from dataclasses import dataclass
from typing import Literal
from unittest.mock import patch

import torch
import torch.nn.functional as F
import triton
from fla.ops.common.chunk_o import chunk_fwd_kernel_o
from prime_rl.configs.trainer import ModelConfig
from prime_rl.trainer import models
from prime_rl.trainer.model import (
    DTYPE_MAP,
    apply_ac,
    freeze_vision_encoder,
    get_model,
    load_dcp_from_hf,
    resolve_auto_attn,
    setup_fsdp,
)
from prime_rl.trainer.models.qwen3_5 import Qwen3_5ForCausalLM
from prime_rl.trainer.models.qwen3_5.rotary_embedding import build_qwen3_5_mrope_position_ids
from prime_rl.trainer.parallel_dims import ParallelDims
from torch import Tensor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

from vtr.router import PatchRouter, anchor_mask
from vtr.vision import VisionModel

Routing = Literal["full", "blind", "sparse"]

chunk_fwd_kernel_o.fn.configs = [triton.Config({"BK": 16, "BV": 16}, num_warps=4, num_stages=3)]


@dataclass
class VTROutput:
    ce: Tensor
    """Per-position cross-entropy (fp32) at the requested loss positions."""

    image_embeds: Tensor
    """Frozen vision features of every image, reusable by later forwards of the same sample."""

    rate: Tensor | None = None
    """Sparse routing: mean keep probability of the routed tokens (the rate term R)."""

    keep_probs: Tensor | None = None
    """Sparse routing: keep probability of every routed token. None if the sample has no routed token."""

    kept_fraction: Tensor | None = None
    """Sparse routing: fraction of the routed tokens the gate kept. None if the sample has no routed token."""


class VTRForCausalLM(Qwen3_5ForCausalLM):
    """Qwen3.5 VLM whose language model reads only the visual tokens a PatchRouter keeps.

    Images are split into blocks of ``block_size`` frames. The first frame of a block (its anchor) is always kept;
    every other visual token is routed. Dropped tokens are removed from the sequence, and kept tokens keep their
    M-RoPE positions, so the language model still knows where each kept patch sits in its frame.

    Routing modes: ``full`` keeps every visual token, ``blind`` keeps only the anchors, and ``sparse`` keeps each
    routed token with its router probability p and scales a kept token's features by p, which makes the sparse
    loss differentiable in p.
    """

    def __init__(self, config) -> None:
        super().__init__(config)
        vision_config = Qwen3_5VisionConfig.from_dict(config.vision_config.to_dict())
        self.model.visual = VisionModel._from_config(
            vision_config, attn_implementation=config.vision_config._attn_implementation
        )
        self.model.router = PatchRouter(config.text_config.hidden_size)

    def forward(
        self,
        input_ids: Tensor,
        pixel_values: Tensor,
        image_grid_thw: Tensor,
        mm_token_type_ids: Tensor,
        loss_positions: Tensor,
        loss_targets: Tensor,
        *,
        routing: Routing,
        block_size: int,
        image_embeds: Tensor | None = None,
        ce_chunk_size: int = 128,
    ) -> VTROutput:
        merge_size = self.config.vision_config.spatial_merge_size
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if image_embeds is None:
            visual = self.model.visual
            image_embeds = visual(pixel_values.to(visual.dtype), image_grid_thw).to(inputs_embeds.dtype)
        position_ids = build_qwen3_5_mrope_position_ids(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            spatial_merge_size=merge_size,
            seq_lens=torch.tensor([input_ids.shape[1]]),
        )
        tokens_per_frame = (image_grid_thw.prod(-1) // merge_size**2).tolist()
        anchors = anchor_mask(tokens_per_frame, block_size, input_ids.device)

        output = VTROutput(ce=torch.empty(0), image_embeds=image_embeds)
        features, keep = image_embeds, None
        if routing == "blind":
            keep = anchors
        elif routing == "sparse":
            features, keep = self.route(image_embeds.detach(), tokens_per_frame, anchors, block_size, output)

        image_mask = input_ids[0] == self.config.image_token_id
        inputs_embeds = inputs_embeds.masked_scatter(image_mask[None, :, None], features)
        if keep is not None:
            keep_sequence = torch.ones_like(image_mask)
            keep_sequence[image_mask] = keep
            inputs_embeds = inputs_embeds[:, keep_sequence]
            position_ids = position_ids[:, :, keep_sequence]
            loss_positions = (keep_sequence.cumsum(0) - 1)[loss_positions]

        hidden = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            seq_lens=torch.tensor([inputs_embeds.shape[1]], device=input_ids.device),
        ).last_hidden_state
        output.ce = self.cross_entropy(hidden[0, loss_positions], loss_targets, ce_chunk_size)
        return output

    def route(
        self, features: Tensor, tokens_per_frame: list[int], anchors: Tensor, block_size: int, output: VTROutput
    ) -> tuple[Tensor, Tensor]:
        if len(set(tokens_per_frame)) != 1:
            raise ValueError(f"The router needs equally sized frames, got {tokens_per_frame} tokens per frame")
        logits = self.model.router(features, tokens_per_frame[0], block_size)
        routed = ~anchors
        if routed.any():
            probs = torch.sigmoid(logits)
            scale = torch.bernoulli(probs) * probs
            scale[anchors] = 1 + logits[anchors] * 0
            output.rate = probs[routed].sum().float() / routed.sum()
            output.keep_probs = probs[routed].float()
        else:
            # Nothing to route. The zero-weight edges keep the router in the autograd graph, so every rank runs
            # the same FSDP collectives in backward.
            scale = 1 + logits * 0
            output.rate = (logits.sum() * 0).float()
        keep = scale > 0
        if routed.any():
            output.kept_fraction = keep[routed].float().mean()
        return features * scale.unsqueeze(-1).to(features.dtype), keep

    def cross_entropy(self, hidden: Tensor, targets: Tensor, chunk_size: int) -> Tensor:
        weight = self.lm_head.weight
        return torch.cat(
            [
                F.cross_entropy(F.linear(chunk, weight).float(), chunk_targets, reduction="none")
                for chunk, chunk_targets in zip(hidden.split(chunk_size), targets.split(chunk_size))
            ]
        )


def setup_vtr_model(config: ModelConfig, parallel_dims: ParallelDims, *, router: bool) -> VTRForCausalLM:
    """Build, shard and load a VTR model from ``config.name``: a Qwen3.5 checkpoint whose safetensors also hold
    the router (``model.router.*``). Reference models (``router=False``) load the same checkpoint without it."""
    resolve_auto_attn(config)
    dtype = DTYPE_MAP[config.optimization_dtype]
    with patch.dict(models._CUSTOM_VLM_MAPPING, {"qwen3_5": VTRForCausalLM}):
        model = get_model(config, device=torch.device("meta"), dtype=dtype)
    if not router:
        model.model.router = None
    freeze_vision_encoder(model.model.visual)
    if config.ac is not None:
        apply_ac(model, config.ac)
    setup_fsdp(model, config, parallel_dims)
    load_dcp_from_hf(model, config)
    model.model.visual.rotary_pos_emb.to(dtype)
    model.model.language_model.rotary_emb.to(dtype)
    return model.train()
