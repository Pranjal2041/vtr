"""Qwen3.5 with the VTR router, served by vLLM.

The vision encoder output of each image is cached unchanged. Before LM prefill the
patched runner calls `vtr_route` once per request, with all of the request's images
in order. The router scores every merged patch from its features in the image and in
up to ten previous images of the same block. Each patch is kept with probability
p = sigmoid(score), as in training; a kept patch enters the LM as p * v at its
original M-RoPE position and a dropped one is removed from the sequence. Block anchors
(images 0, B, 2B, ...) stay whole, and a routed image keeps at least its highest-p patch.
"""

from collections import OrderedDict

import torch
from torch import nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from vtr_inference.config import VTRConfig, is_anchor
from vtr_inference.gate_cache import validate_sample
from vtr_inference.patching import require_patched

ROUTER_HISTORY = 10


def history_indices(start, end, history, block_size, device):
    """For images [start, end): indices of themselves and their `history` predecessors,
    clamped to their block's anchor (or image 0 without blocks)."""
    frames = torch.arange(start, end, device=device)
    anchors = (frames // block_size) * block_size if block_size else torch.zeros_like(frames)
    indices = frames[:, None] + torch.arange(-history, 1, device=device)[None, :]
    return indices.maximum(anchors[:, None])


class PatchRouter(nn.Module):
    """MLP shared across patches, applied to a patch concatenated with the same patch in
    the K previous images. Matches the trained module, so the checkpoint loads as is."""

    def __init__(self, dim: int, K: int = ROUTER_HISTORY, hidden=(1024, 256), p_drop: float = 0.1):
        super().__init__()
        self.K = K
        layers, d = [], (K + 1) * dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, image_embeds, n_per_frame: int, chunk_frames: int = 16, *, block_size: int = 0):
        n_tokens, dim = image_embeds.shape
        assert n_tokens % n_per_frame == 0, f"{n_tokens} image tokens not divisible by {n_per_frame}/frame"
        n_frames = n_tokens // n_per_frame
        feats = image_embeds.view(n_frames, n_per_frame, dim)
        out = []
        for a in range(0, n_frames, chunk_frames):
            b = min(a + chunk_frames, n_frames)
            idx = history_indices(a, b, self.K, block_size, feats.device)
            window = feats[idx].permute(0, 2, 1, 3).reshape(b - a, n_per_frame, (self.K + 1) * dim)
            out.append(self.mlp(window).squeeze(-1).reshape(-1))
        return torch.cat(out)


class VTRQwen3_5ForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # Gates are computed in fp32, as in training.
        self.router = PatchRouter(dim=self.config.text_config.hidden_size).float()
        self.vtr = VTRConfig.from_hf_config(self.config)
        self.vtr_enabled = self.vtr is not None
        if self.vtr_enabled:
            self.vtr.validate(vllm_config)
            require_patched()
        # Gate-cached draws by frame nonce. A request leases the nonces of its plan, so
        # concurrent requests that share a frame reuse one draw until it is published.
        self._draws = OrderedDict()
        self._draw_leases = {}
        self._request_leases = {}

    @torch.inference_mode()
    def vtr_route(self, entries, plan=None, request_id=None):
        """Route one request's images, given as encoder outputs (n_tokens, D) in prompt order.

        Returns, per image, the embeddings to splice in and the kept token indices (CPU).
        """
        if len({entry.shape[0] for entry in entries}) != 1:
            raise ValueError("VTR routes requests whose images all have the same size")
        if plan is None:
            return self._route(entries)
        return self._route_planned(entries, plan, request_id)

    def _probabilities(self, entries):
        n = entries[0].shape[0]
        return torch.sigmoid(self.router(torch.cat(entries).float(), n, block_size=self.vtr.block_size))

    def _route(self, entries):
        n = entries[0].shape[0]
        p = self._probabilities(entries)
        keep = torch.bernoulli(p).bool()
        out = []
        for i, entry in enumerate(entries):
            if is_anchor(i, self.vtr.block_size):
                out.append((entry, torch.arange(n)))
                continue
            p_i, keep_i = p[i * n : (i + 1) * n], keep[i * n : (i + 1) * n].clone()
            if not bool(keep_i.any()):
                keep_i[p_i.argmax()] = True
            out.append(
                (entry[keep_i] * p_i[keep_i].unsqueeze(-1).to(entry.dtype), keep_i.nonzero(as_tuple=True)[0].cpu())
            )
        return out

    def _route_planned(self, entries, plan, request_id):
        """Route with a gate plan: reuse known gates, sample the rest and record them in the plan.

        The router runs only on the blocks that contain an unknown gate.
        """
        frames = plan["frames"]
        if len(entries) != len(frames):
            raise ValueError("Gate plan and encoder outputs cover different images")
        if request_id not in self._request_leases:
            nonces = [frame["nonce"] for frame in frames]
            self._request_leases[request_id] = nonces
            for nonce in nonces:
                self._draw_leases[nonce] = self._draw_leases.get(nonce, 0) + 1
        missing = []
        for i, frame in enumerate(frames):
            draw = self._draws.get(frame["nonce"])
            if frame["keep"] is None and draw is not None:
                frame.update(keep=draw[0], scale=draw[1])
                self._draws.move_to_end(frame["nonce"])
            if frame["keep"] is None and not frame["dense"]:
                missing.append(i)

        n, block_size = entries[0].shape[0], self.vtr.block_size
        offsets, p = {}, None
        if missing:
            if block_size:
                starts = sorted({i // block_size * block_size for i in missing})
                selected = [i for s in starts for i in range(s, min(s + block_size, len(entries)))]
            else:
                selected = list(range(len(entries)))
            offsets = {index: packed * n for packed, index in enumerate(selected)}
            p = self._probabilities([entries[i] for i in selected])

        out = []
        for i, (frame, entry) in enumerate(zip(frames, entries)):
            if entry.shape[0] != frame["length"]:
                raise ValueError("Gate plan image size differs from the encoder output")
            if frame["keep"] is None and frame["dense"]:
                frame.update(keep=list(range(n)), scale=[1.0] * n)
            elif frame["keep"] is None:
                p_i = p[offsets[i] : offsets[i] + n]
                keep = torch.bernoulli(p_i).bool()
                if not bool(keep.any()):
                    keep[p_i.argmax()] = True
                frame.update(
                    keep=keep.nonzero(as_tuple=True)[0].cpu().tolist(),
                    scale=p_i[keep].float().cpu().tolist(),
                )
            validate_sample(frame)
            self._draws[frame["nonce"]] = (frame["keep"], frame["scale"])
            indices = torch.tensor(frame["keep"], device=entry.device, dtype=torch.long)
            scales = torch.tensor(frame["scale"], device=entry.device, dtype=torch.float32)
            out.append((entry[indices] * scales[:, None].to(entry.dtype), indices.cpu()))
        return out

    def vtr_release(self, request_id):
        """Drop a finished request's leases and trim unleased draws to the cache capacity."""
        for nonce in self._request_leases.pop(request_id, []):
            self._draw_leases[nonce] -= 1
            if self._draw_leases[nonce] == 0:
                del self._draw_leases[nonce]
        for nonce in list(self._draws):
            if len(self._draws) <= self.vtr.gate_cache_frames:
                break
            if nonce not in self._draw_leases:
                del self._draws[nonce]

    def load_weights(self, weights):
        """Load a VTR export: router weights go to `self.router`, the rest are renamed to the
        standard Qwen3.5 checkpoint layout."""
        router_state, passthrough = {}, []
        for name, tensor in weights:
            if ".router.mlp." in name:
                router_state[name.split(".router.", 1)[1]] = tensor
            elif (hf_name := _hf_name(name)) is not None:
                passthrough.append((hf_name, tensor))
        loaded = super().load_weights(passthrough)
        self.router.load_state_dict(router_state, strict=True)
        loaded |= {f"router.{name}" for name in router_state}
        # The loader omits some directly assigned parameters (norms, GDN A_log/dt_bias,
        # vision embeddings) from the returned set although it loads them. Credit every
        # parameter that received a checkpoint tensor so the strict load check holds.
        params = {name for name, _ in self.named_parameters()}
        for hf_name, _ in passthrough:
            internal = self.hf_to_vllm_mapper._map_name(hf_name)
            if internal in params:
                loaded.add(internal)
        return loaded


def _hf_name(name):
    """Training-wrapper parameter names -> standard Qwen3.5 names; None for MTP weights."""
    lm = "model.language_model.language_model.language_model."
    if name.startswith("mtp."):
        return None
    if name.startswith(lm):
        return "model.language_model." + name[len(lm) :]
    if name.startswith("model.language_model.language_model.lm_head."):
        return "lm_head." + name.split("lm_head.", 1)[1]
    if name.startswith("model.language_model.visual."):
        return "model.visual." + name[len("model.language_model.visual.") :]
    return name
