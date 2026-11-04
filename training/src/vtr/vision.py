import torch
from torch import Tensor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding
from transformers.vision_utils import get_vision_attention_seqlens, get_vision_position_ids


class VisionRotaryEmbedding(Qwen3_5VisionRotaryEmbedding):
    def reset_parameters(self) -> None:
        self.inv_freq.copy_(self.compute_axial_rope_parameters(self.config, self.inv_freq.device)[0])

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        angles = position_ids[..., None].to(self.inv_freq.dtype) * self.inv_freq
        return self.recomposition_frequencies(angles.cos()), self.recomposition_frequencies(angles.sin())


class VisionModel(Qwen3_5VisionModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.rotary_pos_emb = VisionRotaryEmbedding(config)

    def interpolated_pos_embed(self, grid_thw: Tensor) -> Tensor:
        side, merge = self.num_grid_per_side, self.spatial_merge_size
        device, dtype = self.pos_embed.weight.device, self.pos_embed.weight.dtype
        embeds = []
        for t, h, w in grid_thw.tolist():
            rows, cols = torch.linspace(0, side - 1, h), torch.linspace(0, side - 1, w)
            top, left = rows.int(), cols.int()
            bottom, right = (top + 1).clip(max=side - 1), (left + 1).clip(max=side - 1)
            dh, dw = rows - top, cols - left
            corners = (
                (top, left, 1 - dh, 1 - dw),
                (top, right, 1 - dh, dw),
                (bottom, left, dh, 1 - dw),
                (bottom, right, dh, dw),
            )
            embed = sum(
                self.pos_embed((row[:, None] * side + col[None]).to(device))
                * (row_weight[:, None] * col_weight[None]).to(device, dtype)[..., None]
                for row, col, row_weight, col_weight in corners
            )
            blocks = embed.repeat(t, 1, 1).view(t, h // merge, merge, w // merge, merge, -1)
            embeds.append(blocks.permute(0, 1, 3, 2, 4, 5).flatten(0, 4))
        return torch.cat(embeds)

    def forward(self, pixel_values: Tensor, grid_thw: Tensor) -> Tensor:
        hidden_states = self.patch_embed(pixel_values) + self.interpolated_pos_embed(grid_thw)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        position_embeddings = self.rotary_pos_emb(hidden_states, position_ids)
        cu_seqlens, max_seqlen = get_vision_attention_seqlens(grid_thw, self.config)
        for block in self.blocks:
            hidden_states = block(
                hidden_states, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, position_embeddings=position_embeddings
            )
        return self.merger(hidden_states)
