import torch
from torch import Tensor, nn

HISTORY = 10
HIDDEN = (1024, 256)
DROPOUT = 0.1


def is_anchor(frame: int, block_size: int) -> bool:
    return frame % block_size == 0


def anchor_mask(tokens_per_frame: list[int], block_size: int, device: torch.device) -> Tensor:
    """Per-token mask of the dense block-anchor frames (frames 0, B, 2B, ...)."""
    anchors = torch.tensor([is_anchor(i, block_size) for i in range(len(tokens_per_frame))], device=device)
    counts = torch.tensor(tokens_per_frame, device=device)
    return torch.repeat_interleave(anchors, counts, output_size=sum(tokens_per_frame))


def history_indices(start: int, end: int, history: int, block_size: int, device: torch.device) -> Tensor:
    """Frame indices [t - history, ..., t] for frames t in [start, end), clamped to the frame's block anchor."""
    frames = torch.arange(start, end, device=device)
    anchors = frames // block_size * block_size
    indices = frames[:, None] + torch.arange(-history, 1, device=device)[None, :]
    return indices.maximum(anchors[:, None])


class PatchRouter(nn.Module):
    """Scores every visual token from the features of the same patch in the current and previous frames.

    The input of the MLP for patch ``j`` of frame ``t`` is the concatenation of the frozen vision features of
    patch ``j`` in frames ``t - history, ..., t`` (oldest first); frames before the block anchor repeat it.
    """

    def __init__(self, dim: int, history: int = HISTORY, hidden: tuple[int, ...] = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.history = history
        layers, width = [], (history + 1) * dim
        for size in hidden:
            layers += [nn.Linear(width, size), nn.GELU(), nn.Dropout(dropout)]
            width = size
        layers.append(nn.Linear(width, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: Tensor, tokens_per_frame: int, block_size: int, chunk_frames: int = 16) -> Tensor:
        """Map frame-major features (frames * tokens_per_frame, dim) to keep logits (frames * tokens_per_frame,)."""
        num_tokens, dim = features.shape
        num_frames = num_tokens // tokens_per_frame
        frames = features.view(num_frames, tokens_per_frame, dim)
        logits = []
        # Chunked over frames: the full (frames, tokens, history + 1, dim) window does not fit in memory.
        for start in range(0, num_frames, chunk_frames):
            end = min(start + chunk_frames, num_frames)
            window = frames[history_indices(start, end, self.history, block_size, features.device)]
            window = window.permute(0, 2, 1, 3).reshape(end - start, tokens_per_frame, -1)
            logits.append(self.mlp(window).reshape(-1))
        return torch.cat(logits)
