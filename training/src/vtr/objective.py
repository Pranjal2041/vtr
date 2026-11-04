from torch import Tensor

from vtr.config import ObjectiveConfig


def degradation(l_full: Tensor, l_blind: Tensor, l_sparse: Tensor) -> Tensor:
    """Importance-weighted degradation D = sum(w * (l_sparse - l_full)) / sum(w) over supervised tokens.

    The importance of a token is how much the images help predict it, w = max(0, l_blind - l_full), so tokens
    that do not depend on the images do not constrain the router.
    """
    importance = (l_blind - l_full).clamp_min(0)
    return (importance * (l_sparse - l_full)).sum() / (importance.sum() + 1e-6)


def dual_update(dual: float, degradation_ratio: float, config: ObjectiveConfig) -> float:
    """Projected dual ascent on the constraint r <= r*, where r = sum(l_sparse - l_full) / sum(l_blind - l_full)
    is the batch's degradation ratio. Steps below target are scaled up so that the dual falls faster than it
    rises."""
    error = degradation_ratio - config.target
    scale = config.below_target_scale if error < 0 else 1.0
    return max(0.0, dual + config.dual_lr * error * scale)
