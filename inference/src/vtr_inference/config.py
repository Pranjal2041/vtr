"""VTR serving options, read from the model's HF config (set them with `--hf-overrides`)."""

from dataclasses import dataclass

ARCHITECTURE = "VTRQwen3_5ForConditionalGeneration"


@dataclass(frozen=True)
class VTRConfig:
    """Image-token routing options.

    block_size: images 0, B, 2B, ... are dense anchors and the router's history
        never crosses an anchor. 0 keeps only the first image dense, with no resets.
    gate_cache: sample each image's gate once per prompt prefix and reuse it in later
        requests, which makes routing compatible with prefix caching.
    gate_cache_frames: capacity of the gate cache, in images.
    """

    block_size: int = 10
    gate_cache: bool = False
    gate_cache_frames: int = 4096
    image_token_id: int = 0

    @classmethod
    def from_hf_config(cls, hf_config) -> "VTRConfig | None":
        """Return the options, or None when the model is not routed by VTR.

        `vtr_prune: false` serves the VTR checkpoint densely (no routing).
        """
        if ARCHITECTURE not in (getattr(hf_config, "architectures", None) or []):
            return None
        if not getattr(hf_config, "vtr_prune", True):
            return None
        # Exported checkpoints record their block size as `vlm_router_block_size`.
        block_size = getattr(hf_config, "vtr_block_size", getattr(hf_config, "vlm_router_block_size", 10))
        config = cls(
            block_size=int(block_size),
            gate_cache=bool(getattr(hf_config, "vtr_gate_cache", False)),
            gate_cache_frames=int(getattr(hf_config, "vtr_gate_cache_frames", 4096)),
            image_token_id=int(hf_config.image_token_id),
        )
        if config.block_size < 0:
            raise ValueError(f"vtr_block_size must be non-negative, got {config.block_size}")
        if config.gate_cache_frames < 1:
            raise ValueError(f"vtr_gate_cache_frames must be positive, got {config.gate_cache_frames}")
        return config

    def validate(self, vllm_config) -> None:
        """Reject engine settings the routing path does not support."""
        if vllm_config.cache_config.enable_prefix_caching and not self.gate_cache:
            raise ValueError(
                "Prefix caching needs vtr_gate_cache: a fresh gate per request would "
                "reuse KV computed under a different gate. Set vtr_gate_cache or pass "
                "--no-enable-prefix-caching."
            )
        if vllm_config.parallel_config.world_size != 1:
            raise ValueError("VTR routing supports a single GPU per engine (tensor/pipeline parallel size 1)")
        if vllm_config.speculative_config is not None:
            raise ValueError("VTR routing does not support speculative decoding")


def is_anchor(image_index: int, block_size: int) -> bool:
    """Dense anchors are the first image and, with blocks, every block's first image."""
    return image_index == 0 or (block_size > 0 and image_index % block_size == 0)
