from pathlib import Path
from typing import Literal

from prime_rl.configs.monitors import MonitorsConfig
from prime_rl.configs.sft import SFTDataConfig
from prime_rl.configs.shared import GCConfig, ResumeConfig, TrainerLogConfig
from prime_rl.configs.trainer import (
    AdamWConfig,
    CheckpointConfig,
    ConstantSchedulerConfig,
    ModelConfig,
    SchedulerConfig,
    TokenizerConfig,
    validate_scheduler,
)
from prime_rl.utils.config import BaseConfig
from pydantic import Field, model_validator
from renderers import AutoRendererConfig, RendererConfig


class VTRDataConfig(SFTDataConfig):
    image_sizes: dict[str, tuple[int, int]] = {}
    """Resize every image of rows whose ``modality`` column is a key to that ``(width, height)`` with Lanczos
    resampling before rendering. Rows of other modalities keep their native resolution."""


class VTRValConfig(BaseConfig):
    interval: int = Field(50, ge=1)
    """Validate after every this many optimizer steps."""

    data: VTRDataConfig


class RouterConfig(BaseConfig):
    block_size: int = Field(10, ge=1)
    """Frames per routing block. The first frame of every block is always kept, and the router's temporal
    history does not reach back past it."""

    lr: float = Field(5e-5, ge=0)
    """AdamW learning rate of the router."""

    weight_decay: float = Field(1e-4, ge=0)
    """AdamW weight decay of the router."""


class ObjectiveConfig(BaseConfig):
    target: float = Field(0.1, ge=0)
    """Target degradation r* of the dual update: the fraction of the full-vs-blind loss gap the sparse model may
    give up."""

    lambda_init: float = Field(0.5, ge=0)
    """Initial dual variable, the weight of the degradation term in the router objective R + lambda * D."""

    dual_lr: float = Field(0.2, ge=0)
    """Step size of the dual update lambda <- max(0, lambda + dual_lr * s * (r - r*))."""

    below_target_scale: float = Field(10.0, ge=0)
    """Factor s applied to the dual step while the measured degradation is below target (s = 1 above it)."""

    ce_chunk_size: int = Field(128, ge=1)
    """Supervised positions per LM-head chunk. Full-vocabulary logits are never materialized for the sequence."""


class VTRConfig(BaseConfig):
    model: ModelConfig = ModelConfig()

    tokenizer: TokenizerConfig = TokenizerConfig()

    renderer: RendererConfig = AutoRendererConfig()

    data: VTRDataConfig = VTRDataConfig()

    val: VTRValConfig | None = None
    """Validation on held-out rows. If None, no validation runs."""

    optim: AdamWConfig = AdamWConfig()
    """AdamW of the three language models (full, blind and sparse each have their own state)."""

    scheduler: SchedulerConfig = ConstantSchedulerConfig()

    router: RouterConfig = RouterConfig()

    objective: ObjectiveConfig = ObjectiveConfig()

    ckpt: CheckpointConfig | None = None

    resume: ResumeConfig | None = None
    """Resume from a checkpoint of this run (``resume.step``, or the latest when unset)."""

    log: TrainerLogConfig = TrainerLogConfig()

    monitors: MonitorsConfig = MonitorsConfig()

    output_dir: Path = Path("outputs")
    """Directory for checkpoints and logs of this run."""

    max_steps: int = Field(ge=1)

    seed: int = 0
    """Seed of the router's dropout and Bernoulli gate samples, offset by the rank."""

    matmul_precision: Literal["highest", "high", "medium"] = "high"

    gc: GCConfig | None = GCConfig()

    dist_timeout_seconds: int = 900

    @model_validator(mode="after")
    def resolve_auto_defaults(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        if self.optim.weight_decay == "auto":
            self.optim.weight_decay = 0.01
        if self.model.freeze_moe_router == "auto":
            self.model.freeze_moe_router = False
        validate_scheduler(self.scheduler, self.max_steps)
        return self

    @model_validator(mode="after")
    def validate_supported(self):
        if self.model.vlm is None or not self.model.vlm.freeze_vision_encoder:
            raise ValueError("VTR routes frozen vision features: set [model.vlm] with freeze_vision_encoder = true")
        unsupported = {
            "model.lora": self.model.lora is not None,
            "model.cp > 1": self.model.cp > 1,
            "model.compile": self.model.compile is not None,
            "model.ac_offloading": self.model.ac_offloading is not None,
            "model.fusions": bool(self.model.fusions.enabled),
            "model.offload": self.model.offload != "none",
            "data.micro_batch_size > 1": self.data.micro_batch_size > 1,
            "resume.dir": self.resume is not None and self.resume.dir is not None,
        }
        if names := [name for name, enabled in unsupported.items() if enabled]:
            raise ValueError(f"Not supported by the VTR trainer: {', '.join(names)}")
        return self
