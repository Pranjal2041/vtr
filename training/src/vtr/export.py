"""Write VTR checkpoints: a Hugging Face Qwen3.5 model directory whose weights also hold the router
(``model.router.mlp.*``) and whose ``config.json`` records the routing block size (``vlm_router_block_size``).

``vtr-init`` combines a base model with a cold-start router into the checkpoint joint training starts from.
``vtr-export`` turns a joint-training checkpoint into the same format for inference."""

import json
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from huggingface_hub import snapshot_download
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.weights import load_state_dict, save_state_dict
from pydantic import Field


class InitConfig(BaseConfig):
    base: str = "Qwen/Qwen3.5-2B"
    """Base model (HF name or local directory)."""

    router: Path
    """Cold-start router checkpoint (``router_best.pt`` written by ``vtr-coldstart``)."""

    output: Path

    block_size: int = Field(10, ge=1)


class ExportConfig(BaseConfig):
    checkpoint: Path
    """Joint-training checkpoint step directory (``<output_dir>/checkpoints/step_<N>``)."""

    base: str
    """The model the run trained from (its ``model.name``); its config and tokenizer files are copied."""

    output: Path

    block_size: int = Field(10, ge=1)


def resolve(model: str) -> Path:
    return Path(model) if Path(model).is_dir() else Path(snapshot_download(model))


def write_checkpoint(state_dict: dict[str, torch.Tensor], base: Path, output: Path, block_size: int) -> None:
    save_state_dict(state_dict, output)
    for file in base.iterdir():
        if file.is_file() and not file.name.endswith((".safetensors", ".safetensors.index.json")):
            shutil.copy(file, output / file.name)
    config = json.loads((base / "config.json").read_text())
    config["vlm_router_block_size"] = block_size
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def init(config: InitConfig) -> None:
    base = resolve(config.base)
    state_dict = load_state_dict(base)
    router = torch.load(config.router, map_location="cpu", weights_only=True)["model"]
    state_dict |= {f"model.router.{name}": value.to(torch.bfloat16) for name, value in router.items()}
    write_checkpoint(state_dict, base, config.output, config.block_size)


def export(config: ExportConfig) -> None:
    base = resolve(config.base)
    reader = dcp.FileSystemReader(config.checkpoint / "trainer")
    prefix = "app.model."
    state_dict = {
        name: torch.empty(meta.size, dtype=meta.properties.dtype)
        for name, meta in reader.read_metadata().state_dict_metadata.items()
        if name.startswith(prefix)
    }
    dcp.load(state_dict, storage_reader=reader, no_dist=True)
    # Qwen3.5 checkpoints already use the Hugging Face parameter names.
    state_dict = {name.removeprefix(prefix): value for name, value in state_dict.items()}
    if json.loads((base / "config.json").read_text()).get("tie_word_embeddings"):
        del state_dict["lm_head.weight"]
    write_checkpoint(state_dict, base, config.output, config.block_size)


def init_main() -> None:
    init(cli(InitConfig))


def export_main() -> None:
    export(cli(ExportConfig))
