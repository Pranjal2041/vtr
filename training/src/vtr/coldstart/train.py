"""Cold-start router training: the router alone, on frozen vision features, with binary cross-entropy against the
per-patch labels written by ``vtr-labels``. Only the current (last) frame of each eleven-frame window is
supervised."""

import hashlib
import json
import math
import os
import random
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.weights import load_state_dict
from pydantic import Field
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoImageProcessor

from vtr.data import load_image
from vtr.export import resolve
from vtr.router import HISTORY, PatchRouter
from vtr.vision import VisionModel

# loss sum, patches, positive targets, kept predictions, true positives, false positives, false negatives
NUM_COUNTS = 7


class ColdstartConfig(BaseConfig):
    model: str = "Qwen/Qwen3.5-2B"
    """Model whose frozen vision encoder produces the router's input features."""

    data: Path
    """Directory with the ``train.parquet`` and ``validation.parquet`` written by ``vtr-labels``."""

    output_dir: Path = Path("outputs/coldstart")

    image_sizes: dict[str, tuple[int, int]] = {}
    """(width, height) to resize each modality's frames to. Must match the sizes the labels were made at."""

    batch_sizes: dict[str, int] = {"desktop": 256, "game": 768}
    """Windows of each modality in one optimizer step, across all ranks. An epoch ends once every window of the
    largest modality (in steps) has been seen; smaller modalities cycle."""

    epochs: int = Field(3, ge=1)

    lr: float = 1.5e-3

    weight_decay: float = 1e-4

    vision_batch_size: int = Field(8, ge=1)
    """Windows per frozen-vision forward."""

    num_workers: int = 4

    seed: int = 0


class Windows(torch.utils.data.Dataset):
    def __init__(self, rows: Dataset, processor, image_sizes: dict[str, tuple[int, int]]):
        self.rows, self.processor, self.image_sizes = rows, processor, image_sizes

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        size = self.image_sizes.get(row["modality"])
        images = [load_image(source) for source in row["images"]]
        if size is not None:
            images = [image.resize(size, resample=Image.Resampling.LANCZOS) for image in images]
        encoded = self.processor(images=images, return_tensors="pt")
        height, width = row["grid"]
        label = np.unpackbits(np.frombuffer(row["label"], dtype=np.uint8), count=height * width, bitorder="little")
        return {
            "id": row["id"],
            "modality": row["modality"],
            "pixel_values": encoded["pixel_values"],
            "image_grid_thw": encoded["image_grid_thw"],
            "labels": torch.from_numpy(label.astype(np.float32)),
        }


def load_vision(model: str, device: torch.device) -> VisionModel:
    path = resolve(model)
    config = AutoConfig.from_pretrained(path)
    vision = VisionModel._from_config(config.vision_config, attn_implementation="sdpa", dtype=torch.bfloat16)
    prefix = "model.visual."
    state_dict = {
        name.removeprefix(prefix): value for name, value in load_state_dict(path).items() if name.startswith(prefix)
    }
    vision.load_state_dict(state_dict)
    return vision.to(device).eval().requires_grad_(False)


@torch.no_grad()
def router_inputs(vision: VisionModel, windows: list[dict], device: torch.device) -> list[Tensor]:
    """Per window, the (patches, 11 * dim) router input: each patch's features in every frame, oldest first."""
    grid_thw = torch.cat([window["image_grid_thw"] for window in windows]).to(device)
    pixel_values = torch.cat([window["pixel_values"] for window in windows]).to(device, vision.dtype)
    features = vision(pixel_values, grid_thw)
    patches = (grid_thw.prod(-1) // vision.spatial_merge_size**2).tolist()
    frames = features.split(patches)
    inputs = []
    for index, window in enumerate(windows):
        window_frames = torch.stack(frames[index * (HISTORY + 1) : (index + 1) * (HISTORY + 1)])
        if window_frames.shape[1] != window["labels"].numel():
            raise ValueError(f"{window['id']}: {window_frames.shape[1]} patches but {window['labels'].numel()} labels")
        inputs.append(window_frames.permute(1, 0, 2).reshape(window_frames.shape[1], -1).float())
    return inputs


def epoch_schedule(modalities: list[str], batch_sizes: dict[str, int], epoch: int, seed: int) -> list[list[int]]:
    """Row indices of every optimizer step of one epoch."""
    groups = {name: [i for i, modality in enumerate(modalities) if modality == name] for name in batch_sizes}
    steps = max(math.ceil(len(rows) / batch_sizes[name]) for name, rows in groups.items())
    rng = random.Random(seed + epoch)
    for rows in groups.values():
        rng.shuffle(rows)
    return [
        [
            rows[(step * batch_sizes[name] + i) % len(rows)]
            for name, rows in groups.items()
            for i in range(batch_sizes[name])
        ]
        for step in range(steps)
    ]


def example_seed(context: str, example_id: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{context}/{example_id}".encode()).digest()[:8], "little")


def counts(logits: Tensor, labels: Tensor, seed: int) -> Tensor:
    """Metric counts of one window, with kept patches drawn from the router's keep probabilities."""
    probs = logits.sigmoid()
    generator = torch.Generator(device=logits.device).manual_seed(seed)
    kept = torch.rand(probs.shape, device=logits.device, generator=generator) < probs
    positive = labels > 0.5
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="sum")
    values = [loss, labels.numel(), positive.sum(), kept.sum(), (kept & positive).sum(), (kept & ~positive).sum()]
    values.append((~kept & positive).sum())
    return torch.stack([torch.as_tensor(value, dtype=torch.float64, device=logits.device) for value in values])


def summarize(counts: Tensor, prefix: str) -> dict[str, float]:
    loss, patches, positive, kept, tp, fp, fn = counts.tolist()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        f"{prefix}/loss": loss / max(patches, 1),
        f"{prefix}/f1": 2 * precision * recall / max(precision + recall, 1e-12),
        f"{prefix}/precision": precision,
        f"{prefix}/recall": recall,
        f"{prefix}/kept_fraction": kept / max(patches, 1),
        f"{prefix}/target_fraction": positive / max(patches, 1),
    }


def groups(windows: Iterator[dict], count: int, size: int) -> Iterator[list[dict]]:
    """The next ``count`` windows, in runs of at most ``size`` consecutive windows of the same modality."""
    group = []
    for _ in range(count):
        window = next(windows)
        if group and (len(group) == size or window["modality"] != group[0]["modality"]):
            yield group
            group = []
        group.append(window)
    if group:
        yield group


def main() -> None:
    config = cli(ColdstartConfig)
    rank, world_size, local_rank = (
        int(os.environ.get(name, default)) for name, default in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0))
    )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    logger = setup_logger("info")
    for name, size in config.batch_sizes.items():
        if size % world_size:
            raise ValueError(f"batch_sizes.{name}={size} is not divisible by the world size {world_size}")
    torch.set_float32_matmul_precision("highest")

    processor = AutoImageProcessor.from_pretrained(resolve(config.model))
    vision = load_vision(config.model, device)
    splits = {split: Dataset.from_parquet(str(config.data / f"{split}.parquet")) for split in ("train", "validation")}
    datasets = {split: Windows(rows, processor, config.image_sizes) for split, rows in splits.items()}

    torch.manual_seed(config.seed)
    router = PatchRouter(vision.config.out_hidden_size).to(device)
    # Same initial weights on every rank, different dropout masks.
    torch.manual_seed(config.seed + rank)
    optimizer = torch.optim.AdamW(router.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def loader(indices: list[int], split: str) -> DataLoader:
        return DataLoader(
            Subset(datasets[split], indices),
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    @torch.no_grad()
    def validate(step: int, best_f1: float) -> float:
        router.eval()
        totals = {name: torch.zeros(NUM_COUNTS, dtype=torch.float64, device=device) for name in config.batch_sizes}
        indices = list(range(rank, len(datasets["validation"]), world_size))
        for group in groups(iter(loader(indices, "validation")), len(indices), config.vision_batch_size):
            for window, inputs in zip(group, router_inputs(vision, group, device)):
                seed = example_seed(f"validation/{config.seed}", window["id"])
                totals[window["modality"]] += counts(router.mlp(inputs).squeeze(-1), window["labels"].to(device), seed)
        metrics = {}
        for name, value in totals.items():
            dist.all_reduce(value)
            metrics |= summarize(value, f"val/{name}")
        metrics |= summarize(sum(totals.values()), "val")
        router.train()
        if rank == 0:
            logger.success(f"Validation | Step {step} | " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))
            with (config.output_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps({"step": step, **metrics}) + "\n")
            if metrics["val/f1"] > best_f1:
                torch.save(
                    {"model": router.state_dict(), "step": step, "metrics": metrics},
                    config.output_dir / "router_best.pt",
                )
        return max(best_f1, metrics["val/f1"])

    config.output_dir.mkdir(parents=True, exist_ok=True)
    step, best_f1 = 0, validate(0, -1.0)
    modalities = splits["train"]["modality"]
    patches = [height * width for height, width in splits["train"]["grid"]]
    for epoch in range(config.epochs):
        schedule = epoch_schedule(modalities, config.batch_sizes, epoch, config.seed)
        windows = iter(loader([index for batch in schedule for index in batch[rank::world_size]], "train"))
        for batch in schedule:
            # The loss is the mean over every patch of the global batch.
            num_patches = sum(patches[index] for index in batch)
            step_counts = torch.zeros(NUM_COUNTS, dtype=torch.float64, device=device)
            for group in groups(windows, len(batch[rank::world_size]), config.vision_batch_size):
                labels = torch.cat([window["labels"] for window in group]).to(device)
                logits = router.mlp(torch.cat(router_inputs(vision, group, device))).squeeze(-1)
                (F.binary_cross_entropy_with_logits(logits, labels, reduction="sum") / num_patches).backward()
                step_counts += counts(
                    logits.detach(), labels, example_seed(f"train/{config.seed}/{step}", group[0]["id"])
                )
            dist.all_reduce(step_counts)
            for param in router.parameters():
                dist.all_reduce(param.grad)
            grad_norm = torch.nn.utils.get_total_norm([param.grad for param in router.parameters()])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if rank == 0:
                metrics = summarize(step_counts, "train") | {"train/grad_norm": grad_norm.item(), "epoch": epoch}
                logger.success(
                    f"Step {step} | Loss {metrics['train/loss']:.4f} | F1 {metrics['train/f1']:.4f} | "
                    f"Grad. Norm {grad_norm.item():.4f}"
                )
                with (config.output_dir / "metrics.jsonl").open("a") as file:
                    file.write(json.dumps({"step": step, **metrics}) + "\n")
        best_f1 = validate(step, best_f1)
    if rank == 0:
        torch.save({"model": router.state_dict(), "step": step}, config.output_dir / "router_last.pt")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
