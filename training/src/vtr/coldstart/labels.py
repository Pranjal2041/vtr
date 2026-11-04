"""Per-patch targets for the cold-start router.

A window is eleven consecutive frames; only the last (current) frame is labeled, on the patch grid the vision
encoder gives it. A patch is positive if it changed since the previous frame (temporal) and lies on a SAM 3 mask
of one of the window's concepts (semantic). Each modality has its own rule (``LabelRule``)."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from PIL import Image
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from pydantic import BaseModel
from transformers import AutoImageProcessor, Sam3Model, Sam3Processor
from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

from vtr.data import load_image


class LabelRule(BaseModel):
    detection_threshold: float
    """Score SAM 3 post-processing runs at."""

    score_threshold: float
    """Minimum SAM 3 instance score."""

    concept_batch_size: int

    temporal: Literal["any_pixel", "patch_mean"]
    """``any_pixel``: a patch changed if any of its pixels has a channel difference above ``temporal_threshold``.
    ``patch_mean``: a patch changed if its mean absolute RGB difference is at least ``temporal_threshold``."""

    temporal_threshold: float

    temporal_without_semantic: bool
    """Use the temporal mask alone when SAM 3 grounds no concept in the frame."""


class LabelConfig(BaseConfig):
    manifest: Path
    """JSON lines, one window per line: ``id``, ``modality``, ``split``, ``images`` (eleven image references,
    oldest first, in any form ``vtr.data.load_image`` opens) and ``concepts`` (noun phrases for SAM 3)."""

    output: Path
    """Directory that receives one ``<split>.parquet`` per split."""

    model: str = "Qwen/Qwen3.5-2B"
    """Model whose image processor defines the patch grid."""

    sam: str = "facebook/sam3"

    sam_revision: str = "3c879f39826c281e95690f02c7821c4de09afae7"

    image_sizes: dict[str, tuple[int, int]] = {}
    """(width, height) the router sees each modality's frames at. Temporal masks are computed on the resized
    frames; SAM 3 always segments the original current frame."""

    rules: dict[str, LabelRule] = {
        "desktop": LabelRule(
            detection_threshold=0.25,
            score_threshold=0.25,
            concept_batch_size=8,
            temporal="any_pixel",
            temporal_threshold=5,
            temporal_without_semantic=True,
        ),
        "game": LabelRule(
            detection_threshold=0.2,
            score_threshold=0.5,
            concept_batch_size=16,
            temporal="patch_mean",
            temporal_threshold=10,
            temporal_without_semantic=False,
        ),
    }


def temporal_mask(previous: np.ndarray, current: np.ndarray, grid: tuple[int, int], rule: LabelRule) -> np.ndarray:
    difference = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    if rule.temporal == "any_pixel":
        changed = torch.from_numpy(difference.max(axis=2) > rule.temporal_threshold)
        return F.adaptive_max_pool2d(changed[None].float(), grid)[0].bool().numpy()
    # Non-overlapping patches with edges floor(i * size / grid).
    height, width = difference.shape[:2]
    rows = np.arange(grid[0]) * height // grid[0]
    columns = np.arange(grid[1]) * width // grid[1]
    sums = np.add.reduceat(np.add.reduceat(difference.sum(axis=2, dtype=np.int64), rows, axis=0), columns, axis=1)
    counts = np.diff(rows, append=height)[:, None] * np.diff(columns, append=width)[None, :] * 3
    return sums / counts >= rule.temporal_threshold


@torch.inference_mode()
def semantic_mask(
    image: Image.Image,
    concepts: list[str],
    grid: tuple[int, int],
    rule: LabelRule,
    sam: Sam3Model,
    processor: Sam3Processor,
) -> np.ndarray:
    semantic = torch.zeros(grid, dtype=torch.bool, device=sam.device)
    if not concepts:
        return semantic.numpy(force=True)
    text = processor.tokenizer(
        concepts,
        padding=True,
        truncation=True,
        max_length=sam.text_encoder.config.max_position_embeddings,
        return_tensors="pt",
    ).to(sam.device)
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(sam.device)
    vision = sam.get_vision_features(pixel_values=pixel_values)
    batch_size = rule.concept_batch_size
    for start in range(0, len(concepts), batch_size):
        input_ids = text["input_ids"][start : start + batch_size]
        repeat = input_ids.shape[0]
        output = sam(
            vision_embeds=Sam3VisionEncoderOutput(
                last_hidden_state=vision.last_hidden_state.expand(repeat, -1, -1),
                fpn_hidden_states=[value.expand(repeat, -1, -1, -1) for value in vision.fpn_hidden_states],
                fpn_position_encoding=[value.expand(repeat, -1, -1, -1) for value in vision.fpn_position_encoding],
            ),
            input_ids=input_ids,
            attention_mask=text["attention_mask"][start : start + batch_size],
        )
        results = processor.post_process_instance_segmentation(
            output,
            threshold=rule.detection_threshold,
            mask_threshold=0.5,
            target_sizes=[(image.height, image.width)] * repeat,
        )
        for result in results:
            masks = result["masks"][result["scores"] >= rule.score_threshold]
            if len(masks):
                semantic |= F.adaptive_max_pool2d(masks[:, None].float(), grid)[:, 0].bool().any(0)
    return semantic.numpy(force=True)


def main() -> None:
    config = cli(LabelConfig)
    logger = setup_logger("info")
    image_processor = AutoImageProcessor.from_pretrained(config.model)
    merge_size = image_processor.merge_size
    processor = Sam3Processor.from_pretrained(config.sam, revision=config.sam_revision)
    sam = Sam3Model.from_pretrained(config.sam, revision=config.sam_revision, dtype=torch.bfloat16).cuda().eval()

    splits: dict[str, list[dict]] = {}
    with config.manifest.open() as manifest:
        for index, line in enumerate(manifest):
            row = json.loads(line)
            rule = config.rules[row["modality"]]
            size = config.image_sizes.get(row["modality"])
            original = load_image(row["images"][-1])
            previous, current = load_image(row["images"][-2]), original
            if size is not None:
                previous, current = (image.resize(size, Image.Resampling.LANCZOS) for image in (previous, current))
            _, height, width = image_processor(images=current, return_tensors="pt")["image_grid_thw"][0].tolist()
            grid = (height // merge_size, width // merge_size)
            temporal = temporal_mask(np.asarray(previous), np.asarray(current), grid, rule)
            semantic = semantic_mask(original, row["concepts"], grid, rule, sam, processor)
            label = temporal if rule.temporal_without_semantic and not semantic.any() else temporal & semantic
            splits.setdefault(row["split"], []).append(
                {
                    "id": row["id"],
                    "modality": row["modality"],
                    "images": row["images"],
                    "grid": list(grid),
                    "label": np.packbits(label.reshape(-1), bitorder="little").tobytes(),
                }
            )
            if (index + 1) % 100 == 0:
                logger.info(f"Labeled {index + 1} windows")

    config.output.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        Dataset.from_list(rows).to_parquet(config.output / f"{split}.parquet")
        logger.info(f"Wrote {len(rows)} {split} windows to {config.output / f'{split}.parquet'}")


if __name__ == "__main__":
    main()
