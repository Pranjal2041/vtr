import io
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from datasets import Dataset
from PIL import Image
from prime_rl.trainer.data import CatDataset, cat_collate
from prime_rl.trainer.sft.data import RendererResolver, SFTDataset, load_sft_dataset
from renderers import RendererConfig
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.tokenization_utils import PreTrainedTokenizer

from vtr.config import VTRDataConfig


def load_image(source: str) -> Image.Image:
    """Open a path, a ``file://`` URL, or a ``tar+file:///<archive>?offset=<O>&size=<S>`` URL that names the
    ``S`` bytes of an image stored at byte offset ``O`` of an uncompressed tar archive."""
    url = urlparse(source)
    if url.scheme == "tar+file":
        query = parse_qs(url.query)
        with open(unquote(url.path), "rb") as archive:
            archive.seek(int(query["offset"][0]))
            data = archive.read(int(query["size"][0]))
        return Image.open(io.BytesIO(data)).convert("RGB")
    return Image.open(url.path if url.scheme == "file" else source).convert("RGB")


class VTRDataset(SFTDataset):
    """SFT dataset that loads each row's images itself, resizing them when the row's ``modality`` has a size in
    ``image_sizes``, and hands the decoded images to the renderer."""

    def __init__(self, *args, image_sizes: dict[str, tuple[int, int]], **kwargs):
        super().__init__(*args, **kwargs)
        self.image_sizes = image_sizes

    def _process(self, example: dict) -> dict | None:
        size = self.image_sizes.get(example.get("modality"))
        column = self.columns.messages
        messages = [
            {**message, "content": [self._load_part(part, size) for part in message["content"]]}
            if isinstance(message["content"], list)
            else message
            for message in example[column]
        ]
        return super()._process({**example, column: messages})

    @staticmethod
    def _load_part(part: dict[str, Any], size: tuple[int, int] | None) -> dict[str, Any]:
        if part.get("type") == "image_url":
            image = load_image(part["image_url"]["url"])
        elif part.get("type") == "image" and isinstance(part.get("image"), str):
            image = load_image(part["image"])
        else:
            return part
        if size is not None:
            image = image.resize(size, resample=Image.Resampling.LANCZOS)
        return {"type": "image", "image": image}


class UnpackedDataset(CatDataset):
    """One sample per row. Token dropping gives every forward its own sequence, so samples are never packed."""

    def __init__(self, dataset: SFTDataset):
        super().__init__(dataset, seq_len=dataset.seq_len, pad_to_multiple_of=1)

    def __iter__(self):
        yield from self.dataset


def setup_dataloader(
    config: VTRDataConfig,
    tokenizer: PreTrainedTokenizer,
    renderer: RendererConfig,
    processor: Any,
    *,
    max_epochs: int | None = None,
    raw_dataset: Dataset | None = None,
) -> StatefulDataLoader:
    dataset = VTRDataset(
        load_sft_dataset(config) if raw_dataset is None else raw_dataset,
        RendererResolver(tokenizer, renderer, processor=processor, columns=config.columns.renderer),
        shuffle=config.shuffle,
        seed=config.seed,
        seq_len=config.seq_len,
        loss_mask_config=config.loss_mask,
        max_epochs=max_epochs,
        multimodal=True,
        columns=config.columns,
        image_sizes=config.image_sizes,
    )
    return StatefulDataLoader(
        UnpackedDataset(dataset), batch_size=1, collate_fn=cat_collate, num_workers=config.num_workers, pin_memory=True
    )
