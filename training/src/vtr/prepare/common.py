import hashlib
import json
import tarfile
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as lines:
        return [json.loads(line) for line in lines if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as lines:
        for row in rows:
            lines.write(json.dumps(row, sort_keys=True) + "\n")
    partial.replace(path)


def hash_fraction(text: str) -> float:
    """Map ``text`` to a stable number in [0, 1)."""
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64


def tar_urls(archive: Path) -> dict[str, str]:
    """``tar+file`` image URLs (see ``vtr.data.load_image``) of every member of an uncompressed tar archive."""
    path = archive.resolve()
    with tarfile.open(archive) as members:
        return {m.name: f"tar+file://{path}?offset={m.offset_data}&size={m.size}" for m in members}
