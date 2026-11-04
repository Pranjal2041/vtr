"""Cold-start windows: ten history frames plus one supervised current frame. No frame occurs in two windows."""

import random
from pathlib import Path

from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger

from vtr.prepare.common import hash_fraction, read_jsonl, tar_urls, write_jsonl

WINDOW_FRAMES = 11


class WindowsConfig(BaseConfig):
    desktop: Path | None = None
    """JSON lines of desktop trajectories: ``sha256``, ``run_dir`` and ``frame_count``. A run directory holds
    ``observation_<step>.png`` and ``messages_step_<step>.json``."""

    games: Path | None = Path("data/games")
    """Directory written by ``vtr-games``. Only recordings with extracted frames (``vtr-frames``) give windows."""

    output: Path = Path("data/coldstart/windows.jsonl")

    desktop_windows: int = 50_000

    target_hz: int = 30

    seed: int = 20260817


def random_packing(frame_count: int, key: str, seed: int) -> list[tuple[int, int]]:
    """One random packing of as many disjoint 11-frame windows as fit, as inclusive frame ranges."""
    count = frame_count // WINDOW_FRAMES
    if count == 0:
        return []
    gaps = [0] * (count + 1)
    rng = random.Random(f"{seed}:desktop-k10-packing:{key}")
    for _ in range(frame_count - count * WINDOW_FRAMES):
        gaps[rng.randrange(len(gaps))] += 1
    windows = []
    start = gaps[0]
    for index in range(count):
        windows.append((start, start + WINDOW_FRAMES - 1))
        start += WINDOW_FRAMES + gaps[index + 1]
    return windows


def desktop_windows(trajectories: list[dict], target: int, seed: int) -> list[dict]:
    """Keep one random window of every trajectory, then sample the rest up to ``target`` across all trajectories."""
    packings = [random_packing(int(t["frame_count"]), t["sha256"], seed) for t in trajectories]
    rng = random.Random(f"{seed}:desktop-k10-select")
    chosen: set[tuple[int, int]] = set()
    remainder: list[tuple[int, int]] = []
    for run, windows in enumerate(packings):
        if windows:
            protected = rng.randrange(len(windows))
            chosen.add((run, protected))
            remainder += [(run, index) for index in range(len(windows)) if index != protected]
    chosen.update(rng.sample(remainder, min(target - len(chosen), len(remainder))))

    rows = []
    for run, (trajectory, windows) in enumerate(zip(trajectories, packings, strict=True)):
        bucket = hash_fraction(f"{seed}:desktop-split:{trajectory['sha256']}")
        run_dir = Path(trajectory["run_dir"])
        for index, (start, end) in enumerate(windows):
            if (run, index) in chosen:
                rows.append(
                    {
                        "id": f"desktop-{len(rows):05d}",
                        "modality": "desktop",
                        "split": "train" if bucket < 0.95 else "validation",
                        "images": [str(run_dir / f"observation_{frame}.png") for frame in range(start, end + 1)],
                        "messages": str(run_dir / f"messages_step_{end}.json"),
                    }
                )
    return rows


def game_windows(games: Path, target_hz: int) -> list[dict]:
    """The last eleven frames of every trajectory at ``target_hz``."""
    urls: dict[str, dict[str, str]] = {}
    rows = []
    for trajectory in read_jsonl(games / "trajectories.jsonl"):
        workflow_id = trajectory["workflow_id"]
        archive = games / "frames" / f"{workflow_id}.tar"
        if trajectory["target_hz"] != target_hz or not archive.exists():
            continue
        if workflow_id not in urls:
            urls[workflow_id] = tar_urls(archive)
        members = (f"{trajectory['member_prefix']}/{position:02d}.jpg" for position in range(29, 40))
        rows.append(
            {
                "id": trajectory["id"],
                "modality": "game",
                "split": trajectory["split"],
                "images": [urls[workflow_id][member] for member in members],
                "game": trajectory["game_slug"],
            }
        )
    return rows


def main() -> None:
    config = cli(WindowsConfig)
    logger = setup_logger("info")
    rows = []
    if config.desktop is not None:
        rows += desktop_windows(read_jsonl(config.desktop), config.desktop_windows, config.seed)
    if config.games is not None:
        rows += game_windows(config.games, config.target_hz)
    write_jsonl(config.output, rows)
    logger.info(f"Wrote {len(rows)} windows to {config.output}")


if __name__ == "__main__":
    main()
