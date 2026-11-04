"""Gaming-500: the recording inventory, the 40-frame trajectories sampled from it, and their frames."""

import io
import json
import math
import random
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger

from vtr.prepare.common import hash_fraction, read_jsonl, write_jsonl

REPO_ID = "markov-ai/gaming-500-hours"
REVISION = "5af703f2810306e7d75eb4394ae59591f1f6e8a2"
SOURCE_FPS = 30
TRAJECTORY_FRAMES = 40
RATES = (30, 15, 8, 4, 2)
CANVAS_WIDTH, CANVAS_HEIGHT = 1280, 720
GAME_NAMES = {"age-of-empires": "Age of Empires"}


class GamesConfig(BaseConfig):
    output: Path = Path("data/games")
    """Receives ``recordings.jsonl`` and ``trajectories.jsonl``."""

    trajectories_per_rate: int = 30_000

    margin_seconds: int = 10
    """Skip this much footage at each end of a recording."""

    seed: int = 20260817

    workers: int = 24


class FramesConfig(BaseConfig):
    games: Path = Path("data/games")
    """Directory written by ``vtr-games``. Receives ``frames/<recording>.tar`` and
    ``events/<recording>.frame_events.json``."""

    target_hz: list[int] = [30]
    """Sampling rates to extract."""

    recordings: list[str] = []
    """Workflow ids to extract. Empty means every recording with a trajectory at ``target_hz``."""


def slug(dataset_game: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", dataset_game.lower()).strip("_")


def game_name(dataset_game: str) -> str:
    words = (word.upper() if len(word) <= 3 else word.capitalize() for word in dataset_game.split("-"))
    return GAME_NAMES.get(dataset_game, " ".join(words))


def list_recordings(workers: int) -> list[dict]:
    tree = HfApi().list_repo_tree(REPO_ID, repo_type="dataset", revision=REVISION, recursive=True)
    clips = [e for e in tree if isinstance(e, RepoFile) and e.path.endswith("/clip.mp4") and e.path.count("/") == 2]

    def load(clip: RepoFile) -> dict:
        dataset_game, workflow_id, _ = clip.path.split("/")
        metadata = f"{dataset_game}/{workflow_id}/metadata.json"
        path = hf_hub_download(REPO_ID, metadata, repo_type="dataset", revision=REVISION)
        return {
            "workflow_id": workflow_id,
            "game_slug": slug(dataset_game),
            "game": game_name(dataset_game),
            "clip": clip.path,
            "clip_size_bytes": clip.size,
            "duration_s": json.loads(Path(path).read_text())["total_duration_ms"] / 1000,
        }

    with ThreadPoolExecutor(workers) as executor:
        return sorted(executor.map(load, clips), key=lambda value: (value["game_slug"], value["workflow_id"]))


def nearest_frame_offsets(target_hz: int) -> list[int]:
    """Source frames nearest to the timestamps ``k / target_hz``; ties round up."""
    return [int(Fraction(index * SOURCE_FPS, target_hz) + Fraction(1, 2)) for index in range(TRAJECTORY_FRAMES)]


OFFSETS = {rate: nearest_frame_offsets(rate) for rate in RATES}
SPANS = {rate: OFFSETS[rate][-1] + 1 for rate in RATES}


def hamilton_quotas(target: int, weights: dict[str, int], eligible: set[str]) -> dict[str, int]:
    """Weight-proportional integer quotas with at least one per eligible game."""
    quotas = dict.fromkeys(eligible, 1)
    remaining = target - len(eligible)
    total = sum(weights[game] for game in eligible)
    raw = {game: remaining * weights[game] / total for game in eligible}
    for game in eligible:
        quotas[game] += math.floor(raw[game])
    leftover = target - sum(quotas.values())
    for game in sorted(eligible, key=lambda game: (-(raw[game] % 1), game))[:leftover]:
        quotas[game] += 1
    return quotas


def random_gaps(slack: int, count: int, rng: random.Random) -> list[int]:
    """A uniform split of ``slack`` free frames into ``count`` gaps."""
    if slack == 0:
        return [0] * count
    gaps, previous = [], -1
    for separator in sorted(rng.sample(range(slack + count - 1), count - 1)):
        gaps.append(separator - previous - 1)
        previous = separator
    return [*gaps, slack + count - 2 - previous]


def sample_trajectories(recordings: list[dict], per_rate: int, margin_seconds: int, seed: int) -> list[dict]:
    """Sample ``per_rate`` trajectories at every rate. Trajectories never overlap in source time, a game's share
    follows its duration, and every game long enough for a rate gets at least one trajectory at that rate."""
    by_game: dict[str, list[dict]] = defaultdict(list)
    capacity: dict[str, int] = {}
    first_frame = margin_seconds * SOURCE_FPS
    for recording in recordings:
        last_frame = math.floor((float(recording["duration_s"]) - margin_seconds) * SOURCE_FPS)
        capacity[recording["workflow_id"]] = max(0, last_frame - first_frame)
        if capacity[recording["workflow_id"]]:
            by_game[recording["game_slug"]].append(recording)

    weights = {game: sum(capacity[r["workflow_id"]] for r in values) for game, values in by_game.items()}
    quotas = {}
    for rate in RATES:
        eligible = {
            g for g, values in by_game.items() if any(capacity[r["workflow_id"]] >= SPANS[rate] for r in values)
        }
        quotas[rate] = hamilton_quotas(per_rate, weights, eligible)

    remaining = dict(capacity)
    assignments: dict[str, list[int]] = defaultdict(list)
    unplaced = []
    for game in sorted(by_game):
        workflows = [recording["workflow_id"] for recording in by_game[game]]
        for rate in sorted(RATES, key=lambda value: -SPANS[value]):
            for _ in range(quotas[rate].get(game, 0)):
                candidates = [w for w in workflows if remaining[w] >= SPANS[rate]]
                if not candidates:
                    unplaced.append(rate)
                    continue
                workflow = min(candidates, key=lambda value: (remaining[value] - SPANS[rate], value))
                assignments[workflow].append(rate)
                remaining[workflow] -= SPANS[rate]
    for rate in unplaced:
        candidates = [w for w, free in remaining.items() if free >= SPANS[rate]]
        workflow = min(candidates, key=lambda value: (remaining[value] - SPANS[rate], value))
        assignments[workflow].append(rate)
        remaining[workflow] -= SPANS[rate]

    by_workflow = {recording["workflow_id"]: recording for recording in recordings}
    placements = []
    for workflow_id in sorted(assignments):
        rates = assignments[workflow_id]
        rng = random.Random(f"{seed}:game-layout:{workflow_id}")
        rng.shuffle(rates)
        slack = capacity[workflow_id] - sum(SPANS[rate] for rate in rates)
        gaps = random_gaps(slack, len(rates) + 1, rng)
        start = first_frame + gaps[0]
        for index, rate in enumerate(rates):
            placements.append((RATES.index(rate), rng.random(), workflow_id, start, rate))
            start += SPANS[rate] + gaps[index + 1]

    placements.sort()
    per_rate_index: Counter[int] = Counter()
    trajectories = []
    for _, _, workflow_id, start, rate in placements:
        recording = by_workflow[workflow_id]
        instance_id = f"game-{rate:02d}hz-{per_rate_index[rate]:05d}"
        per_rate_index[rate] += 1
        bucket = hash_fraction(f"{seed}:game-split:{workflow_id}")
        trajectories.append(
            {
                "id": instance_id,
                "split": "train" if bucket < 0.95 else "validation",
                "game_slug": recording["game_slug"],
                "game": recording["game"],
                "workflow_id": workflow_id,
                "target_hz": rate,
                "frame_indices": [start + offset for offset in OFFSETS[rate]],
                "member_prefix": f"{rate:02d}hz/{instance_id}",
            }
        )
    return trajectories


def letterbox(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(CANVAS_WIDTH / width, CANVAS_HEIGHT / height)
    resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=interpolation)
    canvas = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    x, y = (CANVAS_WIDTH - resized_width) // 2, (CANVAS_HEIGHT - resized_height) // 2
    canvas[y : y + resized_height, x : x + resized_width] = resized
    return canvas


def write_frames(clip: Path, targets: dict[int, str], output: Path) -> None:
    """Decode ``clip`` and store each target frame as a 1280x720 JPEG named by ``targets`` in a tar archive."""
    capture = cv2.VideoCapture(str(clip))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".tar.partial")
    with tarfile.open(partial, "w") as archive:
        for index in range(max(targets) + 1):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"{clip} ended at frame {index}")
            if index in targets:
                jpeg = cv2.imencode(".jpg", letterbox(frame), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
                member = tarfile.TarInfo(targets[index])
                member.size = len(jpeg)
                archive.addfile(member, io.BytesIO(jpeg))
    capture.release()
    partial.replace(output)


def games_main() -> None:
    config = cli(GamesConfig)
    logger = setup_logger("info")
    recordings_path = config.output / "recordings.jsonl"
    if not recordings_path.exists():
        write_jsonl(recordings_path, list_recordings(config.workers))
    recordings = read_jsonl(recordings_path)
    trajectories = sample_trajectories(recordings, config.trajectories_per_rate, config.margin_seconds, config.seed)
    write_jsonl(config.output / "trajectories.jsonl", trajectories)
    logger.info(f"Sampled {len(trajectories)} trajectories from {len(recordings)} recordings")


def frames_main() -> None:
    config = cli(FramesConfig)
    logger = setup_logger("info")
    recordings = {recording["workflow_id"]: recording for recording in read_jsonl(config.games / "recordings.jsonl")}
    targets: dict[str, dict[int, str]] = defaultdict(dict)
    for trajectory in read_jsonl(config.games / "trajectories.jsonl"):
        workflow_id = trajectory["workflow_id"]
        if trajectory["target_hz"] in config.target_hz and (not config.recordings or workflow_id in config.recordings):
            for position, frame in enumerate(trajectory["frame_indices"]):
                targets[workflow_id][frame] = f"{trajectory['member_prefix']}/{position:02d}.jpg"

    for index, (workflow_id, frames) in enumerate(sorted(targets.items())):
        stage = config.games / "stage" / workflow_id
        download = dict(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, local_dir=stage)
        clip = recordings[workflow_id]["clip"]
        output = config.games / "frames" / f"{workflow_id}.tar"
        events = config.games / "events" / f"{workflow_id}.frame_events.json"
        if not events.exists():
            events.parent.mkdir(parents=True, exist_ok=True)
            source = hf_hub_download(filename=clip.removesuffix("clip.mp4") + "frame_events.json", **download)
            shutil.move(source, events)
        if not output.exists():
            write_frames(Path(hf_hub_download(filename=clip, **download)), frames, output)
        shutil.rmtree(stage, ignore_errors=True)
        logger.info(f"{index + 1}/{len(targets)}: {len(frames)} frames of {clip}")
