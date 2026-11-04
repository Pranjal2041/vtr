"""Joint-training rows: one agent context per row, folded as the agent harness folds it.

Desktop: a trajectory's first row ends after at most 20 actions. Each later row advances 10 actions, keeps the
previous 10 as context, and supervises only the new ones, so every recorded action is supervised exactly once.

Games: a row is one 40-frame trajectory and every frame is supervised. The target at a frame is a control snapshot:
``k`` held keys, ``b`` held mouse buttons, ``m`` the modifier bitfield, ``p`` the latest pointer event and ``s`` the
latest scroll event of the sampling interval. Events skipped by the sampling are not replayed."""

import bisect
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger

from vtr.prepare.common import hash_fraction, read_jsonl, tar_urls

DESKTOP_IMAGE_MAX, DESKTOP_FOLD_SIZE = 20, 10
GAME_IMAGE_MAX, GAME_FOLD_SIZE = 40, 20
INPUT_EVENTS = {"key", "click", "drag", "mouse_move", "scroll", "modifier_change"}
GAME_SYSTEM_PROMPT = (
    "You control the game shown in the screenshots. After each screenshot, output exactly one compact JSON control "
    "state and no prose. The fields are: k = sorted keyboard keyCodes currently held; b = sorted mouse buttons "
    "currently held; m = modifier bitfield; p = the latest pointer event in this sampling interval, when present; "
    "s = the latest scroll event in this sampling interval, when present. Events omitted by temporal downsampling "
    "must not be replayed."
)

CONTENT = pa.struct(
    [
        pa.field("type", pa.string(), nullable=False),
        pa.field("text", pa.string()),
        pa.field("image_url", pa.struct([pa.field("url", pa.string(), nullable=False)])),
    ]
)
MESSAGE = pa.struct(
    [pa.field("role", pa.string(), nullable=False), pa.field("content", pa.list_(CONTENT), nullable=False)]
)
SCHEMA = pa.schema(
    [
        pa.field("messages", pa.list_(MESSAGE), nullable=False),
        pa.field("message_loss_mask", pa.list_(pa.bool_()), nullable=False),
        pa.field("modality", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("source_group", pa.string(), nullable=False),
        pa.field("image_count", pa.int32(), nullable=False),
        pa.field("supervised_action_count", pa.int32(), nullable=False),
        pa.field("dense_anchor_position", pa.int32(), nullable=False),
        pa.field("routed_positions", pa.list_(pa.int32()), nullable=False),
        pa.field("image_max", pa.int32(), nullable=False),
        pa.field("fold_size", pa.int32(), nullable=False),
        pa.field("target_hz", pa.int32()),
    ]
)


class RowsConfig(BaseConfig):
    desktop: Path | None = None
    """JSON lines of desktop trajectories: ``id``, ``sha256``, ``run_dir`` and ``frame_count``. A run directory holds
    ``observation_<step>.png``, ``messages_step_<step>.json`` and ``responses.json``."""

    games: Path | None = Path("data/games")
    """Directory written by ``vtr-games`` and ``vtr-frames``."""

    output: Path = Path("data/vtr")

    target_hz: int = 30

    seed: int = 20260820

    validation_fraction: float = 0.05
    """Share of desktop trajectories held out. Games use the split of their trajectory."""

    validation_per_stratum: int = 8

    validation_seed: int = 20260821

    workers: int = 12


def text_part(text: str) -> dict:
    return {"type": "text", "text": text, "image_url": None}


def image_part(url: str) -> dict:
    return {"type": "image_url", "text": None, "image_url": {"url": url}}


def row(
    modality: str,
    split: str,
    example_id: str,
    group: str,
    messages: list,
    mask: list,
    images: int,
    image_max: int,
    fold_size: int,
    target_hz: int | None = None,
) -> dict:
    return {
        "messages": messages,
        "message_loss_mask": mask,
        "modality": modality,
        "split": split,
        "example_id": example_id,
        "source_group": group,
        "image_count": images,
        "supervised_action_count": sum(mask),
        "dense_anchor_position": 0,
        "routed_positions": list(range(1, images)),
        "image_max": image_max,
        "fold_size": fold_size,
        "target_hz": target_hz,
    }


def case_ranges(action_count: int) -> list[tuple[int, int]]:
    """Inclusive ``(end, supervision_start)`` step pairs of a trajectory's rows."""
    first_end = min(DESKTOP_IMAGE_MAX, action_count) - 1
    ranges = [(first_end, 0)]
    start = first_end + 1
    while start < action_count:
        end = min(start + DESKTOP_FOLD_SIZE, action_count) - 1
        ranges.append((end, start))
        start = end + 1
    return ranges


def desktop_rows(trajectory: dict, seed: int, validation_fraction: float) -> list[dict]:
    name, run_dir = trajectory["id"], Path(trajectory["run_dir"])
    responses = json.loads((run_dir / "responses.json").read_text())["model_responses"]
    held_out = hash_fraction(f"{seed}:desktop-joint-split:{trajectory['sha256']}") < validation_fraction
    rows = []
    for case, (end, supervision_start) in enumerate(case_ranges(int(trajectory["frame_count"]))):
        messages, images = [], 0
        for message in json.loads((run_dir / f"messages_step_{end}.json").read_text()):
            parts = []
            for part in message["content"]:
                if part["type"] == "text":
                    parts.append(text_part(str(part.get("text") or "")))
                else:
                    url = f"file://{(run_dir / Path(part['image_url']['url']).name).resolve()}"
                    parts.append(image_part(url))
                    images += 1
            messages.append({"role": message["role"], "content": parts})
        messages.append({"role": "assistant", "content": [text_part(str(responses[end] or ""))]})

        # Long rollouts also trim their oldest text turns: the assistant messages left are the turns ending at `end`.
        step = end - sum(message["role"] == "assistant" for message in messages)
        mask = []
        for message in messages:
            trainable = False
            if message["role"] == "assistant":
                step += 1
                response = responses[step]
                trainable = supervision_start <= step and isinstance(response, str) and bool(response.strip())
            mask.append(trainable)
        split = "validation" if held_out else "train"
        example_id = f"desktop:{name}:{case:04d}"
        rows.append(
            row("desktop", split, example_id, name, messages, mask, images, DESKTOP_IMAGE_MAX, DESKTOP_FOLD_SIZE)
        )
    return rows


def parse_screens(value: Any) -> dict[int, dict]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if isinstance(value, dict):
        value = [value] if "width" in value and "height" in value else []
    if not isinstance(value, list):
        return {}
    return {int(screen.get("index", index)): screen for index, screen in enumerate(value) if isinstance(screen, dict)}


def normalized_xy(event: dict, screens: dict[int, dict]) -> list[int] | None:
    if "x" not in event or "y" not in event:
        return None
    screen = screens.get(int(event.get("screen", 0))) or screens.get(0)
    if screen is None:
        screen = {"originX": 0.0, "originY": 0.0, "width": 1920.0, "height": 1080.0}
    width = max(float(screen.get("width", 1920.0)), 1.0)
    height = max(float(screen.get("height", 1080.0)), 1.0)
    x = round(1000 * (float(event["x"]) - float(screen.get("originX", 0.0))) / width)
    y = round(1000 * (float(event["y"]) - float(screen.get("originY", 0.0))) / height)
    return [min(1000, max(0, x)), min(1000, max(0, y))]


def pointer_payload(event: dict) -> dict:
    value: dict[str, Any] = {"t": str(event["type"])}
    xy = normalized_xy(event, event.get("_screens") or {})
    if xy is not None:
        value["xy"] = xy
    if event.get("button") is not None:
        value["button"] = str(event["button"])
    if event.get("isDown") is not None:
        value["down"] = bool(event["isDown"])
    return value


def scroll_payload(event: dict) -> list[float | int]:
    def clean(value: Any) -> float | int:
        number = float(value or 0)
        return int(number) if number.is_integer() else round(number, 4)

    return [clean(event.get("deltaX", 0)), clean(event.get("deltaY", 0))]


def replay_events(path: Path, targets: set[int], last_frame: int) -> tuple[dict[int, dict], list[int], list[dict]]:
    """Walk a recording's events up to ``last_frame``. Returns the held-input snapshot at each target frame and the
    input events with their frames, in time order. A recording without input events cannot be supervised."""
    snapshots: dict[int, dict] = {}
    timed: list[tuple[int, tuple[float, float], dict]] = []
    held_keys: set[int] = set()
    held_buttons: set[str] = set()
    modifiers = 0
    screens: dict[int, dict] = {}
    actions = 0
    with path.open() as lines:
        for line in lines:
            record = json.loads(line)
            frame = int(record["frame"])
            if frame > last_frame:
                break
            for event in record.get("events") or []:
                kind = event.get("type")
                if kind == "screen_config":
                    screens = parse_screens(event.get("screens"))
                    continue
                if kind not in INPUT_EVENTS:
                    continue
                actions += kind != "modifier_change"
                if kind == "key":
                    code = int(event["keyCode"])
                    held_keys.add(code) if event.get("isDown") else held_keys.discard(code)
                    modifiers = int(event.get("modifiers", modifiers) or 0)
                elif kind == "click":
                    button = str(event.get("button", "left"))
                    held_buttons.add(button) if event.get("isDown") else held_buttons.discard(button)
                elif kind == "modifier_change":
                    modifiers = int(event.get("modifiers", 0) or 0)
                moment = (float(event.get("video_t_ms", 0.0)), float(event.get("timestamp", 0.0)))
                timed.append((frame, moment, {**event, "_screens": screens}))
            if frame in targets:
                snapshots[frame] = {"k": sorted(held_keys), "b": sorted(held_buttons), "m": modifiers}
    if not actions:
        return {}, [], []
    timed.sort(key=lambda value: value[:2])
    return snapshots, [value[0] for value in timed], [value[2] for value in timed]


def game_action(snapshot: dict, interval: list[dict]) -> dict:
    action = dict(snapshot)
    pointer = next((e for e in reversed(interval) if e["type"] in {"mouse_move", "click", "drag"}), None)
    scroll = next((e for e in reversed(interval) if e["type"] == "scroll"), None)
    if pointer is not None:
        action["p"] = pointer_payload(pointer)
    if scroll is not None:
        action["s"] = scroll_payload(scroll)
    return action if any(action.values()) else {"wait": True}


def game_rows(job: tuple[Path, Path, list[dict], int]) -> list[dict]:
    archive, events, trajectories, last_frame = job
    urls = tar_urls(archive)
    targets = {frame for trajectory in trajectories for frame in trajectory["frame_indices"]}
    snapshots, event_frames, input_events = replay_events(events, targets, last_frame)
    if not input_events:
        return []
    rows = []
    for trajectory in trajectories:
        target_hz = trajectory["target_hz"]
        frames = trajectory["frame_indices"]
        messages = [{"role": "system", "content": [text_part(GAME_SYSTEM_PROMPT)]}]
        mask = [False]
        previous = frames[0] - max(1, round(30 / target_hz))
        for position, frame in enumerate(frames):
            low = bisect.bisect_right(event_frames, previous)
            high = bisect.bisect_right(event_frames, frame)
            action = game_action(snapshots[frame], input_events[low:high])
            parts = [image_part(urls[f"{trajectory['member_prefix']}/{position:02d}.jpg"])]
            if position == 0:
                parts.append(text_part(f"\nGame: {trajectory['game']}. Visual/control sampling rate: {target_hz} Hz."))
            messages.append({"role": "user", "content": parts})
            target = json.dumps(action, separators=(",", ":"), sort_keys=True)
            messages.append({"role": "assistant", "content": [text_part(target)]})
            mask += [False, True]
            previous = frame
        example_id = f"game:{trajectory['id']}"
        group = trajectory["workflow_id"]
        rows.append(
            row(
                "game",
                trajectory["split"],
                example_id,
                group,
                messages,
                mask,
                len(frames),
                GAME_IMAGE_MAX,
                GAME_FOLD_SIZE,
                target_hz,
            )
        )
    return rows


def game_jobs(games: Path, target_hz: int) -> list[tuple[Path, Path, list[dict], int]]:
    """One job per recording with extracted frames. A job's last frame is the recording's last sampled frame at any
    rate: a recording counts as having input if it has any up to there."""
    by_workflow: dict[str, list[dict]] = defaultdict(list)
    last_frames: dict[str, int] = defaultdict(int)
    for trajectory in read_jsonl(games / "trajectories.jsonl"):
        workflow = trajectory["workflow_id"]
        last_frames[workflow] = max(last_frames[workflow], trajectory["frame_indices"][-1])
        if trajectory["target_hz"] == target_hz and (games / "frames" / f"{workflow}.tar").exists():
            by_workflow[workflow].append(trajectory)
    return [
        (
            games / "frames" / f"{workflow}.tar",
            games / "events" / f"{workflow}.frame_events.json",
            trajectories,
            last_frames[workflow],
        )
        for workflow, trajectories in sorted(by_workflow.items())
    ]


def balanced_validation(rows: list[dict], per_stratum: int, seed: int) -> list[dict]:
    """The ``per_stratum`` rows with the lowest seeded hash, at most one per source group."""

    def rank(value: dict) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}:{value['example_id']}".encode()).hexdigest(), value["example_id"]

    selected, groups = [], set()
    for value in sorted(rows, key=rank):
        if value["source_group"] not in groups and len(selected) < per_stratum:
            groups.add(value["source_group"])
            selected.append(value)
    return selected


def dataset_card(game_config: str) -> str:
    lines = ["---", "configs:"]
    for name, split, path in (
        ("desktop", "train", "train-desktop.parquet"),
        (game_config, "train", f"train-{game_config}.parquet"),
        ("balanced_validation", "validation", "validation-balanced.parquet"),
    ):
        lines += [f"- config_name: {name}", "  data_files:", f"  - split: {split}", f"    path: {path}"]
    return "\n".join([*lines, "---", ""])


def main() -> None:
    config = cli(RowsConfig)
    logger = setup_logger("info")
    config.output.mkdir(parents=True, exist_ok=True)
    game_config = f"game_{config.target_hz}hz"
    validation: dict[str, list[dict]] = {"desktop": [], game_config: []}

    def write(rows: list[dict], name: str, writer: pq.ParquetWriter) -> None:
        train = [value for value in rows if value["split"] == "train"]
        validation[name] += [value for value in rows if value["split"] == "validation"]
        if train:
            writer.write_table(pa.Table.from_pylist(train, schema=SCHEMA))

    with pq.ParquetWriter(config.output / "train-desktop.parquet", SCHEMA, compression="zstd") as writer:
        trajectories = read_jsonl(config.desktop) if config.desktop is not None else []
        for trajectory in trajectories:
            write(desktop_rows(trajectory, config.seed, config.validation_fraction), "desktop", writer)

    with pq.ParquetWriter(config.output / f"train-{game_config}.parquet", SCHEMA, compression="zstd") as writer:
        jobs = game_jobs(config.games, config.target_hz) if config.games is not None else []
        with ProcessPoolExecutor(config.workers) as executor:
            for rows in executor.map(game_rows, jobs):
                write(rows, game_config, writer)

    rows = [
        value
        for stratum in validation.values()
        for value in balanced_validation(stratum, config.validation_per_stratum, config.validation_seed)
    ]
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCHEMA), config.output / "validation-balanced.parquet", compression="zstd"
    )
    (config.output / "README.md").write_text(dataset_card(game_config))
    logger.info(
        f"Wrote {config.output}: {pq.read_metadata(config.output / 'train-desktop.parquet').num_rows} desktop and "
        f"{pq.read_metadata(config.output / f'train-{game_config}.parquet').num_rows} game rows, "
        f"{len(rows)} validation rows"
    )


if __name__ == "__main__":
    main()
