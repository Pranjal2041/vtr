"""Noun phrases for SAM 3 to ground: one list per desktop window (Gemini) and one vocabulary per game (Luna).

Each window of the manifest gains a ``concepts`` field. Answers are cached, so an interrupted run resumes. The Luna
client reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL``; the Gemini client reads ``GEMINI_API_KEY``."""

import base64
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

from huggingface_hub import hf_hub_url
from openai import OpenAI
from prime_rl.utils.config import BaseConfig, cli
from prime_rl.utils.logger import setup_logger
from pydantic import BaseModel, Field

from vtr.prepare.common import read_jsonl, write_jsonl
from vtr.prepare.games import REPO_ID, REVISION

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DESKTOP_PROMPT = """A computer-use agent is performing this task on a desktop computer:

TASK: {instruction}

ACTIONS TAKEN SO FAR:
{history}

The attached screenshot is the agent's current view, taken right before its next action. List the visual elements in THIS screenshot that the agent needs to focus on to decide and carry out its next action toward completing the task — the things that actually matter for the task (the specific control to operate next, values it must read/record, the relevant window/dialog/field). Ignore UI that is irrelevant to this task.

Give each concept as a SHORT groundable noun phrase of 2-6 words that names the element so a segmentation model can locate it — e.g. "Save button", "Debit Note input field", "chi-square results table", "Jamovi folder icon". Do NOT write full sentences, quotes, or explanations."""

GAME_PROMPT = """Inspect every attached frame from {game}. Propose the visual concepts
that an agent needs to locate to choose good game actions.

Return short, concrete, visually groundable noun phrases suitable as text prompts
for a segmentation model. Include gameplay entities, interactable objects, spatial
landmarks, status indicators, and UI controls when they affect action choice.
Exclude decorative scenery, generic background, abstract states, actions, and
anything not visibly supported by at least one attached frame. Distinguish objects
that require different actions, but avoid synonyms and cosmetic variants.

Every phrase must name an object or bounded UI/scene region that SAM3 could outline.
Good forms include "enemy character", "health bar", "minimap", and "interactable
door". Do not emit scene descriptions such as "active combat", "player moving",
"open space", "level progress", or "gameplay screen". Do not emit proper names when
a visually equivalent generic category is more groundable.

Return only the required JSON object. Do not use tools and do not explain it."""

VOCABULARY_PROMPT = """You are constructing the text-prompt vocabulary for SAM3 to produce
patch-level pseudo-labels for a game-playing vision-language agent.

Game: {game}

Below are exact concept phrases proposed from {frames:,} sampled frames. The integer is
the number of {batch_size}-frame batches (out of {batches}) that proposed that exact phrase.

Semantically deduplicate this vocabulary. Preserve concepts that require different
visual grounding or different agent actions. Merge only true synonyms, spelling
variants, and unnecessarily specific cosmetic variants. Remove scene descriptions,
actions, abstract states, decorative background, debugging/performance overlays,
and content from applications outside the named game. Each canonical name must be
a short concrete noun phrase that SAM3 can visually ground. Keep rare concepts when
they are clearly actionable; frequency is evidence, not a hard threshold.

Return only the canonical concept names. Aim for the smallest vocabulary that still
covers every distinct action-relevant visual category supported by these frames.
Return only the required JSON.

PHRASES:
{phrases}
"""


class Concepts(BaseModel):
    concepts: list[str]


class FrameConcepts(BaseModel):
    concepts: list[str] = Field(max_length=40)


class Vocabulary(BaseModel):
    concepts: list[str] = Field(min_length=1, max_length=70)


class ConceptsConfig(BaseConfig):
    windows: Path = Path("data/coldstart/windows.jsonl")
    """Manifest written by ``vtr-windows``."""

    games: Path = Path("data/games")
    """Directory written by ``vtr-games``."""

    cache: Path = Path("data/coldstart/concepts")

    desktop_model: str = "gemini-3-flash-preview"

    temperature: float = 0.2

    game_model: str = "gpt-5.6-luna"
    """Also answers desktop windows on which ``desktop_model`` returns no content."""

    reasoning_effort: str = "low"

    frames_per_game: int = 1000
    """Sampled from the game's largest recording."""

    frames_per_batch: int = 20

    margin_seconds: float = 10.0

    seed: int = 20260815

    desktop_workers: int = 32

    game_workers: int = 8

    frame_workers: int = 20

    retries: int = 4


def ask(client: OpenAI, model: str, prompt: str, images: list[Path], schema: type[BaseModel], **options) -> list[str]:
    parts = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime(image)};base64,{base64.b64encode(image.read_bytes()).decode()}"},
        }
        for image in images
    ]
    messages = [{"role": "user", "content": [*parts, {"type": "text", "text": prompt}]}]
    response = client.chat.completions.parse(model=model, messages=messages, response_format=schema, **options)
    return response.choices[0].message.parsed.concepts


def mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(part.get("text")) for part in content if isinstance(part, dict) and part.get("text"))
    return ""


def parse_context(messages: list[dict]) -> tuple[str, str]:
    """The task instruction and the actions taken so far, from a recorded prompt."""
    instruction = ""
    actions = []
    for message in messages:
        text = message_text(message.get("content"))
        if message.get("role") == "user" and "Instruction:" in text and not instruction:
            match = re.search(r"Instruction:\s*(.*?)(?:\nUnless explicitly|\n\nPrevious actions:|$)", text, re.S)
            if match:
                instruction = " ".join(match.group(1).split())
        if message.get("role") != "assistant" or not text.strip():
            continue
        action = re.search(r"Action:\s*([^\n]+)", re.sub(r"<think>.*?</think>", "", text, flags=re.S))
        if action:
            actions.append(" ".join(action.group(1).split()))
    return instruction, "\n".join(actions) if actions else "(none)"


def clean_desktop(values: list[str]) -> list[str]:
    """First eight words of each phrase, unquoted, without case-insensitive repeats."""
    result: dict[str, str] = {}
    for value in values:
        phrase = " ".join(value.replace('"', " ").split()[:8])
        if phrase:
            result.setdefault(phrase.lower(), phrase)
    return list(result.values())


def game_key(phrase: str) -> str:
    return re.sub(r"\s+", " ", phrase.lower().replace("_", " ").strip()).strip(' .,:;-"')


def clean_game(values: list[str]) -> list[str]:
    result: dict[str, str] = {}
    for value in values:
        phrase = " ".join(value.split())
        if game_key(phrase):
            result.setdefault(game_key(phrase), phrase)
    return list(result.values())


def desktop_concepts(window: dict, config: ConceptsConfig, gemini: OpenAI) -> list[str]:
    path = config.cache / "desktop" / f"{window['id']}.json"
    if path.exists():
        return json.loads(path.read_text())
    instruction, history = parse_context(json.loads(Path(window["messages"]).read_text()))
    prompt = DESKTOP_PROMPT.format(instruction=instruction, history=history)
    image = Path(window["images"][-1])
    for attempt in range(config.retries):
        try:
            answer = ask(gemini, config.desktop_model, prompt, [image], Concepts, temperature=config.temperature)
            break
        except Exception:
            time.sleep(1.5 * 2**attempt)
    else:
        answer = ask(OpenAI(), config.game_model, prompt, [image], Concepts, reasoning_effort=config.reasoning_effort)
    concepts = clean_desktop(answer)
    path.write_text(json.dumps(concepts))
    return concepts


def probe(url: str) -> tuple[float, float]:
    """A video's duration in seconds and its frame rate."""
    command = "ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate:format=duration -of json".split()
    value = json.loads(subprocess.check_output([*command, url], text=True))
    return float(value["format"]["duration"]), float(Fraction(value["streams"][0]["r_frame_rate"]))


def extract_frame(url: str, timestamp: float, output: Path) -> None:
    scale = "scale=960:-2:force_original_aspect_ratio=decrease"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.6f}", "-i", url]
    subprocess.run([*command, "-frames:v", "1", "-vf", scale, "-q:v", "3", "-y", str(output)], check=True)


def sample_frames(recording: dict, config: ConceptsConfig, directory: Path) -> list[Path]:
    """Random frames of a recording, read from the dataset without downloading the whole video."""
    public = hf_hub_url(REPO_ID, recording["clip"], repo_type="dataset", revision=REVISION)
    with urllib.request.urlopen(urllib.request.Request(public, method="HEAD"), timeout=60) as response:
        url = response.geturl()
    duration, fps = probe(url)
    rng = random.Random(f"{config.seed}:{recording['game_slug']}:{recording['workflow_id']}")
    eligible = range(int(config.margin_seconds * fps), int((duration - config.margin_seconds) * fps))
    indices = sorted(rng.sample(eligible, config.frames_per_game))
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{position:04d}_{index:09d}.jpg" for position, index in enumerate(indices)]
    with ThreadPoolExecutor(config.frame_workers) as executor:
        list(executor.map(lambda item: extract_frame(url, item[0] / fps, item[1]), zip(indices, paths, strict=True)))
    return paths


def game_vocabulary(slug: str, recordings: list[dict], config: ConceptsConfig, luna: OpenAI) -> list[str]:
    path = config.cache / "game" / f"{slug}.json"
    if path.exists():
        return json.loads(path.read_text())
    recording = min((r for r in recordings if r["game_slug"] == slug), key=lambda r: (-r["clip_size_bytes"], r["clip"]))
    game = recording["game"]
    frames_dir = config.cache / "frames" / slug
    frames = sample_frames(recording, config, frames_dir)
    batches = [frames[i : i + config.frames_per_batch] for i in range(0, len(frames), config.frames_per_batch)]
    options = {"reasoning_effort": config.reasoning_effort}

    def propose(batch: list[Path]) -> list[str]:
        return clean_game(ask(luna, config.game_model, GAME_PROMPT.format(game=game), batch, FrameConcepts, **options))

    with ThreadPoolExecutor(config.game_workers) as executor:
        proposals = list(executor.map(propose, batches))

    counts: Counter[str] = Counter()
    forms: dict[str, Counter[str]] = {}
    for phrases in proposals:
        for phrase in phrases:
            counts[game_key(phrase)] += 1
            forms.setdefault(game_key(phrase), Counter())[phrase] += 1
    listing = "\n".join(
        f"{index:04d}\t{count:2d}\t{forms[key].most_common(1)[0][0]}"
        for index, (key, count) in enumerate(counts.most_common())
    )
    prompt = VOCABULARY_PROMPT.format(
        game=game, frames=len(frames), batch_size=config.frames_per_batch, batches=len(batches), phrases=listing
    )
    concepts = clean_game(ask(luna, config.game_model, prompt, [], Vocabulary, **options))
    path.write_text(json.dumps(concepts))
    shutil.rmtree(frames_dir)
    return concepts


def main() -> None:
    config = cli(ConceptsConfig)
    logger = setup_logger("info")
    windows = read_jsonl(config.windows)
    desktop = [window for window in windows if window["modality"] == "desktop"]
    games = sorted({window["game"] for window in windows if window["modality"] == "game"})
    (config.cache / "desktop").mkdir(parents=True, exist_ok=True)
    (config.cache / "game").mkdir(parents=True, exist_ok=True)

    if desktop:
        gemini = OpenAI(base_url=GEMINI_URL, api_key=os.environ["GEMINI_API_KEY"])
        with ThreadPoolExecutor(config.desktop_workers) as executor:
            answers = executor.map(lambda window: desktop_concepts(window, config, gemini), desktop)
            for window, concepts in zip(desktop, answers, strict=True):
                window["concepts"] = concepts
    if games:
        luna = OpenAI()
        recordings = read_jsonl(config.games / "recordings.jsonl")
        vocabularies = {slug: game_vocabulary(slug, recordings, config, luna) for slug in games}
        for window in windows:
            if window["modality"] == "game":
                window["concepts"] = vocabularies[window["game"]]
    write_jsonl(config.windows, windows)
    logger.info(f"Added concepts to {len(windows)} windows ({len(games)} games)")


if __name__ == "__main__":
    main()
