"""Apply, check or revert `vllm.patch` on the installed vLLM wheel.

The patch is exact: a target file is patched only when it matches the wheel's RECORD
hash, and counts as patched only if reverting the patch restores that hash.
"""

import base64
import hashlib
import importlib.metadata
import re
from pathlib import Path

VLLM_VERSION = "0.21.0+cu129.r42434.pr39568.a106aa6"
PATCH_FILE = Path(__file__).with_name("vllm.patch")
PRISTINE, PATCHED, MODIFIED = "pristine", "patched", "modified"
_HUNK = re.compile(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _read(path):
    return path.read_bytes().decode()


def _parse(text):
    """Unified diff -> {path: [(old_start, old_lines, new_start, new_lines), ...]}."""
    files, lines, i = {}, text.splitlines(keepends=True), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("+++ "):
            hunks = files.setdefault(line[4:].split("\t")[0].strip().removeprefix("b/"), [])
        elif match := _HUNK.match(line):
            old_start, old_count, new_start, new_count = (int(g) if g else 1 for g in match.groups())
            old, new = [], []
            while len(old) < old_count or len(new) < new_count:
                i += 1
                kind, body = lines[i][0], lines[i][1:]
                if kind in " -":
                    old.append(body)
                if kind in " +":
                    new.append(body)
            hunks.append((old_start, old, new_start, new))
        i += 1
    return files


def _apply(text, hunks, reverse=False):
    """Replace each hunk at its exact line number; None if any context differs."""
    lines = text.splitlines(keepends=True)
    out, cursor = [], 0
    for old_start, old, new_start, new in hunks:
        start, old, new = (new_start, new, old) if reverse else (old_start, old, new)
        begin = start - 1
        if begin < cursor or lines[begin : begin + len(old)] != old:
            return None
        out += lines[cursor:begin] + new
        cursor = begin + len(old)
    return "".join(out + lines[cursor:])


def _digest(text):
    return base64.urlsafe_b64encode(hashlib.sha256(text.encode()).digest()).rstrip(b"=").decode()


def _targets():
    """(installed path, wheel RECORD sha256, hunks) for every patched file."""
    dist = importlib.metadata.distribution("vllm")
    if dist.version != VLLM_VERSION:
        raise RuntimeError(f"VTR patches vLLM {VLLM_VERSION}, found {dist.version}")
    record = {str(f): f.hash.value for f in dist.files if f.hash is not None}
    return [(Path(dist.locate_file(path)), record[path], hunks) for path, hunks in _parse(_read(PATCH_FILE)).items()]


def _state(text, expected, hunks):
    if _digest(text) == expected:
        return PRISTINE
    reverted = _apply(text, hunks, reverse=True)
    return PATCHED if reverted is not None and _digest(reverted) == expected else MODIFIED


def status():
    """{installed file: pristine | patched | modified}."""
    return {path: _state(_read(path), expected, hunks) for path, expected, hunks in _targets()}


def require_patched():
    unpatched = [str(path) for path, state in status().items() if state != PATCHED]
    if unpatched:
        raise RuntimeError("vLLM is not patched for VTR; run `vtr patch`. Unpatched: " + ", ".join(unpatched))


def apply(revert=False):
    """Patch (or revert) every target file. Refuses if any file is neither pristine nor patched."""
    targets = _targets()
    states = {path: _state(_read(path), expected, hunks) for path, expected, hunks in targets}
    modified = [str(path) for path, state in states.items() if state == MODIFIED]
    if modified:
        raise RuntimeError("Files differ from both the wheel and the patch: " + ", ".join(modified))
    for path, _, hunks in targets:
        if states[path] == (PATCHED if revert else PRISTINE):
            path.write_bytes(_apply(_read(path), hunks, reverse=revert).encode())
            for cached in path.parent.glob(f"__pycache__/{path.stem}.*.pyc"):
                cached.unlink()
    return status()
