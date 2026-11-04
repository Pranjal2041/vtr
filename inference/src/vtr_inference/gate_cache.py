"""Gate plans: sampled per-image gates shared by the scheduler, the worker and the KV cache.

With `vtr_gate_cache`, each image's Bernoulli gate is sampled once per prompt prefix
(the tokens up to and including that image) and reused by later requests with the
same prefix. A request carries its plan in `sampling_params.extra_args[PLAN_KEY]`:

    {"frames": [frame, ...], "original_ids": [...]}
    frame = {"offset", "length", "identifier", "dense", "key", "nonce", "keep", "scale"}

`keep`/`scale` are None until the worker samples the gate. The scheduler compacts the
prompt with every known gate, so prefix-cache lookups see the routed tokens; the KV
block hashes include each gate's nonce.
"""

import hashlib
import json
import math
import uuid
from collections import OrderedDict
from copy import deepcopy
from dataclasses import replace

from vtr_inference.config import is_anchor

PLAN_KEY = "_vtr_plan"


def frame_keys(token_ids, frames, salt):
    """Hash each prompt prefix through each image, so a key fixes the image's history."""
    state = hashlib.sha256(json.dumps(salt, sort_keys=True).encode())
    cursor = 0
    keys = []
    for frame in frames:
        end = frame["offset"] + frame["length"]
        state.update(json.dumps(token_ids[cursor:end], separators=(",", ":")).encode())
        state.update(json.dumps([frame["identifier"], frame["dense"], frame["length"]]).encode())
        keys.append(state.hexdigest())
        cursor = end
    return keys


def compact_prompt(token_ids, frames):
    """Drop the image tokens each frame's `keep` rejects; return ids and new (offset, length)s."""
    result, positions, cursor = [], [], 0
    for frame in frames:
        offset, length = frame["offset"], frame["length"]
        if offset < cursor or offset + length > len(token_ids):
            raise ValueError("Overlapping or out-of-range image spans")
        result.extend(token_ids[cursor:offset])
        indices = frame.get("keep")
        if indices is None:
            indices = range(length)
        elif not indices or list(indices) != sorted(set(indices)) or indices[0] < 0 or indices[-1] >= length:
            raise ValueError("Keep indices must be nonempty, sorted, unique and in range")
        positions.append((len(result), len(indices)))
        result.extend(token_ids[offset + i] for i in indices)
        cursor = offset + length
    result.extend(token_ids[cursor:])
    return result, positions


def validate_sample(frame):
    keep, scale = frame["keep"], frame["scale"]
    if keep is None or scale is None or len(keep) != len(scale):
        raise ValueError("Sample must contain matched keep indices and soft scales")
    compact_prompt([0] * frame["length"], [{**frame, "offset": 0}])
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scale):
        raise ValueError("Invalid soft gate scales")
    if frame["dense"] and (keep != list(range(frame["length"])) or any(p != 1 for p in scale)):
        raise ValueError("Dense anchor must remain unchanged")


class GatePlanCache:
    """LRU of sampled frame gates, keyed by prefix; frames in use are never evicted.

    An evicted key gets a new nonce when it is seen again, so a fresh draw never
    hits KV computed under the old one.
    """

    def __init__(self, capacity=4096):
        if capacity < 1:
            raise ValueError("Gate cache capacity must be positive")
        self.capacity = capacity
        self.entries = OrderedDict()
        self.leases = {}

    def acquire(self, key, frame):
        if key not in self.entries:
            self.entries[key] = {**frame, "key": key, "nonce": uuid.uuid4().hex, "keep": None, "scale": None}
        self.entries.move_to_end(key)
        self.leases[key] = self.leases.get(key, 0) + 1
        self.trim()
        return deepcopy(self.entries[key])

    def publish(self, frame):
        validate_sample(frame)
        current = self.entries[frame["key"]]
        if current["nonce"] != frame["nonce"]:
            raise RuntimeError("Gate identity changed while request was active")
        if current["keep"] is not None and (current["keep"] != frame["keep"] or current["scale"] != frame["scale"]):
            raise RuntimeError("Two requests sampled different gates for the same cached prefix")
        current.update(keep=frame["keep"].copy(), scale=frame["scale"].copy())

    def release(self, frames):
        for frame in frames:
            key = frame["key"]
            self.leases[key] -= 1
            if self.leases[key] == 0:
                del self.leases[key]
        self.trim()

    def trim(self):
        for key in list(self.entries):
            if len(self.entries) <= self.capacity:
                break
            if key not in self.leases:
                del self.entries[key]


def request_plan(request):
    params = request.sampling_params
    return (params.extra_args or {}).get(PLAN_KEY) if params is not None else None


def prepare_request(request, cache, block_size):
    """Attach a plan to a new image request and compact its prompt with the known gates."""
    if request_plan(request) is not None or not request.mm_features:
        return
    if any(f.modality != "image" for f in request.mm_features):
        raise ValueError("Gate-cached routing accepts image-only multimodal requests")
    if request.prompt_embeds is not None or request.resumable:
        raise ValueError("Gate-cached routing requires token prompts and non-streaming inputs")
    original = list(request.prompt_token_ids)
    frames = [
        {
            "offset": f.mm_position.offset,
            "length": f.mm_position.length,
            "identifier": f.identifier,
            "dense": is_anchor(i, block_size),
        }
        for i, f in enumerate(request.mm_features)
    ]
    salt = [request.cache_salt, str(request.lora_request), block_size]
    frames = [cache.acquire(key, frame) for key, frame in zip(frame_keys(original, frames, salt), frames)]
    plan = {"frames": frames, "original_ids": original}
    request.sampling_params = deepcopy(request.sampling_params)
    request.sampling_params.extra_args = {**(request.sampling_params.extra_args or {}), PLAN_KEY: plan}
    apply_request_plan(request, plan)


def apply_request_plan(request, plan):
    """Compact the scheduler's request with the plan's known gates and rehash its blocks."""
    ids, positions = compact_prompt(plan["original_ids"], plan["frames"])
    # Update lists in place: the request's ConstantList views must stay valid.
    request._all_token_ids[:] = ids + list(request.output_token_ids)
    request.prompt_token_ids = ids
    request.num_prompt_tokens = len(ids)
    request.mm_features = [
        replace(f, mm_position=replace(f.mm_position, offset=off, length=n, is_embed=None))
        for f, (off, n) in zip(request.mm_features, positions)
    ]
    request.block_hashes.clear()
    request.update_block_hashes()


def original_features(features, plan):
    """The request's multimodal features at their original, uncompacted positions."""
    return [
        replace(f, mm_position=replace(f.mm_position, offset=p["offset"], length=p["length"], is_embed=None))
        for f, p in zip(features, plan["frames"])
    ]


def hash_features(features, plan):
    """Features whose identifiers include the gate nonce, for KV block hashing only."""
    return [replace(f, identifier=f.identifier + ":gate:" + p["nonce"]) for f, p in zip(features, plan["frames"])]


def cacheable_length(request, plan):
    """Prefix-cache hits must stop before the first image whose gate is still unknown."""
    for feature, frame in zip(request.mm_features, plan["frames"]):
        if frame["keep"] is None and not frame["dense"]:
            return feature.mm_position.offset
    return request.num_tokens - 1
