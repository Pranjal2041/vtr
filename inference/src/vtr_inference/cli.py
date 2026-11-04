"""`vtr patch` and `vtr serve`."""

import argparse
import json
import os
import shlex
import sys

from vtr_inference import patching
from vtr_inference.config import ARCHITECTURE

# Engine settings of the released evaluation servers. Arguments after the model
# override them.
SERVE_DEFAULTS = [
    "--dtype", "bfloat16",
    "--seed", "0",
    "--tensor-parallel-size", "1",
    "--max-model-len", "65536",
    "--max-num-batched-tokens", "65536",
    "--max-num-seqs", "4",
    "--gpu-memory-utilization", "0.85",
    "--limit-mm-per-prompt", '{"image": 40, "video": 0}',
    "--mm-processor-cache-gb", "0",
]  # fmt: skip
STANDARD = ["--no-enable-prefix-caching", "--no-enable-chunked-prefill"]
# vLLM requires chunked prefill with prefix caching for this hybrid model.
FAST = ["--enable-prefix-caching"]


def serve_command(args, extra):
    overrides = {"architectures": [ARCHITECTURE], **json.loads(args.hf_overrides)}
    if args.fast:
        overrides["vtr_gate_cache"] = True
    if args.dense:
        overrides["vtr_prune"] = False
    if args.router_block_size is not None:
        overrides["vtr_block_size"] = args.router_block_size
    return [
        "vllm",
        "serve",
        args.model,
        *SERVE_DEFAULTS,
        *(FAST if args.fast else STANDARD),
        "--hf-overrides",
        json.dumps(overrides),
        *extra,
    ]


def main():
    parser = argparse.ArgumentParser(prog="vtr", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)

    patch = commands.add_parser("patch", help="patch the installed vLLM for VTR")
    mode = patch.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report the patch state; exit 1 unless patched")
    mode.add_argument("--revert", action="store_true", help="restore the original vLLM files")

    serve = commands.add_parser(
        "serve",
        allow_abbrev=False,
        help="serve a VTR checkpoint with the released settings",
        description="Other arguments are passed to `vllm serve` and override the defaults.",
    )
    serve.add_argument("model")
    serve.add_argument("--fast", action="store_true", help="gate cache + prefix caching + chunked prefill")
    serve.add_argument("--dense", action="store_true", help="serve without routing (all image tokens)")
    serve.add_argument("--router-block-size", type=int, help="dense-anchor interval (default: checkpoint's, 10)")
    serve.add_argument("--hf-overrides", default="{}", help="JSON merged into the VTR overrides")
    serve.add_argument("--dry-run", action="store_true", help="print the vllm command and exit")

    args, extra = parser.parse_known_args()
    if args.command == "patch":
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        states = patching.status() if args.check else patching.apply(revert=args.revert)
        for path, state in states.items():
            print(f"{state:9} {path}")
        sys.exit(
            0 if all(s == (patching.PRISTINE if args.revert else patching.PATCHED) for s in states.values()) else 1
        )

    command = serve_command(args, extra)
    if args.dry_run:
        print(shlex.join(command))
        return
    patching.require_patched()
    os.execv(sys.executable, [sys.executable, "-m", "vllm.entrypoints.cli.main", *command[1:]])
