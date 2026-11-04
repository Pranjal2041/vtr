# VTR Training

Trains the VTR router and language model on Qwen3.5-2B with [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), and exports a model in the same format as [l3lab/VTR-2B](https://huggingface.co/l3lab/VTR-2B) for [`../inference`](../inference) to serve.

## Install

You need Linux, NVIDIA GPUs with CUDA 13, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

## Prepare Data

Game data comes from [Gaming-500](https://huggingface.co/datasets/markov-ai/gaming-500-hours). Desktop data is your own: `desktop.jsonl` has one line per trajectory with `id`, `sha256`, `run_dir` and `frame_count`, and each run directory holds `observation_<step>.png`, `messages_step_<step>.json` and `responses.json`. `verified.jsonl` is the same format and lists the trajectories to train on.

List the game recordings and sample 40-frame trajectories from them.

```bash
uv run vtr-games
```

Download the clips and extract their frames. OpenCV 4.13 decodes some clips to black frames, so use 4.12 here.

```bash
uv run --with opencv-python-headless==4.12.0.88 vtr-frames
```

Pick the cold-start windows.

```bash
uv run vtr-windows --desktop desktop.jsonl
```

Ask LLMs for the concepts SAM 3 should find in each window. This needs `GEMINI_API_KEY` and `OPENAI_API_KEY`.

```bash
uv run vtr-concepts
```

Turn windows into per-patch labels with SAM 3.

```bash
uv run vtr-labels @ configs/labels.toml
```

Build the rows for joint training.

```bash
uv run vtr-rows --desktop verified.jsonl
```

## Train

Train the router alone on the labels, with frozen vision features.

```bash
uv run torchrun --nproc-per-node=4 -m vtr.coldstart.train @ configs/coldstart.toml
```

Combine the base model and the cold-start router into the checkpoint joint training starts from.

```bash
uv run vtr-init --router outputs/coldstart/router_best.pt --output checkpoints/init
```

Train the router and the language model together. Validation runs every 50 steps and checkpoints are saved every 500. Add `--resume` to continue from the latest checkpoint.

```bash
uv run torchrun --nproc-per-node=8 -m vtr.train @ configs/joint.toml
```

## Export

Write the checkpoint with the best validation loss as one Hugging Face model.

```bash
uv run vtr-export --checkpoint outputs/joint/checkpoints/step_2000 --base checkpoints/init --output checkpoints/vtr
```
