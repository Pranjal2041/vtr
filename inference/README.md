# VTR Inference

Serves VTR with an OpenAI-compatible API, using a patched [vLLM](https://github.com/vllm-project/vllm). The router decides which image tokens enter the language model. Dropped tokens are removed from the prompt, so the prompt and the KV cache get smaller.

## Install

You need Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/). We tested on an NVIDIA L40S.

```bash
uv sync
uv run vtr patch
```

`uv sync` installs the pinned vLLM version. `vtr patch` applies `src/vtr_inference/vllm.patch` to the vLLM in this environment, and it changes vLLM for every model served from here. Use `vtr patch --check` to see its state and `vtr patch --revert` to undo it.

## Serve

```bash
uv run vtr serve l3lab/VTR-2B --port 8000
```

| Flag | What it does |
|---|---|
| `--fast` | Reuses earlier keep/drop choices so an agent's history stays in the prefix cache (see below). |
| `--dense` | Keeps all image tokens, for comparison. |
| `--router-block-size N` | Keeps images 0, N, 2N, ... whole. The default is 10. With 0, only the first image is kept whole. |
| `--dry-run` | Prints the full `vllm serve` command and exits. |

Any other argument goes to `vllm serve`.

### Standard and fast mode

- **Standard** is the default and matches the paper's evaluations. Every request samples a new keep/drop choice for each image, and prefix caching is off.
- **Fast** samples the choice for each image once and reuses it in later requests with the same prefix. An agent's growing history keeps its routing and its KV cache. Choices come from the same probabilities as in standard mode.

### Limits

Each server uses one GPU, so run more servers for more GPUs. Speculative decoding is not supported. All images in a request must have the same resolution.

## Send A Request

```python
import base64
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def image(path):
    data = base64.b64encode(open(path, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}

response = client.chat.completions.create(
    model="l3lab/VTR-2B",
    messages=[{"role": "user", "content": [
        image("step_0.png"), image("step_1.png"), image("step_2.png"),
        {"type": "text", "text": "What changed on the screen, and what should I click next?"},
    ]}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

To see how many image tokens each request kept, set `VLLM_LOGGING_LEVEL=DEBUG` when you start the server.
