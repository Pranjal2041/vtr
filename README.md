<div align="center">
    <h1>Visual Token Routing: Learning to Keep Tokens That Matter</h1>
    <a href="https://pranjal2041.github.io/vtr/"><img src="https://img.shields.io/website?down_message=down&style=for-the-badge&up_message=up&url=https%3A%2F%2Fpranjal2041.github.io%2Fvtr%2F"></a>
    <a href="https://arxiv.org/abs/2611.06854"><img src="https://img.shields.io/badge/arXiv-2611.06854-red.svg?style=for-the-badge"></a>
    <a href="https://huggingface.co/l3lab/VTR-2B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-blue?style=for-the-badge&logo=huggingface"></a>
    <a href="https://github.com/pranjal2041/vtr"><img src="https://img.shields.io/badge/GitHub-Code-black?style=for-the-badge&logo=github"></a>
    <br>
</div>

<br>

VTR trains a small router together with a vision-language model. The router decides which visual tokens enter the language model. VTR matches the full-resolution model on OSWorld, CUA-World and video game action prediction with 2.5 to 5.8× fewer visual tokens.

## Quickstart

You need Linux, an NVIDIA GPU, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Install
git clone https://github.com/pranjal2041/vtr.git
cd vtr/inference
uv sync
uv run vtr patch

# 2. Start the server
uv run vtr serve l3lab/VTR-2B --port 8000
```

The server has an OpenAI-compatible API. See [inference/README.md](inference/README.md) for a sample request and all options.

## Train Your Own Model

You can skip this if you use our model. See [training/README.md](training/README.md) to prepare data, train and export.

## Acknowledgments

We thank the Qwen team for Qwen3.5-2B. This code is built on [vLLM](https://github.com/vllm-project/vllm) and [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl).

## Citation

If you use VTR in your research, please cite:

```bibtex
@misc{aggarwal2026vtr,
  title={Visual Token Routing: Learning to Keep Tokens That Matter},
  author={Pranjal Aggarwal and Sean Welleck},
  year={2026},
  eprint={2611.06854},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2611.06854},
}
```
