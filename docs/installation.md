# Installation

Octorch uses [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:ZW471/octorch.git
cd octorch
uv sync                    # torch (cu126 wheels), numpy, tqdm
```

Optional extras: `--extra dev` (pytest), `--extra training` (pyyaml, gymnasium), `--extra gui`
(pygame, opencv, pillow, matplotlib), `--extra validate` (jax + octax for the cross-validation tests).

## CUDA version

`pyproject.toml` selects the PyTorch wheel index:

```toml
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu126" }
```

`cu126` matches NVIDIA driver 535+ (CUDA 12.6). Replace it with `cu128` / `cu130` for newer drivers or
with `https://download.pytorch.org/whl/cpu` for a CPU-only install, then run `uv sync` again.

## Checking the install

```bash
uv run python -c "import torch, octorch; print(torch.__version__, torch.cuda.is_available())"
uv run pytest -q
```
