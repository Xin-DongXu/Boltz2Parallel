# Installation

## Prerequisites

1. A working [Boltz](https://github.com/jwohlwend/boltz) Singularity image and cache.
2. Linux + NVIDIA GPU(s) with a driver/CUDA stack compatible with that image.
3. Python ≥ 3.8.

## pip (from source)

```bash
git clone https://github.com/Xin-DongXu/Boltz2Parallel.git
cd Boltz2Parallel
pip install .
# optional
pip install ".[extras]"
```

Editable install for development:

```bash
pip install -e ".[extras]"
```

Verify:

```bash
boltz2parallel --version
boltz2parallel --help
python -m boltz2parallel --help
```
