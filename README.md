# Boltz2Parallel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Boltz](https://img.shields.io/badge/Boltz-2-green.svg)](https://github.com/jwohlwend/boltz)

Profile-driven toolkit for running [Boltz](https://github.com/jwohlwend/boltz) (Boltz-1 / Boltz-2) inference at scale on multi-GPU Linux clusters.

Boltz2Parallel wraps the official Boltz Singularity workflow with VRAM-aware scheduling, temporal-wave batching, and companion utilities for profiling and YAML preparation — the **same design** as [AF3Parallel](https://github.com/Xin-DongXu/AF3Parallel), adapted to Boltz YAML inputs.

---

## Overview

| Concept | AF3Parallel | Boltz2Parallel |
| --- | --- | --- |
| Inputs | AF3 JSON | Boltz YAML |
| Runner | `run_alphafold.py` | `boltz predict` |
| Cross-GPU | LPT by tokens | LPT by sequence length / tokens |
| Per-GPU | VRAM packing + temporal waves | Same |
| CLI | `af3parallel <cmd>` | `boltz2parallel <cmd>` |

---

## What's included

| Tool | CLI command | Purpose |
| --- | --- | --- |
| Multi-GPU executor | `boltz2parallel run` | Distribute Boltz jobs across GPUs with LPT, packing, and temporal waves |
| Peak VRAM profiler | `boltz2parallel profile` | One-shot peak-memory scan → TSV profile |
| Time-series profiler | `boltz2parallel profile-ts` | Sub-second VRAM sampling during Boltz runs |
| YAML generator | `boltz2parallel yaml generate` | Build Boltz YAML from FASTA + ligand |
| MSA attachment | `boltz2parallel yaml add-msa` | Write MSA paths into YAML protein entries |

Built-in default: measured **A800 80 GB** linear memory/runtime model (envelope fit). Other GPUs: run `boltz2parallel profile` once.

---

## Installation

### Prerequisites

Complete a Boltz installation first (Singularity `.sif`, model cache). Details: [docs/installation.md](docs/installation.md).

| Component | Required | Notes |
| --- | --- | --- |
| Boltz + Singularity | Yes | `--sif` and `--boltz-cache` |
| Linux + NVIDIA GPU | Yes | |
| Python ≥ 3.8 | Yes | |
| PyYAML | Yes | installed with the package |
| `psutil` | Optional | via `pip install ".[extras]"` |

### From source

```bash
git clone https://github.com/Xin-DongXu/Boltz2Parallel.git
cd Boltz2Parallel
pip install .
```

Verify:

```bash
boltz2parallel --version
boltz2parallel --help
```

---

## Quick start

```bash
# 1. Profile once per GPU model (optional on A800 with built-in linear model)
boltz2parallel profile \
    -i ./yaml_input -o my_gpu_profile.tsv \
    --sif boltz.sif --boltz-cache ~/.boltz

# 2. Run across GPUs
boltz2parallel run \
    -i ./yaml_input -o results.tsv --output-dir ./boltz_output \
    --sif boltz.sif --boltz-cache ~/.boltz \
    --gpus 0,1,2,3 --gpu-preset a800-80g
```

Disable temporal waves (packing only):

```bash
boltz2parallel run ... --no-temporal-waves
```

---

## Typical workflow

```
  Boltz YAML  ──►  boltz2parallel profile  ──►  TSV profile
       │                                           │
       └──────────────────────────►  boltz2parallel run  ──►  results.tsv
```

See [docs/workflow.md](docs/workflow.md).

---

## CLI reference

```bash
boltz2parallel <command> [arguments]
boltz2parallel run --help
python -m boltz2parallel --help
```

| Subcommand | Standalone alias |
| --- | --- |
| `run` | `boltz2parallel-run` |
| `profile` | `boltz2parallel-profile` |
| `profile-ts` | `boltz2parallel-profile-ts` |
| `yaml generate` | `boltz2parallel-yaml-generate` |
| `yaml add-msa` | `boltz2parallel-yaml-add-msa` |

Full guides: [docs/](docs/README.md)

---

## Features

- Token/length-balanced **LPT** multi-GPU distribution
- **VRAM-aware** batching with temporal-wave scheduling
- Built-in **A800** linear (+ optional stepwise) profile
- Streaming TSV logs, per-task retry, SIGINT cleanup
- YAML generation and MSA attachment helpers

---

## Relation to AF3Parallel

AF3Parallel ([PyPI](https://pypi.org/project/af3parallel/), [GitHub](https://github.com/Xin-DongXu/AF3Parallel)) is the manuscript-facing package for AlphaFold 3. Boltz2Parallel demonstrates that the same scheduler transfers to Boltz without modifying the Boltz runtime — answering the modularity question for BIOINF-2026-2000.

---

## License & citation

MIT License — see [LICENSE](LICENSE). Boltz is licensed separately by its authors.

See [CITATION.cff](CITATION.cff).
