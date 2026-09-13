# Boltz2Parallel

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Boltz](https://img.shields.io/badge/Boltz-2-green.svg)](https://github.com/jwohlwend/boltz)

Profile-driven multi-GPU scheduling for [Boltz](https://github.com/jwohlwend/boltz) (Boltz-1 / Boltz-2) inference — the same design principles as [AF3Parallel](https://github.com/Xin-DongXu/AF3Parallel), adapted to Boltz YAML inputs and Singularity execution.

> Companion to the AF3Parallel Applications Note (BIOINF-2026-2000). This repository demonstrates that the **VRAM-profile + LPT + temporal-wave** scheduler is not AF3-specific.

---

## What it does

| Component | Role |
| --- | --- |
| `boltz2parallel.py` | Multi-GPU executor: length/token-aware LPT distribution, VRAM packing, temporal-wave co-scheduling |
| `scripts/Boltz2_GPU_stat.py` | Peak VRAM / runtime profiling → TSV profile |
| `scripts/Boltz2_GPU_memory_timeseries.py` | Sub-second VRAM time series during runs |
| `scripts/Boltz2_YAML_Generator.py` | Build Boltz YAML inputs from sequences / complexes |
| `scripts/Boltz2_Add_MSA_to_YAML.py` | Attach MSA fields to YAML for reuse workflows |
| `profiles/Boltz_A800_stat_All_Len_Checked_2.tsv` | Measured A800 80 GB envelope profile (1,557 points) |

Built-in default is a **linear** memory/runtime model fit on A800 (`slope_mem≈31.14 MB/token`). Pass `--legacy-step-model` for the stepwise staircase, or `--memory-profile` for a custom TSV.

---

## Requirements

- Linux + NVIDIA GPU(s), NVIDIA driver / CUDA stack compatible with your Boltz Singularity image
- Python ≥ 3.8
- `PyYAML` (`pip install pyyaml`)
- A Boltz Singularity `.sif` and Boltz cache directory

---

## Quick start

```bash
pip install pyyaml

# Optional: profile your GPU once
python scripts/Boltz2_GPU_stat.py -h

# Run the scheduler (auto-detect GPUs; temporal waves ON by default)
python boltz2parallel.py \
  -i ./yaml_input \
  -o results.tsv \
  --sif /path/to/boltz.sif \
  --boltz-cache ~/.boltz \
  --gpus 0,1,2,3 \
  --gpu-preset a800-80g

# Ablation: packing without temporal waves
python boltz2parallel.py ... --no-temporal-waves
```

Extra Boltz CLI flags:

```bash
python boltz2parallel.py ... --boltz-extra-args --recycling_steps 10 --diffusion_samples 5
```

---

## Relation to AF3Parallel

| Concept | AF3Parallel | Boltz2Parallel |
| --- | --- | --- |
| Inputs | AF3 JSON | Boltz YAML |
| Token rule | AF3 residue + ligand heavy atoms | Sequence length (protein/NA/ligand as in YAML) |
| Runner | `run_alphafold.py` via Singularity | `boltz predict` via Singularity |
| Cross-GPU | LPT by tokens | LPT by tokens / length |
| Per-GPU | VRAM packing + temporal waves | Same |
| Profile | Stepwise AF3 A800 / 4090 | Linear (default) or stepwise Boltz A800 |

AF3Parallel remains the manuscript-facing production package (PyPI `af3parallel`). Boltz2Parallel is released so reviewers and users can see the same scheduling idea on a second structure model without modifying Boltz itself.

---

## Repository layout

```
Boltz2Parallel/
  boltz2parallel.py          # main scheduler (canonical)
  profiles/                  # measured A800 profile
  scripts/                   # profiling + YAML helpers
  examples/data/             # optional memory timeseries sample
  archive/                   # prior drafts (not for production use)
  LICENSE
  README.md
  requirements.txt
  .gitignore
```

---

## Citation

If you use this scheduler together with AF3Parallel, please cite the AF3Parallel Applications Note (manuscript ID BIOINF-2026-2000) and this repository URL.

---

## License

MIT — see [LICENSE](LICENSE).
