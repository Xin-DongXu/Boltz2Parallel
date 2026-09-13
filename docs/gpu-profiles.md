# GPU profiles

Boltz2Parallel ships a measured A800 80 GB envelope profile:

- Package data: `boltz2parallel/data/Boltz_A800_stat_All_Len_Checked_2.tsv`
- Default scheduler model: **linear** fit (`mem ≈ 31.14 × tokens − 1589`, with floors)
- Optional: `--legacy-step-model` for the stepwise staircase derived from the same data

GPU presets (`--gpu-preset`) set VRAM capacity; only the A800 profile is measured.
Other preset names are VRAM-matched fallbacks — run `boltz2parallel profile` on your
hardware before production use.

External profile:

```bash
boltz2parallel run ... --memory-profile my_gpu_profile.tsv
```
