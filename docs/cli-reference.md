# CLI reference

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

Common `run` flags mirror AF3Parallel: `--gpus`, `--gpu-preset`, `--safety-margin`,
`--vram-margin`, `--no-temporal-waves`, `--skip-vram-overflow`, `--boltz-extra-args`.
