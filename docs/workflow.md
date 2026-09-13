# Typical workflow

```
  Boltz YAML inputs  ──►  boltz2parallel profile  ──►  TSV profile
         │                                                  │
         │                                                  ▼
         └──────────────────────────────►  boltz2parallel run  ──►  results.tsv
```

1. (Optional) build YAML inputs:

```bash
boltz2parallel yaml generate \
    -i proteins.fasta -o ./yaml_input \
    --ligand ATP --smiles "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"
```

2. Profile once per GPU model (skip if using the bundled A800 linear model):

```bash
boltz2parallel profile \
    -i ./yaml_input -o my_gpu_profile.tsv \
    --sif boltz.sif --boltz-cache ~/.boltz
```

3. Run the batch:

```bash
boltz2parallel run \
    -i ./yaml_input -o results.tsv --output-dir ./boltz_output \
    --sif boltz.sif --boltz-cache ~/.boltz \
    --gpus 0,1,2,3 --gpu-preset a800-80g
```

Ablation without temporal waves:

```bash
boltz2parallel run ... --no-temporal-waves
```
