# YAML tools

## generate

Pair FASTA protein sequences with a ligand SMILES to write Boltz YAML inputs.

```bash
boltz2parallel yaml generate -i proteins.fasta -o ./yaml_out \
    --ligand LIG --smiles "CCO"
```

## add-msa

Attach MSA file paths to protein entries (filename → UniProt accession convention).

```bash
boltz2parallel yaml add-msa \
    --yaml-dir ./yaml_input --msa-dir ./msa --msa-ext a3m
```

See each subcommand's `--help` for full flags.
