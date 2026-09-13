"""Attach MSA paths to Boltz YAML protein entries."""

#!/usr/bin/env python3
"""
Add MSA file paths to Boltz style YAML inputs.

This script scans a directory of Boltz YAML files, parses a UniProt
accession from each filename, locates the matching MSA file in a
separate directory, and writes the MSA path into every protein entry
of the YAML.

Filename convention
    YAML : prefix_UNIPROTID.yaml or .yml, for example Ligand_P12345.yaml.
           The trailing underscore separated token is treated as the
           UniProt accession.
    MSA  : UNIPROTID.<ext> in the MSA directory. The extension defaults
           to a3m and can be changed with the --msa-ext flag.

Notes
    PyYAML does not preserve YAML comments. Comments in the input files
    will be lost when the files are rewritten.

    When a YAML contains several protein entries with different
    sequences, the same MSA path is applied to all of them and a
    warning is printed. The user should review such files before
    running predictions.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# Default MSA extension if --msa-ext is not provided.
DEFAULT_MSA_EXT = "a3m"

# Minimum length accepted for a UniProt-like accession parsed from the
# trailing token of a YAML filename. Standard UniProt accessions are
# six or ten characters long.
MIN_UNIPROT_LEN = 6


def parse_uniprot_id(filename: str) -> Optional[str]:
    """Extract the UniProt accession from a YAML filename.

    The filename stem is split on the last underscore and the right
    part is treated as the accession. The accession must be at least
    MIN_UNIPROT_LEN characters long, contain only uppercase letters
    and digits, and contain at least one digit. Anything else returns
    None so that the caller can skip the file.

    Args:
        filename: Bare filename, with or without an extension.

    Returns:
        The UniProt accession, or None if the filename does not match
        the expected layout.
    """
    stem = Path(filename).stem
    if "_" not in stem:
        return None

    suffix = stem.rsplit("_", 1)[1]
    if len(suffix) < MIN_UNIPROT_LEN:
        return None
    if not all(c.isupper() or c.isdigit() for c in suffix):
        return None
    if not any(c.isdigit() for c in suffix):
        return None
    if not any(c.isupper() for c in suffix):
        return None
    return suffix


def find_msa_file(msa_dir: Path, uniprot_id: str, ext: str) -> Optional[Path]:
    """Return the absolute path of the MSA file for a UniProt ID.

    Args:
        msa_dir: Directory containing MSA files. Expected to be already
            resolved to an absolute path by the caller.
        uniprot_id: UniProt accession parsed from the YAML filename.
        ext: MSA file extension without the leading dot.

    Returns:
        Absolute path to the MSA file, or None if it does not exist.
    """
    candidate = msa_dir / f"{uniprot_id}.{ext}"
    return candidate if candidate.is_file() else None


def update_yaml_content(
    yaml_data: Dict[str, Any],
    msa_path: str,
    overwrite: bool,
) -> Tuple[Dict[str, Any], int, int, bool]:
    """Insert the MSA path into every protein entry in the YAML data.

    Args:
        yaml_data: Parsed YAML mapping, modified in place.
        msa_path: MSA path to write into each protein entry.
        overwrite: If False, protein entries that already carry an msa
            field are left unchanged.

    Returns:
        A tuple of (yaml_data, applied, kept_existing, sequences_differ),
        where applied counts entries that received the MSA path,
        kept_existing counts entries left unchanged because they
        already had an msa field, and sequences_differ is True when
        the YAML contains protein entries with non identical sequences.

    Raises:
        ValueError: If the YAML lacks the expected top level structure
            or contains no protein entry.
    """
    if not isinstance(yaml_data, dict):
        raise ValueError("YAML root must be a mapping")
    if "sequences" not in yaml_data:
        raise ValueError("YAML does not contain a top level sequences key")

    sequences = yaml_data["sequences"]
    if not isinstance(sequences, list):
        raise ValueError("sequences must be a list")

    protein_entries: List[Dict[str, Any]] = [
        s["protein"]
        for s in sequences
        if isinstance(s, dict)
        and "protein" in s
        and isinstance(s["protein"], dict)
    ]
    if not protein_entries:
        raise ValueError("No protein entry found in sequences")

    seen_sequences = {p.get("sequence", "") for p in protein_entries}
    sequences_differ = len(seen_sequences) > 1

    applied = 0
    kept_existing = 0
    for protein in protein_entries:
        if "msa" in protein and protein["msa"] not in (None, "") and not overwrite:
            kept_existing += 1
            continue
        protein["msa"] = msa_path
        applied += 1

    return yaml_data, applied, kept_existing, sequences_differ


def collect_yaml_files(yaml_dir: Path, recursive: bool) -> List[Path]:
    """Return a sorted list of YAML files under yaml_dir."""
    extensions = {".yaml", ".yml"}
    iterator = yaml_dir.rglob("*") if recursive else yaml_dir.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in extensions
    )


def resolve_output_path(
    yaml_path: Path,
    yaml_dir: Path,
    output_dir: Optional[Path],
    recursive: bool,
) -> Path:
    """Compute where the updated YAML should be written."""
    if output_dir is None:
        return yaml_path
    if recursive:
        return output_dir / yaml_path.relative_to(yaml_dir)
    return output_dir / yaml_path.name


def write_yaml(target: Path, data: Dict[str, Any]) -> None:
    """Write YAML content to target with stable key order."""
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
            sort_keys=False,
        )


def process_yaml_files(
    yaml_dir: Path,
    msa_dir: Path,
    output_dir: Optional[Path],
    msa_ext: str,
    overwrite_msa: bool,
    dry_run: bool,
    recursive: bool,
    verbose: bool,
) -> int:
    """Iterate over YAML files in a directory and add MSA paths.

    Returns:
        Process exit code: 0 when no failures occurred, 1 otherwise.
    """
    yaml_files = collect_yaml_files(yaml_dir, recursive)
    if not yaml_files:
        print(f"No YAML files found under {yaml_dir}")
        return 0

    print(f"Found {len(yaml_files)} YAML file(s) under {yaml_dir}")
    if dry_run:
        print("Dry run mode: no files will be written")

    counters = {
        "processed": 0,
        "skipped_filename": 0,
        "skipped_no_msa": 0,
        "skipped_existing_msa": 0,
        "failed_parse": 0,
        "failed_structure": 0,
        "failed_write": 0,
        "warned_seq_diff": 0,
    }

    for yaml_path in yaml_files:
        rel = (
            yaml_path.relative_to(yaml_dir)
            if recursive else Path(yaml_path.name)
        )

        uniprot_id = parse_uniprot_id(yaml_path.name)
        if uniprot_id is None:
            print(f"Skip {rel}: filename does not match expected pattern")
            counters["skipped_filename"] += 1
            continue

        msa_path = find_msa_file(msa_dir, uniprot_id, msa_ext)
        if msa_path is None:
            print(f"Skip {rel}: no MSA found for UniProt ID {uniprot_id}")
            counters["skipped_no_msa"] += 1
            continue

        try:
            with yaml_path.open("r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Failed to parse {rel}: {e}")
            counters["failed_parse"] += 1
            continue
        except OSError as e:
            print(f"Failed to read {rel}: {e}")
            counters["failed_parse"] += 1
            continue

        try:
            updated, applied, kept, seq_diff = update_yaml_content(
                yaml_data, str(msa_path), overwrite=overwrite_msa
            )
        except ValueError as e:
            print(f"Failed to update {rel}: {e}")
            counters["failed_structure"] += 1
            continue

        if applied == 0 and kept > 0:
            print(
                f"Skip {rel}: all protein entries already have an MSA "
                f"(use --overwrite-msa to replace)"
            )
            counters["skipped_existing_msa"] += 1
            continue

        if seq_diff:
            print(
                f"Warning: {rel} contains protein entries with different "
                f"sequences; the same MSA was applied to all of them"
            )
            counters["warned_seq_diff"] += 1

        target = resolve_output_path(yaml_path, yaml_dir, output_dir, recursive)

        if dry_run:
            print(
                f"Would update {rel} with MSA {msa_path} "
                f"(applied={applied}, kept={kept})"
            )
            counters["processed"] += 1
            continue

        try:
            write_yaml(target, updated)
        except OSError as e:
            print(f"Failed to write {rel}: {e}")
            counters["failed_write"] += 1
            continue

        if verbose:
            print(
                f"Updated {rel} with MSA {msa_path} "
                f"(applied={applied}, kept={kept})"
            )
        else:
            print(f"Updated {rel}")
        counters["processed"] += 1

    print()
    print("Summary")
    print(f"  Processed              : {counters['processed']}")
    print(f"  Skipped (filename)     : {counters['skipped_filename']}")
    print(f"  Skipped (no MSA found) : {counters['skipped_no_msa']}")
    print(f"  Skipped (existing MSA) : {counters['skipped_existing_msa']}")
    print(f"  Failed (parse or read) : {counters['failed_parse']}")
    print(f"  Failed (structure)     : {counters['failed_structure']}")
    print(f"  Failed (write)         : {counters['failed_write']}")
    if counters["warned_seq_diff"]:
        print(f"  Warnings (seq differ)  : {counters['warned_seq_diff']}")

    failed_total = (
        counters["failed_parse"]
        + counters["failed_structure"]
        + counters["failed_write"]
    )
    return 0 if failed_total == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Add MSA file paths to Boltz YAML inputs by matching "
            "UniProt IDs in filenames against MSA files in a directory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "YAML filenames must end with an underscore followed by the "
            "UniProt accession, for example Ligand_P12345.yaml. The MSA "
            "for accession P12345 is expected at MSA_DIR/P12345.a3m by "
            "default; use --msa-ext to change the extension.\n"
            "\n"
            "When a YAML contains multiple protein entries with "
            "different sequences, the same MSA path is written to all "
            "of them and a warning is printed."
        ),
    )

    parser.add_argument(
        "-y", "--yaml-dir",
        type=Path,
        required=True,
        help="Directory containing input YAML files",
    )
    parser.add_argument(
        "-m", "--msa-dir",
        type=Path,
        required=True,
        help="Directory containing MSA files",
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory to write updated YAML files into. If omitted, "
            "the input files are overwritten in place."
        ),
    )
    parser.add_argument(
        "--msa-ext",
        type=str,
        default=DEFAULT_MSA_EXT,
        help=(
            f"MSA file extension without the leading dot "
            f"(default: {DEFAULT_MSA_EXT})"
        ),
    )
    parser.add_argument(
        "--overwrite-msa",
        action="store_true",
        help=(
            "Replace existing msa fields in protein entries. Without "
            "this flag, files where every protein entry already has an "
            "MSA are left unchanged."
        ),
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search the YAML directory recursively",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report planned actions without writing any file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed information for each processed file",
    )

    args = parser.parse_args()

    yaml_dir = args.yaml_dir.resolve()
    msa_dir = args.msa_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None

    if not yaml_dir.is_dir():
        print(f"YAML directory does not exist: {yaml_dir}")
        return 2
    if not msa_dir.is_dir():
        print(f"MSA directory does not exist: {msa_dir}")
        return 2

    if output_dir is not None:
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"Cannot create output directory {output_dir}: {e}")
            return 2

    msa_ext = args.msa_ext.lstrip(".").strip()
    if not msa_ext:
        print("MSA extension cannot be empty")
        return 2

    if args.verbose:
        print(f"YAML directory   : {yaml_dir}")
        print(f"MSA directory    : {msa_dir}")
        print(f"Output directory : {output_dir if output_dir else 'in place'}")
        print(f"MSA extension    : {msa_ext}")
        print(f"Recursive        : {args.recursive}")
        print(f"Overwrite MSA    : {args.overwrite_msa}")
        print(f"Dry run          : {args.dry_run}")
        print()

    try:
        return process_yaml_files(
            yaml_dir=yaml_dir,
            msa_dir=msa_dir,
            output_dir=output_dir,
            msa_ext=msa_ext,
            overwrite_msa=args.overwrite_msa,
            dry_run=args.dry_run,
            recursive=args.recursive,
            verbose=args.verbose,
        )
    except KeyboardInterrupt:
        print("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
