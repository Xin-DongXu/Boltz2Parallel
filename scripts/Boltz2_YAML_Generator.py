#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Boltz-2 batch YAML generator.

Generates YAML input files for Boltz-2 protein-ligand affinity prediction
by pairing protein sequences from FASTA input with a ligand SMILES string.

Input modes (mutually exclusive):
  -i / --input-file   Single FASTA file (one or many sequences).
  -d / --input-dir    Directory of FASTA files (.fasta, .fa, .fas, .faa);
                      sequences are collected from all matching files.

Output naming:
  {ligand_name}_{protein_id}.yaml
  Optionally with a user-supplied --suffix appended before the extension.

Author: Xin-Dong Xu
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Recognised FASTA file extensions (lower-case; upper-case variants are also
# checked when scanning a directory).
FASTA_EXTENSIONS: Tuple[str, ...] = (".fasta", ".fa", ".fas", ".faa")


# ---------------------------------------------------------------------------
# FASTA parsing
# ---------------------------------------------------------------------------

def parse_fasta_file(fasta_path: str) -> List[Tuple[str, str]]:
    """
    Parse a FASTA file and return all (sequence_id, sequence) pairs it contains.

    The function handles both single-sequence files and multi-sequence files
    (one or more '>' headers per file).  The sequence ID is the first
    whitespace-delimited token following the '>' character.

    Args:
        fasta_path: Path to the FASTA file.

    Returns:
        List of (sequence_id, sequence) tuples.  Returns an empty list if the
        file cannot be read or contains no valid sequences.
    """
    records: List[Tuple[str, str]] = []
    current_id: Optional[str] = None
    current_seq: List[str] = []

    try:
        with open(fasta_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    # Flush the previous record before starting a new one.
                    if current_id is not None:
                        records.append((current_id, "".join(current_seq)))
                    # Take only the first token as the sequence ID.
                    current_id = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            # Flush the final record.
            if current_id is not None:
                records.append((current_id, "".join(current_seq)))

    except FileNotFoundError:
        logger.error(f"FASTA file not found: {fasta_path}")
    except Exception as exc:
        logger.error(f"Error reading FASTA file '{fasta_path}': {exc}")

    return records


def collect_sequences_from_file(fasta_file: str) -> List[Tuple[str, str]]:
    """
    Collect all protein sequences from a single FASTA file.

    Supports both single-sequence and multi-sequence FASTA files.

    Args:
        fasta_file: Path to the FASTA file.

    Returns:
        List of (sequence_id, sequence) tuples.
    """
    path = Path(fasta_file)
    if not path.is_file():
        logger.error(f"Input file does not exist or is not a regular file: {fasta_file}")
        return []

    logger.info(f"Reading FASTA file: {path}")
    records = parse_fasta_file(str(path))
    logger.info(f"  -> {len(records)} sequence(s) found in {path.name}")
    return records


def collect_sequences_from_directory(fasta_dir: str) -> List[Tuple[str, str]]:
    """
    Collect all protein sequences from every FASTA file inside a directory.

    Files with extensions .fasta, .fa, .fas, .faa (case-insensitive) are
    collected.  Sub-directories are not traversed.

    Args:
        fasta_dir: Path to the directory containing FASTA files.

    Returns:
        List of (sequence_id, sequence) tuples aggregated from all files.
    """
    dir_path = Path(fasta_dir)
    if not dir_path.is_dir():
        logger.error(f"Input directory does not exist: {fasta_dir}")
        return []

    # Gather files matching any recognised extension (case-insensitive).
    fasta_files: List[Path] = []
    for child in sorted(dir_path.iterdir()):
        if child.is_file() and child.suffix.lower() in FASTA_EXTENSIONS:
            fasta_files.append(child)

    if not fasta_files:
        logger.warning(
            f"No FASTA files found in directory '{fasta_dir}'. "
            f"Recognised extensions: {', '.join(FASTA_EXTENSIONS)}"
        )
        return []

    logger.info(f"Found {len(fasta_files)} FASTA file(s) in '{fasta_dir}'")

    all_records: List[Tuple[str, str]] = []
    for fasta_file in fasta_files:
        logger.info(f"Processing: {fasta_file.name}")
        records = parse_fasta_file(str(fasta_file))
        logger.info(f"  -> {len(records)} sequence(s) found")
        all_records.extend(records)

    return all_records


# ---------------------------------------------------------------------------
# YAML content generation
# ---------------------------------------------------------------------------

def generate_yaml_content(
    protein_sequence: str,
    smiles: str,
) -> Dict:
    """
    Build the Boltz-2 YAML content dict for a single protein-ligand pair.

    The protein is always assigned chain ID 'A' and the ligand chain ID 'B'.
    Affinity prediction is requested for the ligand chain ('B').

    Args:
        protein_sequence: Amino-acid sequence string.
        smiles:           SMILES string for the ligand.

    Returns:
        Dictionary representing the complete YAML content.
    """
    return {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": protein_sequence,
                }
            },
            {
                "ligand": {
                    "id": "B",
                    "smiles": smiles,
                }
            },
        ],
        "properties": [
            {
                "affinity": {
                    "binder": "B",
                }
            }
        ],
    }


def save_yaml_file(content: Dict, output_path: str) -> bool:
    """
    Write a YAML content dict to disk.

    Args:
        content:     Dictionary to serialise as YAML.
        output_path: Destination file path.

    Returns:
        True on success, False on failure.
    """
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            yaml.dump(
                content,
                fh,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                indent=2,
            )
        return True
    except Exception as exc:
        logger.error(f"Failed to write '{output_path}': {exc}")
        return False


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_smiles(smiles: str) -> bool:
    """
    Perform a basic sanity check on a SMILES string.

    Checks that the string is non-empty and contains no whitespace characters.
    Full chemical validation is outside the scope of this script.

    Args:
        smiles: SMILES string to validate.

    Returns:
        True if the string passes basic checks, False otherwise.
    """
    if not smiles or not isinstance(smiles, str):
        return False
    return not any(ch in smiles for ch in (" ", "\t", "\n", "\r"))


def make_safe_filename_part(text: str) -> str:
    """
    Sanitise a string so that it is safe to embed in a file name.

    Retains only alphanumeric characters, hyphens, and underscores.

    Args:
        text: Raw string (e.g. sequence ID or ligand name).

    Returns:
        Sanitised string suitable for use in a file name.
    """
    return "".join(ch for ch in text if ch.isalnum() or ch in ("_", "-"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Boltz-2 YAML input files for protein-ligand affinity "
            "prediction.  Accepts either a single (multi-)FASTA file or a "
            "directory of FASTA files as protein sequence input."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input modes (exactly one of -i or -d is required):

  Single FASTA file (may contain multiple sequences):
    %(prog)s -i proteins.fasta -o yaml_out/ -l Aspirin -s "CC(=O)Oc1ccccc1C(=O)O"

  Directory of FASTA files:
    %(prog)s -d ./fasta_dir   -o yaml_out/ -l Aspirin -s "CC(=O)Oc1ccccc1C(=O)O"

Output file naming:
  {ligand_name}_{protein_id}.yaml

Optional flags:
  --suffix        Append an extra suffix:  {ligand_name}_{protein_id}_{suffix}.yaml
  --dry-run       Print which files would be created without writing anything.
  -v / --verbose  Show DEBUG-level log messages.
        """,
    )

    # ---- input (mutually exclusive) ----------------------------------------
    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        "-i", "--input-file",
        type=str,
        metavar="FASTA_FILE",
        help=(
            "Path to a single FASTA file.  The file may contain one or more "
            "sequences; one YAML file is generated for each sequence."
        ),
    )

    input_group.add_argument(
        "-d", "--input-dir",
        type=str,
        metavar="FASTA_DIR",
        help=(
            "Path to a directory containing FASTA files "
            "(.fasta, .fa, .fas, .faa).  All matching files are processed and "
            "sequences are collected across all of them."
        ),
    )

    # ---- required output / ligand arguments --------------------------------
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        required=True,
        metavar="OUTPUT_DIR",
        help="Directory where the generated YAML files will be written.",
    )

    parser.add_argument(
        "-l", "--ligand-name",
        type=str,
        required=True,
        metavar="LIGAND_NAME",
        help=(
            "Name of the ligand.  Used as the prefix in output file names: "
            "{ligand_name}_{protein_id}.yaml"
        ),
    )

    parser.add_argument(
        "-s", "--smiles",
        type=str,
        required=True,
        metavar="SMILES",
        help='SMILES string representing the ligand (e.g. "CC(=O)Oc1ccccc1C(=O)O").',
    )

    # ---- optional arguments ------------------------------------------------
    parser.add_argument(
        "--suffix",
        type=str,
        default=None,
        metavar="SUFFIX",
        help=(
            "Optional suffix appended to output file names: "
            "{ligand_name}_{protein_id}_{suffix}.yaml"
        ),
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging.",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        metavar="LOG_FILE",
        help="Optional path to a log file.  When omitted, log messages are "
             "written only to stdout.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Simulate the run: log which files would be created without "
            "actually writing any output."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    # Verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose mode enabled.")

    # Optional file handler
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.info(f"Log file: {os.path.abspath(args.log_file)}")

    # Validate SMILES
    if not validate_smiles(args.smiles):
        logger.error(
            "The provided SMILES string is invalid (empty or contains whitespace)."
        )
        return 1

    # Collect sequences from the chosen input mode
    if args.input_file:
        logger.info("Input mode: single FASTA file")
        sequences = collect_sequences_from_file(args.input_file)
    else:
        logger.info("Input mode: directory of FASTA files")
        sequences = collect_sequences_from_directory(args.input_dir)

    if not sequences:
        logger.error("No protein sequences were found.  Aborting.")
        return 1

    logger.info(f"Total sequences to process: {len(sequences)}")

    # Prepare output directory
    output_path = Path(args.output_dir)
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_path.absolute()}")

    # Sanitise the ligand name once (reused for every file)
    safe_ligand = make_safe_filename_part(args.ligand_name)
    safe_suffix = make_safe_filename_part(args.suffix) if args.suffix else None

    success_count = 0
    failed_count  = 0
    skipped_count = 0

    used_filenames: set = set()

    for protein_id, protein_sequence in sequences:
        if not protein_sequence:
            logger.warning(f"Sequence '{protein_id}' is empty; skipping.")
            skipped_count += 1
            continue

        safe_pid = make_safe_filename_part(protein_id)
        if not safe_pid:
            # All characters were stripped by sanitisation; fall back to a
            # stable hash-derived stem so different IDs do not collide.
            import hashlib
            safe_pid = "id" + hashlib.sha1(
                protein_id.encode("utf-8")).hexdigest()[:8]
            logger.warning(
                f"Sequence ID '{protein_id}' contained no filename-safe "
                f"characters; using fallback '{safe_pid}'.")

        # Build output file name
        if safe_suffix:
            yaml_filename = f"{safe_ligand}_{safe_pid}_{safe_suffix}.yaml"
        else:
            yaml_filename = f"{safe_ligand}_{safe_pid}.yaml"

        # Detect collisions: distinct sequence IDs that sanitise to the same
        # filename would silently overwrite each other.  Append a counter.
        if yaml_filename in used_filenames:
            stem, ext = os.path.splitext(yaml_filename)
            counter = 2
            while f"{stem}_{counter}{ext}" in used_filenames:
                counter += 1
            collided = yaml_filename
            yaml_filename = f"{stem}_{counter}{ext}"
            logger.warning(
                f"Filename collision for sequence '{protein_id}': "
                f"'{collided}' already used; writing as '{yaml_filename}'.")
        used_filenames.add(yaml_filename)

        yaml_filepath = output_path / yaml_filename

        if args.dry_run:
            logger.info(f"[dry-run] Would create: {yaml_filepath}")
            success_count += 1
            continue

        # Build YAML content
        content = generate_yaml_content(protein_sequence, args.smiles)

        # Write to disk
        if save_yaml_file(content, str(yaml_filepath)):
            logger.debug(f"Created: {yaml_filepath}")
            success_count += 1
        else:
            failed_count += 1

    # Summary
    logger.info("")
    logger.info("=" * 50)
    logger.info("Batch processing complete")
    logger.info(f"  Successfully created : {success_count}")
    logger.info(f"  Failed               : {failed_count}")
    if skipped_count:
        logger.info(f"  Skipped (empty seq)  : {skipped_count}")
    if args.dry_run:
        logger.info("  [dry-run] No files were actually written.")
    logger.info("=" * 50)

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
