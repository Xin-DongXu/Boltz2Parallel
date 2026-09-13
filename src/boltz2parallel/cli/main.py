"""Unified CLI dispatcher for all Boltz2Parallel tools."""

from __future__ import annotations

import importlib
import signal
import sys
from typing import Callable, Dict, Optional

from boltz2parallel.__version__ import __version__

_COMMANDS: Dict[str, tuple[str, str, bool]] = {
    "run": ("boltz2parallel.parallel", "main", True),
    "profile": ("boltz2parallel.gpu_memory_profiler", "main", False),
    "profile-ts": ("boltz2parallel.gpu_memory_timeseries_profiler", "main", False),
}

_USAGE = f"""\
Boltz2Parallel v{__version__} - Boltz multi-GPU toolkit

Usage:
  boltz2parallel <command> [arguments]

Commands:
  run         Multi-GPU batch executor (LPT + VRAM packing + temporal waves)
  profile     Peak VRAM / runtime profiler
  profile-ts  Time-series VRAM profiler
  yaml        YAML helpers (generate | add-msa)

Examples:
  boltz2parallel run -i ./yaml_input -o results.tsv --sif boltz.sif --boltz-cache ~/.boltz
  boltz2parallel profile -i ./yaml_input -o profile.tsv --sif boltz.sif
  boltz2parallel yaml generate -i proteins.fasta -o ./yaml_out --ligand ATP --smiles ...

Run 'boltz2parallel <command> --help' for command-specific options.
"""


def _print_usage() -> None:
    sys.stdout.write(_USAGE)


def _load_entry(module_path: str, func_name: str) -> Callable:
    module = importlib.import_module(module_path)
    return getattr(module, func_name)


def _install_parallel_signal_handlers() -> None:
    parallel = importlib.import_module("boltz2parallel.parallel")
    signal.signal(signal.SIGINT, parallel.signal_handler)
    signal.signal(signal.SIGTERM, parallel.signal_handler)


def _dispatch_yaml(rest: list[str]) -> int:
    if not rest or rest[0] in ("-h", "--help"):
        sys.stdout.write(
            "Usage: boltz2parallel yaml <generate|add-msa> [arguments]\n"
            "  generate   Build Boltz YAML from FASTA + ligand SMILES\n"
            "  add-msa    Attach MSA paths to existing YAML files\n"
        )
        return 0
    sub = rest[0]
    args = rest[1:]
    if sub == "generate":
        sys.argv = ["boltz2parallel-yaml-generate"] + args
        result = _load_entry("boltz2parallel.yaml_generator", "main")()
        return int(result) if result is not None else 0
    if sub in ("add-msa", "add_msa"):
        sys.argv = ["boltz2parallel-yaml-add-msa"] + args
        result = _load_entry("boltz2parallel.yaml_add_msa", "main")()
        return int(result) if result is not None else 0
    sys.stderr.write(f"boltz2parallel yaml: unknown subcommand {sub!r}\n")
    return 2


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch to the selected Boltz2Parallel subcommand."""
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help"):
        _print_usage()
        return 0

    if args[0] in ("-V", "--version"):
        sys.stdout.write(f"boltz2parallel {__version__}\n")
        return 0

    command = args[0]
    rest = args[1:]

    if command == "yaml":
        return _dispatch_yaml(rest)

    if command not in _COMMANDS:
        sys.stderr.write(f"boltz2parallel: unknown command {command!r}\n\n")
        _print_usage()
        return 2

    module_path, func_name, needs_signals = _COMMANDS[command]
    sys.argv = [f"boltz2parallel-{command}"] + rest

    if needs_signals:
        _install_parallel_signal_handlers()

    entry = _load_entry(module_path, func_name)
    result = entry()
    return int(result) if result is not None else 0
