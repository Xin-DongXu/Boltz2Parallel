"""Time-series VRAM profiler for Boltz runs."""

#!/usr/bin/env python3
"""Boltz-2 GPU Memory Time-Series Profiler.

Runs a curated set of representative Boltz prediction jobs through a
Singularity container and records sub-second GPU memory time series for
each job into a single TSV file.  The companion peak-memory TSV produced
by Boltz_GPU_parallel.py (or any equivalent file with sequence_length and
peak_memory_mb columns) drives the selection of representatives so that
the resulting curves cover the full input-size range observed in a
prediction campaign.

Pipeline
  1. Read a peak-memory stat TSV and bucket prior runs into token-width
     bins.  Boltz GPU memory scales approximately linearly with total
     token count, so uniform token bins are used by default rather than
     the memory-bucketing approach appropriate for step-wise allocators.
  2. Parse every YAML file in --input-dir in parallel and compute its
     total token count (protein + RNA + DNA residues plus ligand
     entities), matching the rules used by Boltz_GPU_parallel.py.
  3. For each token bin, pick up to --n-per-bin representative YAML files
     spread across the bin (low / mid / high token count).
  4. Run each representative through Singularity, sample GPU memory at
     a fixed sub-second interval through a background thread, and append
     the resulting time series to the output TSV.

Output TSV columns
  yaml_file | total_tokens | protein_length | rna_length | dna_length |
  ligand_count | elapsed_seconds | memory_used_mb | memory_total_mb |
  memory_percent | gpu_util | temperature | success | job_runtime_seconds

Stat file columns (input)
  Required: peak_memory_mb plus one of {sequence_length, token_count, tokens}.
  Extra columns are ignored.

Token counting rules (per Boltz documentation)
  Protein  : 1 token per amino acid residue, multiplied by the YAML count
             field if present (default 1).
  RNA / DNA: 1 token per nucleotide, multiplied by count.
  Ligand   : 1 token per ligand entity (multiplied by count).  This
             matches Boltz_GPU_parallel.py and is intentionally simpler
             than the heavy-atom rule used by AF3.

Requirements
  Python >= 3.8
  nvidia-smi in PATH
  singularity in PATH
  A Boltz Singularity image (.sif) and a populated Boltz cache directory
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple

import yaml


# ============================================================
# Console output helpers
# ============================================================

class _Colors:
    RED    = "\033[0;31m"
    GREEN  = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE   = "\033[0;34m"
    NC     = "\033[0m"


_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _emit(prefix: str, msg: str, color: str) -> None:
    if _USE_COLOR:
        print(f"{color}[{prefix}]{_Colors.NC} {msg}", flush=True)
    else:
        print(f"[{prefix}] {msg}", flush=True)


def info(msg: str) -> None:    _emit("INFO",  msg, _Colors.BLUE)
def warning(msg: str) -> None: _emit("WARN",  msg, _Colors.YELLOW)
def error(msg: str) -> None:   _emit("ERROR", msg, _Colors.RED)
def ok_msg(msg: str) -> None:  _emit("OK",    msg, _Colors.GREEN)


# ============================================================
# Boltz token counter
# ============================================================

def boltz_count_tokens(data: dict) -> Dict[str, int]:
    """Compute Boltz token counts for a parsed YAML object.

    Returns a dict with keys:
      protein_length, rna_length, dna_length, ligand_count,
      sequence_length (= protein + rna + dna),
      total_tokens    (= sequence_length, with a fallback to ligand_count
                       for pure-ligand inputs to keep total_tokens > 0).

    The convention matches Boltz_GPU_parallel.py so that stat files and
    profile fits are interchangeable between the two tools.
    """
    protein_length = 0
    rna_length = 0
    dna_length = 0
    ligand_count = 0

    sequences = (data or {}).get("sequences", []) or []
    for seq_entry in sequences:
        if not isinstance(seq_entry, dict):
            continue
        try:
            count = max(1, int(seq_entry.get("count", 1)))
        except (TypeError, ValueError):
            count = 1

        if "protein" in seq_entry:
            ent = seq_entry["protein"] or {}
            protein_length += len(str(ent.get("sequence", ""))) * count
        elif "rna" in seq_entry:
            ent = seq_entry["rna"] or {}
            rna_length += len(str(ent.get("sequence", ""))) * count
        elif "dna" in seq_entry:
            ent = seq_entry["dna"] or {}
            dna_length += len(str(ent.get("sequence", ""))) * count
        elif "ligand" in seq_entry:
            ligand_count += count

    sequence_length = protein_length + rna_length + dna_length
    total_tokens = sequence_length if sequence_length > 0 else ligand_count

    return {
        "protein_length":  protein_length,
        "rna_length":      rna_length,
        "dna_length":      dna_length,
        "ligand_count":    ligand_count,
        "sequence_length": sequence_length,
        "total_tokens":    total_tokens,
    }


# ============================================================
# Worker for parallel YAML parsing
# ============================================================

def _parse_worker(path_str: str) -> Dict:
    """Top-level (picklable) worker for ProcessPoolExecutor."""
    p = Path(path_str)
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        counts = boltz_count_tokens(data)
        name = data.get("name") if isinstance(data, dict) else None
        return {
            "yaml_file":    p.name,
            "yaml_path":    str(p),
            "yaml_name":    name,
            "tokens":       counts["total_tokens"],
            "protein_len":  counts["protein_length"],
            "rna_len":      counts["rna_length"],
            "dna_len":      counts["dna_length"],
            "ligand_count": counts["ligand_count"],
            "error":        None,
        }
    except (OSError, yaml.YAMLError) as exc:
        return {
            "yaml_file":    p.name,
            "yaml_path":    str(p),
            "yaml_name":    None,
            "tokens":       0,
            "protein_len":  0,
            "rna_len":      0,
            "dna_len":      0,
            "ligand_count": 0,
            "error":        str(exc),
        }


# ============================================================
# Stat file -> token bins
# ============================================================

_STAT_LEN_COLS = ("sequence_length", "token_count", "tokens", "total_tokens")


def parse_stat_file(stat_file: Path,
                    bin_width: int = 200,
                    ) -> List[Dict]:
    """Read a peak-memory TSV and divide it into token-width bins.

    Required columns (any one of):
        sequence_length, token_count, tokens, total_tokens
    Required column:
        peak_memory_mb
    Rows whose token or memory cell cannot be parsed as a number are
    silently skipped.

    Returns a list (sorted by min_token) of:
        {
            "bin_low":         int,    # inclusive left edge of bin
            "bin_high":        int,    # exclusive right edge of bin
            "min_token":       int,    # smallest token count actually seen
            "max_token":       int,    # largest token count actually seen
            "mean_memory_mb":  float,
            "token_list":      List[int],
        }
    """
    if bin_width <= 0:
        raise ValueError("bin_width must be a positive integer")

    rows: List[Tuple[int, float]] = []
    with open(stat_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{stat_file}: file is empty")
        len_col = next((c for c in _STAT_LEN_COLS
                        if c in reader.fieldnames), None)
        if len_col is None or "peak_memory_mb" not in reader.fieldnames:
            raise ValueError(
                f"{stat_file}: missing required columns. "
                f"Need 'peak_memory_mb' and one of {list(_STAT_LEN_COLS)}. "
                f"Got: {reader.fieldnames}"
            )
        for row in reader:
            try:
                tokens = int(float(row[len_col]))
                mem = float(row["peak_memory_mb"])
            except (KeyError, ValueError, TypeError):
                continue
            if tokens <= 0 or mem <= 0:
                continue
            rows.append((tokens, mem))

    if not rows:
        raise ValueError(f"No valid rows found in {stat_file}")

    # Bucket by floor(tokens / bin_width).
    buckets: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for tokens, mem in rows:
        key = (tokens // bin_width) * bin_width
        buckets[key].append((tokens, mem))

    bins: List[Dict] = []
    for key in sorted(buckets):
        items = buckets[key]
        token_list = sorted(t for t, _ in items)
        mems = [m for _, m in items]
        bins.append({
            "bin_low":        key,
            "bin_high":       key + bin_width,
            "min_token":      token_list[0],
            "max_token":      token_list[-1],
            "mean_memory_mb": sum(mems) / len(mems),
            "token_list":     token_list,
        })

    info(f"Detected {len(bins)} token bin(s) (width={bin_width}):")
    for b in bins:
        info(f"  tokens [{b['min_token']:>6}-{b['max_token']:>6}]  "
             f"->  ~{b['mean_memory_mb']:.0f} MB VRAM (mean over "
             f"{len(b['token_list'])} prior run(s))")
    return bins


# ============================================================
# Representative file selection
# ============================================================

def select_representatives(bins: List[Dict],
                           parsed: List[Dict],
                           n: int = 3,
                           ) -> List[Dict]:
    """For each token bin, pick up to n YAML files spread across the
    [low, mid, high] of its observed token range.

    Files whose token count falls inside a bin's [min_token, max_token]
    range are eligible.  Selection avoids duplicate token counts so that
    the resulting set spans as much of the bin as possible.  The full
    list is deduplicated by file name across bins.
    """
    if n <= 0:
        return []

    selected: List[Dict] = []
    seen: set = set()

    for b in bins:
        lo, hi = b["min_token"], b["max_token"]
        cands = sorted(
            [p for p in parsed
             if p["error"] is None and lo <= p["tokens"] <= hi],
            key=lambda x: x["tokens"],
        )
        if not cands:
            warning(f"  bin [{lo}-{hi}]: no matching YAML files; "
                    "skipping.")
            continue

        if len(cands) <= n:
            picks = list(cands)
        else:
            target_idx = [0, len(cands) // 2, len(cands) - 1][:n]
            picks = []
            taken: set = set()
            for tgt in target_idx:
                # Walk outward from tgt to find an unused token count.
                placed = False
                for delta in range(len(cands)):
                    for sign in (0, 1, -1):
                        probe = tgt + sign * delta
                        if 0 <= probe < len(cands):
                            c = cands[probe]
                            if c["tokens"] not in taken:
                                picks.append(c)
                                taken.add(c["tokens"])
                                placed = True
                                break
                    if placed:
                        break
            picks = picks[:n]

        for p in picks:
            if p["yaml_file"] not in seen:
                seen.add(p["yaml_file"])
                selected.append(p)
                info(f"  bin [{lo:>6}-{hi:>6}] -> {p['yaml_file']}  "
                     f"(tokens={p['tokens']})")

    info(f"Total files selected for profiling: {len(selected)}")
    return selected


# ============================================================
# GPU sampling (fine-grained, background thread, interruptible)
# ============================================================

def _gpu_query(gpu_id: int) -> Dict:
    """One nvidia-smi snapshot for a single GPU.  Returns zeros on failure."""
    try:
        r = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.used,memory.total,utilization.gpu,"
             "temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            first = r.stdout.strip().splitlines()[0]
            parts = [x.strip() for x in first.split(",")]
            if len(parts) == 4:
                mu = int(parts[0])
                mt = int(parts[1])
                return {
                    "memory_used_mb":  mu,
                    "memory_total_mb": mt,
                    "memory_percent":  round(mu / mt * 100, 2) if mt else 0.0,
                    "gpu_util":        int(parts[2]),
                    "temperature":     int(parts[3]),
                }
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return {"memory_used_mb": 0, "memory_total_mb": 0,
            "memory_percent": 0.0, "gpu_util": 0, "temperature": 0}


class TimedGPUMonitor:
    """Thread-safe interval sampler.

    stop() returns within one poll even if the monitored child is
    mid-interval, because the worker waits on a stop Event rather than
    sleeping unconditionally.
    """

    def __init__(self, interval: float = 0.5, gpu_id: int = 0) -> None:
        if interval <= 0:
            raise ValueError("interval must be > 0")
        self.interval = float(interval)
        self.gpu_id = int(gpu_id)
        self._stop = Event()
        self._thread: Optional[Thread] = None
        self._lock = threading.Lock()
        self._t0 = 0.0
        self._records: List[Tuple[float, Dict]] = []

    def start(self) -> None:
        with self._lock:
            self._records = []
        self._stop.clear()
        self._t0 = time.time()
        self._thread = Thread(target=self._loop, daemon=True,
                              name="TimedGPUMonitor")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            elapsed = round(time.time() - self._t0, 3)
            sample = _gpu_query(self.gpu_id)
            with self._lock:
                self._records.append((elapsed, sample))
            self._stop.wait(timeout=self.interval)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval + 2.0)

    def get_records(self) -> List[Tuple[float, Dict]]:
        with self._lock:
            return list(self._records)


# ============================================================
# Boltz output detection (Boltz-1 and Boltz-2 layouts)
# ============================================================

def find_boltz_output_dir(out_dir: Path,
                          yaml_stem: str,
                          yaml_name: Optional[str],
                          ) -> Optional[Path]:
    """Locate the Boltz prediction directory for a finished job.

    Searches both layouts:
      Boltz-1: out_dir/predictions/<name>/
      Boltz-2: out_dir/boltz_results_<name>/predictions/<name>/

    The name is taken from the YAML 'name' field when provided, otherwise
    from the YAML file stem.  Returns the matched predictions/<name>/
    directory or None.
    """
    candidates = {yaml_stem.lower()}
    if yaml_name:
        candidates.add(str(yaml_name).lower())

    # Layout A: Boltz-1
    pred_a = out_dir / "predictions"
    if pred_a.is_dir():
        try:
            for child in pred_a.iterdir():
                if child.is_dir() and child.name.lower() in candidates:
                    return child
        except (PermissionError, FileNotFoundError, OSError):
            pass

    # Layout B: Boltz-2
    try:
        for entry in out_dir.iterdir():
            if not (entry.is_dir()
                    and entry.name.lower().startswith("boltz_results_")):
                continue
            pred_b = entry / "predictions"
            if not pred_b.is_dir():
                continue
            for child in pred_b.iterdir():
                if child.is_dir() and child.name.lower() in candidates:
                    return child
    except (PermissionError, FileNotFoundError, OSError):
        pass

    return None


def validate_output(out_dir: Path,
                    yaml_stem: str,
                    yaml_name: Optional[str],
                    ) -> Tuple[bool, str]:
    """Check whether the Boltz prediction produced a structure file."""
    pred_dir = find_boltz_output_dir(out_dir, yaml_stem, yaml_name)
    if pred_dir is None:
        return False, "no prediction directory found under out_dir"

    has_structure = any(pred_dir.glob("*.cif")) or any(pred_dir.glob("*.pdb"))
    if not has_structure:
        return False, f"prediction directory {pred_dir.name} has no .cif/.pdb"
    return True, f"structure files present in {pred_dir.name}"


# ============================================================
# Singularity command builder
# ============================================================

def _build_cmd(parsed: Dict,
               sif: str,
               boltz_cache: str,
               out_dir: str,
               gpu_id: int,
               extra_args: List[str],
               ) -> List[str]:
    """Build the full singularity exec --nv ... boltz predict ... command."""
    yaml_path = Path(parsed["yaml_path"])
    abs_input  = str(yaml_path.parent.resolve())
    abs_output = os.path.abspath(out_dir)
    abs_cache  = os.path.abspath(boltz_cache)
    os.makedirs(abs_output, exist_ok=True)
    os.makedirs(abs_cache, exist_ok=True)

    return [
        "singularity", "exec", "--nv",
        "--writable-tmpfs",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{abs_input}:/boltz_input",
        "--bind", f"{abs_output}:/boltz_output",
        "--bind", f"{abs_cache}:/boltz_cache",
        sif,
        "boltz", "predict",
        f"/boltz_input/{parsed['yaml_file']}",
        "--out_dir", "/boltz_output",
        "--cache", "/boltz_cache",
    ] + list(extra_args)


# ============================================================
# Boltz invocation with concurrent GPU monitoring
# ============================================================

# Track the currently-running Boltz subprocess so signal handlers can kill it.
_CHILD_LOCK = threading.Lock()
_CURRENT_CHILD: Optional[subprocess.Popen] = None


def run_boltz_job(parsed: Dict,
                  sif: str,
                  boltz_cache: str,
                  out_dir: str,
                  gpu_id: int,
                  extra_args: List[str],
                  monitor_interval: float,
                  timeout_seconds: int,
                  log_dir: Optional[Path],
                  ) -> Dict:
    """Execute one Boltz job and record GPU memory over time.

    Returns:
        {
            "success":          bool,
            "runtime_seconds":  float,
            "records":          List[(elapsed_s, gpu_dict)],
            "validate_reason":  str,
        }
    """
    global _CURRENT_CHILD
    cmd = _build_cmd(parsed, sif, boltz_cache, out_dir, gpu_id, extra_args)
    info(f"  CMD: {' '.join(shlex.quote(x) for x in cmd)}")

    log_path: Optional[Path] = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{Path(parsed['yaml_file']).stem}.log"

    monitor = TimedGPUMonitor(interval=monitor_interval, gpu_id=gpu_id)
    monitor.start()
    t0 = time.time()
    rc = -1
    log_f = open(log_path, "w", encoding="utf-8") \
        if log_path else subprocess.DEVNULL
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_f if log_path else subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        with _CHILD_LOCK:
            _CURRENT_CHILD = proc
        try:
            rc = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            warning(f"  Boltz timed out (>{timeout_seconds} s); killing.")
            proc.kill()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            rc = 124
    finally:
        with _CHILD_LOCK:
            _CURRENT_CHILD = None
        if log_path and hasattr(log_f, "close"):
            log_f.close()
        runtime = time.time() - t0
        monitor.stop()

    # Validate that Boltz actually produced output (a zero exit code alone
    # is not sufficient; the structure files must exist).
    yaml_stem = Path(parsed["yaml_file"]).stem
    files_ok, reason = validate_output(
        Path(out_dir), yaml_stem, parsed.get("yaml_name"))
    success_flag = (rc == 0) and files_ok
    if rc != 0 and files_ok:
        # Unusual: subprocess failed but outputs exist.
        reason = f"exit code {rc} but outputs present; treating as failure"

    tag = "SUCCESS" if success_flag else f"FAILED (rc={rc})"
    log_msg = f"  {tag}  runtime={runtime:.1f} s  ({reason})"
    if log_path:
        log_msg += f"  log: {log_path}"
    (ok_msg if success_flag else warning)(log_msg)

    return {
        "success":         success_flag,
        "runtime_seconds": runtime,
        "records":         monitor.get_records(),
        "validate_reason": reason,
    }


# ============================================================
# TSV output
# ============================================================

_TSV_HEADER = (
    "yaml_file", "total_tokens",
    "protein_length", "rna_length", "dna_length", "ligand_count",
    "elapsed_seconds",
    "memory_used_mb", "memory_total_mb", "memory_percent",
    "gpu_util", "temperature",
    "success", "job_runtime_seconds",
)


def append_timeseries(out_tsv: Path,
                      parsed: Dict,
                      records: List[Tuple[float, Dict]],
                      success_flag: bool,
                      runtime: float,
                      write_header: bool,
                      ) -> None:
    """Append the time-series rows for one job to the output TSV."""
    mode = "w" if write_header else "a"
    with open(out_tsv, mode, newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        if write_header:
            w.writerow(_TSV_HEADER)
        for elapsed, gpu in records:
            w.writerow([
                parsed["yaml_file"], parsed["tokens"],
                parsed["protein_len"], parsed["rna_len"],
                parsed["dna_len"], parsed["ligand_count"],
                elapsed,
                gpu["memory_used_mb"], gpu["memory_total_mb"],
                gpu["memory_percent"], gpu["gpu_util"], gpu["temperature"],
                success_flag, f"{runtime:.2f}",
            ])


# ============================================================
# Signal handling
# ============================================================

def _install_signal_handlers() -> None:
    def _handler(signum, _frame):
        sig = signal.Signals(signum).name
        warning(f"Received {sig}; killing Boltz subprocess (if any) and "
                "exiting.")
        with _CHILD_LOCK:
            child = _CURRENT_CHILD
        if child is not None and child.poll() is None:
            try:
                child.kill()
            except OSError:
                pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


# ============================================================
# CLI / Main
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="boltz_GPU_memory_timeseries.py",
        description=("Boltz-2 GPU memory time-series profiler. Buckets "
                     "prior runs from a peak-memory stat file into token "
                     "bins, picks representative YAML files per bin, runs "
                     "Boltz sequentially, and records sub-second GPU "
                     "memory curves into a single TSV."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with all required arguments
  %(prog)s --stat-file Boltz_A800_stat.tsv \\
           --input-dir ./yaml_inputs \\
           --output-dir ./timeseries_output \\
           --sif boltz.sif \\
           --boltz-cache ~/.boltz

  # Smaller bins, finer sampling, three representatives per bin
  %(prog)s --stat-file Boltz_A800_stat.tsv \\
           --input-dir ./yaml_inputs \\
           --sif boltz.sif --boltz-cache ~/.boltz \\
           --bin-width 100 --monitor-interval 0.25 --n-per-bin 3

  # Forward extra flags to boltz predict (use_msa_server, etc.)
  %(prog)s --stat-file Boltz_A800_stat.tsv \\
           --input-dir ./yaml_inputs \\
           --sif boltz.sif --boltz-cache ~/.boltz \\
           --extra-args '--use_msa_server --recycling_steps 10'
        """,
    )

    # --- Stat file and inputs ---
    p.add_argument("--stat-file", type=Path, required=True,
                   help="Peak-memory stat TSV (e.g. produced by "
                        "Boltz_GPU_parallel.py).  Must contain a "
                        "peak_memory_mb column and one of "
                        "sequence_length / token_count / tokens.")
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing Boltz YAML input files "
                        "(.yaml or .yml).  Sub-directories are not "
                        "traversed.")

    # --- Outputs ---
    p.add_argument("--output-dir", type=Path,
                   default=Path("./boltz_timeseries_output"),
                   help="Boltz prediction output directory "
                        "(default: ./boltz_timeseries_output).")
    p.add_argument("--output-tsv", type=Path, default=None,
                   help="Output TSV path "
                        "[default: <output-dir>/gpu_memory_timeseries.tsv].")
    p.add_argument("--log-dir", type=Path, default=None,
                   help="Directory for per-job stdout/stderr logs "
                        "[default: <output-dir>/_boltz_logs].")

    # --- Boltz / Singularity ---
    p.add_argument("--sif", type=str, required=True,
                   help="Path to the Boltz Singularity image (.sif).")
    p.add_argument("--boltz-cache", type=str, default="~/.boltz",
                   help="Path to the Boltz cache directory "
                        "(default: ~/.boltz).")
    p.add_argument(
        "--extra-args", type=str, default="",
        help=("Single quoted string of extra flags forwarded to "
              "'boltz predict', parsed with shlex.  Example: "
              "--extra-args '--use_msa_server --recycling_steps 10 "
              "--diffusion_samples 5'."),
    )

    # --- Profiling controls ---
    p.add_argument("--gpu-id", type=int, default=0,
                   help="GPU index to use for both the Boltz run and the "
                        "memory monitor (default: 0).")
    p.add_argument("--bin-width", type=int, default=200,
                   help="Token-bin width for representative selection "
                        "(default: 200 tokens).")
    p.add_argument("--n-per-bin", type=int, default=3,
                   help="Maximum number of representative YAML files "
                        "to profile per token bin (default: 3).")
    p.add_argument("--monitor-interval", type=float, default=0.5,
                   help="GPU sampling interval in seconds (default: 0.5).")
    p.add_argument("--workers", type=int,
                   default=min(8, multiprocessing.cpu_count()),
                   help="Parallel workers for YAML token counting "
                        "(default: min(8, cpu_count)).")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Per-job timeout in seconds (default: 7200, "
                        "i.e. 2 hours).")

    args = p.parse_args()
    if args.monitor_interval <= 0:
        p.error("--monitor-interval must be > 0.")
    if args.bin_width <= 0:
        p.error("--bin-width must be > 0.")
    if args.n_per_bin <= 0:
        p.error("--n-per-bin must be > 0.")
    if args.workers <= 0:
        p.error("--workers must be > 0.")
    if args.timeout <= 0:
        p.error("--timeout must be > 0.")
    if args.gpu_id < 0:
        p.error("--gpu-id must be >= 0.")
    return args


def main() -> None:
    args = _parse_args()
    _install_signal_handlers()

    # ---- Validation ----
    if not args.stat_file.is_file():
        error(f"Stat file not found: {args.stat_file}")
        sys.exit(1)
    if not args.input_dir.is_dir():
        error(f"Input directory not found: {args.input_dir}")
        sys.exit(1)
    if not Path(args.sif).is_file():
        error(f"Singularity image not found: {args.sif}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output_tsv is None:
        args.output_tsv = args.output_dir / "gpu_memory_timeseries.tsv"
    log_dir = args.log_dir or (args.output_dir / "_boltz_logs")
    extra_args = shlex.split(args.extra_args) if args.extra_args else []
    boltz_cache = os.path.abspath(os.path.expanduser(args.boltz_cache))

    info("=" * 64)
    info("Boltz-2 GPU Memory Time-Series Profiler")
    info("=" * 64)
    info(f"Stat file         : {args.stat_file}")
    info(f"Input dir         : {args.input_dir}")
    info(f"Output dir        : {args.output_dir}")
    info(f"Output TSV        : {args.output_tsv}")
    info(f"Log dir           : {log_dir}")
    info(f"SIF image         : {args.sif}")
    info(f"Boltz cache       : {boltz_cache}")
    info(f"GPU id            : {args.gpu_id}")
    info(f"Bin width         : {args.bin_width} tokens")
    info(f"Files per bin     : {args.n_per_bin}")
    info(f"Monitor interval  : {args.monitor_interval} s")
    info(f"Parse workers     : {args.workers}")
    info(f"Per-job timeout   : {args.timeout} s")
    if extra_args:
        info(f"Extra Boltz flags : {extra_args}")

    # ---- Tool checks ----
    try:
        r = subprocess.run(["nvidia-smi", f"--id={args.gpu_id}"],
                           capture_output=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError(f"nvidia-smi exit={r.returncode}")
        info("nvidia-smi        : OK")
    except (FileNotFoundError, subprocess.TimeoutExpired,
            RuntimeError) as exc:
        error(f"nvidia-smi unavailable for GPU {args.gpu_id}: {exc}")
        sys.exit(1)
    try:
        subprocess.run(["singularity", "--version"],
                       capture_output=True, check=True, timeout=10)
        info("singularity       : OK")
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        error("singularity not available in PATH.")
        sys.exit(1)

    # ---- Step 1: token bin detection ----
    info("=" * 64)
    info("Step 1/4 - Building token bins from stat file ...")
    bins = parse_stat_file(args.stat_file, bin_width=args.bin_width)

    # ---- Step 2: parallel YAML token counting ----
    info("=" * 64)
    info("Step 2/4 - Counting tokens in input YAML files (parallel) ...")
    all_yaml = sorted(list(args.input_dir.glob("*.yaml"))
                      + list(args.input_dir.glob("*.yml")))
    if not all_yaml:
        error(f"No YAML files found in {args.input_dir}")
        sys.exit(1)
    info(f"Found {len(all_yaml)} YAML file(s) - parsing with "
         f"{args.workers} worker(s) ...")

    parsed_all: List[Dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(_parse_worker, str(p)): p for p in all_yaml}
        done = 0
        for fut in as_completed(futs):
            parsed_all.append(fut.result())
            done += 1
            if done % 1000 == 0 or done == len(all_yaml):
                info(f"  Parsed {done}/{len(all_yaml)} ...")

    bad = [p for p in parsed_all if p["error"]]
    if bad:
        warning(f"  {len(bad)} file(s) failed to parse "
                f"(showing first 5):")
        for p in bad[:5]:
            warning(f"    {p['yaml_file']}: {p['error']}")

    parsed_all.sort(key=lambda x: x["tokens"])
    valid = [p for p in parsed_all if p["error"] is None and p["tokens"] > 0]
    if not valid:
        error("No valid YAML files parsed; aborting.")
        sys.exit(1)
    info(f"  Dataset token range: {valid[0]['tokens']} - "
         f"{valid[-1]['tokens']}")

    # ---- Step 3: select representatives ----
    info("=" * 64)
    info(f"Step 3/4 - Selecting up to {args.n_per_bin} "
         f"representative(s) per bin ...")
    selected = select_representatives(bins, parsed_all, n=args.n_per_bin)
    if not selected:
        error("No files selected. Verify that the token ranges in the "
              "input directory overlap with the bins from the stat file. "
              "Consider increasing --bin-width or supplying YAML inputs "
              "covering a wider token range.")
        sys.exit(1)

    # ---- Step 4: run Boltz + monitor ----
    info("=" * 64)
    info("Step 4/4 - Running Boltz jobs and recording GPU memory "
         "curves ...")

    first_write = not args.output_tsv.exists()
    if not first_write:
        info(f"Appending to existing TSV: {args.output_tsv}")

    total = len(selected)
    total_ok = 0
    total_fail = 0
    overall_t0 = time.time()

    for idx, parsed in enumerate(selected, 1):
        info(f"[Job {idx}/{total}] {parsed['yaml_file']} "
             f"(tokens={parsed['tokens']}, protein={parsed['protein_len']}, "
             f"rna={parsed['rna_len']}, dna={parsed['dna_len']}, "
             f"ligand={parsed['ligand_count']})")
        result = run_boltz_job(
            parsed=parsed,
            sif=args.sif,
            boltz_cache=boltz_cache,
            out_dir=str(args.output_dir),
            gpu_id=args.gpu_id,
            extra_args=extra_args,
            monitor_interval=args.monitor_interval,
            timeout_seconds=args.timeout,
            log_dir=log_dir,
        )
        append_timeseries(
            out_tsv=args.output_tsv,
            parsed=parsed,
            records=result["records"],
            success_flag=result["success"],
            runtime=result["runtime_seconds"],
            write_header=first_write,
        )
        first_write = False

        if result["success"]:
            total_ok += 1
        else:
            total_fail += 1

        n_pts = len(result["records"])
        peak = max((r[1]["memory_used_mb"] for r in result["records"]),
                   default=0)
        ok_msg(f"  {n_pts} data point(s)  |  peak VRAM = {peak} MB  |  "
               f"runtime = {result['runtime_seconds']:.1f} s  |  "
               f"{'OK' if result['success'] else 'FAILED'}")

    # ---- Done ----
    overall_runtime = time.time() - overall_t0
    info("=" * 64)
    if total_fail == 0:
        ok_msg(f"All {total} job(s) completed successfully in "
               f"{overall_runtime/60:.1f} min.")
    else:
        warning(f"{total_ok}/{total} job(s) succeeded, "
                f"{total_fail} failed (total wall time "
                f"{overall_runtime/60:.1f} min).")
    ok_msg(f"Time-series TSV: {args.output_tsv}")
    info("Filter rows by 'yaml_file' to plot each job's VRAM curve "
         "independently.")


if __name__ == "__main__":
    main()
