#!/usr/bin/env python3
"""plot_gpu_timeseries.py — GPU Memory Time-Series Plotter for Boltz-2 Profiler Output.

Reads one or more TSV files produced by Boltz2_GPU_memory_timeseries.py and
draws one curve per job (yaml_file) on a single figure, coloured by total
token count from warm (short sequences) to cool (long sequences).

Usage
-----
    python plot_gpu_timeseries.py [OPTIONS] -i FILE [FILE ...] -o OUTPUT.svg

Arguments
---------
  -i / --input     One or more TSV files to read (required).
  -o / --output    Output SVG file path (required).
  --label-col      Column to use for curve labels: 'yaml_file' or
                   'total_tokens' (default: yaml_file).
  --smooth         Apply light Gaussian smoothing to each curve
                   (sigma = N seconds; 0 = off, default: 0).
  --title          Chart title (default: "GPU Memory Usage by Job Length").
  --width          Figure width in inches (default: 12).
  --height         Figure height in inches (default: 6).
  --dpi            DPI used for layout (does not affect SVG resolution;
                   default: 100).
  --alpha          Line transparency 0–1 (default: 0.85).
  --linewidth      Line width in pts (default: 1.6).
  --legend-cols    Number of legend columns (default: 2).

Input TSV columns (produced by Boltz2_GPU_memory_timeseries.py)
---------------------------------------------------------------
  yaml_file | total_tokens | protein_length | rna_length | dna_length |
  ligand_count | elapsed_seconds | memory_used_mb | memory_total_mb |
  memory_percent | gpu_util | temperature | success | job_runtime_seconds
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (sampled from the uploaded box-plot reference image)
#
#  The palette transitions:
#    wheat/sandy  →  peach/salmon  →  muted rose  →  dusty-pink  →
#    mauve/orchid  →  slate-blue  →  steel-blue
#
#  This replicates the warm-to-cool gradient visible in the reference, where
#  shorter jobs (left boxes) are warm and longer jobs (right boxes) are cool.
# ─────────────────────────────────────────────────────────────────────────────
_PALETTE_HEX = [
    "#EDD49E",   # sandy wheat   (shortest)
    "#E8B99B",   # warm peach
    "#E0A09C",   # salmon-peach
    "#D4909A",   # muted rose
    "#C87A8A",   # dusty rose
    "#BB7A90",   # rose-mauve
    "#A8849C",   # mauve-orchid
    "#9490AA",   # lavender-grey
    "#8898BA",   # slate-blue
    "#7A8EB4",   # steel-blue     (longest)
]

def _build_cmap() -> mcolors.LinearSegmentedColormap:
    """Build a LinearSegmentedColormap from the reference palette."""
    rgb = [mcolors.to_rgb(h) for h in _PALETTE_HEX]
    return mcolors.LinearSegmentedColormap.from_list("boltz_ref", rgb)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tsv(path: Path) -> List[Dict]:
    """Return all rows of a TSV file as a list of dicts."""
    rows: List[Dict] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _load_jobs(paths: List[Path]) -> Dict[str, Dict]:
    """
    Aggregate rows from one or more TSV files into a dict keyed by yaml_file.

    Each value is::

        {
            "label":          str,          # yaml_file stem
            "total_tokens":   int,
            "elapsed":        np.ndarray,   # seconds (float)
            "memory_mb":      np.ndarray,   # memory_used_mb (float)
        }
    """
    raw: Dict[str, List[Dict]] = defaultdict(list)

    for p in paths:
        for row in _parse_tsv(p):
            key = row.get("yaml_file", "unknown")
            raw[key].append(row)

    jobs: Dict[str, Dict] = {}
    for key, rows in raw.items():
        try:
            elapsed  = np.array([float(r["elapsed_seconds"])  for r in rows])
            mem_mb   = np.array([float(r["memory_used_mb"])   for r in rows])
        except (KeyError, ValueError) as exc:
            print(f"[WARN] Skipping job '{key}': {exc}", file=sys.stderr)
            continue

        # Sort by elapsed time (safety measure)
        order     = np.argsort(elapsed)
        elapsed   = elapsed[order]
        mem_mb    = mem_mb[order]

        # Token count: use first non-empty row
        tokens = 0
        for r in rows:
            try:
                tokens = int(r.get("total_tokens", 0))
                break
            except (TypeError, ValueError):
                pass

        jobs[key] = {
            "label":        Path(key).stem,   # drop .yaml for readability
            "total_tokens": tokens,
            "elapsed":      elapsed,
            "memory_mb":    mem_mb,
        }

    return jobs


# ─────────────────────────────────────────────────────────────────────────────
# Optional smoothing
# ─────────────────────────────────────────────────────────────────────────────

def _smooth(y: np.ndarray, sigma_pts: float) -> np.ndarray:
    """Gaussian-weighted moving average (edge-preserving via reflection)."""
    if sigma_pts <= 0 or len(y) < 3:
        return y
    from scipy.ndimage import gaussian_filter1d  # type: ignore
    return gaussian_filter1d(y, sigma=sigma_pts)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_timeseries(
    jobs: Dict[str, Dict],
    output: Path,
    label_col: str    = "yaml_file",
    smooth_sigma: float = 0.0,
    title: str        = "GPU Memory Usage by Job Length",
    fig_w: float      = 12.0,
    fig_h: float      = 6.0,
    dpi: int          = 100,
    alpha: float      = 0.85,
    linewidth: float  = 1.6,
    legend_cols: int  = 2,
) -> None:
    if not jobs:
        print("[ERROR] No valid job data found.", file=sys.stderr)
        sys.exit(1)

    # ── sort jobs by token count so colour assignment is monotone ────────────
    sorted_jobs = sorted(jobs.values(), key=lambda j: j["total_tokens"])
    n           = len(sorted_jobs)
    cmap        = _build_cmap()
    colours     = [cmap(i / max(n - 1, 1)) for i in range(n)]

    # ── matplotlib global settings ───────────────────────────────────────────
    plt.rcParams.update({
        "svg.fonttype":         "none",      # keep text as <text>, not paths
        "font.family":          "sans-serif",
        "font.size":            10,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.color":           "#E0E0E0",
        "grid.linewidth":       0.6,
        "legend.frameon":       True,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#CCCCCC",
    })

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    legend_handles: List[Line2D] = []

    for idx, job in enumerate(sorted_jobs):
        elapsed  = job["elapsed"]
        mem_mb   = job["memory_mb"]
        colour   = colours[idx]
        tokens   = job["total_tokens"]

        # Smooth if requested (sigma given in seconds; convert to sample pts)
        if smooth_sigma > 0 and len(elapsed) >= 3:
            dt          = float(np.median(np.diff(elapsed))) or 0.5
            sigma_pts   = smooth_sigma / dt
            mem_mb      = _smooth(mem_mb, sigma_pts)

        # Build label
        if label_col == "total_tokens":
            label = f"{tokens} tokens"
        else:
            label = job["label"]
            if tokens:
                label += f"  [{tokens} tok]"

        line, = ax.plot(
            elapsed, mem_mb,
            color     = colour,
            linewidth = linewidth,
            alpha     = alpha,
            label     = label,
            solid_capstyle = "round",
            solid_joinstyle = "round",
        )
        legend_handles.append(line)

    # ── axes labels & title ──────────────────────────────────────────────────
    ax.set_xlabel("Elapsed Time (s)", fontsize=11)
    ax.set_ylabel("GPU Memory Used (MB)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    # ── colour bar (token-count gradient) on the right ───────────────────────
    if n > 1:
        sm = cm.ScalarMappable(
            cmap  = cmap,
            norm  = mcolors.Normalize(
                vmin = sorted_jobs[0]["total_tokens"],
                vmax = sorted_jobs[-1]["total_tokens"],
            ),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01, aspect=30, shrink=0.85)
        cbar.set_label("Total Tokens", fontsize=10)
        cbar.ax.tick_params(labelsize=9)

    # ── legend ────────────────────────────────────────────────────────────────
    legend = ax.legend(
        handles      = legend_handles,
        ncols        = legend_cols,
        fontsize     = 8,
        loc          = "upper left",
        title        = "Job  [token count]" if label_col != "total_tokens" else "Jobs",
        title_fontsize = 8.5,
    )

    plt.tight_layout()

    # ── save as SVG ───────────────────────────────────────────────────────────
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg", bbox_inches="tight",
                metadata={"Creator": "plot_gpu_timeseries.py"})
    plt.close(fig)
    print(f"[OK] SVG saved → {output}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog        = "plot_gpu_timeseries.py",
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter,
    )

    # ── required ──────────────────────────────────────────────────────────────
    p.add_argument(
        "-i", "--input",
        nargs    = "+",
        required = True,
        metavar  = "FILE",
        help     = "One or more TSV files produced by "
                   "Boltz2_GPU_memory_timeseries.py.",
    )
    p.add_argument(
        "-o", "--output",
        required = True,
        metavar  = "FILE",
        help     = "Output SVG file path (e.g. gpu_memory.svg).",
    )

    # ── optional ──────────────────────────────────────────────────────────────
    p.add_argument(
        "--label-col",
        choices  = ["yaml_file", "total_tokens"],
        default  = "yaml_file",
        help     = "Column used to label each curve in the legend "
                   "(default: yaml_file).",
    )
    p.add_argument(
        "--smooth",
        type     = float,
        default  = 0.0,
        metavar  = "SIGMA_SEC",
        help     = "Gaussian smoothing sigma in seconds; 0 = off "
                   "(default: 0).",
    )
    p.add_argument(
        "--title",
        default  = "GPU Memory Usage by Job Length",
        help     = "Chart title.",
    )
    p.add_argument(
        "--width",
        type     = float,
        default  = 12.0,
        metavar  = "INCHES",
        help     = "Figure width in inches (default: 12).",
    )
    p.add_argument(
        "--height",
        type     = float,
        default  = 6.0,
        metavar  = "INCHES",
        help     = "Figure height in inches (default: 6).",
    )
    p.add_argument(
        "--dpi",
        type     = int,
        default  = 100,
        help     = "DPI for layout computation (default: 100).",
    )
    p.add_argument(
        "--alpha",
        type     = float,
        default  = 0.85,
        help     = "Line transparency 0–1 (default: 0.85).",
    )
    p.add_argument(
        "--linewidth",
        type     = float,
        default  = 1.6,
        help     = "Line width in points (default: 1.6).",
    )
    p.add_argument(
        "--legend-cols",
        type     = int,
        default  = 2,
        metavar  = "N",
        help     = "Number of legend columns (default: 2).",
    )

    args = p.parse_args()

    # Validate
    for f in args.input:
        if not Path(f).is_file():
            p.error(f"Input file not found: {f}")
    if not args.output.lower().endswith(".svg"):
        print(f"[WARN] Output path '{args.output}' does not end with .svg; "
              "proceeding anyway.", file=sys.stderr)
    if not (0.0 < args.alpha <= 1.0):
        p.error("--alpha must be in (0, 1].")
    if args.linewidth <= 0:
        p.error("--linewidth must be > 0.")
    if args.legend_cols < 1:
        p.error("--legend-cols must be >= 1.")

    return args


def main() -> None:
    args = _parse_args()

    input_paths = [Path(f) for f in args.input]
    output_path = Path(args.output)

    print(f"[INFO] Loading {len(input_paths)} file(s) ...")
    jobs = _load_jobs(input_paths)
    print(f"[INFO] Found {len(jobs)} job(s).")

    if not jobs:
        print("[ERROR] No jobs loaded. Check that the input files contain "
              "the expected columns (elapsed_seconds, memory_used_mb, "
              "yaml_file, total_tokens).", file=sys.stderr)
        sys.exit(1)

    plot_timeseries(
        jobs         = jobs,
        output       = output_path,
        label_col    = args.label_col,
        smooth_sigma = args.smooth,
        title        = args.title,
        fig_w        = args.width,
        fig_h        = args.height,
        dpi          = args.dpi,
        alpha        = args.alpha,
        linewidth    = args.linewidth,
        legend_cols  = args.legend_cols,
    )


if __name__ == "__main__":
    main()
