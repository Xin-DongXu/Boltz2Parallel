#!/usr/bin/env python3
"""Boltz Multi-GPU Parallel Executor.

Distributes Boltz protein structure prediction tasks across multiple GPUs
using sequence-length-aware load balancing and a temporal-wave scheduler.

Memory model:
  By default the script uses a linear memory/runtime model fit from
  empirical profiling on NVIDIA A800 80GB hardware:
      memory_mb = slope_mem * tokens + intercept_mem   (with a memory floor)
      runtime_s = slope_rt  * tokens + intercept_rt    (with a runtime floor)
  Pass --legacy-step-model to use a step-wise discrete model instead.

Output layouts supported:
  Boltz-1 layout: out_dir/predictions/<name>/
  Boltz-2 layout: out_dir/boltz_results_<name>/predictions/<name>/

Token counting rules (per Boltz documentation):
  Protein  : 1 token per amino acid residue (sequence length)
  RNA/DNA  : 1 token per nucleotide (sequence length)
  Ligand   : counted as sequence_length from YAML

Container execution:
  Boltz is invoked through Singularity with --nv for GPU access.
  The container image, Boltz cache directory, input directory, and output
  directory are bind-mounted into the container.
"""


import os
import sys
import argparse
import time
import subprocess
import threading
import csv
import re
import signal
import shutil
import multiprocessing
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import bisect
import heapq
import math
import itertools

# ------------------------------------------------------------
#  Built-in Memory Profile: Boltz on A800 80GB
#  Linear model derived from 1557 data points
#  (Boltz_A800_stat_All_Len_Checked_2.tsv)
#  Fit method: max-envelope per 100-token bin
# ------------------------------------------------------------
# Linear model: memory_mb = SLOPE_MEM * tokens + INTERCEPT_MEM
# Linear model: runtime_s = SLOPE_RT  * tokens + INTERCEPT_RT
LINEAR_PROFILE_BOLTZ_A800 = {
    "slope_mem":      31.14,    # MB per token
    "intercept_mem": -1589.0,   # MB (y-intercept; floor enforced below)
    "slope_rt":        0.2162,  # seconds per token
    "intercept_rt":    2.6,     # seconds (y-intercept; floor enforced below)
    "memory_floor_mb": 3000,    # minimum memory estimate (MB)
    "runtime_floor_s": 75.0,    # minimum runtime estimate (seconds)
}

# Legacy step-wise profile (enabled by --legacy-step-model)
MEMORY_PROFILE_STEPS_BOLTZ_A800 = [
    # Sequence length: 0 - 256, Memory: ~12.8 GB
    {"min_token": 0, "max_token": 257, "memory_mb": 13111, "runtime_avg": 90.65},
    # Sequence length: 257 - 512, Memory: ~17.4 GB
    {"min_token": 257, "max_token": 513, "memory_mb": 17811, "runtime_avg": 108.73},
    # Sequence length: 513 - 768, Memory: ~17.4 GB (monotonic: max of 17653,17811)
    {"min_token": 513, "max_token": 769, "memory_mb": 17811, "runtime_avg": 130.47},
    # Sequence length: 769 - 1024, Memory: ~24.5 GB
    {"min_token": 769, "max_token": 1025, "memory_mb": 25127, "runtime_avg": 161.55},
    # Sequence length: 1025 - 1280, Memory: ~31.4 GB
    {"min_token": 1025, "max_token": 1281, "memory_mb": 32201, "runtime_avg": 200.66},
    # Sequence length: 1281 - 1536, Memory: ~43.8 GB
    {"min_token": 1281, "max_token": 1537, "memory_mb": 44837, "runtime_avg": 248.45},
    # Sequence length: 1537 - 2052, Memory: ~60.5 GB
    {"min_token": 1537, "max_token": 2053, "memory_mb": 61967, "runtime_avg": 335.10},
    # Sequence length: 2053 - 2578, Memory: ~78.3 GB
    {"min_token": 2053, "max_token": 2579, "memory_mb": 80169, "runtime_avg": 469.20},
    # Sequence length: >= 2579, Memory: ~79.0 GB (near VRAM ceiling)
    {"min_token": 2579, "max_token": None, "memory_mb": 80871, "runtime_avg": 644.61},
]

# ------------------------------------------------------------
#  GPU Preset Table
# ------------------------------------------------------------
GPU_PRESETS: Dict[str, Dict] = {
    # -- NVIDIA Data-Centre --
    "a800-80g":  {"vram_mb": 80 * 1024, "profile": "a800",    "label": "NVIDIA A800 80 GB"},
    "a100-80g":  {"vram_mb": 80 * 1024, "profile": "a800",    "label": "NVIDIA A100 80 GB"},
    "a100-40g":  {"vram_mb": 40 * 1024, "profile": "a800",    "label": "NVIDIA A100 40 GB"},
    "h100-80g":  {"vram_mb": 80 * 1024, "profile": "a800",    "label": "NVIDIA H100 80 GB"},
    "h100-94g":  {"vram_mb": 94 * 1024, "profile": "a800",    "label": "NVIDIA H100 NVL 94 GB"},
    "a6000-48g": {"vram_mb": 48 * 1024, "profile": "a800",    "label": "NVIDIA RTX A6000 48 GB"},
    "v100-32g":  {"vram_mb": 32 * 1024, "profile": "a800",    "label": "NVIDIA V100 32 GB"},
    # -- NVIDIA Consumer / Workstation --
    "rtx4090":   {"vram_mb": 24 * 1024, "profile": "a800",    "label": "NVIDIA RTX 4090 24 GB"},
    "rtx3090":   {"vram_mb": 24 * 1024, "profile": "a800",    "label": "NVIDIA RTX 3090 24 GB"},
}

DEFAULT_GPU_VRAM_MB = 80 * 1024  # 81920 MB -- A800 80GB default

# Sampling interval (seconds) for the per-task GPU memory monitor.
# A short interval gives a more accurate peak-memory reading.
GPU_MONITOR_INTERVAL_PER_TASK = 2


class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


# ------------------------------------------------------------
#  Streaming Result Writer (Thread-safe)
# ------------------------------------------------------------
class StreamingResultWriter:
    """Thread-safe writer for streaming TSV results in real-time.

    Holds a persistent open file handle instead of opening and closing
    the file on every write_task_result() call.
    """

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self._write_lock = threading.Lock()
        self.header_written = False
        self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._write_lock:
            if self._fh and not self._fh.closed:
                try:
                    self._fh.flush()
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None

    def _ensure_handle(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self.output_file, 'a', newline='', encoding='utf-8')

    def write_header(self):
        with self._write_lock:
            if self._fh and not self._fh.closed:
                self._fh.close()
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([
                    'gpu_id', 'batch_id', 'is_retry',
                    'batch_type', 'wave_id',
                    'batch_peak_memory_mb', 'batch_runtime_seconds',
                    'task_id', 'yaml_file', 'task_name',
                    'sequence_length', 'protein_length', 'rna_length', 'dna_length',
                    'ligand_count', 'total_sequences',
                    'estimated_memory_mb', 'estimated_runtime_s',
                    'task_peak_memory_mb', 'runtime_seconds',
                    'timeout_risk', 'success',
                    'timestamp'
                ])
            self._fh = open(self.output_file, 'a', newline='', encoding='utf-8')
            self.header_written = True

    def write_task_result(self, task: 'PredictionTask', ok: bool, runtime: float,
                         peak_mem: int, gpu_id: int, batch_id: str,
                         batch_peak_memory: int, batch_runtime: float,
                         is_retry: bool = False,
                         batch_type: str = 'normal',
                         wave_id: str = ''):
        with self._write_lock:
            self._ensure_handle()
            writer = csv.writer(self._fh, delimiter='\t')
            yi = task.yaml_info
            writer.writerow([
                gpu_id, batch_id, is_retry,
                batch_type, wave_id,
                batch_peak_memory, f"{batch_runtime:.2f}",
                task.task_id, task.yaml_file.name,
                yi.get('name', 'Unknown'),
                yi.get('sequence_length', 0),
                yi.get('protein_length', 0),
                yi.get('rna_length', 0),
                yi.get('dna_length', 0),
                yi.get('ligand_count', 0),
                yi.get('total_sequences', 0),
                task.estimated_memory, f"{task.estimated_runtime:.1f}",
                peak_mem, f"{runtime:.2f}",
                task.timeout_risk, ok,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            self._fh.flush()

    def write_skipped_task(self, yaml_file: Path, yi: Dict,
                           reason: str = 'skipped_existing_output'):
        with self._write_lock:
            self._ensure_handle()
            writer = csv.writer(self._fh, delimiter='\t')
            writer.writerow([
                'N/A', 'skipped', False,
                'skipped', '',
                'N/A', 'N/A',
                'N/A', yaml_file.name,
                yi.get('name', 'Unknown'),
                yi.get('sequence_length', 0),
                yi.get('protein_length', 0),
                yi.get('rna_length', 0),
                yi.get('dna_length', 0),
                yi.get('ligand_count', 0),
                yi.get('total_sequences', 0),
                'N/A', 'N/A',
                'N/A', 'N/A',
                'N/A', reason,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            self._fh.flush()


# ------------------------------------------------------------
#  Data Classes
# ------------------------------------------------------------
@dataclass
class TokenMemoryProfile:
    token_count: int
    memory_usage_mb: int
    runtime_seconds: float
    success: bool = True


@dataclass
class PredictionTask:
    yaml_file: Path
    yaml_info: Dict
    estimated_memory: int
    estimated_runtime: float
    task_id: str
    timeout_risk: bool = False
    vram_overflow: bool = False


@dataclass
class TaskBatch:
    tasks: List[PredictionTask]
    total_memory: int
    estimated_max_runtime: float
    batch_id: str


@dataclass
class TaskWave:
    tasks: List[PredictionTask]
    total_memory: int
    estimated_max_runtime: float
    wave_id: str


@dataclass
class TemporalWaveBatch:
    """Multi-anchor temporal wave batch."""
    anchor_tasks: List[PredictionTask]
    waves: List[TaskWave]
    anchor_group_memory: int
    wave_memory_budget: int
    estimated_anchor_runtime: float
    batch_id: str

    @property
    def anchor(self) -> Optional[PredictionTask]:
        return self.anchor_tasks[0] if self.anchor_tasks else None

    @property
    def anchor_memory(self) -> int:
        return self.anchor_group_memory

    @property
    def tasks(self) -> List[PredictionTask]:
        all_tasks = list(self.anchor_tasks)
        for wave in self.waves:
            all_tasks.extend(wave.tasks)
        return all_tasks

    @property
    def total_memory(self) -> int:
        wave_max = max((w.total_memory for w in self.waves), default=0)
        return self.anchor_group_memory + wave_max

    @property
    def estimated_max_runtime(self) -> float:
        return self.estimated_anchor_runtime

    @property
    def wave_task_count(self) -> int:
        return sum(len(w.tasks) for w in self.waves)


@dataclass
class GPUWorker:
    gpu_id: int
    tasks: List[PredictionTask]
    total_tokens: int
    batches: List  # List[Union[TaskBatch, TemporalWaveBatch]]
    working_dir: Path


# ------------------------------------------------------------
#  Logging helpers
# ------------------------------------------------------------
def print_colored(message: str, color: str = Colors.NC):
    print(f"{color}{message}{Colors.NC}")

def info(message: str):
    print_colored(f"[INFO] {message}", Colors.BLUE)

def success(message: str):
    print_colored(f"[SUCCESS] {message}", Colors.GREEN)

def warning(message: str):
    print_colored(f"[WARNING] {message}", Colors.YELLOW)

def error(message: str):
    print_colored(f"[ERROR] {message}", Colors.RED)

def debug(message: str):
    print_colored(f"[DEBUG] {message}", Colors.CYAN)


# ------------------------------------------------------------
#  GPU Detection & Helpers
# ------------------------------------------------------------
def detect_gpu_vram_mb(gpu_id: int) -> Optional[int]:
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}',
             '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
            if raw.isdigit():
                return int(raw)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def _guess_gpu_preset_from_vram(vram_mb: int) -> Optional[str]:
    for name, preset in GPU_PRESETS.items():
        if abs(preset['vram_mb'] - vram_mb) <= 512:
            return name
    return None


def detect_available_gpus() -> List[int]:
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return list(range(len(lines)))
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return []


def parse_gpu_list(gpu_str: str) -> List[int]:
    if not gpu_str:
        return detect_available_gpus()
    gpus = set()
    parts = gpu_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            gpus.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            gpus.add(int(part))
    return sorted(list(gpus))


# ------------------------------------------------------------
#  GPU Monitor
# ------------------------------------------------------------
class GPUMonitor:
    MAX_GPU_DATA_ENTRIES = 86_400

    def __init__(self, interval: int = 1, gpu_id: int = 0):
        self.interval = interval
        self.gpu_id = gpu_id
        self.monitoring = False
        self.monitor_thread = None
        self.current_memory = 0
        self.peak_memory = 0
        self.gpu_data = []
        self._lock = threading.Lock()

    def check_nvidia_smi(self) -> bool:
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_gpu_info(self, specific_gpu: int = None) -> Dict:
        gpu_idx = specific_gpu if specific_gpu is not None else self.gpu_id
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_idx}',
                 '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 4:
                    memory_used  = int(parts[0]) if parts[0].isdigit() else 0
                    memory_total = int(parts[1]) if parts[1].isdigit() else 0
                    gpu_util     = parts[2] if parts[2] else 'N/A'
                    temperature  = parts[3] if parts[3] else 'N/A'
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    return {
                        'memory_used': memory_used, 'memory_total': memory_total,
                        'memory_percent': memory_percent, 'gpu_util': gpu_util,
                        'temperature': temperature
                    }
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass
        return {'memory_used': 0, 'memory_total': 0, 'memory_percent': 0,
                'gpu_util': 'N/A', 'temperature': 'N/A'}

    def get_current_memory_usage(self) -> int:
        return self.get_gpu_info()['memory_used']

    def _monitor_loop(self):
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            current_memory = gpu_info['memory_used']
            with self._lock:
                if current_memory > self.peak_memory:
                    self.peak_memory = current_memory
                self.current_memory = current_memory
                if len(self.gpu_data) < self.MAX_GPU_DATA_ENTRIES:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.gpu_data.append({
                        'timestamp': timestamp,
                        'memory_used': current_memory,
                        'memory_total': gpu_info['memory_total'],
                        'memory_percent': gpu_info['memory_percent'],
                        'gpu_util': gpu_info['gpu_util'],
                        'temperature': gpu_info['temperature']
                    })
            time.sleep(self.interval)

    def start_monitoring(self):
        initial_memory = self.get_current_memory_usage()
        with self._lock:
            self.monitoring = True
            self.peak_memory = initial_memory
            self.current_memory = initial_memory
            self.gpu_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)


# ------------------------------------------------------------
#  Token-level Memory Profile Loader
# ------------------------------------------------------------
class TokenMemoryProfileLoader:
    """
    Loads per-sequence-length memory/runtime profiles.

    Default mode: LINEAR MODEL
      memory_mb = slope_mem * tokens + intercept_mem   (floor applied)
      runtime_s = slope_rt  * tokens + intercept_rt    (floor applied)

    Analysis of 1557 Boltz runs on A800 80GB shows a strong linear relationship
    between total token count and peak VRAM (R^2 ~ 0.90, max-envelope fit).
    Unlike AlphaFold3's discrete/gradient allocation pattern, Boltz's memory
    scales linearly, making a linear model more accurate than a step function.

    Legacy mode (--legacy-step-model): step-wise discrete model.
    """
    MEMORY_FLOOR_MB = 3000
    RUNTIME_FLOOR_S = 75.0

    def __init__(self, profile_file: Optional[Path] = None, vram_margin: float = 0.95,
                 builtin_profile: str = 'a800', profile_gap_fill: bool = True,
                 legacy_step_model: bool = False):
        self.vram_margin = vram_margin
        self.profiles: Dict[int, TokenMemoryProfile] = {}
        self.profile_file = profile_file
        self._builtin_profile = builtin_profile
        self._do_gap_fill: bool = profile_gap_fill
        self._legacy_step_model = legacy_step_model
        self.gpu_vram_mb: Optional[int] = None
        self.effective_vram_mb: Optional[int] = None
        self._vram_overflow_token: Optional[int] = None
        self.profile_source: str = 'builtin'
        self.profile_source_label: str = 'Built-in Boltz A800 80GB profile'

        # Linear model parameters (default mode)
        self._slope_mem: float = 0.0
        self._intercept_mem: float = 0.0
        self._slope_rt: float = 0.0
        self._intercept_rt: float = 0.0
        self._memory_floor: int = self.MEMORY_FLOOR_MB
        self._runtime_floor: float = self.RUNTIME_FLOOR_S
        self._use_linear: bool = not legacy_step_model

        # Legacy step model data (only populated when legacy mode is active)
        self._memory_steps: List[Tuple[int, Optional[int], int, float]] = []
        self._step_min_tokens: List[int] = []

        if profile_file and profile_file.exists():
            self._load_external_profiles()
        else:
            if profile_file is not None:
                warning(f"Memory profile file not found: {profile_file}")
                warning("Falling back to built-in Boltz A800 80GB profile")
            self._load_builtin_profiles()

    # -- Linear model helpers --

    @staticmethod
    def _linear_fit_max_envelope(tokens: List[int], values: List[float],
                                  bin_size: int = 100) -> Tuple[float, float]:
        """Fit a line through the max values in each bin.

        Returns (slope, intercept).  Pure-Python implementation --
        no numpy dependency required.
        """
        from collections import defaultdict
        bins: Dict[int, List[float]] = defaultdict(list)
        for t, v in zip(tokens, values):
            b = (t // bin_size) * bin_size
            bins[b].append(v)

        # Collect (bin_center, max_value) pairs
        xs, ys = [], []
        for b in sorted(bins.keys()):
            if len(bins[b]) >= 2:  # require >=2 points per bin for robustness
                xs.append(b + bin_size / 2.0)
                ys.append(max(bins[b]))
        if len(xs) < 2:
            # Fallback: use raw max values
            xs = [float(t) for t in tokens]
            ys = list(values)

        n = len(xs)
        sum_x  = sum(xs)
        sum_y  = sum(ys)
        sum_xy = sum(x * y for x, y in zip(xs, ys))
        sum_x2 = sum(x * x for x in xs)

        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-12:
            # Degenerate: all tokens identical -> flat line at max memory
            return 0.0, max(ys) if ys else 3000.0

        slope = (n * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / n
        return slope, intercept

    def _load_builtin_profiles(self):
        if self._legacy_step_model:
            self._load_builtin_step_profiles()
        else:
            self._load_builtin_linear_profiles()

    def _load_builtin_linear_profiles(self):
        """Load the built-in linear model (default)."""
        lp = LINEAR_PROFILE_BOLTZ_A800
        self._slope_mem     = lp['slope_mem']
        self._intercept_mem = lp['intercept_mem']
        self._slope_rt      = lp['slope_rt']
        self._intercept_rt  = lp['intercept_rt']
        self._memory_floor  = lp['memory_floor_mb']
        self._runtime_floor = lp['runtime_floor_s']
        self._use_linear = True
        self.profile_source = 'builtin'
        self.profile_source_label = (
            f'Built-in Boltz A800 80GB linear model '
            f'(mem = {self._slope_mem:.2f}*tokens + {self._intercept_mem:.0f}, '
            f'floor={self._memory_floor}MB)'
        )
        info(f"Loaded {self.profile_source_label}")

    def _load_builtin_step_profiles(self):
        """Load the legacy step-wise model."""
        self._use_linear = False
        steps_src = MEMORY_PROFILE_STEPS_BOLTZ_A800
        label_base = 'Built-in Boltz A800 80GB step profile (legacy)'

        for step in steps_src:
            self._memory_steps.append((
                step["min_token"], step["max_token"],
                step["memory_mb"], step["runtime_avg"]
            ))
        self.profile_source = 'builtin'
        self.profile_source_label = f'{label_base} ({len(self._memory_steps)} steps)'
        self._rebuild_step_index()
        info(f"Loaded {self.profile_source_label}")

    def _rebuild_step_index(self):
        self._step_min_tokens = [s[0] for s in self._memory_steps]

    def _load_external_profiles(self):
        """Load memory profiles from external TSV file.

        Supports both Boltz-style (sequence_length) and AF3-style (token_count) columns.
        Fits a linear model by default; uses step-wise in legacy mode.
        """
        REQUIRED_COLUMNS_SETS = [
            {'sequence_length', 'peak_memory_mb', 'runtime_seconds'},
            {'token_count', 'peak_memory_mb', 'runtime_seconds'},
        ]
        try:
            with open(self.profile_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                if reader.fieldnames is None:
                    raise RuntimeError("Profile file appears to be empty")

                # Detect which column set is available
                fields_set = set(reader.fieldnames)
                length_col = None
                for reqset in REQUIRED_COLUMNS_SETS:
                    if reqset.issubset(fields_set):
                        length_col = 'sequence_length' if 'sequence_length' in reqset else 'token_count'
                        break
                if length_col is None:
                    raise RuntimeError(
                        f"Profile file missing required columns. "
                        f"Found: {list(reader.fieldnames)}. "
                        f"Need: sequence_length/token_count + peak_memory_mb + runtime_seconds"
                    )

                success_col = None
                for fname in reader.fieldnames:
                    if 'success' in fname.lower():
                        success_col = fname
                        break

                for row in reader:
                    try:
                        if success_col:
                            sv = row.get(success_col, '').strip().lower()
                            if sv not in ('true', '1', 'yes', 'success'):
                                continue
                        token_count = int(float(row[length_col]))
                        memory_mb = int(float(row['peak_memory_mb']))
                        runtime = float(row['runtime_seconds'])
                        if token_count <= 0 or memory_mb <= 0:
                            continue
                        existing = self.profiles.get(token_count)
                        if existing is None:
                            self.profiles[token_count] = TokenMemoryProfile(
                                token_count=token_count,
                                memory_usage_mb=memory_mb,
                                runtime_seconds=runtime, success=True
                            )
                        else:
                            if memory_mb > existing.memory_usage_mb:
                                existing.memory_usage_mb = memory_mb
                            if runtime > existing.runtime_seconds:
                                existing.runtime_seconds = runtime
                    except (ValueError, KeyError) as e:
                        warning(f"Skipping invalid profile row: {type(e).__name__}: {e}")

            if not self.profiles:
                raise RuntimeError("No valid data points found in profile file")

            if self._legacy_step_model:
                self._build_steps_from_profiles()
                n_steps = len(self._memory_steps)
                self.profile_source = 'external'
                self.profile_source_label = (
                    f'External step profile: {self.profile_file.name} '
                    f'({len(self.profiles)} data points -> {n_steps} steps)'
                )
            else:
                self._build_linear_from_profiles()
                self.profile_source = 'external'
                self.profile_source_label = (
                    f'External linear profile: {self.profile_file.name} '
                    f'({len(self.profiles)} data points -> '
                    f'mem = {self._slope_mem:.2f}*tokens + {self._intercept_mem:.0f}, '
                    f'floor={self._memory_floor}MB)'
                )
            info(f"Loaded {self.profile_source_label}")

        except Exception as e:
            warning(f"Failed to load external profiles ({type(e).__name__}: {e})")
            warning("Falling back to built-in Boltz A800 80GB profile")
            self._load_builtin_profiles()

    def _build_linear_from_profiles(self):
        """Build linear model from loaded profile data using max-envelope fit."""
        tokens_list = [p.token_count for p in self.profiles.values()]
        mem_list    = [float(p.memory_usage_mb) for p in self.profiles.values()]
        rt_list     = [p.runtime_seconds for p in self.profiles.values()]

        self._slope_mem, self._intercept_mem = self._linear_fit_max_envelope(
            tokens_list, mem_list, bin_size=100)
        self._slope_rt, self._intercept_rt = self._linear_fit_max_envelope(
            tokens_list, rt_list, bin_size=100)

        # Determine floor from the smallest observed values
        self._memory_floor = max(self.MEMORY_FLOOR_MB,
                                  int(min(mem_list) * 0.95))
        self._runtime_floor = max(self.RUNTIME_FLOOR_S,
                                   min(rt_list) * 0.9)
        self._use_linear = True

        info(f"Linear fit: memory_mb = {self._slope_mem:.4f} * tokens + {self._intercept_mem:.2f} "
             f"(floor={self._memory_floor})")
        info(f"Linear fit: runtime_s = {self._slope_rt:.4f} * tokens + {self._intercept_rt:.2f} "
             f"(floor={self._runtime_floor:.1f})")

    def _build_steps_from_profiles(self):
        """Build step function from loaded profile data (legacy mode)."""
        if not self.profiles:
            return

        canonical_boundaries = [
            (0, 257), (257, 513), (513, 769), (769, 1025),
            (1025, 1281), (1281, 1537), (1537, 2053),
            (2053, 2579), (2579, None),
        ]

        sorted_tokens = sorted(self.profiles.keys())
        self._memory_steps = []

        for lo, hi in canonical_boundaries:
            matching = [
                t for t in sorted_tokens
                if t >= lo and (hi is None or t < hi)
            ]
            if matching:
                max_mem = max(self.profiles[t].memory_usage_mb for t in matching)
                runtimes = [self.profiles[t].runtime_seconds for t in matching]
                avg_rt = sum(runtimes) / len(runtimes)
                self._memory_steps.append((lo, hi, max_mem, round(avg_rt, 2)))

        if not self._memory_steps:
            max_mem = max(p.memory_usage_mb for p in self.profiles.values())
            avg_rt = sum(p.runtime_seconds for p in self.profiles.values()) / len(self.profiles)
            min_tok = min(self.profiles.keys())
            self._memory_steps.append((min_tok, None, max_mem, round(avg_rt, 2)))

        # Enforce monotonically non-decreasing memory
        for i in range(1, len(self._memory_steps)):
            prev_mem = self._memory_steps[i - 1][2]
            curr = self._memory_steps[i]
            if curr[2] < prev_mem:
                self._memory_steps[i] = (curr[0], curr[1], prev_mem, curr[3])

        self._rebuild_step_index()

    def set_gpu_vram(self, vram_mb: int):
        self.gpu_vram_mb = vram_mb
        self.effective_vram_mb = int(vram_mb * self.vram_margin)

        if self._use_linear:
            # Analytical VRAM overflow threshold
            if self._slope_mem > 0:
                overflow_tokens = (self.effective_vram_mb - self._intercept_mem) / self._slope_mem
                if overflow_tokens > 0:
                    self._vram_overflow_token = int(math.ceil(overflow_tokens))
                else:
                    # Even 0 tokens exceed VRAM (shouldn't happen)
                    self._vram_overflow_token = 0
            else:
                # Flat or negative slope -- no overflow
                self._vram_overflow_token = None
        else:
            # Legacy step-based overflow detection
            for min_token, max_token, mem, _ in self._memory_steps:
                if mem > self.effective_vram_mb:
                    self._vram_overflow_token = min_token
                    break

        if self._vram_overflow_token is not None:
            warning(f"GPU VRAM limit: {vram_mb}MB ({vram_mb/1024:.1f}GB)")
            warning(f"Effective limit (with {self.vram_margin*100:.0f}% margin): "
                    f"{self.effective_vram_mb}MB")
            warning(f"VRAM overflow at sequence_length >= {self._vram_overflow_token} "
                    f"(may cause OOM)")
        else:
            info(f"GPU VRAM limit: {vram_mb}MB ({vram_mb/1024:.1f}GB) "
                 f"- all profiled tasks fit within VRAM")

    def _lookup_step(self, token_count: int) -> int:
        """Legacy step lookup (only used in step mode)."""
        if not self._step_min_tokens:
            return -1
        idx = bisect.bisect_right(self._step_min_tokens, token_count) - 1
        return max(0, min(idx, len(self._memory_steps) - 1))

    def estimate_memory_mb(self, token_count: int) -> int:
        if self._use_linear:
            raw = self._slope_mem * token_count + self._intercept_mem
            return max(int(math.ceil(raw)), self._memory_floor)
        else:
            if not self._memory_steps:
                return self.MEMORY_FLOOR_MB
            return self._memory_steps[self._lookup_step(token_count)][2]

    def estimate_runtime_seconds(self, token_count: int) -> float:
        if self._use_linear:
            raw = self._slope_rt * token_count + self._intercept_rt
            return max(raw, self._runtime_floor)
        else:
            if not self._memory_steps:
                return 7200.0
            return self._memory_steps[self._lookup_step(token_count)][3]

    def is_timeout_risk(self, token_count: int, timeout_seconds: float = 7200.0) -> bool:
        return self.estimate_runtime_seconds(token_count) >= timeout_seconds * 0.85

    def is_over_gpu_vram(self, token_count: int) -> bool:
        if self._vram_overflow_token is None:
            return False
        return token_count >= self._vram_overflow_token

    def get_memory_step_summary(self) -> List[Tuple[int, Optional[int], int, bool]]:
        """Return summary for display.

        In linear mode, returns sampled points along the linear model.
        In legacy step mode, returns the step boundaries.
        """
        if self._use_linear:
            # Generate sample points at representative token counts
            sample_tokens = [50, 100, 250, 500, 750, 1000, 1250, 1500,
                             1750, 2000, 2250, 2500, 2750, 3000]
            result = []
            prev_upper = 0
            for i, t in enumerate(sample_tokens):
                mem = self.estimate_memory_mb(t)
                upper = sample_tokens[i + 1] if i + 1 < len(sample_tokens) else None
                exceeds = self.effective_vram_mb is not None and mem > self.effective_vram_mb
                result.append((prev_upper, upper, mem, exceeds))
                prev_upper = upper if upper is not None else t
            return result
        else:
            if not self._memory_steps:
                return []
            result = []
            for min_token, max_token, mem, _ in self._memory_steps:
                exceeds = self.effective_vram_mb is not None and mem > self.effective_vram_mb
                result.append((min_token, max_token, mem, exceeds))
            return result

    def get_vram_overflow_threshold(self) -> Optional[int]:
        return self._vram_overflow_token

    def get_linear_params(self) -> Optional[Dict]:
        """Return the linear model parameters (None if in legacy step mode)."""
        if not self._use_linear:
            return None
        return {
            'slope_mem': self._slope_mem,
            'intercept_mem': self._intercept_mem,
            'slope_rt': self._slope_rt,
            'intercept_rt': self._intercept_rt,
            'memory_floor': self._memory_floor,
            'runtime_floor': self._runtime_floor,
        }


# ------------------------------------------------------------
#  Boltz YAML Token Counter / Parser
# ------------------------------------------------------------
def _parse_yaml_core(data: Dict, yaml_path: Path) -> Optional[Dict]:
    """Shared YAML parsing core used by parse_yaml_file() and
    _parse_single_yaml_for_process().

    Eliminates ~50 lines of duplicated sequence-parsing logic.
    Returns the yaml_info dict or None if no valid sequences found.
    """
    if data is None:
        return None

    protein_length = 0
    rna_length = 0
    dna_length = 0
    ligand_count = 0
    total_sequences = 0

    sequences = data.get('sequences', [])
    for seq_entry in sequences:
        total_sequences += 1
        count = max(1, int(seq_entry.get('count', 1)))

        if 'protein' in seq_entry:
            p = seq_entry['protein']
            protein_length += len(p.get('sequence', '')) * count
        elif 'rna' in seq_entry:
            r = seq_entry['rna']
            rna_length += len(r.get('sequence', '')) * count
        elif 'dna' in seq_entry:
            d = seq_entry['dna']
            dna_length += len(d.get('sequence', '')) * count
        elif 'ligand' in seq_entry:
            ligand_count += count

    sequence_length = protein_length + rna_length + dna_length
    if sequence_length <= 0 and ligand_count <= 0:
        return None

    # For pure-ligand inputs, use a minimum token estimate
    if sequence_length <= 0:
        sequence_length = ligand_count  # minimal estimate: 1 token per ligand entity

    name = data.get('name', yaml_path.stem)
    return {
        'name': name,
        'sequence_length': sequence_length,
        'protein_length': protein_length,
        'rna_length': rna_length,
        'dna_length': dna_length,
        'ligand_count': ligand_count,
        'total_sequences': total_sequences,
        'version': data.get('version', 1),
        'yaml_input_file': yaml_path.name,
        '_yaml_path': str(yaml_path),
    }


def parse_yaml_file(yaml_path: Path) -> Optional[Dict]:
    """Parse a single Boltz YAML file in the main process (logs warnings on error)."""
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return _parse_yaml_core(data, yaml_path)
    except Exception as e:
        warning(f"Failed to parse {yaml_path}: {e}")
        return None


def _parse_single_yaml_for_process(yaml_path_str: str) -> Optional[Dict]:
    """Parse a single YAML file inside a subprocess worker.

    Delegates to _parse_yaml_core() -- no logging (callers count None returns).
    """
    try:
        yaml_path = Path(yaml_path_str)
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return _parse_yaml_core(data, yaml_path)
    except Exception:
        return None


# ------------------------------------------------------------
#  File Collection (Parallel)
# ------------------------------------------------------------
def check_output_exists(output_dir: str, yaml_info: Dict, verbose: bool = False) -> bool:
    """Check whether the output folder for this job already exists.

    Boltz output structure varies by version:
      Boltz-1: out_dir/predictions/<input_name>/
      Boltz-2: out_dir/boltz_results_<input_name>/predictions/<input_name>/
    Checks both the 'name' field from YAML and the yaml file stem, in both layouts.
    """
    output_path = Path(output_dir)
    job_name = yaml_info.get('name', 'Unknown')
    yaml_stem = Path(yaml_info.get('yaml_input_file', '')).stem

    candidates = {job_name.lower(), yaml_stem.lower()} - {'', 'unknown'}
    if not candidates:
        return False

    # Layout A: Boltz-1 style  out_dir/predictions/<name>/
    predictions_dir = output_path / 'predictions'
    if predictions_dir.exists():
        try:
            for folder in predictions_dir.iterdir():
                if folder.is_dir() and folder.name.lower() in candidates:
                    if any(folder.iterdir()):
                        if verbose:
                            debug(f"Found completed output (Boltz-1 layout) for '{job_name}' ({folder})")
                        return True
        except (PermissionError, FileNotFoundError):
            pass

    # Layout B: Boltz-2 style  out_dir/boltz_results_<name>/predictions/<name>/
    try:
        for cand_name in candidates:
            boltz_results_dir = output_path / f'boltz_results_{cand_name}'
            if not boltz_results_dir.exists():
                # Also try original-case names
                for entry in output_path.iterdir():
                    if entry.is_dir() and entry.name.lower() == f'boltz_results_{cand_name}':
                        boltz_results_dir = entry
                        break
            pred_dir = boltz_results_dir / 'predictions'
            if pred_dir.exists():
                for folder in pred_dir.iterdir():
                    if folder.is_dir() and folder.name.lower() in candidates:
                        if any(folder.iterdir()):
                            if verbose:
                                debug(f"Found completed output (Boltz-2 layout) for '{job_name}' ({folder})")
                            return True
    except (PermissionError, FileNotFoundError, OSError):
        pass

    return False


def collect_yaml_files(input_dir: Path, output_dir: str,
                       skip_existing: bool = True,
                       verbose: bool = False,
                       num_workers: int = None) -> Tuple[List[Tuple[Path, Dict]], List[Tuple[Path, Dict]]]:
    """Collect and parse YAML files with parallel processing."""
    info(f"Scanning YAML files in: {input_dir}")
    yaml_files = list(input_dir.glob("*.yaml")) + list(input_dir.glob("*.yml"))
    if not yaml_files:
        error(f"No YAML files found in {input_dir}")
        sys.exit(1)
    info(f"Found {len(yaml_files)} YAML files")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, len(yaml_files), multiprocessing.cpu_count())

    # Pre-check existing outputs
    existing_names = set()
    if skip_existing:
        info("Checking for existing outputs...")
        predictions_dir = Path(output_dir) / 'predictions'
        try:
            # Boltz-1 layout: out_dir/predictions/<name>/
            if predictions_dir.exists():
                for folder in predictions_dir.iterdir():
                    if folder.is_dir() and any(folder.iterdir()):
                        existing_names.add(folder.name.lower())
        except (PermissionError, FileNotFoundError):
            pass
        try:
            # Boltz-2 layout: out_dir/boltz_results_<name>/predictions/<name>/
            out_path = Path(output_dir)
            for entry in out_path.iterdir():
                if entry.is_dir() and entry.name.startswith('boltz_results_'):
                    pred_sub = entry / 'predictions'
                    if pred_sub.exists():
                        for folder in pred_sub.iterdir():
                            if folder.is_dir() and any(folder.iterdir()):
                                existing_names.add(folder.name.lower())
        except (PermissionError, FileNotFoundError):
            pass
        if existing_names:
            info(f"Found {len(existing_names)} existing output folders")

    # Small file counts: single-threaded
    if len(yaml_files) <= 10:
        info("Small file count - using single-threaded parsing")
        pending_files = []
        skipped_files = []
        for yaml_file in yaml_files:
            yaml_info = parse_yaml_file(yaml_file)
            if yaml_info is not None:
                if skip_existing:
                    job_name = yaml_info.get('name', 'Unknown')
                    yaml_stem = yaml_file.stem
                    candidates = {job_name.lower(), yaml_stem.lower()} - {'', 'unknown'}
                    if candidates & existing_names:
                        skipped_files.append((yaml_file, yaml_info))
                        continue
                pending_files.append((yaml_file, yaml_info))
            else:
                if verbose:
                    warning(f"Skipped invalid: {yaml_file.name}")
        info(f"Pending tasks: {len(pending_files)}")
        if skipped_files:
            info(f"Skipped tasks: {len(skipped_files)}")
        return pending_files, skipped_files

    # Parallel parsing
    info(f"Parsing YAML files with {num_workers} parallel processes...")
    yaml_paths_str = [str(yf) for yf in yaml_files]

    pending_files = []
    skipped_files = []
    parse_errors = 0
    start_time = time.time()

    import platform as _platform
    _preferred_ctx = 'fork' if _platform.system() != 'Windows' else 'spawn'
    try:
        ctx = multiprocessing.get_context(_preferred_ctx)
    except ValueError:
        ctx = multiprocessing.get_context('spawn')

    with ctx.Pool(processes=num_workers) as pool:
        completed = 0
        total = len(yaml_paths_str)
        last_progress = -1

        for yaml_info in pool.imap_unordered(_parse_single_yaml_for_process,
                                              yaml_paths_str, chunksize=10):
            completed += 1
            progress = int(completed / total * 100)
            if progress >= last_progress + 2 or completed == total:
                elapsed = time.time() - start_time
                if completed > 0:
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"\r  Parsing: {completed}/{total} ({progress:3d}%) | "
                          f"Elapsed: {elapsed:.1f}s | Speed: {rate:.1f} files/s | "
                          f"ETA: {eta:.1f}s   ", end='', flush=True)
                last_progress = progress

            if yaml_info is None:
                parse_errors += 1
                continue

            yaml_file = Path(yaml_info['_yaml_path'])

            if skip_existing:
                job_name = yaml_info.get('name', 'Unknown')
                yaml_stem = Path(yaml_info.get('yaml_input_file', '')).stem
                candidates = {job_name.lower(), yaml_stem.lower()} - {'', 'unknown'}
                if candidates & existing_names:
                    skipped_files.append((yaml_file, yaml_info))
                    continue

            pending_files.append((yaml_file, yaml_info))

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\r  Parsing: {completed}/{total} (100%) | "
          f"Time: {elapsed:.1f}s | Avg Speed: {rate:.1f} files/s          ")
    print()  # ensure newline after progress bar

    if parse_errors > 0:
        warning(f"Failed to parse {parse_errors} file(s)")
    info(f"Pending tasks: {len(pending_files)}")
    if skipped_files:
        info(f"Skipped tasks: {len(skipped_files)}")
    return pending_files, skipped_files


# ------------------------------------------------------------
#  Dual-dimension Task Optimizer + Temporal Wave Scheduler
# ------------------------------------------------------------
class DualDimensionTaskOptimizer:
    """Packs tasks into batches satisfying both memory and runtime constraints."""

    def __init__(self, max_memory_mb: int,
                 safety_margin: float = 0.1,
                 max_batch_runtime_seconds: float = 7200.0):
        self.max_memory_mb = max_memory_mb
        self.available_memory_mb = int(max_memory_mb * (1 - safety_margin))
        self.safety_margin = safety_margin
        self.max_batch_runtime = max_batch_runtime_seconds

    def _greedy_skeleton(self, sorted_tasks: List[PredictionTask],
                         batch_id_start: int) -> Tuple[List[TaskBatch], int]:
        """First-Fit Decreasing skeleton packing."""
        batches: List[TaskBatch] = []
        batch_id = batch_id_start

        current_tasks: List[PredictionTask] = []
        current_memory = 0
        current_max_runtime = 0.0

        for task in sorted_tasks:
            mem_ok = (current_memory + task.estimated_memory) <= self.available_memory_mb
            new_max_runtime = max(current_max_runtime, task.estimated_runtime)
            runtime_ok = new_max_runtime <= self.max_batch_runtime

            if mem_ok and runtime_ok and current_tasks:
                current_tasks.append(task)
                current_memory += task.estimated_memory
                current_max_runtime = new_max_runtime
            else:
                if current_tasks:
                    batches.append(TaskBatch(
                        tasks=current_tasks.copy(),
                        total_memory=current_memory,
                        estimated_max_runtime=current_max_runtime,
                        batch_id=f"batch_{batch_id:03d}"
                    ))
                    batch_id += 1
                current_tasks = [task]
                current_memory = task.estimated_memory
                current_max_runtime = task.estimated_runtime

                if task.estimated_memory > self.available_memory_mb:
                    warning(f"Task {task.task_id} ({task.estimated_memory}MB) exceeds available "
                            f"memory -- scheduling solo (oversized)")
                    batches.append(TaskBatch(
                        tasks=[task], total_memory=task.estimated_memory,
                        estimated_max_runtime=task.estimated_runtime,
                        batch_id=f"batch_{batch_id:03d}_oversized"
                    ))
                    batch_id += 1
                    current_tasks = []
                    current_memory = 0
                    current_max_runtime = 0.0

        if current_tasks:
            batches.append(TaskBatch(
                tasks=current_tasks, total_memory=current_memory,
                estimated_max_runtime=current_max_runtime,
                batch_id=f"batch_{batch_id:03d}"
            ))
            batch_id += 1
        return batches, batch_id

    def _build_temporal_wave_batches(
        self, all_tasks: List[PredictionTask],
        batch_id_start: int,
        min_anchor_ratio: float = 2.0,
        max_anchor_group_ratio: float = 1.5,
    ) -> Tuple[List[TemporalWaveBatch], List[PredictionTask], int]:
        """Build TemporalWaveBatches from a unified task pool."""
        MIN_WAVE_BUDGET_MB = 500
        sorted_tasks = sorted(all_tasks, key=lambda t: t.estimated_runtime, reverse=True)
        scheduled: Set[str] = set()
        temporal_batches: List[TemporalWaveBatch] = []
        batch_id = batch_id_start

        for seed_idx, seed in enumerate(sorted_tasks):
            if seed.task_id in scheduled:
                continue

            # Build anchor group
            anchor_group: List[PredictionTask] = [seed]
            anchor_group_memory = seed.estimated_memory
            seed_runtime = seed.estimated_runtime

            for cand in sorted_tasks[seed_idx + 1:]:
                if cand.task_id in scheduled:
                    continue
                if cand.estimated_runtime < seed_runtime / max_anchor_group_ratio:
                    break
                if anchor_group_memory + cand.estimated_memory + MIN_WAVE_BUDGET_MB > self.available_memory_mb:
                    continue
                anchor_group.append(cand)
                anchor_group_memory += cand.estimated_memory

            anchor_window = max(t.estimated_runtime for t in anchor_group)
            wave_budget = self.available_memory_mb - anchor_group_memory

            if wave_budget < MIN_WAVE_BUDGET_MB:
                continue

            # Collect wave candidates
            wave_candidates = [
                t for t in sorted_tasks
                if t.task_id not in scheduled
                and t.task_id not in {a.task_id for a in anchor_group}
                and t.estimated_memory <= wave_budget
                and t.estimated_runtime * min_anchor_ratio <= anchor_window
            ]
            wave_candidates.sort(key=lambda t: t.estimated_memory, reverse=True)

            if not wave_candidates:
                continue

            # Build waves
            waves: List[TaskWave] = []
            remaining_candidates = list(wave_candidates)
            wave_num = 0

            while remaining_candidates:
                wave_tasks = []
                wave_mem = 0
                still_remaining = []
                for t in remaining_candidates:
                    if wave_mem + t.estimated_memory <= wave_budget:
                        wave_tasks.append(t)
                        wave_mem += t.estimated_memory
                    else:
                        still_remaining.append(t)
                if not wave_tasks:
                    break
                wave_max_rt = max(t.estimated_runtime for t in wave_tasks)
                waves.append(TaskWave(
                    tasks=wave_tasks, total_memory=wave_mem,
                    estimated_max_runtime=wave_max_rt,
                    wave_id=f"wave_{wave_num:02d}"
                ))
                wave_num += 1
                remaining_candidates = still_remaining

            if not waves:
                continue

            # Commit
            for a in anchor_group:
                scheduled.add(a.task_id)
            for w in waves:
                for t in w.tasks:
                    scheduled.add(t.task_id)

            temporal_batches.append(TemporalWaveBatch(
                anchor_tasks=anchor_group,
                waves=waves,
                anchor_group_memory=anchor_group_memory,
                wave_memory_budget=wave_budget,
                estimated_anchor_runtime=anchor_window,
                batch_id=f"batch_{batch_id:03d}_tw"
            ))
            batch_id += 1

        remaining = [t for t in all_tasks if t.task_id not in scheduled]
        return temporal_batches, remaining, batch_id

    def create_optimal_batches(self, tasks: List[PredictionTask],
                                min_anchor_ratio: float = 2.0,
                                use_temporal_waves: bool = True,
                                max_anchor_group_ratio: float = 1.5) -> List:
        batches: List = []
        batch_id = 1

        risk_tasks     = [t for t in tasks if t.timeout_risk]
        overflow_tasks = [t for t in tasks if t.vram_overflow and not t.timeout_risk]
        normal_tasks   = [t for t in tasks if not t.timeout_risk and not t.vram_overflow]

        # Phase 0: Isolate risk tasks
        for task in risk_tasks:
            batches.append(TaskBatch(
                tasks=[task], total_memory=task.estimated_memory,
                estimated_max_runtime=task.estimated_runtime,
                batch_id=f"batch_{batch_id:03d}_timeout_risk"))
            batch_id += 1
            warning(f"Task {task.task_id} placed in solo batch (timeout risk)")

        for task in overflow_tasks:
            batches.append(TaskBatch(
                tasks=[task], total_memory=task.estimated_memory,
                estimated_max_runtime=task.estimated_runtime,
                batch_id=f"batch_{batch_id:03d}_vram_overflow"))
            batch_id += 1
            warning(f"Task {task.task_id} ({task.estimated_memory}MB) exceeds GPU VRAM, solo batch")

        if not normal_tasks:
            return batches

        # Phase 1: Temporal wave batches
        remaining = normal_tasks
        if use_temporal_waves and len(normal_tasks) >= 2:
            temporal_batches, remaining, batch_id = self._build_temporal_wave_batches(
                normal_tasks, batch_id, min_anchor_ratio, max_anchor_group_ratio
            )
            if temporal_batches:
                batches.extend(temporal_batches)
                info(f"TemporalWave: {len(temporal_batches)} wave batch(es), "
                     f"{len(remaining)} task(s) remain for regular packing")

        # Phase 2: FFD packing for remaining
        if remaining:
            remaining_sorted = sorted(remaining,
                                      key=lambda t: (t.estimated_memory, t.estimated_runtime),
                                      reverse=True)
            regular_batches, batch_id = self._greedy_skeleton(remaining_sorted, batch_id)
            batches.extend(regular_batches)

        return batches


# ------------------------------------------------------------
#  Multi-GPU Task Distribution
# ------------------------------------------------------------
def distribute_tasks_by_tokens(tasks: List[PredictionTask],
                                num_gpus: int) -> List[List[PredictionTask]]:
    """Distribute tasks across GPUs using LPT algorithm with min-heap."""
    if num_gpus <= 0:
        return [tasks] if tasks else []
    if len(tasks) <= num_gpus:
        result: List[List[PredictionTask]] = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            result[i % num_gpus].append(task)
        return result

    heap = [(0, i) for i in range(num_gpus)]
    heapq.heapify(heap)
    gpu_tasks: List[List[PredictionTask]] = [[] for _ in range(num_gpus)]

    for task in sorted(tasks,
                       key=lambda t: t.yaml_info.get('sequence_length', 0),
                       reverse=True):
        tok, gid = heapq.heappop(heap)
        gpu_tasks[gid].append(task)
        heapq.heappush(heap, (tok + task.yaml_info.get('sequence_length', 0), gid))

    return gpu_tasks


def create_gpu_workers(tasks_by_gpu: List[List[PredictionTask]],
                        gpu_ids: List[int],
                        optimizer: DualDimensionTaskOptimizer,
                        workspace_root: Path,
                        min_anchor_ratio: float = 2.0,
                        use_temporal_waves: bool = True,
                        max_anchor_group_ratio: float = 1.5) -> List[GPUWorker]:
    workers = []
    for gpu_id, gpu_task_list in zip(gpu_ids, tasks_by_gpu):
        if not gpu_task_list:
            continue
        total_tokens = sum(t.yaml_info.get('sequence_length', 0) for t in gpu_task_list)
        batches = optimizer.create_optimal_batches(
            gpu_task_list,
            min_anchor_ratio=min_anchor_ratio,
            use_temporal_waves=use_temporal_waves,
            max_anchor_group_ratio=max_anchor_group_ratio,
        )
        gpu_dir = workspace_root / f"gpu_{gpu_id}_work"
        worker = GPUWorker(
            gpu_id=gpu_id, tasks=gpu_task_list,
            total_tokens=total_tokens, batches=batches,
            working_dir=gpu_dir
        )
        workers.append(worker)
    return workers


# ------------------------------------------------------------
#  Boltz Output Helpers
# ------------------------------------------------------------
def filter_harmless_warnings(stderr_text: str) -> str:
    """Filter known-harmless warnings from Boltz stderr."""
    harmless_patterns = [
        "UserWarning:", "FutureWarning:", "DeprecationWarning:",
        "torch.set_default_dtype", "Setting default dtype",
        "WARNING:root:", "WARNING:torch:",
        "GPU available:", "TPU available:", "HPU available:",
        "Using 16bit", "Lightning automatically upgraded",
        "pytorch_lightning", "Trainer(", "Using device",
        "LOCAL_RANK:", "Predicting",
        "Downloading", "Fetching",
        "trifast", "triton",
    ]
    _tf_log_prefix_re = re.compile(r'^[IW]\d{4}\s')

    lines = stderr_text.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.strip():
            continue
        line_lower = line.lower()
        is_harmless = any(p.lower() in line_lower for p in harmless_patterns)
        if not is_harmless and _tf_log_prefix_re.match(line):
            is_harmless = True
        if not is_harmless:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def _find_boltz_output_dir(output_dir: str, task_name: str) -> Optional[Path]:
    """Locate the Boltz output folder for a task, searching both layout styles.

    Returns the predictions/<task_name>/ directory if found, else None.
    Supports:
      Boltz-1: out_dir/predictions/<task_name>/
      Boltz-2: out_dir/boltz_results_<task_name>/predictions/<task_name>/
    """
    base = Path(output_dir)

    # Layout A: Boltz-1  out_dir/predictions/<n>/
    path_a = base / 'predictions' / task_name
    if path_a.exists():
        return path_a

    # Layout B: Boltz-2  out_dir/boltz_results_<n>/predictions/<n>/
    path_b = base / f'boltz_results_{task_name}' / 'predictions' / task_name
    if path_b.exists():
        return path_b

    # Layout B (case-insensitive fallback)
    try:
        target_lower = f'boltz_results_{task_name}'.lower()
        for entry in base.iterdir():
            if entry.is_dir() and entry.name.lower() == target_lower:
                pred_dir = entry / 'predictions' / task_name
                if pred_dir.exists():
                    return pred_dir
    except (PermissionError, FileNotFoundError, OSError):
        pass

    return None


def is_task_successful(output_dir: str, task_name: str,
                       result: subprocess.CompletedProcess,
                       strict_errors: bool = False) -> bool:
    """Determine whether a Boltz task succeeded.

    Checks both Boltz-1 (out_dir/predictions/<n>/) and Boltz-2
    (out_dir/boltz_results_<n>/predictions/<n>/) output layouts.
    """
    # Check for output files
    output_path = _find_boltz_output_dir(output_dir, task_name)
    if output_path is not None:
        success_indicators = ["*.cif", "*.pdb", "confidence_*.json"]
        for pattern in success_indicators:
            if list(output_path.glob(pattern)):
                return True

    if strict_errors:
        return result.returncode == 0

    if result.stderr:
        filtered_stderr = filter_harmless_warnings(result.stderr)
        if not filtered_stderr.strip():
            return True
        real_error_patterns = [
            "CUDA out of memory", "OutOfMemoryError", "RuntimeError",
            "FileNotFoundError", "Permission denied",
            "Segmentation fault", "Killed", "Fatal error",
            "Traceback (most recent call last):",
            "ImportError", "ModuleNotFoundError", "MemoryError",
        ]
        for pattern in real_error_patterns:
            if pattern in filtered_stderr:
                return False

    return result.returncode == 0


# -- Non-retryable error detection --

# Error patterns that indicate a permanent failure -- retrying will not help.
_NON_RETRYABLE_PATTERNS = [
    "pre_affinity_",           # Boltz-2 affinity intermediate missing
    "ImportError",
    "ModuleNotFoundError",
    "SyntaxError",
    "ligand for affinity is too large",
]


def is_non_retryable_error(stderr_text: str) -> bool:
    """Return True if the stderr contains an error that will never succeed on retry."""
    if not stderr_text:
        return False
    for pattern in _NON_RETRYABLE_PATTERNS:
        if pattern in stderr_text:
            return True
    return False


# ------------------------------------------------------------
#  Task Execution -- Singularity
# ------------------------------------------------------------
def run_boltz_task(task: PredictionTask, singularity_image: str,
                   boltz_cache_path: str, output_path: str,
                   extra_args: List[str] = None,
                   strict_errors: bool = False,
                   task_timeout: Optional[int] = 7200,
                   gpu_id: int = 0,
                   is_retry: bool = False) -> Tuple[bool, float, int]:
    """Run a single Boltz prediction task via Singularity.

    When is_retry=True:
      - Partial output (boltz_results_<name>/) is removed first
      - ``--override`` is appended so Boltz re-creates everything
    When is_retry=False (default):
      - ``--override`` is NOT passed, letting Boltz create output cleanly
    """
    yaml_file = task.yaml_file
    yaml_filename = yaml_file.name
    input_dir = os.path.abspath(yaml_file.parent)
    output_dir = os.path.abspath(output_path)
    cache_abs = os.path.abspath(boltz_cache_path)

    task_name = task.yaml_info.get('name', task.yaml_file.stem)

    # On retry, clean stale partial output so Boltz starts fresh
    if is_retry:
        for prefix in (f'boltz_results_{task_name}', f'boltz_results_{yaml_file.stem}'):
            stale_dir = Path(output_dir) / prefix
            if stale_dir.exists():
                try:
                    shutil.rmtree(str(stale_dir))
                    info(f"[GPU{gpu_id}] Cleaned stale output: {stale_dir.name}")
                except OSError as e:
                    warning(f"[GPU{gpu_id}] Could not remove {stale_dir.name}: {e}")

    # Build singularity command
    # Boltz command: boltz predict <input_path> --out_dir <out> --cache <cache>
    command = [
        "singularity", "exec", "--nv",
        "--writable-tmpfs",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{input_dir}:/boltz_input",
        "--bind", f"{output_dir}:/boltz_output",
        "--bind", f"{cache_abs}:/boltz_cache",
        singularity_image,
        "boltz", "predict",
        f"/boltz_input/{yaml_filename}",
        "--out_dir", "/boltz_output",
        "--cache", "/boltz_cache",
    ]
    # Only add --override on retry to avoid disrupting Boltz-2's internal
    # two-phase pipeline (structure -> affinity) on the first run.
    if is_retry:
        command.append("--override")
    if extra_args:
        command.extend(extra_args)

    info(f"[GPU{gpu_id}] {'RETRY ' if is_retry else ''}Running {task.task_id}: {yaml_filename} "
         f"(seq_len={task.yaml_info.get('sequence_length','?')}, "
         f"est.mem={task.estimated_memory}MB)")
    debug(f"Command: {' '.join(command)}")
    os.makedirs(output_dir, exist_ok=True)

    gpu_monitor = GPUMonitor(interval=GPU_MONITOR_INTERVAL_PER_TASK, gpu_id=gpu_id)
    initial_memory = gpu_monitor.get_current_memory_usage()
    gpu_monitor.start_monitoring()
    start_time = time.time()

    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=task_timeout, cwd=os.getcwd())
        end_time = time.time()
        runtime = end_time - start_time
        gpu_monitor.stop_monitoring()

        # Store filtered stderr on the task for later non-retryable detection
        filtered_stderr = ''
        if result.returncode != 0:
            warning(f"[GPU{gpu_id}] Task {task.task_id} exited with code {result.returncode}")
            if result.stderr:
                filtered_stderr = filter_harmless_warnings(result.stderr)
                if filtered_stderr:
                    error(f"[GPU{gpu_id}] Task {task.task_id} errors:\n{filtered_stderr[-1500:]}")
        task._last_stderr = filtered_stderr  # type: ignore[attr-defined]

        success_status = is_task_successful(output_dir, task_name, result, strict_errors)
        status_word = "completed" if success_status else "FAILED"
        (info if success_status else warning)(
            f"[GPU{gpu_id}] {task.task_id} {status_word} in {runtime:.1f}s")

        peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
        return success_status, runtime, peak_memory

    except subprocess.TimeoutExpired:
        warning(f"[GPU{gpu_id}] Task {task.task_id} timed out after {task_timeout}s")
        task._last_stderr = 'TimeoutExpired'  # type: ignore[attr-defined]
        try:
            gpu_monitor.stop_monitoring()
            peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
        except Exception:
            peak_memory = 0
        return False, float(task_timeout), peak_memory

    except Exception as e:
        error(f"[GPU{gpu_id}] Task {task.task_id} error: {e}")
        task._last_stderr = str(e)  # type: ignore[attr-defined]
        try:
            gpu_monitor.stop_monitoring()
            peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
        except Exception:
            peak_memory = 0
        return False, 0.0, peak_memory


def run_batch_parallel(batch: TaskBatch, singularity_image: str,
                       boltz_cache_path: str, output_path: str,
                       extra_args: List[str] = None,
                       strict_errors: bool = False,
                       max_workers: int = None,
                       task_timeout: Optional[int] = 7200,
                       gpu_id: int = 0,
                       on_task_complete=None) -> List[Tuple[PredictionTask, bool, float, int]]:
    workers = len(batch.tasks) if max_workers is None else min(max_workers, len(batch.tasks))
    info(f"[GPU{gpu_id}] Batch {batch.batch_id}: {len(batch.tasks)} tasks, "
         f"{batch.total_memory}MB est. memory")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_boltz_task, task, singularity_image,
                boltz_cache_path, output_path, extra_args,
                strict_errors, task_timeout, gpu_id
            ): task for task in batch.tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                ok, runtime, peak_mem = future.result()
                results.append((task, ok, runtime, peak_mem))
                if on_task_complete:
                    on_task_complete(task, ok, runtime, peak_mem, gpu_id, batch, '')
            except Exception as e:
                error(f"[GPU{gpu_id}] Task {task.task_id} exception: {e}")
                results.append((task, False, 0.0, 0))
                if on_task_complete:
                    on_task_complete(task, False, 0.0, 0, gpu_id, batch, '')

    success_count = sum(1 for _, ok, _, _ in results if ok)
    info(f"[GPU{gpu_id}] Batch {batch.batch_id} done: "
         f"{success_count}/{len(results)} succeeded")
    return results


def run_temporal_wave_batch(
    batch: TemporalWaveBatch, singularity_image: str,
    boltz_cache_path: str, output_path: str,
    extra_args: List[str] = None,
    strict_errors: bool = False,
    max_workers: int = None,
    task_timeout: Optional[int] = 7200,
    gpu_id: int = 0,
    on_task_complete=None
) -> Tuple[List[Tuple[PredictionTask, bool, float, int]], List[PredictionTask]]:
    """Execute temporal wave batch (multi-anchor).

    When all anchors finish before every wave is dispatched, the wave
    tasks that were never launched are NOT silently dropped - they are
    returned as a separate list of "deferred wave tasks" so the caller
    can re-run them in a later parallel pass (rather than losing them).

    Returns:
        (all_results, deferred_wave_tasks)
        - all_results: list of (task, ok, runtime, peak_mem) for tasks that
          actually ran (anchors + completed waves).
        - deferred_wave_tasks: PredictionTask objects from waves that were
          skipped because all anchors finished before those waves were
          dispatched. These have NOT been run yet and must be re-scheduled
          by the caller.
    """
    anchor_tasks = batch.anchor_tasks
    waves = batch.waves
    total_wt = sum(len(w.tasks) for w in waves)

    if len(anchor_tasks) == 1:
        anc_desc = (f"anchor={anchor_tasks[0].task_id} "
                    f"(seq_len={anchor_tasks[0].yaml_info.get('sequence_length','?')}, "
                    f"est.rt={batch.estimated_anchor_runtime:.0f}s)")
    else:
        ids = '+'.join(t.task_id for t in anchor_tasks)
        anc_desc = (f"anchors=[{ids}] ({len(anchor_tasks)} tasks, "
                    f"window={batch.estimated_anchor_runtime:.0f}s)")

    info(f"[GPU{gpu_id}] TemporalWave {batch.batch_id}: {anc_desc} | "
         f"{len(waves)} waves, {total_wt} wave tasks")

    max_wave_size = max((len(w.tasks) for w in waves), default=0)
    needed = len(anchor_tasks) + max_wave_size
    workers = needed if max_workers is None else max(needed, max_workers)

    all_results: List[Tuple[PredictionTask, bool, float, int]] = []
    deferred_wave_tasks: List[PredictionTask] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        anchor_futures = {
            executor.submit(
                run_boltz_task, anc, singularity_image,
                boltz_cache_path, output_path, extra_args,
                strict_errors, task_timeout, gpu_id
            ): anc for anc in anchor_tasks
        }

        waves_completed = 0
        waves_skipped = 0

        for wave_idx, wave in enumerate(waves):
            if all(f.done() for f in anchor_futures):
                # All anchors finished before this wave was dispatched.
                # Collect every task from waves[wave_idx:] as DEFERRED so
                # the caller can re-run them later (instead of silently
                # losing them, which is the original bug).
                remaining_waves = waves[wave_idx:]
                waves_skipped = len(remaining_waves)
                for rw in remaining_waves:
                    deferred_wave_tasks.extend(rw.tasks)
                info(f"[GPU{gpu_id}] {batch.batch_id}: all anchors done early, "
                     f"deferring {waves_skipped} remaining wave(s) "
                     f"({len(deferred_wave_tasks)} task(s)) for end-of-GPU "
                     f"parallel rerun")
                break

            info(f"[GPU{gpu_id}] {batch.batch_id} {wave.wave_id}: "
                 f"launching {len(wave.tasks)} tasks ({wave.total_memory}MB)")

            wave_futures = {
                executor.submit(
                    run_boltz_task, task, singularity_image,
                    boltz_cache_path, output_path, extra_args,
                    strict_errors, task_timeout, gpu_id
                ): task for task in wave.tasks
            }

            for future in as_completed(wave_futures):
                task = wave_futures[future]
                try:
                    ok, runtime, peak_mem = future.result()
                    all_results.append((task, ok, runtime, peak_mem))
                    if on_task_complete:
                        on_task_complete(task, ok, runtime, peak_mem,
                                        gpu_id, batch, wave.wave_id)
                except Exception as e:
                    error(f"[GPU{gpu_id}] Wave task {task.task_id} error: {e}")
                    all_results.append((task, False, 0.0, 0))
                    if on_task_complete:
                        on_task_complete(task, False, 0.0, 0,
                                        gpu_id, batch, wave.wave_id)
            waves_completed += 1

        # Collect anchor results
        for anchor_future, anc in anchor_futures.items():
            try:
                ok, runtime, peak_mem = anchor_future.result()
                all_results.append((anc, ok, runtime, peak_mem))
                status = "OK" if ok else "FAILED"
                info(f"[GPU{gpu_id}] {batch.batch_id} anchor {anc.task_id}: "
                     f"{status} in {runtime:.1f}s")
                if on_task_complete:
                    on_task_complete(anc, ok, runtime, peak_mem,
                                    gpu_id, batch, 'anchor')
            except Exception as e:
                error(f"[GPU{gpu_id}] Anchor {anc.task_id} error: {e}")
                all_results.append((anc, False, 0.0, 0))
                if on_task_complete:
                    on_task_complete(anc, False, 0.0, 0, gpu_id, batch, 'anchor')

    total_ok = sum(1 for _, ok, _, _ in all_results if ok)
    info(f"[GPU{gpu_id}] TemporalWave {batch.batch_id} DONE: "
         f"{total_ok}/{len(all_results)} succeeded | "
         f"{waves_completed} waves run, {waves_skipped} skipped"
         + (f" ({len(deferred_wave_tasks)} task(s) deferred)" if deferred_wave_tasks else ""))
    return all_results, deferred_wave_tasks


# ------------------------------------------------------------
#  GPU Worker Executor
# ------------------------------------------------------------
def run_gpu_worker(worker: GPUWorker, singularity_image: str,
                    boltz_cache_path: str, output_path: str,
                    extra_args: List[str],
                    strict_errors: bool, max_workers: int,
                    task_timeout: int,
                    input_dir: str = None,
                    result_writer: StreamingResultWriter = None,
                    optimizer: 'DualDimensionTaskOptimizer' = None) -> Dict:
    """Run all batches for a single GPU worker, then a two-stage retry pass.

    Stages:
      1. Stage YAML files into the per-GPU working directory.
      2. Walk the batch list sequentially; for each TemporalWaveBatch
         collect any "deferred wave tasks" (tasks from waves whose
         dispatch was skipped because all anchors finished early). For
         every batch also collect tasks that ran but failed.
      3. Parallel rerun pass: ONLY when at least one TemporalWaveBatch
         deferred wave tasks. Combine deferred + (retryable) failed
         tasks into a single rerun pool, repack via the optimizer with
         use_temporal_waves=False (so no task can be lost again), and
         run those rerun batches in parallel on this GPU. Anything that
         still fails drops down to stage 4.
      4. Final per-task retry pass (last resort): rerun each still-failed
         task individually, one at a time, honouring the existing
         non-retryable-error filter.
      5. On the way out, restore staged YAML files to the original input
         directory.
    """
    gpu_id = worker.gpu_id
    info(f"\n{'='*60}")
    info(f"[GPU{gpu_id}] Starting with {len(worker.tasks)} tasks in {len(worker.batches)} batches")
    info(f"[GPU{gpu_id}] Total sequence length: {worker.total_tokens:,}")
    info(f"{'='*60}")

    worker_dir = worker.working_dir
    worker_dir.mkdir(parents=True, exist_ok=True)

    original_yaml_paths = {}
    moved_files = []
    for task in worker.tasks:
        dest = worker_dir / task.yaml_file.name
        if task.yaml_file != dest and not dest.exists():
            original_path = str(task.yaml_file)
            shutil.move(original_path, str(dest))
            task.yaml_file = dest
            original_yaml_paths[dest] = original_path
            moved_files.append(task)
    if moved_files:
        info(f"[GPU{gpu_id}] Moved {len(moved_files)} YAML files to working directory")

    batch_trackers = {}

    def on_task_complete(task, ok, runtime, peak_mem, gid, batch, wave_id=''):
        if batch.batch_id not in batch_trackers:
            batch_trackers[batch.batch_id] = {
                'start_time': time.time(), 'peak_memory': 0, 'tasks': []
            }
        tracker = batch_trackers[batch.batch_id]
        tracker['peak_memory'] = max(tracker['peak_memory'], peak_mem)
        tracker['tasks'].append((task, ok, runtime, peak_mem))
        batch_runtime = time.time() - tracker['start_time']

        if isinstance(batch, TemporalWaveBatch):
            btype = 'temporal_anchor' if wave_id == 'anchor' else 'temporal_wave'
        else:
            btype = 'normal'

        if result_writer:
            result_writer.write_task_result(
                task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                gpu_id=gid, batch_id=batch.batch_id,
                batch_peak_memory=tracker['peak_memory'],
                batch_runtime=batch_runtime,
                is_retry=False, batch_type=btype, wave_id=wave_id
            )

    overall_start = time.time()
    total_ok = 0
    total_fail = 0
    failed_tasks: List[Tuple[PredictionTask, float, int]] = []
    deferred_wave_tasks: List[PredictionTask] = []

    for i, batch in enumerate(worker.batches, 1):
        is_temporal = isinstance(batch, TemporalWaveBatch)
        batch_type_label = "[TEMPORAL WAVE]" if is_temporal else "[NORMAL]"
        info(f"[GPU{gpu_id}] Batch {i}/{len(worker.batches)}: "
             f"{batch.batch_id} {batch_type_label}")

        if is_temporal:
            batch_results, batch_deferred = run_temporal_wave_batch(
                batch, singularity_image, boltz_cache_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_task_complete
            )
            if batch_deferred:
                deferred_wave_tasks.extend(batch_deferred)
                info(f"[GPU{gpu_id}] Batch {batch.batch_id}: deferred "
                     f"{len(batch_deferred)} wave task(s) for parallel rerun")
        else:
            batch_results = run_batch_parallel(
                batch, singularity_image, boltz_cache_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_task_complete
            )

        for task, ok, runtime, peak_mem in batch_results:
            if ok:
                total_ok += 1
            else:
                total_fail += 1
                failed_tasks.append((task, runtime, peak_mem))

    # =========================================================
    # Stage 3: Parallel rerun pass for deferred + failed tasks
    # =========================================================
    # Only triggered when at least one TemporalWaveBatch deferred wave
    # tasks (i.e. its anchor finished before the wave was dispatched).
    # In that case the deferred wave tasks would otherwise be silently
    # lost; we combine them with retryable failed tasks, repack the
    # union into regular FFD parallel batches via the optimizer (with
    # use_temporal_waves=False so we cannot lose any task again), and
    # run those batches in parallel on this GPU. Anything that still
    # fails drops down to the per-task retry below.
    #
    # When NO wave tasks were deferred, behaviour is unchanged: failed
    # tasks (if any) go straight to the per-task retry pass below.
    rerun_recovered = 0
    rerun_batches_run = 0
    rerun_pool: List[PredictionTask] = []
    rerun_pool_size = 0
    deferred_ids: Set[str] = {t.task_id for t in deferred_wave_tasks}
    still_failed_tasks: List[Tuple[PredictionTask, float, int]] = []
    rerun_skipped_non_retryable = 0

    if deferred_wave_tasks:
        # Filter retryable failed tasks (preserve existing semantics:
        # tasks whose stderr matches a non-retryable pattern stay
        # permanently failed and never enter the rerun pool).
        retryable_failed: List[Tuple[PredictionTask, float, int]] = []
        permanently_failed: List[Tuple[PredictionTask, float, int]] = []
        for task, rt, pm in failed_tasks:
            last_stderr = getattr(task, '_last_stderr', '')
            if is_non_retryable_error(last_stderr):
                permanently_failed.append((task, rt, pm))
                rerun_skipped_non_retryable += 1
                warning(f"[GPU{gpu_id}] Skipping rerun for {task.task_id} "
                        f"(non-retryable error detected)")
            else:
                retryable_failed.append((task, rt, pm))

        # Deduplicate by task_id (a task can't be both deferred AND failed
        # in normal flow, but de-dup defensively in case logic changes).
        seen_ids: Set[str] = set()
        for t in deferred_wave_tasks:
            if t.task_id not in seen_ids:
                rerun_pool.append(t)
                seen_ids.add(t.task_id)
        for task, _rt, _pm in retryable_failed:
            if task.task_id not in seen_ids:
                rerun_pool.append(task)
                seen_ids.add(task.task_id)
        rerun_pool_size = len(rerun_pool)

        # Pre-account: deferred tasks were never counted in total_ok or
        # total_fail (they didn't run). Treat them as "currently failing"
        # for the duration of the rerun stage so the running counters
        # stay sensible; we'll decrement total_fail for each one that
        # succeeds below. Tasks coming from `retryable_failed` are
        # already counted in total_fail, so no extra bookkeeping.
        total_fail += len(deferred_wave_tasks)

        warning(f"[GPU{gpu_id}] Parallel rerun pool: "
                f"{len(deferred_wave_tasks)} deferred + "
                f"{len(retryable_failed)} retryable-failed = "
                f"{rerun_pool_size} task(s)")

    if rerun_pool:
        # Build rerun batches via the optimizer. Disable temporal waves
        # so no task can be deferred a second time.
        if optimizer is not None:
            try:
                rerun_batches = optimizer.create_optimal_batches(
                    rerun_pool, use_temporal_waves=False
                )
            except Exception as e:
                error(f"[GPU{gpu_id}] Rerun batch packing failed: {e}; "
                      f"falling back to one solo batch per task")
                rerun_batches = [
                    TaskBatch(tasks=[t], total_memory=t.estimated_memory,
                              estimated_max_runtime=t.estimated_runtime,
                              batch_id=f"rerun_solo_{t.task_id}")
                    for t in rerun_pool
                ]
        else:
            warning(f"[GPU{gpu_id}] No optimizer passed to run_gpu_worker; "
                    f"running rerun pool as solo batches")
            rerun_batches = [
                TaskBatch(tasks=[t], total_memory=t.estimated_memory,
                          estimated_max_runtime=t.estimated_runtime,
                          batch_id=f"rerun_solo_{t.task_id}")
                for t in rerun_pool
            ]

        # Tag rerun batch ids so they don't collide with original ids in TSV.
        for rb_idx, rb in enumerate(rerun_batches, 1):
            if not rb.batch_id.startswith('rerun_'):
                rb.batch_id = f"rerun_{rb_idx:03d}_{rb.batch_id}"

        info(f"[GPU{gpu_id}] Parallel rerun: {len(rerun_batches)} batch(es) "
             f"across {rerun_pool_size} task(s)")

        def on_rerun_task_complete(task, ok, runtime, peak_mem, gid, batch, wave_id=''):
            # Mirror on_task_complete bookkeeping but stamp batch_type=
            # 'rerun_parallel' and is_retry=True for tasks that previously
            # failed, False for tasks that were merely deferred (their
            # first actual run).
            if batch.batch_id not in batch_trackers:
                batch_trackers[batch.batch_id] = {
                    'start_time': time.time(), 'peak_memory': 0, 'tasks': []
                }
            tracker = batch_trackers[batch.batch_id]
            tracker['peak_memory'] = max(tracker['peak_memory'], peak_mem)
            tracker['tasks'].append((task, ok, runtime, peak_mem))
            batch_runtime = time.time() - tracker['start_time']

            is_retry_flag = task.task_id not in deferred_ids
            if result_writer:
                result_writer.write_task_result(
                    task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                    gpu_id=gid, batch_id=batch.batch_id,
                    batch_peak_memory=tracker['peak_memory'],
                    batch_runtime=batch_runtime, is_retry=is_retry_flag,
                    batch_type='rerun_parallel', wave_id=''
                )

        for rb_idx, rb in enumerate(rerun_batches, 1):
            info(f"[GPU{gpu_id}] Rerun batch {rb_idx}/{len(rerun_batches)}: "
                 f"{rb.batch_id} ({len(rb.tasks)} tasks, "
                 f"~{rb.total_memory}MB, ~{rb.estimated_max_runtime/60:.1f}min)")
            rb_results = run_batch_parallel(
                rb, singularity_image, boltz_cache_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_rerun_task_complete
            )
            rerun_batches_run += 1
            for task, ok, runtime, peak_mem in rb_results:
                if ok:
                    total_ok += 1
                    total_fail -= 1
                    rerun_recovered += 1
                else:
                    still_failed_tasks.append((task, runtime, peak_mem))

            n_rb_ok = sum(1 for _, ok, _, _ in rb_results if ok)
            n_rb_fail = len(rb_results) - n_rb_ok
            info(f"[GPU{gpu_id}] Rerun batch {rb.batch_id}: "
                 f"{n_rb_ok} ok / {n_rb_fail} failed")
            if rb_idx < len(rerun_batches):
                time.sleep(1)

        info(f"[GPU{gpu_id}] Parallel rerun complete: "
             f"{rerun_recovered} recovered / {len(still_failed_tasks)} still failing")

    # =========================================================
    # Stage 4: Final per-task retry pass (sequential, solo)
    # =========================================================
    # If we ran the parallel rerun stage, the per-task retry now operates
    # on tasks that still failed there. Otherwise it operates on the
    # original failed_tasks (preserving legacy behaviour).
    retry_ok = 0
    retry_skipped = 0
    if deferred_wave_tasks:
        # After Stage 3: the still-failed pool is what we retry one by one.
        # Non-retryable tasks were filtered out before Stage 3 already.
        final_retry_pool = still_failed_tasks
    else:
        final_retry_pool = failed_tasks

    if final_retry_pool:
        # Filter out non-retryable tasks (only meaningful when we did NOT
        # already do a Stage 3 rerun, which already filtered them).
        retryable: List[Tuple[PredictionTask, float, int]] = []
        if not deferred_wave_tasks:
            for task, rt, pm in final_retry_pool:
                last_stderr = getattr(task, '_last_stderr', '')
                if is_non_retryable_error(last_stderr):
                    retry_skipped += 1
                    warning(f"[GPU{gpu_id}] Skipping retry for {task.task_id} "
                            f"(non-retryable error detected)")
                else:
                    retryable.append((task, rt, pm))
            if retry_skipped:
                info(f"[GPU{gpu_id}] {retry_skipped} task(s) marked non-retryable "
                     f"(e.g. pre_affinity missing / import error)")
        else:
            # In the post-Stage-3 path, non-retryable filter was already
            # applied; some tasks may still match non-retryable patterns
            # if the rerun produced a fresh non-retryable stderr. Re-check
            # to be safe.
            for task, rt, pm in final_retry_pool:
                last_stderr = getattr(task, '_last_stderr', '')
                if is_non_retryable_error(last_stderr):
                    retry_skipped += 1
                    warning(f"[GPU{gpu_id}] Skipping retry for {task.task_id} "
                            f"(non-retryable error detected)")
                else:
                    retryable.append((task, rt, pm))

        if retryable:
            info(f"[GPU{gpu_id}] Retrying {len(retryable)} task(s) individually "
                 f"(sequential)...")
            for task, _, _ in retryable:
                time.sleep(1)
                info(f"[GPU{gpu_id}] Retrying {task.task_id}...")
                ok, runtime, peak_mem = run_boltz_task(
                    task, singularity_image, boltz_cache_path, output_path,
                    extra_args, strict_errors, task_timeout, gpu_id,
                    is_retry=True
                )
                if ok:
                    retry_ok += 1
                    total_ok += 1
                    total_fail -= 1
                    success(f"[GPU{gpu_id}] Retry succeeded: {task.task_id}")
                else:
                    warning(f"[GPU{gpu_id}] Retry failed: {task.task_id}")

                if result_writer:
                    result_writer.write_task_result(
                        task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                        gpu_id=gpu_id, batch_id='retry',
                        batch_peak_memory=peak_mem, batch_runtime=runtime,
                        is_retry=True, batch_type='retry', wave_id=''
                    )

    # Restore YAML files
    restored = 0
    for dest, original_path in original_yaml_paths.items():
        try:
            if dest.exists():
                orig = Path(original_path)
                if not orig.exists():
                    shutil.move(str(dest), original_path)
                    restored += 1
                else:
                    dest.unlink()
        except OSError:
            pass
    if restored:
        info(f"[GPU{gpu_id}] Restored {restored} YAML files to input directory")

    total_time = time.time() - overall_start
    return {
        'gpu_id': gpu_id,
        'total_ok': total_ok,
        'total_fail': total_fail,
        'total_time': total_time,
        'retry_ok': retry_ok,
        'rerun_recovered': rerun_recovered,
        'rerun_pool_size': rerun_pool_size,
        'rerun_batches_run': rerun_batches_run,
        'deferred_wave_tasks': len(deferred_wave_tasks),
        'worker_dir': str(worker_dir),
    }


# ------------------------------------------------------------
#  Display helpers
# ------------------------------------------------------------
def print_gpu_distribution_summary(workers: List[GPUWorker], total_tasks: int):
    print_colored("\n" + "="*80, Colors.CYAN)
    print_colored("GPU TASK DISTRIBUTION SUMMARY", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)

    total_tokens = sum(w.total_tokens for w in workers)
    avg_tokens = total_tokens / len(workers) if workers else 0

    info(f"Total GPUs used: {len(workers)}")
    info(f"Total tasks: {total_tasks}")
    info(f"Total sequence length: {total_tokens:,}")
    info(f"Average per GPU: {avg_tokens:,.0f}")

    print_colored("\nPer-GPU Distribution:", Colors.CYAN)
    for worker in workers:
        deviation = ((worker.total_tokens - avg_tokens) / avg_tokens * 100) if avg_tokens > 0 else 0
        print(f"  GPU {worker.gpu_id}: {len(worker.tasks):3d} tasks, "
              f"{worker.total_tokens:>8,} seq_len ({deviation:+6.2f}% from avg), "
              f"{len(worker.batches):2d} batches")
    print_colored("="*80 + "\n", Colors.CYAN)


def print_optimization_summary(workers: List[GPUWorker], total_tasks: int,
                               max_memory_mb: int, skipped_count: int = 0,
                               vram_overflow_count: int = 0,
                               vram_overflow_token: Optional[int] = None):
    print_colored("\n" + "="*80, Colors.MAGENTA)
    print_colored("BATCH OPTIMIZATION SUMMARY (Multi-GPU) [Temporal Wave]", Colors.MAGENTA)
    print_colored("="*80, Colors.MAGENTA)
    info(f"Total tasks to process: {total_tasks}")
    if skipped_count > 0:
        info(f"Skipped tasks: {skipped_count}")
    if vram_overflow_count > 0:
        warning(f"VRAM overflow tasks: {vram_overflow_count} "
                f"(seq_len >= {vram_overflow_token})")

    total_batches = sum(len(w.batches) for w in workers)
    total_tw = sum(1 for w in workers
                   for b in w.batches if isinstance(b, TemporalWaveBatch))
    info(f"Total batches: {total_batches} ({total_tw} temporal-wave, "
         f"{total_batches - total_tw} normal)")
    info(f"Available GPU memory: {max_memory_mb}MB per GPU")

    for worker in workers:
        if not worker.batches:
            continue
        print_colored(f"\nGPU {worker.gpu_id} Batches:", Colors.BLUE)
        for batch in worker.batches:
            if isinstance(batch, TemporalWaveBatch):
                anchor_tasks = batch.anchor_tasks
                wave_task_count = batch.wave_task_count
                if len(anchor_tasks) == 1:
                    anc = anchor_tasks[0]
                    anc_desc = (f"anchor={anc.task_id} "
                                f"(seq_len={anc.yaml_info['sequence_length']:,})")
                else:
                    ids = '+'.join(t.task_id for t in anchor_tasks)
                    anc_desc = f"anchors=[{ids}] ({len(anchor_tasks)} tasks)"
                print_colored(
                    f"  {batch.batch_id}: [TEMPORAL WAVE] {anc_desc} | "
                    f"{len(batch.waves)} waves, {wave_task_count} wave tasks",
                    Colors.GREEN
                )
            else:
                tokens = [t.yaml_info['sequence_length'] for t in batch.tasks]
                print(f"  {batch.batch_id}: {len(batch.tasks):2d} tasks | "
                      f"mem={batch.total_memory:>5}MB | "
                      f"wall~{batch.estimated_max_runtime/60:>5.1f}min | "
                      f"seq_len {min(tokens):>5,}-{max(tokens):>5,}")
    print_colored("="*80 + "\n", Colors.MAGENTA)


def print_memory_step_summary(profile_loader: TokenMemoryProfileLoader):
    linear_params = profile_loader.get_linear_params()

    print_colored("\n" + "="*80, Colors.CYAN)
    if linear_params:
        print_colored(
            f"LINEAR MEMORY MODEL ({profile_loader.profile_source_label})",
            Colors.CYAN
        )
        print_colored("="*80, Colors.CYAN)
        print(f"  Formula:  memory_mb = {linear_params['slope_mem']:.2f} * tokens "
              f"+ ({linear_params['intercept_mem']:.0f})")
        print(f"  Formula:  runtime_s = {linear_params['slope_rt']:.4f} * tokens "
              f"+ ({linear_params['intercept_rt']:.1f})")
        print(f"  Memory floor: {linear_params['memory_floor']} MB "
              f"({linear_params['memory_floor']/1024:.1f} GB)")
        print(f"  Runtime floor: {linear_params['runtime_floor']:.1f} s")
        print("-"*80)
        print(f"{'Tokens':<10} {'Est. Memory (MB)':<18} {'Est. Memory (GB)':<18} "
              f"{'Est. Runtime':<14} {'Status'}")
        print("-"*80)

        sample_tokens = [50, 100, 250, 500, 750, 1000, 1250, 1500,
                         1750, 2000, 2250, 2500, 2750, 3000]
        for t in sample_tokens:
            mem = profile_loader.estimate_memory_mb(t)
            rt = profile_loader.estimate_runtime_seconds(t)
            exceeds = (profile_loader.effective_vram_mb is not None
                       and mem > profile_loader.effective_vram_mb)
            status = "[!] VRAM OVERFLOW" if exceeds else "[OK] GPU"
            print(f"{t:<10} {mem:<18,} {mem/1024:<18.1f} {rt:<14.1f} {status}")
    else:
        # Legacy step display
        print_colored(
            f"MEMORY ALLOCATION STEP SUMMARY ({profile_loader.profile_source_label})",
            Colors.CYAN
        )
        print_colored("="*80, Colors.CYAN)
        print(f"{'Step':<5} {'Seq Length Range':<22} {'Memory (MB)':<15} "
              f"{'Memory (GB)':<12} {'Status'}")
        print("-"*80)

        step_summary = profile_loader.get_memory_step_summary()
        for i, (min_token, max_token, mem, exceeds) in enumerate(step_summary, 1):
            if max_token is None:
                token_range = f">= {min_token}"
            else:
                token_range = f"{min_token} - {max_token - 1}"
            status = "[!] VRAM OVERFLOW" if exceeds else "[OK] GPU"
            memory_gb = mem / 1024
            print(f"{i:<5} {token_range:<22} {mem:<15,} {memory_gb:<12.1f} {status}")

    vram_threshold = profile_loader.get_vram_overflow_threshold()
    if vram_threshold:
        print("-"*80)
        warning(f"VRAM Overflow Threshold: seq_len >= {vram_threshold}")
    print_colored("="*80 + "\n", Colors.CYAN)


# ------------------------------------------------------------
#  Singularity Test
# ------------------------------------------------------------
def test_singularity_command(singularity_image: str, boltz_cache_path: str,
                             input_dir: str, output_dir: str,
                             gpu_id: int = 0) -> bool:
    """Test that the Singularity container works and 'boltz' is accessible."""
    info(f"Testing singularity command on GPU {gpu_id}...")

    # Test 1: Python is accessible
    test_cmd = [
        "singularity", "exec", "--nv",
        "--writable-tmpfs",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{input_dir}:/boltz_input",
        "--bind", f"{output_dir}:/boltz_output",
        "--bind", f"{boltz_cache_path}:/boltz_cache",
        singularity_image,
        "python", "--version"
    ]
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            error(f"Singularity Python test failed (exit {result.returncode}): {result.stderr}")
            return False
        success(f"Singularity Python: {result.stdout.strip()}")
    except Exception as e:
        error(f"Singularity Python test failed: {e}")
        return False

    # Test 2: 'boltz' command is accessible
    test_cmd2 = [
        "singularity", "exec", "--nv",
        "--writable-tmpfs",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{input_dir}:/boltz_input",
        "--bind", f"{output_dir}:/boltz_output",
        "--bind", f"{boltz_cache_path}:/boltz_cache",
        singularity_image,
        "boltz", "predict", "--help"
    ]
    try:
        result2 = subprocess.run(test_cmd2, capture_output=True, text=True, timeout=30)
        if result2.returncode != 0:
            error(f"'boltz predict --help' failed (exit {result2.returncode}): "
                  f"{result2.stderr[:300]}")
            error("The 'boltz' command may not be installed or in PATH inside the container.")
            return False
        success(f"Singularity 'boltz predict' is accessible on GPU {gpu_id}")
    except Exception as e:
        error(f"'boltz' command test failed: {e}")
        return False

    return True


# ------------------------------------------------------------
#  Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Boltz Multi-GPU Parallel Executor\n"
            "Distributes tasks evenly across multiple GPUs based on sequence length.\n"
            "Uses Singularity container for Boltz execution.\n"
            "Features: Multi-GPU, Temporal Wave Scheduling, Linear Memory Model,\n"
            "  VRAM Overflow Detection, Streaming TSV Results, Auto-Retry.\n"
            "Memory model: linear (mem = slope*tokens + intercept) by default.\n"
            "Use --legacy-step-model for the step-wise discrete model."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect GPUs, enable temporal wave scheduling (default)
  %(prog)s -i ./yaml_input -o results.tsv --sif boltz.sif --boltz-cache ~/.boltz

  # Specify 4 GPUs
  %(prog)s -i ./yaml_input -o results.tsv --sif boltz.sif --boltz-cache ~/.boltz --gpus 0,1,2,3

  # Use external memory profile
  %(prog)s -i ./yaml_input -o results.tsv --sif boltz.sif --boltz-cache ~/.boltz \\
      --memory-profile Boltz_A800_stat.tsv

  # Disable temporal wave scheduling
  %(prog)s -i ./yaml_input -o results.tsv --sif boltz.sif --boltz-cache ~/.boltz --no-temporal-waves
        """
    )

    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('-i', '--input-dir', type=Path, required=True, metavar='DIR',
                          help='Directory containing Boltz YAML input files')
    io_group.add_argument('-o', '--output-file', type=Path, required=True, metavar='FILE',
                          help='Output TSV file for results and performance metrics')
    io_group.add_argument('--output-dir', type=str, default='./boltz_output_parallel',
                          metavar='DIR',
                          help='Directory for Boltz prediction outputs (default: ./boltz_output_parallel)')
    io_group.add_argument('--no-skip-existing', action='store_true',
                          help='Force rerun all tasks (ignore existing outputs)')
    io_group.add_argument('--skip-vram-overflow', action='store_true',
                          help='Skip tasks exceeding GPU VRAM instead of running them')
    io_group.add_argument('--max-seq-length', type=int, default=None, metavar='N',
                          help='Skip tasks with sequence_length > N')
    io_group.add_argument('--temp-dir', type=str, default=None, metavar='DIR',
                          help='Temp working directory for GPU-specific files (default: ./gpu_work)')

    boltz_group = parser.add_argument_group('Boltz Configuration')
    boltz_group.add_argument('--sif', '--singularity-image', type=str, required=True,
                             metavar='FILE',
                             help='Path to Boltz Singularity image (.sif)')
    boltz_group.add_argument('--boltz-cache', type=str, default='~/.boltz', metavar='DIR',
                             help='Path to Boltz cache directory (default: ~/.boltz)')
    boltz_group.add_argument('--boltz-extra-args', type=str, nargs='*', metavar='ARG',
                             help='Additional arguments passed to boltz predict '
                                  '(e.g. --recycling_steps 10 --diffusion_samples 5)')

    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--gpus', type=str, default=None, metavar='LIST',
                          help='Comma-separated GPU IDs or ranges (e.g. "0,1,2,3" or "0-3")')
    gpu_group.add_argument('--gpu-preset', type=str, default=None, metavar='PRESET',
                           dest='gpu_preset',
                           help='GPU hardware preset (a800-80g, a100-80g, h100-80g, rtx4090, etc.)')
    gpu_group.add_argument('--gpu-memory', type=int, default=DEFAULT_GPU_VRAM_MB, metavar='MB',
                           help=f'Total GPU memory in MB per GPU (default: {DEFAULT_GPU_VRAM_MB})')
    gpu_group.add_argument('--vram-margin', type=float, default=0.95, metavar='RATIO',
                           help='VRAM safety margin (0.0-1.0, default: 0.95 = 95%%)')
    gpu_group.add_argument('--safety-margin', type=float, default=0.1, metavar='RATIO',
                           help='Safety margin for batch memory (0.0-0.5, default: 0.1)')
    gpu_group.add_argument('--max-workers', type=int, default=None, metavar='N',
                           help='Max parallel workers per batch (default: batch size)')
    gpu_group.add_argument('--max-batch-runtime', type=float, default=7200.0, metavar='SECONDS',
                           help='Max estimated wall-clock per batch (default: 7200)')
    gpu_group.add_argument('--task-timeout', type=int, default=7200, metavar='SECONDS',
                           help='Hard timeout per individual task (default: 7200)')

    sched_group = parser.add_argument_group('Temporal Wave Scheduling')
    sched_group.add_argument('--no-temporal-waves', action='store_true',
                             help='Disable temporal wave scheduling')
    sched_group.add_argument('--min-anchor-ratio', type=float, default=2.0, metavar='RATIO',
                             help='Min runtime ratio for anchor tasks (default: 2.0)')
    sched_group.add_argument('--max-anchor-group-ratio', type=float, default=1.5,
                             metavar='RATIO', dest='max_anchor_group_ratio',
                             help='Max runtime spread for anchor grouping (default: 1.5)')

    profile_group = parser.add_argument_group('Memory Profiling Options')
    profile_group.add_argument('--memory-profile', type=Path, metavar='FILE',
                               help='External TSV memory profile file')
    profile_group.add_argument('--legacy-step-model', action='store_true',
                               dest='legacy_step_model',
                               help='Use step-wise discrete memory model instead of linear model')
    profile_group.add_argument('--no-profile-gap-fill', action='store_true',
                               dest='no_profile_gap_fill',
                               help='Disable gap interpolation in external profiles (legacy mode)')
    profile_group.add_argument('--memory-estimation-factor', type=float, default=1.0,
                               metavar='FACTOR',
                               help='Multiplicative factor for memory estimates (default: 1.0)')

    monitor_group = parser.add_argument_group('Monitoring & Debug Options')
    monitor_group.add_argument('--monitor-interval', type=int, default=5, metavar='SECONDS',
                               help='GPU monitoring interval (default: 5)')
    monitor_group.add_argument('--cpu-workers', type=int, default=None, metavar='N',
                               help='CPU workers for parallel YAML parsing (default: auto)')
    monitor_group.add_argument('--verbose', '-v', action='store_true',
                               help='Enable verbose output')
    monitor_group.add_argument('--strict-errors', action='store_true',
                               help='Strict error mode: only exit code 0 counts as success')
    monitor_group.add_argument('--test-only', action='store_true',
                               help='Show configuration and batch plan only -- do not run tasks')

    args = parser.parse_args()

    # -- GPU preset resolution --
    preset_profile_key = 'a800'
    preset_label = ''

    if args.gpu_preset is not None:
        key = args.gpu_preset.lower().strip()
        if key not in GPU_PRESETS:
            preset_names = ', '.join(sorted(GPU_PRESETS.keys()))
            error(f"Unknown --gpu-preset '{key}'. Available: {preset_names}")
            sys.exit(1)
        preset = GPU_PRESETS[key]
        preset_profile_key = preset['profile']
        preset_label = preset['label']
        if args.gpu_memory == DEFAULT_GPU_VRAM_MB:
            args.gpu_memory = preset['vram_mb']
        info(f"GPU preset '{key}' ({preset_label}): vram={args.gpu_memory}MB")

    # -- Validation --
    if not (0.0 <= args.safety_margin <= 0.5):
        error("Safety margin must be between 0.0 and 0.5"); sys.exit(1)
    if not (0.5 <= args.vram_margin <= 1.0):
        error("VRAM margin must be between 0.5 and 1.0"); sys.exit(1)
    if args.gpu_memory < 1000:
        error("GPU memory must be at least 1000MB"); sys.exit(1)
    if not (0.5 <= args.memory_estimation_factor <= 5.0):
        error("Memory estimation factor must be between 0.5 and 5.0"); sys.exit(1)
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        error(f"Input directory invalid: {args.input_dir}"); sys.exit(1)
    if not os.path.exists(args.sif):
        error(f"Singularity image not found: {args.sif}"); sys.exit(1)
    if args.min_anchor_ratio < 1.1:
        error("--min-anchor-ratio must be >= 1.1"); sys.exit(1)
    if args.max_anchor_group_ratio < 1.0:
        error("--max-anchor-group-ratio must be >= 1.0"); sys.exit(1)
    if args.max_seq_length is not None and args.max_seq_length < 1:
        error("--max-seq-length must be a positive integer"); sys.exit(1)
    if args.task_timeout is not None and args.task_timeout < 60:
        error("--task-timeout must be at least 60 seconds"); sys.exit(1)

    # -- GPU Detection & Selection --
    available_gpus = detect_available_gpus()
    if not available_gpus:
        error("No GPUs detected! Please check nvidia-smi."); sys.exit(1)

    selected_gpus = parse_gpu_list(args.gpus)
    invalid_gpus = [g for g in selected_gpus if g not in available_gpus]
    if invalid_gpus:
        warning(f"GPUs {invalid_gpus} not found. Available: {available_gpus}")
        selected_gpus = [g for g in selected_gpus if g in available_gpus]
    if not selected_gpus:
        error("No valid GPUs selected!"); sys.exit(1)

    info(f"Available GPUs: {available_gpus}")
    info(f"Selected GPUs: {selected_gpus}")

    # Auto-detect GPU VRAM
    if args.gpu_preset is None and args.gpu_memory == DEFAULT_GPU_VRAM_MB:
        detected_vram = detect_gpu_vram_mb(selected_gpus[0])
        if detected_vram and detected_vram != DEFAULT_GPU_VRAM_MB:
            matched = _guess_gpu_preset_from_vram(detected_vram)
            if matched:
                preset_profile_key = GPU_PRESETS[matched]['profile']
                preset_label = GPU_PRESETS[matched]['label']
                info(f"Auto-detected GPU VRAM: {detected_vram} MB -> preset '{matched}'")
            else:
                warning(f"Auto-detected GPU VRAM: {detected_vram} MB -- no matching preset. "
                        "Using default profile.")
            args.gpu_memory = detected_vram

    # -- Paths --
    boltz_cache_path = os.path.abspath(os.path.expanduser(args.boltz_cache))
    output_path = os.path.abspath(args.output_dir)
    singularity_image = os.path.abspath(args.sif)
    input_dir = os.path.abspath(args.input_dir)

    if args.temp_dir:
        workspace_root = Path(args.temp_dir).resolve()
    else:
        workspace_root = Path(input_dir) / 'gpu_work'
    workspace_root.mkdir(parents=True, exist_ok=True)

    os.makedirs(boltz_cache_path, exist_ok=True)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    # -- Prereq checks --
    gpu_monitor = GPUMonitor(args.monitor_interval)
    if not gpu_monitor.check_nvidia_smi():
        error("nvidia-smi not available"); sys.exit(1)
    try:
        subprocess.run(['singularity', '--version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        error("Singularity not available in PATH"); sys.exit(1)

    use_temporal_waves = not args.no_temporal_waves
    min_anchor_ratio = args.min_anchor_ratio

    # -- Load profile --
    profile_loader = TokenMemoryProfileLoader(
        args.memory_profile, vram_margin=args.vram_margin,
        builtin_profile=preset_profile_key,
        profile_gap_fill=not getattr(args, 'no_profile_gap_fill', False),
        legacy_step_model=getattr(args, 'legacy_step_model', False),
    )
    profile_loader.set_gpu_vram(args.gpu_memory)

    # -- Banner --
    model_type = "Legacy Step-wise" if getattr(args, 'legacy_step_model', False) else "Linear"
    linear_params = profile_loader.get_linear_params()
    print_colored("\n" + "="*80, Colors.MAGENTA)
    print_colored("BOLTZ MULTI-GPU PARALLEL EXECUTOR", Colors.MAGENTA)
    print_colored(f"Sequence-Length-Aware Distribution across {len(selected_gpus)} GPUs",
                  Colors.MAGENTA)
    if linear_params:
        print_colored(
            f"{model_type} Memory Model: "
            f"mem = {linear_params['slope_mem']:.2f}*tokens + ({linear_params['intercept_mem']:.0f}), "
            f"floor={linear_params['memory_floor']}MB",
            Colors.MAGENTA)
    else:
        print_colored(f"{model_type} Memory Model ({len(profile_loader._memory_steps)} Discrete Steps)",
                      Colors.MAGENTA)
    print_colored(f"Profile: {profile_loader.profile_source_label}", Colors.MAGENTA)
    print_colored("Singularity Container Execution", Colors.MAGENTA)
    print_colored("Multi-Anchor TemporalWaveBatch Scheduling", Colors.MAGENTA)
    print_colored("Streaming Write - Results Visible in Real-time", Colors.MAGENTA)
    if use_temporal_waves:
        print_colored(f"Temporal Wave Scheduling ENABLED (min_anchor={min_anchor_ratio:.1f}, "
                      f"max_group={args.max_anchor_group_ratio:.1f})", Colors.GREEN)
    else:
        print_colored("Temporal Wave Scheduling DISABLED", Colors.YELLOW)
    if preset_label:
        print_colored(f"GPU Preset: {preset_label}", Colors.CYAN)
    print_colored("="*80, Colors.MAGENTA)

    info(f"Input directory   : {input_dir}")
    info(f"Output file       : {args.output_file}")
    info(f"Output directory  : {output_path}")
    info(f"Workspace dir     : {workspace_root}")
    info(f"Skip existing     : {'DISABLED' if args.no_skip_existing else 'ENABLED'}")
    if args.skip_vram_overflow:
        info(f"Skip VRAM overflow: ENABLED")
    if args.max_seq_length is not None:
        info(f"Max seq length    : {args.max_seq_length:,}")
    info(f"Singularity image : {singularity_image}")
    info(f"Boltz cache       : {boltz_cache_path}")
    info(f"GPUs used         : {selected_gpus}")
    info(f"GPU memory        : {args.gpu_memory}MB ({args.gpu_memory/1024:.1f}GB) per GPU")
    info(f"VRAM margin       : {args.vram_margin*100:.0f}%")
    info(f"Safety margin     : {args.safety_margin*100:.1f}%")
    info(f"Max batch runtime : {args.max_batch_runtime/60:.0f}min")
    info(f"Task hard timeout : {args.task_timeout/60:.0f}min")
    if use_temporal_waves:
        info(f"Temporal waves    : ENABLED (min_anchor_ratio={min_anchor_ratio:.1f})")
    else:
        info(f"Temporal waves    : DISABLED")
    info(f"Memory model      : {'Linear' if profile_loader._use_linear else 'Step-wise (legacy)'}")
    info(f"Memory profile    : {profile_loader.profile_source_label}")

    print_memory_step_summary(profile_loader)

    # -- Extra args --
    extra_args = []
    if args.boltz_extra_args:
        extra_args.extend(args.boltz_extra_args)
    if extra_args:
        info(f"Extra Boltz arguments: {' '.join(extra_args)}")

    # -- Singularity test --
    if not test_singularity_command(singularity_image, boltz_cache_path,
                                    input_dir, output_path, selected_gpus[0]):
        error("Singularity test failed. Check your configuration."); sys.exit(1)

    # -- Collect YAML files (parallel) --
    cpu_count = multiprocessing.cpu_count()
    if args.cpu_workers is not None:
        cpu_workers = args.cpu_workers
        info(f"CPU workers for YAML parsing: {cpu_workers} (manual; {cpu_count} logical CPUs)")
    else:
        cpu_workers = cpu_count
        info(f"CPU workers for YAML parsing: {cpu_workers} (auto-detected)")

    skip_existing = not args.no_skip_existing
    yaml_files, skipped_files = collect_yaml_files(
        args.input_dir, output_path, skip_existing, args.verbose, cpu_workers
    )

    if not yaml_files:
        if skipped_files:
            success("All tasks already have output. Nothing to do.")
        else:
            error("No tasks to process.")
        sys.exit(0)

    # -- Build tasks --
    tasks: List[PredictionTask] = []
    token_skipped_files: List[Tuple[Path, Dict]] = []
    vram_overflow_count = 0

    for i, (yaml_file, yaml_info) in enumerate(yaml_files, 1):
        seq_len = yaml_info.get('sequence_length', 0)

        if args.max_seq_length is not None and seq_len > args.max_seq_length:
            token_skipped_files.append((yaml_file, yaml_info))
            continue

        raw_memory = profile_loader.estimate_memory_mb(seq_len)
        estimated_memory = int(raw_memory * args.memory_estimation_factor)
        estimated_runtime = profile_loader.estimate_runtime_seconds(seq_len)
        timeout_risk = profile_loader.is_timeout_risk(seq_len, args.task_timeout)
        over_vram = profile_loader.is_over_gpu_vram(seq_len)

        if over_vram:
            vram_overflow_count += 1
            if args.skip_vram_overflow:
                token_skipped_files.append((yaml_file, yaml_info))
                continue

        task = PredictionTask(
            yaml_file=yaml_file, yaml_info=yaml_info,
            estimated_memory=estimated_memory,
            estimated_runtime=estimated_runtime,
            task_id=f"task_{i:04d}",
            timeout_risk=timeout_risk, vram_overflow=over_vram
        )
        tasks.append(task)

    info(f"Created {len(tasks)} prediction tasks")
    if token_skipped_files:
        warning(f"Skipped {len(token_skipped_files)} task(s) due to filters")
    if not tasks:
        warning("No tasks remain after applying filters.")
        result_writer = StreamingResultWriter(args.output_file)
        result_writer.write_header()
        for yaml_file, yi in skipped_files:
            result_writer.write_skipped_task(yaml_file, yi, reason='skipped_existing_output')
        for yaml_file, yi in token_skipped_files:
            result_writer.write_skipped_task(yaml_file, yi, reason='skipped_filter')
        result_writer.close()
        sys.exit(0)

    # -- Create Optimizer --
    optimizer = DualDimensionTaskOptimizer(
        max_memory_mb=args.gpu_memory,
        safety_margin=args.safety_margin,
        max_batch_runtime_seconds=args.max_batch_runtime
    )

    # -- Distribute tasks --
    tasks_by_gpu = distribute_tasks_by_tokens(tasks, len(selected_gpus))
    gpu_workers = create_gpu_workers(
        tasks_by_gpu, selected_gpus, optimizer, workspace_root,
        min_anchor_ratio=min_anchor_ratio,
        use_temporal_waves=use_temporal_waves,
        max_anchor_group_ratio=args.max_anchor_group_ratio,
    )

    _cleanup_state['gpu_workers'] = gpu_workers
    _cleanup_state['input_dir'] = input_dir

    vram_overflow_token = profile_loader.get_vram_overflow_threshold()
    print_gpu_distribution_summary(gpu_workers, len(tasks))
    print_optimization_summary(gpu_workers, len(tasks), args.gpu_memory,
                               len(skipped_files), vram_overflow_count, vram_overflow_token)

    if args.test_only:
        info("Test-only mode -- exiting without running tasks.")
        sys.exit(0)

    # -- Execute --
    result_writer = StreamingResultWriter(args.output_file)
    result_writer.write_header()

    if skipped_files:
        for yaml_file, yi in skipped_files:
            result_writer.write_skipped_task(yaml_file, yi, reason='skipped_existing_output')
        info(f"Recorded {len(skipped_files)} skipped tasks (existing output)")
    if token_skipped_files:
        for yaml_file, yi in token_skipped_files:
            reason = 'skipped_vram_overflow' if args.skip_vram_overflow and \
                profile_loader.is_over_gpu_vram(yi.get('sequence_length', 0)) \
                else 'skipped_seq_limit'
            result_writer.write_skipped_task(yaml_file, yi, reason=reason)

    tw_count = sum(1 for w in gpu_workers
                   for b in w.batches if isinstance(b, TemporalWaveBatch))
    tw_wave_tasks = sum(b.wave_task_count for w in gpu_workers
                        for b in w.batches if isinstance(b, TemporalWaveBatch))
    info(f"\nStarting multi-GPU execution with {len(gpu_workers)} workers...")
    if use_temporal_waves and tw_count > 0:
        info(f"{tw_count} TemporalWaveBatch(es) will process "
             f"{tw_wave_tasks} wave tasks in anchor VRAM shadow")
    info("Streaming mode - results written immediately as tasks complete")

    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=len(gpu_workers)) as executor:
        futures = {
            executor.submit(
                run_gpu_worker, worker, singularity_image,
                boltz_cache_path, output_path, extra_args,
                args.strict_errors, args.max_workers, args.task_timeout,
                input_dir, result_writer, optimizer
            ): worker for worker in gpu_workers
        }

        gpu_results = []
        for future in as_completed(futures):
            worker = futures[future]
            try:
                result = future.result()
                gpu_results.append(result)
                info(f"\n{'='*60}")
                success(f"GPU {worker.gpu_id} COMPLETED: "
                       f"{result['total_ok']} ok / {result['total_fail']} failed "
                       f"in {result['total_time']/3600:.2f}h")
                info(f"{'='*60}")
            except Exception as e:
                error(f"GPU {worker.gpu_id} execution error: {e}")
                gpu_results.append({
                    'gpu_id': worker.gpu_id,
                    'total_ok': 0,
                    'total_fail': len(worker.tasks),
                    'total_time': 0
                })

    # -- Final Summary --
    total_time = time.time() - overall_start
    result_writer.close()

    total_ok = sum(r['total_ok'] for r in gpu_results)
    total_fail = sum(r['total_fail'] for r in gpu_results)
    total_retry_ok = sum(r.get('retry_ok', 0) for r in gpu_results)
    total_rerun_recovered = sum(r.get('rerun_recovered', 0) for r in gpu_results)
    total_rerun_pool = sum(r.get('rerun_pool_size', 0) for r in gpu_results)
    total_deferred = sum(r.get('deferred_wave_tasks', 0) for r in gpu_results)

    print_colored("\n" + "="*80, Colors.GREEN)
    print_colored("MULTI-GPU EXECUTION COMPLETED", Colors.GREEN)
    print_colored("="*80, Colors.GREEN)
    success(f"Total runtime      : {total_time:.1f}s ({total_time/3600:.2f}h)")
    success(f"GPUs used          : {len(gpu_workers)}")
    success(f"Tasks processed    : {total_ok + total_fail}")
    if skipped_files:
        info(f"Tasks skipped      : {len(skipped_files)} (existing output)")
    if token_skipped_files:
        info(f"Tasks skipped      : {len(token_skipped_files)} (filters)")
    success(f"Successful (total) : {total_ok}")
    if total_fail:
        warning(f"Final failures     : {total_fail}")
    if total_deferred:
        info(f"Wave tasks deferred: {total_deferred} (anchor finished early)")
    if total_rerun_pool:
        info(f"Parallel rerun pool: {total_rerun_pool} task(s)")
        if total_rerun_recovered:
            info(f"  -> Recovered      : {total_rerun_recovered} (parallel rerun)")
    if total_retry_ok:
        info(f"  -> Recovered      : {total_retry_ok} (per-task retry)")
    if total_ok + total_fail > 0:
        success(f"Success rate       : {total_ok/(total_ok+total_fail)*100:.1f}%")
    success(f"Results saved      : {args.output_file}")
    print_colored("="*80 + "\n", Colors.GREEN)


# ------------------------------------------------------------
#  Signal Handling & Cleanup
# ------------------------------------------------------------
_cleanup_state = {
    'workers': [],
    'input_dir': None,
    'gpu_workers': [],
}


def restore_yaml_files_from_gpu_work(gpu_work_dir: Path, input_dir: Path):
    if not gpu_work_dir.exists():
        return 0
    restored = 0
    try:
        for yaml_file in gpu_work_dir.glob('*.yaml'):
            dest = input_dir / yaml_file.name
            if not dest.exists():
                shutil.move(str(yaml_file), str(dest))
                restored += 1
            else:
                yaml_file.unlink()
        for yaml_file in gpu_work_dir.glob('*.yml'):
            dest = input_dir / yaml_file.name
            if not dest.exists():
                shutil.move(str(yaml_file), str(dest))
                restored += 1
            else:
                yaml_file.unlink()
        if not any(gpu_work_dir.iterdir()):
            gpu_work_dir.rmdir()
    except Exception as e:
        warning(f"Error during restore: {e}")
    return restored


def signal_handler(signum, frame):
    warning("\nInterrupt received -- attempting to restore files...")
    try:
        if _cleanup_state.get('input_dir') and _cleanup_state.get('gpu_workers'):
            input_dir = Path(_cleanup_state['input_dir'])
            for worker in _cleanup_state['gpu_workers']:
                if hasattr(worker, 'working_dir') and worker.working_dir.exists():
                    restored = restore_yaml_files_from_gpu_work(
                        worker.working_dir, input_dir)
                    if restored > 0:
                        info(f"Restored {restored} files from {worker.working_dir.name}")
    except Exception as e:
        warning(f"Cleanup error: {e}")
    warning("Cleanup attempted. Exiting...")
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
