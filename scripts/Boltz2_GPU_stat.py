#!/usr/bin/env python3
"""Boltz GPU memory monitor.

Iterates over Boltz YAML input files, runs each prediction, samples NVIDIA
GPU memory at a fixed interval, validates that output files were produced,
and writes a tab-separated summary suitable for downstream profiling.

For every distinct protein sequence length found in the input directory,
one representative YAML file is selected (subsequent files of the same
length are skipped) and processed in ascending length order.
"""

import os
import sys
import argparse
import time
import subprocess
import threading
import yaml
import csv
import signal
import select
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import tempfile
import json
import re

class Colors:
    """Color definitions for terminal output"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color

class GPUMonitor:
    """GPU memory monitoring class"""

    def __init__(self, interval: int = 1):
        self.interval = interval
        self.monitoring = False
        self.monitor_thread = None
        self.max_memory = 0
        self.gpu_data = []

    def check_nvidia_smi(self) -> bool:
        """Check if nvidia-smi is available"""
        try:
            result = subprocess.run(['nvidia-smi'],
                                  capture_output=True,
                                  text=True,
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def get_gpu_info(self) -> Dict[str, str]:
        """Get current GPU information"""
        try:
            # Get memory usage
            mem_cmd = ['nvidia-smi', '--query-gpu=memory.used,memory.total',
                      '--format=csv,noheader,nounits']
            mem_result = subprocess.run(mem_cmd, capture_output=True, text=True, timeout=5)

            # Get utilization and temperature
            util_cmd = ['nvidia-smi', '--query-gpu=utilization.gpu,temperature.gpu',
                       '--format=csv,noheader,nounits']
            util_result = subprocess.run(util_cmd, capture_output=True, text=True, timeout=5)

            if mem_result.returncode == 0 and util_result.returncode == 0:
                mem_data = mem_result.stdout.strip().split(', ')
                util_data = util_result.stdout.strip().split(', ')

                memory_used = int(mem_data[0].strip()) if mem_data[0].strip().isdigit() else 0
                memory_total = int(mem_data[1].strip()) if mem_data[1].strip().isdigit() else 0
                gpu_util = util_data[0].strip() if len(util_data) > 0 else 'N/A'
                temperature = util_data[1].strip() if len(util_data) > 1 else 'N/A'

                memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0

                return {
                    'memory_used': memory_used,
                    'memory_total': memory_total,
                    'memory_percent': memory_percent,
                    'gpu_util': gpu_util,
                    'temperature': temperature
                }
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass

        return {
            'memory_used': 0,
            'memory_total': 0,
            'memory_percent': 0,
            'gpu_util': 'N/A',
            'temperature': 'N/A'
        }

    MAX_GPU_DATA_ENTRIES = 86_400  # roughly 24 h at 1 s sampling

    def _monitor_loop(self):
        """GPU monitoring loop."""
        while self.monitoring:
            gpu_info = self.get_gpu_info()
            current_memory = gpu_info['memory_used']

            if current_memory > self.max_memory:
                self.max_memory = current_memory

            # Store data point, capped to avoid unbounded growth on long runs.
            if len(self.gpu_data) < self.MAX_GPU_DATA_ENTRIES:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.gpu_data.append({
                    'timestamp': timestamp,
                    'memory_used': current_memory,
                    'memory_total': gpu_info['memory_total'],
                    'memory_percent': gpu_info['memory_percent'],
                    'gpu_util': gpu_info['gpu_util'],
                    'temperature': gpu_info['temperature'],
                })

            time.sleep(self.interval)

    def start_monitoring(self):
        """Start GPU monitoring"""
        self.monitoring = True
        self.max_memory = 0
        self.gpu_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

def print_colored(message: str, color: str = Colors.NC):
    """Print colored message"""
    print(f"{color}{message}{Colors.NC}")

def info(message: str):
    """Print info message"""
    print_colored(f"[INFO] {message}", Colors.BLUE)

def success(message: str):
    """Print success message"""
    print_colored(f"[SUCCESS] {message}", Colors.GREEN)

def warning(message: str):
    """Print warning message"""
    print_colored(f"[WARNING] {message}", Colors.YELLOW)

def error(message: str):
    """Print error message"""
    print_colored(f"[ERROR] {message}", Colors.RED)

def debug(message: str):
    """Print debug message"""
    print_colored(f"[DEBUG] {message}", Colors.CYAN)

def parse_yaml_file(yaml_path: Path) -> Optional[Tuple[int, Optional[str]]]:
    """Parse YAML file and return (protein_sequence_length, yaml_name).

    yaml_name is the value of the top-level 'name' field when present, or
    None otherwise.  The Boltz YAML name field controls the output folder
    layout and may differ from the file stem.
    """
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        seq_len: Optional[int] = None
        if 'sequences' in data:
            for seq in data['sequences']:
                if 'protein' in seq and 'sequence' in seq['protein']:
                    seq_len = len(seq['protein']['sequence'])
                    break
        if seq_len is None:
            return None
        return seq_len, data.get('name')
    except Exception as e:
        warning(f"Failed to parse {yaml_path}: {e}")
        return None

def collect_yaml_files(input_dir: Path) -> Dict[int, Tuple[Path, Optional[str]]]:
    """Collect YAML files and group by protein sequence length."""
    info(f"Scanning YAML files in: {input_dir}")

    yaml_files = list(input_dir.rglob("*.yaml")) + list(input_dir.rglob("*.yml"))

    if not yaml_files:
        error(f"No YAML files found in {input_dir}")
        sys.exit(1)

    info(f"Found {len(yaml_files)} YAML files")

    # Group by sequence length, keep only one file per length.
    # Stored value is (yaml_path, yaml_name).
    length_to_file: Dict[int, Tuple[Path, Optional[str]]] = {}

    for yaml_file in yaml_files:
        parsed = parse_yaml_file(yaml_file)
        if parsed is None:
            continue
        seq_length, yaml_name = parsed
        if seq_length not in length_to_file:
            length_to_file[seq_length] = (yaml_file, yaml_name)
            info(f"Length {seq_length}: {yaml_file.name}")
        else:
            info(f"Length {seq_length}: Skipping {yaml_file.name} "
                 f"(already have one)")

    if not length_to_file:
        error("No valid YAML files with protein sequences found")
        sys.exit(1)

    info(f"Selected {len(length_to_file)} unique sequence lengths")
    return length_to_file

def validate_output(yaml_file: Path, out_dir: Path,
                    yaml_name: Optional[str] = None,
                    settle_seconds: float = 2.0
                    ) -> Tuple[bool, str, List[str]]:
    """Validate that Boltz produced structure files under out_dir.

    Searches both Boltz-1 and Boltz-2 output layouts:
      Boltz-1: out_dir/predictions/<name>/
      Boltz-2: out_dir/boltz_results_<name>/predictions/<name>/

    The name is taken from the YAML 'name' field when provided, otherwise
    from the YAML file stem.  Returns (success, reason, output_files).
    """
    yaml_stem = yaml_file.stem
    candidates = {yaml_stem.lower()}
    if yaml_name:
        candidates.add(yaml_name.lower())

    # Allow the filesystem a brief moment to flush.
    if settle_seconds > 0:
        time.sleep(settle_seconds)

    if not out_dir.exists():
        return False, f"Output directory does not exist: {out_dir}", []

    found_dirs: List[Path] = []

    # Layout A: Boltz-1
    pred_a = out_dir / "predictions"
    if pred_a.is_dir():
        for child in pred_a.iterdir():
            if child.is_dir() and child.name.lower() in candidates:
                found_dirs.append(child)

    # Layout B: Boltz-2
    try:
        for entry in out_dir.iterdir():
            if not (entry.is_dir() and entry.name.lower().startswith(
                    "boltz_results_")):
                continue
            pred_b = entry / "predictions"
            if not pred_b.is_dir():
                continue
            for child in pred_b.iterdir():
                if child.is_dir() and child.name.lower() in candidates:
                    found_dirs.append(child)
    except (PermissionError, FileNotFoundError, OSError):
        pass

    if not found_dirs:
        return False, (
            f"No prediction directory matching {sorted(candidates)} under "
            f"{out_dir}"
        ), []

    # Collect structure / score files (deliberately excluding checkpoint
    # files such as *.ckpt and *.pt, which are inputs to Boltz, not outputs).
    structure_exts = (".cif", ".pdb")
    score_exts = (".json",)
    output_files: List[Path] = []
    for d in found_dirs:
        for ext in structure_exts + score_exts + (".npz",):
            output_files.extend(d.rglob(f"*{ext}"))

    structure_files = [f for f in output_files
                        if f.suffix.lower() in structure_exts]
    if not structure_files:
        return False, (
            f"Prediction directory found ({found_dirs[0]}) but contains no "
            f"structure file (*.cif or *.pdb)"
        ), [str(f) for f in output_files]

    return True, (
        f"Found {len(structure_files)} structure file(s) in "
        f"{len(found_dirs)} prediction directory(ies)"
    ), [str(f) for f in output_files]

def run_boltz_prediction_with_realtime_output(
        yaml_file: Path,
        checkpoint_path: str,
        affinity_checkpoint_path: str,
        out_dir: Path,
        yaml_name: Optional[str] = None,
        timeout_hours: int = 2,
        ) -> Tuple[bool, float, str]:
    """Run a single Boltz prediction with streaming stdout display.

    Returns (success, runtime_seconds, detailed_reason).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "boltz", "predict", str(yaml_file),
        "--out_dir", str(out_dir),
        "--use_msa_server",
        "--checkpoint", checkpoint_path,
        "--affinity_checkpoint", affinity_checkpoint_path,
    ]

    info(f"Running: {' '.join(command)}")

    start_time = time.time()
    last_msa_message_time = start_time      # last time an MSA-related line was seen
    last_msa_print_time = 0.0                # last time the waiting indicator was printed

    try:
        # Start the process
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )

        output_lines = []
        msa_detected = False

        # Real-time output monitoring
        while True:
            if process.poll() is not None:
                break

            # Check if there's output available
            if sys.platform != 'win32':
                ready, _, _ = select.select([process.stdout], [], [], 0.1)
            else:
                ready = [process.stdout]  # Windows doesn't support select on pipes

            if ready:
                line = process.stdout.readline()
                if line:
                    line = line.strip()
                    output_lines.append(line)

                    # Print real-time output
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print_colored(f"[{current_time}] {line}", Colors.CYAN)

                    # Detect MSA server activity
                    if any(keyword in line.lower() for keyword in ['msa', 'server', 'alignment', 'colabfold']):
                        if not msa_detected:
                            info("MSA server activity detected - this may take several minutes...")
                            msa_detected = True
                        last_msa_message_time = time.time()

                    # Look for error patterns
                    if any(keyword in line.lower() for keyword in ['error', 'failed', 'exception', 'traceback']):
                        warning(f"Potential error detected: {line}")

            # Check for timeout
            current_time = time.time()
            if current_time - start_time > timeout_hours * 3600:
                warning(f"Process timed out after {timeout_hours} hours")
                process.terminate()
                time.sleep(5)
                if process.poll() is None:
                    process.kill()
                return False, timeout_hours * 3600, "Process timed out"

            # MSA waiting indicator: report cumulative wait since the last
            # actual MSA-related message, throttled to one line per 30s.
            silent_for = current_time - last_msa_message_time
            since_last_print = current_time - last_msa_print_time
            if msa_detected and silent_for > 30 and since_last_print > 30:
                print_colored(
                    f"[MSA] Waiting for MSA server response... "
                    f"({silent_for:.0f}s since last message)",
                    Colors.MAGENTA)
                last_msa_print_time = current_time

            time.sleep(0.1)

        # Get final output
        remaining_output = process.stdout.read()
        if remaining_output:
            for line in remaining_output.strip().split('\n'):
                if line.strip():
                    output_lines.append(line.strip())
                    current_time = datetime.now().strftime('%H:%M:%S')
                    print_colored(f"[{current_time}] {line.strip()}", Colors.CYAN)

        end_time = time.time()
        runtime = end_time - start_time

        # Check process exit code
        return_code = process.returncode
        process_success = (return_code == 0)

        if not process_success:
            reason = f"Process failed with exit code {return_code}"
            error(reason)
            # Print last few lines for debugging
            if output_lines:
                error("Last few output lines:")
                for line in output_lines[-10:]:
                    print_colored(f"  {line}", Colors.RED)
            return False, runtime, reason

        # Validate actual output files
        info("Validating output files...")
        output_valid, validation_reason, output_files = validate_output(
            yaml_file, out_dir, yaml_name=yaml_name)

        if not output_valid:
            warning(f"Output validation failed: {validation_reason}")
            return False, runtime, f"Process completed but {validation_reason}"

        info(f"Output validation successful: {validation_reason}")
        debug(f"Output files: {output_files[:5]}...")  # Show first 5 files

        return True, runtime, "Process completed successfully with valid output"

    except Exception as e:
        error(f"Process execution failed: {e}")
        return False, time.time() - start_time, f"Execution error: {str(e)}"

def write_results_header(output_file: Path):
    """Write TSV header"""
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            'yaml_file', 'sequence_length', 'peak_memory_mb',
            'runtime_seconds', 'success', 'failure_reason', 'timestamp'
        ])

def write_result(output_file: Path, yaml_file: Path, seq_length: int,
                peak_memory: int, runtime: float, success: bool, reason: str = ""):
    """Write single result to TSV file"""
    with open(output_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([
            yaml_file.name, seq_length, peak_memory,
            f"{runtime:.2f}", success, reason, datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ])

def print_progress_bar(current: int, total: int, prefix: str = "Progress"):
    """Print a progress bar"""
    length = 50
    percent = current / total
    filled = int(length * percent)
    bar = '#' * filled + '-' * (length - filled)
    print_colored(f"\r{prefix}: |{bar}| {percent:.1%} ({current}/{total})", Colors.GREEN)

def main():
    parser = argparse.ArgumentParser(
        description="GPU memory monitoring wrapper around boltz predict.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s -i /path/to/yaml/files -o results.tsv
    %(prog)s --input-dir ./data --output-file boltz_results.tsv --interval 2
    %(prog)s -i ./yamls -o results.tsv --timeout 3

Behaviour:
    - One representative YAML file is selected per distinct protein sequence
      length and processed in ascending length order.
    - GPU memory is sampled at the configured interval; the peak is recorded.
    - Stdout from boltz is streamed live; MSA server quiet periods are
      reported with a throttled waiting indicator.
    - After each run, output files are validated against the Boltz-1 and
      Boltz-2 directory layouts under --out-dir.
    - Per-task results are appended to the TSV output file.
        """)

    parser.add_argument('-i', '--input-dir',
                       type=Path,
                       required=True,
                       help='Directory containing YAML files')

    parser.add_argument('-o', '--output-file',
                       type=Path,
                       required=True,
                       help='Output TSV file for results')

    parser.add_argument('--checkpoint',
                       type=str,
                       default='~/.boltz/boltz2_conf.ckpt',
                       help='Path to Boltz checkpoint file (default: ~/.boltz/boltz2_conf.ckpt)')

    parser.add_argument('--affinity-checkpoint',
                       type=str,
                       default='~/.boltz/boltz2_aff.ckpt',
                       help='Path to Boltz affinity checkpoint file (default: ~/.boltz/boltz2_aff.ckpt)')

    parser.add_argument('--interval',
                       type=int,
                       default=1,
                       help='GPU monitoring interval in seconds (default: 1)')

    parser.add_argument('--timeout',
                       type=int,
                       default=2,
                       help='Timeout for each prediction in hours (default: 2)')

    parser.add_argument('--out-dir',
                       type=Path,
                       default=Path('./boltz_stat_output'),
                       help='Output directory passed to boltz predict '
                            '(default: ./boltz_stat_output)')

    parser.add_argument('--debug',
                       action='store_true',
                       help='Enable debug output')

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)

    if not args.input_dir.is_dir():
        error(f"Input path is not a directory: {args.input_dir}")
        sys.exit(1)

    # Expand checkpoint paths
    checkpoint_path = os.path.expanduser(args.checkpoint)
    affinity_checkpoint_path = os.path.expanduser(args.affinity_checkpoint)

    # Create output directory for the result TSV if needed
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    # Resolve and create the Boltz output directory
    boltz_out_dir = args.out_dir.resolve()
    boltz_out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(args.interval)

    # Check nvidia-smi
    if not gpu_monitor.check_nvidia_smi():
        error("nvidia-smi not available or GPU not accessible")
        sys.exit(1)

    # Print startup information
    print_colored("=" * 80, Colors.GREEN)
    success("Boltz GPU Memory Monitor")
    print_colored("=" * 80, Colors.GREEN)

    info(f"Input directory: {args.input_dir}")
    info(f"Output file: {args.output_file}")
    info(f"Boltz out dir: {boltz_out_dir}")
    info(f"Checkpoint: {checkpoint_path}")
    info(f"Affinity checkpoint: {affinity_checkpoint_path}")
    info(f"Monitor interval: {args.interval}s")
    info(f"Timeout per prediction: {args.timeout}h")
    info(f"Debug mode: {'ON' if args.debug else 'OFF'}")

    # Collect and organize YAML files
    length_to_file = collect_yaml_files(args.input_dir)

    # Sort by sequence length (shortest first)
    sorted_files = sorted(length_to_file.items())

    # Initialize output file
    write_results_header(args.output_file)

    print_colored("=" * 80, Colors.BLUE)
    info(f"Starting predictions for {len(sorted_files)} files...")
    print_colored("=" * 80, Colors.BLUE)

    # Statistics
    successful_runs = 0
    failed_runs = 0
    total_runtime = 0

    # Process files from shortest to longest
    for i, (seq_length, (yaml_file, yaml_name)) in enumerate(sorted_files, 1):
        print_colored(f"\n{'='*20} FILE {i}/{len(sorted_files)} {'='*20}",
                      Colors.YELLOW)
        info(f"Processing: {yaml_file.name}")
        info(f"Sequence length: {seq_length}")
        info(f"File path: {yaml_file}")
        info(f"Output dir : {boltz_out_dir}")

        # Print progress bar
        print_progress_bar(i - 1, len(sorted_files), "Overall Progress")

        # Start GPU monitoring
        info("Starting GPU monitoring...")
        gpu_monitor.start_monitoring()

        try:
            # Run Boltz prediction with real-time output
            success_status, runtime, detailed_reason = (
                run_boltz_prediction_with_realtime_output(
                    yaml_file,
                    checkpoint_path,
                    affinity_checkpoint_path,
                    out_dir=boltz_out_dir,
                    yaml_name=yaml_name,
                    timeout_hours=args.timeout,
                )
            )

        finally:
            # Stop GPU monitoring
            info("Stopping GPU monitoring...")
            gpu_monitor.stop_monitoring()

        # Get peak memory usage
        peak_memory = gpu_monitor.max_memory
        total_runtime += runtime

        # Update statistics
        if success_status:
            successful_runs += 1
        else:
            failed_runs += 1

        # Write result
        write_result(args.output_file, yaml_file, seq_length,
                    peak_memory, runtime, success_status, detailed_reason)

        # Print detailed summary
        print_colored(f"\n{'='*50}", Colors.GREEN if success_status else Colors.RED)
        status_color = Colors.GREEN if success_status else Colors.RED
        status_text = "[OK] SUCCESS" if success_status else "[X] FAILED"
        print_colored(f"Status: {status_text}", status_color)
        print_colored(f"File: {yaml_file.name}", Colors.BLUE)
        print_colored(f"Sequence length: {seq_length}", Colors.BLUE)
        print_colored(f"Runtime: {runtime:.2f}s ({runtime/60:.1f}min)", Colors.BLUE)
        print_colored(f"Peak GPU memory: {peak_memory}MB", Colors.BLUE)
        print_colored(f"Reason: {detailed_reason}", status_color)
        print_colored(f"{'='*50}", Colors.GREEN if success_status else Colors.RED)

        # Print running statistics
        info(f"Running stats - Success: {successful_runs}, Failed: {failed_runs}, Total time: {total_runtime/60:.1f}min")

        # Small delay between runs for system stability
        if i < len(sorted_files):
            info("Waiting 5 seconds before next prediction...")
            time.sleep(5)

    # Final progress bar
    print_progress_bar(len(sorted_files), len(sorted_files), "Overall Progress")
    print()

    # Print final summary
    print_colored("\n" + "="*80, Colors.GREEN)
    success("ALL PREDICTIONS COMPLETED!")
    print_colored("="*80, Colors.GREEN)

    success(f"Results saved to: {args.output_file}")
    info(f"Total files processed: {len(sorted_files)}")
    success(f"Successful predictions: {successful_runs}")
    if failed_runs > 0:
        warning(f"Failed predictions: {failed_runs}")
    info(f"Total runtime: {total_runtime/60:.1f} minutes ({total_runtime/3600:.1f} hours)")
    info(f"Average runtime per file: {total_runtime/len(sorted_files):.1f} seconds")

    if failed_runs > 0:
        warning(f"Check the output file for detailed failure reasons: {args.output_file}")

def signal_handler(signum, frame):
    """Handle interrupt signals"""
    warning("\nReceived interrupt signal. Cleaning up...")
    sys.exit(130)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    main()
