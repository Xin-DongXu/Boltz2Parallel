"""Console-script entry points with correct initialization per command."""

from __future__ import annotations

import signal


def run_parallel() -> int:
    from boltz2parallel.parallel import main, signal_handler

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
    return 0


def run_gpu_memory_profiler() -> int:
    from boltz2parallel.gpu_memory_profiler import main

    main()
    return 0


def run_gpu_memory_timeseries_profiler() -> int:
    from boltz2parallel.gpu_memory_timeseries_profiler import main

    main()
    return 0


def run_yaml_generator() -> int:
    from boltz2parallel.yaml_generator import main

    result = main()
    return int(result) if result is not None else 0


def run_yaml_add_msa() -> int:
    from boltz2parallel.yaml_add_msa import main

    result = main()
    return int(result) if result is not None else 0
