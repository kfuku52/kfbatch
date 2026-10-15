"""Repeated synthetic report benchmarks; input construction is outside measurement.

Run with ``python -m benchmarks.benchmark_reports``. Compare checksums as well as
wall/CPU times between revisions in the same environment. Peak Python allocation
is measured separately so tracemalloc does not distort the time samples.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import platform
import statistics
import time
import tracemalloc
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass

import pandas

from benchmarks.benchmark_parsers import _qstat_lines, _slurm_job_lines, _slurm_node_lines
from kfbatch.batch_scope import aggregate_jobs
from kfbatch.stat import (
    get_qstat_df,
    get_scontrol_node_df,
    get_slurm_fairshare_rank_summary,
    get_slurm_launch_heuristic_df,
    get_squeue_user_df,
    print_queued_job_summary,
)
from kfbatch.uge_parser import _merge_qstat_iteration_min_availability


def _checksum(result):
    if isinstance(result, pandas.DataFrame):
        result = {
            "data": result.to_json(orient="split", double_precision=15),
            "dtypes": result.dtypes.astype(str).tolist(),
            "attrs": result.attrs,
        }
    elif is_dataclass(result):
        result = asdict(result)
    return hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()


def _measure(name, operation, repeats, *, measure_memory=True):
    expected = _checksum(operation())  # Warm up and retain an output-equivalence check.
    samples = []
    cpu_samples = []
    for _ in range(repeats):
        gc.collect()
        started = time.perf_counter()
        cpu_started = time.process_time()
        result = operation()
        cpu_samples.append(time.process_time() - cpu_started)
        samples.append(time.perf_counter() - started)
        assert _checksum(result) == expected
    peak = "skipped"
    if measure_memory:
        gc.collect()
        tracemalloc.start()
        result = operation()
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert _checksum(result) == expected
        peak = f"{peak_bytes / 1024 / 1024:.2f}MiB"
    print(
        f"{name}: median={statistics.median(samples):.6f}s "
        f"min={min(samples):.6f}s max={max(samples):.6f}s "
        f"cpu_median={statistics.median(cpu_samples):.6f}s "
        f"cpu_min={min(cpu_samples):.6f}s cpu_max={max(cpu_samples):.6f}s "
        f"py_peak={peak} checksum={expected}",
        flush=True,
    )


def _printed_uge_summary(jobs):
    with redirect_stdout(io.StringIO()) as output:
        print_queued_job_summary(jobs, scheduler="uge", current_user="user0")
    return output.getvalue()


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=_positive_int, default=100_000)
    parser.add_argument("--users", type=_positive_int, default=2_000)
    parser.add_argument("--nodes", type=_positive_int, default=5_000)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--skip-memory", action="store_true", help="Skip the separate allocation run."
    )
    args = parser.parse_args()
    print(f"Python {platform.python_version()}, pandas {pandas.__version__}, {platform.platform()}")
    print(f"jobs={args.jobs} users={args.users} nodes={args.nodes} repeats={args.repeats}")

    jobs = get_squeue_user_df(_slurm_job_lines(args.jobs))
    jobs["user"] = [f"user{index % args.users}" for index in range(args.jobs)]
    shares = pandas.DataFrame(
        {
            "user": [f"user{index}" for index in range(args.users)],
            "account": "account",
            "fairshare": [1 / (index + 1) for index in range(args.users)],
        }
    )
    uge_jobs = jobs.assign(
        state=[("r", "qw", "Eqw", "h")[index % 4] for index in range(args.jobs)],
        queue_name=["epyc.q" if index % 4 == 0 else "" for index in range(args.jobs)],
    )
    snapshot = get_qstat_df(_qstat_lines(args.nodes, 0))
    later_snapshot = snapshot.copy()
    later_snapshot.loc[later_snapshot.index[::3], "status"] = "a"
    later_snapshot.loc[later_snapshot.index[::5], "hc:mem_req"] = "100.000G"
    nodes = get_scontrol_node_df(
        _slurm_node_lines(args.nodes), partition_state_map={"epyc": "UP", "short": "UP"}
    )
    own_jobs = jobs.assign(user="user0", req_mem="2Gc")
    operations = {
        "slurm-group-totals": lambda: aggregate_jobs(jobs, "slurm"),
        "uge-group-totals": lambda: aggregate_jobs(uge_jobs, "uge"),
        "uge-summary": lambda: _printed_uge_summary(uge_jobs),
        "slurm-fairshare": lambda: get_slurm_fairshare_rank_summary(jobs, shares, "user0"),
        "qstat-merge": lambda: _merge_qstat_iteration_min_availability(snapshot, later_snapshot),
        "slurm-launch": lambda: get_slurm_launch_heuristic_df(
            nodes, own_jobs, current_user="user0"
        ),
    }
    for name, operation in operations.items():
        _measure(name, operation, args.repeats, measure_memory=not args.skip_memory)


if __name__ == "__main__":
    main()
