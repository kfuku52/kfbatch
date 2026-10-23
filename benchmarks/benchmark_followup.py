"""Synthetic follow-up workloads; compare timings and checksums across revisions.

Inputs are built outside measurement. Use --case to select workloads, and measure
allocations separately from timing. See docs/performance.md for interpretation.
"""

import argparse
import io
import platform
from contextlib import redirect_stdout
from types import SimpleNamespace

import pandas

from benchmarks.benchmark_parsers import (
    _qstat_lines,
    _slurm_job_lines,
    _slurm_node_lines,
    _uge_job_lines,
)
from benchmarks.benchmark_reports import _measure, _positive_int
from kfbatch import batch_scope, render, stat

CASES = (
    "users-slurm",
    "users-uge",
    "one-user-slurm",
    "one-user-uge",
    "totals-slurm",
    "totals-uge",
    "render-slurm",
    "render-uge",
    "top-nodes",
    "launch-queues",
    "squeue-repeated",
    "squeue-unique",
    "uge-array-repeated",
    "uge-array-unique",
    "uge-no-array",
    "qstat-resources",
)


def _printed(operation, *args):
    with redirect_stdout(io.StringIO()) as output:
        operation(*args)
    return output.getvalue()


def _scope_jobs(count, users, scheduler):
    states = ["R", "PD", "FAILED", "SUSPENDED"] if scheduler == "slurm" else ["r", "qw", "Eqw", ""]
    frame = pandas.DataFrame(
        {
            "user": [f"user{i % users:05d}" for i in range(count)],
            "state": [states[i % 4] for i in range(count)],
            "queue_name": ["main.q" if i % 2 else "" for i in range(count)],
            "total_slots": [i % 31 + 1 for i in range(count)],
        }
    )
    for column in range(12):
        frame[f"unused_{column}"] = [f"text{i % 100}" for i in range(count)]
    return frame


def _nodes(count, queues):
    physical_count = count if queues == 1 else (count + 1) // 2
    frame = (
        stat.get_scontrol_node_df(_slurm_node_lines(physical_count), {"epyc": "UP", "short": "UP"})
        .iloc[:count]
        .copy()
    )
    frame["queue_name"] = [f"part{(i + i // physical_count) % queues:04d}" for i in range(count)]
    frame["ncore_available"] = [(i % physical_count) % 57 for i in range(count)]
    frame["hc:mem_req"] = [f"{((i % physical_count) % 129) * 1024}M" for i in range(count)]
    assert not frame.duplicated(["queue_name", "node_name"]).any()
    return frame


def _operation(name, args):
    if name.startswith(("users-", "one-user-", "totals-")):
        scheduler = name.rsplit("-", 1)[1]
        users = 1 if name.startswith("one-user-") else args.users
        jobs = _scope_jobs(args.jobs, users, scheduler)
        if name.startswith("totals-"):
            return lambda: batch_scope.aggregate_jobs(jobs, scheduler)
        members = [f"user{i:05d}" for i in range(users + 2)]
        return lambda: _printed(batch_scope._print_user_breakdown, jobs, scheduler, members)
    if name in {"render-slurm", "render-uge", "top-nodes", "launch-queues"}:
        nodes = _nodes(args.nodes, 1 if name == "top-nodes" else args.queues)
        if name == "launch-queues":
            return lambda: stat.get_slurm_launch_heuristic_df(nodes, None)
        nodes = stat.adjust_ram_unit(nodes)
        options = SimpleNamespace(ntop=1, all_tiers=False, exclude_abnormal_node=True)
        if name == "top-nodes":
            # The renderer strips accounting attrs before selecting its top nodes.
            nodes.attrs = {}
            return lambda: render._format_compact_top_nodes(
                nodes, "ncore_available", "hc:mem_req", options
            )
        if name == "render-uge":
            nodes.attrs = {}
        printer = getattr(render, f"print_{name.rsplit('-', 1)[1]}_compact_summary")
        return lambda: _printed(printer, nodes, None, options)
    if name.startswith("squeue-"):
        lines = _slurm_job_lines(args.jobs)
        if name.endswith("unique"):
            lines = [line.replace("\t2G\t", f"\t{i + 1}M\t") for i, line in enumerate(lines)]
        return lambda: stat.get_squeue_user_df(lines)
    if name.startswith("uge-"):
        lines = _uge_job_lines(args.jobs)
        if name.endswith("unique"):
            lines = [line.replace("1-100:1", f"1-{i + 1}:1") for i, line in enumerate(lines)]
        elif name == "uge-no-array":
            lines = [line.removesuffix(" 1-100:1") for line in lines]
        return lambda: stat.get_user_df(lines)
    lines = _qstat_lines(args.qstat_nodes, 120)
    return lambda: stat.get_qstat_df(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument("--jobs", type=_positive_int, default=100_000)
    parser.add_argument("--nodes", type=_positive_int, default=50_000)
    parser.add_argument("--qstat-nodes", type=_positive_int, default=5_000)
    parser.add_argument("--queues", type=_positive_int, default=100)
    parser.add_argument("--users", type=_positive_int, default=2_000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--skip-memory", action="store_true")
    args = parser.parse_args()
    print(
        f"Python {platform.python_version()}, pandas {pandas.__version__}, {platform.platform()}",
        flush=True,
    )
    print(
        f"jobs={args.jobs} users={args.users} nodes={args.nodes} "
        f"queues={args.queues} qstat_nodes={args.qstat_nodes} repeats={args.repeats}",
        flush=True,
    )
    for name in args.case or CASES:
        operation = _operation(name, args)
        _measure(name, operation, args.repeats, measure_memory=not args.skip_memory)


if __name__ == "__main__":
    main()
