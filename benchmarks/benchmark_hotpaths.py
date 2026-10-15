"""Synthetic parser, rendering and bounded-I/O workloads with output checksums.

Run from the repository root with ``python -m benchmarks.benchmark_hotpaths``.
Use the same file, environment and arguments at both revisions; see performance.md.
"""

import argparse
import io
import json
import platform
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from unittest.mock import patch

import pandas

from benchmarks.benchmark_parsers import _qstat_lines, _slurm_job_lines, _slurm_node_lines
from benchmarks.benchmark_reports import _measure, _positive_int
from kfbatch import command, memory, render, stat, uge_parser


def _printed(operation, *args):
    with redirect_stdout(io.StringIO()) as output:
        operation(*args)
    return output.getvalue()


def _uge_snapshot(resources, jobs):
    def scheduler_output(**kwargs):
        return resources if kwargs["command_name"] == "--stat_command" else jobs

    options = Namespace(
        stat_command="qstat -F",
        example_file="",
        niter=1,
        current_user="current_user",
        uge_job_command="qstat -u '*'",
        uge_job_example_file="",
        scope="overview",
    )
    with patch.object(stat, "get_command_stdout_lines", scheduler_output):
        with redirect_stdout(io.StringIO()) as output:
            nodes, users = stat._get_uge_df(options, 60)
    return {
        "text": output.getvalue(),
        "nodes": nodes.to_json(orient="split"),
        "node_attrs": nodes.attrs,
        "users": users.to_json(orient="split"),
        "user_attrs": users.attrs,
    }


def _captured_command(mib):
    code = f"import os; os.write(1,(b'x'*1023+b'\\n')*{mib * 1024})" if mib else "print('ok')"
    result = command._run_command([sys.executable, "-c", code], "synthetic", 10)
    return [value.hex() if isinstance(value, bytes) else value for value in result]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--case", action="append", help="Select one or more named operations.")
    args = parser.parse_args()
    print(
        f"Python {platform.python_version()}, pandas {pandas.__version__}, {platform.platform()}",
        flush=True,
    )

    nodes = stat.get_scontrol_node_df(_slurm_node_lines(25_000), {"epyc": "UP", "short": "UP"})
    nodes["queue_name"] = [f"part{(i + i // 25_000) % 5:03d}" for i in range(len(nodes))]
    assert not nodes.duplicated(["queue_name", "node_name"]).any()
    nodes = stat.adjust_ram_unit(nodes)
    options = Namespace(ntop=1, all_tiers=False, exclude_abnormal_node=True)
    reservation_nodes = stat.get_scontrol_node_df(
        _slurm_node_lines(5_000), {"epyc": "UP", "short": "UP"}
    )
    explicit = pandas.DataFrame(
        [
            {
                "queue_name": "epyc",
                "node_name": f"node{i:05d}",
                "reservation_name": "reservation_a",
                "reserved_cores": 2,
                "reserved_mem_mb": 4096,
                "whole_node": False,
                "accessible": False,
                "access_users": "other_user",
            }
            for i in range(5_000)
        ]
    )
    wildcard = explicit.iloc[:1].assign(node_name="*")
    shares = [f"account_a|user{i}|1|0.1|{i}|0.01|{1 / (i + 1):.12f}" for i in range(20_000)]
    repeated = pandas.Series(["64G", "128G"] * 50_000)
    unique = pandas.Series([f"{i}M" for i in range(100_000)])
    jobs = stat.get_squeue_user_df(_slurm_job_lines(100_000)).assign(req_mem="2Gc")
    json_jobs = [
        json.dumps(
            {
                "job_info": {
                    "jobs": [
                        {
                            "JB_job_number": i + 1,
                            "JB_owner": f"user{i % 100}",
                            "state": "qw",
                            "slots": 2,
                            "ja_task_id": "1-20",
                        }
                        for i in range(50_000)
                    ]
                }
            }
        )
    ]
    users = " ".join(f"user{i}" for i in range(5_000))
    group = [
        "THE NUMBER OF RUNNING JOBS BY USER IN THE GROUP (group_a)",
        f"QNAME {users}",
        "THE NUMBER OF MEM_REQ BY USER IN THE GROUP (group_a)",
        f"QNAME {users}",
    ]
    data = (b"x" * 1023 + b"\n") * 16_384
    resources = _qstat_lines(5_000, 120)
    all_jobs = ["1 0.5 job current_user qw 10/15/2026 00:00:00 2 1-20"]
    operations = {
        "render-50000": lambda: _printed(render.print_slurm_compact_summary, nodes, None, options),
        "launch-50000": lambda: stat.get_slurm_launch_heuristic_df(nodes, None),
        "reservations-explicit-5000": lambda: stat._expand_reservation_rows(
            reservation_nodes, explicit
        ),
        "reservations-wildcard-5000": lambda: stat._expand_reservation_rows(
            reservation_nodes, wildcard
        ),
        "sshare-20000": lambda: stat.get_sshare_df(shares),
        "memory-repeated-100000": lambda: memory.memory_series_to_gib(repeated).to_frame(),
        "memory-unique-100000": lambda: memory.memory_series_to_gib(unique).to_frame(),
        "pending-100000": lambda: stat._smallest_priority_blocked_request(jobs),
        "uge-json-50000": lambda: uge_parser.get_uge_json_job_df(json_jobs),
        "qfree-members-5000": lambda: uge_parser._qfree_group_context(group),
        "decode-16MiB": lambda: command._decode_bounded_lines(data, "synthetic"),
        "uge-fallback-5000x120": lambda: _uge_snapshot(resources, all_jobs),
        "capture-0MiB": lambda: _captured_command(0),
        "capture-4MiB": lambda: _captured_command(4),
    }
    if args.case and set(args.case) - operations.keys():
        parser.error(f"Unknown cases: {sorted(set(args.case) - operations.keys())}")
    for name, operation in operations.items():
        if not args.case or name in args.case:
            _measure(name, operation, args.repeats, measure_memory=not args.skip_memory)


if __name__ == "__main__":
    main()
