"""Reproducible synthetic benchmarks for kfbatch's largest scheduler inputs."""

from __future__ import annotations

import argparse
import gc
import time
import tracemalloc

from kfbatch.stat import (
    get_qstat_df,
    get_scontrol_node_df,
    get_squeue_user_df,
    get_user_df,
)


def _qstat_lines(node_count, extra_resource_count):
    lines = ["queuename qtype resv/used/tot. load_avg arch states"]
    for node_index in range(node_count):
        lines.append(f"epyc.q@node{node_index:05d} BP 0/1/64 0.10 lx-amd64")
        lines.append("\thc:mem_req=120.000G")
        lines.append("\thl:mem_total=256.000G")
        lines.extend(
            f"\thl:unused_resource_{resource_index}=value"
            for resource_index in range(extra_resource_count)
        )
    return lines


def _uge_job_lines(job_count):
    return [
        (
            f" {job_id} 0.500 job{job_id} user r 07/30/2026 12:00:00 "
            f"epyc.q@node{job_id % 100:03d} 1 1-100:1"
        )
        for job_id in range(job_count)
    ]


def _slurm_job_lines(job_count):
    return [
        (f"{job_id}\tepyc\tjob{job_id}\tuser\taccount\tPD\t0:00\t1\t1\t2G\t1:00:00\t(Priority)")
        for job_id in range(job_count)
    ]


def _slurm_node_lines(node_count):
    return [
        (
            f"NodeName=node{node_index:05d} Arch=x86_64 CPUTot=64 CPUEfctv=64 "
            "CPUAlloc=8 RealMemory=262144 AllocMem=32768 "
            "State=MIXED Partitions=epyc,short"
        )
        for node_index in range(node_count)
    ]


def _measure(name, input_factory, parser):
    lines = input_factory()
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    frame = parser(lines)
    elapsed = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    print(
        f"{name}: rows={len(frame):,} cols={len(frame.columns)} "
        f"time={elapsed:.3f}s py_peak={peak_bytes / 1024 / 1024:.1f}MiB "
        f"frame={frame_bytes / 1024 / 1024:.1f}MiB"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qstat-nodes", type=int, default=5_000)
    parser.add_argument("--qstat-extra-resources", type=int, default=120)
    parser.add_argument("--jobs", type=int, default=100_000)
    parser.add_argument("--slurm-nodes", type=int, default=50_000)
    args = parser.parse_args()

    _measure(
        "qstat",
        lambda: _qstat_lines(args.qstat_nodes, args.qstat_extra_resources),
        get_qstat_df,
    )
    _measure("uge-jobs", lambda: _uge_job_lines(args.jobs), get_user_df)
    _measure("slurm-jobs", lambda: _slurm_job_lines(args.jobs), get_squeue_user_df)
    _measure(
        "slurm-nodes",
        lambda: _slurm_node_lines(args.slurm_nodes),
        lambda lines: get_scontrol_node_df(
            lines,
            partition_state_map={"epyc": "UP", "short": "UP"},
        ),
    )


if __name__ == "__main__":
    main()
