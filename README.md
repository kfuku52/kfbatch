![kfbatch — compact cluster resource summaries for Slurm and AGE/UGE/SGE](https://raw.githubusercontent.com/kfuku52/kfbatch/main/docs/assets/kfbatch-header.png)

[![Tests](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://github.com/kfuku52/kfbatch/blob/main/.github/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kfuku52/kfbatch/blob/main/LICENSE)
[![Schedulers](https://img.shields.io/badge/schedulers-Slurm%20%7C%20AGE%2FUGE%2FSGE-orange)](#overview)

## Overview

`kfbatch` prints a compact, conservative snapshot of batch-cluster resources.

- Slurm: jobs from `squeue`, nodes and partitions from `scontrol`, active reservations,
  and an optional `sprio`-aware single-node launch heuristic.
- Altair Grid Engine (AGE), Univa Grid Engine (UGE), and Sun Grid Engine (SGE):
  queue instances from `qstat -F`, all-user job counts, and optional `qfree` quota
  and launch-slot data.

The AGE support is suitable for sites such as the SHIROKANE Supercomputer. Optional
site commands may fail without suppressing the core resource table; the output prints
an explicit `note: degraded ...` message whenever this happens.

## Installation

```bash
python -m pip install "git+https://github.com/kfuku52/kfbatch@main"
kfbatch -h
```

Python 3.10 or newer is required.

## Quick start

Slurm is the default:

```bash
kfbatch
```

AGE/UGE/SGE:

```bash
kfbatch --stat_command "qstat -F"
```

The AGE defaults use `qstat -u '*'` for all-user task counts and `qfree -c` for
site-specific quota data. Disable `qfree` where it is unavailable:

```bash
kfbatch --stat_command "qstat -F" --uge_qfree_command ""
```

## Output files

Node and job tables have separate, stable output options:

```bash
kfbatch --out_nodes nodes.tsv --out_jobs jobs.tsv
```

`--out` remains a legacy alias of `--out_nodes`. Specifying `--out` and
`--out_nodes` with different paths is an error. If node discovery fails,
`--out_jobs` can still be written, while a node file is never populated with
the incompatible job-table schema.

## Useful options

```bash
# Keep the minimum availability seen across three Grid Engine snapshots.
kfbatch --stat_command "qstat -F" --niter 3

# Limit every scheduler command to 15 seconds. Use 0 to disable timeouts.
kfbatch --command_timeout 15

# Disable the Slurm priority-aware launch heuristic.
kfbatch --show_launch_heuristic no

# Run entirely from the repository's synthetic fixtures.
kfbatch \
  --example_file tests/fixtures/slurm/squeue_full.txt \
  --stat_command squeue \
  --slurm_node_example_file tests/fixtures/slurm/nodes.txt \
  --slurm_partition_example_file tests/fixtures/slurm/partitions.txt \
  --slurm_reservation_example_file tests/fixtures/slurm/reservations.txt
```

## Example output

Slurm:

```text
jobs  self:R/Q/F=0/0/0  all:R/Q/F=4/2/0

part   nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU               topRAM               launch
epyc   2/0/2  48/8/64     144/192      compute02 32c/64GiB  compute01 16c/80GiB  <=32c/64GiB
short  0/1/1  0/0/16      ?/32         -                    -                    <=0c/0GiB

legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total
```

AGE/UGE/SGE with `qfree`:

```text
jobs  self:R/Q/F=0/0/0  all:R/Q/F=4/248/5

queue    nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU               topRAM  quota(s/g/l)  launch2G
mjobs.q  1/0/1  12/4/16     384/512      compute01 12c/24GiB  same    4/8/128       24(+16s)
intr.q   1/0/1  7/1/8       30/32        compute02 7c/20GiB   same    0/0/24        7

legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total
        ram uses qfree request headroom/capacity; topRAM is the best queue-instance request headroom
        quota=self/group/limit slots (inf=unlimited), launch2G=immediate 2G slots (+standby)
```

`?` means that the scheduler did not provide a trustworthy value. Memory is
normalized with binary scheduler units and displayed in GiB (`1G = 1024M`);
available memory is floored rather than rounded up so that the display does not
overstate launch capacity.

## Accuracy and failure behavior

- AGE task states such as `Rq` and `hRq` are pending, while `Eqw` is failed.
- The default text `qstat -u '*'` command preserves pending array ranges. Some
  AGE JSON versions collapse pending arrays and omit the range; when a custom
  JSON command is used, affected counts are marked as estimated.
- With `--niter`, a queue instance missing from any snapshot is treated as
  unavailable. A completely unparseable later snapshot is an error rather than
  silently reusing stale capacity.
- Slurm reservations are subtracted only while active. Reservations explicitly
  accessible to the current user are not subtracted. A node marked reserved but
  not resolvable from reservation output is conservatively unavailable.
- Optional partition, reservation, priority, all-user-job, and `qfree` failures
  produce degradation notes. Required job/resource command failures return a
  non-zero exit status.
- Scheduler output is decoded defensively, so a non-UTF-8 job name cannot crash
  the entire report.

## Development

All checked-in scheduler samples are small synthetic fixtures.

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest
python -m build
```

## License

This program is MIT-licensed. See [LICENSE](LICENSE) for details.
