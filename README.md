[![Tests](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml/badge.svg)](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/kfuku52/kfbatch/blob/master/.github/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](https://github.com/kfuku52/kfbatch/blob/master/LICENSE)

## Overview

**kfbatch** prints a compact resource summary for batch clusters.

It supports both:

- `SLURM` via `squeue`
- `UGE/SGE` via `qstat -F`

On SLURM, `kfbatch` can also combine node state, partition state, active reservations, and
`sprio` output to show a reservation-adjusted, priority-aware single-node launch heuristic
for the current user.

## Installation

```bash
pip install git+https://github.com/kfuku52/kfbatch
```

Confirm the installed options:

```bash
kfbatch -h
```

## What It Prints

Depending on the scheduler, `kfbatch` reports:

- queued/running/failed task counts
- cluster-wide node, CPU, and RAM summaries
- top nodes by available RAM
- top nodes by available cores
- on SLURM, a per-partition launch heuristic for the current user

In SLURM mode, task counts are shown for both the current user and all users.

## Quick Start

Default SLURM mode:

```bash
kfbatch
```

UGE mode:

```bash
kfbatch --stat_command "qstat -F"
```

Legacy alias:

```bash
kfbatch stat --stat_command "qstat -F"
```

## Useful Examples

SLURM with the default live commands:

```bash
kfbatch \
  --stat_command "squeue" \
  --slurm_node_command "scontrol show node -o" \
  --slurm_partition_command "scontrol show partition -o"
```

SLURM with fixture files for debugging:

```bash
kfbatch \
  --example_file squeue_notrunc.txt \
  --stat_command "squeue" \
  --slurm_node_example_file scontrol_show_node_o.txt \
  --slurm_partition_example_file scontrol_show_partition_o.txt
```

UGE using a single snapshot instead of repeated polling:

```bash
kfbatch \
  --stat_command "qstat -F" \
  --niter 1
```

Write the parsed resource table to TSV:

```bash
kfbatch --out kfbatch.tsv
```

Disable the SLURM launch heuristic:

```bash
kfbatch --show_launch_heuristic no
```

## Example Output

```text
# of running job tasks for current user (estimated from squeue): 0
# of running job tasks for all users (estimated from squeue): 271
# of queued job tasks for current user (estimated from squeue): 0
# of queued job tasks for all users (estimated from squeue): 30125

Reporting working/abnormal/total nodes, available/used/reserved/abnormal/total CPUs, and available/total RAM:
epyc: 3/10/13 nodes, 68/267/0/1824/2159 CPUs, and 10/17,112G RAM
rome: 5/1/6 nodes, 404/236/0/128/768 CPUs, and 317/3,093G RAM

Reporting heuristic single-node launch ceilings for kf (reservation-adjusted, priority-aware):
epyc:
  immediate-start ceiling: n/a
  top free node: a004 has 59 CPUs and 925G RAM
  smallest current Priority-blocked request is 1 CPUs / 1G / 00:05:00
  priority gap: 3931
  fairshare gap: 3926
  note: current user has Priority-blocked jobs; no stable immediate-start ceiling can be inferred
```

## Notes

- `kfbatch` auto-detects the scheduler from `--stat_command`.
- In UGE mode, `--niter` controls how many times `qstat -F` is sampled; the reported availability
  is the minimum seen across iterations.
- In SLURM mode, old or truncated `squeue` formats are still accepted for parsing, but the launch
  heuristic falls back to `n/a` if request-size fields are unavailable.
- `kfbatch` is primarily maintained for the author's own cluster workflows, so site-specific output
  formats may still require custom command options.

## License

This program is MIT-licensed. See [LICENSE](LICENSE) for details.
