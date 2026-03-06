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
- on SLURM, a compact one-row-per-partition table
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
jobs  self:R/Q/F=0/5/0  all:R/Q/F=239/7318/0

part    nodes    cpu(a/u/t)   ram(a/t)G  topCPU         topRAM         launch
epyc    3/10/13  14/321/2159  451/17112  a004 14c/5G    a017 0c/352G   PRIO min=1c/1G/5m gap=18097 fs=17960
rome    5/1/6    458/182/768  104/3093   at141 103c/12G at139 94c/49G  PRIO min=1c/1G/5m gap=1701 fs=1037
short   2/0/2    256/0/256    1031/1031  at137 128c/516G same          <=128c/516G

legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total
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
