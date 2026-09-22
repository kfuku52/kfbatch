![kfbatch — compact cluster resource summaries for Slurm and AGE/UGE/SGE](docs/assets/kfbatch-header.png)

[![Tests](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/kfuku52/kfbatch/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.14-blue)](https://github.com/kfuku52/kfbatch/blob/main/.github/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Schedulers](https://img.shields.io/badge/schedulers-Slurm%20%7C%20AGE%2FUGE%2FSGE-orange)](#overview)

## Overview

`kfbatch` prints compact and deliberately conservative snapshots of batch-cluster
resources and filesystem quotas:

- Slurm jobs from `squeue`, nodes/partitions/reservations from `scontrol`,
  priority factors from `sprio`, and association FairShare data from `sshare`.
- Altair Grid Engine (AGE), Univa Grid Engine (UGE), and Sun Grid Engine (SGE)
  queue instances from `qstat -F`, all-user jobs from `qstat -u '*'`, and optional
  site quota/launch-slot data from `qfree -c`.
- Personal and shared group disk usage/limits from `lfsq`, standard `quota`, Lustre
  `lfs quota`, or a site-specific command.

AGE support is tested against the SHIROKANE Supercomputer command formats. SHIROKANE
currently exposes AGE/UGE commands and does not provide Slurm commands on its login
nodes. Optional site commands can be disabled or may degrade independently; the
report labels unavailable or untrusted data instead of treating it as free capacity.

Large scheduler reports are parsed with bounded memory use.

## Installation

Python 3.10 or newer is required; CI covers Python 3.10–3.14. pip installs the
runtime dependency `pandas>=2.2.2`. The Git URL installation also requires Git.
Use a virtual environment if your system Python disallows package installation.

Run live queries on a POSIX cluster host with the relevant scheduler/quota clients
on `PATH` and access to their data. pip does not install these external tools;
standard quota auto-detection expects Linux `quota -w -p -ug` support.

```bash
python -m pip install "git+https://github.com/kfuku52/kfbatch"
kfbatch --version
```

The module entry point is equivalent:

```bash
python -m kfbatch -h
```

## Quick start

The CLI has separate batch and disk-quota commands:

```bash
kfbatch batch
kfbatch quota
```

`batch` auto-detects the locally available scheduler command. It prefers Slurm's
`squeue`, falls back to `qstat -F` for AGE/UGE/SGE installations such as SHIROKANE,
and retains the `squeue` default when neither command is available so the missing
required command is reported clearly.

Bare `kfbatch` remains a permanent compatibility alias for `kfbatch batch`, so
existing invocations continue to work:

```bash
kfbatch
kfbatch --stat_command "qstat -F"
```

The explicit AGE/UGE/SGE form is:

```bash
kfbatch batch --stat_command "qstat -F"
```

For a wrapper command whose executable name does not reveal the scheduler, specify
it explicitly:

```bash
kfbatch batch --scheduler uge --stat_command "ssh cluster qstat -F"
```

The AGE defaults use `qstat -u '*'` and the site-specific `qfree -c`. Disable
`qfree` at sites where it is unavailable:

```bash
kfbatch batch --stat_command "qstat -F" --uge_qfree_command ""
```

## Personal, group, and cluster jobs

`batch` defaults to `--scope overview`, which shows personal and cluster totals
plus group totals when group identity can be established safely:

```text
jobs  self:R/Q/X/O=0/1/0/0  all:R/Q/X/O=2/3/0/0
jobs  group[account_a]:R/Q/X/O=1/2/0/0
```

Use an individual scope or request a per-user group breakdown:

```bash
kfbatch batch --scope self
kfbatch batch --scope group --by-user
kfbatch batch --scope all
```

For Slurm, group identity is an account association reported by `sshare`; jobs are
selected by the account field from `squeue`. For AGE/UGE, `qfree` supplies the group
name and member list. If those sources are unavailable, kfbatch labels the group
view unavailable instead of guessing from currently visible jobs. For Slurm, an
administrator or remote-wrapper user can select a known account explicitly:

```bash
kfbatch batch --scope group --group-id account_a
```

When a user has several Slurm account associations, each account is reported
separately. For AGE/UGE, `--group-id` cannot replace missing `qfree` membership
data; it must agree with the discovered group. With only a qfree running total,
queued and failed group slots are shown as `?`.

Slurm totals count array tasks, whereas AGE/UGE totals count slots (slots per task
multiplied by array-task count). These are current snapshots, not job history.
`--scope` and `--by-user` affect printed job summaries, not the resource table or
TSV rows.

## Disk quota

`quota` normalizes space and file/inode values and clearly separates personal
usage from limits shared by a group:

```text
scope  owner    filesystem    space(used/soft/hard)   files(used/soft/hard)  grace  provider
self   user_a   home_user_a   3.5TiB/-/5.0TiB         420,000/-/500,000      -      custom
group  group_a  home_group_a  71.2TiB/90.0TiB/100TiB  8,200,000/...          3days  custom
```

Common forms are:

```bash
# Auto-detect lfsq first, then standard quota.
kfbatch quota

# Restrict the report.
kfbatch quota --scope self --filesystem home
kfbatch quota --scope group --group-id group_a

# Parse standard Lustre quota output from a site-specific query.
kfbatch quota --provider lustre \
  --quota-command "lfs quota -u user_a home_user_a"
```

On SHIROKANE, `lfsq` must be run after `qlogin`. kfbatch parses its `Gbytes` and
`kfiles` columns with their reported scales, reports the `qlogin` requirement only
when the command itself fails, and never starts an interactive session automatically.
The ordinary `lfs quota` command can instead be supplied with `--quota-command` on
a login node.

Standard `quota`/`lfs quota` tables are parsed directly, including empty grace
columns and scaled file counts. The default Linux quota query uses unrounded
values and raw grace expiry timestamps (Unix seconds; zero means no grace).
Valid POSIX quota output is retained on status 1 with a diagnostic, since quota
uses nonzero status to report exceeded limits. A custom site wrapper can
emit a whitespace-, tab-, or pipe-separated table whose required columns are
`scope owner filesystem bytes_used`; optional columns are `bytes_soft`,
`bytes_hard`, `files_used`, `files_soft`, `files_hard`, and `grace`. Unitless space
values are KiB; binary suffixes from KiB through PiB are supported. Despite the
`bytes_*` column names, unitless wrapper values are also KiB, not bytes. See the
[usage reference](docs/usage-reference.md) for missing values, filters, and safe
offline examples.

## Output

Slurm uses one row per partition and reports task totals as running, queued,
terminal/error, and other/unknown states (`R/Q/X/O`):

```text
jobs  self:R/Q/X/O=0/0/0/0  all:R/Q/X/O=4/2/0/0

fairshare  self=0.500000  account=account_a  assoc_rank=2/3  pending_assoc_rank=n/a/1

part   nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU               topRAM               launch
epyc   2/0/2  48/8/64     144/192      compute02 32c/64GiB  compute01 16c/80GiB  res<=32c/64GiB
short  0/1/1  0/0/16      ?/32         -                    -                    res<=0c/0GiB

legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total,
        launch=res=CPU/RAM-only ceiling
```

`assoc_rank` ranks user/account associations, not unique people. If the current
user has multiple associations and no queued job identifies one unambiguously,
the output explicitly says that the highest-FairShare association was selected.

AGE/UGE/SGE uses the same compact layout with slot totals (`R/Q/F`) and adds
`qfree` quota data when available:

```text
jobs  self:R/Q/F=0/0/0  all:R/Q/F=4/248/5

queue    nodes  cpu(a/u/t)  ram(a/t)GiB  topCPU               topRAM  quota(s/g/l)  launch2G
mjobs.q  1/0/1  12/4/16     384/512      compute01 12c/24GiB  same    4/8/128       24(+16s)
intr.q   1/0/1  7/1/8       30/32        compute02 7c/20GiB   same    0/0/24        7

legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total
        ram uses qfree request headroom/capacity; topRAM is queue-instance request headroom
        quota=self/group/limit slots (inf=unlimited), launch2G=immediate slots (+standby)
```

`?` means that the scheduler did not provide a trustworthy value. Slurm memory
suffixes are binary when their unit is explicit; a unitless `squeue %m` request is
kept ambiguous. Grid Engine follows its documented convention: lowercase suffixes
are decimal and uppercase suffixes are binary. Values are displayed in GiB, and
available memory is floored so the display never rounds launch capacity upward.

## Useful options

```bash
# Merge three Grid Engine snapshots using the minimum trustworthy availability.
kfbatch batch --stat_command "qstat -F" --niter 3

# Show two best nodes for CPU and RAM; include ties with the second node.
kfbatch batch --ntop 2 --all_tiers yes

# Limit each scheduler command to 15 seconds (default: 60).
kfbatch batch --command_timeout 15

# Keep the compact Slurm table but display "-" for launch and skip sprio.
kfbatch batch --show_launch_heuristic no

# Omit the Slurm association FairShare line.
kfbatch batch --show_fairshare_rank no
```

`--command_timeout 0` disables individual command timeouts; Grid Engine resource
sampling still has a 300-second budget. `--niter` accepts 1–100 and applies only
to Grid Engine resource snapshots. Other commands are not repeated.

Run entirely from the repository root using its synthetic fixtures (these paths
are not installed by the wheel):

```bash
kfbatch batch \
  --example_file tests/fixtures/slurm/squeue_full.txt \
  --stat_command squeue \
  --slurm_node_example_file tests/fixtures/slurm/nodes.txt \
  --slurm_partition_example_file tests/fixtures/slurm/partitions.txt \
  --slurm_reservation_example_file tests/fixtures/slurm/reservations.txt \
  --slurm_prio_example_file tests/fixtures/slurm/sprio.txt \
  --slurm_share_example_file tests/fixtures/slurm/sshare.txt \
  --current_user current_user
```

## TSV output

Node and job tables have separate output paths:

```bash
kfbatch --out_nodes nodes.tsv --out_jobs jobs.tsv
```

Each file is written atomically; the pair is not a transaction. Both paths must
resolve to different files. `--out` remains a legacy alias of `--out_nodes`. If Slurm node discovery fails, the job TSV can still
be written, the node TSV is not created or updated, and the command returns a
non-zero status. Existing outputs are replaced on successful writes, not appended;
parent directories must already exist. No files are saved unless output paths
are supplied. Relative paths use the current working directory.

See [TSV columns and units](docs/usage-reference.md#tsv-columns-and-units) before
interpreting the files: node rows are queue/partition instances, memory is in GiB,
and job `total_slots` has scheduler-specific semantics. Diagnostic notes and
parse-quality metadata are not included in TSVs.

## Accuracy and failure behavior

- AGE states such as `Rq` and `hRq` are queued, while `Eqw` is failed.
- Text `qstat -u '*'` preserves pending array ranges. If a custom JSON command omits
  array metadata, affected task counts are labeled as estimated.
- With `--niter`, a queue instance missing from any snapshot is unavailable. Any
  abnormal state or unknown memory seen during the sampling window is retained.
  A completely unparseable later snapshot is an error.
- The AGE queue table is the union of `qstat` and `qfree` queues, so quota-only and
  resource-only queues remain visible.
- Unknown Slurm partition metadata is abnormal, never implicitly `UP`.
- Unknown Slurm job states remain visible in the `O` total instead of disappearing.
- Rejected Slurm job rows fail the report; incomplete all-user Grid Engine output
  is labeled degraded and cannot be presented as complete cluster totals.
- Missing Slurm account fields make group totals unavailable, not zero.
- Active Slurm reservations inaccessible to the current user are subtracted across
  every partition alias of the physical node. Access checks combine configured
  user, group, account, QOS, and partition restrictions. Unresolved or unavailable
  reservation metadata suppresses resource ceilings.
- Slurm `launch` is only a single-node CPU/RAM ceiling. It is not an immediate-start
  prediction and does not model every scheduling constraint.
- Required job/resource command failures return non-zero. Optional command failures
  print a `note: degraded ...` explanation.
- Scheduler commands have bounded stdout, stderr, individual line lengths, and
  execution time. Timeout cleanup includes descendant processes.
- Scheduler output is decoded defensively, so invalid UTF-8 in a job name does not
  crash the report.

## Development

All checked-in scheduler samples are small synthetic fixtures. Do not add live
scheduler captures.

See [CONTRIBUTING.md](CONTRIBUTING.md) for environment setup, the offline smoke
command, change-specific tests, and delivery checks. Agent guidance starts at
[AGENTS.md](AGENTS.md). See also [SECURITY.md](SECURITY.md) and
[CHANGELOG.md](CHANGELOG.md).

## License

This program is MIT-licensed. See [LICENSE](LICENSE) for details.
