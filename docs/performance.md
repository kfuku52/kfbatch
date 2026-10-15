# Performance audit

The 0.4.14 audit covered scheduler parsing, job/group summaries, FairShare ranking,
resource ceilings, snapshot merging, rendering, command capture, quota parsing,
and CI dependency installation. All workloads use generated synthetic data.

## Changes

- Group aggregation reads the state, queue, and slot columns directly instead of
  creating a pandas Series for every job. Invalid/negative slot handling and
  mutually exclusive scheduler-state buckets are retained.
- Grid Engine job summaries normalize paired state/queue values directly instead
  of resolving two indexed cells for every job.
- Slurm FairShare ranking selects only pending user/account columns, deduplicates
  repeated associations, and indexes ranked associations once. Missing account
  inference still considers associations with unknown FairShare, and ambiguous
  accounts still select the best ranked association with the existing disclosure.
- Grid Engine snapshot merging combines status values in input order and assigns
  the resulting column together. Unknown memory, abnormal states, and missing
  snapshot rows retain their conservative handling.
- Slurm request-memory conversion reads columns directly instead of using
  `DataFrame.apply(axis=1)`. Per-CPU/per-node units, ambiguous memory, and the
  single-node resource ceiling remain unchanged.
- The five Python-version CI jobs install the package, pytest, and PyYAML instead
  of every development tool. The quality/package job retains lint, format, type,
  security, dependency audit, coverage, build, and installation validation.
  Workflow events, permissions, runners, and release validation are unchanged.

The existing parsers already discard irrelevant resource fields and reuse common
text. Command capture bounds stdout, stderr, line lengths, time, and cleanup;
these safeguards were retained. Quota parsing and small display tables were
reviewed without changing their units or introducing caches. Scheduler queries
remain sequential: this audit does not establish that extra concurrent requests
would be safe or faster on a live cluster.

## Reproduction

Use CONTRIBUTING's activated development environment at the repository root:

```bash
python -m benchmarks.benchmark_reports --jobs 100000 --users 2000 --nodes 5000 --repeats 5 --skip-memory
python -m benchmarks.benchmark_reports --jobs 100000 --users 2000 --nodes 5000 --repeats 3
python -m benchmarks.benchmark_parsers
```

To measure the same harness against the pre-change source without switching the
working branch:

```bash
kfbatch_baseline_tree=$(mktemp -d)
git archive dc6d499b3642b48ccfc007f2cf002d3d1e44eed2 | tar -x -C "$kfbatch_baseline_tree"
cp benchmarks/benchmark_reports.py "$kfbatch_baseline_tree/benchmarks/"
(
  cd "$kfbatch_baseline_tree"
  python -m benchmarks.benchmark_reports --jobs 100000 --users 2000 --nodes 5000 --repeats 5 --skip-memory
)
```

The report harness creates inputs before measurement, warms each operation once,
reports repeated wall and process-CPU time samples, and measures peak Python
allocation in a separate invocation with `tracemalloc` unless `--skip-memory` is
selected. CPU time helps distinguish algorithm cost from competing host work;
allocation tracing can be much slower than the operation itself. Checksums cover
summary text, totals, FairShare fields, and DataFrame values, indices, column
order, dtypes, and attrs.
Compare checksums between revisions as well as timing. Peak allocation excludes
the prebuilt inputs and is not process RSS; subprocess and CLI startup time are
outside this workload.

The default inputs are 100,000 pending Slurm jobs across 2,000 associations,
100,000 mixed-state Grid Engine jobs, 5,000 Grid Engine queue instances, and 5,000
Slurm nodes exposed through two partitions. The launch case deliberately uses
100,000 priority-blocked jobs belonging to one user to exercise request scanning.
Smaller sizes can be selected with the same options.

## Measurement record

Measured on 2026-10-15 with Python 3.14.7, pandas 3.0.6, and macOS 27.0.1 arm64.
Baseline: `dc6d499b3642b48ccfc007f2cf002d3d1e44eed2`. All six output checksums
matched before and after. These are local synthetic measurements, not a claim
about scheduler response time or every cluster workload.

CPU seconds below are the median of five samples; brackets show min–max. The
host was shared and wall time varied substantially, so CPU time is the primary
comparison. No process affinity or CPU-frequency controls were applied.

| Operation | Before CPU seconds [range] | After CPU seconds [range] | CPU speedup |
| --- | --- | --- | --- |
| Slurm group totals | 1.970 [1.719–2.253] | 0.072 [0.068–0.075] | 27.4× |
| Grid Engine group totals | 1.842 [1.569–1.983] | 0.066 [0.057–0.072] | 27.9× |
| Grid Engine job summary | 1.241 [1.193–1.282] | 0.065 [0.059–0.072] | 19.0× |
| Slurm FairShare | 2.698 [2.589–2.879] | 0.074 [0.069–0.082] | 36.5× |
| Grid Engine snapshot merge | 0.272 [0.241–0.285] | 0.064 [0.060–0.070] | 4.2× |
| Slurm launch ceiling | 0.894 [0.863–0.924] | 0.586 [0.548–0.617] | 1.5× |

The same samples' wall seconds are retained for transparency. In particular,
overlapping launch-case ranges do not establish a precise wall-time gain.

| Operation | Before wall seconds [range] | After wall seconds [range] |
| --- | --- | --- |
| Slurm group totals | 6.375 [4.570–10.016] | 0.300 [0.197–0.330] |
| Grid Engine group totals | 5.678 [2.465–6.627] | 0.179 [0.057–0.367] |
| Grid Engine job summary | 4.012 [3.228–4.527] | 0.162 [0.077–0.341] |
| Slurm FairShare | 6.990 [6.221–7.824] | 0.199 [0.093–0.268] |
| Grid Engine snapshot merge | 0.925 [0.614–1.519] | 0.189 [0.163–0.410] |
| Slurm launch ceiling | 3.226 [2.487–3.345] | 1.502 [1.200–2.502] |

Separate single-operation allocation runs used the same inputs with tracing
enabled and three untraced timing repetitions. They produced the same checksums:

| Operation | Before peak Python allocation (MiB) | After (MiB) |
| --- | --- | --- |
| Slurm group totals | 16.02 | 0.01 |
| Grid Engine group totals | 16.79 | 0.01 |
| Grid Engine job summary | 6.36 | 6.30 |
| Slurm FairShare | 31.87 | 10.52 |
| Grid Engine snapshot merge | 2.10 | 2.10 |
| Slurm launch ceiling | 71.02 | 66.34 |

Beyond the large-workload checksums, 360 exact comparisons against the baseline
covered 60 deterministic generated cases across aggregation, Grid Engine summary
text, FairShare selection/missing data, snapshot values/dtypes, and launch tables.
Checked-in regression tests cover missing columns, malformed slot values, queue
state precedence, repeated associations, unranked shares, and status ordering.
The existing parser benchmark was also executed as the initial parsing baseline;
its parser implementations were not changed by this audit.
