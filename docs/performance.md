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

## 0.4.15 follow-up audit

Baseline: `b291889463dc240acd0aaab9b19c0b3371a11aef` (0.4.14). Measured on
2026-10-15 with Python 3.14.7, pandas 3.0.6, macOS 27.0.1 arm64. Inputs are
synthetic and constructed before measurement. Each operation is warmed once,
then measured five times; peak Python allocation is a separate traced invocation.
All 14 before/after checksums matched. Another 265 exact comparisons covered
values, dtypes, indices, attrs and printed text, including malformed and missing
data, huge integers, reservation aliases, and degraded Grid Engine fallbacks.

```bash
python -m benchmarks.benchmark_hotpaths
python -m benchmarks.benchmark_hotpaths --case capture-4MiB --repeats 5 --skip-memory

kfbatch_hotpath_base=$(mktemp -d)
git archive b291889463dc240acd0aaab9b19c0b3371a11aef | tar -x -C "$kfbatch_hotpath_base"
cp benchmarks/benchmark_hotpaths.py "$kfbatch_hotpath_base/benchmarks/"
(
  cd "$kfbatch_hotpath_base"
  python -m benchmarks.benchmark_hotpaths
)
```

The harness reuses the report benchmark's timing and checksum implementation.
Compare the same arguments and environment across revisions. Medians below are
seconds; allocation excludes prebuilt inputs and is not process RSS. Baseline
and updated runs were consecutive on a shared host, without affinity or frequency
controls. CPU time describes parser work; wall time is the useful measure for
pipe readiness. Child-process CPU and allocations are outside the capture metric.

| Operation | Before → after CPU seconds | CPU ratio | Before → after wall seconds | Before → after peak MiB |
| --- | --- | --- | --- | --- |
| Slurm render, 50,000 queue/node rows | 0.5571 → 0.1077 | 5.2× | 0.5876 → 0.1081 | 9.84 → 4.13 |
| Slurm launch, same rows | 0.3271 → 0.1201 | 2.7× | 0.3312 → 0.1217 | 8.00 → 4.26 |
| 5,000 explicit reservations | 0.0895 → 0.0192 | 4.7× | 0.0937 → 0.0196 | 2.59 → 2.60 |
| Wildcard over 5,000 nodes | 0.0357 → 0.0122 | 2.9× | 0.0364 → 0.0122 | 2.48 → 2.48 |
| 20,000 sshare associations | 0.2970 → 0.1680 | 1.8× | 0.3120 → 0.1730 | 12.76 → 11.88 |
| 100,000 repeated memory tokens | 0.0878 → 0.0091 | 9.6× | 0.0880 → 0.0092 | 7.06 → 4.77 |
| 100,000 unique memory tokens | 0.0909 → 0.0950 | — | 0.0973 → 0.0990 | 7.06 → 7.06 |
| 100,000 pending requests | 0.3841 → 0.1662 | 2.3× | 0.3859 → 0.2013 | 49.77 → 49.78 |
| 50,000 Grid Engine JSON jobs | 0.1437 → 0.1278 | 1.1× | 0.1438 → 0.1384 | 34.18 → 34.18 |
| 5,000 qfree group users | 0.1833 → 0.0044 | 41.5× | 0.1847 → 0.0044 | 0.59 → 1.09 |
| Decode 16MiB of LF output | 0.0194 → 0.0085 | 2.3× | 0.0196 → 0.0098 | 33.39 → 16.76 |
| Grid Engine 5,000 instances × 120 resources | 0.3620 → 0.1227 | 2.9× | 0.3668 → 0.1257 | 3.63 → 3.63 |
| Capture small child output | 0.0109 → 0.0095 | — | 0.0575 → 0.0338 | 0.34 → 0.34 |
| Capture 4MiB child output | 0.0206 → 0.0168 | — | 0.3213 → 0.0621 | 12.00 → 12.00 |

The node workload has 25,000 physical nodes, each in two distinct partitions
distributed across five queues, retaining the topology attrs emitted by the real
parser. Rendering and resource ceilings discard attrs only on private shallow
working copies; original topology remains available for reservation accounting.
This removes repeated deep copies by pandas. Reservations use per-row plain
records and reuse wildcard targets without retaining a second full record list.

Grid Engine parses embedded jobs only when the all-user query is absent or
degraded, and validates JSON required fields once per job. qfree's seen-user set
retains first-seen ordering at a measured cost of about 0.5MiB for 5,000 users.

Repeated scalar conversions use caches of at most 256 entries, scoped to a single
call. Memory/time columns and pending requests sample up to 128 entries across
the input and keep small, mostly unique, or unsuitable samples on the direct
path. Sampling is a heuristic, so unusual distributions can still cause cache
misses. Unique memory timings overlap (CPU ranges 0.0882–0.1001 before and
0.0894–0.0970 after); no improvement is claimed there. Small command-output wall
ranges also overlap (0.0373–0.0615 before and 0.0311–0.0533 after).

sshare retains scalar numeric conversion before its existing final column
conversion. Bulk conversion of mixed floating-point values and huge integers
was rejected because it changes rounding. Slurm request-memory conversion reuses
its parsed match instead of running the same regular expression twice.

POSIX capture waits for readable pipes, retaining output/line limits, deadlines,
process-group termination and bounded cleanup for detached descendants. The
non-POSIX capture path is retained; these measurements establish no Windows
performance result. LF decoding streams binary lines; CR/CRLF uses the existing
split semantics. Live schedulers and site-specific quota providers were not used.

CI retains Python 3.10–3.14, minimum pandas, coverage, security, package-install
and PR merge-ref checks. A push to a branch with an open PR at the same head
uses that PR's validation; default-branch pushes and other branches still run.
Routing is read-only and fails on API errors. A concurrent PR opening can still
race routing and run both validations. Python 3.12's full suite runs once with
coverage, and tag releases reuse the distributions built and installed by their
own validation run. No workflow wall-time saving is claimed from local tests;
PR routing and tag publication need their respective GitHub events to exercise
the hosted paths.
