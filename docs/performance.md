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


## 2026-10-23 local follow-up audit

Baseline: `d10f6daf8d861bfcd68968f0a6e8d01140a95a31` (0.4.15). Ten
proposals were evaluated by ten separate subagents in isolated synthetic source
copies. Nine were adopted; the key/value parser change was rejected. Two
independent follow-up reviews covered the combined state aggregation and node
rendering changes before integration.

| # | Proposal | Decision and final implementation |
| --- | --- | --- |
| 1 | Avoid per-user job DataFrames | Adopt with #2. Group positional indices once and scan only state, queue and slot columns; retain first-seen user evaluation order and sorted display. |
| 2 | Reuse job-state classification | Adopt. Shared per-report classifier caches at most 128 built-in string states, stops caching a diverse vocabulary, and leaves non-string conversion on the original path. UGE queue assignment changes only an otherwise unclassified state. |
| 3 | Avoid repeated queue scans in rendering | Adopt. Build queue indices once; preserve input queue order, missing values, categorical keys and qfree-only queues. |
| 4 | Avoid sorting every node for a small top list | Adopt. For numeric primary columns above 1,000 rows, select candidates with `nlargest(keep="all")`, then apply the existing secondary/name ordering. Small/object inputs retain their existing sort. |
| 5 | Avoid repeated queue scans in launch ceilings | Adopt. Index queue rows once and filter normal nodes inside each queue; retain node sorting and CPU/RAM-only semantics. |
| 6 | Reuse Slurm request-memory validation | Adopt. A parser-call-local LRU caches up to 256 validation results, including invalid tokens. |
| 7 | Remove redundant Grid Engine job-line checks | Adopt. The numeric-first-character check already excludes empty and separator lines, so it no longer builds a character set for every job. |
| 8 | Replace key/value `split` with `partition` | Reject. Micro CPU improved 8–10%, but ordinary 50,000-node parsing changed from an initial 7.4% gain to a 3.8% regression on confirmation. No reliable overall benefit was established. |
| 9 | Reuse repeated job-array range counts | Adopt for UGE text only, with a parser-call-local LRU of at most 256 expressions. Slurm and UGE JSON paths are unchanged. |
| 10 | Skip unused Grid Engine resource fields earlier | Adopt. Only the exact two memory-field prefixes are split into keys and values; whitespace and malformed keys retain their original treatment. |

Final measurements used Python 3.14.7, pandas 3.1.0, macOS 27.0.1 arm64.
All inputs were synthetic and constructed before timing. Each operation was
warmed once and timed seven times; Python allocation peaks were traced in
separate invocations. Candidate benchmarks ran sequentially after subagent
measurements stopped. The shared host still had substantial external load;
there was no CPU affinity or frequency control. CPU medians are the primary
comparison, and wall medians below do not establish consistent wall-time gains.

Default inputs are 100,000 jobs across 2,000 users, 50,000 queue/node instances
across 100 queues, and 5,000 Grid Engine instances with 120 unused resource
fields each. Physical node aliases use distinct queue keys and matching capacity.
The top-node case uses 50,000 distinct physical nodes in one queue. One-user,
unique-memory, unique-array and non-array cases expose unfavorable distributions.
The node and per-user cases were rerun after final workload/order refinements;
unchanged operations retain their seven-sample measurements.

CPU and wall values are seconds. CPU brackets show the min–max of seven samples.

| Operation | Before CPU [range] | After CPU [range] | CPU ratio | Wall median before → after |
| --- | --- | --- | --- | --- |
| users-slurm | 0.238675 [0.217381–0.248347] | 0.058885 [0.056100–0.065761] | 4.05× | 0.735321 → 0.067551 |
| users-uge | 0.205453 [0.190840–0.224376] | 0.063038 [0.056897–0.069879] | 3.26× | 0.488571 → 0.068278 |
| one-user-slurm | 0.093284 [0.086412–0.108948] | 0.058740 [0.050416–0.065046] | 1.59× | 0.134855 → 0.063571 |
| one-user-uge | 0.077951 [0.072560–0.084941] | 0.058899 [0.053275–0.063204] | 1.32× | 0.171371 → 0.061914 |
| totals-slurm | 0.086068 [0.080906–0.091344] | 0.043759 [0.040283–0.045825] | 1.97× | 0.262397 → 0.060961 |
| totals-uge | 0.068147 [0.063252–0.077016] | 0.044766 [0.042480–0.046881] | 1.52× | 0.130501 → 0.052865 |
| render-slurm | 0.412112 [0.382628–0.419059] | 0.263432 [0.243532–0.301649] | 1.56× | 0.546795 → 1.100500 |
| render-uge | 0.434138 [0.399779–0.464886] | 0.251580 [0.241424–0.298582] | 1.73× | 0.781715 → 0.920466 |
| top-nodes | 0.028832 [0.024740–0.033477] | 0.001907 [0.001756–0.002704] | 15.12× | 0.100983 → 0.001911 |
| launch-queues | 0.498607 [0.462476–0.528334] | 0.229896 [0.199944–0.292050] | 2.17× | 1.241463 → 0.531729 |
| squeue-repeated | 0.350475 [0.335988–0.385858] | 0.256270 [0.252925–0.286800] | 1.37× | 0.726811 → 0.701297 |
| squeue-unique | 0.386453 [0.374661–0.401542] | 0.410633 [0.377697–0.431111] | 0.94× | 0.534739 → 0.517731 |
| uge-array-repeated | 0.241450 [0.232763–0.248691] | 0.135501 [0.126878–0.150987] | 1.78× | 0.345165 → 0.203941 |
| uge-array-unique | 0.238749 [0.235109–0.270594] | 0.176469 [0.172355–0.185420] | 1.35× | 0.361835 → 0.190738 |
| uge-no-array | 0.180999 [0.171169–0.210019] | 0.115415 [0.112285–0.121486] | 1.57× | 0.275319 → 0.131637 |
| qstat-resources | 0.133816 [0.124910–0.142133] | 0.054901 [0.053977–0.068855] | 2.44× | 0.165692 → 0.059609 |

All 16 final before/after checksums matched, including summary text and DataFrame
values, indices, column order, dtypes and attrs. Independent combined reviews
compared 1,468 aggregation cases and 336 full-render cases. An evaluation-order
difference for stateful scalar objects was fixed by retaining grouped positional
iteration; regression tests cover state, queue and slot conversions. Individual
proposal comparisons also covered missing fields, malformed tokens, array
estimates, unknown memory, queue-only rows and reservation topology preservation.

| Operation | Before peak Python allocation | After |
| --- | --- | --- |
| users-slurm | 16.50MiB | 4.71MiB |
| render-slurm | 1.90MiB | 2.29MiB |
| top-nodes | 6.98MiB | 0.78MiB |
| launch-queues | 1.90MiB | 2.29MiB |
| uge-array-repeated | 56.73MiB | 56.73MiB |

Peaks exclude prebuilt inputs and are not process RSS. Queue indices cost about
0.39 MiB for 50,000 rows in these render/launch cases. The unique Slurm-memory
case has a 6.3% higher CPU median, with overlapping sample ranges; the bounded
cache trades miss overhead for the repeated-token improvement. On the final
100-row/5-queue small workload, launch CPU increased from 4.62 to 5.59 ms;
Slurm/UGE aggregate totals increased by about 12/22 microseconds. Small workloads
are not uniformly faster. All-primary-tied top-node
inputs showed little benefit (about 1.4% extra CPU in screening), and
`all_tiers` still formats every selected tie. No improvement is claimed for these
unfavorable cases or for live scheduler response time.

Reproduce from CONTRIBUTING's activated environment at the repository root:

```bash
python -m benchmarks.benchmark_followup --repeats 7 --skip-memory
python -m benchmarks.benchmark_followup --case users-slurm --case render-slurm --case top-nodes --case launch-queues --case uge-array-repeated --repeats 1

kfbatch_followup_base=$(mktemp -d)
git archive d10f6daf8d861bfcd68968f0a6e8d01140a95a31 | tar -x -C "$kfbatch_followup_base"
cp benchmarks/benchmark_followup.py "$kfbatch_followup_base/benchmarks/"
(
  cd "$kfbatch_followup_base"
  python -m benchmarks.benchmark_followup --repeats 7 --skip-memory
)
```

Use identical arguments and the same interpreter/dependencies at both revisions;
compare checksums as well as timings. For small inputs, use `--jobs 100 --users 5
--nodes 100 --queues 5 --qstat-nodes 10`. Fixtures, site defaults, units, CLI/TSV
schemas, scheduler queries and command-capture safeguards were not changed.
This audit makes no CI-runtime claim. The full interpreter/minimum-pandas matrix,
live schedulers, SSH wrappers and site-specific quota providers were not exercised.

Local verification: `python -m pytest --cov=kfbatch --cov-branch` passed all 386
tests with 87.40% branch-inclusive coverage. Static checks (`ruff check`,
`ruff format --check`, `mypy`, and `bandit`) passed. The small-input benchmark
also matched all 16 before/after checksums. Package builds and network dependency
audits were outside this local-only change; no version bump or publication was
performed.


## 2026-10-23 second local follow-up

Baseline: `05ae68e5c81b709b74bfa4d2f52f42437cbb2ef8`. Six further proposals
were screened in separate subagent source copies; five were adopted. No scheduler
commands, units, output schemas, or site defaults were changed.

| # | Proposal | Decision |
| --- | --- | --- |
| 1 | Avoid a pandas Series for every displayed top-node row | Adopt. At least ten selected rows with known numeric CPU counts use a common-dtype NumPy row array. Small selections, object/string CPU counts and missing CPU counts retain `iterrows`, preserving its row-type inference. |
| 2 | Reduce launch-ceiling sorting and RAM parsing | Adopt. Numeric queues with at least 256 normal rows first retain maximum-CPU candidates, then use the existing RAM conversion and three-key sort. All-tied, unknown-CPU and nonnumeric cases keep the original ordering behavior. |
| 3 | Reuse repeated partition-field splitting | Adopt. Use the existing sampled, bounded per-call string mapper with an explicit built-in-string mode. Repeated lists are reused before exploding queue rows; output rows and association duplicates remain unchanged. |
| 4 | Precompile the Slurm header regex | Reject. Initial isolated gains shrank to 0.98% for repeated-memory jobs and 0.83% for unique-memory jobs in five alternating comparisons. The small difference did not establish a reliable overall benefit. |
| 5 | Precompile the unchanged memory regex | Adopt. Use the identical expression with `Pattern.fullmatch`; the measurable benefit is unique-token memory-column conversion. Full squeue parsing and repeated-token columns showed no consistent improvement in the final sweep. |
| 6 | Reuse repeated Slurm state normalization | Adopt. Apply the mapper at the job-summary, FairShare and launch entry points. Only built-in strings enter the new cache, preserving non-string/string-subclass conversion behavior. |

The mapper retains its previous default behavior for existing memory/time
callers. Its new `exact_strings=True` mode is explicit, per call and limited to
256 entries. Samples with many distinct values or unsuitable types keep direct
conversion; no global cache was introduced. Array-row rendering preserves the
common dtype of all selected columns instead of changing numeric coercion by
selecting only three columns.

Measured with Python 3.14.7, pandas 3.1.0 and macOS 27.0.1 arm64. Inputs were
constructed before measurement, with one warmup and seven repetitions per
operation. Baseline and integrated timing suites ran sequentially after subagent
work stopped. Allocation tracing used separate invocations. The host was still
shared, without CPU affinity or frequency control; compare CPU ranges and avoid
interpreting the wall medians as guaranteed CLI latency improvements.

The top-node cases use 50,000 distinct nodes, selecting all primary ties or
25,000 rows. Launch cases use either 50,000 node instances over one/100 queues,
or 100,000 priority-pending jobs over eight node instances/four queues. The
multi-partition case places each job in four candidate queues. Summary and
memory-column cases use 100,000 rows. Existing synthetic node generators retain
physical-alias consistency and distinct queue/node keys.

CPU and wall values are seconds; CPU brackets show min–max of seven samples.

| Operation | Before CPU [range] | After CPU [range] | CPU ratio | Wall median before → after |
| --- | --- | --- | --- | --- |
| top-ties | 0.678192 [0.625174–0.810793] | 0.059477 [0.056738–0.070812] | 11.40× | 1.880191 → 0.074824 |
| top-many | 0.390950 [0.351324–0.423213] | 0.043728 [0.038544–0.054366] | 8.94× | 0.698162 → 0.074988 |
| top-nodes | 0.002194 [0.001891–0.002866] | 0.002410 [0.002173–0.003865] | 0.91× | 0.002222 → 0.002416 |
| launch-single | 0.056683 [0.052851–0.064743] | 0.018489 [0.017926–0.020311] | 3.07× | 0.082843 → 0.020147 |
| launch-queues | 0.239212 [0.223100–0.249707] | 0.125679 [0.102742–0.129949] | 1.90× | 0.881018 → 0.200525 |
| launch-jobs | 0.280943 [0.275349–0.303924] | 0.250193 [0.235464–0.263859] | 1.12× | 0.397539 → 0.679505 |
| launch-jobs-multi | 1.013460 [0.960166–1.062230] | 0.807533 [0.774624–0.964370] | 1.26× | 1.856221 → 1.502717 |
| slurm-summary | 0.075076 [0.062616–0.083920] | 0.024787 [0.023569–0.026134] | 3.03× | 0.115802 → 0.032122 |
| slurm-summary-unique | 0.079592 [0.076481–0.088062] | 0.069922 [0.066903–0.085162] | 1.14× | 0.164279 → 0.085184 |
| squeue-repeated | 0.273013 [0.252164–0.296412] | 0.266316 [0.257810–0.272715] | 1.03× | 0.624666 → 0.326202 |
| squeue-unique | 0.408161 [0.398763–0.468396] | 0.411861 [0.362578–0.443301] | 0.99× | 0.842776 → 1.025526 |
| memory-unique | 0.108300 [0.102698–0.118989] | 0.097763 [0.088868–0.105609] | 1.11× | 0.146377 → 0.381825 |
| memory-repeated | 0.018295 [0.017343–0.021766] | 0.019023 [0.017418–0.021462] | 0.96× | 0.019603 → 0.027643 |
| memory-ge-unique | 0.112871 [0.098852–0.121997] | 0.089912 [0.082626–0.095292] | 1.26× | 0.270152 → 0.197957 |

| Operation | Before peak Python allocation | After |
| --- | --- | --- |
| top-ties | 18.70MiB | 18.69MiB |
| launch-single | 12.84MiB | 7.39MiB |
| launch-jobs-multi | 142.92MiB | 120.12MiB |
| slurm-summary | 6.80MiB | 4.78MiB |
| memory-ge-unique | 7.83MiB | 7.83MiB |

Peaks exclude prebuilt inputs and are not process RSS. The additional working
sets for partition lists and state normalization shrink; rendering still needs
the selected output text. Ordinary top1 and repeated-memory timings have
overlapping ranges and no improvement is claimed. Unique-state inputs retain
direct normalization, so their apparent timing gain is not attributed to caching.
The full unique-memory squeue case was effectively unchanged; the earlier audit's
unique-memory overhead is not claimed to have been eliminated.

All 14 large-workload and 26 small-workload output checksums matched before/after,
including DataFrame values, column/index order, dtypes and attrs. Individual
checks included 12,160 row-rendering comparisons, 1,252 launch comparisons and
166,882 memory-conversion comparisons. An independent combined review matched
192 full stdout cases, 33 launch DataFrames and three existing exception paths,
with 460 input/attrs nonmutation checks per implementation. Regression tests
cover numeric coercion, missing values, name/queue ordering and late string
subclasses with conversion side effects.

Reproduce selected final workloads in CONTRIBUTING's activated environment:

```bash
python -m benchmarks.benchmark_followup --case top-ties --case top-many --case launch-single --case launch-jobs-multi --case slurm-summary --case memory-ge-unique --repeats 7 --skip-memory
python -m benchmarks.benchmark_followup --case top-ties --case launch-single --case launch-jobs-multi --case slurm-summary --case memory-ge-unique --repeats 1

kfbatch_round2_base=$(mktemp -d)
git archive 05ae68e5c81b709b74bfa4d2f52f42437cbb2ef8 | tar -x -C "$kfbatch_round2_base"
cp benchmarks/benchmark_followup.py "$kfbatch_round2_base/benchmarks/"
(
  cd "$kfbatch_round2_base"
  python -m benchmarks.benchmark_followup --case top-ties --case top-many --case launch-single --case launch-jobs-multi --case slurm-summary --case memory-ge-unique --repeats 7 --skip-memory
)
```

Select the other recorded operations with repeated `--case` options. Small-input
checks used `--jobs 100 --users 5 --nodes 100 --queues 5 --qstat-nodes 10`.
Use identical arguments, interpreter and dependency versions at both revisions.
Full Python/minimum-pandas CI, live scheduler/SSH/site quota checks, package builds
and network dependency audits were not run for this local-only change.

Verification: the fixture smoke selection passed 42 tests; the full
`python -m pytest --cov=kfbatch --cov-branch` run passed 417 tests with 87.73%
branch-inclusive coverage. Static checks (`ruff check`, `ruff format --check`,
`mypy`, and `bandit`) and `git diff --check` passed. This remains a local commit
with no version bump or publication.
