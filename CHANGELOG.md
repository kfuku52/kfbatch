# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-07-31

### Added

- Add explicit `batch` and `quota` subcommands while preserving bare `kfbatch`
  and its existing options as the batch compatibility interface.
- Add `overview`, `self`, `group`, and `all` batch scopes plus optional per-user
  group breakdowns.
- Discover Slurm groups from `sshare` account associations and AGE/UGE group
  members from `qfree`, with an explicit `--group-id` override.
- Add provider-independent personal/group disk-quota records, standard
  `quota`/`lfs quota` and normalized wrapper parsers, `lfsq`/POSIX auto-detection,
  captured fixture support, and filesystem/owner filtering.

### Changed

- Show group job totals in the default overview when authoritative membership
  data is available, and label the view unavailable instead of inferring it from
  incomplete job data.
- Keep shared group disk usage and limits visually distinct from personal usage.
- Retain Slurm association rows without a FairShare value for group discovery
  while excluding them from FairShare ranks.

### Security

- Never start an interactive `qlogin` automatically; failed SHIROKANE `lfsq`
  queries provide an actionable retry message instead.

## [0.3.0] - 2026-07-31

### Added

- Add an explicit `--current_user` override for remote schedulers and fixture runs.
- Add structured command failures, bounded stdout/stderr and line handling, process
  group termination, and regression coverage for hostile or malformed scheduler
  output.
- Add type checking, wheel-content and metadata validation, subprocess coverage,
  installed-wheel fixture smoke tests, and a tag-driven release workflow with
  checksums, an SBOM, and build provenance attestations.

### Changed

- Require pandas 2.2.2 or newer and test the real minimum without pinning NumPy
  below version 2.
- Interpret Grid Engine lowercase memory suffixes as decimal and uppercase suffixes
  as binary; treat unitless Slurm `%m` request memory as ambiguous rather than GiB.
- Classify all known Slurm job and node states conservatively, preserve `R/Q/X/O`
  totals, and suppress launch ceilings whenever capacity or reservation metadata
  cannot be resolved safely.
- Bound Grid Engine sampling to 100 snapshots and 300 seconds, make repeated
  reservation adjustment idempotent, and pre-index launch heuristic inputs.
- Preserve existing TSV permissions during atomic replacement.

### Fixed

- Reject non-regular fixture inputs without blocking, cap scheduler output without
  deadlocking on full pipes, terminate inherited descendants on timeout, sanitize
  `SQUEUE_*` environment overrides, and avoid exposing command arguments in errors.
- Reject non-empty unrecognized scheduler output and invalid numeric/time fields
  instead of silently reporting zero or free resources.
- Resolve the effective local user independently of spoofable environment variables
  and label FairShare account selection consistently.
- Include every supported Slurm state in job summaries and retain the requested
  number of top rows when the primary metric is unknown.

### Security

- Expand fixture privacy validation to emails, IPv4/IPv6 addresses, absolute paths,
  scheduler identities, and recursively discovered capture-like files.
- Document a private email fallback for vulnerability reports.

## [0.2.1] - 2026-07-30

### Changed

- Parse AGE queue instances and Slurm node blocks in a single pass and retain only
  the documented fixed node schema.
- Build job and node tables from fixed-width tuple records, reuse repeated
  scheduler labels, and replace hot-path array-expression regular expressions
  with direct numeric parsing.
- Spool large command output to disk after 1 MiB and decode it incrementally,
  reducing simultaneous raw-byte and decoded-text retention.
- Add a reproducible synthetic benchmark for large AGE and Slurm parser inputs.

## [0.2.0] - 2026-07-30

### Added

- Compact AGE/UGE/SGE output aligned with the Slurm partition table.
- SHIROKANE AGE and `qfree -c` parsing support.
- Slurm association FairShare and pending-association ranks.
- Explicit scheduler override, module entry point, and `--version`.
- Atomic and schema-separated node/job TSV outputs.
- Synthetic AGE and Slurm regression fixtures with privacy enforcement.

### Changed

- Treat missing partition, reservation, memory, and multi-snapshot data
  conservatively instead of reporting it as available capacity.
- Apply Slurm reservations across every partition alias of a physical node and
  combine user, group, account, QOS, and partition access restrictions.
- Use an explicit stable `sprio` format and label launch values as CPU/RAM-only
  ceilings rather than immediate-start predictions.
- Interpret scheduler memory suffixes as binary units and display GiB.
- Validate positive iteration/display counts and finite command timeouts.

### Fixed

- AGE array-task and modern job-row parsing.
- AGE queue visibility when `qstat` and `qfree` report different queue sets.
- Slurm state classification, hostlist expansion, reservation wildcards, and
  multi-account FairShare labeling.
- Non-zero failure behavior when required Slurm node data is unavailable.

### Security

- Removed live scheduler captures from the maintained tree and added fixture
  checks that reject non-synthetic identities and private network addresses.

[0.4.0]: https://github.com/kfuku52/kfbatch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/kfuku52/kfbatch/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/kfuku52/kfbatch/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kfuku52/kfbatch/releases/tag/v0.2.0
