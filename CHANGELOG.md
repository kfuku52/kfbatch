# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.2.1]: https://github.com/kfuku52/kfbatch/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/kfuku52/kfbatch/releases/tag/v0.2.0
