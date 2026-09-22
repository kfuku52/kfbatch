# Contributing

Contributions are welcome. Please keep changes small enough to review and add a
regression test for behavior changes.

## Development setup

Run the commands below from the repository root in a POSIX shell. Use Python
3.10–3.14 (`python --version`); an OS-provided `python3` may be older. There is no
separate environment file or task runner: `pyproject.toml` defines dependencies
and all check-tool settings.

Create an isolated environment outside the checkout (the fixture privacy check
also scans untracked files inside the checkout):

```bash
python -m venv "${TMPDIR:-/tmp}/kfbatch-dev-venv"
. "${TMPDIR:-/tmp}/kfbatch-dev-venv/bin/activate"
python -m pip install -e ".[dev]"
python -m kfbatch --version
```

Installation needs package-index access. Reuse the activated environment for
subsequent commands; do not change dependency versions to repair a local setup.

## Local verification

The existing pytest suite is the verification entrypoint. It uses synthetic
fixtures, mocks, and local subprocesses; no cluster login or large dataset is
required. CLI output files use pytest's temporary directories. The small smoke
selection covers both schedulers, quota, legacy CLI entrypoints, TSV output, and
failure exits:

```bash
python -m pytest -q tests/test_integration_samples.py tests/test_cli.py tests/test_fixture_privacy.py
```

Success means exit status zero with all selected tests passing. A failed or
uncollected test is not a successful smoke run. Choose additional tests by the
changed behavior, using `python -m pytest -q` followed by the paths below:

| Change | Additional test files |
| --- | --- |
| Scheduler parsing, memory/state semantics, reservations, rendering, command execution, or `stat.py` compatibility imports | `tests/test_stat.py tests/test_review_regressions.py` |
| Personal/group/account scope | `tests/test_batch_scope.py tests/test_review_regressions.py` |
| Quota parsing, units, provider exits or filters | `tests/test_quota.py tests/test_review_regressions.py` |
| Workflow configuration | `tests/test_workflows.py` (also inspect the affected CI jobs) |
| CLI examples or options | Compare `python -m kfbatch -h`, `python -m kfbatch batch -h`, and `python -m kfbatch quota -h` with the edited examples |

For docs-only changes, check the affected links and execute new commands/examples;
do not run bare `kfbatch batch` or `quota` as an offline smoke test. Those query the
local scheduler or filesystem. The README's complete Slurm fixture example is
safe offline; supplying only `--example_file` does not replace optional scheduler
queries. Use a temporary directory for any manual `--out_nodes`/`--out_jobs` output.

## Delivery checks

Before a GitHub push, run the following local checks. This is the local delivery
checklist; `.github/workflows/tests.yml` defines the complete CI matrix and package
installation checks. Stop on any nonzero exit and report the failure.

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
python -m bandit -q -c pyproject.toml -r kfbatch
python -m pytest --cov=kfbatch --cov-branch
```

Expected results are clean lint/format/type/security checks, passing tests, and
coverage (including branches) meeting the configured 70% threshold. Packaging also requires
package-index access for build isolation. Keep distributions separate from stale
local builds:

```bash
kfbatch_dist=$(mktemp -d)
python -m build --outdir "$kfbatch_dist"
python -m twine check --strict "$kfbatch_dist"/*
check-wheel-contents "$kfbatch_dist"/*.whl
```

All three commands must succeed. Build may create ignored `build/` and
`*.egg-info/` intermediates in the checkout; these are not source files to edit.
For packaging changes, additionally reproduce CI's source-archive content checks
and wheel/sdist installation checks in fresh environments outside the checkout.

Keep these distinct from offline verification:

- `python -m pip_audit .` queries the vulnerability service and may resolve/download
  dependencies. It is a CI delivery check; run when network access is available,
  and explicitly report an unavailable service rather than treating it as a pass.
- CI runs Python 3.10–3.14, minimum pandas 2.2.2, clean package installs, and CodeQL.
  One local interpreter does not establish that matrix passed.
- `python -m benchmarks.benchmark_parsers` is an opt-in synthetic performance
  workload (up to 100,000 jobs by default), not a routine smoke check. Consult the
  performance skill before benchmarking; no performance claim follows from pytest.
- Live scheduler, SSH wrapper, `qlogin`, and site quota checks require the relevant
  site environment and explicit task scope. Do not collect private captures to
  compensate for a missing environment.

## Scheduler fixtures

Never commit live scheduler output. It can expose user names, project/account
names, host names, IP addresses, job names, and resource usage.

Create the smallest synthetic fixture that reproduces the parser case:

- use identities such as `current_user`, `other_user`, and `account_a`;
- use hosts such as `node01` and `compute02`;
- remove free-form job text that is irrelevant to the test;
- keep fixtures under `tests/fixtures/age`, `tests/fixtures/slurm`, or `tests/fixtures/quota`; and
- run `python -m pytest tests/test_fixture_privacy.py`.

## Pull requests

Explain the scheduler/version tested, the user-visible change, and any conservative
fallback introduced for missing metadata. Do not weaken unknown-data handling merely
to make a sample output look more complete.

## Releases

Update `kfbatch.__version__` and `CHANGELOG.md` in the release commit. Push an
annotated `vX.Y.Z` tag only after the `main` checks pass. The tag workflow verifies
that the tag and package versions match and runs the full reusable validation
workflow before publishing distributions,
SHA-256 checksums, an SBOM, and build-provenance attestations to a GitHub release.

## Correctness invariants

Parser changes must preserve total task counts across exclusive state buckets,
normalize long and short state names identically, and expose rejected rows and
missing fields. Additional snapshots or missing metadata must never increase
available capacity. Reservation uncertainty applies to every partition alias of
an affected physical node. Use synthetic tests for these properties.

Scheduler parsers live in `slurm_parser.py` and `uge_parser.py`, state semantics
in `job_states.py`, rendering in `render.py`, and command orchestration in
`stat.py`. Its existing parser and rendering imports remain compatibility aliases.
