# Contributing

Contributions are welcome. Please keep changes small enough to review and add a
regression test for behavior changes.

## Development setup

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=kfbatch --cov-branch
python -m build
```

The supported Python range is 3.10 through 3.14.

## Scheduler fixtures

Never commit live scheduler output. It can expose user names, project/account
names, host names, IP addresses, job names, and resource usage.

Create the smallest synthetic fixture that reproduces the parser case:

- use identities such as `current_user`, `other_user`, and `account_a`;
- use hosts such as `node01` and `compute02`;
- remove free-form job text that is irrelevant to the test;
- keep fixtures under `tests/fixtures/age` or `tests/fixtures/slurm`; and
- run `python -m pytest tests/test_fixture_privacy.py`.

## Pull requests

Explain the scheduler/version tested, the user-visible change, and any conservative
fallback introduced for missing metadata. Do not weaken unknown-data handling merely
to make a sample output look more complete.
