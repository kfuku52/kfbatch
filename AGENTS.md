<!-- BEGIN KF AGENT POLICY: source=https://github.com/kfuku52/kf-agent-policy; version=10; sha256=82e3c0eb467582a414d9a6b2feaaaf6f5c8ae330d30f2e3efbf8c303155d0e2e -->
# Common agent policy

Repository-specific instructions override these defaults.

- Follow the user's task scope within higher-priority instructions and execution
  permissions. Complete implementation through affected verification and a result
  report; a plan or investigation ends with its requested deliverable. Continue
  authorized work without repeated approval; identify actual blocking boundaries.
- Inspect the worktree and preserve unrelated changes. Refresh remote information
  when needed; do not merge, rebase, or switch branches merely to inspect it.
- Prefer the default branch when starting work without an established branch.
  Preserve an existing task branch; follow explicit user branch instructions.
  Never create or switch branches solely for a commit, push, release, or PR.
- Change or recommend branch protection only when explicitly asked. Honor explicit
  repository-specific direct-push exceptions; otherwise report a rejected push
  without bypassing protection or inventing a branch or PR.
- Unpublished implementation details may be redesigned; preserve existing public
  APIs, file formats, and saved-data compatibility unless a breaking change is
  authorized. Update affected producers, consumers, tests, examples, and docs.
- Fix verified root causes; do not hide failures with fallbacks or weaker checks.
  Document unavoidable workarounds and their removal conditions.
- Read relevant docs and run the repository's check entrypoint for the change and
  phase. Verify affected behavior; report checks run and omitted. Repeat or broaden
  successful checks only for new changes, failures, or unresolved concerns.
- For library metadata, require demonstrated incompatibility for exact pins or
  upper bounds; keep reproducibility locks separate.
- When editing READMEs, keep them concise with useful visuals inline; put extended
  guides in linked documentation.
- For GitHub push/release work, use `prepare-github-push` in `.agents/skills/`.
  Local-only commits need no version bump; GitHub pushes require one.
- For software performance work, use `benchmark-performance` in `.agents/skills/`.
  Performance claims require comparable measurements and equivalent output.
- For GitHub Actions edits, use `optimize-github-actions` in `.agents/skills/`.
  Preserve required coverage; never run untrusted PR code on self-hosted runners.
<!-- END KF AGENT POLICY -->

# Working in kfbatch

- Start with the README's usage, output, and accuracy sections, then
  [CONTRIBUTING.md](CONTRIBUTING.md) for setup and verification. Check the worktree
  before edits. Commands run from the repository root in its activated development
  environment; `pyproject.toml` is the tool/dependency configuration.
- Entry points: `kfbatch/cli.py` dispatches batch to `stat.py` and quota to
  `quota.py`; scheduler parsers and rendering are separated as described in
  CONTRIBUTING. `command.py` owns subprocess bounds and failure handling.
- Use CONTRIBUTING's **Local verification** table to select tests, and its
  **Delivery checks** before push. It contains the exact pytest, lint, type,
  security, and build commands; do not invent a second check runner. For reusable
  fixture-based validation use
  [.agents/skills/verify-kfbatch-change/SKILL.md](.agents/skills/verify-kfbatch-change/SKILL.md).
- Preserve CONTRIBUTING's correctness invariants and the README's unit semantics:
  unknown data is not free capacity, additional snapshots cannot increase
  availability, and Slurm launch ceilings are not scheduling predictions. Keep
  site defaults and resource assumptions unchanged unless the task requests them.
- Preserve bare CLI invocation as `batch`, `--out` as the node-output alias,
  separate node/job TSV schemas, and existing imports through `kfbatch.stat`.
- Keep fixtures synthetic. Do not edit/commit live captures, private site settings,
  generated TSVs, coverage output, build artifacts, or local environments. There is
  no checked-in research dataset or separate analysis configuration to tune.
- At completion report changed behavior/files, executed checks and results,
  static-only checks, and environment-dependent omissions. Before publication use
  `prepare-github-push`; follow the existing version/changelog scheme and report
  the destination and commit. Do not create a release tag unless requested.
