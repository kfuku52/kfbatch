---
name: verify-kfbatch-change
description: Select and run kfbatch's synthetic fixture checks after CLI, parser, quota, output, or usage-documentation changes. Use for local correctness verification, not performance benchmarking or live cluster validation.
---

# Verify a kfbatch change

Inputs: the diff or affected behavior, an available Python 3.10–3.14 interpreter,
and whether the task includes publication. Run from the repository root.

1. Read [CONTRIBUTING.md](../../../CONTRIBUTING.md), especially Local verification
   and Correctness invariants. Inspect the changed code and the corresponding
   README contract. Identify which scheduler/provider, output schema, or failure
   behavior could change; do not infer test coverage just from module names.
2. Check `python --version` and use the documented development environment. Run
   the existing smoke selection, then the additional files from CONTRIBUTING's
   change table. For a behavior fix, reproduce it with the smallest synthetic
   case and assert an observable count, schema, diagnostic, or exit status.
3. For CLI documentation, compare all affected help screens and run the example
   with complete fixtures. The README has a complete Slurm example; the integration
   tests have AGE and quota examples. Keep the scheduler explicit and substitute
   synthetic identity arguments. A job fixture alone does not replace node,
   reservation, priority, share, or qfree queries. Use pytest's `tmp_path` or a
   new temporary directory for TSVs; never reuse real output paths.
4. If publication is requested, run CONTRIBUTING's delivery checks and continue
   with the existing prepare-github-push skill. Keep network audit, CI matrix,
   packaging installation checks, and opt-in performance work distinct from the
   local fixture result.

Output: a concise record of the behavior checked, exact commands, interpreter,
results, and checks omitted with reasons. Success requires zero exits and the
expected assertions/output; a passing smoke selection is not a passing CI matrix.
On a failure, retain the diagnostic and fix the cause within task scope, then
rerun affected checks. If dependencies or a site environment are unavailable,
report that boundary; do not suppress failures, weaken assertions, change package
requirements, query a real cluster, or add private captures to make the run pass.
