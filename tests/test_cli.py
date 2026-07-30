import os
import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CLI_PATH = REPO_ROOT / "kfbatch" / "kfbatch"


def _run_cli(args):
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        [sys.executable, str(CLI_PATH)] + args,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def test_legacy_stat_alias_is_rejected():
    out = _run_cli(["stat", "--stat_command", "qstat -F"])
    assert out.returncode != 0
    assert "unrecognized arguments" in out.stderr


def test_kfbatch_help_shows_stat_options_without_subcommands():
    out = _run_cli(["-h"])
    assert out.returncode == 0
    assert "--stat_command" in out.stdout
    assert "--slurm_node_command" in out.stdout
    assert "--uge_job_command" in out.stdout
    assert "--uge_qfree_command" in out.stdout
    assert "--out_nodes" in out.stdout
    assert "--out_jobs" in out.stdout
    assert "--command_timeout" in out.stdout
    assert "subcommands" not in out.stdout.lower()


def test_legacy_help_alias_is_rejected():
    out = _run_cli(["help", "stat"])
    assert out.returncode != 0
    assert "unrecognized arguments" in out.stderr


def test_unknown_option_returns_nonzero():
    out = _run_cli(["--this-option-does-not-exist"])
    assert out.returncode != 0
    assert "unrecognized arguments" in out.stderr


def test_negative_command_timeout_is_rejected():
    out = _run_cli(["--command_timeout", "-1"])
    assert out.returncode != 0
    assert "non-negative" in out.stderr


def test_conflicting_node_output_aliases_are_rejected_before_scheduler_access():
    out = _run_cli(["--out", "legacy.tsv", "--out_nodes", "nodes.tsv"])
    assert out.returncode == 1
    assert "--out and --out_nodes" in out.stderr
