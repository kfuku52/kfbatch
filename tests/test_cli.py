import os
import pathlib
import subprocess
import sys

import pytest

import kfbatch.cli as cli_module
from kfbatch import __version__
from kfbatch.errors import KFBatchCommandError

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


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_command_timeout_is_rejected(value):
    out = _run_cli([f"--command_timeout={value}"])
    assert out.returncode != 0
    assert "finite non-negative" in out.stderr


@pytest.mark.parametrize("option", ["--ntop", "--niter"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_integer_options_reject_zero_and_negative(option, value):
    out = _run_cli([option, value])
    assert out.returncode != 0
    assert "positive integer" in out.stderr


def test_version_option_reports_package_version():
    out = _run_cli(["--version"])
    assert out.returncode == 0
    assert out.stdout.strip().endswith(__version__)


def test_package_is_executable_as_a_module():
    out = subprocess.run(
        [sys.executable, "-m", "kfbatch", "--version"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert out.stdout.strip().endswith(__version__)


def test_conflicting_node_output_aliases_are_rejected_before_scheduler_access():
    out = _run_cli(["--out", "legacy.tsv", "--out_nodes", "nodes.tsv"])
    assert out.returncode == 1
    assert "--out and --out_nodes" in out.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [("yes", True), ("ON", True), ("0", False), ("false", False)],
)
def test_parse_bool_supported_spellings(text, expected):
    assert cli_module.parse_bool(text) is expected


def test_build_parser_exposes_scheduler_and_stable_priority_defaults():
    args = cli_module._build_parser().parse_args([])
    assert args.scheduler == "auto"
    assert args.ntop == 1
    assert "%i|%r|%Y|%S|%A|%F|%J|%P" in args.slurm_prio_command


def test_main_forwards_parsed_arguments(monkeypatch):
    observed = {}

    def fake_stat_main(args):
        observed["args"] = args

    monkeypatch.setattr("kfbatch.stat.stat_main", fake_stat_main)
    assert cli_module.main(["kfbatch", "--scheduler", "uge", "--niter", "2"]) == 0
    assert observed["args"].scheduler == "uge"
    assert observed["args"].niter == 2


def test_main_converts_domain_error_to_exit_status(monkeypatch, capsys):
    def fake_stat_main(args):
        raise KFBatchCommandError("synthetic failure")

    monkeypatch.setattr("kfbatch.stat.stat_main", fake_stat_main)
    assert cli_module.main(["kfbatch"]) == 1
    assert "synthetic failure" in capsys.readouterr().err
