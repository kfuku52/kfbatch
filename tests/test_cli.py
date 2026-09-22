import os
import pathlib
import subprocess
import sys

import pytest

import kfbatch.cli as cli_module
from kfbatch import __version__

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
    assert "invalid choice" in out.stderr


def test_kfbatch_help_shows_subcommands():
    out = _run_cli(["-h"])
    assert out.returncode == 0
    assert "batch" in out.stdout
    assert "quota" in out.stdout
    assert "--stat_command" not in out.stdout


def test_negative_command_timeout_is_rejected():
    out = _run_cli(["--command_timeout", "-1"])
    assert out.returncode != 0
    assert "non-negative" in out.stderr


def test_negative_quota_command_timeout_is_rejected():
    out = _run_cli(["quota", "--command-timeout", "-1"])
    assert out.returncode == 2
    assert "non-negative" in out.stderr


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_nonfinite_command_timeout_is_rejected(value):
    out = _run_cli([f"--command_timeout={value}"])
    assert out.returncode != 0
    assert "finite non-negative" in out.stderr


@pytest.mark.parametrize(("option", "value"), [("--ntop", "0"), ("--niter", "-1")])
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
    assert out.returncode == 2
    assert "--out and --out_nodes" in out.stderr


@pytest.mark.parametrize(
    ("text", "expected"),
    [("yes", True), ("ON", True), ("0", False), ("false", False)],
)
def test_parse_bool_supported_spellings(text, expected):
    assert cli_module.parse_bool(text) is expected


@pytest.mark.parametrize(
    ("available_commands", "expected"),
    [
        ({"squeue", "qstat"}, "squeue"),
        ({"qstat"}, "qstat -F"),
        (set(), "squeue"),
    ],
)
def test_build_parser_auto_detects_stat_command(monkeypatch, available_commands, expected):
    monkeypatch.setattr(
        cli_module.shutil,
        "which",
        lambda command: f"/usr/bin/{command}" if command in available_commands else None,
    )

    args = cli_module._build_parser().parse_args([])

    assert args.stat_command == expected


def test_long_options_do_not_accept_ambiguous_abbreviations():
    out = _run_cli(["--sched", "uge"])
    assert out.returncode == 2
    assert "unrecognized arguments" in out.stderr


def test_niter_has_an_operational_upper_bound():
    out = _run_cli(["--niter", str(cli_module.MAX_NITER + 1)])
    assert out.returncode == 2
    assert f"--niter must be <= {cli_module.MAX_NITER}" in out.stderr


def test_custom_quota_provider_requires_command():
    out = _run_cli(["quota", "--provider", "custom"])
    assert out.returncode == 2
    assert "requires --quota-command" in out.stderr


def test_failed_lfsq_provider_explains_qlogin_without_starting_it():
    out = _run_cli(["quota", "--provider", "lfsq", "--quota-command", "false"])
    assert out.returncode == 1
    assert "run qlogin first" in out.stderr


def test_unrecognized_successful_lfsq_output_does_not_explain_qlogin():
    out = _run_cli(["quota", "--provider", "lfsq", "--quota-command", "printf unrecognized"])
    assert out.returncode == 1
    assert "completed successfully" in out.stderr
    assert "qlogin" not in out.stderr
