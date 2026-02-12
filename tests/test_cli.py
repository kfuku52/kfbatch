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
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_kfbatch_direct_mode_matches_legacy_stat_alias():
    args = [
        "--example_file",
        "data/qstat1/qstatF.txt",
        "--stat_command",
        "qstat -F",
        "--niter",
        "1",
    ]
    direct = _run_cli(args)
    legacy = _run_cli(["stat"] + args)
    assert direct.returncode == 0
    assert legacy.returncode == 0
    assert direct.stdout == legacy.stdout


def test_kfbatch_help_shows_stat_options_without_subcommands():
    out = _run_cli(["-h"])
    assert out.returncode == 0
    assert "--stat_command" in out.stdout
    assert "--slurm_node_command" in out.stdout
    assert "subcommands" not in out.stdout.lower()


def test_legacy_help_alias_maps_to_main_help():
    out = _run_cli(["help", "stat"])
    assert out.returncode == 0
    assert "--stat_command" in out.stdout


def test_unknown_option_returns_nonzero():
    out = _run_cli(["--this-option-does-not-exist"])
    assert out.returncode != 0
    assert "unrecognized arguments" in out.stderr
