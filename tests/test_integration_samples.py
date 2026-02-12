import os
import pathlib
import subprocess
import sys

import pandas
import pytest

from kfbatch.stat import (
    get_qstat_df,
    get_scontrol_node_df,
    get_scontrol_partition_df,
    get_squeue_user_df,
)


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


@pytest.mark.parametrize(
    "fixture_path",
    [
        "data/qstat1/qstatF.txt",
        "data/qstat2/qstatF.txt",
        "data/qstat3/qstatF.txt",
        "data/qstat4/qstatF.txt",
    ],
)
def test_qstat_sample_parsing_invariants(fixture_path):
    with open(REPO_ROOT / fixture_path) as fh:
        lines = fh.readlines()
    df = get_qstat_df(lines)
    assert df.shape[0] > 0
    assert (df["ncore_available"] >= 0).all()
    assert "hc:mem_req" in df.columns
    assert "hl:mem_total" in df.columns
    assert not df["queue_name"].isna().any()
    assert not df["node_name"].isna().any()


@pytest.mark.parametrize(
    "fixture_path",
    [
        "data/qstat1/qstatF.txt",
        "data/qstat2/qstatF.txt",
        "data/qstat3/qstatF.txt",
        "data/qstat4/qstatF.txt",
    ],
)
def test_qstat_cli_runs_on_all_sample_snapshots(fixture_path):
    out = _run_cli(
        [
            "--example_file",
            fixture_path,
            "--stat_command",
            "qstat -F",
            "--niter",
            "1",
        ]
    )
    assert out.returncode == 0
    assert "# of CPUs in use for running jobs:" in out.stdout
    assert "Reporting working/abnormal/total nodes" in out.stdout


def test_slurm_sample_parsing_invariants():
    with open(REPO_ROOT / "squeue_notrunc.txt") as fh:
        squeue_lines = fh.readlines()
    with open(REPO_ROOT / "scontrol_show_partition_o.txt") as fh:
        partition_lines = fh.readlines()
    with open(REPO_ROOT / "scontrol_show_node_o.txt") as fh:
        node_lines = fh.readlines()
    df_user = get_squeue_user_df(squeue_lines)
    assert df_user.shape[0] > 300
    assert set(["R", "PD"]).issuperset(set(df_user["state"].unique()))
    df_partition = get_scontrol_partition_df(partition_lines)
    partition_state_map = (
        df_partition.set_index("partition_name")["partition_state"].to_dict()
    )
    df_node = get_scontrol_node_df(node_lines, partition_state_map=partition_state_map)
    assert df_node.shape[0] > 0
    assert (df_node["ncore_available"] >= 0).all()
    assert "login" in df_node["queue_name"].unique()
    df_login = df_node[df_node["queue_name"] == "login"]
    assert (df_login["status"].str.contains("partition_state=INACTIVE", regex=False)).all()


def test_slurm_cli_writes_valid_tsv(tmp_path):
    out_file = tmp_path / "slurm.tsv"
    out = _run_cli(
        [
            "--example_file",
            "squeue_notrunc.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "scontrol_show_node_o.txt",
            "--slurm_partition_example_file",
            "scontrol_show_partition_o.txt",
            "--out",
            str(out_file),
        ]
    )
    assert out.returncode == 0
    assert out_file.exists()
    df = pandas.read_csv(out_file, sep="\t")
    assert df.shape[0] > 0
    expected_cols = {"queue_name", "node_name", "ncore_available", "hc:mem_req", "hl:mem_total"}
    assert expected_cols.issubset(set(df.columns))


def test_slurm_cli_truncated_squeue_reports_estimated_note():
    out = _run_cli(
        [
            "--example_file",
            "squeue.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "scontrol_show_node_o.txt",
            "--slurm_partition_example_file",
            "scontrol_show_partition_o.txt",
        ]
    )
    assert out.returncode == 0
    assert "Note:" in out.stdout
    assert "task counts are estimated" in out.stdout


def test_qstat_cli_writes_valid_tsv(tmp_path):
    out_file = tmp_path / "qstat.tsv"
    out = _run_cli(
        [
            "--example_file",
            "data/qstat1/qstatF.txt",
            "--stat_command",
            "qstat -F",
            "--niter",
            "1",
            "--out",
            str(out_file),
        ]
    )
    assert out.returncode == 0
    assert out_file.exists()
    df = pandas.read_csv(out_file, sep="\t")
    assert df.shape[0] > 0
    expected_cols = {"queue_name", "node_name", "ncore_available", "hc:mem_req", "hl:mem_total"}
    assert expected_cols.issubset(set(df.columns))
