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
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize(
    "fixture_path",
    [
        "tests/fixtures/age/qstat_f_1.txt",
        "tests/fixtures/age/qstat_f_2.txt",
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
        "tests/fixtures/age/qstat_f_1.txt",
        "tests/fixtures/age/qstat_f_2.txt",
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
            "--uge_job_example_file",
            "tests/fixtures/age/qstat_all_users.txt",
            "--uge_qfree_example_file",
            "tests/fixtures/age/qfree.txt",
        ]
    )
    assert out.returncode == 0
    assert "jobs  self:R/Q/F=" in out.stdout
    assert "queue" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "ram(a/t)GiB" in out.stdout
    assert "topCPU" in out.stdout
    assert "topRAM" in out.stdout


def test_slurm_sample_parsing_invariants():
    with open(REPO_ROOT / "tests/fixtures/slurm/squeue_full.txt") as fh:
        squeue_lines = fh.readlines()
    with open(REPO_ROOT / "tests/fixtures/slurm/partitions.txt") as fh:
        partition_lines = fh.readlines()
    with open(REPO_ROOT / "tests/fixtures/slurm/nodes.txt") as fh:
        node_lines = fh.readlines()
    df_user = get_squeue_user_df(squeue_lines)
    assert df_user.shape[0] == 3
    assert set(["R", "PD"]).issuperset(set(df_user["state"].unique()))
    df_partition = get_scontrol_partition_df(partition_lines)
    partition_state_map = df_partition.set_index("partition_name")["partition_state"].to_dict()
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
            "tests/fixtures/slurm/squeue_full.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
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
            "tests/fixtures/slurm/squeue_legacy.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
        ]
    )
    assert out.returncode == 0
    assert "note:" in out.stdout
    assert "task counts are estimated" in out.stdout


def test_slurm_cli_uses_compact_partition_table():
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_full.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
        ]
    )
    assert out.returncode == 0
    assert "jobs  self:R/Q/F=" in out.stdout
    assert "part" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "ram(a/t)GiB" in out.stdout
    assert "launch" in out.stdout


def test_qstat_cli_writes_valid_tsv(tmp_path):
    out_file = tmp_path / "qstat.tsv"
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/age/qstat_f_1.txt",
            "--stat_command",
            "qstat -F",
            "--niter",
            "1",
            "--uge_job_example_file",
            "tests/fixtures/age/qstat_all_users.txt",
            "--uge_qfree_example_file",
            "tests/fixtures/age/qfree.txt",
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


def test_cli_writes_separate_node_and_job_schemas(tmp_path):
    node_file = tmp_path / "nodes.tsv"
    job_file = tmp_path / "jobs.tsv"
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/age/qstat_f_1.txt",
            "--stat_command",
            "qstat -F",
            "--uge_job_example_file",
            "tests/fixtures/age/qstat_all_users.txt",
            "--uge_qfree_example_file",
            "tests/fixtures/age/qfree.txt",
            "--out_nodes",
            str(node_file),
            "--out_jobs",
            str(job_file),
        ]
    )
    assert out.returncode == 0
    node_df = pandas.read_csv(node_file, sep="\t")
    job_df = pandas.read_csv(job_file, sep="\t")
    assert {"node_name", "ncore_available"}.issubset(node_df.columns)
    assert {"job_id", "user", "total_slots"}.issubset(job_df.columns)
    assert "job_id" not in node_df.columns
    assert "node_name" not in job_df.columns
