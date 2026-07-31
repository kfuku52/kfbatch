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
SLURM_OPTIONAL_FIXTURE_ARGS = [
    "--current_user",
    "current_user",
    "--slurm_share_example_file",
    "tests/fixtures/slurm/sshare.txt",
    "--slurm_prio_example_file",
    "tests/fixtures/slurm/sprio.txt",
]


def _run_cli(args, extra_env=None):
    env = os.environ.copy()
    pythonpath = str(REPO_ROOT)
    if env.get("PYTHONPATH"):
        pythonpath += os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath
    if extra_env is not None:
        env.update(extra_env)
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
            *SLURM_OPTIONAL_FIXTURE_ARGS,
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
            *SLURM_OPTIONAL_FIXTURE_ARGS,
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
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ]
    )
    assert out.returncode == 0
    assert "jobs  self:R/Q/X/O=" in out.stdout
    assert "part" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "ram(a/t)GiB" in out.stdout
    assert "launch" in out.stdout


def test_slurm_cli_reports_fairshare_ranks_from_fixture():
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_fairshare.txt",
            "--stat_command",
            "squeue",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
            "--slurm_share_example_file",
            "tests/fixtures/slurm/sshare.txt",
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ],
        extra_env={"USER": "current_user", "LOGNAME": "current_user"},
    )
    assert out.returncode == 0
    assert "fairshare  self=0.500000" in out.stdout
    assert "assoc_rank=2/3" in out.stdout
    assert "pending_assoc_rank=2/2" in out.stdout


def test_batch_subcommand_reports_slurm_group_jobs_by_user():
    out = _run_cli(
        [
            "batch",
            "--example_file",
            "tests/fixtures/slurm/squeue_fairshare.txt",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
            "--slurm_share_example_file",
            "tests/fixtures/slurm/sshare.txt",
            "--slurm_prio_example_file",
            "tests/fixtures/slurm/sprio.txt",
            "--current_user",
            "current_user",
            "--scope",
            "group",
            "--by-user",
        ]
    )
    assert out.returncode == 0
    assert "group[account_a]:R/Q/X/O=0/2/0/0" in out.stdout
    assert "current_user:R/Q/X/O=0/1/0/0" in out.stdout
    assert "user_a:R/Q/X/O=0/1/0/0" in out.stdout


def test_batch_subcommand_reports_uge_group_jobs_by_user():
    out = _run_cli(
        [
            "batch",
            "--scheduler",
            "uge",
            "--stat_command",
            "qstat -F",
            "--example_file",
            "tests/fixtures/age/qstat_f_1.txt",
            "--uge_job_example_file",
            "tests/fixtures/age/qstat_all_users.txt",
            "--uge_qfree_example_file",
            "tests/fixtures/age/qfree.txt",
            "--current_user",
            "user_a",
            "--scope",
            "group",
            "--by-user",
        ]
    )
    assert out.returncode == 0
    assert "group[group_a]:R/Q/F=4/200/0" in out.stdout
    assert "user_a:R/Q/F=4/0/0" in out.stdout
    assert "user_b:R/Q/F=0/200/0" in out.stdout


def test_quota_subcommand_reports_personal_and_group_fixture_rows():
    out = _run_cli(
        [
            "quota",
            "--quota-example-file",
            "tests/fixtures/quota/normalized.txt",
            "--current-user",
            "user_a",
        ]
    )
    assert out.returncode == 0
    assert "user_a" in out.stdout
    assert "group_a" in out.stdout
    assert "shared by all group members" in out.stdout


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


def test_slurm_cli_keeps_compact_layout_when_launch_heuristic_is_disabled():
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_full.txt",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            "tests/fixtures/slurm/reservations.txt",
            "--show_fairshare_rank",
            "no",
            "--show_launch_heuristic",
            "no",
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ]
    )
    assert out.returncode == 0
    assert "part" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "Reporting top" not in out.stdout


def test_slurm_node_failure_writes_jobs_but_returns_nonzero(tmp_path):
    jobs_path = tmp_path / "jobs.tsv"
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_full.txt",
            "--slurm_node_command",
            "false",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--show_fairshare_rank",
            "no",
            "--out_jobs",
            str(jobs_path),
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ]
    )
    assert out.returncode == 1
    assert jobs_path.exists()
    assert "Slurm node/resource data is unavailable" in out.stderr


def test_slurm_reservation_failure_suppresses_resource_ceiling():
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_full.txt",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_command",
            "false",
            "--show_fairshare_rank",
            "no",
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ]
    )
    assert out.returncode == 0
    assert "resource ceilings are suppressed" in out.stdout
    assert "epyc   0/2/2" in out.stdout


def test_cli_rejects_same_node_and_job_output_path_before_scheduler_access(tmp_path):
    output_path = tmp_path / "same.tsv"
    out = _run_cli(
        [
            "--out_nodes",
            str(output_path),
            "--out_jobs",
            str(output_path),
        ]
    )
    assert out.returncode == 2
    assert "must refer to different files" in out.stderr
    assert not output_path.exists()


def test_slurm_cli_rejects_nonempty_unrecognized_job_output(tmp_path):
    malformed = tmp_path / "malformed-squeue.txt"
    malformed.write_text("warning: output format changed\n", encoding="utf-8")
    out = _run_cli(
        [
            "--scheduler",
            "slurm",
            "--example_file",
            str(malformed),
            "--show_fairshare_rank",
            "no",
            "--show_launch_heuristic",
            "no",
        ]
    )
    assert out.returncode == 1
    assert "non-empty but contained no recognized squeue rows" in out.stderr


def test_slurm_cli_suppresses_partition_for_unmatched_active_reservation(tmp_path):
    reservation = tmp_path / "unmatched-reservation.txt"
    reservation.write_text(
        "ReservationName=unknown Nodes=does-not-exist NodeCnt=1 CoreCnt=8 "
        "PartitionName=epyc Users=other State=ACTIVE\n",
        encoding="utf-8",
    )
    out = _run_cli(
        [
            "--example_file",
            "tests/fixtures/slurm/squeue_full.txt",
            "--slurm_node_example_file",
            "tests/fixtures/slurm/nodes.txt",
            "--slurm_partition_example_file",
            "tests/fixtures/slurm/partitions.txt",
            "--slurm_reservation_example_file",
            str(reservation),
            *SLURM_OPTIONAL_FIXTURE_ARGS,
        ]
    )
    assert out.returncode == 0
    assert "resource ceilings are suppressed" in out.stdout
    assert "epyc   0/2/2" in out.stdout
