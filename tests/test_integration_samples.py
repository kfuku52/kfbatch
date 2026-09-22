import os
import pathlib
import subprocess
import sys

import pandas

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
    assert "all:R/Q/X/O=4/2/0/0" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "ram(a/t)GiB" in out.stdout
    assert "launch" in out.stdout
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
    assert "user_c:R/Q/F=" not in out.stdout


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
    assert "self" in out.stdout
    assert "71.2TiB" in out.stdout
    assert "shared by all group members" in out.stdout


def test_quota_subcommand_parses_shirokane_lfsq_units():
    out = _run_cli(
        [
            "quota",
            "--quota-example-file",
            "tests/fixtures/quota/lfsq.txt",
            "--current-user",
            "user_a",
        ]
    )
    assert out.returncode == 0
    assert "8.0GiB/-/-" in out.stdout
    assert "34,000/-/-" in out.stdout
    assert "214GiB/-/6.0TiB" in out.stdout
    assert "327,000/-/6,000,000" in out.stdout


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
    assert "all:R/Q/F=4/248/5" in out.stdout
    assert "cpu(a/u/t)" in out.stdout
    assert "ram(a/t)GiB" in out.stdout
    assert "topCPU" in out.stdout
    assert "topRAM" in out.stdout
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
