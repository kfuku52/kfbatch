"""Synthetic regressions and invariants for the correctness audit."""

import json
import os
import shlex
import signal
import subprocess
import sys
from argparse import Namespace
from dataclasses import astuple

import pandas
import pytest

from kfbatch import stat
from kfbatch.batch_scope import aggregate_jobs, print_group_job_summary
from kfbatch.cli import _build_parser
from kfbatch.command import get_command_result
from kfbatch.errors import KFBatchCommandError
from kfbatch.job_states import SLURM_STATE_NAME_TO_CODE, slurm_bucket
from kfbatch.memory import grid_engine_memory_text_to_gib
from kfbatch.quota import _collect_records, _matches_filesystem, parse_quota_lines


@pytest.mark.parametrize("indent", ["", " ", "        ", "\t"])
@pytest.mark.parametrize("job_id", ["1", "1234567", "123456789012"])
def test_uge_job_indentation_does_not_change_count(indent, job_id):
    frame = stat.get_user_df(
        [
            "job-ID prior name user state submit/start at queue slots ja-task-ID",
            f"{indent}{job_id} 0.5 job current_user qw 09/22/2026 00:00:00 2 1-10",
        ]
    )
    assert frame["total_slots"].tolist() == [20]
    assert frame.attrs["parse_quality"]["rejected_rows"] == 0


@pytest.mark.parametrize("state,code", SLURM_STATE_NAME_TO_CODE.items())
def test_slurm_state_aliases_and_total_conservation(state, code, capsys):
    assert slurm_bucket(state) == slurm_bucket(code)
    frame = pandas.DataFrame(
        {
            "user": ["current_user"] * 2,
            "state": [state, code],
            "total_slots": [2, 3],
            "task_count_estimated": [False, False],
        }
    )
    totals = aggregate_jobs(frame, "slurm")
    assert sum(astuple(totals)) == 5
    stat.print_queued_job_summary(frame, "slurm", "current_user")
    expected = "/".join(str(n) for n in astuple(totals))
    assert f"self:R/Q/X/O={expected}" in capsys.readouterr().out


@pytest.mark.parametrize("state", ["ST", "STOPPED", "SO", "STAGE_OUT"])
def test_stopped_and_staging_jobs_retain_allocation(state):
    assert slurm_bucket(state) == "running"


def _node_frame():
    return stat.get_scontrol_node_df(
        [
            "NodeName=node01 Partitions=main,short CPUTot=32 CPUAlloc=0 "
            "RealMemory=65536 AllocMem=0 State=IDLE",
        ],
        {"main": "UP", "short": "UP"},
    )


@pytest.mark.parametrize(
    "lines",
    [
        ["ReservationName=r Nodes=node01 NodeCnt=1 CoreCnt=8 Users=other_user"],
        [
            "ReservationName=r State=ACTIVE PartitionName=main Nodes=node01 "
            "NodeCnt=2 CoreCnt=8 Users=other_user",
            "NodeName=node01 CoreIDs=garbage",
        ],
        [
            "ReservationName=r State=ACTIVE PartitionName=main Nodes=node[1-999999999] "
            "NodeCnt=2 CoreCnt=8 Users=other_user"
        ],
    ],
)
def test_uncertain_reservations_suppress_every_node_alias(lines, monkeypatch):
    args = _build_parser().parse_args(
        ["--stat_command", "squeue", "--current_user", "current_user"]
    )
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    frame = stat._apply_slurm_reservation_state(_node_frame(), stat.get_squeue_user_df([]), args, 1)
    assert frame["ncore_available"].tolist() == [0, 0]
    assert frame["status"].str.contains("UNKNOWN").all()
    assert not frame["hc:mem_req_known"].any()


@pytest.mark.parametrize(
    "hostlist",
    [
        "node[0-999999999]",
        "rack[0-999]node[0-999]",
        "node[4-1]",
        "node[[1-2]]",
        "node[1-2",
        "node[1]" * 17,
    ],
)
def test_hostlist_expansion_rejects_unbounded_or_invalid_input(hostlist):
    with pytest.raises(ValueError):
        stat._expand_slurm_hostlist(hostlist)


def test_hostlist_preserves_cartesian_product_and_zero_padding():
    assert stat._expand_slurm_hostlist("rack[01-02]node[1,3],node04") == [
        "rack01node1",
        "rack01node3",
        "rack02node1",
        "rack02node3",
        "node04",
    ]


def test_zero_effective_cpus_is_not_missing():
    frame = stat.get_scontrol_node_df(
        [
            "NodeName=node01 Partitions=main CPUTot=32 CPUEfctv=0 CPUAlloc=0 "
            "RealMemory=65536 AllocMem=0 State=IDLE",
        ],
        {"main": "UP"},
    )
    assert frame["ncore_total"].tolist() == [0]
    assert frame["ncore_available"].tolist() == [0]


def test_whole_node_reservation_covers_heterogeneous_nodes_and_partition_aliases():
    nodes = stat.get_scontrol_node_df(
        [
            "NodeName=node01 Partitions=main,short CPUTot=8 CPUAlloc=0 "
            "RealMemory=16384 AllocMem=0 State=IDLE+RESERVED ReservationName=hold",
            "NodeName=node02 Partitions=main,short CPUTot=32 CPUAlloc=0 "
            "RealMemory=65536 AllocMem=0 State=IDLE+RESERVED ReservationName=hold",
        ],
        {"main": "UP", "short": "UP"},
    )
    reservations = stat.get_scontrol_reservation_df(
        [
            "ReservationName=hold Nodes=node[01-02] NodeCnt=2 CoreCnt=40 "
            "PartitionName=main TRES=cpu=40 Users=other_user State=ACTIVE"
        ],
        current_user="current_user",
    )
    result = stat.adjust_ram_unit(stat.apply_slurm_reservations(nodes, reservations))
    assert result["ncore_available"].eq(0).all()
    assert result["hc:mem_req"].eq(0).all()
    launch = stat.get_slurm_launch_heuristic_df(result, stat.get_squeue_user_df([]))
    assert launch["recommended_cores"].eq(0).all()


@pytest.mark.parametrize(
    "topology,expected",
    [
        ("Sockets=1 CoresPerSocket=16 ThreadsPerCore=2", 16),
        ("ThreadsPerCore=2", 16),
        ("Sockets=1 CoresPerSocket=32 ThreadsPerCore=2", 24),
        ("ThreadsPerCore=1", 24),
        ("", 0),
        ("ThreadsPerCore=invalid", 0),
    ],
)
def test_partial_reservation_uses_node_cpu_units(topology, expected, monkeypatch, capsys):
    nodes = stat.get_scontrol_node_df(
        [
            f"NodeName=node01 Partitions=main,short CPUTot=32 CPUAlloc=0 {topology} "
            "RealMemory=65536 AllocMem=0 State=IDLE"
        ],
        {"main": "UP", "short": "UP"},
    )
    reservations = stat.get_scontrol_reservation_df(
        [
            "ReservationName=hold Nodes=node01 NodeCnt=1 CoreCnt=8 "
            "PartitionName=main Users=other_user State=ACTIVE",
            "NodeName=node01 CoreIDs=0-7",
        ],
        current_user="current_user",
    )
    result = stat.apply_slurm_reservations(nodes, reservations)
    assert result["ncore_available"].tolist() == [expected, expected]
    assert set(result.columns) == set(stat.SLURM_NODE_COLUMNS) | {
        "reservation_cores",
        "reservation_mem_mb",
        "reservation_accessible",
    }
    monkeypatch.setattr(stat, "get_scontrol_reservation_df", lambda *_, **__: reservations)
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: [])
    args = _build_parser().parse_args(["--current_user", "current_user"])
    cli_result = stat._apply_slurm_reservation_state(nodes, stat.get_squeue_user_df([]), args, 1)
    assert cli_result["ncore_available"].tolist() == [expected, expected]
    assert ("CPU topology is unknown" in capsys.readouterr().out) is (expected == 0)


@pytest.mark.parametrize(
    "lines",
    [
        ["No reservations in the system"],
        ["  No reservations in the system  "],
        [
            "ReservationName=license_hold Nodes= NodeCnt=0 CoreCnt=0 "
            "PartitionName=(null) Flags=LICENSE_ONLY TRES=license/tool_a=1 "
            "Users=other_user Licenses=tool_a State=ACTIVE"
        ],
        [
            "ReservationName=license_hold Nodes=(null) NodeCnt=0 CoreCnt=0 "
            "PartitionName=(null) Flags=ANY_NODES TRES=license/tool_a=1 "
            "Users=other_user Licenses=tool_a State=ACTIVE"
        ],
        [
            "ReservationName=buffer_hold Nodes=(null) NodeCnt=0 CoreCnt=0 "
            "PartitionName=(null) Flags=ANY_NODES TRES=bb/tool_a=1 "
            "Users=other_user BurstBuffer=tool_a:1 State=ACTIVE"
        ],
    ],
)
def test_reservation_responses_without_compute_resources_preserve_capacity(
    lines, monkeypatch, capsys
):
    args = _build_parser().parse_args(["--current_user", "current_user"])
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    result = stat._apply_slurm_reservation_state(
        _node_frame(), stat.get_squeue_user_df([]), args, 1
    )
    assert result["ncore_available"].tolist() == [32, 32]
    assert result["hc:mem_req"].tolist() == ["65536M", "65536M"]
    assert "degraded" not in capsys.readouterr().out


def test_license_flag_does_not_hide_a_compute_reservation(monkeypatch):
    lines = [
        "ReservationName=hold Nodes=node01 NodeCnt=1 CoreCnt=32 "
        "PartitionName=main Flags=LICENSE_ONLY Users=other_user State=ACTIVE"
    ]
    args = _build_parser().parse_args(["--current_user", "current_user"])
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    result = stat._apply_slurm_reservation_state(
        _node_frame(), stat.get_squeue_user_df([]), args, 1
    )
    assert result["ncore_available"].eq(0).all()


@pytest.mark.parametrize(
    "field,value,missing",
    [
        ("slots", None, "slots"),
        ("slots", "invalid", "slots"),
        ("slots", -1, "slots"),
        ("slots", 1.5, "slots"),
        ("JB_owner", None, "user"),
        ("JB_owner", "", "user"),
        ("state", None, "state"),
        ("state", "", "state"),
    ],
)
def test_incomplete_uge_json_cannot_be_a_complete_cluster_total(
    field, value, missing, monkeypatch, capsys
):
    job = {
        "JB_job_number": 1,
        "JB_owner": "current_user",
        "state": "r",
        "queue_name": "main.q@node01",
        "slots": 16,
    }
    if value is None:
        job.pop(field)
    else:
        job[field] = value
    lines = [json.dumps({"job_info": {"job_list": [job]}})]
    parsed = stat.get_uge_json_job_df(lines)
    assert parsed.attrs["rejected_rows"] == 1
    assert missing in parsed.attrs["missing_fields"]
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    fallback = stat.get_user_df([])
    result, all_users = stat._get_uge_all_user_jobs(
        Namespace(uge_job_command="qstat", uge_job_example_file=""), fallback, 1
    )
    assert result is fallback
    assert not all_users
    assert "degraded" in capsys.readouterr().out


@pytest.mark.parametrize("parent_waits", [False, True])
@pytest.mark.parametrize("allow_failure", [False, True])
def test_detached_child_cannot_hold_command_capture_open(tmp_path, parent_waits, allow_failure):
    pid_file = tmp_path / "child.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"Path({str(pid_file)!r}).write_text(str(p.pid)); "
        + ("time.sleep(60)" if parent_waits else "")
    )
    command = shlex.join([sys.executable, "-c", parent_code])
    runner_code = (
        "import json; from kfbatch.command import get_command_result; "
        "from kfbatch.errors import KFBatchCommandError\n"
        "try:\n"
        f"    result=get_command_result({command!r}, timeout_seconds=0.2, "
        f"allow_failure={allow_failure!r})\n"
        "    print(json.dumps({'result': result}))\n"
        "except KFBatchCommandError as error:\n"
        "    print(json.dumps({'timed_out': error.timed_out, 'message': str(error)}))\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", runner_code],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        failure = json.loads(result.stdout)
        if allow_failure:
            assert failure == {"result": None}
        else:
            assert failure["timed_out"] is parent_waits
            if not parent_waits:
                assert "output pipes" in failure["message"]
    finally:
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    "first,second",
    [
        ("1023.9M", "1023.9M"),
        ("1G", "1023.9M"),
        ("1200M", "1G"),
        ("1g", "1000M"),
    ],
)
def test_additional_snapshots_never_increase_memory(first, second):
    def snapshot(memory):
        return stat.get_qstat_df(
            [
                "main.q@node01 BIP 0/0/8 0 linux",
                f"\thc:mem_req={memory}",
                "\thl:mem_total=2G",
            ]
        )

    frame = snapshot(first)
    for _ in range(3):
        before = grid_engine_memory_text_to_gib(frame.at[0, "hc:mem_req"])
        frame = stat._merge_qstat_iteration_min_availability(frame, snapshot(second))
        after = grid_engine_memory_text_to_gib(frame.at[0, "hc:mem_req"])
        assert after <= before
        assert after <= grid_engine_memory_text_to_gib(second)


def test_partial_squeue_output_fails_before_printing_totals(monkeypatch, capsys):
    lines = ["1 p job current_user R 0:00 1 node01", "2 p job current_user R 0:00 N/A node02"]
    frame = stat.get_squeue_user_df(lines)
    assert frame.attrs["parse_quality"]["rejected_rows"] == 1
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    args = _build_parser().parse_args(["--stat_command", "squeue"])
    with pytest.raises(KFBatchCommandError, match="incomplete"):
        stat.get_df(args)
    assert "all:R" not in capsys.readouterr().out


def test_partial_uge_output_is_not_a_cluster_total(monkeypatch, capsys):
    lines = ["1 0.5 job current_user r 09/22/2026 00:00:00 main.q@node01 1", "2 malformed"]
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    args = Namespace(uge_job_command="qstat", uge_job_example_file="")
    fallback = stat.get_user_df([])
    result, all_users = stat._get_uge_all_user_jobs(args, fallback, 1)
    assert result is fallback
    assert not all_users
    assert "rejected" in capsys.readouterr().out


@pytest.mark.parametrize("accounts", [[""], ["account_a", ""]])
def test_missing_account_data_is_not_zero_group_jobs(accounts, capsys):
    frame = pandas.DataFrame(
        {"account": accounts, "state": ["R"] * len(accounts), "total_slots": [1] * len(accounts)}
    )
    assert not print_group_job_summary(
        frame, scheduler="slurm", current_user="current_user", group_id="account_a"
    )
    assert "unavailable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "row,grace",
    [
        ("/home 100 200 300     10 20 30", ""),
        ("/home 100 200 300 - 10 20 30 -", ""),
        ("/home 100 200 300 3days 10 20 30", "3days"),
        ("/home 100 200 300     10 20 30 2days", "space=-,files=2days"),
        ("/home 100 200 300 0 10 20 30 0", ""),
        ("/home 100 200 300 1234567890 10 20 30 0", "1234567890"),
    ],
)
def test_quota_grace_columns_do_not_shift_counts(row, grace):
    records = parse_quota_lines(
        [
            "Disk quotas for user current_user (uid 1001):",
            "Filesystem blocks quota limit grace files quota limit grace",
            row,
        ],
        "posix",
    )
    assert len(records) == 1
    record = records[0]
    assert (record.bytes_used, record.bytes_soft, record.bytes_hard) == (
        100 * 1024,
        200 * 1024,
        300 * 1024,
    )
    assert (record.files_used, record.files_soft, record.files_hard) == (10, 20, 30)
    assert record.grace == grace


def test_quota_human_counts_and_wrapped_filesystem():
    records = parse_quota_lines(
        [
            "Disk quotas for user current_user (uid 1001):",
            "Filesystem space quota limit grace files quota limit grace",
            "/dev/very-long-filesystem-name",
            "1G 2G 3G - 100k 1.5M 2M -",
        ],
        "posix",
    )
    assert len(records) == 1
    assert (records[0].files_used, records[0].files_soft, records[0].files_hard) == (
        100_000,
        1_500_000,
        2_000_000,
    )


@pytest.mark.parametrize(
    "filesystem,expected",
    [
        ("/home", True),
        ("/home/", True),
        ("/home/current_user", True),
        ("home", True),
        ("/home2", False),
    ],
)
def test_home_filesystem_filter(filesystem, expected):
    record = parse_quota_lines(
        ["scope owner filesystem bytes_used", f"self current_user {filesystem} 1G"], "custom"
    )[0]
    assert _matches_filesystem(record, "home") is expected


def _quota_command(status, output):
    return shlex.join(
        [
            sys.executable,
            "-c",
            f"import sys; print({output!r}); print('diagnostic', file=sys.stderr); sys.exit({status})",
        ]
    )


@pytest.mark.parametrize(
    "status,provider,valid",
    [(0, "posix", True), (1, "posix", True), (2, "posix", False), (1, "custom", False)],
)
def test_quota_exit_status_is_provider_specific(status, provider, valid, capsys):
    args = Namespace(
        quota_example_file="",
        provider=provider,
        quota_command=_quota_command(
            status, "scope owner filesystem bytes_used\nself current_user /home 1G"
        ),
        command_timeout=2,
    )
    if not valid:
        with pytest.raises(KFBatchCommandError):
            _collect_records(args)
    else:
        assert len(_collect_records(args)) == 1
        if status == 1:
            assert "exceeded" in capsys.readouterr().out


def test_quota_nonzero_with_no_records_is_still_an_error():
    args = Namespace(
        quota_example_file="",
        provider="posix",
        quota_command=_quota_command(1, "broken output"),
        command_timeout=2,
    )
    with pytest.raises(KFBatchCommandError):
        _collect_records(args)


def test_command_result_preserves_exit_and_stderr():
    result = get_command_result(_quota_command(1, "data"), accepted_returncodes=(0, 1))
    assert result.returncode == 1
    assert result.stdout_lines == ["data"]
    assert result.stderr == "diagnostic"


@pytest.mark.parametrize(
    "hostlist", ["longprefix[1-9]", "node01,node02,node03", "node[0000001-0000009]"]
)
def test_hostlist_bounds_expanded_bytes(hostlist, monkeypatch):
    monkeypatch.setattr(stat, "MAX_EXPANDED_HOST_BYTES", 16)
    with pytest.raises(ValueError):
        stat._expand_slurm_hostlist(hostlist)


def test_uge_json_rejected_job_is_not_a_complete_total(monkeypatch):
    import json

    lines = [
        json.dumps(
            {
                "job_info": {
                    "job_list": [
                        {"JB_job_number": 1, "JB_owner": "current_user", "state": "qw", "slots": 1},
                        {"JB_owner": "other_user", "state": "qw"},
                    ]
                }
            }
        )
    ]
    frame = stat.get_uge_json_job_df(lines)
    assert frame.attrs["parse_quality"]["rejected_rows"] == 1
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: lines)
    fallback = stat.get_user_df([])
    frame, all_users = stat._get_uge_all_user_jobs(
        Namespace(uge_job_command="qstat", uge_job_example_file=""),
        fallback,
        1,
    )
    assert frame is fallback
    assert not all_users


@pytest.mark.parametrize(
    "group_slots,expected", [([None], "?"), ([3, None], "?"), ([0], "0"), ([3, 2], "5")]
)
def test_uge_group_running_total_preserves_unknown_queues(group_slots, expected, capsys):
    qfree = pandas.DataFrame({"group_slots": group_slots})
    qfree.attrs["group_name"] = "group_a"
    jobs = stat.get_user_df([])
    assert print_group_job_summary(
        jobs, scheduler="uge", current_user="current_user", qfree_frame=qfree
    )
    assert f"group[group_a]:R/Q/F={expected}/?/?" in capsys.readouterr().out


def test_uge_explicit_group_requires_discovered_identity(capsys):
    qfree = pandas.DataFrame({"group_slots": [3]})
    qfree.attrs["group_users"] = ["current_user"]
    jobs = stat.get_user_df([])
    jobs.attrs["all_users"] = True
    assert not print_group_job_summary(
        jobs,
        scheduler="uge",
        current_user="current_user",
        group_id="group_a",
        qfree_frame=qfree,
    )
    assert "unavailable" in capsys.readouterr().out


@pytest.mark.parametrize("allow_failure", [False, True])
@pytest.mark.parametrize("example", [False, True])
def test_command_line_limit_respects_optional_failure(
    monkeypatch, tmp_path, allow_failure, example
):
    import kfbatch.command as command_module

    monkeypatch.setattr(command_module, "MAX_OUTPUT_LINE_BYTES", 8)
    fixture = tmp_path / "output.txt"
    fixture.write_text("x" * 9 + "\n")
    command = shlex.join([sys.executable, "-c", "print('x' * 9)"])
    kwargs = {"allow_failure": allow_failure, "example_file": str(fixture) if example else ""}
    if allow_failure:
        assert get_command_result(command, **kwargs) is None
    else:
        with pytest.raises(KFBatchCommandError) as caught:
            get_command_result(command, **kwargs)
        assert caught.value.output_limited


@pytest.mark.parametrize("header,factor", [("kfiles", 1000), ("mfiles", 1000000)])
def test_quota_fractional_scaled_file_counts(header, factor):
    records = parse_quota_lines(
        [
            "Disk quotas for user current_user (uid 1000):",
            f"Filesystem Gbytes quota limit grace {header} quota limit grace",
            "/home 1 2 3 - 0.5 1.25 2.75 -",
        ],
        provider="lfsq",
    )
    assert len(records) == 1
    assert (records[0].files_used, records[0].files_soft, records[0].files_hard) == (
        factor // 2,
        factor * 5 // 4,
        factor * 11 // 4,
    )


def test_uge_memory_only_qfree_does_not_imply_zero_running_jobs(capsys):
    qfree = stat.get_qfree_df(
        [
            "SUMMARY OF RUNNING JOBS ( MEM_REQ )",
            "mjobs.q 8 16 - 128 24 16 512",
            "THE NUMBER OF MEM_REQ BY USER IN THE GROUP (group_a)",
        ]
    )
    assert print_group_job_summary(
        stat.get_user_df([]), scheduler="uge", current_user="current_user", qfree_frame=qfree
    )
    assert "group[group_a]:R/Q/F=?/?/?" in capsys.readouterr().out
