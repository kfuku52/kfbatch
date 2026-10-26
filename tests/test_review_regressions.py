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

from kfbatch import command as command_module, memory, stat, uge_parser
from kfbatch.batch_scope import aggregate_jobs, print_group_job_summary
from kfbatch.cli import _build_parser
from kfbatch.command import get_command_result
from kfbatch.errors import KFBatchCommandError
from kfbatch.job_states import SLURM_STATE_NAME_TO_CODE, slurm_bucket
from kfbatch.memory import grid_engine_memory_text_to_gib
from kfbatch.quota import _collect_records, _matches_filesystem, parse_quota_lines


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\n\n",
        b"a\nb\n",
        b"a\nb",
        b"a\rb\r",
        b"a\r\nb\r\n",
        b"a\x0bb\x0c",
        b"a\xff\nb\n",
        "日本語\n次\n".encode(),
    ],
)
def test_bounded_decoder_preserves_scheduler_line_semantics(data):
    expected = command_module._decode_scheduler_lines(data.splitlines())
    assert command_module._decode_bounded_lines(data, "synthetic") == expected


@pytest.mark.parametrize("ending", [b"", b"\n", b"\r", b"\r\n"])
def test_bounded_decoder_counts_content_bytes(monkeypatch, ending):
    monkeypatch.setattr(command_module, "MAX_OUTPUT_LINE_BYTES", 8)
    assert command_module._decode_bounded_lines(b"x" * 8 + ending, "synthetic") == ["x" * 8]
    with pytest.raises(KFBatchCommandError) as error:
        command_module._decode_bounded_lines(b"x" * 9 + ending, "synthetic")
    assert error.value.output_limited


def test_command_capture_preserves_both_large_streams():
    code = "import os; [(os.write(1,b'out\\n'*4096),os.write(2,b'err\\n'*4096)) for _ in range(16)]"
    result = get_command_result(shlex.join([sys.executable, "-c", code]), timeout_seconds=10)
    assert result.stdout_lines == ["out"] * 65536
    assert result.stderr == ("err\n" * 16384).strip() + "\n[stderr truncated]"


def test_sshare_keeps_scalar_rounding_for_mixed_huge_integers():
    tokens = ["9223372036854775808", "18446744073709551616", "0.1", "bad", "-1", "1e309"]
    frame = stat.get_sshare_df(
        [f"account_a|user{i}|1|{v}|{v}|{v}|{v}" for i, v in enumerate(tokens)]
    )
    expected = pandas.to_numeric(
        pandas.Series([pandas.to_numeric(v, errors="coerce") for v in tokens]), errors="coerce"
    )
    for column in ["norm_shares", "raw_usage", "effective_usage", "fairshare"]:
        pandas.testing.assert_series_equal(frame[column], expected.rename(column), check_exact=True)


@pytest.mark.parametrize("grid_engine", [False, True])
def test_memory_columns_keep_units_and_unknowns_across_snapshots(grid_engine):
    series = pandas.Series(["64G", "1000m", "invalid", ""] * 256)
    expected = pandas.Series(
        [64.0, 1000 * 1000**2 / 1024**3 if grid_engine else 1000 / 1024, float("nan"), float("nan")]
        * 256
    )
    convert = (
        memory.grid_engine_memory_series_to_gib if grid_engine else memory.memory_series_to_gib
    )
    pandas.testing.assert_series_equal(convert(series), expected)
    changed = series.copy()
    changed.iloc[0] = "128G"
    expected.iloc[0] = 128.0
    pandas.testing.assert_series_equal(convert(changed), expected)


@pytest.mark.parametrize("all_tiers", [False, True])
def test_render_and_launch_preserve_original_topology_attrs(capsys, all_tiers):
    nodes = stat.adjust_ram_unit(_node_frame())
    original = nodes.copy(deep=True)
    clean = nodes.copy(deep=False)
    clean.attrs = {}
    options = Namespace(ntop=1, all_tiers=all_tiers, exclude_abnormal_node=True)
    stat.print_slurm_compact_summary(clean, None, options)
    expected_output = capsys.readouterr().out
    stat.print_slurm_compact_summary(nodes, None, options)
    assert capsys.readouterr().out == expected_output
    pandas.testing.assert_frame_equal(
        stat.get_slurm_launch_heuristic_df(nodes, None),
        stat.get_slurm_launch_heuristic_df(clean, None),
        check_exact=True,
    )
    pandas.testing.assert_frame_equal(nodes, original, check_exact=True)
    assert nodes.attrs == original.attrs
    assert "slurm_cpus_per_core" in nodes.attrs


def test_reservation_expansion_preserves_order_and_unknown_targets():
    reservations = pandas.DataFrame(
        [
            {"reservation_name": "r1", "node_name": "*", "queue_name": "main"},
            {"reservation_name": "r2", "node_name": "*", "queue_name": "main"},
            {"reservation_name": "r3", "node_name": "*", "queue_name": "missing"},
            {"reservation_name": "r4", "node_name": "missing", "queue_name": "short"},
            {"reservation_name": "r5", "node_name": "node01", "queue_name": "short"},
        ]
    )
    expanded = stat._expand_reservation_rows(_node_frame(), reservations)
    assert expanded["reservation_name"].tolist() == ["r1", "r2", "r5"]
    assert expanded["node_name"].tolist() == ["node01"] * 3
    assert expanded.attrs == {
        "unresolved_targets": ["r3:*", "r4:missing"],
        "unresolved_partitions": ["missing", "short"],
    }


@pytest.mark.parametrize("response", [None, ["unrecognized"], [], ['{"job_info":{"jobs":[]}}']])
def test_uge_embedded_fallback_is_parsed_only_when_needed(monkeypatch, response):
    monkeypatch.setattr(stat, "get_command_stdout_lines", lambda **_: response)
    fallback = stat.get_user_df([])
    calls = []

    def embedded():
        calls.append(True)
        return fallback

    result, complete = stat._get_uge_all_user_jobs(
        Namespace(uge_job_command="qstat", uge_job_example_file=""), embedded, 1
    )
    expected_complete = response in ([], ['{"job_info":{"jobs":[]}}'])
    assert complete == expected_complete
    assert len(calls) == (0 if complete else 1)
    if not complete:
        assert result is fallback


def test_qfree_members_keep_first_seen_order_and_group_ambiguity():
    lines = [
        "\x1b[1mTHE NUMBER OF RUNNING JOBS BY USER IN THE GROUP (group_a)\x1b[0m",
        "QNAME user_b user_a user_b invalid!",
        "THE NUMBER OF MEM_REQ BY USER IN THE GROUP (group_a)",
        "QNAME user_c user_a",
    ]
    assert uge_parser._qfree_group_context(lines) == ("group_a", ["user_b", "user_a", "user_c"])
    lines.append("THE NUMBER OF MEM_REQ BY USER IN THE GROUP (group_b)")
    assert uge_parser._qfree_group_context(lines) == ("", ["user_b", "user_a", "user_c"])


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


@pytest.mark.parametrize("job_id", ["1", "١", "１", "²"])
def test_uge_job_row_filter_preserves_quality_with_noise_and_unicode_digits(job_id):
    frame = stat.get_user_df(
        iter(
            [
                "",
                " \t ",
                "  ----  ",
                "JOB-ID prior name user state submit/start at queue slots ja-task-ID",
                "synthetic scheduler warning",
                "12 malformed",
                f"\t{job_id} 0.5 job current_user qw 09/22/2026 00:00:00 2 1-10",
            ]
        )
    )
    assert frame["job_id"].tolist() == [job_id]
    assert frame["total_slots"].tolist() == [20]
    assert frame.attrs["input_nonempty"] is True
    assert frame.attrs["recognized_header"] is True
    assert frame.attrs["candidate_rows"] == 2
    assert frame.attrs["parse_quality"]["recognized_rows"] == 1
    assert frame.attrs["parse_quality"]["rejected_rows"] == 1


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
    # An exiting parent must exercise pipe cleanup, independent of Python startup
    # time under coverage or host contention. The outer five-second bound stays.
    command_timeout = 0.2 if parent_waits else 0
    runner_code = (
        "import json; from kfbatch.command import get_command_result; "
        "from kfbatch.errors import KFBatchCommandError\n"
        "try:\n"
        f"    result=get_command_result({command!r}, timeout_seconds={command_timeout}, "
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
        if not parent_waits:
            assert pid_file.exists()
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


def test_repeated_squeue_memory_validation_preserves_rejected_rows():
    memories = ["2Gc", "NA", "2Gn", "2GiB", "9" * 400 + "P", "0M", "2"]
    memories = memories + [f"{index}M" for index in range(300)] + memories
    lines = [
        f"{index}\tmain\tjob\tcurrent_user\taccount_a\tPD\t0:00\t1\t1\t{value}\t1:00:00\t(Priority)"
        for index, value in enumerate(memories)
    ]

    frame = stat.get_squeue_user_df(lines)

    invalid = {"NA", "2GiB", "9" * 400 + "P"}
    accepted = [index for index, value in enumerate(memories) if value not in invalid]
    assert frame["job_id"].tolist() == [str(index) for index in accepted]
    assert frame["req_mem"].tolist() == [memories[index] for index in accepted]
    assert frame.attrs["candidate_rows"] == len(lines)
    assert frame.attrs["recognized_rows"] == len(accepted)
    assert frame.attrs["rejected_rows"] == 6
    assert frame.attrs["parse_quality"]["rejected_rows"] == 6


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
