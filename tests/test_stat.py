import shlex
import sys
from types import SimpleNamespace

import pandas
import pytest

import kfbatch.command as command_module
import kfbatch.stat as stat_module
from kfbatch.command import decode_scheduler_output
from kfbatch.memory import floor_gib, memory_text_to_gib, slurm_request_memory_gib
from kfbatch.stat import (
    QSTAT_COLUMNS,
    SLURM_SQUEUE_PARSE_FIELDS,
    KFBatchCommandError,
    KFBatchUsageError,
    adjust_ram_unit,
    apply_slurm_reservations,
    get_command_stdout_lines,
    get_df,
    get_qfree_df,
    get_qstat_df,
    get_scheduler_from_command,
    get_scontrol_node_df,
    get_scontrol_reservation_df,
    get_slurm_fairshare_rank_summary,
    get_slurm_launch_heuristic_df,
    get_sprio_df,
    get_squeue_command_for_parsing,
    get_squeue_user_df,
    get_sshare_df,
    get_uge_json_job_df,
    get_user_df,
    mark_unresolved_slurm_reservations,
    print_queued_job_summary,
    print_slurm_compact_summary,
    print_slurm_fairshare_rank_summary,
    print_uge_compact_summary,
)


class OneShotIterable:
    def __init__(self, values):
        self.values = values
        self.started = False

    def __iter__(self):
        assert not self.started, "input was traversed more than once"
        self.started = True
        return iter(self.values)


def test_get_scheduler_from_command_accepts_full_path():
    assert get_scheduler_from_command("/usr/bin/squeue") == "slurm"
    assert get_scheduler_from_command("/opt/sge/bin/qstat -F") == "uge"


def test_get_squeue_command_for_parsing_adds_required_flags():
    command = get_squeue_command_for_parsing("squeue")
    tokens = shlex.split(command)
    assert tokens[0] == "squeue"
    assert "-h" in tokens
    assert "-o" in tokens
    assert SLURM_SQUEUE_PARSE_FIELDS in tokens


def test_get_squeue_command_for_parsing_overrides_explicit_format_equals():
    command = get_squeue_command_for_parsing("squeue --format=%i")
    tokens = shlex.split(command)
    assert "-h" in tokens
    assert not any(token.startswith("--format=") for token in tokens)
    assert "-o" in tokens
    assert SLURM_SQUEUE_PARSE_FIELDS in tokens


def test_get_squeue_command_for_parsing_overrides_short_o_attached():
    command = get_squeue_command_for_parsing("squeue -o%i")
    tokens = shlex.split(command)
    assert "-h" in tokens
    assert not any(token.startswith("-o") and token != "-o" for token in tokens)
    assert tokens.count("-o") == 1
    assert SLURM_SQUEUE_PARSE_FIELDS in tokens


def test_get_squeue_command_for_parsing_preserves_non_format_filters():
    command = get_squeue_command_for_parsing("squeue -u current_user -p epyc --format=%i")
    tokens = shlex.split(command)
    assert tokens[0] == "squeue"
    assert "-u" in tokens
    assert "current_user" in tokens
    assert "-p" in tokens
    assert "epyc" in tokens
    assert SLURM_SQUEUE_PARSE_FIELDS in tokens


def test_get_command_stdout_lines_empty_command_allow_failure():
    out = get_command_stdout_lines("", allow_failure=True, quiet_failure=True)
    assert out is None


def test_get_command_stdout_lines_empty_command_raises():
    with pytest.raises(KFBatchCommandError):
        get_command_stdout_lines("", allow_failure=False, quiet_failure=True)


def test_get_command_stdout_lines_missing_example_file_allow_failure():
    out = get_command_stdout_lines(
        "echo hi",
        example_file="/tmp/this_file_should_not_exist_for_kfbatch_tests",
        allow_failure=True,
        quiet_failure=True,
    )
    assert out is None


def test_get_command_stdout_lines_malformed_command_allow_failure():
    out = get_command_stdout_lines("'", allow_failure=True, quiet_failure=True)
    assert out is None


def test_get_command_stdout_lines_malformed_command_raises():
    with pytest.raises(KFBatchCommandError):
        get_command_stdout_lines("'", allow_failure=False, quiet_failure=True)


def test_get_command_stdout_lines_replaces_invalid_utf8(tmp_path):
    fixture = tmp_path / "invalid-utf8.txt"
    fixture.write_bytes(b"valid\ninvalid-\xff-name\n")
    lines = get_command_stdout_lines(
        "unused",
        example_file=str(fixture),
        command_name="fixture",
    )
    assert lines == ["valid", "invalid-\ufffd-name"]


def test_get_command_stdout_lines_spools_large_output(monkeypatch):
    monkeypatch.setattr(command_module, "STDOUT_SPOOL_MEMORY_LIMIT_BYTES", 16)
    code = "import sys; sys.stdout.write('first\\n' + 'x' * 64 + '\\nlast')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    lines = get_command_stdout_lines(command, command_name="large fixture")
    assert lines == ["first", "x" * 64, "last"]


def test_decode_scheduler_output_falls_back_from_unknown_locale(monkeypatch):
    monkeypatch.setattr(command_module.locale, "getpreferredencoding", lambda _do_setlocale: "x")
    assert decode_scheduler_output("日本語".encode()) == "日本語"


def test_get_command_stdout_lines_bounds_error_detail(monkeypatch):
    monkeypatch.setattr(command_module, "STDERR_DETAIL_LIMIT_BYTES", 8)
    code = "import sys; sys.stderr.write('0123456789'); raise SystemExit(2)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    with pytest.raises(KFBatchCommandError, match=r"01234567\n\[stderr truncated\]"):
        get_command_stdout_lines(command, command_name="failed fixture")


def test_get_command_stdout_lines_times_out():
    command = "{} -c {}".format(
        shlex.quote(sys.executable),
        shlex.quote("import time; time.sleep(1)"),
    )
    with pytest.raises(KFBatchCommandError, match="Timed out"):
        get_command_stdout_lines(
            command,
            command_name="slow fixture",
            timeout_seconds=0.01,
        )


def test_get_squeue_user_df_parses_literal_backslash_t():
    lines = [
        r"2001_[106-239%239]\tepyc\tanalysis_array\tuser_b\tPD\t0:00\t1\t(Priority)",
        "",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "job_id"] == "2001_[106-239%239]"
    assert df.at[0, "partition"] == "epyc"
    assert df.at[0, "total_slots"] == 134
    assert bool(df.at[0, "task_count_estimated"]) is False


def test_get_squeue_user_df_marks_truncated_array_as_estimated():
    lines = [
        "2001_[106-239%\tepyc\tanalysis_array\tuser_b\tPD\t0:00\t1\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "total_slots"] == 134
    assert bool(df.at[0, "task_count_estimated"]) is True


def test_get_squeue_user_df_parses_extended_slurm_fields():
    lines = [
        "2002\tepyc\tanalysis_one\tcurrent_user\taccount_a\tPD\t0:00\t1\t1\t1G\t00:05:00\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "account"] == "account_a"
    assert int(df.at[0, "req_cpus"]) == 1
    assert df.at[0, "req_mem"] == "1G"
    assert df.at[0, "time_limit"] == "00:05:00"
    assert df.at[0, "pending_reason"] == "Priority"
    assert bool(df.at[0, "resource_fields_complete"]) is True


def test_get_squeue_user_df_marks_legacy_slurm_fields_as_incomplete():
    lines = [
        "2002\tepyc\tanalysis_one\tcurrent_user\tPD\t0:00\t1\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "req_cpus"]) == 0
    assert df.at[0, "req_mem"] == ""
    assert bool(df.at[0, "resource_fields_complete"]) is False


def test_get_squeue_user_df_keeps_legacy_account_empty():
    lines = [
        "15243876\tepyc\twrap\tkfuku\tPD\t0:00\t1\t1\t1G\t00:05:00\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "account"] == ""
    assert df.at[0, "pending_reason"] == "Priority"


def test_get_scontrol_node_df_skips_nodes_without_partition_and_marks_reserved():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
        "NodeName=n2 Arch=x86_64 CPUAlloc=8 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=8000 State=MIXED+RESERVED Partitions=p1",
        "NodeName=n3 Arch=x86_64 CPUAlloc=0 CPUEfctv=8 CPUTot=8 RealMemory=16000 FreeMem=15000 State=IDLE Partitions=(null)",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP"})
    assert sorted(df["node_name"].tolist()) == ["n1", "n2"]
    assert df.loc[df["node_name"] == "n1", "status"].iloc[0] == ""
    assert df.loc[df["node_name"] == "n2", "status"].iloc[0] == ""
    marked = mark_unresolved_slurm_reservations(df, pandas.DataFrame())
    assert marked.loc[marked["node_name"] == "n2", "status"].iloc[0] == "reservation_unresolved"


def test_get_scontrol_node_df_marks_inactive_partition_as_abnormal():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "INACTIVE"})
    assert df.shape[0] == 1
    assert df.at[0, "status"] == "partition_state=INACTIVE"


@pytest.mark.parametrize("partition_state_map", [None, {}])
def test_get_scontrol_node_df_treats_unknown_partition_metadata_as_abnormal(
    partition_state_map,
):
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 "
        "RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map=partition_state_map)
    assert df.shape[0] == 1
    assert df.at[0, "status"] == "partition_state=UNKNOWN"


def test_get_scontrol_node_df_treats_lowercase_up_as_up():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "up"})
    assert df.shape[0] == 1
    assert df.at[0, "status"] == ""


def test_get_scontrol_node_df_treats_up_star_as_up():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP*"})
    assert df.shape[0] == 1
    assert df.at[0, "status"] == ""


def test_get_scontrol_node_df_marks_up_plus_drain_partition_as_abnormal():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP+DRAIN"})
    assert df.shape[0] == 1
    assert df.at[0, "status"] == "partition_state=UP+DRAIN"


def test_get_scontrol_node_df_clips_negative_available_core_count():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=20 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP"})
    assert df.shape[0] == 1
    assert int(df.at[0, "ncore_available"]) == 0


def test_get_scontrol_node_df_uses_schedulable_memory_over_free_mem():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 AllocMem=28000 FreeMem=31000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP"})
    assert df.shape[0] == 1
    assert df.at[0, "hc:mem_req"] == "4000M"


def test_get_scontrol_node_df_preserves_unknown_memory():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP"})
    assert df.shape[0] == 1
    assert pandas.isna(df.at[0, "hc:mem_req"])
    assert pandas.isna(df.at[0, "hl:mem_total"])
    assert bool(df.at[0, "hc:mem_req_known"]) is False
    assert bool(df.at[0, "hl:mem_total_known"]) is False


def test_get_scontrol_reservation_df_counts_explicit_core_ids_and_single_node_fallback():
    lines = [
        "ReservationName=r1 StartTime=2026-03-06T12:00:00 EndTime=2026-03-07T12:00:00 Duration=1-00:00:00",
        "Nodes=node20 NodeCnt=1 CoreCnt=8 PartitionName=epyc Flags=IGNORE_JOBS State=ACTIVE TRES=cpu=8,mem=64G,node=1,billing=8",
        "NodeName=node20 CoreIDs=0-2,4,6-8",
        "",
        "ReservationName=r2 StartTime=2026-03-06T12:00:00 EndTime=2026-03-07T12:00:00 Duration=1-00:00:00",
        "Nodes=node21 NodeCnt=1 CoreCnt=6 PartitionName=epyc Flags=IGNORE_JOBS State=ACTIVE",
        "NodeName=node21 CoreIDs=(null)",
    ]
    df = get_scontrol_reservation_df(lines)
    assert df.shape[0] == 2
    assert int(df.loc[df["node_name"] == "node20", "reserved_cores"].iloc[0]) == 7
    assert int(df.loc[df["node_name"] == "node20", "reserved_mem_mb"].iloc[0]) == 65536
    assert int(df.loc[df["node_name"] == "node21", "reserved_cores"].iloc[0]) == 6


def test_get_scontrol_reservation_df_parses_multiline_hostlists_and_access():
    lines = [
        "ReservationName=active_hold StartTime=2026-07-30T09:00:00",
        " Nodes=node[01-02] NodeCnt=2 CoreCnt=16",
        " PartitionName=epyc TRES=cpu=16,mem=32G,node=2",
        " Users=current_user State=ACTIVE",
        "",
        "ReservationName=old_hold Nodes=node03 NodeCnt=1 CoreCnt=4",
        " PartitionName=epyc TRES=cpu=4,mem=4G,node=1",
        " Users=other_user State=INACTIVE",
    ]
    df = get_scontrol_reservation_df(lines, current_user="current_user")
    assert df["node_name"].tolist() == ["node01", "node02"]
    assert df["reserved_cores"].tolist() == [8, 8]
    assert df["reserved_mem_mb"].tolist() == [16384, 16384]
    assert df["accessible"].tolist() == [True, True]


def test_get_scontrol_reservation_df_expands_compound_hostlists():
    lines = [
        "ReservationName=compound Nodes=rack[1-2]n[01-02] NodeCnt=4 CoreCnt=8 "
        "PartitionName=p1 Users=other State=ACTIVE",
    ]
    df = get_scontrol_reservation_df(lines, current_user="current_user")
    assert df["node_name"].tolist() == [
        "rack1n01",
        "rack1n02",
        "rack2n01",
        "rack2n02",
    ]
    assert df["reserved_cores"].tolist() == [2, 2, 2, 2]


def test_reservation_access_requires_every_configured_dimension():
    lines = [
        "ReservationName=restricted Nodes=n1 NodeCnt=1 CoreCnt=4 "
        "PartitionName=p1 Users=ALL Accounts=project_b State=ACTIVE",
    ]
    denied = get_scontrol_reservation_df(
        lines,
        current_user="current_user",
        current_accounts={"project_a"},
    )
    allowed = get_scontrol_reservation_df(
        lines,
        current_user="current_user",
        current_accounts={"project_b"},
    )
    assert denied.at[0, "accessible"] == False  # noqa: E712
    assert allowed.at[0, "accessible"] == True  # noqa: E712


def test_reservation_access_is_conservative_for_specific_qos_and_supports_deny_lists():
    qos_lines = [
        "ReservationName=qos Nodes=n1 NodeCnt=1 CoreCnt=4 "
        "PartitionName=p1 Users=ALL QOS=urgent State=ACTIVE",
    ]
    denied_user_lines = [
        "ReservationName=deny Nodes=n1 NodeCnt=1 CoreCnt=4 "
        "PartitionName=p1 Users=-blocked State=ACTIVE",
    ]
    qos = get_scontrol_reservation_df(qos_lines, current_user="current_user")
    blocked = get_scontrol_reservation_df(denied_user_lines, current_user="blocked")
    allowed = get_scontrol_reservation_df(denied_user_lines, current_user="current_user")
    assert qos.at[0, "accessible"] == False  # noqa: E712
    assert blocked.at[0, "accessible"] == False  # noqa: E712
    assert allowed.at[0, "accessible"] == True  # noqa: E712


def test_get_scontrol_reservation_df_normalizes_null_partition():
    lines = [
        "ReservationName=global Nodes=n1 NodeCnt=1 CoreCnt=4 "
        "PartitionName=(null) Users=other State=ACTIVE",
    ]
    df = get_scontrol_reservation_df(lines, current_user="current_user")
    assert df.at[0, "queue_name"] == ""


def test_apply_slurm_reservations_subtracts_partial_reservations_and_estimated_memory():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node20"],
            "ncore_resv": [0],
            "ncore_available": [32],
            "ncore_total": [64],
            "hl:mem_total": ["64000M"],
            "hc:mem_req": ["32000M"],
            "status": [""],
        }
    )
    df_reservation = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node20"],
            "reservation_name": ["r1"],
            "reserved_cores": [6],
            "reserved_mem_mb": [0],
        }
    )
    out = apply_slurm_reservations(df_node, df_reservation)
    assert int(out.at[0, "ncore_resv"]) == 6
    assert int(out.at[0, "ncore_available"]) == 26
    assert int(out.at[0, "reservation_cores"]) == 6
    assert int(out.at[0, "reservation_mem_mb"]) == 6000
    assert out.at[0, "hc:mem_req"] == "26000M"


def test_apply_slurm_reservations_uses_explicit_reserved_memory_when_available():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node20"],
            "ncore_resv": [0],
            "ncore_available": [32],
            "ncore_total": [64],
            "hl:mem_total": ["64000M"],
            "hc:mem_req": ["32000M"],
            "status": [""],
        }
    )
    df_reservation = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node20"],
            "reservation_name": ["r1"],
            "reserved_cores": [6],
            "reserved_mem_mb": [14000],
        }
    )
    out = apply_slurm_reservations(df_node, df_reservation)
    assert int(out.at[0, "reservation_mem_mb"]) == 14000
    assert out.at[0, "hc:mem_req"] == "18000M"


def test_apply_slurm_reservations_subtracts_from_every_partition_alias():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["p1", "p2"],
            "node_name": ["node20", "node20"],
            "ncore_resv": [0, 0],
            "ncore_available": [32, 32],
            "ncore_total": [64, 64],
            "hl:mem_total": ["64000M", "64000M"],
            "hc:mem_req": ["32000M", "32000M"],
            "status": ["", ""],
        }
    )
    df_reservation = pandas.DataFrame(
        {
            "queue_name": ["p1"],
            "node_name": ["node20"],
            "reservation_name": ["r1"],
            "reserved_cores": [8],
            "reserved_mem_mb": [8000],
            "accessible": [False],
        }
    )
    out = apply_slurm_reservations(df_node, df_reservation)
    assert out["ncore_available"].tolist() == [24, 24]
    assert out["hc:mem_req"].tolist() == ["24000M", "24000M"]


def test_apply_slurm_reservations_expands_all_nodes_within_partition():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["p1", "p1", "p2"],
            "node_name": ["n1", "n2", "n3"],
            "ncore_resv": [0, 0, 0],
            "ncore_available": [8, 8, 8],
            "ncore_total": [8, 8, 8],
            "hl:mem_total": ["8000M", "8000M", "8000M"],
            "hc:mem_req": ["8000M", "8000M", "8000M"],
            "status": ["", "", ""],
        }
    )
    df_reservation = get_scontrol_reservation_df(
        [
            "ReservationName=all_p1 Nodes=ALL NodeCnt=2 CoreCnt=0 "
            "PartitionName=p1 Users=other State=ACTIVE",
        ],
        current_user="current_user",
    )
    out = apply_slurm_reservations(df_node, df_reservation)
    assert out["ncore_available"].tolist() == [0, 0, 8]
    assert out["hc:mem_req"].tolist() == ["0M", "0M", "8000M"]


def test_get_sprio_df_parses_pending_priority_table():
    lines = [
        "          JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE  PARTITION",
        "           2002 epyc           12721          0          0       2708         14      10000",
    ]
    df = get_sprio_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "job_id"] == "2002"
    assert int(df.at[0, "priority"]) == 12721
    assert int(df.at[0, "fairshare"]) == 2708


def test_get_sprio_df_parses_stable_pipe_format():
    df = get_sprio_df(["2002|epyc|12721|0|0|2708|14|10000"])
    assert df.shape[0] == 1
    assert df.at[0, "partition"] == "epyc"
    assert int(df.at[0, "priority"]) == 12721
    assert int(df.at[0, "fairshare"]) == 2708


def test_get_sshare_df_parses_pipe_output_and_skips_account_summary():
    lines = [
        "Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare",
        "general_analysis||1|0.040000|1000000|0.800000|",
        " general_analysis|kfuku|1|0.000429|41192861|0.015753|0.005691",
    ]
    df = get_sshare_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "account"] == "general_analysis"
    assert df.at[0, "user"] == "kfuku"
    assert float(df.at[0, "fairshare"]) == pytest.approx(0.005691)


def test_get_slurm_fairshare_rank_summary_reports_overall_and_pending_ranks():
    df_job = pandas.DataFrame(
        {
            "job_id": ["1", "2", "3"],
            "user": ["alice", "kfuku", "zeta"],
            "account": ["general_analysis", "general_analysis", "general_analysis"],
            "state": ["PD", "PD", "R"],
        }
    )
    df_share = get_sshare_df(
        [
            "Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare",
            " general_analysis|alice|1|0.1|100|0.001|0.900000",
            " general_analysis|bob|1|0.1|200|0.002|0.500000",
            " general_analysis|kfuku|1|0.1|300|0.003|0.005691",
            " general_analysis|zeta|1|0.1|400|0.004|0.001000",
        ]
    )
    summary = get_slurm_fairshare_rank_summary(
        df_job=df_job, df_share=df_share, current_user="kfuku"
    )
    assert summary["overall_rank"] == 3
    assert summary["overall_total"] == 4
    assert summary["pending_rank"] == 2
    assert summary["pending_total"] == 2


def test_print_slurm_fairshare_rank_summary_uses_compact_single_line(capsys):
    print_slurm_fairshare_rank_summary(
        {
            "account": "general_analysis",
            "fairshare": 0.005691,
            "overall_rank": 41,
            "overall_total": 52,
            "pending_rank": 12,
            "pending_total": 20,
            "pending_missing": 0,
        }
    )
    out = capsys.readouterr().out
    assert "fairshare  self=0.005691" in out
    assert "account=general_analysis" in out
    assert "assoc_rank=41/52" in out
    assert "pending_assoc_rank=12/20" in out


def test_fairshare_summary_discloses_multi_account_selection(capsys):
    df_share = get_sshare_df(
        [
            "Account|User|RawShares|NormShares|RawUsage|EffectvUsage|FairShare",
            " account_a|current_user|1|0.1|10|0.1|0.400000",
            " account_b|current_user|1|0.1|20|0.2|0.600000",
        ]
    )
    summary = get_slurm_fairshare_rank_summary(
        df_job=pandas.DataFrame(),
        df_share=df_share,
        current_user="current_user",
    )
    assert summary["account"] == "account_b"
    assert summary["association_count"] == 2
    print_slurm_fairshare_rank_summary(summary)
    assert "selected=best_of_2_associations" in capsys.readouterr().out


def test_get_slurm_launch_heuristic_keeps_resource_ceiling_when_priority_blocks_even_tiny_job():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node04"],
            "status": [""],
            "ncore_available": [67],
            "hc:mem_req": ["925G"],
        }
    )
    df_job = pandas.DataFrame(
        {
            "job_id": ["2002"],
            "partition": ["epyc"],
            "user": ["current_user"],
            "state": ["PD"],
            "req_cpus": [1],
            "req_mem": ["1G"],
            "time_limit": ["00:05:00"],
            "pending_reason": ["Priority"],
            "resource_fields_complete": [True],
        }
    )
    df_prio = pandas.DataFrame(
        {
            "job_id": ["2002", "topjob"],
            "partition": ["epyc", "epyc"],
            "priority": [12721, 16652],
            "fairshare": [2708, 6634],
        }
    )
    out = get_slurm_launch_heuristic_df(
        df_node=df_node,
        df_job=df_job,
        df_prio=df_prio,
        current_user="current_user",
    )
    assert out.shape[0] == 1
    assert int(out.at[0, "recommended_cores"]) == 67
    assert float(out.at[0, "recommended_mem_gb"]) == 925.0
    assert float(out.at[0, "recommended_mem_gib"]) == 925.0
    assert out.at[0, "status"] == "priority_blocked"
    assert int(out.at[0, "priority_gap"]) == 3931
    assert int(out.at[0, "fairshare_gap"]) == 3926
    assert int(out.at[0, "top_node_cores"]) == 67


def test_get_slurm_launch_heuristic_matches_multi_partition_pending_jobs():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node04"],
            "status": [""],
            "ncore_available": [128],
            "hc:mem_req": ["516G"],
        }
    )
    df_job = pandas.DataFrame(
        {
            "job_id": ["2002"],
            "partition": ["medium,rome,epyc"],
            "user": ["current_user"],
            "state": ["PD"],
            "req_cpus": [1],
            "req_mem": ["1G"],
            "time_limit": ["00:05:00"],
            "pending_reason": ["Priority"],
            "resource_fields_complete": [True],
        }
    )
    df_prio = pandas.DataFrame(
        {
            "job_id": ["2002", "topjob"],
            "partition": ["medium,rome,epyc", "epyc"],
            "priority": [14951, 16652],
            "fairshare": [5597, 6634],
        }
    )
    out = get_slurm_launch_heuristic_df(
        df_node=df_node,
        df_job=df_job,
        df_prio=df_prio,
        current_user="current_user",
    )
    assert out.shape[0] == 1
    assert out.at[0, "status"] == "priority_blocked"
    assert int(out.at[0, "recommended_cores"]) == 128
    assert float(out.at[0, "recommended_mem_gb"]) == 516.0
    assert float(out.at[0, "recommended_mem_gib"]) == 516.0
    assert int(out.at[0, "blocked_req_cores"]) == 1
    assert int(out.at[0, "priority_gap"]) == 1701
    assert int(out.at[0, "fairshare_gap"]) == 1037


def test_get_slurm_launch_heuristic_keeps_resource_ceiling_without_zero_sized_request_for_legacy_rows():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["node04"],
            "status": [""],
            "ncore_available": [67],
            "hc:mem_req": ["925G"],
        }
    )
    df_job = pandas.DataFrame(
        {
            "job_id": ["2002"],
            "partition": ["epyc"],
            "user": ["current_user"],
            "state": ["PD"],
            "req_cpus": [0],
            "req_mem": [""],
            "time_limit": [""],
            "pending_reason": ["Priority"],
            "resource_fields_complete": [False],
        }
    )
    out = get_slurm_launch_heuristic_df(
        df_node=df_node,
        df_job=df_job,
        current_user="current_user",
    )
    assert out.shape[0] == 1
    assert int(out.at[0, "recommended_cores"]) == 67
    assert pandas.isna(out.at[0, "blocked_req_cores"])
    assert out.at[0, "status"] == "priority_blocked_missing_fields"


def test_get_user_df_counts_uge_array_with_commas_and_ranges():
    lines = [
        "  123 0.555 test user qw 02/12/2026 12:00:00 4 1,2,4-8:2",
    ]
    df = get_user_df(lines)
    assert df.shape[0] == 1
    # tasks = 1 + 1 + 3 = 5
    assert int(df.at[0, "total_slots"]) == 20


def test_get_user_df_counts_uge_range_without_step():
    lines = [
        "  124 0.111 test user qw 02/12/2026 12:00:00 2 10-12",
    ]
    df = get_user_df(lines)
    assert df.shape[0] == 1
    # tasks = 3
    assert int(df.at[0, "total_slots"]) == 6


def test_get_user_df_handles_no_job_lines():
    df = get_user_df(["queuename qtype", "----"])
    assert df.shape[0] == 0
    assert "total_slots" in df.columns


def test_get_user_df_parses_modern_age_running_rows_with_queue_and_blank_jclass():
    lines = [
        " 1001 0.50004 analysis_a user1 r 07/30/2026 10:08:54 mjobs.q@compute07 20",
        " 1002 0.00000 analysis_b user2 qw 07/30/2026 10:10:43 1",
    ]
    df = get_user_df(lines)
    assert df.shape[0] == 2
    assert df.at[0, "queue_name"] == "mjobs.q"
    assert int(df.at[0, "total_slots"]) == 20
    assert df.at[1, "queue_name"] == ""
    assert int(df.at[1, "total_slots"]) == 1


def test_get_uge_json_job_df_parses_running_pending_and_expanded_tasks():
    lines = [
        '{"queue_info":[{"running jobs":['
        '{"JB_job_number":1,"JAT_prio":0.5,"JB_name":"run","JB_owner":"me",'
        '"state":"r","JAT_start_time":"2026-07-30T10:00:00","queue_name":"mjobs.q@n1","slots":4},'
        '{"JB_job_number":1,"JAT_prio":0.5,"JB_name":"run","JB_owner":"me",'
        '"state":"r","JAT_start_time":"2026-07-30T10:00:00","queue_name":"mjobs.q@n2","slots":4}'
        ']}],"job_info":[{"pending jobs":['
        '{"JB_job_number":2,"JAT_prio":0,"JB_name":"wait","JB_owner":"other",'
        '"state":"hqw","JB_submission_time":"2026-07-30T10:01:00","queue_name":"","slots":2}'
        "]}]}"
    ]
    df = get_uge_json_job_df(lines)
    assert df.shape[0] == 3
    assert int(df.loc[df["state"] == "r", "total_slots"].sum()) == 8
    assert int(df.loc[df["state"] == "hqw", "total_slots"].sum()) == 2
    assert df.loc[df["state"] == "r", "queue_name"].unique().tolist() == ["mjobs.q"]
    assert not df.loc[df["state"] == "r", "task_count_estimated"].any()
    assert df.loc[df["state"] == "hqw", "task_count_estimated"].all()


def test_get_uge_json_job_df_marks_omitted_array_range_as_estimated():
    lines = [
        '{"job_info":[{"pending jobs":[{"JB_job_number":1002,'
        '"JB_name":"analysis_b","JB_owner":"user_b","state":"Rq","slots":2}]}]}'
    ]
    df = get_uge_json_job_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "total_slots"]) == 2
    assert bool(df.at[0, "task_count_estimated"]) is True


def test_get_uge_json_job_df_returns_none_for_non_json():
    assert get_uge_json_job_df(["not json"]) is None


def test_get_qfree_df_parses_slot_and_memory_summaries():
    lines = [
        "SUMMARY OF RUNNING JOBS",
        "                    me       grp  QUOTA      ALL",
        "        QNAME       JOBS      JOBS  LIMIT     JOBS AVAIL STDBY   TOTAL",
        "------------- -------------------- ------ ----------------------------",
        "      mjobs.q          3        10   1024     3828   120   192    5376",
        "       lmem.q          0         1      0      381   195     0     576",
        "THE NUMBER OF RUNNING JOBS BY USER IN THE GROUP (grp)",
        "====================== QUOTA BY MEM_REQ =====================",
        "SUMMARY OF RUNNING JOBS ( MEM_REQ )",
        "                    me       grp  QUOTA      ALL",
        "        QNAME    MEM_REQ   MEM_REQ  LIMIT  MEM_REQ AVAIL STDBY   TOTAL",
        "------------- -------------------- ------ ----------------------------",
        "      mjobs.q          6        20      -    33221   120   192   34330",
        "       lmem.q          0         4      -    30001   195     0   46080",
        "THE NUMBER OF MEM_REQ BY USER IN THE GROUP (grp)",
    ]
    df = get_qfree_df(lines)
    assert df["queue_name"].tolist() == ["mjobs.q", "lmem.q"]
    mjobs = df.loc[df["queue_name"] == "mjobs.q"].iloc[0]
    assert int(mjobs["self_slots"]) == 3
    assert int(mjobs["quota_slots"]) == 1024
    assert int(mjobs["available_slots_2g"]) == 120
    assert int(mjobs["standby_slots"]) == 192
    assert int(mjobs["all_mem_req_gb"]) == 33221
    assert pandas.isna(mjobs["quota_mem_gb"])


def test_get_qstat_df_includes_last_node():
    lines = [
        "epyc.q@node01 BP 0/1/2 0.10 lx-amd64",
        "\\thc:mem_req=4.000G",
        "\\thl:mem_total=8.000G",
    ]
    df = get_qstat_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "ncore_available"]) == 1
    assert df.at[0, "queue_name"] == "epyc.q"
    assert df.at[0, "node_name"] == "node01"


def test_get_qstat_df_skips_malformed_header_lines():
    lines = [
        "this is malformed",
        "epyc.q@node01 BP 0/1/2 0.10 lx-amd64",
        "\\thc:mem_req=4.000G",
        "\\thl:mem_total=8.000G",
    ]
    df = get_qstat_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "node_name"] == "node01"


def test_get_qstat_df_returns_empty_table_for_unparseable_input():
    df = get_qstat_df(["garbage line", "### comment"])
    assert df.shape[0] == 0
    assert "ncore_available" in df.columns
    assert "hc:mem_req" in df.columns
    assert "hl:mem_total" in df.columns


def test_get_qstat_df_ignores_orphan_tab_lines():
    lines = [
        "\\thc:mem_req=4.000G",
        "\\thl:mem_total=8.000G",
    ]
    df = get_qstat_df(lines)
    assert df.shape[0] == 0
    assert "ncore_available" in df.columns


def test_get_qstat_df_clips_negative_available_and_preserves_unknown_memory():
    lines = [
        "epyc.q@node01 BP 2/4/5 0.10 lx-amd64",
        "\thl:mem_total=8.000G",
    ]
    df = get_qstat_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "ncore_available"]) == 0
    assert pandas.isna(df.at[0, "hc:mem_req"])
    assert bool(df.at[0, "hc:mem_req_known"]) is False
    assert df.at[0, "hl:mem_total"] == "8.000G"


def test_get_qstat_df_streams_once_and_discards_unneeded_resources():
    lines = OneShotIterable(
        [
            "epyc.q@node01 BP 0/1/2 0.10 lx-amd64",
            "\thc:mem_req=4.000G",
            "\thl:mem_total=8.000G",
            "\thl:unused_large_resource=123456",
        ]
    )
    df = get_qstat_df(lines)
    assert list(df.columns) == QSTAT_COLUMNS
    assert df.at[0, "hc:mem_req"] == "4.000G"
    assert "hl:unused_large_resource" not in df.columns


def test_adjust_ram_unit_converts_memory_to_binary_gib_consistently():
    df = pandas.DataFrame(
        {
            "hc:mem_req": ["500M", "1.5G", "2T"],
            "hl:mem_total": ["1000M", "4G", "1T"],
        }
    )
    out = adjust_ram_unit(df)
    assert out.at[0, "hc:mem_req"] == 500 / 1024
    assert out.at[0, "hc:mem_req_unit"] == "GiB"
    assert out.at[1, "hc:mem_req"] == 1.5
    assert out.at[2, "hc:mem_req"] == 2048.0
    assert out.at[2, "hl:mem_total"] == 1024.0


def test_memory_parser_supports_pebibyte_suffix():
    assert memory_text_to_gib("1P") == 1024.0 * 1024.0


def test_adjust_ram_unit_handles_lowercase_and_invalid_values():
    df = pandas.DataFrame(
        {
            "hc:mem_req": ["500m", "bad", ""],
            "hl:mem_total": ["1t", "2g", None],
        }
    )
    out = adjust_ram_unit(df)
    assert out.at[0, "hc:mem_req"] == 500 / 1024
    assert out.at[0, "hc:mem_req_unit"] == "GiB"
    assert pandas.isna(out.at[1, "hc:mem_req"])
    assert out.at[1, "hc:mem_req_unit"] == ""
    assert out.at[0, "hl:mem_total"] == 1024.0
    assert out.at[1, "hl:mem_total"] == 2.0
    assert pandas.isna(out.at[2, "hl:mem_total"])
    assert out.at[2, "hl:mem_total_unit"] == ""


def test_slurm_per_cpu_memory_and_display_floor_use_binary_units():
    assert slurm_request_memory_gib("2Gc", req_cpus=6, num_nodes=2) == 6.0
    assert memory_text_to_gib("1535M") == 1535 / 1024
    assert floor_gib(memory_text_to_gib("1535M")) == 1


def test_print_queued_job_summary_slurm_accepts_long_state_names(capsys):
    df_user = pandas.DataFrame(
        {
            "user": ["current_user", "other", "current_user"],
            "state": ["RUNNING", "PENDING", "FAILED"],
            "total_slots": [2, 3, 4],
            "task_count_estimated": [False, False, False],
        }
    )
    print_queued_job_summary(df_user, scheduler="slurm", current_user="current_user")
    out = capsys.readouterr().out
    assert "jobs  self:R/Q/X=2/0/4  all:R/Q/X=2/3/4" in out


def test_print_queued_job_summary_uge_matches_slurm_style(capsys):
    df_user = pandas.DataFrame(
        {
            "user": ["me", "other", "me", "other"],
            "state": ["r", "hqw", "Eqw", "s"],
            "queue_name": ["mjobs.q", "", "", "web.q"],
            "total_slots": [4, 3, 2, 1],
        }
    )
    print_queued_job_summary(df_user, scheduler="uge", current_user="me")
    out = capsys.readouterr().out
    assert "jobs  self:R/Q/F=4/0/2  all:R/Q/F=5/3/2" in out


def test_print_queued_job_summary_uge_treats_rq_and_hrq_as_pending(capsys):
    df_user = pandas.DataFrame(
        {
            "user": ["me", "me", "me", "me"],
            "state": ["Rq", "hRq", "r", "dr"],
            "queue_name": ["", "", "mjobs.q", "mjobs.q"],
            "total_slots": [5, 3, 2, 1],
        }
    )
    print_queued_job_summary(df_user, scheduler="uge", current_user="me")
    out = capsys.readouterr().out
    assert "jobs  self:R/Q/F=2/8/1  all:R/Q/F=2/8/1" in out


def test_print_queued_job_summary_uge_marks_partial_job_source(capsys):
    df_user = pandas.DataFrame(
        {
            "user": ["me"],
            "state": ["r"],
            "queue_name": ["mjobs.q"],
            "total_slots": [4],
        }
    )
    print_queued_job_summary(
        df_user,
        scheduler="uge",
        current_user="me",
        all_users=False,
    )
    out = capsys.readouterr().out
    assert "jobs  observed:R/Q/F=4/0/0  (all-user status unavailable)" in out


def test_print_slurm_compact_summary_uses_single_row_per_partition(capsys):
    df = pandas.DataFrame(
        {
            "queue_name": ["epyc", "epyc", "rome"],
            "node_name": ["node04", "node17", "node41"],
            "status": ["", "", ""],
            "ncore_available": [14, 0, 103],
            "ncore_used": [178, 81, 25],
            "ncore_total": [192, 81, 128],
            "ncore_resv": [0, 0, 0],
            "hc:mem_req": [5.0, 352.0, 12.0],
            "hl:mem_total": [1548.0, 376.0, 516.0],
        }
    )
    df_launch = pandas.DataFrame(
        {
            "queue_name": ["epyc", "rome"],
            "recommended_cores": [14, 103],
            "recommended_mem_gib": [5.0, 12.0],
            "top_node_name": ["node04", "node41"],
            "top_node_cores": [14, 103],
            "top_node_mem_gib": [5.0, 12.0],
            "priority_gap": [18097, 1701],
            "fairshare_gap": [17960, 1037],
            "blocked_req_cores": [1, 1],
            "blocked_req_mem_gib": [1.0, 1.0],
            "blocked_time_limit": ["5:00", "5:00"],
            "status": ["priority_blocked", "priority_blocked"],
        }
    )
    args = SimpleNamespace(exclude_abnormal_node=True)
    print_slurm_compact_summary(df, df_launch, args)
    out = capsys.readouterr().out
    assert "part" in out
    assert "nodes" in out
    assert "cpu(a/u/t)" in out
    assert "epyc" in out
    assert "node04 14c/5GiB" in out
    assert "node17 0c/352GiB" in out
    assert "<=14c/5GiB PRIO min=1c/1GiB/5m gap=18097 fs=17960" in out
    assert "<=103c/12GiB PRIO min=1c/1GiB/5m gap=1701 fs=1037" in out
    assert (
        "legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total" in out
    )


def test_print_uge_compact_summary_uses_qfree_queue_filter_and_quota(capsys):
    df = pandas.DataFrame(
        {
            "queue_name": ["mjobs.q", "mjobs.q", "private.q"],
            "node_name": ["n1", "n2", "n3"],
            "status": ["", "d", ""],
            "ncore_available": [12, 0, 8],
            "ncore_used": [4, 0, 0],
            "ncore_total": [16, 16, 8],
            "ncore_resv": [0, 0, 0],
            "hc:mem_req": [32.0, 0.0, 16.0],
            "hl:mem_total": [64.0, 64.0, 32.0],
        }
    )
    df_qfree = pandas.DataFrame(
        {
            "queue_name": ["mjobs.q"],
            "self_slots": [2],
            "group_slots": [10],
            "quota_slots": [1024],
            "available_slots_2g": [12],
            "standby_slots": [16],
            "all_mem_req_gb": [40],
            "total_mem_gb": [100],
        }
    )
    args = SimpleNamespace(exclude_abnormal_node=True)
    print_uge_compact_summary(df, df_qfree, args)
    out = capsys.readouterr().out
    assert "queue" in out
    assert "mjobs.q" in out
    assert "private.q" in out
    assert "1/1/2" in out
    assert "12/4/32" in out
    assert "60/100" in out
    assert "2/10/1024" in out
    assert "12(+16s)" in out
    assert "launch2G=immediate 2G slots (+standby)" in out


def test_get_df_qstat_requires_niter_at_least_one():
    args = SimpleNamespace(stat_command="qstat -F", niter=0)
    with pytest.raises(KFBatchUsageError):
        get_df(args)


def test_get_df_qstat_merges_memory_numerically(monkeypatch):
    first_lines = [
        "epyc.q@node01 BP 0/0/2 0.10 lx-amd64",
        "\thc:mem_req=1200M",
        "\thl:mem_total=2000M",
    ]
    second_lines = [
        "epyc.q@node01 BP 0/0/2 0.10 lx-amd64",
        "\thc:mem_req=1G",
        "\thl:mem_total=2000M",
    ]
    line_sets = [first_lines, second_lines]
    call_index = {"i": 0}

    def fake_get_command_stdout_lines(**kwargs):
        i = call_index["i"]
        call_index["i"] += 1
        return line_sets[i]

    monkeypatch.setattr(stat_module, "get_command_stdout_lines", fake_get_command_stdout_lines)
    args = SimpleNamespace(stat_command="qstat -F", niter=2, example_file="")
    scheduler, df, _ = get_df(args)
    assert scheduler == "uge"
    assert df.shape[0] == 1
    assert df.at[0, "hc:mem_req"] == "1.000G"


def test_merge_qstat_snapshot_marks_disappearing_nodes_unavailable():
    first = get_qstat_df(
        [
            "mjobs.q@node01 BP 0/0/2 0.10 lx-amd64",
            "\thc:mem_req=2G",
            "\thl:mem_total=8G",
            "mjobs.q@node02 BP 0/0/2 0.10 lx-amd64",
            "\thc:mem_req=2G",
            "\thl:mem_total=8G",
        ]
    )
    second = get_qstat_df(
        [
            "mjobs.q@node01 BP 0/1/2 0.10 lx-amd64",
            "\thc:mem_req=1G",
            "\thl:mem_total=8G",
        ]
    )
    merged = stat_module._merge_qstat_iteration_min_availability(first, second)
    node02 = merged.loc[merged["node_name"] == "node02"].iloc[0]
    assert int(node02["ncore_available"]) == 0
    assert pandas.isna(node02["hc:mem_req"])
    assert "missing_in_snapshot" in node02["status"]


def test_merge_qstat_snapshots_preserves_any_abnormal_status_and_unknown_memory():
    first = get_qstat_df(
        [
            "mjobs.q@node01 BP 0/0/2 0.10 lx-amd64 d",
            "\thc:mem_req=2G",
            "\thl:mem_total=8G",
        ]
    )
    second = get_qstat_df(
        [
            "mjobs.q@node01 BP 0/0/2 0.10 lx-amd64",
            "\thl:mem_total=8G",
        ]
    )
    merged = stat_module._merge_qstat_iteration_min_availability(first, second)
    assert "d" in merged.at[0, "status"]
    assert pandas.isna(merged.at[0, "hc:mem_req"])
    assert bool(merged.at[0, "hc:mem_req_known"]) is False


def test_get_df_qstat_handles_empty_later_iteration(monkeypatch):
    first_lines = [
        "epyc.q@node01 BP 0/0/2 0.10 lx-amd64",
        "\thc:mem_req=2G",
        "\thl:mem_total=8G",
    ]
    second_lines = ["garbage line"]
    line_sets = [first_lines, second_lines]
    call_index = {"i": 0}

    def fake_get_command_stdout_lines(**kwargs):
        i = call_index["i"]
        call_index["i"] += 1
        return line_sets[i]

    monkeypatch.setattr(stat_module, "get_command_stdout_lines", fake_get_command_stdout_lines)
    args = SimpleNamespace(stat_command="qstat -F", niter=2, example_file="")
    with pytest.raises(KFBatchCommandError, match="snapshot 2"):
        get_df(args)
