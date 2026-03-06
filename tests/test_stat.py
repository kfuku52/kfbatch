import shlex
from types import SimpleNamespace

import pandas
import pytest

import kfbatch.stat as stat_module
from kfbatch.stat import (
    KFBatchCommandError,
    KFBatchUsageError,
    SLURM_SQUEUE_PARSE_FIELDS,
    adjust_ram_unit,
    apply_slurm_reservations,
    get_command_stdout_lines,
    get_df,
    get_qstat_df,
    get_scheduler_from_command,
    get_scontrol_reservation_df,
    get_slurm_launch_heuristic_df,
    get_sprio_df,
    print_slurm_compact_summary,
    print_queued_job_summary,
    print_slurm_launch_heuristic,
    get_user_df,
    get_scontrol_node_df,
    get_squeue_command_for_parsing,
    get_squeue_user_df,
)


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
    command = get_squeue_command_for_parsing("squeue -u kfuku -p epyc --format=%i")
    tokens = shlex.split(command)
    assert tokens[0] == "squeue"
    assert "-u" in tokens
    assert "kfuku" in tokens
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


def test_get_squeue_user_df_parses_literal_backslash_t():
    lines = [
        r"14817340_[106-239%239]\tepyc\tpepHsapE115\tktamagawa\tPD\t0:00\t1\t(Priority)",
        "",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "job_id"] == "14817340_[106-239%239]"
    assert df.at[0, "partition"] == "epyc"
    assert df.at[0, "total_slots"] == 134
    assert bool(df.at[0, "task_count_estimated"]) is False


def test_get_squeue_user_df_marks_truncated_array_as_estimated():
    lines = [
        "14817340_[106-239%\tepyc\tpepHsapE115\tktamagawa\tPD\t0:00\t1\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "total_slots"] == 134
    assert bool(df.at[0, "task_count_estimated"]) is True


def test_get_squeue_user_df_parses_extended_slurm_fields():
    lines = [
        "15243876\tepyc\twrap\tkfuku\tPD\t0:00\t1\t1\t1G\t00:05:00\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "req_cpus"]) == 1
    assert df.at[0, "req_mem"] == "1G"
    assert df.at[0, "time_limit"] == "00:05:00"
    assert df.at[0, "pending_reason"] == "Priority"
    assert bool(df.at[0, "resource_fields_complete"]) is True


def test_get_squeue_user_df_marks_legacy_slurm_fields_as_incomplete():
    lines = [
        "15243876\tepyc\twrap\tkfuku\tPD\t0:00\t1\t(Priority)",
    ]
    df = get_squeue_user_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "req_cpus"]) == 0
    assert df.at[0, "req_mem"] == ""
    assert bool(df.at[0, "resource_fields_complete"]) is False


def test_get_scontrol_node_df_skips_nodes_without_partition_and_marks_reserved():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
        "NodeName=n2 Arch=x86_64 CPUAlloc=8 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=8000 State=MIXED+RESERVED Partitions=p1",
        "NodeName=n3 Arch=x86_64 CPUAlloc=0 CPUEfctv=8 CPUTot=8 RealMemory=16000 FreeMem=15000 State=IDLE Partitions=(null)",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "UP"})
    assert sorted(df["node_name"].tolist()) == ["n1", "n2"]
    assert df.loc[df["node_name"] == "n1", "status"].iloc[0] == ""
    assert df.loc[df["node_name"] == "n2", "status"].iloc[0] == "MIXED+RESERVED"


def test_get_scontrol_node_df_marks_inactive_partition_as_abnormal():
    lines = [
        "NodeName=n1 Arch=x86_64 CPUAlloc=4 CPUEfctv=16 CPUTot=16 RealMemory=32000 FreeMem=16000 State=IDLE Partitions=p1",
    ]
    df = get_scontrol_node_df(lines, partition_state_map={"p1": "INACTIVE"})
    assert df.shape[0] == 1
    assert df.at[0, "status"] == "partition_state=INACTIVE"


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


def test_get_scontrol_reservation_df_counts_explicit_core_ids_and_single_node_fallback():
    lines = [
        "ReservationName=r1 StartTime=2026-03-06T12:00:00 EndTime=2026-03-07T12:00:00 Duration=1-00:00:00",
        "Nodes=a020 NodeCnt=1 CoreCnt=8 PartitionName=epyc Flags=IGNORE_JOBS State=ACTIVE TRES=cpu=8,mem=64G,node=1,billing=8",
        "NodeName=a020 CoreIDs=0-2,4,6-8",
        "",
        "ReservationName=r2 StartTime=2026-03-06T12:00:00 EndTime=2026-03-07T12:00:00 Duration=1-00:00:00",
        "Nodes=a021 NodeCnt=1 CoreCnt=6 PartitionName=epyc Flags=IGNORE_JOBS State=ACTIVE",
        "NodeName=a021 CoreIDs=(null)",
    ]
    df = get_scontrol_reservation_df(lines)
    assert df.shape[0] == 2
    assert int(df.loc[df["node_name"] == "a020", "reserved_cores"].iloc[0]) == 7
    assert int(df.loc[df["node_name"] == "a020", "reserved_mem_mb"].iloc[0]) == 64000
    assert int(df.loc[df["node_name"] == "a021", "reserved_cores"].iloc[0]) == 6


def test_apply_slurm_reservations_subtracts_partial_reservations_and_estimated_memory():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["a020"],
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
            "node_name": ["a020"],
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
            "node_name": ["a020"],
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
            "node_name": ["a020"],
            "reservation_name": ["r1"],
            "reserved_cores": [6],
            "reserved_mem_mb": [14000],
        }
    )
    out = apply_slurm_reservations(df_node, df_reservation)
    assert int(out.at[0, "reservation_mem_mb"]) == 14000
    assert out.at[0, "hc:mem_req"] == "18000M"


def test_get_sprio_df_parses_pending_priority_table():
    lines = [
        "          JOBID PARTITION   PRIORITY       SITE        AGE  FAIRSHARE    JOBSIZE  PARTITION",
        "       15243876 epyc           12721          0          0       2708         14      10000",
    ]
    df = get_sprio_df(lines)
    assert df.shape[0] == 1
    assert df.at[0, "job_id"] == "15243876"
    assert int(df.at[0, "priority"]) == 12721
    assert int(df.at[0, "fairshare"]) == 2708


def test_get_slurm_launch_heuristic_returns_na_when_priority_blocks_even_tiny_job():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["a004"],
            "status": [""],
            "ncore_available": [67],
            "hc:mem_req": ["925G"],
        }
    )
    df_job = pandas.DataFrame(
        {
            "job_id": ["15243876"],
            "partition": ["epyc"],
            "user": ["kfuku"],
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
            "job_id": ["15243876", "topjob"],
            "partition": ["epyc", "epyc"],
            "priority": [12721, 16652],
            "fairshare": [2708, 6634],
        }
    )
    out = get_slurm_launch_heuristic_df(df_node=df_node, df_job=df_job, df_prio=df_prio, current_user="kfuku")
    assert out.shape[0] == 1
    assert pandas.isna(out.at[0, "recommended_cores"])
    assert pandas.isna(out.at[0, "recommended_mem_gib"])
    assert out.at[0, "status"] == "priority_blocked"
    assert int(out.at[0, "priority_gap"]) == 3931
    assert int(out.at[0, "fairshare_gap"]) == 3926
    assert int(out.at[0, "top_node_cores"]) == 67


def test_get_slurm_launch_heuristic_returns_na_without_zero_sized_request_for_legacy_rows():
    df_node = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "node_name": ["a004"],
            "status": [""],
            "ncore_available": [67],
            "hc:mem_req": ["925G"],
        }
    )
    df_job = pandas.DataFrame(
        {
            "job_id": ["15243876"],
            "partition": ["epyc"],
            "user": ["kfuku"],
            "state": ["PD"],
            "req_cpus": [0],
            "req_mem": [""],
            "time_limit": [""],
            "pending_reason": ["Priority"],
            "resource_fields_complete": [False],
        }
    )
    out = get_slurm_launch_heuristic_df(df_node=df_node, df_job=df_job, current_user="kfuku")
    assert out.shape[0] == 1
    assert pandas.isna(out.at[0, "recommended_cores"])
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


def test_get_qstat_df_clips_negative_available_and_fills_missing_memory():
    lines = [
        "epyc.q@node01 BP 2/4/5 0.10 lx-amd64",
        "\thl:mem_total=8.000G",
    ]
    df = get_qstat_df(lines)
    assert df.shape[0] == 1
    assert int(df.at[0, "ncore_available"]) == 0
    assert df.at[0, "hc:mem_req"] == "0G"
    assert df.at[0, "hl:mem_total"] == "8.000G"


def test_adjust_ram_unit_converts_mib_to_gib_consistently():
    df = pandas.DataFrame(
        {
            "hc:mem_req": ["500M", "1.5G", "2T"],
            "hl:mem_total": ["1000M", "4G", "1T"],
        }
    )
    out = adjust_ram_unit(df)
    assert out.at[0, "hc:mem_req"] == 0.5
    assert out.at[0, "hc:mem_req_unit"] == "G"
    assert out.at[1, "hc:mem_req"] == 1.5
    assert out.at[2, "hc:mem_req"] == 2000.0
    assert out.at[2, "hl:mem_total"] == 1000.0


def test_adjust_ram_unit_handles_lowercase_and_invalid_values():
    df = pandas.DataFrame(
        {
            "hc:mem_req": ["500m", "bad", ""],
            "hl:mem_total": ["1t", "2g", None],
        }
    )
    out = adjust_ram_unit(df)
    assert out.at[0, "hc:mem_req"] == 0.5
    assert out.at[0, "hc:mem_req_unit"] == "G"
    assert out.at[1, "hc:mem_req"] == 0.0
    assert out.at[1, "hc:mem_req_unit"] == "G"
    assert out.at[0, "hl:mem_total"] == 1000.0
    assert out.at[1, "hl:mem_total"] == 2.0
    assert out.at[2, "hl:mem_total"] == 0.0
    assert out.at[2, "hl:mem_total_unit"] == "G"


def test_print_queued_job_summary_slurm_accepts_long_state_names(capsys):
    df_user = pandas.DataFrame(
        {
            "user": ["kfuku", "other", "kfuku"],
            "state": ["RUNNING", "PENDING", "FAILED"],
            "total_slots": [2, 3, 4],
            "task_count_estimated": [False, False, False],
        }
    )
    print_queued_job_summary(df_user, scheduler="slurm", current_user="kfuku")
    out = capsys.readouterr().out
    assert "jobs  self:R/Q/F=2/0/4  all:R/Q/F=2/3/4" in out


def test_print_slurm_launch_heuristic_uses_multiline_blocks(capsys):
    df_launch = pandas.DataFrame(
        {
            "queue_name": ["epyc"],
            "recommended_cores": [None],
            "recommended_mem_gib": [None],
            "top_node_name": ["a004"],
            "top_node_cores": [59],
            "top_node_mem_gib": [925.0],
            "priority_gap": [3931],
            "fairshare_gap": [3926],
            "blocked_req_cores": [1],
            "blocked_req_mem_gib": [1.0],
            "blocked_time_limit": ["00:05:00"],
            "status": ["priority_blocked"],
        }
    )
    print_slurm_launch_heuristic(df_launch, current_user="kfuku")
    out = capsys.readouterr().out
    assert "epyc:\n" in out
    assert "  immediate-start ceiling: n/a\n" in out
    assert "  top free node: a004 has 59 CPUs and 925G RAM\n" in out
    assert "  smallest current Priority-blocked request is 1 CPUs / 1G / 00:05:00\n" in out
    assert "  note: current user has Priority-blocked jobs; no stable immediate-start ceiling can be inferred\n" in out


def test_print_slurm_compact_summary_uses_single_row_per_partition(capsys):
    df = pandas.DataFrame(
        {
            "queue_name": ["epyc", "epyc", "rome"],
            "node_name": ["a004", "a017", "at141"],
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
            "recommended_cores": [None, None],
            "recommended_mem_gib": [None, None],
            "top_node_name": ["a004", "at141"],
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
    assert "a004 14c/5G" in out
    assert "a017 0c/352G" in out
    assert "PRIO min=1c/1G/5m gap=18097 fs=17960" in out
    assert "legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total" in out


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
    scheduler, df, _ = get_df(args)
    assert scheduler == "uge"
    assert df.shape[0] == 1
    assert int(df.at[0, "ncore_available"]) == 2
