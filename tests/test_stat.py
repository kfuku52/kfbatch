import shlex
from types import SimpleNamespace

import pandas
import pytest

import kfbatch.stat as stat_module
from kfbatch.stat import (
    adjust_ram_unit,
    get_command_stdout_lines,
    get_df,
    get_qstat_df,
    get_scheduler_from_command,
    print_queued_job_summary,
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


def test_get_squeue_command_for_parsing_keeps_explicit_format_equals():
    command = get_squeue_command_for_parsing("squeue --format=%i")
    tokens = shlex.split(command)
    assert "-h" in tokens
    assert any(token.startswith("--format=") for token in tokens)
    assert "-o" not in tokens


def test_get_squeue_command_for_parsing_keeps_short_o_attached():
    command = get_squeue_command_for_parsing("squeue -o%i")
    tokens = shlex.split(command)
    assert "-h" in tokens
    assert any(token.startswith("-o") and token != "-o" for token in tokens)
    assert tokens.count("-o") == 0


def test_get_command_stdout_lines_empty_command_allow_failure():
    out = get_command_stdout_lines("", allow_failure=True, quiet_failure=True)
    assert out is None


def test_get_command_stdout_lines_empty_command_exit():
    with pytest.raises(SystemExit):
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
            "state": ["RUNNING", "PENDING", "FAILED"],
            "total_slots": [2, 3, 4],
            "task_count_estimated": [False, False, False],
        }
    )
    print_queued_job_summary(df_user, scheduler="slurm")
    out = capsys.readouterr().out
    assert "# of running job tasks (estimated from squeue): 2" in out
    assert "# of queued job tasks (estimated from squeue): 3" in out
    assert "# of failed/cancelled job tasks (estimated from squeue): 4" in out


def test_get_df_qstat_requires_niter_at_least_one():
    args = SimpleNamespace(stat_command="qstat -F", niter=0)
    with pytest.raises(SystemExit):
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
