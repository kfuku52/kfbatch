import getpass
import grp
import json
import os
import pathlib
import re
import shlex
import tempfile

import pandas

from kfbatch.command import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    get_command_stdout_lines,
)
from kfbatch.errors import KFBatchCommandError, KFBatchUsageError
from kfbatch.memory import (
    floor_gib,
    memory_series_to_gib,
    memory_text_to_gib,
    memory_text_to_mib,
    slurm_request_memory_gib,
)

SLURM_RUNNING_STATES = {"R", "CG"}
SLURM_PENDING_STATES = {"PD", "CF"}
SLURM_ERROR_STATES = {
    "BF",  # BOOT_FAIL
    "CA",  # CANCELLED
    "DL",  # DEADLINE
    "F",  # FAILED
    "NF",  # NODE_FAIL
    "OOM",  # OUT_OF_MEMORY
    "PR",  # PREEMPTED
    "RV",  # REVOKED
    "SE",  # SPECIAL_EXIT
    "ST",  # STOPPED
    "TO",  # TIMEOUT
}
SLURM_STATE_NAME_TO_CODE = {
    "RUNNING": "R",
    "COMPLETING": "CG",
    "PENDING": "PD",
    "CONFIGURING": "CF",
    "BOOT_FAIL": "BF",
    "CANCELLED": "CA",
    "DEADLINE": "DL",
    "FAILED": "F",
    "NODE_FAIL": "NF",
    "OUT_OF_MEMORY": "OOM",
    "PREEMPTED": "PR",
    "REVOKED": "RV",
    "SPECIAL_EXIT": "SE",
    "STOPPED": "ST",
    "TIMEOUT": "TO",
}
SLURM_NORMAL_NODE_STATES = {"IDLE", "MIXED", "ALLOCATED", "COMPLETING"}
SLURM_UNAVAILABLE_NODE_FLAGS = {
    "DRAIN",
    "DRAINING",
    "DOWN",
    "FAIL",
    "NOT_RESPONDING",
    "MAINT",
    "POWER_DOWN",
    "POWERING_DOWN",
    "POWERED_DOWN",
    "REBOOT_REQUESTED",
    "REBOOT_ISSUED",
    "PLANNED",
}
SLURM_SQUEUE_PARSE_FIELDS = "%i\t%P\t%j\t%u\t%a\t%t\t%M\t%D\t%C\t%m\t%l\t%R"
QSTAT_REQUIRED_NODE_FIELDS = {
    "queue_name",
    "node_name",
    "qtype",
    "ncore_resv",
    "ncore_used",
    "ncore_total",
    "np_load",
    "arch",
    "status",
}


def _parse_uge_task_expression(task_expression):
    if task_expression == "":
        return 1, False
    num_tasks = 0
    estimated = False
    for token in task_expression.split(","):
        token = token.strip()
        if token == "":
            estimated = True
            continue
        m = re.match(r"^([0-9]+)-([0-9]+):([0-9]+)$", token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            step = int(m.group(3))
            if (step <= 0) or (end < start):
                estimated = True
                continue
            num_tasks += int((end - start) / step) + 1
            continue
        m = re.match(r"^([0-9]+)-([0-9]+)$", token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            if end < start:
                estimated = True
                continue
            num_tasks += int(end - start) + 1
            continue
        if re.match(r"^[0-9]+$", token):
            num_tasks += 1
            continue
        estimated = True
    if num_tasks == 0:
        return 1, True
    return num_tasks, estimated


def get_qstat_df(lines):
    columns = [
        "queue_name",
        "node_name",
        "qtype",
        "ncore_resv",
        "ncore_used",
        "ncore_total",
        "np_load",
        "arch",
        "status",
        "hc:mem_req",
        "hl:mem_total",
        "hc:mem_req_known",
        "hl:mem_total_known",
        "ncore_available",
    ]
    lines = [re.sub("\n$", "", line) for line in lines]
    lines = [line for line in lines if line != ""]
    lines = [line for line in lines if not line.startswith("queuename")]
    lines = [line for line in lines if not line.startswith("---")]
    lines = [line for line in lines if not line.startswith("###")]
    lines = [line for line in lines if not line.startswith(" ")]
    lines = [line for line in lines if not line.startswith("\n")]
    node_params = {}
    rows = []
    for line in lines:
        if not line.startswith("\t"):
            if QSTAT_REQUIRED_NODE_FIELDS.issubset(set(node_params.keys())):
                rows.append(node_params)
            node_params = {}
            items = [item for item in line.split(" ") if item != ""]
            if len(items) < 5:
                continue
            m = re.match(r"^([0-9]+)/([0-9]+)/([0-9]+)$", items[2])
            if m is None:
                continue
            node_params["queue_name"] = re.sub("@.*", "", items[0])
            node_params["node_name"] = re.sub(".*@", "", items[0])
            node_params["qtype"] = items[1]
            node_params["ncore_resv"] = m.group(1)
            node_params["ncore_used"] = m.group(2)
            node_params["ncore_total"] = m.group(3)
            node_params["np_load"] = items[3]
            node_params["arch"] = items[4]
            if len(items) > 5:
                node_params["status"] = items[5]
            else:
                node_params["status"] = ""
        else:
            key = re.sub("\t", "", line)
            key = re.sub("=.*", "", key)
            value = re.sub(".*=", "", line)
            node_params[key] = value
    if QSTAT_REQUIRED_NODE_FIELDS.issubset(set(node_params.keys())):
        rows.append(node_params)
    df = pandas.DataFrame.from_records(rows)
    if df.shape[0] == 0:
        return pandas.DataFrame(columns=columns)
    # The live AGE output contains hundreds of resource fields. Constructing the
    # frame once keeps its internal blocks contiguous before adding normalized
    # columns below.
    df = df.copy()
    for col in ["ncore_resv", "ncore_used", "ncore_total"]:
        df[col] = df[col].astype(int)
    for mem_col in ["hc:mem_req", "hl:mem_total"]:
        if mem_col not in df.columns:
            df[mem_col] = pandas.NA
        raw = df[mem_col].astype("string").str.strip()
        is_known = raw.notna() & (raw != "")
        df[mem_col + "_known"] = is_known
        df.loc[~is_known, mem_col] = pandas.NA
    ncore_available = df["ncore_total"] - df["ncore_used"] - df["ncore_resv"]
    ncore_available = ncore_available.clip(lower=0)
    tmp = pandas.DataFrame({"ncore_available": ncore_available.astype(int)})
    df = pandas.concat([df, tmp], axis=1)
    df = df.sort_values(by=["queue_name", "node_name"]).reset_index(drop=True)
    return df


def _memory_series_to_gb(series):
    return memory_series_to_gib(series)


def _memory_text_to_gb(value):
    return memory_text_to_gib(value)


def _memory_text_to_mb(value):
    value_mib = memory_text_to_mib(value)
    if pandas.isna(value_mib):
        return 0
    return int(round(value_mib))


def _extract_tres_resource_value(tres_txt, resource_name):
    txt = str(tres_txt).strip()
    if txt == "":
        return ""
    prefix = f"{resource_name}="
    for token in txt.split(","):
        token = token.strip()
        if token.startswith(prefix):
            return token[len(prefix) :].strip()
    return ""


def _slurm_time_to_minutes(value):
    txt = str(value).strip()
    if txt in ["", "N/A", "UNLIMITED", "NOT_SET"]:
        return float("inf")
    day_part = 0
    if "-" in txt:
        day_txt, txt = txt.split("-", 1)
        day_part = _safe_int(day_txt, default=0)
    items = txt.split(":")
    if len(items) == 3:
        hours = _safe_int(items[0], default=0)
        minutes = _safe_int(items[1], default=0)
        seconds = _safe_int(items[2], default=0)
    elif len(items) == 2:
        hours = 0
        minutes = _safe_int(items[0], default=0)
        seconds = _safe_int(items[1], default=0)
    elif len(items) == 1:
        hours = 0
        minutes = _safe_int(items[0], default=0)
        seconds = 0
    else:
        return float("inf")
    total_minutes = (day_part * 24 * 60) + (hours * 60) + minutes + (seconds / 60.0)
    return float(total_minutes)


def _extract_slurm_pending_reason(node_or_reason):
    txt = str(node_or_reason).strip()
    m = re.match(r"^\((.*)\)$", txt)
    if m is None:
        return ""
    return m.group(1).strip()


def _merge_qstat_common_rows(df_base, df_new, common_index):
    if len(common_index) == 0:
        return
    base_cores = pandas.to_numeric(
        df_base.loc[common_index, "ncore_available"], errors="coerce"
    ).fillna(0)
    new_cores = pandas.to_numeric(
        df_new.loc[common_index, "ncore_available"], errors="coerce"
    ).fillna(0)
    use_new = new_cores <= base_cores
    for col in ["ncore_resv", "ncore_used", "ncore_total", "np_load"]:
        if col in df_base.columns and col in df_new.columns:
            replacement = df_new.loc[common_index, col]
            selected = df_base.loc[common_index, col].copy()
            selected.loc[use_new] = replacement.loc[use_new]
            df_base.loc[common_index, col] = selected
    df_base.loc[common_index, "ncore_available"] = (
        pandas.concat([base_cores, new_cores], axis=1).min(axis=1).astype(int)
    )

    for mem_col in ["hc:mem_req", "hl:mem_total"]:
        base_mem = _memory_series_to_gb(df_base.loc[common_index, mem_col])
        new_mem = _memory_series_to_gb(df_new.loc[common_index, mem_col])
        known_col = mem_col + "_known"
        base_known = (
            df_base.loc[common_index, known_col].fillna(False).astype(bool)
            if known_col in df_base.columns
            else base_mem.notna()
        )
        new_known = (
            df_new.loc[common_index, known_col].fillna(False).astype(bool)
            if known_col in df_new.columns
            else new_mem.notna()
        )
        known = base_known & new_known & base_mem.notna() & new_mem.notna()
        min_mem = pandas.concat([base_mem, new_mem], axis=1).min(axis=1)
        df_base.loc[common_index, mem_col] = min_mem.where(known).map(
            lambda value: pandas.NA if pandas.isna(value) else f"{float(value):.3f}G"
        )
        known_values = df_base[known_col].copy()
        known_values.loc[common_index] = known
        df_base[known_col] = known_values

    if "status" not in df_base.columns:
        df_base["status"] = ""
    new_status = (
        df_new.loc[common_index, "status"]
        if "status" in df_new.columns
        else pandas.Series("", index=common_index)
    )
    for row_index in common_index:
        tokens = []
        for value in [df_base.at[row_index, "status"], new_status.at[row_index]]:
            for token in str(value or "").split("|"):
                token = token.strip()
                if token and token not in tokens:
                    tokens.append(token)
        df_base.at[row_index, "status"] = "|".join(tokens)


def _merge_qstat_iteration_min_availability(df, df_i):
    key_cols = ["queue_name", "node_name"]
    if df.shape[0] == 0:
        return df_i.copy()
    if (not set(key_cols).issubset(set(df.columns))) or (
        not set(key_cols).issubset(set(df_i.columns))
    ):
        return df
    df_base = df.set_index(key_cols, drop=False).copy()
    df_new = df_i.set_index(key_cols, drop=False)
    common_index = df_base.index.intersection(df_new.index)
    _merge_qstat_common_rows(df_base, df_new, common_index)
    missing_from_new = df_base.index.difference(df_new.index)
    new_since_first = df_new.index.difference(df_base.index)
    if "status" not in df_base.columns:
        df_base["status"] = ""
    if len(missing_from_new) > 0:
        df_base.loc[missing_from_new, "ncore_available"] = 0
        df_base.loc[missing_from_new, "hc:mem_req"] = pandas.NA
        if "hc:mem_req_known" in df_base.columns:
            df_base.loc[missing_from_new, "hc:mem_req_known"] = False
        previous = df_base.loc[missing_from_new, "status"].fillna("").astype(str)
        df_base.loc[missing_from_new, "status"] = previous.map(
            lambda value: "|".join(token for token in [value, "missing_in_snapshot"] if token)
        )
    if len(new_since_first) > 0:
        new_rows = df_new.loc[new_since_first].copy()
        if "status" not in new_rows.columns:
            new_rows["status"] = ""
        new_rows["ncore_available"] = 0
        new_rows["hc:mem_req"] = pandas.NA
        if "hc:mem_req_known" in new_rows.columns:
            new_rows["hc:mem_req_known"] = False
        new_rows["status"] = (
            new_rows["status"]
            .fillna("")
            .astype(str)
            .map(
                lambda value: "|".join(
                    token for token in [value, "missing_in_previous_snapshot"] if token
                )
            )
        )
        df_base = pandas.concat([df_base, new_rows], axis=0)
    df_base = df_base.reset_index(drop=True)
    df_base = df_base.sort_values(by=key_cols).reset_index(drop=True)
    return df_base


def _empty_uge_job_df():
    columns = [
        "job_id",
        "prior",
        "name",
        "user",
        "state",
        "submit_or_start_date",
        "submit_or_start_time",
        "queue_name",
        "slots",
        "ja_task_id",
        "total_slots",
        "task_count_estimated",
    ]
    return pandas.DataFrame(columns=columns)


def _parse_uge_text_job_line(line):
    items = re.split(r"\s+", str(line).strip())
    if (len(items) < 8) or (re.match(r"^[0-9]+$", items[0]) is None):
        return None
    tail = items[7:]
    queue_name = ""
    if tail and (("@" in tail[0]) or tail[0].endswith(".q")):
        queue_name = re.sub(r"@.*$", "", tail.pop(0))
    if tail and (re.match(r"^[0-9]+$", tail[0]) is None):
        # AGE may print a non-empty job-class column between queue and slots.
        tail.pop(0)
    if not tail or (re.match(r"^[0-9]+$", tail[0]) is None):
        return None
    slots = int(tail.pop(0))
    ja_task_id = tail[0] if tail else ""
    num_tasks, task_count_estimated = _parse_uge_task_expression(ja_task_id)
    return {
        "job_id": items[0],
        "prior": items[1],
        "name": items[2],
        "user": items[3],
        "state": items[4],
        "submit_or_start_date": items[5],
        "submit_or_start_time": items[6],
        "queue_name": queue_name,
        "slots": slots,
        "ja_task_id": ja_task_id,
        "total_slots": slots * num_tasks,
        "task_count_estimated": task_count_estimated,
    }


def get_user_df(lines):
    rows = []
    for line in lines:
        if re.match(r"^\s+[0-9]+\s+", str(line)) is None:
            continue
        row = _parse_uge_text_job_line(line)
        if row is not None:
            rows.append(row)
    if len(rows) == 0:
        return _empty_uge_job_df()
    return pandas.DataFrame(rows, columns=_empty_uge_job_df().columns)


def _iter_uge_json_jobs(data):
    for section_value in data.values():
        if isinstance(section_value, dict):
            section_items = [section_value]
        elif isinstance(section_value, list):
            section_items = section_value
        else:
            continue
        for section_item in section_items:
            if not isinstance(section_item, dict):
                continue
            for job_list in section_item.values():
                if not isinstance(job_list, list):
                    continue
                for job in job_list:
                    if isinstance(job, dict):
                        yield job


def _uge_json_job_row(job):
    job_id = str(job.get("JB_job_number", job.get("job_id", ""))).strip()
    if job_id == "":
        return None
    slots = max(_safe_int(job.get("slots", 1), default=1), 0)
    task_expression = str(
        job.get("ja_task_id", job.get("ja-task-ID", job.get("tasks", "")))
    ).strip()
    num_tasks, expression_estimated = _parse_uge_task_expression(task_expression)
    queue_name = re.sub(r"@.*$", "", str(job.get("queue_name", "")).strip())
    state = str(job.get("state", ""))
    timestamp = str(job.get("JAT_start_time", job.get("JB_submission_time", ""))).strip()
    timestamp_items = timestamp.split("T", 1)
    return {
        "job_id": job_id,
        "prior": job.get("JAT_prio", ""),
        "name": str(job.get("JB_name", "")),
        "user": str(job.get("JB_owner", "")),
        "state": state,
        "submit_or_start_date": timestamp_items[0] if timestamp_items else "",
        "submit_or_start_time": timestamp_items[1] if len(timestamp_items) > 1 else "",
        "queue_name": queue_name,
        "slots": slots,
        "ja_task_id": task_expression,
        "total_slots": slots * num_tasks,
        # AGE 2023 can omit the range for collapsed pending arrays.
        "task_count_estimated": expression_estimated
        or (not task_expression and not queue_name and "q" in state.lower()),
    }


def get_uge_json_job_df(lines):
    payload = "\n".join([str(line).rstrip("\n") for line in lines]).strip()
    if payload == "":
        return _empty_uge_job_df()
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    recognized_schema = bool({"queue_info", "job_info"} & set(data))
    rows = []
    for job in _iter_uge_json_jobs(data):
        row = _uge_json_job_row(job)
        if row is not None:
            recognized_schema = True
            rows.append(row)
    if not recognized_schema:
        return None
    if len(rows) == 0:
        empty = _empty_uge_job_df()
        empty.attrs["recognized_schema"] = True
        return empty
    frame = pandas.DataFrame(rows, columns=_empty_uge_job_df().columns)
    frame.attrs["recognized_schema"] = True
    return frame


def _optional_int(value):
    txt = str(value).strip().replace(",", "")
    if txt in ["", "-"]:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def get_qfree_df(lines):
    columns = [
        "queue_name",
        "self_slots",
        "group_slots",
        "quota_slots",
        "all_slots",
        "available_slots_2g",
        "standby_slots",
        "total_slots",
        "self_mem_req_gb",
        "group_mem_req_gb",
        "quota_mem_gb",
        "all_mem_req_gb",
        "total_mem_gb",
    ]
    rows_by_queue = {}
    queue_order = []
    mode = ""
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for raw_line in lines:
        line = ansi_escape.sub("", str(raw_line)).strip()
        if line == "SUMMARY OF RUNNING JOBS":
            mode = "slots"
            continue
        if line == "SUMMARY OF RUNNING JOBS ( MEM_REQ )":
            mode = "memory"
            continue
        if line.startswith("THE NUMBER OF ") or line.startswith("======================"):
            mode = ""
            continue
        if mode == "":
            continue
        items = re.split(r"\s+", line)
        if (len(items) != 8) or (items[0] in ["QNAME", "-------------"]):
            continue
        queue_name = items[0]
        if re.match(r"^[A-Za-z0-9_.-]+$", queue_name) is None:
            continue
        values = [_optional_int(value) for value in items[1:]]
        if any(value is None for value in values[0:2] + values[3:7]):
            continue
        if queue_name not in rows_by_queue:
            rows_by_queue[queue_name] = {col: None for col in columns}
            rows_by_queue[queue_name]["queue_name"] = queue_name
            queue_order.append(queue_name)
        row = rows_by_queue[queue_name]
        if mode == "slots":
            (
                row["self_slots"],
                row["group_slots"],
                row["quota_slots"],
                row["all_slots"],
                row["available_slots_2g"],
                row["standby_slots"],
                row["total_slots"],
            ) = values
        else:
            (
                row["self_mem_req_gb"],
                row["group_mem_req_gb"],
                row["quota_mem_gb"],
                row["all_mem_req_gb"],
                _available_slots_2g,
                _standby_slots,
                row["total_mem_gb"],
            ) = values
    rows = [rows_by_queue[queue_name] for queue_name in queue_order]
    return pandas.DataFrame(rows, columns=columns)


def _count_slurm_array_task_expression(task_expression):
    if task_expression == "":
        return 1, True
    num_tasks = 0
    has_ambiguous_pattern = False
    for token in task_expression.split(","):
        token = token.strip()
        if token == "":
            has_ambiguous_pattern = True
            continue
        m = re.match(r"^([0-9]+)-([0-9]+)(?::([0-9]+))?$", token)
        if m:
            start = int(m.group(1))
            end = int(m.group(2))
            step = 1 if m.group(3) is None else int(m.group(3))
            if (step <= 0) or (end < start):
                has_ambiguous_pattern = True
                continue
            num_tasks += int((end - start) / step) + 1
            continue
        if re.match(r"^[0-9]+$", token):
            num_tasks += 1
            continue
        has_ambiguous_pattern = True
    if num_tasks == 0:
        return 1, True
    return num_tasks, has_ambiguous_pattern


def estimate_slurm_task_count(job_id):
    if "_" not in job_id:
        return 1, False
    job_suffix = job_id.split("_", 1)[1]
    if re.match(r"^[0-9]+$", job_suffix):
        return 1, False
    if not job_suffix.startswith("["):
        return 1, True
    task_expression = re.sub(r"^\[", "", job_suffix)
    has_closing_bracket = "]" in task_expression
    if has_closing_bracket:
        task_expression = task_expression.split("]", 1)[0]
    task_expression = task_expression.split("%", 1)[0]
    num_tasks, has_ambiguous_pattern = _count_slurm_array_task_expression(task_expression)
    is_estimated = has_ambiguous_pattern or (not has_closing_bracket)
    return num_tasks, is_estimated


def _split_squeue_row(line):
    if "\t" in line:
        return line.split("\t"), "\t"
    if "\\t" in line:
        # Some captured files may contain literal "\t" separators.
        return line.split("\\t"), "\\t"
    return re.split(r"\s+", line.strip(), maxsplit=11), " "


def _looks_like_slurm_state_token(value):
    return re.fullmatch(r"[A-Za-z_]+", str(value or "").strip()) is not None


def _parse_squeue_row_items(items, rest_separator):
    items = [str(item).strip() for item in items]
    has_account = len(items) >= 12 and _looks_like_slurm_state_token(items[5])
    if has_account:
        return {
            "resource_fields_complete": True,
            "job_id": items[0],
            "partition": items[1],
            "name": items[2],
            "user": items[3],
            "account": items[4],
            "state": items[5],
            "elapsed_time": items[6],
            "num_nodes_txt": items[7],
            "req_cpus_txt": items[8],
            "req_mem": items[9],
            "time_limit": items[10],
            "node_or_reason": rest_separator.join(items[11:]).strip(),
        }
    if len(items) >= 11:
        return {
            "resource_fields_complete": True,
            "job_id": items[0],
            "partition": items[1],
            "name": items[2],
            "user": items[3],
            "account": "",
            "state": items[4],
            "elapsed_time": items[5],
            "num_nodes_txt": items[6],
            "req_cpus_txt": items[7],
            "req_mem": items[8],
            "time_limit": items[9],
            "node_or_reason": rest_separator.join(items[10:]).strip(),
        }
    if len(items) >= 8:
        return {
            "resource_fields_complete": False,
            "job_id": items[0],
            "partition": items[1],
            "name": items[2],
            "user": items[3],
            "account": "",
            "state": items[4],
            "elapsed_time": items[5],
            "num_nodes_txt": items[6],
            "req_cpus_txt": "",
            "req_mem": "",
            "time_limit": "",
            "node_or_reason": rest_separator.join(items[7:]).strip(),
        }
    return None


def get_squeue_user_df(lines):
    columns = [
        "job_id",
        "partition",
        "name",
        "user",
        "account",
        "state",
        "elapsed_time",
        "num_nodes",
        "req_cpus",
        "req_mem",
        "time_limit",
        "node_or_reason",
        "pending_reason",
        "resource_fields_complete",
        "total_slots",
        "task_count_estimated",
    ]
    table = []
    for raw_line in lines:
        line = re.sub("\n$", "", raw_line)
        if line.strip() == "":
            continue
        if line.lstrip().startswith("JOBID "):
            continue
        items, rest_separator = _split_squeue_row(line)
        row = _parse_squeue_row_items(items, rest_separator)
        if row is None:
            continue
        try:
            num_nodes = int(row["num_nodes_txt"])
        except ValueError:
            num_nodes = 1
        req_cpus = _safe_int(row["req_cpus_txt"], default=0)
        num_tasks, is_estimated = estimate_slurm_task_count(row["job_id"])
        total_slots = num_tasks
        table.append(
            {
                "job_id": row["job_id"],
                "partition": row["partition"],
                "name": row["name"],
                "user": row["user"],
                "account": row["account"],
                "state": row["state"],
                "elapsed_time": row["elapsed_time"],
                "num_nodes": num_nodes,
                "req_cpus": req_cpus,
                "req_mem": row["req_mem"],
                "time_limit": row["time_limit"],
                "node_or_reason": row["node_or_reason"],
                "pending_reason": _extract_slurm_pending_reason(row["node_or_reason"]),
                "resource_fields_complete": row["resource_fields_complete"],
                "total_slots": total_slots,
                "task_count_estimated": is_estimated,
            }
        )
    return pandas.DataFrame(table, columns=columns)


def _split_scontrol_node_blocks(lines):
    blocks = []
    current = ""
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            if current != "":
                blocks.append(current.strip())
                current = ""
            continue
        if ("NodeName=" in line) and (current != ""):
            blocks.append(current.strip())
            current = line
            continue
        if current == "":
            current = line
        else:
            current += " " + line
    if current != "":
        blocks.append(current.strip())
    return blocks


def _parse_key_value_fields(line):
    items = [item for item in line.split(" ") if item != ""]
    params = {}
    for item in items:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        params[key] = value
    return params


def _safe_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _partition_state_is_up(partition_state):
    state = str(partition_state).strip().upper()
    if state == "":
        return False
    tokens = re.findall(r"[A-Z_]+", state)
    if len(tokens) == 0:
        return False
    return (tokens[0] == "UP") and (len(tokens) == 1)


def _normalize_slurm_node_state(state_raw):
    if state_raw == "":
        return ""
    m = re.match(r"^([A-Z]+)", state_raw.upper())
    if m is None:
        return state_raw.upper()
    return m.group(1)


def _slurm_state_flags(state_raw):
    if state_raw == "":
        return []
    flags = []
    for token in state_raw.upper().split("+"):
        m = re.match(r"^([A-Z_]+)", token)
        if m is None:
            continue
        flags.append(m.group(1))
    return flags


def get_scontrol_partition_df(lines):
    columns = ["partition_name", "partition_state"]
    rows = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            continue
        if "PartitionName=" not in line:
            continue
        params = _parse_key_value_fields(line)
        partition_name = params.get("PartitionName", "")
        partition_state = params.get("State", "")
        if partition_name == "":
            continue
        rows.append(
            {
                "partition_name": partition_name,
                "partition_state": partition_state,
            }
        )
    return pandas.DataFrame(rows, columns=columns)


def _split_scontrol_named_blocks(lines, anchor_key):
    blocks = []
    current = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            if current:
                blocks.append(current)
                current = []
            continue
        if line.startswith(anchor_key) and current:
            blocks.append(current)
            current = [line]
            continue
        if not current:
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def _count_core_id_expression(core_ids):
    txt = str(core_ids).strip()
    if txt in ["", "(null)", "N/A"]:
        return 0
    total = 0
    for token in txt.split(","):
        token = token.strip()
        if token == "":
            continue
        m = re.match(r"^([0-9]+)-([0-9]+)$", token)
        if m is not None:
            start = int(m.group(1))
            end = int(m.group(2))
            if end >= start:
                total += (end - start) + 1
            continue
        if re.match(r"^[0-9]+$", token):
            total += 1
    return total


def _split_slurm_hostlist(value):
    tokens = []
    current = []
    depth = 0
    for character in str(value):
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(depth - 1, 0)
        if character == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                tokens.append(token)
            current = []
            continue
        current.append(character)
    token = "".join(current).strip()
    if token:
        tokens.append(token)
    return tokens


def _expand_slurm_hostlist(value):
    def expand_token(token):
        match = re.search(r"\[([^\]]+)\]", token)
        if match is None:
            return [token]
        prefix = token[: match.start()]
        expression = match.group(1)
        suffix = token[match.end() :]
        expanded = []
        for item in expression.split(","):
            item = item.strip()
            range_match = re.fullmatch(r"([0-9]+)-([0-9]+)", item)
            if range_match is None:
                values = [item] if item else []
            else:
                start_text, end_text = range_match.groups()
                start = int(start_text)
                end = int(end_text)
                width = max(len(start_text), len(end_text))
                values = [f"{number:0{width}d}" for number in range(start, end + 1)]
            for expanded_value in values:
                expanded.extend(expand_token(f"{prefix}{expanded_value}{suffix}"))
        return expanded

    text = str(value).strip()
    if text.upper() == "ALL":
        return ["*"]
    hosts = []
    for token in _split_slurm_hostlist(text):
        hosts.extend(expand_token(token))
    return hosts


def _normalize_access_values(value):
    return [
        item.strip()
        for item in str(value or "").split(",")
        if item.strip() not in {"", "(null)", "N/A"}
    ]


def _access_list_allows(value, candidates):
    entries = _normalize_access_values(value)
    if not entries:
        return None
    candidates = {str(candidate).strip() for candidate in candidates if str(candidate).strip()}
    if "ALL" in entries:
        return True
    is_deny_list = all(entry.startswith("-") for entry in entries)
    if is_deny_list:
        denied = {entry[1:] for entry in entries}
        return bool(candidates) and bool(candidates - denied)
    if any(entry.startswith("-") for entry in entries):
        return False
    return bool(candidates.intersection(entries))


def _reservation_user_is_authorized(users_value, current_user):
    allowed = _access_list_allows(users_value, [current_user])
    return bool(allowed)


def _current_group_names():
    group_ids = set(os.getgroups())
    group_ids.add(os.getgid())
    names = set()
    for group_id in group_ids:
        try:
            names.add(grp.getgrgid(group_id).gr_name)
        except KeyError:
            continue
    return names


def _reservation_access_is_authorized(
    header_params,
    current_user,
    current_accounts=None,
    current_groups=None,
):
    if not current_user:
        return False
    current_accounts = set(current_accounts or [])
    current_groups = set(current_groups or _current_group_names())
    partition_name = str(header_params.get("PartitionName", "")).strip()
    if partition_name in {"(null)", "N/A"}:
        partition_name = ""
    checks = [
        _access_list_allows(header_params.get("Users", ""), [current_user]),
        _access_list_allows(header_params.get("Groups", ""), current_groups),
        _access_list_allows(header_params.get("Accounts", ""), current_accounts),
        # A QOS-specific reservation cannot be proven accessible without a
        # concrete job QOS. QOS=ALL is still positively resolvable.
        _access_list_allows(header_params.get("QOS", ""), []),
        _access_list_allows(
            header_params.get("AllowedPartitions", ""),
            [partition_name] if partition_name else [],
        ),
    ]
    configured_checks = [check for check in checks if check is not None]
    return bool(configured_checks) and all(configured_checks)


SLURM_RESERVATION_COLUMNS = [
    "queue_name",
    "node_name",
    "reservation_name",
    "reserved_cores",
    "reserved_mem_mb",
    "whole_node",
    "accessible",
    "access_users",
]


def _reservation_header_params(block):
    params = {}
    for line in block:
        if ("=" in line) and (not line.startswith("NodeName=")):
            params.update(_parse_key_value_fields(line))
    return params


def _reservation_resource_defaults(header_params):
    node_count = _safe_int(header_params.get("NodeCnt", ""), default=0)
    reserved_cores = max(_safe_int(header_params.get("CoreCnt", ""), default=0), 0)
    tres = header_params.get("TRES", "") or header_params.get("ReqTRES", "")
    if reserved_cores <= 0:
        reserved_cores = max(
            _safe_int(_extract_tres_resource_value(tres, "cpu"), default=0),
            0,
        )
    reserved_mem_mb = max(
        _memory_text_to_mb(_extract_tres_resource_value(tres, "mem")),
        0,
    )
    return node_count, reserved_cores, reserved_mem_mb


def _reservation_row(context, node_name, reserved_cores, reserved_mem_mb, whole_node):
    return {
        "queue_name": context["partition_name"],
        "node_name": node_name,
        "reservation_name": context["reservation_name"],
        "reserved_cores": reserved_cores,
        "reserved_mem_mb": reserved_mem_mb,
        "whole_node": whole_node,
        "accessible": context["accessible"],
        "access_users": context["access_users"],
    }


def _explicit_reservation_rows(block, context):
    rows = []
    explicit_lines = [line for line in block if line.startswith("NodeName=")]
    for line in explicit_lines:
        params = _parse_key_value_fields(line)
        node_name = params.get("NodeName", "").strip()
        if node_name == "":
            continue
        reserved_cores = _count_core_id_expression(params.get("CoreIDs", ""))
        if reserved_cores == 0 and context["node_count"] == 1:
            reserved_cores = context["default_reserved_cores"]
        if reserved_cores <= 0:
            continue
        reserved_mem_mb = 0
        if context["default_reserved_mem_mb"] > 0 and context["node_count"] > 0:
            reserved_mem_mb = int(round(context["default_reserved_mem_mb"] / context["node_count"]))
        rows.append(
            _reservation_row(
                context,
                node_name,
                reserved_cores,
                reserved_mem_mb,
                whole_node=False,
            )
        )
    return rows, bool(explicit_lines)


def _hostlist_reservation_rows(header_params, context):
    node_names = _expand_slurm_hostlist(header_params.get("Nodes", "").strip())
    if not node_names:
        return []
    node_count = context["node_count"] or len(node_names)
    rows = []
    for node_index, node_name in enumerate(node_names):
        reserved_cores = 0
        if context["default_reserved_cores"] > 0 and node_count > 0:
            reserved_cores = context["default_reserved_cores"] // node_count
            if node_index < context["default_reserved_cores"] % node_count:
                reserved_cores += 1
        reserved_mem_mb = 0
        if context["default_reserved_mem_mb"] > 0 and node_count > 0:
            reserved_mem_mb = int(round(context["default_reserved_mem_mb"] / node_count))
        rows.append(
            _reservation_row(
                context,
                node_name,
                reserved_cores,
                reserved_mem_mb,
                whole_node=context["default_reserved_cores"] <= 0,
            )
        )
    return rows


def _parse_reservation_block(block, current_user, current_accounts, current_groups):
    header_params = _reservation_header_params(block)
    reservation_name = header_params.get("ReservationName", "").strip()
    state = header_params.get("State", "").strip().upper()
    if state == "":
        warning = "reservation {} has no State field and was ignored".format(
            reservation_name or "<unknown>"
        )
        return [], warning
    if state != "ACTIVE":
        return [], None
    partition_name = header_params.get("PartitionName", "").strip()
    if partition_name in {"(null)", "N/A"}:
        partition_name = ""
    node_count, reserved_cores, reserved_mem_mb = _reservation_resource_defaults(header_params)
    context = {
        "partition_name": partition_name,
        "reservation_name": reservation_name,
        "node_count": node_count,
        "default_reserved_cores": reserved_cores,
        "default_reserved_mem_mb": reserved_mem_mb,
        "accessible": _reservation_access_is_authorized(
            header_params,
            current_user=current_user,
            current_accounts=current_accounts,
            current_groups=current_groups,
        ),
        "access_users": header_params.get("Users", "").strip(),
    }
    rows, has_explicit_rows = _explicit_reservation_rows(block, context)
    if has_explicit_rows:
        return rows, None
    rows = _hostlist_reservation_rows(header_params, context)
    if rows:
        return rows, None
    warning = "active reservation {} has no parseable Nodes field".format(
        reservation_name or "<unknown>"
    )
    return [], warning


def get_scontrol_reservation_df(
    lines,
    current_user="",
    current_accounts=None,
    current_groups=None,
):
    rows = []
    warnings = []
    for block in _split_scontrol_named_blocks(lines, "ReservationName="):
        block_rows, warning = _parse_reservation_block(
            block,
            current_user,
            current_accounts,
            current_groups,
        )
        rows.extend(block_rows)
        if warning is not None:
            warnings.append(warning)
    frame = pandas.DataFrame(rows, columns=SLURM_RESERVATION_COLUMNS)
    frame.attrs["warnings"] = warnings
    return frame


def _expand_reservation_rows(df_node, df_reservation):
    available_nodes = set(df_node["node_name"].dropna().astype(str))
    rows = []
    unresolved_targets = []
    for _, reservation in df_reservation.iterrows():
        node_name = str(reservation.get("node_name", "") or "").strip()
        queue_name = str(reservation.get("queue_name", "") or "").strip()
        if node_name == "*":
            if queue_name:
                targets = (
                    df_node.loc[df_node["queue_name"].astype(str) == queue_name, "node_name"]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )
            else:
                targets = sorted(available_nodes)
        else:
            targets = [node_name] if node_name in available_nodes else []
        if not targets:
            unresolved_targets.append(
                f"{reservation.get('reservation_name', '<unknown>')}:{node_name or '<none>'}"
            )
            continue
        for target in targets:
            row = reservation.to_dict()
            row["node_name"] = target
            rows.append(row)
    expanded = pandas.DataFrame(rows, columns=df_reservation.columns)
    expanded.attrs["unresolved_targets"] = unresolved_targets
    return expanded


def apply_slurm_reservations(df_node, df_reservation):
    if (
        df_node is None
        or df_node.shape[0] == 0
        or df_reservation is None
        or df_reservation.shape[0] == 0
    ):
        return df_node
    df = df_node.copy()
    for col in ["reservation_cores", "reservation_mem_mb"]:
        if col not in df.columns:
            df[col] = 0
    if "reservation_accessible" not in df.columns:
        df["reservation_accessible"] = False

    reservation_rows = _expand_reservation_rows(df, df_reservation)
    df.attrs["reservation_unresolved_targets"] = reservation_rows.attrs.get(
        "unresolved_targets", []
    )
    if reservation_rows.shape[0] == 0:
        return df

    accessible_mask = (
        reservation_rows["accessible"].fillna(False).astype(bool)
        if "accessible" in reservation_rows.columns
        else pandas.Series(False, index=reservation_rows.index)
    )
    accessible_rows = reservation_rows.loc[accessible_mask, :]
    accessible_nodes = set(accessible_rows["node_name"].dropna().astype(str))
    if accessible_nodes:
        df.loc[df["node_name"].astype(str).isin(accessible_nodes), "reservation_accessible"] = True

    reservation_rows = reservation_rows.loc[~accessible_mask, :].copy()
    if reservation_rows.shape[0] == 0:
        return df

    node_shape = (
        df.loc[:, ["node_name", "ncore_total", "hl:mem_total"]]
        .drop_duplicates(subset=["node_name"])
        .copy()
    )
    node_shape["node_total_mem_mb"] = node_shape["hl:mem_total"].map(_memory_text_to_mb)
    node_shape["ncore_total"] = (
        pandas.to_numeric(node_shape["ncore_total"], errors="coerce").fillna(0).astype(int)
    )
    reservation_rows = reservation_rows.merge(node_shape, how="left", on="node_name")
    reservation_rows["reserved_cores_effective"] = (
        pandas.to_numeric(reservation_rows["reserved_cores"], errors="coerce").fillna(0).astype(int)
    )
    whole_node = (
        reservation_rows.get(
            "whole_node",
            pandas.Series(False, index=reservation_rows.index),
        )
        .fillna(False)
        .astype(bool)
    )
    reservation_rows.loc[whole_node, "reserved_cores_effective"] = (
        reservation_rows.loc[whole_node, "ncore_total"].fillna(0).astype(int)
    )
    reserved_mem_mb = reservation_rows.get(
        "reserved_mem_mb",
        pandas.Series(0, index=reservation_rows.index),
    )
    reservation_rows["reserved_mem_mb_effective"] = (
        pandas.to_numeric(reserved_mem_mb, errors="coerce").fillna(0).astype(int)
    )
    reservation_rows.loc[whole_node, "reserved_mem_mb_effective"] = (
        reservation_rows.loc[whole_node, "node_total_mem_mb"].fillna(0).astype(int)
    )
    needs_estimate = (
        (reservation_rows["reserved_mem_mb_effective"] <= 0)
        & (reservation_rows["reserved_cores_effective"] > 0)
        & (reservation_rows["ncore_total"] > 0)
    )
    reservation_rows.loc[needs_estimate, "reserved_mem_mb_effective"] = (
        (
            reservation_rows.loc[needs_estimate, "node_total_mem_mb"]
            * reservation_rows.loc[needs_estimate, "reserved_cores_effective"]
            / reservation_rows.loc[needs_estimate, "ncore_total"]
        )
        .round()
        .astype(int)
    )
    grouped = (
        reservation_rows.groupby("node_name", as_index=False)[
            ["reserved_cores_effective", "reserved_mem_mb_effective"]
        ]
        .sum()
        .rename(
            columns={
                "reserved_cores_effective": "reservation_cores_new",
                "reserved_mem_mb_effective": "reservation_mem_mb_new",
            }
        )
    )
    df = df.merge(grouped, how="left", on="node_name")
    for col in ["reservation_cores_new", "reservation_mem_mb_new"]:
        df[col] = pandas.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["reservation_cores"] = pandas.to_numeric(df["reservation_cores"], errors="coerce").fillna(
        0
    ).astype(int) + df.pop("reservation_cores_new")
    df["reservation_mem_mb"] = pandas.to_numeric(df["reservation_mem_mb"], errors="coerce").fillna(
        0
    ).astype(int) + df.pop("reservation_mem_mb_new")
    df["ncore_resv"] = (
        pandas.to_numeric(df["ncore_resv"], errors="coerce").fillna(0).astype(int)
        + df["reservation_cores"]
    )
    df["ncore_available"] = (
        (
            pandas.to_numeric(df["ncore_available"], errors="coerce").fillna(0).astype(int)
            - df["reservation_cores"]
        )
        .clip(lower=0)
        .astype(int)
    )
    available_mem_mb = df["hc:mem_req"].map(memory_text_to_mib)
    adjusted_mem_mb = (available_mem_mb - df["reservation_mem_mb"]).clip(lower=0)
    df["hc:mem_req"] = adjusted_mem_mb.map(
        lambda value: pandas.NA if pandas.isna(value) else f"{int(value)}M"
    )
    if "hc:mem_req_known" in df.columns:
        df["hc:mem_req_known"] = (
            df["hc:mem_req_known"].fillna(False).astype(bool) & adjusted_mem_mb.notna()
        )
    fully_reserved = (df["reservation_cores"] > 0) & (df["ncore_available"] <= 0)
    if fully_reserved.any():
        df.loc[fully_reserved, "status"] = (
            df.loc[fully_reserved, "status"]
            .fillna("")
            .astype(str)
            .map(lambda value: "|".join(token for token in [value, "reserved"] if token))
        )
    return df


def mark_unresolved_slurm_reservations(df_node, df_reservation=None):
    if df_node is None or df_node.shape[0] == 0:
        return df_node
    if "reservation_name" not in df_node.columns:
        return df_node
    df = df_node.copy()
    has_reservation_flag = (df["reservation_name"].fillna("").astype(str).str.strip() != "") | df[
        "slurm_state"
    ].fillna("").astype(str).str.contains("RESERVED", regex=False)
    if not has_reservation_flag.any():
        return df
    accounted = (
        pandas.to_numeric(df["reservation_cores"], errors="coerce").fillna(0) > 0
        if "reservation_cores" in df.columns
        else pandas.Series(False, index=df.index)
    )
    accessible = (
        df["reservation_accessible"].fillna(False).astype(bool)
        if "reservation_accessible" in df.columns
        else pandas.Series(False, index=df.index)
    )
    unresolved = has_reservation_flag & ~accounted & ~accessible
    if unresolved.any():
        df.loc[unresolved, "status"] = (
            df.loc[unresolved, "status"]
            .fillna("")
            .astype(str)
            .map(
                lambda value: "|".join(
                    token for token in [value, "reservation_unresolved"] if token
                )
            )
        )
        df.loc[unresolved, "ncore_available"] = 0
        df.loc[unresolved, "hc:mem_req"] = "0M"
    return df


def mark_slurm_metadata_unknown(df_node, reason):
    if df_node is None or df_node.shape[0] == 0:
        return df_node
    df = df_node.copy()
    df["status"] = (
        df["status"]
        .fillna("")
        .astype(str)
        .map(lambda value: "|".join(token for token in [value, reason] if token))
    )
    df["ncore_available"] = 0
    df["hc:mem_req"] = pandas.NA
    if "hc:mem_req_known" in df.columns:
        df["hc:mem_req_known"] = False
    return df


def get_sprio_df(lines):
    columns = [
        "job_id",
        "partition",
        "priority",
        "site",
        "age",
        "fairshare",
        "jobsize",
        "partition_factor",
    ]
    rows = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            continue
        if line.upper().startswith("JOBID"):
            continue
        items = [item.strip() for item in line.split("|")]
        if len(items) != 8:
            items = re.split(r"\s+", line)
        if len(items) != 8:
            continue
        rows.append(
            {
                "job_id": items[0],
                "partition": items[1],
                "priority": _safe_int(items[2], default=0),
                "site": _safe_int(items[3], default=0),
                "age": _safe_int(items[4], default=0),
                "fairshare": _safe_int(items[5], default=0),
                "jobsize": _safe_int(items[6], default=0),
                "partition_factor": _safe_int(items[7], default=0),
            }
        )
    return pandas.DataFrame(rows, columns=columns)


def get_sshare_df(lines):
    columns = [
        "account",
        "user",
        "raw_shares",
        "norm_shares",
        "raw_usage",
        "effective_usage",
        "fairshare",
    ]
    rows = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            continue
        if line.lower().startswith("account|"):
            continue
        items = line.split("|")
        if len(items) < 7:
            continue
        account = items[0].strip()
        user = items[1].strip()
        if user == "":
            continue
        rows.append(
            {
                "account": account,
                "user": user,
                "raw_shares": _safe_int(items[2], default=0),
                "norm_shares": pandas.to_numeric(items[3], errors="coerce"),
                "raw_usage": pandas.to_numeric(items[4], errors="coerce"),
                "effective_usage": pandas.to_numeric(items[5], errors="coerce"),
                "fairshare": pandas.to_numeric(items[6], errors="coerce"),
            }
        )
    df = pandas.DataFrame(rows, columns=columns)
    if df.shape[0] == 0:
        return df
    for col in ["norm_shares", "raw_usage", "effective_usage", "fairshare"]:
        df[col] = pandas.to_numeric(df[col], errors="coerce")
    df = df.loc[df["fairshare"].notna(), :].copy()
    return df.reset_index(drop=True)


def get_scontrol_node_df(lines, partition_state_map=None):
    columns = [
        "queue_name",
        "node_name",
        "qtype",
        "ncore_resv",
        "ncore_used",
        "ncore_total",
        "ncore_available",
        "np_load",
        "arch",
        "status",
        "hl:mem_total",
        "hc:mem_req",
        "hl:mem_total_known",
        "hc:mem_req_known",
        "slurm_state",
        "reservation_name",
    ]
    rows = []
    node_blocks = _split_scontrol_node_blocks(lines)
    for node_block in node_blocks:
        if "NodeName=" not in node_block:
            continue
        params = _parse_key_value_fields(node_block)
        node_name = params.get("NodeName", "")
        if node_name == "":
            continue
        partition_raw = params.get("Partitions", "")
        partitions = [p.strip().rstrip("*") for p in partition_raw.split(",") if p.strip() != ""]
        partitions = [p for p in partitions if p not in ["(null)", "N/A"]]
        if len(partitions) == 0:
            continue
        ncore_total = _safe_int(params.get("CPUEfctv", ""), default=0)
        if ncore_total <= 0:
            ncore_total = _safe_int(params.get("CPUTot", ""), default=0)
        ncore_total = max(ncore_total, 0)
        ncore_used = max(_safe_int(params.get("CPUAlloc", ""), default=0), 0)
        ncore_resv = 0
        ncore_available = max(ncore_total - ncore_used - ncore_resv, 0)
        mem_total_mb = _safe_int(params.get("RealMemory", ""), default=-1)
        mem_total_known = mem_total_mb >= 0
        if mem_total_known:
            mem_total_mb = max(mem_total_mb, 0)
        alloc_mem_mb = _safe_int(params.get("AllocMem", ""), default=-1)
        if mem_total_known and alloc_mem_mb >= 0:
            # Schedulable memory is constrained by Slurm's allocated memory,
            # not by the OS-level free page count.
            mem_available_mb = max(mem_total_mb - alloc_mem_mb, 0)
        else:
            mem_available_mb = _safe_int(params.get("FreeMem", ""), default=-1)
            if mem_available_mb >= 0:
                mem_available_mb = max(mem_available_mb, 0)
        mem_available_known = mem_available_mb >= 0
        slurm_state = params.get("State", "")
        reservation_name = params.get("ReservationName", "").strip()
        state_base = _normalize_slurm_node_state(slurm_state)
        flags = _slurm_state_flags(slurm_state)
        has_unavailable_flag = any((flag in SLURM_UNAVAILABLE_NODE_FLAGS) for flag in flags)
        node_status = (
            ""
            if ((state_base in SLURM_NORMAL_NODE_STATES) and (not has_unavailable_flag))
            else slurm_state
        )
        arch = params.get("Arch", "")
        for partition in partitions:
            partition_state = (
                ""
                if partition_state_map is None
                else str(partition_state_map.get(partition, "")).strip()
            )
            partition_status = ""
            if not _partition_state_is_up(partition_state):
                partition_status = f"partition_state={partition_state or 'UNKNOWN'}"
            status = node_status
            if (status != "") and (partition_status != ""):
                status = f"{status}|{partition_status}"
            elif partition_status != "":
                status = partition_status
            rows.append(
                {
                    "queue_name": partition,
                    "node_name": node_name,
                    "qtype": "SLURM",
                    "ncore_resv": ncore_resv,
                    "ncore_used": ncore_used,
                    "ncore_total": ncore_total,
                    "ncore_available": ncore_available,
                    "np_load": "",
                    "arch": arch,
                    "status": status,
                    "hl:mem_total": f"{mem_total_mb}M" if mem_total_known else pandas.NA,
                    "hc:mem_req": (f"{mem_available_mb}M" if mem_available_known else pandas.NA),
                    "hl:mem_total_known": mem_total_known,
                    "hc:mem_req_known": mem_available_known,
                    "slurm_state": slurm_state,
                    "reservation_name": reservation_name,
                }
            )
    df = pandas.DataFrame(rows, columns=columns)
    if df.shape[0] == 0:
        return df
    for col in ["ncore_resv", "ncore_used", "ncore_total", "ncore_available"]:
        df[col] = df[col].astype(int)
    df = df.sort_values(by=["queue_name", "node_name"]).reset_index(drop=True)
    return df


def _normalize_slurm_job_state(state_raw):
    if state_raw is None:
        return ""
    state = str(state_raw).strip().upper()
    if state == "":
        return ""
    m = re.match(r"^([A-Z_]+)", state)
    if m is not None:
        state = m.group(1)
    return SLURM_STATE_NAME_TO_CODE.get(state, state)


def _normalize_uge_job_state(state_raw, queue_name=""):
    state = str(state_raw).strip()
    state_lower = state.lower()
    if ("e" in state_lower) or ("d" in state_lower):
        return "F"
    if ("q" in state_lower) or (state_lower in {"h", "w"}):
        return "Q"
    if (str(queue_name).strip() != "") or any(marker in state_lower for marker in ["r", "s", "t"]):
        return "R"
    return ""


def print_queued_job_summary(df_user, scheduler="uge", current_user="", all_users=True):
    if scheduler == "slurm":
        if df_user.shape[0] == 0:
            print("No jobs found in squeue output.")
            print("")
            return
        state_codes = df_user["state"].fillna("").map(_normalize_slurm_job_state)
        is_running = state_codes.isin(SLURM_RUNNING_STATES)
        is_qwaiting = state_codes.isin(SLURM_PENDING_STATES)
        is_error = state_codes.isin(SLURM_ERROR_STATES)
        num_running = int(df_user.loc[is_running, "total_slots"].sum())
        num_qwaiting = int(df_user.loc[is_qwaiting, "total_slots"].sum())
        num_error = int(df_user.loc[is_error, "total_slots"].sum())
        if (current_user != "") and ("user" in df_user.columns):
            is_self = df_user["user"].fillna("") == current_user
            num_running_self = int(df_user.loc[is_running & is_self, "total_slots"].sum())
            num_qwaiting_self = int(df_user.loc[is_qwaiting & is_self, "total_slots"].sum())
            num_error_self = int(df_user.loc[is_error & is_self, "total_slots"].sum())
            print(
                f"jobs  self:R/Q/X={num_running_self}/{num_qwaiting_self}/{num_error_self}  "
                f"all:R/Q/X={num_running}/{num_qwaiting}/{num_error}"
            )
        else:
            print(f"# of running job tasks (estimated from squeue): {num_running}")
            print(f"# of queued job tasks (estimated from squeue): {num_qwaiting}")
            print(f"# of terminal/error job tasks currently visible in squeue: {num_error}")
        num_estimated_rows = int(df_user["task_count_estimated"].sum())
        if num_estimated_rows > 0:
            txt = "note: {} row(s) had truncated/irregular SLURM array IDs; task counts are estimated."
            print(txt.format(num_estimated_rows))
        print("")
        return
    if df_user.shape[0] == 0:
        print("No jobs found in AGE/UGE/SGE output.")
        print("")
        return
    if "queue_name" in df_user.columns:
        queue_names = df_user["queue_name"].fillna("")
    else:
        queue_names = pandas.Series("", index=df_user.index)
    state_codes = pandas.Series(
        [
            _normalize_uge_job_state(df_user.at[i, "state"], queue_names.at[i])
            for i in df_user.index
        ],
        index=df_user.index,
    )
    is_running = state_codes == "R"
    is_qwaiting = state_codes == "Q"
    is_error = state_codes == "F"
    num_running = int(df_user.loc[is_running, "total_slots"].sum())
    num_qwaiting = int(df_user.loc[is_qwaiting, "total_slots"].sum())
    num_error = int(df_user.loc[is_error, "total_slots"].sum())
    if not all_users:
        print(
            f"jobs  observed:R/Q/F={num_running}/{num_qwaiting}/{num_error}  (all-user status unavailable)"
        )
        print("")
        return
    if (current_user != "") and ("user" in df_user.columns):
        is_self = df_user["user"].fillna("") == current_user
        num_running_self = int(df_user.loc[is_running & is_self, "total_slots"].sum())
        num_qwaiting_self = int(df_user.loc[is_qwaiting & is_self, "total_slots"].sum())
        num_error_self = int(df_user.loc[is_error & is_self, "total_slots"].sum())
        print(
            f"jobs  self:R/Q/F={num_running_self}/{num_qwaiting_self}/{num_error_self}  all:R/Q/F={num_running}/{num_qwaiting}/{num_error}"
        )
    else:
        print(f"# of running AGE/UGE/SGE job slots: {num_running}")
        print(f"# of queued AGE/UGE/SGE job slots: {num_qwaiting}")
        print(f"# of AGE/UGE/SGE job slots in error: {num_error}")
    if "task_count_estimated" in df_user.columns:
        estimated_rows = int(df_user["task_count_estimated"].fillna(False).astype(bool).sum())
        if estimated_rows:
            print(
                f"note: {estimated_rows} AGE/UGE/SGE row(s) lack complete array-task metadata; "
                "slot counts may be underestimated."
            )
    print("")


def get_current_user_name():
    try:
        user_name = getpass.getuser().strip()
        if user_name:
            return user_name
    except (KeyError, OSError):
        user_name = ""
    return os.environ.get("USER", "").strip()


def _rank_fairshare_rows(df_share):
    if (df_share is None) or (df_share.shape[0] == 0):
        return pandas.DataFrame(
            columns=[
                "account",
                "user",
                "fairshare",
                "raw_usage",
                "effective_usage",
                "fairshare_rank",
            ]
        )
    df = df_share.copy()
    df = df.loc[df["fairshare"].notna(), :].copy()
    df = df.drop_duplicates(subset=["user", "account"], keep="first")
    df = df.sort_values(
        by=["fairshare", "user", "account"], ascending=[False, True, True]
    ).reset_index(drop=True)
    df["fairshare_rank"] = range(1, df.shape[0] + 1)
    return df


def _resolve_fairshare_account(df_share, user, account=""):
    account = str(account or "").strip()
    user = str(user or "").strip()
    if account != "":
        return account
    if (df_share is None) or (df_share.shape[0] == 0) or user == "":
        return ""
    matches = (
        df_share.loc[df_share["user"] == user, "account"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    matches = [value for value in matches if value != ""]
    if len(matches) == 1:
        return matches[0]
    return ""


def _current_user_fairshare_account(df_job, df_share, current_user):
    if current_user == "":
        return ""
    accounts = []
    if (df_job is not None) and (df_job.shape[0] > 0) and ("user" in df_job.columns):
        df_current = df_job.loc[df_job["user"].fillna("") == current_user, :].copy()
        if df_current.shape[0] > 0:
            account_series = (
                df_current["account"]
                if "account" in df_current.columns
                else pandas.Series([""] * df_current.shape[0])
            )
            account_values = account_series.fillna("").astype(str).str.strip()
            nonempty = sorted([value for value in account_values.unique().tolist() if value != ""])
            if len(nonempty) == 1:
                accounts = nonempty
    if len(accounts) == 1:
        return accounts[0]
    return _resolve_fairshare_account(df_share, current_user)


def get_slurm_fairshare_rank_summary(df_job, df_share, current_user=""):
    if current_user == "" or (df_share is None) or (df_share.shape[0] == 0):
        return None
    df_ranked = _rank_fairshare_rows(df_share)
    if df_ranked.shape[0] == 0:
        return None
    current_account = _current_user_fairshare_account(df_job, df_share, current_user)
    if current_account != "":
        current_rows = df_ranked.loc[
            (df_ranked["user"] == current_user) & (df_ranked["account"] == current_account), :
        ].copy()
    else:
        current_rows = df_ranked.loc[df_ranked["user"] == current_user, :].copy()
    if current_rows.shape[0] == 0:
        return None
    association_count = int(current_rows.shape[0])
    current_row = current_rows.sort_values(by=["fairshare_rank"]).iloc[0]
    current_account = str(current_row["account"])

    pending_rank = None
    pending_total = 0
    pending_missing = 0
    if (df_job is not None) and (df_job.shape[0] > 0) and ("user" in df_job.columns):
        state_codes = df_job["state"].fillna("").map(_normalize_slurm_job_state)
        df_pending = df_job.loc[state_codes.isin(SLURM_PENDING_STATES), :].copy()
        pairs = []
        seen = set()
        for _, row in df_pending.iterrows():
            user = str(row.get("user", "") or "").strip()
            account = str(row.get("account", "") or "").strip()
            if user == "":
                continue
            account = _resolve_fairshare_account(df_share, user, account)
            key = (user, account)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
        pending_rows = []
        for user, account in pairs:
            if account != "":
                matched = df_ranked.loc[
                    (df_ranked["user"] == user) & (df_ranked["account"] == account), :
                ].copy()
            else:
                matched = df_ranked.loc[df_ranked["user"] == user, :].copy()
            if matched.shape[0] == 0:
                pending_missing += 1
                continue
            pending_rows.append(matched.sort_values(by=["fairshare_rank"]).iloc[0].to_dict())
        if len(pending_rows) > 0:
            df_pending_ranked = pandas.DataFrame(pending_rows)
            df_pending_ranked = df_pending_ranked.sort_values(
                by=["fairshare", "user", "account"],
                ascending=[False, True, True],
            ).reset_index(drop=True)
            df_pending_ranked["pending_fairshare_rank"] = range(1, df_pending_ranked.shape[0] + 1)
            pending_total = int(df_pending_ranked.shape[0])
            current_pending = df_pending_ranked.loc[
                (df_pending_ranked["user"] == current_user)
                & (df_pending_ranked["account"] == current_account),
                :,
            ]
            if current_pending.shape[0] == 0:
                current_pending = df_pending_ranked.loc[
                    df_pending_ranked["user"] == current_user, :
                ]
            if current_pending.shape[0] > 0:
                pending_rank = int(
                    current_pending.sort_values(by=["pending_fairshare_rank"]).iloc[0][
                        "pending_fairshare_rank"
                    ]
                )

    return {
        "user": current_user,
        "account": current_account,
        "association_count": association_count,
        "fairshare": float(current_row["fairshare"]),
        "overall_rank": int(current_row["fairshare_rank"]),
        "overall_total": int(df_ranked.shape[0]),
        "pending_rank": pending_rank,
        "pending_total": pending_total,
        "pending_missing": pending_missing,
        "raw_usage": current_row.get("raw_usage", None),
        "effective_usage": current_row.get("effective_usage", None),
    }


def print_slurm_fairshare_rank_summary(summary):
    if summary is None:
        return
    fields = [
        "fairshare",
        "self={:.6f}".format(float(summary["fairshare"])),
    ]
    account = str(summary.get("account", "") or "").strip()
    if account != "":
        fields.append(f"account={account}")
    association_count = int(summary.get("association_count", 1) or 1)
    if association_count > 1:
        fields.append(f"selected=best_of_{association_count}_associations")
    fields.append(
        "assoc_rank={}/{}".format(int(summary["overall_rank"]), int(summary["overall_total"]))
    )
    pending_rank = summary.get("pending_rank", None)
    pending_total = int(summary.get("pending_total", 0) or 0)
    if pending_rank is not None and pending_total > 0:
        fields.append(f"pending_assoc_rank={int(pending_rank)}/{pending_total}")
    elif pending_total > 0:
        fields.append(f"pending_assoc_rank=n/a/{pending_total}")
    pending_missing = int(summary.get("pending_missing", 0) or 0)
    if pending_missing > 0:
        fields.append(f"pending_missing_fairshare={pending_missing}")
    print("  ".join(fields))
    print("")


def _split_slurm_partition_field(partition_field):
    partitions = []
    for token in str(partition_field or "").split(","):
        partition = token.strip().rstrip("*")
        if partition in ["", "(null)", "N/A"]:
            continue
        partitions.append(partition)
    return partitions


def _slurm_partition_field_matches(partition_field, queue_name):
    queue_name = str(queue_name or "").strip().rstrip("*")
    if queue_name == "":
        return False
    return queue_name in _split_slurm_partition_field(partition_field)


def get_slurm_launch_heuristic_df(df_node, df_job, df_prio=None, current_user=""):
    columns = [
        "queue_name",
        "recommended_cores",
        "recommended_mem_gb",
        "recommended_mem_gib",
        "top_node_name",
        "top_node_cores",
        "top_node_mem_gb",
        "top_node_mem_gib",
        "priority_gap",
        "fairshare_gap",
        "blocked_req_cores",
        "blocked_req_mem_gb",
        "blocked_req_mem_gib",
        "blocked_time_limit",
        "status",
    ]
    if (df_node is None) or (df_node.shape[0] == 0):
        return pandas.DataFrame(columns=columns)
    rows = []
    queue_names = sorted(
        [
            q
            for q in df_node["queue_name"].dropna().unique().tolist()
            if not str(q).startswith("login")
        ]
    )
    for queue_name in queue_names:
        df_queue = df_node.loc[
            (df_node["queue_name"] == queue_name) & (df_node["status"] == ""), :
        ].copy()
        if df_queue.shape[0] == 0:
            rows.append(
                {
                    "queue_name": queue_name,
                    "recommended_cores": 0,
                    "recommended_mem_gb": 0.0,
                    "recommended_mem_gib": 0.0,
                    "top_node_name": "",
                    "top_node_cores": 0,
                    "top_node_mem_gb": 0.0,
                    "top_node_mem_gib": 0.0,
                    "priority_gap": None,
                    "fairshare_gap": None,
                    "blocked_req_cores": None,
                    "blocked_req_mem_gb": None,
                    "blocked_req_mem_gib": None,
                    "blocked_time_limit": "",
                    "status": "no_normal_nodes",
                }
            )
            continue
        df_queue["available_mem_gb"] = _memory_series_to_gb(df_queue["hc:mem_req"])
        df_queue = df_queue.sort_values(
            by=["ncore_available", "available_mem_gb", "node_name"], ascending=[False, False, True]
        ).reset_index(drop=True)
        top_node = df_queue.iloc[0]
        top_node_cores = int(top_node["ncore_available"])
        top_node_mem_gb = float(top_node["available_mem_gb"])
        recommended_cores = top_node_cores
        recommended_mem_gb = top_node_mem_gb
        priority_gap = None
        fairshare_gap = None
        blocked_req_cores = None
        blocked_req_mem_gb = None
        blocked_time_limit = ""
        status = "resource_only"
        if (current_user != "") and (df_job is not None) and (df_job.shape[0] > 0):
            state_codes = df_job["state"].fillna("").map(_normalize_slurm_job_state)
            job_partition_matches = df_job["partition"].map(
                lambda partition, target=queue_name: _slurm_partition_field_matches(
                    partition, target
                )
            )
            user_pending = df_job.loc[
                job_partition_matches
                & (df_job["user"] == current_user)
                & state_codes.isin(SLURM_PENDING_STATES),
                :,
            ].copy()
            if user_pending.shape[0] > 0:
                if (df_prio is not None) and (df_prio.shape[0] > 0):
                    prio_partition_matches = df_prio["partition"].map(
                        lambda partition, target=queue_name: _slurm_partition_field_matches(
                            partition, target
                        )
                    )
                    df_prio_queue = df_prio.loc[prio_partition_matches, :].copy()
                    if df_prio_queue.shape[0] > 0:
                        top_priority = int(df_prio_queue["priority"].max())
                        top_fairshare = int(df_prio_queue["fairshare"].max())
                        df_user_prio = df_prio_queue.loc[
                            df_prio_queue["job_id"].isin(user_pending["job_id"]), :
                        ].copy()
                        if df_user_prio.shape[0] > 0:
                            user_best_priority = int(df_user_prio["priority"].max())
                            user_best_fairshare = int(df_user_prio["fairshare"].max())
                            priority_gap = top_priority - user_best_priority
                            fairshare_gap = top_fairshare - user_best_fairshare
                user_priority_pending = user_pending.loc[
                    user_pending["pending_reason"]
                    .fillna("")
                    .str.contains("Priority", case=False, regex=False),
                    :,
                ].copy()
                if user_priority_pending.shape[0] > 0:
                    if "resource_fields_complete" not in user_priority_pending.columns:
                        user_priority_pending["resource_fields_complete"] = False
                    if "num_nodes" not in user_priority_pending.columns:
                        user_priority_pending["num_nodes"] = 1
                    valid_priority_pending = user_priority_pending.loc[
                        user_priority_pending["resource_fields_complete"].fillna(False)
                        & (user_priority_pending["num_nodes"] == 1),
                        :,
                    ].copy()
                    if valid_priority_pending.shape[0] > 0:
                        valid_priority_pending["req_mem_gb"] = valid_priority_pending.apply(
                            lambda row: slurm_request_memory_gib(
                                row["req_mem"],
                                req_cpus=row["req_cpus"],
                                num_nodes=row["num_nodes"],
                            ),
                            axis=1,
                        )
                        valid_priority_pending = valid_priority_pending.loc[
                            valid_priority_pending["req_mem_gb"].notna(), :
                        ].copy()
                    if valid_priority_pending.shape[0] > 0:
                        valid_priority_pending["time_limit_minutes"] = valid_priority_pending[
                            "time_limit"
                        ].map(_slurm_time_to_minutes)
                        valid_priority_pending = valid_priority_pending.sort_values(
                            by=["req_cpus", "req_mem_gb", "time_limit_minutes", "job_id"],
                            ascending=[True, True, True, True],
                        ).reset_index(drop=True)
                        smallest = valid_priority_pending.iloc[0]
                        blocked_req_cores = int(smallest["req_cpus"])
                        blocked_req_mem_gb = float(smallest["req_mem_gb"])
                        blocked_time_limit = str(smallest["time_limit"]).strip()
                        status = "priority_blocked"
                    else:
                        status = "priority_blocked_missing_fields"
        rows.append(
            {
                "queue_name": queue_name,
                "recommended_cores": recommended_cores,
                "recommended_mem_gb": recommended_mem_gb,
                "recommended_mem_gib": recommended_mem_gb,
                "top_node_name": str(top_node["node_name"]),
                "top_node_cores": top_node_cores,
                "top_node_mem_gb": top_node_mem_gb,
                "top_node_mem_gib": top_node_mem_gb,
                "priority_gap": priority_gap,
                "fairshare_gap": fairshare_gap,
                "blocked_req_cores": blocked_req_cores,
                "blocked_req_mem_gb": blocked_req_mem_gb,
                "blocked_req_mem_gib": blocked_req_mem_gb,
                "blocked_time_limit": blocked_time_limit,
                "status": status,
            }
        )
    return pandas.DataFrame(rows, columns=columns)


def _format_slurm_compact_time_limit(time_limit):
    txt = str(time_limit).strip()
    if txt in ["", "nan", "N/A", "NOT_SET"]:
        return "?"
    total_minutes = _slurm_time_to_minutes(txt)
    if total_minutes == float("inf"):
        return "inf"
    total_minutes = int(round(total_minutes))
    days = int(total_minutes / (24 * 60))
    rem_minutes = total_minutes - (days * 24 * 60)
    hours = int(rem_minutes / 60)
    minutes = rem_minutes - (hours * 60)
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or len(parts) == 0:
        parts.append(f"{minutes}m")
    return "".join(parts[:2])


def _format_slurm_compact_node(node_name, ncore_available, mem_gb):
    if str(node_name).strip() == "":
        return "-"
    mem_floor = floor_gib(mem_gb)
    mem_text = "?GiB" if mem_floor is None else f"{mem_floor}GiB"
    return f"{node_name} {int(ncore_available)}c/{mem_text}"


def _format_compact_top_nodes(df_nodes, primary_col, secondary_col, args):
    if df_nodes.shape[0] == 0:
        return "-"
    ordered = df_nodes.sort_values(
        by=[primary_col, secondary_col, "node_name"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ntop = max(int(getattr(args, "ntop", 1)), 1)
    limit = min(ntop, ordered.shape[0])
    if getattr(args, "all_tiers", False):
        threshold = ordered.at[limit - 1, primary_col]
        selected = ordered.loc[ordered[primary_col] >= threshold, :]
    else:
        selected = ordered.iloc[:limit, :]
    return ", ".join(
        _format_slurm_compact_node(
            row["node_name"],
            row["ncore_available"],
            row["hc:mem_req"],
        )
        for _, row in selected.iterrows()
    )


def _format_slurm_compact_launch_row(row):
    if row is None:
        return "-"
    status = str(row.get("status", "")).strip()
    recommended_cores = row.get("recommended_cores", None)
    recommended_mem_gb = row.get("recommended_mem_gb", row.get("recommended_mem_gib", None))
    blocked_req_cores = row.get("blocked_req_cores", None)
    blocked_req_mem_gb = row.get("blocked_req_mem_gb", row.get("blocked_req_mem_gib", None))
    blocked_time_limit = row.get("blocked_time_limit", "")
    priority_gap = row.get("priority_gap", None)
    fairshare_gap = row.get("fairshare_gap", None)
    if pandas.isna(recommended_cores):
        resource_fields = ["n/a"]
    else:
        memory_floor = floor_gib(recommended_mem_gb)
        memory_text = "?GiB" if memory_floor is None else f"{memory_floor}GiB"
        resource_fields = [f"res<={int(recommended_cores)}c/{memory_text}"]
    if status in ["priority_blocked", "priority_blocked_missing_fields"]:
        fields = resource_fields + ["PRIO"]
        if pandas.notna(blocked_req_cores):
            blocked_memory_floor = floor_gib(blocked_req_mem_gb)
            blocked_memory_text = (
                "?GiB" if blocked_memory_floor is None else f"{blocked_memory_floor}GiB"
            )
            fields.append(
                f"min={int(blocked_req_cores)}c/{blocked_memory_text}/{_format_slurm_compact_time_limit(blocked_time_limit)}"
            )
        else:
            fields.append("min=?")
        if pandas.notna(priority_gap):
            fields.append(f"gap={int(priority_gap)}")
        if pandas.notna(fairshare_gap):
            fields.append(f"fs={int(fairshare_gap)}")
        return " ".join(fields)
    return resource_fields[0]


def print_slurm_compact_summary(df, df_launch, args):
    queue_names = [q for q in df["queue_name"].unique().tolist() if not str(q).startswith("login")]
    launch_rows = {}
    if (df_launch is not None) and (df_launch.shape[0] > 0):
        for i in df_launch.index:
            queue_name = df_launch.at[i, "queue_name"]
            launch_rows[queue_name] = df_launch.loc[i, :].to_dict()
    rows = []
    for queue_name in queue_names:
        df_queue = df.loc[(df["queue_name"] == queue_name), :].reset_index(drop=True)
        is_abnormal_status = df_queue["status"] != ""
        num_abnormal_node = int(is_abnormal_status.sum())
        num_node = int(df_queue.shape[0])
        num_working_node = num_node - num_abnormal_node
        ncore_total = int(df_queue.loc[:, "ncore_total"].sum())
        ncore_used = int(df_queue.loc[~is_abnormal_status, "ncore_used"].sum())
        ncore_available = int(df_queue.loc[~is_abnormal_status, "ncore_available"].sum())
        mem_total = df_queue.loc[:, "hl:mem_total"].sum(min_count=1)
        mem_available = df_queue.loc[~is_abnormal_status, "hc:mem_req"].sum(min_count=1)
        if args.exclude_abnormal_node:
            df_normal = df_queue.loc[~is_abnormal_status, :].copy()
        else:
            df_normal = df_queue.copy()
        if df_normal.shape[0] > 0:
            top_cpu = _format_compact_top_nodes(
                df_normal,
                "ncore_available",
                "hc:mem_req",
                args,
            )
            top_ram = _format_compact_top_nodes(
                df_normal,
                "hc:mem_req",
                "ncore_available",
                args,
            )
            if top_cpu == top_ram:
                top_ram = "same"
        else:
            top_cpu = "-"
            top_ram = "-"
        rows.append(
            {
                "part": str(queue_name),
                "nodes": f"{num_working_node}/{num_abnormal_node}/{num_node}",
                "cpu(a/u/t)": f"{ncore_available}/{ncore_used}/{ncore_total}",
                "ram(a/t)GiB": "{}/{}".format(
                    "?" if pandas.isna(mem_available) else floor_gib(mem_available),
                    "?" if pandas.isna(mem_total) else floor_gib(mem_total),
                ),
                "topCPU": top_cpu,
                "topRAM": top_ram,
                "launch": _format_slurm_compact_launch_row(launch_rows.get(queue_name)),
            }
        )
    if len(rows) == 0:
        return
    columns = ["part", "nodes", "cpu(a/u/t)", "ram(a/t)GiB", "topCPU", "topRAM", "launch"]
    widths = {}
    for col in columns:
        widths[col] = len(col)
        for row in rows:
            widths[col] = max(widths[col], len(str(row[col])))
    header = "  ".join(
        [columns[0].ljust(widths[columns[0]])] + [col.ljust(widths[col]) for col in columns[1:]]
    )
    print(header)
    for row in rows:
        print(
            "  ".join(
                [
                    str(row["part"]).ljust(widths["part"]),
                    str(row["nodes"]).ljust(widths["nodes"]),
                    str(row["cpu(a/u/t)"]).ljust(widths["cpu(a/u/t)"]),
                    str(row["ram(a/t)GiB"]).ljust(widths["ram(a/t)GiB"]),
                    str(row["topCPU"]).ljust(widths["topCPU"]),
                    str(row["topRAM"]).ljust(widths["topRAM"]),
                    str(row["launch"]).ljust(widths["launch"]),
                ]
            )
        )
    print("")
    print(
        "legend: nodes=working/abnormal/total, cpu=available/used/total, "
        "ram=available/total, launch=res=CPU/RAM-only ceiling"
    )
    print("")


def _format_qfree_int(value, zero_as_inf=False):
    if value is None or pandas.isna(value):
        return "-"
    number = int(value)
    if zero_as_inf and number == 0:
        return "inf"
    return str(number)


def print_uge_compact_summary(df, df_qfree, args):
    qfree_rows = {}
    queue_names = df["queue_name"].dropna().astype(str).unique().tolist()
    if (df_qfree is not None) and (df_qfree.shape[0] > 0):
        qfree_queue_names = df_qfree["queue_name"].dropna().astype(str).tolist()
        queue_names.extend(queue for queue in qfree_queue_names if queue not in queue_names)
        for i in df_qfree.index:
            qfree_rows[str(df_qfree.at[i, "queue_name"])] = df_qfree.loc[i, :].to_dict()
    rows = []
    for queue_name in queue_names:
        df_queue = df.loc[(df["queue_name"] == queue_name), :].reset_index(drop=True)
        if df_queue.shape[0] > 0:
            is_abnormal_status = df_queue["status"] != ""
            num_abnormal_node = int(is_abnormal_status.sum())
            num_node = int(df_queue.shape[0])
            num_working_node = num_node - num_abnormal_node
            ncore_total = int(df_queue.loc[:, "ncore_total"].sum())
            ncore_used = int(df_queue.loc[~is_abnormal_status, "ncore_used"].sum())
            ncore_available = int(df_queue.loc[~is_abnormal_status, "ncore_available"].sum())
            mem_total = df_queue.loc[:, "hl:mem_total"].sum(min_count=1)
            mem_available = df_queue.loc[~is_abnormal_status, "hc:mem_req"].sum(min_count=1)
            if args.exclude_abnormal_node:
                df_normal = df_queue.loc[~is_abnormal_status, :].copy()
            else:
                df_normal = df_queue.copy()
            if df_normal.shape[0] > 0:
                top_cpu = _format_compact_top_nodes(
                    df_normal,
                    "ncore_available",
                    "hc:mem_req",
                    args,
                )
                top_ram = _format_compact_top_nodes(
                    df_normal,
                    "hc:mem_req",
                    "ncore_available",
                    args,
                )
                if top_cpu == top_ram:
                    top_ram = "same"
            else:
                top_cpu = "-"
                top_ram = "-"
        else:
            num_working_node = 0
            num_abnormal_node = 0
            num_node = 0
            ncore_available = 0
            ncore_used = 0
            ncore_total = 0
            mem_available = float("nan")
            mem_total = float("nan")
            top_cpu = "-"
            top_ram = "-"
        qfree_row = qfree_rows.get(queue_name)
        if qfree_row is None:
            quota = "-"
            launch_2g = "-"
        else:
            quota = "{}/{}/{}".format(
                _format_qfree_int(qfree_row.get("self_slots")),
                _format_qfree_int(qfree_row.get("group_slots")),
                _format_qfree_int(qfree_row.get("quota_slots"), zero_as_inf=True),
            )
            qfree_total_mem = qfree_row.get("total_mem_gb")
            qfree_used_mem = qfree_row.get("all_mem_req_gb")
            if (
                (qfree_total_mem is not None)
                and pandas.notna(qfree_total_mem)
                and (qfree_used_mem is not None)
                and pandas.notna(qfree_used_mem)
            ):
                mem_total = float(qfree_total_mem)
                mem_available = max(mem_total - float(qfree_used_mem), 0.0)
            available_slots = _format_qfree_int(qfree_row.get("available_slots_2g"))
            standby_slots = qfree_row.get("standby_slots")
            if (
                (standby_slots is not None)
                and pandas.notna(standby_slots)
                and (int(standby_slots) > 0)
            ):
                launch_2g = f"{available_slots}(+{int(standby_slots)}s)"
            else:
                launch_2g = available_slots
        rows.append(
            {
                "queue": queue_name,
                "nodes": f"{num_working_node}/{num_abnormal_node}/{num_node}",
                "cpu(a/u/t)": f"{ncore_available}/{ncore_used}/{ncore_total}",
                "ram(a/t)GiB": "{}/{}".format(
                    "?" if pandas.isna(mem_available) else floor_gib(mem_available),
                    "?" if pandas.isna(mem_total) else floor_gib(mem_total),
                ),
                "topCPU": top_cpu,
                "topRAM": top_ram,
                "quota(s/g/l)": quota,
                "launch2G": launch_2g,
            }
        )
    if len(rows) == 0:
        return
    columns = [
        "queue",
        "nodes",
        "cpu(a/u/t)",
        "ram(a/t)GiB",
        "topCPU",
        "topRAM",
        "quota(s/g/l)",
        "launch2G",
    ]
    widths = {}
    for col in columns:
        widths[col] = max([len(col)] + [len(str(row[col])) for row in rows])
    print("  ".join([col.ljust(widths[col]) for col in columns]))
    for row in rows:
        print("  ".join([str(row[col]).ljust(widths[col]) for col in columns]))
    print("")
    print("legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total")
    if qfree_rows:
        print(
            "        ram uses qfree request headroom/capacity; topRAM is the best queue-instance request headroom"
        )
        print(
            "        quota=self/group/limit slots (inf=unlimited), launch2G=immediate 2G slots (+standby)"
        )
    print("")


def get_scheduler_from_command(stat_command):
    try:
        command = shlex.split(stat_command)
    except ValueError:
        return None
    if len(command) == 0:
        return None
    executable = os.path.basename(command[0])
    if executable == "qstat":
        return "uge"
    if executable == "squeue":
        return "slurm"
    return None


def _strip_squeue_parse_options(command):
    stripped = [command[0]]
    skip_next = False
    for token in command[1:]:
        if skip_next:
            skip_next = False
            continue
        if token in ["-h", "--noheader"]:
            continue
        if token.startswith("--noheader="):
            continue
        if token in ["-o", "-O", "--format", "--Format"]:
            skip_next = True
            continue
        if token.startswith("--format=") or token.startswith("--Format="):
            continue
        if token.startswith("-o") and token != "-o":
            continue
        if token.startswith("-O") and token != "-O":
            continue
        stripped.append(token)
    return stripped


def get_squeue_command_for_parsing(stat_command):
    try:
        command = shlex.split(stat_command)
    except ValueError:
        return stat_command
    if len(command) == 0:
        return stat_command
    executable = os.path.basename(command[0])
    if executable != "squeue":
        return stat_command
    command = _strip_squeue_parse_options(command)
    command.append("-h")
    command.extend(["-o", SLURM_SQUEUE_PARSE_FIELDS])
    return " ".join([shlex.quote(item) for item in command])


def _command_timeout_from_args(args):
    return getattr(args, "command_timeout", DEFAULT_COMMAND_TIMEOUT_SECONDS)


def _print_degraded(component, detail):
    print(f"note: degraded {component} data: {detail}")


def _print_slurm_fairshare(args, df_user, current_user, timeout_seconds):
    if not getattr(args, "show_fairshare_rank", True):
        return
    share_lines = get_command_stdout_lines(
        command_str=getattr(args, "slurm_share_command", "sshare -a -P"),
        example_file=getattr(args, "slurm_share_example_file", ""),
        allow_failure=True,
        command_name="--slurm_share_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if share_lines is None:
        _print_degraded("Slurm FairShare", "--slurm_share_command failed or timed out")
        return
    df_share = get_sshare_df(share_lines)
    summary = get_slurm_fairshare_rank_summary(
        df_job=df_user,
        df_share=df_share,
        current_user=current_user,
    )
    if summary is not None:
        print_slurm_fairshare_rank_summary(summary)
    elif df_share.shape[0] == 0:
        _print_degraded(
            "Slurm FairShare",
            "command succeeded but no association rows were parsed",
        )


def _get_slurm_partition_state_map(args, timeout_seconds):
    partition_lines = get_command_stdout_lines(
        command_str=args.slurm_partition_command,
        example_file=args.slurm_partition_example_file,
        allow_failure=True,
        command_name="--slurm_partition_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if partition_lines is None:
        _print_degraded("Slurm partition", "--slurm_partition_command failed or timed out")
        return None
    df_partition = get_scontrol_partition_df(partition_lines)
    if df_partition.shape[0] == 0:
        _print_degraded(
            "Slurm partition",
            "command succeeded but no partition rows were parsed",
        )
        return None
    return df_partition.set_index("partition_name")["partition_state"].to_dict()


def _get_slurm_df(args, timeout_seconds):
    current_user = get_current_user_name()
    lines = get_command_stdout_lines(
        command_str=get_squeue_command_for_parsing(args.stat_command),
        example_file=args.example_file,
        allow_failure=False,
        command_name="--stat_command",
        timeout_seconds=timeout_seconds,
    )
    df_user = get_squeue_user_df(lines)
    print_queued_job_summary(df_user, scheduler="slurm", current_user=current_user)
    _print_slurm_fairshare(args, df_user, current_user, timeout_seconds)
    partition_state_map = _get_slurm_partition_state_map(args, timeout_seconds)
    node_lines = get_command_stdout_lines(
        command_str=args.slurm_node_command,
        example_file=args.slurm_node_example_file,
        allow_failure=True,
        command_name="--slurm_node_command",
        timeout_seconds=timeout_seconds,
    )
    if node_lines is None:
        print("Skipping node resource summary because --slurm_node_command failed.\n")
        return None, df_user
    df_node = get_scontrol_node_df(node_lines, partition_state_map=partition_state_map)
    if df_node.shape[0] == 0:
        print("Skipping node resource summary because SLURM node output could not be parsed.")
        print(
            'Use --slurm_node_command "scontrol show node -o" or provide --slurm_node_example_file.\n'
        )
        return None, df_user
    return df_node, df_user


def _get_uge_all_user_jobs(args, fallback, timeout_seconds):
    command = getattr(args, "uge_job_command", "")
    example_file = getattr(args, "uge_job_example_file", "")
    if command == "" and example_file == "":
        return fallback, False
    job_lines = get_command_stdout_lines(
        command_str=command,
        example_file=example_file,
        allow_failure=True,
        command_name="--uge_job_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if job_lines is None:
        _print_degraded(
            "AGE/UGE/SGE all-user jobs",
            "--uge_job_command failed or timed out; using jobs embedded in qstat -F",
        )
        return fallback, False
    payload = "\n".join(job_lines).lstrip()
    if not payload.startswith(("{", "[")):
        return get_user_df(job_lines), True
    parsed = get_uge_json_job_df(job_lines)
    if parsed is not None:
        return parsed, True
    _print_degraded(
        "AGE/UGE/SGE all-user jobs",
        "JSON schema was not recognized; using jobs embedded in qstat -F",
    )
    return fallback, False


def _get_uge_df(args, timeout_seconds):
    if args.niter < 1:
        raise KFBatchUsageError("Exiting. --niter must be >= 1 when using qstat mode.")
    df = None
    df_user = None
    for iteration in range(args.niter):
        lines = get_command_stdout_lines(
            command_str=args.stat_command,
            example_file=args.example_file,
            allow_failure=False,
            command_name="--stat_command",
            timeout_seconds=timeout_seconds,
        )
        snapshot = get_qstat_df(lines)
        if snapshot.shape[0] == 0:
            raise KFBatchCommandError(
                f"AGE/UGE/SGE resource snapshot {iteration + 1} "
                "contained no parseable queue instances."
            )
        if iteration > 0:
            df = _merge_qstat_iteration_min_availability(df, snapshot)
            continue
        df = snapshot
        df_user, has_all_user_jobs = _get_uge_all_user_jobs(
            args,
            get_user_df(lines),
            timeout_seconds,
        )
        print_queued_job_summary(
            df_user,
            scheduler="uge",
            current_user=get_current_user_name(),
            all_users=has_all_user_jobs,
        )
    return df, df_user


def get_df(args):
    scheduler_override = getattr(args, "scheduler", "auto")
    scheduler = (
        get_scheduler_from_command(args.stat_command)
        if scheduler_override == "auto"
        else scheduler_override
    )
    if scheduler is None:
        raise KFBatchUsageError(f"Exiting. --stat_command does not support: {args.stat_command}")
    timeout_seconds = _command_timeout_from_args(args)
    if scheduler == "slurm":
        df, df_user = _get_slurm_df(args, timeout_seconds)
    else:
        df, df_user = _get_uge_df(args, timeout_seconds)
    return scheduler, df, df_user


def adjust_ram_unit(df):
    for col in ["hc:mem_req", "hl:mem_total"]:
        values = memory_series_to_gib(df[col])
        known_col = col + "_known"
        if known_col in df.columns:
            known = df[known_col].fillna(False).astype(bool) & values.notna()
        else:
            known = values.notna()
            df[known_col] = known
        df[col] = values.where(known, float("nan"))
        df[col + "_unit"] = pandas.Series(
            ["GiB" if value else "" for value in known],
            index=df.index,
            dtype="string",
        )
    return df


def _normalized_output_path(path):
    return pathlib.Path(path).expanduser().resolve(strict=False)


def _atomic_write_tsv(df, output_path, label):
    target = _normalized_output_path(output_path)
    if not target.parent.is_dir():
        raise KFBatchCommandError(
            f"Failed to write {label}: parent directory does not exist: {target.parent}"
        )
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary_path = pathlib.Path(handle.name)
            df.to_csv(handle, sep="\t", index=False)
        os.replace(temporary_path, target)
    except (OSError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise KFBatchCommandError(f"Failed to write {label}: {target}: {error}") from error


def _resolve_output_paths(args):
    legacy_out = getattr(args, "out", "")
    explicit_node_out = getattr(args, "out_nodes", "")
    if legacy_out and explicit_node_out and legacy_out != explicit_node_out:
        raise KFBatchUsageError(
            "--out and --out_nodes refer to the same node table; specify only one path."
        )
    node_output_path = explicit_node_out or legacy_out
    job_output_path = getattr(args, "out_jobs", "")
    if (
        node_output_path
        and job_output_path
        and _normalized_output_path(node_output_path) == _normalized_output_path(job_output_path)
    ):
        raise KFBatchUsageError("--out_jobs and --out_nodes must refer to different files.")
    return node_output_path, job_output_path


def _require_slurm_node_data(df, node_output_path):
    if df is not None:
        return
    print("Skipping cluster/node resource availability.")
    print("Reason: no parsed SLURM node data was available.")
    print('Provide --slurm_node_command or --slurm_node_example_file from "scontrol show node -o".')
    if node_output_path:
        _print_degraded(
            "node TSV",
            f"{node_output_path} was not written because no node table was available",
        )
    raise KFBatchCommandError("Slurm node/resource data is unavailable.")


def _get_current_slurm_accounts(df_user, current_user):
    if "account" not in df_user.columns:
        return set()
    accounts = set(
        df_user.loc[df_user["user"].fillna("") == current_user, "account"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    accounts.discard("")
    return accounts


def _apply_slurm_reservation_state(df, df_user, args, timeout_seconds):
    reservation_lines = get_command_stdout_lines(
        command_str=args.slurm_reservation_command,
        example_file=args.slurm_reservation_example_file,
        allow_failure=True,
        command_name="--slurm_reservation_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if reservation_lines is None:
        _print_degraded(
            "Slurm reservation",
            "--slurm_reservation_command failed or timed out; resource ceilings are suppressed",
        )
        return mark_unresolved_slurm_reservations(
            mark_slurm_metadata_unknown(df, "reservation_state=UNKNOWN"),
            None,
        )
    current_user = get_current_user_name()
    df_reservation = get_scontrol_reservation_df(
        reservation_lines,
        current_user=current_user,
        current_accounts=_get_current_slurm_accounts(df_user, current_user),
    )
    for warning in df_reservation.attrs.get("warnings", []):
        _print_degraded("Slurm reservation", warning)
    if df_reservation.shape[0] > 0:
        df = apply_slurm_reservations(df, df_reservation)
        for target in df.attrs.get("reservation_unresolved_targets", []):
            _print_degraded(
                "Slurm reservation",
                f"target could not be matched to a parsed node: {target}",
            )
    return mark_unresolved_slurm_reservations(df, df_reservation)


def _get_slurm_launch_frame(df, df_user, args, timeout_seconds):
    if not args.show_launch_heuristic:
        return None
    prio_lines = get_command_stdout_lines(
        command_str=args.slurm_prio_command,
        example_file=args.slurm_prio_example_file,
        allow_failure=True,
        command_name="--slurm_prio_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    df_prio = None
    if prio_lines is None:
        _print_degraded(
            "Slurm priority",
            "--slurm_prio_command failed or timed out; launch estimates are resource-only",
        )
    else:
        df_prio = get_sprio_df(prio_lines)
        if df_prio.shape[0] == 0:
            _print_degraded(
                "Slurm priority",
                "command succeeded but no priority rows were parsed",
            )
    return get_slurm_launch_heuristic_df(
        df_node=df,
        df_job=df_user,
        df_prio=df_prio,
        current_user=get_current_user_name(),
    )


def _get_qfree_frame(args, timeout_seconds):
    qfree_command = getattr(args, "uge_qfree_command", "")
    qfree_lines = get_command_stdout_lines(
        command_str=qfree_command,
        example_file=getattr(args, "uge_qfree_example_file", ""),
        allow_failure=True,
        command_name="--uge_qfree_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if qfree_lines is None:
        if qfree_command:
            _print_degraded(
                "qfree",
                "--uge_qfree_command failed or timed out; quota columns are unavailable",
            )
        return None
    df_qfree = get_qfree_df(qfree_lines)
    if df_qfree.shape[0] == 0:
        _print_degraded(
            "qfree",
            "command succeeded but no queue summaries were parsed",
        )
    return df_qfree


def stat_main(args):
    node_output_path, job_output_path = _resolve_output_paths(args)
    timeout_seconds = _command_timeout_from_args(args)
    scheduler, df, df_user = get_df(args)
    if job_output_path:
        _atomic_write_tsv(df_user, job_output_path, "job TSV")
    if scheduler == "slurm":
        _require_slurm_node_data(df, node_output_path)
        df = _apply_slurm_reservation_state(df, df_user, args, timeout_seconds)
    df = adjust_ram_unit(df)
    if scheduler == "slurm":
        print_slurm_compact_summary(
            df,
            _get_slurm_launch_frame(df, df_user, args, timeout_seconds),
            args,
        )
    else:
        print_uge_compact_summary(df, _get_qfree_frame(args, timeout_seconds), args)
    if node_output_path:
        _atomic_write_tsv(df, node_output_path, "node TSV")
