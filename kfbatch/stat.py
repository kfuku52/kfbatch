import getpass
import json
import os
import re
import shlex

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
SLURM_SQUEUE_PARSE_FIELDS = "%i\t%P\t%j\t%u\t%t\t%M\t%D\t%C\t%m\t%l\t%R"
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


def _count_uge_task_expression(task_expression):
    return _parse_uge_task_expression(task_expression)[0]


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


def _memory_series_to_gib(series):
    return _memory_series_to_gb(series)


def _memory_text_to_gb(value):
    return memory_text_to_gib(value)


def _memory_text_to_gib(value):
    return _memory_text_to_gb(value)


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
    if len(common_index) > 0:
        base_cores = pandas.to_numeric(
            df_base.loc[common_index, "ncore_available"], errors="coerce"
        ).fillna(0)
        new_cores = pandas.to_numeric(
            df_new.loc[common_index, "ncore_available"], errors="coerce"
        ).fillna(0)
        min_cores = pandas.concat([base_cores, new_cores], axis=1).min(axis=1)
        df_base.loc[common_index, "ncore_available"] = min_cores.astype(int)
        base_mem = _memory_series_to_gb(df_base.loc[common_index, "hc:mem_req"])
        new_mem = _memory_series_to_gb(df_new.loc[common_index, "hc:mem_req"])
        min_mem = pandas.concat([base_mem, new_mem], axis=1).min(axis=1, skipna=True)
        df_base.loc[common_index, "hc:mem_req"] = min_mem.map(
            lambda value: pandas.NA if pandas.isna(value) else f"{float(value):.3f}G"
        )
    missing_from_new = df_base.index.difference(df_new.index)
    new_since_first = df_new.index.difference(df_base.index)
    if "status" not in df_base.columns:
        df_base["status"] = ""
    if len(missing_from_new) > 0:
        df_base.loc[missing_from_new, "ncore_available"] = 0
        df_base.loc[missing_from_new, "hc:mem_req"] = "0G"
        previous = df_base.loc[missing_from_new, "status"].fillna("").astype(str)
        df_base.loc[missing_from_new, "status"] = previous.map(
            lambda value: "|".join(token for token in [value, "missing_in_snapshot"] if token)
        )
    if len(new_since_first) > 0:
        new_rows = df_new.loc[new_since_first].copy()
        if "status" not in new_rows.columns:
            new_rows["status"] = ""
        new_rows["ncore_available"] = 0
        new_rows["hc:mem_req"] = "0G"
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


def print_stats(df):
    for i in df.index:
        queue_name = df.at[i, "queue_name"]
        num_avail_cpu = df.at[i, "ncore_available"]
        avail_ram = df.at[i, "hc:mem_req"]
        ram_unit = df.at[i, "hc:mem_req_unit"]
        node_name = df.at[i, "node_name"]
        node_status = df.at[i, "status"]
        txt = "{}: {:,} cores and {:,.0f}{} RAM in {}"
        if node_status != "":
            txt += " with the status {}"
        print(txt.format(queue_name, num_avail_cpu, avail_ram, ram_unit, node_name, node_status))


def print_resource_availability(df, args):
    queue_names = df.loc[:, "queue_name"].unique()
    queue_names = [q for q in queue_names if not q.startswith("login")]
    resources = dict()
    resources["RAM"] = "hc:mem_req"
    resources["core"] = "ncore_available"
    for resource_name in resources.keys():
        col = resources[resource_name]
        print(f"Reporting top {resource_name} availability:")
        for queue_name in queue_names:
            if args.exclude_abnormal_node:
                df_queue = df.loc[(df["queue_name"] == queue_name) & (df["status"] == ""), :]
            else:
                df_queue = df.loc[(df["queue_name"] == queue_name), :]
            if df_queue.shape[0] == 0:
                continue
            other_cols = [oc for oc in list(resources.values()) if oc != col]
            sort_by = [
                col,
            ] + other_cols
            df_queue = df_queue.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
            if args.all_tiers:
                descending_values = df_queue[col]
                threshold_value = descending_values.iloc[
                    min(args.ntop - 1, descending_values.shape[0] - 1)
                ]
                df_top_availability = df_queue.loc[(df_queue[col] >= threshold_value), :]
                df_top_availability = df_top_availability.sort_values(
                    by=col, ascending=False
                ).reset_index(drop=True)
            else:
                df_top_availability = df_queue.iloc[0 : args.ntop, :]
            print_stats(df=df_top_availability)
        print("")


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


def get_uge_json_job_df(lines):
    payload = "\n".join([str(line).rstrip("\n") for line in lines]).strip()
    if payload == "":
        return _empty_uge_job_df()
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return None
    rows = []
    if not isinstance(data, dict):
        return None
    recognized_schema = bool({"queue_info", "job_info"} & set(data))
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
                    if not isinstance(job, dict):
                        continue
                    job_id = str(job.get("JB_job_number", job.get("job_id", ""))).strip()
                    if job_id == "":
                        continue
                    recognized_schema = True
                    slots = max(_safe_int(job.get("slots", 1), default=1), 0)
                    task_expression = str(
                        job.get("ja_task_id", job.get("ja-task-ID", job.get("tasks", "")))
                    ).strip()
                    num_tasks, expression_estimated = _parse_uge_task_expression(task_expression)
                    queue_name = re.sub(r"@.*$", "", str(job.get("queue_name", "")).strip())
                    state = str(job.get("state", ""))
                    timestamp = str(
                        job.get("JAT_start_time", job.get("JB_submission_time", ""))
                    ).strip()
                    timestamp_items = timestamp.split("T", 1)
                    submit_or_start_date = timestamp_items[0] if timestamp_items else ""
                    submit_or_start_time = timestamp_items[1] if len(timestamp_items) > 1 else ""
                    rows.append(
                        {
                            "job_id": job_id,
                            "prior": job.get("JAT_prio", ""),
                            "name": str(job.get("JB_name", "")),
                            "user": str(job.get("JB_owner", "")),
                            "state": state,
                            "submit_or_start_date": submit_or_start_date,
                            "submit_or_start_time": submit_or_start_time,
                            "queue_name": queue_name,
                            "slots": slots,
                            "ja_task_id": task_expression,
                            "total_slots": slots * num_tasks,
                            # AGE 2023 can omit the range for collapsed pending arrays.
                            "task_count_estimated": expression_estimated
                            or (not task_expression and not queue_name and "q" in state.lower()),
                        }
                    )
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
    return re.split(r"\s+", line.strip(), maxsplit=10), " "


def _parse_squeue_row_items(items, rest_separator):
    items = [str(item).strip() for item in items]
    if len(items) >= 11:
        return {
            "resource_fields_complete": True,
            "job_id": items[0],
            "partition": items[1],
            "name": items[2],
            "user": items[3],
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
        return True
    tokens = re.findall(r"[A-Z_]+", state)
    if len(tokens) == 0:
        return True
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
    hosts = []
    for token in _split_slurm_hostlist(value):
        match = re.fullmatch(r"([^\[]*)\[([^\]]+)\](.*)", token)
        if match is None:
            hosts.append(token)
            continue
        prefix, expression, suffix = match.groups()
        for item in expression.split(","):
            item = item.strip()
            range_match = re.fullmatch(r"([0-9]+)-([0-9]+)", item)
            if range_match is None:
                hosts.append(f"{prefix}{item}{suffix}")
                continue
            start_text, end_text = range_match.groups()
            start = int(start_text)
            end = int(end_text)
            if end < start:
                continue
            width = max(len(start_text), len(end_text))
            hosts.extend(
                "{}{:0{width}d}{}".format(prefix, number, suffix, width=width)
                for number in range(start, end + 1)
            )
    return hosts


def _reservation_user_is_authorized(users_value, current_user):
    if not current_user:
        return False
    users = {
        item.strip()
        for item in str(users_value).split(",")
        if item.strip() not in {"", "(null)", "N/A"}
    }
    return "ALL" in users or current_user in users


def get_scontrol_reservation_df(lines, current_user=""):
    columns = [
        "queue_name",
        "node_name",
        "reservation_name",
        "reserved_cores",
        "reserved_mem_mb",
        "whole_node",
        "accessible",
        "access_users",
    ]
    rows = []
    warnings = []
    for block in _split_scontrol_named_blocks(lines, "ReservationName="):
        header_params = {}
        for line in block:
            if ("=" in line) and (not line.startswith("NodeName=")):
                header_params.update(_parse_key_value_fields(line))
        state = header_params.get("State", "").strip().upper()
        reservation_name = header_params.get("ReservationName", "").strip()
        if state == "":
            warnings.append(
                "reservation {} has no State field and was ignored".format(
                    reservation_name or "<unknown>"
                )
            )
            continue
        if state != "ACTIVE":
            continue
        partition_name = header_params.get("PartitionName", "").strip()
        if partition_name == "":
            warnings.append(
                "active reservation {} has no PartitionName and was ignored".format(
                    reservation_name or "<unknown>"
                )
            )
            continue
        node_count = _safe_int(header_params.get("NodeCnt", ""), default=0)
        default_reserved_cores = max(_safe_int(header_params.get("CoreCnt", ""), default=0), 0)
        reservation_tres = header_params.get("TRES", "")
        if reservation_tres == "":
            reservation_tres = header_params.get("ReqTRES", "")
        if default_reserved_cores <= 0:
            default_reserved_cores = max(
                _safe_int(
                    _extract_tres_resource_value(reservation_tres, "cpu"),
                    default=0,
                ),
                0,
            )
        default_reserved_mem_mb = max(
            _memory_text_to_mb(_extract_tres_resource_value(reservation_tres, "mem")), 0
        )
        access_users = header_params.get("Users", "").strip()
        accessible = _reservation_user_is_authorized(access_users, current_user)
        has_explicit_node_rows = False
        for line in block:
            if not line.startswith("NodeName="):
                continue
            has_explicit_node_rows = True
            params = _parse_key_value_fields(line)
            node_name = params.get("NodeName", "").strip()
            if node_name == "":
                continue
            reserved_cores = _count_core_id_expression(params.get("CoreIDs", ""))
            if (reserved_cores == 0) and (node_count == 1):
                reserved_cores = default_reserved_cores
            if reserved_cores <= 0:
                continue
            reserved_mem_mb = 0
            if (default_reserved_mem_mb > 0) and (node_count > 0):
                reserved_mem_mb = int(round(float(default_reserved_mem_mb) / float(node_count)))
            rows.append(
                {
                    "queue_name": partition_name,
                    "node_name": node_name,
                    "reservation_name": reservation_name,
                    "reserved_cores": reserved_cores,
                    "reserved_mem_mb": reserved_mem_mb,
                    "whole_node": False,
                    "accessible": accessible,
                    "access_users": access_users,
                }
            )
        if has_explicit_node_rows:
            continue
        node_names = _expand_slurm_hostlist(header_params.get("Nodes", "").strip())
        if not node_names:
            warnings.append(
                "active reservation {} has no parseable Nodes field".format(
                    reservation_name or "<unknown>"
                )
            )
            continue
        if node_count <= 0:
            node_count = len(node_names)
        for node_index, node_name in enumerate(node_names):
            reserved_cores = 0
            if default_reserved_cores > 0 and node_count > 0:
                reserved_cores = default_reserved_cores // node_count
                if node_index < (default_reserved_cores % node_count):
                    reserved_cores += 1
            reserved_mem_mb = 0
            if default_reserved_mem_mb > 0 and node_count > 0:
                reserved_mem_mb = int(round(float(default_reserved_mem_mb) / float(node_count)))
            rows.append(
                {
                    "queue_name": partition_name,
                    "node_name": node_name,
                    "reservation_name": reservation_name,
                    "reserved_cores": reserved_cores,
                    "reserved_mem_mb": reserved_mem_mb,
                    "whole_node": default_reserved_cores <= 0,
                    "accessible": accessible,
                    "access_users": access_users,
                }
            )
    frame = pandas.DataFrame(rows, columns=columns)
    frame.attrs["warnings"] = warnings
    return frame


def apply_slurm_reservations(df_node, df_reservation):
    if (
        (df_node is None)
        or (df_node.shape[0] == 0)
        or (df_reservation is None)
        or (df_reservation.shape[0] == 0)
    ):
        return df_node
    df = df_node.copy()
    if "reservation_cores" not in df.columns:
        df["reservation_cores"] = 0
    if "reservation_mem_mb" not in df.columns:
        df["reservation_mem_mb"] = 0
    reservation_rows = df_reservation.copy()
    if "accessible" in reservation_rows.columns:
        reservation_rows = reservation_rows.loc[
            ~reservation_rows["accessible"].fillna(False).astype(bool), :
        ].copy()
    if reservation_rows.shape[0] == 0:
        return df
    if "reserved_mem_mb" not in reservation_rows.columns:
        reservation_rows["reserved_mem_mb"] = 0
    reservation_rows["reserved_mem_mb"] = (
        pandas.to_numeric(reservation_rows["reserved_mem_mb"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    node_shape = df.loc[:, ["queue_name", "node_name", "ncore_total", "hl:mem_total"]].copy()
    node_shape["node_total_mem_mb"] = node_shape["hl:mem_total"].map(_memory_text_to_mb)
    node_shape["ncore_total"] = (
        pandas.to_numeric(node_shape["ncore_total"], errors="coerce").fillna(0).astype(int)
    )
    reservation_rows = reservation_rows.merge(
        node_shape, how="left", on=["queue_name", "node_name"]
    )
    reservation_rows["reserved_cores_effective"] = (
        pandas.to_numeric(reservation_rows["reserved_cores"], errors="coerce").fillna(0).astype(int)
    )
    if "whole_node" in reservation_rows.columns:
        whole_node = reservation_rows["whole_node"].fillna(False).astype(bool)
        reservation_rows.loc[whole_node, "reserved_cores_effective"] = (
            reservation_rows.loc[whole_node, "ncore_total"].fillna(0).astype(int)
        )
    reservation_rows["reserved_mem_mb_effective"] = reservation_rows["reserved_mem_mb"]
    if "whole_node" in reservation_rows.columns:
        reservation_rows.loc[whole_node, "reserved_mem_mb_effective"] = (
            reservation_rows.loc[whole_node, "node_total_mem_mb"].fillna(0).astype(int)
        )
    needs_estimate = (
        (reservation_rows["reserved_mem_mb_effective"] <= 0)
        & (reservation_rows["reserved_cores_effective"] > 0)
        & (reservation_rows["ncore_total"] > 0)
    )
    if needs_estimate.sum():
        reservation_rows.loc[needs_estimate, "reserved_mem_mb_effective"] = (
            (
                (
                    reservation_rows.loc[needs_estimate, "node_total_mem_mb"]
                    * reservation_rows.loc[needs_estimate, "reserved_cores_effective"]
                )
                / reservation_rows.loc[needs_estimate, "ncore_total"]
            )
            .round()
            .astype(int)
        )
    grouped = (
        reservation_rows.groupby(["queue_name", "node_name"], as_index=False)[
            ["reserved_cores_effective", "reserved_mem_mb_effective"]
        ]
        .sum()
        .rename(
            columns={
                "reserved_cores_effective": "reservation_cores",
                "reserved_mem_mb_effective": "reservation_mem_mb",
            }
        )
    )
    df = df.merge(grouped, how="left", on=["queue_name", "node_name"], suffixes=("", "_new"))
    if "reservation_cores_new" in df.columns:
        new_values = (
            pandas.to_numeric(df["reservation_cores_new"], errors="coerce").fillna(0).astype(int)
        )
        df["reservation_cores"] = (
            pandas.to_numeric(df["reservation_cores"], errors="coerce").fillna(0).astype(int)
            + new_values
        )
        df = df.drop(columns=["reservation_cores_new"])
    if "reservation_mem_mb_new" in df.columns:
        new_values = (
            pandas.to_numeric(df["reservation_mem_mb_new"], errors="coerce").fillna(0).astype(int)
        )
        df["reservation_mem_mb"] = (
            pandas.to_numeric(df["reservation_mem_mb"], errors="coerce").fillna(0).astype(int)
            + new_values
        )
        df = df.drop(columns=["reservation_mem_mb_new"])
    df["reservation_cores"] = (
        pandas.to_numeric(df["reservation_cores"], errors="coerce").fillna(0).astype(int)
    )
    df["reservation_mem_mb"] = (
        pandas.to_numeric(df["reservation_mem_mb"], errors="coerce").fillna(0).astype(int)
    )
    node_available_mem_mb = df["hc:mem_req"].map(memory_text_to_mib)
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
    adjusted_available_mem_mb = (node_available_mem_mb - df["reservation_mem_mb"]).clip(lower=0)
    df["hc:mem_req"] = adjusted_available_mem_mb.map(
        lambda value: pandas.NA if pandas.isna(value) else f"{int(value)}M"
    )
    if "hc:mem_req_known" in df.columns:
        df["hc:mem_req_known"] = (
            df["hc:mem_req_known"].fillna(False).astype(bool) & adjusted_available_mem_mb.notna()
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
    known_pairs = set()
    if df_reservation is not None and df_reservation.shape[0] > 0:
        known_pairs = set(
            zip(
                df_reservation["queue_name"].astype(str),
                df_reservation["node_name"].astype(str),
                strict=False,
            )
        )
    unresolved = pandas.Series(False, index=df.index)
    for row_index in df.index[has_reservation_flag]:
        pair = (str(df.at[row_index, "queue_name"]), str(df.at[row_index, "node_name"]))
        if pair not in known_pairs:
            unresolved.at[row_index] = True
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
        if line.startswith("JOBID ") or line.startswith("JOBID\t"):
            continue
        items = re.split(r"\s+", line)
        if len(items) < 8:
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
            partition_status = ""
            if partition_state_map is not None:
                partition_state = partition_state_map.get(partition, "")
                if not _partition_state_is_up(partition_state):
                    partition_status = f"partition_state={partition_state}"
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
                f"jobs  self:R/Q/F={num_running_self}/{num_qwaiting_self}/{num_error_self}  all:R/Q/F={num_running}/{num_qwaiting}/{num_error}"
            )
        else:
            print(f"# of running job tasks (estimated from squeue): {num_running}")
            print(f"# of queued job tasks (estimated from squeue): {num_qwaiting}")
            print(f"# of failed/cancelled job tasks (estimated from squeue): {num_error}")
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
    except Exception:
        pass
    return os.environ.get("USER", "").strip()


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


def _get_launch_value(df_launch, row_index, preferred_col, legacy_col=None):
    if preferred_col in df_launch.columns:
        return df_launch.at[row_index, preferred_col]
    if (legacy_col is not None) and (legacy_col in df_launch.columns):
        return df_launch.at[row_index, legacy_col]
    return None


def print_slurm_launch_heuristic(df_launch, current_user=""):
    if (df_launch is None) or (df_launch.shape[0] == 0):
        return
    subject = "current user"
    if current_user != "":
        subject = current_user
    print(
        f"Reporting heuristic single-node launch ceilings for {subject} (reservation-adjusted, priority-aware):"
    )
    for i in df_launch.index:
        queue_name = df_launch.at[i, "queue_name"]
        recommended_cores = df_launch.at[i, "recommended_cores"]
        recommended_mem_gb = _get_launch_value(
            df_launch, i, "recommended_mem_gb", "recommended_mem_gib"
        )
        top_node_name = df_launch.at[i, "top_node_name"]
        top_node_cores = int(df_launch.at[i, "top_node_cores"])
        top_node_mem_gb = float(
            _get_launch_value(df_launch, i, "top_node_mem_gb", "top_node_mem_gib")
        )
        status = str(df_launch.at[i, "status"])
        print(f"{queue_name}:")
        if pandas.isna(recommended_cores):
            print("  immediate-start ceiling: n/a")
        elif status in ["priority_blocked", "priority_blocked_missing_fields"]:
            print(
                f"  resource-only ceiling: <= {int(recommended_cores):,} CPUs and {floor_gib(recommended_mem_gb)}GiB RAM"
            )
        else:
            print(
                f"  immediate-start ceiling: <= {int(recommended_cores):,} CPUs and {floor_gib(recommended_mem_gb)}GiB RAM"
            )
        if top_node_name != "":
            print(
                f"  top free node: {top_node_name} has {top_node_cores:,} CPUs and {floor_gib(top_node_mem_gb)}GiB RAM"
            )
        blocked_req_cores = df_launch.at[i, "blocked_req_cores"]
        if pandas.notna(blocked_req_cores):
            blocked_req_mem_gb = float(
                _get_launch_value(df_launch, i, "blocked_req_mem_gb", "blocked_req_mem_gib")
            )
            blocked_time_limit = str(df_launch.at[i, "blocked_time_limit"]).strip()
            blocked_txt = f"smallest current Priority-blocked request is {int(blocked_req_cores)} CPUs / {floor_gib(blocked_req_mem_gb)}GiB"
            if blocked_time_limit not in ["", "nan"]:
                blocked_txt += f" / {blocked_time_limit}"
            print(f"  {blocked_txt}")
        priority_gap = df_launch.at[i, "priority_gap"]
        if pandas.notna(priority_gap):
            print(f"  priority gap: {int(priority_gap)}")
        fairshare_gap = df_launch.at[i, "fairshare_gap"]
        if pandas.notna(fairshare_gap):
            print(f"  fairshare gap: {int(fairshare_gap)}")
        if status == "priority_blocked":
            print(
                "  note: current user has Priority-blocked jobs; no stable immediate-start ceiling can be inferred"
            )
        if status == "priority_blocked_missing_fields":
            print(
                "  note: current user has Priority-blocked jobs, but request size is unavailable in the current squeue format"
            )
    print("")


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
        resource_fields = [f"<={int(recommended_cores)}c/{memory_text}"]
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
            df_top_cpu = df_normal.sort_values(
                by=["ncore_available", "hc:mem_req", "node_name"], ascending=[False, False, True]
            ).reset_index(drop=True)
            df_top_ram = df_normal.sort_values(
                by=["hc:mem_req", "ncore_available", "node_name"], ascending=[False, False, True]
            ).reset_index(drop=True)
            top_cpu = _format_slurm_compact_node(
                df_top_cpu.at[0, "node_name"],
                df_top_cpu.at[0, "ncore_available"],
                df_top_cpu.at[0, "hc:mem_req"],
            )
            top_ram = _format_slurm_compact_node(
                df_top_ram.at[0, "node_name"],
                df_top_ram.at[0, "ncore_available"],
                df_top_ram.at[0, "hc:mem_req"],
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
    print("legend: nodes=working/abnormal/total, cpu=available/used/total, ram=available/total")
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
    if (df_qfree is not None) and (df_qfree.shape[0] > 0):
        queue_names = df_qfree["queue_name"].dropna().astype(str).tolist()
        for i in df_qfree.index:
            qfree_rows[str(df_qfree.at[i, "queue_name"])] = df_qfree.loc[i, :].to_dict()
    else:
        queue_names = df["queue_name"].dropna().astype(str).unique().tolist()
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
                df_top_cpu = df_normal.sort_values(
                    by=["ncore_available", "hc:mem_req", "node_name"],
                    ascending=[False, False, True],
                ).reset_index(drop=True)
                df_top_ram = df_normal.sort_values(
                    by=["hc:mem_req", "ncore_available", "node_name"],
                    ascending=[False, False, True],
                ).reset_index(drop=True)
                top_cpu = _format_slurm_compact_node(
                    df_top_cpu.at[0, "node_name"],
                    df_top_cpu.at[0, "ncore_available"],
                    df_top_cpu.at[0, "hc:mem_req"],
                )
                top_ram = _format_slurm_compact_node(
                    df_top_ram.at[0, "node_name"],
                    df_top_ram.at[0, "ncore_available"],
                    df_top_ram.at[0, "hc:mem_req"],
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


def get_df(args):
    scheduler = get_scheduler_from_command(args.stat_command)
    if scheduler is None:
        raise KFBatchUsageError(f"Exiting. --stat_command does not support: {args.stat_command}")
    timeout_seconds = _command_timeout_from_args(args)
    if scheduler == "slurm":
        current_user = get_current_user_name()
        squeue_command = get_squeue_command_for_parsing(args.stat_command)
        lines = get_command_stdout_lines(
            command_str=squeue_command,
            example_file=args.example_file,
            allow_failure=False,
            command_name="--stat_command",
            timeout_seconds=timeout_seconds,
        )
        df_user = get_squeue_user_df(lines)
        print_queued_job_summary(df_user, scheduler="slurm", current_user=current_user)
        partition_lines = get_command_stdout_lines(
            command_str=args.slurm_partition_command,
            example_file=args.slurm_partition_example_file,
            allow_failure=True,
            command_name="--slurm_partition_command",
            quiet_failure=True,
            timeout_seconds=timeout_seconds,
        )
        partition_state_map = None
        if partition_lines is not None:
            df_partition = get_scontrol_partition_df(partition_lines)
            if df_partition.shape[0] > 0:
                partition_state_map = df_partition.set_index("partition_name")[
                    "partition_state"
                ].to_dict()
            else:
                _print_degraded(
                    "Slurm partition",
                    "command succeeded but no partition rows were parsed",
                )
        else:
            _print_degraded(
                "Slurm partition",
                "--slurm_partition_command failed or timed out",
            )
        node_lines = get_command_stdout_lines(
            command_str=args.slurm_node_command,
            example_file=args.slurm_node_example_file,
            allow_failure=True,
            command_name="--slurm_node_command",
            timeout_seconds=timeout_seconds,
        )
        if node_lines is None:
            print("Skipping node resource summary because --slurm_node_command failed.")
            print("")
            return scheduler, None, df_user
        df_slurm_node = get_scontrol_node_df(node_lines, partition_state_map=partition_state_map)
        if df_slurm_node.shape[0] == 0:
            print("Skipping node resource summary because SLURM node output could not be parsed.")
            print(
                'Use --slurm_node_command "scontrol show node -o" or provide --slurm_node_example_file.'
            )
            print("")
            return scheduler, None, df_user
        return scheduler, df_slurm_node, df_user
    if args.niter < 1:
        raise KFBatchUsageError("Exiting. --niter must be >= 1 when using qstat mode.")
    df_user = None
    has_all_user_jobs = False
    for i in range(args.niter):
        lines = get_command_stdout_lines(
            command_str=args.stat_command,
            example_file=args.example_file,
            allow_failure=False,
            command_name="--stat_command",
            timeout_seconds=timeout_seconds,
        )
        df_i = get_qstat_df(lines)
        if df_i.shape[0] == 0:
            raise KFBatchCommandError(
                f"AGE/UGE/SGE resource snapshot {i + 1} contained no parseable queue instances."
            )
        if i == 0:
            df = df_i
            df_user = get_user_df(lines)
            uge_job_command = getattr(args, "uge_job_command", "")
            uge_job_example_file = getattr(args, "uge_job_example_file", "")
            if (uge_job_command != "") or (uge_job_example_file != ""):
                job_lines = get_command_stdout_lines(
                    command_str=uge_job_command,
                    example_file=uge_job_example_file,
                    allow_failure=True,
                    command_name="--uge_job_command",
                    quiet_failure=True,
                    timeout_seconds=timeout_seconds,
                )
                if job_lines is not None:
                    payload = "\n".join(job_lines).lstrip()
                    if payload.startswith("{") or payload.startswith("["):
                        df_user_json = get_uge_json_job_df(job_lines)
                        if df_user_json is not None:
                            df_user = df_user_json
                            has_all_user_jobs = True
                        else:
                            _print_degraded(
                                "AGE/UGE/SGE all-user jobs",
                                "JSON schema was not recognized; using jobs embedded in qstat -F",
                            )
                    else:
                        df_user = get_user_df(job_lines)
                        has_all_user_jobs = True
                else:
                    _print_degraded(
                        "AGE/UGE/SGE all-user jobs",
                        "--uge_job_command failed or timed out; using jobs embedded in qstat -F",
                    )
            print_queued_job_summary(
                df_user,
                scheduler="uge",
                current_user=get_current_user_name(),
                all_users=has_all_user_jobs,
            )
        else:
            df = _merge_qstat_iteration_min_availability(df, df_i)
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


def print_cluster_summary(df):
    queue_names = df["queue_name"].unique()
    print(
        "Reporting working/abnormal/total nodes, available/used/reserved/abnormal/total CPUs, and available/total RAM:"
    )
    for queue_name in queue_names:
        df_queue = df.loc[(df["queue_name"] == queue_name), :].reset_index(drop=True)
        is_abnormal_status = df_queue["status"] != ""
        num_abnormal_node = is_abnormal_status.sum()
        num_node = df_queue.shape[0]
        num_working_node = num_node - num_abnormal_node
        ncore_total = df_queue.loc[:, "ncore_total"].sum()
        ncore_used = df_queue.loc[~is_abnormal_status, "ncore_used"].sum()
        ncore_reserved = df_queue.loc[~is_abnormal_status, "ncore_resv"].sum()
        ncore_abnormal = df_queue.loc[is_abnormal_status, "ncore_total"].sum()
        ncore_available = df_queue.loc[~is_abnormal_status, "ncore_available"].sum()
        mem_total = df_queue.loc[:, "hl:mem_total"].sum(min_count=1)
        mem_available = df_queue.loc[~is_abnormal_status, "hc:mem_req"].sum(min_count=1)
        mem_available_text = "?" if pandas.isna(mem_available) else floor_gib(mem_available)
        mem_total_text = "?" if pandas.isna(mem_total) else floor_gib(mem_total)
        txt = "{}: {}/{}/{} nodes, {}/{}/{}/{}/{} CPUs, and {}/{}GiB RAM"
        print(
            txt.format(
                queue_name,
                num_working_node,
                num_abnormal_node,
                num_node,
                ncore_available,
                ncore_used,
                ncore_reserved,
                ncore_abnormal,
                ncore_total,
                mem_available_text,
                mem_total_text,
            )
        )
    print("")


def stat_main(args):
    legacy_out = getattr(args, "out", "")
    explicit_node_out = getattr(args, "out_nodes", "")
    if legacy_out and explicit_node_out and legacy_out != explicit_node_out:
        raise KFBatchUsageError(
            "--out and --out_nodes refer to the same node table; specify only one path."
        )
    node_output_path = explicit_node_out or legacy_out
    job_output_path = getattr(args, "out_jobs", "")
    timeout_seconds = _command_timeout_from_args(args)
    scheduler, df, df_user = get_df(args)
    if job_output_path:
        df_user.to_csv(job_output_path, sep="\t", index=False)
    if (scheduler == "slurm") and (df is None):
        print("Skipping cluster/node resource availability.")
        print("Reason: no parsed SLURM node data was available.")
        print(
            'Provide --slurm_node_command or --slurm_node_example_file from "scontrol show node -o".'
        )
        if node_output_path:
            _print_degraded(
                "node TSV",
                f"{node_output_path} was not written because no node table was available",
            )
        return
    if scheduler == "slurm":
        reservation_lines = get_command_stdout_lines(
            command_str=args.slurm_reservation_command,
            example_file=args.slurm_reservation_example_file,
            allow_failure=True,
            command_name="--slurm_reservation_command",
            quiet_failure=True,
            timeout_seconds=timeout_seconds,
        )
        df_reservation = None
        if reservation_lines is not None:
            df_reservation = get_scontrol_reservation_df(
                reservation_lines,
                current_user=get_current_user_name(),
            )
            for warning in df_reservation.attrs.get("warnings", []):
                _print_degraded("Slurm reservation", warning)
            if df_reservation.shape[0] > 0:
                df = apply_slurm_reservations(df, df_reservation)
        else:
            _print_degraded(
                "Slurm reservation",
                "--slurm_reservation_command failed or timed out; reserved nodes are treated as unavailable",
            )
        df = mark_unresolved_slurm_reservations(df, df_reservation)
    df = adjust_ram_unit(df)
    if scheduler == "slurm" and args.show_launch_heuristic:
        prio_lines = get_command_stdout_lines(
            command_str=args.slurm_prio_command,
            example_file=args.slurm_prio_example_file,
            allow_failure=True,
            command_name="--slurm_prio_command",
            quiet_failure=True,
            timeout_seconds=timeout_seconds,
        )
        df_prio = None
        if prio_lines is not None:
            df_prio = get_sprio_df(prio_lines)
            if df_prio.shape[0] == 0:
                _print_degraded(
                    "Slurm priority",
                    "command succeeded but no priority rows were parsed",
                )
        else:
            _print_degraded(
                "Slurm priority",
                "--slurm_prio_command failed or timed out; launch estimates are resource-only",
            )
        current_user = get_current_user_name()
        df_launch = get_slurm_launch_heuristic_df(
            df_node=df, df_job=df_user, df_prio=df_prio, current_user=current_user
        )
        print_slurm_compact_summary(df, df_launch, args)
    elif scheduler == "uge":
        qfree_lines = get_command_stdout_lines(
            command_str=getattr(args, "uge_qfree_command", ""),
            example_file=getattr(args, "uge_qfree_example_file", ""),
            allow_failure=True,
            command_name="--uge_qfree_command",
            quiet_failure=True,
            timeout_seconds=timeout_seconds,
        )
        df_qfree = None
        if qfree_lines is not None:
            df_qfree = get_qfree_df(qfree_lines)
            if df_qfree.shape[0] == 0:
                _print_degraded(
                    "qfree",
                    "command succeeded but no queue summaries were parsed",
                )
        elif getattr(args, "uge_qfree_command", ""):
            _print_degraded(
                "qfree",
                "--uge_qfree_command failed or timed out; quota columns are unavailable",
            )
        print_uge_compact_summary(df, df_qfree, args)
    else:
        print_cluster_summary(df)
        print_resource_availability(df, args)
    if node_output_path:
        df.to_csv(node_output_path, sep="\t", index=False)
