"""Uge parser helpers, independent of command execution."""

import json
import re
from typing import Any

import pandas

from kfbatch.memory import (
    grid_engine_memory_series_to_gib,
)
from kfbatch.parse_quality import attach_parse_quality
from kfbatch.parse_utils import _numeric_task_token_count, _safe_int

QSTAT_COLUMNS = [
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

UGE_JOB_COLUMNS = [
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
        token_count = _numeric_task_token_count(token)
        if token_count is None:
            estimated = True
        else:
            num_tasks += token_count
    if num_tasks == 0:
        return 1, True
    return num_tasks, estimated


def get_qstat_df(lines):
    """Parse queue-instance capacity without retaining unused ``qstat -F`` fields.

    AGE can emit hundreds of host/resource attributes for every queue instance.
    Only the two memory attributes below participate in kfbatch calculations or
    its documented node-table schema, so the parser discards all other dynamic
    fields as it streams the input.
    """

    rows: list[Any] = []
    node = None

    def append_node():
        if node is None:
            return
        mem_request = node.get("hc:mem_req", "").strip()
        mem_total = node.get("hl:mem_total", "").strip()
        mem_request_known = mem_request != ""
        mem_total_known = mem_total != ""
        rows.append(
            (
                node["queue_name"],
                node["node_name"],
                node["qtype"],
                node["ncore_resv"],
                node["ncore_used"],
                node["ncore_total"],
                node["np_load"],
                node["arch"],
                node["status"],
                mem_request if mem_request_known else pandas.NA,
                mem_total if mem_total_known else pandas.NA,
                mem_request_known,
                mem_total_known,
                max(
                    node["ncore_total"] - node["ncore_used"] - node["ncore_resv"],
                    0,
                ),
            )
        )

    for raw_line in lines:
        line = str(raw_line).rstrip("\r\n")
        if line == "":
            continue
        if line.startswith("\t"):
            if node is None:
                continue
            key, separator, value = line[1:].partition("=")
            if separator and key in {"hc:mem_req", "hl:mem_total"}:
                node[key] = value
            continue
        append_node()
        node = None
        if line.startswith(("queuename", "---", "###", " ")):
            continue
        items = line.split()
        if len(items) < 5:
            continue
        core_counts = items[2].split("/")
        if len(core_counts) != 3 or not all(value.isdigit() for value in core_counts):
            continue
        queue_name, separator, node_name = items[0].partition("@")
        if separator == "":
            continue
        node = {
            "queue_name": queue_name,
            "node_name": node_name,
            "qtype": items[1],
            "ncore_resv": int(core_counts[0]),
            "ncore_used": int(core_counts[1]),
            "ncore_total": int(core_counts[2]),
            "np_load": items[3],
            "arch": items[4],
            "status": items[5] if len(items) > 5 else "",
        }
    append_node()
    if not rows:
        return pandas.DataFrame(columns=QSTAT_COLUMNS)
    df = pandas.DataFrame.from_records(rows, columns=QSTAT_COLUMNS)
    df = df.sort_values(by=["queue_name", "node_name"]).reset_index(drop=True)
    return df


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
        base_mem = grid_engine_memory_series_to_gib(df_base.loc[common_index, mem_col])
        new_mem = grid_engine_memory_series_to_gib(df_new.loc[common_index, mem_col])
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
        selected = df_base.loc[common_index, mem_col].copy()
        use_new = new_mem < base_mem
        selected.loc[use_new] = df_new.loc[common_index, mem_col].loc[use_new]
        df_base.loc[common_index, mem_col] = selected.where(known, pandas.NA)
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
    return pandas.DataFrame(columns=UGE_JOB_COLUMNS)


def _parse_uge_text_job_line(line, text_cache):
    items = str(line).split()
    if len(items) < 8 or not items[0].isdigit():
        return None
    tail_index = 7
    queue_name = ""
    if tail_index < len(items) and (("@" in items[tail_index]) or items[tail_index].endswith(".q")):
        queue_name = items[tail_index].split("@", 1)[0]
        tail_index += 1
    if tail_index < len(items) and not items[tail_index].isdigit():
        # AGE may print a non-empty job-class column between queue and slots.
        tail_index += 1
    if tail_index >= len(items) or not items[tail_index].isdigit():
        return None
    slots = int(items[tail_index])
    tail_index += 1
    ja_task_id = items[tail_index] if tail_index < len(items) else ""
    num_tasks, task_count_estimated = _parse_uge_task_expression(ja_task_id)
    reuse = text_cache.setdefault
    return (
        items[0],
        items[1],
        items[2],
        reuse(items[3], items[3]),
        reuse(items[4], items[4]),
        items[5],
        items[6],
        reuse(queue_name, queue_name),
        slots,
        ja_task_id,
        slots * num_tasks,
        task_count_estimated,
    )


def get_user_df(lines):
    rows = []
    text_cache: dict[str, str] = {}
    input_nonempty = False
    recognized_header = False
    candidate_rows = 0
    for line in lines:
        text = str(line)
        stripped = text.strip()
        if stripped:
            input_nonempty = True
        if stripped.lower().startswith("job-id"):
            recognized_header = True
            continue
        if not stripped or set(stripped) <= {"-"}:
            continue
        if not stripped[:1].isdigit():
            continue
        candidate_rows += 1
        row = _parse_uge_text_job_line(text, text_cache)
        if row is not None:
            rows.append(row)
    if not rows:
        frame = _empty_uge_job_df()
    else:
        frame = pandas.DataFrame.from_records(rows, columns=UGE_JOB_COLUMNS)
    frame.attrs.update(
        {
            "input_nonempty": input_nonempty,
            "recognized_header": recognized_header,
            "candidate_rows": candidate_rows,
            "recognized_rows": len(rows),
        }
    )
    return attach_parse_quality(frame, candidate_rows=candidate_rows)


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


def _uge_json_job_row(job, text_cache):
    job_id = str(job.get("JB_job_number", job.get("job_id", ""))).strip()
    if job_id == "":
        return None
    slots = max(_safe_int(job.get("slots", 1), default=1), 0)
    task_expression = str(
        job.get("ja_task_id", job.get("ja-task-ID", job.get("tasks", "")))
    ).strip()
    num_tasks, expression_estimated = _parse_uge_task_expression(task_expression)
    queue_name = str(job.get("queue_name", "")).strip().partition("@")[0]
    state = str(job.get("state", ""))
    priority = job.get("JAT_prio", "")
    timestamp = str(job.get("JAT_start_time", job.get("JB_submission_time", ""))).strip()
    timestamp_items = timestamp.split("T", 1)
    reuse = text_cache.setdefault
    return (
        job_id,
        priority,
        str(job.get("JB_name", "")),
        reuse(str(job.get("JB_owner", "")), str(job.get("JB_owner", ""))),
        reuse(state, state),
        timestamp_items[0] if timestamp_items else "",
        timestamp_items[1] if len(timestamp_items) > 1 else "",
        reuse(queue_name, queue_name),
        slots,
        task_expression,
        slots * num_tasks,
        # AGE 2023 can omit the range for collapsed pending arrays.
        expression_estimated or (not task_expression and not queue_name and "q" in state.lower()),
    )


def get_uge_json_job_df(lines):
    payload = "\n".join(str(line).rstrip("\n") for line in lines).strip()
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
    text_cache: dict[str, str] = {}
    candidate_rows = 0
    for job in _iter_uge_json_jobs(data):
        candidate_rows += 1
        row = _uge_json_job_row(job, text_cache)
        if row is not None:
            recognized_schema = True
            rows.append(row)
    if not recognized_schema:
        return None
    if len(rows) == 0:
        empty = _empty_uge_job_df()
        empty.attrs["recognized_schema"] = True
        return attach_parse_quality(empty, candidate_rows=candidate_rows)
    frame = pandas.DataFrame.from_records(rows, columns=UGE_JOB_COLUMNS)
    frame.attrs["recognized_schema"] = True
    return attach_parse_quality(frame, candidate_rows=candidate_rows)


def _optional_int(value):
    txt = str(value).strip().replace(",", "")
    if txt in ["", "-"]:
        return None
    try:
        return int(txt)
    except ValueError:
        return None


def _qfree_group_context(lines):
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    group_names = []
    group_users: list[str] = []
    in_group_table = False
    for raw_line in lines:
        line = ansi_escape.sub("", str(raw_line)).strip()
        match = re.match(
            r"THE NUMBER OF (?:RUNNING JOBS|MEM_REQ) BY USER IN THE GROUP \(([^)]+)\)",
            line,
        )
        if match is not None:
            group_names.append(match.group(1).strip())
            in_group_table = True
            continue
        if line.startswith(("SUMMARY OF ", "======================")):
            in_group_table = False
            continue
        items = re.split(r"\s+", line)
        if in_group_table and items and items[0].upper() == "QNAME":
            group_users.extend(
                user
                for user in items[1:]
                if re.fullmatch(r"[A-Za-z0-9_.-]+", user) and user not in group_users
            )
    unique_names = set(group_names)
    return (group_names[0] if len(unique_names) == 1 else ""), group_users


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
    lines = list(lines)
    group_name, group_users = _qfree_group_context(lines)
    rows_by_queue: dict[str, dict[str, Any]] = {}
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
    frame = pandas.DataFrame(rows, columns=columns)
    frame.attrs["group_name"] = group_name
    frame.attrs["group_users"] = group_users
    return frame
