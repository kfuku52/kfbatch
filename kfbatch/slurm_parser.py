"""Slurm parser helpers, independent of command execution."""

import re

import pandas

from kfbatch.memory import (
    memory_text_to_mib,
)
from kfbatch.parse_quality import attach_parse_quality
from kfbatch.parse_utils import _numeric_task_token_count, _safe_int

SLURM_NORMAL_NODE_STATES = {"IDLE", "MIXED", "ALLOCATED"}

SLURM_CONDITIONALLY_SAFE_NODE_FLAGS = SLURM_NORMAL_NODE_STATES | {"RESERVED"}

SLURM_UNAVAILABLE_NODE_FLAGS = {
    "BLOCKED",
    "CLOUD",
    "COMPLETING",
    "DRAIN",
    "DRAINED",
    "DRAINING",
    "DOWN",
    "DYNAMIC",
    "FAIL",
    "FAILING",
    "FUTURE",
    "INVALID_REG",
    "NOT_RESPONDING",
    "MAINT",
    "MAINTENANCE",
    "PERFCTRS",
    "POWER_DOWN",
    "POWERING_DOWN",
    "POWERING_UP",
    "POWERED_DOWN",
    "REBOOT_REQUESTED",
    "REBOOT_ISSUED",
    "PLANNED",
}

SLURM_NODE_SUFFIX_FLAGS = {
    "*": "NOT_RESPONDING",
    "~": "POWERED_DOWN",
    "#": "POWERING_UP",
    "!": "POWER_DOWN",
    "%": "POWERING_DOWN",
    "$": "MAINTENANCE",
    "@": "REBOOT_REQUESTED",
    "^": "REBOOT_ISSUED",
    "-": "PLANNED",
}

SLURM_JOB_COLUMNS = [
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

SLURM_NODE_COLUMNS = [
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


def _extract_slurm_pending_reason(node_or_reason):
    txt = str(node_or_reason).strip()
    if not (txt.startswith("(") and txt.endswith(")")):
        return ""
    return txt[1:-1].strip()


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
        token_count = _numeric_task_token_count(token)
        if token_count is None:
            has_ambiguous_pattern = True
        else:
            num_tasks += token_count
    if num_tasks == 0:
        return 1, True
    return num_tasks, has_ambiguous_pattern


def estimate_slurm_task_count(job_id):
    if "_" not in job_id:
        return 1, False
    job_suffix = job_id.split("_", 1)[1]
    if job_suffix.isdigit():
        return 1, False
    if not job_suffix.startswith("["):
        return 1, True
    task_expression = job_suffix[1:]
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
    text = str(value or "").strip()
    return text != "" and text.replace("_", "").isalpha()


def _parse_squeue_row_items(items, rest_separator, text_cache):
    items = [item.strip() for item in items]
    reuse = text_cache.setdefault
    has_account = len(items) >= 12 and _looks_like_slurm_state_token(items[5])
    if has_account:
        node_or_reason = rest_separator.join(items[11:]).strip()
        return (
            items[0],
            reuse(items[1], items[1]),
            items[2],
            reuse(items[3], items[3]),
            reuse(items[4], items[4]),
            reuse(items[5], items[5]),
            items[6],
            items[7],
            items[8],
            reuse(items[9], items[9]),
            reuse(items[10], items[10]),
            reuse(node_or_reason, node_or_reason),
            True,
        )
    if len(items) >= 11:
        node_or_reason = rest_separator.join(items[10:]).strip()
        return (
            items[0],
            reuse(items[1], items[1]),
            items[2],
            reuse(items[3], items[3]),
            "",
            reuse(items[4], items[4]),
            items[5],
            items[6],
            items[7],
            reuse(items[8], items[8]),
            reuse(items[9], items[9]),
            reuse(node_or_reason, node_or_reason),
            True,
        )
    if len(items) >= 8:
        node_or_reason = rest_separator.join(items[7:]).strip()
        return (
            items[0],
            reuse(items[1], items[1]),
            items[2],
            reuse(items[3], items[3]),
            "",
            reuse(items[4], items[4]),
            items[5],
            items[6],
            "",
            "",
            "",
            reuse(node_or_reason, node_or_reason),
            False,
        )
    return None


def get_squeue_user_df(lines):
    table = []
    text_cache: dict[str, str] = {}
    input_nonempty = False
    recognized_header = False
    candidate_rows = 0
    rejected_rows = 0
    for raw_line in lines:
        line = str(raw_line).rstrip("\r\n")
        if line.strip() == "":
            continue
        input_nonempty = True
        if re.match(r"^\s*JOBID(?:\s|\\t|$)", line):
            recognized_header = True
            continue
        candidate_rows += 1
        items, rest_separator = _split_squeue_row(line)
        row = _parse_squeue_row_items(items, rest_separator, text_cache)
        if row is None:
            rejected_rows += 1
            continue
        (
            job_id,
            partition,
            name,
            user,
            account,
            state,
            elapsed_time,
            num_nodes_txt,
            req_cpus_txt,
            req_mem,
            time_limit,
            node_or_reason,
            resource_fields_complete,
        ) = row
        if (
            not str(job_id).strip()
            or not _looks_like_slurm_state_token(state)
            or not str(num_nodes_txt).isdigit()
            or int(num_nodes_txt) < 1
        ):
            rejected_rows += 1
            continue
        num_nodes = int(num_nodes_txt)
        if resource_fields_complete:
            if (
                not str(req_cpus_txt).isdigit()
                or int(req_cpus_txt) < 1
                or pandas.isna(memory_text_to_mib(req_mem, default_unit="M"))
            ):
                rejected_rows += 1
                continue
            req_cpus = int(req_cpus_txt)
        else:
            req_cpus = 0
        num_tasks, is_estimated = estimate_slurm_task_count(job_id)
        total_slots = num_tasks
        table.append(
            (
                job_id,
                partition,
                name,
                user,
                account,
                state,
                elapsed_time,
                num_nodes,
                req_cpus,
                req_mem,
                time_limit,
                node_or_reason,
                _extract_slurm_pending_reason(node_or_reason),
                resource_fields_complete,
                total_slots,
                is_estimated,
            )
        )
    frame = pandas.DataFrame.from_records(table, columns=SLURM_JOB_COLUMNS)
    frame.attrs.update(
        {
            "input_nonempty": input_nonempty,
            "recognized_header": recognized_header,
            "candidate_rows": candidate_rows,
            "recognized_rows": len(table),
            "rejected_rows": rejected_rows,
        }
    )
    missing = ("account",) if not frame.empty and frame["account"].eq("").any() else ()
    return attach_parse_quality(frame, candidate_rows=candidate_rows, missing_fields=missing)


def _iter_scontrol_node_blocks(lines):
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            if current:
                yield " ".join(current)
                current = []
            continue
        if ("NodeName=" in line) and current:
            yield " ".join(current)
            current = [line]
            continue
        current.append(line)
    if current:
        yield " ".join(current)


def _parse_key_value_fields(line):
    params = {}
    for item in line.split():
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        params[key] = value
    return params


def _strict_nonnegative_int(value):
    txt = str(value).strip()
    if not txt.isdigit():
        return None
    number = int(txt)
    return number if number >= 0 else None


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
        for suffix in token[m.end() :]:
            flag = SLURM_NODE_SUFFIX_FLAGS.get(suffix)
            if flag is not None:
                flags.append(flag)
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
    return df.reset_index(drop=True)


def _slurm_node_capacity(params):
    metadata_status = []
    ncore_total = _strict_nonnegative_int(params.get("CPUEfctv", ""))
    if ncore_total is None:
        ncore_total = _strict_nonnegative_int(params.get("CPUTot", ""))
    if ncore_total is None:
        metadata_status.append("cpu_total=UNKNOWN")
        ncore_total = 0
    ncore_used = _strict_nonnegative_int(params.get("CPUAlloc", ""))
    if ncore_used is None or ncore_used > ncore_total:
        metadata_status.append("cpu_alloc=UNKNOWN")
        ncore_used = ncore_total
    ncore_available = max(ncore_total - ncore_used, 0)

    mem_total_mb = _strict_nonnegative_int(params.get("RealMemory", ""))
    mem_total_known = mem_total_mb is not None
    alloc_mem_mb = _strict_nonnegative_int(params.get("AllocMem", ""))
    mem_available_known = (
        mem_total_known and alloc_mem_mb is not None and alloc_mem_mb <= mem_total_mb
    )
    if not mem_total_known:
        metadata_status.append("memory_total=UNKNOWN")
    if not mem_available_known:
        metadata_status.append("memory_alloc=UNKNOWN")
        mem_available_mb = None
    else:
        # Slurm's allocated-memory accounting defines schedulable memory.
        # FreeMem is an OS page statistic and is intentionally not a fallback.
        mem_available_mb = max(mem_total_mb - alloc_mem_mb, 0)
    return {
        "metadata_status": metadata_status,
        "ncore_total": ncore_total,
        "ncore_used": ncore_used,
        "ncore_available": ncore_available,
        "mem_total_mb": mem_total_mb,
        "mem_total_known": mem_total_known,
        "mem_available_mb": mem_available_mb,
        "mem_available_known": mem_available_known,
    }


def _slurm_node_status(slurm_state, metadata_status):
    state_base = _normalize_slurm_node_state(slurm_state)
    flags = _slurm_state_flags(slurm_state)
    unknown_flags = [
        flag
        for flag in flags
        if flag not in SLURM_CONDITIONALLY_SAFE_NODE_FLAGS
        and flag not in SLURM_UNAVAILABLE_NODE_FLAGS
    ]
    has_unavailable_flag = any(flag in SLURM_UNAVAILABLE_NODE_FLAGS for flag in flags)
    if not slurm_state:
        node_status = "node_state=UNKNOWN"
    elif state_base in SLURM_NORMAL_NODE_STATES and not has_unavailable_flag and not unknown_flags:
        node_status = ""
    else:
        node_status = slurm_state
    if metadata_status:
        metadata_text = "|".join(metadata_status)
        node_status = "|".join(token for token in [node_status, metadata_text] if token)
    return node_status


def get_scontrol_node_df(lines, partition_state_map=None):
    rows = []
    for node_block in _iter_scontrol_node_blocks(lines):
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
        capacity = _slurm_node_capacity(params)
        ncore_total = capacity["ncore_total"]
        ncore_used = capacity["ncore_used"]
        ncore_resv = 0
        ncore_available = capacity["ncore_available"]
        mem_total_mb = capacity["mem_total_mb"]
        mem_total_known = capacity["mem_total_known"]
        mem_available_mb = capacity["mem_available_mb"]
        mem_available_known = capacity["mem_available_known"]
        slurm_state = params.get("State", "")
        reservation_name = params.get("ReservationName", "").strip()
        node_status = _slurm_node_status(slurm_state, capacity["metadata_status"])
        arch = params.get("Arch", "")
        mem_total = f"{mem_total_mb}M" if mem_total_known else pandas.NA
        mem_available = f"{mem_available_mb}M" if mem_available_known else pandas.NA
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
            row_ncore_available = ncore_available if status == "" else 0
            row_mem_available = (
                mem_available if status == "" else ("0M" if mem_available_known else pandas.NA)
            )
            row_mem_available_known = mem_available_known
            rows.append(
                (
                    partition,
                    node_name,
                    "SLURM",
                    ncore_resv,
                    ncore_used,
                    ncore_total,
                    row_ncore_available,
                    "",
                    arch,
                    status,
                    mem_total,
                    row_mem_available,
                    mem_total_known,
                    row_mem_available_known,
                    slurm_state,
                    reservation_name,
                )
            )
    df = pandas.DataFrame.from_records(rows, columns=SLURM_NODE_COLUMNS)
    if df.shape[0] == 0:
        return df
    df = df.sort_values(by=["queue_name", "node_name"]).reset_index(drop=True)
    return df
