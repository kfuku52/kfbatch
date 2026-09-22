import os
import pathlib
import re
import shlex
import tempfile
import time
from typing import Any

import pandas

from kfbatch.command import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    get_command_stdout_lines,
)
from kfbatch.errors import KFBatchCommandError, KFBatchUsageError
from kfbatch.job_states import (
    SLURM_ERROR_STATES,
    SLURM_PENDING_STATES,
    SLURM_RUNNING_STATES,
    SLURM_STATE_NAME_TO_CODE as SLURM_STATE_NAME_TO_CODE,
    _normalize_slurm_job_state,
    _normalize_uge_job_state,
)
from kfbatch.memory import (
    grid_engine_memory_series_to_gib,
    memory_series_to_gib,
    memory_text_to_gib,
    memory_text_to_mib,
    slurm_request_memory_gib,
)
from kfbatch.parse_quality import rejected_rows
from kfbatch.parse_utils import (
    _numeric_task_token_count as _numeric_task_token_count,
    _safe_int as _safe_int,
)
from kfbatch.render import (
    _format_compact_top_nodes as _format_compact_top_nodes,
    _format_qfree_int as _format_qfree_int,
    _format_slurm_compact_launch_row as _format_slurm_compact_launch_row,
    _format_slurm_compact_node as _format_slurm_compact_node,
    _format_slurm_compact_time_limit as _format_slurm_compact_time_limit,
    _slurm_time_to_minutes as _slurm_time_to_minutes,
    print_slurm_compact_summary as print_slurm_compact_summary,
    print_uge_compact_summary as print_uge_compact_summary,
)
from kfbatch.slurm_parser import (
    SLURM_CONDITIONALLY_SAFE_NODE_FLAGS as SLURM_CONDITIONALLY_SAFE_NODE_FLAGS,
    SLURM_JOB_COLUMNS as SLURM_JOB_COLUMNS,
    SLURM_NODE_COLUMNS as SLURM_NODE_COLUMNS,
    SLURM_NODE_SUFFIX_FLAGS as SLURM_NODE_SUFFIX_FLAGS,
    SLURM_NORMAL_NODE_STATES as SLURM_NORMAL_NODE_STATES,
    SLURM_UNAVAILABLE_NODE_FLAGS as SLURM_UNAVAILABLE_NODE_FLAGS,
    _count_slurm_array_task_expression as _count_slurm_array_task_expression,
    _extract_slurm_pending_reason as _extract_slurm_pending_reason,
    _iter_scontrol_node_blocks as _iter_scontrol_node_blocks,
    _looks_like_slurm_state_token as _looks_like_slurm_state_token,
    _normalize_slurm_node_state as _normalize_slurm_node_state,
    _parse_key_value_fields as _parse_key_value_fields,
    _parse_squeue_row_items as _parse_squeue_row_items,
    _partition_state_is_up as _partition_state_is_up,
    _slurm_node_capacity as _slurm_node_capacity,
    _slurm_node_status as _slurm_node_status,
    _slurm_state_flags as _slurm_state_flags,
    _split_squeue_row as _split_squeue_row,
    _strict_nonnegative_int as _strict_nonnegative_int,
    estimate_slurm_task_count as estimate_slurm_task_count,
    get_scontrol_node_df as get_scontrol_node_df,
    get_scontrol_partition_df as get_scontrol_partition_df,
    get_sprio_df as get_sprio_df,
    get_squeue_user_df as get_squeue_user_df,
    get_sshare_df as get_sshare_df,
)
from kfbatch.uge_parser import (
    QSTAT_COLUMNS as QSTAT_COLUMNS,
    UGE_JOB_COLUMNS as UGE_JOB_COLUMNS,
    _empty_uge_job_df as _empty_uge_job_df,
    _iter_uge_json_jobs as _iter_uge_json_jobs,
    _merge_qstat_common_rows as _merge_qstat_common_rows,
    _merge_qstat_iteration_min_availability as _merge_qstat_iteration_min_availability,
    _optional_int as _optional_int,
    _parse_uge_task_expression as _parse_uge_task_expression,
    _parse_uge_text_job_line as _parse_uge_text_job_line,
    _qfree_group_context as _qfree_group_context,
    _uge_json_job_row as _uge_json_job_row,
    get_qfree_df as get_qfree_df,
    get_qstat_df as get_qstat_df,
    get_uge_json_job_df as get_uge_json_job_df,
    get_user_df as get_user_df,
)

grp: Any
pwd: Any
try:
    import grp
    import pwd
except ImportError:  # pragma: no cover - schedulers are normally queried on POSIX hosts
    grp = None
    pwd = None

SLURM_KNOWN_JOB_STATES = (
    SLURM_RUNNING_STATES | SLURM_PENDING_STATES | SLURM_ERROR_STATES | {"CD", "RS", "SI", "SO", "S"}
)
SLURM_SQUEUE_PARSE_FIELDS = "%i\t%P\t%j\t%u\t%a\t%t\t%M\t%D\t%C\t%m\t%l\t%R"
MAX_QSTAT_SNAPSHOTS = 100
MAX_QSTAT_SAMPLING_SECONDS = 300.0


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


def _iter_scontrol_named_blocks(lines, anchor_key):
    current: list[str] = []
    for raw_line in lines:
        line = raw_line.strip()
        if line == "":
            if current:
                yield current
                current = []
            continue
        if line.startswith(anchor_key) and current:
            yield current
            current = [line]
            continue
        if not current:
            current = [line]
        else:
            current.append(line)
    if current:
        yield current


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
    current: list[str] = []
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


MAX_EXPANDED_HOSTS = 100_000
MAX_HOSTLIST_GROUPS = 16
MAX_EXPANDED_HOST_BYTES = 16 * 1024 * 1024


def _hostlist_values(expression, budget):
    values: list[str] = []
    size = 0
    for item in expression.split(","):
        interval = re.fullmatch(r"([0-9]+)-([0-9]+)", item)
        if interval is None:
            if not item.isdigit():
                raise ValueError("invalid hostlist range")
            count, width = 1, len(item)
        else:
            first, last = interval.groups()
            start, end = int(first), int(last)
            count, width = end - start + 1, max(len(first), len(last))
        size += count * width
        if count < 1 or count > budget - len(values) or size > MAX_EXPANDED_HOST_BYTES:
            raise ValueError("hostlist expansion exceeds safety limit")
        if interval is None:
            values.append(item)
        else:
            values.extend(f"{number:0{width}d}" for number in range(start, end + 1))
    return values


def _expand_slurm_hostlist(value):
    """Expand a hostlist only after bounding each range and Cartesian product."""
    text = str(value).strip()
    if text.upper() == "ALL":
        return ["*"]
    hosts: list[str] = []
    host_bytes = 0
    for token in _split_slurm_hostlist(text):
        parts = [""]
        groups = 0
        while "[" in token:
            groups += 1
            match = re.search(r"\[([^\[\]]+)\]", token)
            if match is None or groups > MAX_HOSTLIST_GROUPS:
                raise ValueError("invalid or excessively nested hostlist")
            prefix, expression, token = token[: match.start()], match.group(1), token[match.end() :]
            if "[" in prefix or "]" in prefix:
                raise ValueError("nested hostlist is unsupported")
            budget = (MAX_EXPANDED_HOSTS - len(hosts)) // len(parts)
            values = _hostlist_values(expression, budget)
            product_bytes = (sum(map(len, parts)) + len(parts) * len(prefix)) * len(values) + len(
                parts
            ) * sum(map(len, values))
            if product_bytes + host_bytes > MAX_EXPANDED_HOST_BYTES:
                raise ValueError("hostlist expansion exceeds byte limit")
            parts = [part + prefix + value for part in parts for value in values]
        if "]" in token or len(hosts) + len(parts) > MAX_EXPANDED_HOSTS:
            raise ValueError("invalid or excessive hostlist expansion")
        host_bytes += sum(map(len, parts)) + len(parts) * len(token)
        if host_bytes > MAX_EXPANDED_HOST_BYTES:
            raise ValueError("hostlist expansion exceeds byte limit")
        hosts.extend(part + token for part in parts)
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
    if grp is None or not hasattr(os, "getgroups") or not hasattr(os, "getgid"):
        return set()
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
    current_groups = set(_current_group_names()) if current_groups is None else set(current_groups)
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
    rejected_nodes = []
    for line in explicit_lines:
        params = _parse_key_value_fields(line)
        node_name = params.get("NodeName", "").strip()
        if node_name == "":
            rejected_nodes.append("<unknown>")
            continue
        reserved_cores = _count_core_id_expression(params.get("CoreIDs", ""))
        if reserved_cores == 0 and context["node_count"] == 1:
            reserved_cores = context["default_reserved_cores"]
        if reserved_cores <= 0:
            rejected_nodes.append(node_name)
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
    return rows, bool(explicit_lines), rejected_nodes


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
    rows, has_explicit_rows, rejected_nodes = _explicit_reservation_rows(block, context)
    if has_explicit_rows:
        if rejected_nodes:
            warning = "active reservation {} has unparseable CoreIDs for {}".format(
                reservation_name or "<unknown>",
                ",".join(rejected_nodes),
            )
            return rows, warning
        return rows, None
    try:
        rows = _hostlist_reservation_rows(header_params, context)
    except ValueError as error:
        return [], f"active reservation {reservation_name or '<unknown>'}: {error}"
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
    unresolved_partitions = set()
    for block in _iter_scontrol_named_blocks(lines, "ReservationName="):
        header_params = _reservation_header_params(block)
        block_rows, warning = _parse_reservation_block(
            block,
            current_user,
            current_accounts,
            current_groups,
        )
        rows.extend(block_rows)
        if warning is not None:
            warnings.append(warning)
            if str(header_params.get("State", "")).strip().upper() in {"", "ACTIVE"}:
                partition_name = str(header_params.get("PartitionName", "")).strip()
                if partition_name in {"(null)", "N/A"}:
                    partition_name = ""
                unresolved_partitions.add(partition_name)
    frame = pandas.DataFrame(rows, columns=SLURM_RESERVATION_COLUMNS)
    frame.attrs["warnings"] = warnings
    frame.attrs["unresolved_partitions"] = sorted(unresolved_partitions)
    return frame


def _expand_reservation_rows(df_node, df_reservation):
    available_nodes = set(df_node["node_name"].dropna().astype(str))
    rows = []
    unresolved_targets = []
    unresolved_partitions = set()
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
            unresolved_partitions.add(queue_name)
            continue
        for target in targets:
            row = reservation.to_dict()
            row["node_name"] = target
            rows.append(row)
    expanded = pandas.DataFrame(rows, columns=df_reservation.columns)
    expanded.attrs["unresolved_targets"] = unresolved_targets
    expanded.attrs["unresolved_partitions"] = sorted(unresolved_partitions)
    return expanded


def apply_slurm_reservations(df_node, df_reservation):
    if (
        df_node is None
        or df_node.shape[0] == 0
        or df_reservation is None
        or df_reservation.shape[0] == 0
    ):
        return df_node
    if df_node.attrs.get("slurm_reservations_applied", False):
        return df_node
    df = df_node.copy()
    for col in ["reservation_cores", "reservation_mem_mb"]:
        if col not in df.columns:
            df[col] = 0
    if "reservation_accessible" not in df.columns:
        df["reservation_accessible"] = False

    reservation_rows = _expand_reservation_rows(df, df_reservation)
    unresolved_targets = reservation_rows.attrs.get("unresolved_targets", [])
    unresolved_partitions = reservation_rows.attrs.get("unresolved_partitions", [])
    df.attrs["reservation_unresolved_targets"] = unresolved_targets
    df.attrs["reservation_unresolved_partitions"] = unresolved_partitions
    if reservation_rows.shape[0] == 0:
        df.attrs["slurm_reservations_applied"] = True
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
        df.attrs["slurm_reservations_applied"] = True
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
    df.attrs["reservation_unresolved_targets"] = unresolved_targets
    df.attrs["reservation_unresolved_partitions"] = unresolved_partitions
    df.attrs["slurm_reservations_applied"] = True
    return df


def suppress_slurm_resource_ceiling(df_node, partitions, reason):
    if df_node is None or df_node.shape[0] == 0:
        return df_node
    df = df_node.copy()
    partition_set = {str(value or "").strip() for value in partitions}
    if not partition_set or "" in partition_set:
        affected = pandas.Series(True, index=df.index)
    else:
        partition_rows = df["queue_name"].fillna("").astype(str).isin(partition_set)
        nodes = set(df.loc[partition_rows, "node_name"])
        affected = df["node_name"].isin(nodes)
    if not affected.any():
        return df
    df.loc[affected, "status"] = (
        df.loc[affected, "status"]
        .fillna("")
        .astype(str)
        .map(lambda value: "|".join(token for token in [value, reason] if token))
    )
    df.loc[affected, "ncore_available"] = 0
    df.loc[affected, "hc:mem_req"] = "0M"
    if "hc:mem_req_known" in df.columns:
        df.loc[affected, "hc:mem_req_known"] = False
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


def _print_scoped_job_totals(self_text, all_text, scope):
    if scope == "self":
        print(f"jobs  {self_text}")
    elif scope == "all":
        print(f"jobs  {all_text}")
    else:
        print(f"jobs  {self_text}  {all_text}")


def print_queued_job_summary(
    df_user,
    scheduler="uge",
    current_user="",
    all_users=True,
    scope="overview",
):
    if rejected_rows(df_user):
        print(f"note: incomplete job data: {rejected_rows(df_user)} row(s) rejected")
    if scope == "group":
        return
    if scheduler == "slurm":
        if df_user.shape[0] == 0:
            print("No jobs found in squeue output.")
            print("")
            return
        state_codes = df_user["state"].fillna("").map(_normalize_slurm_job_state)
        is_running = state_codes.isin(SLURM_RUNNING_STATES)
        is_qwaiting = state_codes.isin(SLURM_PENDING_STATES)
        is_error = state_codes.isin(SLURM_ERROR_STATES)
        is_other = ~(is_running | is_qwaiting | is_error)
        num_running = int(df_user.loc[is_running, "total_slots"].sum())
        num_qwaiting = int(df_user.loc[is_qwaiting, "total_slots"].sum())
        num_error = int(df_user.loc[is_error, "total_slots"].sum())
        num_other = int(df_user.loc[is_other, "total_slots"].sum())
        if (current_user != "") and ("user" in df_user.columns):
            is_self = df_user["user"].fillna("") == current_user
            num_running_self = int(df_user.loc[is_running & is_self, "total_slots"].sum())
            num_qwaiting_self = int(df_user.loc[is_qwaiting & is_self, "total_slots"].sum())
            num_error_self = int(df_user.loc[is_error & is_self, "total_slots"].sum())
            num_other_self = int(df_user.loc[is_other & is_self, "total_slots"].sum())
            self_text = (
                "self:R/Q/X/O="
                f"{num_running_self}/{num_qwaiting_self}/{num_error_self}/{num_other_self}"
            )
            all_text = f"all:R/Q/X/O={num_running}/{num_qwaiting}/{num_error}/{num_other}"
            _print_scoped_job_totals(self_text, all_text, scope)
        else:
            print(f"# of running job tasks (estimated from squeue): {num_running}")
            print(f"# of queued job tasks (estimated from squeue): {num_qwaiting}")
            print(f"# of terminal/error job tasks currently visible in squeue: {num_error}")
            print(f"# of other-state job tasks currently visible in squeue: {num_other}")
        unknown_states = sorted(
            {
                value
                for value in state_codes.loc[is_other].dropna().astype(str)
                if value and value not in SLURM_KNOWN_JOB_STATES
            }
        )
        if unknown_states:
            print(f"note: unknown SLURM job state(s): {','.join(unknown_states)}")
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
        self_text = f"self:R/Q/F={num_running_self}/{num_qwaiting_self}/{num_error_self}"
        all_text = f"all:R/Q/F={num_running}/{num_qwaiting}/{num_error}"
        _print_scoped_job_totals(self_text, all_text, scope)
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
    if pwd is None or not hasattr(os, "geteuid"):
        return ""
    try:
        return pwd.getpwuid(os.geteuid()).pw_name.strip()
    except (KeyError, OSError):
        return ""


def _current_user_from_args(args):
    explicit = str(getattr(args, "current_user", "") or "").strip()
    return explicit or get_current_user_name()


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
    pending_account = ""
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
                # Keep the pending association distinct from the overall
                # best association instead of silently attributing its rank
                # to current_account.
                current_pending = df_pending_ranked.loc[
                    df_pending_ranked["user"] == current_user, :
                ]
            if current_pending.shape[0] > 0:
                pending_row = current_pending.sort_values(by=["pending_fairshare_rank"]).iloc[0]
                pending_rank = int(pending_row["pending_fairshare_rank"])
                pending_account = str(pending_row["account"])

    return {
        "user": current_user,
        "account": current_account,
        "association_count": association_count,
        "fairshare": float(current_row["fairshare"]),
        "overall_rank": int(current_row["fairshare_rank"]),
        "overall_total": int(df_ranked.shape[0]),
        "pending_rank": pending_rank,
        "pending_account": pending_account,
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
    pending_account = str(summary.get("pending_account", "") or "").strip()
    pending_total = int(summary.get("pending_total", 0) or 0)
    if pending_rank is not None and pending_total > 0:
        if pending_account and pending_account != account:
            fields.append(f"pending_account={pending_account}")
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


def _index_slurm_rows_by_partition(frame):
    if frame is None or frame.shape[0] == 0 or "partition" not in frame.columns:
        return {}
    expanded = frame.copy()
    expanded["_queue_name"] = expanded["partition"].map(_split_slurm_partition_field)
    expanded = expanded.explode("_queue_name")
    expanded = expanded.loc[expanded["_queue_name"].notna(), :]
    return {
        str(queue_name): rows.drop(columns=["_queue_name"]).reset_index(drop=True)
        for queue_name, rows in expanded.groupby("_queue_name", sort=False)
    }


def _slurm_priority_gaps(user_pending, df_prio_queue):
    if df_prio_queue is None or df_prio_queue.shape[0] == 0:
        return None, None
    top_priority = int(df_prio_queue["priority"].max())
    top_fairshare = int(df_prio_queue["fairshare"].max())
    df_user_prio = df_prio_queue.loc[df_prio_queue["job_id"].isin(user_pending["job_id"]), :]
    if df_user_prio.shape[0] == 0:
        return None, None
    return (
        top_priority - int(df_user_prio["priority"].max()),
        top_fairshare - int(df_user_prio["fairshare"].max()),
    )


def _smallest_priority_blocked_request(user_pending):
    user_priority_pending = user_pending.loc[
        user_pending["pending_reason"].fillna("").str.contains("Priority", case=False, regex=False),
        :,
    ].copy()
    if user_priority_pending.shape[0] == 0:
        return None, None, "", "resource_only"
    if "resource_fields_complete" not in user_priority_pending.columns:
        user_priority_pending["resource_fields_complete"] = False
    if "num_nodes" not in user_priority_pending.columns:
        user_priority_pending["num_nodes"] = 1
    valid = user_priority_pending.loc[
        user_priority_pending["resource_fields_complete"].fillna(False)
        & (user_priority_pending["num_nodes"] == 1),
        :,
    ].copy()
    if valid.shape[0] == 0:
        return None, None, "", "priority_blocked_missing_fields"
    valid["time_limit_minutes"] = valid["time_limit"].map(_slurm_time_to_minutes)
    valid["req_mem_gb"] = valid.apply(
        lambda row: slurm_request_memory_gib(
            row["req_mem"],
            req_cpus=row["req_cpus"],
            num_nodes=row["num_nodes"],
        ),
        axis=1,
    )
    known_memory = valid.loc[valid["req_mem_gb"].notna(), :].copy()
    if known_memory.shape[0] > 0:
        smallest = (
            known_memory.sort_values(
                by=["req_cpus", "req_mem_gb", "time_limit_minutes", "job_id"],
                ascending=[True, True, True, True],
                na_position="last",
            )
            .reset_index(drop=True)
            .iloc[0]
        )
        return (
            int(smallest["req_cpus"]),
            float(smallest["req_mem_gb"]),
            str(smallest["time_limit"]).strip(),
            "priority_blocked",
        )
    smallest = (
        valid.sort_values(
            by=["req_cpus", "time_limit_minutes", "job_id"],
            ascending=[True, True, True],
            na_position="last",
        )
        .reset_index(drop=True)
        .iloc[0]
    )
    return (
        int(smallest["req_cpus"]),
        None,
        str(smallest["time_limit"]).strip(),
        "priority_blocked_ambiguous_memory",
    )


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
    user_pending_by_partition = {}
    if (
        current_user
        and df_job is not None
        and df_job.shape[0] > 0
        and {"state", "user", "partition"}.issubset(df_job.columns)
    ):
        state_codes = df_job["state"].fillna("").map(_normalize_slurm_job_state)
        user_pending = df_job.loc[
            (df_job["user"].fillna("") == current_user) & state_codes.isin(SLURM_PENDING_STATES),
            :,
        ].copy()
        user_pending_by_partition = _index_slurm_rows_by_partition(user_pending)
    prio_by_partition = _index_slurm_rows_by_partition(df_prio)
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
        if current_user:
            user_pending = user_pending_by_partition.get(
                queue_name,
                pandas.DataFrame(columns=df_job.columns if df_job is not None else []),
            )
            if user_pending.shape[0] > 0:
                df_prio_queue = prio_by_partition.get(queue_name)
                priority_gap, fairshare_gap = _slurm_priority_gaps(
                    user_pending,
                    df_prio_queue,
                )
                (
                    blocked_req_cores,
                    blocked_req_mem_gb,
                    blocked_time_limit,
                    status,
                ) = _smallest_priority_blocked_request(user_pending)
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


def _parsed_empty_but_input_unrecognized(frame):
    if frame is None or frame.shape[0] > 0:
        return False
    attrs = frame.attrs
    return bool(
        attrs.get("candidate_rows", 0) > 0
        or (attrs.get("input_nonempty", False) and not attrs.get("recognized_header", False))
    )


def _print_slurm_fairshare(args, df_user, current_user, timeout_seconds):
    show_rank = getattr(args, "show_fairshare_rank", True)
    needs_group_discovery = (
        getattr(args, "scope", "overview") in {"overview", "group"}
        and not str(getattr(args, "group_id", "") or "").strip()
    )
    if not show_rank and not needs_group_discovery:
        return None
    share_lines = get_command_stdout_lines(
        command_str=getattr(args, "slurm_share_command", "sshare -a -P"),
        example_file=getattr(args, "slurm_share_example_file", ""),
        allow_failure=True,
        command_name="--slurm_share_command",
        quiet_failure=True,
        timeout_seconds=timeout_seconds,
    )
    if share_lines is None:
        component = "Slurm account/FairShare" if needs_group_discovery else "Slurm FairShare"
        _print_degraded(component, "--slurm_share_command failed or timed out")
        return None
    df_share = get_sshare_df(share_lines)
    if show_rank:
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
    elif df_share.shape[0] == 0:
        _print_degraded(
            "Slurm account",
            "command succeeded but no association rows were parsed",
        )
    return df_share


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
    current_user = _current_user_from_args(args)
    lines = get_command_stdout_lines(
        command_str=get_squeue_command_for_parsing(args.stat_command),
        example_file=args.example_file,
        allow_failure=False,
        command_name="--stat_command",
        timeout_seconds=timeout_seconds,
    )
    df_user = get_squeue_user_df(lines)
    if _parsed_empty_but_input_unrecognized(df_user):
        raise KFBatchCommandError(
            "SLURM job output was non-empty but contained no recognized squeue rows."
        )
    if rejected_rows(df_user):
        raise KFBatchCommandError(
            f"SLURM job output rejected {rejected_rows(df_user)} row(s); totals are incomplete."
        )
    print_queued_job_summary(
        df_user,
        scheduler="slurm",
        current_user=current_user,
        scope=getattr(args, "scope", "overview"),
    )
    df_share = _print_slurm_fairshare(args, df_user, current_user, timeout_seconds)
    if getattr(args, "scope", "overview") in {"overview", "group"}:
        from kfbatch.batch_scope import print_group_job_summary

        print_group_job_summary(
            df_user,
            scheduler="slurm",
            current_user=current_user,
            group_id=getattr(args, "group_id", ""),
            by_user=getattr(args, "by_user", False),
            share_frame=df_share,
        )
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
    first_character = next(
        (stripped[0] for line in job_lines if (stripped := str(line).lstrip()) != ""),
        "",
    )
    if first_character not in {"{", "["}:
        parsed = get_user_df(job_lines)
        if _parsed_empty_but_input_unrecognized(parsed) or rejected_rows(parsed):
            _print_degraded(
                "AGE/UGE/SGE all-user jobs",
                "text schema was not recognized or rows were rejected; using jobs embedded in qstat -F",
            )
            return fallback, False
        return parsed, True
    parsed = get_uge_json_job_df(job_lines)
    if parsed is not None and not rejected_rows(parsed):
        return parsed, True
    _print_degraded(
        "AGE/UGE/SGE all-user jobs",
        "JSON schema was not recognized or rows were rejected; using jobs embedded in qstat -F",
    )
    return fallback, False


def _get_uge_df(args, timeout_seconds):
    if args.niter < 1:
        raise KFBatchUsageError("Exiting. --niter must be >= 1 when using qstat mode.")
    if args.niter > MAX_QSTAT_SNAPSHOTS:
        raise KFBatchUsageError(f"Exiting. --niter must be <= {MAX_QSTAT_SNAPSHOTS}.")
    df = None
    df_user = None
    sampling_started = time.monotonic()
    for iteration in range(args.niter):
        remaining_sampling_seconds = MAX_QSTAT_SAMPLING_SECONDS - (
            time.monotonic() - sampling_started
        )
        if remaining_sampling_seconds <= 0:
            raise KFBatchCommandError(
                f"AGE/UGE/SGE sampling exceeded {MAX_QSTAT_SAMPLING_SECONDS:g} seconds."
            )
        iteration_timeout = remaining_sampling_seconds
        if timeout_seconds is not None and float(timeout_seconds) > 0:
            iteration_timeout = min(float(timeout_seconds), remaining_sampling_seconds)
        lines = get_command_stdout_lines(
            command_str=args.stat_command,
            example_file=args.example_file,
            allow_failure=False,
            command_name="--stat_command",
            timeout_seconds=iteration_timeout,
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
            current_user=_current_user_from_args(args),
            all_users=has_all_user_jobs,
            scope=getattr(args, "scope", "overview"),
        )
        df_user.attrs["all_users"] = has_all_user_jobs
        if "parse_quality" in df_user.attrs:
            df_user.attrs["parse_quality"]["scope"] = "all" if has_all_user_jobs else "observed"
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


def adjust_ram_unit(df, scheduler="slurm"):
    for col in ["hc:mem_req", "hl:mem_total"]:
        if scheduler == "uge":
            values = grid_engine_memory_series_to_gib(df[col])
        else:
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
    try:
        existing_mode = target.stat().st_mode & 0o777
    except FileNotFoundError:
        existing_mode = None
    except OSError as error:
        raise KFBatchCommandError(f"Failed to inspect {label}: {target}: {error}") from error
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
        if existing_mode is not None:
            os.chmod(temporary_path, existing_mode)
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
    current_user = _current_user_from_args(args)
    current_groups: set[str] | None = None
    if str(getattr(args, "current_user", "") or "").strip():
        # A remote/fixture scheduler user cannot safely inherit local groups.
        current_groups = set()
    df_reservation = get_scontrol_reservation_df(
        reservation_lines,
        current_user=current_user,
        current_accounts=_get_current_slurm_accounts(df_user, current_user),
        current_groups=current_groups,
    )
    warnings = df_reservation.attrs.get("warnings", [])
    unresolved_partitions = set(df_reservation.attrs.get("unresolved_partitions", []))
    for warning in warnings:
        _print_degraded("Slurm reservation", warning)
    if df_reservation.shape[0] > 0:
        df = apply_slurm_reservations(df, df_reservation)
        for target in df.attrs.get("reservation_unresolved_targets", []):
            _print_degraded(
                "Slurm reservation",
                f"target could not be matched to a parsed node: {target}",
            )
        unresolved_partitions.update(df.attrs.get("reservation_unresolved_partitions", []))
    if unresolved_partitions:
        affected_text = (
            "all partitions"
            if "" in unresolved_partitions
            else ",".join(sorted(unresolved_partitions))
        )
        _print_degraded(
            "Slurm reservation",
            f"unresolved active reservation affects {affected_text}; "
            "resource ceilings are suppressed",
        )
        df = suppress_slurm_resource_ceiling(
            df,
            unresolved_partitions,
            "reservation_state=UNKNOWN",
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
        current_user=_current_user_from_args(args),
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
    df = adjust_ram_unit(df, scheduler=scheduler)
    if scheduler == "slurm":
        print_slurm_compact_summary(
            df,
            _get_slurm_launch_frame(df, df_user, args, timeout_seconds),
            args,
        )
    else:
        df_qfree = _get_qfree_frame(args, timeout_seconds)
        if getattr(args, "scope", "overview") in {"overview", "group"}:
            from kfbatch.batch_scope import print_group_job_summary

            print_group_job_summary(
                df_user,
                scheduler="uge",
                current_user=_current_user_from_args(args),
                group_id=getattr(args, "group_id", ""),
                by_user=getattr(args, "by_user", False),
                qfree_frame=df_qfree,
            )
        print_uge_compact_summary(df, df_qfree, args)
    if node_output_path:
        _atomic_write_tsv(df, node_output_path, "node TSV")
