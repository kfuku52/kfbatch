"""Render helpers, independent of command execution."""

import re

import pandas

from kfbatch.memory import (
    floor_gib,
)


def _slurm_time_to_minutes(value):
    txt = str(value).strip().upper()
    if txt in ["", "N/A", "UNLIMITED", "NOT_SET", "INFINITE"]:
        return float("inf")
    match = re.fullmatch(
        r"(?:(?P<days>[0-9]+)-)?(?:(?P<hours>[0-9]+):)?"
        r"(?P<minutes>[0-9]+)(?::(?P<seconds>[0-9]+))?",
        txt,
    )
    if match is None:
        return float("nan")
    day_part = int(match.group("days") or 0)
    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes"))
    seconds = int(match.group("seconds") or 0)
    has_days = match.group("days") is not None
    has_hours = match.group("hours") is not None
    has_seconds = match.group("seconds") is not None
    if has_days and (not has_hours or not has_seconds or hours > 23):
        return float("nan")
    if has_hours and not has_seconds:
        # Two-component values are minutes:seconds, not hours:minutes.
        seconds = minutes
        minutes = hours
        hours = 0
    if seconds > 59 or (has_hours and minutes > 59):
        return float("nan")
    total_minutes = (day_part * 24 * 60) + (hours * 60) + minutes + (seconds / 60.0)
    return float(total_minutes)


def _format_slurm_compact_time_limit(time_limit):
    txt = str(time_limit).strip()
    if txt in ["", "nan", "N/A", "NOT_SET"]:
        return "?"
    total_minutes = _slurm_time_to_minutes(txt)
    if total_minutes == float("inf"):
        return "inf"
    if pandas.isna(total_minutes):
        return "?"
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
        if pandas.isna(threshold):
            selected = ordered.iloc[:limit, :]
        else:
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
    if status in [
        "priority_blocked",
        "priority_blocked_ambiguous_memory",
        "priority_blocked_missing_fields",
    ]:
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


def _print_compact_table(rows, columns):
    """Print preformatted cells with shared column widths and two-space separators."""
    widths = {}
    for col in columns:
        widths[col] = max([len(col)] + [len(str(row[col])) for row in rows])
    print("  ".join([col.ljust(widths[col]) for col in columns]))
    for row in rows:
        print("  ".join([str(row[col]).ljust(widths[col]) for col in columns]))


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
    _print_compact_table(rows, columns)
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
    _print_compact_table(rows, columns)
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
