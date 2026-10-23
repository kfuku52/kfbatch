"""Group-aware job aggregation shared by Slurm and Grid Engine reports."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import repeat

from kfbatch.job_states import slurm_bucket as _slurm_bucket, uge_bucket as _uge_bucket


@dataclass(frozen=True)
class JobTotals:
    running: int = 0
    queued: int = 0
    failed: int = 0
    other: int = 0


def _job_bucket_getter(scheduler, row_count):
    """Reuse pure string states within this report, with a bounded vocabulary."""
    is_slurm = scheduler == "slurm"
    cache: dict[str, str] | None = {} if row_count >= 128 else None

    def classify(state, queue):
        nonlocal cache
        if cache is None or type(state) is not str or (not is_slurm and type(queue) is not str):
            return _slurm_bucket(state) if is_slurm else _uge_bucket(state, queue)
        bucket = cache.get(state)
        if bucket is None:
            bucket = _slurm_bucket(state) if is_slurm else _uge_bucket(state, "")
            if len(cache) < 128:
                cache[state] = bucket
            else:
                cache = None
        # An assigned queue changes only an otherwise unclassified UGE state.
        return "running" if not is_slurm and bucket == "other" and queue.strip() else bucket

    return classify


def aggregate_jobs(frame, scheduler):
    counts = {"running": 0, "queued": 0, "failed": 0, "other": 0}
    if frame is None or frame.shape[0] == 0:
        return JobTotals()
    states = frame.get("state", repeat("", len(frame)))
    queues = frame.get("queue_name", repeat("", len(frame)))
    slot_counts = frame.get("total_slots", repeat(0, len(frame)))
    classify = _job_bucket_getter(scheduler, len(frame))
    for state, queue, value in zip(states, queues, slot_counts, strict=True):
        bucket = classify(state, queue)
        try:
            slots = int(value)
        except (TypeError, ValueError):
            slots = 0
        counts[bucket] += max(slots, 0)
    return JobTotals(**counts)


def _print_totals(label, totals, scheduler):
    if scheduler == "slurm":
        print(
            f"jobs  {label}:R/Q/X/O={totals.running}/{totals.queued}/{totals.failed}/{totals.other}"
        )
    else:
        print(f"jobs  {label}:R/Q/F={totals.running}/{totals.queued}/{totals.failed}")


def _print_user_breakdown(frame, scheduler, users=None):
    known_users = set(users or [])
    totals_by_user: dict[str, dict[str, int]] = {}
    if frame is not None and "user" in frame.columns:
        # Retain first-seen user evaluation order without copying all job columns.
        user_indices = frame.groupby(frame["user"].fillna("").astype(str), sort=False).indices
        states = list(frame.get("state", repeat("", len(frame))))
        queues = list(frame.get("queue_name", repeat("", len(frame))))
        slot_counts = list(frame.get("total_slots", repeat(0, len(frame))))
        classify = _job_bucket_getter(scheduler, len(frame))
        for user, positions in user_indices.items():
            if not user:
                continue
            counts = {"running": 0, "queued": 0, "failed": 0, "other": 0}
            totals_by_user[user] = counts
            for position in positions:
                bucket = classify(states[position], queues[position])
                try:
                    slots = int(slot_counts[position])
                except (TypeError, ValueError):
                    slots = 0
                counts[bucket] += max(slots, 0)
        known_users.update(totals_by_user)
    for user in sorted(user for user in known_users if user):
        totals = JobTotals(**totals_by_user[user]) if user in totals_by_user else JobTotals()
        if scheduler == "slurm":
            print(
                f"      {user}:R/Q/X/O="
                f"{totals.running}/{totals.queued}/{totals.failed}/{totals.other}"
            )
        else:
            print(f"      {user}:R/Q/F={totals.running}/{totals.queued}/{totals.failed}")


def _slurm_groups(frame, share_frame, current_user, explicit_group):
    if explicit_group:
        members = []
        if share_frame is not None and share_frame.shape[0] > 0:
            members = (
                share_frame.loc[share_frame["account"] == explicit_group, "user"]
                .dropna()
                .astype(str)
                .tolist()
            )
        return [(explicit_group, members)]
    if share_frame is None or share_frame.shape[0] == 0:
        return []
    accounts = (
        share_frame.loc[share_frame["user"].fillna("") == current_user, "account"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    groups = []
    for account in accounts:
        members = (
            share_frame.loc[share_frame["account"] == account, "user"].dropna().astype(str).tolist()
        )
        groups.append((account, members))
    return groups


def _print_slurm_group_summary(
    frame,
    share_frame,
    current_user,
    explicit_group,
    by_user,
):
    groups = _slurm_groups(frame, share_frame, current_user, explicit_group)
    if not groups:
        print("jobs  group: unavailable (no Slurm account association was discovered)")
        return False
    if "account" not in frame.columns or (
        not frame.empty and frame["account"].fillna("").str.strip().eq("").any()
    ):
        print("jobs  group: unavailable (squeue output does not contain account data)")
        return False
    for account, members in groups:
        group_frame = frame.loc[frame["account"].fillna("") == account, :]
        _print_totals(f"group[{account}]", aggregate_jobs(group_frame, "slurm"), "slurm")
        if by_user:
            _print_user_breakdown(group_frame, "slurm", members)
    return True


def _qfree_group_context(qfree_frame, explicit_group):
    if qfree_frame is None:
        return "", []
    discovered = str(qfree_frame.attrs.get("group_name", "") or "")
    users = [str(user) for user in qfree_frame.attrs.get("group_users", []) if str(user)]
    if not discovered or (explicit_group and explicit_group != discovered):
        return "", []
    return discovered, users


def _print_uge_group_summary(frame, qfree_frame, explicit_group, by_user):
    group_name, members = _qfree_group_context(qfree_frame, explicit_group)
    has_all_users = bool(frame.attrs.get("all_users", False))
    if group_name and members and has_all_users and "user" in frame.columns:
        group_frame = frame.loc[frame["user"].fillna("").isin(members), :]
        _print_totals(f"group[{group_name}]", aggregate_jobs(group_frame, "uge"), "uge")
        if by_user:
            _print_user_breakdown(group_frame, "uge", members)
        return True
    if group_name and qfree_frame is not None and qfree_frame.shape[0] > 0:
        slots = qfree_frame["group_slots"]
        running = str(int(slots.sum())) if slots.notna().all() else "?"
        print(f"jobs  group[{group_name}]:R/Q/F={running}/?/?  (qfree running total only)")
        return True
    if not has_all_users:
        detail = "all-user qstat and qfree group data are unavailable"
    else:
        detail = "qfree did not identify the current AGE/UGE group"
    print(f"jobs  group: unavailable ({detail})")
    return False


def print_group_job_summary(
    frame,
    *,
    scheduler,
    current_user,
    group_id="",
    by_user=False,
    share_frame=None,
    qfree_frame=None,
):
    """Print group totals without guessing a group identity."""

    if scheduler == "slurm":
        printed = _print_slurm_group_summary(
            frame,
            share_frame,
            current_user,
            group_id,
            by_user,
        )
    else:
        printed = _print_uge_group_summary(frame, qfree_frame, group_id, by_user)
    print("")
    return printed
