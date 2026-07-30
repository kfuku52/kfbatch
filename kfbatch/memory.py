"""Scheduler memory parsing and safe display helpers."""

from __future__ import annotations

import math
import re

import pandas

_MIB_PER_UNIT = {
    "K": 1.0 / 1024.0,
    "M": 1.0,
    "G": 1024.0,
    "T": 1024.0 * 1024.0,
}


def memory_text_to_mib(value, default_unit="G"):
    """Convert a Grid Engine or Slurm memory token to mebibytes.

    Scheduler K/M/G/T suffixes are binary. Slurm's optional trailing ``c``,
    ``n``, or ``g`` request-scope marker is ignored by this scalar converter.
    """

    if value is None or pandas.isna(value):
        return float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTkmgt])?([cngCNG])?",
        text,
    )
    if match is None:
        return float("nan")
    number = float(match.group(1))
    unit = (match.group(2) or default_unit).upper()
    return number * _MIB_PER_UNIT[unit]


def memory_text_to_gib(value, default_unit="G"):
    mib = memory_text_to_mib(value, default_unit=default_unit)
    if pandas.isna(mib):
        return float("nan")
    return float(mib) / 1024.0


def memory_series_to_gib(series, default_unit="G"):
    return series.map(lambda value: memory_text_to_gib(value, default_unit=default_unit))


def slurm_request_memory_gib(value, req_cpus=1, num_nodes=1):
    """Return a pending Slurm request's per-node memory in GiB.

    ``c`` requests are per CPU, ``n`` and unsuffixed requests are per node,
    and ``g`` requests cannot be resolved without a GPU count.
    """

    if value is None or pandas.isna(value):
        return float("nan")
    text = str(value).strip()
    match = re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTkmgt])?([cngCNG])?",
        text,
    )
    if match is None:
        return float("nan")
    scope = (match.group(3) or "n").lower()
    if scope == "g":
        return float("nan")
    base_mib = memory_text_to_mib(text, default_unit="M")
    if pandas.isna(base_mib):
        return float("nan")
    if scope == "c":
        node_count = max(int(num_nodes), 1)
        cpus_per_node = int(math.ceil(max(int(req_cpus), 1) / node_count))
        base_mib *= cpus_per_node
    return float(base_mib) / 1024.0


def floor_gib(value):
    if value is None or pandas.isna(value):
        return None
    return max(int(math.floor(float(value))), 0)
