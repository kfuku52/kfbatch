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
    "P": 1024.0 * 1024.0 * 1024.0,
}
_DECIMAL_BYTES_PER_UNIT = {
    "k": 1000.0,
    "m": 1000.0**2,
    "g": 1000.0**3,
    "t": 1000.0**4,
    "p": 1000.0**5,
}


def _memory_match(value):
    if value is None or pandas.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return re.fullmatch(
        r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTPkmgtp])?([cngCNG])?",
        text,
    )


def memory_text_to_mib(value, default_unit="G"):
    """Convert a Grid Engine or Slurm memory token to mebibytes.

    Scheduler K/M/G/T suffixes are binary. Slurm's optional trailing ``c``,
    ``n``, or ``g`` request-scope marker is ignored by this scalar converter.
    """

    match = _memory_match(value)
    if match is None:
        return float("nan")
    number = float(match.group(1))
    if not math.isfinite(number):
        return float("nan")
    unit = (match.group(2) or default_unit).upper()
    value_mib = number * _MIB_PER_UNIT[unit]
    return value_mib if math.isfinite(value_mib) else float("nan")


def grid_engine_memory_text_to_mib(value):
    """Convert a Grid Engine memory token to MiB.

    Grid Engine uses decimal multipliers for lowercase suffixes, binary
    multipliers for uppercase suffixes, and bytes for an unsuffixed value.
    """

    match = _memory_match(value)
    if match is None or match.group(3):
        return float("nan")
    number = float(match.group(1))
    if not math.isfinite(number):
        return float("nan")
    unit = match.group(2)
    if unit is None:
        value_mib = number / (1024.0**2)
    elif unit.islower():
        value_mib = number * _DECIMAL_BYTES_PER_UNIT[unit] / (1024.0**2)
    else:
        value_mib = number * _MIB_PER_UNIT[unit]
    return value_mib if math.isfinite(value_mib) else float("nan")


def memory_text_to_gib(value, default_unit="G"):
    mib = memory_text_to_mib(value, default_unit=default_unit)
    if pandas.isna(mib):
        return float("nan")
    return float(mib) / 1024.0


def memory_series_to_gib(series, default_unit="G"):
    return series.map(lambda value: memory_text_to_gib(value, default_unit=default_unit))


def grid_engine_memory_text_to_gib(value):
    mib = grid_engine_memory_text_to_mib(value)
    if pandas.isna(mib):
        return float("nan")
    return float(mib) / 1024.0


def grid_engine_memory_series_to_gib(series):
    return series.map(grid_engine_memory_text_to_gib)


def slurm_request_memory_gib(value, req_cpus=1, num_nodes=1):
    """Return a pending Slurm request's per-node memory in GiB.

    ``c`` requests are per CPU and ``n`` requests are per node. Unitless
    ``squeue %m`` values lose that distinction, and ``g`` requests cannot be
    resolved without a GPU count, so both are returned as unknown.
    """

    match = _memory_match(value)
    if match is None:
        return float("nan")
    scope_marker = match.group(3)
    if scope_marker is None:
        # squeue %m removes Slurm's MEM_PER_CPU bit. An unsuffixed value
        # therefore cannot safely be interpreted as either per-node or per-CPU.
        return float("nan")
    scope = scope_marker.lower()
    if scope == "g":
        return float("nan")
    base_mib = memory_text_to_mib(value, default_unit="M")
    if pandas.isna(base_mib):
        return float("nan")
    if scope == "c":
        try:
            node_count = max(int(num_nodes), 1)
            cpus_per_node = int(math.ceil(max(int(req_cpus), 1) / node_count))
        except (TypeError, ValueError, OverflowError):
            return float("nan")
        base_mib *= cpus_per_node
    value_gib = float(base_mib) / 1024.0
    return value_gib if math.isfinite(value_gib) else float("nan")


def floor_gib(value):
    if value is None or pandas.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return max(int(math.floor(number)), 0)
