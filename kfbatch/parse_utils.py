"""Parse utils helpers, independent of command execution."""

from functools import lru_cache


def _map_repeated_strings(series, convert):
    """Cache repeated tokens for one conversion, with bounded memory.

    Sample across the input so mostly unique columns keep the direct path.
    Only strings enter the cache; other scalar types retain their own semantics.
    """
    if len(series) < 256:
        return series.map(convert)
    sample = series.iloc[:: max(len(series) // 128, 1)].iloc[:128].tolist()
    if not all(isinstance(value, str) for value in sample) or len(set(sample)) > 64:
        return series.map(convert)
    cached = lru_cache(maxsize=256)(convert)
    return series.map(lambda value: cached(value) if isinstance(value, str) else convert(value))


def _numeric_task_token_count(token):
    """Return the task count represented by one numeric array token."""

    range_text, step_separator, step_text = token.partition(":")
    start_text, range_separator, end_text = range_text.partition("-")
    if range_separator == "":
        return 1 if step_separator == "" and token.isdigit() else None
    if not (start_text.isdigit() and end_text.isdigit()):
        return None
    if step_separator:
        if not step_text.isdigit():
            return None
        step = int(step_text)
    else:
        step = 1
    start = int(start_text)
    end = int(end_text)
    if step <= 0 or end < start:
        return None
    return ((end - start) // step) + 1


def _safe_int(value, default=0):
    if value is None:
        return default
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
