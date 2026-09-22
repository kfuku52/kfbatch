"""Parse utils helpers, independent of command execution."""


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
