"""Safe execution and decoding of scheduler commands."""

from __future__ import annotations

import locale
import shlex
import subprocess

from kfbatch.errors import KFBatchCommandError

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0


def _format_error_message(summary, detail="", quiet=False):
    if quiet or not str(detail).strip():
        return summary
    return f"{summary}\n{detail}"


def decode_scheduler_output(data):
    """Decode scheduler bytes without failing on arbitrary job-name bytes."""

    if isinstance(data, str):
        return data
    preferred = locale.getpreferredencoding(False) or "utf-8"
    encodings = list(dict.fromkeys([preferred, "utf-8"]))
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode(encodings[0], errors="replace")


def get_command_stdout_lines(
    command_str,
    example_file="",
    allow_failure=False,
    command_name="command",
    quiet_failure=False,
    timeout_seconds=DEFAULT_COMMAND_TIMEOUT_SECONDS,
):
    if example_file:
        try:
            with open(example_file, "rb") as handle:
                return decode_scheduler_output(handle.read()).splitlines()
        except OSError as error:
            if allow_failure:
                return None
            summary = f"Failed to read example file for {command_name}: {example_file}"
            raise KFBatchCommandError(
                _format_error_message(summary, str(error), quiet=quiet_failure)
            ) from error
    try:
        command = shlex.split(command_str)
    except ValueError as error:
        if allow_failure:
            return None
        summary = f"Failed to parse {command_name}: {command_str}"
        raise KFBatchCommandError(
            _format_error_message(summary, str(error), quiet=quiet_failure)
        ) from error
    if not command:
        if allow_failure:
            return None
        summary = f"Failed to run {command_name}: command is empty"
        raise KFBatchCommandError(_format_error_message(summary, quiet=quiet_failure))
    timeout = None
    if timeout_seconds is not None and float(timeout_seconds) > 0:
        timeout = float(timeout_seconds)
    try:
        command_out = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        if allow_failure:
            return None
        summary = f"Timed out running {command_name} after {timeout:g}s: {command_str}"
        raise KFBatchCommandError(_format_error_message(summary, quiet=quiet_failure)) from error
    except OSError as error:
        if allow_failure:
            return None
        summary = f"Failed to run {command_name}: {command_str}"
        raise KFBatchCommandError(
            _format_error_message(summary, str(error), quiet=quiet_failure)
        ) from error
    if command_out.returncode != 0:
        if allow_failure:
            return None
        command_stderr = decode_scheduler_output(command_out.stderr).strip()
        summary = f"Failed to run {command_name}: {command_str}"
        raise KFBatchCommandError(
            _format_error_message(summary, command_stderr, quiet=quiet_failure)
        )
    return decode_scheduler_output(command_out.stdout).splitlines()
