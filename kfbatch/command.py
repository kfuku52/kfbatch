"""Safe execution and decoding of scheduler commands."""

from __future__ import annotations

import codecs
import locale
import shlex

# Scheduler command strings are explicit program input and are parsed into argv.
import subprocess  # nosec B404
import tempfile

from kfbatch.errors import KFBatchCommandError

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
STDOUT_SPOOL_MEMORY_LIMIT_BYTES = 1024 * 1024
STDERR_DETAIL_LIMIT_BYTES = 64 * 1024


def _format_error_message(summary, detail="", quiet=False):
    if quiet or not str(detail).strip():
        return summary
    return f"{summary}\n{detail}"


def _scheduler_output_encodings():
    """Return valid, de-duplicated decoder names in preference order."""

    preferred = locale.getpreferredencoding(False) or "utf-8"
    encodings = []
    for encoding in (preferred, "utf-8"):
        try:
            canonical_name = codecs.lookup(encoding).name
        except LookupError:
            continue
        if canonical_name not in encodings:
            encodings.append(canonical_name)
    return tuple(encodings)


def _decode_scheduler_bytes(data, encodings):
    for encoding in encodings:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode(encodings[0], errors="replace")


def decode_scheduler_output(data):
    """Decode scheduler bytes without failing on arbitrary job-name bytes."""

    if isinstance(data, str):
        return data
    return _decode_scheduler_bytes(data, _scheduler_output_encodings())


def _decode_scheduler_lines(binary_lines):
    encodings = _scheduler_output_encodings()
    if len(encodings) == 1:
        encoding = encodings[0]
        return [line.decode(encoding, errors="replace").rstrip("\r\n") for line in binary_lines]
    return [_decode_scheduler_bytes(line, encodings).rstrip("\r\n") for line in binary_lines]


def _read_scheduler_error(stderr):
    stderr.seek(0)
    data = stderr.read(STDERR_DETAIL_LIMIT_BYTES + 1)
    truncated = len(data) > STDERR_DETAIL_LIMIT_BYTES
    detail = decode_scheduler_output(data[:STDERR_DETAIL_LIMIT_BYTES]).strip()
    if truncated:
        detail = f"{detail}\n[stderr truncated]" if detail else "[stderr truncated]"
    return detail


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
                return _decode_scheduler_lines(handle)
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
    # Spool large stdout streams to a temporary file. This avoids simultaneously
    # retaining the raw bytes, a fully decoded string, and the returned line list.
    with (
        tempfile.SpooledTemporaryFile(max_size=STDOUT_SPOOL_MEMORY_LIMIT_BYTES) as stdout,
        tempfile.SpooledTemporaryFile(max_size=STDOUT_SPOOL_MEMORY_LIMIT_BYTES) as stderr,
    ):
        try:
            # The parsed argv list is executed directly; shell=True is never used.
            command_out = subprocess.run(  # nosec B603
                command,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            if allow_failure:
                return None
            summary = f"Timed out running {command_name} after {timeout:g}s: {command_str}"
            raise KFBatchCommandError(
                _format_error_message(summary, quiet=quiet_failure)
            ) from error
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
            command_stderr = _read_scheduler_error(stderr)
            summary = f"Failed to run {command_name}: {command_str}"
            raise KFBatchCommandError(
                _format_error_message(summary, command_stderr, quiet=quiet_failure)
            )
        stdout.seek(0)
        return _decode_scheduler_lines(stdout)
