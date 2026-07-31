"""Safe execution and decoding of scheduler commands."""

from __future__ import annotations

import codecs
import locale
import math
import os
import pathlib
import shlex
import signal
import stat
import subprocess  # nosec B404
import threading
import time

from kfbatch.errors import KFBatchCommandError

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
MAX_STDOUT_BYTES = 128 * 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024 * 1024
MAX_OUTPUT_LINE_BYTES = 4 * 1024 * 1024
STDERR_DETAIL_LIMIT_BYTES = 64 * 1024
PROCESS_TERMINATION_GRACE_SECONDS = 0.5
PROCESS_POLL_INTERVAL_SECONDS = 0.02


class _BoundedCapture:
    def __init__(self, limit):
        self.limit = int(limit)
        self.data = bytearray()
        self.total = 0
        self.exceeded = False

    def append(self, chunk):
        self.total += len(chunk)
        remaining = max((self.limit + 1) - len(self.data), 0)
        if remaining:
            self.data.extend(chunk[:remaining])
        if self.total > self.limit:
            self.exceeded = True


def _capture_stream(stream, capture):
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            capture.append(chunk)
    finally:
        stream.close()


def _format_error_message(summary, detail="", quiet=False):
    if quiet or not str(detail).strip():
        return summary
    return f"{summary}\n{detail}"


def _scheduler_output_encodings():
    """Return valid, de-duplicated decoder names in preference order."""

    preferred = locale.getpreferredencoding(False) or "utf-8"
    encodings = []
    for encoding in ("utf-8", preferred):
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


def _decode_bounded_lines(data, command_name):
    raw_lines = data.splitlines()
    if any(len(line) > MAX_OUTPUT_LINE_BYTES for line in raw_lines):
        raise KFBatchCommandError(
            f"Failed to read {command_name}: an output line exceeded "
            f"{MAX_OUTPUT_LINE_BYTES} bytes.",
            command_name=command_name,
            output_limited=True,
        )
    return _decode_scheduler_lines(raw_lines)


def _read_scheduler_error(data):
    truncated = len(data) > STDERR_DETAIL_LIMIT_BYTES
    detail = decode_scheduler_output(data[:STDERR_DETAIL_LIMIT_BYTES]).strip()
    if truncated:
        detail = f"{detail}\n[stderr truncated]" if detail else "[stderr truncated]"
    return detail


def _safe_label(value):
    return "".join(
        character if character.isprintable() and character not in "\r\n\x1b" else "?"
        for character in str(value)
    )


def _command_summary(command_name, command):
    executable = pathlib.Path(command[0]).name if command else "<empty>"
    return f"{_safe_label(command_name)} ({_safe_label(executable)})"


def _subprocess_environment(command):
    environment = os.environ.copy()
    executable = pathlib.Path(command[0]).name if command else ""
    if executable == "squeue":
        for key in list(environment):
            if key.startswith("SQUEUE_"):
                environment.pop(key, None)
    return environment


def _signal_process_group(process, sig):
    if os.name == "posix":
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass
        return
    if process.poll() is None:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()


def _terminate_process_group(process):
    started = time.monotonic()
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    if os.name == "posix":
        remaining = PROCESS_TERMINATION_GRACE_SECONDS - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(remaining)
    _signal_process_group(process, signal.SIGKILL)
    if process.poll() is None:
        process.wait()


def _read_example_file(example_file, command_name):
    path = pathlib.Path(example_file)
    try:
        open_flags = os.O_RDONLY
        open_flags |= getattr(os, "O_CLOEXEC", 0)
        open_flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, open_flags)
        with os.fdopen(descriptor, "rb") as handle:
            file_stat = os.fstat(handle.fileno())
            if not stat.S_ISREG(file_stat.st_mode):
                raise KFBatchCommandError(
                    f"Failed to read example file for {_safe_label(command_name)}: "
                    "only regular files are accepted.",
                    command_name=command_name,
                )
            if file_stat.st_size > MAX_STDOUT_BYTES:
                raise KFBatchCommandError(
                    f"Failed to read example file for {_safe_label(command_name)}: "
                    f"file exceeds {MAX_STDOUT_BYTES} bytes.",
                    command_name=command_name,
                    output_limited=True,
                )
            data = handle.read(MAX_STDOUT_BYTES + 1)
    except KFBatchCommandError:
        raise
    except OSError as error:
        summary = (
            f"Failed to read example file for {_safe_label(command_name)}: {_safe_label(path)}"
        )
        raise KFBatchCommandError(summary, command_name=command_name) from error
    if len(data) > MAX_STDOUT_BYTES:
        raise KFBatchCommandError(
            f"Failed to read example file for {_safe_label(command_name)}: "
            f"file exceeds {MAX_STDOUT_BYTES} bytes.",
            command_name=command_name,
            output_limited=True,
        )
    return _decode_bounded_lines(data, command_name)


def _run_command(command, command_name, timeout):
    popen_kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "env": _subprocess_environment(command),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    elif hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    process = subprocess.Popen(command, **popen_kwargs)  # nosec B603
    stdout_capture = _BoundedCapture(MAX_STDOUT_BYTES)
    stderr_capture = _BoundedCapture(MAX_STDERR_BYTES)
    stdout_thread = threading.Thread(
        target=_capture_stream,
        args=(process.stdout, stdout_capture),
        name="kfbatch-stdout",
    )
    stderr_thread = threading.Thread(
        target=_capture_stream,
        args=(process.stderr, stderr_capture),
        name="kfbatch-stderr",
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = None if timeout is None else time.monotonic() + timeout
    timed_out = False
    output_limited = False
    while process.poll() is None:
        if stdout_capture.exceeded or stderr_capture.exceeded:
            output_limited = True
            _terminate_process_group(process)
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            _terminate_process_group(process)
            break
        time.sleep(PROCESS_POLL_INTERVAL_SECONDS)
    stdout_thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    stderr_thread.join(timeout=PROCESS_TERMINATION_GRACE_SECONDS)
    if stdout_thread.is_alive() or stderr_thread.is_alive():
        # A descendant inherited the capture pipes. End the command's whole
        # session so background descendants cannot outlive the invocation.
        _terminate_process_group(process)
        stdout_thread.join()
        stderr_thread.join()
    output_limited = output_limited or stdout_capture.exceeded or stderr_capture.exceeded
    return (
        process.returncode,
        bytes(stdout_capture.data),
        bytes(stderr_capture.data),
        timed_out,
        output_limited,
    )


def _command_result_lines(
    *,
    command,
    command_name,
    timeout,
    returncode,
    stdout,
    stderr,
    timed_out,
    output_limited,
    allow_failure,
    quiet_failure,
):
    if timed_out:
        if allow_failure:
            return None
        summary = f"Timed out running {_command_summary(command_name, command)} after {timeout:g}s."
        raise KFBatchCommandError(
            _format_error_message(
                summary,
                _read_scheduler_error(stderr),
                quiet=quiet_failure,
            ),
            command_name=command_name,
            argv=command,
            returncode=returncode,
            timed_out=True,
        )
    if output_limited:
        if allow_failure:
            return None
        summary = (
            f"Failed to run {_command_summary(command_name, command)}: "
            "captured output exceeded its safety limit."
        )
        raise KFBatchCommandError(
            _format_error_message(
                summary,
                _read_scheduler_error(stderr),
                quiet=quiet_failure,
            ),
            command_name=command_name,
            argv=command,
            returncode=returncode,
            output_limited=True,
        )
    if returncode != 0:
        if allow_failure:
            return None
        summary = (
            f"Failed to run {_command_summary(command_name, command)} (exit status {returncode})."
        )
        raise KFBatchCommandError(
            _format_error_message(
                summary,
                _read_scheduler_error(stderr),
                quiet=quiet_failure,
            ),
            command_name=command_name,
            argv=command,
            returncode=returncode,
        )
    return _decode_bounded_lines(stdout, command_name)


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
            return _read_example_file(example_file, command_name)
        except KFBatchCommandError:
            if allow_failure:
                return None
            raise
    try:
        command = shlex.split(command_str)
    except ValueError as error:
        if allow_failure:
            return None
        summary = f"Failed to parse {_safe_label(command_name)}."
        raise KFBatchCommandError(
            _format_error_message(summary, str(error), quiet=quiet_failure),
            command_name=command_name,
        ) from error
    if not command:
        if allow_failure:
            return None
        summary = f"Failed to run {_safe_label(command_name)}: command is empty."
        raise KFBatchCommandError(
            _format_error_message(summary, quiet=quiet_failure),
            command_name=command_name,
        )
    try:
        timeout_value = 0.0 if timeout_seconds is None else float(timeout_seconds)
    except (TypeError, ValueError) as error:
        if allow_failure:
            return None
        raise KFBatchCommandError(
            f"Failed to run {_safe_label(command_name)}: timeout must be a finite "
            "non-negative number.",
            command_name=command_name,
        ) from error
    if not math.isfinite(timeout_value) or timeout_value < 0:
        if allow_failure:
            return None
        raise KFBatchCommandError(
            f"Failed to run {_safe_label(command_name)}: timeout must be a finite "
            "non-negative number.",
            command_name=command_name,
        )
    timeout = timeout_value or None
    try:
        returncode, stdout, stderr, timed_out, output_limited = _run_command(
            command,
            command_name,
            timeout,
        )
    except OSError as error:
        if allow_failure:
            return None
        summary = f"Failed to run {_command_summary(command_name, command)}."
        raise KFBatchCommandError(
            _format_error_message(summary, str(error), quiet=quiet_failure),
            command_name=command_name,
            argv=command,
        ) from error
    return _command_result_lines(
        command=command,
        command_name=command_name,
        timeout=timeout,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        output_limited=output_limited,
        allow_failure=allow_failure,
        quiet_failure=quiet_failure,
    )
