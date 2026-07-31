"""Filesystem quota collection and compact reporting."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any

from kfbatch.command import DEFAULT_COMMAND_TIMEOUT_SECONDS, get_command_stdout_lines
from kfbatch.errors import KFBatchCommandError, KFBatchUsageError

pwd: Any
try:
    import pwd
except ImportError:  # pragma: no cover - quota commands are normally used on POSIX hosts
    pwd = None


@dataclass(frozen=True)
class QuotaRecord:
    """Provider-independent quota values.

    Byte limits and inode/file limits are ``None`` when the provider reports no
    limit or does not expose that value.
    """

    provider: str
    filesystem: str
    scope: str
    owner: str
    bytes_used: int
    bytes_soft: int | None
    bytes_hard: int | None
    files_used: int | None
    files_soft: int | None
    files_hard: int | None
    grace: str = ""


_SECTION_RE = re.compile(
    r"^Disk quotas for\s+(?P<kind>usr|user|grp|group)\s+(?P<owner>\S+)",
    flags=re.IGNORECASE,
)
_BYTE_FACTORS = {
    "": 1024,
    "K": 1024,
    "KI": 1024,
    "M": 1024**2,
    "MI": 1024**2,
    "G": 1024**3,
    "GI": 1024**3,
    "T": 1024**4,
    "TI": 1024**4,
    "P": 1024**5,
    "PI": 1024**5,
}
_QUOTA_SPACE_HEADER_FACTORS = {
    "blocks": 1024,
    "kbytes": 1024,
    "mbytes": 1024**2,
    "gbytes": 1024**3,
    "tbytes": 1024**4,
    "pbytes": 1024**5,
    "space": 1024,
}
_QUOTA_FILE_HEADER_FACTORS = {
    "files": 1,
    "kfiles": 1000,
    "mfiles": 1000**2,
    "gfiles": 1000**3,
}
_UNLIMITED = {"", "-", "none", "unlimited", "inf", "infinite", "0"}


def add_quota_arguments(parser):
    parser.add_argument(
        "--scope",
        choices=["overview", "self", "group"],
        default="overview",
        help="default=%(default)s: Show personal, group, or both quota scopes.",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "lfsq", "posix", "lustre", "custom"],
        default="auto",
        help="default=%(default)s: Quota command/output provider.",
    )
    parser.add_argument(
        "--filesystem",
        metavar="NAME",
        default="all",
        help=(
            "default=%(default)s: Filesystem filter. Use home for /home, all for "
            "every record, or an exact filesystem/path name."
        ),
    )
    parser.add_argument(
        "--current-user",
        "--current_user",
        dest="current_user",
        metavar="NAME",
        default="",
        help="default=effective local user: User record to display.",
    )
    parser.add_argument(
        "--group-id",
        "--group_id",
        dest="group_id",
        metavar="NAME",
        default="",
        help="default=all reported groups: Group quota owner to display.",
    )
    parser.add_argument(
        "--quota-command",
        "--quota_command",
        dest="quota_command",
        metavar="COMMAND",
        default="",
        help="Override the selected provider command.",
    )
    parser.add_argument(
        "--quota-example-file",
        "--quota_example_file",
        dest="quota_example_file",
        metavar="PATH",
        default="",
        help="Read captured quota stdout from a synthetic fixture instead of running a command.",
    )
    parser.add_argument(
        "--command-timeout",
        "--command_timeout",
        dest="command_timeout",
        metavar="SECONDS",
        type=_parse_nonnegative_float,
        default=DEFAULT_COMMAND_TIMEOUT_SECONDS,
        help="default=%(default)s: Quota command timeout; 0 disables it.",
    )


def _parse_nonnegative_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("Expected a finite non-negative number.") from error
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Expected a finite non-negative number.")
    return number


def _clean_numeric_token(value):
    return str(value).strip().rstrip("*")


def _parse_bytes(value, *, unlimited_zero=False, default_factor=1024):
    text = _clean_numeric_token(value).replace(",", "")
    if text.lower() in _UNLIMITED and (unlimited_zero or text != "0"):
        return None
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]i?|)[Bb]?", text, re.IGNORECASE)
    if match is None:
        return None
    unit = match.group(2).upper()
    factor = default_factor if unit == "" else _BYTE_FACTORS.get(unit)
    if factor is None:
        return None
    return int(float(match.group(1)) * factor)


def _parse_count(value, *, unlimited_zero=False):
    text = _clean_numeric_token(value).replace(",", "")
    if text.lower() in _UNLIMITED and (unlimited_zero or text != "0"):
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _quota_header_factors(line):
    fields = re.split(r"\s+", str(line).strip().lower())
    if len(fields) < 6 or fields[0] != "filesystem":
        return None
    space_factor = _QUOTA_SPACE_HEADER_FACTORS.get(fields[1])
    file_factor = _QUOTA_FILE_HEADER_FACTORS.get(fields[5])
    if space_factor is None or file_factor is None:
        return None
    return space_factor, file_factor


def _scaled_count(value, factor, *, unlimited_zero=False):
    count = _parse_count(value, unlimited_zero=unlimited_zero)
    return None if count is None else count * factor


def _parse_standard_quota(lines, provider):
    records = []
    section_scope = ""
    section_owner = ""
    saw_header = False
    pending_filesystem = ""
    space_factor = 1024
    file_factor = 1
    for raw_line in lines:
        line = str(raw_line).strip()
        match = _SECTION_RE.match(line)
        if match is not None:
            section_scope = "self" if match.group("kind").lower() in {"usr", "user"} else "group"
            section_owner = match.group("owner").rstrip(":")
            saw_header = False
            pending_filesystem = ""
            continue
        if section_scope == "" or line == "":
            continue
        header_factors = _quota_header_factors(line)
        if header_factors is not None:
            space_factor, file_factor = header_factors
            saw_header = True
            continue
        if not saw_header:
            continue
        items = re.split(r"\s+", line)
        if len(items) == 1 and ("/" in items[0] or ":" in items[0]):
            pending_filesystem = items[0]
            continue
        if pending_filesystem:
            items.insert(0, pending_filesystem)
            pending_filesystem = ""
        if len(items) < 8:
            continue
        filesystem = items[0]
        values = items[1:]
        bytes_used = _parse_bytes(values[0], default_factor=space_factor)
        files_used = _scaled_count(values[4], file_factor)
        if bytes_used is None:
            continue
        records.append(
            QuotaRecord(
                provider=provider,
                filesystem=filesystem,
                scope=section_scope,
                owner=section_owner,
                bytes_used=bytes_used,
                bytes_soft=_parse_bytes(
                    values[1],
                    unlimited_zero=True,
                    default_factor=space_factor,
                ),
                bytes_hard=_parse_bytes(
                    values[2],
                    unlimited_zero=True,
                    default_factor=space_factor,
                ),
                files_used=files_used,
                files_soft=_scaled_count(
                    values[5],
                    file_factor,
                    unlimited_zero=True,
                ),
                files_hard=_scaled_count(
                    values[6],
                    file_factor,
                    unlimited_zero=True,
                ),
                grace="" if values[3] in {"-", "none"} else values[3],
            )
        )
    return records


def _normalized_columns(line):
    separator = "|" if "|" in line else None
    fields = [field.strip().lower() for field in line.split(separator)]
    aliases = {
        "used": "bytes_used",
        "soft": "bytes_soft",
        "hard": "bytes_hard",
        "quota": "bytes_soft",
        "limit": "bytes_hard",
        "files": "files_used",
        "files_quota": "files_soft",
        "files_limit": "files_hard",
    }
    normalized = [aliases.get(field, field) for field in fields]
    required = {"scope", "owner", "filesystem", "bytes_used"}
    return separator, normalized if required.issubset(normalized) else []


def _parse_normalized_quota(lines, provider):
    records = []
    columns: list[str] = []
    separator = None
    for raw_line in lines:
        line = str(raw_line).strip()
        if line == "" or line.startswith("#"):
            continue
        if not columns:
            separator, columns = _normalized_columns(line)
            continue
        values = [value.strip() for value in line.split(separator)]
        if len(values) != len(columns):
            continue
        row = dict(zip(columns, values, strict=True))
        scope = row["scope"].lower()
        if scope in {"user", "usr", "personal"}:
            scope = "self"
        elif scope in {"grp", "project", "account"}:
            scope = "group"
        if scope not in {"self", "group"}:
            continue
        bytes_used = _parse_bytes(row["bytes_used"])
        if bytes_used is None:
            continue
        records.append(
            QuotaRecord(
                provider=provider,
                filesystem=row["filesystem"],
                scope=scope,
                owner=row["owner"],
                bytes_used=bytes_used,
                bytes_soft=_parse_bytes(row.get("bytes_soft", ""), unlimited_zero=True),
                bytes_hard=_parse_bytes(row.get("bytes_hard", ""), unlimited_zero=True),
                files_used=_parse_count(row.get("files_used", "")),
                files_soft=_parse_count(row.get("files_soft", ""), unlimited_zero=True),
                files_hard=_parse_count(row.get("files_hard", ""), unlimited_zero=True),
                grace=row.get("grace", ""),
            )
        )
    return records


def parse_quota_lines(lines, provider):
    """Parse standard ``quota``/``lfs quota`` or normalized wrapper output."""

    records = _parse_standard_quota(lines, provider)
    if not records:
        records = _parse_normalized_quota(lines, provider)
    return records


def _effective_user(explicit):
    if str(explicit or "").strip():
        return str(explicit).strip()
    if pwd is None or not hasattr(os, "geteuid"):
        return ""
    try:
        return pwd.getpwuid(os.geteuid()).pw_name.strip()
    except (KeyError, OSError):
        return ""


def _provider_candidates(args):
    if args.quota_command:
        label = args.provider if args.provider != "auto" else "custom"
        return [(label, args.quota_command)]
    if args.provider == "lfsq":
        return [("lfsq", "lfsq")]
    if args.provider == "posix":
        return [("posix", "quota -s -ug")]
    if args.provider in {"lustre", "custom"}:
        raise KFBatchUsageError(f"--provider {args.provider} requires --quota-command.")
    candidates = []
    if shutil.which("lfsq"):
        candidates.append(("lfsq", "lfsq"))
    if shutil.which("quota"):
        candidates.append(("posix", "quota -s -ug"))
    return candidates


def _collect_records(args):
    if args.quota_example_file:
        provider = args.provider if args.provider != "auto" else "captured"
        lines = get_command_stdout_lines(
            command_str="",
            example_file=args.quota_example_file,
            command_name="--quota-example-file",
            timeout_seconds=args.command_timeout,
        )
        records = parse_quota_lines(lines, provider)
        if not records:
            raise KFBatchCommandError(
                "Quota fixture contained no recognized standard or normalized quota rows."
            )
        return records

    candidates = _provider_candidates(args)
    if not candidates:
        raise KFBatchCommandError(
            "No supported quota command was found. Install/use lfsq or quota, or provide "
            "--quota-command/--quota-example-file."
        )
    completed_providers = set()
    for provider, command in candidates:
        lines = get_command_stdout_lines(
            command_str=command,
            allow_failure=True,
            command_name="--quota-command",
            quiet_failure=True,
            timeout_seconds=args.command_timeout,
        )
        if lines is None:
            continue
        completed_providers.add(provider)
        records = parse_quota_lines(lines, provider)
        if records:
            return records
    if any(provider == "lfsq" for provider, _command in candidates):
        if "lfsq" in completed_providers:
            raise KFBatchCommandError(
                "lfsq completed successfully but returned no recognized quota rows."
            )
        raise KFBatchCommandError(
            "lfsq did not return parseable quota data. On SHIROKANE, run qlogin first and "
            "then retry `kfbatch quota`; kfbatch never starts qlogin automatically."
        )
    raise KFBatchCommandError("The quota command returned no recognized quota rows.")


def _matches_filesystem(record, requested):
    requested = str(requested or "all").strip()
    if requested in {"", "all"}:
        return True
    if requested == "home":
        return record.filesystem == "home" or record.filesystem.startswith("/home/")
    return requested in {record.filesystem, os.path.basename(record.filesystem.rstrip("/"))}


def _filter_records(records, args):
    current_user = _effective_user(args.current_user)
    selected = []
    for record in records:
        if not _matches_filesystem(record, args.filesystem):
            continue
        if record.scope == "self":
            if args.scope == "group" or (current_user and record.owner != current_user):
                continue
        elif args.scope == "self":
            continue
        if record.scope == "group" and args.group_id and record.owner != args.group_id:
            continue
        selected.append(record)
    return selected


def _format_bytes(value):
    if value is None:
        return "-"
    number = float(value)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    unit_index = 0
    while number >= 1024 and unit_index < len(units) - 1:
        number /= 1024
        unit_index += 1
    precision = 0 if number >= 100 or unit_index == 0 else 1
    return f"{number:.{precision}f}{units[unit_index]}"


def _format_count(value):
    return "-" if value is None else f"{value:,}"


def print_quota_records(records):
    rows = []
    for record in records:
        rows.append(
            {
                "scope": record.scope,
                "owner": record.owner,
                "filesystem": record.filesystem,
                "space(used/soft/hard)": "/".join(
                    [
                        _format_bytes(record.bytes_used),
                        _format_bytes(record.bytes_soft),
                        _format_bytes(record.bytes_hard),
                    ]
                ),
                "files(used/soft/hard)": "/".join(
                    [
                        _format_count(record.files_used),
                        _format_count(record.files_soft),
                        _format_count(record.files_hard),
                    ]
                ),
                "grace": record.grace or "-",
                "provider": record.provider,
            }
        )
    columns = [
        "scope",
        "owner",
        "filesystem",
        "space(used/soft/hard)",
        "files(used/soft/hard)",
        "grace",
        "provider",
    ]
    widths = {
        column: max([len(column)] + [len(str(row[column])) for row in rows]) for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print("  ".join(str(row[column]).ljust(widths[column]) for column in columns))
    if any(record.scope == "group" for record in records):
        print("")
        print(
            "note: group usage and limits are shared by all group members; they are not personal usage."
        )


def quota_main(args):
    records = _filter_records(_collect_records(args), args)
    if not records:
        raise KFBatchCommandError(
            "No quota records matched the requested scope, owner, and filesystem filters."
        )
    print_quota_records(records)
