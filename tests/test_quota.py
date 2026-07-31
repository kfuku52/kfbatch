from argparse import Namespace

import pytest

from kfbatch.errors import KFBatchCommandError
from kfbatch.quota import parse_quota_lines, quota_main


def _args(**overrides):
    values = {
        "scope": "overview",
        "provider": "auto",
        "filesystem": "all",
        "current_user": "user_a",
        "group_id": "",
        "quota_command": "",
        "quota_example_file": "tests/fixtures/quota/normalized.txt",
        "command_timeout": 60.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_parse_standard_lustre_user_and_group_quota():
    records = parse_quota_lines(
        [
            "Disk quotas for usr user_a (uid 1001):",
            "Filesystem kbytes quota limit grace files quota limit grace",
            "home_user_a 1048576 2097152 3145728 - 120 200 300 -",
            "Disk quotas for grp group_a (gid 2001):",
            "Filesystem kbytes quota limit grace files quota limit grace",
            "home_group_a 4194304 8388608 16777216 3days 400 800 1600 -",
        ],
        "lustre",
    )
    assert [(record.scope, record.owner) for record in records] == [
        ("self", "user_a"),
        ("group", "group_a"),
    ]
    assert records[0].bytes_used == 1024**3
    assert records[1].bytes_hard == 16 * 1024**3
    assert records[1].files_hard == 1600


def test_quota_main_prints_personal_and_shared_group_rows(capsys):
    quota_main(_args())
    out = capsys.readouterr().out
    assert "self" in out
    assert "user_a" in out
    assert "group_a" in out
    assert "71.2TiB" in out
    assert "shared by all group members" in out


def test_quota_main_filters_group_owner(capsys):
    quota_main(_args(scope="group", group_id="group_a"))
    out = capsys.readouterr().out
    assert "group_a" in out
    assert "user_a" not in out


def test_quota_main_rejects_empty_filter_result():
    with pytest.raises(KFBatchCommandError, match="No quota records matched"):
        quota_main(_args(scope="group", group_id="missing_group"))


def test_quota_main_rejects_unrecognized_fixture(tmp_path):
    fixture = tmp_path / "quota.txt"
    fixture.write_text("unrecognized output\n", encoding="utf-8")
    with pytest.raises(KFBatchCommandError, match="no recognized"):
        quota_main(_args(quota_example_file=str(fixture)))
