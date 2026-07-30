import ipaddress
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"


def test_scheduler_fixtures_contain_no_private_ip_addresses():
    for path in FIXTURE_ROOT.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            address = ipaddress.ip_address(match.group(0))
            assert not address.is_private, f"private address in {path}: {address}"


def test_legacy_live_capture_files_are_not_checked_in():
    removed_paths = [
        REPO_ROOT / "squeue.txt",
        REPO_ROOT / "squeue_notrunc.txt",
        REPO_ROOT / "scontrol_show_node_o.txt",
        REPO_ROOT / "scontrol_show_partition_o.txt",
        REPO_ROOT / "data" / "qstat1" / "full.tsv",
        REPO_ROOT / "data" / "qstat1" / "qstatF.txt",
        REPO_ROOT / "data" / "qstat2" / "qstatF.txt",
        REPO_ROOT / "data" / "qstat3" / "qstatF.txt",
        REPO_ROOT / "data" / "qstat4" / "qstatF.txt",
    ]
    assert not [path for path in removed_paths if path.exists()]
