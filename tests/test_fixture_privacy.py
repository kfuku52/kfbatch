import ipaddress
import json
import pathlib
import re

from kfbatch.stat import get_user_df

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"


def _fixture_text_files():
    return [path for path in FIXTURE_ROOT.rglob("*") if path.is_file()]


def test_scheduler_fixtures_contain_no_network_addresses_emails_or_local_paths():
    for path in _fixture_text_files():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            address = ipaddress.ip_address(match.group(0))
            raise AssertionError(f"network address in {path}: {address}")
        for token in re.findall(r"[0-9A-Fa-f:.]{2,}", text):
            if ":" not in token:
                continue
            try:
                address = ipaddress.ip_address(token.strip("[](),"))
            except ValueError:
                continue
            raise AssertionError(f"network address in {path}: {address}")
        assert not re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
            text,
        ), f"email address in {path}"
        assert not re.search(
            r"(?<![A-Za-z0-9])/(?:Users|home|etc|opt|private|tmp|var)/",
            text,
        ), f"local absolute path in {path}"


def test_scheduler_identity_fields_use_synthetic_names():
    for path in (FIXTURE_ROOT / "slurm").glob("squeue*.txt"):
        text = path.read_text(encoding="utf-8").replace("\\t", "\t")
        for line in text.splitlines():
            fields = line.split("\t")
            assert len(fields) >= 4
            assert re.fullmatch(r"(?:analysis|job)_[a-z]", fields[2])
            assert re.fullmatch(r"(?:user_[a-z]|current_user)", fields[3])
            if len(fields) >= 12:
                assert re.fullmatch(r"account_[a-z]", fields[4])

    share_path = FIXTURE_ROOT / "slurm" / "sshare.txt"
    for line in share_path.read_text(encoding="utf-8").splitlines()[1:]:
        account, user, *_ = line.split("|")
        assert re.fullmatch(r"account_[a-z]", account)
        assert user == "" or re.fullmatch(r"(?:user_[a-z]|current_user)", user)

    node_text = (FIXTURE_ROOT / "slurm" / "nodes.txt").read_text(encoding="utf-8")
    for node_name in re.findall(r"\bNodeName=([^\s]+)", node_text):
        assert re.fullmatch(r"(?:compute|login)[0-9]+", node_name)

    reservation_text = (FIXTURE_ROOT / "slurm" / "reservations.txt").read_text(encoding="utf-8")
    for users in re.findall(r"\bUsers=([^\s]+)", reservation_text):
        assert all(
            re.fullmatch(r"(?:user_[a-z]|current_user|ALL)", user) for user in users.split(",")
        )

    for path in (FIXTURE_ROOT / "age").glob("qstat*.txt"):
        jobs = get_user_df(path.read_text(encoding="utf-8").splitlines())
        for user in jobs["user"].dropna().astype(str):
            assert re.fullmatch(r"(?:user_[a-z]|current_user)", user)

    json_data = json.loads(
        (FIXTURE_ROOT / "age" / "qstat_all_users.json").read_text(encoding="utf-8")
    )

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "JB_owner":
                    assert re.fullmatch(r"(?:user_[a-z]|current_user)", str(item))
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(json_data)


def test_live_capture_paths_are_not_checked_in():
    capture_name = re.compile(
        r"^(?:qfree|qstat|scontrol|sprio|squeue|sshare)(?:[-_.].*)?\.txt$",
        re.IGNORECASE,
    )
    forbidden_dirs = {"capture", "captures", "data", "dump", "scheduler-data"}
    forbidden = []
    for path in REPO_ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or FIXTURE_ROOT in path.parents
            or path.name == ".DS_Store"
        ):
            continue
        if capture_name.fullmatch(path.name) or forbidden_dirs.intersection(path.parts):
            forbidden.append(path)
    assert not forbidden
