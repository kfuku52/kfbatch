import ipaddress
import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures"


def _fixture_text_files():
    return [path for path in FIXTURE_ROOT.rglob("*") if path.is_file()]


def test_scheduler_fixtures_contain_no_private_ip_addresses():
    for path in _fixture_text_files():
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text):
            address = ipaddress.ip_address(match.group(0))
            assert not address.is_private, f"private address in {path}: {address}"


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


def test_live_capture_paths_are_not_checked_in():
    forbidden_root_globs = [
        "squeue*.txt",
        "sshare*.txt",
        "scontrol*.txt",
    ]
    forbidden = []
    for pattern in forbidden_root_globs:
        forbidden.extend(REPO_ROOT.glob(pattern))
    if (REPO_ROOT / "data").exists():
        forbidden.extend(path for path in (REPO_ROOT / "data").rglob("*") if path.is_file())
    assert not [path for path in forbidden if path.exists() and path.name != ".DS_Store"]
