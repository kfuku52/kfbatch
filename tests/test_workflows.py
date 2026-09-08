import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_codeql_steps_share_one_pinned_revision():
    workflow = (REPO_ROOT / ".github/workflows/codeql.yml").read_text()
    revisions = re.findall(r"uses: github/codeql-action/(?:init|analyze)@([0-9a-f]{40})", workflow)
    assert len(revisions) == 2
    assert len(set(revisions)) == 1


def test_dependabot_groups_codeql_updates():
    configuration = (REPO_ROOT / ".github/dependabot.yml").read_text()
    assert re.search(
        r"(?m)^    groups:\n      codeql-action:\n        patterns:\n"
        r"          - github/codeql-action/\*\s*$",
        configuration,
    )
