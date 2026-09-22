import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_codeql_steps_share_one_pinned_revision():
    workflow = (REPO_ROOT / ".github/workflows/codeql.yml").read_text()
    revisions = re.findall(r"uses: github/codeql-action/(?:init|analyze)@([0-9a-f]{40})", workflow)
    assert len(revisions) == 2
    assert len(set(revisions)) == 1


def _workflow(name):
    import yaml

    return yaml.load((REPO_ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)


def test_release_requires_full_validation_at_the_tag_revision():
    release = _workflow("release.yml")
    checks = _workflow("tests.yml")
    assert "workflow_call" in checks["on"]
    assert release["jobs"]["validate"]["uses"] == "./.github/workflows/tests.yml"
    assert release["jobs"]["release"]["needs"] == "validate"
    assert checks["jobs"]["test"]["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    ]
    assert {"minimum-pandas", "quality-and-package"} <= checks["jobs"].keys()
    for job in checks["jobs"].values():
        checkout = next(
            step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
        )
        assert "ref" not in checkout.get("with", {})  # Default is the caller's commit.
    assert release["concurrency"]["cancel-in-progress"] == "false"


def test_validation_event_routes_and_cancellation_do_not_skip_release_checks():
    checks = _workflow("tests.yml")
    # All branch pushes and PR merge refs retain their existing coverage. Tag
    # validation is supplied exactly once through release's reusable workflow.
    assert checks["on"]["push"]["branches"] == ["**"]
    assert "pull_request" in checks["on"]
    assert _workflow("release.yml")["on"]["push"]["tags"] == ["v*"]
    condition = checks["concurrency"]["cancel-in-progress"]
    assert condition == "${{ github.event_name == 'pull_request' || github.ref_type == 'branch' }}"
    assert "github.event_name" in checks["concurrency"]["group"]
    assert checks["permissions"] == {"contents": "read"}
