import os
import pathlib
import re
import subprocess

import pytest

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
    assert release["jobs"]["validate"]["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert release["jobs"]["release"]["needs"] == "validate"
    versions = checks["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    quality = checks["jobs"]["quality-and-package"]
    quality_python = next(
        step["with"]["python-version"]
        for step in quality["steps"]
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    assert sorted(versions + [quality_python]) == ["3.10", "3.11", "3.12", "3.13", "3.14"]
    assert any("--cov-branch" in step.get("run", "") for step in quality["steps"])
    assert {"minimum-pandas", "quality-and-package"} <= checks["jobs"].keys()
    for name in ("test", "minimum-pandas", "quality-and-package"):
        job = checks["jobs"][name]
        assert job["needs"] == "validation-route"
        assert job["if"] == "needs.validation-route.outputs.run == 'true'"
        checkout = next(
            step for step in job["steps"] if step.get("uses", "").startswith("actions/checkout@")
        )
        assert "ref" not in checkout.get("with", {})  # Default is the caller's commit.
    assert release["concurrency"]["cancel-in-progress"] == "false"


def test_validation_event_routes_and_cancellation_do_not_skip_release_checks():
    checks = _workflow("tests.yml")
    # Branches without a matching open PR, PR merge refs and release tags all
    # receive full validation. A matching open PR replaces duplicate push work.
    assert checks["on"]["push"]["branches"] == ["**"]
    assert "pull_request" in checks["on"]
    assert _workflow("release.yml")["on"]["push"]["tags"] == ["v*"]
    condition = checks["concurrency"]["cancel-in-progress"]
    assert condition == "${{ github.event_name == 'pull_request' || github.ref_type == 'branch' }}"
    assert "github.event_name" in checks["concurrency"]["group"]
    assert checks["permissions"] == {"contents": "read"}


@pytest.mark.parametrize(
    ("event", "ref_type", "branch", "pr_heads", "expected", "queried"),
    [
        ("push", "branch", "main", "", "true", False),
        ("push", "branch", "topic", "", "true", True),
        ("push", "branch", "topic", "owner/repo:head-sha", "false", True),
        ("push", "branch", "topic", "owner/repo:older-sha", "true", True),
        ("push", "branch", "topic", "fork/repo:head-sha", "true", True),
        ("pull_request", "branch", "1/merge", "owner/repo:head-sha", "true", False),
        ("push", "tag", "v0.4.15", "owner/repo:head-sha", "true", False),
        ("workflow_call", "tag", "v0.4.15", "", "true", False),
    ],
)
def test_validation_route(tmp_path, event, ref_type, branch, pr_heads, expected, queried):
    route = _workflow("tests.yml")["jobs"]["validation-route"]
    assert route["permissions"] == {"contents": "read", "pull-requests": "read"}
    fake_gh = tmp_path / "gh"
    fake_gh.write_text('#!/bin/sh\nprintf "%s\\n" "$PR_HEADS"\ntouch "$GH_CALLED"\n')
    fake_gh.chmod(0o755)
    output = tmp_path / "outputs"
    called = tmp_path / "called"
    environment = dict(
        os.environ,
        PATH=f"{tmp_path}:{os.environ['PATH']}",
        EVENT_NAME=event,
        REF_TYPE=ref_type,
        BRANCH=branch,
        DEFAULT_BRANCH="main",
        GITHUB_REPOSITORY="owner/repo",
        GITHUB_REPOSITORY_OWNER="owner",
        GITHUB_SHA="head-sha",
        GITHUB_OUTPUT=str(output),
        PR_HEADS=pr_heads,
        GH_CALLED=str(called),
    )
    subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", route["steps"][0]["run"]],
        env=environment,
        check=True,
    )
    assert output.read_text() == f"run={expected}\n"
    assert called.exists() == queried


def test_release_reuses_only_validated_distributions_from_the_same_revision():
    checks = _workflow("tests.yml")
    release = _workflow("release.yml")
    upload = next(
        step
        for step in checks["jobs"]["quality-and-package"]["steps"]
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    download = next(
        step
        for step in release["jobs"]["release"]["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert upload["if"] == "github.ref_type == 'tag' && startsWith(github.ref, 'refs/tags/v')"
    assert (
        upload["with"]["name"]
        == download["with"]["name"]
        == "release-distributions-${{ github.sha }}"
    )
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["overwrite"] == "true"
    assert upload["with"]["retention-days"] == "1"
    assert download["with"]["path"] == "dist"
    assert not any(
        "python -m build" in step.get("run", "") for step in release["jobs"]["release"]["steps"]
    )
