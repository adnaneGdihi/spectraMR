"""Structural invariants of ``.github/workflows/dev-publish.yml``.

The lane cannot be driven here (no runner, no PyPI, and an upload burns a filename
for good), so each property is decided from the parsed document and planted against
a mutated copy -- the same response ``test_release_workflow.py`` gives the release
lane. A predicate that only ever sees the real file is one nobody has watched fail.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "dev-publish.yml"

# PyYAML resolves the bare key `on` to True (YAML 1.1).
_ON = True
_REPO_GUARD = "github.repository == 'adnaneGdihi/spectraMR'"
_UPLOAD_GATE = "needs.build.outputs.publish == 'true'"


def _load() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def trigger_violations(doc: dict[str, Any]) -> list[str]:
    """The lane must start from a `dev-build-*` tag, and from nothing autonomous."""
    on = doc.get(_ON) or {}
    found: list[str] = []
    push = on.get("push")
    if not isinstance(push, dict) or push.get("tags") != ["dev-build-*"]:
        found.append(f"push must be tags: ['dev-build-*'], got {push!r}")
    elif "branches" in push:
        found.append("push carries a branches filter")
    if "schedule" in on:
        found.append("a schedule never fires: it registers from the default branch")
    return found


def tag_pattern_overlaps_release(doc: dict[str, Any]) -> bool:
    """A `v*`-shaped pattern would also fire release.yml and test-release.yml."""
    tags = ((doc.get(_ON) or {}).get("push") or {}).get("tags") or []
    return any(t.startswith("v") for t in tags)


def upload_not_gated_on_dev_head(doc: dict[str, Any]) -> list[str]:
    """The upload job must wait on the build job's dev-head verdict."""
    jobs = doc.get("jobs") or {}
    pypi = jobs.get("pypi") or {}
    found: list[str] = []
    if _UPLOAD_GATE not in str(pypi.get("if", "")):
        found.append(f"pypi.if lacks {_UPLOAD_GATE!r}")
    build = jobs.get("build") or {}
    if (build.get("outputs") or {}).get("publish") != "${{ steps.gate.outputs.publish }}":
        found.append("build does not export the gate step's verdict")
    gate = [s for s in build.get("steps") or [] if s.get("id") == "gate"]
    if not gate or "refs/heads/dev" not in gate[0].get("run", ""):
        found.append("no gate step comparing against refs/heads/dev")
    return found


def jobs_without_repo_guard(doc: dict[str, Any]) -> list[str]:
    """The file ships to both repositories; only the public one may publish."""
    return sorted(
        n for n, j in (doc.get("jobs") or {}).items() if _REPO_GUARD not in str(j.get("if", ""))
    )


def test_the_real_workflow_is_clean() -> None:
    doc = _load()
    assert trigger_violations(doc) == []
    assert not tag_pattern_overlaps_release(doc)
    assert upload_not_gated_on_dev_head(doc) == []
    assert jobs_without_repo_guard(doc) == []


@pytest.mark.parametrize(
    "push",
    [
        None,
        {"branches": ["dev"]},
        {"tags": ["*"]},
        {"tags": ["dev-build-*"], "branches": ["dev"]},
    ],
    ids=["dispatch-only", "branch-push", "any-tag", "tag-plus-branch"],
)
def test_trigger_violations_fire(push: dict[str, Any] | None) -> None:
    doc = copy.deepcopy(_load())
    doc[_ON].pop("push")
    if push is not None:
        doc[_ON]["push"] = push
    assert trigger_violations(doc)


def test_a_schedule_is_a_trigger_violation() -> None:
    doc = copy.deepcopy(_load())
    doc[_ON]["schedule"] = [{"cron": "0 3 * * *"}]
    assert trigger_violations(doc)


def test_a_release_shaped_tag_is_an_overlap() -> None:
    doc = copy.deepcopy(_load())
    doc[_ON]["push"]["tags"] = ["v*"]
    assert tag_pattern_overlaps_release(doc)


def test_an_upload_gated_on_the_ref_alone_fires() -> None:
    doc = copy.deepcopy(_load())
    doc["jobs"]["pypi"]["if"] = f"github.ref == 'refs/heads/dev' && {_REPO_GUARD}"
    assert upload_not_gated_on_dev_head(doc) == [f"pypi.if lacks {_UPLOAD_GATE!r}"]


def test_a_dropped_build_output_fires() -> None:
    doc = copy.deepcopy(_load())
    doc["jobs"]["build"].pop("outputs")
    assert upload_not_gated_on_dev_head(doc) == ["build does not export the gate step's verdict"]


def test_a_gate_step_that_never_reads_dev_fires() -> None:
    doc = copy.deepcopy(_load())
    for step in doc["jobs"]["build"]["steps"]:
        if step.get("id") == "gate":
            step["run"] = 'echo "publish=true" >> "$GITHUB_OUTPUT"'
    assert upload_not_gated_on_dev_head(doc) == ["no gate step comparing against refs/heads/dev"]


@pytest.mark.parametrize("job", ["build", "pypi"])
def test_a_job_without_the_repo_guard_fires(job: str) -> None:
    doc = copy.deepcopy(_load())
    doc["jobs"][job]["if"] = str(doc["jobs"][job]["if"]).replace(_REPO_GUARD, "true")
    assert jobs_without_repo_guard(doc) == [job]
