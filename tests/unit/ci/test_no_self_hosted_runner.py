"""No workflow job may run on a self-hosted runner.

spectraMR is a public repository and every lane runs on GitHub-hosted runners
since the maintainer's ARM64 runner (``thor``) was retired on 2026-10-28. A
self-hosted job is the one shape that turns a fork pull request, or a queued tag
push, into code execution on a machine nobody is watching -- and an offline
runner does not reject a job, it queues it until the host next comes up.

Both scan roots are read: the in-tree ``.github/workflows/`` and the export
overlay, which is where the public repo's ``pull_request`` lanes come from. A
``runs-on`` may be an expression, so every ``fromJSON`` array and quoted scalar
in it is a candidate label set; a job is self-hosted if ANY candidate is.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[3]
_TREE_WORKFLOWS = _REPO / ".github" / "workflows"
_OVERLAY_WORKFLOWS = _REPO / "scripts" / "release" / "public_overlay" / ".github" / "workflows"

_FROMJSON_RE = re.compile(r"fromJSON\(\s*'(\[[^']*\])'\s*\)")
_QUOTED_RE = re.compile(r"'([^']*)'")


def _workflow_files() -> list[Path]:
    """Every workflow in both scan roots, ``.yml`` and ``.yaml`` alike."""
    if not _TREE_WORKFLOWS.is_dir():
        pytest.skip(f"{_TREE_WORKFLOWS.relative_to(_REPO)} is not shipped in this tree")
    files: list[Path] = []
    for root in (_TREE_WORKFLOWS, _OVERLAY_WORKFLOWS):
        if root.is_dir():
            files += sorted(root.glob("*.yml")) + sorted(root.glob("*.yaml"))
    assert files, f"{_TREE_WORKFLOWS} exists but ships no workflows -- discovery is broken"
    return files


def _candidate_label_sets(runs_on: object) -> list[set[str]]:
    """Every label set a ``runs-on`` value can resolve to."""
    if isinstance(runs_on, list):
        return [{str(x).lower() for x in runs_on}]
    if isinstance(runs_on, dict):
        return _candidate_label_sets(runs_on.get("labels", runs_on.get("group", "")))
    text = str(runs_on)
    if "${{" not in text:
        return [{text.lower()}]
    sets = [{str(x).lower() for x in json.loads(m)} for m in _FROMJSON_RE.findall(text)]
    sets += [{q.lower()} for q in _QUOTED_RE.findall(_FROMJSON_RE.sub("", text))]
    return sets


def _is_self_hosted(job: dict) -> bool:
    """One owner for the label test."""
    return any("self-hosted" in labels for labels in _candidate_label_sets(job.get("runs-on")))


def _jobs() -> list[tuple[str, str, dict]]:
    return [
        (str(path.relative_to(_REPO)), name, job)
        for path in _workflow_files()
        for name, job in ((yaml.safe_load(path.read_text()) or {}).get("jobs") or {}).items()
    ]


def test_discovery_finds_jobs_with_a_runs_on() -> None:
    """Anti-vacuity: the gate below quantifies over these."""
    assert any("runs-on" in job for _, _, job in _jobs()), "no job declares runs-on"


def test_no_job_targets_a_self_hosted_runner() -> None:
    offenders = [f"{wf}:{name}" for wf, name, job in _jobs() if _is_self_hosted(job)]
    assert not offenders, (
        f"{offenders} target a self-hosted runner. spectraMR is public and runs every "
        "lane on GitHub-hosted runners; use `runs-on: ubuntu-latest`."
    )


_RETIRED_THOR_EXPR = (
    "${{ (inputs.runner == 'hosted' || vars.RELEASE_RUNNER == 'hosted') && 'ubuntu-latest' "
    '|| fromJSON(\'["self-hosted", "Linux", "ARM64"]\') }}'
)


@pytest.mark.parametrize(
    ("runs_on", "expected"),
    [
        (["self-hosted", "Linux", "ARM64"], True),
        ("self-hosted", True),
        (_RETIRED_THOR_EXPR, True),
        ("${{ inputs.flag && 'self-hosted' || 'ubuntu-latest' }}", True),
        ({"labels": ["self-hosted", "gpu"]}, True),
        ("ubuntu-latest", False),
        (["ubuntu-24.04"], False),
        ("${{ inputs.flag && 'ubuntu-latest' || 'ubuntu-22.04' }}", False),
    ],
)
def test_label_test_planted_shapes(runs_on: object, expected: bool) -> None:
    """Every shape a self-hosted target can take turns the gate red, hosted ones do not."""
    assert _is_self_hosted({"runs-on": runs_on}) is expected
