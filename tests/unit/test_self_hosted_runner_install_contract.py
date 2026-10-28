"""The ARM64 self-hosted lane must be able to install what it declares.

Two knobs decide whether the maintainer's self-hosted runner (`thor`,
linux/arm64) can run at all, and they live in different files:

* the workflows -- which jobs target ``[self-hosted, Linux, ARM64]`` and which
  extras they install through ``.github/actions/setup-env``. Both scan roots are
  read: the in-tree ``.github/workflows/`` and the export overlay, which is where
  the public repo's ``pull_request`` lanes actually come from;
* ``pyproject.toml`` -- whether every requirement in those extras can actually
  resolve on ``platform_machine == "aarch64"``.

Pinning either alone is blind to its partner. Drop the ``platform_machine !=
'aarch64'`` marker from gudhi and every ARM64 job dies during
``pip install -e ".[dev]"`` -- before a single test runs, with an error that
looks like a network fault rather than a policy change. Point a new job at the
ARM64 labels and the same thing happens silently. So the assertion here is the
*coupling*: for the extras the ARM64 jobs install, the aarch64-resolved
requirement set must not contain a distribution known to be unavailable there.

gudhi is the one such distribution today: it publishes no aarch64 wheel and no
sdist for any release >= 3.12 (PyPI JSON API, 2026-09-04). POT is deliberately
NOT in this set -- it ships manylinux_2_27_aarch64 wheels plus an sdist, and
marking it off would remove a dependency ARM64 installs perfectly well.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement

_REPO = Path(__file__).resolve().parents[2]
_TREE_WORKFLOWS = _REPO / ".github" / "workflows"

# Second scan root, and the one that matters most here. The overlay is the
# export's other write path: it OVERWRITES allowlisted files, so pr-required.yml
# and pr-advisory.yml reach the public repo from HERE, not from the tree above.
# Both are `pull_request`-triggered -- exactly the trigger the fork-PR check at
# the bottom of this module exists to police -- so scanning only the tree would
# leave this module blind in the one place the risk actually lives.
#
# A scan root is an unaudited constant: no violation planted under
# .github/workflows/ can reveal that a second workflow directory exists.
# tests/unit/ci/test_workflow_triggers.py records the same finding for triggers;
# this is the same blindness for runner labels.
_OVERLAY_WORKFLOWS = _REPO / "scripts" / "release" / "public_overlay" / ".github" / "workflows"

# Distributions that must NOT survive extras resolution on linux/aarch64.
#
# Derived by sweep, not by anecdote. The first entry was added from the one
# failure that happened to be observed, which is how a set like this silently
# goes stale -- so it was re-derived exhaustively on 2026-09-04 by two passes:
#   (i)  every one of the 257 packages in the tracked uv.lock, bucketed by
#        whether it offers a pure wheel, an aarch64 wheel, or only an sdist;
#   (ii) the PyPI simple API for each survivor, across ALL published versions,
#        because CI installs with pip and is not bound to the uv.lock pins.
# That pass found scalene, which pass (i) alone would have rated "buildable".
#
# The two entries fail for DIFFERENT reasons and the distinction is load-bearing:
#   gudhi   -- 332 wheels, 0 aarch64, 0 sdists: pip has nothing to install.
#   scalene -- 1231 wheels, 0 linux-aarch64 (its 17 arm wheels are macOS, which
#              reports platform_machine 'arm64'), sdist present, 4 compiled
#              objects: installable only by compiling C++ on the runner.
# Re-run both passes before trusting this set; membership moves with upstream.
UNAVAILABLE_ON_AARCH64 = {"gudhi", "scalene"}

ARM64_LABELS = {"self-hosted", "linux", "arm64"}

# A `runs-on` may be an expression: the release lane's hosted fallback is
# `${{ cond && 'ubuntu-latest' || fromJSON('["self-hosted", ...]') }}`. The label
# test has to read through that, or a self-hosted job behind an expression is
# invisible to every assertion in this module -- the fork-PR one included. Every
# fromJSON array literal and every remaining quoted scalar is a candidate label
# set, and a job counts as ARM64 if ANY candidate reaches the runner, because any
# of them can be the one chosen at run time.
_FROMJSON_RE = re.compile(r"fromJSON\(\s*'(\[[^']*\])'\s*\)")
_QUOTED_RE = re.compile(r"'([^']*)'")


def _workflow_files() -> list[Path]:
    """Every workflow in both scan roots -- one owner for discovery.

    GitHub honours `.yaml` as well as `.yml`. Globbing only `.yml` would let a
    `.yaml` workflow pairing `pull_request` with the ARM64 labels evade every
    check in this module.

    Absent is a state to report, never a state to infer. The export ships
    `.github/` today, but that is a scope decision the allowlist owns and can
    reverse -- a tree without it must SKIP visibly (this module is out of scope
    there) rather than fail, while a tree that HAS the directory and no
    workflows in it is a defect and must raise.
    """
    if not _TREE_WORKFLOWS.is_dir():
        pytest.skip(f"{_TREE_WORKFLOWS.relative_to(_REPO)} is not shipped in this tree")
    files = sorted(_TREE_WORKFLOWS.glob("*.yml")) + sorted(_TREE_WORKFLOWS.glob("*.yaml"))
    assert files, f"{_TREE_WORKFLOWS} exists but ships no workflows -- discovery is broken"
    if _OVERLAY_WORKFLOWS.is_dir():
        files += sorted(_OVERLAY_WORKFLOWS.glob("*.yml"))
        files += sorted(_OVERLAY_WORKFLOWS.glob("*.yaml"))
    return files


def _label(path: Path) -> str:
    """Name a workflow by scan root, so a failure says WHICH copy is at fault."""
    return str(path.relative_to(_REPO))


def _candidate_label_sets(runs_on: object) -> list[set[str]]:
    """Every label set a ``runs-on`` value can resolve to."""
    if isinstance(runs_on, list):
        return [{str(x).lower() for x in runs_on}]
    text = str(runs_on)
    if "${{" not in text:
        return [{text.lower()}]
    sets = [{str(x).lower() for x in json.loads(m)} for m in _FROMJSON_RE.findall(text)]
    sets += [{q.lower()} for q in _QUOTED_RE.findall(_FROMJSON_RE.sub("", text))]
    return sets


def _is_arm64(job: dict) -> bool:
    """One owner for the label test -- every call site below resolves through it."""
    return any(labels >= ARM64_LABELS for labels in _candidate_label_sets(job.get("runs-on")))


def _offers_hosted_fallback(job: dict) -> bool:
    """A self-hosted job must be movable to a hosted runner without an edit."""
    runs_on = job.get("runs-on")
    if not isinstance(runs_on, str) or "${{" not in runs_on:
        return False
    candidates = _candidate_label_sets(runs_on)
    return "inputs.runner" in runs_on and any(c == {"ubuntu-latest"} for c in candidates)


def _fork_reachable_arm64_jobs(doc: dict) -> list[str]:
    """ARM64 job names in a workflow that any fork pull request can trigger."""
    # PyYAML reads the bare key `on` as the boolean True (YAML 1.1).
    triggers = doc.get(True, doc.get("on")) or {}
    names = set(triggers) if isinstance(triggers, dict) else {triggers}
    if not names & {"pull_request", "pull_request_target"}:
        return []
    return [name for name, job in (doc.get("jobs") or {}).items() if _is_arm64(job)]


def _arm64_jobs() -> list[tuple[str, str, dict]]:
    """Every (workflow, job_name, job) whose ``runs-on`` is the ARM64 label set."""
    return [
        (_label(path), name, job)
        for path in _workflow_files()
        for name, job in ((yaml.safe_load(path.read_text()) or {}).get("jobs") or {}).items()
        if _is_arm64(job)
    ]


def _extras_installed_by(job: dict) -> set[str]:
    """Extras the job asks ``setup-env`` for (its input defaults to ``dev``)."""
    extras: set[str] = set()
    for step in job.get("steps") or []:
        uses = str(step.get("uses", ""))
        if "actions/setup-env" not in uses:
            continue
        declared = (step.get("with") or {}).get("extras", "dev")
        extras.update(part.strip() for part in str(declared).split(",") if part.strip())
    return extras


def _resolve(extra: str, machine: str, seen: set[str] | None = None) -> set[str]:
    """Distribution names an extra pulls in on ``machine``, following self-refs."""
    seen = seen if seen is not None else set()
    if extra in seen:
        return set()
    seen.add(extra)
    with (_REPO / "pyproject.toml").open("rb") as fh:
        table = tomllib.load(fh)["project"]["optional-dependencies"]
    names: set[str] = set()
    for spec in table.get(extra, []):
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({"platform_machine": machine}):
            continue
        # `spectramr[a,b]` self-references compose the extras by reference.
        if req.name.replace("_", "-") == "spectramr":
            for nested in req.extras:
                names |= _resolve(nested, machine, seen)
        else:
            names.add(req.name.lower())
    return names


def _base_dependencies(machine: str) -> set[str]:
    """Distribution names in ``project.dependencies`` on ``machine``.

    A second scan root, for the same reason the overlay is one above. Every
    other leg in this module resolves ``optional-dependencies`` -- so the base
    requirement list, which is installed unconditionally and is where torch
    lives, was invisible to all of them. A package that becomes aarch64-hostile
    *there* would break the runner with every extras-based leg still green,
    because the blind spot and any planted violation live in different tables.
    """
    with (_REPO / "pyproject.toml").open("rb") as fh:
        specs = tomllib.load(fh)["project"]["dependencies"]
    names: set[str] = set()
    for spec in specs:
        req = Requirement(spec)
        if req.marker and not req.marker.evaluate({"platform_machine": machine}):
            continue
        names.add(req.name.lower())
    return names


def test_base_dependencies_are_installable_on_aarch64() -> None:
    """The unconditional requirement list must not carry an aarch64 blocker.

    Benign when written (the 2026-09-04 sweep put every blocker in an extra);
    this leg is what keeps it that way, since nothing else here reads the table.
    """
    base = _base_dependencies("aarch64")
    assert base, "project.dependencies resolved to nothing -- lookup is broken"
    offenders = base & UNAVAILABLE_ON_AARCH64
    assert not offenders, (
        f"{sorted(offenders)} are unconditional dependencies but cannot be "
        f"installed on linux/aarch64, so the self-hosted runner cannot install "
        f"this package at all. Move them to an extra with a platform_machine "
        f"marker, as topology/gudhi and profile/scalene already are."
    )


def test_at_least_one_job_targets_the_arm64_runner() -> None:
    """Guard against a vacuous pass.

    Every assertion below quantifies over the ARM64 jobs. If that set were
    empty -- a rename of the labels, a revert -- the checks would pass while
    verifying nothing.
    """
    jobs = _arm64_jobs()
    assert jobs, (
        "No job targets [self-hosted, Linux, ARM64]. Either the release lane was "
        "moved back to hosted runners (then delete this module), or the labels "
        "were changed and these checks silently stopped covering anything."
    )


def test_arm64_jobs_install_extras_that_resolve_on_aarch64() -> None:
    """The coupling: what the ARM64 jobs install must be installable there."""
    checked = 0
    for workflow, job_name, job in _arm64_jobs():
        for extra in _extras_installed_by(job):
            resolved = _resolve(extra, "aarch64")
            assert resolved, f"[{extra}] resolved to nothing -- lookup is broken"
            offenders = resolved & UNAVAILABLE_ON_AARCH64
            assert not offenders, (
                f"{workflow}:{job_name} installs [{extra}], which still pulls "
                f"{sorted(offenders)} on aarch64. Those publish no aarch64 wheel "
                f"and no sdist, so `pip install -e '.[{extra}]'` fails during "
                "resolution and the job dies before running anything. Add a "
                "`; platform_machine != 'aarch64'` marker in pyproject.toml."
            )
            checked += 1
    assert checked, "no ARM64 job installs any extra -- coupling unverified"


@pytest.mark.parametrize("name", sorted(UNAVAILABLE_ON_AARCH64))
def test_marker_is_arch_scoped_not_a_deletion(name: str) -> None:
    """Each excluded distribution must still install on x86_64.

    Deleting the dependency outright would also make the test above pass, and
    would silently disable the cubical_ph_w2 arms (gudhi) or `spectramr profile`
    (scalene) on every machine that can run them. The marker is an architecture
    scope, not a removal, and this is the leg that tells the two apart.
    """
    assert name in _resolve("dev", "x86_64"), f"{name} was deleted, not scoped"
    assert name not in _resolve("dev", "aarch64")


@pytest.mark.parametrize("extra", ["topology", "all", "dev"])
def test_pot_stays_available_on_aarch64(extra: str) -> None:
    """POT ships aarch64 wheels; marking it off would be an over-correction."""
    assert "pot" in _resolve(extra, "aarch64")


def test_no_arm64_job_is_reachable_from_a_fork_pull_request() -> None:
    """spectraMR is public: a fork PR on a self-hosted runner is code execution.

    ``pull_request`` and ``pull_request_target`` both run for forks. A workflow
    carrying either trigger must not contain an ARM64 self-hosted job.
    """
    for path in _workflow_files():
        arm_jobs = _fork_reachable_arm64_jobs(yaml.safe_load(path.read_text()) or {})
        assert not arm_jobs, (
            f"{_label(path)} has a pull_request-family trigger AND self-hosted ARM64 "
            f"job(s) {arm_jobs}. On a public repo that executes fork-authored code on "
            "the maintainer's machine. Keep PR lanes on hosted runners."
        )


_EXPR_WITH_THOR_FALLBACK = (
    "${{ inputs.runner == 'hosted' && 'ubuntu-latest' "
    "|| fromJSON('[\"self-hosted\", \"Linux\", \"ARM64\"]') }}"
)


@pytest.mark.parametrize(
    ("runs_on", "expected"),
    [
        (["self-hosted", "Linux", "ARM64"], True),
        ("ubuntu-latest", False),
        (_EXPR_WITH_THOR_FALLBACK, True),
        ("${{ vars.CUDA_RUNNER_LABEL }}", False),
        ("${{ inputs.runner == 'hosted' && 'ubuntu-latest' || 'ubuntu-22.04' }}", False),
        ("${{ fromJSON('[\"self-hosted\", \"Linux\", \"X64\"]') }}", False),
    ],
)
def test_label_test_reads_through_runs_on_expressions(runs_on: object, expected: bool) -> None:
    """Planted shapes: the label test must see the runner behind an expression.

    Before this, an expression-valued ``runs-on`` was stringified and compared
    as one label, so a self-hosted job behind a fallback expression evaded
    every check in this module.
    """
    assert _is_arm64({"runs-on": runs_on}) is expected


def test_fork_pr_check_sees_an_arm64_job_behind_an_expression() -> None:
    """The fork-PR gate must go red on the expression shape, not only the list."""
    doc = {
        "on": {"pull_request": None},
        "jobs": {"suite": {"runs-on": _EXPR_WITH_THOR_FALLBACK}},
    }
    assert _fork_reachable_arm64_jobs(doc) == ["suite"]
    doc["jobs"]["suite"]["runs-on"] = "ubuntu-latest"
    assert _fork_reachable_arm64_jobs(doc) == []


def test_every_arm64_job_offers_the_hosted_fallback() -> None:
    """thor is offline unless started, and GitHub has no failover between pools.

    A job pinned to the label triple alone queues for 24 h and is cancelled.
    Every ARM64 job therefore carries the `runner` dispatch input's hosted
    branch in its expression, so a person can move it without editing YAML.
    """
    pinned = [
        f"{workflow}:{name}"
        for workflow, name, job in _arm64_jobs()
        if not _offers_hosted_fallback(job)
    ]
    assert not pinned, (
        f"{pinned} target [self-hosted, Linux, ARM64] with no hosted fallback. "
        "Use the runs-on expression release.yml's build job carries."
    )


@pytest.mark.parametrize(
    ("runs_on", "expected"),
    [
        (_EXPR_WITH_THOR_FALLBACK, True),
        (["self-hosted", "Linux", "ARM64"], False),
        ("${{ vars.RELEASE_RUNNER == 'hosted' && 'ubuntu-latest' || fromJSON('[\"self-hosted\", \"Linux\", \"ARM64\"]') }}", False),
    ],
)
def test_fallback_detector_planted_shapes(runs_on: object, expected: bool) -> None:
    """A literal triple, or an expression a dispatch cannot steer, is pinned."""
    assert _offers_hosted_fallback({"runs-on": runs_on}) is expected
