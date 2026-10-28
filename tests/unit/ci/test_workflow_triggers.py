"""Every workflow trigger is a pull request.

Guards one policy: every workflow trigger is a pull request. The full test suite runs on
a cluster rather than on hosted runners, so a `schedule:` cron or a branch `push:` in
`.github/workflows/` spends hosted-runner minutes on work nobody asked for, on hardware
far worse suited to it.

The policy is stated here rather than cited from a page, because the page that carried
it is internal and does not ship: a failure message that names a file the reader does
not have sends them looking for it instead of at the workflow.

Two mechanics make this worth a gate rather than a convention:

* `schedule` / `push` / `workflow_dispatch` / `pull_request_target` / `issue_comment` are
  read from the **default branch**, so a cron added here is invisible on `dev` and fires
  from `main` -- and, in the other direction, `nightly.yml` carried a cron for its whole
  life and never fired once because it was never published to `main`.
* A `branches:` filter on a `pull_request` trigger matches the PR's *base*, and PRs here
  target **both** `dev` (feature work) and `main` (dependabot bumps, `dev`->`main`
  publish PRs). Pinning it to `main` disabled every check on every `dev` PR until
  2026-07-12 while the PR page rendered clean; pinning it to `dev` would do the same to
  the `main` ones. With two live bases the only safe filter is none. That invariant has
  no other home, so it lives here too.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"

# Events that are a pull request, or are scoped to one.
_PR_EVENTS = frozenset(
    {
        "pull_request",
        "pull_request_target",
        "pull_request_review",
        "pull_request_review_comment",
    }
)

# A button in the Actions tab. Never autonomous, so it is allowed anywhere.
_MANUAL_EVENTS = frozenset({"workflow_dispatch"})

# The only non-PR events in the repo, each initiated by a human, with the reason. A new
# entry here is a deliberate exception to the policy and belongs in the same commit as
# the workflow that needs it.
_ALLOWED_NON_PR_EVENTS: dict[str, dict[str, str]] = {
    "release.yml": {
        "push": "tag-only: fires on `git push origin vX.Y.Z`, the publish step",
    },
    "test-release.yml": {
        # NOT the artefact: this lane does `pip install -e .` and `make test-release`,
        # so it exercises the checkout. release.yml is what verifies what is published.
        "push": "tag-only: runs the full test lane against the tagged source tree",
        # `release: types: [published]` was removed 2026-09-04, and this entry with
        # it -- GitHub suppresses runs triggered by a GITHUB_TOKEN-published release
        # (so it never fired on the automated path), while on a manual publish it
        # queued a second full-length run behind the tag's. `_stale_entries` is what
        # made the pair inseparable, which is the point of it.
    },
    "dev-publish.yml": {
        # Not dispatch: dispatch registers from the default branch, which does not
        # carry this file. A tag push runs the file at the tagged commit.
        "push": "tag-only: fires on `git push origin dev-build-<stamp>`; uploads only a "
        "build of public dev's head",
    },
    "claude.yml": {
        "issues": "the @claude bot; the job body requires an @claude mention",
        "issue_comment": "the @claude bot; the job body requires an @claude mention",
    },
    "manual-full-suite.yml": {
        # Declared by the OVERLAY copy only. The private file is dispatch-only,
        # and a cron there would never fire -- crons are read from the default
        # branch, and this tree does not publish .github/ to one. The public
        # export does, which is why the exception is scoped to the overlay and
        # why _stale_entries takes the union over both directories rather than
        # resolving this name against .github/workflows/ alone.
        "schedule": (
            "the PUBLIC maintainer lane: whole-tree suite, sphinx -W, supply-chain "
            "and link checks, weekly. It is not a required check, and a lane that "
            "only runs on dispatch never fails on its own -- which is exactly why "
            "the export used to deny this file."
        ),
    },
}

_ALLOWLIST = _REPO_ROOT / "scripts" / "release" / "public_allowlist.txt"

# Workflows this repository HAS but the public distribution deliberately denies.
#
# An _ALLOWED_NON_PR_EVENTS entry naming one of these is not stale in a tree that
# does not ship the workflow -- it is an entry whose workflow was removed by SCOPE.
# A bare ``path.exists()`` cannot tell the two apart, and they need opposite
# treatment: rot must fail, a scope decision must not. Absent is a state to
# report, never a state to infer.
#
# The excuse is not free-floating. Each name here must appear as a `!` denial in
# scripts/release/public_allowlist.txt, which ships in both trees -- so deleting
# the denial without deleting this entry fails, and the two cannot drift apart
# unnoticed. The excuse covers ABSENCE only: a workflow that is present is
# checked in full, exactly as if it were not listed here.
_NOT_DISTRIBUTED: dict[str, str] = {
    "claude.yml": (
        "issue_comment-triggered and consumes secrets.CLAUDE_CODE_OAUTH_TOKEN; a "
        "long-lived OAuth token behind a comment trigger is an exposure path on a "
        "public repository, and the secret does not exist there."
    ),
}


def _denied_in_public_allowlist(allowlist_text: str) -> set[str]:
    """Workflow basenames the public allowlist denies with a `!` line."""
    denied: set[str] = set()
    for raw in allowlist_text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("!.github/workflows/"):
            denied.add(line.rsplit("/", 1)[-1])
    return denied


def _stale_entries(
    allowed: dict[str, dict[str, str]],
    workflow_dirs: Sequence[Path],
    not_distributed: dict[str, str],
    denied: set[str],
) -> list[str]:
    """The pure core of the stale check, so it can be planted against directly.

    ``workflow_dirs`` is a SEQUENCE, and that is load-bearing rather than
    generality for its own sake. A workflow name can exist in both the private
    directory and the overlay, with DIFFERENT triggers -- the maintainer lane's
    `schedule:` is declared by the overlay copy alone, because a cron in this
    tree would never fire. Resolving the name against one directory reports
    such an event as stale: the register would be telling the truth and the
    check would call it rot. So an entry is stale only when NO copy declares
    the event, and absent only when NO copy exists.
    """
    stale: list[str] = []
    for name, events in allowed.items():
        paths = [d / name for d in workflow_dirs if (d / name).is_file()]
        if not paths:
            if name not in not_distributed:
                stale.append(f"{name}: workflow no longer exists")
            elif name not in denied:
                stale.append(
                    f"{name}: excused by _NOT_DISTRIBUTED but the public allowlist "
                    f"does not deny it -- the excuse names a scope decision that "
                    f"scripts/release/public_allowlist.txt no longer makes"
                )
            continue
        declared: set[str] = set()
        for path in paths:
            declared |= set(_triggers(path))
        stale.extend(
            f"{name}: allowlists `{event}`, which the workflow no longer declares"
            for event in events
            if event not in declared
        )
    return stale


# The public distribution's workflows are stored HERE, not in .github/workflows/,
# so that GitHub does not also run them against this repository. That storage
# location put them outside this file's scan root -- and a scan root is an
# unaudited constant: no violation planted in .github/workflows/ can reveal that
# a second directory of workflows exists and is unchecked. The `branches:` filter
# this module exists to forbid would have been invisible in exactly the files
# that get published, until someone read the exported tree.
_OVERLAY_WORKFLOW_DIR = (
    _REPO_ROOT / "scripts" / "release" / "public_overlay" / ".github" / "workflows"
)


def _workflow_paths() -> list[Path]:
    """Every workflow this repository OWNS, wherever it is stored.

    Both directories, deliberately. The overlay copies are published verbatim, so
    the policy applies to them identically; the only difference is which repo
    runs them.
    """
    dirs = [_WORKFLOW_DIR, _OVERLAY_WORKFLOW_DIR]
    paths = sorted(
        p for d in dirs if d.is_dir() for ext in ("*.yml", "*.yaml") for p in d.glob(ext)
    )
    assert paths, f"no workflows found under any of {dirs}"
    return paths


def _triggers(path: Path) -> dict[str, Any]:
    """The `on:` block, normalised to {event: config}.

    YAML 1.1 resolves the bare key `on` to the boolean `True`, so a plain `doc["on"]`
    raises KeyError on every workflow in the repo. Read both spellings.
    """
    doc = yaml.safe_load(path.read_text())
    block = doc.get("on", doc.get(True))
    assert block is not None, f"{path.name} has no `on:` block"
    if isinstance(block, str):
        return {block: None}
    if isinstance(block, list):
        return dict.fromkeys(block)
    return block


# ``.github/`` is a dropped root in the public export -- Workstream F ports the
# two PR workflows into the new repo rather than exporting this tree's ten.
# Without this skip, ``_workflow_paths()`` below raises during IMPORT, and a
# collection error is not a test failure: it aborts the whole pytest session
# with "Interrupted: 1 error during collection", so ``pytest tests/`` in the
# export collects NOTHING AT ALL. One shipped file made the entire suite
# unrunnable, and the workaround was an ``--ignore`` flag a public user has no
# way to know they need.
#
# The distinction the assert in ``_workflow_paths`` draws is preserved rather
# than weakened: a MISSING directory is a scope decision and skips visibly; a
# PRESENT but empty one is a defect and still raises.
if not _WORKFLOW_DIR.is_dir():
    pytest.skip(
        f"no workflow directory at {_WORKFLOW_DIR}: .github/ is not part of this tree",
        allow_module_level=True,
    )

_WORKFLOWS = _workflow_paths()
# Not p.name: both directories hold a `pr-required.yml`, and two parametrised
# cases sharing an id are indistinguishable in a failure report.
_IDS = [
    (f"overlay:{p.name}" if p.is_relative_to(_OVERLAY_WORKFLOW_DIR) else p.name) for p in _WORKFLOWS
]


def _schedule_violation(
    name: str,
    triggers: dict[str, Any],
    *,
    fires_on_default_branch: bool,
    allowed: dict[str, dict[str, str]],
) -> str | None:
    """Why this workflow may not declare a `schedule:`, or None if it may.

    SCOPE-AWARE, and the asymmetry is the whole point. The blanket ban this
    replaces carried a justification -- "the full suite runs on a cluster rather
    than on hosted runners" -- that is true of THIS repository and false of the
    published one, which has no cluster and whose maintainer lane exists
    precisely to fail on its own. A rule enforced where its reason does not hold
    is a rule that has to be argued with rather than read.

    The private directory keeps the unconditional ban, and _ALLOWED_NON_PR_EVENTS
    does NOT license it there. That is deliberate: a cron in .github/workflows/
    would not fire anyway (crons are read from a default branch this tree never
    publishes .github/ to), so an entry permitting one would document a
    capability that does not exist -- the advertised-but-inert shape. Putting the
    maintainer lane in the overlay is the fix, not registering it here.

    `fires_on_default_branch` is a SEMANTIC and deliberately not a path test.
    The two agree in this tree and diverge in the exported one, where the
    overlay directory is gone and `.github/workflows/` IS the published lane --
    so a caller deriving this flag from "is the file under the overlay path"
    reports the public maintainer lane's own cron as a violation, in the only
    tree where that cron actually fires. Resolve it with
    `_lane_fires_on_default_branch`, which asks the question this parameter
    names.
    """
    if "schedule" not in triggers:
        return None
    if not fires_on_default_branch:
        return (
            f"{name} declares a `schedule:` cron in .github/workflows/. A cron is read "
            f"from the DEFAULT BRANCH, and this tree does not publish .github/ to one -- "
            f"nightly.yml carried a cron for its entire life and fired zero times. Use "
            f"`workflow_dispatch:`. If this is the PUBLIC maintainer lane, it belongs in "
            f"the overlay directory, where it does fire; registering it here would not "
            f"make it run."
        )
    if "schedule" not in allowed.get(name, {}):
        return (
            f"overlay:{name} declares a `schedule:` cron with no `schedule` entry in "
            f"_ALLOWED_NON_PR_EVENTS. The published repository CAN run a scheduled lane, "
            f"but every cron there is a deliberate exception that states its reason next "
            f"to the workflow that needs it."
        )
    return None


def _lane_fires_on_default_branch(path: Path, overlay_dir: Path) -> bool:
    """Would a `schedule:` in this file actually run?

    Two trees, and the same path means different things in each:

    * private tree -- `overlay_dir` exists. `.github/workflows/` is the private
      lane, whose crons are dead because this tree never publishes `.github/` to
      a default branch; the overlay is the published lane and its crons fire.
    * exported tree -- `overlay_dir` is absent by design (the export replaces
      `.github/workflows/` with the overlay's content). What is left IS the
      published lane, on the repository where `main` is the default branch.

    `overlay_dir` is a parameter rather than the module constant so a test can
    build either tree in a tmp_path. That is the whole reason this defect
    survived: the predicate's seven plants each pass the flag as a literal, so
    no plant could observe the CALL SITE computing it wrongly.
    """
    if path.is_relative_to(overlay_dir):
        return True
    return not overlay_dir.is_dir()


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_no_workflow_runs_on_a_schedule(path: Path) -> None:
    violation = _schedule_violation(
        path.name,
        _triggers(path),
        fires_on_default_branch=_lane_fires_on_default_branch(path, _OVERLAY_WORKFLOW_DIR),
        allowed=_ALLOWED_NON_PR_EVENTS,
    )
    assert violation is None, violation


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_push_triggers_are_tag_only(path: Path) -> None:
    push = _triggers(path).get("push")
    if push is None:
        return
    allowed = _ALLOWED_NON_PR_EVENTS.get(path.name, {})
    assert "push" in allowed, (
        f"{path.name} triggers on `push:`. Only the tag-driven release lanes may, and "
        f"they are listed in _ALLOWED_NON_PR_EVENTS."
    )
    assert isinstance(push, dict) and "tags" in push and "branches" not in push, (
        f"{path.name} triggers on a branch push ({push!r}). A push trigger is allowed "
        f"only when scoped to tags -- a branch push is autonomous CI."
    )


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_non_pull_request_triggers_are_allowlisted(path: Path) -> None:
    allowed = _ALLOWED_NON_PR_EVENTS.get(path.name, {})
    unexpected = {
        event
        for event in _triggers(path)
        if event not in _PR_EVENTS | _MANUAL_EVENTS and event not in allowed
    }
    assert not unexpected, (
        f"{path.name} triggers on {sorted(unexpected)}, which is neither a pull-request "
        f"event nor an allowlisted human-initiated one. Every trigger here is a pull "
        f"request; extend _ALLOWED_NON_PR_EVENTS only for an event a human starts."
    )


@pytest.mark.parametrize("path", _WORKFLOWS, ids=_IDS)
def test_pull_request_triggers_have_no_branches_filter(path: Path) -> None:
    """The 2026-07-12 invariant: a `branches:` filter matches the PR's BASE branch."""
    for event in ("pull_request", "pull_request_target"):
        config = _triggers(path).get(event)
        if not isinstance(config, dict):
            continue
        assert "branches" not in config, (
            f"{path.name} filters `{event}` on branches={config['branches']!r}. That "
            f"matches the PR's BASE branch, and PRs here target BOTH `dev` and `main` "
            f"-- pinning it to `main` is what silently disabled every check until "
            f"2026-07-12, and pinning it to `dev` would do the same to the `main` PRs. "
            f"A skipped workflow renders as no check at all, not as a red one."
        )


def test_allowlist_has_no_stale_entries() -> None:
    """An exception that no workflow still needs is a licence nobody is using."""
    allowlist_text = _ALLOWLIST.read_text(encoding="utf-8") if _ALLOWLIST.exists() else ""
    stale = _stale_entries(
        _ALLOWED_NON_PR_EVENTS,
        [_WORKFLOW_DIR, _OVERLAY_WORKFLOW_DIR],
        _NOT_DISTRIBUTED,
        _denied_in_public_allowlist(allowlist_text),
    )
    assert not stale, "stale _ALLOWED_NON_PR_EVENTS entries:\n  " + "\n  ".join(stale)


def test_every_not_distributed_entry_is_denied_by_the_public_allowlist() -> None:
    """The excuse and the denial are one decision; neither may move alone."""
    assert _ALLOWLIST.exists(), f"{_ALLOWLIST} is missing -- the excuse cannot be checked"
    denied = _denied_in_public_allowlist(_ALLOWLIST.read_text(encoding="utf-8"))
    missing = sorted(set(_NOT_DISTRIBUTED) - denied)
    assert not missing, (
        "_NOT_DISTRIBUTED names workflows the public allowlist does not deny: " + ", ".join(missing)
    )


# ---------------------------------------------------------------------------
# Plants against the stale check itself (non-negotiable 15).
#
# The check grew a branch that EXCUSES an absence, and an excuse is the one kind
# of change that makes a detector quieter rather than louder. So each rule and
# each shape it can take is planted here and required to turn the core red --
# committed as tests, not run once by hand. The negative controls matter equally:
# a check that fires on everything is as useless as one that fires on nothing.
# ---------------------------------------------------------------------------

_PR_ONLY_WORKFLOW = "on:\n  pull_request:\n"
_TAG_PUSH_WORKFLOW = "on:\n  push:\n    tags: ['v*']\n"


def _workflows(tmp_path: Path, **files: str) -> Path:
    d = tmp_path / "workflows"
    d.mkdir(parents=True)
    for name, body in files.items():
        (d / name.replace("__", ".")).write_text(body, encoding="utf-8")
    return d


def test_an_absent_workflow_with_no_excuse_is_stale(tmp_path: Path) -> None:
    """The original rule. Deleting a workflow must not leave its licence behind."""
    stale = _stale_entries({"ghost.yml": {"push": "why"}}, [_workflows(tmp_path)], {}, set())
    assert stale == ["ghost.yml: workflow no longer exists"]


def test_an_excused_absence_that_the_allowlist_does_not_deny_is_stale(
    tmp_path: Path,
) -> None:
    """The excuse is anchored to the denial: remove the denial and it fails."""
    stale = _stale_entries(
        {"ghost.yml": {"push": "why"}},
        [_workflows(tmp_path)],
        {"ghost.yml": "scope decision"},
        denied=set(),
    )
    assert len(stale) == 1
    assert "the public allowlist does not deny it" in stale[0]


def test_an_excused_absence_that_the_allowlist_denies_is_clean(tmp_path: Path) -> None:
    """Negative control: the whole point of the branch is that this case passes."""
    assert (
        _stale_entries(
            {"ghost.yml": {"push": "why"}},
            [_workflows(tmp_path)],
            {"ghost.yml": "scope decision"},
            denied={"ghost.yml"},
        )
        == []
    )


def test_the_excuse_does_not_leak_into_a_present_workflow(tmp_path: Path) -> None:
    """The excuse covers ABSENCE only.

    This is the shape that would make the change a net loss: a name listed in
    _NOT_DISTRIBUTED that is nonetheless PRESENT must be checked exactly as if it
    were not listed at all, or listing a workflow there would silently stop
    auditing its triggers in the tree that actually runs it.
    """
    stale = _stale_entries(
        {"ghost.yml": {"issue_comment": "why"}},
        [_workflows(tmp_path, ghost__yml=_PR_ONLY_WORKFLOW)],
        {"ghost.yml": "scope decision"},
        denied={"ghost.yml"},
    )
    assert len(stale) == 1
    assert "no longer declares" in stale[0]


def test_a_present_workflow_that_still_declares_its_event_is_clean(
    tmp_path: Path,
) -> None:
    """Negative control for the branch above."""
    assert (
        _stale_entries(
            {"ghost.yml": {"push": "tag-only"}},
            [_workflows(tmp_path, ghost__yml=_TAG_PUSH_WORKFLOW)],
            {},
            set(),
        )
        == []
    )


def test_denial_parser_reads_a_real_denial_and_ignores_a_commented_one() -> None:
    """Both shapes, because the allowlist is a commented file by construction.

    A denial carrying a trailing comment must still count, and a denial that is
    itself commented out must not -- the second is how a reverted decision would
    otherwise keep excusing an entry forever.
    """
    denied = _denied_in_public_allowlist(
        "\n".join(
            [
                "!.github/workflows/real.yml",
                "!.github/workflows/trailing.yml   # with a reason",
                "# !.github/workflows/commented.yml",
                "  !.github/workflows/indented.yml",
                ".github/                          # an ALLOW, not a denial",
                "!tests/unit/ci/not_a_workflow.py",
            ]
        )
    )
    assert denied == {"real.yml", "trailing.yml", "indented.yml"}


# ---------------------------------------------------------------------------
# Plants against the schedule rule (non-negotiable 15).
#
# The rule gained a SCOPE and a register, and both make a detector quieter: one
# excuses a whole directory, the other excuses a named file. So every shape is
# planted -- each scope, each register state, and the negative controls that
# prove the rule has not simply been switched off. The private+registered case
# is the one that would otherwise rot: nothing else notices if the register
# silently starts licensing the tree it was never meant to reach.
# ---------------------------------------------------------------------------

_CRON = {"schedule": [{"cron": "17 6 * * 1"}]}
_NO_CRON = {"pull_request": None}
_REGISTERED = {"lane.yml": {"schedule": "the public maintainer lane"}}


def test_a_cron_in_the_private_directory_is_a_violation() -> None:
    """The original rule. It must still fire on the shape it always fired on."""
    msg = _schedule_violation("lane.yml", _CRON, fires_on_default_branch=False, allowed={})
    assert msg is not None and "DEFAULT BRANCH" in msg


def test_the_register_does_not_license_a_cron_in_the_private_directory() -> None:
    """The shape that would make this change a net loss.

    Adding a name to _ALLOWED_NON_PR_EVENTS must not quietly permit a cron in
    .github/workflows/. It would not fire there, so permitting it advertises a
    capability the tree does not have -- and the register entry would read as
    proof that someone had thought about it.
    """
    msg = _schedule_violation("lane.yml", _CRON, fires_on_default_branch=False, allowed=_REGISTERED)
    assert msg is not None and "belongs in the overlay directory" in msg


def test_an_unregistered_cron_in_the_overlay_is_a_violation() -> None:
    """Widening the scope must not turn the overlay into a free-for-all."""
    msg = _schedule_violation("lane.yml", _CRON, fires_on_default_branch=True, allowed={})
    assert msg is not None and "_ALLOWED_NON_PR_EVENTS" in msg


def test_a_registered_cron_in_the_overlay_is_clean() -> None:
    """Negative control: the one case the change exists to permit."""
    assert (
        _schedule_violation("lane.yml", _CRON, fires_on_default_branch=True, allowed=_REGISTERED)
        is None
    )


@pytest.mark.parametrize("fires", [False, True])
def test_a_workflow_with_no_cron_is_clean_in_either_scope(fires: bool) -> None:
    """Negative control: a check that fires on everything is as useless as one
    that fires on nothing."""
    assert (
        _schedule_violation("lane.yml", _NO_CRON, fires_on_default_branch=fires, allowed={}) is None
    )


# --------------------------------------------------------------------------- #
# Plants for the RESOLVER, not the predicate. The nine above each hand the flag
# in as a literal, so every one of them passed while the call site computed it
# from `path.is_relative_to(_OVERLAY_WORKFLOW_DIR)` -- a path fact that answers
# a different question. Running the shipped suite inside a real export is what
# exposed it: there the overlay directory is gone, `.github/workflows/` IS the
# published lane, and the maintainer lane's own cron was reported as a violation
# in the only tree where it actually fires.
# --------------------------------------------------------------------------- #


def _tree(root: Path, *, with_overlay: bool) -> tuple[Path, Path]:
    """Build a private-shaped or export-shaped tree; return (workflows, overlay)."""
    workflows = root / ".github" / "workflows"
    workflows.mkdir(parents=True)
    overlay = root / "scripts" / "release" / "public_overlay" / ".github" / "workflows"
    if with_overlay:
        overlay.mkdir(parents=True)
    return workflows, overlay


def test_private_tree_github_workflows_does_not_fire(tmp_path: Path) -> None:
    """The ban's home: overlay present, so `.github/` is the inert private lane."""
    workflows, overlay = _tree(tmp_path, with_overlay=True)
    assert _lane_fires_on_default_branch(workflows / "lane.yml", overlay) is False


def test_private_tree_overlay_fires(tmp_path: Path) -> None:
    _, overlay = _tree(tmp_path, with_overlay=True)
    assert _lane_fires_on_default_branch(overlay / "lane.yml", overlay) is True


def test_exported_tree_github_workflows_fires(tmp_path: Path) -> None:
    """The shape the old call site got wrong, and the reason for this resolver.

    No overlay directory means the export already replaced `.github/workflows/`
    with the overlay's content, on a repository whose default branch is what the
    export writes. A cron there fires, so banning it bans the maintainer lane.
    """
    workflows, overlay = _tree(tmp_path, with_overlay=False)
    assert _lane_fires_on_default_branch(workflows / "lane.yml", overlay) is True


def test_the_export_shape_still_requires_registration(tmp_path: Path) -> None:
    """Composed end-to-end: widening the scope must not empty the register.

    An unregistered cron in the exported tree is still a violation -- otherwise
    the fix above would trade a false positive for a blind spot, which is the
    trade a ratchet must never make.
    """
    workflows, overlay = _tree(tmp_path, with_overlay=False)
    fires = _lane_fires_on_default_branch(workflows / "stray.yml", overlay)
    assert _schedule_violation("stray.yml", _CRON, fires_on_default_branch=fires, allowed={})


def test_an_event_declared_only_by_the_overlay_copy_is_not_stale(tmp_path: Path) -> None:
    """The union rule, and the reason _stale_entries takes a sequence.

    `manual-full-suite.yml` exists in BOTH directories with different triggers:
    dispatch-only in the private tree, dispatch + cron in the overlay. Resolving
    the name against .github/workflows/ alone reports the overlay's `schedule`
    as an event "the workflow no longer declares" -- rot, for a register entry
    that is telling the truth.
    """
    private = _workflows(tmp_path / "a", lane__yml="on:\n  workflow_dispatch:\n")
    overlay = _workflows(
        tmp_path / "b",
        lane__yml="on:\n  workflow_dispatch:\n  schedule:\n    - cron: '0 6 * * 1'\n",
    )
    assert _stale_entries({"lane.yml": {"schedule": "why"}}, [private, overlay], {}, set()) == []
    # Control: against the private directory alone it IS reported stale, which is
    # the bug this signature change fixes.
    assert _stale_entries({"lane.yml": {"schedule": "why"}}, [private], {}, set()) == [
        "lane.yml: allowlists `schedule`, which the workflow no longer declares"
    ]


def test_an_event_declared_by_no_copy_is_still_stale(tmp_path: Path) -> None:
    """The union must not swallow real rot: two directories, neither declaring it."""
    private = _workflows(tmp_path / "a", lane__yml=_PR_ONLY_WORKFLOW)
    overlay = _workflows(tmp_path / "b", lane__yml=_PR_ONLY_WORKFLOW)
    stale = _stale_entries({"lane.yml": {"schedule": "why"}}, [private, overlay], {}, set())
    assert stale == ["lane.yml: allowlists `schedule`, which the workflow no longer declares"]


def test_the_overlay_workflows_are_actually_scanned() -> None:
    """Anti-vacuity for the widened scan root.

    Without this, deleting `_OVERLAY_WORKFLOW_DIR` from `_workflow_paths` leaves
    every test above green -- the parametrised cases simply stop existing, and a
    test that no longer runs reports nothing at all. Assert the published
    workflows are in the set, by name.
    """
    if not _OVERLAY_WORKFLOW_DIR.is_dir():
        pytest.skip(f"{_OVERLAY_WORKFLOW_DIR} is absent -- not the tree that owns it")
    scanned = {p for p in _WORKFLOWS if p.is_relative_to(_OVERLAY_WORKFLOW_DIR)}
    on_disk = {p for ext in ("*.yml", "*.yaml") for p in _OVERLAY_WORKFLOW_DIR.glob(ext)}
    assert on_disk, f"{_OVERLAY_WORKFLOW_DIR} exists but holds no workflow"
    assert scanned == on_disk


# --------------------------------------------------------------------------- #
# A badge is a claim about a workflow, and nothing was checking it.
#
# README.md carried an Actions badge for `test.yml` -- a workflow that does not
# exist and never has. GitHub renders such a badge as a grey "no status" image
# rather than an error, so it reads as "CI configured" indefinitely. The workflow
# names are right here, so the claim is checkable.
# --------------------------------------------------------------------------- #

_BADGE_WORKFLOW = re.compile(r"/actions/workflows/([A-Za-z0-9._-]+\.ya?ml)")


def _workflows_named_in(text: str) -> set[str]:
    return set(_BADGE_WORKFLOW.findall(text))


def test_every_workflow_a_readme_badge_names_actually_exists() -> None:
    readme = _REPO_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover - README always ships
        pytest.skip("no README.md in this tree")
    named = _workflows_named_in(readme.read_text(encoding="utf-8"))
    have = {p.name for p in _WORKFLOWS}
    missing = sorted(named - have)
    assert not missing, (
        "README.md links an Actions badge to workflow(s) that do not exist: "
        + ", ".join(missing)
        + ". GitHub renders a badge for a missing workflow as a grey 'no status' "
        "image, not as an error, so it reads as working CI forever."
    )


def test_the_badge_scan_finds_the_badge_that_is_there() -> None:
    """Anti-vacuity: a regex that matches nothing passes the test above."""
    readme = _REPO_ROOT / "README.md"
    if not readme.exists():  # pragma: no cover
        pytest.skip("no README.md in this tree")
    assert _workflows_named_in(readme.read_text(encoding="utf-8")), (
        "no Actions badge found in README.md -- either the badge was removed, or "
        "the pattern stopped matching and the check above is now vacuous"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://github.com/o/r/actions/workflows/ci.yml/badge.svg", {"ci.yml"}),
        ("…/actions/workflows/pr-required.yaml)", {"pr-required.yaml"}),
        ("…/actions/workflows/a.yml … /actions/workflows/b.yml", {"a.yml", "b.yml"}),
        ("https://github.com/o/r/actions", set()),
        ("see .github/workflows/ci.yml in the tree", set()),
    ],
)
def test_the_badge_pattern_reads_a_url_and_not_prose(text: str, expected: set[str]) -> None:
    """Both directions. The last case is the one that would cause false alarms:
    a prose mention of a workflow path is not a badge and must not be read as one.
    """
    assert _workflows_named_in(text) == expected


# --------------------------------------------------------------------------- #
# docs/known_limitations.rst is the single public owner of "does CI run here?",
# and it enumerates the published lane's jobs. Prose that names jobs is a second
# owner for the workflow's shape (non-negotiable 17) unless something holds the
# two together -- and nothing did. The sentence these tests replace advertised a
# YAML-audit tier the published lane has never carried, and a guard-script count
# measured on the PRIVATE copy of the same filename: both true of a lane, just
# not of the lane the page is about. Nothing in the tree reads shipped prose, so
# the divergence was invisible until it was read by a person.
# --------------------------------------------------------------------------- #

_KNOWN_LIMITATIONS = _REPO_ROOT / "docs" / "known_limitations.rst"
_JOB_PROSE_ANCHOR = "Its blocking jobs are"
_RST_JOB_LITERAL = re.compile(r"``([a-z][a-z0-9-]*)``")


def _jobs_named_in(text: str) -> set[str]:
    """Job names the prose claims, read from the one paragraph that lists them.

    Scoped to a paragraph rather than to the page, and that is the difference
    between a check and a formality. A page-wide literal scan picks up every
    ``...`` on it -- filenames, make targets, flags -- so the comparison below
    would be satisfied by a superset that no rewrite could ever fall out of.
    """
    for paragraph in text.split("\n\n"):
        if _JOB_PROSE_ANCHOR in paragraph:
            return set(_RST_JOB_LITERAL.findall(paragraph))
    return set()


def _published_lane() -> Path | None:
    """The ``pr-required.yml`` that gates PUBLIC merges, in whichever tree this is.

    Overlay first, and the order carries the whole meaning: in the private tree
    BOTH files exist and they are different lanes. ``.github/workflows/`` there
    carries a ``yaml-audit`` job the published lane does not have and lacks the
    ``hygiene`` job it does, so resolving to it would pin the shipped page
    against a lane no public pull request has ever run. In the exported tree the
    overlay is absent by design and the shipped file IS the published lane.
    """
    overlay = _OVERLAY_WORKFLOW_DIR / "pr-required.yml"
    if overlay.is_file():
        return overlay
    shipped = _WORKFLOW_DIR / "pr-required.yml"
    return shipped if shipped.is_file() else None


def _job_keys(path: Path) -> set[str]:
    return set((yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("jobs") or {})


def test_known_limitations_names_the_lane_that_is_actually_published() -> None:
    if not _KNOWN_LIMITATIONS.is_file():  # pragma: no cover
        pytest.skip(f"{_KNOWN_LIMITATIONS} is not part of this tree")
    lane = _published_lane()
    if lane is None:  # pragma: no cover
        pytest.skip("no pr-required.yml in this tree")

    named = _jobs_named_in(_KNOWN_LIMITATIONS.read_text(encoding="utf-8"))
    defined = _job_keys(lane)
    # Both sides asserted non-empty BEFORE they are compared. Two empty sets are
    # equal, so a rotted anchor on one side and an unparseable `jobs:` block on
    # the other would agree with each other and report a page that names nothing
    # as correct.
    assert named, f"no paragraph containing {_JOB_PROSE_ANCHOR!r} in {_KNOWN_LIMITATIONS.name}"
    assert defined, f"{lane} parsed no jobs"
    assert named == defined, (
        f"{_KNOWN_LIMITATIONS.name} names {sorted(named)} but {lane.name} defines "
        f"{sorted(defined)} -- the shipped page is the single public owner of this "
        "list, so a divergence is a wrong answer given to a reader, not a typo"
    )


_LANE_PARAGRAPH = (
    "Its blocking jobs are ``lint-diff``, ``guards`` and ``hygiene``, aggregated\nby ``required``."
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The shape that shipped: prose naming a job the published lane dropped.
        (
            "Its blocking jobs are ``lint-diff`` and ``yaml-audit``.",
            {"lint-diff", "yaml-audit"},
        ),
        # The opposite shape: prose that omits a job the lane gained. `hygiene`
        # is exactly this -- it was added to the lane with no page to match.
        ("Its blocking jobs are ``lint-diff`` and ``guards``.", {"lint-diff", "guards"}),
        # Anchor absent -> empty, which the caller asserts against rather than
        # comparing. A rewrite that drops the sentence must not read as agreement.
        ("The lane runs ``lint-diff`` and ``guards`` on every PR.", set()),
        # Scoping: literals in OTHER paragraphs must not leak in. Without this the
        # named set grows to the whole page and stops being falsifiable.
        (
            f"{_LANE_PARAGRAPH}\n\nSee also ``physics`` and ``security`` elsewhere.",
            {"lint-diff", "guards", "hygiene", "required"},
        ),
        # A dotted literal is a filename, not a job key, and must not be captured:
        # `pr-required.yml` sits one sentence away from this list on the page.
        ("Its blocking jobs are ``lint-diff``; see ``pr-required.yml``.", {"lint-diff"}),
    ],
)
def test_the_job_prose_scan_reads_one_paragraph_and_not_the_page(
    text: str, expected: set[str]
) -> None:
    """Planted violations (non-negotiable 15), one per shape the rule can take:
    a stale name, a missing name, a rotted anchor, a cross-paragraph leak, and a
    filename that looks like a job. The first two are the shapes actually
    observed on this page; the last three are how the detector goes blind.
    """
    assert _jobs_named_in(text) == expected


# ---------------------------------------------------------------------------
# Every `SKIP:` in a workflow names a hook that exists
# ---------------------------------------------------------------------------
#
# pre-commit accepts a SKIP naming an id it has never heard of, in silence: no
# warning, no non-zero exit, the hook simply runs. So a SKIP entry that outlives
# the hook it names reads as protection on the page and is none in the run --
# and the reverse, a hook renamed out from under a SKIP, turns a deliberately
# excused gate back on where nobody expects it. Neither direction announces
# itself. `.pre-commit-config.yaml` is the authority on which ids exist; this is
# the only thing that asks it.

_PRE_COMMIT_CONFIG = _REPO_ROOT / ".pre-commit-config.yaml"


def _live_hook_ids(config_text: str) -> set[str]:
    return {
        str(hook["id"])
        for repo in yaml.safe_load(config_text).get("repos", [])
        for hook in repo.get("hooks", [])
        if hook.get("id")
    }


def _skip_id_lists(node: Any) -> list[list[str]]:
    """Every `SKIP:` value in a workflow document, split into ids.

    Recursive rather than keyed on a fixed path: `env:` is legal at workflow, job
    and step level, and a scan anchored to one of the three is blind to the other
    two. Each entry is stripped, because `a, b` is two ids to pre-commit and the
    naive split yields the id `" b"`, which matches no hook and would read as a
    violation of exactly the rule this checks.
    """
    if isinstance(node, dict):
        found: list[list[str]] = []
        for key, value in node.items():
            if key == "SKIP" and isinstance(value, str):
                found.append([t.strip() for t in value.split(",") if t.strip()])
            else:
                found.extend(_skip_id_lists(value))
        return found
    if isinstance(node, list):
        return [ids for item in node for ids in _skip_id_lists(item)]
    return []


def test_every_precommit_skip_names_a_hook_that_exists() -> None:
    live = _live_hook_ids(_PRE_COMMIT_CONFIG.read_text(encoding="utf-8"))
    assert len(live) >= 10, (
        f"only {len(live)} hook id(s) parsed out of {_PRE_COMMIT_CONFIG.name}; the "
        "config carries more than that, so the parse is wrong and a clean result "
        "here means nothing"
    )

    declared = [
        (path, ids)
        for path in _workflow_paths()
        for ids in _skip_id_lists(yaml.safe_load(path.read_text(encoding="utf-8")))
    ]
    assert declared, (
        "no SKIP found in any workflow, in either directory. The overlay's "
        "pr-required.yml carries one, so this scan has stopped reaching it and "
        "passes over an empty set."
    )

    offenders = sorted(
        f"{path.name}: {hook_id}" for path, ids in declared for hook_id in ids if hook_id not in live
    )
    assert not offenders, (
        "a workflow SKIPs a pre-commit hook id that does not exist. pre-commit "
        "accepts this silently, so the entry protects nothing while reading as "
        "though it does:\n" + "\n".join(f"  {o}" for o in offenders)
    )


_SKIP_PLANTS: list[tuple[str, str, list[list[str]]]] = [
    ("job-level env", "jobs:\n  a:\n    env:\n      SKIP: mypy\n", [["mypy"]]),
    (
        "step-level env",
        "jobs:\n  a:\n    steps:\n      - env:\n          SKIP: mypy,codespell\n",
        [["mypy", "codespell"]],
    ),
    (
        "spaces after the commas",
        "jobs:\n  a:\n    env:\n      SKIP: 'mypy, codespell'\n",
        [["mypy", "codespell"]],
    ),
    ("no SKIP anywhere", "jobs:\n  a:\n    steps:\n      - run: true\n", []),
    ("an empty SKIP", "jobs:\n  a:\n    env:\n      SKIP: ''\n", [[]]),
]


@pytest.mark.parametrize(
    "shape,text,expected", _SKIP_PLANTS, ids=[p[0] for p in _SKIP_PLANTS]
)
def test_the_skip_scan_finds_every_shape_a_skip_can_take(
    shape: str, text: str, expected: list[list[str]]
) -> None:
    """Planted (non-negotiable 15), one per shape rather than one per rule.

    The last two are the shapes that make the scan report a clean tree for the
    wrong reason: an empty result and a result the reader cannot tell from one.
    """
    assert _skip_id_lists(yaml.safe_load(text)) == expected, shape


def test_the_skip_check_flags_a_dead_id() -> None:
    """Red on the real failure: a SKIP that outlived its hook."""
    live = _live_hook_ids(
        "repos:\n  - repo: local\n    hooks:\n      - id: alive\n        entry: x\n"
    )
    ids = _skip_id_lists(yaml.safe_load("jobs:\n  a:\n    env:\n      SKIP: alive,retired\n"))
    assert [i for group in ids for i in group if i not in live] == ["retired"]
