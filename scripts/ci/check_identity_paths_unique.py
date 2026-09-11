#!/usr/bin/env python
"""CI gate: no two experiment arms may write to the same identity path.

An arm's *identity paths* -- where its results, checkpoints, logs and metrics go
-- must name that arm and no other. Two arms resolving the same value do not
error, they interleave: the second run's checkpoints overwrite the first's, both
append to one loss CSV, and a control's metrics land in the treatment's tree, so
the evidence behind a published number is silently a mixture of two runs (#1930).

**This gate resolves; it does not parse**, for the reason
``check_witness_corpus.py`` states: a gate that reads the YAML grades the
document rather than the run. That is load-bearing here.
``checkpoint.checkpoint_dir`` carries ``validation_alias="save_dir"`` and is the
ONLY aliased field in the schema tree, with 293 corpus arms using the alias
spelling. A parsing gate sees two independent keys where the run sees one field.

Two clauses, because the first has a blind spot the second covers:

1. **No two arms resolve the same identity-path value.** Plain duplicate
   detection, deliberately with no stem vocabulary and no group-directory
   heuristic: a group directory is spelled ``<group>/<stem>`` and so is unique
   per arm for free, while every real collision here is an exact repeat.
2. **No arm's identity path names a DIFFERENT arm.** An arm pointing at a sibling
   spelled differently yields no duplicate at all, and a copied config keeps
   naming its template until someone reads it. A segment equal to the arm's OWN
   stem is exempt, which keeps this quiet on the group directories and
   self-abbreviations that make a naive "names this arm" rule unusable.

Scope is values under ``experiments/``. ``logging.identity.experiment`` is
excluded, and the reason is stronger than "it is a label rather than a
location": NOTHING under ``src/`` reads it. Its only two mentions are a
``renames.py`` row and an entry in ``paired_arms_diff_paths.DEFAULT_DIFF_PATHS``
-- the allow-list of paths two paired arms may legitimately DIFFER on. Seven
names are in fact shared by fourteen arms, three of them a control arm carrying
its treatment arm's name, but gating on collisions in a value no run consumes
would report a defect nothing can act on; the unread knob is filed instead.

Two blind spots follow from the scope and are tracked
separately: relative values, which resolve against the launch CWD; and an arm
declaring BOTH spellings of the aliased field, where the alias wins and the
visible ``checkpoint_dir`` never reaches the run. ``docs/contributing/ci.rst``
carries the measurements and the reasoning for all four decisions.

``test_kspace_filling_cohort_invariants.py::test_o_identity_paths_name_this_arm``
enforces a *stricter, different* invariant -- every identity path must name
**this** arm -- over one cohort standardized to satisfy it. Neither subsumes the
other: an arm can name itself and still collide, if its twin also names it.
Non-negotiable 17 forbids two owners of one invariant, not two that overlap.

Ratchet -- the three-way shape of ``check_witness_corpus.py``: a finding absent
from the baseline FAILS, a baseline entry gone clean is info, entries still
present PASS as tracked debt. Entries are scoped to the *finding* -- ``(key,
value)`` for clause 1, ``(arm, key, value)`` for clause 2 -- never to a file,
which would waive every future collision that arm acquires.

Usage: ``python scripts/ci/check_identity_paths_unique.py [root] [--baseline P]
[--strict] [--update-baseline]``.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spectramr.config.settings import TrainingSettings  # noqa: E402
from spectramr.infrastructure.validation.config_health_checker import (  # noqa: E402
    IDENTITY_PATHS,
)

BASELINE = REPO_ROOT / "scripts" / "ci" / "identity_paths_baseline.txt"

#: ``IDENTITY_PATHS`` is imported, not restated (non-negotiable 17). It names the
#: dotted attribute paths on the RESOLVED settings object that locate an arm's
#: artifacts -- canonical spellings only, since the schema has already folded
#: ``save_dir`` into ``checkpoint_dir``, which is the point of resolving rather
#: than parsing. Its owner is ``config_health_checker``, whose sibling check
#: ``check_identity_paths_agree`` applies the same six paths WITHIN one config;
#: this gate applies them ACROSS the corpus. Two copies of the list would let
#: those two answers disagree about what an identity path is.

#: Only values naming a location in the corpus tree are in scope. See the
#: module docstring for why relative values are excluded and where they live.
CORPUS_PREFIX = "experiments/"


class MissingBaselineError(Exception):
    """The baseline file is absent.

    Non-negotiable 18: absent is a state to REPORT, never to infer. Reading a
    missing baseline as ``set()`` grades every known finding as a regression, and
    turns a deleted baseline into a clean gate on the next ``--update-baseline``.
    """


def _rel(path: Path, root: Path) -> str:
    """Report path relative to the repo when it is inside it, else to the scan root.

    A gate that can only address paths under ``REPO_ROOT`` cannot be pointed at a
    planted corpus in a tmp dir, which makes its own blindness untestable
    (non-negotiable 15).
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path.relative_to(root))


def _read_attr(obj: object, dotted: str) -> object | None:
    """Walk a dotted attribute path, returning None if any hop is absent."""
    cur: object | None = obj
    for segment in dotted.split("."):
        cur = getattr(cur, segment, None)
        if cur is None:
            return None
    return cur


def identity_values(settings: object) -> dict[str, str]:
    """The in-scope identity paths this resolved config actually uses."""
    found: dict[str, str] = {}
    for key in IDENTITY_PATHS:
        value = _read_attr(settings, key)
        if isinstance(value, str) and value.startswith(CORPUS_PREFIX):
            found[key] = value.rstrip("/")
    return found


def collect(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, set[str]], int]:
    """Resolve every arm under ``root``.

    Returns the per-arm identity values, the stem -> arms index clause 2 needs,
    and the count of configs that would not load. Schema validity is
    ``check_experiment_configs_load.py``'s debt and it carries its own baseline;
    inheriting it here would make this gate's verdict meaningless.
    """
    resolved: dict[str, dict[str, str]] = {}
    stems: dict[str, set[str]] = defaultdict(set)
    unloadable = 0
    for path in sorted(root.rglob("*.yaml")):
        if "archive" in path.parts:
            continue
        rel = _rel(path, root)
        # The stem index is a property of the FILENAME, so it must include arms
        # that fail to load: an unloadable arm still owns its name, and a
        # sibling pointing at it is still a hijack.
        stems[path.stem].add(rel)
        try:
            settings = TrainingSettings.from_yaml(str(path))
        except Exception:
            unloadable += 1
            continue
        resolved[rel] = identity_values(settings)
    return resolved, stems, unloadable


def find_duplicates(resolved: dict[str, dict[str, str]]) -> dict[str, list[str]]:
    """Clause 1: identity-path values claimed by more than one arm."""
    owners: dict[tuple[str, str], set[str]] = defaultdict(set)
    for rel, values in resolved.items():
        for key, value in values.items():
            owners[(key, value)].add(rel)
    return {
        f"duplicate::{key}::{value}": sorted(arms)
        for (key, value), arms in owners.items()
        if len(arms) > 1
    }


def find_hijacks(
    resolved: dict[str, dict[str, str]], stems: dict[str, set[str]]
) -> dict[str, list[str]]:
    """Clause 2: an identity path naming a different arm than the one that owns it."""
    findings: dict[str, list[str]] = {}
    for rel, values in resolved.items():
        own_stem = Path(rel).stem
        for key, value in values.items():
            for segment in Path(value).parts:
                if segment == own_stem:
                    # Naming yourself is the correct case, at any depth. This is
                    # what exempts group directories and self-abbreviations.
                    break
                named = stems.get(segment)
                if named:
                    findings[f"names-another-arm::{rel}::{key}::{value}"] = sorted(named)
                    break
    return findings


def read_baseline(baseline: Path | None = None) -> set[str]:
    baseline = BASELINE if baseline is None else baseline
    if not baseline.exists():
        raise MissingBaselineError(str(baseline))
    return {
        line.strip()
        for line in baseline.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def write_baseline(entries: list[str], baseline: Path | None = None) -> None:
    baseline = BASELINE if baseline is None else baseline
    baseline.write_text(
        "# Arms whose identity paths collide with another arm's (issue #1930).\n"
        "#\n"
        "# `duplicate::<key>::<value>`         -- two or more arms resolve this value\n"
        "# `names-another-arm::<arm>::<k>::<v>` -- this arm's path names a different arm\n"
        "#\n"
        "# An entry here is tracked DEBT, not permission. Two arms sharing a results\n"
        "# tree interleave checkpoints, logs and metrics with no error, so the\n"
        "# artifacts behind a number become a mixture of two runs. Fix by giving the\n"
        "# arm its own path -- conventionally `experiments/results/<the arm's stem>`.\n"
        "# Entries are scoped to the finding, never to a file.\n"
        "#\n"
        "# Regenerate: python scripts/ci/check_identity_paths_unique.py --update-baseline\n"
        + "".join(f"{entry}\n" for entry in sorted(entries))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="experiments/inprogress")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline file for THIS root. A second tree needs its own; sharing "
        "one makes each gate inherit the other's debt.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on ANY finding, not just ones absent from the baseline.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite the baseline from the current tree.",
    )
    args = parser.parse_args(argv)

    # Absent corpus vs empty corpus: two facts, two answers (non-negotiable 18).
    # `experiments/` is absent from the published tree, so there this gate has no
    # subject -- and it runs as a pre-commit hook. Report the absence and decline
    # rather than crashing or passing in silence. A root that EXISTS and is empty
    # falls through to the scan, where zero arms is a clean run over a real corpus.
    root = REPO_ROOT / args.root
    if not root.is_dir():
        print(
            f"{args.root}/ is not present in this tree, so there are no arms whose "
            "identity paths could collide -- declining rather than reporting a "
            "clean run. This is expected in the published distribution, which "
            "ships configuration templates instead of the experiment corpus."
        )
        return 0

    resolved, stems, unloadable = collect(root)
    duplicates = find_duplicates(resolved)
    hijacks = find_hijacks(resolved, stems)
    detail = {**duplicates, **hijacks}
    current = set(detail)

    if args.update_baseline:
        write_baseline(sorted(current), args.baseline)
        print(f"baseline rewritten with {len(current)} entr(ies)")
        return 0

    colliding_arms = {arm for arms in duplicates.values() for arm in arms}
    print(f"resolved {len(resolved)} config(s); {unloadable} skipped (schema-invalid)")
    print(
        f"colliding identity paths: {len(duplicates)} shared by {len(colliding_arms)} arm(s); "
        f"{len(hijacks)} path(s) naming another arm"
    )

    try:
        baseline = read_baseline(args.baseline)
    except MissingBaselineError as exc:
        print(
            f"\nFAILED: the baseline file is absent ({exc}). Refusing to treat a "
            "missing baseline as an empty one -- that would grade every known "
            "finding as a new regression, and would silently bless the whole "
            "corpus on the next --update-baseline. Restore the file from git, or "
            "create it deliberately with --update-baseline."
        )
        return 1

    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        print(f"\ninfo: {len(fixed)} baseline entr(ies) are now clean — tighten with")
        print("      python scripts/ci/check_identity_paths_unique.py --update-baseline")
        for entry in fixed[:20]:
            print(f"  {entry}")

    failing = sorted(current) if args.strict else new
    if failing:
        label = "finding(s)" if args.strict else "NEW finding(s)"
        print(f"\nFAILED: {len(failing)} {label}")
        for entry in failing[:40]:
            print(f"  {entry}")
            for arm in detail[entry]:
                print(f"      {arm}")
        if len(failing) > 40:
            print(f"  ... and {len(failing) - 40} more")
        print(
            "\nTwo arms sharing a results tree do not error: they interleave. The "
            "second run overwrites the first's checkpoints, both append to the same "
            "loss CSV, and the metrics behind a published number become a mixture "
            "of two experiments."
        )
        return 1

    print(f"\nOK: {len(current)} known finding(s) in the baseline, 0 new.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
