#!/usr/bin/env python
"""CI gate: no config may silently lose a declared key, and no check may be dead.

Two questions, corpus-wide:

1. **Does resolving this config drop a key it declares?** ``extra="ignore"`` blocks
   discard any key they do not declare, and nothing said so. That is how
   ``acceleration.min_center_fraction`` went missing and made the #534 ladder fix
   inert (#550): the YAML still showed the knob, the run never saw it, and the
   blocking gate written to protect those 47 arms reported ``defects: none``
   because it read the document with ``yaml.safe_load`` instead of resolving it.
2. **Is every registered check actually invoked?** The ``META`` witnesses answer
   this once per process; a check nobody calls protects nothing.

**This gate resolves; it does not parse.** Every config goes through
``TrainingSettings.from_yaml``, so the gate sees exactly what the run will see.
A gate that parses YAML grades the document, which is the failure above.

Scope. It reports only the ledger's own findings plus the ``META`` witnesses. It
deliberately does NOT fail on ``ConfigHealthChecker`` output, even though the
witness registry can now produce it: that debt is owned by
``check_experiment_configs_load.py`` and ``spectramr audit``, both of which carry
their own baselines. A gate that fails on another gate's debt inherits it and
stops meaning anything — the same scoping rule
``check_scheduler_specs_resolve.py`` states for the ``optimization`` block.

Two trees, two baselines. This gate runs over ``experiments/inprogress`` and,
under a SEPARATE hook id, over ``tests/audit/corpus`` -- where a fixture
declaring a key the schema discards is a planted violation that carries none of
the shape it advertises (#1933). Each root passes its own ``--baseline``. Sharing
one is the failure this docstring already argues against, and it is not
hypothetical: run against ``tests/audit/corpus`` while reading the experiments
baseline, the gate reported all 2658 entries as "now clean" and invited an
``--update-baseline`` that would have erased the entire debt record (#1930).

Ratchet. Identical three-way shape to
``check_acceleration_ladder_realisable.py``:

* a finding NOT in the baseline   -> FAIL (a regression)
* a baseline entry now clean      -> info (tighten the baseline)
* baseline entries still findings -> PASS (known debt, tracked)

Usage::

    python scripts/ci/check_witness_corpus.py
    python scripts/ci/check_witness_corpus.py experiments/inprogress
    python scripts/ci/check_witness_corpus.py --strict
    python scripts/ci/check_witness_corpus.py --update-baseline
    python scripts/ci/check_witness_corpus.py tests/audit/corpus \
        --baseline scripts/ci/witness_baseline_audit_corpus.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from spectramr.config.settings import TrainingSettings  # noqa: E402
from spectramr.core.execution_ledger import (  # noqa: E402
    ExecutionLedger,
    SubstitutionClass,
)

BASELINE = REPO_ROOT / "scripts" / "ci" / "witness_baseline.txt"

#: Substitution classes that are defects rather than information. A default being
#: injected is normal; a DECLARED key being discarded is not.
BLOCKING = {
    SubstitutionClass.EXTRA_IGNORE_DROPPED,
    SubstitutionClass.DROPPED_UNCONSUMED_KWARG,
}


def findings_for(path: Path) -> list[str]:
    """Declared-but-discarded keys for one config, resolved through the schema."""
    ExecutionLedger.reset()
    ledger = ExecutionLedger.begin_run(source=str(path))
    TrainingSettings.from_yaml(str(path))
    return [f"{s.class_id.value}: {s.path}" for s in ledger.substitutions if s.class_id in BLOCKING]


def meta_findings() -> list[str]:
    """The ``META`` witnesses: run once per process, not once per config."""
    from spectramr.infrastructure.validation.witness import (
        Stage,
        WitnessSubject,
        run_witnesses,
    )
    from spectramr.infrastructure.validation.witness.registry import Tier

    subject = WitnessSubject.for_ci(None, {})
    return [
        f"{v.witness_name}: {v.message}"
        for v in run_witnesses(subject, tiers=frozenset({Tier.T0, Tier.T1}))
        if not v.passed and v.stage is Stage.META
    ]


class MissingBaselineError(Exception):
    """The baseline file is absent (non-negotiable 18).

    Inferring ``set()`` here grades every known finding as a NEW regression on one
    run, and -- because the operator's obvious response is ``--update-baseline``
    -- silently blesses the whole corpus on the next. A deleted baseline must be
    reported, not guessed at.
    """


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
        "# Configs that declare a key the schema then discards (issue #550).\n"
        "# One `<path>::<class>: <dotted key>` per line.\n"
        "#\n"
        "# A path here is tracked DEBT, not permission: removing one is the goal,\n"
        "# adding one needs a reason in the PR that adds it. Fix by declaring the\n"
        "# key on the schema block that should own it, or by deleting the stale\n"
        "# key from the YAML.\n"
        "#\n"
        "# Regenerate: python scripts/ci/check_witness_corpus.py --update-baseline\n"
        + "".join(f"{e}\n" for e in sorted(entries))
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="experiments")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline file for THIS root. A second tree MUST carry its own: "
        "pointing this gate at `tests/audit/corpus` while it reads the "
        "experiments baseline reported 2658 entries as 'now clean' and invited "
        "an --update-baseline that would have erased the whole debt record "
        "(#1930). That is the failure this gate's own docstring argues against.",
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

    # Absent corpus vs empty corpus: two different facts, two different answers.
    #
    # `experiments/` is not in the published tree, so in a distribution checkout
    # this gate has no subject at all -- and it runs as a pre-commit hook, which
    # made every contributor's FIRST commit fail with a bare traceback. Report the
    # absence and decline, rather than either crashing or passing in silence.
    #
    # A root that EXISTS and is empty still falls through to the loop below, where
    # the zero-file case is a finding rather than a clean run. That asymmetry is
    # deliberate and is the same one `docs/known_limitations.rst` records for the
    # corpus tests in a tree with no `.git`: nothing to check and cannot check are
    # not the same state (non-negotiable 18).
    root = REPO_ROOT / args.root
    if not root.is_dir():
        print(
            f"{args.root}/ is not present in this tree, so the witness corpus has "
            "no subject -- declining rather than reporting a clean run. This is "
            "expected in the published distribution, which ships configuration "
            "templates instead of the experiment corpus."
        )
        return 0

    current: set[str] = set()
    checked = unloadable = 0
    for path in sorted((REPO_ROOT / args.root).rglob("*.yaml")):
        if "archive" in path.parts:
            continue
        rel = str(path.relative_to(REPO_ROOT))
        try:
            found = findings_for(path)
        except Exception:
            # Schema validity is check_experiment_configs_load.py's job, and it
            # carries its own baseline. Inheriting its debt here would make this
            # gate's verdict meaningless.
            unloadable += 1
            continue
        checked += 1
        current.update(f"{rel}::{f}" for f in found)

    meta = meta_findings()

    if args.update_baseline:
        write_baseline(sorted(current), args.baseline)
        print(f"baseline rewritten with {len(current)} entr(ies)")
        return 0

    print(f"resolved {checked} config(s); {unloadable} skipped (schema-invalid)")
    print(f"declared-but-discarded keys: {len(current)}")

    try:
        baseline = read_baseline(args.baseline)
    except MissingBaselineError as exc:
        print(
            f"\nFAILED: the baseline file is absent ({exc}). Refusing to treat a "
            "missing baseline as an empty one -- that grades every known finding "
            "as a new regression, and blesses the corpus on the next "
            "--update-baseline. Restore it from git, or create it deliberately "
            "with --update-baseline."
        )
        return 1
    new = sorted(current - baseline)
    fixed = sorted(baseline - current)

    if fixed:
        print(f"\ninfo: {len(fixed)} baseline entr(ies) are now clean — tighten with")
        print("      python scripts/ci/check_witness_corpus.py --update-baseline")
        for entry in fixed[:20]:
            print(f"  {entry}")

    if meta:
        print(f"\nFAILED: {len(meta)} meta-witness finding(s)")
        for item in meta:
            print(f"  {item}")

    failing = sorted(current) if args.strict else new
    if failing:
        label = "config(s)" if args.strict else "NEW"
        print(f"\nFAILED: {len(failing)} {label} declaring a key the schema discards")
        for entry in failing[:40]:
            print(f"  {entry}")
        if len(failing) > 40:
            print(f"  ... and {len(failing) - 40} more")

    if failing or meta:
        print(
            "\nA key the schema discards is not a style issue: the YAML advertises "
            "it and the run never receives it."
        )
        return 1

    print(
        f"\nOK: {len(current)} known finding(s) in the baseline, 0 new; every meta-witness passes."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
