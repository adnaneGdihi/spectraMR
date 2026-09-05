"""The warrant for the loss-weight SSOT: switching resolvers changes no arm's objective.

Every live arm is asserted to resolve, under the SSOT
(:func:`spectramr.models.losses.weights.build_loss_weight_table`), to exactly the weight it
was training at before the migration -- as frozen in :data:`BASELINE_GOLDEN`.

This is what licenses deleting the eight legacy resolvers and their three disagreeing
magic default tables. It only passes because
``scripts/migrations/materialize_loss_weights.py`` first wrote each arm's *actual*
weights into its YAML -- before that migration, 418 declarations resolved to a different
number than the one written in the file and 349 had never been computed at all.

**The baseline is frozen, not recomputed.** This test used to read the *migrated* YAML for
BOTH sides of the comparison: it re-derived the legacy weight from the same file the
migration had just rewritten. A migration that CHANGED an arm's objective therefore moved
both sides together and the proof stayed green -- which is exactly what happened. The first
attempt disabled 74 loss terms that arms were actively computing and reduced 52 arms to an
empty objective (``LossBuilder.validate()``: "No losses were built"), and this file passed.
A migration's proof must compare against the tree as it was BEFORE the migration ran, or it
proves only that the migration agrees with itself.

The second half of that bug: the test and the codemod each rolled their own family selector,
and rolled the same mistake (keying off ``training_mode``, never selecting ``folding``), so
the proof shared the defect it was meant to catch. There is now one selector
(:func:`spectramr.models.losses._legacy_weights.legacy_family_for`, used by the codemod), this
test no longer re-derives the legacy side at all, and
:func:`test_folding_strategy_set_matches_the_source` re-derives the folding set from the
strategy registry so it cannot drift away from the code.

If this test fails, either an arm's weight semantics changed (check the diff) or someone
re-introduced a default table.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import yaml as pyyaml

from spectramr.config.schemas.loss import LossConfigSchema
from spectramr.models.losses._legacy_weights import FOLDING_STRATEGIES
from spectramr.models.losses.weights import build_loss_weight_table, canonical_loss_name
from tests.utils.corpus import tracked_yamls
from tests.utils.repo_scripts import skip_if_public_export

REPO = Path(__file__).resolve().parents[2]
LIVE_TREES = ("experiments/inprogress", "experiments/active")
LOSS_LISTS = ("image_losses", "kspace_losses", "complex_losses")

#: Golden ``declared name -> canonical table key`` for the whole live corpus.
KEY_GOLDEN = Path(__file__).parent / "_loss_canonical_keys.txt"

#: What every declared loss ACTUALLY weighed before the migration. Frozen; emitted from the
#: pre-migration tree by the codemod. Never regenerate it from a migrated tree.
BASELINE_GOLDEN = Path(__file__).parent / "_pre_migration_weights.txt"

#: Arms whose EVERY declared loss resolved to 0.0 under the legacy resolvers. Their old
#: behaviour is not expressible: ``get_enabled_losses()`` gates on the LIST weight (so the
#: module WAS built) while the computer weighed it through ``lambda_<n>`` (so it contributed
#: 0.0), and nothing in the SSOT can say "build it, then weigh it zero". Disabling them all
#: -- the faithful freeze -- empties LossBuilder's stack and the arm cannot start.
#:
#: So the codemod leaves these arms untouched and their DECLARED weight now applies: a dead
#: loss comes back to life. **Accepted deliberately on 2026-07-12** -- the zeroed terms are
#: these arms' PRIMARY fidelity losses (``complex_mse`` x33, ``l2`` x11, ``complex_l1`` x1,
#: ``hfen`` x27), so they were training no fidelity objective at all; restoring the weight the
#: YAML always declared is the repair. See ``docs/loss_weight_ssot.rst``.
#:
#: Pinned so the set cannot GROW silently: derived from the frozen baseline, so a newly-dead
#: arm moves the count and trips this test instead of quietly joining the list.
EXPECTED_REVIVED = 45


def _arms() -> list[Path]:
    found: list[Path] = []
    for tree in LIVE_TREES:
        found.extend(tracked_yamls(REPO / tree))
    return found


ARMS = _arms()


def _baseline() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for line in BASELINE_GOLDEN.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        arm, loss, weight = line.split("\t")
        rows.setdefault(arm, {})[loss] = float(weight)
    return rows


BASELINE = _baseline()

#: An arm is "revived" iff every weight it carried in the baseline was 0.0.
REVIVED = {arm for arm, w in BASELINE.items() if all(v == 0.0 for v in w.values())}

#: Arms whose trained weight was deliberately CHANGED after the baseline was frozen.
#: The baseline still records the pre-change value -- correctly; it is a measurement
#: of history -- so the no-op assertion no longer applies and the arm is pinned to its
#: CURRENT declaration instead, exactly as ``REVIVED`` is. The assertion does not go
#: away; its authority moves.
#:
#: Editing ``_pre_migration_weights.txt`` instead is the tempting fix and the wrong
#: one: its header says a value there is a measurement, not a preference, and
#: rewriting it from a migrated tree is what made this proof circular the first time.
#:
#: Every entry cites the commit that made the change and a journal section, so the set
#: cannot be used to wave a genuine regression through.
SUPERSEDED: dict[str, str] = {
    "experiments/inprogress/mrixfields2026/task3/b34_ablate_euclidean.yaml": (
        "c8c2a7579 -- FisherRaoGeodesicStrategy is inline-loss: it reads "
        "training.fisher_rao_geodesic.lambda_l1 (=1.0) as its sole authority and never "
        "routes losses.image_losses through LossBuilder, so the 10.0 the materializer "
        "wrote came from applying the LossBuilder path to a strategy that does not use "
        "it. docs/deprecation_journal.rst 2026-08-09; pinned by "
        "test_fisher_rao_geodesic_strategy.py"
        "::test_effective_l1_weight_is_lambda_l1_not_image_losses."
    ),
    "experiments/inprogress/mrixfields2026/task3/b34_fisher_rao.yaml": (
        "c8c2a7579 -- the same inline-loss facade as b34_ablate_euclidean above."
    ),
    "experiments/inprogress/vf/exp_vf_01_subvoxel_superres_v2.yaml": (
        "550e3eec9 -- the five terms the baseline records (complex_mse, hfen, ssim, "
        "concomitant_phase_residual, phase_smoothness_complex) were REMOVED as "
        "advertised-but-inert: ConcreteMultiAcquisitionStrategy builds its own "
        "objective and never routes the declarative builder, so the 2026-06-28 run "
        "carried all five as permanently-empty columns. Declaring an unread loss is "
        "CLAUDE.md #15/#16, so the repair was deletion. The baseline cannot express "
        "'this term no longer exists'; the arm's current declaration is asserted "
        "instead. See issue #935 for the contradiction the later `losses:` block "
        "introduced."
    ),
}


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: str(p.relative_to(REPO)))
def test_ssot_weight_matches_the_weight_the_arm_actually_trained_at(arm: Path) -> None:
    doc = pyyaml.safe_load(arm.read_text())
    if not isinstance(doc, dict) or not isinstance(doc.get("losses"), dict):
        pytest.skip("no losses block")

    rel = str(arm.relative_to(REPO))
    if rel not in BASELINE:
        pytest.skip("no declarative loss entries")

    try:
        parsed = LossConfigSchema(**doc["losses"])
    except Exception as exc:  # pragma: no cover - a schema break is a different bug
        pytest.fail(f"{arm.name}: losses block does not parse: {exc}")

    # Raises on a conflicting dual declaration -- which is itself the assertion that no
    # arm declares one loss at two different weights.
    table = build_loss_weight_table(parsed)

    if rel in REVIVED or rel in SUPERSEDED:
        # Two ways the frozen baseline stops being the right authority for an arm:
        #
        #   REVIVED    -- every declared loss was dead, and freezing that would stop the
        #                 arm from starting, so the author's declaration applies instead.
        #   SUPERSEDED -- the arm was deliberately changed after the baseline was frozen,
        #                 with a commit and a journal entry (see the mapping above).
        #
        # Both assert the SAME thing -- the SSOT honours the DECLARED weight, not the
        # baseline -- so neither exemption is a way to stop asserting. The authority
        # moves; the guard stays.
        why = SUPERSEDED.get(
            rel, "every declared loss resolved to 0.0 before the migration"
        )
        kind = "superseded" if rel in SUPERSEDED else "revived"
        for list_name in LOSS_LISTS:
            for entry in doc["losses"].get(list_name) or []:
                if not isinstance(entry, dict) or entry.get("enabled", True) is False:
                    continue
                name = entry.get("name")
                if name is None:
                    continue
                declared = float(entry.get("weight", 1.0))
                ssot = table.weight(canonical_loss_name(name), iteration=10**9)
                assert math.isclose(ssot, declared, rel_tol=1e-9, abs_tol=1e-12), (
                    f"{arm.name}: '{name}' is a {kind} arm, so the SSOT must honour the "
                    f"declared {declared}, but it resolves to {ssot}.\n  {why}"
                )
        return

    # Past the warm-up gate: the gate is a schedule, not a weight, and is covered in
    # tests/unit/models/losses/test_weights.py.
    for loss, legacy in BASELINE[rel].items():
        ssot = table.weight(loss, iteration=10**9)
        assert math.isclose(ssot, legacy, rel_tol=1e-9, abs_tol=1e-12), (
            f"{arm.name}: '{loss}' trained at {legacy} before the migration (frozen in "
            f"{BASELINE_GOLDEN.name}) but the SSOT resolves it to {ssot}. The migration "
            "is not a no-op."
        )


def test_the_revived_set_has_not_grown() -> None:
    """Reviving a dead loss changes an arm's objective. Pin how many arms it happens to."""
    assert len(REVIVED) == EXPECTED_REVIVED, (
        f"{len(REVIVED)} arms now have an all-dead declared objective, expected "
        f"{EXPECTED_REVIVED}. Each one silently gains a live loss under the SSOT. "
        "Review the new arms, then update EXPECTED_REVIVED."
    )


def test_no_arm_is_migrated_to_an_empty_objective() -> None:
    """The migration must never disable an arm's LAST loss.

    ``LossBuilder.validate()`` raises ``No losses were built`` on an empty stack, so an arm
    whose every declaration got ``enabled: false`` cannot start at all -- it does not train
    a degraded objective, it dies. The first migration did this to 52 arms and every test
    stayed green, because nothing asserted the stack was non-empty. Now something does.

    Scoped to arms that HAD declarative losses (i.e. appear in the baseline): the inline-only
    arms carry no ``*_losses`` entries and legitimately build nothing.
    """
    empty = []
    for arm in ARMS:
        rel = str(arm.relative_to(REPO))
        if rel not in BASELINE:
            continue
        doc = pyyaml.safe_load(arm.read_text())
        try:
            parsed = LossConfigSchema(**doc["losses"])
        except Exception:  # pragma: no cover - covered by the no-op test above
            continue
        if not parsed.get_enabled_losses():
            empty.append(rel)

    assert not empty, (
        "these arms declare losses but resolve to an EMPTY enabled set -- "
        "LossBuilder.validate() will refuse to build and the run cannot start:\n  "
        + "\n  ".join(sorted(empty))
    )


def test_folding_strategy_set_matches_the_source() -> None:
    """``FOLDING_STRATEGIES`` must track the strategies that actually fold. Anti-drift.

    A strategy that adopts the ``_apply_builder_image_losses`` seam without landing in the
    set gets classified ``computer``, whose lambda scan cannot see a list weight and answers
    0.0 -- so its live losses read as "never computed" and get disabled. That is precisely
    how ``hfen``/``ms_ssim`` were switched off on 37 arms. Re-derive from source and compare.
    """
    import ast
    import importlib
    import inspect
    import textwrap

    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    fold_names = {"_apply_builder_image_losses", "fold_builder_image_losses"}

    def calls_fold(cls: type) -> bool:
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
        except (OSError, TypeError, SyntaxError):
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Skip the DEFINITION: its body legitimately calls the SSOT. Counting it makes
            # every inheritor of ReconstructionStrategy look like a folder.
            if node.name == "_apply_builder_image_losses":
                continue
            for sub in ast.walk(node):
                if not isinstance(sub, ast.Call):
                    continue
                fn = sub.func
                name = (
                    fn.attr
                    if isinstance(fn, ast.Attribute)
                    else getattr(fn, "id", None)
                )
                if name in fold_names:
                    return True
        return False

    derived = set()
    for key, path in TrainingStrategyFactory.STRATEGY_CLASS_PATHS.items():
        module, _, cls_name = path.rpartition(".")
        try:
            cls = getattr(importlib.import_module(module), cls_name)
        except Exception:  # pragma: no cover - an unimportable strategy is another bug
            continue
        if any(calls_fold(k) for k in cls.__mro__):  # an inherited fold still folds
            derived.add(key)

    assert derived == set(FOLDING_STRATEGIES), (
        "FOLDING_STRATEGIES is out of step with the code.\n"
        f"  fold but are not listed: {sorted(derived - set(FOLDING_STRATEGIES))}\n"
        f"  listed but do not fold : {sorted(set(FOLDING_STRATEGIES) - derived)}\n"
        "A folding strategy that is not listed has its list weights read as 0.0 and its "
        "live losses disabled by the migration."
    )


def test_every_superseded_entry_is_real_and_cites_its_commit() -> None:
    """The exemption set must not outlive what it exempts, nor grow without evidence.

    An allow-list whose entries nobody re-checks is how a genuine regression gets
    waved through. Three conditions, all cheap: the arm still exists, the baseline
    still carries it (otherwise nothing is being exempted and the row is dead), and
    the reason names the commit that made the change.
    """
    skip_if_public_export("experiments/ does not ship, so every superseded entry reads as a deleted arm")
    import re

    for rel, why in SUPERSEDED.items():
        assert (REPO / rel).exists(), f"{rel} no longer exists -- drop the exemption"
        assert rel in BASELINE, (
            f"{rel} is not in the frozen baseline, so the no-op proof never asserted "
            "anything about it and this exemption is dead"
        )
        assert re.match(r"^[0-9a-f]{7,40} -- ", why), (
            f"{rel}: the reason must open with the commit SHA that changed the arm, "
            f"got {why[:40]!r}"
        )
    overlap = set(SUPERSEDED) & REVIVED
    assert (
        not overlap
    ), f"an arm cannot be both revived and superseded: {sorted(overlap)}"


def test_the_corpus_is_actually_covered() -> None:
    """Guard against the parametrization silently collapsing to zero arms."""
    skip_if_public_export("experiments/ does not ship; the live corpus is empty here")
    assert len(ARMS) > 500, f"expected the live corpus, found {len(ARMS)} arms"


def _declared_names() -> set[str]:
    """Every loss name declared anywhere in the live corpus (list entries + ``lambda_*``)."""
    names: set[str] = set()
    for arm in ARMS:
        try:
            doc = pyyaml.safe_load(arm.read_text())
        except Exception:  # pragma: no cover - a YAML break is a different bug
            continue
        losses = doc.get("losses") if isinstance(doc, dict) else None
        if not isinstance(losses, dict):
            continue
        for list_name in LOSS_LISTS:
            for entry in losses.get(list_name) or []:
                if isinstance(entry, dict) and entry.get("name"):
                    names.add(str(entry["name"]))
        for section in losses.values():
            if isinstance(section, dict):
                names.update(f for f in section if f.startswith("lambda_"))
    return names


def _read_golden() -> dict[str, str]:
    rows = {}
    for line in KEY_GOLDEN.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, canon = line.partition(" -> ")
            rows[name] = canon
    return rows


def test_canonical_keys_are_pinned() -> None:
    """An alias change silently RE-KEYS the weight table, and the no-op test cannot see it.

    ``test_ssot_weight_matches_the_weight_the_arm_actually_trained_at`` canonicalises BOTH
    sides through :func:`canonical_loss_name`, so editing the alias table moves the table's
    key and the lookup key *together* -- the 645-arm proof stays green while every consumer
    that resolves the OLD key now finds nothing declared (and ``weights.py`` RAISES). The
    "proven no-op" shield therefore does not cover an alias mutation, which is exactly what
    issue #191 (``gradient_penalty`` is an alias of R1) requires us to perform.

    29 of the 93 declared names canonicalise to a *different* key. The sharpest is
    ``lambda_gp -> r1_regularization``: WGAN-GP's weight is filed under R1's key. Dropping
    that alias re-keys it to ``gradient_penalty`` -- a change this test makes visible and the
    other one cannot.
    """
    current = {n: canonical_loss_name(n) for n in _declared_names()}

    if os.environ.get("SPECTRAMR_UPDATE_LOSS_KEY_GOLDEN") == "1":
        header = "\n".join(
            line for line in KEY_GOLDEN.read_text().splitlines() if line.startswith("#")
        )
        body = "".join(f"{n} -> {c}\n" for n, c in sorted(current.items()))
        KEY_GOLDEN.write_text(header + "\n" + body)
        pytest.skip("golden regenerated")

    golden = _read_golden()
    remapped = {
        n: (golden[n], c) for n, c in current.items() if n in golden and golden[n] != c
    }
    assert not remapped, (
        "A loss name now files its weight under a DIFFERENT canonical key:\n"
        + "\n".join(
            f"  {n}: {was!r} -> {now!r}" for n, (was, now) in sorted(remapped.items())
        )
        + "\n\nEvery consumer resolving the old key now finds nothing declared. Co-edit them "
        "(config/schemas/loss.py, pipelines/training_loop.py, reporting/plotters/generative/"
        "gan_diagnostics.py), then regenerate with SPECTRAMR_UPDATE_LOSS_KEY_GOLDEN=1."
    )


def test_warmup_gate_shape_is_pinned() -> None:
    """Pin what the warm-up gate does, so reconciling the resolvers' ``iteration`` default is visible.

    The no-op test asserts only at ``iteration=10**9`` -- past the gate -- so it is blind to
    the gate entirely. Meanwhile the two callers of the SSOT disagree on the default:
    ``BaseTrainingStrategy._get_loss_weight`` passes ``iteration=1_000_000`` (gate OFF) and
    ``BaseLossComputer._get_loss_weight`` defaults to ``iteration=0`` (gate ON). Both are
    faithful ports of pre-SSOT per-family behaviour, so the divergence is legacy, not a
    regression -- but reconciling it is a NUMERIC change that no current test can observe.

    This asserts the gate's shape: a gated term is 0.0 before ``warmup_iterations`` and its
    full weight after; an ungated term is unaffected by ``iteration``.
    """
    skip_if_public_export("experiments/ does not ship; there are no arms whose weights to check")
    checked = 0
    for arm in ARMS:
        doc = pyyaml.safe_load(arm.read_text())
        if not isinstance(doc, dict) or not isinstance(doc.get("losses"), dict):
            continue
        try:
            table = build_loss_weight_table(LossConfigSchema(**doc["losses"]))
        except Exception:
            continue  # covered by the no-op test above

        # ``warmup_iterations: 0`` is a deliberate opt-out (c_mno.yaml and the cs_mno
        # operator arms: they are wall-clock-limited, would never exit a 1000-iter warm-up,
        # and l1*0 as the only surviving term is the dead-loss collapse). A gated term with
        # no warm-up window is correctly ungated.
        gate_active = table.warmup_iterations > 0

        for name, spec in table.items():
            early = table.weight(name, iteration=0)
            late = table.weight(name, iteration=table.warmup_iterations)
            if gate_active and spec.warmup_gated and spec.enabled and spec.weight:
                assert early == 0.0, (
                    f"{arm.name}: '{name}' is warm-up gated but resolves to {early} at "
                    "iteration=0 -- the gate is not firing."
                )
                assert math.isclose(late, spec.weight), (
                    f"{arm.name}: '{name}' resolves to {late} at the end of warm-up, "
                    f"expected its declared weight {spec.weight}."
                )
            else:
                assert early == late, (
                    f"{arm.name}: '{name}' is NOT warm-up gated yet its weight changes with "
                    f"iteration ({early} -> {late})."
                )
            checked += 1

    assert checked > 1000, f"expected the live corpus's weights, checked only {checked}"
