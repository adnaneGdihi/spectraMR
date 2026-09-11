"""Tests for the loss-lambda -> domain-list migrator.

This migrator deletes config keys across a whole cohort, so its refusals matter
far more than its happy path: a wrong deletion silently changes the objective of
every arm it touches, and the resulting run still trains. Each case here is one of
the violations the migrator was planted with, kept as a test so a future edit that
removes a guard turns this file red instead of the corpus.

The fixture is a REAL cohort arm rather than a hand-built stub: the migrator's
whole safety argument runs through ``TrainingSettings.from_yaml`` and
``build_loss_weight_table``, and a stub config resolves neither.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import shutil
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_MOD_PATH = _REPO / "scripts" / "migrations" / "migrate_loss_lambdas_to_domain_lists.py"
_ARM = (
    _REPO
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "attention_shootout"
    / "experiment_11_attention_none.yaml"
)


def _load():
    spec = importlib.util.spec_from_file_location("_mig", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mig():
    return _load()


@pytest.fixture
def arm(tmp_path: Path) -> Path:
    """A copy of a migrated arm with one redundant lambda put BACK.

    ``log_spectral`` is declared on ``kspace_losses`` at 0.1; re-adding
    ``lambda_log_spectral: 0.1`` recreates the dual declaration the migrator
    exists to collapse, at the same value (a different value is a separate case
    below, and does not even load).
    """
    dst = tmp_path / _ARM.name
    shutil.copy(_ARM, dst)
    text = dst.read_text()
    assert "lambda_log_spectral" not in text, "fixture premise: the arm is migrated"
    dst.write_text(
        text.replace("  reconstruction:\n", "  reconstruction:\n    lambda_log_spectral: 0.1\n", 1)
    )
    return dst


def test_removes_a_redundant_lambda_and_touches_nothing_else(mig, arm):
    before = arm.read_text()
    status, removed = mig.migrate(arm, apply=True)
    assert status == "MIGRATED", status
    assert removed == ["reconstruction.lambda_log_spectral"]
    # every other line byte-identical: the deletion is the ONLY edit
    assert arm.read_text().splitlines() == [
        ln for ln in before.splitlines() if ln.strip() != "lambda_log_spectral: 0.1"
    ]


def test_keeps_a_lambda_that_no_domain_list_declares(mig, arm):
    """``pre_dc_kspace`` is a strategy-inline term with no registry entry.

    It CANNOT have a domain-list entry, so its lambda is the only declaration
    there is -- deleting it would zero a live 0.3-weighted term.
    """
    mig.migrate(arm, apply=True)
    assert "lambda_pre_dc_kspace: 0.3" in arm.read_text()


def test_keeps_an_explicit_zero_whose_schema_default_is_not_zero(mig, arm):
    """``lambda_bloch_residual: 0.0`` pins a term OFF; its schema default is 1.0."""
    from spectramr.config.schemas.loss import PhysicsLossesConfig

    assert PhysicsLossesConfig().lambda_bloch_residual == 1.0, "premise"
    mig.migrate(arm, apply=True)
    assert "lambda_bloch_residual: 0.0" in arm.read_text()


def test_refuses_an_arm_whose_two_surfaces_disagree(mig, arm):
    """A disagreement is the user's to resolve -- never silently 'fixed'.

    ``build_loss_weight_table`` raises on it, so the arm does not load and the
    migrator reports SKIP rather than picking a winner.
    """
    arm.write_text(arm.read_text().replace("lambda_log_spectral: 0.1", "lambda_log_spectral: 0.9"))
    before = arm.read_text()
    status, _ = mig.migrate(arm, apply=True)
    assert status.startswith("SKIP unloadable"), status
    assert arm.read_text() == before, "a refused arm must not be edited"


def test_rolls_back_when_a_removal_would_change_the_resolved_table(mig, arm, monkeypatch):
    """The verifier, not the loader, is what catches an over-broad removal set.

    Planted by widening BOTH candidate surfaces to include the load-bearing
    ``pre_dc_kspace``: the file still loads, so only the before/after table
    comparison can catch it.

    Widening the domain-list pool alone is a DIFFERENT plant (see
    ``test_reports_a_candidate_the_audit_does_not_flag``) -- the audit gate would
    stop the removal first and this test would pass without the verifier running.
    """
    real = mig._declared_in_domain_lists
    monkeypatch.setattr(mig, "_declared_in_domain_lists", lambda p: real(p) | {"pre_dc_kspace"})
    real_sites = mig._audit_reported_sites
    monkeypatch.setattr(
        mig,
        "_audit_reported_sites",
        lambda p: real_sites(p) | {("reconstruction", "pre_dc_kspace")},
    )
    before = arm.read_text()
    status, removed = mig.migrate(arm, apply=True)
    assert status.startswith("ROLLBACK weight table changed"), status
    assert "pre_dc_kspace" in status
    assert "reconstruction.lambda_pre_dc_kspace" in removed
    assert arm.read_text() == before, "a rolled-back arm must be restored byte-for-byte"


def test_reports_a_candidate_the_audit_does_not_flag(mig, arm, monkeypatch):
    """Non-negotiable 17: the health check owns the pairing rule, so it gates removal.

    Plant for the gate itself. Widening only the domain-list pool makes
    ``pre_dc_kspace`` look removable to the text scan while
    ``dual_surface_loss_declarations`` still reports nothing for it -- exactly the
    divergence the linkage exists to catch.

    The verdict must NAME it. Without the gate the key is deleted and the table
    verifier rescues the file with a ``ROLLBACK``, which leaves the same bytes on
    disk -- so asserting only that the key survived would pass either way.
    """
    real = mig._declared_in_domain_lists
    monkeypatch.setattr(mig, "_declared_in_domain_lists", lambda p: real(p) | {"pre_dc_kspace"})

    status, removed = mig.migrate(arm, apply=True)

    assert "reconstruction.lambda_pre_dc_kspace" not in removed, removed
    assert "unreported" in status, status
    assert "reconstruction.lambda_pre_dc_kspace" in status, status
    assert "lambda_pre_dc_kspace: 0.3" in arm.read_text()


def test_an_unparsable_audit_site_is_fatal(mig, arm, monkeypatch):
    """A site the migrator cannot parse must stop the run, not be skipped.

    Silently dropping it makes an audit-reported duplicate unremovable with no
    verdict naming it -- the arm reports ``OK already migrated`` while both
    surfaces still declare the term (non-negotiable 3).
    """
    from spectramr.infrastructure.validation import config_health_checker

    monkeypatch.setattr(
        config_health_checker,
        "dual_surface_loss_declarations",
        lambda _losses: [("log_spectral", 0.1, ["losses.log_spectral"], ["kspace_losses"])],
    )
    with pytest.raises(SystemExit, match="unparsable lambda site"):
        mig.migrate(arm, apply=True)


def test_a_migrated_arm_that_also_diverges_exits_1_and_prints_the_verdict(
    mig, arm, monkeypatch, capsys
):
    """A divergence must survive the reporting layer, not just the verdict string.

    This arm both migrates (``lambda_log_spectral`` is reported and removable) and
    diverges (the widened ``pre_dc_kspace`` is not reported), so it is filed under
    the ``MIGRATED`` head. Two guards keyed on that head used to lose it: the detail
    line was printed only ``if head != "MIGRATED"``, and the exit code was derived
    from the ``ROLLBACK``/``REFUSE`` counts -- so a cohort run reported success while
    the verdict it never printed said the two surfaces disagree.

    Both are keyed on :data:`UNREPORTED_MARKER` instead, which is why one constant
    owns the token (non-negotiable 17).
    """
    real = mig._declared_in_domain_lists
    monkeypatch.setattr(mig, "_declared_in_domain_lists", lambda p: real(p) | {"pre_dc_kspace"})
    monkeypatch.setattr(sys, "argv", ["migrate", str(arm.parent), "--apply"])

    try:
        code = mig.main()
    finally:
        # main() disables logging process-wide and never restores it.
        logging.disable(logging.NOTSET)

    out = capsys.readouterr().out
    assert code == 1, out
    assert "MIGRATED" in out, out
    assert mig.UNREPORTED_MARKER in out, out
    assert "reconstruction.lambda_pre_dc_kspace" in out, out


def test_a_second_run_is_a_no_op(mig, arm):
    mig.migrate(arm, apply=True)
    status, removed = mig.migrate(arm, apply=True)
    assert status == "OK already migrated"
    assert removed == []


# --------------------------------------------------------------------------
# Plants for the three defects the first cohort run actually shipped (836d29785).
# Each is the SHAPE that got past the guards, not a restatement of a case above:
# the migration was verified against the SSOT weight table, and all three are
# invisible to that table by construction.
#
# These use the REAL affected arms as fixtures. A synthesised fixture was tried
# first and was silently vacuous twice over -- inserting a second ``image_losses:``
# key made PyYAML drop the planted entry, so the name never entered the removable
# set and the assertion passed against a migrator that did nothing at all.
# --------------------------------------------------------------------------

# A FROZEN byte-copy, not the live arm: #1923 makes this arm's ``lambda_perceptual``
# removable and this same change migrates it, so a test aimed at the live path would
# go vacuous the moment the duplicate it needs is gone. The copy keeps the real
# dual-surface shape (the file's header records why a synthesised one does not).
_PERCEPTUAL_ARM = _REPO / "tests/fixtures/configs/dual_surface_perceptual_arm_frozen.yaml"
_DIFFUSION_ARM = _REPO / "experiments/inprogress/kspace_filling" / "experiment_130_ti_ccd.yaml"
# Writes ``losses.reconstruction.lambda_l1: 0.1`` AND declares ``image_losses[l1]`` at
# the same weight, so removing the category key leaves the SSOT table identical while
# a raw reader sees 0.1 -> 10.0 (the schema default).
#
# FROZEN, no longer the live arm. It was ``inprogress/ldm_two_stage_ulf_to_hf/
# stage2_ldm_ulf_to_hf.yaml``, kept live on the argument that the census refuses to
# migrate it so no cohort sweep can take the shape away. That argument covers the
# dual-surface half and not the rest: the tests below turn on whether the arm
# declares ``losses.gan`` and which strategy it resolves to, and an owner is free to
# change either without touching a lambda. A frozen copy also lets one fixture carry
# BOTH sides of each pair -- the live arm can only ever be one of them.
_RAW_L1_ARM = _REPO / "tests/fixtures/configs/dual_surface_l1_arm_frozen.yaml"


def _copy(src: Path, tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    dst = tmp_path / src.name
    shutil.copy(src, dst)
    return dst


def test_leaves_a_lambda_outside_the_losses_surface_alone(mig, tmp_path, monkeypatch):
    """``training.diffusion.lambda_*`` is not this migration's surface.

    ``training:`` has a ``diffusion:`` child too, and the block tracker matched any
    ``CATEGORY_BLOCKS`` name at any depth -- so the first cohort run deleted
    ``training.diffusion.lambda_mse`` on 2 arms. The verifier reads only ``losses``,
    so nothing went red.

    Planted with ``complex_l1`` rather than ``mse`` deliberately: ``mse`` is on the
    deny-list, which would mask a regression in the SCOPE guard being tested here.

    The audit gate would mask it the same way -- this arm declares no
    ``losses.diffusion.lambda_complex_l1``, so a broken scope guard would report the
    planted key as merely unreported and this test would stay green. The gate is
    therefore widened to name the pair, leaving the scope guard as the only thing
    between the plant and a deletion.
    """
    arm = _copy(_DIFFUSION_ARM, tmp_path)
    assert "complex_l1" in mig._declared_in_domain_lists(arm), "premise: removable"
    real_sites = mig._audit_reported_sites
    monkeypatch.setattr(
        mig,
        "_audit_reported_sites",
        lambda p: real_sites(p) | {("diffusion", "complex_l1")},
    )
    arm.write_text(
        arm.read_text().replace(
            "\ntraining:\n", "\ntraining:\n  diffusion:\n    lambda_complex_l1: 0.0\n", 1
        )
    )

    _status, removed = mig.migrate(arm, apply=True)

    assert "diffusion.lambda_complex_l1" not in removed, removed
    assert "    lambda_complex_l1: 0.0\n" in arm.read_text(), "key outside `losses:` deleted"


def test_keeps_diffusion_lambda_mse_even_though_a_domain_list_declares_mse(mig, tmp_path):
    """The SSOT maps ``lambda_mse`` -> ``l2``; the diffusion resolver does not.

    ``_resolve_diffusion_weight`` reads the literal path ``losses.diffusion.lambda_mse``.
    Because the table canonicalizes the name onto a DIFFERENT term, the arm's
    ``image_losses`` ``mse`` entry looks like a duplicate declaration when it is not
    one -- ``mse`` really is in the removable set here, so only the deny-list stops it.
    """
    arm = _copy(_DIFFUSION_ARM, tmp_path)
    assert "mse" in mig._declared_in_domain_lists(arm), "premise: mse is removable"
    assert "    lambda_mse: 1.0\n" in arm.read_text(), "premise: declared under losses.diffusion"

    status, removed = mig.migrate(arm, apply=True)

    assert "diffusion.lambda_mse" not in removed, removed
    assert "    lambda_mse: 1.0\n" in arm.read_text()
    # The verdict must SAY a duplicate survived. This used to report
    # "OK already migrated" -- the same string a genuinely clean arm returns -- so a
    # cohort sweep counted this arm as done while both surfaces still declared `mse`.
    assert status.startswith("DENIED"), status
    assert "lambda_mse" in status, status


def test_removes_lambda_perceptual_now_that_the_builder_reads_the_weight_table(mig, tmp_path):
    """The inverse of the denial this arm used to carry (#1923).

    ``LossBuilder._build_composite_gan`` used to construct ``gan_composite`` from the
    RAW ``recon_config.lambda_perceptual``, whose schema default is 10.0 while its
    sibling ``enable_perceptual`` defaults False -- so deleting an explicit 0.1 was a
    latent 100x jump for any arm that declares ``losses.gan``, and that constructor
    argument never appeared in the weight table for the verifier to see. That builder
    now resolves perceptual from the table, so the duplicate is genuinely inert.

    The assertion that matters is the last one: removing the category key must leave
    the DECLARED weight untouched. A test that only checked the line was gone would
    pass just as well against a migrator that silently dropped the loss.
    """
    arm = _copy(_PERCEPTUAL_ARM, tmp_path)
    assert "perceptual" in mig._declared_in_domain_lists(arm), "premise: removable"
    assert "    lambda_perceptual: 0.1\n" in arm.read_text(), "premise: dual-surface"

    status, removed = mig.migrate(arm, apply=True)

    assert "reconstruction.lambda_perceptual" in removed, removed
    assert "    lambda_perceptual: 0.1\n" not in arm.read_text()
    assert status.startswith("MIGRATED"), status

    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.weights import build_loss_weight_table

    spec = build_loss_weight_table(TrainingSettings.from_yaml(str(arm)).losses).get("perceptual")
    assert spec is not None and spec.enabled, "the surviving declaration must still stand"
    assert spec.weight == 0.1, spec


def test_an_arm_with_both_a_denial_and_a_divergence_names_both(mig, tmp_path, monkeypatch):
    """Two different kept-field states must not share one verdict.

    Aimed at the ``losses.diffusion.lambda_mse`` denial, which is the one that
    survives #1923 (the perceptual entry this test used to ride on is retired). The
    arm carries ``reconstruction.lambda_perceptual: 0.0`` but declares ``perceptual``
    in no domain list, so widening the pool with that name manufactures a candidate
    the audit will not report -- putting both states on one arm, the case a
    ``DENIED``-first verdict answered by dropping the divergence entirely and
    reporting a reasoned exemption where a disagreement also existed.

    ``UNREPORTED`` takes the head because it is the actionable one: the denial recurs
    on every run of a fully migrated cohort, while the divergence means the text scan
    and the audit stopped agreeing.
    """
    arm = _copy(_DIFFUSION_ARM, tmp_path)
    assert "perceptual" not in mig._declared_in_domain_lists(arm), "premise: not declared"
    assert "    lambda_perceptual: 0.0\n" in arm.read_text(), "premise: category key present"
    real = mig._declared_in_domain_lists
    monkeypatch.setattr(mig, "_declared_in_domain_lists", lambda p: real(p) | {"perceptual"})

    status, removed = mig.migrate(arm, apply=True)

    assert removed == [], removed
    assert status.startswith("UNREPORTED"), status
    assert "reconstruction.lambda_perceptual" in status, status
    assert "diffusion.lambda_mse" in status, status
    assert "    lambda_perceptual: 0.0\n" in arm.read_text()
    assert "    lambda_mse: 1.0\n" in arm.read_text()


def test_empty_block_guard_scans_past_the_first_category_block(mig):
    """The guard used to ``return`` on the first ``CATEGORY_BLOCKS`` key in the file.

    A non-empty ``training.diffusion:`` therefore masked an emptied ``losses.diffusion:``
    further down -- which is exactly how 2 arms shipped with an empty block.
    """
    masked = "training:\n  diffusion:\n    type: cold\nlosses:\n  diffusion:\n  image_losses: []\n"
    assert mig._empties_a_block(masked) == "diffusion"

    healthy = (
        "training:\n  diffusion:\n    type: cold\nlosses:\n  diffusion:\n    lambda_mse: 1.0\n"
    )
    assert mig._empties_a_block(healthy) is None


def test_removes_lambda_l1_now_that_the_builder_reads_the_weight_table(mig, tmp_path):
    """The inverse of the ROLLBACK this arm used to carry (#1949).

    This test asserted the opposite until the builder moved. ``gan_composite.lambda_l1``
    was pinned in :func:`_raw_reader_census` because
    ``LossBuilder._build_composite_gan`` passed ``recon_config.lambda_l1`` straight
    into the ctor, where the weight table could not see it -- so removing the
    duplicate category key was a silent ``0.1 -> 10.0`` jump, and the census was the
    only thing standing in front of it. That builder now resolves the weight through
    ``LossBuilder._declared_weight`` -> ``_loss_weight_table()``, the pin left with
    the reader it covered, and the removal is a genuine no-op.

    The fixture keeps its ``losses.gan`` block. It is no longer load-bearing for the
    verdict -- nothing is gated on it here any more -- but it keeps this arm on the
    branch that *does* build the composite, which is the case where a regression
    would show. An arm without the block could not tell a fixed builder from a
    deleted one.

    The assertion that matters is the last one, exactly as in the ``perceptual``
    sibling above: removing the category key must leave the DECLARED weight
    untouched. A test that only checked the line was gone would pass just as well
    against a migrator that silently dropped the loss.

    The invariant this test used to carry -- *the census sees removals the table
    cannot* -- is not lost with it. It is planted at
    ``test_a_disentangled_arm_rolls_back_because_a_live_raw_reader_still_sees_the_field``,
    over the one raw reader that is still live.
    """
    arm = _make_gan_declared(_copy(_RAW_L1_ARM, tmp_path))
    assert mig._denied("reconstruction", "l1") is None, "premise: no deny-list entry for l1"
    assert "    lambda_l1: 0.1\n" in arm.read_text(), "premise: dual-surface"

    status, removed = mig.migrate(arm, apply=True)

    assert "reconstruction.lambda_l1" in removed, removed
    assert "    lambda_l1: 0.1\n" not in arm.read_text()
    assert status.startswith("MIGRATED"), status

    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.weights import build_loss_weight_table

    spec = build_loss_weight_table(TrainingSettings.from_yaml(str(arm)).losses).get("l1")
    assert spec is not None and spec.enabled, "the surviving declaration must still stand"
    assert spec.weight == 0.1, spec


_DISENTANGLED = (
    "spectramr.infrastructure.training.strategies."
    "disentangled_strategy.DisentangledTrainingStrategy"
)


def _make_disentangled(arm: Path) -> Path:
    """Repoint an arm's strategy at the one computer that raw-reads the field.

    Exactly ONE knob moves. ``training.strategy_class`` outranks ``training_mode``
    in :meth:`TrainingStrategyFactory.get_strategy_class`, so the arm's
    ``training_mode: diffusion`` is left in place deliberately -- if the pin ever
    starts reading the deprecated key instead, this arm stops resolving to the
    disentangled strategy and the test goes red.
    """
    text = arm.read_text()
    before = text
    text = re.sub(r"(?m)^(\s*strategy_class:\s*).*$", rf"\g<1>{_DISENTANGLED}", text, count=1)
    assert text != before, "fixture premise: the arm declares a strategy_class"
    arm.write_text(text)
    return arm


def test_a_disentangled_arm_rolls_back_because_a_live_raw_reader_still_sees_the_field(
    mig, tmp_path
):
    """#1923 retired the perceptual DENY; this is the hole that retirement left.

    The DENY refused *every* ``lambda_perceptual`` removal, so it happened to cover
    a reader nobody had classified. Retiring it on the strength of one builder is
    only safe if the others are dead -- and one is not:
    ``UnifiedDisentangledLossComputer._get_loss_weight`` maps ``perceptual ->
    lambda_perceptual`` and ``_stack_components`` multiplies by the result, so for
    that computer the raw field is the trained weight and deleting an explicit 0.1
    is the same latent 100x jump, through a different door.

    This is the planted violation (NN15) for the conditional pin. Its falsifier is
    ``test_removes_lambda_perceptual_now_that_the_builder_reads_the_weight_table``
    above, which runs the SAME fixture with its original ``strategy_class`` and
    must still report ``MIGRATED``. The pair is the whole claim: one knob, two
    verdicts. If a later edit makes the pin unconditional, this test stays green
    and that one goes red -- which is why both must be kept.

    0 corpus arms have this shape today (3 resolve to the disentangled strategy,
    none declares perceptual on either surface), so the fixture is manufactured on
    purpose: the pin exists for the first arm that does, not for one that already
    does.
    """
    arm = _make_disentangled(_copy(_PERCEPTUAL_ARM, tmp_path))
    assert "    lambda_perceptual: 0.1\n" in arm.read_text(), "premise: dual-surface"

    status, removed = mig.migrate(arm, apply=True)

    assert "reconstruction.lambda_perceptual" in removed, "premise: still a candidate"
    assert status.startswith("ROLLBACK raw (non-SSOT) reader changed"), status
    assert "unified_disentangled.lambda_perceptual" in status, status
    assert "    lambda_perceptual: 0.1\n" in arm.read_text(), "rollback did not restore"


def test_the_disentangled_pin_is_silent_for_every_arm_that_cannot_reach_that_computer(
    mig, tmp_path
):
    """The census entry must be ``None`` off the disentangled path, not the value.

    Pinning the raw field unconditionally would re-deny the removals #1923 exists to
    unblock -- the arm would merely move ``DENIED -> ROLLBACK``, the same blocked
    state under a new string. Asserted on the census directly rather than through a
    verdict, because a verdict can come out right for the wrong reason.
    """
    from spectramr.config.settings import TrainingSettings

    plain = _copy(_PERCEPTUAL_ARM, tmp_path)
    assert mig._raw_reader_census(plain)["unified_disentangled.lambda_perceptual"] is None

    swapped = _make_disentangled(_copy(_PERCEPTUAL_ARM, tmp_path / "d"))
    declared = TrainingSettings.from_yaml(str(swapped)).losses.reconstruction.lambda_perceptual
    assert mig._raw_reader_census(swapped)["unified_disentangled.lambda_perceptual"] == declared
    assert declared == 0.1, "premise: the arm declares a NON-default weight"


# ---------------------------------------------------------------------------
# Per-reader gates (#1928). Each gate is planted TWICE -- once where the reader
# runs and once where it does not -- because a pin that is always on and a pin
# that is always off each satisfy exactly one half. Only the pair is the claim.
#
# ONE gate lives here now: the disentangled weight map. The composite-GAN gate and
# its pair of plants were retired with the four ``gan_composite.lambda_*`` pins in
# #1949 -- the builder resolves those weights through the weight table, so there is
# no raw reader left to gate. ``_make_gan_declared`` survives the retirement: two
# tests still need an arm that reaches ``_build_composite_gan``.
# ---------------------------------------------------------------------------


def _make_gan_declared(arm: Path) -> Path:
    """Give an arm the ``losses.gan`` block ``_build_composite_gan`` is guarded on.

    One key, and it is the smallest one that makes the block parse: YAML reads a
    childless mapping key as ``null``, so ``gan:`` alone would leave
    ``losses.gan is None`` and change nothing -- the same mechanism
    :func:`_empties_a_block` refuses.
    """
    text = arm.read_text()
    patched = text.replace(
        "  reconstruction:\n", "  gan:\n    lambda_adv: 1.0\n  reconstruction:\n", 1
    )
    assert patched != text, "fixture premise: the arm has a losses.reconstruction block"
    arm.write_text(patched)
    return arm


def test_the_gan_declared_row_stays_unconditional(mig, tmp_path):
    """Removing the ``gan:`` block itself must still move a census row.

    This row's justification changed with #1949 and it is worth stating, because a
    row kept for a reason that has evaporated is how a census accretes dead pins.

    It used to be the safety net under the four gated ``gan_composite.*`` pins: gate
    them on ``losses.gan`` and a rewrite that emptied that block would silence all
    four *together*, so this row was the one that still moved. Those four pins are
    gone (the builder reads the weight table now), and with them that argument.

    What it covers now is its own, and the weight table cannot see it:
    ``_build_composite_gan`` has two call sites and both are guarded on ``losses.gan``
    being truthy, so emptying the block stops the composite being built **at all** --
    a behaviour change no lambda in the table registers, because the table has no
    notion of "is there a gan block". Nothing else in the census would move.

    Mutation note: gating THIS row on a ``builds_composite``-shaped predicate would
    be a semantic no-op, since such a gate's condition is this row's own value. The
    shape it does catch is a row gated on something different (a strategy, a schema
    field), which is the way a future edit would actually break it.
    """
    declared = _make_gan_declared(_copy(_RAW_L1_ARM, tmp_path))
    with_gan = mig._raw_reader_census(declared)

    stripped = declared.read_text().replace("  gan:\n    lambda_adv: 1.0\n", "", 1)
    declared.write_text(stripped)
    without = mig._raw_reader_census(declared)

    assert with_gan["losses.gan_declared"] != without["losses.gan_declared"]
    assert with_gan != without, "the census must not go quiet when the gan block leaves"


def test_an_unresolvable_strategy_consults_the_disentangled_pin(mig):
    """The disentangled gate's fail-closed clause, which lost its witness to #1964.

    ``_reaches_disentangled_weight_map`` asks :class:`TrainingStrategyFactory` to
    resolve the arm and returns ``True`` when that raises, because the two errors
    are not symmetric: over-pinning refuses a removal, which is reported with the
    field named, while under-pinning silently multiplies a declared 0.1 by 100.

    That clause was covered only by proximity until now. The retired
    ``_reaches_diffusion_banner`` gate resolved a strategy the same way and carried
    its own unresolvable-strategy plant, so retiring the banner took the witness for
    *this* gate's ``except`` with it -- a mutation matrix run after the retirement is
    what surfaced it. No YAML expresses the shape (any config that loads resolves
    to something), so it is planted with a stub whose attribute access raises:
    an untested ``except`` is indistinguishable from an absent one.
    """

    class _Hostile:
        def __getattr__(self, name: str):
            raise RuntimeError("config surface unavailable")

    assert mig._reaches_disentangled_weight_map(_Hostile()) is True


def test_the_disentangled_pin_follows_a_subclass_not_the_class_name(mig, monkeypatch):
    """``issubclass``, not ``cls.__name__ ==``, and the difference is a live weight.

    A subclass of :class:`DisentangledTrainingStrategy` inherits the ``__init__``
    that constructs ``UnifiedDisentangledLossComputer``, and therefore inherits the
    raw ``lambda_perceptual`` read with it. A name comparison answers ``False`` for
    every such subclass, so the pin would fall silent on exactly the arms whose
    weight is still resolved from the raw field -- the under-pinning direction, the
    expensive one.

    Both halves are asserted here because either alone is satisfied by a constant:
    a gate stuck at ``True`` passes the subclass clause, and one stuck at ``False``
    passes the unrelated-class clause. Only the pair distinguishes them.
    """
    from spectramr.infrastructure.training import strategy_factory as sf
    from spectramr.infrastructure.training.strategies.disentangled_strategy import (
        DisentangledTrainingStrategy,
    )

    class _VendorDisentangled(DisentangledTrainingStrategy):
        """A subclass that inherits the computer, and so inherits the raw read."""

    assert _VendorDisentangled.__name__ != "DisentangledTrainingStrategy", "premise"

    monkeypatch.setattr(
        sf.TrainingStrategyFactory, "get_strategy_class", lambda self, cfg: _VendorDisentangled
    )
    assert mig._reaches_disentangled_weight_map(object()) is True, "a subclass reaches the reader"

    monkeypatch.setattr(sf.TrainingStrategyFactory, "get_strategy_class", lambda self, cfg: str)
    assert mig._reaches_disentangled_weight_map(object()) is False, "an unrelated class does not"


# ---------------------------------------------------------------------------
# #1948 -- a removed lambda takes the comment block that documented it.
#
# `_rewrite` is driven directly with an explicit pool and audit set, because the
# comment rule is a property of the text walk and nothing about it reaches
# `TrainingSettings.from_yaml`. Eight plants, one per shape -- three red on the
# unfixed migrator, five pinning the drop as narrow, and the last through the
# real `migrate` so the helper is observed on the production path (NN16).
# ---------------------------------------------------------------------------

_DOC = "    # Alpha weights the magnitude spectrum so high-frequency error is\n"
_DOC2 = "    # not swamped by the DC peak.\n"


def _rewritten(mig, text: str):
    """`_rewrite` with a fixed pool/audit set naming one term of each outcome.

    ``alpha`` is removable (pooled and audit-reported), ``beta`` is pooled but
    unreported, and ``diffusion.mse`` is the live :data:`DENY` entry -- reported
    here on purpose, so only the DENY guard is holding it.
    """
    return mig._rewrite(
        text,
        {"alpha", "beta", "mse"},
        {("reconstruction", "alpha"), ("diffusion", "mse")},
    )


def test_a_removed_lambda_takes_the_comment_block_documenting_it(mig):
    """The defect #1948 reports: the key goes, its documentation stays behind."""
    text = (
        "losses:\n"
        "  reconstruction:\n"
        "    lambda_keep: 1.0\n" + _DOC + _DOC2 + "    lambda_alpha: 0.1\n"
        "    lambda_tail: 2.0\n"
    )
    new, removed, _, _ = _rewritten(mig, text)
    assert removed == ["reconstruction.lambda_alpha"]
    assert new == (
        "losses:\n  reconstruction:\n    lambda_keep: 1.0\n    lambda_tail: 2.0\n"
    ), "the whole contiguous block above the removed key goes with it"


def test_a_blank_line_detaches_the_comment_from_the_removed_lambda(mig):
    """A blank line is the separator, so the comment above it is not the key's."""
    text = (
        "losses:\n"
        "  reconstruction:\n"
        "    lambda_keep: 1.0\n" + _DOC + "\n"
        "    lambda_alpha: 0.1\n"
    )
    new, removed, _, _ = _rewritten(mig, text)
    assert removed == ["reconstruction.lambda_alpha"]
    assert new == "losses:\n  reconstruction:\n    lambda_keep: 1.0\n" + _DOC + "\n"


def test_only_the_comment_block_below_the_blank_line_is_dropped(mig):
    """Two blocks split by a blank line: the near one is attached, the far one is not."""
    text = (
        "losses:\n"
        "  reconstruction:\n"
        "    # A standing note about the reconstruction weights.\n"
        "\n" + _DOC + "    lambda_alpha: 0.1\n"
    )
    new, removed, _, _ = _rewritten(mig, text)
    assert removed == ["reconstruction.lambda_alpha"]
    assert new == (
        "losses:\n"
        "  reconstruction:\n"
        "    # A standing note about the reconstruction weights.\n"
        "\n"
    )


def test_an_unreported_lambda_keeps_its_comment(mig):
    """The drop is gated on the removal: a key that survives keeps its prose."""
    text = "losses:\n  reconstruction:\n" + _DOC + "    lambda_beta: 0.1\n"
    new, removed, _, unreported = _rewritten(mig, text)
    assert removed == [] and unreported == ["reconstruction.lambda_beta"]
    assert new == text, "nothing is removed, so nothing is orphaned"


def test_a_denied_lambda_keeps_its_comment(mig):
    """`DENY` runs before the audit, so a denied pair never reaches the drop."""
    text = "losses:\n  diffusion:\n" + _DOC + "    lambda_mse: 0.1\n"
    new, removed, denied, _ = _rewritten(mig, text)
    assert removed == [] and denied == ["diffusion.lambda_mse"]
    assert new == text


def test_a_comment_at_the_blocks_own_level_documents_the_block_and_survives(mig):
    """Dedenting to the category header's column marks a comment as the block's."""
    text = (
        "losses:\n"
        "  reconstruction:\n"
        "    lambda_keep: 1.0\n"
        "  # A note at the block's own level, about the block.\n"
        "    lambda_alpha: 0.1\n"
    )
    new, removed, _, _ = _rewritten(mig, text)
    assert removed == ["reconstruction.lambda_alpha"]
    assert new == (
        "losses:\n"
        "  reconstruction:\n"
        "    lambda_keep: 1.0\n"
        "  # A note at the block's own level, about the block.\n"
    )


def test_a_trailing_comment_on_the_previous_key_survives(mig):
    """A comment sharing a line with a key is not a comment line and stops the walk."""
    text = (
        "losses:\n"
        "  reconstruction:\n"
        "    lambda_keep: 1.0  # keep's own trailing note\n"
        "    lambda_alpha: 0.1\n"
    )
    new, removed, _, _ = _rewritten(mig, text)
    assert removed == ["reconstruction.lambda_alpha"]
    assert new == (
        "losses:\n  reconstruction:\n    lambda_keep: 1.0  # keep's own trailing note\n"
    )


@pytest.fixture
def documented_arm(arm: Path) -> Path:
    """The migrated-arm fixture with a comment block over the re-added key.

    Anchored on the line `arm` itself inserts, so a corpus rewrite that moves the
    arm breaks that fixture's own premise assertion first rather than silently
    re-aiming this plant at a different key.
    """
    text = arm.read_text()
    assert "    lambda_log_spectral: 0.1\n" in text, "premise: the arm fixture re-added the key"
    arm.write_text(
        text.replace(
            "    lambda_log_spectral: 0.1\n",
            "    # Log-spectral weighting: compares magnitude spectra in log space so\n"
            "    # high-frequency error is not swamped by the DC peak.\n"
            "    lambda_log_spectral: 0.1\n",
            1,
        )
    )
    return arm


def test_migrate_drops_the_comment_on_a_real_arm(mig, documented_arm):
    """The production path reaches the drop -- observed, not inferred (NN16)."""
    status, removed = mig.migrate(documented_arm, apply=True)
    assert status == "MIGRATED", status
    assert removed == ["reconstruction.lambda_log_spectral"]
    out = documented_arm.read_text()
    assert "lambda_log_spectral" not in out
    assert "Log-spectral weighting" not in out
    assert "swamped by the DC peak" not in out
