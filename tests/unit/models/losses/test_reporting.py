"""The loss-objective banner renders the DECLARED weight (#1918, #1919).

Two traps make a naive renderer silently wrong, and both are pinned here.

1. ``LossWeightTable.weight(name)`` defaults to ``iteration=0``, so every
   warm-up-gated loss reads back 0.0000. A banner built on it announces
   ``adversarial λ=0.0000`` for an arm that trains it at 0.01 — which reads as
   "disabled" and is the exact failure the empty banner was replaced to fix.
2. An arm with nothing declared must still print a line. Absent is a state to
   report, never one to infer (non-negotiable 18); returning ``[]`` reproduces
   the silence this work removed.

Configs come from the producer (``TrainingSettings.from_yaml``) rather than a
hand-built schema, so a fixture cannot agree with the code by construction.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ARM = (
    _REPO_ROOT
    / "experiments"
    / "inprogress"
    / "kspace_filling"
    / "experiment_11_sense_bridge_critic.yaml"
)


@pytest.fixture(scope="module")
def sense_bridge_table():
    """The real weight table of an arm that declares a warm-up-gated adversarial term."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.models.losses.weights import build_loss_weight_table

    assert _ARM.exists(), f"missing experiment config: {_ARM}"
    cfg = TrainingSettings.from_yaml(str(_ARM))
    table = build_loss_weight_table(cfg.losses)
    assert table.warmup_iterations > 0, (
        "this arm must declare a warm-up window or the gating assertions below are vacuous"
    )
    return table


@pytest.mark.unit
class TestDeclaredWeightNotResolvedWeight:
    def test_a_warmup_gated_loss_shows_its_declared_weight(self, sense_bridge_table):
        """``adversarial`` renders at 0.0100, not the 0.0 the gate returns at step 0."""
        from spectramr.models.losses.reporting import format_loss_objective

        spec = sense_bridge_table["adversarial"]
        assert spec.warmup_gated, "fixture drift: adversarial is no longer warm-up gated"
        assert spec.weight == pytest.approx(0.01)
        # The trap: what a weight()-based renderer would have printed instead.
        assert sense_bridge_table.weight("adversarial", iteration=0) == 0.0

        line = next(
            ln
            for ln in format_loss_objective(sense_bridge_table, prefix="[X]")
            if "adversarial" in ln
        )
        assert "0.0100" in line
        assert "0.0000" not in line

    def test_a_gated_loss_says_so(self, sense_bridge_table):
        """The note is what keeps the declared weight from over-claiming."""
        from spectramr.models.losses.reporting import format_loss_objective

        line = next(
            ln
            for ln in format_loss_objective(sense_bridge_table, prefix="[X]")
            if "adversarial" in ln
        )
        assert "warmup" in line
        assert str(sense_bridge_table.warmup_iterations) in line

    def test_the_source_of_every_weight_is_named(self, sense_bridge_table):
        """A banner without provenance cannot settle 'which surface set this?'."""
        from spectramr.models.losses.reporting import format_loss_objective

        body = [ln for ln in format_loss_objective(sense_bridge_table, prefix="[X]") if "λ=" in ln]
        assert body
        for line in body:
            assert "(losses." in line, f"no config source in: {line}"


@pytest.mark.unit
class TestNothingDeclaredIsStillReported:
    def test_an_empty_table_renders_a_line(self):
        """Non-negotiable 18: report the absence, never render silence."""
        from spectramr.models.losses.reporting import format_loss_objective
        from spectramr.models.losses.weights import build_loss_weight_table

        lines = format_loss_objective(build_loss_weight_table(None), prefix="[X]")
        assert lines, "an arm with no declared losses printed nothing"
        assert len(lines) == 1
        assert "(0)" in lines[0]

    def test_an_all_zero_arm_says_why_it_is_empty(self):
        """'Declared but all zero' and 'declared nothing' are different states."""
        from spectramr.config.schemas.loss import LossConfigSchema
        from spectramr.models.losses.reporting import format_loss_objective
        from spectramr.models.losses.weights import build_loss_weight_table

        empty = format_loss_objective(build_loss_weight_table(None), prefix="[X]")
        schema_only = format_loss_objective(
            build_loss_weight_table(LossConfigSchema()), prefix="[X]"
        )
        assert "no loss weights at all" in empty[0]
        # A bare schema declares nothing either — the defaults are not declarations.
        assert schema_only[0] == empty[0]


@pytest.mark.unit
class TestActivePredicate:
    def test_warmup_gating_does_not_make_a_loss_inactive(self, sense_bridge_table):
        """A gated loss is part of the objective; it just contributes 0 for a while."""
        from spectramr.models.losses.reporting import active_loss_names, is_active

        assert is_active(sense_bridge_table["adversarial"])
        assert "adversarial" in active_loss_names(sense_bridge_table)

    def test_zero_weighted_and_disabled_names_are_excluded(self, sense_bridge_table):
        from spectramr.models.losses.reporting import active_loss_names

        active = active_loss_names(sense_bridge_table)
        for name in active:
            spec = sense_bridge_table[name]
            assert spec.enabled and spec.weight > 0
        inactive = set(sense_bridge_table) - active
        assert inactive, "fixture drift: this arm no longer declares any inactive loss"
        for name in inactive:
            assert not (sense_bridge_table[name].enabled and sense_bridge_table[name].weight > 0)

    def test_strategy_inline_terms_are_part_of_the_objective(self, sense_bridge_table):
        """``pre_dc_kspace`` has no loss module; the banner must still report it.

        It reaches the CSV under a renamed key (``pre_dc_kspace_l1``, owned by
        ``declared_metric_keys``), so the banner is the only place the user sees
        the weight they actually wrote.
        """
        from spectramr.models.losses.reporting import active_loss_names

        assert "pre_dc_kspace" in active_loss_names(sense_bridge_table)
