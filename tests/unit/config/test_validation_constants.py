"""Regression: previously-unregistered training modes are now in the allow-list.

The `privileged_learning`/`privileged` and `multi` strategies were dispatchable
via an explicit ``strategy_class`` but absent from ``VALID_TRAINING_MODES``, so an experiment selecting them by
``training_mode`` would fail the audit. 2026-05-31: registered them so their
experiments can be audit-clean.
"""

from __future__ import annotations

from spectramr.config.validation_constants import (
    CONTRAST_THREADED_STRATEGIES,
    VALID_TRAINING_MODES,
)


def test_contrast_threaded_strategies_is_the_ssot_set() -> None:
    # Single source of truth for "this strategy forwards batch['contrast_id'] to
    # the model" — consumed by both the audit guard
    # (ConfigHealthChecker.check_contrast_conditioning_strategy_threaded) and the
    # per-arm test guard. A widened (use_contrast_conditioning) model on a
    # strategy NOT in this set raises "batch carries no contrast_id" at step 0.
    assert isinstance(CONTRAST_THREADED_STRATEGIES, frozenset)
    # The 12 strategies verified (grep + mechanism test) to thread contrast_id
    # as of the 2026-07-07 broad multicontrast rollout.
    expected = {
        "field_flow",
        "field_bridge",
        "field_guided_diffusion",
        "fisher_rao_geodesic",
        "scattering_besov",
        "steerable_synthesis",
        "brenier_synthesis",
        "field_conditioned_inr",
        "ulf_redegrad_tta",
        "ulf_dps",
        "generative_refiner",
        "field_cold_diffusion",
    }
    assert expected <= CONTRAST_THREADED_STRATEGIES


def test_privileged_and_multi_modes_registered():
    for mode in ("privileged_learning", "privileged", "multi"):
        assert mode in VALID_TRAINING_MODES, f"{mode} missing from VALID_TRAINING_MODES"


def test_generative_mode_is_valid():
    """Regression WS1-core-01: the normalizing-flow arms (glow, equivariant_flow)
    use ``training_mode: generative``."""
    assert "generative" in VALID_TRAINING_MODES


def test_multi_acquisition_mode_registered() -> None:
    assert "multi_acquisition" in VALID_TRAINING_MODES


def test_orphan_training_modes_are_gone() -> None:
    """The three A3 orphans: advertised in VALID_TRAINING_MODES, dispatchable never.

    ``kan`` / ``pretraining`` / ``sensitivity_estimation`` had no
    ``STRATEGY_CLASS_PATHS`` entry -- git history shows none was ever written.
    A YAML naming one passed schema validation and then died at strategy
    resolution (pitfall #9). Removed 2026-07-19; see docs/training_mode_ssot.rst
    for the replacements (``mae``/``ssl``, an architecture axis, and ``pinn``).
    """
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    for mode in ("kan", "pretraining", "sensitivity_estimation"):
        assert mode not in VALID_TRAINING_MODES
        assert mode not in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_advertised_modes_are_all_dispatchable() -> None:
    """VALID_TRAINING_MODES must not re-grow an entry that names no strategy.

    This is the anti-regrowth guard for the orphan class. It is one-directional:
    the table is still missing 121 dispatchable modes, which the follow-up
    reconciliation addresses by deleting the table outright.
    """
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    orphans = set(VALID_TRAINING_MODES) - set(
        TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    )
    assert orphans == set(), (
        f"advertised but undispatchable: {sorted(orphans)} -- either register a "
        "strategy or remove the name"
    )
