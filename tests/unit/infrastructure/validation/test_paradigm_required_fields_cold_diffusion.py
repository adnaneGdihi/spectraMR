"""Regression: the bare 'diffusion' paradigm rule must not false-positive on
'field_cold_diffusion' (B-2.4 review CRITICAL — audit exit 2)."""

from __future__ import annotations

import types

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _cfg(strategy_class: str, has_diffusion_block: bool):
    training = types.SimpleNamespace(
        strategy_class=strategy_class,
        training_mode=strategy_class,
        diffusion=(object() if has_diffusion_block else None),
        field_cold_diffusion=object(),
    )
    return types.SimpleNamespace(training=training)


def _errors(results):
    return [r for r in results if not r.passed and r.severity == "error"]


def test_field_cold_diffusion_not_required_to_set_training_diffusion() -> None:
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("field_cold_diffusion", has_diffusion_block=False)
    )
    # No paradigm_required_fields error: the strategy owns training.field_cold_diffusion
    # and never reads training.diffusion.
    assert _errors(results) == []


def test_field_guided_diffusion_not_required_to_set_training_diffusion() -> None:
    # B-3.5: the self-contained field-CFG strategy owns training.field_guided_diffusion
    # and never reads training.diffusion — the exact-name guard must exempt it too.
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("field_guided_diffusion", has_diffusion_block=False)
    )
    assert _errors(results) == []


def test_genuine_diffusion_still_requires_training_diffusion() -> None:
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("diffusion", has_diffusion_block=False)
    )
    # The real DiffusionTrainingStrategy must still be flagged when the block is absent.
    assert any(r.category == "paradigm_required_fields" for r in _errors(results))


def test_levy_denoising_not_required_to_set_training_diffusion() -> None:
    # LevyDenoisingStrategy owns training.levy_denoising and refuses training.diffusion.
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("levy_diffusion", has_diffusion_block=False)
    )
    assert _errors(results) == []


def test_other_diffusion_named_mode_still_requires_training_diffusion() -> None:
    # The exemption is an exact mode name, not the "diffusion" substring.
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("latent_diffusion", has_diffusion_block=False)
    )
    assert any(r.category == "paradigm_required_fields" for r in _errors(results))


def test_resetting_denoiser_not_required_to_set_training_diffusion() -> None:
    # resetting_diffusion routes to LevyDenoisingStrategy, which refuses training.diffusion.
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("resetting_diffusion", has_diffusion_block=False)
    )
    assert _errors(results) == []


def test_qspace_diffusion_not_required_to_set_training_diffusion() -> None:
    # QSpaceDiffusionStrategy reads training.qspace_diffusion; its reconstruction base
    # refuses training.diffusion, so requiring it here made the arm unlaunchable.
    checker = ConfigHealthChecker()
    results = checker.check_paradigm_required_fields(
        _cfg("qspace_diffusion", has_diffusion_block=False)
    )
    assert _errors(results) == []
