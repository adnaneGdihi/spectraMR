"""Planted violations for the score-SDE config rules and the audit check that applies them."""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from spectramr.config.settings import TrainingSettings
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.score_sde_config import (
    declared_sde,
    validate_score_sde_config,
)
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

pytestmark = pytest.mark.unit

ARMS = Path("experiments/inprogress/diffusion")


def _config(
    model_type: str = "score_based_diffusion",
    model_kwargs: dict | None = None,
    dc_enabled: bool = True,
    dc_method: str = "pc_sampler_consistency",
    mask_types: tuple = ("random",),
    mask_direction: str | None = None,
) -> types.SimpleNamespace:
    kwargs = (
        {"sde_type": "vesde", "sigma_min": 0.01, "sigma_max": 50.0}
        if model_kwargs is None
        else model_kwargs
    )
    dc = types.SimpleNamespace(
        enabled=dc_enabled, method=dc_method, eval_noise_level=0.0, interval=1
    )
    return types.SimpleNamespace(
        model=types.SimpleNamespace(model_type=model_type, model_kwargs=kwargs),
        physics=types.SimpleNamespace(data_consistency=dc),
        data=types.SimpleNamespace(image_undersampling=False, acquisition=None),
        undersampling=types.SimpleNamespace(
            mask_types=list(mask_types), mask_direction=mask_direction
        ),
    )


def _check(cfg) -> object:
    return ConfigHealthChecker().check_score_sde_declaration(cfg)


def test_a_complete_declaration_passes() -> None:
    cfg = _config()
    assert declared_sde(cfg) == ("vesde", 0.01, 50.0)
    validate_score_sde_config(cfg)
    assert _check(cfg).passed


@pytest.mark.parametrize(
    ("model_kwargs", "match"),
    [
        ({"sde_type": "vpsde", "sigma_min": 0.01, "sigma_max": 50.0}, "Unknown sde_type"),
        ({"sigma_min": 0.01, "sigma_max": 50.0}, "without an sde_type"),
        ({"sde_type": "vesde", "sigma_min": 0.01}, "needs both"),
        ({"sde_type": "vesde", "sigma_min": 50.0, "sigma_max": 0.01}, "sigma_min < sigma_max"),
    ],
)
def test_an_illegal_sde_is_rejected(model_kwargs: dict, match: str) -> None:
    cfg = _config(model_kwargs=model_kwargs)
    with pytest.raises(ConfigurationError, match=match):
        declared_sde(cfg)
    result = _check(cfg)
    assert not result.passed and result.severity == "error"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"dc_enabled": False}, "data_consistency enabled"),
        ({"dc_method": "not_a_sampler"}, "not a registered sampler"),
        ({"mask_types": ()}, "exactly one mask_types"),
        ({"mask_types": ("random", "equispaced")}, "exactly one mask_types"),
        ({"mask_direction": "diagonal"}, "mask_direction"),
    ],
)
def test_a_config_the_engine_cannot_run_is_rejected(overrides: dict, match: str) -> None:
    cfg = _config(**overrides)
    with pytest.raises(ConfigurationError, match=match):
        validate_score_sde_config(cfg)
    assert not _check(cfg).passed


def test_other_model_types_and_discrete_arms_are_skipped() -> None:
    other = _config(model_type="mrf_tangent_score", model_kwargs={"sde_type": "vp"})
    assert declared_sde(other) == (None, None, None)
    assert _check(other).passed
    assert _check(_config(model_kwargs={}, dc_enabled=False)).passed


def test_the_audit_fails_exp96_and_passes_exp72() -> None:
    exp96 = TrainingSettings.from_yaml(ARMS / "experiment_96_sde_diffusion.yaml")
    exp72 = TrainingSettings.from_yaml(ARMS / "experiment_72_score_based_diffusion_sde.yaml")
    assert "needs both sigma_min and sigma_max" in _check(exp96).message
    assert not _check(exp96).passed
    assert _check(exp72).passed


def test_the_check_runs_in_the_full_report() -> None:
    exp96 = TrainingSettings.from_yaml(ARMS / "experiment_96_sde_diffusion.yaml")
    report = ConfigHealthChecker().run_all_checks(exp96)
    assert any(r.check_name == "score_sde_declaration" and not r.passed for r in report.results)
