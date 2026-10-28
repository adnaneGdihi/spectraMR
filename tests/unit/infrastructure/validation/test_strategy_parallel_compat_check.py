"""``check_strategy_parallel_compat``: the audit-time view of the director's gate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.config.schemas.base import ParallelismConfigSchema
from spectramr.infrastructure.validation.config_health_checker import (
    FATAL_HEALTH_CHECKS,
    ConfigHealthChecker,
)

pytestmark = pytest.mark.unit

# MRI-SLAM grows per-subject parameters mid-run, so it refuses every multi-process backend.
_MRI_SLAM = "spectramr.infrastructure.training.strategies.mri_slam_strategy.MRISLAMStrategy"


def _checker() -> ConfigHealthChecker:
    return ConfigHealthChecker.__new__(ConfigHealthChecker)


def _config(mode: str, *, strategy: str = "ddp", zero_stage: int = 2):
    kwargs = {"strategy": strategy}
    if strategy == "deepspeed":
        kwargs["deepspeed"] = {"enabled": True, "zero_stage": zero_stage}
    if strategy == "fsdp":
        kwargs["fsdp"] = {"enabled": True}
    return SimpleNamespace(
        parallel=ParallelismConfigSchema(**kwargs),
        training=SimpleNamespace(strategy_class=None, training_mode=mode),
    )


def test_single_process_is_not_applicable() -> None:
    result = _checker().check_strategy_parallel_compat(_config("mri_slam", strategy="none"))
    assert result.passed and result.severity == "info"


def test_planted_owned_module_strategy_under_ddp_is_an_error() -> None:
    result = _checker().check_strategy_parallel_compat(_config("mri_slam"))
    assert not result.passed and result.severity == "error"
    assert "MRISLAMStrategy" in result.message


def test_explicit_dotted_class_is_judged_too() -> None:
    config = _config("mri_slam")
    config.training = SimpleNamespace(strategy_class=_MRI_SLAM, training_mode=None)
    assert not _checker().check_strategy_parallel_compat(config).passed


def test_planted_undeclared_strategy_under_fsdp_is_an_error() -> None:
    result = _checker().check_strategy_parallel_compat(_config("vae", strategy="fsdp"))
    assert not result.passed


def test_planted_diffusion_under_zero3_is_an_error() -> None:
    config = _config("diffusion", strategy="deepspeed", zero_stage=3)
    assert not _checker().check_strategy_parallel_compat(config).passed


def test_diffusion_under_zero2_passes() -> None:
    config = _config("diffusion", strategy="deepspeed", zero_stage=2)
    assert _checker().check_strategy_parallel_compat(config).passed


def test_undeclared_on_a_replicated_backend_is_info_not_a_warning() -> None:
    """A warning would fail ``--strict`` audits of ~300 arms that run today."""
    result = _checker().check_strategy_parallel_compat(_config("vae", strategy="ddp"))
    assert result.passed and result.severity == "info"


def test_reconstruction_keeps_its_fsdp_arm_green() -> None:
    assert (
        _checker().check_strategy_parallel_compat(_config("reconstruction", strategy="fsdp")).passed
    )


def test_unresolvable_strategy_is_left_to_the_other_check() -> None:
    result = _checker().check_strategy_parallel_compat(_config("no_such_strategy"))
    assert result.passed and result.severity == "info"


def test_the_check_is_fatal_because_the_director_raises_on_it_anyway() -> None:
    assert "strategy_parallel_compat" in FATAL_HEALTH_CHECKS
