"""The audit refuses a parallel layout the strategy or model cannot run."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.witness.checks.parallel_compat_checks import (
    parallel_layout_defect,
)

pytestmark = [pytest.mark.unit]


def _settings(strategy="ddp", pipeline=1, model_type="nc_pdnet"):
    return SimpleNamespace(
        parallel=SimpleNamespace(
            strategy=strategy, deepspeed=None, mesh=SimpleNamespace(pipeline=pipeline)
        ),
        model=SimpleNamespace(model_type=model_type),
    )


def _strategies():
    from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    return ReconstructionTrainingStrategy, GANTrainingStrategy


def test_a_pipelined_reconstruction_with_a_hooked_model_passes() -> None:
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    recon, _ = _strategies()
    assert parallel_layout_defect(_settings(pipeline=2), recon) is None


def test_planted_an_undeclared_strategy_under_a_pipeline_is_an_error() -> None:
    _, gan = _strategies()
    assert "pipelined" in parallel_layout_defect(_settings(pipeline=2), gan)


def test_planted_a_model_without_the_hook_is_an_error() -> None:
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    recon, _ = _strategies()
    message = parallel_layout_defect(_settings(pipeline=2, model_type="unet"), recon)
    assert message is not None and "pipeline_stages" in message


def test_single_process_is_not_judged() -> None:
    _, gan = _strategies()
    assert parallel_layout_defect(_settings(strategy="none", pipeline=1), gan) is None
