"""WS1-core-03 (#3): the gated feature-insertion hallucination hook in the
diffusion validation. Disabled by default; runs every interval_validations;
never raises into the validation loop."""

from __future__ import annotations

from types import SimpleNamespace

from tests.utils.config_block_stub import block_stub

import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)


def _stub(*, enabled, interval=1, step=0, predict=None, method="feature_insertion"):
    s = SimpleNamespace()
    s.config = SimpleNamespace(
        validation=block_stub(
            "validation",

            hallucination_test=SimpleNamespace(
                enabled=enabled,
                interval_validations=interval,
                n_features=3,
                method=method,
            )
        )
    )
    s.validation_step_count = step
    s.logging_service = SimpleNamespace(log_warning=lambda *a, **k: None)
    s._generate_validation_prediction = predict
    return s


def _run(stub, target):
    return DiffusionTrainingStrategy._maybe_hallucination_metrics(
        stub, target, None, torch.tensor([1]), None, {}
    )


def test_disabled_returns_empty() -> None:
    assert _run(_stub(enabled=False), torch.zeros(1, 1, 16, 16)) == {}


def test_off_interval_returns_empty() -> None:
    assert _run(_stub(enabled=True, interval=4, step=1), torch.zeros(1, 1, 16, 16)) == {}


def test_enabled_on_interval_returns_index() -> None:
    # identity predict → features preserved → high index
    stub = _stub(enabled=True, interval=1, step=0, predict=lambda p, *a, **k: p)
    out = _run(stub, torch.rand(2, 1, 24, 24))
    assert "val_hallucination_preservation_index" in out
    assert out["val_hallucination_preservation_index"] > 0.9


def test_predict_error_is_swallowed() -> None:
    def boom(*a, **k):
        raise RuntimeError("boom")

    assert _run(_stub(enabled=True, predict=boom), torch.zeros(1, 1, 16, 16)) == {}
