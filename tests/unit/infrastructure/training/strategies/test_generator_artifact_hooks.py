"""The reconstruction strategy folds the detector loss and scores corrupted validation passes."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies import reconstruction
from spectramr.infrastructure.training.strategies.generator_artifact_hooks import (
    auxiliary_loss_keys,
    corrupted_input_metrics,
    fold_auxiliary_losses,
)
from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)
from spectramr.models.factories.model_factory import ModelFactory

BLOCK = {
    "artifacts": {
        "complex_gaussian": {"probability": 1.0, "severity": [0.5, 0.5]},
        "gibbs": {"probability": 1.0, "severity": [0.5, 0.5]},
    },
    "detector_loss_weight": 0.5,
    "eval_severity": 0.5,
}


def _model():
    return ModelFactory().create_generator(
        "artifact_robust_generator",
        in_channels=4,
        out_channels=4,
        features=[4, 8],
        artifact_augmentation=BLOCK,
    )


def _input() -> torch.Tensor:
    torch.manual_seed(0)
    x = torch.randn(2, 4, 16, 16)
    x[..., ::2] = 0.0
    return x


def test_the_detector_term_is_folded_into_the_total_and_recorded() -> None:
    model = _model().train()
    model(_input())
    losses: dict[str, torch.Tensor] = {}
    total = fold_auxiliary_losses(model, torch.tensor(1.0), losses)
    assert set(losses) == {"artifact_detector_bce"}
    assert torch.isclose(total, 1.0 + losses["artifact_detector_bce"])
    assert auxiliary_loss_keys(model) == {"artifact_detector_bce"}


def test_a_plain_generator_contributes_nothing() -> None:
    plain = torch.nn.Identity()
    assert fold_auxiliary_losses(plain, torch.tensor(2.0), {}) == 2.0
    assert auxiliary_loss_keys(plain) == frozenset()
    assert corrupted_input_metrics(plain, {}, lambda: {"val_psnr": 1.0}) == {}


def test_reconstruction_validation_step_emits_per_artifact_scores(monkeypatch) -> None:
    """The production ``validation_step`` runs the clean and the forced passes."""
    model = _model().eval()
    x = _input()
    target = torch.zeros_like(x)

    def fake_pass(self, input_batch, target_batch, *, batch_data=None, **kwargs):
        with torch.no_grad():
            pred = model(input_batch)
        mse = torch.mean((pred - target_batch) ** 2)
        return {"val_psnr": float(-10 * torch.log10(mse)), "val_zf_psnr": 0.0}

    monkeypatch.setattr(ModelValidationMixin, "validation_step", fake_pass)
    strategy = reconstruction.ReconstructionTrainingStrategy.__new__(
        reconstruction.ReconstructionTrainingStrategy
    )
    strategy.env = SimpleNamespace(generator=model)
    metrics = strategy.validation_step(x, target)
    for name in ("complex_gaussian", "gibbs"):
        assert f"val_{name}_psnr" in metrics
        assert f"val_{name}_zf_psnr" not in metrics
        assert metrics[f"val_{name}_psnr_drop"] == metrics["val_psnr"] - metrics[f"val_{name}_psnr"]
    assert metrics["val_gibbs_psnr"] != metrics["val_psnr"]


def test_the_strategy_declares_the_detector_column_for_the_csv_header() -> None:
    strategy = reconstruction.ReconstructionTrainingStrategy.__new__(
        reconstruction.ReconstructionTrainingStrategy
    )
    strategy.config = SimpleNamespace(model=SimpleNamespace(model_type="artifact_robust_generator"))
    strategy.env = SimpleNamespace(generator=_model())
    assert "artifact_detector_bce" in strategy.declared_metric_keys()
