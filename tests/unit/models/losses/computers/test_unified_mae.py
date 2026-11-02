"""``UnifiedMAELossComputer``: construction and the per-step output contract.

``compute`` called ``float()`` twice per step to fill ``metrics``, a dict no
training-path caller reads; the values are now detached tensors, equal to what
the floats were. The objective itself is unchanged here: which weights it reads is
reported separately (it bypasses the loss-weight SSOT).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from spectramr.models.losses.computers import UnifiedMAELossComputer


def _pair() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    return torch.rand(2, 1, 8, 8, requires_grad=True), torch.rand(2, 1, 8, 8)


@pytest.mark.parametrize("device", ["cpu", torch.device("cpu")])
def test_the_device_is_normalised_to_a_torch_device(device: str | torch.device) -> None:
    computer = UnifiedMAELossComputer(config=None, device=device)
    assert computer.device == torch.device("cpu")


def _settings(image_losses: list[dict]):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "losses": {"output_domain": "image", "image_losses": image_losses},
        }
    )


def test_only_the_declared_mse_is_trained() -> None:
    """An undeclared L1 is not trained (#2280): the objective is the declared MSE alone."""
    pred, target = _pair()
    computer = UnifiedMAELossComputer(config=_settings([{"name": "mse", "weight": 1.0}]))
    out = computer.compute(pred, target)

    assert float(out.total) == pytest.approx(float(F.mse_loss(pred, target)))
    assert set(out.components) == {"reconstruction_loss"}
    assert out.total.grad_fn is not None


def test_a_declared_l1_is_weighted_by_the_ssot() -> None:
    pred, target = _pair()
    computer = UnifiedMAELossComputer(
        config=_settings(
            [{"name": "mse", "weight": 1.0}, {"name": "l1", "weight": 0.1}],
        )
    )
    # l1 is warm-up gated (LEGACY_WARMUP_LOSSES), so step past the gate.
    out = computer.compute(pred, target, iteration=2000)

    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    assert float(out.total) == pytest.approx(float(mse + 0.1 * l1))


def test_metrics_are_detached_tensors_with_the_unweighted_values(monkeypatch) -> None:
    pred, target = _pair()

    def no_float(self: torch.Tensor) -> float:
        raise AssertionError("float(tensor) on the training path")

    monkeypatch.setattr(torch.Tensor, "__float__", no_float)
    out = UnifiedMAELossComputer(config=_settings([{"name": "mse", "weight": 1.0}])).compute(
        pred, target
    )
    monkeypatch.undo()

    assert set(out.metrics) == {"reconstruction", "l1"}
    for value in out.metrics.values():
        assert isinstance(value, torch.Tensor) and not value.requires_grad
    assert float(out.metrics["reconstruction"]) == pytest.approx(float(F.mse_loss(pred, target)))
    assert float(out.metrics["l1"]) == pytest.approx(float(F.l1_loss(pred, target)))


def test_an_arm_with_no_reconstruction_term_raises_instead_of_a_zero_objective() -> None:
    """#177: neither mse nor l1 declared is an empty objective, not a zero loss."""
    from spectramr.domain.exceptions import ConfigurationError

    pred, target = _pair()
    with pytest.raises(ConfigurationError, match="neither an mse nor an l1"):
        UnifiedMAELossComputer(config=None).compute(pred, target)
