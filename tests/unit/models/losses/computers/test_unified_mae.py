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


def test_the_default_objective_is_mse_plus_a_tenth_of_l1() -> None:
    pred, target = _pair()
    out = UnifiedMAELossComputer(config=None).compute(pred, target)

    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    assert float(out.total) == pytest.approx(float(mse + 0.1 * l1))
    assert out.total.grad_fn is not None


def test_metrics_are_detached_tensors_with_the_unweighted_values(monkeypatch) -> None:
    pred, target = _pair()

    def no_float(self: torch.Tensor) -> float:
        raise AssertionError("float(tensor) on the training path")

    monkeypatch.setattr(torch.Tensor, "__float__", no_float)
    out = UnifiedMAELossComputer(config=None).compute(pred, target)
    monkeypatch.undo()

    assert set(out.metrics) == {"reconstruction", "l1"}
    for value in out.metrics.values():
        assert isinstance(value, torch.Tensor) and not value.requires_grad
    assert float(out.metrics["reconstruction"]) == pytest.approx(float(F.mse_loss(pred, target)))
    assert float(out.metrics["l1"]) == pytest.approx(float(F.l1_loss(pred, target)))
