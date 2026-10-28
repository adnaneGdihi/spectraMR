"""Tests for FieldWienerStrategy (B-2.7)."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.field_wiener_strategy import (
    FieldWienerStrategy,
    compute_field_wiener_loss,
)
from spectramr.models.generators.field_wiener_net import FieldWienerNet


def _net() -> FieldWienerNet:
    return FieldWienerNet(width=24, n_blocks=2, use_field_noise=True)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 32, 32),
        "target": torch.rand(2, 1, 32, 32),
        "field_strength": torch.tensor([0.1, 1.5]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_field_wiener_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim/sobel_edge) fold onto the inline objective
    via the loss-SSOT seam; the inline l1 placeholder is skipped (no double-count)."""
    from spectramr.data.batch_types import BatchAdapter
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))
    strat._fw_lambda_l1 = 1.0

    tb = BatchAdapter.from_dict(_batch())
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_loss_requires_field_strength() -> None:
    batch = {"input": torch.rand(2, 1, 32, 32), "target": torch.rand(2, 1, 32, 32)}
    with pytest.raises(ValueError, match="field_strength"):
        compute_field_wiener_loss(_net(), batch)


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        out = compute_field_wiener_loss(m, batch)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._fw_lambda_l1 = 1.0
    out = strat._compute_losses_impl(input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb)
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._fw_lambda_l1 = 1.0
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16), target_batch=torch.rand(2, 1, 16, 16),
            epoch=0, batch=torch.rand(2, 1, 16, 16),
        )


def test_validation_forward_in_unit_range() -> None:
    m = _net().eval()
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    pred = strat._validation_forward(
        torch.rand(2, 1, 32, 32), {"field_strength": torch.tensor([0.1, 1.5])}
    ).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_score_wiener_emits_band_and_field_sensitivity() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    out = strat._score_wiener(
        torch.rand(2, 1, 32, 32), {"input": torch.rand(2, 1, 32, 32), "field_strength": torch.tensor([0.1, 1.5])}
    )
    assert "val_recoverable_band" in out
    assert out["val_band_field_sensitivity"] >= 0.0  # field-dependent arm: band(7T) >= band(0.1T)


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_wiener" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "field_wiener" in TrainingStrategyConfigSchema.model_fields


def test_loss_uses_declared_source_field_on_paired_route() -> None:
    # The paired ULF route stamps no per-sample field_strength; the declared 64 mT source
    # field supplies it, so the restorer runs at the stated field rather than raising.
    batch = {"input": torch.rand(2, 1, 32, 32), "target": torch.rand(2, 1, 32, 32)}
    out = compute_field_wiener_loss(_net(), batch, source_field=0.064)
    assert torch.isfinite(out["loss_total"])


def test_validation_forward_uses_declared_source_field() -> None:
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=_net().eval())
    strat._fw_source_field = 0.064
    pred = strat._validation_forward(torch.rand(2, 1, 32, 32), {}).detach()
    assert float(pred.min()) >= 0.0 and float(pred.max()) <= 1.0


def test_validation_forward_raises_without_any_field() -> None:
    strat = object.__new__(FieldWienerStrategy)
    strat.env = types.SimpleNamespace(generator=_net().eval())
    with pytest.raises(ValueError, match="source field"):
        strat._validation_forward(torch.rand(2, 1, 32, 32), {})


def test_source_field_knob_validated() -> None:
    from pydantic import ValidationError

    from spectramr.config.schemas.training.field_wiener import FieldWienerConfig

    assert FieldWienerConfig().source_field is None
    with pytest.raises(ValidationError):
        FieldWienerConfig(source_field=-0.1)
