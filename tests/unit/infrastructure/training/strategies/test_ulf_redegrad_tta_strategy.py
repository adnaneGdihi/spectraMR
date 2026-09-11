"""Tests for UlfReDegradationTTAStrategy (B-2.10)."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from spectramr.infrastructure.training.strategies.ulf_map_strategy import ulf_degrade
from spectramr.infrastructure.training.strategies.ulf_redegrad_tta_strategy import (
    UlfReDegradationTTAStrategy,
    compute_redegrad_tta_train_loss,
    redegrad_tta_render,
)
from spectramr.models.generators.field_conditioned_inr import FieldConditionedINR
from tests.utils.config_block_stub import block_stub


def _net() -> FieldConditionedINR:
    return FieldConditionedINR(feat_dim=8, hidden_features=64, hidden_layers=3, style_dim=32)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_train_loss_keys_and_finite() -> None:
    out = compute_redegrad_tta_train_loss(_net(), _batch())
    assert {"loss_total", "loss_l1"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_builder_image_losses_folded_via_seam() -> None:
    """Declarative image losses (hfen/ms_ssim) fold onto the base-model L1 via the loss-SSOT
    seam; the inline l1 placeholder is skipped (no double-count)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = object.__new__(UlfReDegradationTTAStrategy)
    strat.env = types.SimpleNamespace(
        generator=_net(), losses={"l1": CharbonnierLoss(), "hfen": HFENLoss()})
    strat.config = types.SimpleNamespace(losses=types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[], complex_losses=[]))
    strat._tta_lambda_l1 = 1.0

    out = strat._compute_losses_impl(input_batch=_batch(), target_batch=None, epoch=0)
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["loss_total"])
    assert "prediction" not in out and "target_image" not in out


def test_tta_reduces_redegradation_residual() -> None:
    # HEADLINE (anti-facade): test-time adaptation must lower the self-supervised
    # re-degradation residual ||A(G(src)) - src|| vs the un-adapted model. Proves the
    # inner adaptation loop is genuinely load-bearing.
    torch.manual_seed(0)
    m = _net().eval()
    src = torch.rand(1, 1, 16, 16)
    b_t = torch.tensor([7.0])
    base = redegrad_tta_render(
        m, src, b_t, adaptation_steps=0, adaptation_lr=1e-3,
        consistency_weight=1.0, blur_cutoff=0.5,
    )
    adapted = redegrad_tta_render(
        m, src, b_t, adaptation_steps=25, adaptation_lr=1e-3,
        consistency_weight=1.0, blur_cutoff=0.5,
    )
    res_base = float(F.mse_loss(ulf_degrade(base, 0.5), src))
    res_adapt = float(F.mse_loss(ulf_degrade(adapted, 0.5), src))
    assert res_adapt < res_base, f"adapt {res_adapt} should be < base {res_base}"


def test_rollback_never_increases_residual_at_production_point() -> None:
    # Review #1/#3: rollback-to-best guarantees the re-degradation residual NEVER
    # regresses from the base, even at the production operating point (small lr, where
    # Adam from a near-converged init can otherwise overshoot). Pre-fit the model, then
    # assert adapted residual <= base residual.
    torch.manual_seed(0)
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    for _ in range(40):  # near-converge the supervised prior
        opt.zero_grad(set_to_none=True)
        compute_redegrad_tta_train_loss(m, batch)["loss_total"].backward()
        opt.step()
    m.eval()
    src, b_t = batch["input"][:1], batch["field_strength_target"][:1]
    base = redegrad_tta_render(m, src, b_t, adaptation_steps=0, adaptation_lr=1e-4,
                               consistency_weight=1.0, blur_cutoff=0.5)
    adapted = redegrad_tta_render(m, src, b_t, adaptation_steps=8, adaptation_lr=1e-4,
                                  consistency_weight=1.0, blur_cutoff=0.5)
    res_base = float(F.mse_loss(ulf_degrade(base, 0.5), src))
    res_adapt = float(F.mse_loss(ulf_degrade(adapted, 0.5), src))
    assert res_adapt <= res_base + 1e-8, f"rollback failed: {res_adapt} > {res_base}"


def test_high_lr_does_not_diverge() -> None:
    # Review #4: a huge inner lr must not produce NaN/divergence in the returned image
    # (the NaN-guard + rollback keep the best-residual iterate, falling back to base).
    torch.manual_seed(0)
    m = _net().eval()
    src = torch.rand(1, 1, 16, 16)
    out = redegrad_tta_render(m, src, torch.tensor([7.0]), adaptation_steps=10,
                              adaptation_lr=10.0, consistency_weight=1.0, blur_cutoff=0.5)
    assert torch.isfinite(out).all()
    res = float(F.mse_loss(ulf_degrade(out, 0.5), src))
    base = float(F.mse_loss(ulf_degrade(
        redegrad_tta_render(m, src, torch.tensor([7.0]), adaptation_steps=0,
                            adaptation_lr=1e-4, consistency_weight=1.0, blur_cutoff=0.5), 0.5), src))
    assert res <= base + 1e-8  # divergence rolled back to base


def test_params_restored_after_adaptation() -> None:
    # Per-sample TTA: the base model must be RESTORED after rendering (no cross-sample
    # weight leakage).
    torch.manual_seed(0)
    m = _net().eval()
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    redegrad_tta_render(
        m, torch.rand(1, 1, 16, 16), torch.tensor([7.0]), adaptation_steps=10,
        adaptation_lr=1e-3, consistency_weight=1.0, blur_cutoff=0.5,
    )
    for k, v in m.state_dict().items():
        assert torch.allclose(v, snap[k], atol=1e-7), f"param {k} not restored"


def test_ablation_zero_steps_is_plain_forward() -> None:
    torch.manual_seed(0)
    m = _net().eval()
    src = torch.rand(2, 1, 16, 16)
    b_t = torch.tensor([7.0, 1.5])
    out = redegrad_tta_render(
        m, src, b_t, adaptation_steps=0, adaptation_lr=1e-3,
        consistency_weight=1.0, blur_cutoff=0.5,
    )
    with torch.no_grad():
        plain = m(src, field_strength=b_t).clamp(0.0, 1.0)
    assert torch.allclose(out, plain, atol=1e-6)


def test_output_in_data_range() -> None:
    out = redegrad_tta_render(
        _net().eval(), torch.rand(2, 1, 16, 16), torch.tensor([7.0, 1.5]),
        adaptation_steps=5, adaptation_lr=1e-3, consistency_weight=1.0, blur_cutoff=0.5,
    )
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_validation_forward_raises_without_field() -> None:
    import pytest

    strat = object.__new__(UlfReDegradationTTAStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat.config = types.SimpleNamespace(
        validation=block_stub("validation", val_chunk_size=0)
    )
    strat._tta_steps = 0
    for k, v in (("_tta_lr", 1e-4), ("_tta_weight", 1.0), ("_tta_cutoff", 0.5)):
        setattr(strat, k, v)
    with pytest.raises(ValueError, match="field_strength_target"):
        strat._validation_forward(torch.rand(1, 1, 16, 16), {"use_dc": False})


class _RecINR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.seen: list[int] = []

    def forward(self, x, *, field_strength, **_):
        self.seen.append(int(x.shape[0]))
        return self.c(x)


def test_validation_defaults_to_per_sample_adaptation() -> None:
    # Review #5: val_chunk_size=0 must still adapt PER SAMPLE (chunk defaults to 1), not
    # adapt the whole batch jointly. With batch=2 the model is only ever called at
    # batch size 1.
    rec = _RecINR()
    strat = object.__new__(UlfReDegradationTTAStrategy)
    strat.env = types.SimpleNamespace(generator=rec)
    strat._tta_steps, strat._tta_lr = 2, 1e-3
    strat._tta_weight, strat._tta_cutoff = 1.0, 0.5
    strat.config = types.SimpleNamespace(
        validation=block_stub("validation", val_chunk_size=0)  # unset -> defaults to 1
    )
    strat._validation_forward(
        torch.rand(2, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert set(rec.seen) == {1}  # every forward saw a single-sample micro-batch


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "ulf_redegrad_tta" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "ulf_redegrad_tta" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(UlfReDegradationTTAStrategy)
    strat.env = types.SimpleNamespace(generator=_net())

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the ULF->HF translator (train + render) ---


def test_contrast_id_threaded_train_and_render() -> None:
    from spectramr.infrastructure.training.strategies.ulf_redegrad_tta_strategy import (
        compute_redegrad_tta_train_loss,
        redegrad_tta_render,
    )

    seen: dict = {}

    def spy(x, *, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "input": torch.rand(2, 1, 8, 8),
        "target": torch.rand(2, 1, 8, 8),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_redegrad_tta_train_loss(spy, batch)
    assert seen["cid"] is cid
    seen.clear()
    # adaptation_steps=0 -> plain feed-forward render (still threads contrast)
    redegrad_tta_render(
        spy, torch.rand(2, 1, 8, 8), torch.tensor([7.0, 7.0]),
        adaptation_steps=0, adaptation_lr=1e-4, consistency_weight=1.0,
        blur_cutoff=0.5, contrast_id=cid,
    )
    assert seen["cid"] is cid
