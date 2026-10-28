"""Tests for UlfDpsStrategy (B-2.2 ULF-consistent diffusion posterior sampling)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
    make_alphas_cumprod,
)
from spectramr.infrastructure.training.strategies.ulf_dps_strategy import (
    UlfDpsStrategy,
    compute_ulf_dps_loss,
    ulf_dps_sample,
)
from spectramr.infrastructure.training.strategies.ulf_map_strategy import ulf_degrade
from spectramr.infrastructure.training.strategies.ulf_lowpass_operator import UlfLowpassOperator
from spectramr.models.generators.field_guided_score_unet import FieldGuidedScoreUNet
from tests.utils.config_block_stub import block_stub


def _net() -> FieldGuidedScoreUNet:
    # cond_channels=0: unconditional high-field prior (source enters only via DPS).
    return FieldGuidedScoreUNet(cond_channels=0, width=16, time_dim=16)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),   # ULF source (the measurement)
        "target": torch.rand(2, 1, 16, 16),  # high-field target
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_loss_keys_and_finite() -> None:
    acp = make_alphas_cumprod(50, "cosine")
    out = compute_ulf_dps_loss(_net(), _batch(), alphas_cumprod=acp)
    assert {"loss_total", "loss_eps"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    acp = make_alphas_cumprod(50, "cosine")
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = compute_ulf_dps_loss(m, batch, alphas_cumprod=acp)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_sample_shape_matches_source() -> None:
    acp = make_alphas_cumprod(40, "cosine")
    m = _net().eval()
    y = torch.rand(2, 1, 16, 16)
    out = ulf_dps_sample(
        m, y, torch.tensor([7.0, 1.5]), alphas_cumprod=acp, n_steps=6, likelihood_weight=1.0
    )
    assert out.shape == y.shape


def test_likelihood_guidance_reduces_data_residual() -> None:
    # ANTI-FACADE (#16): the DPS likelihood must be load-bearing — guided sampling
    # (likelihood_weight>0) must end CLOSER to ULF measurement consistency
    # ||A(x_hat) - y_ulf||^2 than the unconstrained prior (weight=0). Train the prior
    # briefly so x0_hat is sensible, then compare with the SAME random sequence.
    torch.manual_seed(0)
    acp = make_alphas_cumprod(40, "cosine")
    m = _net()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        compute_ulf_dps_loss(m, batch, alphas_cumprod=acp)["loss_total"].backward()
        opt.step()
    m.eval()
    y = torch.rand(1, 1, 16, 16)
    b_t = torch.tensor([7.0])
    torch.manual_seed(1)
    guided = ulf_dps_sample(m, y, b_t, alphas_cumprod=acp, n_steps=30, likelihood_weight=5.0)
    torch.manual_seed(1)
    unguided = ulf_dps_sample(m, y, b_t, alphas_cumprod=acp, n_steps=30, likelihood_weight=0.0)
    res_g = float((ulf_degrade(guided, 0.5) - y).pow(2).mean())
    res_u = float((ulf_degrade(unguided, 0.5) - y).pow(2).mean())
    assert res_g < res_u, f"guided residual {res_g} should be < unguided {res_u}"


def test_validation_step_end_to_end_no_batch_context_collision() -> None:
    m = _net().eval()
    strat = object.__new__(UlfDpsStrategy)
    strat.env = types.SimpleNamespace(generator=m)
    strat._dps_timesteps = 30
    strat._dps_schedule = "cosine"
    strat._dps_sampling_steps = 5
    strat._dps_likelihood_weight = 1.0
    strat._ulf_lowpass = UlfLowpassOperator.constant_cutoff(0.5)
    strat._acp_cache = {}
    _cfg = types.SimpleNamespace(
        training=None,
        validation=block_stub("validation", compute_image_metrics=False),
    )
    strat.state = types.SimpleNamespace(config=_cfg)
    strat.config = _cfg
    out = strat.validation_step(
        torch.rand(2, 1, 16, 16),
        torch.rand(2, 1, 16, 16),
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert isinstance(out, dict)  # no TypeError through the full chain


class _RecScore(torch.nn.Module):
    """Records per-call batch size to verify validation chunking is load-bearing."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.seen: list[int] = []

    def forward(self, x, *, timesteps, field_strength, **_):
        self.seen.append(int(x.shape[0]))
        return self.conv(x)


def _val_strat(rec, val_chunk_size):
    strat = object.__new__(UlfDpsStrategy)
    strat.env = types.SimpleNamespace(generator=rec)
    strat._dps_timesteps = 20
    strat._dps_schedule = "cosine"
    strat._dps_sampling_steps = 2
    strat._dps_likelihood_weight = 1.0
    strat._ulf_lowpass = UlfLowpassOperator.constant_cutoff(0.5)
    strat._acp_cache = {}
    strat.config = types.SimpleNamespace(
        validation=block_stub(
            "validation",
            compute_image_metrics=False,
            val_chunk_size=val_chunk_size,
        )
    )
    strat.state = types.SimpleNamespace(config=strat.config)
    return strat


def test_validation_chunking_is_load_bearing() -> None:
    # REGRESSION (review HIGH): val_chunk_size must micro-batch the DPS reverse loop
    # (each step retains an autograd graph -> OOM risk), not be an inert knob. With
    # chunk=1 over batch=3, the score net is only ever called with batch size 1.
    rec = _RecScore()
    strat = _val_strat(rec, val_chunk_size=1)
    out = strat._validation_forward(
        torch.rand(3, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([0.1, 1.5, 7.0]),
    )
    assert out.shape == (3, 1, 16, 16)
    assert set(rec.seen) == {1}  # every forward saw a single-sample micro-batch


def test_validation_no_chunk_uses_full_batch() -> None:
    rec = _RecScore()
    strat = _val_strat(rec, val_chunk_size=0)
    strat._validation_forward(
        torch.rand(3, 1, 16, 16),
        {"use_dc": False},
        field_strength_target=torch.tensor([0.1, 1.5, 7.0]),
    )
    assert set(rec.seen) == {3}


def test_per_sample_normalization_decouples_batch() -> None:
    # Review MEDIUM: the DPS step must use a PER-SAMPLE residual norm. Sample 0's
    # estimate must be invariant to what sample 1 is (no cross-sample coupling).
    torch.manual_seed(0)
    acp = make_alphas_cumprod(30, "cosine")
    m = _net().eval()
    y0 = torch.rand(1, 1, 16, 16)
    bt = torch.tensor([7.0])
    torch.manual_seed(2)
    solo = ulf_dps_sample(m, y0, bt, alphas_cumprod=acp, n_steps=6, likelihood_weight=2.0)
    # Pair y0 with a very different second sample; sample-0 output must be unchanged.
    y_pair = torch.cat([y0, torch.rand(1, 1, 16, 16) * 5.0], dim=0)
    torch.manual_seed(2)
    pair = ulf_dps_sample(
        m, y_pair, torch.tensor([7.0, 1.5]), alphas_cumprod=acp, n_steps=6, likelihood_weight=2.0
    )
    assert torch.allclose(solo[0], pair[0], atol=1e-4)


def test_output_in_data_range() -> None:
    # Review MEDIUM/LOW: the metric-scored output must be clamped to [0,1].
    acp = make_alphas_cumprod(30, "cosine")
    out = ulf_dps_sample(
        _net().eval(), torch.rand(2, 1, 16, 16), torch.tensor([7.0, 1.5]),
        alphas_cumprod=acp, n_steps=6, likelihood_weight=5.0,
    )
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "ulf_dps" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "ulf_dps" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(UlfDpsStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._dps_timesteps = 10
    strat._dps_schedule = "cosine"
    strat._acp_cache = {}

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the score net (train + DPS sampler) ---


def test_contrast_id_threaded_train_and_sampling() -> None:
    # b22 enables use_contrast_conditioning on the widened FieldGuidedScoreUNet. The
    # prior is p(x_HF | field, contrast) — conditioning it on the target contrast is
    # legitimate (which contrast to synthesize), NOT source leakage. The strategy MUST
    # forward contrast_id or the widened net raises at step 0 (the cluster failure).
    seen: dict = {}

    def spy(x, *, timesteps, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 0.1]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    acp = make_alphas_cumprod(20, "cosine")
    compute_ulf_dps_loss(spy, batch, alphas_cumprod=acp)
    assert seen["cid"] is cid  # training eps-matching threads it
    seen.clear()
    ulf_dps_sample(
        spy, torch.rand(2, 1, 16, 16), torch.tensor([7.0, 7.0]),
        alphas_cumprod=acp, n_steps=1, likelihood_weight=0.0, contrast_id=cid,
    )
    assert seen["cid"] is cid  # DPS reverse sampler threads it


def test_validation_step_declares_contrast_id_for_pipeline_seam() -> None:
    import inspect

    params = inspect.signature(UlfDpsStrategy.validation_step).parameters
    assert "contrast_id" in params


_PER_CONTRAST = UlfLowpassOperator(axis_cutoffs={0: (0.15, 0.15), 1: (2.0, 2.0)})


def test_dps_likelihood_uses_the_per_contrast_forward_op() -> None:
    # The DPS residual A(x0)-y runs through the strategy's operator: the same seed and
    # measurement under two contrast ids give two different samples.
    acp = make_alphas_cumprod(30, "cosine")
    m = _net().eval()
    y = torch.rand(1, 1, 16, 16)
    kw = {"alphas_cumprod": acp, "n_steps": 4, "likelihood_weight": 5.0, "forward_op": _PER_CONTRAST}
    torch.manual_seed(0)
    a = ulf_dps_sample(m, y, torch.tensor([3.0]), contrast_id=torch.tensor([0]), **kw)
    torch.manual_seed(0)
    b = ulf_dps_sample(m, y, torch.tensor([3.0]), contrast_id=torch.tensor([1]), **kw)
    assert not torch.allclose(a, b)


def test_setup_raises_without_ulf_lowpass_block() -> None:
    import pytest

    strat = object.__new__(UlfDpsStrategy)
    strat._verify_strategy_config = lambda **_: None
    strat.config = types.SimpleNamespace(
        training=types.SimpleNamespace(ulf_dps=None, ulf_lowpass=None), data=None
    )
    with pytest.raises(ValueError, match=r"training\.ulf_lowpass"):
        strat._setup_strategy_specific_components()
