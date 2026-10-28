"""Tests for DoobBridgeStrategy (B-1.10 Doob h-transform 7T diffusion bridge)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.doob_bridge_strategy import (
    DoobBridgeStrategy,
    compute_doob_loss,
    compute_doob_residual_loss,
    doob_h_transform_sample,
    doob_residual_sample,
)
from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
    make_alphas_cumprod,
)
from spectramr.models.generators.doob_marginal_score_unet import DoobMarginalScoreUNet
from spectramr.models.generators.doob_residual_score_unet import DoobResidualScoreUNet
from tests.utils.config_block_stub import block_stub


def _net() -> DoobMarginalScoreUNet:
    return DoobMarginalScoreUNet(width=16, n_blocks=2)


def _rnet() -> DoobResidualScoreUNet:
    return DoobResidualScoreUNet(width=16, n_blocks=2)


def _acp() -> torch.Tensor:
    return make_alphas_cumprod(40, "cosine", device="cpu", dtype=torch.float32)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 7.0]),
    }


def test_loss_keys_and_finite() -> None:
    out = compute_doob_loss(_net(), _batch(), alphas_cumprod=_acp())
    assert {"loss_total", "loss_eps"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_dsm_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _net()
    acp = _acp()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_doob_loss(m, batch, alphas_cumprod=acp)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_loss_uses_target_marginal_not_source() -> None:
    # The DSM loss must depend ONLY on the 7T target (the marginal whose score we learn),
    # not on the source — anti-facade for "unconditional h". Changing the source must not
    # change the loss (with the same RNG seed and timestep draw).
    m = _net()
    acp = _acp()
    b1 = _batch()
    b2 = dict(b1)
    b2["input"] = torch.rand(2, 1, 16, 16)  # different source, same target
    torch.manual_seed(123)
    l1 = float(compute_doob_loss(m, b1, alphas_cumprod=acp)["loss_total"].detach())
    torch.manual_seed(123)
    l2 = float(compute_doob_loss(m, b2, alphas_cumprod=acp)["loss_total"].detach())
    assert l1 == l2


def test_score_translates_toward_target_and_nopin_is_noise_floor() -> None:
    # ANTI-FACADE (#16/#17/#20), strengthened per review: train the score on a target marginal
    # DISTINCT from the source, then (a) h_scale=1 must TRANSLATE TOWARD the target (closer to
    # the target mean than the noise floor), and (b) h_scale=0 must be a degenerate noise floor
    # (higher variance than the coherent pinned sample), NOT an un-translated source. The weak
    # "pin != source" check it replaces passed even for an UNTRAINED net (review finding).
    torch.manual_seed(0)
    m = DoobMarginalScoreUNet(width=16, n_blocks=2)
    acp = _acp()
    src = torch.rand(4, 1, 16, 16)
    tgt = (0.82 + 0.04 * torch.randn(4, 1, 16, 16)).clamp(
        0.0, 1.0
    )  # bright, low-variance
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        compute_doob_loss(m, {"target": tgt}, alphas_cumprod=acp)[
            "loss_total"
        ].backward()
        opt.step()
    m.eval()
    with torch.no_grad():
        pin = doob_h_transform_sample(
            m, src, alphas_cumprod=acp, n_steps=30, strength=0.6, h_scale=1.0, seed=0
        )
        nopin = doob_h_transform_sample(
            m, src, alphas_cumprod=acp, n_steps=30, strength=0.6, h_scale=0.0, seed=0
        )
    tgt_mean = float(tgt.mean())
    # (a) directional fidelity: the pinned sample's mean is closer to the target than the floor
    assert abs(float(pin.mean()) - tgt_mean) < abs(float(nopin.mean()) - tgt_mean)
    # (b) the h_scale=0 floor is noisier (degenerate) than the coherent pinned sample
    assert float(nopin.std()) > float(pin.std())
    assert not torch.allclose(pin, nopin, atol=1e-3)
    assert float(pin.min()) >= 0.0 and float(pin.max()) <= 1.0  # clamped (#20)


def test_validation_sampling_is_reproducible_with_seed() -> None:
    # REGRESSION (#18): validation must be reproducible — the SDEdit init is the only
    # stochasticity at eta=0, so an unseeded draw made the best-checkpoint monitor noisy.
    # _validation_forward passes a FIXED seed; same seed -> identical, different seed -> differs.
    torch.manual_seed(0)
    m = _net().eval()
    acp = _acp()
    src = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        a = doob_h_transform_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=0)
        b = doob_h_transform_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=0)
        c = doob_h_transform_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=1)
    assert torch.allclose(a, b)  # same seed -> bit-identical (reproducible validation)
    assert not torch.allclose(a, c, atol=1e-4)  # different seed -> different init


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard): the canonical pipeline forwards a TrainingBatch, not a dict.
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(DoobBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._db_timesteps = 40
    strat._db_schedule = "cosine"
    strat._db_residual = False
    strat._acp_cache = {}
    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_compute_losses_rejects_tensor_batch() -> None:
    import pytest

    strat = object.__new__(DoobBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._db_timesteps = 40
    strat._db_schedule = "cosine"
    strat._acp_cache = {}
    with pytest.raises(ValueError, match="mapping batch"):
        strat._compute_losses_impl(
            input_batch=torch.rand(2, 1, 16, 16),
            target_batch=torch.rand(2, 1, 16, 16),
            epoch=0,
            batch=torch.rand(2, 1, 16, 16),
        )


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "doob_bridge" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "doob_bridge" in TrainingStrategyConfigSchema.model_fields


# --- ILVR-style source structure anchor (B-1.10) -------------------------------------------
# The unconditional SDEdit bridge loses subject anatomy (val SSIM plateaus BELOW the copy-source
# identity baseline). anchor_scale > 0 replaces the LOW-FREQUENCY band of each reverse iterate
# with the source's low-frequency band (Choi et al. ILVR), pinning anatomy to the subject while
# the learned 7T score supplies contrast/detail. anchor_scale=0 keeps the bare bridge unchanged.


def _lowpass(z: torch.Tensor, scale: int) -> torch.Tensor:
    import torch.nn.functional as F

    h, w = z.shape[-2:]
    down = F.interpolate(
        z,
        size=(max(1, h // scale), max(1, w // scale)),
        mode="bilinear",
        align_corners=False,
    )
    return F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)


def test_anchor_scale_zero_is_backward_compatible() -> None:
    # anchor_scale=0 (default) must reproduce the bare bridge bit-for-bit (no existing arm moves).
    torch.manual_seed(0)
    m = _net().eval()
    acp = _acp()
    src = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        base = doob_h_transform_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=0)
        zero = doob_h_transform_sample(
            m, src, alphas_cumprod=acp, n_steps=20, seed=0, anchor_scale=0
        )
    assert torch.allclose(base, zero)


def test_anchor_preserves_source_low_frequency_structure() -> None:
    # The defining behaviour: with the anchor ON, the OUTPUT's low-frequency band tracks the
    # SOURCE's far more closely than the bare bridge -> subject anatomy is preserved.
    torch.manual_seed(0)
    m = _net().eval()
    acp = _acp()
    src = torch.rand(4, 1, 16, 16)
    scale = 4
    with torch.no_grad():
        off = doob_h_transform_sample(
            m, src, alphas_cumprod=acp, n_steps=20, strength=0.6, seed=0, anchor_scale=0
        )
        on = doob_h_transform_sample(
            m,
            src,
            alphas_cumprod=acp,
            n_steps=20,
            strength=0.6,
            seed=0,
            anchor_scale=scale,
        )
    src_lf = _lowpass(src, scale)
    err_off = float((_lowpass(off, scale) - src_lf).pow(2).mean())
    err_on = float((_lowpass(on, scale) - src_lf).pow(2).mean())
    assert err_on < 0.5 * err_off  # anchored low-freq is much closer to the source


def test_validation_forward_forwards_anchor_scale(monkeypatch) -> None:
    # ANTI-FACADE (#16): the strategy MUST read _db_anchor_scale and forward it to the sampler.
    import spectramr.infrastructure.training.strategies.doob_bridge_strategy as mod

    captured: dict = {}

    def _fake_sample(gen, source, **kw):
        captured.update(kw)
        return source

    monkeypatch.setattr(mod, "doob_h_transform_sample", _fake_sample)
    strat = object.__new__(DoobBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=_net())
    strat._db_timesteps = 40
    strat._db_schedule = "cosine"
    strat._db_residual = False
    strat._acp_cache = {}
    strat._db_sampling_steps = 10
    strat._db_strength = 0.6
    strat._db_h_scale = 1.0
    strat._db_eta = 0.0
    strat._db_val_seed = 0
    strat._db_anchor_scale = 8
    strat.config = types.SimpleNamespace(
        validation=block_stub("validation", val_chunk_size=0)
    )
    src = torch.rand(2, 1, 16, 16)
    strat._validation_forward(src, {"input": src})
    assert captured.get("anchor_scale") == 8


# --- Source-conditioned RESIDUAL detail bridge (b110_doob_residual / b110_ilvr_residual) ----
# residual=true learns the score of r = target - source CONDITIONED on the source, and the
# sampler composes out = source + r. This gives subject-faithful high-frequency detail the
# unconditional bridge (subject-generic) misses; the paired/registered/normalized volumes make
# r small and mostly high-frequency.


def test_residual_loss_depends_on_both_source_and_target() -> None:
    # The residual DSM target is r0 = target - source, so (unlike the unconditional loss)
    # changing EITHER the source or the target changes the loss. This is the residual seam.
    m = _rnet()
    acp = _acp()
    b1 = _batch()
    torch.manual_seed(7)
    l1 = float(
        compute_doob_residual_loss(m, b1, alphas_cumprod=acp)["loss_total"].detach()
    )
    b2 = dict(b1)
    b2["input"] = (
        b1["input"] + 0.3
    )  # different source, same target -> different residual
    torch.manual_seed(7)
    l2 = float(
        compute_doob_residual_loss(m, b2, alphas_cumprod=acp)["loss_total"].detach()
    )
    assert l1 != l2
    out = compute_doob_residual_loss(m, b1, alphas_cumprod=acp)
    assert {"loss_total", "loss_eps"} <= set(out) and torch.isfinite(out["loss_total"])


def test_residual_dsm_loss_reduces() -> None:
    torch.manual_seed(0)
    m = _rnet()
    acp = _acp()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_doob_residual_loss(m, batch, alphas_cumprod=acp)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_residual_sample_composes_source_plus_r_and_clamps() -> None:
    # Output = (source + r).clamp(0,1): correct shape and range regardless of the score.
    torch.manual_seed(0)
    m = _rnet().eval()
    acp = _acp()
    src = torch.rand(2, 1, 16, 16)
    with torch.no_grad():
        out = doob_residual_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=0)
    assert out.shape == src.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0
    # reproducible: same seed -> identical, different seed -> different init
    with torch.no_grad():
        again = doob_residual_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=0)
        other = doob_residual_sample(m, src, alphas_cumprod=acp, n_steps=20, seed=1)
    assert torch.allclose(out, again)
    assert not torch.allclose(out, other, atol=1e-4)


def test_residual_ilvr_anchor_zeroes_residual_lowfreq() -> None:
    # With the residual-space ILVR anchor ON (anchor_scale>=2), the residual's low-frequency band
    # is zeroed, so the OUTPUT's low-freq tracks the SOURCE far more closely than the bare
    # residual bridge -> output low-freq == source (subject anatomy preserved exactly).
    torch.manual_seed(0)
    m = _rnet().eval()
    acp = _acp()
    src = torch.rand(4, 1, 16, 16)
    scale = 4
    with torch.no_grad():
        off = doob_residual_sample(
            m, src, alphas_cumprod=acp, n_steps=20, strength=0.6, seed=0, anchor_scale=0
        )
        on = doob_residual_sample(
            m,
            src,
            alphas_cumprod=acp,
            n_steps=20,
            strength=0.6,
            seed=0,
            anchor_scale=scale,
        )
    src_lf = _lowpass(src, scale)
    err_off = float((_lowpass(off, scale) - src_lf).pow(2).mean())
    err_on = float((_lowpass(on, scale) - src_lf).pow(2).mean())
    assert err_on < 0.5 * err_off


def test_residual_flag_routes_loss_and_validation(monkeypatch) -> None:
    # ANTI-FACADE (#16): _db_residual MUST route BOTH the loss (compute_doob_residual_loss) and
    # validation (doob_residual_sample). With the flag on, the residual paths are called; off,
    # the unconditional paths are.
    import spectramr.infrastructure.training.strategies.doob_bridge_strategy as mod

    calls: dict = {}

    def _fake_residual_loss(gen, batch, *, alphas_cumprod):
        calls["loss"] = "residual"
        return {"loss_total": torch.tensor(0.0), "loss_eps": torch.tensor(0.0)}

    def _fake_uncond_loss(gen, batch, *, alphas_cumprod):
        calls["loss"] = "uncond"
        return {"loss_total": torch.tensor(0.0), "loss_eps": torch.tensor(0.0)}

    def _fake_residual_sample(gen, source, **kw):
        calls["sample"] = "residual"
        return source

    def _fake_uncond_sample(gen, source, **kw):
        calls["sample"] = "uncond"
        return source

    monkeypatch.setattr(mod, "compute_doob_residual_loss", _fake_residual_loss)
    monkeypatch.setattr(mod, "compute_doob_loss", _fake_uncond_loss)
    monkeypatch.setattr(mod, "doob_residual_sample", _fake_residual_sample)
    monkeypatch.setattr(mod, "doob_h_transform_sample", _fake_uncond_sample)

    strat = object.__new__(DoobBridgeStrategy)
    strat.env = types.SimpleNamespace(generator=_rnet())
    strat._db_timesteps = 40
    strat._db_schedule = "cosine"
    strat._acp_cache = {}
    strat._db_sampling_steps = 10
    strat._db_strength = 0.6
    strat._db_h_scale = 1.0
    strat._db_eta = 0.0
    strat._db_val_seed = 0
    strat._db_anchor_scale = 0
    strat.config = types.SimpleNamespace(
        validation=block_stub("validation", val_chunk_size=0)
    )
    src = torch.rand(2, 1, 16, 16)

    strat._db_residual = True
    strat._compute_losses_impl(batch=_batch())
    strat._validation_forward(src, {"input": src})
    assert calls == {"loss": "residual", "sample": "residual"}

    calls.clear()
    strat._db_residual = False
    strat._compute_losses_impl(batch=_batch())
    strat._validation_forward(src, {"input": src})
    assert calls == {"loss": "uncond", "sample": "uncond"}
