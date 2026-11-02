"""Tests for FieldGuidedDiffusionStrategy (B-3.5 continuous-field guided diffusion)."""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
    FieldGuidedDiffusionStrategy,
    compute_field_guided_diffusion_loss,
    field_guided_sample,
    make_alphas_cumprod,
    q_sample,
)
from spectramr.models.generators.field_guided_score_unet import FieldGuidedScoreUNet
from tests.utils.config_block_stub import block_stub


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 3.0]),
        "field_strength_target": torch.tensor([7.0, 1.5]),
    }


def test_alphas_cumprod_monotone_decreasing() -> None:
    for kind in ("cosine", "linear"):
        acp = make_alphas_cumprod(50, kind)
        assert acp.shape == (50,)
        assert torch.all(acp[1:] <= acp[:-1] + 1e-6)  # non-increasing in t
        assert float(acp[0]) > float(acp[-1])  # genuinely decays


def test_q_sample_shape_and_endpoints() -> None:
    acp = make_alphas_cumprod(100, "cosine")
    x0 = torch.rand(3, 1, 8, 8)
    noise = torch.randn_like(x0)
    xt = q_sample(x0, torch.tensor([0, 50, 99]), noise, acp)
    assert xt.shape == x0.shape


def test_loss_keys_and_finite() -> None:
    acp = make_alphas_cumprod(50, "cosine")
    out = compute_field_guided_diffusion_loss(
        FieldGuidedScoreUNet(width=16, time_dim=16), _batch(), alphas_cumprod=acp
    )
    assert {"loss_total", "loss_eps"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    acp = make_alphas_cumprod(50, "cosine")
    m = FieldGuidedScoreUNet(width=24, time_dim=16)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = compute_field_guided_diffusion_loss(m, batch, alphas_cumprod=acp)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_total"].detach())
    assert out is not None and first is not None
    assert float(out["loss_total"].detach()) < first


def test_sample_shape_matches_source() -> None:
    acp = make_alphas_cumprod(50, "cosine")
    m = FieldGuidedScoreUNet(width=16, time_dim=16).eval()
    src = torch.rand(2, 1, 16, 16)
    out = field_guided_sample(
        m,
        src,
        torch.tensor([7.0, 1.5]),
        alphas_cumprod=acp,
        n_steps=6,
        guidance_scale=3.0,
    )
    assert out.shape == src.shape


def test_guidance_changes_output_after_training() -> None:
    # After a few steps the net is non-identity, so guided (scale=3) and unguided
    # (scale=1) sampling must differ — the field-CFG mechanism is genuinely applied.
    torch.manual_seed(0)
    acp = make_alphas_cumprod(40, "cosine")
    m = FieldGuidedScoreUNet(width=24, time_dim=16)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    batch = _batch()
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        compute_field_guided_diffusion_loss(m, batch, alphas_cumprod=acp)[
            "loss_total"
        ].backward()
        opt.step()
    m.eval()
    src = torch.rand(1, 1, 16, 16)
    b_t = torch.tensor([7.0])
    torch.manual_seed(1)
    guided = field_guided_sample(
        m, src, b_t, alphas_cumprod=acp, n_steps=8, guidance_scale=3.0
    )
    torch.manual_seed(1)
    plain = field_guided_sample(
        m, src, b_t, alphas_cumprod=acp, n_steps=8, guidance_scale=1.0
    )
    assert not torch.allclose(guided, plain)


def test_validation_step_end_to_end_no_batch_context_collision() -> None:
    # Drive the FULL validation_step chain (the batch_context double-pass regression).
    acp_net = FieldGuidedScoreUNet(width=16, time_dim=16).eval()
    strat = object.__new__(FieldGuidedDiffusionStrategy)
    strat.env = types.SimpleNamespace(generator=acp_net)
    strat._fgd_timesteps = 30
    strat._fgd_schedule = "cosine"
    strat._fgd_guidance_scale = 2.0
    strat._fgd_sampling_steps = 5
    strat._acp_cache = {}
    _cfg = types.SimpleNamespace(
        training=None,
        validation=block_stub(
            "validation", compute_image_metrics=False, val_chunk_size=0
        ),
    )
    strat.state = types.SimpleNamespace(config=_cfg)
    strat.config = (
        _cfg  # _validation_forward reads self.config.validation.loader.chunk_size
    )
    out = strat.validation_step(
        torch.rand(2, 1, 16, 16),
        torch.rand(2, 1, 16, 16),
        field_strength_target=torch.tensor([7.0, 1.5]),
    )
    assert isinstance(out, dict)  # returned without TypeError


def test_linear_schedule_total_noise_is_t_invariant() -> None:
    # Fix (#2): the rescaled linear betas keep the terminal noise level ~constant
    # across timesteps (otherwise small T under-noises x_T vs the DDIM init).
    acp_short = make_alphas_cumprod(50, "linear")
    acp_long = make_alphas_cumprod(250, "linear")
    # terminal alphas_cumprod (how much signal survives at t=T) should be close.
    assert abs(float(acp_short[-1]) - float(acp_long[-1])) < 0.05


class _RecScore(torch.nn.Module):
    """Records the per-call batch size to verify validation chunking."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[int] = []

    def forward(
        self, x, *, timesteps, field_strength, cond_image=None, field_cond=None, **_
    ):
        self.seen.append(int(x.shape[0]))
        return torch.zeros_like(x)


def _val_strat(rec, val_chunk_size):
    strat = object.__new__(FieldGuidedDiffusionStrategy)
    strat.env = types.SimpleNamespace(generator=rec)
    strat._fgd_timesteps = 20
    strat._fgd_schedule = "cosine"
    strat._fgd_guidance_scale = 1.0
    strat._fgd_sampling_steps = 2
    strat._acp_cache = {}
    strat.config = types.SimpleNamespace(
        validation=block_stub("validation", val_chunk_size=val_chunk_size)
    )
    return strat


def test_validation_chunking_is_load_bearing() -> None:
    # Fix (#4): val_chunk_size must actually micro-batch the reverse sampler (the OOM
    # guard), not be an inert knob. With chunk=1 over a batch of 3, the score net is
    # only ever called with batch size 1; the output still has the full batch.
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
    assert set(rec.seen) == {3}  # full batch in one pass


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "field_guided_diffusion" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "field_guided_diffusion" in TrainingStrategyConfigSchema.model_fields


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(FieldGuidedDiffusionStrategy)
    strat.env = types.SimpleNamespace(
        generator=FieldGuidedScoreUNet(width=16, time_dim=16)
    )
    strat._fgd_timesteps = 10
    strat._fgd_schedule = "cosine"
    strat._acp_cache = {}

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


# --- Multi-contrast: contrast_id threaded to the score net (train + CFG sampler) ---


def test_contrast_id_threaded_train_and_sampler() -> None:
    from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
        compute_field_guided_diffusion_loss,
        field_guided_sample,
    )

    seen: dict = {}

    def spy(
        x,
        *,
        timesteps,
        field_strength,
        cond_image=None,
        field_cond=None,
        contrast_id=None,
        **_,
    ):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    acp = torch.linspace(0.99, 0.01, 5)
    cid = torch.tensor([0, 2])
    batch = {
        "target": torch.rand(2, 1, 8, 8),
        "input": torch.rand(2, 1, 8, 8),
        "field_strength_target": torch.tensor([7.0, 7.0]),
        "contrast_id": cid,
    }
    compute_field_guided_diffusion_loss(
        spy, batch, alphas_cumprod=acp, guidance_prob=0.0
    )
    assert seen["cid"] is cid
    seen.clear()
    field_guided_sample(
        spy,
        torch.rand(2, 1, 8, 8),
        torch.tensor([7.0, 7.0]),
        alphas_cumprod=acp,
        n_steps=1,
        guidance_scale=1.0,
        contrast_id=cid,
    )
    assert seen["cid"] is cid


def test_cfg_sampling_output_stays_in_data_range() -> None:
    """CFG extrapolates; the x0 estimate must be clamped or the DDIM loop diverges.

    b35_field_cfg runs guidance_scale=3.0, so eps leaves the region the model was
    trained on and the resulting x0_hat is fed straight back into the next step —
    the error compounds over all 50 steps. Unclamped, the full run drove PSNR to
    -20.8 dB with NEGATIVE SSIM (anti-correlated with the target). The sibling
    samplers (ulf_dps, generative_refiner, doob_bridge) already carry this guard;
    its absence here was a divergent-sibling omission.
    """

    def _runaway_model(x, **kwargs):
        # A confidently-wrong eps: guidance then extrapolates it 3x every step.
        return torch.full_like(x, 25.0)

    acp = make_alphas_cumprod(timesteps=20)
    out = field_guided_sample(
        _runaway_model,
        torch.rand(2, 1, 8, 8),
        torch.tensor([7.0, 7.0]),
        alphas_cumprod=acp,
        n_steps=10,
        guidance_scale=3.0,
    )

    assert torch.isfinite(out).all(), "CFG sampling produced non-finite values"
    # The RETURNED sample must land in the [0,1] data range the metrics grade against
    # (mrixfields images are [0,1]). The per-step x0_hat clamp to [-1,2] only kills the
    # divergence; without a FINAL clamp the returned tensor stays in [-1,2] and SSIM/PSNR
    # are computed off-scale (the residual b35_field_cfg -3.22 dB / SSIM 0.004 defect).
    # The healthy siblings (doob_bridge, ulf_dps, generative_refiner) all clamp on return.
    assert float(out.min()) >= 0.0 - 1e-5 and float(out.max()) <= 1.0 + 1e-5, (
        f"sampler returned outside the [0,1] data range: "
        f"[{float(out.min()):.3f}, {float(out.max()):.3f}] — images are [0,1], so an "
        "out-of-range return is what produced negative PSNR / ~0 SSIM in the b35 run"
    )


def test_declared_image_loss_is_folded_onto_the_x0_estimate() -> None:
    """A declared image loss reaches the objective on the x0 estimate the sampler uses."""
    acp = make_alphas_cumprod(50, "cosine")
    batch = _batch()
    model = FieldGuidedScoreUNet(width=16, time_dim=16)

    torch.manual_seed(7)
    off = compute_field_guided_diffusion_loss(model, batch, alphas_cumprod=acp)

    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fold(x0_hat, x0, components):
        seen.append((x0_hat, x0))
        return 0.5 * (x0_hat - x0).abs().mean()

    torch.manual_seed(7)
    on = compute_field_guided_diffusion_loss(
        model, batch, alphas_cumprod=acp, fold_image_losses=fold
    )

    ((x0_hat, x0),) = seen
    assert torch.equal(x0, batch["target"])
    assert torch.isfinite(x0_hat).all()
    assert torch.allclose(on["loss_total"] - off["loss_total"], 0.5 * (x0_hat - x0).abs().mean())
    assert "loss_eps" in on


def test_it_declares_folded_image_losses_with_an_empty_stack_as_design() -> None:
    assert FieldGuidedDiffusionStrategy.__dict__["folds_image_losses"] is True
    assert FieldGuidedDiffusionStrategy.__dict__["empty_loss_stack_is_design"] is True
    assert FieldGuidedDiffusionStrategy.__dict__["inline_losses"] == frozenset()
