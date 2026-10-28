"""Tests for HeteroscedasticULFStrategy (B-2.9)."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.strategies.heteroscedastic_ulf_strategy import (
    HeteroscedasticULFStrategy,
    compute_heteroscedastic_ulf_losses,
)


class _TwoChannelNet(nn.Module):
    """Tiny 1->2 (mean, logvar) net for the strategy tests."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _batch() -> dict:
    return {
        "input": torch.rand(2, 1, 16, 16),
        "target": torch.rand(2, 1, 16, 16),
        "field_strength": torch.tensor([0.1, 0.1]),
    }


def test_compute_losses_keys_and_finite() -> None:
    out = compute_heteroscedastic_ulf_losses(_TwoChannelNet(), _batch())
    assert {"loss_total", "loss_nll", "loss_var_prior"} <= set(out)
    assert torch.isfinite(out["loss_total"])


def test_requires_two_channel_output() -> None:
    one_ch = nn.Conv2d(1, 1, 3, padding=1)
    with pytest.raises(ValueError):
        compute_heteroscedastic_ulf_losses(one_ch, _batch())


def test_nll_reduces_under_training() -> None:
    torch.manual_seed(0)
    net = _TwoChannelNet()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    batch = _batch()
    first = None
    out = None
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        out = compute_heteroscedastic_ulf_losses(net, batch, lambda_var_prior=0.0)
        out["loss_total"].backward()
        opt.step()
        if first is None:
            first = float(out["loss_nll"].detach())
    assert out is not None and first is not None
    assert float(out["loss_nll"].detach()) < first


def test_validation_forward_returns_two_channels_for_calibration() -> None:
    # Validation returns the full [mean, logvar] output so gaussian_nll can score the
    # variance (the B-2.9 calibration claim is measured, not invisible — pitfall #18).
    strat = object.__new__(HeteroscedasticULFStrategy)
    strat.env = types.SimpleNamespace(generator=_TwoChannelNet())
    x = torch.rand(2, 1, 16, 16)
    pred = strat._validation_forward(x, {"use_dc": False})
    assert pred.shape == (2, 2, 16, 16)  # [mean, logvar]
    # the mean channel is recoverable for any mean-only use
    assert pred[:, 0:1].shape == (2, 1, 16, 16)


def test_apply_metric_transforms_is_noop_for_distribution_head() -> None:
    """The 2-channel [mean, logvar] head must NOT be metric-transformed.

    The base auto-heuristic reads ``pred.shape[1] == 2`` as complex real/imag and
    selects the ``magnitude`` transform, which does ``torch.complex(target[:,0],
    target[:,1])`` on the 1-channel target -> IndexError, then a fallback PSNR of
    -14 (the b29 crash). The strategy overrides the transform to a no-op so the
    2-channel head flows to ``gaussian_nll`` (which slices it) untouched.
    """
    strat = object.__new__(HeteroscedasticULFStrategy)
    strat.config = types.SimpleNamespace(validation=None, data=None)
    pred = torch.rand(2, 2, 16, 16)  # [mean, logvar]
    target = torch.rand(2, 1, 16, 16)  # 1-channel clean image
    out_pred, out_target = strat._apply_metric_transforms(pred, target, None)
    assert out_pred.shape == (2, 2, 16, 16)  # 2ch preserved for gaussian_nll
    assert out_target.shape == (2, 1, 16, 16)
    assert torch.equal(out_pred, pred)
    assert torch.equal(out_target, target)


def test_prediction_for_visualization_returns_mean_channel() -> None:
    """The saved fake image must be the MEAN (channel 0), not sqrt(mean^2 + logvar^2).

    ``_validation_forward`` hands the base viz path the full 2-channel [mean, logvar]
    head; the base ``to_magnitude`` would RSS-blend it into the b29 dark-brain-on-grey
    wrong-channel image (issue #371). The strategy overrides the viz reducer to slice
    channel 0 so a single-channel mean flows through ``to_magnitude`` untouched.
    """
    strat = object.__new__(HeteroscedasticULFStrategy)
    mean = torch.rand(2, 1, 16, 16)
    logvar = torch.rand(2, 1, 16, 16)
    pred = torch.cat([mean, logvar], dim=1)  # [mean, logvar]
    vis = strat._prediction_for_visualization(pred)
    assert vis.shape == (2, 1, 16, 16)  # reduced to the mean channel
    assert torch.equal(vis, mean)  # exactly channel 0, NOT an RSS blend
    # a 1-channel prediction is passed through unchanged
    one_ch = torch.rand(2, 1, 16, 16)
    assert torch.equal(strat._prediction_for_visualization(one_ch), one_ch)


def test_gaussian_nll_metric_scores_noop_transformed_head() -> None:
    """End-to-end: after the no-op transform, the registered ``gaussian_nll``
    metric consumes the 2-channel head + 1-channel target and returns a finite
    score (val_gaussian_nll is actually emitted, not the -14 PSNR fallback)."""
    import math

    from spectramr.core.metrics.registry import MetricsRegistry

    strat = object.__new__(HeteroscedasticULFStrategy)
    strat.config = types.SimpleNamespace(validation=None, data=None)
    pred = torch.rand(2, 2, 16, 16)
    target = torch.rand(2, 1, 16, 16)
    p, t = strat._apply_metric_transforms(pred, target, None)
    metric = MetricsRegistry.get("gaussian_nll")
    assert math.isfinite(float(metric(p, t)))


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    assert "heteroscedastic_ulf" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS


def test_hoult_sigma_clamped_to_data_scale() -> None:
    # Regression for the review's CRITICAL scale bug: at 0.1T the raw Hoult std is
    # ~19.2 (absolute units), meaningless on [0,1] data. It MUST be clamped to
    # max_sigma so the variance anchor stays on the data scale.
    from spectramr.infrastructure.training.strategies.heteroscedastic_ulf_strategy import (
        _hoult_sigma_per_sample,
    )

    sig = _hoult_sigma_per_sample(torch.tensor([0.1, 0.1]), sigma_floor=0.05, max_sigma=0.2)
    assert float(sig.max()) <= 0.2 + 1e-6
    # high field is not clamped (sigma_floor < max_sigma)
    sig_hf = _hoult_sigma_per_sample(torch.tensor([3.0]), sigma_floor=0.05, max_sigma=0.2)
    assert abs(float(sig_hf) - 0.05) < 1e-6


def test_var_prior_not_degenerate_at_ulf() -> None:
    # With the clamp, the variance-prior term is the SAME order of magnitude as the
    # NLL on [0,1] data at 0.1T (was ~1e7x larger, gradient-starving the mean).
    net = _TwoChannelNet()
    batch = _batch()  # field_strength 0.1T
    out = compute_heteroscedastic_ulf_losses(net, batch, lambda_var_prior=0.1, max_sigma=0.2)
    nll = float(out["loss_nll"].detach())
    vp = float(out["loss_var_prior"].detach())
    assert vp < 100.0 * max(nll, 1e-3)  # not orders of magnitude larger


def test_compute_losses_accepts_canonical_trainingbatch() -> None:
    # REGRESSION (cohort guard, 2026-06-16): the canonical training pipeline converts the
    # loader dict to a TrainingBatch (BatchAdapter.from_dict) and forwards it as
    # batch=<TrainingBatch>, which is NOT a dict. The original `isinstance(batch, dict)`
    # guard rejected it and crashed every MICCAI arm at step 0; the dict-only helper tests
    # never exercised this path. The guard must accept any mapping exposing .get.
    import types

    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    strat = object.__new__(HeteroscedasticULFStrategy)
    strat.env = types.SimpleNamespace(generator=_TwoChannelNet())

    out = strat._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["loss_total"])


def test_predicts_distribution_params_flag_set() -> None:
    """The strategy advertises a distributional head (2-ch [mean, logvar] for a
    1-ch target) so the base train_step width guard defers instead of raising."""
    assert HeteroscedasticULFStrategy.predicts_distribution_params is True


def test_train_step_width_guard_allows_distribution_head() -> None:
    """REGRESSION (b29_heteroscedastic_ulf, re-reproduced 2026-06-24): the base
    ``train_step`` channel-width guard raised ``[DomainMismatch] Model outputs 2
    channels, but target dataset provided 1`` at iter 1, killing the run before
    any step. The strategy's ``predicts_distribution_params`` flag must make the
    guard defer (mirroring the evidential_unet escape). We assert the guard is
    not the thing that rejects, by driving one full step.
    """
    import types

    st = object.__new__(HeteroscedasticULFStrategy)
    st.config = types.SimpleNamespace(
        model=types.SimpleNamespace(
            model_type="configurable_unet", in_channels=1, out_channels=2
        ),
        training=types.SimpleNamespace(strategy_class="heteroscedastic_ulf"),
    )
    st.adapter_chains = {}
    st.env = types.SimpleNamespace(
        discriminator=None, generator=_TwoChannelNet(), opt_g=None
    )
    st._prepare_model_input = lambda x: x  # type: ignore[method-assign]
    st.device = torch.device("cpu")
    inp = torch.rand(1, 1, 8, 8)
    tgt = torch.rand(1, 1, 8, 8)  # 1-ch target vs 2-ch out — the guard's trigger
    # The step must complete: a stub that stops short of the guard would pass
    # this test without ever exercising it.
    st.train_step((inp, tgt), epoch=0, input_batch=inp, target_batch=tgt, iteration=1)


def test_paired_route_anchor_uses_declared_source_field() -> None:
    # The paired ULF route stamps no per-sample field_strength; the declared 64 mT
    # source field must anchor sigma (clamped to max_sigma), not the 3T floor.
    from spectramr.infrastructure.training.strategies.heteroscedastic_ulf_strategy import (
        _hoult_sigma_per_sample,
    )

    sig_ulf = _hoult_sigma_per_sample(None, sigma_floor=0.05, max_sigma=0.2, source_field=0.064)
    assert abs(float(sig_ulf.max()) - 0.2) < 1e-6
    sig_hf = _hoult_sigma_per_sample(None, sigma_floor=0.05, max_sigma=0.2, source_field=3.0)
    assert abs(float(sig_hf.max()) - 0.05) < 1e-6


def test_missing_source_field_raises_not_floor() -> None:
    from spectramr.infrastructure.training.strategies.heteroscedastic_ulf_strategy import (
        _hoult_sigma_per_sample,
    )

    with pytest.raises(ValueError, match="source field"):
        _hoult_sigma_per_sample(None, sigma_floor=0.05, max_sigma=0.2)
    batch = {"input": torch.rand(2, 1, 16, 16), "target": torch.rand(2, 1, 16, 16)}
    with pytest.raises(ValueError, match="source field"):
        compute_heteroscedastic_ulf_losses(_TwoChannelNet(), batch)


def test_declared_source_field_anchor_is_built_on_the_requested_device() -> None:
    from spectramr.infrastructure.training.strategies.heteroscedastic_ulf_strategy import (
        _hoult_sigma_per_sample,
    )

    sig = _hoult_sigma_per_sample(
        None, sigma_floor=0.05, max_sigma=0.2, source_field=0.064, device=torch.device("meta")
    )
    assert sig.device.type == "meta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_paired_route_losses_run_on_cuda_without_device_mix() -> None:
    net = _TwoChannelNet().cuda()
    batch = {"input": torch.rand(2, 1, 16, 16).cuda(), "target": torch.rand(2, 1, 16, 16).cuda()}
    losses = compute_heteroscedastic_ulf_losses(
        net, batch, lambda_var_prior=0.1, source_field=0.064
    )
    assert all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
