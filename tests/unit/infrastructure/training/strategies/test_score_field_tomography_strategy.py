"""Score-field tomography: validation reconstructs by annealed Langevin and emits val_psnr/val_ssim.

Drives the production resolve path: ``TrainingStrategyFactory`` resolves the
class from ``training_mode``, and ``validation_step`` resolves its sampler
through ``SamplerRegistry``; an oracle two-head stub observes every call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.services.metric_keys import early_stop_monitor_candidates
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.diffusion.samplers.annealed_langevin_dc import (
    AnnealedLangevinDCSampler,
    interleaved_to_complex,
)


class _OracleScoreField(nn.Module):
    """``s_x = -(x - truth) / sigma**2`` and a zero ``s_c``; records each call's ``c``."""

    def __init__(self, truth: torch.Tensor, cond_dim: int) -> None:
        super().__init__()
        self.truth = truth
        self.cond_dim = cond_dim
        self.conds: list[torch.Tensor] = []
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, sigma, c):
        self.conds.append(c)
        s_x = -(x - self.truth) / sigma.view(-1, 1, 1, 1) ** 2
        return s_x, torch.zeros(x.shape[0], self.cond_dim)


def _undersampling(**over) -> SimpleNamespace:
    kw = {
        "acceleration_type": "equispaced",
        "base_acceleration": 4.0,
        "center_fraction": 0.125,
        "density_power": 1.5,
        "mask_seed": None,
        "mask_direction": None,
    }
    kw.update(over)
    return SimpleNamespace(**kw)


def _strategy(model: nn.Module, phase_encode_axis=None, **us_over):
    config = SimpleNamespace(
        training=SimpleNamespace(strategy_class=None, training_mode="score_field_tomography"),
        undersampling=_undersampling(**us_over),
        run=SimpleNamespace(seed=7),
        validation=SimpleNamespace(
            scoring=SimpleNamespace(compute=["psnr", "ssim"], primary="psnr")
        ),
        metrics=SimpleNamespace(domain="image"),
    )
    if phase_encode_axis is not None:
        config.data = SimpleNamespace(
            acquisition=SimpleNamespace(phase_encode_axis=phase_encode_axis)
        )
    cls = TrainingStrategyFactory().get_strategy_class(config)
    strat = cls.__new__(cls)
    strat.config = config
    strat.device = torch.device("cpu")
    strat.env = SimpleNamespace(models={"generator": model}, generator=model, config=config)
    strat.adapter_chains = {}
    strat.sigma_min, strat.sigma_max, strat.cond_dim = 1e-3, 1.0, 5
    strat.val_num_levels, strat.val_steps_per_level, strat.val_step_size = 5, 3, 0.3
    return strat


def test_validation_step_reconstructs_through_the_registered_sampler(monkeypatch) -> None:
    torch.manual_seed(0)
    truth = torch.rand(2, 2, 16, 16)
    model = _OracleScoreField(truth, cond_dim=5)
    strat = _strategy(model)
    acq = torch.arange(10, dtype=torch.float32).view(2, 5)

    built = []
    real_get = AnnealedLangevinDCSampler.__init__

    def spy(self, **kw):
        built.append(kw)
        real_get(self, **kw)

    monkeypatch.setattr(AnnealedLangevinDCSampler, "__init__", spy)
    metrics = strat.validation_step(
        torch.zeros_like(truth), truth, batch_data={"acquisition_params": acq}
    )

    # Bare names, as the default validation step emits; the loop's monitor
    # resolution maps the arm's ``val_psnr`` onto ``psnr``.
    assert {"psnr", "ssim"} <= set(metrics)
    assert "psnr" in early_stop_monitor_candidates("val_psnr")
    assert built and built[0]["num_levels"] == 5 and built[0]["seed"] == 7
    assert len(model.conds) == 5 * 3 + 1
    assert all(torch.equal(c, acq) for c in model.conds)  # fixed c* from the batch
    assert model.training  # train mode restored

    # Beats the zero-filled reconstruction from the same mask.
    mask = strat._validation_mask(16, 16, torch.device("cpu"))
    zf = ifft2c(fft2c(interleaved_to_complex(truth)) * mask).abs()
    zf_metrics = strat._score_validation_output(zf, interleaved_to_complex(truth).abs())
    assert metrics["psnr"] > zf_metrics["psnr"]


def test_validation_mask_reads_the_declared_block() -> None:
    strat = _strategy(_OracleScoreField(torch.zeros(1, 2, 32, 32), 5))
    mask = strat._validation_mask(32, 32, torch.device("cpu"))
    rows = mask[0, 0, :, 0]
    assert mask.shape == (1, 1, 32, 32)
    assert rows[14:18].all()  # ACS = int(0.125 * 32) centred lines
    assert rows[::4].all()  # equispaced R=4
    assert strat._undersampling_record()["seed"] == 7


def test_mask_direction_resolves_through_the_phase_encode_axis() -> None:
    model = _OracleScoreField(torch.zeros(1, 2, 32, 32), 5)
    strat = _strategy(model, phase_encode_axis=-1, mask_direction="phase")
    assert strat._undersampling_record()["line_axis"] == "x"
    mask = strat._validation_mask(32, 24, torch.device("cpu"))
    assert mask.shape == (1, 1, 32, 24)
    assert mask[0, 0, 0, ::4].all() and torch.equal(mask[0, 0, 0], mask[0, 0, 5])  # columns


def test_unsupported_undersampling_raises() -> None:
    model = _OracleScoreField(torch.zeros(1, 2, 8, 8), 5)
    with pytest.raises(ValueError, match="phase-encode axis"):
        _strategy(model, mask_direction="phase")._undersampling_record()
    with pytest.raises(ValueError):
        _strategy(model, acceleration_type="no_such_pattern")._undersampling_record()


def test_first_steps_snapshot_stamps_the_sampler_knobs_and_mask() -> None:
    strat = _strategy(_OracleScoreField(torch.zeros(1, 2, 8, 8), 5))
    stamp = strat._first_steps_extra(0)["score_field_tomography"]
    assert stamp["validation_sampler"] == "annealed_langevin_dc"
    assert (stamp["val_num_levels"], stamp["val_steps_per_level"], stamp["val_step_size"]) == (
        5,
        3,
        0.3,
    )
    assert stamp["validation_undersampling"]["mask_type"] == "equispaced"
    assert stamp["validation_undersampling"]["line_axis"] == "y"


def test_training_conditions_on_the_published_acquisition_vector() -> None:
    truth = torch.rand(2, 2, 8, 8)
    model = _OracleScoreField(truth, cond_dim=5)
    strat = _strategy(model)
    strat._dim_x = None
    strat._declare_model_input = lambda *a, **k: None
    strat._sample_sigma = lambda n, device: torch.full((n,), 1e-4)  # c_noisy ~= c
    acq = torch.full((2, 5), 3.0)
    # ``train_step`` forwards the loader batch as ``batch=``.
    losses = strat._compute_losses_impl(
        input_batch=truth, target_batch=truth, epoch=0, batch={"acquisition_params": acq}
    )
    assert torch.isfinite(losses["g_total_loss"])
    assert model.conds[0].shape == (2, 5)
    assert torch.allclose(model.conds[0], acq, atol=1e-2)


def test_complex_image_from_the_ifft_adapter_is_interleaved_for_the_model() -> None:
    torch.manual_seed(1)
    truth = torch.rand(2, 2, 16, 16)
    model = _OracleScoreField(truth, cond_dim=5)
    strat = _strategy(model)
    z = interleaved_to_complex(truth)
    via_complex = strat.validation_step(z, z, batch_data=None)
    via_real = strat.validation_step(truth, truth, batch_data=None)
    assert via_complex["psnr"] == pytest.approx(via_real["psnr"])
    assert model.conds[0].shape == (2, 5)  # inferred c*, padded to cond_dim
