"""IB active acquisition through the production resolve path, on tiny CPU k-space.

The arm YAML is loaded into ``TrainingSettings`` and resolved through
``TrainingStrategyFactory``; the strategy is then built bare (no model, no DI) and
configured from those settings, so the knob read, the in-step masking, the prior and
the validation rollouts run as under ``train``, with the arm's own ``pre_model``
adapters. The inherited metric pass is stubbed to record what it is handed.
"""

from __future__ import annotations

import pathlib
from types import SimpleNamespace

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.builders.leaf.adapter_builders import AdapterChainBuilder
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.ib_active_acquisition_strategy import (
    IBActiveAcquisitionStrategy,
)
from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import MetricsMixin
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

ARM = (
    pathlib.Path(__file__).resolve().parents[5]
    / "experiments/inprogress/novel_2026/idea_4_ib_active_acquisition.yaml"
)
W = 32


@pytest.fixture(scope="module")
def settings() -> TrainingSettings:
    return TrainingSettings.from_yaml(str(ARM))


def _strategy(settings: TrainingSettings) -> IBActiveAcquisitionStrategy:
    cls = TrainingStrategyFactory().get_strategy_class(settings)
    assert cls is IBActiveAcquisitionStrategy
    s = object.__new__(cls)
    s.config = settings
    s.device = torch.device("cpu")
    s._configure_ib(settings)
    s._ib_mask_generator = torch.Generator().manual_seed(0)
    s.adapter_chains = AdapterChainBuilder(settings.adapters).build()
    return s


def _kspace(batch: int = 2) -> torch.Tensor:
    """Real/imag interleaved single-coil k-space of a smooth random image."""
    torch.manual_seed(0)
    image = torch.nn.functional.avg_pool2d(torch.rand(batch, 1, 24, W), 3, 1, 1)
    k = fft2c(image.to(torch.complex64))
    return torch.cat([k.real, k.imag], dim=1)


def test_the_arm_knobs_reach_the_strategy(settings) -> None:
    s = _strategy(settings)
    assert (s.beta, s.mode, s.acceleration, s.center_fraction) == (
        1.0,
        "linear_gaussian",
        4.0,
        0.08,
    )
    assert s.phase_encode_axis == -1
    assert s.ib_provenance()["validation_rollouts"] == ["random", "ib", "variance", "equispaced"]
    assert IBActiveAcquisitionStrategy.standalone_infer_blocker(settings)


def test_training_input_is_masked_to_the_budget_and_feeds_the_prior(settings) -> None:
    s = _strategy(settings)
    k = _kspace()
    out = s._prepare_model_input(k)
    acquired = out.abs().sum(dim=(1, 2)) > 0
    assert (acquired.sum(-1) == round(W / 4.0)).all()
    assert s.line_prior.seen


def test_the_training_step_never_copies_device_to_host(settings, monkeypatch) -> None:
    s = _strategy(settings)
    k = _kspace()
    s._prepare_model_input(k)  # the once-per-split acquired-line check may read counts

    def _no_host_copy(self, *args, **kwargs):
        raise AssertionError("device-to-host copy inside the training step")

    monkeypatch.setattr(torch.Tensor, "cpu", _no_host_copy)
    s._prepare_model_input(k)


def test_validation_grades_random_masks_and_every_rollout(settings, monkeypatch) -> None:
    s = _strategy(settings)
    for _ in range(3):
        s._prepare_model_input(_kspace(4))
    seen: list[torch.Tensor] = []

    def _grade(self, input_batch, target_batch, **kwargs):
        seen.append((input_batch, target_batch))
        return {"psnr": float(len(seen)), "ssim": 0.5}

    monkeypatch.setattr(MetricsMixin, "validation_step", _grade)
    s.on_validation_start()
    k = _kspace()
    metrics = s.validation_step(k, k, batch_idx=0)
    assert len(seen) == 4
    # pre_model ran: the network gets the 1-channel magnitude image, as in training.
    assert all(inp.shape[1] == 1 and tgt.shape[1] == 1 for inp, tgt in seen)
    assert torch.allclose(seen[0][1], seen[3][1])  # the target is never masked
    assert metrics["val_psnr"] == 1.0 and "psnr" not in metrics
    assert s.acquired_lines_record["val"] == {"declared": "measured", "measured": [W]}
    for policy in ("ib", "variance", "equispaced"):
        assert f"val_{policy}_psnr" in metrics and f"val_{policy}_ssim" in metrics
        assert f"val_{policy}_psnr_gain" in metrics
    assert {int(m.sum()) for m in s._policy_masks.values()} == {round(W / 4.0)}
    assert set(s._policy_masks) == {"ib", "variance"}
    assert -1.0 <= metrics["val_ib_noise_signal_corr"] <= 1.0


def test_the_learned_prior_survives_a_checkpoint_round_trip(settings) -> None:
    s = _strategy(settings)
    s._owned_modules, s._owned_wrappers = {}, {}
    s.env = SimpleNamespace(generator=None, discriminator=None, losses={})
    s._prepare_model_input(_kspace(4))
    state = s.strategy_state_dict()
    fresh = _strategy(settings)
    fresh._owned_modules, fresh._owned_wrappers = {}, {}
    fresh.load_strategy_state_dict(state)
    assert fresh.line_prior.rows == s.line_prior.rows
    assert torch.equal(fresh.line_prior.cov_sum, s.line_prior.cov_sum)


def test_validation_before_training_skips_the_prior_rollouts(settings, monkeypatch) -> None:
    s = _strategy(settings)
    monkeypatch.setattr(MetricsMixin, "validation_step", lambda self, i, t, **kw: {"psnr": 1.0})
    k = _kspace()
    metrics = s.validation_step(k, k)
    assert "val_equispaced_psnr" in metrics and "val_ib_psnr" not in metrics


def test_declared_losses_train_through_the_loss_computer(settings, monkeypatch) -> None:
    s = _strategy(settings)
    calls = {}

    class _Computer:
        def compute(self, *, pred, target, epoch, iteration, losses_dict):
            calls["losses_dict"] = losses_dict
            return SimpleNamespace(total=(pred - target).abs().mean(), components={"l1": 0})

    s.loss_computer = _Computer()
    s.env = SimpleNamespace(losses={"l1": object()})
    identity = torch.nn.Identity()
    monkeypatch.setattr(
        IBActiveAcquisitionStrategy, "generator_model", property(lambda self: identity)
    )
    x = torch.randn(2, 1, 8, 8)
    out = s._compute_losses_impl(x, x, epoch=0)
    assert "g_total_loss" in out and calls["losses_dict"] == s.env.losses


def test_nonlinear_mode_and_a_missing_axis_raise(settings) -> None:
    ib = settings.training.ib_active_acquisition.model_copy(update={"mode": "nonlinear"})
    training = settings.training.model_copy(update={"ib_active_acquisition": ib})
    bad = settings.model_copy(update={"training": training})
    s = object.__new__(IBActiveAcquisitionStrategy)
    with pytest.raises(ValueError, match="nonlinear"):
        s._configure_ib(bad)
    data = settings.data.model_copy(update={"acquisition": None})
    with pytest.raises(ValueError, match="phase_encode_axis"):
        s._configure_ib(settings.model_copy(update={"data": data}))
