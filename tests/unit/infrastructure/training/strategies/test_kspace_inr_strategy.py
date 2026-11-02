"""``kspace_inr``: per-slice SIREN k-space fit with an image-domain term (ULF-PR-4)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import yaml
from pydantic import ValidationError

from spectramr.config.schemas.training.kspace_inr import KSpaceINRTrainingConfigSchema
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.kspace_inr_strategy import KSpaceINRStrategy
from spectramr.infrastructure.validation.witness.checks.loss_ownership_checks import (
    declared_image_loss_names,
    loss_ownership_defect,
)

_ARM = (
    Path(__file__).resolve().parents[5]
    / "experiments/inprogress/promoted/exp_promoted_kspace_inr.yaml"
)
_COILS, _H, _W = 2, 8, 6  # non-square: a transposed grid would not fit


def test_it_declares_its_own_loss_ownership() -> None:
    """The image term folds the builder's image losses; the DC term is not a canonical image loss."""
    assert KSpaceINRStrategy.__dict__["folds_image_losses"] is True
    assert KSpaceINRStrategy.__dict__["inline_losses"] == frozenset()
    assert KSpaceINRStrategy.__dict__["model_input_is_coordinates"] is True


def test_the_arm_declares_no_image_loss_the_strategy_drops() -> None:
    raw = yaml.safe_load(_ARM.read_text(encoding="utf-8"))
    entries = (raw.get("losses") or {}).get("image_losses") or []
    names = declared_image_loss_names(SimpleNamespace(losses=SimpleNamespace(image_losses=entries)))
    assert names == ["l1"]
    assert loss_ownership_defect(names, KSpaceINRStrategy) == (None, True)


@pytest.mark.parametrize("bad", [{"inner_steps": 0}, {"inner_lr": 0.0}, {"lambda_dcc": 1.0}])
def test_the_schema_refuses_an_inert_fit_or_an_unknown_knob(bad: dict) -> None:
    with pytest.raises(ValidationError):
        KSpaceINRTrainingConfigSchema(**bad)


def test_the_production_resolve_path_reaches_this_strategy() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    config = SimpleNamespace(
        training=SimpleNamespace(strategy_class=None, training_mode="kspace_inr")
    )
    assert TrainingStrategyFactory().get_strategy_class(config) is KSpaceINRStrategy


class _L1Computer:
    """The loss builder's l1 on the strategy's image, as ``loss_computer.compute`` returns it."""

    def __init__(self) -> None:
        self.preds: list[torch.Tensor] = []

    def compute(self, pred, target, **_: object) -> SimpleNamespace:
        self.preds.append(pred)
        loss = F.l1_loss(pred, target)
        return SimpleNamespace(total=loss, components={"l1": loss.detach()})


def _strategy(**knobs: object) -> KSpaceINRStrategy:
    import spectramr.models.generators.siren_generator  # noqa: F401  (registers `siren`)
    from spectramr.models.registry import get_model_class

    torch.manual_seed(0)
    gen = get_model_class("siren")(
        in_channels=2, out_channels=2 * _COILS, hidden_features=16, hidden_layers=1
    )
    cfg = KSpaceINRTrainingConfigSchema(**knobs)
    s = object.__new__(KSpaceINRStrategy)
    s.env = SimpleNamespace(
        generator=gen,
        losses={},
        config=SimpleNamespace(model=SimpleNamespace(input_type="image")),
    )
    s.device = torch.device("cpu")
    s.loss_computer = _L1Computer()
    s.inner_steps, s.inner_lr = cfg.inner_steps, cfg.inner_lr
    s.lambda_dc, s.first_order = cfg.lambda_dc, cfg.first_order
    s._grids = {}
    return s


def _slice(
    seed: int, gain: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(measured interleaved [1, 2C, H, W], mask [1, 1, H, W], input, RSS target)``."""
    g = torch.Generator().manual_seed(seed)
    image = gain * torch.randn(1, _COILS, _H, _W, generator=g, dtype=torch.complex64)
    k = fft2c(image)
    mask = torch.zeros(1, 1, _H, _W)
    mask[..., ::2] = 1.0
    measured = torch.view_as_real(k * mask).permute(0, 1, 4, 2, 3).reshape(1, 2 * _COILS, _H, _W)
    target = image.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
    return measured, mask, target.clone(), target


def _batch(*seeds: int, gain: float = 1.0) -> tuple[torch.Tensor, torch.Tensor, dict]:
    parts = [_slice(s, gain) for s in seeds]
    measured, mask, lr, hr = (torch.cat(t) for t in zip(*parts, strict=True))
    return lr, hr, {"batch": {"measured_kspace": measured, "mask": mask}}


def test_the_fit_fires_and_trains_the_initialisation() -> None:
    """Reachability: the SIREN is fitted, imaged by ifft2c/RSS, and the outer loss reaches it."""
    s = _strategy(inner_steps=2, inner_lr=1e-2)
    lr, hr, kw = _batch(0, 1)
    out = s._compute_losses_impl(input_batch=lr, target_batch=hr, epoch=0, **kw)
    assert set(out) >= {"loss_total", "loss_image", "loss_kspace_dc", "l1"}
    assert s.loss_computer.preds[0].shape == hr.shape
    out["loss_total"].backward()
    grads = [p.grad for p in s.env.generator.parameters()]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_each_slice_gets_its_own_fit() -> None:
    """Two measurements give two predictions: the field is not one population mean.

    Driven through ``validation_batch_context``, the context production validation runs on.
    """
    s = _strategy(inner_steps=3, inner_lr=5e-2)
    lr, hr, kw = _batch(0, 1)
    context = s.validation_batch_context(lr, hr, kw["batch"])
    assert context is not None
    pred = s._validation_forward(lr, context)
    assert pred.shape == hr.shape and not pred.requires_grad
    assert not torch.allclose(pred[0], pred[1])


def _residuals(gain: float, steps: tuple[int, ...]) -> list[float]:
    lr, hr, kw = _batch(3, gain=gain)
    out = []
    for n in steps:
        s = _strategy(inner_steps=n, inner_lr=1e-3)
        y, mask = s._measurement(s._prepare_batch_context(lr, hr, **kw))
        with torch.no_grad(), torch.enable_grad():
            pred = s._fit(y, mask, second_order=False)
        out.append((s._dc(pred, y, mask) / s._dc(torch.zeros_like(y), y, mask)).item())
    return out


def test_the_inner_loop_reduces_the_sampled_kspace_residual() -> None:
    first, last = _residuals(1.0, (1, 10))
    assert last < first


def test_the_fit_does_not_depend_on_the_kspace_scale() -> None:
    """Raw scanner k-space is not unit scale; the fit runs on y / RMS(y on the mask)."""
    assert _residuals(1e-4, (1, 10)) == pytest.approx(_residuals(1.0, (1, 10)), rel=1e-3)
    lr, hr, kw = _batch(0)
    small_lr, small_hr, small_kw = _batch(0, gain=1e-4)
    s = _strategy(inner_steps=2, inner_lr=1e-3)
    big = s._validation_forward(lr, s._prepare_batch_context(lr, hr, **kw))
    small = s._validation_forward(
        small_lr, s._prepare_batch_context(small_lr, small_hr, **small_kw)
    )
    torch.testing.assert_close(small, 1e-4 * big, rtol=1e-3, atol=1e-9)


def test_a_missing_measurement_raises() -> None:
    s = _strategy()
    lr, hr, _ = _batch(0)
    with pytest.raises(ValueError, match="measured_kspace"):
        s._compute_losses_impl(input_batch=lr, target_batch=hr, epoch=0, batch={})


def test_a_generator_that_does_not_emit_every_coil_raises() -> None:
    s = _strategy()
    s.env.generator.out_channels = 2
    lr, hr, kw = _batch(0)
    with pytest.raises(ValueError, match="out_channels: 4"):
        s._compute_losses_impl(input_batch=lr, target_batch=hr, epoch=0, **kw)


def test_a_missing_training_block_raises() -> None:
    s = object.__new__(KSpaceINRStrategy)
    s.config = SimpleNamespace(training=SimpleNamespace(kspace_inr=None))
    with pytest.raises(ValueError, match=r"training\.kspace_inr"):
        _init_tail(s)


def _init_tail(s: KSpaceINRStrategy) -> None:
    """Run ``__init__`` with the base constructor stubbed: only the block read is under test."""
    from unittest import mock

    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    with mock.patch.object(ReconstructionTrainingStrategy, "__init__", lambda self, *a, **k: None):
        KSpaceINRStrategy.__init__(s)
