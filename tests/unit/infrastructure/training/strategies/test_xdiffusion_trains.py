"""Regression: ``XDiffusionTrainingStrategy._compute_losses_impl`` must train.

2026-05-31 audit found the X-Diffusion step was a no-op: it called
``generated_output = self.env.generator(input_batch)`` then DISCARDED the
result, and computed the loss as ``loss_fn(target + noise, target)`` — a term
with ZERO gradient w.r.t. any generator parameter. The "forward diffusion" was
plain ``target + noise`` with no schedule and no timestep.

The fix samples a timestep ``t`` and noise ``ε``, forms the cosine-schedule
noised target ``x_t = √ᾱ_t·target + √(1-ᾱ_t)·ε``, passes ``x_t`` through the
generator (with ``t`` if the forward accepts it), and scores the prediction
against the clean target. The KEY property asserted here: the returned total
loss requires grad AND its gradient w.r.t. a generator parameter is nonzero.
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.diffusion import (
    XDiffusionTrainingStrategy,
)


class _TinyGenerator(nn.Module):
    """1x1 conv x_0-predictor; forward(x) only (no timestep kwarg)."""

    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _TimestepGenerator(nn.Module):
    """x_0-predictor whose forward accepts a ``timesteps`` kwarg."""

    def __init__(self, channels: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.seen_timesteps: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor | None = None) -> torch.Tensor:
        self.seen_timesteps = timesteps
        return self.conv(x)


def _make_strategy(generator: nn.Module) -> XDiffusionTrainingStrategy:
    """Build a bare strategy with only the attrs ``_compute_losses_impl`` reads."""
    s = object.__new__(XDiffusionTrainingStrategy)
    s.env = types.SimpleNamespace(generator=generator, losses={})
    s.config = types.SimpleNamespace(
        training=types.SimpleNamespace(
            diffusion=types.SimpleNamespace(timesteps=50)
        ),
        model=types.SimpleNamespace(model_type="x_diffusion"),
    )
    s.logging_service = types.SimpleNamespace(log_info=lambda *a, **k: None)
    s._loss_dict_reuse = {}
    return s


class TestXDiffusionTrains:
    def test_total_loss_flows_gradient_to_generator(self):
        gen = _TinyGenerator(channels=2)
        s = _make_strategy(gen)

        x = torch.randn(2, 2, 8, 8)
        y = torch.randn(2, 2, 8, 8)
        out = s._compute_losses_impl(input_batch=x, target_batch=y, epoch=1)

        total = out["g_total_loss"]
        assert total.requires_grad, "total loss must require grad (it was detached)"

        param = gen.conv.weight
        grad = torch.autograd.grad(total, param, retain_graph=False)[0]
        assert grad is not None
        # The whole point of the fix: the loss DEPENDS on the generator output,
        # so the gradient w.r.t. a generator weight is nonzero.
        assert torch.linalg.vector_norm(grad).item() > 0.0, (
            "generator gradient is zero — loss does not depend on model output"
        )

    def test_timestep_is_passed_when_forward_accepts_it(self):
        gen = _TimestepGenerator(channels=2)
        s = _make_strategy(gen)

        x = torch.randn(1, 2, 8, 8)
        y = torch.randn(1, 2, 8, 8)
        out = s._compute_losses_impl(input_batch=x, target_batch=y, epoch=0)

        assert gen.seen_timesteps is not None, "timestep tensor never reached forward"
        assert gen.seen_timesteps.shape == (1,)
        # Diffusion timesteps must be in [1, T) — never the degenerate t=0.
        assert int(gen.seen_timesteps.min()) >= 1
        assert out["g_total_loss"].requires_grad

    def test_resolves_dict_batch_via_legacy_kwarg(self):
        gen = _TinyGenerator(channels=1)
        s = _make_strategy(gen)

        x = torch.randn(2, 1, 8, 8)
        y = torch.randn(2, 1, 8, 8)
        batch = {"input": x, "target": y}
        # Legacy calling convention: batch dict arrives via kwargs only.
        out = s._compute_losses_impl(epoch=1, batch=batch)

        total = out["g_total_loss"]
        grad = torch.autograd.grad(total, gen.conv.weight)[0]
        assert torch.linalg.vector_norm(grad).item() > 0.0


class TestModelInputContract:
    """Non-negotiable 14 — the *opposite* polarity to the diffusion family.

    ``XDiffusionTrainingStrategy`` extends ``BaseTrainingStrategy``, **not**
    ``DiffusionTrainingStrategy``, so it inherited the default
    ``snapshot_prepared_is_model_input = True`` — asserting that
    ``first_steps/input_prepared`` IS the model input. Its own docstring says
    otherwise: the UNet is fed ``x_t``, the noised TARGET contrast, while
    ``input_prepared`` holds the SOURCE modality. The source reaches the model
    only as a conditioning embedding, and on an unconditioned arm not at all.

    So the base captured the source contrast and labelled it "the model input".
    A reader comparing that snapshot against the prediction would be comparing
    two different contrasts (facade, pitfall #16).
    """

    def test_declares_the_noised_target_the_unet_actually_receives(self, declared):
        gen = _TimestepGenerator(channels=2)
        s = _make_strategy(gen)

        x = torch.randn(2, 2, 8, 8)  # source modality
        y = torch.randn(2, 2, 8, 8)  # target contrast
        s._compute_losses_impl(input_batch=x, target_batch=y, epoch=1)

        assert declared() is not None
        tensors, extra, in_kspace_keys = declared()

        # The declared input is the noised TARGET, not the source modality --
        # which is exactly what the inherited `True` got wrong.
        assert not torch.equal(tensors["model_input"], x)
        assert not torch.equal(tensors["model_input"], y)
        assert torch.equal(tensors["input"], x)
        assert torch.equal(tensors["target"], y)
        assert extra["model_input_key"] == "model_input"
        assert in_kspace_keys == set(), "must be explicit, not None"

    def test_declared_input_is_the_tensor_the_generator_saw(self, declared):
        """Pin the declaration to the forward call, not merely to `x_t`.

        The generator is invoked several lines below the declaration, through a
        kwarg-resolution path that can rebind its argument. Recording what the
        module received is the only assertion that survives that.
        """
        seen: dict = {}

        class _RecordingGen(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.conv = nn.Conv2d(2, 2, kernel_size=1)

            def forward(self, x: torch.Tensor, timesteps=None) -> torch.Tensor:
                seen["x"] = x
                return self.conv(x)

        gen = _RecordingGen()
        s = _make_strategy(gen)
        s._compute_losses_impl(
            input_batch=torch.randn(1, 2, 8, 8),
            target_batch=torch.randn(1, 2, 8, 8),
            epoch=0,
        )

        tensors, _extra, _keys = declared()
        assert torch.equal(tensors["model_input"], seen["x"])
