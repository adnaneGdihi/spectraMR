"""Stochastic Interpolants (PR-19 / Part-I A).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-19.

Reference: Albergo & Vanden-Eijnden 2023, *Stochastic Interpolants —
A Unifying Framework for Flow-Based Generative Models*.

Interpolant ``I_t = (1−t) x_0 + t x_1 + σ(t) z``,  ``z ~ N(0, I)``
with ``σ(t) = sqrt(t(1−t))`` (peaks at ``t=½``).

The target velocity for the deterministic + stochastic parts is

    ``b(t, x) = E[I_t' | I_t = x]
              = (x_1 − x_0) + σ'(t) z``

so we regress ``b_θ(I_t, t)`` against ``(x_1 − x_0) + σ'(t) z`` per
sample.

Public method :meth:`sample` integrates the corresponding probability
flow ODE (or SDE if ``stochastic=True``) for inference.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.data.batch_types import read_batch_field

from .diffusion import DiffusionTrainingStrategy


class StochasticInterpolantsStrategy(DiffusionTrainingStrategy):
    """Albergo–Vanden-Eijnden stochastic-interpolant velocity matching."""

    #: Loss ownership (issue #1918). mse_loss(pred, true_velocity) regresses the interpolant VELOCITY; no
    #: declared image loss is computed and none is folded.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    @staticmethod
    def _sigma(t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((t * (1 - t)).clamp_min(0.0))

    @staticmethod
    def _sigma_dot(t: torch.Tensor) -> torch.Tensor:
        # d/dt sqrt(t(1-t)) = (1 − 2t) / (2 sqrt(t(1-t)))
        eps = 1e-6
        return (1 - 2 * t) / (2 * StochasticInterpolantsStrategy._sigma(t).clamp_min(eps))

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # F8b inheritance: read generator from self.env.generator
        # (the StrategyContext dataclass has no ``generator`` field).
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        # ``read_batch_field``, not ``isinstance(batch, dict)``: the loop delivers a
        # ``TrainingBatch`` dataclass, so the isinstance leg is False and the
        # ``else`` handed the WHOLE BATCH OBJECT on as if it were a tensor.
        target = read_batch_field(batch, "target")
        if gen is None or target is None:
            return {"loss_total": torch.tensor(0.0, device=self.device)}

        x1 = target
        x0 = read_batch_field(batch, "source")
        if x0 is None:
            x0 = torch.randn_like(x1)

        b = x1.shape[0]
        t = torch.rand(b, device=x1.device).view(b, *([1] * (x1.dim() - 1)))
        z = torch.randn_like(x1)
        sigma_t = self._sigma(t)
        sigma_dot = self._sigma_dot(t)

        I_t = (1 - t) * x0 + t * x1 + sigma_t * z
        true_velocity = (x1 - x0) + sigma_dot * z

        # The model is fed the stochastic interpolant, formed here and never
        # leaving this method -- so ``first_steps/input_prepared``, captured
        # before the forward pass, is not it. Declare the real one; the wrapper
        # emits it under ``diffusion_step`` (non-negotiable 14).
        self._declare_model_input(
            {"model_input": I_t, "source_x0": x0, "target": x1},
            # Explicit and empty: none of these is k-space by NAME. The
            # canonical keys are unioned in from the config SSOT when the arm
            # declares k-space data, which is safe here because the interpolant
            # is a linear combination of tensors in the arm's own domain.
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "note": (
                    "Stochastic interpolants feed "
                    "I_t = (1 - t) * x0 + t * x1 + sigma(t) * z, formed inside "
                    "this step -- the sigma(t) * z term is what distinguishes it "
                    "from the deterministic flow-matching path. "
                    "'first_steps/input_prepared' is PRE-interpolation."
                ),
            },
        )

        # Detect time-conditioning via signature introspection (NN#3 /
        # pitfall #9): a ``try: gen(I_t, t) except TypeError: gen(I_t)``
        # fallback would silently swallow a genuine TypeError raised *inside*
        # gen.forward (shape/dtype/kwarg bug) and retry with the wrong
        # signature, masking the real failure. ``_generator_accepts_time``
        # dispatches the correct call unconditionally.
        if self._generator_accepts_time(gen):
            pred = gen(I_t, t.flatten())
        else:
            pred = gen(I_t)

        loss = F.mse_loss(pred, true_velocity)
        return {"loss_total": loss, "loss_si_velocity": loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        n_steps: int = 50,
        stochastic: bool = False,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """Integrate the interpolant ODE / SDE from t=0 to t=1.

        ODE:  dx/dt = b(t, x).
        SDE:  dx = b(t, x) dt + ε dW   (Stratonovich-style with constant ε).
        """
        x = x0.clone()
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
        for k in range(n_steps):
            t_now = ts[k]
            t_next = ts[k + 1]
            dt = t_next - t_now
            t_b = t_now.expand(x.shape[0])
            v = velocity_fn(x, t_b)
            x = x + v * dt
            if stochastic:
                x = x + epsilon * torch.sqrt(dt.clamp_min(0.0)) * torch.randn_like(x)
        return x
