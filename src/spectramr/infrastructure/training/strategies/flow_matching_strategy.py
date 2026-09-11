"""Explicit flow matching with a probability-flow ODE sampler (PR-9).

Reference: Lipman et al. 2023, *Flow Matching for Generative Modeling*;
Liu et al. 2023, *Rectified Flow*.

This is a genuine velocity-field flow-matching loop — distinct from the
``flow_matching`` registry key, which merely aliases the generic
:class:`DiffusionTrainingStrategy`. The new ``flow_matching_pfode`` key
points here.

Conditional probability path (straight line / rectified flow)::

    x_t = (1 - t) * x_0 + t * x_1,   x_0 ~ N(0, I)  (or a source if present),
                                     x_1 = data target.

Because the path is linear in ``t``, the conditional velocity target is
**constant** in ``t``::

    u_t = d/dt x_t = x_1 - x_0.

We train ``v_theta(x_t, t)`` to regress ``u_t`` under an MSE objective. This
is exactly the stochastic-interpolant objective **without** the
``sigma(t) * z`` stochastic term -- a pure probability-flow ODE, which is
what distinguishes flow matching from stochastic interpolants.

Inference (:meth:`sample`) integrates the learned probability-flow ODE::

    dx/dt = v_theta(x, t),   from t=0 (noise) to t=1 (data),

with explicit Euler steps or Heun (2nd-order, trapezoidal) steps.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch.nn import functional as func

from spectramr.data.batch_types import read_batch_field

from .diffusion import DiffusionTrainingStrategy

VelocityFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class FlowMatchingStrategy(DiffusionTrainingStrategy):
    """Lipman/rectified-flow velocity matching with a PF-ODE sampler.

    Time-conditioning is dispatched via the inherited
    :meth:`DiffusionTrainingStrategy._generator_accepts_time` (signature
    introspection, no ``except TypeError`` swallow).
    """

    #: Loss ownership (issue #1918). mse_loss(pred, u_t) regresses the VELOCITY FIELD, not the target image;
    #: the hook does not route to the builder's losses.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Orchestrator-kwargs canonical signature (F6). Read the generator
        # from self.env.generator — the StrategyContext has no such field.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        # ``read_batch_field``, not ``isinstance(batch, dict)``. The batch the
        # loop delivers is a ``TrainingBatch`` dataclass, so the isinstance leg
        # is False and the ``else`` handed the WHOLE BATCH OBJECT on as if it
        # were the target tensor -- which surfaced as
        # ``'TrainingBatch' object has no attribute 'shape'`` / ``randn_like():
        # must be Tensor, not TrainingBatch``. ``TrainingBatch`` implements the
        # mapping protocol precisely so this accessor is shape-agnostic, and its
        # docstring names this pairing 'the single most repeated defect at this
        # seam'.
        target = read_batch_field(batch, "target")
        if gen is None or target is None:
            return {
                "loss_total": torch.tensor(0.0, device=self.device),
                "loss_fm_velocity": torch.tensor(0.0, device=self.device),
            }

        x1 = target
        x0 = read_batch_field(batch, "source")
        if x0 is None:
            x0 = torch.randn_like(x1)

        b = x1.shape[0]
        # Per-sample time, broadcast across spatial/channel dims.
        t = torch.rand(b, device=x1.device).view(b, *([1] * (x1.dim() - 1)))

        # Straight-line conditional path; constant conditional velocity.
        x_t = (1 - t) * x0 + t * x1
        u_t = x1 - x0

        # Expose the conditional velocity so an oracle generator (tests /
        # debugging) can return it exactly. Does not affect real models.
        self._true_velocity = u_t

        # The model is fed the INTERPOLANT, which is built here and never leaves
        # this method -- so ``first_steps/input_prepared``, captured before the
        # forward pass, is not it. Declare the real one; the wrapper emits it
        # under ``diffusion_step`` (non-negotiable 14).
        self._declare_model_input(
            {"model_input": x_t, "source_x0": x0, "target": x1},
            # Explicit and empty: none of these is k-space by NAME. When the arm
            # declares k-space data, ``save_debug_snapshot`` unions the canonical
            # keys in from the config SSOT -- which is why the tensors carry
            # ``model_input``/``target`` rather than descriptive names.
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "note": (
                    "Flow matching feeds the straight-line interpolant "
                    "x_t = (1 - t) * x0 + t * x1, formed inside this step. "
                    "'first_steps/input_prepared' is PRE-interpolation. The "
                    "per-sample time t is a second model argument, not an "
                    "image, so it is described here rather than captured (a "
                    "scalar read would sync the device -- non-negotiable 9)."
                ),
            },
        )

        # Dispatch via signature introspection (never except-TypeError to
        # branch, so an in-forward TypeError propagates — pitfall #9).
        if self._generator_accepts_time(gen):
            pred = gen(x_t, t.flatten())
        else:
            pred = gen(x_t)

        loss = func.mse_loss(pred, u_t)
        return {"loss_total": loss, "loss_fm_velocity": loss.detach()}

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: VelocityFn,
        x0: torch.Tensor,
        n_steps: int = 50,
        method: str = "euler",
    ) -> torch.Tensor:
        """Integrate the probability-flow ODE ``dx/dt = v_theta(x, t)``.

        Integrates from ``t=0`` (noise prior ``x0``) to ``t=1`` (data) in
        ``n_steps`` uniform steps.

        Args:
            velocity_fn: Callable ``(x, t) -> v`` returning the velocity at
                state ``x`` and per-sample time ``t`` (a 1-D tensor of length
                ``batch``).
            x0: Initial state (noise sample), shape ``[B, ...]``.
            n_steps: Number of integration steps.
            method: ``"euler"`` (1st order) or ``"heun"`` (2nd order,
                explicit trapezoidal). No silent fallback — an unknown method
                raises (pitfall #9).

        Returns:
            The integrated sample at ``t=1``, same shape as ``x0``.
        """
        if method not in ("euler", "heun"):
            raise ValueError(f"Unknown integration method {method!r}; expected 'euler' or 'heun'.")

        x = x0.clone()
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=x.device)
        for k in range(n_steps):
            t_now = ts[k]
            t_next = ts[k + 1]
            dt = t_next - t_now
            t_b = t_now.expand(x.shape[0])
            v = velocity_fn(x, t_b)
            if method == "euler":
                x = x + v * dt
            else:  # heun: predictor (Euler) + corrector (trapezoid)
                x_pred = x + v * dt
                t_b_next = t_next.expand(x.shape[0])
                v_next = velocity_fn(x_pred, t_b_next)
                x = x + 0.5 * (v + v_next) * dt
        return x
