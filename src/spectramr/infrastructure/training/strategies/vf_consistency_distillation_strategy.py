"""VF-anchored consistency distillation (PR-5 / idea N-B).

Distil a (probability-flow) diffusion teacher into a 1--4-step *consistency*
student :math:`f_\\theta`, enforcing the consistency self-condition with an EMA
teacher PLUS a **noise-gated virtual-fiducial (VF) anchor** that gives a
target-leak-free fidelity certificate.

Consistency model
------------------
A consistency student :math:`f_\\theta` maps any point on the probability-flow
trajectory back to its origin. We enforce

.. math::

    f_\\theta(x_{t_{n+1}}, t_{n+1}) \\approx f_{\\theta^-}(\\hat{x}_{t_n}, t_n)

where :math:`\\theta^-` is an exponential-moving-average (EMA) of
:math:`\\theta` (the *teacher*), and :math:`\\hat{x}_{t_n}` is one
deterministic solver step (Euler on the PF-ODE, here a single denoising step)
from :math:`x_{t_{n+1}}` toward :math:`t_n`. This is the
Song-et-al-2023 consistency-distillation objective.

The EMA teacher is maintained **inside the strategy** as a frozen
``deepcopy`` of the student, updated each step by
``theta^- <- decay * theta^- + (1 - decay) * theta`` (see
:meth:`_update_ema`).

Virtual-fiducial anchor
-----------------------
We compute a **data-independent** marker :math:`m^*` from
:class:`spectramr.infrastructure.physics.virtual_fiducial.VirtualFiducial`
(a fixed Gaussian-grid probe — no dependence on the batch target, hence no
target leak). The marker-restriction operator :math:`V(\\cdot)` keeps only the
marker support, implemented with
:class:`spectramr.infrastructure.physics.vf_operators.MarkerPriorProjection`'s
diagonal mask. We penalise

.. math::

    \\beta(t)\\,\\bigl\\lVert V\\bigl(f_\\theta(x_{t_n})\\bigr) - m^* \\bigr\\rVert_1

with a noise-gated weight

.. math::

    \\beta(t) = \\beta_0 \\, \\exp\\!\\bigl(-(\\sigma_t / \\sigma_{\\text{data}})^2\\bigr)

so the marker constraint is only imposed at low noise where the marker is
identifiable (large :math:`\\sigma_t` -> :math:`\\beta(t) \\to 0`).

Trivial limit
-------------
``beta0 = 0`` zeroes :math:`\\beta(t)` for all :math:`t`, so ``loss_vf_anchor``
is identically ``0`` and ``loss_total`` reduces to ``loss_consistency`` --
i.e. plain consistency distillation with no marker certificate.

No new YAML schema keys are introduced; all knobs are read from constructor
defaults (and may be overridden by attaching attributes before training).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, ClassVar

import torch
from torch.nn import functional as func

from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.vf_operators import MarkerPriorProjection
from spectramr.infrastructure.physics.virtual_fiducial import VirtualFiducial

from .diffusion import DiffusionTrainingStrategy


class VFConsistencyDistillationStrategy(DiffusionTrainingStrategy):
    """Few-step consistency distillation with a noise-gated VF anchor.

    Constructor-default knobs (set as instance attributes; no YAML keys):

    Attributes:
        beta0: Peak VF-anchor weight :math:`\\beta_0`. ``0`` => plain
            consistency distillation (anchor dropped).
        ema_decay: EMA decay for the teacher :math:`\\theta^-`.
        sigma_data: Data-noise scale :math:`\\sigma_{\\text{data}}` in the gate.
        num_consistency_steps: Number of trajectory knots ``N`` used to draw
            the adjacent pair :math:`(t_n, t_{n+1})`.
        vf_grid_spacing: Gaussian-grid spacing for the virtual fiducial.
        vf_im_size: ``(H, W)`` for the fiducial; inferred from the batch if
            left unset.
    """

    #: Loss ownership (issue #1918). mse_loss(student_pred, teacher_pred) is a DISTILLATION term against the
    #: teacher, not the target; the hook folds nothing from the builder.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    #: Defaults applied when the heavyweight base init has run (training path).
    _DEFAULTS: ClassVar[dict[str, Any]] = {
        "beta0": 0.5,
        "ema_decay": 0.999,
        "sigma_data": 0.5,
        "num_consistency_steps": 4,
        "vf_grid_spacing": 16,
        "vf_im_size": None,
    }

    # ------------------------------------------------------------------ knobs
    def _knob(self, name: str) -> Any:
        if not hasattr(self, name):
            # Prefer the typed training.vf_consistency_distillation block
            # (validated); fall back to the documented default (pitfall #15 —
            # these were previously unconditionally the _DEFAULTS).
            # Sequential single getattrs (tolerate a bare instance with no
            # ``config`` and the optional paradigm sub-block) — not a chain.
            _cfg = getattr(self, "config", None)
            cfg = getattr(_cfg, "training", None)
            cfg = getattr(cfg, "vf_consistency_distillation", None)
            if cfg is not None and hasattr(cfg, name):
                setattr(self, name, getattr(cfg, name))
            else:
                setattr(self, name, self._DEFAULTS[name])
        return getattr(self, name)

    # -------------------------------------------------------------- EMA teacher
    def _ensure_ema_teacher(self, student: torch.nn.Module) -> torch.nn.Module:
        """Create the frozen EMA teacher as a deepcopy of the student."""
        teacher = getattr(self, "_ema_teacher", None)
        if teacher is None:
            teacher = copy.deepcopy(student)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
            self._ema_teacher = teacher
        return teacher

    @torch.no_grad()
    def _update_ema(self, student: torch.nn.Module) -> None:
        """``theta^- <- decay * theta^- + (1 - decay) * theta`` (params+buffers)."""
        teacher = self._ema_teacher
        decay = float(self._knob("ema_decay"))
        for tp, sp in zip(teacher.parameters(), student.parameters(), strict=False):
            tp.mul_(decay).add_(sp.detach(), alpha=1.0 - decay)
        # Buffers (e.g. running stats) tracked verbatim so the teacher stays valid.
        for tb, sb in zip(teacher.buffers(), student.buffers(), strict=False):
            tb.copy_(sb)

    # ------------------------------------------------------------- VF marker
    def _ensure_vf(self, ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (and cache) the data-independent marker ``m*`` and its mask.

        Returns ``(marker, mask)`` both broadcastable to ``ref`` ([B,C,H,W]).
        The marker is a fixed Gaussian-grid probe -- it never sees the target,
        so anchoring to it cannot leak the reconstruction target.
        """
        if getattr(self, "_vf_marker", None) is not None:
            return self._vf_marker, self._vf_mask

        h, w = ref.shape[-2], ref.shape[-1]
        im_size = self._knob("vf_im_size") or (h, w)
        spacing = int(self._knob("vf_grid_spacing"))
        spacing = max(1, min(spacing, min(im_size)))

        fiducial = VirtualFiducial(
            im_size=tuple(im_size), grid_spacing=spacing, learnable=False
        ).to(ref.device)
        # Real-valued magnitude marker, [1, 1, H, W].
        marker = fiducial().abs().to(ref.dtype)
        # Marker support mask -> the restriction operator V(x) = mask * x,
        # realised via MarkerPriorProjection's diagonal mask buffer.
        mask = (marker > 0.05).to(ref.dtype)
        projector = MarkerPriorProjection(marker_mask=mask, marker_prior=marker).to(ref.device)
        # V(x) keeps only the marker region; equivalently mask out the rest.
        self._vf_marker = projector.prior  # ideal marker on the support
        self._vf_mask = projector.mask
        return self._vf_marker, self._vf_mask

    @staticmethod
    def _restrict(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Marker-restriction operator :math:`V(x) = m \\odot x`."""
        return mask * x

    # --------------------------------------------------------------- gate
    def _beta_gate(self, sigma_t: torch.Tensor) -> torch.Tensor:
        """Noise-gated anchor weight :math:`\\beta_0 \\exp(-(\\sigma_t/\\sigma_d)^2)`."""
        beta0 = float(self._knob("beta0"))
        sigma_data = float(self._knob("sigma_data"))
        return beta0 * torch.exp(-((sigma_t / max(sigma_data, 1e-8)) ** 2))

    # ------------------------------------------------------------- main loss
    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        # ``read_batch_field``, not ``isinstance(batch, dict)``: the loop delivers a
        # ``TrainingBatch`` dataclass, so the isinstance leg is False and the
        # ``else`` handed the WHOLE BATCH OBJECT on as if it were a tensor.
        target = read_batch_field(batch, "target")
        if gen is None or target is None:
            return {
                "loss_total": torch.tensor(0.0, device=self.device),
                "loss_consistency": torch.tensor(0.0, device=self.device),
                "loss_vf_anchor": torch.tensor(0.0, device=self.device),
            }

        x0 = target
        b = x0.shape[0]
        dim_pad = [1] * (x0.dim() - 1)

        # Adjacent trajectory knots t_n < t_{n+1} on the PF trajectory.
        n_steps = int(self._knob("num_consistency_steps"))
        n_idx = torch.randint(0, n_steps, (b,), device=x0.device)
        t_next = ((n_idx + 1).float() / n_steps).view(b, *dim_pad)
        t_now = (n_idx.float() / n_steps).view(b, *dim_pad)

        # VP-style noise scales: sigma_t monotone in t (so the gate is meaningful).
        sigma_next = t_next
        sigma_now = t_now

        noise = torch.randn_like(x0)
        x_next = x0 + sigma_next * noise  # x_{t_{n+1}}

        # One deterministic solver step toward t_n (Euler on PF-ODE: rescale noise).
        x_hat_now = x0 + sigma_now * noise  # x̂_{t_n}

        # Both forward inputs are formed here, so ``first_steps/input_prepared``
        # -- captured before the forward pass -- is the clean x0, not either of
        # them. Declare the real ones; the wrapper emits under ``diffusion_step``
        # (non-negotiable 14).
        #
        # ``model_input`` is the STUDENT's: it is the network being trained, and
        # the contract is about the tensor that receives gradient. The teacher is
        # a frozen EMA copy manufacturing targets under ``no_grad``, so its input
        # rides alongside under its own name -- distillation is a two-input
        # mechanism and an artifact showing one of them cannot show the noise
        # gap (sigma_next - sigma_now) the objective is built on.
        self._declare_model_input(
            {
                "model_input": x_next,
                "teacher_input": x_hat_now,
                "target": x0,
            },
            # Explicit and empty: the canonical keys are unioned in from the
            # config SSOT when the arm declares k-space, which is right here --
            # both inputs are x0 plus scaled noise, so they live in x0's domain.
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "note": (
                    "Consistency distillation feeds the student x_{t_{n+1}} = "
                    "x0 + sigma_next * noise and the frozen EMA teacher "
                    "x_hat_{t_n} = x0 + sigma_now * noise -- the SAME noise "
                    "draw at two adjacent trajectory knots. "
                    "'first_steps/input_prepared' is PRE-noising."
                ),
            },
        )

        def _fwd(
            module: torch.nn.Module,
            x: torch.Tensor,
            t: torch.Tensor,
            *,
            takes_time: bool,
        ) -> torch.Tensor:
            """Dispatch on introspection (SAQ-001), never on ``except TypeError``.

            A ``TypeError`` raised *inside* the forward -- a shape bug, a bad
            kwarg deeper down -- is a real failure. Retrying it as ``module(x)``
            swallowed the traceback and trained a time-unconditioned student
            against a time-conditioned teacher, which converges to something and
            so never announces itself. ``takes_time`` is resolved per module
            because student and teacher need not share a signature.
            """
            return module(x, t.flatten()) if takes_time else module(x)

        # Student maps x_{t_{n+1}} -> origin.
        student_pred = _fwd(gen, x_next, t_next, takes_time=self._generator_accepts_time(gen))

        # EMA teacher maps the one-step point x̂_{t_n} -> origin (frozen target).
        # Resolved AFTER the student forward: `_ensure_ema_teacher` deep-copies
        # `gen` on first call, and a train-mode forward updates BatchNorm running
        # stats -- hoisting it above would hand step 0 a teacher with different
        # buffers.
        teacher = self._ensure_ema_teacher(gen)
        with torch.no_grad():
            teacher_pred = _fwd(
                teacher,
                x_hat_now,
                t_now,
                takes_time=self._generator_accepts_time(teacher),
            )

        loss_consistency = func.mse_loss(student_pred, teacher_pred)

        # ---- VF anchor: ||V(f_theta(x_{t_n})) - m*||_1, noise-gated by beta(t).
        beta0 = float(self._knob("beta0"))
        if beta0 == 0.0:
            loss_vf_anchor = torch.zeros((), device=x0.device, dtype=loss_consistency.dtype)
        else:
            marker, mask = self._ensure_vf(x0)
            # Student reconstruction at the (low-noise) origin estimate.
            v_pred = self._restrict(student_pred, mask)
            v_target = self._restrict(marker.expand_as(student_pred), mask)
            beta = self._beta_gate(sigma_now)  # [B,1,..]; large sigma -> ~0
            l1 = (v_pred - v_target).abs()
            loss_vf_anchor = (beta * l1).mean()

        loss_total = loss_consistency + loss_vf_anchor

        # EMA update AFTER computing the targets for this step.
        self._update_ema(gen)

        return {
            "loss_total": loss_total,
            "loss_consistency": loss_consistency.detach()
            if loss_total is not loss_consistency
            else loss_consistency,
            "loss_vf_anchor": loss_vf_anchor.detach(),
        }

    # ------------------------------------------------------------- few-step sample
    @torch.no_grad()
    def sample(
        self,
        student_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_init: torch.Tensor,
        n_steps: int = 1,
    ) -> torch.Tensor:
        """Few-step (1--4) consistency generation.

        Each step evaluates the consistency student to jump to the origin
        estimate, then (for multi-step refinement) re-noises to the next, lower
        sigma and repeats -- the standard multistep consistency sampler.

        Args:
            student_fn: The trained student :math:`f_\\theta`.
            x_init: Starting noisy tensor ``x_{t_N}`` ``[B, C, H, W]``.
            n_steps: Number of refinement steps, ``1 <= n_steps <= 4``.

        Returns:
            Generated tensor, same shape as ``x_init``.
        """
        if not (1 <= n_steps <= 4):
            raise ValueError(f"n_steps must be in [1, 4], got {n_steps}")

        b = x_init.shape[0]
        dim_pad = [1] * (x_init.dim() - 1)

        # Resolved once, outside the loop: `_generator_accepts_time` runs
        # `inspect.signature` and is uncached ("detect ONCE", per its docstring),
        # and `student_fn` is fixed for the whole call. Introspection rather than
        # `except TypeError` so a genuine in-forward TypeError propagates instead
        # of silently degrading every refinement step to an unconditioned call.
        takes_time = self._generator_accepts_time(student_fn)

        def _fwd(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return student_fn(x, t.flatten()) if takes_time else student_fn(x)

        # Descending sigma schedule from t=1 down to a small floor.
        sigmas = torch.linspace(1.0, 0.0, n_steps + 1, device=x_init.device)
        x = x_init
        out = x
        for k in range(n_steps):
            t = sigmas[k].expand(b).view(b, *dim_pad)
            out = _fwd(x, t)  # origin estimate
            if k < n_steps - 1:
                # Re-noise to the next (lower) sigma for refinement.
                sigma_next = sigmas[k + 1]
                x = out + sigma_next * torch.randn_like(out)
        return out
