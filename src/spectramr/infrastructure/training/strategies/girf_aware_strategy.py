"""GIRF-Aware Reconstruction (ULF-PR-9 / II.6).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-9.

The strategy now wires through the dedicated
:mod:`spectramr.infrastructure.physics.girf` primitives:

1. :class:`GIRFKernel` — learnable per-axis FIR filter, initialised
   to delta so the untrained filter is a no-op.
2. :func:`apply_girf` — frequency-domain convolution applied to the
   requested gradient waveform (``batch["gradient_waveform"]``).
3. :func:`waveform_to_trajectory` — converts the corrected waveform
   into a corrected k-space trajectory which is forwarded into the
   batch under ``batch["trajectory"]`` so downstream ops (NUFFT)
   automatically pick up the correction.

Loss = parent recon fidelity
     + ``λ_ridge · ‖h − δ‖²``  (Tikhonov ridge toward identity).
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP
from spectramr.infrastructure.physics.girf import (
    GIRFKernel,
    apply_girf,
    waveform_to_trajectory,
)

from .owned_param_groups import add_owned_param_group
from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class GIRFAwareStrategy(ReconstructionTrainingStrategy):
    """Reconstruction with learnable GIRF correction on requested gradients."""

    # parallel_compat: The GIRF kernel is a bare group on opt_g, refused under ddp. It is built
    # only when a batch carries gradient_waveform, sized and rebuilt from that batch, so its
    # creation is not a rank-symmetric point at which to wrap it.
    parallel_compat = frozenset({"replicated"})

    n_axes: int = 3
    kernel_size: int = 21
    lambda_ridge: float = 1e-3
    dt: float = 1e-5

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_girf", OWNED_PARAM_GROUP))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._girf: GIRFKernel | None = None

    def _ensure_girf(self, n_axes: int, device: torch.device) -> GIRFKernel:
        if self._girf is None or self._girf.kernel.shape[0] != n_axes:
            self._girf = GIRFKernel(n_axes=n_axes, kernel_size=self.kernel_size).to(device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                add_owned_param_group(
                    self,
                    opt,
                    {"params": list(self._girf.parameters()), "lr": 1e-4},
                    owner="its GIRF kernel",
                )
        return self._girf

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (round 6 2026-05-17): orchestrator-kwargs canonical signature.
        # See TODO/audit/smoke_audit_20260516.md §F6.
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )

        waveform = batch.get("gradient_waveform")
        if waveform is not None and waveform.dim() == 3:
            n_axes = waveform.shape[1]
            girf = self._ensure_girf(n_axes, waveform.device)
            corrected = apply_girf(waveform, girf.kernel)
            traj = waveform_to_trajectory(corrected, dt=self.dt)
            patched = dict(batch)
            patched["trajectory"] = traj
            patched["corrected_gradient_waveform"] = corrected
            base = super()._compute_losses_impl(
                input_batch=input_batch,
                target_batch=target_batch,
                epoch=epoch,
                **{**kwargs, "batch": patched},
            )
            ridge = self.lambda_ridge * girf.delta_distance()
            base["loss_girf_ridge"] = ridge
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + ridge
                    break
            return base

        # No waveform → degrade to parent path with a tiny no-op ridge.
        base = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )
        if self._girf is not None:
            ridge = self.lambda_ridge * self._girf.delta_distance()
            base["loss_girf_ridge"] = ridge
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + ridge
                    break
        return base
