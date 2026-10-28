"""Field-Probe-Coupled Reconstruction (ULF-PR-10 / II.7).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-10.

External field-probe sensors (NMR probes, fluxgate magnetometers)
record B0(t) drift during the acquisition.  This strategy now wires
through three real components:

1. A small learnable :class:`PhaseDemodulator` MLP that maps the
   per-line probe trace to a per-line correction phase ``φ̂(line)``
   so we no longer fold the whole readout into a single global
   rotation.
2. Demodulation applied per-readout-line on the raw k-space (input
   to the recon network) using the Fourier-shift theorem.
3. Probe self-consistency loss: applying the demod twice (forward +
   inverse) should be the identity.

Public method :meth:`apply_demod` is exposed so the inference path can
reuse the same demodulator on held-out data.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.distributed.owned_modules import OWNED_PARAM_GROUP

from .owned_param_groups import add_owned_param_group
from .reconstruction import ReconstructionTrainingStrategy

# Gyromagnetic ratio of 1H (rad/s/T)
GAMMA_RAD_PER_S_PER_T = 2 * math.pi * 42.577_478_92e6


class PhaseDemodulator(nn.Module):
    """Map a probe trace (B, T) to a per-readout-line phase (B, H)."""

    def __init__(self, n_readout_lines: int, hidden: int = 64) -> None:
        super().__init__()
        self.n_readout_lines = n_readout_lines
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, b0_trace: torch.Tensor, n_lines: int | None = None) -> torch.Tensor:
        """b0_trace: (B, T) → per-line phase (B, n_lines or self.n_readout_lines).

        Resamples the trace down to the readout lines via uniform
        partitioning; each segment is reduced by mean before the MLP.
        """
        n = n_lines or self.n_readout_lines
        B, T = b0_trace.shape
        if n > T:
            pad = b0_trace.new_zeros(B, n - T)
            b0_trace = torch.cat([b0_trace, pad], dim=1)
            T = n
        # mean-pool over uniform windows
        windowed = b0_trace[:, : (T // n) * n].view(B, n, T // n).mean(dim=-1)
        out = self.net(windowed.unsqueeze(-1) * GAMMA_RAD_PER_S_PER_T)
        return out.squeeze(-1)


class FieldProbeCoupledStrategy(ReconstructionTrainingStrategy):
    """Recon with learnable per-line phase demodulation."""

    # parallel_compat: The phase demodulator is a bare group on opt_g, refused under ddp. It is
    # built only when a batch carries b0_trace and kspace, and rebuilt whenever the readout-line
    # count changes, so its creation is not a rank-symmetric point at which to wrap it.
    parallel_compat = frozenset({"replicated"})

    lambda_self: float = 1e-2

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("_demod", OWNED_PARAM_GROUP))

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._demod: PhaseDemodulator | None = None

    def _ensure_demod(self, n_lines: int, device: torch.device) -> PhaseDemodulator:
        if self._demod is None or self._demod.n_readout_lines != n_lines:
            self._demod = PhaseDemodulator(n_lines).to(device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                add_owned_param_group(
                    self,
                    opt,
                    {"params": list(self._demod.parameters()), "lr": 1e-4},
                    owner="its phase demodulator",
                )
        return self._demod

    def apply_demod(self, kspace: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Multiply kspace by exp(-i φ) per readout line.

        kspace: (B, C, H, W) complex or (B, 2C, H, W) real-imag.
        phase:  (B, H) per-line phase.
        """
        rot = (-1j * phase).exp().unsqueeze(-1).unsqueeze(1)  # (B, 1, H, 1)
        if torch.is_complex(kspace):
            return kspace * rot
        rot_re, rot_im = rot.real, rot.imag
        out = torch.empty_like(kspace)
        for i in range(0, kspace.shape[1], 2):
            r, im = kspace[:, i : i + 1], kspace[:, i + 1 : i + 2]
            out[:, i : i + 1] = r * rot_re - im * rot_im
            out[:, i + 1 : i + 2] = r * rot_im + im * rot_re
        return out

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
        if not isinstance(batch, dict) or "b0_trace" not in batch or "kspace" not in batch:
            return super()._compute_losses_impl(
                input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
            )

        kspace = batch["kspace"]
        b0_trace = batch["b0_trace"]
        H = kspace.shape[-2]
        demod = self._ensure_demod(H, kspace.device)
        phase = demod(b0_trace, n_lines=H)

        kspace_corr = self.apply_demod(kspace, phase)

        # Self-consistency: applying demod with -phase should round-trip.
        kspace_round = self.apply_demod(kspace_corr, -phase)
        self_loss = self.lambda_self * (kspace_round - kspace).abs().pow(2).mean()

        patched = dict(batch)
        patched["kspace"] = kspace_corr
        base = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **{**kwargs, "batch": patched},
        )
        base["loss_demod_self_consistency"] = self_loss
        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + self_loss
                break
        return base
