"""MRI-SLAM — Joint Trajectory-Image Estimation (ULF-PR-5 / II.2).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §ULF-PR-5.

Joint optimisation of a per-subject k-space trajectory δτ and the
reconstructed image. Each step:

1. ``image_pred = recon_model(...)``  — already produced by the parent
   ``ReconstructionTrainingStrategy``.
2. ``k_pred = forward_op(image_pred, traj_nominal + δτ)``  — using the
   NUFFT forward model from
   :mod:`spectramr.infrastructure.physics.nufft_ops` when available, falling
   back to a Cartesian shifted-FFT proxy otherwise.
3. Photometric (data-consistency) loss between ``k_pred`` and the
   measured ``k_meas``.
4. Smoothness + drift regularisers on δτ to keep the trajectory
   physically plausible.

The trajectory deltas are :class:`~torch.nn.Parameter`s registered into
the trainer's optimiser on first sight per ``subject_id``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class MRISLAMStrategy(ReconstructionTrainingStrategy):
    """Joint trajectory-image SLAM-style reconstruction."""

    # parallel_compat: Creates per-subject trajectory deltas mid-training; the set can differ per rank.
    parallel_compat = frozenset()

    # Loss weights + trajectory-delta LR; overridable via the typed
    # training.mri_slam block (previously hardcoded inline constants).
    lambda_smooth: float = 1e-3
    lambda_drift: float = 1e-4
    lambda_photometric: float = 0.5
    trajectory_lr: float = 1e-4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._traj_deltas: nn.ParameterDict = nn.ParameterDict()
        self._traj_running: dict[str, torch.Tensor] = {}
        self._nufft_cache: Any = None
        # Typed sub-block (pitfall #15: read + validate + stamp).
        cfg = getattr(self.config.training, "mri_slam", None)
        if cfg is not None:
            self.lambda_smooth = float(cfg.lambda_smooth)
            self.lambda_drift = float(cfg.lambda_drift)
            self.lambda_photometric = float(cfg.lambda_photometric)
            self.trajectory_lr = float(cfg.trajectory_lr)
        logger.info(
            "MRISLAMStrategy (source=%s): lambda_smooth=%.4g, lambda_drift=%.4g, "
            "lambda_photometric=%.4g, trajectory_lr=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_smooth,
            self.lambda_drift,
            self.lambda_photometric,
            self.trajectory_lr,
        )

    def _delta_for_subject(
        self, subj_id: str, n_samples: int, device: torch.device
    ) -> nn.Parameter:
        if subj_id not in self._traj_deltas:
            self._traj_deltas[subj_id] = nn.Parameter(torch.zeros(n_samples, 2, device=device))
            self._traj_running[subj_id] = torch.zeros(n_samples, 2, device=device)
            opt = getattr(self.env, "opt_g", None)
            if opt is not None:
                opt.add_param_group(
                    {"params": [self._traj_deltas[subj_id]], "lr": self.trajectory_lr}
                )
        return self._traj_deltas[subj_id]

    @staticmethod
    def _trajectory_smoothness(delta: torch.Tensor) -> torch.Tensor:
        if delta.shape[0] < 2:
            return delta.new_zeros(())
        diff = delta[1:] - delta[:-1]
        return diff.pow(2).mean()

    def _photometric_loss(
        self,
        image_pred: torch.Tensor,
        traj: torch.Tensor,
        k_meas: torch.Tensor,
    ) -> torch.Tensor:
        """NUFFT(image, traj) ↔ k_meas photometric residual.

        Falls back to a Cartesian phase-shifted FFT proxy when
        ``torchkbnufft`` is unavailable.  The proxy treats ``traj`` as
        a per-sample (kx, ky) shift of a Cartesian grid and uses the
        Fourier-shift theorem to produce shape-comparable k-space.
        """
        if image_pred.dtype not in (torch.complex64, torch.complex128):
            image_pred = torch.complex(image_pred, torch.zeros_like(image_pred))

        try:
            from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel  # type: ignore

            B, _, H, W = image_pred.shape
            if self._nufft_cache is None or self._nufft_cache.im_size != (H, W):
                self._nufft_cache = NUFFTForwardModel(im_size=(H, W)).to(image_pred.device)
            k_pred = self._nufft_cache.forward_project(image_pred, traj.transpose(-1, -2))
            return (k_pred - k_meas).abs().pow(2).mean()
        except (ImportError, ModuleNotFoundError) as exc:
            # Narrow: torchkbnufft is imported lazily inside NUFFTForwardModel
            # .__init__, so ONLY a genuinely-missing-torchkbnufft case should
            # fall back to the Cartesian proxy. A broad ``except Exception``
            # here would swallow real shape/device/CUDA errors from
            # forward_project and silently backprop the proxy (pitfall #9/#10).
            logger.debug("NUFFT path unavailable (%s); using Cartesian proxy.", exc)
            from spectramr.infrastructure.physics.fft_ops import fft2c

            _, _, H, W = image_pred.shape
            k_full = fft2c(image_pred)
            yy, xx = torch.meshgrid(
                torch.linspace(-0.5, 0.5, H, device=image_pred.device),
                torch.linspace(-0.5, 0.5, W, device=image_pred.device),
                indexing="ij",
            )
            shift = traj.mean(dim=0)  # mean per-volume drift
            phase = torch.exp(-2j * torch.pi * (shift[0] * xx + shift[1] * yy))
            k_shifted = k_full * phase
            if k_meas.shape != k_shifted.shape:
                # No self-comparison fallback: substituting the prediction's own
                # ``k_full`` as the target yields ``(k_full*phase - k_full)`` — a
                # measurement-independent phase-ramp-only loss that learns
                # nothing from ``k_meas`` (pitfall #9). Surface the mismatch.
                raise ValueError(
                    "MRI-SLAM Cartesian proxy: measured k-space shape "
                    f"{tuple(k_meas.shape)} != predicted {tuple(k_shifted.shape)}; "
                    "the trajectory/measurement geometry is inconsistent."
                )
            return (k_shifted - k_meas).abs().pow(2).mean()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # F6 (2026-05-17 round 5): the base orchestrator calls
        # ``_compute_losses_impl(input_batch=..., target_batch=...,
        # epoch=..., **kwargs)``. The pre-F6 ``def ...(self, batch, ...)``
        # signature couldn't bind those kwargs, raising
        # TypeError at the first training step. Now we accept the
        # canonical names AND resolve the original ``batch`` dict via
        # the shared ``_resolve_legacy_batch`` helper, which checks
        # ``kwargs["batch"]`` first and falls back to ``input_batch``.
        base = super()._compute_losses_impl(
            input_batch=input_batch, target_batch=target_batch, epoch=epoch, **kwargs
        )
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if not isinstance(batch, dict):
            return base

        subj = str(batch.get("subject_id", "default"))
        traj = batch.get("trajectory")
        if traj is None:
            return base

        device = traj.device
        delta = self._delta_for_subject(subj, traj.shape[-2], device)
        traj_corrected = traj + delta

        # Update running mean (no grad) — used for drift regulariser
        with torch.no_grad():
            self._traj_running[subj] = 0.99 * self._traj_running[subj] + 0.01 * delta.detach()

        smooth = self.lambda_smooth * self._trajectory_smoothness(delta)
        drift = self.lambda_drift * (delta - self._traj_running[subj]).pow(2).mean()
        base["loss_trajectory_smoothness"] = smooth
        base["loss_trajectory_drift"] = drift

        image_pred = batch.get("prediction")
        k_meas = batch.get("kspace_measured")
        if image_pred is not None and k_meas is not None:
            photo = self.lambda_photometric * self._photometric_loss(
                image_pred, traj_corrected, k_meas
            )
            base["loss_photometric"] = photo
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + photo + smooth + drift
                    break
        else:
            for total_key in ("loss_total", "g_total_loss"):
                if total_key in base:
                    base[total_key] = base[total_key] + smooth + drift
                    break
        return base
