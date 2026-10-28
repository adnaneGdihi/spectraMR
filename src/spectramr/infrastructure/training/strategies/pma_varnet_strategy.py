"""Concrete PMA-VarNet Training Strategy.

Orchestrates the training of the Physics-Driven Marker-Anchored Variational Network
(Task 5 Blueprint). Evaluates the K-cascade unrolled network against the
DigitalTwinSimulator using the unified M4 Objective Loss.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.simulator_builder import (
    build_simulator_from_config,
)
from spectramr.models.losses.m4_losses import PMAGlobalLoss

logger = logging.getLogger(__name__)


class ConcretePMAVarNetStrategy(BaseTrainingStrategy):
    """Training strategy for the unrolled PMA-VarNet architecture.

    Leverages the Digital Twin Simulator to generate physical corruptions,
    then tasks the VarNet with inverting the physics using Operator_A_m4.
    """

    # PMAGlobalLoss is computed here, weighted by losses.reconstruction.lambda_*; no list entry
    # is read, so an empty builder stack is the design and a declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    def __init__(
        self,
        env: Any,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env, device, **kwargs)
        # No device re-resolution here. `BaseTrainingStrategy.__init__` sets
        # `self.device = env.device`, which the environment director already
        # resolved through `core.compute_device` (non-negotiable 9b). The
        # overwrite this replaces ran unconditionally -- the factory never
        # passes `device`, so `device or ...` always took the right-hand side
        # -- and it silently relocated the run from `env.device` to `cuda:0`
        # on a multi-GPU node, or to CPU on a GPU-less one.

    def _setup_strategy_specific_components(self) -> None:
        """Initialize the DigitalTwinSimulator and PMA losses."""
        # 1. Build Single Source of Truth Simulator
        self.simulator = build_simulator_from_config(self.config, self.device)
        logger.info(f"Initialized Simulator: {self.simulator.marker_type} markers.")

        # 2. Extract loss weights from SSOT (config.losses.reconstruction)
        recon = self.config.losses.reconstruction
        l_l1 = recon.lambda_l1
        l_ssim = recon.lambda_ssim
        l_marker = recon.lambda_marker

        # Create the comprehensive PMA-VarNet explicit loss
        self.pma_loss = PMAGlobalLoss(
            lambda_l1=l_l1,
            lambda_ssim=l_ssim,
            lambda_marker=l_marker,
        ).to(self.device)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute unrolled cascade losses for PMA-VarNet.

        Args:
            input_batch: Not used, target is cleanly corrupted natively.
            target_batch: Clean multi-contrast anatomy [B, C, H, W].
            epoch: Current training epoch.

        Returns:
            Dictionary with `g_total_loss` and individual components.
        """
        B, C, H, W = target_batch.shape

        # Complex cast if real-stacked [B, 2, H, W] representing 1 contrast
        if not target_batch.is_complex():
            if C == 2:
                target_complex = torch.complex(target_batch[:, 0:1], target_batch[:, 1:2])
            else:
                target_complex = torch.complex(target_batch, torch.zeros_like(target_batch))
        else:
            target_complex = target_batch

        # 1. Run Digital Twin Simulator
        with torch.no_grad():
            corrupted_complex, marker_prior, _joint_clean = self.simulator(target_complex)
            # The simulator returns (corrupted_image, marker_prior, joint_clean);
            # the binary marker ROI mask is a separate attribute (as in
            # vf_admm / virtual_fiducial). The old 3-tuple unpack mislabeled
            # marker_prior as the mask and joint_clean as the prior.
            marker_mask = self.simulator.marker_mask.to(self.device)

        # 2. Convert corrupted image to k-space as the input to VarNet
        # y_meas: [B, N_con, Nc, N_k]. We treat Nc=1 and use Cartesian FFT for this basic strategy.
        # [FIX] Guard: only apply FFT if corrupted_complex is in image domain.
        # If data pipeline already provides k-space, skip to avoid double-FFT.
        if corrupted_complex.is_complex():
            y_meas_cartesian = fft2c(corrupted_complex)  # [B, N_con, H, W]
        else:
            # Real-valued input: convert to complex then FFT
            if corrupted_complex.shape[1] == 2:
                corrupted_as_complex = torch.complex(
                    corrupted_complex[:, 0:1], corrupted_complex[:, 1:2]
                )
                y_meas_cartesian = fft2c(corrupted_as_complex)
            else:
                y_meas_cartesian = fft2c(
                    torch.complex(corrupted_complex, torch.zeros_like(corrupted_complex))
                )
        # Flatten spatial dims to conceptualize non-Cartesian N_k for the forward operator
        y_meas = y_meas_cartesian.view(B, target_complex.shape[1], 1, H * W)  # [B, N_con, 1, N_k]

        # 3. Construct Analytical Parameter Maps (Phase 1 extracted fields)
        # In a complete pipeline, these come from Task 3 extractors.
        # Here we initialize identity maps so the analytical operator functions properly.
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-0.5, 0.5, H, device=self.device),
            torch.linspace(-0.5, 0.5, W, device=self.device),
            indexing="ij",
        )
        dK = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=0).unsqueeze(0).expand(B, 2, H * W)

        maps_dict = {
            "alpha": None,
            "T_kin": None,
            "B0": None,
            "S_map": None,
            "Psi": None,
            "dK": dK,
            "dcf": None,
        }

        # 4. Execute VarNet K-Cascades
        # Generator signature requires y_meas passed down
        reconstructed = self.generator_model(
            x=y_meas,
            y_meas=y_meas,
            maps_dict=maps_dict,
            marker_mask=marker_mask,
            marker_prior=marker_prior,
        )

        joint_clean, _ = self.simulator.embedder(target_complex)

        # 5. Evaluate the global M4 Objective
        total_loss = self.pma_loss(
            prediction=reconstructed,
            target=joint_clean,
            marker_mask=marker_mask,
            marker_prior=marker_prior,
        )

        self._last_prediction = reconstructed.detach()
        self._last_target = joint_clean
        return {
            "g_total_loss": total_loss,
            "loss_pma_global": total_loss.detach(),
        }

    def _compute_metrics_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute standard metrics."""
        return {}

    def validation_step(self, val_batch: Any, batch_idx: int) -> dict[str, torch.Tensor]:
        """Forward validation pass.

        F3/E22 (2026-05-21 smoke audit): unwrap ``TrainingBatch``
        before passing to ``_compute_losses_impl``. The prior code
        passed the raw ``val_batch`` (a TrainingBatch dataclass) as
        ``target_batch``, which crashed at ``target_batch.shape`` on
        line 79 with ``AttributeError: 'TrainingBatch' object has no
        attribute 'shape'`` across 4 experiments (PMA + JFE-VarNet
        cohort). See TODO/audit/smoke_audit_20260521.md §F3.

        The twin draws its SNR, motion and undersampling mask from the global RNG,
        which ``initialize_accelerator`` seeds from ``run.seed``. Validation does
        not reseed it, since that would clobber the accelerator's seeding contract;
        controlled randomness here should use a local ``torch.Generator``.
        """
        input_batch, target_batch = self._unpack_batch(val_batch)
        # #1190: calling ``_compute_losses_impl`` directly bypasses the
        # ``_compute_losses`` wrapper that emits the ``model_output`` snapshot,
        # so arm it here. ``snapshot_source("val")`` is the OUTER manager so the
        # phase is still "val" when the emit runs on exit -- without it the
        # record would claim the training data chain.
        with (
            self.snapshot_source("val"),
            self._capture_model_output(
                module=self.generator_model,
                input_batch=input_batch,
                target_batch=target_batch,
                step=batch_idx,
            ),
        ):
            out = self._compute_losses_impl(input_batch, target_batch, epoch=0)
        # The arms select on ``val_psnr``; grade the reconstruction against the
        # objective's own target (anatomy plus corner fiducials), on magnitudes,
        # through the registry metrics computer (#1774).
        computer = self._get_validation_metrics_computer(self.config)
        computed = computer.compute(self._last_prediction.abs(), self._last_target.abs())
        out.update({f"val_{name}": value for name, value in computed.items()})
        return out


__all__ = ["ConcretePMAVarNetStrategy"]
