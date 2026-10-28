"""fMRI k-space strategies — §§1 and 2 of the fMRI plan.

* :class:`SpatiotemporalAdaptiveSFCReconStrategy`
  (``spatiotemporal_adaptive_sfc_recon``) — 4D Beltrami SFC over the
  ``(x, y, z, t)`` BOLD volume. The strategy invokes
  :class:`SeparableBeltrami4DSolver` to recover the joint warp, and
  combines an L1 reconstruction loss with a Beltrami regulariser and
  a temporal-smoothness term.

* :class:`BeltramiEPIDistortionStrategy`
  (``beltrami_epi_distortion``) — estimates the susceptibility-driven
  EPI distortion as a B0 field map, then applies the closed-form
  Beltrami coefficient to the predicted displacement. The
  :func:`beltrami_from_b0` helper guarantees the warp is quasi-conformal.

References:
    [1] Feinberg et al., "Multiplexed EPI for sub-second whole brain
        fMRI", *PLOS ONE*, 5(12), 2010.
    [9] Andersson et al., "How to correct susceptibility distortions",
        *NeuroImage*, 20(2), 2003.
    [10] Jezzard & Balaban, *Magn. Reson. Med.*, 34(1), 1995.
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.epi_forward import (
    apply_epi_distortion,
    beltrami_from_b0,
)
from spectramr.infrastructure.physics.manifolds.beltrami_4d import (
    SeparableBeltrami4DSolver,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.models.capabilities import StrategyCapabilities


class SpatiotemporalAdaptiveSFCReconStrategy(BaseTrainingStrategy):
    """4D-SFC fMRI reconstruction strategy.

    Treats ``input_batch`` as ``[B, C, T, H, W]`` BOLD-volume features
    and runs the generator over the spatial slices with a Beltrami
    regularisation term computed from a small μ-head and a separable
    4D Beltrami solver.
    """

    #: The whole objective is inline: an L1 fidelity term plus the two Beltrami
    #: regularisers; no builder module is folded (#1768).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"l1"})
    folds_image_losses: ClassVar[bool | None] = False

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "spatiotemporal_adaptive_sfc_recon", None)
        self.lambda_mu = float(getattr(cfg, "lambda_mu", 0.01)) if cfg else 0.01
        self.lambda_T = float(getattr(cfg, "lambda_t", 0.01)) if cfg else 0.01
        self.solver = SeparableBeltrami4DSolver()

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        # Predict reconstruction at the input rank.
        recon = self.generator_model(input_batch)
        recon_loss = F.l1_loss(recon, target_batch)
        # 4D Beltrami regulariser over the spatiotemporal reconstruction.
        if recon.dim() == 5:  # [B, C, T, H, W]
            # Derive the SFC-Beltrami coefficients from the model OUTPUT
            # (``recon``) so the regulariser carries a gradient back to the
            # model parameters. Previously mu/nu came from the no-grad
            # ``input_batch[:, 0]``, making beltrami_reg/temporal_reg constants
            # (zero gradient) and ``lambda_mu``/``lambda_T`` inert — pitfall #15.
            # This mirrors the MRF sibling fix in mrf_kspace_strategies.py.
            x = recon[:, 0]  # [B, T, H, W] — first channel, grad-carrying
            _, T, _, _ = x.shape
            mu = 0.1 * x.diff(dim=-1, prepend=x[..., :1]).to(torch.cfloat)  # [B,T,H,W]
            nu = 0.05 * torch.tanh(x.mean(dim=(-2, -1)))  # [B, T]
            res = self.solver(mu, nu)  # temporal warp is grad-carrying via nu
            beltrami_reg = mu.abs().pow(2).mean()
            temporal_reg = (
                res.phi_temporal.diff(dim=-1).abs().mean() if T >= 2 else recon.new_zeros(())
            )
        else:
            beltrami_reg = recon.new_zeros(())
            temporal_reg = recon.new_zeros(())
        total = recon_loss + self.lambda_mu * beltrami_reg + self.lambda_T * temporal_reg
        return {
            "g_total_loss": total,
            "spatiotemporal_sfc_recon": recon_loss.detach(),
            "spatiotemporal_sfc_beltrami_reg": beltrami_reg.detach(),
            "spatiotemporal_sfc_temporal_reg": temporal_reg.detach(),
        }


class BeltramiEPIDistortionStrategy(BaseTrainingStrategy):
    """EPI distortion correction with a Beltrami-quasiconformal warp.

    The generator predicts a B0 field map (Hz). The strategy converts
    the field to a displacement, applies it to the undistorted target,
    and matches the simulated EPI to the observation.
    """

    #: EPI readout physics, tagged for BOTH regimes that use an EPI readout -- a
    #: precise claim, not an inflated one. The generator predicts a dB0 field (Hz),
    #: apply_epi_distortion simulates the readout INLINE (real gyromagnetic ratio,
    #: phase-encode displacement only) and the residual matches it to the
    #: observation: physics-in-the-loop with no silent fallback.
    #:
    #: HONESTY NOTE: this models the EPI READOUT, not the BOLD signal. It is what
    #: backs mri_functional's PARTIAL claim, and mri_functional therefore means
    #: 'distortion correction exists', NOT 'BOLD modelling exists'. Reaching LIVE
    #: still needs tSNR/GLM losses, a BOLD metric and a real BOLD recon strategy.
    #: Its four fmri_* siblings are deliberately UNTAGGED (they degrade to plain
    #: L1/DSM without their conformal inputs). See docs/reference/workflow_backlog.md.
    capabilities: ClassVar[StrategyCapabilities] = StrategyCapabilities(
        workflows=frozenset({Regime.FUNCTIONAL, Regime.DIFFUSION_WEIGHTED}),
        tasks=frozenset({Task.RECONSTRUCTION}),
    )

    #: The objective is the simulated-EPI residual plus the mu penalty, both
    #: inline; no builder module is folded (#1768).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset({"beltrami_epi_residual"})
    folds_image_losses: ClassVar[bool | None] = False

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "beltrami_epi_distortion", None)
        self.t_esp = float(getattr(cfg, "t_esp", 0.5e-3)) if cfg else 0.5e-3
        self.lambda_mu = float(getattr(cfg, "lambda_mu", 0.01)) if cfg else 0.01

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        delta_b0 = self.generator_model(input_batch)
        simulated = apply_epi_distortion(target_batch, delta_b0, t_esp=self.t_esp)
        residual = F.l1_loss(simulated, input_batch)
        # The Beltrami coefficient is real and bounded by construction;
        # a soft penalty on its magnitude discourages large warps.
        mu = beltrami_from_b0(delta_b0, t_esp=self.t_esp)
        mu_reg = mu.abs().pow(2).mean()
        total = residual + self.lambda_mu * mu_reg
        return {
            "g_total_loss": total,
            "beltrami_epi_residual": residual.detach(),
            "beltrami_epi_mu_reg": mu_reg.detach(),
        }

    @torch.no_grad()
    def validation_step(
        self, input_batch: torch.Tensor, target_batch: torch.Tensor, **kwargs: Any
    ) -> dict[str, Any]:
        """Grade the predicted field by the EPI image it predicts.

        The generator emits a field in Hz, so scoring its output against the
        undistorted target compares incommensurable quantities (#1768). The target
        warped by the predicted field is graded against the observed EPI instead,
        through the registry metrics computer; there is no exact inverse warp to
        grade a corrected image with.
        """
        del kwargs
        input_batch = input_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)
        self.generator_model.eval()
        delta_b0 = self.generator_model(input_batch)
        simulated = apply_epi_distortion(target_batch, delta_b0, t_esp=self.t_esp)
        computer = self._get_validation_metrics_computer(self.config)
        metrics = {
            f"val_{name}": value
            for name, value in computer.compute(simulated, input_batch).items()
        }
        metrics["val_beltrami_epi_residual"] = F.l1_loss(simulated, input_batch)
        return metrics


__all__ = [
    "BeltramiEPIDistortionStrategy",
    "SpatiotemporalAdaptiveSFCReconStrategy",
]
