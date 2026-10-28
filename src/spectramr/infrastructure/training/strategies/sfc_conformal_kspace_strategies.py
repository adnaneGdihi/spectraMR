"""K-space SFC/conformal strategies — §§1 and 2 of the SFC plan.

Two minimal training strategies sharing this module:

* :class:`AdaptiveSFCHSSCStrategy` (``adaptive_sfc_hssc``) — content-
  adaptive quasi-conformal SFC over k-space for state-space
  reconstruction. Strictly generalises HSSC: when the learned
  Beltrami coefficient is zero the strategy degenerates to a fixed
  Hilbert traversal.
* :class:`ConformalDiffusionReconStrategy`
  (``conformal_diffusion_recon``) — diffusion reconstruction in
  conformally-mapped k-space coordinates with the
  :class:`ConformalDataConsistency` projection used as a hard DC
  step every reverse iteration.

Both strategies are minimal in lines: they reuse the framework's
losses (L1, MSE) and only add the SFC-/conformal-specific terms via
the registered losses in :mod:`spectramr.models.losses.conformal_geometry`.

References (numbers match the plan):
    [1] Ahlfors, *Lectures on Quasiconformal Mappings*, AMS, 2006.
    [3] Ahlfors, *Complex Analysis*, McGraw-Hill, 1979.
    [4] Driscoll & Trefethen, *Schwarz-Christoffel Mapping*, CUP, 2002.
    [8] Lui et al., "Teichmüller mapping", *SIAM J. Imag. Sci.*, 2014.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.data_consistency import ConformalDataConsistency
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.utils import (
    _callable_accepts_kwarg,
)
from spectramr.models.blocks.space_filling_curves import BeltramiSFCBlock

logger = logging.getLogger(__name__)


class AdaptiveSFCHSSCStrategy(BaseTrainingStrategy):
    """HSSC-style reconstruction with a learnable Beltrami SFC.

    The strategy attaches a :class:`BeltramiSFCBlock` to the
    reconstruction model and adds a Beltrami regularisation term to
    the standard L1 reconstruction loss. The SFC indices are passed
    to the model via ``kwargs['sfc_indices']`` for models that accept
    them; models that ignore the kwarg are still trained on the L1
    objective and the regulariser still constrains the SFC head.
    """

    # parallel_compat: The SFC block is an owned module, DDP-wrapped and run via call_owned.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: object) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("sfc", OWNED_MODULE),)

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "adaptive_sfc_hssc", None)
        grid_size = int(getattr(cfg, "grid_size", 32)) if cfg else 32
        in_channels = int(getattr(cfg, "in_channels", 1)) if cfg else 1
        self.lambda_mu = float(getattr(cfg, "lambda_mu", 0.01)) if cfg else 0.01
        self.lambda_traj = float(getattr(cfg, "lambda_traj", 0.01)) if cfg else 0.01
        self.sfc = BeltramiSFCBlock(grid_size=grid_size, in_channels=in_channels)
        if self.device is not None:
            self.sfc = self.sfc.to(self.device)
        # The Beltrami SFC head is learnable: register it for DDP, then put its
        # parameters on opt_g (else its regulariser gradients are never stepped).
        self.register_owned_module("sfc", self.sfc)
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in self.sfc.parameters() if p not in existing]
            if new_params:
                opt.add_param_group({"params": new_params})

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        # Resize input to the SFC grid for the Beltrami head; the
        # reconstructor itself sees the unrestricted input.
        gs = self.sfc.grid_size
        x_sfc = F.interpolate(input_batch, size=(gs, gs), mode="bilinear", align_corners=False)
        sfc_out = self.call_owned("sfc", x_sfc)
        # Only pass ``sfc_indices`` to generators whose ``forward`` advertises it
        # (or accepts ``**kwargs``). A strict ``forward(self, x)`` generator would
        # otherwise raise ``TypeError`` — the docstring promise is that models
        # which ignore the kwarg are still trained on the L1 objective. No silent
        # fallback: a generator that *names* the kwarg gets it; one that does not
        # is simply called positionally.
        fwd = getattr(self.generator_model, "forward", self.generator_model)
        if _callable_accepts_kwarg(fwd, "sfc_indices"):
            recon = self.generator_model(input_batch, sfc_indices=sfc_out["indices"])
        else:
            recon = self.generator_model(input_batch)
        recon_loss = F.l1_loss(recon, target_batch)
        beltrami_reg = sfc_out["mu"].abs().pow(2).mean()
        traj_smooth = (
            sfc_out["phi"].diff(dim=-1).abs().mean() + sfc_out["phi"].diff(dim=-2).abs().mean()
        ) * 0.5
        total = recon_loss + self.lambda_mu * beltrami_reg + self.lambda_traj * traj_smooth
        return {
            "g_total_loss": total,
            "adaptive_sfc_recon": recon_loss.detach(),
            "adaptive_sfc_beltrami_reg": beltrami_reg.detach(),
            "adaptive_sfc_trajectory_smoothness": traj_smooth.detach(),
        }


class ConformalDiffusionReconStrategy(BaseTrainingStrategy):
    """Diffusion reconstruction in conformally-mapped k-space.

    Forward pass: noise injection in k-space, denoising via the
    underlying generator, hard data-consistency step in conformal
    coordinates using :class:`ConformalDataConsistency`. The conformal
    Jacobian and mapped observation are supplied via the batch dict
    (``mask`` + ``conformal_jacobian``, populated by
    :class:`SFCConformalFMRIKeysWrapper` when the arm sets
    ``data.expose_conformal_jacobian: true``). Absent either key the
    strategy **raises** rather than silently running a vanilla MSE — the
    conformal DC step is the method claim (pitfall #16).

    Note: the shipped Jacobian is the identity map (``|Phi'(z)| = 1``),
    which makes the conformal DC mathematically identical to plain hard
    DC. A real non-Cartesian trajectory Jacobian is backlog; the raise
    guarantees the wiring is exercised even while the Jacobian is trivial.
    """

    # parallel_compat: Builds no parameters (ConformalDataConsistency holds a float alpha); the
    # generator runs only through its wrapped forward.
    parallel_compat = frozenset({"replicated"})

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "conformal_diffusion_recon", None)
        alpha = float(getattr(cfg, "dc_alpha", 1.0)) if cfg else 1.0
        self.sigma_min = float(getattr(cfg, "sigma_min", 1e-2)) if cfg else 1e-2
        self.sigma_max = float(getattr(cfg, "sigma_max", 1.0)) if cfg else 1.0
        self.dc = ConformalDataConsistency(alpha=alpha)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        device = target_batch.device
        b = target_batch.shape[0]
        log_lo, log_hi = (
            torch.tensor(self.sigma_min, device=device).log(),
            torch.tensor(self.sigma_max, device=device).log(),
        )
        sigma = (log_lo + torch.rand(b, device=device) * (log_hi - log_lo)).exp()
        noisy = target_batch + sigma.view(-1, 1, 1, 1) * torch.randn_like(target_batch)
        self._declare_model_input({"noisy": noisy, "sigma": sigma}, extra={"model_input_key": "noisy"})
        recon = self.generator_model(noisy)
        batch = kwargs.get("batch", {})
        mask = batch.get("mask")
        jac = batch.get("conformal_jacobian")
        if not (isinstance(mask, torch.Tensor) and isinstance(jac, torch.Tensor)):
            # Fail loud, do NOT warn-and-run-plain-MSE. The conformal
            # data-consistency projection IS this strategy's method claim;
            # without 'mask' + 'conformal_jacobian' the loss is an ordinary
            # Gaussian-denoising MSE and the "conformal" paradigm is inert
            # (pitfall #16 — the run smoke-PASSes while measuring nothing).
            # The keys are populated by the SFCConformalFMRIKeysWrapper when the
            # arm sets ``data.expose_conformal_jacobian: true`` and threads the
            # sampling mask through the DataPipelineDirector batch.
            raise ValueError(
                "ConformalDiffusionReconStrategy requires 'mask' and "
                "'conformal_jacobian' in the batch to run the conformal "
                "data-consistency step; got mask=%s, conformal_jacobian=%s. "
                "Set `data.expose_conformal_jacobian: true` and thread the "
                "sampling mask through the pipeline." % (type(mask).__name__, type(jac).__name__)
            )
        recon = self.dc(
            recon,
            y_tilde=input_batch,
            mask=mask.to(device),
            conformal_jacobian=jac.to(device),
        )
        loss = F.mse_loss(recon, target_batch)
        return {
            "g_total_loss": loss,
            "conformal_diffusion_dsm": loss.detach(),
        }


__all__ = [
    "AdaptiveSFCHSSCStrategy",
    "ConformalDiffusionReconStrategy",
]
