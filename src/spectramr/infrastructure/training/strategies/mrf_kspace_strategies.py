"""MRF k-space + posterior-inference strategies — §§1 and 2.

* :class:`SpatiotemporalMRFReconStrategy`
  (``spatiotemporal_mrf_recon``) — 4D Beltrami SFC over spiral
  ``(k_x, k_y, t)`` MRF acquisitions. Reuses
  :class:`SeparableBeltrami4DSolver`.
* :class:`RiemannianMRFDiffusionStrategy`
  (``riemannian_mrf_diffusion``) — Riemannian DSM on the
  Bloch parameter manifold. Score is provided by
  ``mrf_tangent_score``; the manifold is the 5-D
  :class:`BlochMRFManifold`.

References:
    [1] Ma et al., "Magnetic resonance fingerprinting", *Nature*,
        495, 2013.
    [11] Amari & Nagaoka, *Methods of Information Geometry*, AMS, 2000.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.physics.manifolds.bloch_mrf_manifold import (
    BlochMRFManifold,
    cayley_hilbert_chart,
    inverse_cayley_hilbert_chart,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class SpatiotemporalMRFReconStrategy(BaseTrainingStrategy):
    """4D Beltrami SFC for spiral MRF reconstruction.

    Treats ``input_batch[B, C, T, H, W]`` as the spiral MRF data and
    produces a parameter map ``[B, P, H, W]`` (P = 3 by default for
    (T1, T2, M0)). The Beltrami solver is invoked diagnostically;
    the actual reconstruction is performed by the underlying
    generator.
    """

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        # Typed sub-block (pitfall #15: read validated typed fields + stamp).
        # Previously read via getattr against the extra="allow" dict, which
        # silently returned the defaults for a supplied YAML block.
        cfg = getattr(self.config.training, "spatiotemporal_mrf_recon", None)
        self.lambda_mu = float(cfg.lambda_mu) if cfg is not None else 0.01
        self.lambda_nu = float(cfg.lambda_nu) if cfg is not None else 0.01
        logger.info(
            "SpatiotemporalMRFReconStrategy (source=%s): lambda_mu=%.4g, lambda_nu=%.4g",
            "config" if cfg is not None else "defaults",
            self.lambda_mu,
            self.lambda_nu,
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        recon = self.generator_model(input_batch)
        recon_loss = F.l1_loss(recon, target_batch)
        mu_reg = recon.new_zeros(())
        nu_reg = recon.new_zeros(())
        if input_batch.dim() == 5:
            # Derive the SFC-Beltrami coefficients from the model OUTPUT
            # (``recon``) so the regulariser carries a gradient back to the
            # model parameters. Previously mu/nu came from the no-grad
            # ``input_batch[:, 0]``, making mu_reg/nu_reg constants (zero
            # gradient) and ``lambda_mu``/``lambda_nu`` inert — pitfall #15
            # (audit 2026-06). ``recon`` is the [B, P, H, W] parameter map;
            # the first channel carries the spatial structure the conformal
            # Beltrami prior should constrain.
            # (A dead ``self.solver(mu, nu)`` diagnostic forward whose
            # Beltrami4DResult was discarded was removed in an earlier pass.)
            x = recon[:, 0]  # [B, H, W] — first parameter map (grad-carrying)
            mu = 0.05 * (x.diff(dim=-1, prepend=x[..., :1]).to(torch.cfloat))
            nu = 0.05 * torch.tanh(x.mean(dim=(-2, -1)))
            mu_reg = mu.abs().pow(2).mean()
            nu_reg = nu.abs().pow(2).mean()
        total = recon_loss + self.lambda_mu * mu_reg + self.lambda_nu * nu_reg
        return {
            "g_total_loss": total,
            "stmrf_param_fidelity": recon_loss.detach(),
            "stmrf_mu_reg": mu_reg.detach(),
            "stmrf_nu_reg": nu_reg.detach(),
        }


class RiemannianMRFDiffusionStrategy(BaseTrainingStrategy):
    """Riemannian DSM on the Bloch parameter manifold (MRF §2).

    Forward noising: chart-space Brownian motion. Reverse: tangent
    score computed by ``mrf_tangent_score`` and integrated by the
    manifold's exp map. Training loss is chart-space DSM since the
    Cayley-Hilbert chart is conformal at the centre and the chart
    Euclidean metric is a first-order approximation to the Fisher
    metric there.
    """

    # Debug-snapshot contract (non-negotiable 14): the model is fed a noised
    # tensor, not the prepared input; declared here and emitted under the tag.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def __init__(self, env: TrainingEnvironment, **kwargs: Any) -> None:
        super().__init__(env=env, **kwargs)
        cfg = getattr(self.config.training, "riemannian_mrf_diffusion", None)
        self.sigma_min = float(cfg.sigma_min) if cfg is not None else 1e-2
        self.sigma_max = float(cfg.sigma_max) if cfg is not None else 1.0
        logger.info(
            "RiemannianMRFDiffusionStrategy (source=%s): sigma_min=%.4g, sigma_max=%.4g",
            "config" if cfg is not None else "defaults",
            self.sigma_min,
            self.sigma_max,
        )
        self.manifold = BlochMRFManifold()

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch
        if target_batch.shape[-1] != 5:
            raise ValueError("RiemannianMRFDiffusionStrategy expects [..., 5] Bloch params")
        device = target_batch.device
        b = target_batch.shape[0]
        log_lo, log_hi = (
            torch.tensor(self.sigma_min, device=device).log(),
            torch.tensor(self.sigma_max, device=device).log(),
        )
        sigma = (log_lo + torch.rand(b, device=device) * (log_hi - log_lo)).exp()
        z_clean = cayley_hilbert_chart(target_batch)
        noise = torch.randn_like(z_clean)
        z_noisy = z_clean + sigma.view(-1, 1) * noise
        theta_noisy = inverse_cayley_hilbert_chart(z_noisy)
        fingerprint = kwargs.get("fingerprint")
        if fingerprint is None:
            raise ValueError(
                "RiemannianMRFDiffusionStrategy requires a 'fingerprint' key on "
                "the batch; none was supplied. This used to be "
                "kwargs.get('fingerprint'), which passes None and turns the "
                "strategy into an UNCONDITIONAL Riemannian score model on Bloch "
                "parameters -- a reasonable model, but not fingerprinting, and "
                "the fingerprint conditioning is the entire thing that "
                "distinguishes MRF from generic quantitative parameter "
                "modelling (#347). The loss stayed finite and "
                "riemannian_mrf_dsm decreased smoothly the whole time, and "
                "MRFTangentScore's fingerprint encoder (fp_mamba / fp_proj) "
                "collected zero gradient for the entire run.\n"
                "No producer emits this key today: nothing under data/ writes "
                "'fingerprint', and the cluster's NIST-MRF in-vivo data is "
                "parameter maps (MAP_MASKED), not fingerprint time-series. So "
                "this raises rather than degrading -- if unconditional "
                "operation is wanted it should be a declared config mode, not "
                "an accident of a missing dict key."
            )
        self._declare_model_input(
            {"theta_noisy": theta_noisy, "sigma": sigma},
            extra={"model_input_key": "theta_noisy"},
        )
        score = self.generator_model(theta_noisy, sigma, fingerprint)
        target_score = -noise / sigma.view(-1, 1).clamp_min(1e-6)
        loss = F.mse_loss(score, target_score)
        return {
            "g_total_loss": loss,
            "riemannian_mrf_dsm": loss.detach(),
        }


__all__ = [
    "RiemannianMRFDiffusionStrategy",
    "SpatiotemporalMRFReconStrategy",
]
