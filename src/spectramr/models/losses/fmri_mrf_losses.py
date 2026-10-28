"""Losses for the fMRI + MRF 2026 plan
(``TODO/audit_plan_novel_fmri.md``).

Registered losses:
    * ``beltrami_epi_residual``   — fMRI §2 EPI susceptibility-distortion DC.
    * ``hrf_likelihood``          — fMRI §5 HRF prior coupled to BOLD GLM.
    * ``conformality_jacobian``   — MRF §3 Jacobian conformality regulariser.

References (matching the plan):
    [10] Jezzard & Balaban, *Magn. Reson. Med.*, 34(1), 1995.
    [15] Gardiner & Lakic, *Quasiconformal Teichmüller Theory*, AMS, 2000.
    [26] Friston et al., "Nonlinear responses in fMRI: balloon model",
         *NeuroImage*, 12(4), 2000.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.epi_forward import apply_epi_distortion
from spectramr.models.losses.registry import register_loss


@register_loss(
    "beltrami_epi_residual",
    domain="image",
    workflows=frozenset({Regime.FUNCTIONAL, Regime.DIFFUSION_WEIGHTED}),
    tasks=frozenset({Task.RECONSTRUCTION}),
)
class BeltramiEPIResidual(nn.Module):
    """Data-consistency residual for EPI distortion correction (fMRI §2).

    Caller supplies the undistorted target image and an estimated B0 field; the
    loss synthesises the distorted EPI via ``apply_epi_distortion`` and compares
    it to the observed EPI passed in as ``pred``, in L1.

    Tagged ``{mri_functional, mri_diffusion_weighted}`` — exactly the pairing
    ``BeltramiEPIDistortionStrategy`` carries, and for the same reason: EPI is
    the readout for those two regimes and no others. It models the READOUT, not
    the BOLD signal.

    Requires ``context['delta_b0']``; see :meth:`forward`.
    """

    def __init__(self, weight: float = 1.0, *, t_esp: float = 0.5e-3) -> None:
        super().__init__()
        self.weight = weight
        self.t_esp = t_esp

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        """L1 between the observed EPI and the target distorted by ``delta_b0``.

        Raises on a missing ``delta_b0`` rather than degrading. This previously
        fell back to ``F.l1_loss(pred, target)`` — plain L1, with no EPI physics
        in it at all — **while still reporting under the key
        ``beltrami_epi_residual``**. An arm whose headline method is EPI
        distortion correction would then train as a vanilla denoiser, log a term
        named after the physics it was not doing, and never warn (pitfalls #9,
        #16). This loss now backs mri_functional's LIVE claim in the maturity
        ledger, so the degraded path would have made that claim false at runtime.
        """
        delta_b0 = context.get("delta_b0")
        if not isinstance(delta_b0, torch.Tensor):
            raise ValueError(
                "beltrami_epi_residual requires context['delta_b0'] (the "
                f"estimated B0 field, [B, 1, H, W]); got {type(delta_b0)!r}. "
                "Without it there is no EPI distortion to be consistent with, "
                "and the term would silently become a plain L1 reported under "
                "the name of physics it is not doing."
            )
        simulated = apply_epi_distortion(target, delta_b0, t_esp=self.t_esp)
        residual = F.l1_loss(simulated, pred)
        return {
            "loss": self.weight * residual,
            "beltrami_epi_residual": residual,
        }


@register_loss("hrf_likelihood", domain="image")
class HRFLikelihoodLoss(nn.Module):
    """Couples an HRF prior to the BOLD GLM forward model (fMRI §5).

    Convolves the predicted HRF (``pred``) with a stimulus design
    matrix supplied via ``context['design_matrix']`` (shape ``[B, T]``)
    and matches the result to an observed BOLD time series ``target``
    in MSE.

    Deliberately NOT regime-tagged, though it is real BOLD physics when fed.
    ``mri_functional``'s maturity claim is scoped to the RECON family by its
    ``supported_tasks``, and a GLM likelihood is BOLD *analysis*, not
    reconstruction — tagging it would imply the framework models neural
    activity, which it does not. It is perfectly usable; it just does not back a
    maturity claim.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        """MSE between the GLM-predicted BOLD series and the observed one.

        Requires ``context['design_matrix']`` (the GLM path), or
        ``context['canonical_hrf']`` to grade the HRF shape directly. Raises when
        neither is present.

        The removed fallback was the worst defect in this module (issue #338):
        with no design matrix and no canonical HRF it computed
        ``F.mse_loss(pred, torch.zeros_like(pred))``. That is not a degraded GLM,
        it is a **shrink-the-HRF-to-zero penalty** — it actively optimised the
        haemodynamic response toward nothing while reporting an ordinary,
        steadily-decreasing number under the key ``hrf_likelihood``. It converged
        beautifully, on the opposite of the intended objective (pitfall #9).
        """
        design = context.get("design_matrix")
        if isinstance(design, torch.Tensor):
            if target is None or target.dim() < 2:
                raise ValueError(
                    "hrf_likelihood's GLM path needs an observed BOLD series "
                    f"target of shape [B, T]; got {type(target)!r} / "
                    f"{tuple(target.shape) if target is not None else '-'}."
                )
            if design.dim() != 2 or pred.dim() != 2:
                raise ValueError(
                    "hrf_likelihood's GLM path needs design_matrix [B, T] and an HRF "
                    f"prediction [B, T_hrf]; got {tuple(design.shape)} and "
                    f"{tuple(pred.shape)}."
                )
            n_batch = design.shape[0]
            hrf = pred.expand(n_batch, -1) if pred.shape[0] == 1 else pred
            if hrf.shape[0] != n_batch:
                raise ValueError(
                    f"HRF batch {pred.shape[0]} does not match design_matrix batch {n_batch}."
                )
            # Grouped conv convolves each sample's design with its OWN HRF; an
            # ungrouped conv1d crosses every design with every HRF and the result
            # then broadcasts against the target instead of raising.
            sim = F.conv1d(
                design.unsqueeze(0),
                hrf.flip(-1).unsqueeze(1),
                padding=hrf.shape[-1] - 1,
                groups=n_batch,
            ).squeeze(0)[..., : target.shape[-1]]
            if sim.shape != target.shape:
                raise ValueError(
                    f"GLM-predicted BOLD {tuple(sim.shape)} does not match the observed "
                    f"series {tuple(target.shape)}."
                )
            residual = F.mse_loss(sim, target)
        else:
            canonical = context.get("canonical_hrf")
            if not isinstance(canonical, torch.Tensor):
                raise ValueError(
                    "hrf_likelihood requires context['design_matrix'] (the [B, T] "
                    "stimulus design, for the GLM likelihood) or, failing that, "
                    "context['canonical_hrf'] to grade the HRF shape against. Got "
                    f"design_matrix={type(design)!r}, "
                    f"canonical_hrf={type(canonical)!r}. It previously defaulted "
                    "the reference to zeros, which trains the HRF toward zero."
                )
            residual = F.mse_loss(pred, canonical)
        return {
            "loss": self.weight * residual,
            "hrf_likelihood": residual,
        }


@register_loss("conformality_jacobian", domain="latent")
class ConformalityJacobianLoss(nn.Module):
    """Enforces conformality of a learned embedding (MRF §3).

    Conformality ⇔ all singular values of the Jacobian are equal ⇔
    ``‖J‖_F² = n·σ₁²``. Caller passes the Jacobian via
    ``context['jacobian']`` (shape ``[B, n, n]``); the loss returns
    the squared deviation from the conformality identity.

    Despite the "MRF §3" label this is pure differential geometry on an
    arbitrary Jacobian — no Bloch, no dictionary, no fingerprint — so it is
    deliberately NOT tagged ``mri_fingerprinting``. The regime's honest loss is
    ``mrf_dictionary_match``.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: object,
    ) -> dict[str, torch.Tensor]:
        """Squared deviation from ``‖J‖_F² = n·σ₁²``.

        Raises on a missing or misshaped Jacobian. It previously returned exactly
        ``0.0`` — a loss with no gradient, silently, forever. A zero term is
        indistinguishable in the logs from a *satisfied* constraint, so the
        regulariser could be absent for an entire run while appearing to be
        perfectly converged from step one (pitfall #9).
        """
        J = context.get("jacobian")
        if not isinstance(J, torch.Tensor) or J.dim() < 3:
            raise ValueError(
                "conformality_jacobian requires context['jacobian'] of shape "
                f"[B, n, n]; got {type(J)!r}"
                f"{'' if not isinstance(J, torch.Tensor) else f' with shape {tuple(J.shape)}'}"
                ". It previously returned a hard 0.0 here, which reads in the "
                "logs exactly like a satisfied constraint."
            )
        sv = torch.linalg.svdvals(J)
        n = sv.shape[-1]
        frob_sq = J.pow(2).sum(dim=(-1, -2))
        target_val = n * sv[..., 0].pow(2)
        residual = (frob_sq - target_val).pow(2).mean()
        return {
            "loss": self.weight * residual,
            "conformality_jacobian": residual,
        }


__all__ = [
    "BeltramiEPIResidual",
    "ConformalityJacobianLoss",
    "HRFLikelihoodLoss",
]
