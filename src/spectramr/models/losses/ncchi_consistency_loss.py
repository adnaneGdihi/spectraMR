r"""Non-central-χ (Bessel-ratio) consistency loss (SOTA plan T4).

Penalises deviation of the model's magnitude from the non-central-χ **ML
magnitude** of the noisy measurement — the Bessel-ratio fixed point
``ν = m·R_L(mν/σ²)`` (see
:mod:`spectramr.infrastructure.physics.ncchi_data_consistency`). This is the exact
``L``-coil generalisation of the moment-based
:class:`~spectramr.models.losses.rician_consistency_loss.RicianConsistencyLoss`
(``√max(m²−2σ²,0)``, the high-SNR approximation at ``L=1``), and the
image/magnitude-domain analogue of the T1 GSURE k-space term.

Consumed by the EDM / Rician diffusion strategies as a noise-floor-debiasing
data-consistency term. Corner cases: ``L=1`` ⇒ Rician; high SNR ⇒ ML ≈ the
measurement (Gaussian consistency); below the noise floor ⇒ target → 0.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.ncchi_data_consistency import ncchi_ml_magnitude
from spectramr.models.losses.registry import register_loss


def _magnitude(x: torch.Tensor) -> torch.Tensor:
    """Real magnitude: complex.abs(), interleaved real/imag (C==2) → hypot, else abs."""
    if torch.is_complex(x):
        return x.abs()
    if x.shape[1] == 2:
        return torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2 + 1e-12)
    return x.abs()


@register_loss(
    name="ncchi_consistency",
    aliases=["NcChiConsistency", "ncchi_dc", "noncentral_chi_consistency"],
    domain="image",
)
class NcChiConsistencyLoss(nn.Module):
    r"""nc-χ ML-magnitude consistency loss (Bessel-ratio, ``L``-coil).

    Args:
        sigma: k-space / per-coil noise σ (> 0). Fail-closed (pitfall #15).
        n_coils: coil count ``L`` (≥ 1) for the nc-χ noise floor / Bessel order.
        n_iter: fixed-point iterations for the ML magnitude.
        norm: ``"l2"`` (default) or ``"l1"``.

    Forward: ``prediction`` is the model's reconstruction, ``target`` the noisy
    measurement; the loss is ``norm(|prediction|, ncchi_ML(|target|))``.
    """

    def __init__(
        self,
        sigma: float,
        n_coils: int = 1,
        n_iter: int = 20,
        norm: str = "l2",
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")
        if n_coils < 1:
            raise ValueError(f"n_coils must be >= 1, got {n_coils}")
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.sigma = float(sigma)
        self.n_coils = int(n_coils)
        self.n_iter = int(n_iter)
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        pred_mag = _magnitude(prediction)
        meas_mag = _magnitude(target)
        with torch.no_grad():
            ml_target = ncchi_ml_magnitude(
                meas_mag, self.sigma, n_coils=self.n_coils, n_iter=self.n_iter
            )
        if self.norm == "l1":
            return F.l1_loss(pred_mag, ml_target)
        return F.mse_loss(pred_mag, ml_target)


__all__ = ["NcChiConsistencyLoss"]
