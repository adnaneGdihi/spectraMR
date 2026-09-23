r"""The coil null space as a measurement channel.

ESPIRiT eigendecomposes a per-pixel Gram matrix and keeps the leading
eigenvector as the sensitivity. The remaining :math:`C-k` eigenvectors span
directions no coil combination can produce. A physical multi-coil image is
rank one in coil space, :math:`x_q = s_q m_q`, so it has **exactly zero**
component there; thermal noise is full rank and does not.

That asymmetry makes the null space a free, per-pixel, pure-noise observation.
Measured on the committed M4Raw covariance, :math:`\Sigma_n` is full rank 4
with condition number 2.26, and for a rank-one signal direction roughly 75% of
the noise energy lands in the 3-D null space while none of the signal does.

The quantity this module exports is dimensionless:

.. math::

    E_q \;=\; \frac{\lVert P_q^{\perp} \hat{x}_q \rVert^2}
                   {\operatorname{tr}\!\left(P_q^{\perp}\Sigma_n\right)}

with :math:`P_q^{\perp}` the null projector from
:func:`~spectramr.infrastructure.physics.coil_sensitivity.estimate_csm_espirit`.
:math:`E \approx 1` is a correctly denoised pixel, :math:`> 1` is residual noise
or invented structure, :math:`< 1` is over-smoothing. The denominator is the
expectation under the null hypothesis, so the scale is derived rather than
tuned -- which is what separates this from a regularisation weight.

**Summarise with the MEAN, not the median.** The denominator is an expectation,
so only the mean is calibrated to 1. The null distribution is a scaled
chi-squared on :math:`2(C-k)` real degrees of freedom and is right-skewed:
measured over 40 pure-noise draws on a 4-coil phantom, the mean is
:math:`0.997 \pm 0.012` while the median is :math:`0.874 \pm 0.011`. Reporting
the median and reading 0.87 as over-smoothing is the mistake this paragraph
exists to prevent.

**Not to be confused with two other things this repository calls a null space.**
:mod:`spectramr.models.losses.null_space_loss` penalises the k-space *sampling*
null space :math:`(1-M)` and is supervised; ``ClinicalTrustAnalyzer.
compute_null_space_residual`` computes the data-consistency residual at
*measured* locations, which is the range space. This is the per-pixel COIL
subspace and needs no reference image.
"""

from __future__ import annotations

import torch

__all__ = [
    "coil_subspace_energy_budget",
    "coil_subspace_residual",
    "expected_null_residual",
    "scale_covariance",
]


def _as_coil_vectors(coil_images: torch.Tensor) -> torch.Tensor:
    """``(B, C, H, W)`` complex -> ``(B, H, W, C, 1)`` column vectors."""
    if not torch.is_complex(coil_images):
        raise TypeError(
            f"coil_subspace expects COMPLEX coil images, got {coil_images.dtype}. "
            "The rank-one argument holds in the linear complex domain; a magnitude "
            "or an RSS combination has already destroyed the coil vector."
        )
    if coil_images.ndim != 4:
        raise ValueError(f"expected (B, C, H, W), got {tuple(coil_images.shape)}.")
    return coil_images.permute(0, 2, 3, 1).unsqueeze(-1)


def coil_subspace_residual(coil_images: torch.Tensor, null_projector: torch.Tensor) -> torch.Tensor:
    r"""Per-pixel :math:`\lVert P^{\perp} x \rVert^2`.

    Args:
        coil_images: ``(B, C, H, W)`` complex, uncombined.
        null_projector: ``(B, H, W, C, C)`` from ``estimate_csm_espirit(...,
            return_subspace=True)``.

    Returns:
        ``(B, H, W)`` real, non-negative.
    """
    vectors = _as_coil_vectors(coil_images)
    if null_projector.shape[:3] != vectors.shape[:3]:
        raise ValueError(
            f"projector {tuple(null_projector.shape)} and images "
            f"{tuple(coil_images.shape)} disagree on (B, H, W)."
        )
    projected = null_projector.to(vectors.dtype) @ vectors
    return projected.squeeze(-1).abs().pow(2).sum(-1)


def expected_null_residual(
    null_projector: torch.Tensor, noise_covariance: torch.Tensor
) -> torch.Tensor:
    r"""Per-pixel :math:`\operatorname{tr}(P^{\perp}\Sigma_n P^{\perp})`.

    Evaluated as :math:`\operatorname{tr}(\Sigma_n P^{\perp})`, which is equal
    because :math:`P^{\perp}` is Hermitian and idempotent, and is one matrix
    product cheaper.

    This is the residual a *correct* reconstruction leaves: the null-space part
    of the noise it has not removed. It is the denominator that makes the budget
    calibrated rather than a tuned weight.

    Args:
        null_projector: ``(B, H, W, C, C)`` complex.
        noise_covariance: ``(C, C)`` complex Hermitian, in the SAME units as the
            images the residual is taken on. See :func:`scale_covariance`.

    Returns:
        ``(B, H, W)`` real, strictly positive wherever the projector has rank.
    """
    if noise_covariance.ndim != 2 or noise_covariance.shape[0] != noise_covariance.shape[1]:
        raise ValueError(
            f"expected a square (C, C) covariance, got {tuple(noise_covariance.shape)}."
        )
    n_coils = null_projector.shape[-1]
    if noise_covariance.shape[0] != n_coils:
        raise ValueError(
            f"covariance is {noise_covariance.shape[0]}-coil but the projector is "
            f"{n_coils}-coil. The identity holds per coil; mismatched coils mean the "
            "wrong noise model, not a broadcastable one."
        )
    sigma = noise_covariance.to(null_projector.dtype).to(null_projector.device)
    # trace(Sigma @ P) per pixel, without forming the product.
    return torch.einsum("ij,...ji->...", sigma, null_projector).real


def scale_covariance(noise_covariance: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    r"""Rescale :math:`\Sigma_n` to match normalized data.

    The committed M4Raw covariance is in raw scanner units. Normalization
    divides k-space by a per-subject scalar, and a covariance scales as the
    SQUARE of it -- getting this wrong by one power is the failure mode that
    leaves the budget systematically off by orders of magnitude while still
    looking like a plausible number.

    Pass ``scale`` as the divisor that was applied to the data
    (``subject["kspace_scale"]``); a run with normalization disabled passes 1.
    """
    factor = torch.as_tensor(scale, dtype=torch.float64)
    if torch.any(factor <= 0):
        raise ValueError(f"scale must be positive, got {scale!r}.")
    return noise_covariance / (factor.to(noise_covariance.dtype) ** 2)


def coil_subspace_energy_budget(
    coil_images: torch.Tensor,
    null_projector: torch.Tensor,
    noise_covariance: torch.Tensor,
    *,
    support: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""The dimensionless budget :math:`E_q`, per pixel.

    ``1`` is a correctly denoised pixel, ``> 1`` residual noise or invented
    structure, ``< 1`` over-smoothing.

    Args:
        coil_images: ``(B, C, H, W)`` complex, uncombined.
        null_projector: ``(B, H, W, C, C)`` complex.
        noise_covariance: ``(C, C)``, in the images' own units.
        support: optional ``(B, H, W)`` bool. Outside the ESPIRiT support no
            eigenvector clears the threshold, the projector is the identity, and
            the budget measures the whole coil vector against the whole noise
            power -- true, but a statement about background rather than about
            the reconstruction. Masked pixels return NaN so they cannot be
            averaged in silently.
        eps: floor on the denominator.

    Returns:
        ``(B, H, W)`` real.
    """
    residual = coil_subspace_residual(coil_images, null_projector)
    expected = expected_null_residual(null_projector, noise_covariance)
    budget = residual / expected.clamp(min=eps)
    if support is not None:
        if support.shape != budget.shape:
            raise ValueError(
                f"support {tuple(support.shape)} does not match the budget {tuple(budget.shape)}."
            )
        budget = budget.masked_fill(~support, float("nan"))
    return budget
