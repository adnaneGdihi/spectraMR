r"""Calibrate one scan's coil geometry and noise from that scan alone.

Two consumers need the same per-scan quantities from a single repetition: the
self-calibrated R2R draw needs :math:`\Sigma_n`, and the null-space split
(:mod:`coil_split_r2r`) needs the ESPIRiT maps and, for whitening, the same
:math:`\Sigma_n`. Both come out of one ESPIRiT eigendecomposition, so this is
the one place that runs it and applies the two corrections a real M4Raw volume
requires.

**The slice axis is the estimator's batch axis.** The per-group volume arrives
as ``(S, C, H, W)``; flattening it to ``(1, S*C, H, W)`` hands ESPIRiT ``S*C``
"coils" and a calibration matrix it rightly calls rank-deficient (#2322).

**The sampled fraction.** M4Raw zero-fills ~24% of its phase-encode columns
(partial phase FOV). The null-space fit runs in the image domain, where by
Parseval under ``norm="ortho"`` every pixel carries :math:`f\Sigma_n` of noise
for a sampled fraction :math:`f` -- so the raw fit is low by exactly
:math:`f` (0.758 measured at :math:`f=0.76`) and is divided by it here.

That division is exact for the *noise* under any sampling pattern. What does
not survive every pattern is the estimator's premise, a rank-one signal: a
contiguous band blurs the image along the phase-encode axis with a kernel about
a pixel wide, across which the smooth sensitivities barely change, while
scattered undersampling aliases distant pixels -- with different sensitivities
-- onto each other and pushes signal into the null space. That bias grows with
SNR: on a textured synthetic 4-coil phantom it is absent at the committed noise
level and 2.1x at ten times the signal, against 1.05x for a band; #2322 reports
5-5.3x. The support is therefore required to be one contiguous band by default
(``support_policy="band"``), and anything else raises rather than returning a
covariance that is quietly several times wrong.

**The ACS policy** (``support_policy="acs"``, ``PAPER.md`` Proposition 14) is
the opt-in for undersampled acquisitions. A non-band support is cut down to the
contiguous run of sampled columns through the k-space centre, every other
column is zeroed, and the fit is corrected by that run's fraction
:math:`f_{\mathrm{ACS}}`. The point-spread function is then a Dirichlet kernel
about :math:`1/f_{\mathrm{ACS}}` pixels wide along phase encode, so the signal
the fit absorbs is bounded by how far the unit maps turn across it,
:math:`O(\lVert\nabla\hat s\rVert/f_{\mathrm{ACS}})`. That bound is computed
from the maps (:func:`map_leak_bound`) and logged with every ACS fit rather than
assumed small.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from spectramr.infrastructure.physics.coil_noise_fit import estimate_covariance_from_nullspace
from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.pe_support import (
    VALID_SUPPORT_POLICIES,
    NonBandSupportError,
    fit_support,
    map_leak_bound,
    pe_band_fraction,
)

__all__ = [
    "DEFAULT_CALIBRATION",
    "VALID_SUPPORT_POLICIES",
    "NonBandSupportError",
    "ScanCoilCalibration",
    "as_slice_batch",
    "calibrate_scan",
    "fit_scan_covariance",
    "pe_band_fraction",
]

logger = logging.getLogger(__name__)

#: The ESPIRiT settings every per-scan consumer uses unless it declares its own.
#: One dict, so the precomputed-map store validates files against the values
#: the loader would have estimated with.
DEFAULT_CALIBRATION: dict[str, float | int] = {
    "acs_size": 24,
    "kernel_size": 6,
    "eigen_threshold": 0.95,
}


@dataclass(frozen=True)
class ScanCoilCalibration:
    """What one scan says about its own receive chain.

    Attributes:
        maps: ``(S, C, H, W)`` complex ESPIRiT soft-SENSE maps, one per slice.
        covariance: ``(C, C)`` complex Hermitian :math:`\\Sigma_n` of ONE
            repetition in raw k-space units, corrected for the fraction it was
            fitted on; ``None`` when the caller did not ask for it.
        sampled_fraction: fraction of phase-encode columns that were acquired.
        fit_fraction: the fraction the covariance was divided by -- the band's
            under ``band``, the central run's under ``acs`` on a non-band support.
        support_policy: the policy that produced the fit.
        leak_bound: :func:`map_leak_bound` when the fit was restricted to the
            ACS run, else ``None`` (a band needs no bound, Theorem 13).
    """

    maps: torch.Tensor
    covariance: torch.Tensor | None
    sampled_fraction: float
    fit_fraction: float | None = None
    support_policy: str = "band"
    leak_bound: float | None = None


def as_slice_batch(kspace: torch.Tensor) -> torch.Tensor:
    """``(C, H, W)`` or ``(S, C, H, W)`` complex -> ``(S, C, H, W)``.

    Anything else raises: a 5-D tensor or a combined ``(S, H, W)`` volume has no
    coil axis in the place ESPIRiT reads it from, and guessing one is how the
    72-coil calibration of #2322 happened.
    """
    if not torch.is_complex(kspace):
        raise TypeError(f"expected complex k-space, got {kspace.dtype}.")
    if kspace.dim() == 3:
        return kspace.unsqueeze(0)
    if kspace.dim() == 4:
        return kspace
    raise ValueError(f"expected (C, H, W) or (S, C, H, W) coil k-space, got {tuple(kspace.shape)}.")


def calibrate_scan(
    kspace: torch.Tensor,
    support: torch.Tensor,
    *,
    fit_covariance: bool,
    support_policy: str = "band",
    acs_size: int = int(DEFAULT_CALIBRATION["acs_size"]),
    kernel_size: int = int(DEFAULT_CALIBRATION["kernel_size"]),
    eigen_threshold: float = float(DEFAULT_CALIBRATION["eigen_threshold"]),
) -> ScanCoilCalibration:
    """ESPIRiT maps for every slice and, optionally, the scan's :math:`\\Sigma_n`.

    Args:
        kspace: raw coil k-space, ``(C, H, W)`` or ``(S, C, H, W)``, with the
            phase-encode axis last. Raw, not normalized: the covariance is
            returned in the units the R2R draw is injected in.
        support: 1-D bool over phase-encode columns.
        fit_covariance: also fit :math:`\\Sigma_n` from the coil null space.
        support_policy: ``band`` (default) refuses a non-band support; ``acs``
            fits it on the central sampled run and corrects by that run's
            fraction, logging :func:`map_leak_bound`. A band support is fitted
            whole under either policy.
        acs_size, kernel_size, eigen_threshold: ESPIRiT calibration, as in
            :func:`estimate_csm_espirit`.

    Returns:
        The :class:`ScanCoilCalibration`.

    Raises:
        NonBandSupportError: the support is not a band under ``band``, or has no
            central run at least ``acs_size`` wide under ``acs``.
        UnidentifiableCovarianceError: the null spaces do not determine
            :math:`\\Sigma_n` (propagated from the fit, never replaced by a
            constant).
    """
    slices = as_slice_batch(kspace)
    if support.numel() != slices.shape[-1]:
        raise ValueError(
            f"support covers {support.numel()} phase-encode columns, k-space has "
            f"{slices.shape[-1]}."
        )
    fit_columns, fit_fraction, restricted = fit_support(support, support_policy, min_width=acs_size)
    sampled_fraction = float(support.float().mean())
    result = estimate_csm_espirit(
        slices,
        num_coils=slices.shape[1],
        kernel_size=kernel_size,
        acs_size=acs_size,
        eigen_threshold=eigen_threshold,
        return_subspace=fit_covariance,
    )
    if not fit_covariance:
        return ScanCoilCalibration(
            maps=result,
            covariance=None,
            sampled_fraction=sampled_fraction,
            fit_fraction=fit_fraction,
            support_policy=support_policy,
        )
    fitted = slices * fit_columns.to(slices.device) if restricted else slices
    raw = estimate_covariance_from_nullspace(ifft2c(fitted), result.null_projector)
    leak = map_leak_bound(result.maps, fit_fraction) if restricted else None
    if restricted:
        logger.info(
            "[calibration] ACS-restricted Sigma_n fit: %d of %d columns (f_ACS=%.4f), "
            "leak bound mean|d s_hat|/f_ACS = %.4g; absorbed signal scales as "
            "(bound x SNR)^2, so trust the fit at low SNR only (pe_support.py).",
            int(fit_columns.sum()),
            support.numel(),
            fit_fraction,
            leak,
        )
    return ScanCoilCalibration(
        maps=result.maps,
        covariance=raw / fit_fraction,
        sampled_fraction=sampled_fraction,
        fit_fraction=fit_fraction,
        support_policy=support_policy,
        leak_bound=leak,
    )


def fit_scan_covariance(
    kspace: torch.Tensor, support: torch.Tensor, *, support_policy: str = "band"
) -> torch.Tensor:
    """:math:`\\Sigma_n` of one repetition, from :func:`calibrate_scan`, never ``None``."""
    covariance = calibrate_scan(
        kspace, support, fit_covariance=True, support_policy=support_policy
    ).covariance
    if covariance is None:  # pragma: no cover - fit_covariance=True always fits
        raise RuntimeError("calibrate_scan(fit_covariance=True) returned no covariance.")
    return covariance
