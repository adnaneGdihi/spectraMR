r"""Which phase-encode columns a coil-noise fit may use, and what that costs.

A null-space noise fit runs in the image domain, where every pixel carries
:math:`f\Sigma_n` of noise for a sampled phase-encode fraction :math:`f`. The
division by :math:`f` is exact for the noise under any pattern; the estimator's
rank-one premise is not. A contiguous band blurs the image with a kernel about
a pixel wide, across which smooth maps barely turn; scattered undersampling
aliases distant pixels onto each other and pushes signal into the null space
(``PAPER.md`` Proposition 14; 2-5x the true trace, measured).

Two policies, one owner. ``band`` refuses a support with a gap. ``acs`` fits a
non-band support on its contiguous run through the k-space centre and corrects
by that run's fraction :math:`f_{\mathrm{ACS}}`; the leak that remains is
:math:`O(\lVert\nabla\hat s\rVert/f_{\mathrm{ACS}})`, computed from the maps by
:func:`map_leak_bound` rather than assumed small. Both the per-scan calibration
(:mod:`scan_coil_calibration`) and the per-batch sigma
(:func:`~spectramr.infrastructure.physics.coil_noise_fit.sigma_from_kspace_batch`)
read their support through this module.

**The bound is on amplitude; the bias is on energy.** The signal the fit absorbs
is about (bound x signal/noise)^2 of the noise, so the ACS policy is accurate at
low field and not at high SNR. Measured on a textured 4-coil 256x256 phantom
with 24 ACS columns inside 25 % random sampling (bound 0.10): fitted-to-true
trace 1.003 (0.991-1.009, 4 noise draws) at the committed M4Raw level, against
2.6 at ten times the signal, where the whole random support reads 4.4.
"""

from __future__ import annotations

import torch

__all__ = [
    "VALID_SUPPORT_POLICIES",
    "NonBandSupportError",
    "acs_band",
    "batch_fit_columns",
    "fit_support",
    "map_leak_bound",
    "pe_band_fraction",
    "pe_support_of",
]

#: ``band`` refuses any support that is not one contiguous band; ``acs`` fits a
#: non-band support on its central contiguous run and corrects by its fraction.
VALID_SUPPORT_POLICIES: frozenset[str] = frozenset({"band", "acs"})


class NonBandSupportError(ValueError):
    """The sampled phase-encode columns do not form one contiguous band."""


def _check_mask(support: torch.Tensor) -> None:
    if support.dim() != 1 or support.dtype != torch.bool:
        raise TypeError(
            f"support must be a 1-D bool mask over phase-encode columns, got "
            f"{support.dtype} {tuple(support.shape)}."
        )


def pe_support_of(kspace: torch.Tensor) -> torch.Tensor:
    """1-D bool over the last (phase-encode) axis: columns carrying any energy."""
    return kspace.abs().sum(dim=tuple(range(kspace.dim() - 1))) > 0


def pe_band_fraction(support: torch.Tensor) -> float:
    """Sampled fraction of a phase-encode support that must be one contiguous band.

    Raises:
        NonBandSupportError: no column is sampled, or the sampled columns have
            a gap. See the module docstring for why a gap is not a smaller band.
    """
    _check_mask(support)
    sampled = torch.nonzero(support).flatten()
    if sampled.numel() == 0:
        raise NonBandSupportError("no phase-encode column is sampled.")
    first, last = int(sampled[0]), int(sampled[-1])
    if last - first + 1 != sampled.numel():
        raise NonBandSupportError(
            f"{sampled.numel()} sampled phase-encode columns span [{first}, {last}] "
            "with gaps. The per-scan calibration is exact for one contiguous band "
            "(partial phase FOV) and not for scattered undersampling, where aliased "
            "signal enters the coil null space and the fitted Sigma_n comes out "
            "high by an SNR-dependent factor (2-5x measured). support_policy='acs' "
            "fits the central auto-calibration run instead."
        )
    return sampled.numel() / support.numel()


def acs_band(support: torch.Tensor) -> tuple[int, int]:
    """``[first, last]`` of the contiguous sampled run through the centre column.

    Raises:
        NonBandSupportError: the centre column was not acquired, so there is no
            auto-calibration run to restrict a fit to.
    """
    _check_mask(support)
    sampled = support.tolist()
    centre = len(sampled) // 2
    if not sampled[centre]:
        raise NonBandSupportError(
            f"the centre phase-encode column {centre} is not sampled, so the support "
            "has no auto-calibration run to fit on."
        )
    first, last = centre, centre
    while first > 0 and sampled[first - 1]:
        first -= 1
    while last < len(sampled) - 1 and sampled[last + 1]:
        last += 1
    return first, last


def fit_support(
    support: torch.Tensor, policy: str, *, min_width: int = 1
) -> tuple[torch.Tensor, float, bool]:
    """``(columns_to_fit, their_fraction, restricted_to_acs)`` under ``policy``.

    A band is fitted whole under either policy (Theorem 13 makes it exact).

    Args:
        support: 1-D bool over phase-encode columns.
        policy: one of :data:`VALID_SUPPORT_POLICIES`.
        min_width: the narrowest central run ``acs`` accepts -- the ESPIRiT
            calibration window, when the same run also calibrates the maps.

    Raises:
        ValueError: unknown policy.
        NonBandSupportError: a gap under ``band``; no central run at least
            ``min_width`` wide under ``acs``.
    """
    if policy not in VALID_SUPPORT_POLICIES:
        raise ValueError(f"support_policy={policy!r}; valid: {sorted(VALID_SUPPORT_POLICIES)}.")
    try:
        return support, pe_band_fraction(support), False
    except NonBandSupportError:
        if policy == "band":
            raise
    first, last = acs_band(support)
    width = last - first + 1
    if width < min_width:
        raise NonBandSupportError(
            f"the central sampled run is {width} columns wide, narrower than the "
            f"{min_width} columns required (the ESPIRiT calibration window would "
            "reach zero-filled columns)."
        )
    acs = torch.zeros_like(support)
    acs[first : last + 1] = True
    return acs, width / support.numel(), True


def map_leak_bound(maps: torch.Tensor, fit_fraction: float, *, eps: float = 1e-6) -> float:
    r"""The Proposition 14 leak scale, :math:`\overline{\lVert\partial_{PE}\hat s\rVert}/f`.

    The mean, over pixels where both neighbours carry a map, of how far the
    unit coil direction turns between adjacent phase-encode columns, times the
    Dirichlet kernel's width :math:`1/f` in pixels. Dimensionless: roughly the
    fraction of the signal amplitude the kernel can rotate into the null space.

    Args:
        maps: ``(B, C, H, W)`` complex coil maps.
        fit_fraction: the fraction of phase-encode columns the fit used.
    """
    norm = maps.abs().pow(2).sum(1, keepdim=True).sqrt()
    inside = (norm > eps).squeeze(1)
    unit = maps / norm.clamp(min=eps)
    # A unit coil vector is defined up to a phase; remove the relative phase of
    # the two neighbours so only the geometric turn is measured.
    left, right = unit[..., :-1], unit[..., 1:]
    overlap = (left.conj() * right).sum(1, keepdim=True)
    right = right * torch.exp(-1j * torch.angle(overlap))
    step = (right - left).abs().pow(2).sum(1).sqrt()
    both = inside[..., :-1] & inside[..., 1:]
    if not bool(both.any()):
        raise ValueError("the maps have no two adjacent in-support pixels to difference.")
    return float(step[both].mean()) / fit_fraction


def _sampling_bins(kspace: torch.Tensor, acquired: torch.Tensor | None) -> torch.Tensor:
    """``(B, H, W)`` bool: which k-space samples were acquired.

    From ``acquired`` when the caller holds the mask, else read off the data. A data read
    cannot tell an unacquired sample from an acquired one that happens to be exactly zero
    (a sample is zero in every coil only by accident), and the hole guard below then
    mistakes that accident for a 2-D mask. It tripped the ``robust_ssdu_selfcal`` arm at
    iteration 905 of an otherwise healthy run.
    """
    if acquired is None:
        return kspace.abs().flatten(1, -3).sum(1) > 0
    sampled = acquired if acquired.dtype == torch.bool else acquired.abs() > 0
    bins = sampled.flatten(1, -3).any(1)
    return bins.expand(kspace.shape[0], kspace.shape[-2], kspace.shape[-1])


def batch_fit_columns(
    kspace: torch.Tensor, policy: str, *, acquired: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`fit_support` per sample, on the device, for a ``(B, C, H, W)`` batch.

    The support is the declared ``acquired`` mask when one is given, else it is read
    off the data (a column with no energy was not acquired); either way each sample's
    mask is its own. The arithmetic stays on the device; refusing a sample costs one
    host read of a ``(B,)`` flag, the same order as the ``float(sigma)`` the per-step
    caller already performs.

    Args:
        kspace: ``(B, C, H, W)`` complex measurement.
        policy: one of :data:`VALID_SUPPORT_POLICIES`.
        acquired: optional ``(B, 1 or C, H or 1, W)`` sampling mask, any dtype. A 2-D
            mask is still refused; an isolated exact zero inside an acquired line no
            longer is.

    Returns:
        ``(columns, fraction)``: ``(B, W)`` bool and ``(B,)`` float.

    Raises:
        NonBandSupportError: under ``band`` a sample has a gap; under ``acs`` a
            non-band sample's centre column was not acquired.
    """
    if policy not in VALID_SUPPORT_POLICIES:
        raise ValueError(f"support_policy={policy!r}; valid: {sorted(VALID_SUPPORT_POLICIES)}.")
    bins = _sampling_bins(kspace, acquired)
    support = bins.any(-2)
    width_total = support.shape[-1]
    # A mask that runs along the rows leaves every column with energy, so the
    # column fit would see a fully sampled band and skip the sampled-fraction
    # correction: sigma would come out low by sqrt(f). Refuse it by name.
    rows = bins.any(-1)
    rows_missing = rows.sum(-1) < rows.shape[-1]
    if bool((rows_missing & (support.sum(-1) == width_total)).any()):
        raise NonBandSupportError(
            "the sampled lines run along the first spatial axis while the fit reads "
            "phase-encode columns on the last one; declare undersampling.mask_direction: x "
            "so the Cartesian mask runs along M4Raw's phase-encode axis."
        )
    # A 2-D mask (Poisson-disc, pseudo-radial) leaves holes inside acquired
    # columns. The column fit would count those columns whole and skip the 1/f
    # correction on an aliased support, so it is refused rather than mis-fitted.
    holes = ~bins & support[:, None, :] & bins.any(-1, keepdim=True)
    if bool(holes.any()):
        raise NonBandSupportError(
            "the sampled k-space is not whole phase-encode lines (a 2-D mask): an acquired "
            "column has unsampled rows, and the column fit has no sampled fraction for it."
        )
    idx = torch.arange(width_total, device=support.device)
    count = support.sum(-1)
    first = torch.where(support, idx, width_total).amin(-1)
    last = torch.where(support, idx, -1).amax(-1)
    is_band = (count > 0) & (last - first + 1 == count)
    if policy == "band":
        if not bool(is_band.all()):
            raise NonBandSupportError(
                "a sample's sampled phase-encode columns have gaps; support_policy='acs' "
                "fits the central auto-calibration run instead."
            )
        return support, count / width_total
    centre = width_total // 2
    sampled = support.to(torch.int32)
    right = torch.cumprod(sampled[:, centre:], dim=-1).sum(-1)
    left = torch.cumprod(sampled[:, : centre + 1].flip(-1), dim=-1).sum(-1)
    acs = (idx >= (centre - left + 1)[:, None]) & (idx <= (centre + right - 1)[:, None])
    columns = torch.where(is_band[:, None], support, acs)
    fraction = torch.where(is_band, count, (left + right - 1).clamp(min=0)) / width_total
    if not bool((fraction > 0).all()):
        raise NonBandSupportError(
            "a non-band sample does not acquire the centre phase-encode column, so it has "
            "no auto-calibration run to fit on."
        )
    return columns, fraction
