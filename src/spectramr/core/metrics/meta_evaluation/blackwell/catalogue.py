"""Fourier-diagonal degradation experiments and their per-coefficient information.

An experiment observes ``Y_k = h_k x_k + sigma n_k`` per centred k-space coefficient, with
``n_k ~ CN(0, 1)``. Its Fisher operator is diagonal with entries ``h_k^2 / sigma^2``, so the
whole experiment is its information profile. Profiles are held as ``log n_k`` in float64:
Gaussian blur drives ``h_k`` far below float32 range, and two blurs that both underflow to
zero would otherwise compare as equivalent. Noise is in units of the content's RMS, so a
profile is shared by every content and the relations between experiments do not depend on
image scale.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

#: Families of the default radial catalogue (cycle-6 corollary (v)).
FAMILIES = ("noise", "blur", "truncation", "blur_low_noise", "truncation_low_noise")


@dataclass(frozen=True)
class DiagonalExperiment:
    """One experiment: ``log_info[k] = log(|h_k|^2 / (sigma^2 |g_k|^2))`` at unit image RMS.

    The data are ``Y_k = h_k x_k + sigma g_k n_k``: ``h`` is the signal transfer and ``g``
    shapes the noise (``g = 1`` when an operator acts before the thermal noise, ``g = h``
    after it). Without explicit transfers the experiment is realised from its profile
    with unit gain at its most informative coefficient. ``-inf`` marks an unacquired
    coefficient. It is displayed as zero (zero-filling), which is Blackwell-equivalent to
    observing pure noise there.
    """

    name: str
    family: str
    severity: float
    log_info: torch.Tensor  # (H, W) float64
    signal_transfer: torch.Tensor | None = None  # (H, W) real, h
    noise_transfer: torch.Tensor | None = None  # (H, W) real, g
    sigma: float | None = None

    @property
    def noise_std(self) -> float:
        """``sigma`` at unit RMS; by default ``(max_k n_k)^{-1/2}``."""
        if self.sigma is not None:
            return self.sigma
        return math.exp(-0.5 * float(self.log_info.max()))

    def log_transfer(self) -> torch.Tensor:
        """``log h_k`` of the default realisation (unit gain at the most informative k)."""
        return 0.5 * (self.log_info - self.log_info.max())

    def transfers(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(h, g)`` as real float64 tensors, explicit or from the default realisation."""
        if self.signal_transfer is not None and self.noise_transfer is not None:
            return self.signal_transfer.double(), self.noise_transfer.double()
        log_h = self.log_transfer()
        acquired = torch.isfinite(log_h).double()
        return torch.exp(log_h.clamp_min(-700.0)) * acquired, acquired


def radial_frequency(shape: tuple[int, int], device: torch.device | None = None) -> torch.Tensor:
    """Radial spatial frequency in cycles per pixel on the centred grid (DC at ``N // 2``)."""
    h, w = shape
    fy = (torch.arange(h, dtype=torch.float64, device=device) - h // 2) / h
    fx = (torch.arange(w, dtype=torch.float64, device=device) - w // 2) / w
    return torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)


def info_groups(experiments: Sequence[DiagonalExperiment]) -> torch.Tensor:
    """Index ``(H, W)`` of each coefficient's group of identical profiles across ``experiments``.

    Two coefficients share a group exactly when every experiment has the same information
    at both, so every profile is constant on a group. For the radial catalogue the groups
    are the radius shells, merged where all profiles agree; a line mask gets its own.
    """
    stacked = torch.stack([e.log_info.reshape(-1) for e in experiments])
    _, inverse = torch.unique(stacked, dim=1, return_inverse=True)
    groups: torch.Tensor = inverse.reshape(experiments[0].log_info.shape)
    return groups


def gaussian_log_transfer(rho: torch.Tensor, width: float) -> torch.Tensor:
    """``log h`` of a Gaussian blur with pixel standard deviation ``width``."""
    return -2.0 * math.pi**2 * width**2 * rho**2


def disk_log_transfer(rho: torch.Tensor, cutoff: float) -> torch.Tensor:
    """``log h`` of ideal truncation to ``rho <= cutoff`` cycles per pixel."""
    out = torch.zeros_like(rho)
    out[rho > cutoff] = -math.inf
    return out


def _experiment(
    name: str, family: str, sev: float, log_h: torch.Tensor, sigma: float
) -> DiagonalExperiment:
    return DiagonalExperiment(name, family, float(sev), 2.0 * log_h - 2.0 * math.log(sigma))


def build_catalogue(
    shape: tuple[int, int],
    *,
    noise_levels: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.3),
    blur_widths: Sequence[float] = (0.5, 1.0, 1.5, 2.0, 3.0),
    cutoffs: Sequence[float] = (0.4, 0.3, 0.25, 0.2, 0.15),
    base_noise: float = 0.1,
    low_noise_factor: float = 0.5,
    device: torch.device | None = None,
) -> list[DiagonalExperiment]:
    """The five-family radial catalogue: noise, blur and truncation, the last two also at
    ``low_noise_factor * base_noise``. Blur and truncation act before the noise."""
    if base_noise <= 0 or not 0 < low_noise_factor < 1:
        raise ValueError("base_noise must be > 0 and low_noise_factor in (0, 1)")
    rho = radial_frequency(shape, device)
    flat = torch.zeros_like(rho)
    low = base_noise * low_noise_factor
    out = [_experiment(f"noise_{s:g}", "noise", s, flat, s) for s in noise_levels]
    for fam, sigma in (("blur", base_noise), ("blur_low_noise", low)):
        out += [
            _experiment(f"{fam}_{w:g}", fam, w, gaussian_log_transfer(rho, w), sigma)
            for w in blur_widths
        ]
    for fam, sigma in (("truncation", base_noise), ("truncation_low_noise", low)):
        out += [
            _experiment(f"{fam}_{c:g}", fam, c, disk_log_transfer(rho, c), sigma) for c in cutoffs
        ]
    return out


def mixture(a: DiagonalExperiment, b: DiagonalExperiment, lam: float = 0.5) -> DiagonalExperiment:
    """Time-sharing ``lam`` of the scan to ``a`` and the rest to ``b`` (Theorem C1.8(c)).

    Its information is ``lam n_a + (1 - lam) n_b``. The realisation of
    :meth:`DiagonalExperiment.log_transfer` presents it to a metric as one image.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}")
    log_info = torch.logaddexp(a.log_info + math.log(lam), b.log_info + math.log(1.0 - lam))
    return DiagonalExperiment(f"mix({a.name},{b.name})", "mixture", lam, log_info)


__all__ = [
    "FAMILIES",
    "DiagonalExperiment",
    "build_catalogue",
    "disk_log_transfer",
    "gaussian_log_transfer",
    "info_groups",
    "mixture",
    "radial_frequency",
]
