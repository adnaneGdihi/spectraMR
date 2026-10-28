"""2-D non-line sampling kernels: variable-density Bernoulli (D4) and radial spokes (D5)."""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from spectramr.infrastructure.physics.degradations._common import (
    _calibrate_bernoulli_scale,
    _ensure_complex,
    _generator,
)
from spectramr.infrastructure.physics.fft_ops import centred_freqs, fft2c, ifft2c


def degrade_variable_density_2d(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,  # noqa: N803
    alpha: float = 3.0,
) -> torch.Tensor:
    r"""D4 (2-D) — Variable-density 2-D Bernoulli sampling.

    Sample probability decays as :math:`p(k) \propto (1 + \|k\|)^{-\alpha}`,
    **rescaled so that the expected kept fraction is exactly** :math:`1/R`. This
    is Bernoulli, not Poisson disk (no minimum-distance constraint). Appropriate
    for non-Cartesian acquisitions; for Cartesian see
    :func:`degrade_variable_density_cartesian`.

    M5 (digital-twin review): renamed from
    ``degrade_variable_density_undersampling`` to reflect the actual sampling
    pattern. The old name remains as an alias.

    Issue #246: the mask used to keep each sample with probability ``p / R``.
    Since :math:`\mathbb{E}[p] \approx 0.2` for the centre-peaked density, the
    *realised* acceleration was ``R / E[p]`` — a declared 1.0 -> 8.0x sweep in
    fact ran 6.0 -> 36.3x. The axis was consequently **saturated at its first
    step**: theta = 0.05 already delivered 72% of the theta = 1 distortion, and
    the usable dynamic range of the whole axis was 1.4x. The scale is now
    calibrated (:func:`_calibrate_bernoulli_scale`) so declared R == realised R.
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x
    H, W = x.shape[-2], x.shape[-1]
    device = x.device
    g = _generator(device, seed)

    # Normalised k in [-1, 1) on the fft2c grid, so the density peaks on the DC sample.
    ky = (2.0 * centred_freqs(H, device)).view(-1, 1)
    kx = (2.0 * centred_freqs(W, device)).view(1, -1)
    radius = torch.sqrt(kx**2 + ky**2)
    p = (1.0 + radius) ** (-alpha)
    p = p / p.max()

    # The 5x5 DC block is always acquired; it is part of the sampling budget,
    # so the Bernoulli target is corrected for it rather than added on top.
    n_dc = min(5, H) * min(5, W)
    target = max(0.0, (float(H * W) / R - n_dc) / float(H * W))
    p_keep = (_calibrate_bernoulli_scale(p, target) * p).clamp(max=1.0)

    rand = torch.rand(H, W, generator=g, device=device)
    keep = (rand < p_keep).float()

    cy, cx = H // 2, W // 2
    keep[cy - 2 : cy + 3, cx - 2 : cx + 3] = 1.0

    mask = keep.view(1, 1, H, W)
    return ifft2c(fft2c(x) * mask)


# Back-compatibility alias: the old name still resolves.
degrade_variable_density_undersampling = degrade_variable_density_2d


@lru_cache(maxsize=16)
def _radial_spoke_cover(
    H: int,
    W: int,
    seed: int,
    device_str: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Golden-angle spoke coverage of a Cartesian grid, ordered by spoke index.

    Returns ``(first, coverage)`` where ``first[i, j]`` is the index of the FIRST
    golden-angle spoke that covers cell ``(i, j)`` and ``coverage[n]`` is the
    fraction of the grid covered by spokes ``0..n``. Because golden-angle spoke
    *prefixes* are near-uniform, ``mask(n) = (first < n)`` is the mask of an
    ``n``-spoke acquisition, and ``coverage`` is monotone — which is what lets
    :func:`degrade_radial_undersampling` pick the spoke count that realises a
    requested acceleration.

    The spoke budget is ``ceil(2*pi*r_corner)``, twice the ``pi*r`` needed for
    gapless coverage out to the grid **corner** — so the full spoke set covers
    every cell and ``theta = 0`` is a true identity. Cached per grid+seed: the
    cover is a property of the trajectory, not of the image.
    """
    device = torch.device(device_str)
    g = _generator(device, seed)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    phase = torch.rand(1, generator=g, device=device).item() * 2 * math.pi

    r_corner = math.sqrt(H**2 + W**2) / 2.0
    n_cap = max(8, math.ceil(2.0 * math.pi * r_corner))

    ky = torch.arange(H, device=device, dtype=torch.float32) - H // 2
    kx = torch.arange(W, device=device, dtype=torch.float32) - W // 2
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # noqa: N806

    first = torch.full((H, W), n_cap, device=device, dtype=torch.long)
    for j in range(n_cap):
        ang = phase + j * golden
        # Distance to the line through the origin at this angle: a 1-px-wide spoke.
        d = (KX * math.sin(ang) - KY * math.cos(ang)).abs()
        first = torch.where(
            (d < 0.5) & (first == n_cap),
            torch.full_like(first, j),
            first,
        )

    counts = torch.bincount(first.flatten(), minlength=n_cap + 1).float()
    coverage = counts.cumsum(0) / float(H * W)  # coverage[n] = frac covered by 0..n
    return first, coverage


def degrade_radial_undersampling(
    x: torch.Tensor,
    theta: float,
    seed: int = 0,
    R_max: float = 8.0,  # noqa: N803
) -> torch.Tensor:
    """D5 — Radial spoke undersampling, Cartesian-grid approximation.

    Golden-angle spokes are laid on the Cartesian grid; cells under any spoke are
    kept. Simulates the streaking artefact of a true radial acquisition without
    requiring a NUFFT.

    The severity is the **acceleration**: ``R`` ramps 1.0 (theta=0) to ``R_max``
    (theta=1), and the spoke count is chosen so the realised kept fraction of
    k-space is ``1/R``. ``R = 1`` is a fully-sampled radial acquisition, i.e. the
    identity.

    Issue #246: the axis used to ramp a raw spoke *count* from 200 down to 16, and
    200 golden-angle spokes do **not** cover a 128^2 Cartesian grid — they leave
    ~32% of the cells (the corners the ``r < half`` disk cut discarded, plus the
    peripheral gaps between spokes) unsampled. ``D(x, theta=0)`` therefore came
    back with a rel-L2 of 0.069 against its own input, a pedestal larger than the
    *entire* theta=1 distortion of ``partial_fourier`` (0.088), silently
    contaminating every statistic anchored on the clean baseline.
    """
    x = _ensure_complex(x)
    R = 1.0 + (R_max - 1.0) * float(theta)
    if R <= 1.0:
        return x

    H, W = x.shape[-2], x.shape[-1]
    first, coverage = _radial_spoke_cover(H, W, int(seed), str(x.device))

    # Smallest spoke count whose coverage reaches the target sampling fraction.
    # NOTE: there is deliberately no `n_spokes_min` floor. The declared severity is
    # the ACCELERATION, and a spoke floor silently overrides it on small grids: at
    # 64^2 the old floor of 16 spokes forced 22.4% coverage where R=8 asks for
    # 12.5%, so the axis realised R=7.6 at theta=1 and stopped being monotone
    # between theta=0.5 and theta=1 (both clamped to the same 16-spoke mask).
    target = torch.as_tensor(1.0 / R, device=coverage.device, dtype=coverage.dtype)
    n_spokes = max(1, int(torch.searchsorted(coverage, target).item()) + 1)

    mask = (first < n_spokes).float().view(1, 1, H, W)
    return ifft2c(fft2c(x) * mask)
