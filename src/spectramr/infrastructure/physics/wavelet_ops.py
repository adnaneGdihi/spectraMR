r"""Orthonormal periodized 2-D wavelet transform, the sparsifying :math:`\Psi` of CS.

Compressed-sensing reconstruction penalises :math:`\lambda\lVert\Psi x\rVert_1`. When
:math:`\Psi` is orthonormal the proximal map of that penalty is exact and closed
form, :math:`\operatorname{prox}(v)=\Psi^H\,\mathcal S_\tau(\Psi v)`, which is why
this transform is periodized (circular boundary) rather than symmetric-padded: a
padded transform is redundant, so :math:`\Psi^H\Psi=I` but :math:`\Psi\Psi^H\neq I`
and the same formula would no longer be the proximal map.

The filters are real, so a complex image is transformed linearly in its real and
imaginary parts at once and the coefficients stay complex. The coefficient array
has the image's shape (the pywt ``coeffs_to_array`` layout): after :math:`J` levels
the coarsest approximation occupies the top-left
:math:`H/2^J\times W/2^J` block and every other entry is a detail coefficient.

The one-level Haar inside ``models/blocks/dual_domain_attention_kan.py`` is a
feature-map decomposition for attention, not this operator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

__all__ = ["WAVELET_FILTERS", "OrthogonalWavelet2D", "complex_soft_threshold"]

#: Orthonormal low-pass analysis filters (``pywt.Wavelet(name).dec_lo`` order).
WAVELET_FILTERS: Mapping[str, tuple[float, ...]] = {
    "haar": (1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)),
    "db4": (
        -0.010597401785069032,
        0.0328830116668852,
        0.030841381835560764,
        -0.18703481171909309,
        -0.027983769416859854,
        0.6308807679298589,
        0.7148465705529157,
        0.2303778133088965,
    ),
}


def complex_soft_threshold(z: torch.Tensor, threshold: float | torch.Tensor) -> torch.Tensor:
    r"""Exact proximal map of :math:`\tau\lVert z\rVert_1` for real or complex ``z``.

    :math:`\mathcal S_\tau(z)=\max(0,1-\tau/|z|)\,z`: the magnitude shrinks by
    :math:`\tau` and the phase is kept. Entries with :math:`|z|\le\tau` are exactly
    zero, with no :math:`\varepsilon` in the denominator to bias the survivors.
    """
    mag = z.abs()
    keep = mag > threshold
    scale = torch.where(keep, 1.0 - threshold / torch.where(keep, mag, torch.ones_like(mag)), 0.0)
    return z * scale


class OrthogonalWavelet2D:
    r"""Multi-level orthonormal 2-D DWT :math:`\Psi` with periodic boundaries.

    ``forward`` is :math:`\Psi`, ``inverse`` is :math:`\Psi^H=\Psi^{-1}`. Both act on
    the last two axes of a ``[..., H, W]`` real or complex tensor, which must be
    divisible by :math:`2^{\text{levels}}` and, at the coarsest level, no shorter
    than the filter.

    Raises:
        ValueError: on an unknown wavelet or a non-positive level count.
    """

    def __init__(self, wavelet: str = "db4", levels: int = 3) -> None:
        if wavelet not in WAVELET_FILTERS:
            raise ValueError(
                f"Unknown wavelet {wavelet!r}; the orthonormal filters available are "
                f"{sorted(WAVELET_FILTERS)}."
            )
        if isinstance(levels, bool) or not isinstance(levels, int) or levels < 1:
            raise ValueError(f"wavelet levels must be a positive integer, got {levels!r}.")
        self.wavelet = wavelet
        self.levels = levels
        # Correlating with rec_lo = reversed dec_lo, offset by 1 - L/2, reproduces
        # pywt's ``mode="periodization"`` coefficients exactly (tested).
        low = torch.tensor(WAVELET_FILTERS[wavelet], dtype=torch.float64).flip(0)
        n = torch.arange(low.numel())
        # Quadrature mirror: g[n] = (-1)^n h[L-1-n] makes the bank orthonormal.
        self._low = low
        self._high = ((-1.0) ** n) * low.flip(0)
        self._offset = 1 - low.numel() // 2

    def _check_shape(self, h: int, w: int) -> None:
        step = 2**self.levels
        taps = self._low.numel()
        if h % step or w % step or min(h, w) // (step // 2) < taps:
            raise ValueError(
                f"{self.wavelet} at {self.levels} level(s) needs H and W divisible by "
                f"{step} and a coarsest band of at least {taps} samples; got {h}x{w}."
            )

    def _filters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real = x.real.dtype if x.is_complex() else x.dtype
        return self._low.to(x.device, real), self._high.to(x.device, real)

    def _taps(self, n: int, taps: int, device: torch.device) -> torch.Tensor:
        # Circular index (2k + t + offset) mod n of output k under filter tap t.
        k = torch.arange(n // 2, device=device)[:, None]
        return (2 * k + torch.arange(taps, device=device)[None, :] + self._offset) % n

    def _analysis(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        low, high = self._filters(x)
        x = x.movedim(dim, -1)
        patches = x[..., self._taps(x.shape[-1], low.numel(), x.device)]
        out = torch.cat([(patches * low).sum(-1), (patches * high).sum(-1)], dim=-1)
        return out.movedim(-1, dim)

    def _synthesis(self, c: torch.Tensor, dim: int) -> torch.Tensor:
        low, high = self._filters(c)
        c = c.movedim(dim, -1)
        n = c.shape[-1]
        idx = self._taps(n, low.numel(), c.device)
        contrib = c[..., : n // 2, None] * low + c[..., n // 2 :, None] * high
        out = torch.zeros_like(c).index_add_(-1, idx.reshape(-1), contrib.flatten(-2))
        return out.movedim(-1, dim)

    def analyse_axis(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """One analysis level along one axis (low band first, then high), for N-D separable use."""
        if x.shape[dim] % 2:
            raise ValueError(f"axis {dim} has odd length {x.shape[dim]}")
        return self._analysis(x, dim)

    def synthesise_axis(self, c: torch.Tensor, dim: int) -> torch.Tensor:
        """Inverse of :meth:`analyse_axis`."""
        if c.shape[dim] % 2:
            raise ValueError(f"axis {dim} has odd length {c.shape[dim]}")
        return self._synthesis(c, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r""":math:`\Psi x`, in the image-shaped coefficient layout."""
        h, w = x.shape[-2:]
        self._check_shape(h, w)
        coeffs = x
        for level in range(self.levels):
            hh, ww = h >> level, w >> level
            band = self._analysis(self._analysis(coeffs[..., :hh, :ww], -1), -2)
            coeffs = coeffs.clone()
            coeffs[..., :hh, :ww] = band
        return coeffs

    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        r""":math:`\Psi^H c`, the exact inverse of :meth:`forward`."""
        h, w = coeffs.shape[-2:]
        self._check_shape(h, w)
        x = coeffs
        for level in reversed(range(self.levels)):
            hh, ww = h >> level, w >> level
            band = self._synthesis(self._synthesis(x[..., :hh, :ww], -2), -1)
            x = x.clone()
            x[..., :hh, :ww] = band
        return x

    def detail_mask(self, h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
        """``True`` on detail coefficients, ``False`` on the coarsest approximation."""
        mask = torch.ones(h, w, dtype=torch.bool, device=device)
        mask[: h >> self.levels, : w >> self.levels] = False
        return mask
