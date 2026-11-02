"""Super-Nyquist fidelity: per-band transfer gain above the acquisition Nyquist.

PSNR and SSIM are dominated by the low frequencies every method already gets
right. Neither can tell whether the detail a super-resolution network adds was
*unfolded* from inter-frame aliasing or *invented* from a prior, because both
produce a sharp, plausible image.

This metric splits the comparison at the acquisition Nyquist (see
:mod:`spectramr.infrastructure.physics.band_partition`) and reports the transfer
gain

.. math::

    \\rho_\\ell = \\frac{\\langle P_\\ell, T_\\ell\\rangle}
                       {\\|P_\\ell\\|\\,\\|T_\\ell\\|}

per band, where :math:`P_\\ell, T_\\ell` are the band-``l`` components of the
prediction and the reference. Bands with ``rho > 1`` were never measured.

**What the number means depends entirely on what it is computed over, and the
two readings must not be conflated (pitfall #18).**

On the *anatomy* pair it is a descriptive statistic: a high super-Nyquist
:math:`\\rho_\\ell` on held-out anatomy could still come from a prior that
happens to match the population, since the reference was used to fit the
network's peers.

On a *known exogenous instrument* — the virtual fiducial, whose high-frequency
structure is fixed, anatomy-independent, and known exactly at every frequency —
it is a certificate. The instrument passes through the same decimation as the
anatomy, so :math:`\\rho_\\ell` above Nyquist measures how much unmeasured
detail the network genuinely recovers rather than fabricates. That is the use
:class:`~spectramr.infrastructure.training.strategies.multi_acquisition_strategy.ConcreteMultiAcquisitionStrategy`
puts it to.

References
----------
* R. Tsai, T. S. Huang, "Multiframe image restoration and registration,"
  *Advances in Computer Vision and Image Processing*, 1984. Sub-pixel-shifted
  frames make super-Nyquist content identifiable, which is what makes a
  non-zero transfer gain above Nyquist possible at all.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor

from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.band_partition import (
    PASSBAND_GEOMETRIES,
    acquisition_rho,
    band_edges,
    band_masks,
    band_transfer,
    super_nyquist_band_indices,
)


@register_metric(
    "super_nyquist_fidelity",
    aliases=["snf", "SuperNyquistFidelity"],
    direction="higher",
)
class SuperNyquistFidelity:
    """Mean band transfer gain above the acquisition Nyquist.

    Exactly one passband parameterisation must be given, matching
    :func:`~spectramr.infrastructure.physics.band_partition.acquisition_rho`:
    ``sr_scale`` for the synthetic decimation path, or ``voxel_mm`` plus
    ``effective_voxel_mm`` for real data where the stored grid is finer than
    what the scanner resolved.

    Example::

        snf = SuperNyquistFidelity(sr_scale=2)
        score = snf(pred_hr, true_hr)          # in [-1, 1], higher is better
        per_band = snf.spectrum(pred_hr, true_hr)   # {"snf_band_0": ..., ...}
    """

    def __init__(
        self,
        sr_scale: int | None = None,
        voxel_mm: tuple[float, ...] | None = None,
        effective_voxel_mm: tuple[float, ...] | None = None,
        n_sub_bands: int = 2,
        n_super_bands: int = 2,
        rho_max: float = 2.0,
        min_bins: int = 16,
        geometry: Literal["cartesian", "radial"] = "cartesian",
        **kwargs: Any,  # absorb device= etc. from InfrastructureBuilder
    ) -> None:
        """Initialise the partition.

        Args:
            sr_scale: Decimation factor (synthetic path).
            voxel_mm: Stored grid spacing per axis, mm (real-data path).
            effective_voxel_mm: Resolution the acquisition achieved, mm.
            n_sub_bands: Bands below the acquisition Nyquist.
            n_super_bands: Bands above it.
            rho_max: Outer edge in units of the acquisition Nyquist.
            min_bins: Smallest admissible band population; see
                :func:`~spectramr.infrastructure.physics.band_partition.band_masks`.
            geometry: Passband shape, ``"cartesian"`` (box) or ``"radial"``;
                see :func:`~spectramr.infrastructure.physics.band_partition.acquisition_rho`.
        """
        if sr_scale is None and effective_voxel_mm is None:
            raise ValueError(
                "SuperNyquistFidelity needs a passband: pass sr_scale, or "
                "voxel_mm + effective_voxel_mm. Without one there is no "
                "Nyquist boundary and 'super-Nyquist' is undefined."
            )
        if geometry not in PASSBAND_GEOMETRIES:
            raise ValueError(f"geometry must be one of {PASSBAND_GEOMETRIES}, got {geometry!r}")
        self._sr_scale = sr_scale
        self._voxel_mm = voxel_mm
        self._effective_voxel_mm = effective_voxel_mm
        self._edges = band_edges(n_sub_bands, n_super_bands, rho_max)
        self._super = super_nyquist_band_indices(self._edges)
        self._min_bins = min_bins
        self._geometry = geometry
        self._cache: dict[tuple[Any, ...], Tensor] = {}

    @property
    def name(self) -> str:
        """Metric canonical name."""
        return "super_nyquist_fidelity"

    @property
    def edges(self) -> tuple[float, ...]:
        """Band boundaries in units of the acquisition Nyquist."""
        return self._edges

    @property
    def super_nyquist_bands(self) -> tuple[int, ...]:
        """Indices of the bands lying entirely above the acquisition Nyquist."""
        return self._super

    def masks(self, reference: Tensor) -> Tensor:
        """Band masks for ``reference``'s grid, cached per (shape, device, dtype)."""
        size = tuple(reference.shape[2:])
        key = (size, str(reference.device), str(reference.dtype))
        cached = self._cache.get(key)
        if cached is None:
            rho = acquisition_rho(
                size,
                device=reference.device,
                dtype=(reference.dtype if reference.dtype.is_floating_point else torch.float32),
                sr_scale=self._sr_scale,
                voxel_mm=self._voxel_mm,
                effective_voxel_mm=self._effective_voxel_mm,
                geometry=self._geometry,
            )
            cached = band_masks(rho, self._edges, min_bins=self._min_bins)
            self._cache[key] = cached
        return cached

    def transfer(
        self,
        prediction: Tensor,
        target: Tensor,
        support: Tensor | None = None,
    ) -> Tensor:
        """Per-band transfer gain ``[B, L]``. Differentiable in ``prediction``.

        This is the seam the training-time probe uses; ``__call__`` is the
        detached, scalar reporting view of the same quantity.
        """
        return band_transfer(prediction, target, self.masks(target), support=support)

    def spectrum(
        self,
        prediction: Tensor,
        target: Tensor,
        support: Tensor | None = None,
        prefix: str = "snf",
    ) -> dict[str, float]:
        """Per-band gains as a flat logging dict, plus the sub/super summaries.

        Reporting every band rather than one number is deliberate: a single
        mean hides the shape of the rolloff, and the shape is the finding. A
        network that recovers the first super-Nyquist band and nothing beyond
        it is a very different result from one that recovers none.
        """
        with torch.no_grad():
            rho = self.transfer(prediction, target, support).mean(dim=0)
        out = {
            f"{prefix}_band_{i}_lo{self._edges[i]:.2f}": float(rho[i]) for i in range(rho.shape[0])
        }
        sub = [i for i in range(rho.shape[0]) if i not in self._super]
        if sub:
            out[f"{prefix}_sub_nyquist"] = float(rho[sub].mean())
        out[f"{prefix}_super_nyquist"] = float(rho[list(self._super)].mean())
        return out

    @torch.no_grad()  # type: ignore[misc]
    def __call__(
        self,
        prediction: Tensor,
        target: Tensor,
        support: Tensor | None = None,
        **kwargs: Any,
    ) -> float:
        """Mean transfer gain over the super-Nyquist bands.

        Args:
            prediction: ``[B, C, *spatial]`` model output.
            target: ``[B, C, *spatial]`` reference.
            support: Optional ``[B, 1, *spatial]`` weight restricting the
                comparison to where the reference is informative, e.g. the
                fiducial's own footprint.

        Returns:
            Scalar in ``[-1, 1]``; higher is better. ``1.0`` means the
            unmeasured bands are reproduced exactly up to a scale factor.
        """
        if torch.is_complex(prediction):
            prediction = prediction.abs()
        if torch.is_complex(target):
            target = target.abs()
        rho = self.transfer(prediction, target, support).mean(dim=0)
        return float(rho[list(self._super)].mean())


__all__ = ["SuperNyquistFidelity"]
