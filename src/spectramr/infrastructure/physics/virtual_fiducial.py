"""Virtual Fiducial for Motion Estimation.

A learnable Gaussian grid probe that isolates motion artifacts from anatomical
content. When corrupted by the same motion as the patient scan, the fiducial
reveals the pure geometric distortion pattern without confounding brain structure.

The Virtual Fiducial is used in two phases:
1. **Meta-Training**: Random motion θ corrupts both clean anatomy and fiducial;
   the HyperMamba learns to invert motion from the fiducial's distortion pattern.
2. **TTO**: θ̂ is optimized so the fiducial's predicted corruption matches
   the measured data consistency loss.

Reference:
    Ha et al., "Hypernetworks," ICLR 2017.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VirtualFiducial(nn.Module):
    """Learnable Gaussian grid probe for motion estimation.

    Generates a 2D grid of evenly-spaced Gaussian peaks in complex-valued
    image space. The grid serves as a motion-sensitive "barcode" — any
    rigid-body motion during k-space acquisition creates characteristic
    distortions (ghosts, ringing) that are easy for a CNN to decode.

    Args:
        im_size: Image dimensions (H, W).
        grid_spacing: Spacing between Gaussian peaks in pixels.
        sigma: Standard deviation of each Gaussian peak.
        learnable: If True, the grid parameters are trainable.

    Example:
        >>> fiducial = VirtualFiducial(im_size=(256, 256))
        >>> M = fiducial()  # [1, 1, 256, 256] complex64
        >>> # Corrupt with motion:
        >>> corrupted = kinematic_op(M, theta)
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        grid_spacing: int = 16,
        sigma: float = 2.0,
        learnable: bool = False,
        jitter: float = 0.0,
        seed: int = 0,
        voxel_mm: tuple[float, ...] | None = None,
        effective_voxel_mm: tuple[float, ...] | None = None,
        spacing_mm: tuple[float, ...] | float | None = None,
        sigma_mm: tuple[float, ...] | float | None = None,
        kappa: float = 1.0,
    ) -> None:
        """Initialize Virtual Fiducial.

        Args:
            im_size: Image dimensions (H, W).
            grid_spacing: Pixel spacing between Gaussian centers.
            sigma: Width of each Gaussian peak.
            learnable: Whether grid parameters should be trainable.
            jitter: Peak displacement, in units of ``grid_spacing``, drawn once
                from ``U(-jitter, jitter)`` per peak. ``0.0`` keeps the exactly
                periodic lattice. **Registration needs this above zero** (see the
                class docstring): a periodic lattice has a comb spectrum, so
                phase-correlation has almost no populated adjacent frequency bins
                to read a shift from, and the lattice is in any case ambiguous
                modulo one period under translation. ``0.35`` puts the spectral
                occupancy above 25% without peaks colliding.
            seed: Draws the jitter deterministically, so the fiducial is a fixed
                known pattern across runs and processes and can serve as an
                absolute registration reference.
        """
        super().__init__()
        self.im_size = im_size
        self.grid_spacing = grid_spacing
        self.jitter = jitter

        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")

        ndim = len(im_size)
        if voxel_mm is None:
            # Legacy pixel mode: isotropic, unchanged behaviour.
            spacing_px: tuple[float, ...] = (float(grid_spacing),) * ndim
            sigma_px: tuple[float, ...] = (float(sigma),) * ndim
            self.voxel_mm = None
            self.effective_voxel_mm = None
            self.sigma_mm = None
        else:
            spacing_px, sigma_px = self._physical_to_pixels(
                ndim=ndim,
                voxel_mm=voxel_mm,
                effective_voxel_mm=effective_voxel_mm,
                spacing_mm=spacing_mm,
                sigma_mm=sigma_mm,
                kappa=kappa,
            )
            self.voxel_mm = tuple(float(v) for v in voxel_mm)
            self.effective_voxel_mm = tuple(float(v) for v in (effective_voxel_mm or voxel_mm))
            self.sigma_mm = tuple(s * v for s, v in zip(sigma_px, self.voxel_mm, strict=True))

        self.spacing_px = spacing_px
        self.sigma_px = sigma_px

        # Build the base fiducial grid
        base_grid = self._build_gaussian_grid(im_size, spacing_px, sigma_px, jitter, seed)

        if learnable:
            self.grid = nn.Parameter(base_grid)
        else:
            self.register_buffer("grid", base_grid)

        n_peaks = 1
        for n, sp in zip(im_size, spacing_px, strict=True):
            n_peaks *= max(1, int(n / sp))
        logger.info(
            "[VirtualFiducial] im_size=%s spacing_px=%s sigma_px=%s "
            "sigma_mm=%s effective_voxel_mm=%s jitter=%.2f learnable=%s peaks=%d",
            im_size,
            tuple(round(v, 2) for v in spacing_px),
            tuple(round(v, 2) for v in sigma_px),
            self.sigma_mm,
            self.effective_voxel_mm,
            jitter,
            learnable,
            n_peaks,
        )

    @staticmethod
    def _physical_to_pixels(
        *,
        ndim: int,
        voxel_mm: tuple[float, ...],
        effective_voxel_mm: tuple[float, ...] | None,
        spacing_mm: tuple[float, ...] | float | None,
        sigma_mm: tuple[float, ...] | float | None,
        kappa: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Convert millimetre marker geometry to per-axis pixel widths.

        ``voxel_mm`` is the SAMPLING grid; ``effective_voxel_mm`` is the
        resolution the marker must remain visible at. These differ whenever a
        volume has been resampled onto a finer grid than the scanner resolved,
        which is exactly the ULF→HF case: ``preprocess_ulf_paired.py`` puts the
        64mT volume on the 3T grid, so ``voxel_mm`` is 0.22-0.49 mm while the
        64mT scanner actually resolved 1.6-1.7 mm in-plane. Sizing the marker
        against the grid rather than the effective resolution makes it 3-7x too
        fine to survive in the ULF channel.

        ``sigma_mm`` defaults to ``kappa * effective_voxel_mm``. The Cramer-Rao
        bound for a translation estimate scales as the inverse of the marker's
        mean-square frequency, so narrower atoms register better right up to the
        point where energy crosses Nyquist and the sampled marker stops being a
        translate of a fixed template. ``kappa = 1`` sits at that boundary.
        """

        def _per_axis(v, name: str) -> tuple[float, ...]:
            if isinstance(v, (int, float)):
                return (float(v),) * ndim
            out = tuple(float(x) for x in v)
            if len(out) != ndim:
                raise ValueError(f"{name} has {len(out)} entries but im_size is {ndim}-D")
            return out

        vox = _per_axis(voxel_mm, "voxel_mm")
        eff = _per_axis(effective_voxel_mm, "effective_voxel_mm") if effective_voxel_mm else vox
        if any(v <= 0 for v in vox) or any(v <= 0 for v in eff):
            raise ValueError(f"voxel sizes must be > 0, got {vox} / {eff}")
        if any(e < v for e, v in zip(eff, vox, strict=True)):
            raise ValueError(
                f"effective_voxel_mm {eff} is finer than the sampling grid "
                f"{vox} on some axis. The effective resolution is what the "
                "scanner resolved; it cannot be finer than the grid it is "
                "stored on."
            )

        sig = (
            _per_axis(sigma_mm, "sigma_mm")
            if sigma_mm is not None
            else tuple(kappa * e for e in eff)
        )
        if any(s < e for s, e in zip(sig, eff, strict=True)):
            raise ValueError(
                f"sigma_mm {sig} is below the effective voxel size {eff} on some "
                "axis. Such a marker is sub-resolution at the acquisition it must "
                "survive: it aliases, phase correlation acquires a bias no "
                "averaging removes, and the fiducial is effectively invisible. "
                "Raise sigma_mm or kappa (>= 1.0)."
            )

        spa = (
            _per_axis(spacing_mm, "spacing_mm")
            if spacing_mm is not None
            else tuple(8.0 * s for s in sig)
        )
        if any(s <= 2.0 * g for s, g in zip(spa, sig, strict=True)):
            raise ValueError(
                f"spacing_mm {spa} is under 2*sigma_mm {sig} on some axis; peaks "
                "would merge into a smooth field with no localisable structure."
            )

        spacing_px = tuple(s / v for s, v in zip(spa, vox, strict=True))
        sigma_px = tuple(s / v for s, v in zip(sig, vox, strict=True))
        return spacing_px, sigma_px

    @staticmethod
    def _build_gaussian_grid(
        im_size: tuple[int, ...],
        spacing_px: tuple[float, ...],
        sigma_px: tuple[float, ...],
        jitter: float = 0.0,
        seed: int = 0,
    ) -> torch.Tensor:
        """Build an N-D field of anisotropic Gaussian peaks.

        Args:
            im_size: Image dimensions, 2-D ``(H, W)`` or 3-D ``(D, H, W)``.
            spacing_px: Per-axis pixel spacing between centres.
            sigma_px: Per-axis peak width in pixels. Anisotropic by design: the
                ULF cohort is 1.6 mm in-plane against 5.0 mm through-plane, so a
                scalar sigma is wrong by 3x on one axis whatever value it takes.
            jitter: Per-peak displacement in units of ``spacing_px``, drawn once
                from ``U(-jitter, jitter)``. Breaks the lattice periodicity.
            seed: Makes the jitter draw deterministic.

        Returns:
            Complex tensor ``[1, 1, *im_size]`` with Gaussian peaks.
        """
        ndim = len(im_size)
        if len(spacing_px) != ndim or len(sigma_px) != ndim:
            raise ValueError(
                f"spacing_px/sigma_px must have {ndim} entries for im_size "
                f"{im_size}, got {len(spacing_px)}/{len(sigma_px)}"
            )

        for axis, (n, sp) in enumerate(zip(im_size, spacing_px, strict=True)):
            if sp / 2.0 >= float(n):
                raise ValueError(
                    f"axis {axis} is {n} voxels but the peak spacing is "
                    f"{sp:.1f} voxels, so no marker peak fits on it. This is the "
                    "usual outcome of a thin slab combined with a coarse "
                    "effective resolution (the default spacing is 8*sigma). "
                    "Declare a smaller spacing_mm, or accept that this axis "
                    "cannot carry a fiducial and register in-plane only."
                )
        axes = [
            torch.arange(sp / 2.0, float(n), sp, dtype=torch.float32)
            for n, sp in zip(im_size, spacing_px, strict=True)
        ]
        centres = [c.reshape(-1) for c in torch.meshgrid(*axes, indexing="ij")]

        if jitter > 0.0:
            g = torch.Generator().manual_seed(seed)
            centres = [
                (c + (torch.rand(c.shape, generator=g) - 0.5) * (2.0 * jitter * sp)).clamp(
                    0.0, float(n) - 1.0
                )
                for c, sp, n in zip(centres, spacing_px, im_size, strict=True)
            ]

        # Separable per peak, then summed over peaks: [P, N_i] per axis.
        profiles = [
            torch.exp(
                -((torch.arange(n, dtype=torch.float32).unsqueeze(0) - c.unsqueeze(1)) ** 2)
                / (2 * s**2)
            )
            for n, c, s in zip(im_size, centres, sigma_px, strict=True)
        ]
        letters = "abcde"[:ndim]
        subscripts = ",".join(f"p{ch}" for ch in letters) + f"->{letters}"
        grid = torch.einsum(subscripts, *profiles)

        # Normalize to [0, 1]
        grid = grid / grid.max().clamp(min=1e-8)

        # Convert to complex: real = magnitude, imag = 0 (pure real fiducial)
        grid_complex = torch.complex(grid, torch.zeros_like(grid))

        return grid_complex.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Generate the Virtual Fiducial image.

        Args:
            batch_size: Number of copies in the batch dimension.

        Returns:
            Complex tensor [B, 1, H, W].

        forward method for VirtualFiducial.

        Executes PyTorch tensor operations.

        Args:
            batch_size (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return self.grid.expand(batch_size, *([-1] * (self.grid.ndim - 1)))

    def get_kspace(self, batch_size: int = 1) -> torch.Tensor:
        """Get the fiducial's k-space representation.

        Args:
            batch_size: Batch size.

        Returns:
            Complex k-space [B, 1, H, W].
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c

        image = self.forward(batch_size)
        return fft2c(image)


class MotionTrajectory(nn.Module):
    """Optimizable motion trajectory for TTO.

    Parameterizes the per-readout-line rigid-body motion as a learnable
    tensor θ̂(t) = [dx(t), dy(t), dθ(t)]. During TTO, only this module's
    parameters are updated while the network weights remain frozen.

    Args:
        num_readout_lines: Number of phase encode lines.
        init_mode: 'zero' (static start) or 'random' (warm start).

    Example:
        >>> traj = MotionTrajectory(num_readout_lines=256)
        >>> theta = traj()  # [1, 3, 256]
        >>> # Optimize via TTO loss:
        >>> loss.backward()
        >>> optimizer.step()  # Updates only theta
    """

    def __init__(
        self,
        num_readout_lines: int = 256,
        init_mode: str = "zero",
    ) -> None:
        """Initialize MotionTrajectory.

        Args:
            num_readout_lines: Number of k-space readout lines.
            init_mode: Initialization mode ('zero' or 'random').
        """
        super().__init__()

        if init_mode == "zero":
            init_params = torch.zeros(1, 3, num_readout_lines)
        elif init_mode == "random":
            init_params = torch.randn(1, 3, num_readout_lines) * 0.01
        else:
            raise ValueError(f"Unknown init_mode '{init_mode}'. Supported: 'zero', 'random'.")

        self.theta = nn.Parameter(init_params)

        logger.info(
            "[MotionTrajectory] Initialized: lines=%d, mode=%s",
            num_readout_lines,
            init_mode,
        )

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """Return the current motion estimate.

        Args:
            batch_size: Batch dimension expansion.

        Returns:
            Motion parameters [B, 3, N_lines].

        forward method for MotionTrajectory.

        Executes PyTorch tensor operations.

        Args:
            batch_size (int): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        return self.theta.expand(batch_size, -1, -1)

    @property
    def max_translation(self) -> float:
        """Maximum absolute translation across all readout lines."""
        with torch.no_grad():
            return self.theta[:, :2, :].abs().max().item()

    @property
    def max_rotation(self) -> float:
        """Maximum absolute rotation across all readout lines (radians)."""
        with torch.no_grad():
            return self.theta[:, 2, :].abs().max().item()

    @property
    def max_rotation_degrees(self) -> float:
        """Maximum absolute rotation (degrees)."""
        return self.max_rotation * 180 / math.pi


__all__ = ["MotionTrajectory", "VirtualFiducial"]
