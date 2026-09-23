"""Coil Sensitivity Map (CSM) Utilities
====================================

Utilities for estimating, loading, and working with coil sensitivity maps
in multi-coil MRI reconstruction.

.. note:: ESPIRiT implementation matches the canonical mikgroup/espirit-python
   reference.  Calibration kernels are **flipped + conjugated → FFT** (not IFFT)
   per Uecker et al. MRM 2014.

This module provides:
- ESPIRiT-based CSM estimation
- CSM loading from various formats
- CSM validation and preprocessing
- Basic coil combination methods

.. math::

    S_{rss} = \\sqrt{\\sum_c |x_c|^2}

    S_{sense} = \frac{\\sum_c S_c^* x_c}{\\sum_c |S_c|^2}
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import NamedTuple

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# (device.type, dtype) pairs whose cuSOLVER batched-complex eigh has already
# been proven broken in this process. See :func:`_robust_eigh`.
_EIGH_CPU_FALLBACK: set[tuple[str, torch.dtype]] = set()


def _robust_eigh(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Hermitian eigendecomposition resilient to cuSOLVER's
    batched-complex workspace failure.

    ``torch.linalg.eigh`` on CUDA routes large batches of small *complex*
    Hermitian matrices to ``cusolverDnXsyevBatched``, which requests a
    pathological workspace — e.g. ~33 GiB for a ``(65536, 4, 4)`` complex64
    batch (the per-pixel ESPIRiT Gram). Depending on GPU capacity this raises
    :class:`torch.linalg.LinAlgError` (``CUSOLVER_STATUS_INVALID_VALUE`` from
    the buffer-size query) or :class:`torch.cuda.OutOfMemoryError`. The error's
    "input matrix contains NaN" hint is misleading — the buffer-size query
    never inspects the data; only the batch count and complex dtype trigger it.

    On such a failure we recompute on CPU LAPACK (correct and, for the small
    per-pixel matrices here, ~0.1 s) and move the result back to the original
    device. ``.cpu()`` / ``eigh`` / ``.to`` are all differentiable, so callers
    inside an autograd graph are unaffected. ``magma`` is deliberately not used
    as a fallback: it is deprecated and re-dispatches the calibration SVD to a
    different (also-failing) cuSOLVER path.

    The failure is a property of the (device, dtype) pair, not of the data, so
    for an *accelerator* matrix it is latched in :data:`_EIGH_CPU_FALLBACK` on
    first occurrence: subsequent calls skip the doomed cuSOLVER workspace query
    outright rather than paying for it (and re-warning) once per slice. The
    first fallback warns; the rest log at debug level. The latch never
    suppresses a *new* device/dtype, and CPU input is never latched (it has no
    accelerated path to skip, so every CPU failure still warns).

    Args:
        matrix: ``(..., C, C)`` batch of Hermitian matrices.

    Returns:
        ``(eigenvalues, eigenvectors)`` — eigenvalues ascending — matching
        :func:`torch.linalg.eigh`.
    """

    def _cpu_eigh() -> tuple[torch.Tensor, torch.Tensor]:
        eigvals, eigvecs = torch.linalg.eigh(matrix.cpu())
        return eigvals.to(matrix.device), eigvecs.to(matrix.device)

    # The workspace bug is a cuSOLVER (accelerator) defect. A CPU matrix has no
    # accelerated path to skip -- its "fallback" re-runs the same LAPACK call --
    # so latching a CPU key would buy nothing and would only make the retry
    # order-dependent. Latch accelerators only.
    latchable = matrix.device.type != "cpu"
    key = (matrix.device.type, matrix.dtype)
    if latchable and key in _EIGH_CPU_FALLBACK:
        logger.debug(
            "eigh: using latched CPU LAPACK path for %s/%s (cuSOLVER "
            "batched-complex workspace bug seen earlier this process).",
            matrix.device,
            matrix.dtype,
        )
        return _cpu_eigh()

    try:
        return torch.linalg.eigh(matrix)
    except (torch.linalg.LinAlgError, torch.cuda.OutOfMemoryError) as exc:
        if latchable:
            _EIGH_CPU_FALLBACK.add(key)
        logger.warning(
            "torch.linalg.eigh failed on %s (%s); retrying on CPU LAPACK "
            "(cuSOLVER batched-complex workspace bug). This is a real "
            "device→host fallback, not silent. Further %s/%s eigh calls take "
            "the CPU path directly and log at debug level.",
            matrix.device,
            type(exc).__name__,
            matrix.device.type,
            matrix.dtype,
        )
        return _cpu_eigh()


def estimate_csm_rss(
    kspace: torch.Tensor,
    num_coils: int,
    kernel_size: int = 7,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using Root-Sum-Squares (RSS) normalization.

    NOTE: This is NOT the ESPIRiT method. It uses simple RSS normalization
    which is not robust to signal voids or aliasing. Use for testing or
    simple baselines only.

    Args:
        kspace: Complex k-space data (B, num_coils, H, W)
        num_coils: Number of coils (should match kspace.shape[1])
        kernel_size: Size of the calibration kernel (used for center crop)
        threshold: Unused in RSS method (kept for API compatibility)

    Returns:
        Complex sensitivity maps (B, num_coils, H, W)

    """
    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D")

    batch_size, n_coils, height, width = kspace.shape

    if n_coils != num_coils:
        raise ValueError(f"kspace has {n_coils} coils but num_coils={num_coils}")

    # For simplicity, implement a basic ESPIRiT-like estimation
    # In practice, this would use more sophisticated calibration

    # Convert to image domain for estimation
    from .fft_ops import ifft2c

    images = ifft2c(kspace)  # (B, num_coils, H, W)

    # Use center region for calibration (assuming fully sampled center)
    cal_size = min(kernel_size * 4, min(height, width) // 2)
    center_h = height // 2
    center_w = width // 2

    cal_images = images[
        :,
        :,
        center_h - cal_size // 2 : center_h + cal_size // 2,
        center_w - cal_size // 2 : center_w + cal_size // 2,
    ]

    # Estimate sensitivities as normalized coil images
    # This is a simplified version - real ESPIRiT uses eigenvalue decomposition
    rss = torch.sqrt(torch.sum(torch.abs(cal_images) ** 2, dim=1, keepdim=True))
    smaps = cal_images / (rss + 1e-8)

    # Interpolate back to full size (handle complex tensors)
    smaps_real = F.interpolate(
        smaps.real,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    smaps_imag = F.interpolate(
        smaps.imag,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    smaps_full = torch.complex(smaps_real, smaps_imag)

    return smaps_full


def estimate_csm_power_iter(
    kspace: torch.Tensor,
    num_coils: int,
    kernel_size: int = 7,
    num_iter: int = 10,
    seed: int = 0,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using Power Method (ESPIRiT approximation).

    This method approximates the principal eigenvector of the local calibration
    matrix using power iteration, which is computationally more efficient than
    full SVD while providing similar robustness to ESPIRiT.

    Args:
        kspace: Complex k-space data (B, num_coils, H, W)
        num_coils: Number of coils
        kernel_size: Size of calibration kernel (default: 7)
        num_iter: Number of power iterations (default: 10)

    Returns:
        Complex sensitivity maps (B, num_coils, H, W)
    """
    is_5d = kspace.dim() == 5
    if is_5d:
        B_orig, C_orig, H_orig, W_orig, D_orig = kspace.shape
        kspace = kspace.permute(0, 4, 1, 2, 3).reshape(B_orig * D_orig, C_orig, H_orig, W_orig)

    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace, got {kspace.dim()}D")

    batch_size, n_coils, height, width = kspace.shape
    device = kspace.device

    # 1. IFFT to image domain
    from .fft_ops import ifft2c

    images = ifft2c(kspace)  # (B, C, H, W)

    # 2. Extract calibration region (center of k-space)
    # Note: We perform calibration in image domain using a sliding window approach
    # efficiently implemented via convolution or patch extraction.
    # For efficiency in this implementation, we use a simplified approach:
    # We estimate the dominant eigenvector of the local covariance matrix R = x * x^H
    # smoothed by a boxcar filter (kernel_size).

    # Compute local covariance matrices R (B, H, W, C, C)
    # This is memory intensive, so we use a power iteration approach directly on images
    # v_{k+1} = R * v_k / ||R * v_k||
    # where R is the local correlation matrix.
    # R * v can be computed as: (x * x^H) * v = x * (x^H * v)
    # smoothed over a local neighborhood.

    # Initialize sensitivity map estimate (random or uniform)
    # (B, 1, H, W, C) - last dim is coil dimension for matrix mul.
    # Seed a LOCAL generator (not a global manual_seed — perf/determinism SSOT is
    # initialize_accelerator) so the maps are reproducible run-to-run; power
    # iteration converges to the same dominant eigenvector regardless of init in
    # signal regions, so the seed only pins the degenerate/background pixels.
    gen = torch.Generator(device=device).manual_seed(seed)
    v = torch.randn(
        batch_size,
        1,
        height,
        width,
        n_coils,
        dtype=images.dtype,
        device=device,
        generator=gen,
    )
    v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

    # Prepare smoothing kernel
    box_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device) / (kernel_size**2)

    # Permute images for easier broadcasting: (B, 1, H, W, C)
    imgs_perm = images.permute(0, 2, 3, 1).unsqueeze(1)  # (B, 1, H, W, C)

    for _ in range(num_iter):
        # 1. Compute inner product: x^H * v -> (B, 1, H, W, 1)
        # Sum over coils (last dim)
        # conj(imgs) * v
        inner = torch.sum(torch.conj(imgs_perm) * v, dim=-1, keepdim=True)

        # 2. Apply smoothing (local neighborhood aggregation)
        # Reshape to (B, 1, H, W) for conv2d
        inner_real = inner.real.squeeze(-1)  # (B, 1, H, W)
        inner_imag = inner.imag.squeeze(-1)

        # Apply box filter with explicit padding to avoid UserWarning with padding='same'.
        # For even kernel sizes, padding=k//2 yields output size H+1, so we trim back.
        pad = kernel_size // 2
        B1, C1, H1, W1 = inner_real.shape
        inner_smooth_real = F.conv2d(inner_real, box_kernel, padding=pad)[..., :H1, :W1]
        inner_smooth_imag = F.conv2d(inner_imag, box_kernel, padding=pad)[..., :H1, :W1]

        # Reconstruct complex smoothed inner product
        inner_smooth = torch.complex(inner_smooth_real, inner_smooth_imag)
        # (B, 1, H, W, 1)
        inner_smooth = inner_smooth.unsqueeze(-1)

        # 3. Multiply by x: x * (smoothed_inner) -> (B, 1, H, W, C)
        v_new = imgs_perm * inner_smooth

        # 4. Normalize
        v_norm = torch.norm(v_new, dim=-1, keepdim=True)
        v = v_new / (v_norm + 1e-8)

    # v is now the dominant eigenvector (sensitivity map)
    # Reshape back to (B, C, H, W)
    smaps = v.squeeze(1).permute(0, 3, 1, 2)

    if is_5d:
        smaps = smaps.view(B_orig, D_orig, C_orig, H_orig, W_orig).permute(0, 2, 3, 4, 1)

    return smaps


def load_csm_from_file(
    file_path: str,
    expected_shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """Load coil sensitivity maps from file.

    Args:
        file_path: Path to CSM file (.pt, .pth, .npy, .nii, .nii.gz)
        expected_shape: Expected shape (B, num_coils, H, W)

    Returns:
        Complex sensitivity maps tensor

    """
    import numpy as np

    from spectramr.shared.utils.safe_io import safe_torch_load

    if file_path.endswith((".pt", ".pth")):
        smaps = safe_torch_load(file_path, map_location="cpu")
    elif file_path.endswith(".npy"):
        smaps = torch.from_numpy(np.load(file_path))
    elif file_path.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib

            nii_img = nib.load(file_path)
            smaps = torch.from_numpy(nii_img.get_fdata())
        except ImportError:
            raise ImportError("nibabel required for NIfTI files")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    # Ensure complex dtype
    if not torch.is_complex(smaps):
        if smaps.shape[-1] == 2:
            smaps = torch.view_as_complex(smaps)
        else:
            smaps = torch.complex(smaps, torch.zeros_like(smaps))

    # Validate shape if expected_shape provided -- explicit precondition, fail fast
    # (no silent fallback: CLAUDE.md NN#3/#8, pitfall #9/#15).
    if expected_shape is not None and tuple(smaps.shape) != tuple(expected_shape):
        raise ValueError(
            f"Loaded CSM shape {tuple(smaps.shape)} != expected {tuple(expected_shape)} "
            f"(file_path={file_path!r})"
        )

    return smaps


def create_synthetic_csm(
    shape: tuple[int, int, int, int],
    num_coils: int = 8,
    coil_pattern: str = "birdcage",
) -> torch.Tensor:
    """Create synthetic coil sensitivity maps for testing.

    Args:
        shape: Desired shape (B, 1, H, W) - will be expanded to (B, num_coils, H, W)
        num_coils: Number of coils
        coil_pattern: Type of coil pattern ('birdcage', 'random', 'uniform')

    Returns:
        Complex sensitivity maps (B, num_coils, H, W)

    """
    batch_size, _, height, width = shape

    if coil_pattern == "birdcage":
        # Simple birdcage pattern approximation
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, height),
            torch.linspace(-1, 1, width),
            indexing="ij",
        )

        smaps = []
        for coil_idx in range(num_coils):
            angle = 2 * torch.pi * coil_idx / num_coils
            phase = torch.exp(1j * angle * torch.atan2(y, x))
            sensitivity = torch.exp(-0.5 * (x**2 + y**2)) * phase
            smaps.append(sensitivity)

        smaps = torch.stack(smaps, dim=0)  # (num_coils, H, W)

    elif coil_pattern == "random":
        # Random complex sensitivities
        smaps = torch.complex(
            torch.randn(num_coils, height, width),
            torch.randn(num_coils, height, width),
        )

    elif coil_pattern == "uniform":
        # Uniform sensitivities (all coils have same sensitivity)
        smaps = torch.ones(num_coils, height, width, dtype=torch.complex64)

    else:
        raise ValueError(f"Unknown coil pattern: {coil_pattern}")

    # Normalize each coil
    smaps = smaps / (torch.abs(smaps) + 1e-8)

    # Expand to batch dimension
    smaps = smaps.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return smaps


def validate_csm(
    smaps: torch.Tensor,
    kspace_shape: tuple[int, ...] | None = None,
) -> bool:
    """Validate coil sensitivity maps.

    Args:
        smaps: Sensitivity maps tensor
        kspace_shape: Expected k-space shape for compatibility check

    Returns:
        True if valid

    """
    if not torch.is_complex(smaps):
        logger.warning("CSM should be complex-valued")
        return False

    if smaps.dim() != 4:
        logger.warning(f"CSM should be 4D (B, num_coils, H, W), got {smaps.dim()}D")
        return False

    batch_size, num_coils, height, width = smaps.shape

    # Check for NaN/inf
    if torch.isnan(smaps).any() or torch.isinf(smaps).any():
        logger.warning("CSM contains NaN or inf values")
        return False

    # Check k-space compatibility
    if kspace_shape is not None:
        expected_shape = (batch_size, num_coils, height, width)
        if kspace_shape != expected_shape:
            logger.warning(f"CSM shape {expected_shape} incompatible with k-space {kspace_shape}")
            return False

    return True


def coil_combine_sense(
    coil_images: torch.Tensor,
    smaps: torch.Tensor,
    min_support_frac: float = 1e-2,
) -> torch.Tensor:
    r"""SENSE (Roemer-optimal) coil combination.

    Implements the matched-filter / Roemer optimal combine

    .. math::

        \hat{x} = \frac{\sum_c S_c^* \, x_c}{\sum_c |S_c|^2}

    The :math:`\sum_c |S_c|^2` denominator is what makes this an unbiased
    estimate of the underlying magnetization: with coil images
    :math:`x_c = S_c x` it recovers :math:`x` exactly, independent of the
    overall sensitivity scale. Omitting it (as a bare matched filter does)
    leaves an intensity shading by :math:`\sum_c |S_c|^2` — harmless only
    for unit-RSS-normalized maps, wrong for general ones.

    Outside the object support :math:`\sum_c |S_c|^2 \to 0`, and the division there is
    noise over nothing. The floor is therefore **relative to the map scale** and applied
    as a **clamp**, not as an additive Tikhonov term:

    * *Relative*, because an absolute floor is meaningless against a quantity whose
      scale is set by however the maps were normalised — the same defect class as the
      D2 rician sigma (#576). The previous ``eps = 1e-8`` was 1e-8 of the map maximum,
      i.e. no protection at all: measured on 4-coil M4Raw ESPIRiT maps, 24.6% of voxels
      have :math:`\sum_c|S_c|^2 < 10^{-3}`, the worst sit at ~1e-5, and the combined
      image reached **742x its own p99** — set by 26 air voxels whose RSS was *below*
      the image median, so ``max`` was pure amplified noise.
    * *A clamp rather than ``+ lambda``*, because the additive form destroys the exact
      recovery this function's whole docstring is about. Measured on
      :math:`x_c = S_c x`: additive gives 1.7e-2 relative error, the clamp gives
      6.7e-8 — identical to no floor at all. Well-conditioned voxels are untouched by
      construction; only the ill-conditioned ones move.

    ``min_support_frac = 1e-2`` caps the SENSE noise amplification at
    :math:`1/\sqrt{0.01} = 10\times` the best-conditioned voxel. Measured effect on the
    same M4Raw slice, against the RSS combination's physically bounded peak of 51.4:

    .. code-block:: text

        min_support_frac   |SENSE| max   in-object bias   voxels clamped
        0 (was: +1e-8)         1.6e+06           0.000%             0.0%
        1e-3                     276.5           0.000%            24.6%
        1e-2                      88.5           0.000%            25.5%
        3e-2                      52.9           0.000%            26.5%

    The ~25% that clamp at every setting are the air: the floor bites exactly where the
    maps have no support, which is what it is for.

    Args:
        coil_images: Complex coil images (B, num_coils, H, W)
        smaps: Complex sensitivity maps (B, num_coils, H, W)
        min_support_frac: Lower bound on :math:`\sum_c |S_c|^2`, as a fraction of that
            map's own spatial maximum. ``0.0`` restores the unprotected division.

    Returns:
        Combined image (B, 1, H, W)

    """
    numerator = torch.sum(coil_images * smaps.conj(), dim=1, keepdim=True)
    denominator = torch.sum((smaps.conj() * smaps).real, dim=1, keepdim=True)
    # Per-image maximum: a batch may mix anatomies, and one subject's map scale must not
    # set another's floor.
    peak = denominator.amax(dim=(-2, -1), keepdim=True)
    floor = (float(min_support_frac) * peak).clamp(min=torch.finfo(peak.dtype).tiny)
    return numerator / denominator.clamp(min=floor)


def coil_combine_rss(coil_images: torch.Tensor) -> torch.Tensor:
    """Root sum of squares coil combination.

    Args:
        coil_images: Complex coil images (B, num_coils, H, W)

    Returns:
        Combined magnitude image (B, 1, H, W)

    """
    rss = torch.sqrt(torch.sum(torch.abs(coil_images) ** 2, dim=1, keepdim=True))
    return rss


def estimate_acs_hanning_csm(
    kspace: torch.Tensor,
    center_fraction: float = 0.03,
) -> torch.Tensor:
    """Estimate coil sensitivity maps using center ACS region with 2D Hanning window.

    This captures the low-frequency phase and magnitude profile of each coil,
    useful for SENSE-based validation and cross-contrast applications.

    Args:
        kspace: Complex k-space data (B, num_coils, H, W).
                Must be centered (low frequencies in the middle).
        center_fraction: Fraction of k-space center to extract (default 0.03 -> 3%).

    Returns:
        Complex sensitivity maps (B, num_coils, H, W).
    """
    from .fft_ops import ifft2c

    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D")

    B, C, H, W = kspace.shape

    # 1. Define ACS region size
    acs_h = max(1, int(H * center_fraction))
    acs_w = max(1, int(W * center_fraction))

    # 2. Extract ACS region (center crop)
    center_h, center_w = H // 2, W // 2
    h_start = center_h - acs_h // 2
    h_end = h_start + acs_h
    w_start = center_w - acs_w // 2
    w_end = w_start + acs_w

    kspace_acs = torch.zeros_like(kspace)
    kspace_acs[:, :, h_start:h_end, w_start:w_end] = kspace[:, :, h_start:h_end, w_start:w_end]

    # 3. Apply 2D Hanning window to avoid ringing artifacts
    # Create 1D Hanning windows and use outer product for 2D
    window_h = torch.hann_window(acs_h, device=kspace.device, dtype=kspace.real.dtype)
    window_w = torch.hann_window(acs_w, device=kspace.device, dtype=kspace.real.dtype)
    hanning_2d = torch.outer(window_h, window_w)  # (acs_h, acs_w)

    # Broadcast to (B, C, acs_h, acs_w) and apply the window
    kspace_acs[:, :, h_start:h_end, w_start:w_end] *= hanning_2d.unsqueeze(0).unsqueeze(0)

    # 4. Transform to image domain to retrieve low-res coil images
    low_res_coil_images = ifft2c(kspace_acs)  # (B, C, H, W) complex

    # 5. Calculate Root-Sum-Square (RSS) magnitude across coils to represent Anatomy
    rss_mag = torch.sqrt(torch.sum(torch.abs(low_res_coil_images) ** 2, dim=1, keepdim=True))

    # 6. Normalize per-coil images by the RSS magnitude to isolate Coil Phase/Bias (S_c)
    # Adding small epsilon to prevent division by zero in the background air region
    smaps = low_res_coil_images / (rss_mag + 1e-8)

    return smaps


def espirit_min_acs_size(
    num_coils: int,
    kernel_size: int = 6,
    *,
    patch_margin: float = 1.5,
    max_acs: int | None = None,
) -> int:
    """Smallest square ACS side that keeps ESPIRiT calibration full-rank.

    The block-Hankel calibration matrix has ``(acs - kernel + 1)**2`` patch rows
    against ``kernel**2 * num_coils`` unknowns; :func:`estimate_csm_espirit`
    raises when the rows are fewer than the unknowns. This returns the smallest
    square ACS side with at least ``patch_margin`` rows per unknown (``1.5`` ->
    well-conditioned, not merely square). Callers holding a *fully sampled*
    calibration region (e.g. a NEX-merged pseudo-GT) may enlarge the ACS to this
    value to avoid a rank-deficient fall-back on many-coil data (#309).

    Only valid for fully sampled data — never use it to grow the ACS past an
    undersampling mask, which would pull zero-filled lines into the calibration.

    Args:
        num_coils: Coil count ``C`` in the k-space to calibrate.
        kernel_size: ESPIRiT calibration kernel side (must match the estimate call).
        patch_margin: Required patch-rows-per-unknown ratio (>= 1.0).
        max_acs: Optional upper clamp (the fully-sampled extent ``min(H, W)``);
            when the clamp is below the viable side the estimate may still raise,
            which is correct — the geometry is genuinely infeasible.
    """
    if num_coils < 1:
        raise ValueError(f"num_coils must be >= 1, got {num_coils}")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if patch_margin < 1.0:
        raise ValueError(f"patch_margin must be >= 1.0, got {patch_margin}")
    unknowns = kernel_size * kernel_size * num_coils
    needed_patches = patch_margin * unknowns
    side = kernel_size - 1 + math.ceil(math.sqrt(needed_patches))
    if max_acs is not None:
        side = min(side, max_acs)
    return side


class ESPIRiTSubspace(NamedTuple):
    """Sensitivity maps plus the coil subspace they were read off.

    ``estimate_csm_espirit`` eigendecomposes a per-pixel Gram matrix and keeps
    only the leading eigenvector. The remaining ``C - k`` eigenvectors span the
    coil NULL space: directions no coil combination can produce, so a physical
    signal has exactly zero component there while noise does not. Returning the
    projector makes that measurable instead of discarding it.

    Attributes:
        maps: ``(B, C, H, W)`` complex sensitivity maps -- what the bare
            ``estimate_csm_espirit`` call returns.
        null_projector: ``(B, H, W, C, C)`` complex ``P = I - V V^H``, with ``V``
            the eigenvectors at or above ``eigen_threshold``. Hermitian and
            idempotent per pixel.
        leading_eigenvalue: ``(B, H, W)`` real. Near 1 inside the support,
            near 0 outside it.
        second_eigenvalue: ``(B, H, W)`` real. The rank-one signal model this
            module assumes holds only where this is well below 1; where it is
            not, the pixel needs a second ESPIRiT map (FOV wrap) and its null
            space is ``C - 2`` rather than ``C - 1``. Checking it is how you
            find that out instead of assuming it.
    """

    maps: torch.Tensor
    null_projector: torch.Tensor
    leading_eigenvalue: torch.Tensor
    second_eigenvalue: torch.Tensor


def estimate_csm_espirit(
    kspace: torch.Tensor,
    num_coils: int,
    kernel_size: int = 6,
    acs_size: int | tuple[int, int] = 24,
    sigma_threshold: float = 0.02,
    eigen_threshold: float = 0.95,
    phase_ref_coil: int = 0,
    max_n_keep: int | None = None,
    *,
    return_subspace: bool = False,
) -> torch.Tensor | ESPIRiTSubspace:
    """Estimate coil sensitivity maps using the canonical ESPIRiT algorithm.

    Implements the full 7-step ESPIRiT pipeline from Uecker et al.,
    *ESPIRiT — an eigenvalue approach to autocalibrating parallel MRI:
    Where SENSE meets GRAPPA*, MRM 2014.

    Steps:
        1. Extract the Auto-Calibration Signal (ACS) region from k-space center.
        2. Build calibration matrix **A** from overlapping k-space patches.
        3. SVD of A; retain right singular vectors above ``sigma_threshold``.
        4. IFFT retained vectors to image domain → per-pixel matrix G.
        5. Per-pixel Gram matrix: M_q = G_q^H G_q.
        6. Eigen-decompose M_q; keep eigenvector for λ ≈ 1 (in-support mask).
        7. Phase-reference to ``phase_ref_coil`` and RSS-normalize.

    Args:
        kspace: Complex k-space data ``(B, C, H, W)`` — DC must be at center
            (i.e. after ``torch.fft.fftshift``).
        num_coils: Expected number of coils; must equal ``kspace.shape[1]``.
        kernel_size: Width/height of the sliding calibration patch (default: 6).
        acs_size: Number of ACS lines extracted from the k-space center.
            An ``int`` extracts a square region; a ``(H, W)`` tuple extracts a
            rectangular region.  For ultra-low-field data (e.g. M4Raw at 0.3T)
            use the **full readout dimension** to maximize the number of
            calibration patches: ``acs_size=(256, 30)`` (default: 24).
        sigma_threshold: Fraction of the largest singular value below which
            columns of V are considered noise and discarded (default: 0.02).
        eigen_threshold: Pixels whose leading eigenvalue is below this threshold
            are considered outside-of-support and are zeroed out (default: 0.95).
        phase_ref_coil: Index of the coil used as phase reference (default: 0).
        max_n_keep: Hard upper bound on the number of retained singular vectors.
            ``None`` means no limit (only ``sigma_threshold`` decides).

    Returns:
        Complex sensitivity maps ``(B, C, H, W)``.

    Raises:
        ValueError: If ``kspace`` is not 4-D or coil count mismatches.
    """
    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D.")

    batch_size, n_coils, height, width = kspace.shape
    if n_coils != num_coils:
        raise ValueError(f"kspace has {n_coils} coils but num_coils={num_coils}.")

    device = kspace.device
    dtype = kspace.dtype  # complex64 or complex128

    # -----------------------------------------------------------------------
    # Step 1: Extract ACS region
    # -----------------------------------------------------------------------
    if isinstance(acs_size, int):
        acs_h, acs_w = min(acs_size, height), min(acs_size, width)
    else:
        acs_h, acs_w = min(acs_size[0], height), min(acs_size[1], width)

    if acs_h < kernel_size or acs_w < kernel_size:
        raise ValueError(
            f"ESPIRiT ACS region ({acs_h}x{acs_w}) must be >= kernel_size "
            f"({kernel_size}) in both dimensions; increase acs_size or reduce "
            f"kernel_size."
        )

    # Rank viability: the block-Hankel calibration matrix has n_patches rows
    # against kernel²·coils unknowns. When n_patches < unknowns it is
    # rank-deficient → ill-conditioned / non-finite maps (the exp_11 kernel12/
    # acs24/4-coil divergence). Raise rather than emit a silent bad map (#9/#16).
    n_patches = (acs_h - kernel_size + 1) * (acs_w - kernel_size + 1)
    unknowns = kernel_size * kernel_size * n_coils
    if n_patches < unknowns:
        # Name the value that WOULD work. "increase acs_size" alone leaves the
        # caller solving (acs - k + 1)^2 >= margin*k^2*C by hand, and the whole
        # point of espirit_min_acs_size is that the repo already knows the answer.
        viable = espirit_min_acs_size(n_coils, kernel_size=kernel_size, max_acs=min(height, width))
        raise ValueError(
            f"ESPIRiT calibration is rank-deficient: {n_patches} ACS patches "
            f"({acs_h}x{acs_w}, kernel {kernel_size}) < {unknowns} unknowns "
            f"(kernel²·coils = {kernel_size}²·{n_coils}). Reduce kernel_size or "
            f"raise acs_size to at least {viable} (espirit_min_acs_size for "
            f"{n_coils} coils at kernel {kernel_size}). Only widen the ACS over a "
            f"FULLY SAMPLED region -- growing it past an undersampling mask pulls "
            f"zero-filled lines into the calibration."
        )

    ch, cw = height // 2, width // 2
    h0, h1 = ch - acs_h // 2, ch - acs_h // 2 + acs_h
    w0, w1 = cw - acs_w // 2, cw - acs_w // 2 + acs_w
    # acs: (B, C, acs_h, acs_w)
    acs = kspace[:, :, h0:h1, w0:w1]

    # We operate per-batch item. Collect smaps for each item then stack.
    smaps_batch: list[torch.Tensor] = []
    null_batch: list[torch.Tensor] = []
    lead_batch: list[torch.Tensor] = []
    second_batch: list[torch.Tensor] = []

    for b in range(batch_size):
        acs_b = acs[b]  # (C, acs_h, acs_w)

        # -------------------------------------------------------------------
        # Step 2: Build calibration matrix A  (Hankel / block-Hankel)
        #
        # Each row = one overlapping k×k patch across ALL coils, flattened
        # in **spatial-major** order: (k, k, C).  This matches the
        # BART / SigPy convention and is critical: the right singular
        # vectors V will be reshaped as (k, k, C), so every contiguous
        # block of nc elements describes the same k-space location across
        # all coils.
        #
        # Shape: (n_patches, kernel_size * kernel_size * C)
        # -------------------------------------------------------------------
        patches: list[torch.Tensor] = []
        for row in range(acs_h - kernel_size + 1):
            for col in range(acs_w - kernel_size + 1):
                # (C, k, k) → (k, k, C) then flatten → spatial-major
                patch = acs_b[:, row : row + kernel_size, col : col + kernel_size]
                patches.append(patch.permute(1, 2, 0).reshape(-1))  # (k*k*C,)
        A = torch.stack(patches, dim=0)  # (n_patches, k*k*C)

        # -------------------------------------------------------------------
        # Step 3: SVD of A; threshold singular values
        # -------------------------------------------------------------------
        # torch.linalg.svd returns U (m×k), S (k,), Vh (k×n)
        _U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        s_max = S[0] if S.numel() > 0 else torch.tensor(1.0, device=device)
        sig_keep_mask = (sigma_threshold * s_max) < S
        if max_n_keep is not None:
            # Limit number of kept vectors to avoid memory explosion
            keep_indices = sig_keep_mask.nonzero(as_tuple=False).squeeze(-1)
            if keep_indices.numel() > max_n_keep:
                keep_indices = keep_indices[:max_n_keep]
            sig_keep_mask = torch.zeros_like(sig_keep_mask)
            sig_keep_mask[keep_indices] = True

        V_parallel = Vh[sig_keep_mask, :]  # (n_keep, k*k*C)
        n_keep = V_parallel.shape[0]

        if n_keep == 0:
            # Fallback: keep at least the leading singular vector
            V_parallel = Vh[:1, :]
            n_keep = 1

        # Reshape to (n_keep, k, k, C) then transpose to (n_keep, C, k, k)
        # so the rest of the pipeline works in PyTorch (C, H, W) layout.
        V_reshaped = V_parallel.reshape(n_keep, kernel_size, kernel_size, n_coils).permute(
            0, 3, 1, 2
        )

        # -------------------------------------------------------------------
        # Step 4: Flip kernel, zero-pad, then FFT2
        #
        # Reference: mikgroup/espirit-python (espirit.py)
        #   V_col = Vh.conj().T          # V columns
        #   ker = flip(V_col).conj()     # flip + conj of V column
        #
        # Since our code stores **Vh rows** (= conj of V columns):
        #   flip(conj(Vh_row)).conj()  ≡  flip(Vh_row)
        #
        # So we only need to flip — NO conjugation needed.
        # -------------------------------------------------------------------
        V_flipped = V_reshaped.flip(dims=(-2, -1))

        V_padded = torch.zeros(n_keep, n_coils, height, width, dtype=dtype, device=device)
        # Place the kernel at the center (consistent with the fftshift convention)
        kh0 = ch - kernel_size // 2
        kw0 = cw - kernel_size // 2
        V_padded[:, :, kh0 : kh0 + kernel_size, kw0 : kw0 + kernel_size] = V_flipped

        # Centered FFT2 (ifftshift before, fft2, fftshift after) with ortho norm
        V_padded_shifted = torch.fft.ifftshift(V_padded, dim=(-2, -1))
        G_raw = torch.fft.fft2(V_padded_shifted, dim=(-2, -1), norm="ortho")
        G_raw = torch.fft.fftshift(G_raw, dim=(-2, -1))  # (n_keep, C, H, W)

        # Explicit scaling to match BART eigenvalue range [0, 1]
        # sqrt(H*W) / sqrt(k²) per basis function → (H*W)/k² in Gram matrix
        espirit_scale = math.sqrt(height * width) / math.sqrt(kernel_size**2)
        G = G_raw * espirit_scale

        # -------------------------------------------------------------------
        # Step 5: Per-pixel Gram matrix  M_q = G_q^H G_q
        #
        # The scaling factor is already embedded in G (Step 4), so the
        # eigenvalues naturally fall in [0, 1]:
        #   - In-support:     max eigenvalue → 1.0  (SENSE signal subspace)
        #   - Out-of-support: max eigenvalue → 0.0  (noise-only)
        # -------------------------------------------------------------------
        G_flat = G.reshape(n_keep, n_coils, height * width)  # (n_keep, C, N)
        G_pix = G_flat.permute(2, 1, 0)  # (N, C, n_keep)
        M = G_pix @ G_pix.conj().transpose(-2, -1)  # (N, C, C)

        # -------------------------------------------------------------------
        # Step 6: Per-pixel eigendecomposition and Soft-SENSE Weighting
        # torch.linalg.eigh returns eigenvalues in ascending order
        # -------------------------------------------------------------------
        # CPU-resilient: cuSOLVER's batched complex eigh blows up the workspace
        # (~33 GiB for the (H*W, C, C) Gram) → CUSOLVER_STATUS_INVALID_VALUE / OOM.
        eigenvalues, eigenvectors = _robust_eigh(M)  # (N, C), (N, C, C)
        # Finite-guard: a singular/ill-conditioned Gram (e.g. rank-deficient
        # calibration on low-SNR data) can yield NaN/Inf eig; never propagate
        # that into a coil map feeding a training loss (#9). Raise loudly.
        if not (torch.isfinite(eigenvalues).all() and torch.isfinite(eigenvectors).all()):
            raise ValueError(
                f"ESPIRiT eigendecomposition produced non-finite values "
                f"(acs {acs_h}x{acs_w}, kernel {kernel_size}, {n_coils} coils, "
                f"n_keep {n_keep}); calibration is singular/ill-conditioned."
            )
        # Leading eigenvector = last column (highest eigenvalue)
        lead_eigenval = eigenvalues[:, -1]  # (N,)  real
        lead_eigenvec = eigenvectors[:, :, -1]  # (N, C) complex

        if return_subspace:
            # The C - k eigenvectors BELOW the support threshold span directions
            # no coil combination can produce, so a physical signal has exactly
            # zero component there. Built from the same eigendecomposition the
            # maps come from rather than recomputed, so the two cannot disagree.
            in_support = eigenvalues >= eigen_threshold  # (N, C)
            basis = eigenvectors * in_support.unsqueeze(1).to(eigenvectors.dtype)
            signal_proj = basis @ basis.conj().transpose(-2, -1)  # (N, C, C)
            identity = torch.eye(n_coils, dtype=signal_proj.dtype, device=signal_proj.device)
            null_batch.append(
                (identity - signal_proj).reshape(height, width, n_coils, n_coils)
            )
            lead_batch.append(lead_eigenval.reshape(height, width))
            second_batch.append(eigenvalues[:, -2].reshape(height, width))

        # ESPIRiT Soft-SENSE Weighting (Replaces the hard boolean mask)
        # We linearly map the eigenvalues from the threshold up to 1.0.
        # This smoothly tapers the boundaries to exactly zero, preventing jagged cutouts.
        weights = (lead_eigenval - eigen_threshold) / (1.0 - eigen_threshold)
        weights = torch.clamp(weights, min=0.0, max=1.0)

        # Multiply the normalized eigenvectors by the soft weights
        smaps_flat = lead_eigenvec * weights.unsqueeze(-1)  # (N, C)

        # Reshape to (C, H, W)
        smaps_b = smaps_flat.permute(1, 0).reshape(n_coils, height, width)

        # -------------------------------------------------------------------
        # Step 7: Phase reference
        # Note: torch.linalg.eigh inherently provides orthonormal eigenvectors,
        # so sum(|s_c|^2) = 1. The soft weighting applied above correctly
        # handles the background mask. No further RSS division is needed.
        # -------------------------------------------------------------------
        ref_phase = torch.angle(smaps_b[phase_ref_coil : phase_ref_coil + 1, :, :])
        smaps_b = smaps_b * torch.exp(-1j * ref_phase)

        smaps_batch.append(smaps_b)

    # Stack batch dimension: (B, C, H, W)
    maps = torch.stack(smaps_batch, dim=0)
    if not return_subspace:
        return maps
    return ESPIRiTSubspace(
        maps=maps,
        null_projector=torch.stack(null_batch, dim=0),
        leading_eigenvalue=torch.stack(lead_batch, dim=0),
        second_eigenvalue=torch.stack(second_batch, dim=0),
    )


def estimate_csm_pinn(
    kspace: torch.Tensor,
    num_coils: int,
    epochs: int = 1500,
    lr: float = 1e-4,
    lambda_pde_max: float = 1000.0,
    lambda_norm: float = 10.0,
    acs_fraction: float = 0.08,
    **kwargs,
) -> torch.Tensor:
    r"""Estimate coil sensitivity maps via zero-shot PINN optimization.

    Performs **subject-specific, zero-shot** optimization of a SIREN coordinate
    network :math:`G_\theta` using strictly the data from *this specific scan*.
    No pre-training or external ground-truth maps are required.

    The optimisation objective has exactly three terms:

    .. math::

        \mathcal{L}_{Total}(\theta) = \mathcal{L}_{DC}
            + \lambda_{PDE}\,\mathcal{L}_{PDE}
            + \lambda_{Norm}\,\mathcal{L}_{Norm}

    **A. Data Consistency (DC) Loss** — anchors to measured ACS k-space:

    .. math::

        \mathcal{L}_{DC} = \sum_{c} \left\|
            \mathcal{M}_{ACS} \odot \mathcal{F}\bigl(
                S_c(x,y;\theta) \cdot \hat{m}(x,y)
            \bigr) - \mathbf{y}_{c,ACS}
        \right\|_2^2

    **B. PDE Physics Prior** — Helmholtz/Laplace extrapolation:

    .. math::

        \mathcal{L}_{PDE} = \frac{1}{N} \sum_i \sum_c \left\|
            \nabla^2 S_c(\mathbf{p}_i;\theta) + k^2 S_c(\mathbf{p}_i;\theta)
        \right\|_2^2

    **C. Normalization Prior** — unit-norm coil vectors (Uecker NLINV):

    .. math::

        \mathcal{L}_{Norm} = \sum_{x,y} \left(
            1 - \sum_c |S_c(x,y;\theta)|^2
        \right)^2

    References:
        - Ulyanov et al., "Deep Image Prior", CVPR 2018
        - Uecker et al., "Regularized nonlinear inversion...", MRM 2008
        - Yaman et al., "Self-Supervised Physics-Guided...", MRM 2020

    Args:
        kspace: Complex k-space ``(B, C, H, W)`` with DC at center.
        num_coils: Number of receive coils.
        epochs: Adam optimisation steps per batch item.
        lr: Learning rate for the SIREN model.
        lambda_pde_max: Maximum weight for adaptive PDE gradient scaling.
        lambda_norm: Weight for the normalization loss :math:`\lambda_{Norm}`.
        acs_fraction: Fraction of k-space center used as ACS calibration region.

    Returns:
        Complex sensitivity maps ``(B, C, H, W)``.
    """
    from spectramr.models.generators.siren_pinn import SirenSensNet, get_last_shared_layer
    from spectramr.models.losses.pde_losses import HelmholtzPDELoss

    from .fft_ops import fft2c, ifft2c

    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D.")

    batch_size, n_coils, height, width = kspace.shape
    if n_coils != num_coils:
        raise ValueError(f"kspace has {n_coils} coils but num_coils={num_coils}.")

    device = kspace.device

    # Extract kwargs mapped from YAML
    hidden_features = kwargs.get("hidden_features", 256)
    hidden_layers = kwargs.get("hidden_layers", 4)
    first_omega_0 = kwargs.get("first_omega_0", 30.0)
    hidden_omega_0 = kwargs.get("hidden_omega_0", 30.0)
    collocation_points = kwargs.get("collocation_points", 2048)
    # Note: lambda_tv can be mapped if TV is implemented

    # ──────────────────────────────────────────────────────────────────
    # 1. Build the explicit ACS binary mask  M_acs  (frozen)
    # ──────────────────────────────────────────────────────────────────
    acs_h = max(1, int(height * acs_fraction))
    acs_w = max(1, int(width * acs_fraction))
    center_h, center_w = height // 2, width // 2

    h0, h1 = center_h - acs_h // 2, center_h - acs_h // 2 + acs_h
    w0, w1 = center_w - acs_w // 2, center_w - acs_w // 2 + acs_w

    acs_mask = torch.zeros(1, 1, height, width, device=device)
    acs_mask[:, :, h0:h1, w0:w1] = 1.0  # (1, 1, H, W) — broadcasts over B, C

    # ──────────────────────────────────────────────────────────────────
    # 2. Compute magnitude proxy  m_hat  (RSS of zero-filled ACS)
    # ──────────────────────────────────────────────────────────────────
    kspace_acs = kspace * acs_mask  # (B, C, H, W) — zero outside ACS
    low_res_coil_images = ifft2c(kspace_acs)  # (B, C, H, W) complex
    # RSS across coils → (B, H, W)  real-valued magnitude proxy
    m_hat_batch = torch.sqrt(torch.sum(torch.abs(low_res_coil_images) ** 2, dim=1) + 1e-12)

    # Measured ACS k-space  y_acs  (frozen target)
    y_acs = kspace * acs_mask  # (B, C, H, W)

    # ──────────────────────────────────────────────────────────────────
    # 3. Normalised coordinate grid  [-1, 1]²
    # ──────────────────────────────────────────────────────────────────
    gy = torch.linspace(-1, 1, height, device=device)
    gx = torch.linspace(-1, 1, width, device=device)
    mgrid = torch.stack(torch.meshgrid(gy, gx, indexing="ij"), dim=-1)
    coords_base = mgrid.reshape(-1, 2)  # (H*W, 2)

    smaps_out = []

    # ──────────────────────────────────────────────────────────────────
    # 4. Per-subject zero-shot optimisation loop
    # ──────────────────────────────────────────────────────────────────
    for b in range(batch_size):
        logger.info("PINN CSM: batch %d/%d — %d epochs", b + 1, batch_size, epochs)

        m_hat = m_hat_batch[b]  # (H, W) — frozen anatomy proxy
        y_acs_b = y_acs[b]  # (C, H, W) — frozen measured ACS

        # Fresh untrained SIREN for each subject (Deep Image Prior paradigm)
        model = SirenSensNet(
            in_features=2,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=num_coils * 2,
            first_omega_0=first_omega_0,
            hidden_omega_0=hidden_omega_0,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pde_criterion = HelmholtzPDELoss(k_sq=0.0).to(device)

        coords_dc = coords_base.clone().detach()  # detached for DC path
        coords_pde = coords_base.clone()  # base for random PDE sub-batches

        for epoch in range(epochs):
            optimizer.zero_grad()

            # ── A. Data Consistency Loss ─────────────────────────────
            # Predict S_c on full grid → reshape → apply forward model
            S_r, S_i = model(coords_dc)  # (H*W, C_coils)
            S_complex = torch.complex(S_r, S_i)  # (H*W, C_coils)
            S_2d = S_complex.T.reshape(num_coils, height, width)  # (C, H, W)

            # Forward model:  F(S_c · m_hat) masked to ACS
            weighted = S_2d * m_hat.unsqueeze(0)  # (C, H, W)
            kspace_pred = fft2c(weighted.unsqueeze(0)).squeeze(0)  # (C, H, W)
            kspace_pred_acs = kspace_pred * acs_mask.squeeze(0)  # (C, H, W)

            dc_loss = torch.mean(torch.abs(kspace_pred_acs - y_acs_b) ** 2)

            # ── B. PDE Loss (random sub-batch) ──────────────────────
            n_pde = min(collocation_points, height * width)
            idx = torch.randperm(height * width, device=device)[:n_pde]
            coords_pde_batch = coords_pde[idx].clone().detach().requires_grad_(True)

            S_r_pde, S_i_pde = model(coords_pde_batch)
            pde_loss = pde_criterion(S_r_pde, S_i_pde, coords_pde_batch)

            # ── C. Normalization Loss ───────────────────────────────
            # |S_c|² summed over coils → should equal 1 everywhere
            S_mag_sq = S_r**2 + S_i**2  # (H*W, C_coils)
            sum_mag_sq = S_mag_sq.sum(dim=-1)  # (H*W,)
            norm_loss = torch.mean((1.0 - sum_mag_sq) ** 2)

            # ── Adaptive PDE weighting (GradNorm-style) ─────────────
            shared_w = get_last_shared_layer(model)
            grad_dc = torch.autograd.grad(dc_loss, shared_w, retain_graph=True)[0]
            grad_pde = torch.autograd.grad(pde_loss, shared_w, retain_graph=True)[0]

            lam_pde = (
                torch.max(torch.abs(grad_dc)) / (torch.mean(torch.abs(grad_pde)) + 1e-8)
            ).item()
            lam_pde = min(lam_pde, lambda_pde_max)

            # ── Total loss ──────────────────────────────────────────
            total_loss = dc_loss + lam_pde * pde_loss + lambda_norm * norm_loss
            total_loss.backward()
            optimizer.step()

            if epoch % 200 == 0 or epoch == epochs - 1:
                logger.info(
                    "  [%4d/%d] DC=%.4e  PDE=%.4e (λ=%.1f)  Norm=%.4e  Total=%.4e",
                    epoch,
                    epochs,
                    dc_loss.item(),
                    pde_loss.item(),
                    lam_pde,
                    norm_loss.item(),
                    total_loss.item(),
                )

        # ── Extract final map ───────────────────────────────────────
        with torch.no_grad():
            S_r_final, S_i_final = model(coords_dc)
            S_pinn = torch.complex(S_r_final, S_i_final)
            S_pinn = S_pinn.T.reshape(num_coils, height, width)
            smaps_out.append(S_pinn)

    return torch.stack(smaps_out, dim=0)


def extract_acs_region(kspace: torch.Tensor, acs_size: int | tuple[int, int]) -> torch.Tensor:
    """Crop the central ACS block from DC-centered k-space ``(B, C, H, W)``.

    Coil maps are acceleration-invariant → calibrate only from the dense,
    fully-sampled low-frequency center, never the aliased periphery.
    """
    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D.")
    _, _, height, width = kspace.shape
    if isinstance(acs_size, int):
        acs_h, acs_w = min(acs_size, height), min(acs_size, width)
    else:
        acs_h, acs_w = min(acs_size[0], height), min(acs_size[1], width)
    h0, w0 = height // 2 - acs_h // 2, width // 2 - acs_w // 2
    return kspace[:, :, h0 : h0 + acs_h, w0 : w0 + acs_w]


def estimate_smaps(
    kspace: torch.Tensor,
    method: str = "power_iter",
    *,
    kernel_size: int = 6,
    acs_size: int = 24,
    eigen_threshold: float = 0.95,
    maps_path: str | None = None,
    acs_only: bool = False,
    **_ignored,
) -> torch.Tensor | None:
    """Config-driven sensitivity-map dispatcher.

    Routes to the appropriate ``estimate_csm_*`` primitive based on ``method``.
    Raises on an unknown method (no silent fallback — CLAUDE.md pitfall #9/#15).

    Args:
        kspace: Complex k-space ``(B, C, H, W)``.
        method: One of ``none``, ``power_iter``, ``espirit``, ``pinn``, ``rss``,
            ``file``.
        kernel_size: Calibration kernel size (power_iter / espirit).
        acs_size: Autocalibration-region size (espirit).
        eigen_threshold: ESPIRiT eigenvalue threshold.
        maps_path: Required when ``method == "file"``.
        acs_only: Pre-crop to the central ``acs_size`` block before dispatch so
            out-of-ACS aliasing cannot leak into the calibration (fixes
            ``power_iter``, which otherwise IFFTs the whole tensor).

    Returns:
        Complex sensitivity maps ``(B, C, H, W)``, or ``None`` for ``method="none"``.
    """
    n_coils = kspace.shape[1]
    if method == "none":
        return None
    if acs_only and method in ("power_iter", "espirit"):
        kspace = extract_acs_region(kspace, acs_size)
        if method == "power_iter" and min(kspace.shape[-2:]) < kernel_size:
            raise ValueError(
                f"ACS region {tuple(kspace.shape[-2:])} smaller than "
                f"kernel_size={kernel_size}; increase acs_size or reduce kernel_size."
            )
    if method == "power_iter":
        return estimate_csm_power_iter(kspace, n_coils, kernel_size=kernel_size)
    if method == "espirit":
        return estimate_csm_espirit(
            kspace,
            n_coils,
            kernel_size=kernel_size,
            acs_size=acs_size,
            eigen_threshold=eigen_threshold,
        )
    if method == "rss":
        return estimate_csm_rss(kspace, n_coils)
    if method == "pinn":
        return estimate_csm_pinn(kspace, n_coils)
    if method == "file":
        if not maps_path:
            raise ValueError("estimation method='file' requires maps_path")
        return load_csm_from_file(maps_path)
    raise ValueError(
        f"Unknown sensitivity estimation method {method!r}. Valid: "
        "none, power_iter, espirit, pinn, rss, file."
    )


def confine_to_acs(kspace: torch.Tensor, acs_size: int | tuple[int, int]) -> torch.Tensor:
    """Zero every bin outside the central ACS block, keeping the full grid.

    The same calibration confinement :func:`extract_acs_region` performs by
    cropping, expressed so the estimator still receives a full-resolution grid
    and returns full-resolution maps. Cropping buys the same protection against
    aliased periphery and then costs the resolution back, which a caller can
    only recover by interpolating.
    """
    if kspace.dim() != 4:
        raise ValueError(f"Expected 4D kspace (B, C, H, W), got {kspace.dim()}D.")
    _, _, height, width = kspace.shape
    if isinstance(acs_size, int):
        acs_h, acs_w = min(acs_size, height), min(acs_size, width)
    else:
        acs_h, acs_w = min(acs_size[0], height), min(acs_size[1], width)
    h0, w0 = height // 2 - acs_h // 2, width // 2 - acs_w // 2
    confined = torch.zeros_like(kspace)
    confined[:, :, h0 : h0 + acs_h, w0 : w0 + acs_w] = kspace[
        :, :, h0 : h0 + acs_h, w0 : w0 + acs_w
    ]
    return confined


def estimate_smaps_calibrated(
    kspace: torch.Tensor,
    *,
    method: str = "power_iter",
    out_size: tuple[int, int] | None = None,
    eps: float = 1e-8,
    **estimation_kwargs: object,
) -> torch.Tensor | None:
    """Calibrate coil maps from the ACS and return them RSS-normalised at full size.

    The single owner (non-negotiable 17) of the three-step sequence every
    runtime consumer needs: confine the calibration to the ACS, estimate, and
    normalise so ``sum_c |s_c|^2 == 1``. It replaces three hand-rolled copies --
    training (``DiffusionTrainingStrategy._estimate_smaps_cached``), validation
    (the same class's validation branch) and sampling
    (``ColdDiffusionInferenceStrategy``) -- which had to agree for a checkpoint
    to sample with the maps it trained with, and were kept in step by hand.

    **The order is the whole point.** All three copies normalised on the
    ``acs_size x acs_size`` grid and then bilinearly interpolated ``real`` and
    ``imag`` separately up to the image size. Linear interpolation between two
    unit-modulus complex numbers whose phases differ returns the chord, not the
    arc, so the modulus collapsed everywhere between grid nodes: RSS was 1.0
    exactly at the 24 node positions and 0.0024 between them, i.e. a 256/24 =
    10.67 px mesh, with ``E[sum_c |s_c|^2] = 0.58`` against the contracted 1.0.
    Measured identically on a synthetic 4-coil phantom (0.5824) and on the
    archived ``experiment_11_attention_none`` snapshot (0.5717); #2213.

    So nothing is interpolated here. ``power_iter`` is handed an ACS-confined
    grid at full resolution rather than a crop (:func:`confine_to_acs`), and
    ``espirit`` extracts its own ACS internally and already returns at input
    resolution -- both come back the size they went in, and ``out_size`` exists
    only for a caller whose k-space genuinely differs from its image grid. When
    it does fire, the normalisation runs *after* it.

    Args:
        kspace: Complex k-space ``(B, C, H, W)``, DC centred.
        method: Any :func:`estimate_smaps` method. ``"none"`` returns ``None``.
        out_size: ``(H, W)`` to resize onto when the maps do not already match.
        eps: Floor inside the RSS divide.
        **estimation_kwargs: ``kernel_size`` / ``acs_size`` / ``eigen_threshold``
            / ``maps_path``, as :func:`resolve_estimation_settings` returns them.

    Returns:
        ``(B, C, H, W)`` complex maps with unit coil-RSS, or ``None`` for
        ``method="none"``.
    """
    acs_size = int(estimation_kwargs.get("acs_size", 24))  # type: ignore[call-overload]
    # ``espirit`` crops internally and is measured against the FULL grid's
    # eigenvalues; ``power_iter`` IFFTs whatever it is given, so it is the one
    # that needs the periphery zeroed before it sees the tensor.
    calibration = confine_to_acs(kspace, acs_size) if method == "power_iter" else kspace
    smaps = estimate_smaps(calibration, method=method, acs_only=False, **estimation_kwargs)  # type: ignore[arg-type]
    if smaps is None:
        return None
    smaps = smaps.detach()

    if out_size is not None and tuple(smaps.shape[-2:]) != tuple(out_size):
        smaps = torch.complex(
            F.interpolate(smaps.real, size=out_size, mode="bilinear", align_corners=False),
            F.interpolate(smaps.imag, size=out_size, mode="bilinear", align_corners=False),
        )
    return smaps / torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True) + eps)


_ESTIMATION_SUB_KNOBS = ("kernel_size", "acs_size", "eigen_threshold", "maps_path")


def _read(container: object, key: str) -> object | None:
    """Read ``key`` from a mapping *or* an attribute-bearing config node.

    The estimation block reaches this module as a Pydantic sub-model from the
    training strategies and as a plain ``dict`` from the inference strategies,
    which build themselves from a ``model_dump``.  One resolver has to accept
    both or the knob is honored on exactly one of the two paths — which is the
    defect this helper exists to close.
    """
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def resolve_estimation_settings(
    config: object, *, default: str = "power_iter"
) -> tuple[str, dict[str, object]]:
    """Resolve ``physics.coil_processing.estimation`` into ``estimate_smaps`` args.

    Returns ``(method, kwargs)`` ready to splat into :func:`estimate_smaps`.

    Every consumer that reaches for runtime sensitivity maps must go through
    here: an arm that declares ``method: espirit`` and gets ``power_iter``
    because one call site hardcoded a primitive is an advertised-but-unread
    knob (CLAUDE.md non-negotiable 8 / pitfall #15), and the resulting maps
    differ between the path that trained the network and the path that samples
    it.

    A configured ``"none"`` collapses to ``default`` rather than propagating,
    because every caller of this resolver is on a branch that *requires* maps —
    returning ``None`` there would degrade silently instead of conditioning.
    Sub-knobs left unset are omitted entirely so ``estimate_smaps``' own
    defaults apply.

    Args:
        config: The run config, or any node from which ``physics`` is reachable.
            Mappings and attribute-bearing objects are both accepted, and a
            missing block yields ``(default, {})``.
        default: Method to use when the block is absent or says ``"none"``.

    Returns:
        ``(method, kwargs)`` — ``kwargs`` holds only the sub-knobs actually set.
    """
    physics = _read(config, "physics")
    coil_processing = _read(physics, "coil_processing") if physics is not None else None
    estimation = _read(coil_processing, "estimation") if coil_processing is not None else None
    if estimation is None:
        return default, {}

    method = _read(estimation, "method")
    resolved = str(method) if method and method != "none" else default

    kwargs: dict[str, object] = {}
    for key in _ESTIMATION_SUB_KNOBS:
        value = _read(estimation, key)
        if value is not None:
            kwargs[key] = value
    return resolved, kwargs


def sense_gfactor_map(
    smaps: torch.Tensor,
    accel: int,
    *,
    axis: int = -2,
    noise_cov: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-pixel SENSE g-factor for uniform R along ``axis`` (phase-encode).

    ``g[p] = sqrt( (E^H Psi^-1 E)^-1[p,p] * (E^H Psi^-1 E)[p,p] )`` over the R
    folded pixels. Distinct, well-conditioned maps -> g~1; degenerate maps
    (identical across the folded set) -> g blows up = coil maps carry aliasing.

    Args:
        smaps: Complex maps ``(B, C, H, W)`` (RSS-normalized recommended).
        accel: Integer acceleration R along ``axis``. ``R<=1`` -> all-ones.
        axis: Phase-encode spatial axis (``-2``=H default, ``-1``=W).
        noise_cov: Optional ``(C, C)`` noise covariance Psi (default I).

    Returns:
        Real g-factor map ``(B, 1, H, W)``.
    """
    if smaps.dim() != 4:
        raise ValueError(f"Expected 4D smaps (B, C, H, W), got {smaps.dim()}D.")
    r = int(accel)
    b, _c, h, w = smaps.shape
    if r <= 1:
        return torch.ones(b, 1, h, w, device=smaps.device, dtype=torch.float32)
    if axis in (-1, 3):
        g = sense_gfactor_map(smaps.transpose(-1, -2).contiguous(), r, axis=-2, noise_cov=noise_cov)
        return g.transpose(-1, -2).contiguous()
    if axis not in (-2, 2):
        raise ValueError(f"axis must be a spatial axis (-2/-1), got {axis}.")
    if h % r != 0:
        raise ValueError(f"PE dim H={h} not divisible by accel R={r}.")
    hr = h // r
    idx = (
        torch.arange(hr, device=smaps.device)[:, None]
        + torch.arange(r, device=smaps.device)[None, :] * hr
    )  # (Hr, R) — the folded source rows per reduced pixel
    fold = smaps[:, :, idx, :].permute(0, 2, 4, 1, 3)  # (B, Hr, W, C, R)
    eh = fold.conj().transpose(-2, -1)  # (B, Hr, W, R, C)
    if noise_cov is not None:
        eh = eh @ torch.linalg.inv(noise_cov).to(eh.dtype)
    a = eh @ fold  # (B, Hr, W, R, R), Hermitian PSD
    # Invert via eigendecomposition with an eigenvalue floor — robust to the
    # singular (degenerate-map) case where LU hits a zero pivot; diag(A⁻¹)_kk =
    # Σ_j |V_kj|²/λ_j. Background pixels (diag_a≈0) → g≈0 (no signal, undefined).
    eigvals, eigvecs = torch.linalg.eigh(a)
    inv_eigvals = 1.0 / torch.clamp(eigvals, min=1e-8)
    diag_ai = (eigvecs.abs() ** 2 * inv_eigvals.unsqueeze(-2)).sum(dim=-1)
    diag_a = torch.diagonal(a, dim1=-2, dim2=-1).real
    g = torch.sqrt(torch.clamp(diag_ai * diag_a, min=0.0))  # (B, Hr, W, R)
    gmap = torch.zeros(b, h, w, device=smaps.device, dtype=torch.float32)
    gmap[:, idx.reshape(-1), :] = g.permute(0, 1, 3, 2).reshape(b, hr * r, w)
    return gmap.unsqueeze(1)


# ---------------------------------------------------------------------------
# S-map conditioning for k-space networks
# ---------------------------------------------------------------------------

#: Ceiling on the conditioning peak, as a multiple of the peak of the k-space
#: half it is concatenated with.  ``fft2c`` concentrates a smooth sensitivity map
#: into very few low-frequency bins, so the *peak* grows even though Parseval
#: fixes the RMS.  Note the RMS match is scale-invariant, so a globally huge map
#: is already neutralised by it — what this clamp bounds is *spectral
#: concentration*, the residual way a map can dominate the gradient.
#:
#: Measured on 4-coil analytic phantoms, sweeping resolution (64, 256), map
#: phase (present/absent) and acceleration (R = 1, 4, 8) -- 24 configurations.
#: Realistic smooth maps land at **1.02-1.33x** the reference peak, tightly
#: clustered and resolution-invariant (reference and map peaks scale together).
#: Only a perfectly uniform map -- one carrying no spatial encoding at all --
#: reaches 1.72-2.26x. 2.0 therefore leaves ~1.5x headroom over every realistic
#: map while still binding on a degenerate one, which is what a guard should do:
#: a ceiling that engages on every step is a distortion, not a guard.
#: Deliberately a documented constant and not a config knob: an unread/unstamped
#: knob would violate CLAUDE.md #15, and this is an invariant ("conditioning may
#: not dominate the data it conditions"), not a tuning dial.
SMAP_KSPACE_PEAK_RATIO = 2.0


def _interleaved_to_complex(x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """View a real tensor laid out ``[R1, I1, R2, I2, ...]`` as complex.

    Args:
        x: Real tensor whose ``channel_dim`` interleaves real/imaginary parts.
        channel_dim: Axis holding the ``2 * C`` interleaved channels.

    Returns:
        Complex tensor with ``C`` channels on ``channel_dim``.

    Raises:
        ValueError: If ``channel_dim`` does not have an even extent.
    """
    n = x.shape[channel_dim]
    if n % 2 != 0:
        raise ValueError(
            f"Real-interleaved S-maps need an even channel count on dim "
            f"{channel_dim}, got {n} (shape {tuple(x.shape)})."
        )
    x = x.unflatten(channel_dim, (n // 2, 2))
    x = x.movedim(channel_dim + 1, -1).contiguous()
    return torch.view_as_complex(x)


def _complex_to_interleaved(z: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Inverse of :func:`_interleaved_to_complex`.

    Args:
        z: Complex tensor.
        channel_dim: Axis whose ``C`` channels become ``2 * C`` interleaved.

    Returns:
        Real tensor laid out ``[R1, I1, R2, I2, ...]`` on ``channel_dim``.
    """
    x = torch.view_as_real(z)
    x = x.movedim(-1, channel_dim + 1).contiguous()
    return x.flatten(channel_dim, channel_dim + 1)


def _as_complex_field(x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Return ``x`` as a complex field regardless of its storage layout."""
    return x if torch.is_complex(x) else _interleaved_to_complex(x, channel_dim)


def _as_reference_field(x: torch.Tensor, channel_dim: int) -> torch.Tensor:
    """Return ``x`` in a form whose ``.abs()`` is its physical magnitude.

    Unlike :func:`_as_complex_field` this never raises on an odd channel count.
    The reference is read for statistics only — an RMS and a peak — and is never
    converted back to its original layout, so a real tensor that cannot be an
    interleaved complex field is legitimately a **real** field with zero
    imaginary part: ``|x|`` and ``sqrt(mean(x**2))`` are the same quantities in
    the same units either way.  Single-channel k-space arms (``in_channels: 1``)
    reach this branch; the S-maps themselves keep the strict even-count
    requirement, because they *are* round-tripped.

    Args:
        x: Reference tensor, complex or real.
        channel_dim: Axis carrying the coil channels.

    Returns:
        ``x`` as complex when its layout admits it, else ``x`` unchanged.
    """
    if torch.is_complex(x):
        return x
    if x.shape[channel_dim] % 2 == 0:
        return _interleaved_to_complex(x, channel_dim)
    return x


def _per_sample(values: torch.Tensor, ndim: int) -> torch.Tensor:
    """Reshape a ``[B]`` reduction so it broadcasts against a ``ndim``-D tensor."""
    return values.reshape(-1, *([1] * (ndim - 1)))


def prepare_smaps_for_kspace_conditioning(
    smaps: torch.Tensor,
    reference: torch.Tensor,
    *,
    channel_dim: int = 1,
    peak_ratio: float = SMAP_KSPACE_PEAK_RATIO,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make image-domain S-maps safe to concatenate onto a k-space model input.

    Coil sensitivity maps are estimated and stored **in image space**, but the
    k-space cold-diffusion networks concatenate them channel-wise onto a tensor
    that is **k-space**, then apply a single domain transform to the whole
    stack.  Either setting of ``force_pure_kspace`` therefore mistreats one half
    of the stack, and a convolution can only relate channels sharing an index —
    so an image-space sensitivity at pixel ``(x, y)`` gets aligned with the
    spatial frequency ``(kx, ky) = (x, y)``.  That correspondence is meaningless
    and destroys exactly the spatial encoding the maps exist to supply.

    Three things happen here, in order:

    1. **Domain.**  ``fft2c`` (never raw ``torch.fft``; CLAUDE.md #2) moves the
       maps into k-space.  This is not merely consistency: multiplication in
       image space is convolution in k-space, so S-map spectra *are* the
       GRAPPA-style interpolation kernels a k-space network can act on.
    2. **Level.**  Per sample, the maps are rescaled so their RMS matches the
       reference half.  ``norm="ortho"`` makes ``fft2c`` unitary (Parseval), so
       the *RMS* is unchanged by step 1 — but unit-magnitude maps still sit
       roughly an order of magnitude above the k-space periphery, which is
       where nearly all of the plane lives.
    3. **Amplitude.**  Per sample, a phase-preserving magnitude clamp caps the
       conditioning at ``peak_ratio`` times the reference peak, so a
       pathological map cannot dominate the gradient.

    ``reference`` is the k-space half the maps are concatenated with — the only
    tensor available identically at train, validation and inference time, and
    the one "homogeneous levels" is actually about.  A consequence worth
    knowing: under cold diffusion the reference loses energy as ``t`` grows, so
    the conditioning is scaled down in step with it, and at a fully-masked
    ``t = T`` it goes to zero along with the data.

    Args:
        smaps: Sensitivity maps in **image space**, complex or real-interleaved
            (``[R1, I1, R2, I2, ...]`` on ``channel_dim``).
        reference: The k-space tensor the maps will be concatenated onto, in
            either layout.  Read for statistics only, so an odd real channel
            count is accepted here (it is a real field) even though the same
            layout is rejected for ``smaps``.
        channel_dim: Axis carrying the coil channels (1 for ``[B, C, H, W]``,
            2 for ``[B, D, C, H, W]``).
        peak_ratio: Clamp ceiling as a multiple of the reference peak.
        eps: Division guard.

    Returns:
        ``(prepared, scale)`` — ``prepared`` has the dtype, layout and shape of
        ``smaps``; ``scale`` is the per-sample RMS gain applied in step 2, a
        ``[B]`` tensor left **on device** so callers can stamp it into
        provenance without an ``.item()`` sync (CLAUDE.md #9).

    Raises:
        ValueError: If ``peak_ratio`` is not positive, or ``smaps`` is
            real-interleaved with an odd channel count.
    """
    if peak_ratio <= 0:
        raise ValueError(f"peak_ratio must be > 0, got {peak_ratio}.")

    from .fft_ops import fft2c

    was_complex = torch.is_complex(smaps)
    orig_dtype = smaps.dtype

    smaps_c = _as_complex_field(smaps, channel_dim)
    ref_c = _as_reference_field(reference, channel_dim)

    # 1. Domain: image space -> k-space (centred, orthonormal).
    smaps_k = fft2c(smaps_c)

    # 2. Level: per-sample RMS match against the half it rides next to.
    ref_rms = ref_c.abs().pow(2).flatten(1).mean(dim=1).sqrt()
    smap_rms = smaps_k.abs().pow(2).flatten(1).mean(dim=1).sqrt()
    scale = ref_rms / (smap_rms + eps)
    smaps_k = smaps_k * _per_sample(scale, smaps_k.dim()).to(smaps_k.dtype)

    # 3. Amplitude: phase-preserving magnitude clamp against the reference peak.
    cap = _per_sample(peak_ratio * ref_c.abs().flatten(1).amax(dim=1), smaps_k.dim())
    attenuation = torch.clamp(cap / (smaps_k.abs() + eps), max=1.0)
    smaps_k = smaps_k * attenuation.to(smaps_k.dtype)

    if was_complex:
        prepared = smaps_k.to(orig_dtype)
    else:
        prepared = _complex_to_interleaved(smaps_k, channel_dim).to(orig_dtype)
    return prepared, scale


__all__ = [
    "SMAP_KSPACE_PEAK_RATIO",
    "coil_combine_rss",
    "coil_combine_sense",
    "confine_to_acs",
    "create_synthetic_csm",
    "espirit_min_acs_size",
    "estimate_acs_hanning_csm",
    "estimate_csm_espirit",
    "estimate_csm_pinn",
    "estimate_csm_power_iter",
    "estimate_csm_rss",
    "estimate_smaps",
    "estimate_smaps_calibrated",
    "extract_acs_region",
    "load_csm_from_file",
    "prepare_smaps_for_kspace_conditioning",
    "resolve_estimation_settings",
    "sense_gfactor_map",
    "validate_csm",
]
