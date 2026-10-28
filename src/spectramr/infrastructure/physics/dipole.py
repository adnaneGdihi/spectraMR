from __future__ import annotations

import torch

from spectramr.infrastructure.physics.fft_ops import centred_freqs


def generate_kspace_grid_3d(
    shape: tuple[int, ...],
    voxel_sizes: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates 3D k-space coordinates.

    Args:
        shape: (B, C, D, H, W) or (D, H, W)
        voxel_sizes: tuple of (dz, dy, dx) in mm
    """
    if len(shape) == 5:
        D, H, W = shape[2], shape[3], shape[4]
    else:
        D, H, W = shape

    dz, dy, dx = voxel_sizes

    # Generate frequencies using torch.fft.fftfreq
    kz = torch.fft.fftfreq(D, d=dz)
    ky = torch.fft.fftfreq(H, d=dy)
    kx = torch.fft.fftfreq(W, d=dx)

    # Create meshgrid
    kz_grid, ky_grid, kx_grid = torch.meshgrid(kz, ky, kx, indexing="ij")

    return kx_grid, ky_grid, kz_grid


def get_dipole_kernel_3d(
    shape: tuple[int, ...],
    voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Return the 3D unit dipole kernel D(k) = 1/3 - kz² / |k|².

    This is the reusable kernel shared by forward dipole convolution
    and QSM dipole inversion.

    Args:
        shape: (B, C, D, H, W) or (D, H, W)
        voxel_sizes: tuple of (dz, dy, dx) in mm

    Returns:
        Real-valued dipole kernel tensor of shape (D, H, W),
        expanded to match the number of leading batch/channel
        dimensions in *shape*.
    """
    kx, ky, kz = generate_kspace_grid_3d(shape, voxel_sizes)

    k_sq = kx**2 + ky**2 + kz**2 + 1e-8
    dipole_kernel = (1.0 / 3.0) - (kz**2 / k_sq)

    # Remove singularity at k=0 (DC component)
    dipole_kernel[k_sq < 1e-7] = 0.0

    # Expand to match batch and channel dims if needed
    ndim = len(shape) if isinstance(shape, (tuple, list)) else shape
    if isinstance(shape, (tuple, list)):
        while dipole_kernel.dim() < len(shape):
            dipole_kernel = dipole_kernel.unsqueeze(0)

    return dipole_kernel


def dipole_kernel_centred(
    shape: tuple[int, int, int],
    b0_dim: int,
    *,
    voxel_mm: tuple[float, float, float] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unit dipole kernel ``D(k) = 1/3 - k_b0^2 / |k|^2`` on the ``fftnc`` grid (DC at ``N // 2``).

    The sign of :func:`get_dipole_kernel_3d`, laid out for centred spectra, with ``B0`` along
    ``b0_dim`` of ``(D, H, W)`` (-3, -2 or -1). ``D(0) = 0``: a uniform offset is the receiver's
    centre frequency. ``voxel_mm`` makes ``k`` physical (cycles/mm), so anisotropic voxels keep
    the dipole's angular shape; ``None`` uses cycles/px on every axis.
    """
    if len(shape) != 3 or any(n < 1 for n in shape):
        raise ValueError(f"shape must be three positive sizes (D, H, W); got {shape!r}")
    if b0_dim not in (-3, -2, -1):
        raise ValueError(f"b0_dim must be -3, -2 or -1 of (D, H, W); got {b0_dim!r}")
    spacing = (1.0, 1.0, 1.0) if voxel_mm is None else voxel_mm
    if len(spacing) != 3 or any(v <= 0 for v in spacing):
        raise ValueError(f"voxel_mm must be three positive sizes; got {voxel_mm!r}")
    axes = [
        centred_freqs(n, device=device, dtype=dtype) / v
        for n, v in zip(shape, spacing, strict=True)
    ]
    k = torch.meshgrid(*axes, indexing="ij")
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    kernel = 1.0 / 3.0 - k[b0_dim + 3] ** 2 / k2.clamp_min(torch.finfo(dtype).tiny)
    return torch.where(k2 > 0, kernel, torch.zeros_like(kernel))


def apply_3d_dipole_kernel(
    chi_map_3d: torch.Tensor,
    voxel_sizes: tuple[float, float, float] = (1.0, 1.0, 1.0),
    pad_z: int = 0,
) -> torch.Tensor:
    """Applies the 3D Dipole Kernel to a susceptibility map in k-space.

    The magnetic dipole field D(k) = 1/3 - kz²/|k|² has infinite spatial
    support. On truncated 2.5D slabs (e.g., D=16), the discrete FFT enforces
    periodic boundary conditions, causing the 1/r³ field to wrap around the
    z-axis and generate massive phase shading artifacts.

    Zero-padding the z-axis by ≥1.5×D isolates the circular convolution
    boundary, yielding physically accurate ΔB₀ fields per Maxwell's equations.

    Reference:
        Salomir et al., "A fast calculation method for magnetic field
        inhomogeneity due to an arbitrary distribution of bulk susceptibility,"
        Concepts in Magnetic Resonance Part B, 2003.

    Args:
        chi_map_3d: Susceptibility map of shape [B, 1, D, H, W]
        voxel_sizes: tuple of (dz, dy, dx) in mm
        pad_z: Number of zero-padding slices on each side of z-axis.
            Recommended: ≥1.5×D (e.g., pad_z=24 for D=16).
            When 0, no padding is applied (original behavior).

    Returns:
        Delta B0 field of the same shape as chi_map_3d.
    """
    import torch.nn.functional as F

    # 1. Zero-pad along z to enforce linear (non-circular) convolution
    if pad_z > 0:
        # F.pad format: (W_left, W_right, H_left, H_right, D_left, D_right)
        chi_padded = F.pad(chi_map_3d, (0, 0, 0, 0, pad_z, pad_z), mode="constant", value=0.0)
    else:
        chi_padded = chi_map_3d

    # 2. Generate dipole kernel on the padded grid
    dipole_kernel = get_dipole_kernel_3d(chi_padded.shape, voxel_sizes).to(chi_padded.device)

    # 3. Forward physics: ΔB₀ = χ ⊛ d (convolution = multiplication in k-space)
    chi_k = torch.fft.fftn(chi_padded, dim=(-3, -2, -1))
    field_k = chi_k * dipole_kernel
    delta_b0_padded = torch.fft.ifftn(field_k, dim=(-3, -2, -1)).real

    # 4. Crop back to original z extent
    if pad_z > 0:
        return delta_b0_padded[..., pad_z:-pad_z, :, :]
    return delta_b0_padded
