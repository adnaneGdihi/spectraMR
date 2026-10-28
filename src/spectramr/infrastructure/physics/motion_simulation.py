"""Motion Simulation for MRI.

This module provides tools for simulating physiological motion, including
rigid body motion and elastic respiratory deformations with hysteresis.
"""

import math

import torch
import torch.nn.functional as F

__all__ = [
    "RESPIRATORY_HYSTERESIS",
    "apply_elastic_respiratory_motion",
    "respiratory_trace",
    "respiratory_trace_peak",
    "simulate_elastic_motion_ffd",
]

#: Inhale/exhale asymmetry ``a`` of the breathing trace ``cos^2(n wt / 2) exp(-a sin(wt))``.
RESPIRATORY_HYSTERESIS = 0.2
_BREATHS = (1, 2)


def respiratory_trace(
    time_points: torch.Tensor,
    cycle_period: float,
    hysteresis: float = RESPIRATORY_HYSTERESIS,
    *,
    breaths_per_cycle: int = 2,
) -> torch.Tensor:
    """Asymmetric breathing trace ``cos^2(n wt / 2) exp(-a sin(wt))``, ``w = 2 pi / cycle_period``.

    ``n = breaths_per_cycle`` lobes per ``cycle_period``. The default 2 is the FFD simulator's
    trace (two lobes, skewed in opposite directions); ``n = 1`` is one breath per period with a
    consistent inhale/exhale skew. Non-negative; its maximum is :func:`respiratory_trace_peak`.
    """
    if breaths_per_cycle not in _BREATHS:
        raise ValueError(f"breaths_per_cycle must be one of {_BREATHS}; got {breaths_per_cycle!r}")
    phase = 2 * torch.pi / cycle_period * time_points
    lobe = phase if breaths_per_cycle == 2 else phase / 2
    return torch.cos(lobe) ** 2 * torch.exp(-hysteresis * torch.sin(phase))


def _peak_two_lobes(a: float) -> float:
    # d/dphi = 0 away from cos = 0: a s^2 - 2 s - a = 0, s = sin(phi).
    s = (1.0 - math.sqrt(1.0 + a * a)) / a
    return (1.0 - s * s) * math.exp(-a * s)


def _peak_one_lobe(a: float) -> float:
    # f = (1 + cos phi) / 2 exp(-a sin phi); f' = 0 is g = sin phi + a cos phi (1 + cos phi) = 0.
    phi = -a
    for _ in range(50):
        g = math.sin(phi) + a * math.cos(phi) * (1.0 + math.cos(phi))
        dg = math.cos(phi) - a * math.sin(phi) * (1.0 + 2.0 * math.cos(phi))
        phi -= g / dg
    return 0.5 * (1.0 + math.cos(phi)) * math.exp(-a * math.sin(phi))


_PEAKS = {1: _peak_one_lobe, 2: _peak_two_lobes}


def respiratory_trace_peak(
    hysteresis: float = RESPIRATORY_HYSTERESIS, *, breaths_per_cycle: int = 2
) -> float:
    """Maximum of :func:`respiratory_trace` (slightly above 1 for ``hysteresis > 0``)."""
    if breaths_per_cycle not in _PEAKS:
        raise ValueError(f"breaths_per_cycle must be one of {_BREATHS}; got {breaths_per_cycle!r}")
    a = float(hysteresis)
    return 1.0 if a == 0.0 else _PEAKS[breaths_per_cycle](a)


def simulate_elastic_motion_ffd(
    volume: torch.Tensor,
    time_points: torch.Tensor,
    cycle_period: float = 4.0,
    max_displacement: float = 5.0,
    grid_spacing: int = 16,
    seed: int | None = None,
) -> torch.Tensor:
    """Simulates elastic soft-tissue motion using B-Spline Free-Form Deformation (FFD).

    Generates a dense deformation field by upsampling quasi-periodic perturbations
    from a coarse control point grid.

    Args:
        volume: [B, C, D, H, W]
        time_points: [N_shots] time in seconds
        cycle_period: Respiratory cycle length (s)
        max_displacement: Max voxel displacement
        grid_spacing: Spacing of control points (pixels)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        Deformed k-space data as a COMPLEX tensor of shape
        ``[N_shots, B, C, D, H, W]`` (``torch.stack`` of per-shot
        ``torch.fft.fftn`` outputs) — NOT a real-stacked ``[..., 2]`` layout.
    """
    B, C, D, H, W = volume.shape
    device = volume.device

    # 1. Respiratory Trace (Hysteresis model)
    trace = respiratory_trace(time_points, cycle_period)

    # 2. Coarse Control Points
    # Grid size for control points (ensure at least 2x2x2 for interpolation)
    cp_D = max(2, D // grid_spacing)
    cp_H = max(2, H // grid_spacing)
    cp_W = max(2, W // grid_spacing)

    # Generate random control point flow field [B, 3, cp_D, cp_H, cp_W]
    # We want coherent motion, e.g. principal component along Superior-Inferior (Z)
    if seed is None:
        seed = 42  # Default seed for reproducibility
    rng = torch.Generator(device=device).manual_seed(seed)

    # Random perturbations
    noise = torch.randn(B, 3, cp_D, cp_H, cp_W, device=device, generator=rng)

    # Smoothen control points (low pass) to ensure physical plausibility
    # or just rely on spline upsampling.

    # Biased motion: Diaphragm moves mostly in Z (dim 0 in flow field? No, dim 2 is Z axis of grid?)
    # flow is [B, 3, ...]. 3 channels: dx, dy, dz.
    # Let's verify grid_sample coordinate system: (x, y, z).
    # dim 0 of flow: x displacement?
    # affine_grid returns (x, y, z). Flow should match.

    # Add strong Z bias (SI direction)
    noise[:, 2, :, :, :] += 2.0  # Bias +Z

    # Normalize to max_displacement
    # Handle potentially empty tensor if dimensions are degenerate (though max(2,...) prevents this)
    if noise.numel() > 0:
        max_val = noise.norm(dim=1, keepdim=True).max()
        if max_val > 0:
            noise = noise / (max_val + 1e-6)
    # Scale in pixels.
    # Convert pixels to normalized coordinates [-1, 1].
    # 2.0 range = D/H/W pixels.
    # Scale factor for each dim.

    # We apply the trace scaling later.

    kspace_data = []

    # Grid for sampling
    base_grid = F.affine_grid(
        torch.eye(3, 4, device=device).unsqueeze(0).repeat(B, 1, 1),
        [B, C, D, H, W],
        align_corners=False,
    )  # [B, D, H, W, 3]

    for t_idx, amp in enumerate(trace):
        # Current coarse flow
        current_cp_flow = noise * amp * max_displacement  # [B, 3, cpD, cpH, cpW] (pixels)

        # Upsample to dense flow field [B, 3, D, H, W] via trilinear interpolation (B-Spline approx)
        dense_flow = F.interpolate(
            current_cp_flow, size=(D, H, W), mode="trilinear", align_corners=False
        )

        # Convert pixel displacement to normalized [-1, 1]
        # flow_x_norm = flow_x_pix / (W/2)
        # flow_y_norm = flow_y_pix / (H/2)
        # flow_z_norm = flow_z_pix / (D/2)

        # dense_flow comes as [dx, dy, dz]?
        # CAUTION: interpolate preserves channel order.
        # grid_sample expects last dim to be (x, y, z).
        # We need to permute dense_flow to [B, D, H, W, 3]
        dense_flow_perm = dense_flow.permute(0, 2, 3, 4, 1)  # [B, D, H, W, 3]

        # Apply scaling
        scale = torch.tensor([2.0 / W, 2.0 / H, 2.0 / D], device=device).view(1, 1, 1, 1, 3)
        dense_flow_norm = dense_flow_perm * scale

        # Deform grid
        deformed_grid = base_grid + dense_flow_norm

        # Sample
        deformed_vol = F.grid_sample(
            volume,
            deformed_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )

        # FFT3D to k-space
        kspace_shot = torch.fft.fftn(deformed_vol, dim=(-3, -2, -1), norm="ortho")
        kspace_data.append(kspace_shot)

    return torch.stack(kspace_data)


# Alias for backward compatibility / test compliance
apply_elastic_respiratory_motion = simulate_elastic_motion_ffd
