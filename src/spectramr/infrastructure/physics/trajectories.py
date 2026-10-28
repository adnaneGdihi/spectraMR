"""Trajectory Generation for Non-Cartesian MRI Reconstruction.

This module provides analytic coordinate generators for Non-Uniform FFT (NUFFT)
workflows. Unlike Cartesian masks (binary grids), these generate continuous
float coordinates in [-π, π] for NUFFT operators.

Binary Cartesian masks live in ``sampling.py``; this module owns the continuous
coordinates.

References:
- Jackson et al., "Selection of a convolution function for Fourier inversion" TMI 1991
- Pipe & Menon, "Sampling Density Compensation in MRI" MRM 1999
"""

import math
from typing import Final, Literal

import torch

#: SSOT for the trajectory vocabulary (#1097): consumers import these, so a name cannot
#: be ACCEPTED without being ROUTABLE -- drift between those two sets was the bug.
#: ``cartesian`` is legal in YAML but absent here: it is the ABSENCE of a trajectory,
#: served by the Cartesian mask path rather than by NUFFT.
NON_CARTESIAN_TRAJECTORIES: Final[tuple[str, ...]] = ("radial", "golden_angle", "spiral", "epi")
TRAJECTORY_TYPES: Final[tuple[str, ...]] = ("cartesian", *NON_CARTESIAN_TRAJECTORIES)

#: Full-circle golden angle pi*(3 - sqrt(5)) ~= 137.508 deg between spiral interleaves, which
#: span 2*pi. Radial spokes span pi and take pi*(sqrt(5) - 1)/2 ~= 111.246 deg; each constant
#: is wrong for the other family.
SPIRAL_GOLDEN_ANGLE: Final[float] = math.pi * (3.0 - math.sqrt(5.0))


class TrajectoryFactory:
    """Factory for generating Non-Cartesian k-space trajectories.

    All methods return trajectories in NUFFT-compatible format:
    - Coordinates in range [-π, π]
    - Shape: (2, N) where row 0 = kx, row 1 = ky
    - Density Compensation Function (DCF) for adjoint normalization

    Example:
        >>> traj, dcf = TrajectoryFactory.get_radial_trajectory((256, 256), num_spokes=128)
        >>> traj.shape  # (2, 32768) for 256 samples per spoke
        >>> dcf.shape   # (32768,)
    """

    @staticmethod
    def get_radial_trajectory(
        im_size: tuple[int, int] = (320, 320),
        num_spokes: int = 128,
        samples_per_spoke: int | None = None,
        golden_angle: bool = False,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate radial (spoke-based) trajectory.

        Args:
            im_size: Image dimensions (H, W)
            num_spokes: Number of radial spokes
            samples_per_spoke: Samples per spoke (default: max(im_size) for Nyquist)
            golden_angle: If True, use golden angle increment (~111.25°)
            device: Target device

        Returns:
            Tuple of (trajectory, dcf):
                - trajectory: (2, N) tensor in [-π, π]
                - dcf: (N,) density compensation weights
        """
        N = max(im_size)
        if samples_per_spoke is None:
            samples_per_spoke = N

        # Spoke angles
        if golden_angle:
            # Radial-MRI golden angle: π·(√5 − 1)/2 ≈ 111.246° (Winkelmann 2007),
            # NOT the 2-D phyllotaxis angle π·(3 − √5) ≈ 137.508°.
            ga = torch.pi * (5**0.5 - 1) / 2
            angles = torch.arange(num_spokes, device=device) * ga
        else:
            # Uniform angles
            angles = torch.arange(num_spokes, device=device) * torch.pi / num_spokes

        # Radial samples from -0.5 to 0.5 (normalized k-space)
        r = torch.linspace(-0.5, 0.5, samples_per_spoke, device=device)

        # Outer product: (spokes, samples)
        kx = r.unsqueeze(0) * torch.cos(angles.unsqueeze(1))
        ky = r.unsqueeze(0) * torch.sin(angles.unsqueeze(1))

        # Flatten and scale to [-π, π]
        kx = kx.flatten() * 2 * torch.pi
        ky = ky.flatten() * 2 * torch.pi

        trajectory = torch.stack([kx, ky], dim=0)

        # DCF: Ram-Lak (proportional to |r|, avoid division by zero at DC)
        r_flat = torch.abs(r).repeat(num_spokes)
        dcf = r_flat / (r_flat.sum() + 1e-8) * r_flat.numel()
        dcf[dcf == 0] = dcf[dcf > 0].min()  # Avoid zero weight at DC

        return trajectory, dcf.to(device)

    @staticmethod
    def get_spiral_trajectory(
        im_size: tuple[int, int] = (320, 320),
        num_arms: int = 16,
        num_turns: float = 4.0,
        samples_per_arm: int | None = None,
        variable_density: bool = True,
        max_grad: float = 40.0,  # mT/m
        max_slew: float = 150.0,  # T/m/s
        dt: float = 4e-6,  # Dwell time (4µs)
        device: torch.device = torch.device("cpu"),
        golden_angle: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Archimedean spiral trajectory with slew-rate limiting.

        [PHYSICS FIX] Implements Glover's algorithm for hardware-valid spirals.
        At t=0 (k-space center), naive spirals require infinite slew rate.
        This implementation uses a piecewise trajectory:
        1. Transition region (constant slew rate) near center
        2. Constant amplitude region (max gradient) for periphery

        Args:
            im_size: Image dimensions (H, W)
            num_arms: Number of spiral interleaves
            num_turns: Number of turns from center to edge
            samples_per_arm: Samples per arm (default: 2 * max(im_size))
            variable_density: If True, denser sampling at center
            max_grad: Maximum gradient amplitude (mT/m)
            max_slew: Maximum slew rate (T/m/s)
            dt: Dwell time (seconds)
            device: Target device
            golden_angle: Rotate interleave ``a`` by ``a * SPIRAL_GOLDEN_ANGLE`` rather than
                ``2*pi*a / num_arms``, so the first ``k`` arms do not depend on ``num_arms``.

        Returns:
            Tuple of (trajectory, dcf)

        Reference: Glover, G. H. (1999). "Simple analytic spiral K-space algorithm." MRM.
        """
        N = max(im_size)
        if samples_per_arm is None:
            samples_per_arm = 2 * N

        # Slew-limited to gradient-limited transition time, T_s = G_max / S_max (seconds).
        T_s = (max_grad * 1e-3) / max_slew  # Convert mT/m to T/m

        # Number of samples in transition region
        n_transition = int(T_s / dt)
        n_transition = max(1, min(n_transition, samples_per_arm // 4))  # Cap at 25% of arm

        t = torch.linspace(0, 1, samples_per_arm, device=device)

        # Piecewise radius: r ~ t^2 while slew-limited, then r ~ t while gradient-limited.
        t_transition = n_transition / samples_per_arm

        if variable_density:
            r = torch.where(
                t < t_transition,
                # Quadratic growth (constant slew) - smoother start
                (t / t_transition) ** 2 * t_transition * 0.5,
                # Linear growth (constant grad) for remaining
                t_transition * 0.5
                + (t - t_transition) / (1 - t_transition) * (0.5 - t_transition * 0.5),
            )
        else:
            r = t * 0.5

        theta = t * 2 * torch.pi * num_turns

        all_kx, all_ky, all_dcf = [], [], []

        for arm in range(num_arms):
            arm_angle = arm * SPIRAL_GOLDEN_ANGLE if golden_angle else arm * 2 * torch.pi / num_arms
            kx = r * torch.cos(theta + arm_angle)
            ky = r * torch.sin(theta + arm_angle)
            all_kx.append(kx)
            all_ky.append(ky)

            # DCF for spiral: proportional to radial distance
            dcf_arm = r.clone()
            dcf_arm[dcf_arm == 0] = dcf_arm[dcf_arm > 0].min() if (dcf_arm > 0).any() else 1e-6
            all_dcf.append(dcf_arm)

        # Stack and scale to [-π, π]
        kx = torch.cat(all_kx) * 2 * torch.pi
        ky = torch.cat(all_ky) * 2 * torch.pi
        trajectory = torch.stack([kx, ky], dim=0)

        dcf = torch.cat(all_dcf)
        dcf = dcf / dcf.sum() * dcf.numel()

        return trajectory, dcf

    @staticmethod
    def get_epi_trajectory(
        im_size: tuple[int, int] = (320, 320),
        acceleration: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate Echo-Planar Imaging (EPI) trajectory.

        EPI is Cartesian but with specific readout ordering.
        Useful for distortion correction models.

        Args:
            im_size: Image dimensions (H, W)
            acceleration: Acceleration factor >= 1; num_pe = floor(H / acceleration).
            device: Target device

        Returns:
            Tuple of (trajectory, dcf)
        """
        H, W = im_size
        if acceleration < 1:
            raise ValueError(f"acceleration must be >= 1, got {acceleration}")
        num_pe = H // acceleration  # Phase encoding lines (floor of H / acceleration)

        kx_line = torch.linspace(-0.5, 0.5, W, device=device)
        ky_lines = torch.linspace(-0.5, 0.5, num_pe, device=device)

        all_kx = []
        all_ky = []

        for i, ky in enumerate(ky_lines):
            if i % 2 == 0:
                all_kx.append(kx_line)
            else:
                all_kx.append(kx_line.flip(0))  # Alternating readout direction
            all_ky.append(torch.full((W,), ky, device=device))

        kx = torch.cat(all_kx) * 2 * torch.pi
        ky = torch.cat(all_ky) * 2 * torch.pi
        trajectory = torch.stack([kx, ky], dim=0)

        # Uniform DCF for Cartesian
        dcf = torch.ones(trajectory.shape[1], device=device)

        return trajectory, dcf


def get_trajectory(
    trajectory_type: Literal["radial", "golden_angle", "spiral", "epi"] = "radial",
    im_size: tuple[int, int] = (320, 320),
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience function to get trajectory by name.

    Args:
        trajectory_type: Type of trajectory
        im_size: Image dimensions
        **kwargs: Additional arguments for specific trajectory

    Returns:
        Tuple of (trajectory, dcf)
    """
    if trajectory_type == "radial":
        return TrajectoryFactory.get_radial_trajectory(im_size, **kwargs)
    elif trajectory_type == "golden_angle":
        return TrajectoryFactory.get_radial_trajectory(im_size, golden_angle=True, **kwargs)
    elif trajectory_type == "spiral":
        return TrajectoryFactory.get_spiral_trajectory(im_size, **kwargs)
    elif trajectory_type == "epi":
        return TrajectoryFactory.get_epi_trajectory(im_size, **kwargs)
    else:
        # ``cartesian`` lands here deliberately: it is a legal `data.trajectory` but not
        # a NUFFT family, so asking for it here is a caller bug, not a missing branch.
        raise ValueError(f"Unknown trajectory type: {trajectory_type}")


def compute_dcf(
    trajectory: torch.Tensor,
    im_size: tuple[int, int] = (256, 256),
    grid_size: tuple[int, int] | None = None,
    num_iterations: int = 10,
) -> torch.Tensor:
    """Compute Density Compensation Function (DCF) using Pipe's iterative method.

    Args:
        trajectory: Non-Cartesian trajectory [2, N] (normalized to [-pi, pi])
        im_size: Image size (H, W)
        grid_size: Oversampled grid size (default: 2*im_size)
        num_iterations: Number of iterations for Pipe's method

    Returns:
        dcf: Density compensation weights [N]
    """
    try:
        import torchkbnufft as tkbn
    except ImportError:
        raise ImportError("torchkbnufft is required for iteratively computing DCF.")

    dcf = tkbn.calc_density_compensation_function(
        traj=trajectory,
        im_size=im_size,
        grid_size=grid_size,
        num_iterations=num_iterations,
    )
    return dcf.squeeze()
