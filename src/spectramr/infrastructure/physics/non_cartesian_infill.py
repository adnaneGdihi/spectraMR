"""Filling off-grid k-space: what is exact, what is interpolation, what is invention.

The anatomy-distortion risk in a learned k-space filler is not spread evenly. It
is concentrated in one place -- frequencies with no nearby measurement -- and the
distance to the nearest acquired sample is a property of the **trajectory alone**,
computable before any weight is trained. This module makes that geometry explicit
so the three regimes can be treated differently instead of averaged together.

**Exact.** For a real-valued object k-space is Hermitian, ``S(-k) = conj(S(k))``,
so every acquired sample hands you its antipode for free. This is the identity
partial-Fourier reconstruction has used clinically for decades (homodyne, POCS)
and it is not a prediction, so it cannot hallucinate. It is also not free: an MRI
object is complex (B0 off-resonance, coil phase, flow), and the symmetry holds
only up to that phase. :func:`hermitian_phase_violation` measures the breach on
the samples where the acquired and mirrored sets overlap, which turns the
assumption into a number the arm can report rather than one it has to trust.

**Interpolation.** Inside the sampled support, filling between measurements is
well-posed -- it is what GRAPPA and SPIRiT already do with linear kernels, and
what graph attention over k-nearest neighbours does with learned ones.

**Invention.** Beyond the support there is no measurement to interpolate between,
and a network asked for an answer will produce a confident one.
:func:`sample_density_gate` returns the geometric confidence that separates the
second regime from the third, so an arm can declare how far past its own sampling
it is willing to be believed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

#: Samples whose antipode already lies within this radius (radians) of an
#: acquired point add no coverage, so they are dropped rather than duplicating a
#: measurement and double-weighting it in the adjoint.
DEFAULT_MERGE_RADIUS = 1e-3


def hermitian_extend(
    trajectory: torch.Tensor,
    samples: torch.Tensor,
    *,
    merge_radius: float = DEFAULT_MERGE_RADIUS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mirror an acquisition through the k-space origin by conjugate symmetry.

    Args:
        trajectory: ``[2, N]`` coordinates in radians.
        samples: ``[B, C, N]`` complex measurements at those coordinates.
        merge_radius: Antipodes closer than this to an existing sample are
            dropped as redundant.

    Returns:
        ``(trajectory_ext, samples_ext, is_measured)`` where the first two are the
        acquired set concatenated with the surviving mirrored set, and
        ``is_measured`` is ``[N_ext]``, 1 on genuinely acquired samples and 0 on
        mirrored ones -- so a DC layer can hold the real measurements harder than
        the inferred ones.
    """
    if trajectory.dim() != 2 or trajectory.shape[0] != 2:
        raise ValueError(f"trajectory must be [2, N]; got {tuple(trajectory.shape)}.")
    if samples.shape[-1] != trajectory.shape[-1]:
        raise ValueError(
            f"samples cover {samples.shape[-1]} readouts, trajectory has {trajectory.shape[-1]}."
        )

    mirrored = -trajectory
    # Drop an antipode that lands on an existing sample: keeping both would put
    # two entries at one frequency and let the adjoint count it twice.
    dist = torch.cdist(mirrored.t().unsqueeze(0), trajectory.t().unsqueeze(0)).squeeze(0)
    keep = dist.min(dim=1).values > merge_radius

    traj_ext = torch.cat([trajectory, mirrored[:, keep]], dim=1)
    samples_ext = torch.cat([samples, samples[..., keep].conj()], dim=-1)
    is_measured = torch.cat(
        [
            torch.ones(trajectory.shape[-1], device=trajectory.device),
            torch.zeros(int(keep.sum()), device=trajectory.device),
        ]
    )
    return traj_ext, samples_ext, is_measured


def hermitian_phase_violation(
    trajectory: torch.Tensor,
    samples: torch.Tensor,
    *,
    merge_radius: float = 5e-2,
) -> torch.Tensor:
    """How badly this object breaks the symmetry :func:`hermitian_extend` assumes.

    Where the trajectory already samples both ``+k`` and ``-k`` -- true of every
    radial spoke through the origin -- the object itself says whether it is real:
    a real object satisfies ``S(-k) = conj(S(k))`` exactly, and the residual is
    the phase the assumption ignores.

    Returns a scalar relative violation in ``[0, inf)``. Near 0 the Hermitian
    half of the infill is free; large means the arm is extrapolating on an
    assumption its own data refutes, and should weight those samples down.
    Reported rather than acted on, because the threshold is an owner decision.
    """
    mirrored = -trajectory
    dist = torch.cdist(mirrored.t().unsqueeze(0), trajectory.t().unsqueeze(0)).squeeze(0)
    nearest = dist.min(dim=1)
    paired = nearest.values <= merge_radius
    if not bool(paired.any()):
        return torch.zeros((), device=samples.device)
    idx = nearest.indices[paired]
    lhs = samples[..., paired]
    rhs = samples[..., idx].conj()
    scale = lhs.abs().mean().clamp_min(1e-12)
    return (lhs - rhs).abs().mean() / scale


def sample_density_gate(
    trajectory: torch.Tensor,
    im_size: tuple[int, int],
    *,
    support_sigma: float = 1.5,
    floor: float = 0.0,
) -> torch.Tensor:
    """Per-frequency geometric confidence, in ``[floor, 1]``, on the output grid.

    Rasterises the trajectory onto the reconstruction grid, blurs it by
    ``support_sigma`` bins, and normalises. The result is high where the
    acquisition actually visited and falls to ``floor`` where it did not, so an
    arm can multiply its predicted k-space by this and have its extrapolation
    bounded by its own sampling rather than by whatever the network is confident
    about.

    Args:
        trajectory: ``[2, N]`` radians in ``[-pi, pi]``.
        im_size: Output grid ``(H, W)``.
        support_sigma: Blur width in grid bins. Roughly the radius, in bins,
            over which one sample is taken to support its neighbourhood.
        floor: Confidence assigned to entirely unvisited frequencies. ``0.0``
            forbids extrapolation outright; a small positive value lets a
            network reach past the support under an explicit, declared budget.

    Returns:
        ``[1, 1, H, W]`` real gate, ``fftshift``-centred to match ``fft2c``.
    """
    if not 0.0 <= floor <= 1.0:
        raise ValueError(f"floor must be in [0, 1], got {floor}.")
    height, width = int(im_size[0]), int(im_size[1])
    device = trajectory.device

    # radians [-pi, pi] -> centred bin indices
    ky = (trajectory[0] / torch.pi * 0.5 + 0.5) * (height - 1)
    kx = (trajectory[1] / torch.pi * 0.5 + 0.5) * (width - 1)
    iy = ky.round().long().clamp(0, height - 1)
    ix = kx.round().long().clamp(0, width - 1)

    counts = torch.zeros(height * width, device=device)
    counts.index_add_(0, iy * width + ix, torch.ones_like(ky, dtype=counts.dtype))
    counts = counts.view(1, 1, height, width)

    radius = max(1, int(3 * support_sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=counts.dtype)
    kernel1d = torch.exp(-0.5 * (coords / support_sigma) ** 2)
    kernel1d = kernel1d / kernel1d.sum()
    blurred = F.conv2d(
        F.pad(counts, (radius, radius, 0, 0), mode="circular"),
        kernel1d.view(1, 1, 1, -1),
    )
    blurred = F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode="circular"),
        kernel1d.view(1, 1, -1, 1),
    )

    peak = blurred.amax().clamp_min(1e-12)
    gate = (blurred / peak).clamp(0.0, 1.0)
    return gate * (1.0 - floor) + floor


__all__ = [
    "DEFAULT_MERGE_RADIUS",
    "hermitian_extend",
    "hermitian_phase_violation",
    "sample_density_gate",
]
