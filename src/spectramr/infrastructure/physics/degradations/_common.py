"""Shared helpers of the 2-D degradation kernels: seeded RNG, complex promotion, signal scale."""

from __future__ import annotations

import zlib

import torch

from spectramr.core.quantile import robust_quantile


def _generator(device: torch.device, seed: int | None) -> torch.Generator:
    """Build a torch generator pinned to ``device`` and seeded.

    The Sobol / MI / BT estimators require deterministic outputs for
    a given ``(theta, seed)``. This helper isolates the seeded RNG
    from any global state.
    """
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(int(seed))
    else:
        g.manual_seed(0)
    return g


def derive_axis_seed(master_seed: int, axis: str) -> int:
    """Deterministic per-axis seed, constant across the severity grid.

    Distinct axes get distinct seeds, so two axes never share one artefact
    realisation; but each axis's seed does not depend on theta, so theta stays
    the only free variable (the #244 property).

    The axis hash is ``zlib.crc32``, **not** the builtin ``hash``: the latter is
    salted by ``PYTHONHASHSEED`` and differs between interpreter runs, which
    would reintroduce cross-process non-determinism through the side door.
    The sim2rank sweeps import it from here.
    """
    return (int(master_seed) * 1_000_003 + zlib.crc32(axis.encode("utf-8"))) % (2**31 - 1)


def readout_offset_px(hz: torch.Tensor, bw_hz_per_px: float) -> torch.Tensor:
    """Readout off-resonance offset ``df / BW_px`` in px (Haacke et al. 1999, ch. 20).

    b0_geometric_distortion pulls back ``out(r) = x(r + df / BW_px)``, so content moves by minus
    this offset; metal_artifact forward-maps each source by the same minus.
    """
    return hz / bw_hz_per_px


def _ensure_complex(x: torch.Tensor) -> torch.Tensor:
    if x.is_complex():
        return x
    return torch.complex(x, torch.zeros_like(x))


def _signal_level(mag: torch.Tensor, q: float = 0.99) -> torch.Tensor:
    """Robust signal scale of each sample of a magnitude image: its ``q``-quantile.

    Every noise axis must derive its noise power from the signal so the operator is
    **scale-invariant** — degrading then rescaling has to equal rescaling then
    degrading, because sim2rank sweeps the RAW coil-combined magnitude and normalises
    by p99 only afterwards. p99 rather than ``max`` so one hot voxel cannot set the
    noise floor. Returns ``[B, 1, 1, 1]``: one level per sample, pooled over its coils,
    so a bright neighbour in the batch does not set this image's noise.
    """
    flat = mag.detach().reshape(mag.shape[0], -1).float()
    return robust_quantile(flat, q, dim=1).clamp(min=1e-12).view(-1, 1, 1, 1)


def _sample_rms(k: torch.Tensor) -> torch.Tensor:
    """RMS of each sample over its coils and k-space, ``[B, 1, 1, 1]``."""
    return k.abs().pow(2).mean(dim=(-3, -2, -1), keepdim=True).sqrt().clamp(min=1e-8)


def _calibrate_bernoulli_scale(
    p: torch.Tensor,
    target: float,
    iters: int = 40,
) -> torch.Tensor:
    """Scale ``c`` such that ``mean(clamp(c * p, max=1)) == target``.

    Bisection (``mean(clamp(c*p, 1))`` is monotone non-decreasing in ``c``), run
    branchlessly on-device so it introduces no host syncs.

    Issue #246: a Bernoulli mask that keeps each sample with probability ``p / R``
    realises an acceleration of ``R / E[p]``, **not** ``R`` — and with a
    centre-peaked ``p`` whose mean is ~0.2, a declared R = 8 came out at R = 36.
    Calibrating the scale makes the realised rate equal the declared one.
    """
    tgt = torch.as_tensor(float(target), device=p.device, dtype=p.dtype)
    lo = torch.zeros((), device=p.device, dtype=p.dtype)
    # At c = 1 / min(p) every entry clamps to 1, so the mean is 1 >= any target.
    hi = (1.0 / p.clamp(min=1e-8).min()).reshape(())
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_low = (mid * p).clamp(max=1.0).mean() < tgt
        lo = torch.where(too_low, mid, lo)
        hi = torch.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def _isolated_rng(device: torch.device, seed: int):
    """Context manager that seeds the global RNG without leaking state.

    The simulator primitives (``simulate_b0_inhomogeneity``,
    ``simulate_eddy_current``, ``simulate_spike_noise``,
    ``simulate_rigid_motion_kspace``) use the *global* PyTorch RNG via
    ``torch.randn``/``torch.rand``/``randn_like``. To keep the
    ``(theta, seed) -> identical output`` contract from polluting any
    other RNG consumer in the same process (C3 in the digital-twin
    review), we fork the RNG around each delegated call.

    Forking on CUDA additionally requires listing the device.
    """
    devices: list[torch.device] = []
    if device.type == "cuda":
        devices = [device]
    ctx = torch.random.fork_rng(devices=devices)
    return ctx
