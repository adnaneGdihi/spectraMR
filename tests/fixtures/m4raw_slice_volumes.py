"""Synthetic M4Raw-shaped repetition groups for the slice-route tests (#1757, #2673).

Each group is one (patient, contrast) with three repetitions of a four-coil
volume: smooth coil sensitivities, a disk of anatomy whose brightness tapers
toward the edge slices the way a brain does, a per-(repetition, slice) receiver
phase, unit complex noise, and the trailing phase-encode columns zero-filled as
M4Raw's partial FOV is. Small enough for a unit test, shaped like the real data
in every axis the loader reads.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SLICES, COILS, HEIGHT, WIDTH = 6, 4, 48, 48
#: Zero-filled trailing phase-encode columns (M4Raw leaves ~24 % unsampled).
UNSAMPLED = 8
REPS = 3


def sensitivities() -> np.ndarray:
    """``(C, H, W)`` smooth Gaussian coil profiles, one per quadrant."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, HEIGHT), np.linspace(-1, 1, WIDTH), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    return np.stack([np.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])


def anatomy(slice_index: int, slices: int = SLICES) -> np.ndarray:
    """A disk whose amplitude falls from 20 at the centre slice to ~5 at the edge."""
    yy, xx = np.meshgrid(np.linspace(-1, 1, HEIGHT), np.linspace(-1, 1, WIDTH), indexing="ij")
    taper = 0.15 + 0.85 * np.sin(np.pi * (slice_index + 0.5) / slices)
    return ((xx**2 + yy**2) < 0.5) * 20.0 * taper


def volume(seed: int, slices: int = SLICES, noise_gain: float = 1.0) -> np.ndarray:
    """One repetition's ``(S, C, H, W)`` complex64 k-space."""
    rng = np.random.default_rng(seed)
    sens = sensitivities()
    image = np.empty((slices, COILS, HEIGHT, WIDTH), dtype=np.complex128)
    for s in range(slices):
        phase = np.exp(1j * rng.uniform(-np.pi, np.pi))
        image[s] = sens * anatomy(s, slices) * phase
    shape = image.shape
    image += noise_gain * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
    k = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1))), axes=(-2, -1))
    k = k / np.sqrt(HEIGHT * WIDTH)
    k[..., -UNSAMPLED:] = 0
    return k.astype(np.complex64)


def write_group(root: Path, patient: str, *, contrast: str = "T1", seed: int = 0) -> list[Path]:
    """``<patient>_<contrast>0N.h5`` for N = 1..REPS; returns the paths in order."""
    root.mkdir(parents=True, exist_ok=True)
    paths = []
    for rep in range(REPS):
        path = root / f"{patient}_{contrast}{rep + 1:02d}.h5"
        with h5py.File(str(path), "w") as f:
            f.create_dataset("kspace", data=volume(seed * 10 + rep))
        paths.append(path)
    return paths


def write_maps(maps_root: Path, files: list[Path]) -> None:
    """Precomputed maps for every group in ``files``, as the precompute script stores them."""
    from spectramr.data.datasets.m4raw_coil_maps import (
        estimate_group_maps,
        map_path,
        save_coil_maps,
    )
    from spectramr.data.datasets.m4raw_dataset import _load_kspace
    from spectramr.infrastructure.physics.scan_coil_calibration import DEFAULT_CALIBRATION

    groups: dict[str, list[Path]] = {}
    for path in sorted(files):
        groups.setdefault(path.stem[:-2], []).append(path)
    for reps in groups.values():
        maps, meta = estimate_group_maps([_load_kspace(p) for p in reps], dict(DEFAULT_CALIBRATION))
        save_coil_maps(map_path(maps_root, reps[0].stem), maps, meta)
