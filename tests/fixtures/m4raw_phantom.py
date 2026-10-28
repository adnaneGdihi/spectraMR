"""Synthetic M4Raw files: a smooth 3-D head phantom in multicoil k-space.

The anatomy is a sum of soft-edged ellipsoids, so it changes from slice to slice and
survives the low-pass the registration fit sees. ``shift`` displaces the content by
``(slices, rows, cols)`` voxels, which is how a test plants motion between contrasts;
a negative ``gain`` inverts the contrast, as T1 is to T2.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c

SIZE, SLICES, PIXEL_MM, SLICE_MM = 64, 12, 0.9375, 5.0
_BLOBS = (  # centre and radii as fractions of the grid, then amplitude
    (0.5, 0.5, 0.5, 0.45, 0.38, 0.30, 1.0),
    (0.35, 0.35, 0.40, 0.20, 0.10, 0.08, 0.8),
    (0.65, 0.62, 0.58, 0.18, 0.08, 0.12, -0.6),
    (0.5, 0.45, 0.66, 0.30, 0.06, 0.05, 0.7),
)


def header(prefix: str = "ns0") -> bytes:
    """An ISMRMRD header in M4Raw's ``ns0:``-prefixed form (``prefix=""`` for a default xmlns)."""
    p = f"{prefix}:" if prefix else ""
    xmlns = f"xmlns:{prefix}" if prefix else "xmlns"
    fov = PIXEL_MM * SIZE
    space = (
        f"<{p}matrixSize><{p}x>{SIZE}</{p}x><{p}y>{SIZE}</{p}y><{p}z>1</{p}z></{p}matrixSize>"
        f"<{p}fieldOfView_mm><{p}x>{fov}</{p}x><{p}y>{fov}</{p}y><{p}z>{SLICE_MM}</{p}z>"
        f"</{p}fieldOfView_mm>"
    )
    return (
        f'<?xml version="1.0"?><{p}ismrmrdHeader {xmlns}="http://www.ismrm.org/ISMRMRD">'
        f"<{p}encoding><{p}encodedSpace>{space}</{p}encodedSpace>"
        f"<{p}reconSpace>{space}</{p}reconSpace></{p}encoding></{p}ismrmrdHeader>"
    ).encode()


def phantom(gain: float = 1.0, shift: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """``[SLICES, SIZE, SIZE]`` image with its content displaced by ``shift`` voxels."""
    z, y, x = np.meshgrid(
        np.arange(SLICES) - shift[0],
        np.arange(SIZE) - shift[1],
        np.arange(SIZE) - shift[2],
        indexing="ij",
    )
    image = np.zeros((SLICES, SIZE, SIZE))
    for z0, y0, x0, rz, ry, rx, amplitude in _BLOBS:
        r = np.sqrt(
            ((z - z0 * SLICES) / (rz * SLICES)) ** 2
            + ((y - y0 * SIZE) / (ry * SIZE)) ** 2
            + ((x - x0 * SIZE) / (rx * SIZE)) ** 2
        )
        image += amplitude / (1 + np.exp((r - 1) * 12))
    return image * gain + (1.5 if gain < 0 else 0.0)


def kspace(image: np.ndarray) -> np.ndarray:
    """``[S, 4, H, W]`` complex64 multicoil k-space under smooth corner-coil sensitivities."""
    yy, xx = np.mgrid[:SIZE, :SIZE] / SIZE
    sens = np.stack([np.exp(-((yy - a) ** 2 + (xx - b) ** 2)) for a in (0, 1) for b in (0, 1)])
    coils = torch.tensor(image[:, None] * sens[None], dtype=torch.complex64)
    return fft2c(coils).numpy()


def write(path: Path, image: np.ndarray, *, real_storage: bool = False, **attrs: str) -> Path:
    """An M4Raw-shaped h5: ``kspace`` (complex, or a trailing real/imag axis) and header."""
    data = kspace(image)
    if real_storage:
        data = np.stack([data.real, data.imag], axis=-1).astype(np.float32)
    with h5py.File(path, "w") as f:
        f["kspace"] = data
        f["ismrmrd_header"] = header()
        f.attrs.update(attrs)
    return path
