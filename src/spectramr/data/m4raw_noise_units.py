"""M4Raw images in noise units, so a series' remaining scale is a property of the scanner.

Raw k-space keeps the receive chain's scale, which a per-image percentile rescale would
erase. Dividing by each scan's own noise standard deviation leaves a gain set by the
acquisition (voxel volume, readout time, averages) and, if the receive chain is stable,
by nothing that varies per subject; ``scripts/analysis/m4raw_series_gain_gate.py`` tests
that.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from spectramr.data.datasets.m4raw_dataset import _pe_support
from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.scan_coil_calibration import (
    as_slice_batch,
    fit_scan_covariance,
)


def noise_power(
    kspace: torch.Tensor, fit_slices: int = 8, support: torch.Tensor | None = None
) -> float:
    """Mean per-coil noise variance, ``trace(Sigma_n) / C``, from the central slices.

    ``support`` overrides the acquired-line support read off this k-space. A registered
    copy needs it: the registration's FFT round trip leaves the never-acquired lines
    near zero rather than exactly zero, so its own support reads as fully sampled and
    the sampled-fraction correction is lost. Pass the source file's support instead.
    """
    slices = as_slice_batch(kspace)
    lo = max(0, (slices.shape[0] - fit_slices) // 2)
    central = slices[lo : lo + fit_slices]
    covariance = fit_scan_covariance(central, _pe_support(central) if support is None else support)
    sigma2 = float(torch.diagonal(covariance).real.mean())
    if not sigma2 > 0:
        raise ValueError(f"noise fit returned a non-positive variance {sigma2}")
    return sigma2


def noise_unit_rss(kspace_slice: torch.Tensor, sigma2: float) -> torch.Tensor:
    """Root-sum-of-squares magnitude divided by the scan's noise standard deviation."""
    image = ifft2c(kspace_slice)
    return image.abs().pow(2).sum(0).sqrt() / sigma2**0.5


def complex_kspace(array: np.ndarray) -> torch.Tensor:
    """M4Raw stores k-space complex or as a trailing real/imaginary pair."""
    t = torch.from_numpy(array)
    return t if torch.is_complex(t) else torch.complex(t[..., 0], t[..., 1])


def _central_slices(n_slices: int, fit_slices: int) -> slice:
    lo = max(0, (n_slices - fit_slices) // 2)
    return slice(lo, lo + fit_slices)


def source_support(path: Path, fit_slices: int = 8) -> torch.Tensor | None:
    """The acquired-line support of a registered copy's source; ``None`` if not registered.

    Registration un-zeroes the skipped lines (#2613), so a registered file's own
    support is wrong; ``registration_source_file`` names the file that still has them.
    """
    with h5py.File(path, "r") as f:
        source = f.attrs.get("registration_source_file")
    if source is None:
        return None
    if not Path(source).exists():
        raise ValueError(f"{path}: registration_source_file {source} does not exist")
    with h5py.File(source, "r") as f:
        ds = f["kspace"]
        k = complex_kspace(ds[_central_slices(ds.shape[0], fit_slices)])
    return _pe_support(k)


def scan_noise_power(path: Path, fit_slices: int = 8) -> float:
    """``noise_power`` of one file, reading only its central slices, on the right support."""
    with h5py.File(path, "r") as f:
        ds = f["kspace"]
        k = complex_kspace(ds[_central_slices(ds.shape[0], fit_slices)])
    return noise_power(k, fit_slices=fit_slices, support=source_support(path, fit_slices))
