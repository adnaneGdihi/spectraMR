"""The 2-D NEX pseudo-GT reference read from real ``.h5`` files.

The ESPIRiT / OOM / heterogeneous-rep contracts are pinned in
``tests/unit/infrastructure/physics/test_m4raw_pseudo_gt.py`` through stubbed reads; this
file checks the composition on files.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.datasets.pseudo_gt_reference import synthesize_pseudo_gt
from spectramr.infrastructure.physics.fft_ops import fft2c


def _rep(path, data):
    with h5py.File(path, "w") as f:
        f.create_dataset("kspace", data=data.astype(np.complex64))


def test_the_central_slice_is_the_complex_mean_of_the_reps(tmp_path) -> None:
    rng = np.random.default_rng(0)
    reps = [
        rng.standard_normal((5, 1, 8, 8)) + 1j * rng.standard_normal((5, 1, 8, 8)) for _ in range(2)
    ]
    paths = [tmp_path / f"s_T1{i:02d}.h5" for i in (1, 2)]
    for p, r in zip(paths, reps, strict=True):
        _rep(p, r)
    coil_images, smaps, mag, p99 = synthesize_pseudo_gt(paths)
    assert coil_images.shape == smaps.shape == (1, 1, 8, 8)
    assert mag.shape == (1, 1, 8, 8)
    want = torch.from_numpy(((reps[0][2] + reps[1][2]) / 2).astype(np.complex64))
    assert torch.allclose(fft2c(coil_images)[0], want, atol=1e-5)
    assert float(p99) > 0


def test_every_rep_failing_raises(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="failed to load"):
        synthesize_pseudo_gt([tmp_path / "missing_T101.h5"])
