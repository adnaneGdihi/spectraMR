"""M4Raw serves every coil as k-space; an image-domain arm gets its image from the transform.

``rss_image`` is applied by ``CoilCombineTransform``, which reads the dataset's real-stacked
``(2C, H, W, S)`` tensors as ``C`` complex coils. The dataset itself only accepts ``none``,
``rss`` and ``flatten``, so the two have to meet on ``none``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset
from spectramr.data.transforms.kspace_coil_transforms import CoilCombineTransform
from spectramr.infrastructure.physics.fft_ops import ifft2c

SLICES, COILS, SIZE = 2, 4, 16


def _repetitions(tmp_path: Path) -> list[Path]:
    rng = np.random.default_rng(0)
    files = []
    for rep in (1, 2):
        shape = (SLICES, COILS, SIZE, SIZE)
        k = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        path = tmp_path / f"2022000001_T10{rep}.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=k)
        files.append(path)
    return files


def _item(tmp_path: Path):
    ds = M4RawRepetitionDataset(
        h5_files=_repetitions(tmp_path),
        single_contrast=True,
        coil_processing_mode="none",
        use_repetitions=True,
        target_mode="phase_aligned_mean",
    )
    return ds[0]


def test_the_dataset_serves_every_coil_for_the_transform_to_combine(tmp_path: Path) -> None:
    subject = _item(tmp_path)
    assert tuple(subject["input"].data.shape) == (2 * COILS, SIZE, SIZE, SLICES)


def test_the_combine_transform_turns_those_coils_into_one_real_image(tmp_path: Path) -> None:
    subject = _item(tmp_path)
    stacked = subject["input"].data
    coils = torch.complex(stacked[0::2], stacked[1::2])  # (C, H, W, S)
    expected = torch.sqrt((ifft2c(coils.permute(3, 0, 1, 2)).abs() ** 2).sum(1))  # (S, H, W)

    combined = CoilCombineTransform(method="rss_image")(subject)

    for name in ("input", "target"):
        image = combined[name].data
        assert tuple(image.shape) == (1, SIZE, SIZE, SLICES), name
        assert not torch.is_complex(image) and bool((image >= 0).all()), name
    got = combined["input"].data[0].permute(2, 0, 1)
    assert torch.allclose(got, expected, atol=1e-4)
