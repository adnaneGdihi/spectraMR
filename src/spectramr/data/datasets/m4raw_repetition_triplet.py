"""Repetition-split M4Raw triplets: input and target are two independent acquired repetitions.

M4Raw acquires every contrast more than once. Feeding one repetition as the input and an
independent one as the target makes the target's noise independent of everything the model
sees, so a loss against it estimates the true error plus a constant (Noise2Noise), and a
power-domain version corrects the magnitude bias exactly (``qmap_rep_split``).

The registered triplet (``M4RawTripletDataset``) supplies the subjects, valid slices and
protocols. Each raw repetition is moved into the T1 frame with the rigid fit its contrast's
registered average recorded (``registration_shift_px``, ``registration_slice_offset``;
rotation is never applied there either), so no second registration job is needed.
Repetitions of one contrast are taken as mutually aligned; residual inter-repetition motion
adds a constant shared by every model under comparison.
"""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.data.datasets.m4raw_triplet_dataset import CONTRASTS, M4RawTripletDataset
from spectramr.data.m4raw_noise_units import complex_kspace, noise_unit_rss, scan_noise_power
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.subpixel_registration import fourier_shift


def _fit(path: Path) -> tuple[tuple[float, float], int]:
    """The in-plane shift and slice offset a registered average recorded; none for T1."""
    with h5py.File(path, "r") as f:
        shift = f.attrs.get("registration_shift_px")
        offset = f.attrs.get("registration_slice_offset")
    if shift is None:
        return (0.0, 0.0), 0
    return (float(shift[1]), float(shift[2])), int(offset)


def _stack(images: list[torch.Tensor]) -> tio.ScalarImage:
    """(3, H, W, 1) in the TorchIO layout the triplet dataset uses."""
    return tio.ScalarImage(tensor=torch.stack(images).float().unsqueeze(-1), affine=np.eye(4))


class M4RawRepetitionTripletDataset(Dataset):
    """``input`` = one repetition per contrast, ``target`` = another, both in the T1 frame.

    Args:
        t2_root, flair_root: the registration trees, as for ``M4RawTripletDataset``.
        raw_root: the acquired tree holding ``<subject>_<contrast><NN>.h5`` repetitions.
        split: split directory under every root.
        deterministic: pair the first two repetitions (validation) instead of a random
            pair per item (training).
    """

    def __init__(
        self, t2_root: Path, flair_root: Path, raw_root: Path, split: str, *, deterministic: bool
    ) -> None:
        base = M4RawTripletDataset(t2_root, flair_root, split)
        raw = Path(raw_root) / split
        self.deterministic = deterministic
        self.excluded = dict(base.excluded)
        self._protocol = base._protocol
        self._reps: dict[str, list[list[Path]]] = {}
        self._fits: dict[str, list[tuple[tuple[float, float], int]]] = {}
        self._sigma2: dict[Path, float] = {}
        for subject, paths in base._files.items():
            reps = [sorted(raw.glob(f"{subject}_{c}0[1-9].h5")) for c in CONTRASTS]
            short = [c for c, r in zip(CONTRASTS, reps, strict=True) if len(r) < 2]
            if short:
                self.excluded[subject] = f"fewer than two repetitions of {short}"
                continue
            self._reps[subject] = reps
            self._fits[subject] = [_fit(p) for p in paths]
            for path in (p for r in reps for p in r):  # once here, it pickles into workers
                self._sigma2[path] = scan_noise_power(path)
        self.records = [(s, i) for s, i in base.records if s in self._reps]
        if not self.records:
            raise FileNotFoundError(f"no subject with two repetitions of each contrast under {raw}")

    def __len__(self) -> int:
        return len(self.records)

    def _image(self, path: Path, s: int, fit: tuple[tuple[float, float], int]) -> tuple:
        (dy, dx), offset = fit
        with h5py.File(path, "r") as f:
            k = complex_kspace(f["kspace"][s - offset])
        # Per coil a pixel carries only the acquired samples' noise (M4Raw zero-fills ~24 % of
        # its phase-encode columns), so the noise power in noise units is f C, not C.
        acquired = (k.abs().sum(0) > 0).float().mean()
        if dy or dx:
            shifts = torch.tensor([dy, dx], dtype=torch.float32).expand(1, k.shape[0], 2)
            k = fft2c(fourier_shift(ifft2c(k).unsqueeze(0), shifts)[0])
        return noise_unit_rss(k, self._sigma2[path]), float(acquired) * k.shape[0]

    def __getitem__(self, i: int) -> tio.Subject:
        subject, s = self.records[i]
        pick = (lambda r: r[:2]) if self.deterministic else (lambda r: random.sample(r, 2))
        inputs, targets, noise = [], [], []
        for reps, fit in zip(self._reps[subject], self._fits[subject], strict=True):
            a, b = pick(reps)
            (x, _), (y, power) = self._image(a, s, fit), self._image(b, s, fit)
            inputs.append(x)
            targets.append(y)
            noise.append(power)
        return tio.Subject(
            input=_stack(inputs),
            target=_stack(targets),
            noise_power=torch.tensor(noise),
            subject_id=subject,
            slice_index=s,
            **self._protocol[subject],
        )
