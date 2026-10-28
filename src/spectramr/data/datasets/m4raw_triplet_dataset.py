"""Co-registered M4Raw T1/T2/FLAIR triplets in noise units, with their header protocols.

Reads the two trees ``register_m4raw_contrasts.py`` writes onto T1 (T2->T1, FLAIR->T1).
A subject enters only if both moving contrasts carry ``registration_valid_slices``; a
symlink means the fit was refused, and that series would not share the tissue map.
Images are divided by each scan's own noise std (``m4raw_noise_units``) and by the
square root of its averaging count, not rescaled per image, so the per-series gain stays
a scanner property a loss can treat as known.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.data.m4raw_noise_units import complex_kspace, noise_unit_rss, scan_noise_power
from spectramr.infrastructure.physics.m4raw_pseudo_gt import read_acquisition_params

CONTRASTS = ("T1", "T2", "FLAIR")


def family_of(params: dict, path: Path) -> int:
    """0 = spin echo / TSE, 1 = inversion recovery; GRE-like headers are out of scope."""
    if float(params.get("TI", 0.0) or 0.0) > 0.0:
        return 1
    flip = params.get("flip_angle")
    if flip is not None and 0.0 < float(flip) < 90.0:
        raise ValueError(f"{path}: flip angle {flip} reads as GRE; only SE/TSE and IR are modelled")
    return 0


def _protocol(params: list[dict], paths: tuple[Path, ...]) -> dict[str, torch.Tensor] | str:
    """Per-series protocol tensors, or the reason this subject is excluded."""
    if any(float(p.get("flip_angle", 0.0) or 0.0) == 0.0 for p in params):
        return "header refocusing angle 0"
    etl = [int(p["echo_train_length"]) for p in params]
    # A single-echo train's only echo is at TE; M4Raw writes its echo spacing as N/A.
    esp = [
        float(p["TE"]) if n == 1 else float(p["echo_spacing"])
        for p, n in zip(params, etl, strict=True)
    ]
    return {
        "acq_tr": torch.tensor([float(p["TR"]) for p in params]),
        "acq_te": torch.tensor([float(p["TE"]) for p in params]),
        "acq_ti": torch.tensor([float(p.get("TI", 0.0) or 0.0) for p in params]),
        "acq_esp": torch.tensor(esp),
        "acq_etl": torch.tensor(etl),
        "acq_refocus_deg": torch.tensor([float(p.get("flip_angle", 180.0)) for p in params]),
        "sequence_family": torch.tensor(
            [family_of(p, q) for p, q in zip(params, paths, strict=True)]
        ),
    }


def _valid(path: Path) -> np.ndarray | None:
    with h5py.File(path, "r") as f:
        v = f.attrs.get("registration_valid_slices")
    return None if v is None else np.flatnonzero(np.asarray(v))


def _nex(path: Path) -> int | None:
    with h5py.File(path, "r") as f:
        n = f.attrs.get("nex_average_count")
    return None if n is None else int(n)


class M4RawTripletDataset(Dataset):
    """One item per valid slice of each registered subject: ``input``/``target`` (3, H, W, 1)."""

    def __init__(self, t2_root: Path, flair_root: Path, split: str) -> None:
        a, b = Path(t2_root) / split, Path(flair_root) / split
        self.records: list[tuple[str, int]] = []
        self.excluded: dict[str, str] = {}
        self._files: dict[str, tuple[Path, Path, Path]] = {}
        self._protocol: dict[str, dict[str, torch.Tensor]] = {}
        self._sigma2: dict[Path, float] = {}
        self._nex: dict[Path, int] = {}
        for t1 in sorted(a.glob("*_T100.h5")):
            subject = t1.name[: -len("_T100.h5")]
            t2, flair = a / f"{subject}_T200.h5", b / f"{subject}_FLAIR00.h5"
            if not (t2.exists() and flair.exists()):
                self.excluded[subject] = "missing series"
                continue
            v2, vf = _valid(t2), _valid(flair)
            if v2 is None or vf is None:
                self.excluded[subject] = "unregistered series"
                continue
            paths = (t1, t2, flair)
            proto = _protocol([read_acquisition_params(p) for p in paths], paths)
            if isinstance(proto, str):
                self.excluded[subject] = proto
                continue
            nex = [_nex(p) for p in paths]
            if None in nex:
                self.excluded[subject] = "missing nex_average_count"
                continue
            # Noise fitted once here (central slices, source support for registered
            # copies, #2613), not per item: it pickles into the loader workers.
            for path, n in zip(paths, nex, strict=True):
                self._sigma2[path], self._nex[path] = scan_noise_power(path), n
            self._files[subject], self._protocol[subject] = paths, proto
            self.records += [(subject, int(s)) for s in np.intersect1d(v2, vf)]
        if not self.records:
            raise FileNotFoundError(f"no registered triplet under {a} and {b}")
        self._refuse_mixed_protocols()

    def _refuse_mixed_protocols(self) -> None:
        """A strategy renders one protocol per series; a second protocol would be misrendered."""
        first_subject, first = next(iter(self._protocol.items()))
        for subject, proto in self._protocol.items():
            for key, value in proto.items():
                if not torch.equal(value, first[key]):
                    raise ValueError(
                        f"{subject}: {key}={value.tolist()} differs from {first_subject}'s "
                        f"{first[key].tolist()}; this split mixes protocols"
                    )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> tio.Subject:
        subject, s = self.records[i]
        images = []
        for path in self._files[subject]:
            with h5py.File(path, "r") as f:
                z = noise_unit_rss(complex_kspace(f["kspace"][s]), self._sigma2[path])
            # Per-repetition units: an N-average is sqrt(N) brighter in its own noise
            # units, and the gate's ratios divide that out the same way.
            images.append(z / self._nex[path] ** 0.5)
        x = torch.stack(images).float().unsqueeze(-1)  # (3, H, W, 1), TorchIO layout
        return tio.Subject(
            input=tio.ScalarImage(tensor=x, affine=np.eye(4)),
            target=tio.ScalarImage(tensor=x.clone(), affine=np.eye(4)),
            subject_id=subject,
            slice_index=s,
            **self._protocol[subject],
        )
