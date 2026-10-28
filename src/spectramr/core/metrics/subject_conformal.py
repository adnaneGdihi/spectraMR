"""Split conformal calibrated on held-out SUBJECTS of the anatomy stream (VFU-09).

The fiducial-calibrated construction fits its quantiles on marker residuals
produced by feeding the marker stack through the anatomy slot, an input the
network never trained on, and refits per validation batch. This module is the
standard alternative: residuals of the ordinary anatomy forward pass are
banked per subject across a whole validation pass; the first
``n_calibration`` subjects (sorted by id, so the choice is stable across runs)
calibrate, and coverage is measured on the remaining subjects only.

A calibration subject that was also seen in training would make the guarantee
circular, so :meth:`SubjectResidualBank.evaluate` raises on any overlap.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from spectramr.core.metrics.anchor_conformal import AnchorConformalCalibrator


class SubjectResidualBank:
    """Per-subject residual/difficulty points for one validation pass.

    Args:
        n_calibration: Subjects (sorted by id) that calibrate; the rest test.
        points_per_sample: Points kept per sample, on a fixed stride, so a pass
            costs O(samples x points) memory rather than O(samples x pixels).
    """

    def __init__(self, n_calibration: int, points_per_sample: int = 4096) -> None:
        if n_calibration < 1:
            raise ValueError(f"n_calibration must be >= 1, got {n_calibration}")
        if points_per_sample < 1:
            raise ValueError(f"points_per_sample must be >= 1, got {points_per_sample}")
        self.n_calibration = n_calibration
        self.points_per_sample = points_per_sample
        self._bank: dict[str, tuple[list[Tensor], list[Tensor]]] = {}

    def reset(self) -> None:
        self._bank = {}

    @property
    def subjects(self) -> list[str]:
        return sorted(self._bank)

    def add(self, subjects: Sequence[str], residual: Tensor, difficulty: Tensor) -> None:
        """Bank one batch. ``residual`` and ``difficulty`` are ``[B, ...]``."""
        if residual.shape != difficulty.shape:
            raise ValueError(
                f"residual {tuple(residual.shape)} and difficulty "
                f"{tuple(difficulty.shape)} must match"
            )
        b = residual.shape[0]
        if len(subjects) != b:
            raise ValueError(f"{len(subjects)} subject ids for a batch of {b}")
        r = residual.detach().reshape(b, -1)
        d = difficulty.detach().reshape(b, -1)
        stride = max(1, r.shape[1] // self.points_per_sample)
        # Copy the strided points: a view would keep each full-resolution batch alive.
        r, d = r[:, ::stride].contiguous(), d[:, ::stride].contiguous()
        for i, subject in enumerate(subjects):
            res, dif = self._bank.setdefault(str(subject), ([], []))
            res.append(r[i])
            dif.append(d[i])

    def split(self) -> tuple[list[str], list[str]]:
        """``(calibration, test)`` subject ids; raises without at least one of each."""
        subjects = self.subjects
        if len(subjects) <= self.n_calibration:
            raise ValueError(
                f"held-out conformal calibration needs more than {self.n_calibration} "
                f"validation subject(s) ({self.n_calibration} to calibrate, at least one "
                f"to test), but validation held {len(subjects)}: {subjects}. Declare a "
                "subject-grouped validation split with more subjects."
            )
        return subjects[: self.n_calibration], subjects[self.n_calibration :]

    def _points(self, ids: Iterable[str]) -> tuple[Tensor, Tensor]:
        rs, ds = [], []
        for s in ids:
            res, dif = self._bank[s]
            rs.extend(res)
            ds.extend(dif)
        return torch.cat(rs), torch.cat(ds)

    def evaluate(
        self, calibrator: AnchorConformalCalibrator, *, train_subjects: Iterable[str]
    ) -> dict[str, float]:
        """Fit on the calibration subjects, report coverage on the test subjects."""
        calib, test = self.split()
        leaked = sorted(set(calib) & {str(s) for s in train_subjects})
        if leaked:
            raise ValueError(
                f"conformal calibration subjects {leaked} were also seen in training: "
                "the calibration set overlaps training, so the guarantee is circular. "
                "Use a subject-disjoint train/validation split."
            )
        calibrator.fit(*self._points(calib))
        r_test, d_test = self._points(test)
        out = calibrator.coverage(r_test, d_test).as_dict()
        out["conformal_mean_half_width"] = float(calibrator.half_width(d_test).mean())
        out["conformal_n_calibration_subjects"] = float(len(calib))
        out["conformal_n_test_subjects"] = float(len(test))
        return out


__all__ = ["SubjectResidualBank"]
