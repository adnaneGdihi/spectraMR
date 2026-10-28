"""Every held-out test slice rendered as one PNG: input | prediction | target | |error|.

The test pass attaches a :class:`HeldOutSliceWriter` to the strategy; the validation step hands it
the scoring-domain views of each batch. One panel per slice lands in ``test/slices/`` and
``slice_index.csv`` maps each panel to its loader identity, batch row and realised acceleration,
which is how :mod:`.summary` joins images to the per-slice metric rows.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from spectramr.infrastructure.reporting.test_split.degradation_guard import (
    DegradationGuard,
    realised_rates,
)
from spectramr.infrastructure.training.validation_rows import batch_identifiers

SLICES_SUBDIR = "slices"
SLICE_INDEX_CSV = "slice_index.csv"
#: The |error| panel is windowed on this fraction of the target's window, the same for every
#: slice, so equal grey means equal error across the gallery.
ERROR_WINDOW_FRACTION = 0.25
INDEX_COLUMNS = (
    "batch_index",
    "sample",
    "subject_id",
    "contrast",
    "file_id",
    "slice_index",
    "realised_acceleration",
    "rate_source",
    "image",
)
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def magnitude(x: torch.Tensor) -> np.ndarray:
    """``(B, C, H, W)`` real or complex -> ``(B, H, W)`` root-sum-of-squares over channels."""
    x = x.detach()
    if x.ndim == 3:
        x = x.unsqueeze(1)
    sq = x.abs().square() if x.is_complex() else x.float().square()
    return sq.sum(dim=1).sqrt().float().cpu().numpy()


def _window(img: np.ndarray, hi: float) -> np.ndarray:
    if hi <= 0 or not np.isfinite(hi):
        return np.zeros(img.shape, dtype=np.uint8)
    return (np.clip(img / hi, 0.0, 1.0) * 255.0).astype(np.uint8)


def render_panel(
    input_mag: np.ndarray | None, pred_mag: np.ndarray, target_mag: np.ndarray
) -> np.ndarray:
    """One ``(H, 4W)`` uint8 strip on the target's 99.5th-percentile window.

    An input that does not share the target grid (a super-resolution arm) is drawn blank.
    """
    hi = float(np.percentile(target_mag, 99.5)) or float(target_mag.max())
    blank = np.zeros(target_mag.shape, dtype=np.uint8)
    shown_input = (
        _window(input_mag, hi)
        if input_mag is not None and input_mag.shape == target_mag.shape
        else blank
    )
    error = np.abs(pred_mag - target_mag) if pred_mag.shape == target_mag.shape else None
    return np.concatenate(
        [
            shown_input,
            _window(pred_mag, hi) if pred_mag.shape == target_mag.shape else blank,
            _window(target_mag, hi),
            blank if error is None else _window(error, ERROR_WINDOW_FRACTION * hi),
        ],
        axis=1,
    )


def _cell(value: Any, k: int, n: int) -> str:
    if value is None or (isinstance(value, str) and not value):
        return ""
    if isinstance(value, torch.Tensor):
        flat = value.detach().reshape(-1).cpu().tolist()
        return str(flat[k * len(flat) // n]) if flat else ""
    if isinstance(value, (list, tuple)):
        return str(value[k * len(value) // n]) if value else ""
    return str(value)


class HeldOutSliceWriter:
    """Renders each observed slice and checks the first batch with ``guard``."""

    def __init__(self, out_dir: Path, guard: DegradationGuard) -> None:
        self.out_dir = Path(out_dir)
        self.guard = guard
        self.rows: list[dict[str, Any]] = []
        #: The guard's violation on the first batch; the test pass raises on it.
        self.poisoned: str | None = None
        self._checked = False

    def observe(
        self,
        *,
        input_view: torch.Tensor | None,
        pred_view: torch.Tensor,
        target_view: torch.Tensor,
        batch_data: Any,
        batch_index: int | None,
    ) -> None:
        rates, stamped = realised_rates(batch_data)
        if not self._checked:
            self._checked = True
            if input_view is not None:
                self.poisoned = self.guard.verdict(input_view, target_view, rates, stamped=stamped)
        if self.poisoned is not None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        inputs = magnitude(input_view) if input_view is not None else None
        preds, targets = magnitude(pred_view), magnitude(target_view)
        n = targets.shape[0]
        n_batch = int(rates.numel()) if rates is not None else n
        ids = batch_identifiers(batch_data, n_batch) if batch_data is not None else {}
        rate = rates.cpu().tolist() if rates is not None else None
        source = "" if rates is None else ("loader" if stamped else "mask")
        for k in range(n):
            row = {key: _cell(ids.get(key), k, n) for key in INDEX_COLUMNS[2:6]}
            row["batch_index"] = "" if batch_index is None else int(batch_index)
            row["sample"] = k
            row["realised_acceleration"] = rate[k * len(rate) // n] if rate else ""
            row["rate_source"] = source
            stem = (
                "_".join(
                    v
                    for v in (
                        row["subject_id"],
                        row["contrast"],
                        row["file_id"],
                        row["slice_index"],
                    )
                    if v
                )
                or f"batch{row['batch_index']}"
            )
            name = f"{_UNSAFE.sub('-', stem)}_b{row['batch_index']}_{k}.png"
            panel = render_panel(
                None if inputs is None or k >= inputs.shape[0] else inputs[k],
                preds[k],
                targets[k],
            )
            Image.fromarray(panel).save(self.out_dir / name, optimize=False)
            row["image"] = f"{SLICES_SUBDIR}/{name}"
            self.rows.append(row)

    def write_index(self) -> Path:
        """``slice_index.csv`` beside the panels; one row per rendered slice."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / SLICE_INDEX_CSV
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(INDEX_COLUMNS))
            writer.writeheader()
            writer.writerows(self.rows)
        return path


__all__ = [
    "ERROR_WINDOW_FRACTION",
    "SLICES_SUBDIR",
    "SLICE_INDEX_CSV",
    "HeldOutSliceWriter",
    "magnitude",
    "render_panel",
]
