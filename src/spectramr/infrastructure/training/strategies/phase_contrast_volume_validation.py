"""Assemble phase-contrast validation slices into subject volumes and score them.

Flow items are readout slices, and divergence or mass conservation exist only on
a volume. The validation loader is unshuffled, so a subject's slices arrive
consecutively; each subject is scored the moment its ``n_slices``-th slice
arrives and its slices are freed, so at most one or two subjects are held at a
time. A subject still open when validation ends was cut short (a validation cap)
or split across ranks, and :meth:`FlowVolumeAccumulator.finalize` raises rather
than score a partial volume.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.flow_metrics import (
    ConsistencyOfMass,
    NormalisedDivergence,
    VelocityRMSEROI,
)
from spectramr.core.metrics.outcome import MetricNotApplicableError

_SLICE_FIELDS = ("velocity", "velocity_true", "vessel_mask", "inlet_mask", "outlet_mask")


class FlowVolumeAccumulator:
    """Per-subject slice store with streaming volume metrics."""

    def __init__(self) -> None:
        self._metrics = {
            "divergence_normalised": NormalisedDivergence(),
            "mass_conservation": ConsistencyOfMass(),
            "velocity_rmse_roi_volume": VelocityRMSEROI(),
        }
        self.reset()

    def reset(self) -> None:
        self._open: dict[str, dict[str, Any]] = {}
        self._scores: dict[str, list[torch.Tensor]] = defaultdict(list)

    def add(
        self,
        subject_ids: list[str],
        slice_index: torch.Tensor,
        n_slices: torch.Tensor,
        voxel_size_mm: torch.Tensor,
        fields: dict[str, torch.Tensor | None],
    ) -> None:
        """Store one batch of slices; ``voxel_size_mm`` is ``[B, 3]`` and ``fields``
        holds ``_SLICE_FIELDS`` batched on dim 0.

        ``slice_index`` / ``n_slices`` are read on the host: they index Python
        dicts, once per validation batch.
        """
        for b, (sid, x, n) in enumerate(
            zip(subject_ids, slice_index.tolist(), n_slices.tolist(), strict=True)
        ):
            entry = self._open.setdefault(
                sid, {"n": int(n), "voxel_size_mm": voxel_size_mm[b], "slices": {}}
            )
            entry["slices"][int(x)] = {
                k: (v[b] if v is not None else None) for k, v in fields.items()
            }
            if len(entry["slices"]) == entry["n"]:
                self._score(self._open.pop(sid))

    def _score(self, entry: dict[str, Any]) -> None:
        ordered = [entry["slices"][x] for x in range(entry["n"])]
        if ordered[0]["velocity"] is None:
            return  # no reference velocity to correct against: nothing to score

        def stack(key: str, dim: int) -> torch.Tensor | None:
            if ordered[0][key] is None:
                return None
            return torch.stack([s[key] for s in ordered], dim=dim)

        velocity = stack("velocity", 2)[None]  # [1, n_dir, T, X, H, W]
        truth = stack("velocity_true", 2)
        ctx = MetricContext(
            voxel_size_mm=entry["voxel_size_mm"],
            foreground_mask=stack("vessel_mask", 0),
            inlet_mask=stack("inlet_mask", 1),
            outlet_mask=stack("outlet_mask", 1),
        )
        for name, metric in self._metrics.items():
            if name == "velocity_rmse_roi_volume" and truth is None:
                continue
            try:
                args = (
                    (velocity, truth[None]) if name == "velocity_rmse_roi_volume" else (velocity,)
                )
                self._scores[name].append(metric(*args, context=ctx).detach())
            except MetricNotApplicableError:
                # Declared not-applicable on this subject (no inflow plane, a still
                # field): the metric is absent for it, never a fabricated value.
                continue

    def finalize(self) -> dict[str, float]:
        """Mean of each volume metric over the subjects that defined it, as ``val_<name>``."""
        if self._open:
            partial = {sid: f"{len(e['slices'])}/{e['n']}" for sid, e in self._open.items()}
            self.reset()
            raise RuntimeError(
                f"validation ended with incomplete flow subjects {partial} (slices seen / "
                "slices per subject). A validation cap (validation.loader.num_batches / "
                "num_samples) or a sharded loader cut a volume; volume metrics need every slice."
            )
        out = {f"val_{k}": float(torch.stack(v).mean()) for k, v in self._scores.items() if v}
        self.reset()
        return out


__all__ = ["FlowVolumeAccumulator"]
