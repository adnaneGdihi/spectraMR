"""A ``tio.Compose`` that records each member's output into an active snapshot trace.

Every train and val chain is built through ``_compose_copying_once``, so this is the
one place every loader transformation -- masking, degradation, augmentation,
normalization, coil processing, resampling, cropping -- passes. Outside a snapshot
step it is ``tio.Compose`` plus one ``ContextVar`` read per sample; inside one it
records the subject as the dataset handed it over (``loader/dataset_output``), then
after each member only the tensors whose content that member changed, with the
member's ``applied_record()`` when it has one.
"""

from __future__ import annotations

from typing import Any

import torch
import torchio as tio

from spectramr.core import snapshot_trace


def plane(t: torch.Tensor) -> torch.Tensor:
    """``[C, H, W, D]`` -> the central ``[C, H, W]`` plane; anything else unchanged."""
    if t.dim() == 4:
        return t[..., t.shape[-1] // 2]
    return t


def subject_tensors(subject: Any) -> dict[str, torch.Tensor]:
    """Every image and tensor a subject carries, by key."""
    out: dict[str, torch.Tensor] = {}
    for key, value in subject.items():
        if isinstance(value, tio.Image):
            out[key] = value.data
        elif isinstance(value, torch.Tensor):
            out[key] = value
    return out


def _params(transform: Any) -> dict[str, Any]:
    record = getattr(transform, "applied_record", None)
    if callable(record):
        try:
            return {"applied_record": record()}
        except Exception as exc:  # stamped, never dropped
            return {"applied_record": f"unavailable: {type(exc).__name__}: {exc}"}
    return {}


def _snapshot(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in tensors.items()}


def _differs(before: torch.Tensor | None, after: torch.Tensor) -> bool:
    """By content: a member may copy the subject or edit a tensor in place."""
    if before is None or before.shape != after.shape or before.dtype != after.dtype:
        return True
    return not torch.equal(before, after.detach())


class TracedCompose(tio.Compose):
    """``tio.Compose`` whose members are recorded one by one on a snapshot step."""

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        if not snapshot_trace.is_active():
            return super().apply_transform(subject)
        tensors = subject_tensors(subject)
        before = _snapshot(tensors)
        snapshot_trace.record("loader", "dataset_output", {k: plane(v) for k, v in tensors.items()})
        for i, transform in enumerate(self.transforms):
            subject = transform(subject)
            now = subject_tensors(subject)
            changed = {k: v for k, v in now.items() if _differs(before.get(k), v)}
            before = _snapshot(now)
            params = _params(transform)
            params["changed"] = sorted(changed)
            snapshot_trace.record(
                "loader",
                f"{i:02d}_{type(transform).__name__}",
                {k: plane(v) for k, v in changed.items()},
                params=params,
            )
        return subject


__all__ = ["TracedCompose", "plane", "subject_tensors"]
