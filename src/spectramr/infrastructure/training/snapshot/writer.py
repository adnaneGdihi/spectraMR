"""Write one trace as ``manifest.json``, a PNG per stage tensor, and ``overview.png``.

The manifest lists the stages in recording order with each tensor's statistics, view
and PNG; ``overview.png`` lays the same previews out left to right in that order, so
one picture reads as the pipeline. Coil maps that did not change since an earlier step
are written once and referenced.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import torch

from spectramr.core.snapshot_trace import SnapshotTrace, View
from spectramr.infrastructure.training.snapshot.render import (
    MASK_IN_IMAGE_MIN_ZERO_FRAC,
    image_domain_mask_fraction,
    render_image_preview,
    split_channel_segments,
)
from spectramr.infrastructure.training.snapshot.stats import (
    json_default,
    json_safe,
    report_dict,
    resolve_deferred_extra,
    safe_name,
    tensor_report,
)
from spectramr.infrastructure.training.snapshot_coil_maps import COIL_MAP_KEYS, render_coil_map_grid

logger = logging.getLogger(__name__)

MANIFEST = "manifest.json"
OVERVIEW = "overview.png"
TENSORS = "tensors.pt"


def _is_coil_map(key: str, t: torch.Tensor) -> bool:
    leaf = key.rsplit(".", 1)[-1]
    return t.dim() == 4 and (leaf in COIL_MAP_KEYS or "coil_map" in leaf or "smaps" in leaf)


def _units(key: str, t: torch.Tensor, view: View) -> list[tuple[str, torch.Tensor, View]]:
    """The rendering units: the tensor, or one per declared channel segment."""
    parts = split_channel_segments(key, t, view.segments) if view.segments else None
    if parts is None:
        return [(key, t, view)]
    return [
        (
            name,
            sub,
            View(
                kspace=seg_k,
                authoritative=view.authoritative and seg_k,
                log_scaled=view.log_scaled and (seg_log if seg_log is not None else seg_k),
            ),
        )
        for name, sub, seg_k, seg_log in parts
    ]


class TraceWriter:
    """Writes the traces of one run under ``root``; remembers coil maps across steps."""

    def __init__(
        self,
        root: Path,
        *,
        save_images: bool = True,
        save_json: bool = True,
        save_tensors: bool = False,
    ) -> None:
        self.root = root
        self.save_images = save_images
        self.save_json = save_json
        self.save_tensors = save_tensors
        self._coil_pngs: dict[tuple, str] = {}

    def write(self, trace: SnapshotTrace, head: dict[str, Any]) -> Path:
        out = self.root / trace.phase / f"step_{trace.step:06d}"
        out.mkdir(parents=True, exist_ok=True)
        stages: list[dict[str, Any]] = []
        tiles: list[tuple[str, torch.Tensor]] = []
        for i, stage in enumerate(trace.stages):
            entries = []
            for key, t in stage.tensors.items():
                stem = safe_name(f"{i:02d}_{stage.group}_{stage.name}__{key}")
                entries.append(
                    self._tensor_entry(out, stem, key, t, stage.views[key], tiles, stage)
                )
            stages.append(
                {
                    "index": i,
                    "group": stage.group,
                    "name": stage.name,
                    "source": stage.source,
                    "params": stage.params,
                    "tensors": entries,
                }
            )
        manifest = {
            **head,
            "phase": trace.phase,
            "step": trace.step,
            "extra": resolve_deferred_extra(dict(trace.extra)) or {},
            "stages": stages,
        }
        if self.save_json:
            (out / MANIFEST).write_text(
                json.dumps(json_safe(manifest), indent=2, default=json_default)
            )
        if tiles:
            _overview(tiles, out / OVERVIEW)
        if self.save_tensors:
            torch.save({f"{s.group}/{s.name}": s.tensors for s in trace.stages}, out / TENSORS)
        return out

    def _tensor_entry(
        self,
        out: Path,
        stem: str,
        key: str,
        t: torch.Tensor,
        view: View,
        tiles: list[tuple[str, torch.Tensor]],
        stage: Any,
    ) -> dict[str, Any]:
        from torchvision.utils import save_image

        entry: dict[str, Any] = {
            "key": key,
            "view": {
                "kspace": view.kspace,
                "authoritative": view.authoritative,
                "log_scaled": view.log_scaled,
            },
            **report_dict(tensor_report(key, t)),
            "png": [],
        }
        if not self.save_images:
            return entry
        if _is_coil_map(key, t):
            fp = (tuple(t.shape), round(float(t.float().abs().sum()), 4))
            if fp not in self._coil_pngs:
                grid = render_coil_map_grid(t, in_kspace=view.kspace, log_scaled=view.log_scaled)
                path = out / f"{stem}.png"
                save_image(grid, str(path), nrow=(grid.shape[0] - 1) // 2)
                self._coil_pngs[fp] = str(path.relative_to(self.root))
            entry["png"].append(self._coil_pngs[fp])
            return entry
        notes = [entry["note"]] if entry["note"] else []
        for name, sub, unit_view in _units(key, t, view):
            preview = render_image_preview(
                sub,
                in_kspace=unit_view.kspace,
                authoritative=unit_view.authoritative,
                log_scaled=unit_view.log_scaled,
            )
            if preview is None:
                continue
            if unit_view.kspace:
                frac = image_domain_mask_fraction(preview)
                if frac >= MASK_IN_IMAGE_MIN_ZERO_FRAC:
                    notes.append(
                        f"DOMAIN SUPERPOSITION in {name}: {frac:.0%} of interior image lines are "
                        "zero; a k-space mask appears to have been applied in the image domain"
                    )
            path = out / f"{safe_name(stem + name[len(key) :])}.png"
            save_image(preview[:1], str(path))
            entry["png"].append(str(path.relative_to(self.root)))
            tiles.append((f"{stage.group}/{stage.name}\n{name}", preview[0, 0]))
        entry["note"] = "; ".join(notes)
        return entry


def _overview(tiles: list[tuple[str, torch.Tensor]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = min(6, len(tiles))
    rows = math.ceil(len(tiles) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (label, img) in zip(axes.flat, tiles, strict=False):
        ax.imshow(img.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(label, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


__all__ = ["MANIFEST", "OVERVIEW", "TENSORS", "TraceWriter"]
