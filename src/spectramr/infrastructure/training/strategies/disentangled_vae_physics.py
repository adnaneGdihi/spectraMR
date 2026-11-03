"""Physics axis of :class:`DisentangledVAETrainingStrategy` (``training.disentangled_vae``).

:func:`resolve_physics_axis` reads ``physics_source`` once, checks that the data block can
supply it, and returns a :class:`PhysicsAxis` that gives each row's source and target
physics vector. :class:`FieldSwapReadings` keeps one validation pass of the field-strength
axis and reports the disentanglement readings at the end of it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_args

import torch

from spectramr.config.schemas.training.disentangled_vae import DisentangledVAEConfig, PhysicsSource
from spectramr.core.metrics.disentanglement import DisentanglementAccumulator, cosine_distance
from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.acquisition_registry import AcquisitionRegistry

PHYSICS_SOURCES: frozenset[str] = frozenset(get_args(PhysicsSource))

#: The batch keys ``data.expose.field_strength_pair`` stamps (Tesla, one per sample).
SOURCE_FIELD_KEY = "field_strength_source_t"
TARGET_FIELD_KEY = "field_strength_t"


class PhysicsAxis:
    """Where each row's physics vector comes from, for one of the advertised sources."""

    def __init__(self, source: str, contrasts: tuple[str, str] = ("T1w", "T1w")) -> None:
        if source not in PHYSICS_SOURCES:
            raise ValueError(
                f"training.disentangled_vae.physics_source={source!r}; "
                f"expected one of {sorted(PHYSICS_SOURCES)}"
            )
        self.source = source
        self.contrasts = contrasts
        # Per contrast: (fields, vectors) on the CPU, built at setup so an unregistered
        # contrast fails here; copies per device are cached on first use.
        self._tables: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._on_device: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
        if source == "paired_field_stamps":
            for contrast in contrasts:
                self._tables[contrast] = AcquisitionRegistry.physics_table(contrast)

    @property
    def has_physics(self) -> bool:
        return self.source != "single_field"

    @staticmethod
    def row_fields(batch: Any, key: str, n_rows: int) -> torch.Tensor:
        """The stamped field (Tesla) of each of ``n_rows`` rows, where it sits; raises if absent."""
        value = read_batch_field(batch, key) if batch is not None else None
        if value is None:
            raise ValueError(
                f"physics_source=paired_field_stamps needs batch['{key}'] on every batch; set "
                "data.expose.field_strength_pair: true on a contrast_aware_paired route"
            )
        fields = torch.as_tensor(value).detach().to(dtype=torch.float32).reshape(-1)
        if fields.numel() == 0 or n_rows % fields.numel():
            raise ValueError(f"{fields.numel()} '{key}' values cannot cover {n_rows} rows")
        return fields.repeat_interleave(n_rows // fields.numel())

    def _lookup(self, contrast: str, fields: torch.Tensor, device: torch.device) -> torch.Tensor:
        fields = fields.to(device, non_blocking=True)
        slot = (contrast, str(fields.device))
        if slot not in self._on_device:
            known, table = self._tables[contrast]
            self._on_device[slot] = (known.to(fields.device), table.to(fields.device))
        return AcquisitionRegistry.lookup_fields(fields, *self._on_device[slot])

    def vectors(
        self, batch: Any, n_rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(source, target)`` physics vectors, each ``(n_rows, 4)`` on ``device``."""
        if self.source == "fixed_t1w_64mt_3t":
            src = AcquisitionRegistry.get_physics_vector("T1w", "64mT", device)
            tgt = AcquisitionRegistry.get_physics_vector("T1w", "3T", device)
            return src.expand(n_rows, -1), tgt.expand(n_rows, -1)
        if self.source != "paired_field_stamps":
            raise ValueError(f"physics_source={self.source!r} has no physics vectors")
        src_c, tgt_c = self.contrasts
        src = self._lookup(src_c, self.row_fields(batch, SOURCE_FIELD_KEY, n_rows), device)
        tgt = self._lookup(tgt_c, self.row_fields(batch, TARGET_FIELD_KEY, n_rows), device)
        return src, tgt


def resolve_physics_axis(config: Any) -> PhysicsAxis:
    """Read ``training.disentangled_vae`` and check the data block can feed it."""
    block = getattr(config.training, "disentangled_vae", None)
    if block is None:
        block = DisentangledVAEConfig()
    if not isinstance(block, DisentangledVAEConfig):
        raise TypeError(
            f"training.disentangled_vae must be a DisentangledVAEConfig, got {type(block).__name__}"
        )
    if block.physics_source != "paired_field_stamps":
        return PhysicsAxis(block.physics_source)
    data = config.data
    if data.dataset_type != "contrast_aware_paired" or not data.expose.field_strength_pair:
        raise ValueError(
            "physics_source=paired_field_stamps needs data.dataset_type: contrast_aware_paired "
            "with data.expose.field_strength_pair: true (the only route that stamps both fields); "
            f"got dataset_type={data.dataset_type!r}, "
            f"field_strength_pair={data.expose.field_strength_pair!r}"
        )
    if data.input_contrast is None or data.target_contrast is None:
        raise ValueError(
            "physics_source=paired_field_stamps reads the protocol contrast from "
            "data.input_contrast.name / data.target_contrast.name; declare both"
        )
    return PhysicsAxis(block.physics_source, (data.input_contrast.name, data.target_contrast.name))


class FieldSwapReadings:
    """One validation pass of the field axis: probes, content cycle and swap error ratio.

    * ``val_content_field_leak``: a held-out linear probe reads the field from the content
      code of the source and target images (0 when the anatomy code is field-free).
    * ``val_swap_content_distance``: the content of ``decode(c_src, p_tgt)`` against
      ``c_src`` (0 when a physics swap leaves the anatomy intact).
    * ``val_field_swap_error_ratio``: ``||decode(c_src, p_tgt) - x_tgt||^2`` over
      ``||decode(c_src, p_src) - x_tgt||^2``; below 1 when the physics code moves the render
      toward the target field, about 1 when the decoder ignores it.
    """

    def __init__(self) -> None:
        self._acc = DisentanglementAccumulator()
        self._cycle: list[float] = []
        self._swap_err = 0.0
        self._keep_err = 0.0

    def add(
        self,
        content: tuple[torch.Tensor, torch.Tensor],
        physics: tuple[torch.Tensor, torch.Tensor],
        fields: tuple[torch.Tensor, torch.Tensor],
        groups: Sequence[str] | None,
        swap: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Record a batch.

        ``swap`` is ``(rendered_tgt, rendered_src, target_image, content_of_rendered_tgt)``.
        """

        # Source and target rows interleave per sample, so a row-half split (no subject ids)
        # keeps each pair on one side and still sees both fields on the training side.
        def pairs(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.stack((a, b), dim=1).flatten(0, 1)

        labels = torch.round(pairs(*fields) * 1000.0).long()  # field in mT
        both_groups = None if groups is None else [g for g in groups for _ in range(2)]
        self._acc.add(pairs(*content), pairs(*physics), labels, both_groups)
        rendered_tgt, rendered_src, target, cycled = swap
        self._cycle.append(cosine_distance(cycled, content[0]))
        self._swap_err += float((rendered_tgt - target).pow(2).sum())
        self._keep_err += float((rendered_src - target).pow(2).sum())

    def summary(self) -> dict[str, float]:
        if self._acc.n_rows == 0:
            return {}
        out = self._acc.probes(include_style=False, label_name="field")
        out["val_swap_content_distance"] = sum(self._cycle) / len(self._cycle)
        out["val_field_swap_error_ratio"] = self._swap_err / max(self._keep_err, 1e-12)
        return out


__all__ = [
    "PHYSICS_SOURCES",
    "SOURCE_FIELD_KEY",
    "TARGET_FIELD_KEY",
    "FieldSwapReadings",
    "PhysicsAxis",
    "resolve_physics_axis",
]
