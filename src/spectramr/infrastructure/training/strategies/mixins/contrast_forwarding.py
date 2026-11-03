"""Per-sample contrast conditioning for generators that name ``contrast_idx`` (#2410).

The batch carries one contrast id per record (``contrast_id`` / ``contrast_idx``).
Contrast-conditioned generators take it as ``contrast_idx`` and, beside it, an
``acquisition_params`` mapping of per-sample ``[B]`` tensors built from
``data.multi_contrast.acquisition_params``. The table is built once and indexed on
device, so a step costs one gather and no host sync.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer

__all__ = [
    "ACQUISITION_PARAM_KEYS",
    "AcquisitionLookup",
    "align_contrast_rows",
    "aux_map_loss_context",
    "bloch_loss_context",
    "build_acquisition_lookup",
    "contrast_inputs",
    "declared_acquisition_lookup",
    "names_parameter",
]

#: Numeric columns handed to the model, in table order (ms, ms, ms, degrees, tesla).
ACQUISITION_PARAM_KEYS: tuple[str, ...] = ("TE", "TR", "TI", "FA", "B0")

#: Sequence families the Bloch layer renders; any other declared family raises.
_SEQUENCE_CODES: dict[str, int] = {
    "spin_echo": MultiPhysicsBlochLayer.SEQ_SPIN_ECHO,
    "inversion_recovery": MultiPhysicsBlochLayer.SEQ_INVERSION_RECOVERY,
    "gradient_echo": MultiPhysicsBlochLayer.SEQ_GRADIENT_ECHO,
    "diffusion_weighted": MultiPhysicsBlochLayer.SEQ_DIFFUSION_WEIGHTED,
}


def names_parameter(forward: Callable[..., Any], name: str) -> bool:
    """True when ``forward`` names ``name`` explicitly; a bare ``**kwargs`` does not count."""
    return _names_parameter(getattr(forward, "__func__", forward), name)


@functools.lru_cache(maxsize=256)
def _names_parameter(function: Callable[..., Any], name: str) -> bool:
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def contrast_inputs(forward: Callable[..., Any], multi_contrast: Any) -> tuple[bool, bool]:
    """Which of ``contrast_idx`` / ``acquisition_params`` a forward is fed; train and predict share it.

    ``data.multi_contrast.enabled`` is the switch, so an arm with it off runs the same
    network unconditioned.
    """
    if not multi_contrast.enabled:
        return False, False
    return (
        names_parameter(forward, "contrast_idx"),
        names_parameter(forward, "acquisition_params"),
    )


def declared_acquisition_lookup(multi_contrast: Any) -> AcquisitionLookup | None:
    """The lookup over ``data.multi_contrast.acquisition_params``, or ``None`` when none are declared."""
    if not multi_contrast.acquisition_params:
        return None
    return build_acquisition_lookup(
        dict(multi_contrast.contrast_map),
        [p.model_dump() for p in multi_contrast.acquisition_params],
    )


def bloch_loss_context(generator: Any) -> dict[str, Any]:
    """The ``context`` a Bloch-anchored loss reads, from the generator's last forward.

    A generator that factors its output through tissue parameters exposes them as
    ``last_tissue_params`` beside the ``last_acquisition_params`` it rendered with;
    ``bloch_consistency`` re-synthesises from both. Empty for any other generator.
    """
    tissue = getattr(generator, "last_tissue_params", None)
    acquisition = getattr(generator, "last_acquisition_params", None)
    if tissue is None or acquisition is None:
        return {}
    return {"context": {"tissue_params": tissue, "acquisition_params": acquisition}}


def aux_map_loss_context(generator: Any, batch_context: dict[str, Any]) -> dict[str, Any]:
    """The auxiliary maps a generator wrote on its last forward, with the batch's B0.

    ``cross_field_fm`` with ``tissue_heads`` exposes ``last_aux_maps`` (``t1_ms``,
    ``tissue_probabilities``); ``contrast_consistency`` reads them with the per-sample
    ``field_strength``. Empty for any generator that writes no auxiliary maps.
    """
    aux = getattr(generator, "last_aux_maps", None)
    if aux is None:
        return {}
    return {"aux_maps": aux, "field_strength": batch_context.get("field_strength")}


def align_contrast_rows(contrast: torch.Tensor, rows: int) -> torch.Tensor:
    """One id per generator row: repeat each record's id over its flattened slices."""
    declared = int(contrast.shape[0])
    if declared == rows:
        return contrast
    if declared == 0 or rows % declared:
        raise ValueError(
            f"the batch carries {declared} contrast id(s) for {rows} generator row(s); "
            "no whole-record expansion assigns one id per row."
        )
    return contrast.repeat_interleave(rows // declared, dim=0)


class AcquisitionLookup:
    """Device-resident per-contrast table; ``__call__`` gathers one row per sample."""

    def __init__(self, values: torch.Tensor, sequence: torch.Tensor) -> None:
        self._values = values
        self._sequence = sequence

    def __call__(self, contrast: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._values.device != contrast.device:
            self._values = self._values.to(contrast.device)
            self._sequence = self._sequence.to(contrast.device)
        idx = contrast.long()
        rows = self._values[idx]
        out = {key: rows[:, i] for i, key in enumerate(ACQUISITION_PARAM_KEYS)}
        out["contrast_type"] = self._sequence[idx]
        return out


def build_acquisition_lookup(
    contrast_map: Mapping[str, int], rows: Sequence[Mapping[str, Any]]
) -> AcquisitionLookup:
    """Table row ``i`` is the contrast whose ``contrast_map`` id is ``i``.

    Raises when the ids are not ``0..n-1`` or the rows do not name exactly the
    mapped contrasts, so a sample can never be conditioned on another contrast's
    protocol.
    """
    n = len(contrast_map)
    if sorted(contrast_map.values()) != list(range(n)):
        raise ValueError(
            f"data.multi_contrast.contrast_map ids {sorted(contrast_map.values())} must be "
            f"exactly 0..{n - 1} to index data.multi_contrast.acquisition_params."
        )
    by_name = {str(row["name"]): row for row in rows}
    if len(by_name) != len(rows) or set(by_name) != set(contrast_map):
        raise ValueError(
            "data.multi_contrast.acquisition_params must name each contrast_map contrast "
            f"exactly once; contrast_map {sorted(contrast_map)}, rows "
            f"{[str(row['name']) for row in rows]}."
        )
    values = torch.zeros(n, len(ACQUISITION_PARAM_KEYS))
    sequence = torch.zeros(n, dtype=torch.long)
    for name, idx in contrast_map.items():
        row = by_name[name]
        values[idx] = torch.tensor([float(row[key]) for key in ACQUISITION_PARAM_KEYS])
        family = str(row["contrast_type"])
        if family not in _SEQUENCE_CODES:
            raise ValueError(
                f"contrast {name!r} declares contrast_type {family!r}, which the Bloch layer "
                f"does not render; expected one of {sorted(_SEQUENCE_CODES)}."
            )
        sequence[idx] = _SEQUENCE_CODES[family]
    return AcquisitionLookup(values, sequence)
