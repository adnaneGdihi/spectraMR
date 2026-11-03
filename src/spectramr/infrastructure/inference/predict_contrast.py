"""The contrast id ``predict`` feeds a contrast-conditioned model, resolved or refused.

Training reads ``contrast_idx`` off the batch (the loader raises when a record has none); validation
passes it through ``generator.sample``. ``predict`` had no source for it, so a checkpoint trained with
``num_contrasts > 0`` was run with no contrast embedding at all -- unconditioned, silently. The id now
comes from ``--contrast-idx``, else from a contrast keyword in the file name looked up in the arm's
own contrast map, and is refused when neither applies. It is never defaulted to 0.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.mixins.contrast_forwarding import (
    contrast_inputs,
    declared_acquisition_lookup,
)

#: Generator kwargs a contrast-conditioned forward is fed at train and at predict (#2410).
CONDITIONING_KEYS: tuple[str, ...] = ("contrast_idx", "acquisition_params")


def _from_filename(name: str, contrast_map: Mapping[str, int] | None) -> int | None:
    if not contrast_map:
        return None
    # Longest keyword wins: "AXT1" is a substring of "AXT1POST", and the file names the latter.
    found = [k for k in contrast_map if k.upper() in name.upper()]
    if not found:
        return None
    longest = max(len(k) for k in found)
    hits = {contrast_map[k] for k in found if len(k) == longest}
    return hits.pop() if len(hits) == 1 else None


def resolve_predict_contrast(
    num_contrasts: int,
    input_file: Path,
    explicit: int | None,
    contrast_map: Mapping[str, int] | None,
) -> torch.Tensor | None:
    """Scalar ``contrast_idx`` for one input file, or ``None`` for a single-contrast model."""
    if num_contrasts <= 0:
        if explicit is not None:
            raise ConfigurationError(
                f"--contrast-idx {explicit} was given but the checkpoint has num_contrasts="
                f"{num_contrasts}: no contrast embedding exists to receive it."
            )
        return None
    idx = explicit if explicit is not None else _from_filename(input_file.name, contrast_map)
    if idx is None:
        raise ConfigurationError(
            f"{input_file.name}: the checkpoint was trained with num_contrasts={num_contrasts} "
            "but no contrast id was supplied or found in the file name (known keywords: "
            f"{sorted(contrast_map) if contrast_map else 'none'}). Pass --contrast-idx N; "
            "running without it would feed the model no contrast embedding."
        )
    if not 0 <= int(idx) < num_contrasts:
        raise ConfigurationError(
            f"contrast id {idx} is outside [0, {num_contrasts}) for {input_file.name}."
        )
    return torch.tensor([int(idx)], dtype=torch.long)


def resolve_predict_conditioning(
    multi_contrast: Any,
    forward: Callable[..., Any],
    input_file: Path,
    explicit: int | None,
) -> dict[str, Any] | None:
    """One-sample ``contrast_idx`` / ``acquisition_params`` for a forward training conditioned.

    Mirrors the training seam (``contrast_inputs``): ``None`` when the arm trains the
    network unconditioned, else the kwargs its forward names, resolved from the file or
    refused. The table is ``data.multi_contrast.contrast_map``, the one training indexed.
    """
    takes_idx, takes_acq = contrast_inputs(forward, multi_contrast)
    if not (takes_idx or takes_acq):
        return None
    contrast_map = dict(multi_contrast.contrast_map)
    if not contrast_map:
        raise ConfigurationError(
            "data.multi_contrast.enabled is true and the generator takes contrast_idx, but "
            "data.multi_contrast.contrast_map is empty: predict has no contrast id to resolve."
        )
    contrast = resolve_predict_contrast(len(contrast_map), input_file, explicit, contrast_map)
    assert contrast is not None  # num_contrasts > 0: resolved or raised
    out: dict[str, Any] = {"contrast_idx": contrast} if takes_idx else {}
    lookup = declared_acquisition_lookup(multi_contrast) if takes_acq else None
    if lookup is not None:
        out["acquisition_params"] = lookup(contrast)
    return out


def expand_conditioning(conditioning: Mapping[str, Any], rows: int) -> dict[str, Any]:
    """Broadcast one-sample conditioning tensors to ``rows`` samples (views, no copy)."""
    return {
        key: (
            {k: v.expand(rows) for k, v in value.items()}
            if isinstance(value, Mapping)
            else value.expand(rows)
        )
        for key, value in conditioning.items()
    }
