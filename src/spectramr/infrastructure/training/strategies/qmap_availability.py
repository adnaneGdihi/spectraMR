"""Which contrasts a q-map model sees, and the leave-one-contrast-out readings.

``model_kwargs.contrast_dropout`` hides each contrast of a training sample with that
probability (at least one always stays), so a model learns to work from any subset.
Validation then holds out each contrast in turn and scores only its render: the
pre-registered primary metric of ``DESIGN_tissue_map_cohort.md`` §5.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from spectramr.data.batch_types import read_batch_field

CONTRAST_KEYS = ("t1", "t2", "flair")
#: The header protocol fields M4RawTripletDataset puts in every batch, one value per series.
PROTOCOL_FIELDS = (
    "acq_tr",
    "acq_te",
    "acq_ti",
    "acq_esp",
    "acq_etl",
    "acq_refocus_deg",
    "sequence_family",
)


def validate_contrast_dropout(p: Any) -> float:
    """``model_kwargs.contrast_dropout`` as a float in [0, 1); 1 would hide everything."""
    value = float(p)
    if not 0.0 <= value < 1.0:
        raise ValueError(f"model_kwargs.contrast_dropout={p!r}; expected 0 <= p < 1")
    return value


def draw_availability(batch_size: int, p: float, device: torch.device) -> torch.Tensor | None:
    """A ``(B, 3)`` 0/1 mask, or ``None`` (everything available) when ``p`` is 0."""
    if p == 0.0:
        return None
    keep = (torch.rand(batch_size, len(CONTRAST_KEYS), device=device) >= p).float()
    empty = keep.sum(1) == 0
    if empty.any():  # rescue a sample that lost every contrast with one at random
        pick = torch.randint(len(CONTRAST_KEYS), (int(empty.sum()),), device=device)
        keep[empty.nonzero(as_tuple=True)[0], pick] = 1.0
    return keep


def read_series_protocol(batch: Any) -> list[dict[str, float]]:
    """One host-side protocol dict per series, from the batch's first sample."""
    field = {k: read_batch_field(batch, k) for k in PROTOCOL_FIELDS}
    missing = [k for k, v in field.items() if v is None]
    if missing:
        raise ValueError(f"the batch carries no {missing}; use dataset_type m4raw_triplet")
    return [{n: float(v[0, k]) for n, v in field.items()} for k in range(len(CONTRAST_KEYS))]


def unroll_protocol(series: Sequence[dict[str, float]]) -> list[dict[str, Any]]:
    """The strategy's header-protocol rows in the render's argument names."""
    return [
        {
            "family": int(p["sequence_family"]),
            "tr": p["acq_tr"],
            "te": p["acq_te"],
            "ti": p["acq_ti"],
            "esp": p["acq_esp"],
            "etl": int(p["acq_etl"]),
            "refocus_deg": p["acq_refocus_deg"],
        }
        for p in series
    ]


def leave_one_out(
    render: Callable[[torch.Tensor], list[torch.Tensor]],
    *,
    batch_size: int,
    device: torch.device,
    score: Callable[[int, list[torch.Tensor], torch.Tensor], dict[str, float]],
) -> dict[str, float]:
    """Hold out each contrast in turn and score its render.

    ``render(available)`` returns one render per contrast. ``score(k, renders, available)``
    returns named readings for held-out contrast k; it may use the available contrasts
    (to predict k's gain, say) but must not read k's reference to predict it. Each reading
    is reported per contrast (``<name>_t1`` ...) and as their mean (``<name>``).
    """
    per: dict[str, list[float]] = {}
    out: dict[str, float] = {}
    for k, contrast in enumerate(CONTRAST_KEYS):
        available = torch.ones(batch_size, len(CONTRAST_KEYS), device=device)
        available[:, k] = 0.0
        for name, value in score(k, render(available), available).items():
            out[f"{name}_{contrast}"] = value
            per.setdefault(name, []).append(value)
    out.update({name: sum(v) / len(v) for name, v in per.items()})
    return out
