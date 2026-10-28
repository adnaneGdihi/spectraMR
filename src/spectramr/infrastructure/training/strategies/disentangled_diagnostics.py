"""Validation-time disentanglement readings for :class:`DisentangledTrainingStrategy`.

``core.metrics.disentanglement`` owns the arithmetic; this module owns the parts that need
the model: which validation rows are kept whole, and the swap decode that re-encodes an
image rendered from one row's content and another row's style.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple

import torch

from spectramr.core.metrics.disentanglement import DisentanglementAccumulator, relative_error

#: Ordered (content source, style source) pairs decoded per validation pass. The probes
#: see every row; the swap decode runs the generator and renderer, so it is capped.
MAX_SWAP_PAIRS = 12


class SwapRow(NamedTuple):
    """One validation row kept whole: its content map, style code, label and render batch.

    ``batch`` is a one-row batch holding what the Bloch render reads for this row (its
    sequence family and, when declared, its acquisition). It exists only for a model that
    emits tissue maps; an image decoder renders nothing and keeps ``None``.
    """

    content: torch.Tensor
    style: torch.Tensor
    label: int
    batch: dict[str, Any] | None


def _one_row(chunk: Mapping[str, Any], k: int) -> dict[str, Any]:
    """Row ``k`` of a chunk batch, whose values are tensors or mappings of tensors."""
    return {
        key: (
            {kk: vv[k : k + 1].detach() for kk, vv in val.items()}
            if isinstance(val, Mapping)
            else val[k : k + 1].detach()
        )
        for key, val in chunk.items()
    }


def expand_row_names(names: Any, n_rows: int) -> list[str] | None:
    """One name per validation row from the per-sample names of a batch, or ``None``.

    Validation can score several rows per sample (the slices of a volume), laid out sample
    after sample, as ``contrast_idx`` is expanded.
    """
    if names is None:
        return None
    names = [str(names)] if isinstance(names, str) else [str(n) for n in names]
    if not names or n_rows % len(names):
        raise ValueError(f"{len(names)} per-sample names cannot cover {n_rows} rows")
    each = n_rows // len(names)
    return [name for name in names for _ in range(each)]


def record_validation_rows(
    acc: DisentanglementAccumulator,
    content: torch.Tensor,
    style: torch.Tensor,
    labels: torch.Tensor,
    render_batch_of_rows: Callable[[], Mapping[str, Any]] | None,
    groups: Sequence[str] | None = None,
) -> None:
    """Feed a batch to the probes and keep the rows the swap test still wants.

    ``render_batch_of_rows`` returns this batch's render inputs (per-row family and
    acquisition) and is called only when a row is kept, because resolving them costs a
    batch parse. It is ``None`` for a model with no sequence map, whose decoder output is
    already an image. ``groups`` is the subject id of each row, which makes the probes'
    split subject-disjoint.
    """
    labels = labels.detach().view(-1)
    acc.add(content, style, labels, groups=groups)
    chunk: Mapping[str, Any] | None = None
    for k in range(labels.numel()):
        label = int(labels[k])
        if not acc.wants(label):
            continue
        if chunk is None and render_batch_of_rows is not None:
            chunk = render_batch_of_rows()
        acc.keep(
            label,
            SwapRow(
                content[k : k + 1].detach(),
                style[k : k + 1].detach(),
                label,
                None if chunk is None else _one_row(chunk, k),
            ),
        )


def swap_distances(
    model: Any,
    render: Callable[[Any, Mapping[str, Any]], torch.Tensor],
    acc: DisentanglementAccumulator,
    max_pairs: int = MAX_SWAP_PAIRS,
    include_style: bool = True,
) -> dict[str, float]:
    """Decode row *i*'s content with row *j*'s style, render, and encode again.

    With ``include_style=False`` (a declared protocol owns the contrast, so the render takes
    no style code) the swap is *i*'s content under *j*'s protocol, and only the content
    distance is reported.

    A factorised model returns *i*'s content unchanged and *j*'s style, so both relative
    errors are 0, and about 1 when the decoder dropped the code.

    Args:
        model: The generator (``enc_c``, ``enc_s``, ``condition_style``, ``decode_tissue``).
        render: Maps the decoder's output and the style source's one-row render batch to an
            image (the strategy's Bloch render for a tissue dict).
        acc: The pass's accumulator.
        max_pairs: Cap on decoded pairs.

    Returns:
        ``val_swap_content_distance`` and ``val_swap_style_distance``; empty when the
        accumulator holds no pair of rows with different labels.
    """
    pairs = acc.swap_pairs(max_pairs)
    if not pairs:
        return {}
    content_distance: list[float] = []
    style_distance: list[float] = []
    # validation_step hands the model back in train mode; a sampled style and live norm
    # statistics would make the distances noise.
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for source, target in pairs:
                label = torch.tensor([target.label], device=target.style.device)
                raw = model.decode_tissue(
                    source.content, model.condition_style(target.style, label)
                )
                if isinstance(raw, dict):
                    if target.batch is None:
                        raise ValueError(
                            "the decoder emits tissue maps but no sequence code was kept for "
                            "the style source; declare model_kwargs.bloch_sequence_by_contrast"
                        )
                    image = render(raw, target.batch)
                else:
                    image = raw
                content_distance.append(relative_error(model.enc_c(image), source.content))
                if include_style:
                    style_out = model.enc_s(image)
                    style = model.reparameterize(*style_out) if model.use_vae else style_out
                    style_distance.append(relative_error(style, target.style))
    finally:
        model.train(was_training)
    out = {"val_swap_content_distance": sum(content_distance) / len(content_distance)}
    if include_style:
        out["val_swap_style_distance"] = sum(style_distance) / len(style_distance)
    return out
