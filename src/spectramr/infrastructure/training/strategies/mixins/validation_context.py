"""What a validation batch carries beyond ``(input, target)``, for the metrics that need it.

The validation loop hands ``ModelValidationMixin.validation_step`` two tensors,
flattened ``[B, C, H, W, D] -> [B*D, C, H, W]`` and ComplexGuard-interleaved by
``pipelines/train.py``. Everything else rides the batch untouched, so a metric
that declares ``needs=("coil_maps",)`` had no route to its maps: the recon path
called the computer without a ``MetricContext`` and got ``nan``. This module reads
the extra fields off the batch and brings each to the shape the two tensors have.

**Which metrics get a context.** Only those whose every declared ``need`` is a
field this path supplies (:data:`SUPPLIED_FIELDS`). A metric needing more -- a
sampling ``mask``, a ``reconstructor`` -- keeps its old context-free call rather
than receiving a context that is silently missing half of what it declared.

The flatten is repeated here, not imported: its owner sits in ``pipelines/``,
which this layer may not import (non-negotiable 5). Where the two could
disagree -- square padding of a non-square slice -- the shapes are compared and a
mismatch raises instead of being padded a second way.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

from spectramr.core.coil_map_names import COIL_MAP_ALIASES, REFERENCE_MAPS_KEY
from spectramr.core.metrics.coil_manifold import (
    as_complex_coils,
    as_interleaved,
    project_kspace_onto_manifold,
)
from spectramr.core.metrics.context import MetricContext
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.core.metrics.sense_image_metrics import SENSE_METRIC_NAMES

__all__ = [
    "SUPPLIED_FIELDS",
    "alternate_input",
    "batch_tensor",
    "coil_maps",
    "compute_context_metrics",
    "context_metric_names",
    "flatten_like",
    "measured_kspace",
    "project_prediction",
    "scoring_maps",
    "with_measurement",
]

#: ``MetricContext`` fields the validation path can fill from a batch.
SUPPLIED_FIELDS: frozenset[str] = frozenset({"coil_maps", "y_kspace", "pred_alt"})

#: The complex spelling first: ``espirit_sensitivity`` writes the magnitude under
#: ``sensitivity`` beside it, and ``BatchAdapter`` binds THAT to ``coil_maps``.
_COIL_MAP_KEYS: tuple[str, ...] = ("sensitivity_complex", *COIL_MAP_ALIASES)


def context_metric_names(computer: Any) -> tuple[str, ...]:
    """Configured metrics that need a context this path can fully supply.

    A computer that cannot list its metrics (a test double) has none to route.
    """
    lister = getattr(computer, "get_metric_names", None)
    if not callable(lister):
        return ()
    names: Iterable[str] = lister()
    return tuple(
        name
        for name in names
        if MetricsRegistry.needs_context(name)
        and set(MetricsRegistry.needs(name)) <= SUPPLIED_FIELDS
    )


def batch_tensor(batch_data: Any, key: str) -> torch.Tensor | None:
    """``key`` off a collated dict or a ``TrainingBatch``, TorchIO wrapper removed."""
    if batch_data is None:
        return None
    if isinstance(batch_data, dict):
        value = batch_data.get(key)
    else:
        metadata = getattr(batch_data, "metadata", None) or {}
        value = metadata.get(key)
        if value is None and key in COIL_MAP_ALIASES:
            value = getattr(batch_data, "coil_maps", None)
    if isinstance(value, dict):
        value = value.get("data")
    return value if isinstance(value, torch.Tensor) else None


def fold_depth(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int] | None]:
    """``(B, C, H, W, D)`` -> ``(B*D, C, H, W)`` plus what unfolds it; 4-D passes through.

    The loop hands TorchIO's layout through unchanged, and every consumer of a
    validation batch here is a per-slice coil operation, so depth folds into
    the batch once and unfolds only where a forward expects the original layout.
    """
    if x.ndim != 5:
        return x, None
    b, c, h, w, d = x.shape
    return x.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w), (b, d)


def unfold_depth(x: torch.Tensor, layout: tuple[int, int] | None) -> torch.Tensor:
    """Inverse of :func:`fold_depth`; a ``None`` layout is the 4-D identity."""
    if layout is None:
        return x
    b, d = layout
    return x.reshape(b, d, *x.shape[1:]).permute(0, 2, 3, 4, 1)


def flatten_like(tensor: torch.Tensor, like: torch.Tensor, *, name: str) -> torch.Tensor:
    """The loop's depth-into-batch flatten, then a hard shape check against ``like``."""
    if tensor.ndim == 5:
        b, c, h, w, d = tensor.shape
        tensor = tensor.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
    if tensor.ndim != 4 or tensor.shape[0] != like.shape[0] or tensor.shape[-2:] != like.shape[-2:]:
        raise ValueError(
            f"validation batch field {name!r} has shape {tuple(tensor.shape)} but the "
            f"tensor it must align with is {tuple(like.shape)}. The loop square-pads "
            "input/target to data.sampling.patch_size and this field was not padded; "
            "serve slices at the patch size instead."
        )
    return tensor.to(like.device, non_blocking=True)


def coil_maps(batch_data: Any, like: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    """``(maps, key)`` flattened to ``like``'s batch, or ``(None, None)`` if absent."""
    for key in _COIL_MAP_KEYS:
        maps = batch_tensor(batch_data, key)
        if maps is not None:
            return flatten_like(maps, like, name=key), key
    return None, None


def scoring_maps(batch_data: Any, like: torch.Tensor) -> tuple[torch.Tensor | None, str | None]:
    """The maps the SENSE image metrics combine with: the validation-only scoring
    store when the batch carries it (one map set across arms, #2664), else the
    batch's own maps."""
    reference = batch_tensor(batch_data, REFERENCE_MAPS_KEY)
    if reference is not None:
        return flatten_like(reference, like, name=REFERENCE_MAPS_KEY), REFERENCE_MAPS_KEY
    return coil_maps(batch_data, like)


def alternate_input(batch_data: Any, like: torch.Tensor) -> torch.Tensor | None:
    """``input_alt`` in the exact layout the model received ``input`` in."""
    alt = batch_tensor(batch_data, "input_alt")
    if alt is None:
        return None
    if torch.is_complex(alt):
        alt = as_interleaved(alt)
    alt = flatten_like(alt, like, name="input_alt")
    if alt.shape != like.shape:
        raise ValueError(
            f"input_alt {tuple(alt.shape)} does not match input {tuple(like.shape)}; "
            "the second repetition must be served in the input's layout."
        )
    return alt


def with_measurement(batch_context: dict[str, Any], kspace: torch.Tensor) -> dict[str, Any]:
    """``batch_context`` with ``measured_kspace`` replaced by ``kspace`` in its layout.

    For a strategy whose validation forward reads the measurement from the context
    rather than from the tensor it is handed (EI, SSDU): the second-repetition
    forward runs on this, so ``pred_alt`` is the estimate from ``input_alt`` (#2593).
    The mask and maps stay: ``input_alt`` is masked by the same ``M_1``.
    """
    current = batch_context.get("measured_kspace")
    if current is None:
        raise ValueError("the validation context carries no measured_kspace to replace.")
    if torch.is_complex(current) != torch.is_complex(kspace):
        kspace = as_interleaved(kspace) if torch.is_complex(kspace) else as_complex_coils(
            kspace, name="input_alt"
        )
    # ``input_alt`` arrives with depth folded into the batch; a unit trailing depth
    # is the only difference a plain view can undo without reordering slices.
    if kspace.dim() == current.dim() - 1 and current.shape[-1] == 1:
        kspace = kspace.unsqueeze(-1)
    if kspace.shape != current.shape:
        raise ValueError(
            f"input_alt {tuple(kspace.shape)} cannot replace measured_kspace "
            f"{tuple(current.shape)}: the two repetitions must share one layout."
        )
    return {**batch_context, "measured_kspace": kspace.to(current.device)}


def measured_kspace(input_batch: torch.Tensor) -> torch.Tensor:
    """The model's k-space input as complex coils, the ``y_kspace`` convention."""
    return as_complex_coils(input_batch, name="input")


def project_prediction(
    prediction: torch.Tensor, batch_data: Any, output_type: str
) -> tuple[torch.Tensor, str]:
    """``F P F^H`` on the k-space prediction, with the maps the batch carries.

    Returns the projected prediction and the batch key the maps came from, for
    the provenance line. Raises when the knob cannot do what it says: a
    projection silently skipped would leave the arm's PSNR describing the raw
    output while its config claims the projected one.
    """
    if output_type != "kspace":
        raise ValueError(
            "validation.scoring.coil_manifold_projection projects multi-coil k-space, "
            f"and this model's output_type is {output_type!r}."
        )
    maps, key = coil_maps(batch_data, prediction)
    if maps is None:
        raise ValueError(
            "validation.scoring.coil_manifold_projection is on but the validation "
            "batch carries no coil maps. Declare the `espirit_sensitivity` transform "
            "in data.processing.transforms."
        )
    return project_kspace_onto_manifold(prediction, maps), str(key)


#: The reference-free certificate (PAPER.md § 3): read on the model's raw output
#: even when `coil_manifold_projection` is on, because a residual measured after
#: projecting is identically zero. Every other context metric grades the output
#: the arm reports, projected when the knob says so.
CERTIFICATE_METRICS: frozenset[str] = frozenset(
    {"null_space_residual", "null_space_residual_fraction"}
)


def compute_context_metrics(
    computer: Any,
    names: tuple[str, ...],
    prediction: torch.Tensor,
    target: torch.Tensor,
    input_batch: torch.Tensor,
    batch_data: Any,
    model_config: Any,
    forward: Callable[[torch.Tensor], torch.Tensor],
    projected: Callable[[torch.Tensor], torch.Tensor] | None = None,
    *,
    signal_domain: str,
    image_prediction: torch.Tensor | None = None,
) -> dict[str, float]:
    """The context-needing metrics as ``val_<name>``.

    ``prediction`` is the model's raw output, in ``signal_domain`` -- passed by
    the caller because a strategy may hand these metrics its output re-expressed
    (an image network's ``F S x_hat``), which ``model.output_type`` would
    misname. ``forward`` must return the same representation. The certificate metrics
    (``CERTIFICATE_METRICS``) read it as is; the others read ``projected(prediction)``
    when a projection is given, and ``pred_alt`` goes through the same projection,
    so the repetition-based decomposition describes the estimator the arm reports.
    ``forward`` runs the model on ``input_alt`` and is called only when a
    configured metric needs ``pred_alt``: a second validation forward nothing
    reads would double the cost for no column.
    """
    # Every reference-graded metric reads the shared scoring store when the batch
    # carries it, so the noise term, the NEX floor and the certificate rest on one
    # map set across families (#2664); otherwise the batch's own maps.
    maps, _ = scoring_maps(batch_data, prediction)
    y_kspace = measured_kspace(input_batch) if model_config.input_type == "kspace" else None
    pred_alt = None
    if any("pred_alt" in MetricsRegistry.needs(name) for name in names):
        alt = alternate_input(batch_data, input_batch)
        if alt is None:
            raise ValueError(
                f"{[n for n in names if 'pred_alt' in MetricsRegistry.needs(n)]} need the "
                "model's output on a second repetition, and the validation batch carries "
                "no `input_alt`. Set data.emit_alternate_input: true."
            )
        with torch.no_grad():
            pred_alt, _ = fold_depth(forward(alt))
        # A forward that ignores its input returns the same tensor on both
        # repetitions, and rep_variance would read 0. An all-zero output (a slice
        # whose coil maps are zero everywhere, projected) is identical on both
        # legitimately and leaves its map-based metrics undefined, so it is not
        # taken as that defect. Two host reads per validation batch, outside the
        # training step, and only when a pred_alt metric is configured.
        if torch.equal(pred_alt, prediction) and bool(torch.count_nonzero(prediction)):
            raise ValueError(
                "the model's output on `input_alt` equals its output on `input`, so the "
                "repetition metrics would read zero variance. The validation forward "
                "ignores the input it is handed (it reads the measurement from the batch "
                "context); drop rep_variance/bias_squared/emit_alternate_input for this "
                "arm until the forward runs on the alternate repetition."
            )
    certificate = tuple(n for n in names if n in CERTIFICATE_METRICS)
    sense = tuple(n for n in names if n in SENSE_METRIC_NAMES)
    graded = tuple(n for n in names if n not in CERTIFICATE_METRICS and n not in SENSE_METRIC_NAMES)
    computed: dict[str, float] = {}
    if certificate:
        context = MetricContext(
            coil_maps=maps,
            y_kspace=y_kspace,
            pred_alt=pred_alt,
            signal_domain=signal_domain,
        )
        computed.update(computer.compute(prediction, target, only=certificate, context=context))
    if graded:
        if projected is not None:
            prediction = projected(prediction)
            if pred_alt is not None:
                pred_alt = projected(pred_alt)
        context = MetricContext(
            coil_maps=maps,
            y_kspace=y_kspace,
            pred_alt=pred_alt,
            signal_domain=signal_domain,
        )
        computed.update(computer.compute(prediction, target, only=graded, context=context))
    if sense:
        # The projected output when the arm projects, as for every graded metric,
        # combined with the scoring maps rather than the arm's training maps. An
        # image network's output is already one combined image and is scored as it
        # is: re-expanding it through its training maps and combining with the
        # scoring maps would score |S_ref^H S_train x_hat|, not x_hat.
        graded_prediction = projected(prediction) if projected is not None and not graded else prediction
        if image_prediction is not None:
            graded_prediction = image_prediction
        context = MetricContext(
            coil_maps=scoring_maps(batch_data, graded_prediction)[0],
            signal_domain=signal_domain,
        )
        computed.update(computer.compute(graded_prediction, target, only=sense, context=context))
    return {f"val_{key}": value for key, value in computed.items()}
