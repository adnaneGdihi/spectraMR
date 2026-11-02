"""Model-level compression metrics (``model_size_mb``, ``compression_ratio``).

Like ``param_count`` these are functions of the *model*, not of an image pair, so they are
``MetricType.DOMAIN_SPECIFIC`` in ``scripts/sim2rank/metrics_list.py`` and are read by the
post-training compression export, not by the per-image validation loop.

Storage convention (stamped in every compression report): the weights of the modules in
:data:`QUANTIZABLE_MODULES` are counted at ``weight_bits`` per element plus one fp32 scale per
tensor when ``weight_bits < 32``; every other parameter and buffer at its own element size.
Storage is dense, so pruning changes sparsity and quality but not size.
"""

from __future__ import annotations

import torch.nn as nn

from spectramr.core.metrics.profiling_metrics import _resolve_model
from spectramr.core.metrics.registry import register_metric

#: Module types whose ``weight`` the compression export quantizes and prunes; one owner, read
#: by ``models/model_compression.py`` so the size accounting and the transform cannot drift.
QUANTIZABLE_MODULES: tuple[type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
)

SUPPORTED_WEIGHT_BITS: frozenset[int] = frozenset({4, 8, 32})
_SCALE_BYTES = 4


def model_storage_bytes(model: nn.Module, *, weight_bits: int = 32) -> int:
    """Bytes needed to store ``model`` under the module's storage convention."""
    if weight_bits not in SUPPORTED_WEIGHT_BITS:
        raise ValueError(
            f"weight_bits={weight_bits} is not one of {sorted(SUPPORTED_WEIGHT_BITS)}."
        )
    quantized = {
        id(m.weight)
        for m in model.modules()
        if isinstance(m, QUANTIZABLE_MODULES) and getattr(m, "weight", None) is not None
    }
    total_bits = 0
    for p in model.parameters():
        if id(p) in quantized and weight_bits < 32:
            total_bits += p.numel() * weight_bits + _SCALE_BYTES * 8
        else:
            total_bits += p.numel() * p.element_size() * 8
    for b in model.buffers():
        total_bits += b.numel() * b.element_size() * 8
    return (total_bits + 7) // 8


@register_metric("model_size_mb", aliases=["model_size", "model_size_megabytes"])
class ModelSize:
    """Storage size of a model in MiB at ``weight_bits`` (lower is better)."""

    def __init__(self, weight_bits: int = 32) -> None:
        self.weight_bits = weight_bits

    @property
    def name(self) -> str:
        return "model_size_mb"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: object | None = None,
        target: object | None = None,
        *,
        model: object | None = None,
        weight_bits: int | None = None,
        **kwargs: object,
    ) -> float:
        """Return ``nan`` without a model: the metric is undefined on an image pair."""
        net = _resolve_model(model, kwargs)
        if net is None:
            return float("nan")
        bits = self.weight_bits if weight_bits is None else weight_bits
        return model_storage_bytes(net, weight_bits=bits) / (1024 * 1024)


@register_metric("compression_ratio", aliases=["size_compression_ratio"])
class CompressionRatio:
    """fp32 storage over storage at ``weight_bits`` for the same model (higher is better)."""

    def __init__(self, weight_bits: int = 32) -> None:
        self.weight_bits = weight_bits

    @property
    def name(self) -> str:
        return "compression_ratio"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: object | None = None,
        target: object | None = None,
        *,
        model: object | None = None,
        weight_bits: int | None = None,
        **kwargs: object,
    ) -> float:
        """Return ``nan`` without a model: the metric is undefined on an image pair."""
        net = _resolve_model(model, kwargs)
        if net is None:
            return float("nan")
        bits = self.weight_bits if weight_bits is None else weight_bits
        return model_storage_bytes(net, weight_bits=32) / model_storage_bytes(net, weight_bits=bits)


__all__ = [
    "QUANTIZABLE_MODULES",
    "SUPPORTED_WEIGHT_BITS",
    "CompressionRatio",
    "ModelSize",
    "model_storage_bytes",
]
