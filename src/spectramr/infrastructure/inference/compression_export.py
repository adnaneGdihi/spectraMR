"""Post-training model-compression export (the ``compression:`` block).

The service prunes and/or weight-quantizes the trained network through
:class:`~spectramr.models.model_compression.ModelCompressionPipeline`, swaps the compressed
weights into the live generator for exactly one re-scoring pass, restores the fp32 weights, and
writes ``compression_report.json`` plus an integer-weight artifact. The re-scoring itself is the
caller's (``pipelines/train.py``), so the compressed weights are graded by the same validation
pass as the fp32 ones.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.compression import CompressionExportConfigSchema
from spectramr.config.schemas.enums import CompressionPruning, CompressionQuantization
from spectramr.core.metrics.compression_metrics import QUANTIZABLE_MODULES
from spectramr.core.metrics.registry import MetricsRegistry
from spectramr.models.model_compression import (
    CompressionConfig,
    ModelCompressionPipeline,
    quantize_tensor_symmetric,
)

logger = logging.getLogger(__name__)

#: Bits per quantization method; ``none`` keeps fp32 storage.
_WEIGHT_BITS: dict[CompressionQuantization, int] = {
    CompressionQuantization.NONE: 32,
    CompressionQuantization.INT8_WEIGHT: 8,
    CompressionQuantization.INT4_WEIGHT: 4,
}
REPORT_NAME = "compression_report.json"
ARTIFACT_NAME = "compressed_model.pt"
STORAGE_CONVENTION = (
    "dense; Conv/ConvTranspose/Linear weights at weight_bits plus one fp32 scale per tensor, "
    "every other parameter and buffer at its own element size"
)


def _unwrap(model: nn.Module) -> nn.Module:
    """The bare module under a DDP/DataParallel wrapper; FSDP-sharded weights are refused."""
    inner = getattr(model, "module", model)
    if any(hasattr(p, "device_mesh") for p in inner.parameters()):
        raise NotImplementedError(
            "compression export on FSDP-sharded (DTensor) weights is not implemented; "
            "run the export on an unsharded checkpoint."
        )
    return inner


class ModelCompressionExportService:
    """Compress a trained model per a validated ``compression:`` block and report the result."""

    def __init__(self, cfg: CompressionExportConfigSchema, *, seed: int) -> None:
        if not cfg.enabled:
            raise ValueError(
                "ModelCompressionExportService built from a disabled compression block."
            )
        self.cfg = cfg
        self.weight_bits = _WEIGHT_BITS[cfg.quantization.method]
        self.pipeline_config = CompressionConfig(
            quantize_weights=cfg.quantization.method is not CompressionQuantization.NONE,
            quantize_activations=False,
            weight_bits=self.weight_bits,
            pruning_method=cfg.pruning.method.value,
            pruning_ratio=(
                cfg.pruning.amount if cfg.pruning.method is not CompressionPruning.NONE else 0.0
            ),
            pruning_seed=int(seed),
        )

    def compress(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Return the compressed state dict of ``model`` (left unchanged)."""
        compressed = ModelCompressionPipeline(self.pipeline_config).compress_model(_unwrap(model))
        return {k: v.detach() for k, v in compressed.state_dict().items()}

    @contextmanager
    def swapped_in(self, model: nn.Module, state: dict[str, torch.Tensor]) -> Iterator[None]:
        """Load ``state`` into ``model`` for the block, then restore the original weights."""
        inner = _unwrap(model)
        original = {k: v.detach().clone() for k, v in inner.state_dict().items()}
        inner.load_state_dict(state)
        try:
            yield
        finally:
            inner.load_state_dict(original)

    def model_report(self, model: nn.Module, state: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Size metrics (through the metric registry) and sparsity of the compressed weights."""
        inner = _unwrap(model)
        with self.swapped_in(model, state):
            size_fp32 = MetricsRegistry.get("model_size_mb")(model=inner, weight_bits=32)
            size = MetricsRegistry.get("model_size_mb")(model=inner, weight_bits=self.weight_bits)
            ratio = MetricsRegistry.get("compression_ratio")(
                model=inner, weight_bits=self.weight_bits
            )
            weights = [
                m.weight.detach()
                for m in inner.modules()
                if isinstance(m, QUANTIZABLE_MODULES) and m.weight is not None
            ]
            n_total = sum(w.numel() for w in weights)
            n_zero = int(sum((w == 0).sum() for w in weights)) if weights else 0
        return {
            "declared": self.cfg.model_dump(mode="json"),
            "applied": {
                "weight_bits": self.weight_bits,
                "quantization_scheme": "symmetric per-tensor, weights only",
                "pruning_method": self.cfg.pruning.method.value,
                "pruning_amount": self.cfg.pruning.amount,
                "pruning_seed": self.pipeline_config.pruning_seed,
            },
            "storage_convention": STORAGE_CONVENTION,
            "model_size_mb_fp32": size_fp32,
            "model_size_mb": size,
            "compression_ratio": ratio,
            "prunable_weights": n_total,
            "weight_sparsity": n_zero / n_total if n_total else 0.0,
        }

    def write_artifact(self, model: nn.Module, state: dict[str, torch.Tensor], path: Path) -> Path:
        """Save integer weight levels + scales for quantized tensors, everything else as-is."""
        inner = _unwrap(model)
        quantized = {
            f"{name}.weight" if name else "weight"
            for name, m in inner.named_modules()
            if isinstance(m, QUANTIZABLE_MODULES) and m.weight is not None
        }
        levels: dict[str, torch.Tensor] = {}
        scales: dict[str, torch.Tensor] = {}
        dense: dict[str, torch.Tensor] = {}
        for key, tensor in state.items():
            if key in quantized and self.weight_bits < 32:
                levels[key], scales[key] = quantize_tensor_symmetric(tensor, self.weight_bits)
                levels[key], scales[key] = levels[key].cpu(), scales[key].cpu()
            else:
                dense[key] = tensor.cpu()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"weight_bits": self.weight_bits, "levels": levels, "scales": scales, "dense": dense},
            path,
        )
        return path

    def run(
        self,
        model: nn.Module,
        evaluate: Callable[[], dict[str, float]],
        run_dir: Path | None,
    ) -> dict[str, Any]:
        """Score fp32, compress, score compressed, restore; write the report when ``run_dir``."""
        fp32_metrics = dict(evaluate())
        state = self.compress(model)
        with self.swapped_in(model, state):
            compressed_metrics = dict(evaluate())
        report = self.model_report(model, state)
        report["val_metrics_fp32"] = fp32_metrics
        report["val_metrics_compressed"] = compressed_metrics
        report["val_metrics_delta"] = {
            k: compressed_metrics[k] - v
            for k, v in fp32_metrics.items()
            if isinstance(v, (int, float)) and isinstance(compressed_metrics.get(k), (int, float))
        }
        if run_dir is not None:
            report["artifact"] = str(self.write_artifact(model, state, run_dir / ARTIFACT_NAME))
            (run_dir / REPORT_NAME).write_text(json.dumps(report, indent=2, default=str))
            logger.info("[Compression] report -> %s", run_dir / REPORT_NAME)
        return report


__all__ = ["ARTIFACT_NAME", "REPORT_NAME", "ModelCompressionExportService"]
