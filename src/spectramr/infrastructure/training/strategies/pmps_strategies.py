"""PMPS strategies — Phases 2 / 3 / 4 (bundled).

Three concrete training strategies for the PMPS pipeline. Each is a
thin subclass of :class:`ReconstructionTrainingStrategy` (following the
same alias-pattern used by ~30 other novel-research strategies in the
factory) that adds init-time checkpoint / config validation and routes
the loss aggregation through the YAML-declared losses.

* :class:`CorruptionCalibrationStrategy` — fits :math:`\\psi` for the
  ULF/HF corruption operators against the real paired corpus via MMD
  (Phase 2).
* :class:`PairedSynthesisStrategy` — frozen-model orchestrator that
  emits synthetic paired (x_ULF, x_HF) shards (Phase 3).
* :class:`DataEfficiencyHarnessStrategy` — orchestrates the
  cells × folds × seeds grid for the headline data-efficiency curve
  (Phase 4).

Heavy lifting (corruption operator composition, synthesis writer,
bootstrap CIs) is delegated to dedicated models / losses / use cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


def _resolve_manifest(value: Any, field_name: str) -> Path:
    """Resolve a manifest path, raising if unset / not on disk."""
    if value is None:
        raise ValueError(f"PMPS: {field_name} is unset; the strategy cannot run without it.")
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"PMPS: {field_name} resolved to {path}, but that path does not exist on disk. "
            f"Smoke-test runs may use a placeholder path under tests/fixtures/."
        )
    return path


class CorruptionCalibrationStrategy(ReconstructionTrainingStrategy):
    """Fit psi for (C_ULF, C_HF) against the real paired corpus (Phase 2)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config.training
        logger.info(
            "CorruptionCalibrationStrategy: %d calibration iters.",
            getattr(cfg, "calibration_iters", -1),
        )


class PairedSynthesisStrategy(ReconstructionTrainingStrategy):
    """Frozen-model orchestrator for synthesising paired (x_ULF, x_HF) (Phase 3).

    No learned parameters at this stage — the strategy walks the
    sampling loop (sample theta → run Bloch → apply corruption → write
    pair) under the standard training-step harness, accumulating zero
    gradients. Inherits from :class:`ReconstructionTrainingStrategy` for
    framework integration (DI, logging, validation hooks).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config.training
        n_pairs = getattr(cfg, "n_synthetic_pairs", 0)
        if n_pairs <= 0:
            raise ValueError("PairedSynthesisStrategy: training.n_synthetic_pairs must be > 0.")
        logger.info(
            "PairedSynthesisStrategy: n_synthetic_pairs=%d, output_root=%s.",
            n_pairs,
            getattr(cfg, "output_root", "?"),
        )


class DataEfficiencyHarnessStrategy(ReconstructionTrainingStrategy):
    """Orchestrate the data-efficiency curve grid (Phase 4)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = self.config.training
        settings = list(getattr(cfg, "settings", []) or [])
        if not settings:
            raise ValueError(
                "DataEfficiencyHarnessStrategy: training.settings must "
                "list at least one cell (e.g. R10, R10_S100, R10_S1000)."
            )
        # Manifest paths are validated by Pydantic but the audit ladder
        # benefits from a strategy-level check that points at the right
        # YAML keys when something is missing on disk.
        for field in (
            "real_paired_manifest",
            "synthetic_paired_manifest",
        ):
            val = getattr(cfg, field, None)
            if val is None or "tests/fixtures" in str(val).replace("\\", "/"):
                continue
            _resolve_manifest(val, f"training.{field}")
        logger.info(
            "DataEfficiencyHarnessStrategy: %d settings × %d folds × %d seeds = %d cells.",
            len(settings),
            getattr(cfg, "loso_n_folds", 1),
            getattr(cfg, "n_seeds_per_setting", 1),
            len(settings)
            * int(getattr(cfg, "loso_n_folds", 1))
            * int(getattr(cfg, "n_seeds_per_setting", 1)),
        )


__all__ = [
    "CorruptionCalibrationStrategy",
    "DataEfficiencyHarnessStrategy",
    "PairedSynthesisStrategy",
]
