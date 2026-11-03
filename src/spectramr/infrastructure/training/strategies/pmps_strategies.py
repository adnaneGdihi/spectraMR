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
  emits synthetic paired (x_ULF, x_HF) shards (Phase 3); its loop is
  :mod:`.pmps_synthesis`.
* :class:`DataEfficiencyHarnessStrategy` — orchestrates the
  cells × folds × seeds grid for the headline data-efficiency curve
  (Phase 4).

Heavy lifting (corruption operator composition, synthesis writer,
bootstrap CIs) is delegated to dedicated models / losses / use cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn

from spectramr.config.schemas.training.pmps import TrainingConfigPairedSynthesis
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.models.generators.paired_synthesis_generator import PairedSynthesisGenerator
from spectramr.models.losses.registry import create_loss
from spectramr.models.registry import get_model_class

from .pmps_synthesis import (
    SynthesisGeometry,
    load_frozen,
    require_paired_synthesis_config,
    run_paired_synthesis,
)
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


def build_frozen_prior(
    cfg: TrainingConfigPairedSynthesis, device: torch.device
) -> tuple[nn.Module, str]:
    """Build the Stage-A ``tissue_parameter_diffusion`` and load its checkpoint, frozen."""
    prior_cfg = cfg.tissue_prior
    declared_t = prior_cfg.model_kwargs.get("num_timesteps", prior_cfg.timesteps)
    if declared_t != prior_cfg.timesteps:
        raise ValueError(
            f"training.tissue_prior.model_kwargs.num_timesteps={declared_t} disagrees with "
            f"training.tissue_prior.timesteps={prior_cfg.timesteps}"
        )
    prior = get_model_class("tissue_parameter_diffusion")(
        in_channels=3, out_channels=3, **prior_cfg.model_kwargs
    )
    return prior, load_frozen(prior, prior_cfg.checkpoint, prior_cfg.part, device)


def build_frozen_operators(
    cfg: TrainingConfigPairedSynthesis, device: torch.device
) -> tuple[nn.Module, nn.Module, str]:
    """The calibrated ULF operator from the Stage-B checkpoint and the declared HF operator."""
    corr = cfg.corruption
    ulf = get_model_class("ulf_corruption_operator")(
        in_channels=1, out_channels=1, **corr.ulf_model_kwargs
    )
    source = load_frozen(ulf, corr.checkpoint, corr.part, device)
    hf = get_model_class("hf_corruption_operator")(
        in_channels=1,
        out_channels=1,
        initial_rician_sigma=corr.hf.rician_noise_sigma,
        initial_bias_field_severity=corr.hf.bias_field_severity,
        initial_motion_severity=corr.hf.motion_severity,
    )
    hf.to(device).eval()
    for param in hf.parameters():
        param.requires_grad_(False)
    return ulf, hf, source


class PairedSynthesisStrategy(ReconstructionTrainingStrategy):
    """Frozen-model orchestrator for synthesising paired (x_ULF, x_HF) (Phase 3).

    It LEARNS NOTHING. Construction validates every Stage-C knob, loads the frozen prior and
    operators into the arm's :class:`PairedSynthesisGenerator`, and runs the whole corpus once
    (:func:`run_paired_synthesis`); every training step then returns an exactly-zero loss, the
    ``HardwareWarpCalibrationStrategy`` pattern, and every validation pass reports the corpus
    summary instead of running the generator on a loader batch it cannot condition.
    """

    #: The declared loss scores each pair inside the synthesis pass; no step folds it.
    inline_losses: ClassVar[frozenset[str]] = frozenset({"per_pair_consistency_residual"})
    folds_image_losses: ClassVar[bool] = False

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(expected_modes=("paired_synthesis",))
        cfg = require_paired_synthesis_config(self.config.training)
        generator = self.env.generator
        if not isinstance(generator, PairedSynthesisGenerator):
            raise TypeError(
                "paired_synthesis needs model.model_type: paired_synthesis_generator, got "
                f"{type(generator).__name__}"
            )
        declared = (self.config.model.model_kwargs or {}).get("defensibility_threshold")
        if declared is not None and float(declared) != cfg.defensibility_threshold:
            raise ValueError(
                f"model.model_kwargs.defensibility_threshold={declared} disagrees with "
                f"training.defensibility_threshold={cfg.defensibility_threshold}; declare it "
                "once, under training"
            )
        weights = {e.name: e.weight for e in self.config.losses.image_losses}
        if set(weights) != {"per_pair_consistency_residual"}:
            raise ValueError(
                "paired_synthesis scores pairs with losses.image_losses "
                f"[per_pair_consistency_residual] alone; the arm declares {sorted(weights)}"
            )
        if weights["per_pair_consistency_residual"] != 1.0:
            # The residual is recorded per pair, never weighted into an objective.
            raise ValueError("per_pair_consistency_residual is recorded, not weighted: weight 1.0")
        patch = list(self.config.data.sampling.patch_size)
        geometry = SynthesisGeometry(
            batch_size=int(self.config.training.batch_size), height=patch[0], width=patch[1]
        )
        prior, prior_source = build_frozen_prior(cfg, self.device)
        ulf, hf, corruption_source = build_frozen_operators(cfg, self.device)
        generator.ulf_corruption = ulf
        generator.hf_corruption = hf
        generator.defensibility_threshold = cfg.defensibility_threshold
        prior_cfg = cfg.tissue_prior
        scheduler = DiffusionScheduler(
            num_timesteps=prior_cfg.timesteps,
            beta_schedule=prior_cfg.noise_schedule,
            beta_start=prior_cfg.beta_start,
            beta_end=prior_cfg.beta_end,
            device=self.device,
        )
        self.paired_synthesis_provenance: dict[str, Any] = {
            **cfg.model_dump(
                mode="json",
                include={
                    "n_synthetic_pairs",
                    "protocol_sampling",
                    "output_format",
                    "output_root",
                    "defensibility_threshold",
                    "write_rejected",
                    "parallel_shards",
                    "tissue_prior",
                    "corruption",
                },
            ),
            "geometry": vars(geometry),
            "tissue_prior_source": prior_source,
            "ulf_corruption_source": corruption_source,
            "hf_calibrated": False,
            "prior_b0": "p_hf.B0",
            "concomitant_rendered": False,
        }
        logger.info("PairedSynthesisStrategy knobs: %s", self.paired_synthesis_provenance)
        self.synthesis_summary = run_paired_synthesis(
            generator,
            prior,
            scheduler,
            create_loss("per_pair_consistency_residual"),
            cfg,
            geometry,
            self.paired_synthesis_provenance,
        )

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # Nothing trains; an exactly-zero, graph-carrying loss keeps backward/step valid.
        params = [p for p in self.env.generator.parameters() if p.requires_grad]
        if not params:
            return {"loss_total": torch.zeros((), device=self.device, requires_grad=True)}
        return {"loss_total": sum(p.sum() for p in params) * 0.0}

    def validation_step(
        self, input_batch: Any = None, target_batch: Any = None, **kwargs: Any
    ) -> dict[str, float]:
        """The corpus summary; the loader batch is ignored (generation needs theta and protocols)."""
        summary = self.synthesis_summary
        preview = summary["preview"]
        # Fresh tensors per pass, so the visual capture shows the pair instead of a forward.
        self._last_visual_pred = preview["ulf"].clone()
        self._last_visual_target = preview["hf"].clone()
        return {
            k: float(summary[k])
            for k in ("per_pair_consistency_residual", "acceptance_rate", "n_accepted", "n_rendered")
        }


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
            "DataEfficiencyHarnessStrategy: %d settings x %d folds x %d seeds = %d cells.",
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
    "build_frozen_operators",
    "build_frozen_prior",
]
