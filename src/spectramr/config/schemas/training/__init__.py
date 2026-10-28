"""Training configuration schemas.

This package houses the **base** training-strategy schema plus a
paradigm-specific schema per training mode. The paradigm schemas
specify kwargs and validation conditions that ``training_mode``
dispatch can resolve dynamically (R0 / R4 from
``TODO/audit/CHANGES_RATIONALE.md`` §0).

Public API:

- :class:`BaseTrainingConfigSchema` (also re-exported as
  ``TrainingConfigSchema``) — the base schema referenced by
  ``TrainingSettings.training``.
- :func:`create_training_config` — dispatch a ``config_dict`` to the
  right paradigm-specific schema based on ``training_mode``.
- Per-paradigm schemas: ``TrainingConfigGAN``, ``TrainingConfigDiffusion``,
  ``TrainingConfigVAE``, ``TrainingConfigReconstruction``,
  ``TrainingConfigSSL``, ``TrainingConfigFlow``, ``TrainingConfigFederated``,
  ``TrainingConfigMetaLearning``, ``TrainingConfigMotion``,
  ``TrainingConfigTTO``.

Why these stay (after the 2026-05-15 zero-consumer-restoration pass):

The base schema is the SSOT (``TrainingSettings.training`` is typed
``TrainingStrategyConfigSchema`` with ``extra="allow"``), and strategy
classes duck-type optional fields. But the paradigm schemas remain
because:

1. ``training_mode`` dispatch in ``create_training_config()`` picks the
   right schema at runtime.
2. Downstream forks / experiment YAMLs may import
   ``TrainingConfigGAN`` directly to construct typed dicts.
3. The schemas carry per-paradigm field defaults and constraints that
   are part of the public contract.

An unknown ``training_mode`` falls back to the documented default
``TrainingConfigReconstruction`` — this is an intentional, named
default (not a CLAUDE.md #9 silent fallback), and the factory exposes
the dispatch table so callers can audit the mapping.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseTrainingConfigSchema
from .acq_hypernetwork import AcqHypernetworkConfig, TrainingConfigAcqHypernetwork
from .bloch_manifold_dps import TrainingConfigBlochManifoldDPS
from .diffusion import TrainingConfigDiffusion
from .dispersion_bloch_ae import (
    DispersionBlochAEConfig,
    TrainingConfigDispersionBlochAE,
)
from .equivariance_conformal import TrainingConfigEquivarianceConformal
from .federated import TrainingConfigFederated
from .flow import TrainingConfigFlow
from .gan import TrainingConfigGAN
from .hardware_warp import TrainingConfigHardwareWarpCalibration
from .mcgi import MCGIConfig
from .meta_learning import TrainingConfigMetaLearning
from .motion import TrainingConfigMotion
from .operator_id import OperatorIDConfig, TrainingConfigOperatorID
from .phys_residual_conformal import TrainingConfigPhysResidualConformal
from .pmps import (
    TrainingConfigCorruptionCalibration,
    TrainingConfigDataEfficiencyHarness,
    TrainingConfigPairedSynthesis,
    TrainingConfigTissueDiffusionPretrain,
)
from .reconstruction import TrainingConfigReconstruction
from .se3_navigator import TrainingConfigSE3Navigator
from .spectra_tta import TrainingConfigSpectraTTA
from .ssl import TrainingConfigSSL
from .tto import TrainingConfigTTO
from .vae import TrainingConfigVAE
from .vf_advanced import TrainingConfigIBVF, TrainingConfigTwinDPS

logger = logging.getLogger(__name__)

# Alias preserved for callers using the older name.
TrainingConfigSchema = BaseTrainingConfigSchema


class TrainingConfigLowRankSparse(TrainingConfigReconstruction):
    """Top-level paradigm schema for ``training_mode: low_rank_sparse``.

    The RPCA L+S strategy subclasses ``ReconstructionTrainingStrategy``, so its
    paradigm schema subclasses :class:`TrainingConfigReconstruction`. The five
    RPCA knobs themselves live in the typed ``training.low_rank_sparse``
    sub-block (``LowRankSparseTrainingConfigSchema``, mounted on
    ``TrainingStrategyConfigSchema``); this top-level schema exists so
    ``create_training_config`` dispatches ``low_rank_sparse`` configs through
    the reconstruction objective gating rather than the bare base schema.

    Defined here (not in ``low_rank_sparse.py``) to avoid the import cycle
    base <- low_rank_sparse <- reconstruction <- base.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }


# Dispatch table: ``training_mode`` → attribute name on this module.
# Looking up by attribute name (rather than capturing the class object
# directly) lets tests / forks patch the module-level schema via
# ``unittest.mock.patch("spectramr.config.schemas.training.TrainingConfigGAN")``
# and have the factory respect the patched class. Aliases live here
# too (``kspace_cold_diffusion`` → diffusion).
_MODE_DISPATCH: dict[str, str] = {
    "gan": "TrainingConfigGAN",
    "diffusion": "TrainingConfigDiffusion",
    "kspace_cold_diffusion": "TrainingConfigDiffusion",
    "vae": "TrainingConfigVAE",
    "reconstruction": "TrainingConfigReconstruction",
    "low_rank_sparse": "TrainingConfigLowRankSparse",
    "ssl": "TrainingConfigSSL",
    "flow": "TrainingConfigFlow",
    "federated": "TrainingConfigFederated",
    "meta_learning": "TrainingConfigMetaLearning",
    "motion": "TrainingConfigMotion",
    "tto": "TrainingConfigTTO",
    # SPECTRA test-time adaptation (plan §D) — freeze backbone, adapt norm/adapter.
    "spectra_tta": "TrainingConfigSpectraTTA",
    # PMPS (2026-05-19) — four phases.
    "tissue_diffusion_pretrain": "TrainingConfigTissueDiffusionPretrain",
    "corruption_calibration": "TrainingConfigCorruptionCalibration",
    "paired_synthesis": "TrainingConfigPairedSynthesis",
    "data_efficiency_harness": "TrainingConfigDataEfficiencyHarness",
    # VF advanced (2026-05-19) — Phase-2 IB-VF + Phase-3 Twin-DPS.
    "ib_vf": "TrainingConfigIBVF",
    "twin_dps": "TrainingConfigTwinDPS",
    # VF campaign Phase 2 breakthrough methods (B-1..B-3). B-4
    # (hamiltonian_acquisition) loads via the base schema's extra="allow"
    # (like TTO) — no full TrainingConfig wrapper.
    "equivariance_conformal": "TrainingConfigEquivarianceConformal",
    "phys_residual_conformal": "TrainingConfigPhysResidualConformal",
    "bloch_manifold_dps": "TrainingConfigBlochManifoldDPS",
    "se3_equivariant_navigator": "TrainingConfigSE3Navigator",
    # Operator-ID (Proposal 1) — Lie-algebraic BCH effective-generator ID.
    "operator_id": "TrainingConfigOperatorID",
    # Contrast/field-agnostic bundle (2026-06-29 design) — M3 LCAH and M4
    # DL-BAE. M1 (phys_residual_conformal) is above; M2 (MCGI) rides the
    # supervised paradigm and needs no wrapper.
    "acq_hypernetwork": "TrainingConfigAcqHypernetwork",
    "dispersion_bloch_ae": "TrainingConfigDispersionBlochAE",
    # ULF paired batch-1 Proposal 2 -- hardware warp identified from repeat-session
    # commutators; frozen-parameter, no model or loss registration.
    "hardware_warp_calibration": "TrainingConfigHardwareWarpCalibration",
}

# The documented default mode when ``training_mode`` is missing or
# unrecognised. Kept centralised so the audit / docs and the factory
# agree.
_DEFAULT_MODE = "reconstruction"


def create_training_config(config_dict: dict[str, Any]) -> BaseTrainingConfigSchema:
    """Create a typed training config from a raw dict.

    Dispatches on ``config_dict["training_mode"]`` (defaulting to
    ``reconstruction`` when missing or unrecognised) and returns a
    fully-validated paradigm-specific schema instance.

    The dispatch resolves the schema class by attribute name on this
    module so callers can ``mock.patch("spectramr.config.schemas.training.TrainingConfigGAN")``
    in tests / forks and get a substituted class without rewriting the
    factory.

    Args:
        config_dict: Raw configuration mapping (typically loaded from
            YAML). The ``training_mode`` key drives dispatch.

    Raises:
        ValueError: if ``training_mode`` is neither in the dispatch table nor a
            recognised :data:`VALID_TRAINING_MODES` value — a typo'd or
            unimplemented mode must NOT silently degrade to ``reconstruction``
            (pitfall #9).

    Returns:
        A validated subclass of :class:`BaseTrainingConfigSchema`
        appropriate for the requested mode.
    """
    import sys

    from spectramr.config.validation_constants import VALID_TRAINING_MODES

    mode = str(config_dict.get("training_mode", _DEFAULT_MODE)).lower()

    if mode in _MODE_DISPATCH:
        attr_name = _MODE_DISPATCH[mode]
    elif mode in VALID_TRAINING_MODES:
        # A recognised mode with no dedicated paradigm schema: it intentionally
        # loads through the default (reconstruction/base) schema. Observable —
        # not the previous SILENT ``.get(mode, default)`` fallback (pitfall #9).
        logger.warning(
            "training_mode %r has no dedicated schema; loading through the default %r schema.",
            mode,
            _DEFAULT_MODE,
        )
        attr_name = _MODE_DISPATCH[_DEFAULT_MODE]
    else:
        raise ValueError(
            f"Unknown training_mode {mode!r}: not in the dispatch table "
            f"({sorted(_MODE_DISPATCH)}) nor a recognised VALID_TRAINING_MODES "
            "value. A typo'd or unimplemented mode must not silently fall back "
            f"to {_DEFAULT_MODE!r} (pitfall #9)."
        )

    schema_cls = getattr(sys.modules[__name__], attr_name)

    # ``training_mode`` is a v5.0-removed field; ``reject_deprecated_fields``
    # (base.py) RAISES if it is present. The dispatch above already consumed it,
    # so strip it from a COPY before validating (WS1-st-01 — the documented
    # call form used to always raise on this passthrough).
    validated_dict = {k: v for k, v in config_dict.items() if k != "training_mode"}
    return schema_cls.model_validate(validated_dict)


__all__ = [
    "AcqHypernetworkConfig",
    "BaseTrainingConfigSchema",
    "DispersionBlochAEConfig",
    "MCGIConfig",
    "OperatorIDConfig",
    "TrainingConfigAcqHypernetwork",
    "TrainingConfigCorruptionCalibration",
    "TrainingConfigDataEfficiencyHarness",
    "TrainingConfigDiffusion",
    "TrainingConfigDispersionBlochAE",
    "TrainingConfigFederated",
    "TrainingConfigFlow",
    "TrainingConfigGAN",
    "TrainingConfigHardwareWarpCalibration",
    "TrainingConfigIBVF",
    "TrainingConfigLowRankSparse",
    "TrainingConfigMetaLearning",
    "TrainingConfigMotion",
    "TrainingConfigOperatorID",
    "TrainingConfigPairedSynthesis",
    "TrainingConfigReconstruction",
    "TrainingConfigSSL",
    "TrainingConfigSchema",
    "TrainingConfigSpectraTTA",
    "TrainingConfigTTO",
    "TrainingConfigTissueDiffusionPretrain",
    "TrainingConfigTwinDPS",
    "TrainingConfigVAE",
    "create_training_config",
]
