"""Typed strategy hyperparameter sub-blocks (2026-08 knob-wiring batch).

The unfinished tail of the 2026-06 batch — see
:mod:`spectramr.config.schemas.training.strategy_knobs_2026_06`, whose docstring
states the rule and the cluster incident that produced it. The mechanism is
identical and worth restating, because this batch is where it cost a result:

:class:`~spectramr.config.schemas.training.base.TrainingStrategyConfigSchema` is
``extra="allow"``. A paradigm block the schema never declares is stored as the
plain ``dict`` it arrived as — and every strategy reads its knobs with
``getattr(cfg, "knob", default)``, which on a ``dict`` returns the **default**.
The YAML validates, the key appears in provenance, and the declared value is
discarded. Pitfall #15, and #17 when the discarded value was an ablation's axis.

Measured 2026-08-02 across ``experiments/``: **18 undeclared paradigm blocks
over 20 arms**. The sharpest case is in this module's fMRI pair:

.. code-block:: text

    idea_1_spatial_only_sfc.yaml             declares lambda_t: 0.0
    idea_1_spatiotemporal_adaptive_sfc.yaml  declares lambda_t: 0.01
    ...both resolved to 0.01 -- the ablation and its own baseline were
    runtime-identical, so any measured delta between them was noise.

    idea_2_long_echo_spacing.yaml            declares t_esp: 1.0e-3
    ...resolved to 5.0e-4, and t_esp IS that arm's stated axis.

Declaring the block as a typed model is the whole fix: ``cfg`` stops being a
dict, so the strategies' existing ``getattr`` calls resolve correctly. **No
strategy edit is required.**

Every default below is the value the strategy already falls back to, so an arm
that does not declare the block behaves exactly as it did. Some arms declare
exactly the default — for those the fix is a runtime no-op that restores the
MEANING of the declaration rather than its value.

Landed cohort by cohort: fMRI (4 blocks), MRF (3), novel_2026 (3), spec_tasks (2).

``physics_eq_ssl`` is deliberately EXCLUDED. That strategy already solved the
problem the other way round — it asserts the block IS a dict, reads it with
``pe.get(...)``, and does its own typo rejection
(``physics_equivariant_ssl_strategy.py:138-160``). Mounting a typed block would
make its ``isinstance(pe, dict)`` guard fail and raise ``TypeError`` on the one
arm that uses it. Converting it needs a strategy edit, so it is not part of a
schema-only batch.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..renames import reject_renamed_keys

__all__ = [
    "AdaptiveSFCHSSCTrainingConfigSchema",
    "BeltramiEPIDistortionTrainingConfigSchema",
    "BlochEquivariantTranslationTrainingConfigSchema",
    "CRLBMRFPulseDesignTrainingConfigSchema",
    "ConformalDiffusionReconTrainingConfigSchema",
    "ConformalMRFDictlessReconTrainingConfigSchema",
    "CrossScannerMRFHarmonisationTrainingConfigSchema",
    "DTN2STrainingConfigSchema",
    "IBActiveAcquisitionTrainingConfigSchema",
    "PrivilegedLearningTrainingConfigSchema",
    "RiemannianBlochDiffusionTrainingConfigSchema",
    "SpatiotemporalAdaptiveSFCReconTrainingConfigSchema",
]

#: These blocks are constructor-time hyperparameters, not free-form kwargs, so a
#: typo must raise rather than sit unread — the failure this module exists to
#: end. Matches the 2026-06 batch.
_KNOB_BLOCK = ConfigDict(extra="forbid", frozen=True)


class SpatiotemporalAdaptiveSFCReconTrainingConfigSchema(BaseModel):
    """4-D Beltrami regularisation weights for spatiotemporal fMRI recon.

    Read by ``fmri_kspace_strategies.SpatiotemporalAdaptiveSFCReconStrategy``
    (``:55-57``). ``lambda_t`` is the axis of the ``idea_1_spatial_only_sfc``
    ablation, which declared ``0.0`` and ran at ``0.01``.
    """

    model_config = _KNOB_BLOCK

    lambda_mu: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight of the Beltrami-coefficient (spatial) regulariser.",
    )
    lambda_t: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Weight of the TEMPORAL Beltrami term. Set to 0.0 to ablate the "
            "spatiotemporal coupling and leave a purely spatial regulariser."
        ),
    )


class BeltramiEPIDistortionTrainingConfigSchema(BaseModel):
    """EPI geometric-distortion knobs for the Beltrami distortion strategy.

    Read by ``fmri_kspace_strategies.BeltramiEPIDistortionStrategy``
    (``:133-135``). ``t_esp`` is the axis of the ``idea_2_long_echo_spacing``
    ablation, which declared ``1.0e-3`` and ran at ``5.0e-4``.
    """

    model_config = _KNOB_BLOCK

    t_esp: float = Field(
        default=0.5e-3,
        gt=0.0,
        description=(
            "Echo spacing in SECONDS. Distortion scales with it, so this is the "
            "physical axis an echo-spacing study varies."
        ),
    )
    lambda_mu: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight of the Beltrami-coefficient regulariser on the field.",
    )


class AdaptiveSFCHSSCTrainingConfigSchema(BaseModel):
    """Beltrami-SFC block geometry and loss weights for adaptive HSSC recon.

    Read by ``sfc_conformal_kspace_strategies.AdaptiveSFCHSSCStrategy``
    (``:59-63``). ``grid_size`` / ``in_channels`` are passed to
    ``BeltramiSFCBlock`` at construction, so they size a real module.
    """

    model_config = _KNOB_BLOCK

    grid_size: int = Field(
        default=32,
        ge=1,
        description="Side length of the Beltrami-SFC coefficient grid.",
    )
    in_channels: int = Field(
        default=1, ge=1, description="Input channels of the Beltrami-SFC block."
    )
    lambda_mu: float = Field(
        default=0.01, ge=0.0, description="Weight of the Beltrami-coefficient term."
    )
    lambda_traj: float = Field(
        default=0.01, ge=0.0, description="Weight of the trajectory-fidelity term."
    )


class ConformalDiffusionReconTrainingConfigSchema(BaseModel):
    """Noise range and data-consistency strength for conformal diffusion recon.

    Read by ``sfc_conformal_kspace_strategies.ConformalDiffusionReconStrategy``
    (``:140-143``). ``dc_alpha`` is handed to ``ConformalDataConsistency`` at
    construction.
    """

    model_config = _KNOB_BLOCK

    dc_alpha: float = Field(
        default=1.0,
        ge=0.0,
        description="Strength of the conformal data-consistency projection.",
    )
    sigma_min: float = Field(
        default=1e-2, gt=0.0, description="Lower end of the diffusion noise range."
    )
    sigma_max: float = Field(
        default=1.0, gt=0.0, description="Upper end of the diffusion noise range."
    )


class ConformalMRFDictlessReconTrainingConfigSchema(BaseModel):
    """Conformality weight for dictionary-free MRF reconstruction.

    Read by ``mrf_acquisition_strategies.ConformalMRFDictlessReconStrategy``
    (``:66``). ``idea_3_overcompressed_embedding`` declares ``0.05`` and ran at
    the ``0.01`` default — a 5x under-weighted conformality term.
    """

    model_config = _KNOB_BLOCK

    lambda_conformality: float = Field(
        default=1e-2,
        ge=0.0,
        description=("Weight of the conformality penalty on the fingerprint embedding."),
    )


class CRLBMRFPulseDesignTrainingConfigSchema(BaseModel):
    """Beltrami regularisation weight for CRLB-driven MRF pulse design.

    Read by ``mrf_acquisition_strategies.CRLBMRFPulseDesignStrategy`` (``:185``).
    """

    model_config = _KNOB_BLOCK

    lambda_beltrami: float = Field(
        default=1e-3,
        ge=0.0,
        description="Weight of the Beltrami smoothness term on the pulse train.",
    )


class CrossScannerMRFHarmonisationTrainingConfigSchema(BaseModel):
    """Anchor weight for cross-scanner MRF time-reparameterisation.

    Read by ``mrf_acquisition_strategies.CrossScannerMRFHarmonisationStrategy``
    (``:262``).

    ``tau_eps`` is deliberately NOT declared here. The Beltrami squash epsilon
    lives on the MODEL (``ScannerTimeReparam.forward`` passes its own ``eps`` to
    ``enforce_1d_beltrami_constraint``), so a strategy-level knob would never
    reach it — a silent no-op, which is the defect this module exists to end.
    The strategy says so at ``:257-261``; declaring it here would re-create the
    facade in typed form. Wiring it needs a model-construction change.
    """

    model_config = _KNOB_BLOCK

    lambda_anchor: float = Field(
        default=1e-3,
        ge=0.0,
        description="Weight of the anchor term pinning tau to the reference scanner.",
    )


class BlochEquivariantTranslationTrainingConfigSchema(BaseModel):
    """Bloch sequence parameters and loss weights for ULF->HF translation.

    Read by ``bloch_equivariant_translation_strategy`` (``:63-73``). These are
    physical acquisition constants, not tuning knobs: the ULF/HF T1/T2 scales and
    TR/TE define the forward Bloch model the equivariance term is built on, so a
    discarded value changes the physics the arm claims to enforce.
    """

    model_config = _KNOB_BLOCK

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.bloch_equivariant_translation"))  # type: ignore[arg-type]
    )

    t1_scale_ulf: float = Field(default=0.30, gt=0.0, description="ULF T1 scale factor.")
    t2_scale_ulf: float = Field(default=1.05, gt=0.0, description="ULF T2 scale factor.")
    tr_ulf: float = Field(default=8.0, gt=0.0, description="ULF repetition time (ms).")
    tr_hf: float = Field(default=8.0, gt=0.0, description="HF repetition time (ms).")
    te_ulf: float = Field(default=4.0, gt=0.0, description="ULF echo time (ms).")
    te_hf: float = Field(default=4.0, gt=0.0, description="HF echo time (ms).")
    flip_angle_deg: float = Field(
        default=15.0, gt=0.0, le=180.0, description="Excitation flip angle (degrees)."
    )
    lambda_bloch_eq: float = Field(
        default=1.0, ge=0.0, description="Weight of the Bloch-equivariance term."
    )
    lambda_cycle: float = Field(
        default=0.1, ge=0.0, description="Weight of the cycle-consistency term."
    )
    unpaired_patch: int = Field(default=32, ge=1, description="Patch size for the unpaired branch.")
    enable_bloch_checkpoint: bool = Field(
        default=False, description="Gradient-checkpoint the Bloch simulation."
    )


class IBActiveAcquisitionTrainingConfigSchema(BaseModel):
    """Information-bottleneck weight and forward-model mode for active acquisition.

    Read by ``ib_active_acquisition_strategy`` (``:54-55``). ``mode`` is a
    ``Literal`` here so an unknown value is refused at LOAD; the strategy's own
    ``VALID_MODES`` check then never fires in practice, which is the point — a
    typo should not survive until construction.
    """

    model_config = _KNOB_BLOCK

    beta: float = Field(default=1.0, ge=0.0, description="Information-bottleneck trade-off weight.")
    mode: Literal["linear_gaussian", "nonlinear"] = Field(
        default="linear_gaussian", description="Forward-model family."
    )


class RiemannianBlochDiffusionTrainingConfigSchema(BaseModel):
    """Manifold-diffusion knobs for Riemannian Bloch q-map recon.

    Read by ``riemannian_bloch_diffusion_strategy`` (``:28-39``).
    ``metric_cache_resolution`` defaults to ``0`` (caching OFF);
    ``idea_5_riemannian_qmap_brain`` declares ``32`` and ran with caching
    disabled.
    """

    model_config = _KNOB_BLOCK

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.riemannian_bloch_diffusion"))  # type: ignore[arg-type]
    )

    t_min: float = Field(default=1e-3, gt=0.0, description="Diffusion time floor.")
    t_max: float = Field(default=1.0, gt=0.0, description="Diffusion time ceiling.")
    step_size_init: float = Field(default=0.05, gt=0.0, description="Initial geodesic step size.")
    injectivity_safety_factor: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description="Fraction of the injectivity radius a step may take.",
    )
    metric_cache_resolution: int = Field(
        default=0,
        ge=0,
        description="Metric-tensor cache grid resolution; 0 disables caching.",
    )
    metric_refresh_interval: int = Field(
        default=50, ge=1, description="Epochs between metric-cache refreshes."
    )
    enable_bloch_checkpoint: bool = Field(
        default=False, description="Gradient-checkpoint the Bloch manifold ops."
    )


class PrivilegedLearningTrainingConfigSchema(BaseModel):
    """Loss weights and curriculum for privileged-information distillation.

    Read by ``privileged_learning_strategy`` (``:140-149``).
    ``task_method_iii_privileged`` declares six of these at exactly their
    defaults, so declaring the block is a runtime no-op for it — what changes is
    that the declaration now MEANS something and a typo raises.
    """

    model_config = _KNOB_BLOCK

    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("training.privileged"))  # type: ignore[arg-type]
    )

    alpha: float = Field(
        default=1.0, ge=0.0, description="Weight of the privileged-distillation term."
    )
    beta: float = Field(default=0.5, ge=0.0, description="Weight of the domain-adversarial term.")
    delta: float = Field(default=0.1, ge=0.0, description="Weight of the feature-alignment term.")
    curriculum_tau0: float = Field(
        default=5.0,
        gt=0.0,
        description="Initial curriculum temperature; decays over training.",
    )
    min_keep_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Floor of the marker keep-fraction the curriculum decays towards.",
    )
    initial_keep_fraction: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Marker keep-fraction at epoch 0, decayed by exp(-epoch / curriculum_tau0).",
    )
    disc_hidden_dim: int = Field(
        default=128, ge=1, description="Hidden width of the domain discriminator."
    )


class DTN2STrainingConfigSchema(BaseModel):
    """Receptive window for the DTN2S mask.

    Read by ``dtn2s_strategy`` (``:67-70``). That site is worth noting: it guards
    with ``hasattr(self.config.training, "dtn2s")``, which under ``extra="allow"``
    is True whenever the YAML supplies the block — so the guard passed and the
    value was still discarded, because ``getattr`` then ran against a dict. A
    presence check cannot substitute for a type.
    """

    model_config = _KNOB_BLOCK

    receptive_window: int = Field(
        default=8, ge=1, description="Half-width of the DTN2S coupling window."
    )
