"""Data configuration schema.

This module contains the unified configuration for data loading.
CLEANED: Removed legacy specific path keys (lr_volume, ulf_root, etc.)
in favor of a unified 'data_root' + 'dataset_type' strategy.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.config.schemas.collation import CollationConfigSchema
from spectramr.shared.utils.path_normalizer import PathNormalizer

from .renames import (
    default_knob,
    fold_renamed_keys,
    folded_input_keys,
    folded_input_paths,
    reject_renamed_keys,
)

__all__ = [
    "AcquisitionMetadataConfigSchema",
    "AcquisitionParamsSchema",
    "AugmentationConfigSchema",
    "CachingPolicy",
    "CoilsConfigSchema",
    "CollationConfigSchema",
    "ContrastConfigSchema",
    "CoordinateEmissionConfigSchema",
    "DataConfigSchema",
    "DataExposeConfigSchema",
    "DataLoaderConfigSchema",
    "DatasetSourceSchema",
    "DomainConfigSchema",
    "LatentDiffusionConfigSchema",
    "MRIxFieldsDataConfigSchema",
    "ManifestRoleConfigSchema",
    "MetaLearningConfigSchema",
    "ModeOutputSchema",
    "MultiChannelOutputSchema",
    "MultiContrastConfigSchema",
    "MultiDomainConfigSchema",
    "OutputChannelSchema",
    "PriorLoadingConfigSchema",
    "QuantitativeConfigSchema",
    "ReferenceTissuePanelConfig",
    "TemporalConfigSchema",
    "TemporalOutputSchema",
]


#: The dataset types a config may declare, and the ONLY list any consumer may
#: use to describe them. Hoisted to module scope 2026-08-04 because there were
#: three divergent copies: this schema's validator, the "Known types:" string in
#: ``DatasetInstantiator.create_datasets``' raise -- which omitted ``mrixfields``
#: and ``oracle_bssfp`` (87 arms' worth of real types) while listing ten alias
#: spellings that can never reach it -- and the prose in
#: ``consolidated_dataset_factory``. A user with a typo was shown a list that
#: was wrong in both directions.
#:
#: ``graph_mri`` was REMOVED: it was canonical, in ``_SELF_INDEXED_DATASET_TYPES``
#: and in both collation maps, but no branch in ``DatasetInstantiator`` ever
#: constructed it, so declaring it raised "not recognised" -- from a message that
#: did not list it either. 0 arms declared it. The graph paradigm is served by
#: any dataset_type plus ``data.processing.transforms: [{name: graph_encoding}]``,
#: not by a dedicated dataset type. Re-add it only WITH an instantiator branch.
CANONICAL_DATASET_TYPES: tuple[str, ...] = (
    "kspace",
    "m4raw",
    "nifti",
    "nifti_paired",
    "contrast_aware_paired",
    "npy_slice",
    "image",
    "dicom",
    "synthetic",
    "preprocessed",
    "pde_synthetic",
    "quantitative",
    "cine",
    "bart_kspace",
    "bids_paired",
    "png_paired",
    "field_ref",
    "ismrmrd_kspace",
    "oracle_bssfp",
    "mrixfields",
    "fmri",
)

#: Legacy spellings folded to a canonical type BEFORE dispatch. Because the fold
#: happens in the validator, a downstream ``if dataset_type == "<alias>"`` branch
#: is unreachable -- ten such labels sat in ``DatasetInstantiator`` looking live.
DATASET_TYPE_ALIASES: dict[str, str] = {
    "fastmri_kspace": "kspace",
    "fastmri_knee": "kspace",
    "fastmri_brain": "kspace",
    "m4raw_multicoil": "m4raw",
    "volume_h5": "kspace",
    "3d_volumetric": "nifti",
    "3d": "nifti",
    "image_folder": "image",
    "2d": "image",
    "folder": "image",
    "paired_slices": "npy_slice",
    "slice_paired": "npy_slice",
    "npy_paired": "npy_slice",
    "dicom_series": "dicom",
    "paired_nifti": "nifti_paired",
    "paired_mri": "nifti_paired",
}


class AcquisitionParamsSchema(BaseModel):
    """Physical acquisition parameters for a single contrast.

    Used by ``AcquisitionEmbedding`` (encodes the vector into a
    learnable embedding) and by ``BlochConsistencyLoss`` (drives
    parameter-map generators). Setting this block enables
    physics-informed conditioning beyond a bare contrast-id lookup.

    Field units follow the SI convention used everywhere in
    ``src/infrastructure/physics/``:

    - ``TE``, ``TR``, ``TI``  in milliseconds
    - ``FA``                   in degrees
    - ``B0``                   in tesla
    - ``contrast_type``        ``spin_echo`` | ``inversion_recovery`` | ``gradient_echo`` | ``diffusion_weighted``
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    name: str = Field(..., description="Contrast label (T1w / T2w / FLAIR / PDw / SWI / ...).")
    TE: float = Field(..., gt=0.0, description="Echo time in milliseconds.")
    TR: float = Field(..., gt=0.0, description="Repetition time in milliseconds.")
    TI: float = Field(
        default=-1.0,
        description=("Inversion time in milliseconds, or -1 sentinel for sequences without IR."),
    )
    FA: float = Field(default=90.0, gt=0.0, le=180.0, description="Flip angle in degrees.")
    B0: float = Field(default=3.0, gt=0.0, description="Main field strength in tesla.")
    contrast_type: Literal[
        "spin_echo",
        "inversion_recovery",
        "gradient_echo",
        "diffusion_weighted",
        "ssfp",
        "mprage",
    ] = Field(
        default="spin_echo",
        description="Pulse-sequence family — selects the Bloch equation variant.",
    )
    include_concomitant: bool | None = Field(
        default=None,
        description=(
            "Whether to fold the Maxwell concomitant phase into the Bloch "
            "forward (PMPS Phase-0). ``None`` defers to the auto-rule "
            "(enabled when ``B0 < 0.5`` T); ``True``/``False`` pins the "
            "behaviour explicitly. The Tier-1 check "
            "``concomitant_required_at_ulf`` enforces this auto-rule at "
            "YAML-validation time."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _auto_set_concomitant_at_ulf(cls, data: Any) -> Any:
        """Auto-set ``include_concomitant=True`` when ``B0 < 0.5`` T.

        The 1/B0 scaling of the concomitant phase (Bernstein 1998) makes
        the correction dominant below 0.1 T and material below 0.5 T. We
        enforce it at schema-validation so a downstream YAML cannot
        silently omit the correction at ULF. Explicit ``include_concomitant``
        overrides the auto-rule. Uses ``mode="before"`` so the field is
        seeded on the raw dict before frozen-model validation.
        """
        if isinstance(data, dict):
            if data.get("include_concomitant") is None:
                b0 = data.get("B0", 3.0)
                try:
                    b0_t = float(b0)
                except (TypeError, ValueError):
                    return data
                if b0_t < 0.5:
                    data = {**data, "include_concomitant": True}
        return data


class ReferenceTissuePanelConfig(BaseModel):
    """Reference-tissue panel for :class:`BlochSignalEncoder` (PMPS Phase-0).

    Configures the fixed tissue set whose Bloch signatures form the
    basis of the acquisition-vector embedding. ``tissues`` selects entries
    by name from the default panel
    (``gm | wm | csf | fat | lesion_t1 | lesion_t2``); custom tissue
    physical parameters are added in a future revision.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    tissues: list[Literal["gm", "wm", "csf", "fat", "lesion_t1", "lesion_t2"]] = Field(
        default_factory=lambda: ["gm", "wm", "csf"],
        description="Reference-tissue names selected from the default panel.",
    )
    n_samples_per_tissue: int = Field(
        default=8,
        ge=1,
        description=(
            "Number of perturbed samples per tissue when training the "
            "BlochSignalEncoder. Higher values smooth the embedding "
            "manifold; 8 is sufficient for protocol-design discrimination."
        ),
    )

    @field_validator("tissues")
    @classmethod
    def _at_least_one_tissue(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("reference_panel.tissues must list at least one tissue.")
        return v


class MultiContrastConfigSchema(BaseModel):
    """Pattern C — per-sample contrast-id conditioning (FiLM).

    Opt-in toggle for training a single model on a heterogeneous cohort
    (T1/T2/FLAIR/PD) where each sample carries a contrast label. When
    ``enabled=True`` the data pipeline emits a ``contrast_idx`` long tensor
    on every batch and the strategy passes it to ``model.forward(...,
    contrast_idx=...)``. The model is responsible for consuming it (typically
    via ``nn.Embedding(n_contrasts, embed_dim)`` feeding a FiLM γ/β MLP);
    the Tier-1 audit ``multi_contrast_model_support`` guards against silently
    selecting a model that ignores the id. It checks **both** consumers --
    ``model.model_type`` and, when one is declared,
    ``model.discriminator_component.name`` -- because an unconditioned critic
    drops ``contrast_idx`` just as silently as an unconditioned generator
    (#1931).

    Distinguish from:

    - ``input_contrast`` / ``target_contrast`` — Pattern A (cross-contrast
      *translation*, fixed source→target normalization). Independent of this
      block.
    - ``MultiContrastFusionGenerator`` — Pattern B (channel-stacked fusion
      where every sample contains *all* contrasts). Does not use ``contrast_idx``.
    """

    # extra="ignore" so v6.1 additive sub-fields don't break v6.0 YAMLs that
    # carry stray keys; explicit fields below still type-check at load time.
    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle for Pattern C contrast-id conditioning.",
    )
    n_contrasts: int = Field(
        default=4,
        ge=1,
        description=(
            "Cohort cardinality (e.g. 4 = T1/T2/FLAIR/PD). Must match the "
            "model's ``nn.Embedding(n_contrasts, ...)`` lookup table size."
        ),
    )
    embed_dim: int = Field(
        default=8,
        ge=1,
        description=(
            "Contrast-embedding dimension fed into the FiLM MLP. The model "
            "may override via ``model_kwargs`` if it needs a different size."
        ),
    )
    contrast_map: dict[str, int] = Field(
        default_factory=lambda: {"T1": 0, "T2": 1, "FLAIR": 2, "PD": 3},
        description=(
            "Mapping from contrast name (case-insensitive substring of the "
            "filename) to integer id. Datasets like M4Raw use this to derive "
            "``contrast_idx`` automatically. Override to extend the cohort."
        ),
    )
    acquisition_params: list[AcquisitionParamsSchema] | None = Field(
        default=None,
        description=(
            "Optional per-contrast physical acquisition parameters "
            "(TE/TR/TI/FA/B0/contrast_type). When set, the strategy passes "
            "the per-sample (TE,TR,TI,FA,B0) into the model's "
            "AcquisitionEmbedding instead of (or in addition to) the bare "
            "contrast_idx — allowing harmonisation across vendors and field "
            "strengths and unlocking BlochConsistencyLoss for parameter-map "
            "bottlenecks. Indices must align with `contrast_map`. "
            "See docs/multi_contrast.md."
        ),
    )

    # === v6.1 — multi-contrast SSL extensions (Item 1 of multi-contrast remaining) ===
    positive_sampling: str = Field(
        default="co_registered_subject",
        description=(
            "InfoNCE positive-pair strategy for multi-contrast SSL. "
            "Options: ``co_registered_subject`` | ``same_volume_slice``."
        ),
    )
    negative_sampling: str = Field(
        default="cross_subject",
        description="InfoNCE negative-pair strategy: ``cross_subject`` | ``cross_volume``.",
    )

    # === v6.1 — vendor / protocol soft-prompt extensions (Item 4) ===
    vendor_map: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Optional mapping from vendor / protocol identifier to integer id, "
            "consumed by ``VendorPromptEmbedding``. Empty dict disables vendor "
            "conditioning (v6.0 semantics)."
        ),
    )
    n_vendors: int = Field(
        default=0,
        ge=0,
        description=(
            "Cohort cardinality for vendor/protocol soft-prompt embeddings. "
            "Must equal ``len(vendor_map)`` when both are set."
        ),
    )

    # === PMPS Phase-0 (2026-05-19) — Bloch-grounded acquisition encoding ===
    bloch_grounded: bool = Field(
        default=False,
        description=(
            "Switch contrast conditioning from a categorical lookup to a "
            "Bloch-signature embedding on a fixed reference-tissue panel. "
            "When ``True`` the strategy mounts the ``bloch_signal_encoder`` "
            "block (registered in models/blocks/bloch_signal_encoder.py) and "
            "passes the acquisition vector ``(TE, TR, TI, FA, B0)`` through it "
            "instead of (or in addition to) the bare contrast id. Audit "
            "check ``bloch_grounded_requires_reference_panel`` enforces the "
            "well-formedness of the panel when this is set."
        ),
    )
    reference_panel: ReferenceTissuePanelConfig | None = Field(
        default=None,
        description=(
            "Reference-tissue panel consumed by the BlochSignalEncoder. "
            "Required when ``bloch_grounded=True``."
        ),
    )

    @model_validator(mode="after")
    def _validate_bloch_grounded(self) -> "MultiContrastConfigSchema":
        """Reject Bloch-grounded conditioning without acquisition_params/panel.

        Re-using the existing ``acquisition_params`` field for the
        physical protocol means the Tier-1 audit can validate both
        Bloch-signature embeddings and parameter-map bottlenecks with a
        single sub-block. The audit equivalent of this check
        (``bloch_grounded_requires_reference_panel``,
        ``acquisition_params_required_for_bloch_grounded``) lives in
        ``infrastructure/validation/pmps_checks.py`` for actionable error
        messages at audit time; the schema-level guard catches the same
        cases at load time.
        """
        if self.bloch_grounded:
            if self.reference_panel is None:
                raise ValueError(
                    "multi_contrast.bloch_grounded=True requires "
                    "multi_contrast.reference_panel. Add at minimum "
                    "`reference_panel: {tissues: [gm, wm, csf]}`."
                )
            if not self.acquisition_params:
                raise ValueError(
                    "multi_contrast.bloch_grounded=True requires "
                    "multi_contrast.acquisition_params (non-empty). The "
                    "BlochSignalEncoder needs (TE, TR, TI, FA, B0) per "
                    "contrast to produce embeddings."
                )
        return self


class ContrastConfigSchema(BaseModel):
    """Configuration for MRI contrast-specific normalization.

    Used in contrast-aware paired datasets where input and target
    have different contrasts requiring separate normalization strategies.

    Example YAML:
        input_contrast:
          name: ULF_64mT
          normalization: percentile
          percentile: 99.5
          out_range: [0.0, 1.0]
          clamp: true
          keywords: ['64mt', 'ulf', 'lf']
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)

    name: str = Field(
        ..., description="Contrast identifier (e.g., 'T1w', 'T2w', 'ULF_64mT', 'HF_3T')"
    )
    normalization: Literal["percentile", "zscore", "minmax", "none"] = Field(
        default="percentile",
        description="Normalization strategy: 'percentile' (robust), 'zscore' (mean/std), 'minmax' (0-1), 'none'",
    )
    percentile: float = Field(
        default=99.5,
        ge=0.0,
        le=100.0,
        description="Upper percentile for robust normalization (0-100)",
    )
    out_range: tuple[float, float] = Field(
        default=(0.0, 1.0), description="Output range after normalization (min, max)"
    )
    clamp: bool = Field(default=True, description="Whether to clamp values to out_range")
    noise_gate: bool = Field(
        default=False,
        description="Enable ULF noise gating (hard-clip background noise)",
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to match in filenames for contrast detection (e.g., ['64mt', 'ulf'])",
    )


class ManifestRoleConfigSchema(BaseModel):
    """Configuration for manifest role assignments in data loading.

    Defines which manifest pickles serve as inputs vs targets.
    Supports multiple inputs for SSL and multiple targets for multi-task learning.

    Example YAML:
        manifest_roles:
          inputs:
            - manifest: "fastmri_brain_multicoil_kspace.pkl"
              key: "kspace_input"
            - manifest: "fastmri_brain_multicoil_image_normalized.pkl"
              key: "image_auxiliary"  # SSL secondary input
          targets:
            - manifest: "fastmri_brain_multicoil_image_gt.pkl"
              key: "target"
          auxiliary:
            - manifest: "fastmri_brain_multicoil_coil_sensitivity.pkl"
              key: "sensitivity_maps"
    """

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)

    # Input manifests (can be multiple for SSL)
    inputs: list[dict[str, str]] = Field(
        default_factory=lambda: [{"manifest": "", "key": "input"}],
        description="List of input manifests with keys. Each dict has 'manifest' (pkl path) and 'key' (tensor name)",
    )

    # Target manifests
    targets: list[dict[str, str]] = Field(
        default_factory=lambda: [{"manifest": "", "key": "target"}],
        description="List of target manifests with keys",
    )

    # Auxiliary data (coil sensitivity, etc.)
    auxiliary: list[dict[str, str]] = Field(
        default_factory=list,
        description="List of auxiliary manifests (e.g., coil sensitivity maps)",
    )


class PriorLoadingConfigSchema(BaseModel):
    """Configuration for loading priors from other models."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)
    enabled: bool = Field(default=False)
    source: str = Field(default="")
    checkpoint_path: str = Field(default="")
    freeze: bool = Field(default=True)


class CachingPolicy(BaseModel):
    """Configuration for data staging and caching strategy."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)
    strategy: Literal["none", "ram", "disk"] = Field(default="none")
    cache_dir: str | None = Field(default=None)
    staging_behavior: Literal["full", "progressive"] = Field(default="full")
    cleanup_trigger: Literal["session_end", "epoch_end", "never"] = Field(default="session_end")


class DatasetSourceSchema(BaseModel):
    """Definition of dataset sources for multi-dataset or SSL workflows."""

    model_config = ConfigDict(protected_namespaces=(), extra="ignore", frozen=True)
    name: str = Field(default="default")
    variant: str = Field(default="2d_slices")
    split: Literal["train", "val", "both"] = Field(default="train")
    weight: float = Field(default=1.0, gt=0)
    data_path: str | None = Field(default=None)
    path: str | None = Field(default=None)  # Alias for data_path
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("variant")
    @classmethod
    def validate_variant(cls, value: str) -> str:
        """Validate and consolidate variant types."""
        value = value.lower()

        # Consolidated variant types (4 core types)
        valid = [
            "2d_slices",  # Extract 2D slices from 3D volumes
            "2d_full",  # Full 2D images (no slicing)
            "3d_patches",  # Random 3D patches from volumes
            "3d_full",  # Full 3D volumes
        ]

        # Alias mapping for backward compatibility
        aliases = {
            "2d": "2d_slices",
            "3d": "3d_patches",
            "3d_volumetric": "3d_patches",
            "slices": "2d_slices",
            "patches": "3d_patches",
            "volume": "3d_full",
            "full": "2d_full",
        }

        # Map alias to canonical type
        if value in aliases:
            value = aliases[value]

        # Unknown variants must raise — never silently pass through (NN#3).
        # Legacy model-architecture strings (attention_unet, fast_kan) belong in
        # model.model_type, not data.datasets[].variant.
        if value not in valid:
            raise ValueError(
                f"DatasetSourceSchema.variant {value!r} is not a recognised "
                f"variant. Valid values: {valid}. "
                "Legacy model-architecture strings (attention_unet, fast_kan) "
                "belong in model.model_type, not data.datasets[].variant."
            )

        return value


# ═══════════════════════════════════════════════════════════════════════════
# Mode-aware data schema (Phase 3 of TODO/audit/data_layer_unification_plan.md)
# ═══════════════════════════════════════════════════════════════════════════
#
# Pipelines ask the data layer for a loader keyed by *mode*:
#   - ``train``  — random patches, augmentation on, shuffling on
#   - ``val``    — deterministic patches or full volumes, no aug
#   - ``infer``  — full volume OR sliding-window tiling, no aug, deterministic
#   - ``eval``   — reference loading for post-hoc evaluation
#
# The per-mode behavior lives under ``DataConfigSchema.modes``. When that
# block is absent (legacy YAMLs), the ``derive_modes_from_legacy`` model
# validator fills it in from top-level fields, preserving current behavior.
#
# Later phases extend ``ModeSamplerSchema`` with ``grid`` / ``temporal_*``
# sampler types and ``ModeOutputSchema`` / ``ModeEvalRoleSchema`` for
# inference output ordering and eval reference roles. Phase 3 keeps the
# schema minimal: enough surface for the legacy-fallback shim to populate
# and for builders to consume mode-aware values via a single
# ``DataConfigSchema.resolve_mode(mode)`` helper.


SamplerType = Literal[
    "uniform",  # tio.UniformSampler — random spatial patches (train default)
    "full",  # no sampler — whole-volume forward pass (val/infer default)
    "grid",  # tio.GridSampler — sliding-window tiling (Phase 2)
    "label",  # tio.LabelSampler — Phase 5
    "weighted",  # tio.WeightedSampler — Phase 5
    "temporal_uniform",  # 4D random temporal patches (Phase 4c)
    "temporal_grid",  # 4D sliding-window tiling (Phase 4c)
]


class ModeSamplerSchema(BaseModel):
    """Per-mode sampler configuration.

    Phase-3 surface intentionally narrow: enough to override the legacy
    ``samples_per_volume`` / ``queue_length`` / ``use_queue_for_validation``
    fields on a per-mode basis. Phase 2 fills in ``patch_overlap`` /
    ``aggregator`` for grid sampling; Phase 4c fills in temporal options;
    Phase 5 fills in label_* and weight_* for non-uniform samplers.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    type: SamplerType = Field(
        default="uniform",
        description="Sampler kind. 'uniform' = random patches; 'full' = whole "
        "volume; 'grid' = sliding-window (Phase 2); 'label' = class-balanced "
        "from a label map (Phase 5); 'weighted' = probability-map weighted "
        "(Phase 5); 'temporal_*' = 4D (Phase 4c).",
    )
    samples_per_volume: int = Field(
        default=4,
        ge=1,
        description="Random patches per subject per epoch (for type=uniform).",
    )
    queue_length: int = Field(
        default=100,
        ge=1,
        description="TorchIO patch queue RAM buffer length.",
    )
    shuffle: bool = Field(
        default=True,
        description="Shuffle subjects in the queue. Inferred from mode "
        "(true for train, false for val/infer/eval) when modes are derived "
        "from legacy fields.",
    )

    # Phase-2 fields (declared now so Phase 2 doesn't touch the schema again):
    patch_overlap: tuple[int, int, int] | None = Field(
        default=None,
        description="Per-axis overlap for type='grid' (Phase 2). None = auto.",
    )
    aggregator: Literal["average", "hann", "crop"] | None = Field(
        default=None,
        description="GridAggregator strategy for type='grid' (Phase 2). "
        "'hann' recommended for smooth tile borders.",
    )

    # ── Phase 5 — non-uniform sampler variants ──────────────────────────
    label_name: str | None = Field(
        default=None,
        description="(type='label'): name of the label map in the "
        "TorchIO Subject from which to sample. Required for type='label'.",
    )
    label_probabilities: dict[int, float] | None = Field(
        default=None,
        description="(type='label'): per-class sampling probabilities. "
        "Keys are label integer values, values are probabilities (need not "
        "sum to 1 — TorchIO normalizes). Use to oversample minority classes "
        "(lesions, pathology) relative to background.",
    )
    probability_map: str | None = Field(
        default=None,
        description="(type='weighted'): name of the per-voxel probability "
        "map in the TorchIO Subject. Required for type='weighted'. Use to "
        "bias sampling toward, e.g., foreground voxels via a brain mask.",
    )

    @model_validator(mode="after")
    def validate_sampler_required_fields(self) -> "ModeSamplerSchema":
        """Phase 5: validate sampler-specific required fields."""
        if self.type == "label" and self.label_name is None:
            raise ValueError(
                "sampler.type='label' requires sampler.label_name to be set "
                "(name of the label image in the TorchIO Subject)."
            )
        if self.type == "weighted" and self.probability_map is None:
            raise ValueError(
                "sampler.type='weighted' requires sampler.probability_map "
                "to be set (name of the per-voxel probability image in "
                "the TorchIO Subject)."
            )
        return self


class ModeConfigSchema(BaseModel):
    """Per-mode behavior block (one of train/val/infer/eval).

    Phase 3 covers ``sampler`` and ``augmentation_enabled``. Phase 2 adds
    ``strict_train_parity`` and ``output``; Phase 1.7 fills in ``role``
    and ``reference_paths`` for ``mode=eval``.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    sampler: ModeSamplerSchema = Field(
        default_factory=ModeSamplerSchema,
        description="Sampler config for this mode.",
    )
    augmentation_enabled: bool = Field(
        default=False,
        description="Whether to apply augmentation pipeline in this mode. "
        "Train defaults true; val/infer/eval default false.",
    )
    transforms_override: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Optional per-mode extra transforms (Phase 4+). Empty "
        "list inherits the shared transform chain.",
    )

    # ── Phase-2 placeholders (infer-only) ───────────────────────────────────
    strict_train_parity: bool = Field(
        default=False,
        description="(infer mode, Phase 2): refuse to load if the resolved "
        "transform chain diverges from the training-time chain recorded in "
        "the checkpoint metadata.",
    )

    # ── Phase-4d output ordering (infer-only) ───────────────────────────────
    output: "ModeOutputSchema | None" = Field(
        default=None,
        description="(infer mode, Phase 4d): on-disk writer config. "
        "Controls format, per-channel naming/denormalization, and "
        "frame-order preservation for cine outputs. None ⇒ legacy "
        "single-tensor write path.",
    )

    # ── Phase-1.7 placeholders (eval-only) ──────────────────────────────────
    role: Literal["clean_reference", "metric_ref", "uncertainty", "manifest"] | None = Field(
        default=None,
        description="(eval mode): kind of reference artifact this loader "
        "yields. None outside eval mode.",
    )
    reference_paths: list[str] = Field(
        default_factory=list,
        description="(eval mode): paths for reference loading. Consumed by "
        "spectramr.data.builders.eval_dataset_builder.",
    )


class DataModesSchema(BaseModel):
    """Top-level mode-dispatch block under ``data.modes``.

    Each field is optional; absent modes are populated by the
    ``derive_modes_from_legacy`` validator on :class:`DataConfigSchema`.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    train: ModeConfigSchema | None = Field(default=None)
    val: ModeConfigSchema | None = Field(default=None)
    infer: ModeConfigSchema | None = Field(default=None)
    eval: ModeConfigSchema | None = Field(default=None)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Resampling configurability
# ═══════════════════════════════════════════════════════════════════════════
#
# Today, ``_ResampleToReferenceTransform`` and ``_EnsureMinimumSpatialSize``
# are hardcoded inside ``TorchIOTransformBuilder``. Phase 4 exposes them
# via YAML so users can declare:
#
#   data:
#     resample:
#       enabled: true
#       strategy: target_shape
#       target_shape: [256, 256, 256]
#       interpolation: bspline
#       anti_aliasing: true
#     crop_or_pad:
#       enabled: true
#       target_shape: [256, 256, 1]
#       padding_mode: reflect      # reflect avoids k-space DC bleed
#
# This closes the variable-shape ULF/HF question (different volume sizes
# need to be normalized to a canonical bounding box BEFORE the model
# sees them — anti-aliasing on downsample is non-negotiable for
# k-space-aware training).


ResampleStrategy = Literal["reference", "isotropic"]
Interpolation = Literal["nearest", "linear", "bspline"]
PaddingMode = Literal["zero", "reflect", "edge", "mean"]
CropStrategy = Literal["center", "foreground", "corner"]


class ResampleConfigSchema(BaseModel):
    """Spatial resampling configuration for variable-shape MRI data.

    Three strategies, all opt-in via ``enabled=true``:

    - ``reference``: use the legacy hardcoded ``_ResampleToReferenceTransform``
      which infers a target affine from the first subject. Preserves
      pre-Phase-4 behavior.
    - ``isotropic``: resample every subject to a fixed voxel spacing
      (in mm). Useful when input data has heterogeneous resolution
      (e.g. ULF 1.5×1.5×4 mm and HF 1×1×1 mm).

    For fixed-shape canonical-bounding-box normalization (independent
    of spacing), see :class:`CropOrPadConfigSchema` — the two compose
    cleanly: resample to a canonical spacing, then crop/pad to a
    canonical shape.

    ``anti_aliasing=true`` applies a Gaussian pre-filter on downsample
    paths — critical for k-space training because aliasing in image
    space leaks into k-space high frequencies.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master switch. When False (default), the hardcoded "
        "_ResampleToReferenceTransform path is used (preserves "
        "pre-Phase-4 behavior).",
    )
    strategy: ResampleStrategy = Field(
        default="reference",
        description="Resampling strategy — see class docstring.",
    )
    target_spacing: tuple[float, float, float] | None = Field(
        default=None,
        description="Voxel spacing in mm for strategy='isotropic'. "
        "Required when strategy='isotropic'.",
    )
    interpolation: Interpolation = Field(
        default="bspline",
        description="Interpolation kernel. ``bspline`` is the TorchIO "
        "default for intensity images; use ``nearest`` for label maps.",
    )
    anti_aliasing: bool = Field(
        default=True,
        description="Gaussian pre-filter on downsample. CRITICAL for "
        "k-space-aware training — aliasing in image space leaks into "
        "k-space high frequencies. Leave on unless you have a strong reason.",
    )

    @model_validator(mode="after")
    def validate_strategy_requirements(self) -> "ResampleConfigSchema":
        if not self.enabled:
            return self
        if self.strategy == "isotropic" and self.target_spacing is None:
            raise ValueError(
                "resample.strategy='isotropic' requires "
                "resample.target_spacing to be set (e.g. [1.0, 1.0, 1.0])."
            )
        return self


class CropOrPadConfigSchema(BaseModel):
    """Crop-or-pad to a canonical voxel grid before the model.

    Complements :class:`ResampleConfigSchema` — resample changes voxel
    spacing; crop_or_pad changes voxel COUNT without changing spacing.

    ``padding_mode: reflect`` is the recommended default for MRI:
    zero-padding leaves a hard intensity discontinuity at the volume
    border that bleeds into k-space DC. ``reflect`` keeps the image
    smooth.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, the hardcoded "
        "_EnsureMinimumSpatialSize transform handles min-size padding "
        "(preserves pre-Phase-4 behavior).",
    )
    target_shape: tuple[int, int, int] | None = Field(
        default=None,
        description="Canonical voxel grid shape (H, W, D). Required when enabled.",
    )
    padding_mode: PaddingMode = Field(
        default="reflect",
        description="Padding mode. 'reflect' avoids the zero-padding-induced "
        "k-space DC bleed; 'zero' matches numpy default; 'edge' replicates "
        "the border voxel; 'mean' fills with the volume mean.",
    )
    crop_strategy: CropStrategy = Field(
        default="center",
        description="When the input is larger than target_shape, how to "
        "select the crop. 'center' is symmetric; 'foreground' centers on "
        "the foreground bounding box (better for skull-stripped brains); "
        "'corner' crops the (0,0,0) corner.",
    )

    @model_validator(mode="after")
    def validate_target_shape_when_enabled(self) -> "CropOrPadConfigSchema":
        if self.enabled and self.target_shape is None:
            raise ValueError(
                "crop_or_pad.enabled=true requires crop_or_pad.target_shape "
                "to be set (e.g. [256, 256, 1])."
            )
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4b — Quantitative / Bloch / multi-echo / DWI
# ═══════════════════════════════════════════════════════════════════════════
#
# Two orthogonal additions:
#
#   data.quantitative.*         — per-batch target maps (T1, T2, T2*, PD, ADC).
#                                 Dataset yields a dict-keyed batch with these
#                                 as separate tensors. Strategies that consume
#                                 quantitative outputs (e.g., parameter-map
#                                 generators, BlochConsistencyLoss) read them
#                                 by name.
#   data.acquisition_metadata.* — per-sample acquisition scalars (TE, TR, TI,
#                                 FA, B0). Ingested from BIDS-JSON sidecars,
#                                 DICOM headers, H5 attrs, or an explicit map.
#                                 Threads into model conditioning and Bloch
#                                 loss residuals.
#
# Multi-echo and DWI are special cases of acquisition metadata: each echo /
# diffusion-encoded volume carries its own (TE, b-value, bvec) tuple. The
# dataset emits an extra channel axis with the corresponding metadata list.


QuantitativeMap = Literal["t1", "t2", "t2star", "pd", "adc", "fa_dti"]
AcquisitionMetadataSource = Literal[
    "bids_json",  # Sibling .json sidecars (BIDS-style)
    "dicom",  # DICOM headers (TE = DICOM tag (0018, 0081), etc.)
    "h5_attrs",  # HDF5 root attributes (FastMRI-style)
    "yaml_inline",  # Explicit dict on the config (smoke / synthetic / unit-test)
]


class QuantitativeConfigSchema(BaseModel):
    """Quantitative parameter-map ingestion (T1 / T2 / PD / ...).

    When ``enabled=true``, :class:`~spectramr.data.datasets.quantitative_dataset.
    QuantitativeMapDataset` yields a Subject that collates to::

        {
            "input":  Tensor[B, C_in, X, Y, Z],   # stacked ``input_paths``
            "target": Tensor[B, C_out, X, Y, Z],  # ``target_maps``, in order
            "t1":     Tensor[B, 1, X, Y, Z],      # if "t1" in target_maps
            "t2":     Tensor[B, 1, X, Y, Z],
            "pd":     Tensor[B, 1, X, Y, Z],
        }

    Maps are stored in physical units (T1, T2, T2* in ms; PD unitless 0-1;
    ADC in mm²/s × 1e-3; FA unitless 0-1). The dataset does NOT rescale to
    [-1, 1] — Bloch-aware losses need physical magnitudes.

    Used by:

    - Bloch consistency losses (``src/models/losses/bloch_loss.py``)
    - Parameter-map generators (``src/models/generators/bloch_consistency.py``)
    - Quantitative-MRI eval campaigns (per-map PSNR / RMSE)

    .. warning::

       Until 2026-08-05 this docstring advertised ``input_contrasts`` and
       ``target`` keys that the dataset **never emitted** — it produced
       ``input`` plus the named maps and nothing else, so every batch was
       rejected by ``BatchAdapter.from_dict``, which requires both canonical
       keys. ``dataset_type: quantitative`` could not serve a single step.
       ``target`` is now real (the declared maps, stacked in ``target_maps``
       order) and ``input_contrasts`` is gone, because nothing ever read it.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, no parameter maps are loaded "
        "or emitted on the batch (preserves pre-Phase-4b behavior).",
    )
    target_maps: list[QuantitativeMap] = Field(
        default_factory=list,
        description="Which parameter maps to load and emit on every batch. "
        "Each entry yields a separate dict key in the batch.",
    )
    map_paths: dict[str, str] = Field(
        default_factory=dict,
        description="Optional explicit path template per map (e.g. "
        "{'t1': '{subject}/{session}/anat/{subject}_{session}_T1map.nii.gz'}). "
        "When absent, the dataset falls back to glob-by-suffix.",
    )
    units: Literal["physical", "normalized"] = Field(
        default="physical",
        description="``physical`` keeps maps in SI-derived units (ms, mm²/s); "
        "``normalized`` z-scores or min-max scales for visualization. "
        "Bloch losses require ``physical``.",
    )
    input_source: Literal["contrasts", "maps"] | None = Field(
        default=None,
        description="What each record's ``input_paths`` actually are, and "
        "therefore whether ``input`` and ``target`` are the same tensor. "
        "``contrasts``: separate weighted images (T1w/T2w/PDw) the maps are "
        "predicted from -- the parameter-mapping task, input != target. "
        "``maps``: the input_paths ARE the target maps, so input == target -- "
        "correct for generative modelling of the map distribution (a diffusion "
        "strategy noises the target itself), and what "
        "scripts/data/build_nist_mrf_manifest.py emits. There is no default: "
        "the two are indistinguishable from the config alone, and guessing "
        "``contrasts`` on a maps-style manifest silently trains the identity.",
    )

    @model_validator(mode="after")
    def validate_target_maps_when_enabled(self) -> "QuantitativeConfigSchema":
        if self.enabled and not self.target_maps:
            raise ValueError(
                "quantitative.enabled=true requires quantitative.target_maps "
                "to be a non-empty list (e.g. ['t1', 't2', 'pd'])."
            )
        return self

    @model_validator(mode="after")
    def validate_input_source_is_declared(self) -> "QuantitativeConfigSchema":
        """Refuse to infer input==target from the shape of the index."""
        if self.enabled and self.input_source is None:
            raise ValueError(
                "quantitative.enabled=true requires an explicit "
                "quantitative.input_source. Use 'contrasts' when input_paths "
                "are weighted images the maps are predicted from, or 'maps' "
                "when input_paths ARE the target maps (generative modelling "
                "of the map distribution; what build_nist_mrf_manifest.py "
                "writes). QuantitativeMapDataset enforces the declaration "
                "against the index, so a mislabelled manifest raises instead "
                "of quietly training input -> input."
            )
        return self


class PhaseContrastConfigSchema(BaseModel):
    """Phase-contrast / 4D-flow velocity encoding (regime: ``mri_flow``).

    NOTE ON THE NAME. This is phase-contrast MRI, **not** flow matching. In this
    repository "flow" overwhelmingly means rectified/normalising flow —
    ``training.flow`` is already taken by ``FlowConfig`` (an ODE sampler), and
    there are ``models/blocks/flow/``, ``flow_matching_losses.py``,
    ``physics/rectified_flow.py`` and ``field_flow_strategy.py``. This block is
    named for its physics, matching the registered ``phase_contrast`` operator
    and ``PhaseContrastVelocityLoss``, so the two can never be confused.

    When ``enabled=true`` the dataset yields a batch dict like::

        {
            "input":       Tensor[B, 4, H, W],   # 4-point encoded phases (ref,x,y,z)
            "target":      Tensor[B, 3, H, W],   # (vx, vy, vz), PHYSICAL units (m/s)
            "inlet_mask":  Tensor[B, 1, H, W],   # only if flux_conservation
            "outlet_mask": Tensor[B, 1, H, W],   # only if flux_conservation
        }

    The velocity field IS this arm's target, so it rides the canonical ``target``
    key — ``BatchAdapter.from_dict`` raises without one, so a batch that spelled
    it only ``velocity`` could not traverse the normal pipeline at all.
    ``velocity`` is accepted as an explicit alias and takes precedence. Either
    way the strategy enforces a 3-component shape, so pointing a flow arm at a
    magnitude-image dataset fails loudly instead of grading velocity against an
    image (pitfall #18).

    Velocity is emitted in m/s and is NOT rescaled to [-1, 1]: the encoding
    ``phi = pi * v / venc`` and the ``|v| > venc`` unwrap hinge are only
    meaningful in physical units (cf. ``QuantitativeConfigSchema.units``).

    Used by: ``PhaseContrastFlowStrategy``, ``models/losses/flow_losses.py``,
    ``core/metrics/flow_metrics.py``.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False the block is inert and no "
        "velocity encoding is loaded or emitted.",
    )
    venc: float = Field(
        default=1.0,
        gt=0,
        description="Velocity-encoding value (m/s). Sets phi = pi*v/venc, so "
        "|v| > venc phase-wraps. Must match the acquisition.",
    )
    encoding_scheme: Literal["four_point_reference"] = Field(
        default="four_point_reference",
        description="Velocity-encoding scheme. A closed Literal: only "
        "'four_point_reference' is implemented (flow_encoding."
        "four_point_reference_encode/decode), so 'hadamard' raises at config "
        "load rather than silently degrading (pitfall #9).",
    )
    through_plane_axis: Literal["x", "y", "z"] = Field(
        default="z",
        description="Which velocity component is through-plane, for the flux-conservation term.",
    )
    voxel_area: float = Field(
        default=1.0,
        gt=0,
        description="In-plane voxel area (mm^2) for the flux integral. It lives "
        "here rather than on the loss because ThroughPlaneFluxConservationLoss "
        "has no __init__, so the LossBuilder cannot set it — the strategy "
        "threads it through per call.",
    )
    flux_conservation: bool = Field(
        default=False,
        description="Enable the inlet/outlet flux-conservation term. Requires "
        "the dataset to emit inlet_mask and outlet_mask.",
    )

    @model_validator(mode="after")
    def validate_flux_requires_enabled(self) -> "PhaseContrastConfigSchema":
        if self.flux_conservation and not self.enabled:
            raise ValueError(
                "phase_contrast.flux_conservation=true requires "
                "phase_contrast.enabled=true — the block is otherwise inert and "
                "the flux term would silently never fire."
            )
        return self


class PerfusionConfigSchema(BaseModel):
    """DCE tracer-kinetic perfusion ingestion (regime: ``mri_perfusion``).

    When ``enabled=true`` the dataset yields a batch dict like::

        {
            "input": Tensor[B, T, H, W],   # measured concentration-time curve C_t(t)
            "t_s":   Tensor[T],            # time axis, SECONDS, uniformly sampled
            "aif":   Tensor[T],            # only if aif_source == "measured"
        }

    Concentrations are in mM and the time axis in seconds — both
    ``extended_tofts_forward`` and ``parker_population_aif`` assume it (the AIF
    converts to minutes internally). No rescale to [-1, 1].

    Used by: ``PerfusionKineticMappingStrategy``,
    ``models/losses/perfusion_losses.py``,
    ``infrastructure/physics/signal_models/perfusion_kinetics.py``.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False the block is inert.",
    )
    kinetic_model: Literal["extended_tofts"] = Field(
        default="extended_tofts",
        description="Tracer-kinetic forward model. A closed Literal with one "
        "member: only extended_tofts is wired into ToftsResidualLoss. "
        "'gamma_variate' exists in signal_models but no loss consumes it, so "
        "listing it here would advertise a knob nothing reads (pitfall #15).",
    )
    aif_source: Literal["population", "measured"] = Field(
        default="population",
        description="'population' = the Parker analytic AIF; 'measured' = the "
        "dataset's 'aif' batch key. 'learned' is deliberately ABSENT: it would "
        "need a generator.predict_aif head that no registered model implements "
        "(pitfall #15). Backlogged.",
    )
    num_frames: int = Field(
        default=60,
        ge=2,
        description="T, the number of dynamic frames. Must equal "
        "model.in_channels — the strategy checks this and raises.",
    )
    temporal_resolution_s: float = Field(
        default=5.0,
        gt=0,
        description="Seconds between dynamic frames. Builds the uniform t_s "
        "axis that extended_tofts_forward assumes.",
    )


class SpectroscopyConfigSchema(BaseModel):
    """MRS / MRSI ingestion (regime: ``mri_spectroscopy``).

    When ``enabled=true`` the dataset yields a batch dict like::

        {
            "input": Tensor[B, 2*T, H, W],  # FID, real and imag as channels
            "t_s":   Tensor[T],             # acquisition time axis, SECONDS, from 0
        }

    The FID is **complex**, carried as ``2*T`` interleaved real/imag channels so a
    standard 2-D conv generator can consume it — the same trick ``data.perfusion``
    uses for its dynamic frames. Time starts at 0: an FID is causal, and that is
    why ``fid_to_spectrum`` shifts only its output (see the signal model).

    Used by: ``MRSQuantificationStrategy``,
    ``models/losses/spectroscopy_losses.py``,
    ``infrastructure/physics/signal_models/spectroscopy.py``.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False the block is inert.",
    )
    signal_model: Literal["mrs_lorentzian"] = Field(
        default="mrs_lorentzian",
        description="Spectral forward model, resolved through the "
        "SignalModelRegistry by the strategy. A closed Literal with one member: "
        "only mrs_lorentzian is implemented. A metabolite-basis (LCModel) model "
        "would need a basis set the repo does not ship, so listing it here would "
        "advertise a knob nothing reads (pitfall #15).",
    )
    num_points: int = Field(
        default=1024,
        ge=8,
        description="T, the number of complex FID samples. model.in_channels must "
        "equal 2*T (real and imag); the strategy checks this and raises.",
    )
    dwell_time_s: float = Field(
        default=5e-4,
        gt=0,
        description="Seconds between FID samples. Sets the spectral bandwidth "
        "(1/dwell) and builds the uniform t_s axis the signal model assumes.",
    )
    num_resonances: int = Field(
        default=3,
        ge=1,
        description="M, the number of Lorentzian resonances fitted per voxel. "
        "model.out_channels must equal 4*M — amplitude, frequency, linewidth and "
        "phase per resonance.",
    )
    field_strength_t: float = Field(
        default=3.0,
        gt=0,
        description="B0 in tesla. Only used to convert chemical shift (ppm, "
        "field-independent) to frequency (Hz) for reporting.",
    )


class AcquisitionMetadataConfigSchema(BaseModel):
    """Per-sample acquisition-parameter ingestion (TE / TR / TI / FA / B0).

    When ``enabled=true``, every sample carries an ``acquisition_metadata``
    dict::

        sample["acquisition_metadata"] = {
            "TE":  float (ms),
            "TR":  float (ms),
            "TI":  float (ms),    # -1 sentinel when no IR
            "FA":  float (deg),
            "B0":  float (T),
            "contrast_type": "spin_echo" | "inversion_recovery" | "gradient_echo" | "diffusion_weighted",
        }

    Multi-echo / DWI: when the input has C > 1 contrast channels, each scalar
    becomes a length-C list (one per echo / b-value).

    Used by:

    - ``AcquisitionEmbedding`` (encodes the scalars into a learnable embedding
      consumed via FiLM/cross-attention)
    - ``BlochConsistencyLoss`` (synthesizes the contrast from (T1, T2, PD) and
      compares to the input)
    - Vendor / protocol-aware harmonization

    Distinguish from :class:`MultiContrastConfigSchema.acquisition_params`:
    that one carries fixed per-class parameters (one row per contrast type);
    this one carries actual per-sample values from the scan headers.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, no metadata is loaded or "
        "emitted (preserves pre-Phase-4b behavior; legacy AcquisitionEmbedding "
        "callers can still inject their own dict downstream).",
    )
    source: AcquisitionMetadataSource = Field(
        default="bids_json",
        description="Where to read scan parameters from. ``bids_json`` looks "
        "for a sibling ``.json`` next to each image file (BIDS spec). "
        "``dicom`` reads DICOM headers (TE / TR / FA tags). ``h5_attrs`` reads "
        "HDF5 root attributes (FastMRI-style). ``yaml_inline`` uses the "
        "``defaults`` dict below — primarily for smoke tests and synthetic data.",
    )
    fields: list[Literal["TE", "TR", "TI", "FA", "B0", "contrast_type", "b_value", "bvec"]] = Field(
        default_factory=lambda: ["TE", "TR", "FA"],
        description="Which scan parameters to read. ``TI`` only meaningful for "
        "inversion-recovery sequences; ``b_value``/``bvec`` only for DWI.",
    )
    defaults: dict[str, float | str | list[float]] = Field(
        default_factory=dict,
        description="Fallback values for fields missing from the source (e.g. "
        "{'TI': -1.0, 'B0': 3.0}). Required keys when source='yaml_inline'. "
        "list[float] for multi-echo / multi-bvalue defaults.",
    )
    strict: bool = Field(
        default=False,
        description="When True, raise if a required field is missing from the "
        "source (and not provided in ``defaults``). When False, log a warning "
        "and fall back to the default. CLAUDE.md #9 says strict=True is the "
        "right answer for production training — silent fallbacks corrupt "
        "Bloch-loss residuals.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4c — Temporal / 4D / cine
# ═══════════════════════════════════════════════════════════════════════════
#
# Cine MRI volumes are 4D: [H, W, slices, frames]. The convention this
# codebase uses is to fold the frame axis into TorchIO's channel axis so
# spatial samplers (UniformSampler, GridSampler, Resample) work unchanged.
# Tensor stored on the Subject has shape (frames, W, H, slices) — frames
# in the C slot, spatial dims after.
#
# Temporal samplers slice along the channel/frame axis to extract a
# contiguous window of frames (``temporal_uniform``) or a deterministic
# grid of windows (``temporal_grid``) for inference. The aggregator
# stitches windows back together preserving frame order — the central
# correctness constraint for cine reconstruction.


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4d — Inference output ordering contract
# ═══════════════════════════════════════════════════════════════════════════
#
# Two output shapes need explicit on-disk discipline:
#
# 1. Multi-channel quantitative outputs (Phase 4b): the model emits a
#    (B, C, H, W, D) tensor where C maps to {t1, t2, pd, ...}. Splitting
#    that into per-map NIfTI files with the right names + per-channel
#    denormalization is the writer's responsibility.
#
# 2. Frame-ordered cine outputs (Phase 4c): the aggregator returns
#    (frames, W, H, slices). The writer must respect the original frame
#    order recorded on the Subject so cine loops play correctly.
#
# Both blocks live under ``modes.infer.output.*`` — they're inference-only.


WriterFormat = Literal["nifti", "h5", "npy"]


class OutputChannelSchema(BaseModel):
    """Per-channel naming + denormalization for multi-channel inference.

    Used when the model emits a (B, C, ...) tensor and each channel
    needs to land on disk as a separately-named, separately-scaled
    artifact. Index in the parent list = channel index.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description="Filename suffix and identity (e.g. 't1', 't2', 'pd', "
        "'lr_pred', 'magnitude'). The OutputWriter uses this to name the "
        "per-channel file.",
    )
    scale: float = Field(
        default=1.0,
        description="Multiplicative factor applied at write time "
        "(post-denorm value = raw_value * scale + offset). For physical "
        "units (T1 in ms, T2 in ms, PD in [0,1]), set the scale that "
        "inverts the train-time normalization for THIS channel.",
    )
    offset: float = Field(
        default=0.0,
        description="Additive offset applied at write time (post = raw * scale + offset).",
    )
    dtype: Literal["float32", "float16", "int16", "uint16"] = Field(
        default="float32",
        description="On-disk dtype. ``int16``/``uint16`` clip after "
        "scale+offset — useful for DICOM-compatible 16-bit storage.",
    )


class MultiChannelOutputSchema(BaseModel):
    """Output discipline for multi-channel models (Phase 4b / 4d).

    When enabled, the inference writer splits the channel axis of the
    output tensor into separately-named files, applying per-channel
    scale/offset/dtype. When disabled (default), the writer treats the
    output as a single intensity volume (legacy behavior).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, legacy single-channel "
        "write path is used (preserves pre-Phase-4d behavior).",
    )
    channels: list[OutputChannelSchema] = Field(
        default_factory=list,
        description="Ordered per-channel write specs. Index = output tensor channel index.",
    )

    @model_validator(mode="after")
    def validate_channels_when_enabled(self) -> "MultiChannelOutputSchema":
        if self.enabled and not self.channels:
            raise ValueError(
                "output.multi_channel.enabled=true requires a non-empty "
                "output.multi_channel.channels list."
            )
        return self


class TemporalOutputSchema(BaseModel):
    """Output discipline for cine 4D inference (Phase 4c / 4d).

    Cine inference reassembles per-window predictions into a frame-
    ordered volume (via :class:`TemporalGridAggregator`). This block
    declares the on-disk shape — frame-axis position + filename pattern.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, frame-aware writing is "
        "skipped (preserves pre-Phase-4d behavior).",
    )
    frame_axis_out: Literal[0, 1, 2, 3] = Field(
        default=3,
        description="Which axis of the on-disk tensor holds the frame index. "
        "ACDC NIfTI 4D convention is axis 3 (last).",
    )
    preserve_frame_order: bool = Field(
        default=True,
        description="When True, the writer reads ``subject['frame_order']`` "
        "and permutes the output so frames are emitted in original index "
        "order. The aggregator already restores order — leave True unless "
        "you have a non-standard frame source.",
    )


class ModeOutputSchema(BaseModel):
    """Aggregated inference output config (under ``modes.infer.output``)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    format: WriterFormat = Field(
        default="nifti",
        description="On-disk format. ``nifti`` is the default for 3D/4D "
        "anatomical data; ``h5`` for k-space / multi-coil intermediates; "
        "``npy`` for fastest IO when the consumer is internal.",
    )
    filename_template: str = Field(
        default="{subject_id}_{name}",
        description="Filename pattern (without extension). Supported "
        "placeholders: ``{subject_id}``, ``{file_id}``, ``{name}`` "
        "(per-channel name), ``{idx}`` (sample index).",
    )
    multi_channel: MultiChannelOutputSchema = Field(
        default_factory=MultiChannelOutputSchema,
        description="Per-channel naming + denormalization. See :class:`MultiChannelOutputSchema`.",
    )
    temporal: TemporalOutputSchema = Field(
        default_factory=TemporalOutputSchema,
        description="Frame-ordered cine output. See :class:`TemporalOutputSchema`.",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4e — Multi-domain / dual-loader (domain adaptation)
# ═══════════════════════════════════════════════════════════════════════════
#
# Domain adaptation strategies (M4Raw → FastMRI, ULF → HF, vendor A → B)
# need TWO loaders. Phase 4e adds ``data.multi_domain.*`` and a director
# method that returns one loader per domain (and optionally a site-balancer
# wrapper to control batch composition).


DomainBalancingStrategy = Literal[
    "round_robin",  # Alternate batches: src, tgt, src, tgt, ...
    "stratified",  # Each batch contains samples from both domains
    "concat",  # No balancing — just zip the loaders
]


class DomainConfigSchema(BaseModel):
    """Per-domain manifest / data_root / dataset_type override.

    Used inside :class:`MultiDomainConfigSchema` to declare each domain
    independently. Unspecified fields inherit from the parent
    :class:`DataConfigSchema`.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    name: str = Field(
        ...,
        description="Domain identifier ('source', 'target', 'm4raw', "
        "'fastmri'). Used by site-balancer and strategy code to route batches.",
    )
    data_root: str | None = Field(
        default=None,
        description="Override data_root for this domain. None inherits.",
    )
    index_path: str | None = Field(
        default=None,
        description="Override manifest index path for this domain. None inherits.",
    )
    dataset_type: str | None = Field(
        default=None,
        description="Override dataset_type for this domain. None inherits.",
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Relative sampling weight for stratified batching.",
    )


class MultiDomainConfigSchema(BaseModel):
    """Configuration for two-domain training (domain adaptation)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master switch. When False, the single-loader path is "
        "used (preserves pre-Phase-4e behavior).",
    )
    domains: list[DomainConfigSchema] = Field(
        default_factory=list,
        description="Ordered list of domains. Index 0 = source by convention; "
        "index 1+ = target(s). At least 2 entries required when enabled.",
    )
    balancing: DomainBalancingStrategy = Field(
        default="round_robin",
        description="How to compose batches: round_robin / stratified / concat.",
    )
    target_supervised: bool = Field(
        default=False,
        description="When True, target loader provides labels (supervised "
        "domain adaptation). When False, target loader emits inputs only.",
    )

    @model_validator(mode="after")
    def validate_domains_when_enabled(self) -> "MultiDomainConfigSchema":
        if self.enabled:
            if len(self.domains) < 2:
                raise ValueError(
                    "multi_domain.enabled=true requires at least 2 entries "
                    f"in multi_domain.domains. Got {len(self.domains)}."
                )
            names = [d.name for d in self.domains]
            if len(set(names)) != len(names):
                raise ValueError(f"multi_domain.domains names must be unique. Got: {names}")
        return self


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4f — Latent diffusion stage-2 + coordinate emission + meta-learning
# ═══════════════════════════════════════════════════════════════════════════
#
# Three orthogonal opt-in additions, all defaulting to disabled:
#
#   data.latent_diffusion.*   — Amendment H: stage-2 latent training that
#                                lazy-encodes the input through a frozen
#                                stage-1 checkpoint (VAE / VQ-VAE).
#   data.emit_coordinates.*   — Amendment I: append a normalized coordinate
#                                grid to every batch for INR / coord-MLP
#                                style models.
#   data.meta_learning.*      — Amendment J: expose MetaLearningDataset
#                                support/query/tasks-per-epoch from config
#                                instead of constructor kwargs.


class LatentDiffusionConfigSchema(BaseModel):
    """Lazy-encode wrapper for latent diffusion stage-2 training.

    When enabled, the data layer wraps the inner loader so that every
    sample is passed through a frozen stage-1 encoder (loaded from
    ``stage1_checkpoint``) before being yielded. The model sees
    latents, not raw images.

    Optional disk cache: when ``cache_dir`` is set, the encoded latents
    are persisted on first encounter and reloaded on subsequent epochs —
    avoids redundant forward passes through the stage-1 encoder.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, the inner loader yields "
        "raw images (preserves pre-Phase-4f behavior).",
    )
    stage1_checkpoint: str | None = Field(
        default=None,
        description="Path to the frozen stage-1 VAE / VQ-VAE checkpoint. "
        "Required when enabled. The wrapper instantiates the encoder, "
        "moves it to the dataset's device, and freezes weights.",
    )
    latent_key: str = Field(
        default="latent",
        description="Batch dict key under which encoded latents are stored. "
        "The original image stays under ``input`` for downstream losses.",
    )
    cache_dir: str | None = Field(
        default=None,
        description="When set, encoded latents are cached to disk under "
        "this directory keyed by sample hash. None disables caching.",
    )

    @model_validator(mode="after")
    def validate_checkpoint_when_enabled(self) -> "LatentDiffusionConfigSchema":
        if self.enabled and self.stage1_checkpoint is None:
            raise ValueError(
                "latent_diffusion.enabled=true requires "
                "latent_diffusion.stage1_checkpoint to point at a frozen "
                "stage-1 encoder checkpoint."
            )
        return self


class CoordinateEmissionConfigSchema(BaseModel):
    """Append a normalized coordinate grid to every batch.

    Used by implicit neural representation (INR) models, coordinate-MLPs,
    and any architecture that conditions on spatial position. The
    transform adds two batch keys:

    - ``coords``       : Tensor of normalized coordinates in [-1, 1].
    - ``coord_resolution`` : Tuple recording the grid resolution.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, no coords are emitted.",
    )
    normalize: bool = Field(
        default=True,
        description="When True (default), coords are in [-1, 1] per axis. "
        "When False, coords are raw integer voxel indices.",
    )
    include_batch_dim: bool = Field(
        default=True,
        description="When True, coords have shape (B, ndim, *spatial); "
        "when False, (ndim, *spatial). True is more model-friendly.",
    )


class MetaLearningConfigSchema(BaseModel):
    """Meta-learning task-sampling config (Amendment J)."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, the wrapper is bypassed.",
    )
    support_size: int = Field(
        default=4,
        ge=1,
        description="Number of examples in the support set per task.",
    )
    query_size: int = Field(
        default=4,
        ge=1,
        description="Number of examples in the query set per task.",
    )
    tasks_per_epoch: int = Field(
        default=100,
        ge=1,
        description="Virtual epoch length (number of meta-tasks per epoch).",
    )
    task_params: dict[str, list[Any]] = Field(
        default_factory=dict,
        description="Task-sampling parameters (e.g. "
        "{'acceleration_factors': [2, 4, 8, 16]}). Consumed by the "
        "underlying MetaLearningDataset.",
    )


class TemporalConfigSchema(BaseModel):
    """4D / cine MRI temporal-axis configuration.

    Cine reconstruction tasks operate on 4D volumes shaped
    [H, W, slices, frames]. This block drives indexing, pairing,
    sampling, training, and inference for temporal data:

    - ``glob_pattern`` / ``target_suffix`` / ``target_source``: how the
      index finds cine volumes and how each one gets its target.
    - ``frames_per_window``: number of consecutive frames per training
      patch (8 of 24 typical for cardiac cine).
    - ``frame_axis``: which axis on disk contains the frame index.
      Used by the dataset to permute to TorchIO channel convention.
    - ``temporal_overlap``: frame-overlap between adjacent windows at
      inference (analogous to spatial ``patch_overlap``).

    .. warning::

       ``frames_per_window`` and ``temporal_overlap`` are validated here
       and logged by ``_create_cine_dataset``, but **no tensor is windowed
       by them yet**. Their consumer, ``data.builders.temporal_sampler``,
       has no importer in ``src/`` -- ``CineMRIDataset`` serves whole
       volumes with every frame in the channel slot. Until that sampler is
       wired into the queue, both knobs are declarative only. Two
       consequences follow, so they are stated rather than discovered:
       the epoch is one draw per volume, not per window, and a cohort
       whose volumes disagree on frame count cannot be batched (see
       ``total_frames``).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master switch. When False (default), no temporal "
        "handling is applied (preserves pre-Phase-4c behavior).",
    )
    frames_per_window: int = Field(
        default=8,
        ge=1,
        description="Number of consecutive frames per training patch. "
        "Should evenly divide the total frame count to avoid edge effects.",
    )
    frame_axis: Literal[0, 1, 2, 3] = Field(
        default=3,
        description="Axis of the on-disk 4D tensor that contains the "
        "frame index. ACDC NIfTI 4D = axis 3 (last). FastMRI cine = axis 0.",
    )
    temporal_overlap: int = Field(
        default=2,
        ge=0,
        description="Frame-overlap between adjacent temporal_grid windows "
        "at inference. Higher → smoother reconstruction loops, more compute.",
    )
    total_frames: int | None = Field(
        default=None,
        description="Optional declared total frame count. When set, the "
        "dataset validates each loaded volume against it and raises on the "
        "first disagreement. Leaving it null does NOT make a "
        "heterogeneous-frame-count cohort servable -- frames occupy the "
        "TorchIO channel slot, so volumes of differing frame count cannot be "
        "stacked and collate raises at batch_size>1. Null means 'every volume "
        "already agrees, do not spend a check'; set it to assert that.",
    )
    glob_pattern: str = Field(
        default="**/*4d.nii.gz",
        description="Glob, relative to source.root, matching cine input "
        "volumes. The default is the ACDC NIfTI layout. Widen it to reach the "
        "other formats the loader supports (.nii, .npy, .pt) or a FastMRI-cine "
        "directory -- while it was hardcoded, those were unreachable.",
    )
    target_source: Literal["sibling", "self"] | None = Field(
        default=None,
        description="Where each cine volume's training target comes from. "
        "``sibling``: a paired file on disk named by ``target_suffix`` "
        "(super-resolution / translation). ``self``: the loaded volume is the "
        "target and the input is the same tensor for a degradation transform "
        "to corrupt downstream (the reconstruction setup -- see "
        "``FieldRefDataset``). Required when ``enabled``: silence would have "
        "to be read as one of the two, and reading it as ``self`` on an arm "
        "with no degradation transform trains the identity and reports "
        "excellent PSNR (pitfall #16).",
    )
    target_suffix: str | None = Field(
        default=None,
        description="Filename suffix of the paired target, substituted for "
        "``.nii.gz`` on the input's name (e.g. ``_gt.nii.gz`` beside an input "
        "``*_lr.nii.gz``). Required by, and only meaningful for, "
        "``target_source='sibling'``.",
    )

    @model_validator(mode="after")
    def validate_target_pairing_is_declared(self) -> "TemporalConfigSchema":
        """Refuse to infer the pairing mode from the silence of another field.

        ``target_suffix`` unset could mean "self-paired" or "the author forgot".
        Deriving one from the other is the same defect class as the pre-split
        skip-set that was restated instead of derived (#733): a fact carried in
        a side channel, invisible to the log and to review.
        """
        if not self.enabled:
            return self
        if self.target_source is None:
            raise ValueError(
                "temporal.enabled=true requires an explicit "
                "temporal.target_source. Use 'sibling' (+ target_suffix) when "
                "the target is a paired file on disk, or 'self' when the "
                "loaded volume IS the target and a degradation transform "
                "builds the input. There is no default: guessing 'self' on an "
                "arm with no degradation transform silently trains the "
                "identity."
            )
        if self.target_source == "sibling" and not self.target_suffix:
            raise ValueError(
                "temporal.target_source='sibling' requires "
                "temporal.target_suffix (e.g. '_gt.nii.gz') -- it is the only "
                "thing that names the paired file."
            )
        if self.target_source == "self" and self.target_suffix:
            raise ValueError(
                "temporal.target_suffix is set but target_source='self', so no "
                "sibling is ever read. Drop one: 'sibling' to use the file, or "
                f"the suffix ({self.target_suffix!r}) to self-pair."
            )
        return self

    @model_validator(mode="after")
    def validate_overlap_below_window(self) -> "TemporalConfigSchema":
        if self.enabled and self.temporal_overlap >= self.frames_per_window:
            raise ValueError(
                "temporal.temporal_overlap must be strictly less than "
                "temporal.frames_per_window (otherwise windows don't "
                "advance). Got "
                f"temporal_overlap={self.temporal_overlap}, "
                f"frames_per_window={self.frames_per_window}."
            )
        return self


# Allowed semantic roles for a BART dimension. BART arrays are a fixed 16-element
# hypercube whose dims carry no labels, so the role of each non-singleton dim is
# declared explicitly per dataset (the loader never guesses — pitfalls #9 / #15).
_BART_DIM_ROLES = frozenset(
    {
        "readout",  # frequency-encode / radial-spiral readout samples
        "phase",  # Cartesian phase-encode 1
        "phase2",  # Cartesian phase-encode 2 (3D)
        "coil",  # receive channels
        "echo",  # multi-echo / contrast index (TE)
        "frame",  # cardiac / temporal frame (cine, real-time)
        "spoke",  # non-Cartesian readout index (radial spokes / spiral interleaves)
        "slice",  # 2D slice index
        "map",  # ESPIRiT map / sensitivity set
        "repetition",  # signal averages / repetitions
        "flip",  # flip-angle index (e.g. double-angle B1 mapping: 2 flips)
    }
)


class BartConfigSchema(BaseModel):
    """BART ``.cfl``/``.hdr`` (non-Cartesian / multi-coil) k-space ingestion — spec E2.

    Activated by ``dataset_type='bart_kspace'``. BART stores a complex array as a
    16-element hypercube whose dimensions are *unlabeled*; the same role lands in
    different dim slots across datasets (e.g. readout is dim0 in
    ``cardiac_radial_ir_flash`` but dim1 in ``multiecho_radial_b0_r2star``). The
    loader therefore refuses to infer roles — each non-singleton dim's meaning is
    declared per dataset in ``bart_dim_map`` and validated here (raise on unknown
    role / out-of-range index / duplicate slot). The resolved map is stamped into
    run provenance.

    The reader (``BartCflStrategy``) returns the raw complex array + dim vector;
    this config tells the ``BartKspaceDataset`` how to canonicalize it and, for
    non-Cartesian sampling, how to obtain the trajectory + density compensation
    (feeding the existing ``NUFFTForwardModel`` / ``RadialDCF`` — never a new
    gridding implementation).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle. When False, no BART ingestion occurs.",
    )
    bart_dim_map: dict[str, int] = Field(
        default_factory=dict,
        description="Maps a semantic role (readout/phase/coil/echo/frame/spoke/"
        "slice/...) to its BART dimension index (0-15). Declared per dataset; "
        "required when enabled.",
    )
    sampling: Literal["cartesian", "radial", "spiral", "non_cartesian"] = Field(
        default="cartesian",
        description="Acquisition trajectory family. Non-Cartesian families require "
        "a trajectory_source so the NUFFT operator can be built.",
    )
    trajectory_source: Literal["none", "golden_angle", "sibling_cfl", "ismrmrd"] = Field(
        default="none",
        description="Where the measured/derived k-space trajectory comes from. "
        "'none' is valid only for Cartesian. 'golden_angle' synthesizes a "
        "radial trajectory; 'sibling_cfl' reads a paired *_traj.cfl; 'ismrmrd' "
        "pulls the stored trajectory from the .mrd header (spec E1).",
    )
    density_compensation: Literal["none", "radial", "iterative"] = Field(
        default="none",
        description="Density-compensation weighting for non-Cartesian adjoint "
        "recon. Maps onto RadialDCF / IterativeDCF in infrastructure/physics.",
    )
    b0_from_echo: bool = Field(
        default=False,
        description="Derive a REAL B0 field map (Hz) from the multi-echo data "
        "(per-coil conjugate-product phase difference between the first two "
        "echoes) and emit it as the batch 'b0_map'. Feeds the VF real-reference "
        "field-scoring seam. Requires an 'echo' role and a delta_te.",
    )
    delta_te: float | None = Field(
        default=None,
        description="Echo spacing ΔTE in seconds — required when b0_from_echo is "
        "set, since absolute B0 in Hz = phase_diff / (2π·ΔTE). No silent guess.",
    )
    b1_from_dam: bool = Field(
        default=False,
        description="Derive a REAL B1+ transmit efficiency map from a double-angle "
        "(DAM) acquisition: arccos(|S_2alpha| / (2·|S_alpha|)) / alpha_nominal, over the two "
        "'flip' measurements. Emits batch 'b1_map'. Requires a 'flip' role + "
        "b1_nominal_flip_deg. Feeds the multi-acquisition B1 real-reference seam.",
    )
    b1_nominal_flip_deg: float | None = Field(
        default=None,
        description="Nominal flip angle alpha (degrees) of the first DAM measurement — "
        "required when b1_from_dam is set (B1 efficiency = alpha_actual / alpha_nominal).",
    )
    file_pattern: str | None = Field(
        default=None,
        description="Optional substring filter on the .cfl basename (e.g. 'b1map') "
        "so a mixed dataset directory loads only the matching arrays — the others "
        "(different dim layout) would otherwise fail canonicalization.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "BartConfigSchema":
        if not self.enabled:
            return self
        if not self.bart_dim_map:
            raise ValueError(
                "bart.enabled=true requires a non-empty bart_dim_map "
                "(e.g. {'readout': 1, 'coil': 3, 'spoke': 2}). BART dims are "
                "unlabeled — refusing to guess (pitfall #9)."
            )
        unknown = sorted(set(self.bart_dim_map) - _BART_DIM_ROLES)
        if unknown:
            raise ValueError(
                f"bart_dim_map has unknown role(s) {unknown}; allowed roles: "
                f"{sorted(_BART_DIM_ROLES)}."
            )
        for role, idx in self.bart_dim_map.items():
            if not 0 <= idx <= 15:
                raise ValueError(
                    f"bart_dim_map['{role}']={idx} is outside the BART dim range [0, 15]."
                )
        if len(set(self.bart_dim_map.values())) != len(self.bart_dim_map):
            raise ValueError(
                f"bart_dim_map assigns the same BART dim index to two roles: {self.bart_dim_map}."
            )
        if self.sampling != "cartesian" and self.trajectory_source == "none":
            raise ValueError(
                f"bart.sampling='{self.sampling}' is non-Cartesian and requires a "
                "trajectory_source (golden_angle / sibling_cfl / ismrmrd) so the "
                "NUFFT operator can be constructed."
            )
        if self.b0_from_echo:
            if "echo" not in self.bart_dim_map:
                raise ValueError(
                    "bart.b0_from_echo=true requires an 'echo' role in "
                    "bart_dim_map (the B0 map is derived from the phase "
                    "evolution between echoes)."
                )
            if not self.delta_te or self.delta_te <= 0:
                raise ValueError(
                    "bart.b0_from_echo=true requires a positive delta_te "
                    "(seconds); absolute B0 in Hz cannot be derived without ΔTE."
                )
        if self.b1_from_dam:
            if "flip" not in self.bart_dim_map:
                raise ValueError(
                    "bart.b1_from_dam=true requires a 'flip' role in bart_dim_map "
                    "(the two double-angle measurements)."
                )
            if not self.b1_nominal_flip_deg or self.b1_nominal_flip_deg <= 0:
                raise ValueError(
                    "bart.b1_from_dam=true requires a positive b1_nominal_flip_deg "
                    "(degrees); B1 efficiency = alpha_actual / alpha_nominal."
                )
        return self


class BidsPairedConfigSchema(BaseModel):
    """BIDS low-field/high-field paired-NIfTI ingestion — spec E5.

    Activated by ``dataset_type='bids_paired'``. Indexes a BIDS tree with two
    field-strength subtrees (default ``64mT_data`` / ``3T_data``) and pairs by
    (subject, contrast) for low-field → high-field domain adaptation. A contrast
    present in only one field is excluded — never paired with a fabricated
    partner (pitfall #9).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False, description="Master toggle for BIDS paired-field ingestion."
    )
    low_field_dir: str = Field(
        default="64mT_data",
        description="Name of the low-field subtree directory (the input domain).",
    )
    high_field_dir: str = Field(
        default="3T_data",
        description="Name of the high-field subtree directory (the target domain).",
    )
    contrasts: list[str] = Field(
        default_factory=lambda: ["T1w", "T2w", "FLAIR"],
        description="BIDS suffixes to pair (e.g. T1w/T2w/FLAIR). Only contrasts "
        "present in BOTH fields for a subject are kept.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "BidsPairedConfigSchema":
        if not self.enabled:
            return self
        if not self.contrasts:
            raise ValueError("bids_paired.enabled=true requires a non-empty contrasts list.")
        if self.low_field_dir == self.high_field_dir:
            raise ValueError(
                "bids_paired.low_field_dir and high_field_dir must differ "
                f"(both are {self.low_field_dir!r})."
            )
        return self


class PngPairedConfigSchema(BaseModel):
    """Paired-PNG super-resolution ingestion — spec brats_sr.

    Activated by ``dataset_type='png_paired'``. Pairs a low-res/undersampled PNG
    directory (``lr_dir``, default ``A_LRSI``) against a high-res/fully-sampled
    one (``hr_dir``, default ``A_HRSI``) by common filename; an optional
    ``lesion_dir`` carries a segmentation map. A filename present in only one
    field is excluded — never fake-paired (pitfall #9).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Master toggle for paired-PNG ingestion.")
    lr_dir: str = Field(
        default="A_LRSI",
        description="Low-res / undersampled PNG directory (the model input).",
    )
    hr_dir: str = Field(
        default="A_HRSI",
        description="High-res / fully-sampled PNG directory (the target).",
    )
    lesion_dir: str | None = Field(
        default=None,
        description="Optional lesion-/segmentation-map PNG directory; attached as "
        "a tio.LabelMap when a matching filename exists.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "PngPairedConfigSchema":
        if self.enabled and self.lr_dir == self.hr_dir:
            raise ValueError(
                f"png_paired.lr_dir and hr_dir must differ (both are {self.lr_dir!r})."
            )
        return self


class FieldRefConfigSchema(BaseModel):
    """NIfTI field-reference ingestion (kasper / traveling_heads).

    Activated by ``dataset_type='field_ref'``. Pairs each anatomy NIfTI
    (``anatomy_glob``) with a real field map (``b0_map`` and/or ``b1_map``, paths
    relative to ``data_root`` or absolute) carried on the batch for the VF
    real-reference field-scoring seam — the derivation-free counterpart to BART
    ``b0_from_echo`` / ``b1_from_dam`` (the map is already in Hz / efficiency).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False, description="Master toggle for NIfTI field-reference ingestion."
    )
    anatomy_glob: str = Field(
        default="**/*.nii",
        description="Glob (under data_root) for the anatomy NIfTI files (the VF target).",
    )
    b0_map: str | None = Field(
        default=None,
        description="Path to a real B0 map NIfTI (Hz), relative to data_root or "
        "absolute — carried as batch b0_map.",
    )
    b1_map: str | None = Field(
        default=None,
        description="Path to a real B1+ map NIfTI (efficiency) — carried as batch b1_map.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "FieldRefConfigSchema":
        if self.enabled and self.b0_map is None and self.b1_map is None:
            raise ValueError(
                "field_ref.enabled=true requires b0_map and/or b1_map (the real "
                "field the run grades against)."
            )
        return self


class FmriConfigSchema(BaseModel):
    """4-D BOLD series ingestion.

    Activated by ``dataset_type='fmri'``, which routes to
    :class:`~spectramr.data.datasets.fmri_dataset.FMRIBoldSeriesDataset` -- the
    only loader in the repo that keeps a time axis legible (frame order, frame
    count and TR ride on the Subject). ``dataset_type: nifti`` folds a 4-D
    volume's trailing axis into channels and drops those semantics, which is why
    ``axis_exposure`` annotates ``nifti`` as exposing no non-spatial axis and
    ``fmri`` as exposing ``TEMPORAL``.

    Before this route existed, no ``dataset_type`` selected a temporal loader, so
    ``mri_functional`` / ``mri_dynamic`` / ``mri_perfusion`` -- every regime whose
    ``required_axes`` is ``{TEMPORAL}`` -- could not be declared on any arm in the
    corpus without failing ``check_workflow_required_axes`` (issue #998).
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Master toggle for 4-D BOLD ingestion.")
    volume_glob: str = Field(
        default="**/*.nii*",
        description="Glob (under data_root) for the 4-D BOLD volumes.",
    )
    tr_seconds: float = Field(
        default=0.72,
        gt=0,
        description="Repetition time in seconds, carried on the Subject as ``tr`` "
        "so HRF/temporal strategies can read it. Previously reachable only as a "
        "FMRIVolumeDataset constructor default, with no way to set it from config.",
    )
    phase_encode_axis: int = Field(
        default=-2,
        description="Phase-encode axis index, carried on the Subject for the EPI "
        "distortion strategy.",
    )
    target_source: Literal["sibling"] | None = Field(
        default=None,
        description=(
            "How the arm's TARGET is obtained, and it must be DECLARED -- never "
            "inferred from another field's silence. 'sibling' pairs each volume "
            "with a companion file named by ``target_suffix`` (blip-up/blip-down "
            "opposite phase-encode polarity is the standard fMRI answer, and what "
            "topup/eddy consume). There is deliberately no 'self' option: with "
            "input == target, BeltramiEPIDistortionStrategy is minimised "
            "analytically at Delta_B0 = 0 (residual and mu_reg vanish together) "
            "and SpatiotemporalAdaptiveSFCRecon becomes the identity -- both "
            "train smoothly to a worthless answer. See "
            "TODO/inprogress/backlog_fmri_serving_path_2026_08_05.md #2."
        ),
    )
    target_suffix: str = Field(
        default="_target",
        description="Filename suffix identifying a 'sibling' target beside each "
        "input volume, e.g. sub-01_bold.nii -> sub-01_bold_target.nii.",
    )


class IsmrmrdConfigSchema(BaseModel):
    """ISMRMRD measured-trajectory k-space ingestion (kasper monitored spiral).

    Activated by ``dataset_type='ismrmrd_kspace'``. Reconstructs from raw ISMRMRD
    acquisitions using the file's OWN measured trajectory (closing the
    ``trajectory_source='ismrmrd'`` branch ``BartKspaceDataset`` raises on) via the
    existing ``NUFFTForwardModel`` / ``IterativeDCF`` — never a new gridding path.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(
        default=False,
        description="Master toggle for ISMRMRD measured-trajectory ingestion.",
    )
    im_size: list[int] | None = Field(
        default=None,
        description="Recon image size [H, W]. None ⇒ taken from the ISMRMRD XML "
        "encodedSpace matrixSize.",
    )
    density_compensation: Literal["none", "iterative", "radial"] = Field(
        default="iterative",
        description="Density-compensation for the non-Cartesian adjoint. 'iterative' "
        "(Pipe-Menon) suits arbitrary spiral/non-Cartesian trajectories.",
    )
    trajectory_scale: float = Field(
        default=6.283185307179586,  # 2*pi
        description="Multiplier applied to the stored trajectory to reach the "
        "torchkbnufft [-pi, pi] convention. 2*pi maps the common normalized "
        "[-0.5, 0.5] (1/FOV) ISMRMRD trajectory; confirm vs the dataset's units.",
    )
    emit_paired_trajectory: bool = Field(
        default=False,
        description="Emit both the measured trajectory (the primary file's) and "
        "the paired nominal trajectory (resolved via nominal_file_glob) as "
        "subject['trajectory_measured'] / subject['trajectory_nominal'], so a "
        "trajectory-recovery arm (vf_35) can supervise Δk against the measured "
        "deviation Δk_true = measured - nominal. The nominal files are excluded "
        "from the primary index by the same glob.",
    )
    nominal_file_glob: str | None = Field(
        default=None,
        description="Filename glob (e.g. '*nominal*') that BOTH excludes nominal "
        "files from the primary index AND resolves each measured file's paired "
        "nominal sibling. Required when emit_paired_trajectory=true.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "IsmrmrdConfigSchema":
        if self.enabled and self.im_size is not None and len(self.im_size) != 2:
            raise ValueError("ismrmrd.im_size must be [H, W] (two ints) or null.")
        if self.enabled and self.emit_paired_trajectory and not self.nominal_file_glob:
            raise ValueError(
                "ismrmrd.emit_paired_trajectory=true requires nominal_file_glob "
                "(e.g. '*nominal*') — no silent no-op (pitfall #15)."
            )
        return self


class OracleBssfpConfigSchema(BaseModel):
    """Phase-cycled bSSFP + analytical Hz B0 ingestion (oracle_bssfp, Plaehn 2025).

    Activated by ``dataset_type='oracle_bssfp'`` — the real-data Path A for the
    bSSFP→ΔB0 arm. Loads the EXTRACTED form: a real-interleaved phase-cycled stack
    NIfTI ``[H, W, 2N]`` (per ``stack_glob``) + a Hz B0 NIfTI (``b0_map``) carried
    as the real reference. The raw Siemens TWIX must be extracted/converted on the
    cluster first; a TWIX reader now exists in-repo (``spectramr.data.twix`` /
    ``TwixStrategy``) to decode the raw k-space, but this ingestion consumes the
    extracted NIfTI form and raises if those files are absent.
    """

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    enabled: bool = Field(default=False, description="Master toggle for oracle_bssfp ingestion.")
    stack_glob: str = Field(
        default="**/*bssfp*stack*.nii*",
        description="Glob (under data_root) for the real-interleaved [H,W,2N] "
        "phase-cycled-stack NIfTI files.",
    )
    b0_map: str | None = Field(
        default=None,
        description="Path to the analytical Hz B0 map NIfTI (relative to data_root "
        "or absolute) — carried as batch b0_map. Required when enabled.",
    )

    @model_validator(mode="after")
    def _validate(self) -> "OracleBssfpConfigSchema":
        if self.enabled and not self.b0_map:
            raise ValueError(
                "oracle_bssfp.enabled=true requires b0_map (the real off-resonance "
                "field the run grades against)."
            )
        return self


#: Every phase-9 sub-block. ``forbid`` even though the parent ``data:`` block is
#: ``ignore``: a new block has no legacy corpus to protect, and `ignore` is
#: precisely what let keys disappear silently (#550).
_DATA_SUBBLOCK = {"extra": "forbid", "frozen": True}


class DataLoaderConfigSchema(BaseModel):
    """How samples are batched and moved to the device.

    Purely mechanical: nothing here changes what the model sees, only how fast
    it arrives. ``max_prefetch`` is gone -- it was a deprecated alias folded
    into ``prefetch_factor`` by a hand-written validator, and it is now a rename
    record instead, so the fixer and the shim read one table. No corpus arm sets
    both, so nothing changes behaviour.

    ``use_async_dataloader`` is deliberately NOT here. Its own description says
    no live loader path reads it, and giving an inert knob a tidy home implies
    it works -- the same call made for ``optimization.num_steps`` in phase 8.
    """

    model_config = _DATA_SUBBLOCK

    batch_size: int = Field(default=4, gt=0)

    num_workers: int = Field(
        default=4,
        ge=0,
        description=(
            "DataLoader worker processes. Each worker is a fork of the parent, so its "
            "cost is NOT the whole parent image: measured on the M4Raw corpus on one "
            "host, a worker adds ~320 MB of *resident* memory (PSS), linearly -- the "
            "default 4 adds ~1.3 GB over num_workers=0 and 8 adds ~2.5 GB. Size a node "
            "with PSS (/proc/<pid>/smaps_rollup), which is what the cgroup charges. "
            "Summing per-worker RSS instead counts the ~1.1 GB shared torch fork image "
            "once per worker and overstates the real figure by 2.45x at 8 workers "
            "(9671 MB summed vs 3944 MB actual). The per-worker number is a reading on "
            "one corpus and host, not a constant -- re-measure before sizing on it. "
            "Note this multiplies the per-worker caches declared elsewhere, e.g. "
            "``mrixfields.max_resident_volumes``."
        ),
    )

    persistent_workers: bool = Field(
        default=False,
        description="If True, the data loader will not shutdown the worker processes after a dataset has been consumed once.",
    )

    prefetch_factor: int = Field(
        default=2,
        ge=2,
        description="Number of samples loaded in advance by each worker. 2 means there will be a total of 2 * num_workers samples prefetched across all workers.",
    )

    pin_memory: bool = True


class DataSamplingConfigSchema(BaseModel):
    """How samples are DRAWN from a volume, before anything is done to them.

    The TorchIO queue's three knobs finally sit together: ``patch_size`` is what
    a sample IS, ``samples_per_volume`` is how many are taken per subject per
    epoch, and ``queue_length`` is the RAM buffer holding them. Tuning one blind
    to the others is how a queue build comes to dominate a smoke run's wallclock.

    ``patch_size`` is also the most overloaded name in the repo -- ~134 model
    attributes (ViT/MAE/SwinIR patch embedding) share the spelling and mean
    something entirely different. Inside ``sampling:`` it is unambiguous, which
    is most of the point. Note ``ModeSamplerSchema`` carries its OWN
    ``samples_per_volume``/``queue_length`` (defaults 4/100) for the v6.1 mode
    vocabulary; these are the ``data:``-level ones (8/200) and the two are
    genuinely different fields.

    ``phase_encode_axis`` is deliberately NOT here. Nothing reads it from the
    config: ``FMRIVolumeDataset`` takes it as a constructor argument with its own
    default and is selected by no ``dataset_type`` (already noted in
    ``data/datasets/axis_exposure.py``), and ``epi_forward`` takes it as a
    function parameter. Two arms declare it and it does nothing -- the same call
    as ``use_async_dataloader``, ``data.test_split`` and
    ``data.return_image_domain``.

    ``modes`` stays put; it was already a nested block.
    """

    model_config = _DATA_SUBBLOCK

    patch_size: tuple[int, int, int] | tuple[int, int] = Field(
        default=(320, 320, 1),
        description="Input tensor shape (W, H, D). D=1 for 2D training, D>1 for 3D.",
    )
    samples_per_volume: int = Field(
        default=8,
        description="How many random patches to extract per subject per epoch.",
    )
    queue_length: int = Field(
        default=200,
        # ge=1 declared directly: rejection previously happened only as a side
        # effect of the Phase-3 mode-derivation validator building the v6.1
        # sampler schema (which carries ge=1) from this flat field.
        ge=1,
        description="TorchIO patch queue buffer size (RAM vs CPU balance).",
    )

    enable_slab_mode: bool = Field(
        default=False,
        description="Enable 2.5D slab collation: Flattens input depth to channels [B, C*D, H, W] and keeps middle target slice.",
    )
    enable_slice_2d: bool = Field(
        default=False,
        description=(
            "Per-slice 2D sampling for volumetric paired NIfTI "
            "(``dataset_type: nifti_paired``): expose each axial slice of a 3D "
            "[C,H,W,D] volume as one training sample (a depth-1 [C,H,W,1] "
            "subject; the collation depth-squeeze then yields [B,C,H,W]). This "
            "lets a 2D AutoencoderKL/VAE (``spatial_dims: 2``) consume 3D HF "
            "volumes without a conv2d shape crash, and keeps ``batch_size: 1`` "
            "gradient-light (one slice, not a whole 156-slice volume → no OOM). "
            "A D-deep volume expands to D samples. No-op for non-paired / "
            "already-2D data."
        ),
    )

    num_synthetic_samples: int = Field(
        default=10,
        ge=1,
        description="Number of synthetic samples to generate when dataset is empty (fallback for testing)",
    )

    @field_validator("patch_size")
    @classmethod
    def validate_patch_size(cls, v: tuple[int, ...]) -> tuple[int, int, int]:
        """Normalize patch_size to 3D tuple (W, H, D)."""
        if len(v) == 2:
            return (v[0], v[1], 1)
        if len(v) == 3:
            return (v[0], v[1], v[2])
        raise ValueError("patch_size must be a tuple of 2 or 3 integers")


class DataPairingConfigSchema(BaseModel):
    """What counts as the INPUT and what counts as the TARGET.

    Every field here answers one question -- given a pile of records on disk,
    which two of them form a training pair?  ``contrasts``/``sessions`` filter
    the input side, ``target_contrasts``/``target_sessions`` the target side
    (each defaulting to its input counterpart), ``bidirectional_mode`` says
    which end of a ULF/HF pair is which, ``hf_resolution`` picks among HF
    variants, ``allow_unpaired`` admits records with no partner at all, and
    ``single_contrast`` switches pairing off entirely.

    The plan called this block ``contrast:``, which is wrong twice over.  Three
    of the eight fields are ULF/HF field-strength pairing with no contrast
    content, and ``data.contrast`` would sit next to the pre-existing
    ``data.input_contrast``/``data.target_contrast`` -- which are
    ``ContrastConfigSchema`` NORMALIZATION specs (percentile, out_range, clamp),
    not selection filters.  Two adjacent blocks whose names differ by a plural
    is a worse collision than the stutter the rename was avoiding.

    Leaf names are carried across unchanged.  ``contrasts`` is NOT renamed to
    ``input_contrasts`` for the same reason: singular-vs-plural is the worst
    available disambiguator against ``data.input_contrast``, and the block
    prefix already does that work.

    The three ``[PAIRED]`` fields are read defensively today
    (``getattr(config, "bidirectional_mode", "ulf_to_hf")`` and friends) because
    they are also reached through a duck-typed stand-in -- ``IndexBuilder``
    documents the protocol as "an object exposing ``allow_unpaired``,
    ``bidirectional_mode``, ``contrasts``, and ``hf_resolution``".  Those
    stand-ins now carry a real ``DataPairingConfigSchema``, so the defaults
    encoded at each call site stop being a second, silent source of truth.
    """

    model_config = _DATA_SUBBLOCK

    contrasts: list[str] | None = Field(
        default=None,
        description="Filter specific contrasts (e.g. ['T1w', 'FLAIR']). If None, load all available.",
    )

    sessions: list[str] | None = Field(
        default=None,
        description="Filter specific sessions (e.g. ['01', '02']). If None, load all.",
    )

    target_contrasts: list[str] | None = Field(
        default=None,
        description="Filter specific target contrasts (e.g. ['T2w']). If None, defaults to `pairing.contrasts`.",
    )

    target_sessions: list[str] | None = Field(
        default=None,
        description="Filter specific target sessions. If None, defaults to `pairing.sessions`.",
    )

    single_contrast: bool = Field(
        default=False,
        description=(
            "Single-contrast mode for M4Raw: load each contrast independently "
            "(no cross-contrast pairing). Each sample = 1 contrast with repetition "
            "averaging. Use for contrast-agnostic training (e.g., universal k-space filling)."
        ),
    )

    bidirectional_mode: Literal["ulf_to_hf", "hf_to_ulf", "hf_to_hf", "ulf_to_ulf"] = Field(
        default="ulf_to_hf",
        description=(
            "Direction for paired ULF/HF datasets; name is <input>_to_<target>. "
            "'ulf_to_hf' (default): ULF=input, HF=target. "
            "'hf_to_ulf': HF=input, ULF=target (bidirectional testing). "
            "'hf_to_hf' / 'ulf_to_ulf': autoencode a single field — input≡target "
            "by construction; the OPPOSITE arm is DROPPED (target_path set None, "
            "self-supervised branch aliases target=input). Use for stage-1 VAE "
            "pretraining (a translation direction would train a degradation net)."
        ),
    )

    hf_resolution: Literal["highres", "lowres", "unknown"] | None = Field(
        default=None,
        description=(
            "Filter paired manifest records to a specific HF scan resolution. "
            "'highres': only full 3T resolution scans (acq-highres). "
            "'lowres': only downsampled-to-ULF-FOV scans (acq-lowres). "
            "'unknown': preprocessed data where resolution was merged (acq-3t). "
            "None (default): no filtering — use all available HF resolution variants."
        ),
    )

    allow_unpaired: bool = Field(
        default=False,
        description=(
            "Allow samples without a corresponding target (inference/val mode). "
            "When True, subjects with no target_path are included as inference-only samples. "
            "Should be True for validation splits of the ULF-paired dataset, where more ULF "
            "subjects exist than HF subjects."
        ),
    )


class DataSourceConfigSchema(BaseModel):
    """WHERE the bytes come from -- roots, indices and manifests.

    Six answers to one question, previously scattered across ~300 lines of
    ``data:``: the tree they live under (``root``), how that tree is arranged
    (``layout``), and the four files that enumerate what to read from it.

    ``dataset_type`` is deliberately NOT here. It is a DISPATCH key rather than
    a location -- ``dataset_instantiator``, ``config_health_checker`` and
    ``axis_exposure.DATASET_TYPE_SIGNAL_DOMAINS`` branch on it, and the plan
    binds it to ``workflow.regime``. 636 arms declare it as the first line of
    their ``data:`` block, where nothing about it is ambiguous, so nesting it
    would cost a corpus-wide fold and buy a reader nothing. ``datasets`` and
    ``manifest_roles`` stay for a different reason: both are already nested
    blocks, and this phase groups scalars rather than re-parenting blocks.

    ``data_root`` and ``data_layout`` drop their ``data_`` prefix -- the
    top-level block supplies it, as with ``expose_scanner_id`` ->
    ``expose.scanner_id``. The ``_path`` leaves keep their full names: under the
    plan's rule ``<thing>_path`` is a file, and ``index``/``validation_index``/
    ``paired_manifest`` are the things.

    ``root`` carries its ``field_validator`` across. Leaving it behind would be
    a hard ``PydanticUserError`` at import -- the one failure mode in this
    decomposition that cannot be missed.
    """

    model_config = _DATA_SUBBLOCK

    root: str = Field(
        default="./data",
        description="Root folder containing the dataset (e.g. 'databases/fastmri/knee_singlecoil')",
    )

    layout: Literal["flat", "bids"] = Field(
        default="flat",
        description="Directory layout: 'flat' (all files in root/split) or 'bids' (sub-*/ses-*/anat/*.nii.gz)",
    )

    index_path: str | None = Field(
        default=None,
        description="Path to pre-computed .pkl index for fast startup (required for H5)",
    )

    validation_index_path: str | None = Field(
        default=None,
        description="Path to pre-computed .pkl index for validation set (overrides default split)",
    )

    test_index_path: str | None = Field(
        default=None,
        description=(
            "Index/manifest of the HELD-OUT test set: the records a paper reports "
            "on, never the ones that select the checkpoint (that is validation). "
            "Must differ from index_path and validation_index_path; the "
            "`held_out_test_split_disjoint` witness checks the records too. "
            "Declaring it is a protocol change (cohort review 2026-09-02, T0.3): "
            "numbers stop being comparable with runs that reported on validation."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _held_out_test_index_is_distinct(cls, data: Any) -> Any:
        """A test manifest that is also the train or validation manifest is not held out."""
        if isinstance(data, dict):
            test = data.get("test_index_path")
            if test:
                for key in ("index_path", "validation_index_path"):
                    if data.get(key) and str(data[key]) == str(test):
                        raise ValueError(
                            f"data.source.test_index_path equals data.source.{key} "
                            f"({test!r}): a held-out test set cannot be the set the model "
                            "trains on or selects its checkpoint with."
                        )
        return data

    paired_manifest_path: str | None = Field(
        default=None,
        description=(
            "Path to the v4 paired JSON manifest generated by "
            "scripts/preprocessing/generate_paired_manifest.py. "
            "When set, bypasses the BIDS directory crawl and uses pre-computed pairs. "
            "Supports PathResolver cluster-relative paths."
        ),
    )

    preprocessing_dir: str | None = Field(
        default=None,
        description="Path to preprocessed dataset directory (for dataset_type='preprocessed')",
    )

    @field_validator("root")
    @classmethod
    def normalize_root(cls, value: str) -> str:
        """Expand and normalise the tree root; existence is not required here."""
        return str(PathNormalizer.normalize_and_validate(value, must_exist=False))


class TransformSpecSchema(BaseModel):
    """One entry of ``data.processing.transforms``.

    Typed so a declaration cannot be silently discarded. The previous type was
    ``dict[str, Any]``, and the only consumer scanned the list for the literal
    string ``"graph_encoding"`` and stopped -- every other entry validated and
    then vanished, so an arm named for a transform trained without it.

    ``name`` is REQUIRED, which alone catches the committed arm that spells the
    key ``type:`` instead. Registry membership is checked outside the schema
    (``config/`` may not import ``data/`` -- non-negotiable #5); the builder
    raises on an unregistered name and the Tier-1 audit reports it earlier.

    Kwargs may be nested under ``kwargs:`` (preferred) or written flat beside
    ``name`` -- the flat spelling is what the committed ``graph_encoding`` arms
    use, so ``extra="allow"`` keeps them working. An unknown kwarg is not
    swallowed: it reaches the transform constructor and surfaces as a TypeError.
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(
        ...,
        description=(
            "Registered transform name (see "
            "spectramr.data.transforms.registry.list_transforms). Dotted import "
            "paths are NOT supported and never resolved."
        ),
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Constructor kwargs for the transform.",
    )

    def resolved_kwargs(self) -> dict[str, Any]:
        """Merge the flat extras with the nested ``kwargs`` block.

        Nested ``kwargs`` wins on a collision, and the merge is the single
        place the two accepted spellings are reconciled.
        """
        extras = dict(self.__pydantic_extra__ or {})
        extras.update(self.kwargs)
        return extras


class DataProcessingConfigSchema(BaseModel):
    """What is done to the VALUES between disk and the model.

    Reads as the pipeline it is: k-space is normalised (and optionally log-scaled)
    first, then the image-domain normalisation and rescale run, then any
    config-driven transforms. Previously these thirteen were scattered across the
    flat block with the k-space group ~200 lines from the image group, so nothing
    showed that they compose.

    The four booleans take the ratified ``enable_<thing>`` spelling (165 uses vs
    20 ``use_*``); the nouns they gate keep their names, so
    ``enable_kspace_normalization`` sits directly above ``kspace_percentile`` and
    ``kspace_scale_domain``, which is the group a reader is actually looking for.

    ``resample`` and ``crop_or_pad`` are NOT here: they are already nested
    component blocks, and the phase-9 decomposition groups scalars rather than
    re-parenting blocks that were never flat.
    """

    model_config = _DATA_SUBBLOCK

    # --- k-space scaling ---
    enable_kspace_normalization: bool = Field(
        default=False,
        description="Normalize k-space data by the kspace_percentile magnitude.",
    )
    kspace_percentile: float = Field(
        default=0.99,
        ge=0.0,
        le=1.0,
        description="Percentile (0-1) used for k-space normalization",
    )
    kspace_scale_domain: Literal["kspace", "image"] = Field(
        default="kspace",
        description=(
            "Where KSpaceNormalizationTransform measures the robust scale. "
            "'kspace' takes the percentile of the k-space magnitude (optionally "
            "over the centre patch, see log_scaling_center_fraction). 'image' is "
            "Parseval-compliant: the percentile of the coil-RSS magnitude AFTER "
            "ifft2c, so the reconstructed IMAGE lands at ~unit scale — the right "
            "choice when losses/metrics are graded in the image domain. "
            "'image' ignores log_scaling_center_fraction (a k-space notion)."
        ),
    )
    enable_log_scaling: bool = Field(
        default=False,
        description="Enable robust center-patch dynamic scaling (previously in model)",
    )
    log_scaling_center_fraction: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Fraction of center patch used for log_scaling computation",
    )

    # --- image-domain normalisation ---
    # 'scalar' was removed 2026-08-04: it was an advertised member with no
    # dispatch branch in either build_train_transforms or build_val_transforms,
    # so declaring it applied NO normalization while the run reported success
    # (pitfall #9 / #15). 0 arms declared it. Re-add it here only together with
    # both dispatch branches in torchio_transform_builder.py.
    normalization_type: Literal["none", "standard", "minmax", "percentile", "robust_percentile"] = (
        Field(
            default="none",
            description="Image normalization strategy: 'standard' (Z-score), 'minmax' (Rescale), 'percentile' (Robust Scale), 'robust_percentile' (Alias for percentile)",
        )
    )
    normalization_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments for normalization strategy (e.g. percentile=0.99)",
    )
    enable_image_normalization: bool = Field(
        default=False,
        description="Normalize image data using ZNormalization (zero mean, unit std)",
    )
    enable_image_rescale: bool = Field(
        default=False,
        description="Rescale image intensities to specific range (default -1 to 1)",
    )
    rescale_range: tuple[float, float] = Field(
        default=(-1.0, 1.0),
        description="Target range for intensity rescaling (min, max)",
    )
    rescale_percentiles: tuple[float, float] = Field(
        default=(0.0, 100.0),
        description="Source percentiles for rescaling (min, max). Use (0.5, 99.5) to reject outliers.",
    )

    # --- downstream ---
    data_range: float | None = Field(
        default=None,
        description="Explicit data range for metrics (e.g. 2.0 for [-1,1], 1.0 for [0,1]). Overrides dynamic calculation in PSNR/SSIM.",
    )
    transforms: list[TransformSpecSchema] = Field(
        default_factory=list,
        description=(
            "Config-driven transforms, resolved through the transform registry "
            "(spectramr.data.transforms.registry). Each entry is {name, kwargs}; "
            "flat kwargs alongside `name` are also accepted. An unregistered "
            "name raises -- it used to be silently discarded."
        ),
    )


class DataDomainConfigSchema(BaseModel):
    """What representation the loader hands the model.

    Answers the question that decides what the model's first layer even sees:
    image or k-space, how many target channels, which artifact variants, and
    whether the sample is delivered as a graph. Previously these seven sat
    ~300 lines apart in the flat block.

    Leaves drop the redundant suffix -- ``domain.output`` rather than
    ``domain.output_domain`` -- following the ``expose:`` precedent, where the
    block name supplies the noun. It also disambiguates: ``output_domain``
    exists on ``losses`` too, and the two mean different things.

    ``return_image_domain`` is deliberately NOT here. It is a declared knob with
    no reader in ``src/spectramr`` (already carried in ``KNOWN_UNCONSUMED``); its
    only consumers are an offline audit script and
    ``scripts/evaluation/run_test_inference.py``, which raises before reaching
    it (#665). Twelve arms set it and it does nothing. Giving an inert knob a
    tidy home implies it works -- the call made for ``use_async_dataloader`` and
    ``data.test_split``. It stays flat until something reads it or it is deleted.
    """

    model_config = _DATA_SUBBLOCK

    output: Literal["image", "kspace"] = Field(
        default="image",
        description=(
            "Domain the loader emits: 'image' or 'kspace'. Must agree with what "
            "the model declares it consumes -- the audit's domain_alignment "
            "check grades exactly this."
        ),
    )

    target_channels: int = Field(
        default=1,
        ge=1,
        description=(
            "Channels in the target validation/reconstruction image "
            "(1 for RSS, >1 for multi-contrast)."
        ),
    )

    # --- artifact variant selection (overrides the task default) ---
    input_artifact: str | None = Field(
        default=None,
        description="Explicitly select input artifact type (overrides task default).",
    )
    target_artifact: str | None = Field(
        default=None,
        description="Explicitly select target artifact type (overrides task default).",
    )

    # --- graph delivery (non-Cartesian encoding) ---
    graph_type: str | None = Field(
        default=None,
        description="Graph type for graph-based MRI encoding.",
    )
    enable_graph_encoding: bool = Field(
        default=False,
        description="Enable graph-based encoding for non-Cartesian MRI.",
    )
    graph_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for graph construction (k-NN, radius, ...).",
    )


class DataSplitConfigSchema(BaseModel):
    """How the corpus is partitioned into train and validation.

    Reads as the question a reader actually has: *what decides which records
    train and which validate?* ``type`` answers it, and the rest are that
    choice's companions -- ``loso`` needs ``holdout_site``, ``loso_subject``
    needs exactly one of ``holdout_subject`` / ``loso_fold``, ``random`` needs
    ``validation_fraction``. Grouping them puts the strategy next to the fields
    that make it valid, which is what ``validate_split_strategy`` checks.

    ``validation_split`` becomes ``validation_fraction``: it is a fraction in
    [0, 1], not a split, and ``<thing>_fraction`` is the ratified spelling
    (12 uses vs 5 ``_ratio``).

    ``test_split`` is deliberately NOT here. No code reads it -- its only
    consumer, ``scripts/evaluation/run_test_inference.py``, raises three
    attribute reads before reaching it -- so there is no held-out test set to
    configure. Giving an inert knob a tidy home implies it works; the same call
    was made for ``use_async_dataloader`` above and ``optimization.num_steps``
    in phase 8. It stays flat and visibly odd until issue #665 decides whether
    to wire or delete it.
    """

    model_config = _DATA_SUBBLOCK

    type: Literal["auto", "directory", "manifest", "random", "loso", "loso_subject"] = Field(
        default="auto",
        description=(
            "How to partition data into train/val splits. "
            "'directory': expects train/ and val/ subdirectories in data_root. "
            "'manifest': uses index_path and validation_index_path pkl/json files. "
            "'random': random split using split.validation_fraction. "
            "'loso': leave-one-SITE-out, requires holdout_site. "
            "'loso_subject': leave-one-SUBJECT-out, requires holdout_subject or "
            "loso_fold. "
            "'auto': infers from available config (backward-compatible cascade).\n\n"
            "NOTE the two 'loso' spellings are different designs and the "
            "collision is deliberate-but-dangerous: 'loso' holds out a SITE "
            "(multi-centre generalisation) and needs a site tag the ULF cohort "
            "does not have; 'loso_subject' holds out a SUBJECT, which is what a "
            "10-subject cohort needs so no subject appears in both splits."
        ),
    )

    validation_fraction: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description=(
            "Fraction of records held out for validation under "
            "split.type='random' (and the 'auto' cascade when it lands there)."
        ),
    )

    # --- leave-one-out companions ---
    holdout_site: str | list[str] | None = Field(
        default=None,
        description=(
            "Site ID(s) to exclude from training and use EXCLUSIVELY for "
            "validation (Leave-One-Site-Out). Required by type='loso'."
        ),
    )
    train_sites: list[str] | None = Field(
        default=None,
        description=("Sites to include in training (if None, includes all except holdout_site)."),
    )
    holdout_subject: str | None = Field(
        default=None,
        description=(
            "Subject id held out for validation under type='loso_subject'. "
            "Mutually exclusive with loso_fold."
        ),
    )
    loso_fold: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Index into the SORTED list of subject ids present in the manifest, "
            "so a cohort of N subjects is covered by folds 0..N-1 without "
            "hand-writing each subject id. Deterministic: the sort is on the id "
            "string, not on manifest order, so the same fold means the same "
            "subject across runs and machines."
        ),
    )

    # --- Smoke / debug subsampling ---
    # Cap the number of records materialized per split. The TorchIO
    # ``SubjectsDataset`` builder is eager (every record is loaded +
    # coil-processed up front), so the build cost scales linearly with
    # split size. For a 10-iteration smoke test that build dominates the
    # wallclock; capping the split to a handful of subjects collapses the
    # build from minutes to seconds without weakening config/wiring
    # validation. ``None`` (default) means no cap -- production training
    # uses the full split. Honored in
    # ``spectramr.data.metadata.index_builder`` after the train/val split.
    max_train_subjects: int | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, keep at most this many records in the train split "
            "(smoke/debug subsampling). None = full split."
        ),
    )
    max_val_subjects: int | None = Field(
        default=None,
        gt=0,
        description=(
            "If set, keep at most this many records in the validation split "
            "(smoke/debug subsampling). None = full split."
        ),
    )


class DataExposeConfigSchema(BaseModel):
    """Extra per-sample metadata the loader emits alongside the tensors.

    The block name supplies the verb, so the leaves drop their ``expose_``
    prefix: ``expose.scanner_id: true`` reads as the sentence it is. Each flag
    costs a batch key, so a model that never reads one is paying for it.
    """

    model_config = _DATA_SUBBLOCK

    acquisition_params: bool = Field(
        default=False,
        description=(
            "Expose the acquisition-parameter vector "
            "(TR, TE, TI, alpha, B0) on every batch under "
            "batch['acquisition_params']. Required by "
            "training_mode='score_field_tomography'."
        ),
    )
    # --- SFC / conformal + fMRI/MRF 2026 batch-key exposure flags ---
    # Each flag opts the corresponding key into the batch dict via the
    # `SFCConformalFMRIKeysWrapper` slotted in by `DataPipelineDirector
    # ._apply_optional_wrappers` (audit_plan_novel_fmri.md closure pass).

    conformal_jacobian: bool = Field(
        default=False,
        description=(
            "Expose batch['conformal_jacobian'] for ConformalDataConsistency "
            "(SFC §2). Default identity Jacobian when the underlying "
            "dataset has no override."
        ),
    )

    cortex_flatten_grid: bool = Field(
        default=False,
        description=(
            "Expose batch['cortex_flatten_grid'] for cortical-conformal "
            "reconstruction (SFC §5, fMRI §3). When a CorticalSurfaceDataset "
            "supplies an override (GIFTI/CIFTI/NumPy) it takes precedence; "
            "otherwise an identity grid is populated."
        ),
    )

    field_strength: bool = Field(
        default=False,
        description=(
            "Expose batch['field_strength'] (main-field B0 in Tesla) for "
            "paired field-strength conditioning (ULF↔HF). Pairs with the "
            "'field_strength' conditioning source. Default: physics.field_strength."
        ),
    )

    field_strength_target: bool = Field(
        default=True,
        description=(
            "Emit batch['field_strength_target'] (Tesla, the per-sample target field) "
            "from the mrixfields paired dataset — read by MRIxFieldsPairedDataset to "
            "gate emission and required by cross-field / field-flow validation. "
            "Pairs with expose_field_strength (source field)."
        ),
    )

    glm_design_matrix: bool = Field(
        default=False,
        description=(
            "Expose batch['design_matrix'] for HRF-coupled training "
            "(fMRI §5). Default: simple square-wave stimulus regressor."
        ),
    )

    scanner_id: bool = Field(
        default=False,
        description=(
            "Expose batch['scanner_id'] for cross-vendor harmonisation "
            "(MRF §5). Default: single-scanner id 0 unless the dataset "
            "supplies a per-sample vendor string."
        ),
    )

    site_id: bool = Field(
        default=False,
        description=(
            "Expose batch['site_id'] for multi-site / federated conditioning. "
            "Pairs with the 'site_id' conditioning source and the site Balancer. "
            "Default: single-site id 0 unless the manifest supplies a site key."
        ),
    )


class MRIxFieldsDataConfigSchema(BaseModel):
    """Options specific to the MRIxFields cohort's paired ULF/HF source.

    Six fields sharing one prefix are a sub-block spelled the long way -- the
    same argument that collapsed ``compile_*`` in phase 8.
    """

    model_config = _DATA_SUBBLOCK

    max_resident_volumes: int = Field(
        default=6,
        ge=2,
        le=64,
        description=(
            "How many decoded MRIxFields volumes a DataLoader worker may hold at once "
            "under slice_mode='all_slices' — the RAM bound, ~226 MB per volume (so the "
            "default 6 is ~1.4 GB per worker, times num_workers). It is a budget, not a "
            "hit-rate knob: VolumeBlockedSliceSampler is what turns the budget into reuse "
            "by consuming co-resident containers together. The FLOOR is the widest "
            "container's volume count — multi_source touches N sources plus the target, "
            "so the corpus' four source fields need 5 resident; a budget below that makes "
            "every sample evict the volume the next one needs (the sampler raises rather "
            "than thrash). Raise it to buy larger shuffle blocks, lower it on a RAM-bound "
            "node."
        ),
    )

    output_contrast: str | None = Field(
        default=None,
        description=(
            "For mrixfields_pairing_policy='multi_contrast': which contrast the target "
            "image is (default: the first source contrast in canonical order, T1w)."
        ),
    )

    pairing_policy: Literal[
        "fixed_target",
        "all_pairs",
        "ulf_source",
        "multi_source",
        "multi_contrast",
        "multi_field",
        "prior",
    ] = Field(
        default="all_pairs",
        description=(
            "How MRIxFieldsPairedDataset forms (source,target) field pairs within a "
            "pairing_group (MICCAI MRIxFields2026): 'all_pairs' (any ordered pair; "
            "any-to-any T3), 'fixed_target' (target pinned to mrixfields_target_field; "
            "→ e.g. 7T synthesis T1), 'ulf_source' (source pinned to 0.1T; ULF "
            "enhancement T2), 'multi_source' (B-1.1 travelling-volunteer TUPLE: all "
            "source fields < target stacked as 'sources' against the shared target; "
            "needs mrixfields_target_field), 'multi_contrast' (idea 2.1 relaxometry: all "
            "contrasts at a source field stacked as the encoder input [C,H,W] against "
            "the target-contrast image at mrixfields_target_field; needs "
            "mrixfields_target_field), 'prior' (adaptive: ULF→HF pairs for a group that "
            "has them, identity pairs for unpaired singletons — trains a field-"
            "conditioned prior on a large unpaired pool while validating real recon on "
            "a small paired set), 'multi_field' (the SAME anatomy at every field in "
            "mrixfields_fields stacked as the encoder input [M,H,W], with "
            "mrixfields_heldout_fields emitted separately and never fitted; needs "
            "mrixfields_fields, and emits NO target — the dispersion autoencoder "
            "reconstructs its own input)."
        ),
    )

    fields: list[float] | None = Field(
        default=None,
        description=(
            "For mrixfields_pairing_policy='multi_field': the field strengths (Tesla) "
            "stacked as the encoder input, emitted as channels in THIS declared order "
            "so channel m is fields[m]. Read by no other policy. The order is the "
            "contract the model's `fields` buffer is checked against, so re-ordering "
            "this list re-orders the input channels."
        ),
    )

    heldout_fields: list[float] | None = Field(
        default=None,
        description=(
            "For mrixfields_pairing_policy='multi_field': field strengths (Tesla) "
            "emitted under 'heldout' and NEVER placed in the fitted stack — the "
            "held-out-field extrapolation test (fit 1.5/3/5/7 T, predict 0.1 T). Must "
            "not intersect mrixfields_fields; an overlap would fit the field the test "
            "claims to predict and the test would still produce a number. A group "
            "lacking one of these fields is skipped, exactly as for a missing stack "
            "field."
        ),
    )

    rescale_per_image: bool = Field(
        default=False,
        description=(
            "Re-normalise every SERVED MRIxFields image (slice or volume) to [0,1] by "
            "its own min-max. Default False, because this corpus is ALREADY [0,1] on "
            "disk, so the renorm does not establish a scale — it replaces a meaningful "
            "GLOBAL scale with an arbitrary per-image one.\n\n"
            "That matters because the task is field translation: the model must learn "
            "how intensity maps from one B0 to another. Rescaling the source AND the "
            "target each to exactly [0,1] erases that relationship — the source arrives "
            "pre-scaled to the target's range, so the objective degenerates toward "
            "structure matching and the field conditioning has nothing left to explain.\n\n"
            "The cost grew with slice_mode='all_slices': one slice per volume gave a "
            "roughly comparable gain across samples, but stretching each of ~364 depth "
            "slices independently amplifies a low-dynamic-range slice near the top of "
            "the head to the same [0,1] as mid-brain — a per-slice gain the network "
            "cannot invert, because nothing in the batch records what it was. Enable "
            "only for a corpus that is NOT pre-normalised; flip it cohort-wide, never "
            "per-arm, since it changes the reference the metrics grade against."
        ),
    )

    slice_mode: Literal["central", "all_slices", "volume"] = Field(
        default="all_slices",
        description=(
            "Which slice(s) MRIxFieldsPairedDataset serves per volume. 'all_slices' "
            "(default) expands each pair/tuple to EVERY foreground depth slice — the "
            "volumes are 3-D and every foreground slice is training signal, so serving "
            "one slice would discard ~363/364 of the corpus. A whole-volume-scale "
            "air-slice filter drops the empty MNI end slices, so the per-slice renorm "
            "cannot amplify air to full contrast. 'central' serves only the middle slice "
            "(a debug/smoke shortcut, NOT a training mode). 'volume' emits the whole "
            "[C,H,W,D] for spatial_dims:3 arms (NOT supported with "
            "multi_source/multi_contrast — those raise).\n\n"
            "'all_slices' is affordable because of HOW the epoch is ordered, not because "
            "slices are cheap to fetch. The slices are NOT all held in memory: a worker "
            "keeps at most mrixfields_max_resident_volumes decoded volumes (~226 MB each) "
            "and VolumeBlockedSliceSampler consumes the containers sharing those volumes "
            "together, shuffling within the block. That is tio.Queue's load-few / "
            "emit-many / shuffle-the-buffer algorithm, reimplemented as a torch Sampler "
            "because this dataset emits plain dicts rather than tio.Subject (and because "
            "a Subject would be a PAIR, so a volume shared by N pairs would be decoded N "
            "times). One decode is amortised across every slice of every container that "
            "volume takes part in: a 5-field all_pairs group is 20 containers over 5 "
            "volumes, so 5 decodes serve ~5,000 samples (~240x fewer decodes than a plain "
            "shuffle in simulation).\n\n"
            "Without that ordering the mode is a starvation trap — measured, one full "
            "364x436x364 decode is ~0.51 s and a plain shuffled loader makes nearly every "
            "sample a fresh decode against a ~45-volume / ~10 GB working set. Reading "
            "slices lazily instead does not help: indexed_gzip is NOT installed, so each "
            "dataobj[..., i] on a .nii.gz re-inflates the stream from the start (~14.7 s "
            "for ~250 slices vs ~0.51 s for one full decode). Installing indexed_gzip, or "
            "re-exporting the corpus memmappable, would make the ordering unnecessary "
            "rather than merely faster.\n\n"
            "Note: switching modes changes the population validation metrics average "
            "over, so numbers are not comparable across the flip."
        ),
    )

    target_field: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Target field strength (Tesla) when mrixfields_pairing_policy in "
            "{'fixed_target','multi_source','multi_contrast'}."
        ),
    )


class CoilsConfigSchema(BaseModel):
    """How multi-coil data is reduced before it reaches the model.

    ``coil_processing_mode`` drops its redundant prefix (the block supplies it),
    but ``num_virtual_coils`` keeps its full name: the naming rule is
    ``num_<thing>`` and the thing is virtual coils, so shortening it would
    satisfy the block at the rule's expense.
    """

    model_config = _DATA_SUBBLOCK

    processing_mode: Literal["none", "flatten", "svd", "magnitude", "rss", "rss_image"] = Field(
        default="none",
        description="How to handle multi-coil data: 'none' keeps original, 'flatten' splits real/imag per coil, 'svd' compresses to virtual coils, 'magnitude' gives real-only RSS in source domain, 'rss' returns 2-ch real/imag of the RSS-combined k-space (applies IFFT-RSS-FFT for k-space inputs), 'rss_image' returns 1-ch RSS magnitude image (applies IFFT then RSS for k-space inputs; identity for image inputs)",
    )

    num_virtual_coils: int = Field(
        default=4,
        ge=1,
        description="Number of virtual coils for SVD compression. Only used when coil_processing_mode='svd'. Determines in_channels = 2 * num_virtual_coils.",
    )

    svd_calibration_lines: int | None = Field(
        default=None,
        ge=1,
        description="Central k-space rows (along H) used to estimate the SVD coil-compression basis. None (default) uses the full FoV. Only used when coil_processing_mode='svd'. Set via physics.coil_processing.compression.calibration_lines.",
    )


class DataConfigSchema(BaseModel):
    """The Unified Data Configuration.

    Design Philosophy:
    - 1 Path (`data_root`) to rule them all.
    - 1 Type (`dataset_type`) to define structure.
    - 1 Index (`index_path`) for massive datasets.

    Dataset Type Aliases (Backward Compatibility - v5.0 → v6.0):
    ============================================================

    To maintain backward compatibility with legacy configurations, the following
    dataset_type aliases are automatically normalized to their canonical forms:

    **K-Space Formats** (all normalize to "fastmri_kspace" or "kspace"):
    - "kspace_fastmri" → "fastmri_kspace"
    - "fastmri_knee" → "fastmri_kspace"  (for single-coil knee datasets)
    - "m4raw" → "fastmri_kspace"  (legacy raw format)
    - "kspace" → "fastmri_kspace"  (shorthand)

    **3D Volume Formats** (all normalize to "nifti"):
    - "3d" → "nifti"
    - "3d_volumetric" → "nifti"
    - "volume_h5" → "nifti"  (HDF5 volume format)
    - "nifti_volume" → "nifti"

    **2D Image Formats** (all normalize to "image"):
    - "image_folder" → "image"
    - "2d" → "image"
    - "png" → "image"
    - "jpeg" → "image"

    **Paired Dataset Formats** (for contrast adaptation):
    - "paired_nifti" → "nifti_paired"  (ULF-to-HF pairing)
    - "paired_mri" → "nifti_paired"
    - "contrast_pair" → "nifti_paired"

    **DICOM Formats** (all normalize to "dicom"):
    - "dcm" → "dicom"
    - "dicom_series" → "dicom"
    - "dicom_files" → "dicom"

    Migration Path (v5.0 → v6.0):
    =============================

    If your config uses any of the aliases above, they are **automatically
    normalized** during config loading by the ``validate_dataset_type``
    field-validator on this schema (below). You can continue using old configs
    without modification.

    **Recommended**: Update configs to use canonical names for clarity:
    - Old: `dataset_type: "3d"`
    - New: `dataset_type: "nifti"`

    Example Usage:
    ==============
    v5.0 Config (still works in v6.0):
        data:
          dataset_type: "3d"
          data_root: "./databases/ulf_to_hf/"

    v6.0 Config (canonical):
        data:
          dataset_type: "nifti"
          data_root: "./databases/ulf_to_hf/"

    For more information on config migration, see:
        docs/CONFIG_MIGRATION_GUIDE.md
        the ``validate_dataset_type`` field-validator on this schema
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        extra="ignore",  # CRITICAL: Ignores old keys like 'lr_volume_dir' so configs don't crash
        frozen=True,
    )

    # ---- phase 9: grouped sub-blocks -------------------------------------
    # The flat spellings still LOAD -- `fold_renamed_keys` moves each into its
    # sub-block before validation -- but they are gone from Python. That matters
    # more here than it did for `optimization:`: this block is `extra="ignore"`,
    # so a key the fold table forgot does not raise, it VANISHES and the arm
    # trains on the default (#550). `tests/unit/config/schemas/test_renames.py`
    # pins all 86 pre-phase-9 scalars against that.
    loader: DataLoaderConfigSchema = Field(default_factory=DataLoaderConfigSchema)
    expose: DataExposeConfigSchema = Field(default_factory=DataExposeConfigSchema)
    mrixfields: MRIxFieldsDataConfigSchema = Field(default_factory=MRIxFieldsDataConfigSchema)
    coils: CoilsConfigSchema = Field(default_factory=CoilsConfigSchema)
    split: DataSplitConfigSchema = Field(default_factory=DataSplitConfigSchema)
    domain: DataDomainConfigSchema = Field(default_factory=DataDomainConfigSchema)
    processing: DataProcessingConfigSchema = Field(default_factory=DataProcessingConfigSchema)
    sampling: DataSamplingConfigSchema = Field(default_factory=DataSamplingConfigSchema)
    pairing: DataPairingConfigSchema = Field(default_factory=DataPairingConfigSchema)
    source: DataSourceConfigSchema = Field(default_factory=DataSourceConfigSchema)

    __folded_input_keys__ = folded_input_keys("data")
    __folded_input_paths__ = folded_input_paths("data")

    # Retired flat spellings, BOTH postures. `reject_renamed_keys` raises on
    # the ones already driven to zero; `fold_renamed_keys` moves the rest.
    #
    # The reject half is not optional here even while the raise set is small.
    # This block is `extra="ignore"`, so a raise-posture record with nothing
    # to refuse it is not "retired" -- it is SILENTLY DROPPED, which the
    # renames module docstring calls strictly worse than leaving the fold in
    # place: the key stops working AND stops being visible. Promoting a
    # drained record in a block without this validator converts a working
    # fold into exactly that.
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("data")))
    _fold_renamed = model_validator(mode="before")(classmethod(fold_renamed_keys("data")))

    # --- 1. The Source ---
    dataset_type: str = Field(
        default="kspace",
        description="Canonical dataset type: e.g., 'kspace', 'image', 'nifti', 'nifti_paired', 'preprocessed', 'npy_slice', 'dicom'",
    )

    known_dataset: str | None = Field(
        default=None,
        description="DEPRECATED: Use dataset_type='m4raw' instead. Kept for backward compatibility only.",
    )

    use_repetitions: bool | None = Field(
        default=None,
        description=(
            "NEX repetition averaging: group repetition files by stem and serve "
            "target=mean(reps). None (default) means the ROUTE's own default -- "
            "dataset_type 'm4raw' averages repetitions (True). No other route "
            "reads this knob, so declaring true on one advertises a NEX reference "
            "that is never built (witness `nex_reference_route`, error)."
        ),
    )

    slice_level_records: bool = Field(
        default=False,
        description=(
            "M4Raw only (#1757): index one record per (group, slice) and read "
            "f['kspace'][slice] for every repetition, instead of one record per "
            "(patient, contrast) group that loads every repetition in full to "
            "serve samples_per_volume patches of one slice. Each record is a "
            "depth-1 subject: sampling.patch_size must have depth 1 and the train "
            "sampler must draw samples_per_volume: 1 (or be type 'full'), which "
            "the audit check slice_level_records_queue_shape enforces; an epoch "
            "then visits every slice once. Under coils.processing_mode 'rss' the "
            "phase-reference coil is picked per record, i.e. per slice on this "
            "route. Off by default: no arm changes behaviour."
        ),
    )

    extra_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments for dataset factory",
    )

    # --- 1a. The Datasets (Multi-Source) ---
    datasets: list[DatasetSourceSchema] = Field(
        default_factory=list,
        description="List of dataset sources for multi-dataset loading or explicit path override",
    )

    # --- 2. The Geometry (Slicing & Patching) ---

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_sizes(cls, data: Any) -> Any:
        """Migrate legacy keys and inject M4Raw preset defaults.

        Handles:
        1. Legacy img_size/target_size → patch_size migration.
        2. known_dataset='m4raw_multicoil' → dataset_type='m4raw' migration (G2).
        3. M4Raw preset defaults injection when dataset_type='m4raw' (S5).
        """
        if not isinstance(data, dict):
            return data

        # Map legacy keys to patch_size.
        #
        # The guard must consider the CANONICAL destination, not just the
        # legacy spelling. `data.patch_size` folds to `data.sampling.patch_size`
        # (RENAMES, posture `fold`), so once the key drain rewrites an arm the
        # legacy key is gone and a bare `"patch_size" not in data` fires --
        # injecting a scalar `image_size` beside the arm's declared canonical
        # list, which `_reject_disagreeing_spellings` then refuses. The arm
        # loaded fine before the drain and does not load after it, from a
        # rewrite that is supposed to be a no-op.
        #
        # 97 in-progress arms across 15 cohorts declare both a legacy size key
        # and `patch_size`, so this is the whole drain rather than an edge case.
        # It is also the same failure the M4Raw preset block below was already
        # repaired for -- see its comment on seeding the canonical path.
        sampling_block = data.get("sampling")
        patch_size_declared = "patch_size" in data or (
            isinstance(sampling_block, dict) and "patch_size" in sampling_block
        )
        for legacy_key in ("img_size", "target_size", "image_size"):
            if legacy_key in data and not patch_size_declared:
                data["patch_size"] = data.pop(legacy_key)
                patch_size_declared = True

        # `max_prefetch` -> `loader.prefetch_factor` is a rename record now, so
        # the shim, the fixer and the corpus gate read ONE table. The
        # hand-written fold that used to sit here was a second mechanism that
        # agreed with the table only by coincidence.

        # G2/S2: Migrate known_dataset to dataset_type
        if data.get("known_dataset") == "m4raw_multicoil" and data.get("dataset_type") not in (
            "m4raw",
        ):
            import logging

            logging.getLogger(__name__).warning(
                "[DataConfigSchema] 'known_dataset: m4raw_multicoil' is deprecated. "
                "Use 'dataset_type: m4raw' instead. Auto-migrating."
            )
            data["dataset_type"] = "m4raw"

        # S5: M4Raw preset defaults — inject only if not explicitly set by user.
        # Seeded at the CANONICAL path via `default_knob`: three of these four
        # are folded names, and writing the legacy spelling beside a migrated
        # arm's canonical one made the arm unloadable ("two spellings disagree").
        if data.get("dataset_type") in ("m4raw", "m4raw_multicoil"):
            m4raw_defaults = {
                "use_repetitions": True,
                "coil_processing_mode": "svd",
                "normalize_kspace": True,
                "kspace_percentile": 0.99,
            }
            for key, default_val in m4raw_defaults.items():
                default_knob(data, "data", key, default_val)

        return data

    # [ARCHITECT ADD] 2.5D Slab Collation Mode
    multislice_enabled: bool = Field(
        default=False,
        description=(
            "Treat a [B, C, H, W] batch with C > 2 and C even as stacked "
            "multi-slice complex k-space, rather than inferring it from the "
            "shape. Read by the reconstruction strategy and its mixin. "
            "Previously undeclared: both read it through "
            "`hasattr(config.data, 'multislice_enabled')`, which this block's "
            "extra='ignore' made permanently False, so the multi-slice branch "
            "could never execute and the flag the docstrings advertise could "
            "never fire (pitfall #16). Default False keeps every existing arm "
            "byte-identical."
        ),
    )

    # --- Collation Strategy Configuration ---
    # [PHASE 1] User-configurable collation strategy for batch assembly
    collation: CollationConfigSchema = Field(
        default_factory=CollationConfigSchema,
        description=(
            "Collation strategy configuration for batch assembly. "
            "Controls how individual samples are collated into batches during data loading. "
            "See CollationConfigSchema for available strategies and parameters."
        ),
    )

    # v6.0: No auto-migration - patch_size must be explicit

    # --- 3. The Physics (Degradation & Simulation) ---
    # [SSOT] Acceleration and center_fraction REMOVED from DataConfig.
    # Use the top-level 'acceleration' config block as the Single Source of Truth.

    # [PHYSICS] Trajectory for Non-Cartesian Acquisition
    trajectory: str | None = Field(
        default=None,
        description="Non-Cartesian trajectory type (e.g. spiral, radial) for simulated acquisition",
    )

    # --- 4. The Logistics ---
    test_split: float = Field(default=0.1, ge=0, le=1)
    #: Opt-in switch for the held-out test split. ``test_split`` alone does NOT
    #: carve one out.
    #:
    #: This is deliberate and load-bearing. ``test_split`` was declared and read
    #: by nothing (#665), and it defaults to 0.1, so **800 committed arms carry a
    #: non-zero value they never asked to act on** (657 at 0.1, 143 at 0.15;
    #: exactly 2 set it to 0). Honouring the declared value would retroactively
    #: remove 10-15% of the training data from every one of them and change every
    #: published number -- a silent, repo-wide behaviour change dressed up as
    #: "wiring an existing knob".
    #:
    #: So the fraction stays as declared and the ACTION becomes explicit: an arm
    #: gets a test split when it says so. Flipping this default is a one-line
    #: change if the corpus is ever migrated deliberately.
    enable_test_split: bool = Field(
        default=False,
        description=(
            "Hold out a test split of `test_split` fraction. Default False: "
            "`test_split` has never been read, and 800 arms carry a non-zero "
            "value by default rather than by intent (issue #665)."
        ),
    )

    # --- Smoke / debug subsampling ---
    # Cap the number of records materialized per split. The TorchIO
    # ``SubjectsDataset`` builder is eager (every record is loaded +
    # coil-processed up front), so the build cost scales linearly with
    # split size. For a 10-iteration smoke test that build dominates the
    # wallclock; capping the split to a handful of subjects collapses the
    # build from minutes to seconds without weakening config/wiring
    # validation. ``None`` (default) means no cap — production training
    # uses the full split. Honored in
    # ``spectramr.data.metadata.index_builder`` after the train/val split.

    # --- audit_plan_novel.md Idea 2 ---
    # Score-field tomography needs the acquisition-parameter vector
    # (TR, TE, TI, alpha, B0) on every batch. Datasets that can read this
    # from DICOM / JSON metadata expose it under
    # ``batch["acquisition_params"]`` when this flag is true.
    # Disabled by default for backward compatibility; the Tier-1 audit
    # check ``acquisition_params_required_for_score_field`` enforces
    # it when ``training_mode == "score_field_tomography"``.
    image_undersampling: bool = Field(
        default=False,
        description=(
            "Image→k-space bridge: synthesise an aliased 'input' from a fully-"
            "sampled magnitude image (fft2c → Cartesian mask → ifft2c, physics "
            "SSOT) so an image-domain corpus (e.g. MRIxFields NIfTI) can drive a "
            "k-space-acceleration recon arm (exp_c1 federated VF). The mask uses "
            "the top-level acceleration/center_fraction. Off by default; k-space "
            "datasets that already carry undersampled k-space must NOT enable it."
        ),
    )
    phase_encode_axis: Literal[-2, -1] | None = Field(
        default=None,
        description=(
            "Phase-encode direction for EPI distortion correction "
            "(fMRI §2). The audit "
            "``epi_phase_encode_direction_required`` enforces it when "
            "training_mode='beltrami_epi_distortion'."
        ),
    )

    # --- 4a. Split Strategy ---

    # Synthetic data fallback (for empty datasets in smoke tests)

    # PDE-benchmark dataset selector (active only when
    # dataset_type='pde_synthetic'). Picks the operator-learning
    # problem generated on-the-fly by spectramr.data.datasets.pde_synthetic.
    # See docs/mno_family.rst for the available problems.
    pde_problem: str | None = Field(
        default=None,
        description=(
            "PDE benchmark when dataset_type='pde_synthetic'. "
            "Allowed values: 'burgers_1d', 'darcy_2d'. "
            "Defaults to 'burgers_1d' when unset."
        ),
    )

    # --- 5. Components ---
    augmentation: AugmentationConfigSchema = Field(default_factory=AugmentationConfigSchema)
    caching: CachingPolicy = Field(default_factory=CachingPolicy)
    prior_loading: PriorLoadingConfigSchema = Field(default_factory=PriorLoadingConfigSchema)

    # --- 6. Advanced / Optional ---
    use_async_dataloader: bool = Field(
        default=False,
        description=(
            "DEPRECATED / inert: no live loader path reads this (only the "
            "retired ConsolidatedDatasetFactory did). Kept so existing YAMLs "
            "still load; remove it from configs. Async prefetch is governed by "
            "``persistent_workers`` + ``prefetch_factor``."
        ),
    )
    return_image_domain: bool = Field(
        default=False,
        description="Return image domain target for perceptual losses (SSIM/LPIPS)",
    )
    target_mode: Literal["complex_mean", "phase_aligned_mean", "rep_pair", "r2r"] = Field(
        default="complex_mean",
        description=(
            "M4Raw NEX target. 'r2r' is Recorrupted-to-Recorrupted (Pang 2021): "
            "BOTH halves are built from the input repetition alone as "
            "input = y + alpha*z, target = y - z/alpha with z ~ CN(0, Sigma_n) "
            "drawn fresh per sample, so Cov(input, target) = Sigma_n - Sigma_z = 0 "
            "and the L1/L2 minimiser is the clean image. It is the only mode that "
            "needs ONE excitation, so it is the only one deployable on a "
            "single-repetition scan. R2R is a second-moment construction: a "
            "per-acquisition artefact appears in both halves with the same sign "
            "and is PRESERVED, not removed -- use 'rep_pair' to decorrelate that. "
            "'complex_mean' plain-averages complex k-space "
            "(legacy; cancels signal under inter-rep phase drift). "
            "'phase_aligned_mean' corrects each rep's global phase to rep0 first. "
            "'rep_pair' is the Noise2Noise target: ONE other repetition, phase-"
            "aligned to the input repetition, never averaged, so the target's "
            "noise is independent of the input's and the SNR is that of a single "
            "repetition; it requires nex_target_exclude_input: true and >= 2 "
            "repetitions per group (a 1-repetition group has no pair and raises "
            "whatever nex_fallback says)."
        ),
    )
    val_target_mode: Literal["complex_mean", "phase_aligned_mean", "rep_pair"] | None = Field(
        default=None,
        description=(
            "M4Raw NEX target for the VALIDATION split only. None (default) "
            "serves data.target_mode on both splits. 'r2r' is excluded here at "
            "the TYPE level, and that exclusion is the reason the knob exists: "
            "r2r draws a fresh z on every __getitem__, so an r2r validation "
            "target is a different random image each epoch. PSNR against it "
            "measures the draw rather than the model, and best-checkpoint "
            "selection (track_best_metric) would keep whichever epoch drew the "
            "kindest noise. Declaring the SAME val_target_mode across a cohort "
            "is also what makes its arms comparable: the validation reference "
            "is then ONE fixed definition while the training target varies per "
            "arm, so a PSNR difference is attributable to the training target. "
            "Read by dataset_type 'm4raw' only."
        ),
    )
    r2r_alpha: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Recorruption strength for data.target_mode='r2r'. Read ONLY under "
            "that mode. alpha CANCELS from the decorrelation identity "
            "(Cov(y + alpha*z, y - z/alpha) = Sigma_n - Sigma_z for every "
            "alpha > 0), so it does not trade bias -- it trades VARIANCE between "
            "the two halves: the input carries (1 + alpha^2)*Sigma_n and the "
            "target (1 + 1/alpha^2)*Sigma_n. alpha=1.0 (default) splits it evenly "
            "at 2*Sigma_n each; alpha<1 gives a cleaner input and a noisier "
            "target, alpha>1 the reverse."
        ),
    )
    nex_target_exclude_input: bool = Field(
        default=False,
        description=(
            "M4Raw leave-one-out NEX target: drop the input rep (rep 0) from the "
            "averaged target so target and input noise are uncorrelated (removes "
            "the bias toward preserving 1/N of the input noise). Trades one rep of "
            "sqrt(N) SNR for an unbiased target; only engages with >=3 reps. "
            "Default False keeps the all-reps average. NOTE: enabling this changes "
            "the reference the PSNR/SSIM metrics grade against, so runs are not "
            "numerically comparable to all-reps-target runs — flip cohort-wide."
        ),
    )
    nex_fallback: Literal["error", "all_reps"] = Field(
        default="error",
        description=(
            "Read only under nex_target_exclude_input=true: what a group with "
            "fewer than 3 repetitions does (M4Raw FLAIR ships 2, T1/T2 ship 3). "
            "'error' (default) refuses at dataset construction, so a leave-one-out "
            "run can never silently grade some contrasts against the correlated "
            "all-reps reference (#695). 'all_reps' accepts that reference for the "
            "short groups as a DECLARED, logged choice; the run's metrics then "
            "average over two reference definitions and the paper must say so."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _nex_fallback_is_read_only_under_leave_one_out(cls, data: Any) -> Any:
        """``nex_fallback`` is consumed only by the leave-one-out branch.

        With the all-reps reference the knob is never read, and an unread knob
        must raise rather than sit in the YAML advertising a choice that was
        never made (CLAUDE.md non-negotiable 8).
        """
        if isinstance(data, dict):
            mode = data.get("target_mode")
            if mode != "r2r" and "r2r_alpha" in data:
                raise ValueError(
                    "data.r2r_alpha is read only under data.target_mode='r2r' (it "
                    "scales the recorruption that mode draws); declaring it with "
                    f"target_mode={mode!r} advertises a choice that is never made "
                    "(CLAUDE.md non-negotiable 8). Drop it, or select 'r2r'."
                )
            if mode == "r2r" and data.get("val_target_mode") is None:
                raise ValueError(
                    "data.target_mode='r2r' draws a FRESH recorruption z on every "
                    "__getitem__, so an r2r validation target is a different "
                    "random image each epoch: PSNR against it measures the draw, "
                    "not the model, and best-checkpoint selection would keep the "
                    "luckiest epoch rather than the best one. Declare "
                    "data.val_target_mode (e.g. 'phase_aligned_mean') to give the "
                    "validation split a fixed reference; r2r stays on the "
                    "training split, which is the only split that needs it."
                )
            if data.get("target_mode") == "rep_pair" and not data.get(
                "nex_target_exclude_input", False
            ):
                raise ValueError(
                    "data.target_mode='rep_pair' is the leave-one-out pair by definition "
                    "(the target is another repetition than the input); declare "
                    "data.nex_target_exclude_input: true with it."
                )
            fallback = data.get("nex_fallback", "error")
            if fallback != "error" and not data.get("nex_target_exclude_input", False):
                raise ValueError(
                    "data.nex_fallback is read only when data.nex_target_exclude_input "
                    "is true (it decides what a <3-repetition group does under "
                    f"leave-one-out); nex_fallback={fallback!r} with the all-reps "
                    "reference is an unread knob. Drop it, or enable leave-one-out."
                )
        return data

    @model_validator(mode="after")
    def _r2r_requires_a_linear_path_to_the_loss(self) -> "DataConfigSchema":
        """R2R decorrelation survives linear maps only.

        ``Cov(A a, A b) = A Cov(a, b) A^H``, so the identity
        ``Cov(input, target) = 0`` that makes the L1/L2 minimiser the CLEAN image
        is preserved by every linear stage between injection and loss -- the
        percentile divide (one scalar per subject, applied to both halves),
        ``ifft2c`` (unitary), channel interleaving. It is destroyed by a
        nonlinearity, and destroyed SILENTLY: training converges, the loss falls,
        and the network has learned the wrong fixed point.

        Two stages in this pipeline are nonlinear, so under ``r2r`` both raise
        rather than degrade (CLAUDE.md non-negotiable 3):

        * ``coils.processing_mode`` other than ``none``. ``rss`` computes
          ``|I|_rss * exp(j*angle(I_ref))`` -- phase-preserving, but the MODULUS
          is nonlinear. It maps Gaussian to Rician, so ``E[|target|] != |x|``.
          ``magnitude``/``rss_image`` are worse, ``svd`` fits its basis to the
          data so it is not a fixed linear map either.
        * ``processing.enable_log_scaling``. ``log1p`` on magnitude is nonlinear.
        """
        if self.target_mode != "r2r":
            return self
        if self.coils.processing_mode != "none":
            raise ValueError(
                "data.target_mode='r2r' requires data.coils.processing_mode='none', "
                f"got {self.coils.processing_mode!r}. R2R decorrelates the two "
                "halves in the second moment, which survives LINEAR maps only; a "
                "coil combine that takes a magnitude turns the Gaussian residual "
                "Rician, so E[target] is no longer the clean image and the trained "
                "fixed point is wrong. Nothing downstream can detect this -- the "
                "loss still falls. Serve the coils uncombined (4 coils -> 8 "
                "interleaved real channels for M4Raw)."
            )
        if self.processing.enable_log_scaling:
            raise ValueError(
                "data.target_mode='r2r' is incompatible with "
                "data.processing.enable_log_scaling=True: log1p on magnitude is "
                "nonlinear, so it breaks the R2R decorrelation the same way a "
                "magnitude coil combine does. The percentile divide alone is safe "
                "(one scalar per subject, applied to both halves, and a common "
                "scalar divides out of the covariance)."
            )
        return self

    @model_validator(mode="after")
    def _slice_level_records_is_read_by_m4raw_only(self) -> "DataConfigSchema":
        """``slice_level_records`` is consumed only by the m4raw route.

        On any other ``dataset_type`` the knob would sit in the YAML advertising
        a per-slice index that is never built (CLAUDE.md non-negotiable 8).
        ``mode="after"`` so an alias spelling has already been canonicalised by
        ``validate_dataset_type`` before the comparison.
        """
        if self.slice_level_records and self.dataset_type != "m4raw":
            raise ValueError(
                "data.slice_level_records is read only by dataset_type 'm4raw' (the "
                f"M4Raw repetition loader); dataset_type={self.dataset_type!r} never "
                "builds a per-slice index, so the knob would be unread. Drop it, or "
                "set dataset_type: m4raw."
            )
        return self

    # [ARCHITECT ADD] Explicit Data Range (for Metrics)

    # [ARCHITECT ADD] Image Normalization Strategy

    # [MULTI-COIL] Coil Processing
    # [CONTRAST-AWARE] Per-contrast normalization configuration
    input_contrast: ContrastConfigSchema | None = Field(
        default=None,
        description="Normalization configuration for input contrast (used in contrast_aware_paired dataset)",
    )
    target_contrast: ContrastConfigSchema | None = Field(
        default=None,
        description="Normalization configuration for target contrast (used in contrast_aware_paired dataset)",
    )

    # [PATTERN C] Per-sample contrast-id conditioning (FiLM)
    multi_contrast: MultiContrastConfigSchema = Field(
        default_factory=lambda: MultiContrastConfigSchema(),
        description=(
            "Opt-in: enable per-sample contrast-id conditioning so a single "
            "model can handle T1/T2/FLAIR/PD by reading a `contrast_idx` "
            "tensor on every batch. Requires every model the batch reaches "
            "-- `model.model_type` and, when declared, "
            "`model.discriminator_component.name` -- to declare "
            "`supports_contrast_conditioning=True`; the Tier-1 audit check "
            "`multi_contrast_model_support` enforces this. Datasets that "
            "expose contrast metadata (M4Raw, slice, contrast_aware_paired) "
            "already inject `contrast_idx` into the batch — this flag is the "
            "user-visible toggle that says 'I expect the model to consume it'."
        ),
    )

    # [PAIRED/2D] Per-slice sampling for volumetric paired NIfTI

    slice_cache_size: int = Field(
        default=2,
        ge=1,
        description=(
            "LRU size (number of post-transform volumes) held by "
            "``SliceVolumeDataset`` when ``slice_2d`` is set. Each cache MISS "
            "re-decodes AND re-transforms a whole [C,H,W,D] volume, so with the "
            "default 2 and a shuffled loader most window accesses miss and pay "
            "the full-volume cost. When the corpus is few volumes with many "
            "slices each (e.g. the stage-1 ULF→HF LDM VAEs) raise this to >= the "
            "volume count to hold every volume resident and eliminate "
            "re-decodes. Trades worker RAM for throughput; no effect on "
            "numerics. Per DataLoader worker."
        ),
    )

    # Options: "none" (pass through), "flatten" (2*Coils channels), "svd" (SVD compression), "magnitude" (magnitude RSS), "rss" (complex/real RSS), "rss_image" (1-ch RSS magnitude image)

    # [ARCHITECT ADD] Intensity Rescaling (for Tanh models -1 to 1)

    # [ARCHITECT ADD] Dynamic transforms list

    # [ARCHITECT ADD] Advanced Graph & Encoding

    # [ARCHITECT ADD] Manifest role configuration for input/target assignment
    manifest_roles: ManifestRoleConfigSchema = Field(
        default_factory=ManifestRoleConfigSchema,
        description="Configuration for which manifests serve as inputs vs targets (for SSL multi-input)",
    )

    # ── Mode-aware overrides (Phase 3 of TODO/audit/data_layer_unification_plan.md) ──
    # When absent, ``derive_modes_from_legacy`` populates each mode from the
    # legacy top-level fields. New YAMLs can override per-mode behavior
    # (e.g. ``data.modes.val.sampler.type: uniform`` to enable patch val).
    modes: DataModesSchema = Field(
        default_factory=DataModesSchema,
        description=(
            "Per-mode sampler / augmentation overrides. When omitted, "
            "behavior matches legacy fields (train: uniform patches + aug; "
            "val: full volumes, no aug; infer/eval: full volumes, no aug). "
            "Builders consume via ``DataConfigSchema.resolve_mode('train')`` etc."
        ),
    )

    # ── Resampling configurability (Phase 4) ────────────────────────────────
    # Both opt-in. When ``enabled=false`` (default), the hardcoded
    # ``_ResampleToReferenceTransform`` and ``_EnsureMinimumSpatialSize``
    # paths are used — preserves pre-Phase-4 behavior for ~200 existing YAMLs.
    resample: ResampleConfigSchema = Field(
        default_factory=ResampleConfigSchema,
        description="Spatial resampling for variable-shape MRI data. See "
        "ResampleConfigSchema docstring for the three strategies.",
    )
    crop_or_pad: CropOrPadConfigSchema = Field(
        default_factory=CropOrPadConfigSchema,
        description="Crop-or-pad to a canonical voxel grid. Complements "
        "``resample`` (which changes spacing; crop_or_pad changes count).",
    )

    # ── Quantitative / Bloch (Phase 4b) ─────────────────────────────────────
    # Both opt-in. When ``enabled=false`` (default) both blocks are no-ops and
    # the dataset path is unchanged. See QuantitativeConfigSchema and
    # AcquisitionMetadataConfigSchema docstrings for the emitted batch shape.
    quantitative: QuantitativeConfigSchema = Field(
        default_factory=QuantitativeConfigSchema,
        description="qMRI parameter-map ingestion (T1/T2/PD/...). When "
        "enabled, the dataset emits each map as a separate batch dict key.",
    )
    phase_contrast: PhaseContrastConfigSchema = Field(
        default_factory=PhaseContrastConfigSchema,
        description="Phase-contrast / 4D-flow velocity encoding (venc, scheme, "
        "flux masks). NOT flow matching — see the schema docstring.",
    )
    perfusion: PerfusionConfigSchema = Field(
        default_factory=PerfusionConfigSchema,
        description="DCE tracer-kinetic perfusion ingestion (time axis, AIF "
        "source, kinetic model).",
    )
    spectroscopy: SpectroscopyConfigSchema = Field(
        default_factory=SpectroscopyConfigSchema,
        description="MRS / MRSI ingestion (FID length, dwell time, resonance "
        "count, spectral signal model).",
    )
    acquisition_metadata: AcquisitionMetadataConfigSchema = Field(
        default_factory=AcquisitionMetadataConfigSchema,
        description="Per-sample acquisition-scalar ingestion (TE/TR/TI/FA/B0). "
        "When enabled, every sample carries an ``acquisition_metadata`` dict "
        "consumed by AcquisitionEmbedding and BlochConsistencyLoss.",
    )

    # ── Temporal / 4D / cine (Phase 4c) ─────────────────────────────────────
    temporal: TemporalConfigSchema = Field(
        default_factory=TemporalConfigSchema,
        description="4D cine temporal-axis configuration. When enabled, the "
        "dataset folds frames into TorchIO's channel axis and the sampler "
        "dispatches to temporal_uniform / temporal_grid.",
    )

    # ── Multi-domain / dual-loader (Phase 4e) ───────────────────────────────
    multi_domain: MultiDomainConfigSchema = Field(
        default_factory=MultiDomainConfigSchema,
        description="Two-domain training (domain adaptation). When enabled, "
        "DataPipelineDirector.build_multi_domain_dataloaders returns one "
        "loader per declared domain.",
    )

    # ── BART / non-Cartesian k-space ingestion (spec E2) ────────────────────
    bart: BartConfigSchema = Field(
        default_factory=BartConfigSchema,
        description="BART .cfl/.hdr non-Cartesian / multi-coil k-space ingestion. "
        "When dataset_type='bart_kspace', bart.bart_dim_map declares each BART "
        "dimension's role and the dataset feeds the existing NUFFT/DCF physics.",
    )

    # ── BIDS low-field/high-field paired ingestion (spec E5) ────────────────
    bids_paired: BidsPairedConfigSchema = Field(
        default_factory=BidsPairedConfigSchema,
        description="BIDS paired-field ingestion (e.g. ulf_paired 64mT↔3T). When "
        "dataset_type='bids_paired', pairs by (subject, contrast) for low→high "
        "domain adaptation.",
    )

    # ── Paired-PNG super-resolution ingestion (brats_sr) ────────────────────
    png_paired: PngPairedConfigSchema = Field(
        default_factory=PngPairedConfigSchema,
        description="Paired-PNG super-resolution (e.g. brats_sr A_LRSI↔A_HRSI). "
        "When dataset_type='png_paired', pairs LR↔HR PNGs by common filename.",
    )

    # ── NIfTI field-reference ingestion (kasper / traveling_heads) ──────────
    field_ref: FieldRefConfigSchema = Field(
        default_factory=FieldRefConfigSchema,
        description="NIfTI field-reference (real B0/B1 map). When "
        "dataset_type='field_ref', pairs anatomy with a real b0_map/b1_map for "
        "the VF real-reference field-scoring seam.",
    )

    # ── Per-arm acquisition-axis declaration ────────────────────────────────
    acquisition_axes: list[str] | None = Field(
        default=None,
        description=(
            "Non-spatial axes THIS arm's acquisition carries, e.g. ['echo']. "
            "The generalisation of data.bart.bart_dim_map's per-arm declaration "
            "to arms that are not BART: a dataset_type-keyed table states a fact "
            "about a whole corpus and cannot express a per-arm one. Declared "
            "axes outrank the per-type annotation. Validated against the Axis "
            "enum, so a typo raises rather than silently exposing nothing. "
            "Leave unset to fall back to the dataset_type annotation; an empty "
            "list is the positive claim that the arm carries NO non-spatial "
            "axis, which rejects a regime requiring one."
        ),
    )

    # ── 4-D BOLD ingestion (the temporal route) ─────────────────────────────
    fmri: FmriConfigSchema = Field(
        default_factory=FmriConfigSchema,
        description="4-D BOLD series. When dataset_type='fmri', reads volumes "
        "with the time axis kept legible (frame order / count / TR on the "
        "Subject) -- the route that makes a TEMPORAL axis claim true.",
    )

    # ── ISMRMRD measured-trajectory k-space (kasper monitored spiral) ───────
    ismrmrd: IsmrmrdConfigSchema = Field(
        default_factory=IsmrmrdConfigSchema,
        description="ISMRMRD measured-trajectory ingestion. When "
        "dataset_type='ismrmrd_kspace', reconstructs from the file's own measured "
        "trajectory via the existing NUFFT/DCF physics.",
    )

    # ── oracle_bssfp phase-cycled stack + real Hz B0 (vf_29 Path A) ──────────
    oracle_bssfp: OracleBssfpConfigSchema = Field(
        default_factory=OracleBssfpConfigSchema,
        description="Phase-cycled bSSFP + analytical Hz B0 ingestion. When "
        "dataset_type='oracle_bssfp', loads the extracted real-interleaved stack "
        "+ Hz B0 for real-data ΔB0 grading (Path A).",
    )

    # ── Latent diffusion / coords / meta-learning (Phase 4f) ────────────────
    latent_diffusion: LatentDiffusionConfigSchema = Field(
        default_factory=LatentDiffusionConfigSchema,
        description="Lazy-encode wrapper for stage-2 latent diffusion. "
        "When enabled, the loader yields encoded latents via a frozen "
        "stage-1 checkpoint.",
    )
    emit_coordinates: CoordinateEmissionConfigSchema = Field(
        default_factory=CoordinateEmissionConfigSchema,
        description="Append a normalized coordinate grid to every batch. "
        "For INR / coord-MLP / position-aware models.",
    )
    meta_learning: MetaLearningConfigSchema = Field(
        default_factory=MetaLearningConfigSchema,
        description="Meta-learning task sampling. Exposes support_size / "
        "query_size / tasks_per_epoch / task_params via YAML instead of "
        "MetaLearningDataset constructor kwargs.",
    )

    # --- Validations ---
    @field_validator("acquisition_axes")
    @classmethod
    def validate_acquisition_axes(cls, value: list[str] | None) -> list[str] | None:
        """Reject an axis name the Axis enum does not define.

        A typo must raise here, not resolve to an empty set downstream: the
        whole point of the declaration is to make a regime's required axis
        checkable, and a silently-dropped ``"echoes"`` would turn a positive
        claim into "this arm carries nothing" -- rejecting the very arm it was
        written to admit (pitfall #9). Normalised to lower case so ``ECHO`` and
        ``echo`` are the same declaration.
        """
        if value is None:
            return None
        from spectramr.config.schemas.enums import Axis

        valid = {a.value for a in Axis}
        out: list[str] = []
        for raw in value:
            name = str(raw).strip().lower()
            if name not in valid:
                raise ValueError(
                    f"data.acquisition_axes: {raw!r} is not an Axis. Valid axes: {sorted(valid)}."
                )
            out.append(name)
        # Deterministic and duplicate-free, so two spellings of the same
        # declaration compare equal in provenance.
        return sorted(set(out))

    @field_validator("dataset_type")
    @classmethod
    def validate_dataset_type(cls, value: str) -> str:
        """validate_dataset_type.

        Args:
            value (str): Description.
        Returns:
            str: Description.
        """
        value = value.lower()

        # Single source of truth -- see CANONICAL_DATASET_TYPES /
        # DATASET_TYPE_ALIASES at module scope for why these were hoisted.
        valid = list(CANONICAL_DATASET_TYPES)
        aliases = DATASET_TYPE_ALIASES

        # Map alias to canonical type
        if value in aliases:
            value = aliases[value]

        # ``dataset_type`` is a DISPATCH key resolved through ``DATASET_REGISTRY``
        # (``DataPipelineDirector`` -> ``DatasetInstantiator``), exactly as
        # ``model_type`` is resolved through ``MODEL_REGISTRY``. Validating it
        # against the frozen 21-name literal alone meant a dataset registered
        # OUT OF TREE was rejected at load time however correctly it was
        # registered: models, losses and metrics were pluggable and datasets
        # alone were not.
        #
        # The registry is consulted through a function-local import, the same
        # shape ``TrainingSettings._finalize_from_dict`` already uses to resolve
        # ``model_type`` through ``spectramr.models.registry``. It stays local
        # because this module is imported during config-schema construction, long
        # before the data layer is needed.
        #
        # ``_ensure_registered`` is the ``populate_model_registry`` analogue: the
        # registry is EMPTY until it runs, so consulting it first would reject
        # every name -- the trap CLAUDE.md documents for MODEL_REGISTRY.
        #
        # The check stays HERE rather than moving to settings.py because it is a
        # Tier-0 guarantee: ``tests/unit/data/datasets/test_axis_exposure.py``
        # relies on an invalid ``dataset_type`` dying at schema construction,
        # before any workflow check runs.
        if value not in valid:
            try:
                from spectramr.data.builders.dataset_instantiator import (
                    DatasetInstantiator,
                )
                from spectramr.data.datasets.registry import DATASET_REGISTRY

                DatasetInstantiator._ensure_registered()
                registered = set(DATASET_REGISTRY)
            except Exception:  # pragma: no cover - data layer unavailable
                registered = set()
            if value not in registered:
                raise ValueError(
                    f"dataset_type={value!r} is not recognised. Canonical types: "
                    f"{list(CANONICAL_DATASET_TYPES)}. Registered out-of-tree: "
                    f"{sorted(registered - set(CANONICAL_DATASET_TYPES))}. "
                    f"Accepted aliases (folded to a canonical type before "
                    f"dispatch): {sorted(DATASET_TYPE_ALIASES)}."
                )
        return value

    @model_validator(mode="after")
    def validate_collation_consistency(self) -> "DataConfigSchema":
        """Validate collation configuration consistency with other data settings.

        Checks:
            1. If enable_slab_mode=True, warn if collation.strategy is not 'slab'
            2. If collation.strategy is explicit, it takes precedence

        Returns:
            Validated config instance

        Raises:
            Warning if configuration seems inconsistent
        """
        import logging

        logger = logging.getLogger(__name__)

        # Check slab mode consistency
        if self.sampling.enable_slab_mode and self.collation.strategy is not None:
            if self.collation.strategy != "slab":
                logger.warning(
                    f"[DataConfigSchema] enable_slab_mode=True but collation.strategy='{self.collation.strategy}'. "
                    f"User-specified strategy takes precedence, but consider setting collation.strategy='slab' "
                    f"for consistency with enable_slab_mode."
                )

        return self

    @model_validator(mode="after")
    def validate_split_strategy(self) -> "DataConfigSchema":
        """Validate split_strategy against required companion fields.

        - ``loso`` requires ``holdout_site`` to be set.
        - ``manifest`` requires ``index_path`` to be set.
        - ``directory`` warns if data_root has no ``train/`` subdir.

        Returns:
            Validated config instance.

        Raises:
            ValueError: If required companion fields are missing.
        """
        strategy = self.split.type
        if strategy == "loso_subject" and (self.split.holdout_subject is None) == (
            self.split.loso_fold is None
        ):
            raise ValueError(
                "split.type='loso_subject' requires EXACTLY ONE of "
                "'split.holdout_subject' (an explicit id) or 'split.loso_fold' (an index "
                "into the sorted subject list). Both would disagree; neither "
                "leaves the held-out subject undefined, and the run would "
                "silently validate on training subjects."
            )
        if strategy == "loso":
            if not self.split.holdout_site:
                raise ValueError(
                    "split.type='loso' requires 'split.holdout_site' to be set. "
                    "Specify the site name to hold out for validation."
                )
        elif (
            strategy == "manifest"
            and not self.source.index_path
            and not self.source.paired_manifest_path
        ):
            raise ValueError(
                "split.type='manifest' requires either 'source.index_path' "
                "or 'source.paired_manifest_path' to be set. Provide a path to "
                "the training manifest (pkl/json) or a v4 paired manifest."
            )
        return self

    @model_validator(mode="after")
    def validate_contrast_aware_requirements(self) -> "DataConfigSchema":
        """Validate that ``contrast_aware_paired`` has required contrast configs.

        The ``input_contrast`` and ``target_contrast`` fields are optional in
        the general schema but **required** when
        ``dataset_type == 'contrast_aware_paired'``.  Validating here gives a
        clear error at config-load time instead of a late crash in
        :class:`~spectramr.data.builders.dataset_instantiator.DatasetInstantiator`.

        Raises:
            ValueError: If either contrast config is missing.
        """
        if getattr(self, "dataset_type", None) == "contrast_aware_paired":
            missing = []
            if not getattr(self, "input_contrast", None):
                missing.append("input_contrast")
            if not getattr(self, "target_contrast", None):
                missing.append("target_contrast")
            if missing:
                raise ValueError(
                    f"dataset_type='contrast_aware_paired' requires "
                    f"{', '.join(missing)} to be set in the data config. "
                    f"Each must define at least a 'name' (e.g. 'ULF_64mT') "
                    f"and optionally normalization parameters."
                )
        return self

    # ── Phase 3: mode-dispatch helpers ──────────────────────────────────────

    @model_validator(mode="after")
    def derive_modes_from_legacy(self) -> "DataConfigSchema":
        """Backward-compat shim: populate ``modes.{train,val,infer,eval}``
        from legacy top-level fields when each mode is absent.

        Rules (preserve current behavior):

        - ``train``: sampler=uniform, samples_per_volume from
          ``self.sampling.samples_per_volume``, queue_length from
          ``self.sampling.queue_length``, shuffle=True, augmentation from
          ``self.augmentation.enabled``.
        - ``val``:   sampler from ``self.use_queue_for_validation``
          (True → uniform, False → full), shuffle=False,
          augmentation_enabled=False.
        - ``infer``: sampler=full, shuffle=False,
          augmentation_enabled=False.
        - ``eval``:  sampler=full, shuffle=False,
          augmentation_enabled=False, role=clean_reference.

        A user-supplied ``data.modes.<mode>`` block wins entirely —
        the validator never merges partial overrides; if you declare
        ``data.modes.train``, you declare ALL of train.

        This is intentional: silent merging of legacy + override fields
        would violate CLAUDE.md #9 (silent fallbacks forbidden) — better
        to make the precedence explicit at the block level.
        """
        # ``frozen=True`` requires model_copy/model_construct for mutation.
        # Build new ModeConfigSchemas for any absent mode, then re-build
        # ``modes`` with the populated set.

        # Bail out early when modes are already fully populated.
        if (
            self.modes.train is not None
            and self.modes.val is not None
            and self.modes.infer is not None
            and self.modes.eval is not None
        ):
            return self

        # Build each absent mode from legacy fields.
        derived_train = self.modes.train
        derived_val = self.modes.val
        derived_infer = self.modes.infer
        derived_eval = self.modes.eval

        if derived_train is None:
            derived_train = ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    type="uniform",
                    samples_per_volume=self.sampling.samples_per_volume,
                    queue_length=self.sampling.queue_length,
                    shuffle=True,
                ),
                augmentation_enabled=bool(self.augmentation.enabled),
            )

        if derived_val is None:
            # ``use_queue_for_validation`` is not a real schema field — the
            # queue builder consults it via ``hasattr`` (see
            # torchio_queue_builder.py:164-168) when YAMLs declare it
            # in the data: block (and DataConfigSchema's
            # extra="ignore" keeps it from raising). Mirror the same
            # discipline here so legacy YAMLs that DO declare it get
            # the expected uniform-val behavior.
            use_queue_for_val = bool(getattr(self, "use_queue_for_validation", False))
            derived_val = ModeConfigSchema(
                sampler=ModeSamplerSchema(
                    type="uniform" if use_queue_for_val else "full",
                    samples_per_volume=self.sampling.samples_per_volume,
                    queue_length=self.sampling.queue_length,
                    shuffle=False,
                ),
                augmentation_enabled=False,
            )

        if derived_infer is None:
            derived_infer = ModeConfigSchema(
                sampler=ModeSamplerSchema(type="full", shuffle=False),
                augmentation_enabled=False,
            )

        if derived_eval is None:
            derived_eval = ModeConfigSchema(
                sampler=ModeSamplerSchema(type="full", shuffle=False),
                augmentation_enabled=False,
                role="clean_reference",
            )

        # Rebuild the modes container (frozen → construct anew).
        # Use ``object.__setattr__`` to bypass the frozen lock on this
        # single internal mutation. Done in a model_validator so it runs
        # after construction completes.
        new_modes = DataModesSchema(
            train=derived_train,
            val=derived_val,
            infer=derived_infer,
            eval=derived_eval,
        )
        object.__setattr__(self, "modes", new_modes)
        return self

    def resolve_mode(
        self,
        mode: Literal["train", "val", "infer", "eval"],
    ) -> ModeConfigSchema:
        """Single public entry point for builders to read per-mode config.

        After ``derive_modes_from_legacy`` runs, ``self.modes.<mode>`` is
        always a fully-populated :class:`ModeConfigSchema`. This helper
        gives callers a typed, never-None handle keyed by mode string.

        Args:
            mode: One of ``"train"``, ``"val"``, ``"infer"``, ``"eval"``.

        Returns:
            The fully resolved :class:`ModeConfigSchema` for that mode.
        """
        cfg = getattr(self.modes, mode, None)
        if cfg is None:
            # Should be unreachable after derive_modes_from_legacy runs.
            # Surface loudly if invariant breaks (CLAUDE.md #9).
            raise RuntimeError(
                f"DataConfigSchema.modes.{mode} is None after "
                "derive_modes_from_legacy validator — invariant broken."
            )
        return cfg

    def is_graph_paradigm(self) -> bool:
        """Amendment A: graph datasets bypass the spatial mode dispatcher.

        Intended for ``DataPipelineDirector``, to early-exit before applying
        ``GridSampler`` / ``Resample`` / patch-based transforms that
        don't make sense for graph data (which has no spatial axes in
        the TorchIO sense).

        .. warning::
           **Nothing calls this.** ``DataPipelineDirector`` names no graph
           early-exit, no arm in ``experiments/`` declares the flag, and the
           live graph path is a *different* field of the same name on
           ``TorchIOTransformConfig`` (``torchio_transform_builder.py:469``,
           read at :1229). Tracked separately; do not assume the early-exit
           described above happens.

        Returns:
            True if this config requests graph encoding — in which
            case patch-sampling, resampling, and grid-tiling are skipped.
        """
        # `enable_graph_encoding` folded to `domain.enable_graph_encoding`. A
        # YAML declaration of the legacy spelling still loads, but this Python
        # read raised AttributeError, so the method could not be called at all.
        return bool(self.domain.enable_graph_encoding)


# Resolve the forward reference on ``ModeConfigSchema.output`` now that
# ``ModeOutputSchema`` is in the module scope (Phase 4d).
ModeConfigSchema.model_rebuild()
