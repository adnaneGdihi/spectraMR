"""GeoMamba-ULF training paradigm sub-section schema.

Located at: ``training.geomamba_ulf`` in v6.0 YAML configs.

Used when ``training.strategy_class`` resolves to
``GeoMambaULFStrategy``. Holds parameters for the four genuinely-new
components of the paradigm:

- **metric_sfc**: foreground-restricted metric-aware space-filling
  curve (Kruskal MST + DFS, intensity-weighted edge metric).
- **film**: contrast embedding dimension and FiLM modulation depth
  for the contrast-conditioned bidirectional Mamba blocks.
- **unwarp**: physical unwarp composition (B0 + gradient nonlinearity
  + Maxwell concomitant), with diagnostics.
- **topology**: cubical persistent homology and Beltrami diagnostic
  weighting and stratification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema


class MetricSFCConfig(StrictSchema):
    """Metric-aware space-filling-curve linearizer parameters."""

    beta: float = Field(
        default=10.0,
        ge=0.0,
        description=(
            "Intensity edge-weight coefficient: edge weight is "
            "||u-v|| * sqrt(1 + beta * (I(u)-I(v))^2)."
        ),
    )
    smoothing_sigma: float = Field(
        default=1.0,
        ge=0.0,
        description="Gaussian sigma for the pre-denoise proxy used by the metric.",
    )
    cache_dir: str | None = Field(
        default="cache/metric_sfc",
        description=(
            "On-disk cache for permutations keyed by (volume_shape, mask_hash). "
            "First epoch pays MST cost; subsequent epochs are O(1) lookup. "
            "Set to null to disable caching."
        ),
    )
    connectivity: int = Field(
        default=26,
        description="Voxel connectivity for the foreground graph (6 / 18 / 26).",
    )


class FiLMConfig(StrictSchema):
    """FiLM contrast-conditioning parameters for Mamba blocks."""

    contrast_dim: int = Field(
        default=8,
        ge=1,
        description="Contrast-embedding dimension fed into the FiLM MLP.",
    )
    n_contrasts: int = Field(
        default=3,
        ge=1,
        description="Number of distinct contrasts (T1, T2, FLAIR by default).",
    )
    film_hidden_mult: int = Field(
        default=4,
        ge=1,
        description="Hidden-layer multiplier in the FiLM MLP (hidden = mult * d_model).",
    )


class UnwarpConfig(StrictSchema):
    """Physical unwarp composition parameters."""

    enable_b0: bool = Field(default=True, description="Apply B0 inhomogeneity unwarp.")
    enable_grad_nonlinearity: bool = Field(
        default=True, description="Apply gradient-nonlinearity unwarp."
    )
    enable_maxwell: bool = Field(
        default=True,
        description=(
            "Apply Maxwell concomitant-gradient correction "
            "(dominant at ULF, negligible above ~1.5T)."
        ),
    )
    field_strength_t: float = Field(
        default=0.064,
        gt=0.0,
        description="Static field strength in Tesla (0.064 for Hyperfine, 0.3 for M4Raw).",
    )
    emit_beltrami_diagnostic: bool = Field(
        default=True,
        description=(
            "Emit Beltrami coefficient ||mu||_inf as a per-step diagnostic. "
            "Diagnostic only — not a training regularizer at v0."
        ),
    )
    gradient_amplitudes_mt_per_m: list[float] = Field(
        default_factory=lambda: [5.0, 5.0, 5.0],
        description=(
            "Maximum gradient amplitudes (Gx, Gy, Gz) in mT/m. "
            "Used to compute gradient-nonlinearity warps and Maxwell concomitant fields. "
            "Hyperfine 0.064T scanner: ~5 mT/m typical; M4Raw 0.3T: ~22 mT/m."
        ),
    )
    voxel_size_mm: list[float] = Field(
        default_factory=lambda: [1.0, 1.0, 1.0],
        description=(
            "Voxel size (dx, dy, dz) in mm. Used to scale displacement fields "
            "into physical space when composing the unwarp."
        ),
    )
    shift_mode: Literal["epi", "readout"] = Field(
        default="epi",
        description=(
            "How a frequency offset becomes a pixel shift. 'epi': shift = df * echo_spacing_s * "
            "N_pe along the phase-encode axis. 'readout': shift = df / pixel_bandwidth_hz along "
            "the readout axis."
        ),
    )
    echo_spacing_s: float | None = Field(
        default=None,
        gt=0.0,
        description="EPI echo spacing in seconds. Required when shift_mode is 'epi'.",
    )
    pixel_bandwidth_hz: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Readout pixel bandwidth in Hz (DICOM PixelBandwidth). Required when shift_mode is "
            "'readout'; the batch does not carry it, so it is declared here."
        ),
    )
    b0_map_key: str | None = Field(
        default=None,
        description="Batch field holding the B0 offset map in Hz. Required when enable_b0 is true.",
    )
    gnl_coeff_path: str | None = Field(
        default=None,
        description=(
            "Gradient-nonlinearity coefficient file. Declared for a reader that is not wired yet; "
            "setting it raises."
        ),
    )

    @model_validator(mode="after")
    def _shift_model_is_declared_when_a_term_is_on(self) -> UnwarpConfig:
        if not (self.enable_b0 or self.enable_maxwell):
            return self
        if self.shift_mode == "epi" and self.echo_spacing_s is None:
            raise ValueError(
                "echo_spacing_s is required when shift_mode='epi' and enable_b0 or enable_maxwell "
                "is true: the frequency offset has no pixel shift without it."
            )
        if self.shift_mode == "readout" and self.pixel_bandwidth_hz is None:
            raise ValueError(
                "pixel_bandwidth_hz is required when shift_mode='readout' and enable_b0 or "
                "enable_maxwell is true."
            )
        return self


class TopologyConfig(StrictSchema):
    """Persistent-homology and Beltrami loss parameters."""

    ph_dimensions: list[int] = Field(
        default_factory=lambda: [0, 1],
        description=(
            "Homology dimensions to track (0 = connected components, 1 = loops). "
            "Cubical PH for 3D supports up to dim 2."
        ),
    )
    wasserstein_p: int = Field(
        default=2,
        ge=1,
        description="Wasserstein order for diagram distance (2 by default per plan §4.3).",
    )
    stratify_subvolumes: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of stratified sub-volumes per batch element used to compute PH "
            "(>1 is a memory/time mitigation per plan §10 risk register)."
        ),
    )
    warmup_epochs: int = Field(
        default=0,
        ge=0,
        description=(
            "Linearly ramp the cubical-PH and Beltrami loss weights from 0 to "
            "their target values over the first N epochs. 0 disables warmup "
            "(constant weights from epoch 0). Plan §11.1 P3 promotes the "
            "topology losses from diagnostic to active regulariser; warmup "
            "stabilises that transition."
        ),
    )


class UncertaintyConfig(StrictSchema):
    """Per-contrast uncertainty-weighted loss parameters (Kendall & Gal 2018)."""

    enabled: bool = Field(
        default=False,
        description=(
            "Replace the foreground-masked L1 with per-contrast log-sigma "
            "uncertainty weighting. P2 milestone (single multi-contrast model "
            "vs per-contrast specialists)."
        ),
    )
    init_log_sigma: float = Field(
        default=0.0,
        description="Initial value of every learnable log-sigma slot (sigma = exp(init)).",
    )
    eps: float = Field(
        default=1.0e-6,
        gt=0.0,
        description="Numerical floor on the precision factor 1 / (2 * sigma^2).",
    )


class GeoMambaULFTrainingConfigSchema(StrictSchema):
    """GeoMamba-ULF paradigm-specific training parameters.

    Attached to ``TrainingStrategyConfigSchema.geomamba_ulf``.
    """

    metric_sfc: MetricSFCConfig = Field(default_factory=MetricSFCConfig)
    film: FiLMConfig = Field(default_factory=FiLMConfig)
    # An undeclared block builds no unwarp (GeoMambaULFStrategy._build_ulf_unwarp), so its default
    # has every term off; the all-on field defaults would fail the shift-parameter check.
    unwarp: UnwarpConfig = Field(
        default_factory=lambda: UnwarpConfig(
            enable_b0=False, enable_grad_nonlinearity=False, enable_maxwell=False
        )
    )
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)


__all__ = [
    "FiLMConfig",
    "GeoMambaULFTrainingConfigSchema",
    "MetricSFCConfig",
    "TopologyConfig",
    "UncertaintyConfig",
    "UnwarpConfig",
]
