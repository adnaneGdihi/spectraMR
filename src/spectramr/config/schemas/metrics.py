"""Metrics tracking and profiling configuration schemas."""

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import MetricMode
from .nr_metrics import NRMetricConfig
from .renames import reject_renamed_keys

_UNREAD_BEST_MODE = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). 922 arms declare it "
    'and this block is extra="forbid", so deleting the field would fail every one of them at load. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)
_UNREAD_EVAL_EPOCH = (
    "NOT READ — nothing in src/ consumes it (KNOWN_UNCONSUMED ledger, "
    "tests/unit/config/test_schema_key_consumption.py). ~900 arms declare it "
    'and this block is extra="forbid", so deleting the field would fail every one of them at load. '
    "Wire it or retire it with a corpus migration; do not trust it today."
)


class MetricsConfigSchema(BaseModel):
    """Metrics computation and tracking configuration.

    Defines which evaluation metrics are computed during training.

    SSOT Principle: This schema is the canonical source for metric selection.
    Metrics are extracted ONCE from config.metrics.*, never hard-coded or re-parsed.

    Paradigm-Specific Metrics:
        reconstruction: PSNR, SSIM, MAE, NRMSE, gradient_snr, contrast
        gan: PSNR, SSIM, MSE, FID, Inception Score
        diffusion: PSNR, SSIM, MAE, MSE, NRMSE, gradient_snr
        vae: PSNR, SSIM, MAE, correlation
        ssl: cosine_similarity, correlation, MSE

    Example:
        >>> config = MetricsConfigSchema(
        ...     enable_tracking=True,
        ...     compute_fid=True,
        ...     compute_lpips=True,
        ...     paradigm_specific_metrics={
        ...         "psnr": True,
        ...         "ssim": True,
        ...         "mae": False,
        ...     }
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",  # Strict schema: metrics must be explicitly declared
        "frozen": True,
    }

    # Removed keys (removals.py) are refused with the reason, not a bare extra error.
    _reject_renamed = model_validator(mode="before")(classmethod(reject_renamed_keys("metrics")))

    compute: list[str] = Field(
        default_factory=list,
        description=(
            "Metrics to compute, by registered name: [psnr, ssim, hfen, ...]. "
            "When non-empty this REPLACES the compute_* flags entirely -- the "
            "flags are not merged in, so the list reads as the complete answer "
            "to 'what does this arm measure?'. Names are validated against "
            "MetricsRegistry by the Tier-1 audit."
        ),
    )
    #: Why a list rather than more flags.
    #:
    #: 209 metrics are registered; 86 ``compute_*`` flags exist; so 145 metrics
    #: are UNREACHABLE from any config (issue #343) -- registering a metric does
    #: not make it selectable, and nothing said so. Seventeen flags run the
    #: other way and name a metric that is not registered (#340).
    #:
    #: A list closes both by construction: registry membership IS the validator,
    #: so an unreachable metric cannot exist and a dead name raises instead of
    #: being silently ignored. It also answers the readability question this
    #: whole effort exists for -- 86 booleans do not tell a reader which metrics
    #: run, and a five-element list does.

    enable_tracking: bool = Field(
        default=True,
        description="Enable metrics computation",
    )
    compute_fid: bool = Field(
        default=False,
        description="Compute Fréchet Inception Distance (FID)",
    )
    compute_lpips: bool = Field(
        default=False,
        description="Compute LPIPS (learned perceptual image patch similarity)",
    )
    compute_ssim: bool = Field(
        default=True,
        description="Compute SSIM (structural similarity index)",
    )
    compute_psnr: bool = Field(
        default=True,
        description="Compute PSNR (peak signal-to-noise ratio)",
    )
    compute_robust_mri_psnr: bool = Field(
        default=False,
        description="Compute RobustMRI_PSNR (ROI-masked PSNR with 5% threshold, fixes background zero-inflation)",
    )
    compute_mse: bool = Field(
        default=True,
        description="Compute MSE (mean squared error)",
    )
    compute_mae: bool = Field(
        default=True,
        description="Compute MAE (mean absolute error)",
    )
    compute_inception_score: bool = Field(
        default=False,
        description="Compute Inception Score",
    )
    compute_precision_recall: bool = Field(
        default=False,
        description=(
            "UNIMPLEMENTED — must stay False. `precision_recall` is not in "
            "MetricsRegistry under any name or alias, so enabling this selects a "
            "metric that cannot be computed. Kept as a field only because 29 "
            "arms under experiments/training/ declare it (all False) and the "
            "metrics block is extra='forbid', so deleting it would fail their "
            "load. Setting it True raises (#340/#660). Register the metric, or "
            "drop the key from those 29 arms and delete this field."
        ),
    )

    @field_validator("compute_precision_recall")
    @classmethod
    def _refuse_unimplemented_precision_recall(cls, v: bool) -> bool:
        """An advertised knob must be wired or must refuse (pitfall #15).

        Five sibling flags naming unregistered metrics were deleted outright;
        this one has 29 corpus declarations, so it survives as a field that can
        only hold its default. That is the honest middle: the arm still loads,
        and an arm that tries to turn it ON is told at startup rather than
        discovering a silently missing CSV column after a full run (#173).
        """
        if v:
            raise ValueError(
                "metrics.compute_precision_recall=True selects `precision_recall`, "
                "which is not registered in MetricsRegistry under any name or "
                "alias — no column can ever be produced for it. Register the "
                "metric first, or leave the flag False."
            )
        return v

    compute_hfen: bool = Field(
        default=False,
        description="Compute HFEN (High-Frequency Error Norm) for edge sharpness",
    )
    compute_advanced_metrics: bool = Field(
        default=True,
        description="Compute advanced medical metrics (VIF, FSIM, GMSD, etc.)",
    )

    # ==================== QUALITY METRICS ====================
    compute_nmse: bool = Field(
        default=False,
        description="Compute NMSE (Normalized Mean Squared Error)",
    )
    compute_nrmse: bool = Field(
        default=False,
        description="Compute NRMSE (Normalized Root Mean Squared Error)",
    )
    compute_snr: bool = Field(
        default=False,
        description="Compute SNR (Signal-to-Noise Ratio in dB)",
    )
    compute_gmsd: bool = Field(
        default=False,
        description="Compute GMSD (Gradient Magnitude Similarity Deviation)",
    )
    compute_vif: bool = Field(
        default=False,
        description="Compute VIF (Visual Information Fidelity)",
    )
    compute_fsim: bool = Field(
        default=False,
        description="Compute FSIM (Feature Similarity Index)",
    )
    compute_uqi: bool = Field(
        default=False,
        description="Compute UQI (Universal Quality Index)",
    )
    compute_ms_ssim: bool = Field(
        default=False,
        description="Compute MS-SSIM (Multi-Scale SSIM)",
    )
    compute_kid: bool = Field(
        default=False,
        description="Compute KID (Kernel Inception Distance)",
    )

    # ==================== K-SPACE / PHASE METRICS ====================
    compute_kspace_error: bool = Field(
        default=False,
        description="Compute k-space consistency error",
    )
    compute_phase_mse: bool = Field(
        default=False,
        description="Compute phase MSE for complex-valued data",
    )
    compute_gradient_entropy: bool = Field(
        default=False,
        description="Compute gradient entropy (sharpness measure)",
    )
    compute_ipen: bool = Field(
        default=False,
        description="Compute IPEN (Image Phase Error Norm)",
    )

    # ==================== ARTIFACT / QA METRICS ====================
    compute_efc: bool = Field(
        default=False,
        description="Compute EFC (Entropy Focus Criterion - ghosting measure)",
    )
    compute_fber: bool = Field(
        default=False,
        description="Compute FBER (Foreground-Background Energy Ratio)",
    )
    compute_qi1: bool = Field(
        default=False,
        description="Compute QI1 (artifacts outside brain mask)",
    )
    compute_cjv: bool = Field(
        default=False,
        description="Compute CJV (Coefficient of Joint Variation)",
    )
    compute_cnr: bool = Field(
        default=False,
        description="Compute CNR (Contrast-to-Noise Ratio)",
    )

    # === Quantitative-imaging agreement metrics (PMPS / vf sim arms) ===
    # Bland-Altman and friends are essential when the experiment
    # validates a quantitative map (B0/B1, T1/T2, MRF) against a
    # ground-truth digital twin: PSNR is no longer the right summary.
    # See exp_vf_21..31_sim_v2.yaml (B0/B1 sim arms) and the M4Raw 0.3 T
    # campaigns.
    compute_bland_altman_bias: bool = Field(
        default=False,
        description=(
            "Compute Bland-Altman bias (mean prediction - ground-truth) "
            "for quantitative-map validation."
        ),
    )
    compute_limits_of_agreement_lower: bool = Field(
        default=False,
        description=("Compute lower Bland-Altman Limit of Agreement (bias - 1.96 * SD)."),
    )
    compute_limits_of_agreement_upper: bool = Field(
        default=False,
        description=("Compute upper Bland-Altman Limit of Agreement (bias + 1.96 * SD)."),
    )
    compute_coefficient_of_variation: bool = Field(
        default=False,
        description=(
            "Compute coefficient of variation (CoV = SD/mean) — repeatability "
            "summary for quantitative maps."
        ),
    )
    compute_folding_fraction: bool = Field(
        default=False,
        description=(
            "Compute folding fraction — proportion of voxels with "
            "non-positive Jacobian under a displacement field "
            "(diffeomorphism integrity check)."
        ),
    )
    compute_icc_3_1: bool = Field(
        default=False,
        description=(
            "Compute ICC(3,1) (Shrout-Fleiss two-way mixed-effects "
            "single-measurement absolute-agreement intraclass "
            "correlation). Standard repeatability metric for "
            "quantitative-MRI parameter maps."
        ),
    )

    # ==================== SEGMENTATION METRICS ====================
    compute_dice: bool = Field(
        default=False,
        description="Compute Dice coefficient",
    )
    compute_iou: bool = Field(
        default=False,
        description="Compute IoU (Intersection over Union)",
    )
    compute_hd95: bool = Field(
        default=False,
        description="Compute HD95 (95th percentile Hausdorff Distance)",
    )

    # ==================== TEMPORAL / fMRI METRICS ====================
    compute_tsnr: bool = Field(
        default=False,
        description="Compute tSNR (temporal SNR for fMRI): mean_t/std_t per voxel "
        "after detrending, averaged over the foreground. Reference-free. NOTE: "
        "tSNR rewards temporal smoothing and cannot distinguish noise removal "
        "from dynamics destruction — pair it with compute_temporal_fidelity. "
        "This flag was DEAD until 2026-07-16: no `tsnr` metric existed.",
    )
    compute_temporal_fidelity: bool = Field(
        default=False,
        description="Compute temporal fidelity: mean Pearson correlation between "
        "predicted and reference voxel time-courses. The metric that catches a "
        "reconstruction collapsing to the temporal mean — per-frame PSNR/SSIM "
        "cannot see that, because they reduce over the time axis.",
    )

    # ==================== CORRELATION METRICS ====================
    compute_pearson: bool = Field(
        default=False,
        description="Compute Pearson correlation coefficient",
    )
    compute_cosine_similarity: bool = Field(
        default=False,
        description="Compute cosine similarity",
    )

    # ==================== ADVANCED PERCEPTUAL METRICS ====================
    compute_pdm: bool = Field(
        default=False,
        description="Compute PDM (Perceptual Difference Model)",
    )
    compute_cw_ssim: bool = Field(
        default=False,
        description="Compute CW-SSIM (Complex Wavelet SSIM)",
    )
    compute_mad: bool = Field(
        default=False,
        description="Compute MAD (Most Apparent Distortion)",
    )
    compute_st_mad: bool = Field(
        default=False,
        description="Compute ST-MAD (Spatio-Temporal MAD)",
    )
    compute_dists: bool = Field(
        default=False,
        description="Compute DISTS (Deep Image Structure and Texture Similarity)",
    )

    # ==================== RADIOMICS METRICS ====================
    compute_frd: bool = Field(
        default=False,
        description="Compute FRD (Fréchet Radiomic Distance) - Requires pyradiomics",
    )
    compute_rfs: bool = Field(
        default=False,
        description="Compute RFS (Radiomic Feature Stability) - Requires pyradiomics",
    )

    # ==================== GRADIENT / EDGE METRICS ====================
    compute_gradient_error: bool = Field(
        default=False,
        description="Compute gradient magnitude error",
    )

    # ==================== DIFFUSION MRI METRICS ====================
    compute_cc_snr: bool = Field(
        default=False,
        description="Compute CC-SNR (Corpus Callosum SNR for diffusion)",
    )
    compute_ndc: bool = Field(
        default=False,
        description="Compute NDC (Noise-corrected Diffusion Coefficient)",
    )
    compute_ndc_diffusion: bool = Field(
        default=False,
        description="Compute NDC for diffusion MRI",
    )
    compute_spike_percentage: bool = Field(
        default=False,
        description="Compute spike percentage in diffusion data",
    )

    # ==================== PERFUSION METRICS ====================
    compute_ktrans: bool = Field(
        default=False,
        description="Compute Ktrans (Tofts model parameter)",
    )
    compute_wash_slope: bool = Field(
        default=False,
        description="Compute wash-in/wash-out slope",
    )
    compute_bat: bool = Field(
        default=False,
        description="Compute BAT (Bolus Arrival Time)",
    )
    compute_neg_voxels: bool = Field(
        default=False,
        description="Compute negative voxel count",
    )

    # ==================== SPECTROSCOPY METRICS ====================
    compute_spectral_linewidth: bool = Field(
        default=False,
        description="Compute spectral linewidth (MRS)",
    )
    compute_crlb: bool = Field(
        default=False,
        description="Compute CRLB (Cramer-Rao Lower Bound)",
    )
    compute_freq_domain_snr: bool = Field(
        default=False,
        description="Compute frequency-domain SNR",
    )

    # ==================== FLOW METRICS ====================
    compute_mass_conservation: bool = Field(
        default=False,
        description="Compute mass conservation (4D flow)",
    )
    compute_divergence: bool = Field(
        default=False,
        description="Compute velocity divergence",
    )
    compute_vnr: bool = Field(
        default=False,
        description="Compute VNR (Velocity-to-Noise Ratio)",
    )

    # ==================== ADDITIONAL K-SPACE METRICS ====================
    # NOTE: compute_kspace_entropy / compute_kspace_high_freq_error were removed
    # (2026-07-18) — no metric implements them, so enabling either raised
    # ConfigurationError ("not a registered metric") at build time. Advertising an
    # unimplemented knob is pitfall #15; backlog the metric + registration if wanted.
    compute_complex_hfen: bool = Field(
        default=False,
        description="Compute complex-valued HFEN",
    )
    compute_spike_detection: bool = Field(
        default=False,
        description="Compute spike detection in k-space",
    )
    # NOTE: compute_spike_percent was removed 2026-07-18 — a byte-duplicate of
    # compute_spike_percentage above whose identity target 'spike_percent' was never
    # registered (the registered metric is 'spike_percentage'), so it silently never
    # fired. Enable compute_spike_percentage instead.

    # ==================== ADDITIONAL QA METRICS ====================
    compute_wm2max: bool = Field(
        default=False,
        description="Compute WM2Max ratio",
    )
    compute_fwhm: bool = Field(
        default=False,
        description="Compute FWHM (Full Width at Half Maximum)",
    )
    compute_gsr: bool = Field(
        default=False,
        description="Compute GSR (Ghost-to-Signal Ratio)",
    )
    compute_volume_similarity: bool = Field(
        default=False,
        description="Compute volume similarity",
    )
    compute_rmse: bool = Field(
        default=False,
        description="Compute RMSE (Root Mean Squared Error)",
    )

    # Paradigm-Specific Metrics Configuration
    enable_paradigm_specific: bool = Field(
        default=True,
        description="Enable paradigm-aware metric selection (overrides flags)",
    )
    paradigm_specific_metrics: dict[str, bool] = Field(
        default_factory=dict,
        description="Paradigm-specific metrics config (overrides defaults if provided)",
    )

    # Metric computation intervals
    metric_interval: int = Field(
        default=1000,
        ge=1,
        description=(
            "NOT READ. 855 arms set this, and nothing consults it: the only "
            "mention is a same-named PARAMETER of "
            "`IterationCounterService.should_compute_metrics`, which is called "
            "solely from tests. The throttle that actually runs is "
            "`train_metric_interval` below. Wiring this one would change "
            "behaviour for 418 arms (141 of them would get 20x MORE metric "
            "computation), so it is an owner decision, not a fix."
        ),
    )
    train_metric_interval: int = Field(
        default=100,
        ge=1,
        description=(
            "Compute TRAINING metrics every N steps. This is the throttle that "
            "runs -- `MetricsMixin._compute_training_metrics` reads it at "
            "`metrics_mixin.py:786`. It was never a schema field, and this block "
            'is `extra="forbid"`, so it was unreachable from YAML and hard-wired '
            "to its 100 default. Declared here at that same default, so every "
            "existing arm is unchanged and the knob becomes settable."
        ),
    )
    eval_on_epoch: bool = Field(
        default=True,
        description=("Evaluate metrics at end of epoch. " + _UNREAD_EVAL_EPOCH),
    )

    # Best model tracking (`track_best_metric` was removed, #568)
    best_metric_name: str = Field(
        default="val_loss",
        description="Metric to optimize",
    )
    best_metric_mode: MetricMode = Field(
        default=MetricMode.MIN,
        description=("Whether to minimize or maximize metric. " + _UNREAD_BEST_MODE),
    )

    # Domain-specific configuration
    domain: str | None = Field(
        default=None,
        description="Domain for metric calculation (e.g., 'image', 'kspace')",
    )
    transform: str | None = Field(
        default=None,
        description=(
            "Transform applied before metric calculation: 'ifft_magnitude', "
            "'ifft_sense_adjoint', 'magnitude', or None ('sense_magnitude' is "
            "validation-only: this path has no batch coil maps and raises). "
            "Read on the TRAINING-"
            "metrics path only (MetricsMixin._compute_training_metrics); the "
            "validation path reads validation.scoring.output_transform instead. "
            "Any other name RAISES -- 'ifft_mag_combine' / 'ifft_mag' are "
            "declared on 146 arms and implemented by nothing, and are "
            "deliberately not aliased onto 'ifft_magnitude' because 112 of "
            "those arms output images, where an IFFT gives a Fourier magnitude "
            "rather than a coil combine (#931). "
            "SSOT: infrastructure/training/utils/metric_transform.py."
        ),
    )
    # [FIX] Add this field to whitelist 'output_dir' in the YAML
    output_dir: str | None = Field(
        default=None,
        description="Custom directory to save metrics artifacts (images, CSVs)",
    )

    # No-reference metric battery (spec §4). ``nr.enabled_metrics`` lists the NR
    # registry keys to compute at validation time; ``nr`` tunables/asset paths
    # flow to each metric via MetricContext.nr_params. Read by
    # MetricsMixin._extract_metrics_from_config and the
    # ``nr_metric_context_wired`` audit check.
    nr: NRMetricConfig = Field(
        default_factory=NRMetricConfig,
        description="No-reference (label-free) metric battery configuration.",
    )


class ProfilingConfigSchema(BaseModel):
    """Performance profiling configuration.

    Defines which performance metrics are tracked.

    Example:
        >>> config = ProfilingConfigSchema(
        ...     enabled=True,
        ...     profile_memory=True,
        ...     profile_time=True,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable performance profiling",
    )
    profile_memory: bool = Field(
        default=False,
        description="Profile GPU/CPU memory usage",
    )
    profile_time: bool = Field(
        default=False,
        description="Profile execution time",
    )
    profile_gradients: bool = Field(
        default=False,
        description="Profile gradient computation",
    )
    profile_backward: bool = Field(
        default=False,
        description="Profile backward pass",
    )

    # Profiling interval
    profile_interval: int = Field(
        default=100,
        ge=1,
        description="Profile every N steps",
    )
    profile_warmup_steps: int = Field(
        default=10,
        ge=0,
        description="Warmup steps before profiling",
    )

    # Output
    save_profile: bool = Field(
        default=False,
        description="Save profile data to file",
    )
    profile_output_dir: str = Field(
        default="./profiles",
        description="Directory to save profile data",
    )


__all__ = ["MetricsConfigSchema", "ProfilingConfigSchema"]
