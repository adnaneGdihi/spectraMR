"""Physics-related configuration schema for MRI reconstruction.

This file mirrors the file layout under ``src/infrastructure/physics/``.
Use the table of contents below to navigate.

╭────────────────────── Table of Contents ───────────────────────╮
│  Line  Class                       Concern                     │
├────────────────────────────────────────────────────────────────┤
│   12   DataConsistencyConfig       hard / soft DC + noise sim  │
│   93   CoilSensitivityConfig       ESPIRiT, calibration ACS    │
│  146   KSpaceConfig                Hermitian, zero-pad recon   │
│  190   RegularizationConfig        TV, wavelet, sparsity       │
│  274   IterativeRefinementConfig   unrolled steps, DC weight   │
│  314   PhaseCorrectionConfig       per-voxel phase fitting     │
│  347   B0CorrectionConfig          B0 unwarping during recon   │
│  382   B0SimulationConfig          synthetic B0 augmentation   │
│  423   MotionCorrectionConfig      runtime motion correction   │
│  459   CompressedSensingConfig     CS sampling pattern + recon │
│  533   PINNConfig                  PDE-loss residuals          │
│  599   BlochSimulationConfig       differentiable Bloch sim    │
│  634   DigitalTwinConfig           full digital-twin pipeline  │
│  926   PhysicsConfigSchema         top-level container         │
╰────────────────────────────────────────────────────────────────╯

For the actual physics implementations, see ``src/infrastructure/physics/``
which is the SSOT per CLAUDE.md (no raw FFT calls — use ``fft_ops``).
This module only schematises *which* physics blocks are enabled and how
they're parameterised at config-load time.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .enums import AccelerationSchedule


class DataConsistencyConfig(BaseModel):
    """Data consistency constraint configuration with realistic noise simulation.

    **CRITICAL DATA LEAK FIX**: Added noise level parameters to prevent the model
    from training with perfect ground truth k-space that won't be available at
    inference. Realistic noise is added to measured k-space during training.

    Data consistency enforces that the predicted reconstruction matches
    the acquired k-space measurements in the undersampled regions.

    Example:
        >>> dc_config = DataConsistencyConfig(
        ...     enabled=True,
        ...     weight=1.0,
        ...     interval=1,
        ...     train_noise_level=0.01,  # 1% noise during training
        ...     eval_noise_level=0.005,  # 0.5% noise during inference
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    # ==================== ACS & STABILIZATION (Phase 2) ====================
    acs_mask_center_fraction: float = Field(
        default=0.08,
        ge=0.0,
        le=0.5,
        description="Fraction of center lines to treat as ACS (Auto-Calibration Signal) for masking DC updates in low-frequencies.",
    )
    enable_acs_decoupling: bool = Field(
        default=False,
        description="If True, nullifies DC gradient update in the ACS center to prevent Bessel ringing artifacts (Step 2.2).",
    )
    enable_acs_replacement: bool = Field(
        default=True,
        description="If True, hard-replaces predicted ACS lines with ground truth lines during inference/validation (Step 2.3).",
    )

    # ==================== CORE DC PARAMETERS ====================
    enabled: bool = Field(
        default=False,
        description="Enable physics-based data consistency",
    )
    apply_at_predict: bool = Field(
        default=False,
        description="Project the prediction onto the measured k-space at predict time, as "
        "the last step of inference: every sampled bin takes the measurement (hard "
        "replacement, weight 1.0 by construction) and the unsampled bins keep the "
        "prediction. Read by the inference strategies, never by the training-time "
        "layer, so it is independent of ``enabled`` and ``method``, which govern what "
        "the model trains with. A sampler that already pins the measurement in its own "
        "loop (diffusion under ``method: hard``, cold diffusion given a mask, "
        "physics_driven) is not projected a second time; the run summary records who "
        "projected. The predict batch must carry the mask and the measurement: "
        "``spectramr infer`` reads the ``mask`` dataset of an HDF5 input and takes the "
        "k-space input as the measurement, and raises when either is missing rather "
        "than skipping. The measurement is used as acquired; ``eval_noise_level`` "
        "simulates acquisition noise on a synthesized reference at validation and is "
        "not added here. The projection domain is the model's registered "
        "``output_domain`` (``kspace``, or ``image`` with an odd channel count); an "
        "unannotated model raises. Not a generator kwarg, so deliberately absent from "
        "``spectramr.infrastructure.physics.dc_settings.DC_SSOT_KEYS``.",
    )
    weight: float = Field(
        default=1.0,
        ge=0,
        description="Trust placed in the measurement by the *soft* DC families. Not a "
        "loss weight, and not a universal blend coefficient: it becomes "
        "``lambda_init`` for SoftDataConsistency, ``beta`` for "
        "NoiseAdaptiveDataConsistency, ``hf_lambda`` for TargetAwareFSDC and "
        "``weight`` for the SimpleDataConsistency fallback. **Inert under "
        "``method: hard``** -- HardDataConsistency takes no weight parameter "
        "because its blend ``(1 - m) * recon + m * obs`` already IS weight 1.0 "
        "by construction, so declaring a weight there cannot mean anything "
        "short of turning hard DC into soft DC. See "
        "spectramr.infrastructure.physics.dc_settings.DCKnobReadership, which is "
        "the one place the per-method readership is declared (#1525).",
    )
    interval: int = Field(
        default=1,
        ge=1,
        description="Apply data consistency every N iterations",
    )
    method: str = Field(
        default="projection_2d_consistency",
        description="Method for data consistency: projection_2d_consistency, adaptive_soft_dc, etc.",
    )
    train_noise_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Acquisition-noise standard deviation added to the measurement "
        "before data consistency pins it, during training. Default 0.0: DC holds "
        "the measurement and nothing else. "
        "ABSOLUTE, not a percentage — the description used to read '0.01 (1%)' and "
        "that is the trap, because the k-space it perturbs is log1p-compressed. "
        "Measured on a percentile-normalised phantom at 256x256, 0.01 is 1.8% of "
        "the mean |k| in the DC annulus and 46.5% in the outer one, randomising "
        "observed outer-band phase by up to 23.9 degrees — on the very bins hard "
        "DC exists to hold, so the model cannot correct it. This default is what "
        "every arm gets: 0 of the corpus declares either level, and the resolver "
        "forwards the schema value regardless, so it reaches all 68 "
        "'dc_method: hard' arms. Declare a non-zero value to opt in. "
        "Read only by DC methods whose layer accepts a noise level ('hard' and the "
        "SimpleDataConsistency fallback); inert under soft/learned methods.",
    )
    eval_noise_level: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="As train_noise_level, at eval. Default 0.0: a non-zero value "
        "contradicts the 'sampler_sigma: 0.0' determinism every cohort arm "
        "declares (#1689), and the reverse loop's own hard-DC branch "
        "(_apply_observed_dc) never added noise — so a non-zero value here made "
        "training and the path that reports the numbers two different mechanisms.",
    )
    noise_type: str = Field(
        default="gaussian",
        description="Noise model for the simulated acquisition noise. Only "
        "'gaussian' (complex k-space) is implemented; the DC layer raises on "
        "anything else rather than degrading to Gaussian. 'rician' was "
        "advertised here for a long time and has never had an implementation "
        "in any DC layer -- see spectramr.infrastructure.physics.dc_settings."
        "SUPPORTED_NOISE_TYPES, which is the one place the shipped vocabulary "
        "is declared. Read by 'hard' DC and the SimpleDataConsistency fallback; "
        "the soft/learned methods build layers that take no noise parameters.",
    )


class CoilSensitivityConfig(BaseModel):
    """Coil sensitivity estimation configuration.

    Coil sensitivities describe the spatial sensitivity pattern of each
    receiver coil in multi-coil MRI systems. Can be estimated during training
    or loaded from pre-computed maps.

    Example:
        >>> coil_config = CoilSensitivityConfig(
        ...     enable_estimation=True,
        ...     estimation_method="espirit",
        ...     precomputed_path="/path/to/coil_maps.npy",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enable_estimation: bool = Field(
        default=False,
        description="Estimate coil sensitivities during training",
    )
    estimation_method: str = Field(
        default="espirit",
        description="Method for coil sensitivity estimation: espirit, walsh, sos",
    )
    estimation_type: str = Field(
        default="dynamic",
        description="Type of estimation: dynamic (per-volume), static (per-dataset), or hybrid",
    )
    calibration_region_size: int = Field(
        default=24,
        ge=4,
        description="Size of calibration region for sensitivity estimation",
    )
    num_threads: int = Field(
        default=4,
        ge=1,
        description="Number of threads for sensitivity computation",
    )
    precomputed_path: str | None = Field(
        default=None,
        description="Path to precomputed coil sensitivity maps (overrides estimation if provided)",
    )
    precomputed_type: str = Field(
        default="pt",
        description="File type of precomputed maps: pt (PyTorch tensor), npy, h5, nii, nii.gz",
    )


class KSpaceConfig(BaseModel):
    """K-space domain processing configuration.

    K-space is the Fourier domain of the MRI signal where acquisitions occur.
    Processing in k-space can enforce physics constraints directly.

    Example:
        >>> kspace_config = KSpaceConfig(
        ...     enable_kspace_recon=True,
        ...     enforce_hermitian_symmetry=True,
        ...     data_range=1.0,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enable_kspace_recon: bool = Field(
        default=False,
        description="Reconstruct in k-space domain",
    )
    enforce_hermitian_symmetry: bool = Field(
        default=True,
        description="Enforce Hermitian symmetry in k-space (for real-valued images)",
    )
    zero_pad_reconstruction: bool = Field(
        default=False,
        description="Zero-pad k-space before IFFT for higher resolution",
    )
    data_range: float = Field(
        default=1.0,
        gt=0.0,
        description="K-space data range for PSNR computation. Use 1.0 for ortho-normalized FFT, adjust if different normalization used.",
    )
    max_magnitude: float = Field(
        default=1.0,
        gt=0.0,
        description="Maximum expected k-space magnitude for metrics normalization. Used to track actual k-space scale.",
    )


class RegularizationConfig(BaseModel):
    """Spatial regularization configuration.

    Regularization constraints encourage smooth or sparse solutions
    in image or transform domains.

    Example:
        >>> reg_config = RegularizationConfig(
        ...     enable_tv=True,
        ...     tv_weight=0.01,
        ...     enable_wavelets=True,
        ...     wavelet_weight=0.01,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    # Total Variation (TV) regularization
    enable_tv: bool = Field(
        default=False,
        description="Enable total variation regularization",
    )
    tv_weight: float = Field(
        default=0.01,
        ge=0,
        description="Weight for TV regularization",
    )
    tv_type: str = Field(
        default="isotropic",
        description="TV type: isotropic or anisotropic",
    )

    # Wavelet regularization
    enable_wavelets: bool = Field(
        default=False,
        description="Use wavelet domain constraints",
    )
    wavelet_weight: float = Field(
        default=0.01,
        ge=0,
        description="Weight for wavelet constraints",
    )
    wavelet_family: str = Field(
        default="db4",
        description="Wavelet family: db4, db8, sym5, etc.",
    )
    wavelet_levels: int = Field(
        default=3,
        ge=1,
        description="Number of wavelet decomposition levels",
    )

    # Sparsity regularization
    enable_sparsity: bool = Field(
        default=False,
        description="Enable sparsity regularization",
    )
    sparsity_weight: float = Field(
        default=0.001,
        ge=0,
        description="Weight for sparsity regularization",
    )
    sparsity_type: str = Field(
        default="l1",
        description="Sparsity type: l1, l0, or log",
    )

    # Additional regularization fields (added for schema compliance)
    physics_penalty_weight: float = Field(
        default=0.0,
        ge=0,
        description="Weight for physics penalty regularization",
    )
    l2_penalty: float = Field(
        default=0.0,
        ge=0,
        description="Weight for L2 penalty regularization",
    )


class IterativeRefinementConfig(BaseModel):
    """Iterative refinement configuration.

    Iterative refinement applies the trained model multiple times
    with data consistency corrections between iterations.

    Example:
        >>> iter_config = IterativeRefinementConfig(
        ...     enabled=True,
        ...     num_steps=3,
        ...     dc_weight=1.0,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable iterative refinement during inference",
    )
    num_steps: int = Field(
        default=1,
        ge=1,
        description="Number of refinement iterations",
    )
    dc_weight: float = Field(
        default=1.0,
        ge=0,
        description="Weight for data consistency in refinement iterations",
    )
    residual_learning: bool = Field(
        default=False,
        description="Use residual learning (learn updates rather than direct reconstruction)",
    )


class PhaseCorrectionConfig(BaseModel):
    """Phase correction configuration.

    Phase errors can occur in MRI due to B0 inhomogeneity, motion, etc.
    Phase correction attempts to estimate and correct these errors.

    Example:
        >>> phase_config = PhaseCorrectionConfig(
        ...     enable_phase_estimation=True,
        ...     estimation_method="voxel_wise",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enable_phase_estimation: bool = Field(
        default=False,
        description="Estimate and correct phase errors",
    )
    estimation_method: str = Field(
        default="voxel_wise",
        description="Phase estimation method: voxel_wise, region_wise, or global",
    )
    phase_smoothing: bool = Field(
        default=True,
        description="Apply spatial smoothing to estimated phase",
    )


# Deprecated spelling (doubled ``n``). The YAML key is ``phase_correction`` (a
# field, unaffected by the class rename), so this alias is a pure import-compat
# shim for any external code referencing the old class name. Silent by design:
# a ``DeprecationWarning`` from ``spectramr.*`` is promoted to a test error, so it
# cannot be a warning. Remove 2026-08. WS-1 round-2 closure (2026-07-02).
PhaseCorrectionnConfig = PhaseCorrectionConfig


class B0CorrectionConfig(BaseModel):
    """B0 field inhomogeneity correction configuration.

    B0 inhomogeneity causes frequency shifts and image distortions
    that can be partially corrected.

    Example:
        >>> b0_config = B0CorrectionConfig(
        ...     enable_b0_correction=True,
        ...     num_shim_orders=2,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enable_b0_correction: bool = Field(
        default=False,
        description="Apply B0 field inhomogeneity correction",
    )
    num_shim_orders: int = Field(
        default=2,
        ge=0,
        le=3,
        description="Number of shim orders (0-3) for B0 correction",
    )
    estimate_from_data: bool = Field(
        default=True,
        description="Estimate B0 map from acquired data",
    )


class B0SimulationConfig(BaseModel):
    """B0 field simulation configuration for training augmentation.

    Simulates B0 inhomogeneity maps on-the-fly during training to make
    models robust to off-resonance effects (especially for Non-Cartesian).

    Example:
        >>> b0_sim = B0SimulationConfig(
        ...     enabled=True,
        ...     max_hz=100.0,
        ...     style="anatomical",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable B0 simulation during training",
    )
    max_hz: float | None = Field(
        default=None,
        ge=0.0,
        description="Maximum off-resonance in Hz. If None, uses field-strength default.",
    )
    style: str = Field(
        default="smooth",
        description="B0 map style: 'smooth' or 'anatomical' (with hotspots)",
    )
    probability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Probability of applying B0 simulation per batch",
    )


class MotionCorrectionConfig(BaseModel):
    """Motion correction configuration.

    Motion during acquisition causes artifacts that can sometimes be corrected.

    Example:
        >>> motion_config = MotionCorrectionConfig(
        ...     enable_motion_correction=True,
        ...     motion_type="translational",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enable_motion_correction: bool = Field(
        default=False,
        description="Enable motion artifact correction",
    )
    motion_type: str = Field(
        default="translational",
        description="Type of motion: translational, rigid, or affine",
    )
    reference_frame: str = Field(
        default="middle",
        description="Reference frame for motion estimation: first, middle, or mean",
    )
    method: str = Field(
        default="standard",
        description="Method for motion correction: standard, rigid_slice_to_volume, etc.",
    )


class CompressedSensingConfig(BaseModel):
    """Compressed sensing (CS) configuration.

    Compressed sensing exploits sparsity in image domains to reconstruct
    images from undersampled k-space measurements. Includes sampling patterns,
    sparsity transforms, and reconstruction algorithms.

    Example:
        >>> cs_config = CompressedSensingConfig(
        ...     enabled=True,
        ...     sampling_pattern="cartesian",
        ...     sparsity_transform="wavelet",
        ...     reconstruction_algorithm="iterative_soft_thresh",
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable compressed sensing constraints",
    )
    sampling_pattern: str = Field(
        default="cartesian",
        description="K-space sampling pattern: cartesian, radial, spiral, random",
    )
    acceleration_factor: float = Field(
        default=4.0,
        ge=1.0,
        description="Undersampling acceleration factor",
    )
    sparsity_transform: str = Field(
        default="wavelet",
        description="Sparsity domain: wavelet, dictionary, total_variation, fourier",
    )
    reconstruction_algorithm: str = Field(
        default="iterative_soft_thresh",
        description="Reconstruction method: iterative_soft_thresh, ista, fista, admm, prox_grad",
    )
    num_iterations: int = Field(
        default=10,
        ge=1,
        description="Number of CS reconstruction iterations",
    )
    threshold_scaling: float = Field(
        default=1.0,
        ge=0.0,
        description="Scaling factor for sparsity threshold (lambda parameter)",
    )
    convergence_tolerance: float = Field(
        default=1e-5,
        ge=0.0,
        description="Convergence tolerance for iterative algorithms",
    )
    step_size: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Step size for gradient-based optimization (0 < step_size <= 1)",
    )
    enforce_data_consistency: bool = Field(
        default=True,
        description="Enforce data consistency with acquired k-space in each iteration",
    )
    use_adaptive_threshold: bool = Field(
        default=False,
        description="Adapt threshold dynamically during reconstruction",
    )


class PINNConfig(BaseModel):
    """Physics-Informed Neural Network (PINN) configuration.

    PINNs incorporate PDE constraints into the loss function.

    Example:
        >>> pinn_config = PINNConfig(
        ...     enabled=True,
        ...     pde_type="wave_equation",
        ...     boundary_condition="dirichlet",
        ...     collocation_points=1000,
        ...     lambda_pde=0.1,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable PINN constraints",
    )
    pde_type: str = Field(
        default="wave_equation",
        description="Type of PDE: wave_equation, bloch_equation",
    )
    boundary_condition: str = Field(
        default="dirichlet",
        description="Boundary condition type: dirichlet, neumann, robin",
    )
    collocation_points: int = Field(
        default=2048,
        ge=0,
        description="Number of collocation points for PDE evaluation",
    )
    lambda_pde: float = Field(
        default=0.1,
        ge=0.0,
        description="Weight for PDE residual loss",
    )

    # ==================== PINN TRAINING HYPERPARAMETERS ====================
    warmup_epochs: int = Field(
        default=200,
        ge=0,
        description=(
            "Number of epochs for pure data-anchoring phase before enabling "
            "PDE loss (curriculum learning warmup)"
        ),
    )
    alpha_ema: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="EMA decay factor for adaptive gradient weighting (Wang et al. ICML 2021)",
    )
    map_save_interval: int = Field(
        default=5,
        ge=1,
        description="Save sensitivity map visualizations every N epochs",
    )


class BlochSimulationConfig(BaseModel):
    """Bloch simulation configuration.

    Simulates MRI signal evolution using Bloch equations.

    Example:
        >>> bloch_config = BlochSimulationConfig(
        ...     enabled=True,
        ...     target_tr=3000.0,
        ...     target_te=100.0,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable Bloch simulation",
    )
    target_tr: float = Field(
        default=3000.0,
        ge=0.0,
        description="Target Repetition Time (TR) in ms",
    )
    target_te: float = Field(
        default=100.0,
        ge=0.0,
        description="Target Echo Time (TE) in ms",
    )


class DigitalTwinConfig(BaseModel):
    """Digital Twin MRI Simulator configuration.

    Controls the virtual fiducial marker embedding and physics-based
    corruption pipeline used for training-time data augmentation.

    Example:
        >>> dt_config = DigitalTwinConfig(
        ...     enabled=True,
        ...     marker_type="cross",
        ...     motion_type="periodic",
        ...     enable_chemical_shift=True,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable the Digital Twin simulator during training",
    )

    # ── Progressive Diffusion Settings ──
    progressive_degradations: list[str] = Field(
        default_factory=lambda: ["motion", "noise", "b0", "b1"],
        description=(
            "Degradations applied progressively during diffusion. Two banks are "
            "accepted: the simulator's 14 NATIVE axes (motion, noise, b0, b1, "
            "undersampling, gibbs, chemical_shift, eddy_current, spike_noise, "
            "rf_zipper, rf_crosstalk, b0_distortion, gradient_nonlinearity, "
            "magic_angle), which modulate its built-in pipeline; and any of the "
            "31 keys of infrastructure.physics.digital_twin_extensions."
            "DEGRADATION_REGISTRY (rigid_motion, pulsatile_motion, rician, "
            "spike, nyquist_ghost, susceptibility, t2star_blur, partial_fourier, "
            "...), applied as standalone operators. A name in both banks is "
            "handled natively and never applied twice. Unknown names raise at "
            "simulator construction."
        ),
    )
    degradation_ranges: dict[str, tuple[float, float]] = Field(
        default_factory=lambda: {
            "motion": (0.0, 1.0),
            "noise": (0.0, 1.0),
            "b0": (0.0, 1.0),
            "b1": (0.0, 1.0),
            "rf_zipper": (0.0, 1.0),
            "gradient_nonlinearity": (0.0, 1.0),
        },
        description="Dictionary mapping degradation type to its (min, max) multiplier range",
    )

    # ── Severity ramp over diffusion timesteps ──
    # The counterpart of undersampling's acceleration_schedule. Until these
    # existed, degradation_ranges was remapped from the corruption factor
    # LINEARLY and there was no way to say otherwise -- the ramp shape was a
    # hardcoded assumption rather than a declared one.
    degradation_schedule: AccelerationSchedule = Field(
        default=AccelerationSchedule.LINEAR,
        description=(
            "Shape of the severity ramp over diffusion timesteps, applied to every "
            "axis unless overridden per-axis. Same vocabulary as "
            "undersampling.acceleration_schedule -- linear, polynomial, power_law, "
            "exponential, step -- because it is the same ramp, on severity instead "
            "of acceleration. 'linear' (the default) is the identity and reproduces "
            "the behaviour of every config written before this field existed. "
            "'step' is binary at the midpoint: vmin below, vmax above (undersampling "
            "indexes an explicit acceleration_range list there; severity has only "
            "the (vmin, vmax) pair, so only the binary reading has a meaning)."
        ),
    )
    degradation_schedules: dict[str, AccelerationSchedule] = Field(
        default_factory=dict,
        description=(
            "Per-axis override of degradation_schedule. Axes do not ramp alike: "
            "thermal noise grows smoothly with the corruption factor while motion "
            "is closer to all-or-nothing, and a single shape for both is a modelling "
            "assumption rather than a physical one. Keys must name a known axis "
            "(the 14 native plus the 31 DEGRADATION_REGISTRY operators) AND appear "
            "in progressive_degradations -- an axis absent from that list is held "
            "static at full strength, so a schedule on it could never fire."
        ),
    )
    degradation_schedule_power: float = Field(
        default=2.0,
        gt=0.0,
        description=(
            "Exponent for the 'power_law' / 'polynomial' severity schedules, "
            "mirroring undersampling.schedule_power. Read only by those two."
        ),
    )

    # ── Marker configuration ──
    @model_validator(mode="after")
    def _severity_schedules_can_fire(self) -> "DigitalTwinConfig":
        """A per-axis schedule that names an unknown or static axis is an unread knob.

        Both failures are silent otherwise: an unknown key is simply never looked
        up, and an axis absent from ``progressive_degradations`` is held at
        effective cf 1.0 by ``_get_effective_cf``, so its ramp shape is never
        consulted. Validate against ``known_degradation_axes()`` rather than a
        hand-written set, so a newly registered degradation is legal the day it
        lands (pitfall #15).
        """
        if self.ood_acceleration_range is not None:
            # Both conditions live on this block, so this validator is their one
            # owner; the witness ``ood_acceleration_range_is_read`` keeps only the
            # cross-object question (does the resolved strategy read the range).
            if not self.enabled:
                raise ValueError(
                    "physics.digital_twin.ood_acceleration_range is declared on a twin "
                    "whose `enabled` is false: nothing will re-score the rungs. Enable the "
                    "twin or delete the range."
                )
            if not self.enable_undersampling:
                raise ValueError(
                    "physics.digital_twin.ood_acceleration_range is declared but "
                    "physics.digital_twin.enable_undersampling is false, so no "
                    "in-distribution acceleration exists for the out-of-distribution rungs "
                    "to be out of. Declare enable_undersampling: true (with the training "
                    "acceleration) or delete the range."
                )
        if not self.degradation_schedules:
            return self
        from spectramr.infrastructure.physics.digital_twin_simulator import (
            known_degradation_axes,
        )

        known = known_degradation_axes()
        unknown = sorted(k for k in self.degradation_schedules if k not in known)
        if unknown:
            raise ValueError(
                f"degradation_schedules names unknown axes {unknown}. Valid: {sorted(known)}."
            )
        if self.progressive_degradations:
            static = sorted(
                k for k in self.degradation_schedules if k not in self.progressive_degradations
            )
            if static:
                raise ValueError(
                    f"degradation_schedules schedules {static}, which "
                    "progressive_degradations does not list. An axis absent from a "
                    "non-empty progressive_degradations is held STATIC at full "
                    "strength, so its schedule can never fire. Add it to "
                    "progressive_degradations, or drop the schedule."
                )
        return self

    marker_type: Literal["gaussian", "cross"] = Field(
        default="gaussian",
        description="Marker shape: 'gaussian' (smooth blobs) or 'cross' (sharp crosshairs)",
    )
    corner_offset: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Marker position as fraction from FOV centre (0.8 = near corners)",
    )
    marker_sigma: float = Field(
        default=0.03,
        gt=0.0,
        le=0.2,
        description="Gaussian marker width in normalised coordinates",
    )
    marker_arm_length: float = Field(
        default=0.06,
        gt=0.0,
        le=0.3,
        description="Cross marker arm half-length in normalised coordinates",
    )
    marker_arm_width: float = Field(
        default=0.006,
        gt=0.0,
        le=0.05,
        description="Cross marker arm half-width in normalised coordinates",
    )
    learnable_markers: bool = Field(
        default=False,
        description="Make marker template parameters trainable",
    )

    # ── Motion configuration ──
    motion_type: Literal["rigid", "periodic", "random_shot", "elastic"] = Field(
        default="rigid",
        description="Motion model: rigid (global shift), periodic (respiratory), "
        "random_shot (per-line jitter), elastic (non-rigid deformation)",
    )
    motion_composite: list[Literal["rigid", "periodic", "random_shot", "elastic"]] = Field(
        default_factory=list,
        description="Composite motion: when non-empty, these motion models are "
        "applied sequentially in k-space to build complex mixed patterns "
        "(e.g. ['rigid','periodic','random_shot']). Empty → single motion_type.",
    )
    apply_as_transform: bool = Field(
        default=False,
        description="Apply the digital twin as a transversal data-pipeline "
        "transform (DigitalTwinDegradation) for ANY experiment — corrupts the "
        "acquisition into 'input', keeps the clean 'target'. Default off. Leave "
        "off for VF strategies (they corrupt internally; enabling here would "
        "double-corrupt).",
    )
    transform_degradation_only: bool = Field(
        default=True,
        description="When apply_as_transform is on, skip VF marker embedding "
        "(markers are a VF concept). Off → embed markers in the data transform.",
    )
    motion_severity: float = Field(
        default=1.0,
        ge=0.0,
        description="Constant amplitude multiplier on motion "
        "(translation/rotation/amplitude, not frequency). A sustained floor, "
        "not a schedule: set >1 to keep the corrupted input far enough from "
        "the target to prevent identity collapse. Do NOT anneal this toward a "
        "mild regime — a near-clean input at the end of training leaks the "
        "target.",
    )
    enable_motion: bool = Field(
        default=False,
        # Default False since 2026-08-03: this sub-flag sits under a parent whose
        # own `enabled` defaults False, so a True default meant enabling the twin
        # bought this corruption unasked. The 9 arms that were relying on the old
        # default now declare it explicitly, so nothing changed for them.
        description="Enable motion corruption",
    )
    max_translation: float = Field(
        default=5.0,
        ge=0.0,
        le=30.0,
        description="Max rigid translation in pixels",
    )
    max_rotation: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Max rigid rotation in radians",
    )
    motion_amplitude: float = Field(
        default=3.0,
        ge=0.0,
        le=20.0,
        description="Peak displacement for periodic/random_shot/elastic motion (pixels)",
    )
    motion_frequency: float = Field(
        default=2.0,
        ge=0.1,
        le=10.0,
        description="Number of motion cycles during acquisition (periodic only)",
    )
    elastic_modes: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of random Fourier modes for elastic deformation",
    )

    # ── SNR / Noise ──
    snr_min: float = Field(
        default=10.0,
        ge=1.0,
        description="Minimum target SNR in dB",
    )
    snr_max: float = Field(
        default=25.0,
        ge=1.0,
        description="Maximum target SNR in dB",
    )

    # ── B0 / B1 ──
    enable_b0: bool = Field(
        default=False,
        # Default False since 2026-08-03: this sub-flag sits under a parent whose
        # own `enabled` defaults False, so a True default meant enabling the twin
        # bought this corruption unasked. The 9 arms that were relying on the old
        # default now declare it explicitly, so nothing changed for them.
        description="Enable B0 inhomogeneity (spatially-varying phase distortion)",
    )
    b0_strength: float = Field(
        default=0.3,
        ge=0.0,
        le=2.0,
        description="B0 field inhomogeneity strength (0=none, 1=severe)",
    )
    enable_b1: bool = Field(
        default=False,
        # Default False since 2026-08-03: this sub-flag sits under a parent whose
        # own `enabled` defaults False, so a True default meant enabling the twin
        # bought this corruption unasked. The 9 arms that were relying on the old
        # default now declare it explicitly, so nothing changed for them.
        description="Enable B1 receive field non-uniformity (intensity bias)",
    )
    b1_strength: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="B1 bias field strength (0=none, 1=severe)",
    )

    # ── Undersampling / Gibbs ──
    enable_undersampling: bool = Field(
        default=False,
        description="Enable Cartesian k-space undersampling (aliasing)",
    )
    acceleration: float = Field(
        default=4.0,
        ge=1.0,
        le=16.0,
        description="Undersampling acceleration factor (1=fully sampled)",
    )
    ood_acceleration_range: list[float] | None = Field(
        default=None,
        description=(
            "Out-of-distribution accelerations the twin-driven validation step "
            "re-scores at, each above 1.0 and strictly ascending, e.g. [16.0, 32.0]. "
            "For every entry R the twin undersamples at R instead of `acceleration`, "
            "the model is run again and the scores are written as "
            "`val_ood_{R}x_<metric>` beside the in-distribution `val_<metric>`; "
            "`val_ood_accelerations` counts the entries (0 when absent). Read only "
            "by the strategies that declare `reads_ood_acceleration_range` "
            "(virtual_fiducial, vf_admm and their subclasses), and only when "
            "`enable_undersampling` is true: a range on a twin that does not "
            "undersample or is disabled fails at load (this schema's validator), and "
            "one on a strategy that does not read it fails the witness "
            "`ood_acceleration_range_is_read`. Each entry adds one twin+model pass per "
            "validation batch (cohort review 2026-09-03, VF)."
        ),
    )

    @field_validator("ood_acceleration_range")
    @classmethod
    def _ood_range_is_ascending_and_above_one(cls, value: list[float] | None) -> list[float] | None:
        """Each entry above 1.0 and strictly ascending; an empty list is a typo."""
        if value is None:
            return None
        if len(value) == 0:
            raise ValueError(
                "physics.digital_twin.ood_acceleration_range is empty; delete the key "
                "or list the accelerations to re-score at, e.g. [16.0, 32.0]."
            )
        for index, entry in enumerate(value):
            if entry <= 1.0:
                raise ValueError(
                    f"physics.digital_twin.ood_acceleration_range[{index}]={entry} is not "
                    "above 1.0; an out-of-distribution rung is an acceleration."
                )
            if index and entry <= value[index - 1]:
                raise ValueError(
                    "physics.digital_twin.ood_acceleration_range must be strictly "
                    f"ascending; entry {index} ({entry}) does not exceed {value[index - 1]}."
                )
        return [float(entry) for entry in value]

    enable_gibbs: bool = Field(
        default=False,
        description="Enable Gibbs ringing (k-space truncation)",
    )
    gibbs_truncation: float = Field(
        default=0.85,
        ge=0.3,
        le=1.0,
        description="Fraction of k-space to retain (1.0=no Gibbs)",
    )

    # ── Chemical shift ──
    enable_chemical_shift: bool = Field(
        default=False,
        description="Enable fat-water chemical shift displacement",
    )
    chemical_shift_pixels: float = Field(
        default=3.0,
        ge=0.0,
        le=15.0,
        description="Chemical shift displacement in pixels along readout",
    )

    # ── Eddy currents ──
    enable_eddy_current: bool = Field(
        default=False,
        description="Enable gradient-induced eddy current artefacts",
    )
    eddy_strength: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Eddy current phase accumulation strength",
    )

    # ── Spike noise ──
    enable_spike_noise: bool = Field(
        default=False,
        description="Enable RF spike noise (zipper artefact)",
    )
    spike_probability: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Per-element probability of a k-space spike",
    )

    # ── RF crosstalk ──
    enable_rf_crosstalk: bool = Field(
        default=False,
        description="Enable multi-slice RF crosstalk signal bleeding",
    )
    crosstalk_fraction: float = Field(
        default=0.05,
        ge=0.0,
        le=0.3,
        description="Fraction of blurred neighbouring-slice signal to add",
    )

    # ── B0 geometric distortion (voxel shifting) ──
    enable_b0_distortion: bool = Field(
        default=False,
        description="Enable B0-induced geometric voxel shifting along FE axis",
    )
    b0_dwell_time: float = Field(
        default=1e-5,
        gt=0.0,
        le=1e-3,
        description="ADC dwell time in seconds (controls pixel bandwidth)",
    )

    # ── Gradient non-linearity ──
    enable_gradient_nonlinearity: bool = Field(
        default=False,
        description="Enable gradient non-linearity (barrel/pincushion distortion)",
    )
    gradient_nonlinearity_severity: float = Field(
        default=2e-6,
        ge=0.0,
        le=1e-3,
        description="Gradient non-linearity distortion coefficient α",
    )

    # ── RF zipper ──
    enable_rf_zipper: bool = Field(
        default=False,
        description="Enable RF zipper artifact (coherent CW leak on single FE column)",
    )
    rf_zipper_location: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Normalised FE position of the RF leak (0–1)",
    )
    rf_zipper_intensity: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Zipper amplitude relative to peak k-space magnitude",
    )

    # ── Magic angle effect ──
    enable_magic_angle: bool = Field(
        default=False,
        description="Enable magic angle T2 modulation for ordered structures at 54.7°",
    )
    magic_angle_te: float = Field(
        default=20.0,
        ge=0.0,
        description="Echo time for magic angle T2 computation (ms)",
    )
    magic_angle_t2_base: float = Field(
        default=10.0,
        gt=0.0,
        description="Baseline T2 for ordered collagen (ms)",
    )
    magic_angle_t2_magic: float = Field(
        default=30.0,
        gt=0.0,
        description="Maximum T2 at the magic angle (ms)",
    )
    magic_angle_collagen_fraction: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Approximate fraction of FOV containing collagen-like structures",
    )


# ---------------------------------------------------------------------------
# v6.1 additions (defined before PhysicsConfigSchema so they're in scope
# for default_factory). Each block defaults to v6.0-equivalent semantics
# (disabled / None) — config_version: '6.0' files load unchanged.
# ---------------------------------------------------------------------------


class ConcomitantFieldConfig(BaseModel):
    """Concomitant-field correction at low B0 (idea 1: CAUR).

    Maxwell concomitant fields scale ~ G²/(2 B0) and become non-negligible
    at ULF. ``enabled=False`` preserves v6.0 semantics (no correction).
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Apply concomitant-field phase correction during reconstruction.",
    )
    max_gradient_T_per_m: float = Field(
        default=0.040,
        gt=0.0,
        description="Peak gradient amplitude (T/m) for the Maxwell-term scaling.",
    )
    waveform_uri: str | None = Field(
        default=None,
        description="URI of an explicit gradient waveform. None → trapezoid assumed.",
    )


class RelaxationPriorsConfig(BaseModel):
    """Tissue-class T1/T2 priors for the Bloch-constrained translator (idea 3)."""

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(default=False)
    t1_table_uri: str | None = Field(
        default=None,
        description="CSV/JSON of (tissue_class, T1_ms) per field-strength.",
    )
    t2_table_uri: str | None = Field(
        default=None,
        description="CSV/JSON of (tissue_class, T2_ms) per field-strength.",
    )


class CoilCompressionConfig(BaseModel):
    """Coil-compression sub-block of ``physics.coil_processing``.

    ``svd`` selects the SCC primitive in
    ``infrastructure/physics/coil_compression.py``; ``gcc`` validates at the
    schema level but the pipeline raises ``NotImplementedError`` when selected
    (the enum value is kept so configs can reference it once implemented).
    Unknown methods are rejected at config load (pitfall #9).
    """

    model_config = {"extra": "forbid", "frozen": True}
    method: Literal["none", "svd", "gcc"] = "none"
    num_virtual_coils: int = Field(default=4, ge=1)
    # None = full FoV (the parity default — the SVD basis is estimated from the
    # whole k-space, matching pre-pipeline behavior). Set a value to restrict the
    # basis to a centered ACS band. Must stay None-default so the schema agrees
    # with the runtime: the loader sync only propagates calibration_lines when the
    # user sets it explicitly, so a non-None schema default would advertise an
    # ACS crop that the runtime never applies (CLAUDE.md non-negotiable #8).
    calibration_lines: int | None = Field(default=None, ge=1)


class CoilEstimationConfig(BaseModel):
    """Sensitivity-map estimation sub-block of ``physics.coil_processing``.

    ``method`` dispatches the ``estimate_smaps`` family in
    ``infrastructure/physics/coil_sensitivity.py``. Unknown methods are rejected
    at config load (pitfall #9).
    """

    model_config = {"extra": "forbid", "frozen": True}
    method: Literal["none", "power_iter", "espirit", "pinn", "rss", "file"] = "power_iter"
    enabled: bool = True
    kernel_size: int = Field(default=6, ge=1)
    acs_size: int = Field(default=24, ge=1)
    eigen_threshold: float = Field(default=0.95, gt=0, le=1)
    maps_path: str | None = None


class CoilCombineConfig(BaseModel):
    """Coil-combine sub-block of ``physics.coil_processing``.

    ``none`` keeps the (optionally compressed) coils separate; ``rss`` collapses
    them to one image by root-sum-of-squares; ``sense`` collapses via the SENSE
    adjoint (requires estimation enabled — smaps). ``none`` is the default so a
    bare ``coil_processing`` block is a no-op (keeps complex multi-coil data).
    Unknown methods are rejected at config load (pitfall #9).
    """

    model_config = {"extra": "forbid", "frozen": True}
    method: Literal["none", "rss", "sense"] = "none"


class CoilOutputConfig(BaseModel):
    """Output-representation sub-block of ``physics.coil_processing``.

    The domain and channel format the model receives — the axis that lets the
    unified block express the legacy ``data.coil_processing_mode`` data-load
    formats (``rss_image`` / ``magnitude`` / ``flatten``):

    * ``domain``: ``kspace`` (default) | ``image`` | ``source`` (= the input's own
      domain, no transform — used by the legacy ``magnitude`` mode).
    * ``channels``: ``complex`` (default, keep complex coils) | ``real_interleaved``
      (split each complex coil into [real, imag] channels) | ``magnitude`` (a single
      real magnitude image — requires a combine).

    Unknown values are rejected at config load (pitfall #9). Cross-axis
    consistency (e.g. ``magnitude`` requires a combine) is enforced by the audit
    ``check_coil_processing_consistency``.
    """

    model_config = {"extra": "forbid", "frozen": True}
    domain: Literal["kspace", "image", "source"] = "kspace"
    channels: Literal["complex", "real_interleaved", "magnitude"] = "complex"


class CoilProcessingConfig(BaseModel):
    """Config-driven SSOT for the coil-processing pipeline.

    Bundles ``compression`` → ``estimation`` → ``combine`` → ``output`` so users
    select coil compression, sensitivity estimation, coil-combine, and the output
    representation from one nested block (``physics.coil_processing``). The
    defaults make a bare block a no-op (no compression, no combine, complex
    k-space output) — i.e. the legacy ``coil_processing_mode: none`` behavior.
    """

    model_config = {"extra": "forbid", "frozen": True}
    compression: CoilCompressionConfig = Field(default_factory=CoilCompressionConfig)
    estimation: CoilEstimationConfig = Field(default_factory=CoilEstimationConfig)
    combine: CoilCombineConfig = Field(default_factory=CoilCombineConfig)
    output: CoilOutputConfig = Field(default_factory=CoilOutputConfig)


class SubvoxelRegistrationConfig(BaseModel):
    """Marker-anchored sub-pixel shift recovery for ``subvoxel_sr``.

    The simulator dithers one slice by N random sub-pixel offsets and pools it
    down; the model fuses the frames back to the HR grid. Whether the model is
    TOLD those offsets is the scientific variable, and this block selects it:

    ``blind``
        The offsets are withheld. The backbone must register implicitly. This is
        what exp_vf_01 ran before 2026-07-26, without saying so.
    ``recovered``
        A virtual fiducial (a known, jittered Gaussian probe) is carried through
        the SAME offsets and the same pooling as the anatomy. Phase correlation
        against the un-shifted fiducial recovers each frame's ABSOLUTE offset,
        which is then handed to the model. This is the arm family's namesake
        mechanism, and the only rung where a marker does any work.
    ``oracle``
        The simulator's ground-truth offsets are handed over directly. Not a
        method: the ceiling that upper-bounds ``recovered``.

    Running all three at fixed architecture and fixed parameter count separates
    "does sub-pixel dither carry information" from "can the model exploit it
    unaided", which a single blind arm conflates. See
    ``docs/digital_twin_review_followups.rst``.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    shift_source: Literal["blind", "recovered", "oracle"] = Field(
        default="blind",
        description=(
            "Which rung of the shift-knowledge ladder this arm sits on. Unknown "
            "values are rejected at load (no silent fallback). Default 'blind' "
            "keeps pre-2026-07-26 arms behaviourally unchanged."
        ),
    )
    marker_grid_spacing: int = Field(
        default=16,
        ge=4,
        description="Nominal pixel spacing of the fiducial's Gaussian peaks.",
    )
    marker_sigma: float = Field(
        default=2.0, gt=0.0, description="Width of each fiducial peak, in pixels."
    )
    marker_jitter: float = Field(
        default=0.35,
        ge=0.0,
        description=(
            "Per-peak displacement in units of grid_spacing, breaking the "
            "lattice periodicity. MUST be > 0 for shift_source='recovered': a "
            "periodic lattice has a comb spectrum that phase correlation cannot "
            "read (0.27% of bins populated, 0.55 px error) and is ambiguous "
            "modulo one period; 0.35 gives 23% occupancy and 0.003 px."
        ),
    )
    marker_seed: int = Field(
        default=0,
        description=(
            "Seeds the jitter draw. The fiducial is an ABSOLUTE reference, so it "
            "must be byte-identical across runs, processes and nodes."
        ),
    )
    max_shift_px: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Half-width of the uniform dither, in HR pixels. At sr_scale=2 the "
            "default spans +/-0.5 LR pixel, i.e. the full sub-pixel range."
        ),
    )

    # ---- Physical-units mode (mm), for real anisotropic acquisitions --------
    # The pixel knobs above are correct for the synthetic single-slice M4Raw arm,
    # where the grid IS the resolution and voxels are isotropic. They are wrong
    # for real ULF->HF data on two counts, so a millimetre mode exists alongside.
    voxel_mm: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Spacing of the STORED grid, per axis, in mm. Required whenever "
            "effective_voxel_mm is set, and DISTINCT from it: the volume is "
            "resampled onto the 3T grid (0.22-0.49 mm) while the 64mT scanner "
            "resolved 1.6-1.7 mm. Collapsing the two makes sigma_px = kappa, "
            "i.e. a one-pixel marker on a grid 3-7x finer than the acquisition "
            "-- precisely the sub-resolution marker the physical-units mode "
            "exists to prevent. Read it from the manifest's metadata.hf_voxel_mm "
            "(the grid both volumes are resampled onto)."
        ),
    )
    effective_voxel_mm: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Resolution the marker must remain VISIBLE at, per axis, in mm. "
            "Setting this switches the fiducial to physical units. It is not the "
            "sampling grid: preprocess_ulf_paired.py resamples the 64mT volume "
            "onto the 3T grid, so the stored spacing is 0.22-0.49 mm while the "
            "64mT scanner resolved 1.6-1.7 mm in-plane. Sizing the marker against "
            "the grid makes it 3-7x too fine to survive in the ULF channel, i.e. "
            "sub-resolution and invisible exactly where it has to work. Read it "
            "from the manifest's metadata.ulf_voxel_mm."
        ),
    )
    marker_sigma_mm: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Per-axis marker width in mm. Defaults to kappa * effective_voxel_mm. "
            "Anisotropic by necessity: the ULF protocol is 1.6 mm in-plane and "
            "5.0 mm through-plane, so a scalar width is wrong by ~3x on one axis "
            "whatever value it takes."
        ),
    )
    marker_spacing_mm: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Per-axis spacing between marker peaks in mm. Defaults to 8*sigma_mm. "
            "Must exceed 2*sigma_mm or peaks merge into a smooth field with no "
            "localisable structure (enforced in VirtualFiducial)."
        ),
    )
    marker_kappa: float = Field(
        default=1.0,
        ge=1.0,
        description=(
            "sigma_mm = kappa * effective_voxel_mm when sigma is not given. The "
            "Cramer-Rao bound for a translation estimate scales as the inverse of "
            "the marker's mean-square frequency, so narrower atoms register better "
            "right up to the point where energy crosses Nyquist and the sampled "
            "marker stops being a translate of a fixed template. kappa=1 sits at "
            "that boundary; below 1.0 the marker aliases, hence the floor."
        ),
    )
    # NOTE: no `max_shift_mm`. The dither half-width is consumed by the
    # simulator as a single scalar pixel width, so a millimetre value could not
    # be honoured per-axis on anisotropic data without also reshaping that draw.
    # Adding the knob here would mean advertising a unit the reader cannot
    # actually apply (CLAUDE.md #15). It belongs with the arm that reshapes the
    # draw, not with the fiducial geometry.

    @model_validator(mode="after")
    def _physical_units_are_self_consistent(self) -> "SubvoxelRegistrationConfig":
        """One unit system per arm, chosen explicitly rather than by precedence.

        Accepting both and silently preferring one is how a config ends up
        meaning something other than it says (CLAUDE.md #9).
        """
        physical = self.effective_voxel_mm is not None
        derived = (self.marker_sigma_mm, self.marker_spacing_mm)
        if not physical and any(d is not None for d in derived):
            raise ValueError(
                "marker_sigma_mm / marker_spacing_mm are millimetre "
                "knobs but effective_voxel_mm is unset, so there is no voxel size "
                "to convert them against. Declare effective_voxel_mm (from the "
                "manifest's metadata.ulf_voxel_mm) or use the pixel knobs."
            )
        for name in (
            "voxel_mm",
            "effective_voxel_mm",
            "marker_sigma_mm",
            "marker_spacing_mm",
        ):
            val = getattr(self, name)
            if val is not None and (len(val) not in (2, 3) or any(v <= 0 for v in val)):
                raise ValueError(f"{name}={val} must be a positive 2- or 3-tuple (per axis).")
        if physical and self.voxel_mm is None:
            raise ValueError(
                "effective_voxel_mm is set but voxel_mm (the STORED grid) is "
                "not. They are different numbers on resampled ULF data -- grid "
                "0.22-0.49 mm against 1.6-1.7 mm resolved -- and the marker "
                "width in pixels is their RATIO. Defaulting voxel_mm to "
                "effective_voxel_mm would silently yield sigma_px = kappa, a "
                "one-pixel marker on a grid 3-7x finer than the acquisition, "
                "which is the sub-resolution failure this mode exists to "
                "prevent. Read voxel_mm from the manifest's metadata.hf_voxel_mm."
            )
        if physical and self.voxel_mm is not None:
            if len(self.voxel_mm) != len(self.effective_voxel_mm or ()):
                raise ValueError(
                    f"voxel_mm={self.voxel_mm} and "
                    f"effective_voxel_mm={self.effective_voxel_mm} must have the "
                    "same number of axes."
                )
            bad = [
                (g, e) for g, e in zip(self.voxel_mm, self.effective_voxel_mm, strict=True) if e < g
            ]
            if bad:
                raise ValueError(
                    f"effective_voxel_mm={self.effective_voxel_mm} is FINER than "
                    f"the stored grid voxel_mm={self.voxel_mm} on {len(bad)} "
                    "axis/axes. An acquisition cannot resolve more than the grid "
                    "it is stored on; the two are most likely swapped."
                )
        return self

    # NOTE: there is deliberately no `lambda_shift` here. A supervision term on
    # the recovered shifts would have IDENTICALLY ZERO gradient: phase
    # correlation carries no learnable parameters and the fiducial frames come
    # from the simulator, so MSE(recovered, true) cannot move a single weight.
    # Declaring it would be a loss the run reports and never optimises —
    # CLAUDE.md #16 exactly. Registration accuracy is instead reported as the
    # `shift_mae_px` / `val_shift_mae_px` DIAGNOSTIC, which is what makes a null
    # result attributable: a flat PSNR with sub-0.01 px registration error means
    # the dither carries no information, whereas a flat PSNR with 0.5 px error
    # means the front end failed.


class AcquisitionParamsConfig(BaseModel):
    """One declared SPGR acquisition (the source or the target field)."""

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    field_strength_t: float = Field(gt=0.0, description="Static field, tesla.")
    tr_ms: float = Field(gt=0.0, description="Repetition time, milliseconds.")
    te_ms: float = Field(gt=0.0, description="Echo time, milliseconds.")
    flip_deg: float = Field(gt=0.0, le=180.0, description="Flip angle, degrees.")


class RelaxometricCalibrationConfig(BaseModel):
    """Hand the network the contrast half of ULF->HF, measured on the fiducial.

    A field translation changes voxel intensities for two reasons at once:
    STRUCTURE (the low-field acquisition resolved less) and CONTRAST (T1 depends
    on B0, so the same sequence orders tissues differently at 64 mT than at 3 T).
    Trained end to end, a network learns one intensity map that entangles both,
    and nothing in the output says whether the contrast half is physically right.

    The fiducial is a synthetic phantom, so its relaxation times are STIPULATED
    rather than estimated, and the steady-state SPGR signal it produces at each
    field is a closed form. Their ratio ``kappa`` is a known constant. The
    network is then constrained to ``y = kappa * g(x)``: it is handed the
    contrast transfer and left only the structural problem.

    The factorisation is identified up to a global scalar. With ``kappa`` fixed
    an intensity error must appear in ``g`` rather than masquerading as
    contrast, and the marker gives a direct read on whether ``kappa`` was right,
    because on marker support the true ratio is measurable from the data.

    SCOPE, measured not assumed: over 64 mT to 3 T at these sequences the
    transfer is dominated by the CSF-to-parenchyma contrast (kappa_CSF /
    kappa_WM is about 2.1) and is only a few percent between white and grey
    matter (1.02 at short TR, at best 1.08 near TR 500 ms / 90 degrees). An arm
    claiming a large contrast-transfer benefit on parenchyma alone is claiming
    more than the physics here supplies.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description=(
            "Apply the relaxometric contrast transfer when synthesising the "
            "low-field frames, and report the fiducial's predicted vs measured "
            "gain. Off by default, so existing arms are unchanged."
        ),
    )
    factored: bool = Field(
        default=False,
        description=(
            "Constrain the model output to kappa * g(x) with kappa supplied by "
            "the declared relaxometry. False is the UNFACTORED control: the "
            "same simulated contrast change, learned end to end. The two differ "
            "by exactly this knob, so the benefit of handing over the contrast "
            "is attributable (pitfall #17)."
        ),
    )
    source: "AcquisitionParamsConfig | None" = Field(
        default=None, description="The acquisition being translated FROM (ULF)."
    )
    target: "AcquisitionParamsConfig | None" = Field(
        default=None, description="The acquisition being translated TO (HF)."
    )
    marker_t1_ms: float = Field(
        default=500.0,
        gt=0.0,
        description=(
            "The fiducial's STIPULATED T1 at the SOURCE field. A physical "
            "marker would be a doped-water phantom with a measured value; here "
            "it is declared, which is exactly why kappa is known rather than "
            "estimated."
        ),
    )
    marker_t1_target_ms: float | None = Field(
        default=None,
        description=(
            "The fiducial's T1 at the TARGET field. Defaults to marker_t1_ms, "
            "i.e. a field-independent marker, which makes kappa a pure sequence "
            "ratio. Declare it to model a marker whose own relaxometry "
            "disperses with field, as real doped water does."
        ),
    )
    marker_t2_ms: float = Field(
        default=80.0,
        gt=0.0,
        description=(
            "The fiducial's T2, taken field-independent over this range (the "
            "standard approximation, stated rather than buried)."
        ),
    )

    @model_validator(mode="after")
    def _enabled_needs_both_acquisitions(self) -> "RelaxometricCalibrationConfig":
        if not self.enabled:
            return self
        if self.source is None or self.target is None:
            raise ValueError(
                "relaxometric_calibration.enabled=True requires BOTH `source` "
                "and `target` acquisition blocks. kappa is a ratio between two "
                "declared acquisitions; with one missing there is nothing to "
                "transfer between and the factored form would silently become "
                "the identity."
            )
        if self.source.field_strength_t == self.target.field_strength_t:
            raise ValueError(
                f"source and target are both at "
                f"{self.source.field_strength_t} T, so T1 is identical on both "
                "sides and kappa collapses to a pure sequence ratio. That is "
                "not a field translation; the mechanism would be inert."
            )
        return self

    @model_validator(mode="after")
    def _factored_requires_enabled(self) -> "RelaxometricCalibrationConfig":
        if self.factored and not self.enabled:
            raise ValueError(
                "relaxometric_calibration.factored=True with enabled=False: the "
                "model would be scaled by a kappa the simulator never applied."
            )
        return self


class ForwardPsfConfig(BaseModel):
    """Measure the acquisition's blur from the fiducial instead of assuming it.

    Every super-resolution arm assumes a forward operator, and the blur half of
    it is normally an assumed Gaussian. If that assumption is wrong the network
    spends capacity correcting a model error nobody told it about, and the
    reported degradation and the real one quietly differ.

    The fiducial's un-blurred form is known exactly, so observing what the
    acquisition does to it turns a blind deconvolution into a linear solve. On a
    coarse control grid it recovers the SPATIAL VARIATION too -- and a real
    low-field PSF is neither Gaussian nor uniform.

    Identifiability is computable rather than hoped for: the kernel is only
    recoverable where the marker has spectral energy. A marker at sigma=1.5 px
    excites 40% of the band, one at sigma=0.8 excites 94%, so a narrower marker
    buys a better-determined operator -- reported per run as psf_identifiability
    so a kernel estimated over mostly-interpolated frequencies is visible.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description=(
            "Apply a declared blur before pooling, and measure it back from the "
            "fiducial. Off by default: existing arms pool with no explicit PSF."
        ),
    )
    source: Literal["assumed", "measured"] = Field(
        default="assumed",
        description=(
            "Which kernel the model is CONDITIONED on. 'assumed' hands it the "
            "declared Gaussian; 'measured' hands it the one estimated from the "
            "fiducial. The pair differ by this field alone, so the value of "
            "measuring rather than assuming is attributable (#17). Unknown "
            "values are rejected at load."
        ),
    )
    sigma_px: float = Field(
        default=1.5,
        gt=0.0,
        description=(
            "Width of the ASSUMED Gaussian, in grid pixels. Also the width the "
            "simulator applies when true_sigma_px is unset, in which case the "
            "assumption is correct by construction and the arms should tie -- "
            "which makes that configuration a useful null control."
        ),
    )
    true_sigma_px: float | None = Field(
        default=None,
        description=(
            "Width the SIMULATOR actually applies, when it differs from the "
            "assumption. Setting it creates a genuine model error for the "
            "measured arm to correct; leaving it unset makes the assumption "
            "exact and the comparison a null control."
        ),
    )
    kernel_size: int = Field(
        default=9,
        ge=3,
        description="Odd spatial extent of the estimated kernel.",
    )
    mu: float = Field(
        default=1e-3,
        gt=0.0,
        description=(
            "Ridge weight on the Fourier solve. Swept empirically: 1e-3 "
            "recovers a known FWHM to within 0.6-1.1% across marker widths, "
            "while 1e-2 over-smooths (-6%) and 1e-6 amplifies the marker's "
            "spectral nulls (+4 to +13%). Must be > 0 -- at a null the kernel "
            "is genuinely unidentifiable and an unregularised solve returns "
            "noise scaled by 1/0."
        ),
    )
    control_rows: int = Field(
        default=1, ge=1, description="Control-grid rows for a varying estimate."
    )
    control_cols: int = Field(default=1, ge=1, description="Control-grid columns.")

    @model_validator(mode="after")
    def _kernel_size_is_odd(self) -> "ForwardPsfConfig":
        if self.kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be odd so the kernel has a centre, got {self.kernel_size}"
            )
        return self


class AnchorConformalConfig(BaseModel):
    """Per-voxel prediction intervals calibrated on the fiducial.

    A point estimate cannot say whether a structure is trustworthy; an interval
    can, but only if it carries a coverage guarantee. Split conformal supplies
    one, and needs a calibration set whose residuals are exchangeable with the
    test residuals -- i.e. held-out ground truth, which ULF-to-HF translation
    does not have at inference time.

    The fiducial supplies it: its true value is known everywhere, so residuals
    on marker support are computable with NO ground truth on the anatomy.

    That is an assumption, not a free lunch: a Gaussian probe is not a brain.
    Conditioning on a difficulty score narrows the assumption and makes it
    TESTABLE -- calibrate per stratum on the marker, then measure per-stratum
    coverage against the anatomy, where the digital twin does supply truth. A
    stratum below nominal means exchangeability failed there and the interval
    carries no guarantee. That outcome is reported, not patched around by
    loosening the stratification.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description=(
            "Calibrate on fiducial residuals at validation and report "
            "per-stratum coverage against the anatomy. Reporting only; nothing "
            "here enters the objective."
        ),
    )
    alpha: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Miscoverage level; 0.1 targets 90% coverage.",
    )
    n_strata: int = Field(
        default=4,
        ge=1,
        description=(
            "Difficulty bins. n_strata=1 is MARGINAL calibration, and it is not "
            "equivalent: on heteroscedastic residuals a single quantile "
            "over-covers the easy regime and under-covers the hard one (0.999 "
            "against 0.715 at a nominal 0.9, measured). More strata sharpen the "
            "conditional guarantee and shrink each calibration set; an empty "
            "stratum raises rather than borrowing another's quantile."
        ),
    )
    tolerance: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Absolute slack below nominal before a stratum is failed. Widening "
            "this until the gate passes defeats the gate."
        ),
    )


class BandProbeConfig(BaseModel):
    """Super-Nyquist probe: certify recovered detail against fabricated detail.

    A super-resolution network is graded on PSNR/SSIM, which are dominated by
    the low frequencies every method already gets right. Neither number can
    distinguish detail *unfolded* from inter-frame aliasing from detail
    *invented* by a prior, because both look sharp and plausible.

    The virtual fiducial makes the distinction measurable. It is an exogenous
    instrument: its high-frequency structure is fixed, anatomy-independent and
    known exactly at every frequency, and it passes through the SAME dither and
    the same pooling as the anatomy. Feeding the marker frames through the
    network and partitioning the result at the acquisition Nyquist
    (:mod:`spectramr.infrastructure.physics.band_partition`) gives a per-band
    transfer gain ``rho_l``. Above Nyquist nothing was measured, so ``rho_l``
    there is exactly the fraction of unmeasured detail the network recovers.

    Two knobs, deliberately separate:

    ``enabled``
        Compute and REPORT the spectrum. Always safe; costs one extra forward
        pass on the marker stack.
    ``lambda_band``
        Also OPTIMISE it. ``0.0`` leaves the probe purely diagnostic, which is
        the control arm: both arms report the same spectrum, so the cost of the
        constraint in task PSNR is attributable to exactly one knob (#17).

    Unlike a shift-supervision term (see :class:`SubvoxelRegistrationConfig`),
    this term has a real gradient: it constrains the network's OUTPUT, not the
    parameterless phase correlation that reads the offsets.
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description=(
            "Run the marker through the network and report the per-band "
            "transfer gain. Requires subvoxel_registration.shift_source="
            "'recovered' -- the marker only exists on that rung."
        ),
    )
    lambda_band: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the super-Nyquist probe term, sum_l (1 - rho_l) over the "
            "bands above the acquisition Nyquist. 0.0 = report only (the "
            "control arm). Sub-Nyquist bands are excluded from the LOSS on "
            "purpose: they are directly measured by the LR frames and need no "
            "probe, so including them would blur what the term buys."
        ),
    )
    lambda_anatomy: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the same super-Nyquist term applied to the ANATOMY pair "
            "instead of the fiducial: sum_l (1 - rho_l(pred, hr_target)) over "
            "the bands above the acquisition Nyquist.\n\n"
            "This is the term that actually pushes DETAIL into the output, and "
            "it is a different claim from `lambda_band`. The instrument term is "
            "a CERTIFICATE -- exogenous, anatomy-independent, and computable "
            "without ground truth. This one is ordinary supervision: it uses the "
            "high-resolution target, which exists at training time and does not "
            "at inference. Reporting a super-Nyquist gain earned by this term as "
            "if it were the certificate would be circular.\n\n"
            "It exists because nothing else in this objective targets high "
            "frequency. The strategy builds its own loss and never reads a "
            "`losses:` block, so HFEN / adversarial / perceptual terms are absent "
            "by construction; the objective is MSE plus a smoothness penalty. "
            "The (1 - rho) form is scale-free, so it constrains band STRUCTURE "
            "and leaves amplitude to the MSE -- the two do not fight."
        ),
    )
    n_sub_bands: int = Field(
        default=2,
        ge=1,
        description="Bands spanning [0, 1] in units of the acquisition Nyquist.",
    )
    n_super_bands: int = Field(
        default=2,
        ge=1,
        description=(
            "Bands spanning (1, rho_max]. These carry the certificate; >= 2 so "
            "the rolloff shape is visible rather than a single mean."
        ),
    )
    rho_max: float = Field(
        default=2.0,
        gt=1.0,
        description=(
            "Outer band edge in units of the acquisition Nyquist. Must be "
            "reachable on the grid or band_masks raises: the synthetic path "
            "reaches sqrt(ndim)*sr_scale, and real ULF->HF data reaches "
            "sqrt(ndim)*effective/grid (about 3.3 on T1w, 7.3 on T2w)."
        ),
    )
    min_bins: int = Field(
        default=16,
        ge=1,
        description=(
            "Smallest admissible band population. A band holding a handful of "
            "frequency bins yields a correlation dominated by sampling noise "
            "that still reads as a transfer measurement; raising is honest."
        ),
    )


class MultiAcquisitionConfig(BaseModel):
    """Faithful multi-acquisition synthesis for VF field-mapping arms.

    Several Virtual-Fiducial methods (AFI, double-angle, dual-echo) are defined
    over *multiple* acquisitions. When enabled, the training strategy synthesises
    the genuine acquisition stack with
    :class:`~spectramr.infrastructure.physics.multi_acquisition.MultiAcquisitionSimulator`
    (reusing the Bloch forward model + Bottomley T1) and supervises the model
    against the ground-truth field. ``method`` is rejected at load time if it is
    not one of the advertised set (no silent fallback — CLAUDE.md pitfall #9).

    Example:
        >>> MultiAcquisitionConfig(enabled=True, method="afi")
    """

    model_config = {"protected_namespaces": (), "extra": "forbid", "frozen": True}

    enabled: bool = Field(
        default=False,
        description="Synthesise a faithful multi-acquisition stack for this arm.",
    )
    method: (
        Literal[
            "double_angle",
            "afi",
            "dual_echo",
            "bloch_siegert",
            "mrf",
            "lowrank_temporal",
            "subvoxel_sr",
            "bssfp_banding",
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "Method whose acquisition stack to synthesise. One of double_angle / "
            "afi / dual_echo / bloch_siegert / mrf / lowrank_temporal / subvoxel_sr "
            "/ bssfp_banding."
        ),
    )
    n_timepoints: int = Field(
        default=16,
        ge=2,
        description="MRF transient length (= model in_channels / dictionary atom length).",
    )
    n_phase_cycles: int = Field(
        default=4,
        ge=4,
        description=(
            "Number of RF phase-cycles for bssfp_banding B0 mapping (= model "
            "in_channels). >= 4 is the elliptical-signal-model identifiability "
            "floor (Xiang-Hoff); the first-harmonic phase-cycle DFT needs it to "
            "resolve absolute off-resonance over one period."
        ),
    )
    use_real_stack: bool = Field(
        default=False,
        description=(
            "Path A (real data): feed the dataset's MEASURED phase-cycled stack "
            "(the model input) to the model instead of synthesising one, and grade "
            "the recovered field against the real b0_map. Use with "
            "dataset_type='oracle_bssfp'. Requires a b0_map in the batch."
        ),
    )
    n_frames: int = Field(
        default=8,
        ge=1,
        description=(
            "Frame count for lowrank_temporal (series length) / subvoxel_sr "
            "(input frames). The floor is method-aware (see the validator): "
            "lowrank_temporal needs >= 2 to have a temporal subspace at all, "
            "while subvoxel_sr admits 1 because a single frame is precisely the "
            "single-frame ablation floor its baseline arm is defined at."
        ),
    )
    subvoxel_registration: "SubvoxelRegistrationConfig" = Field(
        default_factory=lambda: SubvoxelRegistrationConfig(),
        description=(
            "Marker-anchored shift recovery for subvoxel_sr. Selects which rung "
            "of the shift-knowledge ladder the arm sits on."
        ),
    )
    sr_scale: int = Field(
        default=2,
        ge=1,
        description="Super-resolution upscaling factor for subvoxel_sr (model scale_factor).",
    )
    forward_psf: "ForwardPsfConfig" = Field(
        default_factory=lambda: ForwardPsfConfig(),
        description="Fiducial-measured blur operator. Off by default.",
    )
    anchor_conformal: "AnchorConformalConfig" = Field(
        default_factory=lambda: AnchorConformalConfig(),
        description="Fiducial-calibrated prediction intervals. Off by default.",
    )
    relaxometric_calibration: "RelaxometricCalibrationConfig" = Field(
        default_factory=lambda: RelaxometricCalibrationConfig(),
        description=(
            "Fiducial-calibrated contrast transfer between two field strengths. Off by default."
        ),
    )
    marker_channels: bool = Field(
        default=False,
        description=(
            "Append the fiducial stack to the model input, so the network sees "
            "the instrument as n EXTRA channels alongside the n anatomy frames "
            "(total n*4 with the 2n shift maps). Required by any model whose "
            "attention derives its queries and keys from the marker. Off by "
            "default, so pre-2026-07-26 arms are behaviourally unchanged.\n\n"
            "The channels are present on EVERY rung of an attention_keys "
            "ablation, carrying the real marker even where the model does not "
            "route through it, so architecture, parameter count and input "
            "geometry stay fixed and only the routing varies -- the same design "
            "as the shift-knowledge ladder."
        ),
    )
    band_probe: "BandProbeConfig" = Field(
        default_factory=lambda: BandProbeConfig(),
        description=(
            "Super-Nyquist probe on the virtual fiducial. Off by default, so "
            "pre-2026-07-26 arms are behaviourally unchanged."
        ),
    )
    normalize_magnitude: bool = Field(
        default=True,
        description=(
            "Normalise the image-domain magnitude (the M0 proxy) to a per-sample "
            "unit scale before synthesis. k-space targets must keep their raw "
            "scale through the loader (clamping at a percentile would clip the DC "
            "peak), so the IFFT'd magnitude carries an arbitrary scanner scale. "
            "Both the supervision target and the synthesised acquisition stack "
            "derive from it, so the objective grows QUADRATICALLY with that scale "
            "— the 2026-07 exp_vf_01 run hit a first-step gradient norm of 6477 "
            "against a clip of 1.0. Set false only to reproduce a pre-2026-07 run."
        ),
    )
    lambda_smooth: float = Field(
        default=0.01,
        ge=0.0,
        description=(
            "Weight of the L1 total-variation term on the PREDICTION. 0.0 "
            "disables it. The default reproduces the value this strategy "
            "hardcoded before 2026-07, so existing arms are unchanged.\n\n"
            "SCOPE: the justification is 'fields are smooth', which is correct "
            "for the B0/B1 field-mapping methods this strategy was built for -- "
            "there the supervised quantity IS a smooth field. It does NOT hold "
            "for subvoxel_sr, where the supervised quantity is high-resolution "
            "ANATOMY. Measured on a Shepp-Logan target, TV(truth) is 1.66x "
            "TV(a blurred reconstruction), so the term's optimum is strictly "
            "smoother than the ground truth and every move toward the correct "
            "answer increases it. Small (about 1.3% of the objective at 0.01) "
            "but pointed the wrong way, which is why it goes unnoticed. Set 0.0 "
            "on super-resolution arms."
        ),
    )

    @model_validator(mode="after")
    def _require_method_when_enabled(self) -> "MultiAcquisitionConfig":
        if self.enabled and self.method is None:
            raise ValueError(
                "physics.multi_acquisition.enabled=True requires 'method' "
                "(one of double_angle / afi / dual_echo)."
            )
        return self

    @model_validator(mode="after")
    def _n_frames_floor_is_method_aware(self) -> "MultiAcquisitionConfig":
        """One frame is meaningless for a series method and is the ablation
        floor for subvoxel_sr, so the floor cannot be a single field constraint.
        Before 2026-07-26 a blanket ``ge=2`` blocked the single-frame control
        that exp_vf_01's declared baseline is defined at."""
        if self.n_frames < 2 and self.method != "subvoxel_sr":
            raise ValueError(
                f"physics.multi_acquisition.n_frames={self.n_frames} is below the "
                f"floor of 2 for method={self.method!r}. Only 'subvoxel_sr' admits "
                "a single frame (its single-frame ablation control)."
            )
        return self

    @model_validator(mode="after")
    def _sr_scale_matches_the_declared_resolution_gap(self) -> "MultiAcquisitionConfig":
        """The decimation the arm SIMULATES must be the gap it CLAIMS to close.

        subvoxel_sr does not read the real 64mT acquisition: it pools the stored
        volume by ``sr_scale`` and asks the network to invert that. So the
        passband the network faces is ``sr_scale * voxel_mm``, NOT the scanner's
        native ``effective_voxel_mm``. An arm that declares a 1.6 mm effective
        resolution on a 0.49 mm grid (a 3.3x gap) while pooling 2x is training
        on one problem and reporting against another -- and every super-Nyquist
        number would be quoted against a Nyquist the run never had.
        """
        reg = self.subvoxel_registration
        if self.method != "subvoxel_sr" or reg.voxel_mm is None or reg.effective_voxel_mm is None:
            return self
        ratios = [e / g for g, e in zip(reg.voxel_mm, reg.effective_voxel_mm, strict=True)]
        in_plane = ratios[:2]  # the dither and the pooling are both in-plane
        if any(round(r) != self.sr_scale for r in in_plane):
            raise ValueError(
                f"sr_scale={self.sr_scale} does not match the declared in-plane "
                f"resolution gap effective/grid={[round(r, 2) for r in in_plane]} "
                f"(rounds to {[round(r) for r in in_plane]}). "
                "subvoxel_sr pools by sr_scale and inverts THAT, so the two must "
                "agree or the arm reports super-Nyquist gains against a Nyquist "
                "it never simulated. Set sr_scale to the rounded gap, or drop "
                "the millimetre declaration and run in pixel units."
            )
        return self

    @model_validator(mode="after")
    def _measured_psf_needs_a_fiducial(self) -> "MultiAcquisitionConfig":
        """A measured operator needs something known to measure it against."""
        psf = self.forward_psf
        if not psf.enabled:
            return self
        if self.method != "subvoxel_sr" or self.subvoxel_registration.shift_source != "recovered":
            raise ValueError(
                "forward_psf requires method='subvoxel_sr' with "
                "subvoxel_registration.shift_source='recovered'. Solving for a "
                "kernel from a blurred image alone is a blind deconvolution; "
                "the fiducial is what makes it a linear solve."
            )
        if psf.source == "measured" and psf.true_sigma_px is None:
            raise ValueError(
                "forward_psf.source='measured' with true_sigma_px unset: the "
                "simulator would apply exactly the assumed blur, so the "
                "measured and assumed kernels agree by construction and the arm "
                "cannot distinguish them. Declare true_sigma_px to create a "
                "model error worth correcting, or run the pair as an explicit "
                "null control on BOTH arms."
            )
        return self

    @model_validator(mode="after")
    def _anchor_conformal_needs_a_fiducial(self) -> "MultiAcquisitionConfig":
        """The calibration set IS the marker residual set."""
        if not self.anchor_conformal.enabled:
            return self
        if self.method != "subvoxel_sr" or self.subvoxel_registration.shift_source != "recovered":
            raise ValueError(
                "anchor_conformal requires method='subvoxel_sr' with "
                "subvoxel_registration.shift_source='recovered'. The whole "
                "construction is calibration on residuals whose truth is known, "
                "and only the fiducial supplies those."
            )
        return self

    @model_validator(mode="after")
    def _relaxometric_calibration_needs_a_fiducial(self) -> "MultiAcquisitionConfig":
        """kappa is measured ON the fiducial, so the fiducial has to exist."""
        if not self.relaxometric_calibration.enabled:
            return self
        if self.method != "subvoxel_sr":
            raise ValueError(
                f"relaxometric_calibration requires method='subvoxel_sr', got "
                f"{self.method!r}; no other method synthesises a fiducial."
            )
        if self.subvoxel_registration.shift_source != "recovered":
            raise ValueError(
                "relaxometric_calibration requires "
                "subvoxel_registration.shift_source='recovered'. The predicted "
                "gain is only checkable against a measured one on marker "
                "support, and there is no marker on the other rungs."
            )
        return self

    @model_validator(mode="after")
    def _marker_channels_need_a_marker(self) -> "MultiAcquisitionConfig":
        """Feeding the model a marker channel requires a marker to feed it."""
        if not self.marker_channels:
            return self
        if self.method != "subvoxel_sr":
            raise ValueError(
                f"marker_channels=True requires method='subvoxel_sr', got "
                f"{self.method!r}; no other method synthesises a fiducial."
            )
        if self.subvoxel_registration.shift_source != "recovered":
            raise ValueError(
                "marker_channels=True requires "
                "subvoxel_registration.shift_source='recovered', got "
                f"{self.subvoxel_registration.shift_source!r}. The fiducial is "
                "only synthesised on that rung, so the extra channels would be "
                "zeros under a name that says otherwise."
            )
        return self

    @model_validator(mode="after")
    def _band_probe_has_a_marker_to_probe(self) -> "MultiAcquisitionConfig":
        """The probe reads the fiducial, so it cannot outlive the fiducial.

        Enabling it on an arm with no marker would produce a loss term computed
        on nothing -- a reported number with no mechanism behind it, which is
        pitfall #16 in its purest form.
        """
        probe = self.band_probe
        if probe.lambda_anatomy > 0.0 and not probe.enabled:
            raise ValueError(
                f"band_probe.lambda_anatomy={probe.lambda_anatomy} > 0 but "
                "band_probe.enabled is False. The anatomy term reuses the same "
                "band partition as the instrument probe, so the partition has "
                "to be built; a weighted term with no partition would be "
                "weighted and never computed."
            )
        if probe.lambda_band > 0.0 and not probe.enabled:
            raise ValueError(
                f"band_probe.lambda_band={probe.lambda_band} > 0 but "
                "band_probe.enabled is False, so the term would be weighted and "
                "never computed. Set enabled=true, or lambda_band=0.0 for the "
                "report-only control arm."
            )
        if not probe.enabled:
            return self
        if self.method != "subvoxel_sr":
            raise ValueError(
                f"band_probe.enabled=True requires method='subvoxel_sr', got "
                f"{self.method!r}. The probe measures transfer across the "
                "decimation the SR method inverts; no other method defines one."
            )
        # The PARTITION is defined by the decimation, not by the marker, so
        # `enabled` alone needs no fiducial: it builds the bands and lets
        # `lambda_anatomy` act on the anatomy pair. Only the INSTRUMENT term
        # needs an instrument. Coupling the two blocked the high-frequency term
        # on the blind and oracle rungs of the shift ladder, which have no
        # marker by construction and still need the detail loss.
        if probe.lambda_band > 0.0 and self.subvoxel_registration.shift_source != "recovered":
            raise ValueError(
                f"band_probe.lambda_band={probe.lambda_band} > 0 requires "
                "subvoxel_registration.shift_source='recovered', got "
                f"{self.subvoxel_registration.shift_source!r}. The fiducial is "
                "only synthesised on that rung, and the instrument term IS the "
                "fiducial passed through the network. `lambda_anatomy` has no "
                "such requirement -- it acts on the anatomy pair."
            )
        return self

    @model_validator(mode="after")
    def _subvoxel_registration_is_coherent(self) -> "MultiAcquisitionConfig":
        reg = self.subvoxel_registration
        if self.enabled and self.method != "subvoxel_sr" and reg.shift_source != "blind":
            raise ValueError(
                "physics.multi_acquisition.subvoxel_registration only applies to "
                f"method='subvoxel_sr', not {self.method!r}. Declaring it here "
                "would advertise a mechanism nothing reads (CLAUDE.md #15)."
            )
        if reg.shift_source == "recovered" and reg.marker_jitter <= 0.0:
            raise ValueError(
                "shift_source='recovered' requires marker_jitter > 0. A perfectly "
                "periodic fiducial lattice has a comb spectrum (0.27% of bins "
                "populated) and is ambiguous modulo one lattice period under "
                "translation, so phase correlation cannot register it: measured "
                "0.55 px error vs 0.003 px at jitter=0.35."
            )
        return self


class PhysicsConfigSchema(BaseModel):
    """Comprehensive physics configuration for MRI reconstruction.

    Aggregates all physics-related parameters including data consistency,
    coil sensitivities, k-space processing, regularization, compressed sensing,
    and corrections.

    Example:
        >>> physics = PhysicsConfigSchema(
        ...     data_consistency=DataConsistencyConfig(enabled=True),
        ...     regularization=RegularizationConfig(
        ...         enable_tv=True,
        ...         tv_weight=0.01,
        ...     ),
        ...     compressed_sensing=CompressedSensingConfig(
        ...         enabled=True,
        ...         sampling_pattern="cartesian",
        ...     ),
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    # Data consistency
    data_consistency: DataConsistencyConfig = Field(
        default_factory=DataConsistencyConfig,
        description="Data consistency constraint configuration",
    )

    # Coil sensitivities
    coil_sensitivity: CoilSensitivityConfig = Field(
        default_factory=CoilSensitivityConfig,
        description="Coil sensitivity estimation configuration",
    )

    # Coil processing pipeline (compress -> estimate -> combine SSOT)
    coil_processing: CoilProcessingConfig = Field(
        default_factory=CoilProcessingConfig,
        description="Config-driven coil-processing pipeline configuration",
    )

    # K-space processing
    kspace: KSpaceConfig = Field(
        default_factory=KSpaceConfig,
        description="K-space domain processing configuration",
    )

    # Regularization
    regularization: RegularizationConfig = Field(
        default_factory=RegularizationConfig,
        description="Spatial regularization configuration",
    )

    # Iterative refinement
    iterative_refinement: IterativeRefinementConfig = Field(
        default_factory=IterativeRefinementConfig,
        description="Iterative refinement configuration",
    )

    # Phase correction
    phase_correction: PhaseCorrectionConfig = Field(
        default_factory=PhaseCorrectionConfig,
        description="Phase error correction configuration",
    )

    # B0 correction
    b0_correction: B0CorrectionConfig = Field(
        default_factory=B0CorrectionConfig,
        description="B0 field inhomogeneity correction configuration",
    )

    # B0 simulation (for training augmentation)
    b0_simulation: B0SimulationConfig = Field(
        default_factory=B0SimulationConfig,
        description="B0 simulation for training-time augmentation",
    )

    # Field strength (for proper B0 scaling)
    field_strength: float = Field(
        default=3.0,
        ge=0.01,
        le=10.0,
        description="Main magnetic field strength in Tesla (e.g., 3.0, 1.5, 0.55, 0.064)",
    )

    # Motion correction
    motion_correction: MotionCorrectionConfig = Field(
        default_factory=MotionCorrectionConfig,
        description="Motion artifact correction configuration",
    )

    # Compressed sensing
    compressed_sensing: CompressedSensingConfig = Field(
        default_factory=CompressedSensingConfig,
        description="Compressed sensing reconstruction configuration",
    )

    # PINN
    pinn: PINNConfig = Field(
        default_factory=PINNConfig,
        description="Physics-Informed Neural Network configuration",
    )

    # Bloch simulation
    bloch_simulation: BlochSimulationConfig = Field(
        default_factory=BlochSimulationConfig,
        description="Bloch simulation configuration",
    )

    # Digital Twin simulator
    digital_twin: DigitalTwinConfig = Field(
        default_factory=DigitalTwinConfig,
        description="Digital Twin MRI simulator for virtual fiducial training",
    )

    # Faithful multi-acquisition synthesis (VF field-mapping arms)
    multi_acquisition: MultiAcquisitionConfig = Field(
        default_factory=MultiAcquisitionConfig,
        description="Faithful multi-acquisition stack synthesis for AFI/DAM/dual-echo",
    )

    # === v6.1 additions (additive, default-disabled, see
    # TODO/integration_plan_ulf_cheap_fast_mri.md §0.1) ===
    concomitant: ConcomitantFieldConfig = Field(
        default_factory=ConcomitantFieldConfig,
        description=(
            "Concomitant-field correction parameters (idea 1: CAUR). "
            "Active only at low field; disabled by default for v6.0 parity."
        ),
    )
    relaxation_priors: RelaxationPriorsConfig = Field(
        default_factory=RelaxationPriorsConfig,
        description=(
            "Tissue-class T1/T2 prior tables for Bloch-constrained "
            "translation (idea 3). None means no priors active."
        ),
    )
    gradient_waveform_uri: str | None = Field(
        default=None,
        description=(
            "URI of a gradient-waveform CSV/JSON for differentiable "
            "pulse-sequence co-design (idea 4). None disables."
        ),
    )


# Re-export the new sub-schemas alongside the existing public surface.
__all__ = [
    "PhysicsConfigSchema",
    "DataConsistencyConfig",
    "CoilSensitivityConfig",
    "KSpaceConfig",
    "RegularizationConfig",
    "IterativeRefinementConfig",
    "PhaseCorrectionConfig",
    "PhaseCorrectionnConfig",  # deprecated alias (doubled-n typo); remove 2026-08
    "B0CorrectionConfig",
    "B0SimulationConfig",
    "MotionCorrectionConfig",
    "CompressedSensingConfig",
    "PINNConfig",
    "BlochSimulationConfig",
    "DigitalTwinConfig",
    # v6.1
    "ConcomitantFieldConfig",
    "RelaxationPriorsConfig",
]
