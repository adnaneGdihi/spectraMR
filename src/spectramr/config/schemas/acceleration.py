"""K-space acceleration configuration schema - SSOT for undersampling patterns.

Defines undersampling patterns, acceleration factor schedules, and k-space
masking strategies for k-space based reconstruction tasks (SENSE, compressed sensing).

This schema is imported and embedded in training/base.py::BaseTrainingConfigSchema
as the "acceleration" field, making it part of the unified training configuration.
"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .enums import AccelerationSchedule


class AdaptiveAccelerationConfig(BaseModel):
    """Learnable / relaxed-Bernoulli acceleration mask (LOUPE / PILOT).

    Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-11 (M3).

    When ``enabled=True``, the dataloader emits a fully-sampled k-space and
    the model wraps it with a learnable mask layer (Gumbel-sigmoid). A
    density penalty pulls the expected sampling rate toward
    ``target_rate``. Annealing parameters control the Gumbel temperature
    schedule.
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",
        "frozen": True,
    }

    enabled: bool = Field(
        default=False,
        description="Enable a learnable Bernoulli mask co-trained with reconstruction.",
    )
    target_rate: float = Field(
        default=0.125,
        gt=0.0,
        le=1.0,
        description="Target expected sampling rate (1/R). 0.125 == 8x acceleration.",
    )
    density_penalty_lambda: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight on the |E[mask] - target_rate| penalty.",
    )
    budget: Literal["soft", "hard_topk"] = Field(
        default="soft",
        description=(
            "Eval-time mask. 'soft' keeps the lines with sigmoid(logit) > 0.5, which can keep "
            "none at a low target_rate. 'hard_topk' keeps the round(target_rate * W) highest "
            "logits, so validation spends the same budget the training mask targets."
        ),
    )
    mask_learning_rate: float = Field(
        default=1e-3,
        gt=0.0,
        description="Learning rate of the mask logits (LOUPE) or trajectory (PILOT).",
    )

    class _AnnealConfig(BaseModel):
        model_config = {"extra": "ignore", "frozen": True}
        tau_init: float = Field(default=5.0, gt=0.0)
        tau_final: float = Field(default=0.1, gt=0.0)
        anneal_steps: int = Field(default=50_000, ge=1)

    annealing: _AnnealConfig = Field(default_factory=_AnnealConfig)


class AccelerationConfigSchema(BaseModel):
    """K-space undersampling configuration.

    Mounted at ``undersampling:`` since phase 11. The block was ``acceleration:``,
    a name that meant two unrelated things: the MRI k-space ACCELERATION FACTOR
    (what these 26 fields configure) and COMPUTE acceleration. A reader could not
    tell from the block name which sense a key belonged to. The legacy spelling
    still loads -- a ROOT fold moves the whole block -- but it is gone from
    Python: read ``config.undersampling``.

    Centralized configuration for k-space masking and acceleration factors
    used during training and inference for MRI reconstruction.
    It does NOT contain training-specific fields (those are in training/base.py).

    **Five fields here are compute knobs, not undersampling knobs, and all five
    are inert** -- zero readers on this block. They are the reason the old name
    read ambiguously. They stay flat rather than being folded away, so their
    inertness stays visible (issue #680):

    * ``mixed_precision`` -> live equivalent ``optimization.precision.enabled``
    * ``use_compile`` -> ``optimization.compile.enabled``
    * ``use_gradient_checkpointing`` -> ``optimization.gradient.enable_checkpointing``
    * ``gradient_accumulation_steps`` -> ``optimization.gradient.accumulation_steps``
    * ``use_distributed`` -> **no equivalent**; ``parallel:`` has ``backend`` /
      ``num_devices`` / ``fsdp`` / ``deepspeed`` but no boolean gate, so there is
      nothing to point it at. That is an owner decision, not a rename.

    ``ground_truth_folder`` / ``ground_truth_type`` are also inert here.

    Example:
        ```yaml
        undersampling:
          base_acceleration: 4.0
          max_acceleration: 8.0
          acceleration_schedule: linear
          acceleration_type: variable_density
          mask_types: [variable_density]
          center_fraction: 0.08
        ```
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "ignore",  # Improved flexibility for experimental configs
        "frozen": True,
    }

    # ============================================================
    # CORE K-SPACE ACCELERATION PARAMETERS
    # ============================================================

    # Acceleration factors
    base_acceleration: float = Field(
        default=4.0,
        ge=1.0,
        description=(
            "Base k-space acceleration factor (undersampling ratio) the loader applies "
            "literally; 1.0 means fully sampled (no masking). Below 1.0 is refused."
        ),
    )
    max_acceleration: float = Field(
        default=8.0,
        gt=0,
        description="Maximum acceleration factor for progressive/curriculum training",
    )
    acceleration_schedule: AccelerationSchedule = Field(
        default=AccelerationSchedule.LINEAR,
        description="Acceleration ramp schedule: linear, polynomial, exponential, step, power_law",
    )

    # Schedule parameters
    schedule_steps: int = Field(
        default=10000,
        ge=1,
        description="Number of steps to reach max acceleration",
    )
    schedule_power: float = Field(
        default=2.0,
        gt=0,
        description="Power parameter for polynomial acceleration schedule",
    )
    schedule_type: AccelerationSchedule | None = Field(
        default=None,
        description="Alias for acceleration_schedule or specific schedule type",
    )

    # ============================================================
    # K-SPACE UNDERSAMPLING PATTERN SPECIFICATION
    # ============================================================

    # Acceleration type specification
    acceleration_type: str = Field(
        default="variable_density",
        description="Primary k-space undersampling pattern",
    )
    mask_direction: str | None = Field(
        default=None,
        description=(
            "Direction of masking: 'phase' (lines indexed by k_y) or 'readout' "
            "(lines indexed by k_x). Mapped onto the accelerator's ``line_axis`` "
            "and honoured only by the line-based Cartesian families; the "
            "trajectory families define their own support."
        ),
    )

    # NOTE: acceleration_type is deliberately NOT validated against the
    # accelerator registry here. Doing so requires importing
    # infrastructure.physics from config/, which is a leftward import that
    # non-negotiable #5 forbids and tests/architecture/test_layer_direction.py
    # catches -- a function-local import does not exempt it. The check belongs in
    # the audit layer, which may import both. Tracked in issue #967.
    enforce_nested: bool = Field(
        default=False,
        description=(
            "Coerce the mask cascade so M_{t+1} is a subset of M_t at every "
            "timestep. Cold diffusion's forward process assumes k-space is only "
            "ever REMOVED as t grows; families that re-draw their pattern per "
            "timestep (radial, spiral, multi_mask, equispaced under a shrinking "
            "ACS) break that, and the reverse loop has no mechanism to undo an "
            "addition. Enforcement intersects the cascade, so it can only remove "
            "samples: a family that re-draws heavily collapses and raises rather "
            "than silently training on a degenerate mask. Applies to the "
            "fixed-seed cascade only (validation, predict and the reverse loops; "
            "training too when enable_dynamic_mask is off). With "
            "enable_dynamic_mask on, the training draws vary per sample and run "
            "unenforced; provenance records the applied scope under mask_scope. "
            "Default False keeps every existing run byte-identical."
        ),
    )
    nested_tolerance: float = Field(
        default=0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Minimum share of the family's OWN raw draw at each timestep that an "
            "enforced mask must retain before enforce_nested raises. Only read "
            "when enforce_nested is true. The denominator is the raw draw, not "
            "the continuous 1/declared_R: Cartesian families quantise in whole "
            "k-space lines and can never match a continuous target exactly, so "
            "measuring against it fired on sub-line rounding and made 1.0 "
            "unsatisfiable even for a family that nests perfectly. Against the "
            "raw draw, 1.0 means 'enforcement must be a no-op'. Whether the raw "
            "draw honours its declared R is a separate check, "
            "KSpaceUndersamplingProcess.declared_ladder_defects."
        ),
    )
    # Pattern-specific parameters
    density_power: float = Field(
        default=2.0,
        ge=0,
        description="Power for variable density sampling (higher = more center-focused)",
    )
    sigma_scaling: float = Field(
        default=0.25,
        gt=0,
        description="Scaling factor for Gaussian sigma",
    )
    std_scale: float = Field(
        default=4.0,
        gt=0,
        description="Standard deviation scale for Gaussian PDF",
    )
    pf_fraction: float = Field(
        default=0.75,
        ge=0.5,
        le=1.0,
        description="Partial Fourier fraction",
    )
    pattern_path: str | None = Field(
        default=None,
        description="Path to learned sampling pattern",
    )

    # Multiple masks support
    mask_types: list[str] = Field(
        default_factory=lambda: ["variable_density"],
        description="List of undersampling mask types to use/rotate through.",
    )
    mask_combination_mode: str = Field(
        default="rotate",
        description="How to handle multiple mask_types",
    )

    # K-space masking parameters
    center_fraction: float = Field(
        default=0.08,
        ge=0,
        le=1,
        description="Fraction of k-space center to always keep",
    )
    min_center_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "ACS fraction at MAXIMUM acceleration. The always-sampled centre band "
            "ramps from ``center_fraction`` at R=base down to this value at R=max, "
            "so a nominally-high rung stays realisable. ``None`` keeps the ACS "
            "static at ``center_fraction``, which caps every rung at "
            "R <= 1/center_fraction regardless of what the ladder declares "
            "(issue #534). Consumed by ``KSpaceUndersamplingProcess`` via "
            "``resolve_undersampling_kwargs``."
        ),
    )
    undersampling_pattern: str = Field(
        default="random",
        description="Undersampling distribution within mask",
    )

    # Dynamic/stochastic masking
    enable_dynamic_mask: bool = Field(
        default=False,
        description="Use different masks for each training sample",
    )
    mask_seed: int | None = Field(
        default=None,
        description=(
            "Random seed for reproducible mask generation. None is not unseeded: the "
            "k-space process falls back to 42 (models/diffusion/kspace_process.py) and "
            "the loader-side MaskGenerator is deterministic, so an unset seed is "
            "reproducible but undocumented; the effective seed is logged at setup. "
            "Scope: the seed of the fixed-seed cascade (validation, predict, the "
            "reverse loops, and training when enable_dynamic_mask is off). With "
            "enable_dynamic_mask on, each training sample draws a fresh seed from "
            "the global RNG instead; provenance records the applied scope under "
            "mask_scope."
        ),
    )
    train_identity_rung: bool = Field(
        default=False,
        description=(
            "Treat t=0 as a trainable rung even when R(0) == 1. Only has an "
            "effect when the ladder's first rung is fully sampled: "
            "``KSpaceUndersamplingProcess.min_meaningful_timestep()`` already "
            "returns 0 whenever R(0) > 1, so on every other arm this knob is a "
            "no-op. At R(0) == 1 the degradation at t=0 IS the identity, but "
            "the supervision is not -- an arm whose target differs from its "
            "input (NEX averaging, denoising, cross-contrast) has a real "
            "fully-sampled task there, which the default floor of 1 excludes "
            "from training while the reverse schedule still terminates on it "
            "(issue #535). Opt in per arm; the resolved floor is stamped into "
            "the debug snapshot's provenance block. Consumed by "
            "``KSpaceUndersamplingProcess`` via ``resolve_undersampling_kwargs``."
        ),
    )

    # ============================================================
    # GROUND TRUTH / REFERENCE DATA
    # ============================================================

    ground_truth_folder: str | None = Field(
        default=None,
        description="Path to ground truth (fully sampled) k-space data folder",
    )
    ground_truth_type: str = Field(
        default="kspace",
        description="Ground truth data domain: 'kspace' or 'image'",
    )

    # ============================================================
    # PROGRESSIVE TRAINING ACCELERATION RANGE
    # ============================================================

    acceleration_range: list[float] = Field(
        default_factory=lambda: [1.0, 16.0],
        description="Min/max acceleration factor for progressive training",
    )

    # ============================================================
    # LEGACY / MIGRATION SUPPORT
    # ============================================================
    mixed_precision: bool | None = Field(
        default=None,
        description="DEPRECATED: Use optimization.precision.enabled instead.",
    )
    gradient_accumulation_steps: int | None = Field(
        default=None,
        description="DEPRECATED: Use optimization.gradient.accumulation_steps instead.",
    )
    use_compile: bool | None = Field(
        default=None,
        description="DEPRECATED: Use optimization.use_compile instead.",
    )
    use_distributed: bool | None = Field(
        default=None,
        description="DEPRECATED: Use parallel.strategy instead.",
    )
    use_gradient_checkpointing: bool | None = Field(
        default=None,
        description="DEPRECATED: Use optimization.gradient.enable_checkpointing instead.",
    )

    # ============================================================
    # v6.1 — LEARNABLE ACQUISITION (LOUPE / PILOT)
    # ============================================================
    adaptive: AdaptiveAccelerationConfig = Field(
        default_factory=AdaptiveAccelerationConfig,
        description=(
            "Learnable mask configuration (LOUPE / PILOT). "
            "Disabled by default; v6.0 semantics preserved."
        ),
    )

    # ============================================================
    # v6.1 — Sampling-pattern alias for non-learned baselines
    # ============================================================
    sampling_pattern: str | None = Field(
        default=None,
        description=(
            "Alias / override for the dominant Cartesian sampling pattern. "
            "Examples: 'uniform_random', 'equispaced', 'variable_density'. "
            "Takes precedence over ``acceleration_type`` when set."
        ),
    )
    rate: float | None = Field(
        default=None,
        gt=0,
        description="Alias for ``base_acceleration`` — convenience for control YAMLs.",
    )

    @property
    def declares_no_acceleration(self) -> bool:
        """True for the explicit fully-sampled declaration.

        ``base_acceleration: 1.0`` alone is ambiguous: a cold-diffusion ladder
        starts at 1x and climbs to ``max_acceleration`` (default 8.0), so a block
        declares no acceleration only when base and max are both 1.0, a declared
        ``acceleration_range`` holds nothing above 1.0, and no dynamic mask is
        requested. The `undersampling_block_is_applied` witness and the
        `acceleration_schedule_steps_match` health check read this predicate; the
        loader applies `base_acceleration` as declared and needs no predicate.
        """
        if float(self.base_acceleration) != 1.0 or float(self.max_acceleration) != 1.0:
            return False
        if "acceleration_range" in self.model_fields_set and any(
            float(a) != 1.0 for a in self.acceleration_range
        ):
            return False
        return not self.enable_dynamic_mask

    @model_validator(mode="before")
    @classmethod
    def _reject_line_axis(cls, data: object) -> object:
        # `undersampling.line_axis` was a second, unread spelling of which axis a
        # line mask removes; the block is extra="ignore", so without this a
        # declared value would vanish silently (#2160).
        if isinstance(data, dict) and "line_axis" in data:
            raise ValueError(
                "undersampling.line_axis is retired and was never read. Declare the "
                "dataset's data.acquisition.phase_encode_axis and an "
                "undersampling.mask_direction (phase/readout, or the raw axis x/y)."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _apply_rate_alias(cls, data: object) -> object:
        # ``rate`` is a documented convenience alias for ``base_acceleration``
        # (WS1-physics-03): wire it so the advertised behaviour is real. When
        # ``rate`` is provided and ``base_acceleration`` is not explicitly set,
        # copy it across; an explicit ``base_acceleration`` always wins. No
        # current YAML sets ``rate``, so this is a no-op for existing arms.
        if isinstance(data, dict):
            rate = data.get("rate")
            if rate is not None and data.get("base_acceleration") is None:
                data["base_acceleration"] = rate
        return data

    @model_validator(mode="after")
    def _check_center_fraction_ordering(self) -> "AccelerationConfigSchema":
        # The ACS ramp in ``CartesianUniformAccelerator.get_acceleration_mask``
        # only fires when ``min_center_fraction < center_fraction``. A reversed
        # pair therefore reads as a configured ramp and silently produces a
        # static ACS, which is exactly the failure mode #534 was about. Raise
        # instead of degrading (CLAUDE.md pitfall #9).
        if self.min_center_fraction is None:
            return self
        if self.min_center_fraction > self.center_fraction:
            raise ValueError(
                f"acceleration.min_center_fraction ({self.min_center_fraction}) must be "
                f"<= acceleration.center_fraction ({self.center_fraction}). The ACS band "
                f"shrinks from center_fraction at R=base to min_center_fraction at "
                f"R=max, so a larger minimum describes a ramp that cannot run and the "
                f"ACS would stay static at center_fraction."
            )
        return self


__all__ = ["AccelerationConfigSchema", "AdaptiveAccelerationConfig"]
