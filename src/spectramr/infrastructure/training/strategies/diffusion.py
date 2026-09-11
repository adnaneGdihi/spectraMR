"""DiffusionXDiffusion Training Strategy Module

This module contains Diffusion and XDiffusion training strategies.
"""

# Config leak prevention: accelerator_kwargs now passed from config
# instead of hardcoded pattern overrides. See test_diffusion_leak.py for validation.

import gc
import inspect
import logging
import math
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.core.cascading_validation import (
    IDENTITY_ACCELERATION,
    build_cascade_row,
    check_round_trip,
    legacy_linear_timestep,
    reconcile_skipped_levels,
    resolve_cascade_levels,
    training_band,
)
from spectramr.core.curriculum import (
    DEFAULT_MAX_ITERATIONS,
    SHORT_RUN_BYPASS_ITERATIONS,
    CurriculumState,
    resolve_curriculum_state,
)
from spectramr.data.batch_types import align_scale_to_batch, read_batch_field
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.coil_sensitivity import (
    SMAP_KSPACE_PEAK_RATIO,
    estimate_smaps,
    prepare_smaps_for_kspace_conditioning,
)
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.debug_snapshot import ChannelSegment
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.strategies.mixins.adversarial import (
    AdversarialMixin,
    _resolve_disc_updates,
    assemble_adversarial_step_configs,
)
from spectramr.infrastructure.training.t0_predc_probe import (
    generator_exposes_pre_dc,
    run_t0_predc_probe,
)
from spectramr.models.capabilities import StrategyCapabilities
from spectramr.models.losses.computers import UnifiedDiffusionLossComputer
from spectramr.models.losses.computers.unified_diffusion_reconstruction import (
    _call_safe_loss,
)
from spectramr.models.losses.critic_domain import (
    DomainAdaptedCritic,
    critic_accepts_complex,
    critic_component_name,
    critic_input_domain,
    generator_output_domain,
    resolve_conversion,
    to_critic_input,
)
from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss
from spectramr.models.losses.registry import create_loss

from ..utils.data_adapters import TorchIOAdapter
from ..utils.diffusion_mixin import DiffusionStrategyMixin
from .base import BaseTrainingStrategy
from .mixins.utils import pick_present

logger = logging.getLogger(__name__)

# Posterior / reconstruction samplers consumed via ``generator.sample(measurement,
# mask)`` during validation (XC.1). Only samplers exposing the World-A
# ``get_sampler(name, model=..., ...)`` + ``sample(measurement, mask)`` convention
# belong here. ``dds`` (DDSReconSampler) is the first; adding another requires a
# call-compatible adapter (mirroring DDSReconSampler) AND an entry here, else the
# generic ``generate``/single-step path is used. Guarded by
# ``check_dds_requires_score_model`` (paradigm) at config time.
_POSTERIOR_RECON_SAMPLERS = frozenset({"dds", "dynamic_dps"})

_VALID_SAMPLING_STRATEGIES: tuple[str, ...] = (
    "uniform",
    "importance",
    "linear_decay",
    "high_t_emphasis",
    "balanced_high_t",
)

# Uniform-floor mixing weight for the ``balanced_high_t`` timestep sampler.
# P(t) = (1 - eps) * (t / sum(t)) + eps * (1 / N): the high-t emphasis keeps
# (1-eps) of the mass while every timestep is guaranteed at least eps/N. This
# stops pure ``high_t_emphasis`` (P(t) ∝ t) from starving the low-t / low-R
# band that validation reverse-samples for R2x (experiment_11 accel-inversion).
# A documented constant, not a per-arm knob: the sampler NAME is the wired,
# schema-validated knob. Promote to a config field only if a sweep needs it.
_BALANCED_HIGH_T_UNIFORM_FLOOR: float = 0.3

# Parameter names, in precedence order, under which the resolved
# ``validation.sampler_steps`` is forwarded to a generator's ``sample()``.
# Samplers spell the step count differently, so the forwarding site probes the
# signature. A named constant (not an inline literal) so the paired test can
# bind to the same tuple: renaming a sampler's parameter without updating this
# list re-creates the silent no-op it exists to prevent (pitfall #15).
_SAMPLER_STEP_PARAM_NAMES: tuple[str, ...] = (
    "num_inference_steps",
    "num_steps",
    "sampling_steps",
    "n_steps",
    "steps",
)


def describe_nonfinite_prediction(hr_fakes: torch.Tensor) -> str | None:
    """Describe a sampler output carrying non-finite values, else ``None``.

    A diverged sampler does not raise, so it slips past
    ``_generate_validation_prediction``'s ``except`` untouched and first becomes
    visible as a solid-black PNG many frames later — by which point
    ``MetricsTracker._normalize_images`` has mapped NaN, constant output and a
    tiny dynamic range onto one indistinguishable artifact. This names the
    condition while the producing path is still on the stack.

    Predictions on the cold-diffusion arms are **k-space**, so a single
    non-finite entry spreads across the entire image through the IFFT: 1 bad
    value out of 500k blacks out every sample in the batch. That is why the
    count matters and why "8/8 samples black" does not imply eight independent
    failures.

    Returns:
        A ready-to-log WARNING message, or ``None`` when every value is finite.
    """
    finite_mask = torch.isfinite(hr_fakes)
    if bool(finite_mask.all()):
        return None

    n_bad = int(hr_fakes.numel() - int(finite_mask.sum().item()))
    finite_vals = hr_fakes[finite_mask]
    if finite_vals.numel():
        extent = (
            f"finite magnitude range [{finite_vals.abs().min().item():.6g}, "
            f"{finite_vals.abs().max().item():.6g}]"
        )
    else:
        extent = "no finite values at all"
    return (
        f"[DiffusionValidation] Sampler returned {n_bad}/{hr_fakes.numel()} "
        f"non-finite value(s) (shape={list(hr_fakes.shape)}, {extent}). On a "
        f"k-space arm a single one propagates to the whole image through the "
        f"IFFT and renders as a solid-black validation PNG. The saved 'fake' "
        f"images for this step are a RENDER OF A DIVERGED SAMPLE, not a "
        f"reconstruction."
    )


def cold_model_input_key(tensors: Mapping[str, Any]) -> str:
    """Which key in a cold-diffusion snapshot holds the tensor the model is fed.

    ``model_input`` is present exactly when sensitivity maps were concatenated
    onto the degraded k-space, which is the default for ``kspace_cold_diffusion``;
    without conditioning the backbone receives ``noisy_kspace`` itself.

    A function rather than the expression written out at each emitter: the two
    sites that answer this question -- the declaring path in
    ``_prepare_diffusion_inputs`` and the direct ``diffusion_step`` emitter --
    must agree, and #1298 is what a divergence between a snapshot and its own
    label costs. One of them getting a later fix the other missed is the exact
    shape of #697.
    """
    return "model_input" if "model_input" in tensors else "noisy_kspace"


class DiffusionTrainingStrategy(BaseTrainingStrategy, DiffusionStrategyMixin, AdversarialMixin):
    """Diffusion training strategy with shared diffusion behavior.

    Implements score-based diffusion models for image reconstruction. Supports both
    forward and reverse process training with configurable noise schedules (linear,
    cosine, etc.).

    ## Training Process

    1. **Forward Process**: Gradually add Gaussian noise to clean images
    2. **Reverse Process**: Train network to predict noise at each timestep
    3. **Loss Computation**: MSE/L1 loss between predicted and actual noise
    4. **Validation**: Generate samples via reverse process (DDIM, DDPM)

    ## Key Configuration Parameters

    - `training.training_mode`: Must be 'diffusion'
    - `training.diffusion.timesteps`: Number of diffusion steps (e.g., 1000)
    - `training.diffusion.noise_schedule`: Schedule type ('linear', 'cosine')
    - `model.model_type`: Generator architecture (e.g., 'standard_unet')

    ## Cross-Domain Support

    Handles both image and k-space domains via FFT transformations:
    - Detects input domain (image vs. k-space complex)
    - Auto-transforms to model domain via `_prepare_model_input()`
    - Supports cold diffusion (deterministic degradation)

    ## Loss Components

    Computes multiple loss terms via `UnifiedDiffusionLossComputer`:
    - **Diffusion Loss**: Denoising objective (primary)
    - **Adversarial Loss**: Optional GAN-style discriminator loss
    - **Perceptual Loss**: Optional VGG/LPIPS consistency
    - **Reconstruction Loss**: L1/L2 with target

    Attributes:
        state: TrainingState containing model, config, device
        loss_computer: UnifiedDiffusionLossComputer for loss aggregation
        num_timesteps: Number of diffusion steps
        beta_schedule: Noise schedule type
        device: Torch device for computation
    """

    #: Validation keys this strategy writes itself, outside the MetricsRegistry
    #: computer (cohort review 2026-09-02, T0.5): the multi-sample ensemble's
    #: per-rung spread and coverage, in the concrete spellings the default
    #: (2, 8, 32) cascade and ``_stamp_accel_mean`` produce. A name FAMILY
    #: cannot be declared today (#1733), so a custom ``validation.cascade.levels``
    #: ladder mints suffixed keys this set does not list -- the audit witness
    #: strips the ``_<R>x`` suffix before resolving, the metrics mixin's exact
    #: match does not. ``empirical_coverage`` is ALSO a registered metric, so its
    #: bare name resolves through the registry; the suffixed spellings are here
    #: for the mixin. Declaring a non-empty set flips both gates from an INFO
    #: census line to an error for every arm that resolves to this class (140
    #: on 2026-09-03, all clean).
    capabilities = StrategyCapabilities(
        emitted_metrics=frozenset(
            {
                "val_ensemble_std_mean",
                "val_ensemble_std_mean_2x",
                "val_ensemble_std_mean_8x",
                "val_ensemble_std_mean_32x",
                "val_ensemble_std_mean_mean",
                "val_empirical_coverage",
                "val_empirical_coverage_2x",
                "val_empirical_coverage_8x",
                "val_empirical_coverage_32x",
                "val_empirical_coverage_mean",
            }
        )
    )

    #: Diffusion degrades the prepared input INSIDE the step -- ``q_sample``
    #: adds noise, or (cold) zero-fills with ``x_0 * mask``. The base class
    #: captures ``first_steps/input_prepared`` before the forward pass, so for
    #: this family it is the CLEAN tensor, not the one the model receives. The
    #: real model input is emitted separately under ``diffusion_step`` (key
    #: ``noisy_kspace``, #1177). Declaring False here is what stops a reader
    #: concluding a cold-diffusion arm was fed fully-sampled data --
    #: ``docs/debug_snapshot_contract.rst``.
    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"
    #: Cold / Gaussian schedules build their own masks (``generate_batch_masks``).
    applies_undersampling = True
    #: The cold branch grades the zero-filled baseline from the MASKED measurement
    #: it built itself (``_zf_measurement``); the generic input-as-prediction
    #: baseline in ``ModelValidationMixin`` must not emit a second ``val_zf_*``.
    _owns_zero_filled_baseline = True

    #: Loss ownership (issue #1918). This strategy computes NO image loss inline:
    #: every ``losses.image_losses`` / ``kspace_losses`` / ``complex_losses`` entry
    #: is built by ``LossBuilder`` into ``env.losses``, forwarded as ``losses_dict``
    #: to ``UnifiedDiffusionLossComputer.compute`` (below, in ``_compute_losses_impl``),
    #: folded into ``components`` by that computer's dynamic-component loop, and
    #: weighted exactly once in ``_stack_components`` off the loss-weight SSOT.
    #: The computer's own ``reconstruction`` slot is a FALLBACK that stands down when
    #: the same term arrives via ``losses_dict`` (``_recon_fallback_name``), so a
    #: declared ``complex_l1`` / ``l1`` is folded, never double-counted -- which is
    #: why ``inline_losses`` is empty rather than carrying the fidelity term.
    #:
    #: Observed end-to-end on ``experiment_11_attention_none`` (2026-09-07), real
    #: builder output through the real computer: all 6 declared entries land in
    #: ``components``, finite, and ``sum(w * c) == total`` exactly --
    #: ``hfen`` (an image loss under ``output_domain: kspace``) arrives as a
    #: ``_BridgedLoss`` and contributes at its declared 0.3.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = True

    @staticmethod
    def _generator_accepts_time(gen: Any) -> bool:
        """Return ``True`` if the generator's forward accepts a time argument.

        Detect time-conditioning ONCE via signature introspection and dispatch
        the correct 1-arg vs 2-arg call unconditionally. Velocity-field
        subclasses (flow-matching, stochastic-interpolants, Schroedinger-bridge)
        use this instead of ``try: gen(x, t) except TypeError: gen(x)``: a
        genuine ``TypeError`` raised *inside* ``gen.forward`` (a shape/dtype/
        kwarg bug) must propagate, not be silently retried with a different
        signature (pitfall #9).
        """
        fwd = getattr(gen, "forward", gen)
        try:
            sig = inspect.signature(fwd)
        except (TypeError, ValueError):
            return True
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        return len(positional) >= 2

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
            device (Optional[torch.device]): Description.
        """
        super().__init__(env=env, device=device, **kwargs)

        #: Tall cascading-validation rows from the most recent
        #: ``validation_step``; the pipeline drains this (#697). Initialised
        #: EMPTY rather than left unset: an arm whose cascade is skipped (the
        #: image/latent-translation early-out) must publish "no rows", not a
        #: missing attribute the writer would have to `getattr`-guard --
        #: a defaulting `getattr` around a declared field is what silently
        #: disabled `rule_spatial_rank` and the SFC wrapper on #644.
        self._last_cascade_rows: list[dict[str, Any]] = []
        #: Declared-vs-effective timestep curriculum (#1296). Resolved at most
        #: ONCE per strategy -- the sampler runs every training step, and this
        #: answer is a pure function of a frozen config, so recomputing it there
        #: would be per-step config traversal for a constant (non-negotiable 9).
        #:
        #: Resolved on first use rather than here. The read compares
        #: `max_iterations` against the short-run bypass, and doing that in
        #: `__init__` puts an arithmetic comparison in the constructor: 59
        #: strategy tests build this class from a `MagicMock` config they never
        #: step, and every one of them turned into `TypeError: '<=' not
        #: supported between instances of 'MagicMock' and 'int'`. Deferring
        #: costs nothing (one `is None` per step) and keeps construction a
        #: construction.
        self._curriculum_state: CurriculumState | None = None

        #: One-shot latch for the linear-fallback clamp warning (#1295). The
        #: cascade runs once per validation BATCH, so an un-latched warning is
        #: thousands of identical lines in a single run.
        self._cascade_clamp_warned: bool = False

        # Initialize diffusion parameters from config (device already set by base class)
        # Extract diffusion parameters from strict v6.0 config (no fallbacks)
        try:
            num_timesteps = self.config.training.diffusion.timesteps
            if not isinstance(num_timesteps, int) or num_timesteps <= 0:
                raise ConfigurationError(
                    f"Invalid diffusion timesteps: {num_timesteps}. Must be positive integer."
                )
        except AttributeError as e:
            raise ConfigurationError(
                "Missing required config: training.diffusion.num_timesteps. "
                "See schema: src/config/schemas/training/diffusion.py"
            ) from e

        try:
            beta_schedule = self.config.training.diffusion.noise_schedule
            if not isinstance(beta_schedule, str):
                raise ConfigurationError(
                    f"Invalid noise schedule: {beta_schedule}. Must be string (e.g., 'linear', 'cosine')."
                )
        except AttributeError as e:
            raise ConfigurationError(
                "Missing required config: training.diffusion.noise_schedule. "
                "See schema: src/config/schemas/training/diffusion.py"
            ) from e

        # Initialize diffusion parameters using mixin.
        # beta_start/beta_end are forwarded explicitly: they are schema fields
        # (training/diffusion.py) with a beta_end > beta_start validator, but
        # nothing passed them through, so DiffusionScheduler silently used its
        # own 1e-4..0.02 defaults and the declared range was inert (pitfall #15).
        self.initialize_diffusion_parameters(
            num_timesteps=num_timesteps,
            beta_schedule=beta_schedule,
            device=self.device,
            beta_start=self.config.training.diffusion.beta_start,
            beta_end=self.config.training.diffusion.beta_end,
        )

        # Wire training.diffusion.prediction_type (pitfall #15: it had a schema
        # field, an enum and 162 declaring arms, but ZERO readers — an arm could
        # ask for v_prediction and silently train something else).
        #
        # The objective this strategy actually optimises is not configurable; it
        # is decided by the path. In compute_losses the loss target is the
        # injected latent noise for latent diffusion (ε-prediction, Rombach
        # Eq. 3) and the clean target_batch otherwise (x0- i.e. sample-
        # prediction). So rather than pretend the knob selects an objective, we
        # validate that the declaration AGREES with the objective that runs and
        # raise when it does not (CLAUDE.md #9 — never let a declared objective
        # silently differ from the trained one).
        self._validate_prediction_type()

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedDiffusionLossComputer(config=self.config, device=self.device)

        # Initialize k-space physics loss (FrequencyWeightedL1Loss)
        # Read alpha from configuration (Phase 5: configuration-driven)
        alpha = self.config.losses.reconstruction.frequency_weighted_l1_kspace_alpha
        self.frequency_loss_fn = FrequencyWeightedL1Loss(alpha=alpha)

        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        """Initialize diffusion-specific components and perform validation.

        ``BaseTrainingStrategy.__init__`` calls this hook before our ``__init__``
        body reaches :meth:`initialize_diffusion_parameters`, so on that first
        (premature) invocation the diffusion schedule does not exist yet and
        ``_bind_generator_reverse_schedule`` below died on ``self.num_timesteps``
        — how ``stage2_ldm_ulf_to_hf`` failed on 2026-07-25. The base justifies
        the early call with "all setup methods are idempotent", but the hazard is
        running *too early*, not twice. Defer to the explicit call at the end of
        ``__init__``, which runs once our invariants hold.
        """
        if not getattr(self, "_diffusion_initialized", False):
            return
        self._verify_strategy_config(expected_modes=("diffusion", "kspace_cold_diffusion"))

        # OPTIMIZATION (Phase 3C): Cache frequently-accessed config values
        self._cached_log_interval = self.config.logging.intervals.log
        self._cached_normalize_kspace = self.config.data.processing.enable_kspace_normalization
        # One warning per run, not one per step: 30,000 copies would bury it.
        # Guards all three "the batch arrived unnormalized" states, not just the
        # identity-scale one it was named for (#1211).
        self._warned_batch_unnormalized = False
        # ✅ SSOT: Direct access to model.in_channels (schema default exists)
        self._cached_in_channels = self.config.model.in_channels
        # ✅ SSOT: Direct access to optimization config (trust schema defaults)
        self._cached_gradient_clip_enabled = self.config.optimization.gradient.clip.enabled
        self._cached_gradient_clip_value = self.config.optimization.gradient.clip.value

        # --- [NEW LOGGING] ---
        gen = self.generator_model
        model_name = type(gen).__name__
        param_count = sum(p.numel() for p in gen.parameters())
        trainable_count = sum(p.numel() for p in gen.parameters() if p.requires_grad)

        self.logging_service.log_info(f"🚀 Active Diffusion Model: {model_name}")
        self.logging_service.log_info(
            f"📊 Model Parameters: {param_count:,} total, {trainable_count:,} trainable"
        )

        if hasattr(gen, "in_channels"):
            self.logging_service.log_info(f"   Input Channels: {gen.in_channels}")
        # ---------------------

        self._bind_generator_reverse_schedule()

        self._log_config_features(self.logging_service)

        # Initialize mask generator and data consistency for k-space cold diffusion
        # Using KspaceMixin setup logic
        # No `hasattr(...) else 1000` fallback: the guard at the top of this
        # method means the schedule is always initialised by the time we get
        # here, and defaulting would silently build k-space components on a
        # 1000-step schedule while the YAML declared another (pitfall #9).
        self.setup_kspace_components(num_timesteps=self.num_timesteps)

        # Initialize prior model if configured
        self._setup_prior_model()

        # Initialize validation step counter for image logging
        self.validation_step_count = 0

        # ✅ Signal to train.py that this strategy handles its own image logging.
        # Without this flag, train.py's fallback path runs generator(input, timesteps=0)
        # and saves images via a DIFFERENT reconstruction operator, overwriting the
        # strategy's correctly processed images (ifft_magnitude + RSS).
        self.logs_validation_images_in_step = True

        # ===== LOG CONFIGURED LOSSES AT STARTUP =====
        # Read the loss-weight SSOT, not ``enable_*``/``lambda_*`` pairs (#1918, #1919).
        # The 11-entry list that stood here mirrored ONE schema block by hand, so it
        # printed nothing for the 58 kspace_filling arms that declare their objective in
        # ``losses.image_losses`` / ``losses.kspace_losses`` — the banner claimed no
        # losses while seven trained. The formatter always emits a line, including for a
        # genuinely empty arm: absent is a state to report, never to infer (NN18).
        self._log_loss_objective("[DiffusionStrategy]")

    def _configured_estimation_method(self, default: str = "power_iter") -> str:
        """Resolve the sensitivity-estimation method for the runtime smaps fallback.

        Honors ``physics.coil_processing.estimation.method`` (the SSOT) instead of
        hardcoding ``power_iter`` (CLAUDE.md pitfall #15 — the advertised knob is
        read, not a silent no-op). The new method is a Literal in
        ``estimate_smaps``'s exact vocabulary, so it is always dispatchable.

        This DC/SENSE branch genuinely requires sensitivity maps, so a configured
        ``"none"`` (or an absent block) maps to ``default`` here rather than
        returning ``None`` and breaking downstream. The legacy
        ``physics.coil_sensitivity.estimation_method`` is intentionally NOT read:
        it uses a different vocabulary (``auto`` / ``low_rank`` / …) that
        ``estimate_smaps`` would reject.
        """
        config = getattr(self, "config", None)
        physics = config.physics if config is not None else None
        coil_processing = getattr(physics, "coil_processing", None)
        estimation = getattr(coil_processing, "estimation", None)
        method = getattr(estimation, "method", None)
        if method and method != "none":
            return method
        return default

    def _configured_estimation_kwargs(self) -> dict[str, Any]:
        """Resolve ``estimate_smaps`` sub-knobs from the configured estimation block.

        Companion to :meth:`_configured_estimation_method`: threads
        ``kernel_size`` / ``acs_size`` / ``eigen_threshold`` / ``maps_path`` from
        ``physics.coil_processing.estimation`` so they are honored at the runtime
        smaps fallback instead of being silent no-ops (CLAUDE.md pitfall #15).

        Returns an empty dict when the block is absent (bare ``__new__`` / no
        config), so ``estimate_smaps`` falls back to its own defaults — which are
        identical to the schema defaults (``kernel_size=6``), keeping the
        no-coil-block path bit-identical to the previous hardcoded call site.
        ``maps_path`` is omitted when ``None`` so the estimate_smaps default
        applies; methods that ignore a given kwarg (e.g. ``power_iter`` ignores
        ``acs_size``/``eigen_threshold``) simply do not read it.
        """
        # Sequential single getattrs (NOT a nested chain): tolerate a bare
        # ``__new__`` instance with no ``config`` attribute, and a config with
        # no ``physics`` block, exactly as before — see this method's docstring.
        _cfg = getattr(self, "config", None)
        _physics = getattr(_cfg, "physics", None)
        coil_processing = getattr(_physics, "coil_processing", None)
        estimation = getattr(coil_processing, "estimation", None)
        if estimation is None:
            return {}
        kwargs: dict[str, Any] = {}
        for key in ("kernel_size", "acs_size", "eigen_threshold", "maps_path"):
            val = getattr(estimation, key, None)
            if val is not None:
                kwargs[key] = val
        return kwargs

    # Max distinct ACS fingerprints to retain. During reverse sampling there is
    # one batch (1-2 entries); during training entries are bounded by this cap.
    _SMAPS_CACHE_MAX = 8

    def _estimate_smaps_cached(self, acs_kspace_t: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Estimate coil sensitivity maps from static ACS k-space, memoized.

        Coil sensitivities are a deterministic function of the ACS input, yet
        the SVD-class ESPIRiT estimate was re-run every step — most wastefully
        during multistep cold-diffusion reverse sampling, where the SAME ACS
        k-space is re-estimated once per sampler step (~28×) for one sample.

        We key an instance cache on a cheap content fingerprint of the input
        (shape + dtype + two reductions). A content key — unlike a subject-id
        key — can never alias maps across train/val contexts or different
        inputs, so it sidesteps the historical smaps context-leak hazard. The
        two reduction syncs are negligible next to the SVD they replace.
        """
        cache = getattr(self, "_smaps_cache", None)
        if cache is None:
            cache = self._smaps_cache = {}

        mag = acs_kspace_t.abs()
        fingerprint = (
            tuple(acs_kspace_t.shape),
            str(acs_kspace_t.dtype),
            round(float(mag.sum().item()), 3),
            round(float(mag.amax().item()), 6),
        )
        cached = cache.get(fingerprint)
        if cached is not None:
            return cached

        smaps = estimate_smaps(
            acs_kspace_t,
            method=self._configured_estimation_method(),
            acs_only=True,  # crop to the dense center so aliasing can't seed calibration
            **self._configured_estimation_kwargs(),
        ).detach()  # DETACH to prevent gradient flow through S-maps

        # Energy-preserving normalization: sum of squared magnitudes across coils = 1
        rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
        smaps = smaps / rss

        # Ensure matching resolution if needed (estimate_smaps returns full size)
        if smaps.shape[-2:] != (h, w):
            if smaps.is_complex():
                smaps_r = torch.nn.functional.interpolate(
                    smaps.real, size=(h, w), mode="bilinear", align_corners=False
                )
                smaps_i = torch.nn.functional.interpolate(
                    smaps.imag, size=(h, w), mode="bilinear", align_corners=False
                )
                smaps = torch.complex(smaps_r, smaps_i)
            else:
                smaps = torch.nn.functional.interpolate(
                    smaps, size=(h, w), mode="bilinear", align_corners=False
                )

        if len(cache) >= self._SMAPS_CACHE_MAX:
            cache.pop(next(iter(cache)))  # FIFO eviction
        cache[fingerprint] = smaps
        return smaps

    def _setup_prior_model(self) -> None:
        """Initialize prior model for conditioning if enabled."""
        if not hasattr(self.config, "data") or not hasattr(self.config.data, "prior_loading"):
            return

        prior_config = self.config.data.prior_loading
        if not prior_config.enabled:
            return

        self.logging_service.log_info(f"🔄 Loading prior model: {prior_config.source}")

        try:
            from spectramr.infrastructure.builders.leaf.model_builders import (
                GeneratorBuilder,
            )

            # ✅ SSOT: Direct access to model channels via leaf builder
            gen_builder = (
                GeneratorBuilder(self.config, self.device)
                .with_architecture(prior_config.source)
                .with_input_channels(self.config.model.in_channels)
                .with_output_channels(self.config.model.out_channels)
            )
            self.prior_model = gen_builder.validate().build()

            if prior_config.checkpoint_path:
                self.logging_service.log_info(
                    f"   Loading weights from: {prior_config.checkpoint_path}"
                )
                import os

                if os.path.exists(prior_config.checkpoint_path):
                    checkpoint = torch.load(prior_config.checkpoint_path, map_location=self.device)
                    state_dict = checkpoint.get("model_state_dict", checkpoint)

                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("module."):
                            new_state_dict[k[7:]] = v
                        else:
                            new_state_dict[k] = v

                    # ``strict=False`` stays: a prior legitimately loads
                    # partially (a head the diffusion arm does not use). What it
                    # must NOT do is accept ZERO matched keys — that is an
                    # architecture mismatch wearing a successful load's clothes,
                    # and it leaves a randomly-initialised "prior" that the log
                    # line below then reports as loaded and frozen.
                    incompatible = self.prior_model.load_state_dict(new_state_dict, strict=False)
                    expected = set(self.prior_model.state_dict())
                    matched = expected - set(incompatible.missing_keys)
                    if not matched:
                        raise RuntimeError(
                            f"Prior checkpoint {prior_config.checkpoint_path} shares "
                            f"NO parameter names with the '{prior_config.source}' "
                            f"prior ({len(expected)} expected keys, "
                            f"{len(incompatible.unexpected_keys)} unexpected in the "
                            "checkpoint). The prior would be random weights. Point "
                            "`data.prior_loading.checkpoint_path` at a checkpoint of "
                            "this architecture, or disable `prior_loading`."
                        )
                    self.logging_service.log_info(
                        f"   Matched {len(matched)}/{len(expected)} prior parameters"
                    )
                else:
                    # No silent fallback (non-negotiable 3). A declared prior
                    # checkpoint that is not on disk used to degrade to random
                    # weights and still log "✅ Prior model loaded and frozen",
                    # so a prior-conditioned arm silently became an unconditioned
                    # one and published as the former.
                    raise FileNotFoundError(
                        f"Prior checkpoint not found: {prior_config.checkpoint_path}. "
                        "`data.prior_loading.enabled: true` with a "
                        "`checkpoint_path` is a declaration that the prior IS "
                        "trained; running it with random weights would mislabel "
                        "the result. Fix the path, or set "
                        "`data.prior_loading.enabled: false`."
                    )

            self.prior_model.eval()
            for param in self.prior_model.parameters():
                param.requires_grad = False

            self.logging_service.log_info("✅ Prior model loaded and frozen")

        except (FileNotFoundError, RuntimeError, KeyError) as e:
            self.logging_service.log_error(f"❌ Failed to load prior model: {e}")
            raise RuntimeError(
                f"Prior model initialization failed: {e}. "
                "Check checkpoint path and model architecture."
            ) from e
        except Exception as e:
            # Handle generic exception like ModelCreationError if imports missing
            self.logging_service.log_error(f"❌ Failed to load prior model: {e}")
            raise

    def _effective_prediction_type(self) -> str:
        """Return the objective this strategy actually optimises.

        Mirrors the target selection in ``compute_losses``: the loss target is
        the injected latent noise when the arm is a latent-diffusion model
        (ε-prediction) and the clean ``target_batch`` otherwise (x0-prediction,
        spelled ``sample`` by :class:`PredictionType`). Keep the two in sync —
        this method is the declaration-side mirror of that branch.
        """
        return "epsilon" if self._is_latent_diffusion() else "sample"

    def _validate_prediction_type(self) -> None:
        """Raise when ``training.diffusion.prediction_type`` is not what runs.

        ``v_prediction`` is never implemented on this strategy, and an
        epsilon/sample declaration that contradicts the resolved path means the
        arm reports an objective it does not train.
        """
        if not self._is_latent_diffusion():
            # Scoped to the latent path deliberately. The pixel-space path
            # targets target_batch (x0/"sample"), yet 9 arms on this strategy
            # declare `epsilon` there — including DDPM baselines whose INTENT is
            # plainly epsilon-prediction. That disagreement is a real defect,
            # but resolving it means deciding whether the YAML or the path is
            # wrong, which is those cohorts' call, not a side effect of an LDM
            # review. Tracked in issue #641 (which lists all 9); until it is
            # settled this check defers rather than crashing them. Once it is,
            # drop this guard so the check covers both paths.
            return

        declared = self.config.training.diffusion.prediction_type
        declared = str(getattr(declared, "value", declared)).lower()
        effective = self._effective_prediction_type()
        if declared == effective:
            return
        raise ConfigurationError(
            f"training.diffusion.prediction_type={declared!r} is not the objective this "
            f"run would optimise. model_type={self.config.model.model_type!r} resolves to the "
            f"{'latent' if self._is_latent_diffusion() else 'pixel'}-space path, whose loss "
            f"target is {'the injected latent noise' if self._is_latent_diffusion() else 'the clean target image'}"
            f" — i.e. {effective!r}-prediction. Declare prediction_type: {effective} or move the "
            "arm to a strategy that implements the objective you want. "
            "(v_prediction is not implemented here.)"
        )

    def _bind_generator_reverse_schedule(self) -> None:
        """Bind a latent-diffusion generator's reverse schedule to the training SSOT.

        ``training.diffusion.{num_timesteps,noise_schedule}`` drives the FORWARD
        process (this strategy's ``q_sample``). ``LatentDiffusionGenerator`` owned a
        SECOND, private schedule (``LatentDiffusionGeneratorConfig.beta_schedule``,
        default ``"linear"``) that no YAML ever wired, and ``sample()`` — the
        validation path — inverted the forward process with it. An arm declaring
        ``noise_schedule: cosine`` therefore trained on a cosine trajectory and
        sampled with linear posterior coefficients: the reverse trajectory diverges
        and the decoded image collapses to ~black (stage-2 LDM shipped train_psnr≈32
        with val_psnr≈6 dB). Re-bind the model to the SSOT so there is one schedule.
        """
        if not self._is_latent_diffusion():
            return

        gen = (
            self.generator_model.module
            if hasattr(self.generator_model, "module")
            else self.generator_model
        )
        if not hasattr(gen, "diffusion"):
            return

        if not hasattr(gen, "set_diffusion_schedule"):
            # #9 — never sample on a schedule that silently disagrees with training.
            raise RuntimeError(
                f"{type(gen).__name__} owns an internal reverse-diffusion schedule but "
                "exposes no set_diffusion_schedule(); its sampler would invert the "
                "forward process with a different schedule than training.diffusion "
                "declares. Add the method or drop the private schedule."
            )

        # Hand over the forward process's ACTUAL betas, not just its name. The
        # two sides implement different `cosine` formulas (s=0 in
        # DiffusionScheduler vs Nichol-Dhariwal s=0.008 in base_diffusion), so a
        # name-only binding still left sampling inverting a trajectory training
        # never ran -- the very desync this method exists to kill.
        gen.set_diffusion_schedule(
            timesteps=self.num_timesteps,
            beta_schedule=self.beta_schedule,
            device=str(self.device),
            betas=self.scheduler.betas,
        )
        self.logging_service.log_info(
            f"🔗 Reverse schedule bound to training SSOT: "
            f"timesteps={self.num_timesteps}, noise_schedule={self.beta_schedule} "
            f"(betas handed over, not rebuilt)"
        )

    def _is_latent_diffusion(self) -> bool:
        """Check if current model is a latent diffusion model.

        Routes ALL LDM variants to the latent-space q_sample / loss path.
        The previous substring-only check (``"latent_diffusion" in model_type``)
        silently failed for ``latent_gaussian_diffusion`` because the shared
        ``"gaussian_"`` token breaks the substring (``"latent_gaussian_diffusion"``
        does **not** contain ``"latent_diffusion"`` as a contiguous substring).
        That regressed exp 32b with the May 2026
        ``hr_fakes [2,1,32,32] vs target [2,1,256,256]`` shape mismatch — the
        strategy ran the LDM through pixel-space diffusion, the model's
        internal ``encode_to_latent`` shrank the prediction to 32×32, and the
        loss compared against the un-encoded 256×256 target.
        """
        mt = str(self.config.model.model_type).lower()
        # Explicit list — matches the registry-dispatcher pattern (CLAUDE.md).
        ldm_types = {
            "latent_diffusion",
            "latent_gaussian_diffusion",
            "ldm",
        }
        if mt in ldm_types:
            return True
        # Fallback: any model_type that names BOTH "latent" and "diffusion"
        # (catches future LDM variants like ``latent_cold_diffusion``).
        return ("latent" in mt) and ("diffusion" in mt)

    def _cold_diffusion_prior_channel_range(self) -> tuple[int, int] | None:
        """Return ``(start, end)`` of channels to keep fully-sampled.

        Mirrors ``KSpaceUndersamplingProcess.prior_channel_range`` so the
        strategy's cold-diffusion q_sample respects the same cross-contrast
        prior contract. Reads from
        ``config.model.model_kwargs.prior_channel_range`` (the same place
        the model factory reads it from).
        """
        try:
            mk = self.config.model.model_kwargs
            pr = None
            if isinstance(mk, dict):
                pr = mk.get("prior_channel_range")
            elif mk is not None:
                pr = getattr(mk, "prior_channel_range", None)
            if pr is None:
                return None
            s, e = int(pr[0]), int(pr[1])
            if s < 0 or e <= s:
                return None
            return (s, e)
        except (AttributeError, TypeError, ValueError, IndexError):
            return None

    def _cold_degradation_source(self) -> str:
        """Which tensor the cold forward process degrades: ``input`` or ``target``.

        Reads ``training.diffusion.degradation_source`` (validated by the schema,
        default ``input``). Stamped into provenance so a run records which
        supervision regime it actually used (issue #536).
        """
        # Guarded DIRECT access, not getattr-fallback: the schema declares
        # ``training.diffusion`` (``| None``) and ``degradation_source`` (with a
        # default), so a getattr here would only mask a schema regression — which
        # is exactly what test_diffusion_reads_config_directly pins on this file.
        diffusion = self.config.training.diffusion
        source = "input" if diffusion is None else diffusion.degradation_source
        if source not in ("input", "target"):
            raise ConfigurationError(
                f"training.diffusion.degradation_source must be 'input' or "
                f"'target', got {source!r}."
            )
        return source

    def _batch_is_already_normalized(
        self,
        batch_data: Any,
        kspace_scale: Any,
        current_step: int,
    ) -> bool:
        """Did the DATA PIPELINE normalize this batch, or only publish a scale?

        This used to be ``kspace_scale is not None``, which conflates two very
        different batches. ``KSpaceNormalizationTransform`` publishes the scale it
        divided by; ``M4RawRepetitionDataset`` publishes an *identity*
        ``kspace_scale = 1.0`` precisely to say "the data leaves the dataset
        unnormalized" (so the published scale always matches the served tensor).
        Presence is therefore not evidence, and reading it as evidence skipped
        ``apply_kspace_normalization`` on exactly the batches that needed it: the
        model then trained on raw k-space, keeping the ~200x DC-vs-periphery
        dynamic range that ``enable_log_scaling`` exists to compress, and
        threading ``scale = 1.0`` downstream as though it were real.

        That is the failure mode ``experiment_11_attention_none`` recorded — its
        step-6 snapshot shows the batch reaching ``train_step`` at
        ``abs_max 2406.9 / std 8.67``, while this arm's own transform chain maps
        the same raw magnitudes to ``2.46 / 0.041``. A ``log1p``-compressed
        float32 tensor cannot exceed ``ln(FLT_MAX) ~ 88.7`` at all (and only
        ``~44`` through this module's ``sqrt(R^2 + I^2)``), so that batch was
        provably never compressed.

        Positive evidence, in order of authority:

        1. ``kspace_normalized`` — set True by the transform, False by a dataset
           that serves raw. An explicit answer; believe it.
        2. Otherwise a *non-identity* ``kspace_scale``. Something divided by it,
           so the tensors are not raw. An identity scale means nothing was
           divided, and re-running the divide is a no-op rather than a
           double-application, so falling through is safe.

        **Every** route to ``False`` announces itself, because each one means the
        declared transform's output is not reaching ``train_step`` and the
        strategy is silently compensating for the data layer. There are three,
        and only one of them used to warn:

        * an explicit ``kspace_normalized=False`` — the pipeline answered, and
          the answer contradicts the declaration;
        * no ``kspace_scale`` and no marker — the pipeline cannot say;
        * an identity ``kspace_scale`` and no marker — the pipeline cannot say.

        The first is the loudest signal available and was the *silent* one: it
        returned before either warning could fire, which is exactly how
        ``experiment_11_attention_none`` (whose dataset publishes ``False``)
        reached 30,000 iterations with a mosaic of DC blobs and nothing in the
        log (CLAUDE.md #3, #4; issues #1211, #1213).
        """
        # Read through the mapping protocol, NOT isinstance-dict + hasattr. A
        # TrainingBatch fails both of those legs -- it is not a mapping, and its
        # metadata is invisible to attribute lookup -- so the pair read a
        # published marker as absent and fell through to the "cannot say" branch
        # below. See ``read_batch_field`` for the measured access table.
        marker = read_batch_field(batch_data, "kspace_normalized")
        if marker is not None:
            # Collated per-subject markers arrive as a tensor/list of one value
            # per sample; reducing with ``all`` keeps a partially-normalized
            # batch (a pipeline bug of its own) out of the fast path, and avoids
            # ``bool(tensor([False, False]))`` raising.
            if torch.is_tensor(marker):
                normalized = bool(marker.all())
            elif isinstance(marker, (list, tuple)):
                normalized = all(bool(m) for m in marker)
            else:
                normalized = bool(marker)
            if normalized:
                return True
            self._warn_batch_reached_step_unnormalized(
                current_step,
                observed="published kspace_normalized=False",
                diagnosis=(
                    "the dataset's own marker says the tensors left it raw, and "
                    "KSpaceNormalizationTransform overwrites that marker with "
                    "True whenever it runs — so it did not run on the tensor "
                    "that reached train_step"
                ),
            )
            return False

        if kspace_scale is None:
            self._warn_batch_reached_step_unnormalized(
                current_step,
                observed="published NEITHER a kspace_scale NOR a 'kspace_normalized' marker",
                diagnosis=(
                    "the transform publishes both, so a batch carrying neither "
                    "was not produced by a chain containing it"
                ),
            )
            return False

        scale_tensor = (
            kspace_scale if torch.is_tensor(kspace_scale) else torch.as_tensor(kspace_scale)
        )
        scale_float = scale_tensor.float()
        if not torch.allclose(scale_float, torch.ones_like(scale_float)):
            return True

        self._warn_batch_reached_step_unnormalized(
            current_step,
            observed="published an IDENTITY kspace_scale (1.0) and no 'kspace_normalized' marker",
            diagnosis=(
                "an identity scale is the placeholder a dataset publishes to say "
                "it served raw k-space; KSpaceNormalizationTransform overwrites "
                "it with the scale it divided by, so it did not run"
            ),
        )
        return False

    def _warn_batch_reached_step_unnormalized(
        self,
        current_step: int,
        *,
        observed: str,
        diagnosis: str,
    ) -> None:
        """Announce, once per run, that the strategy is normalizing for the data layer.

        Rate-limited to a single emission: silence cost a 30,000-iteration run,
        and 30,000 warnings would cost the next one. One flag guards all three
        states, so the loop pays one bool test (CLAUDE.md #9).

        ``observed`` states what the batch actually carried and ``diagnosis`` why
        that implicates the transform, because the three states point at
        different wiring faults and a message that merged them would send the
        reader to the wrong place.
        """
        if self._warned_batch_unnormalized:
            return
        self._warned_batch_unnormalized = True
        self.logging_service.log_warning(
            f"[Step {current_step}] batch {observed}, while "
            f"data.processing.enable_kspace_normalization is true. {diagnosis}. "
            f"Treating the batch as UNNORMALIZED and normalizing in the "
            f"strategy — which keeps this run off raw k-space but also MASKS "
            f"the upstream defect, so do not read a healthy loss as evidence "
            f"the data layer is wired (issue #1213). Check that the transform "
            f"chain built for this dataset is the one the dataloader actually "
            f"uses. Logged once per run."
        )

    def _min_meaningful_timestep(self) -> int:
        """Lowest timestep worth training on, taken from the degradation operator.

        Delegates to ``KSpaceUndersamplingProcess.min_meaningful_timestep`` (the
        SSOT) so the training sampler and the reverse schedule agree by
        construction: the reverse loop's final, decisive step runs at this
        timestep, and training must cover it (issue #535).

        Falls back to 1 for non-cold paradigms and for generators exposing no
        ``kspace_process`` — for Gaussian diffusion t=0 genuinely is the clean
        sample and excluding it is correct.
        """
        if not self._is_cold_diffusion():
            return 1
        gen = (
            self.generator_model.module
            if hasattr(self.generator_model, "module")
            else self.generator_model
        )
        process = getattr(gen, "kspace_process", None)
        floor = getattr(process, "min_meaningful_timestep", None)
        if floor is None:
            return 1
        return int(floor())

    def _resolve_curriculum_once(self) -> CurriculumState:
        """The curriculum state, resolved on first use and cached thereafter.

        Warns -- once -- when a declared curriculum will not run. WARNING and
        not INFO: `LoggingService.setup` clamps every logger and handler to
        `logging.sinks.level`, and the attention_shootout arms -- the ones that
        declare a curriculum -- set `level: warning`, so an INFO line would be
        discarded on exactly the runs that need it. A knob that was set and did
        not take is not narration.

        The startup knob line carries the same fact for every run
        (`provenance.format_runtime_knobs`), which is the disclosure that does
        not depend on reaching a training step; this warning is the second copy
        for a reader watching the log rather than the banner.
        """
        state = self._curriculum_state
        if state is None:
            state = resolve_curriculum_state(self.config)
            self._curriculum_state = state
            if state.suppressed_by is not None:
                self.logging_service.log_warning(
                    f"[Curriculum] training.curriculum_start_timestep="
                    f"{state.start_timestep} / "
                    f"curriculum_ramp_rate={state.ramp_rate} "
                    f"is declared but will NOT be applied: "
                    f"{state.suppressed_by}. Timesteps are sampled from the "
                    f"full range for the whole run. This is the documented "
                    f"behaviour, not a fault -- it is warned about because "
                    f"nothing downstream distinguishes it from a curriculum "
                    f"that ran."
                )
        return state

    def sample_timesteps(self, batch_size: int, iteration: int = 0) -> torch.Tensor:
        """Sample random timesteps respecting curriculum cap and sampling strategy.

        [STABILIZATION FIX 3.3] Enforces the iteration-based curriculum cap.
        [STABILIZATION FIX 4.0] Supports weighted timestep sampling strategies
        to prevent 2× acceleration regression during multi-scale training.

        Strategies:
            - 'uniform': Equal probability across [1, T_max] (default, backward-compatible).
            - 'importance': Quadratic bias P(t) ∝ (T_max - t)^2 toward smaller timesteps.
              Makes low timesteps ~4× more likely, preserving easy-case performance.
            - 'linear_decay': Linearly decreasing probability P(t) ∝ (T_max - t).
            - 'high_t_emphasis': Linearly INCREASING probability P(t) ∝ t — the
              mirror of 'linear_decay'. Over-samples the HARD high-acceleration
              regime (high t) so the shared timestep-conditioned net keeps
              gradient on it instead of sacrificing it to the easy low-t task
              (the experiment_11 R8x/R32x negative-transfer collapse). Linear,
              not quadratic, so low-t (R2x) is rebalanced — not starved.

        Args:
            batch_size: Number of timesteps to sample.
            iteration: Current training iteration (for curriculum cap).

        Returns:
            Tensor of sampled timesteps, shape (batch_size,).
        """
        if self.generator_model.training:
            # 1. Retrieve curriculum config (Step 3.3 SSOT).
            #
            # These are DECLARED ``int | None`` / ``float | None`` fields on
            # TrainingConfigSchema, so the old ``hasattr(...) else 50`` guard was
            # always True and its fallback was dead code: an arm that simply did
            # not opt into the curriculum got ``start_t = None`` and died at the
            # ``start_t + iteration * rate`` below with
            # ``TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'``
            # on the FIRST training step. Both ldm_two_stage_ulf_to_hf stage-2
            # arms declare neither knob and have max_iterations=200000, so they
            # took this path. hasattr on a schema object tests DECLARATION, never
            # population — the guard for an optional field is ``is not None``.
            #
            # The schema documents the None semantics: "When None the strategy
            # defaults to no curriculum" — so None means sample the full range,
            # not some invented start.
            # [SHORT DEBUG RUN BYPASS] When max_iterations <=
            # SHORT_RUN_BYPASS_ITERATIONS, skip the curriculum cap entirely.
            # The curriculum is designed for long training (100k+ iters); in a
            # short smoke test the ramp never reaches meaningful acceleration
            # (R ~ 1.3x), making the model train on near-identity inputs. This
            # mirrors the diagnostic bypass in _run_diffusion_diagnostics.
            #
            # Both conditions are config-only, so they were resolved once in
            # __init__ (#1296) rather than re-derived on every training step,
            # and a curriculum suppressed here was warned about there instead
            # of vanishing silently.
            _state = self._resolve_curriculum_once()
            start_t = _state.start_timestep
            rate = _state.ramp_rate
            if not _state.effective:
                # Use full timestep range for meaningful degradation
                high = self.num_timesteps
            else:
                # 2. Calculate dynamic cap: T_max = min(T, T_start + iteration * rate)
                dynamic_max = int(start_t + iteration * rate)
                current_max = min(self.num_timesteps, dynamic_max)
                high = max(2, current_max)

            # 3. Timestep floor, derived from the DEGRADATION OPERATOR rather than
            # hardcoded to 1 (issue #535). The reverse loop evaluates its final,
            # decisive step at this floor — that step's full reveal writes 54-99%
            # of the reconstruction — so if training never draws it the time
            # embedding is extrapolating exactly where it matters most.
            #
            # The old ``high = max(2, high)`` plus ``randint(1, high)`` excluded
            # t=0 unconditionally, with the rationale "t=0 implies full sampling
            # (R=1)". That holds only when ``base_acceleration == 1``; the exp_11
            # cohort's step schedule realises R(0)=2, so t=0 was a genuine 2x
            # degradation the model never trained on while every validation
            # rollout ended there.
            t_low = self._min_meaningful_timestep()
            high = max(t_low + 2, high)

            # 4. Apply sampling strategy
            strategy = (
                self.config.training.timestep_sampling_strategy
                if hasattr(self.config.training, "timestep_sampling_strategy")
                else "uniform"
            )
            if strategy not in _VALID_SAMPLING_STRATEGIES:
                raise ConfigurationError(
                    f"Unknown timestep_sampling_strategy: {strategy!r}. "
                    f"Must be one of {_VALID_SAMPLING_STRATEGIES}."
                )

            # Every weighted strategy below draws over ``[t_low, high)`` with a
            # strictly positive weight there, so no timestep the reverse loop can
            # evaluate has zero training mass (issue #535).
            t_range = torch.arange(t_low, high, device=self.device, dtype=torch.float32)

            def _draw(weights: torch.Tensor) -> torch.Tensor:
                weights = weights / weights.sum()
                idx = torch.multinomial(weights, batch_size, replacement=True)
                return (idx + t_low).long()

            if strategy == "importance":
                # Quadratic bias: P(t) ∝ (T_max - t)^2
                # Low timesteps (easy, ~2× accel) are ~4× more likely than
                # high timesteps (hard, ~32× accel), preventing catastrophic
                # forgetting of easy-case weights.
                return _draw((float(high) - t_range) ** 2)

            if strategy == "linear_decay":
                # Linear bias: P(t) ∝ (T_max - t)
                return _draw(float(high) - t_range)

            if strategy == "high_t_emphasis":
                # Linear bias toward HIGH timesteps: P(t) ∝ t. Mirror of
                # 'linear_decay'. Counters the experiment_11 negative transfer
                # where the shared net abandons high-t (R8x/R32x) for the easy
                # low-t (R2x) task: over-sampling high t gives the hard regime
                # gradient priority (equivalent in expectation to up-weighting
                # the high-t loss, but without the L1-pulls-toward-blank failure
                # of scaling the fidelity term itself).
                #
                # WARNING: pure P(t) ∝ t OVER-corrects — once the curriculum cap
                # opens the full range (~iter 4800 here), t=1,2 (R2x) receive
                # <2% of draws and the net FORGETS the low-t band that R2x
                # validation reverse-samples, so R2x val PSNR degrades over
                # training and inverts below R32x. Prefer 'balanced_high_t'.
                #
                # ``t_range - t_low + 1`` rather than ``t_range``: at t_low=0 a
                # bare P(t) ∝ t gives the floor timestep EXACTLY zero mass, which
                # is the untrained-final-step bug this change removes. The shift
                # keeps the monotone increase and the shape above the floor.
                return _draw(t_range - t_low + 1.0)

            if strategy == "balanced_high_t":
                # 'high_t_emphasis' with a uniform floor: keep high-t gradient
                # priority WITHOUT starving low t. P(t) = (1-eps)·(t/Σt) + eps·(1/N)
                # so every timestep is guaranteed >= eps/N of the mass (no low-t
                # forgetting) while (1-eps) of the mass still tilts toward high t.
                # Fixes the experiment_11 R2x accel-inversion (pure P(t) ∝ t
                # abandoned the low-t / low-R band that R2x val reverse-samples).
                # Measured on the exp_11 ladder once the cap opens:
                # P(t>=16)=0.611 vs P(t<=3)=0.044 — versus 0.111 under uniform.
                # The floor MITIGATES low-t forgetting; it does not remove it.
                eps = _BALANCED_HIGH_T_UNIFORM_FLOOR
                shifted = t_range - t_low + 1.0
                emphasis = shifted / shifted.sum()
                uniform = torch.full_like(shifted, 1.0 / shifted.numel())
                return _draw((1.0 - eps) * emphasis + eps * uniform)

            # strategy == "uniform" — only valid remaining case
            return torch.randint(t_low, high, (batch_size,), device=self.device).long()

        # During evaluation, use full range or specific steps from config
        return super().sample_timesteps(batch_size)

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward diffusion process (diffuse data).

        Corrupts the input `x_start` at timestep `t` using noise or degradation mask.
        Supports both Gaussian diffusion (additive noise) and Cold Diffusion (masking).

        Args:
            x_start: Clean input data (B, C, H, W).
            t: Timesteps (B,).
            noise: Optional noise tensor. If None, generated based on schedule.

        Returns:
            Corrupted data at timestep t.
        """
        if self._is_cold_diffusion():
            # Cold Diffusion: Degradation is undersampling (masking)

            # Safety check: Ensure x_start is 4D (B, C, H, W)
            if x_start.ndim == 5 and x_start.shape[-1] == 1:
                x_start = x_start.squeeze(-1)

            # If noise (mask) is provided, use it. Otherwise generate it.
            if noise is None:
                image_shape = (x_start.shape[-2], x_start.shape[-1])
                mask = self.mask_generator.generate_batch_masks(
                    batch_size=x_start.shape[0],
                    timesteps=t,
                    image_shape=image_shape,
                )

                mask = mask.to(x_start.device)
                mask = self.mask_generator.expand_mask_to_channels(mask, x_start.shape[1])

                noise = mask

            # Robust shape handling for broadcasting
            if x_start.shape != noise.shape:
                if noise.shape[1] == 1 and x_start.shape[1] > 1:
                    noise = noise.expand(-1, x_start.shape[1], -1, -1)

                if x_start.shape[2:] != noise.shape[2:]:
                    noise = F.interpolate(noise, size=x_start.shape[2:], mode="nearest")

            # Apply degradation: x_t = x_0 * mask_t
            try:
                result = x_start * noise
            except RuntimeError as e:
                logger.error(
                    f"[q_sample] Shape mismatch! x_start={x_start.shape}, noise={noise.shape}"
                )
                logger.error(f"[q_sample] Timesteps shape: {t.shape}")
                raise RuntimeError(
                    f"Diffusion shape mismatch: x={x_start.shape}, noise={noise.shape}. {e}"
                )

            # Cross-contrast prior: when ``model.model_kwargs.prior_channel_range``
            # is set (e.g. ``[0, 8]`` for the cross-contrast cold diffusion
            # arm where ``[T1||target]`` are concatenated along the coil
            # axis), the listed channels carry a fully-sampled prior — they
            # must NOT be masked. The model-internal
            # ``KSpaceUndersamplingProcess.q_sample`` already honours this,
            # but the strategy's own cold-diffusion q_sample (this branch)
            # was masking every channel uniformly, which is why the
            # rendered "noisy input" never looked accelerated: T1's
            # untouched energy dominated the RSS.
            prior_range = self._cold_diffusion_prior_channel_range()
            if prior_range is not None:
                s, e = prior_range
                C = result.shape[1]
                if 0 <= s < e <= C:
                    result[:, s:e] = x_start[:, s:e]

            return result

        return super().q_sample(x_start, t, noise)

    def _save_debug_snapshots(
        self,
        tensors: dict[str, torch.Tensor],
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Save visual snapshots of tensors for debugging.

        Delegates to the paradigm-agnostic
        :meth:`BaseTrainingStrategy.save_debug_snapshot` so the same
        textual / JSON / image-preview format is used across all
        strategies. Tagged ``diffusion_step`` to distinguish from the
        base class's ``first_steps`` auto-snapshot.

        **Not the primary path any more.** ``_prepare_diffusion_inputs`` now
        DECLARES its model input (``_declare_model_input``) and the wrapper
        emits it, so a subclass overriding ``_compute_losses_impl`` cannot drop
        the ``diffusion_step`` snapshot its ``first_steps`` artifact promises.
        This method remains a valid way to satisfy that contract directly:
        its tag equals ``snapshot_model_input_tag``, so
        :meth:`BaseTrainingStrategy.save_debug_snapshot` marks the step's
        model-input requirement met on the way through. Prefer declaring —
        one emitter, and the wrapper owns the "did anything arrive" check.

        Args:
            tensors: Key -> tensor to capture.
            step: Current training step.
            extra: JSON-serialisable provenance recorded alongside the tensors
                -- for the cold path, which tensor the forward process degraded.
                The tag's snapshot is the only artifact showing the *degraded*
                model input, so what it was derived from belongs in it.
        """
        try:
            # Mark only the keys that genuinely contain k-space data. ``smaps``
            # is image-domain even though its channels are real-stacked
            # complex — see findings booklet 2026-05-05 VIS-1.
            #
            # That exclusion is necessary but not sufficient: ``model_input`` is
            # ``cat([noisy_images, smaps])``, so keeping the standalone ``smaps``
            # key out of the k-space set still leaves the SAME maps inside
            # ``model_input`` being IFFT'd as if they were a spectrum.
            # ``channel_segments`` splits that tensor back into its two domains
            # for rendering; it returns ``{}`` when no maps were concatenated,
            # so arms without smaps conditioning render exactly as before.
            self.save_debug_snapshot(
                tensors,
                step=step,
                tag="diffusion_step",
                in_kspace_keys={
                    "input",
                    "target",
                    "noisy_images",
                    "noisy_kspace",
                    "model_input",
                },
                channel_segments=self._model_input_channel_segments(tensors),
                extra=extra,
                # This tag equals ``snapshot_model_input_tag``, so this method
                # satisfies the model-input contract on the way through and owes
                # it the same naming the declaring path gives -- through the
                # same rule, not a second copy of it (#697).
                model_input_key=cold_model_input_key(tensors),
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to save debug snapshot: {e}")

    def _adversarial_loss_computer(self):
        """The GAN loss computer used for the DISCRIMINATOR step only.

        ``self.loss_computer`` is the diffusion one and has no
        ``compute_discriminator_loss`` -- correctly, since the generator's
        objective here is a denoising one. But the critic's objective is a GAN
        objective whatever the generator is doing, so the D step needs its own
        computer. Built lazily and cached, so a diffusion arm WITHOUT a
        discriminator never constructs it.

        Reuses ``UnifiedGANLossComputer`` rather than reimplementing the
        real/fake terms, so the R1 schedule, gradient penalty and pre-weighting
        behave identically to a GAN arm (non-negotiable 17: one owner for the
        adversarial objective).
        """
        computer = getattr(self, "_adv_loss_computer", None)
        if computer is None:
            from spectramr.models.losses.computers import UnifiedGANLossComputer

            computer = UnifiedGANLossComputer(self.config, self.device)
            self._adv_loss_computer = computer
        return computer

    def _fake_for_critic(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        iteration: int,
        batch_data: Any = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
        """The generator sample the discriminator scores, its conditioning, and its real.

        The D step used to call ``self.env.generator(input_batch)`` -- ONE
        positional argument, no timestep, no ``q_sample``. That is not the
        diffusion forward. A denoiser whose ``forward`` requires ``timesteps``
        raises ``TypeError``; one that defaults it runs at an undefined
        timestep. Either way the critic was scoring something the generator
        never produces, while the generator's own loss path went through
        ``_prepare_diffusion_inputs`` -> ``_forward_through_model`` ->
        ``_extract_and_fix_output``. Training a critic on one distribution and
        consulting it about another makes its score meaningless.

        So this runs the SAME sequence, through the SAME helpers (non-negotiable
        17 -- each step keeps its one owner; only the orchestration is repeated,
        because the loss path's copy is fused with LDM target-override logic
        that has no meaning for a critic).

        ``timesteps`` are drawn afresh: the critic must see the generator's
        output distribution across the schedule, not at one fixed t.

        Returns ``(fake, critic_cond, real)``. ``critic_cond`` is the ``t`` and
        ``contrast_idx`` THIS sample was drawn at, which both this method and
        the loss path already computed and this method used to discard (#1931).
        It must be returned rather than recomputed by the caller: ``t`` is
        sampled here, so a caller drawing its own would label the sample with a
        timestep it was not generated at.

        The same payload conditions the REAL side of the D step. That is
        correct for a label-style conditioning -- the critic must separate real
        from fake *within* a condition, and giving the two sides different
        labels would let it win by reading the label instead of the image. Note
        this makes ``t`` a difficulty label on a clean-domain reconstruction,
        NOT a shared corruption level: unlike Diffusion-GAN, the real sample is
        not re-noised to ``t``.

        ``real`` is the third return for the same reason ``critic_cond`` is the
        second: this method PREPARES it and used to throw it away. The
        preparation rebinds ``target_batch`` -- ``apply_kspace_normalization``
        when the batch reaches the step unnormalized, the 5D->4D flatten, and
        ``_extract_and_fix_output``'s complex/real alignment and channel
        truncation. The caller's ``target_batch`` has been through none of
        that, so scoring this ``fake`` against it hands the critic a pair that
        differs by a transform rather than by realism -- silently, since both
        keep the same shape and dtype. One owner for the pair (non-negotiable
        17): whoever builds the fake also names the real it was built against.
        """
        (
            input_batch,
            target_batch,
            model_input,
            noisy_images,
            timesteps,
            is_cold_diffusion,
            is_latent_diffusion,
            mask,
            scale,
            contrast_idx,
            noise,
        ) = self._prepare_diffusion_inputs(
            input_batch, target_batch, target_batch.size(0), iteration, epoch, batch_data
        )
        model_input = self._maybe_condition_on_input(
            model_input,
            input_batch=input_batch,
            noisy_images=noisy_images,
            is_cold_diffusion=is_cold_diffusion,
            is_latent_diffusion=is_latent_diffusion,
        )
        gen_kwargs = self._build_generator_kwargs(
            is_cold_diffusion=is_cold_diffusion,
            is_latent_diffusion=is_latent_diffusion,
            input_batch=input_batch,
            target_batch=target_batch,
            batch_data=batch_data,
            mask=mask,
        )
        predicted_output = self._forward_through_model(
            model_input=model_input,
            timesteps=timesteps,
            is_latent_diffusion=is_latent_diffusion,
            gen_kwargs=gen_kwargs,
            contrast_idx=contrast_idx,
        )
        _eafo_target = noise if (is_latent_diffusion and noise is not None) else target_batch
        fake, aligned_target = self._extract_and_fix_output(
            predicted_output=predicted_output,
            target_batch=_eafo_target,
            scale=scale,
            input_batch=input_batch,
            is_cold_diffusion=is_cold_diffusion,
            mask=mask,
        )
        # On the LDM path ``_extract_and_fix_output`` was handed ``noise``, so
        # what it returns is an aligned NOISE tensor, not data. The critic's
        # real is never noise -- fall back to the prepared ``target_batch``,
        # which is still the rebound one, just without that method's alignment.
        critic_real = (
            target_batch if (is_latent_diffusion and noise is not None) else aligned_target
        )
        return fake, self._critic_conditioning(timesteps, contrast_idx), critic_real

    @staticmethod
    def _align_for_critic(
        fake: torch.Tensor,
        real: torch.Tensor,
        critic_takes_complex: bool = False,
        *,
        generator_domain: str | None = None,
        critic_domain: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Put both critic inputs on one device, in the critic's declared domain.

        The GAN strategy's D step carries the device and realification guards;
        the diffusion copy was written without them. Both matter more here, not
        less: a diffusion arm may well predict in k-space, and a real-valued
        conv critic fed a complex tensor either raises or silently drops the
        imaginary half.

        **This method used to refuse the domain half of that job**, on the
        grounds that inserting an ``ifft2c`` would be an invisible substitution.
        That reasoning held only while the critic's domain was unknown -- a
        transform chosen from a *suspicion* is exactly the silent substitution
        non-negotiable 3 forbids. It no longer applies: the generator declares
        ``losses.policy.output_domain`` and the critic declares ``input_domain``
        on its own registration, so when the two differ the transform is
        *stated by the configuration*, not guessed from tensor shape. That is
        the same declared conversion ``mixins/kspace.py:45-105`` already
        performs for model inputs. Undeclared on either side still converts
        nothing -- 18 of 19 registered critics declare no domain, and their
        behaviour is byte-identical.

        Domain first, representation second: complex tensors are stacked as
        (real, imag) on the channel axis -- the representation
        ``GANTrainingStrategy._train_discriminator_step`` uses -- and doing that
        before an FFT would transform a tensor whose channels no longer mean
        real and imaginary parts.

        ``critic_takes_complex`` is read off the critic's own registration
        (``accepts_complex``) rather than guessed. It matters because
        realification is applied to each side INDEPENDENTLY, and on this path
        the two sides do not arrive alike: ``fake`` comes from the generator
        already real and INTERLEAVED (``[R1,I1,R2,I2,...]``), while ``real`` is
        ``target_batch``, which may still be complex and would be stacked as a
        BLOCK (``[R1,R2,...,I1,I2,...]``). A critic fed that pair separates real
        from fake on channel order alone. Rather than change the block
        convention -- ``GANTrainingStrategy`` shares it, and 25 live GAN arms
        depend on it -- a critic that declares it consumes complex is handed
        both sides untouched and owns the conversion itself. Note the CONVERTED
        path escapes that asymmetry for free: both sides pass through complex
        and come out block.

        The conversion itself lives in ``critic_domain.to_critic_input``, which
        the G step calls through ``DomainAdaptedCritic`` -- one owner, because a
        critic converted on the D step and not the G step would score images
        while training and k-space while scoring the generator, which is worse
        than converting on neither (non-negotiable 17).

        **One ``generator_domain`` governs BOTH sides, and that is not an
        oversight to be "fixed" by giving ``real`` a domain of its own.** The
        reading that invites it -- ``real`` is data, so its domain is a *data*
        fact and belongs to ``data.domain.output`` -- was true of the tensor
        this parameter used to receive and is false of the one it receives now.
        Since the caller was corrected to pass ``prepared_target``, ``real`` is
        the tensor the reconstruction losses are computed against, and that
        tensor is in ``losses.policy.output_domain`` **by construction**:
        ``DifferentiableFourierBridge.forward`` (``physics_losses.py``) projects
        ``k_pred`` AND ``k_target`` through the same ``_kspace_to_image``, so the
        whole loss pipeline already requires prediction and target to share one
        domain. Resolving ``real`` from a second field would be a second
        resolver for a quantity that already has an owner -- both would compute
        a plausible answer, both would have passing tests, and the divergence
        would surface as a wrongly-transformed critic input rather than an error
        (non-negotiable 17). ``test_align_for_critic_uses_one_domain_for_both``
        pins this; it goes red if a second resolver is introduced here.

        Args:
            fake: the generator's output for this step.
            real: the target the ``fake`` was prepared against -- the *prepared*
                target, not the raw batch, and so in ``generator_domain`` by the
                construction described above. Not an independently-resolved
                domain: see the paragraph above before adding one.
            critic_takes_complex: the critic's ``accepts_complex`` capability.
            generator_domain: ``losses.policy.output_domain`` -- the domain of
                BOTH ``fake`` and ``real``, for the reason stated above.
            critic_domain: the critic's declared ``input_domain``, or None.

        Returns:
            ``(fake, real)`` on one device, in the critic's declared domain.
        """
        if fake.device != real.device:
            fake = fake.to(real.device)
        return (
            to_critic_input(
                fake,
                from_domain=generator_domain,
                to_domain=critic_domain,
                takes_complex=critic_takes_complex,
            ),
            to_critic_input(
                real,
                from_domain=generator_domain,
                to_domain=critic_domain,
                takes_complex=critic_takes_complex,
            ),
        )

    @staticmethod
    def _critic_component_name(config: Any) -> str | None:
        """The registered name of this arm's critic, or ``None`` (#1931).

        Thin delegate. The one owner moved to
        ``models.losses.critic_domain.critic_component_name`` in #1921, because
        ``models/losses/computers/unified_gan.py`` needs the same fact and may
        not import ``infrastructure/`` (non-negotiable 5). This staticmethod
        stays as the strategy-side spelling; it holds no logic of its own.

        **The one owner of this resolution** (non-negotiable 17). Two callers
        need the same fact -- ``_critic_accepts_complex`` (does it take complex
        input) and ``_critic_conditioning`` (may it be sent t/contrast) -- and
        they had drifted into two spellings: guarded direct access in one,
        ``getattr(getattr(config, "model", None), "discriminator_component",
        None)`` in the other.

        The getattr spelling is not merely untidy. It answers ``None`` for a
        field that was RENAMED exactly as for one legitimately absent, so a
        schema drift silently reports "this arm has no critic" and the caller
        degrades instead of raising -- non-negotiable 3's silent fallback, and
        the defect ``2e765cc4b`` fixed on the other reader while this one kept
        it. It is also the shape #368 banned and
        ``test_diffusion_reads_config_directly_not_via_getattr_fallback`` pins.

        ``model`` and ``discriminator_component`` are declared schema fields and
        are read directly, so a rename raises ``AttributeError``. Only the two
        states the schema really allows are absorbed: an arm with no model block
        (``model is None``) and one with no critic configured
        (``discriminator_component is None``, or a blank name).
        """
        return critic_component_name(config)

    @staticmethod
    def _critic_accepts_complex(config: Any) -> bool:
        """Whether the configured critic declares ``accepts_complex``.

        Reads the ``ModelCapabilities`` dataclass -- the single owner of every
        capability flag since #1916 -- so either spelling is correct now. This
        one is kept because it reads the owner directly rather than through a
        helper.

        This docstring used to FORBID ``model_supports``, and the ban was
        right at the time: that helper looked the flag up with
        ``entry.get(capability)`` at the *top level* of the registry entry,
        where the nested flags were simply absent. The two readers shared no
        models at all. Measured over 588 entries at the fix, nested vs
        top-level: ``accepts_complex`` 20 vs 0, ``requires_paired_data``
        71 vs 0, ``expects_real_imag_interleaved`` 11 vs 0,
        ``supports_contrast_conditioning`` 0 vs 28 -- 130 disagreements, now
        0. The top-level surface is deleted and
        ``test_no_entry_carries_a_top_level_capability_key`` keeps it deleted.
        """
        return critic_accepts_complex(config)

    def _critic_for_domain(self, critic: Any) -> Any:
        """The critic, wrapped only if the G step must convert its input (#1920).

        Returns the critic ITSELF whenever no conversion is required, which is
        every arm that leaves either side undeclared and every arm whose two
        declarations agree. That path is byte-identical to the behaviour before
        this seam existed -- it matters, because it is the path all 58
        kspace_filling arms take.

        When the declarations differ, the wrapper applies exactly the transform
        ``_align_for_critic`` applies on the D step, from the same owner. The
        two must not diverge: a critic converted on one step and not the other
        sees images while training and k-space while scoring the generator.

        Args:
            critic: ``discriminator_model``, possibly None.

        Returns:
            The critic, or a :class:`DomainAdaptedCritic` wrapping it.
        """
        if critic is None:
            return None
        config = self.env.config if self.env else self.config
        generator_domain = generator_output_domain(config)
        critic_domain = critic_input_domain(config)
        if resolve_conversion(generator_domain, critic_domain) is None:
            return critic
        return DomainAdaptedCritic(
            critic,
            from_domain=generator_domain,
            to_domain=critic_domain,
            takes_complex=self._critic_accepts_complex(config),
        )

    def _critic_conditioning(
        self,
        timesteps: torch.Tensor | None,
        contrast_idx: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """The conditioning payload for this arm's critic, or ``{}`` (#1931).

        Gated on the REGISTRY, never on the critic object. The decision is
        ``does the configured critic name declare
        supports_contrast_conditioning`` -- read from the model registry by the
        name in ``model.discriminator_component.name``, which is the same fact
        the Tier-1 audit check reads when it decides whether a multi-contrast
        arm may carry this critic. One owner, so audit and runtime cannot
        disagree.

        **Deliberately not ``_callable_accepts_kwarg`` or any other signature
        probe.** Introspection-guarded forwarding is precisely the mechanism
        that dropped ``contrast_idx`` at the generator seam and made #1931
        necessary: it turns "this critic cannot take the payload" into silent
        unconditioned training instead of an error. Gating on the declaration
        makes the flag load-bearing at runtime -- a critic that declares it and
        cannot accept the kwargs raises on step 1.

        Returns ``{}`` for every critic that does not declare the flag, which
        reproduces the unconditioned call byte-for-byte. That is the path all
        but one arm in the corpus take.
        """
        config = self.env.config if self.env else self.config
        name = self._critic_component_name(config)
        if name is None:
            return {}
        # ``model_supports`` reads the nested ``ModelCapabilities`` dataclass,
        # the one owner of every capability flag since #1916. This comment
        # said the opposite until that fix -- the flag really did live only at
        # the top level, and the nested reader answered None for all 28
        # critics declaring it -- but that split is deleted, so the two
        # spellings now return the same answer. Still the reader
        # ``_contrast_aware_critics`` uses, so the audit and this gate cannot
        # disagree about one critic.
        from spectramr.models.registry import model_supports

        if not model_supports(name, "supports_contrast_conditioning"):
            return {}
        # A declared-aware critic with nothing to condition ON is a
        # configuration error, not a reason to score unconditioned: the arm
        # asked for a conditioned critic and would otherwise train a whole run
        # without saying the conditioning was missing (non-negotiable 3). Let
        # the critic own the raise so there is exactly one message.
        return {"timesteps": timesteps, "contrast_idx": contrast_idx}

    def _wire_critic_smaps(self, discriminator: Any) -> None:
        """Give a bridging critic the coil sensitivities of the current batch.

        A PROVIDER, not a value. ``_current_smaps`` is populated inside
        ``_prepare_diffusion_inputs``, which the D step reaches through
        ``_fake_for_critic`` and the G step reaches inside the base closure --
        i.e. *after* this wiring runs, on both paths. Pushing a tensor here
        would hand the critic the previous step's maps, or none at all on the
        first step; a callable resolved at forward time is correct in both
        closures with one wiring point.

        A critic that does not expose the seam is left alone, so this is a
        no-op for every existing discriminator.
        """
        setter = getattr(discriminator, "set_smaps_provider", None)
        if setter is None:
            return
        setter(self._select_batch_compatible_smaps)

    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generator step, preceded by N discriminator steps when one is configured.

        ADDITIVE. With no discriminator this returns exactly what
        ``BaseTrainingStrategy.train_step`` returns, so every existing diffusion
        arm is bit-for-bit unchanged -- that is what makes the adversarial term
        an opt-in extra rather than a change of paradigm.

        Before this, a discriminator attached to a diffusion arm was BUILT and
        CONSULTED (the class docstring advertises "Adversarial Loss: Optional
        GAN-style discriminator loss", and ``_compute_losses_impl`` passes
        ``discriminator=`` to the loss computer) but never UPDATED: the class did
        not inherit ``AdversarialMixin``, defined no discriminator step, and had
        no ``train_step`` of its own. The critic stayed at its initialisation and
        fed the generator a meaningless signal -- the pitfall-#16 facade shape.
        """
        discriminator = getattr(self.env, "discriminator", None)
        if discriminator is None:
            return super().train_step(
                batch, epoch, input_batch=input_batch, target_batch=target_batch, **kwargs
            )

        # A discriminator without ``losses.gan`` cannot be trained: there is no
        # declared cadence and no declared adversarial weight.
        # ``_resolve_disc_updates`` raises with that message rather than
        # defaulting to 1, because inventing either would be the silent
        # substitution non-negotiables 3 and 8 forbid.
        config = self.env.config if self.env else self.config
        num_d_updates = _resolve_disc_updates(config)
        critic_takes_complex = self._critic_accepts_complex(config)
        # Both halves of the #1920 seam, resolved once per step rather than per
        # closure: the generator's declared output domain and the critic's
        # declared input domain. Either being None means no conversion.
        generator_domain = generator_output_domain(config)
        critic_domain = critic_input_domain(config)
        self._wire_critic_smaps(discriminator)

        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)
        input_batch = self._to_device(input_batch)
        target_batch = self._to_device(target_batch)
        iteration = resolve_loop_iteration(self)

        self._last_step_metrics = {}

        def make_d_closure():
            def d_closure() -> torch.Tensor:
                # The critic trains on its own step, so its parameters need grad
                # here; the generator step re-freezes them.
                if isinstance(discriminator, nn.Module):
                    discriminator.requires_grad_(True)
                with torch.no_grad():
                    fake, critic_cond, prepared_target = self._fake_for_critic(
                        input_batch, target_batch, epoch, iteration, kwargs.get("batch_data")
                    )
                fake, real = self._align_for_critic(
                    fake,
                    # The target THIS fake was prepared against, not the raw
                    # one this closure closed over: the preparation may have
                    # normalized, flattened or re-aligned it, and scoring
                    # across that gap is separable on scale alone.
                    prepared_target,
                    critic_takes_complex,
                    generator_domain=generator_domain,
                    critic_domain=critic_domain,
                )
                d_out = self._adversarial_loss_computer().compute_discriminator_loss(
                    real=real,
                    fake=fake,
                    discriminator=discriminator,
                    epoch=epoch,
                    iteration=iteration,
                    # The t/contrast the fake was actually drawn at, carried out
                    # of ``_fake_for_critic`` rather than re-derived (#1931).
                    # ``discriminator`` here is the BARE module -- the D step
                    # converts tensors, not the callable -- which is what keeps
                    # R1's ``isinstance(discriminator, nn.Module)`` guard from
                    # silently zeroing the penalty. See ``DomainAdaptedCritic``.
                    critic_cond=critic_cond,
                )
                d_total = d_out.total if hasattr(d_out, "total") else d_out
                with torch.no_grad():
                    self._last_step_metrics["d_total_loss"] = d_total.detach()
                    for k, v in getattr(d_out, "components", {}).items():
                        if torch.is_tensor(v):
                            self._last_step_metrics[f"d_{k}"] = v.detach()
                return d_total

            return d_closure

        base_configs = super().train_step(
            batch, epoch, input_batch=input_batch, target_batch=target_batch, **kwargs
        )
        g_config = base_configs[-1]

        def g_closure() -> torch.Tensor:
            """The base generator closure, with the critic frozen for its duration.

            ``GANTrainingStrategy._train_generator_step`` does this and the
            diffusion path did not. It is not merely a saved backward: with the
            generator-side adversarial term now actually computed,
            ``discriminator(pred)`` runs inside the G step, so D's parameters
            accumulate gradient from the GENERATOR's objective. Under gradient
            accumulation those survive to D's own ``optimizer.step()`` -- the
            "G trains D" leak. ``opt_d.zero_grad`` at the next D step hides it
            whenever accumulation is off, which is what makes it the kind of bug
            that shows up only in the configuration nobody smoke-tests.

            Freezing the PARAMETERS does not cut the graph: the adversarial term
            still reaches ``pred`` through D, which is the whole point. The D
            step re-enables grad at its entry.
            """
            if isinstance(discriminator, nn.Module):
                discriminator.requires_grad_(False)
            return g_config["closure"]()

        return assemble_adversarial_step_configs(
            num_d_updates=num_d_updates,
            d_closure_factory=make_d_closure,
            g_closure=g_closure,
            discriminator=discriminator,
            generator=self.env.generator,
            opt_d=self.env.opt_d,
            opt_g=g_config["optimizer"],
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses for the diffusion training step (refactored orchestrator).

        Refactored to delegate to focused helper methods:
        - _prepare_model_input: Setup model input (normalization, timesteps, forward process)
        - _build_generator_kwargs: Prepare generator kwargs
        - _forward_through_model: Forward pass through generator
        - _extract_and_fix_output: Post-process output to valid shape
        - _compute_and_log_debug_info: Debug logging and anomaly detection
        - loss_computer: Unified loss computation

        Args:
            input_batch: The input tensor (e.g., undersampled k-space).
            target_batch: The target tensor (e.g., fully-sampled k-space).
            epoch: Current epoch index.
            **kwargs: Additional context, including `batch_data` for metadata.

        Returns:
            A dictionary of loss tensors, including `g_total_loss` and individual components.
        """
        # DEBUG: Log input shapes at function entry
        current_step = kwargs.get("iteration", 0)

        # CRITICAL: Log shapes IMMEDIATELY to debug mismatch
        self.logging_service.log_debug(
            f"[DIFFUSION ENTRY Step {current_step}] input_batch.shape={input_batch.shape}, target_batch.shape={target_batch.shape}"
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[Diffusion._compute_losses_impl START] input_batch.shape=%s, target_batch.shape=%s",
                input_batch.shape,
                target_batch.shape,
            )

        # STEP 1: Prepare model input (normalization, timesteps, forward process)
        batch_size = target_batch.size(0)
        batch_data = kwargs.get("batch_data")

        (
            input_batch,
            target_batch,
            model_input,
            noisy_images,
            timesteps,
            is_cold_diffusion,
            is_latent_diffusion,
            mask,
            scale,
            contrast_idx,
            noise,
        ) = self._prepare_diffusion_inputs(
            input_batch, target_batch, batch_size, current_step, epoch, batch_data
        )

        # STEP 1b: optional measurement conditioning. Standard image diffusion
        # denoises noisy(target) with NO measurement input, which admits a
        # measurement-independent solution (#20). When ``condition_on_input`` is
        # set, concat the (LR/ULF) input onto the noised target so the denoiser
        # is conditioned on the measurement. No-op when the flag is off.
        model_input = self._maybe_condition_on_input(
            model_input,
            input_batch=input_batch,
            noisy_images=noisy_images,
            is_cold_diffusion=is_cold_diffusion,
            is_latent_diffusion=is_latent_diffusion,
        )

        # STEP 2: Build generator kwargs (conditioning, masks, etc.)
        gen_kwargs = self._build_generator_kwargs(
            is_cold_diffusion=is_cold_diffusion,
            is_latent_diffusion=is_latent_diffusion,
            input_batch=input_batch,
            target_batch=target_batch,
            batch_data=batch_data,
            mask=mask,
        )

        # The class declares ``snapshot_prepared_is_model_input = False`` and
        # ``snapshot_model_input_tag = "diffusion_step"``, which obliges EVERY
        # path through here to declare the tensor the network is actually fed
        # (non-negotiable 14). Only one path did: the declaration in
        # ``_prepare_diffusion_inputs`` sits inside
        # ``if "kspace_cold_diffusion" in str(self.config.model.model_type).lower()``,
        # so the promise was kept for one model family and broken for every
        # other diffusion arm -- which then died on the contract check with
        # "its _compute_losses_impl neither called _declare_model_input() nor
        # emitted a snapshot tagged 'diffusion_step'".
        #
        # ``model_input`` is that tensor on the general path, and this is the
        # last point before it reaches the network. Guarded on "nothing declared
        # yet" rather than called unconditionally: the k-space branch has
        # already declared a RICHER record (channel_segments for the
        # [noisy_kspace || smaps] concat), and ``_declare_model_input`` is a
        # plain overwrite, so an unconditional call here would silently discard
        # it. The stash is reset per step by ``_compute_losses``, so this reads
        # None on entry for every arm that did not take that branch.
        if getattr(self, "_declared_model_input", None) is None:
            # ``model_input_key`` is required, not decorative: the tag promises a
            # reader "the real model input is in here", and the contract refuses to
            # let them guess which tensor -- a guess is what let #1298's mislabel
            # stand.
            self._declare_model_input(
                {"model_input": model_input},
                extra={"model_input_key": "model_input"},
                in_kspace_keys=set(),
            )

        # STEP 3: Forward pass through generator
        predicted_output = self._forward_through_model(
            model_input=model_input,
            timesteps=timesteps,
            is_latent_diffusion=is_latent_diffusion,
            gen_kwargs=gen_kwargs,
            contrast_idx=contrast_idx,
        )

        # STEP 4: Extract and fix output shape.
        # For LDM the model output and the loss target both live in LATENT
        # space (Rombach et al., CVPR 2022, Eq. 3). Pass `noise` (the
        # latent-space ε that was injected during q_sample) as the target
        # so the channel-matching guard inside _extract_and_fix_output
        # aligns dimensions in latent space rather than truncating the
        # latent-channel axis to match the (different-shape) image-space
        # target. Without this, hr_fakes [B, latent_C, H', W'] is silently
        # truncated to [B, image_C, H', W'] and the downstream override at
        # STEP 4b mismatches against the image-space target — the May 2026
        # exp_32b ``hr_fakes [2,1,32,32] vs target [2,1,256,256]`` regression.
        _eafo_target = noise if (is_latent_diffusion and noise is not None) else target_batch
        hr_fakes_for_loss, target_for_loss = self._extract_and_fix_output(
            predicted_output=predicted_output,
            target_batch=_eafo_target,
            scale=scale,
            input_batch=input_batch,  # Pass input_batch (original undersampled k-space) for DC layer
            is_cold_diffusion=is_cold_diffusion,
            mask=mask,
        )

        # STEP 4b: LDM ε-prediction loss target override
        # For Latent Diffusion Models, the denoise_net output is a noise
        # prediction ε̂ in latent space.  The correct training objective is
        #   L = ||ε̂ − ε||²   (Rombach et al., CVPR 2022, Eq. 3)
        # where ε is the Gaussian noise added during q_sample.
        # The default target_for_loss from _extract_and_fix_output is
        # the pixel-space ground truth, which is dimensionally wrong.
        if is_latent_diffusion and noise is not None:
            target_for_loss = noise
            self.logging_service.log_debug(
                f"[LDM] Using noise tensor as loss target. "
                f"pred_shape={hr_fakes_for_loss.shape}, noise_shape={noise.shape}"
            )

        # STEP 5: Validate shape compatibility (CRITICAL DEBUG)
        if hr_fakes_for_loss.shape != target_for_loss.shape:
            # F-SHAPE-MISMATCH-RATELIMIT / 2026-05-20 — the auto-squeeze
            # below recovers from common 5D↔4D singleton-dim
            # discrepancies and is deterministic for the
            # ``[B, C, 1, H, W]`` case (the dominant smoke fingerprint
            # for older logs). Rate-limit the warning to one
            # notification per strategy instance so it remains
            # diagnosable without spamming the train log on every step
            # (6+ identical lines per epoch in smoke 20260516).
            if not getattr(self, "_warned_shape_mismatch", False):
                logger.warning(
                    f"Shape mismatch! hr_fakes={hr_fakes_for_loss.shape} (complex={torch.is_complex(hr_fakes_for_loss)}), "
                    f"target={target_for_loss.shape} (complex={torch.is_complex(target_for_loss)}) "
                    f"— attempting deterministic squeeze (warning suppressed for further occurrences)"
                )
                self._warned_shape_mismatch = True
            if target_for_loss.ndim == 5 and hr_fakes_for_loss.ndim == 4:
                if target_for_loss.shape[-1] == 1:
                    target_for_loss = target_for_loss.squeeze(-1)
            elif hr_fakes_for_loss.ndim == 5 and target_for_loss.ndim == 4:
                # Model returned 5D output [B, C, D, H, W] but target is 4D [B, C, H, W]
                # First try: squeeze trailing singleton dims
                while hr_fakes_for_loss.ndim > 4 and hr_fakes_for_loss.shape[-1] == 1:
                    hr_fakes_for_loss = hr_fakes_for_loss.squeeze(-1)
                # If still 5D, try squeezing dim 2 if it's singleton
                if hr_fakes_for_loss.ndim == 5 and hr_fakes_for_loss.shape[2] == 1:
                    hr_fakes_for_loss = hr_fakes_for_loss.squeeze(2)
                # Squeeze complex dim: [B, D1, D2, H, W] -> [B, D1*D2, H, W]
                if hr_fakes_for_loss.ndim == 5:
                    B, D1, D2, H, W = hr_fakes_for_loss.shape
                    hr_fakes_for_loss = hr_fakes_for_loss.reshape(B, D1 * D2, H, W)
                elif (
                    hr_fakes_for_loss.ndim == 5
                    and hr_fakes_for_loss.numel() == target_for_loss.numel()
                ):
                    hr_fakes_for_loss = hr_fakes_for_loss.reshape(target_for_loss.shape)
                elif hr_fakes_for_loss.ndim == 5:
                    logger.error(
                        f"Cannot resolve 5D->4D mismatch: hr_fakes {hr_fakes_for_loss.shape} vs target {target_for_loss.shape}"
                    )
                    raise ValueError(
                        f"Tensor shape mismatch cannot be resolved: "
                        f"hr_fakes {hr_fakes_for_loss.shape} vs target {target_for_loss.shape}"
                    )
            elif hr_fakes_for_loss.numel() == target_for_loss.numel():
                target_for_loss = target_for_loss.reshape(hr_fakes_for_loss.shape)
            else:
                logger.error(
                    f"Cannot reshape: {hr_fakes_for_loss.numel()} vs {target_for_loss.numel()} elements. "
                    f"hr_fakes dtype={hr_fakes_for_loss.dtype}, target dtype={target_for_loss.dtype}"
                )
                raise ValueError(
                    f"Tensor shape mismatch cannot be resolved: "
                    f"hr_fakes {hr_fakes_for_loss.shape} vs target {target_for_loss.shape}"
                )
        else:
            pass

        # Log scale normalization effect
        if scale is not None:
            pass

        # STEP 6: Debug logging and anomaly detection (reads the live iteration
        # from self.loop_state, WS-3 PR-3 — no longer threaded as current_step).
        self._compute_and_log_debug_info(
            hr_fakes_for_loss=hr_fakes_for_loss,
            target_for_loss=target_for_loss,
            input_batch=input_batch,
            model_input=model_input,
        )

        # STEP 7: Compute unified losses
        if (
            hasattr(self, "context")
            and self.context
            and hasattr(self.context, "loss_fn")
            and (self.env.losses if self.env else None)
        ):
            # Inject configured losses into computer if available
            if "diffusion" in (self.env.losses if self.env else {}):
                self.loss_computer.diffusion_loss_fn = (self.env.losses if self.env else {})[
                    "diffusion"
                ]

        # SSOT: Retrieve pre-built losses from environment (bootstrap phase responsibility)
        # Do NOT rebuild losses - use what LossBuilder already created in bootstrap
        losses_dict = {}

        if self.env and hasattr(self.env, "losses") and self.env.losses:
            # Filter to component losses (exclude wrapper losses like generator_loss, discriminator_loss)
            for loss_name, loss_fn in self.env.losses.items():
                if loss_name not in ["generator_loss", "discriminator_loss", "main"]:
                    losses_dict[loss_name] = loss_fn

        if losses_dict:
            self.logging_service.log_debug(
                f"Using {len(losses_dict)} pre-built losses from bootstrap: {list(losses_dict.keys())}"
            )

        smaps = getattr(self, "_current_smaps", None)
        # ── DIAGNOSTIC: log what's actually fed to the loss computer ──
        # Triggered by experiment_11_kspace_cold_diffusion CSV showing
        # complex_l1=0.0/log_spectral=0.0 in training. The user asked us to
        # confirm the loss inputs are in the domain we think they are. The
        # output is INFO-level so it appears once per first-iteration; bump
        # to DEBUG once the values are confirmed sensible.
        if not getattr(self, "_loss_input_diag_logged", False):
            try:

                def _stat(t: torch.Tensor) -> str:
                    if torch.is_complex(t):
                        mag = t.abs()
                        im_max = t.imag.abs().max().item()
                    else:
                        mag = t.abs() if t.dim() > 0 else t
                        im_max = float("nan")
                    return (
                        f"shape={tuple(t.shape)} dtype={t.dtype} "
                        f"complex={torch.is_complex(t)} "
                        f"min={float(t.min().item() if not torch.is_complex(t) else mag.min()):.4e} "
                        f"max={float(t.max().item() if not torch.is_complex(t) else mag.max()):.4e} "
                        f"imag_abs_max={im_max:.4e}"
                    )

                self.logging_service.log_info(f"[LOSS-INPUT DIAG] pred={_stat(hr_fakes_for_loss)}")
                self.logging_service.log_info(f"[LOSS-INPUT DIAG] target={_stat(target_for_loss)}")
                self.logging_service.log_info(
                    f"[LOSS-INPUT DIAG] losses_dict_keys={sorted((losses_dict or {}).keys())}"
                )
            except Exception as _exc:  # never let diag break training
                self.logging_service.log_debug(f"[LOSS-INPUT DIAG] failed: {_exc}")
            self._loss_input_diag_logged = True

        loss_output = self.loss_computer.compute(
            pred=hr_fakes_for_loss,
            target=target_for_loss,
            epoch=epoch,
            iteration=current_step,
            timesteps=timesteps,
            # The G step feeds the critic independently of the D step: the
            # computer calls ``discriminator(pred)`` itself
            # (unified_diffusion_reconstruction.py:511). Wrapping it here routes
            # that call through the SAME converter the D step uses, so the
            # critic cannot see one domain while training and another while
            # scoring the generator (#1920). Built per step, never cached: the
            # wrapper holds no tensors, and a cached one goes stale on any
            # EMA/DDP swap of ``discriminator_model``.
            discriminator=self._critic_for_domain(self.discriminator_model),
            # Condition the G step's critic call exactly as the D step conditions
            # its own (#1931). Built from THIS step's ``timesteps`` and
            # ``contrast_idx`` -- both already in scope here; ``contrast_idx``
            # was simply never passed on. A critic trained with labels but
            # queried without them is a different function, so the generator
            # would be chasing a gradient from a critic it never faces.
            critic_cond=self._critic_conditioning(timesteps, contrast_idx),
            losses_dict=losses_dict if losses_dict else None,
            smaps=smaps,
            mask=mask,
        )

        # Explicitly initialize and accumulate total_loss for execution graph validation
        total_loss = loss_output.total

        total_loss = total_loss + 0.0  # Satisfy accumulation pattern

        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(loss_output.components)
        # OPT-IN pre-DC fidelity (DC-blob L1+): an extra k-space L1 on the
        # generator's PRE-DC prediction so the net learns measurement-dependent
        # HF itself instead of leaning on the soft-DC-injected ACS centre.
        # No-op unless losses.reconstruction.lambda_pre_dc_kspace > 0.
        total_loss = self._add_pre_dc_fidelity(total_loss, predicted_output, target_for_loss, mask)
        self._loss_dict_reuse["g_total_loss"] = total_loss

        # STEP 8: Compute training metrics. Read the LIVE iteration from the
        # loop_state seam — the old ``self.env.step`` was a frozen 0, so the
        # ``current_step % train_metric_interval`` throttle in
        # ``_compute_training_metrics`` fired EVERY step (pitfall #16).
        current_step = resolve_loop_iteration(self)
        train_metrics = self._compute_training_metrics(
            pred=hr_fakes_for_loss,
            target=target_for_loss,
            config=self.config,
            current_step=current_step,
        )
        self._loss_dict_reuse.update(train_metrics)

        # ── Model-output TRAINING snapshot (DC-blob diagnostic) ─────────────
        # The legacy snapshot ran in _prepare_diffusion_inputs BEFORE the
        # forward, so it never held the model OUTPUT. Capture the LIVE post-DC
        # (+ pre-DC) reconstruction here at a trained-step cadence
        # (log_interval/10) so a validation-image-SAVE bug can be ruled out and
        # EMA-lag (validation grades the EMA shadow, not these live weights)
        # told apart from a real model failure. Capped by
        # logging.snapshots.max_calls; never let it break training.
        try:
            # Cadence check is INSIDE the guard: this snapshot is purely
            # diagnostic and must never perturb loss computation, even if
            # ``_cached_log_interval`` or the tensors are unexpected (tests/mocks).
            if current_step % max(1, int(self._cached_log_interval) // 10) == 0:
                _dc_snap = self._build_output_snapshot(
                    hr_fakes_for_loss,
                    predicted_output,
                    target_for_loss,
                    input_batch,
                    mask,
                    model_input=model_input,
                )
                self.save_debug_snapshot(
                    _dc_snap,
                    step=current_step,
                    # Its OWN tag, not "model_output" (#706, one level down).
                    # The budget is per-(run_dir, tag), and the base class's
                    # generic `model_output` emitter fires on EVERY early step
                    # while this one fires every `log_interval // 10` (500 steps
                    # on the kspace_filling cohort). Sharing the tag let the
                    # generic emitter spend all 8 calls by step 8, so this
                    # richer capture -- the only one carrying
                    # `model_output_pre_dc` next to `model_output_post_dc` --
                    # never ran. That pair is what distinguishes "data
                    # consistency never fired" from "the model diverged";
                    # without it the experiment_11 collapse could only be
                    # diagnosed by re-deriving the bound offline.
                    #
                    # It bites whenever the cadence exceeds the budget, i.e.
                    # `intervals.log >= 90`: 421 of the 647 arms under
                    # `inprogress/`, and 57 of the 58 in kspace_filling. Below
                    # that the two interleave and merely halve each other; at
                    # `intervals.log <= 10` the cadence is 1 and this capture
                    # starves the GENERIC one instead. Separate tags remove the
                    # competition in all three regimes rather than retuning it.
                    tag="model_output_dc",
                    in_kspace_keys={
                        "model_output_post_dc",
                        "model_output_pre_dc",
                        "target",
                        "input",
                        "model_input",
                    },
                    # ``model_input`` superposes the measured k-space and the
                    # conditioning half on its channel axis; render each half
                    # separately rather than letting one dominate the other's
                    # dynamic range (the crosshair, #1298/#1327).
                    channel_segments=self._model_input_channel_segments(_dc_snap),
                )
        except Exception as _snap_err:  # pragma: no cover — defensive
            logger.warning(f"model_output snapshot failed: {_snap_err}")

        if "loss" not in self._loss_dict_reuse and "g_total_loss" in self._loss_dict_reuse:
            self._loss_dict_reuse["loss"] = self._loss_dict_reuse["g_total_loss"]
        # TODO: this is a dumb fix and should be handled more elegantly in the loss computer or training loop to ensure 'loss' key is always present if 'g_total_loss' is used.
        # Every loss this arm DECLARED must reach the CSV row. The header is built
        # once at startup and the row writer drops unknown keys (``extrasaction=
        # "ignore"``), so a declared loss missing from the dict leaves a blank column
        # for the whole run. The 9-entry list this replaced read only
        # ``losses.reconstruction``, so it covered none of the domain-list arms
        # (#1918, #1919); the owner is now :attr:`_declared_loss_keys`.
        self._ensure_declared_losses_present(self._loss_dict_reuse, iteration=current_step)

        # Also log standard metrics from return dict
        loss_component_keys = [
            k
            for k in self._loss_dict_reuse.keys()
            if k not in ["g_total_loss", "loss", "diffusion"]
        ]
        if loss_component_keys:
            self.logging_service.log_debug(
                f"[Train] Total return dict components: {len(self._loss_dict_reuse)} keys"
            )
        else:
            self.logging_service.log_warning(
                f"[Train] No loss components in return dict! Keys: {list(self._loss_dict_reuse.keys())}"
            )

        # KAN telemetry (gate values per attention branch + trust-map stats
        # from the KAN ADC). Both helpers are no-ops when the generator
        # doesn't expose them. We unwrap DDP/torch.compile wrappers so the
        # lookup works regardless of session-build wrapping.
        gen = getattr(self, "generator_model", None)
        if gen is not None:
            inner = getattr(gen, "module", gen)  # DDP / OptimizedModule unwrap
            inner = getattr(inner, "_orig_mod", inner)  # torch.compile unwrap
            for helper in ("get_kan_gate_telemetry", "get_kan_trust_map_telemetry"):
                if hasattr(inner, helper):
                    for k, v in getattr(inner, helper)().items():
                        self._loss_dict_reuse[k] = torch.as_tensor(float(v))

            # KAN grid extension (plan §9 risk #1). Auto-toggle sample
            # collection at the very first step, then trigger grid updates
            # at 2K/4K/6K/8K intervals during the first 10K iterations.
            # After the warm-up window, disable collection to drop the CPU
            # buffer overhead permanently.
            if hasattr(inner, "set_kan_sample_collection") and hasattr(inner, "update_kan_grids"):
                kan_warmup_iters = 10000
                kan_update_every = 2000
                if current_step == 1:
                    inner.set_kan_sample_collection(True)
                elif (
                    current_step <= kan_warmup_iters
                    and current_step > 0
                    and current_step % kan_update_every == 0
                ):
                    n_updated = inner.update_kan_grids()
                    if n_updated > 0:
                        self.logging_service.log_info(
                            f"[KAN] Grid extension at iter {current_step}: "
                            f"updated {n_updated} layers"
                        )
                elif current_step == kan_warmup_iters + 1:
                    inner.set_kan_sample_collection(False)
                    self.logging_service.log_info(
                        "[KAN] Grid-extension warm-up complete; disabling sample collection"
                    )

        return self._loss_dict_reuse

    @staticmethod
    def _contrast_idx_from_batch(batch_data: Any, device: Any) -> torch.Tensor | None:
        """Read ``contrast_idx`` off a batch of any shape, or ``None`` (#1931).

        **The one owner of this extraction** (non-negotiable 17). There were
        two, and both were wrong in the same way -- ``isinstance(batch_data,
        dict)``. What ``training_loop`` actually hands down is a
        :class:`~spectramr.data.batch_types.TrainingBatch`, a dataclass and not
        a mapping, so neither guard ever matched: every multi-contrast diffusion
        arm has been training UNCONDITIONED, generator included, with no error
        and no log line. ``read_batch_field`` is this seam's declared answer to
        exactly that pairing -- see its docstring, which names it as the single
        most repeated defect here.

        The ``Tensor``/``list``/``tuple`` ladder is kept rather than collapsed
        into a bare assignment. ``read_batch_field``'s trailing ``getattr`` leg
        serves shapes that are neither mapping nor batch, so a ``Mock`` batch
        resolves ``contrast_idx`` to a ``Mock``; the ladder is what leaves those
        as ``None`` instead of forwarding an attribute-generating stub into the
        generator.

        Args:
            batch_data: The batch as the training loop delivered it -- a
                ``TrainingBatch``, a dict, or any object with attributes.
            device: Device to place the resolved index on, so the value is
                usable by both the generator and the critic without a second
                transfer.

        Returns:
            A long tensor of per-sample contrast ids, or ``None`` when the batch
            genuinely carries none.
        """
        c_idx = read_batch_field(batch_data, "contrast_idx")
        if isinstance(c_idx, torch.Tensor):
            return c_idx.to(device)
        if isinstance(c_idx, list | tuple):
            return torch.tensor(c_idx, dtype=torch.long, device=device)
        return None

    def _build_generator_kwargs(
        self,
        is_cold_diffusion: bool,
        is_latent_diffusion: bool,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dictionary for generator forward pass.

        Args:
            is_cold_diffusion: Whether using cold diffusion variant.
            is_latent_diffusion: Whether using latent diffusion variant.
            input_batch: Undersampled/noisy input tensor.
            target_batch: Ground truth tensor.
            batch_data: Batch metadata dictionary.
            mask: K-space mask (for cold diffusion).

        Returns:
            Dictionary of kwargs for generator forward pass.
        """
        gen_kwargs = {}

        if is_cold_diffusion:
            gen_kwargs["mask"] = mask
            # Pass measurements during training so model learns to blend predictions with observations
            gen_kwargs["kspace_measured"] = input_batch

        # Inject smaps into generator kwargs
        if hasattr(self, "_current_smaps") and self._current_smaps is not None:
            gen_kwargs["smaps"] = self._current_smaps

        # Pass accelerator_kwargs from config if available
        if hasattr(self.config, "undersampling") and self.config.undersampling:
            accelerator_kwargs = (
                self.config.undersampling.accelerator_kwargs
                if hasattr(self.config.undersampling, "accelerator_kwargs")
                else None
            )
            if accelerator_kwargs:
                gen_kwargs.update(accelerator_kwargs)

        if hasattr(self, "prior_model") and self.prior_model is not None:
            with torch.no_grad():
                prior = self.prior_model(input_batch)
                gen_kwargs["conditioning"] = prior

                if batch_data is not None and isinstance(batch_data, dict):
                    if "sensitivity" in batch_data:
                        gen_kwargs["sensitivity_maps"] = batch_data["sensitivity"].to(
                            target_batch.device
                        )
                    elif "sensitivity_maps" in batch_data:
                        gen_kwargs["sensitivity_maps"] = batch_data["sensitivity_maps"].to(
                            target_batch.device
                        )

        # Contrast conditioning: pass contrast_idx to generator.
        # ``_forward_through_model`` writes the same key on the train path, from
        # the value ``_prepare_diffusion_inputs`` extracted; sharing one owner
        # for the extraction makes the two writes identical by construction
        # rather than by coincidence. This write is load-bearing on its own for
        # the t=0 pre-DC probe, which builds generator kwargs without going
        # through ``_prepare_diffusion_inputs`` at all.
        contrast_idx = self._contrast_idx_from_batch(batch_data, target_batch.device)
        if contrast_idx is not None:
            gen_kwargs["contrast_idx"] = contrast_idx

        # SR3 conditioning (#16): thread the image-space input (ULF) as the
        # LDM's condition_image ONLY when the generator advertises conditional
        # translation. Without this the latent-diffusion forward runs
        # unconditionally — the ULF input is never seen (the 2026-07 facade).
        if is_latent_diffusion:
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            _gen_cfg = getattr(gen, "config", None)
            if getattr(_gen_cfg, "conditional_translation", False):
                gen_kwargs["condition_image"] = input_batch

        return gen_kwargs

    def _forward_through_model(
        self,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        is_latent_diffusion: bool,
        gen_kwargs: dict[str, Any],
        contrast_idx: torch.Tensor | None = None,
    ) -> Any:
        """Forward pass through generator model.

        Args:
            model_input: Processed input tensor for model.
            timesteps: Timestep tensor for diffusion models.
            is_latent_diffusion: Whether using latent diffusion.
            gen_kwargs: Additional kwargs for generator.

        Returns:
            Raw output from generator model.
        """
        if contrast_idx is not None:
            gen_kwargs["contrast_idx"] = contrast_idx

        if is_latent_diffusion:
            # Forward the SR3 condition + contrast context that
            # _build_generator_kwargs assembled. The latent forward signature is
            # forward(x, timesteps, context=None, condition_image=None); a bare
            # call (the old code) dropped BOTH, running unconditionally (#16).
            ctx_idx = gen_kwargs.get("contrast_idx")
            context = {"contrast_idx": ctx_idx} if ctx_idx is not None else None
            return self.generator_model(
                model_input,
                timesteps=timesteps,
                context=context,
                condition_image=gen_kwargs.get("condition_image"),
            )
        else:
            from .mixins.utils import _callable_accepts_kwarg

            # For wrapped models (DataParallel, DistributedDataParallel),
            # introspect the underlying module's forward signature.
            gen_forward = self.generator_model.forward
            if hasattr(self.generator_model, "module"):
                underlying = self.generator_model.module
                if hasattr(underlying, "forward"):
                    gen_forward = underlying.forward

            # Remove 'timesteps' from gen_kwargs so it is not duplicated when
            # we pass it explicitly below (avoids "multiple values" TypeError).
            gen_kwargs.pop("timesteps", None)

            if _callable_accepts_kwarg(gen_forward, "timesteps"):
                return self.generator_model(model_input, timesteps=timesteps, **gen_kwargs)
            elif _callable_accepts_kwarg(gen_forward, "time"):
                return self.generator_model(model_input, time=timesteps, **gen_kwargs)
            else:
                return self.generator_model(model_input, **gen_kwargs)

    def _extract_and_fix_output(
        self,
        predicted_output: Any,
        target_batch: torch.Tensor,
        scale: torch.Tensor | None,
        input_batch: torch.Tensor,
        is_cold_diffusion: bool,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract prediction from model output and apply post-processing.

        Handles:
        - Output extraction from tuple/dict/tensor formats
        - Channel mismatch fixing
        - Scale normalization
        - Data consistency layer application

        Args:
            predicted_output: Raw output from generator model.
            target_batch: Ground truth tensor.
            scale: Normalization scale factor (if applicable).
            input_batch: Original undersampled input tensor for data consistency.
            is_cold_diffusion: Whether using cold diffusion.
            mask: K-space mask for data consistency.

        Returns:
            Tuple of (hr_fakes_for_loss, target_for_loss) ready for loss computation.
        """

        def _complex_to_real_stacked(t: torch.Tensor) -> torch.Tensor:
            """Convert complex tensor to real-stacked [B, 2C, H, W].

            Handles both 4D [B,C,H,W] and 5D [B,C,H,W,D] complex inputs.
            view_as_real appends a dim of size 2, so 4D→5D and 5D→6D.
            We squeeze trailing singleton D dims first, convert, then restore.
            """
            if not torch.is_complex(t):
                return t
            # Squeeze trailing singleton dims from TorchIO (B,C,H,W,1)
            was_5d = False
            D_orig = 1
            while t.ndim > 4 and t.shape[-1] == 1:
                t = t.squeeze(-1)
            if t.ndim == 5:
                # Volumetric: [B, C, H, W, D] → flatten depth into batch
                was_5d = True
                B, C, H, W, D_orig = t.shape
                t = t.permute(0, 4, 1, 2, 3).reshape(B * D_orig, C, H, W)
            # Now t is 4D complex [B', C, H, W]
            # view_as_real → [B', C, H, W, 2], permute → [B', C, 2, H, W]
            t_real = torch.view_as_real(t).permute(0, 1, 4, 2, 3).contiguous()
            Bp, Cp, Dp, Hp, Wp = t_real.shape  # D is always 2
            t_real = t_real.view(Bp, Cp * Dp, Hp, Wp)  # [B', 2C, H, W]
            if was_5d:
                # Restore: [B*D, 2C, H, W] → [B, 2C, H, W, D]
                t_real = t_real.view(B, D_orig, Cp * Dp, Hp, Wp).permute(0, 2, 3, 4, 1)
            return t_real

        # Extract prediction from various output formats
        if isinstance(predicted_output, tuple):
            hr_fakes_for_loss = predicted_output[0]
        elif isinstance(predicted_output, dict):
            if "image" in predicted_output:
                hr_fakes_for_loss = predicted_output["image"]
            elif "kspace" in predicted_output:
                hr_fakes_for_loss = predicted_output["kspace"]
            else:
                hr_fakes_for_loss = next(iter(predicted_output.values()))
        else:
            hr_fakes_for_loss = predicted_output

        # Fix channel mismatches
        # CRITICAL FIX: Compare complex channels with complex channels, or real with real
        fakes_channels = hr_fakes_for_loss.shape[1]
        target_channels = target_batch.shape[1]

        if not torch.is_complex(hr_fakes_for_loss) and torch.is_complex(target_batch):
            # hr_fakes is real [B, 2C, H, W], target is complex [B, C, H, W]
            # We align them to REAL-STACKED domain to enable real-valued losses (like L1, MSE)
            target_batch = _complex_to_real_stacked(target_batch)
            target_channels = target_batch.shape[1]

            if fakes_channels != target_channels:
                logger.warning(
                    f"Aligned target batch to Real-Stacked, but channels still mismatch: Fakes={fakes_channels}, Target={target_channels}. Truncating Fakes."
                )
                hr_fakes_for_loss = hr_fakes_for_loss[:, :target_channels]
                fakes_channels = target_channels
        elif torch.is_complex(hr_fakes_for_loss) and not torch.is_complex(target_batch):
            # hr_fakes is complex [B, C, H, W], target is real [B, 2C, H, W]
            # Convert fakes to Real-Stacked domain to match target and enable real-valued losses
            hr_fakes_for_loss = _complex_to_real_stacked(hr_fakes_for_loss)
            fakes_channels = hr_fakes_for_loss.shape[1]

            if fakes_channels != target_channels:
                logger.warning(
                    f"Aligned fakes batch to Real-Stacked, but channels still mismatch: Fakes={fakes_channels}, Target={target_channels}. Truncating Fakes."
                )
                hr_fakes_for_loss = hr_fakes_for_loss[:, :target_channels]
                fakes_channels = target_channels
        elif torch.is_complex(hr_fakes_for_loss) and torch.is_complex(target_batch):
            # Both are complex. We convert BOTH to Real-Stacked to avoid complex gradients in PyTorch losses (e.g. L1Loss only supports real scalars)
            target_batch = _complex_to_real_stacked(target_batch)
            target_channels = target_batch.shape[1]

            hr_fakes_for_loss = _complex_to_real_stacked(hr_fakes_for_loss)
            fakes_channels = hr_fakes_for_loss.shape[1]

            if fakes_channels != target_channels:
                logger.warning(
                    f"Both complex, converted to Real-Stacked, but channels still mismatch: Fakes={fakes_channels}, Target={target_channels}. Truncating Fakes."
                )
                hr_fakes_for_loss = hr_fakes_for_loss[:, :target_channels]
                fakes_channels = target_channels
        elif not torch.is_complex(hr_fakes_for_loss) and not torch.is_complex(target_batch):
            # Both are naturally real-stacked (or purely magnitude/real).
            # If channels still mismatch significantly, truncate the predictions.
            # This is expected for multi-repetition models where generator outputs
            # out_channels * num_repetitions > target_channels.
            if fakes_channels != target_channels:
                if not getattr(self, "_logged_real_stacked_truncation", False):
                    logger.info(
                        f"Real-stacked channel alignment: Fakes={fakes_channels}, Target={target_channels}. "
                        f"Truncating {'target' if fakes_channels < target_channels else 'fakes'} to match."
                    )
                    self._logged_real_stacked_truncation = True
                else:
                    logger.debug(
                        f"Real-stacked channel alignment: Fakes={fakes_channels} vs Target={target_channels}"
                    )
                if fakes_channels > target_channels:
                    # Model outputs more channels than target (multi-rep) — truncate fakes
                    hr_fakes_for_loss = hr_fakes_for_loss[:, :target_channels]
                    fakes_channels = target_channels
                else:
                    # Model outputs fewer channels than target (cross-contrast out<in)
                    # Take LAST channels = target contrast (M4Raw stacks [source|target])
                    # Consistent with _slice_to_target_contrast_single which uses [:, -target_ch:]
                    target_batch = target_batch[:, -fakes_channels:]
                    target_channels = fakes_channels

        # Apply scale normalization to BOTH prediction and target
        # Scale is applied per-sample during kspace normalization, so both must be scaled
        # NOTE: target_batch and hr_fakes_for_loss are ALREADY scaled!
        # target_batch was scaled in _prepare_diffusion_inputs -> apply_kspace_normalization
        # hr_fakes_for_loss is the output of the generator, which received scaled inputs
        # Therefore, we DO NOT divide by scale again here to avoid double division.
        target_for_loss = target_batch
        hr_fakes_for_loss_scaled = hr_fakes_for_loss

        if scale is not None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"[Scale Normalization] Scale factor was applied in preprocessing. "
                    f"Scale shape: {scale.shape}, Scale min/max: [{scale.min().item():.4f}, {scale.max().item():.4f}]"
                )

        return hr_fakes_for_loss_scaled, target_for_loss

    @staticmethod
    def _unsampled_weight(mask: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
        """Broadcastable ``(1 - mask)`` weight over UNSAMPLED k-space bins.

        ``ref`` is the real-stacked ``[B, 2C, H, W]`` difference tensor; the
        mask is a per-location sampling indicator (``1`` = acquired). The pre-DC
        L1 must bite where the network cannot lean on the measurement, i.e. the
        unsampled bins, so this returns the complement aligned to ``ref``.

        Returns ``None`` — the caller then falls back to the uniform L1 (legacy
        behaviour) — when the mask is missing, cannot be aligned to ``ref``, or
        leaves no unsampled bins (fully sampled). The term must never crash
        training on a shape edge case.
        """
        if mask is None or not torch.is_tensor(mask):
            return None
        m = mask.real if torch.is_complex(mask) else mask
        m = m.to(dtype=ref.dtype, device=ref.device)
        if m.ndim != ref.ndim or m.shape[-2:] != ref.shape[-2:]:
            return None
        # A multi-coil mask that does not already broadcast collapses to a
        # single plane (a bin is "sampled" if ANY coil acquired it).
        if m.shape[1] not in (1, ref.shape[1]):
            m = m.amax(dim=1, keepdim=True)
        w = (1.0 - m).clamp_(0.0, 1.0)
        if not w.any():
            # Fully sampled -> no unmeasured bins (stays on device; no sync).
            # CONTRACT, not a nicety: returning None is what routes the caller
            # to the uniform ``diff.mean()``, and on an arm running the
            # fully-sampled rung (``undersampling.train_identity_rung``) that
            # fallback is the ONLY gradient reaching the generator at t=0 --
            # hard DC replaces the network's proposal at every acquired bin, so
            # every post-DC term is a constant there. Returning a zeroed weight
            # instead would kill the rung silently: the loss still computes,
            # training still runs, nothing goes red. Pinned by
            # ``test_unsampled_weight_returns_none_when_fully_sampled``.
            return None
        return w

    def declared_metric_keys(self) -> frozenset[str]:
        """Declare ``pre_dc_kspace_l1`` whenever the pre-DC term is enabled (#1682).

        ``_add_pre_dc_fidelity`` stamps this key on BOTH of its ``lam > 0``
        paths -- the active one and the INACTIVE sentinel that fires when the
        generator exposed no pre-DC tuple -- and on neither when ``lam <= 0``.
        The condition below is therefore the producer's own gate verbatim, so
        the column is promised exactly when a value will be written to it and
        never when one will not.

        Why this key in particular matters: it is the only observable of the
        only gradient the terminal (``t = 0``) rung receives. Under
        ``dc_method: hard`` every k-space bin is acquired at ``t = 0``, so hard
        DC replaces the network's proposal everywhere and every POST-DC loss is
        a constant with respect to the weights. Without this column the rung
        trains through one term that nothing records.
        """
        # Direct, unguarded read of the SSOT block (non-negotiable 3, and the
        # `test_diffusion_reads_config_directly_not_via_getattr_fallback` scan).
        # A `getattr(self.config, "losses", None)` fallback would turn a
        # malformed config into a silently-absent column -- the same defect
        # class as the discard #1682 fixes. `losses` is `LossConfigSchema | None`
        # and every diffusion arm in the corpus declares it, so an absent block
        # here is malformed input and must raise rather than resolve to "no
        # column". `reconstruction` is a declared field, so it is always present
        # (possibly None, which correctly means no pre-DC term).
        recon = self.config.losses.reconstruction
        lam = float(getattr(recon, "lambda_pre_dc_kspace", 0.0) or 0.0)
        return frozenset({"pre_dc_kspace_l1"}) if lam > 0.0 else frozenset()

    def _add_pre_dc_fidelity(
        self,
        total_loss: torch.Tensor,
        predicted_output: Any,
        target_for_loss: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """OPT-IN pre-DC fidelity supervision (Experiment-11 DC-blob, L1+).

        The dominant ``complex_l1`` is computed on the generator's POST-DC
        output, so soft-DC's injection of the always-sampled ACS centre
        satisfies it "for free" at sampled bins and the network is never forced
        to produce measurement-dependent high frequencies — the residual
        DC-blob driver. When ``losses.reconstruction.lambda_pre_dc_kspace > 0``
        and the generator exposed its PRE-DC prediction (the 2nd training-mode
        tuple element), this adds a k-space L1 between that pre-DC prediction
        and the target, pressuring the net's OWN output.

        The L1 is weighted by the UNSAMPLED complement ``(1 - mask)`` (when a
        mask is available): a uniform k-space L1 is dominated by the high-energy
        low-frequency centre and the always-sampled ACS bins that DC already
        injects, so it barely pressures the high frequencies the model must
        hallucinate. Concentrating the gradient on the unmeasured bins is where
        it actually closes the across-R gap. Without a mask it degrades to the
        uniform ``mean|pre_dc - target|`` (byte-identical to the legacy form).

        Default weight 0.0 -> exact no-op. Fail-safe: any shape edge case
        returns ``total_loss`` unchanged rather than crashing training. When
        ``lam > 0`` but the pre-DC prediction is unavailable, the term is
        INACTIVE — it stamps ``pre_dc_kspace_l1 = 0.0`` and warns once so the
        gap is visible in the CSV/provenance instead of vanishing silently
        (pitfall #9/#15).

        Args:
            total_loss: The aggregated training loss so far.
            predicted_output: Raw generator output; the pre-DC prediction is
                ``[1]`` when it is a ``(post_dc, pre_dc)`` tuple.
            target_for_loss: The (already real-stacked, aligned) loss target.
            mask: Sampling mask (``1`` = acquired). Drives the unsampled-bin
                weighting; ``None`` -> uniform L1.

        Returns:
            ``total_loss`` plus ``lambda * weighted_L1(pre_dc, target)`` when
            enabled, else ``total_loss`` unchanged.
        """
        recon = self.config.losses.reconstruction
        lam = float(getattr(recon, "lambda_pre_dc_kspace", 0.0) or 0.0)
        if lam <= 0.0:
            return total_loss
        if not (
            isinstance(predicted_output, tuple)
            and len(predicted_output) > 1
            and torch.is_tensor(predicted_output[1])
        ):
            # Advertised (lam > 0) but the generator did not expose its pre-DC
            # prediction (only happens off the training-mode tuple path). Make
            # the inactivity VISIBLE rather than a silent no-op (pitfall #9/#15).
            self._loss_dict_reuse["pre_dc_kspace_l1"] = torch.zeros((), device=total_loss.device)
            if not getattr(self, "_pre_dc_inactive_warned", False):
                logger.warning(
                    "losses.reconstruction.lambda_pre_dc_kspace=%.3g is set but "
                    "the generator did not expose a pre-DC prediction (expected a "
                    "(post_dc, pre_dc) tuple in training mode); the pre-DC "
                    "fidelity term is INACTIVE.",
                    lam,
                )
                self._pre_dc_inactive_warned = True
            return total_loss

        pre_dc = predicted_output[1]
        if torch.is_complex(pre_dc):
            # complex [B, C, H, W] -> real-stacked [B, 2C, H, W] (R,I,R,I,...)
            pre_dc = torch.view_as_real(pre_dc).movedim(-1, 2).flatten(1, 2)

        tgt = target_for_loss
        if pre_dc.ndim != tgt.ndim:
            return total_loss  # fail-safe — unexpected rank
        if pre_dc.shape[1] != tgt.shape[1]:
            # Surface the channel mismatch once instead of silently slicing
            # (a quiet truncation can hide a real wiring bug — pitfall #9).
            if not getattr(self, "_pre_dc_chan_warned", False):
                logger.warning(
                    "pre-DC fidelity channel mismatch: pre_dc=%s vs target=%s; "
                    "comparing the leading %d channels.",
                    tuple(pre_dc.shape),
                    tuple(tgt.shape),
                    min(pre_dc.shape[1], tgt.shape[1]),
                )
                self._pre_dc_chan_warned = True
            c = min(pre_dc.shape[1], tgt.shape[1])
            pre_dc, tgt = pre_dc[:, :c], tgt[:, :c]
        if pre_dc.shape != tgt.shape:
            return total_loss  # fail-safe — never crash training on an edge case

        diff = torch.abs(pre_dc - tgt)
        w = self._unsampled_weight(mask, diff)
        if w is not None:
            # Broadcast the per-location weight over the channel axis BEFORE
            # normalising, so the denominator counts every weighted (channel,
            # bin) position — otherwise a [B,1,H,W] mask under-counts vs a
            # [B,2C,H,W] diff and the mean is scaled by the channel factor.
            w = w.expand_as(diff)
            term = (w * diff).sum() / w.sum().clamp_min(1.0)
        else:
            term = diff.mean()
        self._loss_dict_reuse["pre_dc_kspace_l1"] = term.detach()
        return total_loss + lam * term

    def _build_output_snapshot(
        self,
        hr_fakes_for_loss: torch.Tensor,
        predicted_output: Any,
        target_for_loss: torch.Tensor,
        input_batch: torch.Tensor,
        mask: torch.Tensor | None,
        model_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build the tensor dict for the model-output TRAINING snapshot.

        Captures the LIVE-weights model OUTPUT (post-DC, and pre-DC when the
        generator exposed it) alongside the input/target/mask. The legacy
        training snapshot was emitted from ``_prepare_diffusion_inputs`` —
        STRICTLY BEFORE the generator forward — so it only ever held inputs and
        could never show what the model produced. Surfacing the output here
        lets us (a) rule out a validation-image-SAVE bug as the source of the
        "blob", and (b) distinguish a genuine model failure from EMA-lag:
        validation evaluates the EMA shadow weights, NOT these live weights, so
        a sharp output here next to a blobby EMA validation = EMA-lag confirmed.

        ``model_input`` is the *degraded* tensor actually fed to
        ``model.forward`` (``x_t = target * mask_t``, post any measurement
        conditioning) — distinct from ``input`` (the raw undersampled
        measurement ``input_batch``). Surfacing it lets a reviewer SEE what the
        model received and, via the snapshot's domain-superposition detector,
        confirm the undersampling mask landed in k-space (a dense aliased image)
        and not the image domain (zeroed image lines). 2026-07-06.
        """
        snap: dict[str, torch.Tensor] = {
            "model_output_post_dc": hr_fakes_for_loss.detach(),
            "target": target_for_loss.detach(),
            "input": input_batch.detach(),
        }
        if model_input is not None and torch.is_tensor(model_input):
            snap["model_input"] = model_input.detach()
        if (
            isinstance(predicted_output, tuple)
            and len(predicted_output) > 1
            and torch.is_tensor(predicted_output[1])
        ):
            snap["model_output_pre_dc"] = predicted_output[1].detach()
        if mask is not None and torch.is_tensor(mask):
            snap["mask"] = mask.detach()
        _smaps = getattr(self, "_current_smaps", None)
        if torch.is_tensor(_smaps):
            snap["smaps"] = _smaps.detach()
        return snap

    def _model_input_channel_segments(
        self, snap: Mapping[str, torch.Tensor]
    ) -> dict[str, list[ChannelSegment]]:
        """Declare the domain of each channel block of ``model_input``.

        ``_prepare_diffusion_inputs`` builds the tensor the network is fed as
        ``torch.cat([noisy_images, smaps_k], dim=1)``. ``noisy_images`` is
        real-stacked k-space (``log1p``-compressed when the arm enables k-space
        log scaling). The maps half was image-domain when this helper was
        written, which is what made the previewer's blanket IFFT draw a bright
        DC pixel plus the horizontal/vertical ridge -- the crosshair on
        ``experiment_11_attention_none``'s ``model_input.png`` while the k-space
        the model received was fine.

        Since #1327 that half is ``prepare_smaps_for_kspace_conditioning``'s
        output, i.e. ``fft2c`` of the maps, level-matched and amplitude-capped.
        **Both segments are therefore k-space now**, and the split survives for
        two reasons that outlived the one it was born for. The widths are
        unaffected -- ``prepare_smaps_for_kspace_conditioning`` round-trips
        dtype and shape, so deriving them from ``_current_smaps`` (which stays
        image-domain, for the SENSE projection) remains correct.

        First, the two halves have different level statistics, so rendering
        them jointly still lets the maps' spectrum dominate the k-space half's
        dynamic range. Second -- and this is why the segments carry a fourth
        field -- only the ``x_t`` half is ``log1p``-compressed. The maps go
        into the concat straight from ``fft2c``, so declaring them k-space
        earns them the IFFT they now need but must NOT earn them the arm's
        ``expm1``: that would clamp every bin above
        ``DECOMPRESS_MAGNITUDE_CEILING`` to one value and render a phase-only,
        washed-out map. Domain and compression are declared separately because
        this tensor genuinely disagrees with itself about them.

        The split is derived from the smaps tensor that was concatenated, not
        from a constant: ``num_physical_coils`` is per-arm and the maps reach
        here complex on some paths and already real-stacked on others. When no
        smaps were concatenated, ``model_input`` IS the k-space and no
        declaration is emitted -- an empty mapping leaves rendering unchanged.
        """
        model_input = snap.get("model_input")
        smaps = getattr(self, "_current_smaps", None)
        if not torch.is_tensor(model_input) or not torch.is_tensor(smaps):
            return {}
        if model_input.dim() < 2 or smaps.dim() < 2:
            return {}

        # Width the maps occupy once real-stacked, matching the conversion
        # `_prepare_diffusion_inputs` applies before the concat.
        smaps_width = smaps.shape[1] * (2 if torch.is_complex(smaps) else 1)
        kspace_width = model_input.shape[1] - smaps_width
        if kspace_width <= 0:
            # No concat happened (or the maps are not what we think). Say so
            # rather than guess a split -- a wrong one would put a confident
            # filename on a wrong picture.
            return {}
        return {
            "model_input": [
                ("kspace", kspace_width, True, True),
                # k-space since #1327, but NEVER ``log1p``'d -- see the
                # docstring. The fourth field is what keeps those two facts
                # from being read off one another.
                ("smaps", smaps_width, True, False),
            ]
        }

    @staticmethod
    def _curriculum_r_max(current_iter: int, r_max_val: float, max_iters: int) -> float:
        """The acceleration-curriculum R_max at ``current_iter`` (Phase 4.2).

        R is held at 2x up to ``SHORT_RUN_BYPASS_ITERATIONS``, then linearly
        interpolated to the configured ``r_max_val`` at iteration 50000. Short
        debug runs (``max_iters <= SHORT_RUN_BYPASS_ITERATIONS``) skip the cap
        and use the full ``r_max_val`` so diagnostics are meaningful
        immediately -- the same threshold as the timestep curriculum's bypass,
        which is why it is imported rather than written out again (#1296).

        Pure function of the live iteration (read from ``self.loop_state`` by the
        caller, WS-3 PR-3) so it is unit-testable in isolation. The regression it
        guards: before the loop_state seam, the caller read the frozen
        ``env.step`` (constant 0), so this always collapsed to the 2x branch.
        """
        if max_iters <= SHORT_RUN_BYPASS_ITERATIONS:
            return r_max_val
        if current_iter <= SHORT_RUN_BYPASS_ITERATIONS:
            return 2.0
        if current_iter >= 50000:
            return r_max_val
        return 2.0 + (r_max_val - 2.0) * (
            (current_iter - SHORT_RUN_BYPASS_ITERATIONS) / (50000 - SHORT_RUN_BYPASS_ITERATIONS)
        )

    def _compute_and_log_debug_info(
        self,
        hr_fakes_for_loss: torch.Tensor,
        target_for_loss: torch.Tensor,
        input_batch: torch.Tensor,
        model_input: torch.Tensor,
    ) -> None:
        """Compute and log debug information for anomaly detection.

        Checks for:
        - Identity collapse (output ≈ input)
        - Target leakage (output ≈ target)
        - Missing degradation (input ≈ target)
        - NaN values

        Args:
            hr_fakes_for_loss: Model prediction.
            target_for_loss: Ground truth (possibly normalized).
            input_batch: Undersampled/noisy input.
            model_input: Processed input to model.
        """
        debug_steps = self.config.logging.snapshots.log_steps
        anomaly_check_interval = self.config.logging.intervals.anomaly_check
        # WS-3 PR-3: read the live iteration from the loop_state seam. The old
        # ``self.env.step`` reads were inert — TrainingEnvironment is frozen, so
        # ``env.step`` was a constant 0, which made ``0 % interval == 0`` fire
        # this debug log every step and pinned the curriculum diagnostic below
        # at stage 0 (always the 2x branch) regardless of real progress.
        current_iter = self.loop_state.iteration
        should_log = self.logging_service.logger.isEnabledFor(logging.INFO) and (
            current_iter in debug_steps or current_iter % anomaly_check_interval == 0
        )
        if not should_log:
            return

        self.logging_service.log_info(f"Step {current_iter}:")

        # Batch all tensor operations before GPU→CPU transfer
        with torch.no_grad():
            # Helper to handle complex tensors
            def get_stats(tensor):
                """Get min, max, mean stats, handling complex tensors by taking magnitude."""
                if torch.is_complex(tensor):
                    tensor = tensor.abs()  # Convert complex to real (magnitude)
                # Empty tensor: torch.min/max/mean raise. Debug instrumentation
                # must never kill training — return NaN sentinels and let the
                # caller log them. The real failure (an empty batch reaching
                # the loss) is hard-failed elsewhere with a clearer message.
                if tensor.numel() == 0:
                    nan = torch.full((), float("nan"), device=tensor.device, dtype=tensor.dtype)
                    return torch.stack([nan, nan, nan])
                return torch.stack([tensor.min(), tensor.max(), tensor.mean()])

            input_stats = get_stats(input_batch)
            target_stats = get_stats(target_for_loss)
            fakes_stats = get_stats(hr_fakes_for_loss)

            all_stats = torch.cat([input_stats, target_stats, fakes_stats]).detach().cpu().numpy()
            input_min, input_max, input_mean = all_stats[0:3]
            target_min, target_max, target_mean = all_stats[3:6]
            fakes_min, fakes_max, fakes_mean = all_stats[6:9]

            # Handle NaN checks for complex tensors
            def has_nan(tensor):
                """Check for NaN in real or complex tensors."""
                if torch.is_complex(tensor):
                    return (
                        torch.isnan(tensor.real).any().item()
                        or torch.isnan(tensor.imag).any().item()
                    )
                return torch.isnan(tensor).any().item()

            input_has_nan = has_nan(input_batch)
            target_has_nan = has_nan(target_for_loss)
            fakes_has_nan = has_nan(hr_fakes_for_loss)

        self.logging_service.log_info(
            f"  input_batch: shape={input_batch.shape}, min={input_min:.6f}, "
            f"max={input_max:.6f}, mean={input_mean:.6f}, contains_nan={input_has_nan}"
        )
        self.logging_service.log_info(
            f"  target_for_loss: shape={target_for_loss.shape}, min={target_min:.6f}, "
            f"max={target_max:.6f}, mean={target_mean:.6f}, contains_nan={target_has_nan}"
        )
        self.logging_service.log_info(
            f"  hr_fakes_for_loss: shape={hr_fakes_for_loss.shape}, min={fakes_min:.6f}, "
            f"max={fakes_max:.6f}, mean={fakes_mean:.6f}, contains_nan={fakes_has_nan}"
        )

        with torch.no_grad():
            # Align model_input shape/dtype for diff computation
            mi_aligned = model_input
            if torch.is_complex(mi_aligned) and not torch.is_complex(hr_fakes_for_loss):
                mi_aligned = torch.view_as_real(mi_aligned).permute(0, 1, 4, 2, 3).contiguous()
                mi_aligned = mi_aligned.view(
                    mi_aligned.shape[0], -1, mi_aligned.shape[-2], mi_aligned.shape[-1]
                )
            elif not torch.is_complex(mi_aligned) and torch.is_complex(hr_fakes_for_loss):
                # Reverse process (unlikely but safe)
                mi_reshaped = mi_aligned.view(
                    mi_aligned.shape[0],
                    mi_aligned.shape[1] // 2,
                    2,
                    mi_aligned.shape[-2],
                    mi_aligned.shape[-1],
                )
                mi_aligned = torch.view_as_complex(mi_reshaped.permute(0, 1, 3, 4, 2).contiguous())

            # S-map conditioning concatenates maps to model_input, increasing its channel dim.
            # We slice to match the target's channel dimension to compare the actual image data.
            if mi_aligned.shape[1] > hr_fakes_for_loss.shape[1]:
                mi_aligned_sliced = mi_aligned[:, : hr_fakes_for_loss.shape[1]]
            else:
                mi_aligned_sliced = mi_aligned

            # Shape guard: skip diff computation when pred/target shapes don't match
            # (e.g., 3D trellis output [B,1,D,H,W] vs 2D target [B,1,H,W])
            if hr_fakes_for_loss.shape != target_for_loss.shape:
                self.logging_service.log_warning(
                    f"  [Debug] Shape mismatch in identity check: "
                    f"pred={hr_fakes_for_loss.shape} vs target={target_for_loss.shape}. "
                    f"Skipping diff computation."
                )
                pred_target_diff = float("nan")
                pred_input_diff = float("nan")
                input_target_diff = float("nan")
            else:
                pred_target_diff = (hr_fakes_for_loss - target_for_loss).abs().mean().item()
                pred_input_diff = (hr_fakes_for_loss - mi_aligned_sliced).abs().mean().item()
                input_target_diff = (mi_aligned_sliced - target_for_loss).abs().mean().item()

            self.logging_service.log_debug(
                f"  [Identity Check] pred-target diff={pred_target_diff:.6f}, "
                f"pred-input diff={pred_input_diff:.6f}, input-target diff={input_target_diff:.6f}"
            )

            id_collapse_thresh = self.config.training.identity_collapse_threshold
            if id_collapse_thresh is None:
                id_collapse_thresh = 0.01

            # F-IDCOLLAPSE-SKIP / 2026-05-20 — respect the same
            # ``synthetic_forward_probe_skip = {"identity_collapse"}``
            # opt-out the audit-time forward probe already honours.
            # Models like ``latent_flow`` and ``kspace_cold_diffusion``
            # are *expected* to produce near-identity output at
            # initialisation (invertible flow / fully-sampled cold-DC
            # case at t=0). Without the opt-out the validation-time
            # check spammed warnings on every val step.
            _probe_skip = (
                getattr(self.generator_model, "synthetic_forward_probe_skip", set()) or set()
            )
            if pred_input_diff < id_collapse_thresh and "identity_collapse" not in _probe_skip:
                self.logging_service.log_warning(
                    f"  ⚠️ IDENTITY COLLAPSE DETECTED: pred-input diff={pred_input_diff:.6f} is too small!"
                )
                self.logging_service.log_warning(
                    "     Model is outputting nearly identical to input. Check if timestep embedding is used."
                )
                # F-TIME-DETECT-BROADEN / 2026-05-20 — the previous
                # ``time_embedding`` / ``time_embed`` substring check
                # missed models that wire time-conditioning under
                # different attribute names (e.g. ConsistencyModelGenerator
                # uses ``time_mlp``, DiffusionUNet uses ``_time_embed``).
                # Smoke 20260509 surfaced 30+ false-positive "No
                # time_embedding found" warnings for the consistency
                # model arm even though it DOES condition on t. Broaden
                # the recognition list; the warning now only fires
                # when none of the canonical attribute names is found.
                _TIME_ATTRS = (
                    "time_embedding",
                    "time_embed",
                    "time_mlp",
                    "t_emb",
                    "t_embed",
                    "t_proj",
                    "_time_embed",
                )
                _found = next(
                    (attr for attr in _TIME_ATTRS if hasattr(self.generator_model, attr)),
                    None,
                )
                if _found is not None:
                    self.logging_service.log_warning(
                        f"     Debug: time-conditioning attr '{_found}' "
                        f"exists: {getattr(self.generator_model, _found)}"
                    )
                else:
                    self.logging_service.log_warning(
                        "     Debug: No known time-conditioning attribute "
                        f"({_TIME_ATTRS}) found in model. Model may not "
                        "support timestep conditioning!"
                    )

            early_step_thresh = self.config.training.early_training_steps
            if early_step_thresh is None:
                early_step_thresh = 500

            if current_iter < early_step_thresh and pred_target_diff < id_collapse_thresh:
                self.logging_service.log_warning(
                    f"  ⚠️ POTENTIAL TARGET LEAK: pred-target diff={pred_target_diff:.6f} at early step!"
                )
                self.logging_service.log_warning(
                    "     Output is nearly identical to target. Check data pipeline for leak."
                )

            # Compute the curriculum's current maximum timestep to derive the expected
            # mask-removal fraction before firing the "MASK NOT APPLIED" alarm.
            # At timestep t the power-law schedule retains 1/R(t) of k-space:
            #   R(t) = 1 + (R_max - 1) * (t / (T-1))^p   [p=2, R_max=32, T=1000]
            # For t <= 138 (iters < 1620) R <= 1.58x, so input≈target is EXPECTED.
            #
            # Skip the curriculum/acceleration diagnostic for paradigms without
            # an acceleration schedule (e.g. LDM, vanilla Gaussian diffusion in
            # latent space). Stage-2 LDM has no ``config.acceleration`` block
            # because its forward process is plain Gaussian noise injection;
            # forcing this diagnostic to run produced the May 2026 stage-2
            # ``'NoneType' object has no attribute 'max_acceleration'`` crash.
            # acceleration is a declared ``AccelerationConfigSchema | None`` field,
            # so direct access returns None for stage-2 LDM (no acceleration
            # block); the guard below handles None without the documented crash.
            _accel_cfg = self.config.undersampling
            if _accel_cfg is None or not _accel_cfg.max_acceleration:
                return
            # Direct access (SSOT) — matches the curriculum read at L411/416;
            # configs set these at training top-level (extra="allow") and this
            # block only runs when acceleration is present (guarded above).
            _curr_ramp_rate = self.config.training.curriculum_ramp_rate
            _curr_start_t = self.config.training.curriculum_start_timestep
            # Both are ``| None`` and default to None ("no curriculum" per the
            # schema). Same TypeError as sample_timesteps if either is unset —
            # here it is masked only because this block is behind the
            # acceleration guard above, which stage-2 LDM never passes.
            if _curr_start_t is None or _curr_ramp_rate is None:
                _t_max_now = self.num_timesteps - 1
            else:
                _t_max_now = min(
                    self.num_timesteps - 1,
                    int(_curr_start_t + current_iter * _curr_ramp_rate) - 1,
                )
            _r_max_val = float(_accel_cfg.max_acceleration)

            # Phase 4.2 Curriculum Learning update
            # R=2 up to SHORT_RUN_BYPASS_ITERATIONS, linear interpolation to
            # R=32 at iteration 50000. For short debug runs
            # (max_iterations <= SHORT_RUN_BYPASS_ITERATIONS), skip the cap and use
            # full config R_max so diagnostics and masks are meaningful immediately.
            # ``current_iter`` is the live loop iteration read from the
            # loop_state seam at the top of this method (WS-3 PR-3), superseding
            # the frozen-``env.step`` read that pinned this diagnostic at 0.
            _max_iters = self.config.training.max_iterations or DEFAULT_MAX_ITERATIONS
            _r_max_val_curriculum = self._curriculum_r_max(current_iter, _r_max_val, _max_iters)

            _r_at_tmax = (
                1.0
                + (_r_max_val_curriculum - 1.0) * (_t_max_now / max(1, self.num_timesteps - 1)) ** 2
            )
            # Expected minimum diff = (1 - 1/R) * mean_signal; use 1/R as a proxy fraction.
            # Only raise the warning when the current curriculum SHOULD produce >=2x undersampling.
            _mask_applied_threshold = 0.01 if _r_at_tmax >= 2.0 else 0.0002
            if input_target_diff < _mask_applied_threshold:
                self.logging_service.log_warning(
                    f"  ⚠️ MASK NOT APPLIED: input-target diff={input_target_diff:.6f} "
                    f"< {_mask_applied_threshold:.4f} (curriculum t_max={_t_max_now}, R_max={_r_at_tmax:.2f}x)"
                )
                self.logging_service.log_warning(
                    "     model_input ≈ target, degradation is not working!"
                )

    def _maybe_condition_on_input(
        self,
        model_input: torch.Tensor,
        *,
        input_batch: torch.Tensor,
        noisy_images: torch.Tensor,
        is_cold_diffusion: bool,
        is_latent_diffusion: bool,
    ) -> torch.Tensor:
        """Concatenate the LR/ULF input onto the noised target (conditional diffusion).

        Gated on ``training.diffusion.condition_on_input`` (default off, so every
        existing arm is byte-identical). No-op for cold/latent diffusion and when
        another conditioning step already extended the channel axis (detected via
        ``model_input is not noisy_images`` — e.g. the smaps concat). The model's
        ``in_channels`` must account for the extra channels (noisy + condition).
        """
        cached = getattr(self, "_condition_on_input_flag", None)
        if cached is None:
            # ``is True`` (not ``bool(...)``): the schema field is a validated
            # bool, so a real config gives True/False, while a MagicMock config
            # in unit tests gives a truthy mock — which must NOT enable the
            # concat (it would feed a 2-channel input to a 1-channel model).
            cached = self.config.training.diffusion.condition_on_input is True
            self._condition_on_input_flag = cached
        if not cached or is_cold_diffusion or is_latent_diffusion:
            return model_input
        if model_input is not noisy_images:
            # smaps / other conditioning already concatenated — don't double up.
            return model_input
        cond = input_batch
        if cond.shape[0] != model_input.shape[0]:
            raise ValueError(
                "condition_on_input: batch-size mismatch between input "
                f"({cond.shape[0]}) and noised target ({model_input.shape[0]})."
            )
        if cond.shape[2:] != model_input.shape[2:]:
            mode = "trilinear" if cond.dim() == 5 else "bilinear"
            cond = F.interpolate(cond, size=model_input.shape[2:], mode=mode, align_corners=False)
        return torch.cat([model_input, cond], dim=1)

    def _prepare_diffusion_inputs(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_size: int,
        current_step: int,
        epoch: int,
        batch_data: Any,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
        bool,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Prepare model input: normalization, timesteps, and forward process.

        Handles:
        - Normalization (physics-compliant)
        - Timestep sampling/loading
        - Forward process (q_sample) to corrupt data
        - Model input concatenation for conditional models

        Args:
            input_batch: Undersampled/noisy input.
            target_batch: Ground truth signal.
            batch_size: Batch size.
            current_step: Current training step.
            epoch: Current epoch.
            batch_data: Metadata dictionary.

        Returns:
            Tuple of:
            - input_batch (possibly normalized)
            - target_batch (possibly normalized)
            - model_input (processed input to model)
            - noisy_images (from forward process)
            - timesteps (tensor)
            - is_cold_diffusion (bool)
            - is_latent_diffusion (bool)
            - mask (optional, for cold diffusion)
            - scale (optional, for normalization)
        """
        # ✅ SSOT: Log input shapes at function entry to detect data loader issues
        # CRITICAL DEBUG: Always log first 5 steps to see where size mismatch happens
        if current_step <= 5:
            logger.debug(
                f"[Diffusion._prepare_diffusion_inputs] ENTRY Step {current_step} shapes: "
                f"input_batch={input_batch.shape}, target_batch={target_batch.shape}, "
                f"batch_size={batch_size}"
            )
            if hasattr(self, "logging_service"):
                self.logging_service.log_debug(
                    f"[ENTRY Step {current_step}] input={input_batch.shape}, target={target_batch.shape}"
                )

        scale = None
        mask = None
        timesteps = None

        # Contrast conditioning for BOTH the generator and (via
        # ``_critic_conditioning``) the discriminator. One owner, one guard --
        # see ``_contrast_idx_from_batch`` for why the ``isinstance(..., dict)``
        # this replaces read every real batch as carrying no contrast.
        contrast_idx = self._contrast_idx_from_batch(batch_data, target_batch.device)

        # Log batch info if needed
        should_log = current_step % self._cached_log_interval == 0
        if should_log:
            logger.debug(
                "[Step %d][Epoch %d] Batch: %s",
                current_step,
                epoch,
                type(batch_data).__name__,
            )

        # Apply normalization for cold diffusion
        if self._is_cold_diffusion() and self._cached_normalize_kspace:
            # Check if dataloader already provided kspace_scale. This MUST go
            # through the mapping protocol: ``batch_data`` here is the object the
            # training loop built with ``BatchAdapter.from_dict`` (see
            # ``pipelines/training_loop.py``), i.e. a TrainingBatch, whose
            # ``kspace_scale`` lives in ``.metadata`` and is unreachable by
            # ``hasattr``. Reading it as absent made the branch below normalize
            # an already-normalized batch a SECOND time.
            kspace_scale_from_batch = read_batch_field(batch_data, "kspace_scale")

            if self._batch_is_already_normalized(batch_data, kspace_scale_from_batch, current_step):
                # Dataloader already normalized the tensors. Extract the scale.
                scale = kspace_scale_from_batch
                if not torch.is_tensor(scale):
                    scale = torch.tensor(scale, device=target_batch.device)
                scale = scale.to(target_batch.device)
            else:
                input_batch, target_batch, scale = self.apply_kspace_normalization(
                    input_batch, target_batch, current_step=current_step
                )

        # Handle 5D tensors from dataloader (B, C, H, W, D) → (B*D, C, H, W)
        # This is required for 2D models training on 3D volumes
        should_flatten_5d = True

        if target_batch.ndim >= 5:
            # Reject empty 5D batches up-front. A zero in any dim silently
            # produces a 0-row tensor downstream that propagates all the way
            # to loss. CLAUDE.md #9 — surface this at the boundary instead.
            if 0 in tuple(target_batch.shape):
                # Surface the manifest record (file_id, contrast_idx, anything
                # that helps the user pinpoint the degenerate volume) so the
                # 2026-05-10 stage2_ldm smoke "iter 6" failure is actionable
                # without re-reading 28 MB of cluster logs.
                file_hint = ""
                if batch_data is not None:
                    for key in ("file_id", "subject_id", "path", "filename"):
                        if isinstance(batch_data, dict) and key in batch_data:
                            file_hint = f" file_id/path={batch_data[key]!r}"
                            break
                        if hasattr(batch_data, key):
                            file_hint = f" file_id/path={getattr(batch_data, key)!r}"
                            break
                raise RuntimeError(
                    f"5D target_batch has a zero-sized dim: shape={tuple(target_batch.shape)}.{file_hint} "
                    f"This indicates the dataloader produced an empty patch (e.g. depth=0 from "
                    f"patch_size=[H,W,1] on a volume with insufficient z-extent). Inspect the "
                    f"manifest record at this iteration before retrying. Common cause: a NIfTI "
                    f"in the manifest has fewer Z-slices than the configured patch depth — "
                    f"filter such volumes out of the paired_manifest_path before training."
                )
            # Structurally check if model natively supports 5D (e.g. RepetitionFusion)
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            # Ask the generator whether it can consume 5D -- NOT whether it
            # owns a repetition-fusion layer. Those are two different
            # invariants and this disjunction conflated them (non-negotiable
            # 17): it reached for ``rep_fusion`` to answer a question about 5D
            # input handling, which in this generator is provided by the
            # ``FourierBridgeNetwork`` backbone and has nothing to do with NEX
            # fusion.
            #
            # It also could not return False. ``rep_fusion`` was built on every
            # ``KSpaceColdDiffusionGenerator`` instance, so ``hasattr`` was a
            # constant ``True`` and this "structural check" measured nothing
            # (#1173; CLAUDE.md pitfall #16).
            #
            # Behaviour-preserving: ``FourierBridgeNetwork`` is instantiated at
            # exactly one site (inside that generator), and no other class in
            # the tree defines ``rep_fusion``, so the old disjunction was True
            # for that generator and False for every other -- which is what
            # ``supports_5d_input`` now returns directly.
            #
            # The ``False`` default is deliberate: a generator that does not
            # publish the property has not declared 5D support, so the caller
            # flattens rather than assuming. Absent is reported, not inferred.
            supports_5d = bool(getattr(gen, "supports_5d_input", False))
            # Only keep 5D if the model supports it AND we actually have depth/reps > 1
            if supports_5d and target_batch.shape[-1] > 1:
                should_flatten_5d = False

        if target_batch.ndim >= 5 and should_flatten_5d:
            shape = list(target_batch.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            self.logging_service.log_info(
                f"[5D→4D RESHAPE] Detected 5D target_batch: ({b}, {c}, {h}, {w}, {d})"
            )
            # Flatten: permute (B, C, H, W, D) → (B, D, C, H, W) → reshape to (B*D, C, H, W)
            target_batch = target_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

            if input_batch is not None and input_batch.ndim >= 5:
                input_batch = input_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

            # Update batch_size for downstream processing (timesteps, masks)
            batch_size = b * d

            # Expand scale factor if it was computed per volume
            if scale is not None and scale.ndim == 4 and scale.shape[0] == b:
                scale = scale.repeat_interleave(d, dim=0)

            # Expand contrast_idx if it was provided per volume
            if contrast_idx is not None and contrast_idx.ndim == 1 and contrast_idx.shape[0] == b:
                contrast_idx = contrast_idx.repeat_interleave(d, dim=0)
        elif target_batch.ndim >= 5 and not should_flatten_5d:
            shape = list(target_batch.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            self.logging_service.log_info(
                f"[5D REPETITION] Keeping 5D input_batch: ({b}, {c}, {h}, {w}, {d})"
            )
            # Permute to match Repetition Fusion Block expected dims: [B, Reps, C, H, W]
            target_batch = target_batch.permute(0, 4, 1, 2, 3)
            if input_batch is not None and input_batch.ndim >= 5:
                input_batch = input_batch.permute(0, 4, 1, 2, 3)

        # Log target statistics
        if should_log:
            try:
                if torch.is_complex(target_batch):
                    mag_stat = target_batch.abs()
                elif target_batch.shape[1] % 2 == 0:
                    B, C, H, W = target_batch.shape
                    t_reshaped = (
                        target_batch.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                    )
                    t_complex = torch.view_as_complex(t_reshaped).permute(0, 3, 1, 2)
                    mag_stat = torch.sqrt(
                        torch.sum(t_complex.abs() ** 2, dim=1, keepdim=True) + 1e-8
                    )
                else:
                    mag_stat = target_batch.abs()

                if logger.isEnabledFor(logging.DEBUG):
                    t_mean = mag_stat.mean().item()
                    t_min, t_max = mag_stat.min().item(), mag_stat.max().item()

                    logger.debug(
                        f"[Step {current_step}] Target Stats: "
                        f"MagRange=[{t_min:.4f}, {t_max:.4f}], MeanMag={t_mean:.4f}, Shape={list(target_batch.shape)}"
                    )
            except Exception as e:
                logger.debug(f"[Step {current_step}] Stats logging failed: {e}")

        # Sample or load timesteps
        if batch_data and "timestep" in batch_data:
            timesteps = batch_data["timestep"]
            if isinstance(timesteps, torch.Tensor):
                timesteps = timesteps.to(target_batch.device)
            elif isinstance(timesteps, list | tuple):
                timesteps = torch.tensor(timesteps, device=target_batch.device, dtype=torch.long)
        else:
            timesteps = self.sample_timesteps(batch_size, iteration=current_step).to(
                target_batch.device
            )

        # Prepare forward process based on diffusion type
        is_cold_diffusion = self._is_cold_diffusion()
        is_latent_diffusion = self._is_latent_diffusion()

        # Initialize noise to None — only set for Gaussian / LDM paths
        # (cold diffusion uses deterministic degradation, no additive noise)
        noise: torch.Tensor | None = None

        if is_latent_diffusion:
            # [DATA LEAKAGE FIX] Encode target to latent space and add noise.
            # Previously, model_input was set to raw target_batch (clean GT pixels)
            # without encoding or noise injection — the denoise_net received the
            # clean target and could learn the identity function (zero loss trivially).
            # Correct LDM procedure (Rombach et al. CVPR 2022):
            #   1. Encode GT image → latent z_0 via frozen autoencoder
            #   2. Add Gaussian noise: z_t = q_sample(z_0, t, ε)
            #   3. Denoise_net predicts ε from z_t
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            if hasattr(gen, "encode_to_latent"):
                latent_z0 = gen.encode_to_latent(target_batch)
            else:
                # CLAUDE.md pitfall #9 — no silent fallback. A latent
                # diffusion strategy on a generator without
                # ``encode_to_latent`` is a configuration bug: the
                # pixel-space fallback produces a different training
                # objective and quietly corrupts results. Fail loud so
                # the smoke wrapper catches it.
                raise TypeError(
                    f"[Latent Diffusion] Generator {type(gen).__name__} does not "
                    "implement encode_to_latent(); cannot run latent-diffusion "
                    "training. Either (a) switch to a pixel-space diffusion "
                    "strategy via training.training_mode='diffusion' / set "
                    "training.diffusion.latent=false, or (b) wrap the generator "
                    "in a VAE that exposes encode_to_latent(target_batch) -> "
                    "latent tensor."
                )

            noise = torch.randn_like(latent_z0)
            noisy_images = self.q_sample(latent_z0, timesteps, noise)
            model_input = noisy_images
        elif is_cold_diffusion:
            # Which tensor the forward process degrades (issue #536). VALIDATION
            # always degrades ``input_batch`` — re-masking the fully-sampled target
            # at validation leaks ground truth — so degrading ``target_batch`` here
            # trained the model on a distribution it is never evaluated on: with
            # ``target_mode: phase_aligned_mean`` the target's observed lines are
            # sqrt(N)-averaged and noise-free while the measurement's are
            # single-rep. From this arm's step-1 snapshot, std(rep0)=0.0950 vs
            # std(NEX mean)=0.0756, a per-component residual std of 0.0575 —
            # comparable to the signal in this log-scaled k-space.
            diffusion_target = (
                input_batch if self._cold_degradation_source() == "input" else target_batch
            )

            # [SSOT] Check if generator has a kspace_process (unified physics)
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            if hasattr(gen, "kspace_process") and gen.kspace_process is not None:
                # Use unified physics process for consistent masks
                noisy_images, mask = gen.kspace_process.q_sample(diffusion_target, timesteps)
            else:
                # Legacy path
                mask = self.generate_and_process_mask(
                    batch_size=batch_size,
                    timesteps=timesteps,
                    target_shape=diffusion_target.shape,
                    current_step=current_step,
                    batch_data=batch_data,
                )
                noisy_images = self.q_sample(diffusion_target, timesteps, mask)

            model_input = noisy_images
        else:
            # Check if this is an image-space ColdDiffusion model with custom degradation
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            from spectramr.models.diffusion.cold_diffusion import ColdDiffusion

            if isinstance(gen, ColdDiffusion) and gen.degradation_type != "noise":
                # Delegate entirely to the model's native progressive degradation (e.g., physical, blur)
                noisy_images = gen._degrade(target_batch, timesteps)
                model_input = noisy_images
            else:
                noise = torch.randn_like(target_batch)
                noisy_images = self.q_sample(target_batch, timesteps, noise)
                model_input = noisy_images

        # Handle k-space cold diffusion conditioning
        if "kspace_cold_diffusion" in str(self.config.model.model_type).lower():
            # [PHYSICS INTEGRATION] Explicitly embed physical geometry via coil sensitivity maps.
            # Try all key names that M4Raw / TorchIO collation pipelines may use.
            smaps = None
            if batch_data is not None:
                for _smap_key in (
                    "sensitivity",
                    "smaps",
                    "sensitivity_maps",
                    "coil_sensitivity",
                ):
                    sens_tio = batch_data.get(_smap_key)
                    if sens_tio is not None:
                        break
                if sens_tio is not None:
                    smaps = sens_tio.tensor if hasattr(sens_tio, "tensor") else sens_tio
                    if smaps.ndim >= 5 and should_flatten_5d:
                        s_shape = list(smaps.shape)
                        smaps = smaps.permute(0, 4, 1, 2, 3).reshape(
                            s_shape[0] * s_shape[-1],
                            s_shape[1],
                            s_shape[-3],
                            s_shape[-2],
                        )
                    elif smaps.ndim >= 5 and not should_flatten_5d:
                        smaps = smaps[..., 0]
                    smaps = smaps.to(target_batch.device)

            if smaps is not None:
                self._current_smaps = smaps
            else:
                # [ESPIRiT MAPS] Calibrate from the FULLY-SAMPLED reference only.
                # Coil maps are acceleration-invariant, so the aliased /
                # diffusion-degraded tensor must never seed calibration — prefer
                # the 'kspace' alias, else the clean target; RAISE rather than
                # fall back to noisy_images (CLAUDE.md #9/#16).
                acs_kspace = batch_data.get("kspace") if batch_data is not None else None
                if acs_kspace is None:
                    acs_kspace = target_batch
                if not isinstance(acs_kspace, torch.Tensor):
                    raise ValueError(
                        "[S-Maps] No fully-sampled k-space/target for ESPIRiT "
                        "calibration; refusing to degrade to the diffusion-"
                        "corrupted input (CLAUDE.md #9/#16)."
                    )
                acs_kspace_t: torch.Tensor = acs_kspace.to(target_batch.device)
                # Extract spatial dims BEFORE any dtype conversion so h/w are always defined
                _acs_shape = acs_kspace_t.shape
                h, w = _acs_shape[-2], _acs_shape[-1]
                if not torch.is_complex(acs_kspace_t):
                    # real-stacked → complex: (B, 2*C, H, W) → (B, C, H, W) complex
                    b, c = _acs_shape[0], _acs_shape[1]
                    c2 = c // 2
                    if c2 < 1:
                        # Single-channel magnitude data: treat as 1-channel complex with zero imag
                        acs_kspace_t = torch.complex(acs_kspace_t, torch.zeros_like(acs_kspace_t))
                    else:
                        acs_kspace_t = torch.view_as_complex(
                            acs_kspace_t.view(b, c2, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
                        )

                # Memoized ACS-cropped estimate (the acs_only crop lives in the
                # cache fn); honors physics.coil_processing.estimation.* (#15).
                smaps = self._estimate_smaps_cached(acs_kspace_t, h, w)
                self._current_smaps = smaps
                self.logging_service.log_debug(
                    "[S-Maps] Estimated dynamically via ESPIRiT: shape=%s",
                    smaps.shape,
                )
                smaps = self._current_smaps

            # [ONE OWNER — CLAUDE.md #17] Concatenate only when the generator
            # was actually SIZED for a doubled stack. This strategy used to
            # concatenate whenever S-maps existed, with no knowledge of the
            # backbone's width, so the 6 internal-DC arms (diff_varnet x4,
            # diff_varnet_kan x2) fed a 2*C stack into a C-wide backbone every
            # training step; ``FourierBridgeNetwork`` absorbed the mismatch by
            # building an untrained 1x1 ChannelAdapter, which it now refuses to
            # do. Generators that do not publish the attribute keep the previous
            # behaviour (this method serves every diffusion paradigm).
            _gen_for_width = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            from spectramr.models.generators.kspace_cold_diffusion_generator import (
                model_expects_smaps_concat,
            )

            # default=True: this strategy also drives non-cold diffusion
            # generators that carry neither attribute, and it concatenated for
            # them unconditionally before the width contract existed.
            _expects_concat = model_expects_smaps_concat(_gen_for_width, default=True)
            if smaps is not None and _expects_concat:
                if torch.is_complex(noisy_images) and not torch.is_complex(smaps):
                    shape_except_c = smaps.shape[2:]
                    num_spatial_dims = len(shape_except_c)
                    permute_dims = [0, 1] + list(range(3, 3 + num_spatial_dims)) + [2]
                    smaps = torch.view_as_complex(
                        smaps.view(smaps.shape[0], -1, 2, *shape_except_c)
                        .permute(*permute_dims)
                        .contiguous()
                    )
                elif not torch.is_complex(noisy_images) and torch.is_complex(smaps):
                    shape_except_c = smaps.shape[2:]
                    num_spatial_dims = len(shape_except_c)
                    permute_dims = [0, 1, num_spatial_dims + 2] + list(
                        range(2, 2 + num_spatial_dims)
                    )
                    smaps = (
                        torch.view_as_real(smaps)
                        .permute(*permute_dims)
                        .reshape(smaps.shape[0], -1, *shape_except_c)
                        .contiguous()
                    )

                # DEBUG SHAPES — informational only; was previously logged
                # at error severity (showed up as ✖ ERR in cluster logs)
                # despite never indicating a failure.
                logger.debug(
                    f"[SHAPE DEBUG] noisy_images: {noisy_images.shape}, smaps: {smaps.shape}"
                )

                # Align smaps dimensions with noisy_images if one is [B, C, H, W, D] and the other is [B, D, C, H, W]
                if (
                    smaps.dim() == 5
                    and noisy_images.dim() == 5
                    and smaps.shape != noisy_images.shape
                ):
                    if (
                        smaps.shape[-1] == noisy_images.shape[1]
                        and smaps.shape[1] == noisy_images.shape[2]
                    ):
                        smaps = smaps.permute(0, 4, 1, 2, 3).contiguous()
                        logger.debug(
                            f"[SHAPE ALIGN] Permuted smaps to {smaps.shape} to match noisy_images"
                        )

                try:
                    # If 5D and not flattened, shape is [B, D, C, H, W], so channels are at dim=2
                    concat_dim = (
                        2
                        if noisy_images.dim() == 5
                        and noisy_images.shape[1] > 1
                        and noisy_images.shape[2] <= 4
                        else 1
                    )
                    # [DOMAIN] ``noisy_images`` is k-space here while the maps
                    # are image-domain.  Concatenating them raw aligns a
                    # sensitivity at image pixel (x, y) with the spatial
                    # frequency (kx, ky) = (x, y) — a meaningless
                    # correspondence for a conv, and the single entry transform
                    # in FourierBridgeNetwork mistreats one half of the stack
                    # whichever way ``force_pure_kspace`` is set.  FFT the maps,
                    # match their level to the half they ride next to, and cap
                    # their amplitude.  The rebind is LOCAL: ``_current_smaps``
                    # stays image-domain for the SENSE projection / FiLM stash.
                    smaps_k, _ = prepare_smaps_for_kspace_conditioning(
                        smaps.detach(), noisy_images, channel_dim=concat_dim
                    )
                    model_input = torch.cat([noisy_images, smaps_k], dim=concat_dim)
                except Exception as e:
                    logger.error(
                        f"Failed to concat. noisy_images: {noisy_images.shape}, smaps: {smaps.shape}"
                    )
                    raise e
            else:
                model_input = noisy_images
            # Log model input statistics if needed
            if current_step % self.config.logging.intervals.log == 0:
                if logger.isEnabledFor(logging.INFO):
                    try:
                        with torch.no_grad():
                            # Convert complex tensors to magnitude before stats
                            stats_input = (
                                model_input.abs() if torch.is_complex(model_input) else model_input
                            )
                            stats_tensor = torch.stack(
                                [
                                    stats_input.mean(),
                                    stats_input.std(),
                                    stats_input.min(),
                                    stats_input.max(),
                                ]
                            )
                            mi_mean, mi_std, mi_min, mi_max = stats_tensor.detach().cpu().numpy()

                        mag_info = ""
                        if not torch.is_complex(model_input) and model_input.shape[1] % 2 == 0:
                            b, c, h, w = model_input.shape

                            complex_view = model_input.view(b, c // 2, 2, h, w)
                            mag = torch.sqrt(
                                complex_view[:, :, 0] ** 2 + complex_view[:, :, 1] ** 2
                            )
                            with torch.no_grad():
                                mag_stats = torch.stack([mag.mean(), mag.max()])
                                mag_mean, mag_max = mag_stats.detach().cpu().numpy()
                            mag_info = f" | Mag(Mean/Max): {mag_mean:.4f}/{mag_max:.4f}"

                        self.logging_service.log_info(
                            f"📥 Model Input [Step {current_step}] | "
                            f"Shape: {list(model_input.shape)} | "
                            f"Stats: μ={mi_mean:.4f}, σ={mi_std:.4f}, Range=[{mi_min:.4f}, {mi_max:.4f}]{mag_info}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to compute model_input stats: {e}")

            # Declare the degraded tensor as THE model input. The wrapper
            # (``BaseTrainingStrategy._compute_losses``) emits it under
            # ``snapshot_model_input_tag`` -- the tag ``first_steps`` already
            # points readers to. Declaring rather than emitting here is what
            # closes the dangling pointer: this method is reachable only through
            # ``_compute_losses_impl``, the hook seven subclasses override, so
            # every one of them inherited the ``diffusion_step`` promise and
            # emitted nothing at its target.
            #
            # When ``model_input`` is the concat ``[noisy_kspace || smaps]``,
            # rendering it as a single k-space tensor produces a
            # superposition of brain (iFFT of k-space half) + bright DC
            # spike (iFFT of image-domain smaps half). Pass the parts as
            # separate keys so the visualizer can apply the right
            # transform to each. See findings booklet 2026-05-05 VIS-1.
            #
            # UNCONDITIONAL, and that is load-bearing. Two owners have gated this
            # on a cadence: first ``self._cached_log_interval * 5`` -- i.e.
            # ``logging.intervals.log``, a *logging* knob arms routinely set to
            # 5000, which put this snapshot at step 25 000 and made it
            # unreachable in every shorter run (#706's shape); then
            # ``snapshot_step_is_due``, the RIGHT cadence but still a second
            # gate. The wrapper's contract check is deliberately not
            # step-dependent, so any cadence here makes the two disagree: under
            # ``snapshots.interval_steps: 100`` this would declare on step 0 and
            # the wrapper would raise on step 1 -- a violation manufactured by
            # the gate. Cadence belongs to ``save_debug_snapshot`` alone, and a
            # declaration is a dict of references plus a tensor view, so there is
            # no per-step cost to avoid (non-negotiable 9).
            snap: dict[str, torch.Tensor] = {
                "noisy_kspace": noisy_images,
                "target": target_batch,
                "mask": mask,
            }
            # ``model_input`` differs from ``noisy_images`` only when
            # smaps were concatenated above. Surface the concatenated half
            # separately so the previewer renders it in its OWN domain.
            _segments: dict[str, list[ChannelSegment]] = {}
            smaps_concatenated = False
            if model_input is not noisy_images:
                if model_input.shape[1] > noisy_images.shape[1]:
                    smaps_part = model_input[:, noisy_images.shape[1] :]
                    snap["smaps"] = smaps_part
                    # Record the REAL model input too, not only its two halves.
                    # Splitting the parts (VIS-1, above) kept the PICTURE right
                    # but left the artifact unable to answer the first question a
                    # reader asks of it -- how many channels does the network
                    # actually receive? -- because no row in the table was ever
                    # the concatenated tensor. Reading `noisy_kspace` at 8ch beside
                    # `smaps` at 8ch, with the 16ch concat nowhere, is what makes
                    # "is it single coil? is it a squashed tensor?" unanswerable
                    # from the snapshot. `channel_segments` now renders each half
                    # in its own domain, so the faithful record and the readable
                    # picture stop being in tension (non-negotiable 14).
                    snap["model_input"] = model_input
                    _segments["model_input"] = [
                        ("kspace", noisy_images.shape[1], True, True),
                        # This half is ``fft2c``'d maps since #1327, so it is
                        # k-space -- consistent with ``smaps`` being listed in
                        # ``in_kspace_keys`` just below. Declaring it
                        # image-domain here would have the SAME tensor rendered
                        # two ways in one artifact. It is NOT ``log1p``'d
                        # though: it reaches the concat straight from
                        # ``fft2c``, while ``noisy_images`` descends from the
                        # ``apply_kspace_normalization``-compressed target. The
                        # fourth field is what says so.
                        ("smaps", smaps_part.shape[1], True, False),
                    ]
                    smaps_concatenated = True
            self._declare_model_input(
                snap,
                channel_segments=_segments,
                # Explicit, not None: the fallback is a ``"kspace"`` substring
                # match over key names, which would miss ``target`` and
                # ``smaps``. ``mask`` is omitted on purpose -- a sampling mask
                # IFFTs to noise. ``smaps`` IS listed whenever it was
                # concatenated: this key holds the half actually fed to the
                # model, and since ``prepare_smaps_for_kspace_conditioning``
                # that half is k-space, not the image-domain maps. (The
                # separate ``_current_smaps``-sourced snapshot elsewhere in this
                # class stays image-domain and stays off this list -- VIS-1
                # is about naming the domain of the tensor you actually stored.)
                in_kspace_keys=(
                    {"noisy_kspace", "target", "smaps"}
                    if smaps_concatenated
                    else {"noisy_kspace", "target"}
                ),
                # Explicit for the same reason, pointing the other way.
                # ``smaps`` is k-space (above) but is deliberately ABSENT here:
                # it arrives straight from
                # ``prepare_smaps_for_kspace_conditioning``'s ``fft2c`` and
                # never passed through ``apply_kspace_normalization``'s
                # ``log1p``, whereas ``noisy_kspace`` and ``target`` both
                # descend from the compressed target. ``model_input`` IS listed
                # -- it is a mix, and its ``channel_segments`` above carry the
                # per-half answer. Leaving this None would decompress the maps
                # with ``expm1``, which is what makes the standalone
                # ``smaps.png`` and the ``model_input__smaps.png`` segment --
                # the SAME tensor -- come out as two different pictures.
                log_scaled_keys={"noisy_kspace", "target", "model_input"},
                # Which tensor the forward process degraded, stamped into the
                # artifact that shows the result -- reading the PNG should not
                # require cross-referencing resolved_config.json to learn
                # whether the measurement or the (NEX-averaged) target was
                # re-masked. ``mask``'s mean in the same JSON is the sampling
                # fraction, i.e. 1/R for the rung this step happened to draw.
                # ``model_input_key`` names the key that IS the model input, so
                # it must follow what was actually recorded. It read
                # "noisy_kspace" unconditionally, which is true only when no
                # smaps were concatenated; with conditioning on (the default for
                # kspace_cold_diffusion, and what this arm runs) the network is
                # fed the 16-channel concat and `noisy_kspace` is merely its
                # first half. A reader trusting the label concluded the model
                # sees 8 channels when the first conv has 16 input planes.
                extra={
                    "degradation_source": self._cold_degradation_source(),
                    "model_input_key": cold_model_input_key(snap),
                    "note": (
                        (
                            "'model_input' is the tensor fed to the model: "
                            "cat([noisy_kspace, smaps], dim=1). 'noisy_kspace' "
                            "is its k-space half (the degradation_source tensor "
                            "zero-filled by the forward process, "
                            "q_sample = x_0 * mask); 'smaps' is its "
                            "conditioning half -- fft2c of the coil maps, "
                            "level-matched and amplitude-capped (#1327), so "
                            "k-space like the first half. "
                        )
                        if "model_input" in snap
                        else (
                            "'noisy_kspace' is the tensor fed to the model: the "
                            "degradation_source tensor zero-filled by the "
                            "forward process (q_sample = x_0 * mask). "
                        )
                    )
                    + (
                        "The base strategy's 'first_steps/input_prepared' is "
                        "PRE-degradation -- _prepare_model_input only converts "
                        "domain."
                    ),
                    # DECLARED vs APPLIED for the conditioning half
                    # (non-negotiable 14). The per-sample RMS gain is a device
                    # tensor and this declaration runs EVERY step, so reading it
                    # numerically here would be a per-step sync (non-negotiable
                    # 9). It is recoverable from the artifact instead: the gain
                    # is by construction rms('noisy_kspace') / rms(fft2c(maps)),
                    # and both tensors are in this same snapshot.
                    "smaps_conditioning": (
                        {
                            "domain": "kspace",
                            "transform": "fft2c",
                            "level": "per-sample RMS matched to 'noisy_kspace'",
                            "amplitude_cap_x_reference_peak": (SMAP_KSPACE_PEAK_RATIO),
                        }
                        if smaps_concatenated
                        else None
                    ),
                },
            )

        return (
            input_batch,
            target_batch,
            model_input,
            noisy_images,
            timesteps,
            is_cold_diffusion,
            is_latent_diffusion,
            mask,
            scale,
            contrast_idx,
            noise,
        )

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Perform a single validation step for Diffusion models.

        Validates the model by generating predictions from undersampled/noisy inputs
        and comparing them to ground truth.

        Key features:
        -   **Deterministic Masking:** Uses pre-computed masks from the dataloader to prevent leakage.
        -   **Physics-Compliant Metrics:** Computes metrics in physical units (no arbitrary scaling).
        -   **TensorBoard Logging:** Logs sample images (predictions, targets, error maps).

        Args:
            input_batch: The undersampled/noisy validation input.
            target_batch: The ground truth to grade against.
            batch_data: The batch these tensors were lifted out of, forwarded by
                ``select_validation_extra_fields`` on the strength of this
                declaration. Needed to read the loader's published
                ``kspace_scale`` so the compensator does not re-normalize an
                already-normalized batch. ``None`` when a caller does not supply
                it; every downstream read tolerates that.
            **kwargs: Additional context (``batch_idx`` for logging control).

        Returns:
            A dictionary of validation metrics (e.g., ``val_psnr``, ``val_ssim``).
        """
        batch = (input_batch, target_batch)
        batch_idx = kwargs.get("batch_idx", 0)  # Extract batch_idx for logging control

        # NOTE: there used to be a shim here that tried to reconstruct batch_data
        # from `batch` when it arrived None. It could never fire: `batch` is the
        # tuple assigned on the line above, so `isinstance(batch, dict)` and
        # `hasattr(batch, "mask"/"metadata")` were all False by construction. It
        # read as a safety net while guaranteeing batch_data stayed None for the
        # entire validation path. `batch_data` is now a declared parameter, which
        # is what actually makes the seam live -- do not reintroduce a guard on
        # `batch` here (pitfall #16).

        # Handle potential None return from _prepare_validation_data (legacy support)
        prep_result = self._prepare_validation_data(batch, input_batch, target_batch, batch_data)
        if prep_result is None:
            # Fallback if _prepare_validation_data returns None (legacy/mock path)
            scale_factor = 1.0
        else:
            input_batch, target_batch, scale_factor = prep_result

        # Ensure model is in eval mode for validation
        self.generator_model.eval()

        should_flatten_5d = True

        if input_batch.ndim >= 5:
            # Structurally check if model natively supports 5D (e.g. RepetitionFusion)
            gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            # Ask the generator whether it can consume 5D -- NOT whether it
            # owns a repetition-fusion layer. Those are two different
            # invariants and this disjunction conflated them (non-negotiable
            # 17): it reached for ``rep_fusion`` to answer a question about 5D
            # input handling, which in this generator is provided by the
            # ``FourierBridgeNetwork`` backbone and has nothing to do with NEX
            # fusion.
            #
            # It also could not return False. ``rep_fusion`` was built on every
            # ``KSpaceColdDiffusionGenerator`` instance, so ``hasattr`` was a
            # constant ``True`` and this "structural check" measured nothing
            # (#1173; CLAUDE.md pitfall #16).
            #
            # Behaviour-preserving: ``FourierBridgeNetwork`` is instantiated at
            # exactly one site (inside that generator), and no other class in
            # the tree defines ``rep_fusion``, so the old disjunction was True
            # for that generator and False for every other -- which is what
            # ``supports_5d_input`` now returns directly.
            #
            # The ``False`` default is deliberate: a generator that does not
            # publish the property has not declared 5D support, so the caller
            # flattens rather than assuming. Absent is reported, not inferred.
            supports_5d = bool(getattr(gen, "supports_5d_input", False))
            if supports_5d and input_batch.shape[-1] > 1:
                should_flatten_5d = False

            # Reject empty 5D batches up-front (mirrors target_batch guard).
            if 0 in tuple(input_batch.shape):
                raise RuntimeError(
                    f"5D input_batch has a zero-sized dim: shape={tuple(input_batch.shape)}. "
                    f"Dataloader produced an empty patch — inspect the manifest record at "
                    f"this iteration before retrying."
                )

        if input_batch.ndim >= 5 and should_flatten_5d:
            shape = list(input_batch.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            self.logging_service.log_info(
                f"[5D→4D RESHAPE] Detected 5D input_batch: ({b}, {c}, {h}, {w}, {d})"
            )
            input_batch = input_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            self.logging_service.log_info(
                f"[5D→4D RESHAPE] Flattened input_batch: ({b}, {c}, {h}, {w}, {d}) → ({b * d}, {c}, {h}, {w})"
            )

            if target_batch.ndim >= 5:
                target_batch = target_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                self.logging_service.log_info(
                    f"[5D→4D RESHAPE] Flattened target_batch: ({b}, {c}, {h}, {w}, {d}) → ({b * d}, {c}, {h}, {w})"
                )
        elif input_batch.ndim >= 5 and not should_flatten_5d:
            shape = list(input_batch.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            self.logging_service.log_info(
                f"[5D REPETITION] Keeping 5D input_batch for validation: ({b}, {c}, {h}, {w}, {d})"
            )
            if target_batch.ndim >= 5:
                # Target must collapse Repetitions so we can compute loss
                target_batch = target_batch[..., 0]

            if scale_factor.ndim == 4 and scale_factor.shape[0] == b:
                scale_factor = scale_factor.repeat_interleave(d, dim=0)
                self.logging_service.log_info(
                    f"[5D→4D RESHAPE] Expanded scale_factor: ({b}, 1, 1, 1) → ({b * d}, 1, 1, 1)"
                )

        # Cascading acceleration levels: progressively increasing undersampling.
        # Use 3 representative acceleration levels to reduce GPU memory pressure.
        # Full 6-level cascade ([2,4,8,12,16,32]) requires 6 sequential forward passes;
        # 3 levels cover low/mid/high undersampling regimes with 50% less memory overhead.
        #
        # Schedule-aware inversion (2026-05-28, experiment_11 mosaic triage):
        # The previous hard-coded linear inverse ``t = T*(R-base)/(max-base)``
        # only matched the forward schedule when ``acceleration.schedule_type``
        # was ``linear``. For ``schedule_type: step`` with a non-uniform
        # ``acceleration_range`` it picked timesteps that the step schedule
        # then decoded as a DIFFERENT acceleration (R=8 → t=200 → step decodes
        # back to R=4), so the ``val_*_<R>x`` metric columns silently
        # mis-labeled the acceleration. The cascade now routes through
        # ``KSpaceAccelerator.timestep_for_acceleration`` (the SSOT inverse)
        # via the strategy's mask generator.
        #
        # #697: the list was declared here AND in `pipelines/training_loop.py`,
        # agreeing only by coincidence. One SSOT now; a divergence would have
        # mislabelled every column, and a mislabelled number still reads as a
        # number. `self._last_cascade_rows` carries the same sweep in tall form
        # (level and timestep as DATA) for the CSV -- see the module docstring
        # of `core/cascading_validation.py` for why that shape, given the
        # schedule-inversion bug described just above.
        # #1394: the ladder is now DECLARED. It was a module constant, so an
        # arm could widen `undersampling.acceleration_range` for TRAINING while
        # validation stayed pinned at 2/8/32 with no spelling to say otherwise
        # -- the two halves of the loop disagreed by construction.
        # `resolve_cascade_levels` is the sole reader of `CASCADING_LEVELS`
        # (non-negotiable 17); an arm that declares nothing still gets exactly
        # (2, 8, 32), so the default path is unchanged.
        _CASCADING_LEVELS: list[int | float] = list(resolve_cascade_levels(self.config.validation))
        # Stamp the RESOLVED ladder, not the declared one. A run that skips a
        # rung (step schedule, off-grid level) is diagnosable from the log
        # alone only if the log says what was asked for in the first place.
        self.logging_service.log_info(
            f"[Cascade] validation ladder = {_CASCADING_LEVELS} "
            f"({'declared via validation.cascade.levels' if self._cascade_levels_declared() else 'framework default'})"
        )
        cascade_rows: list[dict[str, Any]] = []

        # Free fragmented CUDA memory accumulated during training before validation.
        # Kept on purpose: the one-shot training->validation phase boundary
        # (cross-phase fragmentation), pinned by
        # test_diffusion_chunked_sampling_no_empty_cache_2026_08.py.
        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            batch_size = input_batch.size(0)
            # ✅ SSOT: Direct access to model.in_channels
            in_channels = self.config.model.in_channels  # noqa: F841

            num_timesteps = self.num_timesteps
            # Only the fallback inverse and the accel-agnostic pass below use
            # this band now; the schedule-aware answer is no longer clamped
            # into it (#1295). See `core.cascading_validation.training_band`.
            min_t, max_t = training_band(num_timesteps)

            all_metrics: dict[str, float] = {}

            # ── Image / latent-translation arms (no k-space acceleration) ──
            # The cascade below evaluates k-space undersampling robustness
            # (R=2/8/32x) and dereferences ``self.config.acceleration``. An
            # image-domain translator (e.g. the ULF→HF latent-diffusion arm) has
            # no ``acceleration`` block, so the cascade both crashes
            # (AttributeError on ``base_acceleration``, diffusion.py:2746) and,
            # even if guarded, would only ever emit ``val_*_{R}x`` columns —
            # never the plain ``val_psnr`` the arm's ``early_stopping.metric``
            # selects on (pitfall #18). Run one accel-agnostic pass that emits
            # unsuffixed metrics instead. The timestep is inert for the latent
            # sampler (``gen.sample`` runs the full reverse loop; see
            # ``_generate_validation_prediction``), so any value is fine.
            if self.config.undersampling is None:
                timestep = torch.full(
                    (batch_size,),
                    max_t,
                    dtype=torch.long,
                    device=input_batch.device,
                )
                hr_fakes = self._generate_validation_prediction(
                    input_batch,
                    target_batch,
                    timestep,
                    batch_data,
                    kwargs,
                    scale_factor,
                )
                if hr_fakes is None:
                    return {"validation_error": 1.0}
                single_metrics = self._compute_validation_metrics(
                    hr_fakes,
                    target_batch,
                    input_batch,
                    timestep,
                    batch_data,
                    scale_factor,
                    batch_idx=batch_idx,
                )
                # The probe runs on BOTH return paths. Emitting it only from
                # the cascade branch would make the key set depend on
                # `config.undersampling`, which is a per-arm config value and so
                # still rank-invariant -- but it would silently drop the
                # terminal-rung readout for every arm that takes this branch,
                # which is the blind spot this whole change exists to close.
                single_metrics.update(
                    self._t0_pre_dc_probe_metrics(
                        target_batch, input_batch, batch_data, scale_factor, batch_idx
                    )
                )
                return single_metrics

            # L4 input-dependence gate: collect the per-level predictions so
            # we can measure whether the output actually VARIES across the
            # acceleration cascade (catching the measurement-independent DC
            # blob). Only populated when the gate is enabled. The DC blob is an
            # IMAGE-domain phenomenon — the net can wiggle high-frequency
            # k-space (so a k-space spread looks healthy) while the
            # image-dominating low frequencies stay nearly constant — so the
            # captured predictions are converted to iFFT RSS magnitude IMAGES
            # below; measuring the gate on raw k-space would under-detect it.
            _l4_enabled = self.config.validation.gates.input_dependence_tol is not None
            _l4_needs_ifft = True
            if _l4_enabled:
                try:
                    from spectramr.infrastructure.training.utils.domain_inference import (
                        needs_ifft_for_visualization,
                    )

                    _l4_needs_ifft, _ = needs_ifft_for_visualization(self.config)
                except Exception:  # pragma: no cover — defensive only
                    _l4_needs_ifft = (self.config.model.input_type or "kspace").lower() == "kspace"
            cascade_predictions: list[torch.Tensor] = []

            # SSOT inverse — match the YAML's ``acceleration.schedule_type``
            # so each cascade level's mask is what the column label claims.
            # Falls back to the historical linear formula only when the
            # mask generator (and therefore the accelerator) is not wired —
            # e.g. legacy non-cold-diffusion strategies that share this loop.
            sched_accelerator = None
            mask_gen = getattr(self, "mask_generator", None)
            if mask_gen is not None and hasattr(mask_gen, "_get_accelerator"):
                try:
                    sched_accelerator = mask_gen._get_accelerator(None)
                except Exception as _exc:  # pragma: no cover — defensive only
                    self.logging_service.log_warning(
                        f"[Cascading Validation] Could not resolve schedule-"
                        f"aware accelerator ({_exc!r}); falling back to "
                        f"linear inverse."
                    )
                    sched_accelerator = None

            # In-distribution cascade levels + (opt-in) held-out severity grid
            # for the H4 robustness eval (WS1-core-03 / plan §12 / A5.8). Held-out
            # points reuse the EXACT same per-severity eval body below; only the
            # metric suffix and the DC-blob-gate bookkeeping differ, so the
            # in-distribution columns/keys are unchanged.
            _severity_points = [(a, False) for a in _CASCADING_LEVELS]
            _severity_points += [(s, True) for s in self._held_out_severity_points()]
            # Which in-distribution rungs actually produced columns, so "the
            # cascade was complete" stops being an assumption made by everything
            # downstream (#1303). An exit from the loop body below SHOULD name
            # its reason here, but the verdict does not rest on it doing so:
            # `reconcile_skipped_levels` derives whatever no exit accounted for
            # before the two are read together. Held-out points are a robustness
            # readout, not part of the contract, so they are tracked separately
            # and never make the in-distribution cascade look incomplete.
            _levels_evaluated: list[int] = []
            _levels_skipped: dict[int, str] = {}
            for accel, _is_heldout in _severity_points:
                # Free memory between cascading levels to avoid OOM on large models.
                # Kept on purpose (documented OOM history), pinned by
                # test_diffusion_chunked_sampling_no_empty_cache_2026_08.py.
                torch.cuda.empty_cache()

                base_accel = self.config.undersampling.base_acceleration
                max_accel = self.config.undersampling.max_acceleration
                span = max(1.0, max_accel - base_accel)

                # ── Declared vs realised (#1295) ──
                # The SSOT inverse already returns a legal timestep in
                # ``[0, T-1]`` (its documented contract) and, for
                # ``schedule_type='step'``, one it has re-asked the forward
                # schedule to confirm. Post-clamping that answer into the
                # ``[T/10, 9T/10]`` training band -- which is what stood here --
                # moved the mask off the rung while the row kept the requested
                # label, so `val_*` at ``acceleration_level=2`` measured R=2.226
                # and ``acceleration_level=32`` measured R=29.444 on the
                # experiment_11 attention arms (T=29, band [2, 27]).
                #
                # The band was never a validity constraint: it arrived in
                # 1031387cc4 alongside the cascade itself, with no separate
                # rationale, and t=1 / t=28 are ordinary trained timesteps. It
                # actively defeats those arms' ladders, which were built with
                # 2/8/32 as EXACT rungs precisely so no level would be lost.
                # So: no clamp on the SSOT path. The band survives only on the
                # linear fallback, where there is no schedule to be faithful to.
                acceleration_realized: float | None = None
                if sched_accelerator is not None and hasattr(
                    sched_accelerator, "timestep_for_acceleration"
                ):
                    try:
                        t_ideal = float(sched_accelerator.timestep_for_acceleration(float(accel)))
                    except ValueError as _exc:
                        # `timestep_for_acceleration` raises ValueError for TWO
                        # different situations, and collapsing them into one skip
                        # is what #1303 is about:
                        #
                        #  (a) the rung is not in `acceleration_range` at all.
                        #      The arm declared a ladder that does not contain
                        #      this severity — a limitation of the CONFIG, not a
                        #      defect. Skipping is the only honest answer (a
                        #      linear fallback would lie, snapping would report a
                        #      neighbouring bucket under this bucket's label), but
                        #      it must leave a trace.
                        #
                        #  (b) #1171: the rung IS declared, and is realised at NO
                        #      timestep because `len(acceleration_range)` exceeds
                        #      `num_timesteps`. That is a config DEFECT, and the
                        #      guard in `timestep_for_acceleration` exists
                        #      precisely to catch it — catching it here and
                        #      continuing converts a raised error back into the
                        #      silent skip it was written to prevent.
                        #
                        # Discriminated by STATE, never by message text.
                        if self._level_is_declared(sched_accelerator, accel):
                            raise
                        self.logging_service.log_warning(
                            f"[Cascading Validation] R={accel}x is not in "
                            f"acceleration.acceleration_range under "
                            f"schedule_type='step' ({_exc}); skipping level."
                        )
                        if not _is_heldout:
                            _levels_skipped[accel] = "not-in-acceleration_range"
                        continue
                    t_used = int(t_ideal)

                    # Make the inverse prove it. A continuous schedule inverts
                    # by clamping ``progress`` into [0, 1], so a request above
                    # ``max_acceleration`` silently returns T-1 -- the same lie
                    # one layer down from the clamp just removed. Verified: on a
                    # linear T=1000 arm capped at R=8, both R=16 and R=64 return
                    # t=999 and decode back to 8.0.
                    _rt = check_round_trip(
                        requested=float(accel),
                        timestep=t_used,
                        forward=sched_accelerator.get_acceleration_factor,
                        num_timesteps=num_timesteps,
                    )
                    if not _rt.ok:
                        self.logging_service.log_warning(
                            f"[Cascading Validation] R={accel}x resolves to "
                            f"t={t_used}, which the schedule decodes back to "
                            f"R={_rt.realized:.4f} — {_rt.reason} (tolerance "
                            f"{_rt.tolerance:.4g}; "
                            f"[base_acceleration, max_acceleration] = "
                            f"[{base_accel}, {max_accel}]). Skipping the level "
                            f"rather than publishing a mislabelled column."
                        )
                        if not _is_heldout:
                            _levels_skipped[accel] = (
                                f"round-trip-mismatch: realized R={_rt.realized:.4f}"
                            )
                        continue
                    if _rt.realized <= IDENTITY_ACCELERATION + 1e-6:
                        self.logging_service.log_warning(
                            f"[Cascading Validation] R={accel}x realises "
                            f"R={_rt.realized:.4f} at t={t_used} -- a fully "
                            f"sampled (identity) mask, which measures nothing "
                            f"and would inflate best-metric selection. "
                            f"Skipping the level."
                        )
                        if not _is_heldout:
                            _levels_skipped[accel] = f"realizes-identity-mask: R={_rt.realized:.4f}"
                        continue
                    acceleration_realized = _rt.realized
                else:
                    # Linear fallback for legacy strategies without a mask
                    # generator. Same formula, and the same band clamp, as
                    # before this fix: with no accelerator wired there is no
                    # forward schedule to check the answer against, so the
                    # conservative band is all that is left. `t_ideal` here can
                    # legitimately exceed T-1 (the formula is unbounded above
                    # `max_acceleration`), which is what the clamp is for.
                    t_ideal = num_timesteps * (accel - base_accel) / span
                    t_used = legacy_linear_timestep(
                        acceleration=accel,
                        base_acceleration=base_accel,
                        max_acceleration=max_accel,
                        num_timesteps=num_timesteps,
                    )
                    if t_used != int(t_ideal) and not self._cascade_clamp_warned:
                        # Once per strategy, not once per validation cycle: the
                        # cascade runs every val batch and this would otherwise
                        # be thousands of identical lines.
                        self._cascade_clamp_warned = True
                        self.logging_service.log_warning(
                            f"[Cascading Validation] No schedule-aware "
                            f"accelerator is wired, so the legacy linear "
                            f"inverse is in use and its timestep for R={accel}x "
                            f"({int(t_ideal)}) was clamped into "
                            f"[{min_t}, {max_t}] -> {t_used}. The "
                            f"`acceleration_level` column therefore names a "
                            f"REQUESTED level that may not be the measured one, "
                            f"and `acceleration_realized` is left empty because "
                            f"nothing here can decode it. Further occurrences "
                            f"are suppressed."
                        )

                # [DATA LEAKAGE FIX] Curriculum capping REMOVED from validation.
                # Validation must evaluate the full acceleration range for honest
                # metrics.  Capping forced all levels to R=2x early in training,
                # producing near-identity masks and inflating PSNR to ~70 dB.

                timestep = torch.full(
                    (batch_size,),
                    t_used,
                    dtype=torch.long,
                    device=input_batch.device,
                )

                self.logging_service.log_debug(
                    f"[Cascading Validation] R={accel}x → timestep={t_used} "
                    f"(realises R="
                    f"{'unknown' if acceleration_realized is None else f'{acceleration_realized:.4f}'}"
                    f", horizon [0, {num_timesteps - 1}])"
                )

                # The reverse trajectory starts where THIS rung's data actually
                # is. Without it every rung replayed the fully-degraded schedule:
                # at R=2 the input sits at t=1, so the 27 steps above it had a
                # ``mask_next`` already inside the observed support and could not
                # write a single coefficient (#535/#1388).
                hr_fakes = self._generate_validation_prediction(
                    input_batch,
                    target_batch,
                    timestep,
                    batch_data,
                    kwargs,
                    scale_factor,
                    start_timestep=t_used,
                )

                if hr_fakes is None:
                    # Held-out severities are an opt-in robustness readout; losing
                    # one degrades that readout and nothing else, so it stays a
                    # recorded skip.
                    if _is_heldout:
                        self.logging_service.log_warning(
                            f"[Cascading Validation] held-out R={accel}x: prediction "
                            f"returned None, skipping point."
                        )
                        continue
                    # An in-distribution rung is different: every consumer below
                    # (the cascade mean, the accel gap, the L4 gate) is defined
                    # over the WHOLE ladder, and dropping a rung silently makes a
                    # partially-failed validation outscore a complete one (#1303).
                    #
                    # Raising is what makes the existing machinery correct rather
                    # than adding new machinery: `_run_validation` catches
                    # per-batch, so this batch contributes NO keys instead of a
                    # partial dict — which also keeps it out of the per-key /
                    # global-count averaging asymmetry in #1323. If the failure is
                    # systematic rather than transient, every batch raises,
                    # `val_count` reaches 0 and the F36 guard turns it fatal.
                    #
                    # `_generate_validation_prediction` has already logged the
                    # underlying traceback as a warning before returning None, so
                    # the cause is on the record; this names the rung.
                    raise RuntimeError(
                        f"[Cascading Validation] in-distribution R={accel}x produced no "
                        f"prediction (_generate_validation_prediction returned None; its "
                        f"traceback was logged above). Refusing to report a cascade that "
                        f"is missing a severity level: val_*_mean would then be an "
                        f"average over whichever levels survived, under the same column "
                        f"name a complete cascade uses, so a degraded run would score "
                        f"HIGHER than a healthy one (#1303)."
                    )

                level_metrics = self._compute_validation_metrics(
                    hr_fakes,
                    target_batch,
                    input_batch,
                    timestep,
                    batch_data,
                    scale_factor,
                    batch_idx=batch_idx,
                    cascade_level=float(accel),
                    heldout=_is_heldout,
                    timestep_used=t_used,
                    acceleration_realized=acceleration_realized,
                )

                # L4 gate: capture the per-level prediction as an IMAGE
                # (iFFT RSS magnitude) before we free it, so the per-pixel
                # structural spread is measured in the domain the DC blob
                # appears (not k-space, which would under-detect it).
                # Only in-distribution levels feed the DC-blob input-dependence
                # gate (it is indexed against _CASCADING_LEVELS); held-out points
                # are an additional robustness readout, not part of that gate.
                if _l4_enabled and not _is_heldout:
                    cascade_predictions.append(
                        self._cascade_prediction_image(hr_fakes.detach(), _l4_needs_ifft)
                    )

                # Explicitly free prediction tensor to reclaim GPU memory before next level.
                del hr_fakes

                # Two representations of ONE evaluation, both from this pass.
                #
                # (a) Suffixed, flat. Load-bearing, not legacy: the L4
                #     input-dependence gate, `_stamp_accel_psnr_gap` and the
                #     acceleration-ladder check below all index `all_metrics`
                #     by `val_*_<R>x`. Dropping it to "clean up" would silently
                #     disable the DC-blob gate (pitfall #20) while everything
                #     still looked green.
                # (b) Tall. One row per severity point with the level and the
                #     timestep as VALUES -- what the CSV consumes (#697).
                suffix = f"_heldout_{accel}x" if _is_heldout else f"_{accel}x"
                for key, value in level_metrics.items():
                    all_metrics[f"{key}{suffix}"] = value

                # `t_used` is already a Python scalar, so this costs no GPU
                # sync (perf rule: no `.item()` in the loop).
                #
                # `acceleration_level` is what was ASKED for; the row now also
                # carries what the schedule decoded `t_used` back to (#1295).
                # The timestep alone made a divergence visible only to a reader
                # willing to re-run the inversion; the pair makes it a
                # subtraction. It is `None` on the linear-fallback path, where
                # nothing can decode it -- an honest blank rather than a
                # restatement of the request.
                cascade_rows.append(
                    build_cascade_row(
                        acceleration_level=accel,
                        heldout=_is_heldout,
                        timestep=t_used,
                        metrics=level_metrics,
                        acceleration_realized=acceleration_realized,
                    )
                )

                if not _is_heldout:
                    _levels_evaluated.append(accel)

            # Published for the pipeline to persist. The strategy computes and
            # the pipeline writes: `IMetricsService` declares no CSV surface, so
            # reaching through it from here would bypass its own contract.
            #
            # EXTEND, not assign: `_run_validation` calls this method once per
            # val batch and the cascade is not gated on `batch_idx`, so assigning
            # would publish the LAST batch while the suffixed columns beside it
            # hold a mean over every batch. The pipeline's drain aggregates and
            # then clears, which is what bounds this list to one validation.
            self._last_cascade_rows.extend(cascade_rows)

            if not all_metrics:
                return {"validation_error": 1.0}

            # ---- Cascade completeness (#1303) ----
            # Everything below is defined over the WHOLE in-distribution ladder.
            # Before this, a cascade that lost a rung was indistinguishable from
            # one that did not: the mean was taken over "whatever survived" under
            # the same column name, so a degraded run scored HIGHER than a healthy
            # one. Stamp the ladder's shape as DATA so the CSV, provenance and any
            # downstream reader can tell the two apart without inferring it.
            # Belt AND braces: a recorded skip OR a short evaluated list both mean
            # incomplete. The count is not redundant -- it is what keeps a future
            # `continue` that forgets to record itself from reading as complete.
            #
            # That covers the VERDICT. The REASON was still by convention -- each
            # `continue` records why before it leaves -- and a convention holds
            # only until the next `continue` is written. #1295 adds two of them
            # (a round-trip mismatch and an identity mask), neither of which
            # records, and the two branches merge without a textual conflict: the
            # count would correctly report the cascade incomplete while the
            # warning printed `skipped {}` beside a short evaluated list, stating
            # a contradiction and naming no cause on the one line an operator
            # reads to find out what happened.
            #
            # So derive the unaccounted-for rungs instead of trusting every exit
            # to speak up. Ordered BEFORE `_cascade_complete` so the verdict is
            # bit-identical either way (a derived entry only ever fires in the
            # case the length check already caught); this buys the diagnosis, not
            # a new failure mode. Held-out points are excluded by construction --
            # `_CASCADING_LEVELS` is the in-distribution ladder only.
            _levels_skipped = reconcile_skipped_levels(
                _CASCADING_LEVELS, _levels_evaluated, _levels_skipped
            )
            _cascade_complete = not _levels_skipped and len(_levels_evaluated) == len(
                _CASCADING_LEVELS
            )
            all_metrics["val_cascade_levels_expected"] = float(len(_CASCADING_LEVELS))
            all_metrics["val_cascade_levels_evaluated"] = float(len(_levels_evaluated))
            all_metrics["val_cascade_complete"] = 1.0 if _cascade_complete else 0.0
            if not _cascade_complete:
                self.logging_service.log_warning(
                    f"[Cascading Validation] INCOMPLETE cascade: evaluated "
                    f"{sorted(_levels_evaluated)} of {list(_CASCADING_LEVELS)}; "
                    f"skipped {_levels_skipped}. `val_*_mean` is NOT stamped for "
                    f"this validation — the partial average is published as "
                    f"`val_*_mean_partial` instead, so it cannot be compared "
                    f"against, or selected on, as if it were the full-ladder mean "
                    f"(#1303)."
                )

            # ---- L4 input-dependence sanity gate (2026-05-30) ----
            self._apply_input_dependence_gate(all_metrics, cascade_predictions, _CASCADING_LEVELS)

            # ---- Across-R PSNR gap (the DC-blob signal, 2026-06-02) ----
            self._stamp_accel_psnr_gap(all_metrics, _CASCADING_LEVELS)

            # ---- Cascade MEAN (the non-pathological selection target, 2026-06-08) ----
            # 2026-07-29: pass the CONFIGURED validation metrics. Previously the set was
            # hardcoded to psnr/robust_mri_psnr, so declaring e.g. `hfen` in
            # validation.metrics produced val_hfen_<R>x columns but no val_hfen_mean —
            # and an arm selecting on it selected on a key that never existed (#18).
            self._stamp_accel_mean(
                all_metrics,
                _CASCADING_LEVELS,
                [*self._configured_validation_metrics(), *self._ENSEMBLE_METRICS],
                complete=_cascade_complete,
            )

            # ---- Gated feature-insertion hallucination test (#3) ----
            all_metrics.update(
                self._maybe_hallucination_metrics(
                    target_batch,
                    input_batch,
                    timestep,
                    batch_data,
                    kwargs,
                    scale_factor,
                )
            )

            # ---- Terminal (t=0) rung, pre-DC (#1682) ----
            # The cascade above evaluates R=2/8/32 (t=1/14/28) and stops. t=0 is
            # never a rung, and post-DC it could not be one: every bin is
            # acquired there, so the output is the input. Read pre-DC instead.
            all_metrics.update(
                self._t0_pre_dc_probe_metrics(
                    target_batch, input_batch, batch_data, scale_factor, batch_idx
                )
            )

            return all_metrics

    @staticmethod
    def _level_is_declared(sched_accelerator: Any, level: float) -> bool:
        """Is ``level`` a rung the configured step schedule actually declares?

        ``timestep_for_acceleration`` raises ``ValueError`` for two unrelated
        reasons and the cascade must treat them differently (#1303):

        * the rung is **not** in ``acceleration_range`` — a declared limitation
          of the arm's ladder, so the level is skipped and recorded;
        * the rung **is** declared but is realised at no timestep (#1171) — a
          config defect, so the error propagates.

        The two are told apart by reading the accelerator's *state*, never by
        matching the exception's message text: a reworded message would silently
        flip every arm to the other branch.

        ``ColdDiffusionAccelerator`` is a wrapper — ``timestep_for_acceleration``
        delegates to ``self.accelerator`` while the schedule state lives on the
        wrapped instance — so the range is read through that hop. Without an
        explicit range there is no declared grid and nothing can be "declared".

        Args:
            sched_accelerator: The accelerator the cascade resolved.
            level: The requested acceleration factor.

        Returns:
            True when ``level`` appears in the explicit ``acceleration_range``.
        """
        inner = getattr(sched_accelerator, "accelerator", sched_accelerator)
        if not getattr(inner, "_acceleration_range_explicit", False):
            return False
        declared = getattr(inner, "acceleration_range", None)
        if not isinstance(declared, list):
            return False
        # Same 1e-6 tolerance the schedule itself uses, so the two sides cannot
        # disagree about membership.
        return any(abs(float(r) - float(level)) < 1e-6 for r in declared)

    @staticmethod
    def _stamp_accel_psnr_gap(all_metrics: dict[str, float], cascading_levels: list[int]) -> None:
        """Stamp the across-acceleration PSNR gap — the DC-blob signal.

        Soft-DC carries the LOW-acceleration PSNR by injecting measured k-space,
        so a healthy-looking ``val_psnr_2x`` can coexist with a blobby HIGH-R
        output: the gap between the least- and most-accelerated buckets is
        exactly what the model itself (not DC) must close. A *shrinking* gap as
        training proceeds is the signal that the pre-DC fidelity supervision is
        working. Stamping it makes the blob signal a first-class, monitorable
        column rather than something inferred by eye from per-R columns.

        For ``psnr`` and ``robust_mri_psnr`` it stamps
        ``val_<metric>_accel_gap = val_<metric>_<lo>x - val_<metric>_<hi>x``
        (lo = least accelerated, hi = most accelerated). No-op for a metric with
        fewer than the two endpoint levels present, or with < 2 levels.

        Args:
            all_metrics: The per-level metric dict (mutated in place).
            cascading_levels: Acceleration levels evaluated this validation.
        """
        if not cascading_levels or len(cascading_levels) < 2:
            # ...and say so. This branch was unreachable while the ladder was a
            # 3-element module constant; `validation.cascade.levels: [8]` makes
            # it live, and returning silently would leave BOTH the gap column
            # and the flag below absent -- reintroducing exactly the
            # indistinguishable-absence the flag was added (#1303) to remove.
            # A one-rung ladder has no gap by construction, which is a fact
            # worth recording, not a reason to record nothing.
            all_metrics["val_accel_gap_unavailable"] = 1.0
            return
        lo, hi = min(cascading_levels), max(cascading_levels)
        stamped = False
        for metric in ("psnr", "robust_mri_psnr"):
            k_lo, k_hi = f"val_{metric}_{lo}x", f"val_{metric}_{hi}x"
            if k_lo in all_metrics and k_hi in all_metrics:
                all_metrics[f"val_{metric}_accel_gap"] = float(all_metrics[k_lo]) - float(
                    all_metrics[k_hi]
                )
                stamped = True
        # An absent gap column used to be indistinguishable from a gap that was
        # never asked for. Record the no-op so "the DC-blob signal is missing"
        # is DATA rather than something a reader has to infer from an absence
        # (#1303, pitfall #16).
        all_metrics["val_accel_gap_unavailable"] = 0.0 if stamped else 1.0

    def _configured_validation_metrics(self) -> list[str]:
        """Base names declared in ``validation.metrics``, as the cascade-mean set.

        ``config.validation`` is ``ValidationConfigSchema | None`` on the frozen
        SSOT and ``metrics`` is a declared field, so both are read directly: the
        old ``getattr(self.config, "validation", None)`` default could never fire
        and only hid the contract (non-negotiable #1). ``metrics`` may be a list,
        a ``{name: enabled}`` mapping, or None, and entries may be enums.
        """
        validation = self.config.validation
        raw = validation.scoring.compute if validation is not None else None
        if isinstance(raw, dict):
            raw = [name for name, enabled in raw.items() if enabled]
        names: list[str] = []
        for entry in raw or ():
            name = getattr(entry, "value", entry)
            if isinstance(name, str) and name:
                names.append(name)
        return names

    @staticmethod
    def _stamp_accel_mean(
        all_metrics: dict[str, float],
        cascading_levels: list[int],
        metric_names: Iterable[str] | None = None,
        *,
        complete: bool = True,
    ) -> None:
        """Stamp the cascade-mean — the non-pathological checkpoint-selection target.

        Selecting on a single acceleration is unsafe in either direction:
        ``val_<metric>_2x`` (the easiest) lets a run "win" while collapsing at
        high R (a DC blob scores well at 2x), while ``val_<metric>_8x`` alone is
        *monotone-degrading* under the experiment_11 negative-transfer collapse,
        so ``restore_best_weights`` provably keeps the earliest (high-t-untrained)
        checkpoint — the exact 2026-06-06 self-defeating-fix failure.

        The mean over the in-distribution cascade (least→most accelerated) is the
        honest target: it rises only when the WHOLE cascade improves, cannot be
        gamed by collapsing high R, and is not self-defeating. Point
        ``metrics.best_metric_name`` / ``early_stopping.metric`` at
        ``val_robust_mri_psnr_mean``.

        Stamps ``val_<metric>_mean`` over the in-distribution ``_<R>x`` columns
        present, for every metric in ``metric_names``. No-op with < 1 level present.

        ``metric_names`` was a hardcoded ``("psnr", "robust_mri_psnr")``, so a metric
        added to ``validation.metrics`` got its per-level ``val_<m>_<R>x`` columns but
        never a ``_mean`` — and pointing ``best_metric_name`` at one selected on a key
        the run never computes (pitfall #18). ``psnr``/``robust_mri_psnr`` stay in the
        set unconditionally so every arm selecting on them is unaffected.

        Args:
            all_metrics: The per-level metric dict (mutated in place).
            cascading_levels: Acceleration levels evaluated this validation.
            metric_names: Metric base names to average. ``None`` keeps the two
                defaults, preserving the pre-2026-07-29 behaviour for callers that
                do not pass the configured validation metrics.
            complete: Whether every in-distribution level was evaluated. When
                False the average is stamped as ``val_<metric>_mean_partial``
                instead of ``val_<metric>_mean`` (#1303) — see below.

        Why an incomplete cascade may not use the ``_mean`` name:
            The mean was previously taken over *whichever levels survived*, under
            the column name a complete cascade uses. Because the ladder is
            monotone in difficulty, losing the hardest rung RAISES the number: a
            run that failed at 32x outscored one that evaluated all three, and
            ``restore_best_weights`` would prefer the broken one. Two
            incomparable quantities cannot share a column name, so the partial
            one is published under its own — an arm selecting on ``_mean`` then
            finds the key missing (loud) rather than a flattering wrong value
            (silent). Nothing in ``experiments/inprogress/`` is affected today:
            all 63 step-scheduled arms declare a ladder covering every level.
        """
        if not cascading_levels:
            return
        names = {"psnr", "robust_mri_psnr"}
        names.update(metric_names or ())
        suffix = "_mean" if complete else "_mean_partial"
        for metric in sorted(names):
            vals = [
                float(all_metrics[f"val_{metric}_{accel}x"])
                for accel in cascading_levels
                if f"val_{metric}_{accel}x" in all_metrics
            ]
            if vals:
                all_metrics[f"val_{metric}{suffix}"] = sum(vals) / len(vals)

    def _apply_input_dependence_gate(
        self,
        all_metrics: dict[str, float],
        cascade_predictions: list[torch.Tensor],
        cascading_levels: list[int],
    ) -> None:
        """L4 measurement-independence (DC-blob) gate.

        Catches the measurement-independent DC blob: an output that is a
        ~fixed (DC-dominated) constant regardless of the acceleration.
        Prefers the per-pixel structural spread across the cascade (the
        per-level predictions captured during the validation loop); falls
        back to the already-stamped ``val_pred_mean_<R>x`` scalars. No extra
        forward pass. ``validation.input_dependence_tol = None`` disables the
        gate entirely (no keys stamped).

        Stamps ``val_input_dependence`` (the spread) and
        ``val_measurement_collapse`` (1.0 when collapsed, else 0.0) into
        ``all_metrics`` so they flow to the CSV / early-stopping /
        provenance, and logs a warning on collapse so the strict smoke audit
        (warnings exit 2) trips.

        Args:
            all_metrics: The per-level metric dict (mutated in place).
            cascade_predictions: Per-level prediction tensors captured during
                the cascade loop (preferred, per-pixel path).
            cascading_levels: The acceleration levels evaluated (for the
                scalar ``val_pred_mean_<R>x`` fallback and logging).
        """
        tol = self.config.validation.gates.input_dependence_tol
        if tol is None:
            # Deliberately disabled by config — not a skip, and stamping a
            # "skipped" column here would make every arm that turned the gate off
            # look degraded.
            return

        # Local import: keep the core.metrics package walk-discovery off the
        # strategy import path to avoid any init-time cycle.
        from spectramr.core.metrics import get_metric

        metric = get_metric("input_dependence_spread")
        gate_kwargs: dict[str, Any] = {}
        if len(cascade_predictions) >= 2:
            gate_kwargs["cascade_predictions"] = cascade_predictions
        else:
            means = [
                all_metrics[f"val_pred_mean_{accel}x"]
                for accel in cascading_levels
                if f"val_pred_mean_{accel}x" in all_metrics
            ]
            if len(means) >= 2:
                gate_kwargs["cascade_pred_means"] = means

        if not gate_kwargs:
            # The gate is ENABLED but had fewer than two comparable levels to
            # measure across — which is exactly what a cascade that lost a rung
            # produces. Returning silently made an un-run DC-blob gate look
            # identical to one that ran and passed (#1303, pitfall #16).
            all_metrics["val_input_dependence_skipped"] = 1.0
            self.logging_service.log_warning(
                "[L4 input-dependence gate] enabled but not evaluated: fewer than "
                "two cascade levels were available to compare. The DC-blob check "
                "did NOT run for this validation."
            )
            return
        all_metrics["val_input_dependence_skipped"] = 0.0

        spread = float(metric(None, None, **gate_kwargs))
        all_metrics["val_input_dependence"] = spread
        collapsed = spread < tol
        all_metrics["val_measurement_collapse"] = 1.0 if collapsed else 0.0
        if collapsed:
            self.logging_service.log_warning(
                "[L4 input-dependence gate] output is measurement-INDEPENDENT: "
                f"spread {spread:.2e} < tol {tol:.2e} across "
                f"R={cascading_levels}. The model is emitting a ~fixed "
                "(DC-dominated) output regardless of acceleration — the DC-blob "
                "collapse."
            )

    def _cascade_prediction_image(self, prediction: torch.Tensor, needs_ifft: bool) -> torch.Tensor:
        """Convert a cascade prediction to a single-channel RSS magnitude IMAGE
        for the L4 input-dependence gate.

        The DC blob is an IMAGE-domain phenomenon: the network can vary its
        high-frequency k-space (so a *k-space* spread looks healthy) while the
        image-dominating low frequencies stay nearly constant across the
        acceleration cascade. Measuring the gate on the iFFT'd RSS magnitude
        closes that blind spot. ``needs_ifft`` mirrors the metrics/
        visualization domain decision (``needs_ifft_for_visualization``) so the
        gate sees the same image the PSNR + saved PNG use.

        Args:
            prediction: The raw model prediction for one cascade level —
                k-space ``[B, 2*coils, H, W]`` (real-stacked) or complex, or an
                already-image tensor when ``needs_ifft`` is False.
            needs_ifft: True when ``prediction`` is k-space (apply a centered
                iFFT via the physics SSOT); False when it is already an image.

        Returns:
            ``[B, 1, H, W]`` real RSS magnitude image.
        """
        from spectramr.infrastructure.physics.fft_ops import ifft2c

        x = prediction
        if needs_ifft:
            if not torch.is_complex(x) and x.ndim == 4 and x.shape[1] % 2 == 0:
                b, c, h, w = x.shape
                x = torch.view_as_complex(
                    x.permute(0, 2, 3, 1).contiguous().view(b, h, w, c // 2, 2)
                ).permute(0, 3, 1, 2)
            elif not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            x = ifft2c(x)
        mag = x.abs()
        return torch.sqrt((mag**2).sum(dim=1, keepdim=True) + 1e-12)

    def get_validation_images(
        self,
        batch: Any,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Generate validation images for visualization.

        Returns:
            Tuple of (prediction, target, input) or None if generation fails.
        """
        batch_data = kwargs.get("batch_data")
        if batch_data is None:
            if isinstance(batch, dict):
                batch_data = batch
            elif hasattr(batch, "mask"):
                batch_data = {"mask": batch.mask}

        # Prepare data (reuse validation logic)
        prep_result = self._prepare_validation_data(batch, input_batch, target_batch, batch_data)
        if prep_result is None:
            return None
        input_batch, target_batch, scale_factor = prep_result

        # Handle 5D input (flatten to 4D for processing)
        if input_batch.ndim >= 5:
            shape = list(input_batch.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            input_batch = input_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            if target_batch.ndim >= 5:
                target_batch = target_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            if scale_factor.ndim == 4 and scale_factor.shape[0] == b:
                scale_factor = scale_factor.repeat_interleave(d, dim=0)

        # Use representative acceleration (e.g., 4x) for visualization
        # Invert R(t) = base + (max - base) * (t/T) -> t = T * (R - base) / (max - base)
        num_timesteps = self.num_timesteps
        base_accel = self.config.undersampling.base_acceleration
        max_accel = self.config.undersampling.max_acceleration
        span = max(1.0, max_accel - base_accel)
        target_accel = 4.0

        t_val = int(num_timesteps * (target_accel - base_accel) / span)
        t_val = max(1, min(num_timesteps - 1, t_val))

        # [DATA LEAKAGE FIX] Curriculum capping REMOVED from validation images.
        # Validation must evaluate at the requested acceleration level regardless
        # of the current training curriculum stage.

        timestep = torch.full(
            (input_batch.size(0),),
            t_val,
            dtype=torch.long,
            device=input_batch.device,
        )

        with torch.no_grad():
            if (
                hasattr(self.generator_model, "simulator")
                and (
                    self.config.model.model_type if hasattr(self.config.model, "model_type") else ""
                )
                == "image_cold_diffusion"
            ):
                simulator = self.generator_model.simulator
                if (
                    hasattr(simulator, "progressive_degradations")
                    and simulator.progressive_degradations
                ):
                    orig_degrads = simulator.progressive_degradations.copy()
                    all_fakes = []

                    # 1. Generate with all combined
                    hr_fakes_all = self._generate_validation_prediction(
                        input_batch,
                        target_batch,
                        timestep,
                        batch_data,
                        kwargs,
                        scale_factor,
                    )
                    all_fakes.append(hr_fakes_all)

                    # 2. Iterate each enabled degradation individually
                    for d in orig_degrads:
                        simulator.progressive_degradations = [d]
                        hr_fakes_d = self._generate_validation_prediction(
                            input_batch,
                            target_batch,
                            timestep,
                            batch_data,
                            kwargs,
                            scale_factor,
                        )
                        if hr_fakes_d is not None:
                            all_fakes.append(hr_fakes_d)

                    # Restore original list
                    simulator.progressive_degradations = orig_degrads

                    if all(f is not None for f in all_fakes):
                        hr_fakes = torch.cat(all_fakes, dim=-1)
                        if target_batch is not None:
                            target_batch = torch.cat([target_batch] * len(all_fakes), dim=-1)
                        if input_batch is not None:
                            input_batch = torch.cat([input_batch] * len(all_fakes), dim=-1)
                    else:
                        hr_fakes = hr_fakes_all
                else:
                    hr_fakes = self._generate_validation_prediction(
                        input_batch,
                        target_batch,
                        timestep,
                        batch_data,
                        kwargs,
                        scale_factor,
                    )
            else:
                hr_fakes = self._generate_validation_prediction(
                    input_batch,
                    target_batch,
                    timestep,
                    batch_data,
                    kwargs,
                    scale_factor,
                )

        # ``_generate_validation_prediction`` stashes the cold-branch measurement
        # for the zero-filled baseline, but the *consumer* is
        # ``_compute_validation_metrics`` and this method never calls it. Discard
        # it here rather than leave a masked k-space batch referenced on the
        # strategy instance until the next validation's first generate clears it
        # (val-time OOM is a documented failure mode on 16 GB V100s).
        self._zf_measurement = None

        if hr_fakes is None:
            return None

        # Denormalize for visualization
        if self._is_cold_diffusion() and self.config.data.processing.enable_kspace_normalization:
            # Recompute scale if needed (or reuse if robust)
            # Simplest to just use the passed scale_factor
            if scale_factor is not None:
                if scale_factor.ndim < hr_fakes.ndim:
                    while scale_factor.ndim < hr_fakes.ndim:
                        scale_factor = scale_factor.unsqueeze(-1)
                hr_fakes = hr_fakes * scale_factor
                target_batch = target_batch * scale_factor
                input_batch = input_batch * scale_factor

        return hr_fakes, target_batch, input_batch

    def _extract_mask_from_batch(
        self,
        batch_data: Any,
        input_batch: torch.Tensor,
    ) -> torch.Tensor | None:
        """Extract and prepare mask from batch data.

        Handles:
        - Multiple mask locations in batch_data
        - Mask generation with validation seed
        - Shape validation and permutation
        - Device placement

        Returns: mask tensor or None if not found/needed
        """
        mask = None

        # Try multiple mask locations
        if batch_data is not None and isinstance(batch_data, dict):
            mask = batch_data.get("mask")
            if mask is None:
                mask = batch_data.get("sampling_mask")
            if mask is None and "physics" in batch_data:
                physics_dict = batch_data["physics"]
                if isinstance(physics_dict, dict):
                    mask = physics_dict.get("mask")
                    if mask is None:
                        mask = physics_dict.get("sampling_mask")

        # Generate mask if not found
        if mask is None:
            if hasattr(self, "mask_generator") and self.mask_generator is not None:
                self.logging_service.log_warning(
                    "[VALIDATION] No pre-computed mask found in batch. "
                    "Generating DETERMINISTIC mask (config.run.seed) to allow validation."
                )
                # `run.seed`, not `training.seed`: phase 4b moved the key and this
                # read was never repointed. `training.seed` is a RAISE-posture
                # rename, so there is no execution path on which the old spelling
                # resolves -- it raised AttributeError on this branch, and only on
                # the no-precomputed-mask fallback, which a happy-path smoke run
                # never reaches.
                validation_seed = self.config.run.seed
                mask = self.mask_generator.generate_mask(input_batch.shape, seed=validation_seed)
                mask = mask.to(input_batch.device)
            else:
                raise ValueError(
                    "[DATA LEAK PREVENTION] No pre-computed mask found in batch_data. "
                    "Random mask generation has been disabled to prevent train/inference "
                    "distribution mismatch."
                )
        else:
            mask = mask.to(input_batch.device).float()

            # [VALIDATION] Check mask coverage matches expected acceleration
            if self.logging_service.logger.isEnabledFor(logging.DEBUG):
                actual_coverage = mask.mean().item()
                # Was `hasattr(self.config.data, "acceleration")` — permanently
                # False, since the acceleration factor lives on its own block,
                # not on `data`. The check is worth keeping, so it now reads the
                # real field instead of never running.
                accel_cfg = self.config.undersampling
                if accel_cfg is not None and accel_cfg.base_acceleration:
                    expected_accel = accel_cfg.base_acceleration
                    expected_coverage = 1.0 / expected_accel

                    # Allow 10% tolerance for variable density masks
                    if abs(actual_coverage - expected_coverage) > 0.1:
                        self.logging_service.log_warning(
                            f"[VALIDATION] Mask coverage mismatch: "
                            f"expected ~{expected_coverage:.2%} (accel={expected_accel}x), "
                            f"got {actual_coverage:.2%}. "
                            f"Check if dataloader mask matches acceleration.base_acceleration. "
                            f"Variable density masks may have different coverage."
                        )

                self.logging_service.log_debug(
                    f"[VALIDATION] Using pre-computed mask from dataloader: "
                    f"coverage={actual_coverage:.2%}"
                )

        # Determine target channels for expansion (prefer target if it exists, otherwise input)
        # We look at batch_data["target"] if available, else fallback to input
        target_ch = input_batch.shape[1]
        if batch_data is not None and isinstance(batch_data, dict) and "target" in batch_data:
            tgt = batch_data["target"]
            if hasattr(tgt, "shape") and len(tgt.shape) > 1:
                target_ch = tgt.shape[1]

        # Expand mask to match the target channels instead of input channels
        if mask.shape[1] != target_ch:
            mask = self.mask_generator.expand_mask_to_channels(mask, target_ch)

        # Handle 4D/5D shape mismatch: flatten if needed
        if input_batch.ndim == 4 and mask.ndim >= 5:
            shape = list(mask.shape)
            b, c = shape[0], shape[1]
            d = shape[-1]
            h, w = shape[-3], shape[-2]
            if b * d == input_batch.shape[0]:
                self.logging_service.log_warning(
                    "Flattening mask (B,C,H,W,D) -> (B*D,C,H,W) to match input"
                )
                mask = mask.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

        # Log shapes for debugging (per-call masking path — keep at DEBUG so it
        # doesn't spam the hot path; backlog_wasted_compute_audit_2026_05_29 PHYS-3)
        self.logging_service.log_debug(f"Input: {input_batch.shape}, Mask: {mask.shape}")

        # Handle 5D/5D shape mismatch: permute if needed
        if input_batch.ndim >= 5 and mask.ndim >= 5:
            if mask.shape[-1] == input_batch.shape[2]:
                self.logging_service.log_warning(
                    "Permuting mask (B,C,H,W,D) -> (B,C,D,H,W) based on depth match"
                )
                mask = mask.permute(0, 1, 4, 2, 3)

        return mask

    def _apply_masking_and_verify(
        self,
        input_batch: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply masking to input and verify coverage.

        Handles:
        - Element-wise masking
        - Data leakage detection
        - Coverage verification
        - Logging of masking effect

        Returns: masked_input tensor
        """
        # [ROBUST MULTI-COIL MAPPING FIX]
        # Force target block channels to match mask dynamically using repeat to handle
        # both perfectly divisible and irregular multi-coil scaling ratios.
        # This prevents 'a (4) must match size of tensor b (2)' broadcasting errors
        # when a spatial 1-channel or 2-channel mask is mapped against multi-coil (e.g. 4/8 channel) targets.
        if input_batch.shape[1] > mask.shape[1]:
            repeats = [1] * mask.dim()
            repeats[1] = input_batch.shape[1] // mask.shape[1]
            mask = mask.repeat(*repeats)
            # Catch remainder if not perfectly divisible
            if mask.shape[1] < input_batch.shape[1]:
                rem = input_batch.shape[1] - mask.shape[1]
                mask = torch.cat([mask, mask[:, :rem]], dim=1)
        elif input_batch.shape[1] < mask.shape[1]:
            repeats = [1] * input_batch.dim()
            repeats[1] = mask.shape[1] // input_batch.shape[1]
            input_batch = input_batch.repeat(*repeats)
            if input_batch.shape[1] < mask.shape[1]:
                rem = mask.shape[1] - input_batch.shape[1]
                input_batch = torch.cat([input_batch, input_batch[:, :rem]], dim=1)

        masked_input = input_batch * mask

        # Verification: Check if masking had significant effect
        if self.logging_service.logger.isEnabledFor(logging.INFO):
            diff = (masked_input - input_batch).abs().sum().item()
            if diff < 1e-6:
                if mask.mean().item() > 0.999:
                    self.logging_service.log_info(
                        "[MASKING-EFFECT] Mask is fully sampled (100% coverage). No leakage warning needed."
                    )
                else:
                    self.logging_service.log_warning(
                        "[LEAKAGE CHECK] masked_input == input_batch! "
                        f"Mask coverage: {mask.mean().item():.2%}. "
                        "Input may already be masked!"
                    )
            else:
                # Compute actual masking effect
                input_mean = input_batch.abs().mean().item()
                masked_mean = masked_input.abs().mean().item()
                reduction = (input_mean - masked_mean) / input_mean if input_mean > 0 else 0
                self.logging_service.log_info(
                    f"[MASKING-EFFECT] Input mean: {input_mean:.4f} → Masked mean: {masked_mean:.4f} | "
                    f"Reduction: {reduction:.1%} (expected: {1 - mask.mean().item():.1%})"
                )

        return masked_input

    def _cascade_levels_declared(self) -> bool:
        """Did the YAML declare ``validation.cascade.levels``, or is this the default?

        Presence, not equality. An arm that explicitly writes out the same
        ladder the framework defaults to has made a decision, and a later
        change to the framework default must
        not silently move it; an arm that omits the key has not. Comparing the
        resolved ladder against ``CASCADING_LEVELS`` would collapse those two
        into one reading, which is precisely the "absent is a state to report,
        never a state to infer" failure of non-negotiable 18.
        """
        val = self.config.validation
        cascade = getattr(val, "cascade", None) if val is not None else None
        return getattr(cascade, "levels", None) is not None

    def _held_out_severity_points(self) -> list[float]:
        """Held-out severity grid for the H4 robustness eval (WS1-core-03 /
        plan §12 / A5.8).

        Returns ``[]`` unless ``validation.held_out_severity_eval`` is set AND
        ``model.model_kwargs.digital_twin_kwargs.held_out_severity_grid`` is
        present, so arms that don't opt in are unaffected. Values are the extra
        (out-of-training-range) acceleration severities to additionally evaluate.
        """
        val = self.config.validation
        if not (val.gates.held_out_severity_eval if val else False):
            return []
        mk = self.config.model.model_kwargs or {}
        dtk = (
            mk.get("digital_twin_kwargs")
            if isinstance(mk, dict)
            else getattr(mk, "digital_twin_kwargs", None)
        ) or {}
        grid = (
            dtk.get("held_out_severity_grid")
            if isinstance(dtk, dict)
            else getattr(dtk, "held_out_severity_grid", None)
        )
        if not grid:
            return []
        try:
            return [float(s) for s in grid]
        except (TypeError, ValueError):
            return []

    def _resolve_validation_sampling_steps(self) -> int | None:
        """Resolve the reverse-diffusion step count to use at validation.

        WS1-core-04. Precedence (per ``ValidationConfigSchema.sampler_steps``
        docs): ``validation.sampler_steps`` → ``training.diffusion.
        sampling_steps`` → ``None``. ``None`` means "leave it to the sampler's
        own default", so an arm that configures neither knob is unaffected — and
        the value is only ever forwarded to a sampler whose signature accepts a
        step-count parameter (see ``_generate_validation_prediction``).
        """
        val = self.config.validation
        steps = val.sampling.steps if val is not None else None
        if steps is None:
            diff = self.config.training.diffusion
            steps = getattr(diff, "sampling_steps", None) if diff is not None else None
        if steps is not None:
            # A reverse pass cannot have more steps than the forward schedule has;
            # clamp so a mis-set value (e.g. experiment_12 sets sampler_steps=50
            # with timesteps=16) can never drive a strided sampler out of range.
            nt = getattr(self, "num_timesteps", None)
            if isinstance(nt, int) and nt > 0:
                steps = min(steps, nt)
        return steps

    def _sample_multistep_chunked(
        self,
        gen: Any,
        measurement: torch.Tensor,
        mask: torch.Tensor | None,
        steps: int | None,
        smaps: torch.Tensor | None = None,
        start_timestep: int | None = None,
        seed_offset: int = 0,
    ) -> torch.Tensor:
        """Run the multi-step cold reverse loop in ``val_chunk_size`` micro-batches.

        The single-step validation path caps peak GPU memory with
        ``val_chunk_size`` micro-chunking (``_forward_chunked``); the multi-step
        ``gen.sample()`` reverse-sampling path used to bypass it and ran the whole
        validation batch at once. That path holds a full fp32 forward's
        activations of the heavy nested complex / dual-domain UNet, so on
        experiment_11 (44 GiB GPU, 0.9 cap → 39.95 GiB allowed) a batch of 2
        peaked ~40 GiB and OOM'd EVERY validation — 40/40 failed,
        ``validation_metrics.csv`` was an all-``1.0`` sentinel, and the
        early-stopping monitor never appeared (2026-06-14).

        The reverse sampler reconstructs each batch element independently
        (``PhysicsInformedColdDiffusion.sample`` builds a per-element timestep
        vector and performs no cross-batch reduction), so splitting ``measurement``
        / ``mask`` along dim 0 and concatenating the per-chunk reconstructions is
        numerically identical to one full-batch call. Peak memory stays at one
        chunk's footprint because only the finished reconstruction is retained
        per chunk; the activations are freed at the end of each iteration and
        the caching allocator hands the same blocks to the next chunk.

        This loop used to call ``torch.cuda.empty_cache()`` between chunks, on
        the stated rationale that it was what kept peak at one chunk. It is not:
        within a single process the allocator already reuses freed blocks, so
        ``empty_cache`` only returns them to the driver and leaves peak
        *allocated* unchanged, while forcing a device synchronise. A Scalene
        profile of experiment_11_attention_none charged it ~18 s (1.99 % of the
        run), and an in-loop ``empty_cache`` is a non-negotiable 9 violation.
        """
        chunk = max(1, int(self.config.validation.loader.chunk_size))
        n = measurement.shape[0]
        # Forwarded only when non-zero: member 0 (and every single-sample call)
        # is the legacy call, byte for byte, so a generator whose ``sample()``
        # takes no ``seed_offset`` keeps working until an ensemble asks for one.
        member: dict[str, int] = {"seed_offset": int(seed_offset)} if seed_offset else {}
        if n <= chunk:
            return gen.sample(
                measurement=measurement,
                mask=mask,
                inference_timesteps=steps,
                smaps=smaps,
                start_timestep=start_timestep,
                **member,
            )

        meas_chunks = measurement.split(chunk, dim=0)
        if mask is not None and mask.shape[0] == n:
            mask_chunks: list[torch.Tensor | None] = list(mask.split(chunk, dim=0))
        else:
            # mask is None (sampler infers it) or not batch-aligned — reuse as-is.
            mask_chunks = [mask] * len(meas_chunks)
        # The maps are per-element, so they must be split in step with the
        # measurement — a whole-batch tensor handed to a chunk would broadcast
        # one subject's coil profile onto another's k-space.
        if smaps is not None and smaps.shape[0] == n:
            smap_chunks: list[torch.Tensor | None] = list(smaps.split(chunk, dim=0))
        else:
            smap_chunks = [smaps] * len(meas_chunks)

        parts: list[torch.Tensor] = []
        for meas_c, mask_c, smaps_c in zip(meas_chunks, mask_chunks, smap_chunks, strict=False):
            parts.append(
                gen.sample(
                    measurement=meas_c,
                    mask=mask_c,
                    inference_timesteps=steps,
                    smaps=smaps_c,
                    start_timestep=start_timestep,
                    **member,
                )
            )
        return torch.cat(parts, dim=0)

    def _resolve_validation_ensemble(self, gen: Any) -> int:
        """Read ``validation.sampling.ensemble_samples`` against the RESOLVED sampler.

        The schema refuses ``ensemble_samples > 1`` unless the multi-step sampler
        is on and ``model.model_kwargs.sampler_sigma`` is declared above 0; this
        is the other half of that check, on the generator that will actually
        run. A generator built without the knob (a direct construction, a plugin)
        exposes no ``sampler_sigma``, and one at sigma 0 would return N identical
        members with a std of exactly 0 and a coverage of exactly the hit rate: a
        facade wearing a number (non-negotiable 16). Both raise.

        Logged and stamped ONCE per strategy: the cascade calls this per rung
        and per batch, and the knob does not change between them.
        """
        val = self.config.validation
        n = int(val.sampling.ensemble_samples) if val is not None else 1
        if n <= 1:
            if self.ensemble_provenance is None:
                self.ensemble_provenance = {"ensemble_samples": 1}
                self.logging_service.log_info(
                    "[Ensemble] validation.sampling.ensemble_samples=1: the multi-sample "
                    "ensemble is off; val_ensemble_std_mean and val_empirical_coverage "
                    "are not emitted."
                )
            return 1
        sigma = getattr(gen, "sampler_sigma", None)
        if sigma is None:
            raise ConfigurationError(
                f"validation.sampling.ensemble_samples={n}, but the resolved generator "
                f"{type(gen).__name__} exposes no sampler_sigma, so the reverse sampler "
                "cannot be shown to be stochastic. An ensemble whose members may be "
                "identical is refused rather than reported."
            )
        if not float(sigma) > 0:
            raise ConfigurationError(
                f"validation.sampling.ensemble_samples={n} needs a stochastic reverse "
                f"sampler, but the resolved generator runs at sampler_sigma="
                f"{float(sigma)!r}: every member would be identical (std 0). Set "
                "model.model_kwargs.sampler_sigma > 0, or leave ensemble_samples at 1."
            )
        if not val.scoring.enable_image_metrics:
            raise ConfigurationError(
                f"validation.sampling.ensemble_samples={n} is read only by the image "
                "metrics path, and validation.scoring.enable_image_metrics is false: "
                "the N-fold sampling would run and nothing would read its spread."
            )
        if self.ensemble_provenance is None:
            k = float(val.sampling.coverage_k)
            seed = getattr(gen, "sampler_seed", None)
            self.ensemble_provenance = {
                "ensemble_samples": n,
                "coverage_k": k,
                "sampler_sigma": float(sigma),
                "sampler_seed": seed,
                "seed_offsets": list(range(n)),
            }
            self.logging_service.log_info(
                f"[Ensemble] validation draws {n} reverse samples per input "
                f"(sampler_sigma={float(sigma):.4g}, sampler_seed={seed!r}, seed "
                f"offsets 0..{n - 1}); the metrics grade the pixelwise mean, and "
                f"val_ensemble_std_mean / val_empirical_coverage (k={k:g}) are emitted "
                f"per rung. Validation costs {n}x the single-sample multistep pass."
            )
        return n

    def _sample_ensemble_chunked(
        self,
        gen: Any,
        measurement: torch.Tensor,
        mask: torch.Tensor | None,
        steps: int | None,
        n_members: int,
        *,
        smaps: torch.Tensor | None = None,
        start_timestep: int | None = None,
    ) -> torch.Tensor:
        """Draw ``n_members`` reverse samples of one input, stacked ``[N, B, C, H, W]``.

        Member ``i`` is the chunked multistep pass with ``seed_offset=i``, so
        member 0 is exactly the single-sample reconstruction and the others
        differ only in the C6 noise stream. The outer loop is over members and
        the inner one is ``_sample_multistep_chunked``'s ``val_chunk_size`` loop,
        so peak memory stays at one chunk's activations plus the N finished
        k-space outputs (N <= 4 by schema). No host sync happens here: the
        reduction to scalars is left to ``_compute_validation_metrics``, once
        per rung.
        """
        members = [
            self._sample_multistep_chunked(
                gen,
                measurement,
                mask,
                steps,
                smaps=smaps,
                start_timestep=start_timestep,
                seed_offset=i,
            )
            for i in range(int(n_members))
        ]
        return torch.stack(members, dim=0)

    def _ensemble_std_in_metric_domain(
        self,
        members: torch.Tensor,
        denom_scale: torch.Tensor,
        log_scaled: bool,
        target_for_metrics: torch.Tensor,
        val_config: Any,
        is_cold_diffusion: bool,
    ) -> torch.Tensor:
        """Pixelwise sample std of the members, in the domain the metrics grade in.

        Each member crosses the SAME seams the prediction crosses on its way to
        ``pred_transformed`` -- decompress, denormalise, channel-match, the arm's
        metric transform -- so the spread is measured where the residual is
        (magnitude images on an ``ifft_magnitude`` arm), not in k-space, where a
        per-pixel interval means nothing. Bessel-corrected sample std over the N
        members (N - 1 in the denominator).
        """
        from spectramr.infrastructure.training.utils.kspace_view import (
            decompress_for_view,
        )

        transformed = []
        for member in members.unbind(0):
            m = member.to(target_for_metrics.device)
            if is_cold_diffusion:
                m = decompress_for_view(m, log_scaled=log_scaled, channel_dim=1)
            m = m * denom_scale
            m, _ = TorchIOAdapter.ensure_channel_match(m, target_for_metrics)
            m_t, _ = self._apply_metric_transforms(m, target_for_metrics, val_config)
            transformed.append(m_t)
        return torch.stack(transformed, dim=0).std(dim=0)

    def _ensemble_metrics(
        self,
        pred_transformed: torch.Tensor,
        target_transformed: torch.Tensor,
        ensemble_std: torch.Tensor,
    ) -> dict[str, float]:
        """``ensemble_std_mean`` and ``empirical_coverage`` for one rung.

        The coverage is the registered ``empirical_coverage`` metric -- the one
        owner of "fraction of residuals inside a radius", shared with the qCRC
        coverage in ``conformal_risk.py`` -- resolved through the registry and
        handed the std on a ``MetricContext``. Deliberately NOT the configured
        metrics computer: its ``only=`` is an intersection with the arm's
        ``metrics.compute``, so an arm that did not list the metric would get
        nothing while paying for N samples. Two host syncs per rung, the cost of
        one registered metric plus one scalar.
        """
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        k = float(self.config.validation.sampling.coverage_k)
        metric = MetricsRegistry.get("empirical_coverage", k=k)
        coverage = float(
            metric(
                pred_transformed,
                target_transformed,
                context=MetricContext(ensemble_std=ensemble_std),
            )
        )
        return {
            "ensemble_std_mean": float(ensemble_std.mean()),
            "empirical_coverage": coverage,
        }

    def _maybe_hallucination_metrics(
        self,
        target_batch: torch.Tensor,
        input_batch: torch.Tensor,
        timestep: torch.Tensor,
        batch_data: Any,
        kwargs: dict,
        scale_factor: torch.Tensor,
    ) -> dict:
        """Gated feature-insertion hallucination test (#3, WS1-core-03).

        Disabled by default (``validation.hallucination_test.enabled``) → zero
        impact on arms that don't opt in. When enabled, runs every
        ``interval_validations`` validation steps and NEVER raises into the
        validation loop (a diagnostic must not fail training).
        """
        cfg = (
            self.config.validation.gates.hallucination_test
            if self.config.validation is not None
            else None
        )
        if cfg is None or not getattr(cfg, "enabled", False):
            return {}
        interval = max(1, int(getattr(cfg, "interval_validations", 4)))
        step = int(getattr(self, "validation_step_count", 0))
        if step % interval != 0:
            return {}
        try:
            from spectramr.core.metrics.hallucination_test import run_hallucination_test

            def _predict(perturbed: torch.Tensor) -> torch.Tensor:
                pred = self._generate_validation_prediction(
                    perturbed, perturbed, timestep, batch_data, kwargs, scale_factor
                )
                return pred if pred is not None else perturbed

            result = run_hallucination_test(
                _predict,
                target_batch,
                n_features=int(getattr(cfg, "n_features", 5)),
                method=str(getattr(cfg, "method", "feature_insertion")),
            )
            return {
                "val_hallucination_preservation_index": result["hallucination_preservation_index"]
            }
        except Exception as e:
            self.logging_service.log_warning(f"[HallucinationTest] skipped: {e}")
            return {}
        finally:
            # ``_predict`` drives ``_generate_validation_prediction`` once per
            # perturbed feature, so the stash outlives this diagnostic with no
            # consumer. Discard on every exit path.
            self._zf_measurement = None

    def _generate_validation_prediction(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        timestep: torch.Tensor,
        batch_data: Any,
        kwargs: dict,
        scale_factor: torch.Tensor,
        *,
        start_timestep: int | None = None,
    ) -> torch.Tensor | None:
        """Generate model predictions for validation (refactored orchestrator).

        Refactored to delegate to helper methods:
        - _extract_mask_from_batch: Mask extraction and validation
        - _apply_masking_and_verify: Apply masking and detect leakage

        Unified implementation with clean control flow.

        ``start_timestep`` is the timestep the input is actually degraded at, and
        it is the HEAD of the cold reverse trajectory. Keyword-only and defaulting
        to ``None`` so the six non-cascading call sites keep the legacy
        fully-degraded head; cascading validation passes the rung's own ``t_used``.
        Taken as a Python ``int`` rather than read off the ``timestep`` tensor
        precisely so it costs no GPU sync inside the validation loop
        (non-negotiable 9).
        """
        is_cold_diffusion = self._is_cold_diffusion()

        # The zero-filled baseline (#1684 follow-up) is measured from the
        # k-space the model ACTUALLY received, which only exists inside this
        # method: the cold branch below builds its own mask from
        # ``generate_and_process_mask`` (seeded, timestep-dependent), so
        # ``_compute_validation_metrics`` -- which is handed the UNMASKED
        # ``input_batch`` -- cannot reconstruct it without becoming a second,
        # silently-diverging owner of the mask.
        #
        # Cleared UNCONDITIONALLY here and popped (read-and-clear) by the
        # consumer, so the failure mode is a MISSING baseline, never rung N-1's
        # measurement reported under rung N's acceleration. The non-cold branch
        # never sets it, so a non-cold arm emits no ``val_zf_*`` keys at all --
        # on every batch and every rank, which is what the DDP all-reduce's
        # sorted-key packing requires (#1690).
        self._zf_measurement = None
        # Same discipline for the ensemble stack (see the class attribute).
        self._ensemble_members = None

        try:
            gen_kwargs = {}

            # Contrast conditioning: pass contrast_idx to the generator during
            # validation too. Extraction goes through the ONE owner,
            # ``_contrast_idx_from_batch`` (non-negotiable 17) -- this site used
            # to re-derive it behind ``isinstance(batch_data, dict)``, the second
            # resolver, with the same tensor/list ladder written out again.
            #
            # That guard could never match. ``select_validation_extra_fields``
            # (pipelines/train.py) forwards ``val_batch`` as ``batch_data``, and
            # ``val_batch`` is a :class:`~spectramr.data.batch_types.TrainingBatch`
            # -- a dataclass, not a mapping, whose non-core fields live in
            # ``.metadata`` and are reachable only through the mapping protocol.
            # So a multi-contrast arm trained CONDITIONED (the train path was
            # fixed in #1931 half 1) and validated UNCONDITIONED, with no error
            # and no log line: the train/val mismatch of pitfall #18, scoring
            # every epoch's checkpoint on a model the metrics never conditioned.
            #
            # The 5D alignment below is kept -- it is this site's own concern,
            # not the owner's, because only validation flattens B*D.
            c_idx = self._contrast_idx_from_batch(batch_data, target_batch.device)
            if c_idx is not None:
                # Expand shape if batch_size was flattened from 5D (B*D)
                b = len(batch_data.get("input", target_batch))  # original batch size
                b_flat = target_batch.shape[0]  # new batch size
                if b > 0 and b_flat > b and b_flat % b == 0:
                    d = b_flat // b
                    c_idx = c_idx.repeat_interleave(d, dim=0)

                gen_kwargs["contrast_idx"] = c_idx

            # SR3 conditioning at validation (#16): mirror the training path so
            # gen.sample() receives the ULF condition. The sample_kwargs filter
            # below keeps only ("contrast_idx","condition_image"), so without
            # this the conditional LDM would sample UNCONDITIONALLY at eval.
            _gen = (
                self.generator_model.module
                if hasattr(self.generator_model, "module")
                else self.generator_model
            )
            _gen_cfg = getattr(_gen, "config", None)
            if getattr(_gen_cfg, "conditional_translation", False):
                gen_kwargs["condition_image"] = input_batch

            if is_cold_diffusion:
                # [DATA LEAKAGE FIX] Use input_batch (dataloader-provided undersampled
                # k-space) instead of re-masking target_batch (fully-sampled ground truth).
                # Re-masking target leaked GT info: at low acceleration, 50%+ of k-space
                # is retained → model trivially reconstructs → 70 dB PSNR.
                # Now we generate a mask for the given acceleration level and apply it
                # to input_batch, which is already degraded by the dataloader.
                mask = self.generate_and_process_mask(
                    batch_size=input_batch.shape[0],
                    timesteps=timestep,
                    target_shape=input_batch.shape,
                    current_step=0,
                    batch_data=None,
                )

                # Apply masking to the INPUT (already undersampled), not target
                masked_input = self._apply_masking_and_verify(input_batch, mask)

                gen_kwargs["mask"] = mask
                model_input = masked_input
                # [PHYSICS DC] Pass the measured (undersampled) k-space for Data
                # Consistency — mirrors training path (L877).  DC replaces predicted
                # k-space at SAMPLED locations with the actual measurements.  This
                # is standard MRI physics, NOT leakage: it enforces fidelity to
                # acquired data, just like real inference.
                gen_kwargs["kspace_measured"] = masked_input

                # Zero-filled baseline source. ``masked_input``, NOT
                # ``target_batch``: validation masks the noisy single
                # repetition the dataloader delivered, so a target-derived ZF
                # would be a noise-free idealisation no reconstructor can
                # reach -- it would read HIGHER than the model on a healthy
                # run and hide exactly the "worse than zero-filled" verdict
                # this baseline exists to surface.
                self._zf_measurement = masked_input
            else:
                model_input = input_batch

            # [STATIC S-MAP CONDITIONING]
            # In validation, dynamically calculate ESPIRiT maps from the input_batch.
            # Must mirror the training path (_prepare_diffusion_inputs L1511) which
            # always estimates + concatenates smaps for kspace_cold_diffusion models.
            # Previous gate `in_channels == 16` was wrong for cross-contrast configs
            # where in_channels=32 (16 kspace + 16 smaps).
            #
            # Bound unconditionally: the multi-step reverse-sampling branch
            # below forwards ``smaps`` to ``gen.sample()``, and its own guard is
            # a SEPARATE expression. Leaving the name bound only inside this
            # block makes the two guards' agreement load-bearing, which is a
            # NameError waiting on the first arm that separates them.
            smaps = None
            if "kspace_cold_diffusion" in str(self.config.model.model_type).lower():
                # Calibrate from the FULLY-SAMPLED reference (kspace alias / clean
                # target), never the undersampled input — coil maps are
                # acceleration-invariant and the aliased periphery corrupts
                # calibration (CLAUDE.md #9/#16). acs_only crops to the dense center.
                acs_kspace = batch_data.get("kspace") if isinstance(batch_data, dict) else None
                if not isinstance(acs_kspace, torch.Tensor):
                    acs_kspace = target_batch
                if not isinstance(acs_kspace, torch.Tensor):
                    raise ValueError(
                        "[S-Maps] No fully-sampled k-space/target for validation "
                        "ESPIRiT calibration (CLAUDE.md #9/#16)."
                    )
                acs_kspace_t = acs_kspace.to(target_batch.device)
                h, w = acs_kspace_t.shape[-2], acs_kspace_t.shape[-1]
                if not torch.is_complex(acs_kspace_t):
                    b, c = acs_kspace_t.shape[0], acs_kspace_t.shape[1]
                    c2 = c // 2
                    if c2 < 1:
                        # Single-channel magnitude data: treat as 1-channel complex with zero imag
                        acs_kspace_t = torch.complex(acs_kspace_t, torch.zeros_like(acs_kspace_t))
                    else:
                        acs_kspace_t = torch.view_as_complex(
                            acs_kspace_t.view(b, c2, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()
                        )

                # Configured-method + sub-knob smaps fallback (pitfall #15).
                # acs_only crops to the dense center (aliasing can't seed calibration).
                smaps = estimate_smaps(
                    acs_kspace_t,
                    method=self._configured_estimation_method(),
                    acs_only=True,
                    **self._configured_estimation_kwargs(),
                ).detach()

                rss = torch.sqrt((smaps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
                smaps = smaps / rss

                # Resize if needed
                if smaps.shape[-2:] != (h, w):
                    smaps_r = torch.nn.functional.interpolate(
                        smaps.real, size=(h, w), mode="bilinear", align_corners=False
                    )
                    smaps_i = torch.nn.functional.interpolate(
                        smaps.imag, size=(h, w), mode="bilinear", align_corners=False
                    )
                    smaps = torch.complex(smaps_r, smaps_i)

                # Persist the complex S-maps so downstream consumers
                # (`_compute_validation_metrics` losses like `sense_adjoint_l1`
                # and metrics_mixin's `ifft_sense_adjoint` transform) can read
                # them. Without this, both paths see `None` and silently degrade
                # — `sense_adjoint_l1` returns 0 and `ifft_sense_adjoint`
                # falls back to per-coil iFFT magnitude (CLAUDE.md #9).
                # NOTE: do NOT also write _cached_smaps here — it causes a
                # validation-batch smaps to persist across the
                # validation→training boundary and crash sense_adjoint at
                # the first training step (iter-1001 tensor-a(2) vs b(36)).
                self._current_smaps = smaps

                # Domain translation
                if not torch.is_complex(model_input) and torch.is_complex(smaps):
                    smaps = (
                        torch.view_as_real(smaps)
                        .permute(0, 1, 4, 2, 3)
                        .reshape(smaps.shape[0], -1, *smaps.shape[2:])
                    )
                elif torch.is_complex(model_input) and not torch.is_complex(smaps):
                    smaps = torch.complex(smaps, torch.zeros_like(smaps))

                # [DOMAIN] Mirror the training path exactly (see
                # ``_prepare_diffusion_inputs``): the maps are image-domain and
                # ``model_input`` is k-space, so FFT + level-match + amplitude
                # cap before the concat.  Skipping it here would reintroduce a
                # train/val skew — the same class of defect as #1295.
                smaps_k, _ = prepare_smaps_for_kspace_conditioning(
                    smaps.detach(), model_input, channel_dim=1
                )
                model_input = torch.cat([model_input, smaps_k], dim=1)

            # [OOM FIX] Process large batches (e.g. 18 slices from 5D→4D flatten) in
            # micro-chunks to stay within GPU memory budget.  val_chunk_size=2 uses ~1/9
            # of the peak allocation compared to running all 18 slices at once.
            val_chunk_size = self.config.validation.loader.chunk_size
            n_slices = model_input.shape[0]

            def _forward_chunked(x_inp, ts, is_latent, gkw):
                """_forward_chunked.

                Args:
                    x_inp (Any): Description.
                    ts (Any): Description.
                    is_latent (Any): Description.
                    gkw (Any): Description.
                Returns:
                    Any: Description.
                """
                if x_inp.shape[0] <= val_chunk_size:
                    return self._forward_through_model(
                        x_inp,
                        timesteps=ts,
                        is_latent_diffusion=is_latent,
                        gen_kwargs=gkw,
                    )
                chunks = x_inp.split(val_chunk_size, dim=0)
                ts_chunks = (
                    ts.split(val_chunk_size, dim=0)
                    if ts is not None and ts.shape[0] == x_inp.shape[0]
                    else [ts] * len(chunks)
                )
                parts = []
                for c_x, c_ts in zip(chunks, ts_chunks, strict=False):
                    out = self._forward_through_model(
                        c_x,
                        timesteps=c_ts,
                        is_latent_diffusion=is_latent,
                        gen_kwargs=gkw,
                    )
                    if isinstance(out, tuple):
                        out = out[0]
                    if isinstance(out, dict):
                        out = out.get("image", out.get("kspace", next(iter(out.values()))))
                    parts.append(out)
                return torch.cat(parts, dim=0)

            if is_cold_diffusion:
                gen = (
                    self.generator_model.module
                    if hasattr(self.generator_model, "module")
                    else self.generator_model
                )
                # 2026-06-08 OPT-IN true multi-step cold restoration. Default
                # (knob absent/False) keeps the single deterministic forward — a
                # one-shot x0 regressor that posterior-mean-blurs at heavy R. With
                # validation.multistep_cold_sampling ON, run the registered
                # cold_mri reverse loop (PhysicsInformedColdDiffusion.sample):
                # x_T=measurement -> ... -> x_0, enforcing data consistency
                # against the MEASURED k-space every step. Use `masked_input` (the
                # PRE-smap-concat undersampled measurement) and `mask`: sample()
                # does its own physics, so it must NOT get the smap-concatenated
                # model_input. sample() returns K-SPACE (matching forward()), so
                # the downstream cold-branch IFFT in _compute_validation_metrics
                # stays correct — calling generate() (image output) would
                # re-create the DC blob. NOTE (pitfall #18): this makes the val
                # metric measure the ITERATIVE recon, a DIFFERENT quantity than
                # the single-step training loss — re-baseline selection thresholds.
                _multistep = (
                    self.config.validation.sampling.enable_multistep_cold
                    if self.config.validation is not None
                    else False
                )
                if _multistep and not hasattr(gen, "sample"):
                    # Non-negotiable 3. The arm DECLARED the multi-step cold
                    # reverse loop; silently running the single-step forward
                    # instead measures a different quantity and is
                    # indistinguishable in the logs from the declared path
                    # working. Census (2026-09-01): all 55 arms setting this
                    # flag resolve to KSpaceColdDiffusionGenerator, which
                    # defines sample() — so this raises on none of them.
                    raise AttributeError(
                        "validation.sampling.enable_multistep_cold=true, but the "
                        f"resolved generator {type(gen).__name__} exposes no "
                        "sample(); the multi-step cold reverse loop cannot run. "
                        "Set enable_multistep_cold=false to validate with the "
                        "single-step forward, or use a generator implementing "
                        "sample()."
                    )
                if _multistep:
                    steps = self._resolve_validation_sampling_steps()
                    n_members = self._resolve_validation_ensemble(gen)
                    # Chunk the reverse-sampling forward by val_chunk_size to cap
                    # peak GPU memory — the single-step path already does this via
                    # _forward_chunked; this multi-step path used to bypass it and
                    # OOM'd every validation on experiment_11 (see
                    # _sample_multistep_chunked).
                    # Pass the S-maps SEPARATELY (never folded into
                    # ``measurement``): ``sample()`` runs its own physics and
                    # must receive pure measured k-space, while the generator
                    # re-derives the k-space-domain map stack per reverse step
                    # against the current state. Omitting them is what made
                    # this path reconstruct through an untrained ChannelAdapter.
                    if n_members > 1:
                        # N stochastic members of the SAME input, one per C6
                        # seed offset. The k-space mean goes on as the
                        # prediction -- by linearity of the IFFT it is the
                        # pixelwise mean of the complex images (on a log-scaled
                        # arm that mean is taken in compressed units, like every
                        # other k-space average on this path). The members are
                        # stashed for the spread, which only means something in
                        # the metric domain, so ``_compute_validation_metrics``
                        # takes it there.
                        members = self._sample_ensemble_chunked(
                            gen,
                            masked_input,
                            mask,
                            steps,
                            n_members,
                            smaps=smaps,
                            start_timestep=start_timestep,
                        )
                        self._ensemble_members = members
                        hr_fakes = members.mean(dim=0)
                    else:
                        hr_fakes = self._sample_multistep_chunked(
                            gen,
                            masked_input,
                            mask,
                            steps,
                            smaps=smaps,
                            start_timestep=start_timestep,
                        )
                else:
                    hr_fakes = _forward_chunked(
                        model_input, timestep, self._is_latent_diffusion(), gen_kwargs
                    )
            elif self._is_latent_diffusion():
                # Latent diffusion: model.forward() returns noise-prediction in
                # latent space, not a decoded image.  Call model.sample() which
                # runs the full reverse diffusion and decodes to image space.
                gen = (
                    self.generator_model.module
                    if hasattr(self.generator_model, "module")
                    else self.generator_model
                )
                if hasattr(gen, "sample"):
                    # Use target shape so output matches [B, C, H, W] of target.
                    sample_kwargs = {
                        k: v
                        for k, v in gen_kwargs.items()
                        if k in ("contrast_idx", "condition_image")
                    }
                    # WS1-core-04: honor the configured validation sampling-step
                    # count. Pass it ONLY under the parameter name this sampler
                    # actually exposes (ddim: num_inference_steps; others vary),
                    # so a sampler that doesn't accept it keeps its own default —
                    # i.e. a strict no-op for arms that set neither knob.
                    steps = self._resolve_validation_sampling_steps()
                    if steps is not None:
                        params = inspect.signature(gen.sample).parameters
                        for _name in _SAMPLER_STEP_PARAM_NAMES:
                            if _name in params:
                                sample_kwargs[_name] = steps
                                break
                        else:
                            # No matching parameter: the configured step count
                            # cannot be honoured. Say so -- silently dropping it
                            # is how ldm_ulf_to_hf's `sampler_steps: 25` ran the
                            # full 1000-step chain for months (pitfall #15).
                            self.logging_service.log_warning(
                                f"validation.sampler_steps={steps} is configured but "
                                f"{type(gen).__name__}.sample() exposes none of "
                                f"{_SAMPLER_STEP_PARAM_NAMES} -- the sampler will run "
                                f"its own default step count, NOT {steps}."
                            )
                    hr_fakes = gen.sample(
                        target_batch.shape,
                        device=str(target_batch.device),
                        **sample_kwargs,
                    )
                else:
                    # Fallback: single-step forward (may produce latent-space output)
                    noisy_input = self.q_sample(
                        model_input, timestep, torch.randn_like(model_input)
                    )
                    hr_fakes = _forward_chunked(noisy_input, timestep, True, gen_kwargs)
            else:
                # F13 (2026-05-17 round 5): score-based and consistency
                # generators return *predicted noise* (score-based) or a
                # high-noise-level approximation (consistency) when called
                # via plain forward(). Saving that as "fake_images" produced
                # pure-noise validation PNGs (V2 in the 2026-05-16 smoke
                # mosaic; persisted after the F12 mtime filter).
                #
                # The fix: detect models with a dedicated sampler method
                # (``generate``, ``p_sample_loop``, or
                # ``multistep_inference``) and call it with the clean input
                # so visualization saves the reverse-diffusion output, not
                # an intermediate noised sample. Falls back to the legacy
                # noisy-forward path only when neither method exists.
                gen = (
                    self.generator_model.module
                    if hasattr(self.generator_model, "module")
                    else self.generator_model
                )
                _diffusion_cfg = (
                    self.config.training.diffusion if self.config.training is not None else None
                )
                _inf_sampler = str(getattr(_diffusion_cfg, "inference_sampler", "") or "").lower()
                if _inf_sampler in _POSTERIOR_RECON_SAMPLERS and hasattr(gen, "sample"):
                    # XC.1: a posterior reconstruction sampler (e.g. dds) is
                    # configured — dispatch the CONDITIONED reverse loop through
                    # the SamplerRegistry with the measurement, rather than the
                    # unconditional ``generate``. This consumes ``inference_sampler:
                    # dds`` end-to-end on a score arm (gated on the posterior set,
                    # so default score arms keep the ``generate`` path).
                    hr_fakes = gen.sample(
                        measurement=gen_kwargs.get("kspace_measured"),
                        mask=gen_kwargs.get("mask"),
                    )
                elif hasattr(gen, "multistep_inference"):
                    # Consistency model: 2-step inference produces a clean
                    # denoised image; matches the paper's recommended setup.
                    hr_fakes = gen.multistep_inference(
                        model_input,
                        steps=2,
                        measurement=gen_kwargs.get("kspace_measured"),
                        mask=gen_kwargs.get("mask"),
                    )
                elif hasattr(gen, "generate"):
                    # Score-based diffusion: ``generate`` runs the full
                    # reverse SDE (p_sample_loop) and returns a clean image.
                    hr_fakes = gen.generate(model_input)
                else:
                    # Legacy single-step forward path (kept for backward
                    # compatibility with non-diffusion generators that route
                    # through DiffusionTrainingStrategy).
                    noisy_input = self.q_sample(
                        model_input, timestep, torch.randn_like(model_input)
                    )
                    hr_fakes = _forward_chunked(
                        noisy_input, timestep, self._is_latent_diffusion(), gen_kwargs
                    )

            if isinstance(hr_fakes, tuple):
                hr_fakes = hr_fakes[0]
            if isinstance(hr_fakes, dict):
                hr_fakes = hr_fakes.get(
                    "image", hr_fakes.get("kspace", next(iter(hr_fakes.values())))
                )

            # [FIX] Truncate conditioning channels (e.g. smaps) if the generator returned them
            if hasattr(hr_fakes, "shape") and hasattr(target_batch, "shape"):
                fakes_ch = hr_fakes.shape[1]
                target_ch = target_batch.shape[1]
                if fakes_ch > target_ch:
                    self.logging_service.log_debug(
                        f"Validation prediction has {fakes_ch} channels, but target has {target_ch}. Truncating."
                    )
                    hr_fakes = hr_fakes[:, :target_ch]

            self.logging_service.log_info(
                f"[DIAGNOSTIC] _generate_validation_prediction RETURN: "
                f"hr_fakes={list(hr_fakes.shape)}, target_batch={list(target_batch.shape)}, "
                f"hr_complex={torch.is_complex(hr_fakes)}"
            )

            # WARNING level (not the INFO above) is required for this to be
            # seen at all: ``LoggingService.setup`` clamps every logger to
            # ``logging.sinks.level``, and the cascading-validation arms run at
            # ``warning`` — every INFO diagnostic in this method is discarded
            # before it reaches the log. See ``describe_nonfinite_prediction``.
            nonfinite_report = describe_nonfinite_prediction(hr_fakes)
            if nonfinite_report is not None:
                self.logging_service.log_warning(nonfinite_report)

            return hr_fakes

        except Exception as e:
            if self.logging_service.logger.isEnabledFor(logging.WARNING):
                self.logging_service.log_warning(f"Validation generation failed: {e!s}")
            import traceback

            self.logging_service.log_warning(traceback.format_exc())
            return None

    def _compute_validation_metrics(
        self,
        hr_fakes: torch.Tensor,
        target_batch: torch.Tensor,
        input_batch: torch.Tensor,
        timestep: torch.Tensor,
        batch_data: Any,
        scale_factor: torch.Tensor,
        batch_idx: int = 0,
        cascade_level: float | None = None,
        heldout: bool = False,
        timestep_used: int | None = None,
        acceleration_realized: float | None = None,
        emit_reports: bool = True,
    ) -> dict[str, float]:
        """Compute validation metrics.

        Args:
            cascade_level: Acceleration factor for the current cascade pass
                (e.g., 2.0, 8.0, 32.0). Threaded through to the validation
                image saver so each level lands in its own PNG instead of
                overwriting earlier levels — the May 2026
                "experiment_11 fake images doubled" symptom was the 32x
                level overwriting the 4x / 8x renders, leaving only the
                worst-case heavily-aliased output on disk. ``None`` falls
                back to the legacy single-prefix behaviour.
            heldout: True for a point from the opt-in held-out severity grid
                rather than the in-distribution ladder. Recorded as a per-case
                COLUMN, not as a naming convention — the flat metrics spell it
                ``_heldout_<R>x``, which means a reader who filters on
                acceleration alone silently pools out-of-distribution points
                with in-distribution ones.
            timestep_used: The scalar timestep this rung actually ran at,
                already resolved by the caller. Passed in rather than reduced
                from ``timestep`` here so the per-case row costs no GPU sync
                (non-negotiable 9); ``None`` leaves the column blank, which is
                the honest reading of "not resolved" — 0 is a real timestep.
            acceleration_realized: What the schedule decoded ``timestep_used``
                back to, as already computed by the caller for the tall row
                (#1295) -- passed in rather than re-inverted here so the two
                surfaces cannot disagree. ``None`` on the linear-fallback path,
                where there is no schedule to invert, and left blank rather than
                restated as the request.
            emit_reports: False suppresses the TensorBoard/report-recorder
                emission while still returning the numbers. Needed because this
                method is called a SECOND time on the same batch by the t=0
                pre-DC probe (#1682) to score a different prediction; the
                emission is not idempotent -- it appends a per-case row through
                ``feed_report_case_recorder`` and, at ``cascade_level=None``,
                takes the legacy single-prefix path that overwrites the
                per-level PNGs. Scoring is side-effect free; emitting is not.
        """
        self.logging_service.log_info(
            f"[DIAGNOSTIC] _compute_validation_metrics ENTRY: "
            f"hr_fakes={list(hr_fakes.shape)}, target={list(target_batch.shape)}, "
            f"hr_complex={torch.is_complex(hr_fakes)}, tgt_complex={torch.is_complex(target_batch)}"
        )
        is_cold_diffusion = self._is_cold_diffusion()

        hr_fakes = hr_fakes.to(target_batch.device)

        denom_scale = torch.ones(
            hr_fakes.size(0), 1, 1, 1, device=hr_fakes.device, dtype=hr_fakes.dtype
        )

        if is_cold_diffusion and self.config.data.processing.enable_kspace_normalization:
            scale_tensor = scale_factor
            if scale_tensor is None:
                # Mapping protocol, not isinstance-dict + hasattr: on a
                # TrainingBatch both legs missed and denom_scale silently stayed
                # at ones, so metrics were graded in normalized units.
                scale_tensor = read_batch_field(batch_data, "kspace_scale")

            if scale_tensor is not None:
                # ``hr_fakes`` is per SLICE while the published scale is per
                # SUBJECT, so their leading dims differ by the depth factor.
                # Reconcile against ``hr_fakes`` -- this read is independent of
                # the one in ``_prepare_validation_data``, so fixing only that
                # site would leave the ``read_batch_field`` fallback here live.
                denom_scale = align_scale_to_batch(
                    scale_tensor,
                    hr_fakes.size(0),
                    field="kspace_scale",
                    device=hr_fakes.device,
                ).to(hr_fakes.dtype)
            else:
                try:
                    if torch.is_complex(input_batch):
                        input_mag = input_batch.abs()
                    elif input_batch.shape[1] % 2 == 0:
                        B, C, H, W = input_batch.shape
                        i_reshaped = (
                            input_batch.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
                        )
                        i_complex = torch.view_as_complex(i_reshaped).permute(0, 3, 1, 2)
                        input_mag = torch.sqrt(
                            torch.sum(i_complex.abs() ** 2, dim=1, keepdim=True) + 1e-8
                        )
                    else:
                        input_mag = input_batch.abs()

                    batch_size = input_mag.size(0)
                    scales = []
                    for i in range(batch_size):
                        sample_mag_flat = input_mag[i].flatten()
                        if sample_mag_flat.numel() > 0:
                            sample_scale = torch.quantile(sample_mag_flat.float(), 0.99)
                            sample_scale = torch.maximum(
                                sample_scale,
                                torch.tensor(
                                    1e-6,
                                    device=sample_scale.device,
                                    dtype=sample_scale.dtype,
                                ),
                            )
                        else:
                            sample_scale = torch.tensor(
                                1e-6, device=input_mag.device, dtype=input_mag.dtype
                            )
                        scales.append(sample_scale)

                    denom_scale = torch.stack(scales).view(batch_size, 1, 1, 1).to(hr_fakes.dtype)
                except Exception as e:
                    if self.logging_service.logger.isEnabledFor(logging.WARNING):
                        self.logging_service.log_warning(
                            f"Failed to compute denormalization scale: {e!s}"
                        )
                    denom_scale = torch.ones_like(denom_scale)

        # Invert the log-compression (if the data pipeline applied it) BEFORE
        # undoing the percentile scale. Forward was k2 = log1p(|k / scale|);
        # the inverse is k = expm1(|k2|) * scale. Skipping the expm1 leaves the
        # prediction in compressed space, so the IFFT'd validation image is
        # dominated by the data-consistency-injected measured DC -> centre blob.
        #
        # ``enable_log_scaling`` is a *declaration*, and applying ``expm1`` on
        # its word alone is the failure #682 keeps reproducing under new
        # symptoms. On a tensor that was never compressed,
        # ``decompress_kspace_log`` clamps the magnitude at
        # ``DECOMPRESS_MAGNITUDE_CEILING`` and ``expm1`` collapses the entire
        # contrast-carrying band onto the single value ``expm1(30) ~ 1.07e13``,
        # leaving a constant-magnitude / varying-phase spectrum whose IFFT is a
        # phase-only render. Measured on this arm's own snapshot tensors: the
        # prediction's magnitude field dropped from 1087 distinct values to 5.
        #
        # This is the render path's *only* pre-IFFT transform, and it is applied
        # to ``log_preds``/``log_targets`` -- the tensors that the saved
        # ``fake_images``/``real_images`` PNGs AND the validation metrics are
        # both derived from. An unverified ``expm1`` here therefore corrupts
        # every validation artifact at once, in silence.
        #
        # ``decompress_for_view`` is the verifying entry point (the same guard
        # the snapshot/figure render paths already use, previously bypassed
        # here): it checks the declaration against ``|k|max`` before acting,
        # skips ``expm1`` with a WARNING naming both diagnoses -- pipeline data
        # declared log-scaled but never compressed, vs. a prediction that
        # diverged in compressed units -- and warns again when decompression
        # fails to expand. It is a no-op when ``log_scaled`` is False and
        # applies its own complex/even-channel shape gate, so the two branches
        # and their duplicated shape tests collapse into one call per tensor.
        #
        # The two ``[LOG-DECOMPRESS]`` INFO diagnostics that used to live here
        # are gone deliberately, not lost: ``LoggingService.setup`` clamps every
        # logger to ``logging.sinks.level``, which is ``warning`` on the arms
        # that reach this path, so neither line could ever be emitted -- while
        # their eagerly-evaluated f-strings still forced three device->host
        # syncs per validation batch. ``decompress_for_view`` reports the same
        # applied-|k|max transition at DEBUG with lazy %-formatting, and raises
        # the *failure* case to WARNING so it survives the clamp.
        _log_scaled = self.config.data.processing.enable_log_scaling

        # Pop (read-and-clear) the k-space the model was actually given, stashed
        # by ``_generate_validation_prediction``. Clearing here as well as at the
        # producer's top makes a stale read structurally impossible in BOTH
        # directions: a rung whose generation never reached the masking step
        # loses its baseline rather than inheriting the previous rung's.
        _zf_measurement = self._zf_measurement
        self._zf_measurement = None
        zf_for_metrics = None
        # Read-and-clear, like the measurement above: the t=0 pre-DC probe
        # scores a DIFFERENT prediction through this method on the same batch
        # (#1682) and must not inherit this rung's spread.
        _ensemble_members = self._ensemble_members
        self._ensemble_members = None
        ensemble_std: torch.Tensor | None = None

        if is_cold_diffusion:
            from spectramr.infrastructure.training.utils.kspace_view import (
                decompress_for_view,
            )

            hr_fakes = decompress_for_view(hr_fakes, log_scaled=_log_scaled, channel_dim=1)
            target_batch = decompress_for_view(target_batch, log_scaled=_log_scaled, channel_dim=1)
            # #1684: ``input_batch`` is rendered beside these two (the
            # ``val/inputs`` panel and the report-case recorder), so it must
            # cross the SAME seam. It used to cross neither, which put the
            # input panel ~65-100x off and made every saved case's ``inputs``
            # array incomparable with its own ``predictions``/``targets``.
            input_batch = decompress_for_view(input_batch, log_scaled=_log_scaled, channel_dim=1)

            # The zero-filled baseline is a MEASUREMENT, not a prediction, but
            # it is scored against the same target on the same axis, so it
            # crosses the identical seam. ``log1p`` is non-linear: an inverse
            # FFT of still-compressed k-space is not the zero-filled
            # reconstruction, it is the reconstruction of a different signal --
            # the #1684 lesson applied to a number instead of a picture.
            if _zf_measurement is not None:
                _zf_measurement = decompress_for_view(
                    _zf_measurement, log_scaled=_log_scaled, channel_dim=1
                )

        hr_fakes_for_metrics = hr_fakes * denom_scale
        target_for_metrics = target_batch * denom_scale
        # Same ``denom_scale`` as predictions/targets -- one scale per batch, so
        # the three tensors stay on a common axis.
        log_inputs = input_batch * denom_scale
        if _zf_measurement is not None:
            zf_for_metrics = _zf_measurement * denom_scale

        self.logging_service.log_debug(
            f"Denormalized k-space: pred_mean={hr_fakes_for_metrics.abs().mean().item():.2f}, "
            f"target_mean={target_for_metrics.abs().mean().item():.2f}"
        )

        log_preds = hr_fakes_for_metrics
        log_targets = target_for_metrics
        # SSOT: domain inference decides whether predictions / targets need an
        # IFFT before image-domain visualization. ``model.input_type`` is NOT
        # the right signal — a generator with ``input_type: image`` may still
        # output k-space (see src/infrastructure/training/utils/domain_inference.py
        # docstring). The mixin path in metrics_mixin.py already uses this
        # SSOT; the diffusion strategy now follows suit.
        try:
            from spectramr.infrastructure.training.utils.domain_inference import (
                needs_ifft_for_visualization,
            )

            _needs_ifft_preds, _needs_ifft_targets = needs_ifft_for_visualization(self.config)
        except Exception:  # pragma: no cover — defensive only
            _input_type = (self.config.model.input_type or "kspace").lower()
            _needs_ifft_preds = _input_type == "kspace"
            # Mirror rather than re-derive: re-implementing the SSOT's
            # dataset_type rule here would make this a second owner of it
            # (non-negotiable 17) inside a branch that is defensive-only.
            _needs_ifft_targets = _needs_ifft_preds
        # ``is_preds_image`` retains its historical name for downstream call
        # sites: True when the prediction tensor is already in image domain
        # (no IFFT needed for display); False when it's k-space.
        is_preds_image = not _needs_ifft_preds
        # ``inputs`` comes off the DATALOADER, so its domain is the *targets*
        # element of the SSOT pair -- not ``is_preds_image``, which describes
        # the MODEL OUTPUT and is recomputed below after
        # ``_apply_metric_transforms``, a transform ``inputs`` never undergoes.
        inputs_are_image = not _needs_ifft_targets

        # Handle complex targets.
        #
        # F5b (2026-05-24 smoke_audit_20260524): mirror the F5a guard
        # (the logger's predictions/targets ``.abs()`` at
        # ``_log_validation_images_to_tensorboard``) at this *upstream*
        # site too. When ``compute_image_metrics`` is on, this
        # ``target_for_metrics`` flows into ``_apply_metric_transforms``
        # (``ifft_magnitude``) and then into ``log_targets`` for the
        # validation mosaic. An unconditional ``.abs()`` on a multi-coil
        # complex k-space target (e.g. ``coil_processing_mode='none'``)
        # phase-strips it to ``|kspace|``; the subsequent ``ifft2c`` of a
        # real (Hermitian) signal is centro-symmetric — the "doubled
        # brain" real-image regression. Only magnitude-convert genuine
        # single-coil complex *image* tensors (``shape[1] <= 2``); leave
        # multi-coil complex for ``ifft_magnitude`` to pair (R, I),
        # per-coil IFFT, and RSS-combine correctly. See
        # TODO/audit/smoke_audit_20260524.md §F5b.
        if torch.is_complex(target_for_metrics):
            if target_for_metrics.dim() < 4 or target_for_metrics.shape[1] <= 2:
                target_for_metrics = torch.abs(target_for_metrics)

        self.logging_service.log_info(
            f"[Metric Input Shapes] hr_fakes={hr_fakes_for_metrics.shape}, "
            f"target={target_for_metrics.shape}, "
            f"hr_cmplx={torch.is_complex(hr_fakes_for_metrics)}, "
            f"tgt_cmplx={torch.is_complex(target_for_metrics)}"
        )

        hr_fakes_for_metrics, target_for_metrics = TorchIOAdapter.ensure_channel_match(
            hr_fakes_for_metrics, target_for_metrics
        )

        self.logging_service.log_info(
            f"[Metric After Adapter] hr_fakes={hr_fakes_for_metrics.shape}, "
            f"target={target_for_metrics.shape}"
        )

        metrics = {}

        env_losses = {}
        if hasattr(self, "context") and self.context and hasattr(self.context, "loss_fn"):
            env_losses = self.env.losses if self.env else {}

        val_loss = torch.tensor(0.0, device=hr_fakes_for_metrics.device)
        if env_losses:
            # Mirror the training-time loss-call contract: forward `smaps`
            # (and any other physics kwargs) via `_call_safe_loss` so signature-
            # aware losses like `sense_adjoint_l1` receive what they need.
            # Direct `loss_fn(pred, target)` calls were silently triggering
            # `SENSEAdjointL1Loss`'s `smaps is None` early-return (== 0.0),
            # masking the loss in CSV/Sim2Rank trajectories.
            loss_kwargs: dict[str, Any] = {
                "smaps": getattr(self, "_current_smaps", None),
            }
            for loss_name, loss_fn in env_losses.items():
                if not callable(loss_fn):
                    continue
                loss_val = _call_safe_loss(
                    loss_fn,
                    hr_fakes_for_metrics,
                    target_for_metrics,
                    **loss_kwargs,
                )
                if isinstance(loss_val, torch.Tensor):
                    # [STABILIZATION] Use large iteration for validation to bypass warm-up masking
                    weight = self._get_loss_weight(loss_name, iteration=1000000)
                    val_loss = val_loss + weight * loss_val
                    metrics[f"val_{loss_name}"] = loss_val.detach()
        else:
            _mse_fn = create_loss("mse")
            val_loss = _mse_fn(hr_fakes_for_metrics, target_for_metrics)

        val_loss_scalar = (
            val_loss.detach() if isinstance(val_loss, torch.Tensor) else float(val_loss)
        )
        metrics["val_loss"] = val_loss_scalar
        metrics["g_total_loss"] = val_loss_scalar

        # ✅ SSOT: Direct access to validation config (trust schema)
        val_config = self.config.validation
        if val_config and val_config.scoring.enable_image_metrics:
            self.logging_service.log_debug(
                f"[Validation Metrics] Before transform - "
                f"hr_fakes: {list(hr_fakes_for_metrics.shape)} (ndim={hr_fakes_for_metrics.ndim}), "
                f"target: {list(target_for_metrics.shape)} (ndim={target_for_metrics.ndim})"
            )

            # ✅ SSOT: Extract transform name from validation config
            transform = val_config.scoring.output_transform or "none"

            self.logging_service.log_debug(
                f"[Validation Transform] Applying transform: {transform}"
            )

            # Use SSOT ValidationMetricsComputer via Base Strategy Helper
            # This unified helper respects both 'domain' and 'transform' settings.
            # Pass denormalized k-space data to transform.
            # Data is already in physical units from denormalization step above.
            # Transform (e.g., ifft_magnitude) will convert to image space while
            # preserving physical amplitude scale for accurate PSNR computation.
            # Pass the CONFIG OBJECT, as gan.py / field_cocycle / the mixin's own
            # validation path all do. This used to pass the resolved `transform`
            # STRING into a parameter read with `_get_config_value`, where every
            # key lookup on a str is None -- so the name was dropped and the
            # function returned its input unchanged (#927).
            pred_transformed, target_transformed = self._apply_metric_transforms(
                hr_fakes_for_metrics, target_for_metrics, val_config
            )
            if _ensemble_members is not None:
                ensemble_std = self._ensemble_std_in_metric_domain(
                    _ensemble_members,
                    denom_scale,
                    _log_scaled,
                    target_for_metrics,
                    val_config,
                    is_cold_diffusion,
                )

            # The prediction is NOT clamped to the target's range.
            #
            # Until 2026-08-18 this read::
            #
            #     if not torch.is_complex(pred_transformed):
            #         pred_transformed = torch.clamp(
            #             pred_transformed, min=0.0,
            #             max=target_transformed.abs().max() * 1.1,
            #         )
            #
            # which is a facade in the sense of pitfall #16: it bounded the
            # prediction by a statistic of the GROUND TRUTH, so a model whose
            # output ran hot was silently rewritten into a flat field at
            # ``1.1 * target.abs().max()`` instead of being reported.
            #
            # experiment_11_attention_none, 2026-08-18, is the worked example.
            # Every saved ``fake`` sample at R2x/R8x/R32x was bit-exactly
            # constant at 3.8493783473968506 == float32(3.49943470954895) *
            # float32(1.1), i.e. the ceiling itself and nothing but the
            # ceiling -- reproduced on two independent clusters. The renders
            # were uniformly white, the ``render_windows.json`` sidecars
            # recorded ``finite_min == finite_max``, and PSNR/SSIM were
            # computed against a target-bounded prediction, which flatters a
            # diverged model precisely when it most needs to be visible.
            #
            # Same shape of fix as f504035a8 ("retract the black-fake
            # mechanism; measure it instead") and #179: retract the rewrite,
            # keep the observation. The scale relation is measured below and
            # stamped into the returned metrics, so the next run quantifies
            # what this clamp used to hide.
            _pred_scale = self._measure_prediction_scale(pred_transformed, target_transformed)
            for _scale_key, _scale_val in _pred_scale.items():
                metrics[f"val_{_scale_key}"] = _scale_val
            if _pred_scale["pred_above_target_fraction"] >= self._PRED_SCALE_WARN_FRACTION:
                self.logging_service.log_warning(
                    "[Validation Scale] "
                    f"{_pred_scale['pred_above_target_fraction']:.1%} of the prediction "
                    f"sits above the target's peak magnitude "
                    f"({_pred_scale['target_abs_max']:.4g}); pred |max| is "
                    f"{_pred_scale['pred_target_scale_ratio']:.3g}x the target's. "
                    "The prediction is reported as produced -- a saturated or "
                    "featureless render here is the model's output, not a saver "
                    "artifact. Check the sampler, the normalization inverse, and "
                    "whether the model has simply not trained yet."
                )

            if self.logging_service.logger.isEnabledFor(logging.INFO):
                p_min = (
                    pred_transformed.abs().min().item()
                    if torch.is_complex(pred_transformed)
                    else pred_transformed.min().item()
                )
                p_max = (
                    pred_transformed.abs().max().item()
                    if torch.is_complex(pred_transformed)
                    else pred_transformed.max().item()
                )
                t_min = (
                    target_transformed.abs().min().item()
                    if torch.is_complex(target_transformed)
                    else target_transformed.min().item()
                )
                t_max = (
                    target_transformed.abs().max().item()
                    if torch.is_complex(target_transformed)
                    else target_transformed.max().item()
                )
                self.logging_service.log_info(
                    f"[METRIC] Unscaled - pred: [{p_min:.4f}, {p_max:.4f}], "
                    f"target: [{t_min:.4f}, {t_max:.4f}]"
                )

            # Debug: Log shapes after transform
            self.logging_service.log_debug(
                f"[Validation Metrics] After transform - "
                f"pred: {list(pred_transformed.shape)}, "
                f"target: {list(target_transformed.shape)} | "
                f"pred.ndim={pred_transformed.ndim}, pred.dtype={pred_transformed.dtype}, "
                f"pred.device={pred_transformed.device}"
            )

            # Set vars for tensorboard logging so we don't accidentally inverse-FFT true images!
            log_preds = pred_transformed
            log_targets = target_transformed
            # ``log_inputs`` is deliberately NOT reassigned here, and
            # ``inputs_are_image`` deliberately NOT recomputed: the metric
            # transform runs on predictions/targets only. Rebinding either to a
            # post-transform value would claim a domain change that never
            # happened to ``inputs`` and skip its IFFT. The asymmetry is the point.
            # DERIVED, not asserted (#927). ``_apply_metric_transforms`` has four
            # paths that hand back their input untouched, and this used to read
            # ``= True`` regardless -- so on a no-op the flag claimed a domain
            # change that never happened. Downstream that reaches
            # ``kspace_to_image(already_image=...)``, whose guard then raises and
            # takes every validation image with it: experiment_11_attention_none
            # on 2026-08-08 logged the failure 135 times and wrote ZERO images,
            # while PSNR -- computed over the same unconverted k-space -- read
            # 58 dB with robust_mri_psnr NaN.
            from spectramr.infrastructure.training.utils.domain_inference import (
                metric_transform_produced_image,
            )

            is_preds_image = metric_transform_produced_image(hr_fakes_for_metrics, pred_transformed)
            if not is_preds_image:
                self.logging_service.log_warning(
                    f"[Validation Metrics] transform={transform!r} returned its "
                    f"input unchanged ({tuple(pred_transformed.shape)}, "
                    f"{pred_transformed.dtype}); predictions are NOT image "
                    "domain. Images will still render (the visualiser "
                    "transforms them), but the metrics for this batch are "
                    "computed in the PRE-transform domain -- PSNR over k-space "
                    "reads implausibly high and robust_mri_psnr can go NaN. "
                    "See issue #927."
                )

            # CRITICAL: Verify shapes are 4D
            if pred_transformed.ndim != 4 or target_transformed.ndim != 4:
                self.logging_service.log_error(
                    f"[SHAPE ERROR] Unexpected dimensions after transform: "
                    f"pred.ndim={pred_transformed.ndim}, target.ndim={target_transformed.ndim}. "
                    f"Expected 4D tensors [B,C,H,W]. "
                    f"pred.shape={pred_transformed.shape}, target.shape={target_transformed.shape}"
                )
                raise ValueError(
                    f"Dimensions after transform are not 4D. "
                    f"pred.shape={pred_transformed.shape}, target.shape={target_transformed.shape}"
                )

            # ✅ SSOT: Direct access to validation domain/transform (trust schema defaults)
            domain = val_config.scoring.domain or "image"

            # transform is already extracted above

            try:
                computer = self.validation_metrics_computer

                with torch.no_grad():
                    dynamic_data_range_tensor = target_transformed.abs().max()
                    dynamic_data_range = (
                        dynamic_data_range_tensor.item()
                        if dynamic_data_range_tensor > 1e-6
                        else 1.0
                    )

                # --- zero-filled baseline -------------------------------
                #
                # Ordered BEFORE the primary compute on purpose:
                # ``ValidationMetricsComputer.compute`` clears
                # ``last_not_applicable`` per call, and ``training_loop.py``
                # reads that attribute after validation. A subset call placed
                # after the full one would leave the N/A reporter able to see
                # only ``psnr``/``hfen``.
                #
                # ``_apply_metric_transforms`` is the SAME call the prediction
                # crosses -- so when the arm grades in image domain the ZF is
                # ``|ifft2c(masked_input)|`` produced by the configured
                # transform, and when the arm grades with ``transform: none``
                # the ZF is graded in k-space too. Comparability with
                # ``val_<k>`` is what makes the number mean anything, so it is
                # the transform, not a hard-coded IFFT, that decides.
                # ``target_transformed`` is REUSED rather than taken from this
                # call's second return value, and ``dynamic_data_range`` is
                # reused rather than re-measured off the baseline: either
                # re-derivation would put ``val_zf_psnr`` on a different axis
                # from ``val_psnr`` while still looking like a PSNR.
                zf_metrics: dict[str, float] = {}
                if zf_for_metrics is not None:
                    zf_transformed, _ = self._apply_metric_transforms(
                        zf_for_metrics, target_for_metrics, val_config
                    )
                    zf_metrics = computer.compute(
                        zf_transformed,
                        target_transformed,
                        data_range=dynamic_data_range,
                        only=self._ZF_BASELINE_METRICS,
                    )
                    for _zf_k, _zf_v in zf_metrics.items():
                        metrics[f"val_zf_{_zf_k}"] = _zf_v

                unscaled_metrics = computer.compute(
                    pred_transformed,
                    target_transformed,
                    data_range=dynamic_data_range,
                )
                # Measurement-aware metrics (nse_hall, ndcr) need the sampling
                # mask and the acquired k-space; both come from the stashed
                # masked measurement, per rung. They are computed on the complex
                # coil-combined images, not on the magnitude the arm's metric
                # transform hands the main computer (cohort review 2026-09-02).
                if is_cold_diffusion and zf_for_metrics is not None:
                    for _ma_k, _ma_v in self._measurement_aware_metrics(
                        hr_fakes_for_metrics, target_for_metrics, zf_for_metrics, computer
                    ).items():
                        metrics[f"val_{_ma_k}"] = _ma_v

                self.logging_service.log_debug(f"Computed metrics: {list(unscaled_metrics.keys())}")

                # Format metrics strings for logging
                unscaled_str = " | ".join(f"{k}: {v:.4f}" for k, v in unscaled_metrics.items())

                # Log diagnostic info with both unscaled and scaled metrics
                self.logging_service.log_info(
                    f"[Validation Metrics] Input: {list(hr_fakes_for_metrics.shape)} (hr_fakes) | "
                    f"Domain: {domain} | Transform: {transform} | "
                    f"Final Shape: {list(pred_transformed.shape)}"
                )
                # Log validation metrics only if logging enabled
                if self.logging_service.logger.isEnabledFor(logging.INFO):
                    self.logging_service.log_info(f"[Validation Metrics] UNSCALED: {unscaled_str}")

                # Store metrics from unscaled predictions only.
                # One key per value. ``val_{k}_unscaled`` was an exact
                # byte-for-byte alias of ``val_{k}`` -- same ``v``, same loop --
                # so it doubled this dict for no information, and the DDP
                # all-reduce packs sorted(keys) positionally, making every
                # duplicate a real collective payload. Its only reader was a
                # ``.get(primary, .get(alias))`` fallback that could never fire.
                for k, v in unscaled_metrics.items():
                    metrics[f"val_{k}"] = v

                # Ensemble spread and coverage, AFTER the loop above: an arm
                # that also lists ``empirical_coverage`` in ``metrics.compute``
                # gets the computer's declared N/A (NaN) for it there, and the
                # loop would overwrite a value written earlier.
                if ensemble_std is not None:
                    for _e_k, _e_v in self._ensemble_metrics(
                        pred_transformed, target_transformed, ensemble_std
                    ).items():
                        metrics[f"val_{_e_k}"] = _e_v

                # Signed, NOT oriented to "positive is better".
                # ``metric_higher_is_better`` -- the single direction resolver
                # for the metrics tracker, ``keep_best_n``, early stopping and
                # the leaderboard -- resolves an unknown key by the longest
                # known metric name inside it, so ``val_zf_delta_hfen``
                # resolves through ``hfen`` to "lower is better". A RAW
                # ``model - baseline`` is correct under exactly that rule for
                # both directions at once (a better PSNR raises the delta, a
                # better HFEN lowers it); an oriented "gain" would be read
                # backwards for every lower-is-better metric by every one of
                # those consumers.
                for k, v in zf_metrics.items():
                    if k in unscaled_metrics:
                        metrics[f"val_zf_delta_{k}"] = unscaled_metrics[k] - v

            except Exception as e:
                self.logging_service.log_error(f"[Validation Metrics] Computer failed: {e}")
                # Log what we tried
                self.logging_service.log_info(
                    f"[Validation Metrics] Input: {list(hr_fakes_for_metrics.shape)} (hr_fakes) | "
                    f"Domain: {domain} | Transform: {transform} | "
                    f"Final Shape: {list(pred_transformed.shape)} | Error: {e!s}"
                )
                raise

        # Federated cross-contrast metadata published by the M4Raw dataset
        # (see m4raw_dataset.py, cross-contrast branch). May be absent for
        # single-contrast datasets and for older cached batches — the logger
        # treats ``None`` as "render the full stack".
        _fed_start = read_batch_field(batch_data, "federated_target_channel_start")

        # Per-case identity for `per_call_metrics.csv`. The cascade feeds that
        # sink once per (batch, rung), so without these columns a 45-batch x
        # 3-rung sweep wrote 135 rows under three `case_id` labels -- every
        # number present, nothing saying which volume or which acceleration
        # produced it. Built here because this is the only frame where the rung
        # (`cascade_level`, `timestep_used`, `heldout`) and the batch identity
        # (`batch_data`) are both in scope.
        _case_context: dict[str, Any] = {
            "acceleration_level": (None if cascade_level is None else float(cascade_level)),
            "acceleration_realized": acceleration_realized,
            "timestep": timestep_used,
            "heldout": bool(heldout),
            "batch_index": int(batch_idx),
        }
        # Local import for the same reason `feed_report_case_recorder` is
        # imported locally further down: `metrics_mixin` reaches back into this
        # package, so a module-level edge closes a cycle.
        from .mixins.metrics_mixin import summarize_batch_identity

        _case_context.update(summarize_batch_identity(batch_data))

        # `emit_reports=False` exists for readouts that SCORE a second
        # prediction from the same batch -- the t=0 pre-DC probe (#1682). This
        # one call is not only the TensorBoard writer: it also carries
        # `context=_case_context` into `feed_report_case_recorder`
        # (`_log_validation_images_to_tensorboard`, further down), so a second
        # invocation would add a report row per batch AND, with
        # `cascade_level=None`, take the legacy single-prefix path that
        # overwrote the cascade renders -- the "experiment_11 fake images
        # doubled" regression this method's own docstring records. The probe
        # wants the numbers, not a second set of images keyed the same way.
        #
        # The third argument is `log_inputs`, NOT the raw `input_batch`, and
        # `inputs_are_image` is its OWN domain flag (#1684): inputs must cross
        # the same decompress_for_view + denom_scale + domain seam as the
        # predictions and targets beside them. Passing `input_batch` here is a
        # pinned regression -- it is one of #1684's planted violations.
        if emit_reports:
            self._log_validation_images_to_tensorboard(
                log_preds,
                log_targets,
                log_inputs,
                metrics,
                batch_idx=batch_idx,
                is_image_domain=is_preds_image,
                inputs_are_image=inputs_are_image,
                cascade_level=cascade_level,
                federated_target_channel_start=_fed_start,
                context=_case_context,
            )

        # Add timestep stats
        metrics["val_timestep_mean"] = timestep.float().mean().detach().item()
        metrics["val_pred_mean"] = (
            hr_fakes_for_metrics.abs().mean().detach().item()
            if torch.is_complex(hr_fakes_for_metrics)
            else hr_fakes_for_metrics.mean().detach().item()
        )
        metrics["val_target_mean"] = (
            target_for_metrics.abs().mean().detach().item()
            if torch.is_complex(target_for_metrics)
            else target_for_metrics.mean().detach().item()
        )

        return self._convert_metrics_to_floats(metrics)

    def _t0_pre_dc_probe_metrics(
        self,
        target_batch: torch.Tensor,
        input_batch: torch.Tensor,
        batch_data: Any,
        scale_factor: torch.Tensor,
        batch_idx: int,
    ) -> dict[str, float]:
        """Score the terminal (t=0) rung pre-DC, as ``val_t0_predc_*``.

        The sampler never evaluates the model at t=0 under ``dc_method: hard``
        (the reveal at t=1 is total and the last step is inert), and after DC at
        t=0 every bin is acquired so the output IS the input. Both halves of the
        terminal rung -- the one gradient-carrying loss and the only denoising
        the arm is asked to learn -- are therefore invisible to every existing
        readout. This makes them a number.

        Returns ``{}`` when the generator exposes no pre-DC proposal. That
        decision is read off the model class, so every rank reaches the same
        answer: ``train.py:_all_reduce_val_metrics`` packs ``sorted(keys)`` into
        a tensor and all-reduces POSITIONALLY, and a rank-dependent key set
        reduces mismatched-length tensors -- a hang, or one metric's value
        landing under another metric's name. Never an error.
        """
        # Unwrap DataParallel / DistributedDataParallel: the capability lives on
        # the real module, and a wrapper would report "no pre-DC" on every rank
        # -- consistent, so no hang, but a silently empty readout. Calling the
        # unwrapped module also keeps the probe out of DDP's reducer
        # bookkeeping; it is a no_grad eval forward with nothing to synchronise.
        generator = (
            self.generator_model.module
            if hasattr(self.generator_model, "module")
            else self.generator_model
        )
        if not generator_exposes_pre_dc(generator):
            return {}

        # At t=0 every coefficient is revealed, so the mask the network sees is
        # all-ones. Built here rather than borrowed from a cascade rung: the
        # rungs carry their own accelerations and none of them is t=0 -- which is
        # the whole reason this probe exists.
        mask = torch.ones_like(input_batch[:, :1])
        forward_kwargs = self._build_generator_kwargs(
            is_cold_diffusion=self._is_cold_diffusion(),
            is_latent_diffusion=False,
            input_batch=input_batch,
            target_batch=target_batch,
            batch_data=batch_data,
            mask=mask,
        )

        def _score(prediction: torch.Tensor, timesteps: torch.Tensor):
            # The one metrics seam, not a second implementation of scoring
            # (non-negotiable 17). ``emit_reports=False`` because that seam also
            # feeds the report case recorder and the TensorBoard renders; a
            # second call would add a row per batch and, at
            # ``cascade_level=None``, overwrite the cascade images.
            return self._compute_validation_metrics(
                prediction,
                target_batch,
                input_batch,
                timesteps,
                batch_data,
                scale_factor,
                batch_idx=batch_idx,
                emit_reports=False,
            )

        return run_t0_predc_probe(
            generator=generator,
            model_input=input_batch,
            forward_kwargs=forward_kwargs,
            score=_score,
        )

    #: Fraction of prediction elements that may sit above the target's peak
    #: magnitude before validation says so out loud. A diverged or untrained
    #: sampler pushes this to 1.0 — every pixel brighter than the brightest
    #: pixel of the ground truth, which renders as a featureless white field.
    _PRED_SCALE_WARN_FRACTION: float = 0.5

    #: Metrics the zero-filled baseline is graded on. INTERSECTED with the
    #: arm's own configured specs, never added to them: a ``val_zf_hfen`` with
    #: no ``val_hfen`` beside it has nothing to be compared against, and the
    #: baseline exists only to answer "did the model beat the measurement on
    #: the axes this arm is already graded on?". Deliberately two cheap
    #: full-reference metrics -- validation already dominates wall clock on
    #: this paradigm, and a second pass over an arm's whole metric set would
    #: mean a second LPIPS/FID-class evaluation per rung per batch.
    _ZF_BASELINE_METRICS: tuple[str, ...] = ("psnr", "hfen")

    #: The k-space the model was actually handed, stashed by
    #: ``_generate_validation_prediction`` and popped by
    #: ``_compute_validation_metrics``. Declared on the CLASS (rather than
    #: reached through ``getattr(self, ..., None)``) so "no measurement
    #: recorded" is a declared state and not an absent-attribute fallback.
    _zf_measurement: torch.Tensor | None = None

    #: Per-rung keys the validation ensemble writes (``val_<name>``, then
    #: ``_<R>x`` by the cascade and ``_mean`` by ``_stamp_accel_mean``).
    _ENSEMBLE_METRICS: tuple[str, ...] = ("ensemble_std_mean", "empirical_coverage")

    #: The N reverse samples (k-space, model units, ``[N, B, C, H, W]``) that
    #: ``_generate_validation_prediction`` draws when
    #: ``validation.sampling.ensemble_samples > 1``, popped by
    #: ``_compute_validation_metrics``. Class-declared for the same reason as
    #: ``_zf_measurement``: "no ensemble" is a declared state, and a stack can
    #: never outlive its rung.
    _ensemble_members: torch.Tensor | None = None

    #: The resolved ensemble knobs, stamped once by
    #: ``_resolve_validation_ensemble`` (non-negotiable 8): the declared N and k
    #: beside the sigma and seed the sampler actually runs with.
    ensemble_provenance: dict[str, Any] | None = None

    #: The registered metrics that read the measurement context; ``only=`` is an
    #: intersection with the arm's configured set, so an arm opts in by listing
    #: them in ``metrics.compute``.
    _MEASUREMENT_AWARE_METRICS: tuple[str, ...] = ("nse_hall", "ndcr")

    def _measurement_aware_metrics(
        self,
        pred_kspace: torch.Tensor,
        target_kspace: torch.Tensor,
        measured_kspace: torch.Tensor,
        computer: Any,
    ) -> dict[str, float]:
        """``nse_hall`` / ``ndcr`` from the stashed masked measurement.

        The registry has carried these two since the trust-functional work, but
        nothing on the training-validation path ever built the
        :class:`~spectramr.core.metrics.context.MetricContext` they need, so a
        config that listed them got ``nan``. The mask is the measurement's own
        support (any coil non-zero), so it follows the rung; the images are the
        complex SENSE-adjoint reconstructions (the null-space projector needs
        the phase, which the arm's magnitude transform discards).
        """
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.nr_consistency import _as_complex_image
        from spectramr.infrastructure.physics.fft_ops import sense_adjoint

        wanted = tuple(self._MEASUREMENT_AWARE_METRICS)
        measured_c = _as_complex_image(measured_kspace)
        mask = (measured_c.abs() > 0).any(dim=1, keepdim=True).to(torch.float32)
        smaps = self._select_batch_compatible_smaps(pred_kspace.shape[0])
        pred_img = sense_adjoint(_as_complex_image(pred_kspace), smaps=smaps)
        target_img = sense_adjoint(_as_complex_image(target_kspace), smaps=smaps)
        context = MetricContext(mask=mask, y_kspace=measured_c, coil_maps=smaps)
        return dict(computer.compute(pred_img, target_img, only=wanted, context=context))

    @staticmethod
    def _measure_prediction_scale(
        pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
    ) -> dict[str, float]:
        """Describe where the prediction's magnitudes sit relative to the target's.

        Replaces the truth-leaking clamp that used to bound ``pred`` by
        ``1.1 * target.abs().max()``. Nothing here modifies a tensor — the
        point is to *report* a scale mismatch so a diverged model shows up in
        the metrics and the log instead of being flattened into a plausible
        looking constant.

        ``pred_above_target_fraction`` is the discriminating number: it is the
        share of prediction elements strictly brighter than the brightest
        element of the target. At ``1.0`` every pixel exceeds the target's
        peak, which is exactly the uniformly-white render observed for
        experiment_11_attention_none on 2026-08-18.

        Magnitudes are compared via ``abs()`` for both real-stacked and
        complex tensors, because that is the quantity the render path
        (``kspace_to_image`` → coil RSS) and the image metrics actually see.
        """
        pred_mag = pred.detach().abs()
        target_mag = target.detach().abs()

        # Compare in the tensors' own dtypes, then widen the 0-dim scalars for
        # the arithmetic below. Validation may run the sampler under autocast,
        # and in fp16 ``eps`` underflows to exactly zero -- the guard on the
        # ratio would divide by 0 and report ``inf`` for a silent target
        # instead of a large finite number. Widening a scalar is free;
        # widening ``pred_mag`` would not be.
        above_fraction = (pred_mag > target_mag.max()).sum() / pred_mag.numel()

        pred_abs_max = pred_mag.max().float()
        pred_abs_min = pred_mag.min().float()
        target_abs_max = target_mag.max().float()

        # One device sync, not five: validation runs per cascade level per
        # batch, and this is a diagnostic, not a reason to stall the queue.
        keys = (
            "pred_abs_max",
            "pred_abs_min",
            "target_abs_max",
            "pred_target_scale_ratio",
            "pred_above_target_fraction",
        )
        values = torch.stack(
            [
                pred_abs_max,
                pred_abs_min,
                target_abs_max,
                pred_abs_max / target_abs_max.clamp(min=eps),
                above_fraction.float(),
            ]
        ).tolist()
        return dict(zip(keys, values, strict=True))

    @staticmethod
    def _resolve_federated_target_start(
        marker: "torch.Tensor | int | None",
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> int | None:
        """Validate and normalize the federated cross-contrast split marker.

        The M4Raw cross-contrast dataset publishes
        ``federated_target_channel_start`` on the subject (see
        :class:`spectramr.data.datasets.m4raw_dataset.M4RawRepetitionDataset`,
        cross-contrast branch). When that boundary is forwarded into the
        validation logger, the visualization can slice the prediction /
        target tensors at the split so the RSS only mixes coils within
        one contrast — otherwise the saved PNG superimposes T1 and T2/FLAIR
        anatomy, which is exactly the experiment_11 mosaic doubled-target
        finding from 2026-05-13.

        Returns ``None`` when the marker is absent, malformed, mixes
        federated and single-contrast samples within a batch, or points
        outside the tensor shape. ``None`` means "render the full stack" —
        the historical behaviour, preserved for backwards compatibility.
        """
        if marker is None:
            return None
        # Per-sample scalar arrives as a 0-d tensor; a stacked batch arrives
        # as a 1-d tensor. All entries should agree because the dataset
        # emits the same boundary for every cross-contrast sample. If they
        # disagree the batch mixed two dataset modes, and a single slice
        # can't honour both — fall back to rendering the full stack.
        if isinstance(marker, torch.Tensor):
            vals = marker.flatten().tolist()
            if not vals:
                return None
            start = int(vals[0])
            if any(int(v) != start for v in vals[1:]):
                return None
        else:
            start = int(marker)
        # Validate against tensor shapes — a stale or mis-collated boundary
        # would otherwise produce an empty slice or silent index errors.
        if start <= 0 or start >= predictions.shape[1] or start >= targets.shape[1]:
            return None
        return start

    def _validation_image_step(self) -> int:
        """The step label for saved validation renders: the TRAINING iteration.

        Never ``validation_step_count`` (issue #585). That counter increments once per
        cascade LEVEL, so the previous ``step = validation_step_count; if step == 0:
        step = resolve_loop_iteration(self)`` told the truth only for whichever level
        happened to observe the counter at 0. One validation event at iteration 3000
        wrote R2x as ``step003000`` but R8x/R32x as ``step000001`` / ``step000002``;
        and because the counter never returns to 0, every subsequent event mislabelled
        all three.

        Anything that sorts these renders by step then reads the later cascade levels
        as the OLDEST files on disk -- the #425/#427 failure class, and the reason the
        R32x renders looked like an untrained snapshot when they were iteration-3000
        outputs.
        """
        return int(resolve_loop_iteration(self))

    def _log_validation_images_to_tensorboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        metrics: dict,
        batch_idx: int = 0,
        is_image_domain: bool = False,
        inputs_are_image: bool = False,
        cascade_level: float | None = None,
        federated_target_channel_start: "torch.Tensor | int | None" = None,
        context: "Mapping[str, Any] | None" = None,
    ) -> None:
        """Log validation predictions and targets to TensorBoard as images.

        Args:
            cascade_level: Acceleration factor for this validation pass.
                Encoded into the saved-image prefix so multiple cascade
                levels (e.g. 2x / 8x / 32x) land in distinct files instead
                of overwriting one another. ``None`` preserves the legacy
                "validation_*.png" naming.
        """
        # DEBUG: Log entry to this method
        if not hasattr(self, "logging_service") or self.logging_service is None:
            return

        try:
            # DEBUG: Log method entry
            self.logging_service.log_info(
                "[DiffusionValidation] _log_validation_images_to_tensorboard ENTRY"
            )

            step = self._validation_image_step()

            # ✅ SSOT: Direct access to logging config (trust schema defaults)
            logging_config = self.config.logging

            # DEBUG: Log config values
            if logging_config:
                self.logging_service.log_info(
                    f"[DiffusionValidation] log_validation_images = {logging_config.images.log_validation}"
                )

            if logging_config and not logging_config.images.log_validation:
                self.logging_service.log_info(
                    "[DiffusionValidation] EARLY RETURN: log_validation_images is False"
                )
                return

            # ✅ SSOT: Extract max_images_per_batch from config schema (default: 4)
            max_images = logging_config.images.max_per_batch or 4

            def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
                """_ensure_4d.

                Args:
                    t (torch.Tensor): Description.
                Returns:
                    torch.Tensor: Description.
                """
                if t.dim() == 2:
                    return t.unsqueeze(0).unsqueeze(0)
                if t.dim() == 3:
                    return t.unsqueeze(1)
                return t

            def to_magnitude(t):
                """Reduce a possibly-multi-channel image-domain tensor to a
                single-channel magnitude.

                Channel-RSS only — we do NOT pair adjacent channels as
                (R, I) here because paired-modality data (ULF/HF, T1/T2)
                also has even channel count and the previous heuristic
                produced the experiment_11 doubled-brain regression and
                the "ULF doubled and odd" symptom. For genuine
                real-stacked complex tensors, channel-RSS gives the same
                result as per-coil-magnitude+RSS (see proof in
                ``debug_snapshot._render_image_preview``).
                """
                if torch.is_complex(t):
                    return torch.abs(t)
                if t.dim() >= 4 and t.shape[1] > 1:
                    return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
                return t

            def _reduce_to_single_channel(t):
                """Ensure tensor has only 1 channel for image saving (PIL limitation)."""
                if t.dim() < 4:
                    return t
                # t shape: (B, C, H, W)
                if t.shape[1] == 1:
                    return t  # Already single channel
                elif t.shape[1] == 3:
                    return t  # RGB, supported
                else:
                    # Multi-channel: RSS combine (root-sum-of-squares)
                    # This is correct for multi-coil magnitude images
                    # (B, C, H, W) -> (B, 1, H, W)
                    return torch.sqrt(torch.sum(t**2, dim=1, keepdim=True) + 1e-8)

            def kspace_to_image(ksp, already_image=False):
                """Convert k-space to image domain for visualization.

                Handles multi-coil real-stacked format [B, 2*C_coils, H, W]
                by pairing channels as complex, applying per-coil iFFT, then RSS.
                """
                if already_image:
                    # F14 (2026-05-17 round 7): even on the
                    # ``already_image=True`` branch, refuse to render a
                    # k-space-shaped tensor as a magnitude image. The
                    # caller may have misclassified the domain (see
                    # ``_log_validation_images_to_tensorboard`` upstream);
                    # silently letting a complex / even-channel real
                    # tensor through produces the DC-spike "white spot
                    # in the middle" artifact on saved fake_images PNGs.
                    looks_like_kspace = torch.is_complex(ksp) or (
                        ksp.dim() in (4, 5) and ksp.shape[1] >= 2 and ksp.shape[1] % 2 == 0
                    )
                    if looks_like_kspace:
                        raise ValueError(
                            f"[DiffusionValidation.kspace_to_image] "
                            f"already_image=True but tensor shape="
                            f"{tuple(ksp.shape)} (dtype={ksp.dtype}, "
                            f"is_complex={torch.is_complex(ksp)}) looks "
                            f"like k-space (complex or even-channel "
                            f"real-stacked). Rendering this as magnitude "
                            f"produces the DC-spike artifact. Fix the "
                            f"upstream domain inference in "
                            f"``_log_validation_images_to_tensorboard``. "
                            f"See TODO/audit/smoke_audit_20260516.md §F14."
                        )
                    return to_magnitude(ksp)
                try:
                    from spectramr.infrastructure.physics.fft_ops import ifft2c

                    # [FIX] Slice to target contrast for federated multi-contrast data
                    ksp = self._slice_to_target_contrast_single(ksp)
                    # NOTE: inverse log-scaling is already applied upstream in
                    # _compute_validation_metrics before data reaches this function.

                    if torch.is_complex(ksp):
                        img = ifft2c(ksp)
                        return torch.abs(img)

                    # Multi-coil real-stacked: [B, 2*C_coils, H, W]
                    # Pair channels as complex, per-coil iFFT, then RSS
                    if ksp.dim() == 4 and ksp.shape[1] >= 2 and ksp.shape[1] % 2 == 0:
                        c = ksp.shape[1]
                        coil_images = []
                        for i in range(0, c, 2):
                            ksp_complex = torch.complex(ksp[:, i], ksp[:, i + 1])
                            img_complex = ifft2c(ksp_complex)
                            coil_images.append(torch.abs(img_complex).unsqueeze(1))
                        # RSS combine all coils → [B, 1, H, W]
                        coil_stack = torch.cat(coil_images, dim=1)
                        rss = torch.sqrt(torch.sum(coil_stack**2, dim=1, keepdim=True) + 1e-8)
                        return rss

                    return ksp
                except Exception as e:
                    # F14 (2026-05-17 round 7): re-raise instead of
                    # falling back to ``to_magnitude(ksp)``. The fallback
                    # rendered raw k-space magnitude (DC spike at center)
                    # as the "image", masking the actual IFFT failure and
                    # producing the misleading "white spot in the middle"
                    # PNGs. CLAUDE.md #9: silent fallbacks forbidden.
                    self.logging_service.log_error(
                        f"[DiffusionValidation.kspace_to_image] IFFT "
                        f"failed: {e}. Refusing silent magnitude "
                        f"fallback (DC-spike artifact). shape="
                        f"{ksp.shape}, dtype={ksp.dtype}. See "
                        f"TODO/audit/smoke_audit_20260516.md §F14."
                    )
                    raise

            images_dict = {}

            # Defensive complex→magnitude conversion ONLY for image-domain
            # data. For k-space tensors we must keep complex so kspace_to_image
            # can apply a centred IFFT; otherwise an .abs() on complex k-space
            # strips phase and the downstream "real-stacked even-channel"
            # branch pairs magnitudes as real/imag → garbage reconstruction
            # (the experiment_11 doubled-brain + k-space-superposition bug).
            #
            # F5a (2026-05-21 smoke audit): even within ``is_image_domain``,
            # refuse to ``.abs()`` a multi-coil complex tensor. For diffusion
            # YAMLs with ``data.coil_processing_mode='none'`` the target
            # tensor reaches this branch as complex multi-coil k-space (the
            # data builder didn't combine coils), and the unconditional
            # ``.abs()`` here was stripping phase. The Hermitian-symmetric
            # magnitude that resulted produced centro-symmetric (doubled-
            # brain) validation images in 80 / 103 experiments — see
            # TODO/audit/smoke_audit_20260521.md §F5a.
            #
            # Guard: ``.abs()`` only when the tensor is *plausibly* a single-
            # coil complex image (channel count ≤ 2). Multi-coil complex
            # falls through to ``kspace_to_image`` which pairs (R, I) as
            # complex, per-coil IFFTs, and RSS-combines — the right path.
            if is_image_domain:
                if torch.is_complex(predictions):
                    if predictions.dim() < 4 or predictions.shape[1] <= 2:
                        predictions = predictions.abs()
                    else:
                        # Multi-coil complex image — leave complex; the
                        # mixin's kspace_to_image handles per-coil
                        # magnitude + RSS without the phase-strip bug.
                        self.logging_service.log_info(
                            f"[DiffusionValidation] F5a: preserving "
                            f"complex multi-coil predictions "
                            f"(shape={tuple(predictions.shape)}); skipping "
                            f"the unconditional .abs() that stripped phase."
                        )
                if torch.is_complex(targets):
                    if targets.dim() < 4 or targets.shape[1] <= 2:
                        targets = targets.abs()
                    else:
                        self.logging_service.log_info(
                            f"[DiffusionValidation] F5a: preserving "
                            f"complex multi-coil targets "
                            f"(shape={tuple(targets.shape)}); skipping "
                            f"the unconditional .abs() that stripped phase."
                        )

            # Federated cross-contrast slice: when the dataset stacked
            # [source_contrast, target_contrast] along the channel axis (M4Raw
            # in cross-contrast mode emits a 16-channel tensor = 8 source + 8
            # target), the user expects to *see* the target-contrast
            # reconstruction, not a T1+T2 RSS-mix. The dataset publishes the
            # boundary as ``federated_target_channel_start`` in the subject
            # metadata; the validation step forwards it here so we slice both
            # prediction and target tensors before kspace_to_image — the RSS
            # then only mixes coils within one contrast.
            fed_start = self._resolve_federated_target_start(
                federated_target_channel_start,
                predictions,
                targets,
            )
            if fed_start is not None:
                self.logging_service.log_info(
                    f"[DiffusionValidation] Federated split active — slicing "
                    f"channels [{fed_start}:] of both prediction and target "
                    f"tensors (was shape {tuple(predictions.shape)})."
                )
                predictions = predictions[:, fed_start:, ...]
                targets = targets[:, fed_start:, ...]

            pred_img = kspace_to_image(predictions, already_image=is_image_domain)
            # kspace_to_image already returns magnitude — do NOT re-apply to_magnitude
            # as it would treat even-channel images as complex pairs (RSS combine)
            pred_mag = _ensure_4d(pred_img)
            pred_mag = _reduce_to_single_channel(pred_mag)
            images_dict["val/predictions"] = pred_mag[:max_images]

            target_img = kspace_to_image(targets, already_image=is_image_domain)
            # kspace_to_image already returns magnitude — do NOT re-apply to_magnitude
            target_mag = _ensure_4d(target_img)
            target_mag = _reduce_to_single_channel(target_mag)
            images_dict["val/targets"] = target_mag[:max_images]

            # ✅ SSOT: Extract log_difference_images and log_input_images from config schema
            # log_difference_images defaults to True, log_input_images defaults to False
            if logging_config and logging_config.images.log_difference:
                diff = _ensure_4d(torch.abs(pred_mag - target_mag))
                diff = _reduce_to_single_channel(diff)
                images_dict["val/difference"] = diff[:max_images]

            # ``inputs`` has exactly two consumers -- the ``val/inputs`` panel
            # and the report-case recorder -- and they used to convert it
            # independently, both wrong the same two ways (#1684). Render it
            # ONCE, and only when one of them will actually read it: this whole
            # block sits inside the method-wide ``except``, so an unconditional
            # render would let an inputs-only failure take the predictions,
            # targets and difference panels down with it on arms that never
            # asked for input rendering.
            recorder = getattr(self, "_report_case_recorder", None)
            sink = getattr(self, "_per_case_metric_sink", None)
            _feeds_recorder = (recorder is not None and getattr(recorder, "enabled", False)) or (
                sink is not None and getattr(sink, "enabled", False)
            )
            _wants_input_panel = bool(logging_config and logging_config.images.log_input)

            # The image branch goes straight to ``to_magnitude``, NOT through
            # ``kspace_to_image(already_image=True)``. That branch's F14 guard
            # exists to second-guess the *predictions* flag, which is re-derived
            # from a tensor comparison after ``_apply_metric_transforms`` -- its
            # own error text says "fix the upstream domain inference". Here the
            # SSOT POSITIVELY determined the dataloader's domain, so there is
            # nothing to second-guess, and F14's shape heuristic cannot tell a
            # 2-channel real-stacked complex *image* from k-space: it would
            # refuse ``sqrt(re^2 + im^2)``, which IS the correct magnitude, and
            # the swallowing ``except`` above would turn that refusal into "no
            # validation images at all". A branch chosen up front by the
            # authoritative flag is not an error path degrading to a default.
            input_mag = None
            if _wants_input_panel or _feeds_recorder:
                input_img = to_magnitude(inputs) if inputs_are_image else kspace_to_image(inputs)
                input_mag = _reduce_to_single_channel(_ensure_4d(input_img))

            if _wants_input_panel and input_mag is not None:
                images_dict["val/inputs"] = input_mag[:max_images]

            self.logging_service.log_images_batch(images_dict, step, max_images=max_images)

            # Feed the report-case recorder (image-domain magnitudes already on
            # CPU here) — the diffusion override does not call super(), so the
            # seam must be duplicated to avoid an unfed (facade) recorder.
            if _feeds_recorder:
                from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
                    feed_report_case_recorder,
                )

                feed_report_case_recorder(
                    recorder,
                    predictions=pred_mag,
                    targets=target_mag,
                    inputs=input_mag,
                    metrics=metrics,
                    step=step,
                    sink=sink,
                    # Tag the case with its acceleration rung. Cascading
                    # validation feeds this seam once per level at the SAME
                    # iteration, so without it every rung lands under one
                    # ``val_step<N>`` label and per-case analysis silently
                    # compares R=2 against R=32 (see ``report_case_id``).
                    cascade_level=cascade_level,
                    # ...and the rest of the row's identity: acceleration,
                    # timestep, heldout flag, contrast, file id. Forwarded
                    # verbatim -- this method renders images and has no business
                    # interpreting a dataset's vocabulary.
                    context=context,
                )

            self.logging_service.log_debug(
                f"Logged validation images to TensorBoard at step {step}"
            )

            # Also save images to filesystem if metrics service available
            # Only save images for the first batch to prevent filling disk (73k+ images issue)
            if batch_idx == 0:
                try:
                    # DEBUG: Log before checking metrics_service
                    has_metrics_service = hasattr(self, "metrics_service")
                    metrics_service_not_none = (
                        self.metrics_service is not None if has_metrics_service else False
                    )

                    self.logging_service.log_info(
                        f"[DiffusionValidation] Before save_images: has_metrics_service={has_metrics_service}, "
                        f"metrics_service_not_none={metrics_service_not_none}"
                    )

                    if hasattr(self, "metrics_service") and self.metrics_service is not None:
                        epoch = getattr(self, "current_epoch", 0)
                        self.logging_service.log_info(
                            f"[DiffusionValidation] Saving images: "
                            f"domain={'image' if is_image_domain else 'kspace'}, "
                            f"pred=[{pred_mag.min().item():.4f}, {pred_mag.max().item():.4f}], "
                            f"target=[{target_mag.min().item():.4f}, {target_mag.max().item():.4f}], "
                            f"pred_shape={list(pred_mag.shape)}, "
                            f"epoch={epoch}, step={step}"
                        )
                        # Tag the saved-image prefix with the cascade
                        # acceleration level so 2x / 8x / 32x renders are
                        # all preserved on disk instead of the worst-case
                        # level overwriting earlier writes.
                        if cascade_level is not None:
                            level_tag = (
                                f"{cascade_level:.0f}x"
                                if float(cascade_level).is_integer()
                                else f"{cascade_level:g}x"
                            )
                            save_prefix = f"validation_R{level_tag}"
                        else:
                            save_prefix = "validation"
                        real_paths, fake_paths = self.metrics_service.save_images_batch(
                            real_images=target_mag,
                            fake_images=pred_mag,
                            prefix=save_prefix,
                            epoch=epoch,
                            step=step,
                            max_images=max_images,
                        )
                        if real_paths or fake_paths:
                            self.logging_service.log_info(
                                f"[DiffusionValidation] Saved {len(real_paths)} real + {len(fake_paths)} fake images"
                            )
                        else:
                            # No paths returned likely means save_images=False in config (intended behavior)
                            pass
                    else:
                        # ✅ FAIL LOUD if metrics_service is expected but not available
                        missing_attr = not hasattr(self, "metrics_service")
                        if missing_attr:
                            self.logging_service.log_error(
                                "[DiffusionValidation] CRITICAL: metrics_service attribute missing on strategy. "
                                "This indicates a training infrastructure issue - metrics_service must be passed from bootstrap. "
                                "Validation images will NOT be saved."
                            )
                        else:
                            self.logging_service.log_error(
                                "[DiffusionValidation] CRITICAL: metrics_service is None on strategy. "
                                "DI resolution failed or bootstrap did not register IMetricsService. "
                                "Validation images will NOT be saved."
                            )
                except Exception as e:
                    self.logging_service.log_error(
                        f"[DiffusionValidation] CRITICAL: Failed to save validation images: {e}"
                    )

            # Increment step counter (defensive with getattr)
            self.validation_step_count = getattr(self, "validation_step_count", 0) + 1

        except Exception as e:
            if self.logging_service.logger.isEnabledFor(logging.WARNING):
                self.logging_service.log_warning(f"Failed to log validation images: {e!s}")

    def get_noise_prediction(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Extract noise prediction from model output.

        For standard diffusion models, the model directly predicts the noise.
        This can be overridden in subclasses for different prediction formats.

        Args:
            model_output: Raw model output from the diffusion model
            x_t: Noisy input at timestep t
            timesteps: Timestep tensor

        Returns:
            Predicted noise tensor

        """
        # For standard diffusion, model output is predicted noise
        return model_output


class _CrossModalConditionEncoder(torch.nn.Module):
    """Maps a source-modality image to a spatial conditioning embedding.

    A small lazy conv stem: the input channel count is bound on the first
    forward (so it adapts to whatever the source modality provides), then the
    body produces an ``embedding_dim``-channel feature map spatially aligned
    with the input. The strategy feeds this ``z_cond`` to the diffusion
    generator's ``forward`` (when the generator accepts a condition kwarg).

    It is owned by :class:`XDiffusionTrainingStrategy` as a plain attribute
    (the *strategy* is not an ``nn.Module``); the strategy registers this
    module's parameters on the generator optimizer so they train.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        # LazyConv2d binds in_channels on first forward — robust to whatever
        # channel count the source modality arrives with.
        self.stem = torch.nn.LazyConv2d(self.embedding_dim, kernel_size=3, padding=1)
        self.act = torch.nn.SiLU()
        self.proj = torch.nn.Conv2d(self.embedding_dim, self.embedding_dim, kernel_size=1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        return self.proj(self.act(self.stem(source)))


class XDiffusionTrainingStrategy(BaseTrainingStrategy):
    """X-Diffusion training strategy for cross-modal multi-contrast synthesis.

    Extends vanilla diffusion to handle multi-modal cross-contrast generation.
    Learns to synthesize diverse MRI contrast images from conditioned source modality
    (e.g., T1-weighted → T2-weighted, FLAIR, PD) via score-based diffusion.\n    ## Extension Over Standard Diffusion

    **Standard Diffusion**: Single modality, unconditional generation (x_0)
    **X-Diffusion**: Multi-contrast, conditional generation

    Input: T1-weighted or any source modality
    Output: Target contrasts (T2, FLAIR, PD, etc.) via diffusion conditioning

    ## Core Concepts

    **Cross-Modal Generation**:
    - Condition network on source modality features
    - Diffusion model generates diverse target contrasts
    - Accounts for contrast-specific physics (T1/T2 relaxation, field effects)

    **Multi-Contrast Architecture**:
    - **Condition Encoder**: Maps source contrast → embedding (T1 → z_cond)
    - **Diffusion UNet**: Conditioned on z_cond for all timesteps
    - **Output**: Target contrast generation (T2, FLAIR, PD, etc.)

    ## Training Process (as currently implemented)

    1. **Sample Timestep**: t ~ [1, T)
    2. **Sample Noise**: ε ~ N(0, I)
    3. **Forward Diffusion**: x_t = √ᾱ_t·x_target + √(1-ᾱ_t)·ε  (cosine ᾱ_t)
    4. **Model**: UNet(x_t, t) → x̂_0  (t passed only if forward accepts it)
    5. **Loss**: L(x̂_0, x_target)  (config losses; MSE fallback)
    6. **Backward**: gradient flows to the UNet

    .. note::
       Cross-modal **condition encoding** (a learned ``z_cond`` head on the
       source modality, step "Prepare Condition" / "UNet(x_t, t, z_cond)") is
       wired when the optional ``cross_modal`` block is present under
       ``training.diffusion`` (see Configuration). When that block is absent the
       strategy degrades to a single-modality denoiser (the source modality
       enters only as the diffusion ``x_0`` target) — and crucially does NOT
       advertise any cross-modal knob it has not read. When the block IS present,
       every advertised knob is read, validated (unknown ``condition_encoder``
       raises, per pitfall #9), the encoder's parameters are registered on the
       generator optimizer so they actually train, and the resolved values are
       stamped into ``self.cross_modal_provenance`` (pitfall #15).

    ## Configuration

    - `training.training_mode`: 'cross_modal_diffusion' or 'diffusion'
    - `training.diffusion.timesteps`: Diffusion steps (1000 typical) — READ
    - `training.diffusion.cross_modal`: optional dict enabling source-modality
      conditioning. Recognized keys (each read + validated + stamped):

      * ``enabled`` (bool): turn cross-modal conditioning on.
      * ``source_modality`` (str): label of the conditioning modality (stamped).
      * ``target_modalities`` (list[str]): target contrast labels (stamped).
      * ``condition_embedding_dim`` (int > 0): size of the ``z_cond`` code.
      * ``num_contrasts`` (int > 0): number of target modalities (stamped).
      * ``condition_encoder`` (str): encoder type; must be ``"conv"`` — any
        other value RAISES at construction (no silent fallback, pitfall #9).

      The condition embedding is fed to the generator's ``forward`` only if it
      accepts a ``cond`` / ``context`` / ``condition`` kwarg; if cross-modal is
      enabled but the generator accepts none of them, construction RAISES
      rather than silently dropping the condition (pitfall #9).

    ## 3D Generation Support

    **Volumetric Extension**:
    - Condition on 2D slice or 3D volume slices
    - Generate full 3D volume per modality
    - Can handle anisotropic (thick slice) data

    **Configuration**:
    - `training.diffusion.volumetric`: true for 3D
    - `training.diffusion.volume_depth`: Z-dimension (40-80 mm)
    - `training.diffusion.slice_thickness`: Acquisition thickness

    ## Loss Components

    1. **Diffusion Loss**: Score matching on target contrast
       - ||ε_θ(x_t, t, z_cond) - ε||²
       - Denoising objective per modality

    2. **Modality Consistency**: Matching modality-specific statistics
       - Histogram/intensity distribution match
       - Tissue-specific contrast ranges

    3. **Condition Matching**: Ensure condition is properly used
       - Optional: Discriminator on (condition, generated) pairs

    4. **Physics Constraints** (optional):
       - Tissue parameter bounds per modality
       - T1/T2 ratios across contrasts

    ## Key Features

    ✅ **Multi-Modal**: Generate any contrast from any conditioner
    ✅ **Flexible**: Works with 2D slices or 3D volumes
    ✅ **Robust**: Conditional score matching handles modality variation
    ✅ **Diverse**: Can generate multiple mode hypotheses via sampling variance
    ✅ **Physics-Aware**: Optional physics constraints per modality

    ## Inference

    1. **Input**: Source modality (T1-weighted volume)
    2. **Encode Condition**: z_cond = Encoder(T1)
    3. **Sample Noise**: x_T ~ N(0, I)
    4. **Reverse Diffusion**: x_{t-1} ~ p(x_{t-1}|x_t, z_cond)
       - For t = T, T-1, ..., 1, 0:
       - μ = (x_t - σ_t²·ε_θ(x_t, t, z_cond)) / √ᾱ_t
       - x_{t-1} = μ + σ_t·z
    5. **Output**: Generated target contrast (T2, FLAIR, PD)

    ## Multi-Contrast Synthesis

    **Sequential Generation**:
    - Generate T2 from T1
    - Generate FLAIR from T1
    - Generate PD from T1
    - Each uses same condition, independent noise → diverse outputs

    **Joint Generation**:
    - Condition on T1, generate T2 + FLAIR simultaneously
    - Shares condition features, reduces computation
    - Enforces anatomical consistency across contrasts

    ## Acquisition Scenarios

    1. **Provided T1 Only**: Generate missing contrasts (synthetic FLAIR, etc.)
    2. **Quality Improvement**: Denoise/super-res existing contrasts
    3. **Cross-Protocol**: Map from 3T protocol to 0.05T protocol
    4. **Time Series**: Generate 4D (x,y,z,t) dynamic contrasts

    Attributes:
        state: TrainingState with multi-contrast model
        loss_computer: UnifiedDiffusionLossComputer for cross-modal loss
        condition_encoder: Strategy-owned ``nn.Module`` mapping the source
            modality → conditioning embedding. ``None`` unless the
            ``training.diffusion.cross_modal`` block is enabled.
        cross_modal_provenance: dict of the resolved cross-modal knob values
            stamped at construction (empty when cross-modal is disabled).
        device: Computation device (CUDA/CPU)

    References:
        - Conditioned Score Matching: Extensions of score-based diffusion
        - Zhao et al. (2020): Synthesizing MRI Images with Diffusion Models
    """

    #: X-Diffusion noises the clean target inside the step
    #: (``x_t = sqrt(a_bar) * x_0 + sqrt(1 - a_bar) * eps``), so
    #: ``first_steps/input_prepared`` -- captured before the forward pass -- is
    #: the source modality, not what the UNet receives.
    #:
    #: Stated explicitly because this class extends ``BaseTrainingStrategy``
    #: rather than ``DiffusionTrainingStrategy``, so it inherited the ``True``
    #: default despite running a textbook forward diffusion. Its own docstrings
    #: describe the noising; the machine-readable flag contradicted them, which
    #: is the half a reader of the artifact sees (non-negotiable 14).
    snapshot_prepared_is_model_input: bool = False

    #: Its own tag, not the parent's ``diffusion_step``: this class does not
    #: share ``DiffusionTrainingStrategy``'s emission path, and the per-tag call
    #: budget in ``save_debug_snapshot`` is keyed on ``(run_dir, tag)`` -- so a
    #: borrowed name would also mean a borrowed allowance.
    snapshot_model_input_tag: str | None = "xdiffusion_step"

    #: Loss ownership (issue #1918). Unlike its sibling this class does not use a
    #: loss computer: ``_compute_losses_impl`` iterates ``self.env.losses`` directly
    #: and accumulates ``weight * loss_val`` with the weight read from the same
    #: loss-weight SSOT (``_get_loss_weight``). Every builder-produced entry --
    #: which is every declared ``losses.*_losses`` entry -- therefore reaches the
    #: objective, so the flag is True. Nothing is computed inline: the ``mse``
    #: fallback fires only when ``env.losses`` is EMPTY, in which case there is no
    #: declared entry for it to double-count.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = True

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
            device (Optional[torch.device]): Description.
        """
        super().__init__(env=env, device=device, **kwargs)

        # Initialize strategy-specific components using unified loss computer
        self.loss_computer = UnifiedDiffusionLossComputer(config=self.config, device=self.device)

        # X-Diffusion supports AMP like other diffusion models
        # FP16 forward pass with FP32 schedule

        self._setup_strategy_specific_components()

    # Condition-encoder types the strategy knows how to build. An unknown value
    # in the YAML must RAISE (pitfall #9), never fall back to a default.
    _VALID_CONDITION_ENCODERS = ("conv",)

    # Generator forward kwargs that can carry the source-modality embedding,
    # in priority order. The first one the generator accepts is used.
    _CONDITION_FORWARD_KWARGS = ("cond", "context", "condition")

    def _setup_strategy_specific_components(self) -> None:
        """Initialize X-Diffusion-specific components and perform validation.

        Reads the optional ``training.diffusion.cross_modal`` block. When the
        block is enabled, every advertised knob is read + validated (raising on
        an unknown ``condition_encoder``) + stamped into
        ``self.cross_modal_provenance`` (pitfall #15), and a strategy-owned
        condition encoder is built, moved to ``self.device`` and registered on
        the generator optimizer so its parameters actually train.
        """
        self._verify_strategy_config(
            expected_modes=("diffusion", "3d_generation", "cross_modal_diffusion"),
        )
        self._log_config_features(self.logging_service)

        # Defaults so downstream code can always read these attributes even when
        # cross-modal conditioning is disabled (the common path).
        self.condition_encoder: torch.nn.Module | None = None
        self.cross_modal_provenance: dict[str, Any] = {}
        self._condition_forward_kwarg: str | None = None

        self._setup_cross_modal_conditioning()

    def _resolve_cross_modal_config(self) -> dict[str, Any] | None:
        """Return the ``training.diffusion.cross_modal`` block, or None.

        The block lives inside the free-form ``training.diffusion`` mapping (or
        is exposed as an attribute on a typed diffusion config). Returns None
        when the block is absent or explicitly disabled, so the strategy stays a
        plain single-modality denoiser and advertises no unread knob.
        """
        diffusion_cfg = self.config.training.diffusion if self.config.training is not None else None
        if diffusion_cfg is None:
            return None

        if isinstance(diffusion_cfg, dict):
            block = diffusion_cfg.get("cross_modal")
        else:
            block = getattr(diffusion_cfg, "cross_modal", None)
        if block is None:
            return None

        if not isinstance(block, dict):
            # Typed sub-config object — coerce to a plain dict of its fields.
            block = {
                k: getattr(block, k)
                for k in getattr(block, "model_fields", {})
                if hasattr(block, k)
            } or vars(block)

        if not block.get("enabled", True):
            return None
        return block

    def _setup_cross_modal_conditioning(self) -> None:
        """Build + register the condition encoder when cross-modal is enabled."""
        block = self._resolve_cross_modal_config()
        if block is None:
            return

        # --- Read + validate each advertised knob (raise on unknown, #9) ------
        encoder_type = str(block.get("condition_encoder", "conv"))
        if encoder_type not in self._VALID_CONDITION_ENCODERS:
            raise ConfigurationError(
                "X-Diffusion cross_modal.condition_encoder must be one of "
                f"{self._VALID_CONDITION_ENCODERS}, got {encoder_type!r}"
            )

        embedding_dim = int(block.get("condition_embedding_dim", 64))
        if embedding_dim <= 0:
            raise ConfigurationError(
                "X-Diffusion cross_modal.condition_embedding_dim must be a "
                f"positive int, got {embedding_dim!r}"
            )

        target_modalities = list(block.get("target_modalities", []) or [])
        num_contrasts = int(block.get("num_contrasts", len(target_modalities) or 1))
        if num_contrasts <= 0:
            raise ConfigurationError(
                "X-Diffusion cross_modal.num_contrasts must be a positive int, "
                f"got {num_contrasts!r}"
            )
        source_modality = block.get("source_modality")

        # --- Build the strategy-owned encoder (lazy: in_channels at 1st call) --
        # The strategy is NOT an nn.Module, so we hold the module as a plain
        # attribute and register its params on opt_g (env.opt_g is read-only).
        self.condition_encoder = _CrossModalConditionEncoder(embedding_dim=embedding_dim)
        device = getattr(self, "device", None)
        if device is not None:
            self.condition_encoder = self.condition_encoder.to(device)

        opt_g = getattr(self.env, "opt_g", None) if self.env is not None else None
        if opt_g is not None and hasattr(opt_g, "add_param_group"):
            opt_g.add_param_group({"params": list(self.condition_encoder.parameters())})

        # --- Stamp resolved knobs into provenance (#15) -----------------------
        self.cross_modal_provenance = {
            "enabled": True,
            "condition_encoder": encoder_type,
            "condition_embedding_dim": embedding_dim,
            "source_modality": source_modality,
            "target_modalities": target_modalities,
            "num_contrasts": num_contrasts,
        }
        self.logging_service.log_info(
            "X-Diffusion cross-modal conditioning enabled",
            **self.cross_modal_provenance,
        )

    def _resolve_condition_forward_kwarg(self, gen_forward: Any) -> str:
        """Return the generator-forward kwarg that carries ``z_cond``.

        Resolved (and cached) lazily on the first conditioned step. Raises if
        cross-modal conditioning is enabled but the generator's ``forward``
        accepts none of ``cond`` / ``context`` / ``condition`` — we never
        silently drop the condition embedding (pitfall #9).
        """
        if self._condition_forward_kwarg is not None:
            return self._condition_forward_kwarg

        from .mixins.utils import _callable_accepts_kwarg

        for kwarg in self._CONDITION_FORWARD_KWARGS:
            if _callable_accepts_kwarg(gen_forward, kwarg):
                self._condition_forward_kwarg = kwarg
                return kwarg

        raise ConfigurationError(
            "X-Diffusion cross-modal conditioning is enabled but the generator "
            f"forward accepts none of {self._CONDITION_FORWARD_KWARGS}; cannot "
            "feed the source-modality embedding (refusing to silently drop it)."
        )

    @staticmethod
    def _cosine_alpha_bar(t: torch.Tensor, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cumulative product ᾱ_t for a Nichol-Dhariwal cosine schedule.

        Returns ᾱ_t ∈ (0, 1] per element of ``t`` (continuous fraction
        ``t / num_timesteps``). Self-contained so the strategy needs no
        precomputed buffer table (it is not an ``nn.Module``).
        """
        f0 = math.cos(s / (1.0 + s) * math.pi / 2.0) ** 2
        frac = (t.float() / float(num_timesteps)).clamp(0.0, 1.0)
        f_t = torch.cos((frac + s) / (1.0 + s) * math.pi / 2.0) ** 2
        return (f_t / f0).clamp(1e-4, 1.0)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """
        Compute losses for X-Diffusion training step with AMP support.

        Real denoising-diffusion training step (replaces the prior no-op that
        discarded the generator output and regressed ``target + noise`` onto
        ``target`` — a loss with zero gradient to the model). Here we:

        1. Sample a timestep ``t`` and Gaussian noise ``ε``.
        2. Form the noised target ``x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε`` using a
           cosine ``ᾱ_t`` schedule (no precomputed buffer — the strategy is
           not an ``nn.Module``).
        3. Pass ``x_t`` (and ``t`` if the generator's ``forward`` accepts a
           ``timesteps``/``time`` kwarg) through ``self.env.generator`` to get
           an ``x_0``-prediction.
        4. Score the prediction against the clean target via the config losses
           (MSE fallback). The loss now depends on the generator output, so
           ``loss.backward()`` yields a nonzero generator gradient.

        Cross-modal conditioning: when ``training.diffusion.cross_modal`` is
        enabled (see the class docstring / ``_setup_cross_modal_conditioning``),
        the source modality ``input_batch`` is encoded to ``z_cond`` by the
        strategy-owned condition encoder and fed to the generator's ``forward``
        via its condition kwarg — so the prediction (and therefore the loss)
        depends on ``input_batch``, and the encoder's parameters receive
        gradient. When the block is absent the strategy degrades to a plain
        single-modality denoiser and ``input_batch`` is used only for batch
        resolution (no unread/unbacked knob is advertised).

        Args:
            input_batch: Low-resolution / source-modality input tensor batch.
            target_batch: High-resolution / target-contrast tensor batch.
            epoch: Current epoch index.
            **kwargs: Additional optional context (``batch``, ``step``, ...).

        Returns:
            Dictionary of loss tensors.
        """
        epoch = kwargs.get("epoch", epoch)
        # Resolve the real training iteration so the spatial-loss warm-up gate in
        # ``_get_loss_weight`` actually engages. The base train loop threads the
        # step under ``iteration`` (see the base diffusion strategy, which reads
        # ``kwargs.get("iteration", 0)``); ``step`` is accepted as a synonym. With
        # no iteration the gate defaults to 1_000_000 and is permanently bypassed.
        iteration = int(kwargs.get("iteration", kwargs.get("step", 0)) or 0)

        # Resolve a dict batch handed via kwargs (legacy calling convention),
        # then pick the present tensors (never ``a or b`` on tensors).
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(batch, dict):
            input_batch = pick_present(input_batch, batch.get("input"), batch.get("lr"))
            target_batch = pick_present(target_batch, batch.get("target"), batch.get("hr"))

        if input_batch is None or target_batch is None:
            raise ValueError(
                "X-Diffusion requires both input and target tensors; "
                f"got input={type(input_batch)}, target={type(target_batch)}"
            )

        # Log AMP usage for diffusion models
        if epoch == 0:
            self.logging_service.log_info(
                "X-Diffusion model detected - using AMP with FP32 schedule for stability",
                model_type=self.config.model.model_type,
                epoch=epoch,
            )

        generator = getattr(self.env, "generator", None)
        if generator is None:
            raise ValueError("X-Diffusion has no generator wired on env")

        # --- Forward diffusion: noise the clean target at timestep t ---------
        _diff = self.config.training.diffusion
        num_timesteps = int(
            getattr(_diff, "timesteps", 0) or getattr(self, "num_timesteps", 0) or 1000
        )
        b = target_batch.shape[0]
        t = torch.randint(1, num_timesteps, (b,), device=target_batch.device).long()
        eps = torch.randn_like(target_batch)

        a_bar = self._cosine_alpha_bar(t, num_timesteps).to(target_batch.device)
        # Broadcast ᾱ_t over (C, H, W[, D]) — keep batch dim, unsqueeze rest.
        view = [b] + [1] * (target_batch.dim() - 1)
        sqrt_ab = a_bar.sqrt().view(view)
        sqrt_1m_ab = (1.0 - a_bar).sqrt().view(view)
        x_t = sqrt_ab * target_batch + sqrt_1m_ab * eps

        # The UNet is fed the NOISED target, formed on the line above and never
        # leaving this method. ``first_steps/input_prepared`` holds the source
        # modality, so without this the only visible "input" is the tensor the
        # model does not receive -- the exact reading the contract exists to
        # prevent (non-negotiable 14).
        self._declare_model_input(
            {"model_input": x_t, "target": target_batch, "input": input_batch},
            # Explicit and empty: the canonical keys are unioned in from the
            # config SSOT for a k-space arm, which is correct here -- x_t is
            # target_batch scaled plus noise, so it lives in the target's domain.
            in_kspace_keys=set(),
            extra={
                "model_input_key": "model_input",
                "note": (
                    "X-Diffusion feeds x_t = sqrt(a_bar_t) * x_0 + "
                    "sqrt(1 - a_bar_t) * eps, formed inside this step from the "
                    "TARGET contrast. 'input' is the source modality, which "
                    "reaches the model only as a conditioning embedding "
                    "(z_cond) when cross-modal is enabled -- never as the "
                    "denoised tensor. 'first_steps/input_prepared' is that "
                    "source modality, i.e. PRE-noising and not the model input."
                ),
            },
        )

        # --- Model prediction (x_0-prediction); pass t if forward accepts it -
        from .mixins.utils import _callable_accepts_kwarg

        gen_forward = getattr(generator, "forward", generator)

        # Cross-modal conditioning: encode the source modality and thread the
        # embedding through the generator so the prediction depends on it. The
        # condition kwarg is resolved once, here, raising if cross-modal is
        # enabled but the generator accepts none of the recognized kwargs (#9).
        gen_kwargs: dict[str, Any] = {}
        encoder = getattr(self, "condition_encoder", None)
        if encoder is not None:
            cond_kwarg = self._resolve_condition_forward_kwarg(gen_forward)
            z_cond = encoder(input_batch)
            if z_cond.device != x_t.device:
                z_cond = z_cond.to(x_t.device)
            gen_kwargs[cond_kwarg] = z_cond

        if _callable_accepts_kwarg(gen_forward, "timesteps"):
            prediction = generator(x_t, timesteps=t, **gen_kwargs)
        elif _callable_accepts_kwarg(gen_forward, "time"):
            prediction = generator(x_t, time=t, **gen_kwargs)
        else:
            prediction = generator(x_t, **gen_kwargs)

        # Models may return tuples/dicts; take the primary tensor.
        if isinstance(prediction, (tuple, list)):
            prediction = prediction[0]
        elif isinstance(prediction, dict):
            prediction = pick_present(
                prediction.get("output"),
                prediction.get("prediction"),
                prediction.get("x0"),
            )

        if prediction.device != target_batch.device:
            prediction = prediction.to(target_batch.device)

        # --- Loss: prediction vs clean target (gradient flows to generator) --
        components: dict[str, torch.Tensor] = {}
        total_loss: torch.Tensor | None = None

        env_losses = self.env.losses if self.env else {}
        if env_losses:
            for loss_name, loss_fn in env_losses.items():
                loss_val = loss_fn(prediction, target_batch)
                if isinstance(loss_val, torch.Tensor):
                    components[loss_name] = loss_val
                    weight = self._get_loss_weight(loss_name, epoch=epoch, iteration=iteration)
                    if total_loss is None:
                        total_loss = weight * loss_val
                    else:
                        total_loss = total_loss + weight * loss_val

        if total_loss is None:
            _mse_fn = create_loss("mse")
            mse_loss = _mse_fn(prediction, target_batch)
            components["mse"] = mse_loss
            total_loss = mse_loss

        self._loss_dict_reuse.clear()
        self._loss_dict_reuse.update(components)
        self._loss_dict_reuse["g_total_loss"] = total_loss

        if "loss" not in self._loss_dict_reuse:
            self._loss_dict_reuse["loss"] = total_loss

        return self._loss_dict_reuse
