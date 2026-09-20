"""Graph Cold Diffusion Training Strategy.

Cartesian k-space cold diffusion. The name is historical: no graph operation runs
here and no arm on this strategy uses a graph model -- see the class warning (#2082).

Physics-Based Degradation (instead of Gaussian Noise):
- Forward Process (t=0 → t=T): Progressively undersample k-space (remove spokes/arms)
- Reverse Process: Learn to restore fully-sampled signal from sparse measurements

The "noise" is defined by:
1. K-space undersampling (acceleration factor increases with t)
2. B0 phase accumulation (off-resonance effects)

References:
- Bansal A et al., "Cold Diffusion: Inverting Arbitrary Image Transforms" NeurIPS 2022
- This implementation adapts Cold Diffusion for Non-Cartesian MRI reconstruction.
"""

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.utils.kspace_masks import KSpaceMaskGenerator
from spectramr.infrastructure.training.utils.transform_ops import FFTTransformer

logger = logging.getLogger(__name__)


class GraphColdDiffusionStrategy(BaseTrainingStrategy):
    """Cold Diffusion Strategy: deterministic k-space-undersampling degradation.

    Unlike standard diffusion (Gaussian noise), cold diffusion defines degradation
    via physics-based k-space undersampling and B0 field effects.

    .. warning::

       **This strategy is CARTESIAN-only, despite its name.** The opening line used to
       claim it "implements deterministic degradation modeling for Non-Cartesian MRI
       undersampling". It never did, and could not:

       * ``_setup_accelerator`` imported ``ACCELERATOR_REGISTRY`` from
         ``physics.sampling``. That name does not exist -- the registry is
         ``_ACCELERATOR_REGISTRY`` and is not in ``__all__`` -- so the import raised
         ``ImportError`` on every run, was caught, and logged at ``debug``.
       * The code it guarded was written against an API that does not exist either:
         it called ``registry[type](min_budget=...)`` and then ``.generate_mask(shape,
         t=, T=)``. No accelerator class accepts ``min_budget`` and none defines
         ``generate_mask``. The real surface is
         ``ColdDiffusionAccelerator.get_acceleration_mask(kspace_shape, timestep)``,
         which returns a MASK, not the ``(trajectory, _, time_vec)`` tuple the caller
         unpacked.
       * With ``self.accelerator`` permanently ``None`` the NUFFT branch was
         unreachable, so every run silently fell through to a Cartesian mask that was
         additionally **hardcoded** to ``random_cartesian``, ignoring
         ``physics.compressed_sensing.sampling_pattern`` entirely.

       The dead branch is gone and the pattern is now read from config (#1092).

    .. note::

       **Non-Cartesian patterns are REFUSED, not merely unimplemented.** The live
       mask stack (``KSpaceMaskGenerator``) does support ``radial`` / ``spiral`` /
       ``golden_angle``, so wiring them would have been a one-line change -- but their
       cold-diffusion SCHEDULE is broken in a way that silently produces a
       meaningless objective. Measured sampled-fraction vs timestep on a 64x64 grid:

       ===================  =====  ======  ======  ======  ======
       pattern              t=0    t=100   t=200   t=400   t=999
       ===================  =====  ======  ======  ======  ======
       random_cartesian     1.000  0.137   0.073   0.038   0.016
       uniform_cartesian    1.000  0.125   0.063   0.047   0.047
       radial               0.087  0.142   0.087   0.087   0.087
       spiral               0.113  0.193   0.198   0.198   0.198
       golden_angle         0.057  0.143   0.086   0.057   0.057
       ===================  =====  ======  ======  ======  ======

       Cold diffusion requires ``t=0`` to be the CLEAN end and degradation to increase
       monotonically with ``t``. The non-Cartesian rows fail both: ``t=0`` is already
       ~90 % undersampled (the model never sees ``x_0``) and the schedule saturates
       flat from about ``t=200`` (the timestep conditioning carries almost no
       information). Selecting one raises rather than training an arm whose forward
       process does not implement the paper it cites.\n    ## Cold Diffusion Concept

    **Standard Diffusion**: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε (Gaussian noise)
    **Cold Diffusion**: x_t = D_t(x_0) (Deterministic degradation)\n    Where D_t progressively removes information via:
    - K-space undersampling (increasing acceleration)
    - B0 phase corruption (off-resonance effects)
    - Not random noise, but physics-based degradation

    **Key Feature**: "Invertibility" - can always recover any level of degradation
    by applying inverse degradation. This contrasts with Gaussian noise where
    recovery is only probabilistic.

    ## Architecture

    **Forward Process** (Training):
    1. Clean fully-sampled k-space: x_0
    2. Progressive undersampling: x_0 → x_1 → x_2 → ... → x_T (fully undersampled)
       - t=0: Fully sampled
       - t=T: Maximally undersampled (few spokes/arms)
    3. Progressive B0 corruption

    **Reverse Process** (Inference):
    1. Heavily undersampled acquisition: x_T (few k-space spokes)
    2. Predict clean sample at next stage: x_{T-1}
    3. Gradually add information back: x_T → x_{T-1} → ... → x_0 (full k-space)
    4. IFFT to image domain

    ## Non-Cartesian Specifics -- ASPIRATIONAL, NOT IMPLEMENTED

    The radial/spiral spoke-removal schedule described below is what this strategy was
    designed for and what its name promises. It has never run; see the warning at the
    top and #1092. Kept as the specification a real implementation should satisfy, and
    labelled so it is not mistaken for a description of current behaviour.

    **K-Space Trajectory** (target design):
    - Radial: Spokes removed gradually (full circle → partial wedge)
    - Spiral: Arms removed (multi-arm → single arm)
    - Custom: Edge/center removal following trajectory structure

    **B0 Phase Accumulation**:
    - Simulate off-resonance effects via frequency shifts
    - Phase accumulation proportional to field inhomogeneity map
    - Reversed during degradation reversion

    **Graph Neural Network**:
    - Models k-space as graph (non-Cartesian samples as nodes)
    - Edges based on k-space proximity
    - Permutation-invariant features (suitable for non-Cartesian)

    ## Training Process

    1. **Sample Degradation Level**: Randomly choose t ∈ [0, T]
    2. **Apply Degradation**: x_t = D_t(x_0) via undersampling + B0
    3. **Forward Pass**: Model predicts x_0 from x_t. The model is whatever the arm
       declares; every arm on this strategy today declares a convolutional UNet.
    4. **Loss**: L(x_0_pred, x_0_true) + optional physics constraints
    5. **Backward**: Update model weights

    ## Degradation Strategy

    **Acceleration Schedule**:
    - t=0: Acceleration = 1 (fully sampled)
    - t=T/2: Acceleration ≈ 4x
    - t=T: Acceleration = 8-10x (target undersampling)

    **Spoke Removal Pattern**:
    - Radial: Outer spokes first (preserve center k-space)
    - Or: Random spokes (variable density)

    **B0 Level**:
    - Proportional to t (small at t=0, large at t=T)
    - Parameterized by field inhomogeneity magnitude

    ## Configuration

    - `training.training_mode`: 'cold_diffusion' or 'reconstruction'
    - `physics.compressed_sensing.sampling_pattern`: the mask pattern that ACTUALLY
      drives degradation (Cartesian family only -- see the note at the top). Its
      schema default, 'cartesian', is aliased to 'random_cartesian'.
    - NB the three keys previously listed here -- `training.diffusion.degradation_type`,
      `.trajectory` and `.target_acceleration` -- were read by NOTHING in this
      strategy. `degradation_type` is a RENAMES alias for `diffusion.degradation`,
      which this strategy also never consults; the other two do not exist. Removed
      rather than left as documentation of knobs that cannot take effect (#15).
    - `training.diffusion.timesteps`: Number of degradation levels (default 50-100)

    ## Loss Components

    1. **Reconstruction Loss**: L1/L2 between predicted and ground truth k-space
    2. **Frequency Penalty**: Higher weight on low-frequency (center) k-space
    3. **Adversarial Loss**: Optional discriminator on reconstructed image
    4. **Consistency**: Ensure degradation operation is reversible

    ## Key Features

    ✅ **Physics-Based**: Degradation matches acquisition physics (not arbitrary noise)
    ✅ **Invertible**: Degradation steps are reversible (unlike Gaussian noise)
    ✅ **Interpretable**: Degradation type has direct clinical meaning (undersampling)
    ✅ **Stable**: Deterministic degradation enables stable score-based matching
    ❌ **Cartesian only**: the non-Cartesian claim this list used to carry contradicted
       the warning above it and survived two fix passes (#2082).

    ## Inference

    1. Load undersampled k-space: x_T (few spokes)
    2. Initialize noise prediction or guidance
    3. Reverse degradation: x_T → x_{T-1} → ... → x_0
    4. IFFT to image domain
    5. Output: Fully-sampled, noise-suppressed image

    ## Advantages vs Gaussian Diffusion

    - ✅ Physics-aligned (degradation = undersampling, not noise)
    - ✅ More stable training (deterministic vs stochastic)
    - ✅ Interpretable (know exact degradation at each step)
    - ✅ Better for undersampled data (matches MRI reality)
    - ❌ More complex (must implement specific degradations)
    - ❌ Slower inference (one model forward per reverse step)

    Attributes:
        fft_transformer: FFT operations with centering/normalization
        mask_generator: Generator for k-space undersampling masks
        timesteps: Number of degradation levels
        loss_computer: Loss computation, built lazily on first step

    References:
        - Bansal et al. (2022): Cold Diffusion: Inverting Arbitrary Image Transforms
        - Chiang et al.: Non-Cartesian MRI reconstruction with score-based models
    """

    #: Masks / corrupts inside the strategy (digital twin or its own schedule),
    #: so the ``undersampling:`` block is applied without a loader trajectory.
    applies_undersampling = True

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: object,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)

        self.fft_transformer = FFTTransformer(device=self.device)
        self.mask_generator = self._build_mask_generator()

        self._setup_cold_diffusion_components()

    def _build_mask_generator(self) -> KSpaceMaskGenerator:
        """Mask generator carrying the arm's declared ``undersampling:`` block.

        The block reached nothing here: the generator was built with only a
        timestep count and a device, so the accelerator fell back to its own
        defaults (#2060). On ``baseline_cdiffmr`` that meant a ladder running to
        R=64 against a declared 32, and ``seed=None`` -- the global RNG, so the
        cascade re-drew per call instead of truncating one ranking and stopped
        being nested, which the cold-diffusion forward process assumes (#1059).
        All three arms on this strategy are the literature baselines for the
        experiment_11 shootout, so the regime they were de-confounded to match
        was not the one they trained on.

        ``accelerator_kwargs_from_config`` is the allowlist
        ``KSpaceUndersamplingProcess`` and ``KspaceMixin`` already resolve
        through, so the three paths build one accelerator from one YAML.

        The PATTERN is deliberately not taken from there. It is owned by
        ``physics.compressed_sensing.sampling_pattern`` and resolved once by
        ``_resolve_degradation_pattern`` (#1092); every mask call passes it
        explicitly, so ``default_pattern`` here would be a second owner.
        """
        # Function-local: infrastructure -> models is a legal edge, but
        # ``spectramr.infrastructure.training.__init__`` eagerly imports
        # ``.strategies``, so importing at module scope closes a cycle back
        # through this package.
        from spectramr.models.diffusion.kspace_process import (
            accelerator_kwargs_from_config,
        )

        accel_config = self.config.undersampling
        accelerator_kwargs = (
            accelerator_kwargs_from_config(accel_config)[1] if accel_config else {}
        )
        return KSpaceMaskGenerator(
            num_timesteps=self.config.training.diffusion.timesteps,
            device=self.device,
            accelerator_kwargs=accelerator_kwargs,
        )

    def _setup_strategy_specific_components(self) -> None:
        """Initialize strategy-specific components."""
        # Accept all diffusion-related modes
        self._verify_strategy_config(
            expected_modes=(
                "reconstruction",
                "diffusion",
                "cold_diffusion",
                "kspace_cold_diffusion",
                "graph_cold_diffusion",
            )
        )
        self._log_config_features(self.logging_service)

    def _setup_cold_diffusion_components(self) -> None:
        """Initialize Cold Diffusion specific components."""

        # Diffusion timesteps from training config
        self.timesteps = self.config.training.diffusion.timesteps

        # Which k-space mask pattern drives the degradation. Resolved ONCE here so an
        # illegal value fails at setup rather than at the first training step.
        self.degradation_pattern = self._resolve_degradation_pattern()

        # B0 Simulation
        self._setup_b0_simulator()

        logger.info(
            "[Cold Diffusion] Initialized: timesteps=%d, pattern=%s, b0_sim=%s",
            self.timesteps,
            self.degradation_pattern,
            self.b0_simulator is not None,
        )

    # Remove obsolete _get_physics_config method - direct access via self.config.physics

    #: ``physics.compressed_sensing.sampling_pattern`` is a free ``str`` whose default
    #: is ``"cartesian"`` -- a value that is NOT a key in the mask registry (which has
    #: ``uniform_cartesian`` / ``random_cartesian``). Every arm using this strategy
    #: declares it, so it is mapped rather than rejected. It resolves to
    #: ``random_cartesian`` specifically to PRESERVE the pattern the old hardcoded
    #: fallback used: this change fixes wiring, and must not silently alter what eight
    #: existing arms actually train on. NB ``random_cartesian`` carries the #1069
    #: defect (collapses to a low-pass disc at high R) -- changing the default is a
    #: separate, deliberate decision, not a side effect of this fix.
    _PATTERN_ALIASES: ClassVar[dict[str, str]] = {"cartesian": "random_cartesian"}

    #: Supported by the mask generator, but with a cold-diffusion schedule that is
    #: inverted at t=0 and flat thereafter -- see the class docstring's table.
    _NON_CARTESIAN_PATTERNS = frozenset({"radial", "spiral", "golden_angle"})

    def _resolve_degradation_pattern(self) -> str:
        """Resolve the configured k-space mask pattern, or refuse to guess.

        Replaces ``_setup_accelerator``, which imported a non-existent name, called a
        non-existent method with a non-existent kwarg, and swallowed the resulting
        ``ImportError`` into a ``logger.debug`` -- leaving every run on a hardcoded
        ``random_cartesian`` mask that ignored this very field (#1092).

        Raises rather than falling back. A cold-diffusion arm quietly degrading on a
        pattern other than the one it declared is precisely the "reports success while
        training something the config did not ask for" failure (CLAUDE.md #9).
        """
        from spectramr.infrastructure.physics.sampling import _ACCELERATOR_REGISTRY

        declared = self.config.physics.compressed_sensing.sampling_pattern
        pattern = self._PATTERN_ALIASES.get(declared, declared)

        if pattern in self._NON_CARTESIAN_PATTERNS:
            raise ValueError(
                f"physics.compressed_sensing.sampling_pattern={declared!r} is a "
                "non-Cartesian family, which this strategy REFUSES for cold "
                "diffusion. The mask stack does produce it, but its schedule is not a "
                "degradation schedule: t=0 is already ~90% undersampled (so the model "
                "never sees x_0) and the sampled fraction saturates flat from about "
                "t=200 (so the timestep carries almost no information). See the class "
                "docstring for the measured table and #1092. Use a Cartesian pattern, "
                "or implement a real trajectory schedule first."
            )

        if pattern not in _ACCELERATOR_REGISTRY:
            raise ValueError(
                f"physics.compressed_sensing.sampling_pattern={declared!r} resolves to "
                f"{pattern!r}, which is not a registered k-space pattern. Available: "
                f"{sorted(_ACCELERATOR_REGISTRY)}. (The field is typed `str`, so a typo "
                "cannot be caught at schema-validation time -- it is caught here.)"
            )

        if declared != pattern:
            logger.info("[Cold Diffusion] sampling_pattern=%r resolved to %r", declared, pattern)
        return pattern

    def _setup_b0_simulator(self) -> None:
        """Setup B0 field simulation for phase corruption."""
        self.b0_simulator = None
        self.field_strength = self.config.physics.field_strength

        b0_config = self.config.physics.b0_simulation
        if b0_config.enabled:
            try:
                from spectramr.infrastructure.physics.field_simulation import B0MapSimulator

                self.b0_simulator = B0MapSimulator(field_strength=self.field_strength)
                logger.debug(
                    "[Cold Diffusion] B0 simulator: field=%sT, max_hz=%.1f",
                    self.field_strength,
                    self.b0_simulator.max_hz,
                )
            except ImportError:
                logger.debug("[Cold Diffusion] Failed to import B0MapSimulator")

    # ``_setup_nufft_operator`` lived here. It eagerly imported and cached
    # ``NUFFTOperator`` for a branch of ``_apply_degradation`` that was gated on a
    # trajectory the (broken) accelerator could never supply, so the operator was
    # loaded on every run and used on none. Removed with that branch (#1092).
    #
    # A genuine non-Cartesian cold diffusion needs a trajectory GENERATOR keyed on the
    # degradation timestep -- something that returns progressively fewer spokes/arms as
    # t rises. The mask stack cannot provide that (it returns grid masks), which is why
    # the removed code had to invent an API. That remains unimplemented; #1092 tracks it.

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute Cold Diffusion loss.

        Executes the forward degradation process (undersampling + B0) and computes
        reconstruction loss in the image domain.

        Args:
            input_batch: Input tensor (image or k-space).
            target_batch: Ground truth target (fully sampled).
            epoch: Current training epoch.
            **kwargs: Additional context (e.g., 'batch_data').

        Returns:
            Dictionary containing 'g_total_loss' and component losses.
        """
        B = target_batch.shape[0]
        device = target_batch.device

        # 1. Sample random timesteps t ∈ [0, T)
        t = torch.randint(0, self.timesteps, (B,), device=device).long()

        # 2. Apply physics-based degradation (forward diffusion)
        x_t, trajectory, target_signal = self._apply_degradation(target_batch, t, **kwargs)

        # 3. Model prediction
        # Pass degraded signal, trajectory, and timestep to model
        forward_kwargs = self._build_forward_kwargs(x_t, t, trajectory, **kwargs)
        pred_signal = self.env.generator(x_t, **forward_kwargs)

        # [STABILITY FIX] NaN/Inf guard on model output.
        # If the generator produces garbage (e.g., from DCF explosion or
        # graph aggregation divergence), return a zero-grad loss rather
        # than crashing backward() with NaN gradients.
        if torch.isnan(pred_signal).any() or torch.isinf(pred_signal).any():
            nan_frac = torch.isnan(pred_signal).float().mean().item()
            inf_frac = torch.isinf(pred_signal).float().mean().item()
            logger.warning(
                "[GraphColdDiffusion] NaN/Inf detected in generator output "
                "(NaN=%.4f%%, Inf=%.4f%%). Returning zero loss for this step.",
                nan_frac * 100,
                inf_frac * 100,
            )
            zero_loss = torch.tensor(0.0, device=device, requires_grad=True)
            return {
                "g_total_loss": zero_loss,
                "timestep_mean": t.float().mean(),
                "nan_skip": torch.tensor(1.0, device=device),
            }

        # 4. Compute loss via SSOT loss_computer
        # Convert prediction to image domain if it is in k-space
        if pred_signal.shape[1] == 2:  # Heuristic for K-Space (2 channels)
            pred_image = self.fft_transformer.ifft2c(pred_signal)
        else:
            pred_image = pred_signal

        target_image = target_batch

        # [STABILITY FIX] Normalize both prediction and target to a bounded range
        # based on the target scale to prevent exploding gradients from
        # unconventionally scaled MRI datasets.
        scale_factor = (
            target_image.abs()
            .amax(dim=tuple(range(1, target_image.dim())), keepdim=True)
            .clamp(min=1e-8)
        )
        pred_image = pred_image / scale_factor
        target_image = target_image / scale_factor

        # GraphColdDiffusionStrategy delegates all reconstruction loss to the loss_computer

        env_losses = self.env.losses if self.env and hasattr(self.env, "losses") else {}
        if not hasattr(self, "loss_computer") or self.loss_computer is None:
            from spectramr.models.losses.computers import UnifiedReconstructionLossComputer

            self.loss_computer = UnifiedReconstructionLossComputer(
                config=self.config, device=self.device
            )

        # Extract sensitivity maps for physics losses
        batch_data = kwargs.get("batch_data", {})
        if "sensitivity_maps" not in kwargs and "sensitivity_maps" in batch_data:
            kwargs["sensitivity_maps"] = batch_data["sensitivity_maps"]
        elif "sensitivity_maps" not in kwargs and "smaps" in batch_data:
            kwargs["sensitivity_maps"] = batch_data["smaps"]

        loss_output = self.loss_computer.compute(
            pred=pred_image,
            target=target_image,
            epoch=epoch,
            losses_dict=env_losses,
            **kwargs,
        )

        total_loss = loss_output.total
        extra_losses = loss_output.components

        return {
            "g_total_loss": total_loss,
            "timestep_mean": t.float().mean(),
            **extra_losses,
        }

    def _apply_degradation(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Apply the cold-diffusion degradation: k-space undersampling (+ optional B0).

        The middle return slot used to carry a non-Cartesian trajectory. It is always
        ``None`` now and the branch that produced it is gone -- it was gated on an
        accelerator that could never be constructed, so it never executed (#1092). The
        slot is KEPT rather than removed from the signature because
        ``_build_forward_kwargs`` already threads it to models that accept a
        ``trajectory`` kwarg, and that is where a real implementation would plug in.

        Args:
            x_0: Clean signal ``[B, C, H, W]`` (or ``[B, C, H, W, D]``).
            t: Timesteps ``[B]``.

        Returns:
            ``(x_t, None, target)`` -- degraded k-space, no trajectory, clean image.
        """
        shape = x_0.shape
        _B, _C, H, W = shape[0], shape[1], shape[2], shape[3]

        # ONE mask per batch, from the first element's timestep. This is the
        # simplification the method already made; it is preserved deliberately rather
        # than quietly upgraded to per-sample masking, which would change what every
        # existing arm trains on.
        t_val = t[0].item()

        # 1. FFT to k-space.
        k_full = self.fft_transformer.fft2c(x_0)

        # 2. Mask at this degradation level, on the CONFIGURED pattern. This was
        #    hardcoded to "random_cartesian" behind a comment claiming it was a
        #    fallback for when NUFFT "fails" -- NUFFT never ran, so the fallback was
        #    the only path, and physics.compressed_sensing.sampling_pattern reached
        #    nothing. `self.degradation_pattern` is validated at setup.
        if t_val > 0:
            mask = self.mask_generator.generate_acceleration_mask(
                timestep=t_val,
                image_shape=(H, W),
                pattern=self.degradation_pattern,
            )
            mask = self.mask_generator.expand_mask_to_channels(mask, k_full.shape[1])
        else:
            # t=0 is the CLEAN end of the schedule: fully sampled, no degradation.
            mask = torch.ones_like(k_full)

        x_t = k_full * mask

        # 3. Optional B0 phase corruption, scaled by position along the schedule.
        #
        # BEHAVIOUR CHANGE, called out because it is a SECOND dead feature in this
        # method rather than part of the accelerator fix. B0 used to be applied ONLY
        # inside the unreachable NUFFT branch, so `physics.b0_simulation.enabled: true`
        # built a B0MapSimulator that was then never used -- four of the eight arms on
        # this strategy declare it. Leaving it dead while rewriting the method would be
        # knowingly preserving a facade, and non-negotiable #8 says a declared knob must
        # be read. So it is applied here, at the same t/T scaling the dead branch used.
        #
        # If that is not wanted, the fix is to set `physics.b0_simulation.enabled:
        # false` on those arms -- an explicit declaration -- rather than to restore a
        # silently-ignored one.
        if self.b0_simulator is not None:
            from spectramr.infrastructure.physics.b0_utils import apply_b0_phase_shift

            b0_map = self.b0_simulator.generate_batch(shape[0], (H, W), device=x_0.device)
            x_t = apply_b0_phase_shift(x_t, b0_map, None, t_val / self.timesteps)

        target_signal = x_0  # loss is against the clean image
        return x_t, None, target_signal

    def _build_forward_kwargs(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        trajectory: torch.Tensor | None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build kwargs for generator forward pass, filtering unsupported ones."""
        from .mixins.utils import _callable_accepts_kwarg

        forward_kwargs = {}

        # Only add timesteps if model accepts it
        # Safely get the underlying model to check its forward signature
        underlying_model = getattr(self.env.generator, "model", self.env.generator)

        if _callable_accepts_kwarg(underlying_model.forward, "timesteps"):
            forward_kwargs["timesteps"] = t
        elif _callable_accepts_kwarg(underlying_model.forward, "time"):
            forward_kwargs["time"] = t
        # If model doesn't accept any timestep arg, skip it

        # Trajectory for graph construction (only if model accepts it)
        if trajectory is not None:
            if _callable_accepts_kwarg(underlying_model.forward, "trajectory"):
                forward_kwargs["trajectory"] = trajectory
            if _callable_accepts_kwarg(underlying_model.forward, "k_trajectory"):
                forward_kwargs["k_trajectory"] = trajectory

        return forward_kwargs

    @torch.no_grad()
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step for Graph Cold Diffusion.

        Args:
            batch: Data batch (dict or tuple)
            input_batch: Low-resolution/degraded input tensor.
            target_batch: High-resolution/clean target tensor.
            **kwargs: Additional arguments.

        Returns:
            Dictionary of validation metrics.
        """
        batch = (input_batch, target_batch)
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        input_batch = input_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)

        self.env.generator.eval()
        # Use t=0 (no degradation) for clean reconstruction
        t = torch.zeros(input_batch.shape[0], device=self.device, dtype=torch.long)

        # Build forward kwargs, filtering unsupported args
        forward_kwargs = self._build_forward_kwargs(input_batch, t, None)

        # Generate reconstruction
        reconstructed = self.env.generator(input_batch, **forward_kwargs)

        # Standardized Metric Computation
        val_config = self.config.validation
        compute_img_metrics = val_config.scoring.enable_image_metrics if val_config else True

        metrics = {}
        if compute_img_metrics:
            # [Refactor] Use SSOT ValidationMetricsComputer
            try:
                computer = self._get_validation_metrics_computer(self.config)
                computed = computer.compute(reconstructed, target_batch)

                # Apply prefix
                for k, v in computed.items():
                    metrics[f"val_{k}"] = v
            except Exception as e:
                if hasattr(self, "logging_service") and logger.isEnabledFor(logging.WARNING):
                    self.logging_service.log_warning("Validation metrics failed: %s", str(e))

        # Log images to TensorBoard (if configured)
        self._log_validation_images_to_tensorboard(
            predictions=reconstructed,
            targets=target_batch,
            inputs=input_batch,
            metrics=metrics,
        )

        return metrics
