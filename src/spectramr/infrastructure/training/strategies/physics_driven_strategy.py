"""Physics-Driven Training Strategy
================================

Training strategy for Pillar III: Physics-Driven Reconstruction.
Supports models that require k-space and mask inputs for Data Consistency (DC).

[FIX 21] AMP Guard:
Complex64 is NOT supported by standard FP16 AMP in PyTorch.
FFT operations have high dynamic range that FP16 cannot represent.
Recommendation: Run physics experiments in FP32.

[B0-AWARE] Off-Resonance Simulation:
Synthesizes B0 field maps on-the-fly for Non-Cartesian training.
"""

import re as _re
from typing import Any

import torch

from spectramr.domain.interfaces.service_interfaces import ILoggingService
from spectramr.infrastructure.physics.coordinate_utils import CoordinateSampler

from .loss_folding import declared_loss_weights
from .reconstruction import ReconstructionTrainingStrategy

# Hoisted to module scope (NN#9): ``_generate_predictions`` runs every training
# step; compiling this pattern per-call repeated the import lookup + compile
# overhead on the hot path. The pattern matches the kwarg name in PyTorch's
# ``...() got an unexpected keyword argument 'X'`` TypeError message.
_KWARG_PAT = _re.compile(r"unexpected keyword argument ['\"]([^'\"]+)['\"]")


class PhysicsDrivenTrainingStrategy(ReconstructionTrainingStrategy):
    """Physics-informed training strategy with data consistency.

    Incorporates MRI physics constraints directly into training objective.
    Enforces k-space consistency and enables model learning from undersampled data.
    Inherits from ReconstructionTrainingStrategy to leverage robust loss computation,
    data consistency layers, and physics-informed helpers (PINN, K-Space Loss).

    ## Physics Constraints

    1. **Data Consistency Layer**: Projects reconstructions to valid k-space measurements
       - Hard projection: x' = F^{-1}(M⊙ỹ + (1-M)⊙Fx)
       - Soft weighting: Gradual enforcement with learned weight parameter

    2. **K-Space Enforcement**: Ensures compliance with observed measurements
       - Prevents physically impossible solutions
       - Improves reconstruction under high acceleration (8x, 10x)

    3. **Sensitivity Maps**: Supports multi-coil SENSE reconstruction
       - Uses coil sensitivity calibration from auto-calibration data
       - Handles parallel imaging geometry and noise correlation

    ## Training Process

    1. Forward Pass: Generator produces reconstruction from undersampled input
    2. Data Consistency: Project to k-space constraint manifold
    3. Loss Computation: Image-domain L1 + k-space penalties
    4. Backward: Update generator with physics guidance signals

    ## Key Configuration

    - `training.training_mode`: Must be 'reconstruction' or 'physics_driven'
    - `physics.data_consistency.enabled`: True to activate DC layer
    - `physics.data_consistency.method`: 'hard' (projection) or 'soft' (weighted)
    - `physics.data_consistency.weight`: DC term weighting (0.0-1.0)
    - `physics.kspace.enforce_hermitian_symmetry`: For real-valued output
    - `physics.b0_simulation.enabled`: Simulate off-resonance effects

    ## Loss Components

    - **Reconstruction Loss**: L1/L2 between prediction and ground truth
    - **Data Consistency**: Penalty for k-space deviation from measurements
    - **Adversarial Loss**: Optional discriminator feedback
    - **Frequency Penalty**: Focus on under-represented frequencies (optional)
    - **Physics Regularization**: Gradient constraints (optional)

    ## AMP Compatibility

    [FIX 21] Complex64 NOT supported by FP16 AMP in PyTorch. FFT operations
    have high dynamic range that FP16 cannot represent. Recommendation:
    Run physics experiments in FP32 (set `optimization.precision.enabled: false`).

    ## B0-Off-Resonance Simulation

    [B0-AWARE] Synthesizes B0 field maps on-the-fly for Non-Cartesian training.
    Adds B0 simulation to batch when `physics.b0_simulation.enabled: true`.
    Compatible with B0-aware models (B0AwareGraphUNet, NUFFTOperator).
    Simulates realistic off-resonance and field inhomogeneity effects.

    Attributes:
        state: TrainingState with physics-aware generators
        dc_operator: Data consistency projection module
        fft_transformer: FFT operations with proper normalization (ortho)
        device: Torch device (typically CUDA)
        _b0_simulator: Optional B0 field map simulator
    """

    def __init__(
        self,
        env: Any = None,
        logging_service: ILoggingService | None = None,
        **kwargs,
    ):
        """Initialize Physics-Driven strategy.

        Args:
            env: TrainingEnvironment containing device, state, models, etc.
            logging_service: Logging service instance
            **kwargs: Additional arguments passed to parent strategy
        """
        super().__init__(
            env=env,
            logging_service=logging_service,
            **kwargs,
        )

    def _setup_strategy_specific_components(self) -> None:
        """Initialize strategy-specific components."""
        # Allow 'physics_driven' mode in addition to 'reconstruction'
        self._verify_strategy_config(expected_modes=("reconstruction", "physics_driven"))
        self._log_config_features(self.logging_service)

        # [FIX 21] Warn if AMP is enabled for physics experiments
        if hasattr(self.config, "optimization"):
            opt = self.config.optimization
            if getattr(opt.precision, "enabled", False) or getattr(opt, "mixed_precision", False):
                import logging

                logging.getLogger(__name__).warning(
                    "[FIX 21] AMP/Mixed Precision detected in PhysicsDrivenStrategy. "
                    "Complex64 and FFT operations may fail or lose precision. "
                    "Consider setting mixed_precision: false in config."
                )

        # [B0-AWARE] Initialize B0 simulator if configured
        self._b0_simulator = None
        self._field_strength = self._get_field_strength()
        if self.config.physics.b0_simulation.enabled:
            try:
                from spectramr.infrastructure.physics.field_simulation import B0MapSimulator

                self._b0_simulator = B0MapSimulator(field_strength=self._field_strength)
                import logging

                logging.getLogger(__name__).info(
                    f"[B0-AWARE] B0 simulation enabled: field={self._field_strength}T, "
                    f"max_hz={self._b0_simulator.max_hz:.1f}, style={self.config.physics.b0_simulation.style}"
                )
            except ImportError as exc:
                # b0_simulation.enabled=True is an advertised, requested knob.
                # Swallowing the import failure and continuing with
                # _b0_simulator=None would silently downgrade the enabled
                # physics to disabled (pitfall #3/#15). Fail fast instead.
                raise RuntimeError(
                    "physics.b0_simulation.enabled=true but the B0MapSimulator "
                    "import failed; cannot honor the requested B0 simulation."
                ) from exc

        # Ensure K-Space loss is enabled for Physics Driven strategy if not explicitly set
        # SSOT: check config.losses.reconstruction.enable_kspace_loss
        pass

    def _get_field_strength(self) -> float:
        """Extract field strength from physics config."""
        if hasattr(self.config, "physics"):
            physics = self.config.physics
            return getattr(physics, "field_strength", 3.0)
        return 3.0

    def synthesize_b0_map(
        self, batch_size: int, shape: tuple, device: torch.device
    ) -> torch.Tensor | None:
        """Synthesize B0 maps for a batch on-the-fly.

        Consumed by :meth:`_generate_predictions`, which offers the result to the
        forward model as ``b0_map=`` (off-resonance-aware physics generators use
        it; others drop it via the kwarg-strip loop). Returns ``None`` when
        simulation is disabled or the per-call probability gate skips this batch.

        Args:
            batch_size: Number of samples
            shape: Image shape (H, W)
            device: Target device

        Returns:
            B0 maps [B, 1, H, W] in Hz, or None if simulation disabled
        """
        if self._b0_simulator is None or not self.config.physics.b0_simulation.enabled:
            return None

        # Probability check
        prob = self.config.physics.b0_simulation.probability
        # [PERF FIX] Avoid GPU→CPU sync: torch.rand generates scalar without .item()
        if torch.rand(()) > prob:
            return None

        # Use field-strength-aware generation
        return self._b0_simulator.generate_batch(
            batch_size=batch_size,
            shape=shape,
            max_hz=self.config.physics.b0_simulation.max_hz,  # None = use field default
            style=self.config.physics.b0_simulation.style,
            include_maxwell=(self._field_strength < 0.1),  # Maxwell terms for ULF
            device=device,
        )

    def _generate_predictions(
        self,
        lr_image: torch.Tensor,
        forward_kwargs: dict[str, Any],
        batch_context: dict[str, Any],
    ) -> tuple[torch.Tensor, Any]:
        """Generate predictions ensuring Complex64/FP32 execution."""

        # [FIX 21] Enforce FP32 for Physics-Driven models (FFT/Complex support)
        # We disable AMP for the forward pass of the generator
        with torch.amp.autocast("cuda", enabled=False):
            # Ensure inputs are FP32/Complex
            if lr_image.is_floating_point():
                lr_image = lr_image.float()

            # [HYBRID-AWARE] Handle coordinate sampling for continuous resolution models
            if getattr(self.env.generator, "use_coordinate_sampling", False):
                # We use the full grid for standard training context.
                # In optimized pipelines, this could be a subset of points.
                forward_kwargs["query_coords"] = CoordinateSampler.get_full_grid(
                    batch_size=lr_image.shape[0],
                    shape=lr_image.shape[-2:],
                    device=self.device,
                )

            # [PAIRED-AWARE] Propagate the high-field target into
            # ``forward_kwargs`` as ``target=`` so registration / paired
            # physics models (e.g. ``b0_estimator``, which needs both
            # moving and fixed scans to estimate B0 inhomogeneity from
            # geometric distortion) can opt in via ``kwargs.get("target")``
            # without a strategy-side branch on model type.  Models that
            # don't consume ``target`` absorb it through their
            # ``**kwargs`` and are unaffected.  Fixes the 2026-05-10
            # ``exp_51_b0_estimation`` smoke crash where the b0_estimator
            # raised "Expected at least 2 channels for concatenated input,
            # got 1" because the strategy never gave it the fixed scan.
            hr_target = batch_context.get("hr") if batch_context else None
            if hr_target is not None and "target" not in forward_kwargs:
                forward_kwargs["target"] = hr_target

            # [B0-AWARE] If physics.b0_simulation.enabled, synthesise an
            # off-resonance field and offer it to the forward model as
            # ``b0_map=``. Off-resonance-aware physics generators (e.g.
            # ``qmap_pinn``) consume it; the progressive-strip loop below drops
            # it for models that don't. This wires the previously-dead
            # ``synthesize_b0_map`` (zero callers -> facade, pitfall #16) so the
            # advertised knob is load-bearing here, mirroring how
            # graph_cold_diffusion consumes the same simulator.
            if self._b0_simulator is not None and "b0_map" not in forward_kwargs:
                b0_map = self.synthesize_b0_map(
                    batch_size=lr_image.shape[0],
                    shape=tuple(lr_image.shape[-2:]),
                    device=lr_image.device,
                )
                if b0_map is not None:
                    forward_kwargs["b0_map"] = b0_map

            # Call parent implementation (which calls generator)
            # We reproduce logic to wrap it in the context
            #
            # F8-r5 (2026-05-21 smoke audit): the prior retry only stripped
            # ``target`` if it was the offending kwarg, but several
            # physics-driven generators (PIMN especially) reject the FULL
            # set of strategy-injected kwargs (``target``, ``query_coords``,
            # ``kspace_measured``, ``mask``, …) one by one. Each rejection
            # raised a fresh ``unexpected keyword argument 'X'`` TypeError
            # and the retry-once-strip-target loop couldn't keep up.
            #
            # Progressive-strip retry: catch each ``unexpected keyword
            # argument 'X'`` TypeError, pop X, retry — up to N times.
            # If we exhaust the kwargs without success the original
            # exception is re-raised so the actual model bug surfaces.
            _max_strip_attempts = len(forward_kwargs) + 1  # bound
            for _attempt in range(_max_strip_attempts):
                try:
                    hr_fakes = self.env.generator(lr_image, **forward_kwargs)
                    break
                except TypeError as exc:
                    m = _KWARG_PAT.search(str(exc))
                    if m and m.group(1) in forward_kwargs:
                        forward_kwargs.pop(m.group(1))
                        continue
                    raise
            else:
                # Exhausted attempts — the generator rejected every kwarg
                # combination. A bare ``raise`` here is outside any ``except``
                # block (each iteration ``continue``d on a matched TypeError),
                # so Python would raise "No active exception to re-raise" and
                # ``hr_fakes`` would be unbound. Raise an informative error.
                raise RuntimeError(
                    f"PhysicsDrivenStrategy: exhausted {_max_strip_attempts} "
                    "kwarg-strip attempts without a successful generator call. "
                    f"Remaining kwargs: {list(forward_kwargs)!r}. "
                    "Check that the generator's forward() signature is compatible "
                    "with the strategy-injected kwargs."
                )

            if isinstance(hr_fakes, tuple):
                hr_fakes = hr_fakes[0]

            intermediate_outputs = None
            if hasattr(self.env.generator, "forward_with_intermediates"):
                # Re-implement safety check if needed, or assume main generator is enough
                pass

            # [DC CONSOLIDATION] DC is now applied internally by the generator
            # if 'kspace_measured' and 'mask' are passed in kwargs.
            pass

        return hr_fakes, intermediate_outputs

    def _physics_forward_fp32(
        self, forward_op, image: torch.Tensor, sensitivity_maps: torch.Tensor = None
    ) -> torch.Tensor:
        """Execute physics forward operator in FP32 even under AMP.

        [FIX 21] Explicitly disables autocast for FFT/NUFFT operations.

        Args:
            forward_op: Forward operator (FFT, NUFFT, etc.)
            image: Image tensor [B, C, H, W]
            sensitivity_maps: Optional coil sensitivity maps

        Returns:
            K-space data
        """
        with torch.amp.autocast("cuda", enabled=False):
            # Cast to FP32/Complex64 explicitly
            image = image.float()
            if sensitivity_maps is not None:
                sensitivity_maps = sensitivity_maps.float()

            # Execute physics operation in full precision
            return forward_op(image, sensitivity_maps)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses including Physics constraints (Bloch Residual)."""
        # Call Parent for Standard Reconstruction Losses (L1, SSIM, Perceptual)
        losses = super()._compute_losses_impl(input_batch, target_batch, epoch, **kwargs)

        # Bloch-residual weight comes from the loss-weight SSOT, NOT from
        # ``config.losses.physics`` directly. That read (which the comment here used
        # to call the SSOT) was wrong in both directions:
        #   * ``lambda_bloch_residual`` DEFAULTS to 1.0 while ``enable_bloch_residual``
        #     defaults to False, so an arm that never mentions the term saw 1.0 > 0 and
        #     RAN a loss its own config disables;
        #   * an arm on the declarative paradigm (``losses.kspace_losses: [bloch_residual]``,
        #     no ``physics:`` block) has ``config.losses.physics is None``, so the guard
        #     short-circuited and the term was SILENTLY SKIPPED at whatever weight the
        #     rest of the system was applying.
        # ``build_loss_weight_table`` reads both surfaces, honours the enable flag, and
        # raises when they disagree. ``.get(..., 0.0)`` — not ``_get_loss_weight``, which
        # RAISES on an undeclared loss — keeps "undeclared means off" intact.
        lambda_bloch = declared_loss_weights(self.config).get("bloch_residual", 0.0)

        if lambda_bloch > 0:
            # We need T1, T2 maps and M_pred.
            # Assuming the model returns a dictionary with 'M_pred' or the output IS M_pred
            # AND the batch contains 'T1', 'T2' maps.

            # Reuse the grad-carrying prediction already produced by the
            # parent ``super()._compute_losses_impl`` call above
            # (``reconstruction.py`` stashes it on ``self._last_prediction``).
            # Re-running ``self.env.generator(input_batch)`` here would double
            # the per-step compute, run under AMP without the FP32 guard, and
            # bypass the strategy-injected ``forward_kwargs`` — see the
            # 2026-05-31 strategy audit (physics_driven_strategy finding #1).
            output = self._last_prediction
            if output is None:
                raise RuntimeError(
                    "Bloch residual requested (lambda_bloch_residual > 0) but no "
                    "prediction was stashed by the parent reconstruction loss path "
                    "(self._last_prediction is None). The generator forward must run "
                    "before the Bloch term can be computed."
                )
            M_pred = output if isinstance(output, torch.Tensor) else output.get("M_pred", output)

            # Retrieve Maps from kwarg 'batch' if passed, or we need to access the full batch
            # 'train_step' passes kwargs. Check if 'batch' is in kwargs.
            full_batch = kwargs.get("batch", {})

            # The Bloch residual is an advertised, *enabled* loss term
            # (lambda_bloch_residual > 0). Silently dropping it when the batch
            # lacks T1/T2 or the model output is the wrong shape would be a
            # "passed-with-warnings" no-op (pitfall #10/#15) — so we raise the
            # misconfiguration loudly instead of degrading to plain recon.
            if "T1" not in full_batch or "T2" not in full_batch:
                raise RuntimeError(
                    "Bloch residual requested (lambda_bloch_residual="
                    f"{lambda_bloch}) but the batch is missing required "
                    "quantitative maps 'T1' and/or 'T2'. A physics_driven arm "
                    "with the Bloch term enabled must supply T1/T2 maps; "
                    "disable lambda_bloch_residual to train without them."
                )

            T1 = full_batch["T1"].to(self.device, non_blocking=True)
            T2 = full_batch["T2"].to(self.device, non_blocking=True)
            PD = full_batch.get("PD", None)
            if PD is not None:
                PD = PD.to(self.device, non_blocking=True)

            # Check shapes for Time Dimension
            # If M_pred is [B, C, H, W], we can't do dM/dt.
            # Need [B, Time, 3, H, W].
            if M_pred.ndim != 5 or M_pred.shape[2] != 3:
                raise RuntimeError(
                    "Bloch residual requested (lambda_bloch_residual="
                    f"{lambda_bloch}) but the model prediction has shape "
                    f"{tuple(M_pred.shape)}; the Bloch residual needs a "
                    "magnetization trajectory of shape [B, Time, 3, H, W] "
                    "(3 = Mx/My/Mz). Either route a Bloch-trajectory model "
                    "or disable lambda_bloch_residual."
                )

            # Initialize Loss if not present
            if not hasattr(self, "bloch_loss_fn"):
                from spectramr.models.losses.physics_losses import BlochResidualLoss

                self.bloch_loss_fn = BlochResidualLoss().to(self.device, non_blocking=True)

            bloch_loss = self.bloch_loss_fn(M_pred, T1, T2, PD)
            losses["bloch_loss"] = bloch_loss
            # Out-of-place add: in-place ``+=`` on a grad-carrying loss tensor
            # is fragile (errors when the operand is a leaf needed for backward).
            losses["g_total_loss"] = losses["g_total_loss"] + lambda_bloch * bloch_loss

        return losses
