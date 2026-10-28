from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.cycle_bloch_critic import resolve_critic
from spectramr.infrastructure.training.strategies.cycle_bloch_physics import (
    ESTIMATOR_RANGES,
    ParameterEstimator,
    build_backward,
)
from spectramr.infrastructure.training.strategies.mixins.reconstruction import _as_aux_mapping
from spectramr.models.losses.gan_loss_library import gradient_penalty_loss
from spectramr.models.losses.registry import create_loss


class CycleBlochStrategy(BaseTrainingStrategy):
    """Cycle-consistent physics training for realistic anatomically-faithful synthesis.

    Ensures generated high-field MRI images are not only visually plausible but also
    physically capable of producing the observed ultra-low-field signal via forward
    Bloch simulation. Combines cycle consistency with biophysical constraints.

    ## Paradigm Overview

    **Problem**: Standard adversarial training on high-field (HF) images can produce
    visually realistic synthetics that violate MRI physics. Generated tissue parameters
    (T1, T2, PD) may be implausible when simulated via Bloch equations.

    **Solution**: Enforce cycle consistency through Bloch simulation:
    - HF image → estimate tissue params → Bloch simulate ULF → compare with measured ULF
    - Ensures generated HF image would produce measured ULF signal
    - Physically-grounded training without explicit parameter labels

    ## Architecture

    **Components**:
    1. **Generator(ULF) → HF**: Main generator, transforms ULF → High-field estimate
    2. **ParameterEstimator**: Lightweight CNN to estimate (PD, T1, T2) from HF image
    3. **BlochSimulator**: Differentiable MRI physics forward model (ULF sequence)
    4. **Discriminator(HF)**: Adversarial network on HF image

    ## Training Loop

    **Standard GAN Step**:
    1. Generate: HF_fake = Generator(ULF_real)
    2. Discriminate: D(HF_fake) vs D(HF_real)
    3. Update: Generator minimizes discriminator loss

    **Cycle-Physics Step** (novel):
    1. Generate: HF_fake = Generator(ULF_real)
    2. **Estimate Parameters**: (PD, T1, T2) = ParameterEstimator(HF_fake)
    3. **Simulate**: ULF_sim = BlochSimulator(PD, T1, T2, ULF_sequence_params)
    4. **Cycle Loss**: L1(ULF_sim, ULF_real) - simulated must match measured!
    5. Total Loss = L_adversarial + λ·L_cycle_physics

    ## Biophysics Modules

    **ParameterEstimator**:
    - 3-layer CNN: 1 channel input → 3-channel output (PD, T1, T2)
    - Softplus activation: Enforces positive tissue parameters
    - Trained jointly with generator (end-to-end)

    **BlochSimulator**:
    - Simulates ULF MRI sequence on estimated tissue map
    - Inputs: (PD, T1, T2), (TR, TE, FA) sequence parameters
    - Output: Synthetic ULF image via Bloch equation evolution
    - Fully differentiable (PyTorch custom ops)

    ## Typical MRI Parameters

    **Ultra-Low-Field (ULF) Sequence** (0.05T - 0.2T):
    - TR (Repetition Time): 500ms (long, slow)
    - TE (Echo Time): 14ms (short, minimize dephasing)
    - FA (Flip Angle): Variable (e.g., 30°)

    **High-Field (HF) Sequence** (3T):
    - TR: Shorter (faster, clinical speedup)
    - TE: Shorter (better SNR)
    - FA: Optimized per tissue (e.g., 90° for T2)

    ## Configuration

    - `training.training_mode`: 'cycle_bloch' or legacy mode
    - `physics.bloch.enabled`: True for simulator
    - `losses.gan.lambda_cycle_bloch`: λ_cycle
    - `training.cycle_bloch.backward_model`: bloch_fixed (spin echo, TR 500 / TE 14 ms)
      or ulf_forward (DifferentiableULFForwardOperator at `physics.field_strength`
      with `ulf_tr_ms`, `ulf_te_ms`, `ulf_flip_deg`, `tissue_field_tesla`)

    ## Loss Components

    1. **Adversarial Loss**: Fool discriminator on HF image
       - `losses.gan.gan_loss_type`: lsgan (default when losses.gan is absent) or
         wgan-gp (requires `enable_gradient_penalty: true`, weight `lambda_gp`)

    2. **Cycle Physics Loss**: Generated HF → simulate ULF → match measurement
       - L1(ULF_simulated, ULF_measured)
       - Critical: Enforces physical realism

    3. **Parameter Realism Loss** (optional):
       - Regularize (PD, T1, T2) to physiological ranges
       - T1 ∈ [100, 4000]ms, T2 ∈ [10, 200]ms

    4. **Reconstruction Loss** (optional):
       - L1(HF_fake, HF_real) if available

    ## Key Features

    ✅ **Physics-Grounded**: Cycle loss via Bloch simulation
    ✅ **End-to-End**: Parameters estimated and simulated differentiably
    ✅ **Interpretable**: Estimated (PD, T1, T2) can be visualized
    ✅ **Constraint-Free**: No need for ground truth parameter maps
    ✅ **Cross-Field**: Enables ULF↔HF synthesis with physics fidelity

    ## Inference

    1. Input: Ultra-low-field MRI
    2. Generate: HF_estimate = Generator(ULF)
    3. Estimate: (PD, T1, T2) = ParameterEstimator(HF_estimate)
    4. Outputs:
       - Synthesized high-field image (HF_estimate)
       - Tissue parameter maps (PD, T1, T2)
       - Optional: Simulated ULF for validation

    ## Registration

    Registered in ``TrainingStrategyFactory.STRATEGY_CLASS_PATHS`` as
    ``"cycle_bloch"`` and inherits from ``BaseTrainingStrategy``.
    Fully integrated with the framework registry.

    Attributes:
        state: TrainingState with generator and discriminator
        estimator: ParameterEstimator (PD, T1, T2 prediction)
        device: Computation device (CUDA/CPU)
        cycle_bloch_provenance: resolved backward model and its sequence parameters

    References:
        - Bloch, F. (1946): Nuclear Induction
        - Practical Bloch equations implementation in MRI physics
    """

    # parallel_compat: The estimator is an owned module, built on every rank's first generator
    # step from the generator's output channels, DDP-wrapped and run via call_owned.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        return (*super().owned_module_declarations(config), ("estimator", OWNED_MODULE))

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

        # Physics Components
        # Estimator is lazily created on first forward pass to match actual generator output channels,
        # since ChannelAdapter may modify the channel count at runtime.
        self._estimator_channels: int | None = None
        self.estimator: ParameterEstimator | None = None

        cb_cfg = getattr(self.config.training, "cycle_bloch", None)
        backward_model = cb_cfg.backward_model if cb_cfg is not None else "bloch_fixed"
        field_strength = getattr(self.config.physics, "field_strength", None)
        self._render_ulf, sequence = build_backward(
            cb_cfg,
            field_strength,
            self._contrast_map() if self.consumes_contrast_id(self.config) else None,
        )
        self._estimate_b1 = cb_cfg is not None and cb_cfg.b1_plus == "estimated"
        self._b1_tv_weight = float(cb_cfg.b1_tv_weight) if cb_cfg is not None else 0.0
        self._b1_tv = create_loss("tv")
        self.cycle_bloch_provenance: dict[str, Any] = {
            "backward_model": backward_model,
            **sequence,
            "b1_plus": "estimated" if self._estimate_b1 else "off",
            "b1_tv_weight": self._b1_tv_weight,
            **ESTIMATOR_RANGES,
        }

        # Loss weights — SSOT: read from config.losses.gan (GANLossesConfig)
        # Safe fallback: config.losses.gan is None when experiments don't configure
        # a losses.gan: YAML section.  Use GANLossesConfig defaults in that case.
        _gan_cfg = self.config.losses.gan
        self.lambda_cycle = _gan_cfg.lambda_cycle_bloch if _gan_cfg else 10.0
        self.lambda_adv = _gan_cfg.lambda_cycle_adv if _gan_cfg else 1.0

        # SSOT: Use registry-based loss creation instead of hardcoded nn modules
        self.criterion_cycle = create_loss("l1")
        self._configure_critic(_gan_cfg)

    def _configure_critic(self, gan_cfg: Any) -> None:
        """Resolve losses.gan into the adversarial loss both closures use."""
        spec = resolve_critic(gan_cfg)
        self.critic = create_loss(spec.loss_name, label_smoothing=spec.label_smoothing)
        self._d_loss_scale, self._uses_gp = spec.d_scale, spec.uses_gp
        self.lambda_gp = spec.lambda_gp
        self.cycle_bloch_provenance.update(
            critic=spec.name, lambda_gp=spec.lambda_gp, label_smoothing=spec.label_smoothing
        )

    @classmethod
    def consumes_contrast_id(cls, config: Any) -> bool:
        """Whether the backward render reads ``contrast_id`` (a per-contrast sequence table).

        The audit reads this: the batch's ``contrast_idx`` then feeds the physics, not the
        generator or critic, which stay contrast-agnostic.
        """
        cb_cfg = getattr(getattr(config, "training", None), "cycle_bloch", None)
        return getattr(cb_cfg, "sequences", None) is not None

    def _contrast_map(self) -> dict[str, int] | None:
        mc = getattr(self.config.data, "multi_contrast", None)
        return dict(mc.contrast_map) if mc is not None and mc.enabled and mc.contrast_map else None

    def _setup_strategy_specific_components(self) -> None:
        """_setup_strategy_specific_components."""
        pass

    def _first_steps_extra(self, epoch: int, batch: Any = None) -> dict[str, Any]:
        """Stamp the resolved backward operator and its sequence into the snapshot."""
        extra = super()._first_steps_extra(epoch, batch=batch)
        extra["cycle_bloch"] = dict(self.cycle_bloch_provenance)
        return extra

    @staticmethod
    def _to_magnitude_image(x: torch.Tensor) -> torch.Tensor:
        # Delegate to the registered ``rss_coils_to_magnitude`` adapter
        # (src/data/adapters/channels.py) so YAMLs can opt in via
        # `adapters.pre_loss_target: [{name: rss_coils_to_magnitude}]`
        # and the audit knows the bridge is in play. Phase 4 will wire
        # the strategy to honor the YAML chain when present and only
        # fall back to this hardcoded call otherwise.
        from spectramr.data.adapters import get_adapter

        return get_adapter("rss_coils_to_magnitude")()(x)

    def _physics_cycle(
        self, fake_hf: torch.Tensor, contrast_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """HF -> (PD, T1, T2[, B1+]) -> simulated ULF, and the B1+ TV penalty (0 when off)."""
        self._get_estimator(fake_hf)
        tissue_maps = self.call_owned("estimator", fake_hf)
        pd, t1, t2 = tissue_maps[:, 0:1], tissue_maps[:, 1:2], tissue_maps[:, 2:3]
        b1 = tissue_maps[:, 3:4] if self._estimate_b1 else None
        rec_ulf = self._render_ulf(pd, t1, t2, b1, contrast_id)
        loss_b1_tv = self._b1_tv(b1) if b1 is not None else fake_hf.new_zeros(())
        return rec_ulf, loss_b1_tv

    def _get_estimator(self, x: torch.Tensor) -> ParameterEstimator:
        """Lazily create ParameterEstimator matching actual generator output channels.

        Args:
            x: Generator output tensor [B, C, H, W]

        Returns:
            Initialized ParameterEstimator module
        """
        actual_channels = x.shape[1]
        if self.estimator is None or self._estimator_channels != actual_channels:
            self._build_estimator(actual_channels, x.device)
        return self.estimator

    def _build_estimator(self, in_channels: int, device: torch.device) -> None:
        """Build the estimator, register it for DDP, and put its parameters on ``opt_g``."""
        self._estimator_channels = in_channels
        self.estimator = ParameterEstimator(in_channels=in_channels, with_b1=self._estimate_b1).to(
            device, non_blocking=True
        )
        self.register_owned_module("estimator", self.estimator)
        # The HF->(PD,T1,T2) physics-inversion head trains only if opt_g holds it.
        opt = getattr(self.env, "opt_g", None)
        if opt is not None:
            existing = {p for g in opt.param_groups for p in g["params"]}
            new_params = [p for p in self.estimator.parameters() if p not in existing]
            if new_params:
                opt.add_param_group({"params": new_params})

    def _materialize_owned_state(self, state: dict[str, Any]) -> None:
        """Rebuild a saved estimator from its first conv's input width before restore.

        Without this a resume raised, because ``estimator`` is ``None`` until the first step.
        """
        super()._materialize_owned_state(state)
        saved = state.get("modules", {}).get("estimator")
        if saved is not None and self.estimator is None:
            self._build_estimator(int(saved["net.0.weight"].shape[1]), self.device)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute Cycle-Bloch losses.

        Args:
            input_batch: Ultra-low-field input [B, C, H, W]
            target_batch: High-field target (optional) [B, C, H, W]
            epoch: Current epoch number
            **kwargs: Additional arguments

        Returns:
            Dictionary with loss components:
                - 'total': Total generator loss
                - 'gan': Adversarial loss
                - 'cycle': Cycle-physics loss
        """
        real_ulf = input_batch

        generator = self.env.generator
        discriminator = self.env.discriminator

        if discriminator is None:
            raise ValueError(
                "CycleBlochStrategy requires a discriminator. Either "
                "(a) add a ``discriminator:`` section to the experiment "
                "YAML under ``model:`` (e.g. ``model.discriminator: "
                "{discriminator_type: patch_gan_2d, ...}``), or "
                "(b) switch to a non-adversarial cycle strategy such as "
                "``CycleConsistencyStrategy`` that does not depend on a "
                "discriminator."
            )

        # --- 1. Forward: ULF -> HF (Hallucination) ---
        fake_hf = generator(real_ulf)

        # --- 2. Physics Inversion: HF -> Tissue Maps ---
        # Estimate T1/T2/PD maps from the generated HF image
        # --- 3. Backward: Tissue Maps -> ULF (Simulation) ---
        rec_ulf, loss_b1_tv = self._physics_cycle(fake_hf, kwargs.get("contrast_id"))

        # --- GAN Loss ---
        loss_gan = self.critic.compute_generator_loss(discriminator(fake_hf))

        # --- Cycle-Bloch Loss ---
        loss_cycle = self.criterion_cycle(rec_ulf, real_ulf)

        # --- Total Generator Loss ---
        total_loss_g = (
            (self.lambda_adv * loss_gan)
            + (self.lambda_cycle * loss_cycle)
            + (self._b1_tv_weight * loss_b1_tv)
        )

        # Canonical key is ``g_total_loss``; the orchestrator at
        # base.py:845 accepts ``total`` as a synonym but the on-disk
        # convention should be the canonical handle. See
        # TODO/audit/07_strategies_v6_2_v6_3_exotic.md F18.
        return {
            "g_total_loss": total_loss_g,
            "gan": loss_gan,
            "cycle": loss_cycle,
        }

    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        iteration: int = 0,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Orchestrate Cycle-Bloch training step via Trainer-compatible closures.

        Args:
            batch: Batch dictionary.
            epoch: Current epoch.
            iteration: Current iteration.
            input_batch: Pre-unpacked input (optional).
            target_batch: Pre-unpacked target (optional).

        Returns:
            List of optimization configurations for Trainer.
        """
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        real_ulf = input_batch
        if real_ulf is None:
            raise ValueError("Batch must contain 'image' or 'input' key.")

        real_ulf = real_ulf.to(self.device, non_blocking=True)
        if target_batch is not None:
            target_batch = target_batch.to(self.device, non_blocking=True)
        aux = _as_aux_mapping(batch)
        contrast_id = aux.get("contrast_id", aux.get("contrast_idx"))

        discriminator = self.env.discriminator
        if discriminator is None:
            raise ValueError(
                "CycleBlochStrategy requires a discriminator. Either "
                "(a) add a ``discriminator:`` section to the experiment "
                "YAML under ``model:`` (e.g. ``model.discriminator: "
                "{discriminator_type: patch_gan_2d, ...}``), or "
                "(b) switch to a non-adversarial cycle strategy such as "
                "``CycleConsistencyStrategy`` that does not depend on a "
                "discriminator."
            )

        # Prepare losses dict for shared state tracking
        losses_dict: dict[str, torch.Tensor] = {}

        # 1. Generator step (Hallucinate HF + Physics Consistency)
        g_closure = self._train_generator_step(
            real_ulf, target_batch, discriminator, epoch, iteration, losses_dict, contrast_id
        )

        # 2. Discriminator step (Adversarial Real/Fake)
        d_closure = self._train_discriminator_step(
            real_ulf, target_batch, discriminator, epoch, iteration, losses_dict
        )

        return [
            {
                "optimizer": self.env.opt_g,
                "closure": g_closure,
                "model": self.env.generator,
                "name": "generator",
            },
            {
                "optimizer": self.env.opt_d,
                "closure": d_closure,
                "model": discriminator,
                "name": "discriminator",
            },
        ]

    def _train_generator_step(
        self,
        real_ulf: torch.Tensor,
        target_hf: torch.Tensor | None,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, torch.Tensor],
        contrast_id: torch.Tensor | None = None,
    ) -> Any:
        """_train_generator_step.

        Args:
            real_ulf (torch.Tensor): Description.
            target_hf (Optional[torch.Tensor]): Description.
            discriminator (nn.Module): Description.
            epoch (int): Description.
            iteration (int): Description.
            losses_dict (dict[str, torch.Tensor]): Description.
        Returns:
            Any: Description.
        """

        def closure():
            """closure.

            Returns:
                Any: Description.
            """
            fake_hf = self.env.generator(real_ulf)

            rec_ulf, loss_b1_tv = self._physics_cycle(fake_hf, contrast_id)

            # Adversarial Loss
            loss_gan = self.critic.compute_generator_loss(discriminator(fake_hf))

            # Cycle-Physics Loss.
            # rec_ulf is 1-ch magnitude (from BlochSimulator); real_ulf may
            # be multi-channel (virtual coils, real/imag interleaved). When
            # the YAML declares `adapters.pre_loss_target: [{name:
            # rss_coils_to_magnitude}]`, apply_adapters runs it (visible to
            # the audit). Otherwise fall back to the inline helper for
            # legacy YAMLs (which delegates to the same registered
            # adapter — see _to_magnitude_image).
            if self.adapter_chains.get("pre_loss_target"):
                rec_ulf_target = self.apply_adapters("pre_loss_target", rec_ulf)
                real_ulf_target = self.apply_adapters("pre_loss_target", real_ulf)
            else:
                rec_ulf_target = self._to_magnitude_image(rec_ulf)
                real_ulf_target = self._to_magnitude_image(real_ulf)
            loss_cycle = self.criterion_cycle(rec_ulf_target, real_ulf_target)

            total_loss_g = (
                (self.lambda_adv * loss_gan)
                + (self.lambda_cycle * loss_cycle)
                + (self._b1_tv_weight * loss_b1_tv)
            )

            losses_dict.update(
                {
                    "loss_g": total_loss_g,
                    "loss_cycle": loss_cycle,
                    "loss_b1_tv": loss_b1_tv,
                }
            )
            self._last_step_metrics = {k: v.detach() for k, v in losses_dict.items()}

            return total_loss_g

        return closure

    def _train_discriminator_step(
        self,
        real_ulf: torch.Tensor,
        target_hf: torch.Tensor | None,
        discriminator: nn.Module,
        epoch: int,
        iteration: int,
        losses_dict: dict[str, torch.Tensor],
    ) -> Any:
        """_train_discriminator_step.

        Args:
            real_ulf (torch.Tensor): Description.
            target_hf (Optional[torch.Tensor]): Description.
            discriminator (nn.Module): Description.
            epoch (int): Description.
            iteration (int): Description.
            losses_dict (dict[str, torch.Tensor]): Description.
        Returns:
            Any: Description.
        """

        def closure():
            """closure.

            Returns:
                Any: Description.
            """
            with torch.no_grad():
                fake_hf = self.env.generator(real_ulf).detach()

            pred_fake_d = discriminator(fake_hf)
            if target_hf is None:
                if self._uses_gp:
                    raise ValueError(
                        "cycle_bloch: the wgan-gp critic needs a real HF target for its "
                        "real term and gradient penalty; this batch has none."
                    )
                loss_d_real = torch.zeros((), device=self.device)
                loss_d_fake = self.critic.compute_discriminator_loss(pred_fake_d, pred_fake_d)[1]
            else:
                loss_d_real, loss_d_fake = self.critic.compute_discriminator_loss(
                    discriminator(target_hf), pred_fake_d
                )

            loss_d = self._d_loss_scale * (loss_d_real + loss_d_fake)
            if self._uses_gp:
                loss_d = loss_d + self.lambda_gp * gradient_penalty_loss(
                    discriminator, target_hf, fake_hf
                )
            losses_dict["loss_d"] = loss_d
            self._last_step_metrics = {k: v.detach() for k, v in losses_dict.items()}
            return loss_d

        return closure
