from typing import Any, final

import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
from spectramr.infrastructure.training.contexts import TrainingEnvironment
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


class TttAdaptationStrategy(BaseTrainingStrategy):
    """Test-Time Training (TTT) strategy for online model adaptation.

    Adapts model weights during inference on patient-specific data to minimize
    self-supervised objectives (e.g., data consistency). Enables dramatic performance
    improvements without retraining on new data distribution.

    ## Core Concept

    **Objective**: At inference time, perform gradient-based optimization on unlabeled
    test patient data to fine-tune model for that specific anatomy/contrast/protocol.
    Leverages unlabeled test data to adapt from training distribution.

    **Key Insight**: Test data contains information about its own expected solutions
    (via self-supervised losses like data consistency). Optimizing these signals
    effectively personalizes the model.

    ## Training Phases

    **Phase 1: Pre-training** (Offline)
    - Train model on diverse training data
    - Learn good initialization for optimization
    - Establish baseline task performance

    **Phase 2: Test-Time Adaptation** (Online/Inference)
    - Load inference test batch
    - Freeze encoder/early layers, update decoder/late layers
    - Minimize self-supervised objective (data consistency, image statistics)
    - Typically 5-10 optimization steps per test sample

    ## Self-Supervised Loss Options

    1. **Data Consistency**: Enforce measured k-space fidelity
       - L2(y, M⊙F(x)) where y=measurements, x=reconstruction
       - Physics-informed, works for undersampled data

    2. **Entropy Minimization**: Maximize model certainty
       - Encourages confident predictions on unlabeled data
       - Works for classification-style problems

    3. **Image Statistics**: Match training data distribution
       - Texture entropy, contrast range, histogram matching
       - Domain adaptation without labels

    4. **Self-Consistency**: Consistency across augmentations
       - Apply transforms (rotation, small shifts), expect predictions match
       - Enforces robustness to nuisance variations

    ## Optimization Strategy

    **Outer Loop** (training, offline):
    - Train model on labeled data
    - No TTT applied

    **Inner Loop** (test-time, online):
    1. Load test batch x_test (unlabeled)
    2. Clone model weights: θ_adapt = θ
    3. For k steps:
       - Compute self-supervised loss L_ssl(x_test, θ_adapt)
       - Backward: ∇L_ssl
       - Step: θ_adapt ← θ_adapt − α∇L_ssl
    4. Use adapted weights θ_adapt for inference on x_test
    5. Discard adapted weights before next test sample (or keep for domain consistency)

    ## Configuration

    - `training.training_mode`: 'test_time_adaptation' or 'reconstruction'
    - `training.ttt.num_adaptation_steps`: Optimization iterations (5-20)
    - `training.ttt.adaptation_lr`: Step size for gradients (1e-3 to 1e-2)
    - `training.ttt.adaptation_loss`: 'data_consistency', 'entropy', 'statistics'
    - `training.ttt.frozen_modules`: Which layers to freeze ('encoder', 'all' except decoder)

    ## Loss Components

    - **Self-Supervised Loss**: Primary signal during test-time (data consistency, etc.)
    - **Reconstruction Loss**: Optional auxiliary signal if weak labels available
    - **Regularization**: Optional L2 on weight changes to stay near initialization

    ## Key Advantages

    ✅ **No Retraining**: Personalizes to new patient without new data collection
    ✅ **Works Offline**: Can be applied post-hoc during inference
    ✅ **Distribution Shift**: Handles domain shift gracefully (3T→0.05T, different protocols)
    ✅ **Interpretable**: Visualize adaptation progress via loss decrease
    ✅ **No Labels**: Uses only unlabeled test data (privacy-friendly)

    ## Disadvantages & Limitations

    ❌ **Slow Inference**: 5-20 extra optimization steps per test sample
    ❌ **Memory**: Backprop graph required during inference (GPU memory)
    ❌ **Mode Collapse**: Self-supervised losses can degrade generalization if not tuned
    ❌ **Unstable**: Adaptation can converge to poor local optima

    ## Practical Considerations

    - **Batch Size**: Test with batch_size=1 (per-sample adaptation most effective)
    - **Learning Rate**: 10-100x smaller than training LR (stability)
    - **Frozen Layers**: Typically freeze encoder (variance), train decoder only
    - **Loss Weight**: Balance self-supervised with any weak task signal
    - **Monitoring**: Track loss decrease to detect adaptation failure

    Attributes:
        state: TrainingState with model and config
        adaptation_steps: Number of optimization steps per test (default 10)
        adaptation_lr: Learning rate for TTT optimization
        adaptation_loss_fn: Self-supervised loss (data consistency, etc.)
        device: Computation device (CUDA/CPU)

    References:
        - Sun et al. (2020): Test-Time Training with Self-Supervision for Generalization
        - Cai et al. (2021): Exploiting Unlabeled Data in Self-Training with Confidence-Amplified Consistency Regularization
    """

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        **kwargs: Any,
    ) -> None:
        """__init__.

        Args:
            env (Optional[TrainingEnvironment]): Description.
        """
        super().__init__(env=env, **kwargs)
        self._setup_strategy_specific_components()

    def _setup_strategy_specific_components(self) -> None:
        # Get TTT parameters from config
        # Check if adaptation config exists (for meta-learning strategies)
        """_setup_strategy_specific_components."""
        # Canonical home is the typed training.ttt block (TTT and meta-learning
        # are distinct paradigms; reading ttt from model.adaptation_config
        # conflated them). Fall back to the legacy adaptation_config for
        # backward compatibility, else raise (pitfall #15).
        ttt = getattr(self.config.training, "ttt", None)
        if ttt is not None:
            self.num_adaptation_steps = ttt.adaptation_steps
            self.adaptation_lr = ttt.adaptation_lr
            self.consistency_weight = ttt.consistency_weight
        elif hasattr(self.config.training, "adaptation_config"):
            adaptation_config = self.config.training.adaptation_config
            self.num_adaptation_steps = adaptation_config.adaptation_steps
            self.adaptation_lr = adaptation_config.adaptation_lr
            # consistency_weight is not in meta-learning config, use default
            self.consistency_weight = 1.0
        else:
            raise ConfigurationError(
                "Test-time adaptation requires adaptation configuration. "
                "Configure the typed training.ttt block (adaptation_steps / "
                "adaptation_lr / consistency_weight) in your YAML file."
            )

        # We might need a separate optimizer for TTT if we don't want to affect global state
        # But BaseTrainingStrategy uses self.state.opt_g
        # We'll use the existing optimizer but maybe we should reset it or use a new one?
        # For simplicity, we use the provided optimizer.
        pass

    @final
    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Optimized training step for TTT using closures.

        Performs the inner loop adaptation as a single optimization step config.
        """
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)

        if input_batch is not None:
            input_batch = self._to_device(input_batch)
        if target_batch is not None:
            target_batch = self._to_device(target_batch)

        def ttt_closure() -> torch.Tensor:
            """Run the N-1 inner adaptation steps, then return the final loss."""
            for step in range(self.num_adaptation_steps - 1):
                self.env.opt_g.zero_grad(set_to_none=True)

                # Forward pass and loss
                losses = self._compute_losses_impl(
                    input_batch, target_batch, epoch, step=step, **kwargs
                )
                loss = losses["g_total_loss"]

                # Inner loop steps manually; the Trainer only owns the final one.
                # Which means these N-1 backwards bypass ``StepExecutor`` and so
                # bypass its pre-backward guards -- the graph check is called
                # here explicitly rather than duplicated (#1952, NN17).
                ensure_backward_ready(loss, name="ttt_adaptation_inner", global_step=step)
                loss.backward()
                self.env.opt_g.step()

                # NOTE: no per-inner-step metric store here. The old code did
                # ``self._last_step_metrics = {k: v.item() ...}`` on EVERY inner
                # adaptation step — ``num_adaptation_steps - 1`` GPU-synchronising
                # ``.item()`` rounds per training step — only to be overwritten
                # by the final-step store below (review 2026-07-01,
                # performance.md). The inner metrics were never read.

            # Final step: Compute loss and return for Trainer to handle backward/step
            self.env.opt_g.zero_grad(set_to_none=True)
            # #1190: bypasses the ``_compute_losses`` wrapper, so arm the
            # ``model_output`` snapshot here. FINAL call only -- the inner-loop
            # calls are pre-adaptation states (see ``_capture_model_output``).
            with self._capture_model_output(
                module=self.env.generator,
                input_batch=input_batch,
                target_batch=target_batch,
                step=int(kwargs.get("iteration", 0) or 0),
            ):
                final_losses = self._compute_losses_impl(
                    input_batch,
                    target_batch,
                    epoch,
                    step=self.num_adaptation_steps - 1,
                    **kwargs,
                )
            final_loss = final_losses["g_total_loss"]

            # Store final metrics
            self._last_step_metrics = {
                k: v.item() if isinstance(v, torch.Tensor) else v for k, v in final_losses.items()
            }
            return final_loss

        return [
            {
                "name": "ttt_adaptation",
                "closure": ttt_closure,
                "optimizer": self.env.opt_g,
                "model": self.env.generator,
            }
        ]

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute the self-supervised adaptation loss."""
        # Precompute invariant k-space once if available
        cached_mask = kwargs.get("mask")
        cached_kspace = kwargs.get("kspace")
        y_measured_cached: torch.Tensor | None = None

        if cached_mask is not None:
            if cached_kspace is not None:
                y_measured_cached = cached_kspace
            else:
                with torch.no_grad():
                    if hasattr(self, "context") and self.context and self.context.fft_transformer:
                        y_measured_cached = (
                            self.context.fft_transformer.fftnc(input_batch, dims=(-2, -1))
                            * cached_mask
                        )

        # Forward pass
        reconstructed = self.env.generator(input_batch)
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        results = {}

        if cached_mask is not None and y_measured_cached is not None:
            if hasattr(self, "context") and self.context and self.context.fft_transformer:
                k_pred = self.context.fft_transformer.fftnc(reconstructed, dims=(-2, -1))
                k_diff = (k_pred - y_measured_cached) * cached_mask
                dc_loss = torch.norm(k_diff) ** 2
                loss = loss + self.consistency_weight * dc_loss
                results["dc_loss"] = dc_loss
        else:
            tv_loss = torch.sum(
                torch.abs(reconstructed[:, :, :, :-1] - reconstructed[:, :, :, 1:])
            ) + torch.sum(torch.abs(reconstructed[:, :, :-1, :] - reconstructed[:, :, 1:, :]))
            loss = loss + 0.01 * tv_loss
            results["tv_loss"] = tv_loss

        results["g_total_loss"] = loss
        return results
