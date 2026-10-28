"""Disentangled Diffusion Training Strategy.

Training strategy for physics-guided diffusion models operating in tissue
parameter space. Uses DDPM training with Bloch Data Consistency (DC) correction
during validation sampling.

The tissue parameter space (ρ, T1, T2, T2*, ADC, χ) IS the latent space —
no pre-trained VAE is required. The Bloch equations serve as the physics-based
"decoder" mapping tissue maps to MRI images.

Training loop:
    1. Encode input → tissue maps (via content encoder)
    2. Normalize tissue maps to [0, 1]
    3. Sample timestep, add noise
    4. Predict noise → compute MSE loss
    5. Optional: Bloch consistency loss on denoised prediction

Validation loop:
    Full DDPM reverse sampling with Bloch DC at each step.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model
from spectramr.infrastructure.distributed.owned_modules import OWNED_CALL
from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy, LossResult
from spectramr.models.losses.computers import UnifiedReconstructionLossComputer
from spectramr.models.losses.registry import create_loss

logger = logging.getLogger(__name__)


class DisentangledDiffusionStrategy(BaseTrainingStrategy):
    """Training strategy for tissue-space diffusion models.

    Combines DDPM noise prediction with Bloch equation physics constraints.
    Operates purely in tissue parameter space without requiring a VAE.

    Debug-snapshot contract (CLAUDE.md non-negotiable 14): the model is fed
    ``x_t``, a noised copy of the prepared input, so ``first_steps/input_prepared``
    is NOT the model input. The class says so and emits the real one under
    ``diffusion_step`` (cohort review 2026-09-02, T0.8).
    """

    # parallel_compat: With lambda_bloch_dc > 0 the training loss calls generator.bloch outside
    # the wrapped forward, which the call declaration refuses under ddp. ddpm_sample runs only
    # in validation, where nothing is reduced; FSDP would read a shard there, so no "sharded".
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        model_kwargs = getattr(getattr(config, "model", None), "model_kwargs", None) or {}
        bloch = float(model_kwargs.get("lambda_bloch_dc", 0.1)) > 0
        own = (("generator.bloch", OWNED_CALL),) if bloch else ()
        return (*super().owned_module_declarations(config), *own)

    snapshot_prepared_is_model_input: bool = False
    snapshot_model_input_tag: str | None = "diffusion_step"

    def _setup_strategy_specific_components(self) -> None:
        """Initialize Bloch layer and loss tracking."""
        super()._setup_strategy_specific_components()
        self._bloch_layer = MultiPhysicsBlochLayer().to(self.device)

        # SSOT: Loss computer for noise prediction and reconstruction
        self.loss_computer = UnifiedReconstructionLossComputer(
            config=self.config, device=self.device
        )
        self._noise_criterion = create_loss("mse")

        # Loss weights from config
        model_kwargs = self.config.model.model_kwargs or {}
        self._lambda_noise = float(model_kwargs.get("lambda_noise", 1.0))
        self._lambda_bloch_dc = float(model_kwargs.get("lambda_bloch_dc", 0.1))
        self._bloch_dc_interval = int(model_kwargs.get("bloch_dc_interval", 5))

        logger.info(
            "DisentangledDiffusion setup: noise=%.2f, bloch_dc=%.2f (every %d steps)",
            self._lambda_noise,
            self._lambda_bloch_dc,
            self._bloch_dc_interval,
        )

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Compute losses satisfying the Template Method of BaseTrainingStrategy.

        Delegates to the existing compute_generator_loss logic.
        """
        batch = kwargs.pop("batch", {})
        # Ensure batch has expected keys for compute_generator_loss
        if not isinstance(batch, dict):
            batch = {}
        if "input" not in batch:
            batch["input"] = input_batch
        if "target" not in batch:
            batch["target"] = target_batch

        # Live global iteration from the loop_state seam. The old
        # ``kwargs.get("step", 0)`` read a key the training loop never populates,
        # forcing iteration=0 every step, which made the Bloch-DC throttle
        # (``iteration % self._bloch_dc_interval == 0`` in compute_generator_loss)
        # fire on EVERY step instead of every N — defeating the intended
        # compute-saving interval (review 2026-07-01).
        kwargs["iteration"] = resolve_loop_iteration(self)
        # Remove 'batch_data' to avoid forwarding duplicate keys
        kwargs.pop("batch_data", None)

        loss_result = self.compute_generator_loss(batch, **kwargs)
        losses = loss_result.losses
        if loss_result.g_total_loss is not None:
            losses["g_total_loss"] = loss_result.g_total_loss
        return losses

    def compute_generator_loss(self, batch: dict[str, Any], **kwargs: Any) -> LossResult:
        """Compute diffusion training loss.

        Implements standard DDPM noise-prediction training:
          1. Sample noise ε ~ N(0, I)
          2. Compute noisy input x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
          3. Predict noise: ε_pred = generator(x_t, timesteps=t)
          4. Loss = MSE(ε_pred, ε)

        Args:
            batch: Training batch with input images and metadata.
            **kwargs: Additional arguments (iteration count, etc.).

        Returns:
            LossResult with noise prediction loss and optional Bloch DC loss.
        """
        device = self.device
        generator = self.env.generator
        iteration = kwargs.get("iteration", 0)

        # Extract input image (this is x_0)
        x = batch.get("input", batch.get("source"))
        if x is None:
            msg = "Batch must contain 'input' or 'source' key"
            raise ValueError(msg)
        x = x.to(device, non_blocking=True)

        # Sample random timesteps
        B = x.shape[0]
        n_steps = unwrap_model(generator).n_steps
        t = torch.randint(0, n_steps, (B,), device=device)

        # Compute DDPM schedule (linear beta schedule)
        if not hasattr(self, "_alphas_cumprod"):
            betas = torch.linspace(1e-4, 0.02, n_steps, device=device)
            alphas = 1.0 - betas
            self._alphas_cumprod = torch.cumprod(alphas, dim=0)

        alpha_cumprod_t = self._alphas_cumprod.to(device)[t][:, None, None, None]

        # Step 1: Sample noise
        noise = torch.randn_like(x)

        # Step 2: Create noisy input x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        x_noisy = alpha_cumprod_t.sqrt() * x + (1.0 - alpha_cumprod_t).sqrt() * noise

        # Step 3: Predict noise via generator. Declare the tensor the model is
        # ACTUALLY fed (a dict-of-references stash; the snapshot service owns
        # enabled / interval / budget).
        self._declare_model_input(
            {"x_noisy": x_noisy, "timesteps": t}, extra={"model_input_key": "x_noisy"}
        )
        noise_pred = generator(x_noisy, timesteps=t)

        # Step 4: Primary loss — noise prediction via SSOT
        loss_noise = self._noise_criterion(noise_pred, noise)
        losses = {"noise_mse": loss_noise * self._lambda_noise}
        total = loss_noise * self._lambda_noise

        # Optional: Bloch DC consistency (every N steps for efficiency)
        if self._lambda_bloch_dc > 0 and iteration % self._bloch_dc_interval == 0:
            with torch.no_grad():
                # Reconstruct clean tissue from noise prediction
                x_0_pred = (
                    x_noisy - (1.0 - alpha_cumprod_t).sqrt() * noise_pred.detach()
                ) / alpha_cumprod_t.sqrt()
                x_0_pred = x_0_pred.clamp(0, 1)

            # Synthesize via Bloch (requires tissue map → image)
            inner = self.inner_for_call(generator, "generator.bloch")
            # Fallback (unchanged): without a bloch() the prediction is treated as the image.
            synth = inner.bloch(x_0_pred) if hasattr(inner, "bloch") else x_0_pred

            # Target image (assumed to be the same as input for self-recon)
            target = batch.get("target", x)
            if target is not None:
                target = target.to(device, non_blocking=True)
                # Resize if needed
                if synth.shape != target.shape:
                    synth = F.interpolate(
                        synth,
                        size=target.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                # SSOT: Use loss_computer for Bloch DC loss
                env_losses = self.env.losses if self.env and hasattr(self.env, "losses") else {}
                dc_output = self.loss_computer.compute(
                    pred=synth,
                    target=target,
                    epoch=0,
                    losses_dict=env_losses,
                )
                loss_dc = dc_output.total
                losses["bloch_dc"] = loss_dc * self._lambda_bloch_dc
                total = total + loss_dc * self._lambda_bloch_dc

        return LossResult(
            losses=losses,
            g_total_loss=total,
            metrics={
                # A 0-d tensor, not ``.item()``: the training loop fuses scalar
                # tensors into one host transfer per log step (non-negotiable 9).
                "noise_mse": loss_noise.detach(),
            },
        )

    def compute_discriminator_loss(self, batch: dict[str, Any], **kwargs: Any) -> LossResult:
        """No discriminator for diffusion models.

        Returns zero loss.
        """
        return LossResult(
            losses={},
            g_total_loss=torch.tensor(0.0, device=self.device),
        )

    @accepts_step_io
    def validation_step(
        self,
        batch: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validation step with full DDPM sampling.

        Args:
            batch: Validation batch.
            **kwargs: Additional arguments.

        Returns:
            Dict of validation metrics.
        """
        generator = self.env.generator
        generator.eval()

        x = batch.get("input", batch.get("source"))
        if x is None:
            return {}
        x = x.to(self.device, non_blocking=True)
        target = batch.get("target", x).to(self.device, non_blocking=True)

        B, C, H, W = x.shape

        # Get sequence metadata for Bloch DC
        seq_metadata = {
            "TR": batch.get("TR", 8.0),
            "TE": batch.get("TE", 4.0),
            "TI": batch.get("TI", 0.0),
            "contrast_type": batch.get("contrast_type", "GRE"),
        }

        # Full DDPM sampling with Bloch DC
        # Sampling reads the inner module: a DDP wrapper hides both, and no step follows.
        inner = unwrap_model(generator)
        tissue_shape = (B, inner.tissue_channels, H, W)
        tissue_dict = inner.ddpm_sample(
            shape=tissue_shape,
            device=self.device,
            bloch_layer=self._bloch_layer,
            target_image=target,
            seq_metadata=seq_metadata,
        )

        # Synthesize final image via Bloch
        synth = self._bloch_layer(tissue_dict, seq_metadata)

        # Resize if needed
        if synth.shape != target.shape:
            synth = F.interpolate(
                synth, size=target.shape[2:], mode="bilinear", align_corners=False
            )

        # Compute metrics via SSOT loss computer
        env_losses = self.env.losses if self.env and hasattr(self.env, "losses") else {}
        loss_output = self.loss_computer.compute(
            pred=synth,
            target=target,
            epoch=0,
            losses_dict=env_losses,
        )
        mse = loss_output.total.item()
        psnr = -10 * torch.log10(torch.tensor(mse + 1e-10)).item()

        return {"val_mse": mse, "val_psnr": psnr}
