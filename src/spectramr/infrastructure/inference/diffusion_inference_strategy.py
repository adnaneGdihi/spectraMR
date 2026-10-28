"""Diffusion Inference Strategy

Inference strategy for diffusion models (DDPM, cold diffusion, etc.).
Handles iterative sampling and noise schedules.
"""

import logging
from typing import Any

import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec
from spectramr.infrastructure.inference.strided_sampling import ddim_step, strided_timesteps
from spectramr.infrastructure.physics.data_consistency import HardDataConsistency
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import (
    DiffusionScheduler,
)

from .base_inference_strategy import BaseInferenceStrategy
from .multi_modal_inference import MultiModalInferenceMixin
from .performance_optimization import InferencePerformanceOptimizer

logger = logging.getLogger(__name__)


def _resolve_diffusion_config(config: Any) -> dict:
    """Resolve the diffusion sub-config, preferring the ``training.diffusion`` SSOT.

    The top-level ``diffusion:`` key is deprecated; trained models carry their
    schedule under ``training.diffusion``. Without this fallback the strategy
    silently used the hardcoded defaults (1000 linear steps, 256², 1 channel)
    regardless of how the model was trained — meaningless output for a model
    trained at e.g. 28 cosine steps (CLAUDE.md pitfalls #15/#16).

    The training schema uses different key names, so the diverging ones are
    mapped into this strategy's inference vocabulary:
    ``sampling_steps``/``timesteps`` → ``num_inference_steps`` and
    ``noise_schedule`` → ``beta_schedule``. A schedule the strategy cannot run
    (e.g. ``sqrt``) is left to fail loudly downstream rather than silently
    degrade to linear.
    """
    cfg = config or {}
    top = cfg.get("diffusion") if isinstance(cfg, dict) else getattr(cfg, "diffusion", None)
    if top:
        return dict(top)

    training = cfg.get("training") if isinstance(cfg, dict) else getattr(cfg, "training", None)
    training = training or {}
    td = (
        training.get("diffusion")
        if isinstance(training, dict)
        else getattr(training, "diffusion", None)
    )
    if td is None:
        return {}
    if not isinstance(td, dict):
        td = td.model_dump() if hasattr(td, "model_dump") else dict(vars(td))

    resolved = dict(td)
    if "num_inference_steps" not in resolved:
        steps = resolved.get("sampling_steps") or resolved.get("timesteps")
        if steps is not None:
            resolved["num_inference_steps"] = steps
    if "beta_schedule" not in resolved and resolved.get("noise_schedule") is not None:
        ns = resolved["noise_schedule"]
        resolved["beta_schedule"] = getattr(ns, "value", ns)  # enum → str
    return resolved


class DiffusionInferenceStrategy(BaseInferenceStrategy, MultiModalInferenceMixin):
    """Inference strategy for diffusion models.

    Handles iterative sampling for diffusion models trained with
    denoising objectives (DDPM, cold diffusion, flow matching, etc.).
    """

    #: The sampler normalises its input and de-normalises the sample itself.
    owns_kspace_normalization: bool = True

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: dict[str, Any] | None = None,
    ):
        """Initialize diffusion inference strategy with validation.

        Args:
            model: The diffusion model (denoiser)
            device: Device to run inference on
            config: Configuration dictionary

        Raises:
            ValueError: If configuration parameters are invalid
            TypeError: If model is not a valid nn.Module
        """
        if not isinstance(model, nn.Module):
            raise TypeError(f"model must be nn.Module, got {type(model)}")

        super().__init__(model, device, config)

        # Validate mixin initialization - with lenient error handling
        try:
            MultiModalInferenceMixin.__init__(self)
        except AttributeError:
            # This is expected if device is not set from parent class yet
            # The mixin will still work, but may not have full functionality
            pass
        except Exception as e:
            logger.warning(f"Mixin initialization warning: {e!s}")

        self.diffusion_config = _resolve_diffusion_config(self.config)

        # Initialize performance optimizer
        self.performance_optimizer = InferencePerformanceOptimizer(device, self.diffusion_config)

        # Sampling parameters with validation
        self.num_inference_steps = self._validate_param(
            self.diffusion_config.get("num_inference_steps", 1000),
            "num_inference_steps",
            min_val=1,
            max_val=10000,
            param_type=int,
        )

        self.guidance_scale = self._validate_param(
            self.diffusion_config.get("guidance_scale", 1.0),
            "guidance_scale",
            min_val=0.0,
            max_val=100.0,
            param_type=float,
        )

        self.eta = self._validate_param(
            self.diffusion_config.get("eta", 0.0),
            "eta",
            min_val=0.0,
            max_val=1.0,
            param_type=float,
        )

        # Advanced sampling parameters with validation
        self.use_classifier_free_guidance = self.diffusion_config.get("use_cfg", True)
        if not isinstance(self.use_classifier_free_guidance, bool):
            raise ValueError(f"use_cfg must be bool, got {type(self.use_classifier_free_guidance)}")

        self.dynamic_thresholding = self.diffusion_config.get("dynamic_thresholding", False)
        if not isinstance(self.dynamic_thresholding, bool):
            raise ValueError(
                f"dynamic_thresholding must be bool, got {type(self.dynamic_thresholding)}"
            )

        self.thresholding_percentile = self._validate_param(
            self.diffusion_config.get("thresholding_percentile", 0.95),
            "thresholding_percentile",
            min_val=0.0,
            max_val=1.0,
            param_type=float,
        )

        # Noise schedule parameters with validation
        self.beta_start = self._validate_param(
            self.diffusion_config.get("beta_start", 0.0001),
            "beta_start",
            min_val=0.0,
            max_val=0.1,
            param_type=float,
        )

        self.beta_end = self._validate_param(
            self.diffusion_config.get("beta_end", 0.02),
            "beta_end",
            min_val=0.0,
            max_val=0.1,
            param_type=float,
        )

        # Validate beta_start < beta_end
        if self.beta_start >= self.beta_end:
            raise ValueError(f"beta_start ({self.beta_start}) must be < beta_end ({self.beta_end})")

        self.beta_schedule = self.diffusion_config.get("beta_schedule", "linear")
        valid_schedules = ["linear", "cosine", "sigmoid"]
        if self.beta_schedule not in valid_schedules:
            raise ValueError(
                f"beta_schedule must be one of {valid_schedules}, got {self.beta_schedule}"
            )

        # The schedule is the one the model was trained on; sampling visits a
        # strided subsequence of its indices. A schedule rebuilt at the sampling
        # length named different noise levels by the same indices (#2305). A
        # legacy top-level `diffusion:` block has no training length, and its
        # num_inference_steps describes the schedule itself.
        self.num_train_timesteps = self._validate_param(
            self.diffusion_config.get("timesteps", self.num_inference_steps),
            "timesteps",
            min_val=1,
            max_val=10000,
            param_type=int,
        )
        self.sampling_timesteps = strided_timesteps(
            self.num_train_timesteps, self.num_inference_steps
        )
        self.scheduler = DiffusionScheduler(
            num_timesteps=self.num_train_timesteps,
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            device=self.device,
        )

        # Map scheduler attributes for compatibility
        self.betas = self.scheduler.betas
        self.alphas = self.scheduler.alphas
        self.alphas_cumprod = self.scheduler.alphas_cumprod
        self.sqrt_alphas_cumprod = self.scheduler.sqrt_alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = self.scheduler.sqrt_one_minus_alphas_cumprod

        # Legacy attribute (unused but kept for safety if external code accesses)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]]
        )

        # [MRI] Initialize Physics Layers
        self.dc_layer = None
        physics_config = self.config.get("physics", {}) if self.config else {}
        dc_config = physics_config.get("data_consistency", {})
        if dc_config.get("enabled", False) and dc_config.get("method") == "hard":
            # ``self.logger`` was never bound on this class, so every arm that
            # declared enabled+hard raised AttributeError here at construction.
            logger.info("Initializing Hard Data Consistency for Inference")
            self.dc_layer = HardDataConsistency()

        # [MRI] Initialize Normalizer from the SSOT spec — the SAME knobs the
        # training transform used (kspace_percentile / log_scaling /
        # kspace_scale_domain). This used to read ``normalization_kwargs``, the
        # IMAGE-normalization block, and never applied ``log_scaling``, so the
        # model saw a different distribution than it trained on (issue #572).
        data_config = self.config.get("data", {}) if self.config else {}
        self.kspace_norm = KSpaceNormalizationSpec.from_data_config(data_config)
        self.kspace_normalizer = self.kspace_norm if self.kspace_norm.enabled else None

    def _validate_param(
        self,
        value: Any,
        name: str,
        min_val: float | None = None,
        max_val: float | None = None,
        param_type: type = float,
    ) -> Any:
        """Validate a configuration parameter.

        Args:
            value: Parameter value
            name: Parameter name for error messages
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            param_type: Expected parameter type

        Returns:
            Validated parameter value

        Raises:
            ValueError: If parameter is invalid
            TypeError: If parameter has wrong type
        """
        if not isinstance(value, param_type):
            raise TypeError(f"{name} must be {param_type.__name__}, got {type(value).__name__}")

        if min_val is not None and value < min_val:
            raise ValueError(f"{name} must be >= {min_val}, got {value}")

        if max_val is not None and value > max_val:
            raise ValueError(f"{name} must be <= {max_val}, got {value}")

        return value

    @property
    def training_mode(self) -> TrainingModeTypes:
        """Return the training mode this strategy handles."""
        return TrainingModeTypes.DIFFUSION

    # _initialize_noise_schedule removed (handled by DiffusionScheduler)

    def _apply_classifier_free_guidance(
        self,
        conditional_noise_pred: torch.Tensor,
        unconditional_noise_pred: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Apply classifier-free guidance to combine conditional and unconditional predictions.

        Args:
            conditional_noise_pred: Noise prediction from conditional model
            unconditional_noise_pred: Noise prediction from unconditional model
            guidance_scale: Strength of guidance (1.0 = no guidance, >1.0 = stronger guidance)

        Returns:
            Guided noise prediction
        """
        return unconditional_noise_pred + guidance_scale * (
            conditional_noise_pred - unconditional_noise_pred
        )

    def _apply_dynamic_thresholding(
        self,
        x_t: torch.Tensor,
        threshold_percentile: float = 0.95,
    ) -> torch.Tensor:
        """Apply dynamic thresholding to prevent saturation.

        Args:
            x_t: Current sample at timestep t
            threshold_percentile: Percentile for thresholding

        Returns:
            Thresholded sample
        """
        # Compute dynamic threshold based on percentile
        # Flatten spatial dimensions to compute percentile across batch
        abs_x = torch.abs(x_t)
        threshold = torch.quantile(
            abs_x.contiguous().view(abs_x.shape[0], -1),
            threshold_percentile,
            dim=1,
            keepdim=True,
        )
        threshold = threshold.unsqueeze(-1).unsqueeze(-1)  # Reshape back to (B, 1, 1, 1)

        # Clamp values above threshold
        x_t = torch.clamp(x_t, -threshold, threshold)

        # Renormalize to maintain scale
        max_val = torch.amax(torch.abs(x_t), dim=[1, 2, 3], keepdim=True)
        x_t = x_t * (threshold / max_val.clamp(min=1e-8))

        return x_t

    def _predict_noise(
        self,
        x_t: torch.Tensor,
        t: int,
        conditioning: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Predict noise for the current timestep using performance optimizations.

        Args:
            x_t: Current sample at timestep t
            t: Current timestep
            conditioning: Conditioning tensor (if any)
            **kwargs: Additional prediction parameters

        Returns:
            Predicted noise
        """
        # Handle classifier-free guidance
        if (
            self.use_classifier_free_guidance
            and conditioning is not None
            and self.guidance_scale != 1.0
        ):
            # `efficient_predict_noise` OWNS the concat -- it does
            # `torch.cat([x_t, conditioning], dim=1)` itself. So pass the two
            # tensors through their own parameters and never pre-concatenate
            # here: this branch used to build `cat([x_t, conditioning])` and
            # hand it over AS `conditioning`, which the callee concatenated a
            # second time, so the model received [x_t, x_t, conditioning] --
            # 3C channels instead of 2C, silently (#1030).
            if conditioning is not None:
                conditional_noise_pred = self.performance_optimizer.run_optimized_inference(
                    self.model, x_t, t, conditioning
                )

                # The unconditional leg drops the condition's INFORMATION while
                # keeping its shape, so both legs present identical geometry to
                # the model -- that is what makes the guidance difference valid.
                unconditional_noise_pred = self.performance_optimizer.run_optimized_inference(
                    self.model, x_t, t, torch.zeros_like(conditioning)
                )

                # Apply classifier-free guidance
                noise_pred = self._apply_classifier_free_guidance(
                    conditional_noise_pred,
                    unconditional_noise_pred,
                    self.guidance_scale,
                )
            else:
                # Fallback to unconditional if no conditioning
                noise_pred = self.performance_optimizer.run_optimized_inference(
                    self.model, x_t, t, None
                )
        else:
            # Standard prediction (conditional or unconditional)
            if conditioning is not None:
                # `x_t` is the sample the reverse loop is building; `conditioning`
                # is the measurement. Both go through their own parameters, and
                # `efficient_predict_noise` concatenates them once (#1030).
                #
                # This previously read `run_optimized_inference(model,
                # conditioning, t, None)` -- conditioning in the SAMPLE slot --
                # under a comment saying the concat "doubles channels and causes
                # mismatch". It does double them, by design: the model for a
                # conditional arm takes 2C in-channels. Substituting the
                # conditioning for `x_t` silenced that shape error by removing
                # the very input the algorithm is defined on, so every timestep
                # saw the same fixed tensor and the DDPM recursion became a map
                # of a constant. A model whose `in_channels` is C now raises at
                # inference instead, which is the correct failure: a shape error
                # is a config bug you can see, a frozen recursion is not.
                noise_pred = self.performance_optimizer.run_optimized_inference(
                    self.model, x_t, t, conditioning
                )
            else:
                noise_pred = self.performance_optimizer.run_optimized_inference(
                    self.model, x_t, t, None
                )

        # Handle different model output formats
        if isinstance(noise_pred, dict):
            noise_pred = noise_pred.get("noise_pred", noise_pred.get("output", noise_pred))

        return noise_pred

    def preprocess_input(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Preprocess input for diffusion inference.

        For diffusion models, input might be conditioning information
        or initial noise for unconditional generation.

        Args:
            input_tensor: Raw input tensor (conditioning or noise)
            **kwargs: Additional preprocessing parameters

        Returns:
            Preprocessed tensor for diffusion sampling
        """
        # Ensure tensor is on correct device
        if input_tensor.device != self.device:
            input_tensor = input_tensor.to(self.device)

        input_tensor = input_tensor.detach()

        # For conditional diffusion, input_tensor is conditioning
        # For unconditional, we might need to generate noise
        if kwargs.get("unconditional", False):
            # Generate random noise for unconditional sampling. in_channels and
            # image_size are SHAPE-critical: a wrong default (1 / 256) silently
            # produces mis-shaped noise the trained model cannot consume (garbage
            # or a downstream shape crash), so require them explicitly rather than
            # defaulting (pitfall #9 — no silent fallback for a shape-critical value).
            batch_size = kwargs.get("batch_size", 1)
            missing = [k for k in ("in_channels", "image_size") if k not in self.diffusion_config]
            if missing:
                raise ValueError(
                    "Unconditional diffusion sampling requires "
                    f"{missing} in the diffusion config (no silent 1/256 default). "
                    f"Configured keys: {sorted(self.diffusion_config.keys())}"
                )
            channels = int(self.diffusion_config["in_channels"])
            height = width = int(self.diffusion_config["image_size"])

            input_tensor = torch.randn(batch_size, channels, height, width, device=self.device)

        return input_tensor

    def run_inference(self, input_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run diffusion sampling with advanced features.

        Args:
            input_tensor: Conditioning tensor or initial noise
            **kwargs: Additional inference parameters

        Returns:
            Sampled output tensor
        """
        # [MRI] Normalize Input/Conditioning
        # We need to capture scale to denormalize later
        norm_scale = None
        if self.kspace_normalizer is not None:
            # If conditioning is provided (e.g. undersampled kspace), normalize it
            if input_tensor is not None and kwargs.get("conditional", False):
                # channel_dim=1: (B, C, H, W) with real/imag interleaved along C.
                input_tensor, norm_scale = self.kspace_norm.normalize(input_tensor, channel_dim=1)

            # Also normalize initial_noise if provided (rare but possible)
            if kwargs.get("initial_noise") is not None:
                init_noise = kwargs.get("initial_noise")
                # Reuse scale from conditioning if available, else compute new (risky if different)
                # Usually initial_noise is a noisy version of conditioning or random
                # If random, it is already N(0,1), so no normalization needed vs normalized data.
                # If it is an image/kspace, normalize.
                # Heuristic: if it looks like data (not N(0,1)), normalize.
                pass

        # Optimize model for inference
        self.performance_optimizer.optimize_model(self.model)

        # Extract parameters
        batch_size = kwargs.get("batch_size", 1)
        conditional = kwargs.get("conditional", False)
        initial_noise = kwargs.get("initial_noise")

        # Determine conditioning and x_t
        if conditional:
            # Conditional sampling: input_tensor is conditioning
            conditioning = input_tensor
            # Start from noise or provided initial noise
            if initial_noise is not None:
                x_t = initial_noise
            else:
                input_batch_size = conditioning.shape[0]  # Use actual batch size from input
                channels = self.diffusion_config.get("in_channels", 1)
                height = self.diffusion_config.get("image_size", 256)
                width = self.diffusion_config.get("image_size", 256)

                x_t = torch.randn(input_batch_size, channels, height, width, device=self.device)
        else:
            # Unconditional sampling: input_tensor is initial noise
            x_t = input_tensor if initial_noise is None else initial_noise
            conditioning = None

        # Prepare for inference with expected dimensions
        channels = self.diffusion_config.get("in_channels", 1)
        height = self.diffusion_config.get("image_size", 256)
        width = self.diffusion_config.get("image_size", 256)
        conditioning_channels = (
            conditioning.shape[1] if conditional and conditioning is not None else None
        )

        self.performance_optimizer.prepare_for_inference(
            self.model, batch_size, channels, height, width, conditioning_channels
        )

        # Iterative denoising over the strided training indices (DDIM; eta=0
        # deterministic, eta=1 the DDPM posterior variance).
        own_dc_applied = False
        visits = self.sampling_timesteps
        for i, t in enumerate(visits):
            noise_pred = self._predict_noise(x_t, t, conditioning, **kwargs)

            if self.dynamic_thresholding:
                x_t = self._apply_dynamic_thresholding(x_t, self.thresholding_percentile)

            alpha_bar_prev = self.alphas_cumprod[visits[i + 1]] if i + 1 < len(visits) else 1.0
            x_t = ddim_step(x_t, noise_pred, self.alphas_cumprod[t], alpha_bar_prev, self.eta)

            # [MRI] Apply Data Consistency
            # We assume conditioning IS the measured k-space for Cold Diffusion
            if self.dc_layer is not None and conditioning is not None:
                # Attempt to get mask from kwargs
                mask = kwargs.get("mask")

                # If mask missing but we have conditioning (measured kspace),
                # we can define mask as non-zeros of conditioning (assuming zero-filled)
                # But safer to require mask.
                if mask is not None:
                    # Enforce consistency
                    # Note: x_t is current estimate. kspace_obs is conditioning.
                    # We assume x_t is in the same domain as conditioning (K-Space).
                    # Check if we should warn about domain?
                    # HardDataConsistency handles domain check inside, but we need to tell it
                    # if we are in kspace domain to avoid unnecessary FFTs.
                    # Cold Diffusion -> pure kspace -> is_kspace_domain=True

                    # We can infer is_kspace_domain from config or context
                    is_kspace = self.diffusion_config.get("force_pure_kspace", False)
                    # Or heuristic:
                    if not is_kspace and torch.is_complex(x_t):
                        is_kspace = True

                    x_t = self.dc_layer(
                        reconstruction=x_t,
                        kspace_obs=conditioning,
                        mask=mask,
                        is_kspace_domain=is_kspace,
                    )
                    own_dc_applied = True

        # The loop pinned the measured bins itself, so the predict-time hook in
        # ``BaseInferenceStrategy.infer`` must not project a second time.
        if own_dc_applied:
            self.note_data_consistency_applied("DiffusionInferenceStrategy.run_inference")

        # [MRI] Denormalize output: bring sample back to the physical scale of
        # the input. Without this, reconstructions are off by 1/scale (typically
        # 10-100x). See findings booklet 2026-05-05 N-1.
        if self.kspace_normalizer is not None and norm_scale is not None:
            # Decompresses the log1p before rescaling when log_scaling is on.
            x_t = self.kspace_norm.denormalize(x_t, norm_scale, channel_dim=1)

        return x_t

    def postprocess_output(self, output_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """Postprocess diffusion model output.

        Args:
            output_tensor: Raw sampled output
            **kwargs: Additional postprocessing parameters

        Returns:
            Postprocessed output tensor
        """
        # Ensure output is detached
        output_tensor = output_tensor.detach().cpu()

        # Apply final activation if specified
        final_activation = self.diffusion_config.get("final_activation", "")
        if final_activation.lower() == "tanh":
            output_tensor = torch.tanh(output_tensor)
        elif final_activation.lower() == "sigmoid":
            output_tensor = torch.sigmoid(output_tensor)

        # Normalize to [0, 1] range
        if self.diffusion_config.get("normalize_output", True):
            output_tensor = (output_tensor + 1.0) / 2.0

        # Clamp outputs to valid range
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)

        return output_tensor

    def get_strategy_info(self) -> dict[str, Any]:
        """Get information about this diffusion inference strategy."""
        info = super().get_strategy_info()
        perf_stats = self.performance_optimizer.get_performance_stats()
        info.update(
            {
                "diffusion_config": self.diffusion_config,
                "num_inference_steps": self.num_inference_steps,
                "num_train_timesteps": self.num_train_timesteps,
                "guidance_scale": self.guidance_scale,
                "beta_schedule": self.beta_schedule,
                "use_classifier_free_guidance": self.use_classifier_free_guidance,
                "dynamic_thresholding": self.dynamic_thresholding,
                "thresholding_percentile": self.thresholding_percentile,
                "performance_stats": perf_stats,
            }
        )
        return info

    def text_to_image_inference(self, text_prompt: str | list[str], **kwargs) -> torch.Tensor:
        """Run text-to-image inference using diffusion sampling.

        Args:
            text_prompt: Text prompt(s) for generation
            **kwargs: Additional inference parameters

        Returns:
            Generated image tensor
        """
        # Encode text conditioning
        text_conditioning = self.text_conditioning.encode_text(text_prompt)

        # Run diffusion sampling with text conditioning
        return self.run_inference(text_conditioning, **kwargs)

    def image_to_image_inference(
        self,
        source_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        strength: float = 0.8,
        **kwargs,
    ) -> torch.Tensor:
        """Run image-to-image inference using diffusion sampling.

        Args:
            source_image: Source image for editing
            mask: Optional mask for inpainting
            strength: Strength of conditioning (0-1)
            **kwargs: Additional inference parameters

        Returns:
            Edited image tensor
        """
        # Encode image conditioning
        image_conditioning = self.image_conditioning.encode_image(source_image, mask)

        # Add noise to source image based on strength
        if strength < 1.0:
            noise = torch.randn_like(source_image)
            # Add noise inversely proportional to strength
            noise_level = 1.0 - strength
            noisy_image = source_image * strength + noise * noise_level
        else:
            noisy_image = source_image

        # Run diffusion sampling with image conditioning
        return self.run_inference(image_conditioning, initial_noise=noisy_image, **kwargs)

    @torch.no_grad()
    def infer_single(self, input_data: torch.Tensor, **kwargs) -> torch.Tensor:
        """Run inference on a single input tensor.

        Args:
            input_data: Single input tensor
            **kwargs: Additional parameters

        Returns:
            Output tensor
        """
        return self.infer(input_data, **kwargs)

    @torch.no_grad()
    def infer_batch(
        self, input_data_list: list[torch.Tensor], batch_size: int = 4, **kwargs
    ) -> list[torch.Tensor]:
        """Run inference on a batch of input tensors.

        Args:
            input_data_list: List of input tensors
            batch_size: Batch size for processing
            **kwargs: Additional parameters

        Returns:
            List of output tensors
        """
        results = []
        for input_data in input_data_list:
            result = self.infer_single(input_data, **kwargs)
            results.append(result)
        return results
