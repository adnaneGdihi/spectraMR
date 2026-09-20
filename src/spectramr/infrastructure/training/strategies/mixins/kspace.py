"""
K-Space Mixin Module

This module contains the KspaceMixin for K-Space domain handling and Cold Diffusion logic.
"""

import logging
from typing import TYPE_CHECKING, Any

import torch

from spectramr.data.batch_types import align_scale_to_batch, read_batch_field
from spectramr.data.transforms.normalization import KSpaceNormalizationSpec
from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.utils.data_adapters import TorchIOAdapter
from spectramr.infrastructure.training.utils.transform_ops import FFTTransformer

from .utils import _get_config_value, pick_present

if TYPE_CHECKING:
    from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class KspaceMixin:
    # NOTE: no ``applies_undersampling`` here. ``BaseTrainingStrategy`` mixes this
    # class in for EVERY strategy, and ``generate_and_process_mask`` below is
    # called only by the diffusion strategy; the flag belongs on the classes
    # that call it (cohort review 2026-09-02, T0.6).

    """Mixin for K-Space domain handling and Cold Diffusion logic."""

    def _prepare_model_input(
        self: "BaseTrainingStrategy",
        data: Any,
    ) -> torch.Tensor:
        """Prepare input for model, handling domain adaptation if needed."""
        # 1. Handle Dictionary Input
        if isinstance(data, dict):
            if "input" in data:
                return data["input"]
            if "kspace" in data:
                return data["kspace"]
            return list(data.values())[0]

        # 2. Handle Tensor Input
        input_batch = data

        model_domain = _get_config_value(self.config, "model.model_domain", "image").lower()

        is_input_kspace = False
        if isinstance(input_batch, torch.Tensor):
            if input_batch.is_complex():
                # Truly complex-typed tensor — definitely k-space
                is_input_kspace = True
            elif (
                input_batch.dim() == 4
                and input_batch.shape[1] % 2 == 0
                and _get_config_value(self.config, "model.input_type", "image").lower() == "kspace"
            ):
                # Real-stacked complex k-space ``(R0, I0, R1, I1, ...)`` has an
                # EVEN channel count: 2 for a single (virtual) coil, 8 for the
                # 4-coil M4Raw ``experiment_11`` cohort, 16 for cross_contrast.
                # The layout is ambiguous from dtype/shape alone (could be
                # real/imag k-space or an even-channel image), so we trust the
                # config: classify as k-space iff ``input_type == "kspace"``.
                # This prevents a double-FFT when the data pipeline already
                # produces k-space real/imag channels.
                #
                # 2026-06-28 doubled-brain fix: the guard previously required
                # ``shape[1] == 2``, so the 8/16-ch multi-coil arms were
                # misclassified as image and ``image_to_kspace`` (``fft2c``)
                # applied a SECOND forward FFT. Because ``F{F{img}} = img(-x)``,
                # the model's "prepared k-space" became a 180-deg-rotated image
                # (Parseval-preserved std, spread DC) while validation did not
                # double-FFT — the train/val mismatch behind the whole cohort's
                # doubled/garbled outputs.
                is_input_kspace = True

        if (model_domain == "kspace" and is_input_kspace) or (
            model_domain == "image" and not is_input_kspace
        ):
            return input_batch

        if not hasattr(self, "_domain_transformer"):
            self._domain_transformer = FFTTransformer(device=self.device)

        if model_domain == "kspace" and not is_input_kspace:
            kspace = self._domain_transformer.image_to_kspace(input_batch)
            if kspace.is_complex():
                if kspace.dim() == 4:
                    kspace = torch.view_as_real(kspace).permute(0, 1, 4, 2, 3).contiguous()
                    B, C, D, H, W = kspace.shape
                    kspace = kspace.view(B, C * D, H, W)
                elif kspace.dim() == 3:
                    kspace = torch.view_as_real(kspace).permute(0, 3, 1, 2).contiguous()
            return kspace

        elif model_domain == "image" and is_input_kspace:
            image = self._domain_transformer.kspace_to_image(input_batch)
            if image.is_complex():
                if image.dim() == 4:
                    image = torch.view_as_real(image).permute(0, 1, 4, 2, 3).contiguous()
                    B, C, D, H, W = image.shape
                    image = image.view(B, C * D, H, W)
                elif image.dim() == 3:
                    image = torch.view_as_real(image).permute(0, 3, 1, 2).contiguous()
            return image

        return input_batch

    def _is_cold_diffusion(self: "BaseTrainingStrategy") -> bool:
        """Check if current model is a k-space cold diffusion model."""
        return (
            hasattr(self.config, "model")
            and hasattr(self.config.model, "model_type")
            and "kspace_cold_diffusion" in str(self.config.model.model_type).lower()
        )

    def setup_kspace_components(self: "BaseTrainingStrategy") -> None:
        """Initialize K-Space specific components (Mask Generator, Data Consistency).

        Took a ``num_timesteps`` argument until the mask generator moved to the
        model (#2056). The process was built from the same
        ``training.diffusion.timesteps``, so the parameter became a second way
        to state one value and nothing read it (non-negotiable 8).
        """
        if not self._is_cold_diffusion():
            return

        generator = self.env.generator
        if hasattr(generator, "dc_layer") and generator.dc_layer is not None:
            self.dc_layer = generator.dc_layer
            if hasattr(self, "logging_service"):
                self.logging_service.log_info(
                    f"🧲 Data Consistency: Using model-integrated {type(self.dc_layer).__name__}"
                )
        else:
            self.dc_layer = None

            # Check physics config (SSOT v6.0)
            is_dc_enabled = False
            if hasattr(self.config, "physics") and self.config.physics is not None:
                # Direct access for Pydantic schema
                is_dc_enabled = getattr(self.config.physics.data_consistency, "enabled", False)

            if hasattr(self, "logging_service") and is_dc_enabled:
                self.logging_service.log_warning(
                    "🧲 Data Consistency: Enabled in config but NOT found in model. Skipping strategy-side DC."
                )

        # The mask generator is BORROWED from the model, never rebuilt (#2056).
        # Both sides resolved the same ``undersampling:`` block through the same
        # allowlist, so the two instances agreed on every accelerator kwarg --
        # which is why the duplication survived: it showed only as
        # ``Creating Accelerator`` logged twice per run. What a strategy-owned
        # copy can never share is the knobs the ACCELERATOR does not carry:
        # ``accelerator_kwargs_from_config`` drops ``enable_dynamic_mask`` by
        # design, because the per-sample seed jitter lives in
        # ``KSpaceUndersamplingProcess.q_sample``. One object, and
        # ``self.training`` decides whether the pattern jitters.
        unwrapped = generator.module if hasattr(generator, "module") else generator
        process = getattr(unwrapped, "kspace_process", None)
        if process is None:
            raise ValueError(
                f"{self.config.model.model_type!r} routes through the cold-diffusion "
                f"setup, but its generator {type(unwrapped).__name__} exposes no "
                f"'kspace_process' to take the mask generator from. Building a second "
                f"one here is what #2056 removed: it would resolve the same "
                f"'undersampling:' block through a second code path nothing compares "
                f"against, and would miss every knob the accelerator does not carry."
            )
        self.mask_generator = process.mask_generator

        pattern = self.mask_generator.default_pattern
        accelerator_kwargs = self.mask_generator._accelerator_kwargs

        if pattern == "multi_mask" and hasattr(self, "logging_service"):
            self.logging_service.log_warning(
                "Using 'multi_mask' pattern - ensure KSpaceMaskGenerator supports it",
                model_type=self.config.model.model_type,
            )

        if hasattr(self, "logging_service"):
            # ``density_power`` is flattened into the kwargs, not nested under a
            # ``schedule_kwargs`` key — reading it from a nested dict reported
            # "default" for every arm that set it.
            self.logging_service.log_info(
                f"🔧 Acceleration config: type={pattern}, "
                f"density_power={accelerator_kwargs.get('density_power', 'default')}, "
                f"max_accel={accelerator_kwargs.get('max_acceleration', 'default')}, "
                f"seed={accelerator_kwargs.get('seed', 'default')}"
            )

    def _prepare_validation_data(
        self: "BaseTrainingStrategy",
        batch: Any,
        input_batch: torch.Tensor | None,
        target_batch: torch.Tensor | None,
        batch_data: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare and normalize validation data tensors."""
        if input_batch is None or target_batch is None:
            lr, hr = self._unpack_batch(batch)
            input_batch = lr if input_batch is None else input_batch
            target_batch = hr if target_batch is None else target_batch

        in_channels = (
            self.config.model.in_channels if hasattr(self.config.model, "in_channels") else 2
        )
        input_batch = TorchIOAdapter.to_batch_format(input_batch, in_channels)
        target_batch = TorchIOAdapter.to_batch_format(target_batch, in_channels)

        # [FIX] Handle 5D tensors from dataloader (B, C, H, W, D) → (B*D, C, H, W)
        if input_batch.dim() == 5:
            b, c, h, w, d = input_batch.shape
            if hasattr(self, "logging_service"):
                self.logging_service.log_info(
                    f"[5D→4D RESHAPE] Detected 5D input_batch: ({b}, {c}, {h}, {w}, {d})"
                )

            # [FIX] Apply RepetitionFusion (if generator supports it) BEFORE flattening.
            # During training the 5D path goes through rep_fusion inside the generator.
            # During validation the generator is called with 4D input, so we must fuse here.
            gen = getattr(self, "generator_model", None)
            if gen is not None and hasattr(gen, "module"):
                gen = gen.module
            rep_fusion = None
            if gen is not None:
                rep_fusion = getattr(gen, "rep_fusion", None)
                if rep_fusion is None:
                    backbone = getattr(gen, "backbone", None)
                    if backbone is not None:
                        rep_fusion = getattr(backbone, "rep_fusion", None)

            if rep_fusion is not None and d > 1:
                # input_batch: (B, C, H, W, D) – D is repetitions / slices
                # rep_fusion expects (B, C, H, W, Reps) and returns (B, C, H, W)
                input_batch = rep_fusion(input_batch)
                if hasattr(self, "logging_service"):
                    self.logging_service.log_info(
                        f"[RepFusion] Applied rep_fusion: ({b},{c},{h},{w},{d}) → {tuple(input_batch.shape)}"
                    )
                if target_batch.dim() == 5:
                    # Target: collapse repetitions by taking the first (fully sampled) rep
                    target_batch = target_batch[..., 0]
            else:
                # Fallback: naive flatten (B, C, H, W, D) → (B*D, C, H, W)
                input_batch = input_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                if target_batch.dim() == 5:
                    target_batch = target_batch.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

        input_batch = input_batch.to(self.device, non_blocking=True)
        target_batch = target_batch.to(self.device, non_blocking=True)

        scale_factor = torch.ones(input_batch.size(0), 1, 1, 1, device=input_batch.device)

        # Channel_adapter auto-rebuilds for any input channel count — no correction needed here.

        if self.config.data.processing.enable_kspace_normalization:
            # Mapping protocol, not isinstance-dict + hasattr -- a TrainingBatch
            # fails both legs and its metadata is unreachable by attribute
            # lookup. Reading the published scale as absent sent this call into
            # the ``else`` branch below, which divides input AND target a second
            # time by a freshly recomputed quantile -- precisely what the
            # branch's own comment forbids ("Do not divide again!").
            scale_from_batch = read_batch_field(batch_data, "kspace_scale")

            if scale_from_batch is not None:
                # The published scale is per SUBJECT, but ``input_batch`` may
                # already be per SLICE: ``train.py._preprocess_validation_tensor``
                # folds depth into the batch axis before the strategy is called,
                # so neither 5D branch above can fire and no expansion has
                # happened. Reconcile against the tensor rather than trusting the
                # producer's length -- the old ``view(-1, 1, 1, 1)`` adopted it
                # blindly and a length-2 scale reached a length-36 prediction.
                scale_factor = align_scale_to_batch(
                    scale_from_batch,
                    input_batch.size(0),
                    field="kspace_scale",
                    device=input_batch.device,
                )
                # Dataloader has ALREADY normalized the tensors. Do not divide again!
            else:
                # Physics-compliant normalization: compute scale from input only
                # to prevent gain jitter in metrics computation.
                # Scale is derived per-sample from the 99th percentile of input magnitude.
                #
                # F8/E28 (2026-05-21 smoke audit): the prior code assumed an
                # interleaved real-stacked complex layout (``input_batch[:,
                # 0:1]`` real, ``[:, 1:2]`` imag). For 1-channel data (PINN /
                # implicit-fields / graph-diffusion family), ``[:, 1:2]``
                # slices PAST the channel axis and returns shape
                # ``(B, 0, H, W)`` — the magnitude reshape then has 0 elements
                # per sample, and ``torch.quantile`` raises
                # ``input tensor must be non-empty``. 6 experiments crashed
                # this way (experiment_12_pinn_reconstruction,
                # experiment_39_pin_inr_mri_implicit_fields,
                # experiment_42b_graph_diffusion, experiment_46_pinn_inverse_solving,
                # experiment_60_implicit_neural, experiment_pipeline_b_quantum_nerf).
                #
                # Dispatch by channel count:
                #   1 ch              → |input|
                #   2 ch (R+iI)       → sqrt(R^2 + I^2)  [original behavior]
                #   N ch even (multi) → RSS of (R, I) coil pairs
                #   N ch odd          → channel-RSS (no complex semantics)
                c = input_batch.shape[1]
                if c == 1:
                    input_mag = input_batch.abs().reshape(input_batch.size(0), -1)
                elif c == 2:
                    input_mag = torch.sqrt(
                        input_batch[:, 0:1] ** 2 + input_batch[:, 1:2] ** 2
                    ).reshape(input_batch.size(0), -1)
                elif c % 2 == 0:
                    # Pair as coils, magnitude per coil, then RSS.
                    re_ = input_batch[:, 0::2]
                    im_ = input_batch[:, 1::2]
                    coil_mags = torch.sqrt(re_**2 + im_**2 + 1e-12)
                    input_mag = torch.sqrt((coil_mags**2).sum(dim=1, keepdim=True)).reshape(
                        input_batch.size(0), -1
                    )
                else:
                    # Odd channels — multi-contrast or single-coil
                    # multi-channel. Channel-RSS (no complex semantics).
                    input_mag = torch.sqrt((input_batch**2).sum(dim=1, keepdim=True)).reshape(
                        input_batch.size(0), -1
                    )

                # Final guard: an empty input still flunks quantile. Skip
                # normalization rather than raise — caller can still train.
                if input_mag.numel() == 0:
                    return input_batch, target_batch, scale_factor

                percentile_vals = torch.quantile(input_mag, 0.99, dim=1)
                scale_factor = percentile_vals.clamp(min=1e-8).view(input_batch.size(0), 1, 1, 1)
                input_batch = input_batch / scale_factor
                target_batch = target_batch / scale_factor

        return input_batch, target_batch, scale_factor

    def _detect_expected_input_channels(self: "BaseTrainingStrategy") -> int | None:
        """Detect actual input channels expected by the generator model.

        Inspects the generator's channel_adapter (lazily initialized on first training
        batch) to determine how many input channels it was initialized with.
        Falls back to checking backbone conv layers.

        Returns:
            Expected number of input channels, or None if cannot be determined
        """
        try:
            gen = self.generator_model
            if gen is None:
                return None

            # Unwrap DataParallel / DistributedDataParallel
            if hasattr(gen, "module"):
                gen = gen.module

            # Primary path: KSpaceColdDiffusionGenerator wraps a FourierBridgeNetwork
            # backbone that lazily initializes its channel_adapter on the first forward.
            # The flag `_channel_adapter_initialized` and the `channel_adapter` attribute
            # live on the backbone (FourierBridgeNetwork), not on the outer generator.
            backbone = getattr(gen, "backbone", None)
            if backbone is not None and getattr(backbone, "_channel_adapter_initialized", False):
                ch_adapter = getattr(backbone, "channel_adapter", None)
                if ch_adapter is not None:
                    # ChannelAdapter stores in_channels directly
                    if hasattr(ch_adapter, "in_channels"):
                        expected = int(ch_adapter.in_channels)
                        if expected > 0:
                            return expected
                    # Fallback: read from inner adapter conv weight
                    inner = getattr(ch_adapter, "adapter", None)
                    if inner is not None and hasattr(inner, "weight"):
                        expected = int(inner.weight.shape[1])
                        if expected > 0:
                            return expected

            # Also check if channel_adapter lives on the outer generator (some variants)
            if hasattr(gen, "channel_adapter") and getattr(
                gen, "_channel_adapter_initialized", False
            ):
                ch_adapter = gen.channel_adapter
                if hasattr(ch_adapter, "in_channels"):
                    expected = int(ch_adapter.in_channels)
                    if expected > 0:
                        return expected
                inner = getattr(ch_adapter, "adapter", None)
                if inner is not None and hasattr(inner, "weight"):
                    expected = int(inner.weight.shape[1])
                    if expected > 0:
                        return expected

            # Secondary path: backbone first convolution
            backbone = getattr(gen, "backbone", None)
            if backbone is not None:
                for attr in ("conv_in", "input_proj", "patch_embed"):
                    conv = getattr(backbone, attr, None)
                    if conv is not None and hasattr(conv, "weight"):
                        expected = int(conv.weight.shape[1])
                        if expected > 0:
                            return expected

            # Tertiary path: top-level generator conv
            for attr in ("conv_in", "input_proj"):
                conv = getattr(gen, attr, None)
                if conv is not None and hasattr(conv, "weight"):
                    expected = int(conv.weight.shape[1])
                    if expected > 0:
                        return expected

            # Quaternary path: UNet encoder first block (for KSpaceColdDiffusionGenerator)
            # The first encoder block may contain the 1x1 adjustment conv that expects certain channels
            if hasattr(gen, "backbone") and hasattr(gen.backbone, "backbone"):
                # FourierBridgeNetwork wraps a UNet
                unet = gen.backbone.backbone
                if hasattr(unet, "encoder") and len(unet.encoder) > 0:
                    # First encoder block
                    first_encoder = unet.encoder[0]
                    # Look for any conv layer that shows input channel requirements
                    if hasattr(first_encoder, "double_conv"):
                        double_conv = first_encoder.double_conv
                        if hasattr(double_conv, "conv1") and hasattr(double_conv.conv1, "weight"):
                            expected = int(double_conv.conv1.weight.shape[1])
                            if expected > 0:
                                return expected
                        elif hasattr(double_conv, "0") and hasattr(double_conv[0], "weight"):
                            expected = int(double_conv[0].weight.shape[1])
                            if expected > 0:
                                return expected
                    elif hasattr(first_encoder, "0"):
                        # Sequential first layer
                        layer0 = first_encoder[0]
                        if hasattr(layer0, "weight"):
                            expected = int(layer0.weight.shape[1])
                            if expected > 0:
                                return expected

            # Fallback: Check config.model.in_channels (Single Source of Truth)
            # This ensures validation uses the same channel count as training
            if hasattr(self, "config") and hasattr(self.config, "model"):
                if hasattr(self.config.model, "in_channels"):
                    config_in_channels = int(self.config.model.in_channels)
                    if config_in_channels > 0:
                        return config_in_channels

            return None
        except Exception as e:
            if hasattr(self, "logging_service"):
                self.logging_service.log_debug(f"Could not detect expected input channels: {e}")
            return None

    def apply_kspace_normalization(
        self: "BaseTrainingStrategy",
        input_batch: torch.Tensor | None,
        target_batch: torch.Tensor,
        current_step: int = 0,
    ) -> tuple[torch.Tensor | None, torch.Tensor, Any]:
        """Apply robust k-space normalization."""
        # Fallback path: only reached when the dataloader did NOT publish a
        # ``kspace_scale`` (normally KSpaceNormalizationTransform does). Resolve
        # from the SSOT so the scale matches what the transform would have
        # produced -- this used to read ``normalization_kwargs`` (the IMAGE
        # normalization block, a different percentile) and never applied
        # ``log_scaling`` (issue #572).
        if not hasattr(self, "kspace_normalizer"):
            self.kspace_normalizer = KSpaceNormalizationSpec.from_data_config(self.config.data)
            # Reaching this function means a caller decided the batch needs
            # normalizing (it read `enable_kspace_normalization` itself). A spec
            # that resolves DISABLED here therefore means the two readers of the
            # same declaration disagree -- and `normalize()` is a silent no-op
            # when disabled, so the disagreement would return the raw batch with
            # a unit scale while reporting success. That is exactly how
            # experiment_11_attention_none trained on raw k-space: this resolver
            # read flat pre-decomposition names that no schema still carries, so
            # it answered False while the caller read the block and said True.
            if not self.kspace_normalizer.enabled:
                raise RuntimeError(
                    "[kspace-norm] apply_kspace_normalization was called, but "
                    "KSpaceNormalizationSpec.from_data_config resolved "
                    "enabled=False. The caller and the resolver disagree about "
                    "data.processing.enable_kspace_normalization; normalizing "
                    "would be a silent no-op returning the RAW batch and a unit "
                    "scale. Fix the resolver rather than letting the model train "
                    "on un-normalized k-space (CLAUDE.md #3)."
                )

        scale = None
        try:
            if input_batch is not None:
                input_batch, scale = self.kspace_normalizer.normalize(input_batch, channel_dim=1)
                target_batch, _ = self.kspace_normalizer.normalize(
                    target_batch, scale=scale, channel_dim=1
                )
            else:
                if getattr(self, "logging_service", None) is not None:
                    self.logging_service.log_warning(
                        "apply_kspace_normalization: input_batch is None; "
                        "computing scale from target_batch"
                    )
                target_batch, scale = self.kspace_normalizer.normalize(target_batch, channel_dim=1)
                input_batch = None

        except Exception as e:
            # CLAUDE.md #3/#10: k-space normalization is correctness-critical.
            # The old code swallowed any failure and returned the ORIGINAL
            # un-normalized tensors (scale=None), and its only warning was
            # gated on ``_strategy_logger`` — an attribute never set anywhere
            # in the strategies tree, so the failure was fully silent. Surface
            # it instead of training on un-normalized k-space.
            if getattr(self, "logging_service", None) is not None:
                self.logging_service.log_error(
                    f"[Step {current_step}] k-space normalization failed: {e}"
                )
            raise

        return input_batch, target_batch, scale

    def generate_and_process_mask(
        self: "BaseTrainingStrategy",
        batch_size: int,
        timesteps: torch.Tensor,
        target_shape: tuple[int, ...],
        current_step: int = 0,
        batch_data: dict | None = None,
    ) -> torch.Tensor:
        """Generate and process acceleration mask for cold diffusion."""
        mask = None
        if batch_data is not None and isinstance(batch_data, dict):
            mask = pick_present(batch_data.get("mask"), batch_data.get("acceleration_mask"))

        # [FIX] Handle 5D mask from dataloader (B, 1, H, W, D) → (B*D, 1, H, W)
        if mask is not None and mask.dim() == 5:
            b, c, h, w, d = mask.shape
            # Flatten: permute (B, C, H, W, D) → (B, D, C, H, W) → reshape to (B*D, C, H, W)
            mask = mask.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
            logger.debug(f"[MASK] Flattened 5D mask to {mask.shape}")

        if mask is None:
            # Detect spatial dimensions based on target_shape
            # For 4D (B, C, H, W): use last two
            # For 5D (B, C, H, W, D): use middle two (assuming 2D slice processing)
            if len(target_shape) == 5:
                image_shape = (target_shape[2], target_shape[3])
            else:
                image_shape = (target_shape[-2], target_shape[-1])

            pattern = None  # Could default from config

            mask = self.mask_generator.generate_batch_masks(
                batch_size=batch_size,
                timesteps=timesteps,
                image_shape=image_shape,
                pattern=pattern,
            )

            # Logging
            if (
                hasattr(self, "_cached_log_interval")
                and current_step % self._cached_log_interval == 0
            ):
                try:
                    sampled_points = mask.sum().item()
                    total_points = mask.numel()
                    calc_accel = total_points / (sampled_points + 1e-6)

                    active_pattern = pattern or self.mask_generator.default_pattern
                    if hasattr(
                        self, "logging_service"
                    ) and self.logging_service.logger.isEnabledFor(logging.INFO):
                        self.logging_service.log_info(
                            "🧊 Cold Diff Sampling [Step %d] | Mask Pattern: %s | Timesteps: [%d - %d] | Eff. Accel: %.2fx",
                            current_step,
                            active_pattern,
                            timesteps.min().item(),
                            timesteps.max().item(),
                            calc_accel,
                        )
                except Exception as _exc:
                    logger.debug("Suppressed exception: %s", _exc)

        mask = mask.to(self.device, non_blocking=True).float()

        # Phase 4.1: Asymmetric Degradation Masking for TI-CCD
        # We want the mask to only degrade the Target channels.
        # Source channels should always have a mask of 1.0 (fully sampled).
        C_total = target_shape[1]

        # Determine the number of target channels based on domain (complex vs real-interleaved)
        # If C_total is 16 (for 2 contrasts, 4 coils, interleaved real/imag), target is 8
        # We dynamically assume the second half of channels belong to Target
        # based on TI-CCD design (concatenated [Source, Target])

        # First, ensure the mask itself is expanded to C_total
        if mask.shape[1] != C_total:
            mask = self.mask_generator.expand_mask_to_channels(mask, C_total)

        # Apply asymmetric logic if we're dealing with multi-contrast concatenated input
        # Standard configs shouldn't be affected if they only have exactly the Target channels
        # But for TI-CCD where C_total == 16, the first 8 are source, last 8 are target
        #
        # [ROBUSTNESS FIX] Use config.data.domain.target_channels (SSOT) to determine the
        # source/target split boundary instead of assuming C_total // 2.
        target_ch = self.config.data.domain.target_channels
        source_channels = 0
        if target_ch is not None and C_total > target_ch:
            source_channels = C_total - target_ch  # e.g. 16 - 8 = 8 source channels
        elif C_total > self.config.model.out_channels and C_total % 2 == 0:
            # Legacy fallback: assume equal split
            source_channels = C_total // 2

        if source_channels > 0:
            # [#1917] This is the ONLY in-place write to a mask that came out of
            # ``expand_mask_to_channels``; the other ten call sites just multiply.
            # Copy on this path alone, for two independent reasons:
            #   1. a widened [B, 1, H, W] mask is a stride-0 broadcast view, so
            #      every channel aliases one row of memory and the write raises
            #      "more than one element of the written-to tensor refers to a
            #      single memory location";
            #   2. a mask that already had C_total channels is returned unchanged,
            #      and the ``.to(device).float()`` above is a no-op for an
            #      already-float tensor on the same device -- so the write would
            #      reach through and corrupt ``batch_data["mask"]`` for every
            #      later consumer of that batch.
            # Copying here keeps the cheap broadcast view for the read-only
            # callers (non-negotiable 9: no needless allocation in the loop).
            mask = mask.clone()
            # Force mask of 1.0 on the Source channels (preserve fully-sampled prior)
            mask[:, :source_channels, ...] = 1.0

        # Saturation check. Gate on the LIVE iteration (loop_state seam): the
        # old ``self.env.step`` was a frozen 0, so ``0 in [0, 1, 2]`` was always
        # true and this DEBUG warning fired on every step (pitfall #16).
        if (
            hasattr(self, "logging_service")
            and self.logging_service.logger.isEnabledFor(logging.DEBUG)
            and resolve_loop_iteration(self) in [0, 1, 2]
        ):
            mask_fraction = mask.mean().item()
            mask_sat_threshold = (
                self.config.training.mask_saturation_threshold
                if hasattr(self.config, "training")
                and hasattr(self.config.training, "mask_saturation_threshold")
                and self.config.training.mask_saturation_threshold is not None
                else 0.95
            )
            if mask_fraction > mask_sat_threshold:
                self.logging_service.logger.warning(
                    "  ⚠️ MASK IS NEARLY ALL ONES! No degradation being applied!"
                )

        return mask
