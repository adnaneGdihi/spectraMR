"""Reconstruction Mixin Module.

This module contains the ReconstructionMixin for handling reconstruction-specific
batch preparation, generator input preparation, and implicit model coordinate generation.
"""

import logging
from typing import Any, cast

import torch
import torch.nn as nn

from spectramr.infrastructure.training.strategies.mixins.utils import (
    _callable_accepts_kwarg,
    pick_present,
)

logger = logging.getLogger(__name__)



def _as_aux_mapping(batch: Any) -> dict[str, Any]:
    """Flatten whatever the loop hands a strategy into a key -> tensor mapping.

    A plain dict passes through. A ``TrainingBatch`` -- what the loop actually
    delivers, since ``BatchAdapter.from_dict`` runs before ``train_step`` -- is
    unpacked to the same shape: its bound fields plus everything ``from_dict``
    parked in ``metadata``. Anything else returns empty, which is the old
    behaviour for an unrecognised batch.
    """
    if isinstance(batch, dict):
        return batch
    fields = ("input", "target", "mask", "coil_maps")
    if not any(hasattr(batch, name) for name in fields):
        return {}
    merged: dict[str, Any] = {}
    metadata = getattr(batch, "metadata", None)
    if isinstance(metadata, dict):
        merged.update(metadata)
    for name in fields:
        value = getattr(batch, name, None)
        if value is not None:
            # `coil_maps` is the dataclass spelling; the key_mapping below reads
            # the dataset spellings, so bind it under one it recognises.
            merged["coil_sensitivities" if name == "coil_maps" else name] = value
    return merged


class ReconstructionMixin:
    """Mixin for reconstruction-specific logic."""

    def _is_implicit_model(self) -> bool:
        """Detect if model is a PURE coordinate-based NeRF/SIREN.

        There are two types of implicit models:
        1. Pure Coordinate NeRF: f(x,y) -> pixel (NO encoder, needs coord grid input)
        2. Encoder-based NeRF (PINNNeRF): f(measurements, coords) -> pixel
           These HAVE an encoder and need the actual measurements as input!

        This method returns True ONLY for pure coordinate NeRFs.
        Encoder-based models like PINNNeRFGenerator handle coords internally.
        """
        if self.env.generator is None:
            return False

        model_name = self.env.generator.__class__.__name__.lower()

        # Check if model has an encoder (image-conditional NeRF)
        # These should receive measurements as input, not coordinates!
        if hasattr(self.env.generator, "encoder"):
            # Model has encoder - it processes measurements, not raw coordinates
            if hasattr(self, "logging_service") and self.logging_service:
                self.logging_service.log_debug(
                    f"[NeRF Detection] {model_name} has encoder - using standard input"
                )
            return False

        # Pure coordinate-based keywords
        pure_coord_keywords = ["siren", "pure_nerf", "coordinate_mlp", "nesvor", "pinn", "implicit"]

        # Check for pure coordinate-based models
        if any(kw in model_name for kw in pure_coord_keywords):
            return True

        # Check config model_type for explicit pure coordinate mode
        config = self.env.config if self.env else getattr(self.state, "config", None)
        model_type = config.model.model_type if config else ""
        if any(kw in str(model_type).lower() for kw in pure_coord_keywords):
            return True

        return False

    def _generate_coordinate_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Generate normalized coordinate grid for NeRF/SIREN input.

        NeRF models require (x, y) coordinates as input, not images.
        This method generates a grid of coordinates in [-1, 1] range.

        Args:
            batch_size: Number of samples in batch
            height: Image height
            width: Image width
            device: Target device

        Returns:
            Coordinate grid [B, 2, H, W] with x in channel 0, y in channel 1
        """
        # Create normalized vectors [-1, 1]
        x = torch.linspace(-1, 1, width, device=device)
        y = torch.linspace(-1, 1, height, device=device)

        # Create meshgrid [H, W]
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")

        # Stack to [2, H, W] (x=channel 0, y=channel 1)
        grid = torch.stack((grid_x, grid_y), dim=0)

        # Expand to batch [B, 2, H, W]
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return grid

    def _prepare_batch_context_reconstruction(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Extract and organize batch context (CC=2 extracted)."""
        batch = {"lr": input_batch, "hr": target_batch}

        # Original kwargs extraction
        for key in (
            "measured_kspace",
            "mask",
            "coil_sensitivities",
            "multimodal_inputs",
            "trajectory",
            "slice_indices",  # Added for MedGS and other folded representations
            "dcf",  # Density Compensation Function
            "sampling_density",  # Alias for DCF
            "field_strength",  # Per-sample B0 for cross-field conditioning (xfield_fm)
            "field_strength_target",  # Target field for cross-field renderers
            "contrast_id",  # Per-sample contrast index for contrast-conditioned models
        ):
            if key in kwargs:
                batch[key] = kwargs[key]

        # [PIPELINE FIX] Extract auxiliary tensors from raw TorchIO/Dataset subject
        # which is usually passed down as kwargs['batch']
        #
        # The loop converts the collated dict to a ``TrainingBatch`` BEFORE
        # calling ``train_step`` (``training_loop.py``: ``BatchAdapter.from_dict``),
        # and that dataclass is not a Mapping -- so the ``isinstance(dict)`` test
        # below was False on the production path and this whole extraction was
        # dead. EquivariantImagingStrategy is where it surfaced: it needs
        # ``mask`` and ``measured_kspace``, the loader produces both, and the
        # strategy raised "Ensure the dataset/director provides both" on the
        # first step of every run. Flatten it back to the shape this reads:
        # ``from_dict`` binds input/target/mask/coil-maps as fields and puts
        # every other key in ``metadata``, so nothing is lost either way.
        raw_batch = _as_aux_mapping(kwargs.get("batch", {}))
        if raw_batch and isinstance(raw_batch, dict):
            # Map canonical model kwargs to TorchIO/Dataset keys
            # (e.g. models usually expect 'coil_sensitivities', but dataset provides 'sensitivity')
            key_mapping = {
                # `sensitivity_complex` FIRST. The subject builder's second
                # branch stores `sensitivity` as `sens_tensor.abs()`, and a
                # magnitude map is not merely lossy for a phase-sensitive
                # consumer -- `I - s s^H` with real `s` is a real projector, so
                # it silently computes something else. Prefer the complex
                # spelling and fall back only where no complex map exists.
                "coil_sensitivities": [
                    "sensitivity_complex",
                    "sensitivity",
                    "coil_sensitivities",
                ],
                "measured_kspace": ["kspace", "measured_kspace"],
                "mask": ["mask"],
                "trajectory": ["trajectory"],
                "slice_indices": ["slice_indices"],
                "dcf": ["dcf", "sampling_density"],
                "field_strength": ["field_strength", "b0", "field"],
                "field_strength_target": ["field_strength_target"],
                # Domain conditioning axes consumed by AdaptiveConditioner via
                # _domain_conditioning_context. `contrast_id` also accepts the
                # `contrast_idx` alias emitted by m4raw_dataset / slice_dataset /
                # multi-contrast collation. Without these the advertised recon
                # conditioning sources silently no-op (pitfall #16).
                "contrast_id": ["contrast_id", "contrast_idx"],
                "scanner_id": ["scanner_id"],
                "site_id": ["site_id"],
                # SCAS scout view (emitted by ScoutAcquisitionTransform as a
                # distinct 'scout' image) — propagate it so the SCAS hypernet
                # in _prepare_generator_inputs can read batch_context['scout']
                # (audit 2026-06: previously no layer produced this key, so the
                # scout-conditioned density penalty was dead every batch).
                "scout": ["scout"],
            }

            for ctx_key, raw_keys in key_mapping.items():
                if ctx_key not in batch:
                    for raw_key in raw_keys:
                        if raw_key in raw_batch:
                            val = raw_batch[raw_key]
                            # Handle TorchIO/Monai nested 'data' dictionaries
                            if hasattr(val, "data") and isinstance(val.data, torch.Tensor):
                                val = val.data
                            elif (
                                isinstance(val, dict)
                                and "data" in val
                                and isinstance(val["data"], torch.Tensor)
                            ):
                                val = val["data"]

                            # Move to device if it's a tensor
                            if isinstance(val, torch.Tensor):
                                val = val.to(input_batch.device, non_blocking=True)

                            batch[ctx_key] = val
                            if hasattr(self, "logging_service") and self.logging_service:
                                self.logging_service.log_debug(
                                    f"Successfully extracted auxiliary tensor '{raw_key}' for context '{ctx_key}'"
                                )
                            break

        # Check if input is k-space based on config
        config = self.env.config if self.env else getattr(self.state, "config", None)
        input_type = (
            config.model.input_type if (config and hasattr(config.model, "input_type")) else "image"
        )
        if input_type == "kspace" and "measured_kspace" not in batch:
            batch["measured_kspace"] = input_batch

        batch["use_dc"] = (
            getattr(self, "dc_layer", None) is not None
            and batch.get("measured_kspace") is not None
            and batch.get("mask") is not None
        )
        return batch

    def _prepare_generator_inputs_reconstruction(
        self,
        batch_context: dict[str, Any],
        input_batch: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Prepare input tensors and forward kwargs (CC=4 extracted)."""
        multimodal_inputs = batch_context.get("multimodal_inputs")
        measured_kspace = batch_context.get("measured_kspace")
        use_dc = batch_context["use_dc"]

        # Handle multimodal inputs
        if multimodal_inputs is not None:
            input_tensors = list(multimodal_inputs.values())
            lr_image = torch.cat(input_tensors, dim=1)
        # NeRF/Implicit models: generate coordinate grid instead of using image
        elif self._is_implicit_model():
            # Get target shape from hr in batch_context or input_batch
            hr = batch_context.get("hr")
            if hr is not None:
                B, C, H, W = hr.shape
            else:
                B, _, H, W = input_batch.shape  # Fallback to input_batch shape

            # Generate coordinate grid [B, 2, H, W]
            lr_image = self._generate_coordinate_grid(B, H, W, input_batch.device)

            if hasattr(self, "logging_service") and self.logging_service:
                self.logging_service.log_debug(
                    f"[NeRF] Generated coordinate grid for implicit model: {lr_image.shape}"
                )
        else:
            lr_image = input_batch

        # Convert k-space to image if needed
        if use_dc and multimodal_inputs is None:
            kspace_for_input = pick_present(measured_kspace, input_batch)
            trajectory = batch_context.get("trajectory")
            dcf = batch_context.get("dcf")

            if trajectory is not None:
                # Handle Non-Cartesian Trajectories (Spiral/Radial)
                # Standard ifft2c fails for (B, C, N) inputs. We must use NUFFT Adjoint.
                try:
                    import torchkbnufft as tkbn
                except ImportError:
                    raise ImportError("torchkbnufft is required for non-Cartesian reconstruction.")

                # Ensure dimensions are correct for tkbn
                # Handle potential extra batch dimension from collation (B, 1, ...)
                if kspace_for_input.ndim == 4 and kspace_for_input.shape[1] == 1:
                    kspace_for_input = kspace_for_input.squeeze(1)

                if trajectory is not None and trajectory.ndim == 4 and trajectory.shape[1] == 1:
                    trajectory = trajectory.squeeze(1)

                dcf = batch_context.get("dcf")
                if dcf is not None and dcf.ndim == 3 and dcf.shape[1] == 1:
                    dcf = dcf.squeeze(1)

                # kspace: (B, C, N) -> (B, C, N) [OK]
                # traj: (B, N, 2) -> (B, 2, N) [Transpose needed if loaded as (N, 2)]
                # traj: (B, 2, N) [OK]
                if trajectory.ndim == 3 and trajectory.shape[-1] == 2:
                    ktraj = trajectory.transpose(1, 2)  # (B, 2, N)
                else:
                    ktraj = trajectory

                # Get expected image size
                im_size = (256, 256)
                if batch_context.get("hr") is not None:
                    im_size = batch_context["hr"].shape[-2:]
                elif hasattr(self.env.generator, "im_size"):
                    im_size = self.env.generator.im_size

                # Check cache for adjoint operator
                if not hasattr(self, "_cached_adj_op") or self._cached_adj_op.im_size != im_size:
                    self._cached_adj_op = tkbn.KbNufftAdjoint(im_size=im_size, device=self.device)

                # Apply Density Compensation Function if available
                # This pre-weights k-space for better conditioning
                # dcf already retrieved above
                if dcf is not None:
                    # Ensure broadcasting: kspace (B, C, N) * dcf (B, 1, N)
                    if dcf.ndim == 2:
                        dcf = dcf.unsqueeze(1)
                    kspace_for_input = kspace_for_input * dcf

                # Perform Adjoint Operation -> (B, C, H, W)
                # tkbn handles batching if trajectory interacts correctly
                lr_image = self._cached_adj_op(kspace_for_input, ktraj)

                if hasattr(self, "logging_service") and self.logging_service:
                    self.logging_service.log_debug(
                        f"Performed NUFFT Adjoint: kspace {kspace_for_input.shape} -> image {lr_image.shape}"
                    )

            else:
                # Cartesian Fallback
                # Since we are in a mixin, we expect self to have fft_transformer
                # But BaseTrainingStrategy does NOT guarantee fft_transformer attribute.
                # reconstruction.py sets it in __init__.
                # We should check for it safely.
                fft_transformer = getattr(self, "fft_transformer", None)
                if fft_transformer:
                    lr_image = fft_transformer.ifft2c(kspace_for_input)
                else:
                    # Fallback to creating one if not present (defensive)
                    from spectramr.infrastructure.training.utils.transform_ops import (
                        FFTTransformer,
                    )

                    transformer = FFTTransformer(device=self.device)
                    lr_image = transformer.ifft2c(kspace_for_input)

        # Track if we reshapped slices (for reshaping auxiliary tensors)
        num_slices = 1
        was_reshaped = False

        # Handle multi-slice k-space data
        # Use explicit config flag instead of shape-based inference
        config = self.config
        model_in_channels = config.model.in_channels if hasattr(config.model, "in_channels") else 2
        multislice_enabled = config.data.multislice_enabled
        is_multislice = (
            multislice_enabled
            and model_in_channels == 2
            and lr_image.ndim == 4
            and lr_image.shape[1] > 2
            and lr_image.shape[1] % 2 == 0
        )

        if is_multislice:
            # Multi-slice k-space data: (B, num_slices*2, H, W)
            # where channels are [slice1_real, slice1_imag, ...]
            # Reshape to (B*num_slices, 2, H, W) for per-slice processing
            B, channels, H, W = lr_image.shape
            num_slices = channels // 2
            was_reshaped = True
            # Reshape: (B, num_slices*2, H, W) -> (B, num_slices, 2, H, W)
            # -> (B*num_slices, 2, H, W)
            # [PERF] Use contiguous().view() per debugger.md Rule 11
            lr_image = lr_image.contiguous().view(B, num_slices, 2, H, W)
            # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
            lr_image = lr_image.contiguous().view(B * num_slices, 2, H, W)
            if hasattr(self, "logging_service") and self.logging_service:
                self.logging_service.log_debug(
                    f"Multi-slice reshape: input (B={B}, channels={channels}, "
                    f"H={H}, W={W}) -> output (B={B * num_slices}, 2, {H}, {W}), "
                    f"num_slices={num_slices}"
                )
        # Handle complex k-space data: (B, num_slices, H, W) complex
        # -> (B*num_slices, 2, H, W) real/imag
        elif model_in_channels == 2 and torch.is_complex(lr_image) and lr_image.ndim == 4:
            # (B, num_slices, H, W) complex -> (B, num_slices, H, W, 2)
            real_imag = torch.view_as_real(lr_image)
            B, num_slices_complex, H, W, _ = real_imag.shape
            num_slices = num_slices_complex
            was_reshaped = True
            # Permute to (B, num_slices, 2, H, W) then reshape
            # [PERF] Use contiguous().view() per debugger.md Rule 11
            real_imag = real_imag.permute(0, 1, 4, 2, 3)  # (B, num_slices, 2, H, W)
            lr_image = real_imag.contiguous().view(B * num_slices, 2, H, W)

        # If we reshaped the main input, also reshape auxiliary tensors to match
        if was_reshaped and num_slices > 1:
            # Reshape measured_kspace if present
            if measured_kspace is not None:
                # measured_kspace shape: (B, C, H, W) or (B, H, W)
                # Only reshape if it can be divided by num_slices
                if measured_kspace.ndim == 4:
                    B_orig, C, H_k, W_k = measured_kspace.shape
                    # Only reshape if C is divisible by num_slices
                    # (accounts for multi-coil data)
                    if C % num_slices == 0:
                        C_per_slice = C // num_slices
                        # (B, C, H, W) -> (B*num_slices, C_per_slice, H, W)
                        # [PERF] Use contiguous().view() per debugger.md Rule 11
                        measured_kspace = measured_kspace.contiguous().view(
                            B_orig, num_slices, C_per_slice, H_k, W_k
                        )
                        # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
                        measured_kspace = measured_kspace.contiguous().view(
                            B_orig * num_slices, C_per_slice, H_k, W_k
                        )
                        batch_context["measured_kspace"] = measured_kspace
                    else:
                        pass
                elif measured_kspace.ndim == 3:
                    # (B, H, W) -> (B*num_slices, H, W) by repeating for each slice
                    B_orig, H_k, W_k = measured_kspace.shape
                    measured_kspace = measured_kspace.repeat_interleave(num_slices, dim=0)
                    batch_context["measured_kspace"] = measured_kspace

            # Reshape mask if present
            if batch_context.get("mask") is not None:
                mask = batch_context["mask"]
                # mask shape: (B, H, W) or (B, 1, H, W)
                if mask.ndim == 4 and mask.shape[1] == 1:
                    # (B, 1, H, W) -> (B*num_slices, 1, H, W)
                    # by repeating for each slice
                    B_orig = mask.shape[0]
                    mask = mask.repeat(num_slices, 1, 1, 1)
                    batch_context["mask"] = mask
                elif mask.ndim == 3:
                    # (B, H, W) -> (B*num_slices, H, W) by repeating for each slice
                    B_orig = mask.shape[0]
                    mask = mask.repeat(num_slices, 1, 1)
                    batch_context["mask"] = mask

            # Reshape coil_sensitivities if present
            if batch_context.get("coil_sensitivities") is not None:
                coil_sens = batch_context["coil_sensitivities"]
                # coil_sens shape: (B, C, H, W)
                if coil_sens.ndim == 4:
                    # (B, C, H, W) -> (B*num_slices, C, H, W)
                    # by repeating for each slice
                    B_orig = coil_sens.shape[0]
                    C = coil_sens.shape[1]
                    coil_sens = coil_sens.repeat(num_slices, 1, 1, 1)
                    batch_context["coil_sensitivities"] = coil_sens

            # Reshape hr if present
            if batch_context.get("hr") is not None:
                hr = batch_context["hr"]
                # hr shape: (B, C_hr, H, W)
                if hr.ndim == 4:
                    B_orig, C_hr, H_hr, W_hr = hr.shape
                    if C_hr == num_slices:
                        # Case: HR has 1 channel per slice (e.g. magnitude)
                        # (B, num_slices, H, W) -> (B, num_slices, 1, H, W)
                        # -> (B*num_slices, 1, H, W)
                        hr = hr.unsqueeze(2)  # (B, num_slices, 1, H, W)
                        hr = hr.contiguous().view(B_orig * num_slices, 1, H_hr, W_hr)
                        batch_context["hr"] = hr
                    elif C_hr == num_slices * 2:  # Assuming 2 output channels per slice?
                        # (B, num_slices*2, H, W) -> (B, num_slices, 2, H, W)
                        # -> (B*num_slices, 2, H, W)
                        hr = hr.contiguous().view(B_orig, num_slices, 2, H_hr, W_hr)
                        # [PERF B10] Removed no-op permute(0,1,2,3,4) (identity permutation)
                        hr = hr.contiguous().view(B_orig * num_slices, 2, H_hr, W_hr)
                        batch_context["hr"] = hr

        # Build forward kwargs
        forward_kwargs: dict[str, Any] = {}
        generator = cast(nn.Module, self.env.generator)
        generator_forward = getattr(generator, "forward", None)

        if generator_forward is not None:
            if measured_kspace is not None and _callable_accepts_kwarg(
                generator_forward, "kspace_measured"
            ):
                forward_kwargs["kspace_measured"] = batch_context.get(
                    "measured_kspace", measured_kspace
                )
            if batch_context.get("mask") is not None and _callable_accepts_kwarg(
                generator_forward, "mask"
            ):
                forward_kwargs["mask"] = batch_context["mask"]
            if batch_context.get("coil_sensitivities") is not None and _callable_accepts_kwarg(
                generator_forward, "sensitivity_maps"
            ):
                forward_kwargs["sensitivity_maps"] = batch_context["coil_sensitivities"]
            if batch_context.get("trajectory") is not None and _callable_accepts_kwarg(
                generator_forward, "k_trajectory"
            ):
                forward_kwargs["k_trajectory"] = batch_context["trajectory"]

            # Inject slice_indices if model requires it (e.g., for MedGS folded Gaussians)
            if batch_context.get("slice_indices") is not None and _callable_accepts_kwarg(
                generator_forward, "slice_indices"
            ):
                forward_kwargs["slice_indices"] = batch_context["slice_indices"]

            # Inject timesteps if model requires it (e.g. for time-conditioned reconstruction or LPD)
            if _callable_accepts_kwarg(generator_forward, "timesteps"):
                # Generate random timesteps [0, 1]
                # Use batch size from lr_image
                B = lr_image.shape[0]
                timesteps = torch.rand((B,), device=lr_image.device)
                forward_kwargs["timesteps"] = timesteps

            # Inject field conditioning for cross-field renderers (AnatomyFieldRenderer
            # & friends) that REQUIRE ``field_strength`` as a keyword-only arg. Prefer
            # the TARGET field (render at the HF field), fall back to the source field.
            # Signature-gated so non-field models are untouched, and never defaulted
            # (a default field would be a silent-fallback, pitfall #9). Without this,
            # a field-conditioned generator routed through the GENERIC reconstruction
            # validation path (e.g. the calibration strategy, which has no field-aware
            # ``_validation_forward``) raised "missing 1 required keyword-only argument:
            # 'field_strength'" → zero successful validation batches (CLAUDE.md #10;
            # the mrixfields b17_dice_risk_calibration crash).
            if _callable_accepts_kwarg(generator_forward, "field_strength"):
                fs = batch_context.get("field_strength_target", batch_context.get("field_strength"))
                if fs is not None:
                    forward_kwargs["field_strength"] = fs
            if batch_context.get("contrast_id") is not None and _callable_accepts_kwarg(
                generator_forward, "contrast_id"
            ):
                forward_kwargs["contrast_id"] = batch_context["contrast_id"]

        return lr_image, forward_kwargs
