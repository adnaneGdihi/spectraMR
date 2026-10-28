"""Reconstruction Mixin Module.

This module contains the ReconstructionMixin for handling reconstruction-specific
batch-context preparation and implicit-model coordinate generation. Generator input
preparation has one owner, ``ReconstructionTrainingStrategy._prepare_generator_inputs``.
"""

import logging
from typing import Any

import torch

from spectramr.core.module_utils import unwrap_model

logger = logging.getLogger(__name__)


#: Context key -> the batch keys it is read from, in priority order. Module level
#: so the resolution order is one owner a test can plant a regression against.
AUX_KEY_MAPPING: dict[str, tuple[str, ...]] = {
    # `sensitivity_complex` FIRST. The subject builder's second
    # branch stores `sensitivity` as `sens_tensor.abs()`, and a
    # magnitude map is not merely lossy for a phase-sensitive
    # consumer -- `I - s s^H` with real `s` is a real projector, so
    # it silently computes something else. Prefer the complex
    # spelling and fall back only where no complex map exists.
    "coil_sensitivities": (
        "sensitivity_complex",
        "sensitivity",
        "coil_sensitivities",
    ),
    # The explicit key FIRST: a masking transform writes the measurement
    # there, while `kspace` is the fully sampled reference on both k-space
    # routes, so reading it first trained EI on the reference (#2324).
    "measured_kspace": ("measured_kspace", "kspace"),
    "mask": ("mask",),
    "trajectory": ("trajectory",),
    "slice_indices": ("slice_indices",),
    "dcf": ("dcf", "sampling_density"),
    "field_strength": ("field_strength", "b0", "field"),
    "field_strength_target": ("field_strength_target",),
    # Domain conditioning axes consumed by AdaptiveConditioner via
    # _domain_conditioning_context. `contrast_id` also accepts the
    # `contrast_idx` alias emitted by m4raw_dataset / slice_dataset /
    # multi-contrast collation. Without these the advertised recon
    # conditioning sources silently no-op (pitfall #16).
    "contrast_id": ("contrast_id", "contrast_idx"),
    "scanner_id": ("scanner_id",),
    "site_id": ("site_id",),
    # SCAS scout view (emitted by ScoutAcquisitionTransform as a
    # distinct 'scout' image) — propagate it so the SCAS hypernet
    # in _prepare_generator_inputs can read batch_context['scout']
    # (audit 2026-06: previously no layer produced this key, so the
    # scout-conditioned density penalty was dead every batch).
    "scout": ("scout",),
}


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
            # `coil_maps` is the dataclass spelling; AUX_KEY_MAPPING reads
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
        # Through unwrap_model: a DDP/DP wrapper hides the inner module's attributes.
        if hasattr(unwrap_model(self.env.generator), "encoder"):
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
            for ctx_key, raw_keys in AUX_KEY_MAPPING.items():
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
