"""TorchIO Transform Builder - Phase T1 Refactoring.

Extracts transform pipeline composition logic from ConsolidatedDatasetFactory.

Provides reusable, testable builders for constructing TorchIO transform pipelines
for training and validation, with support for:
- Smart geometric standardization (crop vs resize)
- Physics-aware k-space regeneration
- MRI-specific augmentations (optional)
- Multiple physics modes (Cartesian, Non-Cartesian)
- K-space and image normalization
- Advanced transforms (graph encoding, etc.)

Usage:
    config = TorchIOTransformConfig(
        patch_size=(320, 320),
        trajectory_type="cartesian",
        normalize_kspace=True,
    )
    transforms = TorchIOTransformBuilder.build_train_transforms(config)

    .. mermaid::

        flowchart TD
            Start[Config] --> Spatial{Ensure Spatial Consistency}
            Spatial --> Coil[Coil Processing]
            Coil --> Geom{Standardization Mode}
            Geom -->|Smart| SmartRes[Smart Resize/Crop]
            Geom -->|Strict| StrictRes[Strict Resize]
            Geom -->|None| NoOp

            SmartRes --> Sync1[Physics Sync]
            StrictRes --> Sync1

            Sync1 --> Aug[Augmentations]
            Aug --> Sync2[Physics Sync]

            Sync2 --> Physics{Trajectory Type}
            Physics -->|Cartesian| Mask[PhysicsInformedMasking]
            Physics -->|Radial/Spiral| Sim[NonCartesianSimulation]

            Mask --> Norm[Normalization]
            Sim --> Norm

            Norm --> End[Output Pipeline]
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import torchio as tio

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.data.signal_domain import is_kspace_dataset_type
from spectramr.data.transforms.augmentation_factory import TorchIOAugmentationFactory

# Re-exported by the transform rather than imported from
# ``infrastructure.physics.trajectories`` directly: ``spectramr.data`` -> infrastructure
# is a layer-direction violation, and ``non_cartesian`` already holds the recorded
# physics-SSOT exception for it. This module already imports that transform (lazily, in
# ``_build_physics_transform``), so taking its vocabulary from the same place keeps the
# dependency honest without adding a second entry to ``_known_violations.json``.
from spectramr.data.transforms.non_cartesian import (
    NON_CARTESIAN_TRAJECTORIES,
    TRAJECTORY_TYPES,
)

# Re-export, not a second definition. Both dispatch chains in this file were
# replaced by the ONE resolver in the transforms module (#760), which is also
# where the fail-loud message for an unconstructible ``normalization_type`` now
# lives — so the set the error quotes and the set the resolver honours cannot
# drift. Re-exported here because this is where the schema-Literal bridge test
# looks for it.
from spectramr.data.transforms.normalization import (
    IMPLEMENTED_NORMALIZATION_TYPES as IMPLEMENTED_NORMALIZATION_TYPES,
)
from spectramr.data.transforms.physics_sync import PhysicsSynchronization
from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

logger = logging.getLogger(__name__)


class _NoOpTransform(tio.Transform):
    """No-op transform that returns the subject unchanged.

    Used as a placeholder in transform pipelines when no actual
    transformation is needed but a transform is required.
    """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Return the subject unchanged."""
        return subject


class _SyncSubjectAttributes(tio.Transform):
    """Re-bind ``Subject.__dict__`` to the mapping, as the LAST chain member (#1213).

    :class:`torchio.Subject` is a ``dict`` subclass that mirrors its entries into
    ``self.__dict__``; ``Subject.__setitem__`` is **not defined**, so a bare
    ``subject[key] = value`` reaches ``dict.__setitem__`` and the two views silently
    diverge. ``tio.Crop.apply_transform`` — the engine behind every
    :class:`torchio.data.PatchSampler` and therefore every ``tio.Queue`` — builds its
    output *solely* from ``subject.__dict__``: replaced images come back as their
    **pre-transform** objects and newly-added non-image keys are **dropped outright**.
    A chain can therefore run in full, mutate exactly what it was asked to, and have
    its entire effect discarded at patch extraction.

    ``EnsureSpatialConsistency`` and ``KSpaceNormalizationTransform`` each re-sync at
    their own site (that is where the defect was proven). This member is the chain-level
    backstop for the remaining ~49 ``subject[...] = ...`` sites across
    ``data/transforms/`` and for the dataset's own post-hoc keys, so a future transform
    cannot reintroduce the corruption by omission.

    This is invariant maintenance, not a silent fallback (non-negotiable 3): torchio's
    own ``Subject.add_image`` is literally ``self[name] = image; self.update_attributes()``,
    so the sync *is* the documented contract. It is one ``dict.update`` per subject in
    the dataloader path — nothing in the training loop (non-negotiable 9).
    """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Mirror every mapping entry into ``__dict__`` and return the subject."""
        subject.update_attributes()
        return subject


class _ValDebugStats(tio.Transform):
    """Log per-subject tensor statistics (shape/min/max/mean) for debugging.

    Defined at MODULE scope (not as a local class inside
    :meth:`TorchIOTransformBuilder.build_val_transforms`) so it stays
    picklable: Python 3.14 defaults the multiprocessing start method to
    ``forkserver`` on non-Mac POSIX, which pickles each DataLoader worker's
    Dataset and its transform Compose. A *local* class's qualname contains
    ``<locals>`` and pickle cannot locate it, so val workers crashed with
    ``PicklingError`` on the cluster. The transform itself is a pass-through.
    """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Log stats for the ``input``/``target`` images and return the subject."""
        for k in subject.get_images_names():
            if k in ["input", "target"]:
                data = subject[k].data
                if data.is_complex():
                    mag = data.abs()
                    logger.info(
                        f"[VAL-DEBUG] {k} (magnitude): min={mag.min():.2f}, max={mag.max():.2f}, mean={mag.mean():.2f}"
                    )
                else:
                    logger.info(
                        f"[VAL-DEBUG] {k}: min={data.min():.2f}, max={data.max():.2f}, mean={data.mean():.2f}"
                    )
        return subject


class _ResampleToReferenceTransform(tio.Transform):
    """Resample all images in a Subject to match a reference image's spatial shape.

    Required for paired NIfTI datasets (e.g., ULF 64mT → HF 3T) where the
    input and target volumes have inherently different spatial dimensions.
    TorchIO spatial transforms (RandomAffine, RandomElasticDeformation) call
    ``subject.check_consistent_spatial_shape()`` which fails when shapes differ.

    Strategy:
    1. Pick a reference image (``image`` > ``input`` > first available).
    2. For every other image whose spatial shape differs:
       a. If the affines are close (<10mm origin offset), use ``tio.Resample``
          which respects the affine-encoded spatial relationship.
       b. If the affines differ significantly (unregistered paired data),
          override the non-reference image's affine to match the reference,
          then use ``tio.CropOrPad`` to harmonize shapes. This assumes the
          volumes are roughly centered on the same anatomy (valid for
          brain MRI from the same subject).
    3. No-op when all shapes already match (typical for single-domain data).

    This must execute **after** ``EnsureSpatialConsistency`` (affine fix) and
    **before** any spatial augmentations.
    """

    # Keys tried in order to find the reference image
    _REFERENCE_PRIORITY = ("image", "input", "kspace")

    # Maximum affine origin offset (mm) before we consider volumes unregistered
    _AFFINE_TOLERANCE_MM = 10.0

    def _affine_origins_close(self, affine_a, affine_b) -> bool:
        """Check if two affines have similar origins (translation component)."""
        import numpy as np

        origin_a = np.array(affine_a[:3, 3], dtype=np.float64)
        origin_b = np.array(affine_b[:3, 3], dtype=np.float64)
        offset = float(np.linalg.norm(origin_a - origin_b))
        return offset < self._AFFINE_TOLERANCE_MM

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Resample non-reference images to the reference spatial shape."""
        import numpy as np
        import torch

        image_names = subject.get_images_names()
        if len(image_names) < 2:
            return subject  # Nothing to harmonize

        # --- Find reference image ---
        ref_name = None
        for candidate in self._REFERENCE_PRIORITY:
            if candidate in image_names:
                ref_name = candidate
                break
        if ref_name is None:
            ref_name = image_names[0]

        ref_image = subject[ref_name]
        ref_shape = ref_image.spatial_shape  # (W, H, D)
        ref_affine = ref_image.affine

        # --- Resample mismatched images ---
        for name in image_names:
            if name == ref_name:
                continue
            other_image = subject[name]
            if other_image.spatial_shape == ref_shape:
                # Shapes match — still ensure affines are consistent
                if not self._affine_origins_close(ref_affine, other_image.affine):
                    logger.debug(
                        f"[RESAMPLE] '{name}' same shape but affine mismatch — "
                        f"overriding affine to match '{ref_name}'"
                    )
                    # Must use remove+add to invalidate TorchIO's attribute cache
                    new_img = tio.ScalarImage(
                        tensor=other_image.data,
                        affine=np.array(ref_affine, dtype=np.float64),
                    )
                    subject.remove_image(name)
                    subject.add_image(new_img, name)
                continue

            affines_close = self._affine_origins_close(ref_affine, other_image.affine)

            if affines_close:
                # Affines are compatible — use tio.Resample which respects
                # the spatial relationship encoded in the affines.
                logger.debug(
                    f"[RESAMPLE] '{name}' {other_image.spatial_shape} → "
                    f"'{ref_name}' {ref_shape} (affine-aware resample)"
                )
                resampler = tio.Resample(target=ref_image, include=[name])
                subject = resampler(subject)
            else:
                # Affines are NOT compatible (unregistered paired data).
                # Directly manipulate the tensor: center-crop oversized dims,
                # symmetric-pad undersized dims, then assign the reference
                # affine. This avoids tio.CropOrPad's spatial consistency check.
                other_shape = other_image.spatial_shape
                logger.info(
                    f"[RESAMPLE] '{name}' {other_shape} → "
                    f"'{ref_name}' {ref_shape} — LARGE AFFINE MISMATCH detected, "
                    f"forcing affine alignment + center-crop/pad"
                )

                tensor = other_image.data  # (C, W, H, D)

                # Process each spatial dim: crop or pad to match ref
                for dim_idx in range(3):
                    current = tensor.shape[dim_idx + 1]  # +1 for channel dim
                    target = ref_shape[dim_idx]

                    if current > target:
                        # Center-crop
                        excess = current - target
                        start = excess // 2
                        slices = [slice(None)] * 4  # C, W, H, D
                        slices[dim_idx + 1] = slice(start, start + target)
                        tensor = tensor[tuple(slices)]
                    elif current < target:
                        # Symmetric zero-pad
                        deficit = target - current
                        pad_lo = deficit // 2
                        pad_hi = deficit - pad_lo
                        # F.pad expects (d_lo, d_hi, h_lo, h_hi, w_lo, w_hi) in reverse
                        pad_spec = [0] * 6
                        # dim_idx=0 → W (last pair), dim_idx=1 → H, dim_idx=2 → D (first pair)
                        rev_idx = 2 - dim_idx
                        pad_spec[rev_idx * 2] = pad_lo
                        pad_spec[rev_idx * 2 + 1] = pad_hi
                        tensor = torch.nn.functional.pad(tensor, pad_spec, mode="constant", value=0)

                # Use remove_image + add_image to properly invalidate
                # TorchIO Subject's attribute cache. Direct dict assignment
                # (subject[name] = img) does NOT update cached attributes,
                # causing the sampler to see stale spatial shapes.
                new_image = tio.ScalarImage(
                    tensor=tensor,
                    affine=np.array(ref_affine, dtype=np.float64),
                )
                subject.remove_image(name)
                subject.add_image(new_image, name)

        return subject


class _EnsureMinimumSpatialSize(tio.Transform):
    """Pad images so every spatial dimension meets a minimum size.

    Required when 3D NIfTI volumes (e.g., ULF brain ~179×218×196) have
    spatial dimensions smaller than the requested ``patch_size``
    (e.g. 256×256×1).  TorchIO's sampler raises an error if any
    spatial dim of the image is smaller than the corresponding patch dim.

    Strategy:
    1. For each spatial axis, compute how much padding is needed so that
       ``image_dim >= target_dim``.
    2. If ``target_dim`` for an axis is 1 (2D slice extraction), skip that
       axis — the sampler will slice along it, so no padding is needed.
    3. Pad symmetrically with zeros (constant padding).
    4. No-op when all dimensions are already large enough.

    This must execute **after** ``_ResampleToReferenceTransform`` (shape
    harmonization) and **before** any TorchIO sampler or augmentation that
    checks spatial dimensions.
    """

    def __init__(self, min_spatial_size: tuple[int, ...]):
        super().__init__()
        # Normalise to 3D: (W, H, D)
        if len(min_spatial_size) == 2:
            self.min_size = (*min_spatial_size, 1)
        else:
            self.min_size = tuple(min_spatial_size[:3])

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Pad subject images to ensure minimum spatial size.

        Uses manual ``F.pad`` + ``remove_image/add_image`` instead of
        ``tio.Pad``.  ``tio.Pad`` records transform history on the Subject;
        when the Queue's ``UniformSampler`` later applies ``tio.Crop`` to
        extract a patch, TorchIO's Crop interacts with that history and
        *silently undoes* the padding, producing patches at the raw volume
        dimensions.  Manual padding avoids this by creating fresh
        ``ScalarImage`` objects with no recorded history.
        """
        import torch.nn.functional as F

        image_names = subject.get_images_names()
        if not image_names:
            return subject

        # Use first image to determine current spatial shape
        ref_image = subject[image_names[0]]
        current_shape = ref_image.spatial_shape  # (W, H, D)

        # Compute per-dim padding amounts
        pad_amounts = []  # [(lo, hi), (lo, hi), (lo, hi)] for W, H, D
        needs_pad = False
        for i in range(3):
            target = self.min_size[i]
            current = current_shape[i]
            # Skip depth axis when patch_size D=1 (2D slice extraction)
            if target <= 1:
                pad_amounts.append((0, 0))
                continue
            if current < target:
                diff = target - current
                lo = diff // 2
                hi = diff - lo
                pad_amounts.append((lo, hi))
                needs_pad = True
            else:
                pad_amounts.append((0, 0))

        if not needs_pad:
            return subject

        logger.debug(
            f"[SPATIAL] Padding from {current_shape} with {pad_amounts} "
            f"to ensure minimum spatial size {self.min_size}"
        )

        # F.pad expects padding in reverse order: (D_lo, D_hi, H_lo, H_hi, W_lo, W_hi)
        fpad = (
            pad_amounts[2][0],
            pad_amounts[2][1],  # D
            pad_amounts[1][0],
            pad_amounts[1][1],  # H
            pad_amounts[0][0],
            pad_amounts[0][1],  # W
        )

        for name in image_names:
            image = subject[name]
            padded_tensor = F.pad(image.data, fpad, mode="constant", value=0)
            new_image = tio.ScalarImage(
                tensor=padded_tensor,
                affine=image.affine,
            )
            subject.remove_image(name)
            subject.add_image(new_image, name)

        return subject


class _KSpaceToInputTransform(tio.Transform):
    """Transform that renames 'kspace' to 'input' for canonical naming.

    Used when acceleration=1 (no undersampling) but we still need to produce
    the canonical 'input' and 'target' keys expected by BatchAdapter.
    """

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Rename kspace to input and set target from kspace_raw or kspace."""
        # Find the k-space source key
        key = "kspace_raw" if "kspace_raw" in subject else "kspace"
        if key not in subject:
            logger.debug(f"No k-space found ({key} not in subject keys: {subject.keys()})")
            return subject

        # Add 'input' as alias for kspace (for diffusion, input=fully sampled)
        subject.add_image(
            tio.ScalarImage(tensor=subject[key].data, affine=subject[key].affine),
            "input",
        )

        # Add 'target' from kspace (same as input when acceleration=1)
        if "target" not in subject:
            subject.add_image(
                tio.ScalarImage(tensor=subject[key].data, affine=subject[key].affine),
                "target",
            )

        return subject


@dataclass
class TorchIOTransformConfig:
    """Configuration for TorchIO transform pipeline.

    Attributes:
        patch_size: Target patch size (H, W) or (H, W, D) for 3D
        standardization_mode: "smart" (crop vs resize), "strict" (always resize), "none"

        augmentation_config: Augmentation configuration (AugmentationConfigSchema)

        trajectory_type: Physics trajectory ("cartesian", "radial", "spiral", "golden_angle", "epi")
        acceleration: k-space undersampling factor
        center_fraction: Center k-space fraction to preserve

        normalize_kspace: Enable k-space normalization
        kspace_percentile: Percentile for k-space normalization
        normalize_images: Enable image normalization (ZNormalization)

        enable_graph_encoding: Enable graph encoding transform
        graph_config: Graph configuration dict (k_neighbors, max_nodes, etc.)
    """

    # Geometry
    patch_size: tuple = (320, 320)
    standardization_mode: str = "smart"  # "smart", "strict", "none"

    # Augmentation
    augmentation_config: AugmentationConfigSchema | None = None

    # Physics
    trajectory_type: str | None = None
    # [SSOT] acceleration/center_fraction default to 1 (no masking at data layer).
    # Actual dynamic masking is handled by training strategies (e.g., DiffusionTrainingStrategy).
    acceleration: int = 1
    center_fraction: float = 1.0

    # Image→k-space bridge (opt-in via data.image_undersampling). When True, a
    # RetrospectiveImageUndersampling transform synthesises an aliased ``input``
    # from a fully-sampled magnitude ``target`` (fft2c → Cartesian mask → ifft2c)
    # so an image-domain corpus (MRIxFields NIfTI) can drive a k-space-
    # acceleration recon arm (exp_c1). Off by default; kspace arms are untouched.
    image_undersampling: bool = False

    # Transversal Digital-Twin degradation (opt-in via physics.digital_twin).
    # When apply is True, a DigitalTwinDegradation transform corrupts the
    # acquisition in the data pipeline (input=corrupted, target=clean), usable
    # by ANY experiment. Off by default; VF strategies corrupt internally.
    digital_twin_config: object | None = None
    digital_twin_apply: bool = False
    digital_twin_degradation_only: bool = True

    # Normalization
    # Normalization (Refactored Phase 4)
    # Normalization (Refactored Phase 4)
    normalize_kspace: bool = False
    kspace_percentile: float = 0.99
    log_scaling: bool = False
    log_scaling_center_fraction: float = 0.25
    # "kspace" (percentile of |k|) or "image" (Parseval: percentile of the
    # coil-RSS magnitude after ifft2c). See DataConfigSchema.kspace_scale_domain.
    kspace_scale_domain: str = "kspace"

    # Image Normalization Strategy
    # Options: "none", "standard" (Z-score), "minmax" (Rescale), "percentile" (Robust Scale)
    normalization_type: str = "none"
    normalization_kwargs: dict = field(default_factory=dict)

    # Legacy flags (kept for backward compatibility, mapped to normalization_type in post_init)
    normalize_images: bool = False  # Maps to "standard"
    rescale_images: bool = False  # Maps to "minmax"
    rescale_range: tuple = (-1.0, 1.0)
    rescale_percentiles: tuple = (0.0, 100.0)

    # Advanced
    enable_graph_encoding: bool = False
    # Registry-resolved config-driven transforms as (name, kwargs)
    # pairs, appended to BOTH the train and val chains.
    extra_transforms: list = field(default_factory=list)
    graph_config: dict = field(default_factory=dict)

    # Dataset type (used to guard transforms against complex data)
    dataset_type: str = "image"

    # Coil Processing (for multi-coil k-space)
    # Options: "none", "flatten" (Option B), "rss" (pre-combined, no-op), "svd" (Option A), "sense" (Option A)
    coil_processing_mode: str = "none"
    preserve_complex: bool = False
    num_virtual_coils: int = 4
    # `None` is a MEANING here, not "unset": it selects the full FoV for SVD
    # calibration. The field was absent from this dataclass, so the two reads
    # below -- written defensively as `getattr(..., None)` -- returned None
    # unconditionally and a declared calibration count never reached the
    # transform, while its two siblings above were forwarded correctly.
    svd_calibration_lines: int | None = None

    # ── Phase 4 (resampling configurability) ──────────────────────────────
    # Carrying the schema as raw dicts (not the Pydantic instances) so the
    # TorchIOTransformConfig stays decoupled from the schema module — the
    # builders consume primitives, not Pydantic models.
    resample_enabled: bool = False
    resample_strategy: str = "reference"  # "reference" | "isotropic"
    resample_target_spacing: tuple[float, float, float] | None = None
    resample_interpolation: str = "bspline"
    resample_anti_aliasing: bool = True

    crop_or_pad_enabled: bool = False
    crop_or_pad_target_shape: tuple[int, int, int] | None = None
    crop_or_pad_padding_mode: str = "reflect"
    crop_or_pad_crop_strategy: str = "center"

    # ── Phase 4b: acquisition metadata ────────────────────────────────────────
    # Reading a Pydantic schema instance (not the underlying dict) here so
    # the transform builder forwards the full config to LoadAcquisitionMetadata.
    acquisition_metadata_config: Any | None = None

    # ── Phase 4f Amendment I: coordinate grid emission ────────────────────────
    coordinate_emission_config: Any | None = None

    # ── Phase 4b-extra: multi-echo bundling + DWI metadata ────────────────────
    multi_echo_enabled: bool = False
    dwi_metadata_enabled: bool = False
    dwi_metadata_strict: bool = False

    def __post_init__(self):
        """Validate configuration."""
        if not self.patch_size or len(self.patch_size) < 2:
            raise ValueError(f"patch_size must be at least (H, W), got {self.patch_size}")

        if self.standardization_mode not in ("smart", "strict", "none"):
            raise ValueError(
                f"standardization_mode must be 'smart', 'strict', or 'none', "
                f"got {self.standardization_mode}"
            )

        if self.trajectory_type and self.trajectory_type not in TRAJECTORY_TYPES:
            # Raises rather than warns (#1097, CLAUDE.md pitfall #9). `data.trajectory`
            # is typed `str | None` rather than a closed enum, so schema validation
            # cannot catch a typo -- this is the only place it can be caught, and the
            # old `logger.warning` meant `trajectory: spiralll` trained Cartesian while
            # the YAML claimed otherwise.
            raise ValueError(
                f"Unknown trajectory_type: {self.trajectory_type!r}. "
                f"Expected one of {TRAJECTORY_TYPES}."
            )

        if self.kspace_percentile <= 0 or self.kspace_percentile > 1:
            raise ValueError(f"kspace_percentile must be in (0, 1], got {self.kspace_percentile}")

        # =====================================================================
        # CRITICAL FIX: Normalization Mutual-Exclusion Validation
        # =====================================================================
        # Ensure only ONE image normalization strategy is applied.
        # Multiple strategies applied sequentially cause data corruption.
        # (E.g., z-score then min-max → double normalization → corrupted data)

        # Count how many normalization strategies are enabled
        normalization_strategies_enabled = []

        # Check v5.0 schema (new way)
        if self.normalization_type == "standard":
            normalization_strategies_enabled.append("standard (z-score)")
        elif self.normalization_type == "minmax":
            normalization_strategies_enabled.append("minmax (RescaleIntensity)")
        elif self.normalization_type == "percentile":
            normalization_strategies_enabled.append("percentile (robust scale)")

        # Check legacy flags (old way, kept for backward compatibility)
        if self.normalize_images and self.normalization_type == "none":
            normalization_strategies_enabled.append("legacy:normalize_images")
        if self.rescale_images and self.normalization_type == "none":
            normalization_strategies_enabled.append("legacy:rescale_images")

        # FAIL-FAST: If multiple strategies detected, raise immediately
        if len(normalization_strategies_enabled) > 1:
            raise ValueError(
                f"[CRITICAL] Multiple image normalization strategies detected. "
                f"Only ONE normalization strategy allowed per pipeline. "
                f"Found: {', '.join(normalization_strategies_enabled)}. "
                f"Config: normalization_type={self.normalization_type}, "
                f"normalize_images={self.normalize_images}, "
                f"rescale_images={self.rescale_images}. "
                f"To fix: Set all normalization flags to False or disable all but one."
            )

        # Warn if mixing old and new config (even if only one is enabled)
        legacy_flags_set = self.normalize_images or self.rescale_images
        new_config_set = self.normalization_type != "none"

        if legacy_flags_set and new_config_set:
            logger.warning(
                f"[CONFIG] Mixing legacy normalization flags with new schema. "
                f"Legacy: normalize_images={self.normalize_images}, rescale_images={self.rescale_images}. "
                f"New: normalization_type={self.normalization_type}. "
                f"Please migrate to v5.0 schema (use normalization_type only)."
            )

        # Map legacy boolean flags to normalization_type if type is "none"
        if self.normalization_type == "none":
            if self.normalize_images:
                self.normalization_type = "standard"
                logger.info(
                    "[COMPAT] Mapped legacy normalize_images=True to normalization_type='standard'"
                )
            elif self.rescale_images:
                self.normalization_type = "minmax"
                logger.info(
                    "[COMPAT] Mapped legacy rescale_images=True to normalization_type='minmax'"
                )

    @staticmethod
    def from_training_config(config) -> "TorchIOTransformConfig":
        """Create from TrainingSettings config object.

        Extracts relevant fields from config and maps to TorchIOTransformConfig.

        Args:
            config: TrainingSettings (or dict-like with getattr support)

        Returns:
            TorchIOTransformConfig instance
        """
        # Extract patch size (safely with getattr to avoid AttributeError)
        patch_size = config.sampling.patch_size

        # Extract augmentation config (safely)
        augmentation_config = getattr(config, "augmentation", None)

        # Extract trajectory (physics mode, safely)
        trajectory = getattr(config, "trajectory", None)

        # Extract normalization settings
        # Extract normalization settings
        normalize_kspace = config.processing.enable_kspace_normalization
        kspace_percentile = config.processing.kspace_percentile
        log_scaling = config.processing.enable_log_scaling
        log_scaling_center_fraction = config.processing.log_scaling_center_fraction
        kspace_scale_domain = config.processing.kspace_scale_domain

        # Image normalization, declared spelling kept verbatim. The
        # ``robust_percentile`` -> ``percentile`` fold that used to sit here
        # was a copy: ``NormalizationStrategy.from_string`` owns the alias
        # table, and both chains reach it through
        # ``resolve_image_normalization`` when they are built.
        normalization_type = config.processing.normalization_type
        normalization_kwargs = config.processing.normalization_kwargs

        # Legacy support
        normalize_images = config.processing.enable_image_normalization
        rescale_images = config.processing.enable_image_rescale

        # Extract graph encoding
        # Config-driven transforms, resolved through the registry.
        #
        # This loop used to scan for the single literal name "graph_encoding"
        # and ``break``, so every other declared entry validated and was then
        # silently discarded -- committed arms named for slice_profile,
        # synthetic_lesion, scout_acquisition and the geomamba_ulf synthetic
        # simulator all trained without the mechanism they are named for
        # (pitfall #16 behind pitfall #15). An unregistered name now RAISES.
        #
        # ``graph_encoding`` keeps its dedicated fields because the graph
        # transform is appended at a specific point in both chains; everything
        # else lands in ``extra_transforms``.
        enable_graph_encoding = False
        graph_config = {}
        extra_transforms: list[tuple[str, dict]] = []
        transforms_config = config.processing.transforms
        if transforms_config:
            from spectramr.data.transforms.registry import get_transform

            for t_config in transforms_config:
                if isinstance(t_config, dict):
                    t_name = t_config.get("name")
                    t_kwargs = dict(t_config.get("kwargs") or {})
                    flat = {k: v for k, v in t_config.items() if k not in ("name", "kwargs")}
                    flat.update(t_kwargs)
                    t_kwargs = flat
                elif hasattr(t_config, "resolved_kwargs"):
                    t_name = t_config.name
                    t_kwargs = t_config.resolved_kwargs()
                else:  # pragma: no cover - legacy stand-ins
                    t_name = getattr(t_config, "name", None)
                    t_kwargs = dict(getattr(t_config, "kwargs", {}) or {})

                if not t_name:
                    raise ValueError(
                        "data.processing.transforms entry has no 'name'. Each "
                        f"entry must be {{name, kwargs}}; got {t_config!r}. "
                        "(A committed arm spells the key 'type:' -- that never "
                        "resolved and the entry was silently dropped.)"
                    )
                # Membership is the validator: raises KeyError listing the
                # registered names, mirroring metrics.compute.
                get_transform(t_name)

                if t_name == "graph_encoding":
                    enable_graph_encoding = True
                    graph_config = {
                        "k_neighbors": t_kwargs.get("k_neighbors"),
                        "max_nodes": t_kwargs.get("max_nodes"),
                    }
                    continue
                extra_transforms.append((t_name, t_kwargs))

        # Extract physics parameters from top-level SSOT acceleration config
        # [FIX] Read from config.acceleration (SSOT) instead of hardcoding to 1
        acceleration = 1
        center_fraction = 1.0

        # [DEBUG] Verify the undersampling block exists.
        # NOTE the two different `config`s in this module. Here `config` is the
        # settings object (or its ConfigProxy), where phase 11 renamed the
        # top-level BLOCK `acceleration:` -> `undersampling:`. Everywhere below,
        # `config` is a TorchIOTransformConfig whose `acceleration` field is a
        # FLOAT factor phase 11 did NOT rename -- the block rename swept up seven
        # of those reads and turned the retrospective k-space bridge into a hard
        # AttributeError. A rename applies to a receiver, not to a name.
        accel_config = config.undersampling

        if accel_config:
            # The loader applies exactly the declared base factor. It used to
            # substitute ``max_acceleration`` (schema default 8.0) whenever
            # ``base_acceleration <= 1.0``, so "1.0 = fully sampled" to the
            # author became "8x" to the loader (cohort review 2026-09-02, T0.3;
            # latent then -- no arm routed through it -- and a trap for the next
            # one). ``1.0`` selects the identity ``_KSpaceToInputTransform``
            # below; the schema refuses anything below 1.0.
            acceleration = accel_config.base_acceleration
            center_fraction = accel_config.center_fraction
            logger.info(
                f"[SSOT-ACCELERATION] Dataloader will use: "
                f"acceleration={acceleration:.1f}x, center_fraction={center_fraction:.2f} "
                f"(from top-level acceleration config)"
            )
        else:
            # acceleration=1 (no undersampling) is the documented default
            # when no ``acceleration:`` section is declared. Log INFO so
            # the choice is observable, but don't WARN — this is the
            # intended path for fully-sampled / non-k-space experiments
            # (image-domain reconstruction, latent diffusion, etc.).
            logger.info(
                "[SSOT-ACCELERATION] No top-level 'acceleration' config — "
                "using acceleration=1 (no undersampling). If this is a "
                "k-space-undersampling experiment, add an 'acceleration:' "
                "section to the YAML."
            )

        # `preserve_complex` is no longer needed to be extracted from `model_kwargs` dynamically
        # as it should not be accessed from the `data` config. `DataConfigSchema` doesn't have `model`.
        # Any physics complex handling should rely on config settings, but since `model` is not in `proxy_config`,
        # `hasattr(config, "model")` would be false anyway.
        # However, for robustness if `config` happens to be `TrainingSettings` we can check it safely.
        preserve_complex = False
        if hasattr(config, "model") and config.model and config.model.model_kwargs:
            preserve_complex = config.model.model_kwargs.get(
                "use_complex_conv", False
            ) or config.model.model_kwargs.get("force_pure_kspace", False)

        # Root-cause fix for the May 2026 experiment_130_universal_multitask
        # SVD-double-application failure: ``num_virtual_coils`` is a TOP-LEVEL
        # data-config field, not a nested entry under ``normalization_kwargs``.
        # The earlier ``normalization_kwargs.get("num_virtual_coils", 4)``
        # always missed and fell back to the schema default ``4``, which
        # desynced from the subject builder's value (e.g. ``1`` in exp_130).
        # The downstream ``SVDCoilCompressionTransform`` dedup check then
        # compared ``shape[0]=2`` against ``2*4=8`` and the strict raise
        # fired even though the data was already correctly compressed.
        # Read directly from the data config; only fall back to the legacy
        # nested location for backward compatibility.
        num_virtual_coils = config.coils.num_virtual_coils
        if num_virtual_coils is None:
            num_virtual_coils = normalization_kwargs.get("num_virtual_coils", 4)

        # Legacy ``_data`` field: absent on the current schema, so getattr falls
        # back to ``config`` itself. Resolved to a local first so this is a single
        # getattr on a resolved object, not a nested config getattr chain.
        _data_cfg = getattr(config, "_data", config)

        # Transversal Digital-Twin degradation: opt-in via
        # physics.digital_twin.apply_as_transform. Read defensively because
        # `config` may be the data sub-config (no `physics`) in some paths.
        dt_cfg = None
        dt_apply = False
        dt_degradation_only = True
        _physics = getattr(config, "physics", None)
        _dt = getattr(_physics, "digital_twin", None) if _physics is not None else None
        if _dt is not None and getattr(_dt, "apply_as_transform", False):
            dt_cfg = _dt
            dt_apply = True
            dt_degradation_only = getattr(_dt, "transform_degradation_only", True)

        # Hoist optional sub-configs once (config may be a TrainingSettings, a
        # ConfigProxy, or a bare DataConfigSchema, so read defensively), then
        # access their fields with single (non-nested) getattr below.
        _resample = getattr(config, "resample", None)
        _crop_or_pad = getattr(config, "crop_or_pad", None)
        _acq_meta = getattr(config, "acquisition_metadata", None)
        _emit_coords = getattr(config, "emit_coordinates", None)
        _quant = getattr(config, "quantitative", None)

        return TorchIOTransformConfig(
            patch_size=patch_size,
            augmentation_config=augmentation_config,
            trajectory_type=trajectory,
            acceleration=acceleration,
            center_fraction=center_fraction,
            image_undersampling=bool(getattr(config, "image_undersampling", False)),
            digital_twin_config=dt_cfg,
            digital_twin_apply=dt_apply,
            digital_twin_degradation_only=dt_degradation_only,
            normalize_kspace=normalize_kspace,
            kspace_percentile=kspace_percentile,
            log_scaling=log_scaling,
            log_scaling_center_fraction=log_scaling_center_fraction,
            kspace_scale_domain=kspace_scale_domain,
            normalization_type=normalization_type,
            normalization_kwargs=normalization_kwargs,
            normalize_images=normalize_images,
            enable_graph_encoding=enable_graph_encoding,
            extra_transforms=extra_transforms,
            graph_config=graph_config,
            dataset_type=getattr(config, "dataset_type", "image"),
            coil_processing_mode=config.coils.processing_mode,
            rescale_images=rescale_images,
            # Declared field with a schema default -- read it directly. The old
            # `if hasattr(config, "rescale_percentiles") else (0.0, 100.0)`
            # restated a default that the schema already owns, which is how the
            # neighbouring `coil_processing_mode` fallback came to say "svd"
            # while the schema says "none".
            rescale_percentiles=config.processing.rescale_percentiles,
            preserve_complex=preserve_complex,
            num_virtual_coils=num_virtual_coils,
            svd_calibration_lines=config.coils.svd_calibration_lines,
            # ── Phase 4: resampling configurability ───────────────────────
            # Read from data.resample.* and data.crop_or_pad.*. The proxy's
            # __getattr__ forwards to data_config, so these resolve cleanly
            # whether config is a TrainingSettings, a ConfigProxy, or a
            # bare DataConfigSchema.
            resample_enabled=bool(getattr(_resample, "enabled", False)),
            resample_strategy=str(getattr(_resample, "strategy", "reference")),
            resample_target_spacing=getattr(_resample, "target_spacing", None),
            resample_interpolation=str(getattr(_resample, "interpolation", "bspline")),
            resample_anti_aliasing=bool(getattr(_resample, "anti_aliasing", True)),
            crop_or_pad_enabled=bool(getattr(_crop_or_pad, "enabled", False)),
            crop_or_pad_target_shape=getattr(_crop_or_pad, "target_shape", None),
            crop_or_pad_padding_mode=str(getattr(_crop_or_pad, "padding_mode", "reflect")),
            crop_or_pad_crop_strategy=str(getattr(_crop_or_pad, "crop_strategy", "center")),
            # ── Phase 4b: acquisition metadata ────────────────────────────
            acquisition_metadata_config=(
                _acq_meta if getattr(_acq_meta, "enabled", False) else None
            ),
            # ── Phase 4f Amendment I: coordinate grid emission ────────────
            coordinate_emission_config=(
                _emit_coords if getattr(_emit_coords, "enabled", False) else None
            ),
            # ── Phase 4b-extra: multi-echo bundling ───────────────────────
            multi_echo_enabled=bool(getattr(_quant, "enabled", False)),
            # ── Phase 4b-extra: DWI metadata loader ───────────────────────
            # Activate when acquisition_metadata is enabled AND the user
            # declares one of the DWI fields (b_value / bvec). Keeps the
            # transform off for non-DWI scans automatically.
            dwi_metadata_enabled=bool(
                getattr(_acq_meta, "enabled", False)
                and any(f in (getattr(_acq_meta, "fields", []) or []) for f in ("b_value", "bvec"))
            ),
            dwi_metadata_strict=bool(getattr(_acq_meta, "strict", False)),
        )


class TorchIOTransformBuilder:
    """Builder for composing TorchIO transform pipelines.

    Handles:
    - Spatial consistency enforcement (before & after)
    - Smart geometric standardization
    - Augmentations
    - Physics dispatch (Cartesian vs Non-Cartesian)
    - K-space and image normalization
    - Graph encoding (optional)

    Key principle: Physics synchronization must happen immediately after
    any geometric transform to regenerate k-space from the new geometry.
    """

    @staticmethod
    def build_train_transforms(config: TorchIOTransformConfig) -> tio.Compose:
        """Build training transform pipeline.

        Includes all augmentations, physics sync, and normalization.

        Args:
            config: TorchIOTransformConfig

        Returns:
            tio.Compose pipeline ready for training
        """
        transforms = []

        # =====================================================================
        # -1. PRE-SPATIAL metadata transforms (Phase 4b / 4b-extra)
        # =====================================================================
        # These read on-disk sidecars (BIDS-JSON, .bval/.bvec) and attach
        # per-sample metadata dicts to the Subject. They MUST run before
        # any spatial transform because they may read the image's source
        # file path (which TorchIO discards after the first geometric op).
        TorchIOTransformBuilder._append_metadata_transforms(transforms, config)

        # =====================================================================
        # 0. CRITICAL: Spatial consistency FIRST
        # =====================================================================
        # This fixes TorchIO "origin mismatch" errors when kspace/target
        # have different affines (e.g., from H5 files with varying metadata)
        transforms.append(TorchIOTransformBuilder._build_spatial_consistency_transform())
        logger.debug("[SPATIAL] EnsureSpatialConsistency applied (FIRST)")

        # =====================================================================
        # 0.05 Multi-echo bundling (Phase 4b-extra)
        # =====================================================================
        # Validates that the channel axis aligns with the per-echo TE list
        # in acquisition_metadata. Runs after spatial-consistency so the
        # channel dim is stable. No-op when multi_echo_enabled=False.
        if config.multi_echo_enabled:
            from spectramr.data.transforms.multi_echo import BundleMultiEcho

            transforms.append(BundleMultiEcho())
            logger.debug("[MULTI-ECHO] BundleMultiEcho applied (quantitative.enabled=True)")

        # =====================================================================
        # 0.1 Resample to common spatial shape (Phase 4 — configurable)
        # =====================================================================
        # When data.resample.enabled is True, dispatch to ``tio.Resample``
        # with the configured strategy (isotropic spacing, etc.). When
        # False, fall back to the legacy ``_ResampleToReferenceTransform``
        # which infers a target affine from the first subject.
        from spectramr.data.builders.resample_dispatch import (
            build_crop_or_pad_transform,
            build_resample_transform,
        )

        resample_t = build_resample_transform(config)
        if resample_t is not None:
            transforms.append(resample_t)
            logger.debug(
                f"[SPATIAL] Resample applied via data.resample.* "
                f"(strategy={config.resample_strategy})"
            )
        else:
            transforms.append(_ResampleToReferenceTransform())
            logger.debug(
                "[SPATIAL] ResampleToReference applied (legacy — data.resample.enabled is False)"
            )

        # =====================================================================
        # 0.2 Ensure minimum / canonical spatial size (Phase 4 — configurable)
        # =====================================================================
        # When data.crop_or_pad.enabled is True, dispatch to
        # ``tio.CropOrPad(target_shape, padding_mode=reflect|...)``. When
        # False, fall back to legacy ``_EnsureMinimumSpatialSize``.
        crop_t = build_crop_or_pad_transform(config)
        if crop_t is not None:
            transforms.append(crop_t)
            logger.debug(
                f"[SPATIAL] CropOrPad applied via data.crop_or_pad.* "
                f"(target_shape={config.crop_or_pad_target_shape}, "
                f"pad={config.crop_or_pad_padding_mode})"
            )
        else:
            transforms.append(_EnsureMinimumSpatialSize(config.patch_size))
            logger.debug(
                f"[SPATIAL] EnsureMinimumSpatialSize({config.patch_size}) applied "
                "(legacy — data.crop_or_pad.enabled is False)"
            )

        # =====================================================================
        # 0.5. Coil Processing (complex->real conversion)
        # =====================================================================
        # This must happen early to ensure channel count is correct for model
        from spectramr.data.transforms.coil_compression import SVDCoilCompressionTransform
        from spectramr.data.transforms.kspace_coil_transforms import (
            CoilCombineTransform,
            ComplexToRealTransform,
        )

        if config.coil_processing_mode == "flatten":
            # Option B: Flatten coils to channels (2*Coils channels)
            transforms.append(ComplexToRealTransform())
            logger.debug("[COIL] ComplexToRealTransform: (C,H,W,D) complex -> (2C,H,W,D) real")
        elif config.coil_processing_mode == "svd":
            # Option A: SVD Coil Compression (Linear, preserves phase)
            transforms.append(
                SVDCoilCompressionTransform(
                    num_virtual_coils=config.num_virtual_coils,
                    calibration_lines=config.svd_calibration_lines,
                )
            )
            if not config.preserve_complex:
                transforms.append(ComplexToRealTransform())
                logger.debug(
                    f"[COIL] SVDCoilCompressionTransform: (C,H,W,D) complex -> ({config.num_virtual_coils},H,W,D) complex -> ({config.num_virtual_coils * 2},H,W,D) real"
                )
            else:
                logger.debug(
                    f"[COIL] SVDCoilCompressionTransform: (C,H,W,D) complex -> ({config.num_virtual_coils},H,W,D) complex (preserve_complex=True)"
                )
        elif config.coil_processing_mode == "sense":
            # Option A: SENSE coil combination using sensitivity maps
            transforms.append(CoilCombineTransform(method="sense"))
            logger.debug("[COIL] CoilCombineTransform (SENSE): (C,H,W,D) complex -> (2,H,W,D) real")
        elif config.coil_processing_mode == "rss_per_channel":
            # Shen 2024's own combination: RSS reduced once per real/imaginary
            # channel, so the result keeps two (non-negative) channels instead of
            # the single magnitude ``rss_image`` returns. The three prior-method
            # arms need 2 channels and none of the three papers takes multi-coil
            # input, so this is the combination their pipelines assume happened
            # upstream.
            transforms.append(CoilCombineTransform(method="rss_per_channel"))
            logger.debug(
                "[COIL] CoilCombineTransform (rss_per_channel): (C,H,W,D) complex -> "
                "(2,H,W,D) real k-space"
            )
        elif config.coil_processing_mode == "rss_image":
            # IFFT → RSS → image-domain magnitude (1 channel, real).
            # Distinct from ``"rss"`` (which round-trips through FFT and
            # strips phase — see audit-2026-05-14 F5). Use this mode for
            # image-domain reconstruction networks that want
            # pre-combined input without the centro-symmetric
            # phase-strip artefact.
            transforms.append(CoilCombineTransform(method="rss_image"))
            logger.debug(
                "[COIL] CoilCombineTransform (rss_image): (C,H,W,D) complex -> (1,H,W,D) real image"
            )
        elif config.coil_processing_mode in ("rss", "magnitude"):
            # ``"rss"`` and ``"magnitude"`` historically meant
            # "data is already pre-combined upstream (e.g. M4Raw
            # single-coil or subject builder processed)". No transform
            # needed — pass through.
            logger.debug(
                f"[COIL] {config.coil_processing_mode} mode: data assumed pre-combined, skipping coil transform"
            )
        elif config.coil_processing_mode != "none":
            # CLAUDE.md pitfall #9 — no silent fallback. An unrecognised
            # ``coil_processing_mode`` is a config error; raise so the
            # smoke wrapper / audit catches it instead of training on
            # the wrong coil layout.
            raise ValueError(
                f"[COIL] Unknown coil_processing_mode: {config.coil_processing_mode!r}. "
                "Valid values: 'sense', 'rss', 'magnitude', 'rss_per_channel', 'rss_image', "
                "'compressed_sensing' (with num_virtual_coils), or 'none'. "
                "Add the new mode to the dispatch in "
                "src/data/builders/torchio_transform_builder.py if "
                "introducing a new combination strategy."
            )

        # =====================================================================
        # 1. Geometric Standardization — DELETED (audit M1)
        # =====================================================================
        # `SmartGeometricStandardization` / `tio.Resize` were gated on
        # `config.enable_geometric_standardization`, which
        # `from_training_config` populates with
        # `getattr(_data_cfg, "enable_geometric_standardization", False)`.
        # That name is NOT a field on `DataConfigSchema` (nor on `.processing`
        # or `.sampling`), so the getattr returned the default on every arm and
        # the branch was constant-False. 11 arms declare
        # `enable_geometric_standardization: true` in YAML; the `data:` block is
        # `extra="ignore"`, so the key was dropped before anything read it.
        #
        # DELETED rather than wired, and the distinction is the whole point:
        # this branch also appended `PhysicsSynchronization()`, which resolves
        # its source key as "input" FIRST — and on a k-space arm `input` IS
        # k-space, so it would apply a SECOND forward FFT (audit A4, filed
        # separately with its arm count). Turning the flag on would have created
        # that bug on exactly the 11 arms asking for it.
        #
        # The CI check `scripts/ci/check_getattr_names_a_real_field.py` now
        # rejects a `getattr(cfg, "<literal>", default)` under data/builders/
        # that names no schema field, so the next one cannot go quiet.

        # =====================================================================
        # 2. Augmentations
        # =====================================================================
        if config.augmentation_config is not None:
            augmentations = TorchIOAugmentationFactory.build(
                config.augmentation_config,
                dataset_type=config.dataset_type,
            )
            if augmentations:
                transforms.append(augmentations)
                logger.info(f"[AUGMENTATION] {len(augmentations.transforms)} augmentations added")

                # Re-sync k-space after geometric augmentation — but ONLY on an
                # image-primary arm (A4).
                #
                # The transform re-derives k-space from the IMAGE. On a k-space
                # arm there is no image to derive from: `subject["input"]` IS
                # the measured k-space, and the transform's key search tries
                # "input" first, so it fed k-space to `fft2c` and overwrote
                # `subject["kspace"]` with a SECOND forward transform. That is
                # not spare compute — `strategies/mixins/kspace.py` returns
                # `data["kspace"]` as the canonical accessor, and
                # `graph_transform` reads `subject["kspace"]` directly, so the
                # doubly-transformed tensor is what recon strategies consume.
                # 174 arms enable augmentation; 49 of them are k-space.
                if is_kspace_dataset_type(config.dataset_type):
                    logger.debug(
                        "[PHYSICS] PhysicsSynchronization skipped: %r serves "
                        "k-space, so there is no image to re-derive it from.",
                        config.dataset_type,
                    )
                else:
                    # `input_is_image=True` is not a new decision -- it is this
                    # `else`, stated to the transform. Reaching here PROVES the
                    # arm is image-primary, so `input` holds an image. Without
                    # saying so, the transform's own `input`/`kspace` ambiguity
                    # guard refuses and the whole pipeline dies on any dataset
                    # that derives a `kspace` key alongside `input`
                    # (`UniversalMRIDataset` does) -- 10 `10_paradigms` arms in
                    # cluster job 8012333. The caller established the fact,
                    # discarded it, and the callee then refused for want of
                    # exactly that fact.
                    transforms.append(PhysicsSynchronization(input_is_image=True))
                    logger.debug("[PHYSICS] PhysicsSynchronization after augmentation")

        # =====================================================================
        # 3. K-Space Normalization (BEFORE Physics Dispatch)
        # =====================================================================
        # CRITICAL FIX: K-space normalization MUST occur BEFORE physics dispatch (masking)
        # to ensure scale factor is computed on FULL k-space, not undersampled version.
        #
        # Otherwise:
        #   - Non-Cartesian: scale computed on radial/spiral samples (sparse)
        #   - Cartesian: scale computed after masking (fewer samples)
        # Result: Scale mismatch between training and inference
        #
        if config.normalize_kspace:
            from spectramr.data.transforms.normalization import (
                KSpaceNormalizationTransform,
            )

            kspace_norm = KSpaceNormalizationTransform(
                percentile=config.kspace_percentile,
                log_scaling=config.log_scaling,
                center_fraction=config.log_scaling_center_fraction,
                scale_domain=config.kspace_scale_domain,
            )
            transforms.append(kspace_norm)
            logger.info(
                f"[NORMALIZATION] K-space normalization enabled BEFORE masking "
                f"({config.kspace_percentile:.2%} percentile on FULL k-space)"
            )

        # =====================================================================
        # 4. Physics Dispatch (The Circuit Switch)
        # =====================================================================
        # Detect if we need Non-Cartesian (Exp 42) or Cartesian (Exp 11/54)
        # This now applies to NORMALIZED k-space (not raw)
        physics_transform = TorchIOTransformBuilder._build_physics_transform(config)
        transforms.append(physics_transform)
        # Log the injection
        logger.info(
            f"[PHYSICS DISPATCH] {config.trajectory_type or 'Cartesian'} applied "
            f"(to normalized k-space)"
        )

        # =====================================================================
        # 5. Image Normalization (Strategy-Based - AFTER k-space transforms)
        # =====================================================================
        # Image normalization is applied AFTER all k-space operations
        # (physics dispatch, k-space normalization)

        # ONE resolver decides the whole step -- the k-space mutual exclusion,
        # the vocabulary fold and the spec -- for this chain, the val chain and
        # ``pipelines/infer.py``. Each used to carry its own copy of the
        # decision; a copy that drifts is not an error but a differently
        # scaled tensor on one side of the train/predict pair (non-negotiable
        # 17). What the resolver builds replaced a three-branch
        # tio.ZNormalization/RescaleIntensity dispatch that ran SECOND on
        # tensors ``ContrastAwarePairedDataset`` had already normalized (#760,
        # the image-domain twin of #571), and the percentile branch stopped
        # clipping at the percentile before rescaling (plan item B5).
        from spectramr.data.transforms.normalization import (
            ImageNormalizationTransform,
            resolve_image_normalization,
        )

        spec = resolve_image_normalization(
            normalization_type=config.normalization_type,
            dataset_type=getattr(config, "dataset_type", None),
            normalization_kwargs=config.normalization_kwargs,
            kspace_normalization_enabled=config.normalize_kspace,
        )
        if spec is not None and spec.enabled:
            transforms.append(ImageNormalizationTransform(spec))

        # =====================================================================
        # 6. Graph Encoding (Optional Advanced)
        # =====================================================================
        if config.enable_graph_encoding:
            from spectramr.data.transforms.graph_transform import GraphEncodingTransform

            graph_transform = GraphEncodingTransform(**config.graph_config)
            transforms.append(graph_transform)
            logger.debug(
                f"[GRAPH] GraphEncodingTransform "
                f"(k={config.graph_config.get('k_neighbors', 8)}, "
                f"max_nodes={config.graph_config.get('max_nodes', 4096)})"
            )

        # =====================================================================
        # 7. CRITICAL: Final spatial consistency enforcement before Queue
        # =====================================================================
        # After all transforms, force all images to have same affine/spacing
        transforms.append(TorchIOTransformBuilder._build_spatial_consistency_transform())
        logger.debug("[SPATIAL] EnsureSpatialConsistency applied (LAST)")

        # =====================================================================
        # 8. Coordinate grid emission (Phase 4f Amendment I)
        # =====================================================================
        # AFTER all spatial ops so coord_resolution matches the final shape.
        TorchIOTransformBuilder._append_coordinate_grid_transform(transforms, config)

        # Config-driven transforms from the registry (both chains).
        TorchIOTransformBuilder._append_registry_transforms(transforms, config, "TRAIN")

        # Image→k-space bridge (LAST, on the final magnitude image): aliases
        # ``input`` for image-domain corpora driving a k-space recon arm (exp_c1).
        TorchIOTransformBuilder._append_image_undersampling(transforms, config)

        # Truly LAST: re-bind Subject.__dict__ so the patch sampler crops what this
        # chain produced rather than what it received (#1213).
        transforms.append(_SyncSubjectAttributes())

        return tio.Compose(transforms)

    @staticmethod
    def _append_image_undersampling(transforms: list, config: TorchIOTransformConfig) -> None:
        """Append the retrospective image→k-space undersampling transform (opt-in).

        Bridges a fully-sampled image corpus to a k-space-acceleration recon task:
        ``data.image_undersampling: true`` synthesises an aliased ``input`` from
        the full ``input`` via fft2c → Cartesian mask → ifft2c (physics SSOT).
        Off by default → kspace arms are untouched (zero blast radius).
        """
        if not getattr(config, "image_undersampling", False):
            return
        from spectramr.data.transforms.image_undersampling import (
            RetrospectiveImageUndersampling,
        )

        transforms.append(
            RetrospectiveImageUndersampling(
                acceleration=float(config.acceleration),
                center_fraction=float(config.center_fraction),
            )
        )
        logger.info(
            f"[IMAGE-UNDERSAMPLING] retrospective k-space bridge @ R="
            f"{float(config.acceleration):.1f}x, ACS={float(config.center_fraction):.2f}"
        )

    @staticmethod
    def build_val_transforms(config: TorchIOTransformConfig) -> tio.Compose:
        """Build validation transform pipeline.

        Same as training but WITHOUT augmentations (for reproducibility).

        Args:
            config: TorchIOTransformConfig

        Returns:
            tio.Compose pipeline ready for validation
        """
        transforms = []

        # -1. PRE-SPATIAL metadata transforms (Phase 4b / 4b-extra) — must run
        # before any spatial op so per-sample sidecars resolve.
        TorchIOTransformBuilder._append_metadata_transforms(transforms, config)

        # 0. Spatial consistency (CRITICAL first)
        transforms.append(TorchIOTransformBuilder._build_spatial_consistency_transform())

        # 0.05 Multi-echo bundling (Phase 4b-extra)
        if config.multi_echo_enabled:
            from spectramr.data.transforms.multi_echo import BundleMultiEcho

            transforms.append(BundleMultiEcho())

        # 0.1 Resample to common spatial shape (Phase 4 — configurable)
        from spectramr.data.builders.resample_dispatch import (
            build_crop_or_pad_transform,
            build_resample_transform,
        )

        resample_t = build_resample_transform(config)
        if resample_t is not None:
            transforms.append(resample_t)
        else:
            transforms.append(_ResampleToReferenceTransform())

        # 0.2 Ensure minimum / canonical spatial size (Phase 4 — configurable)
        crop_t = build_crop_or_pad_transform(config)
        if crop_t is not None:
            transforms.append(crop_t)
        else:
            transforms.append(_EnsureMinimumSpatialSize(config.patch_size))

        # =====================================================================
        # 0.5. Coil Processing (complex->real conversion)
        # =====================================================================
        # This must happen early to ensure channel count is correct for model
        from spectramr.data.transforms.coil_compression import SVDCoilCompressionTransform
        from spectramr.data.transforms.kspace_coil_transforms import (
            CoilCombineTransform,
            ComplexToRealTransform,
        )

        if config.coil_processing_mode == "flatten":
            # Option B: Flatten coils to channels (2*Coils channels)
            transforms.append(ComplexToRealTransform())
            logger.debug("[COIL] ComplexToRealTransform: (C,H,W,D) complex -> (2C,H,W,D) real")
        elif config.coil_processing_mode == "svd":
            # Option A: SVD Coil Compression (Linear, preserves phase)
            transforms.append(
                SVDCoilCompressionTransform(
                    num_virtual_coils=config.num_virtual_coils,
                    calibration_lines=config.svd_calibration_lines,
                )
            )
            if not config.preserve_complex:
                transforms.append(ComplexToRealTransform())
                logger.debug(
                    f"[COIL] SVDCoilCompressionTransform: (C,H,W,D) complex -> ({config.num_virtual_coils},H,W,D) complex -> ({config.num_virtual_coils * 2},H,W,D) real"
                )
            else:
                logger.debug(
                    f"[COIL] SVDCoilCompressionTransform: (C,H,W,D) complex -> ({config.num_virtual_coils},H,W,D) complex (preserve_complex=True)"
                )
        elif config.coil_processing_mode == "sense":
            # Option A: SENSE coil combination using sensitivity maps
            transforms.append(CoilCombineTransform(method="sense"))
            logger.debug("[COIL] CoilCombineTransform (SENSE): (C,H,W,D) complex -> (2,H,W,D) real")
        elif config.coil_processing_mode == "rss_per_channel":
            transforms.append(CoilCombineTransform(method="rss_per_channel"))
            logger.debug(
                "[COIL] CoilCombineTransform (rss_per_channel): (C,H,W,D) complex -> "
                "(2,H,W,D) real k-space"
            )
        elif config.coil_processing_mode == "rss_image":
            transforms.append(CoilCombineTransform(method="rss_image"))
            logger.debug(
                "[COIL] CoilCombineTransform (rss_image): (C,H,W,D) complex -> (1,H,W,D) real image"
            )
        elif config.coil_processing_mode in ("rss", "magnitude"):
            # ``"rss"`` / ``"magnitude"`` — data is already pre-combined
            # upstream (M4Raw single-coil or subject-builder processed).
            logger.debug(
                f"[COIL] {config.coil_processing_mode} mode: data assumed pre-combined, skipping coil transform"
            )
        elif config.coil_processing_mode != "none":
            # CLAUDE.md pitfall #9 — fail loud on unknown mode (see the
            # training-side branch above for the canonical list).
            raise ValueError(
                f"[COIL] Unknown coil_processing_mode: {config.coil_processing_mode!r}. "
                "Valid: 'sense', 'rss', 'magnitude', 'rss_per_channel', 'rss_image', "
                "'compressed_sensing', 'none'."
            )

        # 1. Geometry — deleted with the train-side twin (M1); see above.

        # 2. NO AUGMENTATIONS for validation

        # 3. Physics (same as training)
        physics_transform = TorchIOTransformBuilder._build_physics_transform(config)
        transforms.append(physics_transform)

        # 4. Normalization (consistent with training)
        if config.normalize_kspace:
            from spectramr.data.transforms.normalization import (
                KSpaceNormalizationTransform,
            )

            kspace_norm = KSpaceNormalizationTransform(
                percentile=config.kspace_percentile,
                log_scaling=config.log_scaling,
                center_fraction=config.log_scaling_center_fraction,
                scale_domain=config.kspace_scale_domain,
            )
            transforms.append(kspace_norm)

        # Image normalization -- the SAME resolver the train chain uses.
        #
        # These two chains previously carried independent copies of the
        # decision, and they had already drifted: the train copy warned on an
        # unknown `normalization_type` while this one had no unknown branch at
        # all, so a val chain silently applied nothing where the train chain
        # at least said something. One resolver, both chains and predict.
        from spectramr.data.transforms.normalization import (
            ImageNormalizationTransform,
            resolve_image_normalization,
        )

        spec = resolve_image_normalization(
            normalization_type=config.normalization_type,
            dataset_type=getattr(config, "dataset_type", None),
            normalization_kwargs=config.normalization_kwargs,
            kspace_normalization_enabled=config.normalize_kspace,
        )
        if spec is not None and spec.enabled:
            transforms.append(ImageNormalizationTransform(spec))

        # 5. Graph encoding (if enabled)
        if config.enable_graph_encoding:
            from spectramr.data.transforms.graph_transform import GraphEncodingTransform

            graph_transform = GraphEncodingTransform(**config.graph_config)
            transforms.append(graph_transform)

        # [DEBUG] Probe statistics. ``_ValDebugStats`` is defined at MODULE scope
        # (not as a local class here) so it is picklable: Python 3.14 defaults the
        # multiprocessing start method to ``forkserver`` on non-Mac POSIX, which
        # pickles the DataLoader worker's Dataset + transform Compose — a local
        # class's qualname carries ``<locals>`` and cannot be located by pickle.
        transforms.append(_ValDebugStats())

        # 6. Final spatial consistency (CRITICAL last)
        transforms.append(TorchIOTransformBuilder._build_spatial_consistency_transform())

        # 7. Coordinate grid emission (Phase 4f Amendment I) — AFTER spatial ops.
        TorchIOTransformBuilder._append_coordinate_grid_transform(transforms, config)

        # Config-driven transforms from the registry (both chains).
        TorchIOTransformBuilder._append_registry_transforms(transforms, config, "VAL")

        # Image→k-space bridge (deterministic equispaced mask → reproducible val).
        TorchIOTransformBuilder._append_image_undersampling(transforms, config)

        # Truly LAST: re-bind Subject.__dict__ so the patch sampler crops what this
        # chain produced rather than what it received (#1213). The val chain needs the
        # sync for a second reason that holds with no sampler at all: **attribute**
        # access reads ``__dict__`` too, so ``subject.input`` returns the pre-transform
        # object while ``subject['input']`` returns the right one. Val does not
        # patch-sample today (``build_val_queue`` has no production caller — its only
        # reference is a docstring example, #1210), so that is the live reason here.
        transforms.append(_SyncSubjectAttributes())

        return tio.Compose(transforms)

    @staticmethod
    def _append_registry_transforms(
        transforms: list, config: "TorchIOTransformConfig", chain: str
    ) -> None:
        """Append the registry-resolved ``data.processing.transforms`` entries.

        Appended LAST-but-one (before the image->k-space bridge) and to BOTH
        chains. Applying a declared transform to train only would reproduce the
        normalization split this audit already found: the model would be graded
        on data the transform never touched.

        The name was validated at ``from_training_config`` time; constructing
        here keeps the transform objects out of the frozen config dataclass.

        Read with direct attribute access, not ``getattr(..., None)``: the field
        is declared with ``default_factory=list`` so it is always present, and a
        defaulted read that can never fire is exactly what
        ``check_getattr_names_a_real_field`` exists to reject — it cannot tell a
        dead default from one silently swallowing a rename.
        """
        if not config.extra_transforms:
            return
        from spectramr.data.transforms.registry import build_transform

        for name, kwargs in config.extra_transforms:
            transforms.append(build_transform(name, **kwargs))
            logger.info(
                "[TRANSFORM] %s: registry transform %r appended (kwargs=%s)",
                chain,
                name,
                sorted(kwargs),
            )

    @staticmethod
    def _build_physics_transform(config: TorchIOTransformConfig) -> tio.Transform:
        """Build physics transform based on trajectory type.

        Dispatches to Non-Cartesian simulation if trajectory is spiral/radial.
        """
        # Transversal Digital-Twin degradation takes the physics slot when
        # opted in (produces input=corrupted, target=clean), independent of
        # trajectory. VF strategies leave this off (they corrupt internally).
        if getattr(config, "digital_twin_apply", False) and config.digital_twin_config is not None:
            from spectramr.data.transforms.tio_physics import DigitalTwinDegradation
            from spectramr.infrastructure.physics.digital_twin_simulator import (
                DigitalTwinSimulator,
            )

            im_size = tuple(config.patch_size[:2])
            simulator = DigitalTwinSimulator.from_config(config.digital_twin_config, im_size)
            logger.info(
                "[PHYSICS] Injecting transversal DigitalTwinDegradation "
                f"(im_size={im_size}, degradation_only="
                f"{config.digital_twin_degradation_only})"
            )
            return DigitalTwinDegradation(
                simulator,
                degradation_only=config.digital_twin_degradation_only,
            )

        # [FIX] Skip k-space generation entirely if trajectory_type is None/null
        if config.trajectory_type is None:
            logger.info(
                "[PHYSICS] trajectory_type=None → Skipping k-space generation (image-domain only)"
            )
            return _NoOpTransform()

        # [PHASE 1.3] Non-Cartesian Physics Simulation
        # Consumes the same tuple the validator above checks against, so the accepted
        # set and the ROUTED set cannot drift apart again (#1097): `golden_angle` and
        # `epi` used to pass validation here and fall through to a silent NoOp.
        if config.trajectory_type in NON_CARTESIAN_TRAJECTORIES:
            from spectramr.data.transforms.non_cartesian import (
                NonCartesianSimulationTransform,
            )

            im_size = config.patch_size[:2] if config.patch_size else None

            # Log the injection
            logger.info(
                f"[PHYSICS] Injecting Non-Cartesian Simulation: {config.trajectory_type} "
                f"(x{config.acceleration})"
            )

            return NonCartesianSimulationTransform(
                pattern=config.trajectory_type,
                im_size=im_size,
                acceleration=config.acceleration,
                # Augmentation config might have noise settings, for now default
                enable_noise=False,
            )

        # For Cartesian trajectories, return a no-op transform
        # Physics masking is handled by PhysicsSynchronization or Logic elsewhere
        if config.trajectory_type and config.trajectory_type != "cartesian":
            # Unreachable: __post_init__ rejects anything outside TRAJECTORY_TYPES, and
            # the branch above routes every non-Cartesian member. Kept as a guard so a
            # future name added to the tuple but not to the generator fails loudly here
            # instead of resurrecting the silent NoOp this replaced (#1097).
            raise ValueError(
                f"[PHYSICS] trajectory_type={config.trajectory_type!r} is accepted but "
                f"has no simulation path. Add it to NON_CARTESIAN_TRAJECTORIES and to "
                f"trajectories.get_trajectory, or remove it from TRAJECTORY_TYPES."
            )
        # [SSOT] If acceleration is 1 (default), use key rename transform (no masking).
        # The actual masking is done by training strategies (e.g., DiffusionTrainingStrategy).
        if config.acceleration == 1:
            logger.info(
                "[PHYSICS] Data-loader acceleration=1 (KSpaceToInput). Dynamic masking by strategy."
            )
            return _KSpaceToInputTransform()
        else:
            logger.info(
                f"[PHYSICS] Injecting Cartesian PhysicsInformedMasking "
                f"(acc={config.acceleration}x, cf={config.center_fraction})"
            )
            return PhysicsInformedMasking(
                acceleration=config.acceleration,
                center_fraction=config.center_fraction,
                validate_shapes=True,
            )

    @staticmethod
    def _build_spatial_consistency_transform() -> tio.Transform:
        """Build spatial consistency transform.

        Forces all images in a Subject to share the same affine matrix.
        This is critical for TorchIO Queue/Sampler which requires spatial consistency.

        Returns:
            EnsureSpatialConsistency transform
        """
        from spectramr.data.transforms.geometric import EnsureSpatialConsistency

        return EnsureSpatialConsistency()

    @staticmethod
    def _append_metadata_transforms(transforms: list, config: TorchIOTransformConfig) -> None:
        """Append Phase 4b / 4b-extra metadata-reading transforms.

        These must run BEFORE any spatial transform — they read sidecar
        files keyed off the source image path, which TorchIO discards
        once a geometric op runs.

        - ``LoadAcquisitionMetadata``: BIDS-JSON / DICOM / H5-attrs /
          yaml_inline → per-sample (TE/TR/TI/FA/B0) dict
        - ``LoadDWIMetadata``: .bval/.bvec siblings → b-value + bvec lists

        No-op when the corresponding config block is None / disabled.
        """
        if config.acquisition_metadata_config is not None:
            from spectramr.data.transforms.acquisition_metadata import (
                LoadAcquisitionMetadata,
            )

            transforms.append(LoadAcquisitionMetadata(config.acquisition_metadata_config))
            logger.debug(
                "[METADATA] LoadAcquisitionMetadata applied "
                f"(source={config.acquisition_metadata_config.source})"
            )

        if config.dwi_metadata_enabled:
            from spectramr.data.transforms.dwi_metadata import LoadDWIMetadata

            transforms.append(LoadDWIMetadata(strict=config.dwi_metadata_strict))
            logger.debug(
                f"[METADATA] LoadDWIMetadata applied (strict={config.dwi_metadata_strict})"
            )

    @staticmethod
    def _append_coordinate_grid_transform(transforms: list, config: TorchIOTransformConfig) -> None:
        """Append the Phase 4f coordinate grid transform at the tail of
        the pipeline. Runs AFTER all spatial ops so the recorded
        ``coord_resolution`` matches the final tensor shape the model sees.

        No-op when ``data.emit_coordinates.enabled`` is False.
        """
        if config.coordinate_emission_config is not None:
            from spectramr.data.transforms.coordinate_grid import CoordinateGridTransform

            transforms.append(CoordinateGridTransform(config.coordinate_emission_config))
            logger.debug("[COORDS] CoordinateGridTransform appended")


__all__ = [
    "TorchIOTransformBuilder",
    "TorchIOTransformConfig",
]
