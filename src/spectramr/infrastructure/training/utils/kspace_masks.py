"""K-Space Mask Utilities

This module contains utilities for generating and managing
k-space acceleration masks used in MRI reconstruction and
diffusion training strategies.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from spectramr.infrastructure.physics.sampling import (
    ACCELERATOR_TO_MASK_TYPE,
    ColdDiffusionAccelerator,
    MaskGenerator,
    MaskType,
    create_kspace_accelerator,
)
from spectramr.infrastructure.physics.sampling_registry import SamplingPatternRegistry
from spectramr.infrastructure.training.utils.mask_table_cache import MaskTableCache


class KSpaceMaskGenerator:
    """Utility class for generating k-space acceleration masks.

    The generator now delegates mask production to
    :class:`ColdDiffusionAccelerator` so that there is a single source of
    truth for k-space undersampling patterns across the codebase.
    """

    def __init__(
        self,
        num_timesteps: int,
        device: torch.device | None = None,
        default_pattern: str = "linear",
        accelerator_kwargs: dict[str, Any] | None = None,
        pattern_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the mask generator.

        Args:
            num_timesteps: Diffusion horizon shared with the accelerator.
            device: Target device for produced masks.
            default_pattern: Pattern used when a caller omits the pattern
                argument.
            accelerator_kwargs: Global kwargs forwarded to every accelerator
                instance.
            pattern_overrides: Optional mapping of pattern/acceleration type to
                keyword arguments that should be merged when instantiating the
                accelerator for that pattern.
        """

        self.num_timesteps = num_timesteps
        self.device = device or torch.device("cpu")
        self.default_pattern = default_pattern
        self._accelerator_kwargs = accelerator_kwargs or {}
        overrides = pattern_overrides or {}
        self._pattern_overrides = {k.lower(): v for k, v in overrides.items()}
        self._accelerators: dict[str, ColdDiffusionAccelerator] = {}
        self._warned_about_factor = False
        # Device-resident cascade cache; see ``mask_table_cache`` for why the
        # fixed-seed path is memoisable and the dynamic-mask path is not.
        self._mask_tables = MaskTableCache()

    def _resolve_acceleration_type(self, pattern: str) -> str:
        """Canonical accelerator name for a declared pattern.

        Delegates to :class:`SamplingPatternRegistry` so this module stops being a
        second place a pattern name can mean something (issue #954). An unknown
        name raises there, with the accepted set in the message.
        """
        return SamplingPatternRegistry.resolve(pattern)

    def _accelerator_kwargs_for(self, key: str) -> dict[str, Any]:
        """_accelerator_kwargs_for.

        Args:
            key (str): Description.
        Returns:
            dict[str, Any]: Description.
        """
        overrides = self._pattern_overrides.get(key.lower())
        if overrides is None:
            overrides = self._pattern_overrides.get(
                self._resolve_acceleration_type(key),
            )
        merged: dict[str, Any] = dict(self._accelerator_kwargs)
        if overrides:
            merged.update(overrides)
        return merged

    def applied_record(self, pattern: str | None = None) -> dict[str, Any]:
        """The family and parameters this generator masks with, for the snapshot provenance.

        ``pattern`` is what the caller passes on every mask call; ``None`` means
        the generator's default. Read from the kwargs it forwards, not from built
        accelerators, because the first snapshot precedes the first mask.
        """
        declared = pattern or self.default_pattern
        kwargs = self._accelerator_kwargs_for(declared)
        unset = "<accelerator default>"
        base, peak = kwargs.get("base_acceleration"), kwargs.get("max_acceleration")
        seed = kwargs.get("seed", unset)
        record = {
            "pattern": declared,
            "family": self._resolve_acceleration_type(declared),
            "acceleration": unset if base is None or peak is None else [base, peak],
            "center_fraction": kwargs.get("center_fraction", unset),
            "min_center_fraction": kwargs.get("min_center_fraction", unset),
            "acceleration_schedule": kwargs.get("acceleration_schedule", unset),
            "enforce_nested": kwargs.get("enforce_nested", unset),
            "mask_seed": "None: global RNG, redrawn per call" if seed is None else seed,
        }
        # A step schedule indexes the ladder, so [base, max] alone misstates it.
        for key in ("acceleration_range", "density_power", "schedule_power"):
            if key in kwargs:
                record[key] = kwargs[key]
        return record

    def _get_accelerator(
        self,
        pattern: str | None,
    ) -> ColdDiffusionAccelerator:
        """_get_accelerator.

        Args:
            pattern (str | None): Description.
        Returns:
            ColdDiffusionAccelerator: Description.
        """
        resolved = self._resolve_acceleration_type(
            pattern or self.default_pattern,
        )
        if resolved not in self._accelerators:
            kwargs = self._accelerator_kwargs_for(pattern or resolved)
            self._accelerators[resolved] = create_kspace_accelerator(
                acceleration_type=resolved,
                num_timesteps=self.num_timesteps,
                **kwargs,
            )
        return self._accelerators[resolved]

    def _ensure_batch_dimensions(
        self,
        mask: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """_ensure_batch_dimensions.

        Args:
            mask (torch.Tensor): Description.
            batch_size (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:  # [C, H, W]
            mask = mask.unsqueeze(0)
        elif mask.dim() == 5:  # [B, C, H, W, D] - Support for 3D/multi-slice data
            pass
        elif mask.dim() != 4:
            raise ValueError(f"Unsupported mask shape: {mask.shape}")

        if mask.shape[0] == 1 and batch_size > 1:
            mask = mask.expand(batch_size, -1, -1, -1)
        return mask

    def generate_mask(
        self,
        shape: tuple[int, ...],
        seed: int | None = None,
        acceleration_factor: float = 4.0,
        pattern: str | None = None,
    ) -> torch.Tensor:
        """Generate a static mask for validation/testing.

        Adapts the `MaskGenerator` API to support the signature expected by
        diffusion strategies (shape, seed).

        Args:
            shape: Shape of the input tensor (B, C, H, W) or (C, H, W).
            seed: Random seed for reproducibility.
            acceleration_factor: Undersampling factor (default: 4.0).
            pattern: Mask pattern name (defaults to self.default_pattern).

        Returns:
            Binary mask tensor.
        """
        # Determine pattern
        accel_type = pattern or self.default_pattern

        # Instantiate MaskGenerator with seed
        generator = MaskGenerator(seed=seed)

        # Handle shape: MaskGenerator expects (H, W) or (C, H, W) usually
        # If shape is (B, C, H, W), extract (C, H, W)
        mask_shape = shape
        if len(shape) == 4:
            mask_shape = shape[1:]  # Drop batch dim

        # Translate accelerator/pattern vocabulary onto a concrete MaskType.
        # Pitfall #9 — an untranslatable pattern must RAISE, not silently
        # degrade to ``uniform_cartesian`` (which masked VD/Poisson/golden-angle
        # validation arms as equispaced-uniform with no log).
        # Two vocabularies meet here, and both are owned elsewhere: MaskType
        # names static patterns, SamplingPatternRegistry names accelerators.
        # A name is tried as a native MaskType first, then translated from its
        # canonical accelerator name. Anything neither knows must RAISE, never
        # degrade to uniform sampling (pitfall #9).
        raw = str(accel_type).strip().lower()
        try:
            mask_type = MaskType(raw).value
        except ValueError:
            canonical = SamplingPatternRegistry.resolve(raw)  # raises if unknown
            member = ACCELERATOR_TO_MASK_TYPE.get(canonical)
            if member is None:
                supported = sorted({m.value for m in MaskType} | set(ACCELERATOR_TO_MASK_TYPE))
                raise ValueError(
                    f"KSpaceMaskGenerator.generate_mask cannot render pattern "
                    f"{accel_type!r} (canonical {canonical!r}) on the static-mask "
                    f"path: it is accelerator-only and has no timestep-free "
                    f"equivalent. Supported here: {supported}. Configure a "
                    f"static-capable pattern or supply a precomputed mask."
                ) from None
            mask_type = member.value

        try:
            mask = generator.generate_mask(
                mask_type=mask_type,
                shape=mask_shape,
                acceleration_factor=acceleration_factor,
            )
        except ValueError as exc:
            valid = sorted(m.value for m in MaskType)
            raise ValueError(
                f"KSpaceMaskGenerator.generate_mask cannot render pattern "
                f"{accel_type!r} on the static-mask path. Supported MaskType "
                f"values: {valid}. Accelerator-only patterns (poisson_disk, "
                f"golden_angle, fractional_variable_density, nested, multi_mask, "
                f"…) are not available here — configure a MaskType-compatible "
                f"pattern or supply a precomputed mask in the batch."
            ) from exc

        # Use internal helpers to ensure correct dimensions
        if len(shape) == 4:
            batch_size = shape[0]
            channels = shape[1]

            # First ensure batch dim
            mask = self._ensure_batch_dimensions(mask, batch_size)

            # Then ensure channels
            mask = self.expand_mask_to_channels(mask, channels)

        return mask.to(self.device, non_blocking=True)

    def generate_acceleration_mask(
        self,
        timestep: int,
        image_shape: tuple[int, int],
        acceleration_factor: int = 4,
        pattern: str | None = None,
    ) -> torch.Tensor:
        """Generate an acceleration mask for a given timestep.

        The ``acceleration_factor`` argument is retained for API compatibility
        but the sparsity is ultimately driven by the accelerator configuration.
        """

        if acceleration_factor != 4 and not self._warned_about_factor:
            warnings.warn(
                (
                    "acceleration_factor is ignored when using "
                    "ColdDiffusionAccelerator; configure the accelerator "
                    "directly instead."
                ),
                stacklevel=2,
            )
            self._warned_about_factor = True

        # One owner for the timestep bound (non-negotiable 17). Every CPU entry
        # point funnels through here -- the slow path of ``generate_batch_masks``,
        # ``describe_ladder``, ``_cascade_masks``, the dynamic-mask loop -- and
        # before this an out-of-range timestep was answered SILENTLY with a mask
        # (the accelerator clamps its own schedule lookup), while the same
        # timestep on the CUDA table path raised a device-side index error. Two
        # answers to one question, and the silent one was the default (#1509).
        # The check is free here: ``timestep`` is already a host int, so it costs
        # no sync, which is why it does not live on the fast path.
        if not 0 <= int(timestep) < self.num_timesteps:
            raise ValueError(
                f"timestep {int(timestep)} is outside the schedule "
                f"[0, {self.num_timesteps}). The cascade is only defined on its "
                "own horizon; a timestep past the end used to return the "
                "endpoint mask silently. Check that the caller's "
                "``num_timesteps`` matches this generator's."
            )
        accelerator = self._get_accelerator(pattern or self.default_pattern)
        height, width = image_shape
        mask = accelerator.get_acceleration_mask(
            (1, height, width),
            int(timestep),
            device=self.device,
        )
        return mask.to(self.device, non_blocking=True)

    def generate_batch_masks(
        self,
        batch_size: int,
        timesteps: torch.Tensor,
        image_shape: tuple[int, int],
        acceleration_factor: int = 4,
        pattern: str | None = None,
    ) -> torch.Tensor:
        """Generate masks for every sample in a batch.

        Returns a tensor shaped ``[batch_size, 1, H, W]`` that can be expanded
        to match coil/channel counts via :meth:`expand_mask_to_channels`.
        """

        # One owner for the batch/timestep agreement, ahead of the device branch
        # so BOTH paths answer identically (#1509). They did not: with
        # ``numel() < batch_size`` the CPU path raised ``IndexError`` from
        # ``timestep_list[i]``, while the CUDA path truncated to whatever was
        # there and returned a SHORT tensor -- ``[1, 1, H, W]`` for a batch of 4
        # -- which then broadcast, silently degrading all four samples with
        # sample 0's mask. ``numel()`` is tensor metadata, so this costs no host
        # sync and stays on the fast path (non-negotiable 9).
        if timesteps.numel() < batch_size:
            raise ValueError(
                f"generate_batch_masks got {timesteps.numel()} timestep(s) for "
                f"batch_size={batch_size}; each sample needs its own timestep. "
                "A shorter tensor used to broadcast one sample's mask across "
                "the batch on CUDA and raise IndexError on CPU."
            )

        # Fast path: serve from a memoised on-device cascade so the timestep
        # tensor never leaves the accelerator (the host copy below is BLOCKING;
        # a Scalene profile charged 24.24 % of a run to it, against 0.8 s of
        # actual mask building). Gated on BOTH sides being off-CPU: a host-side
        # table would need the index moved back, reintroducing this very sync.
        # See ``mask_table_cache`` for why memoising is safe here and is not
        # safe on the dynamic-mask training path.
        if timesteps.device.type != "cpu" and self.device.type != "cpu":
            declared_pattern = pattern or self.default_pattern
            accelerator = self._get_accelerator(declared_pattern)
            # Key on the RESOLVED type, not the declared spelling: 9 of the 19
            # canonical types have more than one accepted spelling, and
            # ``_get_accelerator`` already caches one instance per resolved type,
            # so two spellings share an accelerator and produce identical masks.
            # Keying on the spelling held a second, bit-identical table.
            key = MaskTableCache.build_key(
                self._resolve_acceleration_type(declared_pattern),
                image_shape,
                acceleration_factor,
                self.device,
                accelerator,
            )
            table = self._mask_tables.table_for(
                key,
                self.num_timesteps,
                lambda t: self.generate_acceleration_mask(
                    t, image_shape, acceleration_factor, pattern
                ),
            )
            index = (
                timesteps.detach()
                .reshape(-1)[:batch_size]
                .to(device=table.device, dtype=torch.long)
            )
            # A timestep outside [0, num_timesteps) raises a device-side index
            # error here. That is deliberate: it fails loud rather than
            # clamping to a neighbouring mask (non-negotiable 3).
            return table.index_select(0, index)

        # Single host sync for the whole batch instead of one ``.item()`` per
        # sample inside the loop (the loop runs every training iteration for
        # cold-diffusion arms; per-element ``.item()`` serialises the GPU).
        timestep_list = timesteps.detach().to("cpu", non_blocking=False).tolist()
        masks = [
            self.generate_acceleration_mask(
                int(timestep_list[i]),
                image_shape,
                acceleration_factor,
                pattern,
            )
            for i in range(batch_size)
        ]
        return torch.stack(masks, dim=0)

    def apply_mask_to_kspace(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply undersampling mask to k-space data.

        Args:
            kspace: K-space data of shape [B, C, H, W]
            mask: Mask of shape [B, 1, H, W] or [1, H, W]

        Returns:
            Undersampled k-space of shape [B, C, H, W]

        """
        mask = self._ensure_batch_dimensions(mask, kspace.shape[0])
        mask = self.expand_mask_to_channels(mask, kspace.shape[1])
        return kspace * mask

    def expand_mask_to_channels(
        self,
        mask: torch.Tensor,
        num_channels: int,
    ) -> torch.Tensor:
        """Expand mask to match the number of channels in k-space data.

        Args:
            mask: Input mask of shape [B, 1, H, W] or [1, H, W]
            num_channels: Number of channels in k-space data

        Returns:
            Expanded mask of shape [B, num_channels, H, W]

        """
        target_batch = mask.shape[0] if mask.dim() >= 4 else 1
        # [FIX] Ensure mask has 4 dims [B, 1, H, W] before checking channels
        mask = self._ensure_batch_dimensions(mask, target_batch)

        if mask.shape[1] == 1 and num_channels > 1:
            if mask.dim() == 5:
                # [FIX] Handle 5D [B, 1, H, W, D] -> [B, C, H, W, D]
                mask = mask.expand(-1, num_channels, -1, -1, -1)
            else:
                # Standard 4D [B, 1, H, W] -> [B, C, H, W]
                mask = mask.expand(-1, num_channels, -1, -1)
        elif mask.shape[1] != num_channels:
            # If mask has 320 channels and we want 2? Mismatch.
            # Only expand if it's 1.
            pass

        return mask

    def apply_acceleration_to_kspace(
        self,
        kspace: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply acceleration (undersampling) mask to k-space data.

        Args:
            kspace: K-space data of shape [B, C, H, W]
            mask: Acceleration mask of shape [B, 1, H, W] or [B, C, H, W]

        Returns:
            Undersampled k-space of shape [B, C, H, W]

        """
        # Ensure mask has correct shape
        mask = self.expand_mask_to_channels(mask, kspace.shape[1])
        return kspace * mask

    def generate_and_apply_masks(
        self,
        kspace: torch.Tensor,
        timesteps: torch.Tensor,
        provided_mask: torch.Tensor | None = None,
        acceleration_factor: int = 4,
        pattern: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate acceleration masks and apply them to k-space data.

        Args:
            kspace: K-space data of shape [B, C, H, W]
            timesteps: Timestep tensor of shape [B]
            provided_mask: Optional pre-computed mask
            acceleration_factor: Undersampling factor
            pattern: Mask pattern type

        Returns:
            Tuple of (undersampled_kspace, acceleration_masks)

        """
        batch_size = kspace.shape[0]
        image_shape = (kspace.shape[-2], kspace.shape[-1])

        if provided_mask is not None:
            # Use provided mask
            acceleration_masks = self.expand_mask_to_channels(
                provided_mask,
                kspace.shape[1],
            )
        else:
            # Generate new masks
            acceleration_masks = self.generate_batch_masks(
                batch_size,
                timesteps,
                image_shape,
                acceleration_factor,
                pattern,
            )
            acceleration_masks = self.expand_mask_to_channels(
                acceleration_masks,
                kspace.shape[1],
            )

        # Apply acceleration
        undersampled_kspace = self.apply_acceleration_to_kspace(
            kspace,
            acceleration_masks,
        )

        return undersampled_kspace, acceleration_masks


def create_kspace_mask_generator(
    num_timesteps: int,
    device: torch.device | None = None,
    default_pattern: str = "linear",
    accelerator_kwargs: dict[str, Any] | None = None,
    pattern_overrides: dict[str, dict[str, Any]] | None = None,
) -> KSpaceMaskGenerator:
    """Factory function for creating k-space mask generator.

    Args:
        num_timesteps: Number of diffusion timesteps
        device: Device for tensor operations
        default_pattern: Default pattern/acceleration type to use when callers
            omit the pattern argument.
        accelerator_kwargs: Global kwargs forwarded to each accelerator
            instance.
        pattern_overrides: Optional per-pattern kwargs that override the global
            accelerator configuration.

    Returns:
        Configured KSpaceMaskGenerator instance

    """
    return KSpaceMaskGenerator(
        num_timesteps=num_timesteps,
        device=device,
        default_pattern=default_pattern,
        accelerator_kwargs=accelerator_kwargs,
        pattern_overrides=pattern_overrides,
    )
