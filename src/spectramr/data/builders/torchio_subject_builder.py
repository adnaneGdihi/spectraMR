"""Phase T4: Subject Factory - Unified TorchIO Subject creation.

Consolidates duplicated Subject creation logic from:
- UniversalMRIDataset (110 lines of Subject dict building)
- PreprocessedMRIDataset (90 lines of Subject dict building)
- Medical Volume Dataset (50 lines)

Strategy Pattern for Subject Creation:
- SubjectBuilder: Abstract base for building TorchIO Subjects
- FastMRISubjectBuilder: K-space, sensitivity, paired targets
- PreprocessedSubjectBuilder: From preprocessed directories
- MedicalVolumeSubjectBuilder: Medical volume data
- SubjectBuilderFactory: Registry-based dispatch

Benefits:
- Eliminate 250+ lines of duplicated Subject creation logic
- Centralized affine enforcement (critical for Queue)
- Single place to add new data types
- Easier to test Subject creation separately
- Consistent handling of complex data types

.. mermaid::

    flowchart TD
        Record[Data Record] --> Factory{SubjectBuilderFactory}
        Factory -->|fastmri| FastMRI[FastMRISubjectBuilder]
        Factory -->|preprocessed| Pre[PreprocessedSubjectBuilder]
        Factory -->|medical_volume| Med[MedicalVolumeSubjectBuilder]

        FastMRI -->|Load K-Space| IO1[IO Strategy]
        FastMRI -->|Load Target| IO2[IO Strategy]
        FastMRI -->|Load Sensitivity| IO3[IO Strategy]

        IO1 & IO2 & IO3 --> Affine{Enforce Affine}
        Affine --> Subject[TorchIO Subject]

        Pre -->|Load Files| Tensors
        Tensors --> Affine

        Med -->|Load NIfTI| Tensors
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
import torchio as tio

from spectramr.data.metadata.physics_registry import get_physics_vector
from spectramr.data.transforms.image_domain_crop import (
    crop_or_pad_kspace_in_image_domain,
    validate_image_size,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

logger = logging.getLogger(__name__)


class SubjectBuilder(ABC):
    """Abstract base for building TorchIO Subjects.

    Each builder handles a specific data source type and knows how to:
    1. Load data from files/records
    2. Convert to TorchIO format
    3. Build subject dict with proper affines
    4. Enforce spatial consistency
    """

    @abstractmethod
    def build(self, record: dict[str, Any]) -> tio.Subject:
        """Build a Subject from a data record.

        Args:
            record: Data record (dict with file paths, metadata, etc.)

        Returns:
            TorchIO Subject with all images and metadata

        Raises:
            FileNotFoundError: If required files missing
            ValueError: If data format invalid
        """
        pass

    def _enforce_consistent_affines(self, subject: tio.Subject) -> tio.Subject:
        """Force all images in subject to share same affine.

        CRITICAL: TorchIO Queue requires all images to have identical affines.
        This method ensures the requirement is met by extracting the first
        image's affine and applying it to all others.

        Args:
            subject: Subject with potentially inconsistent affines

        Returns:
            Subject with all images using same affine
        """
        images = subject.get_images()
        if not images:
            return subject

        # Use first image's affine as reference
        reference_affine = images[0].affine.copy()

        # Apply reference affine to all images
        for image_name in subject.get_images_names():
            image = subject[image_name]
            if not np.allclose(image.affine, reference_affine):
                logger.debug(f"[SUBJECT] Affine mismatch in {image_name}, applying reference")
                # [FIX] Recreate image safely based on its class
                # TorchIO images hardcode their type (ScalarImage=INTENSITY, LabelMap=LABEL)
                # so we avoid manual 'type' assignment which can cause copy.copy failures.
                if isinstance(image, tio.LabelMap):
                    new_image = tio.LabelMap(tensor=image.tensor, affine=reference_affine)
                else:
                    new_image = tio.ScalarImage(tensor=image.tensor, affine=reference_affine)

                # Copy path if it exists (metadata like path is safe to copy)
                if hasattr(image, "path"):
                    new_image.path = image.path

                subject[image_name] = new_image

        return subject

    def _ensure_4d_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is 4D for TorchIO: (C, H, W, D).

        Args:
            tensor: Input tensor (any dimension)

        Returns:
            4D tensor
        """
        if tensor.ndim == 2:  # (H, W)
            return tensor.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        elif tensor.ndim == 3:  # (C, H, W) or (S, H, W)
            return tensor.unsqueeze(-1)  # Add depth
        elif tensor.ndim == 4:
            return tensor
        else:
            # 5D+: reduce to first batch element
            return tensor[0] if tensor.ndim > 4 else tensor

    def _fastmri_to_torchio(self, tensor: torch.Tensor, single_slice: bool = False) -> torch.Tensor:
        """Convert FastMRI layout (S, H, W) to TorchIO (C, H, W, D).

        FastMRI stores volumes as (Slices, Height, Width).
        TorchIO expects (Channels, Height, Width, Depth).
        Slices should map to Depth dimension.

        Args:
            tensor: FastMRI tensor (2D, 3D, or 4D).
            single_slice: True when this is a per-slice read (the record carried
                ``slice_index``, so the IO strategy already sliced the volume).
                A 3-D tensor is then ``(Coils, H, W)`` — the leading axis is
                COILS, not slices — and must map to ``(Coils, H, W, D=1)``.
                Without this flag a 3-D multicoil slice is mis-read as an
                ``(S, H, W)`` single-coil volume, dumping the coils onto the
                depth axis and forcing ``channels=1``.

        Returns:
            TorchIO-compatible 4D tensor.
        """
        if tensor.ndim == 2:  # (H, W) single slice
            return tensor.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        elif tensor.ndim == 3:
            if single_slice:
                # Per-slice multicoil read: (Coils, H, W) -> (Coils, H, W, 1).
                return tensor.unsqueeze(-1)
            # Full single-coil volume (S, H, W): map Slices -> Depth.
            # (S, H, W) -> (1, H, W, S)
            return tensor.permute(1, 2, 0).unsqueeze(0)
        elif tensor.ndim == 4:  # (S, C, H, W) -> (C, H, W, S)
            # FastMRI 4D is (Slices, Coils, Height, Width).
            # TorchIO expects (Channels, Height, Width, Depth).
            # We must map:
            # - Slices -> Depth (dim 3)
            # - Coils -> Channels (dim 0)
            # - Height -> Height (dim 1)
            # - Width -> Width (dim 2)
            # (S, C, H, W) -> (C, H, W, S)
            return tensor.permute(1, 2, 3, 0)
        else:
            # 5D+: reduce
            return tensor[0] if tensor.ndim > 4 else tensor


class FastMRISubjectBuilder(SubjectBuilder):
    """Builds Subjects from FastMRI data.

    Handles:
    - K-space data (via IO strategy)
    - Sensitivity maps (optional)
    - Paired targets (for ULF→HF experiments)
    - Proper affine enforcement
    """

    def __init__(
        self,
        primary_io,
        target_io=None,
        sensitivity_io=None,
        coil_processing_mode: str = "none",
        num_virtual_coils: int = 4,
        svd_calibration_lines: int | None = None,
        coil_processing=None,
        image_domain_crop_or_pad: tuple[int, int] | None = None,
    ):
        """Initialize with IO strategies.

        Args:
            primary_io: IO strategy for k-space/primary data
            target_io: IO strategy for target data (optional)
            sensitivity_io: IO strategy for sensitivity maps (optional)
            coil_processing_mode: How to handle multi-coil k-space at load time:
                - "none": Keep original complex multi-coil data
                - "flatten": Convert (Coils, H, W, D) complex to (2*Coils, H, W, D) real
                - "rss": RSS coil combination to (2, H, W, D) real
                - "svd": Compress into virtual coils
            num_virtual_coils: Number of virtual coils when mode="svd"
            image_domain_crop_or_pad: When set, the ``(height, width)`` the complex
                k-space is resized to through its image, before coil processing
                (``data.processing.image_domain_crop_or_pad``).
            normalize_kspace: Whether to normalize k-space data by magnitude percentile
            kspace_percentile: Percentile for normalization (default 0.99 to avoid DC spike)
            log_scaling: Apply phase-preserving log1p magnitude compression after the
                percentile divide (tames the ~200x DC dynamic range). The training
                strategy inverts it (``decompress_kspace_robust``) before IFFT/metrics.
        """
        self.primary_io = primary_io
        self.target_io = target_io
        self.sensitivity_io = sensitivity_io
        self.coil_processing_mode = coil_processing_mode
        self.num_virtual_coils = num_virtual_coils
        self.svd_calibration_lines = svd_calibration_lines
        # Resolved physics.coil_processing block (the 4-axis SSOT). When set and it
        # represents a NON-legacy axis combination, the composable data-load
        # pipeline runs; legacy combinations fall through to the byte-parity
        # mode-based branches below. None → pure legacy coil_processing_mode path.
        self.coil_processing = coil_processing
        self.image_domain_crop_or_pad = (
            validate_image_size(image_domain_crop_or_pad)
            if image_domain_crop_or_pad is not None
            else None
        )

    def _is_nifti_strategy(self, io_strategy, path: str = "") -> bool:
        """Check if strategy is NIfTI-based."""
        if path and str(path).lower().endswith((".nii", ".nii.gz")):
            return True
        name = io_strategy.__class__.__name__
        if "NiftiStrategy" in name or "NifTi" in name:
            return True
        return False

    # The 6 legacy data-load combinations → their mode name. The DATA-LOAD combine
    # is none/rss only (``sense`` ≡ ``none`` here — SENSE needs smaps that exist
    # only post-model, so the data-load keeps coils and the recon sense-combines).
    _LEGACY_COIL_COMBOS: ClassVar[dict[tuple[str, str, str, str], str]] = {
        ("none", "none", "kspace", "complex"): "none",
        ("none", "none", "kspace", "real_interleaved"): "flatten",
        ("svd", "none", "kspace", "real_interleaved"): "svd",
        ("none", "rss", "source", "magnitude"): "magnitude",
        ("none", "rss", "kspace", "real_interleaved"): "rss",
        ("none", "rss", "image", "magnitude"): "rss_image",
    }

    def _coil_block_to_legacy_mode(self) -> str | None:
        """Reverse-lookup the resolved block's data-load axes → legacy mode (or None
        for a genuinely-new combination handled by the composable pipeline)."""
        cp = self.coil_processing
        combine = cp.combine.method
        if combine == "sense":  # data-load can't sense-combine → keep coils
            combine = "none"
        sig = (cp.compression.method, combine, cp.output.domain, cp.output.channels)
        return self._LEGACY_COIL_COMBOS.get(sig)

    def _composable_coil_processing(self, tensor: torch.Tensor, is_kspace: bool) -> torch.Tensor:
        """Data-load pipeline for non-legacy axis combinations: compression →
        combine(none/rss) → output(domain, channels). ``sense`` is treated as
        ``none`` (kept for the recon stage). Not parity-constrained — the 6 legacy
        combos never reach here (they use the byte-parity branches)."""
        cp = self.coil_processing
        combine = cp.combine.method
        if combine == "sense":
            combine = "none"
        domain, channels = cp.output.domain, cp.output.channels

        coils = tensor
        if not torch.is_complex(coils) and coils.shape[0] % 2 == 0:
            coils = torch.complex(coils[0::2], coils[1::2])

        # Stage 1: compression (k-space, complex).
        if cp.compression.method == "svd" and torch.is_complex(coils):
            nvc = cp.compression.num_virtual_coils
            if coils.shape[0] > nvc:
                from spectramr.data.transforms.coil_compression import SVDCoilCompression

                compr = SVDCoilCompression(
                    num_virtual_coils=nvc,
                    calibration_lines=cp.compression.calibration_lines,
                ).to(coils.device)
                coils = compr(coils.permute(3, 0, 1, 2)).permute(1, 2, 3, 0)
        elif cp.compression.method == "gcc":
            raise NotImplementedError(
                "physics.coil_processing.compression.method='gcc' not implemented"
            )

        # Stage 2: combine.
        if combine == "rss":
            if domain == "source":
                combined = torch.sqrt((coils.abs() ** 2).sum(0, keepdim=True))
            else:
                imgs = ifft2c(coils.permute(0, 3, 1, 2)).permute(0, 2, 3, 1) if is_kspace else coils
                combined = torch.sqrt((imgs.abs() ** 2).sum(0, keepdim=True))
                if domain == "kspace":
                    combined = fft2c(combined.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:  # none
            combined = coils

        # Stage 3: output channels.
        if channels == "complex":
            return combined
        if channels == "real_interleaved":
            n = combined.shape[0]
            out = torch.empty(
                2 * n, *combined.shape[1:], dtype=torch.float32, device=combined.device
            )
            out[0::2] = combined.real
            out[1::2] = combined.imag
            return out
        # magnitude
        return combined.abs().float() if torch.is_complex(combined) else combined.float()

    def _crop_or_pad_primary(
        self, tensor: torch.Tensor, is_kspace: bool, sensitivity: torch.Tensor | None
    ) -> torch.Tensor:
        """Resize the complex k-space through its image; refuse what cannot follow."""
        if not is_kspace:
            raise ValueError(
                "data.processing.image_domain_crop_or_pad resizes raw k-space, but "
                "this record served an image. Drop the knob or serve k-space."
            )
        if sensitivity is not None:
            raise ValueError(
                "data.processing.image_domain_crop_or_pad would leave the loaded "
                "sensitivity maps on the uncropped grid. Disable sensitivity loading "
                "or resize the maps upstream."
            )
        return crop_or_pad_kspace_in_image_domain(tensor, self.image_domain_crop_or_pad)

    def _apply_coil_processing(
        self,
        tensor: torch.Tensor,
        sensitivity: torch.Tensor | None = None,
        is_kspace: bool = True,
    ) -> torch.Tensor:
        """Apply coil processing based on configured mode.

        Handles both k-space and image domain multi-coil data.

        Args:
            tensor: Input tensor (Coils, H, W, D) potentially complex
            sensitivity: Optional sensitivity maps for SENSE
            is_kspace: If True, tensor is in k-space domain. If False, image domain.

        Returns:
            Processed tensor in format suitable for model
        """
        # SSOT dispatch: when a resolved physics.coil_processing block is present,
        # it is authoritative. A legacy axis combination overrides the legacy
        # fields and falls through to the byte-parity branches below; a new
        # combination takes the composable pipeline.
        if self.coil_processing is not None:
            mode = self._coil_block_to_legacy_mode()
            if mode is None:
                return self._composable_coil_processing(tensor, is_kspace)
            self.coil_processing_mode = mode
            self.num_virtual_coils = self.coil_processing.compression.num_virtual_coils
            self.svd_calibration_lines = self.coil_processing.compression.calibration_lines

        logger.debug(
            f"[_apply_coil_processing] mode={self.coil_processing_mode}, input_shape={tensor.shape}, is_complex={torch.is_complex(tensor)}, is_kspace={is_kspace}"
        )

        if self.coil_processing_mode == "none":
            logger.debug("[_apply_coil_processing] mode=none, returning original tensor")
            return tensor

        # Handle real-valued tensors. For magnitude-producing modes
        # (``rss_image``, ``magnitude``, ``rss``) we MUST still reduce a
        # real-stacked 2*C-channel tensor to a 1-channel magnitude image
        # so the model receives the channel count its registry capability
        # advertises. The earlier early-return passed real-stacked inputs
        # through unchanged, which broke every ``rss_image``-mode VF
        # experiment in the 2026-05-05 cluster smoke (model expects 1 ch,
        # data arrived as 2 ch real+imag of 1 coil → ``Conv2d expected 1
        # channel got 2`` in 9 arms: m2/m3m4/m6/m7/m8 + hyper_mamba_meta /
        # method_b_hyper_mamba / pipeline_a_koopman_advection / vf_19).
        if not torch.is_complex(tensor):
            mag_modes = {"rss_image", "magnitude", "rss"}
            if self.coil_processing_mode in mag_modes and tensor.shape[0] >= 2:
                # Treat even-channel real tensor as interleaved [R0,I0,R1,I1,...]
                # (the codebase-wide layout); odd-channel as already-magnitude.
                C = tensor.shape[0]
                if C % 2 == 0:
                    real = tensor[0::2]
                    imag = tensor[1::2]
                    coil_complex = torch.complex(real, imag)
                    if is_kspace and self.coil_processing_mode == "rss_image":
                        kspace_t = coil_complex.permute(0, 3, 1, 2)
                        coil_complex = ifft2c(kspace_t).permute(0, 2, 3, 1)
                    rss = torch.sqrt((coil_complex.abs() ** 2).sum(dim=0, keepdim=True))
                    return rss.float()
                else:
                    # odd channels — already a magnitude / single coil; RSS over channels
                    return torch.sqrt((tensor.float() ** 2).sum(dim=0, keepdim=True))
            if tensor.shape[0] > 2:
                logger.debug(f"[COIL] Real tensor with {tensor.shape[0]} channels, passing through")
            return tensor

        C, H, W, D = tensor.shape

        if self.coil_processing_mode == "flatten":
            # (Coils, H, W, D) complex -> (2*Coils, H, W, D) real
            # Works for both k-space and image domain
            real_data = torch.zeros(2 * C, H, W, D, dtype=torch.float32, device=tensor.device)
            real_data[0::2] = tensor.real
            real_data[1::2] = tensor.imag
            domain_str = "k-space" if is_kspace else "image"
            logger.debug(
                f"[COIL] Flattened ({domain_str}): ({C}, {H}, {W}, {D}) complex -> ({2 * C}, {H}, {W}, {D}) real"
            )
            return real_data

        elif self.coil_processing_mode == "rss":
            if is_kspace:
                # K-space domain: IFFT to image, RSS combine, FFT back
                # 1. IFFT to image domain (per coil) using physics module
                kspace_transposed = tensor.permute(0, 3, 1, 2)  # (C, D, H, W)
                images = ifft2c(kspace_transposed)

                # 2. RSS combine: sqrt(sum(|img_c|^2))
                rss_combined = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=0, keepdim=True))

                # 3. FFT back to k-space using physics module
                combined_kspace = fft2c(rss_combined)

                # Transpose back: (1, D, H, W) -> (1, H, W, D)
                combined_kspace = combined_kspace.permute(0, 2, 3, 1)

                # Split to Real/Imag: (1, H, W, D) complex -> (2, H, W, D) real
                real_imag = torch.stack(
                    [combined_kspace.real.squeeze(0), combined_kspace.imag.squeeze(0)],
                    dim=0,
                )
                logger.debug(
                    f"[RSS result] real_imag.shape={real_imag.shape}, combined_kspace.shape={combined_kspace.shape}"
                )
                logger.debug(
                    f"[COIL] RSS (k-space): ({C}, {H}, {W}, {D}) complex -> ({2}, {H}, {W}, {D}) real"
                )
                return real_imag
            else:
                # Image domain: Direct RSS combination (no FFT needed)
                # Input: (Coils, H, W, D) complex images
                # RSS combine: sqrt(sum(|img_c|^2)) -> (1, H, W, D) real magnitude
                rss_combined = torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=0, keepdim=True))

                # For image domain, output is typically single-channel magnitude
                # But for consistency with k-space path, we can output (2, H, W, D) with zero imag
                # OR just (1, H, W, D) real. Let's use (1, H, W, D) for image domain.
                logger.debug(
                    f"[COIL] RSS (image): ({C}, {H}, {W}, {D}) complex -> ({1}, {H}, {W}, {D}) real"
                )
                return rss_combined.float()

        elif self.coil_processing_mode == "svd":
            if is_kspace:
                from spectramr.data.transforms.coil_compression import SVDCoilCompression

                # We compress to num_virtual_coils for complex data which splits to 2 real channels each
                num_virtual_coils = self.num_virtual_coils

                if num_virtual_coils < C:
                    # SVD expects [B, C, H, W], we have [C, H, W, D]
                    # Treat D as Batch dimension: [D, C, H, W]
                    kspace_bchw = tensor.permute(3, 0, 1, 2)

                    compressor = SVDCoilCompression(
                        num_virtual_coils=num_virtual_coils,
                        calibration_lines=self.svd_calibration_lines,
                    ).to(tensor.device)
                    k_virtual = compressor(kspace_bchw)

                    # Back to [C_virtual, H, W, D]
                    compressed_tensor = k_virtual.permute(1, 2, 3, 0)

                    logger.debug(
                        f"[COIL] SVD compressed (k-space): ({C}, {H}, {W}, {D}) complex -> ({num_virtual_coils}, {H}, {W}, {D}) complex"
                    )
                else:
                    compressed_tensor = tensor
                    num_virtual_coils = C

                # Split complex coils to 2*num_virtual_coils real channels (real/imag)
                real_imag = torch.empty(
                    2 * num_virtual_coils,
                    H,
                    W,
                    D,
                    dtype=torch.float32,
                    device=tensor.device,
                )
                real_imag[0::2] = compressed_tensor.real
                real_imag[1::2] = compressed_tensor.imag

                logger.debug(f"[COIL] SVD result split to real/imag: {real_imag.shape}")
                return real_imag
            else:
                # Image domain: Currently not supported directly by SVDCoilCompression
                # Fallback to direct RSS
                logger.warning(
                    "[COIL] SVD requested on Image Domain data. Falling back to RSS magnitude."
                )
                rss_combined = torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=0, keepdim=True))
                return rss_combined.float()

        elif self.coil_processing_mode == "rss_image":
            # RSS coil combination returning a 1-ch magnitude image.
            # k-space inputs are IFFT-ed first; image inputs are used directly.
            if is_kspace:
                kspace_transposed = tensor.permute(0, 3, 1, 2)
                images = ifft2c(kspace_transposed)
                rss_combined = torch.sqrt(torch.sum(torch.abs(images) ** 2, dim=0, keepdim=True))
                rss_combined = rss_combined.permute(0, 2, 3, 1)
            else:
                rss_combined = torch.sqrt(torch.sum(torch.abs(tensor) ** 2, dim=0, keepdim=True))
            logger.debug(
                f"[COIL] RSS image: ({C}, {H}, {W}, {D}) complex -> "
                f"({1}, {H}, {W}, {D}) real magnitude image"
            )
            return rss_combined.float()

        elif self.coil_processing_mode == "magnitude":
            # Simple magnitude: works for both domains
            # (Coils, H, W, D) complex -> (Coils, H, W, D) real magnitude
            magnitude = torch.abs(tensor)
            # RSS across coils for single output
            rss = torch.sqrt(torch.sum(magnitude**2, dim=0, keepdim=True))
            domain_str = "k-space" if is_kspace else "image"
            logger.debug(
                f"[COIL] Magnitude ({domain_str}): ({C}, {H}, {W}, {D}) complex -> ({1}, {H}, {W}, {D}) real"
            )
            return rss

        else:
            from spectramr.infrastructure.errors import ConfigurationError

            raise ConfigurationError(
                f"[COIL] Unknown coil_processing_mode: {self.coil_processing_mode}. "
                f"Valid modes are: 'flatten', 'rss', 'rss_image', 'svd', 'magnitude'. "
                f"Ensure your experiment configuration provides a valid mode."
            )

    def build(self, record: dict[str, Any]) -> tio.Subject:
        """Build Subject from FastMRI record.

        Args:
            record: Record dict with paths: primary_path, target_path, sensitivity_path

        Returns:
            TorchIO Subject with k-space, target, sensitivity
        """
        subject_dict = {}

        # DEBUG: Log coil processing mode at subject builder
        logger.debug(
            f"[FastMRISubjectBuilder.build] coil_processing_mode={self.coil_processing_mode}"
        )

        # Load primary data (can be k-space or image domain).
        # Forward the full record as metadata so per-slice records
        # (``variant: 2d_slices`` → one record per ``slice_index``) hit the
        # lazy single-slice read in FastMRIH5Strategy / NiftiStrategy instead
        # of decoding the whole volume once per slice per epoch. No-op when
        # ``slice_index`` is absent (strategies fall back to the full read).
        primary_data = self.primary_io.load(record["primary_path"], metadata=record)
        affine = primary_data.get("affine", np.eye(4))

        # A per-slice record (``variant: 2d_slices``) makes the IO strategy
        # return an already-sliced tensor whose leading axis is COILS, not
        # slices. Thread this through so ``_fastmri_to_torchio`` maps a 3-D
        # ``(Coils, H, W)`` slice to ``(Coils, H, W, 1)`` instead of dumping the
        # coils onto the depth axis.
        single_slice = record.get("slice_index") is not None

        # Determine data domain and load appropriate tensor
        # Priority: 'kspace' key for k-space data, 'data' or 'image' for image domain
        is_kspace = False
        if "kspace" in primary_data and primary_data["kspace"] is not None:
            logger.debug("[BUILD] Loading 'kspace' key (will normalize if enabled)")
            primary_tensor = self._fastmri_to_torchio(
                primary_data["kspace"], single_slice=single_slice
            )
            is_kspace = True
            primary_key = "kspace"
        elif "data" in primary_data and primary_data["data"] is not None:
            # [FIX] NIfTI data is already in correct orientation (C, W, H, D) or (1, W, H, D)
            # _fastmri_to_torchio assumes (S, C, H, W) and permutes, which breaks NIfTI
            logger.debug("[BUILD] Loading 'data' key (is_kspace=False, will NOT normalize)")
            if self._is_nifti_strategy(self.primary_io, record.get("primary_path", "")):
                primary_tensor = self._ensure_4d_tensor(primary_data["data"])
            else:
                primary_tensor = self._fastmri_to_torchio(
                    primary_data["data"], single_slice=single_slice
                )

            # data could be RSS (image) or could be complex image
            # Check if complex to determine domain
            is_kspace = False  # 'data' key typically means image domain
            primary_key = "image"
        elif "image" in primary_data and primary_data["image"] is not None:
            # Same fix for 'image' key if it comes from NIfTI (unlikely but possible)
            logger.debug("[BUILD] Loading 'image' key (is_kspace=False, will NOT normalize)")
            if self._is_nifti_strategy(self.primary_io, record.get("primary_path", "")):
                primary_tensor = self._ensure_4d_tensor(primary_data["image"])
            else:
                primary_tensor = self._fastmri_to_torchio(
                    primary_data["image"], single_slice=single_slice
                )
            is_kspace = False
            primary_key = "image"
        else:
            raise ValueError(
                f"No valid data found in primary_data. Keys: {list(primary_data.keys())}"
            )

        # Load sensitivity maps (optional, needed before coil processing)
        sensitivity = None
        if (
            "sensitivity_path" in record
            and record["sensitivity_path"] is not None
            and self.sensitivity_io
        ):
            try:
                smap_data = self.sensitivity_io.load(record["sensitivity_path"], metadata=record)
                sensitivity = self._ensure_4d_tensor(smap_data["data"])
                subject_dict["sensitivity"] = tio.ScalarImage(tensor=sensitivity, affine=affine)
            except Exception as e:
                logger.debug(f"Failed to load sensitivity maps: {e}")

        if self.image_domain_crop_or_pad is not None:
            primary_tensor = self._crop_or_pad_primary(primary_tensor, is_kspace, sensitivity)

        # Apply coil processing at load time (critical for TorchIO sampling)
        logger.debug(
            f"[BUILD] Before coil processing: is_kspace={is_kspace}, shape={primary_tensor.shape}"
        )
        primary_tensor = self._apply_coil_processing(
            primary_tensor, sensitivity, is_kspace=is_kspace
        )
        logger.debug(f"[BUILD] After coil processing: shape={primary_tensor.shape}")

        # K-space normalization is NOT applied here. It is owned by
        # KSpaceNormalizationTransform, which TorchIOTransformBuilder appends for
        # the same ``data.normalize_kspace`` flag and which the dataset applies
        # via ``self.transform(subject)``. Normalizing at build time as well gave
        # a double percentile divide + double log1p, with the transform
        # overwriting ``kspace_scale`` so only half of it was invertible.
        # The subject builder matches and serves. -> docs/kspace_normalization_ssot.rst
        # Identity scale: overwritten by the transform with the scale it applied,
        # so the published value always describes the tensor served beside it.
        subject_dict["kspace_scale"] = torch.tensor(1.0)

        # Store with appropriate key based on domain
        if is_kspace:
            subject_dict["kspace"] = tio.ScalarImage(tensor=primary_tensor, affine=affine)
            # Add input alias
            subject_dict["input"] = subject_dict["kspace"]
        else:
            subject_dict["image"] = tio.ScalarImage(tensor=primary_tensor, affine=affine)
            # Also add as kspace key for backward compatibility with strategies expecting 'kspace'
            subject_dict["kspace"] = tio.ScalarImage(tensor=primary_tensor, affine=affine)
            # Add input alias
            subject_dict["input"] = subject_dict["image"]

        if self.image_domain_crop_or_pad is not None and any(
            key in primary_data for key in ("mask", "sampling_mask", "acceleration_mask")
        ):
            raise ValueError(
                f"{record.get('primary_path')} carries a stored sampling mask, which "
                "indexes the uncropped k-space grid and cannot follow an image-domain "
                "crop. Point image_domain_crop_or_pad arms at fully sampled data."
            )

        # [FIX] Propagate Mask if available (Critical for validation consistency)
        # Check both top-level and physics dict
        mask_tensor = None
        if "mask" in primary_data:
            mask_tensor = primary_data["mask"]
        elif "sampling_mask" in primary_data:
            mask_tensor = primary_data["sampling_mask"]
        elif "acceleration_mask" in primary_data:
            mask_tensor = primary_data["acceleration_mask"]

        # Check in physics dict if not found
        if mask_tensor is None and "physics" in primary_data:
            phys = primary_data["physics"]
            if isinstance(phys, dict):
                if "mask" in phys:
                    mask_tensor = phys["mask"]
                elif "sampling_mask" in phys:
                    mask_tensor = phys["sampling_mask"]

        if mask_tensor is not None:
            # Ensure mask is tensor
            if not isinstance(mask_tensor, torch.Tensor):
                mask_tensor = torch.tensor(mask_tensor)

            # Ensure 4D for TorchIO
            mask_tensor = self._ensure_4d_tensor(mask_tensor)

            # Add to subject
            subject_dict["mask"] = tio.LabelMap(tensor=mask_tensor, affine=affine)
            logger.debug(f"[SUBJECT] Propagated mask: {mask_tensor.shape}")

        # Load target (optional)
        if "target_path" in record and record["target_path"] is not None and self.target_io:
            target_data = self.target_io.load(record["target_path"], metadata=record)
            # Target is typically image domain
            target_data_value = (
                target_data.get("data")
                if target_data.get("data") is not None
                else target_data.get("image")
            )

            # [FIX] Apply NIfTI check for target IO as well
            if self._is_nifti_strategy(self.target_io, record.get("target_path", "")):
                target_tensor = self._ensure_4d_tensor(target_data_value)
            else:
                target_tensor = self._fastmri_to_torchio(
                    target_data_value, single_slice=single_slice
                )
            # Apply coil processing to the target.
            #
            # Magnitude modes (rss_image / magnitude / rss) MUST collapse the
            # target to a single channel for EVERY coil count — including a
            # single complex coil (shape[0] == 1) or a real-stacked R/I pair
            # (shape[0] == 2). The historical ``is_complex and shape[0] > 2``
            # gate skipped those, so a magnitude-only baseline (in_channels=1)
            # received a 2-channel (real/imag of 1 coil) target; the
            # simulator-derived input was then 2-channel and the first conv
            # raised "expected 1 channel, got 2" at iter 1 (smoke audit
            # 2026-06-03, RC-A: eval_c2/eval_c3/eval_c7/exp_c4). Multi-coil
            # magnitude arms (e.g. exp_c6, 4 coils) already passed this gate,
            # so their result is unchanged. ``_apply_coil_processing`` is a
            # no-op on an already-1-channel real target. Non-magnitude modes
            # keep the original multi-coil-only gate.
            _mag_modes = {"rss_image", "magnitude", "rss"}
            if self.coil_processing_mode in _mag_modes or (
                torch.is_complex(target_tensor) and target_tensor.shape[0] > 2
            ):
                target_tensor = self._apply_coil_processing(
                    target_tensor, sensitivity, is_kspace=False
                )
            subject_dict["target"] = tio.ScalarImage(
                tensor=target_tensor, affine=target_data.get("affine", affine)
            )
        else:
            # Self-supervised: target = clean input (already processed)
            subject_dict["target"] = tio.ScalarImage(tensor=primary_tensor, affine=affine)

        # Create and enforce consistency
        subject = tio.Subject(**subject_dict)
        subject = self._enforce_consistent_affines(subject)

        # [PHYSICS] Inject Physics Vector (P_GT)
        # Derived from filename heuristics defined in CLUSTER_ROADMAP.md
        # P = [TR, TE, TI, B0]
        if "primary_path" in record:
            path_str = str(record["primary_path"])
            file_name = Path(path_str).name
            physics_vec = get_physics_vector(file_name)
            subject["physics"] = physics_vec
            # Record the file this Subject came from. Every image here is built
            # with ``tensor=``, so ``tio.Image.path`` is None and any transform
            # that needs the source file -- sidecar readers, above all
            # ``LoadDWIMetadata`` looking for .bval/.bvec siblings -- has
            # nothing to work from. The string is already in hand two lines up
            # for the physics heuristic; not recording it was what made DWI
            # metadata unreachable on this route.
            subject["source_path"] = path_str

        # `target_physics` was emitted here (from the target's filename, or a
        # copy of the input's for the self-supervised case) and NEVER READ:
        # across the whole source tree the pair `input_physics`/`target_physics`
        # had 9 writes and 0 reads (audit D21). `physics` is different and stays
        # -- `diffusion.py` derives a mask from it.
        #
        # Producing it cost a second `get_physics_vector` lookup per paired
        # sample and, in the synthetic path, was satisfied with `torch.rand(1, 4)`
        # standing in for (TR, TE, TI, B0). That fabrication was the symptom
        # rather than the disease: nobody would ever have noticed random
        # acquisition parameters, because nothing looked at them.

        logger.debug(f"[SUBJECT] Built FastMRI subject: {list(subject_dict.keys())}")
        return subject


class PreprocessedSubjectBuilder(SubjectBuilder):
    """Builds Subjects from preprocessed directory structure.

    Handles:
    - Loading from npy, pt, nii.gz, h5 files
    - Ground truth (nifti_reconstructed)
    - K-space data (if available)
    - Sensitivity maps (coil sensitivity)
    - Metadata (statistics, task info)
    """

    def build(self, record: dict[str, Any]) -> tio.Subject:
        """Build Subject from preprocessed record.

        Args:
            record: Record with paths like gt_image_path, kspace_path, etc.

        Returns:
            TorchIO Subject
        """
        subject_dict = {}
        affine = np.eye(4)

        # Load ground truth image
        if "gt_image_path" in record:
            gt_tensor = self._load_tensor(record["gt_image_path"])
            gt_tensor = self._ensure_4d_tensor(gt_tensor)
            subject_dict["target"] = tio.ScalarImage(tensor=gt_tensor, affine=affine)

        # Load input image
        if "image_path" in record:
            img_tensor = self._load_tensor(record["image_path"])
            img_tensor = self._ensure_4d_tensor(img_tensor)
            subject_dict["input"] = tio.ScalarImage(tensor=img_tensor, affine=affine)

        # Load k-space data (optional)
        if "kspace_path" in record:
            kspace_tensor = self._load_tensor(record["kspace_path"])
            kspace_tensor = self._ensure_4d_tensor(kspace_tensor)
            subject_dict["kspace"] = tio.ScalarImage(tensor=kspace_tensor, affine=affine)

        # Load sensitivity maps (optional)
        if "sensitivity_path" in record:
            sens_tensor = self._load_tensor(record["sensitivity_path"])
            sens_tensor = self._ensure_4d_tensor(sens_tensor.abs())  # Magnitude
            subject_dict["sensitivity"] = tio.ScalarImage(tensor=sens_tensor, affine=affine)
            subject_dict["sensitivity_complex"] = sens_tensor

        # Load statistics metadata
        if "statistics_path" in record:
            try:
                with open(record["statistics_path"]) as f:
                    subject_dict["statistics"] = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load statistics: {e}")

        # [FIX] Load sampling mask (Critical for diffusion validation)
        mask_path = None
        if "mask_path" in record:
            mask_path = record["mask_path"]
        elif "sampling_mask_path" in record:
            mask_path = record["sampling_mask_path"]
        elif "acceleration_mask_path" in record:
            mask_path = record["acceleration_mask_path"]

        if mask_path:
            try:
                mask_tensor = self._load_tensor(mask_path)
                mask_tensor = self._ensure_4d_tensor(mask_tensor)
                # Ensure binary/integer for mask
                if mask_tensor.dtype in (torch.float16, torch.float32, torch.float64):
                    # keep as float if it contains values 0.0/1.0, but usually masks are labels
                    # However, diffusion strategy expects float mask in input batch sometimes
                    # tio.LabelMap converts to int usually?
                    # tio.LabelMap documentation says it handles discrete labels.
                    # But we can cast to float in strategy if needed.
                    pass
                subject_dict["mask"] = tio.LabelMap(tensor=mask_tensor, affine=affine)
                logger.debug(f"[SUBJECT] Loaded mask from {mask_path}: {mask_tensor.shape}")
            except Exception as e:
                logger.warning(f"Failed to load mask from {mask_path}: {e}")

        # Create and enforce consistency
        subject = tio.Subject(**subject_dict)
        subject = self._enforce_consistent_affines(subject)

        # [PHYSICS] Inject Physics Vector (P_GT) for Preprocessed Data
        # Try to derive from image path
        path_key = "image_path" if "image_path" in record else "kspace_path"
        if path_key in record:
            path_str = str(record[path_key])
            file_name = Path(path_str).name
            physics_vec = get_physics_vector(file_name)
            subject["physics"] = physics_vec
            # Same reason as FastMRISubjectBuilder: tensor-backed images carry
            # no ``path``, so a sidecar reader has nothing to resolve against.
            subject["source_path"] = path_str

        # [METADATA] Inject extra metadata from record (added by BIDS preprocessing)
        # e.g. contrast, subject_id, session_id, reconstruction
        for meta_key in [
            "contrast",
            "subject_id",
            "session_id",
            "reconstruction",
            "file_id",
            "dataset",
        ]:
            if meta_key in record:
                subject[meta_key] = record[meta_key]

        logger.debug(f"[SUBJECT] Built preprocessed subject: {list(subject_dict.keys())}")
        return subject

    def _load_tensor(self, path: Path) -> torch.Tensor:
        """Load tensor from various file formats.

        Delegates to ``spectramr.data.io_strategies.load_tensor_from_file`` —
        the canonical multi-format reader. Per CLAUDE.md pitfall #11,
        data-loading dispatch lives under ``src/data/`` in one place,
        not duplicated inline at each builder. See
        ``TODO/audit/16_data_layer.md`` F2.
        """
        from spectramr.data.io_strategies import load_tensor_from_file

        return load_tensor_from_file(path)


class SubjectBuilderFactory:
    """Factory for creating Subject builders.

    Provides registry-based dispatch by builder type.

    Supports:
    - 'fastmri': FastMRISubjectBuilder
    - 'preprocessed': PreprocessedSubjectBuilder
    - 'medical_volume': MedicalVolumeSubjectBuilder
    """

    BUILDERS = {
        "fastmri": FastMRISubjectBuilder,
        "preprocessed": PreprocessedSubjectBuilder,
    }

    @staticmethod
    def create(builder_type: str, **kwargs) -> SubjectBuilder:
        """Create a Subject builder by type.

        Args:
            builder_type: Type of builder ('fastmri', 'preprocessed')
            **kwargs: Arguments to pass to builder constructor

        Returns:
            Instantiated SubjectBuilder

        Raises:
            ValueError: If builder_type not recognized
        """
        if builder_type not in SubjectBuilderFactory.BUILDERS:
            raise ValueError(
                f"Unknown builder type: {builder_type}. "
                f"Available: {list(SubjectBuilderFactory.BUILDERS.keys())}"
            )

        builder_class = SubjectBuilderFactory.BUILDERS[builder_type]
        logger.debug(f"[SUBJECT] Created builder: {builder_type} ({builder_class.__name__})")
        return builder_class(**kwargs)

    @staticmethod
    def register(name: str, builder_class: type) -> None:
        """Register a custom Subject builder.

        Args:
            name: Name to register builder under
            builder_class: SubjectBuilder subclass
        """
        if not issubclass(builder_class, SubjectBuilder):
            raise TypeError(f"{builder_class} must be a SubjectBuilder subclass")

        SubjectBuilderFactory.BUILDERS[name] = builder_class
        logger.info(f"[SUBJECT] Registered builder: {name}")


__all__ = [
    "FastMRISubjectBuilder",
    "PreprocessedSubjectBuilder",
    "SubjectBuilder",
    "SubjectBuilderFactory",
]
