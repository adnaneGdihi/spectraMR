import logging

import torch
import torchio as tio

from spectramr.data.transforms.layout import (
    PeAxis,
    canonical_to_tio,
    check_pe_axis,
    tio_to_canonical,
)
from spectramr.infrastructure.physics.degradations.geometry import AcquisitionMode
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.line_axis import line_axis_for

logger = logging.getLogger(__name__)

#: Subject key a masking transform writes the undersampled measurement to. The
#: reconstruction mixin resolves ``measured_kspace`` from it before falling back to
#: ``kspace``, which on every k-space route is the fully sampled reference.
MEASURED_KSPACE_KEY = "measured_kspace"

#: Subject key carrying ``N / sampled lines`` of the mask actually applied. It
#: differs from the declared ``acceleration``, which applies to the non-ACS lines
#: only (#2382), and travels with the batch because the main process's copy of
#: the transform never builds a mask when the loader runs workers.
REALISED_ACCELERATION_KEY = "realised_acceleration"


def loader_masks_lines(acceleration: float) -> bool:
    """Whether the loader undersamples with :class:`PhysicsInformedMasking`.

    At ``undersampling.base_acceleration`` 1 the loader passes k-space through
    unmasked and the training strategy does all masking, so the loader reads no
    ``mask_direction``. The audit asks the same question (#1082).
    """
    return acceleration != 1


class PhysicsInformedMasking(tio.transforms.Transform):
    """
    Physics-Informed Masking: Simulates Cartesian k-space undersampling.

    CRITICAL: Undersampling applies the SAME mask to ALL coils.
    - Input: (C=coils*2, W, H, D)
    - Output: 'input' with IDENTICAL shape, but with zeros applied by mask
    -         'measured_kspace' (the same undersampled tensor: the measurement y)
    -         'target' (copy of original full kspace)

    Per-coil behavior:
    - Generates 1D mask in frequency encoding direction
    - Expands to 2D (W, H)
    - Broadcasts to ALL coils
    - Result: all coils have same sampling pattern (Cartesian property)

    Sampling convention: ``acceleration`` R applies to the OUTER (non-ACS)
    lines only. The ACS band of ``center_fraction`` is always acquired and
    ``1/R`` of the remaining lines are drawn uniformly at random on top of it,
    so the realised factor ``N / sampled lines`` is below R: R 4 at N 256 with
    ``center_fraction`` 0.08 samples 79 lines, a realised R of 3.241. The
    fastMRI ``RandomMaskFunc`` convention, in which ACS plus outer lines total
    ``N/R``, is NOT what this transform does. The realised factor is written to
    every subject under :data:`REALISED_ACCELERATION_KEY` (#2382).

    Directions. The mask is a 1-D pattern of acquired LINES that varies along one
    axis (``line_axis``) and is constant along the other. Lines along the
    phase-encode axis are what an accelerated Cartesian scan skips; the other axis
    is the readout, always fully sampled. ``y`` is the first spatial axis of the
    ``(C, W, H, D)`` subject tensor (dim -2 of the training plane) and ``x`` the
    second (dim -1). ``mask_direction='phase'`` or ``'readout'`` is resolved
    against ``data.acquisition.phase_encode_axis`` (-2 or -1); a raw ``'y'``/``'x'``
    is taken as given. With ``mask_direction`` unset the lines run along the FIRST
    spatial axis whether or not that is the phase-encode axis, which on M4Raw is the
    readout: declare ``phase_encode_axis`` and ``mask_direction`` to be sure. The
    axis used and whether it is the phase-encode one are stamped in
    :meth:`applied_record`.

    The ACS band is NOT the one ``normalization.acs_band_rows`` defines: this mask
    truncates its width (``int(N * center_fraction)``) and places it on
    ``[N//2 - w//2, N//2 + w//2)``, so an odd width is one line short and the band
    sits one row off the DC row, while ``acs_band_rows`` rounds and centres on
    ``N//2``. Do not combine this mask with the ``acs`` scale domain (the scale is
    invariant to the mask only when both use ``acs_band_rows``); aligning them would
    change this mask for every arm that uses it, so it is kept as published (#2692).
    """

    def __init__(
        self,
        acceleration=4,
        center_fraction=0.08,
        validate_shapes=True,
        mask_direction: str | None = None,
        phase_encode_axis: int | None = None,
        **kwargs,
    ):
        """Cartesian line undersampling at ``acceleration`` with an ACS of ``center_fraction``.

        ``mask_direction`` names the axis the line pattern varies along
        (``undersampling.mask_direction``), resolved through
        ``line_axis_for`` and the dataset's ``phase_encode_axis``
        (``data.acquisition``). In the physics SSOT's spelling ``y`` is the
        first spatial axis of the ``(C, W, H, D)`` subject tensor (dim -2 of
        the training batch plane), ``x`` the second (dim -1). Unset keeps the
        first spatial axis.
        """
        super().__init__(**kwargs)
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.mask_direction = mask_direction
        self.phase_encode_axis = phase_encode_axis
        # TorchIO's reproducibility contract, and what the snapshot provenance
        # reads as the applied parameters. The names are the constructor's own
        # because history replay passes them back as keyword arguments.
        self.args_names = ["acceleration", "center_fraction", "mask_direction", "phase_encode_axis"]
        self.validate_shapes = validate_shapes
        self.line_axis = (
            "y" if mask_direction is None else line_axis_for(mask_direction, phase_encode_axis)
        )
        # Deterministic per-shape mask cache. The old ``torch.randperm`` was
        # unseeded → a genuinely different mask per item, unlike every other
        # generator in the physics SSOT (all seeded). We seed a scoped
        # ``torch.Generator`` per ``width`` so masks are reproducible (and thus
        # cacheable) without touching global RNG. Cache keyed by
        # ``(width, height)``; forkserver-pickle-safe (plain dict + a CPU
        # Generator built lazily / non-stored key).
        self._mask_cache: dict[tuple[int, int], torch.Tensor] = {}
        self._realised_cache: dict[tuple[int, int], float] = {}

    def applied_record(self) -> dict[str, str]:
        """What the mask does beyond the arguments, read by the snapshot provenance.

        ``undersampling.acceleration_type`` and ``undersampling_pattern`` are not
        read on this path, so the applied pattern is stated here instead (#2508).
        """
        if self.phase_encode_axis is None:
            lines_along = "undeclared (data.acquisition.phase_encode_axis is not set)"
        else:
            phase = line_axis_for("phase", self.phase_encode_axis)
            lines_along = "phase_encode" if self.line_axis == phase else "readout"
        return {
            "pattern": "uniform_random_outer_lines",
            "acceleration_applies_to": "non_acs_lines",
            "mask_seed": "1234 + n_lines",
            "line_axis": self.line_axis,
            "line_axis_plane_dim": "-2" if self.line_axis == "y" else "-1",
            "lines_along": lines_along,
        }

    def realised_acceleration(self, width: int, height: int) -> float:
        """``N / sampled lines`` of the mask this transform builds for the shape."""
        self._generate_mask(width, height)
        return self._realised_cache[(width, height)]

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # [PI-FIX] Prefer 'kspace_raw' as source of truth
        """apply_transform.

        Args:
            subject (tio.Subject): Description.
        Returns:
            tio.Subject: Description.
        """
        key = "kspace_raw" if "kspace_raw" in subject else "kspace"
        if key not in subject:
            logger.debug(f"No k-space found ({key} not in subject keys: {subject.keys()})")
            return subject

        # 1. Extract Tensor: (C, W, H, D)
        # C should be even (Re, Im pairs)
        raw_tensor = subject[key].data
        c, w, h, d = raw_tensor.shape
        device = raw_tensor.device

        logger.debug(
            f"PhysicsInformedMasking input: shape={raw_tensor.shape}, "
            f"device={device}, channels (coils*2)={c}"
        )

        # 2. Complex Conversion: (C, W, H, D) -> (C/2, W, H, D) Complex
        if c % 2 != 0:
            raise ValueError(
                f"K-Space channels must be even for Re/Im pairs. Got {c}. "
                f"This suggests coil count mismatch or wrong data format."
            )

        # Reshape to (C//2, 2, W, H, D) then view as complex
        # Assuming layout is [Re1, Im1, Re2, Im2...] or [Re1..N, Im1..N]?
        # FastMRISubjectWrapper outputs: kspace_t.permute(...) -> (C*2, ...)
        # It flattens (C, 2) -> C*2.
        # So we can reshape back.
        num_coils = c // 2
        logger.debug(f"  Reshaping to complex: {c} channels → {num_coils} coils + real/imag")

        try:
            reshaped = raw_tensor.contiguous().view(num_coils, 2, w, h, d)
            logger.debug(f"  After view: {reshaped.shape}")

            permuted = reshaped.permute(0, 2, 3, 4, 1).contiguous()
            logger.debug(f"  After permute: {permuted.shape} (coils, W, H, D, complex)")

            kspace_complex = torch.view_as_complex(permuted)
            logger.debug(f"  After complex view: {kspace_complex.shape}")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to convert to complex format. Input shape {raw_tensor.shape}, "
                f"expected (C={c}, W={w}, H={h}, D={d}) with even C. "
                f"Error: {e}"
            )

        # 3. Generate & Apply Mask
        # CRITICAL: Generate ONE mask, apply to ALL coils (Cartesian property)
        logger.debug(f"  Generating {self.acceleration}x undersampling mask (w={w}, h={h})")
        mask = self._generate_mask(w, h).to(device)
        # float32, not a Python float: collation would make that float64, which
        # the training loop's device move cannot place on MPS.
        subject[REALISED_ACCELERATION_KEY] = torch.tensor(
            self._realised_cache[(w, h)], dtype=torch.float32
        )

        # Reshape for broadcasting: (1, W, H, 1) - One mask per volume (Cartesian)
        mask_complex = mask.unsqueeze(0).unsqueeze(-1).expand(1, w, h, d)
        logger.debug(f"  Mask shape for broadcasting: {mask_complex.shape}")

        # Apply same mask to all coils
        undersampled_complex = kspace_complex * mask_complex
        logger.debug(f"  After masking: {undersampled_complex.shape}")

        # 4. Convert Back to IO Format: (C*2, W, H, D)
        try:
            real_view = torch.view_as_real(undersampled_complex)
            logger.debug(f"  After view_as_real: {real_view.shape}")

            permuted = real_view.permute(0, 4, 1, 2, 3)
            logger.debug(f"  After permute back: {permuted.shape}")

            undersampled_ri = permuted.flatten(0, 1)
            logger.debug(f"  After flatten to IO format: {undersampled_ri.shape}")
        except RuntimeError as e:
            raise ValueError(
                f"Failed to convert back from complex. Complex shape {undersampled_complex.shape}. "
                f"Error: {e}"
            )

        # 5. Store Results & Validate
        # 'input': Undersampled K-Space (Model Input)
        logger.debug(f"PhysicsInformedMasking output: input shape={undersampled_ri.shape}")

        # VALIDATION: Check shapes match (critical for avoiding downstream errors)
        if self.validate_shapes:
            if undersampled_ri.shape != raw_tensor.shape:
                raise ValueError(
                    f"Shape mismatch after undersampling! "
                    f"Input: {raw_tensor.shape}, Output: {undersampled_ri.shape}. "
                    f"Undersampling should preserve tensor shape."
                )

            # Check sparsity increased (mask was applied)
            input_zeros = (raw_tensor == 0).sum().float()
            output_zeros = (undersampled_ri == 0).sum().float()
            input_sparsity = input_zeros / raw_tensor.numel() * 100
            output_sparsity = output_zeros / undersampled_ri.numel() * 100

            logger.debug(f"  Sparsity: input={input_sparsity:.1f}%, output={output_sparsity:.1f}%")

            if output_sparsity <= input_sparsity:
                logger.warning(
                    f"Masking may not have been applied! "
                    f"Output sparsity {output_sparsity:.1f}% not greater than input {input_sparsity:.1f}%"
                )

        # Store as 'input' (Canonical)
        subject.add_image(
            tio.ScalarImage(tensor=undersampled_ri, affine=subject[key].affine),
            "input",
        )
        # The measurement y under its own name. `kspace` keeps the fully sampled
        # tensor, and resolving `measured_kspace` from it trained EI on the
        # reference instead of on y (#2324).
        subject.add_image(
            tio.ScalarImage(tensor=undersampled_ri.clone(), affine=subject[key].affine),
            MEASURED_KSPACE_KEY,
        )

        # Store original as 'target' (Canonical) if not present
        if "target" not in subject:
            target_data = subject[key].data

            # [FIX] Ensure target dtype matches input (real-stack if input was converted)
            if not torch.is_complex(undersampled_ri) and torch.is_complex(target_data):
                # Convert target to real stack matching input's Re/Im layout
                # (C, W, H, D) complex -> (C, W, H, D, 2) real -> (C, 2, W, H, D) real -> (2C, W, H, D) real
                target_real = torch.view_as_real(target_data)
                target_permuted = target_real.permute(0, 4, 1, 2, 3)
                target_data = target_permuted.flatten(0, 1)

            subject.add_image(
                tio.ScalarImage(tensor=target_data, affine=subject[key].affine),
                "target",
            )
        else:
            # [FIX] If target exists but is complex while input is real, convert it
            target_img = subject["target"]
            target_data = target_img.data
            if not torch.is_complex(undersampled_ri) and torch.is_complex(target_data):
                target_real = torch.view_as_real(target_data)
                target_permuted = target_real.permute(0, 4, 1, 2, 3)
                target_data = target_permuted.flatten(0, 1)
                target_img.set_data(target_data)

        # 'mask': The Sampling Mask
        # Expand mask to proper shape for viewing/loss: (1, W, H, D)
        mask_ri = mask.unsqueeze(0).unsqueeze(-1).expand(1, w, h, d)
        subject.add_image(tio.LabelMap(tensor=mask_ri, affine=subject[key].affine), "mask")

        logger.debug(
            f"  Added to subject: input {undersampled_ri.shape}, target, mask {mask_ri.shape}"
        )

        # [PI-FIX] Update 'mri_sub' (Zero-filled Recon) for consistency check?
        # Usually handled by the model or on-the-fly, but having a quick recon is good for debug.
        # Simple IFFT for QC
        # ifft_img = torch.fft.ifft2(undersampled_complex)
        # ifft_mag = ifft_img.abs().sum(dim=0, keepdim=True) # RSS-like
        # subject.add_image(tio.ScalarImage(tensor=ifft_mag, affine=subject[key].affine), 'mri_sub')

        return subject

    def _generate_mask(self, width, height) -> torch.Tensor:
        """Generate a Cartesian undersampling mask.

        CRITICAL DESIGN:
        - The 1-D line pattern varies along ``line_axis`` (``width`` for
          ``y``, ``height`` for ``x``) and is held constant across the other.
        - Expands the 1D line pattern to 2D and applies it to ALL coils.

        NOTE (review 2026-07-01, canonical-homes / pitfall #12): the physics SSOT
        ``infrastructure/physics/sampling.py::KSpaceMaskGenerator`` is the single
        home for k-space masks. This transform is NOT yet delegated to it because
        the sampling distributions differ (this one → uniform-random outside the
        ACS; SSOT → variable-density power-law). Both now resolve the line axis
        through ``line_axis_for`` (#2160). See docs/review_fixes_2026_07_01.rst.

        Args:
            width: First spatial dimension of the subject tensor.
            height: Second spatial dimension.

        Returns:
            Binary mask (W, H) with 1=acquired, 0=missing.
        """
        # Return a cached mask when this shape was already built — the mask is
        # deterministic (seeded below), so recomputing it per item is waste.
        cached = self._mask_cache.get((width, height))
        if cached is not None:
            return cached

        # The 1-D line pattern runs along the declared axis (`line_axis`).
        n_lines = width if self.line_axis == "y" else height
        mask = torch.zeros(n_lines)
        center = n_lines // 2
        cf_width = int(n_lines * self.center_fraction)

        # Safety check for center fraction
        start = max(0, center - cf_width // 2)
        end = min(n_lines, center + cf_width // 2)
        mask[start:end] = 1

        # Random sampling outside center — deterministic via a scoped generator
        # seeded per line count (reproducible across items/runs, no global RNG).
        outer_indices = torch.where(mask == 0)[0]
        if len(outer_indices) > 0 and self.acceleration > 0:
            num_samples = int(len(outer_indices) / self.acceleration)
            gen = torch.Generator(device="cpu").manual_seed(1234 + int(n_lines))
            perm = torch.randperm(len(outer_indices), generator=gen)[:num_samples]
            mask[outer_indices[perm]] = 1

        # Expand the line to (W, H), constant along the other axis. ``contiguous``
        # so the cached tensor is not an expand-view aliasing ``mask``.
        if self.line_axis == "y":
            mask_2d = mask.unsqueeze(1).expand(width, height).contiguous()
        else:
            mask_2d = mask.unsqueeze(0).expand(width, height).contiguous()
        self._mask_cache[(width, height)] = mask_2d

        # Counted off the mask actually built, not predicted from R and cf: the
        # ACS width and the outer count are both truncated, so a closed form drifts.
        sampled = int(mask.sum())
        realised = n_lines / max(sampled, 1)
        self._realised_cache[(width, height)] = realised
        logger.info(
            "[PHYSICS] PhysicsInformedMasking mask over %d lines: %d sampled, "
            "declared R=%s (outer lines), realised R=%.3f",
            n_lines,
            sampled,
            self.acceleration,
            realised,
        )

        return mask_2d


class DigitalTwinDegradation(tio.transforms.Transform):
    """Transversal Digital-Twin degradation as a data-pipeline transform.

    Mirrors :class:`PhysicsInformedMasking` (the accelerator) but applies the
    ``DigitalTwinSimulator`` corruption — composite motion + B0/B1 fields +
    noise — so **any** experiment can opt into synthetic degradation without
    a VF strategy. The twin operates in the image domain, so a k-space source
    is round-tripped ``ifft2c → corrupt → fft2c``.

    Produces the corrupted acquisition as ``input`` and preserves the clean
    ``target``, giving supervised (corrupted → clean) pairs.

    This is a K-SPACE transform. The former ``source_domain`` knob advertised an
    ``"image"`` value that could never work: the apply path resolved its source
    from the k-space key either way, so ``"image"`` meant "read k-space and
    pretend it is an image". No caller ever passed it either, since the transform
    builder hardcodes the k-space route. Corrupting an image-domain subject is a
    real capability but a separate design decision (which key is the clean
    source, and what happens on a dataset whose ``target`` is an independently
    acquired volume), so the knob is gone rather than accepted-but-broken.

    Args:
        simulator: A pre-built ``DigitalTwinSimulator`` (constructed from
            ``config.physics.digital_twin`` by the transform builder).
        degradation_only: skip VF marker embedding (default True — markers are
            a VF concept and have no place in a general degradation).
        pe_axis: which TorchIO axis, ``"W"`` or ``"H"``, is phase-encode; it lands on the
            twin's dim -2. Declared by the caller, never inferred.
        pe_axis_source: where ``pe_axis`` came from, stamped in :meth:`applied_record`.

    Without a volume mode on the twin's scanner profile each depth slice is corrupted as
    an independent 2-D slice; with ``multislice_2d`` the twin receives the canonical
    ``(1, coils, D, PE, RO)`` volume. ``volume_3d`` raises: the raw k-space served here is
    multislice (D is the slice index, not kz).
    """

    #: Where a k-space-serving dataset puts the raw acquisition, in lookup order.
    SOURCE_KEYS: tuple[str, ...] = ("kspace_raw", "kspace")

    def __init__(
        self,
        simulator,
        degradation_only: bool = True,
        *,
        pe_axis: PeAxis,
        pe_axis_source: str = "caller",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        check_pe_axis(pe_axis)
        geometry = getattr(simulator, "acquisition_geometry", None)
        if geometry is not None and geometry.mode is AcquisitionMode.VOLUME_3D:
            raise ValueError(
                "DigitalTwinDegradation cannot run a volume_3d profile: the raw k-space it "
                "reads is multislice (D is the slice index, not kz). Declare multislice_2d."
            )
        self.simulator = simulator
        self.degradation_only = degradation_only
        self.pe_axis = pe_axis
        self.pe_axis_source = pe_axis_source

    def applied_record(self) -> dict[str, object]:
        """The scanner profile the twin simulates and the declared PE axis, for provenance.

        The profile is stamped by name and the hash of its full ``to_meta()``: the snapshot
        record caps a dict at 16 keys, which would cut a profile's fields silently.
        """
        profile = getattr(self.simulator, "scanner_profile", None)
        return {
            "scanner_profile": None if profile is None else profile.name,
            "scanner_profile_sha256": None if profile is None else profile.meta_sha256(),
            "pe_axis": self.pe_axis,
            "pe_axis_source": self.pe_axis_source,
        }

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Corrupt the subject's acquisition with the digital twin."""
        key = next((k for k in self.SOURCE_KEYS if k in subject), None)
        if key is None:
            # Was a DEBUG log and a pass-through. The builder logs "[PHYSICS]
            # Injecting transversal DigitalTwinDegradation" at INFO one step
            # earlier, so silence here meant the run REPORTED a degradation it
            # never applied and trained on clean data (pitfall #16), invisible
            # at default verbosity.
            raise ValueError(
                "DigitalTwinDegradation found no acquisition to corrupt: the "
                f"subject carries {sorted(subject.keys())} but none of "
                f"{list(self.SOURCE_KEYS)}. It is a k-space transform. Serve a "
                "dataset_type that emits k-space (fastmri_kspace, ismrmrd, "
                "m4raw, bart), or turn physics.digital_twin.apply_as_transform "
                "off."
            )

        raw = subject[key].data  # (C, W, H, D)
        c, w, h, d = raw.shape
        if c % 2 != 0:
            raise ValueError(
                f"DigitalTwinDegradation expects even channels (Re/Im pairs); got {c}."
            )
        num_coils = c // 2

        # (C, W, H, D) -> (num_coils, W, H, D) complex
        kspace_complex = torch.view_as_complex(
            raw.contiguous().view(num_coils, 2, w, h, d).permute(0, 2, 3, 4, 1).contiguous()
        )
        # (1, num_coils, D, PE, RO); the twin works in image space.
        kc = tio_to_canonical(kspace_complex, self.pe_axis)
        geometry = getattr(self.simulator, "acquisition_geometry", None)
        if geometry is None or geometry.mode is AcquisitionMode.SLICE_2D:
            # Each depth slice is its own 2-D slice: (D, num_coils, PE, RO).
            slices = ifft2c(kc[0].permute(1, 0, 2, 3).contiguous())
            corrupted = self.simulator(slices, degradation_only=self.degradation_only)[0]
            corrupted_kc = fft2c(corrupted).permute(1, 0, 2, 3).unsqueeze(0)
        else:
            image = ifft2c(kc.contiguous())
            corrupted = self.simulator(image, degradation_only=self.degradation_only)[0]
            corrupted_kc = fft2c(corrupted)
        corrupted_k = canonical_to_tio(corrupted_kc, self.pe_axis).contiguous()

        # (num_coils, W, H, D) complex -> (C, W, H, D) real
        corrupted_ri = (
            torch.view_as_real(corrupted_k).permute(0, 4, 1, 2, 3).flatten(0, 1).contiguous()
        )
        if corrupted_ri.shape != raw.shape:
            raise ValueError(
                f"DigitalTwinDegradation must preserve shape: in {raw.shape}, "
                f"out {corrupted_ri.shape}."
            )

        subject.add_image(tio.ScalarImage(tensor=corrupted_ri, affine=subject[key].affine), "input")
        if "target" not in subject:
            subject.add_image(
                tio.ScalarImage(tensor=raw.clone(), affine=subject[key].affine), "target"
            )
        return subject
