"""TorchIO augmentation factory for MRI data.

Creates comprehensive augmentation pipelines optimized for medical imaging.
Includes geometric, intensity, and MRI-specific artifact simulations.
"""

import logging
import math

import torchio as tio

from spectramr.config.schemas.augmentation import AugmentationConfigSchema
from spectramr.data.transforms.intensity_augmentation import (
    RandomBrightness,
    RandomContrast,
    RandomMotionBlur,
    RandomRicianNoise,
)

logger = logging.getLogger(__name__)

# Dataset types whose tensors may be complex-valued.
# TorchIO spatial transforms (RandomAffine, RandomElasticDeformation) call
# tensor.min() / tensor.max() internally, which are not implemented for
# ComplexFloat in PyTorch.  Additionally, applying spatial warps directly
# in k-space is physically incorrect — one must warp in the image domain
# and re-compute k-space afterwards.
_COMPLEX_DATASET_TYPES = frozenset({"kspace", "kspace_3d"})

# Subject keys holding the ground truth (``mri`` aliases ``target``). Degradations
# apply to ``input`` only: TorchIO draws intensity parameters per image, so
# leaving the target in corrupts the clean label and decouples it from the input.
_TARGET_KEYS = ("target", "mri")


class TorchIOAugmentationFactory:
    """Builds TorchIO transform pipelines from configuration.

    Supports:
    - Geometric: flip, rotation, elastic deformation
    - Intensity: gamma, contrast, brightness, noise
    - MRI-Specific: bias field, motion, blur, Rician noise
    """

    @staticmethod
    def build(
        config: AugmentationConfigSchema,
        dataset_type: str = "image",
    ) -> tio.Compose | None:
        """Build augmentation pipeline from config.

        Args:
            config: AugmentationConfigSchema with enabled transforms
            dataset_type: The experiment's ``dataset_type`` (e.g. ``"image"``,
                ``"kspace"``).  When the data is complex-valued, spatial
                transforms that call ``tensor.min()`` are automatically
                skipped to prevent ``RuntimeError: min_all not implemented
                for ComplexFloat``.

        Returns:
            tio.Compose pipeline or None if disabled
        """
        if not config.enabled:
            return None

        is_complex = dataset_type in _COMPLEX_DATASET_TYPES

        transforms = []
        prob = config.probability
        degrade = {"p": prob, "exclude": _TARGET_KEYS}

        # =====================================================================
        # 1. Geometric Transforms
        # =====================================================================

        if config.enable_flip:
            # Default to (0, 1) for 2D images [H, W] in TorchIO (C, W, H, D)
            # TorchIO axes: 0=W, 1=H, 2=D.
            # If D=1 (2D), axes should be (0, 1).
            axes = tuple(config.flip_axes) if config.flip_axes else (0, 1)
            flip_prob = config.prob_flip
            transforms.append(tio.RandomFlip(axes=axes, p=flip_prob))

        if config.enable_rotation:
            if is_complex:
                logger.info(
                    "[AUGMENTATION] Skipping RandomAffine (rotation) for "
                    f"dataset_type='{dataset_type}' — incompatible with complex tensors"
                )
            else:
                rotate_prob = config.prob_rotate
                # Standard rotation in-plane (XY) for 2D.
                # TorchIO expects (min_a, max_a, min_b, max_b, min_c, max_c) for 3D.
                # We only rotate around the 3rd axis (D) to keep it in-plane for (W, H).
                rot_min, rot_max = config.rotation_range
                transforms.append(
                    tio.RandomAffine(
                        degrees=(0, 0, 0, 0, rot_min, rot_max),
                        scales=0,
                        translation=0,
                        p=rotate_prob,
                    )
                )

        if config.enable_elastic_deformation:
            if is_complex:
                logger.info(
                    "[AUGMENTATION] Skipping RandomElasticDeformation for "
                    f"dataset_type='{dataset_type}' — incompatible with complex tensors"
                )
            else:
                alpha = config.elastic_alpha
                # SAFETY CLAMP: Prevent TorchIO "folding may occur" warning.
                # TorchIO folding threshold = grid_spacing / 2 where:
                #   grid_spacing = image_bounds / (num_control_points - 2)
                # For 7 control points on a 181-pixel ULF NIfTI volume:
                #   spacing = 181 / (7-2) = 36.2,  threshold = 36.2 / 2 = 18.1
                # max_displacement MUST be ≤ threshold to avoid folding.
                # Clamp to 15.0 mm (safe for images ≥ 160 px with 7 CP).
                _MAX_SAFE_DISPLACEMENT = 15.0
                if alpha > _MAX_SAFE_DISPLACEMENT:
                    logger.info(
                        f"[AUGMENTATION] Clamping elastic_alpha {alpha} → "
                        f"{_MAX_SAFE_DISPLACEMENT} to prevent grid folding "
                        f"(num_control_points=7, folding threshold ≈ "
                        f"image_dim / {(7 - 2) * 2} px)"
                    )
                    alpha = _MAX_SAFE_DISPLACEMENT

                # Restricted to 2D by setting num_control_points for depth to 4.
                # locked_borders=0 avoids the "identity transform" error that occurs
                # when num_control_points <= 2*locked_borders on any axis.
                transforms.append(
                    tio.RandomElasticDeformation(
                        max_displacement=alpha,
                        num_control_points=(
                            7,
                            7,
                            4,
                        ),
                        locked_borders=0,
                        p=0.3,
                    )
                )

        # Before the intensity block, so a simulated input is degraded like a measured one.
        if config.enable_forward_operator:
            if is_complex:
                raise ValueError(
                    "data.augmentation.enable_forward_operator simulates a magnitude ULF "
                    f"image and cannot run on dataset_type={dataset_type!r}."
                )
            from spectramr.data.transforms.synthetic_ulf_simulator import SimulateULFFromHF

            transforms.append(
                SimulateULFFromHF(
                    source_image="target",
                    output_image="input",
                    p=config.prob_forward_operator,
                    **config.forward_operator_kwargs,
                )
            )
            logger.info(
                "[AUGMENTATION] forward-operator augmentation: input <- simulate_ulf_from_hf"
                "(target) with p=%.3f, kwargs=%s",
                config.prob_forward_operator,
                sorted(config.forward_operator_kwargs),
            )

        # =====================================================================
        # 2. Intensity Transforms
        # =====================================================================
        # Gamma exponentiates, contrast pivots on a real mean, and Rician noise
        # is a magnitude model: none is defined on a complex tensor, so the whole
        # photometric block is skipped at once for complex dataset types.  This
        # guard is the sole owner of that decision — the adapters in
        # ``intensity_augmentation`` do not re-check it (non-negotiable 17).
        intensity_flags = (
            config.enable_gamma,
            config.enable_brightness,
            config.enable_contrast,
            config.enable_noise,
            config.enable_bias_field,
            config.enable_rician_noise,
            config.enable_motion_blur,
            config.enable_blur,
        )
        skip_intensity = is_complex and any(intensity_flags)
        if skip_intensity:
            logger.info(
                "[AUGMENTATION] Skipping all intensity augmentations for "
                f"dataset_type='{dataset_type}' — gamma/contrast/Rician noise "
                "are magnitude-image models and are undefined on complex tensors"
            )

        if config.enable_gamma and not skip_intensity:
            # TorchIO draws beta ~ U(a, b) and applies gamma = exp(beta), so
            # the schema's `gamma_range` (stated in gamma) must be log-ed on the
            # way in.  Passing (0.8, 1.2) straight through would silently request
            # gamma in (2.23, 3.32) — a far harsher remap than the arm declared.
            gamma_min, gamma_max = config.gamma_range
            transforms.append(
                tio.RandomGamma(
                    log_gamma=(math.log(gamma_min), math.log(gamma_max)),
                    **degrade,
                )
            )

        if config.enable_brightness and not skip_intensity:
            transforms.append(
                RandomBrightness(
                    brightness_range=config.brightness_range,
                    **degrade,
                )
            )

        if config.enable_contrast and not skip_intensity:
            transforms.append(
                RandomContrast(
                    contrast_range=config.contrast_range,
                    **degrade,
                )
            )

        if config.enable_noise and not skip_intensity:
            # Gaussian noise; `enable_rician_noise` below is the magnitude model.
            transforms.append(tio.RandomNoise(std=config.noise_std, **degrade))

        # =====================================================================
        # 3. MRI-Specific Augmentations
        # =====================================================================

        if config.enable_bias_field and not skip_intensity:
            transforms.append(
                tio.RandomBiasField(
                    coefficients=config.bias_field_coefficients,
                    **degrade,
                )
            )

        if config.enable_rician_noise and not skip_intensity:
            transforms.append(
                RandomRicianNoise(
                    noise_level=config.rician_noise_level,
                    **degrade,
                )
            )

        if config.enable_motion_blur and not skip_intensity:
            transforms.append(
                RandomMotionBlur(
                    motion_intensity=config.motion_blur_intensity,
                    **degrade,
                )
            )

        if config.enable_blur and not skip_intensity:
            transforms.append(tio.RandomBlur(std=config.blur_std, **degrade))

        if config.enable_ghosting:
            # Restrict ghosting to in-plane axes (0, 1)
            axes = tuple(config.ghosting_axes) if config.ghosting_axes else (0, 1)
            transforms.append(
                tio.RandomGhosting(
                    num_ghosts=config.num_ghosts,
                    axes=axes,
                    intensity=config.ghosting_intensity,
                    **degrade,
                )
            )

        if config.enable_spike:
            transforms.append(
                tio.RandomSpike(
                    num_spikes=config.num_spikes,
                    intensity=config.spike_intensity,
                    **degrade,
                )
            )

        if config.enable_anisotropy:
            # Restrict anisotropy to in-plane axes or handle 2D properly
            # For 2D slices, anisotropy usually doesn't make sense unless it's within the slice.
            # We use only (0, 1) to avoid the axis 2 warning.
            transforms.append(
                tio.RandomAnisotropy(
                    axes=(0, 1),
                    downsampling=config.anisotropy_downsampling,
                    **degrade,
                )
            )

        if config.enable_b0_distortion:
            from spectramr.data.transforms.realistic_degradations import RandomB0Distortion

            transforms.append(
                RandomB0Distortion(
                    max_displacement=config.b0_max_displacement,
                    p=config.b0_prob,
                    exclude=_TARGET_KEYS,
                )
            )

        # =====================================================================
        # 4. Advanced Transforms (if implemented)
        # =====================================================================

        # DEFERRED, not unimplemented: the two k-space acceleration-augmentation
        # knobs (issue #1407) are declared by 112 / 15 arms under
        # experiments/inprogress/ and set true by none.  Wiring them needs
        # fft_ops.fft2c rather than raw torch.fft (non-negotiable 2) and makes
        # `input_prepared` diverge from the arm's DECLARED acceleration, which
        # the snapshot provenance must then report (non-negotiable 14).  That is
        # the physics lane, not this factory.
        #
        # Their names are deliberately NOT spelled here: the schema-key
        # consumption index (tools/audit/schema_key_consumption.py) is a raw
        # identifier regex over *.py with no comment stripping, so naming a dead
        # knob in a comment scores it "consumed" and silently launders it out of
        # the dead-knob ratchet.  Issue #1409.
        # Mixup/CutMix are batch-level operations, not per-sample

        if not transforms:
            return None

        return tio.Compose(transforms)


__all__ = ["TorchIOAugmentationFactory"]
