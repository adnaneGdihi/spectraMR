"""Loss Dictionary Standardization - SSOT Key Registry.

This module provides unified loss key definitions across all training paradigms.
Ensures consistent naming and structure of loss dictionaries throughout the codebase.

Standard Loss Keys:
- *_total_loss: Final scalar loss for backward pass
- *_{component}_loss: Individual loss components
- metric_*: Computed metrics (not for backward)
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, ClassVar


class LossKeyRegistry:
    """Registry of standard loss keys for SSOT pattern.

    Ensures consistent naming across all loss computers and strategies.
    """

    # ============ GAN Training ============
    GAN_TOTAL_LOSS = "g_total_loss"
    GAN_COMPONENTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "g_loss_l1": "Generator L1 reconstruction",
            "g_loss_l2": "Generator L2 reconstruction",
            "g_loss_adversarial": "Generator adversarial loss",
            "g_loss_perceptual": "Generator perceptual loss",
            "g_loss_feature_matching": "Feature matching loss",
            "d_total_loss": "Discriminator total loss",
            "d_loss_real": "Discriminator real image loss",
            "d_loss_fake": "Discriminator fake image loss",
            "d_loss_r1": "R1 regularization penalty",
            "d_loss_gp": "Gradient penalty",
        }
    )

    # ============ Diffusion Training ============
    DIFFUSION_TOTAL_LOSS = "diffusion_total_loss"
    DIFFUSION_COMPONENTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "diffusion_loss_mse": "Diffusion MSE loss",
            "diffusion_loss_score": "Score matching loss",
            "diffusion_loss_recon": "Diffusion reconstruction loss",
            "diffusion_loss_adversarial": "Diffusion adversarial loss",
        }
    )

    # ============ VAE Training ============
    VAE_TOTAL_LOSS = "vae_total_loss"
    VAE_COMPONENTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "vae_loss_recon": "VAE reconstruction loss",
            "vae_loss_kl": "KL divergence loss",
            "vae_loss_perceptual": "VAE perceptual loss",
            "vae_loss_adversarial": "VAE adversarial loss",
        }
    )

    # ============ VQ-VAE Training ============
    VQVAE_TOTAL_LOSS = "vqvae_total_loss"
    VQVAE_COMPONENTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "vqvae_loss_recon": "VQ-VAE reconstruction loss",
            "vqvae_loss_codebook": "Codebook loss",
            "vqvae_loss_commitment": "Commitment loss",
            "vqvae_loss_perceptual": "VQ-VAE perceptual loss",
        }
    )

    # ============ Reconstruction Training ============
    RECONSTRUCTION_TOTAL_LOSS = "reconstruction_total_loss"
    RECONSTRUCTION_COMPONENTS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "recon_loss_l1": "L1 reconstruction",
            "recon_loss_l2": "L2 reconstruction",
            "recon_loss_perceptual": "Reconstruction perceptual",
            "recon_loss_ssim": "SSIM loss",
            "recon_loss_dists": "DISTS Perceptual Loss",
        }
    )

    # ============ Shared Metric Keys ============
    METRIC_KEYS: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "metric_psnr": "Peak Signal-to-Noise Ratio",
            "metric_ssim": "Structural Similarity Index",
            "metric_lpips": "Learned Perceptual Image Patch Similarity",
            "metric_fid": "Fréchet Inception Distance",
            "metric_nrmse": "Normalized Root Mean Square Error",
            "metric_mae": "Mean Absolute Error",
        }
    )

    @classmethod
    def get_all_loss_keys(cls) -> dict[str, str]:
        """Get all standardized loss keys.

        Returns:
            Dictionary mapping key names to descriptions
        """
        all_keys: dict[str, str] = {}
        all_keys.update(cls.GAN_COMPONENTS)
        all_keys.update(cls.DIFFUSION_COMPONENTS)
        all_keys.update(cls.VAE_COMPONENTS)
        all_keys.update(cls.VQVAE_COMPONENTS)
        all_keys.update(cls.RECONSTRUCTION_COMPONENTS)
        all_keys.update(cls.METRIC_KEYS)

        # Add total loss keys
        all_keys[cls.GAN_TOTAL_LOSS] = "GAN total loss"
        all_keys[cls.DIFFUSION_TOTAL_LOSS] = "Diffusion total loss"
        all_keys[cls.VAE_TOTAL_LOSS] = "VAE total loss"
        all_keys[cls.VQVAE_TOTAL_LOSS] = "VQ-VAE total loss"
        all_keys[cls.RECONSTRUCTION_TOTAL_LOSS] = "Reconstruction total loss"

        return all_keys

    @classmethod
    def get_training_mode_keys(cls, training_mode: str) -> dict[str, str]:
        """Get loss keys for specific training mode.

        Args:
            training_mode: One of "gan", "diffusion", "vae", "vqvae", "reconstruction"

        Returns:
            Dictionary of relevant loss keys
        """
        mode_map = {
            "gan": cls.GAN_COMPONENTS,
            "diffusion": cls.DIFFUSION_COMPONENTS,
            "vae": cls.VAE_COMPONENTS,
            "vqvae": cls.VQVAE_COMPONENTS,
            "reconstruction": cls.RECONSTRUCTION_COMPONENTS,
        }

        # A copy, so a caller that edits the result cannot rewrite the registry.
        return dict(mode_map.get(training_mode, {}))

    @classmethod
    def get_total_loss_key(cls, training_mode: str) -> str:
        """Get total loss key for training mode.

        Args:
            training_mode: Training mode string

        Returns:
            Total loss key (e.g., "g_total_loss" for GAN)
        """
        mode_map = {
            "gan": cls.GAN_TOTAL_LOSS,
            "diffusion": cls.DIFFUSION_TOTAL_LOSS,
            "vae": cls.VAE_TOTAL_LOSS,
            "vqvae": cls.VQVAE_TOTAL_LOSS,
            "reconstruction": cls.RECONSTRUCTION_TOTAL_LOSS,
        }

        return mode_map.get(training_mode, "total_loss")

    @classmethod
    def validate_loss_dict(cls, losses_dict: Mapping[str, Any], training_mode: str) -> bool:
        """Validate that loss dict uses standard keys.

        Args:
            losses_dict: Dictionary of losses
            training_mode: Training mode

        Returns:
            True if all keys are standardized

        Raises:
            ValueError: If non-standard keys found
        """
        all_valid_keys = set(cls.get_all_loss_keys().keys())
        all_valid_keys.add("total_loss")  # Generic total key

        for key in losses_dict:
            if not (key in all_valid_keys or key.startswith("metric_")):
                raise ValueError(
                    f"Non-standard loss key '{key}' in {training_mode} training. "
                    f"Valid keys: {all_valid_keys}"
                )

        return True


class LossKeyValidator:
    """Validates loss dictionaries for consistency."""

    def __init__(self, training_mode: str) -> None:
        """Initialize validator for training mode.

        Args:
            training_mode: Training paradigm
        """
        self.training_mode = training_mode
        self.expected_keys = LossKeyRegistry.get_training_mode_keys(training_mode)

    def validate(self, losses_dict: Mapping[str, Any], strict: bool = False) -> bool:
        """Validate loss dictionary.

        Args:
            losses_dict: Dictionary to validate
            strict: If True, require all expected keys. If False, just check for unknown keys.

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        all_valid_keys = set(LossKeyRegistry.get_all_loss_keys().keys())

        # Check for unknown keys
        for key in losses_dict:
            if not (key in all_valid_keys or key.startswith("metric_") or key == "total_loss"):
                raise ValueError(f"Unknown loss key '{key}' in {self.training_mode} training")

        # Check for required keys if strict mode
        if strict:
            total_key = LossKeyRegistry.get_total_loss_key(self.training_mode)
            if total_key not in losses_dict and "total_loss" not in losses_dict:
                raise ValueError(
                    f"Missing total loss key '{total_key}' in {self.training_mode} training"
                )

        return True


__all__ = [
    "LossKeyRegistry",
    "LossKeyValidator",
]
