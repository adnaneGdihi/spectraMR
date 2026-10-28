"""Balanced Discriminator System for Optimal Generator/Discriminator Ratios

This module provides discriminators with optimal parameter counts to achieve
good generator/discriminator ratios (1.5-4.0) for stable GAN training.

DISCRIMINATOR PAIRING POLICY:
============================

The system automatically selects discriminator configurations based on
generator parameter count to maintain stable G/D ratios:

PARAMETER THRESHOLDS:
- Generator > 100M params: Discriminator width = 224 (for very large models)
- Generator > 30M params: Discriminator width = 192 (for large models)
- Generator > 10M params: Discriminator width = 128 (for medium-large models)
- Generator > 1M params: Discriminator width = 96 (for medium models)
- Generator ≤ 1M params: Discriminator width = 64 (for small models)

TARGET RATIOS:
- Optimal G/D ratio range: 1.5 - 4.0
- Target ratio: 2.5 (ideal balance for stable training)
- Ratios outside this range are avoided when possible

HEURISTICS:
- Uses RealESRGANDiscriminator as the backbone architecture
- Width selection is refined through parameter estimation to hit target ratios
- Falls back to PatchGAN if balanced pairing fails
- Maintains deterministic pairing for reproducible experiments

USAGE:
    discriminator = create_balanced_discriminator(generator, in_channels=1)
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.registry import register_model

# Imported hard. Under `except ImportError: IDiscriminator = nn.Module` the class
# below silently changed base depending on whether a first-party import resolved --
# present meant "not an nn.Module", absent meant "is one" -- so the same source
# defined two different types and nothing reported which one you got (NN3, NN18).
from spectramr.models.interfaces import IDiscriminator

# Import PatchGAN as backbone
try:
    from .patchgan_discriminator import PatchGANDiscriminator
except ImportError:
    PatchGANDiscriminator = None


# Real-ESRGAN's UNetDiscriminatorSN backbone -- a natural-image super-resolution critic
# operating on real-valued magnitude images. No internal transform.
@register_model(
    role="discriminator", name="realesrgan_discriminator", training_mode="gan", input_domain="image"
)
class RealESRGANDiscriminator(IDiscriminator, nn.Module):
    """Thin wrapper around Real-ESRGAN's UNetDiscriminatorSN to adapt output
    to a scalar per-sample score compatible with existing training code.

    - Uses spectral normalization inside the backbone.
    - Global average pooling over spatial dims.
    - Applies sigmoid for BCE compatibility (can be toggled off if needed).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_feat: int = 64,
        skip_connection: bool = True,
        use_sigmoid: bool = True,
        spectral_norm: bool = False,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            num_feat (int): Description.
            skip_connection (bool): Description.
            use_sigmoid (bool): Description.
            spectral_norm (bool): Description.
        """
        super().__init__()
        if PatchGANDiscriminator is None:
            raise ImportError("PatchGANDiscriminator not available in this environment")
        # Use PatchGANDiscriminator as backbone with new parameters
        self.backbone = PatchGANDiscriminator(
            in_channels=in_channels,
            ndf=num_feat,
            n_layers=3,
            spectral_norm=spectral_norm,
        )
        self.use_sigmoid = use_sigmoid

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            return_features (bool): Description.
        Returns:
            torch.Tensor: Description.
        """
        if return_features:
            y, features = self.backbone(x, return_features=True)
        else:
            y = self.backbone(x)
            features = None

        # y is [B,1,H,W]; pool to [B,1]
        if y.dim() == 4:
            y = F.adaptive_avg_pool2d(y, 1)
            y = y.view(y.size(0), -1)
        if self.use_sigmoid:
            y = torch.sigmoid(y)
        # Return shape [B] to match previous discriminators
        output = y.view(-1, 1).squeeze(1)

        if return_features:
            return output, features
        return output

    def discriminate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Discriminate real vs fake samples."""
        return self.forward(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract feature maps for feature matching."""
        features = {}
        if hasattr(self.backbone, "get_feature_maps"):
            # If backbone supports feature extraction
            features = self.backbone.get_feature_maps(x)
        else:
            # Fallback: extract from forward pass with features
            _, features = self.forward(x, return_features=True)
        return features

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "RealESRGANDiscriminator"

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _estimate_disc_params(
    discriminator_class: nn.Module,
    width: int,
    in_channels: int,
) -> int:
    """Instantiate a candidate discriminator and return its parameter count."""
    if discriminator_class is RealESRGANDiscriminator:
        d = discriminator_class(in_channels=in_channels, num_feat=width)
    else:
        raise ValueError(
            f"Unsupported discriminator class for estimation: {discriminator_class}",
        )
    return sum(p.numel() for p in d.parameters())


def _select_width_for_ratio(
    gen_params: int,
    discriminator_class: nn.Module,
    in_channels: int,
    target: float = 2.5,
    low: float = 1.5,
    high: float = 4.0,
) -> int:
    """Search widths and pick one that yields a G/D ratio within
    [low, high], else closest to target.
    """
    candidate_widths = [16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    best_width = candidate_widths[0]
    best_score = float("inf")
    best_in_range = None

    for w in candidate_widths:
        try:
            disc_params = _estimate_disc_params(discriminator_class, w, in_channels)
            if disc_params == 0:
                continue
            ratio = gen_params / disc_params
            # Prefer ratios inside range; score by distance to target
            score = abs(ratio - target)
            if low <= ratio <= high:
                if best_in_range is None or score < best_in_range[1]:
                    best_in_range = (w, score)
            if score < best_score:
                best_score = score
                best_width = w
        except Exception:
            # Skip widths that fail to instantiate
            continue

    if best_in_range is not None:
        return best_in_range[0]
    return best_width


def get_balanced_discriminator(generator_params_millions):
    """Return the preferred discriminator class and a baseline width guess.
    Note: This function now exclusively uses RealESRGANDiscriminator.
    """
    disc_cls = RealESRGANDiscriminator
    # Coarse baseline guess, refined later
    if generator_params_millions > 100:
        return disc_cls, 224
    if generator_params_millions > 30:
        return disc_cls, 192
    if generator_params_millions > 10:
        return disc_cls, 128
    if generator_params_millions > 1:
        return disc_cls, 96
    return disc_cls, 64


def create_balanced_discriminator(
    generator,
    in_channels=1,
    spectral_norm=False,
    **kwargs,
):
    """Create a balanced discriminator for a given generator.

    Args:
        generator: The generator model to balance against
        in_channels: Input channels for discriminator
        spectral_norm: Whether to use spectral normalization
        **kwargs: Additional arguments (ignored)

    Returns:
        Instantiated discriminator with optimal parameter ratio

    """
    # Count generator parameters
    gen_params = sum(p.numel() for p in generator.parameters())
    gen_params_millions = gen_params / 1e6

    # Get balanced discriminator
    discriminator_class, width_guess = get_balanced_discriminator(gen_params_millions)
    # Refine width by probing to hit the target ratio range
    width = _select_width_for_ratio(gen_params, discriminator_class, in_channels)
    if discriminator_class is RealESRGANDiscriminator:
        discriminator = discriminator_class(
            in_channels=in_channels,
            num_feat=width,
            spectral_norm=spectral_norm,
        )
    else:
        raise ValueError(
            "Unsupported discriminator class from "
            f"get_balanced_discriminator: {discriminator_class}",
        )

    # Count discriminator parameters
    disc_params = sum(p.numel() for p in discriminator.parameters())
    ratio = gen_params / disc_params

    logger = logging.getLogger(__name__)
    logger.info("Balanced discriminator created:")
    logger.info(f"  Generator params: {gen_params:,} ({gen_params_millions:.1f}M)")
    logger.info(
        f"  Discriminator params: {disc_params:,} ({disc_params / 1e6:.1f}M)",
    )
    logger.info(f"  Selected width: {width} | G/D ratio: {ratio:.2f}")

    return discriminator
