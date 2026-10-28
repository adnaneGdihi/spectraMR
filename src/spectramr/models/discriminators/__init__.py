"""Discriminator Implementations Package
====================================

This package contains the essential discriminator implementations.
Only PatchGAN and Balanced discriminators are included for simplicity.
"""

from .balanced_discriminators import (
    RealESRGANDiscriminator,
    create_balanced_discriminator,
)
from .contrast_conditioned_sense_bridge import (
    ContrastConditionedSenseBridgeDiscriminator,
)
from .patchgan_discriminator import PatchGANDiscriminator
from .sense_bridge_discriminator import SenseBridgeDiscriminator
from .stargan_v2_discriminator import (
    StarGANv2Discriminator,
)

__all__ = [
    "ContrastConditionedSenseBridgeDiscriminator",
    "PatchGANDiscriminator",
    "RealESRGANDiscriminator",
    "SenseBridgeDiscriminator",
    "StarGANv2Discriminator",
    "create_balanced_discriminator",
]
