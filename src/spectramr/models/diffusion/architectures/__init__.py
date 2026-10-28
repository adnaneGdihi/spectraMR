"""Expose common U-Net variant blocks used across the project.

This module provides a minimal import surface for tests and legacy code that
expect specific names to be available from the ``unet_variants`` package. It is
designed to be resilient to import errors from optional dependencies.
"""

import importlib.util

__all__ = []

# --- Core Building Blocks ---

# AttentionBlock removed - consolidated into registry as "self_attention"

if importlib.util.find_spec(".common_blocks", __name__):
    try:
        from .common_blocks import OutConv  # noqa: F401

        __all__.append("OutConv")
    except ImportError:
        pass

if importlib.util.find_spec(".kan_blocks", __name__):
    try:
        from .kan_blocks import DoubleConvWithKAN, DownWithKAN, UpWithKAN  # noqa: F401

        __all__.extend(["DoubleConvWithKAN", "DownWithKAN", "UpWithKAN"])
    except ImportError:
        pass

# --- Full U-Net Architectures ---

if importlib.util.find_spec(".enhanced_kan_unet", __name__):
    try:
        from .enhanced_kan_unet import EnhancedKANUNet  # noqa: F401

        __all__.append("EnhancedKANUNet")
    except ImportError:
        pass

if importlib.util.find_spec(".kan_unet", __name__):
    try:
        from . import kan_unet as _kan_unet

        RefinedKANUNet = _kan_unet.RefinedKANUNet
        # Import the alias directly
        from .kan_unet import RefinedDiffusionUNet  # noqa: F401

        __all__.extend(["RefinedDiffusionUNet", "RefinedKANUNet"])
    except ImportError:
        pass

if importlib.util.find_spec(".kan_bottleneck_unet", __name__):
    try:
        from .kan_bottleneck_unet import KANBottleneckUNet  # noqa: F401

        __all__.append("KANBottleneckUNet")
    except ImportError:
        pass

if importlib.util.find_spec(".transformer_hybrid_unets", __name__):
    try:
        from .transformer_hybrid_unets import (  # noqa: F401
            LayerNormDoubleConv,
            SwinHybridUNet,
            ViTHybridUNet,
        )

        __all__.extend(["LayerNormDoubleConv", "SwinHybridUNet", "ViTHybridUNet"])
    except ImportError:
        pass

if importlib.util.find_spec(".score_based_unet", __name__):
    try:
        from .score_based_unet import ScoreBasedUNet  # noqa: F401

        __all__.append("ScoreBasedUNet")
    except ImportError:
        pass

# Sort the final list for cleanliness and to prevent duplicates.
__all__ = sorted(set(__all__))
