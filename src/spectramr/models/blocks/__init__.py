"""Reusable Blocks Package - Registry-based access to neural network blocks.

This package provides reusable building blocks for neural network architectures.
Access blocks via the registry pattern:

    from spectramr.models.blocks import create_block, register_block, list_registered_blocks

    # Create a block by name
    attention = create_block("self_attention", channels=64)
    residual = create_block("residual", in_channels=64, out_channels=128)
"""

# Legacy direct imports (for backward compatibility)
# Legacy direct imports (for backward compatibility)
from .base import BaseGANBlock
from .bloch_signal_encoder import (
    BlochSignalEncoder,
    ReferencePanel,
    ReferenceTissue,
)

# Registry API (preferred - use these for dynamic block creation)
from .block_registry import (
    BLOCK_REGISTRY,
    create_block,
    list_registered_blocks,
    register_block,
    unregister_block,
)
from .chronological_stem import ChronologicalKSpaceStem, ChronologicalKSpaceUnstem
from .instrument_keyed_attention import (
    AttentionKeys,
    InstrumentKeyedCrossAttention,
)

# Interfaces
from .interfaces import IBlockStrategy

# Topology / Linearization
from .morton_stem import MortonLocalityStem, QuadScanFusion, compute_morton_2d
from .normalization import AdaptiveInstanceNorm
from .radial_linearizer import MorphologicalRadialLinearizer
from .residual import UnifiedResidualBlock
from .topology_linearizer import ImageTopologyLinearizer
from .velocity_field_head import VelocityFieldHead

__all__ = [
    # Registry API (new pattern)
    "create_block",
    "register_block",
    "unregister_block",
    "list_registered_blocks",
    "BLOCK_REGISTRY",
    "IBlockStrategy",
    # Blocks
    "AdaptiveInstanceNorm",
    "BaseGANBlock",
    "BlochSignalEncoder",
    "ReferencePanel",
    "ReferenceTissue",
    "UnifiedResidualBlock",
    "VelocityFieldHead",
    "AttentionKeys",
    "InstrumentKeyedCrossAttention",
    # Topology / Linearization
    "MortonLocalityStem",
    "QuadScanFusion",
    "compute_morton_2d",
    "ChronologicalKSpaceStem",
    "ChronologicalKSpaceUnstem",
    "ImageTopologyLinearizer",
    "MorphologicalRadialLinearizer",
]
