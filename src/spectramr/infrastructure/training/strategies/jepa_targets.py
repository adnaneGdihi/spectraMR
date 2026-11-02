"""Block-level embedding targets for the I-JEPA objective (Assran et al. 2023).

I-JEPA predicts the target encoder's embedding *of each target block* from the context
encoder's embedding of the visible context, with the predictor told where the block is.
These helpers turn an encoder's spatial feature map into those per-block embeddings:

* :func:`encoder_features` reads the representation a model produced on its last forward
  (the UNet bottleneck through ``LatentAccessMixin``, else the output feature map);
* :func:`pool_over_mask` averages a feature map over a pixel mask resampled to its grid;
* :func:`block_geometry` describes a block by its centre and extent, the predictor's
  positional query.

All three are tensor ops with no host sync, so they run on the training device.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.module_utils import unwrap_model

#: Width of the positional query :func:`block_geometry` returns: centre (y, x), extent (h, w).
GEOMETRY_DIM = 4


def encoder_features(model: Any, output: Any) -> torch.Tensor:
    """The ``(B, C, h, w)`` representation behind ``output``, the model's last forward.

    A model with latent access (``export_latent_state``) is read at its bottleneck, which
    is the encoder representation I-JEPA predicts; reading its decoded image instead would
    reduce the embedding to the output's channel count (one intensity for a magnitude
    UNet). A model without latent access is an encoder whose output is the feature map.

    Raises:
        ValueError: when neither a bottleneck nor a 4-D output feature map is available.
    """
    inner = unwrap_model(model)
    export = getattr(inner, "export_latent_state", None)
    if callable(export):
        bottleneck = export().get("bottleneck")
        if isinstance(bottleneck, torch.Tensor) and bottleneck.dim() == 4:
            return bottleneck
    if isinstance(output, dict):
        output = next(iter(output.values()))
    if isinstance(output, tuple):
        output = output[0]
    if not isinstance(output, torch.Tensor) or output.dim() != 4:
        shape = tuple(output.shape) if isinstance(output, torch.Tensor) else type(output)
        raise ValueError(
            f"JEPA needs a (B, C, h, w) encoder feature map; {type(inner).__name__} exposes "
            f"no 4-D bottleneck and its output is {shape}."
        )
    return output


def pool_over_mask(features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``features`` ``(B, C, h, w)`` over the ``(H, W)`` pixel ``mask``, as ``(B, C)``.

    The mask is area-resampled to the feature grid, so a block covering part of a coarse
    cell weights that cell by the covered fraction.
    """
    weight = F.adaptive_avg_pool2d(mask.to(features.dtype)[None, None], features.shape[-2:])
    total = weight.sum().clamp_min(torch.finfo(features.dtype).eps)
    return (features * weight).sum(dim=(-2, -1)) / total


def block_geometry(mask: torch.Tensor) -> torch.Tensor:
    """``(4,)`` normalised centre ``(y, x)`` and extent ``(h, w)`` of the pixel ``mask``."""
    m = mask.to(torch.float32)
    height, width = m.shape
    rows, cols = m.sum(dim=1), m.sum(dim=0)
    total = m.sum().clamp_min(1.0)
    ys = torch.arange(height, device=m.device, dtype=m.dtype) + 0.5
    xs = torch.arange(width, device=m.device, dtype=m.dtype) + 0.5
    return torch.stack(
        [
            (rows * ys).sum() / total / height,
            (cols * xs).sum() / total / width,
            (rows > 0).sum() / height,
            (cols > 0).sum() / width,
        ]
    )


def flatten_volume_rows(x: torch.Tensor) -> torch.Tensor:
    """``(B, C, D, H, W)`` volumes as ``(B*D, C, H, W)`` slices, sample after sample."""
    if x.dim() == 5:
        b, c, d, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * d, c, h, w)
    return x


__all__ = [
    "GEOMETRY_DIM",
    "block_geometry",
    "encoder_features",
    "flatten_volume_rows",
    "pool_over_mask",
]
