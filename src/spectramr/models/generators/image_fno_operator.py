"""Plain image-domain Fourier Neural Operator on 1-D or 2-D real grids.

The comparator the CS-MNO PDE arms name ("a plain spectral operator"). Each
layer is the FNO update of Li et al. (2021),

    v_{l+1} = sigma( F^{-1}(R_l F(v_l)) + W_l v_l ),

with the spectral term from the same ``SpectralBranch1d`` / ``SpectralBranch2d``
blocks CS-MNO uses and the same 1x1 lift/project convolutions, so a CS-MNO arm
and this one differ by the SFC-Mamba branch alone. The registered ``fno`` takes
two-channel k-space and is 2-D only, so it cannot serve as that control.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from spectramr.models.blocks.spectral_branch import SpectralBranch1d, SpectralBranch2d
from spectramr.models.registry import register_model

__all__ = ["ImageFNOOperator"]

_SPECTRAL: dict[int, type[nn.Module]] = {1: SpectralBranch1d, 2: SpectralBranch2d}
_POINTWISE: dict[int, type[nn.Module]] = {1: nn.Conv1d, 2: nn.Conv2d}
_ACTIVATIONS: dict[str, type[nn.Module]] = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}


class _FNOLayer(nn.Module):
    """One FNO layer: spectral truncation plus a pointwise residual."""

    def __init__(self, dim: int, modes: int, spatial_rank: int, activation: str) -> None:
        super().__init__()
        self.spectral = _SPECTRAL[spatial_rank](dim, modes)
        self.W = _POINTWISE[spatial_rank](dim, dim, kernel_size=1)
        self.act = _ACTIVATIONS[activation]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.spectral(x) + self.W(x))


@register_model(
    name="image_fno_operator",
    training_mode="reconstruction",
    spatial_dims=(1, 2),
    input_domain=("pde_grid", "image"),
    output_domain=("pde_grid", "image"),
    accepts_complex=False,
    requires_paired_data=True,
    strict_kwargs=True,
)
class ImageFNOOperator(nn.Module):
    """Plain FNO on real ``[B, C, L]`` or ``[B, C, H, W]`` tensors.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        spatial_rank: 1 for sequences (Burgers), 2 for images (Darcy).
        n_layers: Number of FNO layers.
        modes: Retained Fourier modes per axis.
        dim: Channel width of the lifted representation.
        activation: One of ``gelu``, ``relu``, ``silu``. Unknown names raise.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        spatial_rank: Literal[1, 2] = 2,
        n_layers: int = 4,
        modes: int = 16,
        dim: int = 32,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        if spatial_rank not in _SPECTRAL:
            raise ValueError(
                f"ImageFNOOperator: spatial_rank must be one of {sorted(_SPECTRAL)}, "
                f"got {spatial_rank!r}"
            )
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"ImageFNOOperator: unknown activation {activation!r}. "
                f"Available: {sorted(_ACTIVATIONS)}"
            )
        if n_layers < 1 or modes < 1 or dim < 1:
            raise ValueError(
                f"ImageFNOOperator: n_layers, modes and dim must be >= 1, "
                f"got n_layers={n_layers}, modes={modes}, dim={dim}"
            )
        self.spatial_rank = int(spatial_rank)
        conv = _POINTWISE[self.spatial_rank]
        self.lift = conv(in_channels, dim, kernel_size=1)
        self.layers = nn.ModuleList(
            _FNOLayer(dim, modes, self.spatial_rank, activation) for _ in range(n_layers)
        )
        self.project = conv(dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, C_in, *spatial]`` to ``[B, C_out, *spatial]``.

        TorchIO yields ``[B, C, X, 1, 1]`` for 1-D PDE grids and ``[B, C, H, W, 1]``
        for 2-D ones; trailing singleton axes are squeezed to the declared rank, as
        ``CSMNOOperator`` does, and any other rank mismatch raises.
        """
        while x.ndim - 2 > self.spatial_rank and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim - 2 != self.spatial_rank:
            raise ValueError(
                f"ImageFNOOperator expects a rank-{self.spatial_rank} spatial input, "
                f"got shape {tuple(x.shape)}"
            )
        h = self.lift(x)
        for layer in self.layers:
            h = layer(h)
        return self.project(h)
