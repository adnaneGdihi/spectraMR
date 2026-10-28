"""P4 (C_4) spatially-rotation-equivariant convolution family (MICCAI B-1.4).

A *genuine spatial* roto-translation-equivariant conv group for the cyclic group
``C_4`` (90-degree rotations), realised exactly with :func:`torch.rot90` — no kernel
interpolation. Distinct from the sibling :class:`~spectramr.models.blocks.equivariant.group_conv.GroupConv2d`,
which only cyclically rolls the *channel* group axis (an abstract internal symmetry used
by the equivariant flows) and does NOT rotate the spatial kernel — so it is not
equivariant to rotation of the image. These blocks are: rotate the input image by 90 deg
and the output rotates by 90 deg, exactly.

Feature layout for a regular-representation P4 field is ``[B, base*4, H, W]`` with the
group axis as the FAST (inner) axis: channel index ``= b*4 + r`` for base channel ``b``
and orientation ``r in {0,1,2,3}``. The group element ``g = rot90^s`` acts on a P4 field
by (a) spatially rotating each map by ``s*90 deg`` AND (b) cyclically shifting the
orientation axis by ``s``. The two pieces together are what make these layers equivariant.

Exactness: ``rot90`` is a pure index permutation (no interpolation), and on a SQUARE image
with symmetric zero-padding the padded region is itself invariant under 90-deg rotation, so
``layer(rot90(x)) == rot90(layer(x))`` to floating-point precision. Pointwise
nonlinearities (ReLU/SiLU) and a group axis-shared :class:`~torch.nn.GroupNorm` (affine
off) commute with the group action, so a whole U-Net stays equivariant. The final
:func:`p4_to_scalar` projects the regular field to a scalar field (image) by averaging the
orientation axis after a P4 conv — an equivariant regular->trivial map (rot in -> rot out).

Reference: T. S. Cohen and M. Welling, "Group equivariant convolutional networks," ICML
2016 (P4 / p4m).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

_GROUP_ORDER = 4  # C_4: 90-degree rotations


def rotate_p4_field(x: torch.Tensor, s: int) -> torch.Tensor:
    """Act with ``g = rot90^s`` on a P4 regular field ``[B, base*4, H, W]``.

    Spatially rotates every map by ``s*90 deg`` AND cyclically shifts the (inner)
    orientation axis by ``s``. This is the group action the P4 layers are equivariant to.
    """
    s = int(s) % _GROUP_ORDER
    b, c, h, w = x.shape
    if c % _GROUP_ORDER != 0:
        raise ValueError(f"P4 field channels must be divisible by 4; got {c}.")
    base = c // _GROUP_ORDER
    xr = torch.rot90(x, s, dims=(-2, -1))  # spatial rotation
    xr = xr.reshape(b, base, _GROUP_ORDER, h, w)
    xr = torch.roll(xr, shifts=s, dims=2)  # cyclic shift of the orientation fibre
    return xr.reshape(b, c, h, w)


class P4LiftingConv2d(nn.Module):
    """Lifting conv ``Z^2 -> C_4``: scalar/image input -> P4 regular field.

    For each of the 4 output orientations ``r`` the SAME base kernel is spatially rotated
    by ``r*90 deg`` (exact, via ``rot90``). Rotating the input image by 90 deg then both
    rotates each orientation map and cyclically shifts the orientation axis -> equivariance.
    """

    def __init__(self, in_channels: int, out_base: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_base = int(out_base)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        self.weight = nn.Parameter(
            torch.empty(out_base, in_channels, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_base))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = []
        for r in range(_GROUP_ORDER):
            w_r = torch.rot90(self.weight, r, dims=(-2, -1))
            outs.append(F.conv2d(x, w_r, self.bias, padding=self.padding))
        # [B, out_base, 4, H, W] -> [B, out_base*4, H, W] (base outer, orientation inner).
        stacked = torch.stack(outs, dim=2)
        b, ob, n, h, w = stacked.shape
        return stacked.reshape(b, ob * n, h, w)


class P4GroupConv2d(nn.Module):
    """Group conv ``C_4 -> C_4``: P4 regular field -> P4 regular field.

    For output orientation ``r`` the kernel is BOTH spatially rotated by ``r*90 deg`` AND
    has its input-orientation axis cyclically rolled by ``r``. Both transforms are required
    for ``conv(g.x) = g.conv(x)``.
    """

    def __init__(self, in_base: int, out_base: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_base = int(in_base)
        self.out_base = int(out_base)
        self.kernel_size = int(kernel_size)
        self.padding = self.kernel_size // 2
        # [out_base, in_base, 4, k, k]; dim=2 is the INPUT orientation axis.
        self.weight = nn.Parameter(
            torch.empty(out_base, in_base, _GROUP_ORDER, self.kernel_size, self.kernel_size)
        )
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        self.bias = nn.Parameter(torch.zeros(out_base))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = _GROUP_ORDER
        outs = []
        for r in range(n):
            # Roll the input-orientation axis by r, then spatially rotate by r*90deg.
            w_r = torch.roll(self.weight, shifts=r, dims=2)
            w_r = torch.rot90(w_r, r, dims=(-2, -1))
            # [out_base, in_base, 4, k, k] -> [out_base, in_base*4, k, k]
            w_r = w_r.reshape(self.out_base, self.in_base * n, self.kernel_size, self.kernel_size)
            outs.append(F.conv2d(x, w_r, self.bias, padding=self.padding))
        stacked = torch.stack(outs, dim=2)  # [B, out_base, 4, H, W]
        b, ob, _, h, w = stacked.shape
        return stacked.reshape(b, ob * n, h, w)


def p4_group_norm(num_base: int) -> nn.GroupNorm:
    """Orientation-shared GroupNorm (affine off) — commutes with the P4 group action.

    Groups the ``base*4`` channels into ``base`` groups of 4 consecutive (one base
    channel's four orientations), so the four orientations share statistics. ``affine=False``
    keeps it equivariant (a per-channel affine would need to permute with the orientations).
    """
    return nn.GroupNorm(num_groups=num_base, num_channels=num_base * _GROUP_ORDER, affine=False)


def p4_to_scalar(field: torch.Tensor, num_base: int) -> torch.Tensor:
    """Project a P4 regular field ``[B, base*4, H, W]`` to a scalar field by averaging the
    orientation axis -> ``[B, base, H, W]``.

    Averaging the regular representation over the group is the equivariant regular->trivial
    projection: rotating the input rotates this output (``mean_r rot90(g[r-1]) = rot90(mean g)``).
    """
    b, _c, h, w = field.shape
    return field.reshape(b, num_base, _GROUP_ORDER, h, w).mean(dim=2)
