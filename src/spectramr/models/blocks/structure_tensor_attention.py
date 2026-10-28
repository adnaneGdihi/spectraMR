r"""Structure-Tensor Modulated Self-Attention Blocks.

Implements Pillar 8 of the ultra-low field synthesis paradigms. Uses the 3D
Structure Tensor S = ∇I ∇I^T extracted from the input 64mT volume to modulate
the self-attention matrix in Transformer blocks. This physically constrains
the network from averaging information across strong structural boundaries.

Mathematical Blueprint:
    Let S_i be the 3x3 structure tensor at voxel i. S_i characterizes the local
    anisotropy. We penalize attention between voxel i and j based on the
    directional distance:
        d(i, j) = (p_i - p_j)^T S_i (p_i - p_j)
    Modulated Attention:
        A = softmax(Q K^T / sqrt(d) - \gamma * d(i,j)^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def compute_3d_structure_tensor(volume: Tensor, sigma: float = 1.0, rho: float = 2.0) -> Tensor:
    """Compute the 3D Structure Tensor of a volume.

    Args:
        volume: Input volume float tensor of shape (B, C, D, H, W)
        sigma: Standard deviation for the Gaussian derivative
        rho: Standard deviation for the spatial integration (Gaussian smoothing)

    Returns:
        Structure tensor of shape (B, 6, D, H, W) where the 6 channels correspond
        to S_xx, S_yy, S_zz, S_xy, S_xz, S_yz.
    """
    # For efficiency and PyTorch friendliness, we implement the gradients
    # using simple finite differences (Sobel) and Gaussian smoothing.

    # 1. Compute gradients
    B, C, D, H, W = volume.shape
    device = volume.device
    dtype = volume.dtype

    # Finite difference kernels
    kx = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 1, 3) / 2.0
    ky = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 1, 3, 1) / 2.0
    kz = torch.tensor([-1.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 1, 3, 1, 1) / 2.0

    if C > 1:
        # We can sum gradients over channels or compute structure tensor per channel
        # Standard approach for color/multi-channel is summing the outer products.
        kx = kx.repeat(C, 1, 1, 1, 1)
        ky = ky.repeat(C, 1, 1, 1, 1)
        kz = kz.repeat(C, 1, 1, 1, 1)
        groups = C
    else:
        groups = 1

    grad_x = F.conv3d(volume, kx, padding=(0, 0, 1), groups=groups).mean(dim=1, keepdim=True)
    grad_y = F.conv3d(volume, ky, padding=(0, 1, 0), groups=groups).mean(dim=1, keepdim=True)
    if D > 1:
        grad_z = F.conv3d(volume, kz, padding=(1, 0, 0), groups=groups).mean(dim=1, keepdim=True)
    else:
        grad_z = torch.zeros_like(grad_x)

    # 2. Compute outer products
    S_xx = grad_x * grad_x
    S_yy = grad_y * grad_y
    S_zz = grad_z * grad_z
    S_xy = grad_x * grad_y
    S_xz = grad_x * grad_z
    S_yz = grad_y * grad_z

    S = torch.cat([S_xx, S_yy, S_zz, S_xy, S_xz, S_yz], dim=1)

    # 3. Apply spatial smoothing (rho)
    # A simple 3x3x3 average pool acts as a surrogate for Gaussian integration
    # given resolution limits, but we can use gaussian blur.
    # We use adaptive average pooling here to simplify dependencies.
    kernel_D = 3 if D > 1 else 1
    pad_D = 1 if D > 1 else 0
    S_smooth = F.avg_pool3d(S, kernel_size=(kernel_D, 3, 3), stride=1, padding=(pad_D, 1, 1))

    return S_smooth


class StructureTensorAttention(nn.Module):
    """Structure-Tensor Modulated Self-Attention block.

    Operates on a local 3D window (Swin-style) to maintain O(N) complexity.
    """

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int] = (8, 8, 8),
        num_heads: int = 8,
        qkv_bias: bool = True,
        gamma: float = 1.0,
        gamma_trainable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(gamma_trainable, bool):
            raise TypeError(f"gamma_trainable must be a bool, got {gamma_trainable!r}.")
        self.dim = dim
        self.window_size = window_size  # (Wd, Wh, Ww)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.gamma = gamma

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        # Structural modulation weight; a frozen one (a buffer) makes gamma=0 an exact ablation.
        if gamma_trainable:
            self.gamma_param = nn.Parameter(torch.tensor(float(gamma)))
        else:
            self.register_buffer("gamma_param", torch.tensor(float(gamma)))

        # Precompute relative position distance vectors (p_i - p_j) within the window
        Wd, Wh, Ww = window_size
        N = Wd * Wh * Ww

        # Grid coordinates [0, W-1]
        coords_d = torch.arange(Wd)
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(
            torch.meshgrid([coords_d, coords_h, coords_w], indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, N

        # Relative coordinates p_i - p_j -> shape (3, N, N)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        self.register_buffer("relative_coords", relative_coords.float())

    def forward(self, x: Tensor, structure_tensor: Tensor) -> Tensor:
        """Forward pass for structure tensor modulated attention.

        Args:
            x: Input features of shape (B * num_windows, N, C)
            structure_tensor: Structure tensor of shape (B * num_windows, 6, N)
                representing the 6 unique components of the 3x3 tensor per voxel.

        forward method for StructureTensorAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            structure_tensor (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, H, N, head_dim)

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B_, H, N, N)

        # Modulate with Structure Tensor
        # S_i has 6 components: Sxx, Syy, Szz, Sxy, Sxz, Syz
        Sxx = structure_tensor[:, 0, :]  # B_, N
        Syy = structure_tensor[:, 1, :]
        Szz = structure_tensor[:, 2, :]
        Sxy = structure_tensor[:, 3, :]
        Sxz = structure_tensor[:, 4, :]
        Syz = structure_tensor[:, 5, :]

        # Relative coords: (3, N, N)
        dx = self.relative_coords[2]  # Note: Z, Y, X order from meshgrid
        dy = self.relative_coords[1]
        dz = self.relative_coords[0]

        # d(i, j) = (p_i - p_j)^T S_i (p_i - p_j)
        # S_i is dependent only on voxel i, so we expand S over the j dimension
        # dist_matrix: (B_, N, N)
        Sxx_exp = Sxx.unsqueeze(2)
        Syy_exp = Syy.unsqueeze(2)
        Szz_exp = Szz.unsqueeze(2)
        Sxy_exp = Sxy.unsqueeze(2)
        Sxz_exp = Sxz.unsqueeze(2)
        Syz_exp = Syz.unsqueeze(2)

        dist_sq = (
            Sxx_exp * dx**2
            + Syy_exp * dy**2
            + Szz_exp * dz**2
            + 2 * Sxy_exp * dx * dy
            + 2 * Sxz_exp * dx * dz
            + 2 * Syz_exp * dy * dz
        )

        # Add modulation M_tensor to attention logits
        # Negative sign: distance increases -> attention decreases
        # Expand for heads: (B_, 1, N, N)
        M_tensor = -torch.abs(self.gamma_param) * dist_sq.unsqueeze(1)

        attn = attn + M_tensor
        attn = attn.softmax(dim=-1)

        # Output projection
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)

        return x


class StructureTensorTransformerBlock(nn.Module):
    """Transformer Block utilizing Structure Tensor Attention."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int, int] = (8, 8, 8),
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        gamma: float = 1.0,
        gamma_trainable: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StructureTensorAttention(
            dim,
            window_size=window_size,
            num_heads=num_heads,
            gamma=gamma,
            gamma_trainable=gamma_trainable,
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x: Tensor, S: Tensor) -> Tensor:
        """
        Args:
            x: (B_, N, C) Windowed features
            S: (B_, 6, N) Windowed structure tensor components

        forward method for StructureTensorTransformerBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            S (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = x + self.attn(self.norm1(x), S)
        x = x + self.mlp(self.norm2(x))
        return x
