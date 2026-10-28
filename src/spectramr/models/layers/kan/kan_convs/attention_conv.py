import logging
from functools import partial
from math import prod

import torch
from torch import nn
from torch.amp import autocast

from spectramr.models.layers.layer_norm_2d import LayerNorm2d

from .fast_kan_conv import FastKANConv1DLayer, FastKANConv2DLayer, FastKANConv3DLayer
from .kabn_conv import KABNConv1DLayer, KABNConv2DLayer, KABNConv3DLayer
from .kacn_conv import KACNConv1DLayer, KACNConv2DLayer, KACNConv3DLayer
from .kagn_bottleneck_conv import (
    BottleNeckKAGNConv1DLayer,
    BottleNeckKAGNConv2DLayer,
    BottleNeckKAGNConv3DLayer,
    MoEBottleNeckKAGNConv1DLayer,
    MoEBottleNeckKAGNConv2DLayer,
    MoEBottleNeckKAGNConv3DLayer,
)
from .kagn_conv import KAGNConv1DLayer, KAGNConv2DLayer, KAGNConv3DLayer
from .kagn_conv_v2 import KAGNConv1DLayerV2, KAGNConv2DLayerV2, KAGNConv3DLayerV2
from .kajn_conv import KAJNConv1DLayer, KAJNConv2DLayer, KAJNConv3DLayer
from .kaln_conv import KALNConv1DLayer, KALNConv2DLayer, KALNConv3DLayer
from .kan_conv import KANConv1DLayer, KANConv2DLayer, KANConv3DLayer
from .relukan_bottleneck_conv import (
    BottleNeckReLUKANConv1DLayer,
    BottleNeckReLUKANConv2DLayer,
    BottleNeckReLUKANConv3DLayer,
)
from .relukan_conv import ReLUKANConv1DLayer, ReLUKANConv2DLayer, ReLUKANConv3DLayer
from .wav_kan import WavKANConv1DLayer, WavKANConv2DLayer, WavKANConv3DLayer

logger = logging.getLogger(__name__)


def _init_coords_nd(*dims):
    """Initializes coordinates for N-dimensional tensors."""
    t = torch.arange(prod(dims), dtype=torch.float32)
    coords = []
    for i, dim_size in enumerate(dims):
        if i == 0:
            # For the fastest changing dimension
            coord = t % dim_size
        else:
            # For other dimensions
            coord = torch.div(t, prod(dims[:i]), rounding_mode="floor") % dim_size
        coords.append(coord.float())
    return coords


def _compute_axial_cis_nd(dim: int, *shape, theta: float = 100.0):
    """Computes axial rotary embeddings for N-dimensions."""
    coords = _init_coords_nd(*shape)
    n_dims = len(shape)

    # Each dimension gets an equal fraction of the feature dimension for
    # embeddings
    dim_per_axis = dim // n_dims

    # Ensure the dimension per axis is even for complex number representation
    if dim_per_axis % 2 != 0:
        dim_per_axis -= 1

    if dim_per_axis == 0:
        return None

    freqs = 1.0 / (theta ** (torch.arange(0, dim_per_axis, 2).float() / dim_per_axis))

    freqs_cis_list = []
    for i in range(n_dims):
        t = coords[i]
        freqs_i = torch.outer(t, freqs)
        freqs_cis_i = torch.polar(torch.ones_like(freqs_i), freqs_i)
        freqs_cis_list.append(freqs_cis_i)

    # Concatenate along the feature dimension
    freqs_cis = torch.cat(freqs_cis_list, dim=-1)

    # Pad if the total dimension is less than dim // 2
    final_dim = dim // 2
    if freqs_cis.shape[1] < final_dim:
        padding = torch.ones(
            freqs_cis.shape[0],
            final_dim - freqs_cis.shape[1],
            device=freqs_cis.device,
        )
        freqs_cis = torch.cat([freqs_cis, padding], dim=-1)

    return freqs_cis


def _init_2d_freqs_for_mixed_rope(
    dim: int,
    num_heads: int,
    theta: float = 10.0,
    rotate: bool = True,
):
    """_init_2d_freqs_for_mixed_rope.

    Args:
        dim (int): Description.
        num_heads (int): Description.
        theta (float): Description.
        rotate (bool): Description.
    Returns:
        Any: Description.
    """
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for _i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)],
            dim=-1,
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)],
            dim=-1,
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def _compute_mixed_cis_2d(
    freqs: torch.Tensor,
    t_x: torch.Tensor,
    t_y: torch.Tensor,
    num_heads: int,
):
    """_compute_mixed_cis_2d.

    Args:
        freqs (torch.Tensor): Description.
        t_x (torch.Tensor): Description.
        t_y (torch.Tensor): Description.
        num_heads (int): Description.
    Returns:
        Any: Description.
    """
    N = t_x.shape[0]
    # Force float32 math to avoid Half/Float dtype issues in trig and complex
    # ops
    with autocast(device_type="cuda", enabled=False):
        freqs_x = (
            (t_x.float().unsqueeze(-1) @ freqs[0].float().unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_y = (
            (t_y.float().unsqueeze(-1) @ freqs[1].float().unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_cis = torch.polar(
            torch.ones_like(freqs_x, dtype=torch.float32),
            (freqs_x + freqs_y).float(),
        )
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """reshape_for_broadcast.

    Args:
        freqs_cis (torch.Tensor): Description.
        x (torch.Tensor): Description.
    Returns:
        Any: Description.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    # Ensure float32 compute path to avoid Half/Float mismatches
    """apply_rotary_emb.

    Args:
        xq (torch.Tensor): Description.
        xk (torch.Tensor): Description.
        freqs_cis (torch.Tensor): Description.
    Returns:
        Any: Description.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2).contiguous())
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2).contiguous())
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(
        xq_.to(torch.complex64) * freqs_cis.to(torch.complex64),
    ).flatten(2)
    xk_out = torch.view_as_real(
        xk_.to(torch.complex64) * freqs_cis.to(torch.complex64),
    ).flatten(2)
    return xq_out.to(dtype=xq.dtype, device=xq.device), xk_out.to(
        dtype=xk.dtype,
        device=xk.device,
    )


class SelfKANtentionND(nn.Module):
    """SelfKANtentionND class."""

    def __init__(
        self,
        input_dim: int,
        conv_kan_layer: type[nn.Module],
        inner_projection: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | tuple[int, ...] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        dropout: float = 0.0,
        norm_layer: type[nn.Module] | None = None,
        affine: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        degree: int = 3,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        grid_range: list[float] | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            input_dim (int): Description.
            conv_kan_layer (type[nn.Module]): Description.
            inner_projection (int | None): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int | tuple[int, ...]): Description.
            dilation (int): Description.
            groups (int): Description.
            bias (bool): Description.
            dropout (float): Description.
            norm_layer (type[nn.Module] | None): Description.
            affine (bool): Description.
            grid_size (int): Description.
            spline_order (int): Description.
            degree (int): Description.
            g (int): Description.
            k (int): Description.
            train_ab (bool): Description.
            grid_range (list[float] | None): Description.
        """
        super().__init__()

        self.input_dim = input_dim
        self.ndim = None
        self.norm_layer = norm_layer

        if conv_kan_layer in [
            FastKANConv1DLayer,
            KANConv1DLayer,
            KALNConv1DLayer,
            KACNConv1DLayer,
            KAGNConv1DLayer,
            WavKANConv1DLayer,
            KAJNConv1DLayer,
            KABNConv1DLayer,
            BottleNeckKAGNConv1DLayer,
            MoEBottleNeckKAGNConv1DLayer,
            ReLUKANConv1DLayer,
            BottleNeckReLUKANConv1DLayer,
        ]:
            self.ndim = 1
            if self.norm_layer is None:
                self.norm_layer = nn.LayerNorm(input_dim)
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)
        elif conv_kan_layer in [
            FastKANConv2DLayer,
            KANConv2DLayer,
            KALNConv2DLayer,
            KACNConv2DLayer,
            KAGNConv2DLayer,
            WavKANConv2DLayer,
            KAJNConv2DLayer,
            KABNConv2DLayer,
            BottleNeckKAGNConv2DLayer,
            MoEBottleNeckKAGNConv2DLayer,
            ReLUKANConv2DLayer,
            BottleNeckReLUKANConv2DLayer,
        ]:
            self.ndim = 2
            if self.norm_layer is None:
                self.norm_layer = LayerNorm2d(input_dim)
            # Handle GroupNorm specially since it has different argument
            # order
            elif self.norm_layer == nn.GroupNorm:
                # Extract num_groups from kwargs if present, else default
                num_groups = kwargs.get(
                    "num_groups",
                    input_dim // groups if groups > 0 else 1,
                )
                # Filter out KAN-specific arguments that GroupNorm doesn't
                # accept
                group_norm_kwargs = {k: v for k, v in kwargs.items() if k in ["eps", "affine"]}
                self.norm_layer = self.norm_layer(
                    num_groups,
                    input_dim,
                    **group_norm_kwargs,
                )
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)
        elif conv_kan_layer in [
            FastKANConv3DLayer,
            KANConv3DLayer,
            KALNConv3DLayer,
            KACNConv3DLayer,
            KAGNConv3DLayer,
            WavKANConv3DLayer,
            KAJNConv3DLayer,
            KABNConv3DLayer,
            BottleNeckKAGNConv3DLayer,
            MoEBottleNeckKAGNConv3DLayer,
            ReLUKANConv3DLayer,
            BottleNeckReLUKANConv3DLayer,
        ]:
            self.ndim = 3

            if self.norm_layer is None:
                self.norm_layer = nn.LayerNorm(input_dim)
            else:
                self.norm_layer = self.norm_layer(input_dim, affine=affine)
        assert self.ndim is not None, "Unsupported conv kan layer"

        self.inner_proj = None
        self.outer_proj = None
        if inner_projection is not None:
            if self.ndim == 1:
                self.inner_proj = nn.Conv1d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv1d(inner_projection, input_dim, 1)
            if self.ndim == 2:
                self.inner_proj = nn.Conv2d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv2d(inner_projection, input_dim, 1)
            if self.ndim == 3:
                self.inner_proj = nn.Conv3d(input_dim, inner_projection, 1)
                self.outer_proj = nn.Conv3d(inner_projection, input_dim, 1)

        dims = input_dim if inner_projection is None else inner_projection

        self.dims = dims

        # Construct kwargs for the layer
        layer_kwargs = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
        }
        # Only add dropout if it's > 0 or if the layer supports it (most do)
        # But some might not. Assuming they do based on imports.
        # Actually, KANConvNDLayer does NOT take dropout in the file I read!
        # KANConvNDLayer in kan_conv.py:
        # def __init__(..., bias=True, grid_size=5, spline_order=3, base_activation=nn.SiLU):
        # No dropout!

        # KAGNConvNDLayer DOES take dropout.
        # FastKANConvNDLayer DOES take dropout.

        layer_name = conv_kan_layer.__name__

        if "KAGN" in layer_name or "FastKAN" in layer_name or "BottleNeck" in layer_name:
            layer_kwargs["dropout"] = dropout

        if "KAGN" in layer_name:
            layer_kwargs["degree"] = degree
        elif "ReLU" in layer_name:
            layer_kwargs["g"] = g
            layer_kwargs["k"] = k
            layer_kwargs["train_ab"] = train_ab
        elif "FastKAN" in layer_name:
            layer_kwargs["grid_size"] = grid_size
            if grid_range is not None:
                layer_kwargs["grid_range"] = grid_range
        else:
            # Default KAN args
            layer_kwargs["grid_size"] = grid_size
            layer_kwargs["spline_order"] = spline_order

        self.proj_k = conv_kan_layer(dims, dims, kernel_size, **layer_kwargs)
        self.proj_q = conv_kan_layer(dims, dims, kernel_size, **layer_kwargs)
        self.proj_v = conv_kan_layer(dims, dims, kernel_size, **layer_kwargs)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def attention(self, q, k, v):
        """attention.

        Args:
            q (Any): Description.
            k (Any): Description.
            v (Any): Description.
        Returns:
            Any: Description.
        """
        input_shape = v.size()
        m_batchsize = input_shape[0]
        total_pixels = prod(input_shape[2:])

        proj_query = q.view(m_batchsize, -1, total_pixels).permute(0, 2, 1)  # B X CX(N)
        proj_key = k.view(m_batchsize, -1, total_pixels)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(m_batchsize, -1, total_pixels)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(*input_shape)

        return out

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for SelfKANtentionND.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.inner_proj is not None:
            att = self.inner_proj(x)
        else:
            att = x

        q = self.proj_q(att)
        k = self.proj_k(att)
        v = self.proj_v(att)

        att = self.attention(q, k, v)

        if self.inner_proj is not None:
            att = self.outer_proj(att)

        return self.norm_layer(x + self.gamma * att)


class RoPESelfKANtentionND(SelfKANtentionND):
    """Multi-head Attention block with rotary position embeddings."""

    def __init__(self, *args, rope_theta=10.0, rope_mixed=True, **kwargs):
        """__init__.

        Args:
            rope_theta (Any): Description.
            rope_mixed (Any): Description.
        """
        super().__init__(*args, **kwargs)

        if self.dims % 2 != 0:
            raise ValueError(
                f"Embedding dimension for RoPE must be even, but got {self.dims}",
            )

        self.rope_mixed = rope_mixed
        self.rope_theta = rope_theta

        # Cache for computed embeddings
        self._freqs_cis_cache = {}
        self._t_coords_cache = {}

        if self.rope_mixed and self.ndim != 2:
            logger.warning(
                "Mixed RoPE is currently only implemented for 2D convolutions. "
                "Falling back to standard Axial RoPE for this layer."
            )
            self.rope_mixed = False

        if self.rope_mixed:
            self.compute_cis = partial(_compute_mixed_cis_2d, num_heads=1)
            freqs = _init_2d_freqs_for_mixed_rope(
                dim=self.dims,
                num_heads=1,
                theta=rope_theta,
                rotate=True,
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)
        else:
            # Use the new N-D axial computation function
            self.compute_cis = partial(
                _compute_axial_cis_nd,
                dim=self.dims,
                theta=self.rope_theta,
            )

    def attention(self, q, k, v):
        """attention.

        Args:
            q (Any): Description.
            k (Any): Description.
            v (Any): Description.
        Returns:
            Any: Description.
        """
        input_shape = v.size()
        m_batchsize = input_shape[0]
        spatial_dims = input_shape[2:]
        total_pixels = prod(spatial_dims)

        proj_query = q.view(m_batchsize, -1, total_pixels).permute(0, 2, 1)  # B x N x C
        proj_key = k.view(m_batchsize, -1, total_pixels).permute(0, 2, 1)  # B x N x C

        # Apply rotary position embedding
        if self.rope_mixed:
            shape_key = tuple(spatial_dims)
            if shape_key not in self._t_coords_cache:
                self._t_coords_cache[shape_key] = _init_coords_nd(*shape_key)
            coords = self._t_coords_cache[shape_key]
            t_x, t_y = coords[0].to(q.device), coords[1].to(q.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            shape_key = tuple(spatial_dims)
            if shape_key not in self._freqs_cis_cache:
                self._freqs_cis_cache[shape_key] = self.compute_cis(*shape_key)
            freqs_cis = self._freqs_cis_cache[shape_key].to(q.device)

        # Apply RoPE to all tokens, not skipping the first one.
        proj_query, proj_key = apply_rotary_emb(
            proj_query,
            proj_key,
            freqs_cis=freqs_cis,
        )
        proj_key = proj_key.permute(0, 2, 1)  # B x C x N
        #########

        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = v.view(m_batchsize, -1, total_pixels)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(*input_shape)

        return out


class SelfKAGNtention1D(SelfKANtentionND):
    """SelfKAGNtention1D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            KAGNConv1DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class SelfKAGNtention2D(SelfKANtentionND):
    """SelfKAGNtention2D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=None,  # Auto-calculate to preserve spatial dims
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        # Calculate padding to preserve spatial dimensions for attention
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        if padding is None:
            padding = (kernel_size - 1) // 2  # Same padding for attention

        super().__init__(
            input_dim,
            KAGNConv2DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class SelfKAGNtention3D(SelfKANtentionND):
    """SelfKAGNtention3D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            KAGNConv3DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckSelfKAGNtention2D(nn.Module):
    """Bottleneck version of SelfKAGNtention2D for efficient attention."""

    def __init__(
        self,
        input_dim,
        bottleneck_dim=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=None,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            bottleneck_dim (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__()

        # Set bottleneck dimension
        if bottleneck_dim is None:
            bottleneck_dim = max(1, input_dim // 4)

        # Calculate padding to preserve spatial dimensions
        if padding is None:
            padding = (kernel_size - 1) // 2

        # Bottleneck projection layers
        self.bottleneck_down = nn.Conv2d(input_dim, bottleneck_dim, 1)
        self.bottleneck_up = nn.Conv2d(bottleneck_dim, input_dim, 1)

        # Core attention mechanism
        self.attention = SelfKAGNtention2D(
            bottleneck_dim,
            inner_projection=None,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )

    def forward(self, x):
        # Apply bottleneck
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for BottleNeckSelfKAGNtention2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        bottleneck = self.bottleneck_down(x)

        # Apply attention in bottleneck space
        attended = self.attention(bottleneck)

        # Project back to original dimension
        out = self.bottleneck_up(attended)

        return out


class BottleNeckSelfKAGNtention1D(SelfKANtentionND):
    """BottleNeckSelfKAGNtention1D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=None,  # Auto-calculate to preserve spatial dims
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        # Calculate padding to preserve spatial dimensions for attention
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        if padding is None:
            padding = (kernel_size - 1) // 2  # Same padding for attention

        super().__init__(
            input_dim,
            BottleNeckKAGNConv1DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckSelfKAGNtention3D(SelfKANtentionND):
    """BottleNeckSelfKAGNtention3D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=None,  # Auto-calculate to preserve spatial dims
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        # Calculate padding to preserve spatial dimensions for attention
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        if padding is None:
            padding = (kernel_size - 1) // 2  # Same padding for attention

        super().__init__(
            input_dim,
            BottleNeckKAGNConv3DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class RoPEBottleNeckSelfKAGNtention1D(RoPESelfKANtentionND):
    """RoPEBottleNeckSelfKAGNtention1D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        rope_theta=10.0,
        rope_mixed=True,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            rope_theta (Any): Description.
            rope_mixed (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckKAGNConv1DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class RoPEBottleNeckSelfKAGNtention2D(RoPESelfKANtentionND):
    """RoPEBottleNeckSelfKAGNtention2D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        rope_theta=10.0,
        rope_mixed=True,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            rope_theta (Any): Description.
            rope_mixed (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckKAGNConv2DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class RoPEBottleNeckSelfKAGNtention3D(RoPESelfKANtentionND):
    """RoPEBottleNeckSelfKAGNtention3D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        degree=3,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        rope_theta=10.0,
        rope_mixed=True,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            degree (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            rope_theta (Any): Description.
            rope_mixed (Any): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckKAGNConv3DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            degree=degree,
            groups=groups,
            padding=padding,
            rope_theta=rope_theta,
            rope_mixed=rope_mixed,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class SelfReLUKANtention1D(SelfKANtentionND):
    """SelfReLUKANtention1D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            ReLUKANConv1DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class SelfReLUKANtention2D(SelfKANtentionND):
    """SelfReLUKANtention2D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            ReLUKANConv2DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class SelfReLUKANtention3D(SelfKANtentionND):
    """SelfReLUKANtention3D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            ReLUKANConv3DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckSelfReLUKANtention1D(SelfKANtentionND):
    """BottleNeckSelfReLUKANtention1D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckReLUKANConv1DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckSelfReLUKANtention2D(SelfKANtentionND):
    """BottleNeckSelfReLUKANtention2D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckReLUKANConv2DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckSelfReLUKANtention3D(SelfKANtentionND):
    """BottleNeckSelfReLUKANtention3D class."""

    def __init__(
        self,
        input_dim,
        inner_projection=None,
        kernel_size=3,
        g=5,
        k=3,
        train_ab=True,
        groups=1,
        padding=0,
        stride=1,
        dilation=1,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            inner_projection (Any): Description.
            kernel_size (Any): Description.
            g (Any): Description.
            k (Any): Description.
            train_ab (Any): Description.
            groups (Any): Description.
            padding (Any): Description.
            stride (Any): Description.
            dilation (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        super().__init__(
            input_dim,
            BottleNeckReLUKANConv3DLayer,
            inner_projection=inner_projection,
            kernel_size=kernel_size,
            g=g,
            k=k,
            train_ab=train_ab,
            groups=groups,
            padding=padding,
            stride=stride,
            dilation=dilation,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class KANFocalModulationND(nn.Module):
    """KANFocalModulationND class."""

    def __init__(
        self,
        dim: int,
        conv_kan_layer: type[nn.Module],
        focal_norm_layer: dict,
        focal_window: int,
        focal_level: int,
        focal_factor: int = 2,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False,
        full_kan: bool = True,
        grid_size: int = 5,
        spline_order: int = 3,
        degree: int = 3,
        g: int = 5,
        k: int = 3,
        train_ab: bool = True,
        grid_range: list[float] | None = None,
        dropout: float = 0.0,
        **kwargs,
    ):
        """__init__.

        Args:
            dim (int): Description.
            conv_kan_layer (type[nn.Module]): Description.
            focal_norm_layer (dict): Description.
            focal_window (int): Description.
            focal_level (int): Description.
            focal_factor (int): Description.
            use_postln_in_modulation (bool): Description.
            normalize_modulator (bool): Description.
            full_kan (bool): Description.
            grid_size (int): Description.
            spline_order (int): Description.
            degree (int): Description.
            g (int): Description.
            k (int): Description.
            train_ab (bool): Description.
            grid_range (list[float] | None): Description.
            dropout (float): Description.
        """
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.use_postln_in_modulation = use_postln_in_modulation
        self.normalize_modulator = normalize_modulator

        conv_kan_layer_focal = conv_kan_layer
        if conv_kan_layer in [
            FastKANConv1DLayer,
            KANConv1DLayer,
            KALNConv1DLayer,
            KACNConv1DLayer,
            WavKANConv1DLayer,
            KAJNConv1DLayer,
            KABNConv1DLayer,
            BottleNeckKAGNConv1DLayer,
            MoEBottleNeckKAGNConv1DLayer,
            ReLUKANConv1DLayer,
            BottleNeckReLUKANConv1DLayer,
        ]:
            self.global_pool = nn.AdaptiveAvgPool1d((1,))
            self.ndim = 1
            if conv_kan_layer in [BottleNeckKAGNConv1DLayer]:
                conv_kan_layer_focal = KAGNConv1DLayerV2
        elif conv_kan_layer in [
            FastKANConv2DLayer,
            KANConv2DLayer,
            KALNConv2DLayer,
            KACNConv2DLayer,
            WavKANConv2DLayer,
            KAJNConv2DLayer,
            KABNConv2DLayer,
            BottleNeckKAGNConv2DLayer,
            MoEBottleNeckKAGNConv2DLayer,
            ReLUKANConv2DLayer,
            BottleNeckReLUKANConv2DLayer,
        ]:
            self.ndim = 2
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            if conv_kan_layer in [BottleNeckKAGNConv2DLayer]:
                conv_kan_layer_focal = KAGNConv2DLayerV2
        elif conv_kan_layer in [
            FastKANConv3DLayer,
            KANConv3DLayer,
            KALNConv3DLayer,
            KACNConv3DLayer,
            WavKANConv3DLayer,
            KAJNConv3DLayer,
            KABNConv3DLayer,
            BottleNeckKAGNConv3DLayer,
            MoEBottleNeckKAGNConv3DLayer,
            ReLUKANConv3DLayer,
            BottleNeckReLUKANConv3DLayer,
        ]:
            self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.ndim = 3
            if conv_kan_layer in [BottleNeckKAGNConv3DLayer]:
                conv_kan_layer_focal = KAGNConv3DLayerV2

        # Construct kwargs for the layer
        layer_kwargs = {}

        layer_name = conv_kan_layer.__name__

        if "KAGN" in layer_name or "FastKAN" in layer_name or "BottleNeck" in layer_name:
            layer_kwargs["dropout"] = dropout

        if "KAGN" in layer_name:
            layer_kwargs["degree"] = degree
        elif "ReLU" in layer_name:
            layer_kwargs["g"] = g
            layer_kwargs["k"] = k
            layer_kwargs["train_ab"] = train_ab
        elif "FastKAN" in layer_name:
            layer_kwargs["grid_size"] = grid_size
            if grid_range is not None:
                layer_kwargs["grid_range"] = grid_range
        else:
            layer_kwargs["grid_size"] = grid_size
            layer_kwargs["spline_order"] = spline_order

        if full_kan:
            self.f = conv_kan_layer(
                dim,
                2 * dim + (self.focal_level + 1),
                1,
                padding=0,
                **layer_kwargs,
            )
            self.h = conv_kan_layer(dim, dim, 1, padding=0, **layer_kwargs)
        elif self.ndim == 1:
            self.f = nn.Conv1d(dim, 2 * dim + (self.focal_level + 1), 1)
            self.h = nn.Conv1d(dim, dim, 1)
        elif self.ndim == 2:
            self.f = nn.Conv2d(dim, 2 * dim + (self.focal_level + 1), 1)
            self.h = nn.Conv2d(dim, dim, 1)
        else:
            self.f = nn.Conv3d(dim, 2 * dim + (self.focal_level + 1), 1)
            self.h = nn.Conv3d(dim, dim, 1)

        self.proj = conv_kan_layer(dim, dim, 1, **layer_kwargs)
        self.focal_layers = nn.ModuleList()

        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                conv_kan_layer_focal(
                    dim,
                    dim,
                    kernel_size,
                    stride=1,
                    groups=dim,
                    padding=kernel_size // 2,
                    **layer_kwargs,
                ),
            )
            self.kernel_sizes.append(kernel_size)

        if use_postln_in_modulation:
            self.norm_layer = focal_norm_layer["layer"](
                dim,
                **focal_norm_layer["params"],
            )

    def forward(self, x):
        """Args:
        x: input features with shape of (B, C, H, W)

        """
        channels = x.shape[1]

        # pre linear projection
        x = self.f(x)
        q, ctx, self.gates = torch.split(
            x,
            [channels, channels, self.focal_level + 1],
            1,
        )

        # context aggregation
        ctx_all = 0
        for level in range(self.focal_level):
            ctx = self.focal_layers[level](ctx)
            ctx_all = ctx_all + ctx * self.gates[:, level : level + 1]
        ctx_global = self.global_pool(ctx_all)
        ctx_all = ctx_all + ctx_global * self.gates[:, self.focal_level :]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level + 1)

        # focal modulation
        modulator = self.h(ctx_all)
        x_out = q * modulator
        if self.use_postln_in_modulation:
            x_out = self.norm_layer(x_out)

        # post projection
        x_out = self.proj(x_out)
        return x_out


class BottleNeckKAGNFocalModulation1D(KANFocalModulationND):
    """BottleNeckKAGNFocalModulation1D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}

        super().__init__(
            input_dim,
            BottleNeckKAGNConv1DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckKAGNFocalModulation2D(KANFocalModulationND):
    """BottleNeckKAGNFocalModulation2D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}
        super().__init__(
            input_dim,
            BottleNeckKAGNConv2DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class BottleNeckKAGNFocalModulation3D(KANFocalModulationND):
    """BottleNeckKAGNFocalModulation3D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}

        super().__init__(
            input_dim,
            BottleNeckKAGNConv3DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class KAGNFocalModulation1D(KANFocalModulationND):
    """KAGNFocalModulation1D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}

        super().__init__(
            input_dim,
            KAGNConv1DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class KAGNFocalModulation2D(KANFocalModulationND):
    """KAGNFocalModulation2D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=LayerNorm2d,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}
        super().__init__(
            input_dim,
            KAGNConv2DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )


class KAGNFocalModulation3D(KANFocalModulationND):
    """KAGNFocalModulation3D class."""

    def __init__(
        self,
        input_dim,
        focal_window=3,
        focal_level=2,
        focal_factor=2,
        use_postln_in_modulation=True,
        normalize_modulator=True,
        full_kan: bool = True,
        degree=3,
        dropout: float = 0.0,
        norm_layer=nn.LayerNorm,
        **norm_kwargs,
    ):
        """__init__.

        Args:
            input_dim (Any): Description.
            focal_window (Any): Description.
            focal_level (Any): Description.
            focal_factor (Any): Description.
            use_postln_in_modulation (Any): Description.
            normalize_modulator (Any): Description.
            full_kan (bool): Description.
            degree (Any): Description.
            dropout (float): Description.
            norm_layer (Any): Description.
        """
        focal_norm_layer = {"layer": norm_layer, "params": norm_kwargs}

        super().__init__(
            input_dim,
            KAGNConv3DLayer,
            focal_norm_layer,
            focal_window,
            focal_level,
            focal_factor=focal_factor,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
            full_kan=full_kan,
            degree=degree,
            dropout=dropout,
            norm_layer=norm_layer,
            **norm_kwargs,
        )
