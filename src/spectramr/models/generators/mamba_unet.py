from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.data_consistency import NoiseSimulatingDataConsistency
from spectramr.infrastructure.physics.fft_ops import _to_complex, ifft2c
from spectramr.models.blocks.mamba_block import MambaBlock
from spectramr.models.registry import register_model

if TYPE_CHECKING:
    from spectramr.models.generators.hyper_mamba_bridge import SSMParameters

logger = logging.getLogger(__name__)


class TimeAwareSequential(nn.Sequential):
    """Sequential container that passes time embeddings through time-aware modules.

    .. deprecated::
        Import from ``spectramr.models.blocks.time_aware_sequential`` instead.
    """

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ssm_override: SSMParameters | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional SSM parameter override.

        Args:
            x: Input tensor [B, C, H, W].
            emb: Optional time embedding [B, D].
            ssm_override: Optional SSM parameters from HyperMambaBridge.

        Returns:
            Output tensor [B, C, H, W].
        """
        for module in self:
            if isinstance(module, MambaLayer2D) or isinstance(module, TimeAwareSequential):
                x = module(x, emb, ssm_override=ssm_override)
            else:
                x = module(x)
        return x


@register_model(name="mamba_reconstruction", training_mode="reconstruction")
class MambaReconstruction(nn.Module):
    """
    Physics-Informed Unrolled Mamba Network for MRI Reconstruction.

    Architecture:
        Zero-Filled Recon -> [Mamba Refinement -> Data Consistency] x N -> Output
    """

    def __init__(
        self,
        in_channels=2,
        out_channels=2,
        features=[32, 64, 128, 256],
        mamba_config=None,
        num_iterations=5,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            features (Any): Description.
            mamba_config (Any): Description.
            num_iterations (Any): Description.
        """
        super().__init__()
        self.num_iterations = num_iterations

        # Backbone: Operates in Image Space
        # Mamba is excellent at global context (anatomy), not spectral features.
        self.mamba_backbone = MambaUNet(
            in_channels=in_channels,
            out_channels=out_channels,
            features=features,
            mamba_config=mamba_config,
            **kwargs,
        )

        # Physics: Operates in K-Space
        self.dc_layer = NoiseSimulatingDataConsistency()

    def forward(self, masked_kspace, mask):
        """
        Args:
            masked_kspace: (B, C, H, W, 2) or (B, 2, H, W) Input undersampled k-space
            mask: (B, C, H, W, 1) or matching masked_kspace Sampling mask
        """

        # 1. Initial Estimate (Zero-Filled Reconstruction)
        # Handle format: if (B, 2, H, W) treat as real/imag channels
        if not torch.is_complex(masked_kspace) and masked_kspace.shape[1] == 2:
            kspace_complex = torch.view_as_complex(masked_kspace.permute(0, 2, 3, 1).contiguous())
        elif not torch.is_complex(masked_kspace):
            kspace_complex = _to_complex(masked_kspace)
        else:
            kspace_complex = masked_kspace

        x_complex = ifft2c(kspace_complex)

        # Convert to model input format (B, 2, H, W)
        x = torch.view_as_real(x_complex).permute(0, 3, 1, 2).contiguous()

        # 2. Unrolled Optimization Loop
        for i in range(self.num_iterations):
            # A. Image Space Refinement (Regularization)
            # The network predicts the residual correction
            x_residual = self.mamba_backbone(x)
            x = x + x_residual

            # B. K-Space Data Consistency (Physics Constraint)
            # "Reset" the known frequencies to their measured values
            x = self.dc_layer(x, masked_kspace, mask)

        return x


class MambaLayer2D(nn.Module):
    """Wraps MambaBlock for 2D inputs [B, C, H, W].

    Flattens spatial dimensions to sequence [B, H*W, C] for Mamba processing.
    Supports optional SSM parameter override from HyperMambaBridge for
    scan-specific motion conditioning.

    Args:
        channels: Number of input/output channels.
        time_embedding_dim: Time embedding dimension for diffusion conditioning.
        **mamba_kwargs: Additional kwargs for MambaBlock.
    """

    def __init__(
        self,
        channels: int,
        time_embedding_dim: int | None = None,
        linearization_mode: str = "raster",
        **mamba_kwargs,
    ) -> None:
        """Initialize MambaLayer2D.

        Args:
            channels: Number of channels (= d_model for Mamba).
            time_embedding_dim: Optional time embedding dimension.
            linearization_mode: Sequence ordering strategy.
                ``"raster"`` — standard row-major flatten (default).
                ``"morton"`` / ``"morton_2d"`` — Z-order curve.
                ``"hilbert_2d"`` — Hilbert fractal curve.
                ``"snake_2d"`` — Boustrophedon scan.
                ``"zigzag_2d"`` — Diagonal zig-zag scan.
                ``"radial_inside_out"`` — Centrifugal concentric sweep.
                ``"radial_outside_in"`` — Centripetal concentric sweep.
        """
        super().__init__()
        self.channels = channels
        self.linearization_mode = linearization_mode
        self.norm = nn.LayerNorm(channels)

        # Strip linearization_mode from mamba_kwargs to avoid passing
        # it to MambaBlock (which doesn't accept it)
        mamba_kwargs.pop("linearization_mode", None)
        self.mamba = MambaBlock(d_model=channels, **mamba_kwargs)

        # Time embedding projection
        self.time_proj = None
        if time_embedding_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, channels))

        # SSM modulation projection: maps external B/C → feature-space modulation
        # Lazily initialized on first use since d_state is not known here
        self._ssm_mod_proj: nn.Linear | None = None

        # Topology indices: lazily initialized on first forward call
        # (we don't know H, W at init time inside a UNet)
        self._topo_fwd: torch.Tensor | None = None
        self._topo_inv: torch.Tensor | None = None
        self._cached_hw: tuple[int, int] | None = None

    def _get_topology_indices(
        self, H: int, W: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lazily compute and cache topology permutation for given (H, W).

        Dispatches to the appropriate generator based on ``linearization_mode``.
        Recomputes if spatial dimensions change (UNet encoder/decoder).

        Args:
            H: Height of the current feature map.
            W: Width of the current feature map.
            device: Target device for index tensors.

        Returns:
            Tuple of (forward, inverse) permutation index tensors.
        """
        if self._cached_hw != (H, W) or self._topo_fwd is None:
            # Map short aliases to canonical mode names
            mode = self.linearization_mode
            if mode == "morton":
                mode = "morton_2d"

            from spectramr.models.blocks.topology_linearizer import _MODE_GENERATORS

            if mode not in _MODE_GENERATORS:
                raise ValueError(
                    f"Unknown linearization mode '{mode}'. "
                    f"Available: raster, {list(_MODE_GENERATORS.keys())}"
                )

            indices = _MODE_GENERATORS[mode]((H, W))
            fwd = torch.tensor(indices, dtype=torch.long, device=device)
            inv = torch.argsort(fwd)
            self._topo_fwd = fwd
            self._topo_inv = inv
            self._cached_hw = (H, W)
        return self._topo_fwd, self._topo_inv

    def _get_ssm_mod_proj(self, ssm_dim: int) -> nn.Linear:
        """Lazily create SSM modulation projection.

        Args:
            ssm_dim: Dimension of the SSM parameter vector (d_model * d_state).

        Returns:
            Linear projection mapping SSM params to channel modulation.
        """
        if self._ssm_mod_proj is None or self._ssm_mod_proj.in_features != ssm_dim * 2:
            self._ssm_mod_proj = nn.Linear(ssm_dim * 2, self.channels, bias=False).to(
                next(self.parameters()).device
            )
            nn.init.zeros_(self._ssm_mod_proj.weight)
            logger.debug(
                "[MambaLayer2D] Initialized SSM modulation proj: %d → %d",
                ssm_dim * 2,
                self.channels,
            )
        return self._ssm_mod_proj

    def forward(
        self,
        x: torch.Tensor,
        emb: torch.Tensor | None = None,
        ssm_override: SSMParameters | None = None,
    ) -> torch.Tensor:
        """Forward pass with optional SSM modulation.

        Args:
            x: Input tensor [B, C, H, W].
            emb: Optional time embedding [B, D].
            ssm_override: Optional SSM parameters from HyperMambaBridge.
                When provided, applies additive feature modulation derived
                from the bridge's B_proj, C_proj, and Δ_proj outputs.

        Returns:
            Output tensor [B, C, H, W].

        forward method for MambaLayer2D.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ssm_override (SSMParameters | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time embedding injection (add to input features)
        if self.time_proj is not None and emb is not None:
            t = self.time_proj(emb)
            # Broadcast: (B, C) -> (B, C, 1, 1)
            x = x + t[..., None, None]

        # Flatten 2D → 1D sequence using configured linearization mode
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        if self.linearization_mode != "raster":
            # Apply space-filling curve permutation for locality-preserving linearization
            topo_fwd, topo_inv = self._get_topology_indices(H, W, x.device)
            x_flat = x_flat[:, topo_fwd, :]

        x_flat = self.norm(x_flat)

        # Ensure contiguity after norm (critical for cuDNN GRU operations)
        x_flat = x_flat.contiguous()

        # Mamba Forward
        out_flat = self.mamba(x_flat)

        # SSM modulation: apply external B/C/Δ as additive gated modulation
        if ssm_override is not None:
            # Concatenate B and C projections → modulation input
            bc_cat = torch.cat(
                [ssm_override.B_proj, ssm_override.C_proj], dim=-1
            )  # [B, 2 * d_model * d_state]

            # Project to feature space
            mod_proj = self._get_ssm_mod_proj(ssm_override.B_proj.shape[-1])
            modulation = mod_proj(bc_cat)  # [B, C]

            # Gate with Δ (discretization step — controls modulation strength)
            # Δ_proj is [B, d_model]; we need to broadcast or reduce to [B, C]
            delta = ssm_override.Delta_proj
            if delta.shape[-1] != C:
                # Adaptive pooling if d_model != channels
                delta = torch.nn.functional.adaptive_avg_pool1d(delta.unsqueeze(1), C).squeeze(1)
            gate = torch.sigmoid(delta)  # [B, C]

            # Apply: modulation is broadcast over sequence dim
            out_flat = out_flat + (modulation * gate).unsqueeze(1)

        # Reverse linearization: 1D → 2D
        if self.linearization_mode != "raster":
            out_flat = out_flat[:, topo_inv, :]

        out = out_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out


@register_model(name="mamba_unet", training_mode="reconstruction", spatial_dims=(2,))
class MambaUNet(nn.Module):
    """
    U-Net with Mamba (State Space Model) blocks in the encoder and decoder.
    Designed for efficient long-range dependency modeling in MRI.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: list[int] = [32, 64, 128, 256],
        mamba_config: dict = None,
        spatial_dims: int = 2,  # Unused, compat arg
        use_complex_conv: bool = False,
        time_embedding_dim: int | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (list[int]): Description.
            mamba_config (dict): Description.
            spatial_dims (int): Description.
            use_complex_conv (bool): Description.
            time_embedding_dim (int | None): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        # Sanitize mamba_config: only pass keys that MambaBlock/MambaLayer2D accept.
        # Bio-harmonic keys (f_resp, f_card, etc.) are model-level context,
        # NOT Mamba SSM constructor args.
        _raw_mamba = mamba_config or {}
        _VALID_MAMBA_KEYS = {
            "d_state",
            "d_conv",
            "expand",
            "dropout",
            "linearization_mode",  # consumed by MambaLayer2D
        }
        self.mamba_config = {k: v for k, v in _raw_mamba.items() if k in _VALID_MAMBA_KEYS}
        # Store rejected keys for potential downstream use (e.g., bio-harmonic conditioning)
        self._mamba_extra = {k: v for k, v in _raw_mamba.items() if k not in _VALID_MAMBA_KEYS}
        self.use_complex_conv = use_complex_conv

        # Time embedding MLP (similar to NAFNet)
        self.time_mlp = None
        self.time_embedding_dim = None
        if time_embedding_dim is not None:
            self.time_embedding_dim = time_embedding_dim
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim, time_embedding_dim * 4),
                nn.SiLU(),
                nn.Linear(time_embedding_dim * 4, time_embedding_dim),
            )

        if use_complex_conv:
            from spectramr.models.layers.complex_conv import ComplexConv2d

            self.Conv2d = ComplexConv2d
        else:
            self.Conv2d = nn.Conv2d

        # Initial Conv
        if use_complex_conv:
            # ComplexConv2d expects separate real/imag channels internally if we looked at other impls,
            # but usually we pass (B, 2C, H, W).
            # The standard ComplexConv2d wrapper processes in_channels//2 complex channels.
            # So if we pass in_channels=2 (1 complex), it works.
            # But we must be careful with 'features' dims.
            # Mamba/Norms see 'features[0]'. If features[0]=32, does that mean 16 complex channels?
            # Yes, usually we keep standard feature definition (total channels).
            self.inc = nn.Sequential(
                self.Conv2d(
                    in_channels // 2 if use_complex_conv else in_channels,
                    features[0] // 2 if use_complex_conv else features[0],
                    kernel_size=3,
                    padding=1,
                ),
                # Norms: InstanceNorm2d works on channels. For complex, we normalize stacked.
                # Or we use GroupNorm. InstanceNorm is fine for stacked.
                nn.InstanceNorm2d(features[0]),
                nn.LeakyReLU(0.2),  # Standard ReLU on stacked (Split-ReLU)
            )
        else:
            self.inc = nn.Sequential(
                nn.Conv2d(in_channels, features[0], kernel_size=3, padding=1),
                nn.InstanceNorm2d(features[0]),
                nn.LeakyReLU(0.2),
            )

        # Encoder Layers
        self.downs = nn.ModuleList()
        # Downsampling operators
        self.pool = nn.MaxPool2d(2)

        in_feat = features[0]
        for feature in features[1:]:
            self.downs.append(self._make_layer(in_feat, feature))
            in_feat = feature

        # Bottleneck
        if use_complex_conv:
            # Mamba operates on the feature dim. If 64 channels (32 complex), Mamba sees 64.
            # This is "Hybrid" mode for Mamba itself (mixing real/imag via real weights).
            self.bottleneck = MambaLayer2D(
                features[-1], time_embedding_dim=time_embedding_dim, **self.mamba_config
            )
        else:
            self.bottleneck = MambaLayer2D(
                features[-1], time_embedding_dim=time_embedding_dim, **self.mamba_config
            )

        # Decoder Layers
        self.ups = nn.ModuleList()
        # Reversing features for decoder
        decoder_features = features[::-1]

        for i in range(len(decoder_features) - 1):
            if use_complex_conv:
                # Use Upsample + ComplexConv instead of TransposeConv
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        # Reduce channels: in -> out
                        # Conv2d Args: (in // 2, out // 2)
                        self.Conv2d(
                            decoder_features[i] // 2,
                            decoder_features[i + 1] // 2,
                            kernel_size=3,
                            padding=1,
                        ),
                    )
                )
            else:
                self.ups.append(
                    nn.ConvTranspose2d(
                        decoder_features[i],
                        decoder_features[i + 1],
                        kernel_size=2,
                        stride=2,
                    )
                )

            # Then the mixing block
            # Input to mixing block is Cat(skip, up_out).
            # skip has decoder_features[i+1]. up_out has decoder_features[i+1].
            # Total in = 2 * decoder_features[i+1].
            self.ups.append(self._make_layer(decoder_features[i + 1] * 2, decoder_features[i + 1]))

        # Final Conv
        if use_complex_conv:
            # 1x1 Complex Conv
            self.final_conv = self.Conv2d(
                features[0] // 2, out_channels // 2, kernel_size=1, padding=0
            )
        else:
            self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _make_layer(self, in_c, out_c):
        """_make_layer.

        Args:
            in_c (Any): Description.
            out_c (Any): Description.
        Returns:
            Any: Description.
        """
        if self.use_complex_conv:
            # Complex Block
            return TimeAwareSequential(
                self.Conv2d(in_c // 2, out_c // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2),
                # Mamba sees full stacked channels
                MambaLayer2D(
                    out_c,
                    time_embedding_dim=self.time_embedding_dim,
                    **self.mamba_config,
                ),
                self.Conv2d(out_c // 2, out_c // 2, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )
        else:
            return TimeAwareSequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2),
                MambaLayer2D(
                    out_c,
                    time_embedding_dim=self.time_embedding_dim,
                    **self.mamba_config,
                ),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_c),
                nn.LeakyReLU(0.2),
            )

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        ssm_params: list[SSMParameters] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Forward pass with optional SSM parameter injection from HyperMambaBridge.

        Args:
            x: Input tensor [B, C, H, W].
            time: Diffusion time [B] (legacy).
            timesteps: Diffusion timesteps [B].
            ssm_params: Optional list of SSMParameters from HyperMambaBridge.
                Each element conditions one Mamba layer (encoder layers first,
                then decoder layers). If the list is shorter than the number
                of Mamba layers, remaining layers run with default weights.
            **kwargs: Additional kwargs (ignored).

        Returns:
            Reconstructed output [B, C_out, H, W].

        forward method for MambaUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            time (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            ssm_params (list[SSMParameters] | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        emb = None
        if self.time_mlp is not None:
            t = timesteps if timesteps is not None else time
            if t is not None:
                t_emb = self._get_sinusoidal_embedding(t, self.time_embedding_dim)
                emb = self.time_mlp(t_emb)

        # Track which SSM param index to use
        ssm_idx = 0

        # Initial
        x1 = self.inc(x)  # Level 0
        skips = [x1]

        feat = x1
        # Encoder — each down_layer is a TimeAwareSequential with a MambaLayer2D
        for _i, down_layer in enumerate(self.downs):
            feat = self.pool(feat)
            ssm_override = None
            if ssm_params is not None and ssm_idx < len(ssm_params):
                ssm_override = ssm_params[ssm_idx]
                ssm_idx += 1
            feat = down_layer(feat, emb, ssm_override=ssm_override)
            skips.append(feat)

        # Bottleneck — standalone MambaLayer2D (no SSM override in bottleneck)
        bot = self.bottleneck(feat, emb)

        # Decoder
        x = bot
        skips = skips[::-1]  # Reverse skips

        skip_idx = 1
        for i in range(0, len(self.ups), 2):
            up_conv = self.ups[i]
            conv_block = self.ups[i + 1]

            x = up_conv(x)
            skip = skips[skip_idx]
            skip_idx += 1

            # Simple resize if mismatch
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            concat = torch.cat([skip, x], dim=1)
            ssm_override = None
            if ssm_params is not None and ssm_idx < len(ssm_params):
                ssm_override = ssm_params[ssm_idx]
                ssm_idx += 1
            x = conv_block(concat, emb, ssm_override=ssm_override)

        out = self.final_conv(x)
        return out

    def _get_sinusoidal_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal time embeddings.

        Args:
            timesteps: (B,) or (B, 1) tensor of timesteps
            dim: Embedding dimension

        Returns:
            (B, dim) tensor of embeddings
        """
        if timesteps.dim() == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)

        device = timesteps.device
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class CSMMambaLayer(nn.Module):
    """
    Cross-Scan Module (CSM) for Mamba.
    Scans the 2D image in 4 directions (Row, Row-Rev, Col, Col-Rev) to capture global context
    without 1D causality violations (striping artifacts).
    """

    def __init__(self, channels: int, time_embedding_dim: int | None = None, **mamba_kwargs):
        """__init__.

        Args:
            channels (int): Description.
            time_embedding_dim (int | None): Description.
        """
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        # We can share weights across directions or use separate ones.
        # Vim uses separate SSMs. We'll implement separate SSMs for maximum expressivity.
        # But to save memory, we can use 2 (Row/Col) or 1 (Shared).
        # Let's use 2: Horizontal and Vertical. Each handles forward/backward?
        # Standard Mamba is causal. To handle bidirectional, we typically need 2 SSMs per axis.
        # So 4 SSMs total for 4 directions.
        self.directions = 4
        self.mamba_blocks = nn.ModuleList(
            [MambaBlock(d_model=channels, **mamba_kwargs) for _ in range(self.directions)]
        )

        # Merge layer (Linear projection)
        self.merge = nn.Linear(channels * 4, channels)

        # Time embedding projection
        self.time_proj = None
        if time_embedding_dim is not None:
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, channels))

    def forward(self, x: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for CSMMambaLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time embedding injection
        if self.time_proj is not None and emb is not None:
            t = self.time_proj(emb)
            x = x + t[..., None, None]

        x_norm = x.permute(0, 2, 3, 1).contiguous()
        x_norm = self.norm(x_norm)  # (B, H, W, C)

        # Ensure contiguity after norm for cuDNN GRU operations
        x_norm = x_norm.contiguous()

        # 1. Row-Major (Top-Left -> Bottom-Right)
        x_r = x_norm.view(B, H * W, C).contiguous()

        # 2. Row-Reverse (Bottom-Right -> Top-Left)
        x_r_rev = torch.flip(x_r, dims=[1]).contiguous()

        # 3. Col-Major (Transposed)
        x_c = x_norm.permute(0, 2, 1, 3).contiguous().view(B, W * H, C).contiguous()

        # 4. Col-Reverse
        x_c_rev = torch.flip(x_c, dims=[1]).contiguous()

        # Run SSMs (all inputs are now contiguous)
        out_r = self.mamba_blocks[0](x_r)
        out_r_rev = self.mamba_blocks[1](x_r_rev)
        out_c = self.mamba_blocks[2](x_c)
        out_c_rev = self.mamba_blocks[3](x_c_rev)

        # Un-scan and merge
        out_r_rev = torch.flip(out_r_rev, dims=[1])

        out_c = out_c.view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        out_c_rev = (
            torch.flip(out_c_rev, dims=[1]).view(B, W, H, C).permute(0, 2, 1, 3).contiguous()
        )
        out_r = out_r.view(B, H, W, C)
        out_r_rev = out_r_rev.view(B, H, W, C)

        # Concatenate features
        # (B, H, W, 4*C)
        merged = torch.cat([out_r, out_r_rev, out_c, out_c_rev], dim=-1)

        # Project back to C
        out = self.merge(merged)

        # Restore layout
        return out.permute(0, 3, 1, 2).contiguous()


@register_model(
    name="mamba_unet_v2",
    training_mode="reconstruction",
    # Mirrors ``mamba_unet``, the base this subclasses (which declares (2,)).
    # Absent is not inherited: an undeclared spatial_dims reads as None and the
    # audit's spatial-rank check simply does not run (#1084).
    #
    # Verified rather than assumed: v2 builds and forwards at rank 2 (32, 32) and
    # FAILS at rank 3 with "Expected 3D (unbatched) or 4D (batched) input to
    # conv2d" -- identical to the base. That failure is in the UNet skeleton's
    # conv2d, not in the Mamba block, so it holds regardless of whether the real
    # mamba_ssm kernel or the Gated-Conv+GRU fallback is in use.
    spatial_dims=(2,),
)
class MambaUNet_v2(MambaUNet):
    """
    Version 2 of MambaUNet using Cross-Scan Module (CSM) to eliminate striping artifacts.
    Inherits structure from MambaUNet but replaces MambaLayer2D with CSMMambaLayer.
    """

    def _make_layer(self, in_c, out_c):
        """_make_layer.

        Args:
            in_c (Any): Description.
            out_c (Any): Description.
        Returns:
            Any: Description.
        """
        return TimeAwareSequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2),
            CSMMambaLayer(out_c, time_embedding_dim=self.time_embedding_dim, **self.mamba_config),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(0.2),
        )

    def __init__(self, *args, **kwargs):
        # We need to override init to use our _make_layer inside the super().__init__
        # But MambaUNet sets bottleneck in __init__.
        # So we must override __init__ mostly to set the bottleneck to use CSM.
        # Or better: Just reimplement __init__ calling super and then replacing bottleneck?
        # MambaUNet calls _make_layer in __init__, so if we define _make_layer BEFORE calling super().__init__, it works?
        # Yes, method resolution finds subclass method.
        # However, the bottleneck is instantiated directly in MambaUNet.__init__ using MambaLayer2D.
        # So we must replace self.bottleneck AFTER super().__init__.

        """__init__."""
        super().__init__(*args, **kwargs)

        # Replace bottleneck
        self.bottleneck = CSMMambaLayer(
            self.features[-1],
            time_embedding_dim=self.time_embedding_dim,
            **self.mamba_config,
        )
