"""Multi-contrast fusion generator for multimodal super-resolution.
Combines information from multiple MRI contrasts (T1, T2, FLAIR, etc.)
for improved super-resolution performance.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.instrument_keyed_attention import (
    AttentionKeys,
    InstrumentKeyedCrossAttention,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class CrossAttentionFusion(nn.Module):
    """Cross-attention mechanism for fusing multiple contrast features.
    Enables each contrast to attend to information from other contrasts.
    """

    def __init__(
        self,
        channels: int,
        num_contrasts: int,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_contrasts (int): Description.
            heads (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.channels = channels
        self.num_contrasts = num_contrasts
        self.heads = heads
        self.head_dim = channels // heads

        assert channels % heads == 0, "channels must be divisible by heads"

        # Query, Key, Value projections for cross-attention
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

        # Contrast-specific position embeddings
        self.contrast_pos_embed = nn.Parameter(torch.randn(1, num_contrasts, channels))

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (batch, num_contrasts, channels, H, W)

        Returns:
            Output tensor of shape (batch, channels, H, W)

        forward method for CrossAttentionFusion.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size, num_contrasts, channels, height, width = x.shape

        # Add position embeddings
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(-1, num_contrasts, channels)
        x_flat = x_flat + self.contrast_pos_embed

        # Reshape for attention: (batch * H * W, num_contrasts, channels)
        x_flat = x_flat.view(-1, num_contrasts, channels)

        # Project Q, K, V
        q = self.q_proj(x_flat)  # (batch*H*W, num_contrasts, channels)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Split into heads
        q = q.view(-1, num_contrasts, self.heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch*H*W, heads, num_contrasts, head_dim)
        k = k.view(-1, num_contrasts, self.heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.view(-1, num_contrasts, self.heads, self.head_dim)
        v = v.transpose(1, 2)

        # Cross-attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(-1, num_contrasts, channels)

        # Output projection
        output = self.out_proj(attn_output)

        # Weighted fusion across contrasts
        # (batch*H*W, heads, num_contrasts)
        weights = torch.softmax(attn_weights.mean(dim=1), dim=-1)
        weights = weights.mean(dim=1)  # (batch*H*W, num_contrasts)

        # Apply weights to get fused features
        fused = torch.sum(
            output * weights.unsqueeze(-1),
            dim=1,
        )  # (batch*H*W, channels)

        # Reshape back to spatial dimensions
        fused = fused.view(batch_size, height, width, channels)
        fused = fused.permute(0, 3, 1, 2)

        return fused


class FusionResBlock(nn.Module):
    """Residual block for decoder layers."""

    def __init__(self, in_channels: int, out_channels: int):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
        """
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for FusionResBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = F.relu(out)

        out = self.upsample(out)
        return out


class AdaptiveFusionBlock(nn.Module):
    """Adaptive fusion block that learns optimal fusion weights for each contrast."""

    def __init__(self, channels: int, num_contrasts: int):
        """__init__.

        Args:
            channels (int): Description.
            num_contrasts (int): Description.
        """
        super().__init__()
        self.num_contrasts = num_contrasts

        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(num_contrasts))

        # Contrast-specific gating
        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, channels // 16, 1),
                    nn.ReLU(),
                    nn.Conv2d(channels // 16, channels, 1),
                    nn.Sigmoid(),
                )
                for _ in range(num_contrasts)
            ],
        )

    def forward(self, contrast_features: torch.Tensor) -> torch.Tensor:
        """Args:
            contrast_features: list of tensors, each of shape
                (batch, channels, H, W)

        Returns:
            Fused tensor of shape (batch, channels, H, W)

        forward method for AdaptiveFusionBlock.

        Executes PyTorch tensor operations.

        Args:
            contrast_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size, channels, height, width = contrast_features[0].shape

        # Apply gating to each contrast
        gated_features = []
        for i, feat in enumerate(contrast_features):
            gate = self.gates[i](feat)
            gated_feat = feat * gate
            gated_features.append(gated_feat)

        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        fused = torch.zeros_like(gated_features[0])

        for i, feat in enumerate(gated_features):
            fused += weights[i] * feat

        return fused

    def gate(self, frames: torch.Tensor) -> torch.Tensor:
        """Per-frame gating that KEEPS the frame axis: ``[B, N, C, H, W]``.

        :meth:`forward` collapses the frame axis, which is why this block could
        not compose with the fusion attention and ended up constructed but never
        called (issue #508): two fusion mechanisms cannot both consume the same
        axis. Splitting the gating out makes it a genuine pre-step, so both
        ``gates`` and ``fusion_weights`` receive gradient instead of sitting in
        the optimizer contributing nothing.

        ``fusion_weights`` is renormalised by ``N`` so a uniform initialisation
        is exactly the identity, and the gate cannot silently rescale the
        features it is meant to reweight.
        """
        weights = F.softmax(self.fusion_weights, dim=0) * self.num_contrasts
        gated = [
            frames[:, i] * self.gates[i](frames[:, i]) * weights[i]
            for i in range(self.num_contrasts)
        ]
        return torch.stack(gated, dim=1)


@register_model(name="multicontrast_fusion", training_mode="reconstruction", spatial_dims=(2,))
class MultiContrastFusionGenerator(nn.Module, IGenerator):
    """Enhanced multi-contrast fusion generator that combines multiple MRI
    contrasts for super-resolution. Uses advanced attention mechanisms and
    adaptive fusion.
    """

    def __init__(
        self,
        n_contrasts: int = 3,  # Number of input contrasts (T1, T2, FLAIR)
        in_channels: int | None = None,  # TOTAL input channels (see below)
        channels_per_stream: int = 1,  # Channels per contrast/frame
        out_channels: int = 1,  # Output channels
        base_channels: int = 64,
        num_layers: int = 4,
        attention_heads: int = 8,
        dropout_rate: float = 0.1,
        use_residual: bool = True,
        use_cross_attention: bool = True,
        use_adaptive_fusion: bool = True,
        output_activation: str | None = None,
        attention_keys: AttentionKeys | None = None,
        instrument_channels: bool = False,
        conditioning_per_stream: int = 0,
        scale: int = 1,
    ):
        """__init__.

        Args:
            n_contrasts (int): Description.
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            attention_heads (int): Description.
            dropout_rate (float): Description.
            use_residual (bool): Description.
            use_cross_attention (bool): Description.
            use_adaptive_fusion (bool): Description.
            output_activation (Optional[str]): Description.
        """
        super().__init__()

        # ``in_channels`` is the TOTAL width of the input tensor, matching every
        # other generator in the framework and what the model factory passes
        # from ``config.model.in_channels``. It used to mean channels-per-
        # contrast here, which no caller could have discovered: the module
        # crashed on every forward pass (issue #508), so nothing depended on the
        # old reading. Per-stream width is now ``channels_per_stream``.
        self.n_contrasts = n_contrasts
        self.in_channels = channels_per_stream
        self.out_channels = out_channels
        self.input_channels = n_contrasts * channels_per_stream
        self.use_cross_attention = use_cross_attention
        self.use_adaptive_fusion = use_adaptive_fusion

        # Per-stream conditioning appended to every frame before encoding:
        # the multi-frame SR path hands each frame its own (dy, dx) offset, and
        # a network that cannot see which frame moved where has to re-derive
        # the registration it was already given.
        if conditioning_per_stream < 0:
            raise ValueError(f"conditioning_per_stream must be >= 0, got {conditioning_per_stream}")
        self.conditioning_per_stream = int(conditioning_per_stream)

        # Initial convolution to standardize input dimensions (per contrast)
        self.input_conv = nn.Conv2d(
            self.in_channels + self.conditioning_per_stream,
            base_channels,
            kernel_size=3,
            padding=1,
        )

        # Contrast-specific feature extractors
        self.contrast_encoders = nn.ModuleList(
            [self._build_encoder(base_channels, num_layers) for _ in range(n_contrasts)],
        )

        # Multi-contrast fusion mechanisms
        encoder_out_channels = base_channels * (2**num_layers)

        # ``attention_keys`` selects WHERE the fusion attention's queries and
        # keys come from, and is the scientific variable of the instrument-keyed
        # arms. Left unset, the legacy boolean chooses as it always did.
        self.attention_keys: AttentionKeys | None = attention_keys
        if attention_keys is not None:
            self.fusion_attention = InstrumentKeyedCrossAttention(
                channels=encoder_out_channels,
                n_frames=n_contrasts,
                keys=attention_keys,
                heads=attention_heads,
                dropout=dropout_rate,
            )
        elif use_cross_attention:
            self.fusion_attention = CrossAttentionFusion(
                channels=encoder_out_channels,
                num_contrasts=n_contrasts,
                heads=attention_heads,
                dropout=dropout_rate,
            )
        else:
            # Fallback to original attention
            self.fusion_attention = MultiContrastAttention(
                channels=encoder_out_channels,
                num_contrasts=n_contrasts,
                heads=attention_heads,
                dropout=dropout_rate,
            )

        if use_adaptive_fusion:
            self.adaptive_fusion = AdaptiveFusionBlock(
                channels=encoder_out_channels,
                num_contrasts=n_contrasts,
            )

        # Decoder with skip connections
        self.decoder = self._build_decoder(base_channels, num_layers, use_residual)

        # Whether the input tensor carries the instrument stack after the
        # anatomy stack. It is present on EVERY rung of an attention_keys
        # ablation, zero-filled where unused, so architecture, parameter count
        # and input geometry are held fixed and only the routing varies (the
        # same design as the shift-knowledge ladder).
        self.instrument_channels = bool(instrument_channels)
        if attention_keys == "marker" and not self.instrument_channels:
            raise ValueError(
                "attention_keys='marker' needs instrument_channels=True: the "
                "marker features are what the queries and keys are derived "
                "from, and there is nowhere else for them to arrive."
            )
        # [anatomy_0..N-1] [instrument_0..N-1]? [cond_0, cond_1, ..., cond_N-1]
        # The conditioning block is frame-major, matching how the multi-
        # acquisition strategy lays out its (dy, dx) maps.
        self.expected_in_channels = (
            self.input_channels * (2 if self.instrument_channels else 1)
            + self.n_contrasts * self.conditioning_per_stream
        )
        if in_channels is not None and int(in_channels) != self.expected_in_channels:
            raise ValueError(
                f"model.in_channels={in_channels} disagrees with the input "
                f"contract this configuration implies: {n_contrasts} streams x "
                f"{channels_per_stream} channels"
                + (" x 2 (anatomy + instrument)" if self.instrument_channels else "")
                + (
                    f" + {n_contrasts} x {conditioning_per_stream} conditioning"
                    if conditioning_per_stream
                    else ""
                )
                + f" = {self.expected_in_channels}. Raising rather than "
                "preferring one of them: a silently-ignored channel count is "
                "how a stream ends up reading another stream's data."
            )

        # Super-resolution tail. The encoders max-pool by 2**num_layers and the
        # decoder returns to the INPUT grid, so without this the module can only
        # ever emit at the resolution it was given (noted in issue #508).
        if scale < 1:
            raise ValueError(f"scale must be a positive integer, got {scale}")
        self.scale = int(scale)
        self.upsampler: nn.Module = (
            nn.Identity()
            if self.scale == 1
            else nn.Sequential(
                nn.Conv2d(base_channels, base_channels * self.scale**2, 3, padding=1),
                nn.PixelShuffle(self.scale),
            )
        )

        # Output convolution
        self.output_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Output activation
        self.output_activation = self._get_activation(output_activation)

        # Initialize weights
        self.apply(self._init_weights)

    def _encode_streams(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """Encode ``n_contrasts`` streams to ``[B, N, C, H', W']``.

        ``cond`` is the frame-major per-stream conditioning block; stream ``i``
        receives its own slice, so the anatomy and instrument streams for the
        same frame see the same offset.
        """
        k = self.conditioning_per_stream
        feats = []
        for i in range(self.n_contrasts):
            lo = i * self.in_channels
            stream = x[:, lo : lo + self.in_channels]
            if cond is not None:
                stream = torch.cat((stream, cond[:, i * k : (i + 1) * k]), dim=1)
            feats.append(self.contrast_encoders[i](self.input_conv(stream)))
        return torch.stack(feats, dim=1)

    def _build_encoder(self, base_channels: int, num_layers: int) -> nn.Sequential:
        """Build encoder layers with increasing channel depth.

        Returns ``nn.Sequential``, not ``nn.ModuleList``. ``forward`` calls the
        result (``self.contrast_encoders[i](feat)``) and a ``ModuleList`` is a
        container with no ``forward``, so every forward pass raised
        ``NotImplementedError`` (issue #508).
        """
        layers: list[nn.Module] = []

        for i in range(num_layers):
            in_channels = base_channels * (2**i)
            out_channels = base_channels * (2 ** (i + 1))

            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                ),
            )

        return nn.Sequential(*layers)

    def _build_decoder(
        self,
        base_channels: int,
        num_layers: int,
        use_residual: bool,
    ) -> nn.Sequential:
        """Build decoder layers with decreasing channel depth.

        Two fixes over the pre-2026-07-26 form, both required for the module to
        run at all (issue #508). It returns ``nn.Sequential`` for the same reason
        the encoder does, and it no longer doubles ``in_channels`` for a skip
        connection: ``forward`` never carried encoder activations across, so the
        first decoder layer expected twice the channels the fusion produced and
        would have raised on a shape mismatch even once the container bug was
        fixed. Encoder skips are a real improvement to make later, with the
        activations actually plumbed through.
        """
        layers: list[nn.Module] = []

        for i in range(num_layers - 1, -1, -1):
            in_channels = base_channels * (2 ** (i + 1))
            out_channels = base_channels * (2**i)

            if use_residual:
                layers.append(FusionResBlock(in_channels, out_channels))
            else:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True),
                        nn.Upsample(
                            scale_factor=2,
                            mode="bilinear",
                            align_corners=False,
                        ),
                    ),
                )

        return nn.Sequential(*layers)

    def _get_activation(self, activation_name: str | None) -> nn.Module | None:
        """Get activation function by name."""
        if activation_name is None:
            return None
        if activation_name.lower() == "tanh":
            return nn.Tanh()
        if activation_name.lower() == "sigmoid":
            return nn.Sigmoid()
        if activation_name.lower() == "relu":
            return nn.ReLU()
        raise ValueError(f"Unknown activation: {activation_name}")

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize network weights."""
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through multi-contrast fusion generator.

        Args:
            x: Input tensor of shape
                (batch_size, n_contrasts * in_channels, H, W)

        Returns:
            Output tensor of shape (batch_size, out_channels, H, W)

        forward method for MultiContrastFusionGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.shape[1] != self.expected_in_channels:
            raise ValueError(
                f"expected {self.expected_in_channels} input channels: "
                f"{self.n_contrasts} x {self.in_channels} anatomy"
                + (
                    f" + {self.n_contrasts} x {self.in_channels} instrument"
                    if self.instrument_channels
                    else ""
                )
                + (
                    f" + {self.n_contrasts} x {self.conditioning_per_stream} conditioning"
                    if self.conditioning_per_stream
                    else ""
                )
                + f". Got {x.shape[1]}."
            )

        n_img = self.input_channels * (2 if self.instrument_channels else 1)
        cond = x[:, n_img:] if self.conditioning_per_stream else None
        anatomy = self._encode_streams(x[:, : self.input_channels], cond)
        # Only the marker keys read the instrument features; encoding them
        # for content/none would be discarded compute.
        instrument = (
            self._encode_streams(x[:, self.input_channels : n_img], cond)
            if self.instrument_channels and self.attention_keys == "marker"
            else None
        )

        if self.use_adaptive_fusion:
            # Per-frame gating that KEEPS the frame axis, so it composes with
            # the attention instead of competing for the same reduction. Before
            # 2026-07-26 this block was constructed and never called (#508).
            anatomy = self.adaptive_fusion.gate(anatomy)
            if instrument is not None:
                instrument = self.adaptive_fusion.gate(instrument)

        if isinstance(self.fusion_attention, InstrumentKeyedCrossAttention):
            fused_features = self.fusion_attention(anatomy, instrument)
        else:
            fused_features = self.fusion_attention(anatomy)

        # Decode, then upsample to the target grid
        decoded = self.upsampler(self.decoder(fused_features))

        # Output convolution
        output = self.output_conv(decoded)

        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)

        return output

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Get output shape for given input shape."""
        # Input: (batch_size, n_contrasts * in_channels, H, W)
        # Output: (batch_size, out_channels, H, W)
        batch_size, _input_channels, height, width = input_shape
        return (
            batch_size,
            self.out_channels,
            height * self.scale,
            width * self.scale,
        )

    def get_parameter_count(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def name(self) -> str:
        """Return descriptive model name."""
        return f"MultiContrastFusion_{self.n_contrasts}C_{self.out_channels}Cls"

    def generate(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate super-resolved output from multi-contrast input.
        For generators, this is equivalent to forward pass.
        """
        return self.forward(x)


class MultiContrastAttention(nn.Module):
    """Multi-head attention mechanism for fusing multiple contrast features."""

    def __init__(
        self,
        channels: int,
        num_contrasts: int,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        """__init__.

        Args:
            channels (int): Description.
            num_contrasts (int): Description.
            heads (int): Description.
            dropout (float): Description.
        """
        super().__init__()
        self.channels = channels
        self.num_contrasts = num_contrasts
        self.heads = heads
        self.head_dim = channels // heads

        assert channels % heads == 0, "channels must be divisible by heads"

        # Query, Key, Value projections (along channel dimension)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (batch, num_contrasts, channels, H, W)

        Returns:
            Output tensor of shape (batch, channels, H, W)

        forward method for MultiContrastAttention.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        batch_size, num_contrasts, channels, height, width = x.shape

        # Reshape for attention: (batch * H * W * num_contrasts, channels)
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(-1, channels)

        # Project Q, K, V
        q = self.q_proj(x_flat)  # (batch*H*W*num_contrasts, channels)
        k = self.k_proj(x_flat)
        v = self.v_proj(x_flat)

        # Reshape for multi-head attention
        # (batch*H*W, num_contrasts, channels)
        spatial_size = batch_size * height * width
        q = q.view(spatial_size, num_contrasts, channels)
        k = k.view(spatial_size, num_contrasts, channels)
        v = v.view(spatial_size, num_contrasts, channels)

        # Split into heads
        q = q.view(spatial_size, num_contrasts, self.heads, self.head_dim)
        q = q.transpose(1, 2)  # (batch*H*W, heads, num_contrasts, head_dim)
        k = k.view(spatial_size, num_contrasts, self.heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.view(spatial_size, num_contrasts, self.heads, self.head_dim)
        v = v.transpose(1, 2)

        # Attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=getattr(self.dropout, "p", 0.0) if self.training else 0.0,
            scale=self.scale,
        )

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(
            batch_size * height * width,
            num_contrasts,
            channels,
        )

        # Output projection
        output = self.out_proj(attn_output)

        # Average across contrasts to get fused features
        output = output.mean(dim=1)  # (batch*H*W, channels)

        # Reshape back to spatial dimensions
        output = output.view(batch_size, height, width, channels)
        output = output.permute(0, 3, 1, 2)

        return output


def create_multicontrast_fusion_generator(
    n_contrasts: int = 3,
    in_channels: int = 1,
    out_channels: int = 1,
    base_channels: int = 64,
    num_layers: int = 4,
    attention_heads: int = 8,
    dropout_rate: float = 0.1,
    use_residual: bool = True,
    output_activation: str | None = None,
) -> MultiContrastFusionGenerator:
    """Factory function for creating MultiContrastFusionGenerator instances.

    Args:
        n_contrasts: Number of input contrasts
        in_channels: Channels per contrast
        out_channels: Output channels
        base_channels: Base number of channels in network
        num_layers: Number of encoder/decoder layers
        attention_heads: Number of attention heads for fusion
        dropout_rate: Dropout rate for attention
        use_residual: Whether to use residual connections
        output_activation: Output activation function name

    Returns:
        Configured MultiContrastFusionGenerator instance

    """
    return MultiContrastFusionGenerator(
        n_contrasts=n_contrasts,
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        attention_heads=attention_heads,
        dropout_rate=dropout_rate,
        use_residual=use_residual,
        output_activation=output_activation,
    )
