#!/usr/bin/env python
"""NAFNet Generator Implementation
Nonlinear Activation Free Network for MRI Super-Resolution

NAFNet uses SimpleGate activation and simplified channel attention
for efficient and effective image restoration.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from spectramr.models.generators.grad_checkpointing import GradCheckpointingMixin
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class SimpleGate(nn.Module):
    """Simple Gate activation function from NAFNet."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for SimpleGate.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


# The local copy of TimeAwareSequential is GONE — it gated embedding delivery on
# ``isinstance(module, (NAFBlock, TimeAwareSequential))``, and ``ComplexNAFBlock``
# is not a subclass of ``NAFBlock``. So under ``use_complex_conv: true`` every
# block failed the check and was called as ``module(x)``: the time embedding was
# computed, then silently dropped before it reached a single block. Measured on
# an arm's own kwargs, the block received ``emb=None``.
#
# The canonical implementation decides by INSPECTING the forward signature, so it
# handles any block that accepts an ``emb`` parameter — which is why the local
# copy was already marked deprecated in favour of it (canonical homes,
# non-negotiable 6). Importing it fixes the drift rather than patching the
# isinstance tuple, which would only work until the next block class appears.
from spectramr.models.blocks.time_aware_sequential import (  # noqa: E402
    TimeAwareSequential,
)


class NAFBlock(nn.Module):
    """NAFNet Block with simplified channel attention and time conditioning."""

    def __init__(
        self,
        c: int,
        DW_Expand: int = 2,
        drop_out_rate: float = 0.0,
        time_embedding_dim: int | None = None,
    ):
        """__init__.

        Args:
            c (int): Description.
            DW_Expand (int): Description.
            drop_out_rate (float): Description.
            time_embedding_dim (int | None): Description.
        """
        super().__init__()

        # Time embedding projection
        self.time_proj = None
        if time_embedding_dim is not None:
            # Simple linear projection for efficient addition
            self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_embedding_dim, c))

        # Depthwise convolution
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(
            in_channels=c,
            out_channels=dw_channel,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv2 = nn.Conv2d(
            in_channels=dw_channel,
            out_channels=dw_channel,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=dw_channel,
            bias=True,
        )
        self.conv3 = nn.Conv2d(
            in_channels=dw_channel // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        # simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=dw_channel,
                out_channels=dw_channel,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
            ),
            nn.Sigmoid(),
        )

        # SimpleGate activation
        self.sg = SimpleGate()

        # Feature modulation
        self.conv4 = nn.Conv2d(
            in_channels=c,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )
        self.conv5 = nn.Conv2d(
            in_channels=c // 2,
            out_channels=c,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
        )

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp: torch.Tensor, emb: torch.Tensor | None = None) -> torch.Tensor:
        """forward.

        Args:
            inp (torch.Tensor): Description.
            emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for NAFBlock.

        Executes PyTorch tensor operations.

        Args:
            inp (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        x = inp

        # Time Embedding Injection (Add to input features)
        if self.time_proj is not None and emb is not None:
            t = self.time_proj(emb)
            # Broadcast: (B, C) -> (B, C, 1, 1)
            x = x + t[..., None, None]

        x = self.conv1(x)
        x = self.conv2(x)
        x = x * self.sca(x)
        x = self.sg(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(y)
        x = self.sg(x)
        x = self.conv5(x)  # Use conv5 to handle c//2 -> c

        x = self.dropout2(x)

        return y + x * self.gamma


@register_model(name="nafnet", training_mode="reconstruction", spatial_dims=(2,))
class NAFNetGenerator(GradCheckpointingMixin, IGenerator, nn.Module):
    """NAFNet Generator for MRI Cold Diffusion.

    Refactored for MRI Physics:
    1. Complex-Valued I/O (2 channels).
    2. Linear Output (Identity) to preserve dynamic range.
    3. Time-Conditioned Feature Modulation.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        enc_blk_nums: list[int] = [1, 1, 1, 28],
        middle_blk_num: int = 1,
        dec_blk_nums: list[int] = [1, 1, 1, 1],
        use_complex_conv: bool = False,
        contrast_emb_dim: int | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            width (int): Description.
            enc_blk_nums (list[int]): Description.
            middle_blk_num (int): Description.
            dec_blk_nums (list[int]): Description.
            use_complex_conv (bool): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.use_complex_conv = use_complex_conv

        # Determine Block Types
        if use_complex_conv:
            # Use Complex Blocks
            from spectramr.models.blocks.complex_blocks import ComplexNAFBlock as BlockClass
            from spectramr.models.layers.complex_conv import ComplexConv2d

            self.Conv2d = ComplexConv2d
            self.NAFBlock = BlockClass
        else:
            self.Conv2d = nn.Conv2d
            self.NAFBlock = NAFBlock  # Uses the global NAFBlock class defined in this file

        # Intro Convolution
        # NAFNet uses 1 conv to lift to 'width'
        if use_complex_conv:
            self.intro = self.Conv2d(in_channels // 2, width // 2, kernel_size=3, padding=1)
        else:
            self.intro = nn.Conv2d(in_channels, width, 3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # Time Embedding
        self.time_embedding_dim = width * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim),
        )

        # NAFNet's conditioning width is width*4, which is NOT the width the
        # generator builds ``contrast_embedding`` at (that is
        # model_kwargs.time_embedding_dim). For experiment_11e_nafnet those are
        # 48*4 = 192 vs 256, so the two genuinely disagree and a bare add would
        # be a shape error at the first forward. Project when the caller declares
        # the incoming width; raise in forward() only when it was never declared,
        # so the mismatch surfaces as a message rather than a broadcast.
        self.contrast_proj: nn.Module | None = None
        if contrast_emb_dim is not None and contrast_emb_dim != self.time_embedding_dim:
            self.contrast_proj = nn.Linear(contrast_emb_dim, self.time_embedding_dim)

        chan = width
        # Encoders
        for num in enc_blk_nums:
            self.encoders.append(
                TimeAwareSequential(
                    *[
                        self.NAFBlock(chan, time_embedding_dim=self.time_embedding_dim)
                        for _ in range(num)
                    ]
                )
            )
            # Downsample
            if use_complex_conv:
                self.downs.append(self.Conv2d(chan // 2, (chan * 2) // 2, 2, stride=2))
            else:
                self.downs.append(nn.Conv2d(chan, chan * 2, 2, stride=2))
            chan = chan * 2

        self.middle_blks = TimeAwareSequential(
            *[
                self.NAFBlock(chan, time_embedding_dim=self.time_embedding_dim)
                for _ in range(middle_blk_num)
            ]
        )

        # Decoders
        for num in dec_blk_nums:
            # Upsample
            if use_complex_conv:
                # Complex Upsample strategy: Upsample + Conv
                self.ups.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        self.Conv2d(
                            chan // 2, (chan // 2) // 2, 1, padding=0
                        ),  # 1x1 conv to reduce channels
                    )
                )
            else:
                self.ups.append(
                    nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False), nn.PixelShuffle(2))
                )

            chan = chan // 2
            self.decoders.append(
                TimeAwareSequential(
                    *[
                        self.NAFBlock(chan, time_embedding_dim=self.time_embedding_dim)
                        for _ in range(num)
                    ]
                )
            )

        self.padder_size = 2 ** len(enc_blk_nums)

        # Ending
        if use_complex_conv:
            self.ending = self.Conv2d(width // 2, out_channels // 2, 3, padding=1)
        else:
            self.ending = nn.Conv2d(
                width, out_channels, 3, padding=1, stride=1, groups=1, bias=True
            )

        # [MRI Fix] Output Activation
        # Use Identity (Linear) to avoid clipping high dynamic range MRI signals.
        # Tanh is strictly forbidden for MRI reconstruction.
        self.output_activation = nn.Identity()


    def _get_sinusoidal_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Generate sinusoidal timestep embeddings."""
        import math

        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    @property
    def name(self) -> str:
        """Returns the model name."""
        return "NAFNet_MRI"

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass through NAFNet with optional time conditioning.

        forward method for NAFNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, H, W = x.shape

        # Time Embedding
        emb = None
        if self.time_mlp is not None and timesteps is not None:
            t_emb = self._get_sinusoidal_embedding(timesteps, self.time_embedding_dim)
            emb = self.time_mlp(t_emb)

        # Contrast conditioning, mirroring ComplexUNet (complex_unet.py:333).
        # Previously swallowed by **kwargs, so every arm declaring num_contrasts
        # ran without it. Added onto the time embedding so it reaches each
        # NAFBlock through the existing ``time_proj`` path.
        #
        # NOTE for anyone testing this: NAFBlock's ``beta``/``gamma`` residual
        # scalers are ZERO-initialised (standard NAFNet init), and ``emb`` enters
        # only through the branch they scale — so neither timestep nor contrast
        # moves the output at initialisation. That is expected, not a wiring bug;
        # assert on the embedding path, or perturb beta/gamma first.
        contrast_emb = kwargs.get("contrast_emb")
        if contrast_emb is not None:
            if (
                self.contrast_proj is not None
                and contrast_emb.shape[-1] == self.contrast_proj.in_features
            ):
                contrast_emb = self.contrast_proj(contrast_emb)
            if contrast_emb.shape[-1] != self.time_embedding_dim:
                raise ValueError(
                    f"contrast_emb width {contrast_emb.shape[-1]} != NAFNet's "
                    f"time_embedding_dim {self.time_embedding_dim} (= width * 4). "
                    "Set model_kwargs.width so width*4 matches "
                    "model_kwargs.time_embedding_dim, which is the width the "
                    "generator builds contrast_embedding at."
                )
            if emb is None:
                emb = contrast_emb
            else:
                if contrast_emb.shape[0] != emb.shape[0]:
                    contrast_emb = contrast_emb[: emb.shape[0]]
                emb = emb + contrast_emb

        # Input
        x = self.intro(x)

        # Encoder
        ckpt = self._checkpointing_active()
        encs = []
        for encoder, down in zip(self.encoders, self.downs, strict=False):
            x = checkpoint(encoder, x, emb, use_reentrant=False) if ckpt else encoder(x, emb)
            encs.append(x)
            x = down(x)

        # Middle
        x = (
            checkpoint(self.middle_blks, x, emb, use_reentrant=False)
            if ckpt
            else self.middle_blks(x, emb)
        )

        # Decoder
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1], strict=False):
            x = up(x)
            x = x + enc_skip
            x = checkpoint(decoder, x, emb, use_reentrant=False) if ckpt else decoder(x, emb)

        # Output
        x = self.ending(x)
        x = self.output_activation(x)

        return x

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples from latent space (for compatibility)."""
        return self.forward(z, **kwargs)

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Returns the output shape for a given input shape."""
        batch_size, channels, height, width = input_shape
        return (batch_size, self.out_channels, height, width)

    def get_parameter_count(self) -> int:
        """Returns the total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
