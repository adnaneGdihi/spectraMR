"""Comprehensive unit tests for UNet building blocks.

Tests the UNet-specific components including Down, Up, OutConv blocks
for both 2D and 3D variants.
"""

from unittest.mock import patch

import torch
from torch import nn

from spectramr.models.blocks.unet import Down, Down3D, OutConv, OutConv3D, Up, Up3D


# @pytest.fixture(autouse=True)
def clean_mocks():
    """Ensure mocks are cleaned up before/after each test."""
    patch.stopall()
    yield
    patch.stopall()


class TestDown:
    """Test Down block."""

    def test_down_initialization(self):
        """Test Down block initialization."""
        down = Down(in_channels=64, out_channels=128)

        assert isinstance(down.maxpool_conv, nn.Sequential)
        assert len(down.maxpool_conv) == 2
        assert isinstance(down.maxpool_conv[0], nn.MaxPool2d)
        assert isinstance(down.maxpool_conv[1], nn.Module)  # DoubleConv

    def test_down_forward(self):
        """Test Down block forward pass."""
        down = Down(in_channels=3, out_channels=64)

        # Input: (batch, channels, height, width)
        x = torch.randn(2, 3, 32, 32)
        output = down(x)

        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 64  # out_channels
        assert output.shape[2] == 16  # height / 2
        assert output.shape[3] == 16  # width / 2

    def test_down_shape_consistency(self):
        """Test Down block output shapes."""
        down = Down(in_channels=128, out_channels=256)

        x = torch.randn(1, 128, 64, 64)
        output = down(x)

        assert output.shape == (1, 256, 32, 32)

    def test_down_gradient_flow(self):
        """Test gradient flow through Down block."""
        down = Down(in_channels=32, out_channels=64)
        x = torch.randn(2, 32, 16, 16, requires_grad=True)

        output = down(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestUp:
    """Test Up block."""

    def test_up_initialization_bilinear(self):
        """Test Up block initialization with bilinear upsampling."""
        up = Up(in_channels=256, out_channels=128, bilinear=True)

        assert hasattr(up, "up")
        assert hasattr(up, "conv")
        assert isinstance(up.up, nn.Upsample)
        assert up.up.mode == "bilinear"
        assert up.up.scale_factor == 2

    def test_up_initialization_transpose(self):
        """Test Up block initialization with transpose convolution."""
        up = Up(in_channels=256, out_channels=128, bilinear=False)

        assert isinstance(up.up, nn.ConvTranspose2d)
        assert up.up.kernel_size == (2, 2)
        assert up.up.stride == (2, 2)

    def test_up_forward_bilinear(self):
        """Test Up block forward pass with bilinear upsampling."""
        up = Up(in_channels=256, out_channels=128, bilinear=True)

        # x1: encoder feature (upsampled), x2: skip connection
        # After upsampling, x1 will have in_channels//2 = 128 channels
        # x2 should also have 128 channels so concat gives 256 total
        x1 = torch.randn(2, 128, 16, 16)  # from deeper layer
        x2 = torch.randn(2, 128, 32, 32)  # skip connection

        output = up(x1, x2)

        assert output.shape[0] == 2
        assert output.shape[1] == 128  # out_channels
        assert output.shape[2] == 32
        assert output.shape[3] == 32

    def test_up_forward_transpose(self):
        """Test Up block forward pass with transpose convolution."""
        up = Up(in_channels=256, out_channels=128, bilinear=False)

        # For transpose conv, x1 should have in_channels = 256 channels
        # After transpose upsampling, it becomes in_channels//2 = 128 channels
        # x2 should have 128 channels so concat gives 256 total
        x1 = torch.randn(2, 256, 16, 16)  # from deeper layer
        x2 = torch.randn(2, 128, 32, 32)  # skip connection

        output = up(x1, x2)

        assert output.shape == (2, 128, 32, 32)

    def test_up_padding_handling(self):
        """Test Up block handles different input sizes with padding."""
        up = Up(in_channels=128, out_channels=64, bilinear=True)

        # x1 will be upsampled from (7,7) to (14,14), should have in_channels//2 = 64 channels
        # x2 has 64 channels, so concat gives 128 total
        x1 = torch.randn(1, 64, 7, 7)  # Will be upsampled to 14x14
        x2 = torch.randn(1, 64, 15, 15)  # Different size

        output = up(x1, x2)

        # Should handle the size difference with padding
        assert output.shape[0] == 1
        assert output.shape[1] == 64
        # Size should match x2 after upsampling and padding
        assert output.shape[2] == 15
        assert output.shape[3] == 15

    def test_up_gradient_flow(self):
        """Test gradient flow through Up block."""
        up = Up(in_channels=128, out_channels=64, bilinear=True)

        # x1 should have in_channels//2 = 64 channels
        x1 = torch.randn(2, 64, 8, 8, requires_grad=True)
        x2 = torch.randn(2, 64, 16, 16, requires_grad=True)

        output = up(x1, x2)
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestOutConv:
    """Test OutConv block."""

    def test_out_conv_initialization(self):
        """Test OutConv initialization."""
        out_conv = OutConv(in_channels=64, out_channels=1)

        assert isinstance(out_conv.conv, nn.Conv2d)
        assert out_conv.conv.kernel_size == (1, 1)
        assert out_conv.conv.in_channels == 64
        assert out_conv.conv.out_channels == 1
        assert isinstance(out_conv.activation, nn.Identity)

    def test_out_conv_with_activation(self):
        """Test OutConv with custom activation."""
        activation = nn.Sigmoid()
        out_conv = OutConv(in_channels=32, out_channels=3, activation=activation)

        assert out_conv.activation is activation

    def test_out_conv_forward(self):
        """Test OutConv forward pass."""
        out_conv = OutConv(in_channels=128, out_channels=1)

        x = torch.randn(2, 128, 64, 64)
        output = out_conv(x)

        assert output.shape == (2, 1, 64, 64)

    def test_out_conv_with_sigmoid(self):
        """Test OutConv with sigmoid activation."""
        out_conv = OutConv(in_channels=64, out_channels=1, activation=nn.Sigmoid())

        x = torch.randn(1, 64, 32, 32)
        output = out_conv(x)

        assert output.shape == (1, 1, 32, 32)
        # Sigmoid should constrain output to (0, 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)

    def test_out_conv_gradient_flow(self):
        """Test gradient flow through OutConv."""
        out_conv = OutConv(in_channels=64, out_channels=3)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)

        output = out_conv(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestDown3D:
    """Test Down3D block."""

    def setUp(self):
        """Set up test fixtures."""
        patch.stopall()

    def test_down3d_initialization(self):
        """Test Down3D block initialization."""
        down3d = Down3D(in_channels=64, out_channels=128)

        assert isinstance(down3d.maxpool_conv, nn.Sequential)
        assert len(down3d.maxpool_conv) == 2
        assert isinstance(down3d.maxpool_conv[0], nn.MaxPool3d)
        assert isinstance(down3d.maxpool_conv[1], nn.Module)  # DoubleConv3D

    def test_down3d_forward(self):
        """Test Down3D block forward pass."""
        down3d = Down3D(in_channels=1, out_channels=32)

        # Input: (batch, channels, depth, height, width)
        x = torch.randn(2, 1, 16, 32, 32)
        output = down3d(x)

        assert output.shape[0] == 2  # batch size
        assert output.shape[1] == 32  # out_channels
        assert output.shape[2] == 8  # depth / 2
        assert output.shape[3] == 16  # height / 2
        assert output.shape[4] == 16  # width / 2

    def test_down3d_gradient_flow(self):
        """Test gradient flow through Down3D block."""
        down3d = Down3D(in_channels=16, out_channels=32)
        x = torch.randn(1, 16, 8, 16, 16, requires_grad=True)

        output = down3d(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestUp3D:
    """Test Up3D block."""

    def setUp(self):
        """Set up test fixtures."""
        patch.stopall()

    def test_up3d_initialization_trilinear(self):
        """Test Up3D block initialization with trilinear upsampling."""
        up3d = Up3D(in_channels=256, out_channels=128, trilinear=True)

        assert isinstance(up3d.up, nn.Upsample)
        assert up3d.up.mode == "trilinear"
        assert up3d.up.scale_factor == 2

    def test_up3d_initialization_transpose(self):
        """Test Up3D block initialization with transpose convolution."""
        up3d = Up3D(in_channels=256, out_channels=128, trilinear=False)

        assert isinstance(up3d.up, nn.ConvTranspose3d)
        assert up3d.up.kernel_size == (2, 2, 2)
        assert up3d.up.stride == (2, 2, 2)

    def test_up3d_forward_trilinear(self):
        """Test Up3D block forward pass with trilinear upsampling."""
        up3d = Up3D(in_channels=192, out_channels=64, trilinear=True)  # 128 + 64 = 192

        # x1 has 128 channels (upsampled keeps same channels), x2 has 64 channels
        x1 = torch.randn(2, 128, 4, 8, 8)  # from deeper layer
        x2 = torch.randn(2, 64, 8, 16, 16)  # skip connection

        output = up3d(x1, x2)

        assert output.shape == (2, 64, 8, 16, 16)

    def test_up3d_forward_transpose(self):
        """Test Up3D block forward pass with transpose convolution."""
        up3d = Up3D(in_channels=256, out_channels=128, trilinear=False)

        x1 = torch.randn(2, 128, 4, 8, 8)
        x2 = torch.randn(2, 128, 8, 16, 16)

        output = up3d(x1, x2)

        assert output.shape == (2, 128, 8, 16, 16)

    def test_up3d_padding_handling(self):
        """Test Up3D block handles different input sizes with padding."""
        up3d = Up3D(in_channels=64, out_channels=32, trilinear=True)

        # x1 has 64 channels, x2 has 32 channels, concat gives 96? Wait, that doesn't match.
        # Let me check: for trilinear, in_channels is the total after concat
        # So x1 + x2 should = in_channels = 64
        x1 = torch.randn(1, 32, 3, 7, 7)  # Will be upsampled to 6x14x14
        x2 = torch.randn(1, 32, 5, 13, 13)  # Different size

        output = up3d(x1, x2)

        # Should handle the size difference with padding
        assert output.shape == (1, 32, 5, 13, 13)

    def test_up3d_gradient_flow(self):
        """Test gradient flow through Up3D block."""
        up3d = Up3D(in_channels=64, out_channels=32, trilinear=True)

        # x1 + x2 should = in_channels = 64
        x1 = torch.randn(1, 32, 2, 4, 4, requires_grad=True)
        x2 = torch.randn(1, 32, 4, 8, 8, requires_grad=True)

        output = up3d(x1, x2)
        loss = output.sum()
        loss.backward()

        assert x1.grad is not None
        assert x2.grad is not None


class TestOutConv3D:
    """Test OutConv3D block."""

    def setUp(self):
        """Set up test fixtures."""
        patch.stopall()

    def test_out_conv3d_initialization(self):
        """Test OutConv3D initialization."""
        out_conv3d = OutConv3D(in_channels=64, out_channels=1)

        assert isinstance(out_conv3d.conv, nn.Conv3d)
        assert out_conv3d.conv.kernel_size == (1, 1, 1)
        assert out_conv3d.conv.in_channels == 64
        assert out_conv3d.conv.out_channels == 1
        assert isinstance(out_conv3d.activation, nn.Identity)

    def test_out_conv3d_with_activation(self):
        """Test OutConv3D with custom activation."""
        activation = nn.Tanh()
        out_conv3d = OutConv3D(in_channels=32, out_channels=3, activation=activation)

        assert out_conv3d.activation is activation

    def test_out_conv3d_forward(self):
        """Test OutConv3D forward pass."""
        out_conv3d = OutConv3D(in_channels=128, out_channels=1)

        x = torch.randn(2, 128, 8, 16, 16)
        output = out_conv3d(x)

        assert output.shape == (2, 1, 8, 16, 16)

    def test_out_conv3d_gradient_flow(self):
        """Test gradient flow through OutConv3D."""
        out_conv3d = OutConv3D(in_channels=64, out_channels=2)
        x = torch.randn(1, 64, 4, 8, 8, requires_grad=True)

        output = out_conv3d(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestUNetIntegration:
    """Test integration of UNet blocks."""

    def setUp(self):
        """Set up test fixtures."""
        patch.stopall()

    def test_unet_down_up_cycle_2d(self):
        """Test a down-up cycle with 2D blocks."""
        down = Down(in_channels=32, out_channels=64)
        up = Up(in_channels=96, out_channels=32, bilinear=True)  # 64 + 32 = 96

        # Simulate encoder-decoder flow
        x = torch.randn(1, 32, 64, 64)

        # Downsample
        x_down = down(x)  # (1, 64, 32, 32)

        # Upsample with skip connection
        x_up = up(x_down, x)  # (1, 32, 64, 64)

        assert x_up.shape == x.shape

    def test_unet_down_up_cycle_3d(self):
        """Test a down-up cycle with 3D blocks."""
        down3d = Down3D(in_channels=16, out_channels=32)
        up3d = Up3D(in_channels=48, out_channels=16, trilinear=True)  # 32 + 16 = 48

        x = torch.randn(1, 16, 16, 32, 32)

        # Downsample
        x_down = down3d(x)  # (1, 32, 8, 16, 16)

        # Upsample with skip connection
        x_up = up3d(x_down, x)  # (1, 16, 16, 32, 32)

        assert x_up.shape == x.shape

    def test_unet_end_to_end_2d(self):
        """Test end-to-end UNet-like flow with 2D blocks."""
        # Encoder
        down1 = Down(in_channels=1, out_channels=64)
        down2 = Down(in_channels=64, out_channels=128)

        # Decoder
        up1 = Up(in_channels=192, out_channels=64, bilinear=True)  # 128 + 64 = 192

        # Output
        out_conv = OutConv(in_channels=64, out_channels=1)

        x = torch.randn(1, 1, 64, 64)

        # Encoder path
        x1 = down1(x)  # (1, 64, 32, 32)
        x2 = down2(x1)  # (1, 128, 16, 16)

        # Decoder path - just test up1
        x3 = up1(x2, x1)  # (1, 64, 32, 32)

        # Output
        output = out_conv(x3)  # (1, 1, 32, 32)

        assert output.shape[0] == 1
        assert output.shape[1] == 1
        assert output.shape[2] == 32
        assert output.shape[3] == 32


# --- Regression tests for spectramr.models.reconstruction.unet -----------------
# These cover the configurable residual block's fail-loud contract:
#   * unimplemented AttentionType members must raise (pitfall #9/#10), never
#     silently degrade to nn.Identity.
#   * BlockType.DENSE is advertised by the enum but unimplemented; construction
#     must raise NotImplementedError instead of crashing mid-forward with an
#     UnboundLocalError on `out`.

import pytest

from spectramr.models.reconstruction.unet import (
    AttentionType,
    BlockType,
    ConfigurableResidualBlock,
)


class TestConfigurableResidualBlockAttentionContract:
    """Fail-loud contract for the attention dispatch."""

    @pytest.mark.parametrize(
        "at",
        [
            AttentionType.SELF,
            AttentionType.CROSS_CONTRAST_OLMPA,
            AttentionType.DUAL_DOMAIN,
            AttentionType.KAN_DUAL_DOMAIN,
            AttentionType.WAVELET_FREQ,
            AttentionType.KERNELIZED,
            AttentionType.SPARSE,
        ],
    )
    def test_residual_block_raises_on_unimplemented_attention(self, at):
        # pitfall #9/#10: unimplemented attention must fail loud, never no-op to
        # Identity.
        with pytest.raises(ValueError, match="does not implement attention_type"):
            ConfigurableResidualBlock(16, use_attention=True, attention_type=at)

    def test_residual_block_supported_attention_builds(self):
        # The two implemented members must build a real (non-Identity) module.
        ch_block = ConfigurableResidualBlock(
            16, use_attention=True, attention_type=AttentionType.CHANNEL
        )
        sp_block = ConfigurableResidualBlock(
            16, use_attention=True, attention_type=AttentionType.SPATIAL
        )
        assert not isinstance(ch_block.attention, nn.Identity)
        assert not isinstance(sp_block.attention, nn.Identity)

    def test_residual_block_no_attention_keeps_identity(self):
        # use_attention=False must keep nn.Identity regardless of attention_type,
        # and must NOT raise even for an otherwise-unimplemented member.
        block = ConfigurableResidualBlock(
            16, use_attention=False, attention_type=AttentionType.SELF
        )
        assert isinstance(block.attention, nn.Identity)


class TestConfigurableResidualBlockBlockTypeContract:
    """Fail-loud contract for the block_type dispatch."""

    def test_dense_block_type_raises_at_construction(self):
        # BlockType.DENSE is advertised by the enum but unimplemented; it must
        # fail fast at construction, not crash mid-forward with UnboundLocalError.
        with pytest.raises(NotImplementedError, match="DENSE|not implement"):
            ConfigurableResidualBlock(16, block_type=BlockType.DENSE)

    def test_supported_block_types_forward_roundtrip(self):
        # The two implemented block types must build and run a forward pass that
        # preserves shape (guards the numerics-unchanged contract of the fix).
        x = torch.randn(2, 16, 8, 8)
        for bt in (BlockType.STANDARD, BlockType.BOTTLENECK):
            block = ConfigurableResidualBlock(16, block_type=bt)
            out = block(x)
            assert out.shape == x.shape


class TestUNetPreservesInputSpatialSize:
    """The UNet contract (docstring): output preserves input H, W — including
    for non-power-of-two-divisible sizes that floor through the strided encoder."""

    def test_non_divisible_input_size_preserved(self):
        # REGRESSION (mrixfields b25_cartoon_texture, 2026-06-24): a non-divisible
        # input (W=436) floored to 432 through the encoder, and the
        # non-deep-supervision output path did NOT restore the input size, so a
        # downstream L1 raised "size of tensor a (432) must match b (436)". The
        # else-branch must interpolate back to (input_height, input_width) like
        # the deep-supervision branch already does.
        from spectramr.models.reconstruction.unet import UNet

        model = UNet(in_channels=1, out_channels=1)
        # Odd / non-2**N-divisible H and W guarantee an encoder floor.
        x = torch.randn(1, 1, 44, 52)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        assert out.shape[-2:] == x.shape[-2:], (
            f"UNet must preserve input H,W; got {tuple(out.shape[-2:])} for "
            f"input {tuple(x.shape[-2:])}"
        )

    def test_divisible_input_size_unchanged(self):
        # The guard keeps the common (divisible) case an exact no-op path.
        from spectramr.models.reconstruction.unet import UNet

        model = UNet(in_channels=1, out_channels=1)
        x = torch.randn(1, 1, 64, 64)
        out = model(x)
        out = out[0] if isinstance(out, tuple) else out
        assert out.shape[-2:] == x.shape[-2:]


class TestUNetDiscardsEveryDataConsistencyKwarg:
    """UNet has no DC layer, so it must drop the whole ``DC_SSOT_KEYS`` set.

    ``generator_kwargs`` step 3c forwards every row of
    :data:`~spectramr.infrastructure.physics.dc_settings.DC_SSOT_KEYS` into any
    generator whose constructor takes ``**kwargs`` — unconditionally, without
    consulting ``physics.data_consistency.enabled``. UNet accepts those kwargs
    only to throw them away, so its discard set has to be the table itself
    (non-negotiable 17: one owner), not a hand-copied subset of it.
    """

    @staticmethod
    def _dc_kwargs_from_schema_defaults() -> dict:
        """What step 3c forwards for a config that declares no ``physics:``."""
        from spectramr.config.schemas.physics import DataConsistencyConfig
        from spectramr.infrastructure.physics.dc_settings import ssot_pairs

        return dict(ssot_pairs(DataConsistencyConfig()))

    def test_default_physics_block_does_not_break_construction(self):
        # REGRESSION: DC_SSOT_KEYS grew the three noise keys, and UNet's
        # hand-written pop-list did not grow with it, so every ``standard_unet``
        # arm raised "Unexpected keyword argument 'train_noise_level' for
        # UNetConfig" at build time — including the 121 arms that declare no
        # physics block at all and the 24 that declare enabled: false.
        from spectramr.models.reconstruction.unet import UNet

        model = UNet(in_channels=1, out_channels=1, **self._dc_kwargs_from_schema_defaults())
        assert model.config.in_channels == 1

    def test_every_ssot_row_is_discarded(self):
        """Each row on its own, so a seventh row cannot land unhandled.

        This pin cannot go red by *addition* any more, and that is the point:
        UNet derives its discard set from ``DC_SSOT_KEYS`` rather than
        restating it, so growing the table grows the discard set. It goes red
        if someone reintroduces a hand-written list.
        """
        from spectramr.infrastructure.physics.dc_settings import DC_SSOT_KEYS
        from spectramr.models.reconstruction.unet import UNet

        forwarded = self._dc_kwargs_from_schema_defaults()
        for kwarg, _field in DC_SSOT_KEYS:
            UNet(in_channels=1, out_channels=1, **{kwarg: forwarded[kwarg]})

    def test_an_unknown_kwarg_still_raises(self):
        """The discard set widened; the strict check did not go away."""
        import pytest

        from spectramr.models.reconstruction.unet import UNet

        with pytest.raises(TypeError, match="not_a_unet_field"):
            UNet(in_channels=1, out_channels=1, not_a_unet_field=1)
