"""Unit tests for CrossAttentionOracleUNet model."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.cross_attention_oracle import (
    CrossAttentionOracleUNet,
    MarkerCrossAttention,
)
from spectramr.models.registry import MODEL_REGISTRY


class TestMarkerCrossAttention:
    """Tests for the cross-attention oracle layer."""

    def test_init(self) -> None:
        """MarkerCrossAttention initializes with embed_dim."""
        mca = MarkerCrossAttention(embed_dim=128)
        assert mca.embed_dim == 128

    def test_forward_shape(self) -> None:
        """Output should match input anatomy features shape."""
        mca = MarkerCrossAttention(embed_dim=128, num_heads=4, oracle_channels=2)
        anatomy = torch.randn(2, 8, 128)  # [B, N, D]
        marker_residual = torch.randn(2, 2, 64, 64)
        out = mca(anatomy, marker_residual)
        assert out.shape == (2, 8, 128)

    def test_forward_2d_anatomy(self) -> None:
        """Works with 2D anatomy features [B, D] (unsqueezed internally)."""
        mca = MarkerCrossAttention(embed_dim=128, num_heads=4)
        anatomy = torch.randn(2, 128)  # [B, D]
        marker_residual = torch.randn(2, 2, 64, 64)
        out = mca(anatomy, marker_residual)
        assert out.shape == (2, 1, 128)


class TestCrossAttentionOracleUNet:
    """Tests for the full CrossAttentionOracleUNet model."""

    @pytest.fixture
    def model(self) -> CrossAttentionOracleUNet:
        return CrossAttentionOracleUNet(
            in_channels=2,
            out_channels=2,
            features=[16, 32],
            embed_dim=64,
            num_heads=4,
        )

    def test_forward_with_marker(self, model: CrossAttentionOracleUNet) -> None:
        """Forward with marker_residual produces correct output shape."""
        x = torch.randn(2, 2, 64, 64)
        marker_res = torch.randn(2, 2, 64, 64)
        out = model(x, marker_residual=marker_res)
        assert out.shape == (2, 2, 64, 64)

    def test_forward_without_marker(self, model: CrossAttentionOracleUNet) -> None:
        """Forward without marker_residual (blind mode) works."""
        x = torch.randn(2, 2, 64, 64)
        out = model(x)
        assert out.shape == (2, 2, 64, 64)

    def test_return_latent(self, model: CrossAttentionOracleUNet) -> None:
        """return_latent=True returns (output, z_latent) tuple."""
        x = torch.randn(2, 2, 64, 64)
        marker_res = torch.randn(2, 2, 64, 64)
        result = model(x, marker_residual=marker_res, return_latent=True)
        assert isinstance(result, tuple)
        output, z_latent = result
        assert output.shape == (2, 2, 64, 64)
        assert z_latent.shape == (2, 64)  # embed_dim

    def test_multicoil_input_with_2ch_oracle(self) -> None:
        """Regression (method_a/c/ib crash, smoke 2026-06-15): the VF / IB /
        distillation strategies real-stack a 2-coil SVD complex anatomy into a
        4-channel ``x`` while the oracle marker stays 2-channel (single marker
        r/i). The model must accept ``in_channels=4`` for the main branch and a
        2-channel ``marker_residual`` — the arms had ``in_channels=2`` and
        crashed at the first conv ('expected 2 channels, got 4'). out_channels
        matches the 4-channel (2-coil) target."""
        model = CrossAttentionOracleUNet(
            in_channels=4, out_channels=4, oracle_channels=2,
            features=[16, 32], embed_dim=64, num_heads=4,
        )
        x = torch.randn(2, 4, 64, 64)          # 2 coils x (real, imag)
        marker_res = torch.randn(2, 2, 64, 64)  # single marker r/i
        out = model(x, marker_residual=marker_res)
        assert out.shape == (2, 4, 64, 64)

    def test_oracle_mechanism_is_not_inert(self) -> None:
        """The cross-attention oracle must actually change the output when a
        marker is supplied (pitfall #16 facade guard) — passing the marker is
        not a no-op vs blind mode."""
        model = CrossAttentionOracleUNet(
            in_channels=4, out_channels=4, oracle_channels=2,
            features=[16, 32], embed_dim=64, num_heads=4,
        ).eval()
        x = torch.randn(2, 4, 64, 64)
        marker_res = torch.randn(2, 2, 64, 64)
        with torch.no_grad():
            blind = model(x, marker_residual=None)
            oracle = model(x, marker_residual=marker_res)
        assert not torch.allclose(blind, oracle), "oracle cross-attention is inert"

    def test_return_latent_without_marker(
        self, model: CrossAttentionOracleUNet
    ) -> None:
        """return_latent without marker returns tensor (no latent available)."""
        x = torch.randn(2, 2, 64, 64)
        result = model(x, return_latent=True)
        # Without marker, should return just the tensor
        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 2, 64, 64)

    def test_registered(self) -> None:
        """Model is registered in MODEL_REGISTRY."""
        assert "cross_attention_oracle_unet" in MODEL_REGISTRY
        assert (
            MODEL_REGISTRY["cross_attention_oracle_unet"]["class"]
            is CrossAttentionOracleUNet
        )

    def test_latent_dim_attribute(self, model: CrossAttentionOracleUNet) -> None:
        """latent_dim attribute exposed for distillation."""
        assert model.latent_dim == 64


class TestUseMarkerOff:
    """``use_marker: false`` is the marker-off control for experiment_vf_method_a."""

    @staticmethod
    def _net(use_marker: bool) -> CrossAttentionOracleUNet:
        torch.manual_seed(0)
        return CrossAttentionOracleUNet(
            features=[4, 8], embed_dim=16, num_heads=2, use_marker=use_marker
        )

    def test_off_builds_no_marker_branch(self) -> None:
        off, on = self._net(False), self._net(True)
        assert not hasattr(off, "cross_attention") and hasattr(on, "cross_attention")
        names = {n for n, _ in off.named_parameters()}
        assert not any(n.startswith(("token_proj", "from_tokens")) for n in names)

    def test_off_ignores_a_supplied_marker(self) -> None:
        net = self._net(False)
        x = torch.randn(1, 2, 16, 16)
        blind = net(x)
        with_marker = net(x, marker_residual=torch.randn(1, 2, 16, 16))
        assert torch.equal(blind, with_marker)

    def test_on_responds_to_the_marker(self) -> None:
        net = self._net(True)
        x = torch.randn(1, 2, 16, 16)
        assert not torch.allclose(net(x), net(x, marker_residual=torch.randn(1, 2, 16, 16)))

    def test_rejects_a_non_bool(self) -> None:
        with pytest.raises(TypeError, match="use_marker"):
            CrossAttentionOracleUNet(features=[4, 8], embed_dim=16, num_heads=2, use_marker="no")
