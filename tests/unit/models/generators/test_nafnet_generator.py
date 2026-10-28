"""NAFNetGenerator's conditioning contract; the embedding pathway itself is in test_backbone_conditioning."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.nafnet_generator import NAFNetGenerator


def _net() -> NAFNetGenerator:
    torch.manual_seed(0)
    return NAFNetGenerator(in_channels=8, out_channels=8, width=64, use_complex_conv=True).eval()


class TestContrastEmbeddingBatch:
    """Exactly one contrast row per sample; a truncated batch or a broadcast row raises."""

    def test_contrast_batch_larger_than_the_timestep_batch_raises(self) -> None:
        net = _net()
        with pytest.raises(ValueError, match="contrast_emb batch 3"), torch.no_grad():
            net(torch.randn(2, 8, 32, 32), torch.tensor([1, 2]), contrast_emb=torch.randn(3, 256))

    @pytest.mark.parametrize(("rows", "t_rows"), [(1, 2), (2, 3)])
    def test_a_broadcast_row_or_a_larger_timestep_batch_raises(self, rows, t_rows) -> None:
        """A single row used to broadcast onto every sample (#2055)."""
        net = _net()
        x = torch.randn(t_rows, 8, 32, 32)
        with pytest.raises(ValueError, match=f"contrast_emb batch {rows}"), torch.no_grad():
            net(x, torch.arange(1, t_rows + 1), contrast_emb=torch.randn(rows, 256))

    @pytest.mark.parametrize("rows", [2])
    def test_per_sample_rows_pass(self, rows) -> None:
        net = _net()
        x = torch.randn(2, 8, 32, 32)
        with torch.no_grad():
            out = net(x, torch.tensor([1, 2]), contrast_emb=torch.randn(rows, 256))
        assert out.shape == x.shape
