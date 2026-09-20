"""Unit tests for the KAN (Kolmogorov-Arnold) layer.

Verifies B-spline edge correctness, fp32-internal numerical safety under
autocast, gradient flow, and basic shape contracts. CPU-only — no CUDA
needed for any assertion here.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.kan_layer import KANMLP, KANLayer


class TestKANLayer:
    """Test the single KANLayer building block."""

    def test_init_validates_dims(self):
        """Constructor rejects non-positive dims and zero spline order."""
        with pytest.raises(ValueError, match="positive dims"):
            KANLayer(in_dim=0, out_dim=4)
        with pytest.raises(ValueError, match="positive dims"):
            KANLayer(in_dim=4, out_dim=-1)
        with pytest.raises(ValueError, match="spline_order"):
            KANLayer(in_dim=4, out_dim=4, spline_order=0)

    def test_forward_shape(self):
        """Forward pass produces ``[..., out_dim]`` regardless of leading dims."""
        layer = KANLayer(in_dim=8, out_dim=16)

        # 2D input (batch, features)
        x_2d = torch.randn(4, 8)
        assert layer(x_2d).shape == (4, 16)

        # 3D input (batch, seq, features)
        x_3d = torch.randn(2, 5, 8)
        assert layer(x_3d).shape == (2, 5, 16)

        # Single sample
        x_1d = torch.randn(1, 8)
        assert layer(x_1d).shape == (1, 16)

    def test_forward_rejects_wrong_input_dim(self):
        """Last-dim mismatch raises a clear error."""
        layer = KANLayer(in_dim=8, out_dim=4)
        with pytest.raises(ValueError, match="expected last dim"):
            layer(torch.randn(2, 7))

    def test_b_spline_basis_partition_of_unity(self):
        """For x inside the grid range, the B-spline basis sums to ~1.

        This is the partition-of-unity property of B-splines and acts as a
        sanity check that the Cox-de Boor recursion is implemented correctly.
        """
        layer = KANLayer(in_dim=4, out_dim=4, grid_size=8, spline_order=3)
        # Sample inside the grid range (-1, 1)
        x = torch.linspace(-0.9, 0.9, 50).unsqueeze(-1).expand(-1, 4)
        bases = layer.b_splines(x)  # [50, 4, grid_size + spline_order]
        sums = bases.sum(dim=-1)  # [50, 4]
        # Tolerance is loose because the recursion has small numerical noise
        # near boundaries even within the nominal range.
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-3)

    def test_gradient_flows(self):
        """Backward pass yields finite gradients on both spline + base weights."""
        layer = KANLayer(in_dim=4, out_dim=8)
        x = torch.randn(3, 4, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()
        assert layer.spline_weight.grad is not None
        assert torch.isfinite(layer.spline_weight.grad).all()
        assert layer.base_weight.grad is not None
        assert torch.isfinite(layer.base_weight.grad).all()

    def test_fp32_internal_under_bf16_autocast_cpu(self):
        """Forward stays finite when wrapped in CPU bf16 autocast.

        The KANLayer disables autocast internally so B-spline ratios run
        in fp32. Without that guard, bf16 ratio terms can NaN.
        """
        layer = KANLayer(in_dim=8, out_dim=8)
        x = torch.randn(4, 8)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            y = layer(x)
        assert torch.isfinite(y).all()
        # Output dtype follows the surrounding autocast context (bf16) because
        # the layer casts back at exit.
        assert y.dtype == torch.float32 or y.dtype == torch.bfloat16

    def test_deterministic_with_seed(self):
        """Same seed -> identical output."""
        torch.manual_seed(0)
        layer1 = KANLayer(in_dim=4, out_dim=4)
        torch.manual_seed(0)
        layer2 = KANLayer(in_dim=4, out_dim=4)
        x = torch.randn(2, 4)
        assert torch.allclose(layer1(x), layer2(x))


class TestKANLayerGridExtension:
    """Test ``KANLayer.update_grid_from_samples`` (Liu et al. 2024 §2.2)."""

    def test_no_op_with_too_few_samples(self):
        """Grid update silently no-ops when given fewer samples than knots."""
        layer = KANLayer(in_dim=4, out_dim=2, grid_size=8, spline_order=3)
        old_grid = layer.grid.clone()
        old_weight = layer.spline_weight.clone()
        x = torch.randn(3, 4)  # only 3 samples; need >= grid_size + spline_order + 1 = 12
        layer.update_grid_from_samples(x)
        assert torch.equal(layer.grid, old_grid)
        assert torch.equal(layer.spline_weight, old_weight)

    def test_grid_replaced_when_enough_samples(self):
        """With enough samples the grid is updated in place."""
        layer = KANLayer(in_dim=2, out_dim=4, grid_size=5, spline_order=3)
        old_grid = layer.grid.clone()
        # Shift the input distribution well outside the default [-1, 1] grid.
        x = torch.randn(200, 2) * 3.0 + 5.0  # roughly N(5, 3)
        layer.update_grid_from_samples(x)
        # Grid should have moved (mean of any input dim's grid is now near 5).
        new_grid = layer.grid
        assert not torch.equal(new_grid, old_grid)
        assert new_grid.mean().item() > 1.0  # moved away from origin

    def test_functional_near_invariance_on_support(self):
        """After grid update, ``f(x)`` on the original support is approximately
        unchanged — the LSQ fit preserves function values up to small error.
        """
        layer = KANLayer(in_dim=3, out_dim=4, grid_size=8, spline_order=3)
        # Use a *fixed* input distribution that lives inside the initial grid.
        torch.manual_seed(0)
        x = torch.rand(500, 3) * 1.4 - 0.7  # samples in [-0.7, 0.7]
        with torch.no_grad():
            y_before = layer(x)
        layer.update_grid_from_samples(x)
        with torch.no_grad():
            y_after = layer(x)
        # The base SiLU residual is unchanged so its contribution cancels;
        # only the spline part is re-fit. Looser tolerance because LSQ on
        # a finite sample is approximate.
        max_err = (y_after - y_before).abs().max().item()
        assert max_err < 0.05, f"max error after grid update: {max_err}"

    def test_grid_alignment_after_distribution_shift(self):
        """After updating with shifted samples, the grid bracket aligns
        with the new sample range (within a small margin)."""
        layer = KANLayer(in_dim=1, out_dim=2, grid_size=10, spline_order=3)
        x = torch.linspace(-3.0, 3.0, 200).unsqueeze(-1)  # uniform in [-3, 3]
        layer.update_grid_from_samples(x, margin=0.05)
        # First inner knot should be near -3, last near +3 (within margin).
        # Inner knots live at indices [spline_order : -spline_order] of the grid.
        inner = layer.grid[0, layer.spline_order : -layer.spline_order]
        assert inner[0].item() < -2.5
        assert inner[-1].item() > 2.5

    def test_sample_collection_buffers_inputs(self):
        """Forward with collect_samples=True buffers detached ON-DEVICE copies.

        The buffer is intentionally kept on the INPUT device (NOT moved to CPU):
        a ``.cpu()`` here is a blocking device->host sync fired on every forward
        during the 10K-iter grid-collection warmup, across ~48 KAN layers x
        grad-accum x grad-checkpoint re-runs — serialising an otherwise
        GPU-bound step (.claude/rules/performance.md). Regression guard for the
        2026-07-01 sync fix (on CUDA this catches a re-introduced ``.cpu()``).
        """
        layer = KANLayer(in_dim=3, out_dim=4)
        layer.set_sample_collection(True)
        assert layer.collect_samples is True
        assert layer._collected_samples == []
        x = torch.randn(8, 3)
        _ = layer(x)
        assert len(layer._collected_samples) == 1
        # Buffered tensor is detached (no grad), fp32, correct shape, and stays
        # on the SAME device as the input (device-agnostic — no host sync).
        buf = layer._collected_samples[0]
        assert buf.shape == (8, 3)
        assert buf.dtype == torch.float32
        assert not buf.requires_grad
        assert buf.device == x.device

    def test_sample_collection_disabled_by_default(self):
        layer = KANLayer(in_dim=3, out_dim=4)
        assert layer.collect_samples is False
        _ = layer(torch.randn(8, 3))
        assert layer._collected_samples == []

    def test_sample_buffer_evicts_oldest(self):
        """FIFO eviction keeps total buffered rows under the cap."""
        layer = KANLayer(in_dim=2, out_dim=2)
        layer.set_sample_collection(True)
        layer._max_buffered_samples = 100
        for _ in range(10):
            _ = layer(torch.randn(40, 2))
        total = sum(t.shape[0] for t in layer._collected_samples)
        assert total <= 100, f"buffer overflowed: {total}"

    def test_set_sample_collection_false_clears_buffer(self):
        layer = KANLayer(in_dim=3, out_dim=4)
        layer.set_sample_collection(True)
        _ = layer(torch.randn(8, 3))
        assert len(layer._collected_samples) > 0
        layer.set_sample_collection(False)
        assert layer._collected_samples == []
        # Subsequent forward does NOT re-buffer
        _ = layer(torch.randn(8, 3))
        assert layer._collected_samples == []

    def test_update_grid_uses_buffer_when_x_is_none(self):
        """Calling update_grid_from_samples() with no arg uses the buffer."""
        layer = KANLayer(in_dim=2, out_dim=2, grid_size=5, spline_order=3)
        layer.set_sample_collection(True)
        old_grid = layer.grid.clone()
        # Feed a distribution far from the default grid range to force a move
        for _ in range(5):
            _ = layer(torch.randn(50, 2) * 2.5 + 4.0)
        layer.update_grid_from_samples()  # no x arg => uses buffer
        assert not torch.equal(layer.grid, old_grid)
        # Buffer is consumed after a successful update
        assert layer._collected_samples == []

    def test_update_grid_no_op_when_buffer_empty(self):
        """update_grid_from_samples() with no buffered samples is a no-op."""
        layer = KANLayer(in_dim=2, out_dim=2)
        old_grid = layer.grid.clone()
        old_weight = layer.spline_weight.clone()
        layer.update_grid_from_samples()  # buffer empty AND no x arg
        assert torch.equal(layer.grid, old_grid)
        assert torch.equal(layer.spline_weight, old_weight)

    def test_optimizer_state_preserved_after_update(self):
        """Calling update_grid_from_samples should not break Optimizer
        bookkeeping. We use ``.data.copy_`` rather than ``= nn.Parameter(...)``
        so the optimizer's reference to the same Parameter object stays valid.
        """
        layer = KANLayer(in_dim=4, out_dim=2)
        opt = torch.optim.Adam(layer.parameters(), lr=1e-3)
        # Take one step to populate optimizer state for both params
        x = torch.randn(20, 4, requires_grad=True)
        loss = layer(x).sum()
        loss.backward()
        opt.step()
        spline_id_before = id(layer.spline_weight)
        # Now update grid
        layer.update_grid_from_samples(torch.randn(200, 4))
        # Same Parameter object -> optimizer state still applies
        assert id(layer.spline_weight) == spline_id_before
        # And it can keep stepping without error
        opt.zero_grad()
        loss2 = layer(x).sum()
        loss2.backward()
        opt.step()


class TestKANMLP:
    """Test the multi-layer KAN wrapper."""

    def test_requires_at_least_two_dims(self):
        with pytest.raises(ValueError, match="at least input and output"):
            KANMLP(dims=[8])

    def test_forward_shape_through_stack(self):
        mlp = KANMLP(dims=[3, 16, 1])
        x = torch.randn(5, 3)
        assert mlp(x).shape == (5, 1)

    def test_forward_three_layers(self):
        """Three-layer KAN [3 -> 8 -> 8 -> 1]."""
        mlp = KANMLP(dims=[3, 8, 8, 1])
        x = torch.randn(2, 3)
        y = mlp(x)
        assert y.shape == (2, 1)
        assert torch.isfinite(y).all()

    def test_gradient_flows_through_stack(self):
        mlp = KANMLP(dims=[4, 8, 1])
        x = torch.randn(2, 4, requires_grad=True)
        loss = mlp(x).sum()
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ── the grid fit must be identical on every rank ─────────────────────────────
def test_the_sample_gather_is_a_passthrough_without_a_process_group():
    """Single-process training must be unaffected by the distributed path."""
    import torch as _t

    from spectramr.models.blocks.kan_layer import _gather_samples_across_ranks

    x = _t.randn(64, 4)
    assert _gather_samples_across_ranks(x) is x


def test_the_grid_update_fits_the_gathered_samples_not_the_local_shard():
    """The rank-local fit is what splits `grid` from `spline_weight`.

    ``update_grid_from_samples`` solves knots and coefficients together. Fitting
    per-rank makes each rank solve a different pair, and DDP then broadcasts
    ``grid`` (a persistent buffer) from rank 0 without re-broadcasting
    ``spline_weight`` (a parameter) -- so every rank evaluates a gate function
    that no rank actually fitted. Pinned by source because a real four-rank
    process group is not available in a unit test.
    """
    import inspect

    from spectramr.models.blocks import kan_layer

    src = inspect.getsource(kan_layer.KANLayer.update_grid_from_samples)
    assert "_gather_samples_across_ranks(x_flat)" in src, (
        "the grid is fitted to the rank-local shard again"
    )
    gather = inspect.getsource(kan_layer._gather_samples_across_ranks)
    assert "all_gather" in gather and "ReduceOp.MIN" in gather, (
        "the gather must be shape-safe across ranks with unequal sample counts"
    )


def test_the_grid_still_moves_when_it_is_refitted():
    """Guards the two checks above from passing on a no-op update."""
    import torch as _t

    from spectramr.models.blocks.kan_layer import KANLayer

    _t.manual_seed(0)
    layer = KANLayer(in_dim=4, out_dim=3, grid_size=5)
    before = layer.grid.clone()
    layer.update_grid_from_samples(_t.randn(256, 4) * 3.0)
    assert not _t.equal(before, layer.grid)
