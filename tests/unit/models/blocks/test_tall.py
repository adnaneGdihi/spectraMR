"""Unit tests for TALL learnable SFC + privileged-learning gradient-reversal.

Pin three contracts:

1. **Motif tables** are valid permutations of 4 cells.
2. **TALL forward** returns a valid permutation (bijection of N voxels)
   in eval mode and a differentiable sequence in train mode.
3. **Gradient reversal** flips the sign of upstream gradients.

Sanity-shape parametrized tests are in ``TestTALLSanityShape``.
"""
from __future__ import annotations

import copy
import pytest
import torch
import torch.nn.functional as F

from spectramr.models.blocks.tall import (
    TALL,
    build_motif_permutations,
    expected_locality_loss,
    locality_loss,
    permutation_coords,
)


def _seeded_tall(seed: int = 0) -> TALL:
    torch.manual_seed(seed)
    return TALL(order=2, feature_dim=4)


class TestDifferentiableLocality:
    """Exact expected step length of the soft traversal (Spec §9.3)."""

    def test_one_hot_expected_equals_hard_permutation(self) -> None:
        m = _seeded_tall()
        m.eval()
        m.track_locality = True
        x = torch.randn(2, 4, 4, 4)
        with torch.no_grad():
            _, perm = m(x)
        choice = F.one_hot(m._last_choice.argmax(dim=-1), m.n_motifs).float()
        soft = expected_locality_loss(choice, m._last_ordered, m.side)
        hard = locality_loss(permutation_coords(perm, m.side), m.side)
        assert torch.allclose(soft, hard, atol=1e-6)

    def test_hard_permutation_locality_has_no_grad(self) -> None:
        # Documents the trap: the integer permutation path cannot carry a gradient.
        m = _seeded_tall()
        m.eval()
        with torch.no_grad():
            _, perm = m(torch.randn(1, 4, 4, 4))
        assert not locality_loss(permutation_coords(perm, m.side), m.side).requires_grad

    def test_uniform_choice_is_not_below_canonical_one_hot_on_side4(self) -> None:
        # Reviewer counterexample: averaging coordinates before differencing used to
        # reward an indecisive policy (uniform 0.114 < canonical one-hot 0.312).
        m = _seeded_tall()
        m.eval()
        m.track_locality = True
        with torch.no_grad():
            m(torch.randn(1, 4, 4, 4))
        ordered = m._last_ordered
        n_cells, K = ordered.shape[0], ordered.shape[1]
        canonical = F.one_hot(torch.zeros(1, n_cells, dtype=torch.long), K).float()
        uniform = torch.full((1, n_cells, K), 1.0 / K)
        one_hot_value = expected_locality_loss(canonical, ordered, m.side)
        uniform_value = expected_locality_loss(uniform, ordered, m.side)
        assert uniform_value >= one_hot_value - 1e-6

    def test_mixing_never_goes_below_best_pure_choice(self) -> None:
        # Multilinear in the cell choices, so the minimum sits at a pure assignment.
        m = _seeded_tall()
        m.eval()
        m.track_locality = True
        with torch.no_grad():
            m(torch.randn(1, 4, 4, 4))
        ordered = m._last_ordered
        n_cells, K = ordered.shape[0], ordered.shape[1]
        pure_idx = torch.cartesian_prod(*[torch.arange(K)] * n_cells)  # (K**n_cells, n_cells)
        pure = F.one_hot(pure_idx, K).float()  # (P, n_cells, K)
        # The function averages over the batch, so evaluate each pure assignment alone.
        best_pure = min(
            float(expected_locality_loss(pure[i : i + 1], ordered, m.side))
            for i in range(pure.shape[0])
        )
        torch.manual_seed(3)
        mixed = torch.distributions.Dirichlet(torch.ones(K)).sample((6, n_cells))
        mixed_vals = torch.stack(
            [expected_locality_loss(mixed[i : i + 1], ordered, m.side) for i in range(6)]
        )
        assert bool((mixed_vals >= best_pure - 1e-6).all())

    def test_soft_locality_backprops_into_policy(self) -> None:
        m = _seeded_tall()
        m.train()
        m.track_locality = True
        m(torch.randn(2, 4, 4, 4), hard=False)
        m.last_locality_loss().backward()
        grads = [p.grad for p in m.policy.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0.0

    def test_locality_weight_changes_policy_gradient(self) -> None:
        x = torch.randn(2, 4, 4, 4)

        def policy_grads(weight: float) -> list[torch.Tensor]:
            m = _seeded_tall()
            m.train()
            m.track_locality = True
            torch.manual_seed(1)  # identical Gumbel noise for both weights
            seq, _ = m(x, hard=False)
            loss = seq.pow(2).mean() + weight * m.last_locality_loss()
            loss.backward()
            return [p.grad.detach().clone() for p in m.policy.parameters()]

        g0 = policy_grads(0.0)
        g1 = policy_grads(1.0)
        assert any(not torch.allclose(a, b) for a, b in zip(g0, g1, strict=True))

    def test_deepcopy_after_tracked_training_forward(self) -> None:
        m = _seeded_tall()
        m.train()
        m.track_locality = True
        m(torch.randn(2, 4, 4, 4), hard=False)
        clone = copy.deepcopy(m)
        assert clone._last_choice is None
        assert clone.track_locality is True

    def test_untracked_forward_keeps_no_graph(self) -> None:
        m = _seeded_tall()
        m.train()
        m(torch.randn(2, 4, 4, 4), hard=False)
        assert m._last_choice is None and m._last_ordered is None
        copy.deepcopy(m)  # must not raise
        with pytest.raises(RuntimeError, match="track_locality"):
            m.last_locality_loss()


class TestMotifPermutations:
    def test_2d_table_shape(self) -> None:
        m = build_motif_permutations(n_dims=2)
        assert m.shape == (4, 4)
        # Each row is a permutation of {0, 1, 2, 3}
        for k in range(4):
            assert torch.equal(torch.sort(m[k]).values, torch.arange(4))

    def test_3d_table_shape(self) -> None:
        m = build_motif_permutations(n_dims=3)
        assert m.shape == (24, 8)
        for k in range(24):
            assert torch.equal(torch.sort(m[k]).values, torch.arange(8))

    def test_invalid_ndims_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            build_motif_permutations(n_dims=4)


class TestTALL3D:
    def test_3d_forward_shape(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)   # 4x4x4 = 64
        m.eval()
        x = torch.randn(1, 4, 4, 4, 4)
        seq, perm = m(x)
        assert seq.shape == (1, 4, 64) and perm.shape == (1, 64)

    def test_3d_permutation_bijection(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)
        m.eval()
        x = torch.randn(1, 4, 4, 4, 4)
        _, perm = m(x)
        for b in range(perm.shape[0]):
            assert torch.equal(torch.sort(perm[b]).values, torch.arange(64))

    def test_3d_train_mode_grad_flow(self) -> None:
        m = TALL(order=2, feature_dim=4, n_dims=3)
        m.train()
        x = torch.randn(1, 4, 4, 4, 4, requires_grad=True)
        seq, _ = m(x, hard=False)
        seq.pow(2).mean().backward()
        assert x.grad is not None and x.grad.abs().max() > 0
        assert m.policy[0].weight.grad is not None


class TestTALL:
    def test_forward_shape(self) -> None:
        m = TALL(order=4, feature_dim=8)   # 16x16 grid
        x = torch.randn(2, 8, 16, 16)
        m.eval()
        seq, perm = m(x)
        assert seq.shape == (2, 8, 256)
        assert perm.shape == (2, 256)

    def test_permutation_is_bijection(self) -> None:
        m = TALL(order=3, feature_dim=4)   # 8x8 = 64
        x = torch.randn(1, 4, 8, 8)
        m.eval()
        _, perm = m(x)
        for b in range(perm.shape[0]):
            sorted_perm = torch.sort(perm[b]).values
            assert torch.equal(sorted_perm, torch.arange(64))

    def test_canonical_init_emits_motif_zero_at_step_0(self) -> None:
        # With canonical_init=True, the initial bias strongly favours
        # motif 0; argmax should pick motif 0 for every cell at step 0.
        m = TALL(order=3, feature_dim=4, canonical_init=True)
        m.eval()
        x = torch.zeros(1, 4, 8, 8)
        # Need to build the cell features manually to avoid randomness
        feat = m._cell_features(x)
        logits = m.policy(feat)
        argmax = logits.argmax(dim=-1)
        assert (argmax == 0).all(), (
            f"canonical_init should put argmax at motif 0; got {argmax}"
        )

    def test_train_mode_returns_differentiable_seq(self) -> None:
        m = TALL(order=3, feature_dim=4)
        m.train()
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        seq, _ = m(x, hard=False)
        loss = seq.pow(2).mean()
        loss.backward()
        # Gradient should flow back to x and to the policy
        assert x.grad is not None and x.grad.abs().max() > 0
        assert m.policy[0].weight.grad is not None

    def test_tau_anneals_after_steps(self) -> None:
        m = TALL(order=3, feature_dim=4, tau_init=1.0, tau_min=0.1, anneal_steps=10)
        m.train()
        # Tau is 1.0 at step 0
        assert abs(m.tau - 1.0) < 1e-6
        x = torch.randn(1, 4, 8, 8)
        for _ in range(10):
            m(x, hard=False)
        # After 10 steps tau should be at the minimum
        assert abs(m.tau - 0.1) < 0.05


class TestLocalityLoss:
    def test_canonical_hilbert_is_low(self) -> None:
        # An identity permutation in raster order has step length 1 mostly,
        # but jumps W at row boundaries → ~W on average.
        # A canonical Hilbert permutation has step length exactly 1.
        from spectramr.models.blocks.hilbert_order import HilbertOrder
        side = 16
        h = HilbertOrder(shape=(side, side), mode="hilbert")
        perm = h.permutation.unsqueeze(0)
        loss_h = float(locality_loss(permutation_coords(perm, side), side))
        r = HilbertOrder(shape=(side, side), mode="raster")
        rperm = r.permutation.unsqueeze(0)
        loss_r = float(locality_loss(permutation_coords(rperm, side), side))
        # Hilbert mean step = 1; raster has occasional row-jumps of ~side
        assert loss_h < loss_r


class TestGradientReversal:
    def test_forward_identity_backward_negate(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            gradient_reversal,
        )
        x = torch.tensor([2.0, 3.0], requires_grad=True)
        y = gradient_reversal(x, lambda_=2.0)
        # Forward is identity
        assert torch.equal(y, x)
        # Backward inverts the sign and scales by lambda
        y.sum().backward()
        assert torch.equal(x.grad, torch.tensor([-2.0, -2.0]))


class TestDomainDiscriminator:
    def test_classifies_features(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            DomainDiscriminator,
        )
        d = DomainDiscriminator(feature_dim=16)
        x = torch.randn(4, 16, 8, 8)
        logits = d(x, lambda_=1.0)
        assert logits.shape == (4,)


class TestStrategyRegistration:
    def test_new_strategies_registered(self) -> None:
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
        factory = TrainingStrategyFactory()
        assert "privileged_learning" in factory.STRATEGY_CLASS_PATHS
        assert "privileged" in factory.STRATEGY_CLASS_PATHS
        assert "slice_to_volume" in factory.STRATEGY_CLASS_PATHS
        assert "2d_to_3d" in factory.STRATEGY_CLASS_PATHS

    def test_strategy_classes_importable(self) -> None:
        from spectramr.infrastructure.training.strategies.privileged_learning_strategy import (
            PrivilegedLearningStrategy,
        )
        from spectramr.infrastructure.training.strategies.slice_to_volume_strategy import (
            SliceToVolumeStrategy,
        )
        assert PrivilegedLearningStrategy is not None
        assert SliceToVolumeStrategy is not None


# ── Sanity-shape parametrization ────────────────────────────────────────


# (order, feature_dim, expected_spatial)  — 2^order × 2^order grid
_TALL_2D_SHAPES = [
    (2, 4, 4),    # 4×4 = 16 tokens
    (3, 8, 8),    # 8×8 = 64 tokens
    (4, 8, 16),   # 16×16 = 256 tokens
]


@pytest.mark.sanity_shape
@pytest.mark.parametrize(
    "order,feat,side",
    _TALL_2D_SHAPES,
    ids=[f"ord{o}feat{f}side{s}" for o, f, s in _TALL_2D_SHAPES],
)
def test_tall_2d_sanity_shape(order: int, feat: int, side: int) -> None:
    """TALL forward returns (B, F, side²) seq and (B, side²) perm in eval mode."""
    m = TALL(order=order, feature_dim=feat)
    m.eval()
    x = torch.randn(1, feat, side, side)
    seq, perm = m(x)
    assert seq.shape == (1, feat, side * side)
    assert perm.shape == (1, side * side)
    assert torch.isfinite(seq).all()
    # Permutation is a bijection of {0, …, N-1}
    assert torch.equal(torch.sort(perm[0]).values, torch.arange(side * side))


def test_tall_wrong_feature_dim_raises() -> None:
    """Passing x with wrong feature_dim produces mismatched shapes (no silent reshape)."""
    m = TALL(order=2, feature_dim=4)  # expects C=4
    m.eval()
    x = torch.randn(1, 8, 4, 4)  # C=8 — wrong
    # The cell-feature MLP will reject the wrong channel count.
    with pytest.raises((RuntimeError, Exception)):
        m(x)
