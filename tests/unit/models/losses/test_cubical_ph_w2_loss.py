"""Unit tests for CubicalPHWassersteinLoss.

Skipped in environments without `gudhi` and `POT` (the optional
``[topology]`` extra). When those deps are available the tests pin:

- Loss is a non-negative finite scalar.
- Loss is autograd-traceable (the existing Euler-characteristic
  approximation in topological_loss.py was a no-op for backprop —
  this loss must not regress to that bug).
- Identical (pred, target) gives a near-zero loss.
- Stability (Cohen-Steiner): loss is bounded above by L2.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
gudhi = pytest.importorskip("gudhi")
ot = pytest.importorskip("ot")

from spectramr.models.losses.cubical_ph_w2_loss import (  # noqa: E402
    CubicalPHWassersteinLoss,
)


def test_returns_finite_scalar() -> None:
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,))(pred, target)
    assert loss.shape == torch.Size([])
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


def test_matching_uses_linfinity_ground_metric() -> None:
    """Per-pair cost is ``max(|Δbirth|, |Δdeath|)^p`` (L∞), not the L₁ sum.

    The L∞ ground metric is the one under which the Cohen-Steiner stability
    theorem for persistence-diagram W_p holds. We pin a deterministic transport
    plan via a stub ``ot`` module so the assertion does not depend on the EMD
    solver, and check the returned matching cost equals the L∞ value.
    """
    import numpy as np

    from spectramr.models.losses.cubical_ph_w2_loss import _wasserstein_matching

    class _DiagonalOT:
        """Stub whose ``emd`` returns the diagonal (identity) transport plan."""

        @staticmethod
        def emd(wa, wb, cost):  # noqa: ANN001
            return np.diag(wa)

    a = np.array([[0.0, 1.0]])  # birth 0, death 1
    b = np.array([[0.2, 1.3]])  # Δbirth 0.2, Δdeath 0.3

    cost, _pairs = _wasserstein_matching(a, b, p=2, ot_module=_DiagonalOT())

    # Plan = diag: the genuine pair costs max(0.2, 0.3)^2 = 0.09; the
    # diagonal-to-diagonal pair costs 0 (Hera reduction). The cost is the sum of
    # p-th powers over the chosen permutation, with no 1/n normalisation.
    assert cost == pytest.approx(0.09, abs=1e-6)


def test_diagonal_block_zeroed_routes_genuine_match() -> None:
    """A pred feature near a target feature must match it (genuine), not the
    diagonal. Pre-fix the diagonal↔diagonal cost block was not zeroed, so its
    spurious penalty inflated the genuine option and the solver dropped the
    pred point to the diagonal (~9% of random diagrams) — flipping the gradient
    from *pull-toward-target* to *shrink-persistence* (F2)."""
    import numpy as np

    from spectramr.models.losses.cubical_ph_w2_loss import _wasserstein_matching

    a = np.array([[0.423, 0.461]])  # pred point
    b = np.array([[0.124, 0.798]])  # a genuine (if imperfect) target match
    _cost, pairs = _wasserstein_matching(a, b, p=2, ot_module=ot)
    # n_b == 1, so j == 0 is the genuine target and j >= 1 is the diagonal.
    assert pairs == [(0, 0)]  # genuine; was (0, 1) [diagonal] pre-fix


def test_identity_backward_is_finite() -> None:
    """pred == target ⇒ loss == 0 with a FINITE gradient. ``loss.sqrt()`` has an
    infinite slope at 0, so a perfect slice produced a NaN gradient (∞·0). The
    identity test only checked ``.item()`` and never back-propagated, so this was
    latent (F5)."""
    target = torch.rand(1, 1, 8, 8)
    pred = target.clone().detach().requires_grad_(True)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,))(pred, target)
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_tie_localization_uses_generating_cell() -> None:
    """The death gradient must land on the actual generating (saddle) cell, not
    the corner voxel. With a flat background (value ties), locating the death
    voxel by ``argmin(|flat - death|)`` collapsed to flat index 0 = corner
    ``[0, 0]``; gudhi's ``cofaces_of_persistence_pairs`` returns the true cell
    (F6). Here the pred's extra well is hallucinated (target lacks it), so it
    diagonal-matches through its birth+death cells — neither of which is the
    corner."""
    target = _one_well()
    pred = _two_wells(0.6).clone().requires_grad_(True)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=2)(
        pred, target
    )
    loss.backward()
    assert pred.grad is not None
    # The extra well at [6, 6] receives gradient; the corner [0, 0] does not
    # (pre-fix the death gradient was wrongly scattered to the corner).
    assert pred.grad[0, 0, 6, 6] != 0.0
    assert pred.grad[0, 0, 0, 0] == 0.0


def test_loss_is_autograd_traceable() -> None:
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,))(pred, target)
    loss.backward()
    assert pred.grad is not None
    # Gradient sparsity is fine (only birth voxels contribute), but
    # at least one entry must be non-zero in expectation.
    assert torch.any(pred.grad != 0)


def test_identity_loss_is_near_zero() -> None:
    target = torch.rand(1, 1, 8, 8)
    pred = target.clone().detach().requires_grad_(True)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,))(pred, target)
    # Diagrams should match exactly so loss = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def _one_well() -> "torch.Tensor":
    x = torch.ones(1, 1, 8, 8)
    x[0, 0, 1, 1] = 0.0  # single deep well -> one H0 feature
    return x


def _two_wells(second_depth: float) -> "torch.Tensor":
    x = _one_well()
    x[0, 0, 6, 6] = 1.0 - second_depth  # hallucinated extra well
    return x


def test_hallucinated_feature_receives_gradient() -> None:
    """A pred topological feature the target lacks matches the diagonal and must
    contribute a NON-zero, differentiable cost pushing it flat. Pre-fix the
    differentiable path dropped diagonal-matched pred points, so the extra well
    was invisible: loss == 0 and no gradient regardless of its depth (F2)."""
    target = _one_well()
    pred = _two_wells(0.6).clone().requires_grad_(True)
    loss = CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=2)(
        pred, target
    )
    assert loss.requires_grad
    assert loss.item() > 1e-4
    loss.backward()
    assert torch.any(pred.grad != 0)


def test_hallucinated_loss_grows_with_persistence() -> None:
    """A deeper extra well (larger persistence) must cost more than a shallow
    one — the diagonal-projection term scales with the feature's persistence."""
    target = _one_well()
    make = lambda d: CubicalPHWassersteinLoss(  # noqa: E731
        homology_dims=(0,), wasserstein_p=2
    )(_two_wells(d), target).item()
    assert make(0.8) > make(0.3) > 0.0


def test_mask_zeros_out_background_features() -> None:
    """Voxels outside the mask must not generate persistence pairs."""
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = pred.detach().clone()  # identical => only mask matters
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., 2:6, 2:6] = 1.0  # 4x4 foreground patch

    loss = CubicalPHWassersteinLoss(homology_dims=(0,))(pred, target, mask=mask)
    # pred == target inside mask, and outside-mask voxels are
    # filtered out of the filtration → loss = 0.
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_stability_bound() -> None:
    """W_p(Dgm(p), Dgm(t)) <= 3^(d/p) ||p - t||_2 for voxel images (Skraba-Turner)."""
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8)
    ph_loss = CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=2)(
        pred, target
    )
    l2_bound = (pred - target).pow(2).sum().sqrt()
    # Allow a small numerical slack — the loss is computed on a
    # detached numpy view and re-attached via birth-voxel scatter.
    # Bound with the voxel constant 3^(d/p) from Skraba-Turner, d=2, p=2.
    assert ph_loss.item() <= 3.0 * l2_bound.item() + 1e-3


@pytest.mark.parametrize("shape", [(7, 11), (12, 12), (4, 5, 6)])
def test_diagram_matches_gudhi_on_non_symmetric_input(shape: tuple[int, ...]) -> None:
    """gudhi numbers cells in Fortran order; reading them in C order transposes
    every voxel, which the symmetric fixtures above cannot see."""
    import numpy as np

    vol = np.random.default_rng(0).random(shape)
    cc = gudhi.CubicalComplex(top_dimensional_cells=vol)
    cc.compute_persistence()
    expected = sorted(
        (b, d) for dim, (b, d) in cc.persistence() if dim in (0, 1) and np.isfinite(d)
    )
    out = CubicalPHWassersteinLoss(homology_dims=(0, 1))._persistence_with_voxels(vol)
    for dim in (0, 1):
        dgm, births, deaths = out[dim]
        expected = sorted((b, d) for q, (b, d) in cc.persistence() if q == dim and np.isfinite(d))
        assert np.allclose(sorted(map(tuple, dgm)), expected)
        assert np.array_equal(vol.reshape(-1)[births], dgm[:, 0])
        assert np.array_equal(vol.reshape(-1)[deaths], dgm[:, 1])


def test_off_diagonal_well_gradient_lands_on_the_well() -> None:
    """A hallucinated well at [1, 6] is pushed flat at [1, 6], not at its transpose."""
    target = _one_well()
    pred = _one_well()
    pred[0, 0, 1, 6] = 0.3
    pred.requires_grad_(True)
    CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=2)(pred, target).backward()
    assert pred.grad is not None
    assert pred.grad[0, 0, 1, 6] != 0.0
    assert pred.grad[0, 0, 6, 1] == 0.0


def _dots(points, n: int = 16) -> torch.Tensor:
    img = torch.ones(1, 1, n, n)
    for y, x in points:
        img[0, 0, y, x] = 0.0
    return img


def test_erased_target_feature_raises_loss() -> None:
    """A prediction missing target features must score above zero (#2374)."""
    dots = [(2, 2), (2, 12), (12, 2), (12, 12)]
    target = _dots(dots)
    loss_fn = CubicalPHWassersteinLoss(homology_dims=(0,))
    assert loss_fn(target.clone(), target).item() == pytest.approx(0.0, abs=1e-6)
    assert loss_fn(_dots(dots[:3]), target).item() > 0.0


def test_h0_point_never_matches_h1_point() -> None:
    """A loop (H1) against two dots (H0) is not a free match across dimensions."""
    ring = torch.ones(1, 1, 16, 16)
    ring[0, 0, 4:12, 4] = 0.0
    ring[0, 0, 4:12, 11] = 0.0
    ring[0, 0, 4, 4:12] = 0.0
    ring[0, 0, 11, 4:12] = 0.0
    dots = _dots([(2, 2), (13, 13)])
    loss_fn = CubicalPHWassersteinLoss(homology_dims=(0, 1))
    assert loss_fn(ring, dots).item() > 0.0


def test_wasserstein_p_reaches_the_loss() -> None:
    torch.manual_seed(0)
    pred = torch.rand(1, 1, 8, 8, requires_grad=True)
    target = torch.rand(1, 1, 8, 8)
    p1 = CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=1)(pred, target)
    p2 = CubicalPHWassersteinLoss(homology_dims=(0,), wasserstein_p=2)(pred, target)
    assert p1.item() != pytest.approx(p2.item())


def test_stratify_subvolumes_above_one_is_refused() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        CubicalPHWassersteinLoss(homology_dims=(0,), stratify_subvolumes=2)
