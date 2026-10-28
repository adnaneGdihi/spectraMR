"""``SobolevKSpaceLoss`` must expose the order its own docstring advertises.

The class documents ``(1 + k_x^2 + k_y^2)^{s/2}`` and hardcoded the ``s=2`` case
as the literal ``1.0 + xx**2 + yy**2``. 56 exp_11 arms declared ``sobolev_order:
1`` in a block ``extra="ignore"`` discarded, so the order was dead at every
layer and nobody could tell (#560, #615).

Note ``sobolev_frequency`` is a *separate* registered loss whose weight is
``(1 + |k|)`` -- the two are two fixed points of one family, which is why the
arms' declared intent was expressible all along, just not that way.
"""

import pytest
import torch

from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss, SobolevKSpaceLoss


def _expected_weight(size: int, order: float) -> torch.Tensor:
    y = (torch.arange(size, dtype=torch.float32) - size / 2.0) / (size / 2.0)
    yy, xx = torch.meshgrid(y, y, indexing="ij")
    return ((1.0 + xx**2 + yy**2) ** (order / 2.0)).unsqueeze(0).unsqueeze(0)


def test_default_order_reproduces_the_historical_hardcoded_weight():
    """s=2 is what the body computed as `1.0 + xx**2 + yy**2`.

    The default must not move, or every arm already using this loss silently
    changes objective.
    """
    loss = SobolevKSpaceLoss()
    assert loss.order == 2.0
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, _expected_weight(8, 2.0))


def test_order_one_gives_the_square_root_weighting():
    loss = SobolevKSpaceLoss(order=1.0)
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, _expected_weight(8, 1.0))


def test_order_zero_is_the_unweighted_l1_consistency_check():
    """At s=0 the weight map is all ones, so the loss degenerates to plain L1.

    Probing at the parameter value where both sides must agree, rather than at
    the operating point.
    """
    loss = SobolevKSpaceLoss(order=0.0)
    w = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(w, torch.ones_like(w))


def test_order_changes_the_weight_map_so_it_is_not_a_dead_knob():
    a = SobolevKSpaceLoss(order=2.0)._get_weight_map(
        8, 8, torch.device("cpu"), torch.float32
    )
    b = SobolevKSpaceLoss(order=1.0)._get_weight_map(
        8, 8, torch.device("cpu"), torch.float32
    )
    assert not torch.allclose(a, b)


def test_shape_only_cache_key_is_sufficient_because_order_is_per_instance():
    """The cache key is ``(height, width)``, which no longer uniquely determines
    the map now that ``order`` varies -- it is safe only because ``order`` is
    fixed per instance. Pin that, so making ``order`` mutable fails here rather
    than silently serving a stale map.
    """
    loss = SobolevKSpaceLoss(order=1.0)
    first = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    assert set(loss._weight_map_cache) == {(8, 8)}

    loss.order = 2.0  # not supported; the cache must not silently honour it
    second = loss._get_weight_map(8, 8, torch.device("cpu"), torch.float32)
    torch.testing.assert_close(first, second)


def test_order_is_reachable_through_the_registry_kwargs_path():
    """The whole point: `kwargs: {order: 1.0}` must now reach the constructor."""
    from spectramr.models.losses.registry import LossRegistry

    loss = LossRegistry.create("sobolev_kspace", order=1.0)
    assert loss.order == 1.0


def test_forward_still_runs_and_differs_by_order():
    pred = torch.randn(1, 2, 8, 8)
    target = torch.randn(1, 2, 8, 8)
    lo = SobolevKSpaceLoss(order=0.0)(pred, target)
    hi = SobolevKSpaceLoss(order=4.0)(pred, target)
    assert torch.isfinite(lo) and torch.isfinite(hi)
    assert not torch.isclose(lo, hi), "order must change the computed loss"


class TestFrequencyWeightedL1OnAcquiredLines:
    """Never-acquired lines are exact zeros in the target; they carried ~32 % of this map."""

    def _plane(self, b=2, h=16, w=16, lo=3, hi=13):
        plane = torch.zeros(b, 1, h, w)
        plane[..., lo:hi] = 1.0
        return plane

    def test_padded_lines_carry_no_weight(self) -> None:
        from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss

        loss = FrequencyWeightedL1Loss(alpha=16.0)
        target = torch.randn(2, 4, 16, 16)
        plane = self._plane()
        pred = target + 5.0 * (1 - plane)  # error only on the padded lines
        assert float(loss(pred, target, acquired=plane)) == 0.0
        # Planted: without the plane the padded error is scored.
        assert float(loss(pred, target)) > 0.0

    def test_the_mean_runs_over_the_acquired_bins(self) -> None:
        from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss

        loss = FrequencyWeightedL1Loss(alpha=0.0)  # flat map: weight 1 everywhere
        target = torch.zeros(2, 4, 16, 16)
        plane = self._plane()
        pred = target + plane  # unit error on every acquired bin
        assert float(loss(pred, target, acquired=plane)) == pytest.approx(1.0)

    def test_a_depth_volume_folds_the_plane_with_it(self) -> None:
        from spectramr.models.losses.kspace_physics_losses import FrequencyWeightedL1Loss

        loss = FrequencyWeightedL1Loss(alpha=0.0)
        target = torch.zeros(2, 4, 3, 16, 16)
        plane = self._plane()
        pred = target + plane[:, :, None]
        assert float(loss(pred, target, acquired=plane)) == pytest.approx(1.0)


def _fw_inputs(seed: int = 1234):
    """Same draw order as the pre-change golden capture (spectral pair first, then k-space)."""
    g = torch.Generator().manual_seed(seed)
    torch.rand(2, 3, 64, 64, generator=g)
    torch.rand(2, 3, 64, 64, generator=g)
    kp = torch.randn(2, 4, 64, 64, generator=g)
    kt = torch.randn(2, 4, 64, 64, generator=g)
    acq = (torch.rand(2, 1, 64, 64, generator=g) > 0.3).float()
    return kp, kt, acq


def test_linear_profile_is_bit_identical_to_the_pre_change_loss():
    """Golden values captured on the pre-change code; the default profile must not move."""
    kp, kt, acq = _fw_inputs()
    loss = FrequencyWeightedL1Loss(64, 64, alpha=8.0)
    assert loss(kp, kt).item() == float.fromhex("0x1.21b9c20000000p+0")
    assert loss(kp, kt, acquired=acq).item() == float.fromhex("0x1.2139a20000000p+0")
    cp = torch.complex(kp[:, :2], kp[:, 2:])
    ct = torch.complex(kt[:, :2], kt[:, 2:])
    got = FrequencyWeightedL1Loss(64, 64, alpha=8.0)(cp, ct, acquired=acq).item()
    assert got == float.fromhex("0x1.c699fc0000000p+0")


def test_explicit_linear_profile_equals_the_default():
    kp, kt, _ = _fw_inputs()
    default = FrequencyWeightedL1Loss(64, 64, alpha=8.0)(kp, kt)
    explicit = FrequencyWeightedL1Loss(64, 64, alpha=8.0, profile="linear")(kp, kt)
    assert torch.equal(default, explicit)


def _ring_masks(size: int, edges=(1 / 8, 1 / 4, 1 / 2, 3 / 4)):
    y = torch.arange(size, dtype=torch.float64) - size // 2
    rho = torch.sqrt(y[:, None] ** 2 + y[None, :] ** 2) / (size / 2)
    ring = torch.bucketize(rho, torch.tensor(edges, dtype=torch.float64), right=True) - 1
    n_rings = len(edges) - 1
    return [ring == r for r in range(n_rings)], (ring < 0) | (ring >= n_rings)


def test_ring_band_gives_each_ring_equal_total_weight_and_unit_mean():
    loss = FrequencyWeightedL1Loss(64, 64, profile="ring_band")
    w = loss._get_weight_map(64, 64, torch.device("cpu"), torch.float32)[0, 0].double()
    assert w.mean().item() == pytest.approx(1.0, rel=1e-5)
    rings, outside = _ring_masks(64)
    totals = [w[m].sum().item() for m in rings] + [w[outside].sum().item()]
    assert all(t == pytest.approx(totals[0], rel=1e-5) for t in totals)


def test_ring_band_total_follows_the_declared_gains():
    loss = FrequencyWeightedL1Loss(
        64, 64, profile="ring_band", ring_gain=[2.0, 1.0, 1.0], ring_gain_outside=0.5
    )
    w = loss._get_weight_map(64, 64, torch.device("cpu"), torch.float32)[0, 0].double()
    rings, outside = _ring_masks(64)
    totals = [w[m].sum().item() for m in rings] + [w[outside].sum().item()]
    assert totals[0] / totals[1] == pytest.approx(2.0, rel=1e-5)
    assert totals[3] / totals[1] == pytest.approx(0.5, rel=1e-5)


def test_ring_band_keeps_the_low_band_above_zero():
    """The DC bin sits in the outside group, which must keep a non-zero weight."""
    loss = FrequencyWeightedL1Loss(64, 64, profile="ring_band")
    w = loss._get_weight_map(64, 64, torch.device("cpu"), torch.float32)[0, 0]
    assert w[32, 32].item() > 0.0
    assert torch.isfinite(w).all()


def test_ring_band_forward_respects_the_acquired_mask():
    kp, kt, acq = _fw_inputs()
    loss = FrequencyWeightedL1Loss(64, 64, profile="ring_band")
    value = loss(kp, kt, acquired=acq)
    assert torch.isfinite(value)
    assert value.item() > 0.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"profile": "log_bands"}, "Unknown profile"),
        ({"profile": "ring_band", "ring_edges": [0.5, 0.25]}, "strictly increasing"),
        ({"profile": "ring_band", "ring_gain": [1.0, 1.0]}, "one value per ring"),
        ({"profile": "ring_band", "ring_gain": [1.0, -1.0, 1.0]}, ">= 0"),
        (
            {"profile": "ring_band", "ring_gain": [0.0, 0.0, 0.0], "ring_gain_outside": 0.0},
            "positive",
        ),
        ({"profile": "linear", "ring_gain": [1.0, 1.0, 1.0]}, "apply only to profile"),
    ],
)
def test_invalid_ring_configs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FrequencyWeightedL1Loss(64, 64, **kwargs)


def test_profile_is_reachable_through_the_registry_kwargs_path():
    from spectramr.models.losses.registry import LossRegistry

    loss = LossRegistry.create("frequency_weighted_l1_kspace", profile="ring_band", alpha=8.0)
    assert loss.profile == "ring_band"
    assert loss.ring_edges == (1 / 8, 1 / 4, 1 / 2, 3 / 4)
