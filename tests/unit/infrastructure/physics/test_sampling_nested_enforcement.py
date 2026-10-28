"""Tests for nesting enforcement and the physics invariants every family must hold.

Cold diffusion's forward process is ``x_t = M_t * x_0`` with the masks assumed
nested (``M_{t+1} subset-of M_t``): k-space is only ever removed as ``t`` grows.
A bin that reappears is an addition the reverse loop has no mechanism to undo,
so the ``enforce_nested`` flag exists to make the guarantee structural.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.sampling import (
    SUPPORTED_ACCELERATION_TYPES,
    VDCartesian1DAccelerator,
    create_kspace_accelerator,
)

MATRIX = 128
TIMESTEPS = 8
_CPU = torch.device("cpu")

BASE_KWARGS = dict(
    num_timesteps=TIMESTEPS,
    base_acceleration=2.0,
    max_acceleration=16.0,
    center_fraction=0.08,
    min_center_fraction=0.02,
    seed=42,
    acceleration_schedule="linear",
)

#: Families whose cascade already nests, so enforcement is a no-op for them.
NESTING_FAMILIES = [
    "variable_density",
    "variable_density_cava",
    "fractional_variable_density",
    "random_cartesian",
    "nested",
    # Golden-angle prefixes of one fixed ordering (#612); they re-drew per rung before.
    "radial",
    "golden_angle",
    "spiral",
]


def _cascade(accelerator, matrix: int = MATRIX) -> list[torch.Tensor]:
    return [accelerator.get_acceleration_mask((1, matrix, matrix), t)[0] for t in range(TIMESTEPS)]


def _violations(masks: list[torch.Tensor]) -> int:
    return sum(1 for i in range(1, len(masks)) if bool((masks[i] & ~masks[i - 1]).any()))


class TestNestingEnforcement:
    """``enforce_nested`` must make nesting structural, not incidental."""

    @pytest.mark.parametrize("family", NESTING_FAMILIES)
    def test_enforced_cascade_is_nested(self, family: str) -> None:
        acc = create_kspace_accelerator(
            acceleration_type=family, enforce_nested=True, **BASE_KWARGS
        )
        assert _violations(_cascade(acc)) == 0

    @pytest.mark.parametrize("family", NESTING_FAMILIES)
    def test_enforcement_is_a_noop_for_families_that_already_nest(self, family: str) -> None:
        """A family that already nests must come through byte-identical.

        Enforcement intersects the cascade; on an already-monotone cascade the
        intersection is the identity, so this pins that the flag does not perturb
        arms that did not need it.
        """
        raw = _cascade(create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS))
        enforced = _cascade(
            create_kspace_accelerator(acceleration_type=family, enforce_nested=True, **BASE_KWARGS)
        )
        for t, (a, b) in enumerate(zip(raw, enforced, strict=True)):
            assert torch.equal(a, b), f"{family} differs at t={t} under enforcement"

    def test_default_is_off(self) -> None:
        """Every existing run must be unaffected until it opts in."""
        acc = create_kspace_accelerator(acceleration_type="variable_density", **BASE_KWARGS)
        assert acc.enforce_nested is False

    @pytest.mark.parametrize("family", NESTING_FAMILIES)
    def test_strict_tolerance_accepts_a_family_that_nests_exactly(self, family: str) -> None:
        """``nested_tolerance=1.0`` must mean "enforcement is a no-op", and pass.

        The regression: the shortfall guard compared the enforced fraction against
        the CONTINUOUS ``1 / declared_R``. Cartesian families quantise in whole
        k-space lines, so the two can never be equal and the strictest setting was
        unsatisfiable for every family — including the ones that nest perfectly.
        On experiment_11_attention_none this raised over 0.018 of ONE line in 256
        and took both schedule witnesses down on all four ranks mid-run.

        Measured against the family's own raw draw the comparison is exact: an
        already-monotone cascade yields ``enforced == raw`` bit-for-bit, so strict
        equality passes rather than losing to a rounding step.
        """
        acc = create_kspace_accelerator(
            acceleration_type=family,
            enforce_nested=True,
            nested_tolerance=1.0,
            **BASE_KWARGS,
        )
        assert _violations(_cascade(acc)) == 0

    @pytest.mark.parametrize("family", NESTING_FAMILIES)
    def test_exact_nester_warns_about_nothing(
        self, family: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A no-op enforcement must be silent — warnings are not OK (#10).

        The old comparison also drove the in-band WARNING branch, which emitted
        self-contradicting lines ("realises R=2.23 ... but declares R=2.23") once
        per timestep. Nothing was wrong, so nothing should be said.
        """
        caplog.set_level("WARNING", logger="spectramr.infrastructure.physics.sampling")
        acc = create_kspace_accelerator(
            acceleration_type=family,
            enforce_nested=True,
            nested_tolerance=1.0,
            **BASE_KWARGS,
        )
        _cascade(acc)
        # ``.getMessage()``, not ``.message``: the latter only exists once a
        # handler has formatted the record, which makes it order-dependent.
        offenders = [r.getMessage() for r in caplog.records if "enforce_nested" in r.getMessage()]
        assert offenders == [], f"{family} warned despite enforcement being a no-op: {offenders}"

    def test_tolerance_measures_enforcement_cost_not_declared_drift(self) -> None:
        """The guard's denominator is the raw draw, not ``1 / declared_R``.

        Pins the two halves apart. A family whose raw draw sits slightly off its
        declared R is not this guard's business — that is
        ``declared_ladder_defects``. This guard only asks what the cumulative
        intersection deleted, so an exactly-nesting family scores a perfect 1.0
        however its raw draw rounds against the declared ladder.
        """
        acc = create_kspace_accelerator(
            acceleration_type="variable_density",
            enforce_nested=True,
            nested_tolerance=1.0,
            **BASE_KWARGS,
        )
        raw = create_kspace_accelerator(acceleration_type="variable_density", **BASE_KWARGS)
        for t in range(TIMESTEPS):
            enforced_frac = float(acc.get_acceleration_mask((1, MATRIX, MATRIX), t)[0].float().mean())
            raw_frac = float(raw.get_acceleration_mask((1, MATRIX, MATRIX), t)[0].float().mean())
            assert enforced_frac == pytest.approx(raw_frac), f"enforcement cost bins at t={t}"
            # And the declared target genuinely does NOT coincide — which is
            # exactly why it cannot be the denominator.
            declared_target = 1.0 / max(1.0, acc.get_acceleration_factor(t))
            assert enforced_frac >= declared_target * 0.9

    @pytest.mark.parametrize("family", ["multi_mask"])
    def test_redrawing_families_still_raise_at_the_default_tolerance(self, family: str) -> None:
        """The fix must not defang the guard it repairs.

        ``multi_mask`` re-draws its pattern per timestep, so the intersection strips
        most of k-space — a genuine collapse, which must still fail loudly at the 0.5
        default rather than train on a gutted cascade (pitfall #9). Radial and spiral
        left this list when they became prefixes of one ordering (#612).
        """
        acc = create_kspace_accelerator(
            acceleration_type=family, enforce_nested=True, **BASE_KWARGS
        )
        with pytest.raises(ValueError, match=f"collapsed the '{family}' cascade"):
            _cascade(acc)

    def test_tolerance_is_validated(self) -> None:
        for bad in (0.0, -0.1, 1.5):
            with pytest.raises(ValueError, match="nested_tolerance"):
                create_kspace_accelerator(
                    acceleration_type="variable_density",
                    enforce_nested=True,
                    nested_tolerance=bad,
                    **BASE_KWARGS,
                )

    def test_cache_is_keyed_by_shape(self) -> None:
        """A second matrix size must build its own cascade, not reuse the first."""
        acc = create_kspace_accelerator(
            acceleration_type="variable_density", enforce_nested=True, **BASE_KWARGS
        )
        small = acc.get_acceleration_mask((1, 64, 64), 3)
        large = acc.get_acceleration_mask((1, MATRIX, MATRIX), 3)
        assert small.shape[-2:] == (64, 64)
        assert large.shape[-2:] == (MATRIX, MATRIX)
        assert _violations(_cascade(acc, matrix=64)) == 0


class TestCartesianLineStructure:
    """A 2-D Cartesian readout is fully sampled: masks must drop whole lines."""

    def test_variable_density_1d_keeps_whole_lines(self) -> None:
        """Regression for issue #947.

        The class built its mask from columns while declaring ``line_axis='y'``,
        so budget enforcement trimmed a ROW and punched one hole through every
        sampled line. The sampled fraction barely moved, which is why no ladder
        or nesting check caught it — only line purity does.
        """
        acc = VDCartesian1DAccelerator(**BASE_KWARGS)
        mask = acc.get_acceleration_mask((1, MATRIX, MATRIX), TIMESTEPS // 2)[0]
        occupancy = mask.sum(dim=1)
        pure = ((occupancy == 0) | (occupancy == MATRIX)).sum()
        assert int(pure) == MATRIX, (
            f"{MATRIX - int(pure)} phase-encode lines are partially sampled; "
            "a Cartesian sequence cannot acquire that."
        )

    def test_variable_density_1d_samples_dc(self) -> None:
        acc = VDCartesian1DAccelerator(**BASE_KWARGS)
        for t in range(TIMESTEPS):
            mask = acc.get_acceleration_mask((1, MATRIX, MATRIX), t)[0]
            assert bool(mask[MATRIX // 2, MATRIX // 2]), f"DC dropped at t={t}"


class TestPhysicsInvariantsAcrossFamilies:
    """Invariants that hold for every registered family, enforced or not."""

    @pytest.mark.parametrize("family", sorted(SUPPORTED_ACCELERATION_TYPES))
    def test_dc_is_always_sampled(self, family: str) -> None:
        """The k-space centre carries the phase reference and the ACS.

        Without it there is no calibration region for coil-sensitivity estimation
        and no DC term, so a mask that drops it is not a physically meaningful
        acquisition at any acceleration.
        """
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        for t in range(TIMESTEPS):
            mask = acc.get_acceleration_mask((1, MATRIX, MATRIX), t)[0]
            assert bool(mask[MATRIX // 2, MATRIX // 2]), (
                f"{family} drops the k-space centre at t={t}"
            )

    @pytest.mark.parametrize("family", sorted(SUPPORTED_ACCELERATION_TYPES))
    def test_masks_are_deterministic(self, family: str) -> None:
        """Same seed, same timestep, same mask — twice."""
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        first = acc.get_acceleration_mask((1, MATRIX, MATRIX), 3)
        second = acc.get_acceleration_mask((1, MATRIX, MATRIX), 3)
        assert torch.equal(first, second)

    @pytest.mark.parametrize("family", sorted(SUPPORTED_ACCELERATION_TYPES))
    def test_mask_is_never_empty(self, family: str) -> None:
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        for t in range(TIMESTEPS):
            assert bool(acc.get_acceleration_mask((1, MATRIX, MATRIX), t)[0].any()), (
                f"{family} produced an empty mask at t={t}"
            )


#: Families whose mask comes from ``MaskGenerator._rasterize_trajectory``, which
#: mirrors the rank of the shape it is handed. They are the ones that used to break
#: the return-rank contract.
TRAJECTORY_FAMILIES = ["radial", "spiral", "golden_angle"]


class TestMaskRankContract:
    """``get_acceleration_mask`` returns rank 3 whatever the rank of the input.

    The rank was never stated, so the two family groups disagreed: Cartesian
    families allocated ``(channels, H, W)`` unconditionally while the trajectory
    families returned ``_rasterize_trajectory`` verbatim, which drops to ``(H, W)``
    for a 2-tuple. ``ColdDiffusionAccelerator._first_drop_map`` then took ``[0]`` of
    the result — selecting k-space ROW 0 and broadcasting it, which turned a declared
    R=7.8 golden-angle mask into R=128 with every row identical.
    """

    @pytest.mark.parametrize("family", sorted(SUPPORTED_ACCELERATION_TYPES))
    def test_two_tuple_shape_still_returns_rank_three(self, family: str) -> None:
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        mask = acc.get_acceleration_mask((MATRIX, MATRIX), 3)
        assert mask.dim() == 3, f"{family} returned rank {mask.dim()} for a 2-tuple"
        assert mask.shape == (1, MATRIX, MATRIX)

    @pytest.mark.parametrize("family", sorted(SUPPORTED_ACCELERATION_TYPES))
    def test_two_tuple_and_three_tuple_agree(self, family: str) -> None:
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        assert torch.equal(
            acc.get_acceleration_mask((MATRIX, MATRIX), 3),
            acc.get_acceleration_mask((1, MATRIX, MATRIX), 3),
        )

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_enforced_trajectory_mask_is_not_a_broadcast_row(self, family: str) -> None:
        """The exact regression: enforcement + a trajectory family + a 2-tuple.

        A single broadcast row is the tell — it is what ``[0]`` of a rank-2 mask
        produces, and it collapses the realised budget by more than an order of
        magnitude while every declared value stays untouched.
        """
        acc = create_kspace_accelerator(
            acceleration_type=family,
            enforce_nested=True,
            nested_tolerance=0.01,
            **BASE_KWARGS,
        )
        mask = acc.get_acceleration_mask((MATRIX, MATRIX), 3)[0]
        assert not bool((mask == mask[0]).all()), (
            f"{family} produced an identical row everywhere — the rank-2 mask was "
            f"indexed as if row 0 were a channel"
        )
        assert torch.equal(mask, acc.get_acceleration_mask((1, MATRIX, MATRIX), 3)[0])

    def test_first_drop_map_rejects_a_rank_violating_family(self) -> None:
        """A family that forgets to normalise is caught, not silently broadcast."""
        acc = create_kspace_accelerator(
            acceleration_type="golden_angle", enforce_nested=True, **BASE_KWARGS
        )
        inner = acc.accelerator
        original = inner.get_acceleration_mask

        def _rank_two(shape, t, device=_CPU):
            return original(shape, t, device=device)[0]

        inner.get_acceleration_mask = _rank_two
        acc._nested_cache.clear()
        with pytest.raises(ValueError, match=r"must return \(channels, height, width\)"):
            acc.get_acceleration_mask((MATRIX, MATRIX), 3)


class TestTrajectoryFamiliesHonourACS:
    """``center_fraction``/``min_center_fraction`` are read, not just stored.

    Every trajectory family assigned ``self.center_fraction`` in ``__init__`` and
    then never read it, so an arm declaring an 8% calibration region trained on
    whatever band the spokes happened to cross. Cold diffusion needs the ACS to
    survive to the HIGHEST acceleration, which is what ``min_center_fraction``
    declares — hence ``_guaranteed_core_fraction``, not ``_current_center_fraction``.
    """

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_center_floor_changes_the_mask(self, family: str) -> None:
        kwargs = dict(BASE_KWARGS)
        kwargs.pop("min_center_fraction")
        narrow = create_kspace_accelerator(
            acceleration_type=family, min_center_fraction=0.01, **kwargs
        )
        wide = create_kspace_accelerator(
            acceleration_type=family, min_center_fraction=0.05, **kwargs
        )
        differing = sum(
            int(
                (
                    narrow.get_acceleration_mask((1, MATRIX, MATRIX), t)
                    ^ wide.get_acceleration_mask((1, MATRIX, MATRIX), t)
                ).sum()
            )
            for t in range(TIMESTEPS)
        )
        assert differing > 0, f"{family} ignores min_center_fraction"

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_core_survives_the_highest_rung(self, family: str) -> None:
        kwargs = dict(BASE_KWARGS)
        kwargs.pop("min_center_fraction")
        acc = create_kspace_accelerator(
            acceleration_type=family, min_center_fraction=0.04, **kwargs
        )
        mask = acc.get_acceleration_mask((1, MATRIX, MATRIX), TIMESTEPS - 1)[0]
        half = int(MATRIX * 0.04**0.5) // 2
        lo, hi = MATRIX // 2 - half, MATRIX // 2 + half
        assert bool(mask[lo:hi, lo:hi].all()), (
            f"{family} leaves holes in the declared ACS core at the top rung"
        )

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_unreachable_center_floor_raises(self, family: str) -> None:
        """The guard the Cartesian families always had, now shared.

        ``min_center_fraction: 0.08`` at ``max_acceleration: 32`` asks for an ACS
        2.6x the entire budget. ``random_cartesian`` refused it; the trajectory
        families accepted it and discarded it.
        """
        kwargs = dict(BASE_KWARGS)
        kwargs.pop("min_center_fraction")
        kwargs.pop("max_acceleration")
        with pytest.raises(ValueError, match="exceeds the sampling budget"):
            create_kspace_accelerator(
                acceleration_type=family,
                max_acceleration=32.0,
                min_center_fraction=0.08,
                **kwargs,
            )


class TestTrajectoryLadderRungs:
    """The identity rung and the coverage ceiling of the trajectory families (#612)."""

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_identity_rung_is_fully_sampled(self, family: str) -> None:
        """R=1 is every bin, not the sparsest mask a failed search fell back to."""
        kwargs = {**BASE_KWARGS, "base_acceleration": 1.0}
        acc = create_kspace_accelerator(acceleration_type=family, **kwargs)
        assert bool(acc.get_acceleration_mask((1, 48, 80), 0).all())

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_rung_above_the_coverage_ceiling_raises_at_build(self, family: str) -> None:
        """Planted: a 1.2x rung asks for 83% of k-space, more than any trajectory covers."""
        kwargs = {**BASE_KWARGS, "base_acceleration": 1.2}
        with pytest.raises(ValueError, match="covers at most"):
            create_kspace_accelerator(acceleration_type=family, **kwargs)

    @pytest.mark.parametrize("family", TRAJECTORY_FAMILIES)
    def test_cascade_nests_on_a_non_square_grid(self, family: str) -> None:
        acc = create_kspace_accelerator(acceleration_type=family, **BASE_KWARGS)
        masks = [acc.get_acceleration_mask((1, 48, 80), t)[0] for t in range(TIMESTEPS)]
        assert _violations(masks) == 0
