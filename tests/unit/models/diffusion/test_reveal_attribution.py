"""Reveal-step band attribution, with a planted violation per claim.

``reveal_attribution`` claims four things, and each is asserted here against an
input constructed to break exactly that one (non-negotiable 15), so a partial
revert cannot pass:

1. a mask that is not a group of complete lines is REFUSED, not approximated —
   planted as a point pattern and as a radial trajectory;
2. the reveal partition is disjoint, exhaustive over the unobserved support, and
   keeps an inert step as an empty band rather than dropping it;
3. the gain's MODULUS reports amplitude shrinkage — planted as a scaled band;
4. the gain's ARGUMENT reports phase error — planted as a rotated band, which a
   magnitude-only or energy-only statistic cannot see. This is the error mode
   the reverse loop's ceiling is phase-invariant against, so a check that misses
   it would be green on the defect it exists to find.
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.diffusion.kspace_process import paired_complex, paired_magnitude
from spectramr.models.diffusion.reveal_attribution import (
    attribute_reveal_bands,
    complex_gain,
    line_indices,
    resolve_line_axis,
    reveal_partition,
)

pytestmark = pytest.mark.unit

H = W = 16


def _line_mask(rows: list[int]) -> torch.Tensor:
    """A ``[1, 1, H, W]`` mask keeping whole k_y lines (rows)."""
    m = torch.zeros(1, 1, H, W)
    for r in rows:
        m[0, 0, r, :] = 1.0
    return m


def _interleaved(value: torch.Tensor) -> torch.Tensor:
    """Lift a complex ``[1, n, H, W]`` to the interleaved real-stacked layout."""
    out = torch.zeros(1, 2 * value.shape[1], H, W)
    out[:, 0::2] = value.real
    out[:, 1::2] = value.imag
    return out


# --------------------------------------------------------------------------
# Claim 1 — non-line geometry is refused
# --------------------------------------------------------------------------


def test_full_line_mask_resolves_to_the_phase_encode_axis() -> None:
    assert resolve_line_axis(_line_mask([0, 3, 9])) == "y"


def test_column_mask_resolves_to_x() -> None:
    m = torch.zeros(1, 1, H, W)
    m[0, 0, :, 5] = 1.0
    assert resolve_line_axis(m) == "x"


def test_point_pattern_is_refused() -> None:
    """PLANTED: a scattered point mask. No band of lines exists."""
    g = torch.Generator().manual_seed(0)
    m = (torch.rand(1, 1, H, W, generator=g) < 0.25).float()
    with pytest.raises(ValueError, match="not line-structured"):
        resolve_line_axis(m)


def test_radial_trajectory_is_refused() -> None:
    """PLANTED: a two-spoke radial mask — 1-D, but spokes are not lines."""
    m = torch.zeros(1, 1, H, W)
    m[0, 0, H // 2, :] = 1.0  # a genuine line
    for i in range(H):  # ... plus a diagonal spoke, which is not one
        m[0, 0, i, i] = 1.0
    with pytest.raises(ValueError, match="not line-structured"):
        resolve_line_axis(m)


def test_one_partial_line_is_enough_to_refuse() -> None:
    """PLANTED: lines, except one row missing a single bin.

    The near-miss shape: a whole-mask heuristic that tolerated 'mostly lines'
    would pass this and then mislabel the band it reports.
    """
    m = _line_mask([2, 7])
    m[0, 0, 7, 3] = 0.0
    with pytest.raises(ValueError, match="not line-structured"):
        resolve_line_axis(m)


def test_line_indices_rejects_an_unknown_axis() -> None:
    with pytest.raises(ValueError, match="axis must be one of"):
        line_indices(_line_mask([1]), "z")


# --------------------------------------------------------------------------
# Claim 2 — the partition is disjoint, exhaustive, and keeps inert steps
# --------------------------------------------------------------------------


def test_partition_is_disjoint_and_covers_the_unobserved_support() -> None:
    obs = _line_mask([0, 1])
    next_masks = [_line_mask([0, 1, 2, 3]), _line_mask(list(range(H)))]
    bands = reveal_partition(next_masks, obs)

    stacked = torch.stack(bands).long().sum(dim=0)
    assert int(stacked.max()) <= 1, "a coefficient was revealed by two steps"
    covered = stacked.bool() | (obs[0, 0] > 0)
    assert bool(covered.all()), "some coefficient was never written"


def test_terminal_step_takes_everything_still_uncommitted() -> None:
    """The last band is the whole remainder — the loop's unconditional reveal."""
    obs = _line_mask([0])
    bands = reveal_partition([_line_mask([0, 1])], obs)
    assert int(bands[0].sum()) == W  # row 1
    assert int(bands[-1].sum()) == (H - 2) * W  # rows 2..H-1


def test_inert_step_is_reported_as_an_empty_band_not_dropped() -> None:
    """PLANTED: a step whose ``mask_next`` is already inside the observed support.

    Under ``dc_method='hard'`` the loop SKIPS this step. Dropping it here would
    slide every later band onto the wrong timestep.
    """
    obs = _line_mask([0, 1, 2])
    next_masks = [_line_mask([0, 1]), _line_mask(list(range(H)))]
    bands = reveal_partition(next_masks, obs)
    assert len(bands) == 3
    assert int(bands[0].sum()) == 0, "the inert step should reveal nothing"
    assert int(bands[1].sum()) == (H - 3) * W, "the full mask_next writes the remainder"


def test_terminal_band_is_empty_once_an_earlier_step_saw_the_full_mask() -> None:
    """The t=0 skip (#2067), as arithmetic.

    ``mask(0)`` is all-ones on an arm with ``train_identity_rung``, so the step
    whose ``mask_next`` is ``mask(0)`` — the one at t=1 — writes the entire
    remainder, and the terminal t=0 step has nothing left to reveal. That is why
    ``_step_reveals_anything`` returns False there and the model is never called
    at the rung that IS R=1.

    At R=2 the schedule is ``[1, 0]`` and this is the whole trajectory: one
    model call, at t=1, writing the outer half of k-space.
    """
    obs = _line_mask(list(range(0, H, 2)))  # R=2: every other line
    bands = reveal_partition([_line_mask(list(range(H)))], obs)
    assert int(bands[0].sum()) == (H // 2) * W, "t=1 writes every unobserved line"
    assert int(bands[-1].sum()) == 0, "t=0 has nothing left — the skipped terminal step"


# --------------------------------------------------------------------------
# Claim 3 / 4 — the gain separates shrinkage from phase
# --------------------------------------------------------------------------


def test_gain_is_unity_when_the_prediction_is_exact() -> None:
    g = torch.Generator().manual_seed(1)
    target = _interleaved(torch.randn(1, 2, H, W, generator=g, dtype=torch.cfloat))
    modulus, phase = complex_gain(target, target, _line_mask([4]))
    assert modulus == pytest.approx(1.0, abs=1e-5)
    assert phase == pytest.approx(0.0, abs=1e-5)


def test_modulus_reports_amplitude_shrinkage() -> None:
    """PLANTED: the band is written at 0.5x the true amplitude."""
    g = torch.Generator().manual_seed(2)
    target_c = torch.randn(1, 2, H, W, generator=g, dtype=torch.cfloat)
    band = _line_mask([4])
    pred_c = target_c.clone()
    pred_c[..., band[0, 0] > 0] *= 0.5
    modulus, phase = complex_gain(_interleaved(pred_c), _interleaved(target_c), band)
    assert modulus == pytest.approx(0.5, abs=1e-5)
    assert phase == pytest.approx(0.0, abs=1e-5)


def test_argument_reports_phase_error_that_energy_cannot_see() -> None:
    """PLANTED: the band is written with the right magnitude and a 30-deg rotation.

    A band energy / radial-spectrum check is exactly equal on these two inputs;
    the gain's argument is what distinguishes them.
    """
    g = torch.Generator().manual_seed(3)
    target_c = torch.randn(1, 2, H, W, generator=g, dtype=torch.cfloat)
    band = _line_mask([4])
    rotation = torch.polar(torch.tensor(1.0), torch.tensor(math.pi / 6))
    pred_c = target_c.clone()
    pred_c[..., band[0, 0] > 0] *= rotation

    modulus, phase = complex_gain(_interleaved(pred_c), _interleaved(target_c), band)
    assert modulus == pytest.approx(1.0, abs=1e-5), "a pure rotation must not shrink"
    assert phase == pytest.approx(math.pi / 6, abs=1e-5)

    sel = band[0, 0] > 0
    energy_pred = (paired_magnitude(_interleaved(pred_c))[..., sel] ** 2).sum()
    energy_true = (paired_magnitude(_interleaved(target_c))[..., sel] ** 2).sum()
    assert energy_pred == pytest.approx(float(energy_true), rel=1e-5)


def test_empty_band_reports_nan_rather_than_a_gain_of_one() -> None:
    g = torch.Generator().manual_seed(4)
    target = _interleaved(torch.randn(1, 2, H, W, generator=g, dtype=torch.cfloat))
    modulus, phase = complex_gain(target, target, torch.zeros(1, 1, H, W))
    assert math.isnan(modulus) and math.isnan(phase)


def test_gain_refuses_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        complex_gain(torch.zeros(1, 4, H, W), torch.zeros(1, 2, H, W), _line_mask([0]))


# --------------------------------------------------------------------------
# The report, and the SSOT witness for the interleaving convention
# --------------------------------------------------------------------------


def test_report_aligns_bands_with_the_schedule_and_scores_the_observed_support() -> None:
    g = torch.Generator().manual_seed(5)
    target_c = torch.randn(1, 2, H, W, generator=g, dtype=torch.cfloat)
    obs = _line_mask([0, 1])
    next_masks = [_line_mask([0, 1, 2, 3]), _line_mask(list(range(H)))]

    records = attribute_reveal_bands(
        _interleaved(target_c), _interleaved(target_c), next_masks, obs, [7, 3, 0]
    )

    assert [r["timestep"] for r in records] == [7, 3, 0, -1]
    assert records[0]["lines"] == [2, 3]
    assert all(r["line_axis"] == "y" for r in records)
    assert records[-1]["step"] == -1, "the observed support is the reference row"
    assert records[-1]["lines"] == [0, 1]
    assert records[-1]["gain_modulus"] == pytest.approx(1.0, abs=1e-5)


def test_report_refuses_a_schedule_that_does_not_match_the_bands() -> None:
    """PLANTED: one timestep too few — the attribution would name wrong steps."""
    z = torch.zeros(1, 2, H, W)
    with pytest.raises(ValueError, match="must be built from the same reverse"):
        attribute_reveal_bands(z, z, [_line_mask([0, 1])], _line_mask([0]), [5])


def test_paired_complex_and_paired_magnitude_share_one_convention() -> None:
    """The SSOT witness: two helpers, one interleaving, asserted rather than assumed."""
    g = torch.Generator().manual_seed(6)
    x = torch.randn(1, 8, H, W, generator=g)
    torch.testing.assert_close(paired_complex(x).abs(), paired_magnitude(x))


def test_paired_complex_refuses_an_odd_channel_count() -> None:
    with pytest.raises(ValueError, match="even channel"):
        paired_complex(torch.zeros(1, 3, H, W))


def test_report_refuses_a_5d_batch() -> None:
    """PLANTED: [B, C, H, W, D]. The interleaving reader would pair the wrong axis.

    ``paired_complex`` reads channels at ``shape[-3]``, which on a 5D batch is H,
    so it would return a plausible complex tensor of the wrong coefficients and
    every gain below would be a number with no meaning.
    """
    z = torch.zeros(1, 2, H, W, 3)
    with pytest.raises(ValueError, match=r"expects \[B, C, H, W\]"):
        attribute_reveal_bands(z, z, [_line_mask([0, 1])], _line_mask([0]), [5, 0])


# ---------------------------------------------------------------------------
# The vocabulary the audit layer refuses arms by.
#
# `resolve_line_axis` is still the authority -- it decides from the plane's
# geometry -- but it cannot run until validation. These constants are what let
# the audit reject an arm at load instead of hours in, and they must keep naming
# the same families the raise below reports.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_additive_is_excluded_from_the_partitioned_modes():
    """`additive` rewrites the whole plane every step, so no step 'wrote' a
    given coefficient -- and it is the CONSTRUCTOR DEFAULT, so an arm that omits
    `reverse_sampling_mode` gets it."""
    from spectramr.models.diffusion.kspace_process import VALID_REVERSE_MODES
    from spectramr.models.diffusion.reveal_attribution import PARTITIONED_REVERSE_MODES

    assert "additive" not in PARTITIONED_REVERSE_MODES
    # Pinned as a set, not a membership test: a new reverse mode has to make the
    # partition decision explicitly rather than inherit one. `replace_freeze_dc_t0`
    # keys its reveal off a different rung but writes each coefficient once and
    # freezes it, so it partitions for the same reason the other two do.
    assert set(PARTITIONED_REVERSE_MODES) == {
        "replace_freeze",
        "replace_freeze_dc",
        "replace_freeze_dc_t0",
    }
    # Every partitioned mode must be a mode the sampler will actually accept.
    assert PARTITIONED_REVERSE_MODES <= VALID_REVERSE_MODES


@pytest.mark.unit
def test_the_family_lists_match_the_families_the_raise_names():
    """Two owners of one vocabulary drift; the raise's message is the reference."""
    import inspect

    from spectramr.models.diffusion import reveal_attribution as ra

    message = inspect.getsource(ra.resolve_line_axis)
    for family in ra.NON_LINE_ACCELERATION_TYPES:
        assert family in message, (
            f"{family!r} is refused by the audit but not named in the runtime raise"
        )
    for family in ra.DIRECTION_DEPENDENT_ACCELERATION_TYPES:
        assert family in message
