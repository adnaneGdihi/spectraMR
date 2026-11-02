"""Banded equal-damage leaderboard: every covered degradation gets one vote per band."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from spectramr.core.metrics.meta_evaluation.banded_leaderboard import (
    BandedResult,
    banded_json,
    banded_leaderboard,
    banded_leaderboards,
    overall_ranking,
)
from spectramr.core.metrics.meta_evaluation.damage import DAMAGE_BANDS
from spectramr.core.metrics.metric_directions import metric_higher_is_better

BANDS = DAMAGE_BANDS["rel_l2"]
LIGHT = [0.03, 0.05, 0.07, 0.09]
MODERATE = [0.12, 0.18, 0.24]
HEAVY = [0.35, 0.45, 0.6]


def _samples(per_axis: dict[str, list[float]]):
    axis = np.concatenate([[a] * len(d) for a, d in per_axis.items()])
    damage = np.concatenate([np.asarray(d, float) for d in per_axis.values()])
    content = np.zeros(axis.shape[0], dtype=int)
    return axis, content, damage


def _run(values, axis, content, damage, hib=None, min_axes=1) -> BandedResult:
    hib = hib or dict.fromkeys(values, False)
    return banded_leaderboard(values, axis, content, damage, hib, BANDS, min_axes=min_axes)


def test_axis_weight_independent_of_steps() -> None:
    a = LIGHT + MODERATE + HEAVY
    b = [0.04, 0.06, 0.08, 0.15, 0.2, 0.28, 0.4, 0.5, 0.7]
    rng = np.random.default_rng(0)
    axis, content, damage = _samples({"A": a, "B": b})
    noisy = damage + rng.normal(0, 0.05, damage.shape)
    base = _run({"m": noisy}, axis, content, damage).overall

    keep = axis == "A"
    axis2 = np.concatenate([axis, axis[keep]])
    content2 = np.concatenate([content, content[keep]])
    damage2 = np.concatenate([damage, damage[keep]])
    noisy2 = np.concatenate([noisy, noisy[keep]])
    doubled = _run({"m": noisy2}, axis2, content2, damage2).overall
    assert doubled["m"] == pytest.approx(base["m"])


def test_range_width_cannot_tilt() -> None:
    # A is swept ten times harder (and with more levels) than B, over the same bands.
    a = LIGHT + MODERATE + [0.4, 0.8, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0, 10.0]
    b = LIGHT + MODERATE + HEAVY
    axis, content, damage = _samples({"A": a, "B": b})
    on_a = axis == "A"
    perfect_a = np.where(on_a, damage, 1.0)
    perfect_b = np.where(on_a, 1.0, damage)
    res = _run({"pa": perfect_a, "pb": perfect_b}, axis, content, damage)
    assert res.overall["pa"] == pytest.approx(res.overall["pb"])
    assert res.overall["pa"] == pytest.approx(0.5)


def test_uncovered_cell_absent() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE + [0.4, 0.5]})
    res = _run({"m": damage.copy()}, axis, content, damage)
    assert ("m", "A", "heavy") not in res.per_cell
    assert ("m", "heavy") not in res.per_band
    assert res.coverage[("A", "heavy")] == 2
    assert res.per_cell[("m", "A", "light")] == pytest.approx(1.0)


def test_orientation_from_directions() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE + HEAVY})
    values = {"psnr": -20.0 * np.log10(damage), "nmse": damage**2}
    hib = {k: metric_higher_is_better(k) for k in values}
    assert hib == {"psnr": True, "nmse": False}
    res = _run(values, axis, content, damage, hib)
    assert np.sign(res.overall["psnr"]) == np.sign(res.overall["nmse"]) == 1.0
    assert res.overall["psnr"] == pytest.approx(res.overall["nmse"])


def test_disqualified_excluded() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE + HEAVY})
    values = {"psnr": -damage, "ssim": -damage}
    out = banded_leaderboards(
        values, axis, content, {"rel_l2": damage}, {"psnr": True, "ssim": True}, min_axes=1
    )
    assert "psnr" not in out["rel_l2"].overall
    assert "ssim" in out["rel_l2"].overall


def test_imperceptible_never_votes() -> None:
    axis, content, damage = _samples({"A": [0.001, 0.005, 0.01, 0.015, *MODERATE]})
    res = _run({"m": damage.copy()}, axis, content, damage)
    assert res.coverage[("A", "imperceptible")] == 4
    assert {band for (_, _, band) in res.per_cell} == {"moderate"}


def test_constant_metric_scores_zero_and_unmeasured_is_incomplete() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE})
    light = damage < 0.1
    values = {
        "flat": np.ones_like(damage),
        "honest": np.where(light, 1.0, damage),  # blind on light: 0.0 there
        "gap": np.where(light, np.nan, damage),  # unmeasured on light
    }
    res = _run(values, axis, content, damage)
    assert res.per_cell[("flat", "A", "light")] == 0.0
    assert res.per_cell[("honest", "A", "light")] == 0.0
    assert ("gap", "A", "light") not in res.per_cell  # never zero-filled
    assert res.incomplete == {"gap": (("A", "light"),)}
    assert res.overall["gap"] == pytest.approx(1.0)  # partial, reported
    assert res.overall["honest"] == pytest.approx(0.5)
    # A partial score never outranks a complete one, however high it is.
    assert [m for m, _ in overall_ranking(res)] == ["honest", "flat", "gap"]


def test_crash_ranks_below_honest() -> None:
    per_axis = {"A": LIGHT + MODERATE + HEAVY, "B": LIGHT + MODERATE + HEAVY}
    axis, content, damage = _samples(per_axis)
    heavy = damage >= 0.3
    values = {
        "honest": np.where(heavy, 1.0, damage),
        "crash": np.where(heavy, np.nan, damage),
    }
    res = _run(values, axis, content, damage)
    assert res.overall["honest"] == pytest.approx(2 / 3)
    assert res.overall["crash"] == pytest.approx(1.0)
    assert res.incomplete["crash"] == (("A", "heavy"), ("B", "heavy"))
    assert [m for m, _ in overall_ranking(res)] == ["honest", "crash"]
    incomplete = banded_json({"rel_l2": res})["banded_leaderboard"]["rel_l2"]["incomplete"]
    assert incomplete == {"crash": [["A", "heavy"], ["B", "heavy"]]}


def test_finer_distinct_grid_leaves_overall_unchanged() -> None:
    """A twice-finer distinct grid on A, over the same bands, leaves ``overall`` unchanged.

    Exact (to float rounding of the means) for responses whose order within a band does
    not depend on the grid: monotone, anti-monotone or blind. A noisy response would
    re-estimate its own τ on the new levels; that is estimation, not axis weighting.
    """
    b = LIGHT + MODERATE + HEAVY
    fine_a = [
        *np.linspace(0.025, 0.095, 8),
        *np.linspace(0.11, 0.29, 6),
        *np.linspace(0.32, 0.9, 6),
    ]

    def score(a_levels: list[float]) -> dict[str, float]:
        axis, content, damage = _samples({"A": list(a_levels), "B": b})
        on_a = axis == "A"
        values = {
            "mono": damage**2,
            "anti": -np.sqrt(damage),
            "blind_on_a": np.where(on_a, 1.0, damage),
            "moderate_only": np.where((damage >= 0.1) & (damage < 0.3), damage, 1.0),
        }
        return _run(values, axis, content, damage).overall

    coarse, fine = score(b), score(fine_a)
    assert coarse.keys() == fine.keys()
    for m in coarse:
        assert fine[m] == pytest.approx(coarse[m], abs=1e-12)


def test_unknown_direction_and_bad_damage_raise() -> None:
    axis, content, damage = _samples({"A": LIGHT})
    with pytest.raises(KeyError, match="direction"):
        banded_leaderboard({"m": damage}, axis, content, damage, {}, BANDS)
    bad = damage.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="damage"):
        _run({"m": damage}, axis, content, bad)


def test_contents_average_within_a_cell() -> None:
    d = np.array(MODERATE * 2)
    axis = np.array(["A"] * 6)
    content = np.array([0, 0, 0, 1, 1, 1])
    m = np.concatenate([np.array(MODERATE), -np.array(MODERATE)])
    res = _run({"m": m}, axis, content, d)
    assert res.per_cell[("m", "A", "moderate")] == pytest.approx(0.0)


def test_json_has_string_keys_and_tied_metrics_share_a_rank() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE + HEAVY})
    values = {"b": damage.copy(), "a": damage.copy(), "c": -damage}
    out = banded_leaderboards(
        values,
        axis,
        content,
        {"rel_l2": damage, "grad_l2": damage},
        dict.fromkeys(values, False),
        min_axes=1,
    )
    assert overall_ranking(out["rel_l2"]) == [("a", 1), ("b", 1), ("c", 3)]
    payload = banded_json(out)
    json.dumps(payload)
    assert payload["banded_leaderboard"]["rel_l2"]["ranks"] == {"a": 1, "b": 1, "c": 3}
    assert set(payload) == {"banded_leaderboard", "coverage", "yardstick_agreement"}
    assert payload["coverage"]["rel_l2"]["A"]["heavy"] == 3
    assert payload["banded_leaderboard"]["rel_l2"]["per_band"]["light"]["a"] == 1.0
    assert payload["yardstick_agreement"]["grad_l2_vs_rel_l2"] == pytest.approx(1.0)


def test_banded_leaderboards_reads_the_bands_it_is_given() -> None:
    from spectramr.core.metrics.meta_evaluation.damage import bands_from_edges

    axis, content, damage = _samples({"A": LIGHT + MODERATE + HEAVY})
    values = {"ssim": -damage.copy()}
    default = banded_leaderboards(
        values, axis, content, {"rel_l2": damage}, {"ssim": True}, min_axes=1
    )
    # One band edge above every level: the whole sweep is imperceptible and nothing votes.
    wide = bands_from_edges({"rel_l2": [5.0, 6.0, 7.0]})
    custom = banded_leaderboards(
        values, axis, content, {"rel_l2": damage}, {"ssim": True}, bands=wide
    )
    assert default["rel_l2"].overall and not custom["rel_l2"].overall
    assert custom["rel_l2"].coverage[("A", "imperceptible")] == len(damage)


def test_vectorised_tau_b_matches_scipy_with_ties():
    from spectramr.core.metrics.meta_evaluation.banded_leaderboard import _tau_b_rows
    from spectramr.core.metrics.meta_evaluation.iqa_recalibration import kendall_tau

    rng = np.random.default_rng(0)
    d = np.round(rng.uniform(0, 1, 9), 1)
    x = np.round(rng.normal(size=(6, 9)), 0)
    x[0] = 1.0  # constant row -> 0.0, as kendall_tau's NaN rule
    got = _tau_b_rows(x, d)
    np.testing.assert_allclose(got, [kendall_tau(row, d) for row in x], atol=1e-12)


# ── fairness fixes: circular metrics, pooled τ, tied damage, band quorum, tie ranks ──


def _phantom(h: int = 48, shift: float = 0.0) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, h), indexing="ij")
    disc = ((xx / 0.7) ** 2 + ((yy - shift) / 0.9) ** 2 < 1).float()
    inner = ((xx - 0.2) ** 2 + (yy + 0.1) ** 2 < 0.15).float()
    return (0.2 + disc * (1 - 0.5 * inner))[None, None]


@pytest.fixture(scope="module")
def noise_darken():
    """Two contents x (noise, darken) x 8 levels, graded by the registered pool metrics."""
    import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the pool)
    from spectramr.core.metrics.meta_evaluation.damage import damage_batch
    from spectramr.core.metrics.registry import compute_metric

    rows = []
    for c, shift in enumerate((0.0, 0.15)):
        clean = _phantom(shift=shift)
        g = torch.Generator().manual_seed(c)
        nz = torch.complex(
            torch.randn(clean.shape, generator=g), torch.randn(clean.shape, generator=g)
        )
        for a in ("noise", "darken"):
            for t in np.linspace(0.05, 1.0, 8):
                deg = (clean + 0.3 * t * nz).abs() if a == "noise" else clean * (1 - 0.6 * t)
                rows.append((a, c, deg, clean))
    axis = np.array([r[0] for r in rows])
    content = np.array([r[1] for r in rows])
    damage = np.array([float(damage_batch(r[3], r[2], "rel_l2")[0]) for r in rows])
    names = ("nrmse_l2", "snr", "ssim")
    values = {m: np.array([float(compute_metric(m, r[2], r[3])) for r in rows]) for m in names}
    hib = {m: metric_higher_is_better(m) for m in names}
    return values, axis, content, damage, hib


def test_circular_metrics_are_dropped_declared_and_measured(noise_darken) -> None:
    values, axis, content, damage, hib = noise_darken
    res = banded_leaderboards(values, axis, content, {"rel_l2": damage}, hib)["rel_l2"]
    assert {"nrmse_l2", "snr"} <= set(res.circular_measured)
    assert {"nrmse_l2", "snr"} <= set(res.circular_declared)
    assert not {"nrmse_l2", "snr"} & set(res.overall)
    assert "measured" in res.circular["snr"] and "declared" in res.circular["snr"]
    block = banded_json({"rel_l2": res})["banded_leaderboard"]["rel_l2"]
    assert "nrmse_l2" in block["circular_measured"] and block["circular_threshold"] == 0.98


def test_ssim_is_not_measured_circular(noise_darken) -> None:
    values, axis, content, damage, hib = noise_darken
    res = banded_leaderboards(values, axis, content, {"rel_l2": damage}, hib)["rel_l2"]
    assert "ssim" not in res.circular_measured
    assert "ssim" in res.overall


def test_measured_circular_needs_two_axes(noise_darken) -> None:
    from spectramr.core.metrics.meta_evaluation.pooled_tau import measured_circular

    values, axis, content, damage, _ = noise_darken
    one = axis == "noise"
    sub = {m: v[one] for m, v in values.items()}
    assert measured_circular(sub, damage[one], axis[one], content[one]) == {}
    assert "snr" in measured_circular(values, damage, axis, content)


def test_two_perfectly_monotone_metrics_share_rank_one() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE, "B": LIGHT + MODERATE})
    values = {"zeta": damage**2, "alpha": damage.copy(), "noisy": damage[::-1].copy()}
    res = _run(values, axis, content, damage, min_axes=2)
    assert overall_ranking(res)[:2] == [("alpha", 1), ("zeta", 1)]
    assert overall_ranking(res)[2] == ("noisy", 3)


def test_tied_damage_cell_is_uncovered() -> None:
    # B saturates in heavy: five levels, one damage value. It must not vote 0 there.
    axis, content, damage = _samples({"A": LIGHT + HEAVY, "B": LIGHT + [0.5] * 5})
    res = _run({"m": damage.copy()}, axis, content, damage)
    assert ("m", "B", "heavy") not in res.per_cell
    assert ("m", "A", "heavy") in res.per_cell
    assert res.coverage[("B", "heavy")] == 1


def test_band_covered_by_one_axis_does_not_vote_overall() -> None:
    axis, content, damage = _samples({"A": LIGHT + MODERATE, "B": LIGHT})
    on_a = axis == "A"
    values = {"m": np.where(on_a & (damage >= 0.1), -damage, damage)}
    res = _run(values, axis, content, damage, min_axes=2)
    assert res.voting_bands == ("light",)
    assert res.per_band[("m", "moderate")] == pytest.approx(-1.0)  # reported
    assert res.overall["m"] == pytest.approx(1.0)  # does not vote
    assert banded_json({"rel_l2": res})["banded_leaderboard"]["rel_l2"]["min_axes_per_band"] == 2


def test_pooled_tau_equals_single_content_tau() -> None:
    from spectramr.core.metrics.meta_evaluation.iqa_recalibration import kendall_tau

    rng = np.random.default_rng(1)
    d = np.array([*LIGHT, 0.095])
    x = d + rng.normal(0, 0.02, d.size)
    axis, content, _ = _samples({"A": list(d)})
    res = _run({"m": x}, axis, content, d)
    assert res.per_cell[("m", "A", "light")] == pytest.approx(kendall_tau(x, d), abs=1e-12)


def test_pooled_tau_weights_contents_by_pair_count() -> None:
    # Content 0: 6 concordant levels (15 pairs); content 1: 3 discordant levels (3 pairs).
    six = [0.11, 0.13, 0.15, 0.17, 0.19, 0.21]
    d = np.array(six + MODERATE)
    axis = np.array(["A"] * 9)
    content = np.array([0] * 6 + [1] * 3)
    m = np.concatenate([np.array(six), -np.array(MODERATE)])
    res = _run({"m": m}, axis, content, d)
    assert res.per_cell[("m", "A", "moderate")] == pytest.approx((15 - 3) / 18)
