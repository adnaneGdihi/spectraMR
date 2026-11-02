"""Damage yardsticks, bands and the yardstick-stability check."""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.core.metrics.meta_evaluation.damage import (
    DAMAGE_BANDS,
    YARDSTICKS,
    band_of,
    bands_from_edges,
    compare_yardsticks,
    damage,
    damage_batch,
    disqualified_metrics,
    parse_yardsticks,
)


@pytest.fixture
def clean() -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.rand(16, 16, generator=g) + 0.5


def test_rel_l2_unclipped(clean: torch.Tensor) -> None:
    assert damage(clean, 3 * clean, "rel_l2") == pytest.approx(2.0)


def test_identity_is_zero(clean: torch.Tensor) -> None:
    for name in YARDSTICKS:
        assert damage(clean, clean, name) == 0.0


def test_complex_input_uses_magnitude(clean: torch.Tensor) -> None:
    rot = clean.to(torch.complex64) * torch.exp(torch.tensor(1j * 0.7))
    assert damage(clean, rot, "rel_l2") == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("yardstick", ["rel_l2", "grad_l2"])
def test_damage_batch_matches_scalar(clean: torch.Tensor, yardstick: str) -> None:
    g = torch.Generator().manual_seed(1)
    stack = clean + 0.1 * torch.randn(5, 16, 16, generator=g)
    batch = damage_batch(clean, stack, yardstick)
    assert batch.shape == (5,)
    for i in range(5):
        assert float(batch[i]) == pytest.approx(damage(clean, stack[i], yardstick), rel=1e-5)


def test_damage_batch_shape_mismatch_raises(clean: torch.Tensor) -> None:
    with pytest.raises(ValueError):
        damage_batch(clean, torch.zeros(3, 8, 8), "rel_l2")


def test_band_edges() -> None:
    assert band_of(0.0, "rel_l2") == "imperceptible"
    assert band_of(0.02, "rel_l2") == "light"
    assert band_of(0.30, "rel_l2") == "heavy"
    assert band_of(5.0, "rel_l2") == "heavy"
    assert DAMAGE_BANDS["grad_l2"][-1][2] == math.inf


@pytest.mark.parametrize("bad", [-0.1, float("nan")])
def test_band_of_rejects_invalid_damage(bad: float) -> None:
    with pytest.raises(ValueError):
        band_of(bad, "rel_l2")


def test_band_of_unknown_yardstick_raises() -> None:
    with pytest.raises(KeyError):
        band_of(0.1, "vibes")


def test_yardstick_declares_the_metrics_it_disqualifies() -> None:
    assert {"psnr", "nmse"} <= disqualified_metrics("rel_l2")
    assert "hfen" in disqualified_metrics("grad_l2")


def test_disqualified_metrics_unknown_raises() -> None:
    with pytest.raises(KeyError):
        disqualified_metrics("vibes")


def test_compare_yardsticks_detects_an_unstable_ranking() -> None:
    up = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    down = {m: -s for m, s in up.items()}
    stable = compare_yardsticks({"rel_l2": up, "grad_l2": up})
    flipped = compare_yardsticks({"rel_l2": up, "grad_l2": down})
    assert stable["grad_l2_vs_rel_l2"] == pytest.approx(1.0)
    assert flipped["grad_l2_vs_rel_l2"] == pytest.approx(-1.0)


def test_band_votes_needs_min_levels_and_never_imperceptible() -> None:
    from spectramr.core.metrics.meta_evaluation.damage import band_votes

    assert band_votes("light", 3) and not band_votes("light", 2)
    assert band_votes("heavy", 5) and not band_votes("moderate", 0)
    assert not band_votes("imperceptible", 20)
    assert band_votes("light", 2, min_levels=2)


def test_bands_from_edges_reproduces_the_defaults() -> None:
    assert bands_from_edges({}) == dict(DAMAGE_BANDS)
    assert bands_from_edges({"rel_l2": [0.02, 0.10, 0.30]}) == dict(DAMAGE_BANDS)


def test_bands_from_edges_sets_one_yardstick_and_keeps_the_rest() -> None:
    got = bands_from_edges({"grad_l2": [0.05, 0.2, 0.6]})
    assert got["grad_l2"] == (
        ("imperceptible", 0.0, 0.05),
        ("light", 0.05, 0.2),
        ("moderate", 0.2, 0.6),
        ("heavy", 0.6, math.inf),
    )
    assert got["rel_l2"] == DAMAGE_BANDS["rel_l2"]
    assert band_of(0.1, "grad_l2", got) == "light"


@pytest.mark.parametrize(
    ("edges", "match"),
    [
        ({"vibes": [0.1, 0.2, 0.3]}, "unknown yardstick"),
        ({"rel_l2": [0.1, 0.2]}, "three edges"),
        ({"rel_l2": "0.1,0.2,0.3"}, "three edges"),
        ({"rel_l2": [0.1, "x", 0.3]}, "numbers"),
        ({"rel_l2": [-0.1, 0.2, 0.3]}, "non-negative"),
        ({"rel_l2": [0.1, 0.2, math.inf]}, "finite"),
        ({"rel_l2": [0.1, 0.1, 0.3]}, "strictly increasing"),
        ({"rel_l2": [0.3, 0.2, 0.1]}, "strictly increasing"),
    ],
)
def test_bands_from_edges_rejects(edges: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bands_from_edges(edges)


def test_parse_yardsticks() -> None:
    assert parse_yardsticks(" rel_l2 , grad_l2 ") == ("rel_l2", "grad_l2")
    with pytest.raises(ValueError, match="at least one"):
        parse_yardsticks(" , ")
    with pytest.raises(ValueError, match="vibes"):
        parse_yardsticks("rel_l2,vibes")


def test_every_declared_circular_name_is_a_registered_metric() -> None:
    import spectramr.core.metrics.evaluation_metrics  # noqa: F401  (registers the pool)
    from spectramr.core.metrics.registry import list_available

    registered = set(list_available())
    for name in YARDSTICKS:
        assert disqualified_metrics(name) <= registered, name


def test_declared_sets_cover_the_exact_monotone_transforms() -> None:
    assert {"nrmse_l2", "snr"} <= disqualified_metrics("rel_l2")
    assert "gradient_error" in disqualified_metrics("grad_l2")
    assert "grad_entropy" not in disqualified_metrics("grad_l2")


def test_compare_yardsticks_ranks_tied_scores_together() -> None:
    # Three metrics tied at the top under both yardsticks: no alphabetical order inside the tie.
    a = {"x": 1.0, "y": 1.0, "z": 1.0, "w": 0.2}
    b = {"z": 1.0, "y": 1.0, "x": 1.0, "w": 0.2}
    assert compare_yardsticks({"rel_l2": a, "grad_l2": b})["grad_l2_vs_rel_l2"] == pytest.approx(
        1.0
    )


def test_min_axes_per_band_is_two() -> None:
    from spectramr.core.metrics.meta_evaluation.damage import MIN_AXES_PER_BAND

    assert MIN_AXES_PER_BAND == 2


def _old_grad_l2(deg: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    def g(t: torch.Tensor) -> torch.Tensor:
        gx = t[..., :, 1:] - t[..., :, :-1]
        gy = t[..., 1:, :] - t[..., :-1, :]
        return torch.cat([gx.flatten(1), gy.flatten(1)], dim=1)

    gc = g(clean)
    return torch.linalg.vector_norm(g(deg) - gc, dim=1) / torch.linalg.vector_norm(
        gc, dim=1
    ).clamp_min(1e-12)


def test_four_d_grad_l2_is_bit_identical_to_the_in_plane_formula() -> None:
    g = torch.Generator().manual_seed(3)
    clean = torch.rand(1, 2, 12, 10, generator=g)
    stack = (clean + 0.2 * torch.randn(4, 2, 12, 10, generator=g)).abs()
    want = _old_grad_l2(stack, clean.expand_as(stack))
    assert torch.equal(damage_batch(clean[0], stack, "grad_l2"), want)


def test_five_d_grad_l2_adds_the_through_plane_difference() -> None:
    clean = torch.zeros(1, 1, 4, 8, 8)
    clean[..., 2:, :, :] = 1.0  # a step along D only: no in-plane gradient
    deg = clean.clone()
    deg[..., 2, :, :] = 0.5
    assert float(damage_batch(clean, deg, "grad_l2", volume=True)) > 0.0


def test_five_d_grad_l2_weights_each_difference_by_its_spacing() -> None:
    g = torch.Generator().manual_seed(4)
    clean = torch.rand(1, 1, 5, 9, 9, generator=g)
    deg = clean.clone()
    deg[..., 2, :, :] += 0.3  # a slab: mostly a through-plane change
    unit = float(damage_batch(clean, deg, "grad_l2", volume=True))
    iso = float(damage_batch(clean, deg, "grad_l2", voxel_mm=(2.0, 2.0, 2.0), volume=True))
    thick = float(damage_batch(clean, deg, "grad_l2", voxel_mm=(6.0, 1.0, 1.0), volume=True))
    assert iso == pytest.approx(unit, rel=1e-6)  # a common scale cancels in the ratio
    assert thick < unit


def test_voxel_mm_on_a_four_d_stack_raises() -> None:
    x = torch.rand(2, 1, 8, 8)
    with pytest.raises(ValueError, match="voxel_mm"):
        damage_batch(x[0], x, "grad_l2", voxel_mm=(1.0, 1.0, 1.0))


def test_grad_l2_dims_are_stamped_per_rank() -> None:
    from spectramr.core.metrics.meta_evaluation.damage import grad_l2_provenance

    assert grad_l2_provenance(volume=False)["grad_l2_dims"] == [-2, -1]
    vol = grad_l2_provenance(volume=True, voxel_mm=(6.0, 1.0, 1.0), spacing_source="encoded_fov_z")
    assert vol == {
        "grad_l2_dims": [-3, -2, -1],
        "grad_l2_weighting": "inverse_voxel_mm",
        "grad_l2_voxel_mm": [6.0, 1.0, 1.0],
        "spacing_source": "encoded_fov_z",
    }
    assert grad_l2_provenance(volume=True)["grad_l2_weighting"] == "unit_voxel"


def test_a_leading_one_clean_broadcasts_over_the_stack() -> None:
    g = torch.Generator().manual_seed(5)
    clean = torch.rand(1, 1, 3, 8, 8, generator=g)
    stack = clean + 0.1 * torch.rand(4, 1, 3, 8, 8, generator=g)
    for y in YARDSTICKS:
        assert torch.equal(damage_batch(clean, stack, y), damage_batch(clean[0], stack, y))
    with pytest.raises(ValueError, match="differ"):
        damage_batch(torch.rand(2, 1, 3, 8, 8), stack, "rel_l2")


def test_a_five_d_stack_gains_no_through_plane_term_unless_declared_a_volume() -> None:
    clean = torch.zeros(1, 1, 4, 8, 8)
    clean[..., 2:, :, :] = 1.0
    deg = clean.clone()
    deg[..., 2, :, :] = 0.5
    assert float(damage_batch(clean, deg, "grad_l2")) == 0.0  # in-plane only
    assert float(damage_batch(clean, deg, "grad_l2", volume=True)) > 0.0
    with pytest.raises(ValueError, match="volume"):
        damage_batch(clean[0, 0], deg[0], "grad_l2", volume=True)
