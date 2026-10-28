"""Tests for the canonical sinusoidal timestep embedding.

The two properties that matter are the ones the previous `nn.Linear(1, ·)`
encoding failed: neighbouring timesteps must be DISTINGUISHABLE, and the
codes must not collapse when the schedule horizon is small (T=28 here, against
a 10000-base frequency ladder built for T~1000).
"""

from __future__ import annotations

import math

import pytest
import torch

from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding


def test_shape_and_dtype():
    emb = sinusoidal_timestep_embedding(torch.arange(5), 64, max_timesteps=28.0)
    assert emb.shape == (5, 64)
    assert emb.dtype == torch.float32


def test_odd_dim_is_padded():
    emb = sinusoidal_timestep_embedding(torch.arange(3), 65, max_timesteps=28.0)
    assert emb.shape == (3, 65)


def test_adjacent_timesteps_are_distinguishable():
    """The defect this replaces: a rank-1 map of t puts every code on one line.
    Adjacent steps must be separated by a non-trivial margin after scaling."""
    t = torch.arange(28)
    emb = sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
    diffs = (emb[1:] - emb[:-1]).norm(dim=-1)
    assert diffs.min() > 1e-3, f"adjacent codes collapse; min gap {diffs.min():.2e}"


def test_all_timesteps_are_pairwise_distinct():
    t = torch.arange(28)
    emb = sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
    d = torch.cdist(emb, emb)
    off_diag = d + torch.eye(len(t)) * 1e9
    assert off_diag.min() > 1e-3, "two distinct timesteps share an embedding"


def test_unscaled_small_horizon_still_separates():
    """Sanity: without max_timesteps the codes are still distinct for small t —
    the scaling matters for CONDITIONING quality, not for raw injectivity."""
    emb = sinusoidal_timestep_embedding(torch.arange(28), 256)
    d = torch.cdist(emb, emb) + torch.eye(28) * 1e9
    assert d.min() > 1e-3


def test_codes_span_many_dimensions_not_a_line():
    """A linear encoding of ``t`` yields a RANK-1 matrix of codes — every
    timestep on one line, which is exactly why ``nn.Linear(1, ·)`` could not
    express "which t". The sinusoidal basis must span the full set instead.

    Computed in float64 deliberately: the frequency ladder spans ~4 orders of
    magnitude, so float32's default rank tolerance truncates a mathematically
    full-rank matrix to ~4. That is a tolerance artifact, not a property of the
    encoding, and asserting on it would pin the wrong thing.
    """
    emb = sinusoidal_timestep_embedding(torch.arange(28), 256, max_timesteps=28.0)
    assert torch.linalg.matrix_rank(emb.double()) == 28


def test_float_input_accepted():
    """Acceleration factors are continuous and are a legitimate conditioning
    signal on this path, so floats must not be silently truncated."""
    a = sinusoidal_timestep_embedding(torch.tensor([2.0]), 64)
    b = sinusoidal_timestep_embedding(torch.tensor([2.5]), 64)
    assert (a - b).norm() > 1e-4


@pytest.mark.parametrize("bad_dim", [0, 1])
def test_dim_below_two_raises(bad_dim):
    """``half_dim - 1`` divides; dim<2 would emit NaN rather than fail."""
    with pytest.raises(ValueError, match="dim must be"):
        sinusoidal_timestep_embedding(torch.arange(3), bad_dim)


@pytest.mark.parametrize("bad_max", [0.0, -1.0])
def test_non_positive_max_timesteps_raises(bad_max):
    with pytest.raises(ValueError, match="max_timesteps"):
        sinusoidal_timestep_embedding(torch.arange(3), 64, max_timesteps=bad_max)


def test_scaling_changes_the_code():
    """max_timesteps must actually be read — a silently ignored argument here
    would reintroduce the collapse it exists to prevent."""
    t = torch.arange(28)
    assert (
        sinusoidal_timestep_embedding(t, 256, max_timesteps=28.0)
        - sinusoidal_timestep_embedding(t, 256, max_timesteps=1000.0)
    ).norm() > 1.0


# ---------------------------------------------------------------------------
# scale="integer" (#2315): the 29-rung ladder must be resolvable
# ---------------------------------------------------------------------------


def _adjacent_distance(scale: str, horizon: int = 29, dim: int = 256) -> float:
    t = torch.arange(horizon)
    e = sinusoidal_timestep_embedding(t, dim, max_timesteps=float(horizon), scale=scale)
    return float((e[1:] - e[:-1]).norm(dim=1).min())


def test_integer_scale_separates_adjacent_rungs():
    """PLANTED pair: the dividing form puts neighbouring rungs 0.094 apart on a
    norm of 11.3; the integer form must separate them by at least 1."""
    assert _adjacent_distance("unit_interval") < 0.2
    assert _adjacent_distance("integer") > 1.0


def test_integer_scale_ignores_the_horizon():
    t = torch.arange(29)
    a = sinusoidal_timestep_embedding(t, 64, max_timesteps=29.0, scale="integer")
    b = sinusoidal_timestep_embedding(t, 64, scale="integer")
    assert torch.equal(a, b)


def test_unit_interval_is_the_legacy_default():
    """Trained checkpoints were fitted to the dividing form; the default keeps it."""
    t = torch.arange(29)
    assert torch.equal(
        sinusoidal_timestep_embedding(t, 64, max_timesteps=29.0),
        sinusoidal_timestep_embedding(t, 64, max_timesteps=29.0, scale="unit_interval"),
    )


def test_unknown_scale_raises():
    with pytest.raises(ValueError, match="timestep embedding scale"):
        sinusoidal_timestep_embedding(torch.arange(3), 64, scale="log")


# ---------------------------------------------------------------------------
# scale="horizon_band": a frequency ladder derived from the schedule's horizon
# ---------------------------------------------------------------------------


def _codes(horizon: int, dim: int = 256) -> torch.Tensor:
    return sinusoidal_timestep_embedding(
        torch.arange(horizon), dim, max_timesteps=float(horizon), scale="horizon_band"
    )


def test_horizon_band_endpoints_are_quarter_turns():
    """The fastest channel turns a quarter cycle per step, the slowest a quarter
    cycle over the T - 1 steps of the horizon; no DDPM constant is involved."""
    T, half = 29, 128
    phases = torch.arcsin(_codes(T)[:, :half].double())
    assert phases[1, 0].item() == pytest.approx(math.pi / 2, abs=1e-4)
    assert phases[-1, half - 1].item() == pytest.approx(math.pi / 2, abs=1e-4)


@pytest.mark.parametrize(("horizon", "spacing"), [(29, 6.59), (31, 6.52)])
def test_horizon_band_separates_every_pair_further_than_integer(horizon, spacing):
    """Measured at d=256: 6.59 (T=29) and 6.52 (T=31) against integer's 2.66."""
    e = _codes(horizon)
    d = torch.cdist(e, e) + torch.eye(horizon) * 1e9
    assert _adjacent_distance("horizon_band", horizon) == pytest.approx(spacing, abs=0.01)
    assert d.min() > 2 * _adjacent_distance("integer", horizon)


def test_horizon_band_keeps_nearer_steps_nearer():
    """Aliasing buys separation at the cost of order; the band keeps 0.99 of
    triples ordered by step gap, where a 1000/T stretch keeps 0.93 at T=29."""
    T = 29
    d = torch.cdist(_codes(T), _codes(T))
    gap = (torch.arange(T)[:, None] - torch.arange(T)[None, :]).abs()
    ok = tot = 0
    for i in range(T):
        closer = gap[i][:, None] < gap[i][None, :]
        tot += int(closer.sum())
        ok += int((closer & (d[i][:, None] < d[i][None, :])).sum())
    assert ok / tot > 0.98


def test_horizon_band_two_rungs_are_a_quarter_turn_apart():
    e = _codes(2, dim=8)
    assert torch.allclose(e[1, :4], torch.ones(4))


@pytest.mark.parametrize("bad", [None, 1.0])
def test_horizon_band_needs_a_horizon_of_two_or_more(bad):
    with pytest.raises(ValueError, match="horizon_band"):
        sinusoidal_timestep_embedding(torch.arange(3), 64, max_timesteps=bad, scale="horizon_band")


@pytest.mark.parametrize("bad_max", [0.0, -1.0])
def test_horizon_band_non_positive_horizon_raises(bad_max):
    with pytest.raises(ValueError, match="max_timesteps must be > 0"):
        sinusoidal_timestep_embedding(
            torch.arange(3), 64, max_timesteps=bad_max, scale="horizon_band"
        )
