"""Vessel maps of the flow axes: the caller's map, or seeded discs; both are validated."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations.catalogue.flow_vessels import (
    as_5d,
    check_positive,
    resolve_vessels,
    sample_draws,
)


def _x(shape=(3, 1, 4, 32, 32)) -> torch.Tensor:
    return torch.ones(shape, dtype=torch.complex64)


def test_seeded_discs_are_binary_deterministic_and_nonempty() -> None:
    x = _x()
    u = sample_draws(x, seed=5)
    v = resolve_vessels(x, None, u, "t", ndim_in=5)
    assert v.shape == (3, 1, 1, 32, 32)
    assert set(v.unique().tolist()) <= {0.0, 1.0}
    assert (v.flatten(1).sum(1) > 0).all()
    assert torch.equal(v, resolve_vessels(x, None, sample_draws(x, seed=5), "t", ndim_in=5))
    assert not torch.equal(v, resolve_vessels(x, None, sample_draws(x, seed=6), "t", ndim_in=5))


def test_discs_do_not_depend_on_the_batch() -> None:
    x = _x()
    full = resolve_vessels(x, None, sample_draws(x, 5), "t", ndim_in=5)
    one = resolve_vessels(x[:1], None, sample_draws(x[:1], 5), "t", ndim_in=5)
    assert torch.equal(full[:1], one)


def test_one_to_three_discs() -> None:
    x = _x((64, 1, 1, 64, 64))
    v = resolve_vessels(x, None, sample_draws(x, 1), "t", ndim_in=5)
    n = v.flatten(1).sum(1)
    assert n.min() > 0 and len({int(a) for a in n}) > 1  # the count of discs varies


def test_caller_map_is_used_as_given() -> None:
    x = _x()
    m = torch.zeros(1, 1, 4, 32, 32)
    m[..., 3, 5] = 1.0
    v = resolve_vessels(x, m, sample_draws(x, 0), "t", ndim_in=5)
    assert torch.equal(v, m)


def test_two_d_map_is_lifted_to_a_single_slice() -> None:
    x = as_5d(torch.ones(2, 1, 16, 16, dtype=torch.complex64))
    v = resolve_vessels(x, torch.ones(2, 1, 16, 16), sample_draws(x, 0), "t", ndim_in=4)
    assert v.shape == (2, 1, 1, 16, 16)


@pytest.mark.parametrize(
    "bad", [torch.ones(1, 1, 32, 32), torch.ones(1, 1, 4, 32, 31), torch.ones(2, 1, 4, 32, 32)]
)
def test_unfit_map_raises(bad: torch.Tensor) -> None:
    x = _x()
    with pytest.raises(ValueError, match="vessel_map"):
        resolve_vessels(x, bad, sample_draws(x, 0), "t", ndim_in=5)


def test_non_tensor_map_raises() -> None:
    x = _x()
    with pytest.raises(ValueError, match="vessel_map"):
        resolve_vessels(x, [[1.0]], sample_draws(x, 0), "t", ndim_in=5)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan"), True, "5"])
def test_check_positive_rejects(bad) -> None:
    with pytest.raises(ValueError, match="te_ms"):
        check_positive("t", te_ms=bad)
