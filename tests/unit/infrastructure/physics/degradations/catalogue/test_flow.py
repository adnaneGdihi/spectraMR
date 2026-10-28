"""flow_ghosts, flow_voids and inflow_enhancement: invariants per mode and closed-form physics."""

from __future__ import annotations

from dataclasses import replace
from itertools import pairwise

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    Realisation,
    UnsupportedAcquisitionMode,
    affected_components,
    apply_degradation,
)

from .flow_helpers import FLOW, GEOMS, PAIRS, VOXEL, M, _apply, _phantom, _rms  # noqa: F401

# ---- invariants, every (axis, mode) ---------------------------------------------------------


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_theta_zero_is_identity(name, mode):
    x = _phantom(mode)
    assert torch.equal(_apply(name, x, 0.0, mode), x)


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_monotone_finite_shape_deterministic(name, mode):
    x = _phantom(mode)
    outs = [_apply(name, x, t, mode) for t in (0.0, 0.3, 0.6, 1.0)]
    for y in outs:
        assert y.shape == x.shape and torch.isfinite(torch.view_as_real(y)).all()
    res = [_rms(x, y) for y in outs]
    # flow_voids saturates at zero signal once v >= 2 thk / TE, so the steps are non-strict.
    assert all(a <= b + 1e-7 for a, b in pairwise(res)) and res[-1] > res[0], res
    assert torch.equal(_apply(name, x, 0.6, mode), outs[2])


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_batch_invariant(name, mode):
    x = _phantom(mode)
    alone = _apply(name, x, 0.6, mode)
    pair = _apply(name, torch.cat([x, 10 * _phantom(mode, seed=9)]), 0.6, mode)
    torch.testing.assert_close(pair[:1], alone, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_runs_without_voxel_size(name, mode):
    geom = None if GEOMS[mode] is None else replace(GEOMS[mode], voxel_mm=None)
    y = apply_degradation(name, _phantom(mode), 0.6, seed=1, geometry=geom)
    assert torch.isfinite(torch.view_as_real(y)).all()


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_seeded_vessels_depend_on_the_seed(name, mode):
    x = _phantom(mode)
    assert not torch.equal(_apply(name, x, 0.6, mode, seed=1), _apply(name, x, 0.6, mode, seed=2))


@pytest.mark.parametrize(("name", "mode"), PAIRS)
def test_optional_vessel_map_is_used(name, mode):
    x = _phantom(mode)
    empty = torch.zeros(*x.shape[:2], *([1] * (x.dim() - 4)), *x.shape[-2:])
    assert torch.equal(_apply(name, x, 0.6, mode, vessel_map=empty), x)
    full = torch.ones_like(empty)
    assert _rms(_apply(name, x, 0.6, mode, vessel_map=full), x) > 1e-3


@pytest.mark.parametrize("name", FLOW)
def test_bad_kwargs_and_maps_raise(name):
    x = _phantom(M.VOLUME_3D)
    with pytest.raises(ValueError, match="vessel_map"):
        _apply(name, x, 0.5, M.VOLUME_3D, vessel_map=torch.ones(1, 1, 24, 24))
    knob = {"flow_ghosts": "tr_ms", "flow_voids": "te_ms", "inflow_enhancement": "tr_ms"}[name]
    with pytest.raises(ValueError, match=knob):
        _apply(name, x, 0.5, M.VOLUME_3D, **{knob: 0.0})
    with pytest.raises(TypeError):
        _apply(name, x, 0.5, M.VOLUME_3D, nonsense=1)


def test_declarations():
    reg = DEGRADATION_REGISTRY
    assert affected_components("flow_ghosts") == {"magnitude", "phase"}
    assert (
        affected_components("flow_voids")
        == affected_components("inflow_enhancement")
        == {"magnitude"}
    )
    for name in FLOW:
        assert reg[name].requires == frozenset()
        for mode in reg[name].supported_modes:
            assert reg[name].realisation[mode] is Realisation.NATIVE
    assert reg["flow_ghosts"].supported_modes == set(M)
    assert reg["flow_voids"].supported_modes == set(M)
    assert reg["inflow_enhancement"].supported_modes == {M.MULTISLICE_2D, M.VOLUME_3D}
    for name, hi in zip(FLOW, (0.8, 50.0, 50.0), strict=True):
        assert reg[name].severity.primary.at_theta_max == hi


def test_affects_matches_what_the_output_changes():
    x = _phantom(M.VOLUME_3D)
    for name in ("flow_voids", "inflow_enhancement"):
        y = _apply(name, x, 1.0, M.VOLUME_3D)
        assert torch.allclose(torch.angle(y), torch.angle(x), atol=1e-6)
        assert not torch.allclose(y.abs(), x.abs())
    y = _apply("flow_ghosts", x, 1.0, M.VOLUME_3D)
    assert not torch.allclose(torch.angle(y), torch.angle(x), atol=1e-3)
    assert not torch.allclose(y.abs(), x.abs())


def test_voids_and_inflow_keep_a_real_input_real():
    x = _phantom(M.VOLUME_3D).abs()
    for name in ("flow_voids", "inflow_enhancement"):
        assert not _apply(name, x, 0.7, M.VOLUME_3D).is_complex()


def test_inflow_is_refused_on_a_single_slice():
    spec = DEGRADATION_REGISTRY["inflow_enhancement"]
    assert "single slice" in spec.severity.notes
    with pytest.raises(UnsupportedAcquisitionMode, match="no entry depth") as err:
        apply_degradation("inflow_enhancement", _phantom(M.SLICE_2D), 0.5)
    assert "multislice_2d" in str(err.value)
    with pytest.raises(UnsupportedAcquisitionMode, match="no entry depth"):
        spec.fn  # noqa: B018


def test_vessel_map_weights_are_clamped_to_unit_range():
    x = _phantom(M.VOLUME_3D)
    wild = torch.full((1, 1, 1, 24, 24), 7.0)
    wild[..., :12] = -3.0
    clipped = wild.clamp(0.0, 1.0)
    for name in ("flow_voids", "flow_ghosts", "inflow_enhancement"):
        a = _apply(name, x, 0.6, M.VOLUME_3D, vessel_map=wild)
        b = _apply(name, x, 0.6, M.VOLUME_3D, vessel_map=clipped)
        assert torch.equal(a, b)
