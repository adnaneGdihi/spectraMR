import pytest
import torch

from spectramr.models.generators.bloch_cycle_maps import (
    T1_RANGE_MS,
    T2_RANGE_MS,
    BlochSequence,
    build_refinement_modules,
    maps_from_logits,
    resolve_fixed_relaxation,
)


def test_maps_lie_in_their_physiological_ranges():
    maps = maps_from_logits(torch.randn(2, 3, 4, 4), None, None)
    assert maps[:, 0].min() > 0.0
    assert T1_RANGE_MS[0] <= maps[:, 1].min() and maps[:, 1].max() <= T1_RANGE_MS[1]
    assert T2_RANGE_MS[0] <= maps[:, 2].min() and maps[:, 2].max() <= T2_RANGE_MS[1]


def test_no_bloch_config_keeps_the_historical_sequence():
    assert BlochSequence.from_config(None, "SE") == BlochSequence("SE", 3000.0, 100.0, 90.0)


def test_unknown_sequence_type_raises():
    with pytest.raises(ValueError, match="sequence_type"):
        BlochSequence.from_config(None, "EPI")


def test_te_must_lie_below_tr():
    with pytest.raises(ValueError, match="TE < TR"):
        BlochSequence.from_config({"TR": 50.0, "TE": 100.0}, "SE")


def test_fixed_relaxation_is_range_checked():
    assert resolve_fixed_relaxation("t2", True, None) is None
    with pytest.raises(ValueError, match="outside"):
        resolve_fixed_relaxation("t2", False, 1000.0)
    with pytest.raises(ValueError, match="learnable is true"):
        resolve_fixed_relaxation("t1", True, 800.0)


def test_refinement_stage_count():
    assert len(build_refinement_modules(1, 1, None, None)) == 0
    assert len(build_refinement_modules(1, 4, 8, 2)) == 3
    with pytest.raises(ValueError, match=">= 1"):
        build_refinement_modules(1, 0, None, None)


def test_pd_is_unbounded_so_the_render_can_exceed_the_t2_ceiling():
    """A PD capped at 1 caps a TE=100 ms spin echo at exp(-100 / T2_max) ~ 0.72."""
    logits = torch.full((1, 3, 2, 2), 6.0)
    maps = maps_from_logits(logits, None, None)
    pd, t1, t2 = maps[:, 0], maps[:, 1], maps[:, 2]
    signal = pd * (1 - torch.exp(-3000.0 / t1)) * torch.exp(-100.0 / t2)
    assert signal.min() > torch.exp(torch.tensor(-100.0 / T2_RANGE_MS[1]))
