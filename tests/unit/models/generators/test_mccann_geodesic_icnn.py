"""Tests for McCannGeodesicICNN (B-3.9 single-potential McCann field path)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.mccann_geodesic_icnn import McCannGeodesicICNN
from tests.utils.repo_scripts import require_repo_file


def _net(**kw) -> McCannGeodesicICNN:
    return McCannGeodesicICNN(hidden_channels=16, n_layers=3, **kw)


def _fields(b_s, b_t, n=2):
    return torch.full((n,), float(b_s)), torch.full((n,), float(b_t))


def test_forward_shape() -> None:
    bs, bt = _fields(0.1, 7.0)
    out = _net().eval()(torch.rand(2, 1, 16, 16), field_strength=bs, field_strength_target=bt)
    assert out.shape == (2, 1, 16, 16)


def test_path_function_is_monotone_in_field() -> None:
    m = _net().eval()
    t = torch.stack([m.path_function(torch.tensor([b])) for b in (0.1, 1.5, 3.0, 5.0, 7.0)]).flatten()
    assert torch.all(t[1:] >= t[:-1] - 1e-6)  # non-decreasing in field strength


def test_same_field_is_identity() -> None:
    # ANTI-DEGENERATE / load-bearing path: when source==target field the McCann increment
    # t(b_t)-t(b_s) is zero, so the output is exactly the source (no spurious transport).
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    bs, bt = _fields(3.0, 3.0)
    assert torch.allclose(m(x, field_strength=bs, field_strength_target=bt).detach(), x, atol=1e-5)


def test_distinct_fields_transport_is_load_bearing() -> None:
    # A non-trivial field pair must actually move the image (the displacement fires); and the
    # increment scales with the field gap (perturb the path so it is non-degenerate).
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        for p in m.icnn.z_raw:
            p.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    bs, bt = _fields(0.1, 7.0)
    out = m(x, field_strength=bs, field_strength_target=bt)
    assert not torch.allclose(out.detach(), x, atol=1e-4)


def test_path_cannot_collapse_to_constant() -> None:
    # REGRESSION (#20): an unfloored path slope a->0 collapses t(b) to a constant -> every
    # field pair maps identically (the whole McCann mechanism is a no-op). The a_min floor
    # keeps the slope > 0 even when path_a_raw is driven very negative.
    m = _net().eval()
    with torch.no_grad():
        m.path_a_raw.fill_(-30.0)  # softplus(path_a_raw) ~ 0
    assert float(m.path_slope().detach()) >= m._path_a_min * 0.99  # floored (~a_min, float32 tol)
    t_lo = float(m.path_function(torch.tensor([0.1])).detach())
    t_hi = float(m.path_function(torch.tensor([7.0])).detach())
    assert t_lo != t_hi  # path still discriminates fields (no collapse)
    assert m.resolved_path()["path_a"] >= m._path_a_min * 0.99  # provenance reflects the floor


def test_enforce_convexity_ablation_builds_and_runs() -> None:
    bs, bt = _fields(0.1, 7.0)
    out = _net(enforce_convexity=False)(
        torch.rand(1, 1, 16, 16), field_strength=bs[:1], field_strength_target=bt[:1]
    )
    assert out.shape == (1, 1, 16, 16)


def test_both_fields_required_for_probe() -> None:
    import inspect

    params = inspect.signature(McCannGeodesicICNN.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty
    assert params["field_strength_target"].default is inspect.Parameter.empty


def test_rejects_complex() -> None:
    bs, bt = _fields(0.1, 7.0, n=1)
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=bs, field_strength_target=bt)


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        McCannGeodesicICNN(in_channels=2)


def test_synthetic_forward_probe_passes_for_both_arms() -> None:
    # The probe synthesises field_strength == field_strength_target, for which McCann is
    # correctly the identity (zero increment) -> trips identity-collapse; the model declares
    # synthetic_forward_probe_skip = {"identity_collapse"} so a healthy arm passes --probe
    # (the input-invariance #20 check stays active). Regression for both arms.
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.forward_probe import synthetic_forward_probe

    for arm in ("b39_mccann_path", "b39_ablate_convexity"):
        cfg = TrainingSettings.from_yaml(
            str(require_repo_file(f"experiments/inprogress/mrixfields2026/task3/{arm}.yaml"))
        )
        report = synthetic_forward_probe(cfg)
        assert report.passed, f"{arm}: probe failed -> {getattr(report, 'message', report)}"


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "mccann_geodesic_icnn" in MODEL_REGISTRY


def test_probe_skip_includes_dead_loss() -> None:
    """McCann is the identity when source field == target field (the probe's
    synthetic case), so the configured placeholder l1 is ~0; the real objective
    is the strategy's compute_mccann_loss. The model must opt the dead_loss gate
    out to avoid a Tier-2 false positive (cluster g_total_loss=0.0758)."""
    assert McCannGeodesicICNN.synthetic_forward_probe_skip >= {
        "identity_collapse",
        "dead_loss",
    }
