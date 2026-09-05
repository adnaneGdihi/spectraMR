"""Tests for BrenierICNN (B-1.5 source-conditioned Brenier OT map)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.brenier_icnn import BrenierICNN
from tests.utils.repo_scripts import require_repo_file


def _net(**kw) -> BrenierICNN:
    return BrenierICNN(hidden_channels=16, n_layers=3, **kw)


def test_forward_shape_and_near_identity_init() -> None:
    m = _net().eval()
    x = torch.rand(2, 1, 16, 16)
    out = m(x, field_strength=torch.tensor([0.1, 3.0]))
    assert out.shape == (2, 1, 16, 16)
    assert float((out.detach() - x).norm() / x.norm()) < 0.5  # near-identity Brenier init


def test_map_is_monotone_in_input() -> None:
    # The output is grad of a convex potential -> a monotone OT map: <T(a)-T(b),a-b> >= 0.
    torch.manual_seed(0)
    m = _net().eval()
    a, b = torch.rand(6, 1, 16, 16), torch.rand(6, 1, 16, 16)
    fs = torch.full((6,), 0.1)
    ta, tb = m(a, field_strength=fs), m(b, field_strength=fs)
    mono = ((ta - tb) * (a - b)).flatten(1).sum(1)
    assert float(mono.min().detach()) >= -1e-3


def test_field_is_load_bearing_after_perturbation() -> None:
    # The source field conditions the potential (zero-init FiLM -> inert at init; perturb the
    # field MLP, then distinct source fields must give distinct maps).
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        m.icnn.field_mlp.weight.normal_(0.0, 0.5)
        m.icnn.field_mlp.bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo.detach(), hi.detach())


def test_enforce_convexity_ablation_builds_and_runs() -> None:
    out = _net(enforce_convexity=False)(torch.rand(1, 1, 16, 16), field_strength=torch.tensor([3.0]))
    assert out.shape == (1, 1, 16, 16)


def test_field_strength_required_for_probe() -> None:
    import inspect

    p = inspect.signature(BrenierICNN.forward).parameters["field_strength"]
    assert p.default is inspect.Parameter.empty  # no default -> Tier-2 probe injects it


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        BrenierICNN(in_channels=2)


def test_synthetic_forward_probe_passes_for_both_arms() -> None:
    # The Brenier map is near-identity at init (designed OT start), which trips the Tier-2
    # identity-collapse heuristic; the model declares synthetic_forward_probe_skip =
    # {"identity_collapse"} so a healthy arm passes --probe (the input-invariance #20 check
    # stays active). Regression for both the convex arm and its enforce_convexity=false control.
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.forward_probe import synthetic_forward_probe

    for arm in (
        "b15_brenier_map",
        "b15_ablate_convexity",
        "b15_brenier_intensity_undamped",
        "b15_brenier_intensity_transfer",
    ):
        cfg = TrainingSettings.from_yaml(
            str(require_repo_file(f"experiments/inprogress/mrixfields2026/task1/{arm}.yaml"))
        )
        report = synthetic_forward_probe(cfg)
        assert report.passed, f"{arm}: probe failed -> {getattr(report, 'message', report)}"


def test_intensity_adaptation_passes_through_to_icnn() -> None:
    m = BrenierICNN(hidden_channels=16, n_layers=3, intensity_adaptation="undamped")
    assert m.icnn.intensity_adaptation == "undamped"


def test_unknown_intensity_adaptation_raises() -> None:
    with pytest.raises(ValueError, match="intensity_adaptation"):
        BrenierICNN(intensity_adaptation="bogus")


def test_transfer_arm_is_monotone_in_input() -> None:
    # Route 2 (field-conditioned monotone intensity transfer) stays a valid Brenier map.
    torch.manual_seed(0)
    m = BrenierICNN(hidden_channels=16, n_layers=3, intensity_adaptation="transfer").eval()
    with torch.no_grad():
        for p in m.icnn.transfer_mlp.parameters():
            p.normal_(0.0, 0.5)
    a, b = torch.rand(6, 1, 16, 16), torch.rand(6, 1, 16, 16)
    fs = torch.full((6,), 0.1)
    ta, tb = m(a, field_strength=fs), m(b, field_strength=fs)
    mono = ((ta - tb) * (a - b)).flatten(1).sum(1)
    assert float(mono.min().detach()) >= -1e-3


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "brenier_icnn" in MODEL_REGISTRY


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_icnn_field_dim() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    assert m.icnn.field_mlp.in_features == 1 + 3  # source field + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    m = _net(use_contrast_conditioning=True, num_contrasts=3)
    with pytest.raises(ValueError, match="contrast_id"):
        m(torch.rand(1, 1, 16, 16), field_strength=torch.tensor([3.0]))


def test_contrast_changes_output_after_perturbation() -> None:
    # Convexity in x is preserved (field enters via the gamma>=0 softplus FiLM); contrast
    # still changes the transport map once the field MLP moves off its zero init.
    torch.manual_seed(0)
    m = _net(use_contrast_conditioning=True, num_contrasts=3).eval()
    with torch.no_grad():
        m.icnn.field_mlp.weight.normal_(0.0, 0.5)
        m.icnn.field_mlp.bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 16, 16)
    fs = torch.tensor([3.0, 3.0])
    a = m(x, field_strength=fs, contrast_id=torch.tensor([0, 0]))
    b = m(x, field_strength=fs, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a.detach(), b.detach())
