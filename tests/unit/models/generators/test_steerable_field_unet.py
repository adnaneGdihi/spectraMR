"""Tests for SteerableFieldUNet (B-1.4 C_4-equivariant cross-field synthesis)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.steerable_field_unet import SteerableFieldUNet
from tests.utils.repo_scripts import require_repo_file


def _net(use_equivariance: bool = True, use_field_conditioning: bool = True) -> SteerableFieldUNet:
    return SteerableFieldUNet(
        base_width=8,
        n_blocks=3,
        use_equivariance=use_equivariance,
        use_field_conditioning=use_field_conditioning,
    )


def test_forward_shape() -> None:
    out = _net()(torch.rand(2, 1, 32, 32), field_strength=torch.tensor([0.1, 7.0]))
    assert out.shape == (2, 1, 32, 32)


@pytest.mark.parametrize("s", [1, 2, 3])
def test_equivariant_to_rotation_with_fixed_field(s: int) -> None:
    # THE headline anti-facade test (#16/#19): the network is GENUINELY spatial-rotation
    # equivariant — rotate the image by s*90deg (field fixed) and the output rotates,
    # exactly. This is the B-1.4 mechanism; it holds at init (structural, not learned).
    torch.manual_seed(0)
    m = _net().eval()
    x = torch.randn(2, 1, 32, 32)
    fs = torch.tensor([0.1, 3.0])
    lhs = m(torch.rot90(x, s, dims=(-2, -1)), field_strength=fs)
    rhs = torch.rot90(m(x, field_strength=fs), s, dims=(-2, -1))
    assert torch.allclose(lhs, rhs, atol=1e-4), f"max err {(lhs - rhs).abs().max():.2e}"


def test_ablation_is_not_equivariant() -> None:
    # use_equivariance=False -> plain Conv2d -> rotation equivariance breaks (the control).
    torch.manual_seed(0)
    m = _net(use_equivariance=False).eval()
    x = torch.randn(1, 1, 32, 32)
    fs = torch.tensor([3.0])
    lhs = m(torch.rot90(x, 1, dims=(-2, -1)), field_strength=fs)
    rhs = torch.rot90(m(x, field_strength=fs), 1, dims=(-2, -1))
    assert float((lhs - rhs).abs().max().detach()) > 1e-2


def test_equivariance_constraint_reduces_parameters() -> None:
    # The sqrt(|G|) mechanism: the equivariant weight-sharing constraint yields a strictly
    # smaller hypothesis class -> fewer parameters than the matched-width plain control.
    equi = _net().num_parameters()
    plain = _net(use_equivariance=False).num_parameters()
    assert equi < plain


def test_field_is_load_bearing_after_perturbation() -> None:
    # FiLM is zero-init (identity) so the field is inert at init by design; once the head
    # is perturbed, the source field genuinely changes the output (#16 anti-facade).
    torch.manual_seed(0)
    m = _net().eval()
    x = torch.rand(2, 1, 32, 32)
    with torch.no_grad():
        m.field_mlp[-1].weight.normal_(0.0, 0.5)
        m.field_mlp[-1].bias.normal_(0.0, 0.5)
    lo = m(x, field_strength=torch.tensor([0.1, 0.1]))
    hi = m(x, field_strength=torch.tensor([7.0, 7.0]))
    assert not torch.allclose(lo, hi)


def test_field_film_preserves_equivariance() -> None:
    # Even with a NON-identity FiLM, equivariance must hold (the FiLM is per-base-channel,
    # broadcast over orientations -> commutes with the group action).
    torch.manual_seed(0)
    m = _net().eval()
    with torch.no_grad():
        m.field_mlp[-1].weight.normal_(0.0, 0.5)
        m.field_mlp[-1].bias.normal_(0.0, 0.5)
    x = torch.randn(1, 1, 24, 24)
    fs = torch.tensor([5.0])
    lhs = m(torch.rot90(x, 1, dims=(-2, -1)), field_strength=fs)
    rhs = torch.rot90(m(x, field_strength=fs), 1, dims=(-2, -1))
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_field_agnostic_mode_ignores_field() -> None:
    m = _net(use_field_conditioning=False).eval()
    x = torch.rand(1, 1, 16, 16)
    a = m(x, field_strength=torch.tensor([0.1]))
    b = m(x, field_strength=torch.tensor([7.0]))
    assert torch.allclose(a, b)  # field accepted (cohort signature) but unused -> identical


def test_requires_field_when_conditioning_enabled() -> None:
    # field_strength is a required kwarg (no default, matching the cohort & the probe); an
    # explicit None with conditioning on still raises the clear ValueError guard.
    with pytest.raises(ValueError, match="field_strength"):
        _net(use_field_conditioning=True).eval()(torch.rand(1, 1, 16, 16), field_strength=None)


def test_synthetic_forward_probe_passes_for_both_arms() -> None:
    # REGRESSION (#10 audit-ladder false-negative): the Tier-2 synthetic forward probe must
    # PASS for both b14 arms. A `= None` default on field_strength made the probe skip
    # filling it -> a healthy model failed the --probe promotion gate. field_strength is now
    # required so the probe injects it.
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.forward_probe import synthetic_forward_probe

    for arm in ("b14_steerable_equivariant_7t", "b14_ablate_equivariance"):
        cfg = TrainingSettings.from_yaml(
            str(require_repo_file(f"experiments/inprogress/mrixfields2026/task1/{arm}.yaml"))
        )
        report = synthetic_forward_probe(cfg)
        assert report.passed, f"{arm}: probe failed -> {getattr(report, 'message', report)}"


def test_rejects_complex() -> None:
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=torch.tensor([1.0]))


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        SteerableFieldUNet(in_channels=2)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "steerable_field_unet" in MODEL_REGISTRY


# --- Multi-contrast conditioning (mrixfields2026 broad rollout) ---


def test_contrast_widens_field_mlp_input() -> None:
    m = _net()
    m2 = SteerableFieldUNet(
        base_width=8, n_blocks=3, use_contrast_conditioning=True, num_contrasts=3
    )
    assert m.field_mlp[0].in_features == 1
    assert m2.field_mlp[0].in_features == 1 + 3  # tau + num_contrasts


def test_contrast_requires_id_when_enabled() -> None:
    m = SteerableFieldUNet(
        base_width=8, n_blocks=3, use_contrast_conditioning=True, num_contrasts=3
    )
    with pytest.raises(ValueError, match="contrast_id"):
        m(torch.rand(1, 1, 32, 32), field_strength=torch.tensor([3.0]))


@pytest.mark.parametrize("s", [1, 2, 3])
def test_contrast_preserves_equivariance(s: int) -> None:
    # Contrast is a GLOBAL per-sample label, so C_4 rotation equivariance still holds
    # with conditioning on (the FiLM stays per-base-channel, broadcast over orientation).
    torch.manual_seed(0)
    m = SteerableFieldUNet(
        base_width=8, n_blocks=3, use_contrast_conditioning=True, num_contrasts=3
    ).eval()
    x = torch.randn(2, 1, 32, 32)
    fs = torch.tensor([0.1, 3.0])
    cid = torch.tensor([1, 2])
    lhs = m(torch.rot90(x, s, dims=(-2, -1)), field_strength=fs, contrast_id=cid)
    rhs = torch.rot90(m(x, field_strength=fs, contrast_id=cid), s, dims=(-2, -1))
    assert torch.allclose(lhs, rhs, atol=1e-4)


def test_contrast_changes_output_after_perturbation() -> None:
    torch.manual_seed(0)
    m = SteerableFieldUNet(
        base_width=8, n_blocks=3, use_contrast_conditioning=True, num_contrasts=3
    ).eval()
    with torch.no_grad():
        m.field_mlp[-1].weight.normal_(0.0, 0.5)
        m.field_mlp[-1].bias.normal_(0.0, 0.5)
    x = torch.rand(2, 1, 32, 32)
    fs = torch.tensor([3.0, 3.0])
    a = m(x, field_strength=fs, contrast_id=torch.tensor([0, 0]))
    b = m(x, field_strength=fs, contrast_id=torch.tensor([2, 2]))
    assert not torch.allclose(a, b)
