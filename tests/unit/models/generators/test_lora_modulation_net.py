"""Tests for LoRAModulationNet (B-3.6 continuous low-rank modulation translator)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.lora_modulation_net import LoRAModulationNet
from tests.utils.repo_scripts import require_repo_file


def _net(**kw) -> LoRAModulationNet:
    return LoRAModulationNet(width=24, n_blocks=2, **kw)


def _f(b_s, b_t, n=2):
    return torch.full((n,), float(b_s)), torch.full((n,), float(b_t))


def _activate(m: LoRAModulationNet) -> None:
    """Escape the zero-init LoRA delta + activate the (bias-free) field MLP."""
    with torch.no_grad():
        for c in [m.lift, *m.block_convs, m.head]:
            c.lora_up.weight.normal_(0.0, 0.3)
            c.alpha_mlp.weight.normal_(0.0, 0.5)


def test_forward_shape() -> None:
    bs, bt = _f(0.1, 7.0)
    out = _net()(torch.rand(2, 1, 16, 16), field_strength=bs, field_strength_target=bt)
    assert out.shape == (2, 1, 16, 16)


@pytest.mark.parametrize("width,n_blocks,spatial", [(24, 2, 16), (48, 3, 64)])
def test_field_is_load_bearing_after_perturbation(width, n_blocks, spatial) -> None:
    # Pinned at BOTH toy and production dims (width=48,n_blocks=3) per the prior-review lesson.
    torch.manual_seed(0)
    m = LoRAModulationNet(width=width, n_blocks=n_blocks, lora_rank=4).eval()
    _activate(m)
    x = torch.rand(2, 1, spatial, spatial)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(o1.detach(), o2.detach())


@pytest.mark.parametrize("width,n_blocks,spatial", [(24, 2, 16), (48, 3, 64)])
def test_rank_zero_is_field_blind_ablation(width, n_blocks, spatial) -> None:
    # lora_rank=0 -> the shared backbone ignores the field (the one-knob control); still
    # translates (non-degenerate), just field-agnostically. Pinned at production dims too.
    m = LoRAModulationNet(width=width, n_blocks=n_blocks, lora_rank=0).eval()
    x = torch.rand(2, 1, spatial, spatial)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert torch.allclose(o1.detach(), o2.detach())  # field-blind
    assert o1.shape == (2, 1, spatial, spatial)


def test_alpha_mlp_is_bias_free_so_zero_weight_is_inert() -> None:
    # bias=False is load-bearing: with a bias, W_alpha->0 would leave a NON-ZERO but
    # field-INVARIANT delta (a naive "delta != 0" probe would false-PASS). Without the bias,
    # zeroing the field-weight gives alpha=0 -> delta=0, so the adapter is inert (field-invariant
    # coincides with the inert solution already covered by the zero-init guard).
    m = _net(lora_rank=4).eval()
    assert m.lift.alpha_mlp.bias is None
    with torch.no_grad():  # activate lora_up but KILL the field-reading weight
        for c in [m.lift, *m.block_convs, m.head]:
            c.lora_up.weight.normal_(0.0, 0.3)
            c.alpha_mlp.weight.zero_()
    x = torch.rand(2, 1, 16, 16)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert torch.allclose(o1.detach(), o2.detach())


def test_freeze_field_is_param_matched_and_field_blind() -> None:
    # PARAM-MATCHED field-blind control (#17): identical params to the field-conditioned arm,
    # but the constant field makes the (active) adapter field-invariant.
    torch.manual_seed(0)
    free = LoRAModulationNet(width=48, n_blocks=3, lora_rank=4, freeze_field=False)
    frozen = LoRAModulationNet(width=48, n_blocks=3, lora_rank=4, freeze_field=True)
    assert sum(p.numel() for p in free.parameters()) == sum(p.numel() for p in frozen.parameters())
    _activate(frozen)
    frozen.eval()
    x = torch.rand(2, 1, 16, 16)
    o1 = frozen(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = frozen(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert torch.allclose(o1.detach(), o2.detach())  # constant field -> field-invariant


def test_freeze_field_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="freeze_field"):
        LoRAModulationNet(freeze_field="yes")  # type: ignore[arg-type]


def test_size1_field_broadcast_raises() -> None:
    # A size-1 field must NOT silently broadcast one (b_s,b_t) across the whole batch (#9).
    m = _net(lora_rank=4)
    x = torch.rand(2, 1, 16, 16)
    with pytest.raises(ValueError, match="field batch"):
        m(x, field_strength=torch.tensor([0.5]), field_strength_target=torch.tensor([3.0]))
    out = m(x, field_strength=torch.tensor([0.5, 1.5]), field_strength_target=torch.tensor([3.0, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_lora_rank_fraction_is_honest() -> None:
    # The compliance bound r / max-rank(W) = r / min(out_ch, in_ch*k^2) = r/width for the
    # dominant width->width convs (NOT r / conv-element-count, which was misleadingly tiny).
    m = _net(lora_rank=4)  # width=24 -> 4/24 = 16.7%
    frac = m.lora_rank_fraction()
    assert abs(frac - 4 / 24) < 1e-9
    assert 0.0 < frac < 0.2  # a real low-rank bound, not the sub-1% element-count ratio


def test_both_fields_required_for_probe() -> None:
    import inspect

    params = inspect.signature(LoRAModulationNet.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty
    assert params["field_strength_target"].default is inspect.Parameter.empty


def test_synthetic_forward_probe_passes_for_all_arms() -> None:
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.forward_probe import synthetic_forward_probe

    for arm in ("b36_lora_modulation", "b36_ablate_rank0", "b36_ablate_param_matched"):
        cfg = TrainingSettings.from_yaml(
            str(require_repo_file(f"experiments/inprogress/mrixfields2026/task3/{arm}.yaml"))
        )
        report = synthetic_forward_probe(cfg)
        assert report.passed, f"{arm}: probe failed -> {getattr(report, 'message', report)}"


def test_rejects_complex() -> None:
    bs, bt = _f(0.1, 7.0, n=1)
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=bs, field_strength_target=bt)


def test_rejects_multichannel() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        LoRAModulationNet(in_channels=2)


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "lora_modulation_net" in MODEL_REGISTRY
