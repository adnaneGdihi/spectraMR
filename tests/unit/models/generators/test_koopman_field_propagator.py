"""Tests for KoopmanFieldPropagator (B-3.10 continuous Koopman cross-field propagator)."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.koopman_operator import koopman_continuous_propagate
from spectramr.models.generators.koopman_field_propagator import KoopmanFieldPropagator
from tests.utils.repo_scripts import require_repo_file


def _net(**kw) -> KoopmanFieldPropagator:
    return KoopmanFieldPropagator(width=24, koopman_dim=16, n_blocks=2, **kw)


def _f(b_s, b_t, n=2):
    return torch.full((n,), float(b_s)), torch.full((n,), float(b_t))


def test_forward_shape() -> None:
    bs, bt = _f(0.1, 7.0)
    for semigroup in (True, False):
        out = _net(use_koopman_semigroup=semigroup)(
            torch.rand(2, 1, 16, 16), field_strength=bs, field_strength_target=bt
        )
        assert out.shape == (2, 1, 16, 16)


def test_koopman_delta_is_zero_at_init() -> None:
    # A is zero-init -> exp(dtau*A)=I -> propagation is identity -> the output is the bare
    # autoencoder regardless of the field (field-blind at init, like a zero-init residual).
    m = _net(use_koopman_semigroup=True).eval()
    x = torch.rand(2, 1, 16, 16)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert torch.allclose(o1.detach(), o2.detach())  # A=0 -> field-blind at init
    assert m.spectral_abscissa() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("width,kdim,spatial", [(24, 16, 16), (48, 32, 64)])
def test_field_is_load_bearing_after_perturbation(width, kdim, spatial) -> None:
    # After the generator escapes zero-init, distinct field gaps give distinct propagation.
    # Pinned at toy AND production dims (width=48, koopman_dim=32).
    torch.manual_seed(0)
    m = KoopmanFieldPropagator(width=width, koopman_dim=kdim, n_blocks=2).eval()
    with torch.no_grad():
        m.generator.normal_(0.0, 0.3)
    x = torch.rand(2, 1, spatial, spatial)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(o1.detach(), o2.detach())


def test_nonlinear_arm_zero_delta_at_init_then_field_dependent() -> None:
    # The nonlinear control's field_mlp final conv is zero-init -> zero latent delta at init,
    # MATCHING the semigroup arm's zero-init A (no init-asymmetry confound, #17). It exposes no
    # Koopman generator, and becomes field-dependent once the MLP escapes zero.
    torch.manual_seed(0)
    m = _net(use_koopman_semigroup=False).eval()
    assert m.spectral_abscissa() is None
    x = torch.rand(2, 1, 16, 16)
    o1 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o2 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert torch.allclose(o1.detach(), o2.detach())  # zero delta at init -> field-blind, like linear arm
    with torch.no_grad():  # escape the zero-init delta
        m.field_mlp[-1].weight.normal_(0.0, 0.3)
    o3 = m(x, field_strength=torch.tensor([0.1, 0.1]), field_strength_target=torch.tensor([7.0, 7.0]))
    o4 = m(x, field_strength=torch.tensor([3.0, 3.0]), field_strength_target=torch.tensor([1.0, 1.0]))
    assert not torch.allclose(o3.detach(), o4.detach())  # field-dependent after escaping zero


def test_semigroup_composition_is_exact() -> None:
    # The distinctive Koopman claim: ONE generator handles ANY gap and composes exactly
    # (s->m->t == s->t) in the observable space. Verified on the model's own encoder + generator.
    torch.manual_seed(0)
    m = _net(use_koopman_semigroup=True)
    with torch.no_grad():
        m.generator.normal_(0.0, 0.3)
    x = torch.rand(2, 1, 16, 16)
    import math

    b_s = torch.tensor([0.1, 0.5])
    b_m = torch.tensor([1.5, 2.0])
    b_t = torch.tensor([7.0, 3.0])
    tau = lambda b: torch.log10(b)  # noqa: E731
    z_s = m.encoder(x)
    one_hop = koopman_continuous_propagate(z_s, m.generator, tau(b_t) - tau(b_s))
    two_hop = koopman_continuous_propagate(
        koopman_continuous_propagate(z_s, m.generator, tau(b_m) - tau(b_s)),
        m.generator,
        tau(b_t) - tau(b_m),
    )
    assert torch.allclose(one_hop, two_hop, atol=1e-4)
    assert math.isfinite(m.spectral_abscissa())


def test_size1_field_broadcast_raises() -> None:
    m = _net(use_koopman_semigroup=True)
    x = torch.rand(2, 1, 16, 16)
    with pytest.raises(ValueError, match="field batch"):
        m(x, field_strength=torch.tensor([0.5]), field_strength_target=torch.tensor([3.0]))
    out = m(x, field_strength=torch.tensor([0.5, 1.5]), field_strength_target=torch.tensor([3.0, 7.0]))
    assert out.shape == (2, 1, 16, 16)


def test_both_fields_required_for_probe() -> None:
    import inspect

    params = inspect.signature(KoopmanFieldPropagator.forward).parameters
    assert params["field_strength"].default is inspect.Parameter.empty
    assert params["field_strength_target"].default is inspect.Parameter.empty


def test_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="magnitude-only"):
        KoopmanFieldPropagator(in_channels=2)
    with pytest.raises(ValueError, match="koopman_dim"):
        KoopmanFieldPropagator(koopman_dim=1)
    with pytest.raises(ValueError, match="use_koopman_semigroup"):
        KoopmanFieldPropagator(use_koopman_semigroup="yes")  # type: ignore[arg-type]


def test_rejects_complex() -> None:
    bs, bt = _f(0.1, 7.0, n=1)
    with pytest.raises(ValueError, match="magnitude"):
        _net()(torch.rand(1, 1, 8, 8, dtype=torch.cfloat), field_strength=bs, field_strength_target=bt)


def test_synthetic_forward_probe_passes_for_both_arms() -> None:
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.validation.forward_probe import synthetic_forward_probe

    for arm in ("b310_koopman", "b310_ablate_nonlinear"):
        cfg = TrainingSettings.from_yaml(
            str(require_repo_file(f"experiments/inprogress/mrixfields2026/task3/{arm}.yaml"))
        )
        report = synthetic_forward_probe(cfg)
        assert report.passed, f"{arm}: probe failed -> {getattr(report, 'message', report)}"


def test_registered() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert "koopman_field_propagator" in MODEL_REGISTRY
