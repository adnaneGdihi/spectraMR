"""``vmf_diffusion`` built through the production factory path (rule 16).

``populate_model_registry`` -> ``ModelFactory.create_generator`` is the path
``GeneratorBuilder.build`` takes; a direct constructor call would not exercise
``strict_kwargs``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def factory():
    from spectramr.models.factories.model_factory import ModelFactory
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    return ModelFactory()


def _build(factory, **declared):
    kwargs = {"hidden": 8, "num_blocks": 1, "timesteps": 20, "sampling_steps": 4, **declared}
    return factory.create_generator(
        "vmf_diffusion",
        in_channels=4,
        out_channels=4,
        _declared_keys=frozenset(kwargs),
        **kwargs,
    )


def test_registered_as_diffusion_latent_model() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    entry = MODEL_REGISTRY["vmf_diffusion"]
    assert entry["mode"] == "diffusion"
    assert entry["capabilities"].input_domain == "latent"


def test_factory_builds_and_mechanism_fires(factory) -> None:
    from spectramr.models.diffusion.vmf_diffusion import VMFDiffusion

    torch.manual_seed(0)
    model = _build(factory)
    assert isinstance(model, VMFDiffusion)
    z0 = F.normalize(torch.randn(2, 4, 6, 6), dim=1)
    loss = model.training_loss(z0)
    loss.backward()
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 2.0
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)

    out = model.sample((2, 4, 6, 6))
    assert out.shape == (2, 4, 6, 6)
    assert torch.allclose(out.norm(dim=1), torch.ones(2, 6, 6), atol=1e-4)
    assert model.provenance()["kappa_schedule"] == "log_linear"


def test_conditioning_is_enforced(factory) -> None:
    model = _build(factory, cond_channels=2)
    z = F.normalize(torch.randn(1, 4, 4, 4), dim=1)
    t = torch.zeros(1, dtype=torch.long)
    assert model(z, t, torch.randn(1, 2, 4, 4)).shape == z.shape
    with pytest.raises(ValueError, match="cond is missing"):
        model(z, t)


def test_unknown_declared_kwarg_raises_through_factory(factory) -> None:
    with pytest.raises(Exception, match="bogus_knob"):
        _build(factory, bogus_knob=1)


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        ({"kappa_schedule": "cosine"}, "Unknown kappa_schedule"),
        ({"inference_sampler": "nope"}, "not registered"),
        ({"sampling_steps": 50}, "sampling_steps <= timesteps"),
        ({"hidden": 7}, "even hidden"),
    ],
)
def test_invalid_knobs_raise(factory, bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        _build(factory, **bad)


def test_scalar_latent_rejected(factory) -> None:
    with pytest.raises(ValueError, match="in_channels >= 2"):
        factory.create_generator("vmf_diffusion", in_channels=1, out_channels=1)
