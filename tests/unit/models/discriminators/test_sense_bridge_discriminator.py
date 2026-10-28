"""The bridge must combine coils the way the rest of the arm does, or not run.

Two properties carry this critic, and neither is visible in a shape assertion.

*Which* combine. ``sense_adjoint`` (matched filter) and ``coil_combine`` with
``method="sense"`` (Roemer) differ by a spatially-varying real factor, so a
critic trained through one while the arm is scored through the other is
sharpening a differently-shaded image than the one being measured. The tests
below pin the exact operator, not just "an image came out".

*Whether it combines at all*. Without sensitivity maps the honest options are
to raise or to degrade to a root-sum-of-squares image. RSS produces a
plausible picture and a training run that looks fine, which is precisely the
silent fallback non-negotiable 3 forbids -- so absence must raise.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.init_registry import populate_model_registry  # noqa: E402
from spectramr.models.registry import (  # noqa: E402
    MODEL_REGISTRY,
    get_model_capabilities,
)

COILS, SIZE, BATCH = 4, 16, 2


def _critic(**kw):
    populate_model_registry()
    from spectramr.models.discriminators import SenseBridgeDiscriminator

    kw.setdefault("critic_kwargs", {"ndf": 8, "n_layers": 2})
    return SenseBridgeDiscriminator(**kw)


def _smaps():
    torch.manual_seed(0)
    return torch.randn(BATCH, COILS, SIZE, SIZE, dtype=torch.complex64)


def _kspace():
    torch.manual_seed(1)
    return torch.randn(BATCH, COILS, SIZE, SIZE, dtype=torch.complex64)


def test_registered_under_its_yaml_name():
    """The arm names it in ``model.discriminator_component``; unregistered = unbuildable."""
    populate_model_registry()
    assert "sense_bridge_patchgan" in MODEL_REGISTRY


def test_declares_accepts_complex_so_the_strategy_stops_realifying():
    """The capability IS the seam.

    ``DiffusionTrainingStrategy._align_for_critic`` reads this to decide whether
    to hand over the complex pair untouched. Drop the flag and both sides arrive
    real -- one interleaved, one block-stacked -- and the critic separates them
    on channel order instead of on image content.
    """
    populate_model_registry()
    caps = get_model_capabilities("sense_bridge_patchgan")
    assert caps is not None and caps.accepts_complex is True


def test_bridge_is_the_sense_adjoint_not_rss_and_not_roemer():
    """Pin the operator. All three produce an image of the same shape."""
    from spectramr.infrastructure.physics.fft_ops import (
        coil_combine_rss,
        ifft2c,
        sense_adjoint,
    )

    d = _critic(in_channels=1)
    smaps, k = _smaps(), _kspace()
    d.set_smaps_provider(lambda b: smaps[:b])

    bridged = d.bridge(k)
    expected = sense_adjoint(k, smaps=smaps).abs()
    assert torch.allclose(bridged, expected, atol=1e-6)

    rss = coil_combine_rss(ifft2c(k))
    assert not torch.allclose(bridged, rss.abs(), atol=1e-4), (
        "the bridge collapsed to a root-sum-of-squares combine, which ignores "
        "coil phase and is exactly the fallback the smaps raise exists to prevent"
    )
    roemer = expected / (smaps.abs() ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)
    assert not torch.allclose(bridged, roemer, atol=1e-4), (
        "the bridge is Roemer-normalised (coil_combine method='sense'), which "
        "shades the image differently from the arm's own validation metrics"
    )


def test_complex_and_interleaved_inputs_give_the_same_image():
    """Both forms reach this critic: complex from the target, interleaved from the generator.

    ``eval()`` matters -- the inner PatchGAN applies spectral norm, whose power
    iteration mutates on every forward in train mode, so two calls on identical
    input legitimately differ there.
    """
    d = _critic(in_channels=1)
    d.eval()
    smaps, k = _smaps(), _kspace()
    d.set_smaps_provider(lambda b: smaps[:b])
    interleaved = torch.view_as_real(k).permute(0, 1, 4, 2, 3).flatten(1, 2)
    assert interleaved.shape[1] == 2 * COILS and not torch.is_complex(interleaved)
    assert torch.equal(d(k), d(interleaved))


def test_complex_repr_keeps_phase_visible_to_the_critic():
    """``magnitude`` discards phase; ``complex`` presents it as two channels."""
    mag, cpx = _critic(in_channels=1), _critic(in_channels=2, image_repr="complex")
    smaps, k = _smaps(), _kspace()
    for d in (mag, cpx):
        d.set_smaps_provider(lambda b: smaps[:b])
    assert mag.bridge(k).shape[1] == 1
    assert cpx.bridge(k).shape[1] == 2


def test_missing_smaps_provider_raises_rather_than_falling_back():
    """Non-negotiable 3: a critic reached outside the seam must not score an RSS image."""
    d = _critic(in_channels=1)
    with pytest.raises(RuntimeError, match="smaps provider"):
        d(_kspace())


def test_a_provider_that_yields_no_maps_raises():
    """The batch-compatibility selector returns None on a stale map; that is not a licence to guess."""
    d = _critic(in_channels=1)
    d.set_smaps_provider(lambda b: None)
    with pytest.raises(RuntimeError, match="returned None"):
        d(_kspace())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"in_channels": 1, "image_repr": "rss"}, "not a known representation"),
        ({"in_channels": 4, "image_repr": "magnitude"}, "contradicts image_repr"),
        ({"in_channels": 1, "inner_critic": "not_a_registered_model"}, "not registered"),
    ],
)
def test_construction_raises_on_an_unusable_declaration(kwargs, match):
    """Every knob is validated at construction, so a typo dies before iteration 1."""
    with pytest.raises(ValueError, match=match):
        _critic(**kwargs)


def test_feature_maps_are_taken_on_the_bridged_image():
    """Feature matching compares activations; reading them off raw k-space compares domains."""
    d = _critic(in_channels=1)
    smaps, k = _smaps(), _kspace()
    d.set_smaps_provider(lambda b: smaps[:b])
    maps = d.get_feature_maps(k)
    assert maps, "no feature maps returned"
    for name, t in maps.items():
        assert not torch.is_complex(t), f"{name} is complex — the bridge was skipped"
