"""These critics are named for k-space and every one of them consumes an IMAGE.

``KSpaceDiscriminator`` and ``FrequencyDomainDiscriminator`` call ``fft2c`` on
their own input (``_to_kspace``); ``KSpaceAwareDiscriminator`` delegates to the
first. The k-space they score is one they MANUFACTURE, so handing them k-space
computes ``F{F{x}}`` -- the spatially reversed image. It is finite,
brain-shaped, and wrong, which is why the declaration must be read off
``forward`` and pinned here rather than inferred from the class name (#1920).

Also pins two repairs to ``KSpaceAwareDiscriminator``: a ``get_output_shape``
that counted modules where it meant levels, and a ``get_feature_maps`` that
returned ``{}``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.discriminators.kspace_discriminator import (  # noqa: E402
    FrequencyDomainDiscriminator,
    KSpaceAwareDiscriminator,
    KSpaceDiscriminator,
)
from spectramr.models.init_registry import populate_model_registry  # noqa: E402
from spectramr.models.registry import get_model_capabilities  # noqa: E402


@pytest.mark.parametrize(
    "name",
    ["kspace_discriminator", "frequency_domain_discriminator", "kspace_aware_discriminator"],
)
def test_every_critic_in_this_module_declares_the_image_domain(name):
    """The seam in ``critic_domain`` reads this; a wrong value doubles the FFT."""
    populate_model_registry()
    caps = get_model_capabilities(name)
    assert caps is not None, f"{name} is not registered"
    assert caps.input_domain == "image", (
        f"{name} declares input_domain={caps.input_domain!r}. It calls fft2c on its "
        "own input, so it consumes images; declaring 'kspace' makes the critic score "
        "F{F{x}} -- finite, brain-shaped and wrong."
    )


@pytest.mark.parametrize("cls", [KSpaceDiscriminator, FrequencyDomainDiscriminator])
def test_the_declaration_matches_what_the_class_actually_does(cls):
    """Behavioural, not textual: an image in gives a real score out.

    Feeding these an image is the declared contract; ``fft2c`` inside
    ``_to_kspace`` is what makes the declaration ``image`` rather than
    ``kspace``.
    """
    d = cls(in_channels=1, base_channels=8, num_layers=2)
    out = d(torch.randn(1, 1, 64, 64))
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("num_layers", [2, 3, 4])
@pytest.mark.parametrize("size", [64, 128])
def test_get_output_shape_matches_a_real_forward(num_layers, size):
    """Pinned against the network, never against the arithmetic.

    The arithmetic is exactly what was wrong: the shipped formula was
    ``2 ** len(self.spatial_disc.spatial_disc)``, and that length counts
    MODULES (three per level), not levels.

    ``.forward`` is called directly because ``KSpaceAwareDiscriminator`` derives
    from ``IDiscriminator``/``IModel``/``ABC`` and not ``nn.Module``, so it has
    no ``__call__`` (#801, out of scope here).
    """
    d = KSpaceAwareDiscriminator(in_channels=1, base_channels=8, num_layers=num_layers)
    real = tuple(d.forward(torch.randn(1, 1, size, size)).shape)
    assert d.get_output_shape((1, 1, size, size)) == real


def test_the_old_output_shape_formula_would_fail_this_test():
    """Anti-vacuity: the previous formula collapses the answer to zero.

    Without this, the test above could pass on an implementation that happened
    to agree for the sizes tried. ``2 ** 12`` against a 64-pixel input floors to
    ``0`` -- a shape no forward can produce, and one nothing raised on.
    """
    d = KSpaceAwareDiscriminator(in_channels=1, base_channels=8, num_layers=3)
    modules = len(d.spatial_disc.spatial_disc)
    assert modules == 3 * (d.num_layers + 1), "levels append three modules each"
    old = 64 // (2**modules)
    assert old == 0, "the shipped formula floored the output edge to zero"
    assert d.get_output_shape((1, 1, 64, 64))[-1] != old


def test_get_feature_maps_refuses_instead_of_returning_an_empty_dict():
    """An empty dict makes a feature-matching term sum to zero, every batch.

    Declared, weighted, logged, and contributing nothing -- the silent-failure
    shape non-negotiable 3 forbids. ``SenseBridgeDiscriminator`` forwards to its
    inner critic, so an arm naming this class now fails at the first step.
    """
    d = KSpaceAwareDiscriminator(in_channels=1, base_channels=8, num_layers=2)
    with pytest.raises(NotImplementedError, match="no feature maps"):
        d.get_feature_maps(torch.randn(1, 1, 64, 64))


def test_the_sense_bridge_critic_declares_the_opposite_domain():
    """The mirror image, and why one blanket rule would have inverted one of them.

    ``sense_bridge_patchgan`` is named for a bridge and consumes k-space
    (``forward`` is ``critic(bridge(x))``); the critics above are named for
    k-space and consume images.
    """
    populate_model_registry()
    caps = get_model_capabilities("sense_bridge_patchgan")
    assert caps.input_domain == "kspace"
    assert get_model_capabilities("kspace_discriminator").input_domain == "image"


# --- #801: the critic is an nn.Module and its delegate registers -------------


def test_the_aware_critic_is_an_nn_module():
    """It implemented ``IDiscriminator`` only, which carries no ``nn.Module``."""
    from torch import nn

    assert issubclass(KSpaceAwareDiscriminator, nn.Module)
    assert KSpaceAwareDiscriminator.__call__ is nn.Module.__call__


def test_the_delegate_registers_so_a_checkpoint_carries_it():
    """``spatial_disc`` is the ``KSpaceDiscriminator`` this class delegates to.

    Unregistered it was absent from ``state_dict``, unmoved by ``.to(device)``
    and invisible to the optimizer -- while ``forward`` kept working.
    """
    d = KSpaceAwareDiscriminator()
    assert "spatial_disc" in dict(d.named_children())
    assert sum(1 for _ in d.parameters()) > 0
    assert any(k.startswith("spatial_disc.") for k in d.state_dict())
