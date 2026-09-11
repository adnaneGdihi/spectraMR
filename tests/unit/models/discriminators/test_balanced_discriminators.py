"""``RealESRGANDiscriminator`` holds a critic; until #801 it did not register it.

The class implemented ``IDiscriminator`` -- which resolves to ``IModel(ABC)``,
carrying no ``nn.Module`` -- so ``self.backbone`` was an ordinary attribute.
``state_dict()`` was empty, ``.to(device)`` moved nothing, and an optimizer
handed ``model.parameters()`` received zero tensors. Nothing raised: the module
worked in the forward direction and simply did not train.

The second contract here is #1957: the interface import must not be able to
change the base class. Under ``except ImportError: IDiscriminator = nn.Module``
this source defined an ``nn.Module`` subclass when the import failed and a
non-module when it succeeded, with no way to tell which one you had.

CPU-only, tiny tensors.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from spectramr.models.discriminators import balanced_discriminators as mod  # noqa: E402
from spectramr.models.interfaces import IDiscriminator  # noqa: E402


def test_the_critic_is_an_nn_module() -> None:
    assert issubclass(mod.RealESRGANDiscriminator, nn.Module)


def test_the_backbone_registers_as_a_submodule() -> None:
    """The defect, stated as its consequence rather than as a base-class list."""
    d = mod.RealESRGANDiscriminator()
    assert "backbone" in dict(d.named_children())
    assert sum(1 for _ in d.parameters()) > 0, "an optimizer would receive no tensors"
    assert any(k.startswith("backbone.") for k in d.state_dict()), (
        "the backbone's weights must survive a checkpoint round trip"
    )


def test_forward_still_returns_one_score_per_sample() -> None:
    """The class pools to ``[B]`` on purpose -- pin that, not a feature map."""
    d = mod.RealESRGANDiscriminator()
    out = d(torch.randn(3, 1, 64, 64))
    assert out.shape == (3,)
    assert bool(((out >= 0.0) & (out <= 1.0)).all()), "use_sigmoid=True by default"


def test_call_is_not_shadowed() -> None:
    """A hand-written ``__call__`` would bypass hooks and never reach ``forward``."""
    assert mod.RealESRGANDiscriminator.__call__ is nn.Module.__call__


def test_the_interface_import_cannot_swap_the_base_class() -> None:
    """#1957: the module-level name is the real interface, never ``nn.Module``."""
    assert mod.IDiscriminator is IDiscriminator
    assert mod.IDiscriminator is not nn.Module
