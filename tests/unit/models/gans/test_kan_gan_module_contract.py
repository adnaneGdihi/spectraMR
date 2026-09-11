"""``kan_gan``'s two wrappers are ``nn.Module``s, and neither shadows ``__call__``.

Both were plain classes holding a live network in ``self.model``, registered in
``MODEL_REGISTRY`` (and, for the generator, in ``ModelFactory``'s ``"kan"`` /
``"kan_gan"`` dispatch) as if they were models. The submodule never registered,
so ``state_dict()`` was empty and ``parameters()`` yielded nothing.

Both also defined a ``__call__`` whose body was identical to their ``forward``.
That is harmless on a plain class and destructive on an ``nn.Module``: a
hand-written ``__call__`` shadows ``nn.Module._call_impl``, so ``forward``
becomes unreachable and no hook fires. Fixing the base without deleting the
facade would have produced a registered model whose hooks silently never ran
(non-negotiable 16). The hook assertion below is what makes that observable
rather than assumed.

Separate file from ``test_kan_gan.py`` on purpose: that one owns the discriminator
getter's import guard and knob forwarding (#1943), this one owns the module
contract (#801). They share a source file and no invariant.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from spectramr.models.gans import kan_gan  # noqa: E402


@pytest.mark.parametrize("name", ["KANGenerator", "KANDiscriminator"])
def test_the_wrapper_is_an_nn_module(name: str) -> None:
    assert issubclass(getattr(kan_gan, name), nn.Module)


@pytest.mark.parametrize("name", ["KANGenerator", "KANDiscriminator"])
def test_the_wrapper_does_not_shadow_call(name: str) -> None:
    """``vars`` rather than ``getattr``: the inherited one is what we want."""
    cls = getattr(kan_gan, name)
    assert "__call__" not in vars(cls)
    assert cls.__call__ is nn.Module.__call__


def test_the_discriminators_network_registers() -> None:
    d = kan_gan.KANDiscriminator()
    assert "model" in dict(d.named_children())
    assert sum(1 for _ in d.parameters()) > 0
    assert any(k.startswith("model.") for k in d.state_dict())


def test_forward_is_reached_through_call_and_hooks_fire() -> None:
    """Observed, not inferred: the hook is the evidence ``forward`` ran."""
    d = kan_gan.KANDiscriminator()
    seen: list[tuple[int, ...]] = []
    d.register_forward_hook(lambda _m, _i, out: seen.append(tuple(out.shape)))
    out = d(torch.randn(2, 1, 64, 64))
    assert seen == [tuple(out.shape)]


def test_the_generator_still_cannot_be_constructed_pending_1998() -> None:
    """Pins the blocker so it announces itself when it is lifted.

    ``KANGenerator`` passes ``norm_layer`` to ``KANU_Net``, which does not accept
    it (#1998). The base-class fix above is therefore verified at the type level
    only for this class. When #1998 lands this test goes red, and whoever fixes it
    is told here that the registration check is still owed.
    """
    with pytest.raises(TypeError, match="norm_layer"):
        kan_gan.KANGenerator()
