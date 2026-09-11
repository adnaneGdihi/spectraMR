"""``kan_gan``'s discriminator getter: no dead guard, and the knobs arrive (#1943).

The defect was a ``try``/``except ImportError`` whose two branches imported the
*identical* symbol, so the handler could only re-raise what it was written to
absorb. It read like the real optional-dependency fallbacks elsewhere in the
package (non-negotiable 18), which is what made it cost attention on every pass.

Deleting it exposed the second half: ``**kwargs`` was accepted and dropped, so
``ndf`` / ``n_layers`` / ``spectral_norm`` -- three real ``PatchGANDiscriminator``
knobs -- were unreachable through this getter (non-negotiable 8). The forwarding
tests below are red on the unfixed builder for that reason, not for the guard.
"""

import ast
import inspect

import pytest

from spectramr.models.discriminators.patchgan_discriminator import PatchGANDiscriminator
from spectramr.models.gans import kan_gan


def _get_discriminator_ast() -> ast.FunctionDef:
    tree = ast.parse(inspect.getsource(kan_gan))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_discriminator":
            return node
    pytest.fail("get_discriminator is not a module-level function in kan_gan")


def test_get_discriminator_holds_no_try_block():
    """The guard is gone -- asserted on the AST, not on the source text.

    An absence pin on a node type cannot be satisfied by a docstring or a comment
    that merely mentions ``try``.
    """
    fn = _get_discriminator_ast()
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Try)]


def test_get_discriminator_returns_a_patchgan():
    assert isinstance(kan_gan.get_discriminator(), PatchGANDiscriminator)


def test_get_discriminator_forwards_every_patchgan_knob():
    """Each knob is probed with a value that is NOT its default.

    Probing ``ndf=64`` would pass whether or not the argument was forwarded.
    """
    d = kan_gan.get_discriminator(1, ndf=16, n_layers=2, spectral_norm=False)
    assert (d.ndf, d.n_layers, d.spectral_norm) == (16, 2, False)


def test_unknown_kwarg_raises_rather_than_being_discarded():
    with pytest.raises(TypeError):
        kan_gan.get_discriminator(1, not_a_patchgan_knob=3)


def test_kan_discriminator_wrapper_forwards_too():
    """The wrapper at ``KANDiscriminator.__init__`` splats ``**kwargs`` into the
    getter, so the knobs have to survive both hops to be reachable from a config.
    """
    wrapper = kan_gan.KANDiscriminator(in_channels=1, ndf=16)
    assert wrapper.model.ndf == 16
