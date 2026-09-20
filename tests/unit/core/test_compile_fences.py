"""The dynamo fence.

Inductor cannot codegen complex operators. It does not fail on one -- it routes
to an eager fallback and warns once per process -- so a complex arm that
declares compilation gets a run reporting a configuration it did not execute.
Fencing the physics SSOT is what makes the compiled graph provably complex-free.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from spectramr.core.compile_fences import dynamo_disable


def _double(x):
    return x * 2


class TestUnderRealTorch:
    def test_returns_a_working_callable(self):
        fenced = dynamo_disable(_double)
        assert callable(fenced)
        assert fenced(3) == 6

    def test_it_actually_fences(self):
        pytest.importorskip("torch")
        assert dynamo_disable(_double) is not _double

    def test_the_name_survives(self):
        """Tracebacks and logs must still name the physics op."""
        assert dynamo_disable(_double).__name__ == "_double"


class TestUnderTheTorchShim:
    """The planted violation.

    ``tests/conftest.py`` installs a ``MagicMock`` for torch on the torch-less
    lane. A bare ``@torch._dynamo.disable`` there returns **another mock**, so
    the module imports, the symbol exists, and every call to a fenced physics op
    returns a mock instead of a tensor. ``callable()`` does not catch it --
    a MagicMock is callable.
    """

    @pytest.fixture
    def mocked_torch(self, monkeypatch):
        import importlib

        import spectramr.core.compile_fences as module

        monkeypatch.setitem(sys.modules, "torch", MagicMock())
        importlib.reload(module)
        yield module
        monkeypatch.undo()
        importlib.reload(module)

    def test_the_original_function_comes_back(self, mocked_torch):
        assert mocked_torch.dynamo_disable(_double) is _double

    def test_it_still_computes(self, mocked_torch):
        """The failure this guards against is silent: a mock returns a mock,
        and the arm trains on garbage rather than crashing."""
        assert mocked_torch.dynamo_disable(_double)(3) == 6


class TestWhenTorchIsAbsent:
    def test_an_unimportable_torch_leaves_the_function_alone(self, monkeypatch):
        import importlib

        import spectramr.core.compile_fences as module

        monkeypatch.setitem(sys.modules, "torch", None)
        importlib.reload(module)
        try:
            assert module.dynamo_disable(_double) is _double
        finally:
            monkeypatch.undo()
            importlib.reload(module)


class TestThePhysicsSSOTIsFenced:
    def test_fft2c_and_ifft2c_carry_the_fence(self):
        """These are the modules that carry complex dtype, and the reason the
        complex opt-out can be declared honestly."""
        import pathlib

        import spectramr.infrastructure.physics.fft_ops as mod

        source = pathlib.Path(mod.__file__).read_text()
        for fn in ("fft2c", "ifft2c"):
            assert f"@dynamo_disable\ndef {fn}(" in source, f"{fn} is not fenced"

    def test_the_round_trip_is_unchanged(self):
        """A fence must not alter numerics -- it only decides what dynamo sees."""
        torch = pytest.importorskip("torch")
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        x = torch.randn(1, 1, 8, 8, dtype=torch.complex64)
        assert (ifft2c(fft2c(x)) - x).abs().max().item() < 1e-5
