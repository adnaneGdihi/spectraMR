"""``analyze_residuals`` raises on an absent or failing pysr instead of warning."""

from __future__ import annotations

import sys
import types

import pytest
import torch

from spectramr.models.generators import symbolic_regression_wrapper as srw


class _Identity(torch.nn.Module):
    def forward(self, x):
        return x


def _wrapper():
    w = srw.SymbolicRegressionWrapper.__new__(srw.SymbolicRegressionWrapper)
    torch.nn.Module.__init__(w)
    w.generator = _Identity()
    w.logger = srw._logger
    return w


def test_absent_pysr_raises(monkeypatch):
    monkeypatch.setattr(srw, "HAVE_PYSR", False)
    with pytest.raises(ImportError, match="pysr"):
        _wrapper().analyze_residuals(torch.zeros(2, 4), torch.ones(2, 4))


def test_pysr_failure_propagates(monkeypatch):
    class _Failing:
        def __init__(self, **kwargs):
            pass

        def fit(self, x, y):
            raise RuntimeError("julia backend crashed")

    monkeypatch.setattr(srw, "HAVE_PYSR", True)
    monkeypatch.setitem(sys.modules, "pysr", types.SimpleNamespace(PySRRegressor=_Failing))
    with pytest.raises(RuntimeError, match="julia backend crashed"):
        _wrapper().analyze_residuals(torch.zeros(2, 4), torch.ones(2, 4))
