"""Tests for :mod:`spectramr.infrastructure.logging.torch_logging`."""

from __future__ import annotations

import logging
import sys

import pytest

from spectramr.infrastructure.logging.torch_logging import apply_level_to_torch


@pytest.fixture(autouse=True)
def _no_torch_logs(monkeypatch):
    monkeypatch.delenv("TORCH_LOGS", raising=False)


class TestApplyLevelToTorch:
    def test_noop_when_torch_is_not_loaded(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "torch", raising=False)
        apply_level_to_torch(logging.WARNING)  # must not import torch
        assert "torch" not in sys.modules

    def test_explicit_torch_logs_is_left_alone(self, monkeypatch):
        import types

        fake = types.ModuleType("torch._logging")
        calls = []
        fake.set_logs = lambda **kw: calls.append(kw)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "torch._logging", fake)
        monkeypatch.setenv("TORCH_LOGS", "+dynamo")
        apply_level_to_torch(logging.WARNING)
        assert calls == []

    def test_level_is_pushed_onto_torch_logging(self, monkeypatch):
        import types

        fake = types.ModuleType("torch._logging")
        calls = []
        fake.set_logs = lambda **kw: calls.append(kw)
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "torch._logging", fake)
        apply_level_to_torch(logging.WARNING)
        assert calls == [{"all": logging.WARNING}]

    def test_a_torch_without_the_logging_module_is_reported_not_inferred(self, monkeypatch, caplog):
        import types

        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
        monkeypatch.setitem(sys.modules, "torch._logging", None)  # makes the import raise
        with caplog.at_level(logging.WARNING):
            apply_level_to_torch(logging.WARNING)
        assert "torch._logging unavailable" in caplog.text
