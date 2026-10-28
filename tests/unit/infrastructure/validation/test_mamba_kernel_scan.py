"""Tests for the static mamba_ssm route scan and the audit check that consumes it."""

from __future__ import annotations

import sys
import types
import uuid

import pytest

from spectramr.infrastructure.validation import mamba_kernel_scan
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker
from spectramr.infrastructure.validation.mamba_kernel_scan import (
    class_requires_mamba_kernel,
    mamba_kernel_route,
)
from spectramr.models.blocks import mamba_block
from spectramr.models.generators.tall_sr import TALLSuperResolution
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY


def _write_planted_module(tmp_path, monkeypatch) -> str:
    """One module: a helper builds a MambaBlock through an alias and a branch.

    Neither class name says Mamba, so only the route scan can see the kernel.
    """
    name = f"planted_kernel_{uuid.uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text(
        "import torch.nn as nn\n"
        "from spectramr.models.blocks.mamba_block import MambaBlock as _Kernel\n"
        "\n"
        "def make_block(d):\n"
        "    return _Kernel(d_model=d)\n"
        "\n"
        "class PlantedStack(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.block = make_block(8)\n"
        "\n"
        "class PlantedPlain(nn.Module):\n"
        "    def __init__(self, flag=False):\n"
        "        super().__init__()\n"
        "        if flag:\n"
        "            self.block = make_block(8)\n"
        "        else:\n"
        "            self.block = nn.Identity()\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return name


@pytest.fixture
def planted(tmp_path, monkeypatch):
    name = _write_planted_module(tmp_path, monkeypatch)
    module = __import__(name)
    yield module
    sys.modules.pop(name, None)
    mamba_kernel_scan._module_index.cache_clear()


def test_tall_sr_reaches_mamba_block_directly() -> None:
    """The observed miss: tall_sr builds MambaBlock in __init__ and its name says nothing."""
    route = mamba_kernel_route(TALLSuperResolution)
    assert route is not None
    assert route[-1] == "spectramr.models.blocks.mamba_block.MambaBlock"


def test_planted_model_reaches_kernel_through_helper_and_alias(planted) -> None:
    route = mamba_kernel_route(planted.PlantedStack)
    assert route is not None
    assert route[-1] == "spectramr.models.blocks.mamba_block.MambaBlock"


def test_model_routed_through_sibling_module_is_found() -> None:
    """geo_mamba_unet builds FiLMMambaBlock, which builds MambaBlock in another module."""
    from spectramr.models.generators.geo_mamba_unet import GeoMambaUNet

    route = mamba_kernel_route(GeoMambaUNet)
    assert route is not None
    assert "film_mamba_block" in route[1]
    assert route[-1] == "spectramr.models.blocks.mamba_block.MambaBlock"


def test_planted_branch_only_route_is_not_required(planted) -> None:
    """A kernel route in one branch of an if is config-gated, so it is not counted."""
    assert class_requires_mamba_kernel(planted.PlantedPlain) is False


def test_non_kernel_model_is_green() -> None:
    from torch import nn

    class Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(1, 1, 3)

    assert class_requires_mamba_kernel(Plain) is False


def _checker_config(model_type: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(model=types.SimpleNamespace(model_type=model_type))


@pytest.fixture
def no_kernel(monkeypatch):
    monkeypatch.setattr(mamba_block, "_mamba_ssm_importable", lambda: False)
    monkeypatch.setattr(mamba_block, "_mamba_fallback_allowed", lambda: False)


def test_audit_errors_for_tall_sr_without_kernel(no_kernel) -> None:
    populate_model_registry()
    res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(_checker_config("tall_sr"))
    assert res.passed is False
    assert res.severity == "error"
    assert "mamba_ssm" in res.message


def test_audit_errors_for_planted_model_without_kernel(planted, monkeypatch, no_kernel) -> None:
    populate_model_registry()
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "planted_stack_probe",
        {"class": planted.PlantedStack, "mode": "reconstruction", "role": "generator"},
    )
    res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(
        _checker_config("planted_stack_probe")
    )
    assert res.passed is False
    assert res.severity == "error"


def test_audit_passes_non_mamba_model(no_kernel) -> None:
    populate_model_registry()
    res = ConfigHealthChecker().check_mamba_models_require_mamba_ssm(_checker_config("unet"))
    assert res.passed is True
    assert res.severity == "info"
