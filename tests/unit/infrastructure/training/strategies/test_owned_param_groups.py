"""A strategy-built module is refused under a multi-process backend, not trained per rank (#2697)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.strategies.owned_param_groups import (
    add_owned_param_group,
)


def _strategy(backend: str | None) -> SimpleNamespace:
    parallel = None if backend is None else SimpleNamespace(strategy=backend)
    return SimpleNamespace(config=SimpleNamespace(parallel=parallel))


def _optimizer_and_group() -> tuple[torch.optim.Optimizer, dict]:
    owned = torch.nn.Linear(2, 2)
    base = torch.nn.Linear(2, 2)
    return torch.optim.SGD(base.parameters(), lr=0.1), {"params": list(owned.parameters())}


@pytest.mark.parametrize("backend", [None, "none", "dp"])
def test_single_process_backends_register_the_group(backend: str | None) -> None:
    opt, group = _optimizer_and_group()
    add_owned_param_group(_strategy(backend), opt, group, owner="its conditioner")
    assert len(opt.param_groups) == 2


@pytest.mark.parametrize("backend", ["ddp", "fsdp", "deepspeed"])
def test_multi_process_backends_refuse_and_register_nothing(backend: str) -> None:
    opt, group = _optimizer_and_group()
    with pytest.raises(ConfigurationError, match="its conditioner") as exc:
        add_owned_param_group(_strategy(backend), opt, group, owner="its conditioner")
    assert backend in str(exc.value)
    assert len(opt.param_groups) == 1


def test_a_strategy_without_a_config_is_treated_as_single_process() -> None:
    opt, group = _optimizer_and_group()
    add_owned_param_group(SimpleNamespace(), opt, group, owner="x")
    assert len(opt.param_groups) == 2


@pytest.mark.parametrize(
    "module",
    [
        "spectramr.infrastructure.training.strategies.mixins.conditioning_mixin",
        "spectramr.infrastructure.training.strategies.virtual_fiducial_strategy",
        "spectramr.infrastructure.training.strategies.diffusion",
    ],
)
def test_no_strategy_calls_add_param_group_directly(module: str) -> None:
    """The #2697 audit sites still on a bare group go through the helper (geomamba's
    uncertainty weights are now a DDP-wrapped owned module, #2629)."""
    import importlib
    import inspect

    source = inspect.getsource(importlib.import_module(module))
    assert "add_owned_param_group(" in source
    assert ".add_param_group(" not in source.replace("optimizer.add_param_group(group)", "")
