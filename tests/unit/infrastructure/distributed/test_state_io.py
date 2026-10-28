"""``state_io``: the passthrough every non-FSDP strategy uses, and how it is resolved."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.distributed.checkpoint_adapters import (
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
    FSDPCheckpointAdapter,
)
from spectramr.infrastructure.distributed.state_io import (
    FSDPStateIO,
    PassthroughStateIO,
    state_io_of,
)

pytestmark = pytest.mark.unit


class TestResolution:
    def test_no_runtime_is_the_passthrough(self) -> None:
        assert type(state_io_of(None)) is PassthroughStateIO

    def test_runtime_without_an_adapter_is_the_passthrough(self) -> None:
        assert type(state_io_of(SimpleNamespace())) is PassthroughStateIO

    def test_the_fsdp_adapter_carries_the_gathering_io(self) -> None:
        runtime = SimpleNamespace(checkpoint_adapter=FSDPCheckpointAdapter())
        assert isinstance(state_io_of(runtime), FSDPStateIO)

    @pytest.mark.parametrize("adapter", [DefaultCheckpointAdapter(), DeepSpeedCheckpointAdapter()])
    def test_replicated_and_engine_adapters_do_not_gather(self, adapter) -> None:
        assert type(adapter.state_io) is PassthroughStateIO


class TestPassthrough:
    def test_wrapped_keys_are_stripped_on_load(self) -> None:
        io = PassthroughStateIO()
        model = nn.Linear(2, 2)
        prefixed = {f"module.{k}": v.clone() + 1 for k, v in model.state_dict().items()}
        io.load_model_state(model, prefixed)
        assert torch.allclose(model.weight, prefixed["module.weight"])

    def test_full_params_is_a_no_op(self) -> None:
        with PassthroughStateIO().full_params(nn.Linear(2, 2)):
            pass

    def test_swap_in_installs_the_shadow_and_restores_the_live_weights(self) -> None:
        io = PassthroughStateIO()
        live, shadow = nn.Linear(2, 2), nn.Linear(2, 2)
        before = live.weight.detach().clone()
        with io.swap_in(live, shadow):
            assert torch.allclose(live.weight, shadow.weight)
        assert torch.allclose(live.weight, before)


class TestFSDPIOLeavesNonRootsAlone:
    """A plain discriminator beside a wrapped generator: the FSDP contexts raise on it."""

    def test_a_plain_module_takes_the_passthrough_path(self) -> None:
        io = FSDPStateIO()
        model = nn.Linear(2, 2)
        opt = torch.optim.Adam(model.parameters())
        state = io.model_state(model)
        io.load_model_state(model, state)
        io.load_optimizer_state(model, opt, io.optimizer_state(model, opt))
        with io.full_params(model), io.swap_in(model, nn.Linear(2, 2)):
            pass
