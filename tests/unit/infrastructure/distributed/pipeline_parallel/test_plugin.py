"""Single-process checks of the pipeline split: what it refuses, and the wire header."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.distributed.pipeline_parallel import transport
from spectramr.infrastructure.distributed.pipeline_parallel.plugin import (
    split_generator_for_pipeline,
)

pytestmark = [pytest.mark.unit]


def _ctx(
    precision=False, compile_=False, accumulation=1, ema=False, microbatches=1, deep=0.0, **ddp
):
    config = SimpleNamespace(
        optimization=SimpleNamespace(
            precision=SimpleNamespace(enabled=precision, dtype="float16" if precision else None),
            compile=SimpleNamespace(enabled=compile_),
            gradient=SimpleNamespace(accumulation_steps=accumulation),
        ),
        ema=SimpleNamespace(enabled=ema),
        losses=SimpleNamespace(lambda_deep_supervision=deep),
    )
    parallel = SimpleNamespace(
        strategy="ddp", mesh=SimpleNamespace(pipeline=2, pipeline_microbatches=microbatches), **ddp
    )
    mesh = SimpleNamespace(shape={"pp": 2})
    return SimpleNamespace(config=config, parallel=parallel, mesh=mesh)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"compile_": True}, "optimization.compile"),
        ({"accumulation": 2}, "accumulation_steps"),
        ({"ema": True}, "ema.enabled"),
        ({"sync_batch_norm": True}, "sync_batch_norm"),
        ({"find_unused_parameters": True}, "find_unused_parameters"),
        ({"static_graph": True}, "static_graph"),
        ({"deep": 0.5}, "lambda_deep_supervision"),
    ],
)
def test_unverified_settings_are_refused_before_any_split(kwargs, match) -> None:
    with pytest.raises(RuntimeError, match=match):
        split_generator_for_pipeline({"generator": torch.nn.Linear(2, 2)}, _ctx(**kwargs))


def test_mixed_precision_is_no_longer_refused() -> None:
    # fp16 passes the refusals and reaches the next check: the missing hook.
    with pytest.raises(RuntimeError, match="no pipeline_stages"):
        split_generator_for_pipeline({"generator": torch.nn.Linear(2, 2)}, _ctx(precision=True))


def test_a_model_without_the_hook_is_refused_and_the_hooked_ones_are_named() -> None:
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with pytest.raises(RuntimeError, match=r"no pipeline_stages.*nc_pdnet"):
        split_generator_for_pipeline({"generator": torch.nn.Linear(2, 2)}, _ctx())


def test_a_discriminator_is_refused() -> None:
    models = {"generator": torch.nn.Linear(2, 2), "discriminator": torch.nn.Linear(2, 1)}
    with pytest.raises(RuntimeError, match="discriminator"):
        split_generator_for_pipeline(models, _ctx())


def test_batchnorm_with_several_microbatches_is_refused() -> None:
    from spectramr.models.generators.unrolled_reconstruction_generator import (
        UnrolledReconstructionGenerator,
    )

    model = UnrolledReconstructionGenerator(
        in_channels=2, out_channels=2, num_unrolls=2, features=4
    )
    with pytest.raises(RuntimeError, match="BatchNorm"):
        split_generator_for_pipeline({"generator": model}, _ctx(microbatches=2))


def test_the_header_round_trips_every_entry_kind() -> None:
    payload = (
        torch.randn(2, 3, requires_grad=True),
        torch.randn(2, 1, 4, dtype=torch.complex64),
        None,
        torch.ones(5, dtype=torch.bool),
    )
    header = transport.encode_header(payload, (True, True, False, False), 4, torch.device("cpu"))
    specs, count = transport.decode_header(header)
    assert count == 4
    assert specs[2] is None
    assert [
        (s.dtype, s.shape, s.requires_grad, s.batched) for s in (specs[0], specs[1], specs[3])
    ] == [
        (torch.float32, (2, 3), True, True),
        (torch.complex64, (2, 1, 4), False, True),
        (torch.bool, (5,), False, False),
    ]
    for entry, spec in zip((payload[1], payload[3]), (specs[1], specs[3]), strict=True):
        assert torch.equal(transport.from_wire(transport.wire(entry), spec), entry)


def test_the_header_is_a_host_tensor_so_decoding_it_never_syncs() -> None:
    entry = torch.randn(2, 3)
    header = transport.encode_header((entry,), (True,), 2, torch.device("cpu"))
    assert header.device.type == "cpu" and transport.new_header().device.type == "cpu"


def test_a_host_entry_beside_an_accelerator_stage_is_flagged_and_keeps_no_grad() -> None:
    """DiffVarNet's ``restore`` (three ints made on the host) must stay on the host, so the
    last stage reads it without a device sync. The device object needs no accelerator."""
    accelerator = torch.device("cuda")
    restore = torch.tensor([2, 3, 4])
    assert transport.is_host(restore, accelerator)
    assert not transport.is_host(restore, torch.device("cpu"))
    header = transport.encode_header((None, restore), (False, False), 1, accelerator)
    spec = transport.decode_header(header)[0][1]
    assert spec.host and spec.shape == (3,)
    with pytest.raises(ValueError, match="requires grad"):
        transport.encode_header((torch.ones(2, requires_grad=True),), (True,), 1, accelerator)
