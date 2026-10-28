"""``optimization.compile.channels_last`` converts the named models and refuses empty claims."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.config.schemas.optimization import CompileConfigSchema
from spectramr.infrastructure.training.builders.memory_format import apply_channels_last


def _config(channels_last):
    return SimpleNamespace(
        optimization=SimpleNamespace(compile=SimpleNamespace(channels_last=channels_last))
    )


def test_named_model_is_converted_and_the_other_left_alone() -> None:
    gen, disc = nn.Conv2d(2, 4, 3), nn.Conv2d(2, 4, 3)
    record = apply_channels_last({"generator": gen, "discriminator": disc}, _config(["generator"]))
    assert gen.weight.is_contiguous(memory_format=torch.channels_last)
    assert not disc.weight.is_contiguous(memory_format=torch.channels_last)
    assert record == {"generator": {"parameters_converted": 1}}


def test_unset_converts_nothing() -> None:
    assert apply_channels_last({"generator": nn.Conv2d(2, 4, 3)}, _config(None)) == {}


def test_a_model_the_arm_does_not_build_raises() -> None:
    with pytest.raises(RuntimeError, match="does not build"):
        apply_channels_last({"generator": nn.Conv2d(2, 4, 3)}, _config(["discriminator"]))


def test_a_model_with_no_4d_weight_raises_instead_of_claiming_conversion() -> None:
    with pytest.raises(RuntimeError, match="no 4-D real weight"):
        apply_channels_last({"generator": nn.Linear(4, 4)}, _config(["generator"]))


def test_an_empty_list_is_refused_by_the_schema() -> None:
    with pytest.raises(ValueError, match="empty list"):
        CompileConfigSchema(channels_last=[])
    assert CompileConfigSchema(channels_last=["generator"]).channels_last == ["generator"]
