"""``GeneratorBuilder`` must tell author-declared kwargs from forwarded ones.

``build()`` deliberately forwards every top-level ``model.*`` field the
constructor *might* accept, relying on the factory's signature filter to drop
what does not fit. That best-effort forward and an arm's own
``model.model_kwargs`` are indistinguishable by the time they reach the factory
-- so the builder captures the author-declared subset **before** the forward and
passes it down as ``_declared_keys``.

Without that split, the ``**kwargs`` recording added for issue #878 fires once
per forwarded schema field per arm, burying the ``model_kwargs`` entries that
are the actual signal.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from spectramr.infrastructure.builders.leaf.model_builders import GeneratorBuilder


def _config(**model_fields):
    """A stand-in config whose ``model`` block exposes ``model_fields``.

    Deliberately a pydantic-shaped stub rather than a bare SimpleNamespace: the
    builder iterates ``type(model_cfg).model_fields``, so a stub without that
    attribute would make the forward loop silently no-op and the test would pass
    for the wrong reason.
    """
    from pydantic import BaseModel

    class _Model(BaseModel):
        model_config = {"extra": "allow"}
        spatial_dims: int | None = None
        base_channels: int | None = None

    return SimpleNamespace(
        model=_Model(**model_fields),
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(enable_checkpointing=False)
        ),
    )


@pytest.fixture
def captured_call():
    """Capture the kwargs `create_generator` is called with."""
    seen = {}

    def _create_generator(**kwargs):
        seen.update(kwargs)
        return MagicMock()

    factory = MagicMock()
    factory.create_generator.side_effect = _create_generator
    with patch(
        "spectramr.models.factories.model_factory.ModelFactory", return_value=factory
    ):
        yield seen


def test_declared_keys_holds_only_author_written_model_kwargs(captured_call):
    config = _config(spatial_dims=3)
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("depth", 4)
    )
    builder.validate().build()

    assert captured_call["_declared_keys"] == frozenset({"depth"}), (
        "only `depth` was author-written; `spatial_dims` was forwarded from the "
        "top-level model block on a best-effort basis"
    )


def test_forwarded_field_still_reaches_the_factory(captured_call):
    """The split must not change WHAT is forwarded, only what is attributed."""
    config = _config(spatial_dims=3)
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("depth", 4)
    )
    builder.validate().build()

    assert captured_call["spatial_dims"] == 3
    assert captured_call["depth"] == 4


def test_in_and_out_channels_are_not_counted_as_declared(captured_call):
    """They are stripped from kwargs to avoid a duplicate-keyword TypeError, so
    they must not appear in the declared set either."""
    config = _config()
    builder = (
        GeneratorBuilder(config, device="cpu")
        .with_architecture("unet")
        .with_input_channels(2)
        .with_output_channels(2)
        .with_parameter("in_channels", 99)
    )
    builder.validate().build()

    assert "in_channels" not in captured_call["_declared_keys"]


class TestWithCheckpointReadsTheRunStatePart:
    """``with_checkpoint`` loads the named RunState part strictly into the built model."""

    @staticmethod
    def _factory(module):
        factory = MagicMock()
        factory.create_generator.return_value = module
        factory.create_discriminator.return_value = module
        return patch("spectramr.models.factories.model_factory.ModelFactory", return_value=factory)

    @staticmethod
    def _run_state(tmp_path):
        import torch

        from tests.fixtures.run_state_files import write_run_state

        return write_run_state(
            tmp_path / "best.pt",
            {
                "generator": {"weight": torch.full((2, 2), 3.0)},
                "discriminator": {"weight": torch.full((1, 2), 7.0)},
            },
        )

    def test_the_generator_part_loads(self, tmp_path):
        import torch
        from torch import nn

        path = self._run_state(tmp_path)
        with self._factory(nn.Linear(2, 2, bias=False)):
            built = (
                GeneratorBuilder(_config(), device="cpu")
                .with_architecture("unet")
                .with_input_channels(2)
                .with_output_channels(2)
                .with_checkpoint(str(path))
                .build()
            )
        assert torch.equal(built.weight, torch.full((2, 2), 3.0))

    def test_the_discriminator_part_loads(self, tmp_path):
        import torch
        from torch import nn

        from spectramr.infrastructure.builders.leaf.model_builders import DiscriminatorBuilder

        path = self._run_state(tmp_path)
        with self._factory(nn.Linear(2, 1, bias=False)):
            built = (
                DiscriminatorBuilder(_config(), device="cpu")
                .with_architecture("patch_gan_2d")
                .with_input_channels(2)
                .with_checkpoint(str(path))
                .build()
            )
        assert torch.equal(built.weight, torch.full((1, 2), 7.0))

    def test_a_mismatched_part_raises(self, tmp_path):
        from torch import nn

        path = self._run_state(tmp_path)
        with self._factory(nn.Linear(3, 3, bias=False)), pytest.raises(RuntimeError):
            (
                GeneratorBuilder(_config(), device="cpu")
                .with_architecture("unet")
                .with_input_channels(2)
                .with_output_channels(2)
                .with_checkpoint(str(path))
                .build()
            )
