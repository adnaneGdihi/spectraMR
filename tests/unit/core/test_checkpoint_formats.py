from typing import get_args

from spectramr.core.checkpoint_formats import CHECKPOINT_FORMATS, CheckpointFormat


def test_the_tuple_is_the_literal() -> None:
    assert CHECKPOINT_FORMATS == get_args(CheckpointFormat) == ("safetensors", "pt", "pth")


def test_layout_uses_the_core_tuple() -> None:
    from spectramr.infrastructure.checkpointing import layout

    assert layout.FORMATS is CHECKPOINT_FORMATS


def test_the_default_format_has_one_owner() -> None:
    """The schema, the converter and the drain read the core default; layout keeps no copy."""
    import inspect

    from spectramr.config.schemas.checkpoint import CheckpointConfigSchema
    from spectramr.core.checkpoint_formats import DEFAULT_CHECKPOINT_FORMAT
    from spectramr.infrastructure.checkpointing import layout
    from tests.unit.scripts.convert_v1_helpers import converter

    assert DEFAULT_CHECKPOINT_FORMAT in CHECKPOINT_FORMATS
    assert CheckpointConfigSchema.model_fields["format"].default == DEFAULT_CHECKPOINT_FORMAT
    assert not hasattr(layout, "DEFAULT_FORMAT")
    source = inspect.getsource(converter().main)
    assert "default=DEFAULT_CHECKPOINT_FORMAT" in source
    assert 'default="safetensors"' not in source
