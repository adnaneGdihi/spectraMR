"""An unknown option value raises and names the key; a known one parses case-insensitively."""

from __future__ import annotations

import pytest

from spectramr.models.generators.option_parsing import parse_enum_option
from spectramr.models.reconstruction.unet import BlockType, NormalizationType


@pytest.mark.parametrize(
    ("enum_cls", "raw", "expected"),
    [
        (NormalizationType, "group", NormalizationType.GROUP),
        (NormalizationType, "INSTANCE", NormalizationType.INSTANCE),
        (BlockType, "Bottleneck", BlockType.BOTTLENECK),
    ],
)
def test_a_known_value_parses(enum_cls, raw, expected) -> None:
    assert parse_enum_option(enum_cls, raw, "x") is expected


@pytest.mark.parametrize(
    ("enum_cls", "raw", "key"),
    [
        (
            NormalizationType,
            "groupnorm",
            "norm_type",
        ),  # a plausible typo the old code mapped to GROUP
        (BlockType, "residual", "block_type"),  # ...and to STANDARD
        (NormalizationType, 3, "norm_type"),  # not a string at all
        (BlockType, None, "block_type"),
    ],
)
def test_an_unknown_value_raises_and_names_the_key(enum_cls, raw, key) -> None:
    with pytest.raises(ValueError, match=key):
        parse_enum_option(enum_cls, raw, key)
