"""The ``compression:`` block validates its enums and refuses unread knobs."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.compression import CompressionExportConfigSchema
from spectramr.config.schemas.enums import CompressionPruning, CompressionQuantization


def test_valid_block_parses_to_enums():
    cfg = CompressionExportConfigSchema.model_validate(
        {
            "enabled": True,
            "quantization": {"method": "int8_weight"},
            "pruning": {"method": "l1_global", "amount": 0.3},
        }
    )
    assert cfg.quantization.method is CompressionQuantization.INT8_WEIGHT
    assert cfg.pruning.method is CompressionPruning.L1_GLOBAL


@pytest.mark.parametrize(
    "block",
    [
        {"enabled": True, "quantization": {"method": "int2_weight"}},
        {"enabled": True, "pruning": {"method": "structured", "amount": 0.2}},
        {"enabled": True},  # no method: would re-score unchanged weights
        {"enabled": True, "pruning": {"method": "l1_global"}},  # amount missing
        {"pruning": {"method": "none", "amount": 0.5}},  # unread amount
        {"enabled": True, "quantization": {"method": "int8_weight"}, "bits": 8},  # extra key
    ],
)
def test_invalid_blocks_raise(block):
    with pytest.raises(ValidationError):
        CompressionExportConfigSchema.model_validate(block)


def test_disabled_default_is_inert():
    assert CompressionExportConfigSchema().enabled is False
