"""Tests for the data.m4raw_triplet block (exp 32, gain-calibrated maps)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.data import CANONICAL_DATASET_TYPES, DataConfigSchema
from spectramr.config.schemas.m4raw_triplet import M4RawTripletConfigSchema

_BLOCK = {
    "t2_registered_root": "databases/m4raw/nex_averaged_registered/T2_to_T1",
    "flair_registered_root": "databases/m4raw/nex_averaged_registered/FLAIR_to_T1",
    "train_split": "multicoil_train/multicoil_train",
    "val_split": "multicoil_val/multicoil_val",
}


def test_a_complete_block_loads_and_is_frozen() -> None:
    block = M4RawTripletConfigSchema(**_BLOCK)
    assert block.train_split == _BLOCK["train_split"]
    with pytest.raises(ValidationError):
        block.train_split = "other"


def test_an_unknown_key_is_refused() -> None:
    with pytest.raises(ValidationError):
        M4RawTripletConfigSchema(**_BLOCK, flair_root="typo")


def test_a_missing_root_is_refused() -> None:
    with pytest.raises(ValidationError):
        M4RawTripletConfigSchema(**{k: v for k, v in _BLOCK.items() if k != "t2_registered_root"})


def test_the_repetition_root_is_optional_and_off_by_default() -> None:
    assert M4RawTripletConfigSchema(**_BLOCK).raw_repetition_root is None
    block = M4RawTripletConfigSchema(**_BLOCK, raw_repetition_root="databases/m4raw/data")
    assert block.raw_repetition_root == "databases/m4raw/data"


def test_the_type_is_canonical_and_the_block_hangs_off_data() -> None:
    assert "m4raw_triplet" in CANONICAL_DATASET_TYPES
    assert "m4raw_triplet" in DataConfigSchema.model_fields
