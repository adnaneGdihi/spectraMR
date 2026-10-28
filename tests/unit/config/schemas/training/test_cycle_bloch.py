"""Schema tests for ``training.cycle_bloch``."""

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.cycle_bloch import CycleBlochConfig

_SEQ = {"ulf_tr_ms": 500.0, "ulf_te_ms": 14.0, "ulf_flip_deg": 90.0, "tissue_field_tesla": 3.0}


def test_default_is_bloch_fixed():
    assert CycleBlochConfig().backward_model == "bloch_fixed"


def test_ulf_forward_with_full_sequence_loads():
    cfg = CycleBlochConfig(backward_model="ulf_forward", **_SEQ)
    assert cfg.ulf_flip_deg == 90.0


@pytest.mark.parametrize("missing", sorted(_SEQ))
def test_ulf_forward_requires_every_sequence_field(missing):
    seq = {k: v for k, v in _SEQ.items() if k != missing}
    with pytest.raises(ValidationError, match=missing):
        CycleBlochConfig(backward_model="ulf_forward", **seq)


def test_bloch_fixed_rejects_sequence_fields_it_would_ignore():
    with pytest.raises(ValidationError, match="ulf_tr_ms"):
        CycleBlochConfig(backward_model="bloch_fixed", ulf_tr_ms=300.0)


def test_unknown_backward_model_and_keys_are_rejected():
    with pytest.raises(ValidationError):
        CycleBlochConfig(backward_model="epg")
    with pytest.raises(ValidationError):
        CycleBlochConfig(critic="wgan_gp")


_EPG = {
    "backward_model": "epg_tse",
    "ulf_tr_ms": 500.0,
    "ulf_te_ms": 14.0,
    "echo_spacing_ms": 7.0,
    "echo_train_length": 4,
    "refocus_deg": 150.0,
}


def test_epg_tse_loads_and_rejects_ulf_forward_fields():
    assert CycleBlochConfig(**_EPG).echo_train_length == 4
    with pytest.raises(ValidationError, match="ulf_flip_deg"):
        CycleBlochConfig(**_EPG, ulf_flip_deg=90.0)
    with pytest.raises(ValidationError, match="refocus_deg"):
        CycleBlochConfig(**{k: v for k, v in _EPG.items() if k != "refocus_deg"})


def test_estimated_b1_needs_a_flip_angle_model():
    assert CycleBlochConfig(**_EPG, b1_plus="estimated", b1_tv_weight=0.1).b1_tv_weight == 0.1
    with pytest.raises(ValidationError, match="bloch_fixed has no flip angle"):
        CycleBlochConfig(b1_plus="estimated")


def test_b1_tv_weight_without_an_estimated_map_is_rejected():
    with pytest.raises(ValidationError, match="b1_tv_weight"):
        CycleBlochConfig(**_EPG, b1_tv_weight=0.1)


_T2W_ERNST = {"T2w": {"tr_ms": 2000.0, "te_ms": 194.8, "flip_deg": 90.0}}
_T2W_EPG = {
    "tr_ms": 2000.0,
    "te_ms": 194.8,
    "echo_spacing_ms": 4.87,
    "echo_train_length": 80,
    "refocus_deg": 180.0,
}


def test_sequence_table_loads_with_an_optional_inversion():
    seqs = {
        **_T2W_ERNST,
        "FLAIR": {"tr_ms": 3500.0, "te_ms": 175.1, "ti_ms": 1290.0, "flip_deg": 90.0},
    }
    cfg = CycleBlochConfig(backward_model="ulf_forward", tissue_field_tesla=3.0, sequences=seqs)
    assert cfg.sequences["FLAIR"].ti_ms == 1290.0 and cfg.sequences["T2w"].ti_ms is None


def test_sequence_table_rejects_a_fixed_model_scalar_fields_and_unread_fields():
    with pytest.raises(ValidationError, match="fixed"):
        CycleBlochConfig(backward_model="bloch_fixed", sequences=_T2W_ERNST)
    with pytest.raises(ValidationError, match="not read"):
        CycleBlochConfig(
            backward_model="ulf_forward",
            tissue_field_tesla=3.0,
            ulf_tr_ms=500.0,
            sequences=_T2W_ERNST,
        )
    with pytest.raises(ValidationError, match="unread"):
        CycleBlochConfig(
            backward_model="epg_tse", sequences={"T2w": {**_T2W_EPG, "flip_deg": 90.0}}
        )
    with pytest.raises(ValidationError, match="tissue_field_tesla"):
        CycleBlochConfig(backward_model="ulf_forward", sequences=_T2W_ERNST)
    with pytest.raises(ValidationError, match="empty"):
        CycleBlochConfig(backward_model="epg_tse", sequences={})
