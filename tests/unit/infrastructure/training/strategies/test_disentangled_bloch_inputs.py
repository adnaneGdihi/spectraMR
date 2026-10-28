"""Tests for disentangled_bloch_inputs: per-sample Bloch inputs and the shared decode (#2611)."""

from __future__ import annotations

import pytest
import torch

from spectramr.data.batch_types import BatchAdapter
from spectramr.infrastructure.training.strategies.disentangled_bloch_inputs import (
    acquisition_params,
    check_flip_angle_is_read,
    check_protocol_matches_families,
    parse_protocol_map,
    parse_sequence_map,
    sequence_codes,
)

_MAP = {"T1": 0, "T2": 0, "FLAIR": 1}
_CPU = torch.device("cpu")


def _batch(contrasts: list[str]):
    x = torch.zeros(len(contrasts), 1, 4, 4)
    return BatchAdapter.from_dict({"input": x, "target": x, "contrast": contrasts})


def test_parse_sequence_map_accepts_families_case_insensitively() -> None:
    assert parse_sequence_map({"t1": "se", "FLAIR": "IR"}) == {"T1": 0, "FLAIR": 1}


def test_parse_sequence_map_rejects_an_unknown_family() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        parse_sequence_map({"T1": "MPRAGE"})


def test_parse_sequence_map_rejects_a_non_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        parse_sequence_map(["T1", "SE"])


def test_sequence_codes_are_per_sample() -> None:
    codes = sequence_codes(_batch(["T1", "FLAIR"]), _MAP, 2, _CPU)
    assert codes.view(-1).tolist() == [0, 1]
    assert codes.shape == (2, 1, 1, 1)


def test_sequence_codes_expand_over_depth() -> None:
    codes = sequence_codes(_batch(["T1", "FLAIR"]), _MAP, 4, _CPU)
    assert codes.view(-1).tolist() == [0, 0, 1, 1]


def test_an_explicit_contrast_type_tensor_wins() -> None:
    batch = {"contrast_type": torch.tensor([1, 0]), "contrast": ["T1", "T1"]}
    assert sequence_codes(batch, _MAP, 2, _CPU).view(-1).tolist() == [1, 0]


def test_an_unknown_contrast_raises_naming_it() -> None:
    with pytest.raises(ValueError, match="'PD'"):
        sequence_codes(_batch(["PD"]), _MAP, 1, _CPU)


def test_a_batch_without_contrast_raises() -> None:
    with pytest.raises(ValueError, match="neither 'contrast_type' nor 'contrast'"):
        sequence_codes((torch.zeros(1), torch.zeros(1)), _MAP, 1, _CPU)


def test_codes_that_cannot_cover_the_rows_raise() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        sequence_codes(_batch(["T1", "FLAIR"]), _MAP, 3, _CPU)


# --- the acquisition protocol of each contrast (#2618) ---------------------------------

_PROTOCOL = {
    "T1": {"TR": 500.0, "TE": 18.4},
    "T2": {"TR": 5500.0, "TE": 128.0},
    "FLAIR": {"TR": 7500.0, "TE": 98.0, "TI": 1655.0},
}
_FAMILIES = {"T1": 0, "T2": 0, "FLAIR": 1}  # SE, SE, IR


class TestParseProtocolMap:
    def test_keys_are_case_insensitive_and_ti_defaults_to_zero(self) -> None:
        out = parse_protocol_map({"t1": {"tr": 500, "te": 18.4}, "FLAIR": _PROTOCOL["FLAIR"]})
        assert out["T1"] == {"TR": 500.0, "TE": 18.4, "TI": 0.0}
        assert out["FLAIR"]["TI"] == 1655.0

    def test_absent_declares_nothing(self) -> None:
        assert parse_protocol_map(None) == {}

    @pytest.mark.parametrize(
        ("entry", "match"),
        [
            ({"TR": 500.0}, "needs TR and TE"),
            ({"TE": 18.4}, "needs TR and TE"),
            ({"TR": 500.0, "TE": 18.4, "TA": 3.0}, "unknown key"),
            ({"TR": 0.0, "TE": 18.4}, "must be positive"),
            ({"TR": 500.0, "TE": -1.0}, "must be positive"),
            ({"TR": 500.0, "TE": 18.4, "TI": 0.0}, "TI must be positive"),
        ],
    )
    def test_an_ill_formed_entry_raises(self, entry: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            parse_protocol_map({"T1": entry})

    def test_a_non_mapping_raises(self) -> None:
        with pytest.raises(TypeError, match="mapping"):
            parse_protocol_map([("T1", 500)])
        with pytest.raises(TypeError, match="mapping"):
            parse_protocol_map({"T1": 500.0})


class TestProtocolAgreesWithFamilies:
    def test_the_m4raw_protocol_agrees(self) -> None:
        check_protocol_matches_families(parse_protocol_map(_PROTOCOL), _FAMILIES)

    def test_a_declared_contrast_without_a_protocol_raises(self) -> None:
        with pytest.raises(ValueError, match="no entry in bloch_protocol_by_contrast"):
            check_protocol_matches_families(parse_protocol_map({"T1": _PROTOCOL["T1"]}), _FAMILIES)

    def test_an_inversion_recovery_contrast_needs_an_inversion_time(self) -> None:
        bad = {**_PROTOCOL, "FLAIR": {"TR": 7500.0, "TE": 98.0}}
        with pytest.raises(ValueError, match=r"inversion recovery.*TI"):
            check_protocol_matches_families(parse_protocol_map(bad), _FAMILIES)

    def test_a_spin_echo_contrast_may_not_declare_an_inversion_time(self) -> None:
        bad = {**_PROTOCOL, "T1": {"TR": 500.0, "TE": 18.4, "TI": 100.0}}
        with pytest.raises(ValueError, match=r"spin echo.*TI"):
            check_protocol_matches_families(parse_protocol_map(bad), _FAMILIES)

    def test_a_protocol_for_an_undeclared_contrast_raises(self) -> None:
        extra = {**_PROTOCOL, "PD": {"TR": 2000.0, "TE": 10.0}}
        with pytest.raises(ValueError, match=r"'PD'.*bloch_sequence_by_contrast"):
            check_protocol_matches_families(parse_protocol_map(extra), _FAMILIES)

    def test_a_gradient_echo_contrast_is_outside_the_echo_train_protocol(self) -> None:
        with pytest.raises(ValueError, match="spin echo or inversion recovery"):
            check_protocol_matches_families(parse_protocol_map({"T1": _PROTOCOL["T1"]}), {"T1": 2})


class TestLearnableFlipAngleIsRead:
    """The flip angle is a parameter of the gradient-echo Bloch layer only."""

    def test_an_echo_train_only_arm_may_not_declare_it(self) -> None:
        with pytest.raises(ValueError, match="gradient-echo"):
            check_flip_angle_is_read(True, _FAMILIES)

    def test_a_gradient_echo_family_reads_it(self) -> None:
        check_flip_angle_is_read(True, {"T1": 2, "T2": 0})

    def test_without_a_sequence_map_the_batch_decides_so_nothing_is_refused(self) -> None:
        check_flip_angle_is_read(True, {})

    def test_declaring_it_off_is_always_fine(self) -> None:
        check_flip_angle_is_read(False, _FAMILIES)


class TestAcquisitionParams:
    def test_each_sample_gets_its_contrasts_tr_te_ti(self) -> None:
        acq = acquisition_params(_batch(["T1", "FLAIR"]), parse_protocol_map(_PROTOCOL), 2, _CPU)
        assert acq["TR"].view(-1).tolist() == [500.0, 7500.0]
        assert acq["TE"].view(-1).tolist() == [pytest.approx(18.4), 98.0]
        assert acq["TI"].view(-1).tolist() == [0.0, 1655.0]
        assert acq["TR"].shape == (2, 1, 1, 1)

    def test_samples_cover_their_slices_in_order(self) -> None:
        acq = acquisition_params(_batch(["T1", "T2"]), parse_protocol_map(_PROTOCOL), 4, _CPU)
        assert acq["TR"].view(-1).tolist() == [500.0, 500.0, 5500.0, 5500.0]

    def test_an_explicit_acquisition_in_the_batch_is_used_as_given(self) -> None:
        rows = {k: torch.full((3, 1, 1, 1), v) for k, v in (("TR", 1.0), ("TE", 2.0), ("TI", 3.0))}
        acq = acquisition_params({"acquisition": rows}, parse_protocol_map(_PROTOCOL), 3, _CPU)
        assert acq["TE"].view(-1).tolist() == [2.0, 2.0, 2.0]

    def test_an_unlisted_contrast_raises(self) -> None:
        with pytest.raises(ValueError, match="'PD'"):
            acquisition_params(_batch(["PD"]), parse_protocol_map(_PROTOCOL), 1, _CPU)

    def test_a_batch_without_contrast_raises(self) -> None:
        with pytest.raises(ValueError, match="neither 'acquisition' nor 'contrast'"):
            acquisition_params(
                (torch.zeros(1), torch.zeros(1)), parse_protocol_map(_PROTOCOL), 1, _CPU
            )
