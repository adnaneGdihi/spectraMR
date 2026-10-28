"""Tests for acquisition-parameter parsing and the SNR prior it implies."""

from __future__ import annotations

import math

import pytest

from spectramr.infrastructure.physics.acquisition_params import (
    AcquisitionParams,
    MissingAcquisitionParameter,
    predicted_snr_delta_db,
    read_acquisition_params,
)

HEADER_3T = """<?xml version="1.0"?>
<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">
  <acquisitionSystemInformation>
    <systemVendor>ACME</systemVendor>
    <systemFieldStrength_T>3.0</systemFieldStrength_T>
    <receiverChannels>16</receiverChannels>
  </acquisitionSystemInformation>
  <sequenceParameters>
    <TR>2000.0</TR>
    <TE>8.5</TE>
    <averages>1</averages>
  </sequenceParameters>
  <userParameters>
    <userParameterDouble><name>bandwidth_hz_px</name><value>250.0</value></userParameterDouble>
  </userParameters>
</ismrmrdHeader>"""

HEADER_MINIMAL = """<?xml version="1.0"?>
<ismrmrdHeader>
  <acquisitionSystemInformation>
    <systemFieldStrength_T>0.3</systemFieldStrength_T>
  </acquisitionSystemInformation>
</ismrmrdHeader>"""


# ── parsing ───────────────────────────────────────────────────────────


def test_reads_every_recorded_parameter_through_a_namespace():
    # Real ISMRMRD headers are namespaced; a parser that ignored that reads nothing.
    p = read_acquisition_params(HEADER_3T)
    assert p.field_strength_t == pytest.approx(3.0)
    assert p.tr_ms == pytest.approx(2000.0)
    assert p.te_ms == pytest.approx(8.5)
    assert p.averages == pytest.approx(1.0)
    assert p.bandwidth_hz_px == pytest.approx(250.0)


def test_absent_optional_parameters_are_none_not_zero():
    # A header that simply does not record TR states a FACT. Zero would be a lie that
    # propagates silently into any ratio.
    p = read_acquisition_params(HEADER_MINIMAL)
    assert p.field_strength_t == pytest.approx(0.3)
    assert p.tr_ms is None
    assert p.bandwidth_hz_px is None


def test_accepts_bytes_as_h5_stores_it():
    assert read_acquisition_params(
        HEADER_3T.encode()
    ).field_strength_t == pytest.approx(3.0)


def test_missing_header_raises():
    with pytest.raises(ValueError, match="no ISMRMRD header"):
        read_acquisition_params(None)


def test_malformed_xml_raises_rather_than_returning_blanks():
    with pytest.raises(ValueError, match="not valid XML"):
        read_acquisition_params("<ismrmrdHeader><unclosed>")


def test_require_raises_on_an_absent_parameter():
    with pytest.raises(MissingAcquisitionParameter, match="bandwidth_hz_px"):
        AcquisitionParams(field_strength_t=3.0).require("bandwidth_hz_px")


# ── the SNR prior ─────────────────────────────────────────────────────


def test_field_strength_drop_predicts_the_expected_db_loss():
    # 3T -> 0.3T is a factor 10 in B0 => 20*log10(0.1) = -20 dB.
    hq = AcquisitionParams(field_strength_t=3.0)
    lq = AcquisitionParams(field_strength_t=0.3)
    assert predicted_snr_delta_db(hq, lq) == pytest.approx(-20.0)


def test_equal_acquisitions_predict_no_change():
    p = AcquisitionParams(field_strength_t=1.5)
    assert predicted_snr_delta_db(p, p) == pytest.approx(0.0)


def test_averaging_recovers_snr_by_root_n():
    # 4 averages at the LQ site = +6 dB back.
    hq = AcquisitionParams(field_strength_t=3.0, averages=1)
    lq = AcquisitionParams(field_strength_t=3.0, averages=4)
    assert predicted_snr_delta_db(hq, lq) == pytest.approx(20 * math.log10(2.0))


def test_wider_bandwidth_costs_snr_by_root_bw():
    # 4x the receiver bandwidth at the LQ site = -6 dB.
    hq = AcquisitionParams(field_strength_t=3.0, bandwidth_hz_px=100.0)
    lq = AcquisitionParams(field_strength_t=3.0, bandwidth_hz_px=400.0)
    assert predicted_snr_delta_db(hq, lq) == pytest.approx(20 * math.log10(0.5))


def test_optional_terms_are_neutral_when_either_side_omits_them():
    # A missing term must contribute a ratio of 1, never be guessed.
    hq = AcquisitionParams(field_strength_t=3.0, averages=1)
    lq = AcquisitionParams(field_strength_t=3.0)  # no averages recorded
    assert predicted_snr_delta_db(hq, lq) == pytest.approx(0.0)


def test_voxel_volume_is_deliberately_excluded():
    """The prior must NOT include the voxel-volume term.

    ``resample_to_spacing`` already puts the volume on the low-quality grid, and
    area-averaging onto coarser voxels raises SNR exactly as larger voxels do in a
    real acquisition. Counting it here too would hand the noise axis a target several
    dB too optimistic. AcquisitionParams carries no voxel field at all, which is what
    makes the double-count structurally impossible rather than merely avoided.
    """
    assert not hasattr(AcquisitionParams(field_strength_t=3.0), "voxel_volume_mm3")
    hq = AcquisitionParams(field_strength_t=3.0)
    lq = AcquisitionParams(field_strength_t=3.0)
    # Same field, different grids (handled elsewhere) => the prior says nothing.
    assert predicted_snr_delta_db(hq, lq) == pytest.approx(0.0)


def test_missing_field_strength_raises_rather_than_defaulting():
    with pytest.raises(MissingAcquisitionParameter, match="field_strength_t"):
        predicted_snr_delta_db(
            AcquisitionParams(), AcquisitionParams(field_strength_t=0.3)
        )


def test_headers_round_trip_into_a_realistic_prior():
    hq = read_acquisition_params(HEADER_3T)
    lq = read_acquisition_params(HEADER_MINIMAL)
    delta = predicted_snr_delta_db(hq, lq)
    # 3T -> 0.3T, LQ records neither averages nor bandwidth, so both are neutral.
    assert delta == pytest.approx(-20.0)


def test_sidecar_reads_bids_keys_and_converts_seconds_to_ms():
    from spectramr.infrastructure.physics.acquisition_params import acquisition_from_sidecar

    got = acquisition_from_sidecar(
        {
            "MagneticFieldStrength": 0.064,
            "RepetitionTime": 1.5,
            "EchoTime": 0.004,
            "NumberOfAverages": 2,
        }
    )
    assert got.field_strength_t == pytest.approx(0.064)
    assert got.tr_ms == pytest.approx(1500.0)
    assert got.te_ms == pytest.approx(4.0)
    assert got.averages == pytest.approx(2.0)
    assert got.bandwidth_hz_px is None


def test_sidecar_without_field_strength_raises():
    from spectramr.infrastructure.physics.acquisition_params import acquisition_from_sidecar

    with pytest.raises(MissingAcquisitionParameter, match="MagneticFieldStrength"):
        acquisition_from_sidecar({"RepetitionTime": 1.5, "EchoTime": 0.004})


def test_sidecar_that_is_not_an_object_raises():
    from spectramr.infrastructure.physics.acquisition_params import acquisition_from_sidecar

    with pytest.raises(ValueError, match="JSON object"):
        acquisition_from_sidecar([0.064])
