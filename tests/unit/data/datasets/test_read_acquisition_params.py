"""``read_acquisition_params``: the ismrmrd_header tags an M4Raw / fastMRI file carries."""

from __future__ import annotations

from spectramr.data.datasets.kspace_volume_stack import read_acquisition_params


class TestReadAcquisitionParams:
    """TR/TE/TI were in the data all along; nothing parsed them (#606).

    ``brc`` and ``qvcr`` reported ``needs acq_params`` and
    ``phys_residual_consistency`` raised *"requires acquisition (TR/TE/TI in ms)"* on
    files whose ``ismrmrd_header`` states ``TR=7500, TE=98, TI=1655, flipAngle=160``
    outright — three metrics declared inapplicable for want of a parser.
    """

    @staticmethod
    def _write(tmp_path, header: str | None):
        import h5py
        import numpy as np

        path = tmp_path / "acq.h5"
        with h5py.File(path, "w") as f:
            f.create_dataset("kspace", data=np.zeros((1, 2, 4, 4), dtype=np.complex64))
            if header is not None:
                f.create_dataset("ismrmrd_header", data=np.bytes_(header.encode()))
        return path

    def test_parses_the_namespaced_tags_m4raw_writes(self, tmp_path) -> None:

        path = self._write(
            tmp_path,
            "<ns0:ismrmrdHeader><ns0:sequenceParameters>"
            "<ns0:TR>7500.0</ns0:TR><ns0:TE>98.0</ns0:TE><ns0:TI>1655.0</ns0:TI>"
            "<ns0:flipAngle_deg>160.0</ns0:flipAngle_deg>"
            "<ns0:sequence_type>FLAIR_TRA</ns0:sequence_type>"
            "</ns0:sequenceParameters></ns0:ismrmrdHeader>",
        )
        got = read_acquisition_params(path)
        assert got["TR"] == 7500.0
        assert got["TE"] == 98.0
        assert got["TI"] == 1655.0
        assert got["flip_angle"] == 160.0
        assert got["sequence_type"] == "FLAIR_TRA"

    def test_parses_an_unnamespaced_header_too(self, tmp_path) -> None:
        """Other vendors write bare ``<TR>``; the regex must not require a prefix."""

        path = self._write(tmp_path, "<ismrmrdHeader><TR>2500</TR></ismrmrdHeader>")
        assert read_acquisition_params(path)["TR"] == 2500.0

    def test_absent_tag_is_omitted_not_defaulted(self, tmp_path) -> None:
        """A fabricated TR is worse than a missing one — the metric must see the gap."""

        path = self._write(tmp_path, "<ismrmrdHeader><TE>98.0</TE></ismrmrdHeader>")
        got = read_acquisition_params(path)
        assert got == {"TE": 98.0}
        assert "TR" not in got

    def test_no_header_returns_the_honest_empty(self, tmp_path) -> None:
        """Keeps the metric ``not_applicable`` rather than feeding it invented physics."""

        assert read_acquisition_params(self._write(tmp_path, None)) == {}

    def test_reads_echo_spacing_train_length_and_encoded_geometry(self, tmp_path) -> None:
        """The series-gain gate and the EPG forward model need these beside TR/TE/TI."""

        path = self._write(
            tmp_path,
            "<ns0:ismrmrdHeader><ns0:encoding><ns0:encodedSpace><ns0:matrixSize>"
            "<ns0:x>256</ns0:x><ns0:y>256</ns0:y><ns0:z>1</ns0:z></ns0:matrixSize>"
            "<ns0:fieldOfView_mm><ns0:x>240</ns0:x><ns0:y>240</ns0:y><ns0:z>5</ns0:z>"
            "</ns0:fieldOfView_mm></ns0:encodedSpace><ns0:reconSpace><ns0:matrixSize>"
            "<ns0:x>512</ns0:x><ns0:y>512</ns0:y><ns0:z>1</ns0:z></ns0:matrixSize>"
            "</ns0:reconSpace><ns0:echoTrainLength>16</ns0:echoTrainLength></ns0:encoding>"
            "<ns0:sequenceParameters><ns0:TR>4000</ns0:TR><ns0:TE>100</ns0:TE>"
            "<ns0:echo_spacing>12.5</ns0:echo_spacing></ns0:sequenceParameters>"
            "</ns0:ismrmrdHeader>",
        )
        got = read_acquisition_params(path)
        assert got["echo_spacing"] == 12.5
        assert got["echo_train_length"] == 16.0
        assert got["encoded_matrix"] == (256, 256)  # not the reconSpace 512
        assert got["fov_mm"] == (240.0, 240.0, 5.0)

    def test_echo_spacing_with_a_unit_suffix_is_a_number_and_na_is_absent(self, tmp_path) -> None:
        """M4Raw writes ``16.3ms`` on its TSE series and ``N/A`` on its single-echo SE T1."""

        tse = self._write(
            tmp_path, "<ismrmrdHeader><echo_spacing>16.3ms</echo_spacing></ismrmrdHeader>"
        )
        assert read_acquisition_params(tse)["echo_spacing"] == 16.3
        se = self._write(
            tmp_path, "<ismrmrdHeader><echo_spacing>N/A</echo_spacing></ismrmrdHeader>"
        )
        assert "echo_spacing" not in read_acquisition_params(se)

    def test_echo_spacing_in_an_unknown_unit_raises(self, tmp_path) -> None:
        """A silently mis-scaled echo spacing would put the TSE echo at the wrong index."""
        import pytest

        path = self._write(
            tmp_path, "<ismrmrdHeader><echo_spacing>0.0163s</echo_spacing></ismrmrdHeader>"
        )
        with pytest.raises(ValueError, match="echo_spacing"):
            read_acquisition_params(path)

    def test_geometry_is_omitted_when_the_header_has_none(self, tmp_path) -> None:

        got = read_acquisition_params(
            self._write(tmp_path, "<ismrmrdHeader><TR>1</TR></ismrmrdHeader>")
        )
        assert "encoded_matrix" not in got and "fov_mm" not in got
