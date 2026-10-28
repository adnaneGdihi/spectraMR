"""The acquired phase-encode range read from an ISMRMRD header (#2572).

Headers are synthetic XML: the repo carries no ISMRMRD data files.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from spectramr.data.ismrmrd_limits import (
    EncodingLimitsUnavailableError,
    acquired_lines_for_input,
    acquired_lines_from_limits,
    parse_header,
    placed_line_range,
    read_acquired_lines,
)

NS = "http://www.ismrm.org/ISMRMRD"


def _xml(
    step1: tuple[int, int] | None, enc: tuple[int, int] = (64, 64), center: int | None = None
) -> bytes:
    centre = "" if center is None else f"<center>{center}</center>"
    limits = (
        ""
        if step1 is None
        else f"<encodingLimits><kspace_encoding_step_1><minimum>{step1[0]}</minimum>"
        f"<maximum>{step1[1]}</maximum>{centre}</kspace_encoding_step_1>"
        "</encodingLimits>"
    )
    return (
        f'<ismrmrdHeader xmlns="{NS}"><encoding>'
        f"<encodedSpace><matrixSize><x>{enc[0]}</x><y>{enc[1]}</y><z>1</z></matrixSize>"
        "</encodedSpace>"
        f"<reconSpace><matrixSize><x>{enc[0]}</x><y>{enc[1]}</y><z>1</z></matrixSize></reconSpace>"
        f"{limits}</encoding></ismrmrdHeader>"
    ).encode()


def _h5(path, header: bytes | None, *, as_attr: bool = False):
    with h5py.File(path, "w") as f:
        f["kspace"] = np.zeros((2, 4, 4), dtype=np.complex64)
        if header is not None:
            if as_attr:
                f.attrs["ismrmrd_header"] = header
            else:
                f["ismrmrd_header"] = header
    return path


def test_parser_reads_matrix_sizes_and_the_step1_limits() -> None:
    parsed = parse_header(_xml((6, 57), enc=(320, 64)))
    assert parsed == {
        "encoded": (320, 64),
        "recon": (320, 64),
        "step1": (6, 57),
        "step1_center": None,
    }
    assert parse_header(_xml((6, 57), center=31))["step1_center"] == 31


@pytest.mark.parametrize("raw", [b"not xml", b"<a/>", _xml(None)])
def test_parser_reports_absent_limits_as_none_or_empty(raw) -> None:
    assert parse_header(raw).get("step1") is None


def test_limits_become_an_inclusive_line_indicator_per_row() -> None:
    lines = acquired_lines_from_limits((6, 57), 64, 3, source="f.h5")
    assert lines.shape == (3, 64) and lines.dtype == torch.bool
    assert lines[0].nonzero().flatten().tolist() == list(range(6, 58))
    assert torch.equal(lines[0], lines[2])


@pytest.mark.parametrize("limits", [(-1, 10), (10, 5)])
def test_a_degenerate_range_raises(limits) -> None:
    with pytest.raises(EncodingLimitsUnavailableError, match="valid line range"):
        acquired_lines_from_limits(limits, 64, 1, source="f.h5")


def test_a_range_past_the_plane_raises() -> None:
    with pytest.raises(ValueError, match="runs past the 64"):
        acquired_lines_from_limits((0, 64), 64, 1, source="f.h5")


def test_no_limits_raises_naming_the_key() -> None:
    with pytest.raises(EncodingLimitsUnavailableError, match="kspace_encoding_step_1"):
        acquired_lines_from_limits(None, 64, 1, source="f.h5")


@pytest.mark.parametrize("as_attr", [False, True])
def test_read_from_a_dataset_or_an_attribute(tmp_path, as_attr) -> None:
    path = _h5(tmp_path / "a.h5", _xml((6, 57)), as_attr=as_attr)
    lines = read_acquired_lines(path, 64, 2)
    assert int(lines.sum()) == 2 * 52


def test_a_file_without_a_header_raises_naming_file_and_key(tmp_path) -> None:
    path = _h5(tmp_path / "nohdr.h5", None)
    with pytest.raises(EncodingLimitsUnavailableError, match=r"nohdr\.h5.*'ismrmrd_header'"):
        read_acquired_lines(path, 64, 1)


def test_a_header_without_step1_raises_naming_the_key(tmp_path) -> None:
    path = _h5(tmp_path / "nolim.h5", _xml(None))
    with pytest.raises(EncodingLimitsUnavailableError, match=r"nolim\.h5.*kspace_encoding_step_1"):
        read_acquired_lines(path, 64, 1)


def test_a_non_hdf5_input_raises_naming_the_key(tmp_path) -> None:
    with pytest.raises(EncodingLimitsUnavailableError, match=r"x\.npy.*ismrmrd_header"):
        read_acquired_lines(tmp_path / "x.npy", 64, 1)


def _occupied(*columns: int, rows: int = 2, n: int = 16) -> torch.Tensor:
    """The ``[rows, n]`` bool of lines a loaded input carries data on."""
    lines = torch.zeros(rows, n, dtype=torch.bool)
    lines[:, list(columns)] = True
    return lines


def test_a_count_in_maximum_resolves_through_the_files_own_data_lines(tmp_path) -> None:
    """M4Raw: maximum is the line COUNT (10 here), the data sits on columns 3..12."""
    p = _h5(tmp_path / "m4raw.h5", _xml((0, 10)))
    lines = acquired_lines_for_input(p, _occupied(*range(3, 13)))
    assert lines[0].nonzero().flatten().tolist() == list(range(3, 13))
    assert lines.shape == (2, 16)


def test_an_inclusive_range_that_contains_the_data_is_the_support(tmp_path) -> None:
    """fastMRI: an undersampled input carries data on 4, 6, 8 only; the range is 3..12."""
    p = _h5(tmp_path / "fastmri.h5", _xml((3, 12)))
    lines = acquired_lines_for_input(p, _occupied(4, 6, 8))
    assert lines[0].nonzero().flatten().tolist() == list(range(3, 13))


def test_the_count_reading_is_not_a_fallback_for_a_header_that_fits_neither(tmp_path) -> None:
    """Data on 3..12 (10 lines) but maximum 5: not inside [0, 5], and 5 is not the count."""
    p = _h5(tmp_path / "bad.h5", _xml((0, 5)))
    with pytest.raises(ValueError, match="fits neither reading"):
        acquired_lines_for_input(p, _occupied(*range(3, 13)))


def test_an_undersampled_count_convention_file_is_refused_not_guessed(tmp_path) -> None:
    """Count 10 but only 4 lines carry data (prospective undersampling): the support is unknowable."""
    p = _h5(tmp_path / "us.h5", _xml((0, 10)))
    with pytest.raises(ValueError, match="fits neither reading"):
        acquired_lines_for_input(p, _occupied(3, 6, 9, 12))


def test_missing_limits_still_raise_through_the_input_reading(tmp_path) -> None:
    p = _h5(tmp_path / "none.h5", None)
    with pytest.raises(EncodingLimitsUnavailableError, match="ismrmrd_header"):
        acquired_lines_for_input(p, _occupied(3, 4))


# fastMRI places the limits by ``center`` (fastmri.data.mri_data): lines run from
# encodedSpace.y // 2 - center for maximum + 1 lines. A 368-wide brain plane with
# maximum 333 and center 167 acquires 17..350, not 0..333.
FASTMRI = {"step1": (0, 333), "step1_center": 167, "encoded": (640, 368)}


def test_fastmri_limits_are_placed_by_center_not_read_as_indices() -> None:
    assert placed_line_range(FASTMRI, 368, source="f.h5") == (17, 350)


def test_read_acquired_lines_places_a_fastmri_header(tmp_path) -> None:
    p = _h5(tmp_path / "brain.h5", _xml((0, 333), enc=(640, 368), center=167))
    lines = read_acquired_lines(p, 368, 1)
    assert lines[0].nonzero().flatten().tolist() == list(range(17, 351))


def test_a_full_fastmri_input_resolves_to_its_placed_range(tmp_path) -> None:
    """334 occupied lines never equal maximum (333), so the count reading cannot claim it."""
    p = _h5(tmp_path / "full.h5", _xml((0, 333), enc=(640, 368), center=167))
    lines = acquired_lines_for_input(p, _occupied(*range(17, 351), n=368))
    assert lines[0].nonzero().flatten().tolist() == list(range(17, 351))


def test_an_undersampled_fastmri_input_gets_the_placed_range_not_the_index_range(
    tmp_path,
) -> None:
    """The index reading would have accepted this and returned 0..333 (planted)."""
    p = _h5(tmp_path / "us.h5", _xml((0, 333), enc=(640, 368), center=167))
    lines = acquired_lines_for_input(p, _occupied(*range(20, 330, 4), n=368))
    support = lines[0].nonzero().flatten().tolist()
    assert support == list(range(17, 351))
    assert support != list(range(0, 334))


def test_a_centred_header_on_a_cropped_plane_raises() -> None:
    with pytest.raises(ValueError, match="cropped or padded"):
        placed_line_range(FASTMRI, 320, source="f.h5")


def test_a_count_header_with_a_center_still_reads_the_data_run(tmp_path) -> None:
    """M4Raw-style count with a centre element: the contiguous data run wins."""
    p = _h5(tmp_path / "m4.h5", _xml((0, 10), enc=(16, 16), center=8))
    lines = acquired_lines_for_input(p, _occupied(*range(3, 13)))
    assert lines[0].nonzero().flatten().tolist() == list(range(3, 13))


def test_an_undersampled_m4raw_input_inside_the_index_range_is_refused(tmp_path) -> None:
    """Count 10, data on 3, 6, 9: all inside [0, 10], so the index reading would pass and
    return 0..10, shifted from the true 3..12 run. Planted: the silent shape (#2555)."""
    p = _h5(tmp_path / "m4us.h5", _xml((0, 10)))
    with pytest.raises(ValueError, match="line-count convention"):
        acquired_lines_for_input(p, _occupied(3, 6, 9))


def test_a_full_plane_index_range_without_center_is_still_read(tmp_path) -> None:
    p = _h5(tmp_path / "full.h5", _xml((0, 15)))
    lines = acquired_lines_for_input(p, _occupied(2, 7, 11))
    assert lines[0].all()
