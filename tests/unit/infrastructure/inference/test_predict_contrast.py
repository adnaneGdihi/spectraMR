"""``predict`` feeds a contrast-conditioned model a real contrast id, or refuses."""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.inference.predict_contrast import resolve_predict_contrast

M4RAW = {"T1": 0, "T2": 1, "FLAIR": 2, "PD": 3}
BRAIN = {"AXT1": 0, "AXT1POST": 1, "AXT1PRE": 2, "AXT2": 3, "AXFLAIR": 4}


def test_single_contrast_model_needs_no_id() -> None:
    assert resolve_predict_contrast(0, Path("a_T1_rep1.h5"), None, M4RAW) is None


def test_an_id_for_a_model_without_an_embedding_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="num_contrasts=0"):
        resolve_predict_contrast(0, Path("a.h5"), 1, M4RAW)


def test_explicit_id_wins_over_the_file_name() -> None:
    out = resolve_predict_contrast(3, Path("sub_T1_rep1.h5"), 2, M4RAW)
    assert out is not None and out.tolist() == [2] and out.dtype.is_floating_point is False


@pytest.mark.parametrize(("name", "idx"), [("m4raw_FLAIR_x.h5", 2), ("p_t2_rep3.h5", 1)])
def test_file_name_keyword_is_read(name: str, idx: int) -> None:
    out = resolve_predict_contrast(3, Path(name), None, M4RAW)
    assert out is not None and out.tolist() == [idx]


def test_longest_keyword_wins_so_post_contrast_is_not_taken_for_plain_t1() -> None:
    out = resolve_predict_contrast(5, Path("file_AXT1POST_001.h5"), None, BRAIN)
    assert out is not None and out.tolist() == [1]


def test_no_id_and_no_keyword_raises_instead_of_defaulting_to_zero() -> None:
    with pytest.raises(ConfigurationError, match="--contrast-idx"):
        resolve_predict_contrast(3, Path("volume_0001.h5"), None, M4RAW)


def test_out_of_range_id_raises() -> None:
    with pytest.raises(ConfigurationError, match="outside"):
        resolve_predict_contrast(3, Path("a.h5"), 3, M4RAW)
