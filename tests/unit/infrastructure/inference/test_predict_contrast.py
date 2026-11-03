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


def _multi_contrast(enabled: bool = True, rows: list[dict] | None = None):
    from types import SimpleNamespace

    params = [SimpleNamespace(model_dump=lambda r=r: r) for r in (rows or [])]
    return SimpleNamespace(
        enabled=enabled, contrast_map={"T1": 0, "T2": 1, "FLAIR": 2}, acquisition_params=params
    )


_ROWS = [
    {"name": "T1", "TE": 18.4, "TR": 500.0, "TI": 0.0, "FA": 90.0, "B0": 0.3,
     "contrast_type": "spin_echo"},
    {"name": "T2", "TE": 128.0, "TR": 5500.0, "TI": 0.0, "FA": 90.0, "B0": 0.3,
     "contrast_type": "spin_echo"},
    {"name": "FLAIR", "TE": 98.0, "TR": 7500.0, "TI": 1655.0, "FA": 90.0, "B0": 0.3,
     "contrast_type": "inversion_recovery"},
]  # fmt: skip


def _conditioned_forward(x, contrast_idx=None, acquisition_params=None):
    return x


def test_predict_conditions_a_generator_training_conditioned() -> None:
    from spectramr.infrastructure.inference.predict_contrast import (
        expand_conditioning,
        resolve_predict_conditioning,
    )

    cond = resolve_predict_conditioning(
        _multi_contrast(rows=_ROWS), _conditioned_forward, Path("sub_FLAIR_rep1.h5"), None
    )
    assert cond is not None and cond["contrast_idx"].tolist() == [2]
    assert cond["acquisition_params"]["TI"].tolist() == [1655.0]
    wide = expand_conditioning(cond, 4)
    assert wide["contrast_idx"].tolist() == [2] * 4
    assert wide["acquisition_params"]["TR"].tolist() == [7500.0] * 4


def test_predict_refuses_a_conditioned_generator_without_a_contrast() -> None:
    from spectramr.infrastructure.inference.predict_contrast import resolve_predict_conditioning

    with pytest.raises(ConfigurationError, match="--contrast-idx"):
        resolve_predict_conditioning(
            _multi_contrast(), _conditioned_forward, Path("file1000123.h5"), None
        )


def test_predict_leaves_an_unconditioned_arm_alone() -> None:
    from spectramr.infrastructure.inference.predict_contrast import resolve_predict_conditioning

    off = resolve_predict_conditioning(
        _multi_contrast(enabled=False), _conditioned_forward, Path("a_T1.h5"), None
    )
    plain = resolve_predict_conditioning(_multi_contrast(), lambda x: x, Path("a_T1.h5"), None)
    assert off is None and plain is None
