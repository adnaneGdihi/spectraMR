"""``synthetic_marker`` produces the ``batch['marker']`` key PrivilegedLearningStrategy requires."""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.registry import build_transform, transforms_producing
from spectramr.data.transforms.synthetic_marker import SyntheticMarkerTransform


def _subject(c: int = 1, h: int = 32, w: int = 32, d: int = 1) -> tio.Subject:
    return tio.Subject(input=tio.ScalarImage(tensor=torch.rand(c, h, w, d) * 4.0))


def test_registered_as_producer_of_marker() -> None:
    assert "synthetic_marker" in transforms_producing("marker")
    assert isinstance(build_transform("synthetic_marker", num_markers=3), SyntheticMarkerTransform)


def test_marker_matches_input_and_peaks_at_amplitude() -> None:
    subj = _subject(c=2, d=3)
    out = SyntheticMarkerTransform(num_markers=4, amplitude=0.5, seed=0)(subj)
    marker, inp = out["marker"].data, out["input"].data
    assert marker.shape == inp.shape
    assert marker.max().item() == pytest.approx(0.5 * inp.abs().max().item(), rel=0.05)
    assert (marker > 0.5 * marker.max()).float().mean() < 0.2  # sparse dots, not a flood


def test_seed_fixes_layout_and_none_redraws() -> None:
    subj = _subject()
    a = SyntheticMarkerTransform(seed=3)(subj)["marker"].data
    b = SyntheticMarkerTransform(seed=3)(subj)["marker"].data
    torch.testing.assert_close(a, b)
    c1 = SyntheticMarkerTransform()(subj)["marker"].data
    c2 = SyntheticMarkerTransform()(subj)["marker"].data
    assert not torch.equal(c1, c2)


def test_input_is_left_untouched() -> None:
    subj = _subject()
    before = subj["input"].data.clone()
    SyntheticMarkerTransform(seed=1)(subj)
    torch.testing.assert_close(subj["input"].data, before)


@pytest.mark.parametrize("kwargs", [{"num_markers": 0}, {"sigma_px": 0.0}, {"amplitude": -1.0}])
def test_bad_knobs_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SyntheticMarkerTransform(**kwargs)


def test_missing_input_raises() -> None:
    subj = tio.Subject(target=tio.ScalarImage(tensor=torch.rand(1, 8, 8, 1)))
    with pytest.raises(KeyError):
        SyntheticMarkerTransform().apply_transform(subj)
