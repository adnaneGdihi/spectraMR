"""DiffSliceSim: the slice profile, the knob validation and the provenance record."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.blocks.block_registry import BLOCK_REGISTRY
from spectramr.models.blocks.diff_slice_sim import (
    DiffSliceSim,
    build_renderer,
    slice_profile_weights,
)


def test_registered_as_a_block() -> None:
    assert BLOCK_REGISTRY["diff_slice_sim"] is DiffSliceSim


def test_hamming_sinc_profile_is_a_normalised_nonnegative_bump() -> None:
    z, w = slice_profile_weights("hamming_windowed_sinc", 3.0, 1.0, True)
    assert z.tolist() == [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    assert torch.all(w >= 0) and torch.isclose(w.sum(), torch.tensor(1.0))
    assert int(w.argmax()) == 3 and torch.allclose(w, w.flip(0))


def test_thickness_changes_the_profile() -> None:
    """The thickness is physical (mm), so it is not cancelled by the sampling grid."""
    _, w2 = slice_profile_weights("hamming_windowed_sinc", 2.0, 1.0, True)
    _, w3 = slice_profile_weights("hamming_windowed_sinc", 3.0, 1.0, True)
    assert w2.numel() != w3.numel() or not torch.allclose(w2, w3)
    assert w2.max() > w3.max()


def test_no_partial_volume_is_an_impulse() -> None:
    _, w = slice_profile_weights("hamming_windowed_sinc", 3.0, 1.0, False)
    assert w.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


def test_render_shape_parameter_parity_and_profile_reaches_output() -> None:
    # Same weights, different profile buffer: shows the profile is applied, not that it is identifiable.
    on = DiffSliceSim(feature_dim=4, out_channels=1, material_dim=3)
    off = DiffSliceSim(feature_dim=4, out_channels=1, material_dim=3, partial_volume_effect=False)
    assert sum(p.numel() for p in on.parameters()) == sum(p.numel() for p in off.parameters())
    off.load_state_dict(on.state_dict())
    x = torch.randn(2, 4, 8, 8)
    y_on, y_off = on(x), off(x)
    assert y_on.shape == (2, 1, 8, 8)
    assert not torch.allclose(y_on, y_off)


@pytest.mark.parametrize(
    ("rendering", "match"),
    [
        ({"engine": "NoSuchEngine"}, r"rendering\.engine"),
        ({"engine": "DiffSliceSim", "extra": 1}, "unknown keys"),
        ({"physics": {"slice_profile": "gauss"}}, "slice_profile"),
        ({"physics": {"slice_thickness_mm": 0}}, "slice_thickness_mm"),
        ({"physics": {"partial_volume_effect": "yes"}}, "partial_volume_effect"),
        ({"physics": {"bogus": 1}}, "unknown keys"),
    ],
)
def test_invalid_rendering_raises(rendering: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build_renderer(rendering, feature_dim=4, out_channels=1, material_dim=2)


def test_record_names_every_knob() -> None:
    record = build_renderer(
        {"engine": "DiffSliceSim", "physics": {"slice_thickness_mm": 2.0}},
        feature_dim=4,
        out_channels=1,
        material_dim=5,
    ).record
    assert record["engine"] == "DiffSliceSim" and record["slice_thickness_mm"] == 2.0
    assert record["material_dim"] == 5 and record["num_subslices"] == 5
