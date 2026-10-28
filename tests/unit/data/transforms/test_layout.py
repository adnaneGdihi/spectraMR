"""The TorchIO ``(C, W, H, D)`` <-> canonical ``(1, C, D, PE, RO)`` boundary."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.data.transforms.layout import (  # noqa: E402
    canonical_to_tio,
    pe_axis_of_plane_dim,
    tio_to_canonical,
)

# Non-square, non-cubic: a swapped pair of axes changes the shape, so it cannot pass.
C, W, H, D = 2, 6, 8, 5


def _tio() -> torch.Tensor:
    return torch.arange(C * W * H * D, dtype=torch.float32).reshape(C, W, H, D)


@pytest.mark.parametrize("pe_axis", ["W", "H"])
def test_round_trip_is_exact(pe_axis: str) -> None:
    t = _tio()
    assert torch.equal(canonical_to_tio(tio_to_canonical(t, pe_axis), pe_axis), t)


@pytest.mark.parametrize(("pe_axis", "pe", "ro"), [("W", W, H), ("H", H, W)])
def test_pe_lands_on_dim_minus_two_and_depth_on_dim_minus_three(pe_axis, pe, ro) -> None:
    out = tio_to_canonical(_tio(), pe_axis)
    assert out.shape == (1, C, D, pe, ro)


def test_pe_axis_w_maps_each_voxel_to_its_canonical_index() -> None:
    t = _tio()
    out = tio_to_canonical(t, "W")
    c, w, h, d = 1, 4, 7, 3
    assert out[0, c, d, w, h] == t[c, w, h, d]


def test_pe_axis_h_maps_each_voxel_to_its_canonical_index() -> None:
    t = _tio()
    out = tio_to_canonical(t, "H")
    c, w, h, d = 1, 4, 7, 3
    assert out[0, c, d, h, w] == t[c, w, h, d]


def test_complex_tensors_cross_the_boundary() -> None:
    t = torch.complex(_tio(), -_tio())
    assert torch.equal(canonical_to_tio(tio_to_canonical(t, "W"), "W"), t)


@pytest.mark.parametrize("pe_axis", ["D", "w", None])
def test_an_undeclared_pe_axis_is_refused(pe_axis) -> None:
    with pytest.raises(ValueError, match="pe_axis"):
        tio_to_canonical(_tio(), pe_axis)
    with pytest.raises(ValueError, match="pe_axis"):
        canonical_to_tio(tio_to_canonical(_tio(), "W"), pe_axis)


def test_wrong_ranks_are_refused() -> None:
    with pytest.raises(ValueError, match="TorchIO"):
        tio_to_canonical(_tio()[0], "W")
    with pytest.raises(ValueError, match="canonical"):
        canonical_to_tio(_tio(), "W")
    with pytest.raises(ValueError, match="one subject"):
        canonical_to_tio(torch.zeros(2, C, D, W, H), "W")


@pytest.mark.parametrize(
    ("declared", "expected"),
    [
        (-2, ("W", "data.acquisition.phase_encode_axis")),
        (-1, ("H", "data.acquisition.phase_encode_axis")),
        (None, ("W", "undeclared")),
    ],
)
def test_the_declared_plane_dim_names_the_torchio_pe_axis(declared, expected) -> None:
    assert pe_axis_of_plane_dim(declared) == expected


def test_an_unknown_plane_dim_is_refused() -> None:
    with pytest.raises(ValueError, match="phase_encode_axis"):
        pe_axis_of_plane_dim(0)
