"""WindowedSequenceDenoiser reads exactly ``receptive_half_width`` tokens either side, and no more."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.windowed_sequence_denoiser import WindowedSequenceDenoiser


@pytest.mark.parametrize(("half_width", "kernel"), [(8, 3), (6, 5), (4, 3)])
def test_the_receptive_field_is_exactly_the_declared_half_width(half_width, kernel) -> None:
    n, centre = 64, 32
    model = WindowedSequenceDenoiser(
        receptive_half_width=half_width, kernel_size=kernel, hidden_channels=4
    )
    x = torch.randn(1, 1, n, requires_grad=True)
    model(x)[0, 0, centre].backward()
    reads = (x.grad[0, 0] != 0).nonzero().flatten()
    assert int(reads.min()) == centre - half_width
    assert int(reads.max()) == centre + half_width
    assert model.receptive_half_width == half_width


def test_output_shape_matches_the_sequence() -> None:
    model = WindowedSequenceDenoiser(in_channels=2, out_channels=3, receptive_half_width=4)
    assert model(torch.randn(2, 2, 40)).shape == (2, 3, 40)


@pytest.mark.parametrize(
    "kw",
    [
        {"kernel_size": 4},
        {"kernel_size": 1},
        {"receptive_half_width": 7, "kernel_size": 5},
        {"receptive_half_width": 0},
        {"hidden_channels": 0},
    ],
)
def test_a_width_the_layers_cannot_sum_to_raises(kw) -> None:
    with pytest.raises(ValueError):
        WindowedSequenceDenoiser(**kw)


def test_an_image_instead_of_a_sequence_raises() -> None:
    with pytest.raises(ValueError, match=r"\(B, C, N\)"):
        WindowedSequenceDenoiser()(torch.randn(1, 1, 8, 8))


def test_the_model_is_registered_for_model_type_resolution() -> None:
    import spectramr.models.generators  # noqa: F401  (registers the generators)
    from spectramr.models.registry import get_model_class

    assert get_model_class("windowed_sequence_denoiser") is WindowedSequenceDenoiser
