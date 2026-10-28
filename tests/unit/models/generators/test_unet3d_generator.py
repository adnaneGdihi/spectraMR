"""UNet3DGenerator keeps the input resolution and builds its channels from ``features``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.generators.unet3d_generator import UNet3DGenerator  # noqa: E402

pytestmark = [pytest.mark.unit]


@pytest.mark.parametrize("features", [(4, 8, 16, 32), (4, 8), (6, 12, 24)])
def test_output_matches_input_resolution(features: tuple[int, ...]) -> None:
    model = UNet3DGenerator(1, 2, features=features)
    x = torch.randn(2, 1, 16, 8, 8)
    assert model(x).shape == (2, 2, 16, 8, 8)
    assert len(model.downs) == len(model.ups) == len(features) - 1


def test_every_skip_reaches_the_output() -> None:
    torch.manual_seed(0)
    model = UNet3DGenerator(1, 1, features=(4, 8, 16)).eval()
    x = torch.randn(1, 1, 8, 8, 8, requires_grad=True)
    model(x).sum().backward()
    unused = [n for n, p in model.named_parameters() if p.grad is None]
    assert not unused, unused


def test_transposed_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="trilinear=True only"):
        UNet3DGenerator(1, 1, features=(4, 8), trilinear=False)
