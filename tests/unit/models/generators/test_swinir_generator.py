"""SwinIRGenerator keeps the input resolution through its RSTBs."""

import pytest
import torch

from spectramr.models.generators.swinir_generator import RSTB, SwinIRGenerator

pytestmark = pytest.mark.unit


def _tiny(depths: tuple[int, ...], upscale: int, img_size: int = 32) -> SwinIRGenerator:
    return SwinIRGenerator(
        img_size=img_size,
        embed_dim=8,
        depths=depths,
        num_heads=(2,) * len(depths),
        window_size=8,
        mlp_ratio=1.0,
        upscale=upscale,
        drop_path_rate=0.0,
    ).eval()


@pytest.mark.parametrize("depths", [(1,), (1, 1), (2, 2, 2), (1, 1, 1, 1)])
@pytest.mark.parametrize("upscale", [1, 2, 4])
def test_output_is_input_times_upscale(depths: tuple[int, ...], upscale: int) -> None:
    model = _tiny(depths, upscale)
    with torch.no_grad():
        out = model(torch.randn(2, 1, 32, 32))
    assert out.shape == (2, 1, 32 * upscale, 32 * upscale)
    assert out.shape == model.get_output_shape((2, 1, 32, 32))


@pytest.mark.parametrize("upscale", [1, 2])
def test_size_off_the_window_grid_is_padded_then_cropped(upscale: int) -> None:
    # 30 is not a multiple of window_size=8: reflect-padded to 32, then cropped back.
    model = _tiny((1, 1), upscale, img_size=32)
    with torch.no_grad():
        out = model(torch.randn(1, 1, 30, 30))
    assert out.shape == (1, 1, 30 * upscale, 30 * upscale)


def test_one_rstb_per_depth_entry_at_embed_dim() -> None:
    model = _tiny((2, 3, 1), 1)
    assert len(model.layers) == 3
    assert all(isinstance(layer, RSTB) for layer in model.layers)
    assert [len(layer.residual_group) for layer in model.layers] == [2, 3, 1]
    assert all(layer.dim == 8 for layer in model.layers)


def test_patch_size_above_one_raises() -> None:
    with pytest.raises(ValueError, match="patch_size"):
        SwinIRGenerator(img_size=32, embed_dim=8, depths=(1,), num_heads=(2,), patch_size=2)


def test_mismatched_num_heads_raises() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        SwinIRGenerator(img_size=32, embed_dim=8, depths=(1, 1), num_heads=(2,))
