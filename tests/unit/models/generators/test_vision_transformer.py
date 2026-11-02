"""VisionTransformer conv stem: total stride equals patch_size (experiment_17)."""

import pytest
import torch

from spectramr.models.generators.vision_transformer import VisionTransformer, _conv_stem

_TINY = {"embed_dim": 16, "depth": 1, "num_heads": 2, "mlp_ratio": 2.0}


@pytest.mark.parametrize("patch_size", [2, 4, 8, 16])
def test_conv_stem_stride_equals_patch_size(patch_size: int) -> None:
    stem = _conv_stem(1, 16, patch_size)
    out = stem(torch.zeros(1, 1, 32, 32))
    assert out.shape == (1, 16, 32 // patch_size, 32 // patch_size)


@pytest.mark.parametrize("patch_size", [1, 6, 12])
def test_conv_stem_rejects_non_power_of_two(patch_size: int) -> None:
    with pytest.raises(ValueError, match="power-of-two"):
        _conv_stem(1, 16, patch_size)


def test_factory_builds_conv_stem_vit_and_forward_runs() -> None:
    from spectramr.models.factories.model_factory import ModelFactory

    model = ModelFactory().create_generator(
        "vision_transformer",
        in_channels=1,
        out_channels=1,
        img_size=32,
        patch_size=8,
        use_conv_stem=True,
        **_TINY,
    )
    assert isinstance(model, VisionTransformer)
    assert model.use_conv_stem
    assert isinstance(model.patch_embed, torch.nn.Sequential)
    x = torch.randn(2, 1, 32, 32)
    assert model(x).shape == x.shape


def test_input_size_mismatch_raises() -> None:
    model = VisionTransformer(img_size=32, patch_size=8, **_TINY)
    with pytest.raises(ValueError, match="img_size=32"):
        model(torch.zeros(1, 1, 16, 16))


def test_img_size_not_divisible_raises() -> None:
    with pytest.raises(ValueError, match="not divisible"):
        VisionTransformer(img_size=30, patch_size=8, **_TINY)
