"""kspace_ar / kspace_masked resolve through the registry and run inside the generator."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.backbone_builders import (
    BackboneBuildContext,
    build_backbone,
    registered_backbone_names,
)
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

NAMES = ["kspace_ar", "kspace_masked"]


def _ctx(**kw) -> BackboneBuildContext:
    return BackboneBuildContext(
        in_channels=8,
        out_channels=8,
        features=(32,),
        img_size=(16, 16),
        feature_domain="kspace",
        kwargs={"depth": 2, **kw},
    )


@pytest.mark.parametrize("name", NAMES)
def test_registered_and_line_axis_reaches_module(name: str) -> None:
    assert name in registered_backbone_names()
    assert build_backbone(name, _ctx(line_axis=-1)).line_axis == -1


@pytest.mark.parametrize("name", NAMES)
def test_unknown_line_axis_raises(name: str) -> None:
    with pytest.raises(ValueError, match="line_axis"):
        build_backbone(name, _ctx(line_axis=0))


@pytest.mark.parametrize("name", NAMES)
def test_generator_forward_with_row_mask(name: str) -> None:
    model = KSpaceColdDiffusionGenerator(
        in_channels=2,
        out_channels=2,
        base_channels=32,
        num_layers=2,
        attention_type="none",
        backbone_type=name,
        condition_with_smaps=False,
        timesteps=29,
        dc_method="none",
        kspace_log_scaled=False,
        force_pure_kspace=True,
        image_size=(32, 32),
    ).eval()
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., ::2] = 1
    with torch.no_grad():
        out = model(torch.randn(1, 2, 32, 32) * mask, timesteps=torch.tensor([3]), mask=mask)
    assert out.shape == (1, 2, 32, 32) and torch.isfinite(out).all()


def test_backbone_kwargs_defects_for_line_backbones() -> None:
    from spectramr.models.generators.kspace_cold_diffusion_generator import (
        backbone_kwargs_defects,
    )

    assert backbone_kwargs_defects({"backbone_type": "kspace_ar", "line_axis": -1}) == []
    assert backbone_kwargs_defects({"backbone_type": "kspace_ar"}) == []
    assert backbone_kwargs_defects({"backbone_type": "kspace_ar", "line_axes": -1})
    assert backbone_kwargs_defects({"backbone_type": "kspace_masked", "line_axis": 0})
    assert backbone_kwargs_defects({"backbone_type": "kspace_ar", "patch_size": 8})
    assert backbone_kwargs_defects({"backbone_type": "dit", "line_axis": -1})
