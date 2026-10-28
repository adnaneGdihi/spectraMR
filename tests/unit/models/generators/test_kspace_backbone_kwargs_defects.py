"""Planted violations for :func:`backbone_kwargs_defects` (#2047).

One minimal ``model_kwargs`` dict per rule and per shape the rule takes, each of
which the function must reject, plus legal configurations it must accept. The
function is pure, so nothing here builds a model; the constructor's delegation is
checked by reading the raise it emits on the same dict.
"""

from __future__ import annotations

import pytest

from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
    backbone_kwargs_defects,
)

_IMAGE_DOMAIN = sorted(KSpaceColdDiffusionGenerator._IMAGE_DOMAIN_DC_BACKBONES)


@pytest.mark.parametrize("backbone_type", _IMAGE_DOMAIN)
def test_force_pure_kspace_on_an_image_domain_dc_backbone(backbone_type: str) -> None:
    defects = backbone_kwargs_defects({"backbone_type": backbone_type, "force_pure_kspace": True})
    assert len(defects) == 1 and "wrong domain" in defects[0]


@pytest.mark.parametrize("backbone_type", ["diff_varnet", "nafnet", "swin_diff_rec"])
def test_kspace_feature_norm_off_complex_unet(backbone_type: str) -> None:
    defects = backbone_kwargs_defects(
        {"backbone_type": backbone_type, "kspace_feature_norm": "rms"}
    )
    assert defects == [
        f"kspace_feature_norm='rms' applies only to backbone_type='complex_unet', "
        f"not {backbone_type!r}."
    ]


def test_kspace_feature_norm_with_the_default_backbone() -> None:
    """An arm that omits backbone_type gets the constructor's default, not complex_unet."""
    defects = backbone_kwargs_defects({"kspace_feature_norm": "rms"})
    assert len(defects) == 1 and "not 'unet'" in defects[0]


def test_radial_band_tokens_off_complex_unet() -> None:
    defects = backbone_kwargs_defects({"backbone_type": "nafnet", "radial_band_tokens_bands": 8})
    assert len(defects) == 1 and "per-up-step seam" in defects[0]


def test_radial_band_tokens_without_force_pure_kspace() -> None:
    defects = backbone_kwargs_defects(
        {"backbone_type": "complex_unet", "radial_band_tokens_bands": 8}
    )
    assert len(defects) == 1 and "needs force_pure_kspace: true" in defects[0]


def test_declared_radial_band_tokens_rungs() -> None:
    defects = backbone_kwargs_defects(
        {
            "backbone_type": "complex_unet",
            "force_pure_kspace": True,
            "radial_band_tokens_bands": 8,
            "radial_band_tokens_rungs": 29,
        }
    )
    assert len(defects) == 1 and "derived from the process's timestep count" in defects[0]


def test_misspelled_radial_band_tokens_knob() -> None:
    defects = backbone_kwargs_defects(
        {"backbone_type": "complex_unet", "radial_band_tokens_bnads": 8}
    )
    assert len(defects) == 1 and "radial_band_tokens_bnads" in defects[0]


def test_every_defect_is_reported_not_just_the_first() -> None:
    defects = backbone_kwargs_defects(
        {"backbone_type": "diff_varnet", "force_pure_kspace": True, "kspace_feature_norm": "rms"}
    )
    assert len(defects) == 2
    assert "applies only to" in defects[0] and "wrong domain" in defects[1]


@pytest.mark.parametrize(
    "model_kwargs",
    [
        {},
        None,
        {"backbone_type": "diff_varnet", "force_pure_kspace": False},
        {"backbone_type": "nafnet", "kspace_feature_norm": "none"},
        {
            "backbone_type": "complex_unet",
            "force_pure_kspace": True,
            "kspace_feature_norm": "rms",
            "radial_band_tokens_bands": 8,
            "radial_band_tokens_dim": 16,
            "radial_band_tokens_max_log": 2.0,
        },
    ],
)
def test_legal_configurations_are_accepted(model_kwargs: dict | None) -> None:
    assert backbone_kwargs_defects(model_kwargs) == []


def test_the_constructor_raises_the_first_defect() -> None:
    """The generator delegates to the function rather than keeping its own copy."""
    kwargs = {"backbone_type": "diff_varnet", "force_pure_kspace": True}
    expected = backbone_kwargs_defects(kwargs)[0]
    with pytest.raises(ValueError) as excinfo:
        KSpaceColdDiffusionGenerator(
            in_channels=2,
            out_channels=2,
            base_channels=8,
            num_layers=2,
            attention_type="none",
            **kwargs,
        )
    assert str(excinfo.value) == expected


@pytest.mark.parametrize("bad", ["ky", ["ky", "ky"], ["kz"]])
def test_kspace_geometry_planes_with_an_illegal_value(bad: object) -> None:
    defects = backbone_kwargs_defects({"backbone_type": "complex_unet", "kspace_geometry_planes": bad})
    assert len(defects) == 1 and "kspace_geometry_planes" in defects[0]


def test_kspace_geometry_planes_on_a_backbone_without_planes() -> None:
    exempt = sorted(KSpaceColdDiffusionGenerator._GEOMETRY_EXEMPT_BACKBONES)[0]
    defects = backbone_kwargs_defects({"backbone_type": exempt, "kspace_geometry_planes": ["ky"]})
    assert len(defects) == 1 and "applies only to backbones built with geometry" in defects[0]


def test_a_legal_kspace_geometry_planes_is_accepted() -> None:
    kw = {"backbone_type": "complex_unet", "kspace_geometry_planes": ["support", "acquired"]}
    assert backbone_kwargs_defects(kw) == []


@pytest.mark.parametrize(
    ("backbone_type", "typo", "fix"),
    [
        ("dit", "patch_sise", "patch_size"),
        ("uvit", "mlp_ration", "mlp_ratio"),
        ("diffit", "detph", "depth"),
        ("hat", "window_sise", "window_size"),
        ("hat", "cab_wieght", "cab_weight"),
        ("dit", "head", "heads"),
    ],
)
def test_misspelt_transformer_knob_is_flagged(backbone_type, typo, fix):
    defects = backbone_kwargs_defects({"backbone_type": backbone_type, typo: 4})
    assert any(typo in d and fix in d for d in defects), defects


@pytest.mark.parametrize(
    ("backbone_type", "knob"),
    [("dit", "window_size"), ("uvit", "cab_weight"), ("diffit", "window_size"), ("hat", "patch_size")],
)
def test_other_backbones_knob_is_flagged_as_inert(backbone_type, knob):
    defects = backbone_kwargs_defects({"backbone_type": backbone_type, knob: 8})
    assert any(knob in d and "never read" in d for d in defects), defects


@pytest.mark.parametrize("backbone_type", ["dit", "u_vit", "diffit", "hat"])
def test_declared_transformer_knobs_are_clean(backbone_type):
    kw = {"backbone_type": backbone_type, "depth": 6, "heads": 6, "timesteps": 29}
    assert backbone_kwargs_defects(kw) == []
