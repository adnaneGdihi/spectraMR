"""Paired tests for ``latent_diffusion_multi_axis_cfg`` (list-form multi-axis CFG)."""

from __future__ import annotations

import pytest
import torch

from spectramr.models.generators.latent_diffusion_multi_axis_cfg import (
    LatentDiffusionMultiAxisCFG,
    translate_axis_lists,
)


class TestTranslate:
    def test_parallel_lists_become_mappings(self):
        drops, weights = translate_axis_lists(["condition", "contrast"], [4.0, 1.5], [0.1, 0.2])
        assert weights == {"condition": 4.0, "contrast": 1.5}
        assert drops == {"condition": 0.1, "contrast": 0.2}

    def test_absent_drops_stay_none_so_the_generator_default_applies(self):
        assert translate_axis_lists(["condition"], [2.0])[0] is None

    def test_mapping_drops_pass_through(self):
        drops, _ = translate_axis_lists(["condition"], [2.0], {"condition": 0.1})
        assert drops == {"condition": 0.1}

    @pytest.mark.parametrize(
        ("axes", "weights", "drops", "match"),
        [
            (["condition"], [1.0, 2.0], None, "parallel"),
            (["condition", "condition"], [1.0, 2.0], None, "repeats"),
            ([], [], None, "at least one"),
            ("condition", [1.0], None, "must be a list"),
            (["condition"], [1.0], [0.1, 0.2], "cfg_drop_probs has"),
            (["condition"], [1.0], {"contrast": 0.1}, "outside cfg_axes"),
        ],
    )
    def test_malformed_lists_raise(self, axes, weights, drops, match):
        with pytest.raises(ValueError, match=match):
            translate_axis_lists(axes, weights, drops)


def _tiny(**over):
    kwargs = {
        "in_channels": 1,
        "out_channels": 1,
        "latent_channels": 4,
        "base_channels": 8,
        "timesteps": 4,
        "spatial_dims": 2,
        "device": "cpu",
        "conditional_translation": True,
        "cfg_axes": ["condition"],
        "cfg_weights": [2.0],
        "cfg_drop_probs": [0.5],
        "cfg_schedule": {"condition": {"type": "linear", "start": 6.0, "end": 1.0}},
    }
    kwargs.update(over)
    return kwargs


@pytest.mark.parametrize(
    ("axes", "match"),
    [(["anatomy"], "unknown axis"), (["pathology"], "cannot be conditioned on")],
)
def test_axes_the_latent_path_cannot_condition_on_raise(axes, match):
    with pytest.raises(ValueError, match=match):
        LatentDiffusionMultiAxisCFG(**_tiny(cfg_axes=axes, cfg_schedule=None))


def test_missing_axis_lists_raise():
    with pytest.raises(ValueError, match="needs cfg_axes"):
        LatentDiffusionMultiAxisCFG(**_tiny(cfg_axes=None))


def test_registry_resolves_the_real_class_and_the_mechanism_fires(monkeypatch):
    """Reachability: factory -> class; training drop and scheduled guidance both act."""
    from spectramr.models.factories.model_factory import get_model_factory
    from spectramr.models.generators import latent_cfg
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert MODEL_REGISTRY.get("latent_diffusion_multi_axis_cfg")["class"] is (
        LatentDiffusionMultiAxisCFG
    )
    gen = get_model_factory().create_generator("latent_diffusion_multi_axis_cfg", **_tiny())
    assert isinstance(gen, LatentDiffusionMultiAxisCFG)
    assert gen.cfg_guidance_record["cfg_weights"] == {"condition": 2.0}
    assert gen.cfg_guidance_record["cfg_schedule"]["condition"]["type"] == "linear"

    drawn: list[dict] = []
    real_masks = latent_cfg.axis_keep_masks

    def _spy_masks(spec, batch, device):
        out = real_masks(spec, batch, device)
        drawn.append(out)
        return out

    monkeypatch.setattr(
        "spectramr.models.generators.latent_diffusion_generator.axis_keep_masks", _spy_masks
    )
    gen.train()
    side = 16 // gen.autoencoder.downsample_factor
    z = torch.randn(2, 4, side, side)
    gen(z, torch.zeros(2, dtype=torch.long), condition_image=torch.randn(2, 1, 16, 16))
    assert drawn and set(drawn[0]) == {"condition"}, "training drop never ran"

    progresses: list[float] = []
    real_guided = latent_cfg.guided_noise

    def _spy_guided(*args, progress=0.0, **kwargs):
        progresses.append(progress)
        return real_guided(*args, progress=progress, **kwargs)

    monkeypatch.setattr(
        "spectramr.models.generators.latent_diffusion_generator.guided_noise", _spy_guided
    )
    gen.eval()
    with torch.no_grad():
        gen.sample((1, 1, 16, 16), condition_image=torch.randn(1, 1, 16, 16))
    assert progresses == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])
