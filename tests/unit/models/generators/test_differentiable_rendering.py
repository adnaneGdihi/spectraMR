"""DifferentiableRenderingGenerator reads the arm's rendering block through the production build."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from spectramr.models.generators.differentiable_rendering import DifferentiableRenderingGenerator
from spectramr.models.registry import is_strict_kwargs

_ARM = (
    Path(__file__).resolve().parents[4]
    / "experiments/inprogress/novel_architectures/experiment_117_differentiable_rendering.yaml"
)


def _arm_model_kwargs(**overrides) -> dict:
    kwargs = yaml.safe_load(_ARM.read_text())["model"]["model_kwargs"]
    kwargs.update(feature_dim=4, num_layers=1, rendering_layers=1, material_dim=3, **overrides)
    return kwargs


def _build(model_kwargs: dict):
    """Build the arm's generator through ModelBuilder.build_generator, as train does."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.builders.model_builder import ModelBuilder

    settings = TrainingSettings.from_yaml(str(_ARM))
    model = settings.model.model_copy(update={"model_kwargs": model_kwargs})
    config = settings.model_copy(update={"model": model})
    return ModelBuilder(config, "cpu").build_generator()._models["generator"]


def test_registered_strict() -> None:
    assert is_strict_kwargs("differentiable_rendering")


def test_arm_builds_and_the_renderer_fires() -> None:
    model = _build(_arm_model_kwargs())
    assert isinstance(model, DifferentiableRenderingGenerator)
    fired = []
    model.renderer.register_forward_hook(lambda *_: fired.append(True))
    y = model(torch.randn(1, 1, 16, 16))
    assert fired and y.shape == (1, 1, 16, 16)
    assert model.renderer.slice_profile == "hamming_windowed_sinc"
    assert model.renderer.num_subslices == 7


def test_record_reaches_provenance() -> None:
    from spectramr.infrastructure.logging.provenance import count_parameters

    solver = count_parameters(_build(_arm_model_kwargs()))["solver"]
    assert solver["engine"] == "DiffSliceSim" and solver["slice_thickness_mm"] == 3.0
    assert solver["partial_volume_effect"] is True and solver["material_dim"] == 3


def test_unknown_engine_raises_through_the_factory() -> None:
    kwargs = _arm_model_kwargs()
    kwargs["rendering"] = {**kwargs["rendering"], "engine": "NoSuchEngine"}
    with pytest.raises(Exception, match=r"rendering\.engine"):
        _build(kwargs)


def test_unread_declared_key_raises() -> None:
    with pytest.raises(Exception, match="features"):
        _build(_arm_model_kwargs(features=[32, 64]))
