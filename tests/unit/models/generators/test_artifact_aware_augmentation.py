"""``artifact_robust_generator`` resolved from the registry with exp 97's own block.

Reachability (non-negotiable 16): the model is built by ``ModelFactory`` from
the arm's ``model_kwargs.artifact_augmentation`` and observed to corrupt its
input in training mode, to stamp the detector loss, and to leave eval inputs
alone unless the robustness pass forces an artifact.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest
import torch
import yaml

from spectramr.core.metrics.coil_manifold import as_complex_coils
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.errors import ModelCreationError
from spectramr.models.factories.model_factory import ModelFactory
from spectramr.models.generators import artifact_corruption
from spectramr.models.generators.artifact_aware_augmentation import DETECTOR_LOSS_KEY

ARM = (
    Path(__file__).resolve().parents[4]
    / "experiments/inprogress/reconstruction/experiment_97_artifact_aware_training.yaml"
)


def _arm_block() -> dict:
    return yaml.safe_load(ARM.read_text())["model"]["model_kwargs"]["artifact_augmentation"]


def _build(**overrides):
    kwargs = {
        "in_channels": 4,
        "out_channels": 4,
        "features": [4, 8],
        "artifact_augmentation": _arm_block(),
        **overrides,
    }
    return ModelFactory().create_generator("artifact_robust_generator", **kwargs)


def _input() -> torch.Tensor:
    x = torch.randn(4, 4, 16, 16)
    x[..., ::2] = 0.0
    return x


def test_training_forward_corrupts_through_the_registry_and_stamps_the_detector_loss() -> None:
    torch.manual_seed(0)
    model = _build().train()
    with mock.patch.object(
        artifact_corruption, "apply_degradation", wraps=artifact_corruption.apply_degradation
    ) as spy:
        for _ in range(5):  # every artifact fires with p < 1; five draws make one certain
            out = model(_input())
    assert spy.call_count > 0
    assert out.shape == (4, 4, 16, 16)
    aux = model.auxiliary_losses()
    assert set(aux) == {DETECTOR_LOSS_KEY} == set(model.auxiliary_loss_keys)
    assert aux[DETECTOR_LOSS_KEY].requires_grad


def test_eval_forward_is_clean_and_deterministic_unless_an_artifact_is_forced() -> None:
    model = _build().eval()
    x = _input()
    with torch.no_grad():
        clean_a, clean_b = model(x), model(x)
        with model.forced_artifact("random_motion"):
            corrupted = model(x)
    assert torch.equal(clean_a, clean_b)
    assert not torch.allclose(clean_a, corrupted)
    assert model.auxiliary_losses() == {}
    assert model.robustness_eval_artifacts() == tuple(_arm_block()["artifacts"])


@pytest.mark.parametrize("name", ["complex_gaussian", "bias", "random_motion", "gibbs"])
def test_the_detector_reads_only_the_measured_support(name) -> None:
    block = _arm_block()
    block["artifacts"] = {name: {"probability": 1.0, "severity": [0.3, 0.3]}}
    block["eval_severity"] = 0.3
    model = _build(artifact_augmentation=block)
    seen: list[torch.Tensor] = []
    model.artifact_detector[0].register_forward_pre_hook(lambda _m, a: seen.append(a[0]))
    with torch.no_grad():
        model.train()(_input())
        with model.eval().forced_artifact(name):
            model(_input())
    for detector_input in seen:
        kspace = fft2c(as_complex_coils(detector_input, name="detector input"))
        assert kspace[..., ::2].abs().max() < 1e-5


def test_a_trailing_depth_axis_round_trips() -> None:
    model = _build().eval()
    with torch.no_grad():
        out = model(_input().unsqueeze(-1))
    assert out.shape == (4, 4, 16, 16, 1)


def test_an_unread_model_kwarg_raises() -> None:
    with pytest.raises((ModelCreationError, TypeError), match="base_channels"):
        _build(base_channels=8, _declared_keys=frozenset({"base_channels"}))


def test_the_declared_widths_reach_the_unet() -> None:
    assert tuple(_build().base_generator.features) == (4, 8)


def test_the_provenance_record_carries_the_applied_policy() -> None:
    record = _build().provenance_record()["artifact_augmentation"]
    assert record["eval_severity"] == _arm_block()["eval_severity"]


def test_the_arm_builds_through_the_production_generator_builder(tmp_path) -> None:
    """The arm's own YAML, widths shrunk, through ``resolve_generator_kwargs`` + ``GeneratorBuilder``."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.builders.generator_kwargs import resolve_generator_kwargs
    from spectramr.infrastructure.builders.leaf.model_builders import GeneratorBuilder

    raw = yaml.safe_load(ARM.read_text())
    raw["model"]["model_kwargs"]["features"] = [4, 8]
    path = tmp_path / "exp97_tiny.yaml"
    path.write_text(yaml.safe_dump(raw))
    config = TrainingSettings.from_yaml(path)
    builder = (
        GeneratorBuilder(config, "cpu")
        .with_architecture(config.model.model_type)
        .with_input_channels(config.model.in_channels)
        .with_output_channels(config.model.out_channels)
    )
    for key, value in resolve_generator_kwargs(config, device="cpu").items():
        builder.with_parameter(key, value)
    model = builder.validate().build().train()
    x = torch.randn(2, 8, 16, 16)
    x[..., ::2] = 0.0
    assert model(x).shape == x.shape
    assert set(model.auxiliary_losses()) == {DETECTOR_LOSS_KEY}
