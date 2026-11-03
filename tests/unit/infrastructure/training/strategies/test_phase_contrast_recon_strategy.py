"""PhaseContrastReconStrategy through the production path, with a tiny model.

YAML -> TrainingSettings -> TrainingEnvironmentDirector -> TrainingStrategyFactory,
then one training step through ``train_step``'s closure and one validation pass
through ``pipelines.train._run_validation``: dispatch, the 6-D batch, the volume
assembly in ``finalize_validation`` and the loop's image capture all run as under
``train``. The DI container is not bootstrapped, so the environment's device is
whatever the default device service resolves (the GPU when one is present); the
batch is moved there, as the training loop does.
"""

from __future__ import annotations

import math
import textwrap
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.data.batch_types import BatchAdapter
from spectramr.data.datasets.axis_exposure import resolve_axes_for
from spectramr.infrastructure.builders.context import BuilderContext
from spectramr.infrastructure.training.builders.director import TrainingEnvironmentDirector
from spectramr.infrastructure.training.strategies.phase_contrast_recon_strategy import (
    PhaseContrastReconStrategy,
    velocity_error,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

GRID_X = 8


def _settings(tmp_path, extra: str = "", image_weight: float = 1.0) -> TrainingSettings:
    cfg = tmp_path / "pc.yaml"
    cfg.write_text(
        textwrap.dedent(f"""
        config_version: '1.0'
        workflow: {{regime: mri_flow, task: reconstruction}}
        model:
          model_type: flow_kt_unrolled
          model_kwargs: {{encodings: 4, cascades: 1, features: 8, layers: 2}}
        training:
          training_mode: phase_contrast_recon
          input_domain: kspace
          output_domain: image
          epochs: 1
          device: cpu
          output_dir: {tmp_path}/out
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        losses:
          physics: {{lambda_phase_contrast_image_nmse: {image_weight}, lambda_phase_contrast_phasor: 1.0}}
        data:
          dataset_type: flow_synthetic
          batch_size: 2
          phase_contrast: {{enabled: true, venc: 1.5}}
          flow_synthetic:
            enabled: true
            train_subjects: 2
            val_subjects: 1
            grid: [{GRID_X}, 12, 10]
            frames: 3
            coils: 2
            acceleration: 2.0
            center_fraction: 0.2
        validation:
          enabled: true
          loader: {{batch_size: 2}}
        logging:
          log_dir: {tmp_path}/logs
          tensorboard_enabled: false
        {extra}
        """)
    )
    return TrainingSettings.from_yaml(str(cfg))


def _environment(settings: TrainingSettings):
    env = TrainingEnvironmentDirector(BuilderContext(config=settings)).build_environment(
        with_test_loader=False
    )
    # The DI container is not bootstrapped here; no-op services stand in for it.
    strategy = TrainingStrategyFactory().create_strategy(
        env, logging_service=MagicMock(), metrics_service=MagicMock()
    )
    return env, strategy


def test_one_training_step_and_one_validation_pass_run_end_to_end(tmp_path) -> None:
    from spectramr.pipelines.train import _run_validation

    settings = _settings(tmp_path)
    env, strategy = _environment(settings)
    assert isinstance(strategy, PhaseContrastReconStrategy)

    raw = next(iter(env.data_loaders["train"]))
    # As the training loop does: adapt, then move to the environment's device.
    batch = BatchAdapter.from_dict(raw, axes=resolve_axes_for(settings.data)).to(env.device)
    (step,) = strategy.train_step(batch, 0, iteration=0, batch_data=batch, scaler=None)
    loss = step["closure"]()
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in env.generator.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

    pipeline = SimpleNamespace(
        config=settings,
        device=env.device,
        data_loaders=env.data_loaders,
        models=env.models,
        generator=env.generator,
        ema=None,
    )
    metrics = _run_validation(pipeline, strategy, iteration=1, epoch=0, logging_service=None)
    for key in (
        "val_loss",
        "val_image_nmse",
        "val_zf_image_nmse",
        "val_velocity_rmse_roi",
        "val_velocity_rmse_roi_volume",
        "val_divergence_normalised",
        "val_mass_conservation",
    ):
        assert key in metrics and math.isfinite(metrics[key]), key


def test_setup_refuses_a_validation_cap(tmp_path) -> None:
    capped = _settings(tmp_path, extra="").model_copy(deep=True)
    loader = capped.validation.loader.model_copy(update={"num_batches": 2})
    capped = capped.model_copy(
        update={"validation": capped.validation.model_copy(update={"loader": loader})}
    )
    with pytest.raises(ValueError, match="cannot be capped"):
        _environment(capped)


def test_setup_refuses_a_phasor_only_objective(tmp_path) -> None:
    with pytest.raises(ValueError, match="lambda_phase_contrast_image_nmse must be > 0"):
        _environment(_settings(tmp_path, image_weight=0.0))


def test_setup_refuses_phase_contrast_knobs_it_does_not_read(tmp_path) -> None:
    settings = _settings(tmp_path)
    pc = settings.data.phase_contrast.model_copy(update={"voxel_area": 4.0})
    data = settings.data.model_copy(update={"phase_contrast": pc})
    with pytest.raises(ValueError, match="voxel_area"):
        _environment(settings.model_copy(update={"data": data}))


def test_multi_process_backends_are_refused() -> None:
    from spectramr.infrastructure.distributed.strategy_compat import evaluate_parallel_compat

    for backend in ("ddp", "fsdp"):
        verdict = evaluate_parallel_compat(PhaseContrastReconStrategy, backend)
        assert verdict.refused, backend


def test_velocity_error_cancels_a_shared_background_and_wraps() -> None:
    g = torch.Generator().manual_seed(0)
    reference = torch.randn(2, 4, 3, 5, 4, generator=g, dtype=torch.complex64)
    background = torch.exp(1j * torch.rand(2, 4, 1, 5, 4, generator=g))
    venc = torch.tensor([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]])
    shift = 0.3  # rad on direction y only
    images = reference * background
    images[:, 2] *= math.e ** (1j * (shift + 2 * math.pi))
    err = velocity_error(images, reference * background, venc)
    expected = torch.zeros_like(err)
    expected[:, 1] = shift * venc[:, 1, None, None, None] / math.pi
    assert torch.allclose(err, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("markers", "expected"),
    [
        ({"kspace_normalized": torch.tensor([True])}, {"input_raw", "input_prepared"}),
        ({"kspace_normalized": torch.tensor([False])}, set()),
        ({}, set()),
    ],
)
def test_the_snapshot_decompresses_the_input_only_after_a_normalizing_loader(markers, expected):
    """``input_raw`` is the loader's k-space: compressed exactly when its transform ran."""
    strategy = object.__new__(PhaseContrastReconStrategy)
    seen: dict = {}
    strategy.save_debug_snapshot = lambda tensors, **kwargs: seen.update(kwargs)
    volume = torch.rand(1, 2, 3, 4, 4)
    batch = SimpleNamespace(input=torch.rand(1, 2, 4, 4), **markers)
    strategy._snapshot(batch, volume, volume, step=0)
    assert seen["log_scaled_keys"]() == expected
