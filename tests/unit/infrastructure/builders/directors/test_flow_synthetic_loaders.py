"""``flow_synthetic`` through the production loader path: YAML -> director -> batches.

Nothing is patched: transform building, the plain-dict route, collate selection,
the validation-worker guard and loader construction all run as they do under
``train``. CPU only, no model is built.
"""

from __future__ import annotations

import textwrap

import torch

from spectramr.config.settings import TrainingSettings
from spectramr.data.collation.strategies import FlowCollateStrategy
from spectramr.data.datasets.flow_sample import REQUIRED_FLOW_KEYS
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    DataPipelineDirector,
)
from spectramr.infrastructure.training.validation_rows import expand_depth

GRID_X = 8


def _settings(tmp_path) -> TrainingSettings:
    cfg = tmp_path / "flow.yaml"
    cfg.write_text(
        textwrap.dedent(f"""
        config_version: '1.0'
        workflow: {{regime: mri_flow, task: reconstruction}}
        model: {{model_type: standard_unet, in_channels: 1, out_channels: 1}}
        training: {{training_mode: reconstruction, epochs: 1}}
        optimization: {{optimizer_type: adam, learning_rate: 0.001}}
        logging: {{log_dir: {tmp_path}/logs, tensorboard_enabled: false}}
        data:
          dataset_type: flow_synthetic
          batch_size: 2
          phase_contrast: {{enabled: true, venc: 1.5}}
          flow_synthetic:
            enabled: true
            train_subjects: 2
            val_subjects: 2
            grid: [{GRID_X}, 12, 10]
            frames: 3
            coils: 2
            acceleration: 2.0
            center_fraction: 0.2
        """)
    )
    return TrainingSettings.from_yaml(str(cfg))


def _assert_flow_batch(batch: dict) -> None:
    assert set(batch) >= REQUIRED_FLOW_KEYS
    assert batch["input"].dtype == torch.complex64 and batch["input"].ndim == 6
    assert batch["target"].shape == batch["input"].shape
    assert batch["velocity"].shape[:2] == (batch["input"].shape[0], 3)


def test_the_director_serves_flow_slices_in_subject_order(tmp_path) -> None:
    train_loader, val_loader = DataPipelineDirector(_settings(tmp_path)).build_dataloaders(
        num_workers=0, pin_memory=False
    )
    for loader in (train_loader, val_loader):
        assert isinstance(loader.collate_fn.__self__, FlowCollateStrategy)

    _assert_flow_batch(next(iter(train_loader)))

    seen = []
    for batch in val_loader:
        _assert_flow_batch(batch)
        seen.extend(zip(batch["subject_id"], batch["slice_index"].tolist(), strict=True))
    expected = [(f"val-{s:03d}", x) for s in range(2) for x in range(GRID_X)]
    assert seen == expected


def test_a_flow_batch_passes_the_test_split_depth_expansion_unchanged(tmp_path) -> None:
    """expand_depth splits 5-D inputs; a 6-D flow batch must come through whole."""
    _, val_loader = DataPipelineDirector(_settings(tmp_path)).build_dataloaders(
        num_workers=0, pin_memory=False
    )
    batch = next(iter(val_loader))
    (out,) = list(expand_depth([batch]))
    assert out is batch
