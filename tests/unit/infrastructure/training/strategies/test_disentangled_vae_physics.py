"""The physics axis of DisentangledVAETrainingStrategy, driven from the arm YAMLs.

The reachability tests load experiment_52 / experiment_55 through ``TrainingSettings``,
resolve the strategy class through ``TrainingStrategyFactory`` and run the real setup,
training-loss and validation methods on a tiny stub generator that records the physics
vectors it is handed.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.physics.acquisition_registry import AcquisitionRegistry
from spectramr.infrastructure.training.strategies.disentangled_vae_physics import (
    FieldSwapReadings,
    PhysicsAxis,
    resolve_physics_axis,
)

_ARMS = Path(__file__).resolve().parents[5] / "experiments" / "inprogress" / "vae_latent"
_LF, _HF = 0.064, 3.0


class _SpyVAE(torch.nn.Module):
    """``(recon, mu, logvar)`` + ``encode_content`` / ``decode``; records every style input."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, 3, padding=1)
        self.out = torch.nn.Conv2d(2, 1, 1)
        self.styles: list[torch.Tensor | None] = []

    def forward(self, x, style=None):
        self.styles.append(style)
        mu = x.mean(dim=(1, 2, 3)).unsqueeze(-1).expand(-1, 4)
        return self.out(self.conv(x)), mu, torch.zeros_like(mu)

    def encode_content(self, x):
        return self.conv(x)

    def decode(self, z, style):
        self.styles.append(style)
        return self.out(z) + style[:, 3:4, None, None]


def _settings(arm: str):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.from_yaml(_ARMS / arm)


def _strategy(arm: str):
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    settings = _settings(arm)
    cls = TrainingStrategyFactory().get_strategy_class(settings)
    s = cls.__new__(cls)
    s.config = settings
    s.device = torch.device("cpu")
    s.logging_service = SimpleNamespace(log_info=lambda *a, **k: None)
    s.env = SimpleNamespace(generator=_SpyVAE(), losses={}, model_type="disentangled_mri")
    s._loss_dict_reuse = {}
    s._setup_strategy_specific_components()
    s.loss_computer = SimpleNamespace(
        compute=lambda pred, target, epoch, iteration, losses_dict: SimpleNamespace(
            total=torch.nn.functional.l1_loss(pred, target)
        )
    )
    s._get_loss_weight = lambda name, epoch: 1.0
    s._compute_training_metrics = lambda **kw: {}
    s._get_validation_metrics_computer = lambda cfg: SimpleNamespace(
        compute=lambda pred, target: {"mse": ((pred - target) ** 2).mean()}
    )
    return s


def _paired_batch(subjects: list[str]) -> dict:
    n = len(subjects)
    return {
        "input": torch.randn(n, 1, 8, 8),
        "target": torch.randn(n, 1, 8, 8),
        "field_strength_source_t": torch.full((n, 1), _LF),
        "field_strength_t": torch.full((n, 1), _HF),
        "subject_id": subjects,
    }


def test_exp52_feeds_each_row_the_physics_of_its_stamped_field():
    s = _strategy("experiment_52_disentangled_vae.yaml")
    assert type(s).__name__ == "DisentangledVAETrainingStrategy"
    assert s._physics.source == "paired_field_stamps"
    batch = _paired_batch(["s1", "s2"])
    batch["field_strength_t"] = torch.tensor([[_HF], [_LF]])  # row 2: a 64 mT target
    losses = s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)
    assert torch.isfinite(losses["g_total_loss"])
    lf_vec = AcquisitionRegistry.get_physics_vector("T1w", "64mT")[0]
    hf_vec = AcquisitionRegistry.get_physics_vector("T1w", "3T")[0]
    src, tgt = s.env.generator.styles[0], s.env.generator.styles[1]
    assert torch.allclose(src, lf_vec.expand(2, -1))
    assert torch.allclose(tgt[0], hf_vec) and torch.allclose(tgt[1], lf_vec)


def test_exp52_without_field_stamps_raises():
    s = _strategy("experiment_52_disentangled_vae.yaml")
    batch = _paired_batch(["s1", "s2"])
    del batch["field_strength_t"]
    with pytest.raises(ValueError, match="field_strength_t"):
        s._compute_losses_impl(batch["input"], batch["target"], epoch=0, batch=batch)


def test_exp52_validation_emits_the_field_disentanglement_readings():
    s = _strategy("experiment_52_disentangled_vae.yaml")
    s.reset_validation_metrics()
    for subjects in (["s1", "s2"], ["s3", "s4"]):
        batch = _paired_batch(subjects)
        metrics = s.validation_step(batch, batch["input"], batch["target"])
        assert "val_mse" in metrics
    summary = s.finalize_validation()
    for key in (
        "val_content_field_leak",
        "val_swap_content_distance",
        "val_field_swap_error_ratio",
    ):
        assert key in summary and summary[key] == summary[key]  # present and not NaN
    assert summary["val_probe_split_by_subject"] == 1.0


def test_exp55_trains_and_validates_a_self_reconstruction():
    s = _strategy("experiment_55_disentangled_representation.yaml")
    assert s._physics.source == "single_field"
    img = torch.randn(2, 1, 8, 8)
    losses = s._compute_losses_impl(img, img, epoch=0, batch={"input": img})
    assert set(losses) >= {"l1_self", "kl", "g_total_loss"} and "l_trans" not in losses
    assert s.env.generator.styles == [None]  # style comes from the image, no physics vector
    s.reset_validation_metrics()
    s.validation_step({"input": img, "target": img}, img, img)
    assert s.finalize_validation() == {}


def test_paired_source_on_a_route_without_stamps_raises():
    settings = _settings("experiment_52_disentangled_vae.yaml")
    data = settings.data.model_copy(update={"dataset_type": "nifti_paired"})
    with pytest.raises(ValueError, match="contrast_aware_paired"):
        resolve_physics_axis(settings.model_copy(update={"data": data}))


def test_unknown_source_and_unregistered_contrast_raise():
    with pytest.raises(ValueError, match="physics_source"):
        PhysicsAxis("stamps")
    with pytest.raises(ValueError, match="no registered protocol"):
        PhysicsAxis("paired_field_stamps", ("ADC", "ADC"))


def test_swap_ratio_reads_whether_the_physics_code_moves_the_render():
    readings = FieldSwapReadings()
    for lo in (0, 2):
        content = torch.randn(2, 2, 4, 4)
        target = torch.ones(2, 1, 4, 4)
        readings.add(
            content=(content, content.clone()),
            physics=(torch.zeros(2, 4), torch.ones(2, 4)),
            fields=(torch.full((2,), _LF), torch.full((2,), _HF)),
            groups=[f"s{lo}", f"s{lo + 1}"],
            swap=(target * 0.9, target * 0.0, target, content),
        )
    out = readings.summary()
    assert out["val_field_swap_error_ratio"] == pytest.approx(0.01)
    assert out["val_swap_content_distance"] == pytest.approx(0.0, abs=1e-6)


def test_one_small_batch_without_subject_ids_still_reads_the_leak():
    """The contrast_aware route stamps no subject_id, and a smoke pass may hold a single
    batch of two: interleaved rows keep both fields on the probe's training side."""
    readings = FieldSwapReadings()
    content = torch.randn(2, 2, 4, 4)
    target = torch.ones(2, 1, 4, 4)
    readings.add(
        content=(content, content + 1.0),
        physics=(torch.zeros(2, 4), torch.ones(2, 4)),
        fields=(torch.full((2,), _LF), torch.full((2,), _HF)),
        groups=None,
        swap=(target, target, target, content),
    )
    out = readings.summary()
    assert out["val_probe_split_by_subject"] == 0.0
    assert "val_content_field_leak" in out
