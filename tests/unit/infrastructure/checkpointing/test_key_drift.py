"""``key_drift``: a spectral-norm knob that changed between a checkpoint and its resume.

torch reports the mismatch as fifteen raw key names. The hint must name the cause when
spectral norm alone explains the difference, and stay silent for any other mismatch, so an
architecture change is never mislabelled.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import store
from spectramr.infrastructure.checkpointing.key_drift import drift_hint, spectral_norm_drift
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.resume import resume
from tests.unit.infrastructure.checkpointing.conftest import make_pipeline

PLAIN = ["model.0.weight", "model.0.bias"]
SPECTRAL = ["model.0.bias", "model.0.weight_orig", "model.0.weight_u", "model.0.weight_v"]


def test_a_checkpoint_with_spectral_norm_into_a_plain_model() -> None:
    assert "WITH spectral norm" in spectral_norm_drift(PLAIN, SPECTRAL)


def test_a_checkpoint_without_spectral_norm_into_a_spectral_model() -> None:
    assert "WITHOUT spectral norm" in spectral_norm_drift(SPECTRAL, PLAIN)


@pytest.mark.parametrize(
    ("model", "state"),
    [
        (PLAIN, PLAIN),
        (PLAIN, ["model.0.weight", "model.0.bias", "model.1.weight"]),
        (PLAIN, ["model.9.weight_orig", "model.9.weight_u", "model.9.weight_v", "model.0.bias"]),
        (PLAIN, ["other.weight", "other.bias"]),
    ],
)
def test_any_other_mismatch_is_left_unlabelled(model, state) -> None:
    assert spectral_norm_drift(model, state) is None


def _pipeline_with_discriminator(*, spectral: bool):
    pipeline = make_pipeline(ema=False)
    disc = nn.Linear(2, 1)
    pipeline.discriminator = nn.utils.spectral_norm(disc) if spectral else disc
    pipeline.optimizer_d = torch.optim.Adam(pipeline.discriminator.parameters(), lr=1e-3)
    return pipeline


def _save(tmp_path: Path, pipeline) -> Path:
    from types import SimpleNamespace

    return store.save_step(
        pipeline,
        layout=CheckpointLayout.for_run(tmp_path),
        config=SimpleNamespace(checkpoint=SimpleNamespace(format="pt", keep_last_n=5)),
        strategy=None,
        parallel=None,
        early_stopping=None,
        reason="periodic",
        epoch=0,
        step=2,
    )


def test_resume_names_the_knob_when_spectral_norm_flipped(tmp_path: Path) -> None:
    """The experiment_50 failure: written with spectral norm, resumed without it."""
    from types import SimpleNamespace

    path = _save(tmp_path, _pipeline_with_discriminator(spectral=True))
    with pytest.raises(RuntimeError, match="WITH spectral norm") as err:
        resume(
            path,
            _pipeline_with_discriminator(spectral=False),
            strategy=None,
            parallel=None,
            config=SimpleNamespace(checkpoint=SimpleNamespace(format="pt", keep_last_n=5)),
        )
    assert "spectral_norm" in str(err.value) and "discriminator" in str(err.value)


def test_drift_hint_is_none_without_a_state_or_models() -> None:
    assert drift_hint(None, None) is None
    assert drift_hint({"models": {}}, {"generator": nn.Linear(1, 1)}) is None
