"""``predict --acceleration R`` samples what validation samples at R (#2883).

Validation degrades a fully sampled input to the rung for ``R`` over its acquired
lines and starts the reverse loop at that rung's timestep. Predict now resolves the
same rung through the same owner (:mod:`spectramr.models.diffusion.cold_rung`); the
parity test drives the real ``_process_single_file`` and the production ``cold_mri``
sampler, and the planted half shows the old predict (inferred mask, ``T - 1``) does not.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)
from spectramr.infrastructure.inference.predict_rung import resolve_predict_rung
from spectramr.models.diffusion.cold_rung import (
    rung_for_acceleration,
    rung_mask,
    schedule_accelerator,
)
from spectramr.models.diffusion.reverse_path import sample_reverse_path
from spectramr.models.diffusion.samplers import get_sampler
from spectramr.pipelines import infer as infer_mod
from tests.unit.models.diffusion import _reverse_phantom as P

h5py = pytest.importorskip("h5py")

LO, HI = 5, 56
R = 4.0
MODE = "replace_freeze_dc"
#: A strided schedule: from T-1 the strided steps land on other rungs than from the
#: input's own, so the head changes the output (at ``steps == T`` it costs compute only).
STEPS = 4


class _Generator(nn.Module):
    """``sample()`` is the production ``cold_mri`` loop over a step-sensitive denoiser."""

    _dc_method = "hard"

    def __init__(self, process, reference) -> None:
        super().__init__()
        self.num_timesteps = P.T
        self.kspace_process = process
        self.denoiser = P.StepSensitive(reference, process)
        self.starts: list[int | None] = []

    def sample(self, measurement, mask=None, inference_timesteps=None, **kwargs):
        self.starts.append(kwargs.get("start_timestep"))
        sampler = get_sampler(
            "cold_mri",
            model=self.denoiser,
            num_timesteps=P.T,
            max_acceleration=self.kspace_process.max_accel,
            center_fraction=0.08,
            sampling_steps=inference_timesteps or P.T,
            reverse_mode=MODE,
            kspace_log_scaled=False,
        )
        if mask is None:
            mask = (measurement.abs().sum(dim=1, keepdim=True) > 1e-6).float()
        return sampler.sample(
            measurement,
            mask,
            start_timestep=kwargs.get("start_timestep"),
            measured_lines=kwargs.get("measured_lines"),
        )


def _reference() -> torch.Tensor:
    band = torch.zeros(1, 1, P.H, 1)
    band[:, :, LO : HI + 1] = 1.0
    return P.kspace() * band


def _scan(tmp_path: Path, mask: torch.Tensor | None = None) -> Path:
    path = tmp_path / "scan.h5"
    header = (
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding><encodingLimits>'
        f"<kspace_encoding_step_1><minimum>{LO}</minimum><maximum>{HI}</maximum>"
        "</kspace_encoding_step_1></encodingLimits></encoding></ismrmrdHeader>"
    )
    with h5py.File(path, "w") as f:
        f["kspace"] = [0.0]
        f["ismrmrd_header"] = header
        if mask is not None:
            f["mask"] = mask.numpy()
    return path


def _strategy(process, reference) -> ColdDiffusionInferenceStrategy:
    config = {
        "training": {"diffusion": {"timesteps": P.T, "sampling_steps": STEPS}},
        "model": {"input_type": "kspace"},
    }
    return ColdDiffusionInferenceStrategy(
        _Generator(process, reference), torch.device("cpu"), config
    )


def _validation(process, reference) -> tuple[torch.Tensor, int]:
    """What the cascade does at R: the rung's timestep, its mask, and a sample from it."""
    rung = rung_for_acceleration(schedule_accelerator(process), R, P.T)
    acquired = process.acquired_lines(reference)
    steps = torch.full((reference.shape[0],), rung.timestep, dtype=torch.long)
    mask = rung_mask(process, steps, reference, acquired)
    gen = _Generator(process, reference)
    out = sample_reverse_path(
        gen,
        reference * mask,
        mask=mask,
        steps=STEPS,
        chunk_size=2,
        start_timestep=rung.timestep,
        measured_lines=acquired,
    )
    return out, rung.timestep


def _predict(tmp_path, monkeypatch, tensor, acceleration, mask=None):
    process = P.process()
    strategy = _strategy(process, _reference())
    saved: list[torch.Tensor] = []
    monkeypatch.setattr(infer_mod, "_load_input", lambda f, d: tensor.clone())
    monkeypatch.setattr(infer_mod, "_preprocess_tensor", lambda t, c, d, m: t)
    monkeypatch.setattr(infer_mod, "_save_output", lambda out, *a, **k: saved.append(out))
    results = {"outputs": [], "num_processed": 0}
    infer_mod._process_single_file(
        _scan(tmp_path, mask),
        tmp_path,
        strategy.model,
        strategy,
        torch.device("cpu"),
        SimpleNamespace(data=SimpleNamespace(dataset_type="kspace", slice_level_records=False)),
        2,
        results,
        acceleration=acceleration,
    )
    return saved[0], strategy.model.starts, results


def test_predict_at_r_reproduces_validation_at_r(tmp_path, monkeypatch) -> None:
    reference = _reference()
    expected, t_rung = _validation(P.process(), reference)
    out, starts, results = _predict(tmp_path, monkeypatch, reference, R)
    assert starts == [t_rung] and t_rung < P.T - 1
    assert torch.equal(out, expected)
    (stamp,) = results["rung_at_predict"]["files"]
    assert stamp["source"] == "acceleration" and stamp["start_timestep"] == t_rung


def test_planted_without_the_rung_predict_starts_at_t_minus_1(tmp_path, monkeypatch) -> None:
    """The old predict on the same degraded input: inferred mask, head at T-1. It differs."""
    reference = _reference()
    expected, t_rung = _validation(P.process(), reference)
    process = P.process()
    acquired = process.acquired_lines(reference)
    steps = torch.full((2,), t_rung, dtype=torch.long)
    degraded = reference * rung_mask(process, steps, reference, acquired)
    out, starts, _ = _predict(tmp_path, monkeypatch, degraded, None)
    assert starts == [None]
    assert not torch.allclose(out, expected)


def test_a_file_mask_sets_the_start_timestep_and_is_not_reapplied(tmp_path, monkeypatch) -> None:
    reference = _reference()
    expected, t_rung = _validation(P.process(), reference)
    process = P.process()
    acquired = process.acquired_lines(reference)
    steps = torch.full((2,), t_rung, dtype=torch.long)
    mask = rung_mask(process, steps, reference, acquired)
    out, starts, results = _predict(tmp_path, monkeypatch, reference * mask, None, mask[:, :1])
    assert starts == [t_rung]
    assert torch.equal(out, expected)
    assert results["rung_at_predict"]["files"][0]["source"] == "mask"


def test_acceleration_on_an_undersampled_input_without_a_mask_raises(tmp_path) -> None:
    reference = _reference()
    reference[:, :, LO + 1 : HI : 2] = 0.0
    process = P.process()
    with pytest.raises(ValueError, match="fully sampled"):
        resolve_predict_rung(
            _strategy(process, reference),
            _scan(tmp_path),
            reference,
            torch.zeros(2, P.H, dtype=torch.bool).index_fill_(1, torch.arange(LO, HI + 1), True),
            R,
        )


def test_an_unrealised_acceleration_is_a_run_invariant_error(tmp_path) -> None:
    reference = _reference()
    process = P.process()
    with pytest.raises(ConfigurationError, match="round-trip-mismatch"):
        resolve_predict_rung(_strategy(process, reference), _scan(tmp_path), reference, None, 64.0)


def test_acceleration_on_a_strategy_without_a_forward_process_raises(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="no cold-diffusion forward process"):
        resolve_predict_rung(SimpleNamespace(), _scan(tmp_path), _reference(), None, R)


def test_nothing_requested_leaves_predict_unchanged(tmp_path) -> None:
    reference = _reference()
    strategy = _strategy(P.process(), reference)
    assert resolve_predict_rung(strategy, _scan(tmp_path), reference, None, None) is None
    assert resolve_predict_rung(SimpleNamespace(), _scan(tmp_path), reference, None, None) is None
