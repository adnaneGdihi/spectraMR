"""Tests for ColdDiffusionInferenceStrategy: predict delegates to ``generator.sample()``.

Since #2067 predict has no reverse loop of its own. These tests pin what the
strategy still owns -- the horizon, the step count and chunk size it hands the
shared path, the coil maps, the DC ledger -- and that it refuses what the single
path cannot honour. Equality with validation is pinned with the production
``sample()`` in ``tests/unit/models/diffusion/test_reverse_path.py``.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)


class _SamplingModel(nn.Module):
    """Records every ``sample()`` call; returns the measurement with its holes filled."""

    def __init__(self, *, num_timesteps=None, dc_method="hard", **contract) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)
        if num_timesteps is not None:
            self.num_timesteps = num_timesteps
        self._dc_method = dc_method
        for name, value in contract.items():
            setattr(self, name, value)
        self.calls: list[dict] = []

    def sample(self, measurement, mask=None, inference_timesteps=None, **kwargs):
        self.calls.append({"batch": measurement.shape[0], "steps": inference_timesteps, **kwargs})
        support = (measurement != 0).float() if mask is None else mask.expand_as(measurement)
        return measurement * support + 0.5 * (1 - support)


class _DDPLike(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


def _config(**extra) -> dict:
    return {"training": {"diffusion": {"timesteps": 8, "sampling_steps": 4}}, **extra}


def _measurement(channels: int = 2, size: int = 16):
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(1, channels, size, size, generator=generator)
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::2] = 1.0
    return full * mask, mask


def _run(model, config=None, **kwargs):
    measurement, mask = _measurement(kwargs.pop("channels", 2))
    strategy = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), config or _config())
    return strategy, strategy.run_inference(measurement.clone(), mask=mask, **kwargs)


def test_unsupported_degradation_raises():
    with pytest.raises(ValueError, match="kspace_mask"):
        ColdDiffusionInferenceStrategy(
            _SamplingModel(), torch.device("cpu"), {"diffusion": {"degradation": "blur"}}
        )


def test_predict_is_one_call_into_sample():
    model = _SamplingModel()
    _run(model)
    assert len(model.calls) == 1
    assert model.calls[0]["steps"] == 4
    assert model.calls[0]["start_timestep"] is None


def test_the_rung_reaches_sample_as_the_trajectory_head():
    model = _SamplingModel()
    _run(model, start_timestep=3)
    assert model.calls[0]["start_timestep"] == 3


def test_the_batch_is_chunked_at_the_validation_chunk_size():
    model = _SamplingModel()
    strategy = ColdDiffusionInferenceStrategy(
        model, torch.device("cpu"), _config(validation={"loader": {"chunk_size": 1}})
    )
    measurement = torch.ones(3, 2, 8, 8)
    strategy.run_inference(measurement, mask=torch.ones(3, 1, 8, 8))
    assert [c["batch"] for c in model.calls] == [1, 1, 1]


def test_an_undeclared_chunk_size_is_the_schema_default():
    strategy = ColdDiffusionInferenceStrategy(_SamplingModel(), torch.device("cpu"), _config())
    assert strategy.chunk_size == 2


def test_a_requested_step_callback_raises():
    """The single path has no per-step hook; dropping the monitor would fake a trajectory."""
    with pytest.raises(ValueError, match="step_callback"):
        _run(_SamplingModel(), step_callback=lambda *a: None)


def test_an_unread_keyword_raises():
    with pytest.raises(TypeError, match="not_a_knob"):
        _run(_SamplingModel(), not_a_knob=1)


def test_a_model_without_sample_raises():
    class _Bare(nn.Module):
        num_timesteps = 8

    with pytest.raises(AttributeError, match="sample"):
        _run(_Bare())


# --- the horizon and the step count ------------------------------------------


def test_the_horizon_is_the_models_own():
    """PLANTED: a 29-rung model with no configured horizon used to walk 1000 steps."""
    strat = ColdDiffusionInferenceStrategy(
        _SamplingModel(num_timesteps=29), torch.device("cpu"), {}
    )
    assert strat.num_timesteps == 29


def test_a_config_that_disagrees_with_the_model_raises():
    with pytest.raises(ValueError, match="num_timesteps=29 but the config"):
        ColdDiffusionInferenceStrategy(
            _SamplingModel(num_timesteps=29),
            torch.device("cpu"),
            {"training": {"diffusion": {"timesteps": 1000}}},
        )


def test_no_horizon_raises_when_the_steps_need_one():
    strat = ColdDiffusionInferenceStrategy(_SamplingModel(), torch.device("cpu"), {})
    with pytest.raises(ValueError, match="no diffusion horizon"):
        _ = strat.sampling_steps


def test_an_undeclared_step_count_leaves_the_generators_own():
    strat = ColdDiffusionInferenceStrategy(
        _SamplingModel(num_timesteps=29), torch.device("cpu"), {}
    )
    assert strat.sampling_steps is None


def test_validation_steps_outrank_diffusion_steps_and_are_clamped():
    config = {
        "training": {"diffusion": {"timesteps": 16, "sampling_steps": 4}},
        "validation": {"sampling": {"steps": 50}},
    }
    strat = ColdDiffusionInferenceStrategy(_SamplingModel(), torch.device("cpu"), config)
    assert strat.sampling_steps == 16


# --- coil maps: estimated for every kspace_cold_diffusion run, as validation does ----


_COLD = {"model": {"model_type": "kspace_cold_diffusion"}}


@pytest.mark.parametrize(
    ("contract", "model_type", "expects_maps"),
    [
        ({"condition_with_smaps": True}, "kspace_cold_diffusion", True),
        # An internal-DC backbone does not concatenate the maps, but sample()
        # stashes them for the mechanisms that read them; validation passes them too.
        ({"expects_smaps_concat": False}, "kspace_cold_diffusion", True),
        ({}, "image_cold_diffusion", False),
    ],
    ids=["conditioned", "internal_dc_backbone", "not_kspace_cold"],
)
def test_maps_follow_the_validation_predicate(contract, model_type, expects_maps):
    model = _SamplingModel(**contract)
    _run(model, _config(model={"model_type": model_type}), channels=8)
    smaps = model.calls[0]["smaps"]
    assert (smaps is not None) is expects_maps
    if expects_maps:
        # Image-domain, real-interleaved like the measurement: 4 coils -> 8 channels.
        assert smaps.shape[1] == 8 and not torch.is_complex(smaps)


def test_maps_reach_sample_through_the_ddp_wrapper():
    inner = _SamplingModel()
    _run(_DDPLike(inner), _config(**_COLD), channels=8)
    assert inner.calls[0]["smaps"] is not None


def test_single_channel_input_lifts_to_one_complex_coil():
    model = _SamplingModel()
    _run(model, _config(**_COLD), channels=1)
    assert model.calls[0]["smaps"].shape[1] == 2


def test_the_configured_estimation_method_reaches_the_estimator(monkeypatch):
    """``physics.coil_processing.estimation`` is honored, not hardcoded away.

    Predict reaches the calibrator through ``estimate_smaps_from_measurement``,
    the estimator validation's measured-map scoring shares, so the spy sits in
    the physics module that owns both.
    """
    from spectramr.infrastructure.physics import coil_sensitivity as mod

    seen: list[tuple] = []
    real = mod.estimate_smaps_calibrated

    def spy(kspace, method="power_iter", **kwargs):
        seen.append((method, kwargs))
        return real(
            kspace, log_scaled=kwargs["log_scaled"], method="power_iter", out_size=kwargs.get("out_size")
        )

    monkeypatch.setattr(mod, "estimate_smaps_calibrated", spy)
    config = _config(
        physics={"coil_processing": {"estimation": {"method": "espirit", "kernel_size": 5}}},
        **_COLD,
    )
    _run(_SamplingModel(), config, channels=8)
    method, kwargs = seen[0]
    assert method == "espirit" and kwargs["kernel_size"] == 5
    assert kwargs["out_size"] == (16, 16)
    assert kwargs["log_scaled"] is False


# --- physics.data_consistency.apply_at_predict ------------------------------------


def _knob_on_strategy(monkeypatch, model):
    from spectramr.infrastructure.inference import predict_data_consistency as pdc
    from spectramr.models.capabilities import ModelCapabilities

    caps = ModelCapabilities(output_domain="kspace")
    monkeypatch.setitem(pdc.MODEL_REGISTRY, "stub_cold", {"capabilities": caps})
    monkeypatch.setattr(pdc, "get_model_capabilities", lambda n: caps if n == "stub_cold" else None)
    return ColdDiffusionInferenceStrategy(
        model,
        torch.device("cpu"),
        _config(
            physics={"data_consistency": {"apply_at_predict": True}},
            model={"model_type": "stub_cold"},
            data={"dataset_type": "kspace"},
        ),
    )


class TestPredictDataConsistencyLedger:
    def test_a_hard_dc_loop_records_its_pin_and_the_hook_does_not_project_again(
        self, monkeypatch
    ):
        strategy = _knob_on_strategy(monkeypatch, _SamplingModel())
        projections: list[int] = []
        real = strategy.predict_dc.project
        monkeypatch.setattr(
            strategy.predict_dc, "project", lambda *a, **k: projections.append(1) or real(*a, **k)
        )
        measurement, mask = _measurement()
        strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)
        assert projections == []
        prov = strategy.predict_dc_provenance()
        assert prov["applied_by"] == {"ColdDiffusionInferenceStrategy.run_inference": 1}

    def test_a_learned_dc_loop_leaves_the_projection_to_the_hook(self, monkeypatch):
        """Soft/adaptive DC does not pin the bins, so the hook still projects."""
        strategy = _knob_on_strategy(monkeypatch, _SamplingModel(dc_method="adaptive"))
        projections: list[int] = []
        real = strategy.predict_dc.project
        monkeypatch.setattr(
            strategy.predict_dc, "project", lambda *a, **k: projections.append(1) or real(*a, **k)
        )
        measurement, mask = _measurement()
        strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)
        assert projections == [1]

    def test_an_unconstrained_loop_claims_no_pin(self, monkeypatch):
        """``reverse_dc='unconstrained'`` skips the per-step pin, so the ledger stays empty."""
        from types import SimpleNamespace

        model = _SamplingModel(_reverse_physics=SimpleNamespace(reverse_dc="unconstrained"))
        strategy = _knob_on_strategy(monkeypatch, model)
        projections: list[int] = []
        real = strategy.predict_dc.project
        monkeypatch.setattr(
            strategy.predict_dc, "project", lambda *a, **k: projections.append(1) or real(*a, **k)
        )
        measurement, mask = _measurement()
        strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)
        assert projections == [1]

    def test_without_a_mask_the_hook_raises(self, monkeypatch):
        from spectramr.domain.exceptions import ConfigurationError

        strategy = _knob_on_strategy(monkeypatch, _SamplingModel())
        measurement, _mask = _measurement()
        with pytest.raises(ConfigurationError, match="carries no mask"):
            strategy.infer(measurement.clone(), measured_kspace=measurement)

    def test_measured_kspace_never_reaches_sample(self, monkeypatch):
        model = _SamplingModel()
        strategy = _knob_on_strategy(monkeypatch, model)
        measurement, mask = _measurement()
        strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)
        assert all("measured_kspace" not in call for call in model.calls)


def test_it_owns_its_kspace_normalisation() -> None:
    """``preprocess_input`` normalises, so predict must not divide again (#2581)."""
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )

    assert ColdDiffusionInferenceStrategy.owns_kspace_normalization is True


# -- predict counts R over the acquired lines, as training does (#2572) -------------------


class _LineProcess:
    line_dim = -2


def _line_model() -> _SamplingModel:
    return _SamplingModel(num_timesteps=8, kspace_process=_LineProcess())


def _support(rows: int = 16, lo: int = 2, hi: int = 13) -> torch.Tensor:
    lines = torch.zeros(1, rows, dtype=torch.bool)
    lines[:, lo : hi + 1] = True
    return lines


def _banded(lo: int = 2, hi: int = 13) -> tuple[torch.Tensor, torch.Tensor]:
    """A measurement whose only nonzero rows are ``lo..hi`` (the rest never acquired)."""
    measurement, mask = _measurement()
    band = torch.zeros(1, 1, 16, 1)
    band[:, :, lo : hi + 1] = 1.0
    return measurement * band + band, mask


def test_measured_lines_reach_sample() -> None:
    model = _line_model()
    measurement, mask = _banded()
    strategy = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), _config())
    support = _support()
    strategy.run_inference(measurement, mask=mask, measured_lines=support)
    assert torch.equal(model.calls[0]["measured_lines"], support)


def test_no_measured_lines_leaves_the_call_unchanged() -> None:
    model = _line_model()
    _run(model)
    assert "measured_lines" not in model.calls[0]


def test_a_nonzero_line_outside_the_support_raises() -> None:
    measurement, mask = _banded(lo=2, hi=13)
    strategy = ColdDiffusionInferenceStrategy(_line_model(), torch.device("cpu"), _config())
    with pytest.raises(ValueError, match="outside the declared acquired support"):
        strategy.run_inference(measurement, mask=mask, measured_lines=_support(lo=4, hi=13))


def test_a_support_of_the_wrong_shape_raises() -> None:
    measurement, mask = _banded()
    strategy = ColdDiffusionInferenceStrategy(_line_model(), torch.device("cpu"), _config())
    with pytest.raises(ValueError, match="expected"):
        strategy.run_inference(measurement, mask=mask, measured_lines=_support(rows=20))


def test_an_arm_without_a_line_axis_refuses_a_support() -> None:
    class _NoAxis:
        line_dim = None

    model = _SamplingModel(num_timesteps=8, kspace_process=_NoAxis())
    measurement, mask = _banded()
    strategy = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), _config())
    assert strategy.acquisition_line_dim is None
    with pytest.raises(ValueError, match="no line axis"):
        strategy.run_inference(measurement, mask=mask, measured_lines=_support())


def test_predict_and_validation_select_identical_rungs_from_equivalent_support(tmp_path) -> None:
    """#2572's planted test, on the production ``cold_mri`` sampler.

    Validation reads the support off the fully sampled reference's zero lines; predict reads
    it off the ISMRMRD limits of the file. The same input must see the same mask at every rung,
    and the full-support call (the old predict) must not.
    """
    import h5py

    from spectramr.data.ismrmrd_limits import read_acquired_lines
    from spectramr.models.diffusion.reverse_path import sample_reverse_path
    from spectramr.models.diffusion.samplers import get_sampler
    from tests.unit.models.diffusion import _reverse_phantom as P

    lo, hi = 5, 56
    process = P.process()
    reference = P.kspace()
    plane = torch.zeros(1, 1, P.H, 1)
    plane[:, :, lo : hi + 1] = 1.0
    reference = reference * plane

    class _Generator(nn.Module):
        num_timesteps = P.T
        _dc_method = "hard"

        def __init__(self) -> None:
            super().__init__()
            self.kspace_process = process
            self.denoiser = P.Oracle(reference, process)
            self.rungs: list[torch.Tensor] = []

        def sample(self, measurement, mask=None, inference_timesteps=None, **kwargs):
            original = process.cascade_batch_masks

            def spy(*args, **kw):
                out = original(*args, **kw)
                self.rungs.append(out.clone())
                return out

            process.cascade_batch_masks = spy
            try:
                sampler = get_sampler(
                    "cold_mri",
                    model=self.denoiser,
                    num_timesteps=P.T,
                    max_acceleration=process.max_accel,
                    center_fraction=0.08,
                    sampling_steps=inference_timesteps or P.T,
                    reverse_mode="replace_freeze_dc",
                    kspace_log_scaled=False,
                )
                return sampler.sample(
                    measurement,
                    mask,
                    start_timestep=kwargs.get("start_timestep"),
                    measured_lines=kwargs.get("measured_lines"),
                )
            finally:
                process.cascade_batch_masks = original

    support = process.acquired_lines(reference)
    head = torch.full((2,), P.HEAD, dtype=torch.long)
    mask = process.cascade_batch_masks(head, (P.H, P.W), "cpu", acquired=support)
    measurement = reference * mask
    config = {"training": {"diffusion": {"timesteps": P.T, "sampling_steps": P.T}}}

    validation =_Generator()
    sample_reverse_path(
        validation, measurement, mask=mask, steps=P.T, chunk_size=2, measured_lines=support
    )

    path = tmp_path / "scan.h5"
    header = (
        '<ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD"><encoding><encodingLimits>'
        f"<kspace_encoding_step_1><minimum>{lo}</minimum><maximum>{hi}</maximum>"
        "</kspace_encoding_step_1></encodingLimits></encoding></ismrmrdHeader>"
    )
    with h5py.File(path, "w") as f:
        f["ismrmrd_header"] = header

    def predict(**kw):
        gen = _Generator()
        ColdDiffusionInferenceStrategy(gen, torch.device("cpu"), config).run_inference(
            measurement, mask=mask, **kw
        )
        return gen.rungs

    from_header = predict(measured_lines=read_acquired_lines(path, P.H, 2))
    assert len(from_header) == len(validation.rungs) > 0
    assert all(torch.equal(a, b) for a, b in zip(from_header, validation.rungs, strict=True))

    full_support = predict()
    assert not all(torch.equal(a, b) for a, b in zip(full_support, validation.rungs, strict=True))


def test_the_rung_process_is_the_generators_forward_process_through_ddp() -> None:
    """``predict --acceleration`` builds its rung masks from this process (#2883)."""
    process = _LineProcess()
    model = _SamplingModel(num_timesteps=8, kspace_process=process)
    strategy = ColdDiffusionInferenceStrategy(_DDPLike(model), torch.device("cpu"), _config())
    assert strategy.rung_process is process
    assert strategy.acquisition_line_dim == -2
    bare = ColdDiffusionInferenceStrategy(_SamplingModel(num_timesteps=8), torch.device("cpu"))
    assert bare.rung_process is None
