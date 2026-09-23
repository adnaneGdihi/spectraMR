"""Tests for ColdDiffusionInferenceStrategy resolution + knob wiring.

Focus: the 2026-06-12 inference audit fixes.
  * ``_resolve_accelerator`` must REUSE the trained generator's own accelerator
    (its ``kspace_process`` SSOT) rather than rebuilding a desynced one — a
    fresh accelerator masks k-space with a pattern the network never saw and
    collapses the reverse loop toward a DC blob (the cold-diffusion desync
    incident, project_exp11_true_cold_diffusion_fix).
  * an advertised-but-unimplemented ``degradation`` must RAISE (pitfall #15).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)


class _SentinelAccelerator:
    """Stands in for the trained ColdDiffusionAccelerator."""


class _FakeMaskGenerator:
    def __init__(self) -> None:
        self.requested_patterns: list = []
        self.accel = _SentinelAccelerator()

    def _get_accelerator(self, pattern):
        self.requested_patterns.append(pattern)
        return self.accel


class _FakeKSpaceProcess:
    def __init__(self, mask_type: str = "equispaced") -> None:
        self.mask_type = mask_type
        self.mask_generator = _FakeMaskGenerator()


class _ModelWithKspaceProcess(nn.Module):
    """Mirrors KSpaceColdDiffusionGenerator: no ``accelerator`` attr, but a
    ``kspace_process`` SSOT carrying the trained forward process."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.kspace_process = _FakeKSpaceProcess(mask_type="equispaced")

    def forward(self, x, t=None, **kwargs):
        return x


class _ModelWithAccelerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)
        self.accelerator = _SentinelAccelerator()

    def forward(self, x, t=None, **kwargs):
        return x


def test_resolve_reuses_kspace_process_accelerator():
    model = _ModelWithKspaceProcess()
    strat = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), {})
    # The strategy must return the generator's OWN trained accelerator object,
    # built with its trained mask_type — not a freshly-constructed one.
    assert strat.accelerator is model.kspace_process.mask_generator.accel
    assert model.kspace_process.mask_generator.requested_patterns == ["equispaced"]


def test_resolve_prefers_explicit_model_accelerator():
    model = _ModelWithAccelerator()
    strat = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), {})
    assert strat.accelerator is model.accelerator


def test_unsupported_degradation_raises():
    model = _ModelWithKspaceProcess()
    with pytest.raises(ValueError, match="kspace_mask"):
        ColdDiffusionInferenceStrategy(
            model, torch.device("cpu"), {"diffusion": {"degradation": "blur"}}
        )


def test_default_degradation_is_accepted():
    model = _ModelWithKspaceProcess()
    strat = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), {})
    assert strat.degradation_type == "kspace_mask"


def test_no_dead_cold_schedule_attribute():
    # The dead self.cold_schedule buffer (built then never read) was removed.
    model = _ModelWithKspaceProcess()
    strat = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), {})
    assert not hasattr(strat, "cold_schedule")


class _RecordingModel(nn.Module):
    """Deterministic denoiser that records the kwargs the strategy forwards."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_kwargs: list[set[str]] = []

    def forward(self, x, t=None, **kwargs):
        self.seen_kwargs.append(set(kwargs))
        return 0.9 * x + 0.05


def _bare_strategy(model=None):
    """Strategy with no accelerator (bare loop): timesteps 8, stride 2 -> [6,4,2,0]."""
    return ColdDiffusionInferenceStrategy(
        model or _RecordingModel(),
        torch.device("cpu"),
        {"training": {"diffusion": {"timesteps": 8, "sampling_steps": 4}}},
    )


def _measurement():
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(1, 2, 16, 16, generator=generator)
    mask = torch.zeros(1, 1, 16, 16)
    mask[..., ::2] = 1.0
    return full * mask, mask


class TestStepCallback:
    def test_fires_once_per_strided_step_descending(self):
        fired: list[tuple] = []
        measurement, mask = _measurement()
        out = _bare_strategy().run_inference(
            measurement.clone(),
            mask=mask,
            step_callback=lambda s, pred, m: fired.append((s, pred.clone(), m)),
        )
        assert [s for s, _, _ in fired] == [6, 4, 2, 0]
        # No accelerator resolved -> the loop has no per-step degradation mask.
        assert all(m is None for _, _, m in fired)
        # The final fire observes the exact tensor the loop returns (identity
        # denormalization here: no k-space normalization was configured).
        assert torch.equal(fired[-1][1], out)

    def test_none_callback_is_byte_identical(self):
        measurement, mask = _measurement()
        out_plain = _bare_strategy().run_inference(measurement.clone(), mask=mask)
        out_hooked = _bare_strategy().run_inference(
            measurement.clone(), mask=mask, step_callback=lambda s, pred, m: None
        )
        assert torch.equal(out_plain, out_hooked)

    def test_callback_is_never_forwarded_to_the_model(self):
        model = _RecordingModel()
        measurement, mask = _measurement()
        _bare_strategy(model).run_inference(
            measurement.clone(), mask=mask, step_callback=lambda s, pred, m: None
        )
        assert model.seen_kwargs, "model was never called"
        assert all("step_callback" not in seen for seen in model.seen_kwargs)

    def test_trajectory_monitor_plugs_in_as_the_callback(self):
        from spectramr.core.metrics.trajectory_metrics import TrajectoryMonitor

        measurement, mask = _measurement()
        monitor = TrajectoryMonitor(measurement, mask)
        _bare_strategy().run_inference(measurement.clone(), mask=mask, step_callback=monitor)
        summary = monitor.summary()
        assert summary["num_steps"] == 4
        assert summary["step_indices"] == [6, 4, 2, 0]
        assert all(k >= 0.0 for k in summary["kappa_per_step"])
        # Hard DC pins the observed support to the measurement every step, so
        # the on-support residual is exactly zero along the whole trajectory.
        assert summary["trajectory_kappa_max"] == 0.0


# ---------------------------------------------------------------------------
# S-maps reach the sampler in k-space, exactly as in training (#1297)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sampling_ffts_smaps_before_the_kspace_concat() -> None:
    """Sampling must build the same stack the training run was fitted on.

    ``x_t`` is k-space and the maps are image-domain, so the concat needs the
    same FFT + level-match + amplitude cap the strategy applies. Getting this
    right only in training would mean the network is sampled with a channel
    layout it never saw -- the reason the reverse loop is worth auditing at all.
    """
    import ast
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(ColdDiffusionInferenceStrategy))
    tree = ast.parse(src)
    prepared = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "prepare_smaps_for_kspace_conditioning"
    ]
    assert len(prepared) == 1, "expected exactly one preparation site"

    concats = [
        [e.id for e in n.args[0].elts if isinstance(e, ast.Name)]
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "cat"
        and n.args
        and isinstance(n.args[0], (ast.List, ast.Tuple))
    ]
    smap_concats = [names for names in concats if "smaps_k" in names]
    assert smap_concats, "the prepared maps are never concatenated"
    for names in smap_concats:
        assert "smaps" not in names, "raw image-domain maps still concatenated"


# ---------------------------------------------------------------------------
# The S-map gate reads the model's contract, not a magic channel count (#1326)
# ---------------------------------------------------------------------------


class _WidthRecordingModel(nn.Module):
    """Stub denoiser exposing a generator's conditioning contract."""

    def __init__(
        self,
        *,
        condition_with_smaps: bool,
        in_channels: int,
        expects_smaps_concat: bool | None = None,
    ) -> None:
        super().__init__()
        # The arm's *declaration* and the *resolved* contract are two different
        # facts, and an internal-DC backbone is where they part company. Leaving
        # ``expects_smaps_concat`` unset models a pre-contract generator, whose
        # answer falls back to the declaration.
        self.condition_with_smaps = condition_with_smaps
        if expects_smaps_concat is not None:
            self.expects_smaps_concat = expects_smaps_concat
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.widths: list[int] = []

    def forward(self, x, t=None, **kwargs):
        self.widths.append(x.shape[1])
        # Return the data half so the reverse loop's arithmetic stays well-shaped
        # whatever width the conditioning added.
        return 0.9 * x[:, : self.in_channels] + 0.05


class _DDPLike(nn.Module):
    """Minimal stand-in for a DistributedDataParallel wrapper."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _run(model, *, in_channels: int, config_in_channels=None, size: int = 16):
    """Drive two reverse steps and hand back the model."""
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )

    config: dict = {"training": {"diffusion": {"timesteps": 4, "sampling_steps": 2}}}
    if config_in_channels is not None:
        config["model"] = {"in_channels": config_in_channels}
    strategy = ColdDiffusionInferenceStrategy(model, torch.device("cpu"), config)
    generator = torch.Generator().manual_seed(0)
    full = torch.randn(1, in_channels, size, size, generator=generator)
    mask = torch.zeros(1, 1, size, size)
    mask[..., ::2] = 1.0
    strategy.run_inference(full * mask, mask=mask)
    return strategy


@pytest.mark.unit
@pytest.mark.parametrize("in_channels", [2, 4, 8, 16])
def test_conditioning_follows_condition_with_smaps_at_every_width(in_channels):
    """The trained width is ``2 * in_channels`` for every arm, not just 16.

    Training and validation concatenate S-maps for *every* cold-diffusion arm,
    so gating inference on ``in_channels == 16`` starved the other 91 arms in
    the corpus of the conditioning half they were fitted with.
    """
    model = _WidthRecordingModel(condition_with_smaps=True, in_channels=in_channels)
    _run(model, in_channels=in_channels, config_in_channels=in_channels)
    assert model.widths, "the model was never called"
    assert set(model.widths) == {2 * in_channels}


@pytest.mark.unit
@pytest.mark.parametrize("in_channels", [4, 8])
def test_internal_dc_backbone_is_sampled_at_its_built_width(in_channels):
    """A declaration of ``condition_with_smaps`` does not make the width 2x.

    ``diff_varnet`` and ``diff_varnet_kan`` honour the declaration and are still
    built at 1x, because they run data consistency inside the backbone. The six
    such arms in the corpus were handed a doubled stack here, which
    ``FourierBridgeNetwork`` absorbed by rebuilding an untrained 1x1
    ChannelAdapter -- so inference ran through random weights (#1326). Both the
    concat gate and ``_assert_trained_width`` must read the resolved contract.

    ``test_conditioning_follows_condition_with_smaps_at_every_width`` above
    cannot catch this: there the declaration and the contract agree.
    """
    model = _WidthRecordingModel(
        condition_with_smaps=True,
        expects_smaps_concat=False,
        in_channels=in_channels,
    )
    _run(model, in_channels=in_channels, config_in_channels=in_channels)
    assert model.widths, "the model was never called"
    assert set(model.widths) == {in_channels}, (
        f"internal-DC backbone sampled at {set(model.widths)}, not its built width {in_channels}"
    )


@pytest.mark.unit
def test_no_conditioning_when_the_model_declares_none():
    """``condition_with_smaps=False`` must reach the network unconditioned."""
    model = _WidthRecordingModel(condition_with_smaps=False, in_channels=8)
    _run(model, in_channels=8, config_in_channels=8)
    assert set(model.widths) == {8}


@pytest.mark.unit
def test_the_retired_in_channels_gate_no_longer_decides():
    """A 16-channel config must not conjure conditioning the model disowns.

    The mirror of the previous test: the old gate keyed off the *config*, so a
    model that declares no conditioning would still have been fed a doubled
    stack whenever the YAML happened to say 16.
    """
    model = _WidthRecordingModel(condition_with_smaps=False, in_channels=16)
    _run(model, in_channels=16, config_in_channels=16)
    assert set(model.widths) == {16}


@pytest.mark.unit
def test_the_contract_is_read_through_the_ddp_wrapper():
    """``condition_with_smaps`` lives on the generator, not on the wrapper."""
    inner = _WidthRecordingModel(condition_with_smaps=True, in_channels=8)
    _run(_DDPLike(inner), in_channels=8, config_in_channels=8)
    assert set(inner.widths) == {16}


@pytest.mark.unit
def test_single_channel_input_lifts_to_one_complex_coil_without_crashing():
    """Mirror of the validation path's ``c2 < 1`` branch.

    A 1-channel real field has no interleaved Re/Im pair, so ``c // 2`` is zero
    and ``view(b, 0, 2, h, w)`` blows up inside ``view_as_complex``.  The maps
    estimated from the lifted single coil add two real channels, giving a
    ``1 + 2 = 3`` stack — which is why the trained-width guard below is asserted
    for even ``in_channels`` only.
    """
    model = _WidthRecordingModel(condition_with_smaps=True, in_channels=1)
    _run(model, in_channels=1, config_in_channels=1)
    assert set(model.widths) == {3}


@pytest.mark.unit
def test_a_stack_that_is_not_the_trained_width_raises():
    """A width skew must be loud.

    ``FourierBridgeNetwork`` rebuilds its ``ChannelAdapter`` for whatever width
    arrives, so without this guard a mismatch is squeezed through an untrained
    1x1 convolution and merely reconstructs badly (#1326).
    """
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )

    model = _WidthRecordingModel(condition_with_smaps=True, in_channels=8)
    with pytest.raises(ValueError, match="trained on 16"):
        ColdDiffusionInferenceStrategy._assert_trained_width(torch.zeros(1, 12, 8, 8), model)


@pytest.mark.unit
def test_the_width_guard_stays_out_of_the_odd_channel_case():
    """An odd ``in_channels`` has no ``2 * C`` trained width to assert."""
    from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
        ColdDiffusionInferenceStrategy,
    )

    model = _WidthRecordingModel(condition_with_smaps=True, in_channels=1)
    ColdDiffusionInferenceStrategy._assert_trained_width(torch.zeros(1, 3, 8, 8), model)


@pytest.mark.unit
def test_the_configured_estimation_method_reaches_the_sampler(monkeypatch):
    """``physics.coil_processing.estimation`` is honored, not hardcoded away.

    Training and validation dispatch through ``estimate_smaps_calibrated``;
    pinning ``power_iter`` at this call site made the arm's declared method a
    silent no-op at sampling time (non-negotiable 8 / pitfall #15).
    """
    from spectramr.infrastructure.inference import cold_diffusion_inference_strategy as mod

    seen: list[tuple] = []
    real = mod.estimate_smaps_calibrated

    def spy(kspace, method="power_iter", **kwargs):
        seen.append((method, kwargs))
        return real(kspace, method="power_iter", out_size=kwargs.get("out_size"))

    monkeypatch.setattr(mod, "estimate_smaps_calibrated", spy)

    model = _WidthRecordingModel(condition_with_smaps=True, in_channels=8)
    strategy = mod.ColdDiffusionInferenceStrategy(
        model,
        torch.device("cpu"),
        {
            "training": {"diffusion": {"timesteps": 4, "sampling_steps": 2}},
            "physics": {"coil_processing": {"estimation": {"method": "espirit", "kernel_size": 5}}},
        },
    )
    gen = torch.Generator().manual_seed(0)
    full = torch.randn(1, 8, 32, 32, generator=gen)
    mask = torch.zeros(1, 1, 32, 32)
    mask[..., ::2] = 1.0
    strategy.run_inference(full * mask, mask=mask)

    assert seen, "estimate_smaps_calibrated was never called"
    method, kwargs = seen[0]
    assert method == "espirit"
    assert kwargs["kernel_size"] == 5
    # The maps must land on the image grid the sampler works on. Confinement to
    # the dense centre is no longer the caller's to pass -- the owner applies it,
    # which is what lets the maps come back full-resolution instead of cropped
    # and then interpolated back up (#2213).
    assert kwargs["out_size"] == (32, 32)


# ---------------------------------------------------------------------------
# physics.data_consistency.apply_at_predict: the loop already pins the
# measured bins (``pred_x0 * (1 - m) + input * m`` at every step), so the base
# hook must record that and not project a second time.
# ---------------------------------------------------------------------------


def _knob_on_strategy(monkeypatch, model=None):
    from spectramr.infrastructure.inference import predict_data_consistency as pdc
    from spectramr.models.capabilities import ModelCapabilities

    caps = ModelCapabilities(output_domain="kspace")
    monkeypatch.setitem(pdc.MODEL_REGISTRY, "stub_cold", {"capabilities": caps})
    monkeypatch.setattr(pdc, "get_model_capabilities", lambda n: caps if n == "stub_cold" else None)
    return ColdDiffusionInferenceStrategy(
        model or _RecordingModel(),
        torch.device("cpu"),
        {
            "training": {"diffusion": {"timesteps": 8, "sampling_steps": 4}},
            "physics": {"data_consistency": {"apply_at_predict": True}},
            "model": {"model_type": "stub_cold"},
            "data": {"dataset_type": "kspace"},
        },
    )


class TestPredictDataConsistencyLedger:
    def test_the_loop_records_its_pin_and_the_hook_does_not_project_again(self, monkeypatch):
        strategy = _knob_on_strategy(monkeypatch)
        assert strategy.predict_dc is not None
        projections: list[int] = []
        real = strategy.predict_dc.project
        monkeypatch.setattr(
            strategy.predict_dc, "project", lambda *a, **k: projections.append(1) or real(*a, **k)
        )
        measurement, mask = _measurement()

        out = strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)

        assert projections == [], "cold diffusion pinned the bins itself"
        on = mask.bool().expand_as(out)
        assert torch.equal(out[on], measurement[on]), "the loop's own pin holds on the output"
        prov = strategy.predict_dc_provenance()
        assert prov["applied_by"] == {"ColdDiffusionInferenceStrategy.run_inference": 1}
        assert prov["skipped_already_applied"] == 1

    def test_measured_kspace_never_reaches_the_model(self, monkeypatch):
        """The loop forwards ``dict(kwargs)`` into the model; the base pops this one."""
        model = _RecordingModel()
        strategy = _knob_on_strategy(monkeypatch, model)
        measurement, mask = _measurement()
        strategy.infer(measurement.clone(), mask=mask, measured_kspace=measurement)
        assert model.seen_kwargs
        assert all("measured_kspace" not in seen for seen in model.seen_kwargs)

    def test_without_a_mask_the_loop_pins_nothing_and_the_hook_raises(self, monkeypatch):
        from spectramr.domain.exceptions import ConfigurationError

        strategy = _knob_on_strategy(monkeypatch)
        measurement, _mask = _measurement()
        with pytest.raises(ConfigurationError, match="carries no mask"):
            strategy.infer(measurement.clone(), measured_kspace=measurement)

    def test_off_knob_is_byte_identical(self):
        measurement, mask = _measurement()
        strategy = _bare_strategy()
        assert strategy.predict_dc is None
        a = _bare_strategy().run_inference(measurement.clone(), mask=mask)
        b = strategy.run_inference(measurement.clone(), mask=mask)
        assert torch.equal(a, b)
