"""Validation and predict reconstruct through one reverse path (#2067, #2162).

The generator here is a stand-in network carrying the production
``KSpaceColdDiffusionGenerator.sample``, so the sampler, the schedule, the DC
step and the centre-patch passthrough are the real ones. Its forward is row-wise
and carries an error proportional to ``t``, which makes the timestep of every
call visible in the reconstruction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.inference.cold_diffusion_inference_strategy import (
    ColdDiffusionInferenceStrategy,
)
from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.models.diffusion.reverse_path import (
    resolve_reverse_steps,
    sample_reverse_path,
    takes_coil_maps,
)
from spectramr.models.diffusion.reverse_physics import resolve_reverse_physics_kwargs
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceColdDiffusionGenerator,
)

T = 8
STEPS = 4


class _BlendDC(torch.nn.Module):
    """A learned-DC stand-in that moves the prediction halfway to the measurement."""

    def forward(self, x, measured, mask, **_kw):
        return 0.5 * x + 0.5 * measured


class _Generator(torch.nn.Module):
    """Stand-in network behind the production ``sample()``."""

    sample = KSpaceColdDiffusionGenerator.sample
    mask_scope = KSpaceColdDiffusionGenerator.mask_scope
    last_reverse_stats = KSpaceColdDiffusionGenerator.last_reverse_stats

    def __init__(self, target, *, mode="replace_freeze_dc", dc="hard", sigma=0.0, passthrough=None):
        super().__init__()
        self.target = target
        # One pattern for every row, so the output of a row never depends on
        # which chunk it is sampled in.
        self.noise = torch.randn(1, *target.shape[1:], generator=torch.Generator().manual_seed(5))
        self.calls: list[int] = []
        self.num_timesteps = T
        self.kspace_process = SimpleNamespace(max_accel=4.0, center_fraction=0.08)
        self.dc_method = dc
        self.dc_layer = _BlendDC()
        self._sampling_steps = STEPS
        self._sampler_name = "cold_mri"
        self._dc_method = dc
        self._dc_weight = 1.0
        self._reverse_mode = mode
        self._reverse_clip_ratio = 4.0
        self._clip_reference = "global_max"
        self._sampler_sigma = sigma
        self._sampler_seed = 7 if sigma else None
        self._selection_rule = "fixed"
        self._refine_alpha = None
        self._integrity_check = "off"
        self._reverse_physics = resolve_reverse_physics_kwargs(mode, dc, {})
        self._kspace_log_scaled = False
        self._last_reverse_stats = None
        self._dc_passthrough_size = passthrough

    def set_current_smaps(self, v):
        self._current_smaps = v

    def set_current_mask(self, v):
        self._current_mask = v

    def set_current_contrast_idx(self, v):
        self._current_contrast_idx = v

    def set_current_reference(self, v):
        self._current_reference = v

    def forward(self, x, t, **_kw):
        tt = int(t.reshape(-1)[0])
        self.calls.append(tt)
        return 0.5 * x + 0.5 * self.target.mean(0, keepdim=True) + 0.05 * tt * self.noise


def _inputs(batch: int = 3):
    g = torch.Generator().manual_seed(0)
    target = torch.randn(batch, 2, 16, 16, generator=g)
    mask = torch.zeros(batch, 1, 16, 16)
    mask[..., ::2] = 1.0
    mask[..., 6:10] = 1.0
    return target, target * mask, mask


def _validate(gen, measurement, mask, chunk):
    strategy = SimpleNamespace(
        config=SimpleNamespace(
            validation=SimpleNamespace(
                sampling=SimpleNamespace(steps=None), loader=SimpleNamespace(chunk_size=chunk)
            ),
            training=SimpleNamespace(diffusion=SimpleNamespace(sampling_steps=STEPS)),
        ),
        num_timesteps=T,
    )
    steps = DiffusionTrainingStrategy._resolve_validation_sampling_steps(strategy)
    return DiffusionTrainingStrategy._sample_multistep_chunked(
        strategy, gen, measurement, mask, steps
    )


def _predict(gen, measurement, mask, chunk):
    config = {
        "training": {"diffusion": {"timesteps": T, "sampling_steps": STEPS}},
        "validation": {"loader": {"chunk_size": chunk}},
    }
    return ColdDiffusionInferenceStrategy(gen, torch.device("cpu"), config).run_inference(
        measurement.clone(), mask=mask
    )


@pytest.mark.parametrize("sigma", [0.0, 0.1], ids=["deterministic", "seeded_noise"])
@pytest.mark.parametrize("mode", ["replace_freeze_dc", "additive"])
def test_validation_and_predict_return_the_same_reconstruction(sigma, mode):
    target, measurement, mask = _inputs()
    val_gen = _Generator(target, mode=mode, sigma=sigma)
    pred_gen = _Generator(target, mode=mode, sigma=sigma)

    val = _validate(val_gen, measurement, mask, chunk=2)
    pred = _predict(pred_gen, measurement, mask, chunk=2)

    assert torch.equal(val, pred)
    assert val_gen.calls == pred_gen.calls, "the model saw the same timesteps in the same order"


def test_the_chunk_size_is_part_of_the_contract_under_noise():
    """Planted: ``sample()`` reseeds per call, so a different partition draws different noise."""
    target, measurement, mask = _inputs()
    val = _validate(_Generator(target, sigma=0.1), measurement, mask, chunk=2)
    pred = _predict(_Generator(target, sigma=0.1), measurement, mask, chunk=3)
    assert not torch.equal(val, pred)


def test_any_chunking_is_exact_without_noise():
    target, measurement, mask = _inputs()
    whole = sample_reverse_path(
        _Generator(target), measurement, mask=mask, steps=STEPS, chunk_size=3
    )
    split = sample_reverse_path(
        _Generator(target), measurement, mask=mask, steps=STEPS, chunk_size=1
    )
    assert torch.equal(whole, split)


def test_a_chunk_size_below_one_raises():
    target, measurement, mask = _inputs()
    with pytest.raises(ValueError, match="chunk_size"):
        sample_reverse_path(_Generator(target), measurement, mask=mask, steps=STEPS, chunk_size=0)


@pytest.mark.parametrize(
    ("val_steps", "diff_steps", "horizon", "expected"),
    [(None, None, 8, None), (None, 4, 8, 4), (3, 4, 8, 3), (50, None, 16, 16)],
)
def test_one_step_rule_serves_both_callers(val_steps, diff_steps, horizon, expected):
    assert resolve_reverse_steps(val_steps, diff_steps, horizon) == expected


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [("kspace_cold_diffusion", True), ("KSpace_Cold_Diffusion", True), ("unet", False), (None, False)],
)
def test_one_coil_map_predicate_serves_every_caller(model_type, expected):
    assert takes_coil_maps(model_type) is expected


class TestCentrePatchPassthrough:
    """#2162: the declared passthrough now runs on the reverse path, after the last DC."""

    @staticmethod
    def _centre(x):
        return x[..., x.shape[-2] // 2, x.shape[-1] // 2]

    def test_it_pins_the_sampled_centre_bin_under_learned_dc(self):
        target, measurement, mask = _inputs(batch=1)
        gen = _Generator(target, dc="adaptive", passthrough=(1, 1))
        out = gen.sample(measurement, mask=mask)
        assert torch.equal(self._centre(out), self._centre(measurement))

    def test_without_it_learned_dc_leaves_the_centre_off_the_measurement(self):
        """Planted: the blend DC alone does not reach the measured value."""
        target, measurement, mask = _inputs(batch=1)
        out = _Generator(target, dc="adaptive").sample(measurement, mask=mask)
        assert not torch.equal(self._centre(out), self._centre(measurement))

    def test_under_hard_dc_it_changes_nothing(self):
        target, measurement, mask = _inputs(batch=1)
        with_pt = _Generator(target, passthrough=(1, 1)).sample(measurement, mask=mask)
        without = _Generator(target).sample(measurement, mask=mask)
        assert torch.equal(with_pt, without)


def test_measured_lines_are_split_with_the_chunks():
    """Each chunk's rungs are built over its own rows' acquired lines (#2160)."""
    seen: list[torch.Tensor] = []

    class _Recorder:
        def sample(self, measurement, mask=None, **kwargs):
            seen.append(kwargs["measured_lines"])
            return measurement

    measurement = torch.randn(5, 2, 4, 6)
    lines = torch.rand(5, 6) > 0.3
    sample_reverse_path(
        _Recorder(), measurement, mask=None, steps=None, chunk_size=2, measured_lines=lines
    )
    assert [s.shape[0] for s in seen] == [2, 2, 1]
    assert torch.equal(torch.cat(seen), lines)
