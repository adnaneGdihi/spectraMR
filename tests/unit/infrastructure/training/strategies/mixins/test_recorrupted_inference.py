"""``recorrupted_inference``: the R2R Monte-Carlo estimator scored at validation (#2329).

Pinned, each with the violation that turns it red:

* the knob is refused on an arm whose training target is not an R2R mode;
* J = 1 with a zero draw IS the plain forward, so the wrapper adds nothing but
  the recorruption;
* on a linear model the output variance falls by 1/J (the estimator averages);
* the draw is in the input's normalized units -- Sigma / kspace_scale^2 --
  and a batch with no scale raises instead of recorrupting 20x too hard;
* the J forwards run one at a time on the unenlarged batch;
* the null-space mode feeds the input half of a real ``split_kspace`` call.
"""

from __future__ import annotations

import types

import pytest
import torch

import spectramr.infrastructure.training.strategies.mixins.recorrupted_inference as ri
from spectramr.config.schemas.validation import RecorruptedInferenceConfigSchema
from spectramr.core.metrics.coil_manifold import as_complex_coils, as_interleaved
from spectramr.data.batch_types import BatchAdapter
from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance

_B, _C, _H, _W = 2, 4, 32, 32


def _config(mode: str = "r2r", *, samples: int = 5, **knob) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        data=types.SimpleNamespace(
            target_mode=mode,
            r2r_alpha=0.5,
            r2r_covariance_source="committed",
            r2r_split_angle_deg=30.0,
            r2r_whitening="committed",
        ),
        validation=types.SimpleNamespace(
            scoring=types.SimpleNamespace(
                recorrupted_inference=RecorruptedInferenceConfigSchema(samples=samples, **knob)
            )
        ),
        run=types.SimpleNamespace(seed=7),
    )


def _input(zero_cols: int = 4, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    y = torch.complex(
        torch.randn(_B, _C, _H, _W, generator=gen), torch.randn(_B, _C, _H, _W, generator=gen)
    )
    y[..., _W - zero_cols :] = 0
    return as_interleaved(y)


def _batch(x: torch.Tensor, scale: float = 10.0, **extra) -> object:
    return BatchAdapter.from_dict(
        {"input": x, "target": x, "kspace_scale": torch.full((_B,), scale), **extra}
    )


def _estimator(mode: str = "r2r", **kw) -> ri.RecorruptedInference:
    spec = ri.resolve_recorrupted_inference(_config(mode, **kw))
    assert spec is not None
    return ri.RecorruptedInference(spec)


class TestTheKnob:
    def test_off_resolves_to_nothing(self) -> None:
        config = _config()
        config.validation.scoring.recorrupted_inference = None
        assert ri.resolve_recorrupted_inference(config) is None

    @pytest.mark.parametrize("mode", ["phase_aligned_mean", "rep_pair", "complex_mean"])
    def test_refused_where_the_network_never_saw_a_recorruption(self, mode) -> None:
        with pytest.raises(ValueError, match="only an R2R-trained network"):
            ri.resolve_recorrupted_inference(_config(mode))

    def test_the_strength_defaults_to_the_trained_one(self) -> None:
        assert ri.resolve_recorrupted_inference(_config("r2r")).alpha == 0.5
        assert ri.resolve_recorrupted_inference(_config("nullspace_r2r")).angle_deg == 30.0

    def test_an_explicit_strength_overrides(self) -> None:
        assert ri.resolve_recorrupted_inference(_config("r2r", alpha=0.2)).alpha == 0.2

    @pytest.mark.parametrize(
        ("mode", "knob"), [("r2r", {"angle_deg": 20.0}), ("nullspace_r2r", {"alpha": 0.3})]
    )
    def test_the_other_modes_strength_is_refused(self, mode, knob) -> None:
        with pytest.raises(ValueError, match="strength"):
            ri.resolve_recorrupted_inference(_config(mode, **knob))

    def test_the_schema_refuses_both_strengths_and_zero_samples(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="never both"):
            RecorruptedInferenceConfigSchema(samples=5, alpha=1.0, angle_deg=45.0)
        with pytest.raises(ValidationError, match="samples"):
            RecorruptedInferenceConfigSchema(samples=0)


class TestTheR2REstimator:
    def test_one_sample_with_a_zero_draw_is_the_plain_forward(self, monkeypatch) -> None:
        monkeypatch.setattr(
            ri, "draw_r2r_noise", lambda chol, shape, gen: torch.zeros(shape, dtype=chol.dtype)
        )
        x = _input()
        out = _estimator(samples=1).predict(lambda v: 3.0 * v + 1.0, x, _batch(x))
        torch.testing.assert_close(out, 3.0 * x + 1.0)

    def test_averaging_cuts_the_variance_by_one_over_j(self) -> None:
        """Linear spy: out - y is the mean recorruption, variance alpha^2 Sigma / (J s^2)."""
        x = _input()
        y = as_complex_coils(x)
        spread = {
            j: float(
                (as_complex_coils(_estimator(samples=j).predict(lambda v: v, x, _batch(x))) - y)
                .abs()
                .pow(2)
                .mean()
            )
            for j in (1, 5)
        }
        assert spread[1] / spread[5] == pytest.approx(5.0, rel=0.1)

    def test_the_draw_is_in_the_inputs_normalized_units(self) -> None:
        """Planted: dividing Sigma by nothing would inflate every coil's draw by s^2 = 100."""
        x = _input(zero_cols=0)
        scale = 10.0
        out = _estimator(samples=1).predict(lambda v: v, x, _batch(x, scale=scale))
        z = as_complex_coils(out) - as_complex_coils(x)
        measured = z.abs().pow(2).mean(dim=(0, 2, 3))
        expected = 0.5**2 * single_repetition_covariance().diagonal().real / scale**2
        torch.testing.assert_close(measured, expected.float(), rtol=0.1, atol=0.0)

    def test_unacquired_columns_stay_zero(self) -> None:
        x = _input(zero_cols=4)
        out = _estimator(samples=3).predict(lambda v: v, x, _batch(x))
        assert torch.count_nonzero(out[..., _W - 4 :]) == 0

    def test_a_batch_without_its_scale_raises(self) -> None:
        x = _input()
        batch = BatchAdapter.from_dict({"input": x, "target": x})
        with pytest.raises(ValueError, match="kspace_scale"):
            _estimator().predict(lambda v: v, x, batch)

    def test_the_forwards_are_sequential_on_the_batch_as_given(self) -> None:
        x = _input()
        seen: list[tuple[int, ...]] = []
        _estimator(samples=5).predict(lambda v: seen.append(tuple(v.shape)) or v, x, _batch(x))
        assert seen == [tuple(x.shape)] * 5

    def test_a_self_calibrated_arm_draws_from_the_published_sigma(self) -> None:
        config = _config()
        config.data.r2r_covariance_source = "self_calibrated"
        estimator = ri.RecorruptedInference(ri.resolve_recorrupted_inference(config))
        x = _input(zero_cols=0)
        with pytest.raises(ValueError, match="noise_covariance"):
            estimator.predict(lambda v: v, x, _batch(x))
        own = 9.0 * torch.eye(_C, dtype=torch.complex64).expand(_B, _C, _C)
        out = estimator.predict(lambda v: v, x, _batch(x, scale=1.0, noise_covariance=own))
        z = as_complex_coils(out) - as_complex_coils(x)
        # alpha^2 * 9 / J with J = 5.
        assert float(z.abs().pow(2).mean()) == pytest.approx(0.25 * 9.0 / 5, rel=0.1)

    def test_a_batch_index_repeats_its_draws_across_epochs(self) -> None:
        """Validation epoch k and k+1 score batch i under the same recorruptions."""
        x = _input()
        estimator = _estimator(samples=2)
        first = estimator.predict(lambda v: v, x, _batch(x), batch_key=3)
        other = estimator.predict(lambda v: v, x, _batch(x), batch_key=4)
        again = estimator.predict(lambda v: v, x, _batch(x), batch_key=3)
        assert torch.equal(first, again)
        assert not torch.allclose(first, other)

    def test_a_second_call_on_one_batch_draws_afresh(self) -> None:
        """pred_alt follows the prediction on the same batch; shared draws would
        cancel out of rep_variance."""
        x = _input()
        estimator = _estimator(samples=2)
        first = estimator.predict(lambda v: v, x, _batch(x), batch_key=0)
        second = estimator.predict(lambda v: v, x, _batch(x))
        assert not torch.allclose(first, second)

    def test_a_single_batch_repeats_across_epochs_too(self) -> None:
        """The index reseeds on every call carrying it, not only when it changes."""
        x = _input()
        estimator = _estimator(samples=2)
        first = estimator.predict(lambda v: v, x, _batch(x), batch_key=0)
        estimator.predict(lambda v: v, x, _batch(x))  # pred_alt of epoch k
        again = estimator.predict(lambda v: v, x, _batch(x), batch_key=0)
        assert torch.equal(first, again)

    def test_its_draws_leave_the_global_rng_alone(self) -> None:
        x = _input()
        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        _estimator().predict(lambda v: v, x, _batch(x))
        assert torch.equal(torch.rand(3), expected)


def _maps() -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, _H), torch.linspace(-1, 1, _W), indexing="ij")
    centres = [(-0.6, -0.6), (0.6, -0.6), (-0.6, 0.6), (0.6, 0.6)]
    sens = torch.stack([torch.exp(-((xx - a) ** 2 + (yy - b) ** 2)) for a, b in centres])
    sens = (sens / sens.pow(2).sum(0, keepdim=True).sqrt()).to(torch.complex64)
    return sens.expand(_B, _C, _H, _W).clone()


class TestTheNullspaceEstimator:
    def test_each_sample_is_the_input_half_of_a_real_split(self, monkeypatch) -> None:
        import spectramr.infrastructure.physics.coil_split_r2r as split_mod

        halves: list[torch.Tensor] = []
        real = split_mod.split_kspace_prepared
        monkeypatch.setattr(
            split_mod,
            "split_kspace_prepared",
            lambda *a, **k: (lambda half: halves.append(half) or half)(real(*a, **k)),
        )
        x = _input()
        seen: list[torch.Tensor] = []
        maps = _maps()
        batch = _batch(x, sensitivity_complex=maps)
        _estimator("nullspace_r2r", samples=3).predict(lambda v: seen.append(v) or v, x, batch)
        assert len(halves) == 3
        for fed, half in zip(seen, halves, strict=True):
            torch.testing.assert_close(fed, as_interleaved(half))
        assert not torch.allclose(seen[0], seen[1])

    def test_magnitude_maps_are_refused(self) -> None:
        x = _input()
        batch = _batch(x, sensitivity=_maps().abs())
        with pytest.raises(ValueError, match=r"(?i)complex"):
            _estimator("nullspace_r2r").predict(lambda v: v, x, batch)

    def test_no_maps_raise(self) -> None:
        x = _input()
        with pytest.raises(ValueError):
            _estimator("nullspace_r2r").predict(lambda v: v, x, _batch(x))


def test_the_per_mode_tables_cover_exactly_the_eligible_modes() -> None:
    """The resolver reads two tables instead of an if-chain; they must stay in step."""
    from spectramr.infrastructure.training.strategies.mixins import recorrupted_inference as mod

    assert set(mod._MODE_FIELDS) == mod.ELIGIBLE_TARGET_MODES
    assert set(mod._OTHER_MODES_STRENGTH) == mod.ELIGIBLE_TARGET_MODES
