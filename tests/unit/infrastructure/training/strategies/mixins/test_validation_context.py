"""Reading metric context off a validation batch: routing, spelling, shape.

The routing rule is the one that protects existing arms: a metric is handed a
context only when this path can fill every field it declares, so an NR metric
that needs a sampling mask keeps the context-free call it had before.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.core.metrics.coil_manifold import as_interleaved
from spectramr.data.batch_types import BatchAdapter
from spectramr.infrastructure.training.strategies.mixins.validation_context import (
    SUPPLIED_FIELDS,
    alternate_input,
    batch_tensor,
    coil_maps,
    context_metric_names,
    flatten_like,
    project_prediction,
)


def _complex(*shape: int, seed: int = 0) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.complex(torch.randn(*shape, generator=gen), torch.randn(*shape, generator=gen))


def _computer(names: list[str]):
    return types.SimpleNamespace(get_metric_names=lambda: names)


class TestRouting:
    def test_only_fully_suppliable_context_metrics_are_routed(self) -> None:
        names = ["psnr", "null_space_residual", "ndcr", "rep_variance", "csoe"]
        routed = context_metric_names(_computer(names))
        assert routed == ("null_space_residual", "rep_variance", "csoe")

    def test_a_metric_needing_a_mask_keeps_its_old_call(self) -> None:
        """ndcr needs `mask`, which this path does not supply."""
        from spectramr.core.metrics.registry import MetricsRegistry

        assert "mask" in MetricsRegistry.needs("ndcr") and "mask" not in SUPPLIED_FIELDS
        assert context_metric_names(_computer(["ndcr"])) == ()

    def test_a_computer_that_cannot_list_its_metrics_routes_nothing(self) -> None:
        assert context_metric_names(object()) == ()


class TestReadingTheBatch:
    def test_the_complex_spelling_wins_over_the_bound_magnitude(self) -> None:
        """BatchAdapter binds `sensitivity` (a magnitude) to coil_maps."""
        maps = _complex(2, 4, 8, 8)
        batch = BatchAdapter.from_dict(
            {
                "input": torch.zeros(2, 8, 8, 8),
                "target": torch.zeros(2, 8, 8, 8),
                "sensitivity": maps.abs(),
                "sensitivity_complex": maps,
            }
        )
        assert not torch.is_complex(batch.coil_maps)
        found, key = coil_maps(batch, torch.zeros(2, 8, 8, 8))
        assert key == "sensitivity_complex" and torch.equal(found, maps)

    def test_a_torchio_wrapper_is_unwrapped_from_a_dict(self) -> None:
        tensor = torch.ones(1, 2, 3, 3)
        assert batch_tensor({"input_alt": {"data": tensor}}, "input_alt") is tensor

    def test_absent_is_none(self) -> None:
        assert batch_tensor(None, "input_alt") is None
        assert coil_maps({"input": torch.zeros(1)}, torch.zeros(1, 2, 3, 3)) == (None, None)


class TestShapes:
    def test_depth_folds_into_the_batch_like_the_loop_does(self) -> None:
        volume = torch.arange(2 * 3 * 4 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 4, 5)
        flat = flatten_like(volume, torch.zeros(10, 6, 4, 4), name="probe")
        assert flat.shape == (10, 3, 4, 4)
        assert torch.equal(flat[1], volume[0, ..., 1])  # sample 0, slice 1

    def test_a_spatial_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="square-pads"):
            flatten_like(torch.zeros(2, 3, 4, 5), torch.zeros(2, 6, 4, 4), name="probe")

    def test_a_complex_input_alt_arrives_interleaved(self) -> None:
        alt = _complex(2, 4, 8, 8, 1, seed=1)
        out = alternate_input({"input_alt": alt}, torch.zeros(2, 8, 8, 8))
        assert torch.equal(out, as_interleaved(alt[..., 0]))

    def test_an_input_alt_in_another_layout_raises(self) -> None:
        with pytest.raises(ValueError, match="does not match input"):
            alternate_input({"input_alt": torch.zeros(2, 4, 8, 8)}, torch.zeros(2, 8, 8, 8))


class TestProjection:
    def test_an_image_output_is_refused(self) -> None:
        with pytest.raises(ValueError, match="output_type is 'image'"):
            project_prediction(torch.zeros(1, 8, 4, 4), {}, "image")

    def test_it_reports_where_the_maps_came_from(self) -> None:
        maps = _complex(1, 4, 4, 4, seed=2)
        _, key = project_prediction(
            torch.randn(1, 8, 4, 4), {"sensitivity_complex": maps}, "kspace"
        )
        assert key == "sensitivity_complex"


class TestComputeContextMetrics:
    """The caller names the domain, and a forward blind to its input is refused."""

    def _args(self):
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig

        computer = ValidationMetricsComputer(
            ValidationMetricsConfig.from_dict({"metrics": ["rep_variance"]})
        )
        prediction = as_interleaved(_complex(2, 4, 6, 10, seed=4))  # non-square
        batch = {"input_alt": as_interleaved(_complex(2, 4, 6, 10, seed=5))}
        model = types.SimpleNamespace(input_type="image", output_type="image")
        return computer, prediction, batch, model

    def test_the_domain_is_an_explicit_argument(self) -> None:
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer, prediction, batch, model = self._args()
        with pytest.raises(TypeError, match="signal_domain"):
            compute_context_metrics(
                computer, ("rep_variance",), prediction, prediction, prediction, batch, model,
                lambda x: x,
            )
        out = compute_context_metrics(
            computer, ("rep_variance",), prediction, prediction, prediction, batch, model,
            lambda x: x, signal_domain="kspace",
        )
        assert float(out["val_rep_variance"]) > 0

    def test_pred_alt_equal_to_the_prediction_raises(self) -> None:
        """Planted: a forward that returns the same tensor for both repetitions."""
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer, prediction, batch, model = self._args()
        with pytest.raises(ValueError, match="equals its output on `input`"):
            compute_context_metrics(
                computer, ("rep_variance",), prediction, prediction, prediction, batch, model,
                lambda _x: prediction.clone(), signal_domain="kspace",
            )

    def test_an_all_zero_output_on_both_repetitions_is_not_the_defect(self) -> None:
        """A zero-map slice projects to zeros on both repetitions; that is no ignored input.

        The calibration script's projection estimator produced exactly this and
        the guard raised on the slice the script exists to count.
        """
        import torch

        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer, prediction, batch, model = self._args()
        zeros = torch.zeros_like(prediction)
        out = compute_context_metrics(
            computer, ("rep_variance",), zeros, zeros, zeros, batch, model,
            lambda _x: zeros.clone(), signal_domain="kspace",
        )
        assert float(out["val_rep_variance"]) == 0.0


class TestFoldDepth:
    """TorchIO's (B, C, H, W, D) folds into the batch once and unfolds exactly."""

    def test_five_d_round_trips_and_four_d_passes_through(self) -> None:
        import torch

        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            fold_depth,
            unfold_depth,
        )

        x = torch.randn(2, 8, 6, 5, 3)
        folded, layout = fold_depth(x)
        assert folded.shape == (6, 8, 6, 5) and layout == (2, 3)
        assert torch.equal(unfold_depth(folded, layout), x)
        assert torch.equal(folded[1], x[0, ..., 1]), "depth is the fast axis inside the batch"
        four = torch.randn(2, 8, 6, 5)
        same, none = fold_depth(four)
        assert none is None and same is four and unfold_depth(same, none) is four


class TestScoringMaps:
    """#2664: sense_* combine with the validation-only scoring maps when present."""

    def _batch(self, *, with_reference: bool):
        from spectramr.core.coil_map_names import REFERENCE_MAPS_KEY

        training_maps = _complex(2, 4, 16, 16, seed=1)
        scoring_maps = _complex(2, 4, 16, 16, seed=2)
        data = {"sensitivity_complex": training_maps}
        if with_reference:
            data[REFERENCE_MAPS_KEY] = scoring_maps
        return data, training_maps, scoring_maps

    def _sense_psnr(self, batch, prediction, target) -> float:
        from spectramr.core.metrics.computer import create_validation_metrics_computer
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer = create_validation_metrics_computer(["sense_psnr"], device="cpu", data_range=1.0)
        model = types.SimpleNamespace(input_type="kspace", output_type="kspace")
        out = compute_context_metrics(
            computer, ("sense_psnr",), prediction, target, prediction, batch, model,
            lambda x: x, signal_domain="kspace",
        )  # fmt: skip
        return float(out["val_sense_psnr"])

    def _expected(self, prediction, target, maps) -> float:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        metric = MetricsRegistry.get("sense_psnr")
        context = MetricContext(coil_maps=maps, signal_domain="kspace")
        return float(metric(prediction, target, context=context, data_range=1.0))

    def test_the_scoring_maps_win_when_the_batch_carries_them(self) -> None:
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            scoring_maps,
        )

        batch, training_maps, scoring = self._batch(with_reference=True)
        prediction = as_interleaved(_complex(2, 4, 16, 16, seed=3))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        assert torch.equal(scoring_maps(batch, prediction)[0], scoring)
        assert torch.equal(coil_maps(batch, prediction)[0], training_maps)  # operators unchanged
        got = self._sense_psnr(batch, prediction, target)
        assert got == pytest.approx(self._expected(prediction, target, scoring), rel=1e-5)

    def test_planted_without_them_the_training_maps_score(self) -> None:
        batch, training_maps, scoring = self._batch(with_reference=False)
        prediction = as_interleaved(_complex(2, 4, 16, 16, seed=3))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        got = self._sense_psnr(batch, prediction, target)
        assert got == pytest.approx(self._expected(prediction, target, training_maps), rel=1e-5)
        assert got != pytest.approx(self._expected(prediction, target, scoring), rel=1e-3)

    def _context(self, batch, prediction, target, names, **kw) -> dict[str, float]:
        from spectramr.core.metrics.computer import create_validation_metrics_computer
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer = create_validation_metrics_computer(list(names), device="cpu", data_range=1.0)
        model = types.SimpleNamespace(input_type="kspace", output_type="kspace")
        return compute_context_metrics(
            computer, tuple(names), prediction, target, prediction, batch, model,
            lambda x: x, signal_domain="kspace", **kw,
        )  # fmt: skip

    def test_the_corrected_mse_reads_the_scoring_maps_too(self) -> None:
        """The noise term must rest on the map set every family is scored with."""
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, training_maps, scoring = self._batch(with_reference=True)
        prediction = as_interleaved(_complex(2, 4, 16, 16, seed=3))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        got = self._context(batch, prediction, target, ["reference_noise_corrected_mse"])
        metric = MetricsRegistry.get("reference_noise_corrected_mse")

        def expected(maps):
            ctx = MetricContext(coil_maps=maps, signal_domain="kspace")
            return float(metric(prediction, target, context=ctx))

        value = got["val_reference_noise_corrected_mse"]
        assert value == pytest.approx(expected(scoring), rel=1e-5)
        # Planted: the training maps, which the noise term used to read.
        assert value != pytest.approx(expected(training_maps), rel=1e-3)

    def test_an_image_prediction_is_scored_as_it_is(self) -> None:
        """An image network's x_hat, not |S_ref^H S_train x_hat| (validation audit 2026-10-09)."""
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, training_maps, scoring = self._batch(with_reference=True)
        image = as_interleaved(_complex(2, 1, 16, 16, seed=5))
        # What a strategy's context adapter grades: F S_train x_hat.
        from spectramr.infrastructure.physics.fft_ops import fft2c

        re_expanded = as_interleaved(fft2c(training_maps * torch.complex(image[:, 0:1], image[:, 1:2])))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        got = self._context(batch, re_expanded, target, ["sense_psnr"], image_prediction=image)
        ctx = MetricContext(coil_maps=scoring, signal_domain="kspace")
        metric = MetricsRegistry.get("sense_psnr")
        assert got["val_sense_psnr"] == pytest.approx(
            float(metric(image, target, context=ctx, data_range=1.0)), rel=1e-5
        )
        planted = self._context(batch, re_expanded, target, ["sense_psnr"])
        assert planted["val_sense_psnr"] != pytest.approx(got["val_sense_psnr"], rel=1e-3)
