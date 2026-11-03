"""Reading metric context off a validation batch: routing, spelling, shape.

The routing rule is the one that protects existing arms: a metric is handed a
context only when this path can fill every field it declares, so an NR metric
that needs a reconstructor keeps the context-free call it had before.
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
        names = ["psnr", "null_space_residual", "dpst", "rep_variance", "csoe"]
        routed = context_metric_names(_computer(names))
        assert routed == ("null_space_residual", "rep_variance", "csoe")

    def test_a_metric_needing_a_prior_model_keeps_its_old_call(self) -> None:
        """dpst needs `prior_model`, which this path does not supply."""
        from spectramr.core.metrics.registry import MetricsRegistry

        assert "prior_model" in MetricsRegistry.needs("dpst")
        assert "prior_model" not in SUPPLIED_FIELDS
        assert context_metric_names(_computer(["dpst"])) == ()

    def test_a_metric_needing_a_reconstructor_is_routed(self) -> None:
        assert "reconstructor" in SUPPLIED_FIELDS
        assert context_metric_names(_computer(["fmrs"])) == ("fmrs",)

    def test_the_held_out_line_metric_is_routed(self) -> None:
        assert "mask" in SUPPLIED_FIELDS
        assert context_metric_names(_computer(["unacquired_bin_nrmse"])) == (
            "unacquired_bin_nrmse",
        )

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


class TestModelRerunMetrics:
    """monotone_remap_invariance gets the validation forward and the model input."""

    def _run(self, rank_transform: bool, input_type: str = "image") -> dict[str, float]:
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )
        from spectramr.models.encoders.mcgi_encoder import MCGIEncoder

        name = "monotone_remap_invariance"
        computer = ValidationMetricsComputer(ValidationMetricsConfig.from_dict({"metrics": [name]}))
        assert context_metric_names(computer) == (name,)
        torch.manual_seed(0)
        model = MCGIEncoder(hidden_channels=4, depth=2, rank_transform=rank_transform).eval()
        x = torch.rand(2, 1, 12, 12)
        with torch.no_grad():
            pred = model(x)
        cfg = types.SimpleNamespace(input_type=input_type, output_type="image")
        return compute_context_metrics(
            computer, (name,), pred, torch.rand_like(pred), x, {}, cfg, model,
            signal_domain="image",
        )

    def test_rank_on_reads_zero_rank_off_does_not(self) -> None:
        assert float(self._run(True)["val_monotone_remap_invariance"]) == 0.0
        assert float(self._run(False)["val_monotone_remap_invariance"]) > 1e-3

    def test_a_non_image_input_is_refused(self) -> None:
        with pytest.raises(ValueError, match="input_type"):
            self._run(True, input_type="latent")


class TestSamplingMask:
    """The loader's mask reaches ``unacquired_bin_nrmse`` (the held-out-line error)."""

    def _call(self, batch):
        from spectramr.core.metrics.computer import ValidationMetricsComputer
        from spectramr.core.metrics.types import ValidationMetricsConfig
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        names = ("unacquired_bin_nrmse", "acquired_bin_nrmse")
        computer = ValidationMetricsComputer(ValidationMetricsConfig.from_dict({"metrics": list(names)}))
        target = as_interleaved(_complex(2, 1, 8, 6, seed=1))
        model = types.SimpleNamespace(input_type="kspace", output_type="kspace")
        return target, lambda pred: compute_context_metrics(
            computer, names, pred, target, target, batch, model, lambda x: x,
            signal_domain="kspace",
        )

    def test_error_lands_only_on_the_withheld_lines(self) -> None:
        mask = torch.zeros(2, 1, 8, 6, 1)  # (B, 1, H, W, D) as the loader collates it
        mask[:, :, ::2] = 1.0
        target, score = self._call({"mask": mask})
        pred = target.clone()
        pred[:, :, 1::2] += 1.0  # wrong only on the rows the mask withheld
        out = score(pred)
        assert out["val_acquired_bin_nrmse"] == 0.0
        assert out["val_unacquired_bin_nrmse"] > 0.0

    def test_a_training_batch_carries_its_mask_as_a_field(self) -> None:
        mask = torch.ones(2, 1, 8, 6)
        batch = BatchAdapter.from_dict({"input": mask, "target": mask, "mask": mask})
        assert batch_tensor(batch, "mask") is batch.mask

    def test_a_batch_without_a_mask_raises(self) -> None:
        target, score = self._call({})
        with pytest.raises(ValueError, match="carries no `mask`"):
            score(target)


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


class TestScoringMapsAreTheBatchMaps:
    """The SENSE, Rep-BV and certificate metrics score with the batch's own coil maps."""

    def _batch(self):
        training_maps = _complex(2, 4, 16, 16, seed=1)
        return {"sensitivity_complex": training_maps}, training_maps

    def _context(self, batch, prediction, target, names) -> dict[str, float]:
        from spectramr.core.metrics.computer import create_validation_metrics_computer
        from spectramr.infrastructure.training.strategies.mixins.validation_context import (
            compute_context_metrics,
        )

        computer = create_validation_metrics_computer(list(names), device="cpu", data_range=1.0)
        model = types.SimpleNamespace(input_type="kspace", output_type="kspace")
        return compute_context_metrics(
            computer, tuple(names), prediction, target, prediction, batch, model,
            lambda x: x, signal_domain="kspace",
        )  # fmt: skip

    def test_sense_psnr_scores_with_the_batch_maps_and_ignores_a_retired_reference_key(self) -> None:
        """The retired reference store (all repetitions, #2510) must never reach a metric."""
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, training_maps = self._batch()
        batch["reference_sensitivity_complex"] = _complex(2, 4, 16, 16, seed=2)
        prediction = as_interleaved(_complex(2, 4, 16, 16, seed=3))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        got = self._context(batch, prediction, target, ["sense_psnr"])["val_sense_psnr"]
        metric = MetricsRegistry.get("sense_psnr")
        ctx = MetricContext(coil_maps=training_maps, signal_domain="kspace")
        assert got == pytest.approx(float(metric(prediction, target, context=ctx, data_range=1.0)), rel=1e-5)

    def test_the_corrected_mse_noise_term_reads_the_batch_maps(self) -> None:
        from spectramr.core.metrics.context import MetricContext
        from spectramr.core.metrics.registry import MetricsRegistry

        batch, training_maps = self._batch()
        prediction = as_interleaved(_complex(2, 4, 16, 16, seed=3))
        target = as_interleaved(_complex(2, 4, 16, 16, seed=4))
        got = self._context(batch, prediction, target, ["reference_noise_corrected_mse"])
        metric = MetricsRegistry.get("reference_noise_corrected_mse")
        ctx = MetricContext(coil_maps=training_maps, signal_domain="kspace")
        assert got["val_reference_noise_corrected_mse"] == pytest.approx(
            float(metric(prediction, target, context=ctx)), rel=1e-5
        )
