"""A context metric configured on a strategy that never supplies its context is refused (#2703).

Each case plants one strategy x metric pair and asserts the verdict; the routed twins prove
the check does not refuse everything.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.validation.context_metric_routing import (
    configured_metric_names,
    unrouted_context_metrics,
)


def test_a_sense_metric_on_the_gan_strategy_is_refused() -> None:
    """Planted: gan's own validation_step never builds the coil-map context."""
    assert unrouted_context_metrics(GANTrainingStrategy, ["psnr", "sense_psnr"]) == ["sense_psnr"]


def test_the_shared_validation_path_routes_the_sense_metrics() -> None:
    assert (
        unrouted_context_metrics(ReconstructionTrainingStrategy, ["sense_psnr", "sense_ssim"]) == []
    )


def test_the_shared_path_still_refuses_a_metric_whose_needs_it_cannot_fill() -> None:
    """``ndcr`` needs the sampling mask, which ``SUPPLIED_FIELDS`` does not carry."""
    assert unrouted_context_metrics(ReconstructionTrainingStrategy, ["ndcr"]) == ["ndcr"]


def test_diffusion_routes_its_trust_functionals_and_nothing_else() -> None:
    assert unrouted_context_metrics(DiffusionTrainingStrategy, ["ndcr", "nse_hall"]) == []
    assert unrouted_context_metrics(DiffusionTrainingStrategy, ["sense_psnr"]) == ["sense_psnr"]


def test_plain_metrics_are_never_flagged() -> None:
    assert unrouted_context_metrics(GANTrainingStrategy, ["psnr", "ssim", "hfen"]) == []


def test_a_class_with_no_routing_declaration_routes_nothing() -> None:
    class _Bare:
        pass

    assert unrouted_context_metrics(_Bare, ["sense_psnr"]) == ["sense_psnr"]


def test_the_names_come_from_the_scoring_block() -> None:
    config = SimpleNamespace(
        validation=SimpleNamespace(
            scoring=SimpleNamespace(compute=["psnr", SimpleNamespace(name="sense_psnr")])
        )
    )
    assert configured_metric_names(config) == ["psnr", "sense_psnr"]
    assert configured_metric_names(SimpleNamespace(validation=None)) == []


def _checker():
    from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

    return ConfigHealthChecker.__new__(ConfigHealthChecker)


def _config(mode: str, metrics: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(strategy_class=None, training_mode=mode),
        validation=SimpleNamespace(scoring=SimpleNamespace(compute=metrics)),
    )


def test_the_audit_refuses_a_sense_metric_on_a_gan_arm() -> None:
    result = _checker().check_context_metrics_routed(_config("gan", ["psnr", "sense_psnr"]))
    assert not result.passed and result.severity == "error"
    assert "GANTrainingStrategy" in result.message and "sense_psnr" in result.message


def test_the_audit_passes_the_same_metric_on_a_reconstruction_arm() -> None:
    result = _checker().check_context_metrics_routed(_config("reconstruction", ["sense_psnr"]))
    assert result.passed


def test_sense_magnitude_on_the_gan_strategy_is_refused() -> None:
    """Planted: gan's validation_step calls the metric transform without the batch."""
    from spectramr.infrastructure.validation.context_metric_routing import (
        unrouted_output_transform,
    )

    assert unrouted_output_transform(GANTrainingStrategy, "sense_magnitude") == "sense_magnitude"
    assert (
        unrouted_output_transform(DiffusionTrainingStrategy, "sense_magnitude") == "sense_magnitude"
    )


def test_sense_magnitude_on_the_shared_validation_path_is_routed() -> None:
    from spectramr.infrastructure.validation.context_metric_routing import (
        unrouted_output_transform,
    )

    assert unrouted_output_transform(ReconstructionTrainingStrategy, "sense_magnitude") is None


@pytest.mark.parametrize("transform", [None, "magnitude", "ifft_magnitude", "ifft_sense_adjoint"])
def test_transforms_that_need_no_batch_maps_are_never_flagged(transform) -> None:
    from spectramr.infrastructure.validation.context_metric_routing import (
        unrouted_output_transform,
    )

    assert unrouted_output_transform(GANTrainingStrategy, transform) is None


def test_the_output_transform_is_read_from_the_scoring_block() -> None:
    from spectramr.infrastructure.validation.context_metric_routing import (
        configured_output_transform,
    )

    config = SimpleNamespace(
        validation=SimpleNamespace(scoring=SimpleNamespace(output_transform="sense_magnitude"))
    )
    assert configured_output_transform(config) == "sense_magnitude"
    assert configured_output_transform(SimpleNamespace(validation=None)) is None


def test_the_audit_refuses_the_transform_on_a_gan_arm() -> None:
    config = _config("gan", ["psnr"])
    config.validation.scoring.output_transform = "sense_magnitude"
    result = _checker().check_context_metrics_routed(config)
    assert not result.passed and "sense_magnitude" in result.message
    assert "validation.scoring.output_transform" in result.yaml_keys


def test_the_audit_is_registered_in_the_run() -> None:
    import inspect

    from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

    assert "self.check_context_metrics_routed(config)" in inspect.getsource(ConfigHealthChecker)
