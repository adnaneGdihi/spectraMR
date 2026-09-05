"""Integration test for VF campaign Phase 2 breakthrough-method wiring.

Covers the shared-registry edits that wire the four breakthrough methods
(B-1 equivariance-conformal, B-2 Bloch-manifold DPS, B-3 SE(3) navigator,
B-4 Hamiltonian acquisition) into the framework:

- ``STRATEGY_CLASS_PATHS`` (strategy factory) resolves each new training_mode
  to an importable class.
- ``_MODE_DISPATCH`` (config schema) mounts the typed schemas (B-4 loads via
  the base schema's ``extra="allow"``).
- the new metrics and losses are registered.
- each new experiment YAML loads via ``TrainingSettings.from_yaml``.
"""

import importlib

import pytest

from tests.utils.repo_scripts import require_repo_file

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.integration

_NEW_MODES = [
    "equivariance_conformal",
    "bloch_manifold_dps",
    "se3_equivariant_navigator",
    "hamiltonian_acquisition",
]

_NEW_YAMLS = {
    "equivariance_conformal": "experiments/inprogress/vf/exp_p1_b1_equivariance_conformal.yaml",
    "bloch_manifold_dps": "experiments/inprogress/vf/exp_p2_b2_bloch_manifold_dps.yaml",
    "se3_equivariant_navigator": "experiments/inprogress/vf/exp_p3_b3_se3_navigator.yaml",
    "hamiltonian_acquisition": "experiments/inprogress/vf/exp_p7_b4_hamiltonian_acquisition.yaml",
}


@pytest.mark.parametrize("mode", _NEW_MODES)
def test_strategy_resolves(mode):
    """Each new training_mode maps to an importable strategy class."""
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[mode]
    module_name, cls_name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    assert isinstance(cls, type)


def test_schema_dispatch_mounts():
    """B-1..B-3 have typed schema mounts; B-4 loads via the base (extra=allow)."""
    from spectramr.config.schemas.training import _MODE_DISPATCH

    for mode in ("equivariance_conformal", "bloch_manifold_dps", "se3_equivariant_navigator"):
        assert mode in _MODE_DISPATCH


@pytest.mark.parametrize("metric", ["equivariance_defect", "bloch_manifold_residual"])
def test_metrics_registered(metric):
    from spectramr.core.metrics.registry import get_metric

    assert get_metric(metric).name == metric


@pytest.mark.parametrize(
    "loss",
    ["se3_equivariance_defect", "hamiltonian_energy_conservation", "gradient_hardware_compliance"],
)
def test_losses_registered(loss):
    import spectramr.models.losses  # noqa: F401 — fires @register_loss
    from spectramr.models.losses.registry import LossRegistry

    assert loss in LossRegistry._custom_losses


@pytest.mark.parametrize("mode", _NEW_MODES)
def test_yaml_loads(mode):
    """Each new arm's YAML loads through the frozen settings model."""
    from spectramr.config.settings import TrainingSettings

    settings = TrainingSettings.from_yaml(str(require_repo_file(_NEW_YAMLS[mode])))
    assert settings.training.training_mode == mode
