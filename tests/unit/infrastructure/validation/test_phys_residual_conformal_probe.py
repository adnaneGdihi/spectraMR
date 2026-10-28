r"""Unit tests for the PR-CC Tier-2 conformal probe.

Targets ``spectramr.infrastructure.validation.phys_residual_conformal_probe``.

The probe is the locally-runnable, scientifically-rigorous proof that PR-CC fires:
on a phantom with KNOWN (PD, T1, T2) it (a) certifies the conformal coverage
guarantee and that an injected hallucination is flagged, and (b) checks the
configured model actually outputs (rho, T1, T2) parameter maps. It never raises
(returns a failed ProbeResult instead) so the audit CLI can map it to an exit code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _config(model_type: str = "bloch_field_bottleneck"):
    from spectramr.config.schemas.training.phys_residual_conformal import (
        PhysResidualConformalConfig,
    )

    model = SimpleNamespace(
        model_type=model_type,
        model_kwargs={},
        in_channels=1,
        out_channels=1,
        input_type="image",
        spatial_dims=2,
    )
    training = SimpleNamespace(
        phys_residual_conformal=PhysResidualConformalConfig(
            tr_ms=3000.0,
            te_ms=90.0,
            ti_ms=0.0,
            contrast_type="spin_echo",
            stratify_by="none",
        )
    )
    return SimpleNamespace(
        model=model, training=training, physics=SimpleNamespace(field_strength=0.3)
    )


def test_probe_passes_on_param_map_model() -> None:
    from spectramr.infrastructure.validation.phys_residual_conformal_probe import (
        phys_residual_conformal_probe,
    )

    res = phys_residual_conformal_probe(_config(), device="cpu")
    assert res.passed, res.message
    assert res.category == "phys_residual_conformal_coverage"
    assert res.severity == "info"


def test_probe_fails_when_model_lacks_param_output() -> None:
    from spectramr.infrastructure.validation.phys_residual_conformal_probe import (
        phys_residual_conformal_probe,
    )

    # configurable_unet (1->1) renders an image, has no predict_parameters and no
    # 3-channel parameter output: the contract check must fail with severity error.
    res = phys_residual_conformal_probe(_config(model_type="configurable_unet"), device="cpu")
    assert not res.passed
    assert res.severity == "error"


def test_probe_never_raises_on_bad_config() -> None:
    from spectramr.infrastructure.validation.phys_residual_conformal_probe import (
        phys_residual_conformal_probe,
    )

    bad = SimpleNamespace(model=None, training=None)
    res = phys_residual_conformal_probe(bad, device="cpu")
    assert not res.passed
    assert res.category == "phys_residual_conformal_coverage"
