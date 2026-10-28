"""Unit tests for the ``bloch_manifold_dps`` Pydantic schema (Idea 2 / B-2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training.bloch_manifold_dps import (
    BlochManifoldDPSConfig,
    TrainingConfigBlochManifoldDPS,
)


def _valid_block(**overrides) -> dict:
    block = {
        "pulse_sequence": "spgr",
        "param_bounds_T1_ms": (100.0, 5000.0),
        "param_bounds_T2_ms": (10.0, 2000.0),
        "param_bounds_M0": (0.0, 2.0),
        "param_bounds_B1": (0.7, 1.3),
        "projection_inner_iterations": 10,
        "projection_gn_damping": 1e-4,
        "projection_every": 1,
        "likelihood_weight_zeta": 1.0,
        "sampling_steps": 1000,
        "sampler_type": "ddim",
        "prediction_type": "epsilon",
    }
    block.update(overrides)
    return block


def test_valid_block_parses():
    cfg = BlochManifoldDPSConfig.model_validate(_valid_block())
    assert cfg.pulse_sequence == "spgr"
    assert cfg.sampler_type == "ddim"
    assert cfg.likelihood_weight_zeta == 1.0


def test_frozen_and_extra_forbid():
    cfg = BlochManifoldDPSConfig.model_validate(_valid_block())
    assert cfg.model_config["frozen"] is True
    assert cfg.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        BlochManifoldDPSConfig.model_validate(_valid_block(unknown_field=1))


def test_defaults_apply():
    cfg = BlochManifoldDPSConfig.model_validate({})
    assert cfg.pulse_sequence == "spgr"
    assert cfg.projection_inner_iterations == 10
    assert cfg.sampling_steps == 1000


def test_inverted_t1_bounds_rejected():
    with pytest.raises(ValidationError):
        BlochManifoldDPSConfig.model_validate(
            _valid_block(param_bounds_T1_ms=(5000.0, 100.0))
        )


def test_negative_bound_rejected():
    with pytest.raises(ValidationError):
        BlochManifoldDPSConfig.model_validate(
            _valid_block(param_bounds_T2_ms=(-5.0, 200.0))
        )


def test_empty_manifold_rejected():
    # T2 lower bound above T1 upper bound -> {T2 <= T1} empty.
    with pytest.raises(ValidationError):
        BlochManifoldDPSConfig.model_validate(
            _valid_block(
                param_bounds_T1_ms=(100.0, 200.0),
                param_bounds_T2_ms=(300.0, 400.0),
            )
        )


def test_bad_pulse_sequence_rejected():
    with pytest.raises(ValidationError):
        BlochManifoldDPSConfig.model_validate(_valid_block(pulse_sequence="epi"))


def test_top_level_schema_extends_diffusion_and_mounts_block():
    from spectramr.config.schemas.training.diffusion import TrainingConfigDiffusion

    # The top-level config extends the diffusion schema (so it inherits
    # timesteps / noise_schedule / sampler / ...) and adds the required
    # bloch_manifold_dps block as a typed field.
    assert issubclass(TrainingConfigBlochManifoldDPS, TrainingConfigDiffusion)
    fields = TrainingConfigBlochManifoldDPS.model_fields
    assert "bloch_manifold_dps" in fields
    assert fields["bloch_manifold_dps"].annotation is BlochManifoldDPSConfig
    # extra='allow' mirrors the sibling twin-DPS schema (carries VF blocks).
    assert TrainingConfigBlochManifoldDPS.model_config["extra"] == "allow"


def test_full_yaml_loads_through_settings():
    """End-to-end: the shipped arm loads via the real TrainingSettings path."""
    from pathlib import Path

    from spectramr.config.settings import TrainingSettings

    yaml_path = Path("experiments/inprogress/vf/exp_p2_b2_bloch_manifold_dps.yaml")
    if not yaml_path.exists():
        pytest.skip("experiment YAML not present in this checkout")
    settings = TrainingSettings.from_yaml(str(yaml_path))
    # The block is reachable (typed once the lead mounts the union; a dict
    # off the extra='allow' base schema until then).
    bm = getattr(settings.training, "bloch_manifold_dps", None)
    assert bm is not None
    get = (lambda b, k: b[k]) if isinstance(bm, dict) else getattr
    assert get(bm, "pulse_sequence") == "spgr"
