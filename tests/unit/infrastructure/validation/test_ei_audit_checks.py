"""Unit tests for the Equivariant Imaging audit checks (Phase A step 3).

``check_ei_sensing_margin`` is the static identifiability gate (EI needs
undersampling so the operator breaks the group symmetry — Tachella sensing
theorems). ``check_ei_robust_key_matches_correction`` refuses the contradiction
of selecting the ``robust_ei`` key with the nc-χ correction turned off
(pitfall #16: a registry key that silently does nothing).
"""

from __future__ import annotations

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker


class _CS:
    def __init__(self, acceleration_factor=4.0):
        self.acceleration_factor = acceleration_factor


class _Physics:
    def __init__(self, acceleration_factor=4.0):
        self.compressed_sensing = _CS(acceleration_factor=acceleration_factor)


class _Accel:
    def __init__(self, base_acceleration=4.0):
        self.base_acceleration = base_acceleration


class _EI:
    def __init__(self, robust_correction=False):
        self.robust_correction = robust_correction


class _Training:
    def __init__(self, training_mode="equivariant_imaging", ei_block=True, robust_correction=False):
        self.training_mode = training_mode
        self.strategy_class = (
            "spectramr.infrastructure.training.strategies."
            "equivariant_imaging_strategy.EquivariantImagingStrategy"
        )
        self.equivariant_imaging = (
            _EI(robust_correction=robust_correction) if ei_block else None
        )


class _Config:
    def __init__(
        self,
        training_mode="equivariant_imaging",
        ei_block=True,
        robust_correction=False,
        acceleration_factor=4.0,
    ):
        self.training = _Training(
            training_mode=training_mode,
            ei_block=ei_block,
            robust_correction=robust_correction,
        )
        self.physics = _Physics(acceleration_factor=acceleration_factor)
        self.acceleration = _Accel(base_acceleration=acceleration_factor)


def _checker():
    return ConfigHealthChecker()


# --- check_ei_sensing_margin -------------------------------------------------
def test_sensing_margin_not_applicable_for_non_ei_arm():
    cfg = _Config(training_mode="reconstruction", ei_block=False)
    r = _checker().check_ei_sensing_margin(cfg)
    assert r.passed and r.severity == "info"


def test_sensing_margin_passes_when_undersampled():
    r = _checker().check_ei_sensing_margin(_Config(acceleration_factor=4.0))
    assert r.passed


def test_sensing_margin_errors_when_fully_sampled():
    r = _checker().check_ei_sensing_margin(_Config(acceleration_factor=1.0))
    assert not r.passed and r.severity == "error"


def _m4raw_route(**mask_kwargs) -> _Config:
    """An EI arm on the M4Raw route: the schema's acceleration defaults stay at 4."""
    from types import SimpleNamespace

    from spectramr.config.schemas.data import TransformSpecSchema

    cfg = _Config(acceleration_factor=4.0)
    cfg.undersampling = _Accel(base_acceleration=4.0)
    spec = TransformSpecSchema(name="complementary_rep_masking", **mask_kwargs)
    cfg.data = SimpleNamespace(processing=SimpleNamespace(transforms=[spec]))
    return cfg


def test_sensing_margin_reads_the_complementary_mask_rate():
    r = _checker().check_ei_sensing_margin(_m4raw_route(kwargs={"acceleration": 4.0}))
    assert r.passed and "4.0" in r.message


def test_sensing_margin_fails_a_planted_fully_sampled_complementary_mask():
    """Planted: R=1 beside the schema's 4.0 defaults used to pass vacuously."""
    r = _checker().check_ei_sensing_margin(_m4raw_route(kwargs={"acceleration": 1.0}))
    assert not r.passed and r.severity == "error"
    assert r.yaml_keys == ["data.processing.transforms[complementary_rep_masking].acceleration"]


def test_sensing_margin_reads_a_flat_kwarg_and_the_transform_default():
    assert not _checker().check_ei_sensing_margin(_m4raw_route(acceleration=1.0)).passed
    defaulted = _checker().check_ei_sensing_margin(_m4raw_route())
    assert defaulted.passed and "2.0" in defaulted.message  # ComplementaryRepMasking's own default


# --- check_ei_robust_key_matches_correction ----------------------------------
def test_robust_key_not_applicable_when_mode_is_plain_ei():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="equivariant_imaging")
    )
    assert r.passed and r.severity == "info"


def test_robust_key_passes_when_correction_on():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="robust_ei", robust_correction=True)
    )
    assert r.passed


def test_robust_key_errors_when_correction_off():
    r = _checker().check_ei_robust_key_matches_correction(
        _Config(training_mode="robust_ei", robust_correction=False)
    )
    assert not r.passed and r.severity == "error"


# --- wiring ------------------------------------------------------------------
def test_ei_checks_wired_into_run_all_checks():
    import inspect

    src = inspect.getsource(ConfigHealthChecker.run_all_checks)
    assert "check_ei_sensing_margin" in src
    assert "check_ei_robust_key_matches_correction" in src
