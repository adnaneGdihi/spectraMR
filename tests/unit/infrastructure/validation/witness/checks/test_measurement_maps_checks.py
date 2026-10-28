"""Planted violations for ``measurement_maps_need_coil_maps``."""

from __future__ import annotations

from spectramr.infrastructure.validation.witness.checks import measurement_maps_checks as mmc
from spectramr.infrastructure.validation.witness.subject import WitnessSubject


def _verdict(mode: object, model_type: str | None):
    raw: dict = {"model": {"model_type": model_type}}
    if mode is not None:
        raw["validation"] = {"scoring": {"measurement_maps": mode}}
    return mmc.measurement_maps_need_coil_maps(WitnessSubject.for_ci("arm.yaml", raw))


def test_planted_the_family_on_a_model_without_coil_maps_is_an_error() -> None:
    verdict = _verdict("metrics", "diffusion_unet")
    assert verdict.passed is False and "takes none" in verdict.message
    assert verdict.yaml_keys == (mmc.KNOB, "model.model_type")


def test_planted_an_unknown_mode_is_an_error_even_before_the_schema() -> None:
    verdict = _verdict("full", "kspace_cold_diffusion")
    assert verdict.passed is False and "expected one of" in verdict.message


def test_the_family_on_a_coil_map_model_passes() -> None:
    assert _verdict("metrics", "kspace_cold_diffusion").passed is True


def test_off_or_absent_passes_on_any_model() -> None:
    assert _verdict("off", "diffusion_unet").passed is True
    assert _verdict(None, "diffusion_unet").passed is True
