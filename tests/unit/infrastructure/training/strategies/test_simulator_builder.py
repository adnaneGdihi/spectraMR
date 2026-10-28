"""Tests for build_simulator_from_config — composite/severity wiring.

The builder is the SSOT seam between ``config.physics.digital_twin`` and the
``DigitalTwinSimulator``. These pin that the 2026-05-26 motion_composite and
motion_severity fields are forwarded (and that the defaults still produce a
single-motion, nominal-amplitude simulator).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.schemas.data import DataConfigSchema  # noqa: E402
from spectramr.config.schemas.physics import DigitalTwinConfig  # noqa: E402
from spectramr.infrastructure.training.strategies.simulator_builder import (  # noqa: E402
    require_epi_profile,
    build_simulator_from_config,
    undersampling_mask_kwargs,
)


def _config(dt: DigitalTwinConfig, field_strength: float = 3.0) -> SimpleNamespace:
    """Stub settings carrying the two blocks the builder reads.

    ``data`` is the **real** ``DataConfigSchema`` rather than a hand-rolled
    namespace: the builder reads ``data.sampling.patch_size``, and a stub
    spelling it flat (the pre-decomposition ``data.patch_size``) is what left
    these tests red on ``dev`` — a stub cannot drift from the schema if it *is*
    the schema.
    """
    return SimpleNamespace(
        physics=SimpleNamespace(digital_twin=dt, field_strength=field_strength),
        data=DataConfigSchema(),
    )


def test_builder_forwards_composite_and_severity() -> None:
    dt = DigitalTwinConfig(
        enabled=True,
        motion_composite=["rigid", "periodic"],
        motion_severity=2.5,
    )
    sim = build_simulator_from_config(_config(dt), torch.device("cpu"))
    assert sim.motion_composite == ["rigid", "periodic"]
    assert sim._motion_severity == pytest.approx(2.5)


def test_builder_defaults_single_motion_constant_amplitude() -> None:
    sim = build_simulator_from_config(
        _config(DigitalTwinConfig(enabled=True)), torch.device("cpu")
    )
    assert sim.motion_composite == []
    assert sim._motion_severity == pytest.approx(1.0)


def test_builder_builds_the_acquisition_profile_at_physics_field_strength() -> None:
    dt = DigitalTwinConfig(
        enabled=True, acquisition={"readout_bw_hz_per_px": 200.0, "sequence": "gre"}
    )
    sim = build_simulator_from_config(_config(dt, field_strength=0.55), torch.device("cpu"))
    assert sim.scanner_profile.b0_T == pytest.approx(0.55)


@pytest.mark.parametrize("sequence", ["spin_echo", "gre"])
def test_vf_field_score_rejects_a_non_epi_profile(sequence) -> None:
    dt = DigitalTwinConfig(
        enabled=True, acquisition={"readout_bw_hz_per_px": 200.0, "sequence": sequence}
    )
    sim = build_simulator_from_config(_config(dt), torch.device("cpu"))
    with pytest.raises(ValueError, match="EPI"):
        require_epi_profile(sim)


def test_vf_field_score_accepts_epi_or_no_profile() -> None:
    epi = DigitalTwinConfig(
        enabled=True, acquisition={"readout_bw_hz_per_px": 200.0, "sequence": "epi"}
    )
    require_epi_profile(build_simulator_from_config(_config(epi), torch.device("cpu")))
    require_epi_profile(
        build_simulator_from_config(_config(DigitalTwinConfig(enabled=True)), torch.device("cpu"))
    )


# ──────────────────────────────────────────────────────────────────────
# digital_twin.enabled must gate the build (F7 — the flag was inert)
# ──────────────────────────────────────────────────────────────────────


def test_builder_raises_when_twin_disabled() -> None:
    """``enabled=False`` must stop the build, not be silently ignored.

    Regression for F7: the four VF strategies (virtual_fiducial, vf_admm,
    pma_varnet, distillation) call this builder unconditionally from
    ``_setup_strategy_specific_components``, so before this gate an arm that
    left ``physics.digital_twin`` undeclared — ``enabled`` defaults to
    ``False`` — still got a fully-built, all-default twin. The knob parsed and
    validated but gated nothing (pitfall #15), which for a VF arm makes the
    headline mechanism unrequested rather than absent (pitfall #16).
    """
    with pytest.raises(ValueError, match="physics.digital_twin.enabled"):
        build_simulator_from_config(
            _config(DigitalTwinConfig(enabled=False)), torch.device("cpu")
        )


def test_builder_disabled_is_the_schema_default() -> None:
    """The gate fires on an *undeclared* block, which is the real-world case.

    An arm that omits ``physics.digital_twin`` entirely gets the schema
    default, so this pins that the default is the refused state rather than
    relying on an explicit ``enabled: false`` that no arm writes.
    """
    assert DigitalTwinConfig().enabled is False
    with pytest.raises(ValueError):
        build_simulator_from_config(_config(DigitalTwinConfig()), torch.device("cpu"))


def test_enabled_toggle_changes_outcome() -> None:
    """The flag must be load-bearing in both directions."""
    cfg_on = _config(DigitalTwinConfig(enabled=True))
    assert build_simulator_from_config(cfg_on, torch.device("cpu")) is not None
    with pytest.raises(ValueError):
        build_simulator_from_config(
            _config(DigitalTwinConfig(enabled=False)), torch.device("cpu")
        )


# ──────────────────────────────────────────────────────────────────────
# undersampling_mask_kwargs — feed DC layers the measured k-space mask
# ──────────────────────────────────────────────────────────────────────


class _Sim:
    def __init__(self, mask: object) -> None:
        self.last_undersampling_mask = mask


def test_kwargs_carry_mask_when_present() -> None:
    mask = torch.ones(1, 1, 8, 1)
    out = undersampling_mask_kwargs(_Sim(mask))
    assert set(out) == {"mask"} and out["mask"] is mask


def test_kwargs_empty_when_mask_none() -> None:
    assert undersampling_mask_kwargs(_Sim(None)) == {}


def test_kwargs_empty_when_attr_missing() -> None:
    assert undersampling_mask_kwargs(object()) == {}


def test_mask_kwarg_activates_data_consistency() -> None:
    """The mask kwarg must activate a hard-DC layer's k-space replacement.

    ``_HardDCLayer`` returns its CNN input unchanged when ``mask`` is None, and
    replaces measured k-space lines when a mask is supplied. Driving it through
    ``undersampling_mask_kwargs`` proves the helper's output lights up data
    consistency rather than being silently ignored. (We test the DC primitive,
    not the full generator: the generator's zero-init output projection masks
    the cascade contribution until trained, so an untrained end-to-end forward
    cannot observe the effect.)
    """
    from spectramr.models.generators.vf_reconstruction_generators import _HardDCLayer

    torch.manual_seed(0)
    layer = _HardDCLayer()
    x_cnn = torch.randn(1, 2, 32, 32)
    x_input = torch.randn(1, 2, 32, 32)  # the "measurement" (≠ CNN estimate)
    mask = torch.zeros(1, 1, 32, 1)
    mask[:, :, 12:20, :] = 1.0  # sampled centre band

    out_plain = layer(x_cnn, x_input, **undersampling_mask_kwargs(_Sim(None)))
    out_masked = layer(x_cnn, x_input, **undersampling_mask_kwargs(_Sim(mask)))

    assert torch.allclose(out_plain, x_cnn), "no-mask path must be a DC no-op"
    assert not torch.allclose(out_plain, out_masked), "DC did not fire with the mask"
