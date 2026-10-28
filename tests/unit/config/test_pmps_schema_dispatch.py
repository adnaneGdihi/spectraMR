"""PMPS schema-dispatch + Phase-0 schema-validator tests."""

from __future__ import annotations

import pytest

from spectramr.config.schemas.data import (
    AcquisitionParamsSchema,
    MultiContrastConfigSchema,
    ReferenceTissuePanelConfig,
)
from spectramr.config.schemas.training import (
    TrainingConfigCorruptionCalibration,
    TrainingConfigDataEfficiencyHarness,
    TrainingConfigPairedSynthesis,
    TrainingConfigTissueDiffusionPretrain,
    _MODE_DISPATCH,
    create_training_config,
)


class TestMultiContrastBlochGroundedValidator:
    def test_bloch_grounded_without_panel_rejected(self) -> None:
        with pytest.raises(ValueError, match="reference_panel"):
            MultiContrastConfigSchema(
                enabled=True,
                bloch_grounded=True,
                acquisition_params=[
                    AcquisitionParamsSchema(name="T1w", TE=10.0, TR=500.0, FA=90.0, B0=3.0)
                ],
            )

    def test_bloch_grounded_without_acquisition_params_rejected(self) -> None:
        with pytest.raises(ValueError, match="acquisition_params"):
            MultiContrastConfigSchema(
                enabled=True,
                bloch_grounded=True,
                reference_panel=ReferenceTissuePanelConfig(tissues=["gm", "wm"]),
            )

    def test_bloch_grounded_with_both_accepted(self) -> None:
        mc = MultiContrastConfigSchema(
            enabled=True,
            bloch_grounded=True,
            reference_panel=ReferenceTissuePanelConfig(tissues=["gm", "wm", "csf"]),
            acquisition_params=[
                AcquisitionParamsSchema(name="T1w", TE=10.0, TR=500.0, FA=90.0, B0=3.0),
            ],
        )
        assert mc.bloch_grounded is True
        assert mc.reference_panel is not None
        assert len(mc.reference_panel.tissues) == 3


class TestAcquisitionConcomitantAutoRule:
    def test_auto_on_below_05t(self) -> None:
        a = AcquisitionParamsSchema(name="ulf", TE=10.0, TR=500.0, FA=90.0, B0=0.064)
        assert a.include_concomitant is True

    def test_auto_off_above_05t(self) -> None:
        a = AcquisitionParamsSchema(name="hf", TE=10.0, TR=500.0, FA=90.0, B0=3.0)
        assert a.include_concomitant is None

    def test_explicit_false_at_ulf_respected(self) -> None:
        # Explicit user override wins. Audit's
        # `concomitant_required_at_ulf` is what catches this at audit time.
        a = AcquisitionParamsSchema(
            name="ulf",
            TE=10.0,
            TR=500.0,
            FA=90.0,
            B0=0.064,
            include_concomitant=False,
        )
        assert a.include_concomitant is False


class TestSchemaDispatch:
    @pytest.mark.parametrize(
        "mode,cls",
        [
            ("tissue_diffusion_pretrain", TrainingConfigTissueDiffusionPretrain),
            ("corruption_calibration", TrainingConfigCorruptionCalibration),
            ("paired_synthesis", TrainingConfigPairedSynthesis),
            ("data_efficiency_harness", TrainingConfigDataEfficiencyHarness),
        ],
    )
    def test_mode_dispatch_present(self, mode: str, cls: type) -> None:
        assert mode in _MODE_DISPATCH, f"mode {mode} not in _MODE_DISPATCH."
        assert _MODE_DISPATCH[mode] == cls.__name__

    def test_construct_tissue_diffusion_pretrain(self) -> None:
        # Direct construction — `training_mode` is deprecated on the
        # schema (callers use `strategy_class`); the _MODE_DISPATCH
        # routing still exists for tools that pre-pop training_mode out
        # of the dict.
        cfg = TrainingConfigTissueDiffusionPretrain(
            timesteps=500,
            noise_schedule="cosine",
            prediction_type="v",
        )
        assert cfg.timesteps == 500
        assert cfg.field_strength_conditioning is True

    def test_construct_data_efficiency_harness_rejects_empty_settings(self) -> None:
        with pytest.raises(Exception, match="settings"):
            TrainingConfigDataEfficiencyHarness(
                real_paired_manifest="/tmp/real.json",
                synthetic_paired_manifest="/tmp/synth.json",
                settings=[],
            )
