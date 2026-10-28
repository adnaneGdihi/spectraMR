r"""Unit tests for the contrast/field-agnostic bundle training schemas.

Targets ``spectramr.config.schemas.training.acq_hypernetwork`` (M3 LCAH),
``spectramr.config.schemas.training.dispersion_bloch_ae`` (M4 DL-BAE) and
``spectramr.config.schemas.training.mcgi`` (M2 MCGI).

All three are ``frozen``/``extra="forbid"`` per the config non-negotiable, and
both wrappers must be reachable through the ``training_mode`` discriminated
union -- a schema that exists but is not dispatched is an unwired knob.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.training import (
    _MODE_DISPATCH,
    AcqHypernetworkConfig,
    DispersionBlochAEConfig,
    MCGIConfig,
    TrainingConfigAcqHypernetwork,
    TrainingConfigDispersionBlochAE,
)

pytestmark = pytest.mark.unit


class TestAcqHypernetworkConfig:
    def test_defaults_keep_the_certificate_valid(self) -> None:
        cfg = AcqHypernetworkConfig()
        assert cfg.spectral_norm is True
        assert cfg.acquisition_key == "acquisition"

    def test_frozen(self) -> None:
        cfg = AcqHypernetworkConfig()
        with pytest.raises(ValidationError):
            cfg.spectral_norm = False

    def test_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            AcqHypernetworkConfig(spectral_normalisation=True)

    def test_rejects_non_positive_lipschitz_target(self) -> None:
        with pytest.raises(ValidationError):
            AcqHypernetworkConfig(lipschitz_target=0.0)

    def test_mode_dispatch_reaches_the_wrapper(self) -> None:
        assert _MODE_DISPATCH["acq_hypernetwork"] == "TrainingConfigAcqHypernetwork"

    def test_wrapper_requires_the_sub_block(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfigAcqHypernetwork()


class TestDispersionBlochAEConfig:
    def test_accepts_an_identifiable_arm(self) -> None:
        cfg = DispersionBlochAEConfig(num_pools=2, fields_present=(0.055, 0.3, 1.5, 3.0, 7.0))
        assert cfg.num_pools == 2

    def test_rejects_duplicate_fields(self) -> None:
        """Repeated fields inflate the apparent M without adding rank."""
        with pytest.raises(ValidationError, match="DISTINCT"):
            DispersionBlochAEConfig(num_pools=1, fields_present=(1.5, 1.5, 3.0))

    def test_rejects_non_positive_field(self) -> None:
        with pytest.raises(ValidationError, match="positive field strengths"):
            DispersionBlochAEConfig(num_pools=1, fields_present=(0.0, 1.5, 3.0))

    def test_rejects_inverted_tau_bounds(self) -> None:
        with pytest.raises(ValidationError, match="0 < min < max"):
            DispersionBlochAEConfig(
                num_pools=1, fields_present=(0.3, 1.5, 3.0), tau_c_bounds=(1e-6, 1e-11)
            )

    def test_fields_present_is_required(self) -> None:
        """There is no sane default field set; the arm must declare it."""
        with pytest.raises(ValidationError):
            DispersionBlochAEConfig(num_pools=1)

    def test_frozen(self) -> None:
        cfg = DispersionBlochAEConfig(num_pools=1, fields_present=(0.3, 1.5, 3.0))
        with pytest.raises(ValidationError):
            cfg.num_pools = 2

    def test_mode_dispatch_reaches_the_wrapper(self) -> None:
        assert _MODE_DISPATCH["dispersion_bloch_ae"] == "TrainingConfigDispersionBlochAE"

    def test_wrapper_requires_the_sub_block(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfigDispersionBlochAE()


class TestMCGIConfig:
    def test_defaults_are_the_exact_invariance(self) -> None:
        cfg = MCGIConfig()
        assert cfg.hard_rank_eval is True
        assert cfg.symmetrize_order_reversal is True

    def test_rejects_non_positive_temperature(self) -> None:
        with pytest.raises(ValidationError):
            MCGIConfig(soft_rank_temperature=0.0)

    def test_frozen_and_strict(self) -> None:
        cfg = MCGIConfig()
        with pytest.raises(ValidationError):
            cfg.hard_rank_eval = False
        with pytest.raises(ValidationError):
            MCGIConfig(hard_rank=True)


def test_the_dlbae_arm_loads_its_block_typed() -> None:
    """#1711: the block reached the strategy as a dict, so getattr read every default."""
    from spectramr.config.settings import TrainingSettings

    cfg = TrainingSettings.from_yaml(
        "experiments/inprogress/dispersion_bloch_ae/dlbae_brain_5field.yaml"
    ).training.dispersion_bloch_ae
    assert isinstance(cfg, DispersionBlochAEConfig)
    assert cfg.num_pools == 2 and len(cfg.fields_present) == 5


def test_the_lcah_arm_loads_its_block_typed() -> None:
    """The block reached the strategy as a dict, so getattr read hyper_hidden=32, not 64."""
    from spectramr.config.settings import TrainingSettings

    cfg = TrainingSettings.from_yaml(
        "experiments/inprogress/acq_hypernetwork/lcah_recon_multifield.yaml"
    ).training.acq_hypernetwork
    assert isinstance(cfg, AcqHypernetworkConfig)
    assert cfg.hyper_hidden == 64
