"""Tests for the regulatory-certification schema (R1/R2/R4/R5 + B-1.7 Dice-risk)."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.certification import (
    CertificationConfigSchema,
    ConformalCertificateConfig,
    DiceRiskCertificateConfig,
    PACBayesCertificateConfig,
)
from spectramr.infrastructure.calibration.pac_bayes_kl import KL_ESTIMATOR_REGISTRY, RISK_LOSSES


def test_dice_risk_defaults() -> None:
    cfg = DiceRiskCertificateConfig()
    assert cfg.enabled is False
    assert cfg.segmenter_backend == "label_dice"


def test_dice_risk_accepts_advertised_backends() -> None:
    for backend in ("label_dice", "synthseg"):
        assert DiceRiskCertificateConfig(segmenter_backend=backend).segmenter_backend == backend


def test_dice_risk_rejects_unknown_backend_at_load() -> None:
    # Regression (B-1.7 / pitfall #9): the advertised segmenter set must be rejected at
    # config-load time, not only when get_segmenter() runs at calibration time. A free-form
    # str previously let an illegal backend pass the Tier-0/1 audit and fail only at runtime.
    with pytest.raises(ValidationError):
        DiceRiskCertificateConfig(segmenter_backend="nnunet")


# ── #3041: certificates whose inputs no source produces are refused at load ────


def test_enabled_pathology_recall_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match="missing lesion detector"):
        CertificationConfigSchema(pathology_recall={"enabled": True})


def test_enabled_pac_bayes_requires_both_checkpoints() -> None:
    with pytest.raises(ValidationError, match="prior_checkpoint_path"):
        CertificationConfigSchema(pac_bayes={"enabled": True, "checkpoint_path": "q.pt"})
    cfg = CertificationConfigSchema(
        pac_bayes={"enabled": True, "checkpoint_path": "q.pt", "prior_checkpoint_path": "p.pt"}
    )
    assert cfg.pac_bayes.kl_estimator == "isotropic_gaussian"


def test_pac_bayes_rejects_unadvertised_options_at_load() -> None:
    for bad in ({"kl_estimator": "laplace"}, {"risk_loss": "mse"}, {"loss_out_of_range": "clip"}):
        with pytest.raises(ValidationError):
            PACBayesCertificateConfig(**bad)


def test_pac_bayes_advertised_sets_match_the_registries() -> None:
    fields = PACBayesCertificateConfig.model_fields
    assert set(get_args(fields["kl_estimator"].annotation)) == set(KL_ESTIMATOR_REGISTRY)
    assert set(get_args(fields["risk_loss"].annotation)) == set(RISK_LOSSES)


def test_foreground_threshold_defaults_and_accepts_none() -> None:
    assert ConformalCertificateConfig().foreground_threshold is None
    assert ConformalCertificateConfig(foreground_threshold=0.05).foreground_threshold == 0.05
    assert ConformalCertificateConfig(foreground_threshold=None).foreground_threshold is None
    with pytest.raises(ValidationError):
        ConformalCertificateConfig(foreground_threshold=1.5)
