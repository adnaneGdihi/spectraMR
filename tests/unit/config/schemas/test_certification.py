"""Tests for the regulatory-certification schema (R1/R2/R4/R5 + B-1.7 Dice-risk)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.certification import (
    CertificationConfigSchema,
    ConformalCertificateConfig,
    DiceRiskCertificateConfig,
)


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


def test_enabled_pac_bayes_is_refused_at_load() -> None:
    with pytest.raises(ValidationError, match=r"missing KL\(Q\|\|P\) estimator"):
        CertificationConfigSchema(pac_bayes={"enabled": True})


def test_foreground_threshold_defaults_and_accepts_none() -> None:
    assert ConformalCertificateConfig().foreground_threshold is None
    assert ConformalCertificateConfig(foreground_threshold=0.05).foreground_threshold == 0.05
    assert ConformalCertificateConfig(foreground_threshold=None).foreground_threshold is None
    with pytest.raises(ValidationError):
        ConformalCertificateConfig(foreground_threshold=1.5)
