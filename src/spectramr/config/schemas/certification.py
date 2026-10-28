"""Regulatory certification configuration (R1 / R2 / R4 / R5 + V.3 badge).

Plan: TODO/backlog_ulf_radiologist_acceptance_roadmap.md §PR-20..26
      TODO/backlog_paradigm_expansion_roadmap.md §PR-3 (H1) / §PR-4 (H2)

Each certificate is opt-in (``enabled=False`` by default). The
``ValidationBadge`` aggregates the per-certificate JSON artefacts into
a single ``eligible`` flag consumed by the regulatory-package CLI.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConformalCertificateConfig(BaseModel):
    """R1 — Conformal Diagnostic Feature Coverage (CDFC)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    alpha: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Target marginal miscoverage rate (1 - alpha == coverage target).",
    )
    score_fn: str = Field(
        default="absolute_residual",
        description=(
            "Non-conformity score dispatched through the calibration-score registry: "
            "absolute_residual | pixel_residual | quantile_regression | cqr | vf_residual. "
            "vf_residual requires marker_basis_path (see below)."
        ),
    )
    marker_basis_path: str | None = Field(
        default=None,
        description=(
            "Path to a .pt file holding a complex tensor of shape (N, k) — "
            "the orthonormal marker basis used by the vf_residual score. "
            "REQUIRED when score_fn=='vf_residual'; ignored otherwise. "
            "The Tier-1 audit ``vf_residual_marker_matrix_well_conditioned`` "
            "verifies the basis exists and has cond(M^H M) < 1e6."
        ),
    )
    n_calibration: int = Field(
        default=500,
        ge=10,
        description=(
            "Calibration images drawn from the subject-split calibration half. The run "
            "raises when the cohort holds fewer; the realised count is stamped."
        ),
    )
    foreground_threshold: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description=(
            "Foreground mask: a target pixel scores only when |target| exceeds this fraction "
            "of its image's peak magnitude. None (default) scores every pixel, background "
            "included; the M4Raw certification arms declare 0.05."
        ),
    )
    output_artefact: str | None = Field(
        default=None,
        description="Path the strategy writes the CalibrationReport JSON to.",
    )

    @model_validator(mode="after")
    def _validate_vf_residual_basis(self) -> ConformalCertificateConfig:
        """vf_residual requires marker_basis_path."""
        if self.score_fn == "vf_residual" and self.marker_basis_path is None:
            raise ValueError(
                "certification.conformal.score_fn='vf_residual' requires "
                "marker_basis_path to point at a (N, k) complex tensor. "
                "Set certification.conformal.marker_basis_path."
            )
        return self


class CHDCertificateConfig(BaseModel):
    """R2 — Calibrated Hallucination Detection (CHD)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    beta: float = Field(default=0.05, gt=0.0, lt=1.0)
    delta: float = Field(default=0.001, gt=0.0, lt=1.0)
    n_calibration: int = Field(
        default=500,
        ge=10,
        description="Calibration images for the threshold fit (same subject split as R1).",
    )
    foreground_threshold: float | None = Field(
        default=None,
        ge=0.0,
        lt=1.0,
        description="Foreground mask on the |pred - target| scores; None (default) keeps every pixel.",
    )
    output_artefact: str | None = Field(default=None)


class PathologyRecallCertificateConfig(BaseModel):
    """R4 — Pathology Recall Certificate (PRC)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    classes: list[str] = Field(
        default_factory=lambda: ["infarct", "meningioma", "hemorrhage"],
    )
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    target_recall: float = Field(default=0.90, gt=0.0, le=1.0)
    output_artefact: str | None = Field(default=None)


class PACBayesCertificateConfig(BaseModel):
    """R5 — Cross-Site PAC-Bayes Generalisation Certificate."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    tv_tolerance_tau: float = Field(default=0.05, gt=0.0, le=1.0)
    loss_bound_M: float = Field(default=1.0, gt=0.0)
    output_artefact: str | None = Field(default=None)


class DiceRiskCertificateConfig(BaseModel):
    """B-1.7 — SynthSeg-Dice RCPS risk-control certificate (MRIxFields2026)."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    alpha: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Target expected Dice-risk budget (1 - Dice) to control.",
    )
    delta: float = Field(default=0.05, gt=0.0, lt=1.0)
    segmenter_backend: Literal["label_dice", "synthseg"] = Field(
        default="label_dice",
        description=(
            "Segmentation backend: 'label_dice' (local proxy) or 'synthseg'. The advertised "
            "set is enforced at config-load (pitfall #9: reject illegal YAML at audit time, not "
            "only when get_segmenter() runs)."
        ),
    )
    n_classes: int = Field(
        default=5,
        ge=2,
        description="Segmentation class count (label_dice proxy bins; SynthSeg=14).",
    )
    independence_unit: Literal["subject", "sample"] = Field(
        default="subject",
        description=(
            "What one Hoeffding draw is. 'subject' (default): the per-subject mean "
            "Dice-risk, n = number of subjects in the calibration cohort, read from "
            "batch['subject_id'] (the run raises when the batch carries none). "
            "'sample': one draw per scored image, the pre-2026-09-03 behaviour, which "
            "on co-registered slice-served volumes inflates n by the slice count and "
            "makes the bound optimistic (#1707); declare it only for a cohort whose "
            "samples are independent."
        ),
    )
    checkpoint_path: str | None = Field(
        default=None,
        description=(
            "Path to the trained synthesiser checkpoint to certify. REQUIRED when "
            "enabled: a Dice-risk certificate over an uninitialised/untrained model "
            "is scientifically meaningless (pitfall #20), so run_calibration raises "
            "if this is unset. Point it at the synthesis arm's "
            "<output_dir>/checkpoints/best.<format>."
        ),
    )
    output_artefact: str | None = Field(default=None)


class ValidationBadgeConfig(BaseModel):
    """V.3 — Validation badge bundling all upstream certificate artefacts."""

    model_config = {"extra": "ignore", "frozen": True}

    enabled: bool = Field(default=False)
    gates: list[str] = Field(
        default_factory=lambda: [
            "iqm",
            "downstream",
            "hallucination",
            "conformal_coverage",
            "pathology_recall",
            "pac_bayes",
        ],
        description="Required gates for ``eligible=true``.",
    )
    output_path: str | None = Field(default=None)
    upstream_certificate_artefacts: dict[str, str] = Field(
        default_factory=dict,
        description="Map of certificate-letter (r1_cdfc, r2_chd, ...) to JSON path.",
    )


class CertificationConfigSchema(BaseModel):
    """Top-level certification block (top-level under TrainingSettings)."""

    model_config = {"extra": "ignore", "frozen": True}

    conformal: ConformalCertificateConfig = Field(
        default_factory=ConformalCertificateConfig,
    )
    chd: CHDCertificateConfig = Field(default_factory=CHDCertificateConfig)
    pathology_recall: PathologyRecallCertificateConfig = Field(
        default_factory=PathologyRecallCertificateConfig,
    )
    pac_bayes: PACBayesCertificateConfig = Field(
        default_factory=PACBayesCertificateConfig,
    )
    dice_risk: DiceRiskCertificateConfig = Field(
        default_factory=DiceRiskCertificateConfig,
    )
    badge: ValidationBadgeConfig = Field(default_factory=ValidationBadgeConfig)

    @model_validator(mode="after")
    def _reject_uncomputable_certificates(self) -> CertificationConfigSchema:
        """Refuse, at load, the certificates whose inputs no source produces yet."""
        if self.pathology_recall.enabled:
            raise ValueError(
                "certification.pathology_recall is not computable: no lesion detector "
                "produces the empirical_recall it certifies (missing lesion detector, #3041). "
                "Keep the arm needs_implementation and set enabled: false."
            )
        if self.pac_bayes.enabled:
            raise ValueError(
                "certification.pac_bayes is not computable: no KL(Q||P) estimator exists "
                "for the posterior (missing KL(Q||P) estimator, #3041). Keep the arm "
                "needs_implementation and set enabled: false."
            )
        return self


__all__ = [
    "CHDCertificateConfig",
    "CertificationConfigSchema",
    "ConformalCertificateConfig",
    "DiceRiskCertificateConfig",
    "PACBayesCertificateConfig",
    "PathologyRecallCertificateConfig",
    "ValidationBadgeConfig",
]
