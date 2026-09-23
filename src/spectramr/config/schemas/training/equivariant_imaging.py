"""Equivariant Imaging (EI) training sub-schema — Phase A step 3.

Knobs for :class:`spectramr.infrastructure.training.strategies.equivariant_imaging_strategy.EquivariantImagingStrategy`
(Chen et al., ICCV 2021; Robust EI, CVPR 2022). The ``group`` literal mirrors
``spectramr.infrastructure.physics.group_actions.GROUP_REGISTRY`` so an
unsupported group is rejected at load time (pitfall #9). The ``robust_correction``
toggle selects the nc-χ-aware GSURE data-consistency term of the T1 keystone and
requires ``noise_std_estimate`` to be set (pitfall #15: an advertised correction
with no σ would be a dead knob).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TrainingConfigEquivariantImaging(BaseModel):
    """Parameters for the Equivariant Imaging self-supervised reconstruction."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    group: Literal["dihedral", "small_rotation"] = Field(
        default="dihedral",
        description=(
            "Symmetry group supplying T_g. 'dihedral' (D4: 90-degree rotations + "
            "flips, exact on square grids); 'small_rotation' (continuous small "
            "rotations — brain-appropriate, the brain is only approximately "
            "rotation-invariant). Must be a key of group_actions.GROUP_REGISTRY."
        ),
    )
    alpha_equivariance: float = Field(
        default=1.0,
        ge=0.0,
        description=(
            "Weight on the equivariance term relative to measurement consistency. "
            "alpha=0 reduces EI to a measurement-consistency reconstruction."
        ),
    )
    n_group_samples: int = Field(
        default=1,
        ge=1,
        description="Number of group elements sampled (and averaged) per step.",
    )
    max_angle_deg: float = Field(
        default=10.0,
        gt=0.0,
        description="Max rotation magnitude (degrees) for group='small_rotation'.",
    )
    num_angles: int = Field(
        default=4,
        ge=1,
        description="Number of discrete non-identity angles for group='small_rotation'.",
    )
    robust_correction: bool = Field(
        default=False,
        description=(
            "Use the nc-χ-aware GSURE data-consistency term (Robust EI, CVPR 2022 / "
            "T1 keystone) instead of the plain ||A x_hat - y||^2 anchor. Required "
            "true when the strategy is selected via the 'robust_ei' key."
        ),
    )
    noise_std_estimate: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "k-space noise σ for the GSURE term. Required when robust_correction "
            "is true (pitfall #15)."
        ),
    )
    noise_model: Literal["gaussian_kspace", "ncchi_magnitude"] = Field(
        default="gaussian_kspace",
        description=(
            "Likelihood model for the GSURE term: complex-Gaussian per-coil "
            "k-space, or non-central-χ RSS magnitude. Dataset-driven, never "
            "hardcoded to 0.3T."
        ),
    )
    n_coils: int = Field(
        default=1,
        ge=1,
        description="Number of coils (degrees of freedom L) for the nc-χ noise floor.",
    )
    multicoil_operator: bool = Field(
        default=False,
        description=(
            "Use the multi-coil forward operator A = M F S (sense_forward) "
            "instead of the coil-combined A = M F. EI exists to recover what a "
            "single A leaves unidentifiable, so the operator decides whether "
            "the arm has a problem to solve: built explicitly at R=4 on a "
            "16-pixel line, the physical signal set has 12 of 16 directions "
            "unidentifiable through one virtual coil and 0 of 16 through four "
            "physical ones (condition number 34.7). The reconstructed image is "
            "the coil-COMBINED one either way -- the coils enter only through "
            "the operator -- so the model and the group action are unchanged. "
            "Requires complex coil sensitivities in the batch (the "
            "`espirit_sensitivity` transform supplies them) and raises without "
            "them rather than silently reverting to the single-coil operator."
        ),
    )
    split_seed: int = Field(
        default=0,
        ge=0,
        description="RNG seed for reproducible group-element sampling.",
    )

    @model_validator(mode="after")
    def _validate_robust_requires_noise(self) -> TrainingConfigEquivariantImaging:
        if self.robust_correction and self.noise_std_estimate is None:
            raise ValueError(
                "equivariant_imaging.robust_correction=true requires "
                "noise_std_estimate (the k-space noise σ); none was set "
                "(pitfall #15: the knob must be wired)."
            )
        return self


__all__ = ["TrainingConfigEquivariantImaging"]
