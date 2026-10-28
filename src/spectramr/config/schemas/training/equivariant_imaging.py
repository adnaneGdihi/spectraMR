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
    measurement_target: Literal["measured", "alternate_repetition"] = Field(
        default="measured",
        description=(
            "What the measurement-consistency anchor scores the reconstruction "
            "against. 'measured' (default): the input measurement y itself, "
            "||A x_hat - y||^2. 'alternate_repetition' (Rep-EI): an independent "
            "repetition of the same slice on the lines `loss_mask` selects, scored "
            "by the declared `masked_kspace_l2` loss -- Lemma 8 of "
            "experiments/inprogress/n2n/PAPER.md with M_j = M_i. Its noise is "
            "independent of the input's, so the anchor is the supervised loss on "
            "those lines plus a constant. Requires `loss_mask` in the batch "
            "(`complementary_rep_masking` emits it), a `masked_kspace_l2` entry "
            "in losses.kspace_losses, and robust_correction off."
        ),
    )

    anchor_mask: Literal["input", "with_structural_zeros"] = Field(
        default="input",
        description=(
            "Which k-space locations the measured anchor scores. 'input': the input "
            "mask M1 only, so every location outside the acquired lines is "
            "unconstrained. 'with_structural_zeros': M1 plus the columns outside the "
            "acquired phase-encode support, where y is zero by construction, so the "
            "reconstruction is pulled to the acquisition's band without any reference. "
            "Needs `support_mask` in the batch (`complementary_rep_masking` emits it)."
        ),
    )
    anchor_normalization: Literal["all_entries", "selected"] = Field(
        default="all_entries",
        description=(
            "How the measured anchor is averaged. 'all_entries' divides by every "
            "k-space entry, zeros included, so its scale falls with the acceleration "
            "and alpha_equivariance means a different balance on every arm. "
            "'selected' divides by the number of scored entries, as masked_kspace_l2 "
            "does, so alpha is comparable across arms."
        ),
    )
    equivariance_target: Literal["transformed", "band_limited"] = Field(
        default="transformed",
        description=(
            "The reference the equivariance term matches f(A T_g x_hat) to. "
            "'transformed': T_g x_hat. 'band_limited': T_g x_hat projected onto the "
            "acquired phase-encode support, so a 90-degree rotation, which moves "
            "full-band readout content onto the band-limited axis, never asks the "
            "network to invent content outside the acquired band. Needs "
            "`support_mask` in the batch."
        ),
    )

    @model_validator(mode="after")
    def _validate_robust_requires_noise(self) -> TrainingConfigEquivariantImaging:
        if self.robust_correction and self.noise_std_estimate is None:
            raise ValueError(
                "equivariant_imaging.robust_correction=true requires "
                "noise_std_estimate (the k-space noise σ); none was set "
                "(pitfall #15: the knob must be wired)."
            )
        if self.robust_correction and self.measurement_target != "measured":
            raise ValueError(
                "equivariant_imaging.robust_correction replaces the anchor on the "
                "MEASURED y with a GSURE risk; it has no meaning against an "
                f"alternate repetition (measurement_target={self.measurement_target!r})."
            )
        anchor_changed = self.anchor_mask != "input" or self.anchor_normalization != "all_entries"
        if anchor_changed and (self.robust_correction or self.measurement_target != "measured"):
            raise ValueError(
                "equivariant_imaging.anchor_mask / anchor_normalization shape the plain "
                "anchor on the measured y; the GSURE risk and the alternate-repetition "
                "anchor (masked_kspace_l2) never read them."
            )
        return self


__all__ = ["TrainingConfigEquivariantImaging"]
