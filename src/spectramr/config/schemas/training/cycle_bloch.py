"""Config for CycleBlochStrategy (``config.training.cycle_bloch``).

The strategy closes ULF -> G -> HF -> (PD, T1, T2[, B1+]) -> ULF. ``backward_model``
picks the operator for the last arrow:

- ``bloch_fixed``: the historical spin-echo render at TR 500 ms / TE 14 ms, with no
  field dependence.
- ``ulf_forward``: ``DifferentiableULFForwardOperator``. It transports T1/T2 from
  ``tissue_field_tesla`` to ``physics.field_strength`` by power-law dispersion, then
  applies the Ernst steady state at the declared sequence.
- ``epg_tse``: the extended-phase-graph TSE echo (``tse_forward.tse_signal``) at the
  declared TR, TE, echo spacing, echo-train length and refocusing angle.

``b1_plus: estimated`` adds a fourth estimator channel, the relative transmit field,
which scales the flip angles of ``ulf_forward`` or ``epg_tse``. ``b1_tv_weight`` sets
the weight of its total-variation penalty.

``sequences`` replaces the scalar sequence fields with one sequence per contrast, keyed
by ``data.multi_contrast.contrast_map`` names; each sample is rendered with its own
contrast's sequence, and ``ti_ms`` makes it inversion-prepared (T1w, FLAIR at 64 mT).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from spectramr.config.schemas.strictness import StrictSchema

#: Sequence fields each backward model reads; every one is required and no other is allowed.
_READS: dict[str, frozenset[str]] = {
    "bloch_fixed": frozenset(),
    "ulf_forward": frozenset({"ulf_tr_ms", "ulf_te_ms", "ulf_flip_deg", "tissue_field_tesla"}),
    "epg_tse": frozenset(
        {"ulf_tr_ms", "ulf_te_ms", "echo_spacing_ms", "echo_train_length", "refocus_deg"}
    ),
}
_SEQUENCE_FIELDS = frozenset().union(*_READS.values())

#: Per-contrast sequence fields each backward model reads; ``ti_ms`` is optional in both.
_TABLE_READS: dict[str, frozenset[str]] = {
    "ulf_forward": frozenset({"tr_ms", "te_ms", "flip_deg"}),
    "epg_tse": frozenset({"tr_ms", "te_ms", "echo_spacing_ms", "echo_train_length", "refocus_deg"}),
}
_TABLE_FIELDS = frozenset({"ti_ms"}).union(*_TABLE_READS.values())


class CycleBlochSequence(StrictSchema):
    """One contrast's ULF acquisition, as the backward model renders it (times in ms)."""

    tr_ms: float = Field(..., gt=0.0, description="Repetition time (ms).")
    te_ms: float = Field(..., gt=0.0, description="(Effective) echo time (ms).")
    ti_ms: float | None = Field(
        default=None,
        gt=0.0,
        description="Inversion time (ms) of an inversion-prepared sequence; None: no inversion.",
    )
    flip_deg: float | None = Field(
        default=None, gt=0.0, le=180.0, description="Excitation flip angle (deg); ulf_forward."
    )
    echo_spacing_ms: float | None = Field(
        default=None, gt=0.0, description="TSE echo spacing (ms); epg_tse."
    )
    echo_train_length: int | None = Field(
        default=None, ge=1, description="TSE echo-train length; epg_tse."
    )
    refocus_deg: float | None = Field(
        default=None, gt=0.0, le=180.0, description="Refocusing flip angle (deg); epg_tse."
    )


class CycleBlochConfig(StrictSchema):
    """Backward-path knobs for the Cycle-Bloch physics cycle."""

    backward_model: Literal["bloch_fixed", "ulf_forward", "epg_tse"] = Field(
        default="bloch_fixed",
        description=(
            "Operator rendering the estimated tissue maps back to the ULF image. "
            "bloch_fixed: spin echo at TR 500 / TE 14 ms, field-independent. "
            "ulf_forward: DifferentiableULFForwardOperator at physics.field_strength. "
            "epg_tse: EPG turbo-spin-echo echo at the declared echo train."
        ),
    )
    ulf_tr_ms: float | None = Field(
        default=None, gt=0.0, description="ULF repetition time (ms); ulf_forward, epg_tse."
    )
    ulf_te_ms: float | None = Field(
        default=None, gt=0.0, description="ULF (effective) echo time (ms); ulf_forward, epg_tse."
    )
    ulf_flip_deg: float | None = Field(
        default=None,
        gt=0.0,
        le=180.0,
        description="ULF excitation flip angle (degrees); ulf_forward only. 90 gives "
        "the spin-echo signal equation of bloch_fixed.",
    )
    tissue_field_tesla: float | None = Field(
        default=None,
        gt=0.0,
        description="Field (T) at which the estimated T1/T2 maps are expressed; the "
        "operator disperses them from here to physics.field_strength. ulf_forward only.",
    )
    echo_spacing_ms: float | None = Field(
        default=None, gt=0.0, description="TSE echo spacing (ms); epg_tse only."
    )
    echo_train_length: int | None = Field(
        default=None, ge=1, description="TSE echo-train length; epg_tse only."
    )
    refocus_deg: float | None = Field(
        default=None,
        gt=0.0,
        le=180.0,
        description="Nominal refocusing flip angle (degrees); epg_tse only.",
    )
    b1_plus: Literal["off", "estimated"] = Field(
        default="off",
        description="off: nominal flip angles. estimated: the parameter estimator also "
        "emits a relative B1+ map in (0, 2) that scales them (ulf_forward, epg_tse).",
    )
    b1_tv_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the total-variation penalty on the estimated B1+ map.",
    )
    sequences: dict[str, CycleBlochSequence] | None = Field(
        default=None,
        description="Per-contrast sequences keyed by data.multi_contrast.contrast_map names; "
        "each sample is rendered with its own contrast's sequence (ulf_forward, epg_tse). "
        "Replaces the scalar ulf_tr_ms/ulf_te_ms/... fields.",
    )

    @model_validator(mode="after")
    def _fields_match_backward_model(self) -> CycleBlochConfig:
        reads = _READS[self.backward_model]
        if self.sequences is not None:
            self._check_table()
            reads = reads & {"tissue_field_tesla"}
        declared = {name for name in _SEQUENCE_FIELDS if getattr(self, name) is not None}
        if missing := sorted(reads - declared):
            raise ValueError(
                f"training.cycle_bloch.backward_model={self.backward_model} requires "
                f"{', '.join(missing)}; the operator has no field-agnostic default."
            )
        if unread := sorted(declared - reads):
            raise ValueError(
                f"training.cycle_bloch.{', '.join(unread)} are not read by "
                f"backward_model={self.backward_model}."
            )
        if self.b1_plus == "estimated" and self.backward_model == "bloch_fixed":
            raise ValueError(
                "training.cycle_bloch.b1_plus=estimated needs a flip-angle-dependent "
                "backward model (ulf_forward or epg_tse); bloch_fixed has no flip angle."
            )
        if self.b1_tv_weight > 0.0 and self.b1_plus != "estimated":
            raise ValueError(
                "training.cycle_bloch.b1_tv_weight > 0 regularises an estimated B1+ map; "
                "set b1_plus: estimated or b1_tv_weight: 0."
            )
        return self

    def _check_table(self) -> None:
        if self.backward_model not in _TABLE_READS:
            raise ValueError(
                f"training.cycle_bloch.sequences needs a sequence-dependent backward model "
                f"({sorted(_TABLE_READS)}); backward_model={self.backward_model} is fixed."
            )
        if not self.sequences:
            raise ValueError("training.cycle_bloch.sequences is empty; name each contrast.")
        reads = _TABLE_READS[self.backward_model]
        for name, seq in self.sequences.items():
            declared = {f for f in _TABLE_FIELDS if getattr(seq, f) is not None} - {"ti_ms"}
            if declared != reads:
                raise ValueError(
                    f"training.cycle_bloch.sequences.{name}: backward_model="
                    f"{self.backward_model} reads exactly {sorted(reads)} (+ optional ti_ms); "
                    f"missing {sorted(reads - declared)}, unread {sorted(declared - reads)}."
                )
