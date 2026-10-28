"""``data.acquisition``: how the dataset's k-space was acquired, declared per arm.

Two facts that a line mask needs and that differ between datasets (issue #2160):

* ``phase_encode_axis`` -- the dimension of the ``[..., H, W]`` k-space plane,
  in the frame the model trains on, along which the scanner stepped
  phase-encode lines. M4Raw T1/T2/FLAIR and fastMRI brain: -1. M4Raw GRE: -2.
  ``undersampling.mask_direction: phase`` resolves through it, and there is no
  default: an arm that declares a ``phase``/``readout`` mask without this block
  raises at build.
* ``acquired_pe_lines`` -- how many phase-encode lines were acquired: one count,
  a count per contrast (M4Raw: ``{FLAIR: 198, T1: 195, T2: 195}``), or
  ``measured`` when the count varies per volume (fastMRI) and is read off the
  data's own support.

``scripts/physics/measure_phase_encode_axis.py --arm <yaml>`` prints the block
from the files the arm's manifests name. Distinct from the top-level
``acquisition:`` block, which configures acquisition *design* (PILOT, BALD).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

__all__ = ["DataAcquisitionSchema"]


class DataAcquisitionSchema(BaseModel):
    """The dataset's phase-encode axis and acquired phase-encode line count."""

    model_config = ConfigDict(protected_namespaces=(), extra="forbid", frozen=True)

    phase_encode_axis: Literal[-1, -2] = Field(
        description=(
            "Dimension of the [..., H, W] k-space plane carrying the phase-encode "
            "lines, in the training batch frame: -1 (columns) or -2 (rows)."
        ),
    )
    acquired_pe_lines: StrictInt | dict[str, StrictInt] | Literal["measured"] = Field(
        description=(
            "Acquired phase-encode lines: one count, a count per contrast, or "
            "'measured' to read each volume's support from its own data."
        ),
    )

    @field_validator("phase_encode_axis", mode="before")
    @classmethod
    def _axis_is_an_int(cls, value: object) -> object:
        # Literal[-1, -2] alone accepts nothing else, but a bool is an int to
        # Python, so name the failure rather than leave it to the Literal.
        if isinstance(value, bool):
            raise ValueError(f"phase_encode_axis must be -1 or -2, got {value!r}.")
        return value

    @field_validator("acquired_pe_lines")
    @classmethod
    def _counts_are_positive(cls, value: int | dict[str, int] | str) -> int | dict[str, int] | str:
        if isinstance(value, int) and value < 1:
            raise ValueError(f"acquired_pe_lines must be >= 1, got {value}.")
        if isinstance(value, dict):
            if not value:
                raise ValueError("acquired_pe_lines mapping must name at least one contrast.")
            bad = {key: count for key, count in value.items() if not key.strip() or count < 1}
            if bad:
                raise ValueError(
                    f"acquired_pe_lines entries {bad} need a contrast name and a count >= 1."
                )
        return value
