"""Which axis of the k-space plane a Cartesian line mask removes lines along.

One owner for the translation ``undersampling.mask_direction`` -> accelerator
``line_axis`` (issue #2160). A line mask on ``line_axis='y'`` keeps or drops whole
rows (dim -2); on ``'x'`` whole columns (dim -1). Which of the two is the
phase-encode axis is a property of the dataset, not of the mask, and datasets
differ: M4Raw T1/T2/FLAIR and fastMRI brain acquire phase-encode on dim -1 of
the training batch, M4Raw GRE on dim -2. So ``phase`` and ``readout`` resolve
only through the arm's declared ``data.acquisition.phase_encode_axis``; there is
no default, because the default this replaced (``phase -> 'y'``) made every
M4Raw and fastMRI-brain line mask remove readout lines.

``'y'`` and ``'x'`` remain accepted as raw axis names and need no declaration.
"""

from __future__ import annotations

from typing import Literal

__all__ = [
    "LINE_AXIS_DIM",
    "MEASURE_SCRIPT",
    "VALID_MASK_DIRECTIONS",
    "VALID_PHASE_ENCODE_AXES",
    "line_axis_for",
]

LineAxis = Literal["y", "x"]

#: ``phase_encode_axis`` values: the dimension of the ``[..., H, W]`` k-space plane.
VALID_PHASE_ENCODE_AXES: frozenset[int] = frozenset({-1, -2})

#: The plane dimension each ``line_axis`` indexes its lines along.
LINE_AXIS_DIM: dict[str, int] = {"y": -2, "x": -1}

_AXIS_OF_DIM: dict[int, LineAxis] = {-2: "y", -1: "x"}

#: Spellings of ``undersampling.mask_direction``; the first two need the axis.
VALID_MASK_DIRECTIONS: frozenset[str] = frozenset({"phase", "readout", "y", "x"})

#: Prints the ``data.acquisition`` block for an arm from the files it reads.
MEASURE_SCRIPT = "scripts/physics/measure_phase_encode_axis.py"


def line_axis_for(mask_direction: str, phase_encode_axis: int | None) -> LineAxis:
    """The accelerator ``line_axis`` that ``mask_direction`` names for this dataset.

    Args:
        mask_direction: ``phase`` (lines indexed along the phase-encode axis),
            ``readout`` (the other axis), or a raw ``y``/``x``.
        phase_encode_axis: ``data.acquisition.phase_encode_axis``, -1 or -2;
            ``None`` when the arm declares none.

    Raises:
        ValueError: an unknown direction or axis, or ``phase``/``readout``
            without a declared axis.
    """
    if mask_direction not in VALID_MASK_DIRECTIONS:
        raise ValueError(
            f"undersampling.mask_direction={mask_direction!r}; expected one of "
            f"{sorted(VALID_MASK_DIRECTIONS)}."
        )
    if mask_direction in _AXIS_OF_DIM.values():
        return mask_direction  # type: ignore[return-value]
    if phase_encode_axis is None:
        raise ValueError(
            f"undersampling.mask_direction={mask_direction!r} needs the dataset's "
            "phase-encode axis, and this arm declares no "
            "data.acquisition.phase_encode_axis. Run "
            f"`python {MEASURE_SCRIPT} --arm <this yaml>` on the data and declare "
            "the block it prints (issue #2160)."
        )
    if isinstance(phase_encode_axis, bool) or phase_encode_axis not in VALID_PHASE_ENCODE_AXES:
        raise ValueError(
            f"data.acquisition.phase_encode_axis={phase_encode_axis!r}; expected one of "
            f"{sorted(VALID_PHASE_ENCODE_AXES)} (a dimension of the [..., H, W] plane)."
        )
    phase = _AXIS_OF_DIM[phase_encode_axis]
    if mask_direction == "phase":
        return phase
    return "x" if phase == "y" else "y"
