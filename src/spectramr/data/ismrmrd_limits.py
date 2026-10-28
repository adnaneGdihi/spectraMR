"""The phase-encode range a file acquired, read from its ISMRMRD header (#2572).

``predict`` has no fully sampled reference to read the acquired support from, and
an already-undersampled input cannot tell never-acquired padding from lines the
undersampling dropped. The header's ``kspace_encoding_step_1`` limits name the
range the scanner actually encoded, so they are the source of truth here.

The limits name lines in the ENCODED k-space. fastMRI places them in the stored plane by
``center`` (``fastmri.data.mri_data``): the acquired lines run from
``encodedSpace.y // 2 - center`` for ``maximum + 1`` lines, so reading ``[minimum, maximum]``
as array indices shifts the support whenever ``center`` is off the midpoint. A header
without ``center`` falls back to the ISMRMRD index reading.

M4Raw writes the line COUNT into ``maximum`` instead (measured over all 52 files;
``scripts/physics/measure_phase_encode_axis.py`` classifies a file either way). An M4Raw
file is a full acquisition, so its data occupies one contiguous run of exactly
``maximum`` lines, and that run is the support. A header that fits neither reading raises,
never guessed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch

_NS = "{http://www.ismrm.org/ISMRMRD}"
_HDF5_SUFFIXES = (".h5", ".hdf5")

__all__ = [
    "EncodingLimitsUnavailableError",
    "acquired_lines_for_input",
    "acquired_lines_from_limits",
    "parse_header",
    "placed_line_range",
    "read_acquired_lines",
    "read_limits_header",
]


class EncodingLimitsUnavailableError(ValueError):
    """The file carries no usable ``kspace_encoding_step_1`` limits."""


def parse_header(raw: Any) -> dict[str, Any]:
    """Encoded/recon matrix sizes and ``kspace_encoding_step_1`` limits, None where absent."""
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes | bytearray) else str(raw)
    try:
        enc = ET.fromstring(text).find(f"{_NS}encoding")
    except ET.ParseError:
        return {}
    if enc is None:
        return {}

    def _center() -> int | None:
        text = enc.findtext(f"{_NS}encodingLimits/{_NS}kspace_encoding_step_1/{_NS}center")
        return None if text is None else int(text)

    def _ints(node_path: str, keys: tuple[str, ...]) -> tuple[int, ...] | None:
        node = enc.find(node_path)
        if node is None:
            return None
        return tuple(int(node.findtext(f"{_NS}{k}", "0")) for k in keys)

    return {
        "encoded": _ints(f"{_NS}encodedSpace/{_NS}matrixSize", ("x", "y")),
        "recon": _ints(f"{_NS}reconSpace/{_NS}matrixSize", ("x", "y")),
        "step1": _ints(f"{_NS}encodingLimits/{_NS}kspace_encoding_step_1", ("minimum", "maximum")),
        "step1_center": _center(),
    }


def acquired_lines_from_limits(
    limits: tuple[int, int] | None, num_lines: int, rows: int, *, source: str
) -> torch.Tensor:
    """``[rows, num_lines]`` bool: lines ``minimum..maximum`` inclusive, the same for every row.

    The layout is what ``KSpaceUndersamplingProcess.acquired_lines`` returns.

    Raises:
        EncodingLimitsUnavailableError: ``limits`` is None or a degenerate range.
        ValueError: the range does not fit the plane's ``num_lines``.
    """
    if limits is None:
        raise EncodingLimitsUnavailableError(
            f"{source}: the ISMRMRD header has no encodingLimits/kspace_encoding_step_1, "
            "so the acquired phase-encode support is unknown."
        )
    lo, hi = limits
    if lo < 0 or hi < lo:
        raise EncodingLimitsUnavailableError(
            f"{source}: kspace_encoding_step_1 [{lo}, {hi}] is not a valid line range."
        )
    if hi >= num_lines:
        raise ValueError(
            f"{source}: kspace_encoding_step_1 [{lo}, {hi}] runs past the {num_lines} "
            "phase-encode lines of the input plane."
        )
    lines = torch.zeros(rows, num_lines, dtype=torch.bool)
    lines[:, lo : hi + 1] = True
    return lines


def read_limits_header(path: str | Path) -> dict[str, Any]:
    """The parsed ISMRMRD header of the HDF5 file at ``path``, which must carry step-1 limits.

    Raises:
        EncodingLimitsUnavailableError: not an HDF5 file, no ``ismrmrd_header`` (dataset or
            attribute), or a header without ``kspace_encoding_step_1``; the message names
            the file and the missing key.
    """
    p = Path(path)
    if p.suffix.lower() not in _HDF5_SUFFIXES:
        raise EncodingLimitsUnavailableError(
            f"{p}: {p.suffix or 'this file'} carries no ISMRMRD header (key 'ismrmrd_header'); "
            "the acquired phase-encode support can only be read from an HDF5 file."
        )
    import h5py

    with h5py.File(p, "r") as f:
        if "ismrmrd_header" in f:
            raw = f["ismrmrd_header"][()]
        elif "ismrmrd_header" in f.attrs:
            raw = f.attrs["ismrmrd_header"]
        else:
            raise EncodingLimitsUnavailableError(
                f"{p}: missing key 'ismrmrd_header' (no dataset, no attribute)."
            )
    header = parse_header(raw)
    if header.get("step1") is None:
        raise EncodingLimitsUnavailableError(
            f"{p}: ismrmrd_header has no encodingLimits/kspace_encoding_step_1 "
            "(or is not valid ISMRMRD XML)."
        )
    return header


def placed_line_range(header: dict[str, Any], num_lines: int, *, source: str) -> tuple[int, int]:
    """``(first, last)`` inclusive plane indices of the encoded lines, by fastMRI's placement.

    With ``center``: ``first = encodedSpace.y // 2 - center`` and ``last = first + maximum``,
    which needs the plane to be the encoded width. Without it: ``(minimum, maximum)``.

    Raises:
        ValueError: the plane is not the encoded width, or the range leaves the plane.
    """
    lo, hi = header["step1"]
    center = header.get("step1_center")
    encoded = header.get("encoded")
    if center is not None and encoded is not None:
        enc_y = int(encoded[1])
        if enc_y != num_lines:
            raise ValueError(
                f"{source}: the header encodes {enc_y} phase-encode lines but the input plane "
                f"has {num_lines}; the limits cannot be placed on a cropped or padded plane."
            )
        lo = enc_y // 2 - int(center)
        hi = lo + int(header["step1"][1])
    if lo < 0 or hi < lo or hi >= num_lines:
        raise ValueError(
            f"{source}: kspace_encoding_step_1 places lines [{lo}, {hi}], outside the "
            f"{num_lines} phase-encode lines of the input plane."
        )
    return lo, hi


def read_acquired_lines(path: str | Path, num_lines: int, rows: int) -> torch.Tensor:
    """The acquired lines of the HDF5 file at ``path`` by the placed range, ``[rows, N]``.

    Raises:
        EncodingLimitsUnavailableError: see :func:`read_limits_header`.
    """
    source = str(Path(path))
    lo, hi = placed_line_range(read_limits_header(path), num_lines, source=source)
    return acquired_lines_from_limits((lo, hi), num_lines, rows, source=source)


def _one_run(occupied: torch.Tensor) -> bool:
    """Whether the occupied lines form a single contiguous run."""
    idx = occupied.nonzero().flatten()
    return bool(idx.numel()) and int(idx[-1] - idx[0]) + 1 == int(idx.numel())


def acquired_lines_for_input(path: str | Path, data_lines: torch.Tensor) -> torch.Tensor:
    """The acquired lines of the file at ``path`` given the lines its data occupies.

    ``data_lines`` is the ``[rows, N]`` bool of lines carrying data in the loaded input
    (``acquired_lines_of``). Two readings of ``kspace_encoding_step_1``, tried in order:

    * **Count in ``maximum``** (M4Raw): the data occupies one contiguous run of exactly
      ``maximum`` lines, so the file is a full acquisition and that run is the support.
      A fastMRI full acquisition occupies ``maximum + 1`` lines and never matches.
    * **Placed range** (fastMRI, by ``center``; see :func:`placed_line_range`): the range is
      the support when it contains every line the data occupies, which an undersampled
      input does.

    Raises:
        EncodingLimitsUnavailableError: no usable header limits.
        ValueError: the limits fit neither reading for this data.
    """
    source = str(Path(path))
    header = read_limits_header(path)
    num_rows, num_lines = data_lines.shape
    occupied = data_lines.any(dim=0)
    maximum = int(header["step1"][1])
    count = int(occupied.sum())
    if count == maximum and _one_run(occupied):
        return occupied.unsqueeze(0).expand(num_rows, num_lines).clone()
    minimum = int(header["step1"][0])
    if header.get("step1_center") is None and minimum == 0 and maximum < num_lines - 1:
        # ``[0, maximum]`` with no ``center`` is how M4Raw writes a line COUNT, and an
        # undersampled input no longer shows the run that count describes. Read as
        # indices it would pin the support to the plane's first lines, a shifted band
        # that still contains every surviving line, so it would pass the check below.
        raise ValueError(
            f"{source}: kspace_encoding_step_1 fits neither reading for this data: [0, "
            f"{maximum}] with no center is M4Raw's line-count convention, and the data "
            f"occupies {count} lines, not one run of {maximum}. An undersampled input cannot "
            "show that run, and reading the range as indices would shift the support. "
            "Predict on the fully sampled file."
        )
    lo, hi = placed_line_range(header, num_lines, source=source)
    placed = acquired_lines_from_limits((lo, hi), num_lines, num_rows, source=source)
    if not bool((occupied & ~placed[0]).any()):
        return placed
    raise ValueError(
        f"{source}: kspace_encoding_step_1 fits neither reading for this data: the data "
        f"occupies {count} of {num_lines} lines, not one run of maximum={maximum} lines and "
        f"not inside the placed range [{lo}, {hi}]. Check the file with "
        "scripts/physics/measure_phase_encode_axis.py."
    )
