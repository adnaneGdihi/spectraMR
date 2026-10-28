"""Acquisition parameters, and the SNR prior they imply for a quality match.

The ISMRMRD header records how a scan was acquired: field strength, echo and
repetition time, receiver bandwidth, averages. Those numbers *predict* part of the
quality gap between two cohorts, so they belong in the fit as a physically-derived
warm start rather than being ignored while a derivative-free search rediscovers them
from a sharpness statistic.

.. note::
   ``spectramr.data.profiler._parse_ismrmrd_xml`` also reads this header. It is
   deliberately not reused: it is *tolerant* (an unparseable header yields ``None``
   fields so a survey can continue), and it extracts a different set (vendor, coils,
   matrix) without TR or bandwidth. A silently ``None`` field strength here would
   degrade the prior invisibly, so this parser raises on malformed XML and reports a
   genuinely absent field as ``None`` -- a fact, not a failure.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

__all__ = [
    "AcquisitionParams",
    "MissingAcquisitionParameter",
    "acquisition_from_sidecar",
    "predicted_snr_delta_db",
    "read_acquisition_params",
]

#: BIDS-style NIfTI sidecar keys. Times are seconds in the sidecar, milliseconds in
#: :class:`AcquisitionParams`. A sidecar keyed differently (e.g. DICOM tags) raises.
_SIDECAR_KEYS = {
    "field_strength_t": "MagneticFieldStrength",
    "tr_ms": "RepetitionTime",
    "te_ms": "EchoTime",
    "averages": "NumberOfAverages",
    "bandwidth_hz_px": "PixelBandwidth",
}


class MissingAcquisitionParameter(ValueError):
    """A parameter the requested prior depends on is absent from the header."""


@dataclass(frozen=True, slots=True)
class AcquisitionParams:
    """How a scan was acquired. ``None`` means the header did not record it."""

    field_strength_t: float | None = None
    te_ms: float | None = None
    tr_ms: float | None = None
    bandwidth_hz_px: float | None = None
    averages: float | None = None

    def require(self, name: str) -> float:
        """Return a parameter or raise -- never substitute a plausible default."""
        value = getattr(self, name)
        if value is None:
            raise MissingAcquisitionParameter(
                f"{name} is absent from the acquisition header, so the prior that "
                f"depends on it cannot be computed. A substituted default would look "
                f"like a measurement."
            )
        return float(value)


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _as_text(header: Any) -> str:
    if isinstance(header, bytes):
        return header.decode("utf-8", errors="replace")
    return str(header)


def _first_float(root: ET.Element, path: str) -> float | None:
    text = root.findtext(path)
    if text is None or not text.strip():
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_acquisition_params(header: Any) -> AcquisitionParams:
    """Parse acquisition parameters from an ISMRMRD header.

    Raises:
        ValueError: the header is absent or is not parseable XML. An unreadable
            header must stop a prior that would otherwise be computed from
            fabricated numbers.
    """
    if header is None:
        raise ValueError(
            "no ISMRMRD header: acquisition parameters are unknowable and a default "
            "would be a fabrication."
        )
    try:
        root = _strip_ns(ET.fromstring(_as_text(header)))
    except ET.ParseError as exc:
        raise ValueError(f"ismrmrd_header is not valid XML: {exc}") from exc

    # TE/TR may repeat (multi-echo); the first is the representative value.
    te = _first_float(root, "sequenceParameters/TE")
    tr = _first_float(root, "sequenceParameters/TR")
    field = _first_float(root, "acquisitionSystemInformation/systemFieldStrength_T")

    # Receiver bandwidth is not a first-class ISMRMRD field; when a vendor records
    # it, it lands in userParameters. Absent is normal, not an error.
    bandwidth = None
    for tag in ("userParameterDouble", "userParameterLong"):
        for node in root.findall(f"userParameters/{tag}"):
            name = (node.findtext("name") or "").strip().lower()
            if name in {"bandwidth_hz_px", "pixelbandwidth", "bandwidthperpixel"}:
                try:
                    bandwidth = float(node.findtext("value") or "")
                except ValueError:
                    bandwidth = None

    averages = _first_float(root, "sequenceParameters/averages")

    return AcquisitionParams(
        field_strength_t=field,
        te_ms=te,
        tr_ms=tr,
        bandwidth_hz_px=bandwidth,
        averages=averages,
    )


def acquisition_from_sidecar(doc: Any) -> AcquisitionParams:
    """Acquisition parameters from a NIfTI JSON sidecar (BIDS keys, see ``_SIDECAR_KEYS``).

    Field strength is required; TR, TE, averages and bandwidth are optional, as in the
    header reader. Raises :class:`MissingAcquisitionParameter` on an absent field
    strength rather than returning a neutral record.
    """
    if not isinstance(doc, dict):
        raise ValueError(f"sidecar must be a JSON object; got {type(doc).__name__}")

    def _num(key: str) -> float | None:
        value = doc.get(key)
        return None if value is None else float(value)

    field = _num(_SIDECAR_KEYS["field_strength_t"])
    if field is None:
        raise MissingAcquisitionParameter(
            f"sidecar has no {_SIDECAR_KEYS['field_strength_t']!r}; the field strength "
            "the SNR prior depends on is unrecorded, and a guessed one would be a "
            "fabrication."
        )
    tr_s, te_s = _num(_SIDECAR_KEYS["tr_ms"]), _num(_SIDECAR_KEYS["te_ms"])
    return AcquisitionParams(
        field_strength_t=field,
        tr_ms=None if tr_s is None else tr_s * 1000.0,
        te_ms=None if te_s is None else te_s * 1000.0,
        averages=_num(_SIDECAR_KEYS["averages"]),
        bandwidth_hz_px=_num(_SIDECAR_KEYS["bandwidth_hz_px"]),
    )


def predicted_snr_delta_db(
    hq: AcquisitionParams,
    lq: AcquisitionParams,
) -> float:
    r"""Predicted SNR change, in dB, going from the ``hq`` to the ``lq`` acquisition.

    Uses the standard proportionality

    .. math::
        \mathrm{SNR} \;\propto\; B_0 \cdot V_{\text{voxel}} \cdot
        \sqrt{\frac{N_{\text{avg}}}{\mathrm{BW}}}

    keeping only the terms the *acquisition* contributes:

    .. math::
        \Delta\mathrm{SNR}_{\mathrm{dB}} = 20\log_{10}\!\left(
          \frac{B_{0,\mathrm{lq}}}{B_{0,\mathrm{hq}}}
          \sqrt{\frac{N_{\mathrm{lq}}}{N_{\mathrm{hq}}}}
          \sqrt{\frac{\mathrm{BW}_{\mathrm{hq}}}{\mathrm{BW}_{\mathrm{lq}}}}
        \right)

    **The voxel-volume term is deliberately excluded.** ``resample_to_spacing``
    already puts the volume on the low-quality grid, and area-averaging onto coarser
    voxels *raises* SNR exactly as larger voxels do in a real acquisition. Including
    :math:`V_{\text{voxel}}` here would count that gain a second time and hand the
    noise axis a target several dB too optimistic -- which is why a coarse-voxel ULF
    protocol is not simply "noisier everywhere".

    Field strength is required. Averages and bandwidth are optional: a header that
    omits them contributes a ratio of 1, which is the correct neutral element, not a
    guess.

    Returns a NEGATIVE number when the low-quality acquisition is noisier.
    """
    ratio = lq.require("field_strength_t") / hq.require("field_strength_t")

    if lq.averages is not None and hq.averages is not None:
        ratio *= math.sqrt(float(lq.averages) / float(hq.averages))
    if lq.bandwidth_hz_px is not None and hq.bandwidth_hz_px is not None:
        ratio *= math.sqrt(float(hq.bandwidth_hz_px) / float(lq.bandwidth_hz_px))

    if ratio <= 0.0:
        raise ValueError(
            f"predicted SNR ratio must be positive; got {ratio}. Check the header's "
            "field strength, averages and bandwidth."
        )
    return 20.0 * math.log10(ratio)
