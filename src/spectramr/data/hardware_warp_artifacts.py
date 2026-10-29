"""Pose sets and calibration artifacts of the hardware-warp identification: the files' one owner.

Two records cross the file system. The **pose set** (JSON) is the input: the rigid motion of
every repeat-session pair of one head on one scanner, with a path to the pair's non-rigid
registration residual (a ``torch.save``-d ``{"points", "residual"}`` pair in scanner-frame mm).
The **artifact** (JSON) is the output: the warp coefficients in the canonical harmonic-gradient
basis, together with the two numbers that basis is rebuilt from, the per-degree identifiability
constant and the covariance. Every reader and writer of either goes through here (non-negotiable
7), and both are validated on the way in and on the way out, so a consumer never has to guess
whether a missing key means "not calibrated" or "wrong file".

Sign convention of ``coeffs``. The artifact's field ``v`` is the *observed-to-true* displacement:
a voxel observed at scanner position ``x`` holds tissue whose true position is ``x + v(x)``. That
is the ``psi = exp(v)`` of the identification ``phi = psi^-1 h psi``, whose first-order residual
``ad_xi v`` the estimator fits; it is fixed by the theory, not by which session is called fixed
(swapping them flips ``xi`` and the residual together). A consumer that corrects an image must
therefore read the observed image at the ``x`` solving ``x + v(x) = r`` to place tissue at its
true position ``r`` -- ``r - v(r)`` to first order -- never at ``r + v(r)``.

Frame. Positions, displacements and generators are in RAS millimetres relative to ``center_mm``,
the field of view's centre that the registration expressed everything about (it is magnet-fixed
on a fixed-FOV scanner, while the NIfTI origin of a Swoop export is the FOV corner). RAS is the
frame of the framework's data path; the registration converts its LPS measurements at its own
boundary. A consumer evaluates ``v`` at ``r - center_mm``.

``held_out`` is ``null`` for a fit on every pair; a hold-one-subject-out fold sets it to the
subject and its pooled residual RMS before and after the fitted commutator is removed. Such an
artifact is a fold's evidence, not the production warp, and an image consumer refuses it.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "ARTIFACT_KEYS",
    "HELD_OUT_KEYS",
    "PAIR_KEYS",
    "POSE_SET_KEYS",
    "identity_artifact",
    "load_artifact",
    "load_pair_residuals",
    "load_pose_set",
    "validate_artifact",
    "validate_pose_set",
    "write_artifact",
    "write_pose_set",
]

POSE_SET_KEYS: frozenset[str] = frozenset(
    {"unit_id", "software_version", "ball_radius_mm", "center_mm", "pairs"}
)
PAIR_KEYS: frozenset[str] = frozenset(
    {"subject_id", "sessions", "omega", "translation", "residual_path"}
)
ARTIFACT_KEYS: frozenset[str] = frozenset(
    {
        "unit_id",
        "software_version",
        "min_degree",
        "max_degree",
        "ball_radius_mm",
        "center_mm",
        "basis_points_per_axis",
        "coeffs",
        "degrees",
        "covariance",
        "constant_per_degree",
        "n_pairs",
        "held_out",
    }
)
HELD_OUT_KEYS: frozenset[str] = frozenset(
    {"subject_id", "n_pairs", "n_points", "rms_before_mm", "rms_predicted_mm", "rms_after_mm"}
)


def _require_keys(doc: Any, keys: frozenset[str], what: str) -> dict[str, Any]:
    if not isinstance(doc, dict):
        raise ValueError(f"{what} must be a JSON object, got {type(doc).__name__}")
    missing, unknown = sorted(keys - doc.keys()), sorted(doc.keys() - keys)
    if missing or unknown:
        raise ValueError(f"{what}: missing keys {missing}, unknown keys {unknown}")
    return doc


def _nonempty_str(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def _positive(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise ValueError(f"{what} must be a positive number, got {value!r}")
    return float(value)


def _int_at_least(value: Any, floor: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < floor:
        raise ValueError(f"{what} must be an integer >= {floor}, got {value!r}")
    return value


def _finite_floats(value: Any, n: int | None, what: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or (n is not None and len(value) != n):
        raise ValueError(f"{what} must be a list of {n if n is not None else 'finite'} numbers")
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ValueError(f"{what} contains a non-finite or non-numeric entry {v!r}")
    return [float(v) for v in value]


def validate_pose_set(doc: Any) -> dict[str, Any]:
    """Return ``doc`` when it satisfies the pose-set contract; raise ``ValueError`` otherwise."""
    _require_keys(doc, POSE_SET_KEYS, "pose set")
    _nonempty_str(doc["unit_id"], "pose set unit_id")
    _nonempty_str(doc["software_version"], "pose set software_version")
    _positive(doc["ball_radius_mm"], "pose set ball_radius_mm")
    _finite_floats(doc["center_mm"], 3, "pose set center_mm")
    pairs = doc["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pose set pairs must be a non-empty list")
    for i, pair in enumerate(pairs):
        _require_keys(pair, PAIR_KEYS, f"pairs[{i}]")
        _nonempty_str(pair["subject_id"], f"pairs[{i}].subject_id")
        sessions = pair["sessions"]
        if not isinstance(sessions, list) or len(sessions) != 2:
            raise ValueError(f"pairs[{i}].sessions must list exactly two session ids")
        for s in sessions:
            _nonempty_str(s, f"pairs[{i}].sessions entry")
        _finite_floats(pair["omega"], 3, f"pairs[{i}].omega")
        _finite_floats(pair["translation"], 3, f"pairs[{i}].translation")
        if pair["residual_path"] is not None:
            _nonempty_str(pair["residual_path"], f"pairs[{i}].residual_path")
    return doc


def _read_json(path: Path | str, what: str) -> Any:
    """An absent file is ``FileNotFoundError``; unreadable JSON is ``ValueError`` naming it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{what} not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{what} at {path} is not readable JSON: {exc}") from exc


def load_pose_set(path: Path | str) -> dict[str, Any]:
    """Read and validate a pose-set JSON; a contract violation is ``ValueError`` naming the key."""
    return validate_pose_set(_read_json(path, "pose set"))


def _write_json(doc: dict[str, Any], path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    return path


def write_pose_set(pose_set: dict[str, Any], path: Path | str) -> Path:
    """Validate and write a pose set; the file is the contract, so nothing unvalidated lands."""
    return _write_json(validate_pose_set(pose_set), path)


def load_pair_residuals(
    pose_set: dict[str, Any], base_dir: Path | str
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """``(points, residual)`` of every pair, in pose-set order, as float64 ``[n, 3]`` tensors.

    ``residual_path`` is relative to ``base_dir`` (the pose set's own directory). A pair without
    one cannot contribute an observation, so it is refused rather than skipped: a pose set may
    be certified for pose spread before its residuals exist, but it cannot be calibrated.
    """
    out = []
    for i, pair in enumerate(pose_set["pairs"]):
        rel = pair["residual_path"]
        if rel is None:
            raise ValueError(
                f"pairs[{i}] ({pair['subject_id']}, {pair['sessions']}) has no residual_path; "
                "the calibration needs one residual per pair"
            )
        path = Path(base_dir) / rel
        if not path.is_file():
            raise FileNotFoundError(f"pairs[{i}] residual not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or set(payload) != {"points", "residual"}:
            raise ValueError(f"{path}: expected a dict with exactly 'points' and 'residual'")
        points = torch.as_tensor(payload["points"], dtype=torch.float64)
        residual = torch.as_tensor(payload["residual"], dtype=torch.float64)
        if points.ndim != 2 or points.shape[1] != 3 or residual.shape != points.shape:
            raise ValueError(
                f"{path}: points must be [n, 3] and residual the same shape, got "
                f"{tuple(points.shape)} and {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(residual).all()):
            raise ValueError(f"{path}: points or residual contain non-finite values")
        out.append((points, residual))
    return out


def validate_artifact(artifact: Any) -> dict[str, Any]:
    """Return ``artifact`` when it satisfies the artifact contract; raise ``ValueError`` otherwise.

    ``degrees`` must be the canonical block order -- ``2l + 1`` entries of each degree from
    ``min_degree`` to ``max_degree`` -- because that is the order ``HarmonicGradientBasis``
    assigns to ``coeffs``; anything else would silently pair coefficients with the wrong fields.
    """
    _require_keys(artifact, ARTIFACT_KEYS, "artifact")
    _nonempty_str(artifact["unit_id"], "artifact unit_id")
    _nonempty_str(artifact["software_version"], "artifact software_version")
    lo = _int_at_least(artifact["min_degree"], 1, "artifact min_degree")
    hi = _int_at_least(artifact["max_degree"], lo, "artifact max_degree")
    _positive(artifact["ball_radius_mm"], "artifact ball_radius_mm")
    _finite_floats(artifact["center_mm"], 3, "artifact center_mm")
    _int_at_least(artifact["basis_points_per_axis"], 2, "artifact basis_points_per_axis")
    _int_at_least(artifact["n_pairs"], 0, "artifact n_pairs")
    expected_degrees = [d for d in range(lo, hi + 1) for _ in range(2 * d + 1)]
    if list(artifact["degrees"]) != expected_degrees:
        raise ValueError(
            f"artifact degrees must be the canonical block order {expected_degrees}, "
            f"got {artifact['degrees']}"
        )
    n = len(expected_degrees)
    _finite_floats(artifact["coeffs"], n, "artifact coeffs")
    cov = artifact["covariance"]
    if cov is not None:
        if not isinstance(cov, list) or len(cov) != n:
            raise ValueError(f"artifact covariance must be null or a {n}x{n} matrix")
        for row in cov:
            _finite_floats(row, n, "artifact covariance row")
    constants = artifact["constant_per_degree"]
    if not isinstance(constants, dict) or set(constants) != {str(d) for d in range(lo, hi + 1)}:
        raise ValueError(
            f"artifact constant_per_degree must carry exactly the degrees {lo}..{hi} as strings"
        )
    if min(_finite_floats(list(constants.values()), None, "artifact constant_per_degree")) < 0:
        raise ValueError("artifact constant_per_degree has a negative entry")
    if artifact["held_out"] is not None:
        _validate_held_out(artifact["held_out"])
    return artifact


def _validate_held_out(held: Any) -> None:
    _require_keys(held, HELD_OUT_KEYS, "artifact held_out")
    _nonempty_str(held["subject_id"], "artifact held_out.subject_id")
    for key in ("n_pairs", "n_points"):
        _int_at_least(held[key], 1, f"artifact held_out.{key}")
    rms = [held[k] for k in ("rms_before_mm", "rms_predicted_mm", "rms_after_mm")]
    if min(_finite_floats(rms, 3, "artifact held_out rms")) < 0:
        raise ValueError("artifact held_out rms values must be non-negative")


def identity_artifact(
    *,
    min_degree: int,
    max_degree: int,
    ball_radius_mm: float,
    basis_points_per_axis: int,
    unit_id: str,
    software_version: str,
    center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, Any]:
    """An artifact whose warp is the identity (every coefficient zero), for pipelines with no
    calibration to apply and for tests; ``n_pairs`` is 0 because nothing was measured."""
    degrees = [d for d in range(min_degree, max_degree + 1) for _ in range(2 * d + 1)]
    return validate_artifact(
        {
            "unit_id": unit_id,
            "software_version": software_version,
            "min_degree": int(min_degree),
            "max_degree": int(max_degree),
            "ball_radius_mm": float(ball_radius_mm),
            "center_mm": [float(c) for c in center_mm],
            "basis_points_per_axis": int(basis_points_per_axis),
            "coeffs": [0.0] * len(degrees),
            "degrees": degrees,
            "covariance": None,
            "constant_per_degree": {str(d): 0.0 for d in range(min_degree, max_degree + 1)},
            "n_pairs": 0,
            "held_out": None,
        }
    )


def write_artifact(artifact: dict[str, Any], path: Path | str) -> Path:
    """Validate and write the artifact as JSON, creating parent directories. Returns the path."""
    return _write_json(validate_artifact(artifact), path)


def load_artifact(path: Path | str) -> dict[str, Any]:
    """Read and validate an artifact JSON."""
    return validate_artifact(_read_json(path, "hardware-warp artifact"))
