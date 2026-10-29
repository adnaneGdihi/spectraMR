r"""Hold one subject out of the hardware-warp fit and score the fit on that subject's pairs.

The calibration's observations are repeat-session pairs, several of one head, so the natural
cross-validation unit is the subject, as it is for the framework's ``loso_subject`` split of the
manifest. A fold fits the warp on every other subject's pairs and predicts the held-out pairs'
registration residual through the same commutator model, :math:`\mathrm{ad}_{\xi} v` with the
held-out pair's own :math:`\xi`; the residual left after subtracting that prediction, against
the residual before, is the out-of-sample evidence that the fitted field is the scanner's and not
the fitted heads'. Nothing here opens a file: the strategy supplies the loaded pose set and
residuals and stamps the report into the artifact it writes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from spectramr.infrastructure.physics.hardware_warp_calibration import project_onto_basis
from spectramr.infrastructure.physics.hardware_warp_commutator import (
    Generator,
    HarmonicGradientBasis,
    ball_points,
    stacked_design,
)

__all__ = [
    "held_out_report",
    "predict_pair_residual",
    "split_pairs_by_subject",
]

Residual = tuple[torch.Tensor, torch.Tensor]


def split_pairs_by_subject(
    pose_set: dict[str, Any], residuals: Sequence[Residual], holdout_subject: str
) -> tuple[dict[str, Any], list[Residual], list[dict[str, Any]], list[Residual]]:
    """``(fit_pose_set, fit_residuals, held_pairs, held_residuals)`` for one held-out subject.

    ``fit_pose_set`` is ``pose_set`` with the held-out subject's pairs removed; ``held_pairs``
    are those pairs' records. Raises ``ValueError`` when the subject is not in the pose set,
    naming the subjects that are, and when holding it out leaves nothing to fit on.
    """
    pairs = pose_set["pairs"]
    if len(residuals) != len(pairs):
        raise ValueError(f"{len(pairs)} pairs but {len(residuals)} residuals")
    subjects = sorted({str(pair["subject_id"]) for pair in pairs})
    if holdout_subject not in subjects:
        raise ValueError(
            f"holdout_subject={holdout_subject!r} is not among the pose set's subjects {subjects}"
        )
    held = [k for k, pair in enumerate(pairs) if str(pair["subject_id"]) == holdout_subject]
    kept = [k for k in range(len(pairs)) if k not in set(held)]
    if not kept:
        raise ValueError(
            f"holding out {holdout_subject!r} leaves no pair to fit on; the pose set has "
            "pairs of that subject only"
        )
    fit_pose_set = {**pose_set, "pairs": [pairs[k] for k in kept]}
    return (
        fit_pose_set,
        [residuals[k] for k in kept],
        [pairs[k] for k in held],
        [residuals[k] for k in held],
    )


def predict_pair_residual(
    artifact: dict[str, Any], generator: Generator, points: torch.Tensor
) -> torch.Tensor:
    """``ad_xi v`` of the artifact's warp at ``points`` ``[n, 3]`` for one pair's ``xi``.

    The artifact's coefficients live in the canonical basis, orthonormal on the canonical ball
    cloud; the prediction needs the same field's commutator on the pair's own cloud, so the
    coefficients are changed to a basis orthonormal on that cloud (exact: both span the harmonic
    gradient fields of the same degrees) and the design of that basis is applied. Points are in
    RAS mm relative to the artifact's ``center_mm``, as the residuals are.
    """
    points = torch.as_tensor(points, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be [n, 3], got {tuple(points.shape)}")
    lo, hi = int(artifact["min_degree"]), int(artifact["max_degree"])
    canonical = HarmonicGradientBasis(
        ball_points(int(artifact["basis_points_per_axis"]), float(artifact["ball_radius_mm"])),
        hi,
        lo,
    )
    if points.shape[0] * 3 < canonical.n_basis:
        raise ValueError(
            f"{points.shape[0]} points cannot carry a basis of {canonical.n_basis} fields"
        )
    held = HarmonicGradientBasis(points, hi, lo)
    coeffs = torch.as_tensor(artifact["coeffs"], dtype=torch.float64)
    coeffs_held = project_onto_basis(canonical, held) @ coeffs
    omega, translation = (torch.as_tensor(g, dtype=torch.float64) for g in generator)
    return (stacked_design(held, [(omega, translation)]) @ coeffs_held).reshape(-1, 3)


def held_out_report(
    artifact: dict[str, Any],
    held_pairs: Sequence[dict[str, Any]],
    held_residuals: Sequence[Residual],
) -> dict[str, Any]:
    """Per-point vector RMS of the held-out residual before and after removing the prediction.

    All held-out pairs of the subject are pooled (a subject with four sessions contributes two
    pairs). ``rms_predicted_mm`` is the size of the modelled commutator itself, so a reader can
    tell a small warp from a fit that failed to generalise when ``after`` is close to
    ``before``. The report is a measurement and is never turned into a verdict here; its keys
    are the artifact contract's ``HELD_OUT_KEYS`` (:mod:`spectramr.data.hardware_warp_artifacts`).
    """
    if not held_pairs or len(held_pairs) != len(held_residuals):
        raise ValueError(
            f"{len(held_pairs)} held-out pairs but {len(held_residuals)} residuals; need at "
            "least one of each"
        )
    subjects = {str(pair["subject_id"]) for pair in held_pairs}
    if len(subjects) != 1:
        raise ValueError(f"held-out pairs must belong to one subject, got {sorted(subjects)}")
    before = predicted = after = 0.0
    n_points = 0
    for pair, (points, residual) in zip(held_pairs, held_residuals, strict=True):
        generator = (
            torch.as_tensor(pair["omega"], dtype=torch.float64),
            torch.as_tensor(pair["translation"], dtype=torch.float64),
        )
        residual = torch.as_tensor(residual, dtype=torch.float64)
        fitted = predict_pair_residual(artifact, generator, points)
        before += float((residual**2).sum())
        predicted += float((fitted**2).sum())
        after += float(((residual - fitted) ** 2).sum())
        n_points += int(residual.shape[0])
    return {
        "subject_id": subjects.pop(),
        "n_pairs": len(held_pairs),
        "n_points": n_points,
        "rms_before_mm": math.sqrt(before / n_points),
        "rms_predicted_mm": math.sqrt(predicted / n_points),
        "rms_after_mm": math.sqrt(after / n_points),
    }
