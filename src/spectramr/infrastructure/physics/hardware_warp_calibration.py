r"""Calibrate a scanner's hardware warp from a pose set and its repeat-pair residuals.

The pure numerical half of the ``hardware_warp_calibration`` strategy: a pose set (the rigid
motions :math:`\xi_k` between repeat sessions of one head) and the non-rigid registration
residuals :math:`\log\phi_k-\xi_k` of those pairs go in; the artifact record of
:mod:`spectramr.data.hardware_warp_artifacts` comes out. Nothing here opens a file -- loading and
writing belong to the data layer, orchestration to the strategy -- and nothing here is a model.

Warp coefficients are only meaningful together with the point cloud the basis was orthonormalised
on, so the artifact carries ``basis_points_per_axis`` and ``ball_radius_mm`` and every consumer
rebuilds ``HarmonicGradientBasis(ball_points(n, R), max_degree, min_degree)`` from them. The
residuals arrive sampled at each pair's own points (a different head, mask and draw per pair),
so they are fitted in a second basis orthonormal on the union of those clouds, each pair
contributing the design block on its own slice, and the estimate is projected onto the
canonical basis. The projection is exact because both bases span the same space of harmonic
gradient fields, and the Gauss-Markov covariance transforms with it. The per-degree constant is
evaluated on the canonical ball so it agrees with the audit's pose-spread certificate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch

from spectramr.infrastructure.physics.hardware_warp_commutator import (
    Generator,
    HarmonicGradientBasis,
    ball_points,
    estimate_warp,
    evaluate_polynomial_gradients,
    identifiability_constant,
)

__all__ = [
    "calibrate_hardware_warp",
    "concatenate_residuals",
    "pose_set_generators",
    "project_onto_basis",
]


def pose_set_generators(pose_set: dict[str, Any]) -> list[Generator]:
    """The ``(omega, t)`` generator of every pair, in pose-set order.

    ``pose_set`` is a record that already satisfies the contract owned by
    :func:`spectramr.data.hardware_warp_artifacts.validate_pose_set`; nothing is re-checked here.
    ``omega`` is already the axis-angle vector, so it is used as is rather than routed through
    :func:`rigid_generator`, which would reject a pure-translation pair -- legitimate data whose
    commutator ``-H_Phi t`` still observes the warp.
    """
    return [
        (
            torch.as_tensor(pair["omega"], dtype=torch.float64),
            torch.as_tensor(pair["translation"], dtype=torch.float64),
        )
        for pair in pose_set["pairs"]
    ]


def concatenate_residuals(
    residuals: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, list[slice]]:
    """``(points [N, 3], residual [N, 3], slices)``: every pair's cloud stacked, one slice each.

    Pairs need not share a cloud. Each real pair is a different head, mask and random draw, so
    the second basis is orthonormalised on the union and pair ``k`` enters the least squares on
    ``slices[k]`` of it; a shared cloud is the special case where the slices coincide.
    """
    if len(residuals) == 0:
        raise ValueError("no residuals were supplied; every pair needs one")
    clouds, stacked, slices, offset = [], [], [], 0
    for k, (points, residual) in enumerate(residuals):
        points = torch.as_tensor(points, dtype=torch.float64)
        residual = torch.as_tensor(residual, dtype=torch.float64)
        if points.ndim != 2 or points.shape[1] != 3 or residual.shape != points.shape:
            raise ValueError(
                f"pair {k}: points must be [n, 3] and residual the same shape, got "
                f"{tuple(points.shape)} and {tuple(residual.shape)}"
            )
        if not bool(torch.isfinite(points).all()) or not bool(torch.isfinite(residual).all()):
            raise ValueError(f"pair {k}: points or residual contain non-finite values")
        clouds.append(points)
        stacked.append(residual)
        slices.append(slice(offset, offset + points.shape[0]))
        offset += points.shape[0]
    return torch.cat(clouds), torch.cat(stacked), slices


def project_onto_basis(
    source: HarmonicGradientBasis, target: HarmonicGradientBasis
) -> torch.Tensor:
    """Matrix ``M`` with ``c_target = M @ c_source`` for one field expressed in both bases.

    ``M[j, k] = <v'_k, v_j>`` over the target cloud, where ``v'_k`` are the source fields
    evaluated there from their polynomial coefficients; the target's orthonormality on its own
    cloud makes this the exact change of coordinates.
    """
    if (source.min_degree, source.max_degree) != (target.min_degree, target.max_degree):
        raise ValueError("bases must span the same degree range to be projected onto each other")
    fields = torch.cat(
        [
            evaluate_polynomial_gradients(poly, exps, target.points)
            for _, (exps, poly) in sorted(source.blocks.items())
        ],
        dim=0,
    )
    return torch.einsum("kni,jni->jk", fields, target.fields)


def calibrate_hardware_warp(
    pose_set: dict[str, Any],
    residuals: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    max_degree: int,
    min_degree: int,
    ball_radius_mm: float,
    basis_points_per_axis: int,
    unit_id: str,
    software_version: str,
    noise_std: float | None = None,
) -> dict[str, Any]:
    """Least-squares warp coefficients, covariance and certificate as one artifact record.

    Raises ``ValueError`` when the pose set and the knobs disagree on the unit, the software
    version or the ball radius, when the residuals do not match the pairs, or when the shared
    cloud is too small to determine the basis.
    """
    if pose_set.get("unit_id") != unit_id or pose_set.get("software_version") != software_version:
        raise ValueError(
            f"pose set is labelled unit {pose_set.get('unit_id')!r} / software "
            f"{pose_set.get('software_version')!r} but the calibration declares "
            f"{unit_id!r} / {software_version!r}; a warp is specific to both"
        )
    radius = float(pose_set.get("ball_radius_mm", float("nan")))
    if not math.isclose(radius, ball_radius_mm, rel_tol=1e-6):
        raise ValueError(
            f"pose set ball_radius_mm={radius} differs from the declared {ball_radius_mm}"
        )
    generators = pose_set_generators(pose_set)
    if len(residuals) != len(generators):
        raise ValueError(
            f"{len(generators)} pairs but {len(residuals)} residuals; every pair needs exactly one"
        )
    points, stacked, slices = concatenate_residuals(residuals)
    canonical = HarmonicGradientBasis(
        ball_points(basis_points_per_axis, ball_radius_mm), max_degree, min_degree
    )
    if points.shape[0] * 3 < canonical.n_basis:
        raise ValueError(
            f"{points.shape[0]} residual points cannot determine {canonical.n_basis} basis fields"
        )
    fitted = HarmonicGradientBasis(points, max_degree, min_degree)
    coeffs_fit, cov_fit = estimate_warp(
        fitted, generators, stacked, noise_std=noise_std, slices=slices
    )
    change = project_onto_basis(fitted, canonical)
    coeffs = change @ coeffs_fit
    covariance = None if cov_fit is None else change @ cov_fit @ change.T
    constants = identifiability_constant(canonical, generators)
    return {
        "unit_id": unit_id,
        "software_version": software_version,
        "min_degree": int(min_degree),
        "max_degree": int(max_degree),
        "ball_radius_mm": float(ball_radius_mm),
        "center_mm": [float(c) for c in pose_set["center_mm"]],
        "basis_points_per_axis": int(basis_points_per_axis),
        "coeffs": [float(c) for c in coeffs],
        "degrees": [int(d) for d in canonical.degrees],
        "covariance": None if covariance is None else covariance.tolist(),
        "constant_per_degree": {str(d): float(v) for d, v in constants.items()},
        "n_pairs": len(generators),
    }
