"""Shared config parsing and verdict helpers for the schedule-certification witnesses.

Split out of :mod:`schedule_certification_checks` in the Wave 0 exit-criterion
work (#1400): that module was 534 LOC against the 300 ceiling (NN20). It holds no
witnesses itself -- the five are in :mod:`schedule_nesting_checks` and
:mod:`schedule_allocation_checks`, and ``schedule_certification_checks`` remains
as the facade the tests import through.
"""

from __future__ import annotations

from typing import Any

import torch

from spectramr.infrastructure.physics.bounded_cache import BoundedLRUCache
from spectramr.infrastructure.validation.witness.registry import (
    Severity,
    Stage,
    Tier,
    WitnessVerdict,
)

_CATEGORY = "schedule_certification"
_DEFAULT_MATRIX = (256, 256)

#: One ``KSpaceUndersamplingProcess`` per distinct construction, shared by the
#: three witnesses that ask for one. Each build materialises a mask generator
#: whose accelerator logs ``Creating Accelerator`` at INFO, so an un-memoised
#: helper printed the same line once per call and made a single-owner run look
#: like it had several. Bounded because a corpus sweep audits many configs;
#: keyed on the resolved kwargs, so two arms that resolve alike legitimately
#: share one process.
_PROCESS_MEMO: BoundedLRUCache[tuple[Any, ...], Any] = BoundedLRUCache()


# ---------------------------------------------------------------------------
# config parsing — mirrors scripts/ci/check_acceleration_ladder_realisable.py
# ---------------------------------------------------------------------------
def _is_cold_diffusion(doc: dict) -> bool:
    model = doc.get("model") or {}
    kwargs = model.get("model_kwargs") or {}
    return "cold_diffusion" in str(model.get("model_type") or "") or (
        "cold_diffusion" in str(kwargs.get("process_type") or "")
    )


def _undersampling(doc: dict) -> dict:
    return doc.get("undersampling") or doc.get("acceleration") or {}


def _matrix(doc: dict) -> tuple[int, int]:
    data = doc.get("data") or {}
    sampling = data.get("sampling") or {}
    patch = sampling.get("patch_size") or data.get("patch_size") or list(_DEFAULT_MATRIX)
    try:
        return int(patch[0]), int(patch[1])
    except (TypeError, ValueError, IndexError):
        return _DEFAULT_MATRIX


def _timesteps(doc: dict) -> int:
    kwargs = (doc.get("model") or {}).get("model_kwargs") or {}
    diffusion = (doc.get("training") or {}).get("diffusion") or {}
    return int(kwargs.get("timesteps") or diffusion.get("timesteps") or 28)


def build_process_from_config(doc: dict) -> Any | None:
    """The ``KSpaceUndersamplingProcess`` this config actually builds, or None.

    Routed through ``AccelerationConfigSchema`` + ``resolve_undersampling_kwargs``
    exactly like the runtime generator and the ladder CI gate (issue #550:
    reading raw YAML makes the audit blind to keys the schema drops). Returns
    ``None`` when the config is not a cold-diffusion arm.
    """
    if not _is_cold_diffusion(doc):
        return None
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema
    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        resolve_undersampling_kwargs,
    )

    accel = _undersampling(doc)
    resolved = resolve_undersampling_kwargs(AccelerationConfigSchema(**accel))
    timesteps = _timesteps(doc)

    # Shared rather than rebuilt. The witnesses only read the process -- the two
    # that touch ``enforce_nested`` restore it in a ``finally`` -- so one
    # instance answers all three, and the accelerator is constructed once.
    key = (timesteps, tuple(sorted((k, repr(v)) for k, v in resolved.items())))
    cached = _PROCESS_MEMO.get(key)
    if cached is not None:
        return cached

    process = KSpaceUndersamplingProcess(num_timesteps=timesteps, **resolved)
    _PROCESS_MEMO[key] = process
    return process


def synthetic_spectral_prior(image_shape: tuple[int, int]) -> torch.Tensor:
    """Deterministic complex k-space batch with a ``1/(1+|k|^2)`` energy law.

    Stand-in for a clean data batch on surfaces with no data access. Centred
    (DC at the matrix centre, matching ``fft2c``'s shifted convention), unit
    peak, zero phase — the allocation audit only reads ``|k|^2``.
    """
    h, w = image_shape
    ky = torch.arange(h, dtype=torch.float32) - h // 2
    kx = torch.arange(w, dtype=torch.float32) - w // 2
    r2 = ky[:, None] ** 2 + kx[None, :] ** 2
    mag = 1.0 / (1.0 + r2)
    return mag.sqrt().to(torch.complex64).view(1, 1, h, w)


def _not_applicable(name: str, tier: Tier, severity: Severity = Severity.INFO) -> WitnessVerdict:
    """A witness that does not apply reports a skip, never a warning.

    The default used to be WARNING, so every non-cold arm failed ``--strict``
    on two schedule witnesses that had nothing to say about it (three ``vf``
    SR arms on 2026-09-02). Non-negotiable 4 makes a warning a failure; a
    not-applicable check is therefore INFO and ``passed=True``.
    """
    return WitnessVerdict(
        witness_name=name,
        passed=True,
        message="not a k-space cold-diffusion arm; schedule certification does not apply",
        severity=severity,
        category=_CATEGORY,
        stage=Stage.CONSTRUCT,
        tier=tier,
    )


def _declared_certification_levels(doc: dict) -> list[dict]:
    """Offline C2/C3 estimates declared under ``undersampling.certification``.

    The reach ``tau_hat``, per-level increment ``delta_hat``, and tangential
    defect ``theta_hat`` are estimator-dependent (they need embeddings of the
    admissible sets, ``core/metrics/manifold_diagnostics.py``) and therefore
    CANNOT be computed on the config surface. A config may *declare* measured
    values instead::

        undersampling:
          certification:
            per_level:
              - {t: 1, delta_hat: 0.9, tau_hat: 2.4, theta_hat: 0.31}

    ``AccelerationConfigSchema`` runs ``extra: ignore`` so the block never
    perturbs the built process. delta_hat and tau_hat must come from the SAME
    estimation law — pairing a synthetic-prior delta with a real-data reach
    produces a ratio that is not ``kappa_t``.
    """
    certification = _undersampling(doc).get("certification") or {}
    levels = certification.get("per_level") or []
    return [lv for lv in levels if isinstance(lv, dict)]


# ---------------------------------------------------------------------------
# witnesses


__all__ = [
    "build_process_from_config",
    "synthetic_spectral_prior",
]
