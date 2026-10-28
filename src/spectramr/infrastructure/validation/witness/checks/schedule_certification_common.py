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
from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS

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
# config parsing — shared with scripts/ci/check_acceleration_ladder_realisable.py (the horizon is imported from here)
# ---------------------------------------------------------------------------
def _is_cold_diffusion(doc: dict) -> bool:
    model = doc.get("model") or {}
    kwargs = model.get("model_kwargs") or {}
    return "cold_diffusion" in str(model.get("model_type") or "") or (
        "cold_diffusion" in str(kwargs.get("process_type") or "")
    )


def _degrades_by_mask(doc: dict) -> bool:
    """False when ``training.diffusion.degradation`` names a forward process with no mask ladder."""
    diffusion = (doc.get("training") or {}).get("diffusion")
    return not (isinstance(diffusion, dict) and diffusion.get("degradation") == "rician")


def _undersampling(doc: dict) -> dict:
    return doc.get("undersampling") or doc.get("acceleration") or {}


def phase_encode_axis_of(doc: dict) -> int | None:
    """``data.acquisition.phase_encode_axis``, validated by its schema, or None if undeclared."""
    from spectramr.config.schemas.data_acquisition import DataAcquisitionSchema

    block = (doc.get("data") or {}).get("acquisition")
    return None if block is None else DataAcquisitionSchema(**block).phase_encode_axis


def _matrix(doc: dict) -> tuple[int, int]:
    data = doc.get("data") or {}
    sampling = data.get("sampling") or {}
    patch = sampling.get("patch_size") or data.get("patch_size") or list(_DEFAULT_MATRIX)
    try:
        return int(patch[0]), int(patch[1])
    except (TypeError, ValueError, IndexError):
        return _DEFAULT_MATRIX


def _timesteps(doc: dict) -> int:
    """The horizon the generator is built with, resolved in the builder's order.

    ``training.diffusion.timesteps`` is injected whenever that block exists (the schema
    defaults it), so the ``model_kwargs`` copy only counts on an arm without the block (#2468).
    """
    diffusion = (doc.get("training") or {}).get("diffusion")
    if isinstance(diffusion, dict):
        declared = diffusion.get("timesteps", DEFAULT_DIFFUSION_TIMESTEPS)
        if declared is not None:
            return int(declared)
    declared = ((doc.get("model") or {}).get("model_kwargs") or {}).get("timesteps")
    return DEFAULT_DIFFUSION_TIMESTEPS if declared is None else int(declared)


def build_process_from_config(doc: dict) -> Any | None:
    """The ``KSpaceUndersamplingProcess`` this config actually builds, or None.

    Routed through ``AccelerationConfigSchema`` + ``resolve_undersampling_kwargs``
    exactly like the runtime generator and the ladder CI gate (issue #550:
    reading raw YAML makes the audit blind to keys the schema drops). Returns
    ``None`` when the config is not a cold-diffusion arm, or when its forward
    process is not a mask cascade (a Rician arm has no ladder to certify).
    """
    if not _is_cold_diffusion(doc) or not _degrades_by_mask(doc):
        return None
    from spectramr.config.schemas.acceleration import AccelerationConfigSchema
    from spectramr.models.diffusion.kspace_process import (
        KSpaceUndersamplingProcess,
        resolve_undersampling_kwargs,
    )

    accel = _undersampling(doc)
    resolved = resolve_undersampling_kwargs(
        AccelerationConfigSchema(**accel), phase_encode_axis=phase_encode_axis_of(doc)
    )
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
    "phase_encode_axis_of",
    "synthetic_spectral_prior",
]
