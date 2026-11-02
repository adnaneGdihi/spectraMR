"""§8.3 Tier 3 — channelized-Hotelling-observer (CHO) task concordance.

A good no-reference (NR) metric should track *task performance*: as a
signal-detection task gets harder (more noise, more blur, more motion), both
human readers and a model observer detect the target less reliably. This tier
builds a **signal-known-exactly** lesion-detection task, sweeps degradation
severity, measures task detectability :math:`d'` with a channelized Hotelling
observer (CHO), and checks that each NR metric's trajectory is concordant with
the :math:`d'` trajectory along the severity axis.

Method (signal-known-exactly CHO, Barrett & Myers)
--------------------------------------------------
For each clean reference and each degradation family:

1. Insert a small 2-D Gaussian blob lesion at the image centre to form the
   ``present`` reference; the ``absent`` reference is the clean image
   (see :func:`_insert_blob`).
2. Build a fixed Gabor channel bank for the image shape
   (:func:`~spectramr.infrastructure.physics.model_observer.gabor_channel_bank`).
3. For each severity ``theta`` on ``linspace(low, 1.0, n_severities)``: draw
   ``n_ensemble`` degraded present realisations and ``n_ensemble`` degraded
   absent realisations (each with a distinct seeded generator), flatten each to
   a pixel column, and compute
   :math:`d' = \\sqrt{\\max(0, d'^2)}` via
   :meth:`ChannelizedHotellingObserver.detectability`.
4. Evaluate each NR metric on the *mean* degraded present image versus the
   present reference (one NR value per severity) via :func:`cards.eval_metric`.

Per metric and family we take ``|Spearman rho_S|`` between the NR trajectory and
the :math:`d'` trajectory over the severity axis (direction-agnostic: a metric
that rises *or* falls with detectability is concordant). The headline score is
the mean ``|rho_S|`` across non-degenerate families; it passes when that mean
clears ``rho_threshold``.

This module is part of the offline, research-mode NR validation harness — the NR
metrics are validation-pending and never run during training. Everything is
CPU-canonical and deterministic (every RNG is an explicitly-seeded
``torch.Generator``); no scorer ever raises (failures are recorded as NaN).
"""

from __future__ import annotations

import math

import torch

from spectramr.application.use_cases.nr_validation.cards import (
    TierResult,
    eval_metric,
    spearman,
)
from spectramr.core.metrics.meta_evaluation.simulator import DEGRADATION_LIBRARY, content_seed
from spectramr.infrastructure.physics.model_observer import (
    ChannelizedHotellingObserver,
    gabor_channel_bank,
)

__all__ = ["score_cho_concordance"]

_DEFAULT_FAMILIES = ["rician_noise", "gaussian_blur", "motion"]


def _insert_blob(
    img: torch.Tensor,
    contrast: float,
    radius_frac: float,
    center: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Add a centred 2-D Gaussian blob lesion to a magnitude image.

    The standard signal-known-exactly (SKE) detection signal: a small Gaussian
    bump whose peak amplitude is ``contrast`` times the image dynamic range,
    added to the image magnitude. Kept self-contained on purpose — the TorchIO
    ``SyntheticLesionTransform`` operates on 3-D ``tio.Subject`` objects and is
    the wrong tool for a flat 2-D CHO task.

    Args:
        img: input image, ``[..., H, W]`` (the last two dims are spatial). The
            blob is added to every leading-dim slice identically.
        contrast: blob peak as a fraction of the image's dynamic range
            (``amax - amin``).
        radius_frac: blob Gaussian std as a fraction of ``min(H, W)``.
        center: ``(cy, cx)`` blob centre in pixel coordinates; defaults to the
            geometric image centre ``((H-1)/2, (W-1)/2)``.

    Returns:
        A new tensor of the same shape/dtype as ``img`` with the blob added.
    """
    h, w = int(img.shape[-2]), int(img.shape[-1])
    if center is None:
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    else:
        cy, cx = float(center[0]), float(center[1])
    sigma = max(radius_frac * float(min(h, w)), 1e-6)
    ys = torch.arange(h, dtype=torch.float64) - cy
    xs = torch.arange(w, dtype=torch.float64) - cx
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    gauss = torch.exp(-(yy * yy + xx * xx) / (2.0 * sigma * sigma))
    img_f = img.to(torch.float64)
    dyn = float(img_f.amax().item() - img_f.amin().item())
    if not math.isfinite(dyn) or dyn <= 0.0:
        dyn = max(float(img_f.abs().amax().item()), 1.0)
    peak = float(contrast) * dyn
    blob = (peak * gauss).to(img.dtype)
    # Broadcast the [H, W] blob across any leading dims (channel, etc.).
    return img + blob.view((1,) * (img.ndim - 2) + (h, w))


def _ensemble(
    ref: torch.Tensor,
    family: str,
    theta: float,
    *,
    base_seed: int,
    n_ensemble: int,
) -> torch.Tensor:
    """Build a ``[n_pixels, n_ensemble]`` degraded ensemble of one hypothesis.

    Each column is one degraded realisation of ``ref`` under
    ``DEGRADATION_LIBRARY[family]`` at severity ``theta``, flattened. Every
    realisation gets a distinct, explicitly seeded ``torch.Generator`` derived
    from ``base_seed`` so the ensemble is reproducible and bit-stable.
    """
    fn = DEGRADATION_LIBRARY[family]
    cols: list[torch.Tensor] = []
    for k in range(n_ensemble):
        seed = (base_seed ^ (k * 0x9E3779B1)) & 0xFFFFFFFF
        gen = torch.Generator().manual_seed(seed)
        deg = fn(ref.detach(), float(theta), gen)
        cols.append(deg.reshape(-1).to(torch.float64))
    return torch.stack(cols, dim=1)


def _mean_image(
    ref: torch.Tensor,
    family: str,
    theta: float,
    *,
    base_seed: int,
    n_ensemble: int,
) -> torch.Tensor:
    """Mean degraded realisation of ``ref`` (same shape as ``ref``).

    Used as the NR metric's input image at a given severity — averaging over the
    ensemble gives a stable, low-variance representative of the degradation.
    """
    fn = DEGRADATION_LIBRARY[family]
    acc: torch.Tensor | None = None
    for k in range(n_ensemble):
        seed = (base_seed ^ (k * 0x9E3779B1)) & 0xFFFFFFFF
        gen = torch.Generator().manual_seed(seed)
        deg = fn(ref.detach(), float(theta), gen).to(torch.float64)
        acc = deg if acc is None else acc + deg
    assert acc is not None  # n_ensemble >= 1 enforced by the severity loop
    return (acc / float(n_ensemble)).to(ref.dtype)


def score_cho_concordance(
    clean_volumes: list[tuple[str, torch.Tensor]],
    nr_names: list[str],
    *,
    families: list[str] | None = None,
    n_severities: int = 6,
    n_ensemble: int = 24,
    lesion_contrast: float = 0.25,
    lesion_radius_frac: float = 0.06,
    rho_threshold: float = 0.5,
    seed: int = 0,
) -> dict[str, TierResult]:
    """§8.3 Tier 3 — score NR-metric concordance with CHO task detectability.

    Builds a signal-known-exactly lesion-detection task, sweeps degradation
    severity, measures detectability :math:`d'` with a channelized Hotelling
    observer, and reports per-metric ``|Spearman rho_S|`` between the NR-metric
    trajectory and the :math:`d'` trajectory, averaged across non-degenerate
    families. Never raises — degenerate families are skipped and any failed
    metric/statistic is recorded as NaN.

    Args:
        clean_volumes: list of ``(content_id, tensor)`` clean references. Each
            tensor is ``[H, W]`` or ``[C, H, W]`` (the last two dims spatial);
            CPU, real-valued magnitude.
        nr_names: registered NR metric names to score.
        families: degradation families to sweep (keys of
            ``DEGRADATION_LIBRARY``); defaults to
            ``["rician_noise", "gaussian_blur", "motion"]``.
        n_severities: number of severities on the ``linspace(low, 1.0)`` grid.
        n_ensemble: realisations per hypothesis per severity.
        lesion_contrast: blob peak as a fraction of the image dynamic range.
        lesion_radius_frac: blob Gaussian std as a fraction of ``min(H, W)``.
        rho_threshold: pass threshold on the mean ``|rho_S|`` across families.
        seed: master seed; per-realisation seeds are derived from it.

    Returns:
        ``{metric_name: TierResult}`` keyed by ``nr_names``. Each ``TierResult``
        has ``tier="cho_concordance"``, ``score`` = mean ``|rho_S|`` across
        non-degenerate families (NaN if all degenerate), and ``detail`` with
        ``per_family`` (family -> rho_S), ``n_families``, ``n_severities``, and
        ``mean_dprime``.
    """
    fams = list(families) if families is not None else list(_DEFAULT_FAMILIES)
    # Drop any unknown family rather than crashing — keep only wired degraders.
    fams = [f for f in fams if f in DEGRADATION_LIBRARY]

    n_sev = max(int(n_severities), 2)
    n_ens = max(int(n_ensemble), 2)
    # Avoid theta=0 (no degradation -> nothing to detect against).
    thetas = torch.linspace(1.0 / (n_sev + 1), 1.0, n_sev, dtype=torch.float64).tolist()

    if not clean_volumes or not fams or not nr_names:
        reason = (
            "no clean volumes"
            if not clean_volumes
            else "no known families"
            if not fams
            else "no metrics"
        )
        return {
            name: TierResult("cho_concordance", False, float("nan"), {"reason": reason})
            for name in nr_names
        }

    # per family: d' trajectory over thetas, and per-metric NR trajectory.
    dprime_by_family: dict[str, list[float]] = {f: [] for f in fams}
    nr_by_family: dict[str, dict[str, list[float]]] = {
        f: {name: [] for name in nr_names} for f in fams
    }

    # Cache one Gabor bank per spatial shape (data-independent, fixed templates).
    bank_cache: dict[tuple[int, int], torch.Tensor] = {}

    for ci, (content_id, clean_raw) in enumerate(clean_volumes):
        clean = clean_raw.detach()
        if clean.ndim < 2:
            continue
        h, w = int(clean.shape[-2]), int(clean.shape[-1])
        shape_key = (h, w)
        bank = bank_cache.get(shape_key)
        if bank is None:
            try:
                bank = gabor_channel_bank(shape_key)
            except Exception:
                continue
            bank_cache[shape_key] = bank

        present_ref = _insert_blob(clean, lesion_contrast, lesion_radius_frac)
        absent_ref = clean
        # The simulator's blake2b content seed (seed_scheme blake2b_v1), mixed with ci.
        cseed = (content_seed(seed, content_id) ^ (ci * 0x85EBCA77)) & 0xFFFFFFFF

        for fi, family in enumerate(fams):
            for ti, theta in enumerate(thetas):
                base = (cseed + fi * 7919 + ti * 13) & 0xFFFFFFFF
                # Distinct seed streams for present / absent ensembles.
                present_seed = base
                absent_seed = (base ^ 0xABCDEF01) & 0xFFFFFFFF
                # --- CHO detectability d' at this severity ---
                try:
                    present = _ensemble(
                        present_ref,
                        family,
                        theta,
                        base_seed=present_seed,
                        n_ensemble=n_ens,
                    )
                    absent = _ensemble(
                        absent_ref,
                        family,
                        theta,
                        base_seed=absent_seed,
                        n_ensemble=n_ens,
                    )
                    obs = ChannelizedHotellingObserver(bank)
                    d_prime_sq = obs.detectability(present, absent)
                    d_prime = (
                        math.sqrt(max(0.0, d_prime_sq))
                        if math.isfinite(d_prime_sq)
                        else float("nan")
                    )
                except Exception:
                    d_prime = float("nan")
                dprime_by_family[family].append(d_prime)
                # --- NR metric trajectory at this severity ---
                try:
                    mean_present = _mean_image(
                        present_ref,
                        family,
                        theta,
                        base_seed=present_seed,
                        n_ensemble=n_ens,
                    )
                except Exception:
                    mean_present = None
                for name in nr_names:
                    if mean_present is None:
                        nr_by_family[family][name].append(float("nan"))
                        continue
                    # eval_metric synthesises context from the reference arg.
                    val = eval_metric(name, mean_present, present_ref)
                    nr_by_family[family][name].append(val)

    # Aggregate: per metric, |rho_S(NR_traj, d'_traj)| averaged over families that
    # have a non-degenerate (varying, finite) d' trajectory.
    all_dprimes: list[float] = [
        d for traj in dprime_by_family.values() for d in traj if math.isfinite(d)
    ]
    mean_dprime = float(sum(all_dprimes) / len(all_dprimes)) if all_dprimes else float("nan")

    # A family is usable if its d' trajectory varies (so a rank correlation is
    # meaningful). Reuse this set across all metrics.
    usable_families: list[str] = []
    for family in fams:
        d_traj = dprime_by_family[family]
        finite = [d for d in d_traj if math.isfinite(d)]
        if len(finite) >= 3 and (max(finite) - min(finite)) > 1e-12:
            usable_families.append(family)

    out: dict[str, TierResult] = {}
    for name in nr_names:
        if not usable_families:
            out[name] = TierResult(
                "cho_concordance",
                False,
                float("nan"),
                {"reason": "CHO degenerate"},
            )
            continue
        per_family: dict[str, float] = {}
        abs_rhos: list[float] = []
        for family in usable_families:
            d_traj = dprime_by_family[family]
            nr_traj = nr_by_family[family][name]
            try:
                rho = spearman(nr_traj, d_traj)
            except Exception:
                rho = float("nan")
            if math.isfinite(rho):
                per_family[family] = float(rho)
                abs_rhos.append(abs(float(rho)))
        if not abs_rhos:
            out[name] = TierResult(
                "cho_concordance",
                False,
                float("nan"),
                {
                    "reason": "no finite rho",
                    "per_family": per_family,
                    "n_families": 0,
                    "n_severities": n_sev,
                    "mean_dprime": mean_dprime,
                },
            )
            continue
        score = float(sum(abs_rhos) / len(abs_rhos))
        passed = math.isfinite(score) and score >= float(rho_threshold)
        out[name] = TierResult(
            "cho_concordance",
            passed,
            score,
            {
                "per_family": per_family,
                "n_families": len(per_family),
                "n_severities": n_sev,
                "mean_dprime": mean_dprime,
            },
        )
    return out
