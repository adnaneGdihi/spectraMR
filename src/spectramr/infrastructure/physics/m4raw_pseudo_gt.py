"""Physics-informed pseudo-ground-truth synthesis for multi-rep MRI data.

Moved here from ``scripts/sim2rank/ground_truth.py`` 2026-05-14 per
``TODO/backlog_ssot_and_layering_cleanup.md`` Phase 4 — the
``src/cli/app.py`` meta-evaluation path was importing this from
``scripts/``, a layer-direction violation (CLAUDE.md pitfall #13:
nothing in ``src/`` may import from ``scripts/``).

Exploits the multi-repetition topology of the m4raw dataset to
synthesize an ultra-high SNR pseudo-ground truth by averaging
uncombined complex k-space across all repetitions, then reconstructing
via the SENSE adjoint.

Everything here is a pure tensor (or header-string) function. The file reads live in
:mod:`spectramr.data.datasets.kspace_volume_stack`, and the path-taking 2-D reference
:func:`~spectramr.data.datasets.pseudo_gt_reference.synthesize_pseudo_gt` composes the two
(rule 7).

Reference:
    Macovski (1996), "Noise in MRI", MRM — complex k-space averaging
    increases baseline SNR by sqrt(N_reps) prior to the non-linear
    magnitude operation.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _select_consistent_reps(
    slices: list[torch.Tensor], paths: list[Path]
) -> tuple[list[torch.Tensor], list[Path]]:
    """Keep the largest coil/matrix-consistent subset of per-rep slices.

    Genuine NEX repetitions share one receive array, so their selected
    ``(Coils, H, W)`` slices are identically shaped. A group whose reps disagree
    on that shape is not a valid NEX set — coil ``c`` of a 20-coil acquisition is
    a different physical element than coil ``c`` of a 16-coil one, so the two
    cannot be complex-averaged. Rather than crash ``torch.stack`` (the cluster's
    ``stack expects each tensor to be equal size`` error) or silently average
    incompatible reps (a facade, CLAUDE.md #16), keep the reps matching the
    winning shape and name the dropped files at WARNING.

    The winner is the shape carried by the most reps (highest √N SNR); ties break
    toward more coils, then more rows/cols, for a deterministic choice.
    """
    signatures = [tuple(s.shape) for s in slices]
    if len(set(signatures)) <= 1:
        return slices, paths

    counts = Counter(signatures)
    winner = max(counts, key=lambda sig: (counts[sig], sig[0], sig[1], sig[2]))

    kept_slices: list[torch.Tensor] = []
    kept_paths: list[Path] = []
    dropped: list[tuple[Path, tuple[int, ...]]] = []
    for sl, p, sig in zip(slices, paths, signatures, strict=True):
        if sig == winner:
            kept_slices.append(sl)
            kept_paths.append(p)
        else:
            dropped.append((p, sig))

    logger.warning(
        "Repetition group has heterogeneous coil/matrix shapes — not a valid NEX "
        "set. Averaging only the %d rep(s) at (Coils, H, W)=%s; dropping %d "
        "mismatched rep(s): %s. NEX averaging requires identical coil geometry, so "
        "reps with a different coil count or matrix size cannot be complex-averaged.",
        len(kept_slices),
        tuple(winner),
        len(dropped),
        ", ".join(f"{p.name}={sig}" for p, sig in dropped),
    )
    return kept_slices, kept_paths


#: ISMRMRD tags carrying acquisition parameters, and the ``MetricContext.acq_params``
#: key each maps to. Times are milliseconds, matching what ``brc`` / ``qvcr`` /
#: ``phys_residual_consistency`` document as their unit.
_ACQ_PARAM_TAGS: dict[str, str] = {
    "TR": "TR",
    "TE": "TE",
    "TI": "TI",
    "flipAngle_deg": "flip_angle",
    "systemFieldStrength_T": "field_strength_T",
    "sliceThickness": "slice_thickness_mm",
    "spacingBetweenSlices": "slice_spacing_mm",
    "echo_train_length": "echo_train_length",
    "echoTrainLength": "echo_train_length",
    "echo_spacing": "echo_spacing",
    "sequence_type": "sequence_type",
    "protocolName": "protocol_name",
}


def _echo_spacing_ms(value: object) -> dict[str, float]:
    """M4Raw writes ``16.3ms`` on its TSE series and ``N/A`` on its single-echo SE T1."""
    if isinstance(value, float):
        return {"echo_spacing": value}
    text = str(value).strip()
    if text.upper() == "N/A":
        return {}
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*ms", text)
    if match is None:
        raise ValueError(f"echo_spacing {text!r} is neither milliseconds nor N/A")
    return {"echo_spacing": float(match.group(1))}


def _encoded_triple(xml: str, block: str) -> tuple[float, float, float] | None:
    """``<block>`` x/y/z inside ``encodedSpace``; ``reconSpace`` repeats the tag names."""
    space = re.search(r"<(?:\w+:)?encodedSpace>(.*?)</(?:\w+:)?encodedSpace>", xml, re.S)
    if space is None:
        return None
    inner = re.search(rf"<(?:\w+:)?{block}>(.*?)</(?:\w+:)?{block}>", space.group(1), re.S)
    if inner is None:
        return None
    axes = [re.search(rf"<(?:\w+:)?{a}>([^<]+)</", inner.group(1)) for a in "xyz"]
    if any(m is None for m in axes):
        return None
    return tuple(float(m.group(1)) for m in axes)  # type: ignore[union-attr]


def parse_acquisition_header(xml: str) -> dict[str, object]:
    """Acquisition parameters from an ISMRMRD header string (M4Raw and fastMRI).

    The data has always carried these; nothing read them. ``brc`` and ``qvcr`` reported
    ``needs acq_params`` and ``phys_residual_consistency`` raised *"requires
    ``acquisition`` (TR/TE/TI in ms)"*, on files whose header states
    ``TR=7500, TE=98, TI=1655, flipAngle=160 deg`` outright (#606).

    Numeric tags are returned as floats and string tags verbatim; a tag that is absent
    is simply omitted rather than defaulted, so a downstream metric sees "not supplied"
    rather than a fabricated value. The file read lives in
    :func:`spectramr.data.datasets.kspace_volume_stack.read_acquisition_params`.
    """
    out: dict[str, object] = {}
    for tag, key in _ACQ_PARAM_TAGS.items():
        # Namespace-agnostic: M4Raw writes ``<ns0:TR>``, other vendors write ``<TR>``.
        match = re.search(rf"<(?:\w+:)?{tag}>([^<]+)</(?:\w+:)?{tag}>", xml)
        if match is None:
            continue
        text = match.group(1).strip()
        try:
            out[key] = float(text)
        except ValueError:
            out[key] = text
    if "echo_spacing" in out:
        out.update(_echo_spacing_ms(out.pop("echo_spacing")))
    matrix = _encoded_triple(xml, "matrixSize")
    if matrix is not None:
        out["encoded_matrix"] = (int(matrix[0]), int(matrix[1]))
    fov = _encoded_triple(xml, "fieldOfView_mm")
    if fov is not None:
        out["fov_mm"] = fov
    return out


def nex_merge(
    reps: list[torch.Tensor], labels: list[Path]
) -> tuple[torch.Tensor, list[Path]]:
    """Complex k-space mean over the largest shape-consistent subset of ``reps``.

    Each rep is ``(Coils, H, W)`` (one slice) or ``(Slices, Coils, H, W)`` (a volume);
    the mean keeps that shape. Returns ``(mean, labels of the reps kept)``.
    """
    kept, kept_labels = _select_consistent_reps(reps, labels)
    n_reps = len(kept)
    logger.info(
        "Synthesizing pseudo-GT from %d repetitions (SNR boost √%d = %.2f)",
        n_reps,
        n_reps,
        n_reps**0.5,
    )
    return torch.stack(kept, dim=0).mean(dim=0), kept_labels


def pseudo_gt_from_kspace(
    y_gt: torch.Tensor, smaps: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Pseudo ground truth from NEX-merged, fully sampled k-space ``(B, Coils, H, W)``.

    Multi-repetition merge is **NEX averaging in complex k-space** (Macovski 1996,
    *Noise in MRI*): averaging :math:`N` complex repetitions before the magnitude
    operation gives the full :math:`\sqrt{N}` SNR boost. The coil maps are the
    canonical 7-step **ESPIRiT** (Uecker 2014) estimated from that merged k-space.

    .. math::

        \hat{x}_{GT} = A^H(Y_{GT}, S) = \sum_c S_c^* \cdot \text{IFFT}(Y_{GT,c})

    ``B`` is one slice for the 2-D reference and every slice of a volume for the
    volume stack; one robust 99th percentile normalises the whole batch, so the
    slices of a volume keep their relative intensity.

    Args:
        y_gt: Complex k-space, ``(B, Coils, H, W)``.
        smaps: Optional pre-estimated maps of ``y_gt``'s shape to **reuse** instead of
            re-estimating, so a single-rep *degraded* acquisition shares the NEX-merged
            reference's coil operator (sim->real transfer).

    Returns:
        ``(coil_images, smaps, x_gt_mag, p99)``: complex image-domain coil images
        ``(B, Coils, H, W)`` (NOT k-space), the maps, the SENSE-adjoint magnitude
        ``(B, 1, H, W)`` divided by ``p99``, and the raw ``p99`` scalar.
    """
    from spectramr.infrastructure.physics.coil_sensitivity import (
        coil_combine_sense,
        espirit_min_acs_size,
        estimate_csm_espirit,
    )
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    if y_gt.dim() != 4 or not y_gt.is_complex():
        raise ValueError(f"y_gt must be complex (B, Coils, H, W); got {tuple(y_gt.shape)}")
    n_coils = y_gt.shape[1]
    logger.info("Pseudo-GT k-space: shape=%s, n_coils=%d", tuple(y_gt.shape), n_coils)

    if smaps is not None:
        # Reuse caller-provided maps (the NEX-merged reference's ESPIRiT maps
        # applied to a single-rep degraded acquisition) — one shared coil operator.
        if tuple(smaps.shape) != tuple(y_gt.shape):
            raise ValueError(
                f"provided smaps shape {tuple(smaps.shape)} does not match the "
                f"reconstructed k-space {tuple(y_gt.shape)}"
            )
        smaps = smaps.to(device=y_gt.device, dtype=y_gt.dtype)
    elif n_coils > 1:
        # The pseudo-GT k-space is fully sampled (all reps merged), so the central
        # ACS is dense calibration data we may safely enlarge. With many coils the
        # historical 24x24 ACS is rank-deficient (n_patches < kernel^2 * coils) and
        # ESPIRiT would silently fall back to RSS, yielding a materially different
        # x_gt (#309). Grow the ACS only when 24 would fail; low-coil references
        # (incl. 4-coil M4Raw) keep acs_size=24 so their pseudo-GT is unchanged.
        kernel_size = 6
        max_acs = min(int(y_gt.shape[-2]), int(y_gt.shape[-1]))
        acs_size = min(24, max_acs)
        if (acs_size - kernel_size + 1) ** 2 < kernel_size * kernel_size * n_coils:
            acs_size = min(espirit_min_acs_size(n_coils, kernel_size=kernel_size), max_acs)
        try:
            # No global linalg-backend pin: the per-pixel batched-complex eigh
            # in estimate_csm_espirit is made cuSOLVER-resilient by
            # coil_sensitivity._robust_eigh (CPU-LAPACK fallback). Pinning a
            # process-wide backend here was the original failing "fix".
            smaps = estimate_csm_espirit(
                y_gt,
                num_coils=n_coils,
                kernel_size=kernel_size,
                acs_size=acs_size,
                sigma_threshold=0.02,
                eigen_threshold=0.95,
            )
        except torch.cuda.OutOfMemoryError as exc:
            # Keep the ESTIMATOR, change the DEVICE. Switching to RSS here was
            # the real defect (#521): it is a *different ground truth*, so a
            # cohort where some volumes OOM'd and others did not carries two
            # incompatible x_gt definitions, and every downstream ADR / SCVR /
            # BT number silently mixes them. Nothing in the artifacts records
            # which volume took which path, so the contamination is not even
            # recoverable after the fact.
            #
            # The OOM is not ESPIRiT being extravagant. On the 2026-07-25 brain
            # run the biggest volume is (1, 20, 768, 396): the per-pixel Gram is
            # 304k x 20 x 20 complex64 ~= 1.9 GiB, and the log shows 23-27 GiB
            # already held by the sweep with only 0.1-1.0 GiB free and 3.6-7.4
            # GiB reserved-but-unallocated. ESPIRiT is the victim of
            # fragmentation, not the cause, so retrying the same computation on
            # host RAM keeps the contract at no accuracy cost.
            #
            # This mirrors coil_sensitivity._robust_eigh, which already does a
            # logged device->host retry for the cuSOLVER workspace bug in this
            # same code path. It is a documented fallback, not the silent CPU
            # degradation non-negotiable 9b forbids.
            free_gib = total_gib = float("nan")
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                free_gib, total_gib = free_b / 2**30, total_b / 2**30
            logger.warning(
                "ESPIRiT ran out of CUDA memory on a %s volume (%.2f of %.2f "
                "GiB free) — retrying the SAME estimator on CPU rather than "
                "substituting RSS, so this volume's x_gt stays comparable with "
                "the rest of the cohort. Original error: %s",
                tuple(y_gt.shape),
                free_gib,
                total_gib,
                exc,
            )
            # Release the blocks the failed attempt reserved before allocating
            # the host copy. Not a training loop — non-negotiable 9's
            # no-empty_cache rule is about the hot path, and this is one-off
            # OOM recovery, which is exactly what empty_cache is for.
            torch.cuda.empty_cache()
            smaps = estimate_csm_espirit(
                y_gt.cpu(),
                num_coils=n_coils,
                kernel_size=kernel_size,
                acs_size=acs_size,
                sigma_threshold=0.02,
                eigen_threshold=0.95,
            ).to(y_gt.device)
        except Exception as exc:  # explicit, logged fallback to RSS
            logger.error(
                "ESPIRiT coil-map estimation FAILED (%s: %s) — pseudo-GT is "
                "falling back to RSS coil maps, which contradicts the documented "
                "ESPIRiT contract and yields a materially different x_gt. "
                "Downstream sim2rank/BT comparisons used coil_estimation='rss', "
                "NOT 'espirit'. A cohort that mixes the two has two ground "
                "truths and its leaderboard is not interpretable (#521).",
                type(exc).__name__,
                exc,
            )
            from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_rss

            smaps = estimate_csm_rss(y_gt, num_coils=n_coils)
    else:
        smaps = torch.ones_like(y_gt)

    coil_images = ifft2c(y_gt)  # (B, Coils, H, W)
    # (B, 1, H, W); a single coil is its own combination.
    x_gt_complex = coil_combine_sense(coil_images, smaps) if n_coils > 1 else coil_images

    x_gt_mag = x_gt_complex.abs().float()
    p99 = torch.quantile(x_gt_mag.flatten(), 0.99).clamp(min=1e-8)
    x_gt_mag = x_gt_mag / p99

    logger.info(
        "Pseudo-GT image: shape=%s, range=[%.4f, %.4f], raw_p99=%.4f",
        x_gt_mag.shape,
        x_gt_mag.min().item(),
        x_gt_mag.max().item(),
        p99.item(),
    )

    # Return image-domain (NOT k-space) — see scripts/sim2rank docstring:
    # DigitalTwinSimulator expects image-domain input and does fft2c
    # internally; returning k-space caused a double-FFT white-out bug.
    return coil_images, smaps, x_gt_mag, p99


__all__ = [
    "nex_merge",
    "parse_acquisition_header",
    "pseudo_gt_from_kspace",
]
