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

import h5py
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


def read_acquisition_params(path: Path) -> dict[str, object]:
    """Acquisition parameters from an M4Raw file's ``ismrmrd_header``.

    The data has always carried these; nothing read them. ``brc`` and ``qvcr`` reported
    ``needs acq_params`` and ``phys_residual_consistency`` raised *"requires
    ``acquisition`` (TR/TE/TI in ms)"*, on files whose header states
    ``TR=7500, TE=98, TI=1655, flipAngle=160 deg`` outright — three metrics declared
    inapplicable for want of a parser (#606).

    Numeric tags are returned as floats and string tags verbatim; a tag that is absent
    is simply omitted rather than defaulted, so a downstream metric sees "not supplied"
    rather than a fabricated value.

    Returns:
        ``{}`` when the file has no header or no recognised tag — the honest empty,
        which keeps the metric ``not_applicable`` instead of feeding it invented physics.
    """
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")
    with h5py.File(str(path), "r") as f:
        if "ismrmrd_header" not in f:
            return {}
        raw = f["ismrmrd_header"][()]
    xml = raw.decode() if isinstance(raw, bytes | bytearray) else str(raw)

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


def _kspace_shape_from_h5(path: Path) -> tuple[int, ...]:
    """The ``kspace`` dataset's shape, read from the HDF5 header only.

    No voxel IO: h5py exposes ``.shape`` from the dataset metadata. Used to pick
    the target slice before deciding what to read, so the reps are never fully
    materialised just to learn how many slices they have (#616).
    """
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")
    with h5py.File(str(path), "r") as f:
        if "kspace" not in f:
            raise KeyError(f"No 'kspace' dataset in {path}. Keys: {list(f.keys())}")
        return tuple(int(d) for d in f["kspace"].shape)


def _load_kspace_from_h5(path: Path, slice_index: int | None = None) -> torch.Tensor:
    """Load complex k-space tensor from an M4Raw H5 file.

    Mirrors the proven logic in
    :class:`~spectramr.data.datasets.m4raw_dataset.M4RawRepetitionDataset`.

    Args:
        path: H5 file.
        slice_index: When given, read ONLY that slice through h5py's partial
            read instead of materialising the whole volume. Pseudo-GT synthesis
            averages one slice across every repetition, so reading all of them
            held N whole multi-slice volumes to use N single slices (#616).

    Returns:
        Complex float32 tensor, ``(Slices, Coils, H, W)`` — or that shape with
        the leading axis dropped when *slice_index* is given.
    """
    if not path.exists():
        raise FileNotFoundError(f"H5 file not found: {path}")

    with h5py.File(str(path), "r") as f:
        if "kspace" not in f:
            raise KeyError(f"No 'kspace' dataset in {path}. Keys: {list(f.keys())}")
        raw = f["kspace"][()] if slice_index is None else f["kspace"][slice_index]

    t = torch.from_numpy(raw)
    if not torch.is_complex(t):
        if t.shape[-1] == 2:
            t = torch.view_as_complex(t.contiguous().float())
        else:
            t = t.float()
    else:
        t = t.to(torch.complex64)
    return t


def extract_contrast_label(path: Path) -> str:
    """Extract an MRI contrast label from an acquisition filename.

    Handles both naming schemes we ingest:

    - **fastMRI brain** — ``file_brain_AXT2FLAIR_201_6002670`` → ``AXT2FLAIR``
      (the ``AX*`` acquisition token; also ``AXT1``/``AXT2``/``AXFLAIR``/
      ``AXT1POST``/``AXT1PRE``). Without this the M4Raw pattern below silently
      fell through to ``return stem`` and mislabelled every brain volume with
      its full filename (issue #307).
    - **M4Raw** — ``2022091411_T101`` → ``T1``, ``..._T201`` → ``T2``,
      ``..._FLAIR01`` → ``FLAIR``. The contrast token is everything after the
      last ``_`` minus the trailing 2-digit repetition index (matching
      ``M4RawSource._group_id``'s ``stem[:-2]``). The old letters-only pattern
      captured just the leading letters, so ``T1`` and ``T2`` both collapsed to
      ``T`` and ``available_contrasts`` conflated them (issue #307).
    """
    stem = path.stem
    # fastMRI brain acquisition token: ``_AX<...>_<digits>`` (T1/T2/FLAIR/POST/PRE).
    fastmri = re.search(r"_(AX[A-Za-z0-9]+?)_\d", stem)
    if fastmri:
        return fastmri.group(1).upper()
    # M4Raw: <contrast><field?> then the 2-digit repetition suffix. Keep the
    # field digit (T1 vs T2); strip only the rep index.
    m4raw = re.search(r"_([A-Za-z]+\d*?)\d{2}$", stem)
    if m4raw:
        return m4raw.group(1).upper()
    logger.warning("Could not extract contrast from %r, using stem.", stem)
    return stem


def synthesize_pseudo_gt(
    rep_paths: list[Path],
    target_slice: int | None = None,
    device: torch.device = torch.device("cpu"),
    smaps: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Synthesize ultra-high SNR pseudo-ground truth from multi-repetition data.

    Multi-repetition merge is **NEX averaging in complex k-space** (Macovski 1996,
    *Noise in MRI*): averaging :math:`N` complex repetitions before the magnitude
    operation gives the full :math:`\sqrt{N}` SNR boost (magnitude-domain averaging
    would not, because of the Rician noise floor). The coil maps are the canonical
    7-step **ESPIRiT** (Uecker 2014) estimated from that high-SNR merged k-space.

    .. math::

        Y_{GT} = \frac{1}{N_{\text{reps}}} \sum_{i=1}^{N_{\text{reps}}} Y_i
        \qquad
        \hat{x}_{GT} = A^H(Y_{GT}, S) = \sum_c S_c^* \cdot \text{IFFT}(Y_{GT,c})

    Args:
        rep_paths: Sorted list of H5 file paths for one anatomy group
            (all repetitions of the same contrast/subject).
        target_slice: If provided, extract only this slice index.
            Otherwise uses the central slice.
        device: Target device for output tensors.
        smaps: Optional pre-estimated ESPIRiT maps ``(1, C_coils, H, W)`` to
            **reuse** instead of re-estimating. The intended use is sim->real
            transfer: estimate ESPIRiT once from the NEX-merged multi-rep
            reference, then pass those maps when reconstructing each single-rep
            *degraded* acquisition so reference and degraded share ONE coil
            operator. The comparison then isolates the noise/averaging
            degradation rather than a per-acquisition recon-basis shift, and a
            single noisy rep never drives its own (poor) map estimate.

    Returns:
        ``(coil_images, smaps, x_gt_mag, p99)``:
            - ``coil_images`` — complex ``(1, C_coils, H, W)`` image-domain
              tensor (NOT k-space — DigitalTwinSimulator expects image domain).
            - ``smaps`` — ESPIRiT coil sensitivity maps ``(1, C_coils, H, W)``
              (the reused maps verbatim when ``smaps`` was provided).
            - ``x_gt_mag`` — SENSE-adjoint magnitude image ``(1, 1, H, W)``,
              normalised to ``[0, 1]`` by robust 99th-percentile.
            - ``p99`` — the raw 99th-percentile scalar (so callers can
              normalise degraded images consistently).
    """
    from spectramr.infrastructure.physics.coil_sensitivity import (
        coil_combine_sense,
        espirit_min_acs_size,
        estimate_csm_espirit,
    )
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    # Two cheap passes instead of one expensive one (#616). Pseudo-GT averages
    # ONE slice across every repetition, but the target slice is derived from
    # the slice count — so the old single pass loaded every rep in full to learn
    # a number, held them all resident, and then used one slice each. For an
    # M4Raw multi-slice multi-coil acquisition that is the whole volume per rep.
    #
    # Pass 1 reads shapes from the HDF5 header (no voxel IO). Pass 2 reads only
    # the chosen slice. The per-file error handling is unchanged and now covers
    # both passes, so a rep that fails either way is still named rather than
    # crashing the run.
    errors: list[str] = []
    shapes: dict[Path, tuple[int, ...]] = {}
    for p in rep_paths:
        try:
            shapes[p] = _kspace_shape_from_h5(p)
        except FileNotFoundError as exc:
            errors.append(f"{p.name}: {exc}")
            logger.warning("File not found: %s", p)
        except KeyError as exc:
            errors.append(f"{p.name}: {exc}")
            logger.warning("No kspace dataset in %s: %s", p, exc)
        except Exception as exc:  # defensive: surfaced via errors[]
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
            logger.warning("Failed to probe %s: %s: %s", p, type(exc).__name__, exc)

    if not shapes:
        raise RuntimeError(
            f"All {len(rep_paths)} repetition files failed to load.\n"
            f"Errors:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    # The slice count comes from the first rep that probed, matching the old
    # "first successfully loaded rep" semantics.
    n_slices = next(iter(shapes.values()))[0]
    if target_slice is None:
        target_slice = n_slices // 2
    target_slice = min(target_slice, n_slices - 1)

    slices: list[torch.Tensor] = []
    loaded_paths: list[Path] = []
    for p, shape in shapes.items():
        try:
            ks = _load_kspace_from_h5(p, slice_index=min(target_slice, shape[0] - 1))
        except Exception as exc:  # defensive: surfaced via errors[]
            errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
            logger.warning("Failed to load %s: %s: %s", p, type(exc).__name__, exc)
            continue
        # The leading (slice) axis is already gone: 4-D -> (Coils, H, W),
        # 3-D single-coil -> (H, W) which still needs a coil axis.
        if ks.dim() == 3:
            slices.append(ks)
        elif ks.dim() == 2:
            slices.append(ks.unsqueeze(0))
        else:
            raise ValueError(f"Unexpected kspace slice shape: {ks.shape}")
        loaded_paths.append(p)
        logger.debug("Loaded rep %s slice %d: shape=%s", p.name, target_slice, ks.shape)

    if not slices:
        raise RuntimeError(
            f"All {len(rep_paths)} repetition files failed to load.\n"
            f"Errors:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    # NEX averaging is only valid across reps that share one receive array. Reps
    # whose selected slice differs in (Coils, H, W) are not a valid NEX set (coil
    # c of a 20-coil scan is a different physical element than coil c of a 16-coil
    # scan), so keep the largest consistent subset and name the drops at WARNING
    # rather than crash torch.stack or silently average incompatible reps (#16).
    slices, _ = _select_consistent_reps(slices, loaded_paths)

    n_reps = len(slices)
    logger.info(
        "Synthesizing pseudo-GT from %d repetitions (SNR boost √%d = %.2f)",
        n_reps,
        n_reps,
        n_reps**0.5,
    )

    stacked = torch.stack(slices, dim=0)  # (N_reps, Coils, H, W)
    y_gt = stacked.mean(dim=0, keepdim=False).unsqueeze(0).to(device)  # (1, Coils, H, W)

    n_coils = y_gt.shape[1]
    logger.info(
        "Pseudo-GT k-space: shape=%s, n_coils=%d, slice=%d/%d",
        y_gt.shape,
        n_coils,
        target_slice,
        n_slices,
    )

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

    coil_images = ifft2c(y_gt)  # (1, Coils, H, W)
    if n_coils > 1:
        x_gt_complex = coil_combine_sense(coil_images, smaps)  # (1, 1, H, W)
    else:
        x_gt_complex = coil_images

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
    "_load_kspace_from_h5",
    "extract_contrast_label",
    "read_acquisition_params",
    "synthesize_pseudo_gt",
]
