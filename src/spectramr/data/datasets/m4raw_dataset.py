"""M4Raw Repetition-Aware Dataset.

Groups multi-repetition H5 files by base filename pattern (files differ only
in the last 2 digits of the stem, directly concatenated, e.g. ``01``, ``02``, ``03``).

Contrast-specific repetition counts (M4Raw, confirmed 2026-08-17 — issue #1172):

- T1: 3 repetitions
- T2: 3 repetitions
- FLAIR: 2 repetitions
- PD: variable

These numbers are load-bearing, not trivia. ``nex_target_exclude_input``
(leave-one-out NEX target, so the target's noise is independent of the input
rep's) is gated at ``len(kspace_reps) >= 3``, because excluding one of two reps
leaves a single-rep "average" with no SNR gain at all. So LOO is **available for
T1/T2 and structurally impossible for FLAIR** — a FLAIR target necessarily
contains the input rep's own noise at ``1/N``. The NEX SNR gain is likewise
``sqrt(3)`` / ``sqrt(2)``, not ``sqrt(4)`` or ``sqrt(6)``.

This docstring previously claimed 6/6/4, contradicting the ``#695`` comment in
``__init__`` below (which had it right). Anything quoting the old figures is
wrong by a factor of ~2.

For each anatomy group the dataset:
- Loads k-space for every repetition file.
- Averages across reps → high-SNR **target**.
- Uses the first repetition as the **input** (thermal-noise limited).
- Injects a ``contrast_idx`` long tensor into the TorchIO Subject metadata.

The subject format is compatible with the existing TorchIO Queue pipeline
and ``ImageCollateStrategy`` collation.

Slice-level route (``data.slice_level_records``, #1757)
--------------------------------------------------------
By default one record is one (patient, contrast) group and ``_load_item`` reads
every repetition in full to serve ``samples_per_volume`` patches of one slice:
``slices / samples_per_volume`` reads per served slice, times the repetition
count, every epoch. With the knob on, :mod:`.m4raw_slice_records` expands each
group into one record per slice from a header-only shape read, ``__len__`` is
the slice count, and ``_load_item`` reads ``f["kspace"][slice]`` for every
repetition of that record, so the subject it yields has depth 1. The NEX
average is per slice already, so the target path is unchanged. A self-calibrated
``Sigma_n`` stays a property of the scan: a record reads the input repetition's
central slices (:func:`sigma_fit_window`) to fit it, never its own slice. The
k-space scale does follow the record, so it is per slice. The queue is
not bypassed: ``samples_per_volume: 1`` with a depth-1 patch serves every slice
exactly once per epoch (see the module docstring of ``m4raw_slice_records`` for
why any other value double-samples).

Usage
-----
Build directly or via :class:`spectramr.infrastructure.builders.directors.data_pipeline_director.DataPipelineDirector`
when ``data.use_repetitions: true`` is set in the training config.
"""

import contextlib
import logging
import pickle
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.core.coil_map_names import REFERENCE_MAPS_KEY
from spectramr.data.datasets.h5_handle_cache import open_h5
from spectramr.data.datasets.m4raw_coil_map_recipe import DEFAULT_RECIPE
from spectramr.data.datasets.m4raw_identity import (
    parse_m4raw_file_id,
    repetition_group_key,
)
from spectramr.data.datasets.m4raw_input_repetition import (
    InputRepetitionDraw,
    validate_input_repetition,
)
from spectramr.data.datasets.m4raw_nex_shift_store import NexShiftStore
from spectramr.data.datasets.m4raw_noise_covariance_store import NoiseCovarianceStore
from spectramr.data.datasets.m4raw_registration import (
    VALID_NEX_REGISTRATIONS,
    group_shifts,
    registration_window,
    shift_kspace,
)
from spectramr.data.datasets.m4raw_slice_records import expand_to_slice_records, retry_step
from spectramr.data.io_strategies import read_h5_kspace
from spectramr.data.transforms.normalization import (
    MEASURED_INPUT_KEY,
    consumes_measured_input,
)

logger = logging.getLogger(__name__)

# One-time OpenMP thread-count clamp. ``torch.set_num_threads(1)`` prevents
# OpenMP/DataLoader deadlocks when FFT ops run in workers, but it is a
# process-global call — running it on every ``_load_item`` was pure per-item
# overhead. A module-level flag makes it fire once per process (main + each
# forked/forkserver worker imports the module fresh, so each worker clamps
# once on its first load).
_NUM_THREADS_CLAMPED = False


def _clamp_worker_threads_once() -> None:
    global _NUM_THREADS_CLAMPED
    if _NUM_THREADS_CLAMPED:
        return
    # PyTorch build without OpenMP support has no set_num_threads.
    with contextlib.suppress(AttributeError):
        torch.set_num_threads(1)
    _NUM_THREADS_CLAMPED = True


# ---------------------------------------------------------------------------
# Contrast mapping (keyword → integer class index)
# ---------------------------------------------------------------------------
CONTRAST_MAP: dict[str, int] = {
    "T1": 0,
    "T2": 1,
    "FLAIR": 2,
    "PD": 3,
}

# ---------------------------------------------------------------------------
# Repetition counts per contrast (the 3/3/2 fact, as data rather than prose)
# ---------------------------------------------------------------------------
#: Contrast -> the number of repetitions M4Raw actually ships (#1172, confirmed
#: 2026-08-17). Until this map existed the fact lived ONLY in prose -- this
#: module's docstring and the ``_MIN_REPS_FOR_LOO`` comment below -- in no form
#: a checker could consume. That is why ``model_kwargs.num_repetitions`` was
#: validated against nothing and four arms shipped an unsatisfiable ``4``
#: (#1173).
#:
#: ``CONTRAST_MAP`` above is NOT this map and must never be substituted for it:
#: its values are integer *class indices*. ``"FLAIR": 2`` there agrees with
#: FLAIR's true repetition count by coincidence alone -- the same lookup gives
#: T1 = 0 and T2 = 1 repetitions, which is nonsense. The collision is precisely
#: the kind that makes a wrong fixture look confirmed.
#:
#: PD is deliberately ABSENT rather than guessed: the module docstring records
#: it as "variable", so no single literal is correct for it. Consumers must read
#: a missing key as "unknown -- cannot validate" and skip, never as zero and
#: never as a default (CLAUDE.md non-negotiable 3: absent is a state to report,
#: not a state to infer).
#:
#: Authority note: at run time the loader's discovered ``len(kspace_reps)`` is
#: what the model actually receives. This map is the *declared* expectation used
#: for pre-flight validation (``spectramr audit``), which runs before any data is
#: opened. A divergence between the two is a finding to report, not something
#: either side should silently absorb.
M4RAW_REPETITIONS_BY_CONTRAST: dict[str, int] = {
    "T1": 3,
    "T2": 3,
    "FLAIR": 2,
}

# Valid coil-processing modes. NN#3 / pitfall #9: an unrecognised mode string
# (e.g. 'rss_kspace', 'magnitude', 'RSS') must raise at __init__ time, never
# silently fall through to RSS-combine. Module-level for pickle/multiprocessing
# worker safety.
_VALID_COIL_MODES: frozenset[str] = frozenset({"none", "rss", "svd", "flatten"})

#: How the cross-contrast route hands the source contrast to the model.
#: ``stacked`` concatenates it onto the target along the channel axis;
#: ``separate`` serves the target alone and the source as ``kspace_source``,
#: so nothing downstream can degrade, overwrite or score the source (#2452).
#: ``target_only`` serves the same target and drops the source: the no-prior
#: control, on exactly the treated arm's pairs, targets and normalization.
VALID_SOURCE_LAYOUTS: frozenset[str] = frozenset({"stacked", "separate", "target_only"})

# Recognized but NOT acted on by this path: only ``rss`` is applied downstream
# (``__getitem__``), so ``svd`` would silently pass raw multi-coil k-space through
# uncompressed (facade, CLAUDE.md pitfall #16 / #15b). SVD virtual-coil
# compression is wired only on the UniversalMRIDataset path. ``flatten`` is a
# no-op alias of ``none`` here (coils are interleaved to real/imag channels
# either way), so it stays accepted.
_UNIMPLEMENTED_COIL_MODES: frozenset[str] = frozenset({"svd"})

# NEX target averaging. ``complex_mean`` (legacy) plain-averages complex k-space;
# because M4Raw reps are separate acquisitions with global phase drift it CANCELS
# signal (SNR below a single rep). ``phase_aligned_mean`` corrects each rep's
# global phase to rep0 first, recovering the coherent sqrt(N) gain.
_VALID_TARGET_MODES: frozenset[str] = frozenset(
    {"complex_mean", "phase_aligned_mean", "rep_pair", "r2r", "nullspace_r2r"}
)

#: Modes that build BOTH halves from the input repetition, mapped to the method
#: that builds them. They never average, never leave anything out, and are
#: served on a one-repetition group; one set, so a guard written for r2r cannot
#: silently miss its sibling.
_SPLIT_METHODS: dict[str, str] = {"r2r": "_recorrupt_r2r", "nullspace_r2r": "_nullspace_split"}
_SINGLE_REP_SPLIT_MODES: frozenset[str] = frozenset(_SPLIT_METHODS)

#: How a Noise2Noise strategy forms its training target from the published
#: repetition stack (``data.n2n_pairing``), and the training ``target_mode`` each
#: one requires. ``ordered`` keeps a split mode's two halves in order, since at a
#: split angle other than 45 degrees they carry different noise variances.
N2N_PAIRINGS: dict[str, frozenset[str]] = {
    "random_pair": frozenset({"rep_pair"}),
    "one_vs_rest_mean": frozenset({"phase_aligned_mean"}),
    "ordered": frozenset({"r2r", "nullspace_r2r"}),
}

#: The subject key holding the repetition stack: slot 0 is the served input,
#: slots 1.. the repetitions (or the split's target half) a pairing draws from.
REPETITION_STACK_KEY = "kspace_reps"

#: Modes whose target is an average of repetitions; the only ones under which
#: ``emit_alternate_input`` publishes a second repetition.
_AVERAGING_TARGET_MODES: frozenset[str] = frozenset({"complex_mean", "phase_aligned_mean"})

#: Where r2r draws its Sigma_n. 'committed' is the matrix measured on one M4Raw
#: study series; 'self_calibrated' fits one per scan from its own coil null
#: space, which needs no second repetition and so costs the arm no excitation.
_VALID_R2R_COVARIANCE_SOURCES: frozenset[str] = frozenset({"committed", "self_calibrated"})

#: Central slices the per-scan Sigma_n is fitted on. Sigma_n is a receive-chain
#: property, one slice already offers ~65,000 null-space observations (~1%
#: error), and ESPIRiT costs ~2.3 s per 256x256 slice in a one-thread loader
#: worker -- 42 s for an 18-slice volume, paid per group per epoch because the
#: Queue respawns its workers and their caches with them.
_SIGMA_FIT_SLICES: int = 2
#: ``rep_pair`` needs one input repetition and one other; the averaged modes need
#: two others so leave-one-out still averages (``_MIN_REPS_FOR_LOO``).
_MIN_REPS_FOR_REP_PAIR: int = 2


def _min_reps_for_leave_one_out(mode: str) -> int:
    """How many repetitions a group needs before the input can be left out."""
    return _MIN_REPS_FOR_REP_PAIR if mode == "rep_pair" else _MIN_REPS_FOR_LOO


#: Minimum repetitions for a leave-one-out NEX target to mean anything. Excluding
#: the input rep from 2 leaves a *single* noisy rep, which is not an average and
#: carries no sqrt(N) gain — so LOO must decline rather than silently substitute
#: a worse reference (#695).
#:
#: Named because it partitions M4Raw by contrast: with 3 reps for T1/T2 and 2 for
#: FLAIR (#1172), LOO is available for T1/T2 and **structurally impossible** for
#: FLAIR. That is not a configuration choice an arm can make — a FLAIR target
#: always contains the input rep's own noise at 1/N.
_MIN_REPS_FOR_LOO: int = 3

#: What a group with fewer than ``_MIN_REPS_FOR_LOO`` repetitions does when the
#: arm asked for a leave-one-out target. ``error`` refuses at construction, so a
#: 2-rep contrast (M4Raw FLAIR) can never silently grade against the correlated
#: all-reps reference inside a leave-one-out run (#695). ``all_reps`` accepts the
#: all-reps average for those groups as a DECLARED choice and logs it once.
_VALID_NEX_FALLBACKS: frozenset[str] = frozenset({"error", "all_reps"})

#: Default contrast on the *source* side of a federated pair.
#: :meth:`M4RawRepetitionDataset._build_federated_pairs` skips any patient
#: without the source and pairs it against every other (or every declared)
#: contrast. Stamped onto the record so :meth:`_index_stats` can attribute a
#: file to a contrast by *reading* the record, rather than re-deriving it from
#: the filename -- which would be a third copy of the ``stem.split("_")``
#: convention already spelled in ``_build_index`` and ``_build_federated_pairs``.
_FEDERATED_SOURCE_CONTRAST: str = "T1"

#: Contrasts the M4Raw release ships; a pairing naming anything else is a typo.
M4RAW_CONTRASTS: frozenset[str] = frozenset({"T1", "T2", "FLAIR"})


def _phase_align(rep: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """``rep`` rotated by the global phase that best aligns it to ``ref``.

    One phase per ``(slice, coil)`` plane: a repetition's receiver reference
    phase is a per-acquisition scalar, which is what this removes.
    """
    dot = (rep * ref.conj()).sum(dim=(-2, -1), keepdim=True)
    return rep * torch.exp(-1j * torch.angle(dot))


def _pe_support(kspace: torch.Tensor) -> torch.Tensor:
    """1-D bool over the phase-encode (last) axis: the columns that were acquired.

    Derived from the DATA, not the header: M4Raw ships ~24 % of its phase-encode
    columns as exact zeros (partial phase FOV), and its ``encodingLimits`` report
    a COUNT where ISMRMRD specifies an INCLUSIVE maximum -- measured over all 52
    files on this host, ``kspace_encoding_step_1/maximum`` equals the number of
    sampled columns exactly (195 for T1/T2, 198 for FLAIR). Sampled columns always
    carry noise, so a zero column-sum discriminates them exactly.
    """
    if kspace.dim() < 3:
        raise ValueError(
            f"[M4Raw] a split target needs at least (C, H, W) k-space, got "
            f"{tuple(kspace.shape)}."
        )
    support = kspace.abs().sum(dim=tuple(range(kspace.dim() - 1))) > 0
    if not bool(support.any()):
        raise ValueError(
            "[M4Raw] found no sampled phase-encode columns in this group (the whole "
            "k-space is zero). Splitting it would train on noise alone."
        )
    return support


def fit_scan_noise_covariance(
    kspace: torch.Tensor, *, support_policy: str = "band"
) -> tuple[torch.Tensor, int]:
    """One repetition's ``Sigma_n`` in raw units, fitted from its own coil null space.

    Fitted on the central :data:`_SIGMA_FIT_SLICES` slices over the acquired
    columns (:func:`_pe_support`), corrected for the sampled phase-encode
    fraction (:mod:`~spectramr.infrastructure.physics.scan_coil_calibration`).
    The loader's per-scan fit and the predict path both call this, so a
    deployed self-calibrated arm draws from the matrix it was validated with.

    Returns:
        ``(covariance, slices_fitted)``: ``(C, C)`` complex and the slice count.
    """
    from spectramr.infrastructure.physics.scan_coil_calibration import (
        as_slice_batch,
        fit_scan_covariance,
    )

    support = _pe_support(kspace)
    slices = as_slice_batch(kspace)
    window = sigma_fit_window(slices.shape[0])
    central = slices[window.start : window.stop]
    return fit_scan_covariance(central, support, support_policy=support_policy), int(
        central.shape[0]
    )


def sigma_fit_window(num_slices: int) -> range:
    """The slices of a ``num_slices`` volume that :func:`fit_scan_noise_covariance` fits on.

    Owned here so the slice-level route reads the same central slices from disk
    that the per-group route slices out of a loaded volume.
    """
    lo = max(0, (int(num_slices) - _SIGMA_FIT_SLICES) // 2)
    return range(lo, min(int(num_slices), lo + _SIGMA_FIT_SLICES))


def load_sigma_fit_slices(path: Path | str, num_slices: int) -> torch.Tensor:
    """The :func:`sigma_fit_window` slices of ``path``, read one slice at a time.

    The slice route's fit input; ``scripts/precompute_m4raw_noise_covariance.py``
    reads through it too, so a stored ``Sigma_n`` is the fit the loader would run.
    """
    return torch.cat([_load_kspace(Path(path), s) for s in sigma_fit_window(num_slices)])


def _average_reps(
    reps: list[torch.Tensor],
    mode: str,
    exclude_index: int | None = None,
    shifts: Sequence[tuple[float, float]] | None = None,
) -> torch.Tensor:
    """Combine complex k-space repetitions per :data:`_VALID_TARGET_MODES`.

    Args:
        reps: The per-repetition complex k-space tensors.
        mode: One of :data:`_VALID_TARGET_MODES`.
        exclude_index: When set (leave-one-out NEX target), that repetition is
            dropped from the average. The intended use is to exclude the rep
            used as the *input*, so the target's noise is statistically
            independent of the input's — otherwise input and target share the
            input rep's noise at amplitude ``1/N`` and the supervision is biased
            toward preserving that noise. ``None`` (default) averages all reps
            (byte-identical to the legacy behaviour).
        shifts: When set, the ``(dy, dx)`` displacement in pixels of each of
            ``reps`` in one common frame (:mod:`m4raw_registration`, #2849). Every
            repetition is translated onto the anchor before its phase is aligned,
            so subject motion between repetitions does not superpose shifted
            images. ``None`` averages them as acquired.

    Phase alignment is anchored to the excluded repetition when there is one,
    else to ``reps[0]``: in both cases the input's frame. Anchoring to the first
    *retained* repetition left the leave-one-out target at that repetition's
    receiver phase (median 113 deg from the input on M4Raw, #1863), which a
    phase-equivariant network cannot produce. The anchor contributes its frame,
    not its noise, which is also how ``rep_pair`` aligns.
    """
    if mode not in _VALID_TARGET_MODES:
        raise ValueError(
            f"[M4Raw] Unknown target_mode: {mode!r}. Valid: {sorted(_VALID_TARGET_MODES)}"
        )
    if mode in _SINGLE_REP_SPLIT_MODES:
        # These set input AND target together from ONE repetition, so they cannot
        # be expressed as "combine these repetitions into a target". Reaching here
        # means the construction-site branch was bypassed and the caller is about
        # to get an average where it asked for a split.
        raise ValueError(
            f"[M4Raw] target_mode={mode!r} builds both halves from the input repetition "
            "and must be handled at the construction site, not by _average_reps."
        )
    excluding = exclude_index is not None and 0 <= exclude_index < len(reps)
    if shifts is not None:
        reps = _register_to_anchor(reps, shifts, exclude_index if excluding else 0)
    if mode == "rep_pair":
        # Noise2Noise: the target is ONE other repetition, never an average, so
        # its noise is independent of the input's. ``exclude_index`` is
        # mandatory: a pair that may contain the input is the identity task.
        if not excluding:
            raise ValueError(
                "[M4Raw] target_mode='rep_pair' needs the input repetition's index "
                "(exclude_index) so the pair never contains it."
            )
        others = [r for i, r in enumerate(reps) if i != exclude_index]
        if not others:
            raise ValueError(
                "[M4Raw] target_mode='rep_pair' needs at least one repetition besides the input."
            )
        return _phase_align(others[0], reps[exclude_index])
    if not reps:
        raise ValueError("[M4Raw] _average_reps received no repetitions to average.")
    anchor = reps[exclude_index] if excluding else reps[0]
    kept = [r for i, r in enumerate(reps) if i != exclude_index] if excluding else list(reps)
    if not kept:
        # Defensive: exclude_index dropped the only rep. Caller guards against
        # this (LOO only engages with >=2 reps), but never return an empty mean.
        raise ValueError("[M4Raw] _average_reps received no repetitions to average.")
    if mode == "complex_mean":
        return kept[0].clone() if len(kept) == 1 else torch.stack(kept, dim=0).mean(dim=0)
    return torch.stack([_phase_align(r, anchor) for r in kept], dim=0).mean(dim=0)


def _register_to_anchor(
    reps: list[torch.Tensor], shifts: Sequence[tuple[float, float]], anchor: int
) -> list[torch.Tensor]:
    """``reps`` translated onto ``reps[anchor]``, given each one's displacement ``shifts``."""
    if len(shifts) != len(reps):
        raise ValueError(
            f"[M4Raw] {len(reps)} repetitions but {len(shifts)} registration shifts."
        )
    ay, ax = shifts[anchor]
    return [
        rep if i == anchor else shift_kspace(rep, ay - dy, ax - dx)
        for i, (rep, (dy, dx)) in enumerate(zip(reps, shifts, strict=True))
    ]


class _SkipSampleError(Exception):
    """Raised internally when a sample is unrecoverable (e.g. all rep files corrupt).

    Caught in :meth:`M4RawRepetitionDataset.__getitem__` to transparently
    retry on the next index rather than crashing the DataLoader worker.
    """


def _load_reps_or_skip(
    rep_paths: list[Path],
    context: str,
    slice_index: int | None = None,
    loaded: list[int] | None = None,
) -> list[torch.Tensor]:
    """Load every repetition of a NEX group, or raise ``_SkipSampleError``.

    One helper for both the single-contrast and cross-contrast paths, which had
    drifted: the cross-contrast one raised ``_SkipSampleError`` (the retry protocol
    ``__getitem__`` implements) while the single-contrast one returned ``None``
    for "collate to filter". The collate m4raw actually selects is
    ``ImageCollateStrategy``, which has no ``_filter_none`` -- only
    ``RobustCollateStrategy`` and ``PhysicsCollateStrategy`` do -- so a ``None``
    sample reached it and died as ``TypeError: 'NoneType' object is not
    subscriptable``, naming neither the index nor the file (audit B14).

    Per-rep failures were logged at DEBUG and dropped (B8). That is the more
    expensive half: the target is the AVERAGE of the reps, so losing reps
    quietly lowers the SNR boost from sqrt(N) toward sqrt(1) while the config
    still says NEX denoising. Losses are now a counted WARNING, and the caller
    is handed the realised count so it can refuse the degenerate case.

    Args:
        rep_paths: The repetition files this group promises.
        context: Human-readable identifier for the log/error message.
        slice_index: ``None`` loads whole volumes; an int loads that one slice
            of every repetition (the slice-level route, #1757).
        loaded: when given, receives the ``rep_paths`` index of each returned
            repetition, so a caller can name the file behind ``result[i]``.

    Returns:
        The repetitions that loaded, in manifest order.

    Raises:
        _SkipSampleError: none of them loaded.
        IndexError: ``slice_index`` is outside a repetition's slice range. That
            is an index defect (the record promised a slice the file does not
            have), not an unreadable file, so it is never censused: dropping
            the repetition would serve the rest as a complete average.
    """
    kspace_reps: list[torch.Tensor] = []
    failures: list[str] = []
    for position, path in enumerate(rep_paths):
        try:
            if not path.exists():
                raise FileNotFoundError(f"No such file or directory: '{path}'")
            # The whole volume (the call shape the per-group path has always
            # made) or one slice of it on the slice-level route.
            kspace_reps.append(
                _load_kspace(path) if slice_index is None else _load_kspace(path, slice_index)
            )
            if loaded is not None:
                loaded.append(position)
        except IndexError:
            raise
        except Exception as exc:  # censused below, never silent
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")

    if failures and kspace_reps:
        logger.warning(
            "[M4Raw] %s: %d of %d repetitions unreadable — the NEX target is an "
            "average of %d, so its SNR boost is sqrt(%d)=%.2f, not sqrt(%d)=%.2f. "
            "Failures: %s",
            context,
            len(failures),
            len(rep_paths),
            len(kspace_reps),
            len(kspace_reps),
            len(kspace_reps) ** 0.5,
            len(rep_paths),
            len(rep_paths) ** 0.5,
            "; ".join(failures),
        )

    if not kspace_reps:
        raise _SkipSampleError(
            f"[M4Raw] {context}: all {len(rep_paths)} repetition files failed to "
            f"load. Failures: {'; '.join(failures)}"
        )
    return kspace_reps


def _validate_federated_contrasts(
    source: str, targets: Sequence[str] | None, single_contrast: bool
) -> tuple[str, frozenset[str] | None]:
    """Normalise and check the cross-contrast pairing; raise on anything unread or unknown."""
    source = str(source).upper()
    target_set = None if targets is None else frozenset(str(t).upper() for t in targets)
    if single_contrast and (source != _FEDERATED_SOURCE_CONTRAST or target_set is not None):
        raise ValueError(
            "[M4Raw] source_contrast/target_contrasts choose the cross-contrast "
            "pairing; single_contrast=True pairs nothing."
        )
    unknown = sorted(({source} | (target_set or set())) - M4RAW_CONTRASTS)
    if unknown:
        raise ValueError(f"[M4Raw] Unknown contrast(s) {unknown}; M4Raw ships {sorted(M4RAW_CONTRASTS)}.")
    if target_set is not None and (not target_set or source in target_set):
        raise ValueError(
            f"[M4Raw] target_contrasts={sorted(target_set)} must be non-empty and "
            f"exclude the source contrast {source!r}."
        )
    return source, target_set


def contrast_index_for_filename(filename: str) -> int | None:
    """Index of the contrast keyword (T1/T2/FLAIR/PD) found in *filename*, else ``None``."""
    name_upper = filename.upper()
    for key, idx in CONTRAST_MAP.items():
        if key in name_upper:
            return idx
    return None


def _extract_contrast_idx(filename: str) -> int:
    """Return integer contrast index by keyword search in *filename*.

    Args:
        filename: Bare filename (no directory), e.g. ``m4raw_T1_rep001.h5``.

    Returns:
        Integer in ``[0, len(CONTRAST_MAP))``; defaults to 0 (T1) if no match.
    """
    found = contrast_index_for_filename(filename)
    if found is not None:
        return found
    # No contrast keyword (T1/T2/FLAIR/PD) in the filename. Surfaced at WARNING
    # (was a silent DEBUG): under multi-contrast conditioning a missed keyword
    # silently mislabels the scan as T1. A full raise is deferred because this
    # helper is called per-sample and UNCONDITIONALLY (lines ~717/875), so
    # raising would also crash single-contrast runs that never read the index.
    # (WS-5 DC-5.)
    logger.warning(
        "[M4Raw] No contrast keyword (T1/T2/FLAIR/PD) in '%s'; defaulting to 0 "
        "(T1) — verify the filename if this dataset uses contrast conditioning.",
        filename,
    )
    return 0


def _load_kspace(path: Path, slice_index: int | None = None) -> torch.Tensor:
    """Load complex k-space tensor from an H5 file.

    Args:
        path: Path to ``.h5`` file.
        slice_index: ``None`` (default) reads the whole volume. An int reads
            that one slice along the leading axis (the slice-level route,
            ``data.slice_level_records``, #1757); the slice axis is kept at
            length 1 so every consumer sees the same ``(S, [C,] H, W)`` layout.

    Returns:
        Complex float32 tensor of shape ``(Slices, [Coils,] H, W)``; ``Slices``
        is 1 when ``slice_index`` was given.

    Raises:
        KeyError: If the H5 file contains no ``kspace`` dataset.
        IndexError: ``slice_index`` is outside the file's slice range.
    """
    # ``swmr``: the single-writer/multiple-reader mode every other HDF5 reader
    # under ``data/`` opens with (``io_strategies.py``), so the loader workers
    # share a file that is still being written and never block one another. The
    # handle is cached per process, since every slice of every repetition reads here.
    f = open_h5(path)
    if "kspace" not in f:
        raise KeyError(f"No 'kspace' dataset found in {path}. Available keys: {list(f.keys())}")
    # The read is the one shared with ``FastMRIH5Strategy`` (numpy, may be
    # complex or float32 with trailing 2); h5py drops the indexed axis, so
    # the slice read re-adds it as a length-1 leading dim.
    raw = read_h5_kspace(f["kspace"], slice_index)
    if slice_index is not None:
        raw = raw[None]

    t = torch.from_numpy(raw)

    # Handle real-valued storage with trailing channel-2 dimension
    if not torch.is_complex(t):
        t = torch.view_as_complex(t.contiguous().float()) if t.shape[-1] == 2 else t.float()
    else:
        t = t.to(torch.complex64)

    return t


def _read_kspace_shape(path: Path | str) -> tuple[int, ...] | None:
    """Return the effective complex k-space shape WITHOUT loading voxels.

    Reads only the HDF5 ``kspace`` dataset header (``.shape`` / ``.dtype``).
    Real storage with a trailing real/imag axis of size 2 (the layout
    :func:`_load_kspace` collapses via ``view_as_complex``) has that axis
    dropped, so the returned shape matches the complex tensor the loader yields:
    H/W are the trailing two axes and the slice/depth axis leads — exactly the
    convention :meth:`TorchIOQueueBuilder._filter_patch_compatible_subjects`
    assumes for its no-voxel-load fast path.

    Returns ``None`` on any read error (missing file, no ``kspace`` key,
    unreadable header) so the caller falls back rather than crashes. This is
    the F1 fix for the queue-build host-OOM (the slow filter path materialised
    the whole 39 GB corpus); see ``project_queue_build_oom_patch_filter``.
    """
    try:
        with h5py.File(str(path), "r", swmr=True) as f:
            ds = f.get("kspace")
            if not isinstance(ds, h5py.Dataset):
                return None
            shape = tuple(int(d) for d in ds.shape)
            is_complex = np.issubdtype(ds.dtype, np.complexfloating)
    except (OSError, KeyError, ValueError, TypeError):
        return None
    if not is_complex and shape and shape[-1] == 2:
        shape = shape[:-1]
    return shape if len(shape) >= 2 else None


def _rss_combine(kspace: torch.Tensor) -> torch.Tensor:
    """Phase-preserving coil combination -> single virtual coil k-space.

    The legacy implementation did ``fft2c(|rss|)`` after coil-RSS magnitude,
    which silently throws phase away. Taking the FFT of a real-valued
    image yields Hermitian-symmetric k-space, so any downstream ``ifft2c``
    on it produces a centro-symmetric magnitude image — visually a
    "doubled brain" (the brain superimposed with its 180°-rotated copy).
    That regression is what this function is now hardened against; see
    findings booklet 2026-05-06 and CLAUDE.md #9 (no silent fallbacks).

    The fix combines coils with phase from the highest-energy coil as
    reference (poor man's adaptive coil combine):

      I_combined(x) = |I(x)|_rss * exp(j * angle(I_ref(x)))

    where I_ref is the coil with the largest mean magnitude. This
    preserves the genuine non-Hermitian structure of the underlying
    complex image so the resulting k-space is NOT Hermitian-symmetric
    and ``ifft2c`` round-trips cleanly.

    Args:
        kspace: Complex tensor of shape ``(Slices, Coils, H, W)``.

    Returns:
        Complex tensor of shape ``(Slices, H, W)`` -- single virtual coil.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

    s, c, h, w = kspace.shape
    ks = kspace.reshape(s * c, h, w)
    images = ifft2c(ks).reshape(s, c, h, w)  # (S, C, H, W) complex

    coil_energy = images.abs().mean(dim=(0, 2, 3))  # (C,)
    ref_idx = int(coil_energy.argmax().item())
    ref_phase = torch.angle(images[:, ref_idx : ref_idx + 1])  # (S, 1, H, W)

    rss_mag = torch.sqrt((images.abs() ** 2).sum(dim=1, keepdim=True) + 1e-12)
    combined = rss_mag * torch.exp(1j * ref_phase.to(rss_mag.dtype))
    return fft2c(combined.squeeze(1))


def _normalize_kspace(kspace: torch.Tensor, percentile: float = 0.99) -> torch.Tensor:
    """Divide by magnitude percentile (robust, avoids DC-spike dominance).

    Args:
        kspace: Any shape complex tensor.
        percentile: Quantile in ``(0, 1]``.

    Returns:
        Normalized complex tensor (same shape and dtype).
    """
    # Use the SSOT quantile (interpolated) to match
    # ``spectramr.data.transforms.normalization.normalize_percentile``. Earlier
    # versions used ``torch.kthvalue`` which returns the exact k-th element
    # without interpolation, producing a subtly different scale than the
    # rest of the codebase. See findings booklet 2026-05-05 N-4.
    flat_mag = kspace.abs().flatten().float()
    scale = torch.quantile(flat_mag, percentile).clamp(min=1e-8)
    return kspace / scale


def _to_torchio_tensor(kspace: torch.Tensor) -> torch.Tensor:
    """Convert complex k-space to TorchIO format ``(C, H, W, D)``.

    TorchIO expects ``(Channels, H, W, Depth)`` layout. We map:
    - ``Slices`` → ``Depth`` OR ``Reps`` → ``Depth`` (if Reps provided and Slices=1)
    - ``real / imag`` → ``Channels [0, 1]``

    Args:
        kspace: Complex tensor ``(Slices, H, W)`` or ``(Slices, H, W, Reps)``.

    Returns:
        Float32 tensor ``(2, H, W, D)``.
    """
    if kspace.dim() == 4:
        # (Slices, H, W, Reps) -> map Reps to TorchIO Depth
        if kspace.shape[0] != 1:
            raise ValueError(
                f"Cannot map both Slices ({kspace.shape[0]}) and Reps to 4D TorchIO format"
            )

        kspace_sq = kspace.squeeze(0)  # (H, W, Reps)
        real = kspace_sq.real  # (H, W, Reps)
        imag = kspace_sq.imag  # (H, W, Reps)
        return torch.stack([real, imag], dim=0).float()  # (2, H, W, Reps)

    real = kspace.real.permute(1, 2, 0)  # (H, W, Slices)
    imag = kspace.imag.permute(1, 2, 0)  # (H, W, Slices)
    return torch.stack([real, imag], dim=0).float()  # (2, H, W, Slices)


def _attach_coil_maps(
    subject: tio.Subject,
    maps: torch.Tensor,
    source: str,
    calibration: dict[str, Any],
    affine: np.ndarray,
) -> None:
    """Publish ``(S, C, H, W)`` complex maps the way ``espirit_sensitivity`` does.

    ``sensitivity_complex`` for phase-sensitive consumers, ``sensitivity`` for the
    magnitude convention, both ``(C, H, W, S)`` so the patch sampler crops them
    with the k-space. The two stamps tell a declared ``espirit_sensitivity``
    that the maps are already there and which calibration made them.
    """
    from spectramr.data.datasets.m4raw_coil_maps import calibration_tag

    complex_maps = maps.permute(1, 2, 3, 0).contiguous()
    subject.add_image(tio.ScalarImage(tensor=complex_maps, affine=affine), "sensitivity_complex")
    subject.add_image(tio.ScalarImage(tensor=complex_maps.abs(), affine=affine), "sensitivity")
    subject["coil_maps_source"] = source
    subject["coil_maps_calibration"] = calibration_tag(calibration)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class M4RawRepetitionDataset(Dataset):
    """M4Raw dataset that groups repetitions and returns averaged k-space targets.

    Args:
        h5_files: All H5 file paths for this split (may include multiple reps).
        normalize_kspace: Apply percentile-based k-space normalization.
        kspace_percentile: Percentile used for normalization (default 0.99).
        transform: Optional TorchIO transform applied to the returned subject.
        use_repetitions: If ``False``, return the first rep only for both
            input and target (useful for ablation / debugging).
        slice_level_records: One record per (group, slice) rather than per
            group; each ``__getitem__`` reads one slice of every repetition and
            yields a depth-1 subject (#1757). Off by default.
    """

    def __init__(
        self,
        h5_files: list[Path],
        # Default False: the dataset itself does not normalize (the transform
        # does), so defaulting to True would advertise a morph it never performs.
        normalize_kspace: bool = False,
        kspace_percentile: float = 0.99,
        transform: Callable | None = None,
        use_repetitions: bool = True,
        coil_processing_mode: str = "rss",
        num_virtual_coils: int = 4,
        single_contrast: bool = False,
        source_layout: str = "stacked",
        source_contrast: str = _FEDERATED_SOURCE_CONTRAST,
        target_contrasts: Sequence[str] | None = None,
        log_scaling: bool = False,
        target_mode: str = "complex_mean",
        r2r_alpha: float = 1.0,
        r2r_covariance_source: str = "committed",
        nex_target_exclude_input: bool = False,
        nex_fallback: str = "error",
        nex_registration: str = "none",
        nex_shift_root: str | Path | None = None,
        nex_shifts_missing: str = "raise",
        slice_level_records: bool = False,
        r2r_split_angle_deg: float = 45.0,
        r2r_whitening: str = "committed",
        emit_alternate_input: bool = False,
        n2n_pairing: str | None = None,
        coil_maps_root: str | Path | None = None,
        coil_maps_reference_root: str | Path | None = None,
        coil_maps_missing: str = "raise",
        noise_fit_support_policy: str = "band",
        noise_covariance_root: str | Path | None = None,
        noise_covariance_missing: str = "raise",
        coil_maps_recipe: str = DEFAULT_RECIPE,
        coil_maps_acs_size: int | None = None,
        input_repetition: str = "first",
    ) -> None:
        """Initialize M4Raw dataset.

        Args:
            h5_files: All H5 file paths for this split.
            normalize_kspace: Apply percentile-based k-space normalization.
            kspace_percentile: Percentile for normalization (default 0.99).
            transform: Optional TorchIO transform.
            use_repetitions: Average repetitions for high-SNR targets.
            coil_processing_mode: Coil processing mode.
            num_virtual_coils: Number of virtual coils for SVD.
            single_contrast: If True, each contrast is loaded independently
                (no cross-contrast pairing). Each sample = 1 contrast with
                repetition averaging. For contrast-agnostic training.
            source_layout: cross-contrast route only, one of
                :data:`VALID_SOURCE_LAYOUTS`. ``separate`` serves the source
                contrast's acquired repetition as ``kspace_source`` beside a
                target-only ``input``/``target``/``kspace``; ``target_only``
                serves the same target without the source.
            source_contrast: cross-contrast route only, the contrast every
                pair takes its source from (``T1`` by default; ``T2`` for
                T2-guided FLAIR).
            target_contrasts: cross-contrast route only, the contrasts paired
                against the source; ``None`` pairs every other contrast.
            log_scaling: After the (Parseval/image-RSS) percentile divide, apply
                phase-preserving log1p magnitude compression
                (``compress_kspace_log``, the SSOT primitive) to tame the DC
                dynamic range. Inverted by the training strategy before metrics.
            slice_level_records: Expand the index to one record per slice and
                read one slice per repetition per item (#1757). ``False``
                keeps the per-group index byte-identical.
            r2r_split_angle_deg: ``nullspace_r2r`` split angle in (0, 90);
                ``tan`` of it is the R2R ``tau``.
            r2r_whitening: ``nullspace_r2r`` whitener, one of
                :data:`~spectramr.infrastructure.physics.coil_split_r2r.VALID_WHITENING`.
            emit_alternate_input: under an averaging target, also publish
                ``input_alt`` -- repetition 1, phase-aligned to the input
                repetition. A split-mode (r2r) training split has no second
                repetition to publish and emits nothing.
            nex_registration: one of
                :data:`~spectramr.data.datasets.m4raw_registration.VALID_NEX_REGISTRATIONS`.
                ``translation`` registers every repetition onto the input before
                the NEX target (and ``input_alt``) is built from it (#2849).
            nex_shift_root: directory holding the shifts written by
                ``scripts/precompute_m4raw_nex_shifts.py``; read instead of estimating
                them in every worker (#2977). Needs ``nex_registration: translation``.
            nex_shifts_missing: ``raise`` (default) or ``estimate`` -- what a group
                absent from the store does.
            coil_maps_root: directory of per-group ESPIRiT maps written by
                ``scripts/precompute_m4raw_coil_maps.py``. When set, every
                subject carries ``sensitivity_complex``/``sensitivity`` from
                ``<root>/<patient>_<contrast>.pt`` and ``nullspace_r2r`` splits
                with them instead of estimating per load (#2339).
            coil_maps_missing: ``raise`` (default) or ``estimate`` -- what a
                group without a map file does. A file that exists but does not
                match always raises.
            coil_maps_recipe: which repetitions the maps are estimated from, one
                of :data:`~spectramr.data.datasets.m4raw_coil_map_recipe.VALID_RECIPES`;
                checked against every file and used by the ``estimate`` fallback.
            coil_maps_acs_size: the ESPIRiT ACS width the maps were estimated
                with; ``None`` keeps the per-scan default calibration.
            noise_fit_support_policy: the per-scan Sigma_n fit's phase-encode
                support policy, one of
                :data:`~spectramr.infrastructure.physics.pe_support.VALID_SUPPORT_POLICIES`.
            noise_covariance_root: directory holding the per-group Sigma_n written by
                ``scripts/precompute_m4raw_noise_covariance.py``; read instead of fitting
                in every worker (#3117). Needs a self-calibrated r2r draw or whitener.
            noise_covariance_missing: ``raise`` (default) or ``fit`` -- what a group
                absent from the store does.
            input_repetition: which loaded repetition is the input, one of
                :data:`~spectramr.data.datasets.m4raw_input_repetition.VALID_INPUT_REPETITIONS`.
                ``random`` draws one per sample and aligns the NEX target to it;
                the instantiator passes it to the training split only.
        """
        self.normalize_kspace = normalize_kspace
        self.kspace_percentile = kspace_percentile
        self.log_scaling = log_scaling
        self.transform = transform
        # The dataset no longer normalizes: KSpaceNormalizationTransform owns
        # it. Asking for normalization with no transform to apply it would
        # silently serve raw k-space (pitfall #9), so fail loud instead.
        if normalize_kspace and transform is None:
            raise ValueError(
                "[M4Raw] normalize_kspace=True but no transform was supplied. "
                "K-space normalization is applied by "
                "KSpaceNormalizationTransform (built by TorchIOTransformBuilder "
                "when data.normalize_kspace is set), not by the dataset — the "
                "dataset matches and serves. Pass the built transform pipeline, "
                "or set normalize_kspace=False."
            )
        self.use_repetitions = use_repetitions
        self.coil_processing_mode = coil_processing_mode
        if coil_processing_mode not in _VALID_COIL_MODES:
            raise ValueError(
                f"[M4Raw] Unknown coil_processing_mode: {coil_processing_mode!r}. "
                f"Valid modes: {sorted(_VALID_COIL_MODES)}"
            )
        if coil_processing_mode in _UNIMPLEMENTED_COIL_MODES:
            raise NotImplementedError(
                f"[M4Raw] coil_processing_mode={coil_processing_mode!r} is recognized "
                "but NOT implemented on the M4Raw repetition path: only 'rss' is "
                "applied, so it would silently train on uncompressed multi-coil "
                "k-space (CLAUDE.md pitfall #16 facade / #15b). SVD virtual-coil "
                "compression is wired only on the UniversalMRIDataset path -- set "
                "dataset_type: kspace (or fastmri_kspace), or use "
                "coil_processing_mode: 'rss' / 'none' here. (M4Raw has 4 coils, so "
                "svd -> num_virtual_coils=4 is a no-op even where wired.)"
            )
        self.num_virtual_coils = num_virtual_coils
        self.single_contrast = single_contrast
        if source_layout not in VALID_SOURCE_LAYOUTS:
            raise ValueError(
                f"[M4Raw] Unknown source_layout: {source_layout!r}. "
                f"Valid layouts: {sorted(VALID_SOURCE_LAYOUTS)}"
            )
        if single_contrast and source_layout != "stacked":
            raise ValueError(
                f"[M4Raw] source_layout={source_layout!r} describes the "
                "cross-contrast route and single_contrast=True has no source "
                "contrast to serve."
            )
        self.source_layout = source_layout
        self.source_contrast, self.target_contrasts = _validate_federated_contrasts(
            source_contrast, target_contrasts, single_contrast
        )
        if target_mode not in _VALID_TARGET_MODES:
            raise ValueError(
                f"[M4Raw] Unknown target_mode: {target_mode!r}. "
                f"Valid modes: {sorted(_VALID_TARGET_MODES)}"
            )
        self.target_mode = target_mode
        # Built eagerly under r2r so a bad covariance/alpha fails at construction
        # rather than at iteration 1 in a worker process, where the traceback is
        # a DataLoader crash with no arm in it.
        self.r2r_alpha = float(r2r_alpha)
        if r2r_covariance_source not in _VALID_R2R_COVARIANCE_SOURCES:
            raise ValueError(
                f"[M4Raw] Unknown r2r_covariance_source: {r2r_covariance_source!r}. "
                f"Valid: {sorted(_VALID_R2R_COVARIANCE_SOURCES)}."
            )
        self.r2r_covariance_source = r2r_covariance_source
        #: Per-scan Sigma_n (a ``(C, C)`` tensor), keyed by repetition group. A
        #: receive-chain property, so one fit serves every slice of a scan; the
        #: ``self_calibrated`` R2R draw and the null-space whitener share it.
        self._r2r_cov_cache: dict[str, torch.Tensor] = {}
        self._r2r_sampler: Any = None
        if target_mode == "r2r" and r2r_covariance_source == "committed":
            from spectramr.infrastructure.physics.m4raw_noise import M4RawNoiseSampler

            self._r2r_sampler = M4RawNoiseSampler(alpha=self.r2r_alpha)
        from spectramr.infrastructure.physics.coil_split_r2r import (
            VALID_WHITENING,
            check_split_angle,
        )

        check_split_angle(r2r_split_angle_deg)
        if r2r_whitening not in VALID_WHITENING:
            raise ValueError(
                f"[M4Raw] Unknown r2r_whitening: {r2r_whitening!r}. "
                f"Valid: {sorted(VALID_WHITENING)}."
            )
        self.r2r_split_angle_deg = float(r2r_split_angle_deg)
        self.r2r_whitening = r2r_whitening
        from spectramr.infrastructure.physics.scan_coil_calibration import VALID_SUPPORT_POLICIES

        if noise_fit_support_policy not in VALID_SUPPORT_POLICIES:
            raise ValueError(
                f"[M4Raw] Unknown noise_fit_support_policy: {noise_fit_support_policy!r}. "
                f"Valid: {sorted(VALID_SUPPORT_POLICIES)}."
            )
        self.noise_fit_support_policy = noise_fit_support_policy
        self._noise_covariance_store = (
            None
            if noise_covariance_root is None
            else NoiseCovarianceStore.declared(
                noise_covariance_root, noise_covariance_missing, noise_fit_support_policy
            )
        )
        if target_mode == "nullspace_r2r" and coil_processing_mode != "none":
            raise ValueError(
                "[M4Raw] target_mode='nullspace_r2r' splits the PHYSICAL coil vector "
                f"and needs coil_processing_mode='none', got {coil_processing_mode!r}."
            )
        # Scoring-only maps (#2664): published under a key no operator, loss or
        # model input reads; the instantiator passes them to the validation split.
        from spectramr.data.datasets.m4raw_coil_map_source import CoilMapSource

        self._reference_maps = (
            None
            if coil_maps_reference_root is None
            else CoilMapSource.declared(Path(coil_maps_reference_root), "raise")
        )
        self._coil_maps = self._coil_map_source(
            coil_maps_root,
            coil_maps_missing,
            coil_processing_mode,
            single_contrast,
            recipe=coil_maps_recipe,
            acs_size=coil_maps_acs_size,
        )
        if n2n_pairing is not None:
            allowed = N2N_PAIRINGS.get(n2n_pairing)
            if allowed is None:
                raise ValueError(
                    f"[M4Raw] Unknown n2n_pairing: {n2n_pairing!r}. Valid: {sorted(N2N_PAIRINGS)}"
                )
            if target_mode not in allowed or not single_contrast or not use_repetitions:
                raise ValueError(
                    f"[M4Raw] n2n_pairing={n2n_pairing!r} needs the single-contrast route "
                    f"with repetitions and target_mode in {sorted(allowed)}; got "
                    f"target_mode={target_mode!r}, single_contrast={single_contrast}, "
                    f"use_repetitions={use_repetitions}."
                )
        self.n2n_pairing = n2n_pairing
        self.emit_alternate_input = bool(emit_alternate_input)
        if self.emit_alternate_input and not use_repetitions:
            raise ValueError(
                "[M4Raw] emit_alternate_input publishes a SECOND repetition, and "
                "use_repetitions=False loads only the first."
            )
        # Leave-one-out NEX target: exclude the input rep (rep 0) from the
        # averaged target so target and input noise are uncorrelated. Default
        # False keeps the all-reps average (byte-identical legacy behaviour).
        # Trades one rep of √N SNR for an unbiased target; only engages when
        # >=3 reps exist (with 2 reps the LOO target would be a single noisy
        # rep, defeating the purpose — it falls back to the all-reps average).
        self.nex_target_exclude_input = nex_target_exclude_input
        # Rep counts already reported by `_note_loo_declined`. The gate above is
        # contrast-dependent (#695): M4Raw ships FLAIR at 2 reps and T1/T2 at 3,
        # so a `single_contrast` run holds all three and LOO fires for some
        # samples and not others. That makes the reported PSNR/SSIM an average
        # over two different references, and it used to be invisible in every
        # log. Per-sample logging would be a hot-path GPU-free but IO-heavy
        # spam, so report each distinct rep count once per worker.
        self._loo_declined_reported: set[int] = set()
        if nex_fallback not in _VALID_NEX_FALLBACKS:
            raise ValueError(
                f"[M4Raw] Unknown nex_fallback: {nex_fallback!r}. "
                f"Valid: {sorted(_VALID_NEX_FALLBACKS)}"
            )
        self.nex_fallback = nex_fallback
        if nex_registration not in VALID_NEX_REGISTRATIONS:
            raise ValueError(
                f"[M4Raw] Unknown nex_registration: {nex_registration!r}. "
                f"Valid: {sorted(VALID_NEX_REGISTRATIONS)}"
            )
        if nex_registration != "none" and target_mode in _SINGLE_REP_SPLIT_MODES:
            raise ValueError(
                f"[M4Raw] nex_registration={nex_registration!r} registers repetitions "
                f"against each other, and target_mode={target_mode!r} builds both halves "
                "from one repetition; there is nothing to register."
            )
        self.nex_registration = nex_registration
        if nex_shift_root is not None and nex_registration == "none":
            raise ValueError(
                "[M4Raw] nex_shift_root holds registration shifts and nex_registration is "
                "'none'; nothing would read them. Declare nex_registration: translation."
            )
        self._shift_store = (
            None if nex_shift_root is None else NexShiftStore.declared(nex_shift_root, nex_shifts_missing)
        )
        # One group's shifts, keyed by the repetition files that loaded; a group
        # is estimated once per worker however many of its slices are drawn.
        self._shift_cache: dict[tuple[str, ...], list[tuple[float, float]]] = {}
        self.input_repetition = validate_input_repetition(
            input_repetition,
            target_mode=target_mode,
            use_repetitions=use_repetitions,
            single_contrast=single_contrast,
            nex_target_exclude_input=nex_target_exclude_input,
            has_coil_maps=self._coil_maps is not None or self._reference_maps is not None,
        )
        self._draw_input = InputRepetitionDraw(self.input_repetition)
        self.slice_level_records = bool(slice_level_records)

        # Build groups: base_stem → sorted list of rep paths (as strings for pickle safety)
        # BUG FIX: Store paths as strings to avoid Path object pickling issues with TorchIO Queue
        raw_groups = self._build_groups(h5_files)

        if single_contrast:
            # Single-contrast mode: each contrast group = one sample
            # No cross-contrast pairing (T1→T2). Each contrast trained independently.
            single_index = self._build_single_contrast_index(raw_groups)
            self._federated_pairs = []  # Not used in single_contrast mode
            self._groups: list[dict[str, Any]] = [
                {
                    "paths": [str(p) for p in entry["paths"]],
                    "contrast": entry["contrast"],
                    "patient_id": entry["patient_id"],
                }
                for entry in single_index
            ]
        else:
            # Cross-contrast mode: pair the source contrast with each target contrast.
            self._federated_pairs = self._build_federated_pairs(
                raw_groups, source=self.source_contrast, targets=self.target_contrasts
            )

            # DEBUG: Log first path before conversion to detect corruption
            if self._federated_pairs and self._federated_pairs[0]["target"]:
                first_path = self._federated_pairs[0]["target"][0]
                logger.debug(
                    f"[M4Raw __init__] First target group[0] PATH BEFORE string conversion: {first_path}"
                )

            # We will use the federated pairs as the source of truth for length and indexing
            self._groups: list[dict[str, Any]] = [
                {
                    "source": [str(p) for p in pair["source"]],
                    "target": [str(p) for p in pair["target"]],
                    "patient_id": pair["patient_id"],
                    "target_contrast": pair["target_contrast"],
                }
                for pair in self._federated_pairs
            ]

        # DEBUG: Log first stored string path to verify storage
        if self._groups:
            g0 = self._groups[0]
            if single_contrast:
                first_str = g0["paths"][0] if g0.get("paths") else "Empty"
                logger.debug(
                    f"[M4Raw __init__] Single-contrast mode: first group paths[0]: {first_str}"
                )
            else:
                first_str = g0["target"][0] if g0.get("target") else "Empty"
                logger.debug(
                    f"[M4Raw __init__] Cross-contrast mode: first group target[0]: {first_str}"
                )
            logger.debug(
                f"[M4Raw __init__] _groups id: {id(self._groups)}, _groups[0] id: {id(g0)}"
            )

        # Leave-one-out needs >= _MIN_REPS_FOR_LOO reps in EVERY group it will
        # serve; refuse here, before any GPU time, rather than at the first
        # short group mid-epoch (or -- the #695 shape -- never).
        self._refuse_undersized_loo_groups()
        self._refuse_groups_without_an_alternate()
        # The same for the precomputed coil maps: every group's file must exist now, not when
        # the first affected sample is drawn.
        for source in (self._coil_maps, self._reference_maps):
            if source is not None:
                source.refuse_absent_maps(
                    Path(g["paths"][0]).stem for g in self._groups if g.get("paths")
                )

        # F1: stamp each group with its raw k-space shape (header read only) so
        # the queue-build patch-compat filter can run its no-voxel-load fast
        # path instead of materialising the whole corpus → host OOM.
        self._attach_shape_metadata()

        n_groups = len(self._groups)
        # Slice-level route (#1757): one record per (group, slice), each a
        # depth-1 subject, so a served sample costs one slice read per
        # repetition instead of every repetition in full. Header reads only.
        if self.slice_level_records:
            self._groups = expand_to_slice_records(self._groups, _read_kspace_shape)

        logger.info(
            "[M4RawRepetitionDataset] %d anatomy groups from %d H5 files "
            "(use_repetitions=%s, target_mode=%s, slice_level_records=%s -> %d records)",
            n_groups,
            len(h5_files),
            use_repetitions,
            self.target_mode,
            self.slice_level_records,
            len(self._groups),
        )

    @staticmethod
    def _coil_map_source(
        root: str | Path | None,
        missing: str,
        coil_mode: str,
        single_contrast: bool,
        *,
        recipe: str = DEFAULT_RECIPE,
        acs_size: int | None = None,
    ) -> Any:
        """The precomputed-map source, or ``None``; refuses the routes that cannot use one.

        The calibration is the per-scan default the loader would estimate with,
        with ``acs_size`` replacing its ACS width when declared, so a file written
        with any other ACS/kernel/threshold or recipe raises at load.
        """
        from spectramr.data.datasets.m4raw_coil_map_recipe import check_recipe
        from spectramr.data.datasets.m4raw_coil_map_source import (
            VALID_MISSING_POLICIES,
            CoilMapSource,
        )

        if missing not in VALID_MISSING_POLICIES:
            raise ValueError(
                f"[M4Raw] Unknown coil_maps_missing: {missing!r}. "
                f"Valid: {sorted(VALID_MISSING_POLICIES)}."
            )
        check_recipe(recipe)
        if root is None:
            return None
        if coil_mode != "none":
            raise ValueError(
                "[M4Raw] coil_maps_root publishes maps of the PHYSICAL coils and needs "
                f"coil_processing_mode='none', got {coil_mode!r}."
            )
        if not single_contrast:
            raise ValueError(
                "[M4Raw] coil_maps_root is implemented on the single-contrast route "
                "only; the cross-contrast input stacks two contrasts' coils."
            )
        return CoilMapSource.declared(root, missing, recipe=recipe, acs_size=acs_size)

    # ------------------------------------------------------------------
    # Leave-one-out NEX reference: construction-time contract
    # ------------------------------------------------------------------

    def _refuse_undersized_loo_groups(self) -> None:
        """Refuse a leave-one-out run whose groups cannot all honour it.

        Both routes apply leave-one-out: in cross-contrast mode each side's input is
        repetition 0 of that side's own group, so every source and target group is
        checked. With ``nex_fallback='all_reps'`` the short groups are served with
        the correlated all-reps average and ``_note_loo_declined`` reports it once
        per rep count at load time.
        """
        if not (self.use_repetitions and self.nex_target_exclude_input):
            return
        if self.target_mode in _SINGLE_REP_SPLIT_MODES:
            # These build both halves from the input repetition alone -- they
            # never average, so they never leave anything out and a 1-rep group
            # is the POINT of the mode. `nex_target_exclude_input` still arrives
            # True because it is ONE config knob shared by both splits, and the
            # VALIDATION dataset (built with `data.val_target_mode`) needs it.
            return
        if self.nex_fallback != "error" and self.target_mode != "rep_pair":
            return
        short: dict[str, int] = {}
        min_reps = _min_reps_for_leave_one_out(self.target_mode)
        for group in self._groups:
            sides = (
                [(group.get("paths"), group.get("contrast"))]
                if self.single_contrast
                else [
                    (group.get("source"), group.get("source_contrast", _FEDERATED_SOURCE_CONTRAST)),
                    (group.get("target"), group.get("target_contrast")),
                ]
            )
            for paths, contrast in sides:
                if len(paths or []) < min_reps:
                    label = str(contrast or "UNKNOWN")
                    short[label] = short.get(label, 0) + 1
        if not short:
            return
        detail = ", ".join(f"{label}: {count} group(s)" for label, count in sorted(short.items()))
        raise ValueError(
            "[M4Raw] nex_target_exclude_input=True (leave-one-out NEX target) but "
            f"{sum(short.values())} group(s) have fewer than {min_reps} "
            f"repetitions ({detail}). Leave-one-out would leave a single noisy rep "
            "there, and the all-reps average is a DIFFERENT reference (the input's "
            "own noise sits in it at 1/N), so serving it silently would mix two "
            "references into one metric (#695). Declare data.nex_fallback: all_reps "
            "to accept that for the short contrasts, or exclude them via "
            "data.pairing.contrasts."
        )

    def _publishes_noise_covariance(self) -> bool:
        """An averaging split of an arm whose R2R noise model is the scan's own fit.

        That is the validation split of a self-calibrated ``r2r``/``nullspace_r2r``
        arm; ``validation.scoring.recorrupted_inference`` reads the published
        ``noise_covariance`` there. One fit per group per worker, cached.
        """
        return self.target_mode not in _SINGLE_REP_SPLIT_MODES and "self_calibrated" in (
            self.r2r_covariance_source,
            self.r2r_whitening,
        )

    def _emits_alternate_input(self) -> bool:
        """``input_alt`` is published only where the target averages repetitions."""
        return self.emit_alternate_input and self.target_mode in _AVERAGING_TARGET_MODES

    def _refuse_groups_without_an_alternate(self) -> None:
        """Every group this dataset serves ``input_alt`` from needs a repetition 1."""
        if not self._emits_alternate_input():
            return
        if not self.single_contrast:
            raise ValueError(
                "[M4Raw] emit_alternate_input is implemented on the single-contrast "
                "route only; the cross-contrast input is a different contrast's "
                "repetition and has no alternate to publish."
            )
        short = sum(1 for group in self._groups if len(group.get("paths") or []) < 2)
        if short:
            raise ValueError(
                f"[M4Raw] emit_alternate_input needs >= 2 repetitions per group, but "
                f"{short} group(s) have fewer. Exclude them via data.pairing.contrasts "
                "or drop the knob."
            )

    # ------------------------------------------------------------------
    # Index / shape metadata (queue-build fast path — F1)
    # ------------------------------------------------------------------

    @property
    def index(self) -> list[dict[str, Any]]:
        """The index records — the SSOT for ``__len__`` / ``__getitem__``.

        One per (patient, contrast) group by default; one per (group, slice)
        under ``slice_level_records`` (each then carries ``slice_index``).

        Exposed so :meth:`TorchIOQueueBuilder._filter_patch_compatible_subjects`
        can read each record's ``shape`` (no-voxel fast path) and prune in place
        via the setter without materialising any subject. See F1.
        """
        return self._groups

    @index.setter
    def index(self, value: list[dict[str, Any]]) -> None:
        self._groups = value

    def _attach_shape_metadata(self) -> None:
        """Stamp each group with its raw k-space ``shape`` (header read, no voxels).

        Reads only the first repetition file's HDF5 header per group. Groups
        whose first file is unreadable are left without a ``shape`` key; the
        queue-build fast path requires *every* record to carry one, so in that
        (rare) case it conservatively falls back to the slow probe rather than
        silently mis-filtering. See :func:`_read_kspace_shape` and F1.
        """
        for rec in self._groups:
            files = rec.get("source") or rec.get("paths") or rec.get("target")
            if not files:
                continue
            shape = _read_kspace_shape(files[0])
            if shape is not None:
                rec["shape"] = shape

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    @staticmethod
    def _build_groups(h5_files: list[Path]) -> list[list[Path]]:
        """Group files by the first (len(stem)-2) characters of their stem.

        Files that differ only in the last 2 digits of the stem (directly concatenated,
        e.g. ``01``, ``02``, ``03``) are considered repetitions of the same anatomy.

        Args:
            h5_files: Flat list of H5 file paths.

        Returns:
            List of lists, each inner list containing the paths of one rep group
            (sorted for reproducibility).
        """
        groups: dict[str, list[Path]] = {}
        for path in h5_files:
            stem = path.name.split(".")[0]  # filename without ANY extensions
            # One owner for the rule (m4raw_identity): it was written here as
            # `stem[:-2]` and again in the manifest generator, and the batch
            # identity fields were about to add a third copy.
            base = repetition_group_key(stem)
            if base not in groups:
                groups[base] = []
            groups[base].append(path)

        return [sorted(paths) for paths in groups.values()]

    @staticmethod
    def _build_single_contrast_index(
        groups_list: list[list[Path]],
    ) -> list[dict]:
        """Create one entry per contrast group (no cross-contrast pairing).

        Each entry contains all repetition paths for ONE contrast of ONE patient.
        Used for contrast-agnostic training where each contrast is treated
        independently.

        Args:
            groups_list: List of Path lists from _build_groups.

        Returns:
            List of dicts with keys:
                - ``paths``: list of repetition file paths for this contrast
                - ``contrast``: contrast name (T1, T2, FLAIR, etc.)
                - ``patient_id``: patient identifier
        """
        entries = []
        for group in groups_list:
            if not group:
                continue
            stem = group[0].name.split(".")[0][:-2]  # strip repetition suffix
            parts = stem.split("_")
            if len(parts) >= 2:
                patient_id = parts[0]
                contrast = parts[1].upper()
            else:
                patient_id = stem
                contrast = "UNKNOWN"
            entries.append(
                {
                    "paths": group,
                    "contrast": contrast,
                    "patient_id": patient_id,
                }
            )
        entries.sort(key=lambda x: f"{x['patient_id']}_{x['contrast']}")
        return entries

    @staticmethod
    def _build_federated_pairs(
        groups_list: list[list[Path]],
        source: str = _FEDERATED_SOURCE_CONTRAST,
        targets: frozenset[str] | None = None,
    ) -> list[dict]:
        """Pair the source contrast with each target contrast from the grouped list.

        Args:
            groups_list: List of Path lists from _build_groups.
            source: Source contrast; patients without it are skipped.
            targets: Target contrasts; ``None`` pairs every other contrast.

        Returns:
            List of dicts {"source": [...T1 paths...], "target": [...Target paths...], "patient_id": str, "target_contrast": str, "source_contrast": "T1"}.
        """
        # Catalog groups by PatientID and Contrast
        inventory = {}
        for group in groups_list:
            if not group:
                continue
            # E.g., baseline stem: "2022091411_FLAIR" -> Patient: "2022091411", Contrast: "FLAIR"
            stem = group[0].name.split(".")[0][:-2]  # strip repetition "01"
            parts = stem.split("_")
            if len(parts) >= 2:
                patient_id = parts[0]
                contrast = parts[1].upper()
                if patient_id not in inventory:
                    inventory[patient_id] = {}
                inventory[patient_id][contrast] = group
            else:
                # Fallback if the naming convention is violated
                patient_id = stem
                contrast = "UNKNOWN"
                if patient_id not in inventory:
                    inventory[patient_id] = {}
                inventory[patient_id][contrast] = group

        federated_pairs = []
        for patient_id, contrasts in inventory.items():
            if source not in contrasts:
                continue

            source_group = contrasts[source]
            for contrast_name, target_group in contrasts.items():
                if contrast_name == source or (
                    targets is not None and contrast_name not in targets
                ):
                    continue
                federated_pairs.append(
                    {
                        "source": source_group,
                        "target": target_group,
                        "patient_id": patient_id,
                        "target_contrast": contrast_name,
                        # Both sides name their own contrast. Without this the
                        # source files are attributable only by re-parsing the
                        # filename, and a reader that keyed them off
                        # ``target_contrast`` instead would file every T1
                        # repetition under T2/FLAIR.
                        "source_contrast": source,
                    }
                )

        # Sort for deterministic ordering
        federated_pairs.sort(key=lambda x: f"{x['patient_id']}_{x['target_contrast']}")
        return federated_pairs

    @classmethod
    def from_manifest(
        cls,
        manifest_path: str | Path,
        data_root: str | Path = "",
        **kwargs,
    ) -> "M4RawRepetitionDataset":
        """Build dataset from a pre-built pickle manifest.

        The manifest format is the same as produced by the index builder
        (version 1 or version 2).

        Args:
            manifest_path: Path to ``.pkl`` manifest file.
            data_root: Base directory to resolve relative paths. If empty,
                uses the ``data_root`` embedded in the manifest (v2) or the
                current working directory.
            **kwargs: Forwarded to ``__init__``.

        Returns:
            Configured :class:`M4RawRepetitionDataset` instance.
        """
        from spectramr.data.metadata.path_resolver import PathResolver

        manifest_path = Path(manifest_path)
        with open(manifest_path, "rb") as fh:
            manifest = pickle.load(fh)

        if isinstance(manifest, dict):
            raw_files = manifest.get("files", [])
            manifest_root = manifest.get("data_root", "")
        else:
            raw_files = manifest
            manifest_root = ""

        effective_root = Path(data_root or manifest_root or ".")

        h5_files = []
        for entry in raw_files:
            rel_path = entry.get("path", "") if isinstance(entry, dict) else str(entry)
            if not rel_path:
                continue
            resolved = PathResolver.resolve(str(rel_path))
            candidate = Path(resolved)
            if not candidate.is_absolute():
                candidate = effective_root / candidate
            if candidate.exists():
                h5_files.append(candidate)
            else:
                logger.debug("[M4Raw] File not found, skipping: %s", candidate)

        logger.info(
            "[M4Raw] Loaded %d valid H5 paths from manifest %s",
            len(h5_files),
            manifest_path,
        )
        return cls(h5_files, **kwargs)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """__len__.

        Returns:
            int: Description.
        """
        return len(self._groups)

    def __getitem__(self, idx: int) -> tio.Subject:
        """Return a TorchIO Subject with *input*, *target*, and *contrast_idx*.

        Subject fields
        --------------
        ``input`` : :class:`tio.ScalarImage`
            K-space from a single repetition, shape ``(2, H, W, Slices)``.
        ``target`` : :class:`tio.ScalarImage`
            Averaged k-space from all repetitions, shape ``(2, H, W, Slices)``.
        ``kspace`` : the fully sampled reference, the same tensor as ``target``;
            the coil-map estimator and renderers read it there.
        ``measured_kspace`` : the measurement the model is handed, the same tensor
            as ``input`` (after an r2r / null-space split, the split half). A
            masking transform downstream overwrites it with the masked input.
        ``contrast_idx`` : ``torch.long`` scalar (0=T1, 1=T2, 2=FLAIR, 3=PD).
        """
        max_retries = min(5, len(self._groups))
        probe = idx
        for attempt in range(max_retries):
            probe_idx = probe % len(self._groups)
            try:
                result = self._load_item(probe_idx)
                if result is None:
                    continue  # single-contrast branch returned None for corrupt file
                return result
            except _SkipSampleError as exc:
                logger.warning(
                    "[M4Raw] Skipping idx=%d (attempt %d/%d): %s",
                    probe_idx,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                # Leave the failing record's GROUP. A group record steps by one
                # (as before); a slice record's neighbours are the other slices
                # of the same unreadable files, and five retries inside one
                # group would read as the systemic failure below.
                probe += retry_step(self._groups[probe_idx])
        # All retries failed. A single occasionally-corrupt file is handled by
        # the retry-to-next above; reaching here means EVERY probed group failed
        # — almost always a systemic problem (wrong/absent ``data_root``, an
        # unmounted cluster share, a broken manifest), not one bad file. The old
        # behaviour returned a zero-filled synthetic ``tio.Subject`` so training
        # "could continue" — which silently trained the whole run on meaningless
        # all-zero input/target with no hard failure (the "loads zero/random"
        # facade, pitfall #9/#16). Fail loud instead so the data problem surfaces
        # at once rather than after a wasted run.
        raise RuntimeError(
            f"[M4Raw] All {max_retries} retries exhausted for idx={idx}: every "
            f"probed repetition group raised _SkipSampleError. This is a systemic "
            f"data-loading failure (check data.data_root resolves on this host, "
            f"the manifest is present, and the k-space files are readable) — "
            f"refusing to substitute a zero-filled sample and train on garbage."
        )

    def _scan_covariance(
        self,
        kspace: torch.Tensor,
        scan_key: str,
        *,
        fit_kspace: Callable[[], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """This scan's ``Sigma_n`` (:func:`fit_scan_noise_covariance`), cached per group.

        ``fit_kspace`` supplies the k-space to fit on a cache miss; ``None`` fits
        ``kspace`` itself, which is the whole volume on the per-group route. A
        slice record must pass :meth:`_sigma_fit_source`, or the fit would use
        that one slice and differ with whichever slice a worker loaded first.

        A failed fit RAISES rather than falling back to the committed matrix
        (pitfall #9): serving the constant would make an A-vs-E comparison
        vacuous while the YAML still said ``self_calibrated``.
        """
        cached = self._r2r_cov_cache.get(scan_key)
        if cached is not None:
            return cached
        stored = (
            None
            if self._noise_covariance_store is None
            else self._noise_covariance_store.get(scan_key)
        )
        if stored is not None:
            self._r2r_cov_cache[scan_key] = stored
            return stored
        if fit_kspace is not None:
            kspace = fit_kspace()
        covariance, fitted = fit_scan_noise_covariance(
            kspace, support_policy=self.noise_fit_support_policy
        )
        self._r2r_cov_cache[scan_key] = covariance
        logger.info(
            "[M4Raw] self-calibrated Sigma_n for scan %s: trace=%.4g over %d coils "
            "(%d central slice(s), sampled PE fraction %.3f)",
            scan_key,
            float(covariance.diagonal().real.sum()),
            int(covariance.shape[0]),
            fitted,
            float(_pe_support(kspace).float().mean()),
        )
        return covariance

    @staticmethod
    def _sigma_fit_source(
        input_path: Path, input_kspace: torch.Tensor, record: dict[str, Any]
    ) -> Callable[[], torch.Tensor]:
        """The k-space a scan's ``Sigma_n`` is fitted on, read only on a cache miss.

        A group record already holds the input volume. A slice record reads the
        input repetition's :func:`sigma_fit_window` slices from disk, so both
        routes fit the same central slices and every slice of a scan draws from
        one matrix.
        """
        if record.get("slice_index") is None:
            return lambda: input_kspace

        def central() -> torch.Tensor:
            try:
                return load_sigma_fit_slices(input_path, int(record["num_slices"]))
            except (OSError, KeyError) as exc:
                raise _SkipSampleError(
                    f"[M4Raw] {input_path.name}: the central slices "
                    f"{list(sigma_fit_window(int(record['num_slices'])))} the "
                    f"scan's Sigma_n is fitted on could not be read: {exc}"
                ) from exc

        return central

    def _sampler_for_scan(
        self,
        kspace: torch.Tensor,
        scan_key: str,
        *,
        fit_kspace: Callable[[], torch.Tensor] | None = None,
    ) -> Any:
        """The R2R sampler for this scan: the committed matrix or its own fit."""
        if self.r2r_covariance_source == "committed":
            if self._r2r_sampler is None:  # pragma: no cover - guarded in __init__
                raise RuntimeError("[M4Raw] r2r requested but no sampler was built.")
            return self._r2r_sampler
        from spectramr.infrastructure.physics.m4raw_noise import M4RawNoiseSampler

        covariance = self._scan_covariance(kspace, scan_key, fit_kspace=fit_kspace)
        return M4RawNoiseSampler(covariance=covariance, alpha=self.r2r_alpha)

    def _recorrupt_r2r(
        self,
        kspace: torch.Tensor,
        scan_key: str = "",
        *,
        maps: torch.Tensor | None = None,
        fit_kspace: Callable[[], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ONE acquired repetition into an R2R ``(input, target)`` pair.

        ``maps`` is accepted so both split methods share one call and is unused:
        the draw needs a covariance, not a coil geometry.

        ``input = y + alpha*z``, ``target = y - z/alpha`` with
        ``z ~ CN(0, Sigma_n)`` drawn fresh per sample, so
        ``Cov(input, target) = Sigma_n - Sigma_z = 0`` and the L2 minimiser is the
        supervised one -- from a single excitation, with no clean reference. The
        identity is an L2 one: under L1 the minimiser is the median of
        ``x + e2`` given the input, not the median of ``x``.
        Noise goes only into acquired columns (:func:`_pe_support`).

        The draw uses the global RNG, which ``DataLoader`` seeds per worker, so a
        run is reproducible from ``run.seed`` and every sample sees a different
        recorruption across epochs (which is the point -- it is an augmentation
        over the noise, not a fixed second copy).
        """
        support = _pe_support(kspace)
        return self._sampler_for_scan(kspace, scan_key, fit_kspace=fit_kspace).recorrupt(
            kspace, support_mask=support
        )

    def _nullspace_split(
        self,
        kspace: torch.Tensor,
        scan_key: str = "",
        *,
        maps: torch.Tensor | None = None,
        fit_kspace: Callable[[], torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ONE repetition along its own coil null space (``nullspace_r2r``).

        No noise is drawn: the two halves are orthogonal projections of the
        whitened coil vector, independent given the image (see
        :mod:`~spectramr.infrastructure.physics.coil_split_r2r`). ``maps`` are
        the group's precomputed ones under ``coil_maps_root``; without them the
        maps are estimated per slice from this repetition on every load, ~2.3 s
        per 256x256 slice in a one-thread worker, since the Queue respawns its
        workers each epoch and a worker cache would never be read back.
        """
        from spectramr.infrastructure.physics.coil_split_r2r import split_kspace, whitener
        from spectramr.infrastructure.physics.m4raw_noise import single_repetition_covariance
        from spectramr.infrastructure.physics.scan_coil_calibration import (
            as_slice_batch,
            calibrate_scan,
        )

        support = _pe_support(kspace)
        slices = as_slice_batch(kspace)
        if maps is None:
            maps = calibrate_scan(slices, support, fit_covariance=False).maps
        elif maps.shape != slices.shape:
            raise ValueError(
                f"[M4Raw] coil maps {tuple(maps.shape)} do not describe k-space "
                f"{tuple(slices.shape)}."
            )
        # Keys are VALID_WHITENING, checked in __init__; lazy, so only the
        # declared source is ever computed.
        sources: dict[str, Callable[[], torch.Tensor | None]] = {
            "none": lambda: None,
            "committed": single_repetition_covariance,
            "self_calibrated": lambda: self._scan_covariance(
                kspace, scan_key, fit_kspace=fit_kspace
            ),
        }
        first, second = split_kspace(
            slices,
            maps,
            support,
            whitening=whitener(sources[self.r2r_whitening](), slices.shape[1]),
            angle_deg=self.r2r_split_angle_deg,
        )
        return first.reshape(kspace.shape), second.reshape(kspace.shape)

    def _refuse_identity_target(
        self, kspace_reps: list[torch.Tensor], n_paths: int, context: str
    ) -> None:
        """Skip a group whose NEX target would be its own input.

        With one surviving repetition the average is repetition 0, which is also the
        input, so the sample would teach the identity while the config says
        denoising. ``__getitem__`` retries the next group. The split modes are exempt:
        a one-repetition group is the case they exist for.
        """
        if (
            self.use_repetitions
            and self.target_mode not in _SINGLE_REP_SPLIT_MODES
            and len(kspace_reps) < 2
        ):
            raise _SkipSampleError(
                f"[M4Raw] {context}: only {len(kspace_reps)} of {n_paths} repetitions "
                "loaded. The NEX target would be the input itself (SNR boost "
                "sqrt(1)=1.0), so the sample teaches the identity rather than "
                "denoising. Fix the unreadable file(s), or set "
                "data.use_repetitions: false to declare that intent."
            )

    def _nex_target(
        self,
        kspace_reps: list[torch.Tensor],
        context: str,
        input_index: int = 0,
        shifts: list[tuple[float, float]] | None = None,
    ) -> torch.Tensor:
        """The target for one repetition group, per ``target_mode``.

        ``kspace_reps[input_index]`` is the input. It is moved to the front, because
        :func:`_average_reps` anchors the phase frame to (and leave-one-out excludes)
        position 0; with ``input_index == 0`` the list is passed through unchanged.
        Leave-one-out drops the input only when at least two repetitions remain;
        otherwise ``nex_fallback`` decides between refusing and the all-reps average.
        ``shifts``, one per repetition, registers them onto the input first.
        """
        if input_index:
            order = [input_index] + [i for i in range(len(kspace_reps)) if i != input_index]
            kspace_reps = [kspace_reps[i] for i in order]
            shifts = None if shifts is None else [shifts[i] for i in order]
        if not (self.use_repetitions and len(kspace_reps) > 1):
            return kspace_reps[0].clone()
        min_reps = _min_reps_for_leave_one_out(self.target_mode)
        loo_requested = self.nex_target_exclude_input
        exclude = 0 if (loo_requested and len(kspace_reps) >= min_reps) else None
        if loo_requested and exclude is None:
            if self.nex_fallback == "error" or self.target_mode == "rep_pair":
                # rep_pair is refused regardless: the all-reps fallback would serve
                # the input as its own target.
                raise ValueError(
                    f"[M4Raw] {context}: leave-one-out NEX target requested but only "
                    f"{len(kspace_reps)} repetition(s) loaded (needs >= {min_reps}); "
                    "data.nex_fallback='error' refuses the correlated all-reps "
                    "reference. Declare data.nex_fallback: all_reps to accept it."
                )
            self._note_loo_declined(len(kspace_reps))
        return _average_reps(kspace_reps, self.target_mode, exclude_index=exclude, shifts=shifts)

    def _group_shifts(
        self, rep_paths: list[Path], kspace_reps: list[torch.Tensor], record: dict[str, Any]
    ) -> list[tuple[float, float]] | None:
        """Each loaded repetition's displacement against the first, or ``None`` unregistered.

        Estimated on the :func:`registration_window` slices of the volume, read from
        disk on the slice-level route, so both routes and every slice of a group use
        one set of shifts. ``rep_paths`` are the files ``kspace_reps`` came from.
        """
        if self.nex_registration == "none" or len(kspace_reps) == 1:
            return None
        key = tuple(str(p) for p in rep_paths)
        cached = self._shift_cache.get(key)
        if cached is not None:
            return cached
        stored = None if self._shift_store is None else self._shift_store.get(rep_paths)
        if stored is not None:
            self._shift_cache[key] = stored
            return stored
        if record.get("slice_index") is None:
            window = registration_window(kspace_reps[0].shape[0])
            volumes = [r[window.start : window.stop] for r in kspace_reps]
        else:
            window = registration_window(int(record["num_slices"]))
            try:
                volumes = [torch.cat([_load_kspace(p, s) for s in window]) for p in rep_paths]
            except (OSError, KeyError, IndexError) as exc:
                raise _SkipSampleError(
                    f"[M4Raw] {rep_paths[0].name}: the slices {list(window)} the "
                    f"repetitions are registered on could not be read: {exc}"
                ) from exc
        shifts = group_shifts(volumes)
        self._shift_cache[key] = shifts
        return shifts

    def _note_loo_declined(self, n_reps: int) -> None:
        """Report that the leave-one-out NEX gate declined, once per rep count.

        The arm asked for an unbiased (leave-one-out) target and is getting the
        all-reps average instead, so its input's own noise sits inside its
        target at amplitude 1/N. That changes the reference PSNR/SSIM grade
        against, which is exactly the comparison `.claude/rules/data.md` says to
        flip cohort-wide rather than per-arm. Declining silently made the
        substitution invisible in every log (#695).
        """
        if n_reps in self._loo_declined_reported:
            return
        self._loo_declined_reported.add(n_reps)
        logger.warning(
            "[M4Raw] nex_target_exclude_input=True but this group has %d "
            "repetition(s); leave-one-out needs >=3 (it would leave a single "
            "noisy rep). Falling back to the all-reps average for these "
            "samples: their target is NOT unbiased and is not comparable to "
            "the >=3-rep samples in the same run.",
            n_reps,
        )

    def _load_item(self, idx: int) -> tio.Subject | None:
        """Internal loader — all actual loading logic (previously in __getitem__)."""
        # Prevent OpenMP DataLoader deadlocks when fft ops run in workers —
        # once per process, not per item.
        _clamp_worker_threads_once()

        # (Removed the per-__getitem__ deepcopy-equality guard on self._groups —
        # it was debug instrumentation that deep-copied the full group index at
        # init and ran a full list-equality compare every load. The historical
        # path-corruption bug it watched for is fixed by storing paths as
        # strings. backlog_wasted_compute_audit_2026_05_29 DATA-7.)

        stored_paths = self._groups[idx]
        # ``None`` on the per-group index; the record's slice on the slice-level
        # route, threaded into every repetition read below (#1757).
        slice_index = stored_paths.get("slice_index")
        if stored_paths and stored_paths.get("target"):
            first_str_path = stored_paths["target"][0]
            # Verbose per-item path logging only when DEBUG is enabled — the
            # ``os.getcwd()`` + repr work ran unconditionally on every load
            # just to feed debug lines.
            if logger.isEnabledFor(logging.DEBUG):
                import os

                logger.debug(
                    f"[M4Raw __getitem__] ACCESSING idx={idx}, "
                    f"_groups[{idx}]['target'][0]={repr(first_str_path)[:100]}"
                )
                logger.debug(f"[M4Raw __getitem__] CWD: {os.getcwd()}")
                logger.debug(
                    f"[M4Raw __getitem__] idx={idx}, "
                    f"_groups[{idx}]['target'][0] RAW STRING: {first_str_path!r}"
                )
                logger.debug(
                    f"[M4Raw __getitem__] Type of stored_paths['target'][0]: {type(first_str_path)}"
                )

            # Corruption guard (fail-loud, always on — cheap string check).
            if "/data/" in first_str_path and "/databases" not in first_str_path:
                import os

                logger.error(
                    f"[M4Raw __getitem__] ⚠️ CORRUPTED STRING PATH DETECTED: {first_str_path}"
                )
                logger.error(
                    "[M4Raw __getitem__] Should be under /databases/m4raw/data/multicoil_train/"
                )
                logger.error(f"[M4Raw __getitem__] Current working directory: {os.getcwd()}")

        # BUG FIX: Convert string paths back to Path objects
        # =====================================================================
        # SINGLE-CONTRAST MODE: one contrast per sample (no cross-contrast pairing)
        # =====================================================================
        if self.single_contrast:
            rep_paths = [Path(p) for p in stored_paths["paths"]]
            contrast_name = stored_paths.get("contrast", "UNKNOWN")

            # Load all repetitions for this contrast. Raises _SkipSampleError when
            # none load -- the protocol __getitem__ already implements (warn,
            # retry the next group, then fail systemically). It used to return
            # None "for collate to filter", but m4raw selects
            # ImageCollateStrategy, which has no _filter_none, so the None died
            # as `TypeError: 'NoneType' object is not subscriptable` naming
            # neither the index nor the file (B14).
            context = f"idx={idx} contrast={contrast_name}"
            loaded: list[int] = []
            kspace_reps = _load_reps_or_skip(
                rep_paths,
                context + (f" slice={slice_index}" if slice_index is not None else ""),
                slice_index,
                loaded,
            )

            self._refuse_identity_target(kspace_reps, len(rep_paths), context)
            # Repetition 0 unless the training split declared `input_repetition:
            # random`; the target below is aligned to whichever is chosen.
            k = self._draw_input(len(kspace_reps))

            # The group's coil maps, read once per load when the arm declares a
            # map root; the split and every consumer downstream share them.
            maps, maps_source = (
                self._coil_maps.maps_for(rep_paths[0].stem, kspace_reps, slice_index=slice_index)
                if self._coil_maps is not None
                else (None, None)
            )

            # Input = one repetition (noisy), Target = averaged reps (high-SNR)
            input_kspace = kspace_reps[k].clone()
            # The repetition as acquired, kept when the split below replaces the
            # input and this dataset's own chain measures the k-space scale on it
            # (deployment sees y, not y + alpha*z). Nothing else reads the key.
            measured_kspace = None
            shifts = None
            # Sigma_n is a receive-chain property, so it is keyed on the group's
            # repetition 0, the file it is fitted on, never on the slice.
            scan_key = str(rep_paths[0])
            fit_kspace = self._sigma_fit_source(rep_paths[0], kspace_reps[0], stored_paths)
            if self.target_mode in _SINGLE_REP_SPLIT_MODES:
                if consumes_measured_input(self.transform):
                    measured_kspace = kspace_reps[0]
                # Both halves come from THIS single repetition, split here on raw
                # k-space: registry transforms run AFTER normalization, which
                # would rescale an r2r draw away from Sigma_n.
                split = getattr(self, _SPLIT_METHODS[self.target_mode])
                input_kspace, target_kspace = split(
                    input_kspace, scan_key=scan_key, maps=maps, fit_kspace=fit_kspace
                )
            else:
                loaded_paths = [rep_paths[i] for i in loaded]
                shifts = self._group_shifts(loaded_paths, kspace_reps, stored_paths)
                # Under a Noise2Noise pairing the strategy forms the target from the
                # stack below, so no averaged target is built for this split.
                target_kspace = (
                    None
                    if self.n2n_pairing is not None
                    else self._nex_target(kspace_reps, context, input_index=k, shifts=shifts)
                )

            # Slot 0 is the served input; the rest are the split's target half or
            # every other repetition in the input's frame, as `input_alt` is built.
            rep_stack: list[torch.Tensor] | None = None
            if self.n2n_pairing is not None:
                if self.target_mode in _SINGLE_REP_SPLIT_MODES:
                    rep_stack = [input_kspace, target_kspace]
                else:
                    rep_stack = [input_kspace]
                    for j, other in enumerate(kspace_reps):
                        if j == k:
                            continue
                        if shifts is not None:
                            other = _register_to_anchor(
                                [kspace_reps[k], other], [shifts[k], shifts[j]], 0
                            )[1]
                        rep_stack.append(_phase_align(other, kspace_reps[k]))
                target_kspace = None

            # A second noisy view of the input, in the input's frame, so a
            # consumer can measure what the model does across repetitions.
            alt_kspace = None
            if self._emits_alternate_input():
                j = (k + 1) % len(kspace_reps)
                alt = kspace_reps[j]
                if shifts is not None:
                    alt = _register_to_anchor([kspace_reps[k], alt], [shifts[k], shifts[j]], 0)[1]
                alt_kspace = _phase_align(alt, kspace_reps[k])

            # Apply coil processing (RSS combine if needed)
            if input_kspace.dim() == 4 and self.coil_processing_mode == "rss":
                input_kspace = _rss_combine(input_kspace)
                if target_kspace is not None:
                    target_kspace = _rss_combine(target_kspace)
                if rep_stack is not None:
                    rep_stack = [_rss_combine(r) for r in rep_stack]
                if alt_kspace is not None:
                    alt_kspace = _rss_combine(alt_kspace)
                if measured_kspace is not None:
                    measured_kspace = _rss_combine(measured_kspace)

            # NOTE: k-space normalization is NOT applied here. The dataset
            # matches and serves; morphing belongs to the transform layer
            # (``KSpaceNormalizationTransform``, canonical home
            # ``data/transforms/``). This block used to run the Parseval scale
            # inline AND the builder appended the transform for the same
            # ``data.normalize_kspace`` flag, so every arm got a double
            # percentile divide + double log1p and the transform overwrote
            # ``kspace_scale`` — leaving the normalization non-invertible.
            # The Parseval scale now lives in ``kspace_image_domain_scale``
            # (select it with ``data.kspace_scale_domain: image``).

            # Convert to TorchIO format — NO cross-contrast concat
            def _to_torchio_stacked(k: torch.Tensor) -> torch.Tensor:
                """Convert (S, C, H, W) complex → (2C, H, W, S) real interleaved."""
                if k.dim() == 3:
                    k = k.unsqueeze(1)  # (S, 1, H, W)
                real = k.real
                imag = k.imag
                interleaved = torch.empty(
                    (k.shape[0], k.shape[1] * 2, k.shape[2], k.shape[3]),
                    dtype=torch.float32,
                )
                interleaved[:, 0::2] = real
                interleaved[:, 1::2] = imag
                return interleaved.permute(1, 2, 3, 0)  # (2C, H, W, S)

            input_t = _to_torchio_stacked(input_kspace)

            contrast_idx = _extract_contrast_idx(rep_paths[0].name)
            affine = np.eye(4)
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=input_t, affine=affine),
                # Published explicitly because the batch context otherwise falls back
                # to `kspace`, the target, and hands it to the model as y (#2761).
                # Under r2r it is the split input, not the acquired repetition, since
                # the two together determine the target exactly.
                measured_kspace=tio.ScalarImage(tensor=input_t, affine=affine),
            )
            if rep_stack is None:
                target_t = _to_torchio_stacked(target_kspace)
                subject.add_image(tio.ScalarImage(tensor=target_t, affine=affine), "target")
                subject.add_image(tio.ScalarImage(tensor=target_t, affine=affine), "kspace")
            else:
                # No target is served: nothing downstream of the loader, ESPIRiT
                # included, sees target information for this training sample.
                stack_t = torch.cat([_to_torchio_stacked(r) for r in rep_stack], dim=0)
                subject.add_image(tio.ScalarImage(tensor=stack_t, affine=affine), REPETITION_STACK_KEY)
            if alt_kspace is not None:
                subject.add_image(
                    tio.ScalarImage(tensor=_to_torchio_stacked(alt_kspace), affine=affine),
                    "input_alt",
                )
            if measured_kspace is not None:
                subject.add_image(
                    tio.ScalarImage(tensor=_to_torchio_stacked(measured_kspace), affine=affine),
                    MEASURED_INPUT_KEY,
                )
            if maps is not None:
                _attach_coil_maps(subject, maps, maps_source, self._coil_maps.calibration, affine)
            if self._reference_maps is not None:
                reference, _ = self._reference_maps.maps_for(
                    rep_paths[0].stem, kspace_reps, slice_index=slice_index
                )
                subject.add_image(
                    tio.ScalarImage(tensor=reference.permute(1, 2, 3, 0).contiguous(), affine=affine),
                    REFERENCE_MAPS_KEY,
                )
            if self._publishes_noise_covariance():
                # The scan's own Sigma_n in raw units, so a validation-time
                # recorruption draws from the source the training split used.
                subject["noise_covariance"] = self._scan_covariance(
                    kspace_reps[0], scan_key, fit_kspace=fit_kspace
                )
            subject["contrast_idx"] = torch.tensor(contrast_idx, dtype=torch.long)
            subject["file_id"] = rep_paths[0].stem
            # The identities latent in that name, made explicit. `subject_id` is
            # the INDEPENDENCE unit a risk certificate needs: two contrasts of
            # one patient share an anatomy, so counting them separately inflates
            # n (#1707). On this corpus the Hoeffding half-width is 0.0100 per
            # slice, 0.0693 per repetition group and 0.1200 per subject -- a
            # certificate built on the wrong unit reads 12x tighter than it is.
            _identity = parse_m4raw_file_id(rep_paths[0].stem)
            subject["subject_id"] = _identity.subject
            subject["repetition_group"] = _identity.repetition_group
            # The served input's repetition, which `input_repetition: random` makes
            # differ from the group's first file.
            subject["repetition_index"] = parse_m4raw_file_id(rep_paths[loaded[k]].stem).repetition
            # The contrast NAME, beside the index. `contrast_idx` alone forces
            # every consumer to carry a copy of the 0=T1/1=T2/2=FLAIR mapping,
            # and the per-case CSV writer is a generic reporting component that
            # has no business knowing M4Raw's vocabulary. Read off the index
            # record rather than re-parsed from the filename -- `stem.split("_")`
            # is already spelled in `_build_index` and `_build_federated_pairs`
            # and a third copy would drift (non-negotiable 17).
            subject["contrast"] = str(contrast_name)
            # Identity scale: the data leaves the dataset unnormalized. When
            # normalization is enabled, KSpaceNormalizationTransform overwrites
            # this with the scale it actually applied, so the value published
            # here always matches the tensor served alongside it.
            subject["kspace_scale"] = torch.tensor(1.0)
            # ...and say so explicitly. An identity scale is indistinguishable
            # from a real one by presence alone, which is how a consumer came to
            # read "kspace_scale exists" as "the tensors are normalized" and skip
            # normalization entirely. The transform overwrites this with True.
            subject["kspace_normalized"] = False
            if slice_index is not None:
                # Which slice of the volume this depth-1 subject is; the
                # per-group path has no such fact and publishes no key.
                subject["slice_index"] = torch.tensor(slice_index, dtype=torch.long)

            if self.transform is not None:
                subject = self.transform(subject)
            return subject

        # =====================================================================
        # CROSS-CONTRAST MODE (original): pair source (T1) with target (T2/FLAIR)
        # =====================================================================
        # Federated architecture requires paths for both source and target.
        source_paths = [Path(p) for p in stored_paths["source"]]
        target_paths = [Path(p) for p in stored_paths["target"]]

        # DEBUG: Log after conversion
        if source_paths and target_paths:
            logger.debug(
                f"[M4Raw __getitem__] After Path() conversion Source: {source_paths[0]} Target: {target_paths[0]}"
            )

        # ---- Load and fuse source & target repetitions ------------------------------------
        def process_reps(rep_paths: list[Path], side: str) -> tuple[torch.Tensor, torch.Tensor]:
            """Input (repetition 0) and NEX target of one side of the pair.

            The target goes through the same builder as the single-contrast route,
            so ``target_mode`` and ``nex_target_exclude_input`` hold here too (#2450).
            """
            context = f"idx={idx} {side}={rep_paths[0].stem[:-2]}"
            loaded: list[int] = []
            kspace_reps = _load_reps_or_skip(rep_paths, context, slice_index, loaded)
            self._refuse_identity_target(kspace_reps, len(rep_paths), context)
            input_kspace = kspace_reps[0].clone()
            shifts = self._group_shifts(
                [rep_paths[i] for i in loaded], kspace_reps, stored_paths
            )
            target_kspace = self._nex_target(kspace_reps, context, shifts=shifts)

            if kspace_reps[0].dim() == 4 and self.coil_processing_mode == "rss":
                input_kspace = _rss_combine(input_kspace)
                target_kspace = _rss_combine(target_kspace)

            return input_kspace, target_kspace

        if self.target_mode in _SINGLE_REP_SPLIT_MODES:
            # The cross-contrast route pairs a SOURCE contrast with a different
            # TARGET contrast; a split's two halves are one contrast split
            # against itself. Composing them would silently produce a
            # contrast-translation target the arm never asked for.
            raise ValueError(
                f"[M4Raw] target_mode={self.target_mode!r} is a single-contrast "
                "denoising target and has no meaning in the cross-contrast route "
                "(which pairs two different contrasts). Declare "
                "data.pairing.single_contrast: true."
            )

        source_in, source_tgt = process_reps(source_paths, "source")
        target_in, target_tgt = process_reps(target_paths, "target")

        # Log SNR improvement if multiple reps
        if self.use_repetitions:
            logger.debug(
                f"[M4Raw __getitem__] Averaged Target Reps: {len(target_paths)} / Source Reps: {len(source_paths)}"
            )

        # ---- Normalization is the transform layer's job ----------------------
        # The Parseval-compliant volume scale that used to run here now lives in
        # ``kspace_image_domain_scale`` and is applied by
        # ``KSpaceNormalizationTransform``. Running it here as well as in the
        # transform (both driven by ``data.normalize_kspace``) double-divided
        # and double-log1p'd the k-space. See the single-contrast branch above.

        # RSS-combined k-space is (Slices, H, W); give it the coil axis the
        # channel layout below expects.
        if source_in.dim() == 3:
            source_in = source_in.unsqueeze(1)
            target_in = target_in.unsqueeze(1)
            source_tgt = source_tgt.unsqueeze(1)
            target_tgt = target_tgt.unsqueeze(1)

        # Convert to TorchIO format (Real/Imag, Coils, H, W, Slices)
        # However, TorchIO is (Channels, H, W, Depth). We will treat Coils*2 as Channels.
        def _to_torchio_multicoil_stacked(k):
            # k: (Slices, Coils, H, W) -> flatten complex to real/imag -> (Slices, 2*Coils, H, W)
            """_to_torchio_multicoil_stacked.

            Args:
                k (Any): Description.
            Returns:
                Any: Description.
            """
            real = k.real
            imag = k.imag

            # Interleave real and imaginary per coil: R1, I1, R2, I2...
            interleaved = torch.empty(
                (k.shape[0], k.shape[1] * 2, k.shape[2], k.shape[3]),
                dtype=torch.float32,
            )
            interleaved[:, 0::2] = real
            interleaved[:, 1::2] = imag

            return interleaved.permute(1, 2, 3, 0)  # (2*Coils, H, W, Slices)

        if self.source_layout != "stacked":
            input_t = _to_torchio_multicoil_stacked(target_in)
            target_t = _to_torchio_multicoil_stacked(target_tgt)
        else:
            input_t = _to_torchio_multicoil_stacked(torch.cat([source_in, target_in], dim=1))
            target_t = _to_torchio_multicoil_stacked(torch.cat([source_tgt, target_tgt], dim=1))

        contrast_idx = _extract_contrast_idx(target_paths[0].name)

        affine = np.eye(4)
        images = {
            "input": tio.ScalarImage(tensor=input_t, affine=affine),
            "target": tio.ScalarImage(tensor=target_t, affine=affine),
            # The strategy's coil-map estimator and several renderers read the
            # fully sampled target under this alias.
            "kspace": tio.ScalarImage(tensor=target_t, affine=affine),
            # Without it the batch context resolves the measurement from `kspace`,
            # the target (#2761).
            "measured_kspace": tio.ScalarImage(tensor=input_t, affine=affine),
        }
        if self.source_layout == "separate":
            # The source is the acquired repetition, not its NEX average: a
            # deployed model receives one fully sampled source scan.
            images["kspace_source"] = tio.ScalarImage(
                tensor=_to_torchio_multicoil_stacked(source_in), affine=affine
            )
        subject = tio.Subject(**images)
        # Scalar metadata — preserved through TorchIO Queue patch sampling
        # and stacked automatically by ImageCollateStrategy
        subject["contrast_idx"] = torch.tensor(contrast_idx, dtype=torch.long)
        subject["file_id"] = target_paths[0].stem
        # See the single-contrast branch: `subject_id` is the independence unit.
        _identity = parse_m4raw_file_id(target_paths[0].stem)
        subject["subject_id"] = _identity.subject
        subject["repetition_group"] = _identity.repetition_group
        subject["repetition_index"] = _identity.repetition
        # The TARGET contrast: what this sample is reconstructed *into*. The
        # source side is T1 by construction for every federated pair, so naming
        # it here would label all of them "T1" and make the column useless.
        subject["contrast"] = str(stored_paths.get("target_contrast", "UNKNOWN"))

        if self.source_layout == "stacked":
            # Source and target carry the same coil count, so the target starts
            # at the channel midpoint; the image renderer slices there.
            subject["federated_target_channel_start"] = torch.tensor(
                int(target_t.shape[0]) // 2, dtype=torch.long
            )
        if slice_index is not None:
            subject["slice_index"] = torch.tensor(slice_index, dtype=torch.long)

        if self.transform is not None:
            subject = self.transform(subject)

        return subject

    def dry_iter(self) -> list[tio.Subject]:
        """Return lightweight Subject shells for TorchIO Queue compatibility.

        TorchIO Queue calls ``dry_iter()`` to compute ``iterations_per_epoch``
        without loading voxel data.  It requires a ``Sequence[Subject]``
        (indexable), not a Generator, and only inspects metadata attributes
        like ``num_samples``.

        Returns:
            List of Subject objects with path metadata but no loaded images.
        """
        _stub = torch.zeros(1, 1, 1, 1)
        subjects = []
        for group in self._groups:
            # Use target paths for file_id (first target rep path)
            target_paths = group.get("target", group.get("source", []))
            file_id = Path(target_paths[0]).stem if target_paths else "unknown"
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=_stub),
                file_id=file_id,
            )
            subjects.append(subject)
        return subjects

    def provenance_counts(self) -> dict[str, Any]:
        """Per-unit counts for the run provenance record.

        The hook ``provenance.describe_dataloader`` looks for. It answers "did
        all my data arrive?" in the units the *user* has -- files on disk and
        patients -- rather than the patch/batch units a DataLoader speaks. For
        this corpus the chain is ``1024 files -> 384 groups (128 patients x 3
        contrasts)``, so a record carrying only ``batches=768`` cannot separate
        a correct run from a 25 % load failure.

        Cheap by construction: this walks ``self._groups``, which is metadata
        already in memory, and opens no HDF5 file (non-negotiable 9).

        Keys:

        ``groups``
            (patient, contrast) groups. One per ``__getitem__`` on the
            per-group index, where it equals ``__len__``; under
            ``slice_level_records`` several records share a group and the
            record count is published separately as ``slice_records``.
        ``patients``
            Distinct ``patient_id``. Deliberately *not* named ``subjects``:
            torchio's ``Queue.num_subjects`` counts index entries (groups, 384
            here) and two different numbers must never share one name.
        ``files``
            Distinct HDF5 paths in the index, deduplicated across ``paths`` /
            ``source`` / ``target``. All three are read because a
            federated-pair record carries real files in **both** ``source`` and
            ``target``; the ``rec.get("source") or rec.get("target")`` idiom at
            ``_attach_shape_metadata`` takes only the first non-empty list and
            would undercount exactly those records, by half.
        ``per_contrast``
            Groups per contrast -- not files, and one entry per pair rather
            than per side; federated records key off ``target_contrast``.
        ``files_per_contrast``
            Distinct files per contrast, deduplicated exactly as ``files`` is,
            and attributed **per side**: a federated pair contributes its
            ``source`` files to ``source_contrast`` and its ``target`` files to
            ``target_contrast``, so the same T1 group paired against both T2 and
            FLAIR is counted once, under T1.

            This is the key that makes M4Raw's repetition asymmetry visible in a
            run record (#1392). ``per_contrast`` counts *groups*, so a uniform
            ``{T1: 30, T2: 30, FLAIR: 30}`` is what a 3/3/2-repetition split
            looks like from there -- it cannot distinguish 90 files from 80.
            ``files_per_contrast`` reads ``{T1: 90, T2: 90, FLAIR: 60}`` and
            states the difference. It matters because the NEX target is an
            average over a group's repetitions: a FLAIR target averages 2 and a
            T1/T2 target averages 3, so the input's own noise persists in its
            target at 1/2 rather than 1/3, and leave-one-out
            (``nex_target_exclude_input``) is structurally impossible for FLAIR
            (:data:`_MIN_REPS_FOR_LOO`).

            ``sum(files_per_contrast.values()) == files`` holds in both index
            shapes; a divergence means a record carried files under neither
            contrast key and is a defect, not a rounding artefact.

            The older ``per_contrast`` is retained rather than renamed so run
            records written before this key stay readable side by side.
        ``slice_records``
            Only under ``slice_level_records``: ``len(self.index)``, the slice
            count the loader serves (#1757). ``groups`` keeps counting groups,
            so the two are never confused for one another.
        """
        patients: set[str] = set()
        files: set[str] = set()
        per_contrast: dict[str, int] = {}
        # Sets, not counters: the same T1 source group is paired against every
        # other contrast, so a federated index visits its files once per pair.
        files_by_contrast: dict[str, set[str]] = {}
        # Group-level counts are taken once per group. Slice records carry
        # ``group_index``; a per-group record is its own group (its position).
        seen_groups: set[int] = set()
        for position, record in enumerate(self._groups):
            group_id = int(record.get("group_index", position))
            first_of_group = group_id not in seen_groups
            seen_groups.add(group_id)
            patient_id = record.get("patient_id")
            if patient_id is not None:
                patients.add(str(patient_id))
            for key in ("paths", "source", "target"):
                for path in record.get(key) or ():
                    files.add(str(path))
            # Exactly one of these is present per record shape, and both are
            # non-empty strings, so ``or`` is safe here -- unlike the path
            # lists above, where it silently drops a populated second list.
            contrast = record.get("contrast") or record.get("target_contrast")
            if contrast is not None and first_of_group:
                per_contrast[str(contrast)] = per_contrast.get(str(contrast), 0) + 1
            # Per SIDE. ``paths``/``target`` belong to the record's own
            # contrast; ``source`` belongs to ``source_contrast``. A record that
            # declares neither is left OUT rather than folded into an arbitrary
            # bucket -- the sum invariant in the docstring is then what reports
            # it, instead of a plausible-looking wrong number.
            for key, contrast_key in (
                ("paths", "contrast"),
                ("target", "target_contrast"),
                ("source", "source_contrast"),
            ):
                side_contrast = record.get(contrast_key)
                if side_contrast is None:
                    continue
                bucket = files_by_contrast.setdefault(str(side_contrast), set())
                for path in record.get(key) or ():
                    bucket.add(str(path))
        counts: dict[str, Any] = {
            "groups": len(seen_groups),
            "patients": len(patients),
            "files": len(files),
            "per_contrast": dict(sorted(per_contrast.items())),
            "files_per_contrast": {
                contrast: len(paths) for contrast, paths in sorted(files_by_contrast.items())
            },
        }
        if self.slice_level_records:
            counts["slice_records"] = len(self._groups)
        return counts
