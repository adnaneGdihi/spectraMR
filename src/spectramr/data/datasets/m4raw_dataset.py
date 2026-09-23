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
average is per slice already, so the target path is unchanged. The queue is
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import torchio as tio
from torch.utils.data import Dataset

from spectramr.data.datasets.m4raw_identity import (
    parse_m4raw_file_id,
    repetition_group_key,
)
from spectramr.data.datasets.m4raw_slice_records import expand_to_slice_records, retry_step
from spectramr.data.io_strategies import read_h5_kspace

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
    {"complex_mean", "phase_aligned_mean", "rep_pair", "r2r"}
)

#: Where r2r draws its Sigma_n. 'committed' is the matrix measured on one M4Raw
#: study series; 'self_calibrated' fits one per scan from its own coil null
#: space, which needs no second repetition and so costs the arm no excitation.
_VALID_R2R_COVARIANCE_SOURCES: frozenset[str] = frozenset({"committed", "self_calibrated"})
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

#: Contrast on the *source* side of a federated pair. T1 by construction --
#: :meth:`M4RawRepetitionDataset._build_federated_pairs` skips any patient
#: without one and pairs that T1 against every other contrast. Declared here and
#: stamped onto the record so :meth:`_index_stats` can attribute a file to a
#: contrast by *reading* the record, rather than re-deriving it from the
#: filename -- which would be a third copy of the ``stem.split("_")`` convention
#: already spelled in ``_build_index`` and ``_build_federated_pairs``.
_FEDERATED_SOURCE_CONTRAST: str = "T1"


def _average_reps(
    reps: list[torch.Tensor],
    mode: str,
    exclude_index: int | None = None,
    anchor: torch.Tensor | None = None,
) -> torch.Tensor:
    """Combine complex k-space repetitions per :data:`_VALID_TARGET_MODES`.

    Convention: ``.claude/rules/data.md`` §"M4Raw handling".

    Args:
        reps: The per-repetition complex k-space tensors.
        mode: One of :data:`_VALID_TARGET_MODES`.
        exclude_index: When set (leave-one-out NEX target), that repetition is
            dropped from the average. The intended use is to exclude the rep
            used as the *input*, so the target's noise is statistically
            independent of the input's — otherwise input and target share the
            input rep's noise at amplitude ``1/N`` and the supervision is biased
            toward preserving that noise. ``None`` (default) averages all reps
            (byte-identical to the legacy behaviour). Phase alignment, when
            applicable, is always anchored to the FIRST retained rep so the
            excluded rep never seeds the reference phase.
    """
    if mode not in _VALID_TARGET_MODES:
        raise ValueError(
            f"[M4Raw] Unknown target_mode: {mode!r}. Valid: {sorted(_VALID_TARGET_MODES)}"
        )
    if mode == "r2r":
        # R2R sets input AND target together from ONE repetition, so it cannot be
        # expressed as "combine these repetitions into a target". It branches at
        # the construction site instead. Reaching here means the branch was
        # bypassed and the caller is about to get an average where it asked for a
        # recorruption -- raise rather than return a plausible wrong tensor.
        raise ValueError(
            "[M4Raw] target_mode='r2r' builds both halves from the input repetition "
            "and must be handled at the construction site, not by _average_reps."
        )
    if mode == "rep_pair":
        # Noise2Noise: the target is ONE other repetition, never an average, so
        # its noise is independent of the input's. ``anchor`` is the input
        # repetition; the pair is phase-aligned to it (a global inter-rep phase
        # offset would otherwise be learned as a phase drift). ``exclude_index``
        # is mandatory: a pair that may contain the input is the identity task.
        if exclude_index is None or not 0 <= exclude_index < len(reps):
            raise ValueError(
                "[M4Raw] target_mode='rep_pair' needs the input repetition's index "
                "(exclude_index) so the pair never contains it."
            )
        others = [r for i, r in enumerate(reps) if i != exclude_index]
        if not others:
            raise ValueError(
                "[M4Raw] target_mode='rep_pair' needs at least one repetition besides the input."
            )
        pair = others[0]
        ref = reps[exclude_index] if anchor is None else anchor
        dot = (pair * ref.conj()).sum(dim=(-2, -1), keepdim=True)
        return pair * torch.exp(-1j * torch.angle(dot))
    if exclude_index is not None and 0 <= exclude_index < len(reps):
        reps = [r for i, r in enumerate(reps) if i != exclude_index]
    if len(reps) == 1:
        return reps[0].clone()
    if not reps:
        # Defensive: exclude_index dropped the only rep. Caller guards against
        # this (LOO only engages with >=2 reps), but never return an empty mean.
        raise ValueError("[M4Raw] _average_reps received no repetitions to average.")
    if mode == "complex_mean":
        return torch.stack(reps, dim=0).mean(dim=0)
    ref = reps[0]
    aligned = [ref]
    for r in reps[1:]:
        dot = (r * ref.conj()).sum(dim=(-2, -1), keepdim=True)
        aligned.append(r * torch.exp(-1j * torch.angle(dot)))
    return torch.stack(aligned, dim=0).mean(dim=0)


class _SkipSampleError(Exception):
    """Raised internally when a sample is unrecoverable (e.g. all rep files corrupt).

    Caught in :meth:`M4RawRepetitionDataset.__getitem__` to transparently
    retry on the next index rather than crashing the DataLoader worker.
    """


def _load_reps_or_skip(
    rep_paths: list[Path], context: str, slice_index: int | None = None
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
    for path in rep_paths:
        try:
            if not path.exists():
                raise FileNotFoundError(f"No such file or directory: '{path}'")
            # The whole volume (the call shape the per-group path has always
            # made) or one slice of it on the slice-level route.
            kspace_reps.append(
                _load_kspace(path) if slice_index is None else _load_kspace(path, slice_index)
            )
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


def _extract_contrast_idx(filename: str) -> int:
    """Return integer contrast index by keyword search in *filename*.

    Args:
        filename: Bare filename (no directory), e.g. ``m4raw_T1_rep001.h5``.

    Returns:
        Integer in ``[0, len(CONTRAST_MAP))``; defaults to 0 (T1) if no match.
    """
    name_upper = filename.upper()
    for key, idx in CONTRAST_MAP.items():
        if key in name_upper:
            return idx
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
    # share a file that is still being written and never block one another.
    with h5py.File(str(path), "r", swmr=True) as f:
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
        log_scaling: bool = False,
        target_mode: str = "complex_mean",
        r2r_alpha: float = 1.0,
        r2r_covariance_source: str = "committed",
        nex_target_exclude_input: bool = False,
        nex_fallback: str = "error",
        slice_level_records: bool = False,
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
            log_scaling: After the (Parseval/image-RSS) percentile divide, apply
                phase-preserving log1p magnitude compression
                (``compress_kspace_log``, the SSOT primitive) to tame the DC
                dynamic range. Inverted by the training strategy before metrics.
            slice_level_records: Expand the index to one record per slice and
                read one slice per repetition per item (#1757). ``False``
                keeps the per-group index byte-identical.
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
        #: Per-scan Sigma_n under 'self_calibrated', keyed by repetition group.
        #: Sigma_n is a property of the RECEIVE CHAIN, not of a slice, so one fit
        #: serves every slice of a scan -- which is also what makes the knob
        #: affordable: the ESPIRiT eigendecomposition behind it costs ~1 s per
        #: slice on CPU, against `samples_per_volume` draws per volume.
        self._r2r_cov_cache: dict[str, Any] = {}
        self._r2r_sampler: Any = None
        if target_mode == "r2r" and r2r_covariance_source == "committed":
            from spectramr.infrastructure.physics.m4raw_noise import M4RawNoiseSampler

            self._r2r_sampler = M4RawNoiseSampler(alpha=self.r2r_alpha)
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
            # Cross-contrast mode: Pair T1 (source) with T2/FLAIR (target)
            self._federated_pairs = self._build_federated_pairs(raw_groups)

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

    # ------------------------------------------------------------------
    # Leave-one-out NEX reference: construction-time contract
    # ------------------------------------------------------------------

    def _refuse_undersized_loo_groups(self) -> None:
        """Refuse a leave-one-out run whose groups cannot all honour it.

        Only the single-contrast path applies leave-one-out (the input rep is
        one of the averaged reps there); in cross-contrast mode the input is a
        different contrast, so ``nex_target_exclude_input`` is not applied and
        this check does not fire. With ``nex_fallback='all_reps'`` the short
        groups are served with the correlated all-reps average and
        ``_note_loo_declined`` reports it once per rep count at load time.
        """
        if not (self.use_repetitions and self.nex_target_exclude_input):
            return
        if self.target_mode == "r2r":
            # R2R builds both halves from the input repetition alone -- it never
            # averages, so it never leaves anything out and a 1-rep group is the
            # POINT of the mode. `nex_target_exclude_input` still arrives True
            # here because it is ONE config knob shared by both splits, and the
            # VALIDATION dataset (built with `data.val_target_mode`, an averaging
            # mode) is the half that needs it. Without this return, an r2r arm
            # would be refused for a leave-one-out its training split does not do.
            return
        if not self.single_contrast:
            logger.warning(
                "[M4Raw] nex_target_exclude_input=True has no effect in cross-contrast "
                "mode: the input is a different contrast's repetition, so it is never "
                "part of the averaged target."
            )
            return
        if self.nex_fallback != "error" and self.target_mode != "rep_pair":
            return
        short: dict[str, int] = {}
        min_reps = _min_reps_for_leave_one_out(self.target_mode)
        for group in self._groups:
            n_reps = len(group.get("paths") or [])
            if n_reps < min_reps:
                label = str(group.get("contrast") or "UNKNOWN")
                short[label] = max(short.get(label, 0), 0) + 1
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
    def _build_federated_pairs(groups_list: list[list[Path]]) -> list[dict]:
        """Pair Source (T1) with Target (FLAIR/T2/PD) from the grouped list.

        Args:
            groups_list: List of Path lists from _build_groups.

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
            if _FEDERATED_SOURCE_CONTRAST not in contrasts:
                continue  # We must have a T1 source

            # Pair T1 with any non-T1 contrast (e.g. FLAIR, T2, PD)
            source_group = contrasts[_FEDERATED_SOURCE_CONTRAST]
            for contrast_name, target_group in contrasts.items():
                if contrast_name == _FEDERATED_SOURCE_CONTRAST:
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
                        "source_contrast": _FEDERATED_SOURCE_CONTRAST,
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
        ``kspace`` : alias for ``input`` (backward-compatible with strategy).
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

    def _sampler_for_scan(self, kspace: torch.Tensor, scan_key: str) -> Any:
        """The R2R sampler for this scan, fitting Sigma_n when self-calibrating.

        Sigma_n describes the RECEIVE CHAIN, not a slice, so one fit serves every
        slice of a scan and the result is cached per repetition group. That is
        also what makes the mode affordable: the ESPIRiT eigendecomposition it
        rests on costs ~1 s per slice on CPU, against `samples_per_volume` draws
        per volume and several epochs.

        A failed fit RAISES rather than falling back to the committed matrix
        (pitfall #9): the arm exists to test the self-calibrated draw, and
        silently serving it the constant would make the comparison vacuous while
        the YAML still said `self_calibrated`.
        """
        if self.r2r_covariance_source == "committed":
            if self._r2r_sampler is None:  # pragma: no cover - guarded in __init__
                raise RuntimeError("[M4Raw] r2r requested but no sampler was built.")
            return self._r2r_sampler

        cached = self._r2r_cov_cache.get(scan_key)
        if cached is None:
            from spectramr.infrastructure.physics.coil_noise_fit import (
                estimate_covariance_from_nullspace,
            )
            from spectramr.infrastructure.physics.coil_sensitivity import estimate_csm_espirit
            from spectramr.infrastructure.physics.fft_ops import ifft2c
            from spectramr.infrastructure.physics.m4raw_noise import M4RawNoiseSampler

            coils = kspace.reshape(-1, *kspace.shape[-2:]).unsqueeze(0)
            subspace = estimate_csm_espirit(
                coils, num_coils=coils.shape[1], return_subspace=True
            )
            covariance = estimate_covariance_from_nullspace(ifft2c(coils), subspace.null_projector)
            cached = M4RawNoiseSampler(covariance=covariance, alpha=self.r2r_alpha)
            self._r2r_cov_cache[scan_key] = cached
            logger.info(
                "[M4Raw] self-calibrated Sigma_n for scan %s: trace=%.4g over %d coils",
                scan_key,
                float(covariance.diagonal().real.sum()),
                int(covariance.shape[0]),
            )
        return cached

    def _recorrupt_r2r(
        self, kspace: torch.Tensor, scan_key: str = ""
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Split ONE acquired repetition into an R2R ``(input, target)`` pair.

        ``input = y + alpha*z``, ``target = y - z/alpha`` with
        ``z ~ CN(0, Sigma_n)`` drawn fresh per sample, so
        ``Cov(input, target) = Sigma_n - Sigma_z = 0`` and the L1/L2 minimiser is
        the clean image -- from a single excitation, with no clean reference.

        The support mask is derived from the DATA, not the header: M4Raw ships
        ~24 % of its phase-encode columns as exact zeros (partial phase FOV), and
        its ``encodingLimits`` report a COUNT where ISMRMRD specifies an
        INCLUSIVE maximum -- measured over all 52 files on this host,
        ``kspace_encoding_step_1/maximum`` equals the number of sampled columns
        exactly (195 for T1/T2, 198 for FLAIR) and ``slice/maximum`` equals the
        slice count (18), so a spec-conforming ``range(min, max + 1)`` over-reads
        by one. Trusting it would put noise in a column the scanner never sampled.
        Sampled columns always carry noise, so a zero column-sum discriminates
        them exactly.

        The draw uses the global RNG, which ``DataLoader`` seeds per worker, so a
        run is reproducible from ``run.seed`` and every sample sees a different
        recorruption across epochs (which is the point -- it is an augmentation
        over the noise, not a fixed second copy).
        """
        if kspace.dim() < 3:
            raise ValueError(
                f"[M4Raw] r2r needs at least (C, H, W) k-space, got {tuple(kspace.shape)}."
            )
        support = kspace.abs().sum(dim=tuple(range(kspace.dim() - 1))) > 0
        if not bool(support.any()):
            raise ValueError(
                "[M4Raw] r2r found no sampled phase-encode columns in this group "
                "(the whole k-space is zero). Recorrupting it would train on noise "
                "alone."
            )
        return self._sampler_for_scan(kspace, scan_key).recorrupt(kspace, support_mask=support)

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
            kspace_reps = _load_reps_or_skip(
                rep_paths,
                f"idx={idx} contrast={contrast_name}"
                + (f" slice={slice_index}" if slice_index is not None else ""),
                slice_index,
            )

            # A NEX target is the AVERAGE of the repetitions. With exactly one
            # surviving rep the code below falls to `target = kspace_reps[0]`,
            # which is the INPUT -- the arm would train the identity while the
            # config says denoising (pitfall #16). Refuse the group instead;
            # __getitem__ retries the next one. Arms that deliberately want
            # input==target set `use_repetitions: false` and never reach here.
            # R2R is exempt: it builds BOTH halves from the input repetition, so a
            # one-repetition group is not a degenerate case for it -- it is the
            # case it exists to serve. Skipping here would discard exactly the
            # single-excitation data the arm is meant to prove it can use.
            if self.use_repetitions and self.target_mode != "r2r" and len(kspace_reps) < 2:
                raise _SkipSampleError(
                    f"[M4Raw] idx={idx} contrast={contrast_name}: only "
                    f"{len(kspace_reps)} of {len(rep_paths)} repetitions loaded. "
                    "The NEX target would be the input itself (SNR boost "
                    "sqrt(1)=1.0), so the sample teaches the identity rather "
                    "than denoising. Fix the unreadable file(s), or set "
                    "data.use_repetitions: false to declare that intent."
                )

            # Input = first repetition (noisy), Target = averaged reps (high-SNR)
            input_kspace = kspace_reps[0].clone()
            if self.target_mode == "r2r":
                # Recorrupted-to-Recorrupted: both halves come from THIS single
                # repetition. Injected here, on raw k-space, for two reasons a
                # registered transform could not satisfy:
                #   * registry transforms are appended AFTER normalization, so the
                #     draw would be scaled by a per-subject factor and no longer
                #     match Sigma_n;
                #   * with ~24 % of phase-encode columns zero-filled, image-space
                #     noise is spatially correlated, so it cannot be drawn there.
                # Keyed on the GROUP, not the slice: Sigma_n is a receive-chain
                # property, so one fit per scan serves every slice of it.
                input_kspace, target_kspace = self._recorrupt_r2r(
                    input_kspace, scan_key=str(rep_paths[0].parent / rep_paths[0].stem[:-2])
                )
            elif self.use_repetitions and len(kspace_reps) > 1:
                # Leave-one-out only when it leaves >=2 reps to average (>=3
                # total); with exactly 2 reps LOO would yield a single noisy
                # rep, so fall back to the all-reps average.
                _loo_requested = self.nex_target_exclude_input
                _min_reps = _min_reps_for_leave_one_out(self.target_mode)
                _exclude = 0 if (_loo_requested and len(kspace_reps) >= _min_reps) else None
                if _loo_requested and _exclude is None:
                    if self.nex_fallback == "error" or self.target_mode == "rep_pair":
                        # A rep_pair group with one repetition has no pair at all:
                        # the all-reps fallback would serve the input as its own
                        # target (an identity task), so it is refused regardless.
                        # The construction-time check covers declared groups; this
                        # covers a group that lost reps to unreadable files.
                        raise ValueError(
                            f"[M4Raw] idx={idx} contrast={contrast_name}: leave-one-out "
                            f"NEX target requested but only {len(kspace_reps)} "
                            f"repetition(s) loaded (needs >= {_min_reps}); "
                            "data.nex_fallback='error' refuses the correlated all-reps "
                            "reference. Declare data.nex_fallback: all_reps to accept it."
                        )
                    self._note_loo_declined(len(kspace_reps))
                target_kspace = _average_reps(kspace_reps, self.target_mode, exclude_index=_exclude)
            else:
                # Single clone suffices: input_kspace is already an independent
                # clone of kspace_reps[0], so one clone keeps target independent
                # too. backlog_wasted_compute_audit_2026_05_29 DATA-10.
                target_kspace = kspace_reps[0].clone()

            # Apply coil processing (RSS combine if needed)
            if input_kspace.dim() == 4 and self.coil_processing_mode == "rss":
                input_kspace = _rss_combine(input_kspace)
                target_kspace = _rss_combine(target_kspace)

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
            target_t = _to_torchio_stacked(target_kspace)

            contrast_idx = _extract_contrast_idx(rep_paths[0].name)
            affine = np.eye(4)
            subject = tio.Subject(
                input=tio.ScalarImage(tensor=input_t, affine=affine),
                target=tio.ScalarImage(tensor=target_t, affine=affine),
                kspace=tio.ScalarImage(tensor=target_t, affine=affine),
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
            subject["repetition_index"] = _identity.repetition
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
        def process_reps(rep_paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
            """process_reps.

            Args:
                rep_paths (list[Path]): Description.
            Returns:
                tuple[torch.Tensor, torch.Tensor]: Description.
            """
            kspace_reps = _load_reps_or_skip(rep_paths, "cross-contrast group", slice_index)

            # Pre-UNet Complex Repetition Fusion (Target 1.2) - fuse natively,
            # or average them here since complex spatial repetition filters belong in the network.
            # However, the network expects 16 channels total (8 T1 + 8 Target).
            # M4Raw is 4 virtual coils = 8 real/imag channels.
            # For data loader, we load all reps and average them to provide the max SNR k-space.
            # (If the network dynamically handles reps via 1x1 conv, we would stack them along the coil dim,
            # but standard models expect fixed channel counts).
            # We will follow the existing averaging strategy for robust SNR.
            input_kspace = kspace_reps[0].clone()
            if self.use_repetitions and len(kspace_reps) > 1:
                target_kspace = torch.stack(kspace_reps, dim=0).mean(dim=0)
            else:
                target_kspace = kspace_reps[0].clone()

            if kspace_reps[0].dim() == 4 and self.coil_processing_mode == "rss":
                input_kspace = _rss_combine(input_kspace)
                target_kspace = _rss_combine(target_kspace)

            return input_kspace, target_kspace

        if self.target_mode == "r2r":
            # The cross-contrast route pairs a SOURCE contrast with a different
            # TARGET contrast; R2R's two halves are one contrast recorrupted
            # against itself. Composing them would silently produce a
            # contrast-translation target the arm never asked for.
            raise ValueError(
                "[M4Raw] target_mode='r2r' is a single-contrast denoising target and "
                "has no meaning in the cross-contrast route (which pairs two "
                "different contrasts). Declare data.pairing.single_contrast: true."
            )

        source_in, source_tgt = process_reps(source_paths)
        target_in, target_tgt = process_reps(target_paths)

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

        # ---- Federated Channel Stacking (Target 1.1) -------------------
        # Stack along coil dimension for network inputs.
        # kspaces are (Slices, Coils, H, W). We cat on the Coils dimension.
        # If dataset is RSS combined, dim is 3 (Slices, H, W), so we add a channel dim.
        if source_in.dim() == 3:
            source_in = source_in.unsqueeze(1)
            target_in = target_in.unsqueeze(1)
            source_tgt = source_tgt.unsqueeze(1)
            target_tgt = target_tgt.unsqueeze(1)

        input_kspace = torch.cat([source_in, target_in], dim=1)
        target_kspace = torch.cat([source_tgt, target_tgt], dim=1)

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

        input_t = _to_torchio_multicoil_stacked(input_kspace)
        target_t = _to_torchio_multicoil_stacked(target_kspace)

        # ---- Extract contrast index from target filename ----------
        contrast_idx = _extract_contrast_idx(target_paths[0].name)

        # ---- Build TorchIO Subject -----------------------------------
        affine = np.eye(4)
        subject = tio.Subject(
            input=tio.ScalarImage(tensor=input_t, affine=affine),
            target=tio.ScalarImage(tensor=target_t, affine=affine),
            # 'kspace' alias should point to target (averaged) for consistency
            # Some strategy code may use this as an alias
            kspace=tio.ScalarImage(tensor=target_t, affine=affine),
        )
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

        # Federated cross-contrast split: input_kspace / target_kspace are built
        # as ``cat([source, target], dim=coils)`` and then interleaved
        # real/imag per coil. The downstream visualization needs to know where
        # the TARGET-contrast portion starts so it can RSS only that half —
        # otherwise it mixes T1 (source) and T2/FLAIR (target) anatomy into one
        # PNG, which renders as a doubled / superimposed brain (the
        # experiment_11 mosaic finding from 2026-05-13). The boundary is the
        # midpoint of the channel axis because source and target carry the
        # same coil count; we still expose it as an explicit index so future
        # asymmetric layouts can override.
        total_channels = int(target_t.shape[0])
        subject["federated_target_channel_start"] = torch.tensor(
            total_channels // 2, dtype=torch.long
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


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def build_m4raw_datasets(
    manifest_path: str | Path,
    data_root: str | Path = "",
    train_split: float = 0.9,
    # See M4RawRepetitionDataset.__init__ — normalization is the transform's job.
    normalize_kspace: bool = False,
    kspace_percentile: float = 0.99,
    use_repetitions: bool = True,
    train_transform=None,
    val_transform=None,
    log_scaling: bool = False,
) -> tuple["M4RawRepetitionDataset", "M4RawRepetitionDataset"]:
    """Load manifest and split into train/val datasets.

    Args:
        manifest_path: Path to ``.pkl`` manifest.
        data_root: Base path for relative manifest entries.
        train_split: Fraction of groups assigned to training (default 0.9).
        normalize_kspace: Enable k-space magnitude normalization.
        kspace_percentile: Normalization percentile.
        use_repetitions: Enable rep averaging.
        train_transform: TorchIO transform for training subjects.
        val_transform: TorchIO transform for validation subjects.

    Returns:
        Tuple of (train_dataset, val_dataset).
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

    h5_files: list[Path] = []
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

    split_n = max(1, int(train_split * len(h5_files)))
    train_files = h5_files[:split_n]
    val_files = h5_files[split_n:]

    train_ds = M4RawRepetitionDataset(
        train_files,
        normalize_kspace=normalize_kspace,
        kspace_percentile=kspace_percentile,
        transform=train_transform,
        use_repetitions=use_repetitions,
        log_scaling=log_scaling,
    )
    val_ds = M4RawRepetitionDataset(
        val_files or train_files,  # fallback for small datasets
        normalize_kspace=normalize_kspace,
        kspace_percentile=kspace_percentile,
        transform=val_transform,
        use_repetitions=use_repetitions,
        log_scaling=log_scaling,
    )

    return train_ds, val_ds
