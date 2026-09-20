"""Per-experiment debug snapshots — paradigm-agnostic capture.

This module provides a single helper that *any* training strategy can
invoke from one line to write a comprehensive debug snapshot to disk:

* A textual ``snapshot.txt`` report covering every named tensor's
  shape, dtype, device, complex/real, channel parity, value range,
  mean / std, and NaN / Inf counts.
* PNG previews of selected image-like tensors (RSS-magnitude reduction
  for multi-channel real-stacked complex; centred-IFFT reduction for
  k-space tensors); skipped silently when a tensor is not displayable.
* A ``snapshot.json`` machine-readable companion of the same data so
  downstream audits / reports can ingest it without parsing text.

The contract: a tensor that is invariant *or* has NaNs *or* whose
domain (k-space vs image) does not match the configured paradigm is
the most informative thing a user can see at the start of an
experiment — those four facts alone explain most of the regressions
caught during the May 2026 smoke-test post-mortem (doubled-brain,
k-space-superposition, channel mismatch, complex-vs-real bias).

The snapshot is gated on ``logging.snapshots.enabled`` (default True) for
the first ``logging.snapshots.max_calls`` calls per ``(run_dir, tag)``,
then auto-suppressed to keep disk usage bounded.

Usage from a strategy::

    from spectramr.infrastructure.training.debug_snapshot import save_debug_snapshot

    save_debug_snapshot(
        run_dir=self.env.run_output_dir,
        step=current_step,
        tag="train_step",
        tensors={
            "input":      input_batch,
            "target":     target_batch,
            "model_input": model_input,
            "prediction": prediction,
        },
        paradigm=type(self).__name__,
        config_section=self.config.logging,
    )

References
----------
* Lustig M., Donoho D., Pauly J. M., "Sparse MRI: The application of
  compressed sensing for rapid MR imaging," MRM 58(6):1182-1195, 2007.
  (k-space layout invariants used by the IFFT-reducer)
* Macovski A., "Noise in MRI," MRM 36(3):494-497, 1996. (RSS-magnitude
  reduction for multi-coil display)
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from spectramr.infrastructure.training.snapshot_provenance import run_identity

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Per-process call budget — keeps disk usage bounded
# ──────────────────────────────────────────────────────────────────────


_call_counts: dict[tuple[str, str], int] = {}


def _budget_check(run_dir: str | Path, tag: str, max_calls: int) -> bool:
    """Enforce a per-(run_dir, tag) call budget to bound disk usage.

    Keyed on the TAG as well as the directory (#706). With one shared counter,
    every capture site in the process competed for a single ``max_calls`` --
    ``first_steps`` (``base.py``), ``model_output`` (``base.py``), ``vf_twin``
    (``virtual_fiducial_strategy.py``) and diffusion's own captures -- so
    whichever fired first consumed the allowance and the later tags were silently
    never written. "This arm produced no model_output snapshot" then reads
    identically to "the model has no output".

    Callers must invoke this only for calls that will actually write; see
    :func:`save_debug_snapshot`, where the interval test now runs first.
    """
    key = (str(run_dir), tag)
    n = _call_counts.get(key, 0) + 1
    _call_counts[key] = n
    return n <= max_calls


# ──────────────────────────────────────────────────────────────────────
# Tensor-level reporting
# ──────────────────────────────────────────────────────────────────────


@dataclass
class TensorReport:
    """Per-tensor diagnostic record."""

    name: str
    shape: list[int]
    dtype: str
    device: str
    is_complex: bool
    even_channel_count: bool  # heuristic for real-stacked complex
    nan_count: int
    inf_count: int
    finite_count: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    abs_max: float | None
    note: str = ""  # human-readable interpretation


def _tensor_report(name: str, t: torch.Tensor) -> TensorReport:
    """Compute a one-shot diagnostic for a single tensor.

    Statistics are computed in fp32 even for bf16 / fp16 inputs to
    avoid silent overflow in mean/std on very large k-space tensors.
    """
    if not isinstance(t, torch.Tensor):
        return TensorReport(
            name=name,
            shape=[],
            dtype=str(type(t).__name__),
            device="cpu",
            is_complex=False,
            even_channel_count=False,
            nan_count=0,
            inf_count=0,
            finite_count=0,
            min=None,
            max=None,
            mean=None,
            std=None,
            abs_max=None,
            note="not a torch.Tensor",
        )

    with torch.no_grad():
        x = t.detach()
        is_cx = x.is_complex()
        x_view = x.abs().float() if is_cx else x.float()

        n_total = x_view.numel()
        nan_count = int(torch.isnan(x_view).sum().item())
        inf_count = int(torch.isinf(x_view).sum().item())
        finite = n_total - nan_count - inf_count

        if finite > 0:
            x_finite = x_view[torch.isfinite(x_view)]
            mn = float(x_finite.min().item())
            mx = float(x_finite.max().item())
            mu = float(x_finite.mean().item())
            sd = float(x_finite.std().item()) if x_finite.numel() > 1 else 0.0
            am = float(x_finite.abs().max().item())
        else:
            mn = mx = mu = sd = am = None

        even_ch = x.dim() >= 4 and x.shape[1] % 2 == 0 and not is_cx

        # Annotate the most common diagnostics
        notes: list[str] = []
        if nan_count > 0:
            notes.append(f"⚠ {nan_count} NaN")
        if inf_count > 0:
            notes.append(f"⚠ {inf_count} Inf")
        if mn is not None and mx is not None:
            if abs(mx - mn) < 1e-12:
                notes.append("⚠ constant tensor (max − min < 1e-12)")
            if mn < 0 and not is_cx and x.dim() == 4 and x.shape[1] == 1:
                notes.append("ℹ negative values present (signed image)")
        if even_ch and x.dim() == 4 and x.shape[1] in (2, 4, 8, 16):
            notes.append(f"ℹ {x.shape[1]}-ch real-stacked → likely complex (R0,I0,R1,I1,...)")

        return TensorReport(
            name=name,
            shape=list(x.shape),
            dtype=str(x.dtype),
            device=str(x.device),
            is_complex=is_cx,
            even_channel_count=even_ch,
            nan_count=nan_count,
            inf_count=inf_count,
            finite_count=finite,
            min=mn,
            max=mx,
            mean=mu,
            std=sd,
            abs_max=am,
            note="; ".join(notes),
        )


# ──────────────────────────────────────────────────────────────────────
# Image rendering (paradigm-aware)
# ──────────────────────────────────────────────────────────────────────


# The k-space DC-signature veto is shared with the validation-image visualiser
# (metrics_mixin) — single source of truth in domain_inference. Re-exported under
# the original private name so existing snapshot tests/imports keep working.
from spectramr.infrastructure.training.utils.domain_inference import (  # noqa: E402
    looks_like_kspace as _looks_like_kspace,
)


def _is_in_kspace_domain(x: torch.Tensor) -> bool:
    """Is a complex tensor in k-space domain (→ needs an IFFT to view as image)?

    Decides by the **spectrum**, not the raw DC magnitude. A non-negative
    image FFTs to a sharp-DC k-space; a k-space — even one whose DC has been
    flattened by ``data.normalize_kspace`` (phase-preserving ``log1p`` /
    robust magnitude compression) — FFTs to a spread, no-DC field. So ``x`` is
    already image-domain iff ``fft2c(x)`` carries the k-space DC signature.

    This is **normalization-invariant**, unlike the legacy
    ``_looks_like_kspace(x.abs())`` raw-DC test, which false-negated on
    normalized k-space and rendered the experiment_11 debug ``target.png`` as
    raw k-space (DC blob / "doubled brain", 2026-06-17 downloaded-run triage).
    VF / IB image-output targets still FFT to a sharp DC, so they are
    correctly recognised as image-domain and left untransformed.
    """
    from spectramr.infrastructure.physics.fft_ops import fft2c

    return not _looks_like_kspace(fft2c(x).abs())


#: One channel block of a domain-superposed tensor, as its emitting strategy
#: declares it: ``(label, num_channels, is_kspace)``, or
#: ``(label, num_channels, is_kspace, log_compressed)`` when the block's
#: compression state does not follow from its domain.
#:
#: The fourth field exists because DOMAIN and COMPRESSION are independent facts
#: about a tensor, and cold diffusion's ``model_input`` is the case that forced
#: them apart. Since #1327 BOTH of its halves are k-space -- but only the
#: ``x_t`` half passed through the arm's ``log1p``. The maps half is ``fft2c``
#: of image-domain sensitivities (``prepare_smaps_for_kspace_conditioning``),
#: level-matched and amplitude-capped, never compressed. Declaring it
#: ``is_kspace=True`` alone would earn it the IFFT it genuinely needs AND the
#: ``expm1`` it must not have.
#:
#: ``None`` -- and the three-tuple spelling, which is every declaration written
#: before this field existed -- means "inherit the parent tensor's answer".
ChannelSegment = tuple[str, int, bool] | tuple[str, int, bool, bool | None]


def _split_channel_segments(
    name: str,
    t: torch.Tensor,
    segments: Sequence[ChannelSegment],
) -> list[tuple[str, torch.Tensor, bool, bool | None]] | None:
    """Slice a superposed tensor into ``(render_name, sub, is_kspace, log)``.

    ``log`` is the segment's declared compression state, or ``None`` to inherit
    the parent tensor's -- see :data:`ChannelSegment` for why a segment may need
    to disagree with its own domain.

    Returns ``None`` when the declaration cannot be honoured -- no channel axis,
    or the segment widths do not sum to the tensor's channel count. The caller
    then renders the tensor whole, having logged why. Silently re-normalising a
    mis-declared split would put a wrong picture under a right-looking filename,
    which is the failure mode this whole mechanism exists to remove.
    """
    if t.dim() < 2:
        return None
    total = t.shape[1]
    declared = sum(seg[1] for seg in segments)
    if declared != total:
        logger.warning(
            "[debug-snapshot] channel_segments for %r declare %d channels but "
            "the tensor has %d; rendering it whole. The split is a claim about "
            "what the emitting strategy concatenated -- fix the declaration at "
            "the call site rather than the tensor.",
            name,
            declared,
            total,
        )
        return None

    out: list[tuple[str, torch.Tensor, bool, bool | None]] = []
    start = 0
    for seg in segments:
        label, width, is_kspace = seg[0], seg[1], seg[2]
        log_compressed = seg[3] if len(seg) == 4 else None
        out.append(
            (
                f"{name}__{label}",
                t[:, start : start + width],
                is_kspace,
                log_compressed,
            )
        )
        start += width
    return out


def _render_image_preview(
    t: torch.Tensor,
    *,
    in_kspace: bool,
    authoritative: bool = False,
    log_scaled: bool = False,
) -> torch.Tensor | None:
    """Reduce a tensor to ``(B, 1, H, W)`` magnitude suitable for save_image.

    Args:
        t: Input tensor of any layout.
        in_kspace: When True, apply a centred IFFT before magnitude.
        authoritative: When True, the ``in_kspace`` flag came from the config
            SSOT (``model.target_domain`` / ``KNOWN_KSPACE_OUTPUT_MODELS`` via
            ``needs_ifft_for_visualization``), not a name/substring guess — so
            the IFFT is applied UNCONDITIONALLY and the fragile spectrum veto
            (``_is_in_kspace_domain``) is bypassed. The veto false-negates on
            real *normalized multicoil* M4Raw k-space and rendered the
            experiment_11 ``input_prepared`` / ``model_output`` snapshots as
            raw, off-centre k-space instead of a brain (2026-06-27 audit).
        log_scaled: When True, ``data.processing.enable_log_scaling`` compressed
            this k-space with ``log1p``, so the magnitude is decompressed before
            the IFFT (#682). Without it the transform runs on a per-bin
            nonlinearity and the render is a high-pass artifact, not a scaled
            image. Threaded in as an argument because this module receives
            ``config.logging`` and cannot reach ``config.data``.

    Returns:
        ``(B, 1, H, W)`` real magnitude in [0, 1] *per-sample* normalised,
        or ``None`` if the tensor is not displayable as an image.
    """
    try:
        x = t.detach().cpu()
        if x.dim() == 5:
            # 3-D volume → take central slice along the depth axis
            x = x[..., x.shape[-1] // 2]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4:
            return None

        # Re-pair real-stacked complex → complex tensor ONLY when the caller
        # has declared this tensor to be k-space. In image domain we cannot
        # safely assume an even channel count means (R,I) interleaved: it
        # might be (ULF,HF) paired modalities, or multi-contrast stacked
        # channels, or per-coil magnitudes already RSS'd elsewhere. RSS of
        # the raw real-stacked channels is mathematically identical to the
        # per-coil-magnitude+RSS chain for properly-encoded complex data
        # (sqrt(Σ R_c² + Σ I_c²) = sqrt(Σ |C_c|²)), so we lose nothing by
        # skipping the pairing in image domain.
        if in_kspace and not x.is_complex() and x.shape[1] >= 2 and x.shape[1] % 2 == 0:
            ev = x[:, 0::2]
            od = x[:, 1::2]
            x = torch.complex(ev, od)

        # Optional IFFT (k-space → image). Guard against a spurious IFFT on a
        # tensor that is flagged k-space but is already image domain (VF / IB
        # image-output arms): only transform when the tensor is genuinely
        # k-space. The decision is spectrum-based (see _is_in_kspace_domain),
        # so it is invariant to data.normalize_kspace's DC-flattening — the
        # raw-DC test used here previously false-negated on normalized k-space
        # and rendered experiment_11 debug target.png as raw k-space (DC blob /
        # "doubled brain", 2026-06-17), while still IFFT'ing an image-domain
        # complex target into a black DC-spike PNG was the older failure (smoke
        # 2026-06-12) the guard prevents.
        if in_kspace and x.is_complex():
            if authoritative or _is_in_kspace_domain(x):
                from spectramr.infrastructure.physics.fft_ops import ifft2c
                from spectramr.infrastructure.training.utils.kspace_view import (
                    decompress_for_view,
                )

                # Undo log1p BEFORE the transform (#682). A no-op when the arm
                # does not compress, so the call is unconditional.
                x = decompress_for_view(x, log_scaled=log_scaled, channel_dim=1)
                x = ifft2c(x)
            else:
                logger.debug(
                    "debug-snapshot: tensor flagged k-space but its spectrum "
                    "carries the k-space DC signature (already image domain); "
                    "skipping IFFT."
                )

        # Magnitude
        if x.is_complex():
            x = x.abs()
        else:
            x = x.float().abs()

        # Multi-channel → RSS reduce. Works for per-coil magnitudes,
        # real-stacked complex (mathematically equivalent), and is the
        # least-wrong fallback for paired modalities (still mixes them,
        # but at least produces a deterministic single-channel image; the
        # caller should slice to one modality upstream when that matters).
        if x.dim() == 4 and x.shape[1] > 1:
            x = torch.sqrt((x**2).sum(dim=1, keepdim=True) + 1e-12)

        # Per-sample [0, 1]
        b = x.shape[0]
        mn = x.view(b, -1).min(1, keepdim=True)[0].view(b, 1, 1, 1)
        mx = x.view(b, -1).max(1, keepdim=True)[0].view(b, 1, 1, 1)
        x = (x - mn) / (mx - mn + 1e-12)
        return x.clamp(0, 1)
    except Exception as exc:
        logger.debug("debug-snapshot preview failed for shape %s: %s", tuple(t.shape), exc)
        return None


# ──────────────────────────────────────────────────────────────────────
# Domain-superposition detector
# ──────────────────────────────────────────────────────────────────────


#: A declared-k-space tensor whose rendered *image* has at least this fraction
#: of interior rows-or-columns fully dark is flagged as a domain superposition
#: (a k-space line mask applied in the image domain). A correct k-space
#: undersampling IFFTs to a *dense* aliased image (≈0 zero lines); only a
#: line mask multiplied against IMAGE data zeroes whole image rows/cols.
_MASK_IN_IMAGE_MIN_ZERO_FRAC = 0.30


def _image_domain_mask_fraction(preview: torch.Tensor) -> float:
    """Fraction of interior image rows-or-cols that are ~fully zero.

    Operates on the FINAL rendered preview ``(B, 1, H, W)`` in ``[0, 1]`` — the
    exact image a reviewer looks at. This is the k-space-mask-in-image-domain
    signature: applying a Cartesian phase-encode line mask to IMAGE data zeroes
    whole rows (or columns) of the image, whereas applying it to k-space and
    IFFT-ing produces a dense, aliased image with no zero lines.

    The FOV borders (outer 15% each side) are excluded — air rows there are
    legitimately dark. A degenerate (near-constant / all-zero) preview returns
    ``0.0`` so a blank tensor is never misread as striped.

    Returns ``max(row_fraction, col_fraction)`` (a mask lives on one axis).
    """
    if preview is None or preview.dim() != 4 or preview.shape[1] != 1:
        return 0.0
    x = preview.detach().float()
    if not torch.isfinite(x).all():
        return 0.0
    if float(x.max() - x.min()) < 1e-6:  # constant / blank frame: no line structure
        return 0.0
    _, _, h, w = x.shape
    eps = 1e-4
    # A row/col is "dark" iff it is dark across every sample and the other axis.
    row_max = x.amax(dim=(0, 1, 3))  # (H,)
    col_max = x.amax(dim=(0, 1, 2))  # (W,)
    ry0, ry1 = int(0.15 * h), int(0.85 * h)
    cx0, cx1 = int(0.15 * w), int(0.85 * w)
    interior_rows = row_max[ry0:ry1]
    interior_cols = col_max[cx0:cx1]
    row_frac = float((interior_rows < eps).float().mean()) if interior_rows.numel() else 0.0
    col_frac = float((interior_cols < eps).float().mean()) if interior_cols.numel() else 0.0
    return max(row_frac, col_frac)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


@dataclass
class SnapshotConfig:
    """Resolved snapshot config (mirrors the relevant LoggingSchema fields)."""

    enabled: bool = True
    max_calls: int = 8
    save_images: bool = True
    save_json: bool = True
    image_interval_steps: int = 0  # 0 = save every call within budget


def _resolve_config(config_section: Any | None) -> SnapshotConfig:
    """Pull the snapshot config from a Pydantic logging section, with safe defaults."""
    if config_section is None:
        return SnapshotConfig()
    # `config_section is None` already returned above, and every leaf is a
    # declared field on the sub-block, so no per-field defaulting is needed.
    snap = config_section.snapshots
    return SnapshotConfig(
        enabled=bool(snap.enabled),
        max_calls=int(snap.max_calls),
        save_images=bool(snap.save_images),
        save_json=bool(snap.save_json),
        image_interval_steps=int(snap.interval_steps),
    )


def snapshots_are_enabled(config_section: Any | None) -> bool:
    """Is the snapshot subsystem on at all? No step and no budget dependency.

    Deliberately NOT :func:`snapshot_step_is_due`. That function answers "may a
    write happen *at this step*", which folds in ``interval_steps`` and so moves
    with the run's step counter; this one answers "does this run produce
    snapshot artifacts", which is a property of the config alone.

    The distinction matters for the model-input contract in
    ``BaseTrainingStrategy``. That contract binds what an *artifact* claims, so
    the check that enforces it must fire on exactly the runs that write
    artifacts -- not on the subset of steps a cadence happens to select. Gating
    it on a step-dependent predicate would resurrect #706's shape: the wrapper
    and the strategy count steps from different places, so they disagree the
    moment a caller omits ``iteration`` or a run resumes mid-interval.
    """
    return _resolve_config(config_section).enabled


def snapshot_step_is_due(config_section: Any | None, step: int) -> bool:
    """Could a snapshot at ``step`` possibly be written? Cheap and stateless.

    Call sites that must build a dict of tensors (or slice one) before calling
    :func:`save_debug_snapshot` use this to skip that work on steps the module
    would reject anyway. It is an *upper bound*: the per-tag ``max_calls``
    budget is stateful and lives inside :func:`save_debug_snapshot`, which
    remains the only authority on whether a call actually writes.

    **Do not re-derive a cadence from an unrelated knob at the call site.**
    ``diffusion.py`` gated its degraded-model-input snapshot on
    ``logging.intervals.log * 5``; arms routinely set ``intervals.log: 5000``,
    which moved that snapshot to step 25 000 and made it unreachable in every
    run shorter than that -- the same silent-disable shape as #706 one gate
    below. Snapshot cadence is ``logging.snapshots``' business, and only its.

    Args:
        config_section: The ``config.logging`` block, or ``None`` for defaults.
        step: The current training step.

    Returns:
        ``False`` when snapshots are disabled or ``step`` misses the configured
        interval; ``True`` when a write is possible, budget permitting.
    """
    cfg = _resolve_config(config_section)
    if not cfg.enabled:
        return False
    if cfg.image_interval_steps > 0:
        return step % cfg.image_interval_steps == 0
    # No interval configured: the stateful per-(run_dir, tag) budget inside
    # `save_debug_snapshot` is the ONLY thing bounding the writes, and this
    # function cannot observe it. So every step is due and the writer decides.
    #
    # This must NOT be `step <= cfg.max_calls`. That compares an ITERATION
    # against a CALL budget -- two different quantities -- and so under-reports
    # the moment a run's step counter starts above the budget. A resumed run
    # begins at `start_iteration + 1` (pipelines/training_loop.py), so with the
    # default `max_calls: 8` every resume past iteration 8 reported "not due"
    # forever while the writer, with a fresh in-process counter, would have
    # written its full allowance. `virtual_fiducial_strategy._snapshot_twin_outputs`
    # carries a comment about this exact confusion biting a previous owner.
    return True


#: Stamped in place of a deferred value whose callable raised. The key SURVIVES
#: with a reason rather than vanishing: a diagnostic that silently disappears is
#: indistinguishable from one that was never requested (pitfall #16).
_UNRESOLVED_EXTRA = "<unresolved: {0}>"


def _resolve_deferred_extra(
    extra: dict[str, Any] | Callable[[], dict[str, Any]] | None,
) -> dict[str, Any] | None:
    """Invoke any zero-arg callable in ``extra`` -- AFTER the write is decided.

    Two accepted shapes, and the difference matters when several values come off
    one computation:

    * a **callable returning the dict** -- one call produces every value, so a
      single device sync feeds all of them;
    * a **dict with callable values** -- each key defers independently.

    Why defer at all: a scale/limit diagnostic wants ``.item()`` or
    ``float(tensor)``, and a device->host sync in the training loop is banned
    (non-negotiable 9). Building ``extra`` eagerly at the call site pays that
    sync on **every** step, while the budget only suppresses the *write* -- so
    the gate the caller believes in never runs. That is issue #1188.

    Resolution happens exactly once, here, past both the interval and budget
    gates, so a suppressed call costs nothing. Deferring rather than exposing a
    "will this write?" predicate is deliberate: :func:`_budget_check` MUTATES the
    counter, so any peek would either consume the allowance or race the real
    call.

    A callable that raises stamps :data:`_UNRESOLVED_EXTRA` rather than dropping
    the key or propagating -- this runs inside the writer, and a diagnostic must
    never be the thing that breaks a training step.
    """
    if callable(extra):
        try:
            extra = extra()
        except Exception as exc:  # diagnostics must never break training
            return {"extra": _UNRESOLVED_EXTRA.format(f"{type(exc).__name__}: {exc}")}
    if not isinstance(extra, dict):
        return extra
    resolved: dict[str, Any] = {}
    for key, value in extra.items():
        if not callable(value):
            resolved[key] = value
            continue
        try:
            resolved[key] = value()
        except Exception as exc:  # diagnostics must never break training
            resolved[key] = _UNRESOLVED_EXTRA.format(f"{type(exc).__name__}: {exc}")
    return resolved


#: Snapshot roots this PROCESS has already scanned for another run's leftovers.
#: The scan is one directory listing, and it only has an answer to give once.
_MARKED_ROOTS: set[str] = set()

#: Which ``<tag>_step_<n>`` directories THIS run wrote, per snapshot root.
_WRITTEN_DIRS: dict[str, list[str]] = {}


def _mark_run(snap_root: Path, snap_dir: Path) -> None:
    """Record which run owns which snapshot directories, and warn about the rest.

    A run directory is reused across runs and a snapshot directory is named
    ``<tag>_step_<n>`` -- no run in it. So a shorter run overwrites the early
    steps IN PLACE and leaves the later ones untouched, and the result reads as
    one coherent set: ``experiment_11_attention_none`` came back from the
    2026-09-16 smoke run holding steps 1-2 from that run beside steps 3-8 and
    ``model_output_dc`` steps 500-4000 from 2026-09-14, with nothing on disk
    saying so. Directory mtimes do not tell them apart either -- rewriting a
    file in place leaves the directory's own mtime alone.

    ``RUN.json`` is that missing statement: it names the run that owns the
    listed directories, so anything in the root and not in the list is a
    leftover. Instrumentation, so a failure to write it warns rather than
    killing the run it was only describing.
    """
    identity = run_identity()
    marker = snap_root / "RUN.json"
    key = str(snap_root)

    if key not in _MARKED_ROOTS:
        _MARKED_ROOTS.add(key)
        try:
            previous = json.loads(marker.read_text()) if marker.is_file() else {}
        except (OSError, ValueError):
            previous = {}
        stale = sorted(
            d.name
            for d in snap_root.iterdir()
            if d.is_dir() and d.name != snap_dir.name
        )
        if stale and previous.get("run_id") != identity.get("run_id"):
            logger.warning(
                "[debug-snapshot] %s already holds %d snapshot director%s from an "
                "earlier run (%s). This run rewrites only the steps it reaches, so "
                "the rest stay as that run left them -- read RUN.json for the ones "
                "this run owns: %s",
                snap_root,
                len(stale),
                "y" if len(stale) == 1 else "ies",
                previous.get("run_id") or "run id not recorded",
                ", ".join(stale[:6]) + (" ..." if len(stale) > 6 else ""),
            )

    written = _WRITTEN_DIRS.setdefault(key, [])
    if snap_dir.name not in written:
        written.append(snap_dir.name)
    try:
        marker.write_text(
            json.dumps({**identity, "directories": sorted(written)}, indent=2, default=str)
        )
    except OSError as exc:
        logger.warning("[debug-snapshot] could not write %s: %s", marker, exc)


def save_debug_snapshot(
    *,
    run_dir: str | Path,
    step: int,
    tag: str,
    tensors: Mapping[str, torch.Tensor | None],
    paradigm: str = "unknown",
    config_section: Any | None = None,
    in_kspace_keys: set[str] | None = None,
    authoritative_kspace_keys: set[str] | None = None,
    log_scaled: bool = False,
    log_scaled_keys: set[str] | None = None,
    channel_segments: Mapping[str, Sequence[ChannelSegment]] | None = None,
    extra: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    model_input_contract: dict[str, Any] | None = None,
) -> Path | None:
    """Write a paradigm-agnostic debug snapshot.

    Args:
        run_dir: Experiment output directory; the snapshot is written to
            ``<run_dir>/debug_snapshots/<tag>_step_<step:06d>/``.
        step: Training step / iteration.
        tag: Short tag for this snapshot (e.g. ``"train_step"``,
            ``"first_validation"``, ``"loss_nan"``).
        tensors: Mapping ``name -> tensor``. ``None`` values are skipped.
        paradigm: Identifying string for the strategy (e.g. ``"GAN"``).
        config_section: ``config.logging`` Pydantic section; if absent
            the defaults from :class:`SnapshotConfig` are used.
        in_kspace_keys: Names of tensors that are k-space (so the
            preview applies IFFT before magnitude). Default: keys
            containing ``"kspace"`` or ``"k_space"``.
        authoritative_kspace_keys: Subset of ``in_kspace_keys`` whose k-space
            domain is declared by the config SSOT (not a name guess). For
            these the IFFT is applied unconditionally — the fragile spectrum
            veto is bypassed (see :func:`_render_image_preview`).
        log_scaled: Whether this ARM compresses k-space with ``log1p``
            (``data.processing.enable_log_scaling``). Says nothing about which
            tensors in ``tensors`` were captured on the compressed side of that
            transform — see ``log_scaled_keys``.
        log_scaled_keys: Names of the tensors that are ACTUALLY log-compressed,
            for the arms where ``log_scaled`` is True. A single per-call flag is
            not enough because one snapshot routinely mixes domains: the base
            strategy's generic ``model_output`` snapshot pairs the generator's
            output (network units, compressed) with ``target``/``input`` as they
            entered ``train_step`` — i.e. BEFORE the strategy's own
            ``apply_kspace_normalization`` compresses them. Decompressing those
            two applies ``expm1`` to a tensor that was never ``log1p``'d, which
            clamps every bin above ``DECOMPRESS_MAGNITUDE_CEILING`` to the same
            value; the magnitude spectrum is flattened while phase survives, so
            the IFFT renders a phase-only, ringing, washed-out "brain" and the
            reader concludes the DATA is broken. Default ``None`` keeps the
            historical behaviour (every declared-k-space tensor is decompressed),
            which is correct for the strategy-side emitters that capture
            exclusively post-normalization tensors.
        channel_segments: For a tensor whose CHANNEL AXIS superposes more than
            one *rendering* treatment, the ordered segments it decomposes into:
            ``name -> [(label, num_channels, is_kspace[, log_compressed]), ...]``
            (see :data:`ChannelSegment`). Each segment is rendered as its own
            ``<name>__<label>.png`` under its own domain and compression, and the
            un-split tensor is still reported in full by the stats table.

            Cold diffusion's ``model_input`` is the motivating case: it is
            ``cat([x_t, smaps_k], dim=1)`` (``strategies/diffusion.py``), and its
            two halves have needed the split for two different reasons in turn.

            **Before #1327** the maps half was image-domain, so the previewer's
            blanket ``expm1`` + ``ifft2c`` mistreated it: the IFFT of a smooth
            image-domain field is a separable sinc-like kernel -- a bright DC
            pixel plus a horizontal and vertical ridge -- which after the RSS
            combine owns the frame's energy, and the closing per-sample min-max
            then crushes the genuine zero-filled brain to near-black. That is the
            ``experiment_11_attention_none`` "crosshair over a black frame"
            render, and it was a rendering defect: the k-space the network was
            fed was fine.

            **Since #1327** the maps half is ``fft2c`` of the sensitivities
            (``prepare_smaps_for_kspace_conditioning``), so it is k-space and
            DOES need the IFFT -- but it still never passed through the arm's
            ``log1p``, and ``expm1`` on it flattens every bin above
            ``DECOMPRESS_MAGNITUDE_CEILING`` onto one value exactly as described
            for ``log_scaled_keys`` above. It therefore declares
            ``is_kspace=True, log_compressed=False``: the two facts are recorded
            independently because the tensor genuinely disagrees with itself.

            Declared by the emitting strategy, which is the only layer that knows
            what it concatenated. Segments whose channel counts do not sum to the
            tensor's channel count are refused with a warning and the tensor
            renders whole -- a mis-declared split must be visible, not silently
            approximated (non-negotiable 3).
        extra: Scalar diagnostics folded into the text report and JSON. May be
            **deferred**: pass a zero-arg callable (or a dict whose values are
            callables) and it is invoked only once this call has cleared the
            interval and budget gates. Use that whenever building the value
            costs a device->host sync -- ``.item()`` / ``float(tensor)`` in the
            training loop is non-negotiable 9, and an eagerly-built ``extra``
            pays it on every step even though the budget suppresses all but the
            first few *writes* (issue #1188). See
            :func:`_resolve_deferred_extra`.
        provenance: What was done to the data before it reached these tensors,
            as built by
            :func:`~spectramr.infrastructure.training.snapshot_provenance.build_snapshot_provenance`
            -- the declared knobs beside the transforms actually applied. Stored
            as its own top-level JSON key rather than folded into ``extra``: it
            is a nested structured record, and ``_coerce_extra`` flattens
            non-scalars to strings. Without it the four numbers per tensor
            cannot be read against what the arm asked for, which is what made an
            unaccelerated-looking ``input_prepared`` unfalsifiable from the
            artifact alone. See ``docs/debug_snapshot_contract.rst``.
        model_input_contract: For the snapshot a strategy declares as its model
            input (``snapshot_model_input_tag``), the verdict from
            :func:`~spectramr.infrastructure.training.model_input_contract.verify_model_input`
            -- which key names the model input, its channel count, and the
            backbone's declared width. ``None`` on every other snapshot, and
            then the key is absent from the JSON rather than present-and-null.
            This writer does not compute it: it cannot reach the model. Passed
            in by the strategy wrapper, which can.

    Returns:
        Path to the snapshot directory, or ``None`` if the call was
        suppressed by the budget or by ``snapshots.enabled=false``.
    """
    cfg = _resolve_config(config_section)
    if not cfg.enabled:
        return None

    # A real filesystem path, or nothing (#693/#476). `run_dir` is declared
    # `str | Path` and was never checked, so a `MagicMock` env -- which
    # auto-creates a truthy `run_output_dir` -- stringified into
    # `MagicMock/mock.run_output_dir/<object-id>/debug_snapshots/...` and wrote
    # real PNGs into the repo root. 420 such files accumulated and 3 were
    # committed. `os.fspath` raises TypeError on a Mock, so ONE guard here fixes
    # every present and future mocked-env test; patching fixtures is
    # N-places-forever and re-breaks on the next one.
    # `isinstance(..., (str, Path))`, NOT `os.fspath`: a MagicMock auto-creates
    # `__fspath__`, so it passes `os.fspath` AND `isinstance(m, os.PathLike)`, and
    # `os.fspath(MagicMock().run_output_dir)` returns the literal string
    # 'MagicMock/mock.run_output_dir/<id>' -- which is exactly the tree found in
    # the repo root. Only the concrete-type test discriminates, and it is the
    # signature this function already declares.
    if not isinstance(run_dir, (str, Path)):
        raise TypeError(
            f"save_debug_snapshot(run_dir=...) needs a real path, got "
            f"{type(run_dir).__module__}.{type(run_dir).__name__}. A test with a "
            "mocked training environment should pass tmp_path; an unresolved Mock "
            "here writes snapshot files into the working directory under a "
            "'MagicMock/' tree (420 such files accumulated, 3 were committed)."
        )

    # Interval BEFORE budget (#706). `_budget_check` MUTATES the counter, so
    # consuming it on a call the interval then rejects meant that setting
    # `snapshots.interval_steps: 100` with the default `max_calls: 8` exhausted
    # the budget by step 7 -- only the step-0 snapshot was ever written. Turning
    # on the cadence knob turned the feature off, silently.
    if cfg.image_interval_steps > 0 and step % cfg.image_interval_steps != 0:
        return None
    if not _budget_check(run_dir, tag, cfg.max_calls):
        return None

    # Everything past this line is a CERTAIN write, which is the only place a
    # deferred `extra` may be resolved: earlier and the caller pays for a
    # diagnostic that the interval or the budget then throws away -- the #1188
    # defect. Resolved once here, before either consumer (`_format_text_report`
    # and `_coerce_extra` both read the same resolved dict), so a callable that
    # performs a GPU sync syncs once per written snapshot, never twice.
    extra = _resolve_deferred_extra(extra)

    snap_root = Path(run_dir) / "debug_snapshots"
    snap_dir = snap_root / f"{tag}_step_{step:06d}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    _mark_run(snap_root, snap_dir)

    if in_kspace_keys is None:
        in_kspace_keys = {
            k for k in tensors.keys() if "kspace" in k.lower() or "k_space" in k.lower()
        }

    # Pre-render image previews for every DECLARED-k-space tensor (torch only,
    # no torchvision) so the report can carry a domain-superposition note and
    # the PNG step below can reuse the render. A k-space tensor whose *image*
    # rendering shows zeroed lines = a k-space mask applied in the image domain.
    def _is_log_scaled(name: str) -> bool:
        """Is THIS tensor on the compressed side of the arm's ``log1p``?"""
        if not log_scaled:
            return False
        return log_scaled_keys is None or name in log_scaled_keys

    # Rendering unit = what becomes one PNG. Usually the tensor itself; for a
    # tensor whose channel axis superposes domains (cold diffusion's
    # ``model_input`` is ``cat([x_t, smaps], dim=1)``) the caller's
    # ``channel_segments`` splits it so each half renders in ITS OWN domain.
    # The stats table below still reports the undivided tensor -- the record of
    # what the model was fed stays whole (non-negotiable 14); only the picture
    # is decomposed.
    render_units: list[tuple[str, torch.Tensor, bool, bool, bool]] = []
    for name, t in tensors.items():
        if t is None:
            continue
        base_kspace = name in in_kspace_keys
        base_auth = authoritative_kspace_keys is not None and name in authoritative_kspace_keys
        parts = (
            _split_channel_segments(name, t, channel_segments[name])
            if channel_segments and name in channel_segments
            else None
        )
        if parts is None:
            render_units.append((name, t, base_kspace, base_auth, _is_log_scaled(name)))
            continue
        for render_name, sub, seg_kspace, seg_log in parts:
            # Domain and compression are decided SEPARATELY. A segment that
            # declares neither (the three-tuple spelling, ``seg_log is None``)
            # inherits the domain answer, which is what it has always meant: an
            # image-domain segment was never ``log1p``'d, so it takes neither
            # the IFFT nor the ``expm1``. A segment that declares
            # ``log_compressed=False`` while being k-space -- cold diffusion's
            # ``smaps`` half since #1327 -- takes the IFFT and skips the
            # ``expm1``, because ``fft2c`` of a sensitivity map is genuine
            # k-space that never passed through the arm's ``log1p``.
            #
            # Note this is NOT "narrowing log_scaled_keys" (forbidden by
            # ``utils/kspace_view``): that would suppress a decompression the
            # tensor genuinely needs. Here the segment states that it was never
            # compressed, so ``expm1`` was never the right operation for it.
            # The arm-level gate stays outermost, so an explicit ``True`` still
            # decompresses nothing on an arm that does not log-scale.
            render_units.append(
                (
                    render_name,
                    sub,
                    seg_kspace,
                    base_auth and seg_kspace,
                    _is_log_scaled(name) and (seg_log if seg_log is not None else seg_kspace),
                )
            )

    previews: dict[str, torch.Tensor] = {}
    for name, t, unit_kspace, unit_auth, unit_log in render_units:
        if not unit_kspace:
            continue
        try:
            p = _render_image_preview(
                t,
                in_kspace=True,
                log_scaled=unit_log,
                authoritative=(unit_auth),
            )
            if p is not None:
                previews[name] = p
        except Exception as exc:  # never let a preview break the snapshot
            logger.debug("debug-snapshot preview (domain-check) skipped %s: %s", name, exc)

    # Build per-tensor diagnostics, augmenting declared-k-space tensors with a
    # loud DOMAIN SUPERPOSITION note when their rendered image is line-zeroed.
    reports: list[TensorReport] = []
    for name, t in tensors.items():
        if t is None:
            continue
        r = _tensor_report(name, t)
        # A tensor with declared ``channel_segments`` is rendered per SEGMENT, so
        # ``previews`` holds it under ``<name>__<label>`` and never under ``name``
        # itself. Look through to the segments rather than losing this check on
        # exactly the keys the split was introduced for -- silently, since the
        # PNGs and the table row both stay correct when the lookup misses.
        _direct = previews.get(name)
        if _direct is not None:
            _checked: list[tuple[str | None, torch.Tensor]] = [(None, _direct)]
        else:
            _prefix = f"{name}__"
            _checked = [
                (k[len(_prefix) :], p) for k, p in previews.items() if k.startswith(_prefix)
            ]
        for _label, preview in _checked:
            frac = _image_domain_mask_fraction(preview)
            if frac >= _MASK_IN_IMAGE_MIN_ZERO_FRAC:
                # Name the guilty segment: on a split tensor the reader needs to
                # know WHICH half is in the wrong domain to act on the finding.
                _where = f" in channel segment {_label!r}" if _label else ""
                dom_note = (
                    f"⚠ DOMAIN SUPERPOSITION: {frac:.0%} of interior image lines are "
                    f"zero{_where} — a k-space undersampling mask appears to have "
                    "been applied in the IMAGE domain (expected k-space). "
                    "See fft_ops.fft2c/ifft2c."
                )
                r.note = f"{r.note}; {dom_note}" if r.note else dom_note
        reports.append(r)

    # Text report
    text = _format_text_report(
        reports,
        step=step,
        tag=tag,
        paradigm=paradigm,
        run=run_identity(),
        extra=extra,
        provenance=provenance,
        model_input_contract=model_input_contract,
    )
    (snap_dir / "snapshot.txt").write_text(text)

    # JSON companion
    if cfg.save_json:
        json_payload = {
            "step": step,
            "tag": tag,
            "paradigm": paradigm,
            # Which run wrote this (#1299). Stamped INTO the snapshot rather
            # than left to the sibling `provenance.json`, which is written once
            # per launch and overwritten -- so relaunching an arm into the same
            # directory used to rewrite the identity of every snapshot already
            # there, and two snapshots in one directory could not be assumed to
            # share a run, a config, or a commit.
            "run": run_identity(),
            "extra": _coerce_extra(extra or {}),
            # Top-level and NOT passed through `_coerce_extra`: this record is
            # deliberately nested (declared / applied / incomplete) and the
            # coercion flattens every non-scalar to `str(v)`, which would turn
            # the transform list into one unparseable line.
            "provenance": provenance,
            # Also top-level and also bypassing `_coerce_extra`, for the same
            # reason: the verdict is nested (declared / applied / status) and
            # the coercion would `str(v)` each half into one unparseable line.
            # It is omitted entirely when the caller passed none, so a snapshot
            # that is not the model-input one does not grow a null key.
            **(
                {"model_input_contract": model_input_contract}
                if model_input_contract is not None
                else {}
            ),
            "tensors": [
                {
                    "name": r.name,
                    "shape": r.shape,
                    "dtype": r.dtype,
                    "device": r.device,
                    "is_complex": r.is_complex,
                    "even_channel_count": r.even_channel_count,
                    "nan_count": r.nan_count,
                    "inf_count": r.inf_count,
                    "finite_count": r.finite_count,
                    "min": r.min,
                    "max": r.max,
                    "mean": r.mean,
                    "std": r.std,
                    "abs_max": r.abs_max,
                    "note": r.note,
                }
                for r in reports
            ],
        }
        (snap_dir / "snapshot.json").write_text(
            json.dumps(json_payload, indent=2, default=_json_default)
        )

    # Image previews (best-effort, never raises)
    if cfg.save_images:
        try:
            from torchvision.utils import save_image

            for name, t, unit_kspace, unit_auth, unit_log in render_units:
                # Reuse the k-space preview rendered for the domain check above;
                # render the (image-domain) remainder here.
                preview = previews.get(name)
                if preview is None:
                    preview = _render_image_preview(
                        t,
                        in_kspace=unit_kspace,
                        log_scaled=unit_log,
                        authoritative=unit_auth,
                    )
                if preview is None:
                    continue
                save_image(
                    preview,
                    str(snap_dir / f"{_safe_name(name)}.png"),
                    nrow=min(4, preview.shape[0]),
                )
        except Exception as exc:
            logger.debug("debug-snapshot image step skipped: %s", exc)

    logger.info(
        "[debug-snapshot] %s wrote %d-tensor snapshot to %s",
        paradigm,
        len(reports),
        snap_dir,
    )
    return snap_dir


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _coerce_extra(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce non-JSON-able extras to strings / floats."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            try:
                out[k] = v.detach().cpu().tolist() if v.numel() < 16 else float(v.mean())
            except Exception:
                out[k] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _format_provenance(provenance: dict[str, Any] | None) -> list[str]:
    """Render the data-provenance record for ``snapshot.txt``.

    The JSON is the machine-readable copy; this is what someone reading the
    snapshot next to the PNGs actually sees, so ``declared`` and ``applied``
    sit adjacent — a divergence between them is the finding.
    """
    if not provenance:
        return []
    lines = ["Data provenance:", f"  source = {provenance.get('source', '?')}"]
    declared = provenance.get("declared")
    if isinstance(declared, dict):
        for k, v in declared.items():
            lines.append(f"  declared.{k} = {v}")
    applied = provenance.get("applied")
    if isinstance(applied, dict):
        chain = applied.get("dataset_chain")
        if chain:
            lines.append(f"  applied.dataset_chain = {' -> '.join(map(str, chain))}")
        transforms = applied.get("transforms")
        if isinstance(transforms, list):
            names = [str(t.get("name")) for t in transforms if isinstance(t, dict)]
            lines.append(f"  applied.transforms = {', '.join(names)}")
    linearization = provenance.get("model_input_linearization")
    if linearization:
        lines.append(f"  model_input_linearization = {linearization}")
    # Loud on purpose: a reader must never mistake a partial record for a
    # complete one. That mistake is the whole failure mode this record exists
    # to prevent (pitfall #16).
    for gap in provenance.get("incomplete") or []:
        lines.append(f"  ⚠ INCOMPLETE: {gap}")
    if "error" in provenance:
        lines.append(f"  ⚠ ERROR: {provenance['error']}")
    lines.append("-" * 78)
    return lines


def _format_model_input_contract(record: dict[str, Any] | None) -> list[str]:
    """Render the model-input verdict as the first thing after provenance.

    Placed above the tensor table rather than inside ``Extra context`` because
    it is the one line that says whether the table below describes the tensor
    the model was fed. A reader who stops at the header must still see it.
    """
    if not record:
        return []
    declared = record.get("declared") or {}
    applied = record.get("applied") or {}
    status = str(record.get("status", "unresolved")).upper()
    lines = [
        f"Model-input contract [{status}]  key={record.get('model_input_key')!r}",
        f"  declared: channels={declared.get('channels')} "
        f"shape={declared.get('shape')} complex={declared.get('is_complex')}",
        f"  applied:  in_channels={applied.get('in_channels')} "
        f"(from {applied.get('resolved_from')})",
    ]
    detail = record.get("detail")
    if detail:
        lines.append(f"  {detail}")
    lines.append("-" * 78)
    return lines


def _format_text_report(
    reports: list[TensorReport],
    *,
    step: int,
    tag: str,
    paradigm: str,
    extra: dict[str, Any] | None,
    provenance: dict[str, Any] | None = None,
    model_input_contract: dict[str, Any] | None = None,
    run: dict[str, Any] | None = None,
) -> str:
    """Pretty-print the snapshot to a single text file."""
    lines: list[str] = [
        f"# Debug Snapshot — {paradigm}",
        f"step={step}  tag={tag}",
    ]
    if run:
        dirty = " (dirty tree)" if run.get("git_dirty") else ""
        lines.append(
            f"run={run.get('run_id')}  git={(run.get('git_sha') or 'unknown')[:12]}"
            f"{dirty}  started={run.get('started_at')}"
        )
        if run.get("identity_source") != "run_provenance":
            # Said out loud: this id was minted here, not by the pipeline, so it
            # will NOT match any `provenance.json` and must not be correlated
            # with one.
            lines.append(f"  identity_source={run.get('identity_source')}")
    lines.append("=" * 78)
    if extra:
        lines.append("Extra context:")
        for k, v in extra.items():
            lines.append(f"  {k} = {v}")
        lines.append("-" * 78)
    lines.extend(_format_provenance(provenance))
    lines.extend(_format_model_input_contract(model_input_contract))

    header = f"{'name':22s} {'shape':22s} {'dtype':14s} {'min':>10s} {'max':>10s} {'mean':>10s} {'std':>10s} {'NaN':>5s}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in reports:
        shape_str = "x".join(str(s) for s in r.shape) if r.shape else "scalar"
        mn = "—" if r.min is None else f"{r.min:.4g}"
        mx = "—" if r.max is None else f"{r.max:.4g}"
        mu = "—" if r.mean is None else f"{r.mean:.4g}"
        sd = "—" if r.std is None else f"{r.std:.4g}"
        lines.append(
            f"{r.name[:22]:22s} {shape_str[:22]:22s} {r.dtype[:14]:14s} "
            f"{mn:>10s} {mx:>10s} {mu:>10s} {sd:>10s} {r.nan_count:>5d}"
        )
        if r.note:
            lines.append(f"{'':22s} {r.note}")
    lines.append("-" * len(header))
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "ChannelSegment",
    "SnapshotConfig",
    "TensorReport",
    "save_debug_snapshot",
    "snapshot_step_is_due",
]
