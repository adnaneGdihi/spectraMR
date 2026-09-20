"""Training Pipeline - Schema-Driven Strategy Dispatch
==================================================

This module provides a polymorphic training pipeline that dispatches to the
correct strategy based on config.training schema.

[2026-01]: Acts as a Composition Root using strict dependency declaration.
Legacy manual service instantiation has been replaced with `spectramr.bootstrap` builder.
Data extraction now enforces `TrainingBatch` structure.
"""

import csv
import inspect
import json
import logging
import math
import os
import time
import traceback
import weakref
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm import tqdm

from spectramr.infrastructure.services.tensorboard_writer import TensorBoardWriter

logger = logging.getLogger(__name__)

#: Consecutive image-less validation passes before the per-pass WARNING is escalated
#: to an ERROR. One miss is a fluke (an odd batch, an early-terminated pass); three in
#: a row is a broken capture seam that will otherwise ship an image-less run marked OK.
_VISUAL_CAPTURE_MISS_LIMIT = 3

#: Run-scoped state for the image-capture seam, keyed by the strategy instance.
#:
#: These counters used to be set as attributes on ``pipeline`` -- the
#: ``TrainingEnvironment``, which is a ``@dataclass(frozen=True)``. Every
#: assignment therefore raised ``FrozenInstanceError: cannot assign to field
#: '_visual_capture_misses'``, and because the assignment sits INSIDE the
#: "we wanted images and got none" branch, the instrumentation written to
#: surface a silent skip instead crashed the whole run at exactly the moment it
#: was meant to report one. Seven of the eleven paradigms in
#: ``tests/smoke/test_fit_paradigms_smoke.py`` died on it.
#:
#: The state has to outlive a single ``_run_validation`` call (a *consecutive*
#: miss count is the whole point), so it needs a carrier that lives for the run.
#: The strategy is that carrier: mutable, one per run, and already passed to
#: every site that touches these counters. A ``WeakKeyDictionary`` keeps this
#: module from pinning strategies alive after their run ends.
_VISUAL_CAPTURE_STATE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _visual_capture_state(owner: Any) -> dict[str, Any]:
    """Mutable per-run scratch space for the image-capture seam.

    Falls back to a strongly-held entry keyed by ``id`` for an owner that cannot
    be weak-referenced. That fallback deliberately still ACCUMULATES rather than
    handing back a fresh dict: a counter that silently resets every call would
    leave the escalation permanently unreachable, which is pitfall #9 wearing the
    costume of the guard it disables.
    """
    try:
        return _VISUAL_CAPTURE_STATE.setdefault(owner, {})
    except TypeError:
        return _VISUAL_CAPTURE_STATE_BY_ID.setdefault(id(owner), {})


#: Fallback store for non-weak-referenceable owners; see ``_visual_capture_state``.
_VISUAL_CAPTURE_STATE_BY_ID: dict[int, dict[str, Any]] = {}

#: Consecutive degenerate validation outputs before the per-pass WARNING is escalated
#: to an ERROR. Mirrors the image-capture limit above: a single pass can catch a model
#: mid-transient, but a sustained run of them means the arm is producing pictures of
#: nothing while its loss curve stays finite and its SSIM stays plausible.
_OUTPUT_SANITY_MISS_LIMIT = 3


def _resync_scheduler_base_lrs(sched: Any) -> None:
    """Extend an LR scheduler's per-group lists to match its optimizer's current
    ``param_groups`` before ``scheduler.step()``.

    PyTorch LR schedulers snapshot ``base_lrs`` (one entry per param group) at
    construction. ~25 strategies register extra trainable params on the base
    loop's ``opt_g`` / ``opt_d`` via ``add_param_group()`` *after* the scheduler
    is built (TTO, IB-VF, Cycle-Bloch, LOUPE, PILOT, …); the next ``step()`` then
    zips mismatched lengths and raises ``zip() argument 2 is longer than
    argument 1`` (smoke audit 2026-06-03, F7d — exp_vf_tto_v2). This is the
    single universal backstop at the step site: idempotent (appends only the
    shortfall), and it also covers a :class:`WarmupScheduler`'s own
    ``base_lrs`` / ``warmup_start_lr`` / ``warmup_end_lr`` plus its wrapped
    ``main_scheduler``. New groups' base LR is read from ``initial_lr`` (set by
    ``add_param_group(lr=...)``), falling back to the live ``lr``.
    """
    main = getattr(sched, "main_scheduler", None)
    opt = getattr(sched, "optimizer", None) or getattr(main, "optimizer", None)
    if opt is None:
        return
    n_groups = len(opt.param_groups)
    seen: set[int] = set()
    for obj in (sched, main):
        if obj is None:
            continue
        for attr in ("base_lrs", "warmup_start_lr", "warmup_end_lr"):
            seq = getattr(obj, attr, None)
            if isinstance(seq, list) and id(seq) not in seen and len(seq) < n_groups:
                seen.add(id(seq))
                seq.extend(
                    g.get("initial_lr", g.get("lr", 0.0)) for g in opt.param_groups[len(seq) :]
                )


# Set CUBLAS workspace config BEFORE importing torch to fix determinism warnings
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Core imports
from spectramr import bootstrap
from spectramr.config.settings import TrainingSettings
from spectramr.core.metrics.output_sanity import measure_output_sanity
from spectramr.core.module_utils import unwrap_model  # noqa: E402
from spectramr.core.wall_clock import resolve_yield_marker_path  # noqa: E402
from spectramr.data.batch_types import BatchAdapter, TrainingBatch
from spectramr.domain.interfaces.service_interfaces import (
    ICheckpointService,
    ILoggingService,
    IMemoryOptimizationService,
    IMetricsService,
)
from spectramr.infrastructure.builders.directors import CheckpointDirector
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from spectramr.infrastructure.optimization.ema_swap import (  # noqa: E402
    ema_weights_swapped_in,
)

# Re-exported from the infrastructure layer (its canonical home) so existing call
# sites — and the loss-schedule controller, which cannot import leftward from
# pipelines/ (CLAUDE.md #13) — share one alias-resolution SSOT.
from spectramr.infrastructure.services.metric_keys import (  # noqa: F401
    early_stop_monitor_candidates,
)
from spectramr.infrastructure.training.builders.director import (
    TrainingEnvironmentDirector,
)
from spectramr.infrastructure.training.mixed_precision import (  # noqa: E402
    resolve_amp_precision,
)
from spectramr.infrastructure.training.strategies.loss_folding import (  # noqa: E402
    declares_inline_objective,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.infrastructure.validation.config_health_checker import (
    FATAL_HEALTH_CHECKS,
    validate_config_health,
)
from spectramr.infrastructure.validation.resolved_config_artifact import (  # noqa: E402
    write_ledger_failure_sentinel,
    write_resolved_config,
)
from spectramr.models.analysis.extensive_gradient_logger import ExtensiveGradientLogger
from spectramr.models.stability.manager import StabilityManager
from spectramr.pipelines.parallel import is_rank_zero, resolve_data_rank
from spectramr.shared.utils.seed_control import set_global_seed


def _is_epoch_boundary(iteration: int, train_loader_len: int, has_train_loader: bool) -> bool:
    """Whether ``iteration`` lands on a training-epoch boundary.

    True only when a non-empty train loader exists AND the iteration is a
    multiple of its length. The ``has_train_loader`` guard is load-bearing:
    ``train_loader_len`` falls back to ``1`` when the loader is missing/empty,
    and ``iteration % 1 == 0`` is always True — so without the guard a
    missing/empty loader would be treated as a *perpetual* epoch boundary
    (triggering epoch-based validation every step). This preserves the original
    inline semantics after the PIPE-1 hoist
    (backlog_wasted_compute_audit_2026_05_29).
    """
    return has_train_loader and (iteration % train_loader_len == 0)


def validation_can_fire(
    *,
    eval_interval: int,
    max_iterations: int,
    eval_on_epoch: bool = False,
    eval_interval_epochs: int = 1,
    train_loader_len: int = 1,
    has_train_loader: bool = False,
) -> bool:
    """Whether ANY validation event is reachable inside this run's budget.

    The loop iterates ``range(first_iteration, max_iterations + 1)`` and fires
    validation from two additive gates (``TrainingLoop.run``): the step gate
    ``iteration % eval_interval == 0`` and, when ``eval_on_epoch``, the epoch
    gate ``_is_epoch_boundary(...) and epoch % eval_interval_epochs == 0``.
    Neither gate is forced on the first or last iteration -- unlike the logging
    and train-metric gates, which are. So an ``eval_interval`` above the budget
    yields *zero* validation events, and the loop has no way to notice.

    That silence is expensive rather than merely untidy: validation is what
    feeds early stopping and ``checkpoint_best.pt``. With no event, neither
    mechanism ever runs, the unresolvable-monitor ``RuntimeError`` that is
    supposed to cap the cost of a typo'd monitor at one validation pass is
    unreachable (it lives *inside* an event), and the run still exits 0
    reporting success. That is the exact consequence pair #178 ruled FATAL.

    Reachability, gate by gate:

    * **Step gate.** Iterations run 1..``max_iterations`` and the schema pins
      ``interval_steps >= 1``, so the earliest event is at ``eval_interval``
      itself. It is reachable iff ``0 < eval_interval <= max_iterations`` --
      independently of ``first_iteration``, since a resumed run can only lose
      the events it already ran.
    * **Epoch gate.** Boundaries land at ``k * train_loader_len``, where the
      loop's own ``epoch = iteration // train_loader_len`` makes the epoch
      index exactly ``k``. The smallest ``k >= 1`` satisfying
      ``k % eval_interval_epochs == 0`` is ``eval_interval_epochs``, so the
      first epoch-gated event sits at ``eval_interval_epochs *
      train_loader_len``. ``has_train_loader`` is load-bearing for the same
      reason it is in :func:`_is_epoch_boundary`.

    The caller owns the two conditions that are not arithmetic: a validation
    loader must exist, and sanity-check mode disables the epoch gate.
    """
    step_gate_fires = 0 < eval_interval <= max_iterations
    return step_gate_fires or epoch_validation_can_fire(
        max_iterations=max_iterations,
        eval_on_epoch=eval_on_epoch,
        eval_interval_epochs=eval_interval_epochs,
        train_loader_len=train_loader_len,
        has_train_loader=has_train_loader,
    )


def epoch_validation_can_fire(
    *,
    max_iterations: int,
    eval_on_epoch: bool,
    eval_interval_epochs: int = 1,
    train_loader_len: int = 1,
    has_train_loader: bool = False,
) -> bool:
    """Whether the epoch-boundary validation gate reaches a single event.

    Split out of :func:`validation_can_fire` because the caller needs the two
    gates separately: the step gate's *degenerate* case (exactly one event, on
    the final iteration) is only worth warning about when the epoch gate is not
    quietly supplying earlier ones.
    """
    return (
        eval_on_epoch
        and has_train_loader
        and train_loader_len > 0
        and max(1, eval_interval_epochs) * train_loader_len <= max_iterations
    )


def ema_should_update(iteration: int, update_frequency: int, warmup_steps: int = 0) -> bool:
    """Whether to apply the EMA shadow-weight update on ``iteration``.

    Honors two previously-inert knobs (CLAUDE.md pitfall #15):
    * ``ema.update_frequency`` — update on every ``update_frequency``-th step
      (non-positive/missing → 1, i.e. every step);
    * ``ema.warmup_steps`` — skip EMA updates for the first ``warmup_steps``
      steps (the shadow holds its init value until then). Default 0 → no skip,
      preserving the previous every-step behavior exactly.

    NOTE: the adaptive-EMA path consumes ``warmup_steps`` internally, so callers
    must pass ``warmup_steps=0`` there to avoid double-counting.
    """
    if warmup_steps and iteration < warmup_steps:
        return False
    freq = update_frequency if (update_frequency and update_frequency > 0) else 1
    return iteration % freq == 0


#: Per-sample fields the Pattern-A validation seam may forward to a strategy's
#: ``validation_step``. Single source of truth for the seam AND its unit test.
_VALIDATION_FORWARD_FIELDS: tuple[str, ...] = (
    "b0_map",
    "b1_map",
    "trajectory_measured",
    "trajectory_nominal",
    "field_strength",
    "field_strength_target",
    "contrast_id",
    # B-1.1 multi-source consensus: the travelling-volunteer tuple, so validation can
    # render the variance-reduced consensus mean over all source fields.
    "sources",
)


def select_validation_extra_fields(val_batch: Any, vs_params: Any) -> dict[str, Any]:
    """Return the per-sample fields to forward to a Pattern-A ``validation_step``.

    A field is forwarded ONLY when (a) the strategy's ``validation_step`` declares
    it (``key in vs_params``) AND (b) the batch carries it (``key in val_batch``).
    That double gate is load-bearing: it keeps train/val conditioning aligned
    (forwarding ``field_strength``/``field_strength_target``/``contrast_id`` so the
    metric grades the actually-conditioned forward, pitfall #18) and prevents a
    strategy that does not declare a field from receiving an unexpected kwarg.
    ``val_batch`` is a ``TrainingBatch`` whose ``in``/``.get`` fall back to its
    metadata, so the same extraction works for dict- or TrainingBatch-yielding
    loaders.

    ``batch_data`` is forwarded on the same declaration gate but is deliberately
    NOT one of the per-sample fields: those are tensors lifted *out* of the batch,
    whereas this is the batch itself. It is gated only on the declaration (not on
    batch shape) because a strategy that asks for the batch should receive
    whatever the loader produced, and every reader goes through
    ``read_batch_field``, which is shape-agnostic.

    Until this existed, no Pattern-A strategy could see its batch at all: the
    dispatch below passes two tensors, and ``DiffusionTrainingStrategy`` tried to
    recover the batch from ``batch = (input_batch, target_batch)`` -- a tuple, so
    its ``isinstance(batch, dict)`` / ``hasattr(batch, "metadata")`` shim could
    never fire. ``batch_data`` was therefore ``None`` for the whole validation
    path, which sent the k-space compensator into its recompute-and-divide branch
    on batches the loader had already normalized.
    """
    out: dict[str, Any] = {}
    if "batch_data" in vs_params:
        out["batch_data"] = val_batch
    if not hasattr(val_batch, "get"):
        return out
    for _fk in _VALIDATION_FORWARD_FIELDS:
        if _fk in vs_params and _fk in val_batch:
            out[_fk] = val_batch.get(_fk)
    return out


#: Checkpoint-discovery spellings accepted by ``--resume``. Anything else must
#: be an existing file; a value that is neither raises rather than being read as
#: a path that happens not to exist yet (non-negotiable 3).
_RESUME_DISCOVERY_MODES = frozenset({"auto", "if-present"})


def _stamp_resume_record(run_dir: Path, record: dict[str, Any], logger_: Any = None) -> None:
    """Record what the resume actually did, beside ``provenance.json``.

    Appends rather than overwrites: a 120 h wall-clock chain produces one entry
    per link, and the sequence of ``start_iteration`` values is what shows the
    chain advanced instead of restarting. Fail-open — a stamp must never take
    down a run that is otherwise ready to train.
    """
    try:
        target = Path(run_dir) / "resume_history.json"
        history = []
        if target.exists():
            loaded = json.loads(target.read_text())
            history = loaded if isinstance(loaded, list) else [loaded]
        record = {**record, "recorded_at": datetime.now().isoformat(timespec="seconds")}
        history.append(record)
        target.write_text(json.dumps(history, indent=2))
        if logger_ is not None:
            logger_.log_info(f"[Resume] Stamped {record['outcome']} -> {target}")
    except Exception:  # pragma: no cover - a stamp never blocks training
        logger.debug("resume record stamp failed", exc_info=True)


def _write_yield_marker(run_dir: Path, result: dict[str, Any], logger_: Any = None) -> None:
    """Tell the launcher this run stopped early and wants requeueing.

    A file rather than an exit code, because a process-group arm yields inside a
    ``torch.distributed.run`` worker whose status the launcher collapses. WHERE
    the file goes is the launcher's decision, not ours — see
    :func:`~spectramr.core.wall_clock.resolve_yield_marker_path`.
    """
    try:
        target = resolve_yield_marker_path(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "resume_from_iteration": result.get("resume_from_iteration"),
                    "iterations_completed": result.get("iterations_completed"),
                    "yielded_at": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
        if logger_ is not None:
            logger_.log_info(f"[WallClock] Wrote yield marker: {target}")
    except Exception:  # pragma: no cover - a marker never blocks teardown
        logger.debug("yield marker write failed", exc_info=True)


def _determinism_from_config(config: Any) -> bool:
    """Resolve ``training.deterministic`` (reproducible-by-default).

    Mirrors ``main._resolved_determinism`` for the pipeline's seed-controller
    mechanism: the pipeline previously hardcoded ``deterministic=True`` into
    ``set_global_seed``, making the schema knob a pitfall-#15 silent no-op on
    this path. An absent knob resolves True (the historical forced behaviour).
    """
    training = getattr(config, "training", None)
    value = getattr(training, "deterministic", None) if training is not None else None
    return True if value is None else bool(value)


def _percentile_window(img: "torch.Tensor") -> "torch.Tensor":
    """Per-sample window a batch of images to ``[0, 1]`` for TensorBoard/disk.

    Uses the 0.5th to 99.5th percentile so single-pixel outliers (k-space DC
    spikes, Gibbs ringing) cannot squash the anatomy into a narrow band.

    When that percentile span collapses the window falls back to full
    **min-max** before conceding black. A sparse-but-real image — e.g. a
    validation panel masked to the anatomical object support, where >99.5% of
    the frame is exactly zero — has a degenerate percentile span but is *not*
    constant, and returning zeros for it renders a plausible-looking black PNG
    that hides the signal (CLAUDE.md #9: no silent fallbacks). Only a genuinely
    constant frame, which has no window to stretch, renders black.

    Regression: the 2026-07 ``exp_vf_01_subvoxel_superres_v2`` run logged
    ``target_mag range before norm: [0.0000, 92.7404] -> after: [0.0000,
    0.0000]`` and wrote every validation ``real``/``fake`` PNG as pure black.
    """
    if img.numel() == 0:
        return torch.zeros_like(img, dtype=torch.float32)

    result = torch.empty_like(img, dtype=torch.float32)
    for i in range(img.shape[0]):
        sample = img[i].float()
        flat = sample.reshape(-1)

        # F-VIZ-EMPTY / 2026-05-20 — torch.quantile rejects an empty tensor. A
        # per-sample slice can be empty when any non-batch dim is 0 (e.g. the
        # broken ``[2, 0, 256, 256]`` predictions from
        # experiment_52_tissue_diffusion_bloch_dc). The outer ``numel() == 0``
        # guard misses that case.
        if flat.numel() == 0:
            result[i] = torch.zeros_like(sample, dtype=torch.float32)
            continue

        vmin = torch.quantile(flat, 0.005).item()
        vmax = torch.quantile(flat, 0.995).item()
        if vmax - vmin < 1e-8:  # degenerate percentile span → widen to min-max
            vmin, vmax = float(flat.min()), float(flat.max())
        rng = vmax - vmin

        if rng < 1e-8:  # genuinely constant frame: nothing to window
            result[i] = torch.zeros_like(sample)
        else:
            result[i] = ((sample - vmin) / rng).clamp(0, 1)

    return result


def _preprocess_validation_tensor(t: Any, config: Any) -> Any:
    """Validation tensor prep: ComplexGuard → 5D→4D (2D nets only) → square-pad.

    The 5D→4D depth-into-batch flatten targets 2D networks. Volumetric models
    (``model.spatial_dims == 3`` — e.g. the SLAT slab→volume VAE) natively
    consume 5D ``[B,C,H,W,D]`` slabs and RE-INFLATE depth in their decoder, so
    flattening their input desyncs the 5D pred from the flattened 4D target and
    trips the VAE shape-mismatch raise (2026-07 ldm slab arms, vae.py:340).
    Gate the flatten on non-3D; an absent ``spatial_dims`` defaults to 2 so 2D
    arms keep their historical behaviour.
    """
    if not isinstance(t, torch.Tensor):
        return t
    # 1. ComplexGuard — interleave real/imag into the channel dim.
    if torch.is_complex(t):
        if t.ndim == 4:
            B, C, H, W = t.shape
            t = torch.stack([t.real, t.imag], dim=2).view(B, C * 2, H, W)
        elif t.ndim == 5:
            B, C, H, W, D = t.shape
            t = torch.stack([t.real, t.imag], dim=2).view(B, C * 2, H, W, D)
    # 2. 5D→4D for 2D networks only (volumetric spatial_dims==3 keeps depth).
    spatial_dims = config.model.spatial_dims
    if t.ndim == 5 and spatial_dims != 3:
        B, C, H, W, D = t.shape
        t = t.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)
    # 3. Square dimension padding (for Hilbert) — use data/patch dims.
    data_cfg = getattr(config, "data", None)
    if t.ndim == 4 and data_cfg is not None:
        _, _, H, W = t.shape
        target_size = None
        if (
            data_cfg.sampling.patch_size
            and isinstance(data_cfg.sampling.patch_size, (list, tuple))
            and len(data_cfg.sampling.patch_size) >= 2
        ):
            target_size = data_cfg.sampling.patch_size[:2]
        # An `elif hasattr(data_cfg, "image_size")` branch stood here. It was
        # doubly dead: DataConfigSchema never declares `image_size` and is
        # extra="ignore", so hasattr was permanently False; and `migrate_legacy_sizes`
        # (data.py) already folds img_size / target_size / image_size into
        # `patch_size`, which the branch above consumes. Redeclaring the key
        # would resurrect a spelling the migration exists to retire.

        if target_size is not None and (target_size[0] != H or target_size[1] != W):
            pad_h = max(0, target_size[0] - H)
            pad_w = max(0, target_size[1] - W)
            if pad_h > 0 or pad_w > 0:
                t = torch.nn.functional.pad(
                    t, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
                )
    return t


class ExperimentStatusRefusedError(RuntimeError):
    """Raised when an arm's own ``metadata.status`` says it cannot produce a result."""


def _refuse_unlaunchable_status(config: Any, allow_status: str | None = None) -> str | None:
    """Refuse to launch an arm whose metadata says its mechanism is not there.

    ~190 inprogress arms admitted in free text that their mechanism was
    unimplemented, inert or blocked, and nothing read the text: the arm trained,
    reported a number, and the number read as the mechanism's result (CLAUDE.md
    non-negotiable 16). ``metadata.status`` is now a closed vocabulary
    (``EXPERIMENT_STATUSES``) and the statuses in ``LAUNCH_REFUSED_STATUSES``
    are refused here, before the container is built, unless ``allow_status``
    names the status explicitly (a wiring smoke of a stub is a legitimate run;
    a silent one is not).

    Returns:
        The status (``None`` when the arm declares none), for the provenance stamp.
    """
    from spectramr.config.schemas.base import LAUNCH_REFUSED_STATUSES

    metadata = getattr(config, "metadata", None)
    status = getattr(metadata, "status", None) if metadata is not None else None
    if status in LAUNCH_REFUSED_STATUSES and allow_status != status:
        reason = getattr(metadata, "status_reason", None)
        raise ExperimentStatusRefusedError(
            f"metadata.status={status!r}"
            + (f" ({reason})" if reason else "")
            + ": this arm's own metadata says its mechanism cannot produce a result "
            "today, so a number from this run would be read as something it is not. "
            f"Refusing to launch. Pass --allow-status {status} to run it anyway as a "
            "wiring / smoke exercise (the status is stamped into provenance either way)."
        )
    return status


def run_training_pipeline(
    config: Any,
    device: str | None = None,
    is_sanity_check: bool = False,
    resume_path: str | None = None,
    env: Any | None = None,
    strategy: Any | None = None,
    allow_status: str | None = None,
) -> dict[str, Any]:
    """Run the complete training pipeline using Strategy Pattern.

    Acts as the Composition Root for the training process.

    Args:
        config: TrainingSettings configuration object (or a dict, normalized via
            ``settings_from_dict``).
        device: Device override (cuda/cpu/auto).
        is_sanity_check: If True, run in sanity-check (overfit) mode.
        resume_path: Path to checkpoint for resume, or 'auto' to find latest.
        env: Optional pre-built :class:`TrainingEnvironment` (scripting / in-process
            path). When provided, the config-driven ``TrainingEnvironmentDirector``
            is skipped and this environment drives the SAME loop. Container,
            services, and provenance are still built unconditionally.
        strategy: Optional pre-built training strategy. When provided, the
            ``TrainingStrategyFactory`` is skipped. Defaults to ``None`` — the
            recommended scripting path injects ``env`` only and lets the pipeline
            resolve the strategy from ``config.training``.
        allow_status: Name of a launch-refused ``metadata.status`` to run anyway
            (``needs_implementation`` / ``inert`` / ``blocked``), for wiring
            smokes. Anything else leaves the refusal in force.
    """
    # Normalize a dict config FIRST (before any attribute access). Route through
    # settings_from_dict so the coil-processing bridges + model_type check run —
    # a raw TrainingSettings(**config) would skip those dict-level transforms
    # (pitfall #15). Previously this cast happened after ``config.run.seed`` below,
    # so a dict config crashed; doing it here also fixes that latent bug.
    if isinstance(config, dict):
        config = TrainingSettings.settings_from_dict(config)

    # 0-. Status gate: an arm whose own metadata says it cannot produce a result
    # does not run (cohort review 2026-09-02, T0.4). Before seeding, container
    # or model, so a refused launch costs nothing.
    experiment_status = _refuse_unlaunchable_status(config, allow_status)
    if experiment_status is not None:
        logger.info("[Pipeline] metadata.status=%s", experiment_status)

    # 0. Global Seeding
    # ONE seed. `training.seed` used to win over the root `seed:`, so 96 arms
    # setting the root key were writing the loser; both now rename into
    # `run.seed` (schemas/renames.py) and there is nothing left to prefer.
    seed = config.run.seed

    # No rank offset HERE, on purpose. This runs before the model is built
    # (stage 3), and identical weights across ranks is a correctness requirement,
    # not a nicety: FSDP shards from rank 0's parameters and DDP's initial
    # broadcast is skippable, while strategy-owned auxiliary modules built at
    # stage 7 are never broadcast at all. The per-rank data stream is seeded
    # separately once construction is finished — see the second call below.
    set_global_seed(seed, deterministic=_determinism_from_config(config))
    # Wall-clock anchors for the run summary (duration + throughput). Cheap
    # stdlib calls; never gated on anything that can fail.
    run_started_at = datetime.now().astimezone()
    run_start_perf = time.perf_counter()
    # Phase-4 logging-hygiene banner — replaces ad-hoc "===" lines.
    try:
        from spectramr.infrastructure.logging import banner as _banner

        _banner(
            f"Training pipeline · seed={seed}",
            logger=logger,
        )
    except Exception:  # pragma: no cover - logging never blocks training
        logger.info(f"[Pipeline] Global seed set to {seed} (Deterministic Mode Enabled)")

    # Run identity, emitted HERE and not with the provenance banner below.
    # `build_container` (a few lines down) runs LoggingService.setup, which
    # pushes `logging.sinks.level` onto the root logger, every existing logger
    # and every handler — so on an arm setting `level: warning` this is the last
    # point at which an INFO line reaches the console at all. What parallelism a
    # run got is not per-step narration that a quiet run is entitled to drop.
    from spectramr.infrastructure.logging.provenance import log_startup_summary

    log_startup_summary(config, logger=logger)

    # 1. Build Container & Resolve Services
    # (dict configs were already normalized to TrainingSettings at function top.)

    # 1.5 Workflow maturity gate — a STUB regime cannot run any pipeline and an
    # EVAL_ONLY regime cannot be trained. Raises WorkflowNotImplementedError
    # (no-op when the arm declares no workflow: block).
    from spectramr.domain.workflows import enforce_pipeline_maturity_for_config

    enforce_pipeline_maturity_for_config(config, "train")

    # 1a. Run Configuration Health Check (FAIL-FAST on the checks that make a run
    # impossible). Those abort before the model + data pipeline are instantiated;
    # every other error falls through with a warning so that configs with
    # non-fatal issues still run.
    #
    # Which checks are terminal is the CHECKER's knowledge, not the pipeline's —
    # it lives in FATAL_HEALTH_CHECKS beside the checks themselves. This used to
    # be the literal name "domain_alignment" here, which meant a new fail-fast
    # check had to be discovered and wired in a second file; deepspeed_extra_
    # installed was written, its message already said the run would die after the
    # environment was built, and the pipeline built it anyway.
    health_report = validate_config_health(config)
    fatal_errors = [r for r in health_report.errors if r.check_name in FATAL_HEALTH_CHECKS]
    if fatal_errors:
        error_msgs = "; ".join(r.message for r in fatal_errors)
        # The fix hint is what makes the abort actionable (e.g. the exact
        # ``pip install`` line), and a cluster user reads the returned error, not
        # only the log — so carry it into both.
        hints = " ".join(f"fix: {r.fix_hint}" for r in fatal_errors if r.fix_hint)
        checks = ", ".join(sorted({r.check_name for r in fatal_errors}))
        logger.error(
            "[Pipeline] FATAL: pre-flight check failed (%s): %s %s",
            checks,
            error_msgs,
            hints,
        )
        return {
            "error": f"[{checks} Pre-Flight] {error_msgs} {hints}".strip(),
            "success": False,
        }
    if not health_report.passed:
        logger.warning("[Pipeline] WARNING: Config health check found non-fatal errors (see logs)")

    # Bootstrap Services
    container = bootstrap.build_container(config, device=device)

    logging_service = container.resolve(ILoggingService)
    checkpoint_service = container.resolve(ICheckpointService)
    metrics_service = container.resolve(IMetricsService)
    memory_service = container.resolve(IMemoryOptimizationService)

    # Log startup
    run_name = "unnamed_run"
    if config.logging.identity.run:
        run_name = config.logging.identity.run
    elif config.metadata:
        run_name = config.metadata.name or "unnamed_run"

    logging_service.log_info(f"Training run initialized: {run_name}")

    # Capture run provenance (git/env/host/config-hash/effective-batch) up
    # front so even a build-time crash leaves a forensic trace. Model + data
    # sizes are filled in once the environment is built (below). Fail-open:
    # provenance must never block training.
    provenance: dict[str, Any] = {}
    try:
        from spectramr.infrastructure.logging.provenance import collect_run_provenance

        provenance = collect_run_provenance(
            config,
            seed=seed,
            device=device,
            run_name=run_name,
            started_at=run_started_at,
        )
    except Exception:  # pragma: no cover - provenance never blocks training
        logger.debug("run provenance capture failed", exc_info=True)

    # Publish that identity so every debug snapshot this process writes carries
    # it (#1299). The record is HANDED OVER rather than rebuilt downstream: a
    # snapshot that minted its own id would differ from `provenance.json`'s by
    # its timestamp alone, and one run with two identities is worse than one
    # with none. Fail-open like everything else on this path -- a snapshot with
    # a fallback identity still says which run it is not from.
    try:
        from spectramr.infrastructure.training.snapshot_provenance import (
            set_run_identity,
        )

        set_run_identity(provenance)
    except Exception:  # pragma: no cover - identity never blocks training
        logger.debug("run identity publication failed", exc_info=True)

    # Per-rank device inventory. `all_gather_object` is a COLLECTIVE, so its
    # placement is load-bearing rather than stylistic: it sits here, before the
    # pipeline build, because every later provenance site is inside `if
    # provenance:` *and* inside the build's `try:` -- and both of those guards
    # are rank-divergent. `collect_run_provenance` fail-opens per rank, so
    # `provenance` can be `{}` on one rank and populated on another; a build
    # failure returns early on one rank only. Either asymmetry strands the
    # remaining ranks at the collective forever. Here the only guard is
    # `is_initialized()`, inside the helper, which is uniform by construction.
    #
    # Note this is also why the gather is NOT rank-gated even though the *write*
    # below is: gating a collective on rank 0 is a guaranteed hang.
    try:
        from spectramr.infrastructure.logging.provenance import rank_device_inventory

        _rank_devices = rank_device_inventory()
        if _rank_devices and provenance:
            provenance["rank_devices"] = _rank_devices
    except Exception:  # pragma: no cover - provenance never blocks training
        logger.debug("per-rank device inventory failed", exc_info=True)

    # Where this run's log actually went. `logging.sinks.dir` is authoritative
    # over the run directory -- correct per non-negotiable 3b, since a declared
    # value must not be replaced by a caller default -- so the log routinely
    # lands somewhere other than beside the artifacts it describes, and nothing
    # recorded which path won. A run could therefore ship provenance, a resolved
    # config, TensorBoard events, debug snapshots and PNGs while its log was
    # unfindable from any of them. `relocated_from` is populated only when the
    # declared directory was unwritable and the whole log was moved to a temp
    # dir, which is wiped at compute-node teardown: recorded rather than
    # silently omitted, so the absence is attributable after the fact.
    try:
        if provenance:
            _log_record: dict[str, Any] = {
                "resolved_path": getattr(logging_service, "resolved_log_path", None),
                "declared_sinks_dir": getattr(
                    getattr(getattr(config, "logging", None), "sinks", None),
                    "dir",
                    None,
                ),
            }
            _relocated = getattr(logging_service, "log_dir_relocated_from", None)
            if _relocated:
                _log_record["relocated_from"] = _relocated
                _log_record["incomplete"] = [
                    "the declared log directory was not writable; the log was "
                    "moved to a temporary directory that a compute node wipes "
                    "at job teardown"
                ]
            provenance["logging"] = _log_record
    except Exception:  # pragma: no cover - provenance never blocks training
        logger.debug("log destination provenance failed", exc_info=True)

    # 2. Initialize Helper Services (that aren't in DI yet or are local)

    # 3. Build Training Pipeline using Director (Phase 2)
    # Orchestrates all training components: models, optimizers, losses, data, physics
    # =====================================================================
    # PHASE 3: TrainingEnvironmentDirector Integration
    # Builds complete training environment with:
    # - Registry-dispatcher pattern for extensible architecture
    # - Immutable component management
    # - Direct TrainingEnvironment usage (no adapter wrappers)
    # =====================================================================
    try:
        if env is not None:
            # Scripting / in-process path: reuse a pre-built TrainingEnvironment
            # instead of running the config-driven director, then drive the SAME
            # loop below. (Container + services were already built above.)
            pipeline = env
        else:
            director = TrainingEnvironmentDirector(config)
            pipeline = director.build_environment()

        generator = pipeline.models.get("generator")
        optimizer_g = pipeline.optimizers.get("opt_g")

        logging_service.log_info(
            f"Training environment built: "
            f"generator={type(generator).__name__}, "
            f"optimizer_g={type(optimizer_g).__name__}, "
            f"train_loader with batch size {config.data.loader.batch_size}"
        )

        # Augment provenance with model + dataset sizes now that they exist,
        # then emit the human-scannable provenance banner. Fail-open.
        try:
            from spectramr.infrastructure.logging.provenance import (
                count_parameters,
                describe_dataloader,
                log_provenance,
            )

            # The ``device`` stamped above is the REQUESTED one — the CLI's
            # ``--device``, which is None whenever the caller did not pass it.
            # The accelerated-run contract (non-negotiable 9b) wants the
            # RESOLVED device on the record, and only ``pipeline`` knows it.
            # Without this every cluster run shipped ``device: null`` while
            # training on a V100, so provenance could not answer "did this run
            # on an accelerator?" — the one question it exists to answer.
            if provenance:
                resolved_device = getattr(pipeline, "device", None)
                if resolved_device is not None:
                    provenance["device"] = str(resolved_device)

                # Resolved parallelism, for the same reason as the device: the
                # config says what was ASKED for, only the runtime knows what
                # was BUILT. Every plugin computes this record during `adopt`
                # and it was being thrown away -- so an FSDP run and a
                # single-process run produced byte-identical provenance, and
                # the one question a multi-GPU run's record must answer ("did
                # this actually shard?") had no answer on disk.
                # MERGE, do not replace. The plugin's record is thin -- often
                # just `{"strategy": ...}` plus a few strategy-specific keys --
                # while `parallel_provenance` had already resolved `rank`,
                # `local_rank`, `launcher`, `initialized`, `backend`,
                # `node_count` and the DECLARED device/node counts. Overwriting
                # discarded exactly the fields that answer "why does this say
                # 1 GPU when I asked for 4", leaving a record that named the
                # strategy and nothing else. The plugin wins collisions: where
                # both speak, the runtime is the authority.
                parallel_runtime = getattr(pipeline, "parallel", None)
                if parallel_runtime is not None:
                    _plugin_record = dict(getattr(parallel_runtime, "provenance", None) or {}) or {
                        "strategy": getattr(parallel_runtime, "strategy", None)
                    }
                    provenance["parallel"] = {
                        **(provenance.get("parallel") or {}),
                        **_plugin_record,
                    }

                # Resolved compilation, for the same reason as the two above:
                # the config says compile.enabled, only the build knows WHERE
                # the wrap landed and whether it landed at all. `DDP(compile(m))`
                # and `compile(DDP(m))` produce identical YAML and different
                # runs, and the director was assembling this record and
                # dropping it.
                from spectramr.infrastructure.logging.provenance import (
                    compile_provenance,
                )

                provenance["compile"] = compile_provenance(
                    config, getattr(pipeline, "compile", None)
                )

            if provenance:
                # ``generator`` gates the PARAMETER COUNT only. It used to gate
                # this whole block, so the in-process ``env=`` entry above (the
                # one path where no model is built here) silently lost its data
                # counts AND its banner -- and "fixed for all strategies" has to
                # mean every entry point, not just the CLI one.
                if generator is not None:
                    provenance["model"] = count_parameters(generator)
                loaders = getattr(pipeline, "data_loaders", {}) or {}
                # ``is not None``, not truthiness: a DataLoader defines
                # ``__len__``, so an EMPTY loader is falsy and used to be
                # recorded as ``None`` -- indistinguishable from "not built".
                # ``batches: 0`` is a finding; ``null`` is a shrug.
                provenance["data"] = {
                    split: describe_dataloader(loaders[split])
                    for split in ("train", "val", "test")
                    if loaders.get(split) is not None
                }
                log_provenance(provenance, logger=logger)
        except Exception:  # pragma: no cover - provenance never blocks training
            logger.debug("run provenance banner failed", exc_info=True)
    except Exception as exc:
        logging_service.log_error(f"Training pipeline build failed: {exc}")
        logger.error("Training pipeline build failed", exc_info=True)
        # Emit whatever provenance we captured (git/env) so a build crash is
        # still traceable to a commit + host in the log.
        try:
            from spectramr.infrastructure.logging.provenance import log_provenance

            if provenance:
                log_provenance(provenance, logger=logger)
        except Exception:  # pragma: no cover
            pass
        return {"error": f"Training pipeline build failed: {exc}", "success": False}

    # 4. Extract pipeline properties directly (no adapter wrapper)
    device_obj = pipeline.device
    logger.info(f"[Pipeline] Using device: {device_obj}")

    # 5. Parallelism is applied by TrainingEnvironmentDirector at two ordered
    # hooks (Stage A before optimizers for FSDP, Stage B after for DP/DDP).
    # It used to happen here, in one pass, after the environment was fully
    # built -- which cannot be correct for FSDP.
    _is_rank_zero = is_rank_zero(config)
    # Resolved once here rather than at the write site, which is inside a
    # `try:` that must not raise on a torch.distributed probe. Same resolver
    # as the data-sharding rank below: one answer to 'which rank am I',
    # because a provenance file named for a different rank than the one that
    # sharded the data would be worse than no per-rank file at all.
    _resolved_rank = resolve_data_rank(config)

    # 6. Stability Manager
    # =====================================================================
    # FIX #4: Direct config access (NO FALLBACK CHAINS)
    # =====================================================================
    # Previous: Used getattr() with defaults for gradient_clip_val and detect_anomalies
    # Problem: Silently defaults missing required fields, hiding config errors.
    #
    # FIX: Use direct access. Config schema ensures these fields exist.

    gradient_clip_val = config.optimization.gradient.clip.value
    detect_anomalies = config.optimization.gradient.detect_anomalies

    stability_manager = StabilityManager(
        clip_grad_norm=gradient_clip_val,
        detect_anomalies=detect_anomalies,
    )
    logging_service.log_info("Stability Manager active")

    # 6. Extract pipeline properties directly
    # Create strategy using factory (pipeline acts as environment)
    # =====================================================================
    # FIX: Pass metrics_service to strategy (SSOT for image saving)
    # =====================================================================
    # Issue: If metrics_service is not passed, strategy tries DI resolution which may fail.
    # This causes silent fallback: images never get saved to disk.
    # Fix: Explicitly pass metrics_service from container.
    if strategy is None:
        factory = TrainingStrategyFactory()
        strategy = factory.create_strategy(
            env=pipeline,
            logging_service=logging_service,
            metrics_service=metrics_service,
            checkpoint_service=checkpoint_service,
        )

    model_type = config.model.model_type
    logging_service.log_info(f"Strategy: {type(strategy).__name__} (model: {model_type})")

    # Resolve losses from pipeline
    # Use the 'losses' property which consolidates losses_dict and single modules
    losses = pipeline.losses

    # Reached only on the ``env=`` scripting path, which skips the director and so
    # never runs ``LossBuilder.validate()``. Both readers ask the strategy the same
    # question (NN17), so a strategy that declares it owns its objective inline is
    # not warned about here either.
    if not losses and not declares_inline_objective(type(strategy)):
        logging_service.log_warning("No losses were built by LossBuilder; training may fail.")

    # Legacy Gradient Logger (Pending refactor to proper service)
    # Using output root from logging service if accessible, or config
    output_dir = "experiments/outputs"
    if config.training and config.training.output_dir:
        output_dir = config.training.output_dir

    if is_sanity_check:
        # Prevent completely overwriting the real experiment output dir
        output_dir = str(Path(output_dir) / "sanity_check")
        logging_service.log_info(f"🧪 SANITY CHECK MODE: Redirecting outputs to {output_dir}")

    # Construct paths map for local usage
    # Write directly to output_dir, not nested under run_name
    run_dir = Path(output_dir)
    paths = {
        "run_output_dir": str(run_dir),
        "csv_log_file": str(run_dir / "logs" / "training_metrics.csv"),
        "log_dir": str(run_dir / "logs"),
    }

    # Clear any yield marker from the PREVIOUS link of this chain before the
    # run starts. A stale marker requeues a run that has already finished.
    try:
        resolve_yield_marker_path(run_dir).unlink(missing_ok=True)
    except OSError:  # pragma: no cover - a read-only dir fails louder below
        logger.debug("could not clear the wall-clock yield marker", exc_info=True)

    # Ensure output directories exist
    try:
        (run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        logging_service.log_info(f"Output directories created at: {run_dir}")
    except Exception as e:
        logging_service.log_warning(f"Failed to create output dirs at {run_dir}: {e}")

    # Persist the run's provenance + the *resolved* config (post-defaults /
    # post-migration SSOT) as self-describing artifacts, so a bundle can be
    # traced back to its commit/host/knobs without re-parsing the source YAML.
    # Both soft-fail — a stamp hiccup must never break a real training run.
    #
    # `resolved_config.json` carries the run's FINAL state; it has never carried
    # the DELTA from what the YAML declared. The `_ledger` block adds that: every
    # key the schema dropped, defaulted or rewrote, plus every kwarg a consumer
    # could not accept. It is an additive top-level key, deliberately not an
    # inline per-key annotation, so existing readers keep working unchanged —
    # `infrastructure/reporting/cohort_ablation.py::_read_resolved_config` pulls
    # `metadata.tags.*` as raw scalars.
    # `run_dir` is rank-INVARIANT: it comes from `config.training.output_dir`
    # on a frozen config, not from a per-rank timestamp. So under DDP every rank
    # wrote the same two paths and the artifact was last-writer-wins -- which
    # looks fine (the file exists, the JSON is valid) while being an arbitrary
    # rank's view. Rank 0's record is the canonical one; the others land beside
    # it under their own names, because they are the only place the per-rank
    # `local_rank`/`device` facts survive and discarding them would make the new
    # inventory unverifiable.
    _provenance_name = (
        "provenance.json" if _is_rank_zero else f"provenance_rank{_resolved_rank}.json"
    )
    try:
        if provenance:
            (run_dir / _provenance_name).write_text(json.dumps(provenance, indent=2, default=str))
            # ...and again under the run's own id. The canonical name keeps
            # working for every existing reader, but it is overwrite-on-launch,
            # so relaunching an arm into the same directory destroys the record
            # of the previous run -- which is how `experiment_11_attention_none`
            # ended up with five launches' artifacts in one directory and no way
            # to tell which config produced the `step_004000` snapshots (#1299).
            # Additive, not a rename: a copy costs a few KB and cannot break a
            # reader, whereas moving the canonical file would.
            #
            # Rank-differentiated for the SAME reason the canonical name above
            # is. `run_dir` is rank-invariant, so an ungated write here has two
            # failure modes and no good one. When ranks agree on `run_id` every
            # rank writes one path and it is last-writer-wins -- an arbitrary
            # rank's view under a name that reads as the run's own record, which
            # is exactly what the block above exists to stop. When they do NOT
            # agree it is worse: `run_id` is `<slug>-<YYYYmmdd_HHMMSS>-<sha>`
            # built from each rank's own `datetime.now()`, and
            # `collect_run_provenance` fail-opens per rank, so two ranks whose
            # calls straddle a second boundary mint different ids and one run
            # scatters N files each claiming to be its record. That is the "one
            # run, two identities" outcome snapshot_provenance calls worse than
            # none.
            _run_id = provenance.get("run_id")
            if _run_id:
                _id_name = (
                    f"provenance_run_{_run_id}.json"
                    if _is_rank_zero
                    else f"provenance_run_{_run_id}_rank{_resolved_rank}.json"
                )
                (run_dir / _id_name).write_text(json.dumps(provenance, indent=2, default=str))
        if hasattr(config, "model_dump") and _is_rank_zero:
            # The artifact's shape is owned by the audit layer, not by this
            # pipeline: it answers an audit question, and two surfaces
            # hand-rolling the same JSON is how the two disjoint validation
            # stacks happened.
            write_resolved_config(
                run_dir,
                config,
                run_id=provenance.get("run_id") if provenance else None,
                health_report=health_report,
                ledger_source="run_training_pipeline",
            )
    except Exception as e:
        logging_service.log_warning(f"Failed to stamp provenance/resolved config: {e}")
        # A ledger that silently fails to record is the very failure class it
        # exists to detect, so the failure becomes a disk artifact rather than a
        # log line that scrolls off a SLURM buffer.
        write_ledger_failure_sentinel(run_dir, e)

    # Ensure device is cast to torch.device for ExtensiveGradientLogger
    if isinstance(device_obj, str):
        device_obj = torch.device(device_obj)

    gradient_logger_service = ExtensiveGradientLogger(
        log_dir=str(run_dir / "analysis"),
        model_type=model_type,
        device=device_obj,
    )

    # 7. Strategy setup (Services are now managed via DI/Environment)
    # stability_manager is available for local pipeline usage if needed

    # 8. TensorBoard is constructed AFTER resume — see below. It needs
    # `start_iteration` for `purge_step`, and building it here would mean a
    # resumed run keeps the pre-crash event tail.

    # 8b. Per-rank data-stream seeding (issue #1124).
    #
    # The second `set_global_seed` of the run, and the reason two `[Accelerator]`
    # lines appear in the log. Stage 0 seeded every rank identically so weight
    # init would agree; from here on the ranks must DISAGREE, or all N processes
    # draw the same random augmentations and patch crops and the effective
    # augmentation diversity of the run is 1/N. `DistributedSampler` is unaffected
    # either way — it partitions by rank without consulting the RNG — so this is
    # about transform randomness, not which samples a rank sees.
    #
    # Placement is pinned on both sides. It must land AFTER stage 7, so every
    # module (generator, discriminator, strategy-owned heads) is already built
    # from the shared seed; and BEFORE stage 9, because checkpoint resume restores
    # the saved RNG state (`core/rng_state.py`, applied by
    # `CheckpointDirector.load_from` on rank 0) and a re-seed after that would
    # throw the resumed stream away. Rank 0 only: the checkpoint carries one
    # process's streams, so the offset below is what keeps the other ranks
    # drawing different augmentations after a resume.
    #
    # DataLoader workers inherit this for free: `core/worker_seeding.py::seed_worker`
    # derives from `torch.initial_seed()`, which is now rank-offset. Same
    # `_determinism_from_config` value as stage 0 so the cuDNN flags cannot flap.
    #
    # Guarded on rank != 0 so this is a NO-OP for every single-process run: rank 0
    # and non-distributed runs keep their current stream (continuing from whatever
    # construction consumed) and their results do not move, while only the ranks
    # that were duplicating rank 0 change. The resulting asymmetry — rank 0's
    # stream continues, ranks >0 restart at `seed + rank` — is intended. All that
    # is required is that the N streams differ; making it unconditional would buy
    # symmetry at the cost of shifting every existing single-GPU baseline.
    _data_rank = resolve_data_rank(config)
    if _data_rank:
        set_global_seed(seed, deterministic=_determinism_from_config(config), rank=_data_rank)

    # 9. Resume from Checkpoint (if requested)
    #
    # `auto` RAISES on an empty directory; `if-present` starts fresh instead.
    # The chain spelling has to be the second one: every link of a wall-clock
    # chain runs the identical command, so link 1 has nothing to resume from
    # while links 2..N do — and softening `auto` would delete the only spelling
    # that can still tell you your checkpoints went missing.
    start_iteration = 0
    _resume_record: dict[str, Any] = {}
    if resume_path:
        try:
            _resume_mode = str(resume_path).strip()
            _resume_path: str | None = _resume_mode

            if _resume_mode in _RESUME_DISCOVERY_MODES:
                checkpoint_dir = str(Path(output_dir) / "checkpoints")
                latest = checkpoint_service.find_latest_checkpoint(checkpoint_dir)
                if latest is None and _resume_mode == "auto":
                    raise FileNotFoundError(
                        f"No checkpoint found in {checkpoint_dir} for auto-resume. "
                        "Use --resume if-present to start fresh when the directory "
                        "is empty (the wall-clock chain spelling)."
                    )
                _resume_path = str(latest) if latest else None
            elif not Path(_resume_mode).exists():
                # A typo resolves to neither a mode nor a file, and must raise
                # rather than be read as a path (non-negotiable 3).
                raise ValueError(
                    f"--resume {_resume_mode!r} is neither an existing checkpoint "
                    f"nor a known discovery mode "
                    f"({', '.join(sorted(_RESUME_DISCOVERY_MODES))})."
                )

            if _resume_path is None:
                logging_service.log_info(
                    f"[Resume] No checkpoint under {output_dir}/checkpoints; "
                    f"--resume {_resume_mode} starts this run FRESH at iteration 0."
                )
                _resume_record = {
                    "declared": _resume_mode,
                    "resolved_checkpoint": None,
                    "outcome": "fresh",
                    "start_iteration": 0,
                }
            else:
                if _resume_mode in _RESUME_DISCOVERY_MODES:
                    logging_service.log_info(
                        f"[Resume] Discovered latest checkpoint: {_resume_path}"
                    )

                # with_strategy lets the director also restore strategy-OWNED
                # learnable modules/params (sfc heads, spin_sde diffusion param,
                # ...) that live on the strategy, not on the generator — see
                # section R design doc.
                #
                # with_parallel_runtime is what selects the checkpoint adapter,
                # and a resume needs it for two distinct reasons. Loudly: a
                # sharded strategy's best checkpoint holds no generic payload at
                # all, so resuming from one without the adapter raises. Quietly,
                # and worse: a PERIODIC checkpoint does carry the generic
                # envelope, so the weights restore and the resume reports success
                # -- while the ZeRO optimizer partitions in the tag directory
                # beside it are never read, silently restarting a resumed run on
                # a zeroed optimizer.
                #
                # `getattr(..., None)` mirrors training_loop.py's own resolution
                # and keeps every single-process arm on DefaultCheckpointAdapter.
                resume_director = CheckpointDirector(config)
                resume_director.with_checkpoint_dir(
                    str(Path(output_dir) / "checkpoints")
                ).with_pipeline(pipeline).with_strategy(strategy).with_parallel_runtime(
                    getattr(pipeline, "parallel", None)
                )

                if not resume_director.load_from(_resume_path):
                    logging_service.log_error(
                        f"[Resume] Failed to load checkpoint: {_resume_path}"
                    )
                    return {
                        "error": f"Failed to load checkpoint: {_resume_path}",
                        "success": False,
                    }

                start_iteration = resume_director._global_step
                _resume_record = {
                    "declared": _resume_mode,
                    "resolved_checkpoint": _resume_path,
                    "outcome": "restored",
                    "start_iteration": start_iteration,
                    "epoch": resume_director._epoch,
                }
                logging_service.log_info(
                    f"[Resume] Restored from checkpoint: "
                    f"epoch={resume_director._epoch}, step={start_iteration}"
                )
                if resume_director._counter_state:
                    logging_service.log_info(
                        f"[Resume] Counter state available: "
                        f"step={resume_director._counter_state.get('current_step')}, "
                        f"epoch={resume_director._counter_state.get('current_epoch')}"
                    )
        except Exception as e:
            logging_service.log_error(f"[Resume] Checkpoint load error: {e}")
            logger.error("Resume checkpoint load failed", exc_info=True)
            return {"error": f"Resume failed: {e}", "success": False}

    # A resume is a fact about the run that provenance.json cannot carry: it is
    # written at stage 8, before this decision exists. Stamped here so a reader
    # can tell a fresh link of a chain from a restored one (non-negotiable 8).
    # Rank-gated: the stamper is a read-modify-write on one shared path, so an
    # ungated call has every rank appending — four entries per link on a 4-rank
    # arm, or a torn read that fails open and records nothing. The process-group
    # arms are the ones that chain, so this is the production path.
    if _resume_record and _is_rank_zero:
        _stamp_resume_record(run_dir, _resume_record, logger_=logging_service)

    # 9b. TensorBoard — the run's single writer.
    #
    # Deliberately after resume: `purge_step=start_iteration` drops the events a
    # crashed run wrote past the checkpoint, so a resumed chart continues rather
    # than folding back on itself.
    #
    # DDP: only rank 0 owns it; otherwise every rank writes the same event dir
    # and the scalars interleave. A non-zero rank gets a writer that is falsy
    # and whose every method is a no-op, so the `if tb_writer:` guards at the
    # write sites keep their exact previous meaning.
    tb_writer = TensorBoardWriter(
        run_dir,
        config.logging,
        is_rank_zero=_is_rank_zero,
        start_iteration=start_iteration,
    )
    if tb_writer:
        logging_service.log_info(f"TensorBoard logging to: {tb_writer.event_dir}")
        # Provenance travels with the events: a TensorBoard pointed at a bare
        # event dir can still answer what produced the curve.
        tb_writer.text(
            "run/config",
            "```yaml\n" + yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False) + "\n```",
        )

    _report_recorder = _build_report_case_recorder(config, strategy)
    _per_case_sink = _build_per_case_metric_sink(config, strategy)

    # 10. Execute Loop — through the TrainingLoop seam (WS-3). Lazy import
    # breaks the cycle (training_loop delegates back to _execute_training_loop
    # for now; PR-2 inlines that body into TrainingLoop.run).
    from spectramr.pipelines.training_loop import TrainingLoop

    try:
        result = TrainingLoop(
            strategy,
            pipeline,
            config,
            model_type,
            tb_writer=tb_writer,
            logging_service=logging_service,
            output_paths=paths,
            checkpoint_service=checkpoint_service,
            metrics_service=metrics_service,
            is_sanity_check=is_sanity_check,
        ).run(start_iteration=start_iteration)
        if tb_writer:
            _write_hparams(tb_writer, config)
            tb_writer.close()

        # Stamp real wall-clock duration + throughput onto the result (the
        # loop only ever set training_time="N/A"). Computed from perf_counter
        # deltas + iteration counts — no GPU sync, safe outside the hot loop.
        duration_sec = time.perf_counter() - run_start_perf
        if isinstance(result, dict):
            iters = result.get("iterations_completed")
            result["training_time"] = _format_duration(duration_sec)
            result["duration_sec"] = round(duration_sec, 2)
            if isinstance(iters, int) and iters > 0 and duration_sec > 0:
                result["iterations_per_sec"] = round(iters / duration_sec, 3)
            if result.get("wall_clock_yield") and _is_rank_zero:
                _write_yield_marker(run_dir, result, logger_=logging_service)

        # ONE writer (#1685), and every write below it too. The sinks write
        # fixed, rank-independent paths (`report_cases/case_*.npz`,
        # `cases_index.json`, `run_summary.json`), so under DDP every rank raced
        # on the same files: a 4-rank run left one good archive and three
        # `Bad CRC-32`. `_is_rank_zero` is the config-driven owner already
        # resolved above (line ~782) and used for provenance; the process group
        # is still alive here -- every teardown site (`cleanup_distributed`,
        # `_destroy_process_group`, `DistributedContext.__exit__`) runs strictly
        # after `run_training_pipeline` returns -- so this is a real guard, not
        # a `not is_initialized()` no-op.
        #
        # Consequence, deliberate: non-zero ranks' cases are dropped rather than
        # merged. These are a handful of qualitative exemplars, not a metric.
        #
        # Safe to gate: unlike the device-inventory `all_gather_object` above,
        # nothing from here to the end of the block is a collective, so no rank
        # is stranded by skipping it.
        #
        # KNOWN LIMIT, stated rather than hidden: each rank validated its own
        # shard, so the cases and per-call rows written here are RANK 0'S SHARD,
        # not the world's. That is the same tradeoff `_drain_cascade_rows`
        # already makes and documents; a cross-rank gather of case arrays is
        # deliberately out of scope rather than approximated silently.
        if _is_rank_zero:
            if _report_recorder is not None:
                subdir = config.logging.report_cases.subdir
                _report_recorder.write(Path(paths["run_output_dir"]), subdir=subdir)
            if _per_case_sink is not None:
                _per_case_sink.write(Path(paths["run_output_dir"]))

        # Rendezvous before the reporting stage. Now that reporting is rank-0
        # only, this is no longer a read-after-write fence *between* ranks -- one
        # rank both writes and reads -- so its job is to hold the others until
        # rank 0's artifacts are on disk, and no rank returns from the pipeline
        # (and tears the process group down) with writes still in flight.
        #
        # It must stay at all-ranks indentation: moved inside the guard above,
        # rank 0 would block in the collective while no other rank ever arrives
        # -- an NCCL timeout, not a no-op, because the group is still alive here.
        # `RankUtility.barrier` is the existing owner of this primitive
        # (non-negotiable 17) and is a no-op when the group is absent or already
        # down. (Imported at module scope already -- a second, function-local
        # import here would be a redundant second binding of the same name.)
        RankUtility.barrier()

        if _is_rank_zero:
            # Phase 4 logging-hygiene: stamp a self-describing run-summary
            # JSON so post-hoc tooling (mosaic, ranking, dispatch logs) can
            # discover what the run was without re-parsing the YAML.
            #
            # ORDERING IS LOAD-BEARING -- this must stay *above* the reporting
            # hook. `run_summary.json` is one of the five artifacts the report
            # aggregator collects (`reporting/aggregator.py`), and two registered
            # plotters read it: `fig_1_16_run_summary_card` (present in all eight
            # task presets) and `fig_1_15_computational_profile` (five of eight).
            # Emitting it *after* the hook meant the file did not exist while the
            # figures were being drawn, so both soft-skipped at training time and
            # rendered only when `report` was re-run by hand afterwards -- the same
            # run yielding a different figure set depending on which entry point
            # drew it. Anything added here that reporting consumes belongs above
            # the hook too.
            _emit_run_summary(
                config,
                run_dir=Path(paths["run_output_dir"]),
                result=result,
                seed=seed,
                logger_=logger,
                provenance=provenance,
                started_at=run_started_at,
            )

            # End-of-training reporting hook (per TODO/report_step).
            # Soft-fails by default so it cannot break a long run's wrap-up.
            #
            # Skipped on a wall-clock yield for two reasons. It would draw the
            # figure set of a FINISHED experiment from a model that is midway
            # through one; and it runs inside the save margin, so a report long
            # enough to reach the wall takes the launcher down with it before
            # the requeue is ever issued.
            if result.get("wall_clock_yield"):
                logger.info(
                    "[WallClock] Skipping end-of-training reporting: this run "
                    "yielded and is not finished."
                )
            else:
                _maybe_run_reporting(
                    config, run_dir=Path(paths["run_output_dir"]), logger_=logger
                )

        return result

    except Exception as e:
        import traceback

        error_msg = f"Critical Training Error: {e}\n{traceback.format_exc()}"
        hint = _async_cuda_fault_hint(e)
        if hint:
            error_msg = f"{error_msg}\n{hint}"
        logging_service.log_error(error_msg)
        logger.error(error_msg, exc_info=True)
        if tb_writer:
            _write_hparams(tb_writer, config)
            tb_writer.close()
        return {"error": str(e), "success": False}


#: CUDA faults the driver reports on a LATER synchronisation than the launch that
#: caused them. The traceback then names whichever op happened to sync, so the
#: frame is evidence of nothing. ``device-side assert`` is included because its
#: own message already says to set the variable; the rest never say so.
_ASYNC_CUDA_FAULTS = (
    "misaligned address",
    "an illegal memory access",
    "unspecified launch failure",
    "device-side assert triggered",
)


def _async_cuda_fault_hint(exc: BaseException) -> str | None:
    """Say that the traceback of an async CUDA fault points at the wrong frame.

    Three ``kspace_cold_diffusion`` arms died with ``CUDA error: misaligned
    address`` on the 2026-09-17 dispatch, all three blaming the same
    ``ChannelAttention`` line -- which is simply where the next synchronisation
    landed. Re-running unchanged reproduces the same useless frame, so the
    operator needs the serialising flag named at the point of failure rather
    than in a doc they have no reason to open.
    """
    text = str(exc)
    if "CUDA error:" not in text and "AcceleratorError" not in type(exc).__name__:
        return None
    if not any(fault in text for fault in _ASYNC_CUDA_FAULTS):
        return None
    return (
        "[async CUDA fault] The traceback above names the operation that "
        "SYNCHRONISED, not the kernel that faulted -- CUDA reports these "
        "asynchronously, so the frame is not the culprit and re-running "
        "unchanged reproduces the same misdirection. Re-run this arm with "
        "CUDA_LAUNCH_BLOCKING=1 to serialise launches and get a traceback that "
        "names the real kernel. If it then lands somewhere else each time, "
        "suspect the node: `cudnn.benchmark` re-autotunes per process, so the "
        "algorithm selection -- and any alignment requirement it carries -- "
        "varies run to run on identical config."
    )


def _format_duration(seconds: float) -> str:
    """Render a wall-clock duration as ``HH:MM:SS`` (or ``Dd HH:MM:SS``)."""
    seconds = max(0, int(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _checkpoint_summary(config: Any, run_dir: Path) -> dict[str, Any]:
    """Where the checkpoints went, and whether they are inside the bundle (#503).

    exp_vf_01 retrieved a **present-but-empty** ``checkpoints/`` directory while
    its log named two files it had written. Both facts were true: the arm's
    ``checkpoint.checkpoint_dir`` pointed outside the collected run directory,
    so the files existed on the cluster and the bundle never contained them.

    An empty directory cannot distinguish that from "no checkpoint was ever
    saved" -- and those need opposite responses (fix the sync vs fix the run).
    Recording the resolved directory, whether it is inside ``run_dir``, and what
    is actually there at the end makes the bundle answer it on its own.

    Never raises: this is a diagnostic in a footer, and a stat() failure on a
    network filesystem must not cost the run its summary.
    """
    out: dict[str, Any] = {
        "dir": None,
        "inside_run_dir": None,
        "files": [],
        "total_bytes": 0,
    }
    try:
        ckpt = getattr(config, "checkpoint", None)
        raw = getattr(ckpt, "checkpoint_dir", None) if ckpt is not None else None
        if raw is None:
            return out
        d = Path(raw).expanduser().resolve()
        out["dir"] = str(d)
        # The load-bearing flag. False means the files, if any, will not travel
        # with the bundle -- which is the exp_vf_01 shape exactly.
        out["inside_run_dir"] = d == run_dir.resolve() or run_dir.resolve() in d.parents
        if d.is_dir():
            files = sorted(p for p in d.glob("*.pt") if p.is_file())
            out["files"] = [{"name": p.name, "bytes": p.stat().st_size} for p in files]
            out["total_bytes"] = sum(f["bytes"] for f in out["files"])
    except Exception:
        pass
    return out


def _emit_run_summary(
    config: Any,
    *,
    run_dir: Path,
    result: dict[str, Any],
    seed: int | None,
    logger_: Any,
    provenance: dict[str, Any] | None = None,
    started_at: datetime | None = None,
) -> None:
    """Emit ``run_summary.json`` -- the run's self-describing footer.

    Stamps the run's primary knobs (seed, mode, strategy, output dir,
    success flag) plus any best-metric summary the training loop wrote
    into ``result``, and folds in the provenance record (run_id, git, host,
    config hash) + wall-clock timing so the footer alone answers "what ran,
    where, from which commit, and for how long". Soft-fails so a path / JSON-
    encoding hiccup never masks an otherwise-successful training run.

    Part of the Phase 4 logging-hygiene pass: every bundle becomes
    self-describing without having to re-parse the source YAML.
    """
    import json

    from spectramr.infrastructure.logging.provenance import (
        hardware_summary as _hardware_summary,
    )

    try:
        run_dir.mkdir(parents=True, exist_ok=True)
        prov = provenance or {}
        summary = {
            "run_id": prov.get("run_id"),
            "run_name": prov.get("run_name")
            or config.logging.identity.run
            or (config.metadata.name if getattr(config, "metadata", None) else None),
            # The arm's declared author, beside the name the line above reads. It was
            # a declared field no code path touched: the reachability ratchet reports
            # a key set by the YAMLs whose only reads cannot execute, and a run record
            # that names the run but not who declared it is the smaller half of the
            # same provenance question (non-negotiable 8).
            "author": (
                getattr(config.metadata, "author", None)
                if getattr(config, "metadata", None)
                else None
            ),
            "seed": seed,
            "success": (bool(result.get("success", True)) if isinstance(result, dict) else None),
            "strategy_class": (
                config.training.strategy_class if config.training is not None else None
            ),
            "model_type": config.model.model_type,
            "output_dir": str(run_dir),
            # --- timing -----------------------------------------------------
            "started_at": (started_at.isoformat(timespec="seconds") if started_at else None),
            "ended_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "duration": (result.get("training_time") if isinstance(result, dict) else None),
            "duration_sec": (result.get("duration_sec") if isinstance(result, dict) else None),
            "iterations_per_sec": (
                result.get("iterations_per_sec") if isinstance(result, dict) else None
            ),
            # --- metrics ----------------------------------------------------
            "best_metrics": (result.get("best_metrics") if isinstance(result, dict) else None),
            "final_loss": (result.get("final_loss") if isinstance(result, dict) else None),
            "error": result.get("error") if isinstance(result, dict) else None,
            # --- traceability (provenance subset) ---------------------------
            "config_sha256": prov.get("config_sha256"),
            "git": prov.get("git"),
            "host": (prov.get("env") or {}).get("hostname"),
            "effective_batch": (prov.get("batch") or {}).get("effective"),
            "model_params": (prov.get("model") or {}).get("total"),
            # GPU count/type + cores/RAM the run actually had. Without it the
            # ``iterations_per_sec`` above is not comparable across arms.
            "hardware": _hardware_summary(prov) or None,
            "slurm": prov.get("slurm") or None,
            # Backend + resolved resources when started via ``spectramr launch``
            # (pitfall #15c); None for a plain ``spectramr train``.
            "launch": prov.get("launch") or None,
            # --- artifacts ---------------------------------------------------
            # #503: an empty `checkpoints/` in a retrieved bundle read as "no
            # checkpoint was saved" when the truth was "written outside the
            # collected directory". Stated here so the bundle answers it alone.
            "checkpoints": _checkpoint_summary(config, run_dir),
        }
        ck = summary["checkpoints"]
        if ck.get("dir") and ck.get("inside_run_dir") is False:
            logger_.warning(
                "Checkpoints were written OUTSIDE the run directory: %s (run_dir "
                "is %s). They will not travel with a retrieved artifact bundle, "
                "and the bundle's checkpoints/ will look empty as though nothing "
                "was saved (#503). Point checkpoint.checkpoint_dir inside the run "
                "directory, or collect that path explicitly.",
                ck["dir"],
                run_dir,
            )
        elif ck.get("dir") and not ck.get("files"):
            logger_.warning(
                "No checkpoint files found in %s at end of run. If the run "
                "reported saving one, it went somewhere else (#503).",
                ck["dir"],
            )
        path = run_dir / "run_summary.json"
        path.write_text(json.dumps(summary, indent=2, default=str))
        logger_.info(
            "Wrote run summary → %s (run_id=%s, took %s)",
            path,
            summary.get("run_id"),
            summary.get("duration"),
        )
    except Exception as exc:
        logger_.warning("run-summary stamp failed (%s)", exc)


def _write_hparams(tb_writer: TensorBoardWriter, config: Any) -> None:
    """Write this run's HParams entry, paired with its final metrics.

    This is the feature that pays for the rest of the TensorBoard surface.
    Pitfall #17 (confounded ablation) is the second-largest failure class in
    this repo's own validation sweep -- 195 findings -- and the HParams view is
    what makes "this variant differs from its control in more than the one knob
    it claims to test" visible across runs instead of per-chart.

    The selection is deliberately the axes an ablation actually varies. Reading
    them off the resolved config rather than `metadata` means what is recorded
    is what the run USED, not what the arm claimed.

    Soft-fails with a warning, like ``_emit_run_summary`` above. Both call
    sites sit in ``run_training_pipeline``'s wrap-up -- one after a successful
    loop, one inside the ``except`` handler -- so a raise here either discards
    a finished run's result or replaces the real training error with this one.
    That is what the ``precision.use_amp`` typo did on 2026-08-12: the success-
    path call raised, the handler logged "Critical Training Error", and then the
    handler's own call raised the same AttributeError uncaught.
    """
    try:
        opt = config.optimization
        # ``precision`` is (enabled, dtype) and ``dtype='float32'`` disables AMP
        # even when ``enabled`` is true -- three states, not two. Ask the
        # resolver BaseTrainingStrategy itself uses, so the recorded axis is
        # what autocast did. (Was ``precision.use_amp``, a field retired by the
        # optimization-block decomposition: RENAMES maps ``optimization.use_amp``
        # -> ``optimization.precision.enabled``.)
        amp_enabled, _ = resolve_amp_precision(opt.precision.enabled, opt.precision.dtype)
        hparams: dict[str, Any] = {
            "model_type": config.model.model_type,
            "training_mode": config.training.training_mode,
            "learning_rate": opt.optimizer.learning_rate,
            "optimizer": opt.optimizer.type,
            "batch_size": config.data.loader.batch_size,
            "amp": amp_enabled,
            "grad_accum": opt.gradient.accumulation_steps,
        }
        # ``metadata`` is the typed ``ExperimentMetadataSchema`` (one owner since
        # the 2026-09-02 cohort review); a bare attribute read, so a renamed field
        # fails loud instead of returning None to the dashboard.
        metadata = config.metadata
        baseline = metadata.baseline if metadata is not None else None
        if baseline:
            # `metadata.baseline` names the sibling this arm is a variant OF, so
            # the dashboard can sort a control next to the arms that cite it.
            hparams["baseline"] = baseline
        tb_writer.hparams(hparams)
    except Exception as exc:
        logger.warning("HParams stamp failed (%s)", exc)


def _build_report_case_recorder(config: Any, strategy: Any):
    """Build + attach a ReportCaseRecorder when reporting + recording are on.

    Returns the recorder (also attached to ``strategy._report_case_recorder``)
    or None when disabled.
    """
    reporting = getattr(config, "reporting", None)
    if reporting is None or not getattr(reporting, "enabled", False):
        return None
    logging_s = getattr(config, "logging", None)
    if logging_s is not None and not (logging_s.report_cases.enabled if logging_s else True):
        return None
    try:
        from spectramr.core.metrics.metric_directions import METRIC_HIGHER_IS_BETTER
        from spectramr.infrastructure.reporting.cases.recorder import ReportCaseRecorder
    except Exception:
        return None
    validation = getattr(config, "validation", None)
    primary = (validation.scoring.primary if validation else "psnr") or "psnr"
    recorder = ReportCaseRecorder(
        n_cases=getattr(reporting, "n_report_cases", 6),
        selection=getattr(reporting, "case_selection", "best_median_worst"),
        primary_metric=primary,
        higher_is_better=METRIC_HIGHER_IS_BETTER.get(primary, True),
        record_volumes=getattr(reporting, "record_volumes", False),
    )
    try:
        strategy._report_case_recorder = recorder
    except Exception:
        return None
    return recorder


def _build_per_case_metric_sink(config: Any, strategy: Any):
    """Build + attach a PerCallMetricSink when reporting + per-case CSV are on.

    Independent of the (bounded) ReportCaseRecorder gate so the unbounded
    ``per_call_metrics.csv`` is produced even when ``n_report_cases=0``. Returns
    the sink (also attached to ``strategy._per_case_metric_sink``) or None.
    """
    reporting = getattr(config, "reporting", None)
    if reporting is None or not getattr(reporting, "enabled", False):
        return None
    if not getattr(reporting, "per_call_metrics", True):
        return None
    try:
        from spectramr.infrastructure.reporting.cases.metric_sink import PerCallMetricSink
    except Exception:
        return None
    sink = PerCallMetricSink(enabled=True)
    try:
        strategy._per_case_metric_sink = sink
    except Exception:
        return None
    return sink


# The end-of-run reporting hook now lives in `infrastructure/reporting/run_hook.py`.
# It moved because `report` has to be able to follow `infer`/`predict` too, and
# nothing in it was ever training-specific -- it reads `config.reporting` and
# hands a run directory to the artifact-driven `generate_report`. Leaving it here
# would have forced `pipelines/infer.py` to import `pipelines/train.py` for a
# capability belonging to neither.
#
# Re-exported under the former private names so every existing call site keeps
# working, including the source-text pins in
# `tests/unit/pipelines/test_train_reporting_hook.py`: `inspect.getsource`
# resolves through the object to its real definition, so those guards now read
# the moved body rather than silently passing over an empty wrapper.
from spectramr.infrastructure.reporting.run_hook import (  # noqa: E402,F401
    UNCONFIGURED_REPORT_TABLES as _UNCONFIGURED_REPORT_TABLES,
)
from spectramr.infrastructure.reporting.run_hook import (  # noqa: E402
    maybe_run_reporting as _maybe_run_reporting,
)
from spectramr.infrastructure.reporting.run_hook import (  # noqa: E402,F401
    run_unconfigured_report as _run_unconfigured_report,
)


def _maybe_run_calibration(strategy, pipeline, output_paths, logging_service) -> None:
    """Post-training certification hook for conformal / calibration strategies.

    A strategy that exposes ``run_calibration(dataloader)`` (the conformal,
    equivariance-conformal, and exchangeability arms) computes its coverage /
    certificate AFTER the reconstructor is trained — there is no per-step loss to
    optimise (``train_step`` returns ``[]``). Until this hook existed those arms
    ran the standard train loop and NEVER produced their declared primary metric
    (``coverage_at_alpha`` / exchangeability p-value) — the certification harness
    was inert (``run_calibration`` had zero call sites). This dispatches it on the
    validation loader and stamps the report into the run dir (pitfall #15: an
    advertised certificate must be produced + recorded). Self-guarded so a
    non-calibration run is untouched.

    **Fail-loud (2026-07-03, mrixfields_b17_dice_risk_calibration).** Every
    strategy that reaches this hook is *certify-only* — ``train_step`` returns
    ``[]`` and no gradient step ever runs, so the certificate is the arm's SOLE
    deliverable. The hook used to swallow a ``run_calibration`` failure (and a
    missing validation loader) into a warning while the run still reported
    ``success: True`` — a facade certificate (pitfall #16) and a
    "passed-with-warnings" (pitfall #10). It now RAISES: the exception escapes
    ``TrainingLoop.run`` and ``run_training_pipeline`` records ``success: False``
    (b17's real failure was its producer checkpoint never existing because the
    ``->7T`` synthesiser arm crashed on a missing data root). The original cause
    is preserved via ``raise ... from exc``.
    """
    run_calibration = getattr(strategy, "run_calibration", None)
    if not callable(run_calibration):
        return
    val_loader = None
    try:
        val_loader = pipeline.data_loaders.get("val")
    except Exception:
        val_loader = None
    if val_loader is None:
        # A certify-only arm cannot produce its certificate without a
        # calibration cohort — fail loud rather than report a facade success.
        raise RuntimeError(
            "calibration hook: no validation loader; a certify-only calibration "
            "arm cannot produce its certificate without a calibration cohort."
        )
    try:
        reports = run_calibration(val_loader)
    except Exception as exc:
        raise RuntimeError(
            "calibration hook: run_calibration failed — the certificate is this "
            f"certify-only arm's sole deliverable, so the run fails ({exc})."
        ) from exc
    if output_paths and output_paths.get("run_output_dir"):
        target = Path(output_paths["run_output_dir"]) / "calibration_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(reports, indent=2, default=str))
        if logging_service:
            logging_service.log_info(
                f"✓ calibration hook: {len(reports or {})} certificate(s) → {target}"
            )


# ---------------------------------------------------------------------------
# Sanity-check (overfit-single-batch) pass/fail thresholds
# ---------------------------------------------------------------------------
# A correctly-wired model+loss MUST be able to memorise a single batch. These
# gate the ``sanity_check`` command so a model that cannot learn — dead
# gradient, frozen weights, divergence, or a target it architecturally cannot
# fit (e.g. unlearned k-space phase, the experiment_11 "DC blob") — FAILS
# loudly instead of returning a green "success". Tuned for short diagnostic
# runs (a few hundred single-batch iterations).
SANITY_MIN_ITERS = 20  # need this many finite-loss steps before judging a trend
SANITY_WINDOW = 10  # trailing/leading steps averaged for "initial"/"final" loss
SANITY_MAX_LOSS_RATIO = 0.5  # final loss must be <= 50% of the initial loss ...
SANITY_ABS_LOSS_FLOOR = 1e-4  # ... unless it is already essentially zero
SANITY_PHASE_MAX_RATIO = 0.7  # if a phase metric is tracked, it must drop too


def _evaluate_sanity_overfit(
    loss_trace: list[float], phase_trace: list[float]
) -> tuple[bool, dict[str, Any]]:
    """Verdict for the overfit-single-batch sanity check.

    A correctly-wired model memorises one repeated batch, so its training
    loss must collapse. If the loss plateaus the model cannot learn the
    target — broken gradient flow / loss wiring, divergence, or an
    architecturally unlearnable target such as k-space phase (the
    experiment_11 "DC blob"). When a phase-coherence component is tracked we
    additionally require it to improve, since total loss can be dominated by
    magnitude terms and mask a stuck phase.

    Returns ``(passed, report)``. ``report`` is JSON-serialisable and is
    stamped into the run result (provenance, per CLAUDE.md #15).
    """
    report: dict[str, Any] = {"n_iters": len(loss_trace)}
    finite = [x for x in loss_trace if math.isfinite(x)]
    if len(finite) != len(loss_trace):
        report["reason"] = "non-finite loss (divergence) during single-batch overfit"
        return False, report
    if len(finite) < SANITY_MIN_ITERS:
        report["reason"] = (
            f"only {len(finite)} finite-loss iters (<{SANITY_MIN_ITERS}); "
            "inconclusive — run the sanity check for more iterations"
        )
        report["inconclusive"] = True
        return True, report

    w = min(SANITY_WINDOW, len(finite))
    initial = sum(finite[:w]) / w
    final = sum(finite[-w:]) / w
    report["initial_loss"] = initial
    report["final_loss"] = final
    report["loss_ratio"] = (final / initial) if initial > 0 else None

    if not (final <= initial * SANITY_MAX_LOSS_RATIO or final < SANITY_ABS_LOSS_FLOOR):
        report["reason"] = (
            f"loss failed to collapse on a single batch: {initial:.4g} → "
            f"{final:.4g} (ratio {final / initial:.2f} > {SANITY_MAX_LOSS_RATIO}). "
            "The model cannot overfit one batch — check gradient flow, loss "
            "wiring, and whether the target is learnable (e.g. k-space phase)."
        )
        return False, report

    if phase_trace:
        pf = [x for x in phase_trace if math.isfinite(x)]
        if len(pf) >= SANITY_MIN_ITERS:
            pw = min(SANITY_WINDOW, len(pf))
            p0 = sum(pf[:pw]) / pw
            p1 = sum(pf[-pw:]) / pw
            report["phase_initial"] = p0
            report["phase_final"] = p1
            report["phase_ratio"] = (p1 / p0) if p0 > 0 else None
            if p0 > 0 and not (p1 <= p0 * SANITY_PHASE_MAX_RATIO or p1 < SANITY_ABS_LOSS_FLOOR):
                report["reason"] = (
                    f"phase metric failed to improve on a single batch: "
                    f"{p0:.4g} → {p1:.4g} (ratio {p1 / p0:.2f} > "
                    f"{SANITY_PHASE_MAX_RATIO}). The model is not learning "
                    "k-space phase — this is the 'DC blob' failure mode "
                    "(see docs/experiment_11_kspace_cold_diffusion.rst)."
                )
                return False, report

    report["reason"] = "loss collapsed as expected on a single batch"
    return True, report


# ---------------------------------------------------------------------------
# Helpers for the per-arm final_metrics.json contract
# ---------------------------------------------------------------------------


def validation_csv_for(training_csv: str | Path) -> Path:
    """Where the validation CSV lives, given the training CSV.

    One derivation, because there are now two callers: `_run_validation` writes
    the file, and `_summarise_best_metrics_from_csv` reads it to build
    `final_metrics.best`. Two copies of a path rule is how a writer and a reader
    end up pointing at different files while both look correct.

    The name swap is preferred over `parent / "validation_metrics.csv"` so a run
    that suffixed its training CSV keeps the pair together; the parent fallback
    covers a training CSV that is not named `training_metrics*`.
    """
    training_csv = Path(training_csv)
    if "training_metrics" in training_csv.name:
        return training_csv.with_name(
            training_csv.name.replace("training_metrics", "validation_metrics")
        )
    return training_csv.parent / "validation_metrics.csv"


def _column_higher_is_better(key: str) -> bool | None:
    """Resolve a CSV column's optimization direction through the metric SSOT.

    Delegates to :func:`~spectramr.core.metrics.metric_directions.resolve_direction`
    — the non-fatal sibling built for exactly this caller shape, a sweep over
    *arbitrary* columns rather than a decision from a named metric.

    This used to re-implement the lookup by scanning ``METRIC_HIGHER_IS_BETTER``
    for the longest entry appearing as a raw character substring. Two failures
    followed from that, both measured: it matched ``mad`` inside ``made``, so
    ``train_made_up`` resolved to a direction instead of ``None``; and it read the
    table directly, so registry-declared directions, aliases and
    ``NON_REGISTRY_METRIC_DIRECTIONS`` were invisible to it. The SSOT matches whole
    underscore-delimited token runs, which is what makes ``val_robust_mri_psnr_2x``
    resolve while ``totally_made_up_metric`` does not.

    Returns ``None`` for a column the SSOT declines to resolve; the caller records
    no ``_best`` for it rather than guessing.
    """
    from spectramr.core.metrics.metric_directions import resolve_direction

    return resolve_direction(key)


def _should_step_schedulers(
    iteration: int, start_iteration: int, gradient_accumulation_steps: int
) -> bool:
    """Whether per-iteration LR schedulers should step this iteration.

    Schedulers must advance on the optimizer cadence — once per
    ``gradient_accumulation_steps`` micro-batches — not on every iteration;
    otherwise the schedule decays N times too fast under accumulation. The first two
    iterations are skipped to avoid PyTorch's "step before optimizer.step()"
    warning. The optimizer boundary mirrors the trainer:
    ``(iteration + 1) % gas == 0``.
    """
    if iteration <= start_iteration + 1:
        return False
    gas = max(1, int(gradient_accumulation_steps))
    return (iteration + 1) % gas == 0


def _select_current_run_rows(rows: list[dict]) -> list[dict]:
    """Keep only the rows the CURRENT run appended to a shared metrics CSV.

    ``logs/training_metrics.csv`` lives in the arm's output directory and is APPENDED
    to by every run that writes there, so a naive pass over the file summarises an
    arbitrary blend of runs (issue #586: an exp_11 CSV held five of them, and
    ``final_metrics.json`` reported a 07-26 run's bests as a 07-28 run's). The
    ``iteration`` column is monotonically increasing within a run and RESETS when a new
    one starts, so the current run is the final non-decreasing segment.

    Rows with no usable ``iteration`` cannot be windowed; they are returned unchanged so
    callers keep the old behaviour rather than silently losing data.
    """
    start = 0
    previous: int | None = None
    for index, row in enumerate(rows):
        raw = row.get("iteration")
        try:
            current = int(float(raw))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return rows
        if previous is not None and current <= previous:
            start = index
        previous = current
    return rows[start:]


def _summarise_best_metrics_from_csv(
    csv_path: str | None, final_iteration: int | None = None
) -> dict[str, float]:
    """Read the run's metrics CSVs and return per-column best values.

    Reads the training CSV **and** the validation CSV beside it
    (:func:`validation_csv_for`). Reading only the training file is issue #481:
    validation writes to its own CSV, `training_metrics.csv` declares `val_*`
    columns it never populates, and this function fed `final_metrics.best` from
    the training file alone -- so every headline artifact of a run said validation
    produced nothing, on runs where validation ran and produced numbers that were
    in fact alarming (val_psnr ~6 dB against train_psnr ~30 dB). The metric was
    computed; the surface a reader consults never showed it.

    "Best" direction comes from the metric-direction SSOT via
    :func:`_column_higher_is_better`. A column it declines to resolve gets no
    entry at all rather than a guessed one.

    Only the CURRENT run's rows are considered (issue #586), via two filters:

    1. :func:`_select_current_run_rows` keeps the final non-decreasing ``iteration``
       segment, dropping earlier runs appended to the same file.
    2. ``final_iteration`` (the last iteration THIS run actually reached) drops rows
       beyond it. This is what catches the case segment-detection cannot: a run shorter
       than ``logging.log_interval`` writes NO rows, so the final segment still belongs
       to the previous run. Every one of its iterations exceeds this run's, so the
       window empties and ``best`` is ``{}`` -- reporting nothing instead of another
       run's curve.
    """
    if not csv_path:
        return {}

    summary: dict[str, float] = {}
    # Training first, then validation. Both are windowed to this run independently
    # because they are written on different cadences (`log_interval` vs
    # `validation.schedule.interval_steps`) and a run can produce rows in one and
    # none in the other.
    undeclared: set[str] = set()
    for path in (Path(csv_path), validation_csv_for(csv_path)):
        _fold_best_from_csv(path, final_iteration, summary, undeclared)
    if undeclared:
        # Say what was dropped. Recording no `_best` for an undeclared column is
        # right -- inventing a direction is what made `best_metric_name: lpips`
        # maximize LPIPS -- but doing it in SILENCE just moves the defect: a
        # reader looking for `val_residual_norm_best` finds nothing and cannot
        # tell "no declared direction" from "never measured". Named once here,
        # the fix is a one-line entry in `metric_directions`.
        logger.info(
            "final_metrics: %d column(s) have no declared optimization direction "
            "and so carry no `_best` entry: %s. Declare them in "
            "core/metrics/metric_directions.py if a best value is meaningful "
            "(it is not, for grad_norm or lr).",
            len(undeclared),
            ", ".join(sorted(undeclared)),
        )
    return summary


def _fold_best_from_csv(
    path: Path,
    final_iteration: int | None,
    summary: dict[str, float],
    undeclared: set[str] | None = None,
) -> None:
    """Fold one CSV's per-column extrema into ``summary`` (mutated in place).

    Columns with no declared direction are collected into ``undeclared`` rather
    than being dropped without trace; the caller reports them once.
    """
    import csv as _csv

    if not path.exists():
        return

    with open(path, newline="") as fh:
        rows = _select_current_run_rows(list(_csv.DictReader(fh)))
        if final_iteration is not None:
            kept = []
            for row in rows:
                try:
                    if int(float(row.get("iteration"))) <= final_iteration:  # type: ignore[arg-type]
                        kept.append(row)
                except (TypeError, ValueError):
                    kept.append(row)
            rows = kept
        for row in rows:
            for key, raw in row.items():
                if key in ("iteration", "epoch") or raw in (None, "", "nan"):
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                # One resolver, and no fallback guess. The deleted fallback was a
                # 6-substring test (`psnr`/`ssim`/`accuracy`/`dice`/`f1`/`iou`) that
                # could only ever FLIP a column to maximize, and it fired precisely
                # where the SSOT had declined to resolve -- i.e. where there was no
                # declaration to justify the flip. `metric_directions` documents that
                # exact heuristic as the bug that made `best_metric_name: lpips`
                # maximize LPIPS; it should not survive one layer up.
                higher_better = _column_higher_is_better(key)
                if higher_better is None:
                    # Undeclared: record no "best" rather than inventing a direction.
                    # A "best grad_norm" or "best lr" is not a quantity. Collected
                    # so the caller can name what it dropped -- a silent omission
                    # is indistinguishable from "that metric was never measured".
                    if undeclared is not None:
                        undeclared.add(key)
                    continue
                stat_key = f"{key}_best"
                if stat_key not in summary:
                    summary[stat_key] = value
                else:
                    summary[stat_key] = (
                        max(summary[stat_key], value)
                        if higher_better
                        else min(summary[stat_key], value)
                    )


def _extract_es_best(early_stopping_service: object | None) -> float | None:
    if early_stopping_service is None:
        return None
    value = getattr(early_stopping_service, "best_value", None)
    return float(value) if value is not None else None


def _extract_es_best_iter(early_stopping_service: object | None) -> int | None:
    if early_stopping_service is None:
        return None
    value = getattr(early_stopping_service, "best_iteration", None) or getattr(
        early_stopping_service, "best_step", None
    )
    return int(value) if value is not None else None


def _validation_batch_sample_count(val_batch, val_input, val_target):
    """How many samples a validation batch contributed, or ``None``.

    The epoch metric is a **sample-weighted** mean (issue #1347), so every batch
    needs a weight. Read it off whichever reference tensor is available -- the
    target first, because that is the tensor every full-reference metric graded
    against -- and fall back through the batch container.

    Args:
        val_batch: The loader's batch, already adapted to a ``TrainingBatch``
            where possible.
        val_input: The unpacked input tensor, or ``None``.
        val_target: The unpacked target tensor, or ``None``.

    Returns:
        The leading-axis length, or ``None`` when no tensor with a batch axis
        could be found. ``None`` is a *reported* outcome, never silently coerced
        -- see the caller, which counts these and warns once.
    """
    candidates = [val_target, val_input]
    if isinstance(val_batch, TrainingBatch):
        candidates += [val_batch.target, val_batch.input]
    elif isinstance(val_batch, dict):
        candidates += [val_batch.get("target"), val_batch.get("input")]
    for candidate in candidates:
        if isinstance(candidate, torch.Tensor) and candidate.ndim >= 1:
            return int(candidate.shape[0])
    return None


def _all_reduce_val_metrics(val_accum, val_weight, device):
    """Sum per-rank validation accumulators across DDP ranks.

    Under DDP the validation loader is wrapped in a ``DistributedSampler``
    (:func:`spectramr.pipelines.parallel._apply_distributed_samplers`) which
    *shards and pads* the val set, so each rank's ``val_accum`` / ``val_weight``
    cover only its shard. Finalising ``v_sum / val_weight`` on the local shard
    therefore reported a single padded shard's metric, and rank-0
    early-stopping diverged from the true full-set value. Summing both the
    metric running-sums and the sample count across ranks before dividing
    yields the correct sample-weighted global mean.

    No-op (returns the inputs unchanged) when ``torch.distributed`` is not
    initialised — i.e. on the default single-process ``spectramr train`` path,
    which is every run that does not go through the ``train-distributed``
    torchrun entry point.

    Args:
        val_accum: ``{metric_name: running_sum}`` where each sum is a float or
            scalar/array tensor, already weighted by its batch's sample count.
        val_weight: Total weight behind ``val_accum`` on this rank -- the number
            of validation **samples**, not batches (issue #1347). This function
            has always documented "the correct sample-weighted global mean"; it
            is the caller that used to hand it a batch count.
        device: Device for the all-reduce tensors (the model/pipeline device).

    Returns:
        ``(reduced_accum, reduced_weight)`` with float-coerced sums.
    """
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return val_accum, val_weight

    def _to_scalar(v):
        if isinstance(v, torch.Tensor):
            v = v.detach()
            return float(v.item() if v.numel() == 1 else v.mean().item())
        return float(v)

    count_t = torch.tensor([float(val_weight)], device=device)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

    # Stable key order so every rank reduces the same vector positions.
    keys = sorted(val_accum.keys())
    if keys:
        sums = torch.tensor([_to_scalar(val_accum[k]) for k in keys], device=device)
        dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        val_accum = {k: sums[i].item() for i, k in enumerate(keys)}

    return val_accum, round(count_t.item())


def _run_validation(
    pipeline,
    strategy,
    iteration,
    epoch,
    logging_service,
    output_paths=None,
    metrics_service=None,
    tb_writer=None,  # TensorBoard writer for validation metrics
    csv_file=None,
):
    """Run validation loop."""
    val_accum = {}
    val_count = 0
    # Epoch metrics are a SAMPLE-weighted mean, not a batch-weighted one
    # (issue #1347). ``val_count`` stays a batch count because the
    # "every batch raised" guard below is about batches; ``val_samples`` is the
    # divisor. With ``drop_last=False`` a short final batch used to weigh exactly
    # as much as a full one -- 0.8 dB on a 24-image set split 4x5 + 1x4.
    val_samples = 0
    val_unweighted_batches = 0

    if logging_service:
        logging_service.log_info(
            f"[VAL] Starting validation at epoch {epoch}, iteration {iteration}..."
        )

    # ✅ SSOT: Extract logging config from strategy (passed from bootstrap)
    logging_config = None
    if hasattr(strategy, "config") and hasattr(strategy.config, "logging"):
        logging_config = strategy.config.logging

    # ✅ SSOT: Determine if we should capture validation images
    # (only if strategy doesn't already log them internally)
    should_log_images = False
    validation_image_interval = 1  # Default from schema
    max_images_per_batch = 4  # Default from schema
    log_input_images = False  # Default from schema
    log_difference_images = True  # Default from schema
    save_validation_images = True  # Default from schema

    if logging_config is not None:
        should_log_images = bool(
            logging_config.images.log_validation or logging_config.images.save_validation
        )
        # Extract interval with config override
        validation_image_interval = max(1, logging_config.intervals.validation_images)
        # Extract max images from schema
        max_images_per_batch = logging_config.images.max_per_batch
        # Extract boolean flags from schema
        log_input_images = logging_config.images.log_input
        log_difference_images = logging_config.images.log_difference
        save_validation_images = logging_config.images.save_validation

    capture_images = (
        should_log_images
        and (tb_writer is not None or metrics_service is not None)
        and (iteration % validation_image_interval == 0)
        and not getattr(strategy, "logs_validation_images_in_step", False)
    )

    visual_samples: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

    logger.info(f"\n[Pipeline] Validating at iter {iteration}...")
    avg_metrics = {}

    # Reset stateful metrics (e.g., FID) before validation
    if hasattr(strategy, "reset_validation_metrics"):
        strategy.reset_validation_metrics()

    # [FIX] Free training memory before validation to prevent OOM
    # Training typically uses most of VRAM; empty_cache releases the allocator's
    # cached blocks so validation forward passes have room. Wired to
    # config.validation.empty_cache_before_validation (default True preserves
    # OOM-safety; backlog_wasted_compute_audit_2026_05_29 PIPE-2) so
    # memory-headroom runs can skip the allocator re-grow on the next train step.
    # (settings are reached via strategy.config in this function.)
    _val_cfg = getattr(getattr(strategy, "config", None), "validation", None)
    _empty_cache_before_val = (
        getattr(_val_cfg, "empty_cache_before_validation", True) if _val_cfg is not None else True
    )
    if _empty_cache_before_val and torch.cuda.is_available():
        torch.cuda.empty_cache()

    ema_active = getattr(pipeline, "ema", None) is not None
    # Unwrap once: the EMA shadow carries BARE keys (it is a deepcopy of the
    # module taken before any parallel wrap), so intersecting it against a
    # wrapped generator's ``module.``/``_orig_mod.``-prefixed keys yields the
    # empty dict -- which ``load_state_dict(..., strict=False)`` accepts in
    # silence, leaving validation to grade the LIVE weights (#2172).
    ema_target = unwrap_model(pipeline.generator) if ema_active else pipeline.generator

    import contextlib

    @contextlib.contextmanager
    def _ema_swap_context():
        """Delegate to the swap SSOT, or do nothing when this arm has no EMA."""
        if not ema_active:
            yield
            return
        with ema_weights_swapped_in(ema_target, pipeline.ema.module):
            yield

    # Resolve a per-call validation cap. The two schema fields below were
    # silent orphans (declared in `validation.py` but never consumed) — a
    # CLAUDE.md #9 hazard that made `--override validation.num_validation_batches`
    # a no-op and forced every smoke test to walk the full val set.
    # See findings booklet 2026-05-05 VAL-1.
    # C8: the same axis identity the train loop resolves, from the same
    # resolver. Resolved once here, not per batch -- ``resolve_axes_for`` walks
    # the config. Train and val MUST agree: a batch whose axes differ between
    # the two would let a metric grade a frame axis it was never told about.
    from spectramr.data.datasets.axis_exposure import resolve_axes_for

    _batch_axes = resolve_axes_for(getattr(pipeline.config, "data", None))

    val_cfg = getattr(pipeline.config, "validation", None)
    _max_val_batches: int | None = None
    if val_cfg is not None:
        _max_val_batches = val_cfg.loader.num_batches if val_cfg else None
        if _max_val_batches is None:
            # ``num_samples`` (legacy) → convert to a batch budget.
            _ns = val_cfg.loader.num_samples if val_cfg else None
            if _ns is not None:
                _bs = (val_cfg.loader.batch_size if val_cfg else None) or 1
                _max_val_batches = max(1, math.ceil(_ns / max(1, _bs)))

    # F-VALBATCH-RATELIMIT / 2026-05-20 — counter visible to all
    # iterations of the val loop so the per-batch except block can
    # collapse the spam.
    _val_warn_count = 0
    _first_val_failure = ""  # full traceback of the first val-batch failure
    # Resolve the strategy's validation_step signature ONCE before the batch
    # loop: it is run-invariant, so introspecting it per batch (N_batches times
    # N_validations) is wasted reflection.
    # backlog_wasted_compute_audit_2026_05_29 PIPE-3.
    _vs_params = list(inspect.signature(strategy.validation_step).parameters.keys())
    # Memo for the fallback generator.forward signature check (PIPE-3): the
    # generator is bound inside the loop, so we cache the run-invariant
    # "has timesteps param" result by id() rather than re-introspecting per batch.
    _gen_timesteps_cache: dict[int, bool] = {}
    with _ema_swap_context(), torch.no_grad():
        for i, val_batch in enumerate(
            tqdm(pipeline.data_loaders.get("val"), desc="Validating", leave=False)
        ):
            if _max_val_batches is not None and i >= _max_val_batches:
                logger.info(
                    "[VAL] Stopping after %d batches (validation.loader.num_batches cap)",
                    i,
                )
                break
            try:
                # Prepare Batch
                if not isinstance(val_batch, TrainingBatch) and isinstance(val_batch, dict):
                    val_batch = BatchAdapter.from_dict(val_batch, axes=_batch_axes)

                if hasattr(val_batch, "to"):
                    val_batch = val_batch.to(pipeline.device)
                elif isinstance(val_batch, dict):
                    val_batch = {
                        k: v.to(pipeline.device) if isinstance(v, torch.Tensor) else v
                        for k, v in val_batch.items()
                    }

                # Validation Preprocessing (ComplexGuard + 5D->4D + Square Padding).
                # SSOT: module-level ``_preprocess_validation_tensor`` — the 5D→4D
                # flatten is gated on ``model.spatial_dims`` so volumetric (3D)
                # models keep their depth (2026-07 ldm slab-arm fix).
                def _preprocess_tensor(t):
                    return _preprocess_validation_tensor(t, pipeline.config)

                if isinstance(val_batch, TrainingBatch):
                    val_batch.input = _preprocess_tensor(val_batch.input)
                    val_batch.target = _preprocess_tensor(val_batch.target)
                elif isinstance(val_batch, dict):
                    if "input" in val_batch:
                        val_batch["input"] = _preprocess_tensor(val_batch["input"])
                    if "target" in val_batch:
                        val_batch["target"] = _preprocess_tensor(val_batch["target"])

                # Strategy Validation Step
                # [FIX] Unpack batch BEFORE calling validation_step.
                # 13/23 strategies expect (input_batch, target_batch) as two positional
                # args, but this call site was passing the whole batch as a single arg.
                # Use signature introspection to dispatch to the correct pattern.
                _val_input, _val_target = None, None
                if hasattr(strategy, "_unpack_batch"):
                    try:
                        _unpacked = strategy._unpack_batch(val_batch)
                        if isinstance(_unpacked, (tuple, list)) and len(_unpacked) >= 2:
                            _val_input, _val_target = _unpacked[0], _unpacked[1]
                    except Exception:
                        pass
                if _val_input is None or _val_target is None:
                    if isinstance(val_batch, TrainingBatch):
                        _val_input = val_batch.input
                        _val_target = val_batch.target
                    elif isinstance(val_batch, dict):
                        _val_input = val_batch.get("input")
                        _val_target = val_batch.get("target")

                # Dispatch based on strategy's validation_step signature
                # (_vs_params resolved once before the loop — PIPE-3).
                # Pattern A: (input_batch, target_batch, **kwargs) — 13 strategies
                # Pattern B: (batch, **kwargs) or (val_batch, batch_idx) — 5 strategies
                # Pattern C: (batch, input_batch=None, target_batch=None) — 3 strategies
                if (
                    _val_input is not None
                    and _val_target is not None
                    and len(_vs_params) >= 3  # self + 2 positional
                    and "target_batch" in _vs_params
                    and "batch" not in _vs_params[:2]  # exclude Pattern C
                ):
                    # Pattern A: two-tensor signature. Gated extra: pass a real
                    # reference field (e.g. b0_map) ONLY to strategies that
                    # declare it (the VF real-reference seam) — no effect on the
                    # other Pattern-A strategies.
                    _vs_extra: dict[str, Any] = {"batch_idx": i}
                    # Gated per-sample field seam (b0_map/b1_map/trajectory_*,
                    # field_strength[_target], contrast_id): forward each field ONLY
                    # to a strategy whose validation_step declares it AND when the
                    # batch carries it. Extracted to select_validation_extra_fields
                    # (testable; single source of truth). Previously this was inline
                    # and gated on isinstance(val_batch, dict) — ALWAYS False after
                    # the TrainingBatch conversion above, so the seam was DEAD; the
                    # VF real-reference + MRIxFields cross-field arms self-graded on
                    # the wrong field. The double gate also prevents the
                    # batch_context/kwargs collision in the field-conditioned arms.
                    _vs_extra.update(select_validation_extra_fields(val_batch, _vs_params))
                    metrics = strategy.validation_step(_val_input, _val_target, **_vs_extra)
                elif "batch" in _vs_params or "val_batch" in _vs_params:
                    # Pattern B/C: whole-batch signature
                    if "input_batch" in _vs_params:
                        # Pattern C: batch + optional keyword tensors
                        metrics = strategy.validation_step(
                            val_batch,
                            input_batch=_val_input,
                            target_batch=_val_target,
                        )
                    elif "batch_idx" in _vs_params:
                        metrics = strategy.validation_step(val_batch, batch_idx=i)
                    else:
                        metrics = strategy.validation_step(val_batch)
                else:
                    # Fallback: try two-tensor if we have both
                    if _val_input is not None and _val_target is not None:
                        metrics = strategy.validation_step(_val_input, _val_target)
                    else:
                        metrics = strategy.validation_step(val_batch, batch_idx=i)

                if capture_images and visual_samples is None:
                    input_batch, target_batch = _val_input, _val_target
                    if input_batch is not None and target_batch is not None:
                        # Skip _unpack_batch — already done above
                        pass
                    elif hasattr(strategy, "_unpack_batch"):
                        unpacked = strategy._unpack_batch(val_batch)
                        if isinstance(unpacked, (tuple, list)) and len(unpacked) >= 2:
                            input_batch, target_batch = unpacked[0], unpacked[1]
                    if input_batch is None or target_batch is None:
                        if isinstance(val_batch, TrainingBatch):
                            input_batch = val_batch.input
                            target_batch = val_batch.target
                        elif isinstance(val_batch, dict):
                            input_batch = val_batch.get("input")
                            target_batch = val_batch.get("target")

                    if input_batch is not None and target_batch is not None:
                        with torch.no_grad():
                            generator = (
                                strategy.env.generator
                                if hasattr(strategy, "env")
                                else pipeline.models.get("generator")
                            )
                            was_training = generator.training
                            generator.eval()
                            try:
                                # Preprocess input via strategy if available to handle
                                # channel/shape mismatches (e.g. cold diffusion multi-coil)
                                if hasattr(strategy, "_prepare_validation_data"):
                                    prep = strategy._prepare_validation_data(
                                        val_batch, input_batch, target_batch, None
                                    )
                                    if prep is not None:
                                        input_batch, target_batch, _ = prep

                                def _get_tensor_ref(obj):
                                    """Recursively find first tensor in dict/list/tuple."""
                                    if isinstance(obj, torch.Tensor):
                                        return obj
                                    if isinstance(obj, dict):
                                        for v in obj.values():
                                            ref = _get_tensor_ref(v)
                                            if ref is not None:
                                                return ref
                                    if isinstance(obj, (list, tuple)):
                                        for v in obj:
                                            ref = _get_tensor_ref(v)
                                            if ref is not None:
                                                return ref
                                    return None

                                # [FIX] Prefer strategy._validation_forward() over raw
                                # generator(input_batch).  The strategy's method handles
                                # batch context preparation (k-space masks, sensitivity
                                # maps, channel adapters, etc.) that the raw call skips.
                                if hasattr(strategy, "_validation_forward"):
                                    _batch_ctx = {
                                        "use_dc": False,
                                        "measured_kspace": None,
                                    }
                                    # Field-conditioned strategies (the mrixfields
                                    # field_* family: field_fno, field_bridge,
                                    # koopman_field, confluence, …) read their
                                    # conditioning (field_strength,
                                    # field_strength_target, contrast_id, sources,
                                    # input) from ``batch_context`` and RAISE when it
                                    # is absent. With the bare ctx that exception is
                                    # swallowed below (preds=None), so the validation
                                    # IMAGE is silently dropped even though metrics
                                    # computed fine via validation_step (the "metrics
                                    # yes, image no" mrixfields symptom). Thread the
                                    # batch's own conditioning + the prepared
                                    # input/target through so _validation_forward can
                                    # render. CLAUDE.md #9 / pitfall #16.
                                    _batch_ctx["input"] = input_batch
                                    _batch_ctx["target"] = target_batch
                                    # NB: by here ``val_batch`` is a TrainingBatch
                                    # (BatchAdapter.from_dict ran upstream), which
                                    # parks every non-core key in ``.metadata`` and
                                    # exposes it via ``.get()`` — dict does too — so
                                    # duck-type on ``.get`` rather than isinstance
                                    # (a getattr on the dataclass misses metadata).
                                    for _ck in (
                                        "field_strength",
                                        "field_strength_target",
                                        "contrast_id",
                                        "sources",
                                    ):
                                        _cv = None
                                        if hasattr(val_batch, "get"):
                                            _cv = val_batch.get(_ck)
                                        if _cv is None and not isinstance(val_batch, dict):
                                            _cv = getattr(val_batch, _ck, None)
                                        if _cv is not None:
                                            _batch_ctx[_ck] = _cv
                                    preds = strategy._validation_forward(input_batch, _batch_ctx)
                                    # Unpack common return formats
                                    if isinstance(preds, (tuple, list)):
                                        preds = preds[0]
                                    if isinstance(preds, dict):
                                        preds = preds.get(
                                            "reconstruction",
                                            preds.get(
                                                "output",
                                                next(iter(preds.values()), None),
                                            ),
                                        )
                                else:
                                    # Fallback: raw generator call (for strategies
                                    # that don't implement _validation_forward)
                                    # Provide a zero timestep so diffusion generators
                                    # don't emit a "No timesteps provided" warning.
                                    _gk = id(generator.forward)
                                    _gen_has_timesteps = _gen_timesteps_cache.get(_gk)
                                    if _gen_has_timesteps is None:
                                        _gen_has_timesteps = (
                                            "timesteps"
                                            in inspect.signature(generator.forward).parameters
                                        )
                                        _gen_timesteps_cache[_gk] = _gen_has_timesteps
                                    if _gen_has_timesteps:
                                        # Safe extraction of batch size and device
                                        _ref = _get_tensor_ref(input_batch)
                                        _batch_size = _ref.shape[0] if _ref is not None else 1
                                        _device = (
                                            _ref.device if _ref is not None else pipeline.device
                                        )

                                        _t_viz = torch.zeros(
                                            _batch_size,
                                            device=_device,
                                            dtype=torch.long,
                                        )
                                        preds = generator(input_batch, timesteps=_t_viz)
                                    else:
                                        preds = generator(input_batch)
                            except Exception as _vis_exc:
                                logger.warning(
                                    "[Visual] Generator/strategy visual sample capture failed: %s\n%s",
                                    _vis_exc,
                                    traceback.format_exc(),
                                )
                                # Remember WHY, so the end-of-pass escalation below can
                                # name the real cause instead of listing three guesses.
                                # The dominant cause is a conditioned generator reached
                                # through the unconditional fallback — see the
                                # persistent-miss error message.
                                _visual_capture_state(strategy)["last_error"] = (
                                    f"{type(_vis_exc).__name__}: {_vis_exc}"
                                )
                                preds = None

                            if preds is not None:
                                # Recursively unpack if preds is a tuple or dict
                                if isinstance(preds, (tuple, list)):
                                    preds = preds[0]

                                # If it's still a dict, extract reconstruction/output
                                if isinstance(preds, dict):
                                    preds = preds.get(
                                        "reconstruction",
                                        preds.get("output", next(iter(preds.values()), None)),
                                    )

                                # Safe device extraction for target_batch
                                _target_ref = _get_tensor_ref(target_batch)
                                _target_device = (
                                    _target_ref.device
                                    if _target_ref is not None
                                    else pipeline.device
                                )

                                if (
                                    preds is not None
                                    and hasattr(preds, "device")
                                    and preds.device != _target_device
                                ):
                                    preds = preds.to(_target_device)

                            if was_training:
                                generator.train()

                        if preds is not None:
                            visual_samples = (preds, target_batch, input_batch)

                        # [OVERRIDE] If the strategy cached a synthesized visual prediction
                        # from its validation_step (e.g. tissue_params model applies Bloch
                        # synthesis and stores the result in _last_visual_pred), use it instead
                        # of whatever the raw generator forward pass returned.  For tissue_params
                        # models the generator returns a tissue-parameter dict whose first value
                        # (rho, Sigmoid → [0,1]) would otherwise produce near-black images.
                        _strat_pred = getattr(strategy, "_last_visual_pred", None)
                        _strat_tgt = getattr(strategy, "_last_visual_target", None)
                        if _strat_pred is not None and _strat_tgt is not None:
                            visual_samples = (_strat_pred, _strat_tgt, input_batch)

                batch_n = _validation_batch_sample_count(val_batch, _val_input, _val_target)
                if batch_n is None or batch_n <= 0:
                    # Reported, never inferred: this batch is weighted as one
                    # sample and the count is surfaced after the loop. Silently
                    # mixing weight-1 and weight-N batches would re-create the
                    # defect this weighting exists to remove.
                    batch_n = 1
                    val_unweighted_batches += 1
                for k, v in metrics.items():
                    val_accum[k] = val_accum.get(k, 0.0) + v * batch_n
                val_count += 1
                val_samples += batch_n
            except Exception as e:
                # F-VALBATCH-RATELIMIT / 2026-05-20 — when validation
                # fails systematically (e.g. CSMNOOperator shape
                # mismatch, OOM, channel-count mismatch), every val
                # batch raises the same error and the full traceback
                # spammed the log dozens of times per epoch (smoke
                # 20260516 showed 50+ identical OOM tracebacks).
                # First failure logs the full traceback for
                # diagnosability; subsequent failures within the same
                # ``_run_validation`` call get the one-line form.
                exc_type = type(e).__name__
                if _val_warn_count == 0:
                    _first_val_failure = traceback.format_exc()
                    logger.warning(
                        "[Warning] Val batch %d failed (%s): %s\n%s",
                        i,
                        exc_type,
                        e,
                        _first_val_failure,
                    )
                else:
                    logger.warning(
                        "[Warning] Val batch %d failed (%s): %s "
                        "(traceback suppressed; first failure has it)",
                        i,
                        exc_type,
                        e,
                    )
                _val_warn_count += 1
                continue

    # F36 / 2026-05-22 — fail loud when EVERY validation batch raised.
    # Previously a systematic validation crash (shape/channel mismatch, OOM)
    # was swallowed per-batch as a warning (see ~line 2040), validation
    # returned empty metrics, no images were saved, and the run still exited
    # 0 = PASS. That "green but image-less" outcome is the CLAUDE.md #10
    # silent-failure class — smoke 20260521 had 36/41 PASS produce no
    # validation images, each one a validation_step that crashed on every
    # batch. If validation was attempted (_val_warn_count > 0) but not a
    # single batch succeeded (val_count == 0), the run is broken: raise so the
    # smoke gate records a FAIL with the first batch's traceback (already
    # logged above) instead of shipping an image-less green run. Transient
    # single-batch failures are unaffected — those leave val_count > 0.
    if val_count == 0 and _val_warn_count > 0:
        raise RuntimeError(
            f"Validation produced zero successful batches: all "
            f"{_val_warn_count} validation batch(es) raised an exception. A run "
            f"cannot be considered passing when validation never executed — fix "
            f"the validation forward path (model/strategy shape or channel "
            f"mismatch) rather than letting the run exit green with no metrics "
            f"and no validation images (CLAUDE.md #10).\n"
            f"--- first validation-batch failure (root cause) ---\n"
            f"{_first_val_failure.rstrip() or '(traceback unavailable)'}"
        )

    # Under DDP each rank validated only its (padded) DistributedSampler shard;
    # sum the running-sums and counts across ranks so the finalized mean below
    # reflects the FULL validation set, not one shard. No-op single-process.
    if val_unweighted_batches:
        logger.warning(
            "[VAL] %d of %d validation batch(es) carried no tensor with a batch "
            "axis, so their metrics were weighted as a single sample each. The "
            "epoch mean is sample-weighted for the rest; treat the value as "
            "mixed-weighting until the strategy's validation_step returns a "
            "batch this loop can size.",
            val_unweighted_batches,
            val_count,
        )

    val_accum, val_samples = _all_reduce_val_metrics(
        val_accum, val_samples, getattr(pipeline, "device", None)
    )

    if val_samples > 0:
        # Aggregate mean metrics — convert to Python scalars immediately to avoid
        # tensor string serialization in CSV (e.g. "tensor(0.24, device='cuda:0')")
        avg_metrics = {}
        for k, v_sum in val_accum.items():
            v_avg = v_sum / val_samples
            if isinstance(v_avg, torch.Tensor):
                v_scalar = v_avg.detach()
                avg_metrics[k] = float(
                    v_scalar.item() if v_scalar.numel() == 1 else v_scalar.mean().item()
                )
            elif isinstance(v_avg, (int, float)):
                avg_metrics[k] = float(v_avg)
            else:
                avg_metrics[k] = v_avg

        # [SSOT] Finalize summary metrics (FID, IS, etc.)
        # These metrics accumulate state during the loop and compute only once here
        summary_metrics = strategy.finalize_validation()
        if summary_metrics:
            avg_metrics.update(summary_metrics)

        # The other half of #181. `val_count == 0` above catches "every batch
        # raised". This catches the case that gets through it: batches SUCCEED,
        # so val_count > 0 and nothing looks wrong, but every metric they
        # produced is NaN. The run then writes an all-NaN row and exits 0.
        #
        # That state is not merely uninformative, it is unrecoverable: a NaN
        # monitor disarms `save_best` (gated on `math.isfinite`) and freezes
        # `wait_count` in `EarlyStoppingService.update`, so no checkpoint is
        # selected and early stopping can never trigger -- the #178 outcome.
        #
        # Reachable today: an arm whose data honours no normalization contract
        # gets NaN from the range resolver for every range-sensitive metric
        # (#180). A row with even ONE finite metric is left alone; individual
        # NaNs are a per-metric matter the outcome contract already handles.
        _numeric = {k: v for k, v in avg_metrics.items() if isinstance(v, (int, float))}
        if _numeric and not any(math.isfinite(float(v)) for v in _numeric.values()):
            raise RuntimeError(
                f"Validation produced {val_count} successful batch(es) but EVERY "
                f"metric is non-finite: {sorted(_numeric)}.\n"
                "A run cannot select a checkpoint from this: the monitor is NaN, "
                "so `save_best` is skipped and early stopping never fires, and "
                "the run would otherwise train to its full budget and exit 0 "
                "with no checkpoint_best.pt.\n"
                "Most likely the data honours neither the [0, 1] nor the [-1, 1] "
                "normalization contract (#180) -- check the metrics log for a "
                "NOT APPLICABLE line naming `data_range_unresolved`, and declare "
                "`metrics.data_range` or normalize the pipeline output."
            )

        # Log results to console AFTER finalization so FID is included
        if logging_service:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
            logging_service.log_info(f"[VAL] Results: {metrics_str}")

        # Display (Phase 8: Robust Metrics View)
        display = " | ".join(
            [
                f"{k}: {v:.4f}"
                for k, v in avg_metrics.items()
                if any(
                    x in k.lower()
                    for x in [
                        "loss",
                        "psnr",
                        "ssim",
                        "nmse",
                        "error",
                        "fid",
                        "lpips",
                        "kid",
                        "hfen",
                    ]
                )
            ][:8]
        )
        logger.info(f"[Validation Iter {iteration}] {display}")

        # TensorBoard validation logging
        if tb_writer:
            tb_writer.scalars(avg_metrics, iteration, "val")
            # One axis for the headline pair, so "did PSNR move because SSIM
            # moved" is readable without flipping between charts.
            tb_writer.grouped_scalars(
                "val/headline",
                {k: v for k, v in avg_metrics.items() if k in ("psnr", "ssim")},
                iteration,
            )
            # Remembered for the HParams dashboard written at close; the last
            # validation is the run's final score.
            for _k, _v in avg_metrics.items():
                tb_writer.record_hparam_metric(f"hparam/{_k}", _v)
            tb_writer.flush()  # Flush after validation

        # [FIX] Write validation metrics to a dedicated CSV alongside training_metrics.csv
        if csv_file is not None:
            val_csv_file = str(validation_csv_for(csv_file))
            val_row = {"iteration": iteration, "epoch": epoch, **avg_metrics}
            try:
                val_csv_path = Path(val_csv_file)
                val_csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not val_csv_path.exists() or val_csv_path.stat().st_size == 0

                if write_header:
                    # First write — create file with header from current keys
                    all_fieldnames = list(val_row.keys())
                    with open(val_csv_file, "w", newline="") as _vf:
                        _vwriter = csv.DictWriter(
                            _vf, fieldnames=all_fieldnames, extrasaction="ignore"
                        )
                        _vwriter.writeheader()
                        _vwriter.writerow(val_row)
                else:
                    # Append — read existing header, merge with new keys
                    with open(val_csv_file, newline="") as _rf:
                        reader = csv.reader(_rf)
                        existing_header = next(reader, [])
                        existing_rows = list(csv.DictReader(_rf, fieldnames=existing_header))

                    new_keys = [k for k in val_row if k not in existing_header]
                    if new_keys:
                        # Schema expanded — rewrite with unified header
                        all_fieldnames = existing_header + new_keys
                        with open(val_csv_file, newline="") as _rf:
                            # Re-read to get all rows with original header
                            _rf.readline()  # skip old header
                            old_reader = csv.DictReader(_rf, fieldnames=existing_header)
                            existing_rows = list(old_reader)
                        with open(val_csv_file, "w", newline="") as _vf:
                            _vwriter = csv.DictWriter(
                                _vf,
                                fieldnames=all_fieldnames,
                                extrasaction="ignore",
                                restval="",
                            )
                            _vwriter.writeheader()
                            _vwriter.writerows(existing_rows)
                            _vwriter.writerow(val_row)
                    else:
                        # Same schema — simple append
                        with open(val_csv_file, "a", newline="") as _vf:
                            _vwriter = csv.DictWriter(
                                _vf,
                                fieldnames=existing_header,
                                extrasaction="ignore",
                                restval="",
                            )
                            _vwriter.writerow(val_row)
                if logging_service:
                    logging_service.log_info(
                        f"[VAL CSV] Saved {len(avg_metrics)} metrics → {val_csv_file}"
                    )
            except Exception as _vce:
                # FATAL (#713). This is the ONLY file carrying validation numbers
                # -- `training_metrics.csv` declares `val_*` columns it never
                # populates (#481) -- and `final_metrics.best` is folded from it.
                # A swallowed failure here produces a run with zero persisted
                # validation metrics that still reports success, which is exactly
                # the "validation never ran" reading #481 was about. Failing here
                # costs one validation pass, not the budget.
                if logging_service:
                    logging_service.log_error(f"[VAL CSV] Failed to write: {_vce}")
                raise RuntimeError(
                    f"Could not write validation metrics to {val_csv_file}: {_vce}. "
                    "This is the only surface carrying validation numbers, so "
                    "continuing would train on with no record of them."
                ) from _vce

        # CLAUDE.md #9 — surface silent skips. If we wanted to capture
        # but visual_samples never landed (validation step ran but no
        # batch produced usable input/target/preds), the run looks
        # green even though metrics/fake_images stays empty. Log a
        # WARNING so the smoke-test post-mortem can spot the regression
        # instead of silently shipping image-less runs.
        if capture_images and not visual_samples and logging_service is not None:
            logging_service.log_warning(
                "[ImageLogging] visual_samples was never captured during this "
                "validation pass — validation images will NOT be saved. "
                "Likely causes: (a) val loop terminated before any batch "
                "produced both input and target tensors, (b) strategy."
                "_validation_forward returned None, or (c) val_batch had a "
                "shape that _unpack_batch didn't recognise. Inspect val_batch "
                "structure and the strategy's _validation_forward."
            )
            # A single miss can be a fluke; a run of them is a broken seam that the
            # per-pass WARNING alone will not surface — `ablate_cocycle` logged 49 of
            # these across an 8-hour run, saved zero images, and still reported
            # status OK because the crash detectors key on the training loss
            # (pitfall #10 at the artefact layer). Escalate once the miss is
            # persistent so the log carries a greppable ERROR.
            _state = _visual_capture_state(strategy)
            _misses = _state.get("misses", 0) + 1
            _state["misses"] = _misses
            if _misses == _VISUAL_CAPTURE_MISS_LIMIT:
                _cause = _state.get("last_error")
                logging_service.log_error(
                    f"[ImageLogging] {_misses} consecutive validation passes saved NO "
                    "image. This run is producing metrics without pictures — its "
                    "contact sheet will show a PREVIOUS run. Last capture error: "
                    f"{_cause or 'none recorded (preds were None, not raised)'}. "
                    "If that is a missing keyword-only argument, the strategy lacks "
                    "`_validation_forward` and fell through to the unconditional "
                    "`generator(input_batch)` fallback — implement it (see "
                    "CrossFieldTranslationStrategy / FieldCocycleTranslationStrategy)."
                )
        elif capture_images and visual_samples:
            _visual_capture_state(strategy)["misses"] = 0
            # We have pictures — now ask whether they are pictures of anything.
            # SSIM/PSNR grade agreement with the reference and cannot answer this:
            # b33_field_bridge posted the cohort's BEST correlation (+0.91) while
            # rendering grey speckle everywhere the reference is air. Nothing in the
            # run noticed, because the loss stayed finite and falling (pitfall #20).
            _sanity = None
            try:
                _sanity = measure_output_sanity(visual_samples[0], visual_samples[1])
            except Exception as _ose:  # never let a diagnostic abort validation
                logger.debug("[OutputSanity] check skipped: %s", _ose)
            if _sanity is not None and _sanity.is_degenerate and logging_service:
                logging_service.log_warning(
                    f"[OutputSanity] validation output looks degenerate "
                    f"({_sanity.verdict}): {_sanity.detail}"
                )
                _sanity_state = _visual_capture_state(strategy)
                _bad = _sanity_state.get("sanity_misses", 0) + 1
                _sanity_state["sanity_misses"] = _bad
                if _bad == _OUTPUT_SANITY_MISS_LIMIT:
                    logging_service.log_error(
                        f"[OutputSanity] {_bad} consecutive validation passes produced "
                        f"a degenerate output ({_sanity.verdict}). This run is still "
                        "reporting metrics, but its images are not usable "
                        "reconstructions — check for a reverse chain that never "
                        "reaches t=0, a collapsed posterior, or off-scale guidance "
                        "before trusting its SSIM/PSNR. Measured: "
                        f"{_sanity.detail}"
                    )
            elif _sanity is not None and not _sanity.is_degenerate:
                _visual_capture_state(strategy)["sanity_misses"] = 0

        # ✅ SSOT: Log captured validation images if collected
        if capture_images and visual_samples and logging_config is not None:

            def _ensure_4d(t: torch.Tensor) -> torch.Tensor:
                """Ensure tensor is 4D BCHW format for TensorBoard logging.

                Handles:
                - 2D: (H, W) → (1, 1, H, W)
                - 3D: (C, H, W) → (1, C, H, W)
                - 4D: (B, C, H, W) → unchanged
                - 5D: (B, C, H, W, D) → (B*D, C, H, W) [flatten batch & depth]
                """
                t = t.float()
                if t.ndim == 2:
                    return t.unsqueeze(0).unsqueeze(0)
                if t.ndim == 3:
                    return t.unsqueeze(0)
                if t.ndim == 4:
                    return t
                if t.ndim >= 5:
                    # Handle volumetric data: (B, C, H, W, D) → (B*D, C, H, W)
                    shape = list(t.shape)
                    b, c = shape[0], shape[1]
                    d = shape[-1]
                    h, w = shape[-3], shape[-2]
                    result = t.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)
                    logger.debug(
                        f"[_ensure_4d] Converted 5D ({b}, {c}, {h}, {w}, {d}) → 4D {result.shape}"
                    )
                    return result
                # For any other dimension, try to squeeze/unsqueeze intelligently
                logger.warning(f"[_ensure_4d] Unusual tensor dimension: {t.ndim}, shape: {t.shape}")
                while t.ndim < 4:
                    t = t.unsqueeze(0)
                while t.ndim > 4:
                    if t.shape[0] == 1:
                        t = t.squeeze(0)
                    else:
                        t = t.squeeze(-1)
                return t.float()

            def _to_magnitude(t: torch.Tensor) -> torch.Tensor:
                """Convert complex-valued tensor to magnitude image.

                For real-valued [B, C, H, W] with even C: treats channels
                as interleaved real/imaginary pairs and computes RSS
                complex magnitude: sqrt(sum(|z_i|^2)).
                """
                if torch.is_complex(t):
                    # RSS for multi-coil complex
                    if t.shape[1] > 1:
                        return torch.sqrt((t.real**2 + t.imag**2).sum(dim=1, keepdim=True) + 1e-8)
                    return torch.abs(t)

                # Handle 5D volumetric: [B, C, H, W, D] -> [B, 1, H, W, D]
                if t.ndim >= 5:
                    if t.shape[1] > 1:
                        return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
                    return t

                # Handle 4D: [B, C, H, W] -> [B, 1, H, W]
                if t.ndim == 4:
                    if t.shape[1] % 2 == 0 and not t.is_complex():
                        # Interleaved real/imaginary → complex magnitude
                        B, C, H, W = t.shape
                        t_reshaped = (
                            t.permute(0, 2, 3, 1).contiguous().float().view(B, H, W, C // 2, 2)
                        )
                        t_complex = torch.view_as_complex(t_reshaped).permute(0, 3, 1, 2)
                        return torch.sqrt(
                            torch.sum(t_complex.abs() ** 2, dim=1, keepdim=True) + 1e-8
                        )
                    elif t.shape[1] > 1:
                        # RSS combine for multi-channel
                        return torch.sqrt((t**2).sum(dim=1, keepdim=True) + 1e-8)
                    return t

                # F-TOMAG-NONE / 2026-05-20 — without this guard the
                # function fell off the end and returned ``None`` for
                # any tensor with ``ndim < 4`` (e.g., the 3-D ``[B, C,
                # L]`` outputs produced by 1-D models like
                # ``hrf_diffusion_prior`` or ``cs_mno_operator``).
                # ``_run_validation`` then tried ``pred_mag.shape``
                # against ``None`` and crashed the entire pipeline with
                # ``AttributeError: 'NoneType' object has no attribute
                # 'shape'`` — observable in
                # tests_experiments/smoke_test/smoke_test_all_20260519_091801.log
                # at 09:28:32 for cs_mno_operator. Real-valued
                # ≤3-D tensors are already magnitude-like; return them
                # as-is so downstream logging still works.
                return t

            def _kspace_to_image(ksp: torch.Tensor) -> torch.Tensor:
                """Convert k-space to image domain.

                Assumes k-space is in either complex dtype or BCHW split format.
                Handles both 4D and 5D volumetric data.
                Output is always 4D or less (never 5D).
                """
                try:
                    from spectramr.infrastructure.physics.fft_ops import ifft2c

                    # If 5D, flatten to 4D first to fulfill docstring contract
                    if ksp.ndim >= 5:
                        shape = list(ksp.shape)
                        b, c = shape[0], shape[1]
                        d = shape[-1]
                        h, w = shape[-3], shape[-2]
                        ksp = ksp.permute(0, 4, 1, 2, 3).reshape(b * d, c, h, w)

                    if torch.is_complex(ksp):
                        img = ifft2c(ksp)
                        return img

                    # Handle 4D volumetric k-space [B*D, C, H, W] after flattening (or native 4D)
                    if ksp.ndim == 4:
                        if ksp.shape[1] % 2 == 0:
                            b, c, h, w = ksp.shape
                            ksp_reshaped = (
                                ksp.permute(0, 2, 3, 1)
                                .contiguous()
                                .float()  # view_as_complex requires float32
                                .view(b, h, w, c // 2, 2)
                            )
                            ksp_complex = torch.view_as_complex(ksp_reshaped).permute(0, 3, 1, 2)
                            img = ifft2c(ksp_complex)
                            img = img.unsqueeze(1) if img.ndim == 3 else img
                            return img
                        if ksp.shape[1] == 1:
                            # Single real channel claimed to be k-space → treat as
                            # the real part with imaginary = 0 and IFFT. This is
                            # the right interpretation for magnitude-real-only
                            # k-space outputs (rare but valid). Falling through
                            # silently here previously rendered the raw CNN
                            # output as if it were image-domain — the
                            # ``unrolled_reconstruction`` "DC blob" PNG signature.
                            ksp_complex = torch.complex(ksp.float(), torch.zeros_like(ksp).float())
                            img = ifft2c(ksp_complex)
                            return img

                    raise ValueError(
                        "[ImageLogging] _kspace_to_image was asked to IFFT a "
                        f"tensor with shape={tuple(ksp.shape)} dtype={ksp.dtype} "
                        "that is neither complex, even-channel real-stacked "
                        "[R0,I0,...], nor single-channel real. Refusing to "
                        "silently fall through (CLAUDE.md #9): the rendered "
                        "PNG would otherwise be misinterpreted k-space."
                    )
                except ValueError:
                    # Loud-fail: the visualization layer must not lie about domain.
                    # Strategy/YAML must be fixed (not the visualization fallback).
                    raise
                except Exception as e:
                    logger.warning(
                        f"[ImageLogging] _kspace_to_image IFFT failed: {e}, "
                        f"input shape={ksp.shape}, dtype={ksp.dtype}, "
                        f"device={ksp.device}. Falling back to raw tensor "
                        f"(magnitude will be applied downstream)."
                    )
                    return ksp

            preds, targets, inputs = visual_samples
            # Distribution-head strategies (e.g. HeteroscedasticULF's [mean, logvar])
            # reduce the captured prediction to its point estimate so preds is not
            # RSS-blended into sqrt(mean^2 + logvar^2) by _to_magnitude below
            # (issue #371). MetricsMixin._prediction_for_visualization defaults to
            # identity, so non-distribution strategies are unaffected.
            _reduce_vis = getattr(strategy, "_prediction_for_visualization", None)
            if callable(_reduce_vis):
                preds = _reduce_vis(preds)
            logger.info(
                f"[ImageLogging] Visual samples shapes: preds={preds.shape}, targets={targets.shape}, inputs={inputs.shape}"
            )

            # If _last_visual_pred was used, visuals are already image-domain
            # magnitude — skip all IFFT logic to avoid corrupting them.
            _visuals_from_strategy = (
                getattr(strategy, "_last_visual_pred", None) is not None
                and getattr(strategy, "_last_visual_target", None) is not None
            )

            # Detect whether data needs IFFT before visualization.
            # Uses the authoritative domain inference utility (SSOT).
            # See docs/DOMAIN_HANDLING_RULES.md, Rules 1-4.
            from spectramr.infrastructure.training.utils.domain_inference import (
                needs_ifft_for_visualization,
            )

            _needs_ifft_preds = False
            _needs_ifft_targets = False
            _is_image_domain = True
            if hasattr(strategy, "config"):
                cfg = strategy.config
                _needs_ifft_preds, _needs_ifft_targets = needs_ifft_for_visualization(cfg)
                _is_image_domain = not _needs_ifft_preds

            logger.info(
                f"[ImageLogging] Domain detection (domain_inference SSOT): "
                f"is_image_domain={_is_image_domain}, "
                f"needs_ifft_preds={_needs_ifft_preds}, needs_ifft_targets={_needs_ifft_targets}"
            )

            if _visuals_from_strategy:
                # Strategy already provided image-domain magnitude tensors
                pred_img = preds
                target_img = targets
                logger.info(
                    "[ImageLogging] Using strategy-cached visuals (already image-domain magnitude)"
                )
            elif _needs_ifft_preds and _needs_ifft_targets:
                pred_img = _kspace_to_image(preds)
                target_img = _kspace_to_image(targets)
            elif _needs_ifft_preds:
                pred_img = _kspace_to_image(preds)
                target_img = targets
            elif _needs_ifft_targets:
                pred_img = preds
                target_img = _kspace_to_image(targets)
            else:
                pred_img = preds
                target_img = targets
            logger.info(f"[ImageLogging] After domain conversion: pred_img.shape={pred_img.shape}")
            logger.info(
                f"[ImageLogging] After domain conversion: target_img.shape={target_img.shape}"
            )

            pred_mag = _to_magnitude(pred_img)
            logger.info(f"[ImageLogging] After _to_magnitude: pred_mag.shape={pred_mag.shape}")

            pred_mag = _ensure_4d(pred_mag)
            logger.info(
                f"[ImageLogging] After _ensure_4d: pred_mag.shape={pred_mag.shape}, dim={pred_mag.ndim}"
            )

            target_mag = _to_magnitude(target_img)
            target_mag = _ensure_4d(target_mag)
            logger.info(
                f"[ImageLogging] After _ensure_4d: target_mag.shape={target_mag.shape}, dim={target_mag.ndim}"
            )

            images_dict = {
                "val/predictions": pred_mag[:max_images_per_batch],
                "val/targets": target_mag[:max_images_per_batch],
            }

            logger.info(
                f"[ImageLogging] images_dict shapes: val/predictions={images_dict['val/predictions'].shape}, val/targets={images_dict['val/targets'].shape}"
            )

            # Per-sample percentile windowing to [0, 1] for TensorBoard/disk.
            # SSOT: module-level ``_percentile_window`` (unit-tested).
            _normalize_for_tb = _percentile_window

            # Apply normalization for TensorBoard
            images_dict_normalized = {}
            for tag, tensor in images_dict.items():
                images_dict_normalized[tag] = _normalize_for_tb(tensor)
            images_dict = images_dict_normalized

            # ✅ SSOT: Use config flags for conditional image logging
            if log_difference_images:
                # [FIX] Handle resolution mismatch between predictions and targets (e.g. n_downsample mismatch)
                if pred_mag.shape[2:] != target_mag.shape[2:]:
                    import torch.nn.functional as F

                    pred_mag = F.interpolate(
                        pred_mag,
                        size=target_mag.shape[2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                # Slab-to-volume models (e.g. ``slat_vae_slab_to_volume``)
                # emit N×slabs predictions per N target volumes, so batch
                # dims diverge by a small integer factor. Clamp both to
                # the smaller dim before differencing — the diff visual is
                # only used for the first ``max_images_per_batch`` samples
                # anyway.
                if pred_mag.shape[0] != target_mag.shape[0]:
                    n_diff = min(pred_mag.shape[0], target_mag.shape[0], max_images_per_batch)
                    pred_mag = pred_mag[:n_diff]
                    target_mag = target_mag[:n_diff]

                diff = _ensure_4d(torch.abs(pred_mag - target_mag))
                images_dict["val/difference"] = diff[:max_images_per_batch]

            if log_input_images:
                # Multi-contrast inputs (e.g. cross-contrast cold diffusion
                # with [T1||target] concatenated along the coil axis): the
                # raw input tensor mixes a fully-sampled prior contrast with
                # an undersampled target contrast. RSS-combining all
                # channels here drowns the target's aliasing in the prior's
                # clean energy → the rendered "input" looks unaccelerated.
                # Match the convention from
                # ``metrics_mixin._slice_to_target_contrast_single``: when
                # ``data.domain.target_channels`` is set and the input channel
                # count is a multiple of it, slice to the LAST
                # ``target_channels`` (the target contrast) before IFFT.
                inputs_for_vis = inputs
                _target_ch = None
                if hasattr(strategy, "config"):
                    _data_cfg = getattr(strategy.config, "data", None)
                    if _data_cfg is not None:
                        _target_ch = _data_cfg.domain.target_channels
                if (
                    _target_ch is not None
                    and inputs_for_vis.dim() >= 4
                    and inputs_for_vis.shape[1] > _target_ch
                    and inputs_for_vis.shape[1] % _target_ch == 0
                ):
                    inputs_for_vis = inputs_for_vis[:, -_target_ch:]
                if _is_image_domain and not _needs_ifft_targets:
                    input_img = inputs_for_vis
                else:
                    input_img = _kspace_to_image(inputs_for_vis)
                input_mag = _ensure_4d(_to_magnitude(input_img))
                images_dict["val/inputs"] = input_mag[:max_images_per_batch]

            # ✅ SSOT: Log to TensorBoard if available
            if tb_writer and logging_config.images.log_validation:
                tb_writer.images(images_dict, iteration)
                tb_writer.flush()

            # ✅ SSOT: Save to disk if config.logging.save_validation_images enabled.
            # DDP: only rank 0 writes images (the val set is sharded across ranks
            # and all write to the same dir, so otherwise they race / overwrite).
            if (
                metrics_service is not None
                and save_validation_images
                and RankUtility.is_main_rank()
            ):
                try:
                    # [FIX] Denormalize predictions and targets to consistent [0, 1] range before saving
                    # Predictions from generator are in [-1, 1], targets from data are in raw range
                    # Both need to be normalized to [0, 1] for proper visualization

                    # Normalize using the robust TensorBoard normalization function
                    pred_mag_normalized = _normalize_for_tb(pred_mag).clamp(0.0, 1.0)
                    target_mag_normalized = _normalize_for_tb(target_mag).clamp(0.0, 1.0)

                    logger.info(
                        f"[ImageLoggingNormalization] pred_mag range before norm: "
                        f"[{pred_mag.min():.4f}, {pred_mag.max():.4f}] -> after: "
                        f"[{pred_mag_normalized.min():.4f}, {pred_mag_normalized.max():.4f}]"
                    )
                    logger.info(
                        f"[ImageLoggingNormalization] target_mag range before norm: "
                        f"[{target_mag.min():.4f}, {target_mag.max():.4f}] -> after: "
                        f"[{target_mag_normalized.min():.4f}, {target_mag_normalized.max():.4f}]"
                    )

                    metrics_service.save_images_batch(
                        real_images=target_mag_normalized,
                        fake_images=pred_mag_normalized,
                        prefix="validation",
                        epoch=epoch,
                        step=iteration,
                    )
                except Exception as e:
                    if logging_service:
                        logging_service.log_warning(f"Validation image save failed: {e}")

        # NOTE: CSV logging is handled by the primary write path above (lines ~1566-1594).
        # A previous duplicate write here with sorted(data.keys()) caused column
        # misalignment — removed to maintain single CSV write per validation trigger.

    return avg_metrics
