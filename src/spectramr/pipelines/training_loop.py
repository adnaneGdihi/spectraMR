"""The iteration-based training loop as a constructable unit (``TrainingLoop``).

PR-1 of the loop-engine extraction (WS-3). ``TrainingLoop`` is the seam that the
config-driven ``run_training_pipeline``, the scripting ``fit()`` path, and the
sanity-check path all route through — *one* loop, one entry point. For now
``run()`` delegates to the existing ``_execute_training_loop`` body in
:mod:`spectramr.pipelines.train` (a same-layer import); a follow-up (PR-2) inlines
that ~1190-line body into ``run()`` — the helpers stay in ``train.py`` (same
layer), so no re-export shim is needed and existing tests that import them are
untouched — and introduces a ``LoopState`` the strategies can read (which finally
retires the dead ``getattr(self.step_executor, "global_step", …)`` curriculum read).

Home: this lives in ``pipelines/`` (not ``infrastructure/training/``) on purpose.
The loop is *orchestration* — it drives the resolved strategy + ``TrainingEnvironment``
+ injected services through iterations, which is a pipelines concern — and the
pipelines home keeps the loop in the same layer as ``train.py``'s loop helpers
(``infrastructure`` may not import ``pipelines``, so a delegating/​helper-sharing
loop cannot live there). It replaces the dead ``infrastructure/training/engine.py``
``TrainerEngine`` (epoch-based legacy, zero references), removed in this change.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from collections.abc import Iterable
from itertools import cycle
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from spectramr.config.overrides import applied_override_paths
from spectramr.core.cascading_validation import aggregate_cascade_rows
from spectramr.core.metrics.flag_map import schema_flag_to_metric
from spectramr.core.metrics.scalar_transfer import fuse_to_host
from spectramr.core.module_utils import unwrap_model
from spectramr.core.topology import resolve_run_topology
from spectramr.data.batch_types import BatchAdapter, TrainingBatch
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.builders.directors import CheckpointDirector
from spectramr.core.wall_clock import (
    VALIDATION_SAFETY_FACTOR,
    resolve_check_interval,
    resolve_wall_clock_budget,
)
from spectramr.infrastructure.distributed.distributed_training import RankUtility
from spectramr.infrastructure.optimization.ema import EMAKeyMismatchError
from spectramr.infrastructure.training.strategies.lifecycle import (
    StrategyLifecycleDriver,
)

# The loop body (moved here in WS-3 PR-2) calls these helpers, which STAY in
# ``pipelines/train.py`` (same layer — pipelines → pipelines is legal, and
# leaving them there means existing tests that import them from ``train`` are
# untouched). ``train`` imports ``TrainingLoop`` lazily, so this top-level
# import does not cycle.
from spectramr.pipelines.train import (
    _evaluate_sanity_overfit,
    _extract_es_best,
    _extract_es_best_iter,
    _is_epoch_boundary,
    _maybe_run_calibration,
    _resync_scheduler_base_lrs,
    _run_validation,
    _should_step_schedulers,
    _summarise_best_metrics_from_csv,
    early_stop_monitor_candidates,
    ema_should_update,
    epoch_validation_can_fire,
    validation_can_fire,
)

logger = logging.getLogger(__name__)


# Keys the STEP EXECUTOR mints, not the strategy: ``execute_step`` records every
# step config's loss as ``f"{name}_loss"`` (``step_executor.py:270``) using the
# config's own ``name`` (``:208``), and 17 sites name theirs ``generator`` or
# ``discriminator`` (the single-step default is ``base.py:1777``). That dict is
# what SEEDS ``result`` at :1605, before ``get_last_metrics()`` merges the
# strategy's own keys on top.
#
# Each of these is an EXACT alias of a total the strategy already stamps --
# ``generator_loss`` is the same tensor as ``g_total_loss`` (the g-closure
# returns ``g_total``), ``discriminator_loss`` the same as ``d_total_loss``.
# Verified numerically on ``experiment_11_attention_none``: equal at all nine
# logged points.
#
# They are deliberately given NO column. A column would entrench a duplicate
# rather than fix it, and nothing is lost -- the identical value lands in the
# strategy-stamped name. Electing the single owner between the executor's
# ``{name}_loss`` and the strategy's ``{g,d}_total_loss`` is non-negotiable 17
# work with adversarial-wide blast radius, tracked in #1687; it is not this
# module's to make unilaterally. Until then the discard check below reports
# them as KNOWN aliases rather than as an unexplained resolver disagreement.
#
# When #1687 lands, DELETE this set and the classification branch that reads
# it -- they exist only to make the deferral legible, and an exclusion set that
# outlives its reason is how an allowlist is born.
_EXECUTOR_ALIAS_KEYS = frozenset({"generator_loss", "discriminator_loss"})


def read_scheduler_lr(sched: Any) -> tuple[float | None, str | None]:
    """One scheduler's current LR as ``(lr, reason_it_is_missing)`` (#1682).

    Exactly one of the two is ``None``. Split out as a pure core so both
    absence paths are testable and plantable (non-negotiable 15); the code
    this replaces expressed both as ``pass``.

    It is a defect *because of* the fix above, not independently of it.
    ``resolve_scheduler_lr_columns`` now PROMISES an ``lr_<name>`` column for
    every entry in ``pipeline.schedulers``, so a scheduler that never yields a
    value leaves a promised column permanently empty -- and an always-empty
    column cannot be told apart from a column the run never reached. That is
    pitfall #15's own direction reappearing inside the change that fixes
    pitfall #15. Absence is reported here, never inferred by the reader.

    The broad ``except`` is not a silent fallback (non-negotiable 3): it does
    not substitute a value, it returns the reason so the caller can say which
    scheduler went missing and why. ``get_last_lr`` returns host floats, so
    nothing here syncs the device (non-negotiable 9).
    """
    if not hasattr(sched, "get_last_lr"):
        return None, f"{type(sched).__name__} has no get_last_lr()"
    try:
        values = sched.get_last_lr()
    except Exception as exc:  # broad by design - the reason is returned, not swallowed
        return None, f"get_last_lr() raised {type(exc).__name__}: {exc}"
    try:
        return float(values[0]), None
    except (IndexError, TypeError, ValueError, KeyError) as exc:
        return None, f"get_last_lr() returned {values!r} ({type(exc).__name__})"


def resolve_scheduler_lr_columns(pipeline: Any) -> frozenset[str]:
    """CSV columns for the per-scheduler learning rates this loop emits (#1682).

    The emission site writes ``losses_scalar[lr_column_name(sched_name)]`` for every
    entry of ``pipeline.schedulers``. Nothing in the config names those columns:
    the scheduler keys (``opt_g``, ``opt_d``, ``main``) are chosen by the
    optimization builder, so a config-derived header can never produce them and
    every emitted LR was discarded by ``extrasaction="ignore"``.

    Reading the same dict the emission site reads is what makes this exact
    rather than a guess. It is already final here: ``schedulers`` is assigned
    once at environment construction (``builders/environment.py:127``) and no
    site mutates it afterwards.

    The ``lr_`` prefix is NOT duplicated here: ``lr_column_name`` is the single
    owner of the spelling (non-negotiable 17), and both this resolver and the
    emission site call it. The earlier version of this docstring called the
    duplication necessary; it is not, and a second literal is exactly the
    divergence that discarded every LR column in the first place.
    """
    schedulers = getattr(pipeline, "schedulers", None) or {}
    return frozenset(lr_column_name(name) for name in schedulers)


def resolve_producer_declared_columns(strategy: Any) -> frozenset[str]:
    """CSV columns the STRATEGY declares because the config cannot name them.

    Most stamped keys are config-derivable -- a loss named in ``config.losses``
    is stamped under that same name. The exceptions are producers that rename
    between knob and metric: ``losses.reconstruction.lambda_pre_dc_kspace`` is
    stamped as ``pre_dc_kspace_l1``. No amount of config traversal bridges that
    rename, so the producer declares the key itself
    (``BaseTrainingStrategy.declared_metric_keys``).

    ``hasattr`` rather than a hard call because strategies are duck-typed here
    and the loop already treats ``get_last_metrics`` the same way; base returns
    an empty set, so an opted-out strategy is unaffected either route.
    """
    if not hasattr(strategy, "declared_metric_keys"):
        return frozenset()
    return frozenset(strategy.declared_metric_keys())


def classify_discarded_keys(discarded: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split discarded CSV keys into KNOWN executor aliases and unexplained ones.

    This does not suppress anything -- both halves are reported. The split
    exists so a reader can tell a decision from a defect: a known alias costs
    nothing (its value reaches the CSV under the strategy-stamped name), while
    an unexplained key is unrecoverable data loss and a real disagreement
    between the header resolver and the loop.

    Returns:
        ``(known_aliases, unexplained)``, each sorted.
    """
    keys = set(discarded)
    return sorted(keys & _EXECUTOR_ALIAS_KEYS), sorted(keys - _EXECUTOR_ALIAS_KEYS)


def _set_optimizer_eval_mode(pipeline: Any, *, train: bool) -> None:
    """Toggle schedule-free optimizers between their gradient and averaged iterates.

    Schedule-free methods (Defazio et al. 2024) maintain an averaged sequence
    distinct from the point the gradient is evaluated at. ``optimizer.eval()``
    swaps the averaged weights in; ``optimizer.train()`` swaps them back.
    Validating without that call measures the wrong point in weight space, and
    the symptom is a metric that looks merely worse rather than a failure --
    which is why this is wired here instead of documented as the user's job.

    Duck-typed on purpose. It must work when the ``[schedulefree]`` extra is
    absent (no class to isinstance against) and for a user-supplied
    implementation. Every ordinary torch optimizer lacks these methods, so this
    is a no-op for them.

    The predicate is ``supports_schedule_free_modes``, not a copy of it. This
    function used to inline its own version that tested only the method for the
    direction being toggled, so an object exposing ``train`` but not ``eval``
    was switched into train mode and never switched back -- two spellings of
    one predicate, already disagreeing (pitfall #13b).
    """
    from spectramr.infrastructure.training.optimizers import (
        supports_schedule_free_modes,
    )

    for optimizer in (getattr(pipeline, "optimizers", None) or {}).values():
        if optimizer is None or not supports_schedule_free_modes(optimizer):
            continue
        getattr(optimizer, "train" if train else "eval")()


def _install_parallel_step_policy(strategy: Any, parallel_runtime: Any) -> None:
    """Hand the parallel strategy's step policy to the strategy's StepExecutor.

    The plugin resolves the policy, the director stores it on
    ``ParallelRuntime`` -- and until this call existed, nothing read it. The
    executor kept the generic ``AMPPolicy``, so ``DeepSpeedStepPolicy``
    (``owns_gradient_accumulation``/``owns_zero_grad``) never took effect: the
    loop ran ``loss.backward()`` + ``optimizer.step()`` instead of
    ``engine.backward()``/``engine.step()``. That is not a crash, it is 1/N^2
    loss scaling with a green test suite.

    The registry deliberately allows a CLASS or an INSTANCE. FSDP hands back
    ``FSDPStepPolicy`` (the class) because only this layer knows the clip
    settings; DeepSpeed hands back a configured instance because only the engine
    knows itself. Resolving that is this function's job -- ``adopt_step_policy``
    rejects a class outright rather than negotiating capability flags against a
    descriptor.
    """
    policy = getattr(parallel_runtime, "step_policy", None)
    if policy is None:
        return
    executor = getattr(strategy, "step_executor", None)
    adopt = getattr(executor, "adopt_step_policy", None)
    if not callable(adopt):
        raise RuntimeError(
            "parallel strategy %r supplied a step policy but the resolved "
            "strategy's step_executor cannot adopt it. Refusing to continue: "
            "silently dropping it yields wrong gradients, not an error."
            % (getattr(parallel_runtime, "strategy", "?"),)
        )
    if isinstance(policy, type):
        # Construct with the clip settings the current policy already resolved,
        # so FSDPStepPolicy inherits the arm's gradient_clip_value rather than
        # AMPPolicy's 1.0 default.
        current = getattr(executor, "amp_policy", None)
        policy = policy(
            max_grad_norm=getattr(current, "max_grad_norm", 1.0),
            enable_gradient_clipping=getattr(current, "enable_gradient_clipping", True),
        )
    adopt(policy)
    logger.info(
        "[Pipeline] installed %s from parallel strategy %r "
        "(owns_accumulation=%s, owns_zero_grad=%s)",
        type(policy).__name__,
        getattr(parallel_runtime, "strategy", "?"),
        getattr(policy, "owns_gradient_accumulation", False) is True,
        getattr(policy, "owns_zero_grad", False) is True,
    )


def _save_loss_schedule_checkpoint(
    *,
    config: Any,
    pipeline: Any,
    strategy: Any,
    output_paths: Any,
    epoch: int,
    iteration: int,
    metrics: dict[str, Any],
    logging_service: Any,
    parallel_runtime: Any = None,
) -> None:
    """Snapshot the model when a loss-schedule trigger changes a loss weight.

    Saved via the same :class:`CheckpointDirector` as the periodic/best saves, so
    the pre-change state is recoverable. Whether and how often to save (the
    user-configured throttle) is decided by the controller before this is called;
    this only performs the I/O — it holds no interval logic. Fail-soft: a save
    error is logged, not raised, so a transient FS issue never aborts training.

    ``parallel_runtime`` is not optional in spirit: without it the director
    resolves the DEFAULT checkpoint adapter, so an FSDP save writes rank 0's
    local SHARD and a DeepSpeed save writes no native artifact at all. It keeps a
    ``None`` default only because the single-process path has no runtime to pass.
    """
    try:
        checkpoint_dir = str(
            Path(output_paths.get("run_output_dir", "experiments/outputs")) / "checkpoints"
        )
        (
            CheckpointDirector(config)
            .with_checkpoint_dir(checkpoint_dir)
            .with_pipeline(pipeline)
            .with_strategy(strategy)
            .with_epoch(epoch)
            .with_global_step(iteration)
            .with_metrics(metrics)
            .with_scaler(getattr(pipeline, "scaler", None))
            .with_parallel_runtime(parallel_runtime)
            .with_counter_state({"current_step": iteration, "current_epoch": epoch})
            .validate()
            .save()
        )
        if logging_service:
            logging_service.log_info(
                f"[loss_schedule] checkpoint saved before weight change at iter {iteration}"
            )
    except Exception as ckpt_err:  # fail-soft: never abort training on a save error
        logger.warning(
            f"[loss_schedule] checkpoint-on-change save failed at iter {iteration}: {ckpt_err}"
        )


class TrainingLoop:
    """The training loop, holding its collaborators for a single ``run()`` call.

    Construct with the resolved strategy, the ``TrainingEnvironment`` (``pipeline``),
    the frozen ``TrainingSettings`` (``config``), the model-type label, and the
    injected services; then ``run(start_iteration=…)`` drives the iteration loop
    and returns the pipeline result dict (``success`` / ``best_metrics`` / …).
    """

    def __init__(
        self,
        strategy: Any,
        pipeline: Any,
        config: Any,
        model_type: str,
        *,
        tb_writer: Any = None,
        logging_service: Any = None,
        output_paths: Any = None,
        checkpoint_service: Any = None,
        metrics_service: Any = None,
        is_sanity_check: bool = False,
    ) -> None:
        self.strategy = strategy
        self.pipeline = pipeline
        self.config = config
        self.model_type = model_type
        self.tb_writer = tb_writer
        self.logging_service = logging_service
        self.output_paths = output_paths
        self.checkpoint_service = checkpoint_service
        self.metrics_service = metrics_service
        self.is_sanity_check = is_sanity_check

    def run(self, *, start_iteration: int = 0) -> dict[str, Any]:
        """Run the iteration loop; returns the pipeline result dict.

        PR-1 delegates to the existing ``_execute_training_loop`` body (imported
        lazily to break the construct-loop / run-loop cycle with ``train.py``);
        PR-2 inlines that body here verbatim.
        """
        return _execute_training_loop(
            self.strategy,
            self.pipeline,
            self.config,
            self.model_type,
            tb_writer=self.tb_writer,
            logging_service=self.logging_service,
            output_paths=self.output_paths,
            checkpoint_service=self.checkpoint_service,
            metrics_service=self.metrics_service,
            is_sanity_check=self.is_sanity_check,
            start_iteration=start_iteration,
        )

    def evaluate(self) -> dict[str, float]:
        """Run one validation pass and return the aggregated metrics.

        The ``run()`` peer for the scripting ``Trainer.evaluate``: it drives the
        SAME ``_run_validation`` the training loop invokes at each
        ``eval_interval`` (eval mode + EMA-swap, **no optimizer steps**), so the
        reported metrics match in-training validation exactly. The data is the
        env's ``data_loaders["val"]`` (the SSOT) — not a parameter, because the
        ``TrainingEnvironment`` is frozen; ``Trainer.evaluate`` builds the env
        with the caller's loader under the ``"val"`` key.
        """
        generator = self.pipeline.generator
        was_training = getattr(generator, "training", False)
        generator.eval()
        # #1353: a standalone eval drives the same validation, so it drives the
        # same lifecycle pair. Its own driver instance -- ``evaluate()`` is an
        # entry point in its own right, not a step of a run, so its first-fire
        # log line belongs to it and its epoch pair is (correctly) never opened.
        lifecycle = StrategyLifecycleDriver(self.strategy)
        lifecycle.begin_validation()
        try:
            metrics = _run_validation(
                self.pipeline,
                self.strategy,
                0,  # iteration — a standalone eval is not at any train step
                0,  # epoch
                self.logging_service,
                output_paths=self.output_paths,
                metrics_service=self.metrics_service,
                tb_writer=self.tb_writer,
                csv_file=None,
            )
        finally:
            # Drain here too (#697). A standalone eval runs the SAME cascade at
            # the same GPU cost; without this the rows sit on the strategy until
            # the next in-training validation absorbs them into its own sweep --
            # measurements computed and silently discarded, which is the failure
            # class the tall record exists to remove. In `finally` so a raising
            # validation cannot leave one eval's rows to be published later under
            # a different caller's iteration.
            _drain_cascade_rows(
                self.strategy,
                self.metrics_service,
                iteration=0,
                epoch=0,
                logging_service=self.logging_service,
                is_main_process=RankUtility.is_main_rank(),
            )
            # Restore the prior mode so evaluate() is side-effect-free on the
            # model (a caller may evaluate mid-training and keep training).
            if was_training:
                generator.train()
        # Outside the ``finally``: a validation that raised produced no metrics,
        # and ``on_validation_end`` promises the aggregated ones. Reporting {} to
        # a hook that freezes stages on "no improvement" would let a crash read
        # as a plateau.
        lifecycle.end_validation(metrics)
        return metrics or {}


#: ``metrics.compute_*`` flag -> metric name, for the ``losses.csv`` HEADER.
#:
#: Derived from the schema via ``flag_map.schema_flag_to_metric`` -- it used to be 78
#: hand-written entries, and the hand-maintenance was the bug. It was a SUPERSET of
#: ``MetricsMixin``'s 43-entry selection map on the theory that "a surplus column costs
#: an empty string, a missing one costs a silently discarded value". Half right: the
#: surplus was not free. 22 of those flags resolve to a registered metric that the
#: mixin could not select, so setting one bought a column header that nothing could
#: ever fill -- indistinguishable from a metric that ran and returned nothing (#340).
#: Both maps now derive from the same schema, so the superset is exactly the flags the
#: mixin also honours, and the asymmetry is gone rather than documented.
_CSV_METRIC_NAME_MAP: dict[str, str] = schema_flag_to_metric()


def _monitor_not_applicable_reason(strategy: Any, monitor_key: str) -> str:
    """Explain a persistently non-finite monitor, when the computer declared why.

    ``ValidationMetricsComputer`` records a machine-readable
    :class:`NotApplicableReason` for metrics it excluded (an unresolvable
    ``data_range``, a missing ROI mask), which a bare NaN cannot express. Recover
    it so the raise says *why* there is no number instead of only that there
    isn't one.

    Defensive by design, and safe to be: this decorates an exception message that
    is raised either way. A failed lookup degrades the explanation, never the
    decision -- which is the only shape in which a ``getattr`` chain over a
    strategy is acceptable here.
    """
    computer = getattr(strategy, "_validation_computer", None)
    reasons = getattr(computer, "last_not_applicable", None) or {}
    # The monitor is `val_<metric>`; the computer keys on the bare metric name.
    for name, reason in reasons.items():
        if monitor_key in (name, f"val_{name}"):
            return f"declared not-applicable ({reason}) by the metrics computer."
    if reasons:
        return f"the metrics computer declared these not-applicable this event: {sorted(reasons)}."
    return (
        "no not-applicable reason was recorded, so the NaN came from the metric "
        "itself (a diverged model, or an all-NaN forward pass)."
    )


#: What a strategy returns when validation could not produce a prediction AT ALL.
#:
#: A bare failure sentinel, NOT a metric vocabulary -- returned at
#: ``diffusion.py:3233`` (``hr_fakes is None``) and ``:3420`` (``not all_metrics``).
#: Any consumer that reads it as "the metrics this arm produces" misattributes an
#: upstream crash to the arm's configuration.
_VALIDATION_FAILURE_SENTINEL = "validation_error"


def _unresolvable_monitor_error(monitor: str, val_metrics: dict[str, Any], iteration: int) -> str:
    """Explain an unresolvable early-stopping monitor, blaming the right thing.

    Fatal in both branches, for the #178 reason at the call site: an unresolvable
    monitor costs early stopping AND ``checkpoint_best.pt`` at once, while the run
    exits 0 reporting success. Only the ATTRIBUTION is decided here.

    The distinction matters because the two causes have opposite remedies. A
    genuinely absent key is a YAML defect and the user must edit
    ``early_stopping.metric``. A ``val_metrics`` that holds *only*
    :data:`_VALIDATION_FAILURE_SENTINEL` means validation itself failed and
    returned its sentinel -- the monitor is absent as a CONSEQUENCE. The Aug-2026
    run that exposed this died with ``available: ['validation_error']`` and was
    told to fix a metric that was entirely correct: ``val_hfen_mean`` is
    synthesized by averaging the per-level ``val_hfen_{accel}x`` entries,
    ``hfen`` was in ``metrics.compute``, and the resolver already tries the
    ``val_``-prefixed spelling, the bare one and every cascading suffix. No edit
    to that YAML could have helped, and the message sent the user looking anyway.
    """
    if set(val_metrics) == {_VALIDATION_FAILURE_SENTINEL}:
        return (
            f"Validation FAILED at iteration {iteration}: it produced no metrics "
            f"at all, only the failure sentinel "
            f"{{'{_VALIDATION_FAILURE_SENTINEL}': ...}}.\n"
            f"  early_stopping.metric: '{monitor}' -- this is NOT the problem; "
            "do not change it.\n"
            "The upstream failure is logged at WARNING as 'Validation "
            "generation failed: ...' with a traceback, immediately before this "
            "line -- search the run log for that string. If the log is missing, "
            "provenance.json records where it went under `logging`.\n"
            "Fatal rather than skipped because early stopping and "
            "best-checkpoint selection both depend on a metric that will never "
            "arrive: every later validation event runs the same code on the same "
            "shapes, so continuing would train to max_iterations and write no "
            "best checkpoint while still reporting success."
        )
    return (
        f"Early-stopping monitor '{monitor}' is not among the validation metrics "
        f"this arm produces.\n"
        f"  tried:     {early_stop_monitor_candidates(monitor)}\n"
        f"  available: {sorted(val_metrics)}\n"
        "Early stopping AND best-checkpoint selection both depend on this key, "
        "so continuing would train to max_iterations and write no best "
        "checkpoint, while still reporting success. Fix `early_stopping.metric` "
        "to name a metric the arm computes, or disable early stopping."
    )


def lr_column_name(scheduler_name: str) -> str:
    """The ``training_metrics.csv`` column carrying one scheduler's LR.

    ONE OWNER (non-negotiable 17). The header builder and the row producer each
    spelled this out, and the header simply never spelled it at all: the
    producer wrote ``lr_opt_g`` while the header promised no ``lr_*`` column, so
    ``extrasaction="ignore"`` discarded the entire learning-rate curve of every
    arm that has a scheduler. Two sites naming one column is how that happens;
    a helper both call is how it stops.
    """
    return f"lr_{scheduler_name}"


def lr_column_names(schedulers: Any) -> set[str]:
    """Every LR column implied by a ``pipeline.schedulers`` mapping.

    Takes the mapping the producer iterates rather than a config block, so the
    header cannot promise a scheduler the loop does not have (or miss one it
    does). Duck-typed and None-safe: a pipeline with no schedulers is normal.
    """
    return {lr_column_name(name) for name in (schedulers or {})}


def _csv_metric_names(metrics_dict: dict[str, Any]) -> set[str]:
    """Metric names that need a ``losses.csv`` column.

    Mirrors ``MetricsMixin._extract_metrics_from_config`` EXACTLY: the
    ``metrics.compute`` list wins outright when non-empty, and the
    ``metrics.compute_*`` flags are consulted only as the fallback for an
    unmigrated arm. The header must promise precisely what the row producer can
    fill -- any other resolver makes this writer a second SSOT and guarantees a
    column nobody can populate.

    This USED TO UNION the two sources, for a real reason worth preserving:
    reading only the flags cost the drained ``kspace_filling`` cohort its
    ``hfen`` / ``kspace_error`` / ``phase_mse`` columns while still computing
    them (#696), because a drained arm declares the list and no flags, so every
    flag falls back to a schema default the arm never asked for. Precedence
    keeps that fix -- a drained arm's list still wins -- while dropping the
    union's cost, which the flags direction paid: ``experiment_11_attention_none``
    declares five metrics in ``compute`` and inherits ``compute_mse`` /
    ``compute_psnr`` / ``compute_ssim`` / ``compute_advanced_metrics`` as schema
    defaults, so the union promised THREE columns the producer never computes.
    An always-empty column is pitfall #15 in artifact form, and it is exactly
    what sent this arm's reader looking for a lost-data bug.

    The union was also serving as a safety net, because ``extrasaction="ignore"``
    silently DISCARDS a computed value with no column. That job moves to a loud
    write-time check at the row writer, which catches any future divergence
    between the two resolvers -- including surplus *loss* keys, which the union
    never covered.
    """
    compute = set(metrics_dict.get("compute") or [])
    if compute:
        return compute
    return {n for f, n in _CSV_METRIC_NAME_MAP.items() if metrics_dict.get(f, False)}


def _drain_cascade_rows(
    strategy: Any,
    metrics_service: Any,
    *,
    iteration: int,
    epoch: int,
    logging_service: Any = None,
    is_main_process: bool = True,
) -> int:
    """Aggregate, persist and clear the strategy's tall cascading sweep (#697).

    Returns the number of rows written, so "wrote nothing" is distinguishable
    from "was never called" -- the distinction the retired implementation could
    not make, because it built 45 column names and dropped every one of them.

    The strategy publishes one row per (val batch x severity point), because
    ``validation_step`` runs per batch and the cascade is not gated on
    ``batch_idx``. ``aggregate_cascade_rows`` collapses those to one row per
    severity point, matching how the suffixed ``val_*_<R>x`` columns beside them
    are averaged -- otherwise the two would carry different numbers under labels
    claiming the same measurement.

    Clearing is UNCONDITIONAL and outside the write branch. Rows accumulate per
    batch, so a rank that returns early without clearing would grow them without
    bound and carry one validation's measurements into the next. Duck-typed:
    most strategies have no cascade and no attribute, and that is not an error.

    Only the main process writes: every rank validates its own shard and so
    reaches this drain, but they share one output dir, and the writer's
    schema-evolution path renames-then-rewrites the file.

    Known limit: under DDP the written row covers the MAIN RANK'S SHARD, while
    the suffixed columns are all-reduced across ranks (``_all_reduce_val_metrics``).
    ``n_batches`` states the row's own provenance; a cross-rank reduction here
    is deliberately left out of scope rather than approximated silently.
    """
    rows = getattr(strategy, "_last_cascade_rows", None)
    if not rows:
        return 0

    if not is_main_process:
        strategy._last_cascade_rows = []
        return 0

    writer = getattr(metrics_service, "log_cascading_validation", None)
    if writer is None:
        # Fail soft but LOUD. A silent return here would recreate exactly the
        # bug this replaced: values computed at real GPU cost and then dropped
        # with nothing in the log to say so (pitfall #16).
        if logging_service is not None:
            logging_service.log_warning(
                f"[cascade] {len(rows)} cascading-validation row(s) computed at "
                f"iteration {iteration} but the metrics service "
                f"({type(metrics_service).__name__}) exposes no "
                "`log_cascading_validation` -- the R-sweep is NOT being recorded."
            )
        strategy._last_cascade_rows = []
        return 0

    written = writer(aggregate_cascade_rows(rows), iteration=iteration, epoch=epoch)
    strategy._last_cascade_rows = []
    return written


def resolve_scheduler_cadence(strategy: Any, config: Any) -> int:
    """How many iterations pass between LR-scheduler steps.

    Reads ``requested_gradient_accumulation_steps``, **not**
    ``gradient_accumulation_steps``. The distinction is the whole bug:
    ``StepExecutor._negotiate_capabilities`` sets the latter to ``1`` whenever
    the backend owns accumulation itself -- which DeepSpeed does
    (``DeepSpeedStepPolicy.owns_gradient_accumulation = True``). That is correct
    for the *step* path (DeepSpeed accumulates internally, so the executor must
    not accumulate again), but reading it for *cadence* made
    ``_should_step_schedulers`` fire on every iteration instead of every
    ``accumulation_steps``-th.

    With ``accumulation_steps: 2`` the cosine curve therefore ran at double rate
    under DeepSpeed **only**, so the parallel run was not comparable to the
    single-GPU baseline it was supposed to be ranked against -- in a shootout
    whose entire purpose is ranking arms against each other.

    The executor deliberately keeps the configured value under the
    ``requested_`` name so cadence decisions can still see it.
    """
    executor = getattr(strategy, "step_executor", None)
    requested = getattr(executor, "requested_gradient_accumulation_steps", None)
    if requested is not None:
        return max(1, int(requested))
    negotiated = getattr(executor, "gradient_accumulation_steps", None)
    if negotiated is not None:
        return max(1, int(negotiated))
    return max(1, int(config.optimization.gradient.accumulation_steps))


def resolve_ema_warmup_gate(ema: Any, ema_cfg: Any) -> int:
    """How many iterations the loop must SKIP before updating the EMA shadow.

    ``ema.warmup_steps`` means "the length of the EMA warmup period" on both
    EMA paths, but the mechanism differs, so exactly one of the two applies:

    * standard — a hard gate, returned here: no update happens at all, and the
      shadow stays at the random init for that long.
    * adaptive — a soft decay ramp ``initial_decay -> final_decay`` applied
      inside :class:`~spectramr.infrastructure.optimization.ema.ModelEma`, where
      the shadow instead TRACKS the live model from the first update. Stacking
      the hard gate on top would double the warmup period, so this returns 0.

    The decision is taken from the CONSTRUCTED EMA object, not re-derived from
    ``config.ema.enable_adaptive_ema`` (#1294). Until the adaptive ramp was
    implemented, reading the config zeroed ``warmup_steps`` on a promise
    nothing kept, and the knob was silently inert on every adaptive arm; an
    object cannot make a promise it did not keep.

    Args:
        ema: The built EMA tracker, or ``None`` when EMA is disabled.
        ema_cfg: ``config.ema``, or ``None``.

    Returns:
        Iterations to skip before the first EMA update. 0 means "update from
        the first iteration".
    """
    if getattr(ema, "adaptive", False):
        return 0
    return getattr(ema_cfg, "warmup_steps", 0) or 0


def resolve_iteration_budget(max_iterations: int, config: Any) -> int:
    """Apply ``training.iteration_budget_scope`` to the resolved loop bound.

    ``max_iterations`` has always been a **per-rank** bound: nothing on this
    path carries a ``world_size`` term -- not the loop range, not
    ``eval_interval``, not ``checkpoint_interval``, not any scheduler ``T_max``.
    Under ``torchrun --nproc_per_node=4`` each rank therefore runs the FULL
    count. The ranks do not split one experiment between them; they each run the
    whole experiment in lockstep, now also paying the ZeRO/DDP collectives at
    every accumulation boundary. Wall-clock is then *mathematically* >= the
    single-GPU run -- which is how an arm came to be slower on four GPUs than on
    one. Data parallelism buys effective batch (x N), never a shorter run.

    This function does not change that default. It makes it a **declared** choice
    and says out loud what the run is actually buying.

    Args:
        max_iterations: The fully-resolved per-rank bound.
        config: The frozen ``TrainingSettings``.

    Returns:
        The loop bound to use.

    Raises:
        ConfigurationError: ``global`` was requested (not supported yet -- see
            the message for the two independent reasons), or the scope is not a
            recognised value.
    """
    topology = resolve_run_topology()
    scope = getattr(config.training, "iteration_budget_scope", "per_rank")
    if scope == "global":
        raise ConfigurationError(
            "training.iteration_budget_scope: global is not supported yet.\n"
            "  Dividing the budget by world_size would be wrong twice over:\n"
            "  (1) Nothing shards the data (issue #1163). tio.Queue.__getitem__\n"
            "      ignores its index, so the DistributedSampler attached to it\n"
            "      shards nothing and each rank would train on a random 1/N\n"
            "      slice of the SAME unsharded stream.\n"
            "  (2) Every iteration-keyed schedule would silently reshape. The\n"
            "      diffusion curriculum ramps as start_t + iteration*rate, so a\n"
            "      ramp that opens the full ladder 16% into a 30k-iteration run\n"
            "      opens it 64% into a 7.5k one -- the same rate, a different\n"
            "      experiment. The EMA horizon and validation cadence move the\n"
            "      same way. A correct 'global' mode must rescale those too.\n"
            "  Use per_rank (the default) and scale max_iterations yourself, or\n"
            "  fix #1163 and rescale the iteration-keyed schedules with it."
        )
    if scope != "per_rank":
        # Unreachable while the schema Literal holds. If it fires, the schema and
        # this consumer have drifted -- raise rather than degrade to a default.
        raise ConfigurationError(
            f"Unknown training.iteration_budget_scope={scope!r}; expected 'per_rank' or 'global'."
        )
    if topology.is_distributed and topology.is_rank_zero:
        logger.info(
            "[BUDGET] iteration_budget_scope=per_rank: each of the %d ranks runs "
            "all %d iterations. That is ~%dx the GPU-hours for ~1x the wall-clock; "
            "what you gain is effective batch (x%d), not a shorter run.",
            topology.world_size,
            max_iterations,
            topology.world_size,
            topology.world_size,
        )
    return max_iterations


def _execute_training_loop(
    strategy,
    pipeline,
    config,
    model_type,
    tb_writer=None,
    logging_service=None,
    output_paths=None,
    checkpoint_service=None,
    metrics_service=None,
    is_sanity_check=False,
    start_iteration=0,
):
    """Iteration-based training loop.

    Args:
        start_iteration: Iteration to resume from (0 = start fresh).
    """

    # Resolve the batch's non-spatial axis identity ONCE (C8).
    #
    # It belongs here rather than in the collate boundary because that boundary
    # sees tensors, not an arm: ``BatchAdapter.from_dict`` has no dataset_type
    # and no config to reach one from. It belongs OUTSIDE the loop because
    # ``resolve_axes_for`` walks the config, and per-batch config traversal is
    # exactly the hot-path work the training-loop rules forbid.
    #
    # ``None`` here means unresolved, so every consumer skips -- identical to the
    # behaviour before this field existed.
    from spectramr.data.datasets.axis_exposure import resolve_axes_for

    batch_axes = resolve_axes_for(getattr(config, "data", None))
    if batch_axes is not None:
        logger.info(
            "[AXES] batch axes resolved: %s",
            sorted(a.value for a in batch_axes) or "(none — positively declared)",
        )

    # Resolve Max Iterations
    # SSOT: config.training.max_iterations
    max_iterations = (
        config.training.max_iterations if config.training.max_iterations is not None else -1
    )

    # WHERE the budget came from, tracked in lockstep with the value so the
    # launch banner can name it. Printing the number alone is not enough: three
    # separate mechanisms can produce one, they disagree silently, and the log
    # is often the only artifact left when a run is questioned days later.
    #
    # The concrete failure this closes: a 4-GPU run of
    # `experiment_11_attention_none` on 2026-08-21 was launched with
    # `-O training.max_iterations=5000`, while sanity-check mode independently
    # forces the budget to a hardcoded 5000 a few lines below. Both print
    # "Starting training for 5000 iterations", so the log could not answer
    # which one was in effect -- and the answers differ completely (one is the
    # operator's budget for a real run, the other is a single-batch overfit
    # probe that also silences the epoch-validation escape hatch at line ~1867).
    #
    # It deliberately does NOT claim "the user typed this". `main.py` injects
    # overrides of its own and the smoke dispatcher injects
    # `training.max_iterations=<cap>`; all arrive by the same route and are
    # indistinguishable here. The honest claim -- and the one a reader needs --
    # is that the value did not come from the config file.
    budget_source = (
        "-O/--override training.max_iterations"
        if "training.max_iterations" in applied_override_paths(config)
        else "training.max_iterations declared in the config file"
    )

    # Handle infinite/-1 or None
    if max_iterations <= 0:
        # SSOT: config.training.epochs
        epochs = config.training.epochs if config.training.epochs is not None else 100

        loader_len = (
            len(pipeline.data_loaders.get("train")) if pipeline.data_loaders.get("train") else 100
        )
        max_iterations = epochs * loader_len
        # Derived, so neither of the two branches above describes it: the arm
        # declared no usable `max_iterations` and this number is a product of
        # the epoch count and the length of the train loader, which is itself a
        # function of the dataset and batch size. Naming it stops a reader
        # hunting for a `max_iterations:` key that is not there.
        budget_source = f"derived from training.epochs={epochs} x train-loader length {loader_len}"
        logger.info(
            f"[Pipeline] Resolved max_iterations defined by epochs: {epochs} * {loader_len} = {max_iterations}"
        )

    # The budget the ARM declared, captured before sanity mode overwrites it
    # below. The validation-budget warning further down needs it to tell two
    # cases apart that otherwise read identically: a run the MODE made
    # unreachable (the arm is fine, nothing to fix) versus an arm that was
    # already unreachable on the budget it chose (a real defect -- three such
    # arms exist in `experiments/inprogress/`, see #1305). Saying "not a defect
    # in the arm" to the second group would be a false all-clear.
    # `resolve_iteration_budget` is applied below to the override rather than to
    # this value, which is safe because it is the identity for the only
    # reachable scope: `per_rank` only logs, and `global` raises.
    declared_max_iterations = max_iterations

    # 🧪 SANITY CHECK MODE OVERRIDE
    if is_sanity_check:
        # Wrapped, not replaced. The mode's budget is the one in force, but the
        # budget it CLOBBERED is exactly what an operator needs to see -- this
        # override lands after `--override/-O` is applied, so a caller who set
        # `training.max_iterations` watches their value silently vanish. Naming
        # both makes the coincidence of two independent 5000s readable instead
        # of misleading.
        budget_source = (
            f"sanity-check mode (forced; overrides the {max_iterations} from {budget_source})"
        )
        max_iterations = 5000
        logger.info(f"🧪 [SANITY CHECK MODE] Overriding max_iterations to {max_iterations}")
        if logging_service:
            logging_service.log_info(
                f"🧪 SANITY CHECK MODE: Forcing iterations to {max_iterations} to test severe overfitting."
            )

    # Iteration-budget scope. See ``resolve_iteration_budget``: max_iterations
    # is a PER-RANK loop bound, which is why a multi-GPU run was not faster.
    max_iterations = resolve_iteration_budget(max_iterations, config)

    # FIX #4: Direct access to required interval fields (with defensive defaults)
    log_interval = config.logging.intervals.log

    # An arm whose entire budget is shorter than its logging cadence produces an
    # empty metrics curve, and used to do so in silence: the header was written
    # unconditionally while `iteration % log_interval == 0` was satisfied ZERO
    # times, so `logs/training_metrics.csv` carried 29 column names and no rows
    # -- and every train TensorBoard scalar was suppressed with it -- while the
    # run exited reporting success. Measured on `experiment_11_attention_none`
    # (`logging.intervals.log: 5000`, schema default 100).
    #
    # The gate below now logs the first and last iteration unconditionally, so
    # the curve is no longer EMPTY. A budget this far under the cadence still
    # means a two-point curve, which is a configuration mistake worth naming
    # rather than absorbing.
    #
    # WARNING rather than INFO, deliberately: `LoggingService.setup` clamps every
    # logger AND handler to `logging.sinks.level`, which is `warning` on the arms
    # this was found on, so an INFO here would be discarded precisely where it is
    # needed.
    if log_interval > 0 and max_iterations < log_interval:
        logger.warning(
            "[Pipeline] logging.intervals.log=%d exceeds this run's whole budget "
            "of %d iterations, so the periodic metrics gate can never fire. The "
            "first and last iterations are logged unconditionally, so the metrics "
            "CSV and the train TensorBoard scalars will hold 2 points rather than "
            "0 -- but set logging.intervals.log to at most %d for a readable curve.",
            log_interval,
            max_iterations,
            max(1, max_iterations // 10),
        )

    # The SAME failure mode, one interval down, and it is the one the user
    # actually hit: `metrics.train_metric_interval` (schema default 100) gates
    # `MetricsMixin._compute_training_metrics`, so a run whose whole budget is
    # under that cadence computes NO train metric at all and every `train_*`
    # column stays empty while the loss columns fill normally. Measured on
    # `experiment_11_attention_none` at `max_iterations: 40`: `hfen` (the LOSS)
    # populated, `train_hfen` (the METRIC) empty, on all 40 rows.
    #
    # This is a REGRESSION INTRODUCED BY A CORRECT FIX. `self.env.step` used to
    # be a frozen 0, so `0 % train_metric_interval == 0` fired on EVERY step
    # (pitfall #16 -- the throttle was a facade). `resolve_loop_iteration`
    # (`loop_state.py:57-74`) now returns the live 1-based iteration, which is
    # right -- but it flipped short runs from always-on to NEVER, and nobody
    # noticed the short-run case had inverted.
    #
    # The mixin gate now also fires on the first and final iteration (see
    # `metrics_mixin._compute_training_metrics`), so the columns are no longer
    # structurally empty. A budget under the cadence still means a two-point
    # curve, which is worth naming rather than absorbing.
    #
    # WARNING for the same reason as the block above: `LoggingService.setup`
    # clamps every logger AND handler to `logging.sinks.level`.
    train_metric_interval = (
        getattr(config.metrics, "train_metric_interval", 0)
        if getattr(config, "metrics", None) is not None
        else 0
    )
    if train_metric_interval > 0 and max_iterations < train_metric_interval:
        logger.warning(
            "[Pipeline] metrics.train_metric_interval=%d exceeds this run's whole "
            "budget of %d iterations, so the periodic train-metric gate can never "
            "fire on its own. The first and last iterations are now computed "
            "unconditionally, so the `train_*` columns will hold 2 points rather "
            "than 0 -- but set metrics.train_metric_interval to at most %d for a "
            "readable curve.",
            train_metric_interval,
            max_iterations,
            max(1, max_iterations // 10),
        )
    eval_interval = (
        config.validation.schedule.interval_steps if config.validation else max_iterations
    )
    eval_on_epoch = config.validation.schedule.on_epoch if config.validation else False
    # The N of epoch-based mode. Declared, documented as "only consulted in
    # epoch-based mode", and read by NOTHING until now (#711) -- because the
    # mode it belongs to could never be entered.
    eval_interval_epochs = config.validation.schedule.interval_epochs if config.validation else 1

    # %-style, not an f-string: the source strings are built above and this line
    # is the one every arm prints, so lazy formatting is the cheap default.
    logger.info(
        "[Pipeline] Starting training for %d iterations (budget source: %s)...",
        max_iterations,
        budget_source,
    )

    # =====================================================================
    # FIX: Initialize training_metrics.csv with all fieldnames upfront
    # =====================================================================
    # DDP rank-safety: only the main rank writes shared artifacts (CSV / TB /
    # checkpoints / images / final_metrics) to the shared output dir; otherwise
    # every rank races on the same files. ``is_main_rank()`` is True on the
    # single-process path, so non-DDP behavior is unchanged. Validation COMPUTE
    # still runs on all ranks (its metric all-reduce is a collective) — only the
    # side effects are gated.
    is_main_process = RankUtility.is_main_rank()

    # Checkpoints are the ONE shared write that rank-0 gating gets wrong.
    #
    # Under FSDP and DeepSpeed, building the checkpoint is a COLLECTIVE:
    # FSDP's state_dict() all-gathers the shards, and engine.save_checkpoint()
    # synchronises across ranks. Gating those on `is_main_process` means rank 0
    # enters a collective that ranks 1..N never enter, so the job HANGS -- no
    # exception, no log line, just a silent job burning GPU-hours until SLURM
    # kills it at walltime. That is strictly more expensive than a crash.
    #
    # One predicate, derived once, used at every checkpoint site: the adapter
    # decides who *writes*, this decides who *participates*. A future
    # checkpoint call site inherits the answer instead of re-deriving it.
    parallel_runtime = getattr(pipeline, "parallel", None)
    # The step policy is resolved by the plugin but the strategy (and its
    # StepExecutor) is built before the parallel runtime exists, so it has to be
    # installed here -- see _install_parallel_step_policy for what silently
    # breaks without it.
    _install_parallel_step_policy(strategy, parallel_runtime)
    checkpoints_need_all_ranks = bool(
        getattr(parallel_runtime, "checkpoints_require_all_ranks", False)
    )
    may_checkpoint = is_main_process or checkpoints_need_all_ranks
    if checkpoints_need_all_ranks and not is_main_process:
        logger.info(
            "[Pipeline] parallel strategy %r requires collective checkpointing; "
            "this non-zero rank participates in save/load (the adapter decides "
            "which rank writes).",
            getattr(parallel_runtime, "strategy", "?"),
        )

    # SSOT: Write to output_paths["csv_log_file"] which comes from config.training.output_dir
    # Initialize with standard fields + all configured losses upfront to avoid duplication
    csv_file = None
    csv_fieldnames = None
    # Keys already reported as column-less, so the check below warns ONCE per
    # key rather than once per row. See the write site for why it exists.
    csv_discarded_keys_warned: set[str] = set()
    # Schedulers already reported as yielding no LR, so the emission site
    # warns ONCE per scheduler rather than once per row.
    csv_empty_lr_warned: set[str] = set()
    csv_initialized_once = False
    # Consecutive validation events whose monitor value was non-finite. Reset by
    # the first finite one; escalates to a raise at `patience` (see the monitor
    # block below), because in that state neither early stopping nor best-
    # checkpoint selection can ever fire again.
    consecutive_nonfinite_monitor = 0
    if output_paths and "csv_log_file" in output_paths and is_main_process:
        csv_file = output_paths["csv_log_file"]
        try:
            # Create parent directories if needed
            Path(csv_file).parent.mkdir(parents=True, exist_ok=True)

            # Initialize fieldnames with standard fields + all configured losses
            csv_fieldnames = ["iteration", "epoch"]

            # Add all configured loss fieldnames from config upfront
            # This prevents dynamic header rewrites that can cause duplication
            expected_loss_keys = {"g_total_loss", "loss"}

            # [SSOT] Get expected keys from strategy itself
            if hasattr(strategy, "loss_weights"):
                for loss_name in strategy.loss_weights.keys():
                    expected_loss_keys.add(loss_name)

            if hasattr(config, "losses") and config.losses is not None:
                if hasattr(config.losses, "get_enabled_losses"):
                    enabled_losses = config.losses.get_enabled_losses()
                    # enabled_losses is a flat dict: {loss_name: weight}, not nested
                    for loss_name, weight in enabled_losses.items():
                        if weight > 0:
                            expected_loss_keys.add(loss_name)
                            # GAN specific cases
                            if loss_name == "adversarial":
                                expected_loss_keys.add("g_adv_loss")
                                expected_loss_keys.add("d_real_loss")
                                expected_loss_keys.add("d_fake_loss")
                                expected_loss_keys.add("d_total_loss")
                            elif loss_name == "gradient_penalty":
                                expected_loss_keys.add("gradient_penalty")

            # [DISENTANGLED] Add strategy-specific metrics
            # Check if strategy is disentangled-based (could be in strategy_class name or task designation)
            is_disentangled = (
                config.training
                and hasattr(config.training, "strategy_class")
                and config.training.strategy_class
                and "disentangled" in config.training.strategy_class.lower()
            ) or (
                config.training
                and hasattr(config.training, "task")
                and config.training.task
                and "disentangled" in str(config.training.task).lower()
            )
            if is_disentangled:
                expected_loss_keys.update(
                    [
                        "recon_a",
                        "recon_b",
                        "recon_ba",
                        "style_latent",
                        "content_latent",
                        "kl",
                        "bloch",
                        "anat_lock",
                        "g_adv_loss",
                        "d_total_loss",
                    ]
                )
                # Add components that might be enabled
                for l_name in [
                    "mind_ssc",
                    "ssim",
                    "ms_ssim",
                    "perceptual",
                    "hfen",
                    "ffl",
                    "hist",
                    "explicit_gradient",
                ]:
                    expected_loss_keys.add(l_name)

            # Learning-rate columns, one per scheduler. The row producer writes
            # `losses_scalar[lr_column_name(sched_name)]` for every entry in
            # `pipeline.schedulers` (see the "[FIX] Log current learning rate(s)"
            # block at the write site), and the header promised NONE of them, so
            # `extrasaction="ignore"` discarded every LR on every row of every
            # arm that has a scheduler. The whole learning-rate curve was being
            # measured and thrown away -- `lr_opt_g` was one of the three keys
            # the write-time discard check named on experiment_11_attention_none.
            #
            # Derived from the SAME mapping the producer iterates, not from a
            # config re-read: a second resolver here is exactly the divergence
            # this header/producer pair keeps being bitten by (non-negotiable
            # 17). Schedulers are built during bootstrap, before the loop is
            # entered, so the names are known at header time.
            expected_lr_keys = lr_column_names(getattr(pipeline, "schedulers", None))

            # Dynamically extract training metrics from config.metrics
            expected_metrics = set()
            if hasattr(config, "metrics") and config.metrics is not None:
                metrics_dict = (
                    config.metrics.model_dump() if hasattr(config.metrics, "model_dump") else {}
                )

                for metric_name in _csv_metric_names(metrics_dict):
                    expected_metrics.add(f"train_{metric_name}")

            # [#1682] A CSV row has THREE upstream sources, and only one of them
            # is the config. Resolving the header from the config alone is what
            # made three measured values vanish every step on
            # ``experiment_11_attention_none``. The sources are:
            #
            #   (a) the step executor -- ``f"{name}_loss"``. Excluded on purpose;
            #       see ``_EXECUTOR_ALIAS_KEYS`` at module scope for why.
            #   (b) the strategy -- ``_last_step_metrics`` / ``_loss_dict_reuse``.
            #       Config-derivable EXCEPT where the producer renames between
            #       knob and metric, which only the producer can bridge, so it
            #       declares those keys itself.
            #   (c) this loop -- ``lr_{sched_name}`` at the emission site below.
            #       Loop-owned start to finish, so the loop seeds them here.
            #
            # (c): ``pipeline.schedulers`` is assigned once when the training
            # environment is constructed (``builders/environment.py:127``) and
            # never mutated afterwards, so its keys are already final here --
            # this is the same dict the emission site iterates, not a guess at
            # what it will hold.
            expected_metrics.update(resolve_scheduler_lr_columns(pipeline))

            # (b): producer-declared keys. These join the LOSS set, not the
            # metric set, so they also appear in the "Configured Losses in
            # Header" startup line -- they are loss terms, and that line is how
            # an operator confirms a declared term reached the header.
            expected_loss_keys.update(resolve_producer_declared_columns(strategy))

            # NO `validation.scoring.compute` NAMES HERE EITHER. This block used
            # to union them in as `train_*`, on the premise that "a metric an arm
            # scores at validation time is normally computed on the training
            # batch too". That premise is false: the training-metric computer is
            # fed ONLY `_extract_metrics_from_config(config.metrics)` -- there is
            # no path by which `validation.scoring.compute` reaches a training
            # batch. On `experiment_11_attention_none` the premise cost a
            # permanently-empty `train_psnr` column, because the arm scores
            # `psnr` at validation and does not list it in `metrics.compute`.

            # NO `val_*` COLUMNS IN THIS FILE. This writer cannot populate them,
            # and a header that promises them is pitfall #15 in artifact form.
            #
            # The row written below is `{"iteration", "epoch", **losses_scalar}`,
            # and `losses_scalar` derives from `losses_history` -- the TRAINING
            # step's losses. Validation metrics go to their own file, written by
            # `train.py` (`validation_csv_for`) on its own cadence. Measured
            # across every populated training CSV in `tests_experiments/`:
            # **70 files, 20,959 data rows, zero `val_*` cells populated.**
            #
            # An always-empty column is worse than an absent one: a reader cannot
            # tell "validation did not run" from "this file never carries this
            # column". Three independent downstream workarounds exist for exactly
            # that ambiguity, and each becomes a no-op once the columns are gone:
            #   * `_summarise_best_metrics_from_csv` reads the validation CSV as
            #     well -- feeding `final_metrics.best` from this file alone made
            #     every headline artifact say validation produced nothing, on runs
            #     where it produced alarming numbers (#481);
            #   * `_select_current_run_rows` plus the `final_iteration` filter
            #     empty the window rather than attribute a previous run's curve to
            #     this one (#586);
            #   * `_melt_metrics_csv` drops all-empty columns so they cannot
            #     surface as phantom all-NaN series in the learning-curve figure.
            # All three key off the header actually present, so dropping the
            # columns is safe for every known consumer.

            # The cascading acceleration sweep is NOT a set of columns here.
            #
            # It used to be: a second copy of the severity levels crossed with a
            # hardcoded 15-name metric list built 45 names -- `val_psnr_2x`,
            # `val_psnr_8x`, ... -- that were assembled into a local and never
            # extended onto `csv_fieldnames`, so the row writer's
            # `extrasaction="ignore"` discarded every value. Dead since it was
            # written; `git log -S` finds one commit and its parent had the same
            # write-only references.
            #
            # It now lands in its own TALL file, one row per (iteration,
            # severity point), with `acceleration_level` and `timestep` as
            # values -- see `core/cascading_validation.py`. This file stays one
            # row per ITERATION: making it tall would repeat every training loss
            # once per level, and `metrics_report_generator.py` reads it with
            # `pd.read_csv(..., on_bad_lines="error")`.
            csv_fieldnames.extend(sorted(expected_loss_keys))
            csv_fieldnames.extend(sorted(expected_metrics))
            csv_fieldnames.extend(sorted(expected_lr_keys))
            csv_fieldnames = list(
                dict.fromkeys(csv_fieldnames)
            )  # Remove duplicates while preserving order

            # Only write header once at startup
            file_exists = os.path.isfile(csv_file)
            if not file_exists:
                with open(csv_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                    writer.writeheader()
                logging_service.log_info(
                    f"Initialized training metrics CSV with {len(csv_fieldnames)} fields: {csv_file}"
                )
                logging_service.log_info(
                    f"[CSV] Configured Losses in Header: "
                    f"{', '.join([k for k in csv_fieldnames if k in expected_loss_keys and k not in ['g_total_loss', 'loss', 'diffusion']])}"
                )
                csv_initialized_once = True
            elif file_exists:
                # ⚠️ CRITICAL: If file exists, MUST update header to include new enabled losses
                # This handles the case where config changed between runs
                try:
                    with open(csv_file, newline="") as f:
                        reader = csv.DictReader(f)
                        existing_fieldnames = list(reader.fieldnames) if reader.fieldnames else []

                    # Sanity-check: if the file was written without a proper header
                    # (e.g. a previous crash left raw data rows as the first line),
                    # the DictReader will misinterpret the data row as the header.
                    # Detect this by checking whether "iteration" appears as a field.
                    is_corrupted = (
                        len(existing_fieldnames) == 0 or "iteration" not in existing_fieldnames
                    )
                    if is_corrupted:
                        logging_service.log_warning(
                            f"[CSV] Existing file appears corrupted (no 'iteration' column). "
                            f"Recreating with correct header. existing_fieldnames={existing_fieldnames[:5]}"
                        )
                        with open(csv_file, "w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
                            writer.writeheader()
                        # csv_fieldnames already correct — nothing more to do
                    else:
                        # Merge: existing fields + newly configured losses
                        # This ensures new losses added to config show up in CSV
                        merged_fieldnames = list(
                            dict.fromkeys(existing_fieldnames + csv_fieldnames)
                        )

                        if merged_fieldnames != existing_fieldnames:
                            # Schema changed: read all data, rewrite with new header
                            logging_service.log_info(
                                f"[CSV] Schema evolution detected: {len(existing_fieldnames)} → {len(merged_fieldnames)} fields"
                            )
                            logging_service.log_info(
                                f"[CSV] Added {len(set(merged_fieldnames) - set(existing_fieldnames))} new columns: "
                                f"{sorted(set(merged_fieldnames) - set(existing_fieldnames))}"
                            )

                            # Read existing data (might be large)
                            data_rows = []
                            try:
                                with open(csv_file, newline="") as f:
                                    reader = csv.DictReader(f)
                                    data_rows = list(reader)

                                # Rewrite with merged headers
                                with open(csv_file, "w", newline="") as f:
                                    writer = csv.DictWriter(f, fieldnames=merged_fieldnames)
                                    writer.writeheader()
                                    writer.writerows(data_rows)

                                logging_service.log_info(
                                    f"[CSV] Migrated {len(data_rows)} rows with new schema"
                                )
                            except Exception as e:
                                logging_service.log_error(f"[CSV] Failed to migrate schema: {e}")

                        csv_fieldnames = merged_fieldnames

                except Exception as e:
                    logging_service.log_warning(f"Failed to merge existing CSV fieldnames: {e}")
        except Exception as e:
            logging_service.log_warning(f"Failed to initialize training_metrics.csv: {e}")

    # Setup
    if is_sanity_check:
        pipeline.models.get("generator").eval()
        if logging_service:
            logging_service.log_info(
                "🧪 SANITY CHECK MODE: Generator set to eval() to freeze BatchNorm and zero Dropout."
            )
    else:
        pipeline.models.get("generator").train()
    # Scaler comes from pipeline
    scaler = getattr(pipeline, "scaler", None)

    # 🧪 SANITY CHECK OVERRIDE: Capture single batch and repeat infinitely
    train_loader = pipeline.data_loaders.get("train")
    _has_train_loader = bool(train_loader and len(train_loader) > 0)
    train_loader_len = len(train_loader) if _has_train_loader else 1

    # The THIRD interval-vs-budget gate, and the only one that was never
    # checked. Its two siblings above (`logging.intervals.log`,
    # `metrics.train_metric_interval`) each get a budget warning AND an
    # unconditional first/last-iteration force. `validation.schedule.
    # interval_steps` has neither: the gate at ~line 1660 is a bare
    # `iteration % eval_interval == 0`. So the one interval whose absence is
    # most expensive is the one with no protection at all.
    #
    # This is checked HERE rather than beside its siblings (~line 754) because
    # it needs the loaders: `max_iterations` can be epoch-derived at runtime
    # (~line 647), and both gates depend on the val loader's existence and the
    # train loader's length. A Pydantic cross-field validator cannot see any
    # of that, which is why this is a runtime guard and not a schema rule.
    #
    # FATAL, matching the #178 ruling at ~line 1946 rather than the WARNING its
    # siblings get. Zero validation events costs BOTH mechanisms at once --
    # early stopping never evaluates and `checkpoint_best.pt` is never written
    # -- and the run still exits 0 reporting success. #178 already called that
    # exact consequence pair fatal; the only difference here is that the
    # RuntimeError it installed lives INSIDE a validation event, so zero events
    # makes the guard itself unreachable (pitfall #16: a mechanism that reads
    # as protection but cannot fire).
    _val_loader_present = bool(pipeline.data_loaders.get("val"))
    _can_fire = validation_can_fire(
        eval_interval=eval_interval,
        max_iterations=max_iterations,
        # Sanity checks obey `eval_interval` strictly (see the gate at ~1674),
        # so the epoch escape hatch does not exist in that mode.
        eval_on_epoch=eval_on_epoch and not is_sanity_check,
        eval_interval_epochs=eval_interval_epochs,
        train_loader_len=train_loader_len,
        has_train_loader=_has_train_loader,
    )
    _epoch_gate_reaches = epoch_validation_can_fire(
        max_iterations=max_iterations,
        eval_on_epoch=eval_on_epoch and not is_sanity_check,
        eval_interval_epochs=eval_interval_epochs,
        train_loader_len=train_loader_len,
        has_train_loader=_has_train_loader,
    )

    # `not is_sanity_check` is load-bearing, and is the one exemption this
    # guard makes. Sanity-check mode OVERWRITES `max_iterations` with 5000 at
    # ~line 671 -- after any `--override/-O` override has been applied -- so an arm
    # declaring a perfectly consistent `max_iterations: 15000` /
    # `interval_steps: 15000` pair is made inconsistent BY THE MODE, over a
    # budget it never chose and cannot override. Raising there would block a
    # sanity check on 57 of the 647 `experiments/inprogress/` arms (measured
    # 2026-08-21) and the only way out would be editing the YAML. So sanity
    # mode gets the diagnosis without the veto: warn, attribute the budget to
    # the mode, and let the check run.
    if _val_loader_present and not _can_fire and not is_sanity_check:
        raise ConfigurationError(
            f"validation.schedule.interval_steps={eval_interval} exceeds this "
            f"run's whole budget of {max_iterations} iterations, so the "
            f"validation gate can NEVER fire: this run would train to "
            f"completion, evaluate early stopping zero times, write no "
            f"checkpoint_best.pt, and still exit reporting success. "
            f"Set validation.schedule.interval_steps to at most "
            f"{max(1, max_iterations // 2)} (>= 2 events), or enable "
            f"validation.schedule.on_epoch with a train loader of "
            f"{train_loader_len} step(s) per epoch. If you shortened the run "
            f"with `-O training.max_iterations={max_iterations}`, pass "
            f"`-O validation.schedule.interval_steps="
            f"{max(1, max_iterations // 2)}` alongside it to match."
        )

    # Same unreachable gate, but the budget is the MODE's -- see above. Still
    # reported, because a sanity check that silently skips validation is a
    # sanity check that cannot catch a val-time defect, and the whole point of
    # running one is to find defects cheaply.
    #
    # Two different findings share this branch, and they must not share a
    # message. If the interval also exceeds the budget the ARM declared, the
    # mode is not the cause: the arm would skip validation just as completely on
    # a full-length run, and it is the raise above that it will meet the moment
    # it runs outside sanity mode. Telling that operator "not a defect in the
    # arm" would be a false all-clear on the one occasion the mode surfaced a
    # real defect early -- so the attribution is conditioned, not asserted.
    elif _val_loader_present and not _can_fire:
        _arm_survives_own_budget = validation_can_fire(
            eval_interval=eval_interval,
            max_iterations=declared_max_iterations,
            eval_on_epoch=eval_on_epoch,
            eval_interval_epochs=eval_interval_epochs,
            train_loader_len=train_loader_len,
            has_train_loader=_has_train_loader,
        )
        if _arm_survives_own_budget:
            logger.warning(
                "[Pipeline] SANITY CHECK MODE compressed this run's budget to %d "
                "iterations, below this arm's validation.schedule.interval_steps="
                "%d, so validation will NOT run at all during this check -- it "
                "exercises the training path only. This is the mode's budget, not "
                "a defect in the arm: on its declared budget of %d, validation "
                "does fire. Override validation.schedule.interval_steps to at "
                "most %d to exercise the validation path too.",
                max_iterations,
                eval_interval,
                declared_max_iterations,
                max(1, max_iterations // 2),
            )
        else:
            logger.warning(
                "[Pipeline] validation.schedule.interval_steps=%d exceeds this "
                "arm's OWN declared budget of %d iterations, so validation can "
                "never fire -- on this sanity check (budget compressed to %d by "
                "the mode) and equally on a full-length run. This is a defect in "
                "the arm, not an artefact of sanity mode, and it is fatal outside "
                "this mode. Set validation.schedule.interval_steps to at most %d, "
                "or enable validation.schedule.on_epoch.",
                eval_interval,
                declared_max_iterations,
                max_iterations,
                max(1, declared_max_iterations // 2),
            )

    # The degenerate-but-legal case: exactly ONE validation event, landing on
    # the final iteration. Not fatal -- the arm does validate, and a run
    # deliberately validating once at the end is a legitimate choice -- but it
    # voids the cost argument the #178 guard states verbatim ("This fires on
    # the FIRST validation event, so the cost of a typo'd monitor is one
    # validation pass rather than the whole budget"). With the only event on
    # the last iteration, ANY validation-time failure -- an unresolvable
    # monitor, a mask-schedule contract violation, a val-time OOM -- costs the
    # entire budget before it is discovered. Observed on
    # `experiment_11_attention_none` overridden to `max_iterations=5000` with
    # the arm's declared `interval_steps: 5000`: 5 h 17 m on 4 GPUs, no best
    # checkpoint, and the failure was deterministic and weight-independent.
    #
    # Gated on `config.validation` being declared: `eval_interval` FALLS BACK
    # to `max_iterations` when the block is absent, and warning about a
    # value the user never wrote would be noise on every validation-less arm.
    #
    # WARNING rather than INFO for the same reason as its two siblings:
    # `LoggingService.setup` clamps every logger AND handler to
    # `logging.sinks.level`, so an INFO is discarded on the arms that need it.
    elif (
        _val_loader_present
        and config.validation is not None
        and eval_interval == max_iterations
        # `max_iterations == 1` is the fixed point where this advice eats
        # itself: the recommended `max_iterations // 2` floors to 1, which is
        # the value already set, so the warning would tell the user to write
        # what they already wrote. A 1-iteration arm validating on iteration 1
        # is also not degenerate -- it is the only schedule available. 13 of
        # the 647 `inprogress/` arms are exactly this shape (measured
        # 2026-08-21), all deliberate 1-step eval arms.
        and max_iterations > 1
        and not _epoch_gate_reaches
    ):
        logger.warning(
            "[Pipeline] validation.schedule.interval_steps=%d equals this run's "
            "whole budget, so validation runs exactly ONCE, on the final "
            "iteration %d. Early stopping can never act on it and any "
            "validation-time failure costs the entire budget before it is "
            "seen. Set validation.schedule.interval_steps to at most %d to get "
            "an early event that fails fast.",
            eval_interval,
            max_iterations,
            max(1, max_iterations // 2),
        )

    def _set_sampler_epoch(loader: Any, current_epoch: int) -> None:
        """Notify the loader's sampler of the new epoch.

        Required for ``DistributedSampler`` to reshuffle deterministically
        across ranks; benign no-op for samplers without ``set_epoch``.
        Without this call, every epoch under DDP uses the same shuffle,
        which silently degrades training-data variability.
        """
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(current_epoch)

    if is_sanity_check and train_loader:
        if logging_service:
            logging_service.log_info(
                "🧪 SANITY CHECK MODE: Capturing exactly 1 batch and repeating it infinitely..."
            )
        single_batch = next(iter(train_loader))

        # 🧪 Freeze diffusion timestep:
        diff_cfg = config.training.diffusion if config.training else None
        if diff_cfg and hasattr(diff_cfg, "timesteps"):
            fixed_ts = diff_cfg.timesteps // 2
            if logging_service:
                logging_service.log_info(
                    f"🧪 SANITY CHECK MODE: Freezing diffusion timestep to {fixed_ts}."
                )

            if isinstance(single_batch, dict):
                # SSOT: "input" is the canonical batch key. A silent fallback
                # to torch.empty(1) would set batch_size=1 and apply a wrong
                # timestep tensor to whatever else is in the batch. Raise so
                # the dataloader/strategy mismatch surfaces immediately.
                # See findings booklet 2026-05-05 PI-3.
                if "input" not in single_batch:
                    raise KeyError(
                        "sanity-check mode requires 'input' in the batch dict; "
                        f"got keys {list(single_batch.keys())}"
                    )
                bs = single_batch["input"].shape[0]
                single_batch["timestep"] = torch.full((bs,), fixed_ts, dtype=torch.long)
            elif hasattr(single_batch, "input"):
                bs = single_batch.input.shape[0]
                if hasattr(single_batch, "metadata"):
                    single_batch.metadata["timestep"] = torch.full(
                        (bs,), fixed_ts, dtype=torch.long
                    )

        # Sanity check intentionally trains on one batch forever — no
        # epoch boundaries, no sampler reshuffle.
        data_iter = iter(cycle([single_batch]))
        data_epoch = -1  # disabled
    else:
        # Normal path: track which epoch the iterator belongs to so we
        # rebuild it (and call sampler.set_epoch) on each StopIteration.
        # On resume, start at the epoch implied by start_iteration so
        # DDP shuffles are consistent with where training left off.
        data_epoch = start_iteration // train_loader_len
        _set_sampler_epoch(train_loader, data_epoch)
        data_iter = iter(train_loader)

    # The loop is `range(start_iteration + 1, max_iterations + 1)`, so the first
    # iteration is NOT 1 on a resumed run. Bound once here because the logging
    # gate compares against it on every step.
    first_iteration = start_iteration + 1
    pbar = tqdm(range(first_iteration, max_iterations + 1), desc="Training")

    # Wall-clock budget, resolved BEFORE the first step so a malformed or
    # already-expired one fails at startup rather than after hours of GPU.
    wall_clock_budget = resolve_wall_clock_budget()
    wall_clock_yield = False
    # Its OWN cadence, capped independently of `log_interval`: 65 corpus arms
    # declare `logging.intervals.log: 5000`, which at ~1s a step would inspect
    # the deadline every ~83 min against a 900s margin -- the job dies before
    # the check fires, on exactly the long arms this is for. Reading the clock
    # is a vDSO call, not a device sync, so non-negotiable 9 does not bear on
    # how often it happens.
    _wall_clock_every = resolve_check_interval(log_interval)
    # Duration of the most recent validation pass, used to decide whether the
    # next one fits before the wall. None until one has been measured; the
    # first pass therefore runs unguarded, which is safe because it happens
    # early in the run, far from any deadline.
    _last_val_seconds: float | None = None
    if start_iteration > 0 and logging_service:
        logging_service.log_info(f"[Resume] Training resumes from iteration {start_iteration + 1}")

    # Initialize Early Stopping Service
    early_stopping_service = None
    if config.early_stopping and config.early_stopping.enabled:
        try:
            from spectramr.infrastructure.services.early_stopping import (
                EarlyStoppingService,
            )

            early_stopping_service = EarlyStoppingService(config.early_stopping)

            if logging_service:
                monitor = config.early_stopping.metric
                patience = config.early_stopping.patience
                logging_service.log_info(
                    f"Early Stopping Enabled: monitor={monitor}, patience={patience}"
                )
        except ImportError:
            logger.warning("[Pipeline] Warning: Could not import EarlyStoppingService")

    # Initialize the loss-schedule controller (dynamic loss-term weights:
    # clock curriculum + plateau triggers + post-change metric monitoring).
    # Gated on ``loss_schedule.enabled``; absent/disabled => no controller and
    # the loop never touches ``loop_state.loss_weight_overrides`` (stays empty =>
    # static-config behavior). See infrastructure/training/loss_schedule_controller.py.
    loss_schedule_controller = None
    if config.loss_schedule and config.loss_schedule.enabled:
        from spectramr.infrastructure.training.loss_schedule_controller import (
            LossScheduleController,
        )

        loss_schedule_controller = LossScheduleController(
            config.loss_schedule,
            loss_config=config,
            # Resume seeding (M4): pre-fire clock triggers already past so a
            # resumed run continues the curriculum rather than replaying it.
            start_iteration=start_iteration,
            start_epoch=start_iteration // train_loader_len,
        )
        if logging_service:
            logging_service.log_info(
                f"Loss schedule enabled: {len(config.loss_schedule.rules)} rule(s)"
            )

    # Path of the best checkpoint saved during early stopping, so we can honor
    # early_stopping.restore_best_weights after the loop (previously inert —
    # training ended on the latest, often-degraded weights; CLAUDE.md #15).
    best_checkpoint_path = None
    # Why the LAST failure and not a list: a best-save that fails once usually
    # fails every time (permissions, disk, a collective mismatch), so a list is
    # the same string N times. What a reader needs is "did this run fail to write
    # its best checkpoint, and why" (#713).
    best_checkpoint_error: str | None = None

    # Metrics history
    losses_history = {}

    # Sanity-check (overfit-single-batch) verdict traces. Populated ONLY in
    # sanity mode so the normal training loop pays no GPU-sync cost. A
    # correctly-wired model memorises one batch; these traces let us assert
    # the loss actually collapses (and, when tracked, the k-space phase
    # metric improves — the experiment_11 "DC blob" failure mode).
    sanity_loss_trace: list[float] = []
    sanity_phase_trace: list[float] = []

    # FIX #4: Direct config access for checkpoint settings (no fallbacks)
    checkpoint_enabled = config.checkpoint.enabled
    checkpoint_interval = config.checkpoint.save_interval

    if wall_clock_budget:
        # A yield that cannot save is just a crash with extra steps, and it
        # would requeue forever making no progress.
        if not (checkpoint_enabled and checkpoint_service):
            raise RuntimeError(
                "A wall-clock budget is declared but checkpointing is off "
                f"(checkpoint.enabled={checkpoint_enabled}). The run would yield "
                "at the wall, save nothing, requeue, and repeat forever. Enable "
                "checkpointing, or set NO_WALL_CLOCK=1 to train to the wall."
            )
        if logging_service:
            logging_service.log_info(
                f"[WallClock] Yielding in "
                f"{wall_clock_budget.seconds_remaining() / 3600:.2f}h "
                f"({wall_clock_budget.margin_s:.0f}s reserved for the final save; "
                f"checked every {_wall_clock_every} steps)."
            )

    # Initialize epoch *before* the loop so the post-loop final-checkpoint
    # block has a defined value even when the loop body never runs (e.g.
    # resume where start_iteration >= max_iterations).
    epoch = start_iteration // train_loader_len
    # Likewise seed ``iteration`` so the post-loop return can report how many
    # steps actually ran (throughput) even on an empty/early-stopped loop.
    iteration = start_iteration - 1

    # PIPE perf: gradient-accumulation + EMA cadence are INVARIANT across the
    # loop (frozen config + a strategy/step_executor built before the loop), so
    # resolve them ONCE here instead of re-``getattr``-ing every iteration.
    _gas = resolve_scheduler_cadence(strategy, config)
    _ema_cfg = getattr(config, "ema", None)
    _ema_update_freq = getattr(_ema_cfg, "update_frequency", 1)
    _ema_warmup = resolve_ema_warmup_gate(pipeline.ema, _ema_cfg)

    # #1353: the strategy lifecycle contract (``on_epoch_start`` /
    # ``on_epoch_end`` / ``on_validation_start`` / ``on_validation_end``) was
    # declared on BaseTrainingStrategy and overridden by real strategies, but
    # nothing under src/spectramr ever called it. This is the driver.
    lifecycle = StrategyLifecycleDriver(strategy)
    # The EPOCH pair is gated exactly as epoch-based validation is (~line 1941):
    #   * ``_has_train_loader`` -- ``train_loader_len`` falls back to 1 when the
    #     loader is missing/empty, which would make ``epoch == iteration`` and
    #     fire ``on_epoch_start`` every single step, while ``_is_epoch_boundary``
    #     stays permanently False and no ``on_epoch_end`` ever balances it.
    #   * ``not is_sanity_check`` -- these hooks mutate persistent strategy state
    #     (stage freezes, early-stopping patience counters). An overfit-one-batch
    #     sanity pass must not spend an arm's patience budget.
    # The VALIDATION pair is NOT gated: a sanity run really does validate, and a
    # hook that stays silent through a validation it did not prevent would be
    # lying about what the run did.
    _drive_epoch_hooks = _has_train_loader and not is_sanity_check
    # Metrics handed to ``on_epoch_end``. Rebound only by a validation that ran
    # this iteration, and cleared at the end of every iteration.
    _epoch_end_metrics: dict[str, Any] = {}

    for iteration in pbar:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch boundary: rebuild the iterator so the sampler
            # reshuffles. Notify DistributedSampler/etc. of the new
            # epoch so DDP ranks shuffle consistently. The old code
            # used itertools.cycle here which never re-invoked
            # __iter__ on the loader, silently freezing the shuffle
            # to whatever the first epoch produced.
            if not is_sanity_check:
                data_epoch += 1
                _set_sampler_epoch(train_loader, data_epoch)
            data_iter = iter(train_loader) if not is_sanity_check else data_iter
            batch = next(data_iter)
        epoch = iteration // train_loader_len

        # Update logging service SSOT state
        if logging_service:
            logging_service.set_step(iteration)
            logging_service.set_epoch(epoch)

        # WS-3 PR-3: advance the strategy's live-iteration seam. ``env.step`` is
        # frozen at 0 (TrainingEnvironment is frozen=True), so strategies read
        # ``self.loop_state.iteration`` for step-gated logic (e.g. the diffusion
        # curriculum diagnostic). Set here, before train_step, so the value the
        # strategy reads matches the ``iteration=`` it is also passed.
        loop_state = getattr(strategy, "loop_state", None)
        if loop_state is not None:
            loop_state.iteration = iteration
            loop_state.epoch = epoch
            # Loss-schedule: resolve clock curriculum (iteration/epoch triggers +
            # active ramps) into the per-step override map the loss computer reads.
            if loss_schedule_controller is not None:
                # DDP: rank 0 owns the schedule decision and broadcasts it so every
                # rank applies identical weights. This matters because loss_plateau
                # reads the per-rank training loss, so independent per-rank
                # evaluation could diverge; broadcasting rank 0's decision keeps all
                # ranks in lockstep. No-op on the single-process path. The map is
                # then published to the loss computer for EVERY paradigm via the
                # base-strategy seam, before train_step consumes it this iteration.
                if is_main_process:
                    _ls_decision = {
                        "overrides": loss_schedule_controller.on_iteration(iteration, epoch),
                        "checkpoint": loss_schedule_controller.consume_checkpoint_request(
                            iteration
                        ),
                    }
                else:
                    _ls_decision = None
                _ls_decision = RankUtility.broadcast_object(_ls_decision)
                loop_state.loss_weight_overrides = _ls_decision["overrides"]
                strategy.sync_scheduled_loss_weights()
                # Snapshot the pre-change model (throttled by the user-set
                # checkpoint_min_interval, decided in the controller).
                # `may_checkpoint`, NOT `is_main_process`: this builds a
                # checkpoint like every other site, so under FSDP/DeepSpeed it is
                # a COLLECTIVE and every rank must enter it or the job hangs.
                if _ls_decision["checkpoint"] and may_checkpoint:
                    _save_loss_schedule_checkpoint(
                        config=config,
                        pipeline=pipeline,
                        strategy=strategy,
                        output_paths=output_paths,
                        epoch=epoch,
                        iteration=iteration,
                        metrics=losses_history,
                        logging_service=logging_service,
                        parallel_runtime=parallel_runtime,
                    )

        # Strategy lifecycle: open the epoch. Placed AFTER the loop_state
        # update above so a hook reading ``self.loop_state.epoch`` sees the
        # epoch it was just handed, and BEFORE train_step so an unfreeze taken
        # here applies to the epoch's very first gradient step. Idempotent
        # within an epoch -- the driver compares an int and returns.
        if _drive_epoch_hooks:
            lifecycle.begin_epoch(epoch)

        try:
            # Prepare Batch (Use Adapter + Device Move)
            if not isinstance(batch, TrainingBatch):
                # Convert dict/other to TrainingBatch
                if isinstance(batch, dict):
                    batch = BatchAdapter.from_dict(batch, axes=batch_axes)
                else:
                    # Best effort: let strategy handle it or fail
                    pass

            # Move to device if it's a TrainingBatch (handles all internal tensors)
            if hasattr(batch, "to"):
                batch = batch.to(pipeline.device, non_blocking=True)
            elif isinstance(batch, dict):
                # Manual move for dicts
                batch = {
                    k: (
                        v.to(pipeline.device, non_blocking=True)
                        if isinstance(v, torch.Tensor)
                        else v
                    )
                    for k, v in batch.items()
                }

            # Strategy Step
            # Strategy returns list of OptimizationStepConfigs
            step_configs = strategy.train_step(
                batch,
                epoch,
                iteration=iteration,
                batch_data=batch,  # For legacy callbacks
                scaler=scaler,
            )

            # Let the step executor orchestrate the standard PyTorch boilerplate
            result = strategy.step_executor.execute_step(
                step_configs,
                epoch,
                global_step=iteration,
                model_type=model_type,
                clip_and_log_fn=getattr(strategy, "_clip_and_log_gradients", None),
            )

            # [FIX] Step LR schedulers (created by OptimizationBuilder but previously never stepped)
            # This is the root cause of late-stage training oscillation: constant LR with no decay.
            # NOTE: Skip scheduler step on first iteration to avoid PyTorch warning:
            # "Detected call of lr_scheduler.step() before optimizer.step()"
            # PyTorch >= 1.1 requires optimizer.step() to be called first, which happens
            # inside execute_step above, but the internal tracking needs one full cycle.
            # ``_gas`` hoisted above the loop (invariant).
            if pipeline.schedulers and _should_step_schedulers(iteration, start_iteration, _gas):
                for sched_name, sched in pipeline.schedulers.items():
                    try:
                        # ReduceLROnPlateau requires a validation metric and a
                        # per-validation cadence; stepping it here (no metric,
                        # every optimizer step) raised every call → a warning
                        # spam that masqueraded as a working scheduler while the
                        # LR stayed constant. Skip it in the per-iteration loop;
                        # plateau stepping belongs at validation time.
                        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            continue
                        if hasattr(sched, "step"):
                            # Re-sync base_lrs with the optimizer's param_groups
                            # first: a strategy may have added a group after the
                            # scheduler was built, which would otherwise make
                            # step() zip mismatched lengths and raise (F7d).
                            _resync_scheduler_base_lrs(sched)
                            sched.step()
                    except Exception as sched_err:
                        # RAISE, do not warn. A scheduler that cannot step will
                        # fail on EVERY iteration, so the warning is emitted
                        # thousands of times into a log nobody reads while the
                        # run trains at a constant LR -- exactly the "late-stage
                        # oscillation" this block was added to fix, silently
                        # reintroduced. The arm declared a schedule; a run that
                        # cannot honour it is not the experiment that was asked
                        # for, and its provenance would claim otherwise.
                        raise RuntimeError(
                            f"Scheduler step failed for {sched_name!r} at iteration "
                            f"{iteration}: {sched_err}. The declared LR schedule "
                            "cannot be honoured, so this run would train at a "
                            "constant learning rate while reporting the schedule "
                            "in its provenance."
                        ) from sched_err

            # [FIX] Update Exponential Moving Average (EMA) shadow weights.
            # Honor ema.update_frequency (previously an inert no-op, CLAUDE.md
            # pitfall #15): update only every Nth step. N=1 → every step.
            if pipeline.ema is not None:
                # EMA cadence (_ema_update_freq / _ema_warmup) hoisted above the
                # loop (frozen config → invariant).
                if ema_should_update(iteration, _ema_update_freq, _ema_warmup):
                    try:
                        # Unwrap: the shadow was deep-copied from the bare module
                        # at build time, while this generator may since have been
                        # wrapped by DDP / FSDP / DeepSpeed / torch.compile. The
                        # blend is key-matched, so a prefixed live model blends
                        # nothing and raises nothing (#2172).
                        pipeline.ema.update(unwrap_model(pipeline.generator))
                    except EMAKeyMismatchError:
                        # Never demote this one to a warning: it means EMA has
                        # been inert, which is the failure the guard exists to
                        # surface.
                        raise
                    except Exception as ema_err:
                        logger.warning(f"Failed to update EMA model weights: {ema_err}")

            # [STABILIZATION] Divergence (NaN/Inf) detection moved below: it
            # now piggybacks on the tqdm-postfix `.item()` sync instead of
            # paying a SECOND per-step GPU sync here (`torch.isfinite` on a
            # CUDA scalar forces a host round-trip). The old block also
            # or-chained the two result keys — tensor truthiness (itself a
            # sync) that made a legitimate 0.0 loss fall through to the
            # wrong key.

            # Fetch extra metric components (PSNR, SSIM, component losses)
            if hasattr(strategy, "get_last_metrics"):
                result.update(strategy.get_last_metrics())

            # Standardize Result - Keep as tensors to avoid GPU sync
            if isinstance(result, torch.Tensor):
                current_losses = {"g_total_loss": result.detach()}
            elif isinstance(result, dict):
                current_losses = {
                    k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in result.items()
                }
            else:
                current_losses = {}

            losses_history = current_losses

        except Exception as e:
            if logging_service:
                logging_service.log_error(f"Training Step Failed at iter {iteration}: {e}")
            else:
                logger.error(f"[Pipeline] Training Step Failed at iter {iteration}: {e}")
            logger.error("Training Step Failed", exc_info=True)
            raise

        # [FIX] Update tqdm every step with the cheap g_total_loss scalar.
        # Full scalar conversion + CSV write happen only every log_interval steps below.
        # None-safe .get() chain (no `or`-truthiness): a legitimate 0.0
        # g_total_loss must not fall through to "loss".
        _quick_loss = losses_history.get("g_total_loss", losses_history.get("loss"))
        if _quick_loss is not None:
            _quick_scalar = (
                _quick_loss.item() if isinstance(_quick_loss, torch.Tensor) else float(_quick_loss)
            )
            pbar.set_postfix({"loss": f"{_quick_scalar:.4f}"}, refresh=False)

            # [STABILIZATION] Divergence guard (NaN/Inf in loss): prevents
            # continuing training with corrupted weights (blacked-out images).
            # Runs every step but reuses the host float the tqdm postfix
            # already paid the GPU sync for — `math.isfinite` adds zero syncs.
            #
            # UNCONDITIONAL since 2026-07-24. It used to be gated on
            # ``optimization.gradient.detect_anomalies``, which bundled this free check
            # with ``torch.autograd.set_detect_anomaly(True)`` (global, 2-4x
            # slower — see StabilityManager). One knob for two wildly different
            # costs meant turning off the debug mode to recover throughput also
            # deleted the divergence tripwire, on exactly the cohorts whose
            # documented failure mode IS divergence. A non-finite loss must stop
            # training regardless of debug settings; the guard costs nothing.
            if not math.isfinite(_quick_scalar):
                if logging_service:
                    logging_service.log_critical(
                        f"❌ DIVERGENCE DETECTED at iteration {iteration}: "
                        f"Loss is {_quick_scalar}. Stopping training to "
                        f"prevent weight corruption."
                    )
                break

            # Sanity mode only: record the loss trend (and a phase-coherence
            # component if the paradigm tracks one) for the end-of-run
            # overfit verdict. The .item() sync here is acceptable because
            # sanity mode is a diagnostic, not a perf path; the guard keeps
            # it out of the normal training loop entirely.
            if is_sanity_check:
                sanity_loss_trace.append(_quick_scalar)
                for _mk, _mv in losses_history.items():
                    if "phase" in _mk.lower():
                        sanity_phase_trace.append(
                            _mv.item() if isinstance(_mv, torch.Tensor) else float(_mv)
                        )
                        break

        # Logging
        #
        # The periodic gate is NOT the whole condition. `logging.intervals.log`
        # defaults to 100 and arms set it as high as 5000, so any run shorter
        # than the cadence -- a smoke arm, a probe, a 40-step debug run, an
        # early-stopped sweep -- satisfied `iteration % log_interval == 0` zero
        # times and left `logs/training_metrics.csv` holding a header and no
        # rows, with the whole train scalar set suppressed alongside it, while
        # exiting successfully.
        #
        # The first and last iterations are therefore logged unconditionally, so
        # every run yields an interpretable curve. The widening is deliberately
        # bounded to two extra iterations rather than a coarser modulo, because
        # this gate is also the ONLY host transfer in the loop (#707) -- it
        # exists so `get_last_metrics` can return on-device tensors, and
        # non-negotiable 9 forbids a `.item()` per step. Two extra syncs per RUN
        # is not a hot-path cost; a smaller `log_interval` would be.
        is_first_iteration = iteration == first_iteration
        is_last_iteration = iteration == max_iterations

        # Rank 0 decides and BROADCASTS -- ranks that disagreed by one iteration
        # would deadlock, one entering the collective checkpoint save while the
        # others ran another step. The cadence is `_wall_clock_every`, resolved
        # above; it is deliberately not the logging gate's.
        if wall_clock_budget is not None and iteration % _wall_clock_every == 0:
            wall_clock_yield = bool(
                RankUtility.broadcast_object(
                    wall_clock_budget.expired() if is_main_process else None
                )
            )
            if wall_clock_yield:
                logger.info(
                    f"[WallClock] Allocation ends in "
                    f"{wall_clock_budget.deadline - time.time():.0f}s; yielding at "
                    f"iteration {iteration} of {max_iterations} after a final save."
                )
                if logging_service:
                    logging_service.log_info(
                        f"[WallClock] Yielding at iteration {iteration}/{max_iterations}; "
                        "resume with the same command to continue."
                    )

        if iteration % log_interval == 0 or is_first_iteration or is_last_iteration:
            # THE converter. `get_last_metrics` returns on-device tensors (#707)
            # precisely so this gate is the only host transfer, and this comment
            # has claimed "batched" since before anything batched: the dict
            # comprehension it replaces paid one `.item()` PER TENSOR, and each
            # sync drains the queue, so metric k+1 could not start launching
            # while k was still in flight.
            #
            # Real scalars go in one fused transfer; anything else keeps the
            # per-item path so this stays value-identical to the comprehension
            # (a multi-element or complex tensor raised there and must still
            # raise here rather than being silently reduced).
            _fusable = {
                k: v
                for k, v in losses_history.items()
                if isinstance(v, torch.Tensor) and v.numel() == 1 and not torch.is_complex(v)
            }
            losses_scalar = {
                k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in losses_history.items()
                if k not in _fusable
            }
            if _fusable:
                losses_scalar.update(
                    zip(_fusable, fuse_to_host(list(_fusable.values())), strict=True)
                )
            # Restore the declaration order the CSV header was built from; the
            # split above would otherwise emit fused keys last.
            losses_scalar = {k: losses_scalar[k] for k in losses_history}

            # [FIX] Log current learning rate(s) from schedulers
            if pipeline.schedulers:
                for sched_name, sched in pipeline.schedulers.items():
                    current_lr, lr_missing_reason = read_scheduler_lr(sched)
                    if current_lr is not None:
                        losses_scalar[lr_column_name(sched_name)] = current_lr
                    elif sched_name not in csv_empty_lr_warned:
                        # Warn ONCE per scheduler, mirroring the discarded-key
                        # set below -- this is per-row code, and the condition
                        # is a property of the scheduler, so it would otherwise
                        # repeat every iteration for the whole run.
                        csv_empty_lr_warned.add(sched_name)
                        logger.warning(
                            "[Pipeline] Scheduler %r yields no learning rate, so the "
                            "promised CSV column %r will stay empty for this run: %s. "
                            "The header names a column for every entry of "
                            "pipeline.schedulers, so an empty one means the emitter "
                            "and the header resolver disagree.",
                            sched_name,
                            lr_column_name(sched_name),
                            lr_missing_reason,
                        )

            pbar.set_postfix(
                {
                    k: f"{v:.4f}"
                    for k, v in losses_scalar.items()
                    if "loss" in k.lower() or "psnr" in k.lower()
                }
            )

            # CSV Logging - append to existing file with initialized fieldnames
            if csv_file is not None and csv_fieldnames is not None:
                csv_data = {"iteration": iteration, "epoch": epoch, **losses_scalar}

                # DEBUG: Log losses being written on first iteration. The
                # earlier prototype emitted a [CSV] WARNING for every
                # enabled-but-not-computed loss (see
                # config_health_checker.py docstring around the
                # ``[CSV] WARNING: Loss ... is in CSV fieldnames but NOT
                # in losses_scalar dict`` warnings). That warning was
                # noisy and false-positive: the row writer below uses
                # ``extrasaction="ignore"`` and missing columns become
                # empty strings, which is the intended representation
                # of "not computed this iteration". Keep the debug
                # breadcrumb but drop the per-loss warning.
                if iteration <= 100 and logging_service is not None:
                    logging_service.log_debug(
                        f"[CSV] Iteration {iteration}: "
                        f"losses_dict_keys={list(losses_scalar.keys())}, "
                        f"csv_fieldnames={len(csv_fieldnames)} fields"
                    )

                # `extrasaction="ignore"` silently DROPS any key with no
                # column, so a divergence between the header resolver and what
                # the loop actually computes costs real measured data with no
                # trace. `_csv_metric_names` used to union both selection sources
                # to make that structurally impossible; it now mirrors the
                # producer's precedence exactly, and this check is what keeps the
                # discard direction loud instead of silent.
                #
                # NOT the warning that was removed above: that one fired on
                # columns with no VALUE (the empty-string direction, which is the
                # intended representation of "not computed this iteration") and
                # was rightly dropped as a false positive. This one fires on
                # VALUES WITH NO COLUMN, which is unrecoverable data loss.
                discarded = set(csv_data) - set(csv_fieldnames)
                new_discarded = discarded - csv_discarded_keys_warned
                if new_discarded:
                    csv_discarded_keys_warned |= new_discarded
                    # Classify, do NOT suppress. Every discarded key is still
                    # named and still warned about, because every one of them IS
                    # being thrown away -- that statement stays true for the
                    # executor aliases too. What the split adds is the ability
                    # to tell a DECIDED exclusion from a resolver disagreement,
                    # which is the difference between a known cost and a bug. An
                    # allowlist that silenced them would instead become a second,
                    # unaudited owner of "which keys are known duplicates".
                    known_aliases, unexplained = classify_discarded_keys(new_discarded)
                    logger.warning(
                        "[Pipeline] %d computed value(s) have no column in "
                        "%s and are being DISCARDED: %s. The header is built "
                        "from the config at startup, so this means the header "
                        "resolver and the loop disagree -- the value is measured "
                        "and then thrown away. Columns present: %d."
                        "%s",
                        len(new_discarded),
                        getattr(csv_file, "name", csv_file),
                        sorted(new_discarded),
                        len(csv_fieldnames),
                        (
                            f" Of these, {known_aliases} are known step-executor "
                            f"aliases of a strategy-stamped total that IS in the "
                            f"header (no data is lost; excluded on purpose -- see "
                            f"_EXECUTOR_ALIAS_KEYS), leaving {unexplained} "
                            f"unexplained."
                            if known_aliases
                            else ""
                        ),
                    )

                try:
                    # Simply append the row with the pre-initialized fieldnames
                    # No dynamic header rewriting - this prevents duplication
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=csv_fieldnames, extrasaction="ignore")
                        writer.writerow(csv_data)

                except Exception as e:
                    if logging_service:
                        logging_service.log_warning(f"CSV write failed: {e}")

            # TensorBoard
            if tb_writer:
                tb_writer.scalars(losses_scalar, iteration, "train")
                # Every loss term on ONE axis. Separate charts cannot answer
                # "is the adversarial term drowning L1", which is the question
                # a multi-term objective is actually read for -- and the shape
                # a dead or warmup-gated term makes obvious.
                tb_writer.grouped_scalars("train/losses", losses_scalar, iteration)
                # Weight/gradient DISTRIBUTIONS, on `logging.intervals.histogram`
                # (default every 1000 steps). A collapsed layer and a saturated
                # one can share a gradient norm but never share a shape, which
                # is what makes this readable where the scalar is not (#20).
                #
                # The writer checks the cadence BEFORE touching a parameter:
                # `add_histogram` copies each one to the host, so an ungated call
                # would be a GPU sync per parameter per step (non-negotiable #9).
                _hist_model = pipeline.models.get("generator")
                if _hist_model is not None:
                    tb_writer.histograms(_hist_model, iteration, "train")
                # Flush to ensure metrics are written promptly
                if iteration % 100 == 0:
                    tb_writer.flush()

        # Validation Trigger Logic
        # 1. Standard interval check
        time_for_eval = iteration % eval_interval == 0

        # 2. Epoch boundary check (if enabled)
        # Reuse the loader length + presence flag bound once before the loop
        # (train_loader_len / _has_train_loader, ~line 956) instead of
        # re-fetching the loader and recomputing len() every iteration. The
        # _has_train_loader guard preserves the original semantics: a
        # missing/empty loader is NOT an epoch boundary (train_loader_len's
        # fallback of 1 would otherwise make every step one).
        # See backlog_wasted_compute_audit_2026_05_29 PIPE-1.
        is_epoch_end = _is_epoch_boundary(iteration, train_loader_len, _has_train_loader)

        # Epoch-based validation. Skipped during sanity checks, which must obey
        # `eval_interval` strictly.
        #
        # The `and eval_interval <= 0` that used to gate this was UNREACHABLE
        # (#711): `eval_interval` is `validation.schedule.interval_steps`, which
        # the schema declares `ge=1`, or `max_iterations` — never <= 0. So the
        # knob could not fire on any config the schema admits, while defaulting
        # to True and reading as enabled on every arm.
        #
        # It is ADDITIVE, matching what the field says ("Run validation at the
        # end of each epoch"): an epoch boundary adds a validation event, it does
        # not replace the step interval. Additive is also the safe direction --
        # it can only add validation events, never remove the one an arm selects
        # its checkpoint from.
        if (
            eval_on_epoch
            and is_epoch_end
            and not is_sanity_check
            and epoch % eval_interval_epochs == 0
        ):
            time_for_eval = True

        # Will this validation finish before the wall?
        #
        # The wall-clock check ~200 lines above runs BETWEEN iterations, and
        # validation is inside one, so a deadline landing mid-pass cannot be
        # acted on: the job is killed with no checkpoint, no marker and no
        # requeue, and the chain stops looking like an ordinary TIMEOUT. That is
        # not a corner -- `experiment_11_attention_none` records one validation
        # event at 6.4h and validation at 41% of its wall clock.
        #
        # Entry is gated only on values every rank computes identically, and the
        # decision is broadcast, because a rank that skipped while another
        # validated would deadlock on the collective save below.
        if wall_clock_budget is not None and time_for_eval and not wall_clock_yield:
            _fits = None
            if is_main_process:
                _fits = bool(
                    _last_val_seconds
                    and wall_clock_budget.seconds_remaining()
                    < _last_val_seconds * VALIDATION_SAFETY_FACTOR
                )
            if bool(RankUtility.broadcast_object(_fits)):
                wall_clock_yield = True
                logger.info(
                    f"[WallClock] Skipping validation at iteration {iteration}: the "
                    f"last pass took {_last_val_seconds:.0f}s and only "
                    f"{wall_clock_budget.seconds_remaining():.0f}s remain before the "
                    "save margin. Yielding instead."
                )
                if logging_service:
                    logging_service.log_info(
                        f"[WallClock] Validation skipped at iteration {iteration} "
                        "(would not finish before the wall); yielding."
                    )

        # Execute Validation
        #
        # `not wall_clock_yield`: a validation event that starts after the yield
        # decision eats the margin reserved for writing the checkpoint. The gate
        # is on the EXECUTION, not on `time_for_eval` -- the epoch-boundary
        # branch above re-sets that flag after the interval branch computes it.
        _val_started = time.monotonic() if time_for_eval else None
        _validation_ran = False
        if pipeline.data_loaders.get("val") and time_for_eval and not wall_clock_yield:
            _validation_ran = True
            pipeline.models.get("generator").eval()
            # Schedule-free optimizers keep an averaged sequence separate from
            # the iterate the gradient is taken at. Validating without swapping
            # to it measures the wrong point in weight space -- which reads as a
            # merely-worse arm, not as a bug. Duck-typed so it also covers a
            # user-supplied schedule-free implementation.
            _set_optimizer_eval_mode(pipeline, train=False)
            lifecycle.begin_validation()
            val_metrics = _run_validation(
                pipeline,
                strategy,
                iteration,
                epoch,
                logging_service,
                output_paths,
                metrics_service=metrics_service,
                tb_writer=tb_writer,  # Pass TensorBoard writer for validation logging
                csv_file=csv_file,
            )
            lifecycle.end_validation(val_metrics)
            _epoch_end_metrics = val_metrics

            # Drain the tall cascading sweep the strategy just published (#697).
            # The strategy computes, the pipeline persists: `IMetricsService`
            # declares no CSV surface, so the strategy writing this itself would
            # reach past its own interface. `_drain_cascade_rows` is a no-op for
            # every paradigm that does not run a cascade.
            _drain_cascade_rows(
                strategy,
                metrics_service,
                iteration=iteration,
                epoch=epoch,
                logging_service=logging_service,
                is_main_process=is_main_process,
            )

            # Restore train mode unless we're strictly enforcing eval mode for a sanity check
            if not is_sanity_check:
                pipeline.models.get("generator").train()
            # Swap the schedule-free optimizer back to its gradient iterate.
            # Unconditional, including under is_sanity_check: leaving the
            # optimizer in eval mode would take every subsequent gradient at the
            # averaged weights, which is a different algorithm.
            _set_optimizer_eval_mode(pipeline, train=True)

            # Loss-schedule: metric/loss plateau + threshold triggers and any
            # post-change monitoring windows (these only have a metric to read at
            # validation cadence). Updates the override map the computer reads.
            if loss_schedule_controller is not None and loop_state is not None:
                # DDP: rank 0 decides + broadcasts (val_metrics is already
                # all-reduced and identical across ranks, but losses_history for
                # loss_plateau is per-rank, so we keep the SAME single-decision
                # path as on_iteration). No-op single-process.
                if is_main_process:
                    _ls_decision = {
                        "overrides": loss_schedule_controller.on_validation(
                            iteration, epoch, val_metrics, losses_history
                        ),
                        "checkpoint": loss_schedule_controller.consume_checkpoint_request(
                            iteration
                        ),
                    }
                else:
                    _ls_decision = None
                _ls_decision = RankUtility.broadcast_object(_ls_decision)
                loop_state.loss_weight_overrides = _ls_decision["overrides"]
                strategy.sync_scheduled_loss_weights()
                # Snapshot before a plateau/threshold-triggered change proceeds to
                # the next iteration (throttled by checkpoint_min_interval).
                # `may_checkpoint`, NOT `is_main_process` -- see the sibling site
                # above: a rank-0-gated collective hangs the job.
                if _ls_decision["checkpoint"] and may_checkpoint:
                    _save_loss_schedule_checkpoint(
                        config=config,
                        pipeline=pipeline,
                        strategy=strategy,
                        output_paths=output_paths,
                        epoch=epoch,
                        iteration=iteration,
                        metrics=losses_history,
                        logging_service=logging_service,
                        parallel_runtime=parallel_runtime,
                    )

            # Early Stopping Integration (must be inside the if-branch where val_metrics is defined)
            if early_stopping_service:
                monitor_key = early_stopping_service.monitor  # e.g. "val_psnr"
                # F-EARLYSTOP-PREFIX / 2026-05-20 — YAMLs use ``val_*``
                # by convention (mirroring TensorBoard's split namespace)
                # but the validator returns unprefixed metric names
                # (``psnr``, ``ssim``, ...). Without this aliasing, every
                # early-stopping check silently misses and the training
                # runs to ``max_iter`` instead of stopping on plateau.
                # Surfaced by the 2026-05-19 cluster smoke (~14
                # "monitor metric not found" warnings).
                resolved_monitor_key: str | None = None
                # Ordered candidate keys for the configured monitor. The
                # alias rules (val_-prefix, loss-key aliases incl. the
                # val_mse min-mode proxy, cascade-suffix add + strip) are
                # extracted into a pure, unit-tested helper. See
                # ``early_stop_monitor_candidates`` and the F-EARLYSTOP-*
                # entries in docs/smoke_audit_20260521_fixes.rst.
                candidates = early_stop_monitor_candidates(monitor_key)
                for cand in candidates:
                    if cand in val_metrics:
                        resolved_monitor_key = cand
                        break
                if resolved_monitor_key is not None:
                    monitor_key = resolved_monitor_key  # used below for save_best
                    monitor_value = float(val_metrics[monitor_key])
                    early_stopping_service.update(monitor_value, iteration)
                    if resolved_monitor_key != early_stopping_service.monitor and logging_service:
                        # Once-per-run note so the user sees which key was actually used.
                        if not getattr(early_stopping_service, "_alias_notice_emitted", False):
                            logging_service.log_info(
                                f"[EarlyStopping] monitor='{early_stopping_service.monitor}' "
                                f"resolved to val_metrics key '{resolved_monitor_key}' "
                                f"(via F-EARLYSTOP-FUZZY alias)."
                            )
                            early_stopping_service._alias_notice_emitted = True

                    # A non-finite monitor value is not an improvement (#181).
                    # ``EarlyStoppingService.update`` returns EARLY on non-finite,
                    # leaving ``wait_count`` at its PRIOR value -- so if the previous
                    # check improved, ``wait_count == 0`` still holds and this block
                    # would overwrite checkpoint_best.pt with a NaN score.
                    # ``CheckpointDirector`` formats it with ``f"{v:.6f}"``, which
                    # renders `nan` without raising, so nothing downstream noticed.
                    monitor_is_finite = math.isfinite(monitor_value)
                    if not monitor_is_finite and logging_service:
                        logging_service.log_warning(
                            f"[EarlyStopping] monitor '{monitor_key}' is "
                            f"{monitor_value} at iter {iteration}: not counted as an "
                            "improvement, and checkpoint_best.pt is left untouched."
                        )

                    # ...and a monitor that is non-finite EVERY time costs both
                    # mechanisms, exactly as an unresolvable key does (#178). The
                    # branch above disarms `save_best`; `EarlyStoppingService.update`
                    # returns early on non-finite, so `wait_count` never moves and
                    # the stop can never fire either. The run then burns its full
                    # budget and exits 0 with no `checkpoint_best.pt`, which is the
                    # #178 failure entered through a VALUE instead of a key.
                    #
                    # #178 can raise on the first event because keys are stable.
                    # Values are not, so a transient NaN must be tolerated. The
                    # threshold is `patience` -- the run's own declared tolerance
                    # for "no progress", so no new knob is invented: had these
                    # values been finite and merely bad, early stopping would have
                    # stopped the run by now. Instead it is frozen.
                    if monitor_is_finite:
                        consecutive_nonfinite_monitor = 0
                    else:
                        consecutive_nonfinite_monitor += 1
                        patience = getattr(early_stopping_service, "patience", 0) or 0
                        if (
                            early_stopping_service.enabled
                            and patience > 0
                            and consecutive_nonfinite_monitor >= patience
                        ):
                            na_reason = _monitor_not_applicable_reason(strategy, monitor_key)
                            raise RuntimeError(
                                f"Early-stopping monitor '{monitor_key}' has been "
                                f"non-finite for {consecutive_nonfinite_monitor} "
                                f"consecutive validation events (patience="
                                f"{patience}), most recently {monitor_value} at "
                                f"iteration {iteration}.\n"
                                f"  {na_reason}\n"
                                "Both selection mechanisms are dead in this state: "
                                "no checkpoint_best.pt can be written and early "
                                "stopping can never trigger, so continuing would "
                                "burn the remaining budget for a run that cannot "
                                "produce a selected model.\n"
                                "  available: "
                                f"{sorted(k for k, v in val_metrics.items() if isinstance(v, (int, float)) and math.isfinite(float(v)))}"
                            )

                    # Save best checkpoint when metric improves (wait_count resets to 0)
                    if (
                        early_stopping_service.wait_count == 0
                        and checkpoint_enabled
                        # rank 0 writes best.pt -- but FSDP/DeepSpeed need every
                        # rank here, or rank 0 deadlocks in the gather.
                        and may_checkpoint
                        and monitor_is_finite
                    ):
                        try:
                            checkpoint_dir = str(
                                Path(output_paths.get("run_output_dir", "experiments/outputs"))
                                / "checkpoints"
                            )
                            best_director = CheckpointDirector(config)
                            best_path = (
                                best_director.with_checkpoint_dir(checkpoint_dir)
                                .with_pipeline(pipeline)
                                .with_strategy(strategy)
                                .with_epoch(epoch)
                                .with_global_step(iteration)
                                .with_metrics(losses_history)
                                .with_scaler(getattr(pipeline, "scaler", None))
                                .with_parallel_runtime(parallel_runtime)
                                .with_counter_state(
                                    {
                                        "current_step": iteration,
                                        "current_epoch": epoch,
                                    }
                                )
                                .validate()
                                .save_best(
                                    metric_name=monitor_key,
                                    metric_value=val_metrics[monitor_key],
                                )
                            )
                            best_checkpoint_path = str(best_path)
                            if logging_service:
                                logging_service.log_info(
                                    f"[EarlyStopping] Best checkpoint saved at iter {iteration}: "
                                    f"{monitor_key}={val_metrics[monitor_key]:.6f} → {best_path}"
                                )
                        except Exception as best_err:
                            # Recorded, not just logged (#713). A failed best-save
                            # leaves the run reporting success with no
                            # checkpoint_best.pt, and a warning in a log nobody
                            # re-reads is indistinguishable from "the metric never
                            # improved". Stamped into run_summary so the absence is
                            # attributable after the fact.
                            best_checkpoint_error = (
                                f"iter {iteration}: {type(best_err).__name__}: {best_err}"
                            )
                            logger.warning(
                                f"[EarlyStopping] Failed to save best checkpoint: {best_err}"
                            )

                    if early_stopping_service.should_stop():
                        logger.info(
                            f"[EarlyStopping] Triggered at iteration {iteration}. Stopping training."
                        )
                        if logging_service:
                            logging_service.log_info(
                                f"Early Stopping triggered at iter {iteration}"
                            )
                        break
                else:
                    # FATAL, not a warning (#178). An unresolvable monitor costs
                    # BOTH mechanisms at once: early stopping never fires (the run
                    # burns its full budget -- ~23 h of GPU on the arms this was
                    # found on) and `checkpoint_best.pt` is never written, because
                    # the save_best block above lives inside the resolved branch.
                    # Neither failure is visible in any artifact: the run exits 0,
                    # `success: true`, with a warning in a log nobody re-reads.
                    #
                    # This fires on the FIRST validation event, so the cost of a
                    # typo'd monitor is one validation pass rather than the whole
                    # budget. Validation-metric keys are stable across events, so
                    # there is nothing to wait for -- a key absent now is absent
                    # for the rest of the run.
                    # The message is built in `_unresolvable_monitor_error`,
                    # which decides whether the monitor is genuinely absent or
                    # whether validation FAILED and left only its sentinel behind.
                    # Fatal either way, for the #178 reason above; only the
                    # ATTRIBUTION differs, and getting it wrong sent the user to
                    # edit a correct YAML.
                    raise RuntimeError(
                        _unresolvable_monitor_error(
                            early_stopping_service.monitor, val_metrics, iteration
                        )
                    )
        elif time_for_eval and not pipeline.data_loaders.get("val"):
            if logging_service:
                logging_service.log_warning(
                    f"[Pipeline] Skipping validation at iter {iteration}: No 'val' dataloader found in pipeline."
                )
        elif time_for_eval and wall_clock_yield and logging_service:
            # Said out loud: a validation event that vanishes without a line is
            # an unexplained gap in the arm's record.
            logging_service.log_warning(
                f"[WallClock] Validation at iter {iteration} skipped -- the run is "
                "yielding at the wall clock and resumes from the checkpoint below."
            )

        if _validation_ran and _val_started is not None:
            _last_val_seconds = time.monotonic() - _val_started

        # Strategy lifecycle: close the epoch that just COMPLETED (``epoch - 1``
        # at this point -- see StrategyLifecycleDriver's module docstring for why
        # ``on_epoch_start(N + 1)`` necessarily precedes ``on_epoch_end(N)`` here).
        # Placed after the validation block so per-stage early stopping scores the
        # epoch on the boundary validation's metrics rather than on a mid-epoch
        # measurement. ``_epoch_end_metrics`` is rebound ONLY when validation
        # actually ran this iteration; a stale dict would let one epoch's numbers
        # be counted twice against a patience counter.
        if _drive_epoch_hooks and is_epoch_end:
            lifecycle.end_epoch(_epoch_end_metrics)
        _epoch_end_metrics = {}

        # Checkpoint Saving
        # FIX #4: checkpoint_enabled and checkpoint_interval already loaded above (line ~330)
        # No need to re-fetch or use getattr() fallbacks

        if (
            checkpoint_enabled
            and checkpoint_service
            # `or wall_clock_yield`: the yield is worthless without the save it
            # exists to make room for.
            and (iteration % checkpoint_interval == 0 or wall_clock_yield)
            # rank 0 writes the shared checkpoint; collective strategies need all.
            and may_checkpoint
        ):
            try:
                # =====================================================================
                # PHASE 3 TASK 3: Checkpoint via Director (Atomic Save)
                # =====================================================================
                # Phase 3 Task 3: Integrated CheckpointDirector for unified checkpoint
                # management. Director automatically handles all model states, optimizer
                # states (including multi-optimizer GANs), scheduler states, and metrics.
                #
                # Key Benefits:
                # - Single method saves all state atomically
                # - Automatic multi-optimizer GAN support (no manual collection)
                # - Scheduler state recovered on resume
                # - Metrics persisted
                # - Better error handling

                # Use CheckpointDirector for atomic save (Phase 3)
                try:
                    checkpoint_dir = str(
                        Path(output_paths.get("run_output_dir", "experiments/outputs"))
                        / "checkpoints"
                    )

                    director = CheckpointDirector(config)
                    checkpoint_path = (
                        director.with_checkpoint_dir(checkpoint_dir)
                        .with_pipeline(pipeline)
                        .with_strategy(strategy)
                        .with_epoch(epoch)
                        .with_global_step(iteration)
                        .with_metrics(losses_history)
                        .with_scaler(getattr(pipeline, "scaler", None))
                        .with_parallel_runtime(parallel_runtime)
                        .with_counter_state(
                            {
                                "current_step": iteration,
                                "current_epoch": epoch,
                            }
                        )
                        .validate()
                        .save()
                    )

                    if logging_service:
                        logging_service.log_info(
                            f"Checkpoint via director at iter {iteration}: {checkpoint_path}"
                        )
                except Exception as director_err:
                    # If director fails, always try fallback
                    logger.warning(
                        f"[Pipeline] CheckpointDirector failed: {director_err}. "
                        "Attempting fallback..."
                    )
                    try:
                        optimizers = getattr(pipeline, "optimizers", None)
                        main_optimizer = optimizers.get("opt_g") if optimizers else None

                        extra_optimizers = {}
                        if optimizers:
                            for key, opt in optimizers.items():
                                if key != "opt_g":
                                    extra_optimizers[key] = opt

                        checkpoint_service.save_checkpoint(
                            model=pipeline.models.get("generator"),
                            optimizer=main_optimizer,
                            epoch=epoch,
                            loss=losses_history.get("g_total_loss", 0.0),
                            step=iteration,
                            extra_optimizers=(extra_optimizers if extra_optimizers else None),
                            # Strategy-owned learnable state (section R) so the
                            # fallback path doesn't silently drop it either.
                            strategy_state=(
                                strategy.strategy_state_dict()
                                if hasattr(strategy, "strategy_state_dict")
                                else None
                            ),
                        )
                        if logging_service:
                            logging_service.log_info(
                                f"Checkpoint (legacy fallback) at iter {iteration}"
                            )
                    except Exception as fallback_err:
                        if logging_service:
                            logging_service.log_warning(
                                f"Both director and fallback failed: {fallback_err}"
                            )
            except Exception as e:
                if logging_service:
                    logging_service.log_warning(f"Checkpoint save failed at iter {iteration}: {e}")

        if wall_clock_yield:
            break

    # DDP: only rank 0 wrote best.pt (and thus holds its path); broadcast it so
    # EVERY rank restores the SAME best weights below, keeping the final model
    # identical across ranks (otherwise non-main ranks would keep their latest
    # weights). No-op single-process; the broadcast is also a rendezvous, so the
    # file rank 0 wrote during the loop is visible to all ranks before they read.
    best_checkpoint_path = RankUtility.broadcast_object(best_checkpoint_path)

    # Honor early_stopping.restore_best_weights (CLAUDE.md pitfall #15): after
    # the loop, reload the best checkpoint so the returned/exported model is the
    # best one, not the latest (often-degraded) weights. Runs before the final
    # checkpoint so that, too, reflects the restored weights.
    if (
        config.early_stopping
        and getattr(config.early_stopping, "restore_best_weights", False)
        and best_checkpoint_path
        # A yielded run is mid-training: swapping in best weights here would
        # hand the next chain link a model that is not where the loop left off.
        and not wall_clock_yield
    ):
        try:
            # with_parallel_runtime is what makes this symmetric with the WRITER
            # at :1897. Without it the director resolves DefaultCheckpointAdapter
            # (writes_native_artifact=False), so a sharded strategy's best
            # checkpoint -- for DeepSpeed a bare consolidated state_dict, because
            # save_best skips the generic payload whenever the adapter wrote a
            # native artifact -- was parsed as a generic payload and raised
            # KeyError('generator'), discarding the best weights of a finished run.
            CheckpointDirector(config).with_pipeline(pipeline).with_strategy(
                strategy
            ).with_parallel_runtime(parallel_runtime).load_from(best_checkpoint_path)
            logger.info(f"[EarlyStopping] Restored best weights from {best_checkpoint_path}")
        except Exception as restore_err:
            logger.warning(
                f"[EarlyStopping] Failed to restore best weights from "
                f"{best_checkpoint_path}: {restore_err}"
            )

    # Final checkpoint on completion (rank 0 writes the shared output dir;
    # FSDP/DeepSpeed need every rank here or the gather deadlocks).
    #
    # `not wall_clock_yield` is load-bearing rather than an optimisation: this
    # save stamps `global_step=max_iterations` unconditionally, so on a yield at
    # iteration 40k of 200k it would assert the run had finished and the next
    # link of the chain would resume at 200k and train nothing. The yield wrote
    # its own correctly-stamped checkpoint inside the loop.
    if checkpoint_enabled and checkpoint_service and may_checkpoint and not wall_clock_yield:
        try:
            # =====================================================================
            # PHASE 3 TASK 3: Final Checkpoint via Director
            # =====================================================================
            checkpoint_dir = str(
                Path(output_paths.get("run_output_dir", "experiments/outputs")) / "checkpoints"
            )

            director = CheckpointDirector(config)
            checkpoint_path = (
                director.with_checkpoint_dir(checkpoint_dir)
                .with_pipeline(pipeline)
                .with_epoch(epoch)
                .with_global_step(max_iterations)
                .with_metrics(losses_history)
                .with_scaler(getattr(pipeline, "scaler", None))
                .with_parallel_runtime(parallel_runtime)
                .with_counter_state(
                    {
                        "current_step": max_iterations,
                        "current_epoch": epoch,
                    }
                )
                .validate()
                .save()
            )

            if logging_service:
                logging_service.log_info(
                    f"Final checkpoint via director at iter {max_iterations}: {checkpoint_path}"
                )
        except Exception as e:
            if logging_service:
                logging_service.log_warning(f"Final checkpoint save failed: {e}")

    # Reporting is NOT invoked here. `MetricsReportGenerator` used to run from
    # this spot on every run, ungated -- while the canonical `generate_report`
    # sat behind `reporting.enabled`, which defaults False. The legacy generator
    # therefore ran always and the SSOT pipeline almost never did.
    #
    # `pipelines/train.py::_maybe_run_reporting` is the one end-of-training
    # report path now: the full pipeline when `reporting.enabled` is set, a
    # tables-only CSV floor otherwise. It fires after this function returns, so
    # it sees `final_metrics.json` written below -- which this call site did not.
    # The class itself stays: `scripts/render_full_reporting_pipeline.py` builds
    # its auto-report section from it.

    # =====================================================================
    # Write per-arm final_metrics.json — campaign-aggregator contract.
    # =====================================================================
    # Hoisted out of the `try` so the RETURN below can carry it (#481). This
    # payload has always been assembled correctly and then written to disk and
    # dropped on the floor: `train.py` reads `result.get("best_metrics")` for
    # `run_summary.json`, and no return path in this function ever set that key,
    # so `run_summary.best_metrics` was structurally `null` on every run ever.
    best_metrics: dict[str, float] = {}
    try:
        if output_paths and output_paths.get("run_output_dir") and is_main_process:
            run_dir = Path(output_paths["run_output_dir"])
            csv_path = output_paths.get("csv_log_file")
            # [#586] Window the CSV to THIS run. ``training_metrics.csv`` is
            # appended to by every run writing into this output dir, so an
            # unwindowed pass reports whichever run happened to score best.
            # ``iteration`` is the last iteration this run reached.
            best_metrics = (
                _summarise_best_metrics_from_csv(csv_path, final_iteration=iteration)
                if csv_path
                else {}
            )
            # `_summarise_best_metrics_from_csv` now folds the validation CSV in
            # too, so `best` finally carries the `val_*` keys the early-stopping
            # monitor was already selecting on (#481).
            _meta = getattr(config, "metadata", None)
            payload = {
                "schema_version": "1",
                "experiment_name": getattr(_meta, "name", None),
                "best": best_metrics,
                "final_loss": float(losses_history.get("g_total_loss", 0.0) or 0.0),
                "early_stopping_best_value": _extract_es_best(early_stopping_service),
                "early_stopping_best_iteration": _extract_es_best_iter(early_stopping_service),
                # Provenance (#15): every loss-schedule fire / rollback, so a
                # dynamic loss curriculum is auditable rather than inferred.
                "loss_schedule_events": (
                    list(loss_schedule_controller.events)
                    if loss_schedule_controller is not None
                    else []
                ),
                "csv_log_file": str(csv_path) if csv_path else None,
                # None on a healthy run. Non-null means checkpoint_best.pt is
                # MISSING and why -- previously a log warning only, which reads
                # identically to "the metric never improved" (#713).
                "best_checkpoint_error": best_checkpoint_error,
            }
            target = run_dir / "final_metrics.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, indent=2, default=str))
            if logging_service:
                logging_service.log_info(f"✓ Wrote final_metrics.json: {target}")
    except Exception as e:
        # FATAL (#713). `final_metrics.json` is the per-arm contract the campaign
        # aggregator reads; a run that trained fine but wrote no summary is
        # indistinguishable downstream from a run that produced nothing, and the
        # only trace was a warning in a log nobody re-reads. Everything expensive
        # is already done at this point, so failing here costs no GPU time -- it
        # just refuses to call an unreportable run a success.
        if logging_service:
            logging_service.log_error(f"Failed to write final_metrics.json: {e}")
        raise RuntimeError(
            f"Training completed but final_metrics.json could not be written to "
            f"{output_paths.get('run_output_dir') if output_paths else '<unset>'}: {e}. "
            "The run's headline artifact is missing, so its results are not "
            "reportable; refusing to exit successfully."
        ) from e

    # Post-training certification hook: conformal / calibration strategies
    # compute their certificate (coverage_at_alpha, exchangeability p-value)
    # AFTER the reconstructor is trained. No-op for every other strategy.
    #
    # `not wall_clock_yield`: a certificate computed from a half-trained
    # reconstructor is a certificate for a model that will not exist by the end
    # of the chain -- and this runs inside the save margin, so on a yield it
    # eats the time reserved for getting out cleanly.
    if not is_sanity_check and not wall_clock_yield:
        _maybe_run_calibration(strategy, pipeline, output_paths, logging_service)

    # Sanity-check verdict: a green "success" is only honest if the model
    # actually learned. Overfitting one batch MUST collapse the loss; if it
    # plateaus the model cannot fit the target (broken gradient/loss/wiring,
    # or an architecturally unlearnable target such as k-space phase — the
    # experiment_11 "DC blob"). Return success=False so main.py's fail-fast
    # (and the smoke gate) catch it instead of a misleading pass.
    if is_sanity_check:
        passed, report = _evaluate_sanity_overfit(sanity_loss_trace, sanity_phase_trace)
        if logging_service:
            logging_service.log_info(
                f"🧪 [SANITY VERDICT] {'PASS' if passed else 'FAIL'}: {report}"
            )
        return {
            "success": bool(passed),
            "sanity_check": True,
            "sanity_report": report,
            "final_loss": report.get("final_loss", losses_history.get("g_total_loss", 0.0)),
            "training_time": "N/A",  # overwritten by run_training_pipeline
            "iterations_completed": max(0, iteration - start_iteration + 1),
            # `train.py` reads this for run_summary.json (#481). Empty rather than
            # absent: a sanity check is too short to have a meaningful best.
            "best_metrics": best_metrics,
            **({} if passed else {"error": report.get("reason", "sanity overfit failed")}),
        }

    return {
        "success": True,
        # A yielded run succeeded at what it was asked to do and is NOT finished.
        # Callers that treat `success` as "training is done" would archive a run
        # that is half-trained, so the distinction is a key, not a log line.
        "wall_clock_yield": wall_clock_yield,
        "resume_from_iteration": iteration if wall_clock_yield else None,
        "final_loss": losses_history.get("g_total_loss", 0.0),
        "training_time": "N/A",  # overwritten by run_training_pipeline with real wall-clock
        "iterations_completed": max(0, iteration - start_iteration + 1),
        # The key `train.py:905` has always read and no return path ever set, so
        # `run_summary.best_metrics` was `null` on every run ever written (#481).
        "best_metrics": best_metrics,
    }
