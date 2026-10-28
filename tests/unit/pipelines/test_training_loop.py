"""Wiring tests for the ``TrainingLoop`` seam (WS-3 PR-1).

``TrainingLoop`` is the single entry point the config-driven
``run_training_pipeline``, the scripting ``fit()`` path, and the sanity-check
path all route through. PR-1's ``run()`` delegates to the existing
``_execute_training_loop`` body; these tests pin the seam (collaborators held +
forwarded verbatim) WITHOUT running a real loop — a full loop OOM-kills a dev
box, so the heavy end-to-end check belongs on the cluster.
"""

from __future__ import annotations

import ast
import contextlib
import inspect
import logging
import pathlib
import re
import textwrap
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from spectramr.config.schemas.logging import LoggingConfigSchema
from spectramr.core.topology import RunTopology
from spectramr.domain.exceptions import ConfigurationError
from spectramr.pipelines import training_loop as tl
from spectramr.pipelines.training_loop import TrainingLoop
from tests.utils.corpus import tracked_yamls

# ---------------------------------------------------------------------------
# AST helpers — source-grep without the formatting fragility.
#
# Substring assertions over ``inspect.getsource`` are cheap and, for a loop that
# OOM-kills a dev box, the only affordable check. They are also brittle in a way
# that is worse than useless: a formatter reflowing
#
#     loss_schedule_controller.on_iteration(iteration, epoch)
#
# across two lines makes ``"...on_iteration(iteration, epoch)" in src`` False
# while the call is still there and still correct. Two tests in this file failed
# for exactly that reason -- reporting a wiring regression against fully-wired
# code, which is the most expensive kind of false positive because it trains you
# to ignore the test.
#
# Parsing instead asks the question the tests actually mean ("is this call
# present, on this object, with these arguments?") and is invariant to line
# breaks, wrapping and whitespace.
# ---------------------------------------------------------------------------


def _ast_of(func) -> ast.AST:
    """Parse a function's source (dedented so nested defs parse)."""
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _method_calls(func, attr: str, *, on: str | None = None) -> list[ast.Call]:
    """Every ``<...>.attr(...)`` call inside *func*, optionally pinned to ``on``."""
    calls: list[ast.Call] = []
    for node in ast.walk(_ast_of(func)):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if not isinstance(target, ast.Attribute) or target.attr != attr:
            continue
        if on is not None:
            base = target.value
            if not isinstance(base, ast.Name) or base.id != on:
                continue
        calls.append(node)
    return calls


def _plain_calls(func, name: str) -> list[ast.Call]:
    """Every bare ``name(...)`` call inside *func*."""
    return [
        node
        for node in ast.walk(_ast_of(func))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _positional_names(call: ast.Call) -> list[str]:
    """Names passed positionally, for asserting *what* was forwarded."""
    return [a.id for a in call.args if isinstance(a, ast.Name)]


def test_training_loop_holds_collaborators():
    loop = TrainingLoop(
        "strat",
        "pipe",
        "cfg",
        "unet",
        tb_writer="tb",
        logging_service="log",
        output_paths={"x": "y"},
        checkpoint_service="ckpt",
        metrics_service="met",
        is_sanity_check=True,
    )
    assert loop.strategy == "strat"
    assert loop.pipeline == "pipe"
    assert loop.config == "cfg"
    assert loop.model_type == "unet"
    assert loop.is_sanity_check is True


def test_training_loop_run_delegates_with_all_collaborators(monkeypatch):
    """``run()`` forwards every held collaborator + ``start_iteration`` to the
    loop body exactly once. The body now lives module-level in
    ``pipelines/training_loop.py`` (WS-3 PR-2), so patch it there."""
    import spectramr.pipelines.training_loop as training_loop_mod

    captured: dict = {}

    def fake_loop(strategy, pipeline, config, model_type, **kw):
        captured.update(
            strategy=strategy,
            pipeline=pipeline,
            config=config,
            model_type=model_type,
            **kw,
        )
        return {"success": True, "iterations_completed": 0}

    monkeypatch.setattr(training_loop_mod, "_execute_training_loop", fake_loop)

    loop = TrainingLoop(
        "strat",
        "pipe",
        "cfg",
        "unet",
        tb_writer="tb",
        logging_service="log",
        output_paths={"x": "y"},
        checkpoint_service="ckpt",
        metrics_service="met",
        is_sanity_check=True,
    )
    result = loop.run(start_iteration=7)

    assert result == {"success": True, "iterations_completed": 0}
    assert captured["strategy"] == "strat"
    assert captured["pipeline"] == "pipe"
    assert captured["config"] == "cfg"
    assert captured["model_type"] == "unet"
    assert captured["tb_writer"] == "tb"
    assert captured["logging_service"] == "log"
    assert captured["output_paths"] == {"x": "y"}
    assert captured["checkpoint_service"] == "ckpt"
    assert captured["metrics_service"] == "met"
    assert captured["is_sanity_check"] is True
    assert captured["start_iteration"] == 7


def test_loop_advances_strategy_loop_state_each_iteration():
    """WS-3 PR-3: the loop body must advance ``strategy.loop_state`` so the
    strategy reads the live iteration (not the frozen, perpetually-zero
    ``env.step``). A full loop OOM-kills a dev box, so pin the wiring at source
    level — the same pattern the repo uses for other loop-body invariants
    (e.g. the step_executor scheduler-cadence read)."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    # The loop writes the live counters through the seam each step.
    assert "loop_state.iteration = iteration" in src
    assert "loop_state.epoch = epoch" in src
    # Guarded by a getattr so a strategy lacking the seam doesn't crash the loop.
    assert 'getattr(strategy, "loop_state", None)' in src


def test_invariant_config_lookups_hoisted_above_loop():
    """WS-E perf: gradient-accumulation (_gas) and EMA cadence
    (_ema_update_freq / _ema_warmup) are invariant across the loop (frozen
    config + a fixed strategy/step_executor), so they must be resolved ONCE
    above the ``for iteration`` header, not re-``getattr``-ed every step. A full
    loop OOM-kills a dev box, so pin this at source level (repo convention)."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    loop_at = src.index("for iteration in pbar:")
    before, inside = src[:loop_at], src[loop_at:]

    # All three invariants are assigned in the pre-loop section...
    for name in ("_gas =", "_ema_update_freq =", "_ema_warmup ="):
        assert name in before, f"{name} must be hoisted above the loop"
    # ...and NOT recomputed inside the loop body (the old per-iteration reads).
    assert "_gas =" not in inside
    assert "_ema_update_freq =" not in inside
    assert 'getattr(config, "ema", None)' not in inside


def test_ema_is_gated_on_the_optimizer_step_cadence():
    """The EMA gate receives the same accumulation cadence as the schedulers;
    without it the shadow was blended on every micro-batch."""
    import spectramr.pipelines.training_loop as training_loop_mod

    loop = training_loop_mod._execute_training_loop
    (call,) = _plain_calls(loop, "ema_should_update")
    assert _positional_names(call)[-1] == "_gas"
    assert any("_gas" in _positional_names(c) for c in _plain_calls(loop, "_should_step_schedulers"))


def test_the_divergence_guard_is_the_monitor_on_every_step():
    """The NaN check is ``NonFiniteMonitor``: a device flag recorded every step and
    read one step late, so a non-log step pays no host sync for it.

    The guard it replaced read the tqdm postfix's per-step ``.item()``; that sync
    is gone, and so is the postfix outside the log gate. The guard must still not
    be gated on ``optimization.gradient.detect_anomalies``, which also drives
    ``torch.autograd.set_detect_anomaly(True)`` (issue #467): turning the debug
    mode off must never delete the tripwire.
    """
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    loop_at = src.index("for iteration in pbar:")
    inside = src[loop_at:]

    # Every rank, every step, unconditionally: end_step is a collective under DDP.
    assert "\n        _verdict = nonfinite_monitor.end_step(iteration)\n" in inside
    assert "any_rank_nonfinite" not in inspect.getsource(training_loop_mod)
    assert not re.search(r"\bif\b[^\n]*detect_anomalies", inside)

    # The total is recorded before end_step closes the step, under a scaler too.
    record_at = inside.index("nonfinite_monitor.record(")
    assert record_at < inside.index("nonfinite_monitor.end_step(iteration)")
    assert "scaler" not in inside[inside.rindex("\n", 0, record_at) : record_at]

    # The per-step .item() survives only for the sanity trace; the postfix is set
    # at the log gate alone.
    assert inside.count("pbar.set_postfix(") == 1
    assert inside.index("pbar.set_postfix(") > inside.index("iteration % log_interval == 0")
    item_at = inside.index("_quick_loss.item()")
    assert "if is_sanity_check" in inside[inside.rindex("\n        if ", 0, item_at) : item_at]

    # A legitimate 0.0 g_total_loss must not fall through to "loss".
    assert 'losses_history.get("g_total_loss", losses_history.get("loss"))' in src
    assert 'result.get("g_total_loss") or result.get("loss")' not in src


def test_every_save_and_validation_resolves_the_monitor_first():
    """Lag 1 leaves the newest step unread, so anything that persists or evaluates
    the weights drains it first; otherwise a NaN step's weights are saved."""
    src = _loop_source()
    loop_at = src.index("for iteration in pbar:")

    # The periodic/yield save, best save and validation are all preceded by one
    # rank-uniform resolve in the same iteration.
    resolve_at = src.index("_verdict = nonfinite_monitor.resolve()", src.index("time_for_eval ="))
    assert resolve_at < src.index("val_metrics = _run_validation(")
    assert resolve_at < src.index("best_path = checkpoint_service.save_best(")
    assert resolve_at < src.index('reason="yield" if wall_clock_yield else "periodic"')
    gate = src[src.rindex("\n        if ", 0, resolve_at) : resolve_at]
    for operand in ("time_for_eval", "wall_clock_yield", "iteration % checkpoint_interval == 0"):
        assert operand in gate, gate
    assert "may_checkpoint" not in gate, "a rank-0-only resolve strands its peers"

    # The loss-schedule save at the top of an iteration resolves too.
    ls_save = src.index("_save_loss_schedule_checkpoint(", loop_at)
    assert "nonfinite_monitor.resolve()" in src[src.rindex("_ls_decision = RankUtility", 0, ls_save) : ls_save]

    # Loop exit drains the final step before the final save: a top-level
    # statement between the iteration loop and the save that holds reason="final".
    func = _ast_of(tl._execute_training_loop).body[0]
    top = [ast.unparse(stmt) for stmt in func.body]
    loop_ix = next(i for i, s in enumerate(top) if s.startswith("for iteration in pbar:"))
    final_ix = next(i for i, s in enumerate(top) if "reason='final'" in s)
    drains = [
        i for i, s in enumerate(top)
        if s.startswith("if diverged_at is None:") and "nonfinite_monitor.resolve()" in s
    ]
    assert drains and loop_ix < drains[0] < final_ix, (loop_ix, drains, final_ix)


def test_evaluate_wraps_run_validation_in_eval_mode(monkeypatch):
    """WS-3 PR-4: ``TrainingLoop.evaluate()`` drives the SAME ``_run_validation``
    the loop uses at each eval_interval — eval mode, no optimizer steps — and
    returns its aggregated metrics. Patch the validator so this stays a light
    wiring test (a real validation pass OOM-kills a dev box)."""
    import spectramr.pipelines.training_loop as training_loop_mod

    captured: dict = {}

    def fake_validation(pipeline, strategy, iteration, epoch, logging_service, **kw):
        captured.update(
            pipeline=pipeline,
            strategy=strategy,
            iteration=iteration,
            epoch=epoch,
            logging_service=logging_service,
            **kw,
        )
        return {"val_psnr": 31.5}

    monkeypatch.setattr(training_loop_mod, "_run_validation", fake_validation)

    pipeline = MagicMock()
    pipeline.generator.training = True  # model was in train mode

    loop = TrainingLoop(
        "strat",
        pipeline,
        "cfg",
        "unet",
        logging_service="log",
        metrics_service="met",
        output_paths={"x": "y"},
    )
    result = loop.evaluate()

    assert result == {"val_psnr": 31.5}
    # eval mode set, then restored (model was training).
    pipeline.generator.eval.assert_called_once()
    pipeline.generator.train.assert_called_once()
    # The held collaborators are forwarded; iteration/epoch are 0 (standalone).
    assert captured["strategy"] == "strat"
    assert captured["pipeline"] is pipeline
    assert captured["iteration"] == 0
    assert captured["epoch"] == 0
    assert captured["logging_service"] == "log"
    assert captured["metrics_service"] == "met"


def test_evaluate_drains_the_validation_image_writer(monkeypatch):
    """A standalone eval must leave its PNGs written and the writer thread stopped."""
    import spectramr.pipelines.training_loop as training_loop_mod

    monkeypatch.setattr(training_loop_mod, "_run_validation", lambda *a, **k: {"m": 1.0})
    metrics_service = MagicMock()
    pipeline = MagicMock()
    pipeline.generator.training = False
    TrainingLoop("strat", pipeline, "cfg", "unet", metrics_service=metrics_service).evaluate()
    metrics_service.shutdown_image_writer.assert_called_once_with()


def test_evaluate_warns_when_the_image_drain_fails(monkeypatch):
    import spectramr.pipelines.training_loop as training_loop_mod

    monkeypatch.setattr(training_loop_mod, "_run_validation", lambda *a, **k: {"m": 1.0})
    metrics_service = MagicMock()
    metrics_service.shutdown_image_writer.side_effect = OSError("disk full")
    logging_service = MagicMock()
    pipeline = MagicMock()
    pipeline.generator.training = False
    loop = TrainingLoop(
        "strat",
        pipeline,
        "cfg",
        "unet",
        metrics_service=metrics_service,
        logging_service=logging_service,
    )
    assert loop.evaluate() == {"m": 1.0}
    assert "disk full" in logging_service.log_warning.call_args.args[0]


def test_evaluate_returns_empty_dict_when_validator_yields_none(monkeypatch):
    """A validator that returns ``None`` (e.g. no val batches) must surface as an
    empty dict, never ``None`` — callers do ``metrics["..."]`` lookups."""
    import spectramr.pipelines.training_loop as training_loop_mod

    monkeypatch.setattr(training_loop_mod, "_run_validation", lambda *a, **k: None)
    pipeline = MagicMock()
    pipeline.generator.training = False
    loop = TrainingLoop("strat", pipeline, "cfg", "unet")
    assert loop.evaluate() == {}


# ---------------------------------------------------------------------------
# Loss-schedule controller wiring (source-level, same rationale as the
# loop_state invariant above: a full loop OOM-kills a dev box).
# ---------------------------------------------------------------------------


def test_loop_constructs_loss_schedule_controller_when_enabled():
    """The loop body builds a LossScheduleController, gated on
    ``config.loss_schedule.enabled``, and passes the full config for base-weight
    resolution. Absent/disabled => no controller (no-op)."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    assert "config.loss_schedule and config.loss_schedule.enabled" in src
    assert "LossScheduleController(" in src
    assert "loss_config=config" in src


def test_loop_invokes_controller_on_iteration_and_validation():
    """Clock triggers resolve every step (on_iteration); plateau/threshold +
    post-change windows resolve at validation cadence (on_validation). Both write
    the override map the loss computer reads."""
    import spectramr.pipelines.training_loop as training_loop_mod

    loop = training_loop_mod._execute_training_loop

    # Parsed, not substring-matched: the previous assertion pinned
    # ``on_iteration(iteration, epoch)`` on ONE line and went red the moment a
    # formatter wrapped the call, reporting a wiring regression against wiring
    # that was never removed.
    on_iteration = _method_calls(loop, "on_iteration", on="loss_schedule_controller")
    assert on_iteration, "the loop never calls loss_schedule_controller.on_iteration"
    assert any(
        _positional_names(call) == ["iteration", "epoch"] for call in on_iteration
    ), "on_iteration must receive the live iteration and epoch"

    assert _method_calls(loop, "on_validation", on="loss_schedule_controller")

    # both paths publish the overrides through the loop_state seam
    src = inspect.getsource(loop)
    assert src.count("loop_state.loss_weight_overrides = ") >= 2


def test_loop_publishes_overrides_to_loss_computer_for_all_paradigms():
    """The loop calls the paradigm-agnostic seam ``sync_scheduled_loss_weights``
    so the override reaches the loss computer for EVERY strategy, not just
    reconstruction (the critical "silent no-op outside reconstruction" finding).
    Source-level pin: a full loop OOM-kills a dev box."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    assert "strategy.sync_scheduled_loss_weights()" in src


def test_loop_stamps_loss_schedule_events_into_final_metrics():
    """Provenance (#15): each fire/rollback is stamped into final_metrics.json so
    a dynamic loss curriculum is auditable, not inferred."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    assert '"loss_schedule_events"' in src
    assert "loss_schedule_controller.events" in src


def test_loop_saves_checkpoint_before_a_triggered_change():
    """When a loss-schedule trigger fires, the loop saves a checkpoint (throttled
    by the controller's configurable interval) before proceeding — checked at both
    the clock seam (on_iteration) and the plateau seam (on_validation)."""
    import spectramr.pipelines.training_loop as training_loop_mod

    loop = training_loop_mod._execute_training_loop

    # both seams ask the controller (which owns the user-set interval throttle)
    requests = _method_calls(
        loop, "consume_checkpoint_request", on="loss_schedule_controller"
    )
    assert len(requests) >= 2, (
        "expected the clock seam AND the plateau seam to ask the controller "
        f"whether to checkpoint; found {len(requests)} call(s)"
    )
    assert all(_positional_names(call) == ["iteration"] for call in requests)

    assert len(_plain_calls(loop, "_save_loss_schedule_checkpoint")) >= 2


def test_loop_gates_shared_writes_on_main_rank():
    """DDP rank-safety: the loop computes a main-rank flag and gates every shared
    artifact write (CSV, final_metrics) on it, so non-zero ranks don't race on
    the shared output dir. Source-level pin (real DDP needs multiple processes).

    **Checkpoints are the exception** and now gate on ``may_checkpoint``, not on
    ``is_main_process`` directly. Under FSDP/DeepSpeed, building a checkpoint is
    a collective: rank-0-only gating makes rank 0 enter an all-gather nobody
    else enters, and the job hangs instead of failing. ``may_checkpoint`` is
    ``is_main_process`` ORed with that requirement, and the adapter -- not this
    predicate -- decides which rank actually writes.
    """
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    assert "is_main_process = RankUtility.is_main_rank()" in src
    # csv creation gated
    assert '"csv_log_file" in output_paths and is_main_process' in src
    # final_metrics still rank-0-only: a plain file write, not a collective
    assert 'output_paths.get("run_output_dir") and is_main_process' in src
    # checkpoints route through the collective-aware predicate, derived from it
    assert "may_checkpoint = is_main_process or checkpoints_need_all_ranks" in src
    assert re.search(r"checkpoint_enabled\s+and may_checkpoint\b", src)
    # the gate reads config, not a service's truthiness (the injected-store check is
    # an explicit ``is None`` test)
    assert not re.search(r"and checkpoint_service(?! is None)", src)
    assert src.count("may_checkpoint") >= 4
    assert src.count("is_main_process") >= 4


def test_loss_schedule_decision_is_broadcast_from_rank0():
    """The loss-schedule decision (override map + checkpoint flag) is computed on
    rank 0 and broadcast, so loss_plateau (which reads the per-rank training loss)
    cannot diverge across ranks.

    The DECISION stays rank-0-and-broadcast; the SAVE does not. This test
    previously asserted ``and is_main_process`` on the save line too, pinning a
    contract that deadlocks under FSDP/DeepSpeed: building a checkpoint there is
    a collective, so rank 0 entering it alone hangs the job. The two halves are
    genuinely different questions -- *what* to do must be single-source and
    identical everywhere, *who participates* in doing it depends on the parallel
    strategy. See TestCollectiveCheckpointGate.
    """
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    # Unchanged: the decision is still computed once and broadcast.
    assert src.count("RankUtility.broadcast_object(_ls_decision)") >= 2
    # Changed: every rank may now enter the save under a collective strategy.
    assert 'if _ls_decision["checkpoint"] and checkpoint_enabled and may_checkpoint' in src
    assert 'if _ls_decision["checkpoint"] and is_main_process' not in src


def test_best_checkpoint_path_broadcast_before_restore():
    """DDP: only rank 0 wrote the best checkpoint, so its path is broadcast before
    restore_best_weights — otherwise non-main ranks keep their latest weights and
    the final model differs across ranks."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    bcast = src.index(
        "best_checkpoint_path = RankUtility.broadcast_object(best_checkpoint_path)"
    )
    restore = src.index("checkpoint_service.best_weights(")
    assert bcast < restore  # broadcast happens before the restore
    # The broadcast value is what the restore reads, not some other path.
    call = src[restore : src.index(")", restore)]
    assert "best_checkpoint_path" in call


def test_the_final_save_precedes_the_best_restore():
    """#2799: the final file is the last state; restore-best then swaps in the best weights."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    restore = src.index("checkpoint_service.best_weights(")
    assert src.index('reason="final"') < restore
    assert "best_checkpoint_path" in src[restore : src.index(")", restore)]


def test_the_final_step_is_the_last_iteration_that_ran():
    src = _loop_source()
    final_at = src.index("# Final checkpoint on completion")
    block = src[final_at : final_at + 1600]
    assert "ran_a_step = iteration >= first_iteration" in block
    assert "step=iteration," in block
    assert "step=max_iterations" not in src


def test_no_step_ran_means_no_final_save():
    src = _loop_source()
    final_at = src.index("# Final checkpoint on completion")
    assert "and ran_a_step" in src[final_at : final_at + 1600]


def test_a_loop_with_nowhere_to_checkpoint_fails_before_step_one():
    src = _loop_source()
    check = src.index("the loop has no run directory")
    assert check < src.index("for iteration in pbar:")


def test_throttle_interval_lives_in_controller_not_save_helper():
    """The user-set throttle interval is read by the controller's
    consume_checkpoint_request, NOT hardcoded/duplicated in the I/O save helper."""
    import spectramr.pipelines.training_loop as training_loop_mod
    from spectramr.infrastructure.training import loss_schedule_controller as lsc

    helper_src = inspect.getsource(training_loop_mod._save_loss_schedule_checkpoint)
    consume_src = inspect.getsource(
        lsc.LossScheduleController.consume_checkpoint_request
    )
    # the configurable interval gate lives in the controller...
    assert "checkpoint_min_interval" in consume_src
    # ...and is NOT re-implemented in the loop's I/O helper.
    assert "checkpoint_min_interval" not in helper_src


# ---------------------------------------------------------------------------
# Schedule-free optimizers: train()/eval() at the validation boundary
#
# Schedule-free methods (Defazio et al. 2024) maintain an averaged sequence
# distinct from the point the gradient is evaluated at. Validating without
# swapping to it measures the WRONG point in weight space -- and the symptom is
# a metric that reads as a merely-worse arm, not as a bug. That is why the swap
# is wired into the loop rather than documented as the user's job.
# ---------------------------------------------------------------------------


class TestScheduleFreeOptimizerModeBoundary:
    @staticmethod
    def _pipeline_with(optimizers):
        from types import SimpleNamespace

        return SimpleNamespace(optimizers=optimizers)

    def _schedule_free(self):
        import torch

        calls: list[str] = []

        class _SF(torch.optim.SGD):
            def train(self_inner):
                calls.append("train")

            def eval(self_inner):
                calls.append("eval")

        param = torch.nn.Parameter(torch.zeros(1))
        return _SF([param], lr=0.1), calls

    def test_eval_then_train_round_trip(self) -> None:
        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        opt, calls = self._schedule_free()
        pipeline = self._pipeline_with({"opt_g": opt})

        _set_optimizer_eval_mode(pipeline, train=False)
        _set_optimizer_eval_mode(pipeline, train=True)
        assert calls == ["eval", "train"]

    def test_is_a_noop_for_ordinary_optimizers(self) -> None:
        """Every torch optimizer lacks these methods; this must not crash on the
        99% of arms that use one."""
        import torch

        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        param = torch.nn.Parameter(torch.zeros(1))
        pipeline = self._pipeline_with({"opt_g": torch.optim.Adam([param])})
        _set_optimizer_eval_mode(pipeline, train=False)
        _set_optimizer_eval_mode(pipeline, train=True)

    def test_tolerates_a_missing_optimizers_mapping_and_none_entries(self) -> None:
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        _set_optimizer_eval_mode(SimpleNamespace(), train=False)
        _set_optimizer_eval_mode(self._pipeline_with({"opt_d": None}), train=True)

    def test_applies_to_every_optimizer_not_just_the_generator(self) -> None:
        """A GAN's discriminator optimizer needs the same swap."""
        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        opt_g, calls_g = self._schedule_free()
        opt_d, calls_d = self._schedule_free()
        _set_optimizer_eval_mode(
            self._pipeline_with({"opt_g": opt_g, "opt_d": opt_d}), train=False
        )
        assert calls_g == ["eval"] and calls_d == ["eval"]

    def test_a_half_implemented_optimizer_is_never_switched_one_way(self) -> None:
        """An object with ``train`` but no ``eval`` must be skipped ENTIRELY.

        This function inlined its own predicate that tested only the method for
        the direction it was toggling, so such an object was switched into train
        mode and never switched back out -- left in the wrong mode for the rest
        of the run. It now defers to ``supports_schedule_free_modes``, which
        requires both (plus ``step``, since every nn.Module has train/eval).
        """
        import torch

        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        calls: list[str] = []

        class _HalfSF(torch.optim.SGD):
            def train(self):
                calls.append("train")

        param = torch.nn.Parameter(torch.zeros(1))
        pipeline = self._pipeline_with({"opt_g": _HalfSF([param], lr=0.1)})
        _set_optimizer_eval_mode(pipeline, train=False)
        _set_optimizer_eval_mode(pipeline, train=True)
        assert calls == []

    def test_train_mode_is_set_before_the_first_step(self) -> None:
        """schedulefree 1.4 constructs in eval mode and raises on a step taken
        there; the post-validation switch was the only one, so step 1 raised."""
        import spectramr.pipelines.training_loop as training_loop_mod

        loop = training_loop_mod._execute_training_loop
        to_train = [
            c
            for c in _plain_calls(loop, "_set_optimizer_eval_mode")
            if any(
                k.arg == "train" and isinstance(k.value, ast.Constant) and k.value.value is True
                for k in c.keywords
            )
        ]
        first_step = min(c.lineno for c in _method_calls(loop, "train_step", on="strategy"))
        assert any(c.lineno < first_step for c in to_train)

    def test_standalone_evaluate_grades_the_averaged_weights(self, monkeypatch) -> None:
        import spectramr.pipelines.training_loop as training_loop_mod

        opt, calls = self._schedule_free()
        seen: list[list[str]] = []
        monkeypatch.setattr(
            training_loop_mod,
            "_run_validation",
            lambda *a, **k: seen.append(list(calls)) or {"val_psnr": 1.0},
        )
        pipeline = MagicMock()
        pipeline.generator.training = True
        pipeline.optimizers = {"opt_g": opt}

        TrainingLoop("strat", pipeline, "cfg", "unet").evaluate()

        assert seen == [["eval"]]
        assert calls == ["eval", "train"]

    def test_it_reaches_through_the_lookahead_wrapper(self) -> None:
        """``optimizers`` holds the WRAPPER once lookahead.enabled is set.

        ``optimizer.type: schedulefree_adamw`` + ``optimizer.lookahead.enabled:
        true`` put a ``Lookahead`` in this mapping, and it exposed no mode API,
        so validation silently graded the un-averaged iterate.
        """
        from spectramr.infrastructure.training.optimizers.lookahead import Lookahead
        from spectramr.pipelines.training_loop import _set_optimizer_eval_mode

        opt, calls = self._schedule_free()
        pipeline = self._pipeline_with({"opt_g": Lookahead(opt, 2, 0.5)})
        _set_optimizer_eval_mode(pipeline, train=False)
        _set_optimizer_eval_mode(pipeline, train=True)
        assert calls == ["eval", "train"]


class TestCollectiveCheckpointGate:
    """One predicate decides who ENTERS the checkpoint block.

    Gating on ``is_main_rank()`` alone is correct for DP/DDP (which replicate)
    and a DEADLOCK for FSDP/DeepSpeed (which shard): rank 0 enters an
    all-gather that ranks 1..N never enter, so the job hangs with no exception
    and no log line until SLURM kills it at walltime.

    Asserted by AST over the loop source rather than by running four ranks --
    the failure is unobservable single-host, which is exactly why it survived.
    """

    @staticmethod
    def _source():
        import inspect

        from spectramr.pipelines import training_loop

        return inspect.getsource(training_loop)

    def test_the_predicate_exists_and_ors_in_the_collective_case(self):
        source = self._source()
        assert (
            "may_checkpoint = is_main_process or checkpoints_need_all_ranks" in source
        )

    def test_no_checkpoint_site_still_gates_on_is_main_process_alone(self):
        """The regression guard: a new checkpoint call site that copies the old
        `and is_main_process` line reintroduces the hang."""
        offenders = _rank0_only_checkpoint_blocks(self._source())
        assert not offenders, (
            f"checkpoint block(s) at line(s) {offenders} gate on is_main_process "
            "alone; use `may_checkpoint` or an FSDP/DeepSpeed run deadlocks"
        )

    @pytest.mark.parametrize(
        "save",
        [
            "store.save_step(pipeline, step=1)",
            "store.save_best(pipeline, metric_name='m', metric_value=1.0)",
            "_save_loss_schedule_checkpoint(config=config)",
            "restore.best_weights(path, pipeline, parallel=parallel_runtime)",
        ],
    )
    def test_the_guard_turns_red_on_every_save_shape(self, save):
        planted = f"def f():\n    if is_main_process:\n        {save}\n"
        assert _rank0_only_checkpoint_blocks(planted) == [2]

    def test_the_loss_schedule_helper_receives_the_parallel_runtime(self):
        """It builds a checkpoint like every other site, so it needs the
        adapter too -- without it an FSDP save writes rank 0's local SHARD."""
        source = self._source()
        assert "parallel_runtime=parallel_runtime," in source
        helper = source[source.index("def _save_loss_schedule_checkpoint") :]
        helper = helper[: helper.index("\nclass ")]
        assert "parallel=parallel_runtime," in helper

    def test_the_runtime_reaches_every_store_call(self):
        """Without the runtime the store resolves the default adapter and writes this
        rank's SHARD as if it were the whole model."""
        for func in (tl._execute_training_loop, tl._save_loss_schedule_checkpoint):
            calls = [*_method_calls(func, "save_step"), *_method_calls(func, "save_best"),
                     *_method_calls(func, "best_weights")]
            assert calls, func.__name__
            for call in calls:
                kw = {k.arg: k.value for k in call.keywords}
                assert isinstance(kw.get("parallel"), ast.Name), call.lineno
                assert kw["parallel"].id == "parallel_runtime", call.lineno


def _rank0_only_checkpoint_blocks(source: str) -> list[int]:
    """Line numbers of ``if`` blocks that gate a checkpoint save on
    ``is_main_process`` alone."""
    import ast

    offenders = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "is_main_process" not in names:
            continue
        # Derived, not enumerated: any callee that saves a checkpoint, or the one
        # restore that is collective under DeepSpeed.
        called = {
            (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", ""))
            for n in ast.walk(node)
            if isinstance(n, ast.Call)
        }
        # CALLEES only, and only those that SAVE. Two false-positive classes
        # this deliberately excludes, both of which rank-0 gating is correct
        # for:
        #   * `checkpoints_need_all_ranks` in a log line -- a Name, not a
        #     call, so scanning every Name node flagged the explanatory
        #     logger.info block itself;
        #   * `consume_checkpoint_request` -- rank 0 DECIDING whether to
        #     checkpoint, which is then broadcast so all ranks agree
        #     (test_loss_schedule_decision_is_broadcast_from_rank0). The
        #     decision must be single-source; only the SAVE is collective.
        if any(
            ("save" in name.lower() and "checkpoint" in name.lower())
            or name in ("save_step", "save_best", "best_weights", "_save_loss_schedule_checkpoint")
            for name in called
        ):
            offenders.append(node.lineno)
    return offenders


class TestSchedulerStepFailuresRaise:
    """A scheduler that cannot step fails on EVERY iteration.

    Warning meant thousands of identical lines in a log nobody reads while the
    run trained at a constant LR -- the "late-stage oscillation" the stepping
    block was added to fix, silently reintroduced, with the declared schedule
    still stamped in provenance.
    """

    def test_the_handler_raises_rather_than_warning(self):
        import ast
        import inspect

        from spectramr.pipelines import training_loop

        tree = ast.parse(inspect.getsource(training_loop))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            if node.name != "sched_err":
                continue
            kinds = {type(n) for n in ast.walk(node)}
            assert ast.Raise in kinds, (
                "the scheduler-step handler must raise; warning lets the run "
                "train at a constant LR while claiming its declared schedule"
            )
            return
        raise AssertionError("scheduler-step exception handler not found")


class TestParallelStepPolicyInstall:
    """The plugin resolves a step policy; something must hand it over.

    Before ``_install_parallel_step_policy`` existed, every plugin computed a
    ``step_policy``, the director stored it on ``ParallelRuntime``, and NOTHING
    in ``src/`` read it. So ``DeepSpeedStepPolicy`` never took effect and the
    loop ran ``loss.backward()``/``optimizer.step()`` instead of the engine's --
    1/N^2 loss scaling with a fully green test suite.
    """

    @staticmethod
    def _install(strategy, runtime):
        from spectramr.pipelines.training_loop import _install_parallel_step_policy

        return _install_parallel_step_policy(strategy, runtime)

    def test_instance_policy_is_adopted(self):
        strategy = MagicMock()
        policy = MagicMock(owns_gradient_accumulation=True, owns_zero_grad=True)
        self._install(strategy, MagicMock(step_policy=policy, strategy="deepspeed"))
        strategy.step_executor.adopt_step_policy.assert_called_once_with(policy)

    def test_a_policy_class_is_constructed_with_the_arms_clip_settings(self):
        """FSDP hands back the CLASS because only this layer knows the clip
        value; constructing it with AMPPolicy's 1.0 default would silently
        change what the arm clips at."""
        strategy = MagicMock()
        strategy.step_executor.amp_policy = MagicMock(
            max_grad_norm=0.25, enable_gradient_clipping=True
        )

        class _Policy:
            def __init__(self, max_grad_norm=1.0, enable_gradient_clipping=True):
                self.max_grad_norm = max_grad_norm
                self.enable_gradient_clipping = enable_gradient_clipping

        self._install(strategy, MagicMock(step_policy=_Policy, strategy="fsdp"))

        (adopted,), _ = strategy.step_executor.adopt_step_policy.call_args
        assert isinstance(adopted, _Policy)
        assert adopted.max_grad_norm == 0.25

    def test_no_runtime_is_a_no_op(self):
        strategy = MagicMock()
        self._install(strategy, None)
        strategy.step_executor.adopt_step_policy.assert_not_called()

    def test_a_policy_that_cannot_be_adopted_raises(self):
        """Dropping it silently yields wrong gradients, not an error, so the
        unadoptable case must be loud."""
        import pytest

        strategy = MagicMock()
        strategy.step_executor = object()  # no adopt_step_policy
        with pytest.raises(RuntimeError, match="step policy"):
            self._install(
                strategy, MagicMock(step_policy=MagicMock(), strategy="deepspeed")
            )


# ---------------------------------------------------------------------------
# losses.csv column selection (#696)
#
# `_csv_metric_names` decides the CSV HEADER. The row writer uses
# `extrasaction="ignore"`, so a metric that is computed but has no fieldname is
# discarded in silence -- which is how the drained `kspace_filling` cohort lost
# `hfen` / `kspace_error` / `phase_mse` while still computing them.
# ---------------------------------------------------------------------------


class TestCsvMetricNames:
    """The header promises EXACTLY what the row producer can fill.

    Was "both selection sources reach the header, and neither shadows the
    other" -- a union. The union's protection (a drained arm's `compute` list
    must not be shadowed by schema-default flags, #696) is kept by precedence;
    its cost is not. See `_csv_metric_names`' docstring.
    """

    def test_the_compute_list_reaches_the_header(self):
        """The regression. A drained arm declares `metrics.compute` and NO
        flags, so every flag sits at a schema default it never asked for.
        Reading only the flags is what dropped the columns."""
        from spectramr.pipelines.training_loop import _csv_metric_names

        # Exactly the drained shape: the list is the arm's answer, and the
        # flags below carry the DEFAULTS, which are the inverse of it.
        drained = {
            "compute": ["hfen", "kspace_error", "phase_mse", "robust_mri_psnr"],
            "compute_hfen": False,
            "compute_kspace_error": False,
            "compute_phase_mse": False,
            "compute_psnr": True,
            "compute_ssim": True,
        }
        assert {"hfen", "kspace_error", "phase_mse"} <= _csv_metric_names(drained)

    def test_flags_still_select_columns_when_no_list_is_declared(self):
        """~500 inprogress arms are undrained. The list must not displace them."""
        from spectramr.pipelines.training_loop import _csv_metric_names

        assert _csv_metric_names({"compute_hfen": True, "compute_psnr": True}) == {
            "hfen",
            "psnr",
        }
        assert _csv_metric_names({}) == set()

    def test_the_compute_list_wins_outright_over_the_flags(self):
        """DELIBERATE INVERSION of `test_the_two_sources_union_rather_than_override`.

        That test pinned the union on the reasoning that "a surplus column costs
        an empty string and a missing one costs a discarded value", so union was
        the cheap direction. The first half of that is wrong: a surplus column
        is not free, it is a permanently-blank column in the one artifact a
        researcher reads to see whether metrics were computed, which is
        pitfall #15 in artifact form. On `experiment_11_attention_none` the union
        promised `train_mse` / `train_psnr` / `train_ssim` from flags sitting at
        schema defaults the arm never declared, and all three were unfillable.

        The discarded-value half is real, and is now caught loudly at the row
        writer instead of being made impossible by over-promising columns.
        """
        from spectramr.pipelines.training_loop import _csv_metric_names

        both = {"compute_psnr": True, "compute": ["hfen"]}
        assert _csv_metric_names(both) == {"hfen"}

    def test_the_flags_are_the_fallback_when_the_list_is_empty(self):
        """#696's protection, restated: precedence only applies when the arm
        actually declared a list. An unmigrated arm still reads its flags."""
        from spectramr.pipelines.training_loop import _csv_metric_names

        assert _csv_metric_names({"compute_psnr": True, "compute": []}) == {"psnr"}
        assert _csv_metric_names({"compute_psnr": True}) == {"psnr"}

    def test_header_resolver_mirrors_the_producer_exactly(self):
        """The invariant that replaces the union: no data loss AND no empty
        column. Asked of both real resolvers, so a future divergence in either
        one fails here."""
        import pytest

        pytest.importorskip("yaml")
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )
        from spectramr.pipelines.training_loop import _csv_metric_names

        arm = Path(
            "experiments/inprogress/kspace_filling/attention_shootout/"
            "experiment_11_attention_none.yaml"
        )
        if not arm.exists():  # published tree excludes experiments/
            pytest.skip(f"{arm} not present in this checkout")

        settings = TrainingSettings.from_yaml(str(arm))
        computed = set(
            MetricsMixin._extract_metrics_from_config(MetricsMixin(), settings.metrics)
        )
        columns = _csv_metric_names(settings.metrics.model_dump())
        assert computed, "arm computes nothing -- this test has gone vacuous"
        assert computed - columns == set(), f"DISCARDED: {computed - columns}"
        assert columns - computed == set(), f"ALWAYS EMPTY: {columns - computed}"

    def test_validation_scoring_names_do_not_become_train_columns(self):
        """`validation.scoring.compute` used to be unioned in as `train_*` on the
        premise that a validation-scored metric is "normally computed on the
        training batch too". The training computer is fed ONLY
        `_extract_metrics_from_config(config.metrics)`, so no such path exists,
        and the premise cost this arm a permanently-empty `train_psnr`.

        Asserted at source level: the header build must not read
        `config.validation.scoring.compute`, and a behavioural test can only
        cover the branches it happens to exercise.
        """
        import ast
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        tree = ast.parse(src)
        # Any `f"train_{...}"` built inside a loop over `.scoring.compute` is the
        # defect. Look for the attribute chain at all -- it has no other use in
        # this function.
        chains = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "compute"
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "scoring"
        ]
        assert not chains, (
            "the header build reads validation.scoring.compute again -- that "
            "promises train_* columns no producer can fill"
        )

    def test_the_header_map_no_longer_needs_to_be_a_superset(self):
        """This test used to pin ``header ⊇ selection`` (78 vs 43) so an "SSOT
        cleanup" could not shrink the header. The direction was right and the
        premise was wrong: the surplus was not free. 22 header-only flags named a
        REGISTERED metric, so each bought a column no path could fill (#340).

        Both maps derive from ``flag_map.schema_flag_to_metric`` now, so the
        containment they guarded holds as EQUALITY. Asserting equality keeps the
        original protection — the header still cannot shrink below the selection
        set — and adds the half it was missing.
        """
        from spectramr.core.metrics.flag_map import schema_flag_to_metric
        from spectramr.pipelines.training_loop import _CSV_METRIC_NAME_MAP

        ssot = schema_flag_to_metric()
        assert ssot, "the SSOT map is empty -- this test has gone vacuous"
        assert dict(_CSV_METRIC_NAME_MAP) == ssot
        # The original invariant, still true, now by construction.
        assert set(ssot) >= set(_CSV_METRIC_NAME_MAP)

    def test_every_metric_a_drained_arm_computes_has_a_column(self):
        """The seam: ask both real resolvers about a real drained arm rather
        than a hand-built dict, which would be a third resolver."""
        import pytest

        pytest.importorskip("yaml")
        from pathlib import Path

        from spectramr.config.settings import TrainingSettings
        from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
            MetricsMixin,
        )
        from spectramr.pipelines.training_loop import _csv_metric_names

        arm = Path(
            "experiments/inprogress/kspace_filling/experiment_11_kan_dual_domain.yaml"
        )
        if not arm.exists():  # published tree excludes experiments/
            pytest.skip(f"{arm} not present in this checkout")

        settings = TrainingSettings.from_yaml(str(arm))
        computed = set(
            MetricsMixin._extract_metrics_from_config(MetricsMixin(), settings.metrics)
        )
        assert computed, "arm computes nothing -- this test has gone vacuous"
        # No `validation.scoring.compute` union here any more -- the source
        # stopped promising those names, so unioning them in the test would
        # measure a resolver that no longer exists.
        columns = _csv_metric_names(settings.metrics.model_dump())
        assert computed <= columns, computed - columns
        # The stronger half, now true: no column is unfillable either.
        assert columns == computed, columns ^ computed


# ---------------------------------------------------------------------------
# #481, second half: `run_summary.best_metrics` was structurally null.
#
# `train.py:905` reads `result.get("best_metrics")`, but no return path of
# `_execute_training_loop` ever set that key — so the field was `null` on every
# run ever written, regardless of what validation measured. The payload that
# carries the answer was assembled a few lines above, written to
# `final_metrics.json`, and then dropped.
# ---------------------------------------------------------------------------


def test_execute_training_loop_returns_best_metrics_on_every_path():
    """Source-level contract: every `return` carries the key train.py reads.

    Asserted structurally rather than by running a full loop — the point is that
    NO path may omit it, and a behavioural test can only ever cover the paths it
    happens to exercise.
    """
    import ast
    import inspect

    from spectramr.pipelines import training_loop

    tree = ast.parse(inspect.getsource(training_loop._execute_training_loop))
    returns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert returns, "no dict-returning path found — test has gone stale"

    for node in returns:
        keys = {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        assert "best_metrics" in keys, (
            f"the return at line {node.lineno} omits 'best_metrics'; "
            "train.py reads it for run_summary.json and would record null (#481)"
        )


def test_run_summary_reads_the_same_key_the_loop_returns():
    """Pin the seam itself: a rename on either side must fail here.

    The defect was exactly this seam disagreeing — a reader and a producer that
    both looked correct in isolation.
    """
    import ast
    import inspect

    from spectramr.pipelines import train

    src = inspect.getsource(train)
    tree = ast.parse(src)
    reads = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "best_metrics"
    }
    assert "best_metrics" in reads, (
        "train.py no longer reads 'best_metrics' — if it was renamed, "
        "training_loop's return must be renamed with it"
    )


# ---------------------------------------------------------------------------
# #178, entered through a VALUE instead of a key.
#
# The unresolvable-monitor raise covers a monitor KEY that does not exist. A
# monitor whose VALUE is non-finite on every event costs exactly the same two
# mechanisms: `save_best` is gated on `math.isfinite(monitor_value)`, and
# `EarlyStoppingService.update` returns early on non-finite so `wait_count`
# never moves and the stop can never trigger. The run then burns its full budget
# and exits 0 with no best checkpoint.
#
# Reachable today: an arm whose data honours no normalization contract gets
# `val_psnr = NaN` from the range resolver (#180), not a wrong-but-finite score.
# ---------------------------------------------------------------------------


class TestPersistentlyNonFiniteMonitorIsFatal:
    def _escalation_source(self):
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        start = src.index("consecutive_nonfinite_monitor += 1")
        return src[start : start + 1600]

    def test_the_counter_escalates_to_a_raise(self):
        block = self._escalation_source()
        assert "raise RuntimeError" in block, (
            "a monitor that is non-finite forever must fail the run, not warn: "
            "neither early stopping nor best-checkpoint selection can fire again"
        )

    def test_the_threshold_is_patience_not_a_new_constant(self):
        """Reusing the arm's declared tolerance avoids inventing a knob."""
        block = self._escalation_source()
        assert "patience" in block
        assert "consecutive_nonfinite_monitor >= patience" in block

    def test_a_finite_value_resets_the_counter(self):
        """A transient NaN must be tolerated; only a PERSISTENT one is fatal."""
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        assert (
            "consecutive_nonfinite_monitor = 0" in src
        ), "without a reset, one transient NaN accumulates toward the raise"

    def test_the_escalation_is_gated_on_early_stopping_being_enabled(self):
        """With early stopping off, patience is meaningless and must not fire."""
        block = self._escalation_source()
        assert "early_stopping_service.enabled" in block


class TestMonitorNotApplicableReason:
    """The raise explains WHY there is no number, when the computer declared it.

    A bare NaN cannot distinguish "undefined on this input" from "crashed", so
    the reason is recovered from the metrics computer for the message.
    """

    @staticmethod
    def _strategy_with(reasons):
        from types import SimpleNamespace

        return SimpleNamespace(
            _validation_computer=SimpleNamespace(last_not_applicable=reasons)
        )

    def test_it_names_the_reason_for_the_monitored_metric(self):
        from spectramr.core.metrics.outcome import NotApplicableReason
        from spectramr.pipelines.training_loop import _monitor_not_applicable_reason

        strategy = self._strategy_with(
            {"psnr": NotApplicableReason.DATA_RANGE_UNRESOLVED}
        )
        # The monitor is `val_psnr`; the computer keys on the bare metric name.
        text = _monitor_not_applicable_reason(strategy, "val_psnr")

        assert "data_range_unresolved" in text

    def test_it_falls_back_to_listing_what_was_excluded(self):
        from spectramr.core.metrics.outcome import NotApplicableReason
        from spectramr.pipelines.training_loop import _monitor_not_applicable_reason

        strategy = self._strategy_with(
            {"ssim": NotApplicableReason.DATA_RANGE_UNRESOLVED}
        )
        text = _monitor_not_applicable_reason(strategy, "val_lpips")

        assert "ssim" in text

    def test_no_recorded_reason_blames_the_metric_not_the_config(self):
        """An unrecorded NaN is a diverged model, and must not be mislabelled."""
        from spectramr.pipelines.training_loop import _monitor_not_applicable_reason

        text = _monitor_not_applicable_reason(self._strategy_with({}), "val_psnr")

        assert "no not-applicable reason" in text

    def test_a_strategy_with_no_computer_still_yields_a_message(self):
        """This decorates an exception that is raised either way -- it must not
        become a second failure inside the first."""
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import _monitor_not_applicable_reason

        text = _monitor_not_applicable_reason(SimpleNamespace(), "val_psnr")

        assert isinstance(text, str) and text


class TestCsvMetricColumnsDeriveFromTheSchema:
    """``_CSV_METRIC_NAME_MAP`` was 78 hand-written entries; it is now derived.

    The hand-written version was justified as a deliberate SUPERSET of the mixin's
    selection map — "a surplus column costs an empty string, a missing one costs a
    silently discarded value". The first half was wrong. 22 of the surplus flags
    named a REGISTERED metric the mixin could not select, so enabling one wrote a
    header for a column that no code path could fill (#340).
    """

    def test_the_map_is_the_ssot_map(self):
        from spectramr.core.metrics.flag_map import schema_flag_to_metric
        from spectramr.pipelines.training_loop import _CSV_METRIC_NAME_MAP

        assert dict(_CSV_METRIC_NAME_MAP) == schema_flag_to_metric()

    def test_no_hand_written_literal_remains(self):
        """Structural: re-introducing a literal is the regression, not a wrong name."""
        import inspect
        import re

        from spectramr import pipelines

        src = inspect.getsource(pipelines.training_loop)
        head = src.split("def _csv_metric_names")[0]
        literals = re.findall(r'"(compute_\w+)":\s*"', head)
        assert not literals, (
            f"a hand-written flag->name literal is back in training_loop: {literals}. "
            "Derive from flag_map.schema_flag_to_metric() instead."
        )

    def test_the_compute_list_wins_outright_over_the_flags(self):
        """DELIBERATE INVERSION of `test_the_compute_list_still_unions_with_the_flags`.

        #696's protection is preserved and is asserted where it actually applies:
        see `TestCsvMetricNames.test_the_flags_are_the_fallback_when_the_list_is_empty`.
        This case never tested it -- its fixture sets a flag AND a list, which a
        drained arm by definition does not (draining REMOVES the flags), so the
        `psnr` it asserted came from a schema default no arm declared. That is
        the surplus column, not the #696 regression.
        """
        from spectramr.pipelines.training_loop import _csv_metric_names

        names = _csv_metric_names({"compute_psnr": True, "compute": ["hfen", "ssim"]})
        assert names == {"hfen", "ssim"}


class TestLearningRateColumnsReachTheHeader:
    """`lr_opt_g` was measured every iteration and discarded on every row.

    The row producer writes one `lr_<scheduler>` key per entry in
    `pipeline.schedulers`; the header builder promised no `lr_*` column at all.
    `csv.DictWriter(..., extrasaction="ignore")` drops a key with no column
    silently, so the entire learning-rate curve was absent from
    `training_metrics.csv` for every arm that has a scheduler -- and it was one
    of the three keys the write-time discard check named on
    experiment_11_attention_none.
    """

    def test_the_column_name_matches_the_key_the_producer_wrote(self):
        """The observed key, pinned. `opt_g` is the generator optimizer's
        scheduler name in this repo, so this is the exact column the run log
        reported as DISCARDED."""
        from spectramr.pipelines.training_loop import lr_column_name

        assert lr_column_name("opt_g") == "lr_opt_g"

    def test_the_header_covers_every_scheduler_the_producer_iterates(self):
        """The planted shape (CLAUDE.md #15): a scheduler present in the
        mapping the loop iterates must have a column."""
        from spectramr.pipelines.training_loop import lr_column_name, lr_column_names

        schedulers = {"opt_g": object(), "opt_d": object()}
        header = lr_column_names(schedulers)
        produced = {lr_column_name(name) for name in schedulers}
        assert produced <= header, (
            f"the producer writes {sorted(produced - header)} with no column; "
            'extrasaction="ignore" discards them silently'
        )

    def test_no_schedulers_is_not_an_error(self):
        """Most arms have none, and a pipeline may not carry the attribute."""
        from spectramr.pipelines.training_loop import lr_column_names

        assert lr_column_names(None) == set()
        assert lr_column_names({}) == set()

    def test_neither_site_re_spells_the_column(self):
        """Structural, because the defect was two sites naming one column and
        only one of them saying it (non-negotiable 17). A returning
        `f"lr_{...}"` literal is the regression, not a wrong name."""
        import inspect
        import re

        from spectramr import pipelines

        src = inspect.getsource(pipelines.training_loop)
        # Exactly one: `lr_column_name`'s own return. Any second occurrence is a
        # caller spelling the column itself again.
        spellings = re.findall(r'f"lr_\{[^}]*\}"', src)
        assert len(spellings) == 1, (
            f"expected 1 hand-spelled lr column name (the helper's own return), "
            f"found {len(spellings)}: {spellings}. Call lr_column_name() so the "
            "header and the producer cannot diverge."
        )


class TestTheGatedConverterActuallyBatches:
    """`get_last_metrics` returns on-device tensors so this gate is the ONLY
    host transfer (#707). The comment there has said "batched GPU sync" since
    before anything batched: the dict comprehension paid one `.item()` per
    tensor, and each sync drains the queue, so metric k+1 could not start
    launching while k was still in flight."""

    @staticmethod
    def _conversion_source():
        """The conversion prologue of the gate that owns the single host transfer.

        This used to be located with ``src.index("if iteration % log_interval ==
        0:")`` -- the exact failure this file's own header warns about. Widening
        the gate to also log the first and last iteration (so a run shorter than
        ``log_interval`` no longer writes an empty CSV) changed that line's text
        and broke three tests whose SUBJECT -- the fused transfer -- was untouched.

        Located by AST instead: the one ``if`` inside the loop whose test mentions
        ``log_interval``, invariant to how the condition is spelled. Scoped to the
        leading statements up to and including the declaration-order restore; the
        old 1800-character window approximated the same region, and returning the
        whole gate body would let these assertions match text from the CSV writer
        or the TensorBoard block instead.
        """
        gate = _find_guarded_block(_execute_loop_ast(), test_contains="iteration % log_interval")
        prologue = []
        for stmt in gate.body:
            rendered = ast.unparse(stmt)
            prologue.append(rendered)
            if "for k in losses_history" in rendered:
                break
        else:  # pragma: no cover - the restore is what the last test pins
            raise AssertionError(
                "the declaration-order restore is gone from the conversion "
                "prologue; the fused split would otherwise emit every fused "
                "key last"
            )
        return "\n".join(prologue)

    def test_it_uses_the_shared_fused_helper(self):
        block = self._conversion_source()
        assert "fuse_to_host" in block, (
            "the designated single converter must use the fused transfer that "
            "shipped for it, not one .item() per tensor"
        )

    def test_the_helper_is_imported_from_core(self):
        """Rightward import only: pipelines -> core is legal, the reverse is not."""
        from spectramr.pipelines import training_loop

        assert training_loop.fuse_to_host.__module__ == (
            "spectramr.core.metrics.scalar_transfer"
        )

    def test_non_scalar_entries_keep_the_per_item_path(self):
        """Value identity: a multi-element or complex tensor raised in the
        comprehension and must still raise, not be silently reduced."""
        block = self._conversion_source()
        assert "numel() == 1" in block
        assert "torch.is_complex" in block

    def test_declaration_order_is_restored(self):
        """The CSV header was built from `losses_history` order; splitting the
        dict would otherwise emit every fused key last."""
        block = self._conversion_source()
        assert "for k in losses_history" in block


class TestEpochBasedValidationIsReachable:
    """#711: the epoch trigger required `eval_interval <= 0`, which cannot happen.

    `eval_interval` is `validation.schedule.interval_steps` (schema `ge=1`) or
    `max_iterations`. Never <= 0. So `on_epoch` could not fire on any config the
    schema admits — while DEFAULTING TO TRUE, so every arm read as having
    epoch-based validation enabled. Its companion `interval_epochs` was read by
    nothing at all, the N of a mode that could not be entered.
    """

    @staticmethod
    def _trigger_source():
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        # Anchored on the trigger's own unique landmark, NOT on "the second
        # occurrence of `eval_on_epoch`". The ordinal anchor silently slid onto
        # an unrelated block the moment another `eval_on_epoch` read was added
        # earlier in the function -- and slid QUIETLY: two of the four
        # assertions in this class kept passing against the wrong window, so
        # only one of them reported the breakage.
        start = src.index("and is_epoch_end")
        return src[start : start + 1400]

    def test_the_impossible_condition_is_gone(self):
        assert (
            "eval_interval <= 0" not in self._trigger_source()
        ), "interval_steps is ge=1, so this can never be true"

    def test_the_companion_n_is_now_consulted(self):
        """`interval_epochs` was declared, documented, and read by nothing."""
        import inspect

        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        assert "eval_interval_epochs" in src
        assert "epoch % eval_interval_epochs == 0" in src

    def test_sanity_checks_still_bypass_it(self):
        """A sanity check must obey `interval_steps` strictly."""
        assert "not is_sanity_check" in self._trigger_source()

    def test_it_is_additive_not_a_replacement(self):
        """Enabling it must not remove the step-interval event an arm selects
        its checkpoint from — the trigger only ever sets the flag True."""
        block = self._trigger_source()
        assert "time_for_eval = True" in block
        assert "time_for_eval = False" not in block


class TestOnEpochDefaultMatchesRealisedBehaviour:
    """Flipping the default True -> False changes nothing, and that is the point.

    The gate never fired, so every run ever executed was step-based. Leaving the
    default True while making the trigger live would have switched epoch-boundary
    validation ON for the entire corpus as a side effect of a bug fix.
    """

    def test_the_default_is_off(self):
        from spectramr.config.schemas.validation import ValidationScheduleConfigSchema

        assert ValidationScheduleConfigSchema().on_epoch is False

    def test_no_arm_asks_for_epoch_mode_in_either_spelling(self):
        """Measured 2026-08-05: 0 arms declare it true under the current name,
        and all 1395 legacy `eval_on_epoch` declarations are `false`. If this
        ever goes red, the default flip is no longer behaviour-preserving and
        the corpus impact must be re-measured before shipping."""
        import pathlib

        import yaml as _yaml

        root = pathlib.Path(__file__).resolve().parents[3] / "experiments"
        asked = []
        for p in tracked_yamls(root):
            try:
                doc = _yaml.safe_load(p.read_text())
            except Exception:
                continue
            if not isinstance(doc, dict):
                continue
            val = doc.get("validation")
            if not isinstance(val, dict):
                continue
            if val.get("eval_on_epoch") is True:
                asked.append(p.name)
            sched = val.get("schedule")
            if isinstance(sched, dict) and sched.get("on_epoch") is True:
                asked.append(p.name)
        assert not asked, f"arms now request epoch mode: {sorted(set(asked))[:10]}"


# ── the cascade drain (issue #697) ───────────────────────────────────────────
#
# The strategy measures the R-sweep and publishes it; the pipeline persists it.
# That split is deliberate: `IMetricsService` declares only update/get/reset/
# log_metrics, so a strategy writing this CSV itself would reach past its own
# interface. The drain also CLEARS, so a paradigm that skips its cascade on a
# later validation cannot re-persist the previous sweep under a new iteration.


class _Recorder:
    def __init__(self):
        self.calls = []

    def log_cascading_validation(self, rows, *, iteration, epoch):
        self.calls.append((list(rows), iteration, epoch))
        return len(rows)


class _Strategy:
    def __init__(self, rows=None):
        if rows is not None:
            self._last_cascade_rows = rows


def test_drain_writes_the_rows_and_clears_them() -> None:
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    rows = [
        {"acceleration_level": 2.0, "heldout": False},
        {"acceleration_level": 8.0, "heldout": False},
    ]
    strategy, recorder = _Strategy(rows), _Recorder()
    assert _drain_cascade_rows(strategy, recorder, iteration=7, epoch=1) == 2
    written, it, ep = recorder.calls[0]
    assert [r["acceleration_level"] for r in written] == [2.0, 8.0]
    assert (it, ep) == (7, 1)
    assert strategy._last_cascade_rows == []


def test_drain_is_a_no_op_for_a_strategy_without_a_cascade() -> None:
    """Most paradigms have no cascade and no attribute. Not an error."""
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    recorder = _Recorder()
    assert _drain_cascade_rows(_Strategy(), recorder, iteration=7, epoch=1) == 0
    assert recorder.calls == []


def test_draining_twice_does_not_republish_the_same_sweep() -> None:
    """The regression the clear exists to prevent: the same measurements
    reappearing under a later iteration as if they were fresh."""
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    rows = [{"acceleration_level": 2.0, "heldout": False}]
    strategy, recorder = _Strategy(rows), _Recorder()
    _drain_cascade_rows(strategy, recorder, iteration=7, epoch=1)
    assert _drain_cascade_rows(strategy, recorder, iteration=8, epoch=1) == 0
    assert len(recorder.calls) == 1


def test_drain_averages_the_batches_instead_of_publishing_the_last_one() -> None:
    """`validation_step` runs once per VAL BATCH, so a sweep arrives as
    n_batches x n_levels rows. Publishing the last batch would disagree with the
    `val_psnr_8x` column beside it, which is a mean over every batch."""
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    rows = [
        {"acceleration_level": 8.0, "heldout": False, "val_psnr": v}
        for v in (30.0, 32.0, 34.0)
    ]
    strategy, recorder = _Strategy(rows), _Recorder()
    assert _drain_cascade_rows(strategy, recorder, iteration=7, epoch=1) == 1
    written, _, _ = recorder.calls[0]
    assert written[0]["val_psnr"] == 32.0  # (30+32+34)/3, exact in binary float
    assert written[0]["n_batches"] == 3


def test_drain_writes_only_on_the_main_process() -> None:
    """Every rank validates its own shard, so every rank reaches the drain. They
    share one output dir, and `_write_csv_header` renames-then-rewrites on
    schema evolution -- concurrent ranks there can truncate the file."""
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    rows = [{"acceleration_level": 2.0, "heldout": False, "val_psnr": 1.0}]
    strategy, recorder = _Strategy(rows), _Recorder()
    written = _drain_cascade_rows(
        strategy, recorder, iteration=7, epoch=1, is_main_process=False
    )
    assert written == 0
    assert recorder.calls == []


def test_a_non_writing_rank_still_clears_its_rows() -> None:
    """The clear cannot be inside the write branch: rows accumulate per batch,
    so a rank that never writes would grow them without bound and carry one
    validation's measurements into the next."""
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    strategy = _Strategy([{"acceleration_level": 2.0, "heldout": False}])
    _drain_cascade_rows(
        strategy, _Recorder(), iteration=7, epoch=1, is_main_process=False
    )
    assert strategy._last_cascade_rows == []


def test_a_standalone_evaluate_publishes_its_sweep_too() -> None:
    """`evaluate()` runs the SAME cascade at the same GPU cost. Without a drain
    the rows sit on the strategy until the next in-training validation absorbs
    them -- computed and silently discarded, the failure this record removes."""
    from spectramr.pipelines.training_loop import TrainingLoop

    recorder = _Recorder()
    strategy = _Strategy([{"acceleration_level": 2.0, "heldout": False}])

    class _Gen:
        training = False

        def eval(self): ...
        def train(self): ...

    class _Pipeline:
        generator = _Gen()

    runner = TrainingLoop(strategy, _Pipeline(), None, "m", metrics_service=recorder)
    # `_run_validation` is the seam under the drain; stub it so this test is
    # about publication, not about running a real validation.
    import spectramr.pipelines.training_loop as tl

    original = tl._run_validation
    tl._run_validation = lambda *a, **k: {"val_psnr": 1.0}
    try:
        runner.evaluate()
    finally:
        tl._run_validation = original

    assert len(recorder.calls) == 1
    assert strategy._last_cascade_rows == []


def test_a_metrics_service_without_the_writer_warns_rather_than_dropping(
    caplog,
) -> None:
    """Fail soft but LOUD. Silently returning would recreate the exact bug this
    replaced -- values computed at real GPU cost, discarded, nothing in the log.
    """
    from spectramr.pipelines.training_loop import _drain_cascade_rows

    class _Log:
        def __init__(self):
            self.warnings = []

        def log_warning(self, msg):
            self.warnings.append(msg)

    strategy, log = _Strategy([{"acceleration_level": 2.0}]), _Log()
    written = _drain_cascade_rows(
        strategy, object(), iteration=7, epoch=1, logging_service=log
    )
    assert written == 0
    assert len(log.warnings) == 1
    assert "NOT being recorded" in log.warnings[0]
    assert strategy._last_cascade_rows == []


# ---------------------------------------------------------------------------
# The per-rank iteration budget, and the DeepSpeed-only scheduler cadence.
#
# Both of these are the framework half of "why was this arm SLOWER on four GPUs
# than on one". The budget is the cause (every rank runs the whole run); the
# cadence is the thing that made the four-GPU run not even comparable to the
# baseline it was meant to be ranked against.
# ---------------------------------------------------------------------------


def _budget_config(scope: str | None = None, *, accumulation_steps: int = 1):
    """The two attribute paths the two helpers read, and nothing else."""
    training = SimpleNamespace()
    if scope is not None:
        training.iteration_budget_scope = scope
    return SimpleNamespace(
        training=training,
        optimization=SimpleNamespace(
            gradient=SimpleNamespace(accumulation_steps=accumulation_steps)
        ),
    )


def _single_gpu_topology() -> RunTopology:
    return RunTopology(
        execution_mode="local",
        world_size=1,
        local_world_size=1,
        num_nodes=1,
        rank=0,
        local_rank=0,
        cpus_on_node=8.0,
    )


def _four_gpu_topology() -> RunTopology:
    return RunTopology(
        execution_mode="slurm",
        world_size=4,
        local_world_size=4,
        num_nodes=1,
        rank=0,
        local_rank=0,
        cpus_on_node=16.0,
    )


class TestIterationBudgetScope:
    """``per_rank`` is byte-identical; ``global`` refuses, and says why."""

    def test_default_is_per_rank_and_returns_the_bound_unchanged(self, monkeypatch):
        # An arm that never heard of the key must resolve exactly as before.
        monkeypatch.setattr(tl, "resolve_run_topology", _single_gpu_topology)
        assert tl.resolve_iteration_budget(30000, _budget_config()) == 30000

    def test_explicit_per_rank_is_also_unchanged(self, monkeypatch):
        monkeypatch.setattr(tl, "resolve_run_topology", _single_gpu_topology)
        assert tl.resolve_iteration_budget(30000, _budget_config("per_rank")) == 30000

    def test_per_rank_under_four_ranks_still_returns_the_full_count(self, monkeypatch):
        # This is the DEFECT, deliberately preserved as the default: four ranks
        # each run all 30000 iterations. The fix this PR ships is that the
        # behaviour is now declared and logged, not that it silently changed.
        monkeypatch.setattr(tl, "resolve_run_topology", _four_gpu_topology)
        assert tl.resolve_iteration_budget(30000, _budget_config("per_rank")) == 30000

    def test_per_rank_says_out_loud_what_the_run_buys(self, monkeypatch, caplog):
        monkeypatch.setattr(tl, "resolve_run_topology", _four_gpu_topology)
        with caplog.at_level(logging.INFO, logger=tl.logger.name):
            tl.resolve_iteration_budget(30000, _budget_config("per_rank"))
        assert "[BUDGET]" in caplog.text
        assert "effective batch" in caplog.text

    def test_single_gpu_does_not_log_the_budget_banner(self, monkeypatch, caplog):
        # A 1-GPU run has nothing to warn about; the banner would be noise.
        monkeypatch.setattr(tl, "resolve_run_topology", _single_gpu_topology)
        with caplog.at_level(logging.INFO, logger=tl.logger.name):
            tl.resolve_iteration_budget(30000, _budget_config("per_rank"))
        assert "[BUDGET]" not in caplog.text

    def test_global_raises_and_names_both_reasons(self, monkeypatch):
        monkeypatch.setattr(tl, "resolve_run_topology", _four_gpu_topology)
        with pytest.raises(ConfigurationError) as excinfo:
            tl.resolve_iteration_budget(30000, _budget_config("global"))
        message = str(excinfo.value)
        # Reason 1: nothing shards the stream.
        assert "#1163" in message
        # Reason 2: dividing the bound silently reshapes every iteration-keyed
        # schedule. Either reason alone is enough to refuse; the message must
        # carry both, or someone lands #1163 and assumes 'global' is now safe.
        assert "curriculum" in message

    def test_global_raises_on_one_gpu_too(self, monkeypatch):
        # world_size == 1 makes the division a no-op, so it would be tempting to
        # let it through. That would mean an arm loads fine on a dev box and
        # raises only on the cluster -- the worst possible place to find out.
        monkeypatch.setattr(tl, "resolve_run_topology", _single_gpu_topology)
        with pytest.raises(ConfigurationError):
            tl.resolve_iteration_budget(30000, _budget_config("global"))

    def test_unknown_scope_raises_rather_than_defaulting(self, monkeypatch):
        # Unreachable while the schema Literal holds -- this pins that a drift
        # between schema and consumer fails loud (non-negotiable 3).
        monkeypatch.setattr(tl, "resolve_run_topology", _single_gpu_topology)
        with pytest.raises(ConfigurationError, match="Unknown"):
            tl.resolve_iteration_budget(30000, _budget_config("per-rank"))

    def test_the_loop_actually_calls_the_resolver(self):
        # The helper is only worth anything if the loop bound goes through it.
        assert _plain_calls(tl._execute_training_loop, "resolve_iteration_budget")


class TestSchedulerCadenceReadsTheRequestedValue:
    """``gradient_accumulation_steps`` is negotiated to 1 by DeepSpeed."""

    def test_prefers_requested_over_negotiated(self):
        # This IS the bug: DeepSpeedStepPolicy.owns_gradient_accumulation makes
        # the executor set the negotiated value to 1, because DeepSpeed
        # accumulates internally. Reading it for CADENCE stepped the LR
        # scheduler on every iteration instead of every 2nd -- the cosine curve
        # ran at double rate under DeepSpeed only.
        strategy = SimpleNamespace(
            step_executor=SimpleNamespace(
                requested_gradient_accumulation_steps=2,
                gradient_accumulation_steps=1,
            )
        )
        assert tl.resolve_scheduler_cadence(strategy, _budget_config()) == 2

    def test_falls_back_to_negotiated_when_requested_is_absent(self):
        # Non-DeepSpeed executors do not carry the requested_ name; there the
        # negotiated value IS the configured one.
        strategy = SimpleNamespace(
            step_executor=SimpleNamespace(gradient_accumulation_steps=4)
        )
        assert tl.resolve_scheduler_cadence(strategy, _budget_config()) == 4

    def test_falls_back_to_the_config_without_an_executor(self):
        strategy = SimpleNamespace()
        config = _budget_config(accumulation_steps=3)
        assert tl.resolve_scheduler_cadence(strategy, config) == 3

    def test_never_returns_zero(self):
        # A 0 cadence would make `iteration % _gas` a ZeroDivisionError deep in
        # the loop, thousands of iterations in.
        strategy = SimpleNamespace(
            step_executor=SimpleNamespace(requested_gradient_accumulation_steps=0)
        )
        assert tl.resolve_scheduler_cadence(strategy, _budget_config()) == 1

    def test_the_loop_uses_the_helper_not_a_raw_attribute_read(self):
        assert _plain_calls(tl._execute_training_loop, "resolve_scheduler_cadence")
        source = inspect.getsource(tl._execute_training_loop)
        # The old spelling must not come back by copy-paste.
        assert "step_executor.gradient_accumulation_steps" not in source


# W5: the metrics CSV must yield rows, and must not promise columns it cannot
# write. `_execute_training_loop` is far too large to invoke, so these tests
# lift the real predicate out of the real source and EVALUATE it, supplying only
# the primitives (`iteration`, `first_iteration`, `max_iterations`,
# `log_interval`). A test that hard-coded `is_first_iteration` would be a mirror
# of the implementation; this one is not -- the source computes it.
# ---------------------------------------------------------------------------


def _execute_loop_ast() -> ast.AST:
    """Parse ``_execute_training_loop``'s source into an AST."""
    from spectramr.pipelines import training_loop

    src = textwrap.dedent(inspect.getsource(training_loop._execute_training_loop))
    return ast.parse(src)


def _find_guarded_block(tree: ast.AST, *, test_contains: str) -> ast.If:
    """The single ``if`` in ``tree`` whose *test* mentions ``test_contains``."""
    hits = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and test_contains in ast.unparse(node.test)
    ]
    assert len(hits) == 1, (
        f"expected exactly one `if` whose test mentions {test_contains!r}, "
        f"found {len(hits)}: {[ast.unparse(h.test) for h in hits]}"
    )
    return hits[0]


def _eval_logging_gate(
    *, iteration: int, first_iteration: int, max_iterations: int, log_interval: int
) -> bool:
    """Run the REAL logging gate, including the two flags it reads.

    Rebuilds the three real statements (`is_first_iteration = ...`,
    `is_last_iteration = ...`, and the gate's own test) from source via
    ``ast.unparse`` and executes them, so the assertion below is about the
    shipped condition rather than a restatement of it.
    """
    tree = _execute_loop_ast()
    gate = _find_guarded_block(tree, test_contains="iteration % log_interval")

    assigns = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (
                "is_first_iteration",
                "is_last_iteration",
            ):
                assigns[target.id] = ast.unparse(node.value)
    # Deliberately NOT asserting the flags exist. If the gate narrows back to the
    # bare modulo, the flags vanish and this still evaluates -- so the failure the
    # class reports is behavioural ("iteration 1 was not logged") rather than
    # structural ("a variable I expected is missing"). A structural failure would
    # also fire on a harmless rename, which is a false alarm, not a regression.
    program = "".join(f"{name} = {expr}\n" for name, expr in sorted(assigns.items()))
    program += f"__fired__ = bool({ast.unparse(gate.test)})\n"
    namespace: dict[str, object] = {
        "iteration": iteration,
        "first_iteration": first_iteration,
        "max_iterations": max_iterations,
        "log_interval": log_interval,
    }
    exec(compile(program, "<logging-gate>", "exec"), namespace)
    return bool(namespace["__fired__"])


class TestEveryRunYieldsAnInterpretableCurve:
    """A run shorter than `logging.intervals.log` wrote ZERO metric rows.

    Measured on `experiment_11_attention_none`: `logging.intervals.log: 5000`
    against a run of a few dozen steps produced `logs/training_metrics.csv` with
    a 29-column header and no data rows, plus an entirely empty train scalar set
    in TensorBoard -- while the run exited reporting success. The periodic gate
    was satisfied exactly zero times, by design.
    """

    def test_a_run_shorter_than_the_interval_still_logs_two_points(self):
        """The exact failure: 40 iterations, interval 5000."""
        fired = [
            it
            for it in range(1, 41)
            if _eval_logging_gate(
                iteration=it,
                first_iteration=1,
                max_iterations=40,
                log_interval=5000,
            )
        ]
        assert fired == [1, 40], (
            f"a 40-step run at log_interval=5000 logged iterations {fired}; "
            "the first and last must always be logged"
        )

    def test_the_periodic_cadence_is_preserved(self):
        """Widening the gate must not disturb the normal case."""
        fired = [
            it
            for it in range(1, 21)
            if _eval_logging_gate(
                iteration=it, first_iteration=1, max_iterations=20, log_interval=5
            )
        ]
        assert fired == [1, 5, 10, 15, 20]

    def test_the_first_iteration_of_a_resumed_run_is_logged(self):
        """`iteration` starts at `start_iteration + 1`, not at 1.

        A resumed run whose first iteration is 501 must log 501, not wait for the
        next multiple of the interval -- otherwise resuming is exactly the case
        that loses its opening data point.
        """
        assert _eval_logging_gate(
            iteration=501, first_iteration=501, max_iterations=600, log_interval=1000
        )
        assert not _eval_logging_gate(
            iteration=502, first_iteration=501, max_iterations=600, log_interval=1000
        )

    def test_the_final_iteration_is_logged_when_it_is_not_a_multiple(self):
        assert _eval_logging_gate(
            iteration=97, first_iteration=1, max_iterations=97, log_interval=10
        )

    def test_the_gate_is_not_vacuously_true(self):
        """Anti-vacuity: an ordinary interior iteration must still be skipped.

        Without this, a gate that had degenerated to `True` would satisfy every
        other assertion in this class.
        """
        assert not _eval_logging_gate(
            iteration=13, first_iteration=1, max_iterations=100, log_interval=10
        )


class TestTheShortRunWarningNamesTheMisconfiguration:
    """The empty curve was silent. Fixing the gate is not enough: a budget under
    the cadence still yields a two-point curve, which is a config mistake.
    """

    def _guard(self) -> ast.If:
        return _find_guarded_block(
            _execute_loop_ast(), test_contains="max_iterations < log_interval"
        )

    def test_the_guard_fires_only_when_the_budget_is_under_the_cadence(self):
        guard = ast.unparse(self._guard().test)
        program = f"__fired__ = bool({guard})\n"

        def fires(max_iterations: int, log_interval: int) -> bool:
            ns: dict[str, object] = {
                "max_iterations": max_iterations,
                "log_interval": log_interval,
            }
            exec(compile(program, "<short-run-guard>", "exec"), ns)
            return bool(ns["__fired__"])

        assert fires(40, 5000), "the measured failing arm must warn"
        assert not fires(5000, 100), "a normal run must stay quiet"
        assert not fires(100, 100), "budget == cadence logs at least once; no warning"
        assert not fires(40, 0), (
            "log_interval=0 must not reach the modulo comparison; the guard has "
            "to short-circuit or `iteration % 0` raises ZeroDivisionError"
        )

    def test_it_warns_rather_than_informs(self):
        """`LoggingService.setup` clamps every logger to `logging.sinks.level`,
        which is `warning` on the arms this was found on -- an INFO here would be
        discarded precisely where it is needed.
        """
        calls = [
            node
            for node in ast.walk(self._guard())
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("warning", "info", "debug", "error")
        ]
        assert calls, "the short-run guard emits nothing"
        assert {c.func.attr for c in calls} == {"warning"}, (
            f"the short-run guard must warn, not inform: {sorted({c.func.attr for c in calls})}"
        )


class TestTheTrainingCsvPromisesOnlyColumnsItCanWrite:
    """`training_metrics.csv` declared `val_*` columns this writer never fills.

    The row is `{"iteration", "epoch", **losses_scalar}` and `losses_scalar`
    derives from the TRAINING step's `losses_history`; validation metrics go to
    their own file. Measured across every populated training CSV in
    `tests_experiments/`: 70 files, 20,959 data rows, **zero** `val_*` cells
    populated. Same defect class as #340 -- a header for a column no code path
    can fill.
    """

    def _header_source(self) -> str:
        """The header-construction region, up to the `writeheader()` call."""
        from spectramr.pipelines import training_loop

        src = textwrap.dedent(inspect.getsource(training_loop._execute_training_loop))
        head, sep, _ = src.partition("writer.writeheader()")
        assert sep, "the header write moved; this test can no longer locate it"
        return head

    def test_no_val_prefixed_column_is_added_to_the_header(self):
        import re

        head = self._header_source()
        offenders = re.findall(r'f?"val_[\w{}]*"', head)
        assert not offenders, (
            f"the training CSV header adds val_* columns again: {offenders}. "
            "This writer cannot populate them (0 of 20,959 rows corpus-wide); "
            "validation metrics belong to validation_csv_for()'s file."
        )

    def test_train_prefixed_columns_are_still_added(self):
        """Anti-vacuity: the previous test passes trivially if the header stopped
        adding metric columns altogether."""
        head = self._header_source()
        assert 'f"train_{metric_name}"' in head, (
            "the header no longer derives train_* columns from the metric names"
        )


# ---------------------------------------------------------------------------
# W9 -- the raise must not blame a correct monitor.
#
# The Aug-2026 `experiment_11_attention_none` run died on
#   Early-stopping monitor 'val_hfen_mean' is not among the validation metrics
#   this arm produces.  available: ['validation_error']
# and its own message told the user to *"Fix `early_stopping.metric`"*. That
# instruction was wrong: all eight `attention_shootout` arms are correct.
# `val_hfen_mean` is synthesized by averaging the per-level `val_hfen_{accel}x`
# entries (`diffusion.py:3543-3550`), `hfen` IS in `metrics.compute`, and the
# resolver already tries the prefixed spelling, the bare one and every cascading
# suffix. `{"validation_error": 1.0}` is a bare failure SENTINEL returned when
# `hr_fakes is None` -- the monitor was absent because validation had FAILED.
#
# Fatal stays fatal (the #178 rationale is untouched). Only attribution changes.
# ---------------------------------------------------------------------------


class TestTheFailureSentinelIsNotReadAsAMetricVocabulary:
    def _sentinel_message(self) -> str:
        from spectramr.pipelines.training_loop import _unresolvable_monitor_error

        return _unresolvable_monitor_error(
            "val_hfen_mean", {"validation_error": 1.0}, 8
        )

    def test_it_says_validation_failed_rather_than_the_metric_is_absent(self):
        msg = self._sentinel_message()
        assert "Validation FAILED" in msg
        assert "no metrics at all" in msg

    def test_it_tells_the_user_not_to_edit_the_monitor(self):
        """The regression that cost the most: a correct YAML sent for editing."""
        msg = self._sentinel_message()
        assert "val_hfen_mean" in msg, "the monitor is still worth naming"
        assert "this is NOT the problem" in msg
        assert "do not change it" in msg
        assert "Fix `early_stopping.metric`" not in msg, (
            "the sentinel branch must not carry the edit-the-YAML instruction; "
            "that is what sent the user after a metric that was correct"
        )

    def test_it_points_at_the_upstream_failure_by_its_log_string(self):
        msg = self._sentinel_message()
        # Generation errors now propagate (#1152), so the cascade is the only
        # remaining producer of the sentinel and the string to grep for.
        assert "[Cascading Validation]" in msg
        assert "Validation generation failed" not in msg

    def test_it_stays_fatal_and_says_why(self):
        msg = self._sentinel_message()
        assert "Fatal rather than skipped" in msg
        assert "no best checkpoint" in msg
        assert "reporting success" in msg


class TestAGenuinelyAbsentMonitorStillGetsTheOriginalMessage:
    """Anti-vacuity. The sentinel branch must not swallow the real YAML defect
    -- #178's raise is the reason best-checkpoint failures are catchable."""

    def _absent_message(self) -> str:
        from spectramr.pipelines.training_loop import _unresolvable_monitor_error

        return _unresolvable_monitor_error(
            "val_typo", {"val_psnr": 30.0, "val_ssim": 0.9}, 100
        )

    def test_it_still_instructs_the_user_to_fix_the_config(self):
        msg = self._absent_message()
        assert "Fix `early_stopping.metric`" in msg
        assert "Validation FAILED" not in msg

    def test_it_lists_what_was_tried_and_what_was_available(self):
        msg = self._absent_message()
        assert "val_psnr" in msg and "val_ssim" in msg
        assert "tried:" in msg and "available:" in msg

    def test_a_sentinel_alongside_real_metrics_is_a_genuine_absence(self):
        """The sentinel branch keys on `val_metrics` being EXACTLY the sentinel.
        A run that produced metrics AND an error entry has a real vocabulary, so
        an unresolvable monitor there IS a config defect."""
        from spectramr.pipelines.training_loop import _unresolvable_monitor_error

        msg = _unresolvable_monitor_error(
            "val_typo", {"validation_error": 1.0, "val_psnr": 30.0}, 100
        )
        assert "Fix `early_stopping.metric`" in msg
        assert "Validation FAILED" not in msg


class TestTheLoopRaisesThroughTheHelper:
    """The helper is only worth testing if the loop actually routes through it."""

    def test_the_unresolved_branch_calls_the_message_helper(self):
        import inspect

        from spectramr.pipelines import training_loop as training_loop_mod

        src = inspect.getsource(training_loop_mod._execute_training_loop)
        assert "_unresolvable_monitor_error(" in src, (
            "the loop no longer builds its message through the helper these "
            "tests assert on"
        )
        assert "Fix `early_stopping.metric`" not in src, (
            "the message was inlined back into the loop, where the sentinel "
            "branch cannot be reached by a unit test"
        )


# ---------------------------------------------------------------------------
# validation-interval budget guard
# ---------------------------------------------------------------------------
#
# Found on a 4-GPU run of `experiment_11_attention_none` overridden to
# `training.max_iterations=5000` while the arm declares
# `validation.schedule.interval_steps: 5000`. The step gate is a bare
# `iteration % eval_interval == 0` with no first/last force, so that override
# collapsed 6 validation events into 1 -- on the final iteration. A
# deterministic, weight-independent mask-schedule failure was therefore not
# discovered until 5 h 17 m of 4-GPU compute had been spent, and no
# best checkpoint was written.
#
# One notch lower (`max_iterations=4000`) the same arm produces ZERO validation
# events and exits 0 reporting success -- early stopping never evaluates, no
# best checkpoint exists, and the fatal unresolvable-monitor RuntimeError that
# #178 installed is unreachable because it lives INSIDE a validation event.
#
# These drive the real `_execute_training_loop`, so they pin the guard's
# placement (before the loop, after the loaders are bound) and not just the
# arithmetic -- that lives in `tests/unit/pipelines/test_train.py`.


def _guard_config(
    max_iterations,
    interval_steps,
    *,
    on_epoch=False,
    interval_epochs=1,
    with_validation=True,
):
    validation = (
        SimpleNamespace(
            schedule=SimpleNamespace(
                interval_steps=interval_steps,
                on_epoch=on_epoch,
                interval_epochs=interval_epochs,
            )
        )
        if with_validation
        else None
    )
    return SimpleNamespace(
        data=None,
        training=SimpleNamespace(max_iterations=max_iterations, epochs=None),
        logging=LoggingConfigSchema(intervals={"log": 100}),
        metrics=SimpleNamespace(train_metric_interval=100),
        validation=validation,
    )


def _guard_pipeline(*, train_len=10, has_val=True):
    train_loader = MagicMock()
    train_loader.__len__ = lambda self: train_len
    loaders = {"train": train_loader}
    if has_val:
        loaders["val"] = MagicMock()
    pipeline = MagicMock()
    pipeline.data_loaders = loaders
    return pipeline


def _guard_verdict(config, pipeline, *, strategy=None, is_sanity_check=False):
    """Enter `_execute_training_loop` and return whatever exception escaped.

    Anything past the guard needs a real strategy, so the loop is expected to
    die of a mock artefact once it gets there. That makes the interesting
    question "was it a `ConfigurationError`?", not "did it raise?" -- hence a
    returned verdict rather than `pytest.raises(Exception)`, which would pass
    on any mock artefact and assert nothing.
    """
    try:
        tl._execute_training_loop(
            strategy=strategy if strategy is not None else MagicMock(),
            pipeline=pipeline,
            config=config,
            model_type="unet",
            output_paths=None,
            is_sanity_check=is_sanity_check,
        )
    except Exception as exc:  # the verdict IS the exception
        return exc
    return None


class TestValidationIntervalBudgetGuard:
    def test_an_interval_above_the_budget_raises_before_training_starts(self):
        verdict = _guard_verdict(_guard_config(4000, 5000), _guard_pipeline())
        assert isinstance(verdict, ConfigurationError), repr(verdict)
        message = str(verdict)
        assert "validation.schedule.interval_steps=5000" in message
        assert "4000" in message
        # The message must name BOTH losses, not just "validation is off":
        # an operator who reads only "no validation" may accept it, while
        # "no best checkpoint" is what makes the run unusable.
        assert "best checkpoint" in message
        assert "early stopping" in message

    def test_the_raise_precedes_any_training_step(self):
        """The whole point is failing at t=0 rather than after the budget."""
        strategy = MagicMock()
        verdict = _guard_verdict(
            _guard_config(4000, 5000), _guard_pipeline(), strategy=strategy
        )
        assert isinstance(verdict, ConfigurationError), repr(verdict)
        assert not strategy.training_step.called
        assert not strategy.train_step.called

    def test_an_interval_inside_the_budget_clears_the_guard(self):
        """The arm as DECLARED (30000/5000, 6 events) must not be rejected."""
        verdict = _guard_verdict(_guard_config(30000, 5000), _guard_pipeline())
        assert not isinstance(verdict, ConfigurationError), str(verdict)

    def test_a_run_without_a_val_loader_is_not_rejected(self):
        """No validation loader means no validation was expected; the interval
        being unreachable is then not a defect, and raising would break every
        reconstruction-only arm that ships a validation block it cannot use."""
        verdict = _guard_verdict(
            _guard_config(4000, 5000), _guard_pipeline(has_val=False)
        )
        assert not isinstance(verdict, ConfigurationError), str(verdict)

    def test_epoch_mode_supplies_the_events_and_clears_the_guard(self):
        verdict = _guard_verdict(
            _guard_config(4000, 5000, on_epoch=True), _guard_pipeline(train_len=10)
        )
        assert not isinstance(verdict, ConfigurationError), str(verdict)

    def test_epoch_mode_with_an_epoch_longer_than_the_budget_still_raises(self):
        verdict = _guard_verdict(
            _guard_config(4000, 5000, on_epoch=True), _guard_pipeline(train_len=99999)
        )
        assert isinstance(verdict, ConfigurationError), repr(verdict)

    def test_interval_equal_to_the_budget_warns_and_names_the_cost(self, caplog):
        """The observed run. Legal, so not fatal -- but the single event lands
        on the final iteration, which voids the cost argument the #178 guard
        states verbatim ("the cost of a typo'd monitor is one validation pass
        rather than the whole budget")."""
        with caplog.at_level(logging.WARNING):
            verdict = _guard_verdict(_guard_config(5000, 5000), _guard_pipeline())
        assert not isinstance(verdict, ConfigurationError), str(verdict)
        # `.getMessage()`, not `.message`: a handler-stripping test elsewhere in
        # the suite makes `.message` vanish in wide runs (#1290).
        warnings = [
            r.getMessage()
            for r in caplog.records
            if "validation.schedule.interval_steps" in r.getMessage()
        ]
        assert warnings, "the once-on-the-last-iteration case warned about nothing"
        assert "ONCE" in warnings[0]
        assert "5000" in warnings[0]

    def test_a_config_without_a_validation_block_does_not_warn(self, caplog):
        """`eval_interval` FALLS BACK to `max_iterations` when the block is
        absent, so an unguarded `==` test would fire the warning on every
        validation-less arm -- about a value the user never wrote."""
        with caplog.at_level(logging.WARNING):
            _guard_verdict(
                _guard_config(5000, 5000, with_validation=False), _guard_pipeline()
            )
        assert not [
            r for r in caplog.records
            if "validation.schedule.interval_steps" in r.getMessage()
        ]

    def test_a_healthy_interval_warns_about_nothing(self, caplog):
        with caplog.at_level(logging.WARNING):
            _guard_verdict(_guard_config(30000, 5000), _guard_pipeline())
        assert not [
            r for r in caplog.records
            if "validation.schedule.interval_steps" in r.getMessage()
        ]


class TestTheGuardDoesNotVetoASanityCheck:
    """Sanity-check mode imposes its OWN budget, so it cannot veto on budget.

    `is_sanity_check` overwrites `max_iterations` with 5000 (training_loop.py
    ~line 671) AFTER any `--override/-O` override is applied. So an arm whose
    declared pair is internally consistent -- `max_iterations: 15000` with
    `interval_steps: 15000` -- is made inconsistent purely by the mode, and the
    user has no override that fixes it: the mode overwrites the very field they
    would override. A corpus scan on 2026-08-21 put 57 of the 647
    `experiments/inprogress/` arms in exactly this position.

    Without these tests the guard reads as correct and quietly turns a
    diagnostic mode into a mode that refuses to start.
    """

    def test_a_sanity_check_is_not_blocked_by_the_modes_own_budget(self):
        # Declared 15000/15000 is consistent; the mode compresses to 5000.
        verdict = _guard_verdict(
            _guard_config(15000, 15000), _guard_pipeline(), is_sanity_check=True
        )
        assert not isinstance(verdict, ConfigurationError), (
            "sanity-check mode forced max_iterations to 5000 and then rejected "
            "the arm for not matching a budget the arm never declared"
        )

    def test_the_same_config_outside_sanity_mode_is_accepted(self):
        # The control: 15000/15000 is the WARN case, never the raise case, so
        # the test above cannot pass merely because the config is healthy.
        verdict = _guard_verdict(_guard_config(15000, 15000), _guard_pipeline())
        assert not isinstance(verdict, ConfigurationError)

    def test_a_config_that_is_unreachable_on_its_own_budget_still_raises(self):
        # And the exemption is scoped to sanity mode only -- 4000/5000 is
        # unreachable on the arm's OWN declared budget and must still be fatal.
        verdict = _guard_verdict(_guard_config(4000, 5000), _guard_pipeline())
        assert isinstance(verdict, ConfigurationError)

    def test_the_skipped_validation_is_still_reported(self, caplog):
        # Not blocking is not the same as staying silent: a sanity check that
        # silently skips validation cannot catch a val-time defect, which is
        # most of what a sanity check is for.
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(
                _guard_config(15000, 15000), _guard_pipeline(), is_sanity_check=True
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "SANITY CHECK MODE" in messages
        assert "validation will NOT run at all" in messages
        # It must attribute the budget to the mode, not to the arm -- the
        # operator's next action differs completely between those two readings.
        assert "not a defect in the arm" in messages


class TestSanityModeStillNamesADefectiveArm:
    """The mode-attributes-the-budget message must not become a false all-clear.

    Demoting the raise to a warning inside sanity mode was correct, but the
    message it emitted asserted "not a defect in the arm" unconditionally. For
    an arm whose interval also exceeds the budget it DECLARED, that sentence is
    false in the most costly direction: the operator reads an all-clear on the
    one occasion the cheap diagnostic surfaced a real defect early, and meets
    the raise later on a full-length run instead. Three `inprogress/` arms are
    in exactly that state (#1305), and they are the arms most likely to be run
    under a sanity check first.
    """

    def test_an_arm_defective_on_its_own_budget_is_not_given_an_all_clear(
        self, caplog
    ):
        # 10000/1500 is `experiment_130_ti_ccd`'s real shape: unreachable on the
        # arm's own declared budget, and unreachable again under the mode's 5000.
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(
                _guard_config(1500, 10000), _guard_pipeline(), is_sanity_check=True
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "not a defect in the arm" not in messages, (
            "sanity mode told the operator the arm was fine while the arm's own "
            "declared budget of 1500 can never reach interval_steps=10000"
        )
        assert "OWN declared budget" in messages
        assert "fatal outside" in messages

    def test_the_advice_is_scaled_to_the_declared_budget_not_the_modes(
        self, caplog
    ):
        # The fix the operator must make lives in the YAML, so the number they
        # are given has to fit the budget the YAML declares (1500 -> 750), not
        # the transient 5000 the mode imposed. Advice sized to the mode's budget
        # (2500) would still leave the arm fatal on a real run.
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(
                _guard_config(1500, 10000), _guard_pipeline(), is_sanity_check=True
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "at most 750" in messages
        assert "at most 2500" not in messages

    def test_an_arm_that_is_fine_on_its_own_budget_still_gets_the_all_clear(
        self, caplog
    ):
        # The control that keeps the assertion above honest: the 57-arm case
        # (15000/15000, consistent as declared) must KEEP the attribution to the
        # mode. If this regressed, the branch would simply be calling every
        # sanity skip a defect, which is the bug in the other direction.
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(
                _guard_config(15000, 15000), _guard_pipeline(), is_sanity_check=True
            )
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "not a defect in the arm" in messages
        assert "OWN declared budget" not in messages


class TestTheDegenerateWarningDoesNotEatItself:
    """`max_iterations == 1` is the fixed point where the advice is circular.

    The warning recommends `max_iterations // 2`, which floors to 1 at a budget
    of 1 -- i.e. it tells the operator to set `interval_steps` to the value it
    already holds. 13 of the 647 `inprogress/` arms are exactly this shape
    (measured 2026-08-21), all deliberate 1-step eval arms, so this is not a
    hypothetical: it is a warning that would fire on real arms with advice they
    have already taken.
    """

    def test_a_single_iteration_arm_is_not_warned_at(self, caplog):
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(_guard_config(1, 1), _guard_pipeline(train_len=1))
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "runs exactly ONCE" not in messages, (
            "warned a 1-iteration arm to shorten an interval it cannot shorten"
        )

    def test_a_single_iteration_arm_is_not_rejected_either(self):
        verdict = _guard_verdict(_guard_config(1, 1), _guard_pipeline(train_len=1))
        assert not isinstance(verdict, ConfigurationError)

    def test_a_two_iteration_arm_still_warns(self, caplog):
        # The boundary immediately above: at a budget of 2 the advice becomes
        # actionable (`interval_steps: 1` buys a real early event), so the
        # suppression must stop here and not one notch later.
        caplog.set_level(logging.WARNING)
        with caplog.at_level(logging.WARNING):
            _guard_verdict(_guard_config(2, 2), _guard_pipeline(train_len=1))
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "runs exactly ONCE" in messages


_LOOP_LOGGER = "spectramr.pipelines.training_loop"


def _banner(config, caplog, *, sanity=False):
    """Drive the real `_execute_training_loop` and return its launch banner."""
    with caplog.at_level(logging.INFO, logger=_LOOP_LOGGER):
        _guard_verdict(config, _guard_pipeline(), is_sanity_check=sanity)
    lines = [
        r.getMessage()
        for r in caplog.records
        if "Starting training for" in r.getMessage()
    ]
    assert len(lines) == 1, f"expected exactly one launch banner, got {lines}"
    return lines[0]


class TestTheLaunchBannerNamesTheBudgetSource:
    """`max_iterations` has three producers that disagree silently, and the
    banner used to print only the number.

    The run this comes from: `experiment_11_attention_none` was launched on
    4 GPUs with `-O training.max_iterations=5000`, and sanity-check mode
    independently forces the budget to a hardcoded 5000
    (`training_loop.py`, `if is_sanity_check:`). Both print
    "Starting training for 5000 iterations", so days later the log could not
    settle which had been in effect -- and the two runs are not comparable:
    one is the operator's budget for a real run, the other a single-batch
    overfit probe that also disables the epoch-validation escape hatch.

    These drive the real `_execute_training_loop`, so they pin the banner
    where an operator actually reads it, not a helper in isolation.
    """

    def test_a_declared_budget_is_attributed_to_the_config_file(self, caplog):
        line = _banner(_guard_config(30000, 5000), caplog)
        assert "30000 iterations" in line
        assert "declared in the config file" in line

    def test_an_overridden_budget_is_attributed_to_the_override(self, caplog):
        """The case that motivated this. `_guard_config` builds a
        `SimpleNamespace`, so the record has to be set the way
        `apply_overrides` sets it on a real `TrainingSettings`."""
        config = _guard_config(5000, 5000)
        config._override_paths = ("training.max_iterations",)
        line = _banner(config, caplog)
        assert "5000 iterations" in line
        assert "--override" in line
        assert "declared in the config file" not in line

    def test_an_epoch_derived_budget_names_both_of_its_factors(self, caplog):
        """Neither of the other two descriptions fits: the arm declared no
        usable `max_iterations` and the number is `epochs x loader length`.
        Saying "declared in the config file" would send a reader hunting for a
        `max_iterations:` key that is not there."""
        line = _banner(_guard_config(-1, 5000), caplog)
        # epochs defaults to 100, `_guard_pipeline` gives a 10-batch loader
        assert "1000 iterations" in line
        assert "training.epochs=100" in line
        # `"10" in line` would be vacuous -- the "1000 iterations" above already
        # contains it. The loader length is the factor a reader cannot derive.
        assert "length 10" in line

    def test_sanity_mode_names_itself_and_the_budget_it_clobbered(self, caplog):
        """The forcing happens AFTER `--override/-O` is applied, so a caller who
        set `training.max_iterations` watches their value silently vanish. The
        banner has to name both halves, or the two independent 5000s stay
        indistinguishable -- which is the whole failure this closes."""
        config = _guard_config(5000, 5000)
        config._override_paths = ("training.max_iterations",)
        line = _banner(config, caplog, sanity=True)
        assert "sanity-check mode" in line
        assert "--override" in line, (
            "the clobbered budget's own source must survive into the message; "
            "without it the operator cannot tell their -O was overridden"
        )

    def test_sanity_mode_reports_the_declared_value_it_replaced(self, caplog):
        """Distinct from the case above: here the 30000 came from the config
        file, and the number the mode replaced is the interesting one."""
        line = _banner(_guard_config(30000, 5000), caplog, sanity=True)
        assert "5000 iterations" in line
        assert "sanity-check mode" in line
        assert "30000" in line

    def test_the_banner_still_reports_the_iteration_count(self, caplog):
        """Negative control. The annotation must not displace the number the
        banner existed to print; every downstream reader (and every operator
        eyeballing a log) reads that first."""
        for declared in (1, 40, 30000):
            caplog.clear()
            assert f"for {declared} iterations" in _banner(
                _guard_config(declared, 1), caplog
            )


# ---------------------------------------------------------------------------
# Prose-vs-argparse: no message may name a flag the CLI does not accept.
# ---------------------------------------------------------------------------

#: Flags belonging to OTHER tools that this module legitimately names in prose.
#: Each entry is a promise that the flag is real *somewhere else*, so the gate
#: stays a gate rather than becoming a list of everything it happens to find.
_EXTERNAL_TOOL_FLAGS = {
    "--nproc_per_node",  # torchrun, in the distributed-launch guidance
}


def _flag_tokens_in_module_strings(module) -> dict[str, int]:
    """Every CLI-flag-looking token in every string literal of *module*.

    Walks the AST rather than the raw text so an f-string split across four
    source lines is still read as the one sentence an operator sees, and so
    ``--`` used as an em-dash in prose (very common in this file) is not
    mistaken for a flag -- the pattern requires a letter to follow.
    """
    source = pathlib.Path(inspect.getfile(module)).read_text()
    pattern = re.compile(r"(?<![\w-])(--[A-Za-z][\w-]*|-[A-Za-z])(?![\w-])")
    found: dict[str, int] = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for match in pattern.finditer(node.value):
                found[match.group(1)] = found.get(match.group(1), 0) + 1
    return found


def _real_cli_flags() -> set[str]:
    """Every option string the real CLI parser accepts, across all subcommands."""
    from spectramr.cli.app import build_parser

    flags: set[str] = set()
    stack = [build_parser()]
    while stack:
        parser = stack.pop()
        for action in parser._actions:
            flags.update(action.option_strings)
            choices = getattr(action, "choices", None)
            if isinstance(choices, dict):
                stack.extend(c for c in choices.values() if hasattr(c, "_actions"))
    return flags


def test_no_message_in_this_module_names_a_flag_the_cli_rejects() -> None:
    """An error message that names a non-existent flag is worse than one that
    names none: it is read at the exact moment an operator is trying to recover
    a failed run, and it sends them to type something that fails too.

    This is the same defect ``test_validation.py`` pins for the ``interval_steps``
    schema description -- and it shipped a second time, in this module, in the
    same change that installed that test: the unreachable-interval
    ``ConfigurationError`` told operators to use ``--set/-o`` when the CLI only
    ever accepted ``--override``/``-O``. The first test could not catch it
    because it reads exactly one Pydantic field description; a raise message is
    not a field description. So the gate is widened to the whole module, where
    the *next* message to be added is covered before anyone remembers to ask.

    Known limit, stated rather than papered over: this catches a flag that does
    not exist (``--set``), not a flag that exists and means something else. The
    original message also said ``-o``, which IS real -- it is ``infer --output``
    -- and would survive this gate. Catching that needs per-flag semantics the
    parser cannot supply.
    """
    named = _flag_tokens_in_module_strings(tl)
    assert named, "the flag scan found no strings at all; the walk is broken"

    real = _real_cli_flags()
    assert "--override" in real and "-O" in real, (
        "the override flag was renamed; this test's own anchor is stale"
    )

    unknown = set(named) - real - _EXTERNAL_TOOL_FLAGS
    assert not unknown, (
        f"training_loop.py names CLI flags that do not exist: {sorted(unknown)}. "
        f"An operator reading one of these messages would type a command that "
        f"fails. Real override flags: {sorted(f for f in real if 'over' in f)}."
    )


# ---------------------------------------------------------------------------
# EMA warmup gate (#1294).
#
# ``config.ema.warmup_steps`` is the length of the EMA warmup period on both
# paths, but the standard path implements it as a hard update-delay gate here
# in the loop, while the adaptive path implements it as a decay ramp inside
# ModelEma. Applying both would double the period; applying neither leaves the
# knob inert — which is what happened for months, because the loop zeroed the
# gate whenever ``enable_adaptive_ema`` was set and no adaptive ramp existed to
# take over.
# ---------------------------------------------------------------------------


class _Ema:
    def __init__(self, adaptive: bool):
        self.adaptive = adaptive


def test_standard_ema_gets_the_hard_warmup_gate():
    cfg = SimpleNamespace(warmup_steps=2000)
    assert tl.resolve_ema_warmup_gate(_Ema(adaptive=False), cfg) == 2000


def test_adaptive_ema_gets_no_hard_gate():
    """The ramp inside ModelEma already spends warmup_steps; stacking the gate
    on top would make the effective warmup 2x the declared value."""
    cfg = SimpleNamespace(warmup_steps=2000)
    assert tl.resolve_ema_warmup_gate(_Ema(adaptive=True), cfg) == 0


def test_gate_reads_the_object_not_the_config():
    """The regression guard.

    A config that DECLARES adaptive EMA but whose builder produced a standard
    ModelEma must still get its warmup honoured — the old config-derived read
    zeroed the gate on a promise the (deleted) adaptive path never kept.
    """
    cfg = SimpleNamespace(warmup_steps=1500, enable_adaptive_ema=True)
    assert tl.resolve_ema_warmup_gate(_Ema(adaptive=False), cfg) == 1500


def test_missing_ema_or_config_is_no_gate():
    assert tl.resolve_ema_warmup_gate(None, None) == 0
    assert tl.resolve_ema_warmup_gate(None, SimpleNamespace()) == 0
    assert tl.resolve_ema_warmup_gate(None, SimpleNamespace(warmup_steps=None)) == 0


# ---------------------------------------------------------------------------
# The best-weights restore in the training loop.
#
# The restore resolves its checkpoint adapter from the runtime it is handed; without
# it a DeepSpeed arm failed to reload its own best checkpoint after an 8h45m run.
# ---------------------------------------------------------------------------


class TestBestRestoreWiringInTheLoop:
    def test_the_best_restore_attaches_the_parallel_runtime(self) -> None:
        (call,) = _method_calls(tl._execute_training_loop, "best_weights", on="checkpoint_service")
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw["parallel"], ast.Name) and kw["parallel"].id == "parallel_runtime"

    def test_the_best_restore_is_handed_the_strategy(self) -> None:
        """Without it a multi-stage run's stages keep their last weights (#2962)."""
        (call,) = _method_calls(tl._execute_training_loop, "best_weights", on="checkpoint_service")
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("strategy"), ast.Name) and kw["strategy"].id == "strategy"

    def test_the_restore_site_can_actually_see_parallel_runtime(self) -> None:
        """The other half: the name must be bound in the enclosing scope.

        Adding the call would otherwise trade a KeyError for a NameError, and
        only on the arms that reach the restore branch.
        """
        tree = _ast_of(tl._execute_training_loop)
        bound = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        assert "parallel_runtime" in bound, (
            "_execute_training_loop calls best_weights but never binds "
            "parallel_runtime; the restore would raise NameError."
        )


# ---------------------------------------------------------------------------
# #1353 — the strategy lifecycle contract is actually driven.
#
# ``BaseTrainingStrategy`` declares ``on_epoch_start`` / ``on_epoch_end`` /
# ``on_validation_start`` / ``on_validation_end``; ``MultiTrainingStrategy`` overrides
# two of them to implement ``end_to_end_finetune_epoch`` and per-stage
# ``early_stopping``; and until this change nothing under ``src/spectramr`` ever
# called any of them (audit dossier D12 §3.1).
#
# The BEHAVIOUR of the dispatch — order, once-per-epoch, the completed-epoch
# index, resume — lives in
# ``tests/unit/infrastructure/training/strategies/test_lifecycle.py``, which
# replays the loop's boundary arithmetic against the driver. What can only be
# checked here is that the loop shell calls that driver, at the right places,
# with the right gates. These are source-level pins for the reason the rest of
# this file is: a full ``_execute_training_loop`` OOM-kills a dev box.
# ---------------------------------------------------------------------------


class TestTheStrategyLifecycleContractIsDriven:
    def test_the_loop_builds_a_lifecycle_driver_before_the_loop(self) -> None:
        src = inspect.getsource(tl._execute_training_loop)
        loop_at = src.index("for iteration in pbar:")
        assert "StrategyLifecycleDriver(strategy)" in src[:loop_at], (
            "the driver must be constructed once above the loop — rebuilding it "
            "per iteration would reset the once-per-epoch state and refire "
            "on_epoch_start every step"
        )

    def test_the_epoch_pair_is_gated_like_epoch_based_validation(self) -> None:
        """Two gates, each load-bearing.

        ``_has_train_loader``: ``train_loader_len`` falls back to 1 for a
        missing/empty loader, which makes ``epoch == iteration`` — ungated,
        ``on_epoch_start`` would fire every single step while
        ``_is_epoch_boundary`` stays permanently False, so no ``on_epoch_end``
        ever balances it.

        ``not is_sanity_check``: the hooks mutate persistent strategy state
        (stage freezes, patience counters); an overfit-one-batch pass must not
        spend an arm's early-stopping budget.
        """
        src = inspect.getsource(tl._execute_training_loop)
        assert "_drive_epoch_hooks = _has_train_loader and not is_sanity_check" in src

    def test_the_epoch_is_opened_before_the_train_step(self) -> None:
        """An unfreeze taken in ``on_epoch_start`` must apply to the epoch's
        first gradient step, not its second."""
        src = inspect.getsource(tl._execute_training_loop)
        begin = src.index("lifecycle.begin_epoch(epoch)")
        step = src.index("strategy.train_step(")
        loop_at = src.index("for iteration in pbar:")
        assert loop_at < begin < step

    def test_the_epoch_is_closed_after_the_validation_block(self) -> None:
        """Per-stage early stopping must score the epoch on the boundary
        validation, not on a mid-epoch measurement."""
        src = inspect.getsource(tl._execute_training_loop)
        run_val = src.index("val_metrics = _run_validation(")
        end = src.index("lifecycle.end_epoch(")
        assert run_val < end

    def test_epoch_end_is_gated_on_the_epoch_boundary(self) -> None:
        src = inspect.getsource(tl._execute_training_loop)
        assert "if _drive_epoch_hooks and is_epoch_end:" in src

    def test_epoch_end_never_receives_a_stale_metrics_dict(self) -> None:
        """``_epoch_end_metrics`` is rebound only by a validation that ran this
        iteration and cleared every iteration. Handing the previous event's
        numbers to a patience counter would count one epoch twice."""
        src = inspect.getsource(tl._execute_training_loop)
        assert "_epoch_end_metrics = val_metrics" in src
        assert "lifecycle.end_epoch(_epoch_end_metrics)" in src
        # ...and cleared UNCONDITIONALLY at loop-body level, immediately after
        # the gated fire — not inside the gate, where a boundary-less iteration
        # would carry the previous validation's numbers forward.
        assert re.search(
            r"\n( +)lifecycle\.end_epoch\(_epoch_end_metrics\)\n        _epoch_end_metrics = \{\}\n",
            src,
        ), "the epoch-end metrics are not cleared at loop-body level after the fire"

        # The bottom-of-body clear is only safe because nothing short-circuits
        # the outer loop between the rebind and it. A `continue` added there
        # would carry one validation's numbers into the next epoch boundary
        # silently — so pin the premise, not just the placement. (The four
        # `break`s are fine: a break leaves the loop, it does not skip an
        # iteration.)
        tree = _ast_of(tl._execute_training_loop)
        outer = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.For) and getattr(n.iter, "id", "") == "pbar"
        )
        inner_loops = [
            n
            for n in ast.walk(outer)
            if isinstance(n, (ast.For, ast.While, ast.AsyncFor)) and n is not outer
        ]
        inner_nodes = [c for loop in inner_loops for c in ast.walk(loop)]
        owned_by_inner = {id(c) for c in inner_nodes if isinstance(c, ast.Continue)}
        stray = [
            c
            for c in ast.walk(outer)
            if isinstance(c, ast.Continue) and id(c) not in owned_by_inner
        ]
        assert not stray, (
            "a `continue` in the outer iteration loop can skip the "
            "_epoch_end_metrics clear, carrying stale validation metrics into "
            "the next epoch boundary"
        )

    def test_the_validation_pair_brackets_the_in_training_validation(self) -> None:
        src = inspect.getsource(tl._execute_training_loop)
        begin = src.index("lifecycle.begin_validation()")
        run_val = src.index("val_metrics = _run_validation(")
        end = src.index("lifecycle.end_validation(val_metrics)")
        assert begin < run_val < end

    def test_the_validation_pair_is_not_gated_on_sanity_check(self) -> None:
        """A sanity run really does validate; a hook silent through a validation
        it did not prevent would misreport what the run did. The epoch pair is
        gated, this one deliberately is not."""
        src = inspect.getsource(tl._execute_training_loop)
        lines = src.splitlines()
        line = next(ln for ln in lines if "lifecycle.begin_validation()" in ln)
        assert line.strip() == "lifecycle.begin_validation()"

    def test_the_standalone_evaluate_drives_the_same_pair(self) -> None:
        """``evaluate()``'s docstring promises metrics that "match in-training
        validation exactly"; a lifecycle hook that fires for one and not the
        other breaks that promise."""
        src = inspect.getsource(TrainingLoop.evaluate)
        assert "StrategyLifecycleDriver(self.strategy)" in src
        assert src.index("lifecycle.begin_validation()") < src.index("_run_validation(")
        assert "lifecycle.end_validation(metrics)" in src

    def test_no_lifecycle_dispatch_is_gated_on_the_main_rank(self) -> None:
        """``MultiTrainingStrategy`` flips ``requires_grad`` on whole stages inside
        these hooks. A rank-0-only dispatch desynchronises DDP parameter groups —
        the failure is a wrong gradient, not an exception."""
        tree = _ast_of(tl._execute_training_loop)
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            test_src = ast.unparse(node.test)
            if "is_main_process" not in test_src:
                continue
            body_src = "\n".join(ast.unparse(st) for st in node.body)
            assert "lifecycle." not in body_src, (
                f"a lifecycle hook is dispatched under `if {test_src}`"
            )
# ---------------------------------------------------------------------------
# #1682 -- the training-CSV column set has ONE owner
#
# `csv.DictWriter(..., extrasaction="ignore")` drops any key with no column, so
# a header resolved from the config alone silently discarded every value whose
# name the config cannot produce. On `experiment_11_attention_none` that was
# `lr_opt_g`, `pre_dc_kspace_l1` and `generator_loss` -- every step, all run.
#
# These tests pin BOTH halves: that each helper resolves the right names, and
# that `_execute_training_loop` actually CALLS it (non-negotiable 16 -- a
# resolver nothing calls is not a fix). The call-site pins walk the AST on
# purpose: a `"resolve_scheduler_lr_columns" in source` assertion is satisfied
# by a comment or a docstring that merely mentions the name.
# ---------------------------------------------------------------------------


def _called_names(source: str) -> set[str]:
    """Bare function names CALLED in ``source``. Pure core, so it can be planted.

    Split out from the pin below for the reason ``_fitness_lib`` splits its own
    cores: a detector that can only be pointed at real production source can
    never be shown a violation, and an unplantable detector is its own kind of
    blind (`.agent/rules/detectors.md` §1).

    Deliberately AST-based and deliberately ``ast.Name`` only:
      * a name inside a comment, a docstring or a plain string literal is NOT a
        call, and must not satisfy the pin -- that is precisely how a
        ``"name" in source`` assertion gets satisfied by prose;
      * an attribute call (``mod.f()``) is a different construct and is not
        counted, which is honest about what this pin does and does not see.
    """
    import ast

    tree = ast.parse(textwrap.dedent(source))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def _membership_guard_names(source: str) -> set[str]:
    """Names used as the CONTAINER of an ``in``/``not in`` test. Pure core."""
    import ast
    import textwrap as _tw

    out: set[str] = set()
    for node in ast.walk(ast.parse(_tw.dedent(source))):
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators, strict=True):
                if isinstance(op, ast.In | ast.NotIn) and isinstance(comp, ast.Name):
                    out.add(comp.id)
    return out


def _attribute_calls(source: str) -> set[str]:
    """``obj.method`` strings for every attribute call in ``source``. Pure core."""
    import ast
    import textwrap as _tw

    out: set[str] = set()
    for node in ast.walk(ast.parse(_tw.dedent(source))):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                out.add(f"{node.func.value.id}.{node.func.attr}")
    return out


def _fstring_prefixes(source: str) -> list[str]:
    """Literal parts of every f-string in ``source``. Pure core, plantable."""
    import ast

    tree = ast.parse(textwrap.dedent(source))
    return [
        "".join(v.value for v in node.values if isinstance(v, ast.Constant))
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
    ]


def _training_loop_source() -> str:
    import inspect

    from spectramr.pipelines.training_loop import _execute_training_loop

    return inspect.getsource(_execute_training_loop)


def _calls_in_training_loop() -> set[str]:
    """Names of every function called directly inside ``_execute_training_loop``."""
    return _called_names(_training_loop_source())


class TestSchedulerLrColumns:
    def test_no_schedulers_yields_no_columns(self):
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import resolve_scheduler_lr_columns

        assert resolve_scheduler_lr_columns(SimpleNamespace(schedulers={})) == frozenset()
        assert resolve_scheduler_lr_columns(SimpleNamespace(schedulers=None)) == frozenset()
        assert resolve_scheduler_lr_columns(SimpleNamespace()) == frozenset()

    def test_column_name_matches_the_emission_site_exactly(self):
        """The resolver and the emitter are two ends of one column name.

        They can no longer diverge by construction: ``lr_column_name`` is the
        single owner of the spelling (non-negotiable 17) and BOTH ends call it,
        so this pins the coupling at its source rather than comparing two
        independent spellings. The original defect -- the emitter writing
        ``lr_opt_g`` while the header promised no ``lr_*`` column at all, so
        ``extrasaction="ignore"`` discarded every learning rate -- is
        unreachable while one function returns the name for both sides.

        The companion pin ``test_neither_site_re_spells_the_column`` asserts
        that owner stays sole; this one asserts the emitter actually calls it.
        """
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import resolve_scheduler_lr_columns

        assert resolve_scheduler_lr_columns(SimpleNamespace(schedulers={"opt_g": object()})) == {
            "lr_opt_g"
        }
        assert resolve_scheduler_lr_columns(
            SimpleNamespace(schedulers={"opt_g": object(), "opt_d": object()})
        ) == {"lr_opt_g", "lr_opt_d"}

        assert "lr_column_name" in _calls_in_training_loop(), (
            "the emission site no longer routes its column name through "
            "lr_column_name(); it is spelling the column itself again, so "
            "resolve_scheduler_lr_columns can start producing dead columns."
        )
        # The owner really does produce the prefix both ends depend on.
        from spectramr.pipelines.training_loop import lr_column_name

        assert _fstring_prefixes(inspect.getsource(lr_column_name)) == ["lr_"]

    def test_production_path_calls_the_resolver(self):
        assert "resolve_scheduler_lr_columns" in _calls_in_training_loop()


class TestProducerDeclaredColumns:
    def test_strategy_without_the_hook_declares_nothing(self):
        from spectramr.pipelines.training_loop import resolve_producer_declared_columns

        assert resolve_producer_declared_columns(object()) == frozenset()

    def test_declared_keys_are_forwarded(self):
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import resolve_producer_declared_columns

        strategy = SimpleNamespace(declared_metric_keys=lambda: frozenset({"pre_dc_kspace_l1"}))
        assert resolve_producer_declared_columns(strategy) == {"pre_dc_kspace_l1"}

    def test_a_forwarding_strategy_declares_its_computers_stamped_keys(self):
        """#2546: the non-finite substitution count needs a CSV column to be seen."""
        from spectramr.pipelines.training_loop import resolve_producer_declared_columns

        class _Forwarding:
            forwards_loss_components = True
            loss_computer = SimpleNamespace(stamped_component_keys=frozenset({"stamped"}))

            def declared_metric_keys(self):
                return frozenset({"own"})

        class _NotForwarding(_Forwarding):
            forwards_loss_components = False

        assert resolve_producer_declared_columns(_Forwarding()) == {"own", "stamped"}
        assert resolve_producer_declared_columns(_NotForwarding()) == {"own"}

    def test_the_strategies_that_forward_components_say_so(self):
        """Each flag matches its step dict: ``update(loss_output.components)``."""
        from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
        from spectramr.infrastructure.training.strategies.diff_siren import DIFFSirenStrategy
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )
        from spectramr.infrastructure.training.strategies.hssc_strategy import HSSCStrategy
        from spectramr.infrastructure.training.strategies.n2n_strategy import (
            NoiseToNoiseStrategy,
        )

        assert BaseTrainingStrategy.forwards_loss_components is False
        for cls in (
            NoiseToNoiseStrategy,
            HSSCStrategy,
            DIFFSirenStrategy,
            DiffusionTrainingStrategy,
        ):
            assert cls.forwards_loss_components is True, cls.__name__
            assert "update(loss_output.components)" in inspect.getsource(cls), cls.__name__

    def test_production_path_calls_the_resolver(self):
        assert "resolve_producer_declared_columns" in _calls_in_training_loop()


class TestClassifyDiscardedKeys:
    def test_executor_aliases_are_named_as_known(self):
        from spectramr.pipelines.training_loop import classify_discarded_keys

        known, unexplained = classify_discarded_keys({"generator_loss"})
        assert known == ["generator_loss"]
        assert unexplained == []

    def test_unknown_key_is_unexplained(self):
        from spectramr.pipelines.training_loop import classify_discarded_keys

        known, unexplained = classify_discarded_keys({"pre_dc_kspace_l1"})
        assert known == []
        assert unexplained == ["pre_dc_kspace_l1"]

    def test_classification_never_drops_a_key(self):
        """Classifying is not suppressing.

        A future 'optimisation' that returned only the unexplained half would
        stop reporting real discards for the alias names -- the exact silent
        direction #1682 is about. Every input key must appear in the output.
        """
        from spectramr.pipelines.training_loop import classify_discarded_keys

        keys = {"generator_loss", "discriminator_loss", "pre_dc_kspace_l1", "lr_opt_g"}
        known, unexplained = classify_discarded_keys(keys)
        assert set(known) | set(unexplained) == keys
        assert not set(known) & set(unexplained)

    def test_alias_set_holds_only_exact_duplicates(self):
        """Guard the exclusion set against growth.

        Every member is an EXACT alias of a strategy-stamped total that IS in
        the header. Adding a key here without such a twin would convert a real
        discard into a silently-blessed one -- an allowlist by accretion.
        """
        from spectramr.pipelines.training_loop import _EXECUTOR_ALIAS_KEYS

        assert frozenset({"generator_loss", "discriminator_loss"}) == _EXECUTOR_ALIAS_KEYS

    def test_production_path_calls_the_classifier(self):
        assert "classify_discarded_keys" in _calls_in_training_loop()


class TestCallSitePinSeesTheShapesItClaimsTo:
    """Planted violations for the call-site pins (`.agent/rules/detectors.md` §1).

    The pins above assert that `_execute_training_loop` calls each resolver. A
    pin like that is worth exactly what it catches, and the failure mode is
    documented in this repo: a presence assertion over source text is satisfied
    by a docstring or a comment that merely names the thing. Each plant below
    feeds the pure core a source that VIOLATES the rule and asserts it is not
    fooled -- and feeds it the compliant shapes and asserts it is not blind.
    """

    def test_absent_call_is_not_detected(self):
        """The plant: the resolver is simply not called."""
        assert "resolve_scheduler_lr_columns" not in _called_names(
            "def f(pipeline):\n    expected_metrics.update(something_else(pipeline))\n"
        )

    def test_name_in_a_comment_does_not_satisfy_the_pin(self):
        assert "resolve_scheduler_lr_columns" not in _called_names(
            "def f(pipeline):\n    # calls resolve_scheduler_lr_columns(pipeline) eventually\n    pass\n"
        )

    def test_name_in_a_docstring_does_not_satisfy_the_pin(self):
        assert "resolve_scheduler_lr_columns" not in _called_names(
            'def f(pipeline):\n    """Delegates to resolve_scheduler_lr_columns(pipeline)."""\n    pass\n'
        )

    def test_name_as_a_string_literal_does_not_satisfy_the_pin(self):
        assert "resolve_scheduler_lr_columns" not in _called_names(
            'def f(pipeline):\n    return "resolve_scheduler_lr_columns"\n'
        )

    def test_name_referenced_without_being_called_does_not_satisfy_the_pin(self):
        """A bare reference is not a call -- passing it around is not wiring it."""
        assert "resolve_scheduler_lr_columns" not in _called_names(
            "def f(pipeline):\n    return [resolve_scheduler_lr_columns]\n"
        )

    def test_direct_call_is_detected(self):
        assert "resolve_scheduler_lr_columns" in _called_names(
            "def f(pipeline):\n    expected_metrics.update(resolve_scheduler_lr_columns(pipeline))\n"
        )

    def test_call_nested_in_a_branch_is_detected(self):
        """Shape variation: the real call site could move under a guard."""
        assert "resolve_scheduler_lr_columns" in _called_names(
            "def f(pipeline):\n"
            "    if pipeline is not None:\n"
            "        for _ in range(1):\n"
            "            x = resolve_scheduler_lr_columns(pipeline)\n"
        )

    def test_call_inside_a_comprehension_is_detected(self):
        assert "resolve_scheduler_lr_columns" in _called_names(
            "def f(ps):\n    return [resolve_scheduler_lr_columns(p) for p in ps]\n"
        )

    def test_attribute_call_is_not_counted_and_that_is_documented(self):
        """Honest limit, asserted rather than assumed.

        `mod.resolve_scheduler_lr_columns()` is a different construct. The pin
        does not see it. If the production call site ever moves behind a module
        attribute, this test is the record that the pin would go red -- a false
        alarm to fix deliberately, not a silent blindness to discover later.
        """
        assert "resolve_scheduler_lr_columns" not in _called_names(
            "def f(pipeline):\n    return mod.resolve_scheduler_lr_columns(pipeline)\n"
        )


class TestEmissionPrefixPinSeesDrift:
    """Planted violations for the `lr_` prefix agreement."""

    def test_drifted_prefix_is_detected_as_missing(self):
        """The plant: the emitter renames its column, the resolver does not."""
        assert "lr_" not in _fstring_prefixes(
            'def f(n):\n    losses_scalar[f"learningrate_{n}"] = 1.0\n'
        )

    def test_matching_prefix_is_found(self):
        assert "lr_" in _fstring_prefixes('def f(n):\n    losses_scalar[f"lr_{n}"] = 1.0\n')

    def test_prefix_in_a_comment_does_not_satisfy_the_pin(self):
        assert "lr_" not in _fstring_prefixes(
            'def f(n):\n    # writes lr_{n} somewhere\n    losses_scalar[f"learningrate_{n}"] = 1.0\n'
        )

    def test_plain_string_is_not_an_fstring(self):
        """Only a JoinedStr counts -- a non-interpolated "lr_" is not the emitter."""
        assert "lr_" not in _fstring_prefixes('def f(n):\n    x = "lr_"\n')


class TestReadSchedulerLr:
    """`read_scheduler_lr` reports absence instead of inferring it (#1682).

    The two branches replaced here were both `pass`. They became defects only
    once `resolve_scheduler_lr_columns` started promising a column per
    scheduler: a silent miss then yields a header column that is empty for the
    whole run, which a reader cannot tell from "validation never ran".
    """

    @staticmethod
    def _read(sched):
        from spectramr.pipelines.training_loop import read_scheduler_lr

        return read_scheduler_lr(sched)

    def test_working_scheduler_yields_the_first_group_lr(self):
        from types import SimpleNamespace

        lr, reason = self._read(SimpleNamespace(get_last_lr=lambda: [0.0003, 0.001]))
        assert lr == pytest.approx(0.0003)
        assert reason is None

    def test_scheduler_without_the_method_names_its_type(self):
        class NoSuchMethod:
            pass

        lr, reason = self._read(NoSuchMethod())
        assert lr is None
        assert "NoSuchMethod" in reason
        assert "get_last_lr" in reason

    def test_raising_scheduler_reports_the_exception_rather_than_passing(self):
        from types import SimpleNamespace

        def _boom():
            raise RuntimeError("scheduler not stepped yet")

        lr, reason = self._read(SimpleNamespace(get_last_lr=_boom))
        assert lr is None
        assert "RuntimeError" in reason
        assert "not stepped yet" in reason

    def test_empty_lr_list_is_reported_not_indexed(self):
        from types import SimpleNamespace

        lr, reason = self._read(SimpleNamespace(get_last_lr=lambda: []))
        assert lr is None
        assert "IndexError" in reason

    def test_non_numeric_lr_is_reported(self):
        from types import SimpleNamespace

        lr, reason = self._read(SimpleNamespace(get_last_lr=lambda: ["warmup"]))
        assert lr is None
        assert reason is not None

    @pytest.mark.parametrize(
        "sched",
        [
            object(),
            type("Raises", (), {"get_last_lr": lambda self: 1 / 0})(),
            type("Empty", (), {"get_last_lr": lambda self: []})(),
            type("Fine", (), {"get_last_lr": lambda self: [0.1]})(),
        ],
    )
    def test_exactly_one_of_lr_and_reason_is_none(self, sched):
        """The contract the caller's `if lr is not None ... elif` depends on."""
        lr, reason = self._read(sched)
        assert (lr is None) != (reason is None)


class TestStructuralCoresSeeWhatTheyClaim:
    """The cores behind the guard pins, exercised directly.

    A pin is only as good as its core: `_membership_guard_names` exists
    precisely because the substring pin it replaced scored a planted
    `elif False:` green.
    """

    def test_membership_core_finds_the_container_not_the_member(self):
        names = _membership_guard_names("if needle not in haystack:\n    pass\n")
        assert "haystack" in names
        assert "needle" not in names

    def test_membership_core_ignores_a_bare_declaration(self):
        """The exact shape that made the substring pin blind."""
        assert _membership_guard_names("seen = set()\nseen.add(x)\n") == set()

    def test_membership_core_sees_plain_in_as_well_as_not_in(self):
        assert _membership_guard_names("if a in b:\n    pass\n") == {"b"}

    def test_attribute_core_reports_object_and_method(self):
        assert _attribute_calls("seen.add(x)\n") == {"seen.add"}

    def test_attribute_core_ignores_a_bare_name_call(self):
        assert _attribute_calls("helper(x)\n") == set()


class TestSchedulerLrCallSiteIsPinned:
    """A revert to the swallowed `pass` must turn this red (non-negotiable 15).

    Pins the CALL, not the helper: a helper-only pin scores a call-site
    regression green, which is how the original `except Exception: pass`
    survived unnoticed.
    """

    def test_training_loop_reads_schedulers_through_the_reporting_helper(self):
        assert "read_scheduler_lr" in _calls_in_training_loop()

    def test_training_loop_no_longer_indexes_get_last_lr_inline(self):
        source = _training_loop_source()
        assert "get_last_lr()[0]" not in source, (
            "The inline read is back; its absence path is silent again."
        )

    def test_the_warn_once_set_actually_guards_the_warning(self):
        """Structural, not substring.

        A substring pin on the name is satisfied by the DECLARATION and by the
        `.add()` call, so replacing the guard with `elif False:` scored it
        green -- caught by planting it. Only the container-of-a-membership-test
        position is unique to the guard.
        """
        assert "csv_empty_lr_warned" in _membership_guard_names(_training_loop_source())

    def test_the_warn_once_set_is_actually_populated(self):
        """Without the `.add`, the guard stays true and the warning fires per row."""
        assert "csv_empty_lr_warned.add" in _attribute_calls(_training_loop_source())


# ---------------------------------------------------------------------------
# Wall-clock yield: stopping in time to write a checkpoint.
#
# A production arm gets 120h per job and needs more. What was missing was never
# resume -- the checkpoint has carried every piece of state all along -- it was
# STOPPING IN TIME TO WRITE ONE. These pin the four places the yield has to
# touch, each of which is independently satisfiable while the feature is dead:
# a decision nobody acts on, a save that never fires, a loop that never leaves,
# an epilogue that overwrites the yield's own checkpoint with a completed one.
#
# Source-level for the same reason as the loop_state and divergence-guard tests
# above: a full loop OOM-kills a dev box. The DECISION itself is executed in
# tests/unit/core/test_wall_clock.py.
# ---------------------------------------------------------------------------


def _loop_source() -> str:
    import spectramr.pipelines.training_loop as training_loop_mod

    return inspect.getsource(training_loop_mod._execute_training_loop)


def test_wall_clock_budget_is_resolved_before_the_loop():
    """A malformed or already-expired budget must fail at startup, not after
    hours of GPU — and re-reading the environment per step is a syscall in the
    hot path (non-negotiable 9)."""
    src = _loop_source()
    loop_at = src.index("for iteration in pbar:")
    before, inside = src[:loop_at], src[loop_at:]

    assert "wall_clock_budget = resolve_wall_clock_budget()" in before
    assert "resolve_wall_clock_budget()" not in inside


def test_the_wall_clock_check_has_its_own_capped_cadence():
    """NOT the logging cadence, which is what this rode first.

    65 corpus arms declare `logging.intervals.log: 5000`; at ~1s a step that
    inspects the deadline every ~83 min against a 900s margin, so the job is
    killed before the check fires -- on exactly the long-running arms the yield
    exists for. A `time.monotonic` read is a vDSO call, not a device sync, so
    non-negotiable 9 never bore on how often it happens.
    """
    src = _loop_source()
    loop_at = src.index("for iteration in pbar:")
    before, inside = src[:loop_at], src[loop_at:]

    assert "_wall_clock_every = resolve_check_interval(log_interval)" in before
    assert "iteration % _wall_clock_every == 0" in inside
    assert "resolve_check_interval" not in inside, "cadence recomputed per step"

    # The logging gate must stay the only `if` whose test mentions the log
    # cadence: TestEveryRunYieldsAnInterpretableCurve locates and executes it by
    # exactly that property, and a second match errors all nine of its tests.
    tree = ast.parse(textwrap.dedent(src))
    matching = [
        ast.unparse(n.test)
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and "iteration % log_interval" in ast.unparse(n.test)
    ]
    assert len(matching) == 1, matching

    # The decision is broadcast: ranks disagreeing by one iteration deadlock,
    # one entering the collective checkpoint save while the others step.
    check = src.index("iteration % _wall_clock_every == 0")
    assert "RankUtility.broadcast_object" in src[check : check + 600], (
        "the yield decision is per-rank -- under DDP the ranks will disagree "
        "and the collective save will hang"
    )


def test_a_budget_that_cannot_be_honoured_is_refused():
    """`checkpoint.enabled: false` plus a budget yields, saves nothing, requeues
    and repeats forever. Same family as the already-inside-the-margin refusal."""
    src = _loop_source()

    assert "A wall-clock budget is declared but checkpointing is off" in src
    guard_at = src.index("A wall-clock budget is declared but checkpointing is off")
    assert "raise RuntimeError(" in src[max(0, guard_at - 200) : guard_at]
    # It must sit above the loop -- refusing at iteration 40k is not a refusal.
    assert guard_at < src.index("for iteration in pbar:")


def test_the_certification_hook_is_skipped_on_a_yield():
    """A certificate from a half-trained reconstructor describes a model that
    will not exist by the end of the chain -- and it runs inside the save
    margin, eating the time reserved for getting out cleanly."""
    src = _loop_source()

    assert "if not is_sanity_check and not wall_clock_yield and diverged_at is None:" in src


def test_the_yield_forces_a_checkpoint_and_then_leaves():
    """A yield without the save it exists to make room for is just a crash."""
    src = _loop_source()

    assert "iteration % checkpoint_interval == 0 or wall_clock_yield" in src
    assert "if wall_clock_yield:\n            break" in src
    # The save must come first, or the break drops the very state it yielded for.
    assert src.index("or wall_clock_yield") < src.index("if wall_clock_yield:\n            break")


def test_the_completion_epilogue_is_skipped_on_a_yield():
    """The final save is the last state of a finished run; on a yield or a
    divergence there is no finished run to stamp."""
    src = _loop_source()

    final_at = src.index("# Final checkpoint on completion")
    guard = src[final_at : final_at + 1600]
    assert "not wall_clock_yield" in guard
    # A diverged run broke out early; a final file would mark it finished.
    assert "diverged_at is None" in guard

    restore_at = src.index('getattr(config.early_stopping, "restore_best_weights", False)')
    assert "not wall_clock_yield" in src[restore_at : restore_at + 400], (
        "a yielded run is mid-training; swapping in best weights hands the next "
        "link a model that is not where the loop left off"
    )


def test_the_result_distinguishes_yielded_from_completed():
    """`success: True` alone would let a caller archive a half-trained run."""
    src = _loop_source()

    assert '"wall_clock_yield": wall_clock_yield,' in src
    assert '"resume_from_iteration": iteration if wall_clock_yield else None,' in src


def test_validation_is_gated_on_the_yield_and_on_the_budget():
    """Validation is the longest thing in the loop body and the wall-clock check
    cannot interrupt it -- the check runs BETWEEN iterations, validation is
    inside one. A deadline landing mid-pass kills the job with no checkpoint, no
    marker and no requeue, so the chain stops looking like an ordinary TIMEOUT.

    `experiment_11_attention_none.yaml:506` records one validation event at 6.4 h
    and validation at 41% of the run's wall clock, so this is the dominant
    exposure rather than a corner.
    """
    src = _loop_source()

    # Gate on the EXECUTION line. Gating `time_for_eval` at its first assignment
    # is bypassed: the epoch-boundary branch re-sets it to True afterwards.
    assert (
        'if pipeline.data_loaders.get("val") and time_for_eval and not wall_clock_yield:'
        in src
    )
    first_assign = src.index("time_for_eval = iteration % eval_interval == 0")
    epoch_reassign = src.index("            time_for_eval = True")
    execution = src.index('if pipeline.data_loaders.get("val") and time_for_eval and not')
    assert first_assign < epoch_reassign < execution, (
        "the gate moved above the epoch-boundary re-assignment, where it is a no-op"
    )

    # And the budget check, which is what covers a deadline arriving mid-pass.
    assert "VALIDATION_SAFETY_FACTOR" in src
    assert "_last_val_seconds" in src


def test_the_budget_check_is_rank_uniform_and_broadcast():
    """A rank that skipped validation while another ran it would arrive at the
    collective checkpoint save an iteration apart and deadlock."""
    src = _loop_source()

    gate = src.index("if wall_clock_budget is not None and time_for_eval and not wall_clock_yield:")
    window = src[gate : gate + 900]

    # Entry depends only on values every rank computes identically; the
    # rank-local duration is read INSIDE, by rank 0, and the verdict broadcast.
    assert "RankUtility.broadcast_object(_fits)" in window
    assert "if is_main_process:" in window
    assert "_last_val_seconds" not in src[gate : gate + src[gate:].index("if is_main_process:")], (
        "the rank-local duration is read in the entry condition -- ranks can "
        "disagree about whether to enter the collective"
    )


def test_the_validation_duration_is_measured_not_assumed():
    """There was no validation timing in the loop; the budget check needs one."""
    src = _loop_source()

    assert "_val_started = time.monotonic() if time_for_eval else None" in src
    assert "_last_val_seconds = time.monotonic() - _val_started" in src
    # Only on eval iterations -- not a clock read every step.
    assert "time.monotonic() if time_for_eval else None" in src


def test_a_skipped_validation_says_so():
    """A validation event that vanishes without a line is an unexplained gap in
    the arm's record."""
    src = _loop_source()

    assert "elif time_for_eval and wall_clock_yield and logging_service:" in src
    assert "Skipping validation at iteration" in src


class TestLoggingServiceCallsResolve:
    """#2254: the divergence tripwire called ``logging_service.log_critical``, a
    method ``ILoggingService`` never declared and ``LoggingService`` never defined.

    The guard therefore raised ``AttributeError`` on the exact step it was built to
    catch, and the ``break`` on the next line never ran — a run with a non-finite
    loss died on the reporting path instead of stopping cleanly, and the log named a
    logging object rather than the iteration or the loss value.

    Pinning the literal name ``log_critical`` would only close this spelling. What is
    checked instead is the shape: every method the loop calls on the logging service
    must exist on the interface it is typed against. A full loop OOM-kills a dev box,
    so this is an AST check over the source, per the file-level note above.
    """

    @staticmethod
    def _logging_service_attrs(func) -> set[str]:
        """Every ``<recv>.<attr>(...)`` where ``<recv>`` names a logging service."""
        receivers = {"logging_service", "_logging_service", "self.logging_service"}
        found: set[str] = set()
        for node in ast.walk(_ast_of(func)):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute):
                continue
            recv = fn.value
            bare_name = isinstance(recv, ast.Name) and recv.id in receivers
            self_attribute = (
                isinstance(recv, ast.Attribute)
                and isinstance(recv.value, ast.Name)
                and recv.value.id == "self"
                and f"self.{recv.attr}" in receivers
            )
            if bare_name or self_attribute:
                found.add(fn.attr)
        return found

    def test_every_logging_call_in_the_loop_exists_on_the_interface(self):
        from spectramr.domain.interfaces.service_interfaces import ILoggingService

        called = self._logging_service_attrs(tl._execute_training_loop)
        assert called, "no logging_service calls found — the walker stopped matching"

        missing = sorted(a for a in called if not hasattr(ILoggingService, a))
        assert not missing, (
            f"_execute_training_loop calls {missing} on the logging service, and "
            f"ILoggingService declares no such method. Each one raises AttributeError "
            f"at the moment its guard fires. Add the rung to the interface AND to "
            f"LoggingService, or call a method that exists."
        )

    def test_the_divergence_guard_calls_a_real_method(self):
        """The specific rung, on the specific guard, with the concrete service."""
        from spectramr.infrastructure.services.logging_service import LoggingService

        src = inspect.getsource(tl._stop_on_divergence)
        called = re.findall(r"logging_service\.(\w+)\(", src)
        assert called, "the divergence guard no longer logs at all"
        for attr in called:
            assert callable(getattr(LoggingService, attr, None)), (
                f"divergence guard calls LoggingService.{attr}, which does not exist"
            )
        assert "try:" not in src, (
            "a try/except around the divergence log would swallow the failure it "
            "reports (non-negotiable 3) -- fix the call, do not guard it"
        )

    def test_the_guard_still_breaks_after_logging(self):
        """The ``break`` is the load-bearing half: logging names the failure, the
        break is what stops training before the weights are saved. Every in-loop
        verdict site sets ``diverged_at`` from the helper and breaks next."""
        loop = next(
            n for n in ast.walk(_ast_of(tl._execute_training_loop))
            if isinstance(n, ast.For) and ast.unparse(n.iter) == "pbar"
        )
        sites = 0
        for node in ast.walk(loop):
            if not isinstance(node, ast.If) or ast.unparse(node.test) != "_verdict is not None":
                continue
            sites += 1
            body = [ast.unparse(stmt) for stmt in node.body]
            assert body[-2].startswith("diverged_at = _stop_on_divergence("), body
            assert body[-1] == "break", body
        assert sites == 4, (
            f"expected end_step, two resolve sites and the step-failure handler, found {sites}"
        )

    def test_a_failed_step_resolves_the_monitor_before_it_reraises(self):
        """A step that fails on the NaN weights an unread step wrote must report
        that step's divergence; only a clean resolve lets the error propagate."""
        handler = next(
            n for n in ast.walk(_ast_of(tl._execute_training_loop))
            if isinstance(n, ast.ExceptHandler)
            and n.type is not None
            and ast.unparse(n.type) == "Exception"
            and any("Training Step Failed" in ast.unparse(s) for s in n.body)
        )
        body = [ast.unparse(stmt) for stmt in handler.body]
        assert body[0] == "_verdict = nonfinite_monitor.resolve()", body
        assert body[1].startswith("if _verdict is not None:"), body
        assert body[-1] == "raise", body


def test_every_checkpoint_save_goes_through_the_store():
    """#2304, #2799: every save hands the store the strategy, the early-stopping
    service and the runtime, so a file's contents do not depend on which event wrote it."""
    import spectramr.pipelines.training_loop as training_loop_mod

    for func in (
        training_loop_mod._execute_training_loop,
        training_loop_mod._save_loss_schedule_checkpoint,
    ):
        saves = _method_calls(func, "save_step") + _method_calls(func, "save_best")
        assert saves, func.__name__
        for call in saves:
            assert isinstance(call.func.value, ast.Name) and call.func.value.id == "checkpoint_service"
            keywords = {k.arg for k in call.keywords}
            assert {"strategy", "early_stopping", "parallel", "layout"} <= keywords, call.lineno


def test_the_final_checkpoint_carries_strategy_state_and_the_live_scaler(
    monkeypatch, tmp_path
):
    """#2799: drives a real run to its final save, then repeats that save with the
    run's strategy holding owned state and a live scaler."""
    import torch

    from spectramr.infrastructure.checkpointing import codecs, store

    real_save = store.save_step
    calls: list = []
    saved: list = []
    _run_deferring_finite_check(
        monkeypatch, tmp_path, diverge=False, saved_steps=saved, saved_calls=calls
    )
    pipeline, kw = calls[-1]
    assert (kw["step"], kw["reason"]) == (4, "final")
    strategy = kw["strategy"]
    assert strategy is not None, "the final save attaches no strategy"

    class _LiveScaler:
        def state_dict(self):
            return {"scale": 1024.0, "_growth_tracker": 7}

    owned = {"modules": {}, "params": {"sigma": torch.tensor([0.25])}}
    monkeypatch.setattr(strategy, "strategy_state_dict", lambda: owned, raising=False)
    monkeypatch.setattr(
        strategy, "amp_helper", SimpleNamespace(scaler=_LiveScaler()), raising=False
    )
    written = codecs.read(real_save(pipeline, **kw))
    assert torch.equal(written["strategy"]["params"]["sigma"], owned["params"]["sigma"])
    assert written["scaler"] == {"scale": 1024.0, "_growth_tracker": 7}


def test_the_resumed_early_stopping_state_is_restored_into_the_service():
    import spectramr.pipelines.training_loop as training_loop_mod

    loop = training_loop_mod._execute_training_loop
    (call,) = _method_calls(loop, "load_state_dict", on="early_stopping_service")
    assert _positional_names(call) == ["early_stopping_state"]
    run_calls = _plain_calls(TrainingLoop.run, "_execute_training_loop")
    assert any(k.arg == "early_stopping_state" for c in run_calls for k in c.keywords)


def test_a_resumed_early_stopped_run_trains_no_step():
    """The final file of an early-stopped run is stamped at the step it stopped, and
    should_stop() is otherwise read only after a validation: without this gate the
    resumed run trains one more validation window and writes a new final file."""
    src = _loop_source()
    restore_at = src.index("early_stopping_service.load_state_dict(early_stopping_state)")
    gate_at = src.index("early_stopping_service.should_stop()")
    loop_at = src.index("for iteration in pbar:")
    assert restore_at < gate_at < loop_at, "the stop check must sit before the iteration loop"

    gate = src[gate_at:loop_at]
    assert "already early-stopped" in gate

    # The rebind must sit inside the should_stop() branch, or every fresh run is emptied.
    tree = ast.parse(textwrap.dedent(src))
    gates = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "early_stopping_service.should_stop()" in ast.unparse(node.test)
    ]
    rebinds = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pbar" for t in node.targets)
        and ast.unparse(node.value) == "make_progress((), config.logging, desc='Training')"
    ]
    assert len(rebinds) == 1, "the gate must empty the iteration range"
    assert any(rebinds[0] in g.body for g in gates if g.lineno < rebinds[0].lineno)
    # ran_a_step then stays False, so no final save follows.
    assert "ran_a_step = iteration >= first_iteration" in src
    assert "iteration = start_iteration - 1" in src[:loop_at]


# ---------------------------------------------------------------------------
# A run the divergence guard stopped must not report success
# ---------------------------------------------------------------------------
#
# The guard stops the loop on a non-finite total, and the loop used to fall
# through to `success: True`. The total reaches it non-finite when the backend
# defers the finite check -- DeepSpeed's `guard_loss` is a no-op and a
# GradScaler defers -- so these drive the real director-built pipeline on CPU
# with that policy and a graph-connected NaN total.


def _divergence_config(
    tmp_path,
    *,
    checkpoint_enabled=False,
    checkpoint=None,
    max_iterations=4,
    logging=None,
    validation=None,
    log_interval=None,
):
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings.settings_from_dict(
        {
            "model": {
                "model_type": "dummy",
                "in_channels": 1,
                "out_channels": 1,
                "spatial_dims": 2,
                "model_kwargs": {"base_channels": 4, "num_res_blocks": 1},
            },
            "data": {
                "dataset_type": "synthetic",
                "patch_size": [16, 16, 1],
                "img_size": [16, 16],
                "batch_size": 1,
                "num_workers": 0,
                "synthetic_samples": 8,
                "target_channels": 1,
            },
            "losses": {"output_domain": "image", "image_losses": [{"name": "l1", "weight": 1.0}]},
            "checkpoint": {"enabled": checkpoint_enabled, **(checkpoint or {})},
            "optimization": {},
            "logging": logging or ({"intervals": {"log": log_interval}} if log_interval else {}),
            **({"validation": validation} if validation else {}),
            "training": {
                "strategy_class": "reconstruction",
                "output_dir": str(tmp_path),
                "max_iterations": max_iterations,
                "device": "cpu",
            },
        }
    )


def _run_deferring_finite_check(
    monkeypatch, tmp_path, *, diverge, saved_steps=None, saved_calls=None, checkpoint=None
):
    """``saved_steps`` (a list) turns checkpointing on and collects the step of
    every ``store.save_step``; ``saved_calls`` collects ``(pipeline, kwargs)``."""
    import spectramr.pipelines.train as train_mod
    from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
    from spectramr.infrastructure.training.step_executor import StepExecutor
    from spectramr.models.losses.computers.base import BaseLossComputer

    # Keep only the graph-connectivity check, as DeepSpeedStepPolicy does.
    monkeypatch.setattr(
        StepExecutor,
        "_guard_loss",
        lambda self, loss, *, name, global_step: ensure_backward_ready(
            loss, name=name, global_step=global_step
        ),
    )
    if diverge:
        stack = BaseLossComputer._stack_components
        monkeypatch.setattr(
            BaseLossComputer,
            "_stack_components",
            lambda self, *a, **k: stack(self, *a, **k) * float("nan"),
        )
    if saved_steps is not None:
        from spectramr.infrastructure.checkpointing import store

        def _record(pipeline, **kw):
            saved_steps.append(kw["step"])
            if saved_calls is not None:
                saved_calls.append((pipeline, kw))
            return tmp_path / "recorded.safetensors"

        monkeypatch.setattr(store, "save_step", _record)
    calibrations = []
    monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: calibrations.append(a))
    config = _divergence_config(
        tmp_path, checkpoint_enabled=saved_steps is not None, checkpoint=checkpoint
    )
    result = train_mod.run_training_pipeline(config, device="cpu")
    return result, calibrations


def _single_artifact(tmp_path, name):
    import json

    (path,) = list(tmp_path.rglob(name))
    return json.loads(path.read_text())


class TestDivergedRunReportsFailure:
    def test_a_diverged_run_returns_failure_and_says_where(self, monkeypatch, tmp_path):
        result, calibrations = _run_deferring_finite_check(monkeypatch, tmp_path, diverge=True)
        assert result["success"] is False
        assert result["diverged_at_iteration"] == 1
        assert "diverged at iteration 1" in result["error"]
        # A certificate for a model with NaN weights is not a certificate.
        assert calibrations == []

    def test_run_summary_and_final_metrics_record_the_divergence(self, monkeypatch, tmp_path):
        _run_deferring_finite_check(monkeypatch, tmp_path, diverge=True)
        summary = _single_artifact(tmp_path, "run_summary.json")
        assert summary["success"] is False
        assert "diverged" in summary["error"]
        assert _single_artifact(tmp_path, "final_metrics.json")["diverged_at_iteration"] == 1

    def test_the_same_run_without_divergence_still_succeeds(self, monkeypatch, tmp_path):
        """Control: the failure above is the divergence, not the harness."""
        result, calibrations = _run_deferring_finite_check(monkeypatch, tmp_path, diverge=False)
        assert result["success"] is True
        assert _single_artifact(tmp_path, "final_metrics.json")["diverged_at_iteration"] is None
        assert len(calibrations) == 1

    def test_a_diverged_run_writes_no_final_checkpoint(self, monkeypatch, tmp_path):
        """The final save stamps the last iteration; on a diverged run that tag
        marks it finished and ``--resume if-present`` would load it."""
        saved = []
        _run_deferring_finite_check(monkeypatch, tmp_path, diverge=True, saved_steps=saved)
        assert saved == []

    def test_the_non_diverged_control_still_writes_the_final_checkpoint(
        self, monkeypatch, tmp_path
    ):
        saved = []
        _run_deferring_finite_check(monkeypatch, tmp_path, diverge=False, saved_steps=saved)
        assert saved and saved[-1] == 4


# ---------------------------------------------------------------------------
# NonFiniteMonitor in the real loop: lag 1, and never a saved NaN step
# ---------------------------------------------------------------------------
#
# The monitor reports a non-finite step one step late, or at the resolve that
# precedes a save, validation, yield or loop exit. These drive the director-built
# dummy run on CPU through the production guard (no `_guard_loss` patch).


def _nan_from(monkeypatch, step, *, only=False):
    """Make the loss non-finite from ``step`` on (the executor's global_step), or
    at ``step`` alone when ``only`` -- the later steps then see the NaN weights
    that step's update wrote, as a real run would."""
    from spectramr.infrastructure.training.step_executor import StepExecutor
    from spectramr.models.losses.computers.base import BaseLossComputer

    current = {"step": None}
    execute = StepExecutor.execute_step

    def _tracking(self, step_configs, epoch, global_step, *a, **k):
        current["step"] = global_step
        return execute(self, step_configs, epoch, global_step, *a, **k)

    stack = BaseLossComputer._stack_components

    def _poisoned(self, *a, **k):
        out = stack(self, *a, **k)
        at = current["step"]
        hit = at == step if only else at is not None and at >= step
        return out * float("nan") if hit else out

    monkeypatch.setattr(StepExecutor, "execute_step", _tracking)
    monkeypatch.setattr(BaseLossComputer, "_stack_components", _poisoned)


def _record_saves(monkeypatch, tmp_path, saved_steps):
    from spectramr.infrastructure.checkpointing import store

    def _record(pipeline, **kw):
        saved_steps.append(kw["step"])
        return tmp_path / "recorded.safetensors"

    monkeypatch.setattr(store, "save_step", _record)


class _StubScaler:
    """A GradScaler stand-in that, like a real one on overflow, skips every step."""

    def __init__(self):
        self._per_optimizer_states = {}

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        return None  # the skipped optimizer step of an overflowing scaler

    def update(self):
        return None


def _run_dummy(monkeypatch, tmp_path, **config_kw):
    import spectramr.pipelines.train as train_mod

    monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: None)
    return train_mod.run_training_pipeline(
        _divergence_config(tmp_path, **config_kw), device="cpu"
    )


def test_run_detaches_the_monitor_when_the_loop_raises(monkeypatch):
    """A raise out of the loop must not leave the executor recording into a
    monitor nobody reads; ``TrainingLoop.run`` owns the detach on every exit."""
    strategy = SimpleNamespace(step_executor=SimpleNamespace(nonfinite_monitor=None))

    def _attach_then_raise(strategy, *a, **k):
        strategy.step_executor.nonfinite_monitor = object()
        raise RuntimeError("step failed")

    monkeypatch.setattr(tl, "_execute_training_loop", _attach_then_raise)
    with pytest.raises(RuntimeError, match="step failed"):
        TrainingLoop(strategy, MagicMock(), None, "m").run()
    assert strategy.step_executor.nonfinite_monitor is None


def test_final_loss_scalar_is_one_host_float_or_none():
    import torch

    assert tl._final_loss_scalar({}) is None
    assert tl._final_loss_scalar({"g_total_loss": torch.tensor(0.25)}) == 0.25
    assert isinstance(tl._final_loss_scalar({"g_total_loss": torch.tensor(0.0)}), float)
    assert tl._final_loss_scalar({"g_total_loss": 3}) == 3.0


def test_stop_on_divergence_names_the_scanned_state_and_the_value(caplog):
    """The scan runs on the weights after ``state_after`` (they already carry the
    NaN step's update), and the critical line says nan vs inf when known."""
    import torch

    from spectramr.infrastructure.training.nonfinite_monitor import NonFiniteVerdict

    lines: list[str] = []
    service = SimpleNamespace(log_critical=lines.append)
    verdict = NonFiniteVerdict(iteration=2, names=("generator",), on_peer_rank=False)
    with caplog.at_level(logging.CRITICAL):
        got = tl._stop_on_divergence(
            verdict,
            logging_service=service,
            generator=torch.nn.Linear(2, 2),
            state_after=3,
            losses={"g_total_loss": torch.tensor(float("inf"))},
        )
    assert got == 2
    assert "iteration 2" in lines[0] and "(total = inf)" in lines[0]
    assert "weights after iteration 3" in lines[0]
    assert "[Divergence] iteration 3" in caplog.text

    tl._stop_on_divergence(
        verdict, logging_service=service, generator=torch.nn.Linear(2, 2), state_after=3
    )
    assert "total =" not in lines[1]


class TestNonFiniteMonitorInTheLoop:
    def test_a_nan_at_the_save_iteration_is_never_saved(self, monkeypatch, tmp_path):
        """(a) Lag 1 alone would save step 2 and report it at step 3; the resolve
        before the periodic save stops the run first and names step 2."""
        saved: list[int] = []
        _record_saves(monkeypatch, tmp_path, saved)
        _nan_from(monkeypatch, 2)
        result = _run_dummy(
            monkeypatch, tmp_path, checkpoint_enabled=True, checkpoint={"save_interval": 2}
        )
        assert result["success"] is False, result
        assert result["diverged_at_iteration"] == 2
        assert saved == [], f"a checkpoint was written at {saved}"

    def test_one_nan_step_stops_at_that_step_when_the_next_one_raises(
        self, monkeypatch, tmp_path
    ):
        """A single NaN at step 2 writes NaN weights; step 3 then fails inside the
        executor (every loss component is dropped, so the total is a leaf) before
        the loop reaches ``end_step``. The failure must be reported as the
        divergence at step 2, not as step 3's secondary error."""
        saved: list[int] = []
        _record_saves(monkeypatch, tmp_path, saved)
        _nan_from(monkeypatch, 2, only=True)
        result = _run_dummy(
            monkeypatch, tmp_path, checkpoint_enabled=True, checkpoint={"save_interval": 1000}
        )
        assert result["success"] is False, result
        assert result["diverged_at_iteration"] == 2
        assert _single_artifact(tmp_path, "final_metrics.json")["diverged_at_iteration"] == 2
        assert saved == [], f"a checkpoint was written at {saved}"

    def test_run_summary_final_loss_is_a_number(self, monkeypatch, tmp_path):
        """The loop keeps losses on the device; the summary must still carry a
        float, not ``str(tensor)`` from ``json.dumps(default=str)``."""
        result = _run_dummy(monkeypatch, tmp_path)
        assert result["success"] is True, result.get("error")
        assert isinstance(result["final_loss"], float)
        final_loss = _single_artifact(tmp_path, "run_summary.json")["final_loss"]
        assert isinstance(final_loss, float), repr(final_loss)

    def test_a_nan_between_saves_is_named_at_its_own_iteration(self, monkeypatch, tmp_path):
        """The verdict reaches the loop one step late but names the NaN step."""
        _nan_from(monkeypatch, 1)
        result = _run_dummy(monkeypatch, tmp_path)
        assert result["success"] is False
        assert result["diverged_at_iteration"] == 1

    def test_a_nan_in_the_last_iteration_is_caught_at_loop_exit(self, monkeypatch, tmp_path):
        """Iteration 5 of 5 is neither a validation (every 2) nor a save step, so
        only the loop-exit resolve can see it before the final save."""
        saved: list[int] = []
        _record_saves(monkeypatch, tmp_path, saved)
        _nan_from(monkeypatch, 5)
        result = _run_dummy(
            monkeypatch,
            tmp_path,
            max_iterations=5,
            checkpoint_enabled=True,
            checkpoint={"save_interval": 1000},
            validation={"schedule": {"interval_steps": 2}},
        )
        assert result["diverged_at_iteration"] == 5
        assert saved == [], "the final checkpoint of a diverged run was written"

    def test_a_scaler_does_not_hide_a_nan_total(self, monkeypatch, tmp_path):
        """(b) Under a GradScaler the per-config guard defers to the scaler, so
        the loop's own record of the total is what stops the run."""
        from spectramr.infrastructure.training.mixed_precision import (
            MixedPrecisionIntegrationHelper,
        )

        init = MixedPrecisionIntegrationHelper.__init__

        def _with_stub_scaler(self, *a, **k):
            init(self, *a, **k)
            self.scaler = _StubScaler()

        monkeypatch.setattr(MixedPrecisionIntegrationHelper, "__init__", _with_stub_scaler)
        _nan_from(monkeypatch, 1)
        result = _run_dummy(monkeypatch, tmp_path)
        assert result["success"] is False, result
        assert result["diverged_at_iteration"] == 1

    def test_a_non_log_step_performs_no_host_sync(self, monkeypatch, tmp_path):
        """(c) Iterations 2..6 of 7 are not log steps (interval 1000; the first
        and last are forced). Count every host read issued from the loop and the
        guard path on those iterations. The monitor's own CPU read is excluded:
        on CPU it is free, and its CUDA path is proven sync-free under
        ``set_sync_debug_mode("error")`` in ``test_nonfinite_monitor.py``.

        Positive controls: the tracked iterations must cover every non-log step,
        and the same counter must see at least one host read from a file outside
        the scoped three on those steps, so an inert patch cannot pass."""
        result, reads = _count_non_log_host_reads(monkeypatch, tmp_path)
        assert result["success"] is True, result.get("error")
        assert reads.seen >= _NON_LOG_STEPS, f"tracked iterations {sorted(reads.seen)}"
        assert reads.elsewhere, "the counter saw no host read at all; it is not live"
        assert reads.owned == [], f"host syncs on non-log steps: {reads.owned}"

    def test_the_host_sync_counter_flags_a_planted_read(self, monkeypatch, tmp_path):
        """Planted violation (rule 15): an ``.item()`` compiled as if it were a
        line of ``training_loop.py`` and run on every step must be counted on
        each non-log step."""
        result, reads = _count_non_log_host_reads(monkeypatch, tmp_path, plant=True)
        assert result["success"] is True, result.get("error")
        flagged = {it for it, path, _, attr in reads.owned if path == tl.__file__ and attr == "item"}
        assert flagged == _NON_LOG_STEPS, reads.owned


_NON_LOG_STEPS = set(range(2, 7))


def _count_non_log_host_reads(monkeypatch, tmp_path, *, plant=False):
    """Run the 7-step dummy with log interval 1000 and count host reads on the
    non-log steps, split by whether the caller is the loop or the guard path."""
    import sys

    import torch

    import spectramr.infrastructure.training.step_executor as executor_mod
    import spectramr.infrastructure.training.strategy_interfaces as interfaces_mod

    owned_files = {tl.__file__, executor_mod.__file__, interfaces_mod.__file__}
    current: dict[str, int | None] = {"iteration": None}
    reads = SimpleNamespace(owned=[], elsewhere=[], seen=set())

    real_make_progress = tl.make_progress

    class _Tracking:
        """The real progress object, plus which iteration is running."""

        def __init__(self, inner):
            self._inner = inner

        def __iter__(self):
            for it in self._inner:
                current["iteration"] = it
                reads.seen.add(it)
                yield it
            current["iteration"] = None

        def set_postfix(self, data):
            self._inner.set_postfix(data)

        def close(self):
            self._inner.close()

    def _counting(attr):
        real = getattr(torch.Tensor, attr)

        def _wrapper(self, *a, **k):
            caller = sys._getframe(1)
            if current["iteration"] in _NON_LOG_STEPS:
                where = (current["iteration"], caller.f_code.co_filename, caller.f_lineno, attr)
                owned = caller.f_code.co_filename in owned_files
                (reads.owned if owned else reads.elsewhere).append(where)
            return real(self, *a, **k)

        return _wrapper

    monkeypatch.setattr(
        tl, "make_progress", lambda *a, **k: _Tracking(real_make_progress(*a, **k))
    )
    for attr in ("item", "tolist", "cpu", "__bool__"):
        monkeypatch.setattr(torch.Tensor, attr, _counting(attr))
    if plant:
        from spectramr.infrastructure.training.step_executor import StepExecutor

        namespace: dict = {}
        exec(compile("def _planted(t):\n    t.item()\n", tl.__file__, "exec"), namespace)
        execute = StepExecutor.execute_step

        def _with_planted_read(self, *a, **k):
            out = execute(self, *a, **k)
            namespace["_planted"](torch.zeros(()))
            return out

        monkeypatch.setattr(StepExecutor, "execute_step", _with_planted_read)
    result = _run_dummy(
        monkeypatch, tmp_path, max_iterations=7, logging={"intervals": {"log": 1000}}
    )
    return result, reads


class TestFinalSaveIsNotDuplicated:
    def _run(self, monkeypatch, tmp_path, interval):
        calls: list = []
        _run_deferring_finite_check(
            monkeypatch,
            tmp_path,
            diverge=False,
            saved_steps=[],
            saved_calls=calls,
            checkpoint={"save_interval": interval},
        )
        return [(kw["step"], kw["reason"]) for _, kw in calls]

    def test_interval_dividing_the_run_saves_the_last_step_once(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, 2) == [(2, "periodic"), (4, "periodic")]

    def test_interval_not_dividing_the_run_still_saves_final(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, 3) == [(3, "periodic"), (4, "final")]

    def test_run_shorter_than_the_interval_saves_final(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path, 100) == [(4, "final")]


def test_evaluate_forwards_the_iteration_and_csv_of_a_validate_only_pass(monkeypatch):
    """``--validate-only`` re-validates a resumed run at its own step and writes
    the row where the in-training passes wrote theirs."""
    import spectramr.pipelines.training_loop as training_loop_mod

    captured: dict = {}

    def fake_validation(pipeline, strategy, iteration, epoch, logging_service, **kw):
        captured.update(iteration=iteration, epoch=epoch, **kw)
        return {"val_nex_equivalent": 9.0}

    drained: dict = {}
    monkeypatch.setattr(training_loop_mod, "_run_validation", fake_validation)
    monkeypatch.setattr(
        training_loop_mod, "_drain_cascade_rows", lambda *a, **k: drained.update(k)
    )
    pipeline = MagicMock()
    pipeline.generator.training = False
    loop = TrainingLoop("strat", pipeline, "cfg", "unet")
    metrics = loop.evaluate(iteration=25000, epoch=48, csv_file="run/logs/training_metrics.csv")
    assert metrics == {"val_nex_equivalent": 9.0}
    assert captured["iteration"] == 25000 and captured["epoch"] == 48
    assert captured["csv_file"] == "run/logs/training_metrics.csv"
    assert drained["iteration"] == 25000 and drained["epoch"] == 48


def test_every_validation_event_is_tallied_and_judged_before_success():
    """#181: the run-level verdict is only as good as its wiring. A full loop
    OOM-kills a dev box, so the placement is pinned at source level (the repo's
    convention for loop-body invariants); the verdict itself is exercised in
    ``test_validation_evidence.py`` and its consumer in ``test_train.py``."""
    import spectramr.pipelines.training_loop as training_loop_mod

    src = inspect.getsource(training_loop_mod._execute_training_loop)
    run_at = src.index("val_metrics = _run_validation(")
    assert src.index("validation_tally.record(val_metrics)") > run_at
    verdict_at = src.index("validation_failure_reason(validation_tally)")
    success_at = src.rindex('"success": True')
    assert verdict_at < success_at, "the verdict must gate the success return"
    assert src.index("validation_failure_reason(validation_tally)") < src.index(
        "if is_sanity_check:\n        passed, report"
    ), "a sanity check must not pass on a validation that measured nothing"
    assert src.count("**validation_tally.as_result_fields()") == 4


class TestPolicyOwnsLrScheduler:
    """The loop must not step a scheduler the step policy already steps."""

    def test_owned_policy_is_detected(self):
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import policy_owns_lr_scheduler

        strategy = SimpleNamespace(step_executor=SimpleNamespace(_policy_owns_lr_scheduler=True))
        assert policy_owns_lr_scheduler(strategy) is True

    @pytest.mark.parametrize("executor", [None, object()])
    def test_absent_or_unflagged_executor_means_the_loop_steps(self, executor):
        from types import SimpleNamespace

        from spectramr.pipelines.training_loop import policy_owns_lr_scheduler

        assert policy_owns_lr_scheduler(SimpleNamespace(step_executor=executor)) is False

    def test_truthy_accident_is_not_ownership(self):
        from unittest.mock import MagicMock

        from spectramr.pipelines.training_loop import policy_owns_lr_scheduler

        assert policy_owns_lr_scheduler(MagicMock()) is False


# ---------------------------------------------------------------------------
# #1972 -- one checkpoint writer, one destination, and a failed save raises
# ---------------------------------------------------------------------------
#
# A failed periodic save used to select a second writer (the pre-1.0 service) by
# exception. It wrote a generator-only file into the retired
# ``checkpoint.checkpoint_dir`` and the run carried on reporting success. These
# drive the real dummy-model run.


def _run_with_checkpoint_dir_elsewhere(monkeypatch, tmp_path, *, save_interval, fail_at=None):
    """Run 4 iterations launched from ``elsewhere``, a directory outside the run.

    The retired ``checkpoint.checkpoint_dir`` defaulted to ``./checkpoints``, which
    the service created under the launch CWD (#2842). ``fail_at`` makes
    ``store.save_step`` raise ENOSPC at those steps. Returns the result
    and the two directories.
    """
    import errno

    import spectramr.pipelines.train as train_mod
    run_dir = tmp_path / "run"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = _divergence_config(
        run_dir,
        checkpoint_enabled=True,
        checkpoint={"save_interval": save_interval},
    )

    if fail_at is not None:
        from spectramr.infrastructure.checkpointing import store

        real_save = store.save_step

        def _save(pipeline, **kw):
            if kw["step"] in fail_at:
                raise OSError(errno.ENOSPC, "No space left on device")
            return real_save(pipeline, **kw)

        monkeypatch.setattr(store, "save_step", _save)
    monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: None)
    result = train_mod.run_training_pipeline(config, device="cpu")
    return result, run_dir / "checkpoints", elsewhere


def test_every_loss_schedule_save_is_gated_on_checkpoint_enabled_and_may_checkpoint():
    """With ``checkpoint.enabled: false`` the loss-schedule save must not run.

    Ungated, it called ``None.save_step`` (swallowed by the fail-soft handler and
    logged as a save failure), or wrote a checkpoint the user had disabled when a
    store was present. ``may_checkpoint`` stays in the gate: the save is a
    collective under FSDP/DeepSpeed.
    """
    import spectramr.pipelines.training_loop as training_loop_mod

    tree = _ast_of(training_loop_mod._execute_training_loop)
    sites = 0

    def _names(test: ast.AST) -> set[str]:
        return {n.id for n in ast.walk(test) if isinstance(n, ast.Name)}

    def _visit(node: ast.AST, guards: list[set[str]]) -> None:
        nonlocal sites
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_save_loss_schedule_checkpoint"
        ):
            sites += 1
            seen = set().union(*guards) if guards else set()
            assert {"checkpoint_enabled", "may_checkpoint"} <= seen, seen
        for child in ast.iter_child_nodes(node):
            if isinstance(node, ast.If) and child in node.body:
                _visit(child, [*guards, _names(node.test)])
            else:
                _visit(child, guards)

    _visit(tree, [])
    assert sites == 2, f"expected the two loss-schedule save sites, found {sites}"


class TestOneCheckpointWriter:
    def test_the_production_save_lands_in_the_run_and_nowhere_else(self, monkeypatch, tmp_path):
        """Control and reachability: the real save path writes under output_dir."""
        result, ckpt_dir, elsewhere = _run_with_checkpoint_dir_elsewhere(
            monkeypatch, tmp_path, save_interval=2
        )
        assert result["success"] is True, result.get("error")
        written = sorted(p.name for p in ckpt_dir.glob("step_*.safetensors"))
        assert "step_00000002.safetensors" in written, written
        assert "step_00000004.safetensors" in written, written
        assert not (elsewhere / "checkpoints").exists(), "a checkpoints/ dir under the launch CWD"

    def test_a_failed_periodic_save_stops_the_run_instead_of_writing_elsewhere(
        self, monkeypatch, tmp_path
    ):
        result, _, elsewhere = _run_with_checkpoint_dir_elsewhere(
            monkeypatch, tmp_path, save_interval=2, fail_at={2}
        )
        assert not (elsewhere / "checkpoints").exists()
        assert result["success"] is False
        assert "No space left on device" in result["error"]

    def test_a_failed_final_save_fails_the_run(self, monkeypatch, tmp_path):
        """No periodic save fires (interval 1000); only the final one, and it fails."""
        result, ckpt_dir, _ = _run_with_checkpoint_dir_elsewhere(
            monkeypatch, tmp_path, save_interval=1000, fail_at={4}
        )
        assert result["success"] is False
        assert "No space left on device" in result["error"]
        assert not list(ckpt_dir.glob("step_*.safetensors"))

    def test_the_loop_holds_no_second_writer(self):
        """Source pin: the loop never calls the service's save."""
        for func in (tl._execute_training_loop, tl._save_loss_schedule_checkpoint):
            assert not _method_calls(func, "save_checkpoint"), func.__name__


def test_enabled_checkpointing_without_a_store_raises():
    src = _loop_source()
    assert "checkpoint.enabled is true but no checkpoint store was injected" in src


def test_the_loss_schedule_save_goes_through_save_step(tmp_path):
    """The helper writes a real step file through the store, with its reason, and
    keep_last_n applies to it like any other step."""
    from types import SimpleNamespace

    from spectramr.infrastructure.checkpointing import codecs
    from spectramr.infrastructure.checkpointing.facade import CheckpointStore
    from tests.unit.infrastructure.checkpointing.conftest import make_pipeline

    config = SimpleNamespace(checkpoint=SimpleNamespace(format="pt", keep_last_n=1))
    for step in (3, 7):
        tl._save_loss_schedule_checkpoint(
            checkpoint_service=CheckpointStore(),
            config=config, pipeline=make_pipeline(), strategy=None,
            output_paths={"run_output_dir": str(tmp_path)}, epoch=0, iteration=step,
            logging_service=None, parallel_runtime=None, early_stopping_service=None,
        )
    files = sorted((tmp_path / "checkpoints").iterdir())
    assert [p.name for p in files] == ["step_00000007.pt"]
    assert codecs.read(files[0])["meta"]["reason"] == "loss_schedule"


# ---------------------------------------------------------------------------
# checkpoint.async_write: one writer per run, drained before anything reads
# ---------------------------------------------------------------------------


def _failing_writer(tmp_path, *, fatal: bool):
    from unittest.mock import patch

    from spectramr.infrastructure.checkpointing import async_writer
    from spectramr.infrastructure.checkpointing.async_writer import CheckpointWriter
    from tests.unit.infrastructure.checkpointing.conftest import make_run_state

    (tmp_path / "checkpoints").mkdir()
    name, role = ("step_00000150.pt", "step") if fatal else ("best.pt", "best")
    path = tmp_path / "checkpoints" / name
    writer = CheckpointWriter()
    with patch.object(async_writer.codecs, "write", side_effect=OSError("disk full")):
        writer.submit(path, make_run_state("pt", role=role), None, fatal=fatal)
        writer._pending.result()
    return writer, path


def test_the_end_of_loop_drain_fails_the_run_on_a_fatal_write(tmp_path):
    """#1972, behaviourally: a periodic/final write that failed off-thread stops the run."""
    from spectramr.infrastructure.checkpointing.async_writer import CheckpointWriteError

    writer, path = _failing_writer(tmp_path, fatal=True)
    with pytest.raises(CheckpointWriteError) as caught:
        tl._finish_checkpoints(writer, None)
    assert caught.value.path == path
    assert writer._executor is None


def test_the_end_of_loop_drain_records_a_failed_best_write(tmp_path):
    """#713: a failed best write is reported as best_checkpoint_error, not raised."""
    writer, path = _failing_writer(tmp_path, fatal=False)
    error = tl._finish_checkpoints(writer, "iter 3: earlier", 3)
    assert error.startswith("iter 150: ") and str(path) in error and "disk full" in error


def test_a_later_synchronous_best_error_is_not_overridden_by_an_earlier_async_one(tmp_path):
    writer, _ = _failing_writer(tmp_path, fatal=False)  # fails at iter 150
    assert tl._finish_checkpoints(writer, "iter 200: sync", 200) == "iter 200: sync"


def test_without_a_writer_the_best_error_passes_through():
    assert tl._finish_checkpoints(None, "iter 3: x") == "iter 3: x"


def test_every_save_site_hands_over_the_run_s_writer():
    src = _loop_source() + inspect.getsource(tl._save_loss_schedule_checkpoint)
    saves = src.count("checkpoint_service.save_step(") + src.count("checkpoint_service.save_best(")
    assert saves == 4
    assert len(re.findall(r"(?<!checkpoint_)writer=checkpoint_writer,", src)) == saves
    assert src.count("checkpoint_writer=checkpoint_writer,") == 2  # both loss-schedule sites


def test_the_fail_soft_saves_let_an_earlier_fatal_write_through():
    """A periodic failure surfacing at a best or loss-schedule submit is not theirs to swallow."""
    for src in (_loop_source(), inspect.getsource(tl._save_loss_schedule_checkpoint)):
        reraise = src.index("except CheckpointWriteError:\n")
        assert src[reraise:].split("\n")[1].strip().startswith("raise")
        assert reraise < src.index("except Exception as", reraise)


def test_writes_are_finished_before_the_best_path_is_broadcast_and_read():
    src = _loop_source()
    finish = src.index("best_checkpoint_error = _finish_checkpoints(")
    assert src.index('reason="final"') < finish
    assert finish < src.index("RankUtility.broadcast_object(best_checkpoint_path)")


# The real store, the real writer thread and the real loop, on the CPU harness above.


def _run_async(monkeypatch, tmp_path, *, save_interval, fail_step=None, budget=None):
    import spectramr.pipelines.train as train_mod
    from spectramr.infrastructure.checkpointing import async_writer, codecs

    config = _divergence_config(
        tmp_path,
        checkpoint_enabled=True,
        checkpoint={"async_write": True, "save_interval": save_interval, "format": "pt"},
    )
    if fail_step is not None:
        real = codecs.write

        def write(path, state):
            if state["meta"]["step"] == fail_step:
                raise OSError("disk full")
            real(path, state)

        monkeypatch.setattr(async_writer.codecs, "write", write)
    if budget is not None:
        monkeypatch.setattr(tl, "resolve_wall_clock_budget", lambda: budget)
        monkeypatch.setattr(tl, "resolve_check_interval", lambda *_: 1)
    monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: None)
    result = train_mod.run_training_pipeline(config, device="cpu")
    ckpt_dir = tmp_path / "checkpoints"
    files = {p.name: codecs.read(p)["meta"]["reason"] for p in sorted(ckpt_dir.glob("step_*.pt"))}
    return result, files


class TestAsyncWriteThroughTheLoop:
    def test_periodic_steps_land_and_the_last_one_is_not_written_twice(
        self, monkeypatch, tmp_path
    ):
        result, files = _run_async(monkeypatch, tmp_path, save_interval=2)
        assert result["success"] is True, result.get("error")
        assert files == {"step_00000002.pt": "periodic", "step_00000004.pt": "periodic"}

    def test_a_failed_periodic_write_fails_the_run_and_names_its_file(
        self, monkeypatch, tmp_path
    ):
        result, _ = _run_async(monkeypatch, tmp_path, save_interval=2, fail_step=2)
        assert result["success"] is False
        assert "step_00000002" in result["error"] and "disk full" in result["error"]

    def test_the_final_file_is_on_disk_when_the_call_returns(self, monkeypatch, tmp_path):
        result, files = _run_async(monkeypatch, tmp_path, save_interval=3)
        assert result["success"] is True, result.get("error")
        assert files == {"step_00000003.pt": "periodic", "step_00000004.pt": "final"}

    def test_a_yield_leaves_its_file_on_disk_when_the_call_returns(self, monkeypatch, tmp_path):
        """Failure mode 4: the yield breaks the loop, and its save must still land."""
        import time
        from types import SimpleNamespace

        calls = []

        def expired():
            calls.append(1)
            return len(calls) >= 2

        budget = SimpleNamespace(
            expired=expired,
            deadline=time.time() + 60,
            seconds_remaining=lambda: 60.0,
            margin_s=0.0,
        )
        result, files = _run_async(monkeypatch, tmp_path, save_interval=1000, budget=budget)
        assert list(files.values()) == ["yield"], (result.get("error"), files)

    def test_an_early_stop_at_a_periodic_step_still_writes_the_final(self, monkeypatch, tmp_path):
        """The stop breaks before that step's periodic save, so no duplicate to skip."""
        src = _loop_source()
        stop = src.index("Early Stopping triggered at iter")
        assert stop < src.index('reason="yield" if wall_clock_yield else "periodic"')
        assert stop < src.index('reason="final"')

    def test_an_exception_mid_loop_drains_the_write_and_keeps_the_real_error(
        self, monkeypatch, tmp_path
    ):
        """The exit-stack close lands the in-flight periodic write and stops the thread."""
        import threading

        import spectramr.pipelines.train as train_mod
        from spectramr.infrastructure.checkpointing import async_writer, codecs

        planted = RuntimeError("planted step failure")
        gate, submitted, writers, caught = threading.Event(), threading.Event(), [], []
        real_write, real_submit, real_run = codecs.write, async_writer.CheckpointWriter.submit, tl.TrainingLoop.run

        def slow_write(path, state):
            assert gate.wait(timeout=10)
            real_write(path, state)

        def submit(self, *a, **k):
            writers.append(self)
            real_submit(self, *a, **k)
            submitted.set()

        def run(self, **kw):
            real_step = self.strategy.train_step

            def step(*a, **k):
                if submitted.is_set():
                    gate.set()  # the write is in flight; let it finish during teardown
                    raise planted
                return real_step(*a, **k)

            self.strategy.train_step = step
            try:
                return real_run(self, **kw)
            except BaseException as exc:
                caught.append(exc)
                raise

        monkeypatch.setattr(async_writer.codecs, "write", slow_write)
        monkeypatch.setattr(async_writer.CheckpointWriter, "submit", submit)
        monkeypatch.setattr(tl.TrainingLoop, "run", run)
        monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: None)
        config = _divergence_config(
            tmp_path,
            checkpoint_enabled=True,
            checkpoint={"async_write": True, "save_interval": 2, "format": "pt"},
        )
        result = train_mod.run_training_pipeline(config, device="cpu")

        assert caught == [planted] and caught[0] is planted
        assert result["success"] is False and result["error"] == "planted step failure"
        [writer] = set(writers)
        assert writer._executor is None
        assert not any(t.name.startswith("ckpt-writer") for t in threading.enumerate())
        assert codecs.read(tmp_path / "checkpoints" / "step_00000002.pt")["meta"]["reason"] == (
            "periodic"
        )


# ---------------------------------------------------------------------------
# training CSV: one persistent handle (Task 4.1)
# ---------------------------------------------------------------------------


class TestCsvRowSink:
    FIELDS = ("iteration", "epoch", "g_total_loss")

    def _rows(self):
        return [{"iteration": i, "epoch": 0, "g_total_loss": 1.0 / i, "extra": 9} for i in range(1, 6)]

    def test_five_rows_open_once_and_match_per_row_append(self, tmp_path, monkeypatch):
        import builtins
        import csv as csv_mod

        from spectramr.pipelines.training_loop import _CsvRowSink

        expected = tmp_path / "expected.csv"
        with open(expected, "w", newline="") as f:
            csv_mod.DictWriter(f, fieldnames=list(self.FIELDS)).writeheader()
        for row in self._rows():
            with open(expected, "a", newline="") as f:
                csv_mod.DictWriter(f, fieldnames=list(self.FIELDS), extrasaction="ignore").writerow(row)

        actual = tmp_path / "actual.csv"
        with open(actual, "w", newline="") as f:
            csv_mod.DictWriter(f, fieldnames=list(self.FIELDS)).writeheader()
        opens = []
        real_open = builtins.open
        monkeypatch.setattr(
            builtins, "open", lambda *a, **k: (opens.append(a), real_open(*a, **k))[1]
        )
        sink = _CsvRowSink(actual, list(self.FIELDS))
        for row in self._rows():
            sink.write(row)
            # flushed: complete on disk before close
            assert actual.read_text().count("\n") >= 1
        assert len([o for o in opens if str(actual) in str(o[0])]) == 1
        assert actual.read_text().splitlines()[-1] == expected.read_text().splitlines()[-1]
        sink.close()
        assert actual.read_text() == expected.read_text()

    def test_rows_complete_on_disk_without_close(self, tmp_path):
        from spectramr.pipelines.training_loop import _CsvRowSink

        p = tmp_path / "m.csv"
        p.write_text("iteration,epoch,g_total_loss\n")
        sink = _CsvRowSink(p, list(self.FIELDS))
        sink.write({"iteration": 1, "epoch": 0, "g_total_loss": 0.5})
        assert p.read_text().splitlines()[-1] == "1,0,0.5"
        sink.close()
        sink.close()  # idempotent

    def test_a_failed_flush_closes_the_handle_and_the_next_row_reopens(self, tmp_path):
        import pytest

        from spectramr.pipelines.training_loop import _CsvRowSink

        p = tmp_path / "m.csv"
        p.write_text("iteration,epoch,g_total_loss\n")
        sink = _CsvRowSink(p, list(self.FIELDS))
        sink.write({"iteration": 1, "epoch": 0, "g_total_loss": 0.5})
        first_fh = sink._fh

        def boom():
            raise OSError("disk full")

        first_fh.flush = boom
        with pytest.raises(OSError, match="disk full"):
            sink.write({"iteration": 2, "epoch": 0, "g_total_loss": 0.25})
        assert sink._fh is None and first_fh.closed
        sink.write({"iteration": 3, "epoch": 0, "g_total_loss": 0.125})
        sink.close()
        lines = p.read_text().splitlines()
        assert lines[0] == "iteration,epoch,g_total_loss"
        assert all(len(ln.split(",")) == 3 for ln in lines)
        assert lines[-1] == "3,0,0.125"


class _RecordingSink(tl._CsvRowSink):
    instances: ClassVar[list[_RecordingSink]] = []

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.opens = 0
        self.rows = 0
        self.closed = False
        _RecordingSink.instances.append(self)

    def write(self, row):
        was_open = self._fh is not None
        super().write(row)
        self.rows += 1
        self.opens += 0 if was_open else 1

    def close(self):
        super().close()
        self.closed = True


class TestCsvSinkClosesOnEveryLoopExit:
    """Drives the real (tiny, CPU) loop; every exit must leave the handle closed."""

    def _run(self, monkeypatch, tmp_path, *, mode):
        import spectramr.pipelines.train as train_mod
        from spectramr.infrastructure.training.backward_guard import ensure_backward_ready
        from spectramr.infrastructure.training.step_executor import StepExecutor
        from spectramr.models.losses.computers.base import BaseLossComputer

        _RecordingSink.instances = []
        monkeypatch.setattr(tl, "_CsvRowSink", _RecordingSink)
        monkeypatch.setattr(tl, "_maybe_run_calibration", lambda *a, **k: None)
        stack = BaseLossComputer._stack_components
        calls = {"n": 0}

        def patched(self, *a, **k):
            calls["n"] += 1
            if mode == "exception" and calls["n"] >= 3:
                raise RuntimeError("boom from train_step")
            out = stack(self, *a, **k)
            # NaN only from the third call: a first-step NaN breaks before any row is logged.
            return out * float("nan") if mode == "diverge" and calls["n"] >= 3 else out

        if mode == "diverge":
            monkeypatch.setattr(
                StepExecutor,
                "_guard_loss",
                lambda self, loss, *, name, global_step: ensure_backward_ready(
                    loss, name=name, global_step=global_step
                ),
            )
        monkeypatch.setattr(BaseLossComputer, "_stack_components", patched)
        config = _divergence_config(tmp_path, log_interval=1)
        with contextlib.suppress(RuntimeError):
            train_mod.run_training_pipeline(config, device="cpu")
        (sink,) = _RecordingSink.instances
        return sink

    def test_normal_completion_opens_once_and_closes(self, monkeypatch, tmp_path):
        sink = self._run(monkeypatch, tmp_path, mode="normal")
        assert sink.rows >= 3 and sink.opens == 1 and sink.closed
        assert sink._fh is None

    def test_exception_in_the_step_closes_the_handle(self, monkeypatch, tmp_path):
        sink = self._run(monkeypatch, tmp_path, mode="exception")
        assert sink.rows >= 1 and sink.closed and sink._fh is None

    def test_divergence_break_closes_the_handle(self, monkeypatch, tmp_path):
        sink = self._run(monkeypatch, tmp_path, mode="diverge")
        assert _RecordingSink.instances
        assert sink.rows >= 1 and sink.closed and sink._fh is None


def test_the_finished_result_names_the_step_its_restored_weights_come_from() -> None:
    """test_metrics.json said iteration 84001 for weights restored from step 54000.

    Every dict literal that reports ``best_weights_restored`` must carry
    ``best_iteration`` beside it, read from the early-stopping service.
    """
    from spectramr.pipelines import training_loop

    tree = ast.parse(inspect.getsource(training_loop))
    results = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict)
        and any(isinstance(k, ast.Constant) and k.value == "best_weights_restored" for k in node.keys)
    ]
    assert results, "the loop no longer reports best_weights_restored"
    for node in results:
        keys = {k.value for k in node.keys if isinstance(k, ast.Constant)}
        assert "best_iteration" in keys
    es = SimpleNamespace(best_iteration=54000)
    assert training_loop._extract_es_best_iter(es) == 54000
    assert training_loop._extract_es_best_iter(None) is None
