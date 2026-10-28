"""Light-seam unit tests for :mod:`spectramr.pipelines.train`.

Only pure-Python helpers that do NOT require a real dataset, GPU,
or the DI container are tested here. Specifically:

* ``early_stop_monitor_candidates`` — validates ordered alias expansion
  for the four mismatch classes (prefix, loss-alias, cascade-add,
  cascade-strip) documented in the function's docstring.

Heavy paths (``run_training_pipeline``, ``run_sanity_check``, etc.) are
skipped with an explicit reason: they require ``TrainingSettings``,
real dataloaders, and GPU infrastructure.  They are exercised by the
smoke / integration suite on the cluster.
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Import guard — if torch or heavy deps are missing, skip the whole module.
# The root conftest installs a MagicMock shim for torch so collection
# always succeeds; we only skip test-*execution* if something unexpected
# fails at import time.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# early_stop_monitor_candidates
# ---------------------------------------------------------------------------


from spectramr.pipelines.train import early_stop_monitor_candidates  # noqa: E402


class TestEarlyStopMonitorCandidates:
    """Exhaustive coverage of the four mismatch classes."""

    # ── invariants ───────────────────────────────────────────────────────

    def test_first_candidate_is_monitor_key(self) -> None:
        """The user-supplied key must always be the first / preferred candidate."""
        for key in ("val_psnr", "loss", "val_loss", "val_psnr_2x"):
            cands = early_stop_monitor_candidates(key)
            assert cands[0] == key

    def test_output_is_list(self) -> None:
        cands = early_stop_monitor_candidates("val_psnr")
        assert isinstance(cands, list)

    def test_no_duplicates(self) -> None:
        for key in ("val_psnr", "val_loss", "loss", "psnr", "val_ssim_2x"):
            cands = early_stop_monitor_candidates(key)
            assert len(cands) == len(set(cands)), f"Duplicate candidates for {key!r}: {cands}"

    # ── prefix mismatch (val_ ↔ bare) ────────────────────────────────────

    def test_val_prefix_stripped(self) -> None:
        """``val_psnr`` → bare ``psnr`` included."""
        cands = early_stop_monitor_candidates("val_psnr")
        assert "psnr" in cands

    def test_bare_key_no_val_prefix_added(self) -> None:
        """Bare ``psnr`` should NOT automatically expand to ``val_psnr``.

        The rule is *strip*, not *add*: the validator emits the bare key.
        We don't want the monitor to try an invented namespace.
        """
        cands = early_stop_monitor_candidates("psnr")
        # "val_psnr" should not appear as a fabricated alias.
        # (cascade-add may introduce val_psnr_<suffix> which is different)
        assert "val_psnr" not in cands

    # ── loss-alias expansion ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "monitor",
        ["val_loss", "loss"],
    )
    def test_loss_aliases_included(self, monitor: str) -> None:
        """Both ``val_loss`` and ``loss`` must expand to the full alias set."""
        cands = early_stop_monitor_candidates(monitor)
        for alias in (
            "val_recon_loss",
            "recon_loss",
            "val_total_loss",
            "total_loss",
            "g_total_loss",
            "val_g_total_loss",
            "val_mse",
            "mse",
        ):
            assert alias in cands, (
                f"Expected alias {alias!r} in candidates for monitor {monitor!r}; got {cands}"
            )

    def test_non_loss_key_no_loss_aliases(self) -> None:
        """``val_psnr`` must NOT pollute candidates with loss aliases."""
        cands = early_stop_monitor_candidates("val_psnr")
        for alias in ("val_recon_loss", "g_total_loss", "val_mse"):
            assert alias not in cands, f"Unexpected alias {alias!r} for monitor 'val_psnr'"

    # ── cascade-add (bare → suffixed) ────────────────────────────────────

    @pytest.mark.parametrize("suffix", ["_2x", "_8x", "_32x"])
    def test_cascade_add_suffixes_produced(self, suffix: str) -> None:
        """``val_psnr`` should gain ``val_psnr_2x``, ``val_psnr_8x``, etc."""
        cands = early_stop_monitor_candidates("val_psnr")
        assert f"val_psnr{suffix}" in cands, (
            f"Expected cascade-add candidate 'val_psnr{suffix}' in {cands}"
        )

    def test_cascade_add_bare_monitor_also_gains_suffix(self) -> None:
        """``psnr`` (bare) should also gain ``psnr_2x``."""
        cands = early_stop_monitor_candidates("psnr")
        assert "psnr_2x" in cands

    # ── cascade-strip (suffixed → bare) ──────────────────────────────────

    @pytest.mark.parametrize("suffix", ["_2x", "_8x", "_32x"])
    def test_cascade_strip_produces_bare_key(self, suffix: str) -> None:
        """``val_psnr_2x`` must produce the stripped key ``val_psnr``."""
        monitor = f"val_psnr{suffix}"
        cands = early_stop_monitor_candidates(monitor)
        assert "val_psnr" in cands, (
            f"Expected cascade-strip 'val_psnr' in candidates for {monitor!r}; got {cands}"
        )

    def test_cascade_strip_also_strips_val_prefix(self) -> None:
        """``val_psnr_2x`` → strip suffix → ``val_psnr`` → strip val_ → ``psnr``."""
        cands = early_stop_monitor_candidates("val_psnr_2x")
        assert "psnr" in cands

    # ── edge cases ───────────────────────────────────────────────────────

    def test_arbitrary_custom_key_no_crash(self) -> None:
        """Custom monitors not matching any pattern should just return [key]."""
        cands = early_stop_monitor_candidates("custom_metric_xyz")
        assert cands[0] == "custom_metric_xyz"
        assert isinstance(cands, list)

    def test_empty_string_no_crash(self) -> None:
        """Empty string is unusual but should not raise."""
        cands = early_stop_monitor_candidates("")
        assert isinstance(cands, list)

    def test_all_cascade_suffixes_are_covered(self) -> None:
        """Smoke-check that all three documented suffixes are in the _CASCADE_SUFFIXES tuple.

        Canonical home is the infrastructure-layer metric_keys module (the
        controller cannot import leftward from pipelines/); train.py re-exports
        only the public ``early_stop_monitor_candidates``.
        """
        from spectramr.infrastructure.services.metric_keys import (  # noqa: PLC0415
            _CASCADE_SUFFIXES,
        )

        assert "_2x" in _CASCADE_SUFFIXES
        assert "_8x" in _CASCADE_SUFFIXES
        assert "_32x" in _CASCADE_SUFFIXES


# ---------------------------------------------------------------------------
# Heavy entry points — assert the SEAM, not the run
# ---------------------------------------------------------------------------
# Executing these needs a real DI container, a validated YAML, dataloaders and
# (for inference) a checkpoint, so the cluster smoke/integration suite owns the
# behaviour. What a unit test CAN own is the calling contract: both used to be
# bare ``pytest.skip(...)`` bodies with no assertion, which cost the same
# collection time while catching nothing. A renamed or dropped keyword is
# exactly the defect class that shipped twice here — a nonexistent
# ``main._apply_overrides`` import, and ``validation.sampler_steps`` silently
# discarded by a signature probe — and it is cheap to pin.


def test_run_training_pipeline_exposes_its_documented_call_contract() -> None:
    import inspect

    from spectramr.pipelines.train import run_training_pipeline

    params = inspect.signature(run_training_pipeline).parameters
    assert callable(run_training_pipeline)
    # ``config`` first (the SSOT object, never a path); the rest are the knobs
    # the CLI, the sanity-check path and the in-process scripting path pass.
    assert next(iter(params)) == "config"
    assert {"device", "is_sanity_check", "resume_path", "env", "strategy"} <= set(params)


def test_run_inference_pipeline_exposes_its_documented_call_contract() -> None:
    import inspect

    from spectramr.pipelines.infer import run_inference_pipeline

    params = inspect.signature(run_inference_pipeline).parameters
    assert callable(run_inference_pipeline)
    assert {
        "config_path",
        "checkpoint_path",
        "input_path",
        "output_path",
        "device",
        "batch_size",
    } <= set(params)


# ---------------------------------------------------------------------------
# _is_epoch_boundary (PIPE-1 hoist correctness)
# ---------------------------------------------------------------------------


from spectramr.pipelines.train import _is_epoch_boundary  # noqa: E402


class TestIsEpochBoundary:
    """The PIPE-1 perf hoist reuses a pre-bound train_loader_len; this helper
    must preserve the original inline semantics — crucially that a missing/empty
    loader is NEVER an epoch boundary even though train_loader_len falls back
    to 1 (and ``iteration % 1 == 0`` is always True)."""

    def test_missing_loader_is_never_boundary(self) -> None:
        # has_train_loader=False with the fallback len=1: must be False at every
        # iteration (the bug the guard fixes — was True every step).
        for it in (0, 1, 2, 7, 100):
            assert _is_epoch_boundary(it, 1, has_train_loader=False) is False

    def test_boundary_on_multiples_of_loader_len(self) -> None:
        assert _is_epoch_boundary(0, 10, has_train_loader=True) is True
        assert _is_epoch_boundary(10, 10, has_train_loader=True) is True
        assert _is_epoch_boundary(20, 10, has_train_loader=True) is True

    def test_not_boundary_between_multiples(self) -> None:
        for it in (1, 5, 9, 11, 19):
            assert _is_epoch_boundary(it, 10, has_train_loader=True) is False

    def test_single_batch_loader_every_iteration_is_boundary(self) -> None:
        # A genuine 1-batch loader (has_train_loader=True, len=1) is an epoch
        # boundary every step — same as the original logic.
        for it in (0, 1, 2, 3):
            assert _is_epoch_boundary(it, 1, has_train_loader=True) is True


# ---------------------------------------------------------------------------
# _resync_scheduler_base_lrs (F7d universal backstop — smoke audit 2026-06-04)
# ---------------------------------------------------------------------------

import torch  # noqa: E402
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts  # noqa: E402

from spectramr.infrastructure.training.scheduler_system import (  # noqa: E402
    WarmupScheduler,
)
from spectramr.pipelines.train import _resync_scheduler_base_lrs  # noqa: E402


class TestResyncSchedulerBaseLrs:
    """A strategy may add a param group to opt_g AFTER the scheduler is built;
    the step-site resync must extend base_lrs so step() doesn't zip-mismatch."""

    @staticmethod
    def _opt():
        m = torch.nn.Linear(8, 4)
        return m, torch.optim.Adam(m.parameters(), lr=1e-4)

    def test_raw_scheduler_resynced_and_steps(self) -> None:
        m, opt = self._opt()
        sched = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
        opt.add_param_group({"params": [torch.nn.Parameter(torch.randn(3))], "lr": 1e-3})
        assert len(sched.base_lrs) == 1  # stale before resync
        _resync_scheduler_base_lrs(sched)
        assert len(sched.base_lrs) == len(opt.param_groups) == 2
        for _ in range(6):
            opt.zero_grad(set_to_none=True)
            m(torch.randn(2, 8)).sum().backward()
            opt.step()
            _resync_scheduler_base_lrs(sched)  # idempotent at every step
            sched.step()
        assert len(sched.base_lrs) == 2  # idempotent — no over-grow

    def test_warmup_wrapper_inner_and_outer_resynced(self) -> None:
        m, opt = self._opt()
        inner = CosineAnnealingWarmRestarts(opt, T_0=5, T_mult=2, eta_min=1e-6)
        sched = WarmupScheduler(optimizer=opt, main_scheduler=inner, warmup_steps=3)
        opt.add_param_group({"params": [torch.nn.Parameter(torch.randn(3))], "lr": 1e-3})
        _resync_scheduler_base_lrs(sched)
        assert len(sched.base_lrs) == 2
        assert len(sched.warmup_start_lr) == 2
        assert len(sched.warmup_end_lr) == 2
        assert len(inner.base_lrs) == 2
        for _ in range(8):
            opt.zero_grad(set_to_none=True)
            m(torch.randn(2, 8)).sum().backward()
            opt.step()
            sched.step()

    def test_no_optimizer_is_noop(self) -> None:
        _resync_scheduler_base_lrs(object())  # must not raise


# ---------------------------------------------------------------------------
# select_validation_extra_fields — the gated Pattern-A per-sample field seam
# (the production caller of the field-conditioned validation_step overrides;
#  was previously inline + untested, and silently dropped contrast_id).
# ---------------------------------------------------------------------------


from spectramr.pipelines.train import (  # noqa: E402
    _VALIDATION_FORWARD_FIELDS,
    select_validation_extra_fields,
)


class TestSelectValidationExtraFields:
    """The double gate ``key in vs_params AND key in val_batch`` keeps train/val
    conditioning aligned and prevents undeclared-kwarg collisions."""

    def test_forwards_declared_and_present_fields(self) -> None:
        # A cross-field strategy declares field_strength_target + contrast_id and
        # the batch carries both -> both are forwarded (the train/val-match fix).
        vs_params = {
            "input_batch",
            "target_batch",
            "field_strength_target",
            "contrast_id",
        }
        batch = {
            "field_strength_target": 7.0,
            "contrast_id": 2,
            "input": 0,
            "target": 0,
        }
        out = select_validation_extra_fields(batch, vs_params)
        assert out == {"field_strength_target": 7.0, "contrast_id": 2}

    def test_does_not_forward_undeclared_field(self) -> None:
        # contrast_id is in the batch but the strategy does NOT declare it ->
        # must not be forwarded (would be an unexpected kwarg / TypeError risk).
        vs_params = {"input_batch", "target_batch", "field_strength_target"}
        batch = {"field_strength_target": 7.0, "contrast_id": 2}
        out = select_validation_extra_fields(batch, vs_params)
        assert out == {"field_strength_target": 7.0}
        assert "contrast_id" not in out

    def test_does_not_forward_absent_field(self) -> None:
        # The strategy declares contrast_id but the batch lacks it -> skipped
        # (no None injected; the strategy keeps its own default).
        vs_params = {"input_batch", "target_batch", "contrast_id"}
        batch = {"field_strength_target": 7.0}
        out = select_validation_extra_fields(batch, vs_params)
        assert out == {}

    def test_non_get_batch_is_safe(self) -> None:
        # A batch without .get (e.g. a bare tensor stand-in) yields no extras.
        assert select_validation_extra_fields(object(), {"contrast_id"}) == {}

    def test_contrast_id_is_a_recognised_forward_field(self) -> None:
        # Regression guard for the dropped-contrast_id bug: the seam's SSOT tuple
        # must include contrast_id (and the field-strength keys). 'sources' (B-1.1) is
        # load-bearing: dropping it silently disables the consensus-mean validation render.
        for key in (
            "field_strength",
            "field_strength_target",
            "contrast_id",
            "sources",
        ):
            assert key in _VALIDATION_FORWARD_FIELDS


class TestBatchDataReachesAStrategyThatDeclaresIt:
    """The whole-batch forward, gated on declaration alone.

    Pattern-A dispatch hands ``validation_step`` two tensors and nothing else, so
    a Pattern-A strategy could not see its own batch. ``DiffusionTrainingStrategy``
    tried to recover it from ``batch = (input_batch, target_batch)`` -- a tuple, so
    every guard in that shim was False by construction and ``batch_data`` stayed
    ``None`` for the entire validation path. That is what sent the k-space
    compensator into its recompute-and-divide branch on batches the loader had
    already normalized.

    The gate is ``"batch_data" in vs_params`` and nothing else: a strategy that
    does not declare the parameter receives no extra kwarg, so the 28 subclass
    overrides with their own signatures are untouched.
    """

    def test_a_declaring_strategy_receives_the_batch(self) -> None:
        batch = {"input": 0, "target": 0, "kspace_scale": 224.359}
        out = select_validation_extra_fields(
            batch, {"input_batch", "target_batch", "batch_data"}
        )
        assert out["batch_data"] is batch

    def test_a_non_declaring_strategy_receives_nothing(self) -> None:
        """Zero blast radius on the other Pattern-A strategies."""
        batch = {"input": 0, "target": 0, "kspace_scale": 224.359}
        out = select_validation_extra_fields(batch, {"input_batch", "target_batch"})
        assert "batch_data" not in out

    def test_the_forward_is_not_gated_on_batch_shape(self) -> None:
        """A declared batch_data is forwarded even for a batch without ``.get``.

        The per-sample fields need the mapping protocol to be lifted out; the batch
        itself does not. Every consumer reads it via ``read_batch_field``, which is
        shape-agnostic, so withholding it here would only reintroduce ``None``.
        """
        bare = object()
        out = select_validation_extra_fields(
            bare, {"input_batch", "target_batch", "batch_data"}
        )
        assert out == {"batch_data": bare}

    def test_the_diffusion_strategy_actually_declares_it(self) -> None:
        """Wires the seam to its consumer: a rename on either side must fail here.

        ``inspect.signature`` resolves the OVERRIDE, so this assertion is what
        stops the parameter being dropped from the signature while train.py keeps
        gating on a name nothing declares.
        """
        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        params = inspect.signature(DiffusionTrainingStrategy.validation_step).parameters
        assert "batch_data" in params
        assert params["batch_data"].default is None, (
            "must stay optional -- 28 subclasses call super() with two positionals"
        )

    def test_the_dead_tuple_shim_is_gone(self) -> None:
        """The decoy must not come back (pitfall #16).

        A guard on ``batch`` cannot work: ``batch`` is the tuple assigned two lines
        above it, so it reads as a safety net while guaranteeing ``batch_data``
        stays ``None``.
        """
        import inspect

        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        src = inspect.getsource(DiffusionTrainingStrategy.validation_step)
        # Comment lines are excluded deliberately: the removal is DOCUMENTED in a
        # note that names the old guard, and matching that note would make this
        # test fail on its own explanation.
        code = "\n".join(
            line for line in src.splitlines() if not line.lstrip().startswith("#")
        )
        assert 'kwargs.get("batch_data")' not in code, (
            "batch_data is a declared parameter now; reading it from kwargs would "
            "shadow the declaration the dispatch gate keys on"
        )
        assert "isinstance(batch, dict)" not in code, "the dead tuple guard returned"


# ---------------------------------------------------------------------------
# ema_should_update — honors ema.update_frequency (pitfall #15)
# ---------------------------------------------------------------------------


from spectramr.pipelines.train import ema_should_update  # noqa: E402


class TestEmaShouldUpdate:
    """The EMA update is gated on update_frequency (was an inert knob)."""

    def test_frequency_one_updates_every_step(self) -> None:
        assert all(ema_should_update(i, 1) for i in range(5))

    def test_frequency_four_updates_every_fourth_step(self) -> None:
        hits = [i for i in range(12) if ema_should_update(i, 4)]
        assert hits == [0, 4, 8]

    def test_nonpositive_frequency_falls_back_to_every_step(self) -> None:
        # 0 / negative must not divide-by-zero or disable EMA entirely.
        assert ema_should_update(3, 0) is True
        assert ema_should_update(3, -1) is True

    def test_warmup_steps_skips_early_updates(self) -> None:
        # No EMA updates before warmup_steps; resume afterwards.
        assert ema_should_update(5, 1, warmup_steps=10) is False
        assert ema_should_update(10, 1, warmup_steps=10) is True
        assert ema_should_update(11, 1, warmup_steps=10) is True

    def test_warmup_zero_is_no_skip(self) -> None:
        # Default warmup_steps=0 preserves every-step behavior.
        assert ema_should_update(0, 1, warmup_steps=0) is True

    @pytest.mark.parametrize("gas", [2, 4])
    def test_updates_once_per_optimizer_step_under_accumulation(self, gas: int) -> None:
        """The weights change only on the boundary micro-batch; a blend on any
        other one re-applies the decay to unchanged weights (decay**gas)."""
        hits = [i for i in range(4 * gas) if ema_should_update(i, 1, 0, gas)]
        assert hits == [gas - 1, 2 * gas - 1, 3 * gas - 1, 4 * gas - 1]

    def test_frequency_and_warmup_count_optimizer_steps(self) -> None:
        gas = 2
        hits = [i for i in range(40) if ema_should_update(i, 5, 0, gas)]
        assert [i // gas for i in hits] == [0, 5, 10, 15]
        first = next(i for i in range(40) if ema_should_update(i, 1, 3, gas))
        assert first // gas == 3


# ---------------------------------------------------------------------------
# restore_best_weights is wired into the training loop (was inert; pitfall #15)
# ---------------------------------------------------------------------------


class TestRestoreBestWeightsWiring:
    """The early-stopping best-weights restore must be present and guarded."""

    def test_restore_block_is_wired(self) -> None:
        import inspect

        # The loop body moved to pipelines/training_loop.py (WS-3 PR-2).
        from spectramr.pipelines import training_loop

        src = inspect.getsource(training_loop._execute_training_loop)
        # (1) the best path is tracked, (2) gated on the flag, (3) reloaded.
        assert "best_checkpoint_path = str(best_path)" in src
        assert "restore_best_weights" in src
        restore = src.index("checkpoint_service.best_weights(")
        assert "best_checkpoint_path" in src[restore : src.index(")", restore)]


# ---------------------------------------------------------------------------
# DDP rank-safety of the train pipeline (source-level; real DDP needs procs)
# ---------------------------------------------------------------------------


def test_tb_writer_created_only_on_rank_zero():
    """Rank-0 ownership is now a property of the writer, not of a source line.

    This asserted on the literal `'tracking_service == "tensorboard" and
    _is_rank_zero'` appearing in `run_training_pipeline`. That was the only
    instrument available while the gate was an inline conditional, but a source
    scan passes on a line that is present and wrong, and breaks on a refactor
    that is correct -- which is what happened when the two writers were
    collapsed into `TensorBoardWriter`.

    The behaviour is now pinned directly in
    `tests/unit/infrastructure/services/test_tensorboard_writer.py`
    (`test_a_non_zero_rank_gets_no_writer`). What remains worth asserting here
    is the SEAM: the pipeline must pass its rank through, because a writer that
    defaults `is_rank_zero=True` would silently give every rank an event file.
    """
    import inspect  # noqa: PLC0415

    from spectramr.pipelines import train  # noqa: PLC0415

    src = inspect.getsource(train.run_training_pipeline)
    assert "is_rank_zero=_is_rank_zero" in src, (
        "the pipeline must forward its rank to TensorBoardWriter; the default "
        "is True, so an omitted argument gives EVERY rank an event dir"
    )


def test_validation_image_save_gated_on_main_rank():
    import inspect  # noqa: PLC0415

    from spectramr.pipelines import train  # noqa: PLC0415

    src = inspect.getsource(train._run_validation)
    assert "RankUtility.is_main_rank()" in src
