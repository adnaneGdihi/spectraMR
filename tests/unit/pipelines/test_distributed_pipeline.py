"""Unit tests for src/pipelines/distributed.py — DDP entry point utilities."""

from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class TestSetupDistributed:
    """Tests for setup_distributed()."""

    def test_missing_env_vars_raises(self):
        """Should raise RuntimeError if RANK/WORLD_SIZE env vars are missing."""
        from spectramr.pipelines.distributed import setup_distributed

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(RuntimeError, match="RANK"):
                setup_distributed()

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_setup_returns_rank_world(self, _mock_set, _mock_init):
        """Should return (rank, world_size) from env vars."""
        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"}
        with patch.dict("os.environ", env, clear=True):
            rank, world_size = setup_distributed(backend="gloo")
            assert rank == 1
            assert world_size == 4

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_a_hand_rolled_srun_rank_comes_from_slurm_procid(self, _mock_set, _mock_init):
        """No RANK, a WORLD_SIZE and SLURM_PROCID: the global rank, not LOCAL_RANK."""
        from spectramr.pipelines.distributed import setup_distributed

        env = {"SLURM_PROCID": "5", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
        with patch.dict("os.environ", env, clear=True):
            rank, world_size = setup_distributed(backend="gloo")
        assert (rank, world_size) == (5, 8)

    @patch("torch.distributed.init_process_group")
    def test_local_rank_alone_is_not_a_global_rank(self, _mock_init):
        """Planted (#2701): a WORLD_SIZE and a LOCAL_RANK but no RANK or SLURM_PROCID would
        have made every node's first process rank 0; it is now refused."""
        from spectramr.pipelines.distributed import setup_distributed

        with (
            patch.dict("os.environ", {"WORLD_SIZE": "4", "LOCAL_RANK": "1"}, clear=True),
            pytest.raises(RuntimeError, match="RANK"),
        ):
            setup_distributed(backend="gloo")

    def test_local_rank_prefers_local_rank_then_slurm_localid_then_rank(self):
        from spectramr.pipelines.distributed import _local_rank

        with patch.dict("os.environ", {"LOCAL_RANK": "2", "SLURM_LOCALID": "1"}, clear=True):
            assert _local_rank(9) == 2
        with patch.dict("os.environ", {"SLURM_LOCALID": "1"}, clear=True):
            assert _local_rank(9) == 1
        with patch.dict("os.environ", {}, clear=True):
            assert _local_rank(9) == 9

    @patch("torch.distributed.init_process_group")
    def test_slurm_procid_without_a_world_size_is_not_a_rank(self, _mock_init):
        from spectramr.pipelines.distributed import setup_distributed

        with (
            patch.dict("os.environ", {"SLURM_PROCID": "3"}, clear=True),
            pytest.raises(RuntimeError, match="WORLD_SIZE"),
        ):
            setup_distributed(backend="gloo")

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_gloo_does_not_select_a_cuda_device(self, mock_set, _mock_init):
        """``set_device`` raises on a CPU host, so a gloo group must not call it."""
        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict("os.environ", env, clear=True):
            setup_distributed(backend="gloo")
        mock_set.assert_not_called()

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_timeout_seconds_reaches_the_process_group(self, _mock_set, mock_init):
        """A declared timeout must be what init_process_group receives."""
        import datetime

        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict("os.environ", env, clear=True):
            setup_distributed(backend="gloo", timeout_seconds=90)

        assert mock_init.call_args.kwargs["timeout"] == datetime.timedelta(seconds=90)

    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_no_timeout_leaves_torchs_default(self, _mock_set, mock_init):
        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict("os.environ", env, clear=True):
            setup_distributed(backend="gloo")

        assert "timeout" not in mock_init.call_args.kwargs

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_nccl_binds_the_group_to_this_ranks_device(
        self, _mock_set, mock_init, _mock_avail
    ):
        """The planted shape (CLAUDE.md #15).

        Without ``device_id`` every device-taking collective infers a device
        from the current context, which emits "barrier(): using the device
        under current context" and, worse, lets a rank barrier on a device its
        peers did not pick.
        """
        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "3", "WORLD_SIZE": "4", "LOCAL_RANK": "2"}
        with patch.dict("os.environ", env, clear=True):
            setup_distributed(backend="nccl")

        device = mock_init.call_args.kwargs["device_id"]
        assert device.type == "cuda"
        # The NODE-LOCAL index, not the global rank: on a multi-node run
        # ``device_id=rank`` names a GPU that does not exist on this node.
        assert device.index == 2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.distributed.init_process_group")
    @patch("torch.cuda.set_device")
    def test_gloo_is_not_handed_a_cuda_device(self, _mock_set, mock_init, _mock_avail):
        """The other shape: ``gloo`` runs on CPU and rejects a CUDA
        ``device_id``, so the bind must follow the backend, not CUDA
        availability."""
        from spectramr.pipelines.distributed import setup_distributed

        env = {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}
        with patch.dict("os.environ", env, clear=True):
            setup_distributed(backend="gloo")

        assert "device_id" not in mock_init.call_args.kwargs


class TestCleanupDistributed:
    """Tests for cleanup_distributed()."""

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_destroys_group(self, mock_destroy, _mock_init):
        from spectramr.pipelines.distributed import cleanup_distributed

        cleanup_distributed()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_noop_when_not_initialized(self, mock_destroy, _mock_init):
        from spectramr.pipelines.distributed import cleanup_distributed

        cleanup_distributed()
        mock_destroy.assert_not_called()


def _ddp_settings(**overrides):
    """A REAL frozen ``TrainingSettings`` declaring ``parallel.strategy='ddp'``.

    These tests used to build the config as a ``types.SimpleNamespace``, on the
    premise that an already-``ddp`` strategy meant ``model_copy`` was never
    reached. That premise expired: the launcher no longer rewrites ``strategy``
    but it ALWAYS stamps the observed ``num_devices``/``num_nodes``, so
    ``settings.model_copy(update={"parallel": parallel.model_copy(...)})`` runs
    unconditionally and a SimpleNamespace raised ``AttributeError: 'SimpleNamespace'
    object has no attribute 'model_copy'`` (#631). A real settings object is also
    the only double that exercises the nested frozen-block copy this code depends
    on (non-negotiable #1).
    """
    from spectramr.config.settings import TrainingSettings

    return TrainingSettings(
        model={"model_type": "standard_unet", "in_channels": 1, "out_channels": 1},
        training={"training_mode": "reconstruction"},
        data={"batch_size": 2},
        optimization={},
        logging={},
        parallel={"strategy": "ddp", "num_devices": 1},
        **overrides,
    )


class TestRunDistributedTrainingOverrides:
    """Regression tests for ``run_distributed_training`` override handling.

    Bug: ``run_distributed_training`` previously imported a nonexistent
    ``spectramr.main._apply_overrides``, so every override-bearing DDP run raised
    ``ImportError``. The fix imports the real ``spectramr.main.apply_overrides``.
    These tests isolate that single code path: heavy deps (process-group setup,
    config YAML loading, the training pipeline) are mocked, and we assert the
    override is applied without an ImportError and the resulting settings reach
    ``run_training_pipeline``.
    """

    def _patches(self, settings_obj, run_pipeline_mock, apply_overrides_mock):
        """Build the standard patch stack as a list of context managers."""
        return [
            # No real process group / CUDA on a CPU-only test box.
            patch(
                "spectramr.pipelines.distributed.setup_distributed",
                return_value=(0, 1),
            ),
            patch("spectramr.pipelines.distributed.cleanup_distributed"),
            # Lazy imports inside run_distributed_training resolve from these:
            patch(
                "spectramr.config.settings.TrainingSettings.from_yaml",
                return_value=settings_obj,
            ),
            patch(
                "spectramr.pipelines.train.run_training_pipeline",
                run_pipeline_mock,
            ),
            # ``run_distributed_training`` imports apply_overrides RIGHTWARD from
            # the config layer (config/overrides.py), not leftward from main —
            # so the patch target is config.overrides, not main (CLAUDE.md #13).
            patch("spectramr.config.overrides.apply_overrides", apply_overrides_mock),
            # Force LOCAL_RANK so device string construction is deterministic.
            patch.dict("os.environ", {"LOCAL_RANK": "0"}, clear=False),
        ]

    def test_overrides_do_not_raise_import_error(self):
        """A one-element override list must NOT raise ImportError.

        This is the core regression: the lazy ``from spectramr.main import
        apply_overrides`` must resolve to a real symbol. We use a passthrough
        mock so the import target is exercised but config validation is skipped.
        """
        from spectramr.pipelines.distributed import run_distributed_training

        settings = _ddp_settings()
        run_pipeline = MagicMock(return_value={"status": "ok"})
        apply_overrides = MagicMock(return_value=settings)

        from contextlib import ExitStack

        with ExitStack() as stack:
            for cm in self._patches(settings, run_pipeline, apply_overrides):
                stack.enter_context(cm)
            try:
                result = run_distributed_training(
                    config_path="dummy.yaml",
                    backend="gloo",
                    overrides=["optimization.optimizer.learning_rate=1e-4"],
                )
            except ImportError as exc:  # pragma: no cover - regression guard
                pytest.fail(
                    f"run_distributed_training raised ImportError on override "
                    f"path (the _apply_overrides regression): {exc}"
                )

        # Override resolver was reached with our one-element list.
        apply_overrides.assert_called_once()
        _settings_arg, overrides_arg = apply_overrides.call_args.args
        assert overrides_arg == ["optimization.optimizer.learning_rate=1e-4"]
        assert result == {"status": "ok"}

    def test_override_reaches_training_pipeline(self):
        """The settings returned by apply_overrides must be forwarded downstream.

        ``apply_overrides`` returns a settings object carrying a distinct marker
        (``seed``); we assert the object reaching ``run_training_pipeline`` has
        it — proving the override propagates rather than the pre-override
        settings being used. Identity is deliberately NOT asserted: the launcher
        legitimately re-copies the object to stamp the observed topology, so an
        ``is`` check would only re-test that copying is skipped.
        """
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        # `seed` is a fact about the RUN. Both `training.seed` and the top-level `seed`
        # were retired to `run.seed` on 2026-07-31 with posture=raise, so a fixture
        # declaring either fails to construct.
        pre_settings = _ddp_settings(run={"seed": 1})
        post_settings = _ddp_settings(run={"seed": 4242})  # the marker
        run_pipeline = MagicMock(return_value={"status": "ok"})
        apply_overrides = MagicMock(return_value=post_settings)

        with ExitStack() as stack:
            for cm in self._patches(pre_settings, run_pipeline, apply_overrides):
                stack.enter_context(cm)
            run_distributed_training(
                config_path="dummy.yaml",
                backend="gloo",
                overrides=["optimization.optimizer.learning_rate=1e-4"],
            )

        # apply_overrides was given the freshly-loaded (pre-override) settings...
        settings_arg, overrides_arg = apply_overrides.call_args.args
        assert settings_arg is pre_settings
        assert overrides_arg == ["optimization.optimizer.learning_rate=1e-4"]

        # ...and the POST-override settings are what the pipeline runs on.
        run_pipeline.assert_called_once()
        forwarded_settings = run_pipeline.call_args.args[0]
        assert forwarded_settings.run.seed == 4242
        # The strategy DECLARATION survives; only the observed topology is stamped.
        assert forwarded_settings.parallel.strategy == "ddp"

    def test_launcher_stamps_observed_topology_without_mutating_the_original(self):
        """``num_devices``/``num_nodes`` are observed facts, so the launcher
        overwrites them — but through ``model_copy`` on the frozen block, leaving
        the caller's settings untouched (non-negotiable #1)."""
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        settings = _ddp_settings()
        run_pipeline = MagicMock(return_value={"status": "ok"})
        apply_overrides = MagicMock(return_value=settings)

        with ExitStack() as stack:
            for cm in self._patches(settings, run_pipeline, apply_overrides):
                stack.enter_context(cm)
            stack.enter_context(
                patch.dict(
                    "os.environ",
                    {"LOCAL_WORLD_SIZE": "2", "LOCAL_RANK": "0"},
                    clear=False,
                )
            )
            run_distributed_training(config_path="dummy.yaml", backend="gloo")

        forwarded = run_pipeline.call_args.args[0]
        # setup_distributed is patched to (rank=0, world_size=1); LOCAL_WORLD_SIZE=2.
        assert forwarded.parallel.num_devices == 2
        assert forwarded.parallel.num_nodes == 1
        # The frozen original is unchanged — the copy did not write through.
        assert settings.parallel.num_devices == 1

    @pytest.mark.parametrize(
        ("declared", "match"),
        [("none", "parallel.strategy is 'none'"), ("dp", "single-process")],
    )
    def test_launcher_refuses_non_ddp_strategies(self, declared, match):
        """``strategy`` is a declaration, not an observation: the launcher raises
        rather than silently rewriting it (that rewrite is what made 'fsdp' and
        'deepspeed' unreachable from ``train-distributed``)."""
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        settings = _ddp_settings()
        settings = settings.model_copy(
            update={
                "parallel": settings.parallel.model_copy(update={"strategy": declared})
            }
        )
        run_pipeline = MagicMock(return_value={"status": "ok"})

        with ExitStack() as stack:
            for cm in self._patches(settings, run_pipeline, MagicMock()):
                stack.enter_context(cm)
            with pytest.raises(ValueError, match=match):
                run_distributed_training(config_path="dummy.yaml", backend="gloo")

        run_pipeline.assert_not_called()

    def test_real_apply_overrides_is_importable(self):
        """Smoke: the symbol the fix points at actually exists and is callable.

        Guards against the import target silently disappearing again. We do NOT
        run the full pipeline here — just confirm the name resolves. The canonical
        home is now ``config/overrides.py`` (imported rightward from pipelines);
        ``spectramr.main`` re-exports the SAME object for backward compatibility.
        """
        from spectramr.config.overrides import apply_overrides as canonical
        from spectramr.main import apply_overrides as re_exported

        assert callable(canonical)
        assert re_exported is canonical

    def test_distributed_imports_apply_overrides_rightward(self):
        """``pipelines/distributed.py`` must NOT import from ``spectramr.main``.

        pipelines→main is a leftward (entry-layer) import that violates the
        inward-only dependency rule; the override util lives in the config layer.
        """
        import inspect

        from spectramr.pipelines import distributed

        src = inspect.getsource(distributed)
        assert "from spectramr.main import apply_overrides" not in src
        assert "from spectramr.config.overrides import apply_overrides" in src


class TestExecutionLedgerIsArmed:
    """The distributed launcher must arm the ledger like ``main.py`` does.

    ``main.py`` calls ``ExecutionLedger.begin_run`` before loading the config in
    all four single-process entry points; ``run_distributed_training`` did not.
    The first consumer then hit ``current_or_begin``, which self-arms with a
    warning -- printed once per rank, four times on a 4-GPU node -- and stamps
    ``ledger armed late`` into the run's own provenance. The substitutions made
    during ``from_yaml`` are gone by then, so a late ledger is not merely late:
    it is empty, and the artifact says "incomplete, not empty".
    """

    def _run(self, run_pipeline):
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        settings = _ddp_settings()
        with ExitStack() as stack:
            for cm in TestRunDistributedTrainingOverrides()._patches(
                settings, run_pipeline, MagicMock(return_value=settings)
            ):
                stack.enter_context(cm)
            return run_distributed_training(config_path="dummy.yaml", backend="gloo")

    def test_ledger_is_armed_before_the_config_loads(self):
        from spectramr.core.execution_ledger import ExecutionLedger

        seen = {}

        def _capture(*_args, **_kwargs):
            ledger = ExecutionLedger.current()
            seen["ledger"] = ledger
            seen["notes"] = list(ledger.notes) if ledger is not None else None
            return {"status": "ok"}

        assert self._run(MagicMock(side_effect=_capture)) == {"status": "ok"}

        assert seen["ledger"] is not None, (
            "no ledger was armed for the distributed run; every config-load "
            "substitution went unrecorded"
        )
        # Existence alone false-passes: ``current_or_begin`` also produces a
        # non-None ledger, just an empty one that announces its own gap. The
        # note is what distinguishes "armed" from "armed too late".
        assert not [n for n in seen["notes"] if "armed late" in n], (
            f"ledger was self-armed by a consumer, not by the launcher: "
            f"{seen['notes']}"
        )

    def test_ledger_source_names_the_config(self):
        """Provenance must attribute the run to its config, not to a consumer."""
        from spectramr.core.execution_ledger import ExecutionLedger

        seen = {}

        def _capture(*_args, **_kwargs):
            seen["source"] = ExecutionLedger.current().source
            return {"status": "ok"}

        self._run(MagicMock(side_effect=_capture))
        assert seen["source"] == "dummy.yaml"


class TestIdleDeviceRefusal:
    """The predicate behind the #1274 guard: allocated GPUs no rank will use.

    ``experiment_11_attention_none`` was launched onto a ``--gpus=4`` allocation
    by a wrapper that hardcoded ``nproc_per_node=1``. It trained for 41 minutes
    with three cards idle, and nothing objected: the process group initialised,
    DeepSpeed adopted, the health report said 141/141, and provenance recorded
    both numbers in one JSON object without ever subtracting them.

    The predicate is pure, so the whole truth table is reachable on a CPU box.
    Its most important property is the one that is easy to lose: it must not fire
    on a CORRECT launch, because a false refusal on a cluster costs a queue slot.
    """

    @staticmethod
    def _refuse(**kw):
        from spectramr.pipelines.distributed import idle_device_refusal

        base = {
            "allocated_on_node": None,
            "visible": None,
            "local_world_size": 1,
            "allow_idle_devices": False,
        }
        return idle_device_refusal(**{**base, **kw})

    def test_the_incident_is_refused(self):
        """4 allocated, 4 visible, 1 rank -- the exact shape of job 17762324."""
        msg = self._refuse(allocated_on_node=4, visible=4, local_world_size=1)
        assert msg is not None
        # Both counts and both exits must be in the text: an operator reading it
        # on a cluster has no other source for either.
        assert "4 GPU(s)" in msg and "1 rank(s)" in msg and "3 allocated" in msg
        assert "--nproc_per_node=4" in msg
        assert "parallel.allow_idle_devices=true" in msg

    def test_matched_ranks_and_devices_pass(self):
        """4 ranks on 4 GPUs is the shape the guard exists to protect."""
        assert self._refuse(allocated_on_node=4, visible=4, local_world_size=4) is None

    def test_no_scheduler_grant_is_not_a_finding(self):
        """A workstation's idle cards are nobody's allocation.

        ``ALLOC_GPU_ENV`` is unset off-scheduler, so ``allocated_on_node`` is
        ``None``. Firing here would break every local single-rank debug run on a
        multi-GPU box -- and there is no grant being wasted.
        """
        assert (
            self._refuse(allocated_on_node=None, visible=4, local_world_size=1) is None
        )

    def test_gpu_bind_masking_is_not_a_finding(self):
        """``srun --gpu-bind=single:1``: 4 allocated on the node, 1 visible here.

        The other three belong to sibling ranks, not to nobody. Bounding the
        allocation by the visible count is what keeps this correct launch quiet;
        comparing the raw grant against this task's rank count would refuse it.
        """
        assert self._refuse(allocated_on_node=4, visible=1, local_world_size=1) is None

    def test_unprobeable_visible_count_still_refuses_on_the_grant(self):
        """``visible=None`` means torch could not be asked, not that there is one.

        The grant is a scheduler fact and survives a failed torch probe, so the
        check falls back to it rather than going quiet -- going quiet is the
        behaviour this issue is about.
        """
        assert (
            self._refuse(allocated_on_node=4, visible=None, local_world_size=1)
            is not None
        )

    def test_more_ranks_than_devices_is_left_to_the_collision_detector(self):
        """The reverse imbalance is NOT decided here.

        ``provenance._rank_device_record`` can see every rank's resolved device
        and so tells real sharing apart from a per-rank mask; this predicate
        would only be guessing.
        """
        assert self._refuse(allocated_on_node=2, visible=2, local_world_size=4) is None

    def test_the_opt_out_silences_it(self):
        assert (
            self._refuse(
                allocated_on_node=4,
                visible=4,
                local_world_size=1,
                allow_idle_devices=True,
            )
            is None
        )


class TestIdleDeviceGuardWiring:
    """The launcher must consult the predicate, and do it before it forgets.

    Two properties that the pure-predicate tests above cannot cover:

    1. The guard reads the ``core.resources`` probes -- the same implementation
       ``RunTopology`` stamps into provenance. If it re-read the environment
       itself, the ``RunTopology`` GPU fields would be reporting-only decoration,
       the exact anti-pattern ``core/resources.py`` was created to end. The patch
       targets below are what pins that: they are the core functions, not a
       launcher-local helper.
    2. It runs UPSTREAM of the ``model_copy`` that overwrites ``num_devices``
       with the observed rank count. Downstream of that, the declaration a reader
       would compare against is already gone.
    """

    @staticmethod
    def _settings(*, allow_idle: bool):
        from spectramr.config.settings import TrainingSettings

        return TrainingSettings(
            model={"model_type": "standard_unet", "in_channels": 1, "out_channels": 1},
            training={"training_mode": "reconstruction"},
            data={"batch_size": 2},
            optimization={},
            logging={},
            parallel={
                "strategy": "ddp",
                "num_devices": 1,
                "allow_idle_devices": allow_idle,
            },
        )

    def _stack(self, settings, run_pipeline, *, allocated, visible):
        return [
            patch(
                "spectramr.pipelines.distributed.setup_distributed",
                return_value=(0, 1),
            ),
            patch("spectramr.pipelines.distributed.cleanup_distributed"),
            patch(
                "spectramr.config.settings.TrainingSettings.from_yaml",
                return_value=settings,
            ),
            patch("spectramr.pipelines.train.run_training_pipeline", run_pipeline),
            # Patched at their home module: the launcher imports them lazily, so
            # the names do not exist in `pipelines.distributed`'s namespace. That
            # these are the CORE probes is the point -- see the class docstring.
            patch(
                "spectramr.core.resources.allocated_gpus_per_node",
                return_value=(allocated, "env:SLURM_GPUS_ON_NODE"),
            ),
            patch(
                "spectramr.core.resources.visible_gpu_count",
                return_value=visible,
            ),
            patch.dict("os.environ", {"LOCAL_RANK": "0"}, clear=False),
        ]

    def test_launcher_refuses_the_incident_shape(self):
        """One rank against a 4-GPU grant must not reach the training pipeline."""
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import (
            IdleDeviceError,
            run_distributed_training,
        )

        run_pipeline = MagicMock(return_value={"status": "ok"})
        with ExitStack() as stack:
            for cm in self._stack(
                self._settings(allow_idle=False),
                run_pipeline,
                allocated=4,
                visible=4,
            ):
                stack.enter_context(cm)
            with pytest.raises(IdleDeviceError, match="3 allocated GPU"):
                run_distributed_training(config_path="dummy.yaml", backend="gloo")

        # The refusal is worth nothing if the run started anyway.
        run_pipeline.assert_not_called()

    def test_the_opt_out_lets_the_launch_through(self):
        """``allow_idle_devices`` is the acknowledged-debug-run escape hatch."""
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        run_pipeline = MagicMock(return_value={"status": "ok"})
        with ExitStack() as stack:
            for cm in self._stack(
                self._settings(allow_idle=True),
                run_pipeline,
                allocated=4,
                visible=4,
            ):
                stack.enter_context(cm)
            result = run_distributed_training(config_path="dummy.yaml", backend="gloo")

        assert result == {"status": "ok"}
        run_pipeline.assert_called_once()

    def test_a_matched_launch_is_untouched(self):
        """No grant in the environment -> the guard must be inert.

        This is the regression that protects every existing caller: the guard was
        added to a function four other suites already drive with mocks, and an
        over-eager predicate would redden all of them on any GPU-bearing box.
        """
        from contextlib import ExitStack

        from spectramr.pipelines.distributed import run_distributed_training

        run_pipeline = MagicMock(return_value={"status": "ok"})
        with ExitStack() as stack:
            for cm in self._stack(
                self._settings(allow_idle=False),
                run_pipeline,
                allocated=None,
                visible=4,
            ):
                stack.enter_context(cm)
            result = run_distributed_training(config_path="dummy.yaml", backend="gloo")

        assert result == {"status": "ok"}
