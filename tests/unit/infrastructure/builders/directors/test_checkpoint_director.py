"""Paired tests for CheckpointDirector BuilderContext migration (WS-D).

CheckpointDirector is a config-only builder that was migrated to the canonical
``def __init__(self, ctx: BuilderContext)`` convention behind the Phase-0
back-compat shim (:func:`accepts_builder_context`). These tests assert it can be
constructed BOTH the legacy way (``CheckpointDirector(config)``) and the
canonical way (``CheckpointDirector(BuilderContext(config=config))``), and that
both forms store the same config.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest

import torch
import torch.nn as nn

from spectramr.infrastructure.builders.directors import (
    checkpoint_director as checkpoint_director_module,
)

from spectramr.infrastructure.builders.context import BuilderContext

try:
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.builders.directors.checkpoint_director import (
        CheckpointDirector,
        CheckpointState,
        _native_tag_for,
    )
    from spectramr.infrastructure.distributed.checkpoint_adapters import (
        DeepSpeedCheckpointAdapter,
    )
    from spectramr.infrastructure.distributed.strategy_registry import ParallelRuntime
except ImportError:  # pragma: no cover - import guard mirrors sibling director tests
    pytest.skip("CheckpointDirector not available", allow_module_level=True)


@pytest.fixture
def config() -> Mock:
    """Minimal stub config; CheckpointDirector.__init__ only stores it."""
    return Mock(spec=TrainingSettings)


def test_init_accepts_legacy_config_arg(config: Mock) -> None:
    """Legacy callers `CheckpointDirector(config)` still work via the shim."""
    director = CheckpointDirector(config)
    assert director._config is config


def test_init_accepts_builder_context(config: Mock) -> None:
    """Canonical callers pass a BuilderContext."""
    ctx = BuilderContext(config=config)
    director = CheckpointDirector(ctx)
    assert director._config is config


def test_both_forms_produce_equivalent_state(config: Mock) -> None:
    """Both construction shapes yield the same stored config + initial state."""
    legacy = CheckpointDirector(config)
    canonical = CheckpointDirector(BuilderContext(config=config))

    assert legacy._config is canonical._config is config
    # Initial builder state is identical regardless of construction shape.
    assert legacy._checkpoint_dir == canonical._checkpoint_dir is None
    assert legacy._pipeline == canonical._pipeline is None
    assert legacy._epoch == canonical._epoch == 0
    assert legacy._global_step == canonical._global_step == 0
    assert legacy._metrics == canonical._metrics == {}
    assert legacy._strategy == canonical._strategy is None


def test_legacy_construction_is_silent(config: Mock) -> None:
    """The compat shim must not emit a DeprecationWarning for legacy calls."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        CheckpointDirector(config)  # must not raise


def test_init_carries_builder_context_marker() -> None:
    """The migrated __init__ is tagged by the accepts_builder_context decorator."""
    assert (
        getattr(CheckpointDirector.__init__, "__accepts_builder_context__", False)
        is True
    )


# `get_best_checkpoint` selection tests removed with the method (#710): it had
# no caller, and had one been added it would have CRASHED -- training_loop passes
# `.with_metrics(losses_history)`, so checkpoint_best.pt['metrics'] holds LOSS
# keys, and the selector resolved them through `metric_higher_is_better`, which
# raises on an unknown key like `g_adv`. The live discovery path is
# `checkpoint_service.discover_best_checkpoint`, which does no metric ranking.

torch = pytest.importorskip("torch")


def _write_ckpt(path, metrics):
    torch.save({"metrics": metrics}, path)










class TestScalerSSOT:
    """``scaler_state`` used to describe an object no backward pass touched.

    ``OptimizationBuilder.build_grad_scaler`` handed the pipeline a plain
    ``GradScaler("cuda")``; the scaler that actually scales losses is built by
    ``MixedPrecisionIntegrationHelper`` on the strategy and is a different
    class. So every fp16 checkpoint stored an untouched scale of 65536 while
    the live dynamic scale was dropped -- and on resume the run re-converged
    its scale from scratch, overflowing and skipping steps, with a
    well-formed ``scaler_state`` key present the whole time.
    """

    @staticmethod
    def _director():
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        return CheckpointDirector.__new__(CheckpointDirector)

    def test_the_strategy_s_live_scaler_wins_over_the_passed_one(self):
        from types import SimpleNamespace

        director = self._director()
        dead = SimpleNamespace(state_dict=lambda: {"scale": 65536.0})
        live = SimpleNamespace(state_dict=lambda: {"scale": 128.0})
        director._scaler = dead
        director._strategy = SimpleNamespace(amp_helper=SimpleNamespace(scaler=live))
        assert director._resolve_scaler() is live

    def test_falls_back_to_the_passed_scaler_when_no_strategy(self):
        from types import SimpleNamespace

        director = self._director()
        passed = SimpleNamespace(state_dict=lambda: {"scale": 4.0})
        director._scaler = passed
        director._strategy = None
        assert director._resolve_scaler() is passed

    def test_a_strategy_without_amp_returns_none(self):
        from types import SimpleNamespace

        director = self._director()
        director._scaler = None
        director._strategy = SimpleNamespace(amp_helper=SimpleNamespace(scaler=None))
        assert director._resolve_scaler() is None

    def test_build_grad_scaler_no_longer_fabricates_one(self):
        """The builder step survives as a no-op (it is part of the documented
        public chain) but must not put a second scaler into circulation."""
        from spectramr.infrastructure.training.builders.optimization_builder import (
            OptimizationBuilder,
        )

        builder = OptimizationBuilder.__new__(OptimizationBuilder)
        builder._scaler = None
        assert builder.build_grad_scaler() is builder
        assert builder._scaler is None


class TestParallelRuntimeWiring:
    """``with_parallel_runtime`` decides who gathers and who writes."""

    @staticmethod
    def _director():
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        return CheckpointDirector.__new__(CheckpointDirector)

    def test_unset_runtime_yields_the_default_adapter(self):
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            DefaultCheckpointAdapter,
        )

        director = self._director()
        director._parallel = None
        assert isinstance(director._adapter, DefaultCheckpointAdapter)

    def test_an_fsdp_runtime_yields_the_gathering_adapter(self):
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            FSDPCheckpointAdapter,
        )
        from spectramr.infrastructure.distributed.strategy_registry import ParallelRuntime

        director = self._director()
        director._parallel = ParallelRuntime(
            strategy="fsdp", checkpoint_adapter=FSDPCheckpointAdapter()
        )
        assert director._adapter.requires_all_ranks is True

    def test_collect_models_does_not_unwrap(self):
        """The adapter needs the ENGINE / the FSDP root -- handing it the bare
        module underneath would make save_native a silent no-op."""
        from types import SimpleNamespace

        import torch.nn as nn

        inner = nn.Linear(2, 2)
        wrapper = SimpleNamespace(module=inner, save_checkpoint=lambda *a, **k: None)
        director = self._director()
        director._pipeline = SimpleNamespace(generator=wrapper, discriminator=None)
        collected = director._collect_models()
        assert collected["generator"] is wrapper
        assert "discriminator" not in collected


class TestBypassedInitIsTolerated:
    """Tests in this area construct the director via ``__new__``.

    A hard ``self._parallel`` / ``self._scaler`` access turns every one of them
    into an ``AttributeError`` raised from INSIDE the save path, where it is
    re-wrapped as ``RuntimeError: Failed to save checkpoint`` -- which reads as
    a checkpointing bug rather than a missing attribute. Same reason
    ``_strategy`` has always been read with ``getattr``.
    """

    @staticmethod
    def _bare():
        from spectramr.infrastructure.builders.directors.checkpoint_director import (
            CheckpointDirector,
        )

        return CheckpointDirector.__new__(CheckpointDirector)

    def test_adapter_resolves_without_init(self):
        from spectramr.infrastructure.distributed.checkpoint_adapters import (
            DefaultCheckpointAdapter,
        )

        assert isinstance(self._bare()._adapter, DefaultCheckpointAdapter)

    def test_scaler_resolves_without_init(self):
        assert self._bare()._resolve_scaler() is None

    def test_collect_models_without_a_pipeline(self):
        director = self._bare()
        director._pipeline = None
        assert director._collect_models() == {}


# ---------------------------------------------------------------------------
# Fluent setters — relocated from tests/unit/builders/test_phase2_directors.py
# ---------------------------------------------------------------------------
#
# That file was deleted with the unreachable director tree it mostly covered
# (TrainingPipelineDirector / InferencePipelineDirector / ExperimentDirector).
# These seven are the part of it that tested a LIVE component and had no
# counterpart here: this file covered construction shapes, scaler resolution and
# model collection, but not the builder API that ``pipelines/training_loop.py``
# actually chains. Deleting the file without moving them would have retired live
# coverage under cover of a dead-code removal.


@pytest.fixture
def dummy_model() -> nn.Module:
    return nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))


class TestCheckpointDirectorFluentSetters:
    """The chained ``with_*`` calls used at training_loop.py:179 and train.py:729."""

    def test_with_checkpoint_dir(self, config: Mock, tmp_path) -> None:
        director = CheckpointDirector(config)
        director.with_checkpoint_dir(str(tmp_path))
        assert director._checkpoint_dir == Path(str(tmp_path))

    def test_with_epoch(self, config: Mock) -> None:
        director = CheckpointDirector(config)
        director.with_epoch(5)
        assert director._epoch == 5

    def test_invalid_epoch_raises(self, config: Mock) -> None:
        """A negative epoch must raise rather than be stored (non-negotiable #3)."""
        director = CheckpointDirector(config)
        with pytest.raises(ValueError):
            director.with_epoch(-1)

    def test_with_global_step(self, config: Mock) -> None:
        director = CheckpointDirector(config)
        director.with_global_step(100)
        assert director._global_step == 100

    def test_with_metrics(self, config: Mock) -> None:
        director = CheckpointDirector(config)
        director.with_metrics({"loss": 0.5, "psnr": 30.2})
        assert director._metrics["loss"] == 0.5
        assert director._metrics["psnr"] == 30.2

    def test_checkpoint_state_size(self, dummy_model: nn.Module) -> None:
        state = CheckpointState(
            epoch=1,
            global_step=100,
            generator_state=dummy_model.state_dict(),
            discriminator_state=None,
            optimizer_g_state={},
            optimizer_d_state=None,
            scheduler_g_state=None,
            scheduler_d_state=None,
            metrics={"loss": 0.5},
        )
        assert state.get_total_size_mb() > 0


# ---------------------------------------------------------------------------
# Restoring a strategy-native ("consolidated") best checkpoint.
#
# A DeepSpeed arm's ``save_best`` writes the sharded tag directory plus a
# consolidated file, and then RETURNS before the generic ``torch.save`` payload
# (checkpoint_director's ``writes_native_artifact`` guard). So ``checkpoint_best.pt``
# is a bare parameter-keyed state_dict with no ``generator`` key, and the restore
# path used to parse it as a generic payload -- KeyError('generator'), after an
# 8h45m run, with the best weights on disk the whole time.
#
# Everything here is CPU-runnable and needs no deepspeed: the adapter duck-types
# on ``save_checkpoint`` precisely so this is testable without the extra.
# ---------------------------------------------------------------------------

class _FakeDeepSpeedEngine(nn.Module):
    """A DeepSpeed engine by duck-typing only -- the three methods the adapter calls.

    ``load_checkpoint`` returns ``(None, None)`` on a miss, which is how the real
    engine reports a missing tag: it does not raise.
    """

    def __init__(self) -> None:
        super().__init__()
        self.module = nn.Linear(4, 4)

    def _shard(self, root: Path, tag: str) -> Path:
        return Path(root) / tag / "mp_rank_00_model_states.pt"

    def save_checkpoint(self, save_dir, tag, client_state=None):
        f = self._shard(save_dir, tag)
        f.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"module": self.module.state_dict(), "client_state": dict(client_state or {})},
            f,
        )

    def save_16bit_model(self, save_dir, filename):
        # The real call takes a directory and a filename and writes a BARE
        # state_dict -- no wrapper key. That shape is the whole bug.
        torch.save(self.module.state_dict(), Path(save_dir) / filename)
        return True

    def load_checkpoint(self, load_dir, tag):
        f = self._shard(load_dir, tag)
        if not f.exists():
            return None, None
        blob = torch.load(f, weights_only=False)
        self.module.load_state_dict(blob["module"])
        return str(f.parent), blob["client_state"]


def _pipeline_for(engine: _FakeDeepSpeedEngine):
    from types import SimpleNamespace

    return SimpleNamespace(
        generator=engine,
        discriminator=None,
        optimizer_g=torch.optim.SGD(engine.module.parameters(), lr=0.1),
        optimizer_d=None,
        scheduler_g=None,
        scheduler_d=None,
        ema=None,
        scaler=None,
        device="cpu",
    )


def _deepspeed_runtime() -> ParallelRuntime:
    return ParallelRuntime(
        strategy="deepspeed",
        checkpoint_adapter=DeepSpeedCheckpointAdapter(save_consolidated_best=True),
    )


def _save_best_the_way_the_loop_does(config, tmp_path, engine, pipeline) -> Path:
    """Mirrors the writer chain at training_loop.py:1897."""
    return (
        CheckpointDirector(config)
        .with_checkpoint_dir(str(tmp_path))
        .with_pipeline(pipeline)
        .with_epoch(5)
        .with_global_step(4000)
        .with_metrics({"val_hfen_mean": 1.30})
        .with_parallel_runtime(_deepspeed_runtime())
        .validate()
        .save_best(metric_name="val_hfen_mean", metric_value=1.30)
    )


class TestNativeTagDerivation:
    """``latest`` is a FILE recording the newest tag, never a tag itself."""

    @pytest.mark.parametrize(
        ("filename", "expected"),
        [
            ("checkpoint_best.pt", "best"),
            ("best.pt", "best"),  # the _publish_best_alias name
            ("checkpoint_epoch_0005_step_004000.pt", "epoch_0005_step_004000"),
        ],
    )
    def test_tag_matches_what_the_writer_wrote(self, filename, expected) -> None:
        assert _native_tag_for(Path("/run/checkpoints") / filename) == expected


class TestNativeConsolidatedBestCheckpoint:
    def test_save_best_writes_a_bare_state_dict(self, config, tmp_path) -> None:
        """Characterises the artifact: no 'generator' key, parameter names only."""
        engine = _FakeDeepSpeedEngine()
        path = _save_best_the_way_the_loop_does(
            config, tmp_path, engine, _pipeline_for(engine)
        )

        blob = torch.load(path, weights_only=False)
        assert "generator" not in blob
        assert set(blob) == set(engine.module.state_dict())

    def test_restore_recovers_the_best_weights(self, config, tmp_path) -> None:
        """The regression. On the unfixed code this raised KeyError('generator')."""
        engine = _FakeDeepSpeedEngine()
        pipeline = _pipeline_for(engine)
        best = engine.module.weight.detach().clone()

        path = _save_best_the_way_the_loop_does(config, tmp_path, engine, pipeline)

        # Training continues past the best iterate and degrades the weights.
        with torch.no_grad():
            engine.module.weight.zero_()

        director = (
            CheckpointDirector(config)
            .with_pipeline(pipeline)
            .with_parallel_runtime(_deepspeed_runtime())
        )
        assert director.load_from(str(path)) is True
        assert torch.equal(engine.module.weight, best)
        # Metadata has only one possible source here: the tag's client_state.
        assert director._epoch == 5
        assert director._global_step == 4000

    def test_restore_without_the_runtime_names_the_missing_wiring(
        self, config, tmp_path
    ) -> None:
        """The exact shape of the reported crash, now a diagnosable error.

        Omitting ``with_parallel_runtime`` resolves DefaultCheckpointAdapter, so
        nothing can read the consolidated export. That must say so rather than
        raise KeyError('generator') from a dict index 80 lines away.
        """
        engine = _FakeDeepSpeedEngine()
        pipeline = _pipeline_for(engine)
        path = _save_best_the_way_the_loop_does(config, tmp_path, engine, pipeline)

        director = CheckpointDirector(config).with_pipeline(pipeline)
        with pytest.raises(RuntimeError, match="with_parallel_runtime"):
            director.load_from(str(path))

    def test_a_missing_tag_directory_is_not_a_successful_restore(
        self, config, tmp_path
    ) -> None:
        """Silent-fallback guard (non-negotiable 3).

        With the generic parse now skipped on the native path, a tag directory
        that never landed must fail loudly -- otherwise the run reports
        'restored best weights' while holding the last iterate.
        """
        engine = _FakeDeepSpeedEngine()
        pipeline = _pipeline_for(engine)
        path = _save_best_the_way_the_loop_does(config, tmp_path, engine, pipeline)

        import shutil

        shutil.rmtree(tmp_path / "deepspeed")

        director = (
            CheckpointDirector(config)
            .with_pipeline(pipeline)
            .with_parallel_runtime(_deepspeed_runtime())
        )
        with pytest.raises((FileNotFoundError, RuntimeError)):
            director.load_from(str(path))

    def test_a_generic_payload_still_takes_the_generic_path(
        self, config, tmp_path
    ) -> None:
        """No regression for every non-sharded arm, and for `save`'s own output.

        ``save`` writes BOTH artifacts, so its generic half must keep being read
        -- it is the only carrier of ema/scaler/counter state.
        """
        engine = _FakeDeepSpeedEngine()
        pipeline = _pipeline_for(engine)
        best = engine.module.weight.detach().clone()

        path = tmp_path / "checkpoint_epoch_0005_step_004000.pt"
        torch.save(
            {
                "generator": engine.module.state_dict(),
                "optimizer_g": pipeline.optimizer_g.state_dict(),
                "epoch": 5,
                "global_step": 4000,
                "metrics": {"val_hfen_mean": 1.30},
            },
            path,
        )
        with torch.no_grad():
            engine.module.weight.zero_()

        director = CheckpointDirector(config).with_pipeline(pipeline)
        assert director.load_from(str(path)) is True
        assert torch.equal(engine.module.weight, best)
        assert director._global_step == 4000


class TestNativeRestorePositionIsNeverFabricated:
    """The run position comes from the tag, or the restore fails saying so.

    On the native path the consolidated file carries no metadata at all, so the
    tag's client_state is the only source for epoch/global_step. Defaulting them
    to 0 restored a week-long run's WEIGHTS while resetting its schedule
    position, logged ``epoch=0, step=0`` as though measured, and returned True.
    Nothing downstream could detect it, because 0 is also a legitimate value.
    """

    @staticmethod
    def _rewrite_client_state(tmp_path: Path, new_state: dict) -> None:
        """Make the landed tag look like one written without full client state.

        Located by glob rather than by rebuilding the layout, so the adapter stays
        free to move its tag root without silently voiding these tests.
        """
        shard = next((tmp_path / "deepspeed").rglob("mp_rank_00_model_states.pt"))
        blob = torch.load(shard, weights_only=False)
        blob["client_state"] = new_state
        torch.save(blob, shard)

    def _restore(self, config, tmp_path, client_state: dict):
        engine = _FakeDeepSpeedEngine()
        pipeline = _pipeline_for(engine)
        path = _save_best_the_way_the_loop_does(config, tmp_path, engine, pipeline)
        self._rewrite_client_state(tmp_path, client_state)

        director = (
            CheckpointDirector(config)
            .with_pipeline(pipeline)
            .with_parallel_runtime(_deepspeed_runtime())
        )
        return director, path

    def test_a_tag_without_client_state_is_not_a_silent_epoch_zero(
        self, config, tmp_path
    ) -> None:
        """The finding.

        ``load_native`` returns {} when the engines loaded from a tag that carried
        no client state -- a success for the weights, and no information at all
        about position.
        """
        director, path = self._restore(config, tmp_path, {})

        with pytest.raises(RuntimeError) as excinfo:
            director.load_from(str(path))

        message = str(excinfo.value)
        assert "epoch" in message
        assert "global_step" in message

    def test_a_partial_client_state_names_only_what_is_missing(
        self, config, tmp_path
    ) -> None:
        """A truthiness check on the whole dict would pass this one through and
        then fabricate ``global_step=0`` beside a real epoch."""
        director, path = self._restore(config, tmp_path, {"epoch": 5})

        with pytest.raises(RuntimeError) as excinfo:
            director.load_from(str(path))

        assert "global_step" in str(excinfo.value)

    def test_a_genuine_epoch_zero_checkpoint_still_restores(self, config, tmp_path) -> None:
        """The reason the absent case cannot be detected by its value.

        A checkpoint saved at epoch 0 step 0 is legitimate -- an arm that best-ed
        on its first validation. It must load, which is why the guard tests for
        key PRESENCE and never for a falsy value.
        """
        director, path = self._restore(config, tmp_path, {"epoch": 0, "global_step": 0})

        assert director.load_from(str(path)) is True
        assert director._epoch == 0
        assert director._global_step == 0

    def test_absent_metrics_and_counter_state_stay_defaulted(self, config, tmp_path) -> None:
        """Not every absence is a fabrication.

        {} for metrics and None for counter_state are honest -- neither is
        mistakable for a measured value the way ``epoch=0`` is -- so the guard
        deliberately does not extend to them.
        """
        director, path = self._restore(config, tmp_path, {"epoch": 5, "global_step": 4000})

        assert director.load_from(str(path)) is True
        assert director._metrics == {}
        assert director._counter_state is None


# --------------------------------------------------------------------------- #
# #1352 -- the director wrote checkpoints non-atomically
# --------------------------------------------------------------------------- #
def test_director_publishes_checkpoints_through_the_atomic_writer():
    """No bare ``torch.save`` may remain on the checkpoint-publishing path.

    The DI container already registered an atomic writer that this director
    bypassed, so the fix is wiring, not new machinery -- and a wiring fix is
    only delivered once the production path actually calls it.
    """
    import inspect

    from spectramr.infrastructure.builders.directors import checkpoint_director

    source = inspect.getsource(checkpoint_director)
    assert "atomic_save_torch(checkpoint_data, checkpoint_path)" in source
    assert "torch.save(checkpoint_data, checkpoint_path)" not in source



# ---------------------------------------------------------------------------
# RNG round-trip (#2071 follow-up)
#
# The director is the writer the training loop actually uses; CheckpointService
# is only its exception fallback. So a resume replays the interrupted run's
# stochastic stream only if THIS pair carries it -- which it did not, while the
# service captured it and nobody called the service (non-negotiable 16).
# ---------------------------------------------------------------------------


def _rng_pipeline() -> SimpleNamespace:
    generator = nn.Linear(4, 4)
    return SimpleNamespace(
        generator=generator,
        discriminator=None,
        optimizer_g=torch.optim.SGD(generator.parameters(), lr=0.1),
        optimizer_d=None,
        scheduler_g=None,
        scheduler_d=None,
        scaler=None,
        device=torch.device("cpu"),
    )


def _save_rng_checkpoint(tmp_path) -> Path:
    return (
        CheckpointDirector(MagicMock())
        .with_checkpoint_dir(str(tmp_path / "checkpoints"))
        .with_pipeline(_rng_pipeline())
        .with_epoch(1)
        .with_global_step(40_000)
        .validate()
        .save()
    )


def _draws(n: int = 4) -> list[float]:
    return [float(torch.rand(1).item()) for _ in range(n)]


def test_save_carries_rng_state(tmp_path) -> None:
    """Without this key the resumed run cannot replay anything."""
    blob = torch.load(_save_rng_checkpoint(tmp_path), map_location="cpu", weights_only=False)

    assert "rng_state" in blob
    assert {"torch", "numpy", "python"} <= set(blob["rng_state"])


def test_resume_replays_the_interrupted_runs_stream(tmp_path) -> None:
    """The reassurance property, end to end on the production writer/reader.

    The draws after the restore must be the draws that would have followed the
    save, not merely draws from a correctly-seeded generator.
    """
    torch.manual_seed(1234)
    path = _save_rng_checkpoint(tmp_path)
    would_have_drawn = _draws()

    # A requeued job is a fresh process whose stream is wherever seeding left
    # it -- deliberately NOT where the interrupted one stopped.
    torch.manual_seed(9999)

    loaded = (
        CheckpointDirector(MagicMock())
        .with_checkpoint_dir(str(tmp_path / "checkpoints"))
        .with_pipeline(_rng_pipeline())
    )
    assert loaded.load_from(str(path)) is True

    assert _draws() == would_have_drawn


def test_non_main_rank_keeps_its_own_stream(tmp_path) -> None:
    """A shared restore would divide augmentation diversity by the world size.

    The file carries rank 0's streams only, so every rank restoring it draws an
    identical sequence -- which is what train.py's rank-offset seeding exists to
    prevent.
    """
    torch.manual_seed(1234)
    path = _save_rng_checkpoint(tmp_path)
    rank0_would_draw = _draws()

    # Built before the measured window: initialising a module draws from the
    # very stream under test, so constructing it inside would consume the
    # numbers the assertion is about.
    loaded = (
        CheckpointDirector(MagicMock())
        .with_checkpoint_dir(str(tmp_path / "checkpoints"))
        .with_pipeline(_rng_pipeline())
    )

    torch.manual_seed(9999)
    expected_own = _draws()

    torch.manual_seed(9999)
    with patch.object(
        checkpoint_director_module.RankUtility, "is_main_rank", return_value=False
    ):
        assert loaded.load_from(str(path)) is True

    drawn = _draws()
    assert drawn == expected_own
    assert drawn != rank0_would_draw


def test_checkpoint_without_rng_state_still_loads(tmp_path) -> None:
    """Every checkpoint on the cluster today predates this key."""
    path = _save_rng_checkpoint(tmp_path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    del blob["rng_state"]
    torch.save(blob, path)

    loaded = (
        CheckpointDirector(MagicMock())
        .with_checkpoint_dir(str(tmp_path / "checkpoints"))
        .with_pipeline(_rng_pipeline())
    )
    assert loaded.load_from(str(path)) is True


# ── cleanup_old_checkpoints is deprecated ────────────────────────────────────
# It has no callers and is not the retention policy the corpus configures, so
# the warning is the only thing standing between a future caller and a SECOND
# retention policy that cannot see keep_best_n (#1972 / #1973).


def _touch_epoch_checkpoints(tmp_path: Path, n: int) -> Path:
    """`n` checkpoints in the director's own filename format, oldest first."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(n):
        (ckpt_dir / f"checkpoint_epoch_{epoch:04d}_step_{epoch * 1000:06d}.pt").touch()
    return ckpt_dir


def _director_at(tmp_path: Path) -> CheckpointDirector:
    return CheckpointDirector(MagicMock()).with_checkpoint_dir(str(tmp_path / "checkpoints"))


def test_cleanup_old_checkpoints_warns_that_it_is_deprecated(tmp_path: Path) -> None:
    """The planted call must turn the warning red, or it guards nothing (NN15)."""
    _touch_epoch_checkpoints(tmp_path, 5)

    with pytest.warns(DeprecationWarning, match="cleanup_old_checkpoints is deprecated"):
        _director_at(tmp_path).cleanup_old_checkpoints(keep_last_n=3)


def test_cleanup_old_checkpoints_names_the_owner_it_defers_to(tmp_path: Path) -> None:
    """A deprecation that does not say what to use instead sends people nowhere."""
    _touch_epoch_checkpoints(tmp_path, 2)

    with pytest.warns(DeprecationWarning) as caught:
        _director_at(tmp_path).cleanup_old_checkpoints(keep_last_n=1)

    message = str(caught[0].message)
    assert "CheckpointService._apply_retention_policy" in message
    assert "keep_best_n" in message


def test_cleanup_old_checkpoints_still_keeps_the_last_n(tmp_path: Path) -> None:
    """Deprecating it must not quietly change what a surviving caller gets."""
    ckpt_dir = _touch_epoch_checkpoints(tmp_path, 5)

    with pytest.warns(DeprecationWarning):
        removed = _director_at(tmp_path).cleanup_old_checkpoints(keep_last_n=3)

    assert removed == 2
    survivors = sorted(p.name for p in ckpt_dir.glob("checkpoint_epoch_*.pt"))
    assert survivors == [
        "checkpoint_epoch_0002_step_002000.pt",
        "checkpoint_epoch_0003_step_003000.pt",
        "checkpoint_epoch_0004_step_004000.pt",
    ]


def test_cleanup_old_checkpoints_with_keep_zero_removes_nothing(tmp_path: Path) -> None:
    """Pins the edge the docstring names: `checkpoints[:-0]` is `checkpoints[:0]`.

    So "keep the last 0" silently means "keep all" -- the opposite of what the
    argument reads as. Asserted rather than fixed, because fixing it would make
    a dead method safer to call and the point is that it should not be called.
    """
    ckpt_dir = _touch_epoch_checkpoints(tmp_path, 4)

    with pytest.warns(DeprecationWarning):
        removed = _director_at(tmp_path).cleanup_old_checkpoints(keep_last_n=0)

    assert removed == 0
    assert len(list(ckpt_dir.glob("checkpoint_epoch_*.pt"))) == 4


def test_cleanup_old_checkpoints_has_no_production_callers() -> None:
    """The premise of the deprecation. If this fails, the warning now fires on a
    real run and the message must be revisited before the call is accepted."""
    src = Path("src/spectramr")
    # The defining module is excluded: it names the method in its own signature,
    # docstring and deprecation message, none of which is a call.
    definer = src / "infrastructure/builders/directors/checkpoint_director.py"
    callers = [
        f"{path}:{n}"
        for path in src.rglob("*.py")
        if path != definer
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "cleanup_old_checkpoints(" in line
    ]
    assert callers == [], f"cleanup_old_checkpoints gained a production caller: {callers}"
