"""The three checkpoint adapters, and the one flag that prevents a hung job.

The bug these exist to prevent has no exception and no log line: rank 0 enters
a collective (FSDP's state_dict all-gather, DeepSpeed's save_checkpoint) that
ranks 1..N never enter, because the training loop gated the whole checkpoint
block on ``is_main_rank()``. Every assertion here is CPU-runnable; the actual
multi-rank behaviour needs a cluster.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from spectramr.infrastructure.distributed.checkpoint_adapters import (
    DEEPSPEED_TAG_PREFIX,
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
    FSDPCheckpointAdapter,
    IParallelCheckpointAdapter,
    NativeLoadResult,
    resolve_checkpoint_adapter,
)
from spectramr.infrastructure.distributed.strategy_registry import (
    ParallelRuntime,
    list_parallel_strategies,
    resolve_parallel_strategy,
)

ALL_ADAPTERS = (
    DefaultCheckpointAdapter(),
    FSDPCheckpointAdapter(),
    DeepSpeedCheckpointAdapter(),
)


class _FakeEngine:
    """Duck-types the two DeepSpeed engine methods the adapter calls."""

    def __init__(self) -> None:
        self.saved: list[tuple[str, str]] = []
        self.loaded: list[tuple[str, str]] = []

    def save_checkpoint(self, path, tag, client_state=None):
        self.saved.append((str(path), tag))
        (Path(path) / tag).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, path, tag, **kwargs):
        self.loaded.append((str(path), tag))
        self.load_kwargs = kwargs
        return str(path), {"epoch": 3, "global_step": 300}


class TestProtocolConformance:
    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_satisfies_the_protocol(self, adapter) -> None:
        assert isinstance(adapter, IParallelCheckpointAdapter)

    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_declares_both_capability_flags(self, adapter) -> None:
        """Read as booleans by the loop and the director; a missing one would
        be ``None`` and read as False, i.e. silently back to the hang."""
        assert isinstance(adapter.requires_all_ranks, bool)
        assert isinstance(adapter.writes_native_artifact, bool)

    @pytest.mark.parametrize("adapter", ALL_ADAPTERS, ids=lambda a: a.name)
    def test_state_io_round_trips_a_plain_module(self, adapter) -> None:
        """Every adapter must tolerate a module it did not wrap: the director
        reads generator AND discriminator through ``state_io`` unconditionally."""
        import torch
        import torch.nn as nn

        src = nn.Linear(2, 2)
        opt = torch.optim.Adam(src.parameters())
        src(torch.ones(1, 2)).sum().backward()
        opt.step()
        state = adapter.state_io.model_state(src)
        opt_state = adapter.state_io.optimizer_state(src, opt)

        dst = nn.Linear(2, 2)
        dst_opt = torch.optim.Adam(dst.parameters())
        adapter.state_io.load_model_state(dst, state)
        adapter.state_io.load_optimizer_state(dst, dst_opt, opt_state)
        assert torch.equal(dst.weight, src.weight)
        assert dst_opt.state_dict()["state"].keys() == opt_state["state"].keys()


class TestRequiresAllRanks:
    """The flag that decides who *participates*, distinct from who *writes*."""

    def test_default_is_rank0_only(self) -> None:
        assert DefaultCheckpointAdapter().requires_all_ranks is False

    @pytest.mark.parametrize(
        "adapter", [FSDPCheckpointAdapter(), DeepSpeedCheckpointAdapter()]
    )
    def test_sharded_strategies_need_every_rank(self, adapter) -> None:
        """This is THE assertion. If either flips to False, a multi-rank run
        hangs at the first checkpoint with no error -- SLURM kills it at
        walltime and the log's last line is a normal training iteration."""
        assert adapter.requires_all_ranks is True

    def test_only_deepspeed_writes_its_own_artifact(self) -> None:
        assert DeepSpeedCheckpointAdapter().writes_native_artifact is True
        assert FSDPCheckpointAdapter().writes_native_artifact is False
        assert DefaultCheckpointAdapter().writes_native_artifact is False


class TestResolveFromRuntime:
    def test_no_runtime_is_the_default_adapter(self) -> None:
        assert isinstance(resolve_checkpoint_adapter(None), DefaultCheckpointAdapter)

    def test_runtime_without_an_adapter_defaults(self) -> None:
        """The scripting API builds a TrainingEnvironment by hand."""
        runtime = ParallelRuntime(strategy="none")
        assert isinstance(resolve_checkpoint_adapter(runtime), DefaultCheckpointAdapter)

    def test_the_runtime_carries_the_plugin_s_adapter(self) -> None:
        runtime = ParallelRuntime(
            strategy="fsdp", checkpoint_adapter=FSDPCheckpointAdapter()
        )
        assert isinstance(resolve_checkpoint_adapter(runtime), FSDPCheckpointAdapter)

    def test_checkpoints_require_all_ranks_tracks_the_adapter(self) -> None:
        assert ParallelRuntime(strategy="none").checkpoints_require_all_ranks is False
        assert (
            ParallelRuntime(
                strategy="deepspeed", checkpoint_adapter=DeepSpeedCheckpointAdapter()
            ).checkpoints_require_all_ranks
            is True
        )


class TestEveryStrategySuppliesOne:
    """A plugin returning ``None`` would read as "rank 0 only" -- the hang."""

    @pytest.mark.parametrize("name", list_parallel_strategies())
    def test_plugin_builds_an_adapter(self, name: str) -> None:
        from types import SimpleNamespace

        plugin = resolve_parallel_strategy(name)
        ctx = SimpleNamespace(
            config=None,
            device="cpu",
            parallel=SimpleNamespace(deepspeed=SimpleNamespace()),
        )
        adapter = plugin.checkpoint_adapter(ctx)
        assert adapter is not None
        assert isinstance(adapter, IParallelCheckpointAdapter)

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("none", False),
            ("dp", False),
            ("ddp", False),
            ("fsdp", True),
            ("deepspeed", True),
        ],
    )
    def test_collective_strategies_are_exactly_the_sharding_ones(
        self, name: str, expected: bool
    ) -> None:
        """DP and DDP REPLICATE rather than shard, so rank 0's state_dict is
        the whole model and gating on rank 0 is correct for them."""
        from types import SimpleNamespace

        plugin = resolve_parallel_strategy(name)
        ctx = SimpleNamespace(
            config=None,
            device="cpu",
            parallel=SimpleNamespace(deepspeed=SimpleNamespace()),
        )
        assert plugin.checkpoint_adapter(ctx).requires_all_ranks is expected


class TestDeepSpeedNativeArtifacts:
    def test_writes_a_tag_directory_per_engine(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        adapter = DeepSpeedCheckpointAdapter()
        adapter.save_native(
            models={"generator": engine},
            checkpoint_dir=tmp_path,
            tag="best",
            client_state={"epoch": 1},
        )
        assert engine.saved == [
            (str(tmp_path / DEEPSPEED_TAG_PREFIX / "generator"), "best")
        ]

    def test_tag_roots_are_where_save_native_writes(self, tmp_path: Path) -> None:
        """Retention prunes under these roots, so they must be the writer's own (#2800)."""
        engines = {"generator": _FakeEngine(), "discriminator": _FakeEngine()}
        models = {**engines, "plain": object()}
        adapter = DeepSpeedCheckpointAdapter()
        adapter.save_native(
            models=models, checkpoint_dir=tmp_path, tag="epoch_0001_step_001000", client_state={}
        )
        roots = adapter.native_tag_roots(models=models, checkpoint_dir=tmp_path)
        written = {Path(path) for engine in engines.values() for path, _tag in engine.saved}
        assert len(roots) == 2
        assert set(roots) == written

    def test_no_engine_means_nothing_to_do(self, tmp_path: Path) -> None:
        """A DeepSpeed runtime whose models were never wrapped (audit/probe)."""
        import torch.nn as nn

        assert (
            DeepSpeedCheckpointAdapter().save_native(
                models={"generator": nn.Linear(2, 2)},
                checkpoint_dir=tmp_path,
                tag="best",
                client_state={},
            )
            is None
        )

    def test_load_returns_the_client_state(self, tmp_path: Path) -> None:
        engine = _FakeEngine()
        state, handled = DeepSpeedCheckpointAdapter().load_native(
            models={"generator": engine}, checkpoint_dir=tmp_path, tag="best"
        )
        assert state == {"epoch": 3, "global_step": 300}
        assert handled == {"generator"}
        assert engine.loaded

    def test_a_plain_module_beside_an_engine_is_not_reported_handled(
        self, tmp_path: Path
    ) -> None:
        """The mixed case: the director must still restore the discriminator."""
        import torch.nn as nn

        result = DeepSpeedCheckpointAdapter().load_native(
            models={"generator": _FakeEngine(), "discriminator": nn.Linear(2, 2)},
            checkpoint_dir=tmp_path,
            tag="best",
        )
        assert result.handled == frozenset({"generator"})

    def test_two_engines_are_both_reported_handled(self, tmp_path: Path) -> None:
        result = DeepSpeedCheckpointAdapter().load_native(
            models={"generator": _FakeEngine(), "discriminator": _FakeEngine()},
            checkpoint_dir=tmp_path,
            tag="best",
        )
        assert result.handled == frozenset({"generator", "discriminator"})

    def test_engines_are_duck_typed_not_isinstance_checked(self) -> None:
        """So this module never imports deepspeed and stays testable without it."""
        import spectramr.infrastructure.distributed.checkpoint_adapters as mod

        assert "import deepspeed" not in Path(mod.__file__).read_text()


class TestDefaultAdapterIsInert:
    def test_save_and_load_are_no_ops(self, tmp_path: Path) -> None:
        adapter = DefaultCheckpointAdapter()
        assert (
            adapter.save_native(
                models={}, checkpoint_dir=tmp_path, tag="t", client_state={}
            )
            is None
        )
        assert adapter.load_native(
            models={}, checkpoint_dir=tmp_path, tag="t"
        ) == NativeLoadResult(None, frozenset())

    @pytest.mark.parametrize(
        "adapter", (DefaultCheckpointAdapter(), FSDPCheckpointAdapter()), ids=lambda a: a.name
    )
    def test_non_native_adapters_have_no_tag_roots(self, adapter, tmp_path: Path) -> None:
        """Retention has nothing to prune for them beyond the ``.pt`` files."""
        roots = adapter.native_tag_roots(models={"generator": _FakeEngine()}, checkpoint_dir=tmp_path)
        assert roots == ()

    @pytest.mark.parametrize(
        "adapter", (DefaultCheckpointAdapter(), FSDPCheckpointAdapter()), ids=lambda a: a.name
    )
    def test_non_native_adapters_handle_no_model(self, adapter, tmp_path: Path) -> None:
        """An empty handled set keeps every key on the generic restore path."""
        result = adapter.load_native(
            models={"generator": _FakeEngine()}, checkpoint_dir=tmp_path, tag="t"
        )
        assert result.handled == frozenset()


class _MissingTagEngine(_FakeEngine):
    """The real engine reports a missing tag by RETURNING ``(None, None)``.

    It does not raise, which is what made the miss swallowable.
    """

    def load_checkpoint(self, path, tag, **_kwargs):
        self.loaded.append((str(path), tag))
        return None, None


class TestLoadNativeDoesNotSwallowAMiss:
    """A restore that loaded nothing must not look like one that loaded everything.

    ``load_from`` now treats a successful native load as authoritative and stops
    parsing the file beside the tag directory. If a miss came back as an empty
    dict, the run would log "restored best weights" while still holding the
    weights it started with -- the silent fallback non-negotiable 3 forbids.
    """

    def test_a_missing_tag_raises(self, tmp_path) -> None:
        adapter = DeepSpeedCheckpointAdapter()
        with pytest.raises(FileNotFoundError, match="no checkpoint for tag"):
            adapter.load_native(
                models={"generator": _MissingTagEngine()},
                checkpoint_dir=tmp_path,
                tag="best",
            )

    def test_the_error_names_the_tag_and_the_directory(self, tmp_path) -> None:
        """A cluster operator reads this message, not the traceback."""
        adapter = DeepSpeedCheckpointAdapter()
        with pytest.raises(FileNotFoundError) as excinfo:
            adapter.load_native(
                models={"generator": _MissingTagEngine()},
                checkpoint_dir=tmp_path,
                tag="best",
            )
        message = str(excinfo.value)
        assert "'best'" in message
        assert str(tmp_path / DEEPSPEED_TAG_PREFIX / "generator") in message

    def test_no_engines_still_returns_none_rather_than_raising(self, tmp_path) -> None:
        """Distinct from a miss: there was nothing to load INTO.

        ``load_from`` discriminates on this to decide whether it can fall back
        to the generic payload, so it must stay a return value.
        """
        import torch.nn as nn

        assert DeepSpeedCheckpointAdapter().load_native(
            models={"generator": nn.Linear(2, 2)},
            checkpoint_dir=tmp_path,
            tag="best",
        ) == NativeLoadResult(None, frozenset())

    def test_a_hit_with_no_client_state_is_an_empty_dict_not_none(
        self, tmp_path
    ) -> None:
        """``{}`` means loaded-but-no-metadata and must not read as failure."""

        class _NoClientState(_FakeEngine):
            def load_checkpoint(self, path, tag, **_kwargs):
                return str(path), {}

        state, handled = DeepSpeedCheckpointAdapter().load_native(
            models={"generator": _NoClientState()},
            checkpoint_dir=tmp_path,
            tag="best",
        )
        assert state == {}
        assert handled == {"generator"}


class _Zero3Engine:
    def __init__(self, rank0: bool) -> None:
        import torch.nn as nn

        self.module = nn.Linear(2, 2)
        self._rank0 = rank0
        self.calls = 0

    def save_checkpoint(self, *a, **k):  # marks it as an engine
        raise AssertionError("not called here")

    def zero_optimization_partition_weights(self) -> bool:
        return True

    def _zero3_consolidated_16bit_state_dict(self):
        self.calls += 1
        return self.module.state_dict() if self._rank0 else None


def test_full_model_state_gathers_at_zero3_on_every_rank() -> None:
    adapter = DeepSpeedCheckpointAdapter()
    rank0, other = _Zero3Engine(True), _Zero3Engine(False)
    assert set(adapter.full_model_state(rank0)) == {"weight", "bias"}
    assert adapter.full_model_state(other) == {}
    assert rank0.calls == other.calls == 1


def test_full_model_state_below_zero3_reads_the_unwrapped_module() -> None:
    engine = _Zero3Engine(True)
    engine.zero_optimization_partition_weights = lambda: False
    assert set(DeepSpeedCheckpointAdapter().full_model_state(engine)) == {"weight", "bias"}
    assert engine.calls == 0


def test_full_model_state_of_a_plain_module_is_its_state_io_read() -> None:
    import torch.nn as nn

    for adapter in ALL_ADAPTERS:
        assert set(adapter.full_model_state(nn.Linear(2, 2))) == {"weight", "bias"}


def test_engine_owned_keys_name_only_the_deepspeed_engines() -> None:
    import torch.nn as nn

    models = {"generator": _Zero3Engine(True), "discriminator": nn.Linear(2, 1)}
    assert DeepSpeedCheckpointAdapter().engine_owned_keys(models) == frozenset({"generator"})
    assert DefaultCheckpointAdapter().engine_owned_keys(models) == frozenset()
    assert FSDPCheckpointAdapter().engine_owned_keys(models) == frozenset()


def test_load_native_passes_the_gates_and_module_only(tmp_path: Path) -> None:
    engine = _FakeEngine()
    DeepSpeedCheckpointAdapter().load_native(
        models={"generator": engine}, checkpoint_dir=tmp_path, tag="best",
        optimizer=False, scheduler=False, module_only=True,
    )
    assert engine.load_kwargs == {
        "load_optimizer_states": False,
        "load_lr_scheduler_states": False,
        "load_module_only": True,
    }


def test_save_native_writes_only_the_tag(tmp_path: Path) -> None:
    engine = _FakeEngine()
    path = DeepSpeedCheckpointAdapter().save_native(
        models={"generator": engine}, checkpoint_dir=tmp_path, tag="best", client_state={}
    )
    assert path is None
    assert engine.saved == [(str(tmp_path / DEEPSPEED_TAG_PREFIX / "generator"), "best")]
    assert sorted(p.name for p in tmp_path.iterdir()) == [DEEPSPEED_TAG_PREFIX]


def test_the_adapter_has_no_consolidation_left() -> None:
    adapter = DeepSpeedCheckpointAdapter()
    assert not hasattr(adapter, "save_consolidated_best")
    assert not hasattr(adapter, "_save_consolidated")


class _StagedEngine(_FakeEngine):
    def __init__(self, stage: int) -> None:
        super().__init__()
        self._stage = stage

    def zero_optimization_stage(self) -> int:
        return self._stage


@pytest.mark.parametrize(("stage", "whole"), [(0, True), (2, True), (3, False)])
def test_ema_is_whole_below_zero3_only(stage: int, whole: bool, caplog) -> None:
    adapter = DeepSpeedCheckpointAdapter()
    assert adapter.ema_is_whole({"generator": _StagedEngine(stage), "plain": object()}) is whole
    assert "sidecar" not in caplog.text


def test_an_engine_hiding_its_stage_is_not_assumed_whole(caplog) -> None:
    assert DeepSpeedCheckpointAdapter().ema_is_whole({"generator": _FakeEngine()}) is False
    assert "partitioned, not whole-parameter" in caplog.text


@pytest.mark.parametrize("adapter", [DefaultCheckpointAdapter(), FSDPCheckpointAdapter()])
def test_non_native_adapters_always_hold_a_whole_shadow(adapter) -> None:
    assert adapter.ema_is_whole({"generator": object()}) is True
