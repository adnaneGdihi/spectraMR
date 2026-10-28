"""Parallel-strategy plugins: dispatch, ordering, and no silent fallbacks.

``strategy: 'fsdp'`` used to be accepted by nothing. ``apply_parallelism``
implemented ``none``/``dp``/``ddp`` in an if/elif and raised ``ValueError`` on
anything else -- *after* the whole training environment had been built -- while
FSDP was reached through a separate ``parallel.fsdp.enabled`` flag read in
``ModelBuilder``. The spelling the reference template advertised was the fatal
one, and the one that worked read as "no parallelism".

The ordering assertions matter more than the dispatch ones. FSDP wraps in
Stage A because ``FSDP.__init__`` re-points parameter storage into a flat shard:
an optimizer built first receives shard-shaped gradients against full-shape
moment buffers. DP/DDP wrap in Stage B because they do not change parameter
identity and everything downstream expects a bare module.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from spectramr.config.schemas.base import ParallelismConfigSchema  # noqa: E402
from spectramr.infrastructure.distributed.strategy_registry import (  # noqa: E402
    ParallelContext,
    ParallelStrategyPlugin,
    list_parallel_strategies,
    resolve_parallel_strategy,
)


def _ctx(**kw) -> ParallelContext:
    parallel = ParallelismConfigSchema(**kw)
    return ParallelContext(
        config=object(), device=torch.device("cpu"), parallel=parallel
    )


def _models() -> dict[str, nn.Module]:
    return {"generator": nn.Sequential(nn.Conv2d(2, 2, 3), nn.BatchNorm2d(2))}


class TestDispatch:
    @pytest.mark.parametrize("name", ["none", "dp", "ddp", "fsdp"])
    def test_every_implemented_strategy_resolves(self, name: str) -> None:
        assert resolve_parallel_strategy(name).name == name

    def test_fsdp_is_reachable_at_all(self) -> None:
        """The headline gap: this used to be a ValueError thrown deep inside the
        build, not a strategy."""
        assert "fsdp" in list_parallel_strategies()

    def test_every_plugin_satisfies_the_protocol(self) -> None:
        for name in list_parallel_strategies():
            assert isinstance(resolve_parallel_strategy(name), ParallelStrategyPlugin)

    def test_unknown_strategy_raises_naming_the_registered_set(self) -> None:
        with pytest.raises(ValueError, match="Unknown parallel strategy"):
            resolve_parallel_strategy("horovod")

    def test_deepspeed_is_registered(self) -> None:
        """This asserted the opposite until the DeepSpeed backend landed, which
        is how the gap stayed visible instead of being forgotten."""
        assert resolve_parallel_strategy("deepspeed").name == "deepspeed"

    def test_the_schema_vocabulary_and_the_registry_agree_exactly(self) -> None:
        """No advertised strategy resolves to nothing, and nothing is registered
        that a config cannot ask for. Compared, never assumed -- the two lists
        drifting apart is the whole failure this file exists for."""
        from typing import get_args

        from spectramr.config.schemas.base import ParallelStrategy

        assert set(get_args(ParallelStrategy)) == set(list_parallel_strategies())


class TestNoSilentFallbacks:
    """A misconfiguration must abort, not produce a run that reports success."""

    def test_ddp_without_a_process_group_raises(self) -> None:
        plugin = resolve_parallel_strategy("ddp")
        with pytest.raises(RuntimeError, match="process group"):
            plugin.adopt(_models(), {}, {}, _ctx(strategy="ddp"))

    def test_fsdp_without_a_process_group_raises(self) -> None:
        """This is the behaviour change. ``maybe_wrap_with_fsdp`` used to warn and
        return the UNWRAPPED model, so a user who forgot torchrun got a
        full-length unsharded run whose provenance claimed it was sharded --
        while DDP aborted on the identical mistake."""
        plugin = resolve_parallel_strategy("fsdp")
        with pytest.raises(RuntimeError, match="process group"):
            plugin.prepare_models(
                _models(), _ctx(strategy="fsdp", fsdp={"enabled": True})
            )

    def test_dp_on_a_single_device_raises_rather_than_warning(self) -> None:
        """Was a warning + return, so the arm's provenance said multi-GPU while
        it ran on one."""
        if torch.cuda.device_count() > 1:
            pytest.skip("needs a single-GPU (or CPU) host to exercise the guard")
        plugin = resolve_parallel_strategy("dp")
        with pytest.raises(RuntimeError, match="GPU"):
            plugin.adopt(_models(), {}, {}, _ctx(strategy="dp"))


class TestStageOrdering:
    """Which hook each strategy uses is a correctness constraint, not a style."""

    def test_fsdp_wraps_in_stage_a(self) -> None:
        """Stage A runs BEFORE optimizers exist. FSDP must be there: it flattens
        parameters into a shard and re-points their storage, so an optimizer
        built first would hold parameters whose storage was swapped underneath
        it."""
        import inspect

        plugin = resolve_parallel_strategy("fsdp")
        assert "maybe_wrap_with_fsdp" in inspect.getsource(plugin.prepare_models)

    def test_ddp_wraps_in_stage_b(self) -> None:
        """Stage B runs AFTER optimizers. DDP does not change parameter identity,
        and wrapping earlier would make count_parameters and the optimizer
        builder's model selection see a DistributedDataParallel."""
        import inspect

        plugin = resolve_parallel_strategy("ddp")
        assert "wrap_with_ddp" in inspect.getsource(plugin.adopt)
        assert "wrap_with_ddp" not in inspect.getsource(plugin.prepare_models)

    def test_only_process_group_strategies_substitute_samplers(self) -> None:
        loaders = {"train": object(), "val": object()}
        for name in ("none", "dp"):
            plugin = resolve_parallel_strategy(name)
            assert plugin.prepare_data_loaders(loaders, _ctx()) == loaders


class TestStepPolicyHandoff:
    def test_fsdp_supplies_a_sharding_correct_step_policy(self) -> None:
        """clip_grad_norm_(model.parameters(), n) computes a PER-SHARD norm under
        FSDP, so each rank scales differently -- silently, and it reads as
        training instability. The strategy knows how the model was wrapped, so it
        is what supplies the right policy."""
        import inspect

        from spectramr.infrastructure.training.optimizers import FSDPStepPolicy

        source = inspect.getsource(resolve_parallel_strategy("fsdp").adopt)
        assert FSDPStepPolicy.__name__ in source

    @pytest.mark.parametrize("name", ["none", "dp"])
    def test_unsharded_strategies_leave_the_default_policy(self, name: str) -> None:
        if name == "dp" and torch.cuda.device_count() <= 1:
            pytest.skip("needs >=2 visible GPUs; the <=1 case is asserted below")
        plugin = resolve_parallel_strategy(name)
        result = plugin.adopt(_models(), {}, {}, _ctx(strategy=name))
        assert result.step_policy is None

    def test_dp_refuses_a_single_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``dp`` over one GPU is a no-op that still pays scatter/gather, and it
        used to warn-and-continue — so the arm's provenance claimed multi-GPU
        while it ran on one. It must raise (non-negotiable #3).

        ``device_count`` is patched rather than read, so the contract is checked
        on any box: previously this branch was only ever *skipped*, never
        asserted, on the 1-GPU dev machines and CPU-only cluster nodes where it
        is precisely the reachable one.
        """
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        plugin = resolve_parallel_strategy("dp")

        with pytest.raises(RuntimeError, match=r"strategy='dp' but 1 GPU\(s\)"):
            plugin.adopt(_models(), {}, {}, _ctx(strategy="dp"))


class TestProvenance:
    def test_records_the_strategy_and_world_size(self) -> None:
        record = resolve_parallel_strategy("ddp").provenance(_ctx(strategy="ddp"))
        assert record["strategy"] == "ddp"
        assert record["world_size"] == 1

    def test_fsdp_records_the_sharding_knobs_that_were_actually_used(self) -> None:
        ctx = _ctx(
            strategy="fsdp",
            fsdp={"enabled": True, "sharding_strategy": "shard_grad_op"},
        )
        record = resolve_parallel_strategy("fsdp").provenance(ctx)
        assert record["fsdp"]["sharding_strategy"] == "shard_grad_op"
        assert "transformer_layer_cls" in record["fsdp"]


class TestCheckpointAdapterSeam:
    """``checkpoint_adapter`` is a plugin METHOD, not a result field.

    A field on ``ParallelizationResult`` would have to be set by all five
    ``adopt`` implementations; a new plugin that forgot one would silently get
    ``None``, which reads as "rank 0 only" -- the deadlock. As a method the
    base class supplies the default and there is exactly one call site.
    """

    def test_the_protocol_declares_it(self):
        from spectramr.infrastructure.distributed.strategy_registry import (
            ParallelStrategyPlugin,
        )

        assert hasattr(ParallelStrategyPlugin, "checkpoint_adapter")

    def test_it_is_not_a_parallelization_result_field(self):
        from spectramr.infrastructure.distributed.strategy_registry import (
            ParallelizationResult,
        )

        assert "checkpoint_adapter" not in ParallelizationResult.__dataclass_fields__

    def test_runtime_defaults_to_no_adapter(self):
        from spectramr.infrastructure.distributed.strategy_registry import ParallelRuntime

        assert ParallelRuntime.single_process().checkpoint_adapter is None

    def test_checkpoints_require_all_ranks_is_false_without_an_adapter(self):
        """The scripting API builds a runtime by hand; it must not accidentally
        opt into collective checkpointing."""
        from spectramr.infrastructure.distributed.strategy_registry import ParallelRuntime

        assert ParallelRuntime.single_process().checkpoints_require_all_ranks is False

    def test_the_director_threads_it_onto_the_runtime(self):
        import inspect

        from spectramr.infrastructure.training.builders import director

        assert "checkpoint_adapter=parallel_plugin.checkpoint_adapter(" in (
            inspect.getsource(director)
        )
