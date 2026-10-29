"""The five strategy plugins, and the ordering constraints they encode.

Wrap ordering is not a style choice here: FSDP in Stage B or DeepSpeed in
Stage A would build, run, and produce wrong numbers rather than raising.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.distributed.checkpoint_adapters import (
    DeepSpeedCheckpointAdapter,
    DefaultCheckpointAdapter,
    FSDPCheckpointAdapter,
)
from spectramr.infrastructure.distributed.strategies import (
    DataParallelStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    FSDPStrategy,
    NoParallelStrategy,
)


def _ctx(*, world_size: int = 1):
    return SimpleNamespace(
        config=None,
        device="cpu",
        world_size=world_size,
        parallel=SimpleNamespace(
            deepspeed=SimpleNamespace(),
            sync_batch_norm=False,
        ),
    )


class TestCheckpointAdapterPerStrategy:
    @pytest.mark.parametrize(
        ("plugin", "expected"),
        [
            (NoParallelStrategy(), DefaultCheckpointAdapter),
            (DataParallelStrategy(), DefaultCheckpointAdapter),
            (DDPStrategy(), DefaultCheckpointAdapter),
            (FSDPStrategy(), FSDPCheckpointAdapter),
            (DeepSpeedStrategy(), DeepSpeedCheckpointAdapter),
        ],
        ids=lambda v: getattr(v, "name", getattr(v, "__name__", str(v))),
    )
    def test_each_supplies_the_right_adapter(self, plugin, expected) -> None:
        assert isinstance(plugin.checkpoint_adapter(_ctx()), expected)

    def test_replicating_strategies_do_not_need_all_ranks(self) -> None:
        """DP and DDP replicate the model, so rank 0's state_dict IS the model
        and gating the write on rank 0 is correct for them."""
        for plugin in (DataParallelStrategy(), DDPStrategy(), NoParallelStrategy()):
            assert plugin.checkpoint_adapter(_ctx()).requires_all_ranks is False

    def test_sharding_strategies_do(self) -> None:
        for plugin in (FSDPStrategy(), DeepSpeedStrategy()):
            assert plugin.checkpoint_adapter(_ctx()).requires_all_ranks is True


class TestTheRetiredKnobStaysOut:
    """The knob is retired; it must not come back as a DeepSpeed key either."""

    def test_it_is_not_a_deepspeed_config_key(self) -> None:
        """If it came back as one, the rendered ds_config would carry a key nothing reads."""
        from spectramr.infrastructure.distributed.deepspeed_backend.config_builder import (
            DERIVED_KEYS,
        )

        assert "save_consolidated_best" not in DERIVED_KEYS

    def test_the_adapter_takes_no_knob(self) -> None:
        adapter = DeepSpeedStrategy().checkpoint_adapter(_ctx())
        assert isinstance(adapter, DeepSpeedCheckpointAdapter)
        assert not hasattr(adapter, "save_consolidated_best")


class TestWrapOrderingIsEncoded:
    def test_fsdp_wraps_in_stage_a(self) -> None:
        """FSDP.__init__ re-points parameter storage; an optimizer built first
        gets shard-shaped gradients against full-shape moment buffers."""
        import inspect

        source = inspect.getsource(FSDPStrategy.prepare_models)
        assert "maybe_wrap_with_fsdp" in source

    def test_deepspeed_does_nothing_in_stage_a(self) -> None:
        """initialize() CONSUMES an already-built optimizer, so it cannot run
        before OptimizationBuilder."""
        models = {"generator": object()}
        assert DeepSpeedStrategy().prepare_models(models, _ctx()) is models

    def test_ddp_does_not_wrap_in_stage_a(self) -> None:
        """DDP could wrap early (it preserves parameter identity) but then
        count_parameters and the optimizer builder would see the wrapper."""
        import inspect

        source = inspect.getsource(DDPStrategy.prepare_models)
        assert "DistributedDataParallel" not in source


# --------------------------------------------------------------------------
# `requires_process_group` — the flag `cli.profile_preflight` reads to refuse a
# single-process profiling run before it costs twelve minutes.
# --------------------------------------------------------------------------


def test_requires_process_group_values():
    """`dp` is the one that matters: DataParallel needs NO process group.

    A caller spelling this rule as `strategy != "none"` would block it, which is
    why the flag exists instead of that expression.
    """
    from spectramr.infrastructure.distributed.strategies import (
        DataParallelStrategy,
        DDPStrategy,
        DeepSpeedStrategy,
        FSDPStrategy,
        NoParallelStrategy,
    )

    assert NoParallelStrategy.requires_process_group is False
    assert DataParallelStrategy.requires_process_group is False
    assert DDPStrategy.requires_process_group is True
    assert FSDPStrategy.requires_process_group is True
    assert DeepSpeedStrategy.requires_process_group is True


def test_the_flag_agrees_with_the_require_process_group_call_sites():
    """The drift lock (non-negotiable 17): the flag is not a second declaration.

    Read off the AST rather than asserted as a literal list, so a backend added
    tomorrow that calls `_require_process_group` without setting the flag — or
    sets it without calling — fails here instead of silently profiling into a
    twelve-minute RuntimeError.
    """
    src = Path("src/spectramr/infrastructure/distributed/strategies.py")
    tree = ast.parse(src.read_text(encoding="utf-8"))

    callers, flagged = set(), set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for call in ast.walk(node):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id == "_require_process_group"
            ):
                callers.add(node.name)
        for stmt in node.body:
            if (
                isinstance(stmt, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "requires_process_group"
                    for t in stmt.targets
                )
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is True
            ):
                flagged.add(node.name)

    assert callers, "AST found no _require_process_group call sites — test is blind"
    assert callers == flagged, (
        f"drift: classes calling _require_process_group={sorted(callers)} but "
        f"classes setting requires_process_group=True={sorted(flagged)}"
    )


class TestTimeoutSeconds:
    """``parallel.timeout_seconds`` is read by setup_distributed and stamped here."""

    def test_default_is_null_so_torchs_backend_default_applies(self):
        from spectramr.config.schemas.base import ParallelismConfigSchema

        assert ParallelismConfigSchema().timeout_seconds is None

    def test_non_positive_timeout_is_refused(self):
        import pytest
        from pydantic import ValidationError

        from spectramr.config.schemas.base import ParallelismConfigSchema

        with pytest.raises(ValidationError):
            ParallelismConfigSchema(timeout_seconds=0)

    def test_every_strategy_stamps_the_timeout_into_provenance(self):
        from types import SimpleNamespace

        from spectramr.config.schemas.base import ParallelismConfigSchema
        from spectramr.infrastructure.distributed.strategies import NoParallelStrategy

        ctx = SimpleNamespace(
            parallel=ParallelismConfigSchema(timeout_seconds=120), world_size=2
        )
        assert NoParallelStrategy().provenance(ctx)["timeout_seconds"] == 120


class TestSpatialProbeShape:
    """The gate's synthetic volume: declared when given, else the training patch."""

    @staticmethod
    def _ctx(patch, probe=None, in_channels=2):
        from types import SimpleNamespace

        from spectramr.infrastructure.distributed.strategy_registry import ParallelContext

        config = SimpleNamespace(
            data=SimpleNamespace(sampling=SimpleNamespace(patch_size=patch)),
            model=SimpleNamespace(in_channels=in_channels),
        )
        parallel = SimpleNamespace(spatial_probe_shape=probe)
        return ParallelContext(config=config, device=None, parallel=parallel, mesh=None)

    def test_the_training_patch_is_the_default(self) -> None:
        from spectramr.infrastructure.distributed.strategies import _spatial_probe_shape

        assert _spatial_probe_shape(self._ctx((16, 8, 8))) == (1, 2, 16, 8, 8)

    def test_a_declared_probe_wins(self) -> None:
        from spectramr.infrastructure.distributed.strategies import _spatial_probe_shape

        assert _spatial_probe_shape(self._ctx((16, 8, 8), probe=[32, 8, 8])) == (1, 2, 32, 8, 8)

    def test_a_2d_patch_needs_a_declared_probe(self) -> None:
        from spectramr.infrastructure.distributed.strategies import _spatial_probe_shape

        with pytest.raises(ValueError, match="spatial_probe_shape"):
            _spatial_probe_shape(self._ctx((64, 64)))


class TestMultiEngineAdmission:
    """The #2631 gate: every refusal raises from config alone, before initialize."""

    @staticmethod
    def _ds(**overrides):
        base = {"allow_multi_engine": True, "zero_stage": 2}
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_knob_off_refuses_and_names_the_gpu_test(self) -> None:
        from spectramr.infrastructure.distributed.strategies import (
            _require_multi_engine_admissible,
        )

        with pytest.raises(RuntimeError, match="test_deepspeed_multi_engine_gpu"):
            _require_multi_engine_admissible(
                self._ds(allow_multi_engine=False), {"opt_g": 1, "opt_d": 2}
            )

    @pytest.mark.parametrize("stage", [0, 1, 2])
    def test_knob_on_admits_zero_0_to_2_pair(self, stage: int) -> None:
        from spectramr.infrastructure.distributed.strategies import (
            _require_multi_engine_admissible,
        )

        _require_multi_engine_admissible(
            self._ds(zero_stage=stage), {"opt_g": 1, "opt_d": 2}
        )

    def test_planted_zero3_is_refused_even_with_the_knob(self) -> None:
        from spectramr.infrastructure.distributed.strategies import (
            _require_multi_engine_admissible,
        )

        with pytest.raises(RuntimeError, match="zero_stage=3"):
            _require_multi_engine_admissible(
                self._ds(zero_stage=3), {"opt_g": 1, "opt_d": 2}
            )

    def test_planted_third_optimizer_is_refused(self) -> None:
        from spectramr.infrastructure.distributed.strategies import (
            _require_multi_engine_admissible,
        )

        with pytest.raises(RuntimeError, match="opt_x"):
            _require_multi_engine_admissible(
                self._ds(), {"opt_g": 1, "opt_d": 2, "opt_x": 3}
            )
