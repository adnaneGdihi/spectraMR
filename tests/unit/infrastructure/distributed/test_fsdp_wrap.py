"""Tests for the FSDP wrap helper.

Distributed init needs a multi-process world, so these exercise the
single-process case -- which is all the test runner has, and where the
behaviour used to be wrong.

The old docstring here read: "The single-process behaviour is critical: anyone
who forgets to launch with torchrun must still get a working model." That
reasoning is what produced the bug. A "working model" in that situation is an
UNSHARDED one, so the run completes, reports success, and stamps ``fsdp`` into
its own provenance while never having sharded anything -- and DDP raised on the
identical mistake, so the two strategies disagreed about whether it was fatal.
Forgetting ``torchrun`` is now an error for both.
"""

from __future__ import annotations

import pytest
import torch.nn as nn
from pydantic import ValidationError

from spectramr.config.schemas.base import FSDPConfigSchema, ParallelismConfigSchema
from spectramr.infrastructure.distributed.fsdp_wrap import maybe_wrap_with_fsdp


class _Bag:
    """Tiny attribute bag standing in for ParallelismConfigSchema."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_fsdp_disabled_returns_input_unchanged() -> None:
    parallel = ParallelismConfigSchema(fsdp=FSDPConfigSchema(enabled=False))
    model = nn.Linear(4, 4)
    wrapped = maybe_wrap_with_fsdp(model, parallel)
    assert wrapped is model


def test_fsdp_enabled_without_distributed_init_raises() -> None:
    """Enabling FSDP with no process group must RAISE.

    This test previously asserted the opposite -- that it returns the bare model
    with a logged warning, "never raise". That was the bug, not the contract: a
    user who forgot ``torchrun`` got a full-length UNSHARDED run that completed,
    reported success, and stamped ``fsdp`` into its own provenance. DDP has
    always raised on the identical mistake, so the two strategies disagreed about
    whether the same misconfiguration was fatal.

    Note the config now needs BOTH halves -- ``strategy='fsdp'`` and
    ``fsdp.enabled=True`` -- because they were independent switches and could
    contradict each other (#620 F2).
    """
    parallel = ParallelismConfigSchema(
        strategy="fsdp", fsdp=FSDPConfigSchema(enabled=True)
    )
    model = nn.Linear(4, 4)
    with pytest.raises(RuntimeError, match="not initialised"):
        maybe_wrap_with_fsdp(model, parallel)


def test_fsdp_requires_both_halves_of_the_declaration() -> None:
    """``fsdp.enabled`` alone used to silently shard while the config read
    'no parallelism'."""
    with pytest.raises(ValidationError):
        ParallelismConfigSchema(fsdp=FSDPConfigSchema(enabled=True))


def test_fsdp_no_parallel_block_passthrough() -> None:
    """Calling with a config that doesn't have an .fsdp attribute is a no-op."""
    parallel = _Bag()  # no .fsdp at all
    model = nn.Linear(4, 4)
    wrapped = maybe_wrap_with_fsdp(model, parallel)
    assert wrapped is model


class TestFSDPPrecisionFollowsTheGlobalPrecision:
    """``optimization.precision`` owns the arm's precision; FSDP only overrides it."""

    @staticmethod
    def _resolve(local, enabled=False, dtype=None):
        from types import SimpleNamespace

        from spectramr.infrastructure.distributed.fsdp_wrap import resolve_fsdp_precision

        return resolve_fsdp_precision(
            SimpleNamespace(mixed_precision=local), SimpleNamespace(enabled=enabled, dtype=dtype)
        )

    def test_the_default_is_single_precision(self) -> None:
        assert self._resolve(None) is None

    def test_no_precision_block_is_single_precision(self) -> None:
        from types import SimpleNamespace

        from spectramr.infrastructure.distributed.fsdp_wrap import resolve_fsdp_precision

        assert resolve_fsdp_precision(SimpleNamespace(mixed_precision=None), None) is None

    def test_float32_wins_even_when_enabled(self) -> None:
        assert self._resolve(None, enabled=True, dtype="float32") is None

    def test_bfloat16_must_be_declared_and_is_then_followed(self) -> None:
        assert self._resolve(None, enabled=True, dtype="bfloat16") == "bf16"

    def test_mixed_precision_without_a_type_is_float16(self) -> None:
        assert self._resolve(None, enabled=True) == "fp16"

    def test_the_fsdp_override_wins_over_the_global_block(self) -> None:
        assert self._resolve("bf16", enabled=False) == "bf16"
        assert self._resolve("bf16", enabled=True, dtype="float32") == "bf16"

    def test_the_schema_default_defers_to_the_global_block(self) -> None:
        assert FSDPConfigSchema(enabled=True).mixed_precision is None


class TestShardUnitSelection:
    """Which submodules become their own ``fully_shard`` unit, deepest first."""

    @staticmethod
    def _units(model, **cfg):
        from types import SimpleNamespace

        from spectramr.infrastructure.distributed.fsdp_wrap import _select_units

        base = {"auto_wrap_policy": "size_based", "min_num_params": 1, "transformer_layer_cls": []}
        base.update(cfg)
        return _select_units(model, SimpleNamespace(**base))

    @staticmethod
    def _net():
        return nn.Sequential(
            nn.Sequential(nn.Linear(4, 64), nn.Linear(64, 4)), nn.Linear(4, 4)
        )

    def test_none_shards_only_the_root(self) -> None:
        assert self._units(self._net(), auto_wrap_policy="none") == []

    def test_size_based_claims_parameters_once_from_the_deepest_module(self) -> None:
        units = self._units(self._net(), min_num_params=200)
        # 0.0 (320 params) and 0.1 (260) qualify; their parent then has nothing left.
        assert units == ["0.0", "0.1"]

    def test_size_based_parent_takes_what_its_children_did_not(self) -> None:
        assert self._units(self._net(), min_num_params=500) == ["0"]

    def test_transformer_block_selects_by_class_name(self) -> None:
        units = self._units(
            self._net(), auto_wrap_policy="transformer_block", transformer_layer_cls=["Linear"]
        )
        assert set(units) == {"0.0", "0.1", "1"}

    def test_transformer_block_without_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            self._units(self._net(), auto_wrap_policy="transformer_block")

    def test_transformer_block_matching_nothing_raises(self) -> None:
        with pytest.raises(ValueError, match="matched no submodule"):
            self._units(
                self._net(), auto_wrap_policy="transformer_block", transformer_layer_cls=["Conv2d"]
            )

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown FSDP auto_wrap_policy"):
            self._units(self._net(), auto_wrap_policy="layer_count")


def test_shard_like_refuses_a_reference_without_a_shard_plan() -> None:
    from spectramr.infrastructure.distributed.fsdp_wrap import shard_like

    with pytest.raises(RuntimeError, match="no shard plan"):
        shard_like(nn.Linear(2, 2), nn.Linear(2, 2))


def test_unknown_sharding_strategy_raises() -> None:
    from spectramr.infrastructure.distributed.fsdp_wrap import fsdp_shard_mesh

    # The strategy is checked before the mesh is indexed, so no mesh is needed.
    with pytest.raises(ValueError, match="Unknown FSDP sharding_strategy"):
        fsdp_shard_mesh(object(), "zero3")
