"""The compile-placement table.

One placement served every strategy and was wrong for most of them. Each row
below is a measurement, so each test names what was measured rather than
restating the table.
"""

from __future__ import annotations

import pytest

from spectramr.infrastructure.training.builders.compile_placement import (
    CompilePlacement,
    resolve_compile_placement,
)


class TestTheTable:
    def test_single_device_compiles_last(self):
        """After every wrap, which for this path also keeps the optimizer --
        built before Stage B -- looking at a bare module (#2174)."""
        assert resolve_compile_placement("none").stage == "after_stage_b"

    def test_an_absent_parallel_block_is_the_single_device_path(self):
        """1443 of 1528 arms have no ``parallel:`` block at all."""
        assert resolve_compile_placement(None).stage == "after_stage_b"

    def test_ddp_compiles_after_the_wrap(self):
        """DDPOptimizer applies only when dynamo compiles a DDP-wrapped module,
        so ``DDP(compile(m))`` -- what shipped -- forfeits the allreduce/backward
        overlap entirely."""
        assert resolve_compile_placement("ddp").stage == "after_stage_b"

    def test_fsdp_compiles_after_its_stage_a_wrap(self):
        """``torch.compile(FSDP(m))`` is the recommended order; the reverse is
        what shipped."""
        assert resolve_compile_placement("fsdp").stage == "after_stage_a"

    @pytest.mark.parametrize("zero_stage", [1, 2, 3])
    def test_every_sharded_stage_routes_to_deepcompile(self, zero_stage):
        """`torch.compile` runs on the bare module before `deepspeed.initialize`,
        so the ZeRO collectives are opaque calls it never traces across -- it
        optimises the arithmetic and leaves the communication, which is the part
        a sharded run pays for. DeepCompile has a pass for every one of these
        stages (`z1` covers 1 AND 2, `z3` covers 3), so it is the route.
        """
        placement = resolve_compile_placement("deepspeed", zero_stage=zero_stage)
        assert placement.refused
        assert placement.stage is None
        assert "DeepCompile" in placement.refusal

    @pytest.mark.parametrize("zero_stage", [1, 2, 3])
    def test_each_refusal_names_the_pass_that_replaces_it(self, zero_stage):
        """A refusal that does not say what to write instead gets worked around
        rather than followed."""
        refusal = resolve_compile_placement("deepspeed", zero_stage=zero_stage).refusal
        assert "parallel.deepspeed.compile.enabled" in refusal
        expected_pass = "z3" if zero_stage == 3 else "z1"
        assert expected_pass in refusal

    def test_zero3_is_refused_as_a_crash_not_a_preference(self):
        """Planted: the two refusals are different claims and must not be
        collapsed. At ZeRO-3 `torch.compile` raises; at 1/2 it runs and is
        simply the weaker option. Overstating the second would be a false
        measurement claim, and understating the first hides a crash.
        """
        crash = resolve_compile_placement("deepspeed", zero_stage=3).refusal
        assert "_in_forward" in crash, "the observed error should be quoted"

        for stage in (1, 2):
            preference = resolve_compile_placement("deepspeed", zero_stage=stage).refusal
            assert "_in_forward" not in preference
            assert "does run" in preference, "must not claim a crash it did not observe"

    def test_zero_stage_zero_keeps_torch_compile(self):
        """ZeRO off means no collectives, so DeepCompile has no pass to apply.
        Refusing here would leave the arm with no compilation at all."""
        placement = resolve_compile_placement("deepspeed", zero_stage=0)
        assert not placement.refused
        assert placement.stage == "after_stage_a"
        assert placement.advisory and "ZeRO is off" in placement.advisory

    def test_data_parallel_is_advised_against_but_not_refused(self):
        """No arm uses ``dp`` and it is not formally deprecated upstream, so
        there is nothing to measure against. Stating the expectation is honest;
        refusing on an unmeasured expectation is not."""
        placement = resolve_compile_placement("dp")
        assert not placement.refused
        assert placement.advisory and "replicates" in placement.advisory


class TestTheResidualConflict:
    """Some strategies must wrap before the optimizer exists, so #2174 survives
    for them and the table says so rather than pretending otherwise."""

    @pytest.mark.parametrize("strategy,zero", [("fsdp", None), ("deepspeed", 0)])
    def test_pre_optimizer_placements_declare_it(self, strategy, zero):
        placement = resolve_compile_placement(strategy, zero_stage=zero)
        assert placement.optimizer_sees_wrapper is True

    @pytest.mark.parametrize("strategy", ["none", "ddp", "dp"])
    def test_post_optimizer_placements_are_clear_of_it(self, strategy):
        assert resolve_compile_placement(strategy).optimizer_sees_wrapper is False


class TestUnknownStrategies:
    def test_an_undeclared_strategy_raises(self):
        """The planted violation. A new backend that forgets to declare a
        placement must fail the lookup, not inherit whichever row is first."""
        with pytest.raises(ValueError, match="No compile placement is declared"):
            resolve_compile_placement("tensor_parallel")

    def test_the_error_lists_what_is_known(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_compile_placement("nonexistent")
        for known in ("none", "ddp", "fsdp", "deepspeed"):
            assert known in str(excinfo.value)

    def test_every_registered_strategy_has_a_placement(self):
        """Pins the table against the strategy registry, so adding a plugin
        without a placement fails here rather than at build time."""
        from spectramr.infrastructure.distributed.strategy_registry import (
            list_parallel_strategies,
        )

        for name in list_parallel_strategies():
            placement = resolve_compile_placement(name, zero_stage=2)
            assert isinstance(placement, CompilePlacement)
