"""DeepSpeed's optimizer rules, the audit check over them, and the FSDP/DeepSpeed schema rules.

Each rule has a planted violation (CLAUDE.md #15): the point of a check is the
shape it has been watched to refuse.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from spectramr.config.schemas.base import (
    DeepSpeedConfigSchema,
    FSDPConfigSchema,
    ParallelismConfigSchema,
)
from spectramr.infrastructure.distributed.deepspeed_backend.optimizer_support import (
    CPU_OFFLOAD_OPTIMIZERS,
    ZERO_SUPPORTED,
    qualified_name,
    zero_optimizer_problems,
)
from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker

pytestmark = pytest.mark.unit


class TestZeroOptimizerProblems:
    def test_adamw_is_accepted_under_zero2(self) -> None:
        assert not zero_optimizer_problems(
            torch.optim.AdamW, zero_stage=2, offload_optimizer="none"
        )

    def test_planted_sgd_is_refused_under_zero1(self) -> None:
        problems = zero_optimizer_problems(torch.optim.SGD, zero_stage=1, offload_optimizer="none")
        assert problems and "not a ZeRO-supported optimizer" in problems[0]

    def test_any_optimizer_is_fine_at_stage_zero_without_offload(self) -> None:
        assert not zero_optimizer_problems(torch.optim.SGD, zero_stage=0, offload_optimizer="none")

    @pytest.mark.parametrize("offload", ["cpu", "nvme"])
    def test_planted_torch_adamw_with_offload_is_refused(self, offload) -> None:
        problems = zero_optimizer_problems(
            torch.optim.AdamW, zero_stage=2, offload_optimizer=offload
        )
        assert any("DeepSpeedCPUAdam" in p for p in problems)


class TestSetsMatchTheInstalledDeepSpeed:
    """The sets are copied from DeepSpeed, so they must be checked against it."""

    def test_supported_set_has_not_drifted(self) -> None:
        zero_utils = pytest.importorskip("deepspeed.runtime.zero.utils")
        installed = {qualified_name(c) for c in zero_utils.ZERO_SUPPORTED_OPTIMIZERS}
        # apex's FusedAdam is optional and absent on most hosts.
        installed -= {n for n in installed if n.startswith("apex.")}
        assert installed <= ZERO_SUPPORTED

    def test_offload_optimizers_are_the_two_deepspeed_checks(self) -> None:
        adam = pytest.importorskip("deepspeed.ops.adam")
        lion = pytest.importorskip("deepspeed.ops.lion")
        expected = {
            qualified_name(adam.DeepSpeedCPUAdam),
            qualified_name(lion.DeepSpeedCPULion),
        }
        assert expected == CPU_OFFLOAD_OPTIMIZERS


def _checker() -> ConfigHealthChecker:
    return ConfigHealthChecker.__new__(ConfigHealthChecker)


def _config(optimizer: str, *, strategy="deepspeed", zero_stage=2, offload="none"):
    deepspeed = {"enabled": True, "zero_stage": zero_stage}
    if offload != "none":
        deepspeed["offload_optimizer"] = offload
    return SimpleNamespace(
        parallel=ParallelismConfigSchema(
            strategy=strategy, **({"deepspeed": deepspeed} if strategy == "deepspeed" else {})
        ),
        optimization=SimpleNamespace(optimizer=SimpleNamespace(type=optimizer)),
    )


class TestHealthCheck:
    def test_not_deepspeed_is_not_applicable(self) -> None:
        result = _checker().check_deepspeed_optimizer_supported(_config("sgd", strategy="ddp"))
        assert result.passed and result.severity == "info"

    def test_adamw_under_zero2_passes(self) -> None:
        assert _checker().check_deepspeed_optimizer_supported(_config("adamw")).passed

    def test_planted_sgd_under_zero1_is_an_error(self) -> None:
        result = _checker().check_deepspeed_optimizer_supported(_config("sgd", zero_stage=1))
        assert not result.passed and result.severity == "error"

    def test_planted_adamw_with_cpu_offload_is_an_error(self) -> None:
        result = _checker().check_deepspeed_optimizer_supported(
            _config("adamw", zero_stage=2, offload="cpu")
        )
        assert not result.passed and "DeepSpeedCPUAdam" in result.message

    def test_unresolvable_optimizer_is_left_to_the_registry_check(self) -> None:
        result = _checker().check_deepspeed_optimizer_supported(_config("no_such_optimizer"))
        assert result.passed and result.severity == "info"


class TestSchemaRules:
    def test_planted_fsdp_fp16_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="ShardedGradScaler"):
            FSDPConfigSchema(enabled=True, mixed_precision="fp16")

    def test_fsdp_bf16_and_null_are_accepted(self) -> None:
        FSDPConfigSchema(enabled=True, mixed_precision="bf16")
        FSDPConfigSchema(enabled=True, mixed_precision=None)

    def test_disabled_fsdp_block_does_not_trip_the_fp16_rule(self) -> None:
        FSDPConfigSchema(enabled=False, mixed_precision="fp16")

    def test_reduce_and_buffer_dtype_default_to_float32(self) -> None:
        cfg = FSDPConfigSchema(enabled=True)
        assert (cfg.reduce_dtype, cfg.buffer_dtype) == ("float32", "float32")

    def test_unknown_reduce_dtype_is_refused(self) -> None:
        with pytest.raises(ValidationError):
            FSDPConfigSchema(enabled=True, reduce_dtype="float64")

    def test_planted_round_robin_outside_stage2_is_refused(self) -> None:
        with pytest.raises(ValidationError, match="stage-2 option"):
            DeepSpeedConfigSchema(enabled=True, zero_stage=3, round_robin_gradients=True)

    def test_round_robin_at_stage2_is_accepted(self) -> None:
        DeepSpeedConfigSchema(enabled=True, zero_stage=2, round_robin_gradients=True)


class TestMixedPrecisionPolicy:
    def test_gradients_reduce_in_float32_by_default(self) -> None:
        from spectramr.infrastructure.distributed.fsdp_wrap import _resolve_mixed_precision_policy

        policy = _resolve_mixed_precision_policy("bf16")
        assert policy.param_dtype == torch.bfloat16
        assert policy.reduce_dtype == torch.float32
        assert policy.buffer_dtype == torch.float32

    def test_match_param_restores_the_old_all_half_policy(self) -> None:
        from spectramr.infrastructure.distributed.fsdp_wrap import _resolve_mixed_precision_policy

        policy = _resolve_mixed_precision_policy("bf16", "match_param", "match_param")
        assert policy.reduce_dtype == torch.bfloat16 and policy.buffer_dtype == torch.bfloat16

    def test_null_disables_the_policy(self) -> None:
        from spectramr.infrastructure.distributed.fsdp_wrap import _resolve_mixed_precision_policy

        assert _resolve_mixed_precision_policy(None) is None

    def test_the_knobs_are_stamped_into_provenance(self) -> None:
        from spectramr.infrastructure.distributed.strategies import FSDPStrategy

        parallel = ParallelismConfigSchema(
            strategy="fsdp", fsdp={"enabled": True, "reduce_dtype": "match_param"}
        )
        precision = SimpleNamespace(enabled=True, dtype="bfloat16")
        config = SimpleNamespace(optimization=SimpleNamespace(precision=precision))
        ctx = SimpleNamespace(parallel=parallel, world_size=2, config=config)
        record = FSDPStrategy().provenance(ctx)["fsdp"]
        assert record["reduce_dtype"] == "match_param"
        assert record["buffer_dtype"] == "float32"
        # The precision stamped is the one that ran: followed from optimization.precision.
        assert record["mixed_precision"] == "bf16"
