"""The strategy x parallel-backend gate: every rule has a planted violation.

A strategy that owns a module outside ``env.models`` trains a different copy on
each rank under DDP/ZeRO, and nothing errors. The gate is only worth its name for
the shapes it has been watched to refuse (CLAUDE.md #15), so each test below
plants one.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.infrastructure.distributed.strategy_compat import (
    REPLICATED,
    SHARDED,
    backend_kind,
    check_strategy_for_parallel,
    evaluate_parallel_compat,
)

pytestmark = pytest.mark.unit


class _NoOwned:
    """A fake owns no strategy-built parameters, so the owned-module check has nothing to judge."""

    @classmethod
    def owned_module_declarations(cls, config):
        return ()


class _Undeclared(_NoOwned):
    parallel_compat = None


class _ReplicatedOnly(_NoOwned):
    parallel_compat = frozenset({REPLICATED})


class _Both(_NoOwned):
    parallel_compat = frozenset({REPLICATED, SHARDED})


class _SingleProcessOnly(_NoOwned):
    parallel_compat = frozenset()


class _ChildOfReplicatedOnly(_ReplicatedOnly):
    """Undeclared of its own: inherits the parent's restriction, not its permission."""


class _ChildOfBoth(_Both):
    pass


class TestBackendKind:
    @pytest.mark.parametrize(
        ("strategy", "stage", "kind"),
        [
            ("none", None, None),
            (None, None, None),
            ("dp", None, REPLICATED),
            ("ddp", None, REPLICATED),
            ("fsdp", None, SHARDED),
            ("deepspeed", 0, REPLICATED),
            ("deepspeed", 2, REPLICATED),
            ("deepspeed", 3, SHARDED),
        ],
    )
    def test_classification(self, strategy, stage, kind) -> None:
        assert backend_kind(strategy, stage) == kind

    def test_unknown_backend_raises_rather_than_defaulting(self) -> None:
        with pytest.raises(ValueError, match="cannot classify"):
            backend_kind("horovod")


class TestRefusals:
    @pytest.mark.parametrize("backend", ["dp", "ddp", "fsdp", "deepspeed"])
    def test_empty_declaration_refuses_every_parallel_backend(self, backend) -> None:
        verdict = evaluate_parallel_compat(_SingleProcessOnly, backend, 2)
        assert verdict.refused

    def test_replicated_only_refuses_fsdp_and_zero3(self) -> None:
        assert evaluate_parallel_compat(_ReplicatedOnly, "fsdp").refused
        assert evaluate_parallel_compat(_ReplicatedOnly, "deepspeed", 3).refused

    def test_replicated_only_allows_ddp_and_zero2(self) -> None:
        assert not evaluate_parallel_compat(_ReplicatedOnly, "ddp").refused
        assert not evaluate_parallel_compat(_ReplicatedOnly, "deepspeed", 2).refused

    def test_undeclared_is_refused_on_a_sharded_backend(self) -> None:
        verdict = evaluate_parallel_compat(_Undeclared, "fsdp")
        assert verdict.refused and verdict.undeclared

    def test_a_restriction_is_inherited(self) -> None:
        assert evaluate_parallel_compat(_ChildOfReplicatedOnly, "fsdp").refused

    def test_a_permission_is_not_inherited(self) -> None:
        """A subclass can add the very module that made its parent safe."""
        verdict = evaluate_parallel_compat(_ChildOfBoth, "fsdp")
        assert verdict.refused and verdict.undeclared

    def test_a_subclass_cannot_widen_past_its_own_restriction(self) -> None:
        class _Narrowed(_Both):
            parallel_compat = frozenset({REPLICATED})

        assert evaluate_parallel_compat(_Narrowed, "fsdp").refused

    def test_refusal_names_the_class_and_the_backend(self) -> None:
        message = evaluate_parallel_compat(_SingleProcessOnly, "ddp").message
        assert "_SingleProcessOnly" in message and "ddp" in message


class TestAllowed:
    def test_single_process_is_never_gated(self) -> None:
        verdict = evaluate_parallel_compat(_SingleProcessOnly, "none")
        assert not verdict.refused and not verdict.undeclared

    def test_declared_support_passes(self) -> None:
        verdict = evaluate_parallel_compat(_Both, "fsdp")
        assert not verdict.refused and not verdict.undeclared

    def test_undeclared_on_a_replicated_backend_runs_but_is_flagged(self) -> None:
        """Most arms run this way today, so it must be advisory, not a refusal."""
        verdict = evaluate_parallel_compat(_Undeclared, "ddp")
        assert not verdict.refused and verdict.undeclared


class TestDeclarationsAreValidated:
    def test_unknown_token_raises(self) -> None:
        class _Typo:
            parallel_compat = frozenset({"replicate"})

        with pytest.raises(ValueError, match="unknown token"):
            evaluate_parallel_compat(_Typo, "ddp")


class TestFromConfig:
    @staticmethod
    def _config(strategy, stage=None):
        return SimpleNamespace(
            parallel=SimpleNamespace(strategy=strategy, deepspeed=SimpleNamespace(zero_stage=stage))
        )

    def test_zero_stage_is_read_from_the_deepspeed_block(self) -> None:
        assert not check_strategy_for_parallel(
            self._config("deepspeed", 2), _ReplicatedOnly
        ).refused
        assert check_strategy_for_parallel(self._config("deepspeed", 3), _ReplicatedOnly).refused

    def test_no_parallel_block_is_single_process(self) -> None:
        assert not check_strategy_for_parallel(SimpleNamespace(), _SingleProcessOnly).refused


class TestRealStrategies:
    """The shipped declarations, so a refactor cannot quietly drop one."""

    @pytest.mark.parametrize(
        ("module", "name"),
        [
            ("cyclegan_strategy", "CycleGANTrainingStrategy"),
            ("cut_strategy", "CUTTrainingStrategy"),
            ("stargan_v2_strategy", "StarGANv2TrainingStrategy"),
            ("privileged_learning_strategy", "PrivilegedLearningStrategy"),
            ("jepa_strategy", "JEPAStrategy"),
            ("multi_contrast_contrastive_strategy", "MultiContrastContrastiveStrategy"),
            ("distillation_strategy", "ConcreteDistillationStrategy"),
            ("vf_consistency_distillation_strategy", "VFConsistencyDistillationStrategy"),
        ],
    )
    def test_owners_migrated_off_the_refuse_all_list_keep_fsdp_refused(self, module, name) -> None:
        """Their own modules are DDP-wrapped or frozen copies (#2629); none is sharding-safe."""
        import importlib

        cls = getattr(
            importlib.import_module(f"spectramr.infrastructure.training.strategies.{module}"),
            name,
        )
        assert not evaluate_parallel_compat(cls, "ddp").refused
        assert evaluate_parallel_compat(cls, "fsdp").refused

    def test_reconstruction_keeps_its_fsdp_arm(self) -> None:
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        assert not evaluate_parallel_compat(ReconstructionTrainingStrategy, "fsdp").refused

    def test_diffusion_stays_runnable_on_zero2_but_not_zero3(self) -> None:
        from spectramr.infrastructure.training.strategies.diffusion import (
            DiffusionTrainingStrategy,
        )

        assert not evaluate_parallel_compat(DiffusionTrainingStrategy, "deepspeed", 2).refused
        assert evaluate_parallel_compat(DiffusionTrainingStrategy, "deepspeed", 3).refused

    def test_vf_distillation_declares_its_own_permission(self) -> None:
        from spectramr.infrastructure.training.strategies.vf_consistency_distillation_strategy import (
            VFConsistencyDistillationStrategy,
        )

        assert "parallel_compat" in vars(VFConsistencyDistillationStrategy)
        assert not evaluate_parallel_compat(VFConsistencyDistillationStrategy, "deepspeed", 2).refused
        assert evaluate_parallel_compat(VFConsistencyDistillationStrategy, "deepspeed", 3).refused

    def test_every_declared_token_in_the_tree_is_valid(self) -> None:
        """A typo in a declaration would otherwise only surface on a cluster."""
        from spectramr.infrastructure.distributed.strategy_compat import VALID_TOKENS
        from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

        factory = TrainingStrategyFactory()
        bad = []
        for key in factory.STRATEGY_CLASS_PATHS:
            try:
                cls = factory._load_strategy_class(key)
            except Exception:  # an optional dependency absent here is not this test's concern
                continue
            declared = vars(cls).get("parallel_compat")
            if declared is not None and not set(declared) <= VALID_TOKENS:
                bad.append((key, sorted(declared)))
        assert not bad


OWNED_PARAM_STRATEGIES = [
    ("scas_strategy", "SCASStrategy"),
    ("mri_slam_strategy", "MRISLAMStrategy"),
    ("pipeline_strategy", "MultiTrainingStrategy"),
    ("tto_strategy", "ConcreteTTOStrategy"),
]


class TestOwnedParameterStrategies:
    """Strategies that train parameters outside ``env.models`` (audited line by line, #2629).

    Under DDP each rank would train its own copy; under ZeRO the optimizer has no
    ``add_param_group`` at all, so the arm crashes on the first step. Every one of them
    must refuse every parallel backend, and must not borrow a parent's permission.
    """

    @staticmethod
    def _cls(module, name):
        import importlib

        return getattr(
            importlib.import_module(f"spectramr.infrastructure.training.strategies.{module}"),
            name,
        )

    @pytest.mark.parametrize(("module", "name"), OWNED_PARAM_STRATEGIES)
    @pytest.mark.parametrize(
        ("backend", "stage"), [("ddp", None), ("dp", None), ("deepspeed", 2), ("fsdp", None)]
    )
    def test_refuses_every_parallel_backend(self, module, name, backend, stage) -> None:
        verdict = evaluate_parallel_compat(self._cls(module, name), backend, stage)
        assert verdict.refused, f"{name} under {backend}"

    @pytest.mark.parametrize(("module", "name"), OWNED_PARAM_STRATEGIES)
    def test_single_process_is_untouched(self, module, name) -> None:
        assert not evaluate_parallel_compat(self._cls(module, name), "none").refused

    def test_the_reconstruction_parent_keeps_its_own_permission(self) -> None:
        from spectramr.infrastructure.training.strategies.reconstruction import (
            ReconstructionTrainingStrategy,
        )

        assert not evaluate_parallel_compat(ReconstructionTrainingStrategy, "fsdp").refused
        assert evaluate_parallel_compat(
            self._cls("loupe_strategy", "LOUPEStrategy"), "fsdp"
        ).refused

    def test_a_subclass_of_a_refused_strategy_stays_refused(self) -> None:
        """A restriction is inherited by design: a child cannot borrow permission."""
        slam = self._cls("mri_slam_strategy", "MRISLAMStrategy")

        class _Child(slam):
            pass

        assert evaluate_parallel_compat(_Child, "ddp").refused


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("field_cocycle_strategy", "FieldCocycleTranslationStrategy"),
        ("mrf_acquisition_strategies", "ConformalMRFDictlessReconStrategy"),
        ("operator_id_bch_strategy", "OperatorIdBCHTrainingStrategy"),
        ("se3_equivariant_navigator_strategy", "SE3EquivariantNavigatorStrategy"),
    ],
)
class TestGeneratorMethodCallers:
    """Strategies that call a non-forward method on the generator (#2696)."""

    @staticmethod
    def _cls(module: str, name: str) -> type:
        import importlib

        return getattr(
            importlib.import_module(f"spectramr.infrastructure.training.strategies.{module}"), name
        )

    @pytest.mark.parametrize("backend", ["ddp", "fsdp", "deepspeed"])
    def test_refused_under_every_multi_process_backend(self, module, name, backend) -> None:
        verdict = evaluate_parallel_compat(self._cls(module, name), backend, 2)
        assert verdict.refused

    def test_single_process_is_not_gated(self, module, name) -> None:
        assert not evaluate_parallel_compat(self._cls(module, name), "none").refused
