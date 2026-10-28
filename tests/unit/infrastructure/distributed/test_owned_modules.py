"""Strategy-owned modules under each parallel backend (#2629)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.distributed.owned_modules import wrap_owned_module
from spectramr.infrastructure.distributed.strategy_compat import evaluate_parallel_compat
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy


def _parallel(strategy: str) -> SimpleNamespace:
    return SimpleNamespace(
        strategy=strategy,
        find_unused_parameters=False,
        gradient_as_bucket_view=False,
        static_graph=False,
    )


@pytest.mark.parametrize("backend", ["none", "dp"])
def test_single_process_backends_run_the_module_unwrapped(backend):
    head = nn.Linear(2, 2)
    assert (
        wrap_owned_module(
            head, owner="S", name="h", parallel=_parallel(backend), device=torch.device("cpu")
        )
        is None
    )


@pytest.mark.parametrize("backend", ["fsdp", "deepspeed"])
def test_sharding_backends_refuse_a_module_built_after_the_wrap(backend):
    with pytest.raises(ConfigurationError, match="builds 'h' itself"):
        wrap_owned_module(
            nn.Linear(2, 2),
            owner="S",
            name="h",
            parallel=_parallel(backend),
            device=torch.device("cpu"),
        )


def test_unknown_backend_raises_rather_than_falling_back():
    with pytest.raises(ConfigurationError, match=r"unknown parallel\.strategy"):
        wrap_owned_module(
            nn.Linear(2, 2),
            owner="S",
            name="h",
            parallel=_parallel("zero3x"),
            device=torch.device("cpu"),
        )


def test_ddp_without_a_process_group_raises():
    with pytest.raises(RuntimeError):
        wrap_owned_module(
            nn.Linear(2, 2),
            owner="S",
            name="h",
            parallel=_parallel("ddp"),
            device=torch.device("cpu"),
        )


def test_base_registry_records_module_and_runs_it_unwrapped_single_process():
    probe = SimpleNamespace(
        config=SimpleNamespace(parallel=None),
        device=torch.device("cpu"),
        _owned_modules={},
        _owned_wrappers={},
    )
    head = nn.Linear(3, 1)
    assert BaseTrainingStrategy.register_owned_module(probe, "_head", head) is head
    assert BaseTrainingStrategy.owned_modules(probe) == {"_head": head}
    out = BaseTrainingStrategy.call_owned(probe, "_head", torch.ones(1, 3))
    assert out.shape == (1, 1)


def test_base_owns_no_modules_by_default():
    probe = SimpleNamespace(_owned_modules={}, _owned_wrappers={})
    assert BaseTrainingStrategy.owned_modules(probe) == {}


@pytest.mark.parametrize(
    ("dotted", "expect_refused"),
    [
        ("spectramr.infrastructure.training.strategies.xfield_fm_strategy:XFieldFMStrategy", False),
        (
            "spectramr.infrastructure.training.strategies.coord_kspace_gen_strategy:CoordKSpaceGenStrategy",
            False,
        ),
        (
            "spectramr.infrastructure.training.strategies.calibrated_qmap_strategy:CalibratedQMapStrategy",
            False,
        ),
        (
            "spectramr.infrastructure.training.strategies.multi_param_mapping_strategy:OneShotMultiParameterStrategy",
            False,
        ),
        (
            "spectramr.infrastructure.training.strategies.geomamba_ulf_strategy:GeoMambaULFStrategy",
            False,
        ),
        (
            "spectramr.infrastructure.training.strategies.virtual_fiducial_strategy:ConcreteVirtualFiducialStrategy",
            False,
        ),
    ],
)
def test_census_declarations_under_ddp(dotted, expect_refused):
    """Planted-violation pin: an owner that adds an unreduced parameter must be refused under ddp."""
    import importlib

    module_name, cls_name = dotted.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    assert evaluate_parallel_compat(cls, "ddp").refused is expect_refused


def test_batch_gated_builders_are_refused_under_ddp_and_admitted_single_process():
    """Rule 17: XFieldFM and CoordKSpaceGen build (and run) their module only for a batch
    that carries a key, as girf/field_probe/universal do, so they take the same verdict."""
    import importlib

    from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel

    for dotted in (
        "spectramr.infrastructure.training.strategies.xfield_fm_strategy:XFieldFMStrategy",
        "spectramr.infrastructure.training.strategies.coord_kspace_gen_strategy:CoordKSpaceGenStrategy",
    ):
        module_name, cls_name = dotted.split(":")
        cls = getattr(importlib.import_module(module_name), cls_name)
        flags = {"cross_modal": False, "conditioning": False}
        assert check_strategy_for_parallel(_config("ddp", None, **flags), cls).refused
        assert not check_strategy_for_parallel(_config("dp", None, **flags), cls).refused


# --- The static gate and the runtime helpers must agree (#2629, one owner) -----------

_OWNED_OWNERS = [
    "spectramr.infrastructure.training.strategies.xfield_fm_strategy:XFieldFMStrategy",
    "spectramr.infrastructure.training.strategies.coord_kspace_gen_strategy:CoordKSpaceGenStrategy",
    "spectramr.infrastructure.training.strategies.teichmuller_cold_diffusion_strategy:TeichmullerColdDiffusionStrategy",
    "spectramr.infrastructure.training.strategies.beltrami_motion_cortical_strategies:BeltramiMotionCorrectionStrategy",
    "spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies:AdaptiveSFCHSSCStrategy",
    "spectramr.infrastructure.training.strategies.diffusion:DiffusionTrainingStrategy",
    "spectramr.infrastructure.training.strategies.reconstruction:ReconstructionTrainingStrategy",
    "spectramr.infrastructure.training.strategies.multi_param_mapping_strategy:OneShotMultiParameterStrategy",
    "spectramr.infrastructure.training.strategies.spin_sde_strategy:SpinSDEStrategy",
    "spectramr.infrastructure.training.strategies.loupe_strategy:LOUPEStrategy",
    "spectramr.infrastructure.training.strategies.pilot_strategy:PILOTStrategy",
    "spectramr.infrastructure.training.strategies.cycle_bloch_strategy:CycleBlochStrategy",
    "spectramr.infrastructure.training.strategies.girf_aware_strategy:GIRFAwareStrategy",
    "spectramr.infrastructure.training.strategies.field_probe_coupled_strategy:FieldProbeCoupledStrategy",
    "spectramr.infrastructure.training.strategies.universal_reconstruction_strategy:UniversalReconstructionStrategy",
    "spectramr.infrastructure.training.strategies.physics_equivariant_ssl_strategy:PhysicsEquivariantSSLStrategy",
    "spectramr.infrastructure.training.strategies.cyclegan_strategy:CycleGANTrainingStrategy",
    "spectramr.infrastructure.training.strategies.cut_strategy:CUTTrainingStrategy",
    "spectramr.infrastructure.training.strategies.stargan_v2_strategy:StarGANv2TrainingStrategy",
    "spectramr.infrastructure.training.strategies.privileged_learning_strategy:PrivilegedLearningStrategy",
    "spectramr.infrastructure.training.strategies.pnp_strategy:PnPStrategy",
    "spectramr.infrastructure.training.strategies.ulf_map_strategy:UlfMapStrategy",
    "spectramr.infrastructure.training.strategies.jepa_strategy:JEPAStrategy",
    "spectramr.infrastructure.training.strategies.geomamba_ulf_strategy:GeoMambaULFStrategy",
    "spectramr.infrastructure.training.strategies.virtual_fiducial_strategy:ConcreteVirtualFiducialStrategy",
    "spectramr.infrastructure.training.strategies.ib_vf_strategy:IBVFTrainingStrategy",
    "spectramr.infrastructure.training.strategies.calibrated_qmap_strategy:CalibratedQMapStrategy",
    "spectramr.infrastructure.training.strategies.disentangled_strategy:DisentangledTrainingStrategy",
    "spectramr.infrastructure.training.strategies.disentangled_vae_strategy:DisentangledVAETrainingStrategy",
    "spectramr.infrastructure.training.strategies.disentangled_diffusion_strategy:DisentangledDiffusionStrategy",
    "spectramr.infrastructure.training.strategies.padnet_strategy:PaDNetTrainingStrategy",
]
_BACKENDS = [("none", None), ("dp", None), ("ddp", None), ("deepspeed", 0), ("deepspeed", 2)]


def _config(backend: str, zero_stage, *, cross_modal: bool, conditioning: bool) -> SimpleNamespace:
    return SimpleNamespace(
        parallel=SimpleNamespace(
            strategy=backend,
            deepspeed=SimpleNamespace(zero_stage=zero_stage),
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            static_graph=False,
        ),
        # One flag drives every optional owner, so each row covers both declaration states.
        training=SimpleNamespace(
            diffusion={"cross_modal": {"enabled": cross_modal}},
            multi_parameter=SimpleNamespace(uncertainty_weighting=cross_modal),
            pnp=SimpleNamespace(enabled=cross_modal),
            geomamba_ulf=SimpleNamespace(uncertainty=SimpleNamespace(enabled=cross_modal)),
        ),
        undersampling=SimpleNamespace(adaptive=SimpleNamespace(enabled=conditioning)),
        losses=SimpleNamespace(gan=SimpleNamespace(domain_grl=SimpleNamespace(enabled=cross_modal))),
        model=SimpleNamespace(
            conditioning=SimpleNamespace(
                enabled=conditioning, sources=["field_strength"] if conditioning else []
            )
        ),
    )


def _runtime_refuses(mechanism: str, config: SimpleNamespace) -> bool:
    """Whether the runtime helper for ``mechanism`` raises ConfigurationError here.

    Under ddp a module is wrapped, which needs a process group; the RuntimeError it
    raises there is the wrap being reached, so it counts as admitted.
    """
    from spectramr.infrastructure.distributed.owned_modules import (
        OWNED_BUFFERS,
        OWNED_CALL,
        OWNED_MODULE,
        refuse_outside_forward_call,
        sync_owned_buffers,
    )
    from spectramr.infrastructure.training.strategies.owned_param_groups import (
        add_owned_param_group,
    )

    try:
        if mechanism == OWNED_BUFFERS:
            sync_owned_buffers([], owner="S", name="h", parallel=config.parallel)
        elif mechanism == OWNED_CALL:
            refuse_outside_forward_call(config.parallel, owner="S", name="h.encode")
        elif mechanism == OWNED_MODULE:
            wrap_owned_module(
                nn.Linear(2, 2),
                owner="S",
                name="h",
                parallel=config.parallel,
                device=torch.device("cpu"),
            )
        else:
            strategy = SimpleNamespace(config=config)
            add_owned_param_group(
                strategy, _FakeOptimizer(), {"params": []}, owner="its parameters"
            )
    except ConfigurationError:
        return True
    except RuntimeError:
        return False
    return False


class _FakeOptimizer:
    def add_param_group(self, group) -> None:
        self.last = group


@pytest.mark.parametrize("dotted", _OWNED_OWNERS)
@pytest.mark.parametrize(("backend", "stage"), _BACKENDS)
@pytest.mark.parametrize("cross_modal", [False, True])
@pytest.mark.parametrize("conditioning", [False, True])
def test_gate_refuses_an_owned_module_exactly_when_the_runtime_would(
    dotted, backend, stage, cross_modal, conditioning
):
    """Planted-violation parity: each owned parameter the gate reports is refused by the
    runtime helper that builds it, and only then. On the base branch the deepspeed rows
    (and diffusion's cross-modal ddp row) pass the gate and raise at build time."""
    import importlib

    from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel

    module_name, cls_name = dotted.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    config = _config(backend, stage, cross_modal=cross_modal, conditioning=conditioning)
    declared = cls.owned_module_declarations(config)
    gate_refused = check_strategy_for_parallel(config, cls).refused
    runtime_refused = any(_runtime_refuses(mech, config) for _name, mech in declared)
    if declared:
        assert gate_refused == runtime_refused, (dotted, backend, stage, declared)
    else:
        assert not gate_refused


@pytest.mark.parametrize(
    ("dotted", "name"),
    [
        (_OWNED_OWNERS[7], "_log_sigmas"),
        (_OWNED_OWNERS[8], "diffusion"),
        (_OWNED_OWNERS[9], "_mask_logits"),
        (_OWNED_OWNERS[10], "_trajectory"),
    ],
)
def test_parameter_owners_declare_their_holder_so_the_parity_rows_are_not_vacuous(dotted, name):
    import importlib

    module_name, cls_name = dotted.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    config = _config("ddp", None, cross_modal=True, conditioning=True)
    assert (name, "module") in cls.owned_module_declarations(config)


def test_conditioner_is_owned_only_by_steps_that_apply_it():
    """Diffusion never reaches _apply_input_conditioning, so enabling conditioning on a
    diffusion arm builds no conditioner and must not be refused under deepspeed."""
    from spectramr.infrastructure.training.strategies.diffusion import DiffusionTrainingStrategy
    from spectramr.infrastructure.training.strategies.reconstruction import (
        ReconstructionTrainingStrategy,
    )

    config = _config("deepspeed", 2, cross_modal=False, conditioning=True)
    assert DiffusionTrainingStrategy.owned_module_declarations(config) == ()
    assert ReconstructionTrainingStrategy.owned_module_declarations(config) == (
        ("_conditioner", "param_group"),
    )


def test_owned_verdict_table_is_the_one_decision():
    from spectramr.infrastructure.distributed.owned_modules import (
        OWNED_BUFFERS,
        OWNED_CALL,
        OWNED_MODULE,
        OWNED_PARAM_GROUP,
        owned_verdict,
    )

    assert owned_verdict(OWNED_MODULE, "ddp") == "wrap"
    assert owned_verdict(OWNED_PARAM_GROUP, "ddp") == "refuse"
    assert owned_verdict(OWNED_CALL, "ddp") == "refuse"
    assert owned_verdict(OWNED_BUFFERS, "ddp") == "wrap"
    for mech in (OWNED_MODULE, OWNED_PARAM_GROUP, OWNED_CALL, OWNED_BUFFERS):
        assert owned_verdict(mech, "none") == "bare"
        assert owned_verdict(mech, "dp") == "bare"
        assert owned_verdict(mech, "fsdp") == "refuse"
        assert owned_verdict(mech, "deepspeed") == "refuse"
    with pytest.raises(ConfigurationError, match=r"unknown parallel\.strategy"):
        owned_verdict(OWNED_MODULE, "zero3x")


def test_pnp_declares_its_spectral_norm_vectors_only_when_enabled():
    """The parity rows for PnP are not vacuous: the declaration follows training.pnp.enabled."""
    from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel
    from spectramr.infrastructure.training.strategies.pnp_strategy import PnPStrategy

    on = _config("deepspeed", 2, cross_modal=True, conditioning=False)
    off = _config("deepspeed", 2, cross_modal=False, conditioning=False)
    assert ("_denoiser", "buffers") in PnPStrategy.owned_module_declarations(on)
    verdict = check_strategy_for_parallel(on, PnPStrategy)
    assert verdict.refused and "adds '_denoiser' to the wrapped model" in verdict.message
    assert not check_strategy_for_parallel(off, PnPStrategy).refused
    ddp = _config("ddp", None, cross_modal=True, conditioning=False)
    assert not check_strategy_for_parallel(ddp, PnPStrategy).refused
