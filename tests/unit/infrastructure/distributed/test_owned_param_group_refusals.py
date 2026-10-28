"""Strategies whose owned parameters have no rank-symmetric creation point (#2629).

Each builds a module from a batch (or draws per-rank operators), so it cannot be
DDP-wrapped. It declares the parameters as a bare group: the audit refuses ddp, fsdp and
deepspeed naming the attribute, and the runtime builder raises the same refusal if the
audit is bypassed. Single-process backends (none, dp) are admitted.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel

_PKG = "spectramr.infrastructure.training.strategies"

_CASES = [
    ("girf_aware_strategy", "GIRFAwareStrategy", "_girf"),
    ("field_probe_coupled_strategy", "FieldProbeCoupledStrategy", "_demod"),
    ("universal_reconstruction_strategy", "UniversalReconstructionStrategy", "_prompt_embedder"),
    ("physics_equivariant_ssl_strategy", "PhysicsEquivariantSSLStrategy", "latent_ops"),
    ("xfield_fm_strategy", "XFieldFMStrategy", "_hypernet"),
    ("coord_kspace_gen_strategy", "CoordKSpaceGenStrategy", "_siren"),
    ("virtual_fiducial_strategy", "ConcreteVirtualFiducialStrategy", "_conditioner"),
]


def _cls(module: str, name: str) -> type:
    return getattr(importlib.import_module(f"{_PKG}.{module}"), name)


def _config(backend: str, zero_stage: int | None = None, *, attr: str = "") -> SimpleNamespace:
    from spectramr.config.schemas.conditioning import ConditioningConfig

    # VF declares its conditioner only when conditioning is on; the others run without it.
    on = attr == "_conditioner"
    cond = ConditioningConfig(enabled=on, sources=["severity_vec"] if on else [], embed_dim=8)
    return SimpleNamespace(
        model=SimpleNamespace(conditioning=cond),
        parallel=SimpleNamespace(
            strategy=backend,
            deepspeed=SimpleNamespace(zero_stage=zero_stage),
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            static_graph=False,
        ),
    )


@pytest.mark.parametrize(("module", "name", "attr"), _CASES)
@pytest.mark.parametrize(("backend", "stage"), [("ddp", None), ("deepspeed", 2), ("fsdp", None)])
def test_the_audit_refuses_multi_process_backends_naming_the_attribute(
    module, name, attr, backend, stage
):
    verdict = check_strategy_for_parallel(_config(backend, stage, attr=attr), _cls(module, name))
    assert verdict.refused
    if backend != "fsdp":  # fsdp is refused by the class token, before the declarations
        assert repr(attr) in verdict.message


@pytest.mark.parametrize(("module", "name", "attr"), _CASES)
@pytest.mark.parametrize("backend", ["none", "dp"])
def test_single_process_backends_are_admitted(module, name, attr, backend):
    assert not check_strategy_for_parallel(_config(backend, attr=attr), _cls(module, name)).refused


def _shell(module: str, name: str, backend: str):
    """The strategy with only what its owned-parameter builder reads."""
    s = object.__new__(_cls(module, name))
    s.config = _config(backend, attr="_conditioner")
    s.device = torch.device("cpu")
    s.env = SimpleNamespace(opt_g=torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1))
    return s


def _build(s, attr: str) -> None:
    if attr == "_girf":
        s._girf = None
        s._ensure_girf(3, torch.device("cpu"))
    elif attr == "_demod":
        s._demod = None
        s._ensure_demod(8, torch.device("cpu"))
    elif attr == "_prompt_embedder":
        s._get_prompt_embedder(4)
    elif attr == "_hypernet":
        s._hypernet = None
        s._ensure_hypernet(4, torch.device("cpu"))
    elif attr == "_conditioner":
        s._conditioner = None
        s._ensure_conditioner(2)
    elif attr == "_siren":
        s._siren, s._siren_in_features = None, 2
        s._ensure_siren(2, torch.device("cpu"))
    else:
        s._register_on_opt_g(nn.Linear(2, 2))


@pytest.mark.parametrize(("module", "name", "attr"), _CASES)
def test_the_runtime_builder_raises_under_ddp(module, name, attr):
    with pytest.raises(ConfigurationError, match="would not reduce its gradients"):
        _build(_shell(module, name, "ddp"), attr)


@pytest.mark.parametrize(("module", "name", "attr"), _CASES)
def test_the_runtime_builder_adds_the_group_single_process(module, name, attr):
    s = _shell(module, name, "none")
    _build(s, attr)
    assert len(s.env.opt_g.param_groups) == 2
