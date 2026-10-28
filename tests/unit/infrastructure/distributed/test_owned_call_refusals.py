"""A non-forward call on a wrapped env model is refused by name, at audit and at build (#2629).

CycleGAN's GRL domain head reads gen_ab's bottleneck through ``forward_with_anatomy``,
which bypasses the DDP forward, so the generator gradient it carries is never
all-reduced. With ``losses.gan.domain_grl`` on, the audit refuses ddp and names the call,
and the strategy raises the same refusal when built. With it off, ddp is admitted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.distributed.strategy_compat import check_strategy_for_parallel
from spectramr.infrastructure.training.strategies.cyclegan_strategy import (
    CycleGANTrainingStrategy,
)

pytestmark = pytest.mark.unit

_CALL = "gen_ab.forward_with_anatomy"


def _config(backend: str, grl: bool) -> SimpleNamespace:
    return SimpleNamespace(
        parallel=SimpleNamespace(
            strategy=backend,
            deepspeed=SimpleNamespace(zero_stage=2),
            find_unused_parameters=False,
            gradient_as_bucket_view=False,
            static_graph=False,
        ),
        losses=SimpleNamespace(
            gan=SimpleNamespace(
                domain_grl=SimpleNamespace(enabled=grl),
                lambda_domain=1.0,
                lambda_cycle=10.0,
                lambda_identity=0.5,
            )
        ),
        model=SimpleNamespace(in_channels=1, out_channels=1),
    )


@pytest.mark.parametrize("backend", ["ddp", "deepspeed"])
def test_the_audit_refuses_the_grl_tap_naming_the_call_and_the_reason(backend):
    verdict = check_strategy_for_parallel(_config(backend, grl=True), CycleGANTrainingStrategy)
    assert verdict.refused
    if backend == "ddp":  # deepspeed is refused earlier, by the owned modules
        assert repr(_CALL) in verdict.message
        assert "bypasses the wrapper's forward" in verdict.message


def test_ddp_is_admitted_when_the_grl_tap_is_off():
    assert not check_strategy_for_parallel(
        _config("ddp", grl=False), CycleGANTrainingStrategy
    ).refused


@pytest.mark.parametrize("backend", ["none", "dp"])
def test_single_process_backends_admit_the_grl_tap(backend):
    assert not check_strategy_for_parallel(
        _config(backend, grl=True), CycleGANTrainingStrategy
    ).refused


def test_the_strategy_raises_the_same_refusal_when_built_under_ddp():
    strat = CycleGANTrainingStrategy.__new__(CycleGANTrainingStrategy)
    strat._owned_modules, strat._owned_wrappers = {}, {}
    strat.config = _config("ddp", grl=True)
    strat.env = SimpleNamespace(generator=None, discriminator=None, opt_g=None, opt_d=None)
    strat.device = torch.device("cpu")
    with pytest.raises(ConfigurationError, match=r"gen_ab\.forward_with_anatomy"):
        strat.setup_models()


# --- Generator methods called outside the wrapped forward (#3280's raises, one owner) ---

_PKG = "spectramr.infrastructure.training.strategies"
_CALL_CASES = [
    ("calibrated_qmap_strategy", "CalibratedQMapStrategy", "generator.render_triplet"),
    ("disentangled_strategy", "DisentangledTrainingStrategy", "generator.enc_c"),
    ("disentangled_vae_strategy", "DisentangledVAETrainingStrategy", "generator.encode_content"),
    ("disentangled_diffusion_strategy", "DisentangledDiffusionStrategy", "generator.bloch"),
    ("padnet_strategy", "PaDNetTrainingStrategy", "generator.predict_q_maps"),
]


def _call_cls(module: str, name: str) -> type:
    import importlib

    return getattr(importlib.import_module(f"{_PKG}.{module}"), name)


@pytest.mark.parametrize(("module", "name", "call"), _CALL_CASES)
def test_the_audit_refuses_ddp_naming_the_generator_call(module, name, call):
    verdict = check_strategy_for_parallel(_config("ddp", grl=False), _call_cls(module, name))
    assert verdict.refused
    assert repr(call) in verdict.message and "bypasses the wrapper's forward" in verdict.message


@pytest.mark.parametrize(("module", "name", "call"), _CALL_CASES)
@pytest.mark.parametrize("backend", ["none", "dp"])
def test_single_process_backends_admit_the_generator_call(module, name, call, backend):
    assert not check_strategy_for_parallel(
        _config(backend, grl=False), _call_cls(module, name)
    ).refused


def test_bloch_dc_off_admits_disentangled_diffusion_under_ddp():
    config = _config("ddp", grl=False)
    config.model.model_kwargs = {"lambda_bloch_dc": 0.0}
    cls = _call_cls("disentangled_diffusion_strategy", "DisentangledDiffusionStrategy")
    assert not check_strategy_for_parallel(config, cls).refused


def _helper_shell(backend: str):
    from spectramr.infrastructure.training.strategies.padnet_strategy import (
        PaDNetTrainingStrategy,
    )

    s = object.__new__(PaDNetTrainingStrategy)
    s.config = _config(backend, grl=False)
    return s


def test_the_runtime_helper_refuses_with_gradients_under_ddp():
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ConfigurationError, match=r"generator\.predict_q_maps"):
        _helper_shell("ddp").inner_for_call(model, "generator.predict_q_maps")


def test_the_runtime_helper_admits_no_grad_and_single_process():
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        assert _helper_shell("ddp").inner_for_call(model, "g.x") is model
    assert _helper_shell("dp").inner_for_call(torch.nn.DataParallel(model), "g.x") is model
