"""Regression: strategy-owned auxiliary learnable modules must be registered
with the generator optimizer (``self.env.opt_g``) so they actually train.

2026-05-31 audit found two failure modes:

* **Wrong accessor** — `getattr(self.env, "optimizer_g", None) or
  getattr(self.env, "optimizer", None)`; the real attribute is ``opt_g``,
  so the lookup returned ``None`` and `add_param_group` never ran
  (girf kernel, multi_param log-sigmas, field_probe demodulator, ...).
* **No registration at all** — the module was built (and sometimes left on
  CPU) but never handed to any optimizer (cycle_bloch ParameterEstimator,
  geomamba uncertainty weights, beltrami mu-head).

Each test drives the strategy's lazy builder with a stub env carrying a real
optimizer and asserts the new params land in ``opt_g.param_groups``.
"""

from __future__ import annotations

import types

import torch

from spectramr.infrastructure.training.strategies.cycle_bloch_strategy import (
    CycleBlochStrategy,
)
from spectramr.infrastructure.training.strategies.field_probe_coupled_strategy import (
    FieldProbeCoupledStrategy,
)
from spectramr.infrastructure.training.strategies.girf_aware_strategy import (
    GIRFAwareStrategy,
)
from spectramr.infrastructure.training.strategies.physics_equivariant_ssl_strategy import (
    PhysicsEquivariantSSLStrategy,
)
from spectramr.infrastructure.training.strategies.universal_reconstruction_strategy import (
    UniversalReconstructionStrategy,
)


def _env_with_opt():
    dummy = torch.nn.Parameter(torch.zeros(1))
    return types.SimpleNamespace(opt_g=torch.optim.SGD([dummy], lr=0.1))


def _param_ids(opt):
    return {id(p) for g in opt.param_groups for p in g["params"]}


class TestAuxParamRegistration:
    def test_girf_kernel_registered(self):
        # c1: wrong accessor previously meant opt was None -> kernel untrained.
        s = object.__new__(GIRFAwareStrategy)
        s.env = _env_with_opt()
        s.kernel_size = 21
        s._girf = None
        before = _param_ids(s.env.opt_g)
        girf = s._ensure_girf(3, torch.device("cpu"))
        after = _param_ids(s.env.opt_g)
        assert after - before, "GIRF kernel params were not added to opt_g"
        assert all(id(p) in after for p in girf.parameters())

    def test_field_probe_demodulator_registered(self):
        s = object.__new__(FieldProbeCoupledStrategy)
        s.env = _env_with_opt()
        s._demod = None
        before = _param_ids(s.env.opt_g)
        demod = s._ensure_demod(16, torch.device("cpu"))
        after = _param_ids(s.env.opt_g)
        assert after - before, "PhaseDemodulator params were not added to opt_g"
        assert all(id(p) in after for p in demod.parameters())

    def test_cycle_bloch_estimator_registered(self):
        # c2: estimator was built lazily but never handed to any optimizer.
        s = object.__new__(CycleBlochStrategy)
        s.env = _env_with_opt()
        s.estimator = None
        s._estimator_channels = None
        s._estimate_b1 = False
        s.config = types.SimpleNamespace(parallel=None)  # read when the estimator is registered
        s.device = torch.device("cpu")
        s._owned_modules, s._owned_wrappers = {}, {}
        before = _param_ids(s.env.opt_g)
        est = s._get_estimator(torch.randn(1, 2, 8, 8))
        after = _param_ids(s.env.opt_g)
        assert after - before, "ParameterEstimator params were not added to opt_g"
        assert all(id(p) in after for p in est.parameters())

    def test_universal_recon_prompt_embedder_registered(self):
        s = object.__new__(UniversalReconstructionStrategy)
        s.env = _env_with_opt()
        s.device = torch.device("cpu")
        before = _param_ids(s.env.opt_g)
        emb = s._get_prompt_embedder(4)
        after = _param_ids(s.env.opt_g)
        assert after - before, "PhysicsPromptEmbedding params were not added to opt_g"
        assert all(id(p) in after for p in emb.parameters())

    def test_physics_eq_ssl_lazy_projection_registered(self):
        s = object.__new__(PhysicsEquivariantSSLStrategy)
        s.env = _env_with_opt()
        s.latent_dim = 128
        # generator_model property returns self.env.generator; emit a richer
        # latent so _encode builds + registers the lazy projection head.
        s.env.generator = lambda x: torch.randn(2, 256)
        before = _param_ids(s.env.opt_g)
        z = s._encode(torch.randn(2, 3, 4, 4))
        after = _param_ids(s.env.opt_g)
        assert after - before, "lazy _projection params were not added to opt_g"
        assert z.shape == (2, 128)
