"""DDIM sampler: the compiled-step alias that never compiled anything.

``self._step_compiled = self._ddim_step`` was a plain rebinding -- no
``torch.compile``, no configuration, nothing that could ever make it one. The
call site read ``x = self._step_compiled(...)``, so every reader of the sampling
loop was told the step was compiled and every profile said otherwise. That is
pitfall 16 in its cheapest form: a name promising a mechanism that does not
exist, which is worse than the absence because it stops anyone looking.

Compilation is a build-time decision here (``optimization.compile.*``, placed by
``builders/compile_placement``), not something a sampler arranges for itself, so
the alias is deleted rather than wired.
"""

from __future__ import annotations

import torch

from spectramr.models.diffusion.ddim_sampler import DDIMSampler


class _StubDiffusion:
    """The three attributes `_precompute_ddim_parameters` reads.

    A stub rather than a real `Diffusion`: this suite is about the sampler's
    own wiring, and a real schedule would make the test depend on a module it
    is not testing.
    """

    def __init__(self, steps: int = 8) -> None:
        self.device = torch.device("cpu")
        betas = torch.linspace(1e-4, 0.02, steps)
        alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])


def _sampler(**kwargs) -> DDIMSampler:
    return DDIMSampler(_StubDiffusion(), **kwargs)


class TestNoCompiledStepFacade:
    def test_the_alias_is_gone(self) -> None:
        """Planted: reintroducing the rebinding restores the false promise."""
        assert not hasattr(_sampler(), "_step_compiled")

    def test_the_real_step_is_what_the_loop_calls(self) -> None:
        import inspect

        source = inspect.getsource(DDIMSampler.sample)
        assert "_step_compiled" not in source
        assert "self._ddim_step(" in source


class TestConstruction:
    def test_the_schedule_is_precomputed(self) -> None:
        sampler = _sampler()
        assert sampler.ddim_alphas.shape == sampler.alphas_cumprod.shape
        assert torch.isfinite(sampler.ddim_alphas).all()

    def test_eta_zero_is_deterministic(self) -> None:
        """`eta` scales the noise term; zero must leave no stochasticity."""
        assert torch.equal(_sampler(eta=0.0).ddim_sigmas, torch.zeros(8))

    def test_a_positive_eta_admits_noise(self) -> None:
        assert (_sampler(eta=1.0).ddim_sigmas[1:] > 0).any()

    def test_data_consistency_is_opt_in(self) -> None:
        """`dc_weight: 0` must not build the module -- a no-op DC that still
        runs is indistinguishable in a profile from one that is doing work."""
        assert _sampler(dc_weight=0.0).data_consistency is None
        assert _sampler(dc_weight=0.5).data_consistency is not None
