"""Posterior diffusion samplers (PR-20 / Part-I B).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-20.

Three measurement-aware samplers for inverse problems:

- :class:`DPSSampler` — Diffusion Posterior Sampling (Chung et al. 2023).
  Adds ``∇_x_t || A · x_0(x_t) - y ||²`` to the predicted score every
  step.
- :class:`PiGDMSampler` — Pseudo-inverse Guided Diffusion (Song et al.
  2023). Replaces the score correction with the pseudo-inverse of the
  forward operator.
- :class:`DDSSampler` — Decomposed Diffusion Sampling (Chung et al. 2024).
  Splits each step into a diffusion-prior update and a CG-based
  data-consistency update.

All three plug into the standard diffusion-strategy sampling loop via
the existing ``@register_sampler`` registry (CC-1).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.tangent_cg import cg_data_consistency

from .registry import register_sampler


class _PosteriorSamplerBase:
    """Shared scaffolding for measurement-aware diffusion samplers."""

    def __init__(
        self,
        n_steps: int = 50,
        eta: float = 0.0,
        guidance_weight: float = 1.0,
    ) -> None:
        self.n_steps = n_steps
        self.eta = eta
        self.guidance_weight = guidance_weight

    @staticmethod
    def _x0_from_eps(x_t: torch.Tensor, eps: torch.Tensor, alpha_bar: torch.Tensor) -> torch.Tensor:
        return (x_t - (1 - alpha_bar).sqrt() * eps) / alpha_bar.sqrt().clamp(min=1e-12)


@register_sampler("dps_posterior", aliases=["DPSPosterior"])
class DPSSampler(_PosteriorSamplerBase):
    """Diffusion Posterior Sampling — Chung et al. 2023.

    Registered as ``dps_posterior`` to avoid colliding with the
    pre-existing ``DPSMRISampler`` (alias ``dps``) in
    ``spectramr.models.diffusion.pula_sampler``. The two are different
    formulations — the PULA one is k-space-aware and SENSE-coupled;
    this one is the canonical pixel-space DPS algorithm.
    """

    def sample(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        shape: tuple[int, ...],
        alphas_cumprod: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, self.n_steps, dtype=torch.long)
        for t in ts:
            x_t = x_t.detach().requires_grad_(True)
            ab = alphas_cumprod[t]
            eps_pred = score_fn(x_t, t.to(device))
            x0_hat = self._x0_from_eps(x_t, eps_pred, ab)
            # Data-consistency gradient term.
            res = (forward_op(x0_hat) - y).pow(2).sum()
            grad = torch.autograd.grad(res, x_t, retain_graph=False)[0]
            # Standard DDIM step + guidance correction.
            ab_prev = alphas_cumprod[t - 1] if t > 0 else torch.ones_like(ab)
            sigma = self.eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
            mean = ab_prev.sqrt() * x0_hat + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
            x_t = (mean - self.guidance_weight * grad + sigma * noise).detach()
        return x_t


@register_sampler("pi_gdm", aliases=["PiGDM", "pseudoinverse_guided"])
class PiGDMSampler(_PosteriorSamplerBase):
    """Pseudo-inverse Guided Diffusion Model — Song et al. 2023."""

    def sample(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        shape: tuple[int, ...],
        alphas_cumprod: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, self.n_steps, dtype=torch.long)
        for t in ts:
            ab = alphas_cumprod[t]
            with torch.no_grad():
                eps_pred = score_fn(x_t, t.to(device))
                x0_hat = self._x0_from_eps(x_t, eps_pred, ab)
                # Pseudo-inverse correction in measurement space.
                residual = y - forward_op(x0_hat)
                correction = adjoint_op(residual) * self.guidance_weight
                x0_corrected = x0_hat + correction
                ab_prev = alphas_cumprod[t - 1] if t > 0 else torch.ones_like(ab)
                sigma = self.eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
                mean = (
                    ab_prev.sqrt() * x0_corrected
                    + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
                )
                noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
                x_t = mean + sigma * noise
        return x_t


class DDSSampler(_PosteriorSamplerBase):
    """Decomposed Diffusion Sampling — Chung et al. 2024 (model-agnostic core).

    Split each step into:
      (a) prior update via the score model (DDIM step), and
      (b) CG data-consistency update on the resulting predicted x0.

    This is the low-level, model-agnostic algorithm: ``sample`` takes an explicit
    ``score_fn`` and ``forward_op``/``adjoint_op``. For the config-reachable,
    generator-dispatch convention (``get_sampler("dds", model=..., ...)`` +
    ``sample(measurement, mask)``) use :class:`DDSReconSampler`, which builds
    those callables from a score model and delegates the CG step to the shared
    ``cg_data_consistency`` primitive.
    """

    def __init__(
        self, n_steps: int = 50, eta: float = 0.0, cg_iters: int = 5, cg_lambda: float = 1e-2
    ) -> None:
        super().__init__(n_steps=n_steps, eta=eta, guidance_weight=1.0)
        self.cg_iters = cg_iters
        self.cg_lambda = cg_lambda

    def sample(
        self,
        score_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        shape: tuple[int, ...],
        alphas_cumprod: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x_t = torch.randn(shape, device=device)
        ts = torch.linspace(self.n_steps - 1, 0, self.n_steps, dtype=torch.long)

        for t in ts:
            ab = alphas_cumprod[t]
            with torch.no_grad():
                eps_pred = score_fn(x_t, t.to(device))
                x0_hat = self._x0_from_eps(x_t, eps_pred, ab)
                # Stage (b): CG data-consistency on x0 via the shared physics
                # primitive (correct Hermitian inner product for complex k-space).
                x0_dc = cg_data_consistency(
                    x0_hat,
                    forward_op,
                    adjoint_op,
                    y,
                    n_iter=self.cg_iters,
                    lam=self.cg_lambda,
                )
                ab_prev = alphas_cumprod[t - 1] if t > 0 else torch.ones_like(ab)
                sigma = self.eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).sqrt()
                mean = (
                    ab_prev.sqrt() * x0_dc + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
                )
                noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
                x_t = mean + sigma * noise
        return x_t


@register_sampler("dds", aliases=["DDS", "decomposed_diffusion"])
class DDSReconSampler(nn.Module):
    """Generator-dispatch adapter for Decomposed Diffusion Sampling (SOTA T1 / step 2).

    Makes ``dds`` reachable from YAML (``inference_sampler: dds``) via the same
    convention every config-routed sampler uses — construction with
    ``get_sampler("dds", model=..., num_timesteps=..., max_acceleration=...,
    center_fraction=..., dc_method=..., dc_weight=..., sampling_steps=...)`` and
    invocation as ``sample(measurement, mask)``. The bare :class:`DDSSampler`
    crashed on both counts (``TypeError`` on the ``model=`` kwarg; a 7-argument
    ``sample`` signature) — the dead-knob-that-crashes this fixes (pitfall #15).

    **Model contract (DDS requires a score / eps-prediction diffusion model).**
    The model must expose:

    * an eps predictor — ``model._model_forward(x_t, t)`` (e.g.
      ``ScoreBasedDiffusionGenerator``) or a plain callable ``model(x_t, t)``; and
    * a variance schedule — ``model.alphas_cumprod`` or ``model.sqrt_alphas_cumprod``.

    A model lacking the contract (e.g. a cold-diffusion generator) raises here at
    sample time (fail-loud), and the ``dds_requires_score_model`` audit check
    rejects such an arm at config time — so the knob never silently degrades.

    Extra constructor kwargs from the dispatch call (``max_acceleration``,
    ``center_fraction``, ``dc_method``, ``dc_weight``) are accepted and ignored;
    the masked-FFT forward operator is derived from the supplied ``mask`` at
    sample time (single-coil; SENSE coil maps are a future extension).
    """

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        sampling_steps: int | None = None,
        cg_iters: int = 5,
        cg_lambda: float = 1e-2,
        eta: float = 0.0,
        **_ignored: object,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_timesteps = int(num_timesteps)
        self.n_steps = int(sampling_steps or num_timesteps)
        self.cg_iters = int(cg_iters)
        self.cg_lambda = float(cg_lambda)
        self.eta = float(eta)

    def _alphas_cumprod(self, device: torch.device) -> torch.Tensor:
        m = self.model
        if hasattr(m, "alphas_cumprod"):
            return m.alphas_cumprod.to(device)
        if hasattr(m, "sqrt_alphas_cumprod"):
            return (m.sqrt_alphas_cumprod**2).to(device)
        raise ValueError(
            "DDS requires a score/diffusion model exposing 'alphas_cumprod' or "
            f"'sqrt_alphas_cumprod'; {type(m).__name__} has neither. DDS is a "
            "score-model posterior sampler — see the dds_requires_score_model "
            "audit check."
        )

    def _score(self, x_t: torch.Tensor, t_idx: int, device: torch.device) -> torch.Tensor:
        t = torch.full((x_t.shape[0],), int(t_idx), device=device, dtype=torch.long)
        m = self.model
        if hasattr(m, "_model_forward"):
            return m._model_forward(x_t, t)
        if callable(m):
            return m(x_t, t)
        raise TypeError(f"DDS model {type(m).__name__} is not callable and lacks _model_forward.")

    @torch.no_grad()
    def sample(self, measurement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reverse-diffuse from undersampled k-space to an image estimate.

        Args:
            measurement: Acquired k-space ``y`` (complex or real), ``[B, C, H, W]``.
            mask: Sampling mask ``[B, 1, H, W]`` (broadcastable, 1 = sampled).

        Returns:
            The real-valued image reconstruction, ``[B, C, H, W]``.
        """
        device = measurement.device
        alphas_cumprod = self._alphas_cumprod(device)  # fail-loud on non-score model
        mask_c = mask.to(measurement.real.dtype if measurement.is_complex() else mask.dtype)

        def forward_op(x: torch.Tensor) -> torch.Tensor:
            return mask_c * fft2c(x)

        def adjoint_op(k: torch.Tensor) -> torch.Tensor:
            return ifft2c(mask_c * k).real

        shape = (measurement.shape[0], measurement.shape[1], *measurement.shape[-2:])
        x_t = torch.randn(shape, device=device)
        n_sched = alphas_cumprod.shape[0]
        # DDIM-style stride across the trained schedule.
        ts = torch.linspace(n_sched - 1, 0, self.n_steps, device=device).round().long()
        for i in range(self.n_steps):
            t = int(ts[i].item())
            ab = alphas_cumprod[t]
            eps_pred = self._score(x_t, t, device)
            x0_hat = _PosteriorSamplerBase._x0_from_eps(x_t, eps_pred, ab)
            # (b) CG data-consistency on x0 via the shared physics primitive.
            x0_dc = cg_data_consistency(
                x0_hat,
                forward_op,
                adjoint_op,
                measurement,
                n_iter=self.cg_iters,
                lam=self.cg_lambda,
            )
            t_prev = int(ts[i + 1].item()) if i + 1 < self.n_steps else 0
            ab_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.ones_like(ab)
            sigma = self.eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
            mean = ab_prev.sqrt() * x0_dc + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
            x_t = mean + sigma * noise
        return x_t


@register_sampler("dynamic_dps", aliases=["ddps", "dynamic_diffusion_posterior"])
class DynamicDPSSampler(nn.Module):
    r"""Dynamic Diffusion Posterior Sampling — three dynamic-guidance variants.

    A generator-dispatch (World-A) posterior sampler in the same decomposed form
    as :class:`DDSReconSampler` (denoise → guide :math:`\hat{x}_0` → DDIM step),
    but the data-consistency guidance is *dynamic* — it adapts per reverse step.
    ``guidance_mode`` selects the variant (validated; unknown raises, pitfall #9):

    * ``adaptive_step`` (Chung et al. 2023, DPS): residual-normalized gradient
      step :math:`\hat{x}_0 \mathrel{-}= (\zeta/\lVert r\rVert)\,A^{*}r`,
      :math:`r = A\hat{x}_0 - y`. The step shrinks as the residual grows.
    * ``pigdm`` (Song et al. 2023, ΠGDM): noise-level-weighted correction
      :math:`\hat{x}_0 \mathrel{+}= \zeta\,\bar{\alpha}_t\, A^{*}(y - A\hat{x}_0)`
      — guidance strengthens as denoising completes (:math:`\bar{\alpha}_t\to 1`).
    * ``ncchi`` (novel, low-field): ``adaptive_step`` down-weighted by the
      non-central-χ noise floor :math:`2L\sigma^2` so the guidance trusts the
      measurement less where it is noise-dominated. **Corner case:** as
      :math:`\sigma\to 0` the floor vanishes and ``ncchi`` reduces exactly to
      ``adaptive_step``. ``ncchi`` requires ``sigma`` (fail-closed, pitfall #15).
      This is a heuristic low-SNR guidance correction (the project's nc-χ thesis
      applied to the guidance step), not the full Bessel nc-χ posterior.

    Model contract identical to :class:`DDSReconSampler` (``_model_forward`` /
    callable + ``alphas_cumprod`` / ``sqrt_alphas_cumprod``); a non-score model
    fails loud (guarded at config time by ``dds_requires_score_model``).
    """

    _MODES = frozenset({"adaptive_step", "pigdm", "ncchi"})

    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        sampling_steps: int | None = None,
        guidance_mode: str = "adaptive_step",
        guidance_scale: float = 1.0,
        sigma: float | None = None,
        n_coils: int = 1,
        eta: float = 0.0,
        **_ignored: object,
    ) -> None:
        super().__init__()
        if guidance_mode not in self._MODES:
            raise ValueError(
                f"Unknown guidance_mode={guidance_mode!r}; "
                f"supported: {sorted(self._MODES)} (pitfall #9: no silent fallback)."
            )
        if guidance_mode == "ncchi" and sigma is None:
            raise ValueError(
                "guidance_mode='ncchi' requires sigma (the k-space noise σ); "
                "none was set (pitfall #15: the knob must be wired)."
            )
        self.model = model
        self.num_timesteps = int(num_timesteps)
        self.n_steps = int(sampling_steps or num_timesteps)
        self.guidance_mode = guidance_mode
        self.guidance_scale = float(guidance_scale)
        self.sigma = None if sigma is None else float(sigma)
        self.n_coils = int(n_coils)
        self.eta = float(eta)

    # Schedule + score resolution mirror DDSReconSampler's model contract.
    _alphas_cumprod = DDSReconSampler._alphas_cumprod
    _score = DDSReconSampler._score

    def _guide(
        self,
        x0: torch.Tensor,
        forward_op: Callable[[torch.Tensor], torch.Tensor],
        adjoint_op: Callable[[torch.Tensor], torch.Tensor],
        y: torch.Tensor,
        ab: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the mode-specific dynamic data-consistency guidance to ``x0``."""
        residual = forward_op(x0) - y
        if self.guidance_mode == "pigdm":
            # Noise-level-adaptive: weight grows with alpha_bar (late steps).
            weight = self.guidance_scale * ab
            return x0 - weight * adjoint_op(residual)
        # adaptive_step / ncchi share the residual-normalized step, reduced per sample
        # so one image's step does not depend on the rest of its batch.
        bshape = (-1, *([1] * (residual.dim() - 1)))
        r_norm = residual.abs().pow(2).flatten(1).sum(1).sqrt().view(bshape)
        step = self.guidance_scale / (r_norm + 1e-8)
        if self.guidance_mode == "ncchi":
            # Down-weight the step where the measurement is noise-floor-dominated;
            # at sigma=0 the factor is 1 (== adaptive_step).
            floor = 2.0 * self.n_coils * (self.sigma or 0.0) ** 2
            sig_energy = forward_op(x0).abs().pow(2).flatten(1).mean(1).view(bshape)
            snr_factor = (1.0 - floor / (sig_energy + 1e-8)).clamp(min=0.0, max=1.0)
            step = step * snr_factor
        return x0 - step * adjoint_op(residual)

    @torch.no_grad()
    def sample(self, measurement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reverse-diffuse from undersampled k-space with dynamic DC guidance."""
        device = measurement.device
        alphas_cumprod = self._alphas_cumprod(device)  # fail-loud on non-score model
        mask_c = mask.to(measurement.real.dtype if measurement.is_complex() else mask.dtype)

        def forward_op(x: torch.Tensor) -> torch.Tensor:
            return mask_c * fft2c(x)

        def adjoint_op(k: torch.Tensor) -> torch.Tensor:
            return ifft2c(mask_c * k).real

        shape = (measurement.shape[0], measurement.shape[1], *measurement.shape[-2:])
        x_t = torch.randn(shape, device=device)
        n_sched = alphas_cumprod.shape[0]
        ts = torch.linspace(n_sched - 1, 0, self.n_steps, device=device).round().long()
        for i in range(self.n_steps):
            t = int(ts[i].item())
            ab = alphas_cumprod[t]
            eps_pred = self._score(x_t, t, device)
            x0_hat = _PosteriorSamplerBase._x0_from_eps(x_t, eps_pred, ab)
            x0_dc = self._guide(x0_hat, forward_op, adjoint_op, measurement, ab)
            t_prev = int(ts[i + 1].item()) if i + 1 < self.n_steps else 0
            ab_prev = alphas_cumprod[t_prev] if t_prev > 0 else torch.ones_like(ab)
            sigma = self.eta * ((1 - ab_prev) / (1 - ab) * (1 - ab / ab_prev)).clamp(min=0).sqrt()
            mean = ab_prev.sqrt() * x0_dc + (1 - ab_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
            noise = torch.randn_like(x_t) if t_prev > 0 else torch.zeros_like(x_t)
            x_t = mean + sigma * noise
        return x_t


__all__ = [
    "DDSReconSampler",
    "DDSSampler",
    "DPSSampler",
    "DynamicDPSSampler",
    "PiGDMSampler",
]
