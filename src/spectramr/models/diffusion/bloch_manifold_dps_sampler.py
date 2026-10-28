r"""Bloch-Manifold-Constrained Diffusion Posterior Sampling (B-2 sampler).

Idea 2 of ``TODO/VF_CAMPAIGN_IMPLEMENTATION_PLAN.md``. A DPS reverse
sampler that, at every reverse step, interleaves

1. the **measurement likelihood gradient** (Chung et al. 2023, DPS):

   .. math::

       \nabla_{x_t}\log p(y\mid x_t)\;\approx\;
       -\,\zeta\,\nabla_{x_t}\tfrac12\lVert A\,\hat x_0(x_t)-y\rVert_2^2,

   with :math:`\hat x_0` the Tweedie posterior-mean estimate and
   :math:`A` the (SENSE or single-coil) forward operator from
   ``infrastructure.physics``, and

2. the **Bloch-manifold projection** :math:`P_{\mathcal M}` from
   :class:`~spectramr.models.physics.bloch_manifold_projector.BlochManifoldProjector`,
   which snaps the Tweedie estimate onto the manifold of physically
   realisable signals for the configured pulse sequence.

Training of the underlying score network is *unchanged* from standard
DDPM (ε-prediction on clean samples); the novelty lives entirely in this
sampler. At :math:`\zeta=0` and with the projector disabled the sampler
degenerates to plain DDIM/DDPM.

The score network is injected (composition, not inheritance) so the
sampler is agnostic to the concrete denoiser. The module registers under
``training_mode="diffusion"`` so it can be resolved through the
registry-dispatcher and composed by the strategy.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import (
    fft2c,
    ifft2c,
    sense_adjoint,
    sense_forward,
)
from spectramr.models.physics.bloch_manifold_projector import BlochManifoldProjector
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

_EPS = 1e-8


#: Ho et al. (2020) define the linear β range over a 1000-step chain; Nichol &
#: Dhariwal rescale it by ``_REFERENCE_STEPS / T`` so a T-step chain reaches the same noise.
_REFERENCE_STEPS = 1000
_REFERENCE_BETA_RANGE = (1e-4, 2e-2)
_BETA_BOUNDS = (1e-4, 0.999)


def rescaled_linear_betas(timesteps: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """Linear β schedule over ``timesteps`` steps, rescaled from the 1000-step reference.

    The clamp keeps ``1 - β`` positive: unclamped, ``β_end = 20 / T`` reaches 1 at
    ``T <= 20`` and every ``1 / sqrt(ᾱ)`` becomes inf (#2441).
    """
    if timesteps < 1:
        raise ValueError(f"timesteps must be >= 1, got {timesteps}.")
    scale = _REFERENCE_STEPS / timesteps
    start, end = (scale * beta for beta in _REFERENCE_BETA_RANGE)
    betas = torch.linspace(start, end, timesteps, device=device, dtype=torch.float32)
    return betas.clamp(*_BETA_BOUNDS)


def check_reverse_stride(sampling_steps: int, num_timesteps: int) -> None:
    """Raise unless the reverse loop's step count strides the trained schedule."""
    if not 1 <= sampling_steps <= num_timesteps:
        raise ValueError(
            f"sampling_steps={sampling_steps} must lie in [1, num_timesteps="
            f"{num_timesteps}]: the reverse loop strides the trained schedule."
        )


@register_model(
    name="bloch_manifold_dps_sampler",
    training_mode="diffusion",
    spatial_dims=(2, 3),
    input_domain="kspace",
    output_domain="image",
    accepts_complex=True,
)
class BlochManifoldDPSSampler(nn.Module):
    r"""DPS sampler interleaving a likelihood gradient with manifold projection.

    Args:
        score_network: The denoiser :math:`s_\phi(x_t, t)`. Called as
            ``score_network(x, timesteps=t)`` (falling back to
            ``score_network(x, t)`` then ``score_network(x)``). May predict
            either noise (``prediction_type='epsilon'``) or the clean image
            (``prediction_type='x0'``).
        projector: A :class:`BlochManifoldProjector` (or any module with the
            same ``forward`` / ``residual`` contract). If ``None``, one is
            built from ``projector_kwargs``.
        sampling_steps: Number of reverse diffusion steps, strided over the
            ``num_timesteps`` schedule.
        num_timesteps: Horizon the score network was trained on; the ᾱ schedule
            and the step indices fed to it span this. Defaults to ``sampling_steps``.
        sampler_type: ``"ddim"`` (deterministic) or ``"ddpm"`` (stochastic).
        likelihood_weight_zeta: DPS guidance scale :math:`\zeta \ge 0`.
        prediction_type: ``"epsilon"`` or ``"x0"``.
        projection_every: Apply the manifold projection every ``k`` reverse
            steps (1 = every step).
        projector_kwargs: Forwarded to :class:`BlochManifoldProjector` when
            ``projector`` is ``None``.

    The forward operator is SENSE when ``coil_sensitivities`` is supplied to
    :meth:`sample`, single-coil centred FFT otherwise — both via the physics
    SSOT (``fft_ops``), never a raw ``torch.fft``.
    """

    def __init__(
        self,
        score_network: nn.Module,
        projector: BlochManifoldProjector | None = None,
        forward_operator: str = "sense",
        sampling_steps: int = 1000,
        sampler_type: str = "ddim",
        likelihood_weight_zeta: float = 1.0,
        prediction_type: str = "epsilon",
        projection_every: int = 1,
        projector_kwargs: dict | None = None,
        num_timesteps: int | None = None,
    ) -> None:
        super().__init__()
        if sampler_type not in ("ddim", "ddpm"):
            raise ValueError(f"sampler_type must be 'ddim' or 'ddpm', got {sampler_type!r}.")
        if prediction_type not in ("epsilon", "x0"):
            raise ValueError(f"prediction_type must be 'epsilon' or 'x0', got {prediction_type!r}.")
        if likelihood_weight_zeta < 0:
            raise ValueError("likelihood_weight_zeta must be >= 0.")
        self.score_network = score_network
        self.projector = projector or BlochManifoldProjector(**(projector_kwargs or {}))
        self.forward_operator = str(forward_operator).lower()
        self.sampling_steps = int(sampling_steps)
        self.num_timesteps = int(sampling_steps if num_timesteps is None else num_timesteps)
        check_reverse_stride(self.sampling_steps, self.num_timesteps)
        self.sampler_type = sampler_type
        self.zeta = float(likelihood_weight_zeta)
        self.prediction_type = prediction_type
        self.projection_every = max(int(projection_every), 1)

        betas = rescaled_linear_betas(self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)

    # ------------------------------------------------------------------ #
    # Forward model A and adjoint A^H (physics SSOT)
    # ------------------------------------------------------------------ #
    def _forward_op(
        self,
        x: torch.Tensor,
        smaps: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """A x: image -> (masked) k-space."""
        if self.forward_operator == "sense" and smaps is not None:
            return sense_forward(x, smaps=smaps, mask=mask)
        kspace = fft2c(x if torch.is_complex(x) else x.to(torch.complex64))
        if mask is not None:
            while mask.dim() < kspace.dim():
                mask = mask.unsqueeze(1)
            kspace = kspace * mask.to(kspace.real.dtype)
        return kspace

    def _adjoint_op(
        self,
        k: torch.Tensor,
        smaps: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """A^H k: k-space -> image."""
        if self.forward_operator == "sense" and smaps is not None:
            return sense_adjoint(k, smaps=smaps, mask=mask)
        if mask is not None:
            while mask.dim() < k.dim():
                mask = mask.unsqueeze(1)
            k = k * mask.to(k.real.dtype)
        return ifft2c(k)

    # ------------------------------------------------------------------ #
    # Tweedie posterior-mean estimate
    # ------------------------------------------------------------------ #
    def _call_score(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Invoke the score network, tolerating common call signatures."""
        try:
            return self.score_network(x, timesteps=t)
        except TypeError:
            pass
        try:
            return self.score_network(x, t)
        except TypeError:
            return self.score_network(x)

    def tweedie_x0(self, x_t: torch.Tensor, t: torch.Tensor, t_idx: int) -> torch.Tensor:
        r"""Tweedie posterior-mean :math:`\hat x_0(x_t)`.

        For ε-prediction:
        :math:`\hat x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon)/\sqrt{\bar\alpha_t}`.
        For x0-prediction the network output is returned directly.
        """
        out = self._call_score(x_t, t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get("image", out.get("sample", next(iter(out.values()))))
        if self.prediction_type == "x0":
            return out
        a_bar = self.alphas_cumprod[t_idx].to(x_t.real.dtype)
        sqrt_ab = torch.sqrt(a_bar.clamp_min(_EPS))
        sqrt_1mab = torch.sqrt((1.0 - a_bar).clamp_min(0.0))
        return (x_t - sqrt_1mab * out) / sqrt_ab.clamp_min(_EPS)

    # ------------------------------------------------------------------ #
    # The novel guidance + projection step
    # ------------------------------------------------------------------ #
    def likelihood_gradient(
        self,
        x_t: torch.Tensor,
        x0_fn: Callable[[torch.Tensor], torch.Tensor],
        measured_kspace: torch.Tensor,
        smaps: torch.Tensor | None,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        r"""DPS gradient :math:`-\zeta\,\nabla_{x_t}\tfrac12\lVert A\hat x_0-y\rVert^2`.

        Computed with autograd through the Tweedie map ``x0_fn``. Returns a
        tensor the same shape/dtype as ``x_t``.
        """
        if self.zeta == 0.0:
            return torch.zeros_like(x_t)
        # sample() runs under @torch.no_grad(); re-enable grad locally so the
        # DPS gradient through the Tweedie map can be taken.
        with torch.enable_grad():
            x_leaf = x_t.detach().clone().requires_grad_(True)
            x0 = x0_fn(x_leaf)
            residual = self._forward_op(x0, smaps, mask) - measured_kspace
            # Mean (not sum) keeps the guidance scale image-size-invariant so
            # zeta transfers across resolutions and stays numerically stable.
            loss = 0.5 * (residual.abs() ** 2).mean()
            (grad,) = torch.autograd.grad(loss, x_leaf, create_graph=False)
        return -self.zeta * grad.detach()

    def project(self, x0: torch.Tensor) -> torch.Tensor:
        """Project a Tweedie estimate onto the Bloch manifold."""
        return self.projector(x0)

    def manifold_residual(self, x0: torch.Tensor) -> torch.Tensor:
        """Scalar mean per-voxel manifold residual of a Tweedie estimate."""
        return self.projector.residual(x0).mean()

    # ------------------------------------------------------------------ #
    # Reverse sampling loop
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        measured_kspace: torch.Tensor,
        shape: tuple[int, ...] | None = None,
        coil_sensitivities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        return_trace: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        r"""Run Bloch-manifold-constrained DPS reverse sampling.

        Args:
            measured_kspace: Acquired (complex) k-space :math:`y`.
            shape: Image shape ``[B, C, H, W]``. Inferred from the adjoint
                of ``measured_kspace`` when ``None``.
            coil_sensitivities: Complex SENSE maps ``[B, Ncoil, H, W]``.
            mask: Sampling mask ``[B, 1, H, W]``.
            num_steps: Override the reverse-step count (<= ``sampling_steps``).
            generator: RNG for reproducible DDPM noise / initial latent.
            return_trace: If True, also return the per-step manifold residual.

        Returns:
            Complex reconstruction ``[B, C, H, W]`` (and, if requested, the
            list of per-step manifold residuals — non-increasing on a
            consistent prior).
        """
        device = measured_kspace.device
        if shape is None:
            x_adj = self._adjoint_op(measured_kspace, coil_sensitivities, mask)
            shape = tuple(x_adj.shape)
        else:
            x_adj = None

        x_t = torch.randn(*shape, device=device, dtype=torch.float32, generator=generator).to(
            torch.complex64
        )

        n = self.sampling_steps if num_steps is None else min(num_steps, self.sampling_steps)
        # Evenly spaced timestep indices, descending.
        step_idx = torch.linspace(self.num_timesteps - 1, 0, n, device=device).round().long()

        trace: list[float] = []
        for i, t_idx_t in enumerate(step_idx):
            t_idx = int(t_idx_t.item())
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)

            def _x0_fn(z: torch.Tensor, _ti: int = t_idx) -> torch.Tensor:
                tt = torch.full((z.shape[0],), _ti, device=z.device, dtype=torch.long)
                with torch.enable_grad():
                    return self.tweedie_x0(z, tt, _ti)

            # 1. Likelihood gradient (DPS) on x_t.
            grad = self.likelihood_gradient(x_t, _x0_fn, measured_kspace, coil_sensitivities, mask)

            # 2. Tweedie estimate of the clean image.
            x0_hat = self.tweedie_x0(x_t, t, t_idx)

            # 3. Manifold projection (the novelty) — snap to physical signals.
            if (i % self.projection_every) == 0:
                if return_trace:
                    trace.append(float(self.projector.residual(x0_hat).mean().item()))
                x0_hat = self.project(x0_hat)

            # 4. Reverse-step update toward x_{t-1} using the projected x0.
            x_t = self._reverse_step(x_t, x0_hat, t_idx, step_idx, i, grad, generator)

        x_final = x_t
        if return_trace:
            return x_final, trace
        return x_final

    def _reverse_step(
        self,
        x_t: torch.Tensor,
        x0_hat: torch.Tensor,
        t_idx: int,
        step_idx: torch.Tensor,
        i: int,
        likelihood_grad: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """One DDIM / DDPM reverse step from ``x_t`` toward ``x_{t_prev}``."""
        a_bar_t = self.alphas_cumprod[t_idx].to(x_t.real.dtype)
        if i + 1 < len(step_idx):
            t_prev = int(step_idx[i + 1].item())
            a_bar_prev = self.alphas_cumprod[t_prev].to(x_t.real.dtype)
        else:
            a_bar_prev = torch.ones_like(a_bar_t)

        sqrt_ab_t = torch.sqrt(a_bar_t.clamp_min(_EPS))
        sqrt_1mab_t = torch.sqrt((1.0 - a_bar_t).clamp_min(0.0))
        # Recover the implied noise from the (projected) x0 and current x_t.
        eps = (x_t - sqrt_ab_t * x0_hat) / sqrt_1mab_t.clamp_min(_EPS)

        if self.sampler_type == "ddim":
            x_prev = (
                torch.sqrt(a_bar_prev.clamp_min(_EPS)) * x0_hat
                + torch.sqrt((1.0 - a_bar_prev).clamp_min(0.0)) * eps
            )
        else:  # ddpm
            noise = torch.randn(
                *x_t.shape, device=x_t.device, dtype=torch.float32, generator=generator
            ).to(torch.complex64)
            x_prev = (
                torch.sqrt(a_bar_prev.clamp_min(_EPS)) * x0_hat
                + torch.sqrt((1.0 - a_bar_prev).clamp_min(0.0)) * eps
            )
            if t_idx > 0:
                sigma = torch.sqrt(
                    ((1.0 - a_bar_prev) / (1.0 - a_bar_t).clamp_min(_EPS))
                    * (1.0 - a_bar_t / a_bar_prev.clamp_min(_EPS))
                ).clamp_min(0.0)
                x_prev = x_prev + sigma * noise

        # Fold in the DPS likelihood gradient (already negated & scaled).
        return x_prev + likelihood_grad

    def forward(
        self,
        measured_kspace: torch.Tensor,
        coil_sensitivities: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        """Alias for :meth:`sample` so the module is callable like a model."""
        return self.sample(
            measured_kspace,
            coil_sensitivities=coil_sensitivities,
            mask=mask,
            num_steps=num_steps,
        )


__all__ = ["BlochManifoldDPSSampler"]
