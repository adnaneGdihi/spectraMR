r"""RED-diff — re-noising-based regularisation by a frozen diffusion prior.

When a frozen diffusion model :math:`f_\phi` is available as an
implicit prior on natural images, RED-diff (Mardani *et al.* 2024)
defines an inverse-problem reconstruction loss as

.. math::
    \mathcal{L}_{\text{RED-diff}} = \mathbb{E}_t\!\left[w(t)\,\|f_\phi(\hat{\mathbf{x}}+\sigma_t\boldsymbol{\epsilon},t) - \hat{\mathbf{x}}\|^2\right],

i.e. the candidate reconstruction :math:`\hat{\mathbf{x}}` (either a
single test-time variable or a network output) should equal the
diffusion prior's denoiser of itself once Gaussian noise is added.
This is the *implicit-regularisation by re-noising* family of inverse
solvers — closely related to RED (Romano *et al.* 2017) and to the
Score-MRI / DDS family (Feng *et al.* 2023).

For training a feed-forward reconstructor with a diffusion teacher,
the same loss form regularises the student toward the prior's
denoised manifold, so it doubles as a distillation term.

Reference:
    [68] B. T. Feng *et al.*, "Score-based diffusion models as
        principled priors for inverse imaging," *Proc. IEEE/CVF ICCV*,
        2023. (Re-noising / DDS / RED-diff family.)

    Mardani *et al.*, "A variational perspective on solving inverse
    problems with diffusion models," *ICLR*, 2024.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


def _sample_t(
    batch: int, t_min: float, t_max: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return t_min + (t_max - t_min) * torch.rand(batch, device=device, dtype=dtype)


@register_loss(
    name="red_diff",
    aliases=["REDDiffLoss", "renoising_reg", "red_diffusion"],
    domain="image",
)
class REDDiffLoss(nn.Module):
    r"""Re-noising regularisation via a frozen diffusion denoiser.

    Forward expects:

    * ``prediction``: candidate reconstruction :math:`\hat{\mathbf{x}}`.
    * ``target``: ignored (the regulariser has no clean target — the
      prior **is** the target).
    * ``context["denoiser"]``: callable
      ``denoiser(noisy, t) -> denoised`` (a frozen diffusion network in
      eval mode). The strategy is responsible for keeping it in
      ``no_grad`` and on the right device.
    * ``context["sigma_schedule"]`` (optional): callable
      ``t -> sigma(t)``. Defaults to a linear schedule
      :math:`\sigma(t) = t \cdot \sigma_{\max}`.

    Args:
        n_levels: Number of (independently sampled) noise levels per
            forward call. Default 1.
        sigma_max: Upper end of the per-step noise scale. Default 0.5.
        t_min, t_max: Range of :math:`t` to sample uniformly.
        weight_fn: One of ``"uniform"`` (default) or
            ``"snr"``. ``"snr"`` weights samples by
            :math:`1/(1+\sigma^2)` to emphasise low-noise
            (high-information) levels — the original RED-diff
            recommendation.
    """

    def __init__(
        self,
        n_levels: int = 1,
        sigma_max: float = 0.5,
        t_min: float = 1e-3,
        t_max: float = 1.0,
        weight_fn: str = "uniform",
    ) -> None:
        super().__init__()
        if n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {n_levels}")
        if not (sigma_max > 0.0):
            raise ValueError(f"sigma_max must be > 0, got {sigma_max}")
        if not (0.0 <= t_min < t_max <= 1.0):
            raise ValueError(f"need 0 <= t_min < t_max <= 1, got {t_min}, {t_max}")
        if weight_fn not in ("uniform", "snr"):
            raise ValueError(f"weight_fn must be uniform or snr, got {weight_fn!r}")
        self.n_levels = n_levels
        self.sigma_max = sigma_max
        self.t_min, self.t_max = t_min, t_max
        self.weight_fn = weight_fn

    def _sigma(self, t: torch.Tensor) -> torch.Tensor:
        return t * self.sigma_max  # default linear schedule

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None or "denoiser" not in context:
            raise ValueError(
                "REDDiffLoss requires context['denoiser'] (a frozen diffusion denoiser callable)."
            )
        denoiser = context["denoiser"]
        if not callable(denoiser):
            raise TypeError("context['denoiser'] must be callable.")
        custom_sigma = context.get("sigma_schedule")

        loss = prediction.new_zeros(())
        for _ in range(self.n_levels):
            t = _sample_t(
                prediction.shape[0],
                self.t_min,
                self.t_max,
                prediction.device,
                prediction.dtype,
            )
            sigma = custom_sigma(t) if custom_sigma is not None else self._sigma(t)
            sigma_b = sigma.view(-1, *([1] * (prediction.dim() - 1)))
            eps = torch.randn_like(prediction)
            noisy = prediction + sigma_b * eps
            with torch.no_grad():
                # The denoiser is frozen — gradients flow back to
                # `prediction` only through the residual on the
                # left-hand side.
                denoised = denoiser(noisy, t)
            if denoised.shape != prediction.shape:
                raise ValueError(
                    f"denoiser output shape {tuple(denoised.shape)} != "
                    f"prediction shape {tuple(prediction.shape)}"
                )
            residual = denoised.detach() - prediction
            if self.weight_fn == "snr":
                w = 1.0 / (1.0 + sigma * sigma)
                w_b = w.view(-1, *([1] * (prediction.dim() - 1)))
                loss = loss + (w_b * residual.pow(2)).mean()
            else:
                loss = loss + F.mse_loss(prediction, denoised.detach())
        return loss / self.n_levels
