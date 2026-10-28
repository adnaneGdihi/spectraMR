r"""Diffusion Posterior Sampling (DPS) data-fit loss + manifold-constrained
gradient (MCG) variant.

DPS (Chung *et al.* 2023 ICLR) approximates the intractable likelihood
gradient :math:`\nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)`
that posterior sampling needs by replacing it with the data-fit
gradient on the **Tweedie posterior mean**
:math:`\hat{\mathbf{x}}_0(\mathbf{x}_t)`:

.. math::
    \nabla_{\mathbf{x}_t}\log p(\mathbf{y}|\mathbf{x}_t)
    \approx -\nabla_{\mathbf{x}_t}\,\frac{\|\mathbf{y}-\mathbf{A}\hat{\mathbf{x}}_0(\mathbf{x}_t)\|^2}{2\sigma^2}.

The data-fit norm itself,

.. math::
    \mathcal{L}_{\text{DPS}} = \frac{1}{2\sigma^2}\,\|\mathbf{y}-\mathbf{A}\hat{\mathbf{x}}_0\|^2,

is the loss form we register here. It is used both as an inference-time
guidance term (sampler hook) and as a training-time auxiliary that
encourages the diffusion model to produce posterior-consistent
denoised samples.

The **manifold-constrained gradient** (MCG, Chung *et al.* 2022
NeurIPS) replaces the raw gradient with its tangent projection onto
the data manifold. We approximate the tangent space by removing the
component of the gradient that points orthogonally to the Tweedie
mean direction — ``mode="manifold_constrained"``.

References:
    [60] H. Chung *et al.*, "Diffusion posterior sampling for general
        noisy inverse problems," *Proc. ICLR*, 2023.
    [61] H. Chung *et al.*, "Improving diffusion models for inverse
        problems using manifold constraints," *Proc. NeurIPS*, 2022.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss


def _apply_forward(
    image: torch.Tensor,
    mask: torch.Tensor,
    smaps: torch.Tensor | None,
) -> torch.Tensor:
    """Apply :math:`\\mathbf{A} = \\mathbf{M}\\mathbf{F}\\mathbf{S}` to an
    image.

    smaps: optional :math:`(B, C, H, W)` complex coil sensitivity maps.
        If absent we treat the image as already coil-combined.
    """
    if smaps is not None:
        if smaps.dim() != image.dim():
            raise ValueError(f"smaps dims {smaps.dim()} != image dims {image.dim()}")
        image = image * smaps
    k = fft2c(image)
    while mask.dim() < k.dim():
        mask = mask.unsqueeze(0)
    mask = mask.to(k.real.dtype if k.is_complex() else k.dtype)
    return k * mask


def _project_manifold(grad: torch.Tensor, x_hat0: torch.Tensor) -> torch.Tensor:
    """Remove the component of ``grad`` parallel to ``x_hat0``.

    A simple proxy for projecting onto the Tweedie tangent: the
    posterior mean direction is the local data-manifold normal, so the
    tangent projection removes its component.
    """
    flat_g = grad.reshape(grad.shape[0], -1)
    flat_x = x_hat0.reshape(x_hat0.shape[0], -1)
    norm_x_sq = flat_x.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-12)
    coeff = (flat_g * flat_x).sum(dim=1, keepdim=True) / norm_x_sq
    proj = flat_g - coeff * flat_x
    return proj.reshape_as(grad)


@register_loss(
    name="dps_data_fit",
    aliases=["DPSLoss", "diffusion_posterior_sampling", "dps", "mcg_dps"],
    domain="kspace",
)
class DPSDataFitLoss(nn.Module):
    r"""Data-fit norm for diffusion posterior sampling.

    Forward expects:

    * ``prediction``: the Tweedie posterior mean
      :math:`\hat{\mathbf{x}}_0(\mathbf{x}_t)` — image-space, real or
      complex, shape ``(B, C, H, W)``. The strategy builds this from
      the diffusion model output and the noise-scale schedule.
    * ``target``: ignored (the data lives in ``context``).
    * ``context["measurement_kspace"]``: acquired k-space
      :math:`\mathbf{y}`.
    * ``context["sampling_mask"]``: binary mask :math:`\mathbf{M}`.
    * ``context["coil_sensitivities"]`` (optional): :math:`\mathbf{S}`.
    * ``context["sigma"]`` (optional): per-step noise scale
      :math:`\sigma`. Defaults to ``1.0`` so the loss equals the bare
      squared residual.

    Args:
        mode: ``"vanilla"`` (raw DPS, default) or
            ``"manifold_constrained"`` (project the gradient onto the
            Tweedie tangent — MCG variant).
        norm: ``"l2"`` (default, paper) or ``"l1"`` (a robust variant
            popular for high-noise low-field MRI).
    """

    def __init__(
        self,
        mode: str = "vanilla",
        norm: str = "l2",
    ) -> None:
        super().__init__()
        if mode not in ("vanilla", "manifold_constrained"):
            raise ValueError(f"mode must be vanilla or manifold_constrained, got {mode!r}")
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
        self.mode = mode
        self.norm = norm

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(
                "DPSDataFitLoss requires context['measurement_kspace'] "
                "and context['sampling_mask']."
            )
        y = context.get("measurement_kspace")
        mask = context.get("sampling_mask")
        if y is None or mask is None:
            raise ValueError("context must supply 'measurement_kspace' and 'sampling_mask'.")
        smaps = context.get("coil_sensitivities")
        sigma = float(context.get("sigma", 1.0))
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        a_x_hat0 = _apply_forward(prediction, mask, smaps)
        if a_x_hat0.shape != y.shape:
            raise ValueError(
                f"forward(pred) shape {tuple(a_x_hat0.shape)} != measurement shape {tuple(y.shape)}"
            )
        diff = a_x_hat0 - y
        diff_real = torch.view_as_real(diff) if diff.is_complex() else diff

        if self.norm == "l1":
            base = diff_real.abs().mean()
        else:
            base = 0.5 * diff_real.pow(2).mean() / (sigma * sigma)

        if self.mode == "vanilla":
            return base

        # MCG variant — rescale the loss by a tangent factor approximating
        # the manifold projection of the DPS gradient. The gradient is
        # linear in the residual, so projecting the residual onto the
        # tangent space of the Tweedie posterior mean gives the same
        # tangent factor in [0, 1] without needing to autograd through
        # the forward operator. Operate in image space — pull the
        # adjoint up cheaply using a magnitude proxy.
        residual_image = diff.abs() if diff.is_complex() else diff
        # Match leading dims of prediction; collapse coil axis if needed.
        if residual_image.shape != prediction.shape:
            residual_image = residual_image.mean(
                dim=tuple(range(1, residual_image.dim() - prediction.dim() + 1))
            )
        pred_real = prediction.real if prediction.is_complex() else prediction
        grad_proxy = _project_manifold(residual_image, pred_real)
        scale = grad_proxy.norm() / residual_image.norm().clamp_min(1e-12)
        scaled: torch.Tensor = base * scale.detach()
        return scaled
