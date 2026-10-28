r"""Test-Time Adaptation (TTA) consistency loss for accelerated MRI.

At inference time we may not have ground truth :math:`\mathbf{x}`, but
we always have the measurements :math:`\mathbf{y}` and the forward
operator :math:`\mathbf{A}`. The TTA consistency loss simply enforces

.. math::
    \mathcal{L}_{\text{TTA}} = \big\|\,\mathbf{A}\,f_{\boldsymbol{\theta}'}(\mathbf{y}) - \mathbf{y}\,\big\|_2^2

where :math:`f_{\boldsymbol{\theta}'}` is a copy of the trained model
whose parameters (typically the BatchNorm/GroupNorm affine ones) are
updated for a few iterations on the unlabelled measurement. This is
the MRI specialisation of TENT (Wang *et al.* 2021): an entropy
objective is replaced by a measurement-consistency objective that
applies cleanly to regression tasks.

The forward operator is applied via the existing FFT centring helpers
to avoid raw FFT calls.

Reference:
    D. Wang *et al.*, "Tent: Fully test-time adaptation by entropy
    minimization," *ICLR*, 2021. (Adapted to regression / MRI:
    Darestani *et al.*, "Test-time training can close the natural
    distribution shift performance gap in deep learning based
    compressed sensing," ICML 2022.)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss


@register_loss(
    name="tta_consistency",
    aliases=["TestTimeConsistencyLoss", "tta", "tent_mri"],
    domain="kspace",
)
class TestTimeConsistencyLoss(nn.Module):
    r"""Forward-operator residual at test time.

    Forward expects:

    * ``prediction``: image-space output of the model on the
      undersampled measurement.
    * ``target``: ignored.
    * ``context["measurement_kspace"]``: the *acquired* k-space
      :math:`\mathbf{y}`, typically multi-coil :math:`(B, C, H, W)`
      complex.
    * ``context["sampling_mask"]``: binary mask :math:`\mathbf{M}`
      indicating which k-space lines were acquired (broadcastable to
      ``measurement_kspace``).

    Args:
        norm: ``"l2"`` (default, paper) or ``"l1"``.
        coil_sensitivities: Optional ``(B, C, H, W)`` complex tensor.
            If supplied, the forward op becomes :math:`\mathbf{M F S}`;
            otherwise plain :math:`\mathbf{M F}`. Most strategies will
            instead pass the post-SENSE prediction and skip this arg.
    """

    def __init__(self, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be l1 or l2, got {norm!r}")
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
                "TestTimeConsistencyLoss requires "
                "context['measurement_kspace'] and context['sampling_mask']."
            )
        y = context.get("measurement_kspace")
        mask = context.get("sampling_mask")
        if y is None or mask is None:
            raise ValueError("context must supply 'measurement_kspace' and 'sampling_mask'.")
        smaps = context.get("coil_sensitivities")
        x_pred = prediction
        if smaps is not None:
            if smaps.shape != y.shape:
                raise ValueError(
                    f"coil_sensitivities shape {tuple(smaps.shape)} != "
                    f"measurement_kspace shape {tuple(y.shape)}"
                )
            x_pred = x_pred * smaps
        pred_k = fft2c(x_pred)
        if pred_k.shape != y.shape:
            raise ValueError(
                f"forward(pred) shape {tuple(pred_k.shape)} != measurement shape {tuple(y.shape)}"
            )
        while mask.dim() < y.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(y.real.dtype if y.is_complex() else y.dtype)
        diff: torch.Tensor = (pred_k - y) * mask
        if diff.is_complex():
            diff = torch.view_as_real(diff)
        if self.norm == "l1":
            return diff.abs().mean()
        return diff.pow(2).mean()
