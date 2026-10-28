r"""Multi-field data-consistency loss for DL-BAE (M4).

The reconstruction half of the dispersion-latent Bloch autoencoder objective:
the *single* field-invariant latent must explain **every** observed field
simultaneously,

.. math::

   \mathcal L_{\mathrm{DC}}
     = \frac1M\sum_{m=1}^{M} w_m\,
       \bigl\| \hat x(B_0^{(m)}) - x(B_0^{(m)}) \bigr\|_1 ,

which is what forces the latent to encode the dispersion *law* rather than a
per-field lookup. Fitting each field independently would trivially minimise a
per-field loss while learning nothing transferable, so the shared latent is the
whole point of summing here instead of training M models.

Optional ``field_weights`` compensate the SNR imbalance across fields: a 0.05 T
acquisition is far noisier than 3 T, and unweighted L1 lets the high-field terms
dominate the gradient.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

__all__ = ["MultiFieldDataConsistency"]


@register_loss(name="multifield_data_consistency", domain="image")
class MultiFieldDataConsistency(nn.Module):
    r"""Field-summed reconstruction fidelity for a shared dispersion latent.

    Args:
        field_weights: Optional per-field weights :math:`w_m`, length :math:`M`.
            ``None`` weights all fields equally.
        reduction_p: ``1`` for L1 (default, robust to low-field outliers) or
            ``2`` for squared error.
    """

    def __init__(
        self,
        field_weights: tuple[float, ...] | None = None,
        reduction_p: int = 1,
    ) -> None:
        super().__init__()
        if reduction_p not in (1, 2):
            raise ValueError(f"reduction_p must be 1 or 2; got {reduction_p}.")
        self.reduction_p = reduction_p
        if field_weights is None:
            self.field_weights: torch.Tensor | None = None
        else:
            if any(w < 0 for w in field_weights):
                raise ValueError(f"field_weights must be non-negative; got {field_weights!r}.")
            if sum(field_weights) <= 0:
                raise ValueError(
                    f"field_weights must not all be zero; got {field_weights!r}. The "
                    "weighted mean would read 0 and train nothing."
                )
            self.register_buffer("field_weights", torch.tensor(field_weights, dtype=torch.float32))

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the field-summed fidelity.

        Args:
            prediction: Rendered multi-field stack ``[B, M, H, W]``.
            target: Observed multi-field stack ``[B, M, H, W]``.

        Returns:
            Scalar loss.

        Raises:
            ValueError: on a shape mismatch, or when ``field_weights`` does not
                match the field count :math:`M`.
        """
        if prediction.shape != target.shape:
            raise ValueError(
                "multifield_data_consistency needs matching shapes [B, M, H, W]; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        if prediction.ndim != 4:
            raise ValueError(
                "multifield_data_consistency expects a 4-D multi-field stack "
                f"[B, M, H, W]; got {tuple(prediction.shape)}."
            )
        residual = prediction - target
        per_field = (residual.abs() if self.reduction_p == 1 else residual.pow(2)).mean(
            dim=(0, 2, 3)
        )  # [M]

        weights = self.field_weights
        if weights is None:
            return per_field.mean()
        if weights.numel() != per_field.numel():
            raise ValueError(
                f"field_weights has {weights.numel()} entries but the stack carries "
                f"{per_field.numel()} fields."
            )
        w = weights.to(device=per_field.device, dtype=per_field.dtype)
        return (w * per_field).sum() / w.sum().clamp_min(1e-12)
