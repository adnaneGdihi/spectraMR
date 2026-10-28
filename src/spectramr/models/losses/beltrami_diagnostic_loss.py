"""Beltrami coefficient diagnostic + (optional) regularizer.

The input is a displacement :math:`d = u + i v`, and the map it induces is
:math:`f(z) = z + d(z)`. Its Beltrami coefficient is

.. math::

    \\mu(f)
        = \\frac{\\partial f / \\partial \\bar z}{\\partial f / \\partial z}
        = \\frac{(u_x - v_y) + i (v_x + u_y)}
                {(2 + u_x + v_y) + i (v_x - u_y)},

so a zero displacement (the identity map) has :math:`\\mu = 0`, matching
:func:`spectramr.infrastructure.physics.beltrami_faithfulness.beltrami_from_displacement`.

The map is quasiconformal iff :math:`\\| \\mu \\|_\\infty < 1`. The
loss returns :math:`\\| \\mu \\|_\\infty` (or its squared mean) as a
differentiable diagnostic.

Per plan §12.2 — *"reframe QC as a diagnostic, not a guarantee"* —
the v0 YAML attaches this with ``weight = 0`` so it is logged only.
P2 promotes it to a soft regularizer once the training trajectory
is stable.

Input convention:

- 2D: ``(B, 2, H, W)`` interpreted as the displacement channels ``u`` (along
  W, x) and ``v`` (along H, y), in pixels.
- 3D: ``(B, 2 or 3, D, H, W)`` — Beltrami is computed per axial slice on the
  in-plane components ``(u, v)`` = channels 0 and 1, the standard MRI-distortion
  proxy. A third (z) component is ignored by this diagnostic.

The GeoMamba displacement head supplies the 3D field through ``displacement=``.
The fallback ``forward(pred, target)`` uses the residual ``pred - target``, which
must have 2 or 3 channels. A 1-channel intensity residual raises, because it is
not a displacement.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="beltrami_diagnostic",
    aliases=["BeltramiDiagnostic", "beltrami"],
)
class BeltramiDiagnosticLoss(nn.Module):
    """Differentiable Beltrami coefficient ``|mu|`` of the map ``z + d`` for a displacement ``d``.

    Args:
        reduction: ``"infty"`` returns the per-batch maximum
            ``||mu||_inf``; ``"mean"`` returns the per-pixel mean
            ``|mu|``; ``"penalty"`` returns the soft penalty
            ``relu(|mu| - 1).pow(2).mean()`` which is zero exactly
            when the map is quasiconformal.
        eps: stabilizer for the denominator.
    """

    def __init__(
        self,
        reduction: Literal["infty", "mean", "penalty"] = "mean",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if reduction not in {"infty", "mean", "penalty"}:
            raise ValueError(
                f"reduction must be one of {{infty, mean, penalty}}, got {reduction!r}"
            )
        self.reduction = reduction
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        displacement: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """Compute the Beltrami diagnostic.

        Pass either an explicit ``displacement`` tensor (preferred),
        or rely on the standard ``(pred, target)`` convention which
        uses the residual ``pred - target`` as the displacement
        field — only valid when both are 2-channel (u, v).
        """
        field = self._resolve_field(pred, target, displacement)
        if field.ndim == 4:
            if field.shape[1] != 2:
                raise ValueError(f"2D Beltrami needs (B,2,H,W), got {tuple(field.shape)}")
            mu_abs = self._beltrami_2d(field)
        elif field.ndim == 5:
            if field.shape[1] not in (2, 3):
                raise ValueError(
                    f"3D Beltrami needs (B,2|3,D,H,W) displacement, got {tuple(field.shape)}"
                )
            # Per-slice Beltrami on the in-plane (u, v): vectorize batch and depth.
            B, _, D, H, W = field.shape
            inplane = field[:, :2].permute(0, 2, 1, 3, 4).reshape(B * D, 2, H, W)
            mu_abs = self._beltrami_2d(inplane)
        else:
            raise ValueError(
                "Beltrami input must be (B,2,H,W) or (B,2|3,D,H,W), got "
                f"{field.ndim}D"
            )

        if self.reduction == "infty":
            # Per-element max — keep grad via amax.
            return mu_abs.amax()
        if self.reduction == "mean":
            return mu_abs.mean()
        # "penalty": softplus-style hinge above the QC bound.
        return torch.relu(mu_abs - 1.0).pow(2).mean()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_field(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None,
        displacement: torch.Tensor | None,
    ) -> torch.Tensor:
        if displacement is not None:
            return displacement
        if target is None:
            return pred
        if pred.shape != target.shape:
            raise ValueError(
                "pred and target must have the same shape when displacement is None "
                f"(got {tuple(pred.shape)} vs {tuple(target.shape)})"
            )
        if pred.shape[1] not in (2, 3):
            raise ValueError(
                "Beltrami fallback (pred, target) needs a displacement with 2 or 3 channels; "
                f"got {pred.shape[1]}. A 1-channel intensity residual is not a displacement: "
                "pass displacement= from the model's displacement head."
            )
        return pred - target

    def _beltrami_2d(self, field: torch.Tensor) -> torch.Tensor:
        """Compute |mu(f)| pointwise for a (B, 2, H, W) field."""
        u = field[:, 0]
        v = field[:, 1]
        # Use central differences via roll; pad-by-replicate at edges
        # to avoid spurious wrap-around (we discard the last row/col).
        u_x = self._dx(u)
        u_y = self._dy(u)
        v_x = self._dx(v)
        v_y = self._dy(v)

        # 2 f_zbar over 2 f_z for f = z + d: the identity contributes 2 to the
        # real part of 2 f_z and nothing to 2 f_zbar.
        num_re = u_x - v_y
        num_im = v_x + u_y
        den_re = 2.0 + u_x + v_y
        den_im = v_x - u_y

        num_sq = num_re.pow(2) + num_im.pow(2)
        den_sq = den_re.pow(2) + den_im.pow(2) + self.eps
        ratio = num_sq / den_sq
        # sqrt has an infinite derivative at 0, which a conformal region (mu = 0)
        # hits exactly; route those entries through a constant so their gradient is 0.
        positive = ratio > 0
        safe = torch.where(positive, ratio, torch.ones_like(ratio))
        return torch.where(positive, safe.sqrt(), torch.zeros_like(ratio))

    @staticmethod
    def _dx(t: torch.Tensor) -> torch.Tensor:
        # Central finite difference along the W axis with replicate
        # padding so the output shape matches the input.
        tp = F.pad(t.unsqueeze(1), (1, 1, 0, 0), mode="replicate").squeeze(1)
        return 0.5 * (tp[..., :, 2:] - tp[..., :, :-2])

    @staticmethod
    def _dy(t: torch.Tensor) -> torch.Tensor:
        # Central finite difference along the H axis with replicate
        # padding so the output shape matches the input.
        tp = F.pad(t.unsqueeze(1), (0, 0, 1, 1), mode="replicate").squeeze(1)
        return 0.5 * (tp[..., 2:, :] - tp[..., :-2, :])


__all__ = ["BeltramiDiagnosticLoss"]
