"""Hyperelastic Jacobian Loss — Incompressibility Regularization.

Enforces the physical law of mass conservation on predicted deformation
fields by penalising deviations of the Jacobian determinant from unity.

Physics:
    Human soft tissue is ~70 % water and effectively incompressible.
    For a deformation field :math:`\\Phi`, conservation of mass requires:

    .. math::

        \\det\\bigl(J_{\\Phi}(\\mathbf{x})\\bigr) = 1
        \\quad \\forall\\, \\mathbf{x}

    where :math:`J_{\\Phi} = \\nabla \\Phi` is the spatial Jacobian.
    Positive det(J) > 0  guarantees diffeomorphism (no folding).

Loss:
    .. math::

        \\mathcal{L}_{hyper} = \\frac{1}{N}\\sum_{\\mathbf{x}}
            \\bigl(\\det J(\\mathbf{x}) - 1\\bigr)^2
            + \\lambda_{fold} \\cdot \\text{ReLU}\\bigl(-\\det J(\\mathbf{x})\\bigr)

    The second term is a hard penalty for topological violations (folding).

Reference:
    Rühaak et al., "Estimation of Large Motion in Lung CT," *Medical
    Image Analysis*, 2013.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from spectramr.models.losses.registry import register_loss


@register_loss("hyperelastic_jacobian", aliases=["jacobian_det", "incompressibility"])
class HyperelasticJacobianLoss(nn.Module):
    r"""Penalise deformation fields that violate tissue incompressibility.

    Computes the Jacobian determinant via finite differences and penalises
    both volume change ``(det(J)-1)²`` and topological folding ``det(J)<0``.

    Args:
        fold_penalty_weight: Extra penalty weight for negative det(J).
            Set to 0 to disable folding penalty.
        eps: Small constant to avoid numerical issues in gradients.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Hyperelastic} = \mathbb{E}[(|J_u| - 1)^2] + \lambda \mathbb{E}[\max(0, -|J_u|)]"""

    def __init__(
        self,
        fold_penalty_weight: float = 10.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.fold_penalty_weight = fold_penalty_weight
        self.eps = eps

    @staticmethod
    def _jacobian_det_2d(deformation_field: Tensor) -> Tensor:
        """Compute det(J) for a 2-D displacement field.

        Args:
            deformation_field: ``[B, 2, H, W]`` — 2-channel (dx, dy)
                displacement vectors at each pixel.

        Returns:
            Jacobian determinant ``[B, 1, H-2, W-2]``.
        """
        # Central differences for partial derivatives
        # ∂u_x/∂x, ∂u_x/∂y, ∂u_y/∂x, ∂u_y/∂y
        u_x = deformation_field[:, 0:1]  # [B, 1, H, W]
        u_y = deformation_field[:, 1:2]

        # ∂u_x/∂x = (u_x[..., x+1] - u_x[..., x-1]) / 2
        du_x_dx = (u_x[:, :, 1:-1, 2:] - u_x[:, :, 1:-1, :-2]) / 2.0
        # ∂u_x/∂y = (u_x[..., y+1, :] - u_x[..., y-1, :]) / 2
        du_x_dy = (u_x[:, :, 2:, 1:-1] - u_x[:, :, :-2, 1:-1]) / 2.0
        # ∂u_y/∂x
        du_y_dx = (u_y[:, :, 1:-1, 2:] - u_y[:, :, 1:-1, :-2]) / 2.0
        # ∂u_y/∂y
        du_y_dy = (u_y[:, :, 2:, 1:-1] - u_y[:, :, :-2, 1:-1]) / 2.0

        # J = I + ∇u  →  det(J) = (1 + ∂ux/∂x)(1 + ∂uy/∂y) - (∂ux/∂y)(∂uy/∂x)
        det_j = (1.0 + du_x_dx) * (1.0 + du_y_dy) - du_x_dy * du_y_dx

        return det_j

    @staticmethod
    def _jacobian_det_3d(deformation_field: Tensor) -> Tensor:
        """Compute det(J) for a 3-D displacement field.

        Args:
            deformation_field: ``[B, 3, D, H, W]`` — 3-channel (dx, dy, dz).

        Returns:
            Jacobian determinant ``[B, 1, D-2, H-2, W-2]``.
        """
        u = deformation_field  # [B, 3, D, H, W]

        # Central differences for all 9 partial derivatives
        # Slice interior only: [1:-1, 1:-1, 1:-1]
        s = (slice(None), slice(None), slice(1, -1), slice(1, -1), slice(1, -1))

        du = []
        for spatial_dim in (2, 3, 4):
            sl_fwd = list(s)
            sl_bwd = list(s)
            sl_fwd[spatial_dim] = slice(2, None)
            sl_bwd[spatial_dim] = slice(None, -2)
            du.append((u[tuple(sl_fwd)] - u[tuple(sl_bwd)]) / 2.0)

        # du[0] = ∂u/∂z, du[1] = ∂u/∂y, du[2] = ∂u/∂x
        # J_ij = δ_ij + ∂u_i/∂x_j
        j00 = 1.0 + du[2][:, 0:1]  # 1 + ∂ux/∂x
        j01 = du[1][:, 0:1]  # ∂ux/∂y
        j02 = du[0][:, 0:1]  # ∂ux/∂z
        j10 = du[2][:, 1:2]  # ∂uy/∂x
        j11 = 1.0 + du[1][:, 1:2]  # 1 + ∂uy/∂y
        j12 = du[0][:, 1:2]  # ∂uy/∂z
        j20 = du[2][:, 2:3]  # ∂uz/∂x
        j21 = du[1][:, 2:3]  # ∂uz/∂y
        j22 = 1.0 + du[0][:, 2:3]  # 1 + ∂uz/∂z

        det_j = (
            j00 * (j11 * j22 - j12 * j21)
            - j01 * (j10 * j22 - j12 * j20)
            + j02 * (j10 * j21 - j11 * j20)
        )
        return det_j

    def forward(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        intermediate_outputs: list[Tensor] | None = None,
        **kwargs: Any,
    ) -> Tensor:
        """Compute the hyperelastic Jacobian loss.

        Args:
            prediction: Primary prediction (often the reconstructed image).
            target: Ignored — this is a regularisation loss, not a
                reconstruction loss.  Accepts ``target`` for API
                compatibility with ``ILoss`` protocol.
            intermediate_outputs: List of intermediate outputs from the generator.
                If provided, the first element is assumed to be the deformation field.
                Otherwise, falls back to using `prediction` directly.

        Returns:
            Scalar loss value.

        forward method for HyperelasticJacobianLoss.

        Executes PyTorch tensor operations.

        Args:
            prediction (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            intermediate_outputs (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if intermediate_outputs is not None and len(intermediate_outputs) > 0:
            # Assume deformation field is provided in the intermediates
            u = intermediate_outputs[0]
        else:
            u = prediction

        # The rank guard below existed already; the CHANNEL guard did not, and
        # the channel axis is the one the Jacobian maths actually indexes. On a
        # C=1 input `_jacobian_det_2d` slices `u[:, 1:2]` to an EMPTY tensor,
        # every derivative downstream is empty, and `torch.mean` of an empty
        # tensor is nan -- silently, and with a zero gradient. That nan then
        # propagates into a real weighted total (reproduced independently by
        # tests/fuzz/loss_composition_fuzz on shape (1, 1, 8, 8)). Per
        # non-negotiable #3 a violated input contract must RAISE, not degrade.
        if u.dim() == 4:
            expected_c = 2
        elif u.dim() == 5:
            expected_c = 3
        else:
            raise ValueError(f"Expected 4-D or 5-D deformation field, got {u.dim()}-D")
        if u.shape[1] != expected_c:
            raise ValueError(
                f"HyperelasticJacobianLoss needs a displacement field with "
                f"{expected_c} channels for a {u.dim() - 2}-D field (one per "
                f"spatial axis); got shape {tuple(u.shape)}. A single-channel "
                f"image is not a deformation field."
            )
        det_j = self._jacobian_det_2d(u) if u.dim() == 4 else self._jacobian_det_3d(u)

        # Volume preservation: (det(J) - 1)²
        volume_loss = torch.mean((det_j - 1.0) ** 2)

        # Topology preservation: penalise negative det(J) (folding)
        if self.fold_penalty_weight > 0:
            fold_loss = torch.mean(torch.relu(-det_j))
            total = volume_loss + self.fold_penalty_weight * fold_loss
        else:
            total = volume_loss

        return total


__all__ = ["HyperelasticJacobianLoss"]
