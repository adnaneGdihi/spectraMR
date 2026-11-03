"""Cross-field translation losses (MICCAI MRIxFields2026).

``latent_cycle`` enforces encode-once / render-anywhere identifiability for the
:class:`AnatomyFieldRenderer` backbone: the latent re-extracted from a rendered
image must match the original latent. This is the empirical surrogate for the
nonlinear-ICA identifiability argument (the field index is the auxiliary variable)
that breaks the generic non-identifiability of single-view autoencoders.
"""

from __future__ import annotations

import torch
from torch import nn

from spectramr.models.losses.registry import register_loss


@register_loss(
    name="latent_cycle",
    domain="image",
    compatible_with=["cross_field_translation", "reconstruction"],
)
class LatentCycleLoss(nn.Module):
    r"""Latent-cycle identifiability loss :math:`\lVert E(R(q,b)) - q \rVert_1`.

    The strategy supplies ``prediction = E(R(q, b))`` (the re-encoded latent) and
    ``target = q`` (the original latent). Zero at perfect cycle-consistency.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(prediction - target))


@register_loss(
    name="cocycle_consistency",
    domain="image",
    compatible_with=["field_cocycle", "cross_field_translation"],
)
class CocycleConsistencyLoss(nn.Module):
    r"""Cocycle-consistency residual :math:`\lVert G(G(x;s,t);t,u) - G(x;s,u)\rVert_1`.

    This is the quantitative form of Theorem 6 (a flat groupoid cocycle over a
    connected base is a coboundary) for the encode-once / render-anywhere
    ``AnatomyFieldRenderer``: because ``encode`` factors the source field out of the latent, the
    composite map ``G(G(x;s,t);t,u) = R(E(R(E(x,b_s),b_t),b_t),b_u)`` must equal the direct map
    ``G(x;s,u) = R(E(x,b_s),b_u)``. The :class:`FieldCocycleTranslationStrategy` builds
    both images (``prediction`` = composite, ``target`` = direct) over a freely
    sampled intermediate field and passes them here. Exactly zero on a family that
    factorises through a single canonicaliser; the running value is the logged
    ``cocycle_residual`` :math:`\varepsilon_{\mathrm{coc}}` (Corollary 5), so the
    mechanism is measured, not assumed (pitfall #16).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(prediction - target))


@register_loss(
    name="field_identity",
    domain="image",
    compatible_with=["field_cocycle", "cross_field_translation"],
)
class FieldIdentityLoss(nn.Module):
    r"""Field-identity residual :math:`\lVert G(x;s,s) - x \rVert_1`.

    The cocycle law's identity axiom ``G(\cdot;s,s) = Id``: rendering the source
    image back at its OWN field must reproduce it. The strategy supplies
    ``prediction = R(E(x), b_s)`` and ``target = x``. Together with a non-zero
    paired-fidelity/adversarial term (enforced by the
    ``field_cocycle_fidelity_nonzero`` Tier-1 guard) this pins the family away from
    the trivial ``G = Id`` collapse that identity+cocycle alone would admit.
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
        return self.weight * torch.mean(torch.abs(prediction - target))


@register_loss(
    name="field_flow_velocity",
    aliases=["field_flow"],
    domain="image",
    compatible_with=["field_flow", "cross_field_translation"],
)
class FieldFlowVelocityLoss(nn.Module):
    r"""Conditional-flow-matching velocity regression for field-flow (B-3.1).

    The strategy supplies ``prediction = v_theta(x(s), 10^{tau(s)})`` and
    ``target`` = the conditional velocity the model must match. For field-flow that
    target is the **per-unit-tau** secant slope ``u = (x_t - x_s)/(tau_t - tau_s)``
    (NOT the raw displacement ``x_t - x_s``), so that integrating ``dx/dtau = v``
    over ``[tau_s, tau_t]`` reproduces the endpoint. Zero when the predicted
    velocity matches the conditional velocity. Defaults to squared error
    (``norm='l2'``, the flow-matching paper choice); ``norm='l1'`` is also accepted.
    """

    def __init__(self, weight: float = 1.0, norm: str = "l2") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2'; got {norm!r}")
        self.weight = weight
        self.norm = norm

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, **_: object) -> torch.Tensor:
        diff = prediction - target
        per = diff.pow(2) if self.norm == "l2" else diff.abs()
        return self.weight * torch.mean(per)
