r"""Acquisition-flow equivariance via Lie-derivative regularization (AFE).

Continuous-parameter generalization requires the representation to transform
predictably as acquisition parameters vary. AFE defines the equivariance
*generator* from the MRI signal-equation derivative,

.. math::

   \mathbf v_{\boldsymbol\varphi}(\mathbf x)
     = \partial_{\boldsymbol\varphi}\,\mathcal F^{\mathrm{Bloch}}
       (\boldsymbol\xi;\boldsymbol\varphi),

so an infinitesimal change of an acquisition parameter acts as
:math:`\mathbf x\mapsto\mathbf x+\mathbf v_{\boldsymbol\varphi}\,d\boldsymbol
\varphi`. By the equivalence of consistency regularization and Lie-derivative-norm
minimisation (Gruver et al. 2023), penalising the feature change along this flow
drives the representation toward equivariance/invariance under the acquisition
flow. The penalty here is the invariant case (generator :math:`A_{\boldsymbol
\varphi}=0`): the squared feature defect between the base and flowed inputs.

Consistency (the decisive check): where the contrast is insensitive to a
parameter (:math:`\mathrm{TR}\gg T_1`, so :math:`\partial_{\mathrm{TR}}\mathcal
F\to0`), the generator vanishes and the penalty imposes no constraint -- the
representation is correctly left unconstrained where no contrast change occurs.

The guarantee is first-order; the penalty weight is annealed and several
infinitesimal generators composed for large acquisition excursions.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss

# Clean contrast labels handled by the analytic generator (no normalisation
# coupling, so the per-voxel derivative is exact and local).
_SUPPORTED_CONTRASTS = ("spin_echo", "se", "t2", "inversion_recovery", "ir", "flair")
_RELAXATION_FLOOR_MS = 1e-3


def _signal(
    contrast_type: str,
    rho: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    tr: torch.Tensor,
    te: torch.Tensor,
    ti: torch.Tensor,
) -> torch.Tensor:
    """Analytic (un-normalised) magnitude signal for the supported contrasts."""
    ct = contrast_type.lower()
    e1 = torch.exp(-tr / t1)
    e2 = torch.exp(-te / t2)
    if ct in ("spin_echo", "se", "t2"):
        return rho * (1.0 - e1) * e2
    if ct in ("inversion_recovery", "ir", "flair"):
        e_ti = torch.exp(-ti / t1)
        return rho * torch.abs(1.0 - 2.0 * e_ti + e1) * e2
    raise ValueError(
        f"Unknown contrast_type {contrast_type!r}; expected one of "
        f"{sorted(set(_SUPPORTED_CONTRASTS))}."
    )


def acquisition_flow_generator(
    params: torch.Tensor,
    acquisition: dict[str, float],
    *,
    wrt: str = "TE",
    contrast_type: str = "spin_echo",
) -> torch.Tensor:
    r"""Per-voxel acquisition-flow generator :math:`\partial_{\varphi}\mathcal F`.

    Computed by autograd of the analytic signal equation w.r.t. the chosen
    acquisition parameter, broadcast to a full-resolution tensor so each output
    voxel depends only on its own parameter value (exact, local derivative).

    Args:
        params: ``[B, 3, ...]`` ``(rho, T1_ms, T2_ms)`` map.
        acquisition: ``{"TR","TE","TI"}`` in ms.
        wrt: Which acquisition parameter to differentiate (``"TR"``/``"TE"``/``"TI"``).
        contrast_type: Contrast label.

    Returns:
        :math:`\partial\mathcal F/\partial\varphi` per voxel, shape ``[B, 1, ...]``.

    Raises:
        ValueError: unknown ``wrt`` or ``contrast_type``.
    """
    if wrt not in ("TR", "TE", "TI"):
        raise ValueError(f"wrt must be one of TR/TE/TI, got {wrt!r}.")
    rho = params[:, 0:1]
    # Background voxels of a parameter atlas carry T1 = T2 = 0, where the
    # derivative of exp(-TR/T1) is 0 * inf = NaN; the floored value gives the
    # physical limit (a zero generator) instead.
    t1 = params[:, 1:2].clamp_min(_RELAXATION_FLOOR_MS)
    t2 = params[:, 2:3].clamp_min(_RELAXATION_FLOOR_MS)

    base = {
        "TR": float(acquisition["TR"]),
        "TE": float(acquisition["TE"]),
        "TI": float(acquisition.get("TI", 0.0)),
    }
    full = {key: torch.full_like(rho, val) for key, val in base.items()}
    full[wrt] = full[wrt].clone().requires_grad_(True)

    with torch.enable_grad():
        signal = _signal(contrast_type, rho, t1, t2, full["TR"], full["TE"], full["TI"])
        # A contrast that does not depend on ``wrt`` (TI under spin echo) has a
        # zero generator, which is the "no constraint" case the module promises.
        if not signal.requires_grad:
            return torch.zeros_like(rho)
        (grad,) = torch.autograd.grad(
            signal.sum(), full[wrt], create_graph=False, allow_unused=True
        )
    if grad is None:
        return torch.zeros_like(rho)
    return grad


@register_loss(name="acq_flow_lie", domain="image")
class AcquisitionFlowLieLoss(nn.Module):
    r"""Lie-derivative invariance penalty along the acquisition flow.

    The penalty is the squared feature defect between the base and flowed
    representations (invariant case :math:`A_{\boldsymbol\varphi}=0`):
    :math:`\mathcal R = \lVert \Phi(\mathbf x) - \Phi(\mathbf x + \epsilon\,
    \mathbf v_{\boldsymbol\varphi}) \rVert^2`. The flowed features are supplied
    by the strategy (which builds :math:`\mathbf v_{\boldsymbol\varphi}` with
    :func:`acquisition_flow_generator`).

    Args:
        weight: Static multiplier (the strategy anneals it from zero).
    """

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        """Squared feature defect between base (``prediction``) and flowed (``target``)."""
        if prediction.shape != target.shape:
            raise ValueError(
                f"AFE expects matching base/flowed feature shapes; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        return self.weight * (prediction - target).pow(2).mean()


__all__ = ["AcquisitionFlowLieLoss", "acquisition_flow_generator"]
