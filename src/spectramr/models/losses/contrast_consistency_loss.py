r"""Bottomley-derived contrast-consistency loss (XField-FM / idea 2).

Penalises predictions whose implied tissue T1 (under the standard
Bottomley closed form) disagrees with the prior table at the
acquisition's actual ``B0``. In other words: the network is allowed
to be wrong about the *image*, but it should remain physically
plausible about the *contrast*.

The proposal phrases this as a Bottomley-derived inequality
(``TODO/integration_plan_ulf_cheap_fast_mri.md`` §2.2). For a tissue
class :math:`c` and field strength :math:`B_0`, the prior gives a
target T1 :math:`\hat T_1^c(B_0)`. The loss is the mean-squared
deviation between the network's predicted (or fitted) T1 map and the
per-pixel prior, weighted by the tissue-class probability map.

Inputs
------
- ``predicted_t1_ms``: ``[B, 1, H, W]`` real tensor, the network's
  inferred T1 in milliseconds (typically derived from a relaxometry
  head or estimated from the contrast map).
- ``tissue_probabilities``: ``[B, K, H, W]`` soft tissue-class
  segmentation. Channels are ordered consistently with
  :class:`TissueClass` enum.
- ``field_strength_T``: scalar tensor or float — the acquisition's B0.

The loss is differentiable wrt ``predicted_t1_ms`` (so the network
learns to match the priors) and *constant* wrt the priors (the priors
are physical truth, not learnable).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from spectramr.config.schemas.enums import Regime, Task
from spectramr.infrastructure.physics.relaxation_priors import (
    TissueClass,
    bottomley_t1,
)
from spectramr.models.losses.registry import register_loss

# Default tissue-class ordering: matches the enum's declaration order so
# the segmentation channel order is well-defined.
DEFAULT_TISSUE_ORDER: tuple[TissueClass, ...] = (
    TissueClass.WHITE_MATTER,
    TissueClass.GRAY_MATTER,
    TissueClass.CSF,
)


@register_loss(
    name="contrast_consistency",
    aliases=["ContrastConsistencyLoss", "bottomley_consistency"],
    domain="image",
    workflows=frozenset({Regime.QUANTITATIVE}),
    tasks=frozenset({Task.PARAMETER_MAPPING}),
)
class ContrastConsistencyLoss(nn.Module):
    r"""MSE between predicted T1 and the Bottomley prior, tissue-weighted.

    Tagged ``mri_quantitative``: it grades a predicted T1 map in ms against the
    Bottomley field-strength-dependent relaxometry prior, so both the quantity
    and the prior are relaxometry. It raises on shape/channel mismatch rather
    than broadcasting.

    Args:
        weight: ``λ`` multiplier. Must be ≥ 0.
        tissue_order: Iterable of :class:`TissueClass` values matching
            the channel order of ``tissue_probabilities``. Defaults to
            ``(WHITE_MATTER, GRAY_MATTER, CSF)``.
        reduction: ``"mean"`` (default) or ``"sum"``.

    The loss is zero when the network's T1 prediction matches the
    Bottomley target everywhere the tissue probability is non-zero.

    Mathematical formulation
    ------------------------
    .. math::

        \mathcal{L}_{cc} = \lambda \cdot \frac{
            \sum_{c} \int p_c(\mathbf r)
            (\hat T_1(\mathbf r) - \hat T_1^c(B_0))^2 \mathrm d\mathbf r
        }{ \sum_{c} \int p_c(\mathbf r) \mathrm d\mathbf r }

    where :math:`p_c` is the tissue-class probability for class
    :math:`c` and :math:`\hat T_1^c(B_0)` is the Bottomley prior.
    """

    def __init__(
        self,
        weight: float = 1.0,
        tissue_order: Sequence[TissueClass] | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be ≥ 0; got {weight}")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum'; got {reduction!r}")
        self.weight = float(weight)
        self.reduction = reduction
        self.tissue_order: tuple[TissueClass, ...] = tuple(
            tissue_order if tissue_order is not None else DEFAULT_TISSUE_ORDER
        )

    def forward(
        self,
        predicted_t1_ms: torch.Tensor,
        tissue_probabilities: torch.Tensor,
        field_strength_T: float | torch.Tensor,  # noqa: N803 — unit suffix Tesla
        **_: object,
    ) -> torch.Tensor:
        r"""Compute the contrast-consistency loss.

        Args:
            predicted_t1_ms: ``[B, 1, H, W]`` predicted T1 in ms.
            tissue_probabilities: ``[B, K, H, W]`` tissue probabilities.
                ``K`` must equal ``len(self.tissue_order)``.
            field_strength_T: scalar B0 in Tesla.

        Returns:
            Scalar loss tensor.

        Raises:
            ValueError: shape / channel-count mismatches.
        """
        if predicted_t1_ms.dim() != 4 or predicted_t1_ms.shape[1] != 1:
            raise ValueError(
                f"predicted_t1_ms must be [B, 1, H, W]; got {tuple(predicted_t1_ms.shape)}"
            )
        if tissue_probabilities.dim() != 4:
            raise ValueError(
                f"tissue_probabilities must be 4-D; got {tuple(tissue_probabilities.shape)}"
            )
        K = tissue_probabilities.shape[1]
        if len(self.tissue_order) != K:
            raise ValueError(
                f"tissue_probabilities has {K} channels but tissue_order has "
                f"{len(self.tissue_order)} entries — must match."
            )

        b0_value = (
            float(field_strength_T.item())
            if isinstance(field_strength_T, torch.Tensor)
            else float(field_strength_T)
        )
        if b0_value <= 0:
            raise ValueError(f"field_strength_T must be > 0; got {b0_value}")

        # Build the per-class target T1 in ms.  Stored as a
        # buffer-equivalent tensor on the same device / dtype.
        targets = torch.tensor(
            [bottomley_t1(b0_value, t) for t in self.tissue_order],
            dtype=predicted_t1_ms.dtype,
            device=predicted_t1_ms.device,
        )  # [K]

        # Squared error between prediction and each tissue's target,
        # broadcast across spatial dims.
        # predicted_t1_ms: [B, 1, H, W] → [B, K, H, W]
        sq_err = (predicted_t1_ms - targets.view(1, K, 1, 1)) ** 2  # [B, K, H, W]

        # Weight by tissue probability and aggregate.
        weighted = sq_err * tissue_probabilities
        if self.reduction == "mean":
            denom = tissue_probabilities.sum().clamp_min(1e-12)
            penalty = weighted.sum() / denom
        else:
            penalty = weighted.sum()
        return self.weight * penalty


__all__ = ["DEFAULT_TISSUE_ORDER", "ContrastConsistencyLoss"]
