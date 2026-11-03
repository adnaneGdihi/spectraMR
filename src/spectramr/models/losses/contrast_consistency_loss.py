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
    bottomley_t1_tensor,
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
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        aux_maps: dict[str, torch.Tensor] | None = None,
        field_strength: torch.Tensor | None = None,
        **_: object,
    ) -> torch.Tensor:
        r"""Compute the contrast-consistency loss from the model's auxiliary maps.

        Args:
            pred, target: the image pair every image loss receives; unused here.
            aux_maps: the generator's ``t1_ms`` ``[B, 1, H, W]`` and
                ``tissue_probabilities`` ``[B, K, H, W]`` from its last forward.
            field_strength: ``[B]`` per-sample B0 in Tesla.

        Raises:
            ValueError: a missing input or a shape / channel-count mismatch.
        """
        del pred, target
        if not aux_maps or "t1_ms" not in aux_maps or "tissue_probabilities" not in aux_maps:
            raise ValueError(
                "contrast_consistency needs the generator's aux_maps (t1_ms, "
                "tissue_probabilities); the model must build its tissue heads"
            )
        if field_strength is None:
            raise ValueError("contrast_consistency needs the batch's per-sample field_strength")
        predicted_t1_ms, tissue_probabilities = aux_maps["t1_ms"], aux_maps["tissue_probabilities"]
        if predicted_t1_ms.dim() != 4 or predicted_t1_ms.shape[1] != 1:
            raise ValueError(f"t1_ms must be [B, 1, H, W]; got {tuple(predicted_t1_ms.shape)}")
        k = tissue_probabilities.shape[1]
        if tissue_probabilities.dim() != 4 or len(self.tissue_order) != k:
            raise ValueError(
                f"tissue_probabilities {tuple(tissue_probabilities.shape)} must be [B, "
                f"{len(self.tissue_order)}, H, W] (one channel per tissue_order entry)"
            )
        b0 = field_strength.reshape(-1).to(predicted_t1_ms)
        if b0.shape[0] != predicted_t1_ms.shape[0]:
            raise ValueError(
                f"{b0.shape[0]} field strengths for {predicted_t1_ms.shape[0]} samples"
            )
        # Per-sample, per-class Bottomley target [B, K]; no host read of B0.
        targets = torch.stack([bottomley_t1_tensor(b0, t) for t in self.tissue_order], dim=1)
        sq_err = (predicted_t1_ms - targets[:, :, None, None]) ** 2  # [B, K, H, W]
        weighted = sq_err * tissue_probabilities
        if self.reduction == "mean":
            penalty = weighted.sum() / tissue_probabilities.sum().clamp_min(1e-12)
        else:
            penalty = weighted.sum()
        return self.weight * penalty


__all__ = ["DEFAULT_TISSUE_ORDER", "ContrastConsistencyLoss"]
