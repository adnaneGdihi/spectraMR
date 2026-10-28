r"""Bloch-consistency loss for parameter-map / quantitative-MRI bottlenecks.

For a generator that outputs *quantitative tissue parameters*
:math:`(\\hat\\rho, \\hat T_1, \\hat T_2)` rather than weighted images, the
contrast :math:`c` of any sequence is recovered through the closed-form
Bloch signal :math:`B(\\rho, T_1, T_2; \\theta_c)` where :math:`\\theta_c`
encodes :math:`(TE, TR, TI, \\alpha)`. The consistency loss

.. math::

   \\mathcal{L}_{\\text{phys}}
   \\;=\\;
   \\sum_c
   \\big\\|
     \\mathbf{x}_c
     - B(\\hat\\rho, \\hat T_1, \\hat T_2; \\theta_c)
   \\big\\|_p

drives the network to produce *physically valid* parameter maps: ones that,
when run through the Bloch equations of every observed contrast, reproduce
that contrast's measured weighting. This is the anchoring loss of the
Bloch-bottleneck idea (§10a of the multi-contrast brainstorm) and the same
training signal that powers SynthSR / SynthSeg generalisation across
unseen contrasts [2, 3].

The loss is a thin wrapper around the existing
:class:`~spectramr.infrastructure.physics.multi_physics_bloch.MultiPhysicsBlochLayer`,
which already implements SE / IR / GRE / DWI signal equations with proper
unit handling. Per CLAUDE.md the Bloch layer is the single source of truth
for tissue-to-signal conversion — this loss does not duplicate the equations.

References
----------
[1] E. M. Haacke, R. W. Brown, M. R. Thompson, R. Venkatesan,
    *Magnetic Resonance Imaging: Physical Principles and Sequence Design*,
    Wiley, 2014.
[2] J. E. Iglesias *et al.*, "Joint super-resolution and synthesis of 1 mm
    isotropic MP-RAGE volumes from clinical MRI exams with scans of
    different orientation, resolution and contrast," *NeuroImage*, vol. 237,
    p. 118206, 2021.
[3] B. Billot *et al.*, "SynthSeg: Segmentation of brain MRI scans of any
    contrast and resolution without retraining," *Med. Image Anal.*,
    vol. 86, p. 102789, 2023.

Section 2 / §10a of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.infrastructure.physics.multi_physics_bloch import MultiPhysicsBlochLayer
from spectramr.models.losses.registry import register_loss

__all__ = ["BlochConsistencyLoss"]


@register_loss(
    name="bloch_consistency",
    domain="physics",
    compatible_with=["image"],
)
class BlochConsistencyLoss(nn.Module):
    """Anchor predicted parameter maps to observed contrast images.

    The generator is expected to expose its tissue-parameter prediction in
    the ``context`` dict passed to the loss, under the key
    ``tissue_params`` — itself a dict with keys ``rho``, ``t1``, ``t2``
    (each of shape ``(B, 1, H, W)``). The strategy is responsible for
    populating this dict; see ``docs/multi_contrast.md`` for the contract.

    For every observed contrast :math:`c`, the loss synthesises
    :math:`B(\\hat\\rho, \\hat T_1, \\hat T_2; \\theta_c)` and compares against
    the measured :math:`\\mathbf{x}_c`.

    Args:
        norm: Distance norm — ``"l1"``, ``"l2"``, or ``"smooth_l1"``.
        learnable_flip_angle: Forwarded to the underlying
            ``MultiPhysicsBlochLayer``. Default False so the flip angle is
            taken from the metadata supplied in ``context["acquisition_params"]``.
    """

    _NORM_FNS: ClassVar[dict[str, Callable[..., torch.Tensor]]] = {
        "l1": F.l1_loss,
        "l2": F.mse_loss,
        "smooth_l1": F.smooth_l1_loss,
    }

    def __init__(
        self,
        norm: str = "l1",
        learnable_flip_angle: bool = False,
    ) -> None:
        super().__init__()
        if norm not in self._NORM_FNS:
            raise ValueError(f"norm must be one of {list(self._NORM_FNS)}, got {norm!r}")
        self.norm = norm
        self.bloch = MultiPhysicsBlochLayer(
            learnable_flip_angle=learnable_flip_angle,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if context is None:
            raise ValueError(
                "BlochConsistencyLoss requires a `context` dict containing "
                "`tissue_params` and `acquisition_params`."
            )
        tissue_params = context.get("tissue_params")
        acq = context.get("acquisition_params")
        if tissue_params is None or acq is None:
            raise ValueError(
                "context must contain both `tissue_params` "
                "(dict with rho/t1/t2 tensors) and `acquisition_params` "
                "(dict with TR/TE/TI/contrast_type/...)"
            )
        synth = self.bloch(tissue_params=tissue_params, metadata=acq)
        if synth.shape != target.shape:
            raise ValueError(
                f"Bloch synthesis shape {tuple(synth.shape)} does not match "
                f"target shape {tuple(target.shape)} — check acquisition metadata."
            )
        return self._NORM_FNS[self.norm](synth, target)
