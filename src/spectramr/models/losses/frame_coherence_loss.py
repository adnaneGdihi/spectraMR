r"""Mutual-coherence penalty for a learnable sparsifying frame (A-6.5 *SCO-Frame*).

The coherence-optimisation half of A-6.5: a registered loss that drives the
off-diagonal of a frame's atom Gram matrix toward zero, so the learned atoms
become mutually incoherent. Mutual incoherence is a classical sparse-recovery
principle (low coherence certifies unique :math:`\ell_1` recovery for a *fixed*
dictionary, Donoho-Elad) but **that guarantee does not transfer to a learned,
task-specific frame** — here it is an *empirical* regulariser whose effect is the
testable ablation, not a recovery theorem. It is the **one-knob** of the
SCO-Frame ablation — turning ``lambda_coherence`` from 0 to >0 is the only change
between control and treatment.

For the atom matrix :math:`W\in\mathbb{R}^{m\times n}` (``m`` unit-normalised
atoms :math:`\hat d_i`), the penalty is the Gram off-diagonal Frobenius energy

.. math::

    \mathcal{L}_{\mathrm{coh}}=\sum_{i\neq j}\langle\hat d_i,\hat d_j\rangle^2
        =\lVert \hat W\hat W^\top - I_m\rVert_F^2 ,

which is 0 iff the atoms are mutually orthogonal and rises with coherence.

**Anti-facade contract (CLAUDE.md #16).** This is a *parameter* penalty on the
frame's atoms, not a ``(prediction, target)`` loss. Invoked without a frame /
atom tensor it **raises** — it cannot be silently dropped into a YAML ``losses``
block (which would call it ``loss(pred, target)`` and it would have nothing to
penalise). Wire it through the sparse-frame training strategy, which passes
``frame=`` and applies the ``lambda_coherence`` weight.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from spectramr.models.losses.registry import register_loss


def gram_offdiagonal_penalty(atoms: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Gram off-diagonal Frobenius energy of unit-normalised ``atoms``.

    :math:`\sum_{i\neq j}\langle\hat d_i,\hat d_j\rangle^2`. Differentiable in
    ``atoms``; 0 iff the atoms are mutually orthogonal.

    Contract: **axis 0 is the atom axis**; any trailing axes are flattened to the
    feature axis (so a raw conv weight ``[m, C, k, k]`` is accepted and becomes
    ``[m, C*k*k]``). A rank-<2 tensor has no atom/feature split and raises.
    """
    if atoms.dim() < 2:
        raise ValueError(
            f"gram_offdiagonal_penalty expects atoms with >=2 dims (axis 0 = atoms); "
            f"got shape {tuple(atoms.shape)}."
        )
    if atoms.dim() != 2:
        atoms = atoms.reshape(atoms.shape[0], -1)
    wn = atoms / (atoms.norm(dim=1, keepdim=True) + eps)
    gram = wn @ wn.t()
    m = gram.shape[0]
    off = gram - torch.eye(m, device=gram.device, dtype=gram.dtype)
    penalty: torch.Tensor = (off**2).sum()
    return penalty


@register_loss(
    name="frame_coherence",
    aliases=["mutual_coherence_penalty", "frame_incoherence"],
    domain="agnostic",
)
class FrameCoherenceLoss(nn.Module):
    r"""Mutual-coherence penalty on a learnable frame's atoms (A-6.5 one-knob).

    **DOMAIN**: AGNOSTIC — a parameter penalty, independent of the data domain.

    Reads the frame from ``frame=`` (a module exposing ``coherence_penalty()`` —
    e.g. :class:`~spectramr.models.generators.tight_frame_learner.TightFrameLearner`)
    or ``atoms=`` (a raw ``[m, n]`` atom tensor). Returns the un-weighted Gram
    off-diagonal penalty (the strategy multiplies by ``lambda_coherence``). Raises
    if neither is supplied (anti-facade #16).
    """

    @staticmethod
    def _module_penalty(frame: nn.Module) -> torch.Tensor:
        # Attribute access would be typed ``Tensor | Module`` by ``nn.Module.__getattr__``.
        penalty_fn = getattr(frame, "coherence_penalty")  # noqa: B009
        penalty: torch.Tensor = penalty_fn()
        return penalty

    def forward(
        self,
        prediction: Any = None,
        target: Any = None,
        *,
        frame: nn.Module | None = None,
        atoms: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(frame, nn.Module) and hasattr(frame, "coherence_penalty"):
            return self._module_penalty(frame)
        if isinstance(atoms, torch.Tensor):
            return gram_offdiagonal_penalty(atoms)
        # A positional module (a strategy may pass the frame first-positionally).
        if isinstance(prediction, nn.Module) and hasattr(prediction, "coherence_penalty"):
            return self._module_penalty(prediction)
        raise ValueError(
            "FrameCoherenceLoss is a PARAMETER penalty on a learnable frame's atoms; "
            "it needs frame=<module with coherence_penalty()> or atoms=<[m,n] tensor>. "
            "It cannot be used as a (prediction, target) loss — wire it through the "
            "sparse-frame training strategy, not a YAML losses block (#16)."
        )
