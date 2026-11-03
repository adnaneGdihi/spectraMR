r"""Mutual-coherence penalty for a learnable sparsifying frame (A-6.5 *SCO-Frame*).

The coherence-optimisation half of A-6.5: a registered loss that penalises the
off-diagonal entries of a frame's atom Gram matrix, pushing the learned atoms apart.
Mutual incoherence is a classical sparse-recovery principle (low coherence certifies
unique :math:`\ell_1` recovery for a *fixed* dictionary, Donoho-Elad) but **that
guarantee does not transfer to a learned, task-specific frame** — here it is an *empirical* regulariser whose effect is the
testable ablation, not a recovery theorem. It is the **one-knob** of the
SCO-Frame ablation — turning ``lambda_coherence`` from 0 to >0 is the only change
between control and treatment.

For the atom matrix :math:`W\in\mathbb{R}^{m\times n}` (``m`` unit-normalised
atoms :math:`\hat d_i`, Gram :math:`G=\hat W\hat W^\top`) two penalties are offered,
selected by ``penalty`` (``training.sparse_frame.coherence_penalty``):

* ``smooth_max`` (default) — the log-sum-exp smooth maximum of the squared off-diagonal
  Gram entries (``experiments/inprogress/sparse_frame/PROPOSAL.md`` item e, #2368),

  .. math::

      L_\beta=\beta^{-1}\log\sum_{i<j}\exp\!\big(\beta\,G_{ij}^2\big),
      \qquad \max_{i<j}G_{ij}^2\le L_\beta\le\max_{i<j}G_{ij}^2+\frac{\log N}{\beta},

  with :math:`N=m(m-1)/2`. Its minimiser approaches the minimiser of the mutual
  coherence :math:`\mu=\max_{i\ne j}|G_{ij}|` as :math:`\beta\to\infty`, so the
  treatment minimises the quantity its hypothesis names.
* ``frobenius`` — the Gram off-diagonal energy
  :math:`\sum_{i\neq j}G_{ij}^2=\lVert G-I_m\rVert_F^2`. For an over-complete frame
  (:math:`m>n`) its floor :math:`m^2/n-m` is attained by every unit-norm tight frame
  (Benedetto-Fickus 2003), a set that contains frames far above the Welch bound, so it
  lowers the *average* squared coherence and reaches :math:`\mu` only indirectly.

**Anti-facade contract (CLAUDE.md #16).** This is a *parameter* penalty on the
frame's atoms, not a ``(prediction, target)`` loss. Invoked without a frame /
atom tensor it **raises** — it cannot be silently dropped into a YAML ``losses``
block (which would call it ``loss(pred, target)`` and it would have nothing to
penalise). Wire it through the sparse-frame training strategy, which passes
``frame=`` and applies the ``lambda_coherence`` weight.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from spectramr.models.losses.registry import register_loss

COHERENCE_PENALTIES: tuple[str, ...] = ("smooth_max", "frobenius")
DEFAULT_COHERENCE_BETA = 1000.0


def _unit_gram(atoms: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Gram of unit-normalised atoms; axis 0 = atoms, trailing axes flatten."""
    if atoms.dim() < 2:
        raise ValueError(
            f"frame coherence penalties expect atoms with >=2 dims (axis 0 = atoms); "
            f"got shape {tuple(atoms.shape)}."
        )
    if atoms.dim() != 2:
        atoms = atoms.reshape(atoms.shape[0], -1)
    if atoms.dtype in (torch.float16, torch.bfloat16):
        atoms = atoms.float()  # keep the Gram out of reduced precision
    wn = atoms / (atoms.norm(dim=1, keepdim=True) + eps)
    return wn @ wn.t()


def gram_offdiagonal_penalty(atoms: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    r"""Gram off-diagonal Frobenius energy of unit-normalised ``atoms``.

    :math:`\sum_{i\neq j}\langle\hat d_i,\hat d_j\rangle^2`. Differentiable in
    ``atoms``. Floor is 0 for ``m <= n`` (orthogonal atoms) and ``m**2/n - m`` for
    ``m > n`` (unit-norm tight frames).

    Contract: **axis 0 is the atom axis**; any trailing axes are flattened to the
    feature axis (so a raw conv weight ``[m, C, k, k]`` is accepted and becomes
    ``[m, C*k*k]``). A rank-<2 tensor has no atom/feature split and raises.
    """
    gram = _unit_gram(atoms, eps)
    off = gram - torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    penalty: torch.Tensor = (off**2).sum()
    return penalty


def smooth_max_coherence_penalty(
    atoms: torch.Tensor, beta: float = DEFAULT_COHERENCE_BETA, eps: float = 1e-12
) -> torch.Tensor:
    r"""Log-sum-exp smooth maximum :math:`\beta^{-1}\log\sum_{i<j}\exp(\beta G_{ij}^2)`.

    Same atom-axis contract as :func:`gram_offdiagonal_penalty`. Needs ``m >= 2``
    atoms (with one atom there is no pair to take a maximum over).
    """
    gram = _unit_gram(atoms, eps)
    m = gram.shape[0]
    if m < 2:
        raise ValueError(f"smooth-max coherence needs >= 2 atoms; got {m}.")
    iu = torch.triu_indices(m, m, offset=1, device=gram.device)
    sq = gram[iu[0], iu[1]] ** 2
    penalty: torch.Tensor = torch.logsumexp(beta * sq, dim=0) / beta
    return penalty


def validate_coherence_knobs(penalty: str, beta: float) -> None:
    """Raise on an unknown penalty name or a non-positive / non-finite ``beta``."""
    if penalty not in COHERENCE_PENALTIES:
        raise ValueError(
            f"Unknown frame coherence penalty {penalty!r}; expected one of {COHERENCE_PENALTIES}."
        )
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError(f"coherence beta must be finite and > 0; got {beta!r}.")


@register_loss(
    name="frame_coherence",
    aliases=["mutual_coherence_penalty", "frame_incoherence"],
    domain="agnostic",
)
class FrameCoherenceLoss(nn.Module):
    r"""Mutual-coherence penalty on a learnable frame's atoms (A-6.5 one-knob).

    **DOMAIN**: AGNOSTIC — a parameter penalty, independent of the data domain.

    ``penalty`` is one of :data:`COHERENCE_PENALTIES` (``smooth_max`` or
    ``frobenius``) and ``beta`` the smooth-max sharpness; both raise when invalid.
    Reads the atoms from ``frame=`` (a module exposing ``atoms()`` — e.g.
    :class:`~spectramr.models.generators.tight_frame_learner.TightFrameLearner`) or
    ``atoms=`` (a raw ``[m, n]`` atom tensor). Returns the un-weighted penalty (the
    strategy multiplies by ``lambda_coherence``). Raises if neither is supplied
    (anti-facade #16).
    """

    def __init__(self, penalty: str = "smooth_max", beta: float = DEFAULT_COHERENCE_BETA) -> None:
        super().__init__()
        beta = float(beta)
        validate_coherence_knobs(penalty, beta)
        self.penalty = penalty
        self.beta = beta

    def penalty_offset(self, num_atoms: int) -> float:
        """Constant ``log(N)/beta`` the smooth-max sits above ``max G^2`` at most (0 for Frobenius)."""
        if self.penalty != "smooth_max" or num_atoms < 2:
            return 0.0
        return math.log(num_atoms * (num_atoms - 1) / 2) / self.beta

    def _penalty(self, atoms: torch.Tensor) -> torch.Tensor:
        if self.penalty == "smooth_max":
            return smooth_max_coherence_penalty(atoms, self.beta)
        return gram_offdiagonal_penalty(atoms)

    @staticmethod
    def _frame_atoms(frame: nn.Module) -> torch.Tensor:
        # Attribute access would be typed ``Tensor | Module`` by ``nn.Module.__getattr__``.
        atoms_fn = getattr(frame, "atoms")  # noqa: B009
        atoms: torch.Tensor = atoms_fn()
        return atoms

    def forward(
        self,
        prediction: Any = None,
        target: Any = None,
        *,
        frame: nn.Module | None = None,
        atoms: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if isinstance(frame, nn.Module) and hasattr(frame, "atoms"):
            return self._penalty(self._frame_atoms(frame))
        if isinstance(atoms, torch.Tensor):
            return self._penalty(atoms)
        # A positional module (a strategy may pass the frame first-positionally).
        if isinstance(prediction, nn.Module) and hasattr(prediction, "atoms"):
            return self._penalty(self._frame_atoms(prediction))
        raise ValueError(
            "FrameCoherenceLoss is a PARAMETER penalty on a learnable frame's atoms; "
            "it needs frame=<module with atoms()> or atoms=<[m,n] tensor>. "
            "It cannot be used as a (prediction, target) loss — wire it through the "
            "sparse-frame training strategy, not a YAML losses block (#16)."
        )
