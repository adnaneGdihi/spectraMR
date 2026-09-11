r"""Learnable sparsifying tight frame with coherence optimisation (A-6.5 *SCO-Frame*).

The framework has *fixed*-transform sparsity priors (``wavelet_sparsity_l1``,
``kspace_l1_sparsity`` in ``models/losses/regularizers.py``) but no **learnable**
sparsifying transform whose atoms are optimised for **mutual incoherence**. Mutual
incoherence is a classical sparse-recovery principle — for a *fixed* dictionary, a
low coherence :math:`\mu` certifies unique :math:`\ell_1` recovery up to sparsity
:math:`s<\tfrac12(1+1/\mu)` (Donoho-Elad). **That theorem does NOT transfer to a
learned, task-specific conv frame trained end-to-end with soft-thresholding** (not
:math:`\ell_1` minimisation) on real anatomy: here incoherence is an *empirical*
regulariser, and the testable claim is the ablation (does ``lambda_coherence>0``
lower the learned :math:`\mu` at matched reconstruction?), not a recovery guarantee.

A :class:`TightFrameLearner` is a convolutional analysis/synthesis pair
:math:`(D, D^\top)` (an over-complete filter bank). ``analyze`` applies the
``Conv2d`` filter bank; ``synthesize`` applies its exact adjoint via
``conv_transpose2d`` *with the same weight* — ``conv_transpose2d`` is the adjoint
of ``conv2d`` for **stride 1 with matching padding** (the configuration used here;
a strided variant would need ``output_padding`` to stay adjoint). ``forward`` is a
sparse-coding autoencoder pass: analyse -> soft-threshold (sparsify) -> synthesise.

Two structural penalties, on the atom matrix :math:`W\in\mathbb{R}^{m\times n}`
(``m`` atoms, each a flattened :math:`n=C\,k^2` filter):

* **Parseval tightness** :math:`\lVert W^\top W - (m/n) I_n\rVert_F^2` — a
  penalty on the *atom-matrix* Gram. This is a **heuristic proxy** for tightness
  of the full convolutional frame operator :math:`D D^\top` (the overlapping-patch
  conv structure adds energy couplings the atom Gram cannot fully control), so it
  *encourages* but does not *guarantee* an energy-preserving round-trip. The
  adjoint property of the pair is exact (verified by test); the tight-round-trip is
  the optimisation target, not a proven invariant.
* **Mutual coherence** :math:`\lVert \tilde G - I_m\rVert_F^2` with
  :math:`\tilde G = \hat W\hat W^\top` the Gram of the *unit-normalised* atoms —
  the off-diagonal Frobenius penalty that drives the atoms apart. The diagnostic
  :math:`\mu = \max_{i\neq j}|\tilde G_{ij}|` is the classical mutual coherence.

Tightness and incoherence are **distinct** objectives (a tight frame can still be
coherent), which is exactly why the A-6.5 one-knob ablation isolates the coherence
penalty (``frame_coherence`` loss) from the always-on tightness term.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.registry import register_model


@register_model(
    name="tight_frame_learner",
    training_mode="sparse_frame",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
)
class TightFrameLearner(nn.Module):
    r"""Learnable convolutional sparsifying frame (A-6.5 SCO-Frame).

    A sparse-coding autoencoder over a learned over-complete filter bank, with
    Parseval-tightness and mutual-coherence penalties on the atoms.

    Args:
        in_channels: Image channels (== ``out_channels``; this is an autoencoder).
        out_channels: Must equal ``in_channels``.
        num_atoms: Number of frame atoms (filters); over-complete when
            ``num_atoms > in_channels * kernel_size**2``.
        kernel_size: Odd filter size (stride-1, ``same`` padding, so the frame is
            a redundant filter bank that preserves spatial size).
        sparsity_threshold: Soft-threshold level applied to the analysis
            coefficients in ``forward`` (the sparsifying non-linearity).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        num_atoms: int = 64,
        kernel_size: int = 7,
        sparsity_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                "TightFrameLearner is a sparse autoencoder (in==out); "
                f"got in={in_channels}, out={out_channels}."
            )
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd (same-padding), got {kernel_size}.")
        self.in_channels = int(in_channels)
        self.num_atoms = int(num_atoms)
        self.kernel_size = int(kernel_size)
        self._pad = kernel_size // 2
        self.sparsity_threshold = float(sparsity_threshold)
        # Analysis filter bank D: [num_atoms, in_channels, k, k]. Synthesis reuses
        # this weight via conv_transpose2d (the adjoint D^T) — TIED weights.
        self.analysis = nn.Conv2d(
            in_channels, num_atoms, kernel_size, padding=self._pad, bias=False
        )

    def atoms(self) -> torch.Tensor:
        """The atom matrix ``W`` of shape ``[num_atoms, in_channels*k*k]``."""
        return self.analysis.weight.reshape(self.num_atoms, -1)

    def analyze(self, x: torch.Tensor) -> torch.Tensor:
        """Analysis coefficients ``D x`` of shape ``[B, num_atoms, H, W]``."""
        return self.analysis(x)

    def _soft_threshold(self, c: torch.Tensor) -> torch.Tensor:
        return torch.sign(c) * torch.relu(c.abs() - self.sparsity_threshold)

    def synthesize(self, c: torch.Tensor) -> torch.Tensor:
        """Synthesis ``D^T c`` (exact adjoint of ``analyze``) -> ``[B, in, H, W]``."""
        return F.conv_transpose2d(c, self.analysis.weight, padding=self._pad)

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        if torch.is_complex(x):
            raise ValueError("TightFrameLearner expects magnitude (real) input.")
        coeffs = self._soft_threshold(self.analyze(x))
        return self.synthesize(coeffs)

    def parseval_penalty(self) -> torch.Tensor:
        r"""Frame-tightness penalty :math:`\lVert W^\top W - (m/n) I_n\rVert_F^2`."""
        w = self.atoms()
        n_feat = w.shape[1]
        gram = w.t() @ w  # [n, n]
        scale = self.num_atoms / n_feat
        eye = torch.eye(n_feat, device=w.device, dtype=w.dtype)
        return ((gram - scale * eye) ** 2).sum()

    def coherence_penalty(self) -> torch.Tensor:
        r"""Mutual-coherence penalty :math:`\sum_{i\neq j}\tilde G_{ij}^2` (Gram off-diagonal)."""
        w = self.atoms()
        wn = w / (w.norm(dim=1, keepdim=True) + 1e-12)
        gram = wn @ wn.t()  # [m, m], unit diagonal
        off = gram - torch.eye(self.num_atoms, device=gram.device, dtype=gram.dtype)
        return (off**2).sum()

    @torch.no_grad()
    def mutual_coherence(self) -> float:
        r"""Diagnostic :math:`\mu = \max_{i\neq j}|\tilde G_{ij}|` (classical mutual coherence)."""
        if self.num_atoms < 2:
            return 0.0  # no off-diagonal pairs for a single atom -> coherence is 0 by convention
        w = self.atoms()
        wn = w / (w.norm(dim=1, keepdim=True) + 1e-12)
        gram = (wn @ wn.t()).abs()  # diagonal == 1, off-diagonals in [0, 1]
        # Zero the diagonal so max() returns the largest OFF-diagonal coherence.
        off = gram * (1.0 - torch.eye(self.num_atoms, device=gram.device, dtype=gram.dtype))
        return float(off.max())
