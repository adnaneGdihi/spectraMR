"""Multi-coil Cartesian experiments: Fisher operators, relations and the Lanczos witness.

Coil ``c`` observes ``M F S_c x + sigma n_c`` (white noise per coil and per acquired
coefficient), so ``J = sum_c S_c^H F^H M F S_c / sigma^2``. That operator is not
diagonal in k-space. Two certificates are rigorous anyway:

* **dominance, structurally.** If the per-coefficient weights ``M / sigma^2`` are ordered
  elementwise, then ``J_a - J_b = sum_c S_c^H F^H diag(w_a - w_b) F S_c`` is positive
  semidefinite. A Lanczos Ritz value could not show this, since it can miss an
  eigenvalue.
* **incomparability, by witness.** Real vectors ``u, v`` with ``u^T J_a u > u^T J_b u`` and
  ``v^T J_a v < v^T J_b v`` (Theorem C1.2). They are searched for by Lanczos on the real
  symmetric operator ``Re(J_a - J_b)``, because the contents are real images and a
  complex witness would certify incomparability over complex images only. Each
  quadratic form is recomputed in float64 and accepted only past a relative margin,
  since ``fft2c`` works in complex64.

A **random mask** acquires phase-encode line ``k`` independently with probability ``p_k``,
and the reader sees which mask was drawn (Theorem C1.4). Dominance is then certified by
coupling: if ``p >= p'`` per line and ``sigma <= sigma'``, thinning the denser mask
reproduces the sparser law exactly. Incomparability uses the expected Fisher operator
(lines weighted by ``p_k``), because ``a >= b`` forces ``E J_a >= E J_b``, so a witness of
each sign refutes both directions.

Pairs that are neither are **uncertified**. With coil overlap, a pair of non-nested masks
can be comparable (Theorem C1.4(c)), so the elementwise order alone would overclaim
incomparability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigsh

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c, sense_adjoint, sense_forward

from ..simulator import _synth_smaps
from .order import Relation, RelationGraph, compare

#: Relative margin a witness quadratic form must clear (complex64 round-off ~1e-6).
WITNESS_MARGIN = 1e-4
#: Lanczos iteration cap per search direction.
WITNESS_MAXITER = 300


@dataclass(frozen=True)
class CoilExperiment:
    """A Cartesian sampling mask ``(H, W)`` and the per-coil noise ``sigma``.

    With ``random`` the mask holds each line's acquisition probability (constant along
    the readout) and a fresh mask is drawn per draw. Otherwise it is 0/1.
    """

    name: str
    family: str
    severity: float
    mask: torch.Tensor  # (H, W) float64
    sigma: float
    random: bool = False

    def log_weights(self) -> torch.Tensor:
        """``log(M / sigma^2)``, ``-inf`` off the mask: the structural order's profile."""
        out = torch.full_like(self.mask, -math.inf)
        out[self.mask > 0] = -2.0 * math.log(self.sigma)
        return out


def coil_maps(
    shape: tuple[int, int], n_coils: int, device: torch.device | None = None
) -> torch.Tensor:
    """Shared synthetic maps ``(1, C, H, W)``; shape-only and normalised, ``sum |S|^2 = 1``."""
    return _synth_smaps(torch.zeros(shape, device=device), n_coils=n_coils)


def quadratic(v: torch.Tensor, exp: CoilExperiment, smaps: torch.Tensor) -> float:
    """``v^T J v = sum_k M_k |(F S v)_k|^2 / sigma^2`` for a real image ``v``, in float64.

    ``M`` enters once: for a random mask that is the expected operator (weights ``p_k``).
    """
    k = sense_forward(v[None, None].float(), smaps).abs().double().pow(2)
    return float((k * exp.mask.to(smaps.device)).sum()) / exp.sigma**2


def _apply(v: torch.Tensor, exp: CoilExperiment, smaps: torch.Tensor) -> torch.Tensor:
    k = sense_forward(v[None, None], smaps) * exp.mask.to(smaps.device, torch.float32)
    return sense_adjoint(k, smaps)[0, 0] / exp.sigma**2


def _signs(a: CoilExperiment, b: CoilExperiment, smaps: torch.Tensor) -> tuple[bool, bool]:
    """Whether real witnesses of ``u^T (J_a - J_b) u > 0`` and ``< 0`` were found."""
    shape = tuple(a.mask.shape)
    n = shape[0] * shape[1]

    def matvec(x: np.ndarray) -> np.ndarray:
        v = torch.from_numpy(np.asarray(x, dtype=np.float32).reshape(shape)).to(smaps.device)
        d = (_apply(v, a, smaps) - _apply(v, b, smaps)).real
        return d.double().cpu().numpy().reshape(-1)

    op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    v0 = np.random.default_rng(0).standard_normal(n)
    found = {1: False, -1: False}
    for which in ("LA", "SA"):
        try:
            _, vecs = eigsh(op, k=1, which=which, v0=v0, maxiter=WITNESS_MAXITER, tol=1e-3)
        except ArpackNoConvergence as exc:  # any vector with a margined sign still witnesses
            vecs = exc.eigenvectors
        except ArpackError:  # e.g. a zero operator: no witness, so the pair stays uncertified
            continue
        for col in np.asarray(vecs).reshape(n, -1).T:
            u = torch.from_numpy(col.reshape(shape).astype(np.float32)).to(smaps.device)
            qa, qb = quadratic(u, a, smaps), quadratic(u, b, smaps)
            if abs(qa - qb) > WITNESS_MARGIN * max(qa, qb):
                found[1 if qa > qb else -1] = True
    return found[1], found[-1]


def _coupled(a: CoilExperiment, b: CoilExperiment) -> Relation:
    """Coupling order of mask laws: line probabilities and noise ordered together."""
    a_ge = bool((a.mask >= b.mask).all()) and a.sigma <= b.sigma
    b_ge = bool((b.mask >= a.mask).all()) and b.sigma <= a.sigma
    if a_ge and b_ge:
        return Relation.EQUIVALENT
    if a_ge or b_ge:
        return Relation.DOMINATES if a_ge else Relation.DOMINATED
    return Relation.INCOMPARABLE


def certify_coils(experiments: Sequence[CoilExperiment], smaps: torch.Tensor) -> RelationGraph:
    """Structural dominance, witnessed incomparability, and the uncertified rest."""
    strict, equivalent, incomparable, uncertified = [], [], [], []
    for i in range(len(experiments)):
        for j in range(i + 1, len(experiments)):
            a, b = experiments[i], experiments[j]
            if a.random or b.random:
                rel = _coupled(a, b)
            else:
                rel = compare(a.log_weights(), b.log_weights())
            if rel is Relation.DOMINATES:
                strict.append((i, j))
            elif rel is Relation.DOMINATED:
                strict.append((j, i))
            elif rel is Relation.EQUIVALENT:
                equivalent.append((i, j))
            elif all(_signs(a, b, smaps)):
                incomparable.append((i, j))
            else:
                uncertified.append((i, j))
    return RelationGraph(tuple(strict), tuple(equivalent), tuple(incomparable), tuple(uncertified))


class CoilRealiser:
    """Displays for :func:`~.evaluate.metric_table`: a normalised SENSE combine, magnitude.

    The combine is a fixed linear map of the data, so it is a post-processing kernel and
    the experiment's information is unchanged. The Gaussian envelope then filters the
    combined image in k-space, which is again a composition-closed class on the data.
    """

    def __init__(self, smaps: torch.Tensor, n_draws: int = 1, seed: int = 0) -> None:
        self.smaps = smaps
        # One uniform per draw and line, shared by every experiment and content: with
        # ordered probabilities, every draw's masks are nested (the coupling itself).
        gen = torch.Generator().manual_seed(seed)
        self.uniform = torch.rand((n_draws, smaps.shape[-2], 1), generator=gen)

    def __call__(
        self,
        xk: torch.Tensor,
        exp: CoilExperiment,
        noise: torch.Tensor,
        rms: float,
        envelope: torch.Tensor,
    ) -> torch.Tensor:
        smaps = self.smaps.to(xk.device)
        mask = exp.mask.to(xk.device, torch.float32)
        if exp.random:
            draws = noise.shape[0]
            mask = (self.uniform[:draws].to(xk.device) < mask[:, :1]).float()[:, None]
        x = ifft2c(xk)
        y = sense_forward(x[None, None], smaps) * mask + (exp.sigma * rms) * mask * noise
        combined = sense_adjoint(y, smaps, normalize=True)[:, 0]  # (R, H, W)
        filtered = envelope[:, None].to(combined.dtype) * fft2c(combined)[None]
        return ifft2c(filtered).abs()


__all__ = [
    "WITNESS_MARGIN",
    "CoilExperiment",
    "CoilRealiser",
    "certify_coils",
    "coil_maps",
    "quadratic",
]
