"""Phase-only registry degradations as Blackwell experiments, by unitary absorption.

Motion shifts, drifts, eddy currents and ghosts act on the signal as a k-space phase:
``T = F^H diag(e^{i phi}) F``, a unitary map, applied before white noise. Because white
noise is invariant under a unitary map, ``T x + n`` has the law of ``T (x + n)``
(Theorem C1.6(d)). Two cases follow:

* the phase does not depend on the seed: ``T`` is a fixed invertible map, and undoing it
  is a kernel that does not depend on ``x``. The experiment is Blackwell-equivalent to
  noise alone at the same ``sigma``, which makes it a test of display sensitivity;
* the phase depends on the seed (an unobserved trajectory drawn per scan): the mixture
  over seeds is a garbling of noise alone. ``noise_s`` therefore dominates it for every
  ``s <= sigma``. That is a weak dominance, which the violation test reads correctly.

Nothing else is certified: a mixture against noisier experiments, or against another
mixture, is uncertified. Every code is probed to be linear, diagonal and unit-modulus,
first when the catalogue is built and again for each seed a realiser uses.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from spectramr.core.metrics.meta_evaluation.blackwell.order import RelationGraph
from spectramr.core.metrics.meta_evaluation.blackwell.relations import RelationCatalogue
from spectramr.infrastructure.physics.degradations.registry import apply_degradation
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Codes that probe as unit-modulus diagonal k-space operators. ``chemical_shift`` is not
#: one: it adds a shifted fat copy to the water signal, so ``|h|`` varies by about 0.3.
UNITARY_CODES = (
    "b0_drift",
    "eddy",
    "mechanical_vibration",
    "nyquist_ghost",
    "random_motion",
    "sudden_motion",
    "translation_drift",
)
PROBE_TOL = 1e-4


@dataclass(frozen=True)
class NuisanceExperiment:
    """Noise alone (``code is None``) or a phase-only code at ``severity``, then noise."""

    name: str
    family: str
    severity: float
    code: str | None
    sigma: float
    seeded: bool = False


def probe_unitary(code: str, theta: float, seed: int, shape: tuple[int, int]) -> torch.Tensor:
    """The complex unit-modulus transfer of ``code``, or raise if it is not one."""
    gen = torch.Generator().manual_seed(4321)
    x1, x2 = (torch.randn((1, 1, *shape), dtype=torch.complex64, generator=gen) for _ in "ab")

    def op(x: torch.Tensor) -> torch.Tensor:
        return apply_degradation(code, x, theta, seed=seed)

    y1, y2, y12 = op(x1), op(x2), op(x1 + x2)
    h1 = (fft2c(y1) / fft2c(x1))[0, 0].to(torch.complex128)
    h2 = (fft2c(y2) / fft2c(x2))[0, 0].to(torch.complex128)
    res = {
        "linearity": float((y12 - y1 - y2).norm() / y12.norm()),
        "diagonality": float((h1 - h2).abs().max()),
        "modulus": float((h1.abs() - 1.0).abs().max()),
    }
    bad = {k: v for k, v in res.items() if not v <= PROBE_TOL}
    if bad:
        raise ValueError(f"{code} at theta={theta}, seed={seed}: not a unitary phase, {bad}")
    return (h1 + h2) / 2


class NuisanceRealiser:
    """Displays with the nuisance drawn per draw: seed ``seed + r`` on draw ``r``.

    The same seeds serve every experiment and content, so draws are matched across them.
    Each seed's transfer is probed once on first use.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self._checked: set[tuple[str, float, int, tuple[int, int]]] = set()

    def _signal(self, x: torch.Tensor, exp: NuisanceExperiment, r: int) -> torch.Tensor:
        assert exp.code is not None
        seed = self.seed + r if exp.seeded else self.seed
        key = (exp.code, exp.severity, seed, (int(x.shape[-2]), int(x.shape[-1])))
        if key not in self._checked:
            probe_unitary(exp.code, exp.severity, seed, key[3])
            self._checked.add(key)
        y = apply_degradation(exp.code, x[None, None].to(torch.complex64), exp.severity, seed=seed)
        return fft2c(y[0, 0])

    def __call__(
        self,
        xk: torch.Tensor,
        exp: NuisanceExperiment,
        noise: torch.Tensor,
        rms: float,
        envelope: torch.Tensor,
    ) -> torch.Tensor:
        draws = noise.shape[0]
        if exp.code is None:
            k = xk.expand(draws, *xk.shape)
        else:
            x = ifft2c(xk)
            k = torch.stack([self._signal(x, exp, r) for r in range(draws)])
        y = k + (exp.sigma * rms) * noise
        filtered = envelope[:, None].to(y.dtype) * y[None]
        return ifft2c(filtered).abs()


def _graph(exps: Sequence[NuisanceExperiment], base_noise: float) -> RelationGraph:
    strict: set[tuple[int, int]] = set()
    equivalent: set[tuple[int, int]] = set()
    uncertified: set[tuple[int, int]] = set()
    for i, a in enumerate(exps):
        for j, b in enumerate(exps):
            if i >= j:
                continue
            na, nb = a.code is None, b.code is None
            if na and nb:  # the noise ladder
                strict.add((i, j) if a.sigma < b.sigma else (j, i))
                continue
            if not na and not nb:
                both_fixed = not a.seeded and not b.seeded
                (equivalent if both_fixed else uncertified).add((i, j))
                continue
            noise_i, mix_i = (i, j) if na else (j, i)
            noise, mix = exps[noise_i], exps[mix_i]
            if not mix.seeded and math.isclose(noise.sigma, base_noise):
                equivalent.add((min(i, j), max(i, j)))
            elif noise.sigma <= base_noise:
                strict.add((noise_i, mix_i))  # noise_s >= fixed T (s < sigma) or a mixture
            elif not mix.seeded:
                strict.add((mix_i, noise_i))  # a fixed T at sigma beats noisier noise
            else:
                uncertified.add((min(i, j), max(i, j)))
    # A fixed phase is equivalent to noise at sigma, which dominates every mixture.
    for i, a in enumerate(exps):
        for j, b in enumerate(exps):
            if a.code and b.code and not a.seeded and b.seeded:
                uncertified.discard((min(i, j), max(i, j)))
                strict.add((i, j))
    return RelationGraph(
        tuple(sorted(strict)), tuple(sorted(equivalent)), (), tuple(sorted(uncertified))
    )


def nuisance_catalogue(
    contents: Sequence[torch.Tensor],
    *,
    codes: Sequence[str] = UNITARY_CODES,
    thetas: Sequence[float] = (0.5, 1.0),
    base_noise: float = 0.1,
    noise_levels: Sequence[float] = (0.05, 0.1, 0.15, 0.2),
    seed: int = 0,
) -> RelationCatalogue:
    """The noise ladder plus each phase-only code at each theta, at ``base_noise``."""
    if not contents or any(c.dim() != 2 for c in contents):
        raise ValueError("nuisance_catalogue needs 2-D (H, W) contents")
    unknown = [c for c in codes if c not in UNITARY_CODES]
    if unknown:
        raise ValueError(f"codes {unknown} are not audited phase-only codes: {UNITARY_CODES}")
    shape = (int(contents[0].shape[-2]), int(contents[0].shape[-1]))
    exps = [NuisanceExperiment(f"noise_{s:g}", "noise", s, None, s) for s in noise_levels]
    entries = []
    for code in codes:
        for theta in thetas:
            h0 = probe_unitary(code, theta, seed, shape)
            h1 = probe_unitary(code, theta, seed + 1, shape)
            seeded = float((h0 - h1).abs().max()) > PROBE_TOL
            exps.append(
                NuisanceExperiment(f"{code}_{theta:g}", code, theta, code, base_noise, seeded)
            )
            entries.append({"name": exps[-1].name, "seeded": seeded})
    provenance: dict[str, object] = {
        "catalogue": "nuisance_unitary",
        "seed": seed,
        "thetas": list(thetas),
        "base_noise": base_noise,
        "noise_levels": list(noise_levels),
        "rule": "unitary absorption (Theorem C1.6(d)); seeded codes enter as weak dominance",
        "entries": entries,
    }
    return RelationCatalogue(
        experiments=exps,
        graph=_graph(exps, base_noise),
        realiser=lambda n_draws, s: NuisanceRealiser(s),
        provenance=provenance,
    )


__all__ = [
    "UNITARY_CODES",
    "NuisanceExperiment",
    "NuisanceRealiser",
    "nuisance_catalogue",
    "probe_unitary",
]
