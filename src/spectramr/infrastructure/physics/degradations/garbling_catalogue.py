"""Nonlinear post-processing codes as Blackwell garblings (rule R2).

A registry code applied to already-noisy data is a map of that data alone. However
nonlinear it is (clipping, quantisation, an iterative reconstruction), it is a garbling,
so the base experiment dominates it: ``b >= K o b``. By transitivity ``a >= K o b``
whenever ``a >= b``. Nothing else is certified, not even ``K o a`` against ``K o b``,
because a garbling of the better experiment need not dominate the same garbling of the
worse one.

A code qualifies when it reads only its input. ``apply_degradation`` hands it nothing
but the data, a severity and a seed. Each code is still probed: deterministic under a
fixed seed, and not the identity at the chosen severity. ``pi_residual_aliasing`` is
left out, because it needs multi-coil input.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from spectramr.core.metrics.meta_evaluation.blackwell.catalogue import (
    DiagonalExperiment,
    build_catalogue,
)
from spectramr.core.metrics.meta_evaluation.blackwell.order import RelationGraph, certify
from spectramr.core.metrics.meta_evaluation.blackwell.relations import RelationCatalogue
from spectramr.infrastructure.physics.degradations.registry import apply_degradation
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

#: Audited post-noise codes: receiver saturation and quantisation, and a CS reconstruction.
GARBLING_CODES = ("adc_clipping", "adc_quantization", "cs_recon_artifacts")
#: Base experiments drawn from the radial catalogue.
DEFAULT_BASES = ("noise_0.05", "noise_0.1", "noise_0.15", "noise_0.2", "truncation_0.25")


@dataclass(frozen=True)
class GarbledExperiment:
    """``code`` applied with a fixed seed to the data of ``base`` (``code`` None: the base)."""

    name: str
    family: str
    severity: float
    base: DiagonalExperiment
    code: str | None = None
    seed: int = 0


def probe_garbling(code: str, theta: float, seed: int, shape: tuple[int, int]) -> float:
    """Relative change the code makes, after checking it is deterministic; raise otherwise."""
    gen = torch.Generator().manual_seed(2468)
    y = torch.randn((1, 1, *shape), dtype=torch.complex64, generator=gen) + 1.0
    out1 = apply_degradation(code, y, theta, seed=seed)
    out2 = apply_degradation(code, y, theta, seed=seed)
    if not torch.equal(out1, out2):
        raise ValueError(f"{code} at theta={theta}: not deterministic under a fixed seed")
    change = float((out1 - y).abs().norm() / y.abs().norm())
    if change < 1e-6:
        raise ValueError(f"{code} at theta={theta}: leaves the data unchanged")
    return change


class GarblingRealiser:
    """Base displays, then the code on each draw's complex image, then the envelope."""

    def __call__(
        self,
        xk: torch.Tensor,
        exp: GarbledExperiment,
        noise: torch.Tensor,
        rms: float,
        envelope: torch.Tensor,
    ) -> torch.Tensor:
        h, g = (t.to(device=xk.device, dtype=torch.float32) for t in exp.base.transfers())
        y = (h * xk)[None] + (exp.base.noise_std * rms * g) * noise
        if exp.code is not None:
            img = ifft2c(y)[:, None]
            out = apply_degradation(exp.code, img, exp.severity, seed=exp.seed)[:, 0]
            y = fft2c(out.to(torch.complex64))
        filtered = envelope[:, None].to(y.dtype) * y[None]
        return ifft2c(filtered).abs()


def _graph(exps: Sequence[GarbledExperiment]) -> RelationGraph:
    bases = [k for k, e in enumerate(exps) if e.code is None]
    base_graph = certify([exps[k].base for k in bases])
    strict = {(bases[i], bases[j]) for i, j in base_graph.strict}
    equivalent = {(bases[i], bases[j]) for i, j in base_graph.equivalent}
    incomparable = {(bases[i], bases[j]) for i, j in base_graph.incomparable}
    above = {k: {bases[i] for i, j in base_graph.strict if bases[j] == k} | {k} for k in bases}
    for i, j in base_graph.equivalent:
        above[bases[i]].add(bases[j])
        above[bases[j]].add(bases[i])
    base_of = {exps[k].base.name: k for k in bases}
    for c, e in enumerate(exps):
        if e.code is not None:
            for a in above[base_of[e.base.name]]:
                strict.add((a, c))
    certified = strict | equivalent | incomparable
    uncertified = {
        (i, j)
        for i in range(len(exps))
        for j in range(i + 1, len(exps))
        if (i, j) not in certified and (j, i) not in certified
    }
    return RelationGraph(
        tuple(sorted(strict)),
        tuple(sorted(equivalent)),
        tuple(sorted(incomparable)),
        tuple(sorted(uncertified)),
    )


def garbling_catalogue(
    contents: Sequence[torch.Tensor],
    *,
    codes: Sequence[str] = GARBLING_CODES,
    thetas: Sequence[float] = (0.5, 1.0),
    bases: Sequence[str] = DEFAULT_BASES,
    garbled_bases: Sequence[str] = ("noise_0.1", "truncation_0.25"),
    seed: int = 0,
) -> RelationCatalogue:
    """The base experiments, plus each code at each theta on each garbled base."""
    if not contents or any(c.dim() != 2 for c in contents):
        raise ValueError("garbling_catalogue needs 2-D (H, W) contents")
    unknown = [c for c in codes if c not in GARBLING_CODES]
    if unknown:
        raise ValueError(f"codes {unknown} are not audited garblings: {GARBLING_CODES}")
    shape = (int(contents[0].shape[-2]), int(contents[0].shape[-1]))
    radial = {e.name: e for e in build_catalogue(shape)}
    missing = [b for b in (*bases, *garbled_bases) if b not in radial]
    if missing or not set(garbled_bases) <= set(bases):
        raise ValueError(
            f"bases must be radial catalogue names; garbled bases among them: {missing}"
        )
    exps = [GarbledExperiment(b, radial[b].family, radial[b].severity, radial[b]) for b in bases]
    entries = []
    for code in codes:
        for theta in thetas:
            change = probe_garbling(code, theta, seed, shape)
            for b in garbled_bases:
                exps.append(
                    GarbledExperiment(f"{code}_{theta:g}({b})", code, theta, radial[b], code, seed)
                )
            entries.append({"code": code, "theta": theta, "probe_change": change})
    provenance: dict[str, object] = {
        "catalogue": "garbling_post_noise",
        "codes": list(codes),
        "thetas": list(thetas),
        "bases": list(bases),
        "garbled_bases": list(garbled_bases),
        "seed": seed,
        "rule": "a >= K o b when a >= b (R2); K o a vs K o b uncertified",
        "entries": entries,
    }
    return RelationCatalogue(
        experiments=exps,
        graph=_graph(exps),
        realiser=lambda n_draws, s: GarblingRealiser(),
        provenance=provenance,
    )


__all__ = [
    "DEFAULT_BASES",
    "GARBLING_CODES",
    "GarbledExperiment",
    "GarblingRealiser",
    "garbling_catalogue",
    "probe_garbling",
]
