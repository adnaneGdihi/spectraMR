"""The dimensionally typed term library and the symbolic head of ``symbolic_regression``.

Every term has the normal form ``v * prod_k exp(s * a_k / b_k)``: a prefactor variable times at
most ``MAX_DECAY_FACTORS`` exponential factors of variable ratios, with ``s = -1`` when ``neg`` is
an allowed operator. Which forms exist is decided by ``allowed_operators`` (``exp`` and ``div`` make
the exponential factors, ``mul`` joins factors, ``add`` sums terms) and, under
``dimensional_consistency``, by the declared units: an exponent must be dimensionless and every
term must carry ``OUTPUT_UNIT``. Spin-echo and inversion-recovery signal terms such as
``M0 * exp(-TE/T2)`` and ``M0 * exp(-TR/T1) * exp(-TE/T2)`` are members of the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import combinations
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.generators.option_parsing import parse_enum_option

__all__ = [
    "MAX_DECAY_FACTORS",
    "OUTPUT_UNIT",
    "SUPPORTED_OPERATORS",
    "SUPPORTED_UNITS",
    "PhysicsSymbolicHead",
    "SymbolicHeadType",
    "SymbolicSpec",
    "build_symbolic_spec",
]


class SymbolicHeadType(Enum):
    PHYSICS_PYSR = "physicspysr"


SUPPORTED_OPERATORS = frozenset({"exp", "add", "mul", "div", "neg"})
SUPPORTED_UNITS = frozenset({"time", "magnetization", "dimensionless"})
#: The reconstructed image is a signal intensity, so every term carries this unit.
OUTPUT_UNIT = "magnetization"
MAX_DECAY_FACTORS = 2
#: Bound on the exponent magnitude so a growing factor (no ``neg``) cannot overflow float32.
EXPONENT_CLAMP = 30.0
_EPS = 1e-6


@dataclass(frozen=True)
class SymbolicSpec:
    """The validated head declaration and the enumerated term library."""

    head_type: SymbolicHeadType
    variables: tuple[str, ...]
    units: tuple[str, ...]
    operators: tuple[str, ...]
    dimensional_consistency: bool
    num_terms: int
    ratios: tuple[tuple[int, int], ...]
    terms: tuple[tuple[int, tuple[int, ...]], ...]

    def expression(self, index: int) -> str:
        prefactor, decays = self.terms[index]
        sign = "-" if "neg" in self.operators else ""
        factors = [self.variables[prefactor]] if prefactor >= 0 else []
        for r in decays:
            a, b = self.ratios[r]
            factors.append(f"exp({sign}{self.variables[a]}/{self.variables[b]})")
        return "*".join(factors)

    def record(self) -> dict[str, Any]:
        return {
            "type": "PhysicsPySR",
            "allowed_operators": list(self.operators),
            "dimensional_consistency": self.dimensional_consistency,
            "units": dict(zip(self.variables, self.units, strict=True)),
            "output_unit": OUTPUT_UNIT if self.dimensional_consistency else None,
            "num_symbolic_terms": self.num_terms,
            "max_decay_factors": MAX_DECAY_FACTORS,
            "library_size": len(self.terms),
            "library": [self.expression(i) for i in range(len(self.terms))],
            "selection": "top-k |coefficient|, straight-through",
        }


def _validate_units(constraints: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    unknown = sorted(set(constraints) - {"dimensional_consistency", "units"})
    if unknown:
        raise ValueError(f"symbolic_regression constraints keys {unknown} are not read.")
    consistency = constraints.get("dimensional_consistency")
    if not isinstance(consistency, bool):
        raise ValueError(
            f"constraints.dimensional_consistency must be true or false, got {consistency!r}."
        )
    units = constraints.get("units")
    if not isinstance(units, dict) or not units:
        raise ValueError("constraints.units must map each symbolic variable to its unit.")
    for name, unit in units.items():
        if unit not in SUPPORTED_UNITS:
            raise ValueError(
                f"constraints.units.{name} '{unit}' is unknown; choose one of "
                f"{sorted(SUPPORTED_UNITS)}."
            )
    return consistency, {str(k): str(v) for k, v in units.items()}


def build_symbolic_spec(
    head_type: object,
    constraints: dict[str, Any] | None,
    allowed_operators: list[str] | None,
    num_terms: int,
) -> SymbolicSpec:
    """Validate the declaration and enumerate the term library, raising on anything unread."""
    kind = parse_enum_option(SymbolicHeadType, head_type, "model_kwargs.type")
    if not isinstance(constraints, dict):
        raise ValueError("symbolic_regression requires model_kwargs.constraints with units.")
    consistency, units = _validate_units(constraints)
    if not allowed_operators:
        raise ValueError("symbolic_regression requires a non-empty model_kwargs.allowed_operators.")
    unknown_ops = sorted(set(allowed_operators) - SUPPORTED_OPERATORS)
    if unknown_ops:
        raise ValueError(
            f"allowed_operators {unknown_ops} are not implemented; choose from "
            f"{sorted(SUPPORTED_OPERATORS)}."
        )
    if isinstance(num_terms, bool) or not isinstance(num_terms, int) or num_terms < 1:
        raise ValueError(f"num_symbolic_terms must be a positive integer, got {num_terms!r}.")
    ops = frozenset(allowed_operators)
    if num_terms > 1 and "add" not in ops:
        raise ValueError(
            "num_symbolic_terms > 1 sums terms, which needs 'add' in allowed_operators."
        )

    names, kinds = tuple(units), tuple(units.values())
    idx = range(len(names))
    ratios: tuple[tuple[int, int], ...] = ()
    if {"exp", "div"} <= ops:
        ratios = tuple(
            (a, b) for a in idx for b in idx if a != b and (not consistency or kinds[a] == kinds[b])
        )
    if consistency:
        prefactors = [i for i in idx if kinds[i] == OUTPUT_UNIT]
        if not prefactors:
            raise ValueError(
                f"dimensional_consistency needs a variable with unit '{OUTPUT_UNIT}' for the "
                f"signal terms; constraints.units declares none."
            )
    else:
        prefactors = list(idx)
    max_factors = MAX_DECAY_FACTORS if "mul" in ops else 0
    terms: list[tuple[int, tuple[int, ...]]] = []
    for p in prefactors:
        for n in range(max_factors + 1):
            terms.extend((p, decays) for decays in combinations(range(len(ratios)), n))
    if not consistency and "mul" not in ops:
        terms.extend((-1, (r,)) for r in range(len(ratios)))
    if len(terms) < num_terms:
        raise ValueError(
            f"num_symbolic_terms={num_terms} exceeds the {len(terms)} terms the declared "
            f"operators and units admit."
        )
    return SymbolicSpec(
        kind, names, kinds, tuple(sorted(ops)), consistency, num_terms, ratios, tuple(terms)
    )


class PhysicsSymbolicHead(nn.Module):
    """Per-pixel latent variables combined through the K selected library terms."""

    def __init__(self, in_dim: int, spec: SymbolicSpec) -> None:
        super().__init__()
        self.spec = spec
        n_vars, n_terms = len(spec.variables), len(spec.terms)
        self.num_terms = spec.num_terms
        self.to_variables = nn.Conv2d(in_dim, n_vars, 1)
        self.coefficients = nn.Parameter(torch.randn(n_terms) * 0.01)
        ratio = torch.zeros(max(len(spec.ratios), 1), n_vars)
        for r, (a, b) in enumerate(spec.ratios):
            ratio[r, a], ratio[r, b] = 1.0, -1.0
        prefactor = torch.zeros(n_terms, n_vars)
        decay = torch.zeros(n_terms, ratio.shape[0])
        for t, (p, decays) in enumerate(spec.terms):
            if p >= 0:
                prefactor[t, p] = 1.0
            for r in decays:
                decay[t, r] = 1.0
        self.register_buffer("ratio_matrix", ratio, persistent=False)
        self.register_buffer("prefactor_matrix", prefactor, persistent=False)
        self.register_buffer("decay_matrix", decay, persistent=False)
        self.exponent_sign = -1.0 if "neg" in spec.operators else 1.0

    def selection_mask(self) -> torch.Tensor:
        top = torch.topk(self.coefficients.detach().abs(), self.num_terms).indices
        return torch.zeros_like(self.coefficients).scatter_(0, top, 1.0)

    def terms(self, h: torch.Tensor) -> torch.Tensor:
        """``[B, T, H, W]`` values of every library term, batched as two einsums and one exp."""
        log_v = torch.log(F.softplus(self.to_variables(h)) + _EPS)
        ratios = torch.exp(torch.einsum("rv,bvhw->brhw", self.ratio_matrix, log_v))
        exponent = self.exponent_sign * torch.einsum("tr,brhw->bthw", self.decay_matrix, ratios)
        exponent = exponent.clamp(-EXPONENT_CLAMP, EXPONENT_CLAMP)
        log_prefactor = torch.einsum("tv,bvhw->bthw", self.prefactor_matrix, log_v)
        return torch.exp(log_prefactor + exponent)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        mask = self.selection_mask()
        c = self.coefficients
        # The forward value uses only the K selected terms; every coefficient keeps a gradient.
        effective = mask * c + (1.0 - mask) * (c - c.detach())
        return torch.einsum("t,bthw->bhw", effective, self.terms(h)).unsqueeze(1)

    def selected_expressions(self) -> list[str]:
        """The currently selected terms as strings (host read; call outside the training loop)."""
        top = torch.topk(self.coefficients.detach().abs(), self.num_terms).indices.tolist()
        return [self.spec.expression(i) for i in sorted(top)]
