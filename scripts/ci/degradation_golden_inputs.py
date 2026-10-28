"""The golden gate's declared exemption table and the side inputs its hashed cases need.

The 2-D gate (``check_degradation_golden.py``) and the registry-wide tests read the same
tables from here, so an axis's exemption kind and its synthetic inputs have one owner.
"""

from __future__ import annotations

import torch

#: (shape, complex?) -- the multi-coil case is real so both input kinds are hashed.
SHAPES = (((2, 1, 64, 64), True), ((1, 4, 48, 40), False), ((1, 1, 33, 31), True))


#: Axes the 2-D hashing must treat specially, one declared set per kind, each asserted equal to
#: the set the registry implies (``exemption_errors``), so neither a wrongly exempted nor an
#: undeclared axis hides. Volume-only axes have no SLICE_2D kernel and get no case;
#: multicoil-only axes are hashed on the multi-coil shapes only; q-map axes get ``KIND_INPUTS``.
VOLUME_ONLY_AXES = frozenset(
    {
        "inflow_enhancement",
        "slice_crosstalk",
        "through_plane_pve",
        "spin_history",
        "off_resonance_slice_shift",
    }
)
MULTICOIL_ONLY_AXES = frozenset({"correlated_coil_noise", "coil_dropout", "pi_residual_aliasing"})
QMAPS_AXES = frozenset({"imperfect_inversion", "field_contrast"})


def declared_exemptions() -> dict[str, frozenset[str]]:
    return {
        "volume-only": VOLUME_ONLY_AXES,
        "multicoil-only": MULTICOIL_ONLY_AXES,
        "qmaps": QMAPS_AXES,
    }


def _requirement_kind(req) -> str:
    """The declared kind a ``Requirement`` belongs to; an unlisted one is its own (undeclared) kind."""
    return {"multicoil": "multicoil-only", "qmaps": "qmaps"}.get(req.value, f"requires {req.value}")


def registry_exemptions() -> dict[str, frozenset[str]]:
    """The same kinds, derived from the registry's implementations and requirements."""
    from spectramr.infrastructure.physics.degradations import DEGRADATION_REGISTRY
    from spectramr.infrastructure.physics.degradations.registry_queries import (
        slice_capable_degradations,
    )

    kinds = {
        "volume-only": set(DEGRADATION_REGISTRY) - set(slice_capable_degradations()),
        "multicoil-only": set(),
        "qmaps": set(),
    }
    for name, spec in DEGRADATION_REGISTRY.items():
        for req in spec.requires:
            kinds.setdefault(_requirement_kind(req), set()).add(name)
    return {kind: frozenset(names) for kind, names in kinds.items()}


def exemption_errors(
    required: dict[str, frozenset[str]] | None = None,
    exempt: dict[str, frozenset[str]] | None = None,
) -> list[str]:
    """Per kind, disagreements between the declared set and the registry-derived one.

    Both sides are arguments (default: the registry and the declared sets) so a planted test can
    corrupt either.
    """
    required = registry_exemptions() if required is None else required
    exempt = declared_exemptions() if exempt is None else exempt
    errors = []
    for kind in sorted(required.keys() | exempt.keys()):
        need, have = required.get(kind, frozenset()), exempt.get(kind, frozenset())
        errors += [f"{kind}: wrongly exempted: {n}" for n in sorted(have - need)]
        errors += [f"{kind}: undeclared: {n}" for n in sorted(need - have)]
    return errors


def case_shapes(name: str) -> tuple[tuple[int, ...], ...]:
    """The input shapes an axis is hashed on: none if volume-only, coil shapes if multicoil-only."""
    if name in VOLUME_ONLY_AXES:
        return ()
    shapes = tuple(shape for shape, _ in SHAPES)
    if name in MULTICOIL_ONLY_AXES:
        return tuple(shape for shape in shapes if shape[1] >= 2)
    return shapes


def misplaced_cases(cases: dict[str, object]) -> list[str]:
    """Registry case keys on a shape their axis is not hashed on (see ``case_shapes``)."""
    bad = []
    for key in cases:
        parts = key.split("|")
        if len(parts) != 4:
            continue  # simulator / tio cases
        shape = tuple(int(v) for v in parts[1].split("x"))
        if shape not in case_shapes(parts[0].split("+")[0]):
            bad.append(key)
    return bad


def _synthetic_qmaps(x: torch.Tensor) -> dict[str, object]:
    """Smooth T1/T2 (ms) over the image grid with a zero-T1 background corner, PD = |x|; no RNG."""
    h, w = x.shape[-2:]
    dtype = x.real.dtype
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=dtype) / h, torch.arange(w, dtype=dtype) / w, indexing="ij"
    )
    t1 = 900.0 + 400.0 * torch.sin(6.2831853 * (xx + yy))
    t2 = 80.0 + 30.0 * torch.cos(6.2831853 * (xx - yy))
    t1[: h // 8, : w // 8] = 0.0
    lead = (1,) * (x.dim() - 2)
    maps = {"t1": t1.view(*lead, h, w).expand(*lead[:2], *x.shape[2:]).clone()}
    maps["t2"] = t2.view(*lead, h, w).expand_as(maps["t1"]).clone()
    return {"qmaps": {**maps, "pd": x.abs()}}


#: The inputs each kind's axes need besides the image, built from the image: the one place
#: registry-wide tests and the hashing get them (``required_kwargs``).
KIND_INPUTS = {"qmaps": _synthetic_qmaps}
#: Scalar kwargs an axis has no default for, passed on every one of its cases.
AXIS_KWARGS: dict[str, dict[str, object]] = {"imperfect_inversion": {"ti_ms": 600.0}}


def required_kwargs(name: str, x: torch.Tensor, builders=None) -> dict[str, object]:
    """The kwargs ``name`` needs besides the image (empty for an axis that requires nothing)."""
    builders = KIND_INPUTS if builders is None else builders
    kwargs: dict[str, object] = {}
    for kind, axes in declared_exemptions().items():
        if name in axes and kind in builders:
            kwargs.update(builders[kind](x))
    return {**kwargs, **AXIS_KWARGS.get(name, {})}


def validate_required_inputs(specs, builders=None, probes=None) -> None:
    """Raise unless the kind builders meet every ``requires`` of ``specs`` on 2-D and 5-D probes.

    Red for: a requirement the builders leave unmet (including a requirement no kind builds); a
    builder for a kind that is not declared; ``AXIS_KWARGS`` for an axis that does not exist.
    """
    from spectramr.infrastructure.physics.degradations.registry import check_requirements

    builders = KIND_INPUTS if builders is None else builders
    probes = probes or (torch.ones(1, 2, 8, 8), torch.ones(1, 2, 3, 8, 8))
    problems = [
        f"builder for a kind that is not declared: {k}"
        for k in sorted(set(builders) - set(declared_exemptions()))
    ]
    problems += [
        f"{n}: AXIS_KWARGS for an axis that does not exist"
        for n in sorted(set(AXIS_KWARGS) - set(specs))
    ]
    for name in sorted(n for n, s in specs.items() if s.requires):
        for x in probes:
            try:
                check_requirements(specs[name], x, required_kwargs(name, x, builders))
            except ValueError as exc:
                problems.append(f"{name}: builders leave {exc} on {tuple(x.shape)}")
    if problems:
        raise RuntimeError(
            "required-input builders disagree with the registry: " + "; ".join(problems)
        )
