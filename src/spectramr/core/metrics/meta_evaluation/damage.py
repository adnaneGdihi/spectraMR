"""Neutral damage yardsticks: how far a degraded image sits from its clean reference.

θ stays each degradation's own range; fairness between degradations comes from comparing
metrics at equal *measured* damage, in bands. A yardstick's circular pool metrics are
disqualified from any ranking built on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

_EPS = 1e-12

#: ``(band, lo, hi)`` per band, in order of increasing damage.
Bands = tuple[tuple[str, float, float], ...]

#: Band names, lightest first; ``bands_from_edges`` builds them from three lower edges.
BAND_NAMES: tuple[str, ...] = ("imperceptible", "light", "moderate", "heavy")

#: Edges are half-open ``[lo, hi)``; the last band is open-ended. ``grad_l2`` runs larger
#: than ``rel_l2``; set its edges from a coverage preview with ``--damage-bands``.
_DEFAULT_BANDS: Bands = (
    ("imperceptible", 0.0, 0.02),
    ("light", 0.02, 0.10),
    ("moderate", 0.10, 0.30),
    ("heavy", 0.30, math.inf),
)
DAMAGE_BANDS: Mapping[str, Bands] = {
    "rel_l2": _DEFAULT_BANDS,
    "grad_l2": _DEFAULT_BANDS,
}
MIN_LEVELS_PER_BAND: int = 3
#: Covered degradations a band needs before it counts in ``overall``; below it the band
#: is reported but one degradation would decide a whole band's share of the score.
MIN_AXES_PER_BAND: int = 2
NON_VOTING_BANDS: frozenset[str] = frozenset({"imperceptible"})


def band_votes(band: str, n_levels: int, min_levels: int = MIN_LEVELS_PER_BAND) -> bool:
    """Whether a degradation with ``n_levels`` levels in ``band`` votes there."""
    return band not in NON_VOTING_BANDS and n_levels >= min_levels


def _magnitude(t: torch.Tensor) -> torch.Tensor:
    return t.abs().to(torch.float32)


#: ``(D, H, W)`` voxel size in mm, in tensor dim order; ``None`` is a unit voxel.
VoxelMM = tuple[float, float, float] | None


def _rel_l2_batch(
    deg: torch.Tensor, clean: torch.Tensor, voxel_mm: VoxelMM = None, volume: bool = False
) -> torch.Tensor:
    num = torch.linalg.vector_norm((deg - clean).flatten(1), dim=1)
    den = torch.linalg.vector_norm(clean.flatten(1), dim=1).clamp_min(_EPS)
    return num / den


def _grad(t: torch.Tensor, voxel_mm: VoxelMM = None, volume: bool = False) -> torch.Tensor:
    gx = t[..., :, 1:] - t[..., :, :-1]
    gy = t[..., 1:, :] - t[..., :-1, :]
    if not volume:
        return torch.cat([gx.flatten(1), gy.flatten(1)], dim=1)
    gz = t[..., 1:, :, :] - t[..., :-1, :, :]
    if voxel_mm is not None:
        sd, sh, sw = (float(v) for v in voxel_mm)
        gx, gy, gz = gx / sw, gy / sh, gz / sd
    return torch.cat([gx.flatten(1), gy.flatten(1), gz.flatten(1)], dim=1)


def _grad_l2_batch(
    deg: torch.Tensor, clean: torch.Tensor, voxel_mm: VoxelMM = None, volume: bool = False
) -> torch.Tensor:
    gc = _grad(clean, voxel_mm, volume)
    num = torch.linalg.vector_norm(_grad(deg, voxel_mm, volume) - gc, dim=1)
    return num / torch.linalg.vector_norm(gc, dim=1).clamp_min(_EPS)


def grad_l2_provenance(
    *, volume: bool, voxel_mm: VoxelMM = None, spacing_source: str | None = None
) -> dict[str, object]:
    """What ``grad_l2`` differenced, stamped beside the damage (follows the ``volume`` flag).

    A volume adds the dim -3 difference; each difference is divided by its voxel size,
    so a slice-thickness spacing (``spacing_source: encoded_fov_z``) on a gapped stack
    over-weights the through-plane term.
    """
    if not volume:
        return {"grad_l2_dims": [-2, -1], "grad_l2_weighting": "unweighted"}
    out: dict[str, object] = {
        "grad_l2_dims": [-3, -2, -1],
        "grad_l2_weighting": "unit_voxel" if voxel_mm is None else "inverse_voxel_mm",
    }
    if voxel_mm is not None:
        out["grad_l2_voxel_mm"] = [float(v) for v in voxel_mm]
        out["spacing_source"] = spacing_source
    return out


@dataclass(frozen=True)
class Yardstick:
    """A declared damage yardstick and the pool metrics it disqualifies."""

    name: str
    #: ``(degraded[N, ...], clean[N, ...], voxel_mm, volume) -> [N]``, float32 magnitudes.
    fn: Callable[[torch.Tensor, torch.Tensor, VoxelMM, bool], torch.Tensor]
    #: Pool metrics that are monotone functions of this yardstick; ranking them against
    #: damage measured on it is circular, so the caller must exclude them.
    disqualifies: frozenset[str]
    rationale: str


YARDSTICKS: dict[str, Yardstick] = {
    "rel_l2": Yardstick(
        name="rel_l2",
        fn=_rel_l2_batch,
        disqualifies=frozenset({"mse", "nmse", "rmse", "psnr", "nrmse", "nrmse_l2", "snr"}),
        rationale="Relative L2 of the residual; every squared-error metric is monotone in it.",
    ),
    "grad_l2": Yardstick(
        name="grad_l2",
        fn=_grad_l2_batch,
        disqualifies=frozenset({"hfen", "complex_hfen", "gradient_error"}),
        rationale="Relative L2 of the residual in the gradient domain; weights edge damage.",
    ),
}


def _yardstick(name: str) -> Yardstick:
    if name not in YARDSTICKS:
        raise KeyError(f"Unknown yardstick {name!r}. Available: {sorted(YARDSTICKS)}.")
    return YARDSTICKS[name]


def parse_yardsticks(text: str) -> tuple[str, ...]:
    """``"rel_l2,grad_l2"`` -> ``("rel_l2", "grad_l2")``; an unknown name raises ``ValueError``."""
    names = tuple(n.strip() for n in text.split(",") if n.strip())
    if not names:
        raise ValueError("--damage-yardsticks needs at least one yardstick")
    unknown = [n for n in names if n not in YARDSTICKS]
    if unknown:
        raise ValueError(
            f"unknown damage yardstick(s) {unknown}; expected from {sorted(YARDSTICKS)}"
        )
    return names


def bands_from_edges(edges: Mapping[str, Sequence[float]]) -> dict[str, Bands]:
    """Damage bands from ``{yardstick: [light_lo, moderate_lo, heavy_lo]}``.

    Each yardstick named gets imperceptible ``[0, light_lo)``, light, moderate and heavy
    ``[heavy_lo, inf)``; the others keep :data:`DAMAGE_BANDS`. Edges must be finite,
    non-negative and strictly increasing; anything else raises ``ValueError``.
    """
    out = dict(DAMAGE_BANDS)
    for name, raw in edges.items():
        if name not in YARDSTICKS:
            raise ValueError(
                f"damage bands for unknown yardstick {name!r}; have {sorted(YARDSTICKS)}"
            )
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or len(raw) != 3:
            raise ValueError(
                f"{name}: give three edges [light_lo, moderate_lo, heavy_lo], got {raw!r}"
            )
        try:
            lo = [float(v) for v in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}: edges must be numbers, got {raw!r}") from exc
        if not all(math.isfinite(v) and v >= 0.0 for v in lo):
            raise ValueError(f"{name}: edges must be finite and non-negative, got {lo}")
        if not lo[0] < lo[1] < lo[2]:
            raise ValueError(f"{name}: edges must be strictly increasing, got {lo}")
        uppers = (*lo, math.inf)
        lowers = (0.0, *lo)
        out[name] = tuple(zip(BAND_NAMES, lowers, uppers, strict=True))
    return out


def disqualified_metrics(yardstick: str) -> frozenset[str]:
    """Pool metrics to exclude when ranking against damage measured on ``yardstick``."""
    return _yardstick(yardstick).disqualifies


def damage_batch(
    clean: torch.Tensor,
    degraded: torch.Tensor,
    yardstick: str,
    voxel_mm: VoxelMM = None,
    *,
    volume: bool = False,
) -> torch.Tensor:
    """Damage of each of ``N`` degraded images against ``clean``: a ``[N]`` float32 tensor.

    ``clean`` is ``[N, ...]`` or a single image broadcast over the stack. Magnitudes of
    complex input, unclipped, so heavy damage may exceed 1. ``volume=True`` declares a
    ``(N, C, D, H, W)`` volume: ``grad_l2`` adds the through-plane difference, weighted by
    ``voxel_mm`` (``(D, H, W)`` mm; ``None`` is a unit voxel), which only a volume takes.
    The rank alone never adds the term.
    """
    y = _yardstick(yardstick)
    if volume and degraded.ndim != 5:
        raise ValueError(f"volume=True needs a (N, C, D, H, W) stack; got {degraded.ndim}-D")
    if voxel_mm is not None and not volume:
        raise ValueError("voxel_mm weights a volume's differences; pass volume=True")
    deg = _magnitude(degraded)
    ref = _magnitude(clean)
    if ref.ndim == deg.ndim - 1:
        ref = ref.unsqueeze(0)  # broadcast, never materialised N times
    if ref.shape[1:] != deg.shape[1:] or ref.shape[0] not in (1, deg.shape[0]):
        raise ValueError(f"clean {tuple(ref.shape)} and degraded {tuple(deg.shape)} differ")
    return y.fn(deg, ref, voxel_mm, volume)


def damage(clean: torch.Tensor, degraded: torch.Tensor, yardstick: str) -> float:
    """Damage of one degraded image against ``clean``."""
    return float(damage_batch(clean[None], degraded[None], yardstick).item())


def band_of(d: float, yardstick: str, bands: Mapping[str, Bands] = DAMAGE_BANDS) -> str:
    """Name of the half-open band ``[lo, hi)`` that contains damage ``d``."""
    if yardstick not in bands:
        raise KeyError(f"No damage bands for yardstick {yardstick!r}. Have: {sorted(bands)}.")
    if math.isnan(d) or d < 0:
        raise ValueError(f"damage must be a non-negative number, got {d!r}")
    for name, lo, hi in bands[yardstick]:
        if lo <= d < hi:
            return name
    raise ValueError(f"damage {d!r} falls outside the {yardstick!r} bands")


def compare_yardsticks(scores: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    """Spearman rho between metric scores built under different yardsticks.

    A ranking stable across structurally different yardsticks reflects the metrics; one
    that flips reflects the yardstick choice and supports no conclusion. Rho is taken on
    the scores, so tied metrics share an average rank instead of a list position.

    Args:
        scores: ``{yardstick_name: {metric: score, higher is better}}``.

    Returns:
        ``{"a_vs_b": rho}`` for every pair (NaN under three common metrics, or when one
        side is constant).
    """
    from scipy.stats import spearmanr

    out: dict[str, float] = {}
    names = sorted(scores)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = sorted(set(scores[a]) & set(scores[b]))
            if len(common) < 3:
                out[f"{a}_vs_{b}"] = float("nan")
                continue
            sa = [float(scores[a][m]) for m in common]
            sb = [float(scores[b][m]) for m in common]
            if len(set(sa)) < 2 or len(set(sb)) < 2:
                out[f"{a}_vs_{b}"] = float("nan")
                continue
            out[f"{a}_vs_{b}"] = float(spearmanr(sa, sb).statistic)
    return out
