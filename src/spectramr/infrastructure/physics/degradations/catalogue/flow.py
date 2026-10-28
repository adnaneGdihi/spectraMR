"""Flow axes: flow_ghosts, flow_voids, inflow_enhancement, on a slice, a multislice stack or a volume.

A vessel map selects the flowing pixels (the caller's ``vessel_map`` or seeded discs, see
``flow_vessels``). Ghosts modulate those pixels line by line in k-space; voids and inflow are
image-space gains on them. Axes: dim -3 slice/kz, -2 phase-encode, -1 readout.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex
from spectramr.infrastructure.physics.degradations.blocks.acquisition_timing import line_times
from spectramr.infrastructure.physics.degradations.catalogue.flow_vessels import (
    K_DIMS,
    as_5d,
    check_positive,
    resolve_vessels,
    sample_draws,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
    voxel_mm_or_unit,
)
from spectramr.infrastructure.physics.fft_ops import fftnc, ifftnc

__all__ = [
    "DEFAULT_SLICE_THICKNESS_MM",
    "FLOW_GHOST_MAX_DEPTH",
    "FLOW_VOID_MAX_CM_S",
    "INFLOW_MAX_CM_S",
    "flow_ghosts",
    "flow_ghosts_2d",
    "flow_voids",
    "flow_voids_2d",
    "inflow_enhancement",
    "inflow_gain",
    "washout_factor",
]

FLOW_GHOST_MAX_DEPTH = 0.8  # modulation depth a at theta = 1 (dimensionless)
FLOW_VOID_MAX_CM_S = 50.0  # theta = 1: the first zero, 2 thk / TE, at the defaults (20 ms, 5 mm)
INFLOW_MAX_CM_S = 50.0
DEFAULT_SLICE_THICKNESS_MM = 5.0  # used when the geometry carries no slice thickness
_MM_PER_MS_PER_CM_S = 0.01  # 1 cm/s = 10 mm/s = 0.01 mm/ms
_SLICE_2D = AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=None)


def flow_ghosts(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    vessel_map: Tensor | None = None,
    tr_ms: float = 50.0,
    t_card_ms: float = 1000.0,
) -> Tensor:
    """Pulsatile-flow ghosts: vessel signal scaled by ``1 + a cos(2 pi t_n / T_card + phi0)`` per line.

    The factor multiplies the vessel only (Haacke et al., *MRI: Physical Principles and Sequence
    Design*, 1999, ch. 24; Bernstein et al., *Handbook of MRI Pulse Sequences*, 2004, 6.3), so
    ``out = x + ifft(a cos(.) * fft(V x))``. A modulation of ``f = TR / T_card`` cycles per line
    puts ghosts of the vessel alone at ``+/- N_pe f`` px along phase-encode. Line time ``t_n`` is
    ``acquisition_timing.line_times``: ``ky`` for a slice, ``(slice, ky)`` honouring slice_order,
    ``(kz, ky)`` honouring line_order. ``a = 0.8 theta``; ``phi0`` is drawn once per sample.
    """
    check_positive("flow_ghosts", tr_ms=tr_ms, t_card_ms=t_card_ms)
    ndim = x.dim()
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    x5 = as_5d(x)
    b, d, h = x5.shape[0], x5.shape[-3], x5.shape[-2]
    u = sample_draws(x5, seed)
    vessel = resolve_vessels(x5, vessel_map, u, "flow_ghosts", ndim_in=ndim)
    times = line_times(geometry, d, h, tr_s=tr_ms * 1e-3).double()
    cycles = ((times / (t_card_ms * 1e-3)) % 1.0).to(device=x5.device, dtype=x5.real.dtype)
    phi0 = u[:, -1].to(x5.real.dtype).view(b, 1, 1, 1, 1)
    depth = float(theta) * FLOW_GHOST_MAX_DEPTH
    mod = depth * torch.cos(2.0 * math.pi * (cycles.view(1, 1, d, h, 1) + phi0))
    dims = K_DIMS[geometry.mode]
    ghosts = ifftnc(fftnc(vessel * x5, dims=dims) * mod, dims=dims)
    return (x5 + ghosts).view_as(x)


def flow_ghosts_2d(x: Tensor, theta: float, seed: int = 0, **kw) -> Tensor:
    """The slice kernel of :func:`flow_ghosts`."""
    return flow_ghosts(x, theta, seed, _SLICE_2D, **kw)


def washout_factor(v_cm_s: float, te_ms: float, thickness_mm: float) -> float:
    """Spin-echo washout ``max(0, 1 - v (TE / 2) / thk)`` of through-plane flow.

    Spins excited by the 90 pulse that leave the slice before the 180 pulse at TE / 2 give no
    echo (Bradley, in Stark & Bradley, *Magnetic Resonance Imaging*, 1988, ch. 'Flow
    phenomena'; Bernstein et al. 2004).
    """
    return max(0.0, 1.0 - v_cm_s * _MM_PER_MS_PER_CM_S * (te_ms / 2.0) / thickness_mm)


def _thickness_mm(geometry: AcquisitionGeometry, depth: int) -> float:
    """Slab thickness ``D * voxel_mm[0]`` (3-D, 1 mm voxels without ``voxel_mm``); a slice's
    ``voxel_mm[0]``; a stack's ``slice_thickness_mm``, else the pitch minus the gap; else 5 mm."""
    vox = voxel_mm_or_unit(geometry.voxel_mm)
    if geometry.mode is AcquisitionMode.VOLUME_3D:
        return depth * vox[0]
    if geometry.mode is AcquisitionMode.SLICE_2D and geometry.voxel_mm is not None:
        return geometry.voxel_mm[0]
    if geometry.slice_thickness_mm:
        return geometry.slice_thickness_mm
    if geometry.voxel_mm is not None and geometry.mode is AcquisitionMode.MULTISLICE_2D:
        thickness = vox[0] - geometry.slice_gap_mm
        if thickness <= 0:
            raise ValueError(
                f"flow_voids: derived slice thickness voxel_mm[0] - slice_gap_mm = {thickness} mm "
                f"is not positive (voxel_mm[0]={vox[0]}, slice_gap_mm={geometry.slice_gap_mm}); "
                "set slice_thickness_mm"
            )
        return thickness
    return DEFAULT_SLICE_THICKNESS_MM


def flow_voids(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    vessel_map: Tensor | None = None,
    te_ms: float = 20.0,
) -> Tensor:
    """Flow voids: vessel signal times :func:`washout_factor` at ``v = 50 theta`` cm/s.

    The gain is an image-space multiplication, the same in every mode; only the thickness
    differs (slice: ``voxel_mm[0]``; stack: ``slice_thickness_mm``, else ``voxel_mm[0] -
    slice_gap_mm``; either: 5 mm without them; volume: the slab, ``D * voxel_mm[0]``, with 1 mm
    voxels if ``voxel_mm`` is unset). A real input stays real.
    """
    check_positive("flow_voids", te_ms=te_ms)
    if float(theta) <= 0.0:
        return x
    x5 = as_5d(x)
    u = sample_draws(x5, seed)
    vessel = resolve_vessels(x5, vessel_map, u, "flow_voids", ndim_in=x.dim())
    f = washout_factor(
        float(theta) * FLOW_VOID_MAX_CM_S, te_ms, _thickness_mm(geometry, x5.shape[-3])
    )
    return (x5 * (1.0 - vessel * (1.0 - f))).view_as(x)


def flow_voids_2d(
    x: Tensor, theta: float, seed: int = 0, *, voxel_mm: tuple[float, ...] | None = None, **kw
) -> Tensor:
    """The slice kernel of :func:`flow_voids`; ``voxel_mm[0]`` is the thickness (None: 5 mm)."""
    geometry = AcquisitionGeometry(mode=AcquisitionMode.SLICE_2D, voxel_mm=voxel_mm)
    return flow_voids(x, theta, seed, geometry, **kw)


def inflow_gain(
    depth_mm: Tensor, v_cm_s: float, *, tr_ms: float, t1_ms: float, fa_deg: float
) -> Tensor:
    """Vessel signal over stationary-tissue steady state at ``depth_mm`` into the volume (float64).

    Inflowing spins start fully relaxed (``M_0 = 1``) and see ``n = depth / (v TR)`` pulses, with
    ``M_{n+1} = M_n cos(a) E1 + (1 - E1)``, ``E1 = exp(-TR / T1)``; the signal after pulse ``n`` is
    ``M_n sin(a)``. The recursion solves to ``M_n = M_ss + (1 - M_ss)(E1 cos a)^n`` with
    ``M_ss = (1 - E1) / (1 - E1 cos a)``, which also serves fractional ``n``; tissue sits at
    ``M_ss sin(a)``, the spoiled-GRE steady state (Ernst & Anderson 1966; Haacke et al. 1999,
    ch. 18 and 24), so the gain is ``M_n / M_ss``.
    """
    e1 = math.exp(-tr_ms / t1_ms)
    ec = e1 * math.cos(math.radians(fa_deg))
    m_ss = (1.0 - e1) / (1.0 - ec)
    n = depth_mm.double() / (v_cm_s * _MM_PER_MS_PER_CM_S * tr_ms)
    return (m_ss + (1.0 - m_ss) * ec**n) / m_ss


def inflow_enhancement(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    *,
    vessel_map: Tensor | None = None,
    tr_ms: float = 30.0,
    t1_ms: float = 1500.0,
    fa_deg: float = 30.0,
) -> Tensor:
    """Inflow enhancement: vessel signal times :func:`inflow_gain` at ``v = 50 theta`` cm/s.

    Depth is measured at the slice centre, ``(k + 1/2) voxel_mm[0]`` for index ``k`` along D with entry at
    index 0 (1 mm if ``voxel_mm`` is unset), so the gain tends to 1 as ``v -> 0``. In a stack it is the spatial index, not the excitation rank: the
    pulses a spin has seen depend on where it has travelled, not on when the stack is excited.
    A real input stays real.
    """
    check_positive("inflow_enhancement", tr_ms=tr_ms, t1_ms=t1_ms, fa_deg=fa_deg)
    if fa_deg > 90.0:
        raise ValueError(f"inflow_enhancement: fa_deg must be <= 90 (cos a >= 0); got {fa_deg}")
    if float(theta) <= 0.0:
        return x
    x5 = as_5d(x)
    d = x5.shape[-3]
    u = sample_draws(x5, seed)
    vessel = resolve_vessels(x5, vessel_map, u, "inflow_enhancement", ndim_in=x.dim())
    depth = (torch.arange(d, dtype=torch.float64) + 0.5) * voxel_mm_or_unit(geometry.voxel_mm)[0]
    gain = inflow_gain(
        depth, float(theta) * INFLOW_MAX_CM_S, tr_ms=tr_ms, t1_ms=t1_ms, fa_deg=fa_deg
    )
    gain = gain.to(device=x5.device, dtype=x5.real.dtype).view(1, 1, d, 1, 1)
    return (x5 * (1.0 + vessel * (gain - 1.0))).view_as(x)
