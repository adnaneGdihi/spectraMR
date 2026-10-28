"""off_resonance_slice_shift: off-resonance moves the excited slice along the slice-select axis.

A spin off resonance by ``df`` is excited where the slice-select gradient makes up the
difference, ``dz = df / (gamma_bar G_ss)`` (Bernstein et al., Handbook of MRI Pulse Sequences,
2004, sec. 2.3). With the gradient set by the RF bandwidth, ``G_ss = BW_rf / (gamma_bar thk)``,

    dz = thk * df / BW_rf,

so the shift in slice pitches is ``(thk / pitch) df / BW_rf``. Multislice: slice k samples the
object at ``z + dz(r)`` (pull-back, linear interpolation along D). 3-D: the slab is excited as
a whole and partitions are phase-encoded, so only the slab boundary moves: ``thk`` is the slab
(``D`` pitches) and each voxel keeps the fraction of its extent inside the shifted slab.
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.infrastructure.physics.degradations._common import _ensure_complex, _generator
from spectramr.infrastructure.physics.degradations.blocks.volume_fields import smooth_random_field
from spectramr.infrastructure.physics.degradations.blocks.volume_warp import warp_volume
from spectramr.infrastructure.physics.degradations.catalogue.metal_chemistry import (
    slice_select_shift_mm,
)
from spectramr.infrastructure.physics.degradations.catalogue.multislice_common import (
    positive,
    slab_coverage,
    thickness_mm,
    thickness_over_pitch,
)
from spectramr.infrastructure.physics.degradations.geometry import (
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.volume.fields_maps import check_volume
from spectramr.infrastructure.physics.digital_twin_simulator import GAMMA_BAR_HZ_PER_T

__all__ = [
    "DEFAULT_BW_RF_HZ",
    "PEAK_OFF_RESONANCE_HZ",
    "off_resonance_slice_shift",
    "slice_select_gradient_mt_per_m",
]

DEFAULT_BW_RF_HZ = 1000.0
PEAK_OFF_RESONANCE_HZ = 200.0


def slice_select_gradient_mt_per_m(bw_rf_hz: float, thickness_mm_: float) -> float:
    """``G_ss = BW_rf / (gamma_bar thk)`` in mT/m."""
    return 1e3 * bw_rf_hz / (GAMMA_BAR_HZ_PER_T * thickness_mm_ * 1e-3)


def _field_hz(x: Tensor, theta: float, seed: int, b0_map_hz, peak_hz: float, field_scale: float):
    shape = (x.shape[0], 1, *x.shape[-3:])
    if b0_map_hz is None:
        field = smooth_random_field(
            shape[0], shape[2:], generator=_generator(x.device, seed), device=x.device
        )
        return float(theta) * peak_hz * field_scale * field
    if not torch.is_tensor(b0_map_hz) or b0_map_hz.is_complex() or b0_map_hz.dim() != 5:
        raise ValueError(
            "off_resonance_slice_shift: b0_map_hz must be a real (B, 1, D, H, W) tensor"
        )
    try:
        torch.broadcast_shapes(tuple(b0_map_hz.shape), shape)
    except RuntimeError:
        raise ValueError(
            f"off_resonance_slice_shift: b0_map_hz {tuple(b0_map_hz.shape)} does not broadcast "
            f"to {shape}"
        ) from None
    return float(theta) * b0_map_hz.to(device=x.device, dtype=x.real.dtype).expand(shape)


def off_resonance_slice_shift(
    x: Tensor,
    theta: float,
    seed: int,
    geometry: AcquisitionGeometry,
    b0_map_hz: Tensor | None = None,
    bw_rf_hz: float = DEFAULT_BW_RF_HZ,
    peak_hz: float | None = None,
    field_scale: float | None = None,
    gradient_max_mT_per_m: float | None = None,  # noqa: N803
    profile_field_scale: float | None = None,
) -> Tensor:
    """Resample along D at ``z + dz(r)``, ``dz = thk * df / BW_rf``; df = theta * the field (Hz).

    ``b0_map_hz`` is the field at theta = 1; without it a seeded smooth field of peak
    ``peak_hz * field_scale`` is drawn (``peak_hz`` default 200 Hz, ``field_scale`` default 1).
    A scanner profile supplies ``profile_field_scale`` = B0 / 3 T, which scales the seeded field
    only: a measured map is already in Hz, so it is used as given, and an explicit ``peak_hz`` or
    ``field_scale`` with it raises.
    """
    name = "off_resonance_slice_shift"
    check_volume(x, geometry, name)
    bw = positive(name, "bw_rf_hz", bw_rf_hz)
    if b0_map_hz is not None and (peak_hz is not None or field_scale is not None):
        raise ValueError(
            f"{name}: b0_map_hz is a measured field in Hz; peak_hz and field_scale scale the "
            "seeded field only, so drop them or the map"
        )
    if field_scale is not None and profile_field_scale is not None:
        raise ValueError(f"{name}: field_scale and a profile's B0 scaling were both given")
    peak = positive(name, "peak_hz", PEAK_OFF_RESONANCE_HZ if peak_hz is None else peak_hz)
    seeded_scale = profile_field_scale if field_scale is None else field_scale
    scale = positive(name, "field_scale", 1.0 if seeded_scale is None else seeded_scale)
    volume = geometry.mode is AcquisitionMode.VOLUME_3D
    d = x.shape[-3]
    if gradient_max_mT_per_m is not None:
        thk = (
            d * geometry.voxel_mm[0]
            if volume and geometry.voxel_mm
            else thickness_mm(geometry, name)
        )
        g = slice_select_gradient_mt_per_m(bw, thk)
        if g > positive(name, "gradient_max_mT_per_m", gradient_max_mT_per_m):
            raise ValueError(
                f"{name}: slice-select gradient {g:.3g} mT/m (BW_rf {bw} Hz, thickness {thk} mm) "
                f"exceeds gradient_max_mT_per_m={gradient_max_mT_per_m}"
            )
    x = _ensure_complex(x)
    if float(theta) <= 0.0:
        return x
    ratio = float(d) if volume else thickness_over_pitch(geometry, name)
    # Excited locus displacement in pitches, +thk df / BW_rf: minus the assignment shift.
    shift_vox = -slice_select_shift_mm(_field_hz(x, theta, seed, b0_map_hz, peak, scale), ratio, bw)
    if volume:
        return x * slab_coverage(shift_vox, d)
    disp = torch.zeros(x.shape[0], 3, *x.shape[-3:], device=x.device, dtype=x.real.dtype)
    disp[:, 0] = shift_vox[:, 0]
    return warp_volume(x, disp)
