"""A one-slice multislice stack degrades exactly like the 2-D slice it holds."""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
)
from spectramr.infrastructure.physics.degradations.registry_queries import (
    slice_capable_degradations,
)
from tests.utils.degradation_inputs import apply_degradation

M = AcquisitionMode
GEOM = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)  # interleaved by default
MULTISLICE = [n for n, s in DEGRADATION_REGISTRY.items() if M.MULTISLICE_2D in s.supported_modes]
# Task 11's native fields are drawn on the volume grid (2 * centred_freqs, 3-D modes), not the
# 2-D kernel's linspace grid, so a one-slice stack is the same physics but not the same draw.
NATIVE_FIELDS = {
    "b0",
    "b0_geometric_distortion",
    "bias",
    "coil_miscal",
    "susceptibility",
    "concomitant_phase",
}
# Task 13's native geometric operators draw one 3-D field per subject through per_sample_rng.
NATIVE_GEOMETRIC = {"geometric_warp", "gradient_nonlinearity", "magic_angle"}
# Task 14's native rigid and pulsatile motion draw per-shot poses / a CPU cardiac phase on the
# volume grid; the motion trio stays the 2-D kernel for the default interleaved order.
NATIVE_MOTION = {"rigid_motion", "pulsatile_motion"}
# Task 18's catalogue motion. respiratory_motion: volumes default SI to the slice axis (2-D: PE).
# local_intermittent_motion: its direction is planar only in SLICE_2D, so a one-slice stack draws
# a 3-D direction whose dz component is a no-op.
NATIVE_CATALOGUE = {"respiratory_motion", "local_intermittent_motion"}
NATIVE = NATIVE_FIELDS | NATIVE_GEOMETRIC | NATIVE_MOTION | NATIVE_CATALOGUE


def test_multislice_set_is_the_declared_one() -> None:
    # 16 per-slice declarations, the native noise of Task 9, the native fields of Task 11,
    # the native geometric operators of Task 13, rigid / pulsatile motion of Task 14 and the
    # catalogue axes that followed.
    native_noise = {"complex_gaussian", "rician", "emi", "resolution_snr"}
    assert native_noise | NATIVE <= set(MULTISLICE)
    assert len(MULTISLICE) == 69


# A volume-only axis has no single-slice implementation, hence no 2-D reference to match.
@pytest.mark.parametrize(
    "name", [n for n in MULTISLICE if n not in NATIVE and n in slice_capable_degradations()]
)
def test_one_slice_stack_matches_slice_2d(name: str) -> None:
    g = torch.Generator().manual_seed(7)
    shape = (1, 2, 16, 16)
    x = torch.complex(torch.randn(shape, generator=g), torch.randn(shape, generator=g))
    ref = apply_degradation(name, x, 0.6, seed=4)
    out = apply_degradation(name, x.unsqueeze(2), 0.6, seed=4, geometry=GEOM)
    torch.testing.assert_close(out, ref.unsqueeze(2), rtol=0, atol=1e-6)
