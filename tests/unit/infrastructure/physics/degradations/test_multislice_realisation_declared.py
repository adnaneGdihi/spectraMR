"""A multislice declaration must match the draw: shared realisation means identical residuals."""

from __future__ import annotations

import torch

from spectramr.infrastructure.physics.degradations import (
    DEGRADATION_REGISTRY,
    AcquisitionGeometry,
    AcquisitionMode,
    Realisation,
)

M = AcquisitionMode
GEOM = AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None)


def _identical_slice_residuals(name: str) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    one = torch.complex(
        torch.randn(1, 2, 1, 16, 16, generator=g), torch.randn(1, 2, 1, 16, 16, generator=g)
    )
    stack = one.expand(1, 2, 4, 16, 16).clone()
    impl = DEGRADATION_REGISTRY[name].implementations[M.MULTISLICE_2D]
    return impl(stack, 0.6, 5, GEOM) - stack


def test_iq_ghost_residuals_are_shared_across_slices_and_declared_so() -> None:
    res = _identical_slice_residuals("iq_ghost")
    shared = bool((res - res[:, :, :1]).abs().max() == 0)
    declared = DEGRADATION_REGISTRY["iq_ghost"].realisation[M.MULTISLICE_2D]
    expected = Realisation.PER_SLICE_EXACT if shared else Realisation.PER_SLICE_INDEPENDENT
    assert declared is expected, f"iq_ghost draws shared={shared} but declares {declared}"


def test_iq_ghost_dc_spike_is_one_stack_constant() -> None:
    # A receiver DC offset is one hardware constant, so the spike does not scale per slice.
    g = torch.Generator().manual_seed(3)
    one = torch.complex(
        torch.randn(1, 1, 1, 16, 16, generator=g), torch.randn(1, 1, 1, 16, 16, generator=g)
    )
    stack = torch.cat([one, 4.0 * one], dim=2)
    impl = DEGRADATION_REGISTRY["iq_ghost"].implementations[M.MULTISLICE_2D]
    zero_iq = impl(stack, 0.6, 5, GEOM, max_iq_imbalance=0.0) - stack
    spikes = zero_iq[0, 0, :, 8, 8]
    torch.testing.assert_close(spikes[0], spikes[1])
