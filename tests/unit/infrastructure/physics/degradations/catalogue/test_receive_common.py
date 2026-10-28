"""Plumbing of the receive-chain axes: line ranks, conditions, the 2-D adapter, control stream."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from spectramr.infrastructure.physics.degradations import (
    AcquisitionGeometry,
    AcquisitionMode,
    apply_degradation,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_common import (
    ENCODED_DIMS,
    acquisition_ranks,
    check_n_coils,
    coil_covariance_condition,
    control_generator,
    n_coils_condition,
    positive,
    subject_peak,
)
from spectramr.infrastructure.physics.degradations.catalogue.receive_noise import (
    emi_burst,
    emi_burst_2d,
)
from spectramr.infrastructure.physics.degradations.geometry import LineOrder, SliceOrder
from spectramr.infrastructure.physics.scanner_presets import FASTMRI_3T, M4RAW_03T

M = AcquisitionMode
ORDERS = [
    AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None, slice_order=SliceOrder.SEQUENTIAL),
    AcquisitionGeometry(mode=M.MULTISLICE_2D, voxel_mm=None, slice_order=SliceOrder.INTERLEAVED),
    AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=LineOrder.LINEAR),
    AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=LineOrder.CENTRIC),
    AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None, line_order=LineOrder.KZ_INNER),
]


def test_encoded_dims_follow_the_acquisition_mode() -> None:
    assert ENCODED_DIMS[M.SLICE_2D] == ENCODED_DIMS[M.MULTISLICE_2D] == (-2, -1)
    assert ENCODED_DIMS[M.VOLUME_3D] == (-3, -2, -1)


@pytest.mark.parametrize("geometry", ORDERS)
def test_ranks_are_a_permutation_and_tau_is_ordered_by_rank(geometry) -> None:
    rank, tau = acquisition_ranks(geometry, 6, 10, torch.device("cpu"))
    assert sorted(rank.flatten().tolist()) == list(range(60))
    assert float(tau.min()) >= 0.0 and float(tau.max()) < 1.0
    order = rank.flatten().argsort()
    assert bool((tau.flatten()[order].diff() >= 0).all())


def test_single_slice_ranks_are_the_phase_encode_order() -> None:
    geometry = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    rank, tau = acquisition_ranks(geometry, 1, 8, torch.device("cpu"))
    assert rank.tolist() == [list(range(8))]
    assert torch.allclose(tau, torch.arange(8) / 8.0)


def test_subject_peak_is_per_subject_over_re_and_im() -> None:
    k = torch.zeros(2, 2, 3, 4, 5, dtype=torch.complex64)
    k[0, 1, 2, 3, 4] = torch.complex(torch.tensor(-3.0), torch.tensor(1.0))
    k[1, 0, 0, 0, 0] = torch.complex(torch.tensor(0.5), torch.tensor(-7.0))
    assert subject_peak(k).flatten().tolist() == [3.0, 7.0]


def test_positive_and_n_coils_validation() -> None:
    assert positive("v", 2) == 2.0
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="v must be"):
            positive("v", bad)
    x = torch.zeros(1, 4, 1, 2, 2)
    check_n_coils(x, None)
    check_n_coils(x, 4)
    with pytest.raises(ValueError, match="n_coils=15"):
        check_n_coils(x, 15)


def test_conditions_read_the_profiles_coil_facts() -> None:
    cov = ((1.0, 0.2), (0.2, 1.0))
    profile = replace(M4RAW_03T, n_coils=2, coil_covariance=cov)
    args = (profile, 0.5, profile.geometry, (1, 2, 8, 8))
    assert n_coils_condition(*args) == {"n_coils": 2}
    assert coil_covariance_condition(*args) == {"n_coils": 2, "coil_covariance": cov}
    assert coil_covariance_condition(FASTMRI_3T, 1.0, FASTMRI_3T.geometry, ()) == {"n_coils": 15}


def test_a_profile_without_a_covariance_lets_the_caller_supply_one() -> None:
    x = torch.randn(2, 4, 16, 16, dtype=torch.complex64)
    cov = torch.eye(4) + 0.2
    out = apply_degradation(
        "correlated_coil_noise", x, 0.5, seed=1, profile=M4RAW_03T, coil_covariance=cov
    )
    assert out.shape == x.shape


def test_profile_coil_count_must_match_the_input() -> None:
    x = torch.randn(1, 4, 8, 8, dtype=torch.complex64)
    apply_degradation("coil_dropout", x, 0.5, profile=M4RAW_03T)
    with pytest.raises(ValueError, match="n_coils=15"):
        apply_degradation("coil_dropout", x, 0.5, profile=FASTMRI_3T)
    with pytest.raises(ValueError, match="n_coils=15"):
        apply_degradation("correlated_coil_noise", x, 0.5, profile=FASTMRI_3T)


def test_the_2d_kernel_is_the_slice_mode_implementation() -> None:
    x = torch.randn(2, 3, 12, 12, dtype=torch.complex64)
    geometry = AcquisitionGeometry(mode=M.SLICE_2D, voxel_mm=None)
    torch.testing.assert_close(
        emi_burst_2d(x, 0.7, seed=3), emi_burst(x, 0.7, 3, geometry), rtol=0, atol=0
    )


def test_an_input_rank_that_disagrees_with_the_mode_raises() -> None:
    x4 = torch.randn(1, 2, 8, 8, dtype=torch.complex64)
    with pytest.raises(ValueError, match="does not match acquisition mode"):
        emi_burst(x4, 0.5, 0, AcquisitionGeometry(mode=M.VOLUME_3D, voxel_mm=None))


def test_control_stream_is_not_the_noise_stream() -> None:
    noise = torch.Generator().manual_seed(5)
    control = control_generator(torch.device("cpu"), 5)
    assert not torch.equal(torch.rand(4, generator=noise), torch.rand(4, generator=control))
