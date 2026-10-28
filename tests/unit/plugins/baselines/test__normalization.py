"""The shared scaling and undersampling helpers the three adapters build on."""

from __future__ import annotations

import pytest
import torch
from spectramr_baselines._normalization import (
    peak_magnitude,
    phase_encode_mask,
    require_image_size,
    to_complex,
    to_real_stack,
    zero_filled,
)


def test_to_complex_refuses_anything_but_one_complex_channel() -> None:
    with pytest.raises(ValueError, match=r"\[B, 2, H, W\]"):
        to_complex(torch.randn(2, 4, 8, 8))


def test_to_real_stack_inverts_to_complex() -> None:
    x = torch.randn(3, 2, 8, 8)
    assert torch.equal(to_real_stack(to_complex(x)), x)


def test_peak_magnitude_is_the_complex_peak_per_sample() -> None:
    x = torch.zeros(2, 2, 4, 4)
    x[0, 0, 1, 1], x[0, 1, 1, 1] = 3.0, 4.0
    x[1, 1, 2, 3] = -2.0

    peak = peak_magnitude(x)

    assert peak.shape == (2, 1, 1, 1)
    torch.testing.assert_close(peak.flatten(), torch.tensor([5.0, 2.0]))


def test_peak_magnitude_floors_an_empty_slice_rather_than_dividing_by_zero() -> None:
    assert torch.isfinite(torch.zeros(1, 2, 4, 4) / peak_magnitude(torch.zeros(1, 2, 4, 4))).all()


def test_phase_encode_mask_samples_the_last_axis() -> None:
    mask_1d = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])

    mask = phase_encode_mask(mask_1d, height=3)

    assert mask.shape == (1, 1, 3, 5)
    assert torch.equal(mask[0, 0, 2], mask_1d), "every readout row carries the same PE pattern"


def test_phase_encode_mask_refuses_a_2d_mask() -> None:
    with pytest.raises(ValueError, match="1-D"):
        phase_encode_mask(torch.ones(4, 4), height=4)


def test_zero_filled_through_a_full_mask_is_the_identity() -> None:
    x = torch.randn(2, 2, 8, 8)
    torch.testing.assert_close(zero_filled(x, torch.ones(1, 1, 8, 8)), x, atol=1e-5, rtol=1e-5)


def test_zero_filled_removes_the_unsampled_lines() -> None:
    from spectramr.infrastructure.physics.fft_ops import fft2c

    x = torch.randn(1, 2, 8, 8)
    mask = phase_encode_mask(torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]), height=8)

    kspace = fft2c(to_complex(zero_filled(x, mask)))

    assert kspace[..., 2].abs().max() < 1e-5 and kspace[..., 5].abs().max() < 1e-5


def test_zero_filled_refuses_a_mask_of_another_size() -> None:
    with pytest.raises(ValueError, match="does not match"):
        zero_filled(torch.randn(1, 2, 8, 8), torch.ones(1, 1, 8, 16))


def test_require_image_size_accepts_the_configured_square() -> None:
    require_image_size(torch.zeros(1, 2, 320, 320), 320, "Adapter", "model.model_kwargs.image_size")


@pytest.mark.parametrize("shape", [(1, 2, 256, 256), (1, 2, 320, 256), (1, 2, 256, 320)])
def test_require_image_size_names_both_keys_on_a_mismatch(shape: tuple[int, ...]) -> None:
    """PLANTED VIOLATION: a patch of any other size than the adapter's masks."""
    with pytest.raises(ValueError, match=r"image_size.*data\.sampling\.patch_size"):
        require_image_size(torch.zeros(shape), 320, "Adapter", "model.model_kwargs.image_size")


def test_shen_prepare_transforms_the_complex_magnitude() -> None:
    """Planted: ``x.abs()`` on the stack is |Re| + i|Im|, not the complex modulus.

    The two agree in magnitude, so a magnitude-only check cannot tell them apart;
    the k-space of a real (magnitude) image is Hermitian, which |Re| + i|Im| is not.
    """
    import inspect

    from spectramr_baselines import shen2024

    src = inspect.getsource(shen2024.Shen2024Baseline._prepare)
    assert "to_complex(x.abs())" not in src
    assert "to_complex(x).abs()" in src
