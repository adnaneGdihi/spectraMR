"""``svd_coil_compression``: fitted on the masked input, applied to every k-space key.

Pinned, each with the shape it can fail in: the compression commutes with the
mask; the basis is a function of the masked input alone, so changing the target
or the unacquired lines leaves it unchanged; the transform refuses to run before
the masking and beside physical-coil maps. Non-square slices throughout.
"""

from __future__ import annotations

import pytest
import torch
import torchio as tio

from spectramr.data.transforms.complementary_rep_masking import ComplementaryRepMasking
from spectramr.data.transforms.registry import build_transform, list_transforms
from spectramr.data.transforms.svd_coil_compression import compress_subject
from spectramr.infrastructure.physics.coil_compression import (
    apply_svd_compression,
    fit_svd_basis,
)

_C, _H, _W, _S = 4, 12, 40, 2


def _kspace(seed: int) -> torch.Tensor:
    """``(2C, H, W, S)`` real-interleaved, as the M4Raw loader serves it."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(2 * _C, _H, _W, _S, generator=gen)


def _subject(target_seed: int = 1) -> tio.Subject:
    subject = tio.Subject(
        input=tio.ScalarImage(tensor=_kspace(0)),
        target=tio.ScalarImage(tensor=_kspace(target_seed)),
        kspace=tio.ScalarImage(tensor=_kspace(target_seed)),
        input_alt=tio.ScalarImage(tensor=_kspace(2)),
    )
    subject["file_id"] = "2022061001_T101"
    return subject


def _masked(target_seed: int = 1) -> tio.Subject:
    return ComplementaryRepMasking(acceleration=4.0, center_fraction=0.1, mode="same")(
        _subject(target_seed)
    )


def _slices(image: tio.ScalarImage) -> torch.Tensor:
    return image.data.permute(3, 0, 1, 2)


def test_it_is_registered_and_builds_from_yaml_kwargs() -> None:
    assert "svd_coil_compression" in list_transforms()
    assert build_transform("svd_coil_compression", num_virtual_coils=1).num_virtual_coils == 1
    with pytest.raises(ValueError, match="num_virtual_coils"):
        build_transform("svd_coil_compression", num_virtual_coils=0)


def test_every_kspace_key_reaches_k_virtual_coils() -> None:
    out = build_transform("svd_coil_compression", num_virtual_coils=1)(_masked())
    for key in ("input", "measured_kspace", "target", "kspace", "input_alt"):
        assert out[key].data.shape == (2, _H, _W, _S), key
    assert out["mask"].data.shape == (1, _H, _W, _S)
    assert out["coil_compression"] == "svd_on_masked_input;num_virtual_coils=1"


def test_it_commutes_with_the_mask() -> None:
    """``U^H (M y) == M (U^H y)`` with the basis the transform fitted."""
    masked = _masked()
    mask = _slices(masked["mask"])
    raw_input = _slices(_subject()["input"])
    bases = compress_subject(masked, 2)
    for s, basis in enumerate(bases):
        compressed_then_masked = apply_svd_compression(raw_input[s : s + 1], basis) * mask[s]
        torch.testing.assert_close(_slices(masked["input"])[s : s + 1], compressed_then_masked)
    assert torch.equal(_slices(masked["measured_kspace"]), _slices(masked["input"]))


def test_the_basis_is_invariant_to_the_target() -> None:
    """Planted: a basis fitted on `target` would move when the target does."""
    first, second = _masked(target_seed=1), _masked(target_seed=7)
    bases_a = compress_subject(first, 1)
    bases_b = compress_subject(second, 1)
    for a, b in zip(bases_a, bases_b, strict=True):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(first["input"].data, second["input"].data)
    target_basis_a = fit_svd_basis(_slices(_subject(1)["target"])[:1], 1)
    target_basis_b = fit_svd_basis(_slices(_subject(7)["target"])[:1], 1)
    assert not torch.allclose(target_basis_a, target_basis_b)


def test_the_basis_is_fitted_on_the_masked_input_not_the_whole_one() -> None:
    masked = _masked()
    unmasked_basis = fit_svd_basis(_slices(_subject()["input"])[:1], 1)
    masked_basis = fit_svd_basis(_slices(masked["input"])[:1], 1)
    bases = compress_subject(masked, 1)
    torch.testing.assert_close(bases[0], masked_basis)
    assert not torch.allclose(bases[0], unmasked_basis)


def test_it_refuses_to_run_before_the_masking() -> None:
    """Planted: declared first, the basis would see every unacquired line."""
    with pytest.raises(ValueError, match="after complementary_rep_masking"):
        build_transform("svd_coil_compression", num_virtual_coils=1)(_subject())


@pytest.mark.parametrize("key", ["sensitivity_complex", "noise_covariance"])
def test_it_refuses_physical_coil_keys(key) -> None:
    subject = _masked()
    if key == "noise_covariance":
        subject[key] = torch.eye(_C, dtype=torch.complex64)
    else:
        maps = torch.ones(_C, _H, _W, _S, dtype=torch.complex64)
        subject.add_image(tio.ScalarImage(tensor=maps), key)
    with pytest.raises(ValueError, match="physical"):
        build_transform("svd_coil_compression", num_virtual_coils=1)(subject)
