"""The one predicate for legal ``validation.scoring.compute`` names (#2819)."""

from __future__ import annotations

import pytest

from spectramr.core.metrics.validation_names import (
    UNIVERSAL_VALIDATION_KEYS,
    is_emitted,
    partition_validation_compute,
)

_REGISTERED = {"psnr", "ssim"}


def _registered(name: str) -> bool:
    return name.lower() in _REGISTERED


_U = UNIVERSAL_VALIDATION_KEYS


def test_val_loss_is_owned_by_a_path_that_writes_it() -> None:
    assert "val_loss" in UNIVERSAL_VALIDATION_KEYS
    assert partition_validation_compute(
        ["val_loss", "loss"], frozenset(), _registered, universal=_U
    ) == ([], [])


def test_val_loss_is_unowned_on_a_path_that_does_not_write_it() -> None:
    """PLANTED (#2838): the exemption is the caller's to grant, per validation path."""
    assert partition_validation_compute(
        ["val_loss", "loss"], frozenset(), _registered, universal=frozenset()
    ) == ([], ["val_loss", "loss"])


def test_the_universal_set_has_no_default() -> None:
    """A caller that forgets the path question must fail, not inherit the exemption."""
    with pytest.raises(TypeError):
        partition_validation_compute(["val_loss"], frozenset(), _registered)  # type: ignore[call-arg]


def test_registered_names_go_to_the_computer_and_emitted_names_do_not() -> None:
    registry, unowned = partition_validation_compute(
        ["psnr", "field_mse", "SSIM"], frozenset({"val_field_mse"}), _registered, universal=_U
    )
    assert registry == ["psnr", "SSIM"]
    assert unowned == []


def test_a_name_no_producer_owns_is_unowned() -> None:
    assert partition_validation_compute(
        ["val_not_a_metric"], frozenset(), _registered, universal=_U
    ) == (
        [],
        ["val_not_a_metric"],
    )


def test_a_cascade_suffix_is_not_stripped() -> None:
    """The computer receives the name verbatim, so ``psnr_4x`` is not ``psnr``."""
    assert partition_validation_compute(["psnr_4x"], frozenset(), _registered, universal=_U) == (
        [],
        ["psnr_4x"],
    )
    assert not is_emitted("field_mse_4x", frozenset({"val_field_mse"}), universal=_U)


def test_the_val_prefix_is_normalised_both_ways() -> None:
    assert is_emitted("field_mse", frozenset({"val_field_mse"}), universal=_U)
    assert is_emitted("val_field_mse", frozenset({"field_mse"}), universal=_U)
