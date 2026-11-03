"""Which ``dataset_type`` values serve k-space (A4's supporting fact).

``subject["input"]`` means opposite things across this line: on an image arm it
is an image, on a k-space arm it IS the measured k-space. A transform that
resolves its source by key name must know which — not knowing is how
``PhysicsSynchronization`` came to apply a second forward FFT to measured data.
"""

from __future__ import annotations

import pytest

from spectramr.data.signal_domain import KSPACE_DATASET_TYPES, is_kspace_dataset_type


@pytest.mark.parametrize(
    "dataset_type",
    ["kspace", "m4raw", "fastmri", "fastmri_brain", "fastmri_multicoil"],
)
def test_known_kspace_families(dataset_type: str) -> None:
    assert is_kspace_dataset_type(dataset_type)


@pytest.mark.parametrize("dataset_type", ["bart_kspace", "fastmri_kspace", "my_kspace_v2"])
def test_the_substring_rule_catches_spellings_not_enumerated(dataset_type: str) -> None:
    """The corpus spells the family several ways and enumerating every one has
    already drifted, so membership is the set OR a substring test."""
    assert is_kspace_dataset_type(dataset_type)


@pytest.mark.parametrize(
    "dataset_type", ["nifti", "nifti_paired", "mrixfields", "preprocessed", "synthetic"]
)
def test_image_families_are_not_kspace(dataset_type: str) -> None:
    assert not is_kspace_dataset_type(dataset_type)


def test_none_is_image_domain() -> None:
    """Matches the schema default; a missing dataset_type must not be treated
    as k-space, since that would skip a sync an image arm needs."""
    assert not is_kspace_dataset_type(None)
    assert not is_kspace_dataset_type("")


def test_spec_card_uses_this_and_does_not_keep_its_own_copy() -> None:
    """It was a function-local constant in `infrastructure/validation/`, which
    the data layer cannot import (#5). Two copies of a fact that decides
    behaviour is how they drift."""
    import inspect

    from spectramr.infrastructure.validation import spec_card

    source = inspect.getsource(spec_card)
    assert "is_kspace_dataset_type" in source
    assert "KSPACE_DATASET_TYPES = {" not in source, (
        "spec_card redeclared the set instead of importing it"
    )


def test_the_set_is_immutable() -> None:
    """A mutable module-level set invites a caller to 'just add one'."""
    assert isinstance(KSPACE_DATASET_TYPES, frozenset)


def test_phase_contrast_slices_are_kspace() -> None:
    """flow_synthetic's input is multicoil k-space; calling it an image fails the audit's domain check."""
    assert is_kspace_dataset_type("flow_synthetic")
