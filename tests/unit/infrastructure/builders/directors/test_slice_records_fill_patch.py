from types import SimpleNamespace

import pytest

from spectramr.config.schemas.validation import ValidationLoaderConfigSchema
from spectramr.infrastructure.builders.directors.data_pipeline_director import (
    _slice_records_fill_patch,
    _slice_records_val_inherits_workers,
)


def _case(*, dataset_type="m4raw", slice_level=True, sampler="uniform", spv=1, patch=(8, 8, 1)):
    data = SimpleNamespace(dataset_type=dataset_type, slice_level_records=slice_level)
    queue = SimpleNamespace(sampler_type=sampler, samples_per_volume=spv, patch_size=patch)
    ds = SimpleNamespace(index=[{"shape": (1, 4, 8, 8)}, {"shape": (1, 4, 8, 8)}])
    return data, ds, queue


def test_full_slice_patch_on_slice_records_bypasses_the_queue():
    assert _slice_records_fill_patch(*_case()) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset_type": "fastmri_kspace"},
        {"slice_level": False},
        {"sampler": "label"},
        {"spv": 2},
        {"patch": (4, 4, 1)},
    ],
)
def test_every_other_shape_keeps_the_queue(kwargs):
    assert _slice_records_fill_patch(*_case(**kwargs)) is False


def test_a_dataset_without_shape_records_keeps_the_queue():
    data, _, queue = _case()
    assert _slice_records_fill_patch(data, SimpleNamespace(index=None), queue) is False
    assert (
        _slice_records_fill_patch(data, SimpleNamespace(index=[{"shape": (1, 8, 9)}]), queue)
        is False
    )


def _val(declared: bool):

    loader = (
        ValidationLoaderConfigSchema(num_workers=0) if declared else ValidationLoaderConfigSchema()
    )
    return SimpleNamespace(loader=loader)


@pytest.mark.parametrize(
    ("dataset_type", "slice_level", "declared", "expected"),
    [
        ("m4raw", True, False, True),
        ("m4raw", True, True, False),
        ("m4raw", False, False, False),
        ("fastmri_kspace", True, False, False),
    ],
)
def test_validation_inherits_workers_only_when_undeclared(
    dataset_type, slice_level, declared, expected
):
    data = SimpleNamespace(dataset_type=dataset_type, slice_level_records=slice_level)
    assert _slice_records_val_inherits_workers(data, _val(declared)) is expected
