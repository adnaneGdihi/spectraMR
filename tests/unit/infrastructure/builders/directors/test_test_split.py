"""The held-out test loader is the validation pipeline over the test manifest, uncapped."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.builders.directors.test_split import (
    config_for_test_split,
    declares_test_set,
)

ARM = "ei_multicoil_m4raw_r4"


@pytest.fixture(scope="module")
def config():
    from tests.fixtures.self_supervised_batches import load_arm

    return load_arm(ARM)


def test_arm_declares_its_test_set(config):
    assert declares_test_set(config)


def test_validation_split_becomes_the_test_manifest(config):
    swapped = config_for_test_split(config)
    source = swapped.data.source
    assert source.validation_index_path == config.data.source.test_index_path
    assert source.index_path == config.data.source.test_index_path
    assert source.test_index_path is None
    assert config.data.source.validation_index_path != config.data.source.test_index_path


def test_the_original_config_is_untouched(config):
    before = config.data.source.model_dump()
    config_for_test_split(config)
    assert config.data.source.model_dump() == before


def test_a_capped_validation_is_not_a_capped_test(config):
    capped_loader = config.validation.loader.model_copy(update={"num_batches": 3, "num_samples": 9})
    capped = config.model_copy(
        update={"validation": config.validation.model_copy(update={"loader": capped_loader})}
    )
    loader = config_for_test_split(capped).validation.loader
    assert (loader.num_batches, loader.num_samples) == (None, None)


def test_no_declared_test_set_raises_instead_of_borrowing_validation(config):
    source = config.data.source.model_copy(update={"test_index_path": None})
    bare = config.model_copy(update={"data": config.data.model_copy(update={"source": source})})
    assert not declares_test_set(bare)
    with pytest.raises(ValueError, match="No held-out test set"):
        config_for_test_split(bare)


def test_config_owned_paired_test_is_declared_only_when_the_partition_names_one():
    from types import SimpleNamespace

    from spectramr.config.schemas.data import DataSplitConfigSchema
    from spectramr.infrastructure.builders.directors.test_split import declares_test_set

    def cfg(**split):
        data = SimpleNamespace(
            source=SimpleNamespace(test_index_path=None, paired_manifest_path="m.json"),
            split=DataSplitConfigSchema(**split),
            enable_test_split=False,
        )
        return SimpleNamespace(data=data)

    assert declares_test_set(cfg(type="subjects", test_subjects=["07"]))
    assert not declares_test_set(cfg(type="subjects", val_subjects=["02"]))


def test_the_test_loader_scores_one_slice_per_batch(config, monkeypatch):
    from spectramr.infrastructure.builders.directors import data_pipeline_director, test_split

    seen = {}

    class _Director:
        def __init__(self, cfg):
            pass

        def build_dataloaders(self, **kw):
            seen.update(kw)
            return None, "loader"

    monkeypatch.setattr(data_pipeline_director, "DataPipelineDirector", _Director)
    assert test_split.build_test_loader(config, num_workers=0) == "loader"
    assert seen == {"num_workers": 0, "val_batch_size": test_split.TEST_BATCH_SIZE}
