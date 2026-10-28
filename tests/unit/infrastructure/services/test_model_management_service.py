"""ModelManagementService owns the model registry, not checkpoint IO (#2875)."""

import pytest

from spectramr.infrastructure.services.model_management_service import ModelManagementService


def test_checkpoint_io_lives_in_the_checkpointing_package_only(tmp_path):
    service = ModelManagementService(base_dir=str(tmp_path / "models"))
    assert not hasattr(service, "save_model_checkpoint")
    assert not hasattr(service, "load_model_checkpoint")
    assert not hasattr(service, "checkpoint_service")


def test_a_checkpoint_service_argument_is_refused(tmp_path):
    with pytest.raises(TypeError):
        ModelManagementService(base_dir=str(tmp_path / "models"), checkpoint_service=object())
