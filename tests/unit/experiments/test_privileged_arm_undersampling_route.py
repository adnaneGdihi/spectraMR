"""The privileged-marker arm masks its input, and the marker is sized from the masked image.

Without ``data.trajectory: cartesian`` the arm's ``undersampling`` block reached no route and
the student trained on fully sampled input. With it, ``rss_image`` is deferred until after the
mask, so ``synthetic_marker`` must still run after the coil combine, on the image ``input``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torchio as tio

from spectramr.config.settings import TrainingSettings
from spectramr.data.builders.torchio_transform_builder import (
    TorchIOTransformBuilder,
    TorchIOTransformConfig,
)
from spectramr.data.transforms.kspace_coil_transforms import CoilCombineTransform
from spectramr.data.transforms.synthetic_marker import SyntheticMarkerTransform
from spectramr.data.transforms.tio_physics import PhysicsInformedMasking

ARM = (
    Path(__file__).resolve().parents[3]
    / "experiments/inprogress/spec_tasks/task_method_iii_privileged.yaml"
)


def _index(chain: tio.Compose, cls: type) -> int:
    kinds = [type(t) for t in chain.transforms]
    assert cls in kinds, f"{cls.__name__} missing from {[k.__name__ for k in kinds]}"
    return kinds.index(cls)


@pytest.mark.parametrize("build", ["build_train_transforms", "build_val_transforms"])
def test_mask_then_combine_then_marker(build: str) -> None:
    config = TorchIOTransformConfig.from_settings(TrainingSettings.from_yaml(str(ARM)))
    assert config.trajectory_type == "cartesian"
    chain = getattr(TorchIOTransformBuilder, build)(config)
    mask = _index(chain, PhysicsInformedMasking)
    combine = _index(chain, CoilCombineTransform)
    if build == "build_train_transforms":
        assert mask < combine < _index(chain, SyntheticMarkerTransform)
    else:
        assert mask < combine


def test_domain_term_is_on_and_fed_by_the_labelled_balancer() -> None:
    settings = TrainingSettings.from_yaml(str(ARM))
    md = settings.data.multi_domain
    assert md.enabled and md.balancing == "domain_labelled"
    assert [d.name for d in md.domains] == ["synthetic", "real"]
    assert md.domains[1].index_path == "data/manifests/m4raw_motion_subset_held_out.json"
    assert settings.training.privileged.delta > 0


def test_the_director_dispatches_the_arm_to_the_labelled_balancer(monkeypatch) -> None:
    from spectramr.data.builders.site_balancer import DomainLabelledBalancer
    from spectramr.infrastructure.builders.directors.data_pipeline_director import (
        DataPipelineDirector,
    )

    seen: list[str] = []

    def fake_build(self, **_kwargs):
        seen.append(self._config.data.source.index_path)
        return [{"input": torch.zeros(1, 1, 2, 2)}], None

    monkeypatch.setattr(DataPipelineDirector, "build_dataloaders", fake_build)
    director = DataPipelineDirector(TrainingSettings.from_yaml(str(ARM)))
    balancer = director.build_multi_domain_dataloaders(num_workers=0, pin_memory=False)
    assert isinstance(balancer, DomainLabelledBalancer)
    assert seen == [
        "data/manifests/m4raw_train.json",
        "data/manifests/m4raw_motion_subset_held_out.json",
    ]
    assert next(iter(balancer))["domain_label"].tolist() == [1.0, 0.0]
