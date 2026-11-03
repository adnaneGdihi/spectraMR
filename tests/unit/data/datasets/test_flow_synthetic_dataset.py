"""``dataset_type: flow_synthetic``: contract, physics consistency and the real batch path."""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.config.schemas.data import DataConfigSchema
from spectramr.data.datasets.flow_sample import check_flow_sample
from spectramr.data.datasets.flow_synthetic_dataset import create_flow_synthetic_datasets
from spectramr.infrastructure.physics.fft_ops import ifftnc
from spectramr.infrastructure.physics.signal_models.flow_encoding import referenced_decode_images

GRID = (16, 24, 20)


def _config(**block) -> DataConfigSchema:
    flow = {
        "enabled": True,
        "train_subjects": 2,
        "val_subjects": 1,
        "grid": GRID,
        "voxel_size_mm": (1.0, 1.0, 1.0),
        "frames": 4,
        "coils": 2,
        "noise_std": 0.0,
        "background_amplitude": 0.0,
        "acceleration": 3.0,
        "center_fraction": 0.2,
        **block,
    }
    return DataConfigSchema(
        dataset_type="flow_synthetic",
        phase_contrast={"enabled": True, "venc": 1.5},
        flow_synthetic=flow,
    )


def _datasets(**block):
    return create_flow_synthetic_datasets(_config(**block), None, None)


def test_items_honour_the_flow_contract() -> None:
    train, val = _datasets()
    assert len(train) == 2 * GRID[0] and len(val) == GRID[0]
    item = train[3]
    check_flow_sample(item)
    assert item["input"].shape == (2, 4, 4, 24, 20) and item["input"].dtype == torch.complex64
    assert item["velocity"].shape == (3, 4, 24, 20)
    assert (item["subject_id"], item["slice_index"], item["n_slices"]) == ("train-000", 3, 16)


def test_noise_free_kspace_is_the_coil_weighted_reference_and_input_is_masked() -> None:
    item = _datasets()[0][5]
    coil_images = ifftnc(item["target"], dims=(-2, -1))
    assert torch.allclose(
        coil_images, item["coil_maps"][:, None, None] * item["image_ref"][None], atol=1e-5
    )
    assert torch.equal(item["input"], item["target"] * item["mask"])


def test_decoding_the_reference_images_returns_the_velocity() -> None:
    """Without background phase, the conjugate-product decode of image_ref is the target velocity."""
    item = _datasets()[0][7]
    decoded = referenced_decode_images(
        item["image_ref"], item["venc"], "four_point_reference", dim=0
    )
    signal = item["image_ref"][0].abs() > 0
    assert torch.allclose(decoded[:, signal], item["velocity"][:, signal], atol=1e-4)


def test_a_subject_keeps_one_mask_and_one_phantom_across_slices() -> None:
    train, val = _datasets()
    assert torch.equal(train[0]["mask"], train[9]["mask"])
    assert not torch.equal(train[0]["mask"], train[GRID[0]]["mask"])
    assert not torch.equal(train[4]["velocity"], val[4]["velocity"])
    assert torch.equal(train[4]["target"], train[4]["target"])


def test_assembled_volume_conserves_mass_through_the_emitted_planes() -> None:
    """Stack one subject's slices along X; component i must pair with axis i."""
    from spectramr.core.metrics.context import MetricContext
    from spectramr.core.metrics.flow_metrics import ConsistencyOfMass

    _, val = _datasets()
    items = [val[x] for x in range(GRID[0])]
    velocity = torch.stack([i["velocity"] for i in items], dim=2)[None]  # [1, 3, T, X, Y, Z]
    ctx = MetricContext(
        voxel_size_mm=items[0]["voxel_size_mm"],
        inlet_mask=torch.stack([i["inlet_mask"] for i in items], dim=1),
        outlet_mask=torch.stack([i["outlet_mask"] for i in items], dim=1),
    )
    assert ctx.inlet_mask.sum() > 0 and ctx.outlet_mask.sum() > 0
    assert ConsistencyOfMass()(velocity, context=ctx).item() < 5.0


@pytest.mark.parametrize(
    ("data", "match"),
    [
        ({"phase_contrast": {"enabled": False}}, "data.phase_contrast"),
        ({"augmentation": {"enabled": True}}, "augmentation"),
        ({"flow_synthetic": {"enabled": False}}, "flow_synthetic.enabled"),
    ],
)
def test_the_creator_refuses_inert_or_missing_blocks(data, match) -> None:
    cfg = _config().model_copy(
        update={k: getattr(_config(), k).model_copy(update=v) for k, v in data.items()}
    )
    with pytest.raises(ValueError, match=match):
        create_flow_synthetic_datasets(cfg, None, None)


def test_index_paths_are_refused_so_no_test_pass_scores_validation_phantoms() -> None:
    cfg = _config()
    source = cfg.source.model_copy(update={"test_index_path": "/data/test.json"})
    with pytest.raises(ValueError, match="test_index_path"):
        create_flow_synthetic_datasets(cfg.model_copy(update={"source": source}), None, None)


def test_a_batch_survives_the_real_collate_adapter_and_validation_preprocess() -> None:
    """Every key comes back bit-identical through the path train/val batches take."""
    from spectramr.data.batch_types import BatchAdapter, reject_targetless_samples
    from spectramr.data.collation.strategies import CollateStrategyFactory
    from spectramr.data.collation.strategy_selector import CollationStrategySelector
    from spectramr.data.collation.unpad import split_padded_batches
    from spectramr.data.datasets.axis_exposure import resolve_axes_for
    from spectramr.pipelines.train import _preprocess_validation_tensor

    cfg = _config()
    _, val = create_flow_synthetic_datasets(cfg, None, None)
    name = CollationStrategySelector.AUTO_SELECT_MAP[cfg.dataset_type]
    batch = CollateStrategyFactory.create(name).collate([val[2], val[3]])
    reference = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    (batch,) = list(split_padded_batches([batch]))
    tb = BatchAdapter.from_dict(batch, axes=resolve_axes_for(cfg))
    reject_targetless_samples(tb, strategy="PhaseContrastReconStrategy")
    tb = tb.to("cpu")
    pipeline_cfg = types.SimpleNamespace(model=types.SimpleNamespace(spatial_dims=2), data=cfg)
    tb.input = _preprocess_validation_tensor(tb.input, pipeline_cfg)
    tb.target = _preprocess_validation_tensor(tb.target, pipeline_cfg)

    assert torch.equal(tb.input, reference["input"]) and tb.input.dtype == torch.complex64
    assert torch.equal(tb.target, reference["target"])
    assert torch.equal(tb.mask, reference["mask"])
    assert torch.equal(tb.coil_maps, reference["coil_maps"])
    for key, value in reference.items():
        if key in {"input", "target", "mask", "coil_maps"}:
            continue
        got = tb.get(key)
        if isinstance(value, torch.Tensor):
            assert torch.equal(got, value) and got.dtype == value.dtype, key
        else:
            assert got == value, key
    assert reference["subject_id"] == ["val-000", "val-000"]
