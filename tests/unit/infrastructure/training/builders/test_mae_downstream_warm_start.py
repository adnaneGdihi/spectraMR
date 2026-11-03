"""exp_06 MAE pretraining -> exp_06_mae_downstream_probe ULF->HF synthesis.

The downstream arm is graded on val_psnr only if it really starts from the MAE weights:
``model.checkpoint_path`` loads with the ``overlap`` policy, which accepts a partial match,
so a drifted architecture kwarg would warm-start part of the trunk without complaint.
"""

from pathlib import Path

import pytest
import torch

from spectramr.config.settings import TrainingSettings
from spectramr.infrastructure.training.builders.model_builder import ModelBuilder
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from tests.fixtures.run_state_files import write_run_state

_COHORT = Path("experiments/inprogress/10_paradigms")
_PRETRAIN = _COHORT / "exp_06_masked_autoencoder.yaml"
_DOWNSTREAM = _COHORT / "exp_06_mae_downstream_probe.yaml"
_TINY = {
    "spatial_dims": 2,
    "img_size": [16, 16],
    "patch_size": 4,
    "embed_dim": 16,
    "encoder_depth": 1,
    "decoder_depth": 1,
    "num_heads": 2,
    "mlp_ratio": 2.0,
}


@pytest.fixture(scope="module")
def arms() -> tuple[TrainingSettings, TrainingSettings]:
    return TrainingSettings.from_yaml(str(_PRETRAIN)), TrainingSettings.from_yaml(str(_DOWNSTREAM))


def test_downstream_architecture_is_the_pretraining_architecture(arms) -> None:
    pre, down = arms
    pre_kw = {k: v for k, v in pre.model.model_kwargs.items() if k != "mask_ratio"}
    down_kw = {k: v for k, v in down.model.model_kwargs.items() if k != "mask_ratio"}
    assert down.model.model_type == pre.model.model_type == "mae_mri"
    assert down_kw == pre_kw
    # Every patch is encoded, so the decoder synthesises the whole HF image.
    assert down.model.model_kwargs["mask_ratio"] == 0.0


def test_downstream_warm_starts_from_the_pretraining_run(arms) -> None:
    pre, down = arms
    ckpt = Path(down.model.checkpoint_path)
    assert ckpt.parent.parent == Path(pre.training.output_dir)
    assert down.checkpoint.produced_by_arm == pre.metadata.name


def test_downstream_is_graded_on_the_cohort_axis(arms) -> None:
    _, down = arms
    assert down.metrics.best_metric_name == "val_psnr"
    assert down.early_stopping.metric == "val_psnr"
    assert down.data.pairing.bidirectional_mode == "ulf_to_hf"
    cls = TrainingStrategyFactory().get_strategy_class(down)
    assert cls.__name__ == "ReconstructionTrainingStrategy"


def test_production_builder_loads_the_mae_weights_into_the_downstream_model(arms, tmp_path) -> None:
    """ModelBuilder(warm_start=True), the director's call, on the downstream arm."""
    pre, down = arms
    pre_cfg = pre.model_copy(
        update={
            "model": pre.model.model_copy(update={"model_kwargs": {**_TINY, "mask_ratio": 0.75}})
        }
    )
    torch.manual_seed(0)
    mae = ModelBuilder(pre_cfg, torch.device("cpu")).build_generator().build()["generator"]
    ckpt = write_run_state(tmp_path / "best.safetensors", {"generator": mae.state_dict()})

    down_model = down.model.model_copy(
        update={"model_kwargs": {**_TINY, "mask_ratio": 0.0}, "checkpoint_path": str(ckpt)}
    )
    down_cfg = down.model_copy(update={"model": down_model})
    torch.manual_seed(1)
    builder = ModelBuilder(down_cfg, torch.device("cpu"), warm_start=True)
    synth = builder.build_generator().build()["generator"]

    for (name, a), (_, b) in zip(mae.state_dict().items(), synth.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    x = torch.rand(2, 1, 16, 16)
    mae.eval()
    synth.eval()
    with torch.no_grad():
        expected = mae(x, disable_spatial_masking=True)
        assert torch.allclose(synth(x), expected)
