"""UniversalReconstructionStrategy: the acquisition prompt reaches the prompt-aware backbone.

The strategy resolves through ``TrainingStrategyFactory`` and the backbone through the model
registry. The heavy ``ReconstructionTrainingStrategy.__init__`` is bypassed (``__new__``) and
the parent's input preparation is stubbed to the identity, so these tests exercise only the
seam this strategy owns: metadata -> ``[B, 4]`` prompt -> ``acquisition_prompt`` kwarg.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from spectramr.data.batch_types import TrainingBatch
from spectramr.infrastructure.training.strategies import universal_reconstruction_strategy as urs
from spectramr.infrastructure.training.strategies.mixins.model_validation import (
    ModelValidationMixin,
)
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.models.registry import get_model_class

_META_A = {"TR": [2750.0, 2750.0], "TE": [27.0, 27.0], "FA": [90.0, 90.0], "B0": [2.89, 2.89]}
_META_B = {"TR": [3000.0, 3000.0], "TE": [33.0, 33.0], "FA": [140.0, 140.0], "B0": [1.49, 1.49]}


def _resolved_strategy_cls() -> type:
    config = SimpleNamespace(
        training=SimpleNamespace(strategy_class=None, training_mode="universal_reconstruction")
    )
    return TrainingStrategyFactory().get_strategy_class(config)


def _backbone() -> nn.Module:
    model_cls = get_model_class("prompt_lora_hermitian_fno")
    torch.manual_seed(0)
    model = model_cls(
        in_channels=2, out_channels=2, width=8, modes=4, depth=2, lora_rank=2, prompt_embed_dim=8
    )
    # B is zero-initialised (identity at step 0); give it weight so the prompt is visible.
    for block in model.pointwise:
        nn.init.normal_(block.lora_up.weight, std=0.5)
    return model


def _strategy(generator: nn.Module, monkeypatch: pytest.MonkeyPatch) -> Any:
    cls = _resolved_strategy_cls()
    assert cls is urs.UniversalReconstructionStrategy
    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_prepare_generator_inputs",
        lambda self, ctx, x, validation=False: (x, {}),
    )
    strategy = object.__new__(cls)
    strategy.env = SimpleNamespace(generator=generator)
    return strategy


def _forward(strategy: Any, x: torch.Tensor, meta: Any) -> torch.Tensor:
    _, kwargs = strategy._prepare_generator_inputs({urs.ACQUISITION_METADATA_KEY: meta}, x)
    return strategy.env.generator(x, **kwargs)


def test_prompt_changes_the_reconstruction_and_trains_the_adapters(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    x = torch.randn(2, 2, 16, 16)
    out_a = _forward(strategy, x, _META_A)
    out_b = _forward(strategy, x, _META_B)
    assert not torch.allclose(out_a, out_b), "acquisition prompt had no effect on the output"
    out_a.abs().mean().backward()
    gen = strategy.env.generator
    assert gen.pointwise[0].alpha_mlp.weight.grad.abs().sum() > 0
    assert gen.prompt_embedding.mlp[0].weight.grad.abs().sum() > 0


def test_per_sample_metadata_dicts_are_accepted(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    x = torch.randn(2, 2, 16, 16)
    rows = [{k: v[i] for k, v in _META_A.items()} for i in range(2)]
    assert torch.allclose(_forward(strategy, x, rows), _forward(strategy, x, _META_A))


def test_batch_context_lifts_metadata_from_the_training_batch(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_prepare_batch_context_reconstruction",
        lambda self, x, y, **kw: {"use_dc": False, "measured_kspace": None},
        raising=False,
    )
    x = torch.randn(2, 2, 16, 16)
    batch = TrainingBatch(input=x, target=x, metadata={urs.ACQUISITION_METADATA_KEY: _META_A})
    ctx = strategy._prepare_batch_context(x, x, batch=batch)
    assert ctx[urs.ACQUISITION_METADATA_KEY] is _META_A


def test_validation_step_conditions_on_the_batch_prompt(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    seen: dict[str, Any] = {}

    def _capture(self, input_batch, target_batch, *, batch_data=None, **kwargs):
        seen.update(kwargs["batch_context"])
        return {}

    monkeypatch.setattr(ModelValidationMixin, "validation_step", _capture)
    x = torch.randn(2, 2, 16, 16)
    batch = TrainingBatch(input=x, target=x, metadata={urs.ACQUISITION_METADATA_KEY: _META_B})
    strategy.validation_step(x, x, batch_data=batch)
    assert seen[urs.ACQUISITION_METADATA_KEY] is _META_B
    # The context the validation forward receives conditions the backbone the same way.
    assert torch.allclose(
        _forward(strategy, x, seen[urs.ACQUISITION_METADATA_KEY]), _forward(strategy, x, _META_B)
    )


def test_missing_metadata_raises(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    with pytest.raises(KeyError, match="acquisition_metadata"):
        strategy._prepare_generator_inputs({}, torch.randn(2, 2, 16, 16))


def test_missing_field_and_nan_raise(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    x = torch.randn(2, 2, 16, 16)
    with pytest.raises(KeyError, match="FA"):
        _forward(strategy, x, {k: v for k, v in _META_A.items() if k != "FA"})
    with pytest.raises(ValueError, match="not finite"):
        _forward(strategy, x, {**_META_A, "TE": [float("nan"), 27.0]})


def test_row_mismatch_raises(monkeypatch) -> None:
    strategy = _strategy(_backbone(), monkeypatch)
    with pytest.raises(ValueError, match="row"):
        _forward(strategy, torch.randn(3, 2, 16, 16), _META_A)


def test_a_backbone_that_cannot_take_the_prompt_is_refused(monkeypatch) -> None:
    strategy = _strategy(
        get_model_class("hermitian_fno")(in_channels=2, out_channels=2, width=8, modes=4, depth=1),
        monkeypatch,
    )
    with pytest.raises(ValueError, match="inert"):
        strategy._prepare_generator_inputs(
            {urs.ACQUISITION_METADATA_KEY: _META_A}, torch.randn(2, 2, 16, 16)
        )


def test_the_declared_losses_run_through_the_parent_objective() -> None:
    cls = urs.UniversalReconstructionStrategy
    assert "_compute_losses_impl" not in cls.__dict__
    assert cls.forwards_loss_components


_FASTMRI_HEADER = (
    '<?xml version="1.0"?><ismrmrdHeader xmlns="http://www.ismrm.org/ISMRMRD">'
    "<acquisitionSystemInformation><systemFieldStrength_T>{b0}</systemFieldStrength_T>"
    "</acquisitionSystemInformation><sequenceParameters><TR>{tr}</TR><TE>{te}</TE>"
    "<TI>100</TI><flipAngle_deg>{fa}</flipAngle_deg></sequenceParameters></ismrmrdHeader>"
)


def test_the_arm_metadata_block_emits_the_prompt_from_a_fastmri_header(
    tmp_path, monkeypatch
) -> None:
    """File header -> LoadAcquisitionMetadata (the arm's block) -> collate -> batch -> prompt."""
    h5py = pytest.importorskip("h5py")
    import torchio as tio
    from torch.utils.data import default_collate

    from spectramr.config.schemas.data import AcquisitionMetadataConfigSchema
    from spectramr.data.batch_types import BatchAdapter
    from spectramr.data.transforms.acquisition_metadata import LoadAcquisitionMetadata

    scans = [(2750.0, 27.0, 90.0, 2.89362), (3000.0, 33.0, 140.0, 1.494)]
    transform = LoadAcquisitionMetadata(
        AcquisitionMetadataConfigSchema(
            enabled=True, source="ismrmrd_header", fields=["TR", "TE", "FA", "B0"], strict=True
        )
    )
    samples = []
    for i, (tr, te, fa, b0) in enumerate(scans):
        path = tmp_path / f"file{i}.h5"
        with h5py.File(path, "w") as f:
            f["ismrmrd_header"] = _FASTMRI_HEADER.format(tr=tr, te=te, fa=fa, b0=b0)
        image = torch.zeros(2, 16, 16, 1)
        subject = tio.Subject(input=tio.ScalarImage(tensor=image), source_path=str(path))
        meta = transform(subject)[urs.ACQUISITION_METADATA_KEY]
        samples.append(
            {"input": image[..., 0], "target": image[..., 0], urs.ACQUISITION_METADATA_KEY: meta}
        )
    batch = BatchAdapter.from_dict(default_collate(samples))

    strategy = _strategy(_backbone(), monkeypatch)
    monkeypatch.setattr(
        ReconstructionTrainingStrategy,
        "_prepare_batch_context_reconstruction",
        lambda self, x, y, **kw: {"use_dc": False, "measured_kspace": None},
        raising=False,
    )
    ctx = strategy._prepare_batch_context(batch.input, batch.target, batch=batch)
    _, kwargs = strategy._prepare_generator_inputs(ctx, batch.input)
    assert torch.allclose(kwargs["acquisition_prompt"], torch.tensor(scans))
