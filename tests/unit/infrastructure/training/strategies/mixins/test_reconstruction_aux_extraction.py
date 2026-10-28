"""Auxiliary tensors must survive the loop's batch conversion.

The loop converts the collated dict to a ``TrainingBatch`` *before* calling
``train_step`` (``training_loop.py``: ``BatchAdapter.from_dict``). That
dataclass is not a Mapping, so ``_prepare_batch_context_reconstruction``'s
``isinstance(raw_batch, dict)`` guard was False on the production path and the
whole auxiliary-tensor extraction underneath it never ran.

Nothing caught it. The audit passes 165 checks on an arm whose loader demonstrably
produces both tensors, ``audit --probe`` builds the model on a synthetic phantom
and never opens the dataset, and the strategy's own unit tests exercise the EI
math directly rather than through a batch. The failure surfaced on the first
gradient step of a cluster run:

    ValueError: EquivariantImagingStrategy requires batch_context['measured_kspace']
    (undersampled k-space y) and batch_context['mask'] (acquisition mask).

These tests drive the real ``BatchAdapter.from_dict`` rather than a hand-built
fixture, because the bug lives in the gap between what that produces and what
the mixin expects -- a fixture agreeing with the mixin would be green and blind.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.data.batch_types import BatchAdapter
from spectramr.infrastructure.training.strategies.mixins.reconstruction import (
    ReconstructionMixin,
    _as_aux_mapping,
)

_B, _C, _S = 2, 2, 16


class _Probe(ReconstructionMixin):
    """Minimal host: the mixin only reads ``config.model.input_type``."""

    def __init__(self, input_type: str = "image") -> None:
        self.env = None
        self._input_type = input_type

    @property
    def state(self):
        input_type = self._input_type

        class _Model:
            pass

        _Model.input_type = input_type

        class _Config:
            model = _Model()

        class _State:
            config = _Config()

        return _State()


def _collated() -> dict[str, torch.Tensor]:
    """What the loader hands the loop for a masked k-space arm."""
    return {
        "input": torch.randn(_B, _C, _S, _S),
        "target": torch.randn(_B, _C, _S, _S),
        "mask": torch.ones(_B, 1, _S, _S),
        "kspace": torch.randn(_B, _C, _S, _S),
    }


class TestTheRegression:
    def test_a_training_batch_is_not_a_mapping(self) -> None:
        """The premise. If this ever becomes True the guard was never the bug."""
        batch = BatchAdapter.from_dict(_collated())
        assert not isinstance(batch, dict)

    def test_from_dict_keeps_everything_the_mixin_needs(self) -> None:
        """mask is a bound field; every other key lands in metadata."""
        batch = BatchAdapter.from_dict(_collated())
        assert batch.mask is not None
        assert "kspace" in batch.metadata

    @pytest.mark.parametrize("input_type", ["image", "kspace"])
    def test_both_tensors_reach_the_context_from_a_training_batch(self, input_type: str) -> None:
        """The exact pair EquivariantImagingStrategy raises without."""
        batch = BatchAdapter.from_dict(_collated())
        context = _Probe(input_type)._prepare_batch_context_reconstruction(
            batch.input, batch.target, batch=batch
        )
        assert context.get("measured_kspace") is not None, "measured_kspace was dropped"
        assert context.get("mask") is not None, "mask was dropped"

    def test_a_plain_dict_still_works(self) -> None:
        """The pre-existing path must not regress."""
        raw = _collated()
        context = _Probe()._prepare_batch_context_reconstruction(
            raw["input"], raw["target"], batch=raw
        )
        assert context.get("measured_kspace") is not None
        assert context.get("mask") is not None


class TestTheFlattener:
    def test_a_dict_passes_through_unchanged(self) -> None:
        raw = _collated()
        assert _as_aux_mapping(raw) is raw

    def test_coil_maps_is_rebound_to_the_dataset_spelling(self) -> None:
        """The dataclass field is ``coil_maps``; the key mapping reads other names."""
        raw = _collated()
        raw["coil_sensitivities"] = torch.randn(_B, 4, _S, _S)
        flat = _as_aux_mapping(BatchAdapter.from_dict(raw))
        assert "coil_sensitivities" in flat
        assert "coil_maps" not in flat

    def test_metadata_does_not_shadow_a_bound_field(self) -> None:
        """A bound tensor wins over a same-named metadata entry."""
        batch = BatchAdapter.from_dict(_collated())
        batch.metadata["mask"] = torch.zeros(1)
        flat = _as_aux_mapping(batch)
        assert flat["mask"].shape == (_B, 1, _S, _S)

    @pytest.mark.parametrize(
        "value", [None, {}, 42, "not-a-batch", object()], ids=["none", "empty", "int", "str", "obj"]
    )
    def test_an_unrecognised_batch_yields_nothing_rather_than_raising(self, value) -> None:
        """Absent stays absent; the strategy's own error is the one that should fire."""
        assert _as_aux_mapping(value) in ({}, value)


def test_the_strategy_that_this_unblocks_still_demands_both() -> None:
    """The guard EI raises with must stay -- this fix feeds it, not removes it.

    Deleting that check would turn a loud first-step failure into a silent
    wrong-objective run, which is the trade this fix exists to avoid.
    """
    import inspect

    from spectramr.infrastructure.training.strategies.equivariant_imaging_strategy import (
        EquivariantImagingStrategy,
    )

    source = inspect.getsource(EquivariantImagingStrategy._compute_losses_impl)
    assert "measured_kspace" in source
    assert "raise ValueError" in source


class TestTheMissingManifestMessage:
    """A manifest that does not exist yet is the commonest first-run failure.

    Manifests are gitignored and never rsynced, so "not found" almost always
    means "not generated on this host". Naming only the path costs a cluster
    round trip to work out the command; naming the command does not.
    """

    def test_it_names_the_generator(self) -> None:
        from spectramr.data.builders.manifest_loader import _missing_manifest_message

        message = _missing_manifest_message("data/manifests/m4raw_multicoil_val.json")
        assert "data/manifests/m4raw_multicoil_val.json" in message
        assert "regenerate_cluster_manifests.py" in message

    def test_a_nex_manifest_also_names_the_filter_that_builds_it(self) -> None:
        from spectramr.data.builders.manifest_loader import _missing_manifest_message

        message = _missing_manifest_message("data/manifests/m4raw_train_nex3.json")
        assert "--min-reps 3" in message

    def test_a_plain_manifest_does_not_mention_the_filter(self) -> None:
        """Otherwise the hint is noise on the common path."""
        from spectramr.data.builders.manifest_loader import _missing_manifest_message

        assert "--min-reps" not in _missing_manifest_message("data/manifests/m4raw_train.json")
