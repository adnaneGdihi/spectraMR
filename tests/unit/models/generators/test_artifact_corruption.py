"""The artifact-augmentation block is validated, and corruption keeps the acquired support."""

from __future__ import annotations

import copy

import pytest
import torch

from spectramr.models.generators.artifact_corruption import (
    ArtifactCorruptor,
    parse_artifact_augmentation,
)

BLOCK = {
    "artifacts": {
        "complex_gaussian": {"probability": 1.0, "severity": [0.3, 0.3]},
        "gibbs": {"probability": 1.0, "severity": [0.2, 0.4]},
    },
    "detector_loss_weight": 0.1,
    "eval_severity": 0.3,
}


def _masked_kspace(batch: int = 2, coils: int = 2, size: int = 16) -> torch.Tensor:
    x = torch.randn(batch, 2 * coils, size, size)
    x[..., ::2] = 0.0  # every other phase-encode line unacquired
    return x


def test_a_valid_block_parses_and_records_itself() -> None:
    spec = parse_artifact_augmentation(BLOCK)
    assert spec.names == ("complex_gaussian", "gibbs")
    assert spec.record()["artifacts"]["gibbs"]["severity"] == [0.2, 0.4]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda b: b.pop("eval_severity"), "exactly"),
        (lambda b: b.update(extra=1), "unknown"),
        (
            lambda b: b["artifacts"].update(rician={"probability": 0.5, "severity": [0, 1]}),
            "rician",
        ),
        (lambda b: b["artifacts"]["gibbs"].update(probability=0.0), r"\(0, 1\]"),
        (lambda b: b["artifacts"]["gibbs"].update(severity=[0.5, 0.1]), "low"),
        (lambda b: b.update(detector_loss_weight=-1), ">= 0"),
    ],
)
def test_a_malformed_block_raises(mutate, match) -> None:
    block = copy.deepcopy(BLOCK)
    mutate(block)
    with pytest.raises(ValueError, match=match):
        parse_artifact_augmentation(block)


def test_an_absent_block_raises() -> None:
    with pytest.raises(ValueError, match="required"):
        parse_artifact_augmentation(None)


def test_corruption_changes_acquired_lines_and_leaves_unacquired_ones_empty() -> None:
    corruptor = ArtifactCorruptor(parse_artifact_augmentation(BLOCK), seed=0)
    x = _masked_kspace()
    out, images, labels = corruptor.corrupt(x)
    assert out.shape == x.shape and images.is_complex()
    assert torch.equal(out[..., ::2], torch.zeros_like(out[..., ::2]))
    assert not torch.allclose(out[..., 1::2], x[..., 1::2])
    assert torch.equal(labels, torch.ones(2, 2))


def test_the_same_seed_gives_the_same_corruption() -> None:
    spec = parse_artifact_augmentation(BLOCK)
    x = _masked_kspace()
    a = ArtifactCorruptor(spec, seed=7).corrupt(x)[0]
    b = ArtifactCorruptor(spec, seed=7).corrupt(x)[0]
    assert torch.equal(a, b)


def test_forcing_applies_only_the_named_artifact() -> None:
    corruptor = ArtifactCorruptor(parse_artifact_augmentation(BLOCK), seed=0)
    with corruptor.forced("gibbs", 0.3, seed=1):
        assert corruptor.active
        _, _, labels = corruptor.corrupt(_masked_kspace())
    assert not corruptor.active
    assert labels[:, 0].sum() == 0 and labels[:, 1].sum() == 2
    with pytest.raises(ValueError, match="not declared"), corruptor.forced("spike", 0.3, 1):
        pass
