"""The versioned RunState every checkpoint file holds."""

from __future__ import annotations

import pytest

from spectramr.infrastructure.checkpointing.payload import (
    CONVERTER,
    SCHEMA_VERSION,
    NotARunStateError,
    SchemaVersionError,
    build_meta,
    validate,
)

TOPOLOGY = {"world_size": 1, "parallel_strategy": "none"}


def _meta(**over):
    kwargs = {
        "role": "step",
        "reason": "periodic",
        "epoch": 1,
        "step": 100,
        "fmt": "pt",
        "topology": TOPOLOGY,
        **over,
    }
    return build_meta(**kwargs)


def test_meta_carries_version_and_identity() -> None:
    meta = _meta(config_fingerprint="abc", git_sha="deadbeef")
    assert meta["schema_version"] == SCHEMA_VERSION == 1
    assert (meta["role"], meta["reason"], meta["step"], meta["format"]) == (
        "step",
        "periodic",
        100,
        "pt",
    )
    assert meta["spectramr_version"]
    assert meta["best_metric"] is None


def test_best_needs_its_metric_and_its_reason() -> None:
    meta = _meta(role="best", reason="best", best_metric={"name": "psnr", "value": 31.5})
    assert meta["best_metric"] == {"name": "psnr", "value": 31.5}
    with pytest.raises(ValueError, match="best_metric"):
        _meta(role="best", reason="best")
    with pytest.raises(ValueError, match="reason"):
        _meta(role="best", reason="periodic", best_metric={"name": "psnr", "value": 1.0})
    with pytest.raises(ValueError, match="reason"):
        _meta(role="step", reason="best")


@pytest.mark.parametrize(
    "over",
    [{"role": "final"}, {"reason": "crash"}, {"fmt": "ckpt"}, {"step": -1}, {"epoch": -1}],
)
def test_meta_refuses_unknown_values(over) -> None:
    with pytest.raises(ValueError):
        _meta(**over)


def test_validate_accepts_a_minimal_state() -> None:
    state = {"meta": _meta(), "models": {"generator": {}}}
    assert validate(state, source="x") is state


@pytest.mark.parametrize(
    "state",
    [
        {"generator": {}, "optimizer_g": {}, "epoch": 1},  # pre-1.0 director envelope
        {"meta": {"schema_version": 2}, "models": {}},
        {"meta": {}, "models": {}},
    ],
)
def test_validate_refuses_other_versions_and_names_the_converter(state) -> None:
    with pytest.raises(SchemaVersionError, match=CONVERTER):
        validate(state, source="old.pt")


@pytest.mark.parametrize(
    ("state", "source"),
    [
        ({"conv.weight": 1, "conv.bias": 2}, "pretrained/encoder.pt"),  # a bare state dict
        ({"state_dict": {}, "hyper_parameters": {}}, "lightning.ckpt.pt"),
    ],
)
def test_a_foreign_file_is_named_foreign_not_sent_to_the_converter(state, source) -> None:
    with pytest.raises(NotARunStateError, match="not a spectramr RunState") as caught:
        validate(state, source=source)
    assert CONVERTER not in str(caught.value)
    assert "external weights" in str(caught.value)


def test_a_bare_tensor_file_under_a_pre_1_0_name_names_both_remedies() -> None:
    """DeepSpeed's pre-1.0 consolidated best was a bare state dict, but other tools also
    name weights ``checkpoint_best.pth``: the name alone cannot tell them apart."""
    with pytest.raises(SchemaVersionError, match=CONVERTER) as caught:
        validate({"w": 1}, source="run/checkpoints/checkpoint_best.pt")
    assert not isinstance(caught.value, NotARunStateError)
    assert "external weights" in str(caught.value)


def test_the_pre_1_0_best_alias_is_judged_by_the_file_it_points_at(tmp_path) -> None:
    """Pre-1.0 runs published ``best.pt`` as a symlink to ``checkpoint_best.pt``, and the
    orchestrator handed children that alias, so its target's name is the signal."""
    ckpt = tmp_path / "run" / "checkpoints"
    ckpt.mkdir(parents=True)
    (ckpt / "checkpoint_best.pt").write_bytes(b"x")
    (ckpt / "best.pt").symlink_to("checkpoint_best.pt")
    with pytest.raises(SchemaVersionError, match=CONVERTER) as caught:
        validate({"w": 1}, source=str(ckpt / "best.pt"))
    assert not isinstance(caught.value, NotARunStateError)


def test_a_pre_1_0_envelope_names_only_the_converter() -> None:
    with pytest.raises(SchemaVersionError, match=CONVERTER) as caught:
        validate({"global_step": 3, "generator": {}}, source="pretrained.pt")
    assert "external weights" not in str(caught.value)


def test_validate_refuses_unknown_parts_and_missing_models() -> None:
    with pytest.raises(ValueError, match="unknown part"):
        validate({"meta": _meta(), "models": {}, "counter_state": {}}, source="x")
    with pytest.raises(ValueError, match="models"):
        validate({"meta": _meta()}, source="x")
    with pytest.raises(ValueError, match="models"):
        validate({"meta": _meta(), "models": {"generator": [1, 2]}}, source="x")


def test_validate_refuses_non_mappings() -> None:
    with pytest.raises(TypeError):
        validate([1, 2], source="x")
