"""weights(), read_part() and best_weights(): reading parts of a checkpoint back."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from spectramr.infrastructure.checkpointing import codecs, store
from spectramr.infrastructure.checkpointing.layout import CheckpointLayout
from spectramr.infrastructure.checkpointing.restore import (
    CheckpointPartMissingError,
    ZeroOverlapError,
    best_weights,
    read_part,
    weights,
)
from tests.unit.infrastructure.checkpointing.conftest import (
    FakeEarlyStopping,
    FakeNativeAdapter,
    make_pipeline,
    native_runtime,
)


@pytest.fixture(params=["pt", "safetensors"])
def written(request, tmp_path, run_state):
    state = run_state(request.param, role="best")
    (tmp_path / "checkpoints").mkdir()
    path = tmp_path / "checkpoints" / f"best.{request.param}"
    codecs.write(path, state)
    return path, state


def _fresh():
    return nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2))


def test_exact_loads_the_generator(written) -> None:
    path, state = written
    model = _fresh()
    report = weights(path, "generator", model, policy="exact")
    assert report.matched == report.total == 4
    for k, v in state["models"]["generator"].items():
        assert torch.equal(model.state_dict()[k], v)


def test_a_run_reference_selects_best(written) -> None:
    path, _ = written
    report = weights({"run_dir": str(path.parent.parent)}, "generator", _fresh(), policy="exact")
    assert report.source == path


def test_exact_refuses_a_different_architecture(written) -> None:
    path, _ = written
    grown = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2), nn.Linear(2, 2))
    with pytest.raises(RuntimeError):
        weights(path, "generator", grown, policy="exact")


def test_overlap_transfers_the_shared_trunk(written) -> None:
    path, state = written
    grown = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2), nn.Linear(2, 2))
    report = weights(path, "generator", grown, policy="overlap")
    assert (report.matched, report.total) == (4, 6)
    assert report.missing == ("3.bias", "3.weight")
    assert torch.equal(grown.state_dict()["0.weight"], state["models"]["generator"]["0.weight"])


def test_overlap_with_nothing_in_common_raises(written) -> None:
    path, _ = written
    target = nn.Conv2d(1, 1, 3)
    with pytest.raises(ZeroOverlapError, match="zero parameter names") as info:
        weights(path, "generator", target, policy="overlap")
    # A ValueError, so callers that predate the subclass keep catching it.
    assert isinstance(info.value, ValueError)
    assert info.value.expected == len(target.state_dict())
    assert info.value.unexpected == len(read_part(path, "generator"))


def test_ema_and_stage_parts(written) -> None:
    path, state = written
    model = _fresh()
    weights(path, "ema", model, policy="exact")
    for k, v in state["ema"]["generator"].items():
        assert torch.equal(model.state_dict()[k], v)
    from tests.unit.infrastructure.checkpointing.conftest import Tied

    weights(path, "stage:tied", Tied(), policy="exact")


def test_missing_part_names_what_the_file_holds(written) -> None:
    path, _ = written
    with pytest.raises(CheckpointPartMissingError, match="generator"):
        weights(path, "discriminator", _fresh(), policy="exact")


def test_wrapper_prefixes_are_stripped(tmp_path, run_state) -> None:
    state = run_state("pt")
    state["models"]["generator"] = {
        f"_orig_mod.{k}": v for k, v in state["models"]["generator"].items()
    }
    path = tmp_path / "step_00000150.pt"
    codecs.write(path, state)
    assert weights(path, "generator", _fresh(), policy="exact").matched == 4


@pytest.mark.parametrize("policy", ["strict", "", None])
def test_unknown_policy_raises_before_reading(tmp_path, policy) -> None:
    with pytest.raises(ValueError, match="policy"):
        weights(tmp_path / "absent.pt", "generator", _fresh(), policy=policy)


def _best_file(tmp_path: Path, pipeline=None, parallel=None) -> Path:
    cfg = SimpleNamespace(checkpoint=SimpleNamespace(format="pt", keep_last_n=5))
    return store.save_best(
        pipeline or make_pipeline(),
        layout=CheckpointLayout.for_run(tmp_path),
        config=cfg,
        strategy=None,
        parallel=parallel,
        early_stopping=FakeEarlyStopping(),
        epoch=1,
        step=4,
        metric_name="val_psnr",
        metric_value=21.5,
    )


def test_read_part_returns_the_bare_state_dict(tmp_path: Path) -> None:
    assert set(read_part(_best_file(tmp_path), "generator")) == {"weight", "bias"}


def test_read_part_strips_wrapper_prefixes(tmp_path: Path) -> None:
    path = _best_file(tmp_path)
    state = codecs.read(path)
    state["models"]["generator"] = {
        f"module._orig_mod.{k}": v for k, v in state["models"]["generator"].items()
    }
    codecs.write(path, state)
    assert set(read_part(path, "generator")) == {"weight", "bias"}


def test_best_weights_restores_models_and_the_ema_shadow_with_its_counter(tmp_path: Path) -> None:
    saved = make_pipeline()
    path = _best_file(tmp_path, saved)
    live = make_pipeline()
    nn.init.zeros_(live.generator.weight)
    live.ema.num_updates = 0
    best_weights(path, live, parallel=None, strategy=None)
    assert torch.equal(live.generator.weight, saved.generator.weight)
    assert live.ema.num_updates == 1


def test_best_weights_leaves_optimizer_scheduler_scaler_and_rng_alone(tmp_path: Path) -> None:
    saved = make_pipeline(scaler=True)
    saved.scaler.scale(torch.ones(1))  # the scale is created lazily by the first scale()
    saved.scaler.update(1024.0)
    path = _best_file(tmp_path, saved)
    live = make_pipeline(scaler=True)
    live_scale = live.scaler.get_scale()
    live.optimizer_g.state.clear()
    live.scheduler_g.last_epoch = 1
    torch.manual_seed(9)
    expected = torch.rand(2)
    torch.manual_seed(9)
    best_weights(path, live, parallel=None, strategy=None)
    assert live.optimizer_g.state_dict()["state"] == {}
    assert live.scheduler_g.last_epoch == 1
    assert live.scaler.get_scale() == live_scale != 1024.0
    assert torch.equal(torch.rand(2), expected)


def test_best_weights_refuses_ema_on_without_a_best_shadow(tmp_path: Path) -> None:
    path = _best_file(tmp_path, make_pipeline(ema=False))
    with pytest.raises(RuntimeError, match="EMA is enabled"):
        best_weights(path, make_pipeline(), parallel=None, strategy=None)


def test_a_refused_best_restore_leaves_every_model_untouched(tmp_path: Path) -> None:
    """A refusal after a partial load would leave best weights beside the last shadow,
    while the loop reports the LAST weights; every check runs before any module moves."""
    path = _best_file(tmp_path, make_pipeline(ema=False))
    live = make_pipeline()
    nn.init.zeros_(live.generator.weight)
    before = {k: v.clone() for k, v in live.generator.state_dict().items()}
    with pytest.raises(RuntimeError, match="EMA is enabled"):
        best_weights(path, live, parallel=None, strategy=None)
    after = live.generator.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_a_refused_best_restore_never_enters_the_native_load(tmp_path: Path) -> None:
    native = FakeNativeAdapter(frozenset({"generator"}))
    path = _best_file(tmp_path, make_pipeline(ema=False))
    state = codecs.read(path)
    state["meta"]["native_tag"] = "best"
    codecs.write(path, state)
    with pytest.raises(RuntimeError, match="EMA is enabled"):
        best_weights(path, make_pipeline(), parallel=native_runtime(native), strategy=None)
    assert not getattr(native, "seen", None)


def test_best_weights_restores_engine_models_through_the_native_best_tag(tmp_path: Path) -> None:
    native = FakeNativeAdapter(frozenset({"generator"}))
    path = _best_file(tmp_path)
    state = codecs.read(path)
    state["meta"]["native_tag"] = "best"
    codecs.write(path, state)
    live = make_pipeline()
    nn.init.zeros_(live.generator.weight)
    best_weights(path, live, parallel=native_runtime(native), strategy=None)
    assert (native.seen["tag"], native.seen["module_only"]) == ("best", True)
    assert torch.count_nonzero(live.generator.weight) == 0  # the engine loaded it, not the payload
