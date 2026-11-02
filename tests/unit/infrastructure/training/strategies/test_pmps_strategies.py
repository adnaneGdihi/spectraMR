"""Reachability of the PMPS Stage-C orchestrator: the arm's resolve path runs the loop.

The factory resolves the real ``stage_c_synthesis`` arm to ``PairedSynthesisStrategy``; the
strategy's setup hook then runs on a tiny CPU config with the registry-built generator, a real
Stage-B checkpoint of the ULF operator and a stub prior in place of the Stage-A UNet (no
model-building locally). Observed: operators injected, pairs written, a nonzero residual.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from spectramr.infrastructure.checkpointing import codecs
from spectramr.infrastructure.checkpointing.payload import build_meta
from spectramr.infrastructure.training.strategies import pmps_strategies
from spectramr.infrastructure.training.strategies.pmps_strategies import PairedSynthesisStrategy
from spectramr.models.registry import get_model_class

from .test_pmps_synthesis import ConstantX0Prior, stage_c_extras

pytestmark = [pytest.mark.unit]

ARM = Path("experiments/inprogress/pmps/stage_c_synthesis.yaml")


def _write_ulf_checkpoint(path: Path) -> torch.nn.Module:
    torch.manual_seed(1)
    ulf = get_model_class("ulf_corruption_operator")(in_channels=1, out_channels=1)
    with torch.no_grad():
        ulf.log_noise_gain.fill_(0.25)
    state = {
        "meta": build_meta(
            role="step",
            reason="periodic",
            epoch=0,
            step=1,
            fmt="pt",
            topology={"world_size": 1, "parallel_strategy": "none"},
        ),
        "models": {"generator": ulf.state_dict()},
    }
    codecs.write(path, state)
    return ulf


def _strategy(tmp_path: Path, **overrides: object) -> PairedSynthesisStrategy:
    config = SimpleNamespace(
        training=SimpleNamespace(model_extra=stage_c_extras(tmp_path, **overrides), batch_size=2),
        model=SimpleNamespace(model_kwargs={}),
        losses=SimpleNamespace(
            image_losses=[SimpleNamespace(name="per_pair_consistency_residual", weight=1.0)]
        ),
        data=SimpleNamespace(sampling=SimpleNamespace(patch_size=[8, 8, 1])),
    )
    strategy = object.__new__(PairedSynthesisStrategy)
    strategy.config = config
    strategy.device = torch.device("cpu")
    strategy.env = SimpleNamespace(generator=get_model_class("paired_synthesis_generator")())
    return strategy


def test_arm_resolves_to_paired_synthesis_strategy() -> None:
    from spectramr.config.settings import TrainingSettings
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    settings = TrainingSettings.from_yaml(str(ARM))
    assert TrainingStrategyFactory().get_strategy_class(settings) is PairedSynthesisStrategy


def test_setup_runs_the_loop_with_loaded_operators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saved = _write_ulf_checkpoint(tmp_path / "ulf.pt")
    monkeypatch.setattr(
        pmps_strategies, "build_frozen_prior", lambda cfg, device: (ConstantX0Prior(), "stub")
    )
    strategy = _strategy(tmp_path)
    strategy._setup_strategy_specific_components()

    gen = strategy.env.generator
    assert torch.equal(gen.ulf_corruption.log_noise_gain, saved.log_noise_gain)
    assert gen.hf_corruption is not None
    assert not any(p.requires_grad for p in gen.ulf_corruption.parameters())

    manifest = json.loads((tmp_path / "pairs" / "manifest.json").read_text())
    assert manifest["n_accepted"] == 3
    assert {r["shard"] for r in manifest["pairs"]} == {0, 1}
    assert all(
        r["residual_ulf"] > 0 and r["per_pair_consistency_residual"] > 0 for r in manifest["pairs"]
    )
    assert manifest["provenance"]["ulf_corruption_source"].endswith("ulf.pt")
    written = [f for r in manifest["pairs"] for f in r["files"]]
    assert written and all((tmp_path / "pairs" / f).is_file() for f in written)
    assert strategy._compute_losses_impl()["loss_total"].item() == 0.0

    # Validation reports the corpus, never calling the generator on the 1-channel loader batch.
    batch = torch.zeros(2, 1, 8, 8)
    metrics = strategy.validation_step(batch, batch)
    accepted = [r["per_pair_consistency_residual"] for r in manifest["pairs"] if r["accepted"]]
    assert metrics["n_accepted"] == 3.0
    assert metrics["per_pair_consistency_residual"] == pytest.approx(sum(accepted) / 3)
    assert strategy._last_visual_pred.shape == strategy._last_visual_target.shape == (1, 1, 8, 8)
    first = strategy._last_visual_pred
    strategy.validation_step(batch, batch)
    assert strategy._last_visual_pred is not first


def test_rejected_pairs_are_dropped_without_write_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ulf_checkpoint(tmp_path / "ulf.pt")
    monkeypatch.setattr(
        pmps_strategies,
        "build_frozen_prior",
        lambda cfg, device: (ConstantX0Prior(striped=True), "stub"),
    )
    # Large HF motion on a striped map, against a tiny HF noise, pushes every pair over the gate.
    strategy = _strategy(
        tmp_path,
        n_synthetic_pairs=6,
        write_rejected=False,
        defensibility_threshold=1.0,
        corruption={
            "checkpoint": str(tmp_path / "ulf.pt"),
            "part": "generator",
            "hf": {"rician_noise_sigma": 0.001, "motion_severity": 1.0},
        },
    )
    torch.manual_seed(0)
    strategy._setup_strategy_specific_components()
    pairs = json.loads((tmp_path / "pairs" / "manifest.json").read_text())["pairs"]
    rejected = [r for r in pairs if not r["accepted"]]
    assert rejected and all(r["files"] == [] for r in rejected)
    assert all(r["residual_hf"] >= 1.0 for r in rejected)
    assert not (tmp_path / "pairs" / "rejected").exists()


def test_too_few_accepted_pairs_raise_after_the_render_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_ulf_checkpoint(tmp_path / "ulf.pt")
    monkeypatch.setattr(
        pmps_strategies,
        "build_frozen_prior",
        lambda cfg, device: (ConstantX0Prior(striped=True), "stub"),
    )
    monkeypatch.setattr(
        "spectramr.infrastructure.training.strategies.pmps_synthesis.MAX_RENDERS_PER_PAIR", 1
    )
    strategy = _strategy(
        tmp_path,
        n_synthetic_pairs=6,
        defensibility_threshold=1.0,
        corruption={
            "checkpoint": str(tmp_path / "ulf.pt"),
            "part": "generator",
            "hf": {"rician_noise_sigma": 0.001, "motion_severity": 1.0},
        },
    )
    torch.manual_seed(0)
    with pytest.raises(ValueError, match="short of n_synthetic_pairs=6"):
        strategy._setup_strategy_specific_components()
    assert (tmp_path / "pairs" / "manifest.json").is_file()


def test_disagreeing_threshold_raises(tmp_path: Path) -> None:
    strategy = _strategy(tmp_path)
    strategy.config.model.model_kwargs = {"defensibility_threshold": 2.0}
    with pytest.raises(ValueError, match="declare it once"):
        strategy._setup_strategy_specific_components()
