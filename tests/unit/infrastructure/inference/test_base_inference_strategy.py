"""``BaseInferenceStrategy``: the predict-time data-consistency hook in ``infer``.

The mechanism lives in ``predict_data_consistency.py`` and is tested there;
this file pins what the base class does with it: resolve the knob once at
construction, project after ``run_inference`` and before ``postprocess_output``,
consume ``measured_kspace`` instead of forwarding it, skip when the strategy's
own loop already pinned the measurement, and change nothing when the knob is off.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

from spectramr.config.schemas.enums import TrainingModeTypes
from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.inference import predict_data_consistency as pdc
from spectramr.infrastructure.inference.base_inference_strategy import BaseInferenceStrategy
from spectramr.models.capabilities import ModelCapabilities
from tests.fixtures.run_state_files import write_run_state

KNOB = "physics.data_consistency.apply_at_predict"


class _Plain(BaseInferenceStrategy):
    """A reconstruction-shaped strategy: one forward, nothing else."""

    def __init__(self, model, device, config):
        super().__init__(model, device, config)
        self.seen: dict[str, dict[str, Any]] = {}
        self.postprocess_saw: torch.Tensor | None = None

    @property
    def training_mode(self) -> TrainingModeTypes:
        return TrainingModeTypes.reconstruction

    def preprocess_input(self, input_tensor, **kwargs):
        self.seen["preprocess"] = dict(kwargs)
        return input_tensor

    def run_inference(self, input_tensor, **kwargs):
        self.seen["run"] = dict(kwargs)
        return self.model(input_tensor)

    def postprocess_output(self, output_tensor, **kwargs):
        self.seen["post"] = dict(kwargs)
        self.postprocess_saw = output_tensor.clone()
        return output_tensor.clamp(-0.5, 0.5)


class _PinsItself(_Plain):
    """A sampler whose loop pins the measurement, as cold diffusion does."""

    def run_inference(self, input_tensor, **kwargs):
        out = self.model(input_tensor)
        mask = kwargs["mask"]
        out = out * (1 - mask) + input_tensor * mask
        self.note_data_consistency_applied("_PinsItself.run_inference")
        return out


class _Perturb(nn.Module):
    """A model whose output differs from its input everywhere."""

    def forward(self, x):
        return x + 1.0


def _register(monkeypatch, output_domain="kspace"):
    caps = ModelCapabilities(output_domain=output_domain)
    monkeypatch.setitem(pdc.MODEL_REGISTRY, "stub", {"capabilities": caps})
    monkeypatch.setattr(
        pdc, "get_model_capabilities", lambda name: caps if name == "stub" else None
    )


def _cfg(on: bool):
    return {
        "physics": {"data_consistency": {"apply_at_predict": on}},
        "model": {"model_type": "stub"},
    }


def _mask():
    mask = torch.zeros(1, 1, 8, 8)
    mask[..., ::2] = 1.0
    return mask


class TestTheHookFiresThroughInfer:
    def test_the_prediction_becomes_the_measurement_on_the_mask(self, monkeypatch):
        """The fires-test: through ``infer``, not by calling the layer."""
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        assert strategy.predict_dc is not None
        meas = torch.randn(1, 2, 8, 8) * 0.1
        mask = _mask()

        out = strategy.infer(meas.clone(), mask=mask, measured_kspace=meas)

        on = mask.bool().expand_as(out)
        assert torch.equal(out[on], meas[on]), "sampled bins must be the measurement"
        assert torch.equal(out[~on], (meas + 1.0).clamp(-0.5, 0.5)[~on]), (
            "unsampled bins must be the model's own prediction (then postprocessed)"
        )

    def test_the_projection_runs_before_postprocess(self, monkeypatch):
        """Postprocess clamps; the projection must see the model's raw scale."""
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        meas = torch.full((1, 2, 8, 8), 3.0)  # outside the clamp range
        mask = _mask()
        strategy.infer(meas.clone(), mask=mask, measured_kspace=meas)
        on = mask.bool().expand_as(meas)
        assert strategy.postprocess_saw is not None
        assert torch.equal(strategy.postprocess_saw[on], meas[on]), (
            "postprocess must receive the projected tensor, unclamped"
        )

    def test_measured_kspace_is_consumed_not_forwarded(self, monkeypatch):
        """Cold diffusion hands its kwargs to the model; this one must never reach it."""
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        meas = torch.randn(1, 2, 8, 8)
        strategy.infer(meas.clone(), mask=_mask(), measured_kspace=meas)
        for stage in ("preprocess", "run", "post"):
            assert "measured_kspace" not in strategy.seen[stage], stage
            assert "mask" in strategy.seen[stage], "mask is in the existing kwarg vocabulary"

    def test_no_mask_raises_naming_the_knob(self, monkeypatch):
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        meas = torch.randn(1, 2, 8, 8)
        with pytest.raises(ConfigurationError, match=KNOB.replace(".", r"\.")):
            strategy.infer(meas.clone(), measured_kspace=meas)

    def test_provenance_counts_the_projection(self, monkeypatch):
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        meas = torch.randn(1, 2, 8, 8)
        strategy.infer(meas.clone(), mask=_mask(), measured_kspace=meas)
        strategy.infer(meas.clone(), mask=_mask(), measured_kspace=meas)
        prov = strategy.predict_dc_provenance()
        assert prov["calls"] == 2
        assert prov["projections_by_predict_step"] == 2
        assert prov["applied_by"] == {"predict_step": 2}


class TestNotTwice:
    def test_a_strategy_that_pinned_the_measurement_is_not_projected_again(self, monkeypatch):
        _register(monkeypatch)
        strategy = _PinsItself(_Perturb(), torch.device("cpu"), _cfg(True))
        projections: list[int] = []
        real = strategy.predict_dc.project
        monkeypatch.setattr(
            strategy.predict_dc, "project", lambda *a, **k: projections.append(1) or real(*a, **k)
        )
        meas = torch.randn(1, 2, 8, 8) * 0.1
        mask = _mask()
        out = strategy.infer(meas.clone(), mask=mask, measured_kspace=meas)
        assert projections == [], "the base hook must not project a second time"
        on = mask.bool().expand_as(out)
        assert torch.equal(out[on], meas[on]), "the strategy's own pin still holds"
        prov = strategy.predict_dc_provenance()
        assert prov["applied_by"] == {"_PinsItself.run_inference": 1}
        assert prov["projections_by_predict_step"] == 0
        assert prov["skipped_already_applied"] == 1

    def test_the_ledger_is_per_call(self, monkeypatch):
        """A pin in call one must not silence the projection in call two."""
        _register(monkeypatch)
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(True))
        meas = torch.randn(1, 2, 8, 8)
        strategy.predict_dc.note_applied("stale")  # before any call
        strategy.infer(meas.clone(), mask=_mask(), measured_kspace=meas)
        assert strategy.predict_dc_provenance()["projections_by_predict_step"] == 1


class TestOffState:
    def test_off_means_no_projection_state_at_all(self):
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(False))
        assert strategy.predict_dc is None
        assert strategy.predict_dc_provenance() == {"apply_at_predict": False}

    def test_off_output_is_byte_identical_with_or_without_the_kwargs(self):
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(False))
        x = torch.randn(1, 2, 8, 8)
        plain = strategy.infer(x.clone())
        with_kwargs = strategy.infer(x.clone(), mask=_mask(), measured_kspace=x)
        assert torch.equal(plain, with_kwargs)
        assert "measured_kspace" not in strategy.seen["run"]

    def test_off_note_is_a_no_op(self):
        strategy = _Plain(_Perturb(), torch.device("cpu"), _cfg(False))
        strategy.note_data_consistency_applied("anything")  # must not raise
        assert strategy.predict_dc_provenance() == {"apply_at_predict": False}

    def test_no_config_at_all_is_off(self):
        strategy = _Plain(_Perturb(), torch.device("cpu"), None)
        assert strategy.predict_dc is None


class TestConstructionRefusals:
    def test_an_unannotated_model_refuses_at_construction(self, monkeypatch):
        """Run-invariant, so it fails before the first file, not inside the loop."""
        monkeypatch.setitem(pdc.MODEL_REGISTRY, "stub", {"capabilities": None})
        monkeypatch.setattr(pdc, "get_model_capabilities", lambda name: None)
        with pytest.raises(ConfigurationError, match="declares no output_domain"):
            _Plain(_Perturb(), torch.device("cpu"), _cfg(True))

    def test_a_non_boolean_knob_refuses_at_construction(self):
        cfg = {
            "physics": {"data_consistency": {"apply_at_predict": "yes"}},
            "model": {"model_type": "stub"},
        }
        with pytest.raises(ConfigurationError, match="must be a boolean"):
            _Plain(_Perturb(), torch.device("cpu"), cfg)


def test_the_pipeline_owns_kspace_normalisation_by_default() -> None:
    """Only a strategy that normalises inside ``infer`` opts out (#2581)."""
    from spectramr.infrastructure.inference.base_inference_strategy import BaseInferenceStrategy
    from spectramr.infrastructure.inference.reconstruction_inference_strategy import (
        ReconstructionInferenceStrategy,
    )

    assert BaseInferenceStrategy.owns_kspace_normalization is False
    assert ReconstructionInferenceStrategy.owns_kspace_normalization is False


class TestEmaPartIsPreferred:
    """Inference loads the run's ``ema`` part when ``use_ema`` (default on) and the
    file kept one, else its ``generator``; both strict, into the bare model."""

    @staticmethod
    def _write(tmp_path, *, ema=None, generator=None):
        return write_run_state(
            tmp_path / "best.pt",
            {"generator": generator or {"weight": torch.zeros(2, 2)}},
            ema=ema,
        )

    def test_loads_the_ema_weights_by_default(self, tmp_path) -> None:
        path = self._write(tmp_path, ema={"weight": torch.ones(2, 2)})
        strategy = _Plain(nn.Linear(2, 2, bias=False), torch.device("cpu"), {})
        strategy.load_checkpoint(str(path))
        assert torch.equal(strategy.model.weight, torch.ones(2, 2))

    def test_use_ema_false_loads_the_generator(self, tmp_path) -> None:
        path = self._write(tmp_path, ema={"weight": torch.ones(2, 2)})
        strategy = _Plain(nn.Linear(2, 2, bias=False), torch.device("cpu"), {"use_ema": False})
        strategy.load_checkpoint(str(path))
        assert torch.equal(strategy.model.weight, torch.zeros(2, 2))

    def test_without_an_ema_part_the_generator_loads(self, tmp_path) -> None:
        path = self._write(tmp_path, generator={"weight": torch.full((2, 2), 5.0)})
        model = nn.Linear(2, 2, bias=False)
        _Plain(model, torch.device("cpu"), {}).load_checkpoint(str(path))
        assert torch.equal(model.weight, torch.full((2, 2), 5.0))

    def test_a_partitioned_shadow_raises_instead_of_loading_garbage(self, tmp_path) -> None:
        """ZeRO-3 builds the shadow from local partitions; the strict load must refuse it."""
        path = self._write(tmp_path, ema={"weight": torch.ones(1, 2)})
        strategy = _Plain(nn.Linear(2, 2, bias=False), torch.device("cpu"), {})
        with pytest.raises(RuntimeError):
            strategy.load_checkpoint(str(path))

    def test_the_gan_loader_makes_the_same_choice(self, tmp_path) -> None:
        from spectramr.infrastructure.inference.gan_inference_strategy import (
            GANInferenceStrategy,
        )

        path = self._write(tmp_path, ema={"weight": torch.ones(2, 2)})
        model = nn.Linear(2, 2, bias=False)
        GANInferenceStrategy(model, torch.device("cpu"), {}).load_checkpoint(str(path))
        assert torch.equal(model.weight, torch.ones(2, 2))

        model = nn.Linear(2, 2, bias=False)
        GANInferenceStrategy(model, torch.device("cpu"), {"use_ema": False}).load_checkpoint(
            str(path)
        )
        assert torch.equal(model.weight, torch.zeros(2, 2))

    def test_the_gan_loader_restores_a_configured_discriminator(self, tmp_path, monkeypatch) -> None:
        from types import SimpleNamespace

        from spectramr.infrastructure.inference.gan_inference_strategy import (
            GANInferenceStrategy,
        )
        from spectramr.models.factories import model_factory

        fake = SimpleNamespace(create_discriminator=lambda *_a, **_k: nn.Linear(2, 1, bias=False))
        monkeypatch.setattr(model_factory, "get_model_factory", lambda: fake)
        path = write_run_state(
            tmp_path / "best.pt",
            {"generator": {"weight": torch.zeros(2, 2)}, "discriminator": {"weight": torch.ones(1, 2)}},
        )
        strategy = GANInferenceStrategy(
            nn.Linear(2, 2, bias=False),
            torch.device("cpu"),
            {"gan": {"discriminator": {"type": "patch_gan"}}},
        )
        strategy.load_checkpoint(str(path))
        assert torch.equal(strategy.discriminator.weight, torch.ones(1, 2))
        assert not strategy.discriminator.training
