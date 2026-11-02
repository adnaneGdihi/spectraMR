"""Complex k-space diffusion: resolves through the factory, trains epsilon, samples under hard DC.

Bare-instance pattern: the class comes from ``TrainingStrategyFactory`` (the production
resolve path) and only the seams the methods read are set, so no model or DI build is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.training.schedulers.diffusion_scheduler import DiffusionScheduler
from spectramr.infrastructure.training.strategies.complex_kspace_diffusion_strategy import (
    ComplexKspaceDiffusionStrategy,
    flat_rows,
    measurement_scale,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

T = 20


def _resolved_class() -> type:
    cfg = SimpleNamespace(training=SimpleNamespace(strategy_class="complex_kspace_diffusion"))
    return TrainingStrategyFactory().get_strategy_class(cfg)


def _config(
    *,
    method: str = "hard",
    enabled: bool = True,
    out: str = "kspace",
    prediction: str = "epsilon",
    conditional: bool = False,
    chunk: int = 2,
) -> SimpleNamespace:
    diffusion = SimpleNamespace(prediction_type=prediction, condition_on_input=conditional)
    return SimpleNamespace(
        training=SimpleNamespace(diffusion=diffusion, output_domain=out),
        physics=SimpleNamespace(data_consistency=SimpleNamespace(enabled=enabled, method=method)),
        validation=SimpleNamespace(
            sampling=SimpleNamespace(steps=5), loader=SimpleNamespace(chunk_size=chunk)
        ),
    )


class _EpsGen(torch.nn.Module):
    """Predicts zero noise and records what it saw and the horizon it was given."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[torch.Tensor, int]] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor, **kwargs: object) -> torch.Tensor:
        self.calls.append((x, kwargs.get("max_timesteps")))
        return torch.zeros_like(x)


def _bare(config: SimpleNamespace | None = None) -> ComplexKspaceDiffusionStrategy:
    cls = _resolved_class()
    s = object.__new__(cls)
    s.config = config or _config()
    s.device = torch.device("cpu")
    s.num_timesteps = T
    s.beta_schedule = "linear"
    s.scheduler = DiffusionScheduler(num_timesteps=T, device=torch.device("cpu"))
    s._declared_model_input = None
    s._resolve_legacy_batch = lambda input_batch, kwargs: input_batch  # type: ignore[method-assign]
    s._resolve_validation_sampling_steps = lambda: 5  # type: ignore[method-assign]
    s._resolved_kspace_diffusion = s._resolve_settings()
    gen = _EpsGen()
    s.env = SimpleNamespace(generator=gen)
    return s


def _kspace(b: int = 2, c: int = 2, n: int = 8) -> torch.Tensor:
    return torch.randn(b, c, n, n, 1)


def test_factory_resolves_the_short_name() -> None:
    assert _resolved_class() is ComplexKspaceDiffusionStrategy


def test_training_feeds_the_noised_kspace_and_regresses_epsilon() -> None:
    s = _bare()
    k0, y = _kspace(), _kspace()
    out = s._compute_losses_impl(input_batch={"target": k0, "input": y})

    seen, horizon = s.env.generator.calls[-1]
    assert horizon == T, "the horizon must reach complex_unet's timestep embedding"
    assert seen.shape == (2, 2, 8, 8)
    assert not torch.equal(seen, flat_rows(k0) / measurement_scale(flat_rows(y)))
    tensors, extra, keys = s._declared_model_input
    assert torch.equal(tensors["model_input"], seen)
    assert extra["provenance"]["dc_method"] == "hard"
    assert keys == {"model_input", "target"}
    assert torch.isfinite(out["loss_total"]) and "loss_eps_kspace" in out


def test_epsilon_round_trip_recovers_x0() -> None:
    s = _bare()
    x0 = torch.randn(3, 2, 8, 8)
    eps = torch.randn_like(x0)
    t = torch.tensor([0, 7, T - 1])
    x_t = s.scheduler.q_sample(x0, t, eps)
    torch.testing.assert_close(s.scheduler.predict_x0_from_eps(x_t, t, eps), x0)


def test_sampler_pins_every_measured_bin_to_the_measurement() -> None:
    s = _bare()
    y_full = torch.randn(2, 2, 8, 8)
    mask = torch.zeros(2, 1, 8, 8)
    mask[..., ::3] = 1.0
    y = y_full * mask
    out = s._generate_validation_prediction(
        y, y_full, torch.zeros(2), {"mask": mask.unsqueeze(-1)}, {}
    )
    m = mask.expand_as(out).bool()
    torch.testing.assert_close(out[m], y[m])
    assert torch.equal(s._zf_measurement, y), "the zero-filled baseline is the measurement"
    assert s._rung_mask is not None and s._ensemble_members is None
    assert torch.isfinite(out).all()
    assert len(s.env.generator.calls) == 5, "validation.sampling.steps bounds the chain"


def test_sampler_chunks_every_network_call_by_the_validation_chunk_size() -> None:
    s = _bare(_config(chunk=2))
    y_full = torch.randn(5, 2, 8, 8)
    mask = torch.zeros(5, 1, 8, 8)
    mask[..., ::2] = 1.0
    out = s.sample_with_dc(s.env.generator, y_full * mask, mask)
    sizes = [x.shape[0] for x, _ in s.env.generator.calls]
    assert sizes == [2, 2, 1] * 5, "5 slices at chunk_size 2, over 5 reverse steps"
    assert out.shape == y_full.shape


def test_infer_refuses_the_arm_with_a_named_reason() -> None:
    from spectramr.config.settings import TrainingSettings
    from spectramr.pipelines.infer import _refuse_unservable_strategy

    config = TrainingSettings.from_yaml(
        "experiments/inprogress/diffusion/experiment_47_complex_valued_latent_diffusion.yaml"
    )
    with pytest.raises(ConfigurationError, match="infer cannot serve ComplexKspace"):
        _refuse_unservable_strategy(config)


def test_missing_mask_raises() -> None:
    s = _bare()
    y = torch.randn(1, 2, 8, 8)
    with pytest.raises(ValueError, match="sampling mask"):
        s._generate_validation_prediction(y, y, torch.zeros(1), {}, {})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"method": "soft"}, "data_consistency"),
        ({"enabled": False}, "data_consistency"),
        ({"out": "image"}, "output_domain"),
        ({"prediction": "sample"}, "epsilon"),
        ({"conditional": True}, "condition_on_input"),
    ],
)
def test_unsupported_settings_raise(kwargs: dict, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        _bare(_config(**kwargs))


def test_it_declares_its_own_loss_ownership() -> None:
    d = ComplexKspaceDiffusionStrategy.__dict__
    assert d["folds_image_losses"] is False
    assert d["empty_loss_stack_is_design"] is True
    assert d["inline_losses"] == frozenset()
