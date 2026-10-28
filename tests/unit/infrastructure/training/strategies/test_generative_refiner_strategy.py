"""Tests for GenerativeRefinerStrategy — the shared Track-A refiner chassis.

White-box on the pure pieces (the full training/validation pipeline is exercised by the
Tier-2 mechanism-fires probe): the forward-operator dispatch, the config guards, the
non-saturating eps-matching objective, and the load-bearing property that the DPS
guidance term actually FIRES (guidance-on diverges from guidance-off, i.e. the mechanism
is not inert — pitfall #16) and that guidance_weight=0 recovers the plain prior (the
one-knob ablation).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from spectramr.config.schemas.training.generative_refiner import GenerativeRefinerConfig
from spectramr.infrastructure.training.strategies.field_guided_diffusion_strategy import (
    make_alphas_cumprod,
)
from spectramr.infrastructure.training.strategies.generative_refiner_strategy import (
    _FORWARD_OPERATORS,
    _wiener_band_op,
    _wiener_recoverable_noise,
    compute_ulf_dps_loss,
    refiner_sample,
    resolve_forward_operator,
)

pytestmark = pytest.mark.unit


class _StubEps(nn.Module):
    """A minimal trainable eps-net: eps = w * x (time/field accepted, unused)."""

    def __init__(self, w: float = 0.1) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor(float(w)))

    def forward(
        self,
        x: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        field_strength: torch.Tensor,
        contrast_id: torch.Tensor | None = None,
    ):
        return self.w * x


def _acp(timesteps: int = 20) -> torch.Tensor:
    return make_alphas_cumprod(timesteps, "cosine", device="cpu", dtype=torch.float32)


# ── forward-operator dispatch (no if/elif; unknown raises, pitfall #9) ────────


def test_resolve_forward_operator_known() -> None:
    op = resolve_forward_operator("field_lowpass")
    assert callable(op)
    assert "field_lowpass" in _FORWARD_OPERATORS


def test_resolve_forward_operator_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown generative-refiner forward_operator"):
        resolve_forward_operator("does_not_exist")


def test_wiener_band_registered() -> None:
    assert "wiener_band" in _FORWARD_OPERATORS
    assert callable(resolve_forward_operator("wiener_band"))


# ── wiener_band operator: the b27 mechanism as guidance ──────────────────────


def test_wiener_noise_rises_as_field_drops() -> None:
    """N(b) is negative-slope in log10(b): lower field => larger noise ratio."""
    assert _wiener_recoverable_noise(0.1) > _wiener_recoverable_noise(7.0)


def test_wiener_band_is_dc_preserving() -> None:
    """The gain is DC-normalized, so a constant image passes through unchanged."""
    x = torch.ones(1, 1, 16, 16)
    out = _wiener_band_op(x, {"source_field": 0.1})
    assert torch.allclose(out, x, atol=1e-5)


def test_wiener_band_field_monotone_energy() -> None:
    """Higher source field => wider recoverable band => more retained energy.

    The b27 mechanism's defining property: the recoverable band shrinks as the field
    drops, so a lower-field operator attenuates more high-frequency content.
    """
    torch.manual_seed(5)
    x = torch.rand(1, 1, 32, 32)
    e_low = float(_wiener_band_op(x, {"source_field": 0.1}).pow(2).sum())
    e_high = float(_wiener_band_op(x, {"source_field": 7.0}).pow(2).sum())
    assert e_high >= e_low, f"7T band should retain >= 0.1T energy; got {e_high} < {e_low}"


def test_wiener_band_differentiable() -> None:
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    _wiener_band_op(x, {"source_field": 0.1}).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_wiener_band_distinct_from_field_lowpass() -> None:
    """The two operators are genuinely different responses (not a relabeled low-pass)."""
    torch.manual_seed(9)
    x = torch.rand(1, 1, 24, 24)
    a = _wiener_band_op(x, {"source_field": 0.1})
    b = resolve_forward_operator("field_lowpass")(x, {"blur_cutoff": 0.5})
    assert float((a - b).abs().max()) > 1e-3


def test_config_accepts_wiener_band_and_source_field() -> None:
    cfg = GenerativeRefinerConfig(forward_operator="wiener_band", source_field=0.1)
    assert cfg.forward_operator == "wiener_band"
    assert cfg.source_field == 0.1
    with pytest.raises(ValueError):
        GenerativeRefinerConfig(source_field=9.0)  # > 7 T


# ── config guards ────────────────────────────────────────────────────────────


def test_config_rejects_sampling_steps_gt_timesteps() -> None:
    with pytest.raises(ValueError, match="sampling_steps"):
        GenerativeRefinerConfig(timesteps=50, sampling_steps=100)


def test_config_allows_zero_guidance_ablation() -> None:
    cfg = GenerativeRefinerConfig(guidance_weight=0.0)
    assert cfg.guidance_weight == 0.0


def test_config_rejects_unknown_forward_operator() -> None:
    # The closed Literal rejects an unimplemented mechanism at load time.
    with pytest.raises(ValueError):
        GenerativeRefinerConfig(forward_operator="tv_prior")  # not yet implemented


# ── non-saturating eps-matching objective ────────────────────────────────────


def test_compute_refiner_loss_is_finite_and_trainable() -> None:
    torch.manual_seed(0)
    model = _StubEps()
    batch = {
        "target": torch.rand(2, 1, 16, 16),
        "field_strength_target": torch.tensor([3.0, 7.0]),
    }
    out = compute_ulf_dps_loss(model, batch, alphas_cumprod=_acp())
    loss = out["loss_total"]
    assert loss.requires_grad, "eps-matching loss must carry a gradient to the prior."
    assert math.isfinite(float(loss.detach()))


# ── sampling: shape, range, and the mechanism-fires property ──────────────────


def test_refiner_sample_shape_and_range() -> None:
    torch.manual_seed(1)
    model = _StubEps()
    y = torch.rand(2, 1, 16, 16)
    b_t = torch.tensor([3.0, 7.0])
    out = refiner_sample(
        model, y, b_t,
        alphas_cumprod=_acp(), forward_op=resolve_forward_operator("field_lowpass"),
        op_kwargs={"blur_cutoff": 0.5}, n_steps=5, guidance_weight=1.0,
    )
    assert out.shape == y.shape
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0


def test_guidance_fires_changes_output() -> None:
    """The DPS guidance term must be non-inert: with the SAME initial noise, turning
    guidance on (weight>0) must change the sampled output vs. off (weight=0). If they
    were identical, the mechanism would be a facade (pitfall #16)."""
    model = _StubEps()
    y = torch.rand(2, 1, 16, 16)
    b_t = torch.tensor([3.0, 7.0])
    op = resolve_forward_operator("field_lowpass")
    acp = _acp()

    torch.manual_seed(123)
    guided = refiner_sample(
        model, y, b_t, alphas_cumprod=acp, forward_op=op,
        op_kwargs={"blur_cutoff": 0.5}, n_steps=6, guidance_weight=1.0,
    )
    torch.manual_seed(123)  # identical initial x ~ randn_like(y)
    unguided = refiner_sample(
        model, y, b_t, alphas_cumprod=acp, forward_op=op,
        op_kwargs={"blur_cutoff": 0.5}, n_steps=6, guidance_weight=0.0,
    )
    diff = float((guided - unguided).abs().max())
    assert diff > 1e-5, f"guidance did not change the output (diff={diff}); mechanism inert."


def test_zero_guidance_is_deterministic_prior() -> None:
    """guidance_weight=0 (the ablation) is the plain prior: same seed => same output."""
    model = _StubEps()
    y = torch.rand(1, 1, 16, 16)
    b_t = torch.tensor([5.0])
    op = resolve_forward_operator("field_lowpass")
    acp = _acp()

    outs = []
    for _ in range(2):
        torch.manual_seed(7)
        outs.append(
            refiner_sample(
                model, y, b_t, alphas_cumprod=acp, forward_op=op,
                op_kwargs={"blur_cutoff": 0.5}, n_steps=5, guidance_weight=0.0,
            )
        )
    assert torch.allclose(outs[0], outs[1])


# ── registration ─────────────────────────────────────────────────────────────


def test_strategy_registered_in_factory() -> None:
    from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

    paths = TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "generative_refiner" in paths
    assert paths["generative_refiner"].endswith(
        "generative_refiner_strategy.GenerativeRefinerStrategy"
    )


# ── Multi-contrast: contrast_id threaded to the score net (train + sampler) ───


def test_contrast_id_threaded_train_and_sampling() -> None:
    # b27_wiener enables use_contrast_conditioning on the widened FieldGuidedScoreUNet.
    # Training reuses compute_ulf_dps_loss (shared with b22), so threading that helper
    # fixes both; the refiner's own sampler must forward it too or the widened net
    # raises at step 0 (the cluster failure).
    seen: dict = {}

    def spy(x, *, timesteps, field_strength, contrast_id=None, **_):
        seen["cid"] = contrast_id
        return torch.zeros_like(x)

    cid = torch.tensor([0, 2])
    batch = {
        "target": torch.rand(2, 1, 16, 16),
        "field_strength_target": torch.tensor([3.0, 7.0]),
        "contrast_id": cid,
    }
    compute_ulf_dps_loss(spy, batch, alphas_cumprod=_acp())
    assert seen["cid"] is cid  # shared eps-matching threads it
    seen.clear()
    refiner_sample(
        spy, torch.rand(2, 1, 16, 16), torch.tensor([3.0, 7.0]),
        alphas_cumprod=_acp(), forward_op=resolve_forward_operator("field_lowpass"),
        op_kwargs={"blur_cutoff": 0.5}, n_steps=1, guidance_weight=0.0, contrast_id=cid,
    )
    assert seen["cid"] is cid  # mechanism-guided sampler threads it


def test_field_lowpass_threads_contrast_id_to_the_shared_operator() -> None:
    from spectramr.infrastructure.training.strategies.ulf_lowpass_operator import (
        UlfLowpassOperator,
    )

    op = UlfLowpassOperator(axis_cutoffs={0: (0.15, 0.15), 1: (2.0, 2.0)})
    y = torch.rand(1, 1, 16, 16)
    kw = {
        "alphas_cumprod": _acp(),
        "forward_op": resolve_forward_operator("field_lowpass"),
        "op_kwargs": {"forward_op": op},
        "n_steps": 5,
        "guidance_weight": 5.0,
    }
    torch.manual_seed(0)
    a = refiner_sample(_StubEps(), y, torch.tensor([3.0]), contrast_id=torch.tensor([0]), **kw)
    torch.manual_seed(0)
    b = refiner_sample(_StubEps(), y, torch.tensor([3.0]), contrast_id=torch.tensor([1]), **kw)
    assert not torch.allclose(a, b)


def _refiner_setup(gr_cfg: GenerativeRefinerConfig, ulf_lowpass):
    import types

    from spectramr.infrastructure.training.strategies.generative_refiner_strategy import (
        GenerativeRefinerStrategy,
    )

    strat = object.__new__(GenerativeRefinerStrategy)
    strat._verify_strategy_config = lambda **_: None
    strat.config = types.SimpleNamespace(
        training=types.SimpleNamespace(generative_refiner=gr_cfg, ulf_lowpass=ulf_lowpass),
        data=None,
    )
    strat._setup_strategy_specific_components()
    return strat


def test_field_lowpass_setup_requires_ulf_lowpass_and_wiener_does_not() -> None:
    from spectramr.config.schemas.training.ulf_lowpass import UlfLowpassConfig

    with pytest.raises(ValueError, match=r"training\.ulf_lowpass"):
        _refiner_setup(GenerativeRefinerConfig(forward_operator="field_lowpass"), None)
    wiener = _refiner_setup(GenerativeRefinerConfig(forward_operator="wiener_band"), None)
    assert "forward_op" not in wiener._gr_op_kwargs
    lowpass = _refiner_setup(
        GenerativeRefinerConfig(forward_operator="field_lowpass"),
        UlfLowpassConfig(cutoff_mode="constant", blur_cutoff=0.4),
    )
    assert lowpass._gr_op_kwargs["forward_op"].constant == 0.4
