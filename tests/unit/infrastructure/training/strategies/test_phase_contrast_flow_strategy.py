"""Tests for PhaseContrastFlowStrategy — the trainable backing for mri_flow.

No 4D-flow data exists on the cluster, but the physics is analytic, so the
strategy's actual scientific claim IS testable here: encode a known velocity
field, hand the phases to an optimiser, and recover the field. That is a proof of
the objective, not a smoke check.
"""

from __future__ import annotations

import types

import pytest
import torch

from spectramr.infrastructure.training.strategies.phase_contrast_flow_strategy import (
    PhaseContrastFlowStrategy,
    decode_velocity_four_point,
    encode_velocity_four_point,
)

_VENC = 1.0


def _velocity(batch: int = 2, size: int = 8) -> torch.Tensor:
    """A [B, 3, H, W] velocity field strictly inside +/- venc (no wrap)."""
    torch.manual_seed(0)
    return (torch.rand(batch, 3, size, size) - 0.5) * 1.2 * _VENC  # |v| <= 0.6*venc


class _VelocityNet(torch.nn.Module):
    """[B, 4, H, W] phases -> [B, 3, H, W] velocity. No terminal tanh, on purpose."""

    def __init__(self) -> None:
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Conv2d(4, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 3, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


def _strategy(**overrides: object) -> PhaseContrastFlowStrategy:
    """Hand-built strategy — no DI container, no real TrainingEnvironment."""
    strat = object.__new__(PhaseContrastFlowStrategy)
    strat.env = types.SimpleNamespace(generator=_VelocityNet(), losses={})
    strat.config = types.SimpleNamespace(
        losses=types.SimpleNamespace(
            image_losses=[], kspace_losses=[], complex_losses=[]
        )
    )
    strat._venc = _VENC
    strat._voxel_area = 1.0
    strat._through_plane_index = 2
    strat._flux_conservation = False
    strat._validate_encode_roundtrip = False
    strat._roundtrip_checked = True
    strat._lambda_velocity = 1.0
    strat._lambda_flux = 0.0
    strat._lambda_unwrap = 0.0
    for key, value in overrides.items():
        setattr(strat, key, value)
    return strat


def _batch(**extra: object) -> dict:
    """A canonical batch: the velocity field rides the `target` key.

    BatchAdapter.from_dict REQUIRES ('input', 'target'), so a flow batch that
    spelled the field only 'velocity' could not traverse the real pipeline.
    """
    velocity = _velocity()
    batch = {
        "input": encode_velocity_four_point(velocity, _VENC),
        "target": velocity,
    }
    batch.update(extra)
    return batch


# ---------------------------------------------------------------------------
# Registration / mounting.
# ---------------------------------------------------------------------------


def test_strategy_registered_and_config_mounted() -> None:
    from spectramr.config.schemas.training.base import TrainingStrategyConfigSchema
    from spectramr.infrastructure.training.strategy_factory import (
        TrainingStrategyFactory,
    )

    assert "phase_contrast_flow" in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
    assert "phase_contrast" in TrainingStrategyConfigSchema.model_fields


def test_data_block_is_mounted_not_silently_dropped() -> None:
    """DataConfigSchema is extra="ignore" — an unmounted block vanishes quietly.

    Without the mounted field a `data.phase_contrast:` YAML block is dropped with
    no error at all, and the strategy would read defaults and train the wrong
    physics. The mount is the load-bearing step, so it gets its own test.
    """
    from spectramr.config.schemas.data import DataConfigSchema

    assert "phase_contrast" in DataConfigSchema.model_fields
    data = DataConfigSchema(phase_contrast={"enabled": True, "venc": 2.5})
    assert data.phase_contrast.venc == 2.5


def test_strategy_is_flow_tagged_for_the_ledger() -> None:
    from spectramr.config.schemas.enums import Regime, Task

    caps = PhaseContrastFlowStrategy.__dict__["capabilities"]
    assert caps.workflows == frozenset({Regime.FLOW})
    assert Task.PARAMETER_MAPPING in caps.tasks


# ---------------------------------------------------------------------------
# The physics adapters.
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrips_within_venc() -> None:
    """The channel-first permute must not scramble the component axis."""
    velocity = _velocity()
    phases = encode_velocity_four_point(velocity, _VENC)
    assert phases.shape == (2, 4, 8, 8)
    torch.testing.assert_close(
        decode_velocity_four_point(phases, _VENC), velocity, rtol=1e-4, atol=1e-5
    )


# ---------------------------------------------------------------------------
# The objective.
# ---------------------------------------------------------------------------


def test_loss_keys_and_finite() -> None:
    out = _strategy()._compute_losses_impl(input_batch=_batch())
    assert "g_total_loss" in out
    assert "loss_phase_contrast_velocity" in out
    assert torch.isfinite(out["g_total_loss"])


def test_recovers_a_known_velocity_field() -> None:
    """The OBJECTIVE recovers a known velocity field, on analytic physics.

    Encode a known field, optimise the velocity maps against the phase-contrast
    loss, and check the field comes back. Needs no real data — which matters,
    because there is no 4D-flow data on this cluster.

    SCOPE: this optimises a free tensor against the registered loss; it does NOT
    call the strategy, so it verifies the loss + encoding, not the wiring. The
    wiring is covered by test_loss_reduces (the strategy's own objective descends)
    and test_losses_are_called_by_keyword_not_position.

    The lr is decayed because the loss is L1 in the phase domain (``p=1``): a
    constant gradient magnitude means a fixed lr oscillates around the optimum at
    scale ~lr instead of settling. The assertion is on VELOCITY recovery in m/s,
    not on the loss value — the loss lives in phase units and its magnitude alone
    says nothing about whether the physical field was recovered.
    """
    torch.manual_seed(0)
    from spectramr.models.losses.flow_losses import PhaseContrastVelocityLoss

    target = _velocity()
    estimate = torch.zeros_like(target, requires_grad=True)
    opt = torch.optim.Adam([estimate], lr=5e-2)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.985)
    loss_fn = PhaseContrastVelocityLoss(venc=_VENC)

    first = None
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(v_pred=estimate, v_target=target, venc=_VENC)
        loss.backward()
        opt.step()
        sched.step()
        if first is None:
            first = float(loss.detach())

    assert first is not None and float(loss.detach()) < first / 10.0
    torch.testing.assert_close(estimate.detach(), target, rtol=0.05, atol=0.02)


def test_loss_reduces() -> None:
    torch.manual_seed(0)
    strat = _strategy()
    opt = torch.optim.Adam(strat.env.generator.parameters(), lr=2e-3)
    batch = _batch()
    first = None
    out = None
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        out = strat._compute_losses_impl(input_batch=batch)
        out["g_total_loss"].backward()
        opt.step()
        if first is None:
            first = float(out["g_total_loss"].detach())
    assert out is not None and first is not None
    assert float(out["g_total_loss"].detach()) < first


def test_builder_image_losses_folded_via_seam() -> None:
    """Without the fold, losses.image_losses is an inert decoy (pitfall #16)."""
    from spectramr.models.losses.charbonnier_loss import CharbonnierLoss
    from spectramr.models.losses.hfen_loss import HFENLoss

    strat = _strategy()
    strat.env.losses = {"l1": CharbonnierLoss(), "hfen": HFENLoss()}
    strat.config.losses = types.SimpleNamespace(
        image_losses=[{"name": "l1", "weight": 1.0}, {"name": "hfen", "weight": 0.2}],
        kspace_losses=[],
        complex_losses=[],
    )

    out = strat._compute_losses_impl(input_batch=_batch())
    assert "loss_builder_aux" in out and "loss_hfen" in out
    assert torch.isfinite(out["g_total_loss"])


# ---------------------------------------------------------------------------
# The anti-facade gates.
# ---------------------------------------------------------------------------


def test_losses_are_called_by_keyword_not_position() -> None:
    """The registered flow losses are positional and NOT (pred, target)-shaped.

    ThroughPlaneFluxConservationLoss.forward is
    (v_through_plane, inlet_mask, outlet_mask, voxel_area). Bound positionally
    via the generic ComposedLoss path it would silently take the wrong tensors —
    and voxel_area, which has no __init__ to carry it, would never arrive at all.
    """
    from spectramr.models.losses import flow_losses

    seen: dict[str, object] = {}
    original = flow_losses.ThroughPlaneFluxConservationLoss.forward

    def _record(self, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return original(self, *args, **kwargs)

    flow_losses.ThroughPlaneFluxConservationLoss.forward = _record
    try:
        strat = _strategy(_lambda_flux=0.5, _flux_conservation=True, _voxel_area=4.0)
        strat._compute_losses_impl(
            input_batch=_batch(
                inlet_mask=torch.ones(2, 1, 8, 8), outlet_mask=torch.ones(2, 1, 8, 8)
            )
        )
    finally:
        flow_losses.ThroughPlaneFluxConservationLoss.forward = original

    assert seen["args"] == (), "flux loss was called positionally"
    assert seen["kwargs"]["voxel_area"] == 4.0
    assert set(seen["kwargs"]) == {
        "v_through_plane",
        "inlet_mask",
        "outlet_mask",
        "voxel_area",
    }


def test_raises_when_flux_is_weighted_but_masks_are_absent() -> None:
    """A weighted loss must never be silently skipped."""
    strat = _strategy(_lambda_flux=0.5, _flux_conservation=True)
    with pytest.raises(ValueError, match="inlet_mask"):
        strat._compute_losses_impl(input_batch=_batch())


def test_velocity_alias_takes_precedence_over_target() -> None:
    """Both spellings work; the explicit one wins."""
    batch = _batch()
    explicit = _velocity() * 0.5
    batch["velocity"] = explicit
    strat = _strategy()
    resolved = strat._resolve_velocity_target(batch)
    torch.testing.assert_close(resolved, explicit)


def test_rejects_a_one_channel_image_target() -> None:
    """The shape contract is what actually prevents pitfall #18.

    Point a flow arm at a magnitude-image dataset and `target` is [B, 1, H, W].
    Reading it without a shape check would broadcast a scalar image against a
    3-component velocity field and produce a plausible, meaningless number.
    """
    batch = _batch()
    batch["target"] = torch.rand(2, 1, 8, 8)
    with pytest.raises(ValueError, match="3-component field"):
        _strategy()._compute_losses_impl(input_batch=batch)


def test_raises_without_any_reference_velocity() -> None:
    batch = {"input": encode_velocity_four_point(_velocity(), _VENC)}
    with pytest.raises(ValueError, match="reference velocity"):
        _strategy()._compute_losses_impl(input_batch=batch)


def test_roundtrip_guard_catches_a_venc_mismatch() -> None:
    """A venc mismatch rescales every velocity and nothing else would notice."""
    velocity = _velocity()
    batch = {
        "input": encode_velocity_four_point(velocity, _VENC * 2.0),  # wrong venc
        "velocity": velocity,
    }
    strat = _strategy(_validate_encode_roundtrip=True, _roundtrip_checked=False)
    with pytest.raises(ValueError, match="four-point encoded at this venc"):
        strat._compute_losses_impl(input_batch=batch)


def test_compute_losses_rejects_a_bare_tensor_batch() -> None:
    with pytest.raises(ValueError, match="mapping batch"):
        _strategy()._compute_losses_impl(input_batch=torch.rand(2, 4, 8, 8))


def test_compute_losses_accepts_a_canonical_trainingbatch() -> None:
    """REGRESSION (cohort-wide): the real pipeline passes a TrainingBatch.

    It converts the loader dict via BatchAdapter.from_dict and calls
    _compute_losses_impl with input_batch=<tensor>, target_batch=<tensor> and
    batch=<TrainingBatch> — so the batch arrives in **kwargs, not as
    input_batch. A guard written `isinstance(batch, dict)` passes every dict-fed
    unit test and then crashes every real arm at step 0; that exact bug once took
    out the whole MICCAI cohort. The guard must accept any mapping exposing .get.

    This also pins that BatchAdapter can carry a flow batch at all: it REQUIRES
    canonical ('input', 'target'), which is why the velocity field rides `target`.
    """
    from spectramr.data.batch_types import BatchAdapter

    tb = BatchAdapter.from_dict(_batch())
    out = _strategy()._compute_losses_impl(
        input_batch=tb.input, target_batch=tb.target, epoch=0, batch=tb
    )
    assert torch.isfinite(out["g_total_loss"])
    assert "loss_phase_contrast_velocity" in out


def test_flux_loss_does_not_broadcast_across_the_batch() -> None:
    """REGRESSION: slicing `[:, idx]` instead of `[:, idx:idx+1]` cross-contaminates.

    `velocity_pred[:, idx]` drops the channel axis to [B, H, W], which then
    broadcasts against the documented [B, 1, H, W] masks into [B, B, H, W] —
    every sample's velocity multiplied by every OTHER sample's mask. The flux
    reduces over (-2, -1) only, so the result is a [B, B] matrix of cross-sample
    fluxes whose off-diagonal entries are physically meaningless. It never
    raises: the loss stays finite and plausible.

    Asserted by recording the tensor the loss receives. A value-only assertion
    would not catch it — the contaminated loss is a perfectly ordinary number.
    """
    from spectramr.models.losses import flow_losses

    seen: dict[str, torch.Tensor] = {}
    original = flow_losses.ThroughPlaneFluxConservationLoss.forward

    def _record(self, **kwargs):
        seen["v"] = kwargs["v_through_plane"]
        seen["inlet"] = kwargs["inlet_mask"]
        return original(self, **kwargs)

    flow_losses.ThroughPlaneFluxConservationLoss.forward = _record
    try:
        strat = _strategy(_lambda_flux=0.5, _flux_conservation=True)
        strat._compute_losses_impl(
            input_batch=_batch(
                inlet_mask=torch.ones(2, 1, 8, 8), outlet_mask=torch.ones(2, 1, 8, 8)
            )
        )
    finally:
        flow_losses.ThroughPlaneFluxConservationLoss.forward = original

    v, inlet = seen["v"], seen["inlet"]
    assert v.ndim == 4 and v.shape[1] == 1, (
        f"the through-plane slice must keep its channel axis, got {tuple(v.shape)}"
    )
    assert (v * inlet).shape == v.shape, (
        f"velocity {tuple(v.shape)} x mask {tuple(inlet.shape)} broadcasts to "
        f"{tuple((v * inlet).shape)} — samples are being cross-multiplied"
    )


# ---------------------------------------------------------------------------
# Units guard, rank-generic adapters, setup reads encoding_scheme.
# ---------------------------------------------------------------------------


def test_adapters_work_on_a_volume_with_frames() -> None:
    """The adapters encode along the channel axis at any rank, not only [B, C, H, W]."""
    from spectramr.infrastructure.training.strategies.phase_contrast_flow_strategy import (
        decode_velocity_four_point,
    )

    v = (torch.rand(2, 3, 4, 5, 6, 7) - 0.5) * 1.8 * _VENC
    phases = encode_velocity_four_point(v, _VENC)
    assert phases.shape == (2, 4, 4, 5, 6, 7)
    assert torch.allclose(decode_velocity_four_point(phases, _VENC), v, atol=1e-5)


@pytest.mark.parametrize(
    ("units", "activation", "tail", "fragment"),
    [
        ("Hz", None, None, "output_field_units='Hz'"),
        (None, "tanh", None, "output_activation='tanh'"),
        (None, None, torch.nn.Sigmoid(), "last module is Sigmoid"),
    ],
)
def test_bounded_output_is_named(units, activation, tail, fragment) -> None:
    from spectramr.infrastructure.training.strategies.phase_contrast_flow_strategy import (
        bounded_velocity_output,
    )

    net = _VelocityNet() if tail is None else torch.nn.Sequential(_VelocityNet(), tail)
    reason = bounded_velocity_output(units, activation, net)
    assert reason is not None and fragment in reason


def test_unbounded_undeclared_generator_passes() -> None:
    from spectramr.infrastructure.training.strategies.phase_contrast_flow_strategy import (
        bounded_velocity_output,
    )

    assert bounded_velocity_output(None, None, _VelocityNet()) is None
    assert bounded_velocity_output("m/s", None, _VelocityNet()) is None


def _setup_strategy(generator: torch.nn.Module, **model: object) -> PhaseContrastFlowStrategy:
    from spectramr.config.schemas.data import PhaseContrastConfigSchema

    strat = object.__new__(PhaseContrastFlowStrategy)
    strat.env = types.SimpleNamespace(generator=generator, losses={})
    strat.logging_service = None
    strat._verify_strategy_config = lambda **_: None
    strat._get_loss_weight = lambda name: 1.0 if name == "phase_contrast_velocity" else 0.0
    model_cfg = {"model_type": "not_a_registered_model", "in_channels": 4, "out_channels": 3}
    model_cfg.update(model)
    strat.config = types.SimpleNamespace(
        data=types.SimpleNamespace(phase_contrast=PhaseContrastConfigSchema(enabled=True, venc=1.5)),
        model=types.SimpleNamespace(output_activation=None, **model_cfg),
        training=types.SimpleNamespace(phase_contrast=None),
    )
    strat._setup_strategy_specific_components()
    return strat


def test_setup_reads_the_encoding_scheme() -> None:
    strat = _setup_strategy(_VelocityNet())
    assert strat._encoding_scheme == "four_point_reference"


def test_setup_rejects_a_wrong_input_channel_count() -> None:
    with pytest.raises(ValueError, match="4 encoded phases"):
        _setup_strategy(_VelocityNet(), in_channels=2)


def test_setup_rejects_a_terminal_tanh() -> None:
    with pytest.raises(ValueError, match="needs velocity in m/s"):
        _setup_strategy(torch.nn.Sequential(_VelocityNet(), torch.nn.Tanh()))
