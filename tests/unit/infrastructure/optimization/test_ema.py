"""Tests for ModelEma warmup-decay (Experiment-11 EMA-lag regression).

The DC blob persisted in *validation* because validation swaps in the EMA
shadow weights, and the shadow — a deepcopy of the random init updated with a
fixed ``decay=0.9999`` and NO warmup — was still ~74% random init at the
~3000-iter early-stop (``0.9999**3000 = 0.74``). These tests pin the
num_updates-aware decay ramp that makes the shadow track the live model early.
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from spectramr.infrastructure.optimization.ema import ModelEma


def _const_model(val: float) -> nn.Module:
    m = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        m.weight.fill_(val)
    return m


def test_num_updates_starts_zero_and_increments():
    ema = ModelEma(_const_model(0.0), decay=0.9999)
    assert ema.num_updates == 0
    ema.update(_const_model(1.0))
    assert ema.num_updates == 1


def test_warmup_decay_tracks_model_early():
    """With the warmup ramp, ONE update moves the shadow most of the way to the
    live model (effective_decay = min(0.9999, 2/11) = 0.1818), instead of the
    ~0.9999 that would leave it ~stuck at the random init."""
    ema = ModelEma(_const_model(0.0), decay=0.9999)  # shadow starts at 0.0
    ema.update(_const_model(1.0))  # live weights = 1.0
    w = ema.module.weight
    # ema = decay*0 + (1-decay)*1 = 1 - 2/11
    assert torch.allclose(w, torch.full_like(w, 1.0 - 2.0 / 11.0), atol=1e-5)
    # After 50 updates the (constant) live model is clearly tracked — NOT the
    # ~26% tracking a fixed 0.9999 decay would give over the same span.
    for _ in range(49):
        ema.update(_const_model(1.0))
    assert ema.module.weight.mean().item() > 0.95


def test_effective_decay_caps_at_configured_decay():
    """Once num_updates is large the ramp saturates at the configured decay."""
    ema = ModelEma(_const_model(0.0), decay=0.99)
    ema.num_updates = 10_000
    eff = min(ema.decay, (1.0 + ema.num_updates) / (10.0 + ema.num_updates))
    assert eff == 0.99  # (1+1e4)/(10+1e4)=0.9999 > 0.99 -> cap wins


def test_fixed_decay_would_stay_near_init_without_warmup():
    """Documents WHY the warmup matters: a plain 0.9999 blend over 50 steps on
    a constant model leaves the shadow far from it; the warmup ramp (tested
    above) does not."""
    shadow = 0.0
    for _ in range(50):
        shadow = 0.9999 * shadow + (1.0 - 0.9999) * 1.0
    assert shadow < 0.01  # plain fixed decay: still ~stuck at init after 50


def test_warmup_default_true():
    assert ModelEma(_const_model(0.0), decay=0.9999).warmup is True


def test_warmup_false_uses_fixed_decay():
    """warmup=False -> the EMA-lag BASELINE arm: a fixed decay every step (no
    ramp), so one update on a 0->1 model gives exactly (1-decay), not the
    warmup ramp's larger first step."""
    ema = ModelEma(_const_model(0.0), decay=0.9, warmup=False)
    ema.update(_const_model(1.0))
    w = ema.module.weight
    # fixed decay 0.9: ema = 0.9*0 + 0.1*1 = 0.1 (NOT warmup's 1 - 2/11 = 0.818)
    assert torch.allclose(w, torch.full_like(w, 0.1), atol=1e-6)


def test_num_updates_persisted_in_state_dict():
    """The warmup counter must survive a checkpoint round-trip; otherwise a
    SLURM requeue restarts the ramp at num_updates=0 and the first post-resume
    update uses effective_decay~=0.1, stomping ~90% of the restored shadow."""
    ema = ModelEma(_const_model(0.0), decay=0.9999)
    for _ in range(37):
        ema.update(_const_model(1.0))
    assert ema.num_updates == 37
    sd = ema.state_dict()

    restored = ModelEma(_const_model(0.0), decay=0.9999)
    restored.load_state_dict(sd)
    assert restored.num_updates == 37
    # And the shadow weights themselves round-tripped.
    assert torch.allclose(restored.module.weight, ema.module.weight)


def test_next_update_after_restore_uses_ramped_decay_not_reset():
    """After restoring num_updates, the next update continues the ramp (decay
    close to the configured value) rather than the ~0.1 first-step value."""
    ema = ModelEma(_const_model(1.0), decay=0.99)
    ema.num_updates = 1000  # deep into the ramp; effective_decay -> ~0.99
    before = ema.module.weight.clone()
    ema.update(_const_model(0.0))  # live weights differ -> pull toward 0
    # With decay ~0.99 the shadow barely moves (stays near 1.0); a reset to
    # num_updates=0 would apply decay ~0.1 and crash it toward 0.
    assert ema.module.weight.mean().item() > 0.95
    assert not torch.allclose(ema.module.weight, before)  # it did update


def test_update_same_device_still_blends_correctly():
    """The added device-reconciliation must be a no-op when shadow and model
    share a device (regression guard for the CPU path)."""
    ema = ModelEma(_const_model(0.0), decay=0.9, warmup=False)
    ema.update(_const_model(1.0))
    assert torch.allclose(ema.module.weight, torch.full((4, 4), 0.1), atol=1e-6)


@pytest.mark.gpu
def test_update_reconciles_cpu_ema_with_gpu_model():
    """A memory-saving CPU-EMA / GPU-model config must not raise a device
    mismatch: the live values are moved onto the shadow's device before blending.
    Skipped without CUDA."""
    model = _const_model(1.0).cuda()  # live weights on GPU
    ema = ModelEma(_const_model(0.0), decay=0.9, warmup=False)  # shadow on CPU
    # Force the shadow module onto CPU explicitly.
    ema.module.cpu()
    ema.update(model)  # must not raise despite the device mismatch
    assert ema.module.weight.device.type == "cpu"
    assert torch.allclose(
        ema.module.weight, torch.full((4, 4), 0.1), atol=1e-6
    )


def test_load_legacy_checkpoint_without_counter_defaults_to_zero():
    """Backward-compat: a pre-fix checkpoint has no counter key; strict load
    must still succeed and leave num_updates at its default 0."""
    legacy = ModelEma(_const_model(0.5), decay=0.9999)
    sd = legacy.state_dict()
    sd.pop("_ema_num_updates", None)  # simulate an old checkpoint

    fresh = ModelEma(_const_model(0.0), decay=0.9999)
    fresh.load_state_dict(sd)  # must not raise under strict=True
    assert fresh.num_updates == 0
    assert torch.allclose(fresh.module.weight, legacy.module.weight)


# ---------------------------------------------------------------------------
# Adaptive EMA (#1294).
#
# ``config.ema.enable_adaptive_ema`` and its three deterministic companions
# (``warmup_steps`` / ``initial_decay`` / ``final_decay``) were schema-only for
# months: the implementation lived at ``src/core/models/utils/adaptive_ema.py``
# until ff0efff9f deleted it, orphaning the caller in ``training/state.py``.
# These tests pin the restored semantics — a LINEAR decay ramp
# ``initial_decay -> final_decay`` over ``warmup_steps`` updates, exactly the
# historical behaviour (adaptive_ema.py lines 69-73).
# ---------------------------------------------------------------------------


def _adaptive_ema(model: nn.Module, **kw) -> ModelEma:
    defaults = dict(
        decay=0.5,  # deliberately NOT the ramp endpoint: adaptive must supersede it
        adaptive=True,
        warmup_steps=100,
        initial_decay=0.0,
        final_decay=0.9,
    )
    defaults.update(kw)
    return ModelEma(model, **defaults)


@pytest.mark.parametrize(
    ("num_updates", "expected"),
    [
        (1, 0.009),  # first update sits just above initial_decay
        (50, 0.45),  # half-way through the ramp
        (99, 0.891),  # last ramped step
        (100, 0.9),  # ramp complete -> final_decay
        (10_000, 0.9),  # and it HOLDS at final_decay afterwards
    ],
)
def test_adaptive_ramp_pins_exact_decay(num_updates, expected):
    """decay = initial + (n / warmup_steps) * (final - initial), then final."""
    ema = _adaptive_ema(_const_model(0.0))
    ema.num_updates = num_updates
    assert ema._current_decay() == pytest.approx(expected)


def test_adaptive_supersedes_the_timm_warmup_ramp():
    """warmup=True must NOT win over adaptive=True.

    The timm ramp would give min(0.5, (1+1)/(10+1)) = 0.1818 at n=1; the
    adaptive ramp gives 0.009. Silently applying the wrong one would make the
    declared initial_decay unreadable from the observed behaviour.
    """
    ema = _adaptive_ema(_const_model(0.0), warmup=True)
    ema.num_updates = 1
    assert ema._current_decay() == pytest.approx(0.009)
    assert ema._current_decay() != pytest.approx(2.0 / 11.0)


def test_adaptive_supersedes_fixed_decay():
    """Past the ramp the effective decay is final_decay, never ``decay``."""
    ema = _adaptive_ema(_const_model(0.0), warmup=False, decay=0.5)
    ema.num_updates = 500
    assert ema._current_decay() == pytest.approx(0.9)


def test_non_adaptive_path_is_unchanged():
    """Regression guard: the standard path keeps the timm ramp exactly."""
    ema = ModelEma(_const_model(0.0), decay=0.9999, warmup=True)
    ema.num_updates = 1
    assert ema._current_decay() == pytest.approx(2.0 / 11.0)
    ema.warmup = False
    assert ema._current_decay() == pytest.approx(0.9999)


def test_adaptive_first_update_blends_with_ramped_decay():
    """End-to-end: shadow 0.0, live 1.0, decay 0.009 -> shadow = 0.991."""
    ema = _adaptive_ema(_const_model(0.0))
    ema.update(_const_model(1.0))
    assert ema.num_updates == 1
    assert torch.allclose(
        ema.module.weight, torch.full_like(ema.module.weight, 0.991), atol=1e-6
    )


def test_adaptive_ramp_position_survives_checkpoint_roundtrip():
    """A SLURM requeue must resume the ramp, not restart it at initial_decay."""
    ema = _adaptive_ema(_const_model(0.0))
    for _ in range(60):
        ema.update(_const_model(1.0))
    assert ema.num_updates == 60

    restored = _adaptive_ema(_const_model(0.0))
    restored.load_state_dict(ema.state_dict())
    assert restored.num_updates == 60
    # The ramp resumes at n=60 (0.6 of the way), not back at initial_decay.
    assert restored._current_decay() == pytest.approx(0.54)


def test_adaptive_without_warmup_steps_raises():
    """A zero-length ramp is a declaration that does nothing — refuse it
    rather than silently degrading to a fixed final_decay (non-negotiable 3)."""
    with pytest.raises(ValueError, match="warmup_steps"):
        ModelEma(_const_model(0.0), adaptive=True, warmup_steps=0)


def test_adaptive_with_inverted_endpoints_raises():
    with pytest.raises(ValueError, match="final_decay"):
        ModelEma(
            _const_model(0.0),
            adaptive=True,
            warmup_steps=10,
            initial_decay=0.99,
            final_decay=0.5,
        )


# ---------------------------------------------------------------------------
# Fused ``_foreach_lerp_`` blend
#
# ``update`` batches the floating-point blend into ``torch._foreach_lerp_``
# instead of running ``mul_().add_()`` per tensor: a Scalene profile of
# experiment_11_attention_none charged the update ~18 s over 300 steps, which
# is launch overhead proportional to the parameter COUNT rather than real work.
# ``lerp_(a, b, w) == (1 - w) * a + w * b``, so ``w = 1 - decay`` reproduces the
# old blend -- in one fused op instead of two, hence tolerance not bitwise.
# ---------------------------------------------------------------------------


class _MixedNet(nn.Module):
    """Float params + a non-float buffer (``num_batches_tracked``)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(self.conv(x)).mean((2, 3)))


def _reference_blend(shadow: nn.Module, live: nn.Module, decay: float) -> None:
    """The per-tensor blend this implementation replaced."""
    live_sd = live.state_dict()
    with torch.no_grad():
        for key, value in shadow.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(live_sd[key], alpha=1.0 - decay)
            else:
                value.copy_(live_sd[key])


def test_foreach_blend_matches_per_tensor_reference():
    """Every shadow tensor must land where the two-op blend would have put it."""
    torch.manual_seed(0)
    live = _MixedNet()
    ema = ModelEma(live, decay=0.99, warmup=False)
    reference = copy.deepcopy(ema.module)

    for _ in range(5):
        with torch.no_grad():
            for param in live.parameters():
                param.add_(torch.randn_like(param) * 0.1)
        live(torch.randn(2, 2, 8, 8))  # advances BN buffers
        _reference_blend(reference, live, 0.99)
        ema.update(live)

    shadow_sd = ema.module.state_dict()
    for key, expected in reference.state_dict().items():
        assert torch.allclose(shadow_sd[key].float(), expected.float(), atol=1e-6), key


def test_every_float_tensor_actually_moves():
    """Guards against a bucket that silently drops tensors it never blended."""
    torch.manual_seed(0)
    live = _MixedNet()
    ema = ModelEma(live, decay=0.5, warmup=False)
    before = {k: v.clone() for k, v in ema.module.state_dict().items()}
    with torch.no_grad():
        for param in live.parameters():
            param.add_(1.0)
    ema.update(live)
    after = ema.module.state_dict()
    moved = {
        k for k, v in before.items() if v.dtype.is_floating_point and not torch.equal(v, after[k])
    }
    # Only the PARAMETERS were perturbed above; BN's running buffers are
    # untouched in the live model, so shadow and live already agree there and a
    # correct blend leaves them equal. Asserting over the full float state_dict
    # would fail on exactly those, for the wrong reason.
    perturbed = {name for name, _ in live.named_parameters()}
    assert perturbed <= moved, f"unblended parameters: {perturbed - moved}"


def test_non_float_buffer_is_copied_not_blended():
    """``num_batches_tracked`` is a count; a 0.99 blend of it is meaningless."""
    torch.manual_seed(0)
    live = _MixedNet()
    ema = ModelEma(live, decay=0.99, warmup=False)
    for _ in range(3):
        live(torch.randn(2, 2, 8, 8))
    ema.update(live)
    assert int(ema.module.state_dict()["bn.num_batches_tracked"]) == int(
        live.state_dict()["bn.num_batches_tracked"]
    )


def test_mixed_dtype_pair_falls_back_and_still_blends():
    """An fp32 shadow against an fp16 live model cannot share a foreach bucket.

    ``_foreach_*`` requires a uniform bucket, so this pair takes the per-tensor
    path; it must still blend rather than be skipped.
    """
    torch.manual_seed(0)
    live = _MixedNet()
    ema = ModelEma(live, decay=0.5, warmup=False)
    before = ema.module.conv.weight.detach().clone()
    live.half()
    ema.update(live)
    assert ema.module.conv.weight.dtype is torch.float32
    assert not torch.equal(before, ema.module.conv.weight)


# ---------------------------------------------------------------------------
# Wrapper invariance (#2172).
#
# The blend loop is key-matched, so a shadow whose keys are all absent from the
# live model blends NOTHING and raises nothing. Measured against a real
# DeepSpeed engine: shadow "0.weight" vs engine "module.0.weight", overlap 0,
# on 75 production arms. These tests plant that exact shape.
#
# Note what the pre-existing tests above could not catch: they call
# ``ema.state_dict()`` directly, while the checkpoint director called
# ``unwrap_model(ema).state_dict()`` -- which peels ModelEma's own ``.module``
# and so never runs the override. The tests and the production path were
# exercising different code (non-negotiable 15).
# ---------------------------------------------------------------------------


class _PrefixWrapper(nn.Module):
    """Stands in for DDP / DeepSpeedEngine: registers the model under ``.module``.

    That is the whole mechanism -- every state-dict key gains a ``module.``
    prefix -- so a fake reproduces it exactly and needs no GPU or process group.
    """

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module


def test_update_raises_when_the_live_model_is_wrapped():
    """Zero key overlap must raise, not silently blend nothing."""
    from spectramr.infrastructure.optimization.ema import EMAKeyMismatchError

    model = nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.99, warmup=True)

    with pytest.raises(EMAKeyMismatchError, match="share no state-dict keys"):
        ema.update(_PrefixWrapper(model))


def test_the_raise_names_both_key_spellings():
    """The message has to be actionable -- the shape is invisible otherwise."""
    from spectramr.infrastructure.optimization.ema import EMAKeyMismatchError

    model = nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.99)
    with pytest.raises(EMAKeyMismatchError) as excinfo:
        ema.update(_PrefixWrapper(model))
    message = str(excinfo.value)
    assert "weight" in message and "module.weight" in message
    assert "unwrap_model" in message


def test_unwrapping_the_live_model_restores_blending():
    """The prescribed fix actually works on the wrapper that broke it."""
    from spectramr.core.module_utils import unwrap_model

    model = nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.5, warmup=False)
    before = ema.module.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)

    ema.update(unwrap_model(_PrefixWrapper(model)))

    assert not torch.allclose(before, ema.module.weight)


def test_a_partial_key_overlap_still_blends():
    """Only TOTAL mismatch is a wiring bug.

    A rebuilt ``channel_adapter`` legitimately leaves some shadow keys
    unmatched, and that path must keep working rather than start raising.
    """
    model = nn.Linear(4, 4)
    ema = ModelEma(model, decay=0.5, warmup=False)
    before = ema.module.weight.detach().clone()
    with torch.no_grad():
        model.weight.add_(1.0)

    partial = {"weight": model.state_dict()["weight"]}
    ema.update(_StateDictOnly(partial))

    assert not torch.allclose(before, ema.module.weight)


class _StateDictOnly(nn.Module):
    """A stand-in exposing a caller-chosen ``state_dict`` and nothing else."""

    def __init__(self, sd: dict) -> None:
        super().__init__()
        self._sd = sd

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        return self._sd


# ---------------------------------------------------------------------------
# The checkpoint form: bare weight keys PLUS the warmup counter.
# ---------------------------------------------------------------------------


def test_shadow_state_dict_has_bare_keys_and_the_counter():
    """Weight keys stay bare (the on-disk contract); the counter rides along."""
    ema = ModelEma(nn.Linear(4, 4), decay=0.99)
    ema.num_updates = 777
    sd = ema.shadow_state_dict()

    assert ModelEma._NUM_UPDATES_KEY in sd
    weight_keys = [k for k in sd if k != ModelEma._NUM_UPDATES_KEY]
    assert weight_keys and not any(k.startswith("module.") for k in weight_keys)


def test_shadow_state_dict_roundtrip_restores_the_ramp_position():
    """A resume must continue the decay ramp, not restart it at 0."""
    ema = ModelEma(nn.Linear(4, 4), decay=0.99, warmup=True)
    ema.num_updates = 777

    restored = ModelEma(nn.Linear(4, 4), decay=0.99, warmup=True)
    restored.load_shadow_state_dict(ema.shadow_state_dict())

    assert restored.num_updates == 777
    assert torch.allclose(restored.module.weight, ema.module.weight)


def test_load_shadow_state_dict_accepts_a_pre_fix_checkpoint():
    """Checkpoints written before the counter existed still load."""
    ema = ModelEma(nn.Linear(4, 4), decay=0.99)
    legacy = dict(ema.module.state_dict())  # bare, no counter -- the old format
    assert ModelEma._NUM_UPDATES_KEY not in legacy

    restored = ModelEma(nn.Linear(4, 4), decay=0.99)
    restored.num_updates = 5
    restored.load_shadow_state_dict(legacy)

    assert restored.num_updates == 5  # left alone, never invented


def test_unwrapping_before_state_dict_drops_the_counter():
    """Pins WHY shadow_state_dict exists, so the old spelling cannot return.

    ``unwrap_model`` peels ModelEma's ``.module``, so ``state_dict()`` runs on
    the inner module and the override that injects the counter never fires.
    That is the shape the checkpoint director shipped: every resume restarted
    the decay ramp at 0.
    """
    from spectramr.core.module_utils import unwrap_model

    ema = ModelEma(nn.Linear(4, 4), decay=0.99)
    ema.num_updates = 1234

    assert ModelEma._NUM_UPDATES_KEY not in unwrap_model(ema).state_dict()
    assert ModelEma._NUM_UPDATES_KEY in ema.shadow_state_dict()
