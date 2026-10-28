"""Tests for the multi-contrast fusion generator.

This module was registered, selectable from YAML, referenced by three
experiment configs — and could not complete a single forward pass (issue #508).
The first test here is that exact repro. A paired test asserting a forward pass
would have caught it at authoring time, which is why one exists now.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.models.blocks.instrument_keyed_attention import (  # noqa: E402
    InstrumentKeyedCrossAttention,
)
from spectramr.models.generators.multicontrast_fusion_generator import (  # noqa: E402
    MultiContrastFusionGenerator,
)


def _gen(**kw) -> MultiContrastFusionGenerator:
    torch.manual_seed(0)
    base = {
        "n_contrasts": 8,
        "channels_per_stream": 1,
        "out_channels": 1,
        "base_channels": 16,
        "num_layers": 2,
        "attention_heads": 4,
        "dropout_rate": 0.0,
    }
    return MultiContrastFusionGenerator(**{**base, **kw})


# ── issue #508 ───────────────────────────────────────────────────────────────


def test_forward_pass_completes() -> None:
    """The verbatim repro from #508. Was: NotImplementedError, because
    `_build_encoder` returned an nn.ModuleList and `forward` called it."""
    model = MultiContrastFusionGenerator(
        n_contrasts=8, channels_per_stream=1, out_channels=1, base_channels=32
    )
    assert model(torch.zeros(2, 8, 128, 128)).shape == (2, 1, 128, 128)


def test_decoder_does_not_expect_skips_it_never_receives() -> None:
    """The decoder doubled its input channels for a skip connection `forward`
    never carried across, so it would have raised on a shape mismatch even once
    the container bug was fixed. Deep stacks exercise every decoder stage."""
    for num_layers in (1, 2, 3):
        model = _gen(num_layers=num_layers)
        assert model(torch.randn(1, 8, 32, 32)).shape == (1, 1, 32, 32)


def test_adaptive_fusion_receives_gradient() -> None:
    """It was constructed under a default-True knob and never called: parameters
    in the optimizer, no gradient, an advertised knob with no effect (#508,
    pitfalls #15 and #16)."""
    model = _gen(use_adaptive_fusion=True)
    model(torch.randn(1, 8, 32, 32)).pow(2).mean().backward()
    grads = [float(p.grad.norm()) for p in model.adaptive_fusion.parameters() if p.grad is not None]
    assert grads and max(grads) > 0.0


def test_adaptive_gate_is_the_identity_at_init() -> None:
    """Uniform fusion weights renormalised by N, so the gating reweights the
    frames without silently rescaling them."""
    model = _gen(use_adaptive_fusion=True)
    # base_channels * 2**num_layers = 16 * 4: the gate lives at the encoder
    # OUTPUT width, not the base width.
    frames = torch.randn(2, 8, 64, 8, 8)
    gated = model.adaptive_fusion.gate(frames)
    assert gated.shape == frames.shape
    # only the SE gate should modulate; the frame weighting must be neutral
    weights = torch.softmax(model.adaptive_fusion.fusion_weights, dim=0) * 8
    assert torch.allclose(weights, torch.ones(8), atol=1e-6)


# ── the super-resolution tail ────────────────────────────────────────────────


@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_scale_reaches_the_target_grid(scale: int) -> None:
    """Encoders pool by 2**num_layers and the decoder returns to the INPUT grid,
    so without this tail the module could only emit at the resolution it was
    given (noted in #508)."""
    model = _gen(scale=scale)
    out = model(torch.randn(1, 8, 24, 24))
    assert out.shape == (1, 1, 24 * scale, 24 * scale)
    assert model.get_output_shape((1, 8, 24, 24)) == tuple(out.shape)


def test_non_positive_scale_raises() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _gen(scale=0)


# ── the input contract ───────────────────────────────────────────────────────


def test_instrument_channels_double_the_image_width() -> None:
    model = _gen(attention_keys="marker", instrument_channels=True)
    assert model.expected_in_channels == 16
    assert model(torch.randn(1, 16, 24, 24)).shape == (1, 1, 24, 24)


def test_conditioning_is_frame_major_and_per_stream() -> None:
    """Each frame gets its own (dy, dx), matching how the multi-acquisition
    strategy lays the maps out. A network handed the whole block per stream
    could not tell which offset was its own."""
    model = _gen(attention_keys="marker", instrument_channels=True, conditioning_per_stream=2)
    assert model.expected_in_channels == 8 + 8 + 16
    assert model(torch.randn(1, 32, 24, 24)).shape == (1, 1, 24, 24)


def test_wrong_channel_count_raises() -> None:
    model = _gen(conditioning_per_stream=2)
    with pytest.raises(ValueError, match="expected 24 input channels"):
        model(torch.randn(1, 8, 24, 24))


def test_declared_in_channels_must_match_the_implied_contract() -> None:
    """`in_channels` is the TOTAL width, as everywhere else in the framework.
    Raising beats preferring one of the two numbers: a silently-ignored channel
    count is how one stream ends up reading another stream's data."""
    with pytest.raises(ValueError, match="disagrees with the input contract"):
        _gen(in_channels=24, instrument_channels=True, conditioning_per_stream=2)
    # the consistent declaration is accepted
    assert (
        _gen(
            in_channels=32, instrument_channels=True, conditioning_per_stream=2
        ).expected_in_channels
        == 32
    )


def test_marker_keying_without_instrument_channels_raises() -> None:
    """There would be nowhere for the marker features to arrive from."""
    with pytest.raises(ValueError, match="needs instrument_channels=True"):
        _gen(attention_keys="marker", instrument_channels=False)


# ── the ablation ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("keys", ["content", "marker", "none"])
def test_every_routing_runs_at_a_fixed_input_geometry(keys: str) -> None:
    """All three see the identical tensor, so only the routing varies."""
    model = _gen(
        attention_keys=keys,
        instrument_channels=True,
        conditioning_per_stream=2,
        scale=3,
    )
    assert model.expected_in_channels == 32
    assert model(torch.randn(2, 32, 24, 24)).shape == (2, 1, 72, 72)


def test_content_and_marker_arms_are_capacity_matched() -> None:
    """The only comparison in the cohort not confounded by parameter count."""
    kw = {"instrument_channels": True, "conditioning_per_stream": 2, "scale": 3}
    n = {
        k: sum(p.numel() for p in _gen(attention_keys=k, **kw).parameters())
        for k in ("content", "marker", "none")
    }
    assert n["content"] == n["marker"]
    assert n["none"] < n["content"]  # inherent to removing the mechanism


@pytest.mark.parametrize(("keys", "expected_passes"), [("content", 1), ("none", 1), ("marker", 2)])
def test_instrument_stream_is_encoded_only_when_marker_keys_read_it(
    keys: str, expected_passes: int
) -> None:
    """Content and none keys discard the instrument features, so encoding them is wasted compute."""
    model = _gen(
        attention_keys=keys,
        instrument_channels=True,
        conditioning_per_stream=2,
        scale=3,
    )
    calls = []
    original = model._encode_streams

    def counting(x, cond=None):
        calls.append(x.shape[1])
        return original(x, cond)

    model._encode_streams = counting
    model(torch.randn(2, 32, 24, 24))
    assert len(calls) == expected_passes


def test_attention_keys_selects_the_instrument_keyed_block() -> None:
    """Leaving it unset keeps the legacy boolean behaviour untouched."""
    assert isinstance(
        _gen(attention_keys="content").fusion_attention, InstrumentKeyedCrossAttention
    )
    assert not isinstance(_gen().fusion_attention, InstrumentKeyedCrossAttention)
