"""The physics options of the bounded reverse loops are validated once, and refused loudly."""

from __future__ import annotations

import pytest

from spectramr.models.diffusion.kspace_process import VALID_REVERSE_MODES
from spectramr.models.diffusion.reverse_decay_axis import DECAY_AXES, LINE_DECAY_AXES
from spectramr.models.diffusion.reverse_hooks import _PROJECTORS
from spectramr.models.diffusion.reverse_physics import (
    DECAY_ENVELOPE_SHAPES,
    HARD_PINNED_REVERSE_MODE,
    HOOKED_REVERSE_MODES,
    LINE_ENVELOPE_AXIS,
    REVERSE_PHYSICS_KEYS,
    SUPPORT_ORDERED_REVERSE_MODE,
    UNBOUNDED_REVERSE_MODE,
    VALID_DECAY_ENVELOPES,
    VALID_PROJECTIONS,
    UNCONSTRAINABLE_REVERSE_MODES,
    VALID_REVERSE_DC,
    ReversePhysics,
    resolve_reverse_physics,
    resolve_reverse_physics_kwargs,
)

pytestmark = pytest.mark.unit


def test_defaults_are_every_option_off() -> None:
    assert resolve_reverse_physics("replace_freeze_dc", "hard") == ReversePhysics()


def test_the_hooked_modes_and_projectors_have_one_owner() -> None:
    assert HOOKED_REVERSE_MODES <= VALID_REVERSE_MODES
    assert SUPPORT_ORDERED_REVERSE_MODE in VALID_REVERSE_MODES
    assert set(_PROJECTORS) == VALID_PROJECTIONS
    assert {"off", *DECAY_ENVELOPE_SHAPES} == VALID_DECAY_ENVELOPES
    # ``line`` resolves to the mask's own line axis (#2160); every binning it can
    # resolve to, plus the named ones, is exactly the axis registry.
    named = {axis for _, axis in DECAY_ENVELOPE_SHAPES.values()} - {LINE_ENVELOPE_AXIS}
    assert named | set(LINE_DECAY_AXES.values()) == set(DECAY_AXES)
    assert DECAY_ENVELOPE_SHAPES["per_sample"][1] == LINE_ENVELOPE_AXIS


@pytest.mark.parametrize(
    ("mode", "dc", "kwargs", "match"),
    [
        ("replace_freeze_dc", "hard", {"decay_envelope": "global"}, "reverse_decay_envelope"),
        ("replace_freeze_dc", "hard", {"projection": "grappa"}, "reverse_projection"),
        ("replace_freeze_dc", "hard", {"dc_trust": "wiener"}, "reverse_dc_trust"),
        ("replace_freeze_dc", "hard", {"projection": 1}, "reverse_projection"),
        ("additive", "hard", {"projection": "spirit"}, "not read by"),
        ("replace_freeze", "hard", {"decay_envelope": "per_sample"}, "not read by"),
        ("replace_freeze_dc", "hard", {"decay_envelope": "per_sample_radia"}, "reverse_decay_envelope"),
        ("replace_freeze", "hard", {"decay_envelope": "per_sample_radial"}, "not read by"),
        ("additive", "hard", {"decay_envelope": "batch_pooled_radial"}, "not read by"),
        (
            "replace_freeze_dc",
            "noise_adaptive",
            {"dc_trust": "fitted_decay", "decay_envelope": "per_sample_radial"},
            "radial",
        ),
        (SUPPORT_ORDERED_REVERSE_MODE, "hard", {}, "requires reverse_support_radius"),
        (SUPPORT_ORDERED_REVERSE_MODE, "hard", {"support_radius": 0}, ">= 1"),
        (SUPPORT_ORDERED_REVERSE_MODE, "hard", {"support_radius": True}, ">= 1"),
        (SUPPORT_ORDERED_REVERSE_MODE, "hard", {"support_radius": 2.0}, ">= 1"),
        ("replace_freeze_dc", "hard", {"support_radius": 2}, "read only by"),
        (
            "replace_freeze_dc",
            "hard",
            {"dc_trust": "fitted_decay", "decay_envelope": "per_sample"},
            "noise_adaptive",
        ),
        ("replace_freeze_dc", "noise_adaptive", {"dc_trust": "fitted_decay"}, "decay envelope"),
    ],
)
def test_each_illegal_shape_raises(mode, dc, kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_reverse_physics(mode, dc, **kwargs)


@pytest.mark.parametrize("envelope", ["per_sample_radial", "batch_pooled_radial"])
def test_the_radial_envelopes_resolve_on_a_hooked_loop(envelope) -> None:
    physics = resolve_reverse_physics("replace_freeze_dc", "hard", decay_envelope=envelope)
    assert physics == ReversePhysics(decay_envelope=envelope)
    assert DECAY_ENVELOPE_SHAPES[envelope][1] == "radial"


@pytest.mark.parametrize("mode", sorted(HOOKED_REVERSE_MODES))
def test_every_hooked_mode_accepts_every_option(mode) -> None:
    radius = 2 if mode == SUPPORT_ORDERED_REVERSE_MODE else None
    physics = resolve_reverse_physics(
        mode,
        "noise_adaptive",
        decay_envelope="batch_pooled",
        projection="joint_tv",
        dc_trust="fitted_decay",
        support_radius=radius,
    )
    assert physics == ReversePhysics("batch_pooled", "joint_tv", "fitted_decay", radius)


def test_the_kwargs_spelling_reads_the_yaml_keys() -> None:
    kwargs = {
        REVERSE_PHYSICS_KEYS["decay_envelope"]: "per_sample",
        REVERSE_PHYSICS_KEYS["projection"]: "spirit",
        REVERSE_PHYSICS_KEYS["support_radius"]: 4,
        "unrelated": 1,
    }
    physics = resolve_reverse_physics_kwargs(SUPPORT_ORDERED_REVERSE_MODE, "hard", kwargs)
    assert physics == ReversePhysics("per_sample", "spirit", "measured", 4)


@pytest.mark.parametrize("integrity", [None, "off"])
def test_a_clip_ratio_under_additive_raises(integrity) -> None:
    """``additive`` applies no bound, so a declared ratio is an unread knob (#2428)."""
    kwargs = {"reverse_clip_ratio": 1.3, "reverse_integrity_check": integrity}
    with pytest.raises(ValueError, match="reverse_clip_ratio is not read"):
        resolve_reverse_physics_kwargs(UNBOUNDED_REVERSE_MODE, "hard", kwargs)


@pytest.mark.parametrize(
    ("mode", "kwargs"),
    [
        # The integrity verifier reads the ratio as the ceiling it reports against.
        (UNBOUNDED_REVERSE_MODE, {"reverse_clip_ratio": 1.3, "reverse_integrity_check": "raise"}),
        (UNBOUNDED_REVERSE_MODE, {}),
        ("replace_freeze_dc", {"reverse_clip_ratio": 1.3}),
        ("additive_normalized", {"reverse_clip_ratio": 1.3}),
    ],
)
def test_a_clip_ratio_a_loop_reads_is_accepted(mode, kwargs) -> None:
    resolve_reverse_physics_kwargs(mode, "hard", kwargs)


@pytest.mark.parametrize("dc", ["soft", "noise_adaptive"])
def test_replace_freeze_refuses_a_dc_method_it_never_reads(dc) -> None:
    """``replace_freeze`` pins hard whatever ``dc_method`` says, so a blend is an unread knob."""
    with pytest.raises(ValueError, match="never reads dc_method"):
        resolve_reverse_physics(HARD_PINNED_REVERSE_MODE, dc)


def test_replace_freeze_accepts_hard_dc() -> None:
    resolve_reverse_physics(HARD_PINNED_REVERSE_MODE, "hard")


def test_reverse_dc_defaults_to_per_step_and_reads_its_yaml_key() -> None:
    assert ReversePhysics().reverse_dc == "per_step"
    physics = resolve_reverse_physics_kwargs(
        "replace_freeze_dc", "hard", {REVERSE_PHYSICS_KEYS["reverse_dc"]: "unconstrained"}
    )
    assert physics.reverse_dc == "unconstrained"
    assert {"per_step", "unconstrained"} == VALID_REVERSE_DC


@pytest.mark.parametrize("mode", sorted(VALID_REVERSE_MODES - UNCONSTRAINABLE_REVERSE_MODES))
def test_unconstrained_is_refused_where_it_would_not_change_the_trajectory(mode) -> None:
    radius = 2 if mode == SUPPORT_ORDERED_REVERSE_MODE else None
    with pytest.raises(ValueError, match="reverse_dc='unconstrained' is read only by"):
        resolve_reverse_physics(mode, "hard", support_radius=radius, reverse_dc="unconstrained")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"reverse_dc": "none"}, "reverse_dc"),
        ({"reverse_dc": "unconstrained", "dc_passthrough_center_size": 16}, "centre patch"),
        (
            {
                "reverse_dc": "unconstrained",
                "reverse_dc_trust": "fitted_decay",
                "reverse_decay_envelope": "per_sample",
            },
            "would be unread",
        ),
    ],
)
def test_reverse_dc_rejects_unknown_values_and_unread_companions(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_reverse_physics_kwargs("replace_freeze_dc", "noise_adaptive", kwargs)
