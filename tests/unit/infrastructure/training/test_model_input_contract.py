"""Width check of the model input the snapshot hooks record (#1298).

Width failures rest on a heuristic read of the backbone, so they warn and stamp;
a false positive must not abort a real run over a diagnostic.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spectramr.infrastructure.training.model_input_contract import (
    STATUS_MATCH,
    STATUS_MISMATCH,
    STATUS_UNRESOLVED,
    resolve_model_in_channels,
    verify_model_input,
)



def test_a_declared_in_channels_attribute_wins() -> None:
    """Most authoritative: what the architecture says it takes. It also survives
    a first layer this resolver would not otherwise recognise."""

    class _Net(nn.Module):
        in_channels = 16

        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(3, 4, 1)  # deliberately disagrees

    channels, source = resolve_model_in_channels(_Net())
    assert channels == 16
    assert source == "module.in_channels"


def test_the_first_conv_is_used_when_nothing_is_declared() -> None:
    net = nn.Sequential(nn.Conv2d(7, 4, 3), nn.ReLU(), nn.Conv2d(4, 4, 3))
    assert resolve_model_in_channels(net) == (7, "first Conv2d")


def test_conv3d_counts_too() -> None:
    net = nn.Sequential(nn.Conv3d(5, 2, 3))
    channels, source = resolve_model_in_channels(net)
    assert channels == 5
    assert "Conv3d" in source


def test_a_linear_first_layer_falls_back_to_in_features() -> None:
    net = nn.Sequential(nn.Linear(12, 4))
    assert resolve_model_in_channels(net) == (12, "first Linear.in_features")


def test_a_conv_is_preferred_over_a_linear_that_comes_first() -> None:
    """Ordering, not position: a channel width is what the caller compares
    against, and a projection head placed early would answer a different
    question."""
    net = nn.Sequential(nn.Linear(12, 4), nn.Conv2d(9, 4, 3))
    assert resolve_model_in_channels(net) == (9, "first Conv2d")


def test_an_unrecognisable_backbone_is_unresolved_not_zero() -> None:
    assert resolve_model_in_channels(nn.Sequential(nn.ReLU())) == (None, "unresolved")
    assert resolve_model_in_channels(None) == (None, "unresolved")


def test_a_nonsense_in_channels_does_not_win() -> None:
    """A Mock's auto-attribute is truthy and not an int; taking it would make
    every mock-fed strategy test report a mismatch."""

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = "sixteen"
            self.head = nn.Conv2d(6, 4, 1)

    assert resolve_model_in_channels(_Net()) == (6, "first Conv2d")


def test_a_bool_in_channels_does_not_win() -> None:
    """``bool`` is a subclass of ``int``, so ``in_channels = True`` used to
    resolve as a one-channel backbone and report a mismatch against every real
    tensor. A width of 1 is indistinguishable in the artifact from a genuine
    single-channel model, which is the reading that makes it worth excluding."""

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = True
            self.head = nn.Conv2d(6, 4, 1)

    assert resolve_model_in_channels(_Net()) == (6, "first Conv2d")


# ── Wrapper backbones: the width the SNAPSHOT is compared against ──────────
#
# The shapes below are the ones the resolver was blind to. A module that adapts
# channels before delegating holds two different true answers, and the tiers
# above return the one that does not describe the tensor under test.


def test_backbone_in_channels_outranks_a_wrappers_own_in_channels() -> None:
    """The planted shape (CLAUDE.md #15).

    ``KSpaceColdDiffusionGenerator`` sets ``in_channels = 8`` -- the bare
    measurement its ``forward`` concat gate keys on -- while building its
    backbone at 16 for the ``[noisy_kspace || smaps]`` stack the training path
    feeds it. Both numbers are true; only the second is what ``model_input``
    must match.
    """

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 8
            self.backbone_in_channels = 16
            self.backbone = nn.Conv2d(16, 4, 1)

    assert resolve_model_in_channels(_Wrapper()) == (16, "module.backbone_in_channels")


def test_a_conditioned_wrapper_stops_reporting_a_false_mismatch() -> None:
    """End to end, the regression this tier closes.

    Pre-fix this emitted "carries 16 channels ... but the backbone takes 8" on
    every smaps-conditioned cold-diffusion arm -- most of the cohort. A
    diagnostic that cries wolf across a whole paradigm gets muted, not read.
    """

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 8
            self.backbone_in_channels = 16
            self.backbone = nn.Conv2d(16, 4, 1)

    channels, source = resolve_model_in_channels(_Wrapper())
    verdict = verify_model_input(
        tensors={"model_input": torch.zeros(1, 16, 8, 8)},
        model_input_key="model_input",
        in_channels=channels,
        in_channels_source=source,
    )
    assert verdict.status == STATUS_MATCH


def test_the_new_tier_does_not_blind_the_check() -> None:
    """The other half of #15: a detector that stops firing is not a fix.

    A wrapper publishing 16 must still report a genuinely wrong width, or this
    tier would have traded a false positive for a false negative.
    """

    class _Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 8
            self.backbone_in_channels = 16
            self.backbone = nn.Conv2d(16, 4, 1)

    channels, source = resolve_model_in_channels(_Wrapper())
    verdict = verify_model_input(
        tensors={"model_input": torch.zeros(1, 5, 8, 8)},
        model_input_key="model_input",
        in_channels=channels,
        in_channels_source=source,
    )
    assert verdict.status == STATUS_MISMATCH
    assert "backbone_in_channels" in verdict.in_channels_source


def test_a_nonsense_backbone_in_channels_falls_through() -> None:
    """Same fail-open reasoning as ``in_channels``: a Mock's auto-attribute
    must not become a width, or every mock-fed strategy test reports a
    mismatch from the most-preferred tier."""

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone_in_channels = "sixteen"
            self.in_channels = 6
            self.head = nn.Conv2d(3, 4, 1)

    assert resolve_model_in_channels(_Net()) == (6, "module.in_channels")


def test_a_plain_backbone_without_the_attribute_is_unaffected() -> None:
    """The tier is opt-in: a model that never adapts channels keeps answering
    from ``in_channels`` exactly as before."""

    class _Net(nn.Module):
        in_channels = 12

        def __init__(self) -> None:
            super().__init__()
            self.head = nn.Conv2d(12, 4, 1)

    assert resolve_model_in_channels(_Net()) == (12, "module.in_channels")


# ── The comparison ─────────────────────────────────────────────────────────


def _verdict(tensor, in_channels, *, key="model_input", source="test"):
    return verify_model_input(
        tensors={key: tensor},
        model_input_key=key,
        in_channels=in_channels,
        in_channels_source=source,
    )


def test_matching_widths_report_a_match() -> None:
    v = _verdict(torch.randn(2, 16, 8, 8), 16)
    assert v.status == STATUS_MATCH
    assert v.declared_channels == 16
    assert v.model_in_channels == 16


def test_the_exact_1298_defect_reports_a_mismatch() -> None:
    """8 recorded channels against the 16-plane first conv the arm really had."""
    v = _verdict(torch.randn(1, 8, 320, 320), 16, key="noisy_kspace")
    assert v.status == STATUS_MISMATCH
    assert v.declared_channels == 8
    assert v.model_in_channels == 16


def test_the_detail_says_which_of_the_two_readings_it_could_be() -> None:
    """A mismatch is ambiguous by nature -- a mislabelled key or real miswiring
    -- and a verdict that asserted one would send half its readers the wrong
    way."""
    detail = _verdict(torch.randn(1, 8, 8, 8), 16).detail
    assert "wrong tensor" in detail
    assert "did not declare" in detail


def test_a_complex_tensor_may_feed_twice_its_channel_count() -> None:
    """Half this repo's k-space backbones take a real view of a complex input.
    The snapshot records the tensor as the strategy holds it, before that view."""
    v = _verdict(torch.randn(1, 8, 8, 8, dtype=torch.complex64), 16)
    assert v.status == STATUS_MATCH
    assert v.is_complex is True
    assert v.accepted_channels == (8, 16)


def test_a_complex_tensor_also_matches_its_own_width() -> None:
    """A native complex conv takes C, not 2C. Both are legal, so both accept."""
    assert _verdict(torch.randn(1, 8, 8, 8, dtype=torch.complex64), 8).status == STATUS_MATCH


def test_a_real_tensor_does_not_get_the_doubling_allowance() -> None:
    """The allowance exists for the real-view transform; extending it to real
    inputs would wave through exactly the 8-vs-16 confusion of #1298."""
    v = _verdict(torch.randn(1, 8, 8, 8), 16)
    assert v.status == STATUS_MISMATCH
    assert v.accepted_channels == (8,)


def test_an_unresolvable_width_is_unresolved_and_still_records_the_tensor() -> None:
    v = _verdict(torch.randn(1, 16, 8, 8), None, source="unresolved")
    assert v.status == STATUS_UNRESOLVED
    assert v.declared_channels == 16, "the knowable half is still recorded"
    assert v.model_in_channels is None


def test_a_tensor_without_a_channel_axis_is_unresolved_not_a_mismatch() -> None:
    v = _verdict(torch.randn(8), 16)
    assert v.status == STATUS_UNRESOLVED


def test_a_non_tensor_under_the_named_key_is_unresolved() -> None:
    """`_declare_model_input` takes `dict[str, Any]`; a scalar or None there is
    a defect, but not one this comparison can adjudicate."""
    assert _verdict(None, 16).status == STATUS_UNRESOLVED


# ── The record that reaches the artifact ───────────────────────────────────


def test_the_record_carries_declared_and_applied_side_by_side() -> None:
    """Non-negotiable 14: the divergence must be readable as a subtraction, not
    re-derived by a reader who still happens to have the model."""
    record = _verdict(torch.randn(1, 8, 8, 8), 16, key="noisy_kspace").as_record()
    assert record["model_input_key"] == "noisy_kspace"
    assert record["declared"]["channels"] == 8
    assert record["declared"]["shape"] == [1, 8, 8, 8]
    assert record["applied"]["in_channels"] == 16
    assert record["applied"]["resolved_from"] == "test"
    assert record["status"] == STATUS_MISMATCH
    assert record["detail"]


def test_the_record_is_json_serialisable() -> None:
    """It is stamped as a top-level JSON key, bypassing `_coerce_extra` -- so
    nothing downstream will stringify a stray tuple for it."""
    import json

    json.dumps(_verdict(torch.randn(1, 16, 8, 8), 16).as_record())
