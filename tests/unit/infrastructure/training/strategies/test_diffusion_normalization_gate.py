"""An identity ``kspace_scale`` must not read as "already normalized".

``DiffusionTrainingStrategy`` decided whether the data pipeline had normalized a
batch by asking whether ``kspace_scale`` was present. Two very different batches
answer yes:

* ``KSpaceNormalizationTransform`` publishes the scale it divided by.
* ``M4RawRepetitionDataset`` publishes an identity ``kspace_scale = 1.0`` to say
  "the data leaves the dataset unnormalized", so the published scale always
  matches the tensor served beside it.

Reading presence as evidence skipped ``apply_kspace_normalization`` on exactly
the second kind, and the model trained on raw k-space with the ~200x
DC-vs-periphery range that ``enable_log_scaling`` exists to compress.

``experiment_11_attention_none`` recorded this: its step-6 snapshot has the batch
reaching ``train_step`` at ``abs_max 2406.9``, and a float32 ``log1p`` output
cannot exceed ``ln(FLT_MAX) ~ 88.7`` — so that batch was provably never
compressed.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.infrastructure.training.strategies.diffusion import (
    DiffusionTrainingStrategy,
)

#: ``ln(FLT_MAX)``. A phase-preserving ``log1p`` magnitude compression cannot
#: produce a float32 component above this, which is what makes the snapshot's
#: 2406.9 proof of absence rather than a judgement call.
LOG1P_FLOAT32_CEILING = 88.72


class _Recorder:
    """Minimal logging service: the gate only calls ``log_warning``."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def log_warning(self, message: str) -> None:
        self.warnings.append(message)


@pytest.fixture
def gate():
    """The decision function, detached from strategy construction.

    Built with ``__new__`` deliberately: instantiating the strategy needs a full
    config, container and model, none of which this decision touches. Binding the
    two attributes it does read keeps the test on the branch logic.
    """
    strategy = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
    strategy._warned_batch_unnormalized = False
    strategy.logging_service = _Recorder()
    return strategy


def test_identity_scale_alone_is_not_evidence_of_normalization(gate) -> None:
    """The regression: 1.0 with no marker must route to the fallback."""
    assert gate._batch_is_already_normalized({"kspace_scale": torch.tensor(1.0)}, 1.0, 6) is False


def test_a_real_scale_is_evidence(gate) -> None:
    """224.36 is what this arm's own chain produces; something divided by it."""
    scale = torch.tensor(224.359)
    assert gate._batch_is_already_normalized({"kspace_scale": scale}, scale, 6) is True
    assert gate.logging_service.warnings == [], (
        "a normalized batch must not warn — that would fire every run and "
        "CLAUDE.md #4 makes warnings a failure, not noise"
    )


def test_the_explicit_marker_outranks_the_scale_value(gate) -> None:
    """``kspace_normalized`` is an answer; the scale's value is an inference.

    Both directions matter. A transform that legitimately measured a scale of
    1.0 must still count as normalized, and a dataset that says False must not be
    overridden by a coincidentally non-unit scale.
    """
    scale_one = torch.tensor(1.0)
    assert (
        gate._batch_is_already_normalized(
            {"kspace_scale": scale_one, "kspace_normalized": True}, scale_one, 6
        )
        is True
    )
    assert (
        gate._batch_is_already_normalized(
            {"kspace_scale": torch.tensor(7.0), "kspace_normalized": False},
            torch.tensor(7.0),
            6,
        )
        is False
    )


def test_a_collated_per_sample_marker_is_reduced_not_truthy_tested(gate) -> None:
    """Collation turns a per-subject bool into one value per sample.

    ``bool(tensor([False, False]))`` raises rather than being False, and a
    half-normalized batch must not take the fast path on the strength of its
    first sample.
    """
    assert (
        gate._batch_is_already_normalized(
            {"kspace_normalized": torch.tensor([True, True])}, None, 6
        )
        is True
    )
    assert (
        gate._batch_is_already_normalized(
            {"kspace_normalized": torch.tensor([True, False])}, None, 6
        )
        is False
    )


def test_an_absent_scale_still_routes_to_the_fallback(gate) -> None:
    """The original ``is not None`` behaviour, preserved."""
    assert gate._batch_is_already_normalized({}, None, 6) is False


def test_the_identity_scale_warning_names_the_wiring_point_once(gate) -> None:
    """Silence here cost a 30,000-iteration run; 30,000 warnings would too."""
    for step in range(3):
        gate._batch_is_already_normalized(
            {"kspace_scale": torch.tensor(1.0)}, torch.tensor(1.0), step
        )
    assert len(gate.logging_service.warnings) == 1
    message = gate.logging_service.warnings[0]
    assert "IDENTITY" in message
    assert "KSpaceNormalizationTransform" in message, (
        "the warning must name the component that did not run, or it is not "
        "actionable from a cluster log"
    )


class TestEveryUnnormalizedRouteAnnouncesItself:
    """#1211: three routes reach ``False``; only the identity one used to warn.

    Each route means the declared ``KSpaceNormalizationTransform`` did not reach
    ``train_step`` and the strategy is compensating for the data layer. A silent
    route is a 30,000-iteration run with nothing in the log, so the gate is not
    allowed to have one.
    """

    def test_an_explicit_false_marker_warns(self, gate) -> None:
        """The loudest signal available was the silent one.

        ``M4RawRepetitionDataset`` publishes ``kspace_normalized=False``, so this
        is the route ``experiment_11_attention_none`` actually took — and it
        returned before either existing warning could fire.
        """
        assert gate._batch_is_already_normalized({"kspace_normalized": False}, None, 6) is False
        assert len(gate.logging_service.warnings) == 1, (
            "an explicit False, with normalization declared, is the strongest "
            "evidence of the #1213 wiring defect — it must not be silent"
        )
        assert "kspace_normalized=False" in gate.logging_service.warnings[0]

    def test_an_absent_scale_warns_and_says_which_state_it_saw(self, gate) -> None:
        """The state the issue was filed for: ``return False`` before the warning."""
        assert gate._batch_is_already_normalized({}, None, 6) is False
        assert len(gate.logging_service.warnings) == 1
        message = gate.logging_service.warnings[0]
        assert "NEITHER" in message, (
            "'no scale published' and 'identity scale published' point at "
            "different wiring faults; a merged message sends the reader to the "
            "wrong place"
        )
        assert "IDENTITY" not in message

    def test_the_three_states_are_distinguishable(self, gate) -> None:
        """Same one-shot warning, different diagnosis text."""
        cases = {
            "marker": ({"kspace_normalized": False}, None),
            "absent": ({}, None),
            "identity": ({"kspace_scale": torch.tensor(1.0)}, torch.tensor(1.0)),
        }
        seen = {}
        for name, (batch, scale) in cases.items():
            fresh = DiffusionTrainingStrategy.__new__(DiffusionTrainingStrategy)
            fresh._warned_batch_unnormalized = False
            fresh.logging_service = _Recorder()
            assert fresh._batch_is_already_normalized(batch, scale, 6) is False
            seen[name] = fresh.logging_service.warnings[0]

        assert len({*seen.values()}) == 3, (
            f"the three states must be distinguishable in the log, got {seen}"
        )
        for message in seen.values():
            assert "#1213" in message, (
                "the compensation masks the upstream defect, so the message must "
                "say so — otherwise a healthy loss reads as a wired data layer"
            )

    def test_one_flag_rate_limits_all_three_states(self, gate) -> None:
        """A run that trips two states must still log once, not twice."""
        gate._batch_is_already_normalized({"kspace_normalized": False}, None, 1)
        gate._batch_is_already_normalized({}, None, 2)
        gate._batch_is_already_normalized({"kspace_scale": torch.tensor(1.0)}, torch.tensor(1.0), 3)
        assert len(gate.logging_service.warnings) == 1

    def test_a_normalized_batch_never_warns(self, gate) -> None:
        """CLAUDE.md #4 makes warnings a failure, so the healthy path stays quiet."""
        scale = torch.tensor(224.359)
        assert (
            gate._batch_is_already_normalized(
                {"kspace_scale": scale, "kspace_normalized": True}, scale, 6
            )
            is True
        )
        assert gate._batch_is_already_normalized({"kspace_scale": scale}, scale, 6) is True
        assert gate.logging_service.warnings == []


class TestWhyTheSnapshotProvesItWasNeverCompressed:
    """The arithmetic the diagnosis rests on, pinned rather than asserted."""

    def test_log1p_compression_cannot_reach_the_snapshot_magnitude(self) -> None:
        """No float32 input compresses to 2406.9, so the batch was raw."""
        from spectramr.data.transforms.normalization import compress_kspace_log

        # ``ln(FLT_MAX)`` is the ceiling in exact arithmetic, but the magnitude
        # goes through ``sqrt(R^2 + I^2)``, so the largest input that survives
        # the square is ``sqrt(FLT_MAX) ~ 1.84e19`` and the reachable ceiling is
        # ``ln(sqrt(FLT_MAX)) ~ 44.4`` — tighter still. Sweep the decades up to
        # it rather than pinning one brittle constant.
        largest_squarable = float(torch.finfo(torch.float32).max) ** 0.5
        peaks = []
        for exponent in range(0, 20, 4):
            magnitude = min(10.0**exponent, largest_squarable)
            extreme = torch.zeros(2, 8, 8)
            extreme[0] = magnitude  # imag 0, so |k| is exactly the real part
            compressed = compress_kspace_log(extreme, channel_dim=0)
            assert torch.isfinite(compressed).all(), f"overflow at 1e{exponent}"
            peaks.append(float(compressed.abs().max()))

        assert max(peaks) < LOG1P_FLOAT32_CEILING, (
            f"compressed peaks {peaks} must stay under ln(FLT_MAX)"
        )
        assert max(peaks) < 2406.9, (
            "if the compressed ceiling could reach experiment_11's recorded "
            "abs_max, that snapshot would not prove the batch was uncompressed"
        )
        assert float(compressed.abs().max()) < 2406.9, (
            "if the compressed ceiling could reach the snapshot's abs_max, the "
            "snapshot would not be evidence of anything"
        )

    def test_the_marker_is_no_longer_write_only(self) -> None:
        """``kspace_log_scaled`` was set and never read; don't repeat that.

        ``kspace_normalized`` exists to be consumed by the gate, so a grep that
        finds only the write site is a regression in this change specifically.
        The gate reads it through the shared ``loader_normalization``.
        """
        import inspect

        from spectramr.data.transforms import normalization

        writer = inspect.getsource(normalization.KSpaceNormalizationTransform)
        reader = inspect.getsource(normalization.loader_normalization)
        gate = inspect.getsource(DiffusionTrainingStrategy._batch_is_already_normalized)
        assert "kspace_normalized" in writer
        assert "kspace_normalized" in reader
        assert "loader_normalization(" in gate


class TestTheVerdictDoesNotDependOnTheContainerType:
    """The gate must read the DATA, not the shape of the box it arrived in.

    Every other test in this module feeds a plain ``dict`` -- and that is exactly
    why they all passed while the real training path failed. The loop does not
    hand ``train_step`` a dict: ``pipelines/training_loop.py`` converts the
    collated batch with ``BatchAdapter.from_dict`` first, so what actually
    arrives is a :class:`TrainingBatch` that files every non-core key into
    ``.metadata``.

    Against that object the old guard -- ``isinstance(batch_data, dict)`` with a
    ``hasattr`` fallback -- missed on *both* legs: a dataclass is not a mapping,
    and attribute lookup cannot see ``.metadata``. The published marker read as
    absent, so an already-normalized batch was normalized a second time and the
    run warned that the batch had published nothing while it had published both.

    A dict-only fixture cannot catch that, which is the lesson worth pinning:
    these tests assert the two containers agree, so the mock-shaped path can
    never again diverge from the production one.
    """

    #: The scale this arm's own chain produces, from the step-6 snapshot.
    REAL_SCALE = 224.359

    @staticmethod
    def _as_batch(payload: dict):
        """The same payload, through the conversion the training loop performs."""
        from spectramr.data.batch_types import BatchAdapter

        tensor = torch.randn(1, 2, 8, 8)
        return BatchAdapter.from_dict({"input": tensor, "target": tensor.clone(), **payload})

    def test_a_training_batch_marker_is_seen_exactly_like_a_dict_marker(self, gate) -> None:
        """The regression, stated as an invariant over container type.

        ``kspace_scale=None`` is deliberate and load-bearing. The production call
        site derives that argument from the batch with the *same* broken read, so
        handing the predicate a ready-made scale lets it answer True through the
        parameter and never exercises the marker lookup at all -- the first draft
        of this test did exactly that and passed against the unfixed source.
        ``None`` is what the broken read actually produced, so the marker in
        ``.metadata`` becomes the only route to a True verdict.
        """
        payload = {
            "kspace_scale": torch.tensor(self.REAL_SCALE),
            "kspace_normalized": torch.tensor(True),
        }

        as_batch = gate._batch_is_already_normalized(self._as_batch(payload), None, 6)

        gate._warned_batch_unnormalized = False
        gate.logging_service.warnings.clear()
        as_dict = gate._batch_is_already_normalized(payload, None, 6)

        assert as_batch is as_dict is True, (
            "the batch published kspace_normalized=True either way; a TrainingBatch "
            "answering False is the double-normalization bug"
        )

    def test_a_training_batch_that_published_both_markers_does_not_warn(self, gate) -> None:
        """The warning claimed the batch published NEITHER. It published both."""
        payload = {
            "kspace_scale": torch.tensor(self.REAL_SCALE),
            "kspace_normalized": torch.tensor(True),
        }
        gate._batch_is_already_normalized(self._as_batch(payload), None, 6)
        assert gate.logging_service.warnings == [], (
            f"warned about a fully-marked batch: {gate.logging_service.warnings}"
        )

    def test_an_explicit_false_marker_is_honoured_through_a_training_batch(self, gate) -> None:
        """A published ``False`` must still route to the fallback AND warn.

        The fix must not overshoot into "any TrainingBatch is normalized". This
        arm's dataset publishes ``False``, which is the loudest signal available
        and must survive the container change intact (issues #1211, #1213).
        """
        payload = {
            "kspace_scale": torch.tensor(1.0),
            "kspace_normalized": torch.tensor(False),
        }
        verdict = gate._batch_is_already_normalized(
            self._as_batch(payload), payload["kspace_scale"], 6
        )
        assert verdict is False
        assert len(gate.logging_service.warnings) == 1
        assert "kspace_normalized=False" in gate.logging_service.warnings[0]

    def test_a_training_batch_carrying_no_markers_still_routes_to_the_fallback(
        self, gate
    ) -> None:
        """Absence must remain absence -- the helper adds reach, not optimism."""
        assert gate._batch_is_already_normalized(self._as_batch({}), None, 6) is False
        assert len(gate.logging_service.warnings) == 1
