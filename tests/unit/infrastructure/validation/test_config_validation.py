"""Pins for training-mode *existence* validation.

``training_mode`` means exactly one thing: a key in
``TrainingStrategyFactory.STRATEGY_CLASS_PATHS``. Enforcing that was previously
a side effect of membership in ``TRAINING_MODE_CONSTRAINTS`` -- a hand-maintained
second table that had drifted 90 keys behind the dispatch table. Real configs
(``physics_driven``, ``cyclegan``, ``meta_learning``) were rejected at bootstrap
with "Unknown training mode" while genuinely-undispatchable modes sailed through
whenever ``strategy_class`` happened to be set too.

The check now lives in ``infrastructure/`` where ``STRATEGY_CLASS_PATHS`` lives,
because ``config/`` may not import ``infrastructure/`` (non-negotiable #5).
"""

import pytest

from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory
from spectramr.infrastructure.validation.config_validation import (
    ConfigValidationError,
    ConfigValidator,
    _dispatchable_training_modes,
    _validate_training_mode_dispatchable,
)


def _cfg(mode, **training_extra):
    """Minimal validator-shaped dict carrying a training_mode."""
    training = {"training_mode": mode}
    training.update(training_extra)
    return {"training": training}


class TestDispatchableModes:
    def test_registered_mode_passes(self):
        assert _validate_training_mode_dispatchable(_cfg("gan")) == []

    def test_unregistered_mode_is_rejected(self):
        messages = _validate_training_mode_dispatchable(_cfg("not_a_real_mode_xyz"))
        assert len(messages) == 1
        assert "names no registered strategy" in messages[0]

    def test_rejection_suggests_a_close_match(self):
        """A typo should point at the intended key, not just fail."""
        (message,) = _validate_training_mode_dispatchable(_cfg("diffusionn"))
        assert "Did you mean" in message
        assert "diffusion" in message

    def test_absent_mode_is_not_an_error(self):
        """v6.0 configs dispatch on strategy_class and may omit training_mode."""
        assert _validate_training_mode_dispatchable({"training": {}}) == []
        assert _validate_training_mode_dispatchable({}) == []

    def test_none_training_section_is_not_an_error(self):
        """`training` defaults to None -- guard must match sibling rules."""
        assert _validate_training_mode_dispatchable({"training": None}) == []

    def test_non_string_mode_is_reported_not_crashed(self):
        """experiments/shared/*schema_reference*.yaml carry a dict here.

        A bare `mode not in some_set` would raise TypeError on an unhashable
        dict and get flattened into "Unexpected validation error".
        """
        (message,) = _validate_training_mode_dispatchable(_cfg({"nested": "dict"}))
        assert "must be a string" in message

    def test_unresolvable_mode_rejected_even_with_strategy_class(self):
        """The hole check_strategy_registry leaves open.

        That check is gated on `strategy_class is None`, so a bogus mode label
        rode along unnoticed whenever an explicit strategy_class was present.
        """
        messages = _validate_training_mode_dispatchable(
            _cfg(
                "bogus_label",
                strategy_class="spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy",
            )
        )
        assert len(messages) == 1


class TestDispatchableModeSet:
    def test_includes_every_static_registry_key(self):
        modes = _dispatchable_training_modes()
        assert set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS).issubset(modes)

    @pytest.mark.parametrize("mode", ["physics_driven", "cyclegan", "meta_learning"])
    def test_previously_blocked_modes_now_validate(self, mode):
        """Regression: these are dispatchable but were absent from
        TRAINING_MODE_CONSTRAINTS, so bootstrap raised ConfigValidationError on
        every YAML using them."""
        assert mode in TrainingStrategyFactory.STRATEGY_CLASS_PATHS
        assert _validate_training_mode_dispatchable(_cfg(mode)) == []

    @pytest.mark.parametrize("mode", ["kan", "pretraining", "sensitivity_estimation"])
    def test_deleted_orphan_modes_are_rejected(self, mode):
        """The three A3 orphans: advertised for months, dispatchable never."""
        assert mode not in _dispatchable_training_modes()
        assert _validate_training_mode_dispatchable(_cfg(mode)) != []


class _StubSettings:
    """Duck-types the one thing ``ConfigValidator._config_to_dict`` uses.

    ``_config_to_dict`` is exactly ``config.model_dump()``, so this drives the
    REAL ``validate`` — registry pass, splice, and raise — without paying for a
    fully-populated ``TrainingSettings``.
    """

    def __init__(self, payload):
        self._payload = payload

    def model_dump(self):
        return self._payload


def _registry_clean(mode):
    """A config whose ONLY defect is ``training_mode`` — nothing else errors.

    Verified against the live registry: batch_size / learning_rate / epochs are
    the three unconditional error rules, and no registry rule rejects an unknown mode.
    So any error this config produces came from the splice.
    """
    return _StubSettings(
        {
            "training": {"training_mode": mode, "epochs": 1},
            "data": {"batch_size": 2},
            "optimization": {"learning_rate": 1e-4},
        }
    )


class TestValidateFunnelActuallyRunsTheCheck:
    """The splice must stay wired into ``ConfigValidator.validate`` itself.

    Every other test here calls ``_validate_training_mode_dispatchable``
    directly, which proves the helper is correct but says nothing about whether
    anything CALLS it. Move the splice inside the ``if error_messages:`` block or
    below the raise and all of those stay green while the check goes inert at
    bootstrap — the landed-but-unwired failure this repo keeps hitting
    (pitfall #16). These tests fail in that scenario.
    """

    def test_undispatchable_mode_alone_fails_the_funnel(self):
        """Load-bearing: the check must raise with NO other error present.

        If the splice were nested under ``if error_messages:``, this config
        (registry-clean) would validate successfully and the bad mode would
        reach ``TrainingStrategyFactory`` at bootstrap instead.
        """
        with pytest.raises(ConfigValidationError) as exc:
            ConfigValidator.validate(_registry_clean("not_a_real_mode_xyz"))

        message = str(exc.value)
        assert "[training_mode_dispatchable]" in message
        assert "not_a_real_mode_xyz" in message

    def test_registry_clean_config_has_no_other_errors(self):
        """Pins the premise of the test above rather than assuming it.

        If a future rule starts erroring on this payload, the test above would
        pass for the wrong reason (some other error triggering the raise) and
        would stop guarding the splice. This fails first and says why.
        """
        with pytest.raises(ConfigValidationError) as exc:
            ConfigValidator.validate(_registry_clean("not_a_real_mode_xyz"))

        offending_rules = [
            line.split("]", 1)[0] + "]"
            for line in str(exc.value).splitlines()
            if line.startswith("[")
        ]
        assert offending_rules == ["[training_mode_dispatchable]"], (
            "expected the mode check to be the ONLY error on this payload, got: "
            f"{offending_rules}"
        )

    def test_dispatchable_mode_passes_the_funnel(self):
        """The mirror case: a real mode must not be rejected by the splice.

        ``physics_driven`` is deliberate — it is dispatchable and was missing from
        the old constraints table, so it is exactly the population this PR unblocked. Before the fix this raised "Unknown training mode".
        """
        ConfigValidator.validate(_registry_clean("physics_driven"))
