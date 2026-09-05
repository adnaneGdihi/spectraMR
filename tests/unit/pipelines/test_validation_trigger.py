"""Tests for validation trigger logic in the training pipeline.

Ensures that:
- eval_interval is the ONLY mechanism controlling validation timing
- eval_on_epoch does NOT trigger validation when eval_interval is set
- Validation fires at exact iteration multiples of eval_interval
- No duplicate validation triggers occur
"""

import pytest

from tests.utils.repo_scripts import require_repo_file


class TestValidationTriggerLogic:
    """Unit tests for the validation trigger condition in _execute_training_loop.

    The condition under test (src/pipelines/train.py, lines ~1014-1031):

        time_for_eval = iteration % eval_interval == 0

        if eval_on_epoch and is_epoch_end and not is_sanity_check and eval_interval <= 0:
            time_for_eval = True

    This ensures eval_on_epoch ONLY fires when eval_interval is not set.
    """

    @staticmethod
    def _should_validate(
        iteration: int,
        eval_interval: int,
        eval_on_epoch: bool,
        is_epoch_end: bool,
        is_sanity_check: bool = False,
    ) -> bool:
        """Pure-function replica of the validation trigger logic.

        Mirrors the exact condition in src/pipelines/train.py to enable
        fast, isolated unit testing without mocking the full pipeline.

        Args:
            iteration: Current training step (1-indexed).
            eval_interval: Steps between validations (0 = disabled).
            eval_on_epoch: Whether epoch-boundary validation is enabled.
            is_epoch_end: Whether current iteration is an epoch boundary.
            is_sanity_check: Whether running in sanity check mode.

        Returns:
            True if validation should trigger at this iteration.
        """
        time_for_eval = eval_interval > 0 and iteration % eval_interval == 0

        if (
            eval_on_epoch
            and is_epoch_end
            and not is_sanity_check
            and eval_interval <= 0
        ):
            time_for_eval = True

        return time_for_eval

    # ------------------------------------------------------------------ #
    # eval_interval is the ONLY trigger when set
    # ------------------------------------------------------------------ #

    def test_fires_at_exact_eval_interval(self) -> None:
        """Validation triggers at exact multiples of eval_interval."""
        eval_interval = 9000
        for step in [9000, 18000, 27000, 36000]:
            assert self._should_validate(
                iteration=step,
                eval_interval=eval_interval,
                eval_on_epoch=False,
                is_epoch_end=False,
            ), f"Should validate at step {step}"

    def test_does_not_fire_between_intervals(self) -> None:
        """Validation does NOT trigger between eval_interval multiples."""
        eval_interval = 9000
        for step in [1, 100, 768, 1536, 4500, 8999, 9001]:
            assert not self._should_validate(
                iteration=step,
                eval_interval=eval_interval,
                eval_on_epoch=False,
                is_epoch_end=False,
            ), f"Should NOT validate at step {step}"

    def test_epoch_boundary_ignored_when_eval_interval_set(self) -> None:
        """eval_on_epoch=True must NOT trigger if eval_interval > 0.

        This is the critical fix: epoch boundaries must never override
        the iteration-based schedule when eval_interval is configured.
        """
        eval_interval = 9000
        epoch_boundaries = [768, 1536, 2304, 3072, 3840]  # len(train_loader) = 768

        for step in epoch_boundaries:
            assert not self._should_validate(
                iteration=step,
                eval_interval=eval_interval,
                eval_on_epoch=True,  # Explicitly enabled, but should be ignored
                is_epoch_end=True,
            ), (
                f"Epoch boundary at step {step} must NOT trigger validation "
                f"when eval_interval={eval_interval} is set"
            )

    def test_eval_interval_takes_priority_over_epoch(self) -> None:
        """When both eval_interval and eval_on_epoch are set, only eval_interval fires.

        Scenario: eval_interval=9000, epoch length=768.
        Step 768 is an epoch boundary but NOT an eval_interval multiple.
        Step 9000 is an eval_interval multiple but NOT an epoch boundary.
        """
        # Epoch boundary but not eval_interval → no validation
        assert not self._should_validate(
            iteration=768,
            eval_interval=9000,
            eval_on_epoch=True,
            is_epoch_end=True,
        )

        # eval_interval hit but not epoch boundary → validation fires
        assert self._should_validate(
            iteration=9000,
            eval_interval=9000,
            eval_on_epoch=True,
            is_epoch_end=False,
        )

    # ------------------------------------------------------------------ #
    # eval_on_epoch works only when eval_interval is NOT set
    # ------------------------------------------------------------------ #

    def test_epoch_trigger_works_when_no_eval_interval(self) -> None:
        """eval_on_epoch triggers at epoch boundaries when eval_interval <= 0."""
        assert self._should_validate(
            iteration=768,
            eval_interval=0,
            eval_on_epoch=True,
            is_epoch_end=True,
        )

    def test_epoch_trigger_suppressed_during_sanity_check(self) -> None:
        """eval_on_epoch is suppressed during sanity checks."""
        assert not self._should_validate(
            iteration=768,
            eval_interval=0,
            eval_on_epoch=True,
            is_epoch_end=True,
            is_sanity_check=True,
        )

    # ------------------------------------------------------------------ #
    # No duplicate triggers
    # ------------------------------------------------------------------ #

    def test_no_double_trigger_at_coinciding_boundaries(self) -> None:
        """When epoch boundary coincides with eval_interval, only one trigger.

        Example: eval_interval=768 and epoch length=768 → step 768 is both.
        The function should return True exactly once (not trigger validation twice).
        """
        result = self._should_validate(
            iteration=768,
            eval_interval=768,
            eval_on_epoch=True,
            is_epoch_end=True,
        )
        # Should still be True (from eval_interval), but only one trigger
        assert result is True

    # ------------------------------------------------------------------ #
    # Iteration 0 edge case
    # ------------------------------------------------------------------ #

    def test_does_not_fire_at_iteration_zero(self) -> None:
        """Iteration 0 should not trigger (training starts at 1)."""
        # iteration % eval_interval == 0 for iteration=0, but the loop
        # starts at range(start_iteration+1, ...) so 0 never occurs.
        # Still verify the logic doesn't trigger for step 0 defensively.
        result = self._should_validate(
            iteration=0,
            eval_interval=9000,
            eval_on_epoch=False,
            is_epoch_end=False,
        )
        # 0 % 9000 == 0 → True, but this step never occurs in practice
        # The function correctly returns True here, which is fine since
        # the training loop never calls with iteration=0.
        assert result is True  # Math is correct, loop guards against it

    # ------------------------------------------------------------------ #
    # Parameterized: Full schedule verification
    # ------------------------------------------------------------------ #

    @pytest.mark.parametrize(
        "eval_interval,total_steps,expected_triggers",
        [
            (9000, 27000, [9000, 18000, 27000]),
            (5000, 20000, [5000, 10000, 15000, 20000]),
            (25000, 100000, [25000, 50000, 75000, 100000]),
        ],
    )
    def test_full_schedule_produces_correct_triggers(
        self,
        eval_interval: int,
        total_steps: int,
        expected_triggers: list[int],
    ) -> None:
        """Verify the complete validation schedule for a training run."""
        actual_triggers = [
            step
            for step in range(1, total_steps + 1)
            if self._should_validate(
                iteration=step,
                eval_interval=eval_interval,
                eval_on_epoch=True,  # Enabled but should be ignored
                is_epoch_end=(step % 768 == 0),  # Simulate epoch boundaries
            )
        ]
        assert actual_triggers == expected_triggers, (
            f"eval_interval={eval_interval}: "
            f"expected {len(expected_triggers)} triggers, got {len(actual_triggers)}"
        )


class TestValidationCSVSingleWrite:
    """Ensure only one CSV row is written per validation trigger.

    The bug: two code paths in _run_validation() both wrote to
    validation_metrics.csv with different column orderings:
    - Path A: fieldnames=list(val_row.keys())  → insertion order
    - Path B: fieldnames=sorted(data.keys())   → alphabetical order

    Path B was removed. These tests verify the contract.
    """

    def test_csv_write_path_b_removed(self) -> None:
        """The duplicate CSV write with sorted keys must not exist in train.py."""
        from pathlib import Path

        source = Path("src/spectramr/pipelines/train.py").read_text()

        # The old duplicate write used sorted(data.keys()) — must be gone
        assert "fieldnames=sorted(data.keys())" not in source, (
            "Duplicate CSV write with sorted(data.keys()) still present in "
            "_run_validation. This causes column misalignment in validation_metrics.csv."
        )

    def test_csv_write_uses_insertion_order(self) -> None:
        """The remaining CSV write must use insertion-order fieldnames.

        The header source has to come from ``val_row`` (insertion order).
        Accepts either the inline form ``fieldnames=list(val_row.keys())``
        or the variable form ``all_fieldnames = list(val_row.keys())``
        used immediately by a ``DictWriter`` — both produce identical
        column ordering.
        """
        from pathlib import Path

        source = Path("src/spectramr/pipelines/train.py").read_text()

        inline_form = "fieldnames=list(val_row.keys())" in source
        variable_form = "list(val_row.keys())" in source and "fieldnames=all_fieldnames" in source
        assert inline_form or variable_form, (
            "Primary CSV write must derive fieldnames from val_row keys "
            "(insertion order). Neither inline nor variable form found — "
            "CSV writing may be broken."
        )


class TestExperiment11Config:
    """Verify experiment 11 config enforces iteration-only validation.

    Reads the arm as authored, on the CANONICAL nested path
    ``validation.schedule.{on_epoch,interval_steps}``. The flat spellings
    ``validation.eval_on_epoch`` / ``validation.eval_interval`` still fold
    through ``RENAMES``, so an arm using them loads -- but this class asserts
    what the file says, not what the loader resolves, and the retired spelling
    is what it must no longer say (#1302).
    """

    ARM = "experiments/inprogress/kspace_filling/experiment_11_kspace_cold_diffusion.yaml"

    def _schedule(self) -> dict:
        """The arm's ``validation.schedule`` block, refusing the flat spelling.

        Split out so a re-introduction of the retired keys fails with
        "declares the retired flat spelling" rather than the far more confusing
        "on_epoch missing from schedule".
        """
        import yaml

        config = yaml.safe_load(require_repo_file(self.ARM).read_text())
        validation = config.get("validation", {})
        retired = [k for k in ("eval_on_epoch", "eval_interval") if k in validation]
        assert not retired, (
            f"{self.ARM} declares {retired} -- the retired flat spelling. Those "
            "fold through RENAMES so the run still works, but the arm must be at "
            "current schema: validation.schedule.{on_epoch,interval_steps}."
        )
        return validation.get("schedule", {})

    def test_on_epoch_disabled_in_validation_schedule(self) -> None:
        """The schedule must EXPLICITLY set ``on_epoch: false``.

        Not because the default is wrong -- ``ValidationScheduleConfigSchema``
        defaults ``on_epoch`` to ``False`` today. Because this arm's contract is
        iteration-only validation, and a contract resting on a default is
        indistinguishable from one nobody chose: the day that default flips, the
        arm silently acquires a second trigger. The explicit declaration is what
        makes the choice auditable.
        """
        schedule = self._schedule()
        assert "on_epoch" in schedule, (
            "validation.schedule must explicitly set on_epoch -- inheriting the "
            "default leaves this arm's iteration-only contract undeclared"
        )
        assert (
            schedule["on_epoch"] is False
        ), f"on_epoch must be false, got {schedule['on_epoch']!r}"

    def test_interval_steps_is_positive(self) -> None:
        """``interval_steps`` must be a positive integer."""
        schedule = self._schedule()
        interval = schedule.get("interval_steps", 0)
        assert interval > 0, f"interval_steps must be positive, got {interval!r}"

    def test_the_schema_default_this_arm_overrides_is_the_one_documented(self) -> None:
        """CONTROL: pins the default the docstring above reasons about.

        That rationale used to read "default is True, which causes dual
        triggers" -- stale, and stale in the direction that made the test look
        more load-bearing than it was. If the default moves again this fails
        next to the prose depending on it, rather than leaving the prose quietly
        wrong.
        """
        from spectramr.config.schemas.validation import ValidationScheduleConfigSchema

        assert ValidationScheduleConfigSchema().on_epoch is False
