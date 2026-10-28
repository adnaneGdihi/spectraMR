"""The cascading-sweep SSOT and its tall record (issue #697).

The levels were declared twice -- `pipelines/training_loop.py` and
`infrastructure/training/strategies/diffusion.py` -- and agreed only by
coincidence. The strategy evaluated at one list while the CSV was labelled from
the other, so a divergence would have mislabelled every column silently. A
mislabelled number is worse than a missing one: it still reads as a number.

The record shape is pinned here rather than in a strategy test because it is
the contract between the strategy (which measures) and the pipeline (which
persists).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from spectramr.core.cascading_validation import (
    CASCADE_ID_COLUMNS,
    CASCADING_LEVELS,
    IDENTITY_ACCELERATION,
    UNRECORDED_SKIP_REASON,
    aggregate_cascade_rows,
    build_cascade_row,
    check_round_trip,
    reconcile_skipped_levels,
)

# ── the SSOT itself ──────────────────────────────────────────────────────────


def test_levels_are_immutable() -> None:
    """A tuple, not a list. A shared mutable would let one consumer's
    `.append()` silently change what the other evaluates."""
    assert isinstance(CASCADING_LEVELS, tuple)
    with pytest.raises((AttributeError, TypeError)):
        CASCADING_LEVELS.append(64)  # type: ignore[attr-defined]


def test_levels_are_the_documented_sweep() -> None:
    assert CASCADING_LEVELS == (2, 8, 32)


def test_levels_are_ascending_and_unique() -> None:
    """Ascending matters: `_apply_input_dependence_gate` is indexed positionally
    against this list, so a reorder would compare the wrong severities."""
    assert list(CASCADING_LEVELS) == sorted(set(CASCADING_LEVELS))


def test_only_one_module_defines_the_sweep() -> None:
    """Anti-regression for the defect this SSOT exists to remove.

    Reads the two former sites for a literal re-declaration. A grep-shaped test,
    because the failure mode IS a second literal -- there is no object to
    introspect once someone writes `[2, 8, 32]` again.
    """
    import pathlib

    repo = pathlib.Path(__file__).resolve().parents[3]
    offenders = []
    for rel in (
        "src/spectramr/pipelines/training_loop.py",
        "src/spectramr/infrastructure/training/strategies/diffusion.py",
    ):
        text = (repo / rel).read_text()
        for lit in ("[2, 8, 32]", "[2,8,32]"):
            if lit in text:
                offenders.append(f"{rel} re-declares {lit}")
    assert not offenders, offenders


# ── the tall record ──────────────────────────────────────────────────────────


def test_level_and_timestep_are_values_not_name_parts() -> None:
    row = build_cascade_row(
        acceleration_level=8, heldout=False, timestep=200, metrics={"val_psnr": 31.0}
    )
    assert row["acceleration_level"] == 8.0
    assert row["timestep"] == 200.0
    assert "val_psnr" in row
    assert "val_psnr_8x" not in row


def test_metric_names_pass_through_unfiltered() -> None:
    """No name list is consulted, which is what stops the drift.

    The retired hardcoded 15-name list did not contain `val_hfen`, so a drained
    `kspace_filling` arm emitting it had nowhere to put the value.
    """
    metrics = {"val_hfen": 0.4, "val_kspace_error": 0.1, "anything_at_all": 7}
    row = build_cascade_row(
        acceleration_level=2, heldout=False, timestep=1, metrics=metrics
    )
    assert metrics.items() <= row.items()


def test_heldout_is_a_boolean_column() -> None:
    """It used to be a third naming convention (`_heldout_{R}x`) that appeared
    in no column list at all."""
    row = build_cascade_row(
        acceleration_level=64, heldout=True, timestep=900, metrics={}
    )
    assert row["heldout"] is True
    assert row["acceleration_level"] == 64.0


def test_identity_columns_win_a_name_collision() -> None:
    """A metric literally named `timestep` must not overwrite the severity
    point the row is about -- that would be a wrong number, not a missing one."""
    row = build_cascade_row(
        acceleration_level=8,
        heldout=False,
        timestep=200,
        metrics={"timestep": -1.0, "acceleration_level": -1.0, "heldout": "nonsense"},
    )
    assert row["timestep"] == 200.0
    assert row["acceleration_level"] == 8.0
    assert row["heldout"] is False


def test_absent_timestep_is_none_not_zero() -> None:
    """0 is a legal timestep, so the absence of one must not read as t=0."""
    row = build_cascade_row(
        acceleration_level=2, heldout=False, timestep=None, metrics={}
    )
    assert row["timestep"] is None


def test_a_bare_row_declares_exactly_the_non_writer_id_columns() -> None:
    """`iteration`/`epoch` are stamped by the writer, which knows the step; the
    strategy that builds the row does not."""
    row = build_cascade_row(
        acceleration_level=2, heldout=False, timestep=100, metrics={}
    )
    assert set(row) == set(CASCADE_ID_COLUMNS) - {"iteration", "epoch"}


# ── aggregation across validation batches ────────────────────────────────────
#
# `_run_validation` calls `validation_step` ONCE PER VAL BATCH (train.py:1768),
# and the cascade is not gated on `batch_idx` -- it runs every batch. The
# suffixed `val_*_<R>x` keys are accumulated (`val_accum[k] += v`) and divided
# by `val_count`, so those columns are a mean over every batch. A sidecar that
# published one batch would silently disagree with the column beside it, which
# is the same "a number that is not what its label says" failure the tall
# record exists to prevent.


def test_repeated_severity_points_collapse_to_their_mean() -> None:
    """Three batches at R=8 must publish ONE row holding the mean, matching how
    `val_psnr_8x` is computed -- not the last batch's value."""
    rows = [
        build_cascade_row(
            acceleration_level=8, heldout=False, timestep=200, metrics={"val_psnr": v}
        )
        for v in (30.0, 32.0, 34.0)
    ]
    (out,) = aggregate_cascade_rows(rows)
    assert out["val_psnr"] == pytest.approx(32.0)
    assert out["acceleration_level"] == 8.0


def test_aggregation_records_how_many_batches_it_averaged() -> None:
    """Provenance as DATA: a reader can tell a 1-batch row from a 40-batch one
    without reading the pipeline."""
    rows = [
        build_cascade_row(
            acceleration_level=2, heldout=False, timestep=10, metrics={"val_psnr": 1.0}
        )
        for _ in range(4)
    ]
    (out,) = aggregate_cascade_rows(rows)
    assert out["n_batches"] == 4


def test_each_severity_point_stays_its_own_row() -> None:
    """Aggregation is per (level, heldout) -- it must not collapse the sweep."""
    rows = [
        build_cascade_row(
            acceleration_level=lvl,
            heldout=False,
            timestep=lvl,
            metrics={"val_psnr": lvl},
        )
        for lvl in CASCADING_LEVELS
        for _ in range(2)
    ]
    out = aggregate_cascade_rows(rows)
    assert [r["acceleration_level"] for r in out] == [
        float(x) for x in CASCADING_LEVELS
    ]


def test_heldout_does_not_merge_into_the_in_distribution_point() -> None:
    """R=8 held-out and R=8 in-distribution are different measurements; merging
    them would average a robustness probe into the training regime."""
    rows = [
        build_cascade_row(
            acceleration_level=8,
            heldout=False,
            timestep=200,
            metrics={"val_psnr": 30.0},
        ),
        build_cascade_row(
            acceleration_level=8, heldout=True, timestep=200, metrics={"val_psnr": 10.0}
        ),
    ]
    out = aggregate_cascade_rows(rows)
    assert len(out) == 2
    assert {r["heldout"] for r in out} == {True, False}


def test_non_numeric_metrics_survive_without_being_averaged() -> None:
    """A string-valued diagnostic must not crash the mean, nor be silently
    dropped -- dropping is how the retired implementation lost 45 columns."""
    rows = [
        build_cascade_row(
            acceleration_level=2,
            heldout=False,
            timestep=10,
            metrics={"val_psnr": 30.0, "note": "ok"},
        ),
        build_cascade_row(
            acceleration_level=2,
            heldout=False,
            timestep=10,
            metrics={"val_psnr": 32.0, "note": "ok"},
        ),
    ]
    (out,) = aggregate_cascade_rows(rows)
    assert out["val_psnr"] == pytest.approx(31.0)
    assert out["note"] == "ok"


def test_a_metric_only_a_later_batch_computed_still_gets_a_column() -> None:
    """The group's key set is a UNION, not the first row's keys.

    Iterating `group[0]` would drop anything the first batch happened not to
    emit -- the same silent-drop class as the retired 45-name list, one level up.
    `n_batches` stays the group size, so a sparse column's mean is over the
    batches that HAD it while `n_batches` says how many the sweep saw.
    """
    rows = [
        build_cascade_row(
            acceleration_level=8,
            heldout=False,
            timestep=200,
            metrics={"val_psnr": 30.0},
        ),
        build_cascade_row(
            acceleration_level=8,
            heldout=False,
            timestep=200,
            metrics={"val_psnr": 32.0, "val_hfen": 0.4},
        ),
    ]
    (out,) = aggregate_cascade_rows(rows)
    assert out["val_hfen"] == 0.4
    assert out["val_psnr"] == 31.0
    assert out["n_batches"] == 2


def test_aggregating_nothing_yields_nothing() -> None:
    assert aggregate_cascade_rows([]) == []


def test_every_row_of_a_sweep_has_identical_keys() -> None:
    """The tall writer's schema-evolution path rewrites the file (and backs it
    up) on any new key. Rows of one sweep must not trigger that per row."""
    rows = [
        build_cascade_row(
            acceleration_level=lvl,
            heldout=False,
            timestep=lvl * 10,
            metrics={"val_psnr": 30.0, "val_ssim": 0.9},
        )
        for lvl in CASCADING_LEVELS
    ]
    assert len({frozenset(r) for r in rows}) == 1


# ── skip-reason reconciliation ───────────────────────────────────────────────


class TestReconcileSkippedLevels:
    """A lost rung must never be reported without a reason.

    The completeness VERDICT is already double-guarded in the strategy (a
    recorded skip OR a short evaluated list). These pin the other half: that the
    REASON survives a `continue` nobody taught to record itself, which is the
    shape #1295's two new exits arrive in.
    """

    def test_a_complete_cascade_reconciles_to_nothing(self) -> None:
        """Negative control, and the case that runs on every healthy validation:
        the helper must not invent a skip where no rung was lost."""
        assert reconcile_skipped_levels(CASCADING_LEVELS, list(CASCADING_LEVELS), {}) == {}

    def test_a_recorded_reason_is_preserved_verbatim(self) -> None:
        """Derivation is a backstop, not an override. An exit that DID explain
        itself must keep its own words -- replacing a real diagnosis with
        `skipped-without-recorded-reason` would lose information, which is the
        opposite of the point."""
        evaluated = [lvl for lvl in CASCADING_LEVELS if lvl != 8]
        out = reconcile_skipped_levels(
            CASCADING_LEVELS, evaluated, {8: "not-in-acceleration_range"}
        )
        assert out == {8: "not-in-acceleration_range"}

    def test_a_rung_lost_without_a_reason_is_named_anyway(self) -> None:
        """The regression this exists for. A `continue` that records nothing --
        #1295's round-trip and identity-mask exits -- previously left the warning
        printing `skipped {}` beside a short evaluated list."""
        evaluated = [lvl for lvl in CASCADING_LEVELS if lvl != 8]
        assert reconcile_skipped_levels(CASCADING_LEVELS, evaluated, {}) == {
            8: UNRECORDED_SKIP_REASON
        }

    def test_recorded_and_unrecorded_losses_coexist(self) -> None:
        """Two rungs lost different ways in one sweep: one explained, one not.
        Both must appear, and the explained one must not be overwritten."""
        out = reconcile_skipped_levels(
            CASCADING_LEVELS, [2], {8: "not-in-acceleration_range"}
        )
        assert out == {8: "not-in-acceleration_range", 32: UNRECORDED_SKIP_REASON}

    def test_every_declared_rung_is_accounted_for(self) -> None:
        """The invariant the strategy's warning depends on: after reconciliation
        each in-distribution rung is either evaluated or explained -- never
        neither -- for any partition of the ladder."""
        for cut in range(len(CASCADING_LEVELS) + 1):
            evaluated = list(CASCADING_LEVELS[:cut])
            out = reconcile_skipped_levels(CASCADING_LEVELS, evaluated, {})
            assert set(evaluated) | set(out) == set(CASCADING_LEVELS)
            assert not set(evaluated) & set(out)

    def test_the_input_mapping_is_not_mutated(self) -> None:
        """The call site rebinds the result; a helper that also mutated its
        argument would make the two paths silently disagree if that ever
        changed."""
        original: dict[int, str] = {}
        reconcile_skipped_levels(CASCADING_LEVELS, [], original)
        assert original == {}

    def test_held_out_points_are_not_the_caller_s_ladder(self) -> None:
        """Held-out severities are a robustness readout, not part of the cascade
        contract. Passing one in would report it as a lost rung -- so the
        contract is that `levels` is the in-distribution ladder, and this pins
        what happens if that is violated rather than leaving it to be discovered
        in a log."""
        out = reconcile_skipped_levels([*CASCADING_LEVELS, 64], list(CASCADING_LEVELS), {})
        assert out == {64: UNRECORDED_SKIP_REASON}


# ── declared vs realised (#1295) ─────────────────────────────────────────────
#
# The cascade asks an inverse for "the timestep whose mask realises R=X", then
# labels a whole row `acceleration_level=X`. Nothing made the inverse prove it,
# and two mechanisms quietly moved the measurement off the label: a `[T/10,
# 9T/10]` band clamp the caller applied to the inverse's answer, and continuous
# schedules whose inverse saturates instead of refusing an out-of-domain R.


def _linear_forward(*, base: float, top: float, num_timesteps: int):
    """`get_acceleration_factor` for a linear schedule, as a plain closure.

    A closure rather than a real accelerator on purpose: `check_round_trip` is
    pure, and pinning it against a fake keeps this file free of torch and of
    the accelerator's own construction rules.
    """

    def forward(t: int) -> float:
        t = max(0, min(num_timesteps - 1, int(t)))
        return base + (top - base) * (t / (num_timesteps - 1))

    return forward


def test_the_exact_timestep_round_trips() -> None:
    fwd = _linear_forward(base=1.0, top=8.0, num_timesteps=1000)
    rt = check_round_trip(requested=8.0, timestep=999, forward=fwd, num_timesteps=1000)
    assert rt.ok
    assert rt.realized == pytest.approx(8.0)


def test_quantisation_alone_does_not_fail_the_check() -> None:
    """t=143 realises 2.0020, not 2.0. That residual is the schedule's own
    resolution and must not be reported as a mislabelling."""
    fwd = _linear_forward(base=1.0, top=8.0, num_timesteps=1000)
    rt = check_round_trip(requested=2.0, timestep=143, forward=fwd, num_timesteps=1000)
    assert rt.ok
    assert rt.realized != pytest.approx(2.0, abs=1e-9)


def test_tolerance_is_the_schedule_resolution_not_a_fixed_percentage() -> None:
    """The reason a relative tolerance would be wrong.

    One timestep moves R by 0.007 on a T=1000 arm and by ~1.1 on a T=29 one.
    A 2% rule would reject the coarse arm's unavoidable quantisation; a rule
    loose enough for the coarse arm would wave through real drift on the fine
    one. Half the local step is the only bound that is right for both.
    """
    fine = check_round_trip(
        requested=2.0,
        timestep=143,
        forward=_linear_forward(base=1.0, top=8.0, num_timesteps=1000),
        num_timesteps=1000,
    )
    coarse = check_round_trip(
        requested=2.0,
        timestep=1,
        forward=_linear_forward(base=1.0, top=32.0, num_timesteps=29),
        num_timesteps=29,
    )
    assert fine.ok and coarse.ok
    # Three orders of magnitude apart -- a single constant cannot serve both.
    assert coarse.tolerance > 100 * fine.tolerance


def test_an_out_of_domain_request_is_refused_not_saturated() -> None:
    """A continuous inverse clamps `progress` into [0, 1], so R=64 on an arm
    capped at R=8 returns t=T-1 and decodes back to 8.0. Verified against the
    real accelerator: linear/power_law/exponential all do this."""
    fwd = _linear_forward(base=1.0, top=8.0, num_timesteps=1000)
    rt = check_round_trip(requested=64.0, timestep=999, forward=fwd, num_timesteps=1000)
    assert not rt.ok
    assert rt.realized == pytest.approx(8.0)


def test_a_clamped_timestep_is_caught_on_a_step_schedule() -> None:
    """The #1295 witness in miniature: the answer was t=28, the caller used
    t=27, and bucket 27 is a different rung.

    A residual bound cannot do this. The rungs here are 2.226 apart at the top,
    so half a local step is wider than the whole error -- which is exactly why
    the check also requires local optimality.
    """
    ladder = [1.0, 2.0, 2.226, 29.444, 32.0]

    def forward(t: int) -> float:
        t = max(0, min(4, int(t)))
        return ladder[t]

    moved = check_round_trip(requested=32.0, timestep=3, forward=forward, num_timesteps=5)
    assert not moved.ok
    assert not moved.locally_optimal
    assert moved.realized == pytest.approx(29.444)
    # The residual half of the check would have PASSED this one.
    assert abs(moved.realized - 32.0) <= moved.tolerance

    kept = check_round_trip(requested=32.0, timestep=4, forward=forward, num_timesteps=5)
    assert kept.ok
    assert kept.reason is None


def test_the_two_conditions_catch_different_things() -> None:
    """Neither half is redundant, and the reason says which one fired."""
    ladder = [1.0, 2.0, 2.226, 29.444, 32.0]

    def stepped(t: int) -> float:
        return ladder[max(0, min(4, int(t)))]

    repointed = check_round_trip(
        requested=2.0, timestep=2, forward=stepped, num_timesteps=5
    )
    assert not repointed.locally_optimal
    assert "moved it" in (repointed.reason or "")

    saturated = check_round_trip(
        requested=64.0,
        timestep=999,
        forward=_linear_forward(base=1.0, top=8.0, num_timesteps=1000),
        num_timesteps=1000,
    )
    # Saturation is locally optimal -- t=999 really is the closest available --
    # so only the residual bound catches it.
    assert saturated.locally_optimal
    assert not saturated.ok
    assert "outside what this schedule can realise" in (saturated.reason or "")


def test_a_single_timestep_horizon_demands_an_exact_match() -> None:
    """T=1 leaves no neighbours, so there is no resolution to spend and the
    tolerance must not silently widen to accept anything."""
    rt = check_round_trip(
        requested=4.0, timestep=0, forward=lambda _t: 8.0, num_timesteps=1
    )
    assert not rt.ok
    assert rt.tolerance == pytest.approx(1e-6)


def test_identity_acceleration_is_the_fully_sampled_factor() -> None:
    """R=1 is no undersampling at all. A rung there measures an identity mask,
    so its PSNR is a property of the data pipeline, not of the model."""
    assert IDENTITY_ACCELERATION == 1.0


def test_the_row_carries_what_was_measured_beside_what_was_asked() -> None:
    row = build_cascade_row(
        acceleration_level=32.0,
        heldout=False,
        timestep=27,
        metrics={"val_psnr": 24.0},
        acceleration_realized=29.444,
    )
    assert row["acceleration_level"] == 32.0
    assert row["acceleration_realized"] == pytest.approx(29.444)


def test_an_undecodable_level_records_none_not_the_request() -> None:
    """The linear-fallback path has no schedule to decode with. Restating the
    request there would turn an unknown into a false confirmation -- exactly
    the mislabelling this column exists to expose."""
    row = build_cascade_row(
        acceleration_level=8.0, heldout=False, timestep=200, metrics={}
    )
    assert row["acceleration_realized"] is None


def test_realised_acceleration_survives_aggregation() -> None:
    """It is numeric, so the mean is meaningful and it must not be dropped the
    way the retired 45 columns were."""
    rows = [
        build_cascade_row(
            acceleration_level=2.0,
            heldout=False,
            timestep=2,
            metrics={"val_psnr": 30.0},
            acceleration_realized=2.226,
        )
        for _ in range(3)
    ]
    (merged,) = aggregate_cascade_rows(rows)
    assert merged["acceleration_realized"] == pytest.approx(2.226)


class TestResolveCascadeLevels:
    """`validation.cascade.levels` is the declared ladder; the constant is the default.

    Before #1394 the ladder was a module constant read directly by the strategy
    AND copied into `config_health_checker`. These pin the single resolver both
    now read, so the copy cannot come back without a red test.
    """

    def test_nothing_declared_yields_the_framework_default(self):
        from spectramr.core.cascading_validation import resolve_cascade_levels

        assert resolve_cascade_levels(None) == CASCADING_LEVELS
        assert resolve_cascade_levels(SimpleNamespace()) == CASCADING_LEVELS
        assert (
            resolve_cascade_levels(SimpleNamespace(cascade=SimpleNamespace(levels=None)))
            == CASCADING_LEVELS
        )

    def test_a_declared_ladder_replaces_the_default(self):
        from spectramr.core.cascading_validation import resolve_cascade_levels

        cfg = SimpleNamespace(cascade=SimpleNamespace(levels=[2, 4, 6]))
        assert resolve_cascade_levels(cfg) == (2, 4, 6)

    def test_a_mapping_shaped_config_resolves_identically(self):
        """Duck-typed on purpose so `core/` needs no import from `config/`."""
        from spectramr.core.cascading_validation import resolve_cascade_levels

        assert resolve_cascade_levels({"cascade": {"levels": [3, 9]}}) == (3, 9)

    def test_integral_levels_stay_int_so_flat_column_names_do_not_move(self):
        """`f"_{accel}x"` renders 2 as `_2x` and 2.0 as `_2.0x`.

        The L4 input-dependence gate and `_stamp_accel_psnr_gap` look those
        names up in `all_metrics` and do not raise on a miss — a float here
        would silently disconnect both gates while every run stayed green.
        """
        from spectramr.core.cascading_validation import normalize_cascade_levels

        levels = normalize_cascade_levels([2.0, 8.0, 32.0])
        assert [f"val_psnr_{a}x" for a in levels] == [
            "val_psnr_2x",
            "val_psnr_8x",
            "val_psnr_32x",
        ]
        assert all(isinstance(a, int) for a in levels)

    def test_a_fractional_level_survives_as_a_float(self):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        assert normalize_cascade_levels([2, 4.5]) == (2, 4.5)

    def test_levels_are_deduped_and_sorted_ascending(self):
        """The accel-gap readout subtracts the last rung from the first.

        "First rung is the mildest" is an invariant of the consumers, not
        something a user's typing order should decide.
        """
        from spectramr.core.cascading_validation import normalize_cascade_levels

        assert normalize_cascade_levels([32, 2, 8, 2]) == (2, 8, 32)

    def test_an_empty_ladder_is_refused_not_defaulted(self):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="empty"):
            normalize_cascade_levels([])

    def test_a_sub_1x_level_is_refused(self):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="below 1x"):
            normalize_cascade_levels([2, 0.5])

    @pytest.mark.parametrize("bad", [["8"], ["abc"], [None], [object()]])
    def test_a_non_numeric_level_is_refused(self, bad):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="not a number"):
            normalize_cascade_levels(bad)

    def test_a_boolean_level_is_refused(self):
        """`bool` is an `int` subclass, so `True` would pass as R=1 unchecked."""
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="not a number"):
            normalize_cascade_levels([True])

    def test_a_non_finite_level_is_refused(self):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="not finite"):
            normalize_cascade_levels([float("inf")])

    @pytest.mark.parametrize("bad", [8, "2,8,32", None])
    def test_a_non_sequence_ladder_is_refused(self, bad):
        from spectramr.core.cascading_validation import normalize_cascade_levels

        with pytest.raises(ValueError, match="must be a sequence"):
            normalize_cascade_levels(bad)


class TestFlattenBandRecords:
    """Per-reveal-band attribution, summarised into cascade-row columns (#2067).

    Planted violations first: each assertion below fails against one specific
    way the summary could lie about what was measured.
    """

    @staticmethod
    def _records():
        nan = float("nan")
        return [
            # An OOD-written band: shrunk and rotated.
            {"step": 0, "timestep": 7, "n_bins": 32, "gain_modulus": 0.74, "gain_phase_rad": 0.30},
            # An inert step -- the loop revealed nothing and skipped it.
            {"step": 1, "timestep": 3, "n_bins": 0, "gain_modulus": nan, "gain_phase_rad": nan},
            # The decisive band, written at a trained timestep.
            {"step": 2, "timestep": 1, "n_bins": 64, "gain_modulus": 0.93, "gain_phase_rad": -0.10},
            # The observed support, pinned to the measurement by hard DC.
            {"step": -1, "timestep": -1, "n_bins": 16, "gain_modulus": 1.0, "gain_phase_rad": 0.0},
        ]

    def test_inert_bands_are_excluded_from_every_aggregate(self):
        """PLANTED: a nan band. Averaging it in would report an unmeasured step."""
        from spectramr.core.cascading_validation import flatten_band_records

        out = flatten_band_records(self._records())
        assert out["val_band_count"] == 2.0
        assert out["val_band_gain_modulus_mean"] == pytest.approx((0.74 + 0.93) / 2)

    def test_the_observed_support_is_reported_separately_not_pooled(self):
        """PLANTED: a perfect observed gain that would flatter the inferred mean."""
        from spectramr.core.cascading_validation import flatten_band_records

        out = flatten_band_records(self._records())
        assert out["val_band_observed_gain_modulus"] == 1.0
        assert out["val_band_gain_modulus_min"] == 0.74
        assert out["val_band_gain_modulus_mean"] < 1.0, "the hard-DC floor must not pool in"

    def test_the_worst_band_names_its_timestep(self):
        from spectramr.core.cascading_validation import flatten_band_records

        assert flatten_band_records(self._records())["val_band_worst_timestep"] == 7.0

    def test_phase_is_summarised_by_absolute_value(self):
        """A +0.30 and a -0.10 band must not cancel to a healthy-looking mean."""
        from spectramr.core.cascading_validation import flatten_band_records

        out = flatten_band_records(self._records())
        assert out["val_band_phase_rad_absmax"] == pytest.approx(0.30)
        assert out["val_band_phase_rad_absmean"] == pytest.approx(0.20)

    def test_no_measurable_band_reports_none_not_zero(self):
        """PLANTED: every band inert. Zero would read as a total loss of signal."""
        from spectramr.core.cascading_validation import flatten_band_records

        nan = float("nan")
        out = flatten_band_records(
            [{"step": 0, "timestep": 1, "n_bins": 0, "gain_modulus": nan, "gain_phase_rad": nan}]
        )
        assert out["val_band_count"] == 0.0
        assert out["val_band_gain_modulus_min"] is None
        assert out["val_band_worst_timestep"] is None
        assert out["val_band_observed_gain_modulus"] is None


# ---------------------------------------------------------------------------
# An unmeasurable band among measurable ones.
#
# `min` seeds on element 0 and every `k < nan` is False, so selecting the worst
# band from the UNFILTERED list returned a nan record whenever it came first --
# while `val_band_gain_modulus_min` came from the filtered one. The two columns
# then named different reverse steps, and the one labelled "worst" was the band
# that was never measured.
#
# The tests already here plant nan only with `n_bins: 0`, which is exactly what
# `inferred` excludes, so none of them could see it (non-negotiable 15).
# ---------------------------------------------------------------------------


def _band(step, timestep, modulus, *, n_bins=4):
    return {
        "step": step,
        "timestep": timestep,
        "n_bins": n_bins,
        "gain_modulus": modulus,
        "gain_phase_rad": 0.1 if modulus == modulus else float("nan"),
    }


@pytest.mark.unit
def test_worst_band_and_min_modulus_name_the_same_step():
    """A band with n_bins > 0 and zero target energy yields nan and survives the
    n_bins filter -- reachable at high |k| on a cropped acquisition."""
    from spectramr.core.cascading_validation import flatten_band_records

    nan = float("nan")
    out = flatten_band_records([_band(0, 9, nan), _band(1, 5, 0.31), _band(2, 1, 0.88)])

    assert out["val_band_gain_modulus_min"] == pytest.approx(0.31)
    assert out["val_band_worst_timestep"] == pytest.approx(5.0), (
        "the worst-band column named the UNMEASURED band while the minimum came "
        "from a measured one"
    )


@pytest.mark.unit
def test_a_trailing_unmeasurable_band_is_also_ignored():
    """The non-leading case worked by accident before; it must still hold."""
    from spectramr.core.cascading_validation import flatten_band_records

    nan = float("nan")
    out = flatten_band_records([_band(0, 1, 0.88), _band(1, 9, nan), _band(2, 5, 0.31)])

    assert out["val_band_gain_modulus_min"] == pytest.approx(0.31)
    assert out["val_band_worst_timestep"] == pytest.approx(5.0)


@pytest.mark.unit
def test_all_bands_unmeasurable_reports_nothing_not_a_gain():
    """An unmeasured quantity is reported as unmeasured, never as a gain."""
    from spectramr.core.cascading_validation import flatten_band_records

    nan = float("nan")
    out = flatten_band_records([_band(0, 9, nan), _band(1, 5, nan)])

    assert out["val_band_gain_modulus_min"] is None
    assert out["val_band_worst_timestep"] is None
    assert out["val_band_count"] == 2.0, "the bands existed; only their gain did not"
