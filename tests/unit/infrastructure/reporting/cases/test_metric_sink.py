"""Tests for ``PerCallMetricSink``.

Targets ``spectramr.infrastructure.reporting.cases.metric_sink``: enable gate,
scalar-only filtering, stable column order, memory cap, and CSV round-trip.
"""

from __future__ import annotations

import pandas as pd

from spectramr.infrastructure.reporting.cases.metric_sink import PerCallMetricSink


def test_disabled_sink_is_noop(tmp_path):
    sink = PerCallMetricSink(enabled=False)
    sink.observe(case_id="c0", metrics={"psnr": 30.0})
    assert sink.n_rows == 0
    assert sink.write(tmp_path) is None
    assert not (tmp_path / "per_call_metrics.csv").exists()


def test_observe_keeps_only_finite_scalars(tmp_path):
    sink = PerCallMetricSink(enabled=True)
    sink.observe(case_id="c0", metrics={"psnr": 30.0, "ok": True, "name": "x",
                                        "ssim": 0.9}, step=5)
    out = sink.write(tmp_path)
    assert out is not None and out.exists()
    df = pd.read_csv(out)
    assert set(df.columns) >= {"case_id", "split", "step", "psnr", "ssim"}
    # bool + str metrics are dropped
    assert "ok" not in df.columns
    assert "name" not in df.columns
    assert df.loc[0, "psnr"] == 30.0
    assert df.loc[0, "step"] == 5


def test_column_order_is_stable(tmp_path):
    sink = PerCallMetricSink(enabled=True)
    sink.observe(case_id="c0", metrics={"ssim": 0.9, "psnr": 30.0}, step=1)
    df = pd.read_csv(sink.write(tmp_path))
    # lead cols first, metrics sorted alphabetically after
    assert list(df.columns) == ["case_id", "split", "step", "psnr", "ssim"]


def test_multiple_observations_one_row_each(tmp_path):
    sink = PerCallMetricSink(enabled=True)
    for i in range(4):
        sink.observe(case_id=f"val_step{i}", metrics={"psnr": 20.0 + i}, step=i)
    assert sink.n_rows == 4
    df = pd.read_csv(sink.write(tmp_path))
    assert len(df) == 4
    assert list(df["psnr"]) == [20.0, 21.0, 22.0, 23.0]


def test_memory_cap_evicts_oldest(tmp_path):
    sink = PerCallMetricSink(enabled=True, max_rows=3)
    for i in range(5):
        sink.observe(case_id=f"c{i}", metrics={"psnr": float(i)}, step=i)
    assert sink.n_rows == 3
    df = pd.read_csv(sink.write(tmp_path))
    # oldest two evicted → keeps c2, c3, c4
    assert list(df["case_id"]) == ["c2", "c3", "c4"]


def test_empty_sink_writes_nothing(tmp_path):
    sink = PerCallMetricSink(enabled=True)
    assert sink.write(tmp_path) is None


# ── the context block: identity as columns, not as substructure in case_id ───


def test_context_becomes_columns_between_step_and_the_metrics(tmp_path):
    """The rung and the sample are DATA, not a substring of ``case_id``.

    Cascading validation feeds this sink once per (batch, rung), so a 45-batch
    x 3-rung sweep wrote 135 rows sharing three ``case_id`` values -- every
    number present and nothing saying which volume produced any of them.
    """
    sink = PerCallMetricSink(enabled=True)
    sink.observe(
        case_id="val_step1500_R8x",
        metrics={"val_psnr": 18.9},
        step=1500,
        context={
            "acceleration_level": 8.0,
            "timestep": 27,
            "heldout": False,
            "contrast": "FLAIR",
            "file_id": "2022080410_FLAIR01",
        },
    )
    frame = pd.read_csv(sink.write(tmp_path))
    assert list(frame.columns) == [
        "case_id",
        "split",
        "step",
        "acceleration_level",
        "timestep",
        "heldout",
        "contrast",
        "file_id",
        "val_psnr",
    ]
    assert frame.loc[0, "contrast"] == "FLAIR"
    assert frame.loc[0, "acceleration_level"] == 8.0


def test_context_values_are_not_coerced_to_float(tmp_path):
    """Metrics are floated; context is not.

    ``float("FLAIR")`` raises and ``float(False)`` is ``0.0`` -- coercing the
    context would drop the contrast entirely and render the held-out flag as a
    number indistinguishable from a metric.
    """
    sink = PerCallMetricSink(enabled=True)
    sink.observe(
        case_id="c0",
        metrics={"psnr": 30.0},
        context={"contrast": "T1", "heldout": True},
    )
    row = sink._rows[0]
    assert row["contrast"] == "T1"
    assert row["heldout"] is True


def test_a_none_context_value_is_omitted_not_zeroed(tmp_path):
    """A timestep that could not be resolved must read blank, not ``t=0``.

    0 is a real timestep -- the fully-clean end of the diffusion schedule -- so
    substituting it for "unknown" produces a plausible wrong answer rather than
    a visible gap (non-negotiable 3).
    """
    sink = PerCallMetricSink(enabled=True)
    sink.observe(case_id="c0", metrics={"psnr": 30.0}, context={"timestep": None})
    assert "timestep" not in sink._rows[0]


def test_a_context_key_colliding_with_a_metric_raises():
    """One would overwrite the other and the CSV would look correct.

    A number under a name that means something else is unreadable *and*
    undetectable on read -- so this fires at the call site, on the first row.
    """
    import pytest

    sink = PerCallMetricSink(enabled=True)
    with pytest.raises(ValueError, match="collides"):
        sink.observe(
            case_id="c0", metrics={"timestep": 5.0}, context={"timestep": 27}
        )


def test_a_context_key_colliding_with_an_identity_column_raises():
    import pytest

    sink = PerCallMetricSink(enabled=True)
    with pytest.raises(ValueError, match="collides"):
        sink.observe(case_id="c0", metrics={"psnr": 1.0}, context={"step": 3})


def test_undeclared_context_keys_land_after_the_declared_ones(tmp_path):
    """Declared order is stable; an unknown identity is written, not refused.

    This component does not own the vocabulary of what a dataset can publish,
    so an unrecognised key is appended rather than dropped -- but it never
    reshuffles the leading columns between runs.
    """
    sink = PerCallMetricSink(enabled=True)
    sink.observe(
        case_id="c0",
        metrics={"psnr": 30.0},
        context={"zeta": "z", "contrast": "T1", "alpha": "a"},
    )
    frame = pd.read_csv(sink.write(tmp_path))
    assert list(frame.columns) == ["case_id", "split", "contrast", "alpha", "zeta", "psnr"]


def test_rows_without_context_still_write(tmp_path):
    """A strategy that feeds no context is unaffected -- columns simply absent."""
    sink = PerCallMetricSink(enabled=True)
    sink.observe(case_id="c0", metrics={"psnr": 30.0}, step=7)
    frame = pd.read_csv(sink.write(tmp_path))
    assert list(frame.columns) == ["case_id", "split", "step", "psnr"]


def test_every_declared_context_column_is_actually_emitted():
    """A column this module DECLARES must be one a producer actually sets.

    ``CONTEXT_COLUMNS`` is a promise about what ``per_call_metrics.csv`` can
    contain, and the docs render it as a table of columns the reader may filter
    on. Declaring a name costs nothing and emits nothing, so the two halves
    drift silently: ``acceleration_realized`` shipped in the declaration and in
    the docs while no producer ever set it, which is pitfall #16 (a capability
    is not delivered until the production path calls it) wearing a column name.

    Anchored at the PRODUCERS, not at the repo: every one of these names also
    occurs elsewhere in ``src/`` as an ordinary identifier -- ``build_cascade_row``
    takes ``acceleration_realized`` as a keyword -- so a tree-wide grep scores
    the missing column GREEN. The check is only a check where the dict is built.
    """
    import inspect

    from spectramr.infrastructure.reporting.cases.metric_sink import CONTEXT_COLUMNS
    from spectramr.infrastructure.training.strategies.diffusion import (
        DiffusionTrainingStrategy,
    )
    from spectramr.infrastructure.training.strategies.mixins.metrics_mixin import (
        summarize_batch_identity,
    )

    producers = inspect.getsource(
        DiffusionTrainingStrategy._compute_validation_metrics
    ) + inspect.getsource(summarize_batch_identity)

    never_emitted = [name for name in CONTEXT_COLUMNS if f'"{name}"' not in producers]
    assert not never_emitted, (
        f"declared in CONTEXT_COLUMNS but no producer sets it: {never_emitted}. "
        "Either emit it where the context dict is built, or stop declaring it "
        "(and drop its row from docs/reporting_pipeline.rst)."
    )


def test_the_loop_feeds_only_batches_the_strategy_did_not():
    import pytest
    import torch

    from spectramr.infrastructure.reporting.cases.metric_sink import (
        PerCallMetricSink,
        feed_from_validation_loop,
    )

    sink = PerCallMetricSink(enabled=True)
    mark = sink.n_observed
    assert feed_from_validation_loop(
        sink, {"psnr": 30.0}, step=5, batch_index=0, observed_before=mark
    )
    mark = sink.n_observed
    sink.observe(case_id="own", metrics={"psnr": 31.0}, step=5)
    assert not feed_from_validation_loop(
        sink, {"psnr": 31.0}, step=5, batch_index=1, observed_before=mark
    )
    assert sink.n_rows == 2
    mark = sink.n_observed
    feed_from_validation_loop(
        sink, {"ssim": torch.tensor(0.9), "flag": True}, step=5, batch_index=2, observed_before=mark
    )
    assert sink._rows[-1]["ssim"] == pytest.approx(0.9)
    assert "flag" not in sink._rows[-1]


def test_n_observed_keeps_counting_past_the_row_cap():
    from spectramr.infrastructure.reporting.cases.metric_sink import PerCallMetricSink

    sink = PerCallMetricSink(enabled=True, max_rows=2)
    for i in range(5):
        sink.observe(case_id=str(i), metrics={"psnr": 1.0})
    assert sink.n_rows == 2 and sink.n_observed == 5


def test_a_disabled_or_absent_sink_is_left_alone():
    from spectramr.infrastructure.reporting.cases.metric_sink import (
        PerCallMetricSink,
        feed_from_validation_loop,
    )

    assert not feed_from_validation_loop(
        None, {"psnr": 1.0}, step=1, batch_index=0, observed_before=0
    )
    off = PerCallMetricSink(enabled=False)
    assert not feed_from_validation_loop(
        off, {"psnr": 1.0}, step=1, batch_index=0, observed_before=0
    )

def test_numpy_scalars_are_recorded():
    import numpy as np

    from spectramr.infrastructure.reporting.cases.metric_sink import (
        PerCallMetricSink,
        feed_from_validation_loop,
    )

    sink = PerCallMetricSink(enabled=True)
    assert feed_from_validation_loop(
        sink, {"psnr": np.float32(30.5)}, step=1, batch_index=0, observed_before=0
    )
    assert sink._rows[0]["psnr"] == 30.5


def test_a_scalar_tensor_is_recorded_and_a_map_is_skipped():
    """``val_loss`` arrives as a 0-dim device tensor; it used to be dropped from every row."""
    import torch

    from spectramr.infrastructure.reporting.cases.metric_sink import (
        PerCallMetricSink,
        feed_from_validation_loop,
    )

    sink = PerCallMetricSink(enabled=True)
    metrics = {"val_loss": torch.tensor(0.25), "psnr": 30.0, "error_map": torch.ones(2, 2)}
    assert feed_from_validation_loop(sink, metrics, step=1, batch_index=0, observed_before=0)
    assert sink._rows[0]["val_loss"] == 0.25
    assert "error_map" not in sink._rows[0]
