"""Checkpoint selection is declared four ways and exactly one of them is live.

``early_stopping.metric`` is the one owner: ``EarlyStoppingService`` resolves it
and ``save_best`` consumes it. The other three do nothing —
``metrics.best_metric_name`` is read only by the witness check and the report,
``checkpoint.monitor``/``mode`` are swallowed by the block's ``extra="ignore"``
(#2205), and ``CheckpointDirector._resolve_monitor_metric`` had zero callers.

That last one was the dangerous one, which is why this module also pins its
deletion. It tried ``metrics.best_metric_name`` FIRST, so wiring it — the
obvious repair for an inert knob — would have silently flipped the selection
metric on every arm where the two disagree, and defaulted the rest to a magic
``"loss"`` that is not a registered metric name.

Measured across ``experiments/`` when this landed: **126** arms declare both
spellings with different values, and on **75** of them early stopping is
enabled, so a wrong selection really happens. The remaining 51 have early
stopping off, where nothing selects and the key is merely decorative — which is
why the check passes there rather than crying wolf.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.validation.config_health_checker import ConfigHealthChecker


def _config(
    *,
    live: str | None,
    declared: str | None,
    live_mode: str = "max",
    declared_mode: str = "max",
    enabled: bool = True,
    supplied: set[str] | None = None,
) -> SimpleNamespace:
    """The two selector blocks, at the shapes the checker reads.

    ``supplied`` mimics pydantic's ``model_fields_set``: which keys the YAML
    actually wrote, as opposed to which ones parsing filled in.
    """
    if supplied is None:
        supplied = {"best_metric_name", "best_metric_mode"}
    return SimpleNamespace(
        early_stopping=SimpleNamespace(metric=live, mode=live_mode, enabled=enabled),
        metrics=SimpleNamespace(
            best_metric_name=declared,
            best_metric_mode=declared_mode,
            model_fields_set=supplied,
        ),
    )


def _run(cfg) -> object:
    return ConfigHealthChecker().check_best_metric_matches_early_stopping(cfg)


class TestItFiresOnlyWhereSelectionActuallyHappens:
    def test_agreement_passes(self) -> None:
        assert _run(_config(live="val_psnr", declared="val_psnr")).passed

    def test_a_different_metric_is_an_error(self) -> None:
        result = _run(_config(live="val_loss", declared="val_psnr"))
        assert not result.passed
        assert result.severity == "error"
        assert "val_loss" in result.message and "val_psnr" in result.message

    def test_a_different_mode_is_an_error(self) -> None:
        """`min` vs `max` keeps the WORST checkpoint, which is the worse half."""
        result = _run(
            _config(live="val_loss", declared="val_loss", live_mode="min", declared_mode="max")
        )
        assert not result.passed
        assert "mode" in result.message

    def test_early_stopping_disabled_passes_even_when_they_disagree(self) -> None:
        """Nothing selects, so the inert key claims nothing about a selection.

        51 corpus arms are in this state; erroring on them would make the check
        unsatisfiable without teaching anyone anything.
        """
        assert _run(_config(live="val_loss", declared="val_psnr", enabled=False)).passed

    def test_only_one_spelling_declared_passes(self) -> None:
        assert _run(_config(live="val_psnr", declared=None)).passed
        assert _run(_config(live=None, declared="val_psnr")).passed

    def test_a_missing_block_passes(self) -> None:
        assert _run(SimpleNamespace(early_stopping=None, metrics=None)).passed


class TestTheModeIsComparedByValueNotRepr:
    """Both spellings resolve to a ``MetricMode``; comparing reprs would be fine
    but PRINTING one puts ``MetricMode.MAX`` in a message whose job is to be
    matched against a YAML line reading ``max``."""

    def test_enum_and_string_modes_agree(self) -> None:
        from spectramr.config.schemas.enums import MetricMode

        result = _run(
            _config(
                live="val_psnr",
                declared="val_psnr",
                live_mode=MetricMode.MAX,
                declared_mode=MetricMode.MAX,
            )
        )
        assert result.passed
        assert "MetricMode" not in result.message, result.message

    def test_the_error_message_prints_the_yaml_spelling(self) -> None:
        from spectramr.config.schemas.enums import MetricMode

        result = _run(
            _config(
                live="val_loss",
                declared="val_loss",
                live_mode=MetricMode.MIN,
                declared_mode=MetricMode.MAX,
            )
        )
        assert not result.passed
        assert "'min'" in result.message and "'max'" in result.message
        assert "MetricMode" not in result.message


class TestTheDeadResolverIsGone:
    """It is deleted, not wired: its precedence was the wrong way round."""

    def test_checkpoint_director_no_longer_defines_it(self) -> None:
        from spectramr.infrastructure.builders.directors import checkpoint_director

        assert not hasattr(checkpoint_director.CheckpointDirector, "_resolve_monitor_metric")

    def test_nothing_reintroduces_a_best_metric_name_first_precedence(self) -> None:
        """The shape to keep out, not just the name."""
        from spectramr.infrastructure.builders.directors import checkpoint_director

        source = inspect.getsource(checkpoint_director)
        body = source.split("# `_resolve_monitor_metric` was deleted here")[-1]
        assert '("metrics", "best_metric_name")' not in body


class TestItIsInvokedByTheAudit:
    """A check that run_all_checks never calls protects nothing — the repo's own
    orphan detector caught exactly that during this change."""

    def test_run_all_checks_calls_it(self) -> None:
        source = inspect.getsource(ConfigHealthChecker.run_all_checks)
        assert "check_best_metric_matches_early_stopping" in source


@pytest.mark.parametrize(
    ("live", "declared", "live_mode", "declared_mode", "enabled", "should_pass"),
    [
        ("val_psnr", "val_psnr", "max", "max", True, True),
        ("val_loss", "val_psnr", "min", "max", True, False),
        ("val_psnr", "val_psnr", "min", "max", True, False),
        ("val_loss", "val_psnr", "min", "max", False, True),
    ],
    ids=["agree", "both-differ", "mode-only", "disabled"],
)
def test_the_predicate_over_every_planted_shape(
    live: str, declared: str, live_mode: str, declared_mode: str, enabled: bool, should_pass: bool
) -> None:
    result = _run(
        _config(
            live=live,
            declared=declared,
            live_mode=live_mode,
            declared_mode=declared_mode,
            enabled=enabled,
        )
    )
    assert result.passed is should_pass


class TestItComparesWhatTheYamlWroteNotWhatParsingFilledIn:
    """The defaults are not neutral, and this check would have been unusable.

    ``metrics.best_metric_name`` defaults to ``'val_loss'`` and
    ``best_metric_mode`` to ``MIN``. An arm that declares only
    ``early_stopping.metric: val_psnr`` therefore PARSES with
    ``best_metric_name='val_loss'`` — a value nobody wrote. Comparing the parsed
    value flagged a contradiction the author never made on **109**
    ``inprogress/`` arms; comparing what was supplied leaves exactly the arms
    that really declared two different selectors.
    """

    def test_an_undeclared_inert_key_is_not_a_contradiction(self) -> None:
        result = _run(_config(live="val_psnr", declared="val_loss", supplied=set()))
        assert result.passed, result.message
        assert "not declared" in result.message

    def test_an_undeclared_mode_is_not_a_contradiction(self) -> None:
        result = _run(
            _config(
                live="val_psnr",
                declared="val_psnr",
                live_mode="max",
                declared_mode="min",
                supplied={"best_metric_name"},
            )
        )
        assert result.passed, result.message

    def test_a_declared_one_still_errors(self) -> None:
        """The guard must not swallow the real finding."""
        result = _run(_config(live="val_psnr", declared="val_loss", supplied={"best_metric_name"}))
        assert not result.passed
        assert result.severity == "error"

    def test_a_config_without_fields_set_is_still_compared(self) -> None:
        """A plain stub (no pydantic) must not silently disable the check."""
        cfg = SimpleNamespace(
            early_stopping=SimpleNamespace(metric="val_psnr", mode="max", enabled=True),
            metrics=SimpleNamespace(best_metric_name="val_loss", best_metric_mode="max"),
        )
        assert not _run(cfg).passed
