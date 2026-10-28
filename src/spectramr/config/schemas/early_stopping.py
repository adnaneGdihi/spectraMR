"""Early stopping configuration schema."""

from pydantic import BaseModel, Field, model_validator

from .enums import MetricMode
from .renames import reject_renamed_keys


class EarlyStoppingConfigSchema(BaseModel):
    """Early stopping configuration.

    Defines how to stop training based on validation metrics.

    Example:
        >>> config = EarlyStoppingConfigSchema(
        ...     enabled=True,
        ...     patience=10,
        ...     metric="val_loss",
        ...     mode=MetricMode.MIN,
        ... )
    """

    model_config = {
        "protected_namespaces": (),
        "extra": "forbid",
        "frozen": True,
    }

    # Removed keys (removals.py) are refused with the reason, not a bare extra error.
    _reject_renamed = model_validator(mode="before")(
        classmethod(reject_renamed_keys("early_stopping"))  # type: ignore[arg-type]
    )

    enabled: bool = Field(
        default=False,
        description="Enable early stopping",
    )
    patience: int = Field(
        default=10,
        ge=1,
        description=(
            "Number of consecutive validation passes with no improvement after which "
            "training stops. Early stopping is evaluated on every validation pass, so "
            "its cadence is validation.schedule.interval_steps."
        ),
    )
    patience_min_iterations: int | None = Field(
        default=None,
        ge=0,
        description="Minimum number of iterations to wait before early stopping (overrides patience checks if set)",
    )
    metric: str = Field(
        default="val_loss",
        description="Metric to monitor for early stopping",
    )
    mode: MetricMode = Field(
        default=MetricMode.MIN,
        description="Whether to minimize or maximize metric",
    )
    min_delta: float = Field(
        default=0.0,
        ge=0,
        description="Minimum change in monitored metric to qualify as an improvement",
    )
    restore_best_weights: bool = Field(
        default=True,
        description="Restore best model after early stopping",
    )

    @model_validator(mode="after")
    def _mode_must_match_the_metric_direction(self) -> "EarlyStoppingConfigSchema":
        """Refuse a ``mode`` that contradicts the metric's own direction.

        ``mode: max`` on a loss stops the run when the loss stops RISING, and
        writes the best checkpoint at the WORST iterate. Nothing downstream
        notices: the comparison is well-defined, the counter advances, the file
        is written, and every artifact reports success. The selected model is
        simply the wrong one -- pitfall #16, decided by a single enum.

        The direction is not guessed. ``metric_higher_is_better`` is the single
        resolver already shared by the metrics tracker, early stopping itself
        and the campaign ranker, and it RAISES rather than
        guessing on an unresolvable key. An unknown metric is therefore left
        alone here: naming a metric this schema cannot resolve is a different
        defect with a different owner (the monitor resolution at first
        validation, #178), and duplicating that rejection here would report one
        problem as two.

        A DISABLED block is left alone, matching the sibling check in
        ``infrastructure/services/early_stopping.py``, which has always opened
        with ``if not self.enabled ... return``. Its ``mode`` selects nothing --
        no monitor runs and no checkpoint is chosen through it -- so there is no
        wrong answer for it to give. Until this exemption existed the two checks
        disagreed, and because the schema runs first (at config load) the
        service's exemption was unreachable: a disabled block with a
        contradictory ``mode`` was refused by one owner and excused by the other.
        Measured when closing that gap: 165 corpus arms disable early stopping
        and NONE of them contradicts its metric, so this changes no arm's
        loadability today -- it settles which of two rules is the real one.

        Imported inside the method, mirroring ``renames.py``'s use of
        ``core.execution_ledger``: the direction table is a ``core/`` fact and
        importing it at module scope would make every config import pull the
        metric registry in behind it.
        """
        if not self.enabled:
            return self

        from spectramr.core.metrics.metric_directions import (
            UnknownMetricDirectionError,
            metric_higher_is_better,
        )

        try:
            higher_is_better = metric_higher_is_better(self.metric)
        except UnknownMetricDirectionError:
            return self

        declared_max = self.mode == MetricMode.MAX
        if declared_max != higher_is_better:
            better = "higher" if higher_is_better else "lower"
            wanted = MetricMode.MAX.value if higher_is_better else MetricMode.MIN.value
            raise ValueError(
                f"early_stopping.mode={self.mode.value!r} contradicts "
                f"metric={self.metric!r}, for which {better} is better. As "
                f"written, early stopping would wait for the metric to stop "
                f"getting WORSE and the best checkpoint would hold the worst "
                f"iterate -- silently, since every comparison is still "
                f"well-defined. Set mode: {wanted}."
            )
        return self


__all__ = ["EarlyStoppingConfigSchema"]
