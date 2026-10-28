r"""Registry adapters for figures whose callable is not ``make(df, out_path, **kw)``.

Certificate and learnable-acquisition modules expose ``render_*(out_path, ...)``;
each adapter soft-skips (returns ``None``) until its payload is routed, via
``report_artifacts/`` or an explicit caller kwarg, so they can sit in the
registry without breaking data-less runs.

The cross-run plotters are imported lazily: ``comparison`` reads run directories
through ``batch``, which imports ``pipeline``, which imports the registry.
"""

from __future__ import annotations

from . import certificate_summary, ksd_certificate, learnable_acquisition_pareto


def _save_kw(kw: dict) -> dict:
    """The output options the dispatcher passes, forwarded to a ``render_*``."""
    return {k: kw[k] for k in ("formats", "dpi") if k in kw}


def _certificate_summary(df, out_path, *, certificate_paths=None, metadata=None, **kw):
    if not certificate_paths:
        return None
    return certificate_summary.render_certificate_summary(
        out_path, metadata=metadata, **_save_kw(kw), **certificate_paths
    )


def _ksd_certificate(df, out_path, *, ksd_result=None, metadata=None, **kw):
    if ksd_result is None:
        return None
    return ksd_certificate.render_ksd_certificate(
        out_path, result=ksd_result, metadata=metadata, **_save_kw(kw)
    )


def _learnable_acquisition_pareto(df, out_path, *, acquisition_arms=None, **kw):
    if not acquisition_arms:
        return None
    return learnable_acquisition_pareto.render_learnable_acquisition_pareto(
        out_path, acquisition_arms, **_save_kw(kw)
    )


def _cmp_mosaic(df, out_path, **kwargs):
    from ..comparison.mosaic import make

    return make(df, out_path, **kwargs)


def _cmp_metric_bars(df, out_path, **kwargs):
    from ..comparison.metric_bars import make

    return make(df, out_path, **kwargs)


def _cohort_figure(module: str):
    """An adapter that defers the import of ``comparison.<module>`` until the figure is drawn."""

    def adapter(df, out_path, **kwargs):
        from importlib import import_module

        return import_module(f"..comparison.{module}", __package__).make(df, out_path, **kwargs)

    return adapter


def _cmp_phase_metrics(df, out_path, **kwargs):
    from ..comparison.phase_metrics import make

    return make(df, out_path, **kwargs)


#: ``{fig_id: adapter}`` registered by ``plotters/__init__``.
ADAPTERS = {
    "certificate_summary": _certificate_summary,
    "ksd_certificate": _ksd_certificate,
    "learnable_acquisition_pareto": _learnable_acquisition_pareto,
    "cmp_mosaic": _cmp_mosaic,
    "cmp_metric_bars": _cmp_metric_bars,
    "cmp_phase_metrics": _cmp_phase_metrics,
    "cmp_coverage": _cohort_figure("coverage_matrix"),
    "cmp_effect_forest": _cohort_figure("effect_forest"),
    "cmp_rank_reversal": _cohort_figure("rank_reversal"),
    "cmp_subject_lines": _cohort_figure("subject_lines"),
    "cmp_metric_ladder": _cohort_figure("metric_ladder"),
    "cmp_group_bars": _cohort_figure("group_bars"),
    "cmp_effect_bars": _cohort_figure("effect_bars"),
    "cmp_metric_group_bars": _cohort_figure("metric_group_bars"),
    "cmp_phase_bars": _cohort_figure("phase_bars"),
}
