"""Sim2Rank figure entry points must not leak matplotlib rcParams.

Each plotter used to write its house theme into the global ``rcParams``, so
the next figure drawn in the same process (e.g. a spectramr reporting
plotter) inherited it. Each entry point below runs on tiny synthetic inputs
and the rcParams snapshot must be identical afterwards. The ``backend`` key
is excluded: matplotlib's ``rc_context`` does not restore it, and the CLI
mains select Agg on purpose.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


class _StopError(Exception):
    """Raised by a patched loader to end a CLI main right after its setup."""


def _boom(*_args, **_kwargs):
    raise _StopError


def _rc() -> dict:
    return {k: v for k, v in matplotlib.rcParams.items() if k != "backend"}


@dataclass
class _Spec:
    registry_key: str
    display_name: str


def test_sim2rank_figure_entry_points_restore_rcparams(tmp_path, monkeypatch):
    import scripts.sim2rank.degradation_snapshots as ds
    import scripts.sim2rank.per_axis_figures as pa
    import scripts.sim2rank.run_all_generations as rag
    import scripts.sim2rank.run_meta_eval_all_datasets as rme
    import scripts.sim2rank.sim2rank_diagnostics as diag
    import scripts.sim2rank.simulator_calibration as sc
    from scripts.sim2rank.figures import plot_srocc_bootstrap_ci
    from scripts.sim2rank.visualization import plot_inter_metric_correlation

    rng = np.random.default_rng(0)
    specs = [_Spec(f"m{i}", f"M{i}") for i in range(3)]

    monkeypatch.setattr(pa, "load_sim2rank", _boom)
    monkeypatch.setattr(sc, "list_degradations", _boom)
    monkeypatch.setattr(ds, "build_phantom", _boom)

    def per_axis_main():
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "per_axis_figures",
                "--sim2rank-json",
                str(tmp_path / "x.json"),
                "--output-dir",
                str(tmp_path / "pa"),
            ],
        )
        pa.main()

    def calibration_main():
        monkeypatch.setattr(
            sys, "argv", ["simulator_calibration", "--output-dir", str(tmp_path / "sc")]
        )
        sc.main()

    def snapshots_main():
        monkeypatch.setattr(
            sys, "argv", ["degradation_snapshots", "--output-dir", str(tmp_path / "ds")]
        )
        ds.main()

    cases = {
        "visualization.plot_inter_metric_correlation": lambda: plot_inter_metric_correlation(
            rng.random((3, 8)), specs, output_path=tmp_path / "corr.png"
        ),
        "figures.plot_srocc_bootstrap_ci": lambda: plot_srocc_bootstrap_ci(
            {"a": rng.random(3), "b": rng.random(3)},
            specs,
            n_bootstrap=10,
            output_path=tmp_path / "srocc.png",
        ),
        "run_all_generations.plot_tri_generation_bootstrap": lambda: (
            rag.plot_tri_generation_bootstrap(
                ["A", "B"], {"A": 0.9, "B": 0.5}, {"B"}, output_path=tmp_path / "boot.png"
            )
        ),
        "run_meta_eval_all_datasets._plot_rank_scatter": lambda: rme._plot_rank_scatter(
            {"d1": {"k1": 1, "k2": 2}, "d2": {"k1": 2, "k2": 1}},
            ["k1", "k2"],
            ["d1", "d2"],
            {"k1": "K1", "k2": "K2"},
            tmp_path / "rank.png",
        ),
        "sim2rank_diagnostics.plot_pearson_correlation_map": lambda: (
            diag.plot_pearson_correlation_map(
                torch.rand(1, 1, 16, 16),
                {diag.HARD_AXES[0]: torch.rand(1, 1, 16, 16)},
                tmp_path / "pearson.png",
                kernel_size=5,
            )
        ),
        "per_axis_figures.main": per_axis_main,
        "simulator_calibration.main": calibration_main,
        "degradation_snapshots.main": snapshots_main,
    }

    for label, call in cases.items():
        before = _rc()
        with contextlib.suppress(_StopError):
            call()
        assert _rc() == before, f"{label} leaked rcParams"
