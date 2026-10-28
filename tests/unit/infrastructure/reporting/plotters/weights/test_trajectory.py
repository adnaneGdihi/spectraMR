import pandas as pd

from spectramr.infrastructure.reporting.catalog.examples_payloads import make_checkpoint_runs
from spectramr.infrastructure.reporting.plotters.weights import trajectory


def test_several_arms_render_with_a_swatch_legend(tmp_path):
    runs = make_checkpoint_runs(3)
    out = trajectory.make(
        pd.DataFrame(), tmp_path / "f.pdf", weight_runs=runs, formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()


def test_one_arm_is_coloured_by_step_with_a_colour_bar(tmp_path):
    run = make_checkpoint_runs(1)
    out = trajectory.make(
        pd.DataFrame(), tmp_path / "f.pdf", weight_runs=run, formats=("pdf",), dpi=72
    )
    assert out is not None and out.exists()


def test_soft_skips_without_runs_or_with_a_single_checkpoint(tmp_path):
    assert trajectory.make(pd.DataFrame(), tmp_path / "f.pdf", weight_runs=None) is None
    run = make_checkpoint_runs(1, steps=())["arm0"]
    (run / "checkpoints" / "best.safetensors").unlink()
    assert trajectory.make(pd.DataFrame(), tmp_path / "f.pdf", weight_runs={"a": run}) is None


def test_arms_with_a_different_architecture_are_left_out():
    runs = make_checkpoint_runs(2)
    import torch
    from safetensors.torch import save_file

    odd = runs["arm1"]
    for path in (odd / "checkpoints").glob("*.safetensors"):
        save_file({"0:models.generator.only.weight": torch.zeros(4)}, str(path))
    assert list(trajectory.read_series(runs)) == ["arm0"]
