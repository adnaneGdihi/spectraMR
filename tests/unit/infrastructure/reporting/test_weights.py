import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from spectramr.infrastructure.reporting import weights
from spectramr.infrastructure.reporting.catalog.examples_payloads import make_checkpoint_runs


def test_checkpoints_are_in_step_order_with_best_last():
    run = make_checkpoint_runs(1)["arm0"]
    found = weights.checkpoint_files(run)
    assert [step for step, _ in found] == [50, 2000, 4000, None]
    assert found[-1][1].name == "best.safetensors"


def test_a_long_run_is_thinned_but_keeps_first_and_last(tmp_path):
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for step in range(0, 40 * 100, 100):
        save_file({"0:models.g.w": torch.zeros(2)}, str(folder / f"step_{step:08d}.safetensors"))
    found = weights.checkpoint_files(tmp_path, limit=6)
    steps = [s for s, _ in found]
    assert len(steps) == 6 and steps[0] == 0 and steps[-1] == 3900


def test_the_same_positions_are_sampled_in_every_checkpoint(tmp_path):
    a, b = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    base = torch.arange(1000, dtype=torch.float32)
    save_file({"0:models.g.w": base}, str(a))
    save_file({"0:models.g.w": base + 1}, str(b))
    sa, sb = weights.sample_weights(a, 100), weights.sample_weights(b, 100)
    assert len(sa.values) == 100
    np.testing.assert_allclose(sb.values - sa.values, 1.0)


def test_only_float_model_tensors_are_sampled_and_grouped_by_layer(tmp_path):
    path = tmp_path / "c.safetensors"
    save_file(
        {
            "0:models.generator.encoder.0.weight": torch.ones(40),
            "1:models.generator.decoder.0.weight": torch.ones(40),
            "2:optim.generator.state[0].exp_avg": torch.full((40,), 9.0),
            "3:models.generator.counter": torch.ones(4, dtype=torch.uint8),
        },
        str(path),
    )
    sample = weights.sample_weights(path, 20)
    assert set(sample.group_names) == {"generator.encoder", "generator.decoder"}
    assert 9.0 not in sample.values


def test_a_checkpoint_without_model_parameters_is_refused(tmp_path):
    path = tmp_path / "d.safetensors"
    save_file({"0:optim.generator.state[0].step": torch.ones(2)}, str(path))
    with pytest.raises(ValueError, match="no floating-point model parameters"):
        weights.sample_weights(path)
