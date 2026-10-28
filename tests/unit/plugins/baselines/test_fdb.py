"""FDB validation runs the authors' reverse bridge, observed rather than read.

Small builds only: at 32 x 32 with ``bridge_steps=8`` the test schedule is 12 steps.
``monkeypatch.chdir`` keeps upstream's ``w.npy`` out of the repository root.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from spectramr_baselines._normalization import peak_magnitude
from spectramr_baselines.fdb import _FDB_SCRIPT_UTIL, FDBBaseline

pytestmark = pytest.mark.skipif(
    not _FDB_SCRIPT_UTIL.exists(), reason="FDB vendored upstream absent"
)

_SIZE = 32


def _build(**kwargs) -> FDBBaseline:
    adapter = FDBBaseline(
        image_size=_SIZE,
        bridge_steps=8,
        undersampling_rate=2,
        model_kwarg_overrides={"num_channels": 32},
        **kwargs,
    )
    return adapter.eval()


@pytest.fixture(autouse=True)
def _isolate_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)


def _kspace(bridge, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return bridge.fft2c(torch.complex(x[:, 0:1], x[:, 1:2])) * mask


def test_validation_runs_the_upstream_loop_at_the_test_schedule(monkeypatch) -> None:
    adapter = _build()
    assert adapter.test_bridge_steps == 12
    seen: list[int] = []
    real = adapter._new_bridge

    def spy(steps):
        bridge = real(steps)
        loop = bridge.p_sample_loop_condition_progressive

        def counted(*a, **k):
            seen.append(bridge.num_timesteps)
            yield from loop(*a, **k)

        bridge.p_sample_loop_condition_progressive = counted
        return bridge

    monkeypatch.setattr(adapter, "_new_bridge", spy)
    x = torch.randn(2, 2, _SIZE, _SIZE)
    out = adapter.validation_sample(x, x)
    assert out.shape == x.shape and out.dtype == torch.float32
    assert seen == [12], "upstream's reverse loop did not run over the test schedule"


def test_the_output_keeps_the_acquired_kspace_in_the_inputs_units() -> None:
    """The last step is upstream's data consistency, so acquired samples survive exactly."""
    adapter = _build()
    x = 7.0 * torch.randn(2, 2, _SIZE, _SIZE)
    out = adapter.validation_sample(x, x)
    mask = adapter.acquisition_mask_tensor(x.device)
    bridge = adapter._test_bridge()
    scale = peak_magnitude(x)
    expected = _kspace(bridge, x / scale, mask)
    got = _kspace(bridge, out / scale, mask)
    assert torch.allclose(got, expected, atol=1e-4 * float(expected.abs().max()))
    assert int(mask.sum()) == _SIZE * _SIZE // 4


def test_complex_input_comes_back_complex() -> None:
    adapter = _build()
    z = torch.randn(1, 1, _SIZE, _SIZE, dtype=torch.complex64)
    assert adapter.validation_sample(z, z).shape == z.shape
    assert torch.is_complex(adapter.validation_sample(z, z))


def test_validation_is_repeatable_and_leaves_the_training_rng_alone() -> None:
    adapter = _build()
    x = torch.randn(1, 2, _SIZE, _SIZE)
    np.random.seed(5)
    expected = np.random.rand(2)
    np.random.seed(5)
    first = adapter.validation_sample(x, x)
    assert np.array_equal(np.random.rand(2), expected)
    assert torch.equal(first, adapter.validation_sample(x, x))


def test_multicoil_is_refused_not_guessed() -> None:
    adapter = _build()
    adapter._model_kwargs["data_type"] = "multicoil"
    x = torch.randn(1, 2, _SIZE, _SIZE)
    with pytest.raises(NotImplementedError, match="singlecoil"):
        adapter.validation_sample(x, x)


def test_a_size_other_than_image_size_is_refused() -> None:
    x = torch.randn(1, 2, _SIZE * 2, _SIZE * 2)
    with pytest.raises(ValueError, match="image_size"):
        _build().validation_sample(x, x)


def test_the_test_bridge_carries_the_trained_calibration() -> None:
    adapter = _build()
    adapter._diffusion.w = np.linspace(0.0, 0.7, 8)
    bridge = adapter._test_bridge()
    assert bridge.num_timesteps == 12
    assert bridge.w.shape == (12,)
    assert bridge.w[0] == 0.0 and bridge.w[-1] == pytest.approx(0.7)


def test_a_resumed_run_loads_its_calibration(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    np.save(run / "w.npy", np.full(8, 0.25))
    adapter = _build()
    adapter.set_calibration_dir(run)
    assert np.allclose(adapter._diffusion.w, 0.25)


def test_provenance_records_every_test_knob() -> None:
    record = _build(test_acceleration=8, test_mask_sigma=0.2, test_mask_seed=3).provenance()
    assert record["test_acceleration"] == "8"
    assert record["test_bridge_steps"] == str(8 * 7 * 2 // 8)
    assert record["test_mask"] == "normal_density sigma=0.2 seed=3"


def test_the_calibration_travels_with_the_state_dict() -> None:
    """A fresh process with no ``w.npy`` gets ``w`` back from the checkpoint alone."""
    trained = _build().train()
    trained.training_loss(torch.randn(1, 2, _SIZE, _SIZE))
    assert trained.calibration_w.abs().sum() > 0, "training wrote no calibration"

    fresh = _build()
    fresh._diffusion.w = np.zeros_like(np.asarray(fresh._diffusion.w))
    fresh.load_state_dict(trained.state_dict())
    np.testing.assert_allclose(fresh.calibration(), trained.calibration_w.numpy())


def test_an_uncalibrated_bridge_is_reported(caplog) -> None:
    adapter = _build()
    adapter.calibration_w.zero_()
    adapter._diffusion.w = np.zeros_like(np.asarray(adapter._diffusion.w))
    adapter.calibration()
    assert "all zero" in caplog.text


def test_a_checkpoint_without_the_calibration_buffer_loads_strictly(caplog) -> None:
    """A state dict saved before ``calibration_w`` existed loads under strict=True."""
    legacy = {k: v for k, v in _build().state_dict().items() if k != "calibration_w"}
    adapter = _build()
    adapter._diffusion.w = np.zeros_like(np.asarray(adapter._diffusion.w))
    adapter.load_state_dict(legacy, strict=True)
    assert not adapter.calibration_w.any()
    assert "carries no calibration_w" in caplog.text
    adapter.calibration()
    assert "all zero" in caplog.text
