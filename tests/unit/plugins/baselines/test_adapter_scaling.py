"""Each adapter applies its own authors' normalisation and seeds validation from the input.

The upstream objects are stubbed where they would need an accelerator; what is checked is
the adapter's side of the contract -- what it hands the upstream, and what it hands back.
"""

from __future__ import annotations

import pytest
import torch
from spectramr_baselines._cdiffmr_upstream import _UPSTREAM_NETWORK_FILE
from spectramr_baselines._normalization import peak_magnitude
from spectramr_baselines.cdiffmr import CDiffMRBaseline
from spectramr_baselines.fdb import _FDB_SCRIPT_UTIL, FDBBaseline
from spectramr_baselines.shen2024 import Shen2024Baseline

needs_cdiffmr = pytest.mark.skipif(
    not _UPSTREAM_NETWORK_FILE.exists(), reason="CDiffMR vendored upstream absent"
)
needs_fdb = pytest.mark.skipif(not _FDB_SCRIPT_UTIL.exists(), reason="FDB vendored upstream absent")


def _scaled(size: int = 32, peak: float = 37.0) -> torch.Tensor:
    x = torch.randn(2, 2, size, size)
    return x / peak_magnitude(x) * peak


@needs_cdiffmr
def test_cdiffmr_trains_on_a_peak_normalised_image(monkeypatch) -> None:
    adapter = CDiffMRBaseline(resolution=32, num_steps=5)
    seen: list[torch.Tensor] = []

    def process(x):
        seen.append(x)
        return x, x

    monkeypatch.setattr(adapter, "_process", lambda: process)

    adapter.training_loss(_scaled())

    torch.testing.assert_close(peak_magnitude(seen[0]).flatten(), torch.ones(2))


@needs_cdiffmr
def test_cdiffmr_validation_seeds_from_the_masked_input_and_restores_its_units(
    monkeypatch,
) -> None:
    adapter = CDiffMRBaseline(resolution=32, num_steps=5, sampling_step=3)
    monkeypatch.setattr(adapter, "acquisition_mask_tensor", lambda device: torch.ones(1, 1, 32, 32))
    calls: list[dict] = []

    def sample(x_start, x_obs=None, mask=None, *, batch_size=1, timesteps=None):
        calls.append({"x_start": x_start, "x_obs": x_obs, "mask": mask, "t": timesteps})
        return x_start

    monkeypatch.setattr(adapter, "sample", sample)
    x = _scaled()

    out = adapter.validation_sample(x, torch.zeros_like(x))

    assert calls[0]["t"] == 3
    assert calls[0]["mask"].shape == (2, 1, 32, 32)
    assert calls[0]["x_obs"] is calls[0]["x_start"]
    torch.testing.assert_close(peak_magnitude(calls[0]["x_start"]).flatten(), torch.ones(2))
    torch.testing.assert_close(out, x, atol=1e-4, rtol=1e-5)


@needs_cdiffmr
def test_cdiffmr_refuses_a_sampling_step_outside_the_ladder() -> None:
    with pytest.raises(ValueError, match="sampling_step=6"):
        CDiffMRBaseline(resolution=32, num_steps=5, sampling_step=6)


@needs_cdiffmr
def test_cdiffmr_missing_mask_names_what_the_upstream_ships() -> None:
    adapter = CDiffMRBaseline(resolution=32)
    with pytest.raises(FileNotFoundError, match=r"random_af4_cf0\.08_pe32\.npy"):
        adapter.acquisition_mask_tensor(torch.device("cpu"))


@needs_cdiffmr
def test_cdiffmr_published_mask_is_tiled_over_the_readout() -> None:
    adapter = CDiffMRBaseline(resolution=48)

    mask = adapter.acquisition_mask_tensor(torch.device("cpu"))

    assert mask.shape == (1, 1, 48, 48)
    assert torch.equal(mask[0, 0, 0], mask[0, 0, -1])
    assert 0.0 < float(mask.mean()) < 0.5, "an AF4 mask keeps roughly a quarter of the lines"


@needs_cdiffmr
def test_cdiffmr_provenance_stamps_the_sampling_knobs() -> None:
    record = CDiffMRBaseline(resolution=48, num_steps=5).provenance()
    assert record["acquisition_mask"] == "random_af4_cf0.08_pe48.npy"
    assert record["sampling_step"] == "5"


def test_shen2024_validation_reconstructs_the_input_not_the_target(monkeypatch) -> None:
    adapter = Shen2024Baseline(image_size=32, timesteps=5)
    seen: list[torch.Tensor] = []
    monkeypatch.setattr(adapter, "sample", lambda x, **_: seen.append(x) or x)
    x = _scaled()

    adapter.validation_sample(x, torch.zeros_like(x))

    assert seen and seen[0] is x


@needs_fdb
def test_fdb_trains_on_a_peak_normalised_image(monkeypatch) -> None:
    adapter = FDBBaseline(image_size=32)
    seen: list[torch.Tensor] = []

    def training_losses(model, x_0, t, model_kwargs):
        seen.append(x_0)
        return {"loss": x_0.abs().flatten(1).mean(1)}

    monkeypatch.setattr(adapter._diffusion, "training_losses", training_losses)

    adapter.training_loss(_scaled())

    torch.testing.assert_close(peak_magnitude(seen[0]).flatten(), torch.ones(2))
