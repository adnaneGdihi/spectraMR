"""The precomputed Sigma_n store: exact round trip, refused mismatches, and the loader reads it (#3117)."""

from __future__ import annotations

import pytest
import torch

import spectramr.data.datasets.m4raw_dataset as m4raw_mod
from spectramr.data.datasets.m4raw_dataset import M4RawRepetitionDataset
from spectramr.data.datasets.m4raw_noise_covariance_store import (
    NoiseCovarianceMismatchError,
    NoiseCovarianceMissingError,
    NoiseCovarianceStore,
    covariance_key,
    read_covariances,
    write_covariances,
)
from tests.fixtures import m4raw_slice_volumes as vols

_R2R_SELFCAL = {
    "single_contrast": True,
    "coil_processing_mode": "none",
    "target_mode": "r2r",
    "r2r_covariance_source": "self_calibrated",
    "slice_level_records": True,
}


def _cov(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(4, 4, dtype=torch.complex64, generator=g)
    return a @ a.conj().T


def test_the_covariance_round_trips_exactly(tmp_path) -> None:
    cov = _cov(0)
    write_covariances(tmp_path, {"2022061001_T1": ("2022061001_T101.h5", cov)}, "band")
    (name, back) = read_covariances(tmp_path, "band")["2022061001_T1"]
    assert name == "2022061001_T101.h5"
    assert back.dtype == torch.complex64 and torch.equal(back, cov)


def test_a_store_fitted_under_another_policy_is_refused(tmp_path) -> None:
    write_covariances(tmp_path, {"k": ("k01.h5", _cov(0))}, "band")
    with pytest.raises(NoiseCovarianceMismatchError, match="support_policy"):
        read_covariances(tmp_path, "acs")


def test_an_empty_root_is_an_error_not_an_empty_cache(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="precompute_m4raw_noise_covariance"):
        NoiseCovarianceStore.declared(tmp_path, "raise", "band")


def test_missing_and_mismatched_groups(tmp_path) -> None:
    groups = {covariance_key("p_T101.h5"): ("p_T101.h5", _cov(1))}
    with pytest.raises(NoiseCovarianceMissingError):
        NoiseCovarianceStore(groups, "raise").get("/x/q_T101.h5")
    assert NoiseCovarianceStore(groups, "fit").get("/x/q_T101.h5") is None
    with pytest.raises(NoiseCovarianceMismatchError, match="fits it"):
        NoiseCovarianceStore(groups, "raise").get("/x/p_T102.h5")
    with pytest.raises(ValueError, match="noise_covariance_missing"):
        NoiseCovarianceStore(groups, "estimate")


def test_the_loader_serves_the_stored_fit_without_fitting(tmp_path, monkeypatch) -> None:
    """Rule 16: the store is read on the production path, and the fit never runs."""
    files = vols.write_group(tmp_path, "2022061001", seed=1)
    fitted = m4raw_mod.fit_scan_noise_covariance(
        m4raw_mod.load_sigma_fit_slices(files[0], vols.SLICES)
    )[0]
    write_covariances(
        tmp_path / "store", {covariance_key(files[0]): (files[0].name, fitted)}, "band"
    )

    def _refuse(*_a, **_k):
        raise AssertionError("fitted despite the store")

    monkeypatch.setattr(m4raw_mod, "fit_scan_noise_covariance", _refuse)
    ds = M4RawRepetitionDataset(files, noise_covariance_root=tmp_path / "store", **_R2R_SELFCAL)
    _ = ds[0]
    assert torch.equal(ds._r2r_cov_cache[str(files[0])], fitted.to(torch.complex64))


def test_the_stored_fit_equals_the_loaders_own(tmp_path) -> None:
    """Without a store the slice route fits the same slices to the same matrix."""
    files = vols.write_group(tmp_path, "2022061001", seed=1)
    ds = M4RawRepetitionDataset(files, **_R2R_SELFCAL)
    _ = ds[0]
    stored = m4raw_mod.fit_scan_noise_covariance(
        m4raw_mod.load_sigma_fit_slices(files[0], vols.SLICES)
    )[0]
    torch.testing.assert_close(ds._r2r_cov_cache[str(files[0])], stored, rtol=0, atol=0)
