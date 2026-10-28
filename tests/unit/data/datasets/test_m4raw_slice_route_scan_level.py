"""What stays a property of the SCAN when a record becomes one slice (#2673).

Two quantities the group route computed over a whole volume:

* the self-calibrated ``Sigma_n`` (n2n arm E, the self-calibrated SSDU arm and
  the validation-time recorruption): a receive-chain property, fitted on the
  scan's central slices. A slice record must still fit those slices, read once
  per scan and cached, not the one slice it was handed -- that would be a
  different estimator per slice and per worker, and an air slice's on an edge.
* the k-space scale: one robust quantile per subject, so per volume on the
  group route and per slice on this one. Validation and prediction follow the
  same flag, so the three agree by construction; the change of definition is
  measured here and is the owner decision named in the PR.
"""

from __future__ import annotations

import pytest
import torch

import spectramr.data.datasets.m4raw_dataset as m4raw_mod
from spectramr.data.datasets.m4raw_dataset import (
    M4RawRepetitionDataset,
    fit_scan_noise_covariance,
    sigma_fit_window,
)
from spectramr.data.transforms.normalization import KSpaceNormalizationTransform
from spectramr.infrastructure.inference.predict_kspace import PredictKSpace
from tests.fixtures import m4raw_slice_volumes as vols


def _files(tmp_path) -> list:
    return vols.write_group(tmp_path, "2022061001", seed=1) + vols.write_group(
        tmp_path, "2022061002", seed=2
    )


def _datasets(files, **kw):
    kw = {"single_contrast": True, "coil_processing_mode": "none", **kw}
    return M4RawRepetitionDataset(files, **kw), M4RawRepetitionDataset(
        files, slice_level_records=True, **kw
    )


_R2R_SELFCAL = {"target_mode": "r2r", "r2r_covariance_source": "self_calibrated"}
#: Arm E's validation split: an averaging target that publishes the scan's fit.
_VAL_SELFCAL = {
    "target_mode": "phase_aligned_mean",
    "nex_target_exclude_input": True,
    "r2r_covariance_source": "self_calibrated",
}


def _scan_fit(path) -> torch.Tensor:
    return fit_scan_noise_covariance(m4raw_mod._load_kspace(path))[0]


def test_the_fit_window_is_the_central_pair() -> None:
    assert list(sigma_fit_window(18)) == [8, 9]
    assert list(sigma_fit_window(vols.SLICES)) == [2, 3]
    assert list(sigma_fit_window(1)) == [0]


@pytest.mark.parametrize("first", [0, vols.SLICES - 1, 2])
def test_every_record_of_a_scan_draws_from_the_scans_central_fit(tmp_path, first) -> None:
    """Whichever slice a worker loads first, the cached Sigma_n is the scan's."""
    files = _files(tmp_path)
    vol, sl = _datasets(files, **_R2R_SELFCAL)
    _ = vol[0]
    _ = sl[first]
    key = str(files[0])
    torch.testing.assert_close(sl._r2r_cov_cache[key], vol._r2r_cov_cache[key])
    torch.testing.assert_close(sl._r2r_cov_cache[key], _scan_fit(files[0]))


def test_the_own_slice_fit_is_a_different_estimator(tmp_path) -> None:
    """The planted shape: fitting the record's own slice (what the slice route
    did before the fit read the scan's central slices) moves Sigma_n."""
    files = _files(tmp_path)
    volume = m4raw_mod._load_kspace(files[0])
    central = _scan_fit(files[0])
    own = fit_scan_noise_covariance(volume[0:1])[0]
    assert not torch.allclose(own, central, rtol=1e-3, atol=0)


def test_the_central_slices_are_read_once_per_scan(tmp_path, monkeypatch) -> None:
    files = _files(tmp_path)
    _vol, sl = _datasets(files, **_R2R_SELFCAL)
    reads: list = []
    real = m4raw_mod._load_kspace

    def spy(path, slice_index=None):
        reads.append((path.name, slice_index))
        return real(path, slice_index)

    monkeypatch.setattr(m4raw_mod, "_load_kspace", spy)
    _ = sl[0]
    window = [(files[0].name, s) for s in sigma_fit_window(vols.SLICES)]
    assert reads == [(f.name, 0) for f in files[:3]] + window
    reads.clear()
    _ = sl[1]
    assert reads == [(f.name, 1) for f in files[:3]], "the fit was not cached per scan"


def test_validation_publishes_the_scans_fit_on_every_slice(tmp_path) -> None:
    files = _files(tmp_path)
    vol, sl = _datasets(files, **_VAL_SELFCAL)
    expected = vol[0]["noise_covariance"]
    for s in range(vols.SLICES):
        torch.testing.assert_close(sl[s]["noise_covariance"], expected)


# --- the k-space scale: per volume on groups, per slice on records ---------------


def _norm() -> KSpaceNormalizationTransform:
    """The n2n arms' declaration: image-domain p99, measured on the input."""
    return KSpaceNormalizationTransform(
        percentile=0.99, scale_domain="image", scale_reference="input"
    )


def _predict_scales(input_volume: torch.Tensor, per_slice: bool) -> torch.Tensor:
    """``PredictKSpace._scale`` as the predict path resolves it for this flag."""
    predict = PredictKSpace.__new__(PredictKSpace)
    predict.norm = _norm().spec
    predict.per_slice_scale = per_slice
    return predict._scale(input_volume)


def test_training_validation_and_predict_share_one_scale_per_route(tmp_path) -> None:
    files = _files(tmp_path)
    vol, sl = _datasets(files, target_mode="phase_aligned_mean", transform=_norm())
    raw = m4raw_mod._load_kspace(files[0])
    per_volume = _predict_scales(raw, per_slice=False)
    per_slice = _predict_scales(raw, per_slice=True)
    torch.testing.assert_close(vol[0]["kspace_scale"].reshape(()), per_volume[0])
    for s in range(vols.SLICES):
        torch.testing.assert_close(sl[s]["kspace_scale"].reshape(()), per_slice[s])
    # The definition moved: the edge slices of a tapered volume scale lower.
    ratio = per_slice / per_volume
    assert float(ratio.min()) < 0.6 and 0.95 < float(ratio.max()) < 1.1
