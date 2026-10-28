"""Unit tests for bidirectional ContrastAwarePairedDataset.

Tests the swap mechanism in ContrastAwareSubjectBuilder.build() and
the bidirectional_mode / allow_unpaired flags in ContrastAwarePairedDataset.
All tests run without real NIfTI files by mocking the IO layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.data.datasets.contrast_aware import (
    ContrastAwarePairedDataset,
    ContrastAwareSubjectBuilder,
    ContrastConfig,
)

# The IOStrategyFactory is imported inside ContrastAwareSubjectBuilder.__init__
# from spectramr.data.io_strategies, so we must patch it at the *source* location.
_IO_FACTORY_PATCH = "spectramr.data.io_strategies.IOStrategyFactory.get"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ulf_config():
    return ContrastConfig(name="ULF_64mT", normalization="percentile", percentile=99.5)


@pytest.fixture
def hf_config():
    return ContrastConfig(name="HF_3T", normalization="percentile", percentile=99.5)


def _fake_loader(path: str) -> torch.Tensor:
    """Return deterministic fake tensor whose values encode the path."""
    seed = hash(path) % 1000
    torch.manual_seed(seed)
    return torch.rand(1, 16, 16, 8)


def _make_record(primary: str, target: str | None = None, file_id: str = "test"):
    return {
        "primary_path": primary,
        "target_path": target,
        "file_id": file_id,
    }


def _mock_io():
    """Create a mock IO strategy whose .load() calls _fake_loader."""
    io = MagicMock()
    io.load.side_effect = _fake_loader
    return io


# ---------------------------------------------------------------------------
# ContrastAwareSubjectBuilder.build() — swap param
# ---------------------------------------------------------------------------


class TestSubjectBuilderSwap:
    @patch(_IO_FACTORY_PATCH)
    def test_no_swap_uses_primary_as_input(self, mock_get, ulf_config, hf_config):
        mock_get.return_value = _mock_io()

        builder = ContrastAwareSubjectBuilder(
            input_contrast=ulf_config,
            target_contrast=hf_config,
            io_strategy="nifti",
        )
        record = _make_record("/data/ulf_scan.nii.gz", "/data/hf_scan.nii.gz")
        subject = builder.build(record, swap=False)

        # input_path should be primary_path
        assert subject["input_path"] == "/data/ulf_scan.nii.gz"
        assert subject["target_path"] == "/data/hf_scan.nii.gz"
        assert subject["input_contrast"] == "ULF_64mT"
        assert subject["target_contrast"] == "HF_3T"

    @patch(_IO_FACTORY_PATCH)
    def test_swap_uses_target_as_input(self, mock_get, ulf_config, hf_config):
        mock_get.return_value = _mock_io()

        builder = ContrastAwareSubjectBuilder(
            input_contrast=ulf_config,
            target_contrast=hf_config,
            io_strategy="nifti",
        )
        record = _make_record("/data/ulf_scan.nii.gz", "/data/hf_scan.nii.gz")
        subject = builder.build(record, swap=True)

        # HF is now input, ULF is now target
        assert subject["input_path"] == "/data/hf_scan.nii.gz"
        assert subject["target_path"] == "/data/ulf_scan.nii.gz"
        assert subject["input_contrast"] == "HF_3T"
        assert subject["target_contrast"] == "ULF_64mT"

    @patch(_IO_FACTORY_PATCH)
    def test_inference_mode_no_target(self, mock_get, ulf_config, hf_config):
        """No target_path present → subject has no 'target' key."""
        mock_get.return_value = _mock_io()

        builder = ContrastAwareSubjectBuilder(
            input_contrast=ulf_config,
            target_contrast=hf_config,
        )
        record = _make_record("/data/ulf_scan.nii.gz", target=None)
        subject = builder.build(record, swap=False)

        assert "input" in subject
        assert "target" not in subject


# ---------------------------------------------------------------------------
# ContrastAwarePairedDataset — bidirectional_mode / allow_unpaired
# ---------------------------------------------------------------------------


def _make_index(n_paired: int, n_unpaired: int) -> list[dict]:
    idx = []
    for i in range(n_paired):
        idx.append(
            {
                "primary_path": f"/data/ulf_{i}.nii.gz",
                "target_path": f"/data/hf_{i}.nii.gz",
                "file_id": f"paired_{i}",
                "pairing_status": "paired",
            }
        )
    for j in range(n_unpaired):
        idx.append(
            {
                "primary_path": f"/data/ulf_un_{j}.nii.gz",
                "target_path": None,
                "file_id": f"unpaired_{j}",
                "pairing_status": "unpaired_ulf",
            }
        )
    return idx


class TestContrastAwarePairedDataset:
    @patch(_IO_FACTORY_PATCH)
    def test_default_mode_is_ulf_to_hf(self, mock_get, ulf_config, hf_config):
        mock_get.return_value = _mock_io()

        ds = ContrastAwarePairedDataset(
            index=_make_index(2, 0),
            input_contrast=ulf_config,
            target_contrast=hf_config,
            bidirectional_mode="ulf_to_hf",
            allow_unpaired=False,
        )
        assert ds._swap is False

    @patch(_IO_FACTORY_PATCH)
    def test_hf_to_ulf_sets_swap_flag(self, mock_get, ulf_config, hf_config):
        mock_get.return_value = _mock_io()

        ds = ContrastAwarePairedDataset(
            index=_make_index(2, 0),
            input_contrast=ulf_config,
            target_contrast=hf_config,
            bidirectional_mode="hf_to_ulf",
            allow_unpaired=False,
        )
        assert ds._swap is True

    @patch(_IO_FACTORY_PATCH)
    def test_unknown_bidirectional_mode_raises_value_error(self, mock_get, ulf_config, hf_config):
        """A mis-typed bidirectional_mode must raise (NN#3, pitfall #9).

        Previously any unknown value (e.g. 'bad_mode', 'auto') silently mapped
        to swap=False — the ulf_to_hf direction — corrupting training pairs.
        """
        mock_get.return_value = _mock_io()

        with pytest.raises(ValueError, match="bidirectional_mode"):
            ContrastAwarePairedDataset(
                index=_make_index(2, 0),
                input_contrast=ulf_config,
                target_contrast=hf_config,
                bidirectional_mode="bad_mode",
                allow_unpaired=False,
            )

    @pytest.mark.parametrize("mode", ["hf_to_hf", "ulf_to_ulf"])
    @patch(_IO_FACTORY_PATCH)
    def test_autoencode_modes_raise_not_implemented(self, mock_get, ulf_config, hf_config, mode):
        """Single-field autoencode modes are honored only on the nifti_paired
        path; this two-file dataset has no self-supervised branch, so it must
        raise NotImplementedError rather than silently run as ulf_to_hf."""
        mock_get.return_value = _mock_io()

        with pytest.raises(NotImplementedError, match="nifti_paired"):
            ContrastAwarePairedDataset(
                index=_make_index(2, 0),
                input_contrast=ulf_config,
                target_contrast=hf_config,
                bidirectional_mode=mode,
                allow_unpaired=False,
            )

    @patch(_IO_FACTORY_PATCH)
    def test_len_includes_unpaired_when_allowed(self, mock_get, ulf_config, hf_config):
        mock_get.return_value = _mock_io()

        index = _make_index(n_paired=3, n_unpaired=2)
        ds = ContrastAwarePairedDataset(
            index=index,
            input_contrast=ulf_config,
            target_contrast=hf_config,
            allow_unpaired=True,
        )
        assert len(ds) == 5

    @patch(_IO_FACTORY_PATCH)
    def test_unpaired_record_in_a_paired_split_raises(self, mock_get, ulf_config, hf_config):
        """A target-less record under allow_unpaired=False raises; it used to be
        silently replaced by the next record (a second, silent admission policy)."""
        from spectramr.data.batch_types import TargetlessSampleError

        mock_get.return_value = _mock_io()
        ds = ContrastAwarePairedDataset(
            index=_make_index(n_paired=1, n_unpaired=1)[::-1],  # unpaired first
            input_contrast=ulf_config,
            target_contrast=hf_config,
            allow_unpaired=False,
            skip_nan_samples=False,
        )
        with pytest.raises(TargetlessSampleError, match="unpaired_0"):
            ds[0]
        assert ds[1]["file_id"] == "paired_0"

    @patch(_IO_FACTORY_PATCH)
    def test_has_target_is_stamped_for_the_loop_guard(self, mock_get, ulf_config, hf_config):
        """allow_unpaired=True keeps the record and stamps has_target=False, so
        reject_targetless_samples refuses it in a supervised step."""
        from torch.utils.data import default_collate

        from spectramr.data.batch_types import TargetlessSampleError, reject_targetless_samples

        mock_get.return_value = _mock_io()
        ds = ContrastAwarePairedDataset(
            index=_make_index(n_paired=1, n_unpaired=1),
            input_contrast=ulf_config,
            target_contrast=hf_config,
            allow_unpaired=True,
            skip_nan_samples=False,
        )
        paired, unpaired = ds[0], ds[1]
        assert bool(paired["has_target"]) is True
        assert bool(unpaired["has_target"]) is False and "target" not in unpaired
        reject_targetless_samples(default_collate([{"has_target": paired["has_target"]}]), strategy="s")
        with pytest.raises(TargetlessSampleError):
            reject_targetless_samples(
                default_collate([{"has_target": unpaired["has_target"]}]), strategy="s"
            )
