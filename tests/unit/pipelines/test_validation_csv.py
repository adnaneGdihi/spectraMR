"""Unit tests for validation CSV writing in the training pipeline.

Tests the CSV writer logic for handling evolving metric keys across
validation runs, ensuring header/row consistency.
"""

import csv
from pathlib import Path


class TestValidationCSVWriter:
    """Tests for the validation CSV writer in _run_validation."""

    @staticmethod
    def _write_val_row(csv_path: Path, val_row: dict) -> None:
        """Replicate the CSV writing logic from _run_validation in train.py."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0

        if write_header:
            all_fieldnames = list(val_row.keys())
            with open(csv_path, "w", newline="") as _vf:
                _vwriter = csv.DictWriter(
                    _vf, fieldnames=all_fieldnames, extrasaction="ignore"
                )
                _vwriter.writeheader()
                _vwriter.writerow(val_row)
        else:
            with open(csv_path, newline="") as _rf:
                reader = csv.reader(_rf)
                existing_header = next(reader, [])

            new_keys = [k for k in val_row.keys() if k not in existing_header]
            if new_keys:
                all_fieldnames = existing_header + new_keys
                with open(csv_path, newline="") as _rf:
                    _rf.readline()
                    old_reader = csv.DictReader(_rf, fieldnames=existing_header)
                    existing_rows = list(old_reader)
                with open(csv_path, "w", newline="") as _vf:
                    _vwriter = csv.DictWriter(
                        _vf,
                        fieldnames=all_fieldnames,
                        extrasaction="ignore",
                        restval="",
                    )
                    _vwriter.writeheader()
                    _vwriter.writerows(existing_rows)
                    _vwriter.writerow(val_row)
            else:
                with open(csv_path, "a", newline="") as _vf:
                    _vwriter = csv.DictWriter(
                        _vf,
                        fieldnames=existing_header,
                        extrasaction="ignore",
                        restval="",
                    )
                    _vwriter.writerow(val_row)

    def test_consistent_columns_are_written_correctly(self, tmp_path):
        """Test that rows with same keys produce valid CSV."""
        csv_path = tmp_path / "validation_metrics.csv"
        row1 = {"iteration": 100, "epoch": 1, "psnr": 25.0, "ssim": 0.8}
        row2 = {"iteration": 200, "epoch": 2, "psnr": 26.0, "ssim": 0.85}

        self._write_val_row(csv_path, row1)
        self._write_val_row(csv_path, row2)

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert list(df.columns) == ["iteration", "epoch", "psnr", "ssim"]
        assert df.iloc[1]["psnr"] == 26.0

    def test_expanding_columns_are_merged(self, tmp_path):
        """Test that new keys in row 2 cause schema expansion without parse failure."""
        csv_path = tmp_path / "validation_metrics.csv"
        row1 = {"iteration": 100, "epoch": 1, "psnr": 25.0}
        row2 = {"iteration": 200, "epoch": 2, "psnr": 26.0, "noise_var": 0.05}

        self._write_val_row(csv_path, row1)
        self._write_val_row(csv_path, row2)

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "noise_var" in df.columns
        # Row 1 should have empty/NaN noise_var
        assert pd.isna(df.iloc[0]["noise_var"]) or df.iloc[0]["noise_var"] == ""
        # Row 2 should have the value
        assert df.iloc[1]["noise_var"] == 0.05

    def test_shrinking_columns_are_handled(self, tmp_path):
        """Test that fewer keys in row 2 still writes valid CSV."""
        csv_path = tmp_path / "validation_metrics.csv"
        row1 = {"iteration": 100, "epoch": 1, "psnr": 25.0, "noise_var": 0.1}
        row2 = {"iteration": 200, "epoch": 2, "psnr": 26.0}  # no noise_var

        self._write_val_row(csv_path, row1)
        self._write_val_row(csv_path, row2)

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 2
        assert "noise_var" in df.columns
        assert df.iloc[0]["noise_var"] == 0.1
        # Row 2 should have empty/NaN noise_var
        assert pd.isna(df.iloc[1]["noise_var"]) or df.iloc[1]["noise_var"] == ""

    def test_multiple_expansions(self, tmp_path):
        """Test multiple schema expansions across 3 rows."""
        csv_path = tmp_path / "validation_metrics.csv"
        row1 = {"iteration": 100, "psnr": 25.0}
        row2 = {"iteration": 200, "psnr": 26.0, "ssim": 0.8}
        row3 = {"iteration": 300, "psnr": 27.0, "ssim": 0.85, "fid": 12.3}

        self._write_val_row(csv_path, row1)
        self._write_val_row(csv_path, row2)
        self._write_val_row(csv_path, row3)

        import pandas as pd

        df = pd.read_csv(csv_path)
        assert len(df) == 3
        assert set(df.columns) == {"iteration", "psnr", "ssim", "fid"}
        assert df.iloc[2]["fid"] == 12.3


class TestWriteValidationCsvRow:
    """The production writer, ``train._write_validation_csv_row``."""

    def test_unchanged_keys_never_read_data_rows(self, tmp_path, monkeypatch):
        from spectramr.pipelines import train

        path = str(tmp_path / "validation_metrics.csv")
        for it in (1, 2, 3):
            train._write_validation_csv_row(path, {"iteration": it, "psnr": 20.0 + it})

        rows_read = []
        real_reader = csv.DictReader

        def counting(*args, **kwargs):
            for row in real_reader(*args, **kwargs):
                rows_read.append(row)
                yield row

        monkeypatch.setattr(train.csv, "DictReader", counting)
        train._write_validation_csv_row(path, {"iteration": 4, "psnr": 24.0})

        assert rows_read == []
        with open(path, newline="") as f:
            assert len(list(csv.DictReader(f))) == 4

    def test_new_key_rewrites_with_merged_header(self, tmp_path):
        from spectramr.pipelines import train

        path = tmp_path / "validation_metrics.csv"
        train._write_validation_csv_row(str(path), {"iteration": 1, "psnr": 20.0})
        train._write_validation_csv_row(str(path), {"iteration": 2, "psnr": 21.0, "kid": 0.5})

        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert list(rows[0].keys()) == ["iteration", "psnr", "kid"]
        assert rows[0]["kid"] == "" and rows[1]["kid"] == "0.5"

    def test_failure_raises(self, tmp_path):
        import pytest

        from spectramr.pipelines import train

        blocker = tmp_path / "file"
        blocker.write_text("x")
        with pytest.raises(OSError):
            train._write_validation_csv_row(str(blocker / "v.csv"), {"iteration": 1})
