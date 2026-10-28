"""Unit tests for the Metrics Tracker service."""

import csv
from unittest.mock import MagicMock, patch

import pytest
import torch

from spectramr.infrastructure.services.metrics_tracker import MetricsTracker


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    @pytest.fixture
    def metrics_dir(self, tmp_path):
        """Temporary directory for metrics output."""
        metrics_out = tmp_path / "metrics_logs"
        return metrics_out

    @pytest.fixture
    def tracker(self, metrics_dir):
        """Provide a clean MetricsTracker for each test."""
        return MetricsTracker(
            output_dir=str(metrics_dir),
            save_images=True,
            image_format="png",
            device="cpu",
            model_type="test_model",
        )

    def test_initialization_creates_directories(self, tracker, metrics_dir):
        """Test that initialization creates the correct directory structure."""
        assert metrics_dir.exists()
        assert (metrics_dir / "images").exists()
        assert (metrics_dir / "real_images").exists()
        assert (metrics_dir / "fake_images").exists()

    def test_update_and_get_metrics(self, tracker):
        """Test tracking current metric state."""
        tracker.update_metric("loss_g", 0.5)
        tracker.update_metrics({"loss_d": 0.2, "psnr": 30.0})

        current = tracker.get_current_metrics()
        assert current["loss_g"] == 0.5
        assert current["loss_d"] == 0.2
        assert current["psnr"] == 30.0

    def test_reset_metrics(self, tracker):
        """Test clearing current metrics."""
        tracker.update_metric("loss_g", 0.5)
        tracker.reset_metrics()

        current = tracker.get_current_metrics()
        assert len(current) == 0

    def test_log_metrics_writes_to_csv_and_updates_schema(self, tracker, metrics_dir):
        """Test CSV writing and dynamic schema evolution (new keys)."""
        # Step 1: Log initial metric
        tracker.log_metrics({"loss_g": 0.5}, step=10, epoch=1, prefix="training")

        csv_file = metrics_dir / "training_metrics_test_model.csv"
        assert csv_file.exists()

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert "loss_g" in rows[0]
            assert "timestamp" in rows[0]
            assert rows[0]["loss_g"] == "0.5"

        # Step 2: Log new metrics and trigger schema evolution
        tracker.log_metrics(
            {"loss_d": 0.2, "loss_g": 0.3}, step=20, epoch=1, prefix="training"
        )

        with open(csv_file, newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            rows = list(reader)
            assert "loss_d" in headers
            assert "loss_g" in headers
            assert len(rows) == 2

            # The first row shouldn't have loss_d initially so it pads with empty string
            assert rows[0]["loss_d"] == ""
            assert rows[1]["loss_d"] == "0.2"
            assert rows[1]["loss_g"] == "0.3"

    @patch("spectramr.infrastructure.services.metrics_tracker.compute_metric")
    def test_compute_image_metrics_delegation(self, mock_compute_metric, tracker):
        """Test that metric computation correctly delegates to SSOT registry."""
        mock_compute_metric.return_value = 42.0

        preds = torch.randn(1, 1, 32, 32)
        targets = torch.randn(1, 1, 32, 32)

        metrics = tracker.compute_image_metrics(
            preds, targets, metric_names=["psnr", "ssim"]
        )

        # Verify it delegated to compute_metric for each requested metric
        assert mock_compute_metric.call_count == 2
        assert metrics["psnr"] == 42.0
        assert metrics["ssim"] == 42.0

    def test_normalize_images_handles_various_ranges(self, tracker):
        """Test that _normalize_images correctly detects ranges and normalizes to [0,1]."""
        # Range [-1, 1]
        t1 = torch.tensor([[[-1.0, 1.0], [0.0, -0.5]]])
        norm1 = tracker._normalize_images(t1)
        assert norm1.min() >= 0.0
        assert norm1.max() <= 1.0
        assert torch.allclose(norm1[0, 0, 0], torch.tensor(0.0))  # -1 -> 0
        assert torch.allclose(norm1[0, 0, 1], torch.tensor(1.0))  # 1 -> 1

        # Large range [0, 255]
        t2 = torch.tensor([[[0.0, 255.0], [128.0, 10.0]]])
        norm2 = tracker._normalize_images(t2)
        assert norm2.min() >= 0.0
        assert norm2.max() <= 1.0
        assert torch.allclose(norm2[0, 0, 0], torch.tensor(0.0))
        assert torch.allclose(norm2[0, 0, 1], torch.tensor(1.0))  # 255 -> 1

        # Near zero (black)
        t3 = torch.zeros(1, 2, 2)
        norm3 = tracker._normalize_images(t3)
        assert torch.all(norm3 == 0)

    @patch("spectramr.infrastructure.services.metrics_tracker.Image.fromarray")
    def test_save_images_batch(self, mock_fromarray, tracker):
        """Test saving a batch of images writes to disk correctly."""
        mock_img = MagicMock()
        mock_fromarray.return_value = mock_img

        real_images = torch.randn(2, 1, 16, 16)  # Grayscale
        fake_images = torch.randn(2, 1, 16, 16)

        real_paths, fake_paths = tracker.save_images_batch(
            real_images, fake_images, prefix="test", epoch=1, step=10, max_images=2
        )

        tracker.drain()
        # Mock Image.save should be called for each image (2 real, 2 fake)
        assert mock_img.save.call_count == 4
        assert len(real_paths) == 2
        assert len(fake_paths) == 2
        assert "real_images" in real_paths[0]
        assert "fake_images" in fake_paths[0]

    def test_save_images_batch_disabled(self, metrics_dir):
        """Test that no images are saved if save_images=False."""
        tracker = MetricsTracker(output_dir=str(metrics_dir), save_images=False)

        real_images = torch.randn(2, 1, 16, 16)
        fake_images = torch.randn(2, 1, 16, 16)

        real_paths, fake_paths = tracker.save_images_batch(real_images, fake_images)
        assert len(real_paths) == 0
        assert len(fake_paths) == 0
