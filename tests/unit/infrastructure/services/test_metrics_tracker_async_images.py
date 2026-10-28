"""The MetricsTracker PNG writer runs off the calling thread; drain() is the join point."""

import threading
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.services.metrics_tracker import MetricsTracker


class TestAsyncImageWriter:
    """PNG writes run on a worker thread; drain() is the join point."""

    @staticmethod
    def _tracker(tmp_path):
        return MetricsTracker(
            output_dir=str(tmp_path), save_images=True, image_format="png", device="cpu"
        )

    def test_drain_makes_all_pngs_exist(self, tmp_path):
        tracker = self._tracker(tmp_path)
        real, fake = tracker.save_images_batch(
            torch.rand(3, 1, 16, 16), torch.rand(3, 1, 16, 16), prefix="v", epoch=0, step=1
        )
        tracker.drain()
        assert len(real) == len(fake) == 3
        assert all(Path(p).exists() for p in real + fake)
        tracker.shutdown_image_writer()

    def test_failed_save_reraises_from_drain(self, tmp_path, monkeypatch):
        tracker = self._tracker(tmp_path)

        def boom(array, mode, path):
            raise OSError("disk full")

        monkeypatch.setattr(MetricsTracker, "_write_image", staticmethod(boom))
        tracker.save_images_batch(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), prefix="v")
        with pytest.raises(OSError, match="disk full"):
            tracker.drain()
        tracker.drain()  # error is reported once, not on every later drain
        tracker.shutdown_image_writer()

    def test_next_submit_reraises_previous_error(self, tmp_path, monkeypatch):
        tracker = self._tracker(tmp_path)
        monkeypatch.setattr(
            MetricsTracker,
            "_write_image",
            staticmethod(lambda a, m, p: (_ for _ in ()).throw(OSError("bad"))),
        )
        tracker.save_images_batch(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), prefix="a")
        tracker._pending_images.exception()  # let batch "a" fail before "b" is submitted
        written: list[Path] = []
        monkeypatch.setattr(
            MetricsTracker,
            "_write_image",
            staticmethod(lambda a, m, p: written.append(Path(p))),
        )
        with pytest.raises(OSError, match="bad"):
            tracker.save_images_batch(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), prefix="b")
        tracker.drain()  # the current batch was queued, not dropped
        assert len(written) == 2
        assert all("b" in p.name for p in written)
        tracker.shutdown_image_writer()


def test_save_returns_before_the_write_completes(tmp_path, monkeypatch):
    tracker = MetricsTracker(
        output_dir=str(tmp_path), save_images=True, image_format="png", device="cpu"
    )
    release = threading.Event()
    real_write = MetricsTracker._write_image

    def gated(array, mode, path):
        assert release.wait(timeout=30)
        real_write(array, mode, path)

    monkeypatch.setattr(MetricsTracker, "_write_image", staticmethod(gated))
    real, fake = tracker.save_images_batch(
        torch.rand(2, 1, 16, 16), torch.rand(2, 1, 16, 16), prefix="v"
    )
    try:
        assert not release.is_set()
        assert not any(Path(p).exists() for p in real + fake)
    finally:
        release.set()
    tracker.drain()
    assert all(Path(p).exists() for p in real + fake)
    tracker.shutdown_image_writer()
