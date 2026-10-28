"""Unit tests for MetricsTracker image saving functionality.

This test suite verifies that:
1. Images are saved when save_images=True
2. Images are NOT saved when save_images=False (intended behavior)
3. Returned paths are correct
4. Directories are created properly
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from spectramr.infrastructure.services.metrics_tracker import MetricsTracker


@pytest.fixture(autouse=True)
def _synchronous_image_writes(monkeypatch):
    """These tests inspect files right after saving; the writer is asynchronous."""
    real = MetricsTracker.save_images_batch

    def save_then_drain(self, *args, **kwargs):
        try:
            return real(self, *args, **kwargs)
        finally:
            self.drain()

    monkeypatch.setattr(MetricsTracker, "save_images_batch", save_then_drain)


class TestMetricsTrackerImageSaving:
    """Test suite for MetricsTracker image saving."""

    @pytest.fixture
    def temp_metrics_dir(self):
        """Create temporary metrics directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def metrics_tracker_enabled(self, temp_metrics_dir):
        """Create MetricsTracker with image saving ENABLED."""
        return MetricsTracker(
            output_dir=str(temp_metrics_dir),
            save_images=True,  # ✅ Explicitly enabled
            image_format="png",
            device="cpu",
            model_type="test_model",
        )

    @pytest.fixture
    def metrics_tracker_disabled(self, temp_metrics_dir):
        """Create MetricsTracker with image saving DISABLED."""
        return MetricsTracker(
            output_dir=str(temp_metrics_dir),
            save_images=False,  # ❌ Explicitly disabled
            image_format="png",
            device="cpu",
            model_type="test_model",
        )

    def test_image_directories_created(self, metrics_tracker_enabled, temp_metrics_dir):
        """Test that image directories are created on initialization."""
        assert (temp_metrics_dir / "real_images").exists()
        assert (temp_metrics_dir / "fake_images").exists()
        assert (temp_metrics_dir / "images").exists()

    def test_save_images_batch_when_enabled(self, metrics_tracker_enabled):
        """Test that images ARE saved when save_images=True."""
        real_img = torch.randn(2, 3, 64, 64)
        fake_img = torch.randn(2, 3, 64, 64)

        real_paths, fake_paths = metrics_tracker_enabled.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="test",
            epoch=0,
            step=100,
        )

        # ✅ Should return paths
        assert len(real_paths) == 2
        assert len(fake_paths) == 2
        assert all(isinstance(p, str) for p in real_paths)
        assert all(isinstance(p, str) for p in fake_paths)

        # ✅ Files should actually exist on disk
        for path in real_paths:
            assert Path(path).exists(), f"Real image file not found: {path}"
        for path in fake_paths:
            assert Path(path).exists(), f"Fake image file not found: {path}"

    def test_save_images_batch_when_disabled(self, metrics_tracker_disabled):
        """Test that images are NOT saved when save_images=False (intended behavior)."""
        real_img = torch.randn(2, 3, 64, 64)
        fake_img = torch.randn(2, 3, 64, 64)

        real_paths, fake_paths = metrics_tracker_disabled.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="test",
            epoch=0,
            step=100,
        )

        # ✅ Should return EMPTY lists (not an error, intended behavior)
        assert real_paths == []
        assert fake_paths == []

    def test_save_images_batch_returns_correct_filenames(self, metrics_tracker_enabled):
        """Test that returned filenames follow expected pattern."""
        real_img = torch.randn(1, 3, 32, 32)
        fake_img = torch.randn(1, 3, 32, 32)

        real_paths, fake_paths = metrics_tracker_enabled.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="val",
            epoch=5,
            step=1000,
        )

        # ✅ Filenames should include prefix, epoch, step
        real_path = real_paths[0]
        fake_path = fake_paths[0]

        assert "val" in real_path
        assert "epoch005" in real_path
        assert "step001000" in real_path
        assert "real" in real_path

        assert "val" in fake_path
        assert "epoch005" in fake_path
        assert "step001000" in fake_path
        assert "fake" in fake_path

    def test_save_images_batch_without_epoch_step(self, metrics_tracker_enabled):
        """Test saving images without epoch/step (uses timestamp)."""
        real_img = torch.randn(1, 3, 32, 32)
        fake_img = torch.randn(1, 3, 32, 32)

        real_paths, fake_paths = metrics_tracker_enabled.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="snapshot",
        )

        # ✅ Should still save files with timestamp
        assert len(real_paths) == 1
        assert len(fake_paths) == 1
        assert Path(real_paths[0]).exists()
        assert Path(fake_paths[0]).exists()

    def test_save_images_handles_batches(self, metrics_tracker_enabled):
        """Test that different batch sizes are handled correctly."""
        for batch_size in [1, 2, 4, 8]:
            real_img = torch.randn(batch_size, 3, 32, 32)
            fake_img = torch.randn(batch_size, 3, 32, 32)

            real_paths, fake_paths = metrics_tracker_enabled.save_images_batch(
                real_images=real_img,
                fake_images=fake_img,
                prefix=f"batch{batch_size}",
                epoch=0,
                step=0,
                max_images=batch_size,  # Override default cap to test full batch handling
            )

            # ✅ Number of returned paths should match batch size
            assert len(real_paths) == batch_size
            assert len(fake_paths) == batch_size

    def test_save_images_with_different_channels(self, metrics_tracker_enabled):
        """Test saving images with 1 and 3 channels (supported by PIL)."""
        # MetricsTracker supports only 1 (grayscale) and 3 (RGB) channels
        for channels in [1, 3]:
            real_img = torch.randn(1, channels, 32, 32)
            fake_img = torch.randn(1, channels, 32, 32)

            # Should not raise error for supported channel counts
            real_paths, fake_paths = metrics_tracker_enabled.save_images_batch(
                real_images=real_img,
                fake_images=fake_img,
                prefix=f"ch{channels}",
                epoch=0,
                step=0,
            )

            assert len(real_paths) == 1
            assert len(fake_paths) == 1

    def test_save_images_device_agnostic(self):
        """Test that save_images works regardless of tensor device."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                output_dir=tmpdir,
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

            # Test with CPU tensors
            real_cpu = torch.randn(1, 3, 32, 32)
            fake_cpu = torch.randn(1, 3, 32, 32)

            real_paths, fake_paths = tracker.save_images_batch(
                real_images=real_cpu,
                fake_images=fake_cpu,
                prefix="cpu_test",
                epoch=0,
                step=0,
            )

            assert len(real_paths) == 1
            assert len(fake_paths) == 1
            assert Path(real_paths[0]).exists()


class TestMetricsTrackerImageSavingWithLogging:
    """Test integration with logging service."""

    def test_save_images_logs_info_on_success(self):
        """Test that save_images_batch doesn't hide errors in logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = MetricsTracker(
                output_dir=tmpdir,
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

            real_img = torch.randn(1, 3, 32, 32)
            fake_img = torch.randn(1, 3, 32, 32)

            # Should complete without error
            real_paths, fake_paths = tracker.save_images_batch(
                real_images=real_img,
                fake_images=fake_img,
                prefix="test",
                epoch=0,
                step=0,
            )

            # Verify files were actually created (not silently skipped)
            assert len(real_paths) > 0
            assert len(fake_paths) > 0
            for p in real_paths + fake_paths:
                assert Path(p).exists()


class TestMetricsTrackerMultiChannelImageSaving:
    """Test multi-channel tensor image saving (RSS magnitude fallback).

    Verifies that the image saver handles tensors with >3 channels
    by computing RSS magnitude instead of raising ValueError.
    """

    @pytest.fixture
    def tracker(self):
        """Create MetricsTracker with image saving enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MetricsTracker(
                output_dir=tmpdir,
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

    def test_4_channel_kspace_saves_as_grayscale(self, tracker):
        """4-channel (2-coil complex) k-space data should save."""
        real_img = torch.randn(1, 4, 32, 32)
        fake_img = torch.randn(1, 4, 32, 32)
        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="multi_ch",
            epoch=0,
            step=0,
        )
        assert len(fake_paths) == 1
        assert Path(fake_paths[0]).exists()

    def test_8_channel_multi_coil_saves(self, tracker):
        """8-channel (4-coil complex) data should save."""
        real_img = torch.randn(1, 8, 32, 32)
        fake_img = torch.randn(1, 8, 32, 32)
        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="multi_ch",
            epoch=0,
            step=0,
        )
        assert len(fake_paths) == 1
        assert Path(fake_paths[0]).exists()

    def test_16_channel_saves(self, tracker):
        """16-channel data should save via RSS fallback."""
        real_img = torch.randn(1, 16, 32, 32)
        fake_img = torch.randn(1, 16, 32, 32)
        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="multi_ch",
            epoch=0,
            step=0,
        )
        assert len(fake_paths) == 1
        assert Path(fake_paths[0]).exists()

    def test_batch_of_multi_channel_saves_all(self, tracker):
        """Batch of 4-channel images should save all samples."""
        real_img = torch.randn(3, 4, 32, 32)
        fake_img = torch.randn(3, 4, 32, 32)
        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="batch_multi",
            epoch=0,
            step=0,
        )
        assert len(fake_paths) == 3
        for p in fake_paths:
            assert Path(p).exists()

    def test_2_channel_paired_modality_saves_via_rss(self, tracker):
        """Regression: 2-channel tensors (e.g. ULF/HF paired stacks)
        previously fell into a special branch that treated channel 0 as
        ``real`` and channel 1 as ``imag`` and computed
        ``sqrt(t[0]^2 + t[1]^2)``. For paired modalities that mixed the
        two contrasts into a single magnitude — the May 2026 ULF
        "doubled and odd" symptom. The branch is now removed: 2-channel
        tensors flow through the same multi-channel RSS path as 4 / 8 /
        16-channel tensors.
        """
        # Construct a tensor where channel 0 and channel 1 carry distinct
        # spatial structures so RSS produces a non-degenerate output.
        ch0 = torch.zeros(1, 1, 32, 32)
        ch0[..., :16, :] = 0.6
        ch1 = torch.zeros(1, 1, 32, 32)
        ch1[..., :, :16] = 0.4
        real_img = torch.cat([ch0, ch1], dim=1)  # [1, 2, 32, 32]
        fake_img = real_img.clone()
        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real_img,
            fake_images=fake_img,
            prefix="paired_2ch",
            epoch=0,
            step=0,
        )
        assert len(real_paths) == 1
        assert Path(real_paths[0]).exists()
        # Open the PNG and confirm it is NOT all-zero / NOT all-saturated
        # — the RSS of two non-overlapping rectangular masks should yield
        # a structured grayscale image.
        import numpy as np
        from PIL import Image as _PIL

        with _PIL.open(real_paths[0]) as arr:
            pixels = np.asarray(arr).flatten()
        unique_values = len(np.unique(pixels))
        assert unique_values >= 2, "RSS of paired modalities must preserve structure"


class TestWhatABlackPngActuallyMeans:
    """Which predictions save as a bit-exact black PNG, and which do not.

    A black `fake_*.png` beside a clean `real_*.png` is the single most common
    "the model collapsed" report on the cold-diffusion arms, and it is read as a
    magnitude problem: the prediction is off-scale, so the image saturates away.

    That reading is wrong, and it sent the 2026-08-16 experiment_11 triage down
    a false mechanism before these cases were measured. `_normalize_images` does
    percentile windowing (0.5% / 99.5%, foreground-aware), NOT min-max, so an
    out-of-range but FINITE prediction still renders across the full 0-255
    range. Bit-exact black is a much narrower signal than "too big".

    2026-08-17: narrower still, and now always reported. The first version of
    this class pinned FOUR black cases and thereby documented two renderer bugs
    as expected behaviour:

    * a sample containing a single NaN went black, because `torch.quantile`
      propagates NaN through the whole window — one bad pixel destroyed 65k
      good ones. The window is now taken over finite pixels only;
    * a healthy sample whose dynamic range fell below an ABSOLUTE `1e-8` floor
      went black, so the same image in smaller units rendered differently. The
      degeneracy test is now relative.

    What remains black is only what carries no renderable content — all-zero,
    constant, all-NaN — and each of those now emits a WARNING, because a
    black PNG is otherwise indistinguishable from a legitimate render and
    nothing downstream registers the failure (pitfall #16).

    2026-08-18: that taxonomy gained a FIFTH, legitimate cause, and it is not
    visible from this class. Every case here saves the same tensor as both real
    and fake and asserts on the REAL png, which is always windowed by its own
    percentiles. The fake is now drawn under the real's window instead, so a
    perfectly healthy, non-constant prediction lying entirely below that window
    clamps to bit-exact black — and entirely above, to solid white. Neither is
    constant nor non-finite, so none of the tests below can see it. It is
    detected and reported separately; see
    ``TestTheFakeIsRenderedUnderTheRealsWindow``. Read a black *fake* against
    the shared window recorded in its sidecar, not against this list.
    """

    @pytest.fixture
    def tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MetricsTracker(
                output_dir=tmpdir,
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

    @staticmethod
    def _png(tracker, image):
        import numpy as np
        from PIL import Image

        paths, _ = tracker.save_images_batch(
            real_images=image, fake_images=image, prefix="blk", epoch=0, step=0
        )
        with Image.open(paths[0]) as arr:
            return np.asarray(arr)

    @pytest.mark.parametrize(
        ("name", "factory"),
        [
            ("all_zero", lambda: torch.zeros(1, 1, 32, 32)),
            ("constant", lambda: torch.full((1, 1, 32, 32), 3.7)),
            ("all_nan", lambda: torch.full((1, 1, 32, 32), float("nan"))),
        ],
    )
    def test_a_genuinely_degenerate_sample_saves_as_bit_exact_black(self, tracker, name, factory):
        """These three carry no renderable content, so black is honest.

        ``contains_nan`` used to be the fourth entry here and is not any more —
        see ``test_one_nan_pixel_no_longer_blacks_out_the_whole_sample``.
        """
        pixels = self._png(tracker, factory())
        assert pixels.max() == 0, f"{name} should be bit-exact black, got max {pixels.max()}"

    @pytest.mark.parametrize(
        ("name", "factory"),
        [
            ("all_zero", lambda: torch.zeros(1, 1, 32, 32)),
            ("constant", lambda: torch.full((1, 1, 32, 32), 3.7)),
            ("all_nan", lambda: torch.full((1, 1, 32, 32), float("nan"))),
        ],
    )
    def test_a_black_render_is_always_reported(self, tracker, caplog, name, factory):
        """A black PNG must never be the ONLY trace of a degenerate sample.

        The artifact is indistinguishable from a legitimate render, so without
        this warning nothing downstream registers a failure — pitfall #16. It
        is asserted at WARNING because ``LoggingService.setup`` clamps every
        logger to ``logging.sinks.level``, and the cold-diffusion arms that hit
        this path run at ``warning``: an INFO diagnostic would be discarded
        before reaching the log.
        """
        import logging as _logging

        with caplog.at_level(_logging.WARNING):
            self._png(tracker, factory())
        warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING]
        assert warnings, f"{name} rendered black with no warning emitted"
        assert any("Degenerate image render" in r.getMessage() for r in warnings)

    def test_one_nan_pixel_no_longer_blacks_out_the_whole_sample(self, tracker):
        """The experiment_11 blackout: 1 bad pixel destroyed 65k good ones.

        ``torch.quantile`` propagates NaN, so a single non-finite entry made
        the whole percentile window NaN, every pixel normalised to NaN, and
        ``(NaN * 255).astype(np.uint8)`` — undefined behaviour — landed on 0.
        The window is now computed over finite pixels only, so the anatomy
        survives and only the bad pixels go black.
        """
        torch.manual_seed(0)
        image = torch.rand(1, 1, 32, 32) + 0.5
        image[0, 0, 5, 5] = float("nan")

        pixels = self._png(tracker, image)
        assert pixels.max() == 255, "a single NaN must not black out an otherwise healthy sample"
        assert len(set(pixels.flatten().tolist())) > 2, "structure must survive"

    @pytest.mark.parametrize("sign", ["+", "-"])
    def test_an_infinite_pixel_renders_black_not_white(self, tracker, sign):
        """A diverged pixel must not read as the brightest anatomy in the image.

        `((sample - vmin) / rng).clamp(0, 1)` runs BEFORE `nan_to_num`, and
        clamp maps `+inf` to 1.0 and `-inf` to 0.0 — so the `posinf=1.0` /
        `neginf=0.0` arguments could never fire, and only NaN was ever caught.
        `+inf` therefore rendered WHITE while the surrounding comment and the
        method docstring both claimed non-finite pixels went to black.

        White is the worst available answer: it is indistinguishable from
        maximum real signal, so a diverged pixel reads as the brightest tissue
        in the slice — the precise misreading this render path exists to stop.
        Non-finite pixels are now masked out before the cast.
        """
        torch.manual_seed(0)
        image = torch.rand(1, 1, 32, 32) + 0.5
        image[0, 0, 5, 5] = float(f"{sign}inf")

        pixels = self._png(tracker, image)
        assert pixels[5, 5] == 0, f"{sign}inf must render black, not {pixels[5, 5]}"
        assert pixels.max() == 255, "the healthy anatomy around it must still render"

    def test_normalisation_is_scale_invariant(self, tracker):
        """The same image in smaller units must render identically.

        The degeneracy test used to compare the percentile range against an
        absolute ``1e-8``, which silently assumes every arm renders data of
        order 1. A healthy reconstruction scaled down tripped it and was
        written out as solid black.
        """
        torch.manual_seed(0)
        image = torch.rand(1, 1, 32, 32) + 0.5

        big = self._png(tracker, image)
        small = self._png(tracker, image * 1e-12)
        assert small.max() == 255, "small-magnitude data must not render black"
        assert (big == small).all(), "normalisation must be scale invariant"

    def test_a_merely_off_scale_prediction_still_renders(self, tracker):
        """The case the "collapse" reading assumes, which does NOT go black.

        Bulk at a plausible compressed-domain spread with 1% of coefficients
        driven to a decompression-clamp-sized magnitude: percentile windowing
        rejects the outliers and the anatomy still fills the range.
        """
        torch.manual_seed(0)
        image = torch.rand(1, 1, 64, 64) * 1.6
        flat = image.view(-1)
        flat[torch.randperm(flat.numel())[: flat.numel() // 100]] = 1.07e13

        pixels = self._png(tracker, image)
        assert pixels.max() == 255, (
            "an off-scale but finite prediction must still render — if this "
            "starts going black, the black-PNG signal has widened and the "
            "diagnosis above no longer holds"
        )


@pytest.mark.unit
class TestTheFakeIsRenderedUnderTheRealsWindow:
    """A comparison figure drawn under two transfer functions is not a comparison.

    ``_normalize_images`` fits its window to the tensor it is given, so windowing
    real and fake separately re-fits the affine map to whatever each side happens
    to contain. Intensity error is then absorbed by the very transform that is
    supposed to display it: a prediction 4x too bright comes out **byte-identical**
    to the ground truth (measured, below). That is pitfall #16 exactly — the
    artifact looks like evidence and is not — and it is why experiment_11's
    validation figures read as "fine" while the fake mean was 197.7 against a real
    mean of 55.1.

    The fake is now rendered under the real's window. Sharing costs one new
    failure mode, and the second half of this class is about paying for it: a
    healthy non-constant fake outside the shared window clamps flat, which the
    degeneracy tests in ``TestWhatABlackPngActuallyMeans`` cannot see because the
    sample is neither constant nor non-finite. Reporting that is what keeps this
    change from re-opening the silent black-PNG path that finite-only windowing
    closed one commit earlier.
    """

    @pytest.fixture
    def tracker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield MetricsTracker(
                output_dir=tmpdir,
                save_images=True,
                image_format="png",
                device="cpu",
                model_type="test",
            )

    @staticmethod
    def _phantom(n: int = 32, seed: int = 0) -> torch.Tensor:
        """A bright disc on a dim floor: structure, and percentiles that differ.

        Uniform noise would give an almost symmetric histogram, where the
        foreground-aware ceiling and the whole-image 99.5% coincide and a scaling
        error is hard to distinguish from a windowing artefact.
        """
        g = torch.Generator().manual_seed(seed)
        y, x = torch.meshgrid(torch.linspace(-1, 1, n), torch.linspace(-1, 1, n), indexing="ij")
        disc = ((x**2 + y**2) < 0.5).float()
        return (disc * 0.8 + 0.05 + 0.02 * torch.rand(n, n, generator=g)).reshape(1, 1, n, n)

    @staticmethod
    def _pair(tracker, real, fake, **kwargs):
        """Save one pair; return (real_px, fake_px, sidecar)."""
        import numpy as np
        from PIL import Image

        real_paths, fake_paths = tracker.save_images_batch(
            real_images=real, fake_images=fake, prefix="cmp", epoch=0, step=0, **kwargs
        )
        with Image.open(real_paths[0]) as arr:
            real_px = np.asarray(arr).astype(float)
        with Image.open(fake_paths[0]) as arr:
            fake_px = np.asarray(arr).astype(float)
        sidecar = json.loads(
            next(tracker.fake_images_dir.glob("*_render_windows.json")).read_text()
        )
        return real_px, fake_px, sidecar

    def test_a_prediction_four_times_too_bright_no_longer_renders_as_a_perfect_picture(
        self, tracker
    ):
        """The experiment_11 signature: a pure scaling error, fully invisible.

        Percentile windowing is affine and scale-covariant, so ``4 * x`` and ``x``
        produce the SAME normalised tensor when each is windowed by its own
        percentiles. The bug is not an approximation that got close — the two PNGs
        were bit-for-bit equal.
        """
        real = self._phantom()
        _, fake_px, _ = self._pair(tracker, real, real * 4.0)

        assert fake_px.min() > 0, (
            "a 4x-too-bright prediction must not render its floor as black; "
            "under the real's window its dimmest tissue sits well above vmin"
        )
        assert fake_px.mean() > 120, (
            f"the fake must read as too bright, got mean {fake_px.mean():.1f}; "
            "if this drops back to the real's ~94 the window is no longer shared"
        )

    def test_windowing_each_side_separately_is_what_hid_the_error(self, tracker):
        """Anti-vacuity: without the shared window the test above cannot fail.

        Asserted on the normaliser rather than through a monkeypatch, so it pins
        the mechanism and not a way of disabling it. If the first assertion ever
        goes red, percentile windowing has stopped being scale-covariant and the
        premise of this whole class needs re-measuring.
        """
        real = self._phantom()
        fake = real * 4.0

        per_side_real, real_windows = tracker._normalize_images_windowed(real, context="real")
        per_side_fake, _ = tracker._normalize_images_windowed(fake, context="fake")
        assert torch.allclose(per_side_real, per_side_fake, atol=1e-5), (
            "the facade being fixed: windowed separately, a 4x scaling error "
            "normalises away completely"
        )

        shared_fake, _ = tracker._normalize_images_windowed(
            fake, context="fake", reference_windows=real_windows
        )
        assert not torch.allclose(per_side_real, shared_fake, atol=1e-5), (
            "under the real's window the same error must survive into the pixels"
        )

    @pytest.mark.parametrize(
        ("name", "factor", "direction", "expected"),
        [
            ("below", 1e-4, "BELOW", 0.0),
            ("above", 1e4, "ABOVE", 255.0),
        ],
    )
    def test_a_fake_fully_outside_the_shared_window_is_reported(
        self, tracker, caplog, name, factor, direction, expected
    ):
        """The failure mode sharing introduces, and the one it must not hide.

        A healthy, non-constant, all-finite fake that lies entirely outside the
        borrowed window clamps to solid black or solid white. Nothing in the
        degeneracy tests can see it — it is neither constant nor non-finite — so
        without this report window sharing would hand back exactly the
        black-PNG-with-no-warning facade the finite-only windowing removed.

        The warning must also name the fake's OWN range: a flat render is only
        diagnosable against the dynamic range it was forced out of.
        """
        import logging as _logging

        real = self._phantom()
        with caplog.at_level(_logging.WARNING):
            _, fake_px, _ = self._pair(tracker, real, real * factor)

        assert (fake_px == expected).all(), f"{name} case should render flat at {expected}"
        messages = [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]
        assert any(f"fell {direction} the shared window" in m for m in messages), (
            f"a fully-clamped fake rendered flat with no warning: {messages}"
        )
        assert any("its own range is" in m for m in messages), (
            "the report must carry the fake's own range, or a flat render "
            "cannot be told from a correct one"
        )

    def test_a_collapsed_fake_is_reported_without_claiming_a_blackout(self, tracker, caplog):
        """Constant fake + healthy real: a real finding whose consequence is NOT black.

        Borrowed into a valid window, a constant sample renders as a flat mid-tone
        — a picture, with nothing visibly wrong. So the diagnosis must survive
        (the producer collapsed) while the consequence stays honest: nothing was
        blacked out here, and ``_warn_degenerate_render`` must not say otherwise.
        A warning that overstates its own finding is the same facade in miniature.
        """
        import logging as _logging

        real = self._phantom()
        with caplog.at_level(_logging.WARNING):
            _, fake_px, _ = self._pair(tracker, real, torch.full_like(real, 0.42))

        assert len(set(fake_px.flatten().tolist())) == 1, "a constant fake renders flat"
        assert fake_px.max() > 0, "borrowed into a valid window it is a tone, not black"
        fake_msgs = [
            r.getMessage()
            for r in caplog.records
            if r.levelno >= _logging.WARNING and "[fake/" in r.getMessage()
        ]
        assert any("rendered as a flat tone" in m for m in fake_msgs), fake_msgs
        assert any("0 sample(s) were written as black" in m for m in fake_msgs), (
            "the constant fake DID render, so the warning must not claim a blackout"
        )
        assert not any("RENDER FAILURE" in m for m in fake_msgs), (
            "'RENDER FAILURE' is reserved for a sample actually written as black"
        )

    def test_a_degenerate_real_makes_the_fake_fall_back_and_say_so(self, tracker, caplog):
        """No window to borrow: render the fake on its own, and admit the pair is not one.

        The fallback is the only way to render the fake at all, but it silently
        restores the two-transfer-function comparison this change exists to stop.
        So it has to be stated — otherwise the figure is read as a comparison it
        is not.
        """
        import logging as _logging

        real = self._phantom()
        with caplog.at_level(_logging.WARNING):
            real_px, fake_px, sidecar = self._pair(tracker, torch.zeros_like(real), real)

        assert real_px.max() == 0, "an all-zero real is still black, as before"
        assert fake_px.max() == 255, "the fake must still render, under its own window"
        messages = [r.getMessage() for r in caplog.records if r.levelno >= _logging.WARNING]
        assert any("NOT comparable" in m for m in messages), messages
        assert sidecar["samples"][0]["fake"]["source"] == "own (counterpart degenerate)"
        assert sidecar["samples"][0]["real"]["rendered"] is False

    def test_a_healthy_comparable_pair_emits_no_warning(self, tracker, caplog):
        """None of the shared-window findings may fire on a good render.

        Three of them are new WARNINGs on a path that runs at every validation
        interval for every rung. If a healthy pair tripped any of them the
        diagnostics would be tuned out inside one run, which is how the original
        black-PNG blackout stayed unexamined.
        """
        import logging as _logging

        real = self._phantom()
        with caplog.at_level(_logging.WARNING):
            real_px, fake_px, _ = self._pair(tracker, real, real.clone() * 1.05)

        assert real_px.max() == 255 and fake_px.max() == 255
        assert not [
            r.getMessage() for r in caplog.records if "Degenerate image render" in r.getMessage()
        ], "a healthy pair must render silently"

    def test_the_sidecar_records_the_shared_window_beside_the_fakes_own_range(self, tracker):
        """The window must be recoverable from the artifact, not only from the log.

        ``logging.sinks.level`` on the arms that hit this path is ``warning``, and
        a WARNING on every healthy render would be spam — so a normal shared
        render leaves no log trace at all. Without the sidecar the pixels are
        uninterpretable: ``[0, 255]`` says nothing about the intensities behind
        them, and the 4x error of the first test is unquantifiable after the run.
        """
        real = self._phantom()
        _, _, sidecar = self._pair(tracker, real, real * 4.0)

        entry = sidecar["samples"][0]
        assert entry["real"]["source"] == "own"
        assert entry["fake"]["source"] == "shared"
        assert entry["fake"]["vmin"] == entry["real"]["vmin"]
        assert entry["fake"]["vmax"] == entry["real"]["vmax"]
        # The whole point: the error is quantifiable from the JSON alone.
        assert entry["fake"]["finite_max"] > entry["fake"]["vmax"] * 3, (
            "the fake's own range must sit beside the borrowed window, or a "
            "saturated render cannot be read back"
        )

    def test_the_sidecar_describes_only_the_images_actually_written(self, tracker):
        """``max_images`` truncates the save loop; the record must truncate with it.

        A record covering the whole normalised batch would name PNGs that do not
        exist — the artifact asserting more than it holds, which is the class of
        defect this whole changeset is about.
        """
        real = torch.cat([self._phantom(seed=s) for s in range(5)], dim=0)
        _, _, sidecar = self._pair(tracker, real, real * 2.0, max_images=2)

        assert len(sidecar["samples"]) == 2, sidecar
        assert [s["index"] for s in sidecar["samples"]] == [0, 1]
        assert len(list(tracker.fake_images_dir.glob("*.png"))) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
