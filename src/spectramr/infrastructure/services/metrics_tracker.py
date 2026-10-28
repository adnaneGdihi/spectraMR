#!/usr/bin/env python3
"""Metrics Tracker Service

Provides CSV logging and image management for metrics.
All metric computation delegates to spectramr.core.metrics (SSOT).
"""

import csv
import dataclasses
import datetime
import json
import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from spectramr.core.metrics import compute_metric, list_available
from spectramr.core.metrics.metric_directions import resolve_direction
from spectramr.domain.interfaces.service_interfaces import IMetricsService

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class _RenderWindow:
    """The transfer function one sample was actually rendered under.

    Carried out of :meth:`MetricsTracker._normalize_images_windowed` so the
    *fake* side can be drawn under the *real* side's window, and so the window
    survives into a sidecar. Without the record a saved PNG is uninterpretable
    after the fact: ``[0, 255]`` says nothing about the physical intensities it
    came from, and the run that motivated this carried
    ``logging.sinks.level: warning``, which would have discarded an INFO
    diagnostic before it reached the log.

    ``rendered=False`` marks a sample written as solid black. ``vmin``/``vmax``
    are then the window that *failed*, not one that was applied -- kept rather
    than nulled because the failing window is the diagnosis (a constant sample
    reports its constant value here). Such a record must never be borrowed as a
    reference; :meth:`MetricsTracker._normalize_images_windowed` gates on this
    flag.

    ``finite_min``/``finite_max`` are the raw extremes of the finite pixels,
    beside the percentile window rather than instead of it. That pairing is what
    makes a saturated render readable without re-running: a fake clamped flat
    against a borrowed window is only diagnosable if its own dynamic range is
    recorded next to the window it was forced through. They are ``None`` only
    when the sample carried no finite pixel at all.
    """

    vmin: float
    vmax: float
    source: str
    rendered: bool
    finite_min: float | None
    finite_max: float | None

    def can_be_lent(self) -> bool:
        """Whether this window may be used to render a counterpart tensor.

        Only a window that was actually applied and has non-zero width can draw
        anything. A ``rendered=False`` record carries the window that *failed*,
        so lending it would divide by a zero range and write a second black PNG
        while the sidecar claimed a shared window had been used -- the artifact
        asserting a comparison that never happened.
        """
        return self.rendered and self.vmax > self.vmin


def _metric_higher_is_better(key: str) -> bool | None:
    """Resolve best-direction from the metric SSOT, or ``None`` if undeclared.

    Uses the **non-strict** resolver
    (:func:`spectramr.core.metrics.metric_directions.resolve_direction`) because
    the call sites here fold *every numeric key in a training record* — including
    non-metric columns a strategy happens to log (``epoch``, ``lr``,
    ``grad_norm``). Those have no meaningful "best" and must be skipped, not
    guessed: the old resolver assumed higher-is-better for anything without
    "loss"/"error" in its name, which is how a "best grad_norm" could be recorded
    under an invented direction.

    Decision paths (checkpoint retention, early stopping, leaderboard ranking)
    deliberately use the strict ``metric_higher_is_better`` instead — there an
    assumed direction silently retains the wrong weights (#208).
    """
    return resolve_direction(key)


class MetricsTracker(IMetricsService):
    """Metrics tracking service with CSV logging and image management.

    Delegates all metric computation to spectramr.core.metrics (SSOT) registry.
    Handles:
    - CSV file logging with schema evolution
    - Image saving for real/fake/reconstructed samples
    - Metrics history tracking
    - Metric aggregation and reporting
    """

    def __init__(
        self,
        output_dir: str = "./metrics_logs",
        save_images: bool = True,
        image_format: str = "png",
        device: str = "auto",
        model_type: str = "model",
        history_maxlen: int | None = 50_000,
    ):
        """Initialize metrics tracker.

        Args:
            output_dir: Directory to save metrics and images
            save_images: Whether to save generated images
            image_format: Format for saved images (png, jpg)
            device: Device for computation ("auto", "cuda", "cpu")
            model_type: Model type identifier for filename generation
            history_maxlen: Cap on in-memory metric records per stream (a long
                run logging every step would otherwise grow these lists without
                bound). ``None`` keeps the old unbounded behavior. Best-metric
                tracking is incremental and unaffected by eviction.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_type = model_type
        self.save_images = save_images
        self.image_format = image_format

        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Create subdirectories for images. Only the two that are written to:
        # a third ``images/`` dir was created here and never written by any code
        # in src/, scripts/ or tests/, so every run shipped an empty directory
        # that reads as "the images failed to save" rather than "nothing was ever
        # meant to land here".
        self.real_images_dir = self.output_dir / "real_images"
        self.fake_images_dir = self.output_dir / "fake_images"

        if self.save_images:
            self.real_images_dir.mkdir(exist_ok=True)
            self.fake_images_dir.mkdir(exist_ok=True)

        # CSV file paths
        self.training_csv = self.output_dir / f"training_metrics_{self.model_type}.csv"
        self.inference_csv = self.output_dir / f"inference_metrics_{self.model_type}.csv"
        self.validation_csv = self.output_dir / f"validation_metrics_{self.model_type}.csv"
        #: Tall cascading-acceleration sweep (#697): one row per
        #: (iteration, severity point), with the level and timestep as VALUES
        #: rather than encoded in column names. Its own file because
        #: `training_metrics.csv` is one row per ITERATION -- making that tall
        #: would repeat every training loss once per level.
        self.cascading_csv = self.output_dir / f"cascading_validation_{self.model_type}.csv"

        # Metrics history — bounded so long runs don't grow RSS without limit.
        from collections import deque

        self.training_history: deque[dict[str, Any]] = deque(maxlen=history_maxlen)
        self.inference_history: deque[dict[str, Any]] = deque(maxlen=history_maxlen)
        self.validation_history: deque[dict[str, Any]] = deque(maxlen=history_maxlen)

        # Incremental running-best over the FULL training stream, independent of
        # the bounded history above (so a best isn't lost when its record is
        # evicted). ``get_best_metrics`` merges this with a scan of whatever is
        # currently in ``training_history``.
        self._best_metrics: dict[str, float] = {}

        # Current metrics state
        self.current_metrics: dict[str, Any] = {}

        # PNG encode + write run on one worker thread so validation does not
        # wait on the disk. At most one batch is in flight: the next submit
        # waits for it, which bounds the held arrays and surfaces a failed
        # write at the following save rather than never.
        self._image_executor: ThreadPoolExecutor | None = None
        self._pending_images: Future[None] | None = None
        self._image_lock = threading.Lock()

        # CSV headers cache (IOPS reduction)
        self._csv_headers_cache: dict[Path, list[str]] = {}

        logger.info("[OK] MetricsTracker initialized")
        logger.info(f"   Output directory: {self.output_dir}")
        logger.info(f"   Device: {self.device}")
        logger.info(f"   Available metrics (SSOT): {', '.join(list_available()[:5])}...")

    def update_metrics(self, metrics: dict[str, Any]) -> None:
        """Update current metrics state.

        Args:
            metrics: Dictionary of metrics to update
        """
        self.current_metrics.update(metrics)

    def update_metric(self, key: str, value: Any) -> None:
        """Update a single metric value.

        Args:
            key: Metric key/name
            value: Metric value
        """
        self.current_metrics[key] = value

    def set_gauge(self, key: str, value: Any) -> None:
        """Set a gauge metric value (alias for update_metric).

        Args:
            key: Metric key/name
            value: Metric value
        """
        self.update_metric(key, value)

    def log_metrics(
        self,
        metrics_or_step: dict[str, Any] | int,
        step: int | None = None,
        epoch: int | None = None,
        prefix: str = "validation",
    ) -> None:
        """Log metrics to CSV.

        Args:
            metrics_or_step: Either metrics dict or step number
            step: Step number (if metrics_or_step is dict)
            epoch: Epoch number
            prefix: Log prefix (training/inference/validation)
        """
        # Handle both calling patterns
        if isinstance(metrics_or_step, int):
            metrics = self.current_metrics.copy()
            step = metrics_or_step
        else:
            metrics = metrics_or_step
            step = step or -1

        # Ensure step is set
        if step is None:
            step = -1

        # Create combined metrics dictionary
        log_data = {
            "timestamp": datetime.datetime.now().isoformat(),
            "epoch": epoch if epoch is not None else -1,
            "step": step,
            "model_type": self.model_type,
        }
        log_data.update(metrics)

        # Determine which CSV to write to based on prefix
        if prefix == "training":
            csv_file = self.training_csv
        elif prefix == "inference":
            csv_file = self.inference_csv
        else:
            csv_file = self.validation_csv

        self._append_to_csv(csv_file, log_data)
        logger.debug(f"📊 {prefix.capitalize()} metrics logged: {list(metrics.keys())}")

    def get_latest_metrics(self) -> dict[str, Any]:
        """Get latest metrics.

        Returns:
            Copy of current metrics dictionary
        """
        return self.current_metrics.copy()

    def get_current_metrics(self) -> dict[str, float]:
        """Get current metrics (IMetricsService implementation).

        Returns:
            Current metrics dictionary
        """
        return self.current_metrics.copy()

    def reset_metrics(self) -> None:
        """Reset metrics to empty state."""
        self.current_metrics.clear()
        logger.debug("📊 Metrics reset to empty state")

    def save_images_batch(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        prefix: str = "",
        epoch: int | None = None,
        step: int | None = None,
        max_images: int = 4,
    ) -> tuple[list[str], list[str]]:
        """Save batch of real and fake images.

        Args:
            real_images: Tensor of real images [B, C, H, W]
            fake_images: Tensor of fake images [B, C, H, W]
            prefix: Prefix for saved files
            epoch: Current epoch number
            step: Current step number
            max_images: Maximum number of images to save from this batch (default 4)

        Returns:
            Tuple of (real_images_paths, fake_images_paths)
            Returns empty lists if save_images=False (as configured in bootstrap)
        """
        # ✅ SSOT: save_images flag is set in bootstrap from config.logging.save_validation_images
        # If False, no images are saved (by design - user configured it this way)
        if not self.save_images:
            # This is NOT a failure - user explicitly disabled image saving in config
            logger.debug("[MetricsTracker] save_images=False, skipping image save (by design)")
            return [], []

        logger.info(
            f"[MetricsTracker] save_images_batch: save_images=True, prefix={prefix}, epoch={epoch}, step={step}, real_shape={real_images.shape}, fake_shape={fake_images.shape}"
        )

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create unique identifier
        if epoch is not None and step is not None:
            identifier = f"{prefix}_epoch{epoch:03d}_step{step:06d}_{timestamp}"
        else:
            identifier = f"{prefix}_{timestamp}"

        real_paths = []
        fake_paths = []

        # Normalize images to [0, 1] range
        # Tag each side so a degenerate-render warning names which tensor
        # failed. "real" and "fake" go through identical code, so without the
        # tag the log cannot distinguish a diverged prediction from a broken
        # ground-truth load — the exact ambiguity that left experiment_11's
        # all-black fakes undiagnosed while its reals rendered perfectly.
        #
        # The fake is rendered under the REAL's window, not its own. Windowing
        # each side by its own percentiles makes the pair incomparable: a
        # prediction 4x too bright renders as a flawless picture because the
        # transfer function absorbs precisely the error the figure exists to
        # show (pitfall #16 -- the artifact looks like evidence and is not).
        # Measured on the one experiment_11 batch that did render: fake mean
        # 197.7 against real mean 55.1, i.e. the pair was drawn under two
        # different transfer functions and could not be compared by eye.
        real_normalized, real_windows = self._normalize_images_windowed(
            real_images, context=f"real/{prefix}" if prefix else "real"
        )
        fake_normalized, fake_windows = self._normalize_images_windowed(
            fake_images,
            context=f"fake/{prefix}" if prefix else "fake",
            reference_windows=real_windows,
        )

        logger.debug(
            f"[MetricsTracker] Saving images to: real={self.real_images_dir}, fake={self.fake_images_dir}"
        )

        # Rasterise on this thread (``.cpu()`` and the finite check belong to the
        # caller's device context); only numpy/PIL work is handed to the worker.
        jobs: list[tuple[np.ndarray, str, Path]] = []
        for i, (real_img, fake_img) in enumerate(
            zip(real_normalized, fake_normalized, strict=False)
        ):
            if i >= max_images:
                break
            real_path = self.real_images_dir / f"{identifier}_real_{i:03d}.{self.image_format}"
            fake_path = self.fake_images_dir / f"{identifier}_fake_{i:03d}.{self.image_format}"
            try:
                for img, path in ((real_img, real_path), (fake_img, fake_path)):
                    array, mode = self._rasterize_tensor(img, path)
                    jobs.append((array, mode, path))
            except Exception as e:
                logger.error(f"[MetricsTracker] Failed to rasterise image {i}: {e}")
                raise
            real_paths.append(str(real_path))
            fake_paths.append(str(fake_path))

        self._submit_image_writes(jobs)

        self._write_render_windows(identifier, len(fake_paths), real_windows, fake_windows)

        logger.info(
            f"[MetricsTracker] Successfully saved {len(real_paths)} real + {len(fake_paths)} fake images"
        )
        return real_paths, fake_paths

    def _write_render_windows(
        self,
        identifier: str,
        n_written: int,
        real_windows: list[_RenderWindow | None],
        fake_windows: list[_RenderWindow | None],
    ) -> None:
        """Record the transfer functions a saved pair was drawn under.

        One JSON per ``save_images_batch`` call, sharing the ``identifier`` stem
        with the PNGs it describes, so ``<stem>_fake_007.png`` maps to entry 7
        of ``<stem>_render_windows.json`` without ambiguity. Per call and not
        per image on purpose: cascading validation saves every rung at every
        interval, and a sidecar per PNG would double the file count of the run
        directory for no extra information.

        The window has to be recoverable **from the artifact**, not only from
        the log. Sharing a window is what makes the pair comparable, and it is
        also what makes a fake render meaningless on its own -- ``[0, 255]`` in
        the PNG says nothing about the intensities behind it. A log line cannot
        carry that: ``LoggingService.setup`` clamps every logger to
        ``logging.sinks.level``, the arms on this path run at ``warning``, and a
        WARNING on every healthy render would be pure spam.

        Written next to the fake images because the fake is the side that
        borrows, so a reader staring at a suspicious fake finds the record in
        the same directory. Safe against the two readers that scan these dirs
        (``reporting/cases/loader.py``, ``reporting/plotters/contact_sheet.py``)
        -- both glob ``*.png`` specifically.

        Failure to write is logged, never raised: the images are already on disk
        and losing their provenance must not lose them too. It is a ``warning``
        rather than a ``debug`` because a missing sidecar means the next reader
        cannot interpret the renders.
        """
        if not self.save_images or n_written <= 0:
            return

        def entry(window: _RenderWindow | None) -> dict[str, Any] | None:
            return None if window is None else dataclasses.asdict(window)

        # Only the samples actually written: ``zip(..., strict=False)`` stops at
        # the shorter side and ``max_images`` truncates further, so recording
        # the full normalized batch would describe PNGs that do not exist.
        record = {
            "identifier": identifier,
            "samples": [
                {
                    "index": i,
                    "real": entry(real_windows[i] if i < len(real_windows) else None),
                    "fake": entry(fake_windows[i] if i < len(fake_windows) else None),
                }
                for i in range(n_written)
            ],
        }
        path = self.fake_images_dir / f"{identifier}_render_windows.json"
        try:
            path.write_text(json.dumps(record, indent=2))
        except OSError as exc:
            logger.warning(
                "[MetricsTracker] Could not write render-window sidecar %s: %s. "
                "The PNGs were saved but their transfer functions are now "
                "unrecoverable, so intensity error in them cannot be read back.",
                path,
                exc,
            )

    def compute_image_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        metric_names: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute image quality metrics using SSOT registry.

        Args:
            predictions: Model predictions [B, C, H, W]
            targets: Ground truth targets [B, C, H, W]
            metric_names: List of metric names to compute. If None, uses defaults.

        Returns:
            Dictionary of metric name -> value
        """
        if metric_names is None:
            # Default metrics (fast ones)
            metric_names = ["psnr", "ssim", "mae"]

        metrics = {}

        for metric_name in metric_names:
            try:
                # Use SSOT registry to compute metric
                value = compute_metric(
                    metric_name,
                    predictions,
                    targets,
                    device=self.device,
                )
                metrics[metric_name] = float(value) if value is not None else 0.0
                logger.debug(f"✓ Computed {metric_name}: {metrics[metric_name]:.4f}")
            except KeyError:
                logger.debug(f"⚠ Metric '{metric_name}' not in registry")
            except Exception as e:
                # Do NOT substitute 0.0 — for higher-is-better metrics (PSNR/SSIM)
                # a fake 0.0 is a catastrophic value that corrupts running
                # averages and best-checkpoint selection. Omit the failed metric
                # so consumers see it as absent rather than collapsed.
                logger.warning(f"✗ Failed to compute {metric_name}: {e}")

        return metrics

    def log_training_metrics(
        self,
        epoch: int,
        step: int,
        model_type: str,
        learning_rate: float,
        generator_loss: float,
        discriminator_loss: float,
        real_images: torch.Tensor | None = None,
        fake_images: torch.Tensor | None = None,
        compute_metrics_every: int = 100,
        metric_names: list[str] | None = None,
    ) -> None:
        """Log training metrics including losses and optional image quality metrics.

        Args:
            epoch: Current epoch
            step: Current training step
            model_type: Type of model being trained
            learning_rate: Current learning rate
            generator_loss: Generator loss value
            discriminator_loss: Discriminator loss value
            real_images: Optional batch of real images for quality metrics
            fake_images: Optional batch of generated images for quality metrics
            compute_metrics_every: Compute quality metrics every N steps
            metric_names: List of metrics to compute (uses defaults if None)
        """
        timestamp = datetime.datetime.now().isoformat()
        total_loss = generator_loss + discriminator_loss

        # Save images if provided
        real_paths, fake_paths = [], []
        if real_images is not None and fake_images is not None:
            real_paths, fake_paths = self.save_images_batch(
                real_images,
                fake_images,
                "training",
                epoch,
                step,
            )

        # Prepare base metrics
        metrics_data = {
            "timestamp": timestamp,
            "epoch": epoch,
            "step": step,
            "model_type": model_type,
            "learning_rate": learning_rate,
            "generator_loss": generator_loss,
            "discriminator_loss": discriminator_loss,
            "total_loss": total_loss,
        }

        # Optionally compute image quality metrics
        if (
            step % compute_metrics_every == 0
            and real_images is not None
            and fake_images is not None
        ):
            quality_metrics = self.compute_image_metrics(fake_images, real_images, metric_names)
            metrics_data.update(quality_metrics)

        # Save to CSV
        self._append_to_csv(self.training_csv, metrics_data)

        # Add to (bounded) history and fold into the incremental running-best so
        # a best value survives even after its record is evicted from history.
        self.training_history.append(metrics_data)
        self._update_running_best(metrics_data)

        logger.info(f"📊 Training metrics - Epoch {epoch}, Step {step}")
        logger.info(f"   Losses - G: {generator_loss:.4f}, D: {discriminator_loss:.4f}")

    def log_cascading_validation(
        self,
        rows: list[dict[str, Any]],
        *,
        iteration: int,
        epoch: int,
    ) -> int:
        """Persist one tall cascading-validation sweep (issue #697).

        Each row is a severity point -- ``acceleration_level``, ``heldout`` and
        ``timestep`` are columns, not parts of a column NAME. The metric columns
        are whatever the strategy actually computed, so this cannot drift from
        an arm's declared metric set the way the retired hardcoded 15-name list
        did (that list never made it into any header, so all 45 values were
        discarded by the row writer's ``extrasaction="ignore"``).

        Routed through ``_append_to_csv`` deliberately: it owns header caching,
        schema evolution and the backup-on-rewrite, and duplicating any of that
        here would make a second CSV writer to keep in sync.

        Args:
            rows: severity-point records, one per point, already collapsed
                across validation batches by
                ``core.cascading_validation.aggregate_cascade_rows`` (which also
                stamps ``n_batches``, so a row states how many batches it means
                over). Passing raw per-batch rows here would write one line per
                (batch, level) under a single iteration.
            iteration: global step the sweep was measured at.
            epoch: epoch the sweep was measured at.

        Returns:
            Number of rows written -- so a caller can log it, and so a test can
            tell "wrote nothing" from "was never called".
        """
        if not rows:
            return 0

        # Stamped here, not in `build_cascade_row`: the strategy that measures
        # a sweep does not know which global step it is on -- the pipeline does.
        for row in rows:
            self._append_to_csv(
                self.cascading_csv,
                {"iteration": iteration, "epoch": epoch, **row},
            )

        logger.debug(
            "📊 Cascading validation: %d row(s) at iteration %d -> %s",
            len(rows),
            iteration,
            self.cascading_csv.name,
        )
        return len(rows)

    def log_inference_metrics(
        self,
        model_checkpoint: str,
        model_type: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inference_time: float,
        num_samples: int,
        metric_names: list[str] | None = None,
    ) -> None:
        """Log inference metrics including generation quality and timing.

        Args:
            model_checkpoint: Path to model checkpoint used
            model_type: Type of model
            predictions: Generated/reconstructed images
            targets: Ground truth/reference images
            inference_time: Total inference time
            num_samples: Number of samples generated
            metric_names: List of metrics to compute (uses defaults if None)
        """
        timestamp = datetime.datetime.now().isoformat()
        avg_generation_time = inference_time / num_samples if num_samples > 0 else 0

        # Compute quality metrics using SSOT
        quality_metrics = self.compute_image_metrics(predictions, targets, metric_names)

        # Prepare metrics data
        metrics_data = {
            "timestamp": timestamp,
            "model_checkpoint": model_checkpoint,
            "model_type": model_type,
            "num_samples": num_samples,
            "inference_time": inference_time,
            "avg_generation_time": avg_generation_time,
        }
        metrics_data.update(quality_metrics)

        # Save to CSV
        self._append_to_csv(self.inference_csv, metrics_data)

        # Add to history
        self.inference_history.append(metrics_data)

        logger.info("📊 Inference metrics logged")
        logger.info(f"   Samples: {num_samples}, Time: {inference_time:.2f}s")
        for metric_name, metric_value in quality_metrics.items():
            logger.info(f"   {metric_name}: {metric_value:.4f}")

    def log_epoch_summary(
        self,
        epoch: int,
        model_type: str,
        total_steps: int,
        epoch_time: float,
        avg_generator_loss: float,
        avg_discriminator_loss: float,
    ) -> None:
        """Log epoch summary.

        Args:
            epoch: Completed epoch number
            model_type: Type of model
            total_steps: Total steps in epoch
            epoch_time: Total epoch time
            avg_generator_loss: Average generator loss
            avg_discriminator_loss: Average discriminator loss
        """
        timestamp = datetime.datetime.now().isoformat()

        metrics_data = {
            "timestamp": timestamp,
            "epoch": epoch,
            "model_type": model_type,
            "total_steps": total_steps,
            "epoch_time": epoch_time,
            "avg_generator_loss": avg_generator_loss,
            "avg_discriminator_loss": avg_discriminator_loss,
        }

        # Save to CSV
        self._append_to_csv(self.validation_csv, metrics_data)

        # Add to history
        self.validation_history.append(metrics_data)

        logger.info(f"📈 Epoch {epoch} summary logged")
        logger.info(f"   Steps: {total_steps}, Time: {epoch_time:.1f}s")
        logger.info(f"   Avg Losses - G: {avg_generator_loss:.4f}, D: {avg_discriminator_loss:.4f}")

    def _append_to_csv(self, csv_path: Path, data: dict[str, Any]) -> None:
        """Append data row to CSV file with schema evolution support.

        Handles new keys appearing in data by updating headers.

        Args:
            csv_path: Path to CSV file
            data: Dictionary of data to append
        """
        # Get cached headers
        headers = self._csv_headers_cache.get(csv_path, [])
        if not headers:
            # Cache miss: read headers ONCE
            if csv_path.exists():
                with open(csv_path, newline="") as f:
                    reader = csv.reader(f)
                    headers = next(reader, [])
            else:
                headers = []
            self._csv_headers_cache[csv_path] = headers

        # Check for new keys
        data_keys = set(data.keys())
        header_keys = set(headers)
        new_keys = data_keys - header_keys

        if new_keys:
            logger.debug(f"Schema update: Adding keys {new_keys} to {csv_path.name}")
            # Extend headers with new keys (sorted for stability)
            headers.extend(sorted(new_keys))
            self._csv_headers_cache[csv_path] = headers

            # Rewrite CSV with new headers
            self._write_csv_header(csv_path, headers)

        # Append data row
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            row = [data.get(header, "") for header in headers]
            writer.writerow(row)

    def _write_csv_header(self, csv_path: Path, headers: list[str]) -> None:
        """Write CSV header, backing up old file if it exists.

        Args:
            csv_path: Path to CSV file
            headers: List of header names
        """
        if csv_path.exists():
            # Backup old file
            backup_path = csv_path.with_suffix(".bak")
            csv_path.rename(backup_path)

            # Read data from backup
            data_rows = []
            try:
                with open(backup_path, newline="") as f:
                    reader = csv.DictReader(f)
                    data_rows = [row for row in reader]
            except Exception as e:
                logger.warning(f"Failed to read backup file: {e}")

            # Rewrite with new headers
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                for row in data_rows:
                    writer.writerow(row)

            logger.debug(f"CSV schema updated at {csv_path}")
        else:
            # New file: write header
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    #: Degeneracy threshold for the percentile window, **relative** to the
    #: sample's own magnitude. The predecessor compared ``rng`` against an
    #: absolute ``1e-8``, which is only meaningful if every arm happens to
    #: render data of order 1. It does not: this repo's k-space magnitudes
    #: run to ~2e3, and the identical reconstruction expressed in smaller
    #: units (a healthy image scaled by 1e-12) tripped the guard and was
    #: written out as a solid-black PNG. Normalisation must be scale
    #: invariant — ``healthy`` and ``healthy * 1e-12`` have to render the same.
    _DEGENERATE_RANGE_RTOL = 1e-6

    def _normalize_images(self, images: torch.Tensor, context: str = "") -> torch.Tensor:
        """Normalize images to [0, 1] range for saving.

        Plain-tensor entry point over :meth:`_normalize_images_windowed`, for
        callers rendering a tensor in isolation with no counterpart to share a
        window with. Every window decision and every diagnostic lives in that
        method; this one exists so call sites that only want pixels are not made
        to unpack a window they would discard.
        """
        normalized, _ = self._normalize_images_windowed(images, context=context)
        return normalized

    def _normalize_images_windowed(
        self,
        images: torch.Tensor,
        context: str = "",
        *,
        reference_windows: list[_RenderWindow | None] | None = None,
    ) -> tuple[torch.Tensor, list[_RenderWindow | None]]:
        """Normalize images to [0, 1], returning the window each sample used.

        Foreground-aware per-sample percentile windowing. The earlier
        whole-image [0.5%, 99.5%] approach was correct for normal
        targets but at training-step 0 a kspace cold-diffusion model
        often produces an output whose IFFT magnitude is dominated by a
        small DC-peak blob plus a vast noisy background. The 99.5%
        percentile then sits in the *background-noise* tail, the DC
        blob saturates white, and the rest of the brain is squeezed
        into a narrow band — the May 2026 "bright spot + halo" symptom.

        Fix: compute the upper percentile over **foreground pixels
        only** (above the median + a small slack). When >5% of the
        image is foreground we use the foreground percentile; otherwise
        we fall back to the whole-image percentile so cases where the
        model genuinely outputs near-empty tensors don't divide by
        zero.

        **Degenerate samples are reported, never silently blacked out.**
        This routine used to collapse four *distinct* upstream failures onto
        one byte-identical solid-black PNG, which is a facade (pitfall #16):
        the artifact looks like a render, so nothing downstream registers a
        failure, and the diagnosis is destroyed at exactly the point it was
        available. The four were

        * a single non-finite pixel — ``torch.quantile`` propagates NaN, so
          ``vmin``/``vmax`` went NaN, every pixel normalised to NaN, and
          ``(NaN * 255).astype(np.uint8)`` is undefined behaviour that lands
          on 0. One bad pixel blacked out an otherwise perfect image;
        * an all-non-finite sample (a diverged sampler);
        * a genuinely constant sample (a collapsed model);
        * a *healthy* sample whose dynamic range fell below the absolute
          ``1e-8`` floor — see :attr:`_DEGENERATE_RANGE_RTOL`.

        Two of those are now rendered correctly rather than reported: the
        window is computed over **finite pixels only**, so a sample with a
        few NaNs renders its real content (the non-finite pixels alone go to
        0), and the degeneracy test is relative, so small-magnitude data
        renders identically to the same data scaled up. The remaining two
        are true degeneracies; they still render black, but emit one
        aggregated ``WARNING`` naming the affected sample indices and their
        raw statistics.

        WARNING level is deliberate and load-bearing: ``LoggingService.setup``
        clamps every logger and handler to ``logging.sinks.level``, and the
        arms that hit this path run at ``warning``. An INFO diagnostic here
        would be discarded before it reached the log.

        **A shared window is what makes a pair comparable.** Windowed by its own
        percentiles, every render is a flattering render: the affine map is
        re-fitted to whatever the tensor happens to contain, so a prediction 4x
        too bright, or too dim, or with the wrong contrast, comes out looking
        correct. The transfer function absorbs precisely the error the figure
        exists to display -- pitfall #16, an artifact that looks like evidence.
        Passing ``reference_windows`` renders this tensor under the counterpart's
        window instead, so intensity error survives into the PNG.

        **Sharing opens one new failure mode, and it is reported.** A healthy,
        non-constant sample lying entirely below the borrowed window clamps to
        bit-exact black; entirely above, to solid white. Neither is constant nor
        non-finite, so the degeneracy tests above cannot see it -- exactly the
        silent black-PNG facade that finite-only windowing closed. Full clamping
        against a borrowed window is therefore detected and folded into the same
        aggregated warning.

        The sample's own window is still computed even when a reference is in
        play, because the own window is the *diagnosis*: a collapsed producer
        borrowed into a healthy window renders as a flat tone -- a picture, with
        nothing visibly wrong -- and dropping that test to save three reductions
        would delete the finding.

        Args:
            images: Tensor of images [B, C, H, W]
            context: Optional caller tag (e.g. ``"fake/validation_R32x"``)
                folded into the diagnostic so the warning names *which*
                tensor degenerated rather than just "a batch".
            reference_windows: Per-sample windows from the counterpart render
                (index-aligned), or ``None`` to window each sample by its own
                percentiles. An entry that cannot be lent -- the counterpart was
                blacked out -- falls back to this sample's own window and says
                so in the warning, because the resulting pair is *not* a
                comparison. Indices past the end of the list have no counterpart
                at all and render under their own window silently: the caller's
                ``zip(..., strict=False)`` never writes them.

        Returns:
            ``(normalized, windows)`` -- the image tensor on CPU in [0, 1], and
            one :class:`_RenderWindow` per sample recording the transfer
            function actually applied (``rendered=False`` where the sample was
            written as solid black). The windows are what the counterpart
            borrows and what the sidecar records.
        """
        images = images.detach().cpu().float()

        if images.numel() == 0:
            return torch.zeros_like(images, dtype=torch.float32), []

        B = images.shape[0]
        result = torch.empty_like(images)
        windows: list[_RenderWindow | None] = []
        nonfinite_reports: list[str] = []
        degenerate_reports: list[str] = []
        # Findings about the SHARED window, kept apart from ``degenerate_reports``
        # so the warning cannot claim a blackout that did not happen -- see
        # ``_warn_degenerate_render``.
        window_reports: list[str] = []

        for i in range(B):
            sample = images[i]
            flat = sample.reshape(-1)

            # Window over FINITE pixels only. A NaN anywhere in ``flat``
            # makes every ``torch.quantile``/``torch.median`` below return
            # NaN, which is what silently blacked out healthy renders.
            finite_mask = torch.isfinite(flat)
            n_nonfinite = int(flat.numel() - int(finite_mask.sum().item()))
            if n_nonfinite:
                nonfinite_reports.append(f"sample {i}: {n_nonfinite}/{flat.numel()} non-finite")
            if n_nonfinite == flat.numel():
                # Nothing to window against — a fully diverged sample.
                degenerate_reports.append(f"sample {i}: all pixels non-finite")
                result[i] = torch.zeros_like(sample)
                # No finite pixel means no window was computed at all: the
                # ``0.0`` bounds are placeholders and ``finite_min is None`` is
                # the tell. ``rendered=False`` keeps it from being lent.
                windows.append(
                    _RenderWindow(
                        vmin=0.0,
                        vmax=0.0,
                        source="none (all pixels non-finite)",
                        rendered=False,
                        finite_min=None,
                        finite_max=None,
                    )
                )
                continue
            finite = flat[finite_mask] if n_nonfinite else flat
            # The RAW extremes, recorded beside the percentile window rather
            # than instead of it: a sample clamped flat against a borrowed
            # window is only diagnosable if its own dynamic range sits next to
            # the window it was forced through.
            finite_min = float(finite.min().item())
            finite_max = float(finite.max().item())

            # Whole-image percentiles (the legacy floor).
            vmin = torch.quantile(finite, 0.005).item()
            vmax_whole = torch.quantile(finite, 0.995).item()

            # Foreground-aware ceiling. ``threshold`` separates
            # background noise from anatomy: median is robust to a few
            # bright pixels (e.g. the DC peak) and is below most brain
            # tissue. The foreground fraction guard prevents this path
            # from kicking in for nearly-empty outputs.
            median_val = torch.median(finite).item()
            slack = 0.05 * (vmax_whole - vmin) if vmax_whole > vmin else 0.0
            threshold = median_val + slack
            fg_mask = finite > threshold
            fg_fraction = float(fg_mask.float().mean().item())
            if fg_fraction > 0.05:
                fg_pixels = finite[fg_mask]
                # 95th percentile of FOREGROUND clamps the bright DC
                # spike without saturating; combined with the legacy
                # 0.5% lower percentile the brain fills the dynamic
                # range cleanly.
                vmax = torch.quantile(fg_pixels, 0.95).item()
                if vmax <= vmin:
                    vmax = vmax_whole
            else:
                vmax = vmax_whole

            rng = vmax - vmin
            # Scale-invariant degeneracy test — see _DEGENERATE_RANGE_RTOL.
            scale = max(abs(vmin), abs(vmax))
            own_degenerate = rng <= self._DEGENERATE_RANGE_RTOL * scale or rng <= 0.0

            # Resolve which window actually draws this sample. The own window
            # computed above stays the diagnosis either way.
            reference: _RenderWindow | None = None
            counterpart_unusable = False
            if reference_windows is not None and i < len(reference_windows):
                candidate = reference_windows[i]
                if candidate is not None and candidate.can_be_lent():
                    reference = candidate
                else:
                    counterpart_unusable = True

            source = "own"
            render_min, render_max = vmin, vmax
            if reference is not None:
                source = "shared"
                render_min, render_max = reference.vmin, reference.vmax
            elif counterpart_unusable:
                source = "own (counterpart degenerate)"
                window_reports.append(
                    f"sample {i}: counterpart was blacked out, so this side "
                    f"rendered under its OWN window [{vmin:.6g}, {vmax:.6g}]; "
                    "the pair is NOT comparable"
                )
            render_rng = render_max - render_min

            if own_degenerate and reference is None:
                # Genuinely constant sample: black is the honest render, but
                # it is reported rather than passed off as a picture.
                degenerate_reports.append(
                    f"sample {i}: constant at {vmin:.6g} "
                    f"(window [{vmin:.6g}, {vmax:.6g}], range {rng:.6g})"
                )
                result[i] = torch.zeros_like(sample)
                windows.append(
                    _RenderWindow(
                        vmin=vmin,
                        vmax=vmax,
                        source=source,
                        rendered=False,
                        finite_min=finite_min,
                        finite_max=finite_max,
                    )
                )
            else:
                if own_degenerate:
                    # Constant, but a lendable reference window renders it as a
                    # flat tone rather than black. Its own finding: the producer
                    # collapsed (diagnosis) and the sample did NOT black out
                    # (consequence). Folding it into ``degenerate_reports``
                    # would make the warning assert a blackout that never
                    # happened, which is the same facade in miniature.
                    window_reports.append(
                        f"sample {i}: constant at {vmin:.6g}, rendered as a flat "
                        f"tone under the shared window "
                        f"[{render_min:.6g}, {render_max:.6g}]"
                    )
                normalized = ((sample - render_min) / render_rng).clamp(0, 1)

                if reference is not None:
                    # Checked on ``normalized`` BEFORE the non-finite mask below
                    # writes zeros, and over finite pixels only -- masked zeros
                    # would read as "fell below the window".
                    norm_flat = normalized.reshape(-1)
                    norm_finite = norm_flat[finite_mask] if n_nonfinite else norm_flat
                    if float(norm_finite.max().item()) <= 0.0:
                        window_reports.append(
                            f"sample {i}: every finite pixel fell BELOW the "
                            f"shared window [{render_min:.6g}, {render_max:.6g}] "
                            f"and rendered black; its own range is "
                            f"[{finite_min:.6g}, {finite_max:.6g}]"
                        )
                    elif float(norm_finite.min().item()) >= 1.0:
                        window_reports.append(
                            f"sample {i}: every finite pixel fell ABOVE the "
                            f"shared window [{render_min:.6g}, {render_max:.6g}] "
                            f"and rendered white; its own range is "
                            f"[{finite_min:.6g}, {finite_max:.6g}]"
                        )

                # Send non-finite pixels to black so only THEY are lost, not
                # the whole sample -- masked, NOT via ``nan_to_num``.
                #
                # ``clamp`` runs first and maps ``+inf`` to 1.0 and ``-inf`` to
                # 0.0, so a ``nan_to_num(posinf=..., neginf=...)`` after it can
                # never fire: those arguments were dead, and ``+inf`` rendered
                # WHITE while this comment claimed black. White is the worst
                # available answer -- it reads as maximum signal, so a diverged
                # pixel looks like the brightest real anatomy in the image,
                # which is the exact misreading this render path exists to
                # stop. Only NaN was actually being caught.
                if n_nonfinite:
                    result[i] = torch.where(
                        finite_mask.reshape(sample.shape),
                        normalized,
                        torch.zeros_like(normalized),
                    )
                else:
                    result[i] = normalized
                windows.append(
                    _RenderWindow(
                        vmin=render_min,
                        vmax=render_max,
                        source=source,
                        rendered=True,
                        finite_min=finite_min,
                        finite_max=finite_max,
                    )
                )

        if nonfinite_reports or degenerate_reports or window_reports:
            self._warn_degenerate_render(
                context, nonfinite_reports, degenerate_reports, images, window_reports
            )

        return result, windows

    def _warn_degenerate_render(
        self,
        context: str,
        nonfinite_reports: list[str],
        degenerate_reports: list[str],
        images: torch.Tensor,
        window_reports: list[str] | None = None,
    ) -> None:
        """Emit one aggregated WARNING for a batch that rendered degenerately.

        Aggregated per call rather than per sample: cascading validation saves
        every rung at every validation interval, so a per-sample warning would
        emit B x rungs lines per interval and get tuned out.

        ``degenerate_reports`` means exactly "written as solid black" -- that is
        what the consequence clause claims, so a finding that did *not* black a
        sample out belongs in ``window_reports`` instead. The two are kept apart
        for one reason: a sample that is constant but rendered under a borrowed
        window is a real finding whose consequence is the opposite of a
        blackout, and a warning that overstates its own finding is the same
        facade in miniature. None of the ``window_reports`` cases fires on a
        healthy comparable pair, so this stays a diagnostic rather than spam.
        """
        window_reports = window_reports or []
        finite = images[torch.isfinite(images)]
        if finite.numel():
            stats = f"finite range [{finite.min().item():.6g}, {finite.max().item():.6g}]"
        else:
            stats = "no finite values in the batch"
        tag = f" [{context}]" if context else ""
        details = "; ".join(nonfinite_reports + degenerate_reports + window_reports)
        # Only claim a blackout when one actually happened — a warning that
        # overstates its own finding is the same facade in miniature.
        if degenerate_reports:
            consequence = (
                f"{len(degenerate_reports)} sample(s) were written as solid "
                "black; a solid-black PNG is a RENDER FAILURE, not a picture "
                "of the model output"
            )
        elif window_reports:
            consequence = (
                "the sample(s) rendered, but the shared-window findings above "
                "change how the picture must be read"
            )
        else:
            consequence = (
                "the finite content still rendered (non-finite pixels alone were written as black)"
            )
        logger.warning(
            "[MetricsTracker]%s Degenerate image render — %d sample(s) carry "
            "non-finite pixels, %d sample(s) were written as black, %d "
            "shared-window finding(s). %s. %s. %s: check the producing path "
            "(sampler divergence, a collapsed model, or an unscaled tensor) "
            "rather than the saver.",
            tag,
            len(nonfinite_reports),
            len(degenerate_reports),
            len(window_reports),
            stats,
            details,
            consequence,
        )

    def _submit_image_writes(self, jobs: list[tuple[np.ndarray, str, Path]]) -> None:
        """Queue one batch of writes, then re-raise the previous batch's error (if any).

        The current batch is submitted first so a failed predecessor costs only its own pictures.
        """
        if not jobs:
            return
        with self._image_lock:
            previous, self._pending_images = self._pending_images, None
            if self._image_executor is None:
                self._image_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="metrics-png"
                )
            self._pending_images = self._image_executor.submit(self._write_images, jobs)
            if previous is not None:
                previous.result()

    def _wait_pending_locked(self) -> None:
        pending, self._pending_images = self._pending_images, None
        if pending is not None:
            pending.result()

    def _write_images(self, jobs: list[tuple[np.ndarray, str, Path]]) -> None:
        for array, mode, path in jobs:
            self._write_image(array, mode, path)
            logger.debug(f"[MetricsTracker] Saved image: {path}")

    @staticmethod
    def _write_image(array: np.ndarray, mode: str, path: Path) -> None:
        Image.fromarray(array, mode=mode).save(path)

    def drain(self) -> None:
        """Block until queued PNG writes finish; re-raise the worker's error, if any."""
        with self._image_lock:
            self._wait_pending_locked()

    def shutdown_image_writer(self) -> None:
        """Drain, then stop the worker thread. Call once after the last validation."""
        try:
            self.drain()
        finally:
            with self._image_lock:
                executor, self._image_executor = self._image_executor, None
            if executor is not None:
                executor.shutdown(wait=True)

    def _rasterize_tensor(self, tensor: torch.Tensor, path: Path) -> tuple[np.ndarray, str]:
        """Convert a tensor to a ``(uint8 array, PIL mode)`` pair on the calling thread.

        Handles 1-channel (grayscale), 2-channel (complex MRI → magnitude),
        and 3-channel (RGB) tensors.

        Args:
            tensor: Image tensor [C, H, W]
            path: Output file path
        """
        # Last line of defence before the uint8 cast. ``float -> uint8`` is
        # undefined for NaN/inf in NumPy; on x86 NaN lands on 0, so a
        # non-finite tensor that reached here would be written as a
        # plausible-looking solid-black PNG on this platform and as arbitrary
        # noise on another. ``_normalize_images`` already scrubs its output,
        # so a hit here means a caller bypassed it — say so instead of
        # emitting a silently-wrong image.
        tensor = tensor.detach().cpu()
        if not torch.isfinite(tensor).all():
            n_bad = int((~torch.isfinite(tensor)).sum().item())
            logger.warning(
                "[MetricsTracker] %s: %d/%d non-finite pixel(s) reached the "
                "uint8 cast (the cast is undefined on them). Rendering those "
                "pixels as black; the tensor was NOT normalised by "
                "_normalize_images.",
                path.name,
                n_bad,
                tensor.numel(),
            )
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)

        if tensor.dim() == 3:  # [C, H, W]
            if tensor.shape[0] == 1:  # Grayscale
                img_array = tensor.squeeze(0).numpy()
                return (img_array * 255).astype(np.uint8), "L"
            elif tensor.shape[0] == 3:  # RGB
                img_array = tensor.permute(1, 2, 0).numpy()
                return (img_array * 255).astype(np.uint8), "RGB"
            else:
                # Multi-channel (incl. C=2): RSS-combine all channels.
                # Previously a special C==2 branch treated the channels as
                # interleaved (real, imag) and computed sqrt(R^2 + I^2). That
                # is unsafe for paired-modality data (e.g. ULF/HF, T1/T2)
                # which appears here with shape [2, H, W] but is NOT complex
                # — the assumption produced the "doubled and odd" mixed
                # appearance the May 2026 ULF reports flagged. Upstream
                # callers MUST reduce to a single channel when modality
                # semantics matter; here we do RSS as a deterministic
                # last-resort that is mathematically equivalent to the old
                # behaviour for genuine real-stacked complex data.
                magnitude = torch.sqrt((tensor**2).sum(dim=0) + 1e-8)
                mag_max = magnitude.max()
                if mag_max > 1e-6:
                    magnitude = magnitude / mag_max
                img_array = magnitude.numpy()
                return (img_array * 255).astype(np.uint8), "L"
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    # Metadata columns that are not quality metrics and must not be "best"-tracked.
    _BEST_EXCLUDE_KEYS = frozenset(
        {
            "timestamp",
            "epoch",
            "step",
            "model_type",
            "learning_rate",
            "generator_loss",
            "discriminator_loss",
            "total_loss",
        }
    )

    def _update_running_best(self, record: dict[str, Any]) -> None:
        """Fold one training record into the incremental running-best (O(#keys))."""
        for key, value in record.items():
            if key in self._BEST_EXCLUDE_KEYS:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            higher = _metric_higher_is_better(key)
            if higher is None:
                # Undeclared direction (a non-metric log column such as `lr` or
                # `grad_norm`). Skip rather than invent one — a "best" under a
                # guessed direction is worse than no entry at all.
                continue
            cur = self._best_metrics.get(key)
            if cur is None or (higher and value > cur) or (not higher and value < cur):
                self._best_metrics[key] = float(value)

    def get_best_metrics(self) -> dict[str, float]:
        """Get best metrics achieved during training.

        Merges the incremental running-best (folded on every training log, so it
        survives history eviction) with a scan of whatever is currently in
        ``training_history`` — the latter keeps callers that set
        ``training_history`` directly (e.g. tests) working.

        Returns:
            Dictionary of ``best_<metric>`` -> best_value
        """
        # Start from the incremental tracker, then fold in the current history so
        # a directly-assigned history is still honored.
        best: dict[str, float] = dict(self._best_metrics)
        for record in self.training_history:
            for key, value in record.items():
                if key in self._BEST_EXCLUDE_KEYS:
                    continue
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    continue
                if value == "":  # defensive: stringified empties
                    continue
                higher = _metric_higher_is_better(key)
                if higher is None:
                    continue  # undeclared direction — skip, never guess (see above)
                cur = best.get(key)
                if cur is None or (higher and value > cur) or (not higher and value < cur):
                    best[key] = float(value)

        return {f"best_{key}": val for key, val in best.items()}

    def save_metrics_csv(
        self,
        filename: str | None = None,
        iteration: int | None = None,
    ) -> None:
        """Save all accumulated metrics (compatibility method).

        Args:
            filename: Optional custom filename (ignored)
            iteration: Optional iteration number (ignored)
        """
        logger.debug("Metrics already logged to CSV files")

    def export_summary_report(self, output_path: str | None = None) -> str:
        """Export a summary report of all metrics.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to generated report
        """
        if output_path is None:
            output_path = self.output_dir / "metrics_summary_report.txt"

        # Aggregate statistics
        summary_lines = [
            "=" * 60,
            "METRICS SUMMARY REPORT",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"Training Records: {len(self.training_history)}",
            f"Inference Records: {len(self.inference_history)}",
            f"Epoch Records: {len(self.validation_history)}",
            "",
            "Best Metrics Achieved:",
            "-" * 40,
        ]

        for metric_name, metric_value in self.get_best_metrics().items():
            summary_lines.append(f"  {metric_name}: {metric_value:.6f}")

        summary_lines.extend(["", "=" * 60])

        # Write report
        with open(output_path, "w") as f:
            f.write("\n".join(summary_lines))

        logger.info(f"📋 Summary report saved to: {output_path}")
        return str(output_path)
