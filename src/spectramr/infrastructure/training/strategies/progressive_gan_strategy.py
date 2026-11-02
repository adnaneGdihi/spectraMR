"""Progressive-growing GAN training strategy.

Subclasses :class:`~spectramr.infrastructure.training.strategies.gan.GANTrainingStrategy`
and adds the phase scheduler of Karras *et al.* [Karras et al., ICLR 2018]:
training advances through resolution phases, and within each phase the
newly-added layers are faded in via a coefficient ``alpha`` annealed from
0 to 1. The schedule is driven by a per-phase step budget so the
generator and discriminator grow together.

The strategy reuses the parent's full GAN optimisation loop (D/G
alternation, WGAN-GP via the framework's ``gan_wgan`` loss); it only
mutates ``current_phase`` / ``alpha`` on the generator and discriminator
before each step. The model still trains correctly under the plain
``gan`` mode without this strategy — the schedule is the only difference.

Config: reads ``config.training.gan.phase_schedule`` (a list of per-phase
step counts). When absent, a uniform schedule is derived from the model's
``num_phases`` and ``training.max_iterations``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.infrastructure.training.loop_state import resolve_loop_iteration
from spectramr.infrastructure.training.step_io import accepts_step_io
from spectramr.infrastructure.training.strategies.gan import GANTrainingStrategy
from spectramr.infrastructure.training.utils.metric_views import MetricViews

logger = logging.getLogger(__name__)


class ProgressiveGANStrategy(GANTrainingStrategy):
    """GAN strategy with progressive resolution growth and fade-in annealing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Fail fast on a non-growable model: the progressive_gan paradigm is a
        # silent no-op (the schedule is dropped) unless the generator — and the
        # discriminator, when present — expose ``set_phase``. Raise here rather
        # than letting ``_apply_phase`` skip the growth step unnoticed.
        self._require_phase_capability()
        self._phase_schedule = self._resolve_phase_schedule()
        # Cumulative step boundaries: phase s spans [boundaries[s], boundaries[s+1]).
        self._phase_boundaries = self._cumulative(self._phase_schedule)

    def _require_phase_capability(self) -> None:
        """Validate once that the model(s) can grow, raising otherwise."""
        gen = self.env.generator if self.env else None
        if not callable(getattr(gen, "set_phase", None)):
            raise TypeError(
                "ProgressiveGANStrategy requires a generator exposing a callable "
                f"'set_phase(phase, alpha)', got {type(gen).__name__}. Use a "
                "progressive_gan model_type (gans/progressive_gan.py); without "
                "'set_phase' the growth schedule would be silently dropped."
            )
        disc = self.env.discriminator if self.env else None
        if disc is not None and not callable(getattr(disc, "set_phase", None)):
            raise TypeError(
                "ProgressiveGANStrategy requires the discriminator to expose a "
                f"callable 'set_phase(phase, alpha)', got {type(disc).__name__}. "
                "Use a progressive_gan discriminator (gans/progressive_discriminator.py); "
                "without 'set_phase' the growth schedule would be silently dropped."
            )

    # ------------------------------------------------------------------ #
    # Schedule construction
    # ------------------------------------------------------------------ #
    def _num_phases(self) -> int:
        gen = self.env.generator if self.env else None
        n = getattr(gen, "num_phases", None)
        if isinstance(n, int) and n >= 1:
            return n
        raise TypeError(
            "ProgressiveGANStrategy requires a phase-growable generator exposing an "
            f"integer 'num_phases' >= 1, got {n!r} on {type(gen).__name__}. "
            "Use a progressive_gan model_type (gans/progressive_gan.py) under "
            "progressive_gan training mode instead of degrading to a single-phase no-op."
        )

    def _resolve_phase_schedule(self) -> list[int]:
        """Return the per-phase step budget.

        Prefers ``config.training.gan.phase_schedule``; otherwise spreads
        ``training.max_iterations`` uniformly across the model phases.
        """
        num_phases = self._num_phases()
        schedule: list[int] | None = None
        try:
            gan_cfg = getattr(self.config.training, "gan", None)
            schedule = getattr(gan_cfg, "phase_schedule", None) if gan_cfg else None
        except AttributeError:
            schedule = None

        if schedule:
            schedule = [int(s) for s in schedule]
            if any(s <= 0 for s in schedule):
                raise ValueError(f"phase_schedule entries must be positive, got {schedule}")
            if len(schedule) != num_phases:
                raise ValueError(
                    "config.training.gan.phase_schedule length "
                    f"({len(schedule)}) must equal the model's num_phases "
                    f"({num_phases}); a mismatch silently truncates the schedule "
                    "or raises IndexError mid-training."
                )
            return schedule

        max_iters = int(getattr(self.config.training, "max_iterations", 0) or 0)
        if max_iters <= 0:
            # No budget known: keep the model at its final phase from the start.
            return [1] * num_phases
        per_phase = max(max_iters // num_phases, 1)
        return [per_phase] * num_phases

    @staticmethod
    def _cumulative(schedule: list[int]) -> list[int]:
        boundaries = [0]
        for s in schedule:
            boundaries.append(boundaries[-1] + s)
        return boundaries

    # ------------------------------------------------------------------ #
    # Phase / alpha computation
    # ------------------------------------------------------------------ #
    def _phase_and_alpha(self, global_step: int) -> tuple[int, float]:
        """Compute (phase, alpha) for a given global training step.

        ``alpha`` linearly anneals 0 -> 1 over the first half of each phase's
        budget and is held at 1 for the remainder; phase 0 has no fade-in.
        """
        num_phases = self._num_phases()
        boundaries = self._phase_boundaries
        total = boundaries[-1]
        if total <= 0:
            return num_phases - 1, 1.0
        step = min(max(global_step, 0), total - 1)

        phase = num_phases - 1
        for s in range(num_phases):
            if boundaries[s] <= step < boundaries[s + 1]:
                phase = s
                break

        if phase == 0:
            return 0, 1.0

        phase_len = boundaries[phase + 1] - boundaries[phase]
        offset = step - boundaries[phase]
        fade_len = max(phase_len // 2, 1)
        alpha = min(offset / fade_len, 1.0)
        return phase, float(alpha)

    def _apply_phase(self, global_step: int) -> tuple[int, float]:
        # Capability is validated at construction (``_require_phase_capability``),
        # so ``set_phase`` is guaranteed present here — call it unconditionally
        # rather than silently skipping growth when the attribute is absent.
        phase, alpha = self._phase_and_alpha(global_step)
        gen = self.env.generator if self.env else None
        disc = self.env.discriminator if self.env else None
        if gen is not None:
            gen.set_phase(phase, alpha)
        if disc is not None:
            disc.set_phase(phase, alpha)
        return phase, alpha

    # ------------------------------------------------------------------ #
    # Phase-resolution targets
    # ------------------------------------------------------------------ #
    def _to_phase_resolution(self, target: torch.Tensor) -> torch.Tensor:
        """Box-downsample ``target`` to the generator's active output resolution.

        At phase ``s`` the generator emits ``4 * 2**s`` pixels, so the real batch
        the critic scores and the target of the reconstruction losses are reduced
        to that size (Karras et al.), with the same average pooling the
        discriminator's fade-in path uses (#2385).
        """
        res = self.env.generator.current_resolution()
        if tuple(target.shape[-2:]) == (res, res):
            return target
        if target.shape[-2] % res or target.shape[-1] % res:
            raise ValueError(
                f"target spatial size {tuple(target.shape[-2:])} is not a multiple of the "
                f"active ProgressiveGAN resolution {res}; set the patch size to "
                "model_kwargs.max_resolution."
            )
        return F.adaptive_avg_pool2d(target, res)

    # ------------------------------------------------------------------ #
    # Step override
    # ------------------------------------------------------------------ #
    @accepts_step_io
    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        iteration: int = 0,
        batch_data: Any = None,
        scaler: Any = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Advance the growth phase, resize the target to it, then run the GAN step."""
        phase, alpha = self._apply_phase(resolve_loop_iteration(self))
        if input_batch is None or target_batch is None:
            input_batch, target_batch = self._unpack_batch(batch)
        target_batch = self._to_phase_resolution(self._to_device(target_batch))
        steps = super().train_step(
            batch,
            epoch,
            input_batch=input_batch,
            target_batch=target_batch,
            iteration=iteration,
            batch_data=batch_data,
            scaler=scaler,
            **kwargs,
        )
        # The base step resets ``_last_step_metrics``, so the phase is recorded after it.
        self._last_step_metrics["progan_phase"] = float(phase)
        self._last_step_metrics["progan_alpha"] = float(alpha)
        return steps

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Run the critic at the active phase; score the image metrics at full resolution.

        The critic and the visual capture see the phase-resolution pair, while the
        metrics compare the prediction upsampled to the target size with the full
        target, so ``val_psnr`` stays comparable across phases and with
        full-resolution baselines. ``progan_phase`` is stamped into the result.
        """
        full = target_batch
        if torch.is_complex(full):
            full = torch.cat([full.real, full.imag], dim=1)
        self._val_full_target = full
        try:
            results = super().validation_step(
                input_batch, self._to_phase_resolution(target_batch), **kwargs
            )
        finally:
            self._val_full_target = None
        results["progan_phase"] = float(self.env.generator.current_phase)
        return results

    def _metric_transform_views(
        self, pred: torch.Tensor, target: torch.Tensor, metrics_config: Any, **kwargs: Any
    ) -> MetricViews:
        full = getattr(self, "_val_full_target", None)
        if full is not None:
            target = full.to(pred.device)
            if pred.shape[-2:] != target.shape[-2:]:
                pred = F.interpolate(
                    pred, size=target.shape[-2:], mode="bilinear", align_corners=False
                )
        return super()._metric_transform_views(pred, target, metrics_config, **kwargs)

    def _log_validation_images_to_tensorboard(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        inputs: torch.Tensor,
        metrics: dict,
    ) -> None:
        # The pipeline's visual capture pairs a fresh forward with the full-size target;
        # handing it this pass's phase-resolution pair keeps the two shapes equal.
        self._last_visual_pred = predictions.detach()
        self._last_visual_target = targets.detach()
        super()._log_validation_images_to_tensorboard(predictions, targets, inputs, metrics)
