"""Train a vendored baseline through its AUTHORS' forward process, not this repo's.

Every other strategy here owns a degradation schedule and an objective. This one owns
neither: it hands the clean target to the adapter and the adapter runs the upstream
diffusion object -- CDiffMR's ``GaussianDiffusion``, FDB's ``DiffusionBridge``, Shen's
``KspaceDiffusion`` -- which draws its own timestep, applies its own degradation and
returns its own loss.

That is the whole difference between a reproduction and an architecture comparison, and
it is the difference #2080 records. Running these arms on ``GraphColdDiffusionStrategy``
gave three papers one Cartesian mask ladder and one loss; the three published methods
degrade in three different ways, and only their own code knows how.

The dispatch is polymorphic. There is no branch on baseline name anywhere in this file:
:meth:`BaselineAdapter.training_loss` is abstract, and each adapter answers for itself
(non-negotiable 6).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy

logger = logging.getLogger(__name__)


class UpstreamProcessStrategy(BaseTrainingStrategy):
    """Delegate the whole training step to the model's upstream diffusion object.

    The model must be a :class:`~spectramr.models.baselines._base.BaselineAdapter`;
    anything else is refused at setup rather than at the first backward pass.
    """

    #: Every loss term is computed inside the upstream object, so no declarative loss
    #: list reaches the objective. Declaring this is what stops the loss-ownership gate
    #: from reporting an arm's ``image_losses`` as UNVERIFIED (#1918) -- the honest
    #: answer here is "the strategy computes it", not "nothing checked".
    inline_losses: ClassVar[frozenset[str]] = frozenset({"l1", "l2"})
    folds_image_losses: ClassVar[bool] = False

    def setup(self, *args: Any, **kwargs: Any) -> None:
        """Bind the environment, then check the arm agrees with the upstream method."""
        super().setup(*args, **kwargs)
        adapter = self._adapter()
        self._reject_objective_mismatch(adapter)
        self._point_calibration_at_the_run(adapter)

    def _adapter(self) -> Any:
        """The generator, refused unless it can run an upstream process."""
        model = getattr(self.env, "generator", None)
        unwrapped = getattr(model, "module", model)
        if not hasattr(unwrapped, "training_loss"):
            raise TypeError(
                f"{type(self).__name__} trains a model through its authors' own "
                f"forward process, so the model must be a BaselineAdapter that "
                f"implements `training_loss`. Got {type(unwrapped).__name__}. Use a "
                "strategy that owns a degradation schedule instead."
            )
        return unwrapped

    def _reject_objective_mismatch(self, adapter: Any) -> None:
        """An arm may not advertise an objective the upstream will not compute.

        The arms keep a declarative ``losses:`` block -- ``LossBuilder.validate``
        refuses an empty stack at the director's step 3/6 -- but nothing in it reaches
        the objective here. Left unchecked that is an unread knob every arm believes it
        set (non-negotiable 8). So the declaration is read and must AGREE: FDB minimises
        MSE on x_0, CDiffMR and Shen minimise L1.
        """
        declared = self._declared_loss_names()
        if not declared:
            return
        family = str(adapter.UPSTREAM_LOSS_FAMILY)
        if declared != {family}:
            raise ValueError(
                f"{type(adapter).__name__} minimises {family!r} inside its upstream "
                f"training step, but this arm declares {sorted(declared)}. The "
                "declaration does not reach the objective on this strategy, so a "
                f"disagreement is silent. Declare exactly [{family}] at weight 1.0, or "
                "train this arm on a strategy that owns its loss."
            )

    def _declared_loss_names(self) -> set[str]:
        """Enabled entries across the domain loss lists, by canonical name."""
        losses = getattr(self.config, "losses", None)
        if losses is None:
            return set()
        names: set[str] = set()
        for list_name in ("image_losses", "kspace_losses", "complex_losses", "latent_losses"):
            for entry in getattr(losses, list_name, None) or []:
                if getattr(entry, "enabled", True) and getattr(entry, "name", None):
                    names.add(str(entry.name))
        return names

    def _point_calibration_at_the_run(self, adapter: Any) -> None:
        """Give an upstream that writes calibration to disk somewhere to write it.

        FDB's ``q_sample`` does ``np.save("w.npy")`` into the working directory on every
        call and reads it back at construction, so two runs started from the same shell
        would seed each other. Only adapters that declare the hook are touched.
        """
        if not hasattr(adapter, "set_calibration_dir"):
            return
        output_dir = getattr(getattr(self.config, "training", None), "output_dir", None)
        if not output_dir:
            return
        target = Path(output_dir) / "upstream_calibration"
        adapter.set_calibration_dir(target)
        logger.info("[upstream-process] %s calibration -> %s", type(adapter).__name__, target)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Run the authors' training step.

        ``input_batch`` is deliberately unused: all three upstreams degrade the CLEAN
        target themselves, and handing them an already-undersampled measurement would
        apply the degradation twice.

        Args:
            input_batch: The loader's model input. Unused here.
            target_batch: The clean target, ``[B, 2, H, W]`` real / imaginary.
            epoch: Unused -- the upstream draws its own timestep per step.
            **kwargs: Passed through as the batch for upstreams that need more.

        Returns:
            ``{"g_total_loss": <the upstream's own loss>}``.
        """
        del input_batch, epoch
        adapter = self._adapter()
        loss = adapter.training_loss(target_batch, kwargs.get("batch_data"))
        return {"g_total_loss": loss}

    def _run_reverse_process(
        self,
        adapter: Any,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        batch_data: Any = None,
    ) -> torch.Tensor:
        """Call the adapter's own reverse process; translate its refusal, never its silence.

        Left with no override, :class:`~spectramr.models.baselines._base.BaselineAdapter`'s
        ``validation_sample`` raises ``NotImplementedError``. Swallowing that and falling
        back to a bare ``forward()`` is exactly the shape this strategy exists to close
        (finding 21): reported here as a named ``RuntimeError`` instead, which ``train.py``'s
        "every validation batch raised" gate turns into a failed run rather than a green one
        reporting a number that grades nothing (non-negotiable 3).
        """
        try:
            output = adapter.validation_sample(input_batch, target_batch, batch=batch_data)
        except NotImplementedError as exc:
            raise RuntimeError(
                f"{type(adapter).__name__} has no reverse-sampling process this strategy "
                f"can drive during validation, so no honest reconstruction metric exists "
                f"for it: {exc}"
            ) from exc
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output

    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        *,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Validate through the adapter's OWN reverse process -- never a bare ``forward()``.

        Left undefined here, ``validation_step`` would resolve to
        ``MetricsMixin.validation_step`` -- one ``generator(input_batch)`` call with no
        timestep, which every adapter here defaults to t=0: the fully-sampled identity
        rung for CDiffMR/Shen, and outside FDB's trained timestep distribution entirely.
        This drives :meth:`_run_reverse_process` instead and scores its output exactly
        like the fallback would have, so the number changes meaning, not the metric
        plumbing.
        """
        del kwargs
        adapter = self._adapter()
        device = getattr(self, "device", target_batch.device)
        self.env.generator.eval()
        try:
            with torch.no_grad():
                input_batch = input_batch.to(device, non_blocking=True)
                target_batch = target_batch.to(device, non_blocking=True)
                output = self._run_reverse_process(adapter, input_batch, target_batch, batch_data)
                if output.device != target_batch.device:
                    output = output.to(target_batch.device)
                val_config = getattr(getattr(self, "config", None), "validation", None)
                output, target_batch = self._apply_metric_transforms(
                    output, target_batch, val_config
                )
                return self.validation_metrics_computer.compute(output, target_batch)
        finally:
            self.env.generator.train()


__all__ = ["UpstreamProcessStrategy"]
