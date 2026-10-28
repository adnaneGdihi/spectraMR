r"""Multi-axis classifier-free guidance (CFG) mixin.

Implements §8 of the multi-contrast brainstorm — generalises classifier-
free guidance [5] from a single binary condition to a *vector* of
condition axes :math:`\mathbf{c} = (c_{\text{contrast}}, c_{\text{thickness}},
c_{\text{pathology}}, \dots)`. Each axis is dropped independently during
training; at sampling time, the guidance is a weighted sum over per-axis
guidance directions:

.. math::

   \tilde{\boldsymbol\epsilon}
   \;=\;
   \boldsymbol\epsilon_\theta(\mathbf{x}, \emptyset)
   \;+\;
   \sum_i w_i \big[
     \boldsymbol\epsilon_\theta(\mathbf{x}, c_i)
     -\boldsymbol\epsilon_\theta(\mathbf{x}, \emptyset)
   \big].

The advantage over single-axis CFG is independent dial knobs:
:math:`w_{\text{contrast}}` controls how strictly the output adheres to
the requested contrast, :math:`w_{\text{pathology}}` controls how
strongly any pathology mask is honoured, and so on. This is critical
when conditions can pull in opposite directions (e.g. a lesion-mask
condition that wants strong local edits vs. a contrast condition that
wants smooth brain-tissue priors).

The mixin is *model-side* — no diffusion-strategy edits required.
Models inherit ``MultiAxisCFGMixin`` and implement
``_predict_noise(x, conditions)`` where ``conditions`` is a dict of
``{"axis_name": tensor_or_None}``. The mixin provides:

- ``forward_with_cfg_drop(x, conditions, drop_probs)`` — training-time
  helper that masks out a subset of condition axes per sample.
- ``cfg_sample_step(x, conditions, weights)`` — inference-time
  combination of per-axis guidance.

Reference
---------
[5] J. Ho, T. Salimans, "Classifier-free diffusion guidance," NeurIPS
    Workshop on Deep Generative Models and Downstream Applications,
    2021.

§8 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from spectramr.models.blocks.cfg_schedule import PerAxisSchedule

__all__ = ["MultiAxisCFGMixin"]


class MultiAxisCFGMixin(nn.Module):
    r"""Per-axis classifier-free guidance for any condition-aware model.

    Subclasses must implement::

        def _predict_noise(
            self,
            x: torch.Tensor,
            conditions: dict[str, torch.Tensor | None],
        ) -> torch.Tensor: ...

    where each entry of ``conditions`` is either a tensor (axis active)
    or ``None`` (axis dropped — the model should treat this as the
    unconditional path). Standard convention: a model checks
    ``cond is None`` and substitutes a learnt null embedding.

    Forward pass during training: call
    ``forward_with_cfg_drop(x, conditions, drop_probs)`` to mask out
    each axis with its independent dropout probability.
    """

    def _predict_noise(
        self,
        x: torch.Tensor,
        conditions: Mapping[str, torch.Tensor | None],
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _predict_noise(x, conditions)."
        )

    def _predict_noise_many(
        self,
        x: torch.Tensor,
        condition_sets: Sequence[Mapping[str, torch.Tensor | None]],
    ) -> list[torch.Tensor]:
        """One prediction per condition set. Override to batch them into one call."""
        return [self._predict_noise(x, conditions) for conditions in condition_sets]

    @staticmethod
    def _drop_axes(
        conditions: Mapping[str, torch.Tensor | None],
        drop_probs: Mapping[str, float],
        device: torch.device,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor | None]:
        """Return a new conditions dict with axes randomly nulled per sample."""
        out: dict[str, torch.Tensor | None] = {}
        for name, value in conditions.items():
            p = drop_probs.get(name, 0.0)
            if value is None or p <= 0.0:
                out[name] = value
                continue
            if not (0.0 < p <= 1.0):
                raise ValueError(f"drop probability for {name!r} must be in (0, 1]; got {p}.")
            mask = torch.rand(batch_size, generator=generator, device=device) < p
            if mask.any():
                # Broadcast mask along the trailing dims of the condition tensor
                # so we can write zeros into the dropped slots. Convention:
                # downstream model treats an all-zero condition row as "null".
                masked = value.clone()
                trailing = (slice(None),) + (None,) * (value.dim() - 1)
                masked = torch.where(
                    mask[trailing].expand_as(masked), torch.zeros_like(masked), masked
                )
                out[name] = masked
            else:
                out[name] = value
        return out

    def forward_with_cfg_drop(
        self,
        x: torch.Tensor,
        conditions: Mapping[str, torch.Tensor | None],
        drop_probs: Mapping[str, float],
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Training-time forward with per-axis condition dropout."""
        dropped = self._drop_axes(
            conditions=conditions,
            drop_probs=drop_probs,
            device=x.device,
            batch_size=x.shape[0],
            generator=generator,
        )
        return self._predict_noise(x, dropped)

    @torch.no_grad()
    def cfg_sample_step(
        self,
        x: torch.Tensor,
        conditions: Mapping[str, torch.Tensor | None],
        weights: Mapping[str, float],
        schedule: PerAxisSchedule | None = None,
        t: float = 0.0,
    ) -> torch.Tensor:
        r"""Inference-time multi-axis classifier-free guided prediction.

        Args:
            x: Current noisy sample.
            conditions: Dict of per-axis conditions (each a tensor).
            weights: Per-axis guidance scale :math:`w_i`. Axes absent from
                ``weights`` default to 0 (no guidance applied for that axis).
            schedule: Optional time-varying weight schedule bundle
                (see :class:`spectramr.models.blocks.cfg_schedule.PerAxisSchedule`).
                When supplied, the effective per-axis weight at progress
                ``t`` is ``w_i * schedule.weight_at(axis, t)``.
                When ``None``, the multiplier defaults to 1.0 for every
                axis (identity) — the output then matches the original
                constant-weight behaviour bit-exact.
            t: Normalised diffusion progress ``t in [0, 1]`` consumed by
                ``schedule``. Ignored when ``schedule`` is ``None``.

        Returns:
            The CFG-combined noise prediction
            :math:`\boldsymbol\epsilon_\theta(\mathbf{x}, \emptyset)
            + \sum_i w_i\, w_i(t)\, (\boldsymbol\epsilon_\theta(\mathbf{x}, c_i)
            - \boldsymbol\epsilon_\theta(\mathbf{x}, \emptyset))`.
        """
        # Unconditional baseline drops every axis; each guided axis is then
        # evaluated alone. All 1+k predictions go through one batched hook.
        active: list[tuple[str, float]] = []
        for axis, weight in weights.items():
            if weight == 0.0:
                continue
            if axis not in conditions or conditions[axis] is None:
                # Cannot guide on an axis whose condition is missing.
                continue
            effective_weight = weight
            if schedule is not None:
                effective_weight = weight * schedule.weight_at(axis, t)
            if effective_weight == 0.0:
                continue
            active.append((axis, effective_weight))
        sets: list[dict[str, torch.Tensor | None]] = [dict.fromkeys(conditions)]
        for axis, _ in active:
            single_axis = dict.fromkeys(conditions)
            single_axis[axis] = conditions[axis]
            sets.append(single_axis)
        predictions = self._predict_noise_many(x, sets)
        eps_null = predictions[0]
        result = eps_null.clone()
        for (_, effective_weight), eps_axis in zip(active, predictions[1:], strict=True):
            result = result + effective_weight * (eps_axis - eps_null)
        return result

    @staticmethod
    def make_default_drop_schedule(
        axes: Sequence[str],
        prob: float = 0.1,
        joint_drop_prob: float = 0.05,
    ) -> dict[str, float]:
        """Convenience: a uniform per-axis drop schedule with optional joint drop.

        The training-time per-axis dropout is ``prob`` plus a small additional
        ``joint_drop_prob`` representing the probability of all axes being
        dropped together (the pure unconditional path needed for stable CFG).
        Returns a flat dict of per-axis probabilities; the joint case is
        handled implicitly because each axis's draw is independent.
        """
        return dict.fromkeys(axes, prob + joint_drop_prob)
