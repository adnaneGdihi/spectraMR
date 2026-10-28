r"""Multi-axis classifier-free guidance for :class:`LatentDiffusionGenerator`.

Two condition axes exist on the latent ULF->HF path:

* ``condition``: the SR3 ULF latent concatenated to the noisy latent. Null is
  a zero latent, the convention the generator's training drop already uses.
* ``contrast``: the contrast embedding added to the U-Net decoder embedding.
  Null is a zero embedding row, which equals the ``contrast_idx=None`` path.
  The drop acts on the embedding, never on ``contrast_idx``, because index 0
  is T1w, not "no contrast".

Training drops each axis independently per sample with ``cfg_drop_probs``.
Sampling combines the axes listed in ``cfg_weights`` with
:meth:`MultiAxisCFGMixin.cfg_sample_step`,

.. math::

   \tilde\epsilon = \epsilon(\emptyset) + \sum_i w_i\,[\epsilon(c_i) - \epsilon(\emptyset)],

where :math:`\emptyset` nulls the guided axes only. An axis that is declared
but not guided stays on in every term, so a single-axis control keeps its
contrast conditioning at sampling, as it had in training. The 1+k predictions
run as one batched U-Net call.

Absent knobs reproduce the generator's historical behaviour exactly: a 10%
drop of the ULF condition, drawn with the same ``torch.bernoulli`` call, and
unguided sampling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import torch

from spectramr.models.blocks.multi_axis_cfg import MultiAxisCFGMixin

__all__ = [
    "LATENT_CFG_AXES",
    "LEGACY_CONDITION_DROP",
    "LatentCFGGuide",
    "LatentCFGSpec",
    "axis_keep_masks",
    "guided_noise",
    "resolve_latent_cfg",
]

#: Axes the latent ULF->HF path can condition on, in the order masks are drawn.
LATENT_CFG_AXES: tuple[str, ...] = ("condition", "contrast")

#: The ULF-condition dropout the generator always applied in training.
LEGACY_CONDITION_DROP = 0.1

_NO_DATA_AXES = {
    "thickness": "no slice-thickness label reaches the batch",
    "pathology": "no pathology mask reaches the batch",
}


@dataclass(frozen=True)
class LatentCFGSpec:
    """Resolved per-axis dropout (training) and guidance weights (sampling)."""

    drop_probs: Mapping[str, float] = field(default_factory=dict)
    weights: Mapping[str, float] = field(default_factory=dict)

    @property
    def guided(self) -> bool:
        return any(w != 0.0 for w in self.weights.values())

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"cfg_drop_probs": dict(self.drop_probs), "cfg_weights": dict(self.weights)}


def _check_axes(name: str, mapping: Mapping[str, object], available: set[str]) -> None:
    for axis in mapping:
        if axis in _NO_DATA_AXES:
            raise ValueError(
                f"{name}[{axis!r}]: {_NO_DATA_AXES[axis]}, so this axis cannot be "
                f"conditioned on. Supported axes: {list(LATENT_CFG_AXES)}."
            )
        if axis not in LATENT_CFG_AXES:
            raise ValueError(f"{name}: unknown axis {axis!r}; expected {list(LATENT_CFG_AXES)}.")
        if axis not in available:
            raise ValueError(
                f"{name}[{axis!r}]: this model does not condition on {axis!r} "
                "(condition needs conditional_translation, contrast needs "
                "use_contrast_guidance)."
            )


def resolve_latent_cfg(
    drop_probs: Mapping[str, object] | None,
    weights: Mapping[str, object] | None,
    *,
    has_condition: bool,
    has_contrast: bool,
) -> LatentCFGSpec:
    """Validate ``cfg_drop_probs`` / ``cfg_weights``; unknown or unusable axes raise."""
    available = {a for a, on in (("condition", has_condition), ("contrast", has_contrast)) if on}
    if drop_probs is None:
        drops = {"condition": LEGACY_CONDITION_DROP} if has_condition else {}
    else:
        if not isinstance(drop_probs, Mapping):
            raise ValueError(f"cfg_drop_probs must be a mapping of axis -> p; got {drop_probs!r}.")
        _check_axes("cfg_drop_probs", drop_probs, available)
        drops = {}
        for axis, p in drop_probs.items():
            p = float(p)  # type: ignore[arg-type]
            if not 0.0 <= p < 1.0:
                raise ValueError(f"cfg_drop_probs[{axis!r}] must be in [0, 1); got {p}.")
            drops[axis] = p
    guide: dict[str, float] = {}
    if weights is not None:
        if not isinstance(weights, Mapping):
            raise ValueError(f"cfg_weights must be a mapping of axis -> w; got {weights!r}.")
        _check_axes("cfg_weights", weights, available)
        for axis, w in weights.items():
            w = float(w)  # type: ignore[arg-type]
            if w < 0.0:
                raise ValueError(f"cfg_weights[{axis!r}] must be >= 0; got {w}.")
            if w != 0.0 and drops.get(axis, 0.0) <= 0.0:
                raise ValueError(
                    f"cfg_weights[{axis!r}]={w} guides an axis that is never dropped in "
                    "training, so its null branch is untrained. Set cfg_drop_probs for it."
                )
            guide[axis] = w
    return LatentCFGSpec(drop_probs=drops, weights=guide)


def axis_keep_masks(
    spec: LatentCFGSpec, batch: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """Per-sample keep masks (1 = keep, 0 = null) for every axis with a drop."""
    masks: dict[str, torch.Tensor] = {}
    for axis in LATENT_CFG_AXES:
        p = spec.drop_probs.get(axis, 0.0)
        if p > 0.0:
            masks[axis] = torch.bernoulli(torch.zeros(batch, device=device) + (1 - p))
    return masks


#: ``(unet_input, timesteps, contrast_keep) -> eps`` for one stacked batch.
DenoiseFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


class LatentCFGGuide(MultiAxisCFGMixin):
    """Adapter that lets the mixin's combination formula drive the LDM denoiser.

    Composition rather than a base class of the generator, so the generator's
    subclasses (``LatentGaussianDiffusion``, the residual head) keep their
    inheritance depth. Built per call, holding no parameters.
    """

    def __init__(
        self,
        denoise: DenoiseFn,
        timesteps: torch.Tensor,
        cond_latent: torch.Tensor | None,
        has_contrast: bool,
    ) -> None:
        super().__init__()
        self._denoise = denoise
        self._t = timesteps
        self._cond = cond_latent
        self._has_contrast = has_contrast

    def _predict_noise(self, x, conditions):
        return self._predict_noise_many(x, [conditions])[0]

    def _predict_noise_many(
        self, x: torch.Tensor, condition_sets: Sequence[Mapping[str, torch.Tensor | None]]
    ) -> list[torch.Tensor]:
        inputs, keeps = [], []
        ones = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        for conditions in condition_sets:
            if self._cond is not None:
                cond = conditions.get("condition", self._cond)
                inputs.append(torch.cat([x, torch.zeros_like(x) if cond is None else cond], dim=1))
            else:
                inputs.append(x)
            if "contrast" in conditions:
                keeps.append(ones if conditions["contrast"] is not None else torch.zeros_like(ones))
            else:
                keeps.append(ones)
        n = len(condition_sets)
        stacked = torch.cat(inputs, dim=0)
        keep = torch.cat(keeps, dim=0) if self._has_contrast else None
        eps = self._denoise(stacked, self._t.repeat(n), keep)
        return list(eps.chunk(n, dim=0))


def guided_noise(
    denoise: DenoiseFn,
    z_t: torch.Tensor,
    timesteps: torch.Tensor,
    cond_latent: torch.Tensor | None,
    contrast_available: bool,
    spec: LatentCFGSpec,
) -> torch.Tensor:
    """The multi-axis CFG noise estimate for one reverse step (2-D latents)."""
    conditions: dict[str, torch.Tensor | None] = {}
    if "condition" in spec.weights:
        conditions["condition"] = cond_latent
    if "contrast" in spec.weights:
        conditions["contrast"] = (
            torch.ones(z_t.shape[0], device=z_t.device) if contrast_available else None
        )
    guide = LatentCFGGuide(denoise, timesteps, cond_latent, contrast_available)
    return guide.cfg_sample_step(z_t, conditions, dict(spec.weights))
