r"""Latent diffusion with multi-axis classifier-free guidance declared as axis lists.

``latent_diffusion_multi_axis_cfg`` is :class:`LatentDiffusionGenerator` with the
CFG knobs spelled as parallel lists::

    cfg_axes:       [condition, contrast]
    cfg_weights:    [4.0, 1.5]
    cfg_drop_probs: [0.1, 0.1]      # or a mapping axis -> p
    cfg_schedule:   {condition: {type: linear, start: 6.0, end: 1.0}}

The lists are translated into the generator's mapping form, so the one owner of
axis validation, training dropout and guided sampling stays
:func:`~spectramr.models.generators.latent_cfg.resolve_latent_cfg`. An axis outside
the latent path's set (``condition``, ``contrast``) raises there, as does an axis
with no data in the batch (``pathology``, ``thickness``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spectramr.models.generators.latent_diffusion_generator import LatentDiffusionGenerator
from spectramr.models.registry import register_model

__all__ = ["LatentDiffusionMultiAxisCFG", "translate_axis_lists"]


def _as_list(name: str, value: object) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a list; got {value!r}.")
    return list(value)


def translate_axis_lists(
    cfg_axes: object,
    cfg_weights: object,
    cfg_drop_probs: object = None,
) -> tuple[dict[str, float] | None, dict[str, float]]:
    """Turn ``cfg_axes`` + list ``cfg_weights`` (+ ``cfg_drop_probs``) into mappings.

    Returns ``(drop_probs, weights)``; ``drop_probs`` is ``None`` when undeclared so
    the generator's own default applies. Shape errors raise; axis legality is
    checked downstream by ``resolve_latent_cfg``.
    """
    axes = [str(a) for a in _as_list("cfg_axes", cfg_axes)]
    if not axes:
        raise ValueError("cfg_axes must name at least one axis.")
    if len(set(axes)) != len(axes):
        raise ValueError(f"cfg_axes repeats an axis: {axes}.")
    weights = _as_list("cfg_weights", cfg_weights)
    if len(weights) != len(axes):
        raise ValueError(
            f"cfg_weights has {len(weights)} entries but cfg_axes names {len(axes)}: "
            "the lists are parallel, one weight per axis."
        )
    weight_map = {axis: float(w) for axis, w in zip(axes, weights, strict=True)}
    if cfg_drop_probs is None:
        return None, weight_map
    if isinstance(cfg_drop_probs, Mapping):
        unknown = sorted(set(map(str, cfg_drop_probs)) - set(axes))
        if unknown:
            raise ValueError(f"cfg_drop_probs names axes outside cfg_axes: {unknown}.")
        return {str(a): float(p) for a, p in cfg_drop_probs.items()}, weight_map
    drops = _as_list("cfg_drop_probs", cfg_drop_probs)
    if len(drops) != len(axes):
        raise ValueError(f"cfg_drop_probs has {len(drops)} entries but cfg_axes names {len(axes)}.")
    return {axis: float(p) for axis, p in zip(axes, drops, strict=True)}, weight_map


# Capabilities mirror ``latent_diffusion`` exactly (test_registry_capability_parity.py).
@register_model(
    name="latent_diffusion_multi_axis_cfg",
    training_mode="diffusion",
    supports_contrast_conditioning=True,
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class LatentDiffusionMultiAxisCFG(LatentDiffusionGenerator):
    """Latent diffusion whose CFG axes, weights and schedule are declared as lists."""

    def __init__(
        self,
        *args: Any,
        cfg_axes: object = None,
        cfg_weights: object = None,
        cfg_drop_probs: object = None,
        **kwargs: Any,
    ) -> None:
        if cfg_axes is None or cfg_weights is None:
            raise ValueError(
                "latent_diffusion_multi_axis_cfg needs cfg_axes and cfg_weights (parallel "
                "lists); without them it is plain latent_diffusion."
            )
        drops, weights = translate_axis_lists(cfg_axes, cfg_weights, cfg_drop_probs)
        if drops is not None:
            kwargs["cfg_drop_probs"] = drops
        super().__init__(*args, cfg_weights=weights, **kwargs)
