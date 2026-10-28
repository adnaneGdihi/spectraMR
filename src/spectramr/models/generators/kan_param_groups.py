"""The KAN parameter group: its ratios, its membership and its provenance record.

``kan_lr_ratio`` and ``kan_weight_decay_ratio`` scale the learning rate and weight
decay of the B-spline coefficients, ``KANLayer.spline_weight``, and of nothing else
(#2105). Selecting whole KAN-tagged blocks put 25.9 % of the reference arm's model on
the reduced LR, so the KAN-vs-MLP matrix confounded the gate's parameterisation with
a 10x learning rate on the attention block. Base weights, gates and biases stay at
the base LR. The ratios are validated by :func:`resolve_kan_param_group_ratios`,
which the build and the audit share.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import nn

#: The model_kwargs keys this module owns, with the value each takes when undeclared.
KAN_RATIO_DEFAULTS: dict[str, float] = {"kan_lr_ratio": 1.0, "kan_weight_decay_ratio": 1.0}

#: The key the KAN group's record is stamped under, inside ``provenance["model"]``.
KAN_PARAM_GROUP_PROVENANCE_KEY = "kan_param_group"


def _ratio(key: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"model_kwargs.{key} must be a positive number, got {value!r} ({type(value).__name__})."
        )
    ratio = float(value)
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"model_kwargs.{key} must be a positive finite number, got {ratio!r}.")
    return ratio


def resolve_kan_param_group_ratios(model_kwargs: Mapping[str, Any]) -> tuple[float, float]:
    """``(kan_lr_ratio, kan_weight_decay_ratio)``; raises ``ValueError`` on a malformed value."""
    # Literal reads: the model_kwargs read census collects ``.get("...")`` string literals.
    lr = _ratio("kan_lr_ratio", model_kwargs.get("kan_lr_ratio", 1.0))
    wd = _ratio("kan_weight_decay_ratio", model_kwargs.get("kan_weight_decay_ratio", 1.0))
    return lr, wd


def kan_spline_coefficients(model: nn.Module) -> list[nn.Parameter]:
    """The trainable ``spline_weight`` of every ``KANLayer`` under *model*, in module order."""
    # Function-local: the audit imports this module and must not pull in the block library.
    from spectramr.models.blocks.kan_layer import KANLayer

    return [
        m.spline_weight
        for m in model.modules()
        if isinstance(m, KANLayer) and m.spline_weight.requires_grad
    ]


def build_kan_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    lr_ratio: float,
    weight_decay_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split *model*'s trainable parameters into a ``main`` and a ``kan`` optimizer group.

    Returns the groups and the record stamped into provenance. The ``kan`` group is
    omitted when the model has no spline coefficients; the record is not, so a declared
    ratio that reaches no parameter reads ``n_params: 0`` rather than vanishing.
    """
    kan_params = kan_spline_coefficients(model)
    kan_ids = {id(p) for p in kan_params}
    trainable = [p for p in model.parameters() if p.requires_grad]
    main_params = [p for p in trainable if id(p) not in kan_ids]

    groups: list[dict[str, Any]] = []
    if main_params:
        groups.append({"params": main_params, "lr": base_lr, "weight_decay": weight_decay})
    if kan_params:
        groups.append(
            {
                "params": kan_params,
                "lr": base_lr * lr_ratio,
                "weight_decay": weight_decay * weight_decay_ratio,
                "name": "kan",
            }
        )
    n_kan = sum(p.numel() for p in kan_params)
    n_trainable = sum(p.numel() for p in trainable)
    record = {
        "selector": "KANLayer.spline_weight",
        "kan_lr_ratio": lr_ratio,
        "kan_weight_decay_ratio": weight_decay_ratio,
        "n_tensors": len(kan_params),
        "n_params": n_kan,
        "n_trainable_params": n_trainable,
        "fraction_of_trainable": (n_kan / n_trainable) if n_trainable else 0.0,
    }
    return groups, record


__all__ = [
    "KAN_PARAM_GROUP_PROVENANCE_KEY",
    "KAN_RATIO_DEFAULTS",
    "build_kan_param_groups",
    "kan_spline_coefficients",
    "resolve_kan_param_group_ratios",
]
