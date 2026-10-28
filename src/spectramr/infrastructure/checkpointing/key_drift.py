"""Name the config knob behind a state-dict key mismatch that torch reports as raw keys.

``nn.utils.spectral_norm`` replaces ``<m>.weight`` with ``<m>.weight_orig`` plus the
``weight_u``/``weight_v`` power-iteration buffers. A run whose ``spectral_norm`` knob changed
between the checkpoint and the resume therefore fails with fifteen key names and no hint at
the cause (experiment_50 strict-physical: the YAML flipped the knob after the checkpoint was
written). The refusal itself is correct; this only says why.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from spectramr.core.module_utils import strip_wrapper_prefixes

_SPECTRAL_SUFFIXES = ("weight_orig", "weight_u", "weight_v")


def _plain_name(key: str) -> str | None:
    """``a.b.weight_orig`` -> ``a.b.weight``; ``None`` when ``key`` is not a spectral-norm name."""
    for suffix in _SPECTRAL_SUFFIXES:
        if key == suffix or key.endswith("." + suffix):
            return key[: len(key) - len(suffix)] + "weight"
    return None


def spectral_norm_drift(model_keys: Iterable[str], state_keys: Iterable[str]) -> str | None:
    """Which side carries spectral norm, when that alone explains the key mismatch.

    Returns ``None`` when the key sets agree or differ for any other reason, so an unrelated
    architecture change is never mislabelled.
    """
    model, state = set(model_keys), set(state_keys)
    missing, unexpected = model - state, state - model
    if not missing or not unexpected:
        return None
    as_plain_unexpected = {_plain_name(k) for k in unexpected}
    as_plain_missing = {_plain_name(k) for k in missing}
    if None not in as_plain_unexpected and as_plain_unexpected == missing:
        return "the checkpoint was written WITH spectral norm; this model is built without it"
    if None not in as_plain_missing and as_plain_missing == unexpected:
        return "the checkpoint was written WITHOUT spectral norm; this model is built with it"
    return None


def drift_hint(state: Mapping[str, Any] | None, live: Mapping[str, Any] | None) -> str | None:
    """A one-line explanation for the first live model whose keys differ only by spectral norm."""
    if not state or not live:
        return None
    for name, module in live.items():
        saved = (state.get("models") or {}).get(name)
        if saved is None or not hasattr(module, "state_dict"):
            continue
        found = spectral_norm_drift(module.state_dict(), strip_wrapper_prefixes(saved))
        if found:
            return (
                f"{name}: {found}. Set that model's `spectral_norm` knob (for the "
                "discriminator, model.discriminator_component.kwargs.spectral_norm) to what "
                "the checkpoint used, or start the run fresh."
            )
    return None
