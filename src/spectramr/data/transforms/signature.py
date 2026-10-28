"""Stable, deterministic signatures for TorchIO transform chains.

Phase 2 of ``TODO/audit/data_layer_unification_plan.md`` introduces a
**transform parity contract**: inference must use the same transform
chain that produced the training data, modulo augmentations. The
mechanism is a stable hash recorded in the checkpoint at training
time, then re-computed at inference time and diffed.

The signature is a short hex string derived from:

- Each transform's class name (canonical, not instance ``repr``).
- A small set of "load-bearing" kwargs per transform — the ones that
  meaningfully change the output (e.g. ``out_min_max`` for
  ``RescaleIntensity``, ``percentiles`` for normalization). Stochastic
  state (RNG seeds, augmentation probabilities) is intentionally
  excluded — augmentations are expected to differ between train and
  infer, and asserting equality on a probability would always trip.

Design notes:

- **Order matters.** Two chains with the same transforms but reordered
  produce different signatures — which is correct, since reordering
  normalization vs. resampling changes the output.
- **Augmentation classes are dropped from the signature.** The parity
  contract asserts "train transforms minus augmentation == infer
  transforms"; we exclude aug classes from both sides before hashing.
- **Forward-compatible.** When a new transform with new kwargs lands,
  it shows up in the signature automatically via its class name. Only
  if its behavior depends on a non-default kwarg do we need to add it
  to ``_LOAD_BEARING_KWARGS``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)


# Classes whose presence/absence between train and infer is *expected*
# to differ — they are augmentations applied only at train time.
_AUGMENTATION_CLASS_NAMES: frozenset[str] = frozenset(
    {
        "RandomAffine",
        "RandomFlip",
        "RandomNoise",
        "RandomBiasField",
        "RandomElasticDeformation",
        "RandomMotion",
        "RandomGhosting",
        "RandomSpike",
        "RandomBlur",
        "RandomGamma",
        "RandomSwap",
        "RandomLabelsToImage",
        "RandomAnisotropy",
        "OneOf",
        # spectraMR-specific augmentations
        "PhaseAugmentation",
        "RandomPhaseAug",
        "BenchmarkingDegradations",
        "RealisticDegradations",
        "SyntheticLesion",
    }
)


# Per-transform-class, the kwargs that materially change the output.
# Any kwarg NOT listed here is dropped from the signature — keeps the
# hash stable when an unrelated default changes between TorchIO versions.
_LOAD_BEARING_KWARGS: dict[str, tuple[str, ...]] = {
    "Resample": ("target", "image_interpolation", "label_interpolation"),
    "CropOrPad": ("target_shape", "padding_mode", "mask_name"),
    "RescaleIntensity": ("out_min_max", "percentiles"),
    "ZNormalization": ("masking_method",),
    "ToCanonical": (),
    "EnsureShapeMultiple": ("target_multiple",),
    "_ResampleToReferenceTransform": ("target_shape",),
    "_EnsureMinimumSpatialSize": ("min_size",),
    "CoilCombineTransform": ("method",),
    "Compose": (),  # Composite — children handled by recursion.
}


def _classname(obj: Any) -> str:
    """Return the canonical class name of ``obj``."""
    return type(obj).__name__


def _is_augmentation(obj: Any) -> bool:
    return _classname(obj) in _AUGMENTATION_CLASS_NAMES


def _serialize_kwargs(transform: Any) -> dict[str, Any]:
    """Extract load-bearing kwargs from ``transform`` as a JSON-safe dict.

    Falls back to an empty dict for transforms we don't know about —
    their class name alone goes into the signature.
    """
    cls = _classname(transform)
    keys = _LOAD_BEARING_KWARGS.get(cls)
    if keys is None:
        return {}
    out: dict[str, Any] = {}
    for k in keys:
        if hasattr(transform, k):
            v = getattr(transform, k)
            # Convert tuples → lists for JSON; leave primitives alone.
            if isinstance(v, tuple):
                v = list(v)
            try:
                json.dumps(v)  # round-trip check
                out[k] = v
            except (TypeError, ValueError):
                out[k] = repr(v)
    return out


def _flatten_compose(transforms: Iterable[Any]) -> list[Any]:
    """Flatten ``tio.Compose`` (or any nested ``transforms`` attribute)
    into a flat list. Compose itself is not recorded — only its
    children, in order."""
    out: list[Any] = []
    for t in transforms:
        if _classname(t) == "Compose" and hasattr(t, "transforms"):
            out.extend(_flatten_compose(t.transforms))
        else:
            out.append(t)
    return out


def compute_transform_signature(
    transforms: Iterable[Any] | None,
    *,
    drop_augmentations: bool = True,
) -> str:
    """Compute a stable SHA-256 signature of a transform chain.

    Args:
        transforms: Iterable of TorchIO transforms, a ``tio.Compose``,
            or ``None``. ``None`` returns the empty-chain signature so
            checkpoints without transforms produce a deterministic
            value rather than failing.
        drop_augmentations: When True (default), augmentation classes
            are stripped before hashing. Set False for "exact-train
            chain" signatures (rarely useful — parity is the common case).

    Returns:
        64-character hex string. Two chains with the same canonical
        class-name + load-bearing-kwargs sequence hash identically.
    """
    if transforms is None:
        flat: list[Any] = []
    elif hasattr(transforms, "transforms"):  # tio.Compose
        flat = _flatten_compose(transforms.transforms)
    else:
        flat = _flatten_compose(transforms)

    if drop_augmentations:
        flat = [t for t in flat if not _is_augmentation(t)]

    chain: list[dict[str, Any]] = []
    for t in flat:
        chain.append(
            {
                "cls": _classname(t),
                "kwargs": _serialize_kwargs(t),
            }
        )
    blob = json.dumps(chain, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def diff_signatures(
    train_sig: str | None,
    infer_sig: str | None,
) -> str | None:
    """Compare two signatures. Returns ``None`` when equal, else a
    human-readable message naming the divergence.

    Both ``None`` is treated as equal (no signature recorded at either
    end → no constraint to enforce).

    Used by the Phase-2 ``strict_train_parity`` check at inference load
    time, and by ``tests/unit/data/test_inference_train_parity.py``.
    """
    if train_sig is None and infer_sig is None:
        return None
    if train_sig is None:
        return (
            "Training-time transform signature missing (checkpoint pre-dates "
            "Phase 2). Cannot enforce inference/train parity."
        )
    if infer_sig is None:
        return (
            "Inference transform signature could not be computed (likely "
            "empty transform chain). Refuse to load when strict_train_parity=True."
        )
    if train_sig == infer_sig:
        return None
    return (
        f"Transform parity broken: train_sig={train_sig[:16]}… ≠ "
        f"infer_sig={infer_sig[:16]}…. Inference transforms diverge from "
        "the chain recorded at training time. Inspect "
        "src/data/builders/torchio_transform_builder.py and your YAML's "
        "data.* fields for changes since the checkpoint was written."
    )


def compute_infer_signature(config: Any) -> str:
    """Signature of the val-mode transform chain for a resolved config.

    Inference reuses the val chain: it is the post-training "frozen"
    pipeline, modulo augmentation. This is the value compared against the
    ``transform_signature`` the checkpoint recorded at training time.
    """
    from spectramr.data.builders.torchio_transform_builder import (
        TorchIOTransformBuilder,
        TorchIOTransformConfig,
    )

    tcfg = TorchIOTransformConfig.from_settings(config)
    return compute_transform_signature(TorchIOTransformBuilder.build_val_transforms(tcfg))


def enforce_train_infer_parity(
    config: Any,
    checkpoint_signature: str | None,
) -> str:
    """Refuse to proceed when the inference chain diverges from training's.

    Gated on ``data.modes.infer.strict_train_parity``. When the flag is off
    this is a no-op that still returns the computed signature, so callers can
    log or stamp it either way.

    Args:
        config: Resolved ``TrainingSettings``.
        checkpoint_signature: ``transform_signature`` recorded in the
            checkpoint at training time; ``None`` for pre-Phase-2 checkpoints.

    Returns:
        The computed inference-time signature.

    Raises:
        RuntimeError: When the flag is on and the signatures diverge (which
            includes a checkpoint that recorded no signature at all -- that
            cannot satisfy parity, so the user must opt out explicitly).
    """
    # A config with no `modes:` block simply has no infer mode to consult, so
    # the gate is off. Anything ELSE that goes wrong while resolving it must
    # propagate: the inherited `except Exception: strict = False` meant a
    # malformed modes block silently DISABLED the one check whose entire job is
    # to fail loud (#9), and 132 arms declare `modes:`.
    resolve_mode = getattr(config.data, "resolve_mode", None)
    if resolve_mode is None:
        strict = False
    else:
        infer_mode = resolve_mode("infer")
        strict = bool(getattr(infer_mode, "strict_train_parity", False))

    infer_signature = compute_infer_signature(config)
    if not strict:
        return infer_signature

    divergence = diff_signatures(checkpoint_signature, infer_signature)
    if divergence is not None:
        raise RuntimeError(f"strict_train_parity=true rejected the inference load. {divergence}")
    return infer_signature


__all__ = [
    "compute_infer_signature",
    "compute_transform_signature",
    "diff_signatures",
    "enforce_train_infer_parity",
]
