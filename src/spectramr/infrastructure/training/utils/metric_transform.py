"""Single resolver for the transform applied before validation metrics (#931).

Issue #931 reported ``metrics.transform`` as an inert knob with no reader.
Measurement says otherwise, and the difference decides the fix.

``MetricsMixin._compute_training_metrics`` -- reached from nine live strategies
(``diffusion``, ``gan``, ``vae``, ``reconstruction``, the two disentangled
strategies, ``field_cocycle``, and the ``adversarial`` mixin) -- passes the
``metrics`` block straight into ``_apply_metric_transforms``. So the knob *is*
read on the training-metrics path; the validation path reads
``validation.scoring.output_transform`` instead. Probed on the pre-fix tree:

===========================  =====  ==================
declared value               arms   behaviour
===========================  =====  ==================
``ifft_magnitude``              12   fires
``ifft_sense_adjoint``          57   fires
``ifft_mag_combine``           143   **silent no-op**
``ifft_mag``                     3   **silent no-op**
===========================  =====  ==================

What is dead is not the knob. It is the *spelling*: the dispatcher's
``if/elif`` chain has no ``else``, so a name no branch matches falls out of the
bottom and the tensors are returned unchanged. 146 arms name a transform that
does not exist and are told nothing.

**``ifft_mag_combine`` is deliberately NOT aliased to ``ifft_magnitude``.** The
two are operationally identical -- the implementation deleted in ``e57c21021``
was pair-R/I, ``ifft2c``, ``abs``, RSS, exactly today's ``ifft_magnitude`` --
so an alias is tempting and was the first thing tried. It is wrong here:
aliasing activates all 146, and **112 of them have ``losses.output_domain`` and
``infer_output_domain`` BOTH equal to ``image``**. Firing an IFFT on an image
output does not give those arms "what they declared", it gives them the Fourier
magnitude of an image. The declaration is a copy-paste from a k-space sibling,
not an intent, and the honest response to 146 arms naming a non-existent
transform is to say so (pitfall #9), not to guess which one they meant.

**``ifft_target_magnitude`` is the one asymmetric transform.** Every other name
treats prediction and target alike, which is right only when both live in one
domain. An image-domain network over a k-space loader (the ``self_supervised``
EI arms) predicts an image while its target is the fully sampled k-space; with no
transform the auto-``magnitude`` gate scored the image against ``|k|``, whose
maximum is the DC sample at the grid centre (#2325), and ``ifft_magnitude`` would
take the Fourier magnitude of the image instead. This one brings only the target
to the image domain (:func:`kspace_rss_magnitude`) and takes the prediction's
magnitude (:func:`image_magnitude`). It is declared, never inferred: whether a
target is k-space is not recoverable from the config on every route.

**``sense_magnitude`` needs the batch's coil maps.** It scores
:math:`|\\hat S^H F^H \\cdot|` with unit-normalised maps instead of the RSS image:
the target is always the SENSE combination of its multi-coil k-space, a coil
k-space prediction is combined the same way, and a single-image prediction (EI,
SSDU) is taken as it is. The combine is owned by
:func:`spectramr.core.metrics.sense_image_metrics.sense_scoring_pair`, which the
``sense_psnr``/``sense_ssim`` metrics also call; with no maps it raises rather
than falling back to RSS.

Precedence for the value that runs, highest first:

1. an explicit name handed in by the caller (``diffusion.py`` resolves its own);
2. ``validation.scoring.transform`` / ``validation.scoring.output_transform``.

``metrics_config`` contributes **suppression only** (``domain: none``); it never
supplies a name here. On the training path the metrics block arrives *as* the
first argument, which is how that path keeps reading ``metrics.transform`` --
unchanged from before this module existed.

**Declaring the winning key at all is the decision.** A source present but set
to ``none`` resolves to "no transform" and stops the cascade rather than
falling through. An *unset* field (Python ``None``) is absent and does fall
through.

``domain: none`` is a separate, stronger switch: it suppresses transforms
entirely, including the caller's auto-magnitude gate, and is reported as
``suppressed`` rather than as ``name is None`` because the two drive different
behaviour downstream.
"""

from __future__ import annotations

import logging
from typing import Any, NamedTuple

import torch

from spectramr.infrastructure.training.strategies.mixins.utils import (
    _get_config_value,
    _scoring_leaf,
)

logger = logging.getLogger(__name__)

#: Names ``MetricsMixin._apply_metric_transforms`` actually dispatches. Anything
#: else reaching the dispatcher is a misconfiguration and raises (pitfall #9).
IMPLEMENTED_METRIC_TRANSFORMS: frozenset[str] = frozenset(
    {
        "ifft_magnitude",
        "ifft_sense_adjoint",
        "ifft_target_magnitude",
        "magnitude",
        "sense_magnitude",
    }
)

#: Values that mean "no transform". ``"none"`` is a *truthy string*: before this
#: module it became the transform name, matched no dispatcher branch and fell
#: through silently. Normalising it had to land in the same change as
#: raise-on-unknown, or the 21 arms declaring it would start crashing.
_NO_TRANSFORM_SENTINELS: frozenset[str] = frozenset({"", "none", "null"})

#: The value of ``domain`` that switches metric transforms off wholesale.
_SUPPRESSION_SENTINEL = "none"


class MetricTransformResolution(NamedTuple):
    """Which transform won, where it was declared, and whether any may run."""

    name: str | None
    source: str | None
    suppressed: bool = False

    @property
    def is_implemented(self) -> bool:
        """``False`` only for a declared name the dispatcher cannot execute."""
        return self.name is None or self.name in IMPLEMENTED_METRIC_TRANSFORMS

    def describe(self) -> str:
        """One-line provenance for the run log (pitfall #15, obligation (c))."""
        if self.suppressed:
            return "suppressed by domain: none"
        if self.name is None:
            return "none declared"
        return f"{self.name} (from {self.source})"


def canonical_metric_transform(value: Any) -> str | None:
    """Normalise one declared transform name.

    Returns ``None`` for every spelling of "no transform" and the name itself,
    lower-cased and never swallowed, for anything else -- so the caller can
    decide whether to raise.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in _NO_TRANSFORM_SENTINELS:
        return None
    return text


def _is_suppressed(validation_config: Any, metrics_config: Any) -> bool:
    """``domain: none`` on either block switches metric transforms off."""
    for declared in (
        _scoring_leaf(validation_config, "domain"),
        _get_config_value(metrics_config, "domain", None),
    ):
        if declared is not None and str(declared).strip().lower() == (_SUPPRESSION_SENTINEL):
            return True
    return False


def resolve_metric_transform(
    validation_config: Any = None,
    metrics_config: Any = None,
    *,
    caller_override: Any = None,
    fallback_validation_config: Any = None,
) -> MetricTransformResolution:
    """Resolve the transform that will actually run.

    ``fallback_validation_config`` preserves the two-stage lookup the dispatcher
    has always had: callers hand in a config, and only if it declares nothing
    does ``self.config.validation`` get consulted. Collapsing that to a single
    source would silently drop the fallback for any caller passing a narrowed
    or per-level config.

    See the module docstring for precedence and why ``metrics_config``
    contributes suppression but never a name.
    """
    if _is_suppressed(validation_config, metrics_config) or _is_suppressed(
        fallback_validation_config, None
    ):
        return MetricTransformResolution(name=None, source=None, suppressed=True)

    candidates: tuple[tuple[str, Any], ...] = (
        ("caller", caller_override),
        (
            "validation.scoring.transform",
            _scoring_leaf(validation_config, "transform"),
        ),
        (
            "validation.scoring.output_transform",
            _scoring_leaf(validation_config, "output_transform"),
        ),
        (
            "validation.scoring.transform",
            _scoring_leaf(fallback_validation_config, "transform"),
        ),
        (
            "validation.scoring.output_transform",
            _scoring_leaf(fallback_validation_config, "output_transform"),
        ),
    )

    for source, declared in candidates:
        if declared is None:
            continue  # unset -- fall through to the next source
        return MetricTransformResolution(name=canonical_metric_transform(declared), source=source)

    return MetricTransformResolution(name=None, source=None)


def declared_metric_transforms(
    validation_config: Any = None, metrics_config: Any = None
) -> tuple[tuple[str, str], ...]:
    """Every ``(source, canonical name)`` declared, regardless of precedence.

    The audit needs this rather than :func:`resolve_metric_transform`: a name
    that loses the precedence contest on the validation path may still be the
    one the *training* path dispatches, so an unimplementable spelling has to be
    reported wherever it is written.
    """
    sources = (
        ("validation.scoring.transform", _scoring_leaf(validation_config, "transform")),
        (
            "validation.scoring.output_transform",
            _scoring_leaf(validation_config, "output_transform"),
        ),
        ("metrics.transform", _get_config_value(metrics_config, "transform", None)),
    )
    return tuple(
        (source, canonical_metric_transform(raw))
        for source, raw in sources
        if raw is not None and canonical_metric_transform(raw) is not None
    )


def image_magnitude(image: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W]`` magnitude of an image tensor, root-sum-of-squares over channels.

    Complex channels are taken as they are; an even real channel count is read as
    real/imag-interleaved complex channels (the layout every recon network emits);
    a single real channel is already a magnitude.
    """
    if not torch.is_complex(image) and image.shape[1] > 1:
        if image.shape[1] % 2 != 0:
            raise ValueError(
                f"image_magnitude: {image.shape[1]} real channels cannot be paired "
                "into real/imag complex channels."
            )
        image = torch.complex(image[:, 0::2], image[:, 1::2])
    return image.abs().pow(2).sum(dim=1, keepdim=True).sqrt()


def kspace_rss_magnitude(kspace: torch.Tensor) -> torch.Tensor:
    """``[B, 1, H, W]`` root-sum-of-squares image of multi-coil k-space.

    Pairs real/imag-interleaved channels into complex coils, moves each coil to
    the image domain through the physics SSOT, and RSS-combines. One coil gives
    its magnitude. An odd real channel count has no coil layout and raises.
    """
    from spectramr.infrastructure.physics.fft_ops import ifft2c

    if not torch.is_complex(kspace):
        if kspace.shape[1] % 2 != 0:
            raise ValueError(
                f"kspace_rss_magnitude: {kspace.shape[1]} real channels are not a "
                "real/imag-interleaved coil stack, so they cannot be k-space."
            )
        kspace = torch.complex(kspace[:, 0::2], kspace[:, 1::2])
    return image_magnitude(ifft2c(kspace))


__all__ = [
    "IMPLEMENTED_METRIC_TRANSFORMS",
    "MetricTransformResolution",
    "canonical_metric_transform",
    "declared_metric_transforms",
    "image_magnitude",
    "kspace_rss_magnitude",
    "resolve_metric_transform",
]
