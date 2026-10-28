r"""Image metrics on the SENSE coil combination, ``sense_psnr`` and ``sense_ssim``.

``psnr`` and ``ssim`` on a multi-coil arm read the root-sum-of-squares image
(``ifft_magnitude`` / ``ifft_target_magnitude``). For a noisy reference the RSS
magnitude keeps a non-central :math:`\chi_{2C}` floor everywhere, the background
included, so an output that removes noise is graded down for it (n2n PAPER.md
§ 8.4, § 8.6), and an output that is already an :math:`\hat S^H` combination cannot
reach an RSS reference at all (#2509). The SENSE combination with unit-normalised
ESPIRiT maps,

.. math::

    \hat m = \frac{\sum_c S_c^{*}\,(F^H y)_c}{\sum_c |S_c|^2},

recovers :math:`m` inside the map support, carries one coil-equivalent noise
(Rician rather than :math:`\chi_{2C}`), and is zero where the maps are zero.

The two metrics are registered with ``needs=("coil_maps",)``, so the validation
path routes them through ``compute_context_metrics`` with the batch's maps. They
are extra columns beside ``psnr``/``ssim``, not replacements: which combination
the paper reports is an open owner decision.

:func:`sense_scoring_pair` is the one owner of the combine. The
``sense_magnitude`` metric transform calls it too.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from spectramr.core.metrics.coil_manifold import as_complex_coils
from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.evaluation_metrics import PSNR, SSIMMetric
from spectramr.core.metrics.registry import register_metric
from spectramr.infrastructure.physics.fft_ops import sense_adjoint

__all__ = [
    "SENSE_METRIC_NAMES",
    "SenseMapsMissingError",
    "SensePSNR",
    "SenseSSIM",
    "sense_scoring_pair",
]


#: The metrics that combine with maps rather than grade coil data; the validation
#: path hands them the scoring maps when the batch carries them (#2664).
SENSE_METRIC_NAMES: frozenset[str] = frozenset({"sense_psnr", "sense_ssim"})


class SenseMapsMissingError(ValueError):
    """A SENSE-combined score was asked for and the batch carries no coil maps.

    A distinct type so the validation loop re-raises it rather than degrading to
    its fallback PSNR, which would score the RSS image under a SENSE name.
    """


def _sense_combine(kspace: Tensor, maps: Tensor) -> Tensor:
    """Complex ``[B, 1, H, W]`` :math:`\\hat S^H F^H y` with normalised maps."""
    return sense_adjoint(kspace, maps, normalize=True)


def _image_prediction(prediction: Tensor) -> Tensor | None:
    """The one image a combined prediction holds, or ``None`` if it is coil data."""
    if torch.is_complex(prediction):
        return prediction if prediction.shape[1] == 1 else None
    if prediction.shape[1] == 1:
        return prediction.abs()
    if prediction.shape[1] == 2:
        return torch.complex(prediction[:, 0:1], prediction[:, 1:2])
    return None


def sense_scoring_pair(
    prediction: Tensor, target: Tensor, maps: Tensor | None, *, metric: str = "sense_magnitude"
) -> tuple[Tensor, Tensor]:
    """``(prediction, target)`` as SENSE images, ``[B, 1, H, W]``.

    The target is multi-coil k-space and always goes through the combine. The
    prediction goes through it when it is coil k-space (C coils, as the maps);
    a single image, complex or magnitude, is already a combination and is taken
    as it is. A real 1-channel prediction stays real; every other result is
    complex, so the caller can keep the phase for a phase metric.

    Raises:
        SenseMapsMissingError: no maps reached the metric.
        ValueError: maps that are not complex, a single coil (an image and coil
            k-space are then the same shape, and there is nothing to combine),
            or a prediction that is neither coil k-space nor one image.
    """
    if maps is None:
        raise SenseMapsMissingError(
            f"{metric} needs coil sensitivity maps and the batch carries none. Declare "
            "data.coil_maps_root or the `espirit_sensitivity` transform; an arm with one "
            "virtual coil has no maps and scores psnr/ssim on its magnitude instead."
        )
    if not torch.is_complex(maps):
        raise ValueError(f"{metric} needs COMPLEX coil maps, got {maps.dtype}.")
    coils = maps.shape[1]
    if coils < 2:
        raise ValueError(
            f"{metric}: one coil has nothing to combine, and its image cannot be told "
            "from its k-space by shape."
        )
    target_k = as_complex_coils(target, name="target")
    if target_k.shape[1] != coils:
        raise ValueError(
            f"{metric}: target has {target_k.shape[1]} coils and the maps {coils}; the "
            "target must be the reference's multi-coil k-space."
        )
    image = _image_prediction(prediction)
    if image is None:
        pred_k = as_complex_coils(prediction, name="prediction")
        if pred_k.shape[1] != coils:
            raise ValueError(
                f"{metric}: prediction has {pred_k.shape[1]} coils and the maps {coils}."
            )
        image = _sense_combine(pred_k, maps)
    return image, _sense_combine(target_k, maps)


def _magnitudes(
    name: str, prediction: Tensor, target: Tensor | None, ctx: MetricContext
) -> tuple[Tensor, Tensor]:
    if target is None:
        raise ValueError(f"{name} is a full-reference metric and no target reached it.")
    pred, ref = sense_scoring_pair(prediction, target, ctx.coil_maps, metric=name)
    return pred.abs(), ref.abs()


class _SenseImageMetric:
    """Combine both tensors, then hand them to the canonical image metric.

    ``data_range`` and ``scale_contract`` are forwarded unchanged, so the
    column shares the range the arm declares for ``psnr``/``ssim``.
    ``domain`` is the computer's tag and is not forwarded: after the combine
    the tensors are images whatever the arm's output domain.
    """

    name = ""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        data_range: float | None = None,
        scale_contract: Any = None,
        domain: str | None = None,
        **kwargs: Any,
    ) -> Tensor:
        del domain
        ctx = resolve_context(context, kwargs)
        pred, ref = _magnitudes(self.name, prediction, target, ctx)
        return self._inner(pred, ref, data_range=data_range, scale_contract=scale_contract)


@register_metric("sense_psnr", requires_reference=True, needs=("coil_maps",), direction="higher")
class SensePSNR(_SenseImageMetric):
    """PSNR on :math:`|\\hat S^H F^H \\cdot|` of prediction and reference."""

    name = "sense_psnr"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__(PSNR(device=device))


@register_metric("sense_ssim", requires_reference=True, needs=("coil_maps",), direction="higher")
class SenseSSIM(_SenseImageMetric):
    """SSIM on :math:`|\\hat S^H F^H \\cdot|` of prediction and reference."""

    name = "sense_ssim"

    def __init__(self, device: str | torch.device = "cpu") -> None:
        super().__init__(SSIMMetric(device=device))
