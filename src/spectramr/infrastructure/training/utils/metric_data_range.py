"""Which normalization contract the image metrics may resolve ``data_range`` from (#2241).

The one place a run's configuration is turned into the
:class:`~spectramr.core.metrics.data_range_contract.QuantileScaleContract` that
:func:`~spectramr.core.metrics.evaluation_metrics.resolve_image_data_range`
consumes. Both metrics computers (training and validation) and the fallback PSNR
read it through the computer, so ``val_psnr`` means one thing on every path.
"""

from __future__ import annotations

from typing import Any

from spectramr.core.metrics.data_range_contract import QuantileScaleContract

__all__ = ["scale_contract_from_config"]

#: The scale domains whose quantile is taken on the coil-RSS image the metrics grade.
#: ``acs`` and ``acs_window`` are left out: they take the quantile of the image of
#: the centred calibration region only, whose level relative to the full-resolution
#: image has not been measured, so an arm on them declares its range.
_IMAGE_QUANTILE_DOMAINS = frozenset({"image"})


def scale_contract_from_config(config: Any) -> QuantileScaleContract | None:
    """Resolve the percentile scale the data path applies, if it fixes the image level.

    Read through :meth:`KSpaceNormalizationSpec.from_data_config`, the one
    resolver of the k-space normalization knobs, so the M4Raw preset defaults it
    injects are seen here as they are by the transform that applies them.

    Returns ``None`` when the normalization is off, measures the quantile in
    k-space (which says nothing about the image level), or ``log1p``-compresses
    k-space: the metric transforms IFFT the compressed spectrum on every
    non-diffusion path, so its image is not on the declared scale. Also ``None``
    at ``kspace_percentile: 1.0``, which is max-normalization and is verified by
    the sign test the resolver falls back to.

    Args:
        config: The run's ``TrainingSettings``, or any object with ``data``.

    Returns:
        The contract, or ``None`` when the configuration declares none.
    """
    from spectramr.data.transforms.normalization import KSpaceNormalizationSpec

    data = getattr(config, "data", None)
    if data is None or getattr(data, "processing", None) is None:
        return None
    spec = KSpaceNormalizationSpec.from_data_config(data)
    if not spec.enabled or spec.scale_domain not in _IMAGE_QUANTILE_DOMAINS or spec.log_compressed:
        return None
    if spec.percentile >= 1.0:
        return None
    return QuantileScaleContract(
        quantile=spec.percentile,
        source=(
            "data.processing: enable_kspace_normalization=true, "
            f"kspace_scale_domain={spec.scale_domain}, kspace_percentile={spec.percentile:g}"
        ),
    )
