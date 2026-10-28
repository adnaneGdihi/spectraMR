"""The normalization contract a range-sensitive image metric may resolve ``data_range`` from.

:func:`spectramr.core.metrics.evaluation_metrics.resolve_image_data_range` is the
one resolver; this module only carries what it needs to know about the data path,
so that ``core/`` never has to import the transform layer that applied it.

Two kinds of contract exist in the framework. Max-normalized data puts the
*peak* at magnitude 1.0 (``[0, 1]`` or ``[-1, 1]``), and the resolver infers it
from the sign of the target. A percentile scale (``kspace_scale_domain: image``)
puts a *quantile* of the coil-RSS image at 1.0 and leaves the peak free: M4Raw's
brightest voxels land near 2.05 under the 99th-percentile scale (#2241). The peak
test that verifies the first kind therefore rejects every correctly scaled
batch of the second, so the second is declared, not inferred.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

__all__ = ["DataRangeContractError", "QuantileScaleContract"]


class DataRangeContractError(ValueError):
    """The target contradicts the normalization contract the run declares.

    Not a :class:`~spectramr.core.metrics.outcome.MetricNotApplicableError`:
    that one means "no contract was declared, so there is no range", and the
    metrics computer records it as NaN. This one means the data path did not do
    what the configuration says it did, which is a pipeline defect and must stop
    the run instead of turning every score into NaN.
    """


@dataclass(frozen=True)
class QuantileScaleContract:
    """The data path divides each volume so its ``quantile`` image magnitude is 1.0.

    The reference level of such data is the quantile, so PSNR's ``MAX_I`` and
    SSIM's ``L`` are :attr:`anchor`, the value the declared ``data_range: 1.0``
    workaround of the n2n and self-supervised arms restates per arm.

    Attributes:
        quantile: The quantile of the magnitude image pinned to :attr:`anchor`,
            in ``[0.9, 1)``. ``1.0`` is max-normalization, which the sign test
            already verifies, and below 0.9 :attr:`max_peak` would be tighter than
            the dynamic range of anatomy (every corpus arm declares 0.95 or 0.99).
        source: The configuration that declared the contract, for the log line
            and for the error a violation raises.
    """

    quantile: float
    source: str

    #: The magnitude the declared quantile is scaled to.
    anchor: ClassVar[float] = 1.0

    def __post_init__(self) -> None:
        if not 0.9 <= self.quantile < 1.0:
            raise ValueError(
                f"QuantileScaleContract needs a quantile in [0.9, 1); got {self.quantile!r} "
                f"from {self.source}."
            )

    @property
    def max_peak(self) -> float:
        """The largest target magnitude the contract tolerates: ``anchor / (1 - quantile)``.

        The declaration leaves the top ``1 - quantile`` of each volume free, so no
        bound on the peak follows from it exactly, and none from a batch, which
        holds only some of a volume's slices. This one widens as the declared
        quantile moves into the tail it leaves free: 100x at the 99th percentile,
        about 50x the 2.05 measured on M4Raw, and 20x at the 95th. A scale that was
        never applied sits orders of magnitude off (``experiment_11_attention_none``
        trained on raw k-space at |k|max ~ 2400 while its config declared one).
        A scale applied twice, or to raw data in small units, lands *below* the
        anchor, and neither this test nor the sign test sees it.
        """
        return self.anchor / (1.0 - self.quantile)

    def describe(self) -> str:
        """One-line provenance for the run log."""
        return (
            f"quantile {self.quantile:g} of the magnitude image at {self.anchor:g} ({self.source})"
        )
