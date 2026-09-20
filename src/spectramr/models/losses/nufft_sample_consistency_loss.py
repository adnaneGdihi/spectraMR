"""Sample-domain data consistency for a non-Cartesian acquisition.

``pre_dc_kspace`` is the Cartesian cohort's fidelity term: an L1 between the
network's pre-DC k-space and the truth, weighted by the bins the rung did not
acquire. It has no correct non-Cartesian reading, because the weight it uses is
a Cartesian sampling indicator and an off-grid acquisition measures no Cartesian
bin. This is its twin on the axis where the measurement lives:

.. math::

    \\mathcal{L} = \\frac{\\lVert m_t \\odot (A(\\hat{x}) - y) \\rVert_1}
                        {\\lVert m_t \\rVert_1}

with :math:`A` the NUFFT onto the trajectory and :math:`m_t` the rung's spoke
mask. It asks the only question the measurement can answer: does the
reconstruction reproduce the samples that were actually collected?

The gridded state :math:`A^H(\\mathrm{dcf} \\odot m_t \\odot A(x_0))` does NOT
satisfy this, which is why the term exists. Gridding is one adjoint step, not a
projection -- ``F_NU F_NU^H != I`` when the samples are off-grid and fewer than
the image has pixels -- so the reconstruction the network starts from is already
inconsistent with its own measurements.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.models.losses.null_space_loss import _to_complex_image
from spectramr.models.losses.registry import register_loss

__all__ = ["NUFFTSampleConsistencyLoss"]

#: Distinguishes "this call site does not thread the sample axis at all"
#: from "it does, and the measurement is missing". ``self.training`` cannot
#: make that distinction: nothing in the training infrastructure switches
#: ``env.losses`` modules to eval, so a loss is in training mode during
#: validation too.
_NOT_THREADED = object()


@register_loss(
    name="nufft_sample_consistency",
    domain="physics",
    compatible_with=["kspace", "image"],
)
class NUFFTSampleConsistencyLoss(nn.Module):
    """L1 between predicted and measured samples, over acquired samples only.

    Takes the grid ``(pred, target)`` pair every ``losses.kspace_losses`` entry
    receives and projects ``pred`` onto the trajectory itself, so the term is
    an ordinary registered loss rather than a second objective the strategy
    computes on the side. The measurement arrives by keyword through
    ``_call_safe_loss``'s signature filtering -- the route ``sense_adjoint_l1``
    already uses for ``smaps``.

    Args:
        norm: ``"l1"`` or ``"l2"``. L1 matches ``pre_dc_kspace``, whose weight
            this term inherits on a non-Cartesian arm.
    """

    def __init__(self, norm: str = "l1") -> None:
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        self.norm = norm

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        *,
        sample_measurement: Any = _NOT_THREADED,
        **kwargs: Any,
    ) -> Tensor | None:
        """Score ``pred`` against the samples the rung actually collected.

        Args:
            pred: k-space ``[B, 2C, H, W]`` interleaved or ``[B, C, H, W]``
                complex -- whatever the arm's ``output_domain: kspace`` emits.
            target: unused. The measurement is the reference here, and the grid
                target is an image of it that this term exists not to trust.
            sample_measurement: a
                :class:`~spectramr.models.diffusion.sample_measurement.SampleMeasurement`
                -- the rung's samples, mask, trajectory and operator. Duck-typed
                rather than imported, so ``models.losses`` takes no dependency on
                ``models.diffusion`` and a test can pass a stand-in. Keyword-only
                so a grid tensor cannot land in it, and defaulted to a sentinel
                so an absent kwarg and a ``None`` one are different states.
            **kwargs: the other physics the computer forwards (``smaps``,
                ``mask``); absorbed, not read.

        Returns:
            Scalar loss, or ``None`` when the caller does not thread the sample
            axis at all -- the validation loop, which scores a sampled
            reconstruction with no rung measurement in scope. That writes NO
            component, so the absence reads as absent rather than as a
            perfectly consistent zero.

        Raises:
            ValueError: when the sample axis IS threaded and the measurement is
                missing. A weighted fidelity term that silently reads zero is
                the DC-blob shape (``sense_adjoint_l1`` carries the same guard
                for ``smaps``).
        """
        if sample_measurement is _NOT_THREADED:
            return None
        if sample_measurement is None:
            raise ValueError(
                "NUFFTSampleConsistencyLoss was threaded sample_measurement=None. "
                "This term is the non-Cartesian arm's only data-fidelity signal, "
                "so contributing zero would train it with no measurement pressure "
                "at all -- the DC-blob shape. Either the forward process is "
                "Cartesian (in which case this loss does not belong on the arm) or "
                "its q_sample has not run this step."
            )

        measurement = sample_measurement.to(pred.device)
        image = ifft2c(_to_complex_image(pred))
        # torchkbnufft's forward carries no 1/sqrt(HW), so its samples sit a
        # factor sqrt(H*W) above the ortho scale every other term in the arm is
        # measured on -- 31.9995 / 63.9995 / 128.0008 / 256.0007 against
        # fft2c's DC at 32/64/128/256. Without this the declared 0.3 would act
        # as roughly 17 beside the other five k-space terms.
        ortho = (image.shape[-2] * image.shape[-1]) ** 0.5
        predicted = measurement.project(image) / ortho
        measured = measurement.samples / ortho
        if predicted.shape != measured.shape:
            raise ValueError(
                f"projected {tuple(predicted.shape)} and measured "
                f"{tuple(measured.shape)} samples must have one shape; the "
                "prediction's coil count must match the measurement's."
            )

        mask = measurement.mask
        while mask.dim() < predicted.dim():
            mask = mask.unsqueeze(-2) if mask.dim() == predicted.dim() - 1 else mask.unsqueeze(0)
        mask = mask.to(dtype=torch.float32)

        residual = (predicted - measured).abs()
        if self.norm == "l2":
            residual = residual.pow(2)
        acquired = mask.expand_as(residual)
        # Normalising by the acquired count rather than the total keeps the
        # term comparable across rungs: at R=32 only a thirtieth of the samples
        # are present, and a total-count mean would read as a thirtieth of the
        # error rather than the same error over fewer samples.
        return (acquired * residual).sum() / acquired.sum().clamp_min(1.0)
