"""Data consistency in the SAMPLE domain, for acquisitions that never touch a grid.

Every DC layer in this package takes an image, ``fft2c``s it, replaces the
measured Cartesian bins, and ``ifft2c``s back. That round trip presumes a uniform
lattice: ``fft2c`` has no meaning at an off-grid sample, and a Cartesian mask has
no bin to index. So a spiral or radial arm routed through those layers is either
refused (the ``GRID_DC_METHODS`` guard) or silently enforcing consistency against
frequencies it never acquired.

This layer replaces the lattice with the trajectory. The operator is the NUFFT,
and the "mask" is a **per-sample** selector over the ``N`` trajectory points --
which readouts survived the acceleration -- not a ``[B, 1, H, W]`` grid.

Two formulations, and the default is the one that is physically honest:

``gradient`` (default)
    ``x <- x - (step / ||A^H W A||) * A^H W m (A x - y)``. A descent step on the
    sample-domain fidelity. It touches the image only where a measurement
    disagrees, so unmeasured k-space is left to the network rather than being
    reshaped by the round trip.

    The normaliser is not cosmetic. ``torchkbnufft``'s forward and adjoint are
    not a unit-gain pair -- measured on a 1024-point spiral onto 32x32,
    ``||A^H W A|| = 2.2e3`` -- and that gain moves with the matrix size, the
    oversampling factor and the trajectory. A literal step of 1.0 diverges by
    twenty orders of magnitude in six iterations; dividing it out makes
    ``step_size: 1.0`` mean "a full Landweber step at the stability limit" on
    every arm instead of meaning nothing portable.

``replace``
    Blend in the sample domain, then adjoint. This mirrors the Cartesian layers'
    semantics and is **approximate**: ``A^H W A`` is the Gram (Toeplitz) operator,
    not the identity, so re-adjointing rewrites the unmeasured content too. Kept
    because it is the only mode that reproduces hard-DC behaviour sample for
    sample, which some baselines are defined against.

Feed it samples when the model already emits samples
----------------------------------------------------

``is_sample_domain=True`` is the non-Cartesian analogue of the Cartesian layers'
``is_kspace_domain``, and it matters far more here. Enforcing consistency on a
prediction that is ALREADY ``[B, C, N]`` needs no transform at all -- it is a
masked blend -- so routing it through the image domain buys a full NUFFT round
trip. Measured on a 128x128 spiral:

=========================  ===============  ========  ======
round trip                 relative error   time      vs FFT
=========================  ===============  ========  ======
``fft2c`` then ``ifft2c``  2.9e-07          8.7 ms    1x
``A^H W`` then ``A``       **0.85**         470 ms    **54x**
=========================  ===============  ========  ======

The Cartesian round trip is free and lossless, which is why doing grid DC in
image space costs only time. The NUFFT round trip is neither: 54x the work AND
it corrupts the samples by 85 %, because ``A A^H W`` is not the identity. A
sample-native model sent through the image domain would have its own
measurements overwritten by the reconstruction operator on the way through.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.nufft_ops import NUFFTForwardModel

logger = logging.getLogger(__name__)

#: How the layer reconciles a prediction with the acquired samples.
NC_DC_MODES: frozenset[str] = frozenset({"gradient", "replace"})


def align_sample_mask(mask: torch.Tensor | None, batch: int, num_samples: int) -> torch.Tensor:
    """Broadcast a per-sample selector to ``[B, 1, N]``.

    The non-Cartesian analogue of ``align_dc_mask``: the same question (whose
    layout did the caller hand me?) with a different answer, because here the
    axis being selected is the readout index, not a coil.

    Accepts ``None`` (every sample acquired), ``[N]``, ``[B, N]`` or ``[B, 1, N]``.
    A complex mask is taken by its real part; a bool mask becomes float.
    """
    if mask is None:
        return torch.ones(batch, 1, num_samples)
    if torch.is_complex(mask):
        mask = mask.real
    if mask.dtype == torch.bool:
        mask = mask.float()
    mask = mask.float()
    if mask.ndim == 1:
        mask = mask.view(1, 1, -1).expand(batch, 1, -1)
    elif mask.ndim == 2:
        mask = mask.unsqueeze(1)
    elif mask.ndim != 3:
        raise ValueError(
            f"A non-Cartesian sample mask is indexed by readout, so it must be "
            f"[N], [B, N] or [B, 1, N]; got {tuple(mask.shape)}. A [B, 1, H, W] "
            f"grid mask means the caller still thinks this acquisition is Cartesian."
        )
    if mask.shape[-1] != num_samples:
        raise ValueError(
            f"sample mask covers {mask.shape[-1]} readouts but the trajectory has "
            f"{num_samples}. These must agree -- a mask trimmed to a different "
            f"acceleration silently enforces consistency at the wrong samples."
        )
    return mask


class NonCartesianDataConsistency(nn.Module):
    """Enforce fidelity at acquired off-grid samples.

    Args:
        im_size: Image matrix the NUFFT reconstructs onto.
        mode: One of :data:`NC_DC_MODES`.
        step_size: Descent step for ``gradient`` mode, in units of the Landweber
            stability limit (see above). Learned when ``learn_step`` is set,
            which is what makes an unrolled cascade anneal its own fidelity
            weight.
        lambda_dc: Blend weight for ``replace`` mode; ``1.0`` is hard DC.
        learn_step: Expose ``step_size`` as a parameter.
        grid_size_factor: NUFFT oversampling.
    """

    def __init__(
        self,
        im_size: tuple[int, int] = (256, 256),
        mode: str = "gradient",
        step_size: float = 1.0,
        lambda_dc: float = 1.0,
        learn_step: bool = True,
        grid_size_factor: float = 2.0,
    ):
        super().__init__()
        if mode not in NC_DC_MODES:
            raise ValueError(
                f"Unknown non-Cartesian DC mode {mode!r}. Valid: {sorted(NC_DC_MODES)}."
            )
        if not 0.0 <= lambda_dc <= 1.0:
            raise ValueError(f"lambda_dc must be in [0, 1], got {lambda_dc}.")
        self.im_size = tuple(im_size)
        self.mode = mode
        self.lambda_dc = float(lambda_dc)
        # Imported eagerly by NUFFTForwardModel: a non-Cartesian DC without a
        # NUFFT has no degraded mode worth running, so the ImportError is the
        # correct outcome rather than something to guard into a fallback.
        self.operator = NUFFTForwardModel(im_size=self.im_size, grid_size_factor=grid_size_factor)
        raw = torch.tensor(float(step_size))
        self.step_size = nn.Parameter(raw) if learn_step else None
        if self.step_size is None:
            self.register_buffer("_step_const", raw)
        #: Gram-norm cache keyed by (num_samples, device). Estimated ONCE: the
        #: power iteration costs two NUFFTs and a host sync, which is exactly
        #: what must not run per training step (non-negotiable 9). A trajectory
        #: that rotates per frame but keeps its length reuses the estimate --
        #: the Gram norm of a rotated trajectory is the same to within the
        #: tolerance a step size cares about.
        self._gram_norm: dict[tuple[int, str], float] = {}

    def _step(self) -> torch.Tensor:
        return self.step_size if self.step_size is not None else self._step_const

    def _gram_scale(
        self, trajectory: torch.Tensor, weights: torch.Tensor, like: torch.Tensor
    ) -> float:
        """Spectral norm of ``A^H W A``, by power iteration, cached."""
        key = (trajectory.shape[-1], str(like.device))
        cached = self._gram_norm.get(key)
        if cached is not None:
            return cached
        with torch.no_grad():
            v = torch.randn(1, 1, *self.im_size, dtype=like.dtype, device=like.device)
            v = v / v.abs().norm()
            for _ in range(8):
                av = self.operator.forward_project(v, trajectory)
                v = self.operator.adjoint_project(av * weights[:1, :1], trajectory)
                norm = v.abs().norm()
                if norm <= 0:
                    return 1.0
                v = v / norm
            estimate = float(norm)
        self._gram_norm[key] = max(estimate, 1e-12)
        return self._gram_norm[key]

    def forward(
        self,
        image: torch.Tensor,
        measured_samples: torch.Tensor | None,
        trajectory: torch.Tensor | None,
        sample_mask: torch.Tensor | None = None,
        dcf: torch.Tensor | None = None,
        is_sample_domain: bool = False,
    ) -> torch.Tensor:
        """Reconcile a prediction with the acquired samples.

        Args:
            image: Complex image ``[B, C, H, W]``, or ``[B, C, N]`` samples when
                ``is_sample_domain`` is set.
            measured_samples: Acquired k-space ``[B, C, N]`` (complex).
            trajectory: Coordinates ``[2, N]`` or ``[B, 2, N]``, radians in
                ``[-pi, pi]``.
            sample_mask: Per-readout selector -- see :func:`align_sample_mask`.
            dcf: Density weights ``[N]`` or ``[B, 1, N]``. Absent means uniform,
                which is a real choice: Pipe's weights are what stop the densely
                sampled centre from dominating the adjoint.
            is_sample_domain: The prediction is already ``[B, C, N]``. Skips both
                NUFFTs -- see the table in the module docstring for what that
                saves, and more importantly what it stops corrupting.

        Returns:
            A tensor in the same domain and layout as the input.
        """
        if measured_samples is None or trajectory is None:
            raise ValueError(
                "NonCartesianDataConsistency was reached without "
                f"{'measured_samples' if measured_samples is None else 'trajectory'}. "
                "Returning the input unchanged would report a data-consistent "
                "reconstruction that never saw its measurement (pitfall 9); the "
                "batch must carry `measured_kspace`, `trajectory` and `dcf` for "
                "every non-Cartesian arm."
            )
        if not torch.is_complex(image):
            raise ValueError(
                f"The NUFFT needs a complex image; got dtype={image.dtype}. "
                "Fold an interleaved real/imag field with "
                "`attention_domains.interleaved_to_complex` before this layer."
            )

        batch, _, num_samples = measured_samples.shape
        mask = align_sample_mask(sample_mask, batch, num_samples).to(image.device)
        weights = self._density(dcf, batch, num_samples, image.device, mask.dtype)

        if is_sample_domain:
            if image.dim() != 3 or image.shape[-1] != num_samples:
                raise ValueError(
                    f"is_sample_domain=True expects a [B, C, N] prediction matching "
                    f"the {num_samples}-sample trajectory; got {tuple(image.shape)}."
                )
            # No transform at all: consistency between two sample sets IS a
            # blend, and the gradient step reduces to the same family, so the
            # two modes coincide here.
            if self.mode == "gradient":
                pull = self._step().to(image.dtype) * mask.to(image.dtype)
                return image - pull * (image - measured_samples)
            return torch.where(
                mask.bool(),
                (1.0 - self.lambda_dc) * image + self.lambda_dc * measured_samples,
                image,
            )

        predicted = self.operator.forward_project(image, trajectory)
        if self.mode == "gradient":
            residual = (predicted - measured_samples) * mask * weights
            scale = self._gram_scale(trajectory, weights, image)
            step = (self._step() / scale).to(image.dtype)
            return image - step * self.operator.adjoint_project(residual, trajectory)

        blended = torch.where(
            mask.bool(),
            (1.0 - self.lambda_dc) * predicted + self.lambda_dc * measured_samples,
            predicted,
        )
        return self.operator.adjoint_project(blended * weights, trajectory)

    @staticmethod
    def _density(
        dcf: torch.Tensor | None,
        batch: int,
        num_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalise density weights to ``[B, 1, N]``."""
        if dcf is None:
            return torch.ones(batch, 1, num_samples, device=device, dtype=dtype)
        if torch.is_complex(dcf):
            dcf = dcf.real
        dcf = dcf.to(device=device, dtype=dtype)
        if dcf.ndim == 1:
            dcf = dcf.view(1, 1, -1).expand(batch, 1, -1)
        elif dcf.ndim == 2:
            dcf = dcf.unsqueeze(1)
        if dcf.shape[-1] != num_samples:
            raise ValueError(f"dcf covers {dcf.shape[-1]} samples, trajectory has {num_samples}.")
        return dcf


__all__ = ["NC_DC_MODES", "NonCartesianDataConsistency", "align_sample_mask"]
