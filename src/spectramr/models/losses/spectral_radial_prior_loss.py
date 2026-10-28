r"""Spectral radial prior loss for contrast-aware k-space recovery.

Empirically, the magnitude spectrum :math:`|F(\\mathbf{x}_c)|` of an MRI image
exhibits a *contrast-dependent radial decay*: T2 carries broader low-frequency
support, FLAIR shows sharper high-frequency edge content from CSF suppression,
and SWI/QSM acquisitions concentrate energy in mid-frequency phase bands.
A model that ignores this prior wastes capacity recovering plausible but
contrast-incongruent spectra — particularly at high acceleration where the
image-domain cues are washed out and the spectrum becomes the more reliable
signal.

The loss compares the *radially binned* energy of the prediction against an
empirical or analytic target spectrum :math:`\\bar{E}_c(k_r)`:

.. math::

   \\mathcal{L}_{\\text{spec}}
   \\;=\\;
   \\sum_{k_r}
   \\left\\|
     \\mathbb{E}_{\\mathbf{r} \\in S(k_r)} | F(\\hat{\\mathbf{x}}_c) |^2
     - \\bar{E}_c(k_r)
   \\right\\|^2,

where :math:`S(k_r)` is the annulus of radius :math:`k_r` in the k-space
plane. The expectation is computed by binning the squared FFT magnitude into
``n_bins`` annular bins.

The target spectrum can be:

- **Estimated on-the-fly from the ground truth** (default behaviour when
  ``target`` is supplied) — useful when no prior over the cohort is known.
- **Pre-fit per contrast and frozen** — pass ``reference_spectrum`` as a
  ``(n_contrasts, n_bins)`` tensor or ``(n_bins,)`` for a single contrast.
  This is the form used as a true prior at high acceleration.

The implementation uses the centred FFT from
:mod:`spectramr.infrastructure.physics.fft_ops` (per CLAUDE.md: never raw
``torch.fft.fft2``) so it composes correctly with the rest of the physics
stack.

References
----------
[9]  H. K. Aggarwal, M. P. Mani, M. Jacob, "MoDL: Model-based deep learning
     architecture for inverse problems," *IEEE TMI*, vol. 38, no. 2, 2019.
[11] H. Chung, J. C. Ye, "Score-based diffusion models for accelerated MRI,"
     *Med. Image Anal.*, vol. 80, p. 102479, 2022.

Section 3 of ``docs/multi_contrast.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.models.losses.registry import register_loss

__all__ = ["COMPARISONS", "SpectralRadialPriorLoss", "compute_radial_spectrum"]

COMPARISONS: tuple[str, ...] = ("log1p_difference", "log_ratio")
#: Bins whose centre is at or beyond this fraction of Nyquist hold the noise floor (as in PR1).
NOISE_FROM = 0.85
#: Keeps ``log`` finite when a target has no outer-band energy at all.
_EPS_FLOOR = 1e-12


def compute_radial_spectrum(
    image: torch.Tensor,
    n_bins: int = 32,
) -> torch.Tensor:
    """Return the average radial energy spectrum.

    Args:
        image: ``(B, C, H, W)`` real-valued image tensor.
        n_bins: Number of radial bins between :math:`k_r=0` and the Nyquist
            radius. The bin edges are linearly spaced.

    Returns:
        ``(B, n_bins)`` tensor of mean :math:`|F|^2` per radial bin,
        averaged over the batch's channel dimension.
    """
    if image.dim() != 4:
        raise ValueError(f"image must be (B, C, H, W); got {tuple(image.shape)}")
    B, C, H, W = image.shape
    # Centred FFT (norm=ortho); fft2c handles the real/imag-as-channels case
    # internally via _to_complex when given an even C, but here we pass the
    # magnitude image so we treat C as a generic batch axis.
    img = image.reshape(B * C, 1, H, W)
    spec = fft2c(img)  # complex (B*C, 1, H, W)
    energy = spec.abs().pow(2).reshape(B, C, H, W).mean(dim=1)  # (B, H, W)

    # Build radial coordinate (centred at H//2, W//2) once per call; cheap.
    yy = torch.arange(H, device=image.device, dtype=image.dtype) - H / 2
    xx = torch.arange(W, device=image.device, dtype=image.dtype) - W / 2
    rr = torch.sqrt(yy.view(-1, 1) ** 2 + xx.view(1, -1) ** 2)
    r_max = float(min(H, W)) / 2.0
    bin_idx = (rr / r_max * n_bins).clamp(max=n_bins - 1).long()  # (H, W)

    # scatter_add the energy into bins.
    out = energy.new_zeros((B, n_bins))
    counts = energy.new_zeros((B, n_bins))
    flat_bins = bin_idx.reshape(-1)
    flat_energy = energy.reshape(B, -1)
    out.scatter_add_(1, flat_bins.unsqueeze(0).expand(B, -1), flat_energy)
    ones = torch.ones_like(flat_energy)
    counts.scatter_add_(1, flat_bins.unsqueeze(0).expand(B, -1), ones)
    return out / counts.clamp_min(1.0)


@register_loss(name="spectral_radial_prior", domain="image")
class SpectralRadialPriorLoss(nn.Module):
    """Soft contrast-aware radial-spectrum prior.

    Args:
        n_bins: Number of radial annular bins.
        reference_spectrum: Optional ``(n_bins,)`` or ``(n_contrasts, n_bins)``
            tensor used as the *fixed* target spectrum. When ``None``, the
            target spectrum is computed on-the-fly from the ground truth
            ``target`` passed to ``forward``. The "frozen prior" form is
            preferred at high acceleration (R≥6).
        log_space: If True, compare in :math:`\\log(1+E)` space — more robust
            when high-frequency bins are orders of magnitude smaller than the
            DC bin.
    """

    def __init__(
        self,
        n_bins: int = 32,
        reference_spectrum: torch.Tensor | None = None,
        log_space: bool | None = None,
        comparison: str = "log1p_difference",
        eps_noise_mult: float = 1.0,
        min_ring_snr: float = 4.0,
    ) -> None:
        super().__init__()
        if n_bins < 4:
            raise ValueError(f"n_bins must be >= 4, got {n_bins}")
        if comparison not in COMPARISONS:
            raise ValueError(f"Unknown comparison {comparison!r}. Valid: {list(COMPARISONS)}.")
        if comparison == "log_ratio" and log_space:
            raise ValueError(
                "comparison='log_ratio' already takes the log of each ring; "
                "log_space: true would apply log1p on top. Drop log_space."
            )
        if not (eps_noise_mult > 0 and math.isfinite(eps_noise_mult)):
            raise ValueError(f"eps_noise_mult must be a finite number > 0, got {eps_noise_mult}")
        if not min_ring_snr >= 0:
            raise ValueError(f"min_ring_snr must be >= 0, got {min_ring_snr}")
        self.n_bins = n_bins
        self.comparison = comparison
        # None resolves to the comparison's own default; log_ratio is log-space already.
        self.log_space = (comparison == "log1p_difference") if log_space is None else log_space
        self.eps_noise_mult = float(eps_noise_mult)
        self.min_ring_snr = float(min_ring_snr)
        # First bin whose centre lies at or beyond NOISE_FROM of Nyquist: the bins are contiguous.
        self._noise_start = next(i for i in range(n_bins) if (i + 0.5) / n_bins >= NOISE_FROM)
        self._first_call = True
        self._empty_batches: torch.Tensor | None = None
        if reference_spectrum is not None:
            ref = reference_spectrum.detach().clone().float()
            if ref.dim() == 1:
                ref = ref.unsqueeze(0)  # (1, n_bins)
            if ref.shape[-1] != n_bins:
                raise ValueError(
                    f"reference_spectrum last dim must be {n_bins}, got {tuple(ref.shape)}"
                )
            self.register_buffer("_reference", ref)
        else:
            self._reference = None

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        contrast_id: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        pred_spec = compute_radial_spectrum(prediction, self.n_bins)
        if self._reference is not None:
            if contrast_id is None or self._reference.shape[0] == 1:
                ref_spec = self._reference.expand(pred_spec.shape[0], -1)
            else:
                ref_spec = self._reference[contrast_id.long()]
        else:
            if target is None:
                raise ValueError(
                    "SpectralRadialPriorLoss needs either reference_spectrum "
                    "or a `target` tensor at forward time."
                )
            ref_spec = compute_radial_spectrum(target, self.n_bins)
        if self.comparison == "log_ratio":
            return self._log_ratio(pred_spec, ref_spec)
        if self.log_space:
            pred_spec = torch.log1p(pred_spec)
            ref_spec = torch.log1p(ref_spec)
        return (pred_spec - ref_spec).pow(2).mean()

    @property
    def empty_ring_batches(self) -> int:
        """Batches on which no ring passed ``min_ring_snr`` (zero loss returned for them)."""
        return 0 if self._empty_batches is None else int(self._empty_batches.item())

    def _log_ratio(self, pred_spec: torch.Tensor, ref_spec: torch.Tensor) -> torch.Tensor:
        """Mean squared log-ratio over the rings whose target SNR clears ``min_ring_snr``.

        ``log(E + eps)`` per ring, with ``eps`` a multiple of the target's own outer-band
        level, so the term is invariant to the image's absolute scale. The noise level and
        the SNR mask come from the target, so they carry no gradient.
        """
        ref = ref_spec.detach()
        noise = ref[:, self._noise_start :].median(dim=1).values
        keep = (ref > 0) & (ref >= self.min_ring_snr * noise[:, None])
        eps = (self.eps_noise_mult * noise).clamp_min(_EPS_FLOOR)[:, None]
        diff = torch.log(pred_spec + eps) - torch.log(ref + eps)
        kept = keep.to(diff.dtype)
        if self._first_call:
            # One host read, first call only: a threshold that empties every ring on the
            # first batch is a config error, not a batch property.
            self._first_call = False
            if not bool(keep.any()):
                raise ValueError(
                    f"min_ring_snr={self.min_ring_snr} keeps no ring of the target spectrum "
                    "on the first batch; lower min_ring_snr or check the target's scale."
                )
        empty = (kept.sum() == 0).long()
        self._empty_batches = empty if self._empty_batches is None else self._empty_batches + empty
        return (diff.pow(2) * kept).sum() / kept.sum().clamp_min(1.0)
