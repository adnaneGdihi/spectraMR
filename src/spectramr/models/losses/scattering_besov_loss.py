"""Scattering + Besov multi-scale detail prior (MICCAI MRIxFields2026, B-1.3).

A magnitude-image detail prior for Task-1 (->7T synthesis) built from a FIXED (non-learnable)
oriented Gabor wavelet filterbank — two complementary, parameter-free components that L1 (which
under-produces high-frequency detail -> blur) and the existing Fourier ``spectral_*`` losses do
not capture:

* **Scattering** (Mallat): pooled wavelet-MODULUS coefficients
  ``S = pool( sqrt((x⊛ψ_cos)^2 + (x⊛ψ_sin)^2) )`` over scales x orientations. The modulus
  nonlinearity + spatial localisation make it deformation-stable and texture-morphology aware —
  distinct from a LINEAR global Fourier-magnitude loss.
* **Besov** (B^s_{1,1} norm of the residual): ``sum_j 2^(level_j*s) * mean| D_j(pred) - D_j(target) |``
  over the linear band-pass detail ``D_j`` at scale ``j``. The filterbank sets ``freq = 0.25/2^j``,
  so the loop index ``j`` increases as the band gets COARSER (``j=0`` is the finest band). The
  Besov octave ``level_j = (n_scales-1-j)`` therefore increases toward FINER scales, so the
  ``2^(level_j*s)`` weighting up-weights FINER scales, directly driving high-frequency detail
  synthesis (the SR regularity prior).

Both filterbanks are fixed buffers (no learnable parameters, no encoder confound) and fully
differentiable, so the loss back-props detail-matching gradients into the generator.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.losses.registry import register_loss


def _gabor_quadrature(
    size: int, sigma: float, freq: float, theta: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Real Gabor quadrature pair (cos, sin), zero-mean band-pass, L1-normalised."""
    half = size // 2
    ys, xs = torch.meshgrid(
        torch.arange(-half, half + 1, dtype=torch.float32),
        torch.arange(-half, half + 1, dtype=torch.float32),
        indexing="ij",
    )
    rot = xs * math.cos(theta) + ys * math.sin(theta)
    envelope = torch.exp(-(xs**2 + ys**2) / (2.0 * sigma**2))
    cos_k = envelope * torch.cos(2.0 * math.pi * freq * rot)
    sin_k = envelope * torch.sin(2.0 * math.pi * freq * rot)
    cos_k = cos_k - cos_k.mean()  # zero-mean -> band-pass (kills DC)
    sin_k = sin_k - sin_k.mean()
    cos_k = cos_k / (cos_k.abs().sum() + 1e-8)
    sin_k = sin_k / (sin_k.abs().sum() + 1e-8)
    return cos_k, sin_k


@register_loss(name="scattering_besov", domain="image")
class ScatteringBesovLoss(nn.Module):
    """Fixed-filterbank scattering + Besov detail prior (B-1.3)."""

    def __init__(
        self,
        n_scales: int = 3,
        n_orientations: int = 4,
        besov_s: float = 1.0,
        scatter_weight: float = 1.0,
        besov_weight: float = 1.0,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        if n_scales < 1 or n_orientations < 1:
            raise ValueError(
                f"n_scales/n_orientations must be >= 1; got {n_scales}, {n_orientations}."
            )
        self.n_scales = int(n_scales)
        self.n_orientations = int(n_orientations)
        self.besov_s = float(besov_s)
        self.scatter_weight = float(scatter_weight)
        self.besov_weight = float(besov_weight)
        cos_bank, sin_bank, scale_index = [], [], []
        for j in range(self.n_scales):
            sigma = 1.5 * (2.0**j)
            freq = 0.25 / (2.0**j)  # coarser scale -> lower frequency
            for k in range(self.n_orientations):
                theta = math.pi * k / self.n_orientations
                cos_k, sin_k = _gabor_quadrature(kernel_size, sigma, freq, theta)
                cos_bank.append(cos_k)
                sin_bank.append(sin_k)
                scale_index.append(j)
        # [F, 1, ks, ks] fixed (non-learnable) conv weights.
        self.register_buffer("cos_filters", torch.stack(cos_bank).unsqueeze(1))
        self.register_buffer("sin_filters", torch.stack(sin_bank).unsqueeze(1))
        self.register_buffer("scale_index", torch.tensor(scale_index, dtype=torch.long))
        # Besov B^s seminorm weight 2^(level*s) per band, with `level` the FREQUENCY octave
        # (finer = larger). The loop index j indexes COARSENING (freq = 0.25/2**j: larger j =
        # coarser), so level_j = (n_scales-1-j) makes the weight up-weight the FINER / high-
        # frequency bands as the SR detail prior intends. (Pre-fix bug used 2**(j*s), which
        # up-weighted the COARSEST band — the opposite of the documented behaviour.)
        besov_w = [2.0 ** ((self.n_scales - 1 - j) * self.besov_s) for j in range(self.n_scales)]
        self.register_buffer("besov_scale_weights", torch.tensor(besov_w, dtype=torch.float32))
        self._pad = kernel_size // 2

    def _responses(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos band-pass responses, wavelet-modulus maps), each [B, F, H, W]."""
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)  # magnitude: collapse to single channel
        cos_r = F.conv2d(x, self.cos_filters, padding=self._pad)
        sin_r = F.conv2d(x, self.sin_filters, padding=self._pad)
        modulus = torch.sqrt(cos_r**2 + sin_r**2 + 1e-8)
        return cos_r, modulus

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        out = self.compute(pred, target)
        return out["loss_total"]

    def compute(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        cos_p, mod_p = self._responses(pred)
        cos_t, mod_t = self._responses(target)
        # Scattering: pooled (translation-invariant) wavelet-modulus coefficients.
        s_p = F.adaptive_avg_pool2d(mod_p, 8)
        s_t = F.adaptive_avg_pool2d(mod_t, 8)
        loss_scatter = F.l1_loss(s_p, s_t)
        # Besov B^s_{1,1} norm of the residual: 2^(level*s)-weighted per-scale linear detail,
        # with the weight up-weighting the finer (higher-frequency) bands (see __init__).
        loss_besov = pred.new_zeros(())
        for j in range(self.n_scales):
            mask = self.scale_index == j
            w = self.besov_scale_weights[j]
            loss_besov = loss_besov + w * (cos_p[:, mask] - cos_t[:, mask]).abs().mean()
        loss_besov = loss_besov / self.n_scales
        total = self.scatter_weight * loss_scatter + self.besov_weight * loss_besov
        return {"loss_total": total, "loss_scatter": loss_scatter, "loss_besov": loss_besov}

    def detail_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Mean wavelet-modulus energy of an image — a scalar 'detail richness' witness."""
        _, mod = self._responses(x)
        return mod.mean().detach()
