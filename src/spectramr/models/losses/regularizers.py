"""MRI-Specific Regularizers (registered as losses).

Each regularizer is registered via @register_loss with an explicit ``domain``
parameter so the loss-builder can validate placement at config-load time
(e.g., a kspace regularizer in a losses.image_losses block fails fast).

Domain split (matches existing loss layout):

- ``image``    — operates on magnitude / complex-image tensors [B, C, H, W]
- ``kspace``   — operates on raw or complex k-space tensors [B, C, H, W]
- ``complex``  — operates on complex-valued tensors regardless of domain

Conventions:

- ``forward()`` returns a single scalar tensor (no Python floats, no .item()).
- No ``.item()``, ``.cpu()``, ``.numpy()`` inside forward — keeps autograd
  intact and avoids GPU sync per CLAUDE.md training-loop performance rules.
- All public knobs (weights, eps, etc.) are constructor args so the
  YAML-driven LossBuilder can pass them through ``**kwargs``.

References:

- Wavelet sparsity: Lustig, Donoho, Pauly (MRM 2007), "Sparse MRI: The
  Application of Compressed Sensing for Rapid MR Imaging".
- Hessian Schatten-norm: Lefkimmiatis et al. (TIP 2013), "Hessian-Based
  Norm Regularization for Image Restoration with Biomedical Applications".
- Hankel low-rank (LORAKS): Haldar (TMI 2014), "Low-Rank Modeling of Local
  k-Space Neighborhoods (LORAKS)".
- Hermitian symmetry: standard partial-Fourier MRI assumption that real-valued
  images yield k[k] = conj(k[-k]).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss

# ---------------------------------------------------------------------------
# Image-domain regularizers
# ---------------------------------------------------------------------------


@register_loss(name="wavelet_sparsity_l1", aliases=["wavelet_l1"], domain="image")
class WaveletSparsityLoss(nn.Module):
    """L1 sparsity in a Daubechies-style wavelet domain.

    Uses a single-level discrete wavelet transform implemented as a fixed
    set of separable analysis filters (PyTorch convolution). Avoids a
    third-party PyWavelets dependency and remains autograd-friendly.

    Args:
        wavelet: ``"haar"`` (default) or ``"db2"`` (4-tap Daubechies-2).
        levels: Number of decomposition levels (>=1). Default 1 — additional
            levels apply the analysis to the LL band recursively.
        weight: Scale factor multiplied into the returned scalar.
        include_ll: If True, also penalises the LL (low-low) coefficients.
            Default False — the LL band carries most of the signal energy
            and penalising it shrinks the image.
    """

    _lo: torch.Tensor
    _hi: torch.Tensor

    def __init__(
        self,
        wavelet: str = "haar",
        levels: int = 1,
        weight: float = 1.0,
        include_ll: bool = False,
    ) -> None:
        super().__init__()
        if wavelet not in ("haar", "db2"):
            raise ValueError(f"wavelet must be 'haar' or 'db2', got {wavelet!r}")
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.wavelet = wavelet
        self.levels = levels
        self.weight = float(weight)
        self.include_ll = include_ll

        # Analysis filter taps (low-pass and high-pass).
        if wavelet == "haar":
            lo = torch.tensor([1.0, 1.0]) / math.sqrt(2.0)
            hi = torch.tensor([-1.0, 1.0]) / math.sqrt(2.0)
        else:  # db2
            lo = torch.tensor(
                [
                    -0.12940952255092145,
                    0.22414386804185735,
                    0.836516303737469,
                    0.48296291314469025,
                ]
            )
            hi = torch.tensor(
                [
                    -0.48296291314469025,
                    0.836516303737469,
                    -0.22414386804185735,
                    -0.12940952255092145,
                ]
            )
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)

    @staticmethod
    def _conv2d_separable(
        x: torch.Tensor, h_row: torch.Tensor, h_col: torch.Tensor
    ) -> torch.Tensor:
        """Apply separable 1D filters along H and W with stride 2 (decimation)."""
        c = x.shape[1]
        # Build 2D kernel from outer product.
        k2d = torch.outer(h_row, h_col).to(dtype=x.dtype, device=x.device)
        k2d = k2d.view(1, 1, *k2d.shape).expand(c, 1, *k2d.shape).contiguous()
        # depthwise, stride 2
        return F.conv2d(x, k2d, stride=2, padding=0, groups=c)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        # target arg accepted for builder uniformity but unused — this is a prior on prediction.
        del target
        if torch.is_complex(prediction):
            prediction = prediction.abs()
        x = prediction
        loss = x.new_zeros(())
        for _ in range(self.levels):
            ll = self._conv2d_separable(x, self._lo, self._lo)
            lh = self._conv2d_separable(x, self._lo, self._hi)
            hl = self._conv2d_separable(x, self._hi, self._lo)
            hh = self._conv2d_separable(x, self._hi, self._hi)
            loss = loss + lh.abs().mean() + hl.abs().mean() + hh.abs().mean()
            if self.include_ll:
                loss = loss + ll.abs().mean()
            x = ll
        return self.weight * loss


@register_loss(name="hessian_schatten", aliases=["hessian"], domain="image")
class HessianSchattenLoss(nn.Module):
    """Hessian Schatten-1 (nuclear) norm regularizer.

    Penalises the sum of singular values of the per-pixel 2x2 Hessian
    of the magnitude image. Edge-preserving: rank-1 Hessians (purely
    horizontal or vertical edges) cost the same as their largest singular
    value, while textured / second-order regions accrue both eigenvalues.

    Args:
        weight: scale factor.
        eps: numerical stabiliser for the closed-form 2x2 SVD.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-12) -> None:
        super().__init__()
        self.weight = float(weight)
        self.eps = float(eps)

    @staticmethod
    def _laplacian_components(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (Hxx, Hxy, Hyy) finite differences with replicate padding."""
        # Hxx = u(x+1) - 2u(x) + u(x-1)
        x_pad = F.pad(x, (1, 1, 0, 0), mode="replicate")
        hxx = x_pad[..., :, 2:] - 2.0 * x_pad[..., :, 1:-1] + x_pad[..., :, :-2]
        y_pad = F.pad(x, (0, 0, 1, 1), mode="replicate")
        hyy = y_pad[..., 2:, :] - 2.0 * y_pad[..., 1:-1, :] + y_pad[..., :-2, :]
        # Hxy via central difference of dx along y.
        xy_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        # mixed second derivative: (u(x+1, y+1) - u(x-1, y+1) - u(x+1, y-1) + u(x-1, y-1)) / 4
        hxy = (
            xy_pad[..., 2:, 2:]
            - xy_pad[..., 2:, :-2]
            - xy_pad[..., :-2, 2:]
            + xy_pad[..., :-2, :-2]
        ) / 4.0
        return hxx, hxy, hyy

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        if torch.is_complex(prediction):
            prediction = prediction.abs()
        hxx, hxy, hyy = self._laplacian_components(prediction)
        # Closed-form singular values of [[a, b], [b, c]]:
        #   s1, s2 = | (a+c)/2 +/- sqrt(((a-c)/2)^2 + b^2) |
        a, b, c = hxx, hxy, hyy
        half_sum = 0.5 * (a + c)
        half_diff = 0.5 * (a - c)
        radical = torch.sqrt(half_diff.pow(2) + b.pow(2) + self.eps)
        s1 = (half_sum + radical).abs()
        s2 = (half_sum - radical).abs()
        nuclear = (s1 + s2).mean()
        return self.weight * nuclear


@register_loss(name="patch_low_rank", aliases=["nlr"], domain="image")
class PatchLowRankLoss(nn.Module):
    """Patch-based low-rank prior on stacked image patches.

    Forms a non-overlapping patch matrix (each patch flattened to a row) and
    penalises its nuclear norm — encouraging redundancy across patches that
    is characteristic of clean MR images.

    Args:
        patch_size: side length of the (square) patch.
        weight: scale factor.
        max_patches: cap on patch matrix rows for memory; oldest patches
            beyond the cap are dropped (in spatial order).
    """

    def __init__(
        self,
        patch_size: int = 8,
        weight: float = 1.0,
        max_patches: int = 4096,
    ) -> None:
        super().__init__()
        if patch_size < 2:
            raise ValueError(f"patch_size must be >= 2, got {patch_size}")
        self.patch_size = patch_size
        self.weight = float(weight)
        self.max_patches = int(max_patches)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        if torch.is_complex(prediction):
            prediction = prediction.abs()
        h, w = prediction.shape[-2:]
        ps = self.patch_size
        # Round H/W down to a multiple of patch_size.
        h_eff = (h // ps) * ps
        w_eff = (w // ps) * ps
        x = prediction[..., :h_eff, :w_eff]
        # [B, C, H/ps, ps, W/ps, ps] -> [B, C, num_patches, ps*ps]
        patches = x.unfold(2, ps, ps).unfold(3, ps, ps)
        b2, c2, n_h, n_w, _, _ = patches.shape
        patches = patches.contiguous().view(b2, c2, n_h * n_w, ps * ps)
        # Optional row cap.
        if patches.shape[2] > self.max_patches:
            patches = patches[:, :, : self.max_patches, :]
        # Nuclear norm per (batch, channel) — torch.linalg.svdvals retains autograd.
        # The SVD kernels have no fp16/bf16 implementation, so AMP outputs are upcast.
        if patches.dtype in (torch.float16, torch.bfloat16):
            patches = patches.float()
        sv: torch.Tensor = torch.linalg.svdvals(patches)
        return self.weight * sv.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# K-space-domain regularizers
# ---------------------------------------------------------------------------


@register_loss(name="hermitian_symmetry", aliases=["herm_symmetry"], domain="kspace")
class HermitianSymmetryLoss(nn.Module):
    """Penalty on departure from Hermitian symmetry: k[r] vs conj(k[-r]).

    For real-valued images the k-space satisfies ``k[k_x, k_y] = conj(k[-k_x, -k_y])``.
    This loss measures the mean-squared deviation from that property and
    is a valid prior whenever the underlying image is (close to) real.

    Expects centred k-space (DC at index ``N // 2``, as :func:`fft2c` emits);
    even and odd H/W are both paired exactly.

    Args:
        weight: scale factor.
        normalize: if True, divide the squared deviation by ``mean(|k|^2 + eps)``
            so the loss is dimensionless.
        eps: numerical stabiliser for the normaliser.
    """

    def __init__(
        self,
        weight: float = 1.0,
        normalize: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.normalize = normalize
        self.eps = float(eps)

    @staticmethod
    def _to_complex(t: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(t):
            return t
        # If even-channel real, treat as interleaved Re/Im (common 2*C layout).
        if t.dim() >= 4 and t.shape[1] % 2 == 0:
            real = t[:, 0::2]
            imag = t[:, 1::2]
            return torch.complex(real, imag)
        return t.to(torch.complex64)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        k = self._to_complex(prediction)
        # With DC at index N // 2 (fft2c), the Hermitian partner of index i is
        # N - i on an even axis (flip gives N-1-i, so roll by one) and N-1-i on
        # an odd axis (flip alone); on an even axis index 0 is its own partner.
        h, w = k.shape[-2], k.shape[-1]
        k_flip = torch.conj(
            torch.roll(torch.flip(k, dims=(-2, -1)), shifts=(1 - h % 2, 1 - w % 2), dims=(-2, -1))
        )
        diff = k - k_flip
        num = (diff.real.pow(2) + diff.imag.pow(2)).mean()
        if self.normalize:
            denom = (k.real.pow(2) + k.imag.pow(2)).mean() + self.eps
            num = num / denom
        return self.weight * num


@register_loss(name="kspace_l1_sparsity", aliases=["kspace_l1"], domain="kspace")
class KspaceL1SparsityLoss(nn.Module):
    """L1 sparsity directly on k-space magnitude.

    Args:
        weight: scale factor.
        center_mask_radius: optional radius (fraction of H/2) of a centre
            ACS region excluded from the penalty so low-frequency structure
            is not over-shrunk. ``0.0`` (default) disables masking.
    """

    def __init__(self, weight: float = 1.0, center_mask_radius: float = 0.0) -> None:
        super().__init__()
        self.weight = float(weight)
        if not (0.0 <= center_mask_radius < 1.0):
            raise ValueError(f"center_mask_radius must be in [0, 1), got {center_mask_radius}")
        self.center_mask_radius = float(center_mask_radius)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        # The floor only keeps sqrt'(0) finite; sqrt(floor) is what a zero sample
        # scores, so it sits far below any |k|^2 a measurement carries.
        if torch.is_complex(prediction):
            mag = (prediction.real.pow(2) + prediction.imag.pow(2)).clamp(min=1e-24).sqrt()
        else:
            # Treat 2*C real layout as complex magnitude.
            if prediction.dim() >= 4 and prediction.shape[1] % 2 == 0:
                real = prediction[:, 0::2]
                imag = prediction[:, 1::2]
                mag = (real.pow(2) + imag.pow(2)).clamp(min=1e-24).sqrt()
            else:
                mag = prediction.abs()
        if self.center_mask_radius > 0:
            h, w = mag.shape[-2], mag.shape[-1]
            cy, cx = h // 2, w // 2
            ry = self.center_mask_radius * (h / 2.0)
            rx = self.center_mask_radius * (w / 2.0)
            yy = torch.arange(h, device=mag.device).view(-1, 1) - cy
            xx = torch.arange(w, device=mag.device).view(1, -1) - cx
            outside = ((yy / max(ry, 1e-6)).pow(2) + (xx / max(rx, 1e-6)).pow(2)) > 1.0
            mag = mag * outside.to(dtype=mag.dtype)
        return self.weight * mag.mean()


@register_loss(name="hankel_low_rank", aliases=["loraks"], domain="kspace")
class HankelLowRankLoss(nn.Module):
    """Approximate LORAKS-style low-rank prior on Hankel-structured k-space neighbourhoods.

    Constructs a Hankel matrix from sliding ``(kw, kh)`` k-space patches and
    penalises its nuclear norm. Captures the limited-support property of
    real-image k-space (which produces low-rank Hankel structure).

    Args:
        kernel_size: side length of the (square) k-space neighbourhood.
        weight: scale factor.
        max_rows: cap on Hankel matrix rows (memory).
    """

    def __init__(
        self,
        kernel_size: int = 5,
        weight: float = 1.0,
        max_rows: int = 4096,
    ) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2, got {kernel_size}")
        self.kernel_size = int(kernel_size)
        self.weight = float(weight)
        self.max_rows = int(max_rows)

    @staticmethod
    def _to_complex(t: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(t):
            return t
        if t.dim() >= 4 and t.shape[1] % 2 == 0:
            real = t[:, 0::2]
            imag = t[:, 1::2]
            return torch.complex(real, imag)
        return t.to(torch.complex64)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        k = self._to_complex(prediction)
        h, w = k.shape[-2:]
        ks = self.kernel_size
        if h < ks or w < ks:
            return prediction.new_zeros(()).to(
                prediction.dtype if not torch.is_complex(prediction) else torch.float32
            )
        # Build Hankel rows by extracting all sliding ks*ks patches.
        # Real and imag parts handled jointly via complex SVD (autograd-safe).
        patches = k.unfold(2, ks, 1).unfold(3, ks, 1)  # [B, C, n_h, n_w, ks, ks]
        b2, c2, n_h, n_w, _, _ = patches.shape
        rows = patches.contiguous().view(b2, c2, n_h * n_w, ks * ks)
        if rows.shape[2] > self.max_rows:
            rows = rows[:, :, : self.max_rows, :]
        sv: torch.Tensor = torch.linalg.svdvals(rows)
        return self.weight * sv.sum(dim=-1).mean()


# ---------------------------------------------------------------------------
# Complex-domain regularizers
# ---------------------------------------------------------------------------


@register_loss(name="phase_smoothness_complex", aliases=["phase_smoothness"], domain="complex")
class PhaseSmoothnessComplexLoss(nn.Module):
    """Total-variation penalty on the unwrapped image phase.

    Uses the wrap-aware phase difference ``angle(exp(i * (phi[i+1] - phi[i])))``
    so the regulariser stays well-defined across the +/- pi boundary.

    Args:
        weight: scale factor.
        magnitude_weighted: if True, weight phase-TV by magnitude so noisy
            background pixels contribute less.
        eps: numerical stabiliser for the magnitude weighting.
    """

    def __init__(
        self,
        weight: float = 1.0,
        magnitude_weighted: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.weight = float(weight)
        self.magnitude_weighted = magnitude_weighted
        self.eps = float(eps)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        del target
        if not torch.is_complex(prediction):
            # If 2*C real layout assume interleaved Re/Im.
            if prediction.dim() >= 4 and prediction.shape[1] % 2 == 0:
                real = prediction[:, 0::2]
                imag = prediction[:, 1::2]
                prediction = torch.complex(real, imag)
            else:
                # Real-only input has zero phase; nothing to regularise.
                return prediction.new_zeros(()).to(prediction.dtype)
        # Wrap-aware finite differences via angle of conj(adjacent) * current.
        dx = prediction[..., 1:, :] * torch.conj(prediction[..., :-1, :])
        dy = prediction[..., :, 1:] * torch.conj(prediction[..., :, :-1])
        phase_dx = torch.angle(dx)
        phase_dy = torch.angle(dy)
        if self.magnitude_weighted:
            mag_dx = prediction[..., 1:, :].abs() + prediction[..., :-1, :].abs() + self.eps
            mag_dy = prediction[..., :, 1:].abs() + prediction[..., :, :-1].abs() + self.eps
            tv = (mag_dx * phase_dx.abs()).mean() + (mag_dy * phase_dy.abs()).mean()
        else:
            tv = phase_dx.abs().mean() + phase_dy.abs().mean()
        return self.weight * tv


__all__ = [
    "HankelLowRankLoss",
    "HermitianSymmetryLoss",
    "HessianSchattenLoss",
    "KspaceL1SparsityLoss",
    "PatchLowRankLoss",
    "PhaseSmoothnessComplexLoss",
    "WaveletSparsityLoss",
]
