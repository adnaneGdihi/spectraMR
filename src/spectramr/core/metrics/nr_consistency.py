r"""No-reference *consistency* metrics — self-consistency with the forward model.

This module implements the ``nr_consistency`` block of the NR metric battery
(spec §3). Every metric here answers "is the reconstruction self-consistent
with the acquisition physics?" rather than "how close is it to a reference?".
They consume a :class:`~spectramr.core.metrics.context.MetricContext` (acquired
k-space, sampling mask, coil maps, reconstructor, acquisition order). When the
context is absent the metric returns ``nan`` so it degrades gracefully in the
sim2rank sweep (never raising — ``core/metrics/computer.py`` re-raises
``ValueError`` and aborts the whole validation pass).

Forward model: :math:`\mathbf{y} = \mathbf{M}\mathbf{F}\mathbf{S}\mathbf{x} +
\mathbf{n}` with :math:`\mathbf{A} = \mathbf{M}\mathbf{F}\mathbf{S}`
(``sense_forward``), adjoint :math:`\mathbf{A}^H` (``sense_adjoint``).

Metrics
-------
- **ndcr** — Normalised Data-Consistency Residual (↓); also the cold-diffusion
  trust functional T1 :math:`\kappa_T` (alias ``kappa_T``)
- **prp_snr** — Pseudo-Replica Partition SNR (↑)
- **fmrs** — Forward-Model Round-Trip Stability (↓)
- **csoe** — Coil-Subspace Orthogonal Energy (↓, multi-coil only)
- **aoqv** — Acquisition-Order Quadratic Variation (↓)
- **eta_null** — Invented Null Energy, trust functional T2 (↓, no-reference)
- **fabrication_excess** — Fabrication Excess :math:`\mathcal{H}_T` (↓, full-reference)
- **null_band_energy_deficit** — unfilled fraction of the unacquired band (↓, full-reference)
"""

from __future__ import annotations

import torch
from torch import Tensor

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.registry import register_metric

NAN = float("nan")


# ---------------------------------------------------------------------------
# layout helpers (local; do not drag infrastructure into ``core``)
# ---------------------------------------------------------------------------
def _as_complex_image(x: Tensor) -> Tensor:
    """Coerce a prediction ``[B,C,H,W]`` / ``[B,1,H,W]`` / complex to complex ``[B,1,H,W]``.

    - complex input  → unchanged (kept as ``[B,C,H,W]`` complex)
    - even C real    → interleaved real/imag → ``[B,C//2,H,W]`` complex
    - odd/1 C real   → zero-imag complex
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    if torch.is_complex(x):
        return x.to(torch.complex64)
    c = x.shape[1]
    if c % 2 == 0 and c >= 2:
        xr = x.view(x.shape[0], c // 2, 2, x.shape[-2], x.shape[-1])
        return torch.complex(xr[:, :, 0], xr[:, :, 1]).to(torch.complex64)
    return x.to(torch.complex64)


def _single_channel_complex(x: Tensor) -> Tensor:
    """Reduce a complex image to a single-channel ``[B,1,H,W]`` (RSS over channels)."""
    xc = _as_complex_image(x)
    if xc.shape[1] == 1:
        return xc
    # RSS magnitude with zero phase — single-channel surrogate for sense_forward.
    mag = torch.sqrt((xc.real**2 + xc.imag**2).sum(dim=1, keepdim=True))
    return mag.to(torch.complex64)


def _coil_count(kspace: Tensor) -> int:
    return int(kspace.shape[1]) if kspace.dim() >= 4 else 1


def _norm(x: Tensor) -> Tensor:
    """Complex/real L2 norm over all dims of a (possibly batched) tensor → scalar per batch."""
    if torch.is_complex(x):
        return torch.sqrt((x.real**2 + x.imag**2).sum())
    return torch.sqrt((x**2).sum())


# ---------------------------------------------------------------------------
# ndcr — Normalised Data-Consistency Residual
# ---------------------------------------------------------------------------
@register_metric(
    "ndcr",
    aliases=["NDCR", "data_consistency_residual", "kappa_T", "t1_inconsistency"],
    requires_reference=False,
    needs=("y_kspace", "mask", "coil_maps"),
)
class NormalisedDataConsistencyResidual:
    r"""Normalised Data-Consistency Residual (↓).

    .. math::

        \mathrm{NDCR} = \frac{\lVert \mathbf{M}\odot(\mathbf{F}\mathbf{S}
        \hat{\mathbf{x}}) - \mathbf{y}\rVert_2}{\lVert\mathbf{y}\rVert_2}

    Measures how well the reconstruction explains the *acquired* k-space. A
    physically-consistent reconstruction re-projects to the measured samples;
    NDCR → noise floor (≈0) on a noiseless synthetic where
    :math:`\hat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{y}`, and rises
    monotonically with injected k-space error.

    Needs ``y_kspace`` + ``mask`` (``coil_maps`` optional → single-coil).
    Returns ``nan`` if the measurement context is absent.

    This is also the cold-diffusion trust functional **T1**, the (relative)
    inconsistency :math:`\kappa_T = d(D_T\hat x_0, x_T)/\lVert x_T\rVert` of
    the k-space cold-diffusion papers (aliases ``kappa_T`` /
    ``t1_inconsistency``): re-apply the terminal degradation to the output and
    compare against the actual measurement. Read it TOGETHER with ``eta_null``
    — low :math:`\kappa_T` with anomalously high :math:`\eta^{\mathrm{null}}`
    is the confident-fabrication signature.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "ndcr"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        # ``mask`` is optional: ``mask=None`` is a *fully-sampled* measurement
        # (the convention ``sense_forward``/``ifft2c`` already honour), so the
        # data-consistency residual is still well-defined on the whole of
        # k-space. Requiring a non-None mask here silently NaN'd NDCR on every
        # fully-sampled sweep (the default synthesised measurement) — the flagship
        # NR metric returned no value at all. Only ``y_kspace`` is mandatory.
        if not ctx.has("y_kspace"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import sense_forward

            y = ctx.y_kspace
            assert y is not None
            mask = ctx.mask
            smaps = ctx.coil_maps  # may be None → single-coil
            y = y if torch.is_complex(y) else y.to(torch.complex64)

            # Predicted image: single-channel when single-coil, else keep complex.
            if smaps is None:
                pred_img = _single_channel_complex(prediction)
            else:
                pred_img = _single_channel_complex(prediction)

            k_pred = sense_forward(pred_img, smaps=smaps, mask=mask)
            # Apply the same mask to y so we compare on acquired samples only.
            m = mask
            if m is not None and torch.is_complex(m):
                m = m.real
            y_obs = y * m if m is not None else y
            resid = _norm(k_pred - y_obs)
            denom = _norm(y_obs) + self.eps
            return float((resid / denom).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# prp_snr — Pseudo-Replica Partition SNR
# ---------------------------------------------------------------------------
def _split_mask_columns(mask: Tensor, *, parity: int) -> Tensor:
    """Split a sampling mask into two interleaved halves along the PE (column) axis.

    Returns the sub-mask keeping every other *sampled* column (``parity`` in
    {0,1}). Works on ``[H,W]`` / ``[C,H,W]`` / ``[B,?,H,W]`` masks by acting on
    the last (column) dim; a column is "sampled" if any of its entries is on.
    """
    m = mask.real if torch.is_complex(mask) else mask
    m = m.float()
    # Collapse everything but the last (column) axis to find sampled columns.
    col_energy = m
    while col_energy.dim() > 1:
        col_energy = col_energy.sum(dim=0)
    sampled = (col_energy > 0).nonzero(as_tuple=False).flatten()
    keep = sampled[parity::2]
    out = torch.zeros_like(m)
    out[..., keep] = m[..., keep]
    return out


@register_metric(
    "prp_snr",
    aliases=["PRP_SNR", "pseudo_replica_snr"],
    requires_reference=False,
    needs=("y_kspace", "mask", "coil_maps", "reconstructor"),
)
class PseudoReplicaPartitionSNR:
    r"""Pseudo-Replica Partition SNR (↑).

    Partition the acquired lines into two disjoint, interleaved halves
    :math:`\mathbf{M}_1,\mathbf{M}_2`, reconstruct
    :math:`\hat{\mathbf{x}}_1,\hat{\mathbf{x}}_2`, and form the difference
    :math:`\mathbf{d}=\hat{\mathbf{x}}_1-\hat{\mathbf{x}}_2` and the mean
    :math:`\bar{\mathbf{x}}=\tfrac12(\hat{\mathbf{x}}_1+\hat{\mathbf{x}}_2)`:

    .. math::

        \widehat{\sigma}^2 = \tfrac12\,\mathbb{E}[|d|^2],\qquad
        \mathrm{PRP\text{-}SNR} = \frac{\mathbb{E}|\bar x|}{\widehat\sigma}.

    With independent noise across the two halves, :math:`\widehat{\sigma}^2`
    is an unbiased estimate of the per-pixel noise variance, so SNR drops
    monotonically with injected noise. Reconstruction uses
    ``ctx.reconstructor`` when supplied, else the SENSE adjoint of each
    partition. Returns ``nan`` without ``y_kspace`` + ``mask``.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "prp_snr"

    @property
    def higher_is_better(self) -> bool:
        return True

    def _recon(
        self,
        ctx: MetricContext,
        sub_mask: Tensor,
    ) -> Tensor:
        y = ctx.y_kspace
        assert y is not None
        y = y if torch.is_complex(y) else y.to(torch.complex64)
        m = sub_mask.real if torch.is_complex(sub_mask) else sub_mask
        y_part = y * m
        if ctx.reconstructor is not None:
            out = ctx.reconstructor(y_part, sub_mask, ctx.coil_maps)
            return _single_channel_complex(out)
        from spectramr.infrastructure.physics.fft_ops import sense_adjoint

        img = sense_adjoint(y_part, smaps=ctx.coil_maps, mask=m)
        return _single_channel_complex(img)

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("y_kspace", "mask"):
            return NAN
        try:
            assert ctx.mask is not None
            m1 = _split_mask_columns(ctx.mask, parity=0)
            m2 = _split_mask_columns(ctx.mask, parity=1)
            x1 = self._recon(ctx, m1)
            x2 = self._recon(ctx, m2)
            d = x1 - x2
            xbar = 0.5 * (x1 + x2)
            sigma2 = 0.5 * (d.real**2 + d.imag**2).mean()
            sigma = torch.sqrt(sigma2 + self.eps)
            signal = xbar.abs().mean()
            return float((signal / sigma).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# fmrs — Forward-Model Round-Trip Stability
# ---------------------------------------------------------------------------
@register_metric(
    "fmrs",
    aliases=["FMRS", "round_trip_stability"],
    requires_reference=False,
    needs=("reconstructor", "mask"),
)
class ForwardModelRoundTripStability:
    r"""Forward-Model Round-Trip Stability (↓).

    .. math::

        \mathrm{FMRS} = \frac{1}{K}\sum_{k=1}^{K}
        \frac{\lVert \mathcal{R}(\mathbf{M}^{(k)}\odot\mathbf{F}\mathbf{S}
        \hat{\mathbf{x}}) - \hat{\mathbf{x}}\rVert_2}{\lVert\hat{\mathbf{x}}\rVert_2}

    Re-samples the reconstruction through ``K`` fresh masks of the *same*
    density and re-reconstructs; a projection-like (idempotent)
    reconstructor returns ≈0, an unstable one inflates the residual. Uses
    ``ctx.reconstructor`` when present, else the round trip
    ``sense_adjoint ∘ sense_forward``. Needs ``mask`` (for the density);
    returns ``nan`` otherwise.
    """

    def __init__(self, num_masks: int = 4, eps: float = 1e-8) -> None:
        self.num_masks = int(num_masks)
        self.eps = eps

    @property
    def name(self) -> str:
        return "fmrs"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("mask"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import (
                sense_adjoint,
                sense_forward,
            )
            from spectramr.infrastructure.physics.sampling import MaskGenerator

            xhat = _single_channel_complex(prediction)
            h, w = xhat.shape[-2], xhat.shape[-1]
            mask = ctx.mask
            m = mask.real if torch.is_complex(mask) else mask
            density = float((m > 0).float().mean().clamp(min=1e-3, max=1.0))
            # The VD mask generator needs R > 1 (a full-density resample is a
            # degenerate identity); floor the acceleration so the round trip is
            # actually an under-sampling round trip.
            accel = max(1.5, 1.0 / density)
            smaps = ctx.coil_maps

            # Deterministic seeds from ctx.generator if present, else fixed.
            base_seed = 0
            if ctx.generator is not None:
                base_seed = int(torch.randint(0, 2**31 - 1, (1,), generator=ctx.generator).item())

            xn = _norm(xhat) + self.eps
            acc = 0.0
            n = 0
            for k in range(self.num_masks):
                gen = MaskGenerator(seed=base_seed + k)
                mk = gen.generate_mask("variable_density", (h, w), acceleration_factor=accel)
                mk = mk.to(xhat.device)
                if ctx.reconstructor is not None:
                    yk = sense_forward(xhat, smaps=smaps, mask=mk)
                    rk = _single_channel_complex(ctx.reconstructor(yk, mk, smaps))
                else:
                    yk = sense_forward(xhat, smaps=smaps, mask=mk)
                    rk = _single_channel_complex(sense_adjoint(yk, smaps=smaps, mask=mk))
                acc += float((_norm(rk - xhat) / xn).detach())
                n += 1
            if n == 0:
                return NAN
            return acc / n
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# csoe — Coil-Subspace Orthogonal Energy
# ---------------------------------------------------------------------------
@register_metric(
    "csoe",
    aliases=["CSOE", "coil_subspace_orthogonal_energy"],
    requires_reference=False,
    needs=("y_kspace", "coil_maps"),
)
class CoilSubspaceOrthogonalEnergy:
    r"""Coil-Subspace Orthogonal Energy (↓, multi-coil only).

    .. math::

        \mathrm{CSOE} = \frac{\lVert(\mathbf{I}-\mathbf{V}\mathbf{V}^{H})
        \mathbf{Y}_{\mathrm{coils}}\rVert_2^2}{\lVert\mathbf{Y}_{\mathrm{coils}}
        \rVert_2^2}

    The ESPIRiT signal subspace :math:`\mathbf{V}` is the per-pixel unit coil
    vector spanned by the estimated sensitivity maps. Coil images produced by
    a single underlying image times smooth maps lie entirely in that 1-D
    subspace, so CSOE ≈ 0; inter-coil inconsistency (motion, phase drift)
    pushes energy into the orthogonal complement and CSOE rises.

    Single-coil cohorts return ``nan`` (the metric is undefined). Returns
    ``nan`` if ``y_kspace`` is absent.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "csoe"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("y_kspace"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            y = ctx.y_kspace
            assert y is not None
            y = y if torch.is_complex(y) else y.to(torch.complex64)
            if y.dim() != 4:
                return NAN
            num_coils = _coil_count(y)
            if num_coils < 2:
                return NAN  # single-coil — undefined

            # Per-pixel coil vectors in image domain.
            coil_imgs = ifft2c(y)  # [B, C, H, W] complex

            smaps = ctx.coil_maps
            if smaps is None:
                from spectramr.infrastructure.physics.coil_sensitivity import (
                    estimate_csm_espirit,
                )

                acs = min(24, y.shape[-2], y.shape[-1])
                smaps = estimate_csm_espirit(y, num_coils, acs_size=acs)
            smaps = smaps if torch.is_complex(smaps) else smaps.to(torch.complex64)

            # V = normalised per-pixel coil vector (1-D signal subspace).
            v = smaps  # [B, C, H, W]
            vnorm = torch.sqrt((v.real**2 + v.imag**2).sum(dim=1, keepdim=True) + self.eps)
            v = v / vnorm  # unit per pixel

            # Projection of Y_coils onto span(V): (V^H y) V.
            coeff = (torch.conj(v) * coil_imgs).sum(dim=1, keepdim=True)  # [B,1,H,W]
            proj = coeff * v  # [B, C, H, W]
            ortho = coil_imgs - proj

            num = (ortho.real**2 + ortho.imag**2).sum()
            den = (coil_imgs.real**2 + coil_imgs.imag**2).sum() + self.eps
            return float((num / den).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# aoqv — Acquisition-Order Quadratic Variation
# ---------------------------------------------------------------------------
@register_metric(
    "aoqv",
    aliases=["AOQV", "acq_order_quadratic_variation"],
    requires_reference=False,
    needs=("acq_order", "y_kspace", "reconstructor"),
)
class AcquisitionOrderQuadraticVariation:
    r"""Acquisition-Order Quadratic Variation (↓).

    Running reconstruction :math:`\hat{\mathbf{x}}^{(m)}=\mathcal{R}
    (\mathbf{y}_{1:m})` over the acquisition order:

    .. math::

        \mathrm{AOQV}=\sum_{m}\lVert\hat{\mathbf{x}}^{(m+1)}-
        \hat{\mathbf{x}}^{(m)}\rVert_2^2.

    A stationary acquisition yields a bounded, smoothly-growing running
    reconstruction; a non-stationary jump (motion / spike) of magnitude
    :math:`J` between two breakpoints adds ≈\ :math:`J^2`. Evaluated at a
    coarse set of breakpoints (``num_breaks``) to stay cheap. If
    ``acq_order`` is absent, columns are visited in index order. Uses
    ``ctx.reconstructor`` when present, else the SENSE adjoint. Returns
    ``nan`` without ``y_kspace``.
    """

    def __init__(self, num_breaks: int = 8, eps: float = 1e-8) -> None:
        self.num_breaks = int(num_breaks)
        self.eps = eps

    @property
    def name(self) -> str:
        return "aoqv"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _sampled_columns(self, ctx: MetricContext, w: int) -> Tensor:
        """Acquisition order of sampled columns (acq_order if given, else index)."""
        if ctx.acq_order is not None:
            order = ctx.acq_order.flatten().long()
            return order
        # Derive sampled columns from the mask (or all columns if no mask).
        if ctx.mask is not None:
            m = ctx.mask.real if torch.is_complex(ctx.mask) else ctx.mask
            col = m.float()
            while col.dim() > 1:
                col = col.sum(dim=0)
            return (col > 0).nonzero(as_tuple=False).flatten()
        return torch.arange(w)

    def _recon_prefix(
        self,
        ctx: MetricContext,
        cols: Tensor,
        h: int,
        w: int,
    ) -> Tensor:
        y = ctx.y_kspace
        assert y is not None
        y = y if torch.is_complex(y) else y.to(torch.complex64)
        prefix_mask = torch.zeros((h, w), device=y.device)
        prefix_mask[:, cols] = 1.0
        y_pref = y * prefix_mask
        if ctx.reconstructor is not None:
            return _single_channel_complex(ctx.reconstructor(y_pref, prefix_mask, ctx.coil_maps))
        from spectramr.infrastructure.physics.fft_ops import sense_adjoint

        return _single_channel_complex(sense_adjoint(y_pref, smaps=ctx.coil_maps, mask=prefix_mask))

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("y_kspace"):
            return NAN
        try:
            y = ctx.y_kspace
            assert y is not None
            h, w = y.shape[-2], y.shape[-1]
            order = self._sampled_columns(ctx, w)
            n = int(order.numel())
            if n < 2:
                return NAN

            # Coarse breakpoints over the acquisition order.
            k = min(self.num_breaks, n)
            idx = torch.linspace(1, n, steps=k).round().long().clamp(1, n)
            idx = torch.unique(idx)

            prev: Tensor | None = None
            total = 0.0
            for m in idx.tolist():
                cols = order[:m].to(y.device).long()
                xm = self._recon_prefix(ctx, cols, h, w)
                if prev is not None:
                    diff = xm - prev
                    total += float((diff.real**2 + diff.imag**2).sum().detach())
                prev = xm
            return total
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# nse_hall — Null-Space-Energy Hallucination metric (A-8.2)
# ---------------------------------------------------------------------------
@register_metric(
    "nse_hall",
    aliases=["NSE_Hall", "null_space_energy_hallucination"],
    requires_reference=True,
    needs=("mask",),
    direction="lower",
)
class NullSpaceEnergyHallucination:
    r"""Null-Space-Energy Hallucination fraction (↓).

    Decomposes the reconstruction error ``e = x̂ - x`` against the **Fourier
    sampling** operator ``A = M F`` (acquired k-space lines ``M``). The Fourier
    sampling **null space** is the set of images that change no *acquired Fourier
    line* — where a model is free to *hallucinate*. With the projector
    ``P_null = I - A⁺A = F^H (1-M) F`` (idempotent because ``M`` is a 0/1 mask and
    ``F`` is unitary),

    .. math::

        \mathrm{NSE\text{-}Hall} =
        \frac{\lVert P_{\text{null}}\,(\hat{\mathbf{x}}-\mathbf{x})\rVert_2^2}
             {\lVert \hat{\mathbf{x}}-\mathbf{x}\rVert_2^2} \in [0,1],

    the **fraction of the reconstruction error that lives in the unmeasured
    Fourier null space**. It is the orthogonal complement of the Fourier
    range/data-consistency error: together they partition the total error
    (Parseval). ``NSE-Hall → 1`` means the error is pure unacquired-line content;
    ``→ 0`` means it is confined to the measured lines. Unlike a global MSE/PSNR it
    is sensitive to *where* in the sampling geometry the error sits.

    **Scope (read this).**

    * **Single-coil** (``y = M F x``): the projector is *exact* — ``P_null`` is the
      true measurement null space, and NSE-Hall is the exact hallucination fraction.
    * **Multi-coil SENSE** (``y = M F S x``): this is a *dominant-subspace
      approximation*. The true null space is ``null(M F S)``; coil-sensitivity phase
      coupling lets a fraction of a Fourier-null image still perturb the acquired
      samples, so a Fourier-null error is partly SENSE-*measurable*. NSE-Hall
      therefore **over-counts** hallucination by that coupling fraction (empirically
      ~5-25 % depending on coil-phase diversity). It still ranks recons sensibly but
      is not operator-exact here. A true SENSE-null projector needs an iterative
      ``(MFS)⁺`` and is out of scope for a per-batch metric.

    **Not a standalone score / pairing requirement.** A near-perfect recon can have
    a high null-space *fraction* of a tiny error, and a recon can *lower* NSE-Hall
    by pushing error into the measured lines (breaking data consistency). So this is
    a hallucination *diagnostic* that MUST be read alongside a total-error metric
    (e.g. ``ndcr`` / PSNR); do not use it alone as an early-stopping / ranking
    monitor (config-audit guard backlogged — A-8.2 note).

    Full-reference; needs ``mask`` (multi-coil predictions are RSS-combined first).
    Returns ``nan`` if ``mask`` is absent or the error is ~zero (a perfect
    reconstruction has no error to decompose — pitfall #20, no measurement-
    independent output).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "nse_hall"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if target is None or not ctx.has("mask"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

            pred = _single_channel_complex(prediction)
            tgt = _single_channel_complex(target)
            if pred.shape != tgt.shape:
                return NAN
            err = pred - tgt  # complex [B,1,H,W]
            total = (err.real**2 + err.imag**2).sum()
            if float(total) < self.eps:
                return NAN  # perfect recon: no error to decompose (pitfall #20)

            m = ctx.mask.to(dtype=torch.float32, device=err.device)
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
            # P_null e = F^H (1 - M) F e — error in the UNMEASURED k-space lines.
            null_img = ifft2c((1.0 - m) * fft2c(err))
            null_energy = (null_img.real**2 + null_img.imag**2).sum()
            return float((null_energy / total).clamp(0.0, 1.0).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# eta_null — Invented Null Energy (cold-diffusion trust functional T2)
# ---------------------------------------------------------------------------
@register_metric(
    "eta_null",
    aliases=["null_energy_fraction", "invented_null_energy", "t2_null_energy"],
    requires_reference=False,
    needs=("mask",),
)
class InventedNullEnergy:
    r"""Invented Null Energy fraction :math:`\eta^{\mathrm{null}}` (↓).

    .. math::

        \eta^{\mathrm{null}} =
        \frac{\lVert(\mathbf{1}-\mathbf{M})\odot\mathbf{F}\hat{\mathbf{x}}\rVert_2}
             {\lVert\mathbf{F}\hat{\mathbf{x}}\rVert_2} \in [0,1]

    The fraction of the reconstruction's k-space energy sitting on
    **unacquired** lines — literally "how much of this image was invented".
    No-reference: needs only the sampling mask. Unlike ``nse_hall`` (which
    decomposes the *error* against a reference), this scores the output
    itself, so it deploys at inference time.

    Interpretation is calibration-relative, not absolute: every non-trivial
    image has null-band energy, so the trust read-out compares
    :math:`\eta^{\mathrm{null}}` against its clean-data law (calibration
    tier), with low ``ndcr``/:math:`\kappa_T` + anomalously high
    :math:`\eta^{\mathrm{null}}` as the confident-fabrication signature.

    Returns ``nan`` if ``mask`` is absent: with a fully-sampled measurement
    the value is identically 0 for every input — a measurement-independent
    constant (pitfall #20), not a score.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "eta_null"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if not ctx.has("mask"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import fft2c

            pred = _single_channel_complex(prediction)
            k_pred = fft2c(pred)
            m = ctx.mask
            assert m is not None
            if torch.is_complex(m):
                m = m.real
            m = m.to(dtype=torch.float32, device=k_pred.device)
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
            null_energy = _norm((1.0 - m) * k_pred)
            total = _norm(k_pred) + self.eps
            return float((null_energy / total).clamp(0.0, 1.0).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# fabrication_excess — null-band error energy (cold-diffusion H_T)
# ---------------------------------------------------------------------------
@register_metric(
    "fabrication_excess",
    aliases=["H_T", "null_band_error"],
    requires_reference=True,
    needs=("mask",),
)
class FabricationExcess:
    r"""Fabrication Excess :math:`\mathcal{H}_T` (↓, full-reference).

    .. math::

        \mathcal{H}_T =
        \frac{\lVert(\mathbf{1}-\mathbf{M})\odot
              (\mathbf{F}\hat{\mathbf{x}}-\mathbf{F}\mathbf{x})\rVert_2}
             {\lVert\mathbf{F}\mathbf{x}\rVert_2}

    The distance from the reconstruction to the true fibre of the sampling
    operator: for the linear masking degradation the fibre through the ground
    truth is exactly "same acquired lines, anything on the complement", so the
    fibre distance is the **unacquired-line** discrepancy against the truth.
    Normalised by *total* truth energy (the null band alone can be nearly
    empty, which would blow the ratio up).

    Complements ``nse_hall``: NSE-Hall reports the null-space *fraction* of
    the error (where the error sits), :math:`\mathcal{H}_T` reports its *magnitude*
    relative to the truth (how much was fabricated). A hard-DC reconstruction
    has ALL of its error in the null band (``nse_hall`` ≈ 1) yet may still
    have tiny :math:`\mathcal{H}_T`.

    Validation-tier (needs ground truth). Returns ``nan`` when ``mask`` or
    the reference is absent, or on a shape mismatch.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "fabrication_excess"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if target is None or not ctx.has("mask"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import fft2c

            pred = _single_channel_complex(prediction)
            tgt = _single_channel_complex(target)
            if pred.shape != tgt.shape:
                return NAN
            k_err = fft2c(pred) - fft2c(tgt)
            m = ctx.mask
            assert m is not None
            if torch.is_complex(m):
                m = m.real
            m = m.to(dtype=torch.float32, device=k_err.device)
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
            null_err = _norm((1.0 - m) * k_err)
            denom = _norm(fft2c(tgt)) + self.eps
            return float((null_err / denom).detach())
        except Exception:
            return NAN


# ---------------------------------------------------------------------------
# null_band_energy_deficit — how much of the unacquired band was left empty
# ---------------------------------------------------------------------------
@register_metric(
    "null_band_energy_deficit",
    aliases=["null_fill_deficit"],
    requires_reference=True,
    needs=("mask",),
    direction="lower",
)
class NullBandEnergyDeficit:
    r"""Unfilled fraction of the unacquired band's energy (↓, full-reference).

    .. math::

        \mathcal{D} = \max\!\left(0,\; 1 -
        \frac{\lVert(\mathbf{1}-\mathbf{M})\odot\mathbf{F}\hat{\mathbf{x}}\rVert_2}
             {\lVert(\mathbf{1}-\mathbf{M})\odot\mathbf{F}\mathbf{x}\rVert_2}\right)
        \in [0,1]

    A zero-filled reconstruction leaves the unacquired band empty and scores
    **1.0**; one carrying the truth's null-band energy scores **0.0**. It is the
    direct answer to "how much of what was missing did the model actually put
    back?", and it is the one direction the rest of the battery cannot see: a
    reconstruction whose null band is uniformly attenuated keeps ``ndcr`` at the
    noise floor (hard DC pins the acquired lines), keeps ``nse_hall`` near 1
    (all of its error *is* in the null band, whatever its size), and moves
    ``eta_null`` only in proportion — none of them reports the deficit itself.

    **Why the bare ratio is not what is registered.** The underlying quantity is
    two-sided: below 1 is blur, above 1 is fabrication. ``direction`` admits only
    ``"higher"``/``"lower"``, so a two-sided value registered under either label
    would be read backwards by ``metric_higher_is_better`` and by every consumer
    that resolves through it (``keep_best_n``, early stopping, the leaderboard).
    Clamping at 0 makes the registered value monotone and honest, and the
    fabrication direction keeps its existing single owner in
    :class:`FabricationExcess` (non-negotiable 17) — read the two together.

    **Scope.** Like ``eta_null`` this uses the *Fourier* sampling operator, so on
    multi-coil SENSE data it is a dominant-subspace approximation (see
    :class:`NullSpaceEnergyHallucination` for the size of that bias). Both the
    reconstruction and any baseline it is compared against cross the identical
    code, so the bias largely cancels in a ``zf_delta``.

    Returns ``nan`` when ``mask`` or the reference is absent, on a shape
    mismatch, or when the target's null band is empty (a fully-sampled
    measurement has nothing to fill — a measurement-independent constant is not
    a score, pitfall #9).
    """

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = eps

    @property
    def name(self) -> str:
        return "null_band_energy_deficit"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: Tensor,
        target: Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: object,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        if target is None or not ctx.has("mask"):
            return NAN
        try:
            from spectramr.infrastructure.physics.fft_ops import fft2c

            pred = _single_channel_complex(prediction)
            tgt = _single_channel_complex(target)
            if pred.shape != tgt.shape:
                return NAN
            m = ctx.mask
            assert m is not None
            if torch.is_complex(m):
                m = m.real
            k_pred = fft2c(pred)
            m = m.to(dtype=torch.float32, device=k_pred.device)
            if m.dim() == 2:
                m = m[None, None]
            elif m.dim() == 3:
                m = m[:, None]
            null = 1.0 - m
            target_null = _norm(null * fft2c(tgt))
            # A fully-sampled mask leaves nothing to fill, so the ratio is 0/0.
            # Report it as absent rather than as a perfect or a total deficit.
            if float(target_null) <= self.eps:
                return NAN
            ratio = _norm(null * k_pred) / target_null
            return float((1.0 - ratio).clamp(0.0, 1.0).detach())
        except Exception:
            return NAN


__all__ = [
    "AcquisitionOrderQuadraticVariation",
    "CoilSubspaceOrthogonalEnergy",
    "FabricationExcess",
    "ForwardModelRoundTripStability",
    "InventedNullEnergy",
    "NormalisedDataConsistencyResidual",
    "NullBandEnergyDeficit",
    "NullSpaceEnergyHallucination",
    "PseudoReplicaPartitionSNR",
]
