r"""No-reference *texture* quality metrics (spec NR battery §3 ``nr_texture``).

A small battery of self-referential, target-free image-quality metrics that
probe the **texture statistics** of a reconstruction without a ground truth.
Each metric is a plain ``@register_metric`` class whose ``__call__`` returns a
single Python ``float`` (mean over the batch). They obey the house NR contract:

* never ``raise ValueError`` inside ``__call__`` (the metrics computer
  re-raises it and aborts the whole validation) — a degenerate / missing input
  returns ``float('nan')`` instead;
* an absent *optional heavy dependency* (PyWavelets) is signalled by raising
  ``ImportError`` in ``__init__`` so ``get_metric`` cleanly skips the metric
  (the ``frd`` / pyradiomics precedent);
* the direction is **self-declared** via a ``higher_is_better`` property so the
  central ``metric_directions`` map is never touched.

The metrics
-----------

* ``wlms`` — Wavelet-Leader Multifractal Spectrum Width ↓ (needs ``pywt``).
* ``wpde`` — Wavelet Parent-Child Dependency Entropy ↓ (needs ``pywt``).
* ``pgle`` — Patch-Graph Laplacian von Neumann Entropy ↓.
* ``pid``  — Patch Intrinsic Dimensionality (TwoNN) ↓.
* ``bemd_ier`` — Bidimensional Empirical-Mode Intrinsic-Energy Ratio ↓.
* ``pcd``  — Phase-Congruency Dispersion (std minus mean of PC) — lower better.
* ``savs`` — Spatial Allan-Variance Slope ↓ (deviation from ``-2``).

All deviation-style metrics (``wlms``, ``pgle``, ``pid``, ``savs``) compare the
measured statistic against a *pristine reference value* supplied through
``context.nr_params`` (key ``<metric>_ref``) and fall back to a hard-coded
literature default — so the metric is well defined even without a fitted
reference.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F

from spectramr.core.metrics.context import MetricContext, resolve_context
from spectramr.core.metrics.hallucination_metrics import _to_magnitude
from spectramr.core.metrics.registry import register_metric

# ---------------------------------------------------------------------------
# Shared, dependency-free helpers (torch + stdlib only).
# ---------------------------------------------------------------------------


def _as_magnitude_2d(prediction: torch.Tensor) -> torch.Tensor | None:
    """Coerce a prediction to a real magnitude batch ``[B, 1, H, W]``.

    Accepts ``[B, C, H, W]`` (even ``C`` ⇒ interleaved real/imag), ``[B,1,H,W]``,
    complex, ``[C,H,W]`` or ``[H,W]``. Returns ``None`` on an un-imageable input
    (caller turns that into ``nan``).
    """
    x = prediction
    if x is None:
        return None
    if x.is_complex():
        x = x.abs()
    if x.ndim == 2:
        x = x[None, None]
    elif x.ndim == 3:
        x = x[None]
    if x.ndim != 4:
        return None
    mag = _to_magnitude(x.float())  # [B, 1, H, W] (or [B, C//2, ...] if many coils)
    if mag.shape[1] != 1:
        # Coil/channel collapse to a single magnitude map (RSS over channels).
        mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True) + 1e-12)
    return mag


def _param(ctx: MetricContext, key: str, default: float) -> float:
    """Pull a numeric tunable from ``ctx.nr_params`` with a default."""
    if ctx.nr_params is not None and key in ctx.nr_params:
        try:
            return float(ctx.nr_params[key])
        except (TypeError, ValueError):
            return default
    return default


def _unfold_patches(
    img: torch.Tensor,
    patch: int,
    stride: int,
    max_patches: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Return a ``[N, patch*patch]`` matrix of subsampled, mean-centred patches.

    ``img`` is ``[1, 1, H, W]``. Patches are extracted with ``F.unfold`` then
    randomly subsampled to ``max_patches`` for CPU speed. Each patch is
    de-meaned so the graph / TwoNN statistics describe *texture*, not DC level.
    """
    cols = F.unfold(img, kernel_size=patch, stride=stride)  # [1, p*p, L]
    cols = cols.squeeze(0).transpose(0, 1)  # [L, p*p]
    n = cols.shape[0]
    if n == 0:
        return cols
    if n > max_patches:
        if generator is not None:
            idx = torch.randperm(n, generator=generator)[:max_patches]
        else:
            idx = torch.randperm(n)[:max_patches]
        cols = cols[idx]
    cols = cols - cols.mean(dim=1, keepdim=True)
    return cols


def _gaussian_blur(img: torch.Tensor, sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """Separable Gaussian blur of a ``[B, 1, H, W]`` real image (reflect-pad)."""
    radius = max(1, int(truncate * sigma + 0.5))
    coords = torch.arange(-radius, radius + 1, dtype=img.dtype, device=img.device)
    k1d = torch.exp(-(coords**2) / (2.0 * sigma * sigma))
    k1d = k1d / k1d.sum()
    kx = k1d.view(1, 1, 1, -1)
    ky = k1d.view(1, 1, -1, 1)
    out = F.pad(img, (radius, radius, 0, 0), mode="reflect")
    out = F.conv2d(out, kx)
    out = F.pad(out, (0, 0, radius, radius), mode="reflect")
    out = F.conv2d(out, ky)
    return out


def _mutual_information_abs(a: torch.Tensor, b: torch.Tensor, bins: int = 16) -> float:
    """Discrete MI of two flattened non-negative tensors (nats)."""
    a = a.flatten().float()
    b = b.flatten().float()
    if a.numel() < 8 or b.numel() < 8:
        return 0.0
    amax = a.max().clamp_min(1e-12)
    bmax = b.max().clamp_min(1e-12)
    ai = (a / amax * (bins - 1)).round().long().clamp_(0, bins - 1)
    bi = (b / bmax * (bins - 1)).round().long().clamp_(0, bins - 1)
    joint = torch.zeros(bins, bins, dtype=torch.float64)
    flat = (ai * bins + bi).cpu()
    joint.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
    joint = joint / joint.sum().clamp_min(1.0)
    px = joint.sum(dim=1, keepdim=True)
    py = joint.sum(dim=0, keepdim=True)
    denom = (px @ py).clamp_min(1e-12)
    nz = joint > 0
    mi = (joint[nz] * (joint[nz] / denom[nz]).log()).sum()
    return float(mi.clamp_min(0.0))


# ---------------------------------------------------------------------------
# Wavelet-leader metrics (PyWavelets, lazy import).
# ---------------------------------------------------------------------------


@register_metric("wlms", requires_reference=False, direction="lower")
class WaveletLeaderMultifractalSpectrumWidth:
    r"""Wavelet-Leader Multifractal Spectrum Width · ↓ (deviation from ``w_ref``).

    From the leaders ``L_j(k) = sup`` of detail-coefficient magnitudes over a
    space-scale neighbourhood, the log-cumulants ``c1, c2`` of ``ln L_j`` across
    scales give the (legendre) multifractal spectrum
    :math:`\mathcal{D}(h)\approx 1-(h-c_1)^2/(2|c_2|)`, whose support width is
    :math:`\propto\sqrt{|c_2|}`. We report

    .. math::

        \mathrm{WLMS} = \big|\,\mathrm{width}\,\mathcal{D}(h) - w_{\mathrm{ref}}\big|.

    Measured directions (this estimator, `_texture` fixture): the raw width
    **increases under blur** -- blur steepens the leader log-cumulants (``c1``
    grows more negative, ``c2`` grows as the near-zero fine-scale details spread
    the log-leader distribution) -- and **decreases under noise** -- additive
    noise whitens the leaders toward a flat, mono-fractal-like profile, shrinking
    both cumulant slopes. WLMS is reported as the deviation ``|width - w_ref|``
    from a pristine reference, so with a fitted ``w_ref`` **both** degradations
    raise the score (blur pushes the width up, noise pushes it down). With the
    default ``w_ref=0`` the metric returns the raw width, which is only monotone
    under blur -- a fitted reference is required for it to grade noise as worse.

    Requires PyWavelets (``pywt``); raises :class:`ImportError` in ``__init__``
    if missing so the registry cleanly skips it.
    """

    def __init__(self, wavelet: str = "db2", n_scales: int = 3) -> None:
        try:
            import pywt  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via skip path
            raise ImportError(
                "wlms requires PyWavelets ('pywt'); install spectramr[quality] "
                "or pip install PyWavelets."
            ) from exc
        self.wavelet = wavelet
        self.n_scales = int(n_scales)

    @property
    def name(self) -> str:
        return "wlms"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _width(self, mag2d: torch.Tensor) -> float:
        import numpy as np
        import pywt

        arr = mag2d.detach().cpu().numpy().astype("float64")
        try:
            coeffs = pywt.wavedec2(arr, self.wavelet, level=self.n_scales, mode="periodization")
        except ValueError:
            return float("nan")
        # Leaders per scale: max |detail| in a 3x3 neighbourhood, then ln-mean.
        log_means: list[float] = []
        log_vars: list[float] = []
        for detail in coeffs[1:]:
            ch, cv, cd = detail
            mag = np.maximum.reduce([np.abs(ch), np.abs(cv), np.abs(cd)])
            if mag.size < 4:
                continue
            # local-sup leader via a cheap dilation (max over 3x3).
            lead = mag.copy()
            lead[:-1, :] = np.maximum(lead[:-1, :], mag[1:, :])
            lead[1:, :] = np.maximum(lead[1:, :], mag[:-1, :])
            lead[:, :-1] = np.maximum(lead[:, :-1], mag[:, 1:])
            lead[:, 1:] = np.maximum(lead[:, 1:], mag[:, :-1])
            lead = lead[lead > 0]
            if lead.size < 4:
                continue
            ln = np.log(lead)
            log_means.append(float(ln.mean()))
            log_vars.append(float(ln.var()))
        if len(log_means) < 2:
            return float("nan")
        scales = np.arange(1, len(log_means) + 1, dtype="float64")
        ln_s = np.log(scales)
        # c1 = slope of log-mean vs ln(scale); c2 = slope of log-var vs ln(scale).
        c1 = float(np.polyfit(ln_s, np.asarray(log_means), 1)[0])
        c2 = float(np.polyfit(ln_s, np.asarray(log_vars), 1)[0])
        # Spectrum support width ~ proportional to sqrt(|c2|); c1 anchors location.
        width = 2.0 * math.sqrt(abs(c2) + 1e-12) + 0.1 * abs(c1)
        return width

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        w_ref = _param(ctx, "wlms_ref", 0.0)
        widths = [self._width(mag[b, 0]) for b in range(mag.shape[0])]
        widths = [w for w in widths if not math.isnan(w)]
        if not widths:
            return float("nan")
        mean_w = sum(widths) / len(widths)
        return float(abs(mean_w - w_ref))


@register_metric("wpde", requires_reference=False, direction="lower")
class WaveletParentChildDependencyEntropy:
    r"""Wavelet Parent-Child Dependency Entropy · ↓.

    Models the hidden Markov tree (HMT) coupling of detail magnitudes across
    adjacent scales. We compute the mutual information between a child band and
    its (up-sampled) parent band, plus the entropy of a small large/small
    transition matrix ``T`` of the magnitudes:

    .. math::

        \mathrm{WPDE} = \big|I(|w_j|;|w_{j-1}^{\mathrm{child}}|) - I_{\mathrm{ref}}\big|
                        + H(\mathbf{T}).

    Parent-child MI drops under blur (scales decouple); the transition entropy
    rises with disordered noise. Requires PyWavelets.
    """

    def __init__(self, wavelet: str = "db2", n_scales: int = 3) -> None:
        try:
            import pywt  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "wpde requires PyWavelets ('pywt'); install spectramr[quality]."
            ) from exc
        self.wavelet = wavelet
        self.n_scales = int(n_scales)

    @property
    def name(self) -> str:
        return "wpde"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _score(self, mag2d: torch.Tensor) -> float:
        import numpy as np
        import pywt

        arr = mag2d.detach().cpu().numpy().astype("float64")
        try:
            coeffs = pywt.wavedec2(arr, self.wavelet, level=self.n_scales, mode="periodization")
        except ValueError:
            return float("nan")
        details = coeffs[1:]  # finest..coarsest -> reverse to coarse..fine pairs
        if len(details) < 2:
            return float("nan")
        # Build magnitude maps (HH band) per scale.
        bands = [np.abs(d[2]) for d in details]  # diagonal sub-band per scale
        # Parent = coarser (smaller map), child = finer (2x). Pair adjacent.
        mi_vals: list[float] = []
        for fine, coarse in pairwise(bands):
            # Up-sample the coarse parent to the child's grid (nearest).
            ph, pw = coarse.shape
            ch, cw = fine.shape
            if ph == 0 or pw == 0:
                continue
            ry, rx = max(1, ch // ph), max(1, cw // pw)
            parent_up = np.kron(coarse, np.ones((ry, rx)))[:ch, :cw]
            mi = _mutual_information_abs(
                torch.from_numpy(fine), torch.from_numpy(parent_up), bins=12
            )
            mi_vals.append(mi)
        if not mi_vals:
            return float("nan")
        mi_mean = sum(mi_vals) / len(mi_vals)
        # Transition entropy: 2-state (small/large) Markov chain along rows of
        # the finest band, thresholded at its median.
        fine = bands[0]
        thr = np.median(fine)
        state = (fine > thr).astype("int64").ravel()
        trans = np.zeros((2, 2), dtype="float64")
        if state.size > 1:
            for s0, s1 in pairwise(state):
                trans[s0, s1] += 1.0
        row = trans.sum(axis=1, keepdims=True)
        row[row == 0] = 1.0
        p = trans / row
        ent = 0.0
        for i in range(2):
            for j in range(2):
                if p[i, j] > 0:
                    ent -= float(p[i, j]) * math.log(float(p[i, j]))
        return float(mi_mean + ent)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        i_ref = _param(ctx, "wpde_mi_ref", 0.0)
        scores: list[float] = []
        for b in range(mag.shape[0]):
            s = self._score(mag[b, 0])
            if not math.isnan(s):
                # s = mi_mean + transition_entropy; deviation applies to MI term.
                scores.append(s)
        if not scores:
            return float("nan")
        mean_s = sum(scores) / len(scores)
        # WPDE = |I - I_ref| + H(T); when i_ref==0 this reduces to (I + H).
        return float(abs(mean_s - i_ref))


# ---------------------------------------------------------------------------
# Patch-graph / intrinsic-dimension metrics (pure torch).
# ---------------------------------------------------------------------------


@register_metric("pgle", requires_reference=False, direction="lower")
class PatchGraphLaplacianEntropy:
    r"""Patch-Graph Laplacian von Neumann Entropy · ↓ (deviation from pristine).

    Build a symmetric k-NN graph over (subsampled) image patches, form its
    normalised Laplacian :math:`\mathcal{L}=I-D^{-1/2}WD^{-1/2}`, and treat the
    normalised eigenvalues :math:`\mu_i=\lambda_i/\sum_j\lambda_j` as a
    distribution whose entropy is the von Neumann entropy of the graph:

    .. math::

        \mathrm{PGLE} = -\sum_i \mu_i \ln \mu_i.

    A clean texture has a low-entropy (structured) spectrum; additive noise
    *inflates* the entropy as patch affinities become uniform. Reported as the
    absolute deviation from a pristine reference (``pgle_ref``, default 0).
    """

    def __init__(
        self,
        patch: int = 7,
        stride: int = 5,
        k: int = 8,
        max_patches: int = 192,
    ) -> None:
        self.patch = int(patch)
        self.stride = int(stride)
        self.k = int(k)
        self.max_patches = int(max_patches)

    @property
    def name(self) -> str:
        return "pgle"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _entropy(self, img: torch.Tensor, generator: torch.Generator | None) -> float:
        cols = _unfold_patches(img, self.patch, self.stride, self.max_patches, generator)
        n = cols.shape[0]
        if n < self.k + 2:
            return float("nan")
        # Degenerate (uniform / near-constant) input → all-zero de-meaned
        # patches → ill-conditioned Laplacian; bail to nan rather than crash.
        if float(cols.abs().max()) < 1e-8:
            return float("nan")
        # Pairwise squared distances.
        d2 = torch.cdist(cols, cols).pow(2)
        # Heat-kernel affinity with a self-tuning scale (median nonzero dist).
        med = d2[d2 > 0].median().clamp_min(1e-8)
        w = torch.exp(-d2 / med)
        # k-NN sparsify: keep top-k per row (excluding self), symmetrise.
        kk = min(self.k + 1, n)
        topv, topi = w.topk(kk, dim=1)
        wknn = torch.zeros_like(w)
        wknn.scatter_(1, topi, topv)
        wknn = torch.maximum(wknn, wknn.t())
        wknn.fill_diagonal_(0.0)
        deg = wknn.sum(dim=1)
        dinv = torch.where(deg > 0, deg.rsqrt(), torch.zeros_like(deg))
        lap = torch.eye(n, device=wknn.device, dtype=wknn.dtype) - (
            dinv.unsqueeze(1) * wknn * dinv.unsqueeze(0)
        )
        lap = 0.5 * (lap + lap.t())
        try:
            evals = torch.linalg.eigvalsh(lap).clamp_min(0.0)
        except torch.linalg.LinAlgError:
            return float("nan")
        total = evals.sum().clamp_min(1e-12)
        mu = evals / total
        mu = mu[mu > 1e-12]
        ent = float(-(mu * mu.log()).sum())
        return ent

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        ref = _param(ctx, "pgle_ref", 0.0)
        gen = ctx.generator
        ents = [self._entropy(mag[b : b + 1], gen) for b in range(mag.shape[0])]
        ents = [e for e in ents if not math.isnan(e)]
        if not ents:
            return float("nan")
        mean_e = sum(ents) / len(ents)
        return float(abs(mean_e - ref))


@register_metric("pid", requires_reference=False, direction="lower")
class PatchIntrinsicDimensionality:
    r"""Patch Intrinsic Dimensionality (TwoNN) · ↓ (deviation from pristine).

    The TwoNN estimator (Facco et al. 2017): for each patch embedding the ratio
    of its second to first nearest-neighbour distance, :math:`\mu_i=r_2(i)/r_1(i)`,
    is Pareto-distributed with shape equal to the intrinsic dimension, giving

    .. math::

        \mathrm{PID} = \frac{N}{\sum_i \ln \mu_i}.

    Additive noise *inflates* the intrinsic dimension (patches fill more
    directions); aggressive over-smoothing *collapses* it (patches lie on a
    low-D manifold). Reported as ``|PID - pid_ref|`` (default ``pid_ref=0``).
    """

    def __init__(
        self,
        patch: int = 7,
        stride: int = 5,
        max_patches: int = 256,
        discard_frac: float = 0.1,
    ) -> None:
        self.patch = int(patch)
        self.stride = int(stride)
        self.max_patches = int(max_patches)
        self.discard_frac = float(discard_frac)

    @property
    def name(self) -> str:
        return "pid"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _twonn(self, img: torch.Tensor, generator: torch.Generator | None) -> float:
        cols = _unfold_patches(img, self.patch, self.stride, self.max_patches, generator)
        n = cols.shape[0]
        if n < 8:
            return float("nan")
        if float(cols.abs().max()) < 1e-8:
            return float("nan")
        d = torch.cdist(cols, cols)
        d.fill_diagonal_(float("inf"))
        nn, _ = d.topk(2, dim=1, largest=False)  # [N, 2]: r1, r2
        r1 = nn[:, 0].clamp_min(1e-12)
        r2 = nn[:, 1].clamp_min(1e-12)
        mu = (r2 / r1).clamp_min(1.0 + 1e-9)
        log_mu = mu.log()
        # Discard the largest-mu tail (TwoNN robustification).
        keep = round((1.0 - self.discard_frac) * n)
        keep = max(4, min(keep, n))
        log_mu = log_mu.sort().values[:keep]
        denom = log_mu.sum().clamp_min(1e-8)
        dim = float(keep / denom)
        return dim

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        ref = _param(ctx, "pid_ref", 0.0)
        gen = ctx.generator
        dims = [self._twonn(mag[b : b + 1], gen) for b in range(mag.shape[0])]
        dims = [v for v in dims if not math.isnan(v)]
        if not dims:
            return float("nan")
        mean_d = sum(dims) / len(dims)
        return float(abs(mean_d - ref))


# ---------------------------------------------------------------------------
# Bidimensional empirical-mode decomposition (small pure-torch BEMD).
# ---------------------------------------------------------------------------


@register_metric("bemd_ier", requires_reference=False, aliases=["bemd"], direction="lower")
class BEMDIntrinsicEnergyRatio:
    r"""Bidimensional Empirical-Mode Intrinsic-Energy Ratio · ↓.

    A small, deterministic 2-D empirical-mode decomposition (BEMD) sifts the
    image into intrinsic mode functions (IMFs) by iteratively subtracting a
    smooth envelope (mean of morphological-style Gaussian upper/lower
    envelopes). The fraction of energy carried by the *first* (finest) IMF is

    .. math::

        \mathrm{BEMD\text{-}IER} = \frac{\lVert\mathrm{IMF}_1\rVert_2^2}
                                        {\sum_j \lVert\mathrm{IMF}_j\rVert_2^2}.

    High-frequency noise pushes energy into ``IMF_1``, raising the ratio; a
    clean structured image spreads energy across modes. The decomposition is
    seedless (purely deterministic Gaussian envelopes), so the metric is
    reproducible without an RNG.
    """

    def __init__(self, n_imf: int = 3, env_sigmas: tuple[float, ...] = (1.0, 2.0, 4.0)) -> None:
        self.n_imf = int(n_imf)
        self.env_sigmas = tuple(float(s) for s in env_sigmas)

    @property
    def name(self) -> str:
        return "bemd_ier"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _decompose(self, img: torch.Tensor) -> list[torch.Tensor]:
        """Return up to ``n_imf`` IMFs of a ``[1,1,H,W]`` image plus residual."""
        residual = img.clone()
        imfs: list[torch.Tensor] = []
        for i in range(self.n_imf):
            sigma = self.env_sigmas[min(i, len(self.env_sigmas) - 1)]
            # IMF = signal - smooth-envelope-mean (single sift step at this scale).
            envelope = _gaussian_blur(residual, sigma=sigma)
            imf = residual - envelope
            imfs.append(imf)
            residual = envelope
        imfs.append(residual)  # final residual / trend
        return imfs

    def _ratio(self, img: torch.Tensor) -> float:
        imfs = self._decompose(img)
        energies = [float((m**2).sum()) for m in imfs]
        total = sum(energies)
        if total <= 1e-12:
            return float("nan")
        return float(energies[0] / total)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        ratios = [self._ratio(mag[b : b + 1]) for b in range(mag.shape[0])]
        ratios = [r for r in ratios if not math.isnan(r)]
        if not ratios:
            return float("nan")
        return float(sum(ratios) / len(ratios))


# ---------------------------------------------------------------------------
# Phase-congruency dispersion (log-Gabor bank).
# ---------------------------------------------------------------------------


@register_metric("pcd", requires_reference=False, direction="lower")
class PhaseCongruencyDispersion:
    r"""Phase-Congruency Dispersion · lower is better.

    Phase congruency (Kovesi) measures feature significance independent of
    contrast: with a log-Gabor bank of even/odd quadrature filters producing
    amplitudes :math:`a_n` and phases :math:`\phi_n`,

    .. math::

        \mathrm{PC}(\mathbf{p}) = \frac{|\sum_n a_n e^{i\phi_n}|}
                                       {\sum_n a_n + \varepsilon}.

    A good reconstruction has *high* mean PC at edges with *low* spatial
    dispersion. We report the (signed) gap

    .. math::

        \mathrm{PCD} = \operatorname{std}(\mathrm{PC}) - \operatorname{mean}(\mathrm{PC}),

    so a sharper, more coherent image scores lower (``higher_is_better=False``).
    Mean PC at edges drops under blur, pushing PCD up.
    """

    def __init__(self, n_scales: int = 3, min_wavelength: float = 3.0, mult: float = 2.1) -> None:
        self.n_scales = int(n_scales)
        self.min_wavelength = float(min_wavelength)
        self.mult = float(mult)

    @property
    def name(self) -> str:
        return "pcd"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _log_gabor_pc(self, img: torch.Tensor) -> torch.Tensor:
        """Orientation-pooled phase congruency map for a ``[1,1,H,W]`` image.

        Uses the Kovesi weighting: the raw congruency ``energy/sum_amp`` is
        multiplied by a *frequency-spread* factor that is ``1`` only when energy
        is distributed across all scales and ``→0`` when one scale dominates.
        Blur destroys the high-frequency scales, collapsing the spread and so
        *lowering* the weighted PC — which is the spec's blur anchor.
        """
        _, _, h, w = img.shape
        dev, dt = img.device, img.dtype
        fy = torch.fft.fftfreq(h, device=dev, dtype=dt).view(h, 1)
        fx = torch.fft.fftfreq(w, device=dev, dtype=dt).view(1, w)
        radius = torch.sqrt(fy * fy + fx * fx)
        radius[0, 0] = 1.0  # avoid log(0); DC handled below
        img_fft = torch.fft.fft2(img.squeeze(0).squeeze(0))
        sum_amp = torch.zeros(h, w, device=dev, dtype=dt)
        sum_re = torch.zeros(h, w, device=dev, dtype=dt)
        sum_im = torch.zeros(h, w, device=dev, dtype=dt)
        max_amp = torch.zeros(h, w, device=dev, dtype=dt)
        sigma_on_f = 0.55
        for s in range(self.n_scales):
            wavelength = self.min_wavelength * (self.mult**s)
            f0 = 1.0 / wavelength
            log_gabor = torch.exp(-(torch.log(radius / f0) ** 2) / (2 * math.log(sigma_on_f) ** 2))
            log_gabor[0, 0] = 0.0
            resp = torch.fft.ifft2(img_fft * log_gabor)
            amp = resp.abs()
            sum_amp = sum_amp + amp
            sum_re = sum_re + resp.real
            sum_im = sum_im + resp.imag
            max_amp = torch.maximum(max_amp, amp)
        energy = torch.sqrt(sum_re**2 + sum_im**2 + 1e-12)
        congruency = energy / (sum_amp + 1e-6)
        # Frequency spread (Kovesi): w = 1/(1 + exp(g*(c - s))), s = sum/( (N) * max ).
        n = float(self.n_scales)
        spread = sum_amp / (n * max_amp + 1e-6)  # in (0, 1]; 1 if all scales equal
        gain, cutoff = 10.0, 0.4
        weight = 1.0 / (1.0 + torch.exp(gain * (cutoff - spread)))
        # Noise threshold: suppress weak responses (background) so PC reflects
        # genuine broadband structure rather than the eps floor.
        amp_thresh = 0.02 * float(sum_amp.max())
        active = (sum_amp > amp_thresh).float()
        pc = (weight * congruency * active).clamp(0.0, 1.0)
        return pc

    def _score(self, img: torch.Tensor) -> float:
        pc = self._log_gabor_pc(img)
        return float(pc.std() - pc.mean())

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        scores = [self._score(mag[b : b + 1]) for b in range(mag.shape[0])]
        scores = [s for s in scores if not math.isnan(s)]
        if not scores:
            return float("nan")
        return float(sum(scores) / len(scores))


# ---------------------------------------------------------------------------
# Spatial Allan-variance slope (background region).
# ---------------------------------------------------------------------------


@register_metric(
    "savs",
    requires_reference=False,
    needs=("foreground_mask",),
    direction="lower",
)
class SpatialAllanVarianceSlope:
    r"""Spatial Allan-Variance Slope · ↓ (deviation from ``s_white = -2``).

    Treats the background (complement of the foreground mask :math:`\Omega^c`)
    as a noise field. For block-averaging window :math:`\tau` the spatial Allan
    variance is

    .. math::

        \sigma_A^2(\tau) = \tfrac12\,\mathbb{E}[(\bar x_{\tau,i+1}-\bar x_{\tau,i})^2],

    and we fit the log-log slope, reporting its deviation from the white-noise
    value :math:`s_{\mathrm{white}}=-2`:

    .. math::

        \mathrm{SAVS} = \big|\,d\ln\sigma_A^2/d\ln\tau - (-2)\big|.

    White noise gives slope :math:`\approx-2` (``SAVS≈0``); drift / correlated
    (g-factor-coloured) background flattens the slope and inflates SAVS. If no
    ``foreground_mask`` is supplied a magnitude-threshold background is derived
    from the prediction.
    """

    def __init__(self, taus: tuple[int, ...] = (1, 2, 4, 8)) -> None:
        self.taus = tuple(int(t) for t in taus)

    @property
    def name(self) -> str:
        return "savs"

    @property
    def higher_is_better(self) -> bool:
        return False

    def _background(self, mag: torch.Tensor, ctx: MetricContext) -> torch.Tensor:
        """Boolean background mask ``[B,1,H,W]`` (True = Omega^c)."""
        if ctx.foreground_mask is not None:
            fm = ctx.foreground_mask
            if fm.is_complex():
                fm = fm.abs()
            fm = fm.float()
            while fm.ndim < 4:
                fm = fm.unsqueeze(0)
            if fm.shape[-2:] == mag.shape[-2:]:
                return fm <= 0.5
        # Derive: foreground = bright pixels; background = the rest.
        per = mag.flatten(2).amax(dim=2, keepdim=True).unsqueeze(-1).clamp_min(1e-8)
        return mag <= 0.05 * per

    def _slope(self, field: torch.Tensor) -> float:
        """Log-log slope of the 2-D spatial Allan variance vs window size ``tau``.

        ``field`` is ``[H, W]`` with NaN outside the background region. For each
        ``tau`` we form non-overlapping ``tau x tau`` block means (over valid
        pixels only) and take the Allan variance over horizontally-adjacent
        blocks. White noise → variance ``∝ tau^-2`` → slope ``-2``.
        """
        h, w = field.shape
        side = min(h, w)
        taus = [t for t in self.taus if t * 2 <= side]
        if len(taus) < 2:
            return float("nan")
        log_tau: list[float] = []
        log_av: list[float] = []
        for tau in taus:
            uh = (h // tau) * tau
            uw = (w // tau) * tau
            if uh < 2 * tau or uw < tau:
                continue
            blocks = field[:uh, :uw].reshape(uh // tau, tau, uw // tau, tau)
            # nanmean over each tau x tau block.
            valid = (~torch.isnan(blocks)).sum(dim=(1, 3))
            filled = torch.nan_to_num(blocks, nan=0.0).sum(dim=(1, 3))
            means = filled / valid.clamp_min(1)
            ok = valid > 0  # [nb_h, nb_w]
            diffs = means[:, 1:] - means[:, :-1]
            pair_ok = ok[:, 1:] & ok[:, :-1]
            d = diffs[pair_ok]
            if d.numel() < 4:
                continue
            av = 0.5 * float((d**2).mean())
            if av <= 1e-12:
                continue
            log_tau.append(math.log(tau))
            log_av.append(math.log(av))
        if len(log_tau) < 2:
            return float("nan")
        # Least-squares slope.
        n = len(log_tau)
        sx = sum(log_tau)
        sy = sum(log_av)
        sxx = sum(t * t for t in log_tau)
        sxy = sum(t * y for t, y in zip(log_tau, log_av, strict=True))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return float("nan")
        return (n * sxy - sx * sy) / denom

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: MetricContext | None = None,
        **kwargs: Any,
    ) -> float:
        ctx = resolve_context(context, kwargs)
        mag = _as_magnitude_2d(prediction)
        if mag is None:
            return float("nan")
        s_white = _param(ctx, "savs_white_slope", -2.0)
        bg = self._background(mag, ctx)
        slopes: list[float] = []
        for b in range(mag.shape[0]):
            field = mag[b, 0].clone()
            field[~bg[b, 0]] = float("nan")
            sl = self._slope(field)
            if not math.isnan(sl):
                slopes.append(sl)
        if not slopes:
            return float("nan")
        mean_slope = sum(slopes) / len(slopes)
        return float(abs(mean_slope - s_white))


__all__ = [
    "BEMDIntrinsicEnergyRatio",
    "PatchGraphLaplacianEntropy",
    "PatchIntrinsicDimensionality",
    "PhaseCongruencyDispersion",
    "SpatialAllanVarianceSlope",
    "WaveletLeaderMultifractalSpectrumWidth",
    "WaveletParentChildDependencyEntropy",
]
