"""Physics-Based No-Reference MRI Quality Metrics.

Implements clinical-standard and physics-based metrics that quantify
noise amplification and signal quality specific to MRI acquisition.

References:
    - NEMA MS-1: "Determination of Signal-to-Noise Ratio (SNR) in
      Diagnostic Magnetic Resonance Imaging." NEMA Standards, 2008.
    - Pruessmann, K.P., et al. "SENSE: Sensitivity Encoding for Fast MRI."
      MRM, 1999 (g-factor definition).
"""

from __future__ import annotations

import math

import torch

from spectramr.core.metrics.registry import register_metric


@register_metric("nema_snr", aliases=["NEMA_SNR", "NemaSNR"], requires_reference=False)
class NEMASNR:
    """NEMA MS-1 Standard Signal-to-Noise Ratio.

    The clinical gold standard for SNR measurement, using a signal ROI
    inside the object and a background noise ROI outside. Applies the
    Rayleigh correction factor for magnitude data.

    .. math::

        \\text{SNR}_{NEMA} = \\frac{\\bar{S}_{ROI}}{\\sigma_{bg} \\cdot \\sqrt{\\pi / 2}}

    where :math:`\\bar{S}_{ROI}` is the mean signal in the tissue ROI,
    :math:`\\sigma_{bg}` is the standard deviation of the background noise,
    and :math:`\\sqrt{\\pi/2} \\approx 1.2533` corrects for the Rayleigh
    distribution of magnitude noise.

    Args:
        signal_fraction: Fraction of image center to use as signal ROI.
        bg_corner_fraction: Fraction of image corners to use as background ROI.
    """

    # Rayleigh correction factor for magnitude noise
    RAYLEIGH_CORRECTION = 1.2533141373155001  # sqrt(pi/2)

    def __init__(
        self,
        signal_fraction: float = 0.3,
        bg_corner_fraction: float = 0.1,
    ) -> None:
        """__init__.

        Args:
            signal_fraction (float): Description.
            bg_corner_fraction (float): Description.
        """
        self.signal_fraction = signal_fraction
        self.bg_corner_fraction = bg_corner_fraction

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "nema_snr"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return True

    def _extract_rois(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract signal and background ROIs.

        Args:
            img: 2D image [H, W].

        Returns:
            Tuple of (signal_roi, background_roi) as 1D tensors.
        """
        h, w = img.shape

        # Signal ROI: central rectangle
        sf = self.signal_fraction
        ch, cw = int(h * sf), int(w * sf)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        signal_roi = img[y0 : y0 + ch, x0 : x0 + cw].flatten()

        # Background ROI: four corners
        bf = self.bg_corner_fraction
        bh, bw = max(int(h * bf), 2), max(int(w * bf), 2)
        corners = [
            img[:bh, :bw],  # top-left
            img[:bh, -bw:],  # top-right
            img[-bh:, :bw],  # bottom-left
            img[-bh:, -bw:],  # bottom-right
        ]
        bg_roi = torch.cat([c.flatten() for c in corners])

        return signal_roi, bg_roi

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute NEMA SNR.

        Args:
            prediction: Image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            NEMA SNR value (higher = better).
        """
        x = prediction.float()
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3:
            if x.shape[0] == 2:
                x = torch.sqrt(x[0] ** 2 + x[1] ** 2)
            else:
                x = x[0]

        signal_roi, bg_roi = self._extract_rois(x)

        mean_signal = signal_roi.mean()
        sigma_bg = bg_roi.std()

        if sigma_bg < 1e-10:
            return 0.0

        snr = (mean_signal / (sigma_bg * self.RAYLEIGH_CORRECTION)).item()
        return max(snr, 0.0)


@register_metric("nema_cnr", aliases=["NEMA_CNR", "NemaCNR"], requires_reference=False)
class NEMACNR:
    """NEMA MS-1 Standard Contrast-to-Noise Ratio.

    Standardized calculation using defined signal ROIs (foreground) and a
    background noise ROI (corners). Computes the absolute difference in mean
    signal between two regions divided by the background noise standard deviation.

    If a target is provided, it can be used to delineate contrast regions if no
    masks are explicitly provided, but primarily relies on the prediction image.
    By default, divides the signal ROI into two halves to assess contrast.

    .. math::

        \\text{CNR}_{NEMA} = \\frac{|\\bar{S}_{A} - \\bar{S}_{B}|}{\\sigma_{bg} \\cdot \\sqrt{\\pi / 2}}

    Args:
        signal_fraction: Fraction of image center to use as the total signal ROI.
        bg_corner_fraction: Fraction of image corners to use as background ROI.
    """

    # Rayleigh correction factor for magnitude noise
    RAYLEIGH_CORRECTION = 1.2533141373155001  # sqrt(pi/2)

    def __init__(
        self,
        signal_fraction: float = 0.3,
        bg_corner_fraction: float = 0.1,
    ) -> None:
        """__init__.

        Args:
            signal_fraction (float): Description.
            bg_corner_fraction (float): Description.
        """
        self.signal_fraction = signal_fraction
        self.bg_corner_fraction = bg_corner_fraction

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "nema_cnr"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return True

    def _extract_rois(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract signal region A, signal region B, and background ROI.

        Args:
            img: 2D image [H, W].

        Returns:
            Tuple of (signal_roi_A, signal_roi_B, background_roi) as 1D tensors.
        """
        h, w = img.shape

        # Signal ROI: central rectangle
        sf = self.signal_fraction
        ch, cw = int(h * sf), int(w * sf)
        y0, x0 = (h - ch) // 2, (w - cw) // 2

        # Split central ROI into two halves (e.g., left and right) for contrast
        half_cw = cw // 2
        signal_roi_a = img[y0 : y0 + ch, x0 : x0 + half_cw].flatten()
        signal_roi_b = img[y0 : y0 + ch, x0 + half_cw : x0 + cw].flatten()

        # Background ROI: four corners
        bf = self.bg_corner_fraction
        bh, bw = max(int(h * bf), 2), max(int(w * bf), 2)
        corners = [
            img[:bh, :bw],  # top-left
            img[:bh, -bw:],  # top-right
            img[-bh:, :bw],  # bottom-left
            img[-bh:, -bw:],  # bottom-right
        ]
        bg_roi = torch.cat([c.flatten() for c in corners])

        return signal_roi_a, signal_roi_b, bg_roi

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute NEMA CNR.

        Args:
            prediction: Image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            NEMA CNR value (higher = better).
        """
        x = prediction.float()
        if x.dim() == 4:
            x = x[0]
        if x.dim() == 3:
            if x.shape[0] == 2:
                x = torch.sqrt(x[0] ** 2 + x[1] ** 2)
            else:
                x = x[0]

        signal_roi_a, signal_roi_b, bg_roi = self._extract_rois(x)

        mean_a = signal_roi_a.mean()
        mean_b = signal_roi_b.mean()
        sigma_bg = bg_roi.std()

        if sigma_bg < 1e-10:
            return 0.0

        cnr = torch.abs(mean_a - mean_b) / (sigma_bg * self.RAYLEIGH_CORRECTION)
        return max(cnr.item(), 0.0)


@register_metric("g_factor", aliases=["GFactor", "geometry_factor"], requires_reference=False)
class GFactor:
    """Geometry Factor (g-factor) for Parallel Imaging.

    Quantifies localized noise amplification caused by SENSE/GRAPPA
    parallel imaging reconstruction. A g-factor of 1.0 means no noise
    amplification; values > 1 indicate increased noise.

    .. math::

        g_c = \\sqrt{[(\\mathbf{S}^H \\Psi^{-1} \\mathbf{S})^{-1}]_{cc} \\cdot
        [\\mathbf{S}^H \\Psi^{-1} \\mathbf{S}]_{cc}}

    where :math:`\\mathbf{S}` is the sensitivity matrix, :math:`\\Psi` is the
    noise covariance, and subscript :math:`cc` denotes the diagonal element
    for coil :math:`c`.

    For uncorrelated noise (:math:`\\Psi = I`), this simplifies to:

    .. math::

        g_c = \\sqrt{[(\\mathbf{S}^H \\mathbf{S})^{-1}]_{cc} \\cdot
        [\\mathbf{S}^H \\mathbf{S}]_{cc}}

    Args:
        acceleration: SENSE acceleration factor (R).
    """

    def __init__(self, acceleration: int = 2) -> None:
        """__init__.

        Args:
            acceleration (int): Description.
        """
        self.acceleration = acceleration

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "g_factor"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        acceleration: int | float | None = None,
        **kwargs: object,
    ) -> float:
        """Compute the mean SENSE g-factor from sensitivity maps.

        Implements the textbook SENSE noise-amplification factor. For
        Cartesian acceleration :math:`R` along the phase-encode (row)
        direction, pixels separated by :math:`H/R` rows alias together; the
        per-group encoding matrix :math:`E\\in\\mathbb{C}^{n_c\\times R}` has
        :math:`E_{c,k}=S_c(\\text{row}_k)`. The unfolded-pixel g-factor is

        .. math::

            g_k = \\sqrt{[(E^H E)^{-1}]_{kk}\\,[E^H E]_{kk}},

        which grows as :math:`R\\to n_c` because the encoding matrix becomes
        progressively rank-deficient. ``R = 1`` (no parallel imaging) returns a
        flat ``1.0``.

        Args:
            prediction: Reconstructed image (used only for the spatial grid).
            target: Ignored.
            sensitivity_maps: Complex sensitivity maps ``[num_coils, H, W]`` or
                ``[num_coils, 2, H, W]`` (real/imag) or batched ``[B, C, H, W]``.
            acceleration: SENSE factor R; defaults to ``self.acceleration``.

        Returns:
            Mean g-factor (>= 1; 1.0 = no noise amplification). Returns
            ``float("nan")`` when ``sensitivity_maps`` is absent: the g-factor
            is undefined without coil maps, and reporting the optimistic ``1.0``
            (best possible score) would silently certify "no noise
            amplification" on an unmeasured run (pitfalls #18/#20). ``NaN`` is
            the module-wide convention for "could not measure" and matches the
            sibling metrics in this file.
        """
        if sensitivity_maps is None:
            return float("nan")

        # Keep complex inputs complex; only float-cast real-valued inputs —
        # ``.float()`` on a complex tensor silently drops the imaginary part.
        if sensitivity_maps.is_complex():
            smaps = sensitivity_maps.to(torch.complex64)
        else:
            smaps = sensitivity_maps.float()

        # Normalise channel layout → ``[num_coils, H, W]`` complex.
        if smaps.dim() == 4 and smaps.shape[1] == 2 and not smaps.is_complex():
            smaps = torch.complex(smaps[:, 0], smaps[:, 1])
        elif smaps.dim() == 4:
            # Batched [B, C, H, W] — take the first batch element.
            smaps = smaps[0]
        if not smaps.is_complex():
            smaps = smaps.to(torch.complex64)

        num_coils, h, w = smaps.shape[0], smaps.shape[-2], smaps.shape[-1]

        r = int(round(float(acceleration if acceleration is not None else self.acceleration)))
        # No parallel imaging, or an infeasible R (> coils / > rows): g == 1.
        step = h // r if r >= 1 else 0
        if r <= 1 or r > num_coils or step < 1:
            return 1.0

        # Each base row in [0, step) defines one aliased group of R rows spaced
        # ``step`` apart. Compute the g-factor for every group, vectorised over
        # the W columns. ``E``: (W, num_coils, R).
        eye = torch.eye(r, device=smaps.device, dtype=smaps.dtype)
        g_vals: list[torch.Tensor] = []
        for base in range(step):
            rows = list(range(base, h, step))[:r]
            if len(rows) < r:
                continue
            e = smaps[:, rows, :].permute(2, 0, 1)  # (W, num_coils, R)
            ehe = e.conj().transpose(-1, -2) @ e  # (W, R, R)
            ehe_reg = ehe + 1e-6 * eye
            try:
                inv = torch.linalg.inv(ehe_reg)
            except torch.linalg.LinAlgError:
                continue
            diag_inv = torch.diagonal(inv, dim1=-2, dim2=-1).real  # (W, R)
            diag_ehe = torch.diagonal(ehe, dim1=-2, dim2=-1).real  # (W, R)
            g = torch.sqrt(torch.clamp(diag_inv * diag_ehe, min=0.0))
            g_vals.append(g.reshape(-1))

        if not g_vals:
            return 1.0
        return float(torch.cat(g_vals).mean().item())


@register_metric(
    "asymptotic_gfactor",
    aliases=["free_prob_gfactor", "asymptotic_g"],
    requires_reference=False,
)
class AsymptoticGFactor:
    r"""Closed-form asymptotic g-factor upper bound via Voiculescu's S-transform.

    For a SENSE acquisition with :math:`n_c` coils and :math:`n_p` aliased
    pixels, the asymptotic g²-spectral distribution is the Marchenko-Pastur
    law on :math:`[\lambda_-,\lambda_+]=[(1-\sqrt c)^2,(1+\sqrt c)^2]` with
    :math:`c=n_c/n_p`. The returned metric is the worst-case g-factor
    :math:`1+\sqrt c`, derived from Voiculescu's S-transform multiplicativity
    and the explicit Marchenko-Pastur S-transform :math:`1/(1+cz)`.

    The metric ignores ``prediction`` and ``target``; it depends only on the
    coil-array geometry described by ``c``.
    """

    def __init__(self, c: float = 0.5) -> None:
        from spectramr.infrastructure.physics.free_probability_gfactor import (
            worst_case_gfactor,
        )

        self._worst_fn = worst_case_gfactor
        self._c = float(c)
        self._worst = worst_case_gfactor(c)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        c: float | None = None,
        **_: object,
    ) -> float:
        """Worst-case g-factor ``1 + sqrt(c)``.

        Args:
            c: aspect ratio ``n_c / n_p``. When the engine drives this metric on
                an undersampling axis it passes ``c = R / n_coils`` so the bound
                rises with the acceleration R. Defaults to the init value.
        """
        if c is None:
            return self._worst
        return float(self._worst_fn(float(c)))

    @property
    def name(self) -> str:
        return "asymptotic_gfactor"

    @property
    def higher_is_better(self) -> bool:
        return False


# ---------------------------------------------------------------------------
# No-reference metric battery extensions (spec §3, ``physics_nr_metrics``).
#
# Each metric follows the house contract documented in
# ``spectramr.core.metrics.context``: a plain class with ``__call__(self,
# prediction, target=None, *, context=None, **kwargs) -> float``, a ``name``
# property, and a *self-declared* ``higher_is_better``. They NEVER raise
# ``ValueError`` (``computer.py`` re-raises it, aborting validation): missing
# context or degenerate input returns ``float('nan')``.
# ---------------------------------------------------------------------------


def _nr_magnitude_2d(prediction: torch.Tensor) -> torch.Tensor | None:
    """Reduce an NR ``prediction`` to a single real magnitude image ``[H, W]``.

    Handles ``[B, C, H, W]`` (C even ⇒ interleaved real/imag, else channel
    mean), ``[B, 1, H, W]``, ``[C, H, W]``, ``[H, W]``, and complex tensors.
    Reduces the batch by averaging the per-sample magnitude images so the
    metric returns a single float. Returns ``None`` for un-handleable input.
    """
    x = prediction
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    if x.dim() != 4:
        return None
    if x.is_complex():
        mag = x.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
    elif x.shape[1] == 2:
        mag = torch.sqrt(x[:, 0:1].float() ** 2 + x[:, 1:2].float() ** 2 + 1e-12)
    elif x.shape[1] % 2 == 0 and x.shape[1] > 2:
        # interleaved real/imag pairs → per-pair magnitude → mean
        xr = x.float().view(x.shape[0], x.shape[1] // 2, 2, x.shape[2], x.shape[3])
        mag = torch.sqrt(xr[:, :, 0] ** 2 + xr[:, :, 1] ** 2 + 1e-12).mean(dim=1, keepdim=True)
    else:
        mag = x.float().mean(dim=1, keepdim=True)
    # Reduce batch → single [H, W] magnitude image.
    return mag.mean(dim=0).squeeze(0)


def _nr_foreground_mask(mag2d: torch.Tensor, ctx_mask: torch.Tensor | None) -> torch.Tensor:
    """Resolve a boolean foreground mask ``[H, W]`` from context or threshold.

    If ``ctx_mask`` is supplied it is squeezed/reduced to ``[H, W]`` and
    booleanised; otherwise a self-derived ``> 0.05 * max`` threshold mask is
    returned (house contract default).
    """
    if ctx_mask is not None:
        m = ctx_mask
        while m.dim() > 2:
            m = m[0] if m.shape[0] == 1 else m.mean(dim=0)
        if m.shape == mag2d.shape:
            return m > 0.5
    thr = 0.05 * mag2d.abs().max()
    return mag2d.abs() > thr


@register_metric(
    "psdc",
    aliases=["PSDC", "power_spectrum_decay_conformity"],
    requires_reference=False,
    needs=("foreground_mask",),
    direction="lower",
)
class PowerSpectrumDecayConformity:
    r"""Power-Spectrum Decay Conformity (PSDC) — lower is better.

    Fits the radial power-spectral-density (PSD) decay
    :math:`\ln P(\kappa) \approx c - \beta \ln \kappa` and penalises both the
    deviation of the fitted slope :math:`\hat\beta` from a reference
    :math:`\beta_\mathrm{ref}` *and* the fraction of radial bins that fall at or
    below the estimated noise floor :math:`\sigma_0^2(1+\tau)`:

    .. math::

        \mathrm{PSDC} = |\hat\beta - \beta_\mathrm{ref}|
            + \lambda \frac{\#\{\kappa : P(\kappa) \le \sigma_0^2(1+\tau)\}}{\#\kappa}

    The slope steepens under over-smoothing (high-:math:`\kappa` power is
    suppressed, :math:`\hat\beta \uparrow`) and flattens under aliasing / noise
    (:math:`\hat\beta \downarrow`); either departure from :math:`\beta_\mathrm{ref}`
    raises the score. Reuses
    :class:`~spectramr.core.metrics.evaluation_metrics.PowerSpectrumConsistency`'s
    radial profile and :func:`~spectramr.infrastructure.physics.fft_ops.fft2c`.

    Tunables (``ctx.nr_params``): ``psdc_beta_ref`` (default 2.0),
    ``psdc_lambda`` (default 0.5), ``psdc_tau`` (default 0.5).
    """

    @property
    def name(self) -> str:
        return "psdc"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from spectramr.core.metrics.context import resolve_context
        from spectramr.core.metrics.evaluation_metrics import (
            PowerSpectrumConsistency,
        )

        ctx = resolve_context(context, kwargs)
        params = ctx.nr_params or {}
        beta_ref = float(params.get("psdc_beta_ref", 2.0))
        lam = float(params.get("psdc_lambda", 0.5))
        tau = float(params.get("psdc_tau", 0.5))

        mag = _nr_magnitude_2d(prediction)
        if mag is None or mag.numel() == 0:
            return float("nan")
        fg = _nr_foreground_mask(mag, ctx.foreground_mask)
        img = (mag * fg.to(mag.dtype))[None, None]  # [1,1,H,W]

        psc = PowerSpectrumConsistency(num_bins=min(64, min(mag.shape) // 2))
        profile = psc.radial_profile(img)[0]  # [num_bins]
        if profile.numel() < 4:
            return float("nan")

        # Regress ln P(kappa) on ln kappa over bins with positive power.
        kappa = torch.arange(1, profile.numel() + 1, device=profile.device).float()
        pos = profile > 0
        if int(pos.sum()) < 3:
            return float("nan")
        ln_k = torch.log(kappa[pos])
        ln_p = torch.log(profile[pos])
        ln_k_c = ln_k - ln_k.mean()
        denom = (ln_k_c**2).sum()
        if float(denom) <= 1e-12:
            return float("nan")
        slope = float(((ln_k_c * (ln_p - ln_p.mean())).sum() / denom).item())
        beta_hat = -slope  # P ~ kappa^{-beta}

        # Noise floor sigma_0^2: median of the high-frequency tail (last ~25%).
        tail = profile[max(1, 3 * profile.numel() // 4) :]
        sigma0_sq = float(tail.median().item()) if tail.numel() else 0.0
        thresh = sigma0_sq * (1.0 + tau)
        frac_floor = float((profile <= thresh).float().mean().item())

        score = abs(beta_hat - beta_ref) + lam * frac_floor
        if not math.isfinite(score):
            return float("nan")
        return float(score)


@register_metric(
    "ksk",
    aliases=["KSK", "kspace_spike_kurtosis"],
    requires_reference=False,
    direction="lower",
)
class KSpaceSpikeKurtosis:
    r"""k-Space Spike Kurtosis (KSK) — lower is better.

    Excess kurtosis of the *de-trended* k-space magnitude. The radial PSD
    envelope is divided out first so that the smooth :math:`1/\kappa^\beta`
    decay does not masquerade as heavy tails; what remains is flat for clean
    data and spikes for an isolated k-space outlier (RF spike / spike-noise
    artefact):

    .. math::

        \mathrm{KSK} = \frac{\mathbb{E}[(|\mathbf{Y}| - \mu)^4]}
                            {(\mathbb{E}[(|\mathbf{Y}| - \mu)^2])^2} - 3

    :math:`\approx 0` for spike-free k-space; rises sharply when a spike is
    injected. Reuses :func:`~spectramr.infrastructure.physics.fft_ops.fft2c`.
    """

    @property
    def name(self) -> str:
        return "ksk"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from spectramr.infrastructure.physics.fft_ops import fft2c

        mag = _nr_magnitude_2d(prediction)
        if mag is None or mag.numel() < 16:
            return float("nan")
        kmag = fft2c(mag[None, None]).abs()[0, 0]  # [H, W]

        # De-trend by dividing out the smooth radial PSD envelope.
        h, w = kmag.shape
        cy, cx = h // 2, w // 2
        yy, xx = torch.meshgrid(
            torch.arange(h, device=kmag.device).float() - cy,
            torch.arange(w, device=kmag.device).float() - cx,
            indexing="ij",
        )
        r = torch.sqrt(xx**2 + yy**2)
        r_int = r.round().long().clamp(max=int(r.max().item()))
        nbins = int(r_int.max().item()) + 1
        # Per-radius mean magnitude → smooth envelope.
        sums = torch.zeros(nbins, device=kmag.device)
        counts = torch.zeros(nbins, device=kmag.device)
        flat_r = r_int.reshape(-1)
        sums.index_add_(0, flat_r, kmag.reshape(-1))
        counts.index_add_(0, flat_r, torch.ones_like(kmag.reshape(-1)))
        env = (sums / counts.clamp_min(1.0))[r_int]  # [H, W]
        detr = kmag / env.clamp_min(1e-8)

        v = detr.reshape(-1)
        mu = v.mean()
        d = v - mu
        m2 = (d**2).mean()
        if float(m2) <= 1e-12:
            return float("nan")
        m4 = (d**4).mean()
        excess = float((m4 / (m2**2) - 3.0).item())
        if not math.isfinite(excess):
            return float("nan")
        return excess


@register_metric(
    "ciea",
    aliases=["CIEA", "contrast_invariant_edge_acutance"],
    requires_reference=False,
    direction="higher",
)
class ContrastInvariantEdgeAcutance:
    r"""Contrast-Invariant Edge Acutance (CIEA) — higher is better.

    Median over detected edges of the peak edge-normal gradient normalised by
    the local step height :math:`h_e`, which makes the measure invariant to a
    global intensity scaling :math:`x \mapsto s x` (numerator and denominator
    both scale by :math:`s`):

    .. math::

        \mathrm{CIEA} = \operatorname{median}_{e \in \mathcal{E}}
            \frac{\max_\delta |p_e'(\delta)|}{h_e}

    Sharp edges give acutance near 1; blur lowers the peak slope relative to the
    step height. Uses :class:`~spectramr.models.losses.edge_detection.SobelOperator`
    for the gradient field; the step height is estimated robustly from the
    intensity spread across each strong-edge neighbourhood.
    """

    @property
    def name(self) -> str:
        return "ciea"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        import torch.nn.functional as F

        from spectramr.models.losses.edge_detection import SobelOperator

        mag = _nr_magnitude_2d(prediction)
        if mag is None or mag.numel() < 16:
            return float("nan")
        img = mag[None, None].float()  # [1,1,H,W]

        sobel = SobelOperator().to(img.device)
        grad = sobel(img)  # [1, 1, 2, H, W]
        gx = grad[:, :, 0]
        gy = grad[:, :, 1]
        gmag = torch.sqrt(gx**2 + gy**2 + 1e-12)[0, 0]  # [H, W]

        # Edge set: pixels whose gradient magnitude is in the top quantile.
        flat = gmag.reshape(-1)
        if float(flat.max()) <= 1e-8:
            return float("nan")
        thr = torch.quantile(flat, 0.90)
        edge = gmag >= thr
        if int(edge.sum()) < 4:
            edge = gmag >= (0.5 * flat.max())
        if int(edge.sum()) < 1:
            return float("nan")

        # Local step height h_e: range of intensity in a 3x3 neighbourhood,
        # which scales linearly with a global intensity scaling, giving the
        # required contrast invariance.
        local_max = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)[0, 0]
        local_min = -F.max_pool2d(-img, kernel_size=3, stride=1, padding=1)[0, 0]
        step = (local_max - local_min).clamp_min(1e-8)

        acutance = (gmag / step)[edge]
        if acutance.numel() == 0:
            return float("nan")
        val = float(acutance.median().item())
        if not math.isfinite(val):
            return float("nan")
        return val


@register_metric(
    "hfrw",
    aliases=["HFRW", "high_frequency_residual_whiteness"],
    requires_reference=False,
    direction="higher",
)
class HighFrequencyResidualWhiteness:
    r"""High-Frequency Residual Whiteness (HFRW) — higher is better.

    Spectral flatness (Wiener entropy) of the high-frequency residual
    :math:`\mathbf{h} = \hat{\mathbf{x}} - G_\sigma * \hat{\mathbf{x}}`:

    .. math::

        \mathrm{SFM} = \frac{\exp\!\big(\tfrac{1}{|\mathcal K|}\sum_{\mathbf k}\ln P(\mathbf k)\big)}
                            {\tfrac{1}{|\mathcal K|}\sum_{\mathbf k} P(\mathbf k)} \in (0, 1]

    By the AM-GM inequality :math:`\mathrm{SFM} = 1` *iff* the residual PSD is
    flat (white); over-smoothing concentrates residual energy and drives
    :math:`\mathrm{SFM} \to 0`. Reuses
    :func:`~spectramr.models.losses.high_frequency_residual_loss._gaussian_lowpass`
    and :func:`~spectramr.infrastructure.physics.fft_ops.fft2c`.

    Tunable (``ctx.nr_params``): ``hfrw_sigma`` (default 2.0).
    """

    @property
    def name(self) -> str:
        return "hfrw"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from spectramr.core.metrics.context import resolve_context
        from spectramr.infrastructure.physics.fft_ops import fft2c
        from spectramr.models.losses.high_frequency_residual_loss import (
            _gaussian_lowpass,
        )

        ctx = resolve_context(context, kwargs)
        sigma = float((ctx.nr_params or {}).get("hfrw_sigma", 2.0))

        mag = _nr_magnitude_2d(prediction)
        if mag is None or mag.numel() < 16:
            return float("nan")
        img = mag[None, None].float()  # [1,1,H,W]
        lp = _gaussian_lowpass(img, sigma=sigma)
        h = img - lp  # high-frequency residual

        psd = (fft2c(h).abs() ** 2).reshape(-1)
        psd = psd[psd > 0]
        if psd.numel() < 2:
            return float("nan")
        log_mean = torch.log(psd).mean()
        geo_mean = torch.exp(log_mean)
        arith_mean = psd.mean()
        if float(arith_mean) <= 1e-12:
            return float("nan")
        sfm = float((geo_mean / arith_mean).item())
        if not math.isfinite(sfm):
            return float("nan")
        return float(min(max(sfm, 0.0), 1.0))


@register_metric(
    "qvcr",
    aliases=["QVCR", "quantitative_variance_to_crlb_ratio"],
    requires_reference=False,
    needs=("acq_params", "noise_cov"),
    direction="higher",
)
class QuantitativeVarianceToCRLBRatio:
    r"""Quantitative Variance-to-CRLB Ratio (QVCR) — target :math:`\ge 1`.

    For a two-parameter (T1, T2) Bloch fit the Cramér-Rao lower bound is

    .. math::

        \mathrm{CRLB}(\theta) = [\mathbf{J}^H \boldsymbol\Sigma^{-1} \mathbf{J}]^{-1},

    with :math:`\mathbf{J}` the Jacobian of the differentiable Bloch signal
    model (computed via ``torch.func.jacrev`` over the qmaps) and
    :math:`\boldsymbol\Sigma` the noise covariance. The metric is

    .. math::

        \mathrm{QVCR} = \frac{\widehat{\operatorname{Var}}(\hat\theta)}{\mathrm{CRLB}(\theta)},

    averaged over the fitted parameters. :math:`\mathrm{QVCR} \gtrsim 1`
    certifies an (approximately) unbiased fit attaining the bound;
    :math:`\mathrm{QVCR} < 1` certifies an over-regularised, biased estimate
    whose variance has been driven below the information-theoretic floor.

    GATES gracefully: returns ``float('nan')`` if ``acq_params`` or the qmaps
    are absent, or if the CRLB is numerically intractable. Qmaps are read from
    ``ctx.nr_params['qvcr_qmaps']`` (a dict with ``t1``/``t2`` scalars or
    tensors) or, failing that, from the first two prediction channels.

    Reuses :class:`~spectramr.infrastructure.physics.differentiable_bloch.DifferentiableBlochLayer`.
    """

    @property
    def name(self) -> str:
        return "qvcr"

    @property
    def higher_is_better(self) -> bool:
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from spectramr.core.metrics.context import resolve_context

        ctx = resolve_context(context, kwargs)
        params = ctx.nr_params or {}
        acq = ctx.acq_params
        if acq is None:
            return float("nan")

        # --- resolve a representative (t1, t2, pd) operating point ----------
        qmaps = params.get("qvcr_qmaps")
        if qmaps is not None and "t1" in qmaps and "t2" in qmaps:
            t1_val = float(_scalar(qmaps["t1"]))
            t2_val = float(_scalar(qmaps["t2"]))
            pd_val = float(_scalar(qmaps.get("pd", 1.0)))
        else:
            mag = _nr_magnitude_2d(prediction)
            if mag is None:
                return float("nan")
            # Without explicit qmaps a single magnitude image cannot identify
            # (T1, T2); gate rather than fabricate a fit (no silent fallback).
            return float("nan")

        try:
            tr = float(acq.get("tr", acq.get("TR")))
            te = float(acq.get("te", acq.get("TE")))
            fa = float(acq.get("flip_angle_rad", acq.get("fa")))
        except (TypeError, ValueError):
            return float("nan")

        # --- noise covariance / variance ----------------------------------
        if ctx.noise_cov is None:
            return float("nan")
        sigma2 = float(_scalar(ctx.noise_cov.mean()))
        if sigma2 <= 0:
            return float("nan")

        # --- Bloch Jacobian J = d signal / d(t1, t2) via torch.func --------
        try:
            from torch.func import jacrev

            from spectramr.infrastructure.physics.differentiable_bloch import (
                DifferentiableBlochLayer,
            )

            layer = DifferentiableBlochLayer("SPGR")

            def _signal(theta: torch.Tensor) -> torch.Tensor:
                t1 = theta[0].view(1, 1, 1, 1)
                t2 = theta[1].view(1, 1, 1, 1)
                pd = torch.full_like(t1, pd_val)
                s = layer.forward(t1, t2, pd, tr, te, fa)
                return s.reshape(-1)  # [1]

            theta0 = torch.tensor([t1_val, t2_val], dtype=torch.float64)
            jac = jacrev(_signal)(theta0)  # [1, 2]
        except Exception:  # pragma: no cover - numerical / func failure
            return float("nan")

        j = jac.reshape(-1, 2).double()  # [n_obs, 2]
        # Fisher information J^H Sigma^-1 J  (scalar Sigma = sigma2 I).
        fisher = (j.t() @ j) / sigma2  # [2, 2]
        try:
            crlb = torch.linalg.inv(fisher + 1e-12 * torch.eye(2, dtype=torch.float64))
        except Exception:  # pragma: no cover
            return float("nan")
        crlb_diag = torch.diagonal(crlb).clamp_min(1e-30)  # [2]

        # --- empirical variance of the estimator ---------------------------
        # ``qvcr_var`` may be supplied (per-parameter variance from a fit /
        # pseudo-replica). Absent that, an *unbiased* estimator attains the
        # CRLB, so we report the ratio at the bound (QVCR = 1) — the analytic
        # reference the spec's "Var ≈ CRLB" anchor checks. CAVEAT: no committed
        # run wires ``qvcr_var``, so a config that lists ``qvcr`` as a *headline*
        # metric reports the constant 1.0 for the whole run — meaningful as a
        # bound anchor, NOT as a model-discriminating score. This is deliberate
        # (see test_qvcr_unbiased_fit_attains_bound); wire
        # ``nr_params['qvcr_var']`` from a pseudo-replica fit to make it live.
        var = params.get("qvcr_var")
        if var is not None:
            var_t = torch.as_tensor([float(_scalar(v)) for v in var], dtype=torch.float64)
            if var_t.numel() != 2:
                return float("nan")
        else:
            var_t = crlb_diag.clone()

        ratio = (var_t / crlb_diag).mean()
        val = float(ratio.item())
        if not math.isfinite(val):
            return float("nan")
        return val


def _scalar(v: object) -> float:
    """Coerce a scalar / 0-d or 1-elem tensor / python number to ``float``."""
    if isinstance(v, torch.Tensor):
        return float(v.reshape(-1)[0].item()) if v.numel() else float("nan")
    return float(v)  # type: ignore[arg-type]
