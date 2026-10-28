"""Ultra-low-field physics-noise MAP via half-quadratic splitting (B-2.1).

ULF→HF restoration framed as the MAP estimate

    x_hat = argmin_x  w/2 ‖y − A x‖²  +  R_θ(x),   w = 1/σ²,

solved by half-quadratic splitting / ADMM. Two physical ingredients make this a
genuine inverse problem rather than a denoiser:

* **Forward operator ``A`` = a low-pass** (the ULF effective-resolution loss):
  ``A x = ifft2c(H · fft2c(x))`` with a Gaussian OTF ``H`` whose cutoff comes from
  ``training.ulf_lowpass`` (a constant, or derived per contrast from the voxel sizes;
  see :mod:`.ulf_lowpass_operator`). The data term ``‖y - A x‖²`` therefore
  constrains the *recoverable low-frequency band*, and the learned prior fills the
  high-frequency band the low field cannot measure. (FFT round-trips go through the
  physics SSOT :func:`fft2c`/:func:`ifft2c`.)
* **Hoult-calibrated data weight** ``w = 1/σ²``, ``σ = sigma_floor/(B0/3)^{7/4}``
  (reusing :func:`_hoult_snr_scale`, the physics SSOT): a 0.1 T measurement is
  ~hundreds of times noisier than 3 T, so the data term is correctly weak at ULF
  and the prior dominates — the method does not over-trust a noise-dominated
  measurement. This SNR-calibrated balance is the contribution over an
  unconstrained generator. The ``hoult`` weight is ~6e-4 of rho at 64 mT (the data term
  is inert there), so ``data_weight_mode: normalised`` anchors ``w`` to rho at a declared
  calibration field instead (:func:`ulf_data_weight_normalised`).

The prior ``R_θ`` is a **learned restoration prior**: the spectral-norm-bounded
generator inherited from :class:`PnPStrategy` (``training.pnp``) as the prox. (We
do not claim strict Romano-Elad-Milanfar RED convergence — the generator is a
learned restoration network, not a Gaussian denoiser.)

The pure helpers (:func:`ulf_sigma`, :func:`ulf_degrade`, :func:`ulf_map_solve`)
hold the testable math; the strategy wires them to the trained generator and runs
the solve at validation so the physics mechanism is exercised, not a facade.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.ulf_lowpass import (  # noqa: F401  (re-exported)
    _fft2c_dc_index,
    _lowpass_otf,
    lowpass_dc_prox,
    ulf_degrade,
)
from spectramr.models.generators.pmps_corruption_operators import _hoult_snr_scale

from .pnp_strategy import PnPStrategy

logger = logging.getLogger(__name__)


def ulf_data_weight_normalised(
    source_field: float, calibration_field: float, ratio: float, rho: float
) -> float:
    """Data weight anchored to the penalty: ``w = ratio * rho * (B_src / B_cal)^{7/2}``.

    Since ``sigma ∝ B0^{-7/4}``, ``1/sigma^2 ∝ B0^{7/2}``: the Hoult law fixes only how
    ``w`` varies with field, while ``ratio`` (w/rho at the calibration field) sets the
    absolute scale, so the data term can bind at 64 mT.
    """
    return float(ratio) * float(rho) * (float(source_field) / float(calibration_field)) ** 3.5


def ulf_sigma(source_field: float, sigma_floor: float) -> float:
    """Hoult ULF noise std: ``sigma_floor / (source_field/3)^{7/4}``.

    Lower field ⇒ smaller Hoult scale ⇒ larger σ (noisier measurement).
    """
    return float(sigma_floor) / max(_hoult_snr_scale(float(source_field)), 1e-6)


def ulf_map_solve(
    y: torch.Tensor,
    denoiser: Callable[[torch.Tensor], torch.Tensor],
    *,
    sigma: float,
    rho: float,
    iterations: int,
    blur_cutoff: float = 0.5,
    data_weight_max: float = 1.0e3,
    data_weight: float | None = None,
    h_otf: torch.Tensor | None = None,
) -> torch.Tensor:
    """Half-quadratic-splitting MAP solve with a field-dependent low-pass ``A``.

    Alternates a closed-form Fourier data-consistency prox (for
    ``A = ifft2c ∘ H ∘ fft2c``) and the learned-restoration prox. The closed form
    of ``argmin_x w/2‖y−Ax‖² + ρ/2‖x−(z−u)‖²`` in the (unitary) fft2c domain is

        X = (w·H·Y + ρ·F(z−u)) / (w·H² + ρ),   x = ifft2c(X).real

    so the data term constrains the passband (``H≈1``) and the prior fills the
    stopband (``H≈0``). The data weight ``w = min(1/σ², data_weight_max)`` is the
    Hoult-calibrated precision: tiny at ULF (prior dominates), large at high field.

    Args:
        y: the ULF measurement image ``[B, 1, H, W]`` (real magnitude).
        denoiser: prox-on-prior callable ``x -> D(x)`` (same shape).
        sigma: ULF noise std (see :func:`ulf_sigma`).
        rho: ADMM penalty (``training.pnp.rho``).
        iterations: ADMM iterations (``training.pnp.iterations``).
        blur_cutoff: low-pass cutoff of ``A`` (smaller ⇒ more ULF resolution loss).
        data_weight_max: clamp on ``1/σ²``.
        data_weight: explicit ``w`` (normalised mode); overrides ``sigma`` when given.
        h_otf: precomputed OTF ``[1|B, 1, H, W]``; overrides ``blur_cutoff`` when given.
    """
    if data_weight is None:
        w = min(1.0 / max(sigma**2, 1e-12), float(data_weight_max))
    else:
        w = min(float(data_weight), float(data_weight_max))
    if h_otf is None:
        h_otf = _lowpass_otf(y.shape[-2], y.shape[-1], blur_cutoff, y.device, torch.float32)
    y_k = fft2c(y)
    x = y.clone()
    z = x.clone()
    u = torch.zeros_like(x)
    for _ in range(int(iterations)):
        # x-update: closed-form Fourier data prox for the low-pass A (H real).
        x = lowpass_dc_prox(y_k, z - u, h_otf, w, rho)
        # z-update: learned-restoration prox on the prior.
        z = _denoiser_tensor(denoiser, x + u)
        # dual update.
        u = u + (x - z)
    return z


def _denoiser_tensor(denoiser: Callable[[torch.Tensor], Any], x: torch.Tensor) -> torch.Tensor:
    out = denoiser(x)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        out = out.get("reconstruction", next(iter(out.values())))
    return out


def resolve_ulf_map_config(training: Any) -> Any:
    """Return ``training.ulf_map``; raise when the block is absent (no silent defaults)."""
    cfg = getattr(training, "ulf_map", None)
    if cfg is None:
        raise ValueError(
            "strategy_class 'ulf_map' requires a training.ulf_map block (source_field, "
            "sigma_floor, ...); none is declared."
        )
    return cfg


class UlfMapStrategy(PnPStrategy):
    """ULF→HF physics-noise MAP (B-2.1): PnP denoiser + Hoult-weighted data term."""

    # parallel_compat: Adds no parameters to PnP's (the low-pass operator is a plain object);
    # PnP's _denoiser declaration is inherited. A permission is not, so it is restated here.
    parallel_compat = frozenset({"replicated"})

    def _setup_strategy_specific_components(self) -> None:
        # ulf_map reuses the reconstruction/PnP plumbing; accept its mode
        # explicitly (the parent only accepts reconstruction-family modes).
        self._verify_strategy_config(expected_modes=("ulf_map", "reconstruction"))
        if getattr(self, "logging_service", None):
            self._log_config_features(self.logging_service)
        self._ulf_cfg = resolve_ulf_map_config(self.config.training)
        from .ulf_lowpass_operator import build_ulf_lowpass_operator  # cycle: it imports the OTF

        self._ulf_lowpass = build_ulf_lowpass_operator(self.config)
        logger.info("ulf_map forward operator: %s", self._ulf_lowpass.describe())

    def _ulf_sigma(self) -> float:
        cfg = self._ulf_cfg
        return ulf_sigma(float(cfg.source_field), float(cfg.sigma_floor))

    @torch.no_grad()
    def run_ulf_map(self, y: torch.Tensor, contrast_id: torch.Tensor | None = None) -> torch.Tensor:
        """Solve the ULF→HF physics-noise MAP for measurement ``y``.

        ``solve_mode='prior_only'`` returns the denoiser applied once (the single-pass
        control). Otherwise runs the HQS solve with the Hoult weight (``hoult``) or the
        normalised weight (``normalised``). PnP must be enabled: no silent pass-through.
        """
        denoiser = getattr(self, "_denoiser", None) or self._build_denoiser()
        pnp = getattr(self, "_pnp_cfg", None)
        if denoiser is None or pnp is None or not getattr(pnp, "enabled", False):
            raise ValueError("ulf_map requires training.pnp.enabled: true (the denoiser prox).")
        ulf = self._ulf_cfg
        if ulf.solve_mode == "prior_only":
            return _denoiser_tensor(denoiser, y)
        data_weight = None
        if ulf.data_weight_mode == "normalised":
            data_weight = ulf_data_weight_normalised(
                ulf.source_field, ulf.calibration_field, ulf.relative_data_weight, pnp.rho
            )
        return ulf_map_solve(
            y,
            denoiser,
            sigma=self._ulf_sigma(),
            rho=float(pnp.rho),
            iterations=int(pnp.iterations),
            data_weight_max=float(ulf.data_weight_max),
            data_weight=data_weight,
            h_otf=self._ulf_lowpass.otf(y.shape[-2], y.shape[-1], y.device, contrast_id),
        )

    def validation_step(
        self, input_batch: Any, target_batch: Any, contrast_id: Any = None, **kwargs: Any
    ) -> Any:
        """Fold the per-sample ``contrast_id`` into ``batch_context`` (gated seam)."""
        bc = kwargs.pop("batch_context", None) or {}
        if contrast_id is not None:
            bc["contrast_id"] = contrast_id
        return super().validation_step(input_batch, target_batch, batch_context=bc, **kwargs)

    def _validation_forward(
        self, input_batch: Any, batch_context: dict[str, Any], **_: Any
    ) -> torch.Tensor:
        """Validation prediction = the physics-noise MAP solve on the ULF input.

        Wires the headline mechanism (Hoult-weighted data term + denoiser prox)
        into the validated path so the image metrics score the actually-restored
        image (anti-facade: the σ(B0) likelihood is exercised, not inert).
        """
        x0 = batch_context.get("input", input_batch)
        return self.run_ulf_map(x0, batch_context.get("contrast_id"))
