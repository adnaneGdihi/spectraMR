"""Faithful multi-acquisition field-mapping strategy.

For the VF field-mapping arms (AFI, double-angle, dual-echo), this strategy
synthesises the genuine multi-acquisition stack each method is defined over —
via :class:`MultiAcquisitionSimulator` (Bloch forward model + Bottomley T1) —
feeds it to the arm's model, and supervises the model's field estimate against
the synthesiser's ground-truth field. This replaces the single-frame VF path,
under which the method algebra ran on coil/real-imag channels (see
``docs/digital_twin_review_followups.rst`` Pass 5/6).

All synthesis parameters come from ``config.physics.multi_acquisition`` (SSOT);
an unknown / unimplemented ``method`` raises at startup (no silent fallback).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch.nn.functional import interpolate, mse_loss

from spectramr.config.schemas.physics import subvoxel_input_channels
from spectramr.core.metrics.anchor_conformal import (
    AnchorConformalCalibrator,
    local_detail_score,
)
from spectramr.core.metrics.b0_field_rmse import B0FieldRMSE
from spectramr.core.metrics.quantitative.qmri_agreement import qmri_agreement_metrics
from spectramr.core.metrics.subject_conformal import SubjectResidualBank
from spectramr.core.metrics.super_nyquist_fidelity import SuperNyquistFidelity
from spectramr.infrastructure.physics.marker_realisation import realisation_seeds, wrong_seed
from spectramr.infrastructure.physics.multi_acquisition import (
    AcquisitionResult,
    MultiAcquisitionSimulator,
)
from spectramr.infrastructure.physics.psf_conditioning import otf_channel
from spectramr.infrastructure.physics.psf_estimation import (
    estimate_psf,
    gaussian_psf,
    psf_fwhm_map,
    psf_identifiability,
)
from spectramr.infrastructure.physics.relaxometric_calibration import (
    AcquisitionParams,
    measured_gain,
    relaxometric_gain,
    tissue_gain_map,
)
from spectramr.infrastructure.physics.subpixel_registration import (
    estimate_subpixel_shifts,
    fourier_shift,
)
from spectramr.infrastructure.physics.tissue_factor import (
    N_TISSUES,
    class_agreement,
    gain_vector,
    tissue_kappa_map,
)
from spectramr.infrastructure.training.strategies.base import BaseTrainingStrategy
from spectramr.infrastructure.training.strategies.mixins.utils import pick_present
from spectramr.models.capabilities import StrategyCapabilities

logger = logging.getLogger(__name__)

# Methods whose recovered field is an off-resonance B0 map (graded against the
# real ``b0_map``); every other method recovers a B1 transmit scale (``b1_map``).
_B0_METHODS = ("dual_echo", "bssfp_banding")
# Fixed seed for the validation dither: one checkpoint must score the same
# every time it is evaluated, or the selection metric is noise.
_VALIDATION_SEED = 0


class ConcreteMultiAcquisitionStrategy(BaseTrainingStrategy):
    """Supervise a field map against a faithfully-synthesised acquisition stack.

    Per step: synthesise ``(stack, field)`` from the clean image, predict the
    field with the arm model, and minimise ``MSE(pred_field, field)`` plus a
    light spatial-smoothness term (fields are smooth).
    """

    # Computes its own objective and reads no `losses:` block, so an empty builder stack
    # is the design (LossBuilder.validate) and any declared image loss is a decoy.
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = False

    #: Validation keys this strategy's own ``validation_step`` writes (cohort
    #: review 2026-09-02, T0.5); ``validation.scoring.compute`` may name them.
    capabilities = StrategyCapabilities(
        emitted_metrics=frozenset(
            {
                "val_b0_field_rmse",
                "val_field_abs_bias",
                "val_field_bias",
                "val_field_mae",
                "val_field_mse",
                "val_field_reference_real",
                "val_kappa_class_agreement",
                "val_kappa_error",
                "val_kappa_measured",
                "val_kappa_predicted",
                "val_psnr",
                "val_shift_mae_px",
            }
        )
    )

    def __init__(self, env: Any, device: torch.device | None = None, **kwargs: Any) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # No second ``self._setup_strategy_specific_components()``: the base __init__
        # already runs it, and it is not idempotent once it adds optimizer param groups.

    def _setup_strategy_specific_components(self) -> None:
        cfg = self.config.physics.multi_acquisition
        if not cfg.enabled or cfg.method is None:
            raise ValueError(
                "ConcreteMultiAcquisitionStrategy requires "
                "physics.multi_acquisition.enabled=True and a 'method'."
            )
        self._method = cfg.method
        # Config-driven (CLAUDE.md #15). Was hardcoded to 0.01 until 2026-07,
        # so an arm YAML could declare a smoothness weight nothing ever read.
        self._lambda_smooth = float(cfg.lambda_smooth)
        self._normalize_magnitude = bool(cfg.normalize_magnitude)
        reg = cfg.subvoxel_registration
        self._shift_source = reg.shift_source
        self._max_shift_px = float(reg.max_shift_px)
        self._marker_seed = int(reg.marker_seed)
        self._per_sample_marker = reg.marker_realisation == "per_sample"
        self._marker_epoch, self._marker_ordinal = -1, 0
        self._probe_target = cfg.band_probe.target_realisation
        self._input_stream = cfg.input_stream
        self._fov_mask = cfg.fov_mask
        # The fiducial is synthesised when something reads it: the registrar, or
        # the model itself (marker_channels, which is allowed under 'blind').
        marker_enabled = self._method == "subvoxel_sr" and (
            reg.shift_source == "recovered" or bool(cfg.marker_channels)
        )
        self.macq = MultiAcquisitionSimulator(
            cfg.method,
            field_strength_t=float(self.config.physics.field_strength),
            n_timepoints=cfg.n_timepoints,
            n_frames=cfg.n_frames,
            sr_scale=cfg.sr_scale,
            n_phase_cycles=getattr(cfg, "n_phase_cycles", 4),
            subvoxel_max_shift_px=reg.max_shift_px,
            marker_enabled=marker_enabled,
            marker_grid_spacing=reg.marker_grid_spacing,
            marker_sigma=reg.marker_sigma,
            marker_jitter=reg.marker_jitter,
            marker_seed=reg.marker_seed,
            # Physical-units fiducial. Unset on the synthetic M4Raw arm, where
            # the grid IS the resolution; required on real ULF->HF data, where
            # the volume sits on a 3T grid 3-7x finer than the 64mT scanner
            # resolved and a grid-sized marker would be invisible at ULF.
            # voxel_mm is the STORED grid, effective_voxel_mm what the scanner
            # resolved; sigma in pixels is their RATIO. Passing effective for
            # both (as this call site did until 2026-07-26) collapses the ratio
            # to 1 and yields a ONE-PIXEL marker on a grid 3-7x finer than the
            # acquisition -- the exact sub-resolution invisibility the physical
            # mode exists to prevent. The schema now requires both.
            marker_voxel_mm=reg.voxel_mm and tuple(reg.voxel_mm),
            marker_effective_voxel_mm=(
                tuple(reg.effective_voxel_mm) if reg.effective_voxel_mm else None
            ),
            marker_sigma_mm=tuple(reg.marker_sigma_mm) if reg.marker_sigma_mm else None,
            marker_spacing_mm=(tuple(reg.marker_spacing_mm) if reg.marker_spacing_mm else None),
            marker_kappa=reg.marker_kappa,
        ).to(self.device)
        # Registration accuracy, in HR pixels. Reported, never optimised (phase
        # correlation has no parameters), and it is what makes a null result
        # attributable: flat PSNR at 0.003 px error means the dither carries no
        # information; flat PSNR at 0.5 px error means the front end failed.
        self._last_shift_mae: torch.Tensor | None = None
        self._last_shifts: torch.Tensor | None = None
        self._last_band_loss: torch.Tensor | None = None
        self._last_hf_loss: torch.Tensor | None = None
        self._last_task_loss: torch.Tensor | None = None
        self._last_mse: torch.Tensor | None = None
        # Super-Nyquist probe. The passband is keyed to the decimation the
        # network actually inverts (sr_scale), NOT to the scanner's native
        # resolution: the simulator pools the stored volume, so that pooling IS
        # the acquisition here. A schema validator refuses an arm whose declared
        # effective/grid gap disagrees with sr_scale, so the two cannot drift.
        probe_cfg = cfg.band_probe
        self._band_probe_enabled = bool(probe_cfg.enabled)
        self._lambda_band = float(probe_cfg.lambda_band)
        self._lambda_anatomy = float(probe_cfg.lambda_anatomy)
        self._snf = (
            SuperNyquistFidelity(
                sr_scale=cfg.sr_scale,
                n_sub_bands=probe_cfg.n_sub_bands,
                n_super_bands=probe_cfg.n_super_bands,
                rho_max=probe_cfg.rho_max,
                min_bins=probe_cfg.min_bins,
            )
            if self._band_probe_enabled
            else None
        )
        self._last_band_spectrum: dict[str, float] = {}
        if self._band_probe_enabled:
            logger.info(
                "[MultiAcq] band_probe on: edges=%s super_bands=%s lambda_band=%.4g",
                tuple(round(e, 3) for e in self._snf.edges),
                self._snf.super_nyquist_bands,
                self._lambda_band,
            )
        # Fiducial-measured forward operator.
        psf = cfg.forward_psf
        self._psf_cfg = psf
        self._psf_enabled = bool(psf.enabled)
        self._last_psf: dict[str, float] = {}
        self._last_psf_channel: torch.Tensor | None = None
        self._assumed_otf: torch.Tensor | None = None
        self._assumed_otf_key: tuple[Any, ...] | None = None
        if self._psf_enabled:
            # Declared in HR pixels, compared and fed on the pooled grid, where
            # the measured kernel lives -- one unit system for both sources.
            assumed = gaussian_psf(psf.kernel_size, max(psf.sigma_px / cfg.sr_scale, 1e-3))
            self._assumed_lr_kernel = assumed.reshape(1, 1, 1, *assumed.shape).to(self.device)
            # The simulator applies true_sigma_px when declared, so the assumed
            # kernel is WRONG on purpose and the measured arm has something to
            # correct. With it unset the two agree by construction, which the
            # schema only permits as an explicit null control.
            true_sigma = psf.true_sigma_px or psf.sigma_px
            truth = gaussian_psf(psf.kernel_size, true_sigma)
            self.macq.set_forward_psf(truth.reshape(1, 1, 1, *truth.shape))
            logger.info(
                "[MultiAcq] forward_psf: assumed sigma=%.2f true sigma=%.2f source=%s grid=%dx%d",
                psf.sigma_px,
                true_sigma,
                psf.source,
                psf.control_rows,
                psf.control_cols,
            )

        # Fiducial-calibrated prediction intervals. Reporting only.
        conf = cfg.anchor_conformal
        self._conformal = (
            AnchorConformalCalibrator(
                alpha=conf.alpha, n_strata=conf.n_strata, tolerance=conf.tolerance
            )
            if conf.enabled
            else None
        )
        self._last_coverage: dict[str, float] = {}
        self._last_result: Any = None
        # Held-out-subject calibration (VFU-09): None in the marker mode.
        self._subject_bank = (
            SubjectResidualBank(conf.calibration_subject_count)
            if conf.enabled and conf.calibration_source == "held_out_subjects"
            else None
        )
        self._train_subjects: set[str] = set()

        # Relaxometric contrast transfer. kappa is computed ONCE from declared
        # acquisition parameters and declared marker relaxometry -- nothing here
        # is learned, which is what lets it be handed to the model.
        relax = cfg.relaxometric_calibration
        self._relax_enabled = bool(relax.enabled)
        self._relax_factored = bool(relax.factored)
        self._factor_form = relax.factor_form
        self._marker_kappa: float | None = None
        self._tissue_gains: torch.Tensor | None = None
        self._last_kappa_logits: torch.Tensor | None = None
        if self._relax_enabled:
            src = AcquisitionParams(
                relax.source.field_strength_t,
                relax.source.tr_ms,
                relax.source.te_ms,
                relax.source.flip_deg,
            )
            tgt = AcquisitionParams(
                relax.target.field_strength_t,
                relax.target.tr_ms,
                relax.target.te_ms,
                relax.target.flip_deg,
            )
            gains = {k.value: v for k, v in tissue_gain_map(src, tgt).items()}
            self._marker_kappa = relaxometric_gain(
                relax.marker_t1_ms,
                relax.marker_t1_target_ms or relax.marker_t1_ms,
                relax.marker_t2_ms,
                src,
                tgt,
            )
            self.macq.set_field_transfer(gains, self._marker_kappa)
            if self._relax_factored and self._factor_form == "tissue_map":
                self._tissue_gains = gain_vector(gains, device=self.device)
                expected_out = 1 + N_TISSUES
                declared_out = int(self.config.model.out_channels)
                if declared_out != expected_out:
                    raise ValueError(
                        f"factor_form='tissue_map' needs model.out_channels={expected_out} "
                        f"(1 structure channel + {N_TISSUES} tissue-class logits), got "
                        f"{declared_out}."
                    )
            logger.info(
                "[MultiAcq] relaxometric transfer %.3fT->%.3fT: kappa=%s "
                "marker_kappa=%.4f factored=%s factor_form=%s",
                src.field_strength_t,
                tgt.field_strength_t,
                {k: round(v, 4) for k, v in gains.items()},
                self._marker_kappa,
                self._relax_factored,
                self._factor_form,
            )
        self._last_kappa_measured: torch.Tensor | None = None

        # Feed the fiducial to the MODEL, not only to the registrar. Required
        # by any backbone whose attention keys on the instrument.
        self._marker_channels = bool(getattr(cfg, "marker_channels", False))
        if self._method == "subvoxel_sr":
            expected_in = subvoxel_input_channels(cfg)
            declared_in = int(self.config.model.in_channels)
            if declared_in != expected_in:
                extra = f" plus {cfg.n_frames} fiducial frames" if self._marker_channels else ""
                if self._psf_enabled:
                    extra += " plus 1 forward-PSF channel"
                raise ValueError(
                    f"subvoxel_sr feeds the model {cfg.n_frames} frames plus "
                    f"{2 * cfg.n_frames} shift-conditioning maps{extra} = "
                    f"{expected_in} channels, but model.in_channels={declared_in}. "
                    "Both the maps and the fiducial channels are present on every "
                    "rung of the ladder (zero-filled where the rung does not use "
                    "them), so all rungs share one architecture and one parameter "
                    "count."
                )
            logger.info(
                "[MultiAcq] subvoxel_sr shift_source=%s marker=%s in_channels=%d "
                "marker_realisation=%s probe_target=%s input_stream=%s fov_mask=%s",
                self._shift_source,
                marker_enabled,
                expected_in,
                reg.marker_realisation,
                self._probe_target,
                self._input_stream,
                self._fov_mask,
            )
        # Path A (oracle_bssfp): feed the dataset's real phase-cycled stack to the
        # model instead of synthesising one, grading vs the real b0_map.
        self._use_real_stack = bool(getattr(cfg, "use_real_stack", False))
        logger.info(
            "[MultiAcq] method=%s field_strength=%.2fT use_real_stack=%s",
            self._method,
            float(self.config.physics.field_strength),
            self._use_real_stack,
        )

    # ── helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _to_complex(batch: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(batch):
            return batch
        c = batch.shape[1]
        if c >= 2 and c % 2 == 0:
            return torch.complex(batch[:, 0::2], batch[:, 1::2])
        return batch.to(torch.complex64)

    @staticmethod
    def _unit_scale(mag: torch.Tensor) -> torch.Tensor:
        """Divide a magnitude batch by its per-sample 99.5th percentile.

        A robust reference, not ``amax``: a single hot voxel would otherwise set
        the scale and push the whole object toward zero. The tissue bulk lands in
        ``[0, 1]`` and the surviving tail sits just above 1.
        """
        b, c = mag.shape[0], mag.shape[1]
        ref = torch.quantile(mag.reshape(b, c, -1).float(), 0.995, dim=-1, keepdim=True)
        ref = ref.unsqueeze(-1).to(mag.dtype)
        # an all-zero (or near-degenerate) sample has no percentile to key off
        ref = torch.where(ref > 0, ref, mag.amax(dim=(-2, -1), keepdim=True))
        return mag / ref.clamp(min=1e-8)

    def _clean_magnitude(self, target_batch: torch.Tensor) -> torch.Tensor:
        """Single-channel magnitude (M0 proxy) ``[B, 1, H, W]`` from the target.

        The k-space target must keep its raw scale through the loader — clamping
        at a percentile would clip the DC peak — so the IFFT'd magnitude carries
        an arbitrary scanner scale. Normalising it HERE, at the single seam every
        method draws its ``pd`` from, conditions the whole chain at once: the
        supervision target *and* the synthesised acquisition stack both derive
        from this tensor, which is why the objective otherwise grows
        **quadratically** with that scale (2026-07 exp_vf_01: first-step gradient
        norm 6477 against ``gradient_clip_value: 1.0``).

        Scale-invariance of the reported metric is preserved:
        ``val_psnr = 10*log10(rng**2 / mse)`` divides out any common factor, so
        converged PSNR remains comparable across the change.
        """
        if target_batch.ndim == 5:  # [B, C, H, W, D] -> mid slice
            target_batch = target_batch[..., target_batch.shape[-1] // 2]
        x = self._to_complex(self._to_device(target_batch))
        if self.config.data.dataset_type == "kspace":
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            x = ifft2c(x)
        mag = x.abs()
        if mag.shape[1] > 1:  # RSS coil-combine
            mag = torch.sqrt((mag**2).sum(dim=1, keepdim=True) + 1e-8)
        return self._unit_scale(mag) if self._normalize_magnitude else mag

    @staticmethod
    def _spatial_smoothness(field: torch.Tensor) -> torch.Tensor:
        dx = field[..., 1:, :] - field[..., :-1, :]
        dy = field[..., :, 1:] - field[..., :, :-1]
        return dx.abs().mean() + dy.abs().mean()

    @staticmethod
    def _object_mask(anat: torch.Tensor, field_shape: torch.Size) -> torch.Tensor:
        """Binary object-support mask ``[B, 1, H, W]`` from an anatomy magnitude.

        A field map (B0/B1) is only physically meaningful where there is signal;
        outside the object a real/DAM-derived field is the ratio of two noise
        floors and fills the whole dynamic range. See :meth:`_store_field_visuals`.

        The bright reference is a per-sample **99.5th percentile**, not ``amax``.
        Percentile-normalised M4Raw targets carry a tail far above the tissue
        bulk (the 2026-07 exp_vf_01 run had ``target_mag`` spanning [0, 92.7]
        with the brain near 1), so an ``amax``-relative threshold landed above
        the whole object: the mask kept ~0.1% of the frame and every saved
        validation PNG — real and fake — rendered pure black.
        """
        m = anat.abs() if torch.is_complex(anat) else anat
        if m.shape[1] > 1:  # RSS coil/echo-combine to a single anatomy channel
            m = torch.sqrt((m**2).sum(dim=1, keepdim=True) + 1e-8)
        if m.shape[-2:] != field_shape[-2:]:
            m = interpolate(m, size=tuple(field_shape[-2:]), mode="bilinear", align_corners=False)
        b, c = m.shape[0], m.shape[1]
        ref = torch.quantile(m.reshape(b, c, -1).float(), 0.995, dim=-1, keepdim=True)
        ref = ref.unsqueeze(-1).to(m.dtype)
        # an all-zero sample has no percentile to key off — fall back to amax so
        # the mask stays empty rather than selecting the whole (empty) frame
        ref = torch.where(ref > 0, ref, m.amax(dim=(-2, -1), keepdim=True))
        thr = 0.05 * ref
        return (m > thr).to(torch.float32)

    def _store_field_visuals(self, pred_field: torch.Tensor, field: torch.Tensor) -> None:
        """Cache the (pred, reference) field pair for the validation visualiser.

        DISPLAY-ONLY: the loss/metrics use the unmasked ``field`` / ``pred_field``
        (already computed by the caller). Here we mask the visual copies to the
        anatomical object support so the saved ``real`` / ``fake`` PNGs render on
        the same black-air background. Without this, an external/real B1 map —
        noise outside the object — is stretched to mid-grey speckle by the
        per-sample windowing in ``MetricsTracker._normalize_images`` and the
        *reference* looks more degraded than the model's smoother estimate
        (the 2026-06-21 vf_22-24 "target more degraded than output" report).
        """
        anat = getattr(self, "_last_anat", None)
        if anat is not None:
            mask = self._object_mask(anat, field.shape)
            self._last_visual_pred = (pred_field * mask).detach()
            self._last_visual_target = (field * mask).detach()
        else:  # no anatomy reference available → fall back to the raw fields
            self._last_visual_pred = pred_field.detach()
            self._last_visual_target = field.detach()

    def _real_field_from_batch(self, batch: Any) -> torch.Tensor | None:
        """Pick the real reference field the method recovers from the batch:
        b1_map for B1 methods (double_angle/afi/bloch_siegert), b0_map for B0
        (dual_echo, bssfp_banding). Returns None → synthesise a random field
        (legacy behaviour).

        Accepts either a raw dict or a ``TrainingBatch`` (the trainer converts
        the loaded batch to a ``TrainingBatch`` before ``train_step``; the field
        lives in ``metadata`` and is reachable via ``.get()`` with the metadata
        fallback). The old ``isinstance(batch, dict)`` guard rejected the
        ``TrainingBatch`` form, so the real-field supervision path was dead at
        train time exactly as it was at validation time."""
        if batch is None or not hasattr(batch, "get"):
            return None
        key = "b0_map" if self.macq.method in _B0_METHODS else "b1_map"
        field = batch.get(key)
        return self._to_device(field) if field is not None else None

    def _condition_on_shifts(self, res: Any) -> torch.Tensor:
        """Append per-frame shift-conditioning maps to the acquisition stack.

        Every rung of the ladder produces the SAME tensor shape
        ``[B, 3n, h, w]`` — n frames plus 2n constant maps carrying the frame's
        ``(dy, dx)`` normalised to ``[-1, 1]``. Only the CONTENT of those maps
        changes, so architecture, parameter count and input geometry are held
        fixed and the ablation moves exactly one variable:

        ``blind``     zeros (the model is told nothing)
        ``recovered`` phase correlation of the fiducial against the un-shifted
                      reference, so the offsets come from the marker
        ``oracle``    the simulator's ground truth

        The recovered offsets are detached: they are a measurement handed to the
        model, and phase correlation has no parameters for a gradient to reach.
        """
        if self._method != "subvoxel_sr":
            return res.stack

        stack = res.stack
        b, _n, h, w = stack.shape
        true_shifts = res.shifts

        if self._shift_source == "oracle":
            shifts = true_shifts
        elif self._shift_source == "recovered":
            if res.marker_stack is None:
                raise ValueError(
                    "shift_source='recovered' but the simulator produced no marker "
                    "stack. The fiducial is what carries the offsets; without it "
                    "there is nothing to register against."
                )
            if res.marker_hr is None:
                raise ValueError("the marker stack carries no marker_hr to register against")
            ref = self.macq.pooled(res.marker_hr)
            # Absolute registration against the KNOWN un-shifted fiducial, not
            # frame-to-frame: that absolute anchor is what a fiducial buys.
            shifts = (
                estimate_subpixel_shifts(ref.expand(b, 1, h, w), res.marker_stack)
                * self.macq.sr_scale
            ).detach()
        else:  # "blind" — Literal-validated upstream, so no other value reaches here
            shifts = torch.zeros_like(true_shifts)

        if true_shifts is not None and self._shift_source != "blind":
            self._last_shift_mae = (shifts - true_shifts).abs().mean().detach()
        elif true_shifts is not None:
            self._last_shift_mae = true_shifts.abs().mean().detach()

        self._last_shifts = shifts
        if self._psf_enabled:
            self._last_psf_channel = self._psf_channel(res, shifts)
        if self._marker_channels:
            if res.marker_stack is None:
                raise ValueError(
                    "marker_channels=True but the simulator produced no marker "
                    "stack. marker_channels forces marker_enabled at setup, so "
                    "this means the simulator was built some other way."
                )
            # Anatomy frames first, then the instrument, then the shift maps.
            # `_append_shift_maps` puts the maps last, so a model can slice the
            # two stacks by a fixed offset.
            stack = torch.cat((stack, res.marker_stack), dim=1)
        return self._append_shift_maps(stack, shifts)

    def _append_shift_maps(self, stack: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
        """Concatenate the 2n constant shift maps onto an n-frame stack.

        Split out of :meth:`_condition_on_shifts` so the super-Nyquist probe can
        feed the MARKER stack through the identical input construction. Reusing
        one builder is what makes the probe a measurement of the same network on
        the same input geometry, rather than a second, subtly different path.
        """
        b, _c, h, w = stack.shape
        n = shifts.shape[1]
        maps = (shifts / self._max_shift_px).clamp(-1.0, 1.0)
        maps = maps.reshape(b, 2 * n, 1, 1).expand(b, 2 * n, h, w)
        if not self._psf_enabled:
            return torch.cat((stack, maps), dim=1)
        if self._last_psf_channel is None:
            raise RuntimeError(
                "forward_psf is enabled but no PSF channel was resolved; "
                "_condition_on_shifts must run before any model input is built."
            )
        return torch.cat((stack, maps, self._last_psf_channel), dim=1)

    def _calibrate_conformal(self, res: Any, pred_field: torch.Tensor) -> dict[str, float]:
        """Calibrate on the fiducial, then TEST the guarantee on the anatomy.

        Two residual sets. The marker's is computable with no ground truth,
        which is what makes the construction usable at inference; the anatomy's
        exists here only because the digital twin supplies its target, and it is
        used ONLY to check the assumption, never to calibrate. Calibrating on it
        would be assuming what the arm is meant to test.
        """
        if (
            self._conformal is None
            or self._subject_bank is not None
            or res is None
            or res.marker_stack is None
        ):
            return {}
        with torch.no_grad():
            marker_hr = res.marker_hr.to(pred_field.dtype)
            marker_pred = self.generator_model(
                self._append_shift_maps(
                    (
                        torch.cat((res.marker_stack, res.marker_stack), dim=1)
                        if self._marker_channels
                        else res.marker_stack
                    ),
                    self._last_shifts,
                )
            )[:, :1]
            support = (marker_hr > 0.05 * marker_hr.amax()).to(marker_hr.dtype)
            try:
                self._conformal.fit(marker_pred - marker_hr, local_detail_score(marker_hr), support)
            except ValueError as exc:  # too few points for the declared strata
                logger.warning("[MultiAcq] conformal calibration skipped: %s", exc)
                return {}
            report = self._conformal.coverage(pred_field - res.field, local_detail_score(res.field))
        out = report.as_dict()
        # Interval width matters as much as coverage: an infinitely wide band
        # covers everything and says nothing.
        out["conformal_mean_half_width"] = float(
            self._conformal.half_width(local_detail_score(res.field)).mean()
        )
        return out

    def _estimate_lr_kernel(self, res: Any, shifts: torch.Tensor) -> torch.Tensor:
        """Solve for the blur the acquisition applied, ``[B, 1, 1, k, k]``.

        BOTH sides at the POOLED resolution. Upsampling the observed frame back
        to HR instead would fold the pooling and the interpolation into the
        estimate (it read 18% low that way); here the pooling is common to both
        sides and divides out, so what is left is the blur alone.
        """
        s_scale = self.macq.sr_scale
        with torch.no_grad():
            observed = res.marker_stack[:, :1]
            known = self.macq.pooled(res.marker_hr).to(observed.dtype)
            # The frame was dithered before pooling, so align first or the
            # displacement is absorbed into the kernel as a shifted centroid.
            known = fourier_shift(known, shifts[:, :1] / s_scale)
            cfg = self._psf_cfg
            return estimate_psf(
                observed,
                known,
                kernel_size=cfg.kernel_size,
                mu=cfg.mu,
                control_grid=(cfg.control_rows, cfg.control_cols),
            )

    def _psf_channel(self, res: Any, shifts: torch.Tensor) -> torch.Tensor:
        """The kernel ``forward_psf.source`` names, rendered as ``|OTF|`` ``[B,1,h,w]``."""
        b, _n, h, w = res.stack.shape
        if self._psf_cfg.source == "measured":
            if res.marker_stack is None:
                raise ValueError(
                    "forward_psf.source='measured' but the simulator produced no "
                    "marker stack to measure the kernel from."
                )
            return otf_channel(self._estimate_lr_kernel(res, shifts), (h, w)).detach()
        # "assumed" (Literal-validated upstream): a constant, so rendered once per grid.
        key = (h, w, res.stack.device, res.stack.dtype)
        if self._assumed_otf_key != key:
            kernel = self._assumed_lr_kernel.to(res.stack.device, res.stack.dtype)
            self._assumed_otf = otf_channel(kernel, (h, w)).detach()
            self._assumed_otf_key = key
        return self._assumed_otf.expand(b, -1, -1, -1)

    def _measure_psf(self, res: Any) -> dict[str, float]:
        """Report the measured kernel against the assumed one, on both arms.

        The measurement is a property of the simulator, not of the model, so
        the assumed arm has to carry the same number for the comparison to have
        a shared reference. ``psf_identifiability`` rides alongside because a
        kernel estimated where the marker has no spectral energy is
        interpolation, and a FWHM quoted without it hides that.
        """
        if not self._psf_enabled or res.marker_stack is None:
            self._last_psf = {}
            return {}
        shifts = getattr(self, "_last_shifts", None)
        if shifts is None:
            self._last_psf = {}
            return {}
        measured = self._estimate_lr_kernel(res, shifts)
        with torch.no_grad():
            known = self.macq.pooled(res.marker_hr).to(measured.dtype)
            assumed = self._assumed_lr_kernel.to(measured.device, measured.dtype)
            out = {
                "psf_fwhm_measured": float(psf_fwhm_map(measured).mean()),
                "psf_fwhm_assumed": float(psf_fwhm_map(assumed).mean()),
                "psf_identifiability": float(psf_identifiability(known).mean()),
            }
            out["psf_fwhm_error"] = out["psf_fwhm_measured"] - out["psf_fwhm_assumed"]
        self._last_psf = out
        return out

    def _measure_kappa(self, res: Any) -> None:
        """Check the DECLARED gain against the one the data actually shows.

        On marker support the true ratio is measurable, so the arm can report
        whether the constant it handed the network was right. A factored model
        whose measured gain disagrees with its predicted one is applying a wrong
        constant confidently, which is worse than not factoring at all -- hence
        this is reported on the unfactored control too.
        """
        if not self._relax_enabled or res.marker_stack is None:
            self._last_kappa_measured = None
            return
        shifts = getattr(self, "_last_shifts", None)
        if shifts is None:
            self._last_kappa_measured = None
            return
        with torch.no_grad():
            lr = res.marker_stack[:, :1]
            ref = self.macq.pooled(res.marker_hr).to(lr.dtype)
            # Align before dividing. The marker frames are pooled views of a
            # SHIFTED marker, so an un-shifted reference biases the ratio by the
            # displacement itself: 6.3% at a 1.5 px dither, which would read as a
            # 6% error in the declared relaxometry. The shift is applied in LR
            # pixels, matching the pooled grid the comparison happens on.
            ref = fourier_shift(ref, shifts[:, :1] / self.macq.sr_scale)
            support = (ref > 0.05 * ref.amax()).to(lr.dtype)
            self._last_kappa_measured = measured_gain(ref, lr, support).mean()

    def _anatomy_band_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        """``mean(1 - rho)`` over the super-Nyquist bands of the ANATOMY pair.

        Scale-free by construction, so it constrains band STRUCTURE and leaves
        amplitude to the MSE rather than competing with it. Returns ``None``
        when the partition is not built.
        """
        if self._snf is None:
            return None
        rho = self._snf.transfer(pred, target)
        return (1.0 - rho[:, list(self._snf.super_nyquist_bands)]).mean()

    def _band_probe(self, res: Any) -> torch.Tensor | None:
        """Push the KNOWN fiducial through the network; measure per-band transfer.

        The marker frames are ``sr_scale``-pooled, sub-pixel-shifted views of a
        signal whose every spatial frequency is known exactly and which carries
        no anatomy. Restoring it is the same inverse problem the anatomy path
        solves, so the per-band gain ``rho_l`` above the acquisition Nyquist is
        the fraction of UNMEASURED detail the network genuinely recovers. Below
        Nyquist the content was measured and a high gain proves nothing.

        Returns the loss over the super-Nyquist bands, or ``None`` when the
        probe is off. The spectrum is cached for reporting either way, so the
        control arm (``lambda_band=0``) reports the same numbers it does not
        optimise.
        """
        if self._snf is None or res.marker_stack is None:
            # Clear rather than leave the previous step's numbers in place: a
            # stale spectrum reported against a run that did not compute one is
            # indistinguishable from a real measurement.
            self._last_band_spectrum = {}
            return None
        shifts = getattr(self, "_last_shifts", None)
        if shifts is None:
            raise RuntimeError(
                "band probe ran before the shifts were resolved; "
                "_condition_on_shifts must precede it."
            )
        marker_lr = res.marker_stack
        b, _, h, w = marker_lr.shape
        hr_size = (h * self.macq.sr_scale, w * self.macq.sr_scale)
        probe_stack = marker_lr
        if self._marker_channels:
            # The model expects [anatomy | instrument | maps]. In the probe pass
            # the fiducial IS the signal under test, so it occupies both slots.
            # Note what that means for a marker-keyed arm: queries, keys and
            # values all come from the same tensor, so the probe measures
            # transfer under a SELF-keyed routing rather than the cross-keyed one
            # the anatomy path uses. It is still the same weights on the same
            # input geometry, but the certificate is not a measurement of the
            # cross-keying itself -- state that when quoting it.
            probe_stack = torch.cat((marker_lr, marker_lr), dim=1)
        pred = self.generator_model(self._append_shift_maps(probe_stack, shifts))[:, :1]
        target = res.marker_hr.to(pred.dtype)
        if pred.shape[-2:] != target.shape[-2:]:
            raise ValueError(
                f"band probe: model returned {tuple(pred.shape[-2:])} for a "
                f"{hr_size} target. The probe reuses the anatomy path's input "
                "geometry, so a mismatch here means the model's scale factor "
                "disagrees with physics.multi_acquisition.sr_scale."
            )
        # No support mask: in the probe pass the marker IS the whole signal, so
        # its band components legitimately occupy the full field and excluding
        # the ringing would discard real information. A support weight becomes
        # meaningful only once the marker is embedded alongside anatomy.
        rho = self._snf.transfer(pred, self._probe_loss_target(res, target))
        spectrum = self._snf.spectrum(pred, target)
        # The FLOOR: what a plain interpolator scores on the same instrument.
        # It is not zero, and it is arm-dependent. Boxcar pooling is not an
        # ideal anti-aliasing filter, so a single LR frame retains folded-back
        # super-Nyquist energy that bilinear interpolation partially unfolds by
        # accident -- measured at rho_super = 0.53 on the synthetic marker at
        # sr_scale=2, against 1.00 for an exact multi-frame inversion and -0.09
        # for a plausible-but-wrong marker. An absolute rho quoted without this
        # reference would read as recovery when it is interpolation, so both
        # numbers are reported and neither is silently folded into the other.
        floor = interpolate(marker_lr[:, :1], size=hr_size, mode="bilinear", align_corners=False)
        spectrum["snf_floor_super_nyquist"] = self._snf(floor, target)
        spectrum["snf_gain_over_floor"] = (
            spectrum["snf_super_nyquist"] - spectrum["snf_floor_super_nyquist"]
        )
        self._last_band_spectrum = spectrum
        sup = list(self._snf.super_nyquist_bands)
        return (1.0 - rho[:, sup]).mean()

    def _probe_loss_target(self, res: Any, matched: torch.Tensor) -> torch.Tensor:
        """The marker the instrument term is optimised against.

        ``mismatched`` (the wrong-marker control) swaps in another realisation
        of the same geometry, so the term keeps its magnitude and gradient path
        and loses its information. The reported spectrum still scores ``matched``.
        """
        if self._probe_target != "mismatched":
            return matched
        b = matched.shape[0]
        seeds = res.marker_seeds or (self._marker_seed,) * b
        wrong = tuple(wrong_seed(s) for s in seeds)
        return self.macq.marker_realisations(
            wrong, tuple(matched.shape[-2:]), matched.device, matched.dtype
        )

    def _marker_seeds(self, family: str, epoch: int, b: int, batch_idx: Any = None):
        """Per-sample realisation seeds, or ``None`` for the fixed fiducial."""
        if not self._per_sample_marker:
            return None
        if family == "val":
            if batch_idx is None:
                raise ValueError(
                    "marker_realisation='per_sample' keys validation markers on "
                    "batch_idx; validation_step was called without it."
                )
            # A stride wider than any batch: a short last batch cannot collide.
            ordinals = [int(batch_idx) * 65536 + i for i in range(b)]
        else:
            if epoch != self._marker_epoch:
                self._marker_epoch, self._marker_ordinal = epoch, 0
            ordinals = list(range(self._marker_ordinal, self._marker_ordinal + b))
            self._marker_ordinal += b
        return realisation_seeds(self._marker_seed, family, epoch, ordinals)

    @property
    def _needs_loaded_input(self) -> bool:
        return self._input_stream == "acquired" or self._fov_mask != "none"

    def _acquired_result(self, ulf: torch.Tensor, m0: torch.Tensor) -> AcquisitionResult:
        """The REAL 64 mT slice as a one-frame, undithered acquisition (VFU-07)."""
        if ulf.shape[-2:] != m0.shape[-2:]:
            raise ValueError(
                f"acquired input {tuple(ulf.shape[-2:])} and target {tuple(m0.shape[-2:])} "
                "are not on one grid; the paired manifest resamples ULF onto the 3T grid."
            )
        return AcquisitionResult(
            stack=self.macq.pooled(ulf), field=m0, shifts=m0.new_zeros(m0.shape[0], 1, 2)
        )

    def _field_loss(
        self,
        target_batch: torch.Tensor,
        external_field: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        marker_seeds: tuple[int, ...] | None = None,
        acquired_input: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m0 = self._clean_magnitude(target_batch)
        self._last_anat = m0.detach()  # object support for display masking
        ulf = None
        if self._needs_loaded_input:
            if acquired_input is None:
                raise ValueError(
                    "input_stream='acquired' / fov_mask need the loaded 64 mT input, "
                    "and none reached the loss."
                )
            ulf = self._clean_magnitude(acquired_input)
        if self._input_stream == "acquired":
            res = self._acquired_result(ulf, m0)
        else:
            res = self.macq(
                m0, external_field=external_field, generator=generator, marker_seeds=marker_seeds
            )
        model_input = self._condition_on_shifts(res)
        self._last_result = res
        pred = self.generator_model(model_input)
        # The target may be a 1-channel field (B0/B1), 2-channel params (MRF
        # T1/T2), or a multi-channel image/series — supervise the matching head.
        k = res.field.shape[1]
        if self._tissue_gains is not None:
            # y = kappa(x) * g(x), kappa(x) a posterior over the DECLARED gains.
            logits = pred[:, k : k + N_TISSUES]
            self._last_kappa_logits = logits.detach()
            pred = pred[:, :k] * tissue_kappa_map(logits, self._tissue_gains)
        elif self._relax_factored and self._marker_kappa is not None:
            # y = kappa * g(x) with the fiducial's one constant: the scalar control.
            pred = pred * self._marker_kappa
        self._measure_kappa(res)
        pred_field, field = pred[:, :k], res.field
        if self._fov_mask == "acquired_support":
            # Outside the ULF field of view the resample wrote zeros; the relative
            # floor only absorbs FFT round-off on a k-space-stored input.
            floor = 1e-6 * ulf.amax(dim=(-2, -1), keepdim=True)
            support = (ulf > floor).to(pred_field.dtype)
            pred_field, field = pred_field * support, field * support
        mse = mse_loss(pred_field, field)
        loss = mse + self._lambda_smooth * self._spatial_smoothness(pred_field)
        # The TASK objective, before the probe term. Kept separate so a reader
        # can attribute a change in g_total_loss to the task or to the probe;
        # `field_mse` used to be logged as the total, which silently renamed
        # whatever else was in the sum.
        self._last_task_loss = loss.detach()
        self._last_mse = mse.detach()
        # Unlike a shift-supervision term, this one reaches the weights: it
        # constrains the network's OUTPUT on the instrument, not the
        # parameterless phase correlation that reads the offsets.
        probe = self._band_probe(res)
        self._last_band_loss = None if probe is None else probe.detach()
        if probe is not None and self._lambda_band > 0.0:
            loss = loss + self._lambda_band * probe
        # The DETAIL term. Same partition, anatomy pair, ordinary supervision --
        # it uses the HR target, which exists now and will not at inference, so
        # a gain it earns is NOT the instrument's certificate. Nothing else in
        # this objective targets high frequency: the strategy never reads a
        # `losses:` block, so the base loss is MSE plus a smoothness penalty.
        hf = self._anatomy_band_loss(pred_field, field)
        self._last_hf_loss = None if hf is None else hf.detach()
        if hf is not None and self._lambda_anatomy > 0.0:
            loss = loss + self._lambda_anatomy * hf
        return loss, pred_field, field

    def _real_stack_loss(
        self, stack: torch.Tensor, external_field: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Path A: grade the model on the REAL stack vs the real field (no synthesis)."""
        if external_field is None:
            raise ValueError(
                "multi_acquisition.use_real_stack=true requires a real b0_map in the "
                "batch (oracle_bssfp emits it) — nothing to grade against."
            )
        field = self._to_device(external_field)
        stack = self._to_device(stack)
        # Anatomy support for display masking: RSS magnitude of the real stack.
        self._last_anat = (stack.abs() if torch.is_complex(stack) else stack).detach()
        pred = self.generator_model(stack)
        k = field.shape[1]
        pred_field = pred[:, :k]
        loss = mse_loss(pred_field, field)
        return loss, pred_field, field

    # ── training ──────────────────────────────────────────────────────────
    def train_step(
        self,
        batch: Any,
        epoch: int,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Generator step that BYPASSES the base raw-batch channel guards.

        ``BaseTrainingStrategy.train_step`` validates the *loaded* batch against
        ``model.in_channels`` / ``out_channels`` and real-stacks it for the
        generator (a 4-D ``[B, C, H, W]`` assumption). For multi-acquisition the
        model never consumes the loaded batch: the strategy synthesises an
        acquisition *stack* whose channel count equals the number of
        acquisitions (2 echoes, 16 MRF timepoints, …), decoupled from the coil
        count, inside :meth:`_field_loss`, and supervises a field map. The base
        ComplexGuard / ``[DomainMismatch]`` pre-checks therefore (a) reject a
        perfectly valid batch whenever ``raw_coil_channels != stack_channels``
        and (b) cannot unpack a 5-D (3-D-patch) volume. We skip those raw-batch
        checks and reuse only the base loss/AMP/anomaly machinery via
        :meth:`_compute_losses`. (cluster smoke 2026-06-03: exp_vf_01 crashed at
        ``base.py`` 4-D unpack; arms 21-24/28 would raise ``[DomainMismatch]``.)
        """
        if target_batch is None or (self._needs_loaded_input and input_batch is None):
            _lr, hr = self._unpack_batch(batch)
            target_batch = pick_present(target_batch, hr)
            input_batch = pick_present(input_batch, _lr)
        target_batch = self._to_device(target_batch)
        if self._needs_loaded_input:
            kwargs["acquired_input"] = self._to_device(input_batch)
        if not self.generator_model.training:
            self.generator_model.train()

        def g_closure() -> torch.Tensor:
            losses, metrics_tensor = self._compute_losses(
                None, target_batch, epoch, batch=batch, **kwargs
            )
            loss_total = losses.get("g_total_loss")
            if loss_total is None:
                raise RuntimeError("ConcreteMultiAcquisitionStrategy produced no 'g_total_loss'.")
            detached = {k: v.detach() if hasattr(v, "detach") else v for k, v in losses.items()}
            self._last_step_metrics = self._handle_anomalies(detached, metrics_tensor, epoch)
            return loss_total

        return [
            {
                "optimizer": self.env.opt_g,
                "closure": g_closure,
                "model": self.generator_model,
                "name": "generator",
            }
        ]

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        # A REAL field (b0_map/b1_map) from the batch supervises the model against
        # a real off-resonance / B1 pattern instead of a random synthetic field.
        external_field = self._real_field_from_batch(kwargs.get("batch"))
        if self._subject_bank is not None:
            self._train_subjects.update(
                self._batch_subjects(kwargs.get("batch"), int(target_batch.shape[0]))
            )
        if self._use_real_stack:  # Path A: the model input IS the real stack
            loss, pred_field, field = self._real_stack_loss(input_batch, external_field)
        else:
            seeds = self._marker_seeds("train", epoch, int(target_batch.shape[0]))
            loss, pred_field, field = self._field_loss(
                target_batch,
                external_field,
                marker_seeds=seeds,
                acquired_input=kwargs.get("acquired_input"),
            )
        self._store_field_visuals(pred_field, field)
        out: dict[str, torch.Tensor] = {"g_total_loss": loss}
        if getattr(self, "_last_mse", None) is not None:
            out["field_mse"] = self._last_mse
            out["task_loss"] = self._last_task_loss
        else:  # real-stack path: the objective IS the MSE
            out["field_mse"] = loss.detach()
        # Diagnostic, never optimised: see SubvoxelRegistrationConfig on why a
        # lambda_shift term would have identically zero gradient.
        if self._last_shift_mae is not None:
            out["shift_mae_px"] = self._last_shift_mae
        if getattr(self, "_last_band_loss", None) is not None:
            out["band_probe_loss"] = self._last_band_loss
        if getattr(self, "_last_hf_loss", None) is not None:
            out["anatomy_hf_loss"] = self._last_hf_loss
        if self._last_kappa_measured is not None:
            out["kappa_measured"] = self._last_kappa_measured
        return out

    @staticmethod
    def _batch_subjects(batch: Any, b: int) -> list[str]:
        """Per-sample subject ids from a loaded batch; raises when absent."""
        ids = batch.get("subject_id") if batch is not None and hasattr(batch, "get") else None
        if ids is None:
            raise ValueError(
                "anchor_conformal.calibration_source='held_out_subjects' needs "
                "subject_id on every batch, and this batch carries none."
            )
        ids = [ids] if isinstance(ids, str) else [str(i) for i in ids]
        if len(ids) != b:
            raise ValueError(f"{len(ids)} subject ids for a batch of {b}")
        return ids

    def reset_validation_metrics(self) -> None:
        super().reset_validation_metrics()
        if self._subject_bank is not None:
            self._subject_bank.reset()

    def finalize_validation(self) -> dict[str, float]:
        out = dict(super().finalize_validation())
        if self._subject_bank is not None and self._conformal is not None:
            report = self._subject_bank.evaluate(
                self._conformal, train_subjects=self._train_subjects
            )
            out.update({f"val_{k}": v for k, v in report.items()})
            self._subject_bank.reset()
        return out

    # ── validation ────────────────────────────────────────────────────────
    def validation_step(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        b0_map: torch.Tensor | None = None,
        b1_map: torch.Tensor | None = None,
        batch_data: Any = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        # A real reference field (gated trainer pass) grades the model against a
        # real B0/B1 instead of the synthesised field (real-reference seam).
        external_field = b0_map if self.macq.method in _B0_METHODS else b1_map
        if external_field is not None:
            external_field = self._to_device(external_field)
        with torch.no_grad():
            if self._use_real_stack:  # Path A: grade model(real stack) vs real b0
                _loss, pred_field, field = self._real_stack_loss(input_batch, external_field)
            else:
                gen = torch.Generator(device=target_batch.device).manual_seed(_VALIDATION_SEED)
                seeds = self._marker_seeds(
                    "val", 0, int(target_batch.shape[0]), kwargs.get("batch_idx")
                )
                _loss, pred_field, field = self._field_loss(
                    target_batch,
                    external_field,
                    generator=gen,
                    marker_seeds=seeds,
                    acquired_input=(
                        self._to_device(input_batch) if self._needs_loaded_input else None
                    ),
                )
            self._store_field_visuals(pred_field, field)
            bias = (pred_field - field).mean()
            mae = (pred_field - field).abs().mean()
            # Per-sample data range: a batch-wide range would make the PSNR
            # depend on which samples share a batch. B0 fields are in Hz, so a
            # fixed unit range would be wrong for them.
            flat_p, flat_f = pred_field.reshape(pred_field.shape[0], -1), field.reshape(
                field.shape[0], -1
            )
            rng = (flat_f.amax(dim=1) - flat_f.amin(dim=1)).clamp(min=1e-8)
            mse_s = ((flat_p - flat_f) ** 2).mean(dim=1)
            psnr = (10.0 * torch.log10(rng**2 / (mse_s + 1e-10))).mean()
        result = {
            "val_field_mse": float(mse_loss(pred_field, field)),
            # SIGNED systematic offset -- a DIAGNOSTIC, never a selection target.
            # Minimising a signed bias picks the most NEGATIVE field estimate, so
            # this key carries no direction and monitoring it now raises (#230).
            "val_field_bias": float(bias),
            # The selectable form: |mean(pred - field)| -> 0 is the goal. Distinct
            # from `val_field_mae` = mean(|pred - field|), which also counts random
            # error -- a model can be unbiased (abs_bias 0) and still have a large
            # MAE, and the two answer different questions.
            "val_field_abs_bias": float(bias.abs()),
            "val_field_mae": float(mae),
            "val_psnr": float(psnr),
            # 1.0 = graded vs a REAL field from the batch; 0.0 = self-consistency
            # on the synthesised field (lets a reader tell the regimes apart).
            "val_field_reference_real": 1.0 if external_field is not None else 0.0,
        }
        # Registration accuracy in HR pixels. Under shift_source='blind' this is
        # the error of the zeros the model was handed, so the three ladder rungs
        # are directly comparable on one axis.
        if self._last_shift_mae is not None:
            result["val_shift_mae_px"] = float(self._last_shift_mae)
        # Per-band transfer gain on the fiducial. Reported whenever the probe is
        # enabled, INCLUDING the lambda_band=0 control, so the two arms differ on
        # exactly one knob and the constraint's cost in task PSNR is attributable.
        result.update({f"val_{k}": v for k, v in self._last_band_spectrum.items()})
        result.update(
            {f"val_{k}": v for k, v in self._measure_psf(self._last_result).items()}
            if self._last_result is not None
            else {}
        )
        if self._subject_bank is not None:
            self._subject_bank.add(
                self._batch_subjects(batch_data, int(pred_field.shape[0])),
                pred_field - field,
                local_detail_score(field),
            )
        # Coverage is GATED, not asserted: val_conformal_guaranteed is 0.0 when
        # any stratum missed its level, and that is a reportable result rather
        # than something to widen the tolerance until it disappears.
        result.update(
            {
                f"val_{k}": v
                for k, v in self._calibrate_conformal(self._last_result, pred_field).items()
            }
        )
        # Predicted vs measured contrast transfer on the fiducial. Reported on
        # BOTH arms, so the factored arm's constant is checkable rather than
        # assumed, and a disagreement is visible instead of absorbed.
        if self._marker_kappa is not None:
            result["val_kappa_predicted"] = float(self._marker_kappa)
        if self._last_kappa_logits is not None and self._last_result is not None:
            # Report-only: the simulator's classes come from the HR target here.
            classes = self.macq.tissue_classes(self._last_result.field)
            result["val_kappa_class_agreement"] = float(
                class_agreement(self._last_kappa_logits, classes)
            )
        if self._last_kappa_measured is not None:
            result["val_kappa_measured"] = float(self._last_kappa_measured)
            if self._marker_kappa:
                result["val_kappa_error"] = float(
                    self._last_kappa_measured / self._marker_kappa - 1.0
                )
        # Absolute Hz accuracy for the B0 methods, graded vs the REAL b0_map via
        # the guarded b0_field_rmse metric. The guard returns NaN (skip) if the
        # model handed an image instead of an Hz field (pitfall #16) — so this
        # number is only ever a true Hz-vs-Hz comparison.
        if self.macq.method in _B0_METHODS and external_field is not None:
            result["val_b0_field_rmse"] = B0FieldRMSE()(pred_field, field)
        # qMRI agreement battery on the (pred_field, reference_field) pair the
        # strategy already produces — Bland-Altman bias + limits of agreement,
        # ICC(3,1), CoV (deferred_source_backlog.json: "run the qMRI metrics
        # battery on (pred_field, field)"). These grade inverter SELF-CONSISTENCY
        # vs the simulator's own field, not real-scanner accuracy.
        result.update({f"val_{k}": v for k, v in qmri_agreement_metrics(pred_field, field).items()})
        return result


__all__ = ["ConcreteMultiAcquisitionStrategy"]
