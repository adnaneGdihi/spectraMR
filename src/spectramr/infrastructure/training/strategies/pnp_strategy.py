"""Plug-and-Play / RED Strategy (PR-9 / M1).

Plan: TODO/backlog_paradigm_expansion_roadmap.md §PR-9.

ADMM iteration with a spectral-norm-bounded denoiser plays the role of
the proximal operator on the prior. The denoiser is wrapped via
``SpectralNormalizedDenoiser`` to enforce a Lipschitz constant ≤ 1, which
is the convergence requirement for PnP / RED.

Training objective (the paradigm-specific math that makes ``pnp`` more
than plain reconstruction — backlog #15): on top of the supervised
fidelity term inherited from :class:`ReconstructionTrainingStrategy`, we
add a **Regularization-by-Denoising** (RED) prior.  RED (Romano, Elad &
Milanfar, "The Little Engine that Could", SIAM J. Imaging Sci. 2017)
defines the prior gradient as the denoising residual ``x - D(x)``;
integrating it yields the tractable surrogate energy

    ρ_red(x) = ½ ‖x - D(x)‖²,

where the generator ``D`` plays the role of the denoiser acting on the
current reconstruction ``x``.  Minimising this drives ``x`` toward the
denoiser's fixed-point set (the learned image manifold), which is exactly
the prior PnP/RED enforce at inference time via :meth:`run_pnp`.  The term
flows gradient into the generator (the denoiser), so it shapes training —
the arm is no longer byte-identical to pure reconstruction.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.distributed.owned_modules import OWNED_BUFFERS, sync_owned_buffers

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)

# Default RED prior weight when no ``losses.reconstruction.lambda_red`` knob
# is supplied (genuinely-optional; small so it regularizes without dominating
# the supervised fidelity term).
_DEFAULT_LAMBDA_RED = 0.05


class PnPStrategy(ReconstructionTrainingStrategy):
    """Plug-and-Play / RED strategy.

    For training time we operate on the supervised regression objective
    inherited from the parent **plus** a RED prior term that uses the
    generator as the denoiser on the current reconstruction (see module
    docstring). Inference-time ADMM iterates are surfaced via ``run_pnp``
    as a **manual API** (call ``strategy.run_pnp(y, mask)`` directly); it is
    not yet dispatched from the predict CLI / training loop.
    """

    #: Loss ownership (mrixfields review 2026-09-03): folds the builder's other declared entries (calls the fold or the parent path).
    inline_losses: ClassVar[frozenset[str] | None] = frozenset()
    folds_image_losses: ClassVar[bool | None] = True

    # parallel_compat: The denoiser reparameterises the wrapped generator in place; its
    # spectral-norm vectors are added after the wrap and broadcast once under ddp.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        pnp = getattr(getattr(config, "training", None), "pnp", None)
        own = (("_denoiser", OWNED_BUFFERS),) if getattr(pnp, "enabled", False) else ()
        return (*super().owned_module_declarations(config), *own)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._pnp_cfg = getattr(self.config.training, "pnp", None)
        # Spectral-norm-wrapped denoiser view of the generator (Lipschitz<=1).
        # Built lazily on first use so a stub-constructed strategy (tests) that
        # never has a generator stays inert; populated by _build_denoiser().
        self._denoiser: Any = None
        # Resolved spectral-norm knobs, stamped into provenance (pitfall #15).
        self.pnp_provenance: dict[str, Any] = {}
        self._build_denoiser()

    def _build_denoiser(self) -> Any:
        """Build (once) a spectral-norm-constrained denoiser over the generator.

        Reads ``training.pnp.spectral_norm_iters`` and
        ``training.pnp.denoiser_lipschitz_constant`` (pitfall #15: every
        advertised knob is read + validated + stamped), wraps the generator in
        :class:`SpectralNormalizedDenoiser` to enforce the Lipschitz<=1
        contraction PnP/RED convergence requires, moves it to ``self.device``
        and stamps the resolved values into ``self.pnp_provenance``.

        Returns the wrapper (or ``None`` when PnP is disabled / no generator).
        The wrapper shares the generator's parameters via spectral-norm
        reparameterisation (``weight_orig`` replaces ``weight`` in-place), so it
        adds no NEW trainable parameters and needs no ``add_param_group``.
        """
        existing = getattr(self, "_denoiser", None)
        if existing is not None:
            return existing
        cfg = getattr(self, "_pnp_cfg", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            return None
        env = getattr(self, "env", None)
        gen = getattr(env, "generator", None) if env is not None else None
        if gen is None:
            return None

        from spectramr.models.wrappers import SpectralNormalizedDenoiser

        # Validate the Lipschitz target (raise on illegal value, pitfall #9).
        lipschitz = float(getattr(cfg, "denoiser_lipschitz_constant", 1.0))
        if lipschitz <= 0.0:
            raise ValueError(
                f"training.pnp.denoiser_lipschitz_constant must be > 0; got {lipschitz}"
            )
        n_iters = int(getattr(cfg, "spectral_norm_iters", 1))
        if n_iters < 1:
            raise ValueError(f"training.pnp.spectral_norm_iters must be >= 1; got {n_iters}")

        denoiser = SpectralNormalizedDenoiser(gen, n_power_iterations=n_iters)
        device = getattr(self, "device", None)
        if device is not None:
            denoiser = denoiser.to(device)
        if not denoiser.is_lipschitz_constrained:
            raise RuntimeError(
                "PnP requires a Lipschitz-constrained denoiser, but the "
                "generator exposed no Conv/Linear layers to spectral-normalise."
            )
        # Each rank draws its own power-iteration start vectors, and DDP broadcasts only
        # the buffers it saw at the wrap. Once equal, they stay equal: they iterate on
        # the synchronised weights.
        sync_owned_buffers(
            [b for n, b in denoiser.named_buffers() if n.endswith(("weight_u", "weight_v"))],
            owner=type(self).__name__,
            name="_denoiser",
            parallel=getattr(self.config, "parallel", None),
        )
        self._denoiser = denoiser
        # Stamp the resolved knobs into provenance (pitfall #15).
        self.pnp_provenance = {
            "spectral_norm_iters": n_iters,
            "denoiser_lipschitz_constant": lipschitz,
            "spectral_normed_layers": denoiser.normalised_layers,
        }
        log = getattr(self, "logging_service", None)
        if log is not None:
            # LoggingService exposes ``log_info`` (the ILoggingService API), not a
            # bare ``info`` — calling ``info`` crashed every PnP arm at setup
            # (b21_ulf_map, 2026-06-28: "'LoggingService' object has no attribute
            # 'info'" -> zero validation batches, CLAUDE.md #10).
            log.log_info(
                f"[PnP] spectral-normed denoiser built: {self.pnp_provenance}",
                model_type=getattr(getattr(self, "env", None), "model_type", ""),
            )
        return self._denoiser

    def _red_weight(self) -> float:
        """Resolve the RED prior weight from nested config (SSOT).

        Reads ``config.losses.reconstruction.lambda_red`` when present,
        falling back to :data:`_DEFAULT_LAMBDA_RED` for this
        genuinely-optional knob (the schema does not yet declare it).
        """
        losses = getattr(self.config, "losses", None)
        recon = getattr(losses, "reconstruction", None) if losses is not None else None
        return float(getattr(recon, "lambda_red", _DEFAULT_LAMBDA_RED))

    def _denoise(self, x: torch.Tensor) -> torch.Tensor | None:
        """Apply the spectral-normed denoiser ``D`` to ``x`` (Lipschitz<=1).

        Prefers the spectral-norm-constrained wrapper (the PnP/RED convergence
        requirement). Falls back to the raw generator ONLY when PnP is disabled
        (the wrapper is intentionally not built). Returns ``None`` when no
        generator is available at all.
        """
        denoiser = getattr(self, "_denoiser", None) or self._build_denoiser()
        if denoiser is None:
            env = getattr(self, "env", None)
            denoiser = getattr(env, "generator", None) if env is not None else None
        if denoiser is None:
            return None
        out = denoiser(x)
        if isinstance(out, tuple):  # models returning (output, aux...)
            out = out[0]
        if isinstance(out, dict):
            out = out.get("reconstruction", next(iter(out.values())))
        return out

    def _red_prior(self, reconstruction: torch.Tensor) -> torch.Tensor:
        """RED prior surrogate ½‖x - D(x)‖² with the spectral-normed denoiser.

        ``D`` is the Lipschitz<=1 spectral-norm-wrapped generator (see
        :meth:`_denoise`); applying it to the current reconstruction ``x`` and
        penalising the denoising residual is the tractable RED energy whose
        gradient ``x - D(x)`` is the RED prior gradient. Gradient flows into
        the generator/denoiser params (the wrapper shares them).
        """
        denoised = self._denoise(reconstruction)
        if denoised is None:
            return reconstruction.new_zeros(())
        return 0.5 * (reconstruction - denoised).pow(2).mean()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Supervised fidelity (parent) + RED denoiser-prior regularizer.

        F6 (orchestrator-kwargs canonical signature): resolve the legacy
        batch dict, forward to the parent with NAMED kwargs, then add the
        RED prior on the parent's reconstruction.
        """
        base = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )

        # RED must regularize the MODEL's RECONSTRUCTION (so the prior gradient
        # reaches the denoiser), NOT the ground-truth target. The parent does
        # not surface its hr_fakes, so recompute the reconstruction from the
        # generator here. Falling back to target_batch (the old behaviour) made
        # the RED term regularize the ground truth — a no-op for the model.
        x = base.get("prediction")
        if not isinstance(x, torch.Tensor):
            gen = getattr(self.env, "generator", None)
            if gen is not None and isinstance(input_batch, torch.Tensor):
                recon = gen(input_batch)
                if isinstance(recon, dict):
                    recon = recon.get("reconstruction", next(iter(recon.values())))
                if isinstance(recon, (tuple, list)):
                    recon = recon[0]
                x = recon
        if not isinstance(x, torch.Tensor):
            return base

        lambda_red = self._red_weight()
        red = lambda_red * self._red_prior(x)
        base["loss_red"] = red

        for total_key in ("loss_total", "g_total_loss"):
            if total_key in base:
                base[total_key] = base[total_key] + red
                break
        return base

    def _to_denoiser_layout(self, z: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Coerce a (possibly complex) image to the denoiser's input layout.

        When the model expects 2 real channels (interleaved real/imag) and the
        tensor is complex, split into a real 2-channel tensor; otherwise pass
        through unchanged. Returns ``(tensor, was_complex)`` so the caller can
        invert the conversion (finding: a real 2-channel denoiser would error
        on, or silently drop the imaginary part of, a complex input).
        """
        wants_two = int(self.config.model.in_channels or 0) == 2
        if z.is_complex() and wants_two:
            # [B, C, H, W] complex -> [B, 2C, H, W] real (interleaved).
            real = torch.view_as_real(z)  # [..., 2]
            real = real.movedim(-1, 2).flatten(1, 2)
            return real, True
        return z, False

    def _from_denoiser_layout(self, out: torch.Tensor, was_complex: bool) -> torch.Tensor:
        """Invert :meth:`_to_denoiser_layout` when the input was converted."""
        if not was_complex:
            return out
        # [B, 2C, H, W] real -> [B, C, H, W] complex.
        b, c2, h, w = out.shape
        out = out.view(b, c2 // 2, 2, h, w).movedim(2, -1).contiguous()
        return torch.view_as_complex(out)

    @torch.no_grad()
    def run_pnp(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """One-shot ADMM PnP solve at inference (manual API; not the train loop).

        Uses the Lipschitz<=1 spectral-normed denoiser (see :meth:`_denoise`)
        as the prox-on-prior. The k-space x-update splits by the sampling mask:
        sampled bins are the data-fidelity prox, unsampled bins equal the FREE
        prior ``F(z-u)`` (NOT ``rho*F(z-u)/(1+rho)`` — that older form scaled
        every unsampled bin by ``rho/(1+rho)``).
        """
        from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c

        denoiser = getattr(self, "_denoiser", None) or self._build_denoiser()
        cfg = getattr(self, "_pnp_cfg", None)
        if denoiser is None or cfg is None or not cfg.enabled:
            return ifft2c(y) if y.is_complex() else y

        rho = cfg.rho
        mask_b = mask.bool()
        x = ifft2c(y)
        z = x.clone()
        u = torch.zeros_like(x)
        for _ in range(cfg.iterations):
            # x-update: data-fidelity prox at sampled bins; free prior elsewhere.
            kz = fft2c(z - u)
            kx = torch.where(mask_b, (y + rho * kz) / (1.0 + rho), kz)
            x = ifft2c(kx)
            # z-update: spectral-normed denoiser prox (complex<->interleaved).
            z_in, was_complex = self._to_denoiser_layout(x + u)
            z_out = denoiser(z_in)
            if isinstance(z_out, tuple):
                z_out = z_out[0]
            if isinstance(z_out, dict):
                z_out = z_out.get("reconstruction", next(iter(z_out.values())))
            z = self._from_denoiser_layout(z_out, was_complex)
            # Dual update.
            u = u + (x - z)
        return z
