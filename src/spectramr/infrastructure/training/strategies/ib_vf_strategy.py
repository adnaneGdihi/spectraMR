r"""IB-VF training strategy (Phase 2 / Idea 1).

Composes Virtual-Fiducial training with an information-bottleneck
InfoNCE objective on (navigator, motion) pairs. The standard VF
data-consistency loss is preserved; this strategy *adds* two
mutual-information terms:

.. math::

    L_{IB} = -\beta_+ \cdot I(z_m;\hat\theta) + \beta_- \cdot I(z_m; x)

where :math:`z_m = f_\xi(m)` is the navigator latent
(:class:`NavEncoder1D` output) and :math:`\hat\theta` is the estimated
rigid-motion vector. Both MI terms are estimated through the
:class:`InfoNCECritic` lower bound; the IB sign convention (maximise
the first, minimise the second) is implemented as a sign flip on the
critic outputs before they are added to the total loss.

Consistency limit: when ``beta_plus == beta_minus == 0`` the IB term
vanishes and the strategy reduces to the parent VF strategy. The
Tier-1 audit ``ib_beta_positive`` warns in this case.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from spectramr.infrastructure.distributed.owned_modules import OWNED_MODULE
from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.training.strategies.mixins.utils import pick_present
from spectramr.infrastructure.training.strategies.virtual_fiducial_strategy import (
    ConcreteVirtualFiducialStrategy,
)
from spectramr.models.encoders.nav_encoder_1d import NavEncoder1D
from spectramr.models.losses.infonce_critic import InfoNCECritic

logger = logging.getLogger(__name__)


class IBVFTrainingStrategy(ConcreteVirtualFiducialStrategy):
    """VF + IB-regularised InfoNCE objective on (navigator, motion).

    Subclassing :class:`ConcreteVirtualFiducialStrategy` reuses the
    digital-twin simulator, the marker-injection pipeline, and the
    data-consistency loss. The new logic is confined to the loss
    aggregation: ``_ib_term`` is added to whatever the parent
    strategy already returned.
    """

    # parallel_compat: The two critics and the navigator encoder are owned modules, built in
    # __init__ on every rank and run via call_owned.
    parallel_compat = frozenset({"replicated"})

    @classmethod
    def owned_module_declarations(cls, config: Any) -> tuple[tuple[str, str], ...]:
        own = tuple((name, OWNED_MODULE) for name in ("critic_max", "critic_min", "nav_encoder"))
        return (*super().owned_module_declarations(config), *own)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cfg = getattr(self.config.training, "ib_vf", None)
        if cfg is None:
            raise ValueError(
                "IBVFTrainingStrategy requires training.ib_vf block. "
                "Add it to the YAML or switch to ConcreteVirtualFiducialStrategy."
            )
        # The mode-dispatch to the typed training schema does not fire for ib_vf
        # arms (config.training loads as the generic TrainingStrategyConfigSchema),
        # so the sub-block arrives as a raw dict whose ``.beta_plus`` attribute
        # access crashed with "'dict' object has no attribute 'beta_plus'"
        # (cluster smoke 20260605). Coerce to the SSOT schema — same dict→typed
        # gap the config health-checker already works around via
        # _typed_training_block.
        if isinstance(cfg, dict):
            from spectramr.config.schemas.training.vf_advanced import IBVFConfig

            cfg = IBVFConfig(**cfg)
        self.beta_plus = float(cfg.beta_plus)
        self.beta_minus = float(cfg.beta_minus)
        self.infonce_negatives_K = int(cfg.infonce_negatives_K)
        critic_kwargs = dict(
            temperature=float(cfg.infonce_temperature),
            critic_hidden_dim=int(cfg.critic_hidden_dim),
            negative_pool="momentum_bank" if cfg.memory_bank_size > 0 else "in_batch",
            bank_size=int(cfg.memory_bank_size),
            feature_dim=int(cfg.nav_latent_dim),
        )
        # Two critics — one for the maximisation term, one for the
        # minimisation (parameters are independent so each can specialise).
        self.critic_max = InfoNCECritic(**critic_kwargs)
        self.critic_min = InfoNCECritic(**critic_kwargs)

        # The 1-D navigator encoder is a STRATEGY-OWNED module, NOT the main
        # generator: the main generator reconstructs the 2-D image for the
        # parent VF loss path, while this encoder maps the DC-navigator sequence
        # m(t) -> z_m for the IB term. Wiring NavEncoder1D as the sole model
        # (config model_type=nav_encoder_1d) made the parent VF reconstruction
        # feed a 1-D encoder the full 2-D image -> "NavEncoder1D expected channel
        # dim 2, got (B, 8, 256, 256)" (cluster smoke 20260605, job 7095209).
        # Built here from the ib_vf block; its params join opt_g below.
        self.nav_encoder = NavEncoder1D(
            in_channels=2,
            nav_latent_dim=int(cfg.nav_latent_dim),
            depth=int(cfg.nav_encoder_depth),
        )

        # The critics + navigator encoder are strategy-owned learnable modules.
        # They must live on
        # the training device and their parameters must be registered with the
        # generator optimizer, otherwise the IB term contributes a gradient to
        # nothing and the InfoNCE bound never trains (CLAUDE.md #9 — a loss
        # term that updates no parameter is a silent no-op).
        self._register_ib_modules()

        logger.info(
            "IBVFTrainingStrategy initialised with beta_+=%.3f, beta_-=%.3f, K=%d, bank_size=%d.",
            self.beta_plus,
            self.beta_minus,
            self.infonce_negatives_K,
            cfg.memory_bank_size,
        )

    def _register_ib_modules(self) -> None:
        """Move the IB modules to ``self.device`` and register params on opt_g.

        Mirrors the lazy-builder registration pattern used across the
        VF strategies: collect the params already tracked by the generator
        optimizer, then add only the genuinely-new params (both InfoNCE critics
        and the navigator encoder) as a fresh param group so they are optimised
        alongside the generator.
        """
        opt = getattr(self.env, "opt_g", None)
        ib_modules = (self.critic_max, self.critic_min, self.nav_encoder)
        for module in ib_modules:
            module.to(self.device)
        for name in ("critic_max", "critic_min", "nav_encoder"):
            self.register_owned_module(name, getattr(self, name))
        if opt is None:
            logger.warning(
                "[IBVFTrainingStrategy] no generator optimizer (env.opt_g) found; "
                "InfoNCE critic + navigator-encoder params will NOT be trained."
            )
            return
        existing = {p for group in opt.param_groups for p in group["params"]}
        new = [p for module in ib_modules for p in module.parameters() if p not in existing]
        if new:
            opt.add_param_group({"params": new})

    def _ib_term(
        self,
        z_m: torch.Tensor,
        theta_hat: torch.Tensor,
        anatomy_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute :math:`L_{IB} = -\\beta_+ \\hat I(z_m;\\hat\\theta) +
        \\beta_- \\hat I(z_m; x)`.

        Args:
            z_m: navigator latent ``(B, d)``.
            theta_hat: motion vector ``(B, d_theta)``.
            anatomy_features: image-side features ``(B, d_x)``.

        Returns:
            Scalar tensor — the IB regulariser.
        """
        loss_max = self.call_owned("critic_max", z_m, theta_hat)
        loss_min = self.call_owned("critic_min", z_m, anatomy_features)
        # critic returns -lower_bound; for maximisation we flip the sign.
        return -self.beta_plus * (-loss_max) + self.beta_minus * (-loss_min)

    def _ib_features(
        self, target_complex: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Derive ``(z_m, theta_hat, anatomy_features)`` for the IB term.

        The navigator is the DC k-space line :math:`m(t) = y(0, \cdot)` of the
        corrupted acquisition. We obtain the corrupted image from the same
        digital-twin simulator the parent VF strategy uses (SSOT), take a
        centred FFT, read the central readout line as the per-sample navigator,
        and encode it with the generator (:class:`NavEncoder1D`) to get
        :math:`z_m \in \mathbb{R}^{B \times d}`.

        - ``theta_hat`` is the motion proxy: the encoding of the navigator
          *residual* (corrupted minus clean DC line), which carries the rigid
          motion signature the IB term should make :math:`z_m` maximally
          informative about.
        - ``anatomy_features`` is the encoding of the *clean* navigator — the
          nuisance content the IB term should squeeze out of :math:`z_m`.

        All three are :math:`(B, d)` and flow gradients to both the generator
        and the InfoNCE critics. Returns ``None``-free tensors on ``self.device``.
        """
        # Run the SSOT simulator (same instance/config as the parent loss path)
        # to obtain the corrupted acquisition; never re-parse config or build a
        # second simulator.
        corrupted_image, _marker_prior, joint_clean = self.simulator(target_complex)

        # Centred k-space of corrupted and clean acquisitions; fft2c handles
        # [B, C, H, W] (operates on the trailing two dims). Read the central
        # readout line (DC line) as the 1-D navigator m(t) per sample.
        k_corrupt = fft2c(corrupted_image)
        k_clean = fft2c(joint_clean)
        h = k_corrupt.shape[-2]
        dc_row = h // 2
        nav_corrupt = k_corrupt[..., dc_row, :]  # [B, C, W] complex
        nav_clean = k_clean[..., dc_row, :]
        # Collapse channels into the temporal axis so the navigator is [B, T].
        nav_corrupt = nav_corrupt.reshape(nav_corrupt.shape[0], -1)
        nav_clean = nav_clean.reshape(nav_clean.shape[0], -1)
        nav_residual = nav_corrupt - nav_clean

        # Encode with the strategy-owned 1-D navigator encoder (NOT the main
        # generator, which reconstructs the 2-D image for the parent VF path).
        # NavEncoder1D accepts a complex (B, T) sequence and splits real/imag.
        z_m = self.call_owned("nav_encoder", nav_corrupt)
        theta_hat = self.call_owned("nav_encoder", nav_residual)
        anatomy_features = self.call_owned("nav_encoder", nav_clean)
        return z_m, theta_hat, anatomy_features

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor | None = None,
        target_batch: torch.Tensor | None = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """VF reconstruction losses + the IB / InfoNCE regulariser.

        Forwards to the parent VF loss path (NAMED kwargs, per the canonical
        contract) for the reconstruction + marker terms, then adds
        :math:`L_{IB}` computed from the navigator latent ``z_m``, the motion
        proxy ``theta_hat`` and the anatomy nuisance ``anatomy_features``.
        """
        # Resolve a dict batch (legacy callers) and forward the available
        # tensors to the parent under the canonical contract.
        resolved = self._resolve_legacy_batch(input_batch, kwargs)
        if isinstance(resolved, dict):
            kwargs.setdefault("batch", resolved)

        base = super()._compute_losses_impl(
            input_batch=input_batch,
            target_batch=target_batch,
            epoch=epoch,
            **kwargs,
        )

        # Reconstruct the same complex target the parent consumed (nested
        # config only; no re-parse of YAML).
        target = pick_present(target_batch, input_batch)
        if target is None:
            return base
        if target.ndim == 5:
            mid = target.shape[-1] // 2
            target = target[..., mid]
        target_complex = self._to_complex(target)
        if hasattr(self.config.data, "dataset_type") and self.config.data.dataset_type == "kspace":
            from spectramr.infrastructure.physics.fft_ops import ifft2c

            target_complex = ifft2c(target_complex)

        z_m, theta_hat, anatomy_features = self._ib_features(target_complex)
        ib_loss = self._ib_term(z_m, theta_hat, anatomy_features)

        # Component entries are detached (parent convention); the live IB tensor
        # is folded into the total so its gradient reaches the generator + critics.
        base["loss_ib"] = ib_loss.detach()
        for total_key in ("g_total_loss", "g_loss_total", "loss_total", "total"):
            if total_key in base:
                base[total_key] = base[total_key] + ib_loss
                break
        return base


__all__ = ["IBVFTrainingStrategy"]
