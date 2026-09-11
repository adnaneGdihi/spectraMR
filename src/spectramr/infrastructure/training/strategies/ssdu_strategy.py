"""Self-Supervised learning via Data Undersampling (SSDU) reconstruction strategy.

Trains an unrolled/recon generator from undersampled k-space alone -- no
fully-sampled reference -- by splitting the acquired sample into a network-input
partition Lambda and a held-out loss partition Theta (Yaman et al., MRM 2020).
The held-out residual is scored by the already-registered ``ssdu`` /
``multi_mask_ssdu`` losses.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)

logger = logging.getLogger(__name__)


def split_acquired_mask(
    acquired_mask: torch.Tensor,
    theta_fraction: float = 0.4,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split an acquired-sample mask into disjoint Lambda (input) and Theta (loss) masks.

    Exactly ``round(theta_fraction * n_acquired)`` of the *acquired* k-space
    points are moved into Theta, selected by random key (rank-based) so the
    fraction is correct regardless of the sampling pattern.

    Args:
        acquired_mask: real tensor, 1 where k-space was sampled, broadcastable
            to ``(B, 1, H, W)`` / ``(1, 1, H, W)`` / ``(H, W)``.
        theta_fraction: fraction of acquired points held out into Theta.
        generator: optional RNG for reproducible splits.

    Returns:
        ``(lambda_mask, theta_mask)`` with ``lambda + theta == acquired`` and
        ``lambda * theta == 0``.
    """
    if not 0.0 < theta_fraction < 1.0:
        raise ValueError(f"theta_fraction must be in (0, 1), got {theta_fraction}")
    acq = (acquired_mask > 0).to(acquired_mask.dtype)
    n_acq = int(acq.sum().item())
    theta = torch.zeros_like(acq)
    if n_acq == 0:
        return acq.clone(), theta
    n_theta = round(theta_fraction * n_acq)
    if n_theta == 0:
        return acq.clone(), theta
    # Random key per cell; push non-acquired cells to +inf so they never rank in.
    key = torch.rand(acq.shape, generator=generator).to(acq.device)
    key = torch.where(acq > 0, key, torch.full_like(key, float("inf")))
    flat = key.reshape(-1)
    theta_idx = torch.topk(flat, n_theta, largest=False).indices
    theta_flat = theta.reshape(-1)
    theta_flat[theta_idx] = 1.0
    theta = theta_flat.reshape(acq.shape)
    lam = acq - theta
    return lam, theta


def inject_noisier_kspace(
    target_kspace: torch.Tensor,
    acquired_mask: torch.Tensor,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add synthetic complex-Gaussian noise on the acquired support (Robust SSDU).

    Implements the Noisier2Noise (Millard & Chiew 2024) input augmentation: the
    network sees a *noisier* measurement ``z = y + ñ`` with
    ``ñ ~ CN(0, noise_std²)`` on the acquired k-space points, while the held-out
    loss target stays the original (less-noisy) ``y``. At ``noise_std == 0`` the
    function is the identity, so Robust SSDU reduces exactly to vanilla SSDU.

    Args:
        target_kspace: acquired (complex) measurement ``y``.
        acquired_mask: real mask, 1 where k-space was sampled.
        noise_std: magnitude std σ of the added complex-Gaussian noise.
        generator: optional RNG for reproducibility.

    Returns:
        The noisier measurement ``z`` (same shape/dtype as ``target_kspace``).
    """
    if noise_std <= 0.0:
        return target_kspace
    real_dtype = target_kspace.real.dtype
    acq = (acquired_mask > 0).to(real_dtype)
    real = torch.randn(target_kspace.shape, generator=generator).to(target_kspace.device)
    imag = torch.randn(target_kspace.shape, generator=generator).to(target_kspace.device)
    # Magnitude std == noise_std => per-component std == noise_std / sqrt(2).
    noise = (noise_std / (2.0**0.5)) * torch.complex(real.to(real_dtype), imag.to(real_dtype))
    return target_kspace + noise * acq


class SSDUReconstructionStrategy(ReconstructionTrainingStrategy):
    """Reference-free reconstruction via SSDU k-space self-supervision.

    Reuses ``ReconstructionTrainingStrategy`` plumbing (multi-slice handling,
    k-space->image conversion, generator forward) and overrides only the loss:
    the acquired mask is split into Lambda (network input) and Theta (held-out
    target); the generator reconstructs from the Lambda-undersampled k-space;
    the held-out residual ``||M_Theta(F x_hat - y)||`` is scored by the
    registered ``ssdu`` loss.
    """

    #: Loss ownership (issue #1918). Iterates env.losses (route 5) but the loop body is `if "ssdu" in name`:
    #: every declared image loss whose name lacks that substring is SKIPPED. The
    #: textual route marker cannot see the filter, so the flag -- not the pin --
    #: is what records that this strategy folds only its own ssdu terms.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        # Typed SSDUTrainingConfigSchema (frozen, extra="forbid"); read the
        # validated fields directly so a renamed/missing field raises instead
        # of silently reverting to the default (pitfall #15).
        cfg = getattr(self.config.training, "ssdu", None) if self.config.training else None
        if cfg is not None:
            self.theta_fraction = float(cfg.theta_fraction)
            self.num_masks = int(cfg.num_masks)
            seed = int(cfg.split_seed)
            self.noisier2noise = bool(cfg.noisier2noise_correction)
            self.noise_std = cfg.noise_std_estimate
            self.noise_model = str(cfg.noise_model)
            self.n_coils = int(cfg.n_coils)
        else:
            self.theta_fraction = 0.4
            self.num_masks = 1
            seed = 0
            self.noisier2noise = False
            self.noise_std = None
            self.noise_model = "gaussian_kspace"
            self.n_coils = 1

        # Robust SSDU correction requires a noise estimate (pitfall #15); the
        # schema validator already enforces this, but guard at construction too.
        if self.noisier2noise and self.noise_std is None:
            raise ValueError(
                "Robust SSDU (noisier2noise_correction=true) requires "
                "ssdu.noise_std_estimate (the k-space noise σ); none was set."
            )
        # The 'robust_ssdu' key advertises the Noisier2Noise correction; refuse
        # the contradiction of selecting it with the correction off (pitfall #16).
        training_mode = str(getattr(self.config.training, "training_mode", "") or "").lower()
        if training_mode == "robust_ssdu" and not self.noisier2noise:
            raise ValueError(
                "training_mode 'robust_ssdu' requires "
                "ssdu.noisier2noise_correction=true; use 'ssdu' for vanilla SSDU."
            )

        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(seed)
        logger.info(
            "SSDUReconstructionStrategy: theta_fraction=%.3g, num_masks=%d, "
            "split_seed=%d, noisier2noise=%s (source=%s)",
            self.theta_fraction,
            self.num_masks,
            seed,
            self.noisier2noise,
            "config" if cfg is not None else "defaults",
        )

    def _setup_strategy_specific_components(self) -> None:
        # Accept the SSDU mode in addition to the recon-family modes.
        self._verify_strategy_config(
            expected_modes=(
                "reconstruction",
                "mae_pretraining",
                "volumetric",
                "self_supervised_reconstruction",
                "ssdu",
                "robust_ssdu",
            )
        )
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)

        acquired = batch_context.get("mask")
        target_kspace = batch_context.get("measured_kspace")
        if acquired is None or target_kspace is None:
            raise ValueError(
                "SSDU requires batch_context['mask'] (acquired sample mask) and "
                "batch_context['measured_kspace'] (original k-space y). Ensure the "
                "dataset/director provides both -- SSDU has no fully-sampled target."
            )

        # Generate ``num_masks`` independent Lambda/Theta splits. The network
        # reconstructs once from the first split's Lambda; the held-out Theta
        # partitions are scored by the SSDU loss(es). This wires the previously
        # silent ``num_masks`` knob and supplies the ``theta_masks`` list that
        # ``multi_mask_ssdu`` requires (it raised on the missing key).
        splits = [
            split_acquired_mask(acquired, theta_fraction=self.theta_fraction, generator=self._rng)
            for _ in range(max(1, self.num_masks))
        ]
        lam, theta = splits[0]
        theta_masks = [t for (_, t) in splits]

        # Build the Lambda-undersampled k-space the network sees. Robust SSDU
        # (Noisier2Noise): feed a NOISIER measurement to the network while the
        # held-out Theta loss target below stays the ORIGINAL (less-noisy)
        # ``target_kspace``. At noise_std→0 ``network_kspace == target_kspace``,
        # so this reduces exactly to vanilla SSDU.
        mask_dtype = target_kspace.real.dtype if target_kspace.is_complex() else target_kspace.dtype
        network_kspace = target_kspace
        if self.noisier2noise:
            network_kspace = inject_noisier_kspace(
                target_kspace, acquired, float(self.noise_std), generator=self._rng
            )
        lam_k = network_kspace * lam.to(mask_dtype)
        batch_context["measured_kspace"] = lam_k
        batch_context["mask"] = lam

        lr_image, forward_kwargs = self._prepare_generator_inputs(batch_context, lam_k)
        hr_fakes, _ = self._generate_predictions(lr_image, forward_kwargs, batch_context)

        env_losses: dict[str, Any] = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}

        # Route through the registered SSDU loss(es) via context. The single
        # ``ssdu`` loss reads ``theta_mask`` (the first split); ``multi_mask_ssdu``
        # reads the full ``theta_masks`` list (single-prediction fallback).
        context = {
            "target_kspace": target_kspace,
            "theta_mask": theta,
            "theta_masks": theta_masks,
        }
        total = hr_fakes.new_zeros(())
        components: dict[str, torch.Tensor] = {}
        ran_ssdu = False
        for name, loss_fn in env_losses.items():
            if "ssdu" in name:
                val = loss_fn(hr_fakes, target_batch, context=context)
                total = total + val
                components[f"loss_{name}"] = val
                ran_ssdu = True
        if not ran_ssdu:
            raise ValueError(
                "SSDU strategy requires an 'ssdu' or 'multi_mask_ssdu' loss in "
                "config.objectives -- none was built. (pitfall #9: no silent fallback.)"
            )

        result: dict[str, torch.Tensor] = {"g_total_loss": total, "loss": total}
        result.update(components)
        return result
