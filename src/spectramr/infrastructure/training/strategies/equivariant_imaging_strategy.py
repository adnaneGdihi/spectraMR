r"""Equivariant Imaging (EI) reconstruction strategy — Phase A step 3 (un-trap P1.2).

Equivariant Imaging (Chen et al., ICCV 2021) trains a reconstruction network
from undersampled measurements **with no fully-sampled target**. The registered
:class:`~spectramr.models.losses.equivariant_recon_loss.EquivariantSSLReconLoss`
already encodes the equivariance penalty, but it was an **inert facade**: no
strategy emitted ``context["transformed_recon"]`` (the reference branch
:math:`T_g f(A^* y)`), so the term could never fire on a real run (pitfall #16).
This strategy makes it fire.

Each step computes, for a forward operator :math:`A = M F` built from the
acquisition mask through the complex physics SSOT (``fft2c`` / ``ifft2c``):

* the reconstruction :math:`\hat{x} = f(A^* y)`;
* a measurement-consistency anchor :math:`\lVert A\hat{x} - y\rVert^2`
  (or the nc-χ-aware GSURE risk of the T1 keystone when
  ``robust_correction`` is set — that is the *Robust EI* / ``robust_ei`` arm,
  Chen et al., CVPR 2022); and
* the equivariance term :math:`\lVert f(A T_g \hat{x}) - T_g \hat{x}\rVert`
  for a group element :math:`g` sampled from a brain-appropriate group
  (small-rotation / dihedral, **not** full ``SO(2)``).

The consistency anchor is essential: equivariance alone admits the trivial
solution :math:`f \equiv \text{const}`.

The group :math:`T_g` acts on the real-valued recon; the forward operator
round-trips through complex k-space. Identifiability (Tachella et al.,
sensing theorems) requires :math:`A` to break the group symmetry, which the
``ei_sensing_margin`` audit enforces. Corner case: ``alpha_equivariance == 0``
reduces this to a measurement-consistency reconstruction.

This strategy assumes a single (coil-combined) complex forward operator
:math:`A = M F`; multi-coil EI (:math:`A = M F S` via ``sense_forward``) is a
future extension.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, ClassVar

import torch

from spectramr.infrastructure.physics.fft_ops import fft2c
from spectramr.infrastructure.physics.group_actions import GroupAction, get_group_action
from spectramr.infrastructure.training.builders.environment import TrainingEnvironment
from spectramr.infrastructure.training.strategies.reconstruction import (
    ReconstructionTrainingStrategy,
)
from spectramr.models.losses.equivariant_recon_loss import EquivariantSSLReconLoss
from spectramr.models.losses.sure_n2self_losses import GSUREKspaceLoss

logger = logging.getLogger(__name__)


def _to_complex_image(x: torch.Tensor) -> torch.Tensor:
    """Coerce a real image to complex (zero imaginary part); pass complex through."""
    return x if x.is_complex() else torch.complex(x, torch.zeros_like(x))


def equivariant_imaging_branches(
    reconstruct: Callable[[torch.Tensor], torch.Tensor],
    forward_op: Callable[[torch.Tensor], torch.Tensor],
    group: GroupAction,
    measured_kspace: torch.Tensor,
    g_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute the two EI branches for one group element.

    Args:
        reconstruct: maps measured k-space :math:`y` to a recon image
            :math:`f(A^* y)` (the network, including the adjoint/zero-fill).
        forward_op: the complex masked-FFT forward operator :math:`A = M F`.
        group: the symmetry group supplying :math:`T_g`.
        measured_kspace: the undersampled measurement :math:`y` (complex).
        g_index: the group element index (``0`` is the identity).

    Returns:
        ``(x_hat, prediction, transformed_recon)`` where
        ``x_hat`` :math:`= f(A^* y)`,
        ``transformed_recon`` :math:`= T_g \hat{x}` (the reference branch), and
        ``prediction`` :math:`= f(A^* A T_g \hat{x})` (the re-measured branch).
        The EI penalty is :math:`\lVert prediction - transformed\_recon\rVert`.
    """
    x_hat = reconstruct(measured_kspace)  # f(A^* y)
    transformed_recon = group.apply(x_hat, g_index)  # T_g x_hat  (reference)
    re_measured = forward_op(transformed_recon)  # A T_g x_hat  (complex k-space)
    prediction = reconstruct(re_measured)  # f(A^* A T_g x_hat)
    return x_hat, prediction, transformed_recon


class EquivariantImagingStrategy(ReconstructionTrainingStrategy):
    r"""Self-supervised reconstruction via Equivariant Imaging.

    Reuses the reconstruction plumbing (k-space → image conversion, generator
    forward) and overrides only the loss: a measurement-consistency anchor plus
    the equivariance penalty routed through the registered
    ``EquivariantSSLReconLoss``. No fully-sampled target is read.

    The ``robust_ei`` registry key selects the *Robust EI* arm; it requires
    ``equivariant_imaging.robust_correction = true`` (the key is a promise, the
    knob must honour it — a contradiction raises rather than silently degrading).
    """

    #: Loss ownership (issue #1918). Self-supervised equivariance/measurement-consistency objective -- no ground
    #: truth is used, so no declared image loss can be or is computed.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(
        self,
        env: TrainingEnvironment | None = None,
        device: torch.device | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, device=device, **kwargs)
        cfg = (
            getattr(self.config.training, "equivariant_imaging", None)
            if self.config.training
            else None
        )
        if cfg is None:
            raise ValueError(
                "EquivariantImagingStrategy requires a "
                "training.equivariant_imaging config block (group, "
                "alpha_equivariance, ...). None was provided."
            )
        self.group_name = str(cfg.group)
        self.alpha = float(cfg.alpha_equivariance)
        self.n_group_samples = int(cfg.n_group_samples)
        self.robust = bool(cfg.robust_correction)
        self.noise_model = str(cfg.noise_model)
        self.n_coils = int(cfg.n_coils)
        self.noise_std = cfg.noise_std_estimate

        group_kwargs: dict[str, Any] = {}
        if self.group_name == "small_rotation":
            group_kwargs = {
                "max_angle_deg": float(cfg.max_angle_deg),
                "num_angles": int(cfg.num_angles),
            }
        self._group = get_group_action(self.group_name, **group_kwargs)

        # The ``robust_ei`` key advertises the Jensen/SURE noise correction;
        # refuse the contradiction of selecting it with the correction off
        # (pitfall #9 — a registry key that silently does nothing is a facade).
        # Both keys map to the same dotted strategy_class, so the distinguishing
        # signal is training_mode.
        training_mode = str(getattr(self.config.training, "training_mode", "") or "").lower()
        if training_mode == "robust_ei" and not self.robust:
            raise ValueError(
                "training_mode 'robust_ei' requires "
                "training.equivariant_imaging.robust_correction=true; "
                "use 'equivariant_imaging' for the non-robust arm."
            )

        # Robust EI consumes the T1 keystone GSURE risk as its data-consistency
        # term; build it now so a missing noise estimate fails at construction.
        self._gsure: GSUREKspaceLoss | None = None
        if self.robust:
            if self.noise_std is None:
                raise ValueError(
                    "equivariant_imaging.robust_correction=true requires "
                    "noise_std_estimate (the k-space noise σ); none was set "
                    "(pitfall #15: the knob must be wired)."
                )
            self._gsure = GSUREKspaceLoss(
                sigma=float(self.noise_std),
                noise_model=self.noise_model,
                n_coils=self.n_coils,
            )

        self._owned_ei_loss: EquivariantSSLReconLoss | None = None
        self._rng = torch.Generator(device="cpu").manual_seed(int(cfg.split_seed))
        logger.info(
            "EquivariantImagingStrategy: group=%s, alpha=%.3g, n_samples=%d, "
            "robust=%s, noise_model=%s",
            self.group_name,
            self.alpha,
            self.n_group_samples,
            self.robust,
            self.noise_model,
        )

    def _setup_strategy_specific_components(self) -> None:
        self._verify_strategy_config(
            expected_modes=(
                "reconstruction",
                "self_supervised_reconstruction",
                "equivariant_imaging",
                "robust_ei",
            )
        )
        if self.logging_service:
            self._log_config_features(self.logging_service)

    def _resolve_equivariant_loss(self) -> EquivariantSSLReconLoss:
        """Route through the registered EI loss (the facade target).

        Prefer the loss built from config; otherwise the strategy owns the
        canonical registered loss so the equivariance term always fires.
        """
        env_losses: dict[str, Any] = {}
        if self.env and hasattr(self.env, "losses"):
            env_losses = self.env.losses or {}
        for fn in env_losses.values():
            if isinstance(fn, EquivariantSSLReconLoss):
                return fn
        if self._owned_ei_loss is None:
            self._owned_ei_loss = EquivariantSSLReconLoss(norm="l2")
        return self._owned_ei_loss

    def _compute_losses_impl(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        epoch: int,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        del epoch  # EI is self-supervised; no epoch-gated schedule here.
        batch_context = self._prepare_batch_context(input_batch, target_batch, **kwargs)
        y = batch_context.get("measured_kspace")
        mask = batch_context.get("mask")
        if y is None or mask is None:
            raise ValueError(
                "EquivariantImagingStrategy requires batch_context['measured_kspace'] "
                "(undersampled k-space y) and batch_context['mask'] (acquisition "
                "mask). EI is self-supervised — there is no fully-sampled target. "
                "Ensure the dataset/director provides both."
            )

        mask_c = mask if mask.is_complex() else mask.to(torch.complex64)

        def forward_op(x: torch.Tensor) -> torch.Tensor:
            return mask_c * fft2c(_to_complex_image(x))

        def reconstruct(kspace: torch.Tensor) -> torch.Tensor:
            bc = dict(batch_context)
            bc["measured_kspace"] = kspace
            bc["mask"] = mask
            lr_image, forward_kwargs = self._prepare_generator_inputs(bc, kspace)
            recon, _ = self._generate_predictions(lr_image, forward_kwargs, bc)
            return recon

        ei_loss_fn = self._resolve_equivariant_loss()

        x_hat_ref: torch.Tensor | None = None
        ei_terms: list[torch.Tensor] = []
        last_g = 0
        for _ in range(max(1, self.n_group_samples)):
            g = self._group.sample(self._rng)
            last_g = g
            x_hat, prediction, transformed = equivariant_imaging_branches(
                reconstruct, forward_op, self._group, y, g
            )
            if x_hat_ref is None:
                x_hat_ref = x_hat
            ei_terms.append(
                ei_loss_fn(
                    prediction=prediction,
                    target=target_batch,
                    context={"transformed_recon": transformed},
                )
            )
        equivariance = torch.stack(ei_terms).mean()

        assert x_hat_ref is not None  # loop runs at least once
        a_xhat = forward_op(x_hat_ref)
        if self.robust:
            assert self._gsure is not None
            consistency = self._gsure(a_xhat, y)
        else:
            resid = a_xhat - y
            consistency = (resid.conj() * resid).real.mean()

        total = consistency + self.alpha * equivariance
        return {
            "g_total_loss": total,
            "loss": total,
            "loss_consistency": consistency.detach(),
            "loss_equivariance": equivariance.detach(),
            "ei_group_element": torch.tensor(float(last_g), device=total.device),
        }


__all__ = ["EquivariantImagingStrategy", "equivariant_imaging_branches"]
