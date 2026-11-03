"""k-Space Implicit Neural Representation (ULF-PR-4 / II.1).

A SIREN coordinate MLP ``f_theta(k_x, k_y) -> (Re, Im)`` per coil represents one
slice's multi-coil k-space. Each slice gets its own fit: starting from the
network's weights, ``training.kspace_inr.inner_steps`` gradient steps minimise the
mask-weighted k-space MSE on the sampled points of ``measured_kspace``. The fitted
field is then taken to the image domain (``ifft2c`` per coil, RSS) and scored
against the reference by the declared ``losses.image_losses``, plus
``lambda_dc`` times the fitted field's k-space residual. The outer optimiser
therefore learns the initialisation the per-slice fit starts from (Tancik et al.
2021, learned initialisations for coordinate networks; Sitzmann et al. 2020).

Validation runs the same per-slice fit, so the graded image is what the INR
recovers from that slice's measurement alone.

``coord_kspace_gen`` also evaluates a SIREN over k-space, but as one population
field conditioned on contrast with a radial-moment prior; it shares only the
coordinate convention (``kspace_coordinate_grid``) with this strategy.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import torch
from torch.func import functional_call, vmap

from spectramr.infrastructure.physics.fft_ops import ifft2c
from spectramr.infrastructure.physics.multicoil_ei_operator import interleaved_to_complex
from spectramr.models.blocks.siren_inr import kspace_coordinate_grid

from .reconstruction import ReconstructionTrainingStrategy

logger = logging.getLogger(__name__)


class KSpaceINRStrategy(ReconstructionTrainingStrategy):
    """Per-slice SIREN k-space fit with a learned initialisation and an image-domain term."""

    #: Loss ownership (issue #1918): the image-domain term is the builder's declared
    #: image losses on RSS(ifft2c(k)); the k-space DC term is not a canonical image loss.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = True
    #: The fit reads the measurement and mask off the context, in validation too.
    validates_on_batch_context: ClassVar[bool] = True
    validation_reads_context_measurement = True
    #: The network is handed coordinates, never the prepared input image; the audit's
    #: coil-channel alignment reads this.
    model_input_is_coordinates: ClassVar[bool] = True
    # parallel_compat: the fit evaluates the bare generator through functional_call, outside
    # any wrapper's forward, so DDP never all-reduces and a sharded backend reads shards.
    parallel_compat = frozenset()

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        training = getattr(self.config, "training", None)
        cfg = getattr(training, "kspace_inr", None) if training is not None else None
        if cfg is None:
            raise ValueError(
                "KSpaceINRStrategy requires a training.kspace_inr block: the per-slice fit's "
                "inner_steps, inner_lr, lambda_dc and first_order have no implicit values."
            )
        self.inner_steps = int(cfg.inner_steps)
        self.inner_lr = float(cfg.inner_lr)
        self.lambda_dc = float(cfg.lambda_dc)
        self.first_order = bool(cfg.first_order)
        self._grids: dict[tuple[int, int, torch.device], torch.Tensor] = {}
        logger.info(
            "KSpaceINRStrategy: per-slice SIREN k-space fit, inner_steps=%d, inner_lr=%.3g, "
            "lambda_dc=%.3g, first_order=%s; image term = declared image_losses on RSS(ifft2c(k)).",
            self.inner_steps,
            self.inner_lr,
            self.lambda_dc,
            self.first_order,
        )

    def builds_own_model_input(self) -> bool:
        """The network is evaluated on k-space coordinates, not the loader's input."""
        return True

    @classmethod
    def standalone_infer_blocker(cls, config: object) -> str | None:
        return (
            "kspace_inr fits the SIREN to each slice's measured k-space inside the step; "
            "infer would feed the image to a coordinate network"
        )

    def _measurement(self, context: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """``(y complex [B, C, H, W], mask [B, 1, H, W])`` off the batch context."""
        measured = context.get("measured_kspace")
        mask = context.get("mask")
        if measured is None or mask is None:
            raise ValueError(
                "kspace_inr requires batch_context['measured_kspace'] and ['mask']: route the "
                "arm through the loader's Cartesian masking (data.trajectory: cartesian)."
            )
        measured, mask = _squeeze_unit_depth(measured), _squeeze_unit_depth(mask)
        y = interleaved_to_complex(measured)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        if mask.shape[-2:] != y.shape[-2:] or mask.shape[1] != 1:
            raise ValueError(
                f"kspace_inr mask {tuple(mask.shape)} does not broadcast over the measurement "
                f"{tuple(y.shape)} as one Cartesian mask per slice."
            )
        return y, mask.to(y.real.dtype)

    def _grid(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        key = (height, width, device)
        if key not in self._grids:
            self._grids[key] = kspace_coordinate_grid(height, width, device)
        return self._grids[key]

    def _generator(self, coils: int) -> torch.nn.Module:
        gen = getattr(self.env, "generator", None)
        if gen is None:
            raise RuntimeError("kspace_inr requires self.env.generator (the SIREN field).")
        out_channels = getattr(gen, "out_channels", None)
        if out_channels != 2 * coils:
            raise ValueError(
                f"kspace_inr: the generator emits {out_channels} channels but the measurement "
                f"carries {coils} coils; set model.out_channels: {2 * coils} (Re, Im per coil)."
            )
        return gen

    def _field(
        self, gen: torch.nn.Module, params: dict[str, torch.Tensor], coords: torch.Tensor, shape
    ) -> torch.Tensor:
        """Per-slice parameters ``[B, ...]`` -> complex k-space ``[B, C, H, W]``."""
        batch, coils, height, width = shape
        out = vmap(lambda p: functional_call(gen, p, (coords,)))(params)
        out = out.view(batch, height, width, coils, 2)
        return torch.complex(out[..., 0], out[..., 1]).permute(0, 3, 1, 2)

    @staticmethod
    def _dc(pred_k: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-slice mean squared k-space residual over the sampled points, ``[B]``."""
        residual = (pred_k - y).abs().pow(2) * mask
        sampled = (mask.sum(dim=(1, 2, 3)) * y.shape[1]).clamp(min=1.0)
        return residual.sum(dim=(1, 2, 3)) / sampled

    def _fit(
        self, y: torch.Tensor, mask: torch.Tensor, *, second_order: bool | None = None
    ) -> torch.Tensor:
        """Fit the field to each slice's sampled k-space from the network's weights.

        The field represents ``y / s`` with ``s`` the RMS of the slice's sampled points,
        so ``inner_lr`` and ``lambda_dc`` do not depend on the scanner's k-space scale.
        """
        gen = self._generator(y.shape[1])
        coords = self._grid(y.shape[-2], y.shape[-1], y.device)
        scale = self._dc(torch.zeros_like(y), y, mask).sqrt().clamp(min=1e-12).detach()
        scale = scale.view(-1, 1, 1, 1)
        y_unit = y / scale
        create_graph = (not self.first_order) if second_order is None else second_order
        batch = y.shape[0]
        theta = {n: p.unsqueeze(0).expand(batch, *p.shape) for n, p in gen.named_parameters()}
        for _ in range(self.inner_steps):
            # Summed over slices: each slice's parameters see only their own residual.
            inner = self._dc(self._field(gen, theta, coords, y.shape), y_unit, mask).sum()
            grads = torch.autograd.grad(inner, list(theta.values()), create_graph=create_graph)
            theta = {
                n: t - self.inner_lr * g for (n, t), g in zip(theta.items(), grads, strict=True)
            }
        return scale * self._field(gen, theta, coords, y.shape)

    @staticmethod
    def _image(pred_k: torch.Tensor) -> torch.Tensor:
        """RSS magnitude of the per-coil ``ifft2c`` images, ``[B, 1, H, W]``."""
        power = ifft2c(pred_k).abs().pow(2).sum(dim=1, keepdim=True)
        return power.clamp(min=1e-12).sqrt()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        context = self._prepare_batch_context(input_batch, target_batch, **kwargs)
        y, mask = self._measurement(context)
        pred_k = self._fit(y, mask)
        scale2 = self._dc(torch.zeros_like(y), y, mask).clamp(min=1e-24)
        dc = (self._dc(pred_k, y, mask) / scale2).mean()
        env_losses: dict[str, Any] = (getattr(self.env, "losses", None) or {}) if self.env else {}
        image_out = self.loss_computer.compute(
            pred=self._image(pred_k),
            target=context["hr"],
            epoch=epoch,
            iteration=int(kwargs.get("iteration", 0) or 0),
            losses_dict=env_losses,
        )
        total = image_out.total + self.lambda_dc * dc
        return {
            **image_out.components,
            "loss_total": total,
            "loss_image": image_out.total.detach(),
            "loss_kspace_dc": dc.detach(),
        }

    def _validation_forward(
        self,
        input_batch: torch.Tensor,
        batch_context: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        """The image the per-slice fit recovers from this slice's measurement."""
        y, mask = self._measurement(batch_context)
        # Validation runs under no_grad; the fit itself is gradient descent, and no outer
        # gradient is taken, so the inner loop never builds a second-order graph.
        with torch.enable_grad():
            pred_k = self._fit(y, mask, second_order=False)
        return self._image(pred_k).detach()


def _squeeze_unit_depth(x: torch.Tensor) -> torch.Tensor:
    """Drop a trailing singleton slice axis (``[B, C, H, W, 1]`` -> ``[B, C, H, W]``)."""
    if x.dim() == 5:
        if x.shape[-1] != 1:
            raise ValueError(f"kspace_inr is a 2-D fit; got a {x.shape[-1]}-slice tensor.")
        return x.squeeze(-1)
    return x
