"""GAN loss library with strategies and a composite interface.

Provides both a flexible strategy-based system and a small compatibility
facade used by tests (CompositeLoss, LossConfig, etc.).
"""

import logging

logger = logging.getLogger(__name__)

import enum
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import autograd, nn

# Import SSOT Perceptual Loss
from spectramr.models.losses.perceptual_loss import PerceptualLoss
from spectramr.models.losses.registry import register_loss

# Adversarial loss strategies


class AdversarialLossStrategy(nn.Module):
    """AdversarialLossStrategy class."""

    def compute_generator_loss(self, fake_outputs_d, **kwargs):
        # TODO: Subclasses must implement this method
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        raise NotImplementedError(
            "Subclasses of AdversarialLossStrategy must implement "
            "`compute_generator_loss` to define how generator loss is calculated.",
        )

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        # TODO: Subclasses must implement this method
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        raise NotImplementedError(
            "Subclasses of AdversarialLossStrategy must implement "
            "`compute_discriminator_loss` to define how discriminator loss is calculated.",
        )

    def forward(self, *args, **kwargs):
        """Dummy forward to handle being called like a standard loss (pred, target)
        during validation loops."""
        device = (
            args[0].device if args and isinstance(args[0], torch.Tensor) else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)


@register_loss(
    name="gan_standard", aliases=["gan_vanilla", "gan_bce", "StandardGANLoss"], domain="agnostic"
)
class StandardGANLoss(AdversarialLossStrategy):
    """Vanilla GAN with BCEWithLogitsLoss and optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        """__init__.

        Args:
            label_smoothing (float): Description.
        """
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.label_smoothing = label_smoothing

    def compute_generator_loss(self, fake_outputs_d, **kwargs):
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        real_target = 1.0 - self.label_smoothing
        # Ensure loss function is on the same device as inputs
        if hasattr(self.loss, "to"):
            self.loss = self.loss.to(fake_outputs_d.device)
        return self.loss(fake_outputs_d, torch.full_like(fake_outputs_d, real_target))

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        real_target = 1.0 - self.label_smoothing
        fake_target = 0.0 + self.label_smoothing
        # Ensure loss function is on the same device as inputs
        if hasattr(self.loss, "to"):
            self.loss = self.loss.to(real_outputs_d.device)
        real_loss = self.loss(
            real_outputs_d,
            torch.full_like(real_outputs_d, real_target),
        )
        fake_loss = self.loss(
            fake_outputs_d,
            torch.full_like(fake_outputs_d, fake_target),
        )

        # Safety check for numerical stability
        if not torch.isfinite(real_loss):
            real_loss = torch.tensor(0.0, device=real_outputs_d.device)
        if not torch.isfinite(fake_loss):
            fake_loss = torch.tensor(0.0, device=fake_outputs_d.device)

        return real_loss, fake_loss


@register_loss(name="r1_regularization", aliases=["R1RegularizationLoss"], domain="agnostic")
class R1RegularizationLoss(nn.Module):
    """R1 Regularization for GANs.

    Penalizes gradients of the discriminator to stabilize training.
    Loss = 0.5 * ||grad(D(real))||^2

    NOTE: this is R1 (grad penalty on *real* samples only, driving ‖∇D‖→0),
    NOT WGAN-GP (interpolates real↔fake, drives ‖∇D‖→1). The two are distinct
    regularizers. ``gradient_penalty`` was an alias here and silently handed R1
    to anyone asking for WGAN-GP (issue #191); it is removed so the registry
    fails loud on the ambiguous name. The genuine two-sided penalty lives in
    :func:`gradient_penalty_loss` and is applied by ``CompositeGANLoss``.
    Mathematical Formulation:
    .. math::

        \\mathcal{L}_{R1} = \frac{\\gamma}{2} \\mathbb{E}_{x} \\left[ \\| \nabla_x D(x) \\|_2^2 \right]
    """

    def __init__(self, weight: float = 10.0):
        """__init__.

        Args:
            weight (float): Description.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        discriminator: Any,
        real_images: torch.Tensor,
        *args,
        critic_cond: "Mapping[str, Any] | None" = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            discriminator: Discriminator network (or prediction tensor in dummy calls)
            real_images: Real input images [B, C, H, W] (or target tensor in dummy calls)
            critic_cond: Conditioning for the ``discriminator(real_images)`` call
                below (#1931). An explicit parameter rather than a read out of
                ``**kwargs``: this signature already absorbs stray keywords
                silently, so a caller that passed conditioning would have had it
                accepted and dropped, and the penalty would have regularized a
                gradient the critic never takes. ``None`` reproduces the
                unconditioned call exactly.
        """
        if self.weight <= 0:
            return torch.tensor(
                0.0,
                device=(
                    real_images.device if hasattr(real_images, "device") else torch.device("cpu")
                ),
            )

        # Safety check: if discriminator is not a Module, this was likely called
        # generically during validation iteration (i.e. loss_fn(pred, target)).
        if not isinstance(discriminator, nn.Module):
            dev = discriminator.device if hasattr(discriminator, "device") else torch.device("cpu")
            return torch.tensor(0.0, device=dev)

        real_images.requires_grad_(True)
        real_logits = discriminator(real_images, **(critic_cond or {}))

        # Compute gradients of logits w.r.t input images
        grads = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,  # Allow unused inputs
        )[0]

        if grads is None:
            return torch.tensor(0.0, device=real_images.device)

        # R1 penalty: 0.5 * ||grad||^2, flattened over all dims except batch.
        #
        # ``(g.conj() * g).real`` rather than ``g.pow(2)`` because this critic's
        # input may be complex (#1931). For complex ``g = a + bi``, ``g**2`` is
        # ``a^2 - b^2 + 2abi`` -- not a squared magnitude, and not even real, so
        # the penalty poisons the total loss and ``backward()`` dies with "grad
        # can be implicitly created only for real scalar outputs". Torch's
        # convention for a real loss gives ``g == dL/da + i*dL/db``, so
        # ``(g.conj() * g).real == (dL/da)^2 + (dL/db)^2`` is exactly the squared
        # Euclidean norm over the real view, and is identical to ``g**2`` when
        # ``g`` is real. ``g.abs().pow(2)`` computes the same value but is
        # differentiated by ``create_graph=True`` and is undefined at zero.
        # Same defect, same fix as :func:`gradient_penalty_loss`.
        grads = grads.view(grads.size(0), -1)
        r1_penalty = 0.5 * (grads.conj() * grads).real.sum(dim=1).mean()

        if torch.isnan(r1_penalty) or torch.isinf(r1_penalty):
            return torch.tensor(0.0, device=real_images.device)

        return self.weight * r1_penalty


@register_loss(name="gan_lsgan", aliases=["lsgan", "LSGANLoss"], domain="agnostic")
class LSGANLoss(AdversarialLossStrategy):
    """Least Squares GAN (LSGAN) loss implementation.

    This loss function helps to stabilize GAN training by using a least squares
    loss instead of cross-entropy, penalizing predictions based on their distance
    from the decision boundary.

    Attributes:
        label_smoothing (float): Factor to smooth real labels. Real labels
            will be (1.0 - label_smoothing). Defaults to 0.0.
    """

    def __init__(self, label_smoothing: float = 0.0):
        """Initializes the LSGAN loss.

        Args:
            label_smoothing (float, optional): Smoothing factor for labels. Defaults to 0.0.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def compute_generator_loss(self, fake_outputs_d, **kwargs):
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        real_target = 1.0 - self.label_smoothing
        return torch.mean((fake_outputs_d - real_target) ** 2)

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        real_target = 1.0 - self.label_smoothing
        fake_target = 0.0 + self.label_smoothing
        real_loss = torch.mean((real_outputs_d - real_target) ** 2)
        fake_loss = torch.mean((fake_outputs_d - fake_target) ** 2)
        return real_loss, fake_loss


@register_loss(name="gan_ralsgan", aliases=["ralsgan", "RALSGANLoss"], domain="agnostic")
class RALSGANLoss(AdversarialLossStrategy):
    r"""RALSGANLoss class.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{RALSGAN} = \mathbb{E}[(D(x) - \mathbb{E}[D(\hat{x})] - 1)^2] + \mathbb{E}[(D(\hat{x}) - \mathbb{E}[D(x)] + 1)^2]
    """

    def __init__(self, label_smoothing: float = 0.0):
        """__init__.

        Args:
            label_smoothing (float): Description.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def compute_generator_loss(self, fake_outputs_d, real_outputs_d=None, **kwargs):
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
            real_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        if real_outputs_d is None:
            raise ValueError("RALSGAN requires real_outputs_d for generator loss")
        # Relativistic average margins are a fixed ±1 (Jolicoeur-Martineau);
        # label_smoothing is NOT part of RaLSGAN. The pre-fix code substituted
        # ``fake_target`` (= label_smoothing) for the relativistic ``+1`` term,
        # pushing toward the mean instead of the margin.
        mean_fake = torch.mean(fake_outputs_d)
        mean_real = torch.mean(real_outputs_d)
        loss_fake = torch.mean((fake_outputs_d - mean_real - 1.0) ** 2)
        loss_real = torch.mean((real_outputs_d - mean_fake + 1.0) ** 2)
        return (loss_real + loss_fake) * 0.5

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        # Relativistic average margins are a fixed ±1 (see generator loss).
        mean_fake = torch.mean(fake_outputs_d.detach())
        mean_real = torch.mean(real_outputs_d)
        loss_real = torch.mean((real_outputs_d - mean_fake - 1.0) ** 2)
        loss_fake = torch.mean((fake_outputs_d - mean_real + 1.0) ** 2)
        return loss_real, loss_fake


@register_loss(name="gan_wgan", aliases=["wgan", "WGANLoss"], domain="agnostic")
class WGANLoss(AdversarialLossStrategy):
    """WGANLoss class."""

    def __init__(self, label_smoothing: float = 0.0):
        """__init__.

        Args:
            label_smoothing (float): Description.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def compute_generator_loss(self, fake_outputs_d, **kwargs):
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        return -torch.mean(fake_outputs_d)

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        return -torch.mean(real_outputs_d), torch.mean(fake_outputs_d)


@register_loss(name="gan_hinge", aliases=["hinge", "HingeLoss"], domain="agnostic")
class HingeLoss(AdversarialLossStrategy):
    """Hinge loss implementation for GANs.

    This loss function effectively pushes predictions beyond a margin of 1.0,
    which is often used in combination with Spectral Normalization for
    stable training in modern GAN architectures (like SAGAN, BigGAN).

    Attributes:
        label_smoothing (float): Unused in standard Hinge loss, but kept
            for API compatibility. Defaults to 0.0.
    """

    def __init__(self, label_smoothing: float = 0.0):
        """Initializes the Hinge loss.

        Args:
            label_smoothing (float, optional): Smoothing factor. Defaults to 0.0.
        """
        super().__init__()
        self.label_smoothing = label_smoothing

    def compute_generator_loss(self, fake_outputs_d, **kwargs):
        """compute_generator_loss.

        Args:
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        return -torch.mean(fake_outputs_d)

    def compute_discriminator_loss(self, real_outputs_d, fake_outputs_d, **kwargs):
        """compute_discriminator_loss.

        Args:
            real_outputs_d (Any): Description.
            fake_outputs_d (Any): Description.
        Returns:
            Any: Description.
        """
        # Canonical hinge margins are ±1 (Lim & Ye / SAGAN / BigGAN):
        #   L_D = E[relu(1 - D(real))] + E[relu(1 + D(fake))].
        # label_smoothing applies one-sided to the REAL margin only (Salimans
        # 2016); the fake side keeps the fixed +1 margin. The pre-fix fake term
        # ``relu(D(fake) + label_smoothing)`` dropped that margin entirely.
        real_target = 1.0 - self.label_smoothing
        real_loss = torch.mean(F.relu(real_target - real_outputs_d))
        fake_loss = torch.mean(F.relu(1.0 + fake_outputs_d))
        return real_loss, fake_loss


def gradient_penalty_loss(discriminator, real_data, fake_data, eps=1e-8, critic_cond=None):
    """WGAN-GP: penalise the critic gradient norm away from 1 on interpolates.

    This is the **fourth** critic call on a discriminator step -- ``real``,
    ``fake``, R1 and this one -- and it is the one the conditioning seam missed
    (#1931). ``critic_cond`` is forwarded here exactly as the real and fake
    calls forward it, so a critic declaring ``supports_contrast_conditioning``
    is scored under its condition instead of raising on step 1. Passing
    ``None`` reproduces the old unconditioned call byte-for-byte, which is what
    every caller outside :class:`CompositeGANLoss` still does.

    Real and fake share one payload (see
    ``UnifiedGANLossComputer.compute_discriminator_loss``), so a point on the
    segment between them carries that same payload -- there is no interpolation
    to do on the condition itself.

    **The norm is complex-safe.** ``_align_for_critic`` (#1920) hands this
    function complex tensors whenever the configured critic declares
    ``accepts_complex``, and ``autograd.grad`` then returns a complex gradient.
    ``grads ** 2`` on a complex tensor is the complex square
    ``a**2 - b**2 + 2abi``, not ``a**2 + b**2``: the penalty came out complex,
    ``d_total_loss`` with it, and ``backward()`` died with "grad can be
    implicitly created only for real scalar outputs but got torch.complex64".
    Under torch's convention a real loss gives ``g = dL/da + i * dL/db``, so
    ``(g.conj() * g).real`` is ``(dL/da)**2 + (dL/db)**2`` -- the squared
    Euclidean norm over the real view -- and is identical to ``g ** 2`` when
    ``g`` is real. It is preferred over ``g.abs() ** 2`` because ``abs`` is not
    differentiable at zero, and ``create_graph=True`` differentiates through it.

    Args:
        discriminator (Any): The critic; called once, on the interpolates.
        real_data (Any): Real samples in the critic's own domain.
        fake_data (Any): Generated samples, same domain and shape.
        eps (Any): Added under the square root so its gradient stays finite.
        critic_cond (Any): Keyword payload forwarded to the critic, or ``None``.
    Returns:
        Any: The scalar penalty. Always real, whatever the input dtype.
    """
    batch = real_data.size(0)
    alpha = torch.rand(batch, 1, 1, 1, device=real_data.device)
    interp = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
    disc_interp = discriminator(interp, **(critic_cond or {}))
    grads = autograd.grad(
        outputs=disc_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(disc_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    grads = grads.view(batch, -1)
    grad_norm = torch.sqrt(torch.sum((grads.conj() * grads).real, dim=1) + eps)
    return ((grad_norm - 1) ** 2).mean()


_SSIM_IMPORT_HINT = (
    "CompositeGANLoss: lambda_ssim/lambda_ms_ssim > 0 requires "
    "spectramr.models.losses.ssim_loss, an in-repo module. An ImportError "
    "here indicates a broken installation — reinstall with:\n"
    '    pip install -e ".[dev]"'
)

_LPIPS_INSTALL_HINT = (
    "CompositeGANLoss: lambda_lpips > 0 requires the 'lpips' package. "
    "Install with:\n"
    "    pip install lpips"
)


@register_loss(name="gan_composite", aliases=["CompositeGANLoss"], domain="agnostic")
class CompositeGANLoss(nn.Module):
    r"""CompositeGANLoss class.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{CompositeGAN} = \sum_i w_i \mathcal{L}_{GAN}^{(i)}"""

    def __init__(
        self,
        adv_strategy: AdversarialLossStrategy,
        perceptual_loss: nn.Module | None,
        lambda_l1: float,
        lambda_perceptual: float,
        lambda_adv: float,
        lambda_feat_match: float,
        lambda_gp: float,
        lambda_ssim: float = 0.0,
        lambda_lpips: float = 0.0,
        **kwargs,
    ):
        """__init__.

        Args:
            adv_strategy (AdversarialLossStrategy): Description.
            perceptual_loss (Optional[nn.Module]): Description.
            lambda_l1 (float): Description.
            lambda_perceptual (float): Description.
            lambda_adv (float): Description.
            lambda_feat_match (float): Description.
            lambda_gp (float): Description.
            lambda_ssim (float): Description.
            lambda_lpips (float): Description.
        """
        super().__init__()
        self.adv_strategy = adv_strategy
        self.perceptual_loss = perceptual_loss
        # Use unified loss factory for L1 loss (SSOT)
        from .registry import create_loss

        self.l1_loss = create_loss("l1")
        self.complex_l1_loss = create_loss("complex_l1")
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_adv = lambda_adv
        self.lambda_feat_match = lambda_feat_match
        self.lambda_gp = lambda_gp
        self.lambda_ssim = lambda_ssim
        self.lambda_lpips = lambda_lpips
        self.lambda_ms_ssim = kwargs.get("lambda_ms_ssim", 0.0)

        # A lambda > 0 advertises the mechanism: failing to build it must
        # RAISE, never silently zero the weight or skip the term at forward
        # time (pitfall #9/#16). Deferred imports avoid circular imports.
        if lambda_ssim > 0:
            try:
                from .ssim_loss import SSIMLoss
            except ImportError as exc:
                raise ImportError(_SSIM_IMPORT_HINT) from exc
            self.ssim_loss = SSIMLoss()
        else:
            self.ssim_loss = None

        if self.lambda_ms_ssim > 0:
            try:
                from .ssim_loss import MSSSIMLoss
            except ImportError as exc:
                raise ImportError(_SSIM_IMPORT_HINT) from exc
            self.ms_ssim_loss = MSSSIMLoss()
        else:
            self.ms_ssim_loss = None

        if lambda_lpips > 0:
            try:
                # LPIPSLoss (not create_lpips_loss) so a missing `lpips`
                # package raises here instead of degrading to L1.
                from .lpips_loss import LPIPSLoss

                self.lpips_loss = LPIPSLoss(net="vgg")
            except ImportError as exc:
                raise ImportError(_LPIPS_INSTALL_HINT) from exc
        else:
            self.lpips_loss = None

    def forward(self, *args, **kwargs):
        """Dummy forward for audit compliance.

        forward method for CompositeGANLoss.

        Executes PyTorch tensor operations.

        Args:
            None

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Return 0.0 scalar tensor on correct device
        if args and isinstance(args[0], torch.Tensor):
            return torch.tensor(0.0, device=args[0].device)
        return torch.tensor(0.0)

    def to(self, *args, **kwargs):
        """Override to ensure loss functions are on correct device."""
        super().to(*args, **kwargs)
        # Move loss functions to the same device
        if hasattr(self.l1_loss, "to"):
            self.l1_loss = self.l1_loss.to(*args, **kwargs)
        if hasattr(self.complex_l1_loss, "to"):
            self.complex_l1_loss = self.complex_l1_loss.to(*args, **kwargs)
        if self.perceptual_loss and hasattr(self.perceptual_loss, "to"):
            self.perceptual_loss = self.perceptual_loss.to(*args, **kwargs)
        if self.ssim_loss and hasattr(self.ssim_loss, "to"):
            self.ssim_loss = self.ssim_loss.to(*args, **kwargs)
        if self.lpips_loss and hasattr(self.lpips_loss, "to"):
            self.lpips_loss = self.lpips_loss.to(*args, **kwargs)
        return self

    def _feature_matching_loss(
        self,
        real_feats: Sequence[torch.Tensor] | None,
        fake_feats: Sequence[torch.Tensor] | None,
    ) -> torch.Tensor:
        """_feature_matching_loss.

        Args:
            real_feats (Sequence[torch.Tensor] | None): Description.
            fake_feats (Sequence[torch.Tensor] | None): Description.
        Returns:
            torch.Tensor: Description.
        """
        if not real_feats or not fake_feats:
            dev = real_feats[0].device if real_feats else fake_feats[0].device
            return torch.tensor(0.0, device=dev)
        loss = torch.tensor(0.0, device=real_feats[0].device)
        for rf, ff in zip(real_feats, fake_feats, strict=False):
            loss = loss + self.l1_loss(ff, rf.detach())
        return loss / len(real_feats)

    def compute_generator_loss(
        self,
        fake_outputs_d: torch.Tensor,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        real_outputs_d: torch.Tensor | None = None,
        fake_features_d: Sequence[torch.Tensor] | None = None,
        real_features_d: Sequence[torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """compute_generator_loss.

        Args:
            fake_outputs_d (torch.Tensor): Description.
            real_images (torch.Tensor): Description.
            fake_images (torch.Tensor): Description.
            real_outputs_d (Optional[torch.Tensor]): Description.
            fake_features_d (Optional[Sequence[torch.Tensor]]): Description.
            real_features_d (Optional[Sequence[torch.Tensor]]): Description.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        losses: dict[str, torch.Tensor] = {}
        if isinstance(self.adv_strategy, RALSGANLoss):
            g_adv = self.adv_strategy.compute_generator_loss(
                fake_outputs_d=fake_outputs_d,
                real_outputs_d=real_outputs_d,
            )
        else:
            g_adv = self.adv_strategy.compute_generator_loss(
                fake_outputs_d=fake_outputs_d,
            )
        losses["g_adv_loss"] = g_adv * self.lambda_adv

        # Use ComplexL1Loss if inputs are complex (Fix 9: Phase Blindness)
        if fake_images.is_complex() or (fake_images.dim() == 4 and fake_images.shape[1] % 2 == 0):
            losses["l1_loss"] = self.complex_l1_loss(fake_images, real_images) * self.lambda_l1
        else:
            losses["l1_loss"] = self.l1_loss(fake_images, real_images) * self.lambda_l1

        if self.lambda_perceptual > 0 and self.perceptual_loss:
            p_loss = self.perceptual_loss(fake_images, real_images)
            if isinstance(p_loss, tuple):
                p_loss = p_loss[0]
            losses["perceptual_loss"] = p_loss * self.lambda_perceptual
        else:
            losses["perceptual_loss"] = torch.tensor(0.0, device=real_images.device)

        if self.lambda_feat_match > 0 and fake_features_d and real_features_d:
            losses["feat_match_loss"] = (
                self._feature_matching_loss(real_features_d, fake_features_d)
                * self.lambda_feat_match
            )
        else:
            losses["feat_match_loss"] = torch.tensor(0.0, device=real_images.device)

        if self.lambda_ssim > 0 and self.ssim_loss:
            losses["ssim_loss"] = self.ssim_loss(fake_images, real_images) * self.lambda_ssim
        else:
            losses["ssim_loss"] = torch.tensor(0.0, device=real_images.device)

        if self.lambda_ms_ssim > 0 and self.ms_ssim_loss:
            losses["ms_ssim_loss"] = (
                self.ms_ssim_loss(fake_images, real_images) * self.lambda_ms_ssim
            )
        else:
            losses["ms_ssim_loss"] = torch.tensor(0.0, device=real_images.device)

        if self.lambda_lpips > 0 and self.lpips_loss:
            losses["lpips_loss"] = self.lpips_loss(fake_images, real_images) * self.lambda_lpips
        else:
            losses["lpips_loss"] = torch.tensor(0.0, device=real_images.device)
        losses["g_total_loss"] = sum(losses.values())
        return losses

    def compute_discriminator_loss(
        self,
        real_outputs_d: torch.Tensor,
        fake_outputs_d: torch.Tensor,
        discriminator: nn.Module,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        critic_cond: "Mapping[str, Any] | None" = None,
    ) -> dict[str, torch.Tensor]:
        """compute_discriminator_loss.

        ``real_outputs_d``/``fake_outputs_d`` were already scored by the
        caller, so ``discriminator`` is used for one thing here: the gradient
        penalty's own call on the interpolates. ``critic_cond`` exists for that
        single call (#1931) -- declared explicitly rather than absorbed into
        ``**kwargs`` so that an implementer which cannot honour it fails at the
        call rather than dropping the payload silently.

        Args:
            real_outputs_d (torch.Tensor): Description.
            fake_outputs_d (torch.Tensor): Description.
            discriminator (nn.Module): Description.
            real_images (torch.Tensor): Description.
            fake_images (torch.Tensor): Description.
            critic_cond (Optional[Mapping[str, Any]]): Conditioning forwarded to
                the gradient penalty's critic call; ``None`` leaves it
                unconditioned, as every non-conditioned arm expects.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        real_loss, fake_loss = self.adv_strategy.compute_discriminator_loss(
            real_outputs_d=real_outputs_d,
            fake_outputs_d=fake_outputs_d,
        )
        losses: dict[str, torch.Tensor] = {
            "d_loss_real": real_loss * self.lambda_adv,
            "d_loss_fake": fake_loss * self.lambda_adv,
        }
        total = losses["d_loss_real"] + losses["d_loss_fake"]
        if self.lambda_gp > 0:
            gp = gradient_penalty_loss(
                discriminator, real_images, fake_images, critic_cond=critic_cond
            )
            losses["gp_loss"] = gp * self.lambda_gp
            total = total + losses["gp_loss"]
        losses["d_total_loss"] = total
        return losses


# Backward-compatible small facade for tests


class GANLossType(enum.Enum):
    """GANLossType class."""

    standard = "standard"
    wgan_gp = "wgan-gp"
    hinge = "hinge"
    lsgan = "lsgan"
    feature_matching = "feature_matching"
    spectral_norm = "spectral_norm"


@dataclass
class LossConfig:
    """LossConfig class."""

    gan_loss_type: str = "standard"
    lambda_l1: float = 10.0
    lambda_perceptual: float = 0.0
    lambda_adv: float = 1.0
    lambda_feature_match: float = 0.0
    lambda_gp: float = 10.0
    lambda_ssim: float = 0.0
    lambda_lpips: float = 0.0


class VGG19FeatureExtractor(nn.Module):
    """VGG19 feature extractor for perceptual features.

    Returns list of feature maps at multiple VGG19 layers.
    Handles 1-channel, 2-channel (k-space), and 3-channel inputs.
    """

    def __init__(self, use_input_norm: bool = True, feature_layers: list[str] | None = None):
        """Initialize VGG19 feature extractor.

        Args:
            use_input_norm: Whether to normalize input with ImageNet stats.
            feature_layers: Which layers to extract. Default: relu1_2, relu2_2, relu3_3, relu4_3, relu5_4
        """
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise ImportError(
                "VGG19FeatureExtractor requires torchvision, which is a core "
                "dependency (see pyproject.toml) -- this failure means the "
                "install is broken or ABI-mismatched, not that an extra is "
                "missing. torchvision ships compiled ops built against a fixed "
                "torch ABI, so it must resolve from the same index as torch "
                "('pytorch-cu126' in [tool.uv.sources]); a torchvision resolved "
                "from PyPI instead raises RuntimeError: operator torchvision::nms "
                "does not exist. Reinstall torch and torchvision together from "
                "that index."
            ) from exc

        super().__init__()
        self.use_input_norm = use_input_norm

        if feature_layers is None:
            feature_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_4"]
        self.feature_layers = feature_layers

        # Load VGG19 features. Modern weights= API only — the deleted
        # `pretrained=True` fallback masked the real error (a weights-download
        # failure surfaced as a second exception from the fallback arm).
        # Corrected 2026-08-13: this comment previously said that kwarg "raises
        # TypeError on torchvision >= 0.15". Measured on the pinned 0.28.0, it
        # does not — it is deprecated (two UserWarnings), not removed (#961).
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.vgg_layers = vgg
        self.vgg_layers.eval()

        # Freeze parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Layer name to VGG index mapping
        self.layer_indices = {
            "relu1_1": 1,
            "relu1_2": 3,
            "relu2_1": 6,
            "relu2_2": 8,
            "relu3_1": 11,
            "relu3_2": 13,
            "relu3_3": 15,
            "relu3_4": 17,
            "relu4_1": 20,
            "relu4_2": 22,
            "relu4_3": 24,
            "relu4_4": 26,
            "relu5_1": 29,
            "relu5_2": 31,
            "relu5_3": 33,
            "relu5_4": 35,
        }

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract features from input.

        Args:
            x: Input tensor [B, C, H, W] where C can be 1, 2, or 3.

        Returns:
            List of feature tensors at specified layers.

        forward method for VGG19FeatureExtractor.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            list[torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        # Handle multi-channel complex data: compute magnitude
        if x.shape[1] % 2 == 0 and not x.is_complex():
            B, C, H, W = x.shape
            x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(B, H, W, C // 2, 2)
            x_complex = torch.view_as_complex(x_reshaped).permute(0, 3, 1, 2)
            magnitude = torch.sqrt(torch.sum(x_complex.abs() ** 2, dim=1, keepdim=True))
            x = magnitude.repeat(1, 3, 1, 1)
        # Handle 1-channel grayscale
        elif x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Normalize with ImageNet stats
        if self.use_input_norm:
            x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        # Extract features
        features = []
        target_indices = {
            self.layer_indices[name] for name in self.feature_layers if name in self.layer_indices
        }

        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if idx in target_indices:
                features.append(x.clone())

            # Stop early if past all needed layers
            if idx > max(target_indices):
                break

        return features


class VGGPerceptualLoss(PerceptualLoss):
    """VGG19-based perceptual loss. Compatibility wrapper for SSOT PerceptualLoss."""

    def __init__(self, _feature_extractor=None, _loss_func=None, _weights=None):
        # Ignore legacy args and init SSOT PerceptualLoss
        """__init__.

        Args:
            _feature_extractor (Any): Description.
            _loss_func (Any): Description.
            _weights (Any): Description.
        """
        super().__init__()


class CompositeLoss(nn.Module):
    r"""A combined loss module encompassing multiple loss components for training.

    This module delegates to underlying adversarial, reconstruction, and perceptual
    losses, aggregating their values based on provided weights to return a unified
    loss signal.

    Attributes:
        config (LossConfig): The configuration governing the active losses and weights.
        adv (AdversarialLossStrategy): The strategy for computing GAN adversarial losses.
        perceptual (nn.Module, optional): An external module computing perceptual loss.
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Composite} = \sum_i w_i \mathcal{L}_i"""

    def __init__(self, config: LossConfig = LossConfig()):
        """Initializes the composite loss module.

        Args:
            config (LossConfig, optional): The configuration specifying loss weights
                and active strategies. Defaults to an empty LossConfig.
        """
        super().__init__()
        self.config = config
        adv = _select_adv_strategy(config.gan_loss_type)
        perceptual = None
        if config.lambda_perceptual > 0:
            try:
                perceptual = VGGPerceptualLoss()
            except Exception:
                perceptual = None
        self._core: CompositeGANLoss = CompositeGANLoss(
            adv_strategy=adv,
            perceptual_loss=perceptual,
            lambda_l1=config.lambda_l1,
            lambda_perceptual=config.lambda_perceptual,
            lambda_adv=config.lambda_adv,
            lambda_feat_match=config.lambda_feature_match,
            lambda_gp=config.lambda_gp,
            lambda_ssim=config.lambda_ssim,
            lambda_lpips=config.lambda_lpips,
        )

    def compute_generator_loss(
        self,
        fake_outputs: torch.Tensor | None = None,
        real_images: torch.Tensor | None = None,
        fake_images: torch.Tensor | None = None,
        *,
        fake_outputs_d: torch.Tensor | None = None,
        real_outputs: torch.Tensor | None = None,
        fake_features: list[torch.Tensor] | None = None,
        real_features: list[torch.Tensor] | None = None,
        **_unused: object,
    ) -> dict[str, torch.Tensor]:
        """compute_generator_loss.

        Args:
            fake_outputs (Optional[torch.Tensor]): Description.
            real_images (Optional[torch.Tensor]): Description.
            fake_images (Optional[torch.Tensor]): Description.
            fake_outputs_d (Optional[torch.Tensor]): Description.
            real_outputs (Optional[torch.Tensor]): Description.
            fake_features (Optional[list[torch.Tensor]]): Description.
            real_features (Optional[list[torch.Tensor]]): Description.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        resolved_fake = fake_outputs_d if fake_outputs_d is not None else fake_outputs
        if resolved_fake is None:
            raise ValueError(
                (
                    "CompositeLoss.compute_generator_loss requires "
                    "'fake_outputs_d' or positional 'fake_outputs'."
                ),
            )
        if real_images is None or fake_images is None:
            raise ValueError(
                ("CompositeLoss.compute_generator_loss requires 'real_images' and 'fake_images'."),
            )

        out: dict[str, torch.Tensor] = self._core.compute_generator_loss(
            fake_outputs_d=resolved_fake,
            real_images=real_images,
            fake_images=fake_images,
            real_outputs_d=real_outputs,
            fake_features_d=fake_features,
            real_features_d=real_features,
        )

        zero = resolved_fake.new_zeros(())
        total = out.get("g_total_loss")
        if total is None:
            total = sum(out.values(), zero) if out else zero
            out["g_total_loss"] = total

        result: dict[str, torch.Tensor] = dict(out)
        alias_map = {
            "g_loss_total": "g_total_loss",
            "g_loss_adv": "g_adv_loss",
            "g_adv_loss": "g_adv_loss",
            "g_loss_l1": "l1_loss",
            "g_l1_loss": "l1_loss",
            "g_loss_perceptual": "perceptual_loss",
            "g_loss_feat_match": "feat_match_loss",
            "g_loss_ssim": "ssim_loss",
            "g_loss_lpips": "lpips_loss",
        }

        for alias, source in alias_map.items():
            result[alias] = out.get(source, zero.clone())

        return result

    def compute_discriminator_loss(
        self,
        real_outputs: torch.Tensor | None = None,
        fake_outputs: torch.Tensor | None = None,
        discriminator: nn.Module | None = None,
        real_images: torch.Tensor | None = None,
        fake_images: torch.Tensor | None = None,
        *,
        real_outputs_d: torch.Tensor | None = None,
        fake_outputs_d: torch.Tensor | None = None,
        critic_cond: "Mapping[str, Any] | None" = None,
        **_unused: object,
    ) -> dict[str, torch.Tensor]:
        """compute_discriminator_loss.

        Args:
            real_outputs (Optional[torch.Tensor]): Description.
            fake_outputs (Optional[torch.Tensor]): Description.
            discriminator (Optional[nn.Module]): Description.
            real_images (Optional[torch.Tensor]): Description.
            fake_images (Optional[torch.Tensor]): Description.
            real_outputs_d (Optional[torch.Tensor]): Description.
            fake_outputs_d (Optional[torch.Tensor]): Description.
            critic_cond (Optional[Mapping[str, Any]]): Forwarded to the core
                loss, and from there to the gradient penalty's critic call
                (#1931). Named explicitly: ``**_unused`` would have swallowed
                it and left the penalty unconditioned with no error at all.
        Returns:
            dict[str, torch.Tensor]: Description.
        """
        resolved_real = real_outputs_d if real_outputs_d is not None else real_outputs
        resolved_fake = fake_outputs_d if fake_outputs_d is not None else fake_outputs
        if resolved_real is None or resolved_fake is None:
            raise ValueError(
                (
                    "CompositeLoss.compute_discriminator_loss requires "
                    "'real_outputs_d'/'fake_outputs_d' or positional inputs."
                ),
            )
        if discriminator is None:
            raise ValueError(
                ("CompositeLoss.compute_discriminator_loss requires a discriminator."),
            )
        if real_images is None or fake_images is None:
            raise ValueError(
                (
                    "CompositeLoss.compute_discriminator_loss requires "
                    "'real_images' and 'fake_images'."
                ),
            )

        out: dict[str, torch.Tensor] = self._core.compute_discriminator_loss(
            real_outputs_d=resolved_real,
            fake_outputs_d=resolved_fake,
            discriminator=discriminator,
            real_images=real_images,
            fake_images=fake_images,
            critic_cond=critic_cond,
        )
        zero = resolved_real.new_zeros(())
        return {
            "d_total_loss": out["d_total_loss"],
            "d_loss_real": out["d_loss_real"],
            "d_loss_fake": out["d_loss_fake"],
            "gp_loss": out.get("gp_loss", zero.clone()),
        }


def _select_adv_strategy(
    name: str,
    label_smoothing: float = 0.0,
) -> AdversarialLossStrategy:
    """_select_adv_strategy.

    Args:
        name (str): Description.
        label_smoothing (float): Description.
    Returns:
        AdversarialLossStrategy: Description.
    """
    name = (name or "").lower()
    if "wgan" in name or "wasserstein" in name:
        return WGANLoss(label_smoothing=label_smoothing)
    if "hinge" in name:
        return HingeLoss(label_smoothing=label_smoothing)
    if "lsgan" in name:
        return LSGANLoss(label_smoothing=label_smoothing)
    if "ralsgan" in name:
        return RALSGANLoss(label_smoothing=label_smoothing)
    return StandardGANLoss(label_smoothing=label_smoothing)


def create_gan_loss_system(
    gan_loss_type: str,
    model_type: str,
    lambda_l1: float,
    lambda_perceptual: float,
    lambda_adv: float,
    lambda_feat_match: float,
    lambda_gp: float,
    lambda_ssim: float = 0.0,
    device: str = "cuda",
    label_smoothing: float = 0.0,
) -> dict[str, Any]:
    """create_gan_loss_system.

    Args:
        gan_loss_type (str): Description.
        model_type (str): Description.
        lambda_l1 (float): Description.
        lambda_perceptual (float): Description.
        lambda_adv (float): Description.
        lambda_feat_match (float): Description.
        lambda_gp (float): Description.
        lambda_ssim (float): Description.
        device (str): Description.
        label_smoothing (float): Description.
    Returns:
        dict[str, Any]: Description.
    """
    gan_loss_type = gan_loss_type.lower()
    if "wgan" in gan_loss_type or "wasserstein" in gan_loss_type:
        adv_strategy = WGANLoss(label_smoothing=label_smoothing)
    elif "hinge" in gan_loss_type:
        adv_strategy = HingeLoss(label_smoothing=label_smoothing)
    elif "ralsgan" in gan_loss_type:
        adv_strategy = RALSGANLoss(label_smoothing=label_smoothing)
    elif "lsgan" in gan_loss_type:
        adv_strategy = LSGANLoss(label_smoothing=label_smoothing)
    elif "standard" in gan_loss_type or "vanilla" in gan_loss_type:
        adv_strategy = StandardGANLoss(label_smoothing=label_smoothing)
    else:
        logger.warning(f"Unknown loss type '{gan_loss_type}', default to 'wgan-gp'.")
        adv_strategy = WGANLoss(label_smoothing=label_smoothing)

    perceptual_loss = None
    if lambda_perceptual > 0:
        try:
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA unavailable; perceptual loss on CPU.")
                effective_device = "cpu"
            else:
                effective_device = device
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA unavailable; perceptual loss on CPU.")
                effective_device = "cpu"
            else:
                effective_device = device
            # Instantiating PerceptualLoss directly
            perceptual_loss = PerceptualLoss().to(effective_device)
        except Exception as e:
            logger.warning(f"PerceptualLoss init failed: {e}; disabled.")
            lambda_perceptual = 0.0

    if "diffusion" in model_type.lower():
        logger.warning("Diffusion model; disabling adv/perceptual/GP losses.")
        lambda_adv = 0.0
        lambda_perceptual = 0.0
        lambda_gp = 0.0
        lambda_feat_match = 0.0

    composite_loss = CompositeLoss(
        config=LossConfig(
            gan_loss_type=gan_loss_type,
            lambda_l1=lambda_l1,
            lambda_perceptual=lambda_perceptual,
            lambda_adv=lambda_adv,
            lambda_feature_match=lambda_feat_match,
            lambda_gp=lambda_gp,
            lambda_ssim=lambda_ssim,
        ),
    )

    return {
        "loss_function": composite_loss,
        "adversarial_strategy": adv_strategy,
        "perceptual_loss_module": perceptual_loss,
    }


def get_gan_loss(
    name: str,
) -> tuple[
    Callable[..., torch.Tensor],
    Callable[..., torch.Tensor],
]:
    """Simple registry for GAN loss functions.

    Returns a tuple of (discriminator_loss_func, generator_loss_func)
    callables.

    Args:
        name: Loss type name ('bce', 'hinge', 'ralsgan', 'lsgan', 'wgan')

    Returns:
        Tuple of (d_loss_func, g_loss_func) where each is a callable
        that takes discriminator outputs and returns loss

    Raises:
        ValueError: If loss type is not supported

    """
    name = name.lower()

    if name in ("bce", "standard", "vanilla"):
        strategy = StandardGANLoss()

        def d_loss(
            real_outputs: torch.Tensor,
            fake_outputs: torch.Tensor,
        ) -> torch.Tensor:
            """d_loss.

            Args:
                real_outputs (torch.Tensor): Description.
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            real_loss, fake_loss = strategy.compute_discriminator_loss(
                real_outputs,
                fake_outputs,
            )
            return real_loss + fake_loss

        def g_loss(fake_outputs: torch.Tensor) -> torch.Tensor:
            """g_loss.

            Args:
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return strategy.compute_generator_loss(fake_outputs)

    elif name == "hinge":
        strategy = HingeLoss()

        def d_loss(
            real_outputs: torch.Tensor,
            fake_outputs: torch.Tensor,
        ) -> torch.Tensor:
            """d_loss.

            Args:
                real_outputs (torch.Tensor): Description.
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            real_loss, fake_loss = strategy.compute_discriminator_loss(
                real_outputs,
                fake_outputs,
            )
            return real_loss + fake_loss

        def g_loss(fake_outputs: torch.Tensor) -> torch.Tensor:
            """g_loss.

            Args:
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return strategy.compute_generator_loss(fake_outputs)

    elif name == "ralsgan":
        strategy = RALSGANLoss()

        def d_loss(
            real_outputs: torch.Tensor,
            fake_outputs: torch.Tensor,
        ) -> torch.Tensor:
            """d_loss.

            Args:
                real_outputs (torch.Tensor): Description.
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            real_loss, fake_loss = strategy.compute_discriminator_loss(
                real_outputs,
                fake_outputs,
            )
            return real_loss + fake_loss

        def g_loss(
            fake_outputs: torch.Tensor,
            real_outputs: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """g_loss.

            Args:
                fake_outputs (torch.Tensor): Description.
                real_outputs (Optional[torch.Tensor]): Description.
            Returns:
                torch.Tensor: Description.
            """
            return strategy.compute_generator_loss(
                fake_outputs,
                real_outputs,
            )

    elif name == "lsgan":
        strategy = LSGANLoss()

        def d_loss(
            real_outputs: torch.Tensor,
            fake_outputs: torch.Tensor,
        ) -> torch.Tensor:
            """d_loss.

            Args:
                real_outputs (torch.Tensor): Description.
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            real_loss, fake_loss = strategy.compute_discriminator_loss(
                real_outputs,
                fake_outputs,
            )
            return real_loss + fake_loss

        def g_loss(fake_outputs: torch.Tensor) -> torch.Tensor:
            """g_loss.

            Args:
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return strategy.compute_generator_loss(fake_outputs)

    elif name in ("wgan", "wasserstein"):
        strategy = WGANLoss()

        def d_loss(
            real_outputs: torch.Tensor,
            fake_outputs: torch.Tensor,
        ) -> torch.Tensor:
            """d_loss.

            Args:
                real_outputs (torch.Tensor): Description.
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            real_loss, fake_loss = strategy.compute_discriminator_loss(
                real_outputs,
                fake_outputs,
            )
            return real_loss + fake_loss

        def g_loss(fake_outputs: torch.Tensor) -> torch.Tensor:
            """g_loss.

            Args:
                fake_outputs (torch.Tensor): Description.
            Returns:
                torch.Tensor: Description.
            """
            return strategy.compute_generator_loss(fake_outputs)

    else:
        raise ValueError(
            f"Unsupported GAN loss type: {name}. Supported: bce, hinge, ralsgan, lsgan, wgan",
        )

    return d_loss, g_loss
