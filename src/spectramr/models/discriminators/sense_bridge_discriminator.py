"""Image-space critic for k-space arms, bridged by the SENSE adjoint.

Every discriminator in this package scores whatever domain the generator
emits. For the ``kspace_filling`` cohort that is k-space, so a critic there
judges Fourier coefficients: it can reward a spectrum whose coefficients look
plausible while the reconstructed image carries coil-cancellation artefacts,
because nothing in a per-coefficient score is sensitive to how the coils
combine. This module closes that gap by moving the pair into image space
*before* the critic sees it -- the bridge is part of the critic, not a
transform bolted onto the training loop, so the same arm YAML that names the
critic also gets the bridge.

Which image, exactly
--------------------
``sense_adjoint`` (the matched filter ``sum_c conj(S_c) I_c``), NOT
``coil_combine(method="sense")`` (Roemer, which divides by ``sum_c |S_c|^2``).
The two differ by a spatially-varying real factor, so a critic trained on one
and metrics computed on the other disagree about what "sharp" means in the
periphery. ``sense_adjoint`` is what this cohort's own validation metrics use
(``diffusion.py`` ``_measurement_aware_metrics``), so this critic and the
numbers the arm is ranked by see the same shading.

That choice deserves a caveat rather than silence. The arm also declares
``losses.image_losses: [hfen]``, and how that term reaches the objective on
the diffusion path is **not established**. What was checked: the shared fold
(``loss_folding.fold_builder_image_losses``) is called only by
``reconstruction.py`` and ``field_cocycle_strategy.py``, and
``DiffusionTrainingStrategy`` declares neither ``inline_losses`` nor
``folds_image_losses`` (both inherit ``None`` -- "ownership not declared"), so
the audit's ``image_losses_reach_the_objective`` witness reports it UNVERIFIED
rather than passing it. Whether ``hfen`` is computed inline, folded by some
other route, or silently dropped was NOT traced. Tracked as #1918.

The consequence for this critic is the same either way, which is why it is
noted and not waited on: the bridge here is chosen to match the arm's ranking
metrics (``_measurement_aware_metrics``), not to match ``hfen``. Do not couple
the two until #1918 says what ``hfen`` actually receives.

Layout, and why this class refuses to guess
-------------------------------------------
A real tensor with an even channel count is ambiguous: ``[R1,I1,R2,I2,...]``
(interleaved, what the generator emits) and ``[R1,R2,...,I1,I2,...]`` (block,
what ``_align_for_critic`` and ``GANTrainingStrategy`` produce from a complex
tensor) are the same shape and dtype. Decoding the wrong one pairs the real
part of coil 1 with the real part of coil 2 and calls it a complex number.

So this critic declares ``accepts_complex=True`` on its registration, and
``DiffusionTrainingStrategy._align_for_critic`` reads that capability and
hands it the tensors *unrealified*. ``_as_complex_image`` -- the framework's
one owner for "coerce a prediction to a complex image" -- then passes complex
through untouched and decodes a real tensor as interleaved, which is the only
layout that can still arrive here.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from spectramr.core.metrics.nr_consistency import _as_complex_image
from spectramr.infrastructure.physics.fft_ops import sense_adjoint
from spectramr.models.interfaces.models import IDiscriminator
from spectramr.models.registry import MODEL_REGISTRY, register_model

logger = logging.getLogger(__name__)

#: How the complex bridged image is presented to the inner critic. Closed set:
#: an unknown value raises rather than falling back (non-negotiable 3).
IMAGE_REPRS: frozenset[str] = frozenset({"magnitude", "complex"})

#: Channels the inner critic must be built for, per representation. The arm
#: declares ``in_channels`` and this table is what validates it, so a mismatch
#: is a config error at construction rather than a shape error at step 1.
_REPR_CHANNELS: dict[str, int] = {"magnitude": 1, "complex": 2}


class SenseBridge(nn.Module):
    """k-space -> SENSE-combined image. The bridge alone, with no critic.

    Extracted so that more than one critic can stand on it without a second
    copy of the adjoint, the provider seam, or the layout decision
    (non-negotiable 17). :class:`SenseBridgeDiscriminator` composes it, and so
    does the contrast-conditioned critic in
    ``contrast_conditioned_sense_bridge.py``, which needs the *same* bridged
    image but a wider inner critic -- widening cannot be expressed by
    subclassing, because every concrete discriminator here already sits at the
    inheritance-depth ceiling (non-negotiable 20).

    Holds no parameters; it exists as an ``nn.Module`` only so composing
    critics can register it as a submodule and keep ``.to(device)`` uniform.
    """

    def __init__(self, image_repr: str = "magnitude") -> None:
        super().__init__()
        if image_repr not in IMAGE_REPRS:
            raise ValueError(
                f"image_repr={image_repr!r} is not a known representation. "
                f"Choose one of {sorted(IMAGE_REPRS)}."
            )
        self.image_repr = image_repr
        #: Set by the training strategy. Left None so that a critic reached
        #: without the seam raises in ``forward`` instead of silently scoring
        #: a root-sum-of-squares image the arm never asked for.
        self._smaps_provider: Callable[[int], torch.Tensor | None] | None = None

    def set_smaps_provider(self, provider: Callable[[int], torch.Tensor | None]) -> None:
        """Install the callable that yields coil sensitivities for a batch size.

        A *provider*, not a value: the strategy populates ``_current_smaps``
        inside ``_prepare_diffusion_inputs``, which runs **during** the
        generator closure. A value pushed before that closure would be the
        previous step's maps (or absent on the first step), so the critic pulls
        at forward time instead.
        """
        self._smaps_provider = provider

    def _resolve_smaps(self, batch: int) -> torch.Tensor:
        """Coil sensitivities for this batch, or a raise naming the cause."""
        if self._smaps_provider is None:
            raise RuntimeError(
                "A SENSE-bridged critic was called without a smaps provider. The "
                "SENSE bridge has no meaning without coil sensitivities, and falling "
                "back to a root-sum-of-squares combine would change what this critic "
                "scores without saying so (non-negotiable 3). The diffusion strategy "
                "installs the provider in train_step; a critic reached by another path "
                "must call set_smaps_provider first."
            )
        smaps = self._smaps_provider(batch)
        if smaps is None:
            raise RuntimeError(
                "SENSE bridge: the smaps provider returned None for batch "
                f"size {batch}. This arm declares a SENSE-bridged critic, so the "
                "dataset must supply coil sensitivity maps."
            )
        return smaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """k-space (complex, or real interleaved) -> real image for the critic."""
        kspace = _as_complex_image(x)
        smaps = self._resolve_smaps(kspace.shape[0])
        image = sense_adjoint(kspace, smaps=smaps)
        if self.image_repr == "magnitude":
            return image.abs()
        # "complex": interleave real/imag on the channel axis, matching the
        # convention the generator is registered for.
        return torch.view_as_real(image).permute(0, 1, 4, 2, 3).flatten(1, 2)


# ``input_domain="kspace"`` (#1920) -- the MIRROR IMAGE of the critics in
# ``kspace_discriminator.py``, which are named for k-space and consume images.
# This one is named for a bridge and consumes k-space: ``forward`` is
# ``critic(bridge(x))``, so the image it scores is one it manufactures. A blanket
# rule for 'the k-space critics' would have inverted exactly one of the two.
@register_model(
    role="discriminator",
    name="sense_bridge_patchgan",
    training_mode="gan",
    accepts_complex=True,
    input_domain="kspace",
)
class SenseBridgeDiscriminator(IDiscriminator, nn.Module):
    """Bridge k-space to a SENSE-combined image, then score it with a critic.

    Args:
        in_channels: channels the inner critic is built for. Must agree with
            ``image_repr`` (1 for ``magnitude``, 2 for ``complex``); the
            builder passes this through from
            ``model.discriminator_component.kwargs.in_channels``.
        image_repr: ``"magnitude"`` scores ``|x|``; ``"complex"`` scores the
            real/imag pair interleaved on the channel axis, which keeps phase
            visible to the critic.
        inner_critic: registry name of the critic applied after the bridge.
        critic_kwargs: forwarded to that critic's constructor.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_repr: str = "magnitude",
        inner_critic: str = "patch_gan",
        critic_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        # The bridge validates ``image_repr`` against IMAGE_REPRS (#3).
        self.sense_bridge = SenseBridge(image_repr=image_repr)
        expected = _REPR_CHANNELS[image_repr]
        if in_channels != expected:
            raise ValueError(
                f"in_channels={in_channels} contradicts image_repr={image_repr!r}, which "
                f"presents {expected} channel(s) to the inner critic. Set "
                f"model.discriminator_component.kwargs.in_channels={expected}."
            )
        if inner_critic not in MODEL_REGISTRY:
            raise ValueError(
                f"inner_critic={inner_critic!r} is not registered. Register it with "
                f"@register_model, or name one of the discriminators that already is."
            )

        # ``image_repr`` is NOT stored here -- it lives on ``sense_bridge`` and is
        # read back through the property below, so the two can never disagree.
        self.in_channels = in_channels
        self.inner_critic_name = inner_critic

        entry = MODEL_REGISTRY[inner_critic]
        critic_cls = entry["class"] if isinstance(entry, dict) else entry
        self.critic = critic_cls(in_channels=in_channels, **(critic_kwargs or {}))

    @property
    def image_repr(self) -> str:
        """The bridge's representation. One owner, read through."""
        return self.sense_bridge.image_repr

    def set_smaps_provider(self, provider: Callable[[int], torch.Tensor | None]) -> None:
        """Install the coil-sensitivity provider on the bridge.

        Kept on this class (rather than left to callers to reach through to
        ``sense_bridge``) because ``diffusion.py:1700`` wires the seam with
        ``getattr(discriminator, "set_smaps_provider", None)`` and silently
        does nothing when the attribute is absent -- a delegation this class
        dropped would disable the bridge without an error.
        """
        self.sense_bridge.set_smaps_provider(provider)

    def bridge(self, x: torch.Tensor) -> torch.Tensor:
        """k-space (complex, or real interleaved) -> real image for the critic."""
        return self.sense_bridge(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score ``x`` after bridging it into image space.

        The loss computer calls ``discriminator(x)`` positionally, so this is
        the entry point for both the real and the fake side.
        """
        return self.critic(self.bridge(x))

    def discriminate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """IDiscriminator entry point; identical to ``forward``."""
        return self.forward(x)

    def get_feature_maps(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inner-critic feature maps, taken on the bridged image.

        Feature matching compares the critic's internal activations, so these
        must come from the same tensor ``forward`` scores -- reading them off
        the raw k-space would compare features from a different domain.
        """
        bridged = self.bridge(x)
        inner = getattr(self.critic, "get_feature_maps", None)
        if callable(inner):
            return inner(bridged)
        return {"out": self.critic(bridged)}
