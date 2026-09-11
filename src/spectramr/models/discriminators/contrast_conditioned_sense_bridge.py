"""A SENSE-bridged critic that is told *which contrast* and *which t* it scores.

Why a conditioned critic at all
-------------------------------
The ``kspace_filling`` arms train one generator over three contrasts
(``data.multi_contrast.n_contrasts: 3``) with ``pairing.single_contrast: true``,
so ``contrast_idx`` varies *across* a batch, and over a diffusion schedule, so
the reconstruction quality a sample can possibly have depends on ``t``. An
unconditioned critic sees that mixture as one distribution. It is then rewarded
for the easiest available discriminator: "this looks like a T2 image" separates
real from fake at chance-beating rates without saying anything about
reconstruction quality, and "this is blurry" is *correct* at high ``t`` and
*wrong* at low ``t``. Both push the generator toward the marginal over
contrasts and timesteps -- the mode-averaging that non-negotiable 15's
``supports_contrast_conditioning`` flag exists to make visible.

Telling the critic the condition removes both shortcuts: it must separate real
from fake *within* a contrast and *at* a timestep, which is the comparison the
generator is actually trying to win.

What this is NOT
----------------
**Not Diffusion-GAN.** That scheme corrupts the real sample to the same ``t``
as the fake and has the critic separate two equally-noised samples. Here the
fake is ``_extract_and_fix_output``'s clean-domain reconstruction (an estimate
of x0, not a noised sample) and the real is the untouched target, so the pair
is clean-vs-clean and ``t`` is a *label describing how hard this example was*,
not a shared corruption level. That is a weaker and cheaper construction, and
it is deliberate -- it needs no change to the sampler and no second noising of
the target. Do not describe this module as Diffusion-GAN, and do not assume the
critic sees matched noise.

Mechanism, and the two not taken
--------------------------------
**Channel concatenation.** The conditioning vector is projected to
``cond_channels``, broadcast over the spatial dims, and concatenated onto the
bridged image, so the inner critic is simply built ``cond_channels`` wider.
This matches ``ConditionalPatchGANDiscriminator``'s ``torch.cat([x, cond], 1)``
and works with *any* registered inner critic, which is the whole point: the arm
picks ``inner_critic`` by name and this class must not care which one it got.

*FiLM* (per-channel scale/shift on the critic's internal activations) and
*projection* (Miyato & Koyama's ``<embedding, features>`` added to the logit)
are both better-conditioned in the literature. Neither is used here because
both require reaching inside the inner critic -- FiLM needs its block
structure, projection needs its penultimate features -- and the inner critic is
whatever the registry hands back. Revisit if this class is ever pinned to one
critic.

Where the numbers come from
---------------------------
``timesteps`` is fed to the embedding **unnormalized** (no ``max_timesteps``).
That is not an oversight: passing a horizon would be a second declaration of
the schedule length beside ``model.params`` (pitfall #17), and the failure mode
is silent -- a horizon that disagrees with the sampler's collapses distinct t
into indistinguishable codes rather than raising. Raw integer t is what DDPM
implementations embed, and it removes the knob entirely.

``num_contrasts`` *is* declared here, matching how every conditioned generator
in this framework takes it. It is validated against the ids at runtime by
``build_contrast_sequence`` -> ``one_hot``, which raises on an out-of-range id
(#9) instead of clamping.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from spectramr.models.blocks.contrast_conditioning import (
    broadcast_conditioning_map,
    contrast_sequence_dim,
)
from spectramr.models.discriminators.sense_bridge_discriminator import (
    _REPR_CHANNELS,
    IMAGE_REPRS,
    SenseBridge,
)
from spectramr.models.interfaces.models import IDiscriminator
from spectramr.models.registry import MODEL_REGISTRY, register_model

logger = logging.getLogger(__name__)


# ``supports_contrast_conditioning=True`` is the load-bearing declaration here.
# ``config_health_checker.check_multi_contrast_model_support`` reads it for the
# critic named by ``model.discriminator_component.name`` and fails a
# multi-contrast arm whose critic does not declare it (#1931). The flag is a
# CLAIM about this ``forward``: it accepts ``contrast_idx`` and uses it. Nothing
# introspects this class at runtime -- the strategy forwards the payload because
# the *registry* says to -- so a class that declares the flag and cannot take
# the kwargs raises on step 1 rather than training unconditioned.
#
# ``accepts_complex`` / ``input_domain`` are copied from the unconditioned
# critic deliberately, not defaulted: ``critic_input_domain`` and
# ``_critic_accepts_complex`` resolve them BY REGISTERED NAME, so omitting them
# here would make ``_align_for_critic`` stack real/imag into a real tensor and
# hand the bridge a k-space it cannot decode.
@register_model(
    role="discriminator",
    name="sense_bridge_patchgan_conditioned",
    training_mode="gan",
    accepts_complex=True,
    input_domain="kspace",
    supports_contrast_conditioning=True,
)
class ContrastConditionedSenseBridgeDiscriminator(IDiscriminator, nn.Module):
    """SENSE-bridge critic conditioned on ``(timesteps, contrast_idx)``.

    Composes :class:`SenseBridge` rather than subclassing
    :class:`SenseBridgeDiscriminator`. That is forced, not stylistic:
    ``SenseBridgeDiscriminator`` is already at inheritance depth 2
    (``-> IDiscriminator -> IModel``), so a subclass would be depth 3 and trip
    ``test_no_new_deep_inheritance`` (non-negotiable 20). Composition also
    expresses the actual relationship better -- this critic needs the same
    bridge but a *wider* inner critic, which subclassing cannot express because
    the parent's ``__init__`` fixes the inner critic's channel count.

    Args:
        in_channels: channels of the BRIDGED IMAGE, before conditioning is
            concatenated. Same meaning as on the unconditioned critic, so an
            arm switching between the two does not have to change it. Must
            agree with ``image_repr``; the inner critic is built for
            ``in_channels + cond_channels``.
        image_repr: ``"magnitude"`` or ``"complex"``, as for the unconditioned
            critic.
        inner_critic: registry name of the critic applied after the bridge.
        critic_kwargs: forwarded to that critic's constructor.
        num_contrasts: one-hot width for ``contrast_idx``. Must match the arm's
            ``data.multi_contrast.n_contrasts``.
        time_embed_dim: width of the sinusoidal ``t`` embedding.
        cond_channels: channels the conditioning is projected to before being
            broadcast and concatenated. Small on purpose -- it widens the inner
            critic's first convolution.
    """

    def __init__(
        self,
        in_channels: int = 1,
        image_repr: str = "magnitude",
        inner_critic: str = "patch_gan",
        critic_kwargs: dict[str, Any] | None = None,
        num_contrasts: int = 3,
        time_embed_dim: int = 64,
        cond_channels: int = 8,
    ) -> None:
        super().__init__()
        if image_repr not in IMAGE_REPRS:
            raise ValueError(
                f"image_repr={image_repr!r} is not a known representation. "
                f"Choose one of {sorted(IMAGE_REPRS)}."
            )
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
        if time_embed_dim < 2:
            raise ValueError(
                f"time_embed_dim must be >= 2 for a sin/cos split; got {time_embed_dim}."
            )
        if cond_channels < 1:
            raise ValueError(
                f"cond_channels={cond_channels} would concatenate nothing, leaving this "
                "critic unconditioned while it declares supports_contrast_conditioning. "
                "Use the unconditioned `sense_bridge_patchgan` instead."
            )

        self.sense_bridge = SenseBridge(image_repr=image_repr)
        self.in_channels = in_channels
        self.inner_critic_name = inner_critic
        self.num_contrasts = num_contrasts
        self.time_embed_dim = time_embed_dim
        self.cond_channels = cond_channels

        # ``contrast_sequence_dim`` is the one owner of "how wide is t + contrast"
        # and raises when num_contrasts < 2, which is the case where a one-hot
        # carries no information.
        self.cond_dim = contrast_sequence_dim(time_embed_dim, num_contrasts, enabled=True)
        self.cond_proj = nn.Linear(self.cond_dim, cond_channels)

        entry = MODEL_REGISTRY[inner_critic]
        critic_cls = entry["class"] if isinstance(entry, dict) else entry
        self.critic = critic_cls(in_channels=in_channels + cond_channels, **(critic_kwargs or {}))

    @property
    def image_repr(self) -> str:
        """The bridge's representation. One owner, read through."""
        return self.sense_bridge.image_repr

    def set_smaps_provider(self, provider: Callable[[int], torch.Tensor | None]) -> None:
        """Install the coil-sensitivity provider on the bridge.

        Same seam as the unconditioned critic: ``diffusion.py`` wires it with a
        ``getattr`` that does nothing when absent, so this delegation must exist
        or the bridge is silently unprovisioned.
        """
        self.sense_bridge.set_smaps_provider(provider)

    def _cond_map(
        self,
        image: torch.Tensor,
        timesteps: torch.Tensor | None,
        contrast_idx: torch.Tensor | None,
    ) -> torch.Tensor:
        """Bind this critic's own widths to the shared conditioning-map builder.

        The construction, the raises and the batch check all live in
        ``blocks.contrast_conditioning`` beside ``build_contrast_sequence``,
        which is the framework's declared single home for contrast conditioning
        (non-negotiable 6). Only ``cond_proj`` is ours -- its output width is
        what widened the inner critic.
        """
        return broadcast_conditioning_map(
            image,
            timesteps,
            contrast_idx,
            time_embed_dim=self.time_embed_dim,
            num_contrasts=self.num_contrasts,
            projection=self.cond_proj,
            owner=type(self).__name__,
        )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        contrast_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score ``x`` after bridging it to image space and appending its label.

        Both loss computers call ``discriminator(x, **critic_cond)``, so the
        conditioning arrives as keywords with these exact names.
        """
        image = self.sense_bridge(x)
        cond_map = self._cond_map(image, timesteps, contrast_idx)
        return self.critic(torch.cat([image, cond_map], dim=1))

    def discriminate(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """IDiscriminator entry point; identical to ``forward``."""
        return self.forward(
            x,
            timesteps=kwargs.get("timesteps"),
            contrast_idx=kwargs.get("contrast_idx"),
        )

    def get_feature_maps(self, x: torch.Tensor, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Inner-critic feature maps, taken on the conditioned bridged image.

        Takes the conditioning too, and raises without it. A conditioned
        critic's activations are only defined given a condition, so returning
        unconditioned features for feature matching would compare activations
        of a function this critic never computes.
        """
        image = self.sense_bridge(x)
        cond_map = self._cond_map(image, kwargs.get("timesteps"), kwargs.get("contrast_idx"))
        conditioned = torch.cat([image, cond_map], dim=1)
        inner = getattr(self.critic, "get_feature_maps", None)
        if callable(inner):
            return inner(conditioned)
        return {"out": self.critic(conditioned)}


__all__ = ["ContrastConditionedSenseBridgeDiscriminator"]
