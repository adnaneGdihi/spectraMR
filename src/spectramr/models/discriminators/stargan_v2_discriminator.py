"""StarGAN v2 multi-domain discriminator for MRIxFields any-to-any FIELD translation (Task 9).

The discriminator maps an image ``x`` to one real/fake logit per sample by:

1. Running a shared convolutional downsampling trunk.
2. Producing ``num_domains`` scalar logits via a final linear layer.
3. Selecting the logit at position ``y`` (the sample's **own** domain index)
   using the same vectorised gather that the generator's ``MappingNetwork``
   and ``StyleEncoder`` use.

This follows StarGAN v2 (Choi et al., CVPR 2020) with ``img_channels=1``
and a modest ``base_channels=64`` cap so 64×64 forward passes are fast
enough for unit tests and the audit probe.

Domain-index bounds are checked explicitly: an out-of-range ``y`` **raises**
``ValueError`` rather than clamping (pitfall #9).
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from spectramr.models.registry import register_model

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _discriminator_select_domain(logits: Tensor, y: Tensor, num_domains: int) -> Tensor:
    """Return the ``y``-th domain logit per sample.

    Args:
        logits: Raw domain logits, shape ``[B, num_domains]``.
        y: Long tensor of domain indices, shape ``[B]``.
        num_domains: Number of registered domains.

    Returns:
        Tensor of shape ``[B]`` — the selected real/fake logit per sample.

    Raises:
        ValueError: If any index in ``y`` is outside ``[0, num_domains)``.
            Silent clamping is forbidden (pitfall #9).
    """
    if y.ndim != 1 or y.shape[0] != logits.shape[0]:
        raise ValueError(
            f"Domain label y must be a 1-D tensor of length "
            f"{logits.shape[0]} (batch), got shape {tuple(y.shape)}."
        )
    # Explicit bounds check: two host-syncs (y.min / y.max) per forward call.
    # With num_domains=5 this is negligible for a baseline discriminator.
    # Do NOT replace with IndexError-based control flow to avoid the sync —
    # the intent is a *loud raise* (pitfall #9), not a performance trade-off.
    y_min = int(y.min())
    y_max = int(y.max())
    if y_min < 0 or y_max >= num_domains:
        raise ValueError(
            f"Domain index out of range: y has values in [{y_min}, {y_max}] "
            f"but only {num_domains} domains are registered "
            f"(valid range [0, {num_domains - 1}]). Refusing to clamp (pitfall #9)."
        )
    idx = torch.arange(logits.shape[0], device=logits.device)
    return logits[idx, y]


class _DownBlock(nn.Module):
    """Pre-activation residual block with average-pool downsampling.

    Mirrors the ``ResBlk`` used in the StarGAN v2 generator/style-encoder
    without importing from ``models/generators/`` (keeps the discriminator
    self-contained and avoids intra-``models/`` circular imports).
    """

    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.learned_sc = dim_in != dim_out
        self.act = nn.LeakyReLU(0.2)
        self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
        self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: Tensor) -> Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        return F.avg_pool2d(x, 2)

    def _residual(self, x: Tensor) -> Tensor:
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = F.avg_pool2d(x, 2)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return (self._shortcut(x) + self._residual(x)) / math.sqrt(2)


# ---------------------------------------------------------------------------
# Registered discriminator
# ---------------------------------------------------------------------------


@register_model(
    role="discriminator",
    name="stargan_v2_discriminator",
    training_mode="stargan_v2",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    requires_paired_data=False,
)
class StarGANv2Discriminator(nn.Module):
    """StarGAN v2 multi-domain discriminator.

    Architecture (single-channel, ``base_channels=64``, capped at ``512``):

    * ``from_rgb``: ``img_channels → base_channels`` stem conv.
    * Shared trunk: ``num_downsamples`` downsampling :class:`_DownBlock` blocks,
      each halving spatial resolution and (up to the cap) doubling channels.
    * Global pool: :class:`~torch.nn.LeakyReLU` + :class:`~torch.nn.AdaptiveAvgPool2d`
      (1×1) + flatten → feature vector of ``trunk_dim`` channels.
    * Output head: :class:`~torch.nn.Linear` (``trunk_dim → num_domains``) yielding
      raw domain logits.
    * Domain selection: ``logits[arange(B), y]`` — vectorised gather of the
      sample's **own** domain head (pitfall #9: out-of-range raises).

    Args:
        img_channels: Number of input image channels (``1`` for magnitude MRI).
        num_domains: Number of field-strength domains (5 for MRIxFields: 0.1 T,
            1.5 T, 3 T, 5 T, 7 T).
        base_channels: Initial feature-map width after ``from_rgb``.
        max_channels: Upper bound on feature-map width at any trunk stage.
        num_downsamples: Number of downsampling blocks in the shared trunk.

    Example::

        d = StarGANv2Discriminator(img_channels=1, num_domains=5)
        logit = d(image, domain_label)   # [B] — one logit per sample
    """

    def __init__(
        self,
        img_channels: int = 1,
        num_domains: int = 5,
        base_channels: int = 64,
        max_channels: int = 512,
        num_downsamples: int = 4,
    ) -> None:
        super().__init__()
        self.num_domains = num_domains

        self.from_rgb = nn.Conv2d(img_channels, base_channels, 3, 1, 1)

        blocks: list[nn.Module] = []
        dim_in = base_channels
        for _ in range(num_downsamples):
            dim_out = min(dim_in * 2, max_channels)
            blocks.append(_DownBlock(dim_in, dim_out))
            dim_in = dim_out
        self.trunk = nn.ModuleList(blocks)

        self.act = nn.LeakyReLU(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # One linear layer produces all domain logits jointly.
        self.fc = nn.Linear(dim_in, num_domains)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Classify ``x`` as real or fake under domain ``y``.

        Args:
            x: Input image, shape ``[B, img_channels, H, W]``.
            y: Long tensor of domain indices, shape ``[B]``.

        Returns:
            Real/fake logit for each sample's own domain head, shape ``[B]``.

        Raises:
            ValueError: If any index in ``y`` is outside ``[0, num_domains)``.
        """
        h = self.from_rgb(x)
        for block in self.trunk:
            h = block(h)
        h = self.act(h)
        h = self.pool(h).flatten(1)  # [B, trunk_dim]
        logits = self.fc(h)  # [B, num_domains]
        return _discriminator_select_domain(logits, y, self.num_domains)
