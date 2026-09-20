"""Registry of ``backbone_type`` builders for ``FourierBridgeNetwork``.

The generator selects its backbone through a 15-branch ``if/elif`` chain that is
already in ``tests/architecture/baselines/dispatch_hell.txt``. Adding a branch
per new architecture grows a baselined offender and is the dispatch shape
non-negotiable 6 exists to stop, so new backbones register here instead and the
chain's terminal ``else`` performs one lookup.

The chain's ``else`` used to raise with a hand-written list of supported names.
That list had already drifted -- it omitted aliases the chain accepted -- which is
the second reason this is a registry: :func:`build_backbone` derives the message
from what is actually reachable.

Existing backbones are deliberately NOT migrated here in this change. Each
carries its own kwarg massaging, and moving them is a mechanical rewrite that
would have to be executed and re-tested per backbone (non-negotiable 19) rather
than ridden in on a feature.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class BackboneBuildContext:
    """Everything a registered builder is allowed to depend on.

    A frozen record rather than the raw ``**kwargs`` dict so a builder cannot
    consume a key the next builder needed -- the chain above pops from a shared
    dict, and the ordering coupling that creates is invisible until a backbone is
    reordered.
    """

    in_channels: int
    out_channels: int
    features: tuple[int, ...]
    img_size: tuple[int, int]
    feature_domain: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    # The generator's contrast embedding (``nn.Embedding(num_contrasts,
    # time_embedding_dim)``) is built at ``time_embedding_dim`` width, which
    # ``FourierBridgeNetwork.__init__`` passes explicitly rather than through
    # ``backbone_kwargs`` -- so it never reaches ``kwargs`` above. A builder
    # that needs to size a contrast projection at construction time reads it
    # from here instead. ``None`` means "the caller did not declare a
    # contrast width", not "no contrast conditioning" -- a backbone still
    # raises if it receives a ``contrast_emb`` it cannot reconcile.
    contrast_emb_dim: int | None = None

    @property
    def width(self) -> int:
        """First feature width -- the model's base embedding dimension."""
        return int(self.features[0])

    @property
    def side(self) -> int:
        """Square side used for position-table sizing."""
        return int(self.img_size[0])

    def get(self, key: str, default: Any) -> Any:
        """Read a model_kwarg without mutating the shared dict."""
        value = self.kwargs.get(key, default)
        return default if value is None else value

    @property
    def heads(self) -> int:
        """Head count: the arm's if it declared one, else derived from ``width``.

        ``width`` comes from ``model.model_kwargs.base_channels``, so a fixed
        default (6, the DiT-S figure) raises on every arm that picks a width it
        does not divide. Derived the way the ``swinir`` branch already derives
        its own, and always a divisor. An EXPLICIT head count that does not
        divide still raises inside the attention block -- that one is the arm's
        mistake to see, not a default to paper over.
        """
        declared = self.kwargs.get("heads")
        if declared is not None:
            return int(declared)
        return max(1, self.width // 64)


BACKBONE_BUILDERS: dict[str, Callable[[BackboneBuildContext], nn.Module]] = {}


def register_backbone(
    *names: str,
) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Bind one builder to one or more ``backbone_type`` spellings."""

    def decorate(builder: Callable[[BackboneBuildContext], nn.Module]):
        for name in names:
            if name in BACKBONE_BUILDERS:
                raise ValueError(
                    f"backbone_type {name!r} is already registered by "
                    f"{BACKBONE_BUILDERS[name].__module__}; two builders for one "
                    "name means the winner depends on import order."
                )
            BACKBONE_BUILDERS[name] = builder
        return builder

    return decorate


def registered_backbone_names() -> tuple[str, ...]:
    """Every name this registry can build, sorted."""
    return tuple(sorted(BACKBONE_BUILDERS))


def build_backbone(
    backbone_type: str,
    ctx: BackboneBuildContext,
    *,
    also_supported: tuple[str, ...] = (),
) -> nn.Module:
    """Build a registered backbone, or raise naming everything reachable.

    ``also_supported`` carries the names the caller's own ``if/elif`` handles.
    Listing only this registry would tell a user that ``complex_unet`` does not
    exist, which is how the hand-written message this replaced went stale.
    """
    builder = BACKBONE_BUILDERS.get(backbone_type)
    if builder is None:
        reachable = sorted({*registered_backbone_names(), *also_supported})
        raise ValueError(f"Unknown backbone_type {backbone_type!r}. Supported: {reachable}.")
    return builder(ctx)


@register_backbone("dit")
def _build_dit(ctx: BackboneBuildContext) -> nn.Module:
    from spectramr.models.generators.dit_backbone import DiTBackbone

    return DiTBackbone(
        in_channels=ctx.in_channels,
        out_channels=ctx.out_channels,
        image_size=ctx.side,
        dim=ctx.width,
        depth=ctx.get("depth", 12),
        heads=ctx.heads,
        patch_size=ctx.get("patch_size", 8),
        mlp_ratio=ctx.get("mlp_ratio", 4.0),
        feature_domain=ctx.feature_domain,
        max_timesteps=ctx.get("max_timesteps", None),
        contrast_emb_dim=ctx.contrast_emb_dim,
    )


@register_backbone("uvit", "u_vit")
def _build_uvit(ctx: BackboneBuildContext) -> nn.Module:
    from spectramr.models.generators.uvit_backbone import UViTBackbone

    return UViTBackbone(
        in_channels=ctx.in_channels,
        out_channels=ctx.out_channels,
        image_size=ctx.side,
        dim=ctx.width,
        depth=ctx.get("depth", 13),
        heads=ctx.heads,
        patch_size=ctx.get("patch_size", 8),
        mlp_ratio=ctx.get("mlp_ratio", 4.0),
        feature_domain=ctx.feature_domain,
        max_timesteps=ctx.get("max_timesteps", None),
        contrast_emb_dim=ctx.contrast_emb_dim,
    )


@register_backbone("hat")
def _build_hat(ctx: BackboneBuildContext) -> nn.Module:
    from spectramr.models.generators.hat_backbone import HATBackbone

    return HATBackbone(
        in_channels=ctx.in_channels,
        out_channels=ctx.out_channels,
        dim=ctx.width,
        depth=ctx.get("depth", 6),
        heads=ctx.heads,
        window_size=ctx.get("window_size", 8),
        cab_weight=ctx.get("cab_weight", 0.05),
        feature_domain=ctx.feature_domain,
        max_timesteps=ctx.get("max_timesteps", None),
        contrast_emb_dim=ctx.contrast_emb_dim,
    )


@register_backbone("diffit")
def _build_diffit(ctx: BackboneBuildContext) -> nn.Module:
    from spectramr.models.generators.diffit_backbone import DiffiTBackbone

    return DiffiTBackbone(
        in_channels=ctx.in_channels,
        out_channels=ctx.out_channels,
        image_size=ctx.side,
        dim=ctx.width,
        depth=ctx.get("depth", 12),
        heads=ctx.heads,
        patch_size=ctx.get("patch_size", 8),
        mlp_ratio=ctx.get("mlp_ratio", 4.0),
        feature_domain=ctx.feature_domain,
        max_timesteps=ctx.get("max_timesteps", None),
        contrast_emb_dim=ctx.contrast_emb_dim,
    )
