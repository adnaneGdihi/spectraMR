"""The registry that keeps new backbones out of the generator's if/elif chain.

``FourierBridgeNetwork.__init__`` is a baselined dispatch offender, so the value
here is as much about what does NOT grow as about what builds (non-negotiable 6).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from spectramr.models.generators.backbone_builders import (
    BACKBONE_BUILDERS,
    BackboneBuildContext,
    build_backbone,
    register_backbone,
    registered_backbone_names,
)

NEW = ["dit", "uvit", "u_vit", "hat", "diffit"]


def _ctx(domain: str = "kspace", **kw) -> BackboneBuildContext:
    kwargs = {"depth": 3, "patch_size": 4, "window_size": 8}
    kwargs.update(kw)
    return BackboneBuildContext(
        in_channels=8,
        out_channels=8,
        features=(64, 128),
        img_size=(32, 32),
        feature_domain=domain,
        kwargs=kwargs,
    )


@pytest.mark.parametrize("name", NEW)
def test_every_new_backbone_is_reachable_by_name(name: str) -> None:
    assert name in registered_backbone_names()
    assert isinstance(build_backbone(name, _ctx()), nn.Module)


@pytest.mark.parametrize("name", NEW)
@pytest.mark.parametrize("domain", ["kspace", "image"])
def test_the_declared_domain_reaches_the_built_module(name: str, domain: str) -> None:
    """A builder that dropped ``feature_domain`` would leave every arm on the
    stem's default and the knob would be inert (pitfall 15)."""
    assert build_backbone(name, _ctx(domain)).feature_domain == domain


@pytest.mark.parametrize("name", NEW)
def test_the_built_module_round_trips_the_field(name: str) -> None:
    net = build_backbone(name, _ctx())
    out = net(torch.randn(2, 8, 32, 32), timesteps=torch.zeros(2))
    assert out.shape == (2, 8, 32, 32)


def test_an_unknown_name_raises_and_lists_what_is_reachable() -> None:
    """The message this replaced was hand-written and had already drifted off
    the branches beside it."""
    with pytest.raises(ValueError, match="Unknown backbone_type") as exc:
        build_backbone("definitely_not_a_backbone", _ctx(), also_supported=("complex_unet",))
    message = str(exc.value)
    assert "complex_unet" in message
    for name in NEW:
        assert name in message


def test_registering_a_name_twice_raises() -> None:
    """Two builders for one name means the winner depends on import order."""
    with pytest.raises(ValueError, match="already registered"):
        register_backbone("dit")(lambda ctx: nn.Identity())


def test_head_count_is_derived_from_width_when_undeclared() -> None:
    """``base_channels`` is the arm's to choose, so a fixed default head count
    raises on every width it does not divide."""
    assert _ctx().heads == 1
    wide = BackboneBuildContext(
        in_channels=8,
        out_channels=8,
        features=(384,),
        img_size=(32, 32),
        feature_domain="kspace",
        kwargs={},
    )
    assert wide.heads == 6


def test_a_declared_head_count_wins() -> None:
    assert _ctx(heads=4).heads == 4


def test_an_explicitly_indivisible_head_count_still_raises() -> None:
    """Deriving a default must not swallow an arm's own bad declaration."""
    with pytest.raises(ValueError, match="divide evenly"):
        build_backbone("dit", _ctx(heads=7))


def test_the_context_does_not_let_one_builder_consume_anothers_kwarg() -> None:
    """The chain above pops from a shared dict; this record must not."""
    ctx = _ctx()
    before = dict(ctx.kwargs)
    build_backbone("dit", ctx)
    build_backbone("diffit", ctx)
    assert ctx.kwargs == before


def test_a_none_valued_kwarg_falls_back_to_the_default() -> None:
    """YAML writes an explicit null for an unset key more often than it omits it."""
    assert _ctx(depth=None).get("depth", 12) == 12


def test_every_registered_builder_returns_a_module() -> None:
    for name in registered_backbone_names():
        assert isinstance(BACKBONE_BUILDERS[name](_ctx()), nn.Module)


def test_contrast_emb_dim_defaults_to_none_and_does_not_disturb_existing_arms() -> None:
    """New field, default value: every pre-existing ``_ctx()`` call is unaffected."""
    assert _ctx().contrast_emb_dim is None


@pytest.mark.parametrize("name", NEW)
def test_contrast_emb_dim_reaches_every_new_backbone(name: str) -> None:
    """Finding 27/31: ``FourierBridgeNetwork`` builds the contrast embedding at
    ``time_embedding_dim`` width, which never enters ``backbone_kwargs`` --
    so a builder that needs to size a contrast projection reads it from the
    context's dedicated field, not from ``ctx.kwargs``. All four backbones
    share one owner for this reconciliation (non-negotiable 17), so this is
    parametrized across them rather than pinned to ``dit`` alone."""
    ctx = BackboneBuildContext(
        in_channels=8,
        out_channels=8,
        features=(64, 128),
        img_size=(32, 32),
        feature_domain="kspace",
        kwargs={"depth": 3, "patch_size": 4, "window_size": 8},
        contrast_emb_dim=256,
    )
    net = build_backbone(name, ctx)
    assert net.contrast_proj is not None
    assert net.contrast_proj.in_features == 256
