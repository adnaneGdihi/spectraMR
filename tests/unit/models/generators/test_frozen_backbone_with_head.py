"""frozen_backbone_with_head: the stage-1 encoder loads strictly, stays frozen, and only the head trains."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.contrast_agnostic_encoder import ContrastAgnosticEncoder
from spectramr.models.generators.frozen_backbone_with_head import FrozenBackboneWithHead

_ENC = {"base_channels": 4, "depth": 2, "projector_depth": 2, "projector_hidden_dim": 8}
_HEAD = {"type": "unet_decoder", "contrast_conditioning": True}


def _model(**kw):
    return FrozenBackboneWithHead(**_ENC, embedding_dim=8, head=_HEAD, **kw)


def test_registry_builds_it_with_the_encoder_frozen() -> None:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

    kw = {**_ENC, "embedding_dim": 8, "head": _HEAD}
    model = ModelFactory().create_generator(
        model_type="frozen_backbone_with_head",
        in_channels=1,
        out_channels=1,
        _declared_keys=frozenset(kw),
        **kw,
    )
    assert MODEL_REGISTRY["frozen_backbone_with_head"]["class"] is FrozenBackboneWithHead
    assert not any(p.requires_grad for p in model.backbone.parameters())


def test_only_the_head_trains_and_the_encoder_stays_in_eval() -> None:
    model = _model().train()
    groups = model.get_parameter_groups()
    backbone_ids = {id(p) for p in model.backbone.parameters()}
    assert groups and not any(id(p) in backbone_ids for p in groups)
    assert not model.backbone.training and model.ups.training
    before = [p.clone() for p in model.backbone.parameters()]
    opt = torch.optim.SGD(groups, lr=0.1)
    out = model(torch.randn(2, 1, 16, 16), contrast_idx=torch.tensor([0, 1]))
    assert out.shape == (2, 1, 16, 16)
    out.pow(2).mean().backward()
    opt.step()
    assert all(torch.equal(a, b) for a, b in zip(before, model.backbone.parameters(), strict=True))


def test_the_stage1_checkpoint_loads_strictly() -> None:
    torch.manual_seed(1)
    stage1 = ContrastAgnosticEncoder(out_channels=8, **_ENC)
    model = _model()
    model.load_backbone_state(stage1.state_dict(), source="stage1")
    for a, b in zip(stage1.parameters(), model.backbone.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    wrong = ContrastAgnosticEncoder(
        out_channels=8, base_channels=8, depth=2, projector_depth=2, projector_hidden_dim=8
    )
    with pytest.raises(RuntimeError, match="not a matching"):
        model.load_backbone_state(wrong.state_dict(), source="wrong")


def test_contrast_conditioning_needs_the_id_and_bad_heads_raise() -> None:
    with pytest.raises(ValueError, match="contrast_idx"):
        _model()(torch.randn(1, 1, 16, 16))
    with pytest.raises(ValueError, match=r"head\.type"):
        FrozenBackboneWithHead(**_ENC, head={"type": "mlp"})
    with pytest.raises(ValueError, match="base_channels"):
        FrozenBackboneWithHead(**_ENC, head={"type": "unet_decoder", "base_channels": 64})
