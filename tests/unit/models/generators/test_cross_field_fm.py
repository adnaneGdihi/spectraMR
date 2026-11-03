"""cross_field_fm: per-stage field FiLM on Swin-V2, and the T1 / tissue heads."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.cross_field_fm import CrossFieldFM

_TINY = {"base_channels": 8, "depths": (1, 2), "num_heads": (1, 2), "window_size": 4}


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_registry_builds_the_real_model_and_refuses_an_unread_knob() -> None:
    from spectramr.models.factories.model_factory import ModelCreationError
    from spectramr.models.registry import MODEL_REGISTRY

    kw = {**_TINY, "field_conditioning_dim": 8}
    model = _factory().create_generator(
        model_type="cross_field_fm",
        in_channels=1,
        out_channels=1,
        _declared_keys=frozenset(kw),
        **kw,
    )
    assert MODEL_REGISTRY["cross_field_fm"]["class"] is CrossFieldFM and isinstance(
        model, CrossFieldFM
    )
    assert set(model.films) == {"encoders_0", "encoders_1", "decoders_0"}
    with pytest.raises(ModelCreationError, match="bottomley_alpha"):
        _factory().create_generator(
            model_type="cross_field_fm",
            in_channels=1,
            out_channels=1,
            _declared_keys=frozenset({"bottomley_alpha"}),
            bottomley_alpha={"csf": 0.05},
        )


def test_the_field_reaches_every_stage() -> None:
    torch.manual_seed(0)
    model = CrossFieldFM(**_TINY, field_conditioning_dim=8).eval()
    x = torch.randn(2, 1, 16, 16)
    field = torch.tensor([0.064, 3.0])
    base = model(x, field_strength=field)
    with torch.no_grad():  # identity at init; move one stage's FiLM off it
        model.films["decoders_0"].mlp[-1].weight.normal_()
    moved = model(x, field_strength=field)
    assert not torch.allclose(base, moved)
    assert not torch.allclose(moved[0], model(x, field_strength=field.flip(0))[0])


def test_conditioning_is_required_when_built_and_absent_otherwise() -> None:
    x = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError, match="field_strength"):
        CrossFieldFM(**_TINY)(x)
    plain = CrossFieldFM(**_TINY, field_conditioning=False)
    assert len(plain.films) == 0 and plain(x).shape == (1, 1, 16, 16)
    with pytest.raises(ValueError, match="sequence_features"):
        CrossFieldFM(**_TINY, sequence_conditioning_dim=3)(x, field_strength=torch.ones(1))


def test_tissue_heads_publish_t1_and_tissue_probabilities() -> None:
    model = CrossFieldFM(**_TINY, field_conditioning=False, tissue_heads=True).eval()
    model(torch.randn(2, 1, 16, 16))
    aux = model.last_aux_maps
    assert aux["t1_ms"].shape == (2, 1, 16, 16) and bool((aux["t1_ms"] > 0).all())
    torch.testing.assert_close(aux["tissue_probabilities"].sum(1), torch.ones(2, 16, 16))
