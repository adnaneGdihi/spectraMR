"""vendor_aware_translation: the registry builds the real model and the vendor id conditions it."""

from __future__ import annotations

import warnings

import pytest
import torch

from spectramr.models.generators.vendor_aware_translation import VendorAwareTranslation

_TINY = {"base_channels": 4, "depth": 2, "acquisition_embedding_dim": 8, "n_vendors": 2}


def _factory():
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from spectramr.models.factories.model_factory import ModelFactory

        return ModelFactory()


def test_registry_builds_the_real_model_and_declares_vendor_conditioning() -> None:
    from spectramr.models.registry import MODEL_REGISTRY

    model = _factory().create_generator(
        model_type="vendor_aware_translation",
        in_channels=1,
        out_channels=1,
        _declared_keys=frozenset(_TINY),
        **_TINY,
    )
    entry = MODEL_REGISTRY["vendor_aware_translation"]
    assert entry["class"] is VendorAwareTranslation and isinstance(model, VendorAwareTranslation)
    assert entry["capabilities"].supports_vendor_conditioning is True


def test_vendor_id_changes_the_output_once_the_prompt_is_trained() -> None:
    torch.manual_seed(0)
    model = VendorAwareTranslation(**_TINY).eval()
    x = torch.randn(1, 1, 16, 16)
    a = model(x, vendor_id=torch.tensor([0]))
    torch.testing.assert_close(a, model(x, vendor_id=torch.tensor([1])))  # zero-init prompt
    with torch.no_grad():
        model.vendor_prompt.vendor_lookup.weight[1].fill_(1.0)
    assert not torch.allclose(a, model(x, vendor_id=torch.tensor([1])))


def test_missing_ids_and_bad_knobs_raise() -> None:
    model = VendorAwareTranslation(**{**_TINY, "n_contrasts": 2}).eval()
    x = torch.randn(1, 1, 16, 16)
    with pytest.raises(ValueError, match="vendor_id"):
        model(x)
    with pytest.raises(ValueError, match="contrast_idx"):
        model(x, vendor_id=torch.tensor([0]))
    with pytest.raises(ValueError, match="vendor_init"):
        VendorAwareTranslation(vendor_init="normal")
