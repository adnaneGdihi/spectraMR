"""Names `stubs.py` must not alias: each is registered by its own class."""

from __future__ import annotations

import pytest

from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

#: name -> the class its own @register_model binds (the former stubs.py target in the comment).
_OWN_CLASS = {
    "uncertainty_wrapper": "UncertaintyWrapperGenerator",  # not MCDropoutGenerator
    "pin_nerf": "PINNeRFGenerator",  # not PINNNeRFGenerator
    "monai_swin_unetr": "MonaiSwinUNETR",  # not UNETRGenerator
    "qmri_network": "QMRINetworkGenerator",  # not QMapGenerator
    "dvs": "DVSGenerator",  # not DifferentiableSlicer
    "diffusion_recon": "DiffusionRecon",  # not ReconstructionWithDiffusionPrior
    "implicit_representation": "ImplicitRepresentationGenerator",  # not ImplicitMRIField
    "latent_diffusion_hierarchical": "LatentDiffusionHierarchical",  # not LatentGaussianDiffusion
}


@pytest.mark.parametrize(("name", "cls_name"), sorted(_OWN_CLASS.items()))
def test_name_binds_its_own_class(name: str, cls_name: str) -> None:
    populate_model_registry()
    assert MODEL_REGISTRY[name]["class"].__name__ == cls_name


def test_live_aliases_still_bind() -> None:
    populate_model_registry()
    for alias, canonical in {
        "pinn": "physics_driven",
        "neural_ca": "neural_cache",
        "foundation_model": "mamba",
        "neural_ode_adaptive": "neural_ode",
    }.items():
        assert MODEL_REGISTRY[alias]["class"] is MODEL_REGISTRY[canonical]["class"]
