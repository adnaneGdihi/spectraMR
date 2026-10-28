"""Regression test for the Wave-2 ledger-restored model aliases.

Twenty name-variant ``model_type`` strings were purged from
``VALID_MODEL_TYPES`` in commit ``d8ccb8452`` and then re-added as
aliases in ``spectramr.models.stubs.register_aliases`` (Phase 1 of
``TODO/deleted_model_types_reimplementation_plan.md``).

``ALIAS_TABLE`` below is the frozen source of truth. Each alias MUST
resolve to the *same class object* as its canonical name after the
registry is populated — otherwise the advertised name is an
"audit-surface lie" (green Tier-0 audit, ``KeyError`` at build time).
If a future edit drops an alias from ``stubs.py``, this test fails
rather than silently regressing the registry surface.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from spectramr.config.validation_constants import VALID_MODEL_TYPES  # noqa: E402
from spectramr.models.init_registry import populate_model_registry  # noqa: E402
from spectramr.models.registry import MODEL_REGISTRY  # noqa: E402

# alias -> canonical registered name
ALIAS_TABLE: dict[str, str] = {
    "MambaReconstruction": "mamba_reconstruction",
    "capsule_networks": "capsule_network",
    "kspace_gpt_foundation": "kspace_gpt",
    "mixture_density": "mixture_density_network",
    "moe_kan": "moe_kan_generator",
    "nafnets": "nafnet",
    "normalizing_flow_image": "normalizing_flow",
    "reversible_network": "invertible_network",
    "standard_vit": "vision_transformer",
    "stylegan": "stylegan2",
    "swin_transformer_kan": "swin_kan_transformer",
    "trellis_gaussian": "trellis_gaussian_vae",
    "trellis_mesh": "trellis_mesh_vae",
    "unrolled_list": "unrolled_reconstruction",
    "vit": "vision_transformer",
    "vit_with_kan": "vit_kan",
    "vq_vae": "vqvae",
    # conditional_diffusion_cfg / multimodal_conditional_diffusion retired
    # 2026-08-12 (diffusion cleanup, phase 1.3) — zero arms, and both named a
    # KNOB on ConditionalDiffusionGenerator rather than a distinct model, so the
    # name read as a selector and selected nothing. Their absence is now pinned
    # by ``test_retired_names_stay_retired`` below.
    "neural_ode_adaptive": "neural_ode",
}


@pytest.fixture(scope="module", autouse=True)
def _populated() -> None:
    populate_model_registry()


@pytest.mark.registry_contract
@pytest.mark.parametrize("alias,canonical", sorted(ALIAS_TABLE.items()))
def test_alias_resolves_to_canonical(alias: str, canonical: str) -> None:
    assert canonical in MODEL_REGISTRY, (
        f"canonical '{canonical}' missing from registry; alias '{alias}' "
        f"cannot be wired"
    )
    assert alias in MODEL_REGISTRY, (
        f"alias '{alias}' not registered — check stubs.register_aliases()"
    )
    assert MODEL_REGISTRY[alias]["class"] is MODEL_REGISTRY[canonical]["class"], (
        f"alias '{alias}' resolves to a DIFFERENT class than canonical "
        f"'{canonical}'"
    )


@pytest.mark.registry_contract
@pytest.mark.parametrize("alias", sorted(ALIAS_TABLE))
def test_alias_is_advertised(alias: str) -> None:
    """Every wired alias must also be advertised in VALID_MODEL_TYPES."""
    assert alias in set(VALID_MODEL_TYPES), (
        f"alias '{alias}' is wired in stubs.py but not advertised in "
        f"VALID_MODEL_TYPES"
    )


# ---------------------------------------------------------------------------
# One name, one class (diffusion cleanup phase 1.3, 2026-08-12)
# ---------------------------------------------------------------------------
#
# The tests above pin what IS registered. Nothing pinned what must NOT be, and
# nothing checked the two registries against each other -- which is how
# ``zero_shot_transfer`` came to mean ZeroRFReconstructor in the ModelFactory
# registry and RectifiedFlowGenerator in MODEL_REGISTRY at the same time.
# Which model a config got depended on which registry the build path asked.

#: Names retired in phase 1.3. Absence is the assertion: re-adding any of them
#: revives a name that either duplicated a canonical key or bound to the wrong
#: class. Each had ZERO declarations across all 1508 committed experiment YAMLs.
RETIRED_NAMES: dict[str, str] = {
    "sde_diffusion": "duplicate of score_based_diffusion (same class, same mode)",
    "zero_shot_transfer": "bound to two different classes; use zerorf or rectified_flow",
    "metal_diffusion": "facade alias of cold_diffusion_process; set degradation instead",
    "conditional_diffusion_cfg": "a knob on conditional_diffusion, not a model",
    "multimodal_conditional_diffusion": "a knob on conditional_diffusion, not a model",
}


@pytest.mark.registry_contract
@pytest.mark.parametrize("name,reason", sorted(RETIRED_NAMES.items()))
def test_retired_names_stay_retired(name: str, reason: str) -> None:
    assert name not in MODEL_REGISTRY, f"'{name}' is back in MODEL_REGISTRY — {reason}"
    assert name not in set(VALID_MODEL_TYPES), (
        f"'{name}' is back in VALID_MODEL_TYPES — {reason}. Advertising a name the "
        f"registry does not carry is the audit-surface lie registry.py:158 warns about."
    )


@pytest.mark.registry_contract
def test_zerorf_is_not_rectified_flow() -> None:
    """The specific swap that motivated this section.

    ``experiment_88_zero_shot_transfer.yaml`` declares ``model_type: zerorf``.
    init_registry aliased BOTH ``zerorf`` and ``zero_shot_transfer`` onto
    ``rectified_flow``. ``zerorf`` was saved only by a ``not in MODEL_REGISTRY``
    guard that happened to short-circuit because the factory registered the real
    class first — an import-order accident, not a contract.
    """
    zerorf = MODEL_REGISTRY["zerorf"]["class"]
    assert zerorf.__name__ == "ZeroRFReconstructor", (
        f"model_type 'zerorf' resolves to {zerorf.__name__}; experiment_88 would "
        f"train a different model than its config names"
    )
    assert zerorf is not MODEL_REGISTRY["rectified_flow"]["class"]


#: Names bound to a DIFFERENT class in ModelFactory's registry than in
#: MODEL_REGISTRY, as measured on 2026-08-12. This is a ratchet, not an
#: allowlist: the set may only shrink. Some pairs are benign (a ``*Generator``
#: wrapper around the same model); others look like genuine swaps -- ``cycle_gan``
#: -> CycleGAN vs ResNetGenerator, ``rl_scanner`` -> DeepQScannerAgent vs
#: ActiveScanner. Untangling them changes real model bindings and is tracked
#: separately; this test exists so the count cannot grow unnoticed.
# ``enhanced_unet`` left this set when ``register_aliases`` stopped running as an
# import-time side effect and became an explicit call from
# ``populate_model_registry``. It is the one alias bound by direct assignment
# rather than ``setdefault`` -- stubs.py calls that binding "unconditional" --
# but the assignment only ever ran if the pass fired after the generators had
# registered. It now always does, so the two registries agree. Do not re-add it
# without re-reading that comment; a re-appearance means the timing regressed.
KNOWN_CROSS_REGISTRY_DISAGREEMENTS = frozenset(
    (
        "attention_unet", "cycle_gan", "disentangled_vae",
        "enhanced_kan_unet", "equivariant_unet",
        "fusion_unet", "gaussian_splatting", "laplace_diffusion", "medical_dino",
        "motion_inr", "pggs_pipeline", "physics_aware_gaussian_splatter", "pin_nerf",
        "refined_kan_unet", "rl_scanner", "sparse_vae", "stable_vit", "structured_vae",
        "symbolic_regression", "uncertainty_wrapper", "vae", "vit_kan", "vq_vae",
        "vqgan", "vqvae",
    )  # fmt: skip
)


@pytest.mark.registry_contract
# Constructing a ModelFactory is not deprecated (the warning lives on
# ``create_model``), and it is still the only thing that populates the
# generator registry this test compares against — reading it IS the point.
def test_no_new_name_binds_to_two_different_classes() -> None:
    from spectramr.models.factories.model_factory import ModelFactory

    populate_model_registry()
    generators = ModelFactory()._registry._generators

    disagreements = {
        name
        for name, factory_cls in generators.items()
        if isinstance(MODEL_REGISTRY.get(name), dict)
        and MODEL_REGISTRY[name].get("class") is not factory_cls
    }
    new = disagreements - KNOWN_CROSS_REGISTRY_DISAGREEMENTS
    assert not new, (
        f"{len(new)} name(s) now mean different classes in the two registries: "
        f"{sorted(new)}. Which model a config gets would depend on which registry "
        f"the build path asks. Bind the name once, or give the second class its own name."
    )

    fixed = KNOWN_CROSS_REGISTRY_DISAGREEMENTS - disagreements
    assert not fixed, (
        f"{sorted(fixed)} no longer disagree — remove them from "
        f"KNOWN_CROSS_REGISTRY_DISAGREEMENTS so the ratchet keeps tightening."
    )
