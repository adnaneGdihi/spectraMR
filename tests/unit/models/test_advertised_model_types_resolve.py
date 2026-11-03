"""Registry-integrity guard for advertised-vs-registered model types.

Audit finding H6 (2026-05-28): the ``latent_diffusion`` model package was
never imported by the registry discovery scan in
``spectramr.models.init_registry.populate_model_registry``. Its
``@register_model`` decorators therefore never fired, yet
``ldm_latent_discriminator`` (and its ``ldm_patch_latent_discriminator`` /
``ldm_multiscale_latent_discriminator`` siblings) are advertised as valid
``model_type`` values in
``spectramr.config.validation_constants.VALID_MODEL_TYPES``. The framework
dispatches models dynamically by name from YAML config, so an
advertised-but-unregistered name is a latent ``KeyError`` at build time,
not dead code.

The fix wired ``spectramr.models.latent_diffusion`` into the
``populate_model_registry`` scan list. This test is the regression guard:
it triggers full registry population and asserts the three ``ldm_*`` names
resolve in ``MODEL_REGISTRY``. It also asserts that every name resolvable
in ``MODEL_REGISTRY`` is recognized by the advertisement set for the
specific ``ldm_*`` family (a focused slice of the broader
advertised-vs-registered contract).

NB: a *blanket* "every advertised name resolves" assertion is intentionally
NOT made here — at the time of writing, 84 other entries in
``VALID_MODEL_TYPES`` remain unresolved after population (aliases applied
elsewhere, or genuinely-unwired names tracked by separate audit findings).
Tightening the full contract is a separate task; this guard locks down the
H6 family so it cannot silently regress.
"""

from __future__ import annotations

import pytest

# The three names that H6 made resolvable. ``ldm_latent_discriminator`` is the
# one explicitly named in the audit; the other two share the same package +
# module (latent_diffusion/latent_discriminator.py) and the same root cause.
LDM_ADVERTISED_NAMES = (
    "ldm_latent_discriminator",
    "ldm_patch_latent_discriminator",
    "ldm_multiscale_latent_discriminator",
)


@pytest.fixture(scope="module")
def populated_registry() -> dict:
    """Trigger full model-registry population once and return MODEL_REGISTRY.

    Uses the production entry point ``populate_model_registry`` (the same one
    bootstrap drives) so the test exercises the real discovery scan chain
    rather than a hand-rolled import list.
    """
    from spectramr.models.init_registry import populate_model_registry

    populate_model_registry()
    from spectramr.models.registry import MODEL_REGISTRY

    return MODEL_REGISTRY


def test_ldm_latent_discriminator_is_advertised() -> None:
    """The H6 name must be in the advertised model_type set (precondition)."""
    from spectramr.config.validation_constants import VALID_MODEL_TYPES

    assert "ldm_latent_discriminator" in VALID_MODEL_TYPES


@pytest.mark.parametrize("name", LDM_ADVERTISED_NAMES)
def test_ldm_name_is_advertised(name: str) -> None:
    """All three ldm_* discriminator names are advertised as model_types."""
    from spectramr.config.validation_constants import VALID_MODEL_TYPES

    assert name in VALID_MODEL_TYPES, f"{name!r} should be advertised in VALID_MODEL_TYPES"


@pytest.mark.parametrize("name", LDM_ADVERTISED_NAMES)
def test_ldm_name_resolves_in_registry(name: str, populated_registry: dict) -> None:
    """Regression guard for audit finding H6.

    After ``populate_model_registry`` runs, every advertised ldm_* name must
    resolve in MODEL_REGISTRY. Before the fix these were absent because the
    ``latent_diffusion`` package was missing from the discovery scan list.
    """
    assert name in populated_registry, (
        f"{name!r} is advertised in VALID_MODEL_TYPES but does not resolve in "
        "MODEL_REGISTRY after populate_model_registry(). The latent_diffusion "
        "package must be in the init_registry scan list (audit finding H6)."
    )


def test_ldm_latent_discriminator_resolves(populated_registry: dict) -> None:
    """Explicit single-name assertion for the headline H6 model_type."""
    assert "ldm_latent_discriminator" in populated_registry


# ==============================================================================
# Regression: v6.1 registration import must not abort on a duplicate-name collision
# ==============================================================================

# These names are defined in spectramr/models/_v6_1_registrations.py *after* the
# former ``latent_diffusion_residual_head`` placeholder. Because the registry
# collision guard RAISES, that duplicate aborted the whole module import and
# silently dropped every registration below it. Removing the placeholder (audit
# 2026-05-28) must restore them.
_V6_1_NAMES_AFTER_COLLISION = (
    "latent_diffusion_multi_axis_cfg",
    "vendor_aware_translation",
    "multi_param_mapping",
    "cross_field_fm",
    "scas",
    "diffusion",
)


@pytest.mark.parametrize("name", _V6_1_NAMES_AFTER_COLLISION)
def test_v6_1_registration_survives_collision_guard(name: str, populated_registry: dict) -> None:
    """Every v6.1 model_type defined after the removed placeholder must resolve."""
    assert name in populated_registry, (
        f"{name!r} did not register — the _v6_1_registrations import likely "
        "aborted on a duplicate-name collision before reaching it (audit 2026-05-28)."
    )


def test_latent_diffusion_residual_head_resolves_to_real_impl(
    populated_registry: dict,
) -> None:
    """The advertised name resolves to the real generator, not the removed
    ``_BackboneAlias`` placeholder that used to shadow it."""
    assert "latent_diffusion_residual_head" in populated_registry
    cls = populated_registry["latent_diffusion_residual_head"]["class"]
    assert cls.__name__ == "LatentDiffusionWithResidualHead"


# ==============================================================================
# Regression: create_latent_discriminator must RAISE on an unknown
# discriminator_type rather than silently degrading to the "standard" variant
# (pitfall #9 / doc-contract). The docstring advertises exactly three types.
# ==============================================================================


def test_create_latent_discriminator_resolves_advertised_types() -> None:
    """Each advertised discriminator_type returns its concrete class."""
    from spectramr.models.latent_diffusion.latent_discriminator import (
        LatentDiscriminator,
        MultiScaleLatentDiscriminator,
        PatchLatentDiscriminator,
        create_latent_discriminator,
    )

    standard = create_latent_discriminator(discriminator_type="standard")
    patch = create_latent_discriminator(discriminator_type="patch")
    multiscale = create_latent_discriminator(discriminator_type="multiscale")

    assert isinstance(standard, LatentDiscriminator)
    assert isinstance(patch, PatchLatentDiscriminator)
    assert isinstance(multiscale, MultiScaleLatentDiscriminator)


def test_create_latent_discriminator_rejects_unknown_type() -> None:
    """An unknown discriminator_type must raise ValueError, not silently return
    the standard discriminator (pitfall #9: no silent fallback)."""
    from spectramr.models.latent_diffusion.latent_discriminator import (
        create_latent_discriminator,
    )

    with pytest.raises(ValueError, match="Unknown discriminator_type"):
        create_latent_discriminator(discriminator_type="pacth")  # codespell:ignore
