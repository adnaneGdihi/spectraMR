"""``LossConfigSchema``'s domain lists, and the keys that used to vanish.

This class is ``extra="ignore"``. That is load-bearing for the legacy keys other
migrations own, and it was catastrophic for the loss LISTS: an invented or
misspelled ``*_losses`` key was dropped in silence, ``uses_list_based_losses``
stayed False, the "output_domain required" validator never fired, and the arm
trained its default objective while advertising another one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from spectramr.config.schemas.enums import SignalDomain
from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS, LossConfigSchema
from spectramr.models.losses.build_set import declared_build_set


class TestLossListDomainsIsTheSSOT:
    def test_every_list_maps_to_a_real_signal_domain(self) -> None:
        for block, domain in LOSS_LIST_DOMAINS.items():
            assert isinstance(domain, SignalDomain), block

    def test_every_mapped_list_is_a_declared_field(self) -> None:
        """A mapping entry for a field that does not exist would send the block
        check looking for a list nothing can populate."""
        for block in LOSS_LIST_DOMAINS:
            assert block in LossConfigSchema.model_fields, block

    def test_every_declared_list_is_mapped(self) -> None:
        """The other direction: an unmapped list is invisible to the domain
        check, so entries in it are graded against nothing.

        Matched on the field TYPE, not the name. Two fields end in ``_losses``
        and are not loss lists at all -- ``normalize_losses`` is a bool and
        ``disable_default_losses`` is a ``list[str]`` of names to switch off --
        so a name-suffix comparison would demand domains for them.
        """
        declared = {
            n
            for n, f in LossConfigSchema.model_fields.items()
            if n.endswith("_losses") and "LossComponentConfig" in str(f.annotation)
        }
        assert declared == set(LOSS_LIST_DOMAINS), (
            "declared loss lists and the domain map disagree: "
            f"{declared ^ set(LOSS_LIST_DOMAINS)}"
        )

    def test_the_complex_irregularity_is_pinned(self) -> None:
        """``complex_image`` is spelled ``complex_losses`` in YAML, so the map
        is NOT ``f'{domain.value}_losses'``. That irregularity is why it has to
        be written down once instead of derived in two places."""
        assert LOSS_LIST_DOMAINS["complex_losses"] is SignalDomain.COMPLEX_IMAGE


class TestUndeclaredLossListsRaise:
    def test_custom_losses_raises(self) -> None:
        """Two corpus arms declared ``custom_losses`` -- not a domain at all --
        and lost the entry in silence (issue #655)."""
        with pytest.raises((ValidationError, ValueError)) as exc:
            LossConfigSchema(
                output_domain="image",
                custom_losses=[{"name": "l1", "weight": 1.0}],
            )
        msg = str(exc.value)
        assert "custom_losses" in msg
        assert "silently discarded" in msg

    def test_the_message_names_the_legal_lists(self) -> None:
        """A rejection that does not say where the entries go is worse than the
        silent drop it replaces."""
        with pytest.raises((ValidationError, ValueError)) as exc:
            LossConfigSchema(output_domain="image", spectrum_losses=[])
        msg = str(exc.value)
        for block in LOSS_LIST_DOMAINS:
            assert block in msg

    def test_a_declared_list_still_loads(self) -> None:
        cfg = LossConfigSchema(
            output_domain="image", image_losses=[{"name": "l1", "weight": 1.0}]
        )
        assert cfg.image_losses[0].name == "l1"

    def test_non_loss_extra_keys_are_still_ignored(self) -> None:
        """The guard is scoped to the ``*_losses`` suffix. Flipping the whole
        class to ``extra='forbid'`` is a separate, larger migration (backlog
        W8) and would reject legacy keys this phase does not own."""
        cfg = LossConfigSchema(output_domain="image", some_legacy_knob=3)
        assert not hasattr(cfg, "some_legacy_knob")


class TestLatentLosses:
    def test_the_list_exists(self) -> None:
        """Five losses are registered ``domain='latent'`` and the vocabulary has
        admitted ``latent`` since phase 4a, but there was nowhere to declare
        one -- so every arm that tried lost the entry."""
        assert "latent_losses" in LossConfigSchema.model_fields

    def test_it_counts_toward_list_based_losses(self) -> None:
        """Otherwise an arm with ONLY latent losses reads as having none, and
        falls through to the default objective."""
        cfg = LossConfigSchema(
            output_domain="latent",
            latent_losses=[{"name": "physics_equivariance", "weight": 1.0}],
        )
        assert cfg.uses_list_based_losses is True

    def test_it_requires_a_latent_output_domain(self) -> None:
        """A latent comes out of a LEARNED encoder, so unlike kspace<->image
        there is no transform that manufactures one from another domain."""
        with pytest.raises((ValidationError, ValueError)) as exc:
            LossConfigSchema(
                output_domain="image",
                latent_losses=[{"name": "physics_equivariance", "weight": 1.0}],
            )
        msg = str(exc.value)
        assert "no bridge exists" in msg
        assert "output_domain='latent'" in msg


class TestOutputDomainLegalSet:
    """The legal set is the domains a loss LIST can grade in, not every
    ``SignalDomain``.

    History matters here, because this test previously asserted the opposite. A
    hardcoded tuple knew ``latent`` but not ``spectrum``, so it was replaced with
    ``{d.value for d in SignalDomain}`` to let a spectroscopy arm declare its own
    output domain. That widened the schema past what the builder can do:
    ``loss_builder.py`` bridges four domains and RAISES for ``spectrum`` /
    ``pde_grid`` / ``mesh``, so an arm could validate at load and fail at build.
    Acceptance without buildability is the exact shape this campaign removes.

    The legal set is now ``LOSS_LIST_DOMAINS.values()``. Spectroscopy support
    needs a ``spectrum_losses`` list and a bridge for it, not a wider enum — zero
    corpus arms declare any of the three today.
    """

    def test_every_buildable_domain_is_accepted(self) -> None:
        from spectramr.config.schemas.loss import LOSS_LIST_DOMAINS

        for domain in LOSS_LIST_DOMAINS.values():
            LossConfigSchema(output_domain=domain.value)

    def test_domains_the_builder_cannot_bridge_are_refused_at_load(self) -> None:
        """Both directions: the enum still has them, the schema must not."""
        unbuildable = set(SignalDomain) - set(LOSS_LIST_DOMAINS.values())
        assert unbuildable, "no unbuildable domains left -- update this test"
        for domain in unbuildable:
            with pytest.raises((ValidationError, ValueError)):
                LossConfigSchema(output_domain=domain.value)

    def test_an_unknown_domain_raises(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            LossConfigSchema(output_domain="not_a_domain")

    def test_list_losses_require_an_output_domain(self) -> None:
        with pytest.raises((ValidationError, ValueError), match="output_domain"):
            LossConfigSchema(image_losses=[{"name": "l1", "weight": 1.0}])


class TestEveryListReachesTheBuilder:
    """``get_enabled_losses`` gates the whole list-based path.

    ``LossBuilder._build_all_dynamic`` early-returns when this dict is empty,
    and that return sits BEFORE the branch calling ``_build_list_based_losses``.
    So a list absent from the harvest loop is not under-reported -- it disables
    list-based building entirely for an arm that declares nothing else.

    ``latent_losses`` shipped in exactly that state: declared on the schema,
    looped in the builder, and invisible to the harvest -- the inert mechanism
    it was added to remove.
    """

    def test_each_list_alone_produces_an_enabled_entry(self) -> None:
        for block, (name, domain) in {
            "image_losses": ("l1", "image"),
            "kspace_losses": ("l1", "kspace"),
            "complex_losses": ("l1", "complex_image"),
            "latent_losses": ("physics_equivariance", "latent"),
        }.items():
            cfg = LossConfigSchema(
                output_domain=domain, **{block: [{"name": name, "weight": 1.0}]}
            )
            assert declared_build_set(cfg).get(name) == 1.0, (
                f"{block} does not reach get_enabled_losses, so an arm "
                "declaring only that list builds nothing"
            )

    def test_the_harvest_covers_every_mapped_list(self) -> None:
        """The weight table reads its lists from LOSS_LIST_DOMAINS, so a new list cannot be
        forgotten: it is derived, not re-enumerated (#1966)."""
        from spectramr.models.losses import weights

        assert tuple(LOSS_LIST_DOMAINS) == weights.LOSS_LISTS

    def test_a_disabled_entry_is_still_excluded(self) -> None:
        cfg = LossConfigSchema(
            output_domain="latent",
            latent_losses=[
                {"name": "physics_equivariance", "weight": 1.0, "enabled": False}
            ],
        )
        assert "physics_equivariance" not in declared_build_set(cfg)


class TestComposedComponentsUseTheSameActivationRule:
    """``get_enabled_losses`` held two rules for one question.

    ``composed.components`` and every list in ``LOSS_LIST_DOMAINS`` are lists of
    the *same* type, ``LossComponentConfig``. Twelve lines apart in one method,
    the list loop honoured ``enabled`` and ``weight > 0`` and the composed loop
    honoured neither -- so ``enabled: false``, the spelling
    ``ComposedLossConfig``'s own docstring advertises, silently did nothing.
    Non-negotiable 17: one owner, and the divergence surfaces as a wrong number
    rather than an error.
    """

    @staticmethod
    def _composed(components: list[dict]) -> LossConfigSchema:
        return LossConfigSchema.model_validate({"composed": {"components": components}})

    def test_disabled_component_is_excluded(self) -> None:
        """The regression the fix exists for.

        Planted-violation guard (non-negotiable 15): drop ``component.enabled``
        from ``is_active`` and this assert fails -- ``ssim`` reappears.
        """
        cfg = self._composed(
            [
                {"name": "l1", "weight": 10.0, "enabled": True},
                {"name": "ssim", "weight": 0.1, "enabled": False},
            ]
        )
        assert declared_build_set(cfg) == {"l1": 10.0}

    def test_zero_weight_component_is_excluded(self) -> None:
        """``weight: 0`` is the list loop's other guard, and validates here.

        ``LossComponentConfig.weight`` is ``ge=0.0``, so zero is a legal
        declaration a user can write; it must mean the same thing in both
        blocks. Planted-violation guard: drop ``component.weight > 0`` from
        ``is_active`` and ``perceptual`` reappears with weight 0.0.
        """
        cfg = self._composed(
            [
                {"name": "l1", "weight": 1.0},
                {"name": "perceptual", "weight": 0.0},
            ]
        )
        assert declared_build_set(cfg) == {"l1": 1.0}

    def test_enabled_component_keeps_its_declared_weight(self) -> None:
        """Excluding must not be over-eager: the happy path still passes through."""
        cfg = self._composed([{"name": "l1", "weight": 10.0}])
        assert declared_build_set(cfg) == {"l1": 10.0}

    def test_exclude_defaults_still_applies_to_composed(self) -> None:
        """The one rule the composed loop *did* enforce must survive the merge."""
        cfg = LossConfigSchema.model_validate(
            {
                "composed": {"components": [{"name": "l1", "weight": 1.0}]},
                "policy": {"exclude_defaults": ["l1"]},
            }
        )
        assert "l1" not in declared_build_set(cfg)

    def test_both_blocks_answer_identically_for_identical_components(self) -> None:
        """The property, stated directly rather than through its symptoms.

        A future edit that re-specialises either loop breaks this even if it
        keeps the individual cases above green.
        """
        components = [
            {"name": "l1", "weight": 1.0, "enabled": True},
            {"name": "ssim", "weight": 0.5, "enabled": False},
            {"name": "perceptual", "weight": 0.0, "enabled": True},
        ]
        via_composed = declared_build_set(self._composed(components))

        list_block = next(iter(LOSS_LIST_DOMAINS))
        via_list = declared_build_set(
            LossConfigSchema.model_validate(
                {list_block: components, "output_domain": LOSS_LIST_DOMAINS[list_block]}
            )
        )

        assert via_composed == via_list


class TestReconstructionRetiredSpellingsRaise:
    """#421: the eight keys retired out of `losses.reconstruction` must RAISE.

    `ReconstructionLossesConfig` is ``extra="ignore"``. Without the
    ``reject_renamed_keys("losses.reconstruction")`` mount these keys would not
    be retired -- they would be silently swallowed, and an author would keep
    writing them while training at the schema default. That is the exact
    "stops working AND stops being visible" outcome the renames module docstring
    calls strictly worse than leaving the ambiguity in place.

    The mount is on `ReconstructionLossesConfig` rather than `LossConfigSchema`
    because `renames_for_block` selects records by `mount_path`, and these keys
    live one level below `losses`. That distinction is not cosmetic: the records
    were committed and inert first, and `rename_mounts.audit_mounts` reported
    them as `missing_mount` -- these tests are what turns the mount from
    "declared" into "observed firing".
    """

    RETIRED_TO_PHYSICS = (
        "lambda_snr_preserving",
        "lambda_bloch_residual",
        "lambda_physics_constraint",
        "enable_snr_preserving",
        "enable_bloch_residual",
        "enable_physics_constraint",
    )

    @pytest.mark.parametrize("leaf", RETIRED_TO_PHYSICS)
    def test_physics_duplicate_raises_and_names_its_new_home(self, leaf):
        value = 0.5 if leaf.startswith("lambda_") else True
        with pytest.raises(ValueError) as exc:
            LossConfigSchema(reconstruction={leaf: value})
        assert f"losses.physics.{leaf}" in str(exc.value), (
            "the error must name the surviving spelling; 'unknown key' would "
            "leave the author guessing"
        )

    def test_lambda_content_raises_and_names_both_successors(self):
        """The key was overloaded, so the message has to disambiguate.

        `lambda_content` meant the VGG content/perceptual weight to a
        reconstruction author and the content-consistency weight to a
        disentanglement author (`KNOWN_LOSS_COMPONENTS` lists
        `content_consistency` and `perceptual` as separate components). A `fold`
        would have silently picked one; `raise` makes the author say which.
        """
        with pytest.raises(ValueError) as exc:
            LossConfigSchema(reconstruction={"lambda_content": 0.5})
        message = str(exc.value)
        assert "lambda_perceptual" in message
        assert "lambda_content_consistency" in message

    def test_enable_content_raises_and_names_its_rename(self):
        with pytest.raises(ValueError) as exc:
            LossConfigSchema(reconstruction={"enable_content": True})
        assert "enable_content_consistency" in str(exc.value)

    def test_surviving_spellings_still_load(self):
        """Negative control -- the retirement must not have over-reached."""
        cfg = LossConfigSchema(
            physics={
                "lambda_snr_preserving": 0.5,
                "lambda_bloch_residual": 0.2,
                "lambda_physics_constraint": 0.3,
            },
            reconstruction={"lambda_perceptual": 0.1, "lambda_content_consistency": 0.7},
        )
        assert cfg.physics.lambda_bloch_residual == 0.2
        assert cfg.reconstruction.lambda_perceptual == 0.1
        assert cfg.reconstruction.lambda_content_consistency == 0.7

    def test_content_consistency_does_not_re_create_the_collision(self):
        """The new name must canonicalise to ITSELF.

        `content` was aliased to `perceptual` while being unregistered, which is
        how the original collision hid. If `content_consistency` were aliased the
        same way, the split would be cosmetic and the crash would return.
        """
        from spectramr.models.losses.registry import LossRegistry
        from spectramr.models.losses.weights import canonical_loss_name

        assert canonical_loss_name("content_consistency") == "content_consistency"
        assert "content_consistency" not in LossRegistry._aliases
