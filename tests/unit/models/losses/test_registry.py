"""The loss registry's domain vocabulary, and the two ratchets over it.

Phase 6 bound loss placement to the declared signal domain. Doing that surfaced
that "this loss has no domain constraint" and "nobody ever annotated this loss"
were the same value, so half the registry read as deliberate when it was merely
un-audited. These tests pin the distinction and drain the backlog.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

import spectramr.models.losses  # noqa: F401  -- import-time registration
from spectramr.config.schemas.enums import SignalDomain
from spectramr.models.losses.registry import (
    AGNOSTIC_DOMAIN,
    LossRegistry,
    compatible_domains,
    get_loss_capabilities,
)


class TestAgnosticIsAPositiveDeclaration:
    """``domain="agnostic"`` must survive the adaptation to ``LossCapabilities``.

    It used to not. "agnostic" simply had no ``Domain`` equivalent, so it fell
    out of ``_LEGACY_DOMAIN_TO_LITERAL`` as ``None`` -- indistinguishable from a
    loss nobody had annotated. The generic losses passed the block check for the
    same reason an un-audited one did, so any tightening of the unannotated
    branch would have silently started rejecting ``l1``.
    """

    def test_the_flag_is_set_and_the_domain_stays_none(self) -> None:
        caps = get_loss_capabilities("l1")
        assert caps is not None
        assert caps.domain_agnostic is True
        assert caps.domain is None, (
            "an agnostic loss must NOT claim a concrete domain -- that would "
            "put a non-domain into the SignalDomain vocabulary"
        )

    def test_a_domained_loss_does_not_set_the_flag(self) -> None:
        caps = get_loss_capabilities("physics_equivariance")
        assert caps is not None
        assert caps.domain == "latent"
        assert caps.domain_agnostic is False

    def test_the_flag_resolves_through_aliases(self) -> None:
        """``mse`` and ``mae`` are aliases of ``l2``/``l1``; the corpus declares
        ``mse`` 378 times, so alias resolution is the common path."""
        for alias in ("mse", "mae", "mean_squared_error"):
            caps = get_loss_capabilities(alias)
            assert caps is not None and caps.domain_agnostic, alias

    def test_agnostic_is_not_a_signal_domain(self) -> None:
        assert AGNOSTIC_DOMAIN not in {d.value for d in SignalDomain}

    def test_the_adversarial_family_is_agnostic(self) -> None:
        """A GAN loss grades DISCRIMINATOR LOGITS, not the signal.

        `StandardGANLoss.compute_generator_loss(fake_outputs_d)` is
        `BCEWithLogitsLoss` over the discriminator's output. The
        discriminator's *input* has a domain; the adversarial loss's input does
        not, so pairing one with any loss block is meaningless rather than
        wrong. They were all unannotated, which said the same thing by
        accident.
        """
        for name in (
            "gan_standard",
            "gan_lsgan",
            "gan_hinge",
            "gan_wgan",
            "gan_ralsgan",
            "gan_composite",
        ):
            caps = get_loss_capabilities(name)
            assert caps is not None, f"{name} lost its registration"
            assert caps.domain_agnostic, f"{name} must be domain-agnostic"

    def test_the_elementwise_distances_are_all_declared_agnostic(self) -> None:
        """The user-visible contract: these are legal under every loss block.

        The corpus relies on it -- ``l1`` sits under ``image_losses`` 334 times
        AND under ``kspace_losses`` 7 times, and both are correct.
        """
        for name in ("l1", "l2", "huber", "charbonnier", "smooth_l1"):
            caps = get_loss_capabilities(name)
            assert caps is not None, f"{name} lost its registration"
            assert caps.domain_agnostic, f"{name} must be domain-agnostic"


class TestCompatibleDomainsFiltersTheSecondVocabulary:
    """``compatible_with=`` collected strategy names alongside domains.

    The block check compares its entries against a domain, so an entry holding
    a strategy name can never match -- the exemption those registrations asked
    for has never once fired.
    """

    #: Losses whose ``compatible_with`` holds NO usable domain. Each needs its
    #: owner to say whether it meant a domain or meant a strategy. This set may
    #: shrink, never grow.
    KNOWN_NON_DOMAIN_COMPATIBLE: ClassVar[frozenset[str]] = frozenset(
        {
            "bloch_source_consistency",
            "cocycle_consistency",
            "dispersion_prior",
            "field_flow_velocity",
            "field_identity",
            "gw_cross_field",
            "heteroscedastic_ulf",
            "latent_cycle",
        }
    )

    def test_only_real_domains_come_back(self) -> None:
        for name in LossRegistry._loss_domains:
            assert compatible_domains(name) <= {d.value for d in SignalDomain}, name

    def test_the_legacy_spelling_is_mapped(self) -> None:
        """``complex`` must arrive as ``complex_image``; leaving it raw is how
        the checker's private map agreed with the registry by luck."""
        assert "complex_image" in compatible_domains("teichmuller_geodesic")
        assert "complex" not in compatible_domains("teichmuller_geodesic")

    def test_a_populated_entry_still_resolves(self) -> None:
        """Anti-vacuity: if the filter dropped everything, the ratchet below
        would pass while the escape hatch was entirely dead."""
        assert compatible_domains("sense_adjoint_l1") == {"image", "kspace"}

    @pytest.mark.parametrize("name", sorted(KNOWN_NON_DOMAIN_COMPATIBLE))
    def test_the_polluted_entries_are_still_the_known_ones(self, name: str) -> None:
        raw = LossRegistry._loss_domains.get(name, {}).get("compatible_with") or []
        assert raw, f"{name} no longer declares compatible_with -- drop it here"
        assert not compatible_domains(name), (
            f"{name} now resolves to a real domain; remove it from "
            "KNOWN_NON_DOMAIN_COMPATIBLE"
        )

    def test_no_new_pollution(self) -> None:
        polluted = {
            n
            for n, m in LossRegistry._loss_domains.items()
            if (m.get("compatible_with") or []) and not compatible_domains(n)
        }
        assert polluted <= self.KNOWN_NON_DOMAIN_COMPATIBLE, (
            f"new compatible_with entries name something that is not a domain: "
            f"{sorted(polluted - self.KNOWN_NON_DOMAIN_COMPATIBLE)}. "
            "compatible_with holds SignalDomain values; a strategy or workflow "
            "name there is silently inert."
        )


class TestUnannotatedRatchet:
    """104 of 214 registrations carry no domain, so the block check has no
    opinion on half the registry.

    Seeded from today's count and allowed only to fall. Annotating is a judgment
    call per loss -- a wrong one turns a working arm into a hard error -- so this
    forces the work without guessing it in bulk.
    """

    #: Measured 2026-07-31 after the second annotation pass (24 more losses:
    #: the adversarial family, the documented image-space set, and the
    #: documented agnostic set). 104 -> 96 -> 72.
    CEILING = 72

    def test_the_unannotated_count_does_not_grow(self) -> None:
        unannotated = set(LossRegistry._custom_losses) - set(LossRegistry._loss_domains)
        assert len(unannotated) <= self.CEILING, (
            f"{len(unannotated)} losses carry no domain metadata, up from "
            f"{self.CEILING}. A new loss must declare `domain=` -- either a "
            f"SignalDomain value or {AGNOSTIC_DOMAIN!r}. Newly unannotated: "
            f"{sorted(unannotated)[:10]}"
        )

    def test_lowering_the_ceiling_is_not_forgotten(self) -> None:
        unannotated = set(LossRegistry._custom_losses) - set(LossRegistry._loss_domains)
        assert len(unannotated) >= self.CEILING - 4, (
            f"only {len(unannotated)} unannotated losses remain (ceiling "
            f"{self.CEILING}) -- lower CEILING so the ratchet keeps its grip. "
            "The tolerance is deliberately tight: a wide one lets the ceiling "
            "drift above the real count, which is slack the ratchet then never "
            "recovers."
        )


class TestThereIsNoSecondLookupPath:
    def test_an_unknown_loss_raises_rather_than_falling_back(self) -> None:
        """The docstrings promised a ``UnifiedLossFunctionFactory`` fallback for
        "built-in" losses. No such class exists anywhere in the tree, and every
        loss including ``l1`` arrives through the decorator."""
        with pytest.raises(ValueError, match="Unknown loss"):
            LossRegistry.create("definitely_not_a_registered_loss")

    def test_no_docstring_still_promises_the_fallback(self) -> None:
        """Two docstrings told a reader to look for a class that is not there.

        The module docstring keeps one mention, explaining that it does not
        exist; the class and method docstrings must not describe it as a live
        lookup path.
        """
        for obj in (LossRegistry, LossRegistry.create):
            assert "UnifiedLossFunctionFactory" not in (obj.__doc__ or "")


class TestDomainStringsThatMapToNothing:
    """The blind spot is CLOSED: no loss resolves to unannotated-by-accident.

    A loss declares its domain as a free string, adapted through
    ``_LEGACY_DOMAIN_TO_LITERAL`` into ``LossCapabilities.domain``. Four losses
    declared ``physics`` or ``bloch_synthesis`` — neither a ``SignalDomain``
    member — so the adapter yielded ``domain=None, domain_agnostic=False`` and
    they fell into a hole visible to neither guard:

    * the unannotated-count ratchet skipped them, because they ARE in
      ``_loss_domains``;
    * ``check_loss_domain_block_match`` skipped them, because their resolved
      domain is ``None``.

    This class used to PIN that set of four so it could shrink but not grow. It
    has now shrunk to zero, by the route its own docstring left open: not by
    inventing a ``SignalDomain`` for ``physics``, but by declaring those strings
    :data:`NON_SIGNAL_DOMAINS` and carrying them on ``domain_agnostic``. Those
    losses grade gradient waveforms, Hamiltonians and Bloch simulations — they
    name no ``losses.<domain>_losses`` block, so the exemption is the correct
    answer and is now stated rather than reached by a failed dict lookup.

    Behaviour is unchanged (they passed the block check before and pass it now);
    what changed is that the pass is deliberate, and survives any tightening of
    the ``None`` branch.
    """

    #: Emptied 2026-08-02. Never grow: a domain string that maps to nothing now
    #: raises at registration, so an addition here means someone widened
    #: REGISTRABLE_DOMAINS without deciding what the new value MEANS.
    UNMAPPED: ClassVar[frozenset[str]] = frozenset()

    @staticmethod
    def _unmapped() -> set[str]:
        from spectramr.models.losses.registry import (
            LossRegistry,
            get_loss_capabilities,
        )

        out = set()
        for name in LossRegistry._loss_domains:
            caps = get_loss_capabilities(name)
            if caps is not None and caps.domain is None and not caps.domain_agnostic:
                out.add(name)
        return out

    def test_the_set_does_not_grow(self) -> None:
        new = sorted(self._unmapped() - self.UNMAPPED)
        assert not new, (
            f"{len(new)} loss(es) declare a domain string that maps to no "
            f"SignalDomain, so they are invisible to the unannotated ratchet AND "
            f"to check_loss_domain_block_match: {new}. Declare a real "
            f"SignalDomain value, AGNOSTIC_DOMAIN, or a NON_SIGNAL_DOMAINS entry."
        )

    def test_the_pin_has_no_stale_entries(self) -> None:
        stale = sorted(self.UNMAPPED - self._unmapped())
        assert not stale, (
            f"{stale} now resolve to a real domain — remove them from UNMAPPED "
            f"so the set keeps shrinking"
        )

    def test_the_detector_still_fires(self) -> None:
        """Anti-vacuity: an EMPTY result must mean 'none', not 'cannot see any'.

        The detector can no longer be exercised through ``register_loss`` — an
        unmappable domain raises there now — so the hole is injected directly
        into the metadata the adapter reads.
        """
        from spectramr.models.losses.registry import LossRegistry

        LossRegistry._loss_domains["_probe_unmapped"] = {"domain": "nowhere"}
        try:
            assert "_probe_unmapped" in self._unmapped()
        finally:
            LossRegistry._loss_domains.pop("_probe_unmapped", None)
        # Deliberately no "and now it is empty" assertion here: `_loss_domains`
        # is class-level state that sibling tests mutate, so under a shuffled
        # order that would assert their cleanup rather than anything about the
        # registry. `test_the_set_does_not_grow` owns that claim.


class TestAnUnregistrableDomainRaises:
    """A typo and a deliberate non-signal domain used to be indistinguishable.

    ``domain=`` was a free string dropped through ``_LEGACY_DOMAIN_TO_LITERAL``,
    so ``domain="imagee"`` silently became "unannotated" and skipped every check
    it was meant to face. ``register_loss``'s own docstring meanwhile advertised
    ``'physics'`` as legal while the map discarded it — the advertised set and
    the honoured set had drifted apart (CLAUDE.md #9/#15).
    """

    def test_a_typo_is_refused_at_registration(self) -> None:
        from spectramr.models.losses.registry import register_loss

        with pytest.raises(ValueError, match="not a registrable domain"):
            register_loss("_probe_typo", domain="imagee")

    def test_every_advertised_domain_is_accepted(self) -> None:
        """The control. Refusing a legal value would be worse than the bug."""
        import torch.nn as nn

        from spectramr.models.losses.registry import (
            REGISTRABLE_DOMAINS,
            LossRegistry,
            register_loss,
        )

        for i, dom in enumerate(sorted(REGISTRABLE_DOMAINS)):
            name = f"_probe_ok_{i}"
            try:
                register_loss(name, domain=dom)(type(f"P{i}", (nn.Module,), {}))
            finally:
                LossRegistry._custom_losses.pop(name, None)
                LossRegistry._loss_domains.pop(name, None)

    def test_omitting_the_domain_is_still_legal(self) -> None:
        """``domain=None`` means unannotated, which remains a valid state."""
        import torch.nn as nn

        from spectramr.models.losses.registry import LossRegistry, register_loss

        try:
            register_loss("_probe_none")(type("PN", (nn.Module,), {}))
        finally:
            LossRegistry._custom_losses.pop("_probe_none", None)
            LossRegistry._loss_domains.pop("_probe_none", None)

    def test_the_non_signal_domains_are_exempt_not_unannotated(self) -> None:
        """The four losses now carry a STATED exemption."""
        from spectramr.models.losses.registry import get_loss_capabilities

        for name in (
            "hamiltonian_energy_conservation",
            "gradient_hardware_compliance",
            "bloch_consistency",
            "bloch_signal_synthesis_consistency",
        ):
            caps = get_loss_capabilities(name)
            assert caps is not None and caps.domain_agnostic, name

    def test_signal_domain_losses_are_untouched(self) -> None:
        """Control: the exemption must not leak onto real signal domains."""
        from spectramr.models.losses.registry import get_loss_capabilities

        for name, expected in (
            ("ssim", "image"),
            ("complex_spatial_gradient", "kspace"),
            ("sense_adjoint_l1", "kspace"),
        ):
            caps = get_loss_capabilities(name)
            assert caps is not None
            assert caps.domain == expected and not caps.domain_agnostic, name


class TestGetLossClass:
    """The class accessor must mirror `create`'s resolution, minus construction.

    `loss_builder` needs the CLASS to ask what its constructor accepts. Reaching
    into `_custom_losses` directly would key on the raw name, so `mae` and `l1`
    would read as two losses -- the exact trap `canonical_name` exists to stop.
    """

    def test_resolves_a_canonical_name(self):
        from spectramr.models.losses.registry import LossRegistry

        cls = LossRegistry.get_loss_class("sobolev_kspace")
        assert cls is not None
        assert cls.__name__ == "SobolevKSpaceLoss"

    def test_resolves_through_the_alias_table(self):
        from spectramr.models.losses.registry import LossRegistry

        for name in LossRegistry.list_available():
            aliases = LossRegistry.get_loss_aliases(name)
            if aliases:
                assert LossRegistry.get_loss_class(aliases[0]) is (
                    LossRegistry.get_loss_class(name)
                ), f"alias {aliases[0]!r} must resolve to the same class as {name!r}"
                return
        pytest.skip("no aliased loss in the registry")

    def test_returns_none_for_an_unregistered_name(self):
        from spectramr.models.losses.registry import LossRegistry

        assert LossRegistry.get_loss_class("definitely_not_a_loss") is None

    def test_agrees_with_is_registered_across_the_whole_registry(self):
        """One resolver, not two that agree until they don't."""
        from spectramr.models.losses.registry import LossRegistry

        for name in LossRegistry.list_available():
            assert LossRegistry.is_registered(name)
            assert LossRegistry.get_loss_class(name) is not None


class TestAliasesCannotShadowCanonicalNames:
    """``create`` reads the alias table first, so an alias equal to another loss's
    canonical name hijacks it, and a canonical name equal to an existing alias is
    unreachable. Which one wins depended on import order.
    """

    @staticmethod
    def _cleanup(*names: str) -> None:
        for name in names:
            LossRegistry._custom_losses.pop(name, None)
            LossRegistry._loss_domains.pop(name, None)
            for alias, canon in list(LossRegistry._aliases.items()):
                if canon == name:
                    LossRegistry._aliases.pop(alias)

    def test_an_alias_naming_another_loss_is_refused(self) -> None:
        import torch.nn as nn

        from spectramr.models.losses.registry import register_loss

        l1_cls = LossRegistry.get_loss_class("l1")
        try:
            with pytest.raises(ValueError, match="hijack create"):
                register_loss("_probe_hijack", aliases=["l1"])(type("PH", (nn.Module,), {}))
            assert LossRegistry.get_loss_class("l1") is l1_cls
            assert "_probe_hijack" not in LossRegistry._custom_losses
        finally:
            self._cleanup("_probe_hijack")

    def test_a_canonical_name_that_is_already_an_alias_is_refused(self) -> None:
        import torch.nn as nn

        from spectramr.models.losses.registry import register_loss

        assert LossRegistry.canonical_name("mae") == "l1"
        try:
            with pytest.raises(ValueError, match="already an alias of 'l1'"):
                register_loss("mae")(type("PM", (nn.Module,), {}))
            assert "mae" not in LossRegistry._custom_losses
        finally:
            LossRegistry._custom_losses.pop("mae", None)

    def test_a_refused_alias_remap_leaves_no_partial_entry(self) -> None:
        """The remap check used to run after the class was already stored."""
        import torch.nn as nn

        from spectramr.models.losses.registry import register_loss

        try:
            with pytest.raises(ValueError, match="refusing to remap"):
                register_loss("_probe_remap", aliases=["mse"])(type("PR", (nn.Module,), {}))
            assert "_probe_remap" not in LossRegistry._custom_losses
            assert LossRegistry.canonical_name("mse") == "l2"
        finally:
            self._cleanup("_probe_remap")

    def test_reregistering_the_same_class_stays_idempotent(self) -> None:
        """Control: a module reload re-runs every decorator with identical arguments."""
        from spectramr.models.losses.registry import register_loss

        l1_cls = LossRegistry.get_loss_class("l1")
        assert l1_cls is not None
        register_loss("l1", aliases=["mae", "mean_absolute_error"], domain="agnostic")(l1_cls)
        assert LossRegistry.get_loss_class("mae") is l1_cls

    def test_no_live_alias_collides_with_a_canonical_name(self) -> None:
        collisions = {
            alias: canon
            for alias, canon in LossRegistry._aliases.items()
            if alias in LossRegistry._custom_losses and alias != canon
        }
        assert collisions == {}


class TestGetLossInfoResolvesAliases:
    def test_an_alias_returns_the_canonical_losses_info(self) -> None:
        info = LossRegistry.get_loss_info("mae")
        assert info["description"] == LossRegistry.get_loss_info("l1")["description"]
        assert "mae" in info["aliases"]

    def test_an_unknown_name_still_raises(self) -> None:
        with pytest.raises(KeyError):
            LossRegistry.get_loss_info("definitely_not_a_loss")
