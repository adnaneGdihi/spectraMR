"""The conditioned SENSE-bridge critic, and the gate that keeps its flag honest.

``supports_contrast_conditioning`` is a CLAIM about a class's ``forward``, and
``DiffusionTrainingStrategy._critic_conditioning`` acts on it without checking:
it forwards ``{timesteps, contrast_idx}`` because the registry says the critic
takes them. That is deliberate (introspection-guarded forwarding is what
silently dropped ``contrast_idx`` in the first place, #1931 half 1), and it puts
the burden here -- if the flag and the signature disagree, a run dies on step 1.

So this file asserts both halves: that the class honours what it declares, and
-- in :class:`TestEveryDeclaredFlagIsHonoured` -- that every OTHER registered
model declaring the flag could honour it too.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from spectramr.models.discriminators import ContrastConditionedSenseBridgeDiscriminator
from spectramr.models.init_registry import populate_model_registry
from spectramr.models.registry import MODEL_REGISTRY

_COILS = 3
_SIZE = 32


def _critic(**kw):
    defaults = {
        "in_channels": 1,
        "image_repr": "magnitude",
        "inner_critic": "patch_gan",
        "critic_kwargs": {"ndf": 8, "n_layers": 2},
        "num_contrasts": 3,
    }
    defaults.update(kw)
    populate_model_registry()
    d = ContrastConditionedSenseBridgeDiscriminator(**defaults)
    smaps = torch.randn(4, _COILS, _SIZE, _SIZE, dtype=torch.complex64)
    d.set_smaps_provider(lambda b: smaps[:b])
    return d


def _kspace(batch: int = 4):
    return torch.randn(batch, _COILS, _SIZE, _SIZE, dtype=torch.complex64)


class TestConditioningIsConsumed:
    """The payload must reach the score, not merely be accepted."""

    def test_the_score_changes_when_the_contrast_changes(self) -> None:
        """The load-bearing test of this whole change.

        A critic that accepts ``contrast_idx`` and ignores it would satisfy
        every signature check, every registry assertion and every "was it
        forwarded" spy in this repo, and would still be an unconditioned critic
        wearing the flag (pitfall #16). The only proof is that the OUTPUT moves.
        """
        d = _critic().eval()
        k = _kspace()
        t = torch.full((4,), 100)
        a = d(k, timesteps=t, contrast_idx=torch.zeros(4, dtype=torch.long))
        b = d(k, timesteps=t, contrast_idx=torch.ones(4, dtype=torch.long))
        assert not torch.allclose(a, b), (
            "the same k-space scored identically under two different contrasts — "
            "the conditioning is accepted and discarded"
        )

    def test_the_score_changes_when_the_timestep_changes(self) -> None:
        d = _critic().eval()
        k = _kspace()
        c = torch.zeros(4, dtype=torch.long)
        a = d(k, timesteps=torch.full((4,), 5), contrast_idx=c)
        b = d(k, timesteps=torch.full((4,), 900), contrast_idx=c)
        assert not torch.allclose(a, b), (
            "the same k-space scored identically at t=5 and t=900 — the timestep "
            "embedding is not reaching the critic"
        )

    def test_the_inner_critic_is_widened_by_cond_channels(self) -> None:
        """The conditioning occupies real channels, not a discarded tensor."""
        d = _critic(cond_channels=5)
        first = next(m for m in d.critic.modules() if isinstance(m, torch.nn.Conv2d))
        assert first.in_channels == 1 + 5, (
            f"inner critic takes {first.in_channels} channels; expected "
            "in_channels(1) + cond_channels(5)"
        )


class TestItRaisesRatherThanScoringUnconditioned:
    """Non-negotiable 3: no silent fallback to the unconditioned behaviour."""

    def test_missing_timesteps_raises(self) -> None:
        d = _critic()
        with pytest.raises(ValueError, match="timesteps"):
            d(_kspace(), contrast_idx=torch.zeros(4, dtype=torch.long))

    def test_missing_contrast_idx_raises(self) -> None:
        d = _critic()
        with pytest.raises(ValueError, match="contrast_idx"):
            d(_kspace(), timesteps=torch.zeros(4, dtype=torch.long))

    def test_both_missing_raises(self) -> None:
        """The bare ``discriminator(x)`` call an unconditioned path would make."""
        d = _critic()
        with pytest.raises(ValueError):
            d(_kspace())

    def test_a_batch_mismatch_raises_rather_than_broadcasting(self) -> None:
        """Broadcasting would label each sample with another sample's condition.

        This is the silent-corruption shape: shapes that *almost* line up
        broadcast successfully and train on wrong labels.
        """
        d = _critic()
        with pytest.raises(ValueError, match="batch mismatch"):
            d(
                _kspace(4),
                timesteps=torch.zeros(2, dtype=torch.long),
                contrast_idx=torch.zeros(4, dtype=torch.long),
            )

    def test_an_out_of_range_contrast_id_raises(self) -> None:
        """#9, inherited from ``build_contrast_sequence`` -> ``one_hot``."""
        d = _critic(num_contrasts=3)
        with pytest.raises(RuntimeError):
            d(
                _kspace(),
                timesteps=torch.zeros(4, dtype=torch.long),
                contrast_idx=torch.tensor([0, 1, 2, 7]),
            )

    def test_feature_maps_require_the_conditioning_too(self) -> None:
        """A conditioned critic's activations are undefined without a condition."""
        d = _critic()
        with pytest.raises(ValueError):
            d.get_feature_maps(_kspace())

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"image_repr": "phase"}, "not a known representation"),
            ({"in_channels": 2}, "contradicts image_repr"),
            ({"inner_critic": "not_registered_anywhere"}, "not registered"),
            ({"time_embed_dim": 1}, "time_embed_dim"),
            ({"cond_channels": 0}, "cond_channels"),
        ],
    )
    def test_construction_refuses_a_contradictory_config(self, kwargs, match) -> None:
        populate_model_registry()
        args = {"inner_critic": "patch_gan", "critic_kwargs": {"ndf": 8, "n_layers": 2}}
        args.update(kwargs)
        with pytest.raises(ValueError, match=match):
            ContrastConditionedSenseBridgeDiscriminator(**args)

    def test_a_missing_smaps_provider_raises_rather_than_falling_back(self) -> None:
        """Inherited from the shared :class:`SenseBridge`, asserted here too.

        The bridge is composed, not inherited, so nothing about the parent's
        test coverage carries over automatically.
        """
        populate_model_registry()
        d = ContrastConditionedSenseBridgeDiscriminator(
            inner_critic="patch_gan", critic_kwargs={"ndf": 8, "n_layers": 2}
        )
        with pytest.raises(RuntimeError, match="smaps provider"):
            d(
                _kspace(),
                timesteps=torch.zeros(4, dtype=torch.long),
                contrast_idx=torch.zeros(4, dtype=torch.long),
            )


class TestRegistration:
    """The registry entry is what the audit and the strategy both read."""

    def test_it_declares_contrast_conditioning(self) -> None:
        populate_model_registry()
        entry = MODEL_REGISTRY["sense_bridge_patchgan_conditioned"]
        # Nested, not top level (#1916). ``register_model`` no longer fans the
        # conditioning flags out to the entry dict -- there is one owner now --
        # so the old ``entry["supports_contrast_conditioning"]`` is a KeyError.
        assert entry["capabilities"].supports_contrast_conditioning is True

    def test_it_copies_the_domain_capabilities_of_the_unconditioned_critic(self) -> None:
        """The easiest miss in this change, and a silent one.

        ``critic_input_domain`` and ``_critic_accepts_complex`` resolve BY NAME.
        A registration that omitted these would make ``_align_for_critic`` stack
        real/imag into a real tensor and hand the bridge a k-space it cannot
        decode -- no error, just a critic scoring garbage.
        """
        populate_model_registry()
        plain = MODEL_REGISTRY["sense_bridge_patchgan"]["capabilities"]
        cond = MODEL_REGISTRY["sense_bridge_patchgan_conditioned"]["capabilities"]
        assert cond.accepts_complex == plain.accepts_complex is True
        assert cond.input_domain == plain.input_domain == "kspace"

    def test_the_census_is_no_longer_zero(self) -> None:
        """Half 1 asserted 0 aware critics; half 2's whole point is to move it."""
        from spectramr.infrastructure.validation.config_health_checker import (
            _contrast_aware_critics,
        )

        populate_model_registry()
        census = _contrast_aware_critics(MODEL_REGISTRY)
        assert census is not None
        aware, total = census
        assert "sense_bridge_patchgan_conditioned" in aware
        assert total > len(aware) >= 1, (
            "the aware set must be a strict subset — if every critic declares the "
            "flag, this test has stopped discriminating"
        )


#: Registered names whose ``forward`` can receive the conditioning they declare
#: by neither delivery route -- empty since #1938, where ``lcah_encoder`` and
#: ``moe_vae`` dropped a flag neither model has a mechanism for. Never add a
#: name here to make a red run green: fix the signature, or drop the flag from
#: the registration.
_FLAG_DECLARED_BUT_UNHONOURED: frozenset[str] = frozenset()


class TestEveryDeclaredFlagIsHonoured:
    """Fitness gate: the flag is a promise about ``forward``, so check ``forward``.

    Lives here rather than beside the registry helpers because those tests run
    against an isolated ``clean_registry`` fixture; this one must see the REAL
    populated registry, which is the only place the claim can be false.
    """

    @staticmethod
    def _unhonoured() -> set[str]:
        populate_model_registry()
        offenders = set()
        declarers = 0
        for name, entry in MODEL_REGISTRY.items():
            if not isinstance(entry, dict):
                continue
            # Read the NESTED capability (#1916 elected one owner). A
            # ``entry.get("supports_contrast_conditioning")`` here answers None
            # for every model, so ``offenders`` empties and all three tests in
            # this class pass VACUOUSLY -- the same silent zero this gate exists
            # to catch, relocated into the gate itself.
            if not getattr(entry.get("capabilities"), "supports_contrast_conditioning", None):
                continue
            declarers += 1
            cls = entry.get("class")
            if not isinstance(cls, type):
                continue
            try:
                params = inspect.signature(cls.forward).parameters
            except AttributeError:
                # No ``forward`` AT ALL -- a real production shape, since 9 of
                # 588 registered models are not nn.Module subclasses (#801),
                # and previously uncaught, so such an entry CRASHED this census
                # instead of being counted by it.
                #
                # This is the strongest offender there is: the model declares
                # it accepts contrast conditioning and has no forward to accept
                # it through. Skipping it would make the worst case the only
                # invisible one. 0 of the 28 declarers are in this state today
                # (measured), so recording it changes no verdict now -- it just
                # means the census reports the case instead of hiding it.
                offenders.add(name)
                continue
            except (TypeError, ValueError):
                # A ``forward`` whose SIGNATURE cannot be read (C-implemented,
                # or a callable inspect refuses). Not recorded as an offender:
                # such a forward may well accept ``**kwargs``, so an offence
                # cannot be established. Known blind spot, 0 cases today.
                continue
            # Mirror BOTH conditioning delivery routes owned by
            # ``DiffusionTrainingStrategy._forward_through_model``, since that
            # method is the single owner of generator-side delivery (NN17) and a
            # model is honest if it can receive the payload by EITHER of them:
            #
            #   1. as a keyword -- the non-latent branch splats ``**gen_kwargs``,
            #      which carries ``contrast_idx``, so the forward must name the
            #      parameter (either spelling) or take ``**kwargs``;
            #   2. inside ``context`` -- the ``is_latent_diffusion`` branch wraps
            #      it as ``context={"contrast_idx": ...}``, so a forward taking
            #      ``context`` receives it without ever naming ``contrast_idx``.
            #
            # Route 2 was absent when this gate first shipped, and without it
            # three latent models that really do read ``context`` were accused.
            #
            # Signature-level only: this cannot prove the body USES the payload
            # (pitfall #16). :class:`TestConditioningIsConsumed` does that, and
            # only for the one critic this change owns.
            accepts = (
                any(p.kind == p.VAR_KEYWORD for p in params.values())
                or "contrast_idx" in params
                or "contrast_id" in params
                or "context" in params
            )
            if not accepts:
                offenders.add(name)
        # Non-vacuity: a reader wired to the wrong surface makes every assertion
        # below trivially true. Fail loud here instead (NN3), naming the cause.
        assert declarers, (
            "0 models declare supports_contrast_conditioning -- this census is "
            "reading the wrong surface, so every test in this class is vacuous "
            "(#1916 moved the flag onto ModelCapabilities)"
        )
        return offenders

    def test_no_new_model_declares_a_flag_its_forward_cannot_honour(self) -> None:
        new = self._unhonoured() - _FLAG_DECLARED_BUT_UNHONOURED
        assert not new, (
            "these registered models declare supports_contrast_conditioning=True "
            "but their forward() can receive it by neither delivery route — no "
            "contrast_idx/contrast_id, no **kwargs (the keyword route) and no "
            f"context (the latent route): {sorted(new)}. Conditioning them then "
            "either raises TypeError or is silently dropped, depending on which "
            "caller drives them. Fix the signature or drop the flag — do not add "
            "them to the baseline."
        )

    def test_the_baseline_has_no_stale_entries(self) -> None:
        """A ratchet that never shrinks stops being read (non-negotiable 20)."""
        stale = _FLAG_DECLARED_BUT_UNHONOURED - self._unhonoured()
        assert not stale, (
            f"these baselined names are no longer unhonoured: {sorted(stale)}. "
            "Either the signature grew a delivery route or the registration "
            "dropped the flag; both are exits. Delete them from "
            "_FLAG_DECLARED_BUT_UNHONOURED."
        )

    def test_the_new_critic_is_not_in_the_baseline(self) -> None:
        """A gate that exempts the thing it was written for is not a gate."""
        assert "sense_bridge_patchgan_conditioned" not in _FLAG_DECLARED_BUT_UNHONOURED
        assert "sense_bridge_patchgan_conditioned" not in self._unhonoured()
