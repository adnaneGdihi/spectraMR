"""The consumption gate must ask "can this read run?", not "does this text exist?".

``test_schema_key_consumption.py`` built its consumer index with ripgrep: a key
counted as consumed if any line under ``src/`` mentioned it. Whether that line
can ever execute was not part of the question. That is how
``logging.tracking.tensorboard_dir`` passed as consumed while its only reader
sat in ``ComprehensiveLoggingService.__init__`` -- a class the bootstrap never
constructs (#928 / #932).

The predicate under test answers the harder question, and its failure mode is
deliberately one-sided: **ambiguity resolves to reachable**. This repo dispatches
through registries and a DI container by design, so a class whose name escapes
into a container, a registry table or a call argument cannot be proven dead, and
a verdict of "unreachable" would license deleting live code.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from spectramr.config.key_reachability import (
    PACKAGE_DIR,
    ReachabilityVerdict,
    ReadEvidence,
    is_key_reachable,
)


def test_package_dir_is_this_repos_own_checkout() -> None:
    """``key_reachability`` must analyse THIS checkout, not whatever ``import
    spectramr`` happens to resolve to.

    ``PACKAGE_DIR = Path(__file__).resolve().parents[1]`` inside
    ``key_reachability.py`` is derived from that module's own file location, so
    it is safe by construction -- but nothing previously checked that the
    module doing the deriving is actually the one under this repo root rather
    than a `spectramr` importable from a different checkout on `sys.path` /
    `PYTHONPATH`. Today a wrong-tree import fails loudly for an unrelated
    reason (this module did not exist on `dev` before this branch); once this
    branch merges that accidental protection is gone, so this guard replaces
    it with a real one.
    """
    this_repo_root = Path(__file__).resolve().parents[3]
    assert this_repo_root / "src" / "spectramr" == PACKAGE_DIR, (
        f"key_reachability.PACKAGE_DIR ({PACKAGE_DIR}) does not sit under this "
        f"test file's own repo root ({this_repo_root}) -- the reachability "
        "analysis would be scanning a different checkout's source tree "
        "entirely. Check PYTHONPATH / which `spectramr` import resolved."
    )


@pytest.fixture(scope="module")
def unreachable_key() -> str:
    """A key whose only read sits in a class nothing constructs.

    Chosen from the complement of ``KNOWN_UNCONSUMED`` (measured 2026-08-12; it
    is not on that list, so text-matching calls it consumed). Its leaf,
    ``max_steps``, has exactly one read in ``src/spectramr/`` outside
    ``config/schemas/``: ``MRIEnv.__init__`` in
    ``models/reasoning/rl_acquisition.py``. ``MRIEnv`` is constructed nowhere in
    ``src/``, ``runners/``, ``scripts/`` or ``tools/`` -- only in
    ``tests/smoke/exotic/test_rl_acquisition.py`` -- so the rg index counted a
    read that no production path can run. That is the whole defect, in one key.

    Pinned by name: constructing ``MRIEnv`` for real, or deleting it, turns this
    test red rather than leaving it silently vacuous.
    """
    return "model.trellis_vae.trainer.max_steps"


def test_a_read_inside_a_never_constructed_class_is_not_consumption() -> None:
    """The rg index cannot tell a live read from a dead one. This must.

    Fixture: a read that exists in src/ but sits in a class no construction path
    reaches. Text-matching says consumed; reachability says no.
    """
    verdict = is_key_reachable("logging.tracking.enable_tensorboard")
    assert verdict.sites, "no read found at all -- fixture is stale, pick another key"
    assert verdict.reachable, "post-#932 this key IS reachable; the predicate says otherwise"


def test_an_unconstructed_class_read_is_reported_unreachable(unreachable_key: str) -> None:
    verdict = is_key_reachable(unreachable_key)
    assert not verdict.reachable
    assert "not constructed" in verdict.reason


def test_the_verdict_is_a_frozen_dataclass_carrying_its_evidence() -> None:
    """``sites`` is part of the contract, not debug output.

    Wave 4 makes wire-or-delete calls off this list, so a verdict that says
    "unreachable" without naming the reads it judged is not actionable.
    """
    verdict = is_key_reachable("logging.tracking.enable_tensorboard")

    assert dataclasses.is_dataclass(verdict)
    assert isinstance(verdict, ReachabilityVerdict)
    assert isinstance(verdict.sites, tuple)
    assert all(site.count(":") >= 1 for site in verdict.sites)
    with pytest.raises(dataclasses.FrozenInstanceError):
        verdict.reachable = False  # type: ignore[misc]


def test_a_key_no_source_line_mentions_is_unreachable_for_that_reason() -> None:
    """No sites at all is a different verdict from dead sites, and must say so."""
    verdict = is_key_reachable("nothing.in.this.tree.spells_this_leaf_name")

    assert not verdict.reachable
    assert verdict.sites == ()
    assert "no read" in verdict.reason


class TestTheTwoCausesOfUnreachableAreDistinguishable:
    """``reachable=False`` collapses two facts; ``evidence`` must separate them.

    Wave 4 consumes this predicate *programmatically* to make wire-or-delete
    calls, and the two causes carry very different weight:

    * ``NO_LIVE_READ`` -- reads exist and the call graph showed them dead. As
      strong as this analysis gets.
    * ``NO_READ_FOUND`` -- no token found at all. Usually nothing reads the key,
      but ``**cfg.model_dump()`` splatting and a runtime-built field name
      (``getattr(cfg, f"compute_{name}")``) look exactly the same from here, and
      those are two of the patterns the design constraint named. Absence of
      evidence, not evidence of absence.

    Without the discriminator a consumer would have to string-match ``reason``,
    or would silently treat the weaker case as the stronger one.
    """

    def test_a_dead_read_is_reported_as_a_call_graph_finding(self) -> None:
        verdict = is_key_reachable("model.trellis_vae.trainer.max_steps")

        assert not verdict.reachable
        assert verdict.evidence is ReadEvidence.NO_LIVE_READ
        assert verdict.sites, "NO_LIVE_READ without sites is a contradiction"

    def test_a_key_with_no_token_anywhere_is_reported_as_absent_evidence(self) -> None:
        verdict = is_key_reachable("nothing.in.this.tree.spells_this_leaf_name")

        assert not verdict.reachable
        assert verdict.evidence is ReadEvidence.NO_READ_FOUND
        assert verdict.sites == ()

    def test_a_reachable_key_is_reported_as_a_live_read(self) -> None:
        verdict = is_key_reachable("logging.tracking.enable_tensorboard")

        assert verdict.reachable
        assert verdict.evidence is ReadEvidence.LIVE_READ

    def test_the_three_states_are_exhaustive_and_not_derived_from_reachable(
        self,
    ) -> None:
        """Anti-vacuity: a field that merely mirrored ``reachable`` would be noise.

        Both falsy states must be observable, and they must differ.
        """
        dead = is_key_reachable("model.trellis_vae.trainer.max_steps")
        absent = is_key_reachable("nothing.in.this.tree.spells_this_leaf_name")

        assert dead.reachable is absent.reachable is False
        assert dead.evidence is not absent.evidence
        assert {v.evidence for v in (dead, absent)} < set(ReadEvidence)


class TestAmbiguityResolvesToReachable:
    """The design constraint that most needs a regression guard.

    An analysis that guessed "unreachable" whenever it could not resolve a
    dynamic dispatch would call most of this repo dead: components are resolved
    through ``@register_*`` tables and ``resolve_service(...)``, never by direct
    instantiation. Every one of those is an ambiguity, and every ambiguity must
    come back reachable with the ambiguity named.
    """

    def test_a_class_that_is_only_referenced_never_called_is_live(self) -> None:
        """The escape branch: a bare name handed somewhere, never instantiated.

        ``LoggingServiceFactory`` is never written as ``LoggingServiceFactory(...)``
        anywhere -- ``bootstrap.py`` reaches it as a namespace,
        ``LoggingServiceFactory.create(...)``, and ``infrastructure/logging/
        __init__.py`` puts the bare name in a lazy re-export table. Neither is a
        construction this analysis can confirm, and neither is one it may rule
        out: the same shape covers ``register_service(SomeType, ...)`` and every
        ``name -> class`` registry in the tree.
        """
        from spectramr.config.key_reachability import class_liveness

        verdict = class_liveness("LoggingServiceFactory")

        assert verdict.live
        assert "live by ambiguity" in verdict.reason
        assert verdict.evidence, "a live verdict must name the site that made it live"

    def test_a_decorator_registered_class_is_live_without_a_call_site(self) -> None:
        """``@register_metric("brenner_focus", ...)`` -- the registry owns the call.

        Nothing in the tree writes ``BrennerFocus(...)``; the metrics registry
        instantiates it from a YAML string. A call-graph analysis that demanded a
        literal call site would declare all 992 decorator-registered classes dead.
        """
        from spectramr.config.key_reachability import class_liveness

        verdict = class_liveness("BrennerFocus")

        assert verdict.live
        assert "carries a decorator" in verdict.reason

    def test_a_base_class_is_live_through_its_live_subclass(self) -> None:
        """Constructing a subclass runs the base's methods, so a read there is live.

        The reverse does not hold, and must not: a live base says nothing about
        an unused subclass.
        """
        from spectramr.config.key_reachability import class_liveness

        verdict = class_liveness("MetricsMixin")

        assert verdict.live
        assert "base of live class" in verdict.reason

    def test_the_dotted_import_path_of_a_dynamic_strategy_counts(self) -> None:
        """The one that decides whether this analysis is safe to act on.

        ``strategy_factory.py`` maps ``training_mode`` to a dotted class path
        *string* and imports it. Not reading that string reported 40 live
        training strategies as never constructed -- the exact wrong-direction
        verdict that would license deleting them.
        """
        from spectramr.config.key_reachability import class_liveness

        for name in ("FieldBridgeStrategy", "RecoverabilityVIBStrategy"):
            assert class_liveness(name).live, name


class TestAccessorMediatedReads:
    """`external_reads` (#1925), and the four ways it could be wrong.

    The index matches a read by finding the field name as a source *token*.
    `build_loss_weight_table` names none of the 112 paths it reads -- it builds
    every name at runtime from `LAMBDA_SECTIONS` and `model_fields_set` -- so 52
    of them look identical to "nothing reads this". That is the case
    `ReadEvidence.NO_READ_FOUND`'s own docstring says a human must read the
    consumer before acting on; `external_reads` is how that reading is carried
    back in.

    Every plant below is a *measured* case on this checkout, not a constructed
    one, so each is also a live pin on the shape it names.
    """

    #: The path #1925 was filed about: fully wired through `losses.physics`, and
    #: named by no token anywhere in the tree.
    TOKENLESS = "losses.physics.lambda_bloch_residual"

    #: Same LEAF as `TOKENLESS`, under a prefix the accessor never receives --
    #: `config_health_checker` hands it the root `config.losses`. 54 paths have
    #: this shape. It is the mutation-killer: key the lookup on
    #: `dotted_path.rsplit(".", 1)[-1]` and this one goes green.
    SAME_LEAF_OTHER_PREFIX = "training.multi.stages.stage_config.loss.physics.lambda_bloch_residual"

    #: In the accessor's read set AND read by a live token elsewhere.
    ALSO_LIVE = "losses.gan.lambda_adv"

    #: Real read sites, every one of them dead. Sits in the gate's own
    #: `UNREACHABLE_READ` ledger, so it is a pinned example of the shape.
    DEAD_SITES = "ema.stability_threshold"

    @staticmethod
    def _map():
        from spectramr.models.losses.weights import accessor_read_paths

        return accessor_read_paths()

    def test_without_the_map_the_tokenless_read_is_invisible(self) -> None:
        """The bug, pinned. Delete `external_reads` and this is the whole story."""
        verdict = is_key_reachable(self.TOKENLESS)
        assert not verdict.reachable
        assert verdict.evidence is ReadEvidence.NO_READ_FOUND
        assert verdict.sites == ()

    def test_the_map_turns_it_into_an_accessor_read(self) -> None:
        verdict = is_key_reachable(self.TOKENLESS, external_reads=self._map())
        assert verdict.reachable
        assert verdict.evidence is ReadEvidence.ACCESSOR_READ
        assert "build_loss_weight_table" in verdict.reason

    def test_the_same_leaf_under_another_prefix_stays_unread(self) -> None:
        """Keyed by full path, never by leaf.

        This is the assertion that dies if anyone "simplifies" the lookup to
        match the leaf, the way the token index above legitimately does.
        """
        assert self.SAME_LEAF_OTHER_PREFIX.endswith(self.TOKENLESS.rsplit(".", 1)[-1])
        verdict = is_key_reachable(self.SAME_LEAF_OTHER_PREFIX, external_reads=self._map())
        assert not verdict.reachable
        assert verdict.evidence is ReadEvidence.NO_READ_FOUND

    def test_a_live_token_read_outranks_the_declaration(self) -> None:
        """Order decides the evidence, never `reachable`.

        A call graph is stronger evidence than a declaration, so a path that is
        both must report the call graph. Swap the two checks and this goes
        `ACCESSOR_READ` -- reachable either way, but the verdict stops naming the
        site a maintainer would go read.
        """
        assert self.ALSO_LIVE in self._map()
        verdict = is_key_reachable(self.ALSO_LIVE, external_reads=self._map())
        assert verdict.reachable
        assert verdict.evidence is ReadEvidence.LIVE_READ
        assert verdict.sites

    def test_the_declaration_outranks_dead_token_reads(self) -> None:
        """The shape no real path in the loss map has, so it is injected.

        A key with real-but-dead sites that an accessor also reads IS read. Check
        the map *after* the dead-sites case and this answers `NO_LIVE_READ` --
        "every read is dead" -- which is false. `sites` must survive: the ledger
        tests ask whether an entry really has read sites, and losing them here
        would refile the key under the wrong ledger section.
        """
        before = is_key_reachable(self.DEAD_SITES)
        assert before.evidence is ReadEvidence.NO_LIVE_READ and before.sites

        after = is_key_reachable(self.DEAD_SITES, external_reads={self.DEAD_SITES: "a probe"})
        assert after.reachable
        assert after.evidence is ReadEvidence.ACCESSOR_READ
        assert after.sites == before.sites

    def test_the_map_is_honoured_rather_than_hardcoded(self) -> None:
        """A fabricated path proves the parameter is read, not a baked-in list."""
        fake = "nothing.in.this.tree.spells_this_leaf_name"
        assert is_key_reachable(fake).evidence is ReadEvidence.NO_READ_FOUND
        verdict = is_key_reachable(fake, external_reads={fake: "a fabricated accessor"})
        assert verdict.reachable
        assert verdict.evidence is ReadEvidence.ACCESSOR_READ
        assert "a fabricated accessor" in verdict.reason

    def test_omitting_the_map_changes_nothing(self) -> None:
        """The parameter is additive: every existing caller keeps its verdict."""
        for key in (self.TOKENLESS, self.ALSO_LIVE, self.DEAD_SITES):
            assert is_key_reachable(key) == is_key_reachable(key, external_reads=None)
            assert is_key_reachable(key) == is_key_reachable(key, external_reads={})
