"""Tests for the metrics registry singleton.

Targets ``spectramr.core.metrics.registry``. Singleton dict-of-classes with
case-insensitive aliases; canonical entry-points are
``register_metric``, ``get_metric``, ``compute_metric``,
``list_available``.

Categories:

- ``IMetric`` runtime-checkable Protocol
- Decorator registration: name + aliases, alias dispatches to canonical
- ``get_metric`` raises ``KeyError`` on unknown names with available list
- ``list_available`` is sorted
- ``is_registered`` works for both names and aliases
- ``get_aliases`` returns the alias list for a metric
- ``compute_metric`` shorthand: instantiate + call in one step

Implementation note: the registry is a class-level singleton populated
by import-time decorators. Tests register an isolated dummy class so
they don't depend on (or pollute) production registrations.
"""

from __future__ import annotations

from typing import ClassVar

import pytest
import torch

from spectramr.core.metrics.registry import (
    IMetric,
    MetricsRegistry,
    compute_metric,
    get_metric,
    list_available,
    register_metric,
)

# ---------------------------------------------------------------------------
# Test dummy
# ---------------------------------------------------------------------------


class _DummyMetric:
    """Minimal IMetric-compatible test double."""

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor, **kwargs) -> float:
        return float(((prediction - target).abs().mean() * self.scale).item())

    @property
    def name(self) -> str:
        return "dummy"

    @property
    def higher_is_better(self) -> bool:
        return False


@pytest.fixture(autouse=True)
def _register_dummy():
    """Register the dummy metric for the duration of each test."""
    register_metric("__dummy__", aliases=["__dummy_alias__", "DummyAlias"])(_DummyMetric)
    yield
    # Clean up — purge only the dummy entries to leave production state intact.
    MetricsRegistry._metrics.pop("__dummy__", None)
    for alias in ["__dummy_alias__", "dummyalias"]:
        MetricsRegistry._aliases.pop(alias, None)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_dummy_is_imetric() -> None:
    """Runtime ``isinstance`` against the protocol passes."""
    assert isinstance(_DummyMetric(), IMetric)


def test_non_metric_object_rejected() -> None:
    """Plain object lacking ``__call__`` / ``name`` is not an IMetric."""

    class _NotAMetric:
        pass

    assert not isinstance(_NotAMetric(), IMetric)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_get_by_canonical_name() -> None:
    """``get_metric('__dummy__')`` returns a fresh instance."""
    metric = get_metric("__dummy__")
    assert isinstance(metric, _DummyMetric)


def test_get_by_alias() -> None:
    """Aliases dispatch to the same canonical class."""
    metric = get_metric("__dummy_alias__")
    assert isinstance(metric, _DummyMetric)


def test_get_alias_is_case_insensitive() -> None:
    """Aliases are stored lower-cased on registration."""
    metric = get_metric("DummyAlias")
    assert isinstance(metric, _DummyMetric)


def test_get_unknown_name_raises_keyerror() -> None:
    """Unknown name → ``KeyError`` listing available metrics."""
    with pytest.raises(KeyError, match="Unknown metric"):
        get_metric("totally_unknown_metric_12345")


def test_get_constructor_kwargs_forwarded() -> None:
    """``get_metric(..., **kwargs)`` forwards kwargs to the constructor."""
    metric = get_metric("__dummy__", scale=2.0)
    assert metric.scale == 2.0


# ---------------------------------------------------------------------------
# Listing & introspection
# ---------------------------------------------------------------------------


def test_list_available_includes_dummy() -> None:
    """The registered dummy appears in ``list_available``."""
    assert "__dummy__" in list_available()


def test_list_available_is_sorted() -> None:
    """``list_available`` is alphabetically sorted."""
    available = list_available()
    assert available == sorted(available)


def test_get_aliases_returns_list() -> None:
    """``get_aliases`` returns all aliases for a metric."""
    aliases = MetricsRegistry.get_aliases("__dummy__")
    assert "__dummy_alias__" in aliases


def test_is_registered_by_name_and_alias() -> None:
    """``is_registered`` is true for both canonical and alias forms."""
    assert MetricsRegistry.is_registered("__dummy__") is True
    assert MetricsRegistry.is_registered("__dummy_alias__") is True
    assert MetricsRegistry.is_registered("__unknown__") is False


# ---------------------------------------------------------------------------
# compute_metric shorthand
# ---------------------------------------------------------------------------


def test_compute_metric_calls_metric() -> None:
    """``compute_metric`` instantiates and invokes in one step."""
    pred = torch.zeros(4)
    target = torch.ones(4)
    score = compute_metric("__dummy__", pred, target)
    assert pytest.approx(score) == 1.0  # |0 - 1| = 1


# ---------------------------------------------------------------------------
# Re-registration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# higher_is_better injection (see metric_directions.py)
# ---------------------------------------------------------------------------


def test_injects_higher_is_better_from_direction_map(monkeypatch) -> None:
    """A metric that doesn't self-declare direction gets it injected at
    registration time from the central SSOT map."""
    from spectramr.core.metrics import metric_directions

    monkeypatch.setitem(metric_directions.METRIC_HIGHER_IS_BETTER, "__inject_test__", True)

    class _NoDirection:
        def __call__(self, p, t, **k):
            return 0.0

    try:
        register_metric("__inject_test__")(_NoDirection)
        assert _NoDirection.higher_is_better is True
    finally:
        MetricsRegistry._metrics.pop("__inject_test__", None)


def test_self_declared_direction_not_overwritten(monkeypatch) -> None:
    """A class that declares its own ``higher_is_better`` is never clobbered
    by the injection, even if the map disagrees."""
    from spectramr.core.metrics import metric_directions

    monkeypatch.setitem(metric_directions.METRIC_HIGHER_IS_BETTER, "__self_decl__", True)

    class _SelfDeclared:
        higher_is_better = False

        def __call__(self, p, t, **k):
            return 0.0

    try:
        register_metric("__self_decl__")(_SelfDeclared)
        assert _SelfDeclared.higher_is_better is False  # map value ignored
    finally:
        MetricsRegistry._metrics.pop("__self_decl__", None)


def test_unmapped_metric_left_untouched() -> None:
    """A metric neither self-declaring nor in the map is registered without
    a ``higher_is_better`` (the gate tests catch this for real metrics)."""

    class _Unmapped:
        def __call__(self, p, t, **k):
            return 0.0

    try:
        register_metric("__unmapped_test__")(_Unmapped)
        assert not hasattr(_Unmapped, "higher_is_better")
    finally:
        MetricsRegistry._metrics.pop("__unmapped_test__", None)


def test_reregister_same_class_is_no_op() -> None:
    """Re-registering the same class doesn't raise."""
    register_metric("__dummy__")(_DummyMetric)
    assert "__dummy__" in MetricsRegistry._metrics


# --- capability metadata (spec §1.2: requires_reference / needs / context) ---


def test_capability_metadata_from_decorator_kwargs() -> None:
    """needs / requires_measurement_context / requires_reference are stored and
    introspectable via the registry classmethods."""

    class _NRMetric:
        higher_is_better = False

        def __call__(self, p, t=None, **k):
            return 0.0

    try:
        register_metric(
            "__nr_ctx__",
            requires_reference=False,
            needs=("y_kspace", "mask"),
            direction="lower",
        )(_NRMetric)
        assert MetricsRegistry.needs("__nr_ctx__") == ("y_kspace", "mask")
        assert MetricsRegistry.needs_context("__nr_ctx__") is True
        assert MetricsRegistry.requires_reference("__nr_ctx__") is False
        # direction="lower" maps to higher_is_better when the class did not
        # self-declare it (here it self-declares, so it stays False).
        assert _NRMetric.higher_is_better is False
    finally:
        for store in (
            MetricsRegistry._metrics,
            MetricsRegistry._needs,
            MetricsRegistry._requires_context,
            MetricsRegistry._requires_reference,
        ):
            store.pop("__nr_ctx__", None)


def test_full_reference_metric_defaults() -> None:
    """A metric registered without NR kwargs reports no context need and
    requires a reference (full-reference default)."""

    class _FR:
        higher_is_better = True

        def __call__(self, p, t, **k):
            return 1.0

    try:
        register_metric("__fr_default__")(_FR)
        assert MetricsRegistry.needs("__fr_default__") == ()
        assert MetricsRegistry.needs_context("__fr_default__") is False
        assert MetricsRegistry.requires_reference("__fr_default__") is True
    finally:
        for store in (
            MetricsRegistry._metrics,
            MetricsRegistry._needs,
            MetricsRegistry._requires_context,
            MetricsRegistry._requires_reference,
        ):
            store.pop("__fr_default__", None)


def test_needs_context_resolves_through_alias() -> None:
    """needs()/needs_context() resolve an alias to the canonical key."""

    class _NRAlias:
        higher_is_better = False

        def __call__(self, p, t=None, **k):
            return 0.0

    try:
        register_metric("__nr_alias__", aliases=["NRAliasUpper"], needs=("coil_maps",))(_NRAlias)
        assert MetricsRegistry.needs("NRAliasUpper") == ("coil_maps",)
        assert MetricsRegistry.needs_context("NRAliasUpper") is True
    finally:
        MetricsRegistry._aliases.pop("nraliasupper", None)
        for store in (
            MetricsRegistry._metrics,
            MetricsRegistry._needs,
            MetricsRegistry._requires_context,
            MetricsRegistry._requires_reference,
        ):
            store.pop("__nr_alias__", None)


# ---------------------------------------------------------------------------
# direction-knob validation (regression: WS-2 fix)
# ---------------------------------------------------------------------------
#
# FIX under test (registry.py): ``register_metric(..., direction=...)`` now
# validates ``direction in ('higher','lower')`` UNCONDITIONALLY, before the
# ``declares_own`` / ``METRIC_HIGHER_IS_BETTER`` branching. Previously the
# membership check lived inside the ``elif`` that only runs when the class
# neither self-declares ``higher_is_better`` nor is named in the central SSOT
# map — so an invalid direction was *silently accepted* in the other two
# situations (CLAUDE.md #15: a wired knob must validate-or-raise on every path).


def _purge_metric(name: str) -> None:
    """Remove ``name`` (canonical) from every per-name registry store."""
    for store in (
        MetricsRegistry._metrics,
        MetricsRegistry._needs,
        MetricsRegistry._requires_context,
        MetricsRegistry._requires_reference,
    ):
        store.pop(name, None)


@pytest.mark.unit
class TestDirectionValidationUnconditional:
    """``direction`` is validated on every registration path, not just one."""

    def test_invalid_direction_plain_class_raises(self) -> None:
        """(a) Plain class, no self-declared direction, name not in the SSOT
        map: invalid ``direction`` must raise (this was the only path that
        validated even before the fix)."""

        class _Plain:
            def __call__(self, p, t, **k):
                return 0.0

        try:
            with pytest.raises(ValueError, match="direction must be 'higher' or 'lower'"):
                register_metric("__dir_plain__", direction="sideways")(_Plain)
        finally:
            _purge_metric("__dir_plain__")

    def test_invalid_direction_self_declared_class_raises(self) -> None:
        """(b) Class that self-declares ``higher_is_better``: BEFORE the fix the
        ``elif`` short-circuited and the invalid direction was silently
        accepted. It must now raise."""

        class _SelfDecl:
            higher_is_better = True

            def __call__(self, p, t, **k):
                return 0.0

        try:
            with pytest.raises(ValueError, match="direction must be 'higher' or 'lower'"):
                register_metric("__dir_self__", direction="sideways")(_SelfDecl)
            # Self-declared direction is untouched by the (rejected) registration.
            assert _SelfDecl.higher_is_better is True
        finally:
            _purge_metric("__dir_self__")

    def test_invalid_direction_mapped_name_raises(self, monkeypatch) -> None:
        """(c) Name present in ``METRIC_HIGHER_IS_BETTER``: BEFORE the fix the
        ``if name in METRIC_HIGHER_IS_BETTER`` branch ran first and the invalid
        direction was silently accepted. It must now raise.

        Uses an existing-key shape — a unique throwaway key inserted into the
        central map for the test — to exercise the "mapped name" branch without
        colliding with a real production registration.
        """
        from spectramr.core.metrics import metric_directions

        # The key is genuinely *present in* METRIC_HIGHER_IS_BETTER for the
        # duration of the test (monkeypatch restores the map afterward), so the
        # registration follows the same mapped-name code path that "psnr",
        # "ssim", etc. take.
        monkeypatch.setitem(metric_directions.METRIC_HIGHER_IS_BETTER, "__dir_mapped__", True)

        class _Mapped:
            def __call__(self, p, t, **k):
                return 0.0

        try:
            with pytest.raises(ValueError, match="direction must be 'higher' or 'lower'"):
                register_metric("__dir_mapped__", direction="sideways")(_Mapped)
            # The injection from the map must NOT have happened on the rejected
            # class — validation precedes the ``METRIC_HIGHER_IS_BETTER``
            # injection in the fixed code, so no ``higher_is_better`` is set.
            assert not hasattr(_Mapped, "higher_is_better")
        finally:
            _purge_metric("__dir_mapped__")

    def test_valid_direction_higher_is_accepted(self) -> None:
        """``direction='higher'`` is accepted and, for an unmapped /
        non-self-declaring class, maps to ``higher_is_better is True``."""

        class _GoodHigher:
            def __call__(self, p, t, **k):
                return 0.0

        try:
            register_metric("__dir_ok__", direction="higher")(_GoodHigher)
            assert "__dir_ok__" in MetricsRegistry._metrics
            assert _GoodHigher.higher_is_better is True
        finally:
            _purge_metric("__dir_ok__")


class TestWorkflowTagIntegrity:
    """The workflow-tag table backs the maturity ledger's anti-facade check.

    Its only consumer is `metrics_tagged(regime)`, so a stale or malformed entry
    here is a facade *inside the guard* — the ledger would report coverage that
    does not exist, which is worse than having no ledger.
    """

    def test_clear_also_clears_workflow_tags(self) -> None:
        """clear() omitted _workflow_tags, so tags outlived their metrics.

        After a test called clear(), metrics_tagged(regime) still returned True
        for metrics that no longer existed.
        """
        from spectramr.config.schemas.enums import Regime
        from spectramr.core.metrics.registry import MetricsRegistry

        # snapshot()/restore() rather than three hand-picked dicts: this test
        # used to save _metrics, _aliases and _workflow_tags and NOT _needs,
        # _requires_context or _requires_reference, so the clear() below left
        # those three empty for the rest of the process. Every later
        # MetricsRegistry.needs() read then returned () -- a wrong value, not an
        # error -- and the release suite could not be run under xdist because the
        # failure set depended on which files shared a worker (#1846).
        saved = MetricsRegistry.snapshot()
        try:

            @MetricsRegistry.register("_tag_integrity_probe", workflows=frozenset({Regime.FLOW}))
            class _Probe:
                higher_is_better = True

                def __call__(self, prediction, target, **kwargs):
                    return 0.0

            assert "_tag_integrity_probe" in MetricsRegistry._workflow_tags
            MetricsRegistry.clear()
            assert MetricsRegistry._workflow_tags == {}, (
                "clear() left workflow tags behind — the ledger would keep "
                "reporting coverage for metrics that no longer exist."
            )
        finally:
            MetricsRegistry.restore(saved)

    def test_a_bare_string_workflows_tag_is_rejected(self) -> None:
        """`workflows="mri_flow"` would SUBSTRING-match in the ledger.

        The ledger does `regime in workflows`. Against a frozenset that is
        membership; against a string it is a substring search, so a typo'd tag
        could silently satisfy — or silently miss — a maturity claim. `direction`
        next door already validated-or-raised; this did not (CLAUDE.md #15).
        """
        import pytest

        from spectramr.core.metrics.registry import MetricsRegistry

        with pytest.raises(TypeError, match="substring-match"):

            @MetricsRegistry.register("_bad_string_tag", workflows="mri_flow")
            class _Bad:
                higher_is_better = True

                def __call__(self, prediction, target, **kwargs):
                    return 0.0

    def test_a_non_regime_member_in_workflows_is_rejected(self) -> None:
        import pytest

        from spectramr.core.metrics.registry import MetricsRegistry

        with pytest.raises(TypeError, match="non-Regime"):

            @MetricsRegistry.register("_bad_member_tag", workflows=frozenset({"flow"}))
            class _Bad:
                higher_is_better = True

                def __call__(self, prediction, target, **kwargs):
                    return 0.0


class TestWorkflowsAccessor:
    """The public read side of the ``workflows=`` tag.

    The tag existed only in the private ``_workflow_tags`` dict, so no consumer
    could ask "does this metric apply to the regime I am measuring?" — and
    sim2rank did not: it scored ``cbf_rmse`` (perfusion) and ``velocity_rmse``
    (4D-flow) on structural brain magnitude images, where both reduce to plain
    RMSE, bit-identical to the registered ``rmse``.
    """

    def test_declared_workflows_are_returned(self) -> None:
        from spectramr.config.schemas.enums import Regime
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.workflows("cbf_rmse") == frozenset({Regime.PERFUSION})

    def test_agnostic_metric_returns_none(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.workflows("psnr") is None

    def test_alias_resolves_to_the_canonical_tag(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.workflows("CBFRMSE") == MetricsRegistry.workflows("cbf_rmse")

    def test_unknown_name_is_agnostic(self) -> None:
        """Unknown must read as "applies everywhere" — the loud direction.

        Returning a narrow set for an unrecognised key would let a typo
        silently delete a metric from a run.
        """
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.workflows("_no_such_metric") is None

    def test_applies_to_regime_matches_and_rejects(self) -> None:
        from spectramr.config.schemas.enums import Regime
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.applies_to_regime("cbf_rmse", Regime.PERFUSION)
        assert not MetricsRegistry.applies_to_regime("cbf_rmse", Regime.STRUCTURAL)

    def test_applies_to_regime_with_no_regime_is_a_no_op(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        assert MetricsRegistry.applies_to_regime("cbf_rmse", None)

    def test_agnostic_metric_applies_to_every_regime(self) -> None:
        from spectramr.config.schemas.enums import Regime
        from spectramr.core.metrics.registry import MetricsRegistry

        assert all(MetricsRegistry.applies_to_regime("psnr", r) for r in Regime)


# ─────────────────────────────────────────────────────────────────────────────
# The kwarg filter must trust __init__ OWNERSHIP, not the advertised signature
# ─────────────────────────────────────────────────────────────────────────────


class TestInheritedInitRejectsKwargs:
    """``nn.Module.__init__`` advertises ``(*args, **kwargs)`` and accepts neither.

    It raises "unexpected keyword argument" for any kwarg unless the subclass sets
    ``call_super_init`` (default False), purely for backward compatibility. So a
    metric that subclasses ``nn.Module`` WITHOUT defining its own ``__init__``
    presents a permissive signature and then refuses everything.

    ``MetricsRegistry.get`` believed the signature and forwarded ``device`` -- the
    one kwarg the validation computer always sends. Eight registered metrics died
    there, including the entire flow/perfusion battery, so an arm that named them
    crashed in validation instead of being graded (#343 / #340).
    """

    def test_a_module_subclass_without_its_own_init_accepts_device(self):
        """The reproducer, as a type built here rather than a name from the tree."""

        @register_metric("_test_inherited_init_metric")
        class _Inherited(torch.nn.Module):
            higher_is_better = True

            def forward(self, pred, target, **kwargs):
                return torch.tensor(0.0)

        try:
            # Pre-fix this raised TypeError: got an unexpected keyword argument 'device'
            assert MetricsRegistry.get("_test_inherited_init_metric", device="cpu")
        finally:
            MetricsRegistry._metrics.pop("_test_inherited_init_metric", None)

    def test_the_advertised_signature_really_is_permissive(self):
        """Anti-vacuity: if nn.Module stopped lying, the test above proves nothing."""
        import inspect

        params = inspect.signature(torch.nn.Module).parameters.values()
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params), (
            "nn.Module no longer advertises **kwargs -- the ownership check in "
            "MetricsRegistry.get may now be unnecessary."
        )
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            torch.nn.Module(device="cpu")

    def test_device_is_still_honoured_when_the_ctor_declares_it(self):
        """The fix must not become 'drop every kwarg'."""

        @register_metric("_test_declares_device_metric")
        class _Declares(torch.nn.Module):
            higher_is_better = True

            def __init__(self, device=None):
                super().__init__()
                self.device_arg = device

            def forward(self, pred, target, **kwargs):
                return torch.tensor(0.0)

        try:
            inst = MetricsRegistry.get("_test_declares_device_metric", device="cpu")
            assert inst.device_arg == "cpu"
        finally:
            MetricsRegistry._metrics.pop("_test_declares_device_metric", None)


def _unlisted_tables(registry: type) -> set[str]:
    """Registration tables ``registry`` declares that ``_TABLES`` does not name.

    A free function taking the class, rather than a method, so the detector can
    be pointed at a PLANTED violation as well as at the real registry. A check
    that can only ever run against the real object is a check nobody has watched
    fail, and the whole reason this file grew these tests is that an enumeration
    which had never been watched fail was wrong (non-negotiable 15).
    """
    declared = set(getattr(registry, "_TABLES", ()))
    live = {
        name
        for name, value in vars(registry).items()
        if isinstance(value, dict) and name.startswith("_") and not name.startswith("__")
    }
    return live - declared


class TestRegistrationTableOwnership:
    """``_TABLES`` is the single owner of "what a registration table is".

    ``clear()``, ``snapshot()`` and ``restore()`` were three independent
    enumerations of the same set and one of them was wrong: the workflow-tag
    isolation test saved three of the six tables ``clear()`` wipes, leaving
    ``_needs``, ``_requires_context`` and ``_requires_reference`` empty for the
    remainder of the process.

    Nothing raised. ``needs()`` returns ``()`` for an unregistered name, so the
    reads that followed came back *wrong* rather than *loud*, and which tests
    saw it depended on ordering -- serial runs mostly escaped because
    ``test_registry.py`` sorts after ``test_nr_*.py``, while under
    ``pytest-xdist --dist loadfile`` it depended on which files shared a worker.
    That is why the release lane was pinned to serial (#1846).
    """

    def test_no_declared_table_is_missing_from_tables(self) -> None:
        """The ratchet: declare a seventh table and this goes red."""
        from spectramr.core.metrics.registry import MetricsRegistry

        assert _unlisted_tables(MetricsRegistry) == set(), (
            "a registration table is not listed in _TABLES, so clear()/snapshot()"
            "/restore() silently skip it -- the exact shape of #1846"
        )

    def test_the_detector_sees_a_planted_unlisted_table(self) -> None:
        """Plant the violation, or the ratchet above is untested."""

        class _Drifted:
            _TABLES = ("_listed",)
            _listed: ClassVar[dict] = {}
            _forgotten: ClassVar[dict] = {}

        assert _unlisted_tables(_Drifted) == {"_forgotten"}

    def test_the_detector_ignores_non_table_class_attributes(self) -> None:
        """``_instance``/``_initialized`` are not tables; a bool is not a dict."""

        class _WithScalars:
            _TABLES = ("_listed",)
            _listed: ClassVar[dict] = {}
            _instance = None
            _initialized = False

        assert _unlisted_tables(_WithScalars) == set()

    def test_clear_empties_every_listed_table(self) -> None:
        from spectramr.core.metrics.registry import MetricsRegistry

        saved = MetricsRegistry.snapshot()
        try:
            for table in MetricsRegistry._TABLES:
                getattr(MetricsRegistry, table)["_probe"] = ()
            MetricsRegistry.clear()
            for table in MetricsRegistry._TABLES:
                assert getattr(MetricsRegistry, table) == {}, table
        finally:
            MetricsRegistry.restore(saved)

    def test_snapshot_restore_round_trips_every_table(self) -> None:
        """The test that would have caught #1846: it asserts over ALL tables.

        The original saved the three tables the test it lived in was *about*.
        Iterating ``_TABLES`` means a table cannot be forgotten by omission.
        """
        from spectramr.core.metrics.registry import MetricsRegistry

        saved = MetricsRegistry.snapshot()
        try:
            for table in MetricsRegistry._TABLES:
                getattr(MetricsRegistry, table)["_probe"] = ()
            mid = MetricsRegistry.snapshot()
            MetricsRegistry.clear()
            MetricsRegistry.restore(mid)
            for table in MetricsRegistry._TABLES:
                assert "_probe" in getattr(MetricsRegistry, table), table
        finally:
            MetricsRegistry.restore(saved)

    def test_restore_raises_when_the_snapshot_predates_a_table(self) -> None:
        """Absent is a state to report: a partial snapshot must not half-restore."""
        import pytest

        from spectramr.core.metrics.registry import MetricsRegistry

        incomplete = {k: v for k, v in MetricsRegistry.snapshot().items() if k != "_needs"}
        with pytest.raises(KeyError, match="_needs"):
            MetricsRegistry.restore(incomplete)

    def test_restore_replaces_rather_than_merges(self) -> None:
        """A registration made after the snapshot must not outlive the restore."""
        from spectramr.core.metrics.registry import MetricsRegistry

        saved = MetricsRegistry.snapshot()
        try:
            MetricsRegistry._needs["_probe_after_snapshot"] = ("mask",)
            MetricsRegistry.restore(saved)
            assert "_probe_after_snapshot" not in MetricsRegistry._needs
        finally:
            MetricsRegistry.restore(saved)

    def test_a_declared_need_survives_a_clear_and_restore(self) -> None:
        """The regression, through the real decorator rather than a raw dict.

        Registering with ``needs=`` and reading it back via the public
        ``needs()`` is the exact call the poisoned runs got ``()`` from.
        """
        from spectramr.core.metrics.registry import MetricsRegistry

        saved = MetricsRegistry.snapshot()
        try:

            @MetricsRegistry.register("_needs_probe", needs=("sibling_contrasts",))
            class _Probe:
                higher_is_better = True

                def __call__(self, prediction, target, **kwargs):
                    return 0.0

            assert MetricsRegistry.needs("_needs_probe") == ("sibling_contrasts",)
            mid = MetricsRegistry.snapshot()
            MetricsRegistry.clear()
            assert MetricsRegistry.needs("_needs_probe") == ()
            MetricsRegistry.restore(mid)
            assert MetricsRegistry.needs("_needs_probe") == ("sibling_contrasts",)
        finally:
            MetricsRegistry.restore(saved)
