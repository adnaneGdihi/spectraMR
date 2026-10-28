"""Unit tests for the env-var NAME registry.

``env_names`` was split out of ``core/env.py`` in the Wave 0 exit-criterion work
(#1400). The split creates a blind spot that these tests exist to close.

``test_env.py::test_all_constants_are_exported`` scans ``vars(env)``. After the
split, a constant declared in this module but omitted from *its* ``__all__``
never reaches ``env`` at all -- so that test cannot see it and passes vacuously.
The same scan has to run against the registry itself, or the split would have
quietly disabled the guard that caught ``SPECTRAMR_GPU_MEMORY_FRACTION``.
"""

from __future__ import annotations

from spectramr.core import env, env_names


def _declared(module: object) -> set[str]:
    """Every ``NAME = "NAME"`` self-naming constant in a module."""
    return {
        name
        for name, value in vars(module).items()
        if isinstance(value, str) and name == value and not name.startswith("_")
    }


class TestRegistryExportCoverage:
    def test_all_constants_are_exported(self) -> None:
        """The guard test_env.py structurally cannot run after the split."""
        unexported = sorted(_declared(env_names) - set(env_names.__all__))
        assert not unexported, (
            f"{unexported} are declared in core/env_names.py but missing from its "
            "__all__. env.py imports through that list, so an unexported constant "
            "never reaches core.env -- which also makes "
            "test_env.py::test_all_constants_are_exported blind to it."
        )

    def test_the_check_can_fire(self) -> None:
        """Anti-vacuity: the scan must actually find the constants."""
        assert len(_declared(env_names)) >= 40

    def test_every_export_is_a_declared_constant(self) -> None:
        """The reverse direction: no __all__ entry naming something absent."""
        phantom = sorted(set(env_names.__all__) - _declared(env_names))
        assert not phantom, f"__all__ names constants that do not exist: {phantom}"


class TestFacadeIntegrity:
    def test_env_reexports_every_registry_constant(self) -> None:
        missing = sorted(set(env_names.__all__) - set(vars(env)))
        assert not missing, f"core.env does not re-export: {missing}"

    def test_the_two_modules_agree_by_identity_not_by_copy(self) -> None:
        """One owner (NN17): env must hold the registry's objects, not copies."""
        for name in env_names.__all__:
            assert getattr(env, name) is getattr(env_names, name)

    def test_names_is_exactly_the_registry_in_declaration_order(self) -> None:
        """env.names() is spliced from the registry, so order must survive."""
        assert env.names() == tuple(env_names.__all__)

    def test_registry_holds_no_accessors(self) -> None:
        """Names only -- a reader here would re-create the cycle the split broke."""
        assert all(n.isupper() for n in env_names.__all__)


class TestDispatchSnapshotNames:
    def test_the_drift_opt_in_and_its_handoff_are_registered(self) -> None:
        """#2662: the dispatcher reads one and writes the other; both are advertised."""
        for name in ("SPECTRAMR_DISPATCH_ALLOW_DRIFT", "SPECTRAMR_DISPATCH_SNAPSHOT_RECORD"):
            assert name in env_names.__all__
            assert name in env.names()
