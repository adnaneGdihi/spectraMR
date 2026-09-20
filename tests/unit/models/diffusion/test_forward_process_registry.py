"""Contract tests for the forward-process registry."""

from __future__ import annotations

import pytest

from spectramr.models.diffusion import forward_process_registry as reg


def test_both_acquisition_models_are_reachable_by_name() -> None:
    """An arm selects its acquisition through these names."""
    names = reg.list_forward_processes()
    assert "cartesian_mask" in names
    assert "golden_angle_spokes" in names


def test_a_partially_filled_registry_is_repaired_not_short_circuited() -> None:
    """Planted violation for the partial-registry shape (non-negotiable 15).

    Guarding population on ``if FORWARD_PROCESS_REGISTRY:`` returns as soon as
    ONE entry is present, leaving the rest missing and the registry reading
    plausibly-but-partly full -- the failure ``.claude/rules/registries.md``
    records for the model registry. Leave exactly one entry behind and assert
    the next call restores the other.
    """
    saved = dict(reg.FORWARD_PROCESS_REGISTRY)
    try:
        reg.FORWARD_PROCESS_REGISTRY.clear()
        reg.FORWARD_PROCESS_REGISTRY["cartesian_mask"] = saved["cartesian_mask"]
        restored = reg.list_forward_processes()
        assert set(restored) >= {"cartesian_mask", "golden_angle_spokes"}, (
            f"one population call yielded only {restored}; population is "
            f"short-circuiting on a non-empty registry"
        )
    finally:
        reg.FORWARD_PROCESS_REGISTRY.clear()
        reg.FORWARD_PROCESS_REGISTRY.update(saved)


def test_an_empty_registry_is_rebuilt_from_this_module_alone() -> None:
    """Recovery must not depend on re-running an import side effect.

    Python caches modules, so a decorator-populated registry that loses every
    entry can never be refilled. Naming the members in the registry makes this
    recoverable, and this is the test that holds it that way.
    """
    saved = dict(reg.FORWARD_PROCESS_REGISTRY)
    try:
        reg.FORWARD_PROCESS_REGISTRY.clear()
        assert set(reg.list_forward_processes()) == set(saved)
    finally:
        reg.FORWARD_PROCESS_REGISTRY.clear()
        reg.FORWARD_PROCESS_REGISTRY.update(saved)


def test_an_unknown_name_raises_and_names_the_alternatives() -> None:
    """No silent fallback to Cartesian acquisition (non-negotiable 3)."""
    with pytest.raises(ValueError, match="Unknown kspace_process_type"):
        reg.build_forward_process("no_such_acquisition")


def test_every_declared_member_resolves() -> None:
    """A member naming a class that moved would raise here, not at build time."""
    for name in reg._MEMBERS:
        assert name in reg.list_forward_processes()
        assert isinstance(reg.FORWARD_PROCESS_REGISTRY[name], type)


def test_build_returns_the_registered_class() -> None:
    """The name resolves to the acquisition it advertises."""
    process = reg.build_forward_process(
        "golden_angle_spokes",
        num_spokes=128,
        samples_per_spoke=64,
        im_size=(32, 32),
        num_timesteps=4,
        max_acceleration=16.0,
        base_acceleration=1.0,
        schedule_kwargs={"acceleration_range": [1.0, 2.0, 4.0, 16.0]},
    )
    assert type(process).__name__ == "NonCartesianSpokeProcess"
