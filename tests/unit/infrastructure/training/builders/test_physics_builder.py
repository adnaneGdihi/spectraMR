"""`PhysicsBuilder` builds the physics operators the training path resolves.

Paired with ``src/spectramr/infrastructure/training/builders/physics_builder.py``.

``build_mask_generator`` was removed in #2056. It built a third
``KSpaceMaskGenerator`` into ``env.physics["mask_generator"]`` on every run --
one nothing read, from the same ``undersampling:`` block the model and the
strategy already resolve. The tests it carried pinned the shared allowlist
(``mask_seed`` -> ``seed``, unread defaults not forwarded), and that shape now
lives at its one owner, ``accelerator_kwargs_from_config``, in
``tests/unit/models/diffusion/test_kspace_process.py``.
"""

from __future__ import annotations

from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder
from tests.utils.minimal_settings import minimal_settings_for


def test_the_builder_no_longer_owns_a_mask_generator() -> None:
    """The deletion is the assertion (#2056, non-negotiable 17).

    Re-adding the step would restore a generator with no readers and a third
    accelerator per run, and nothing else in the suite would notice: the copy
    was invisible precisely because it agreed with the other two.
    """
    assert not hasattr(PhysicsBuilder, "build_mask_generator")

    components = (
        PhysicsBuilder(minimal_settings_for("gan"), "cpu")
        .build_fft_transformer()
        .build_data_consistency()
        .build_coil_sensitivity()
        .validate()
        .build()
    )
    assert "mask_generator" not in components


def test_build_coil_sensitivity_is_an_honest_no_op():
    """The step never worked, and could not have been noticed from outside.

    It imported ``ESPIRiTSensitivity`` from ``physics/coil_sensitivity.py``, which
    exports FUNCTIONS and has never defined that class. The import never actually
    raised, though: the two guards above it read
    ``config.physics.parallel_imaging.enabled``, and ``parallel_imaging`` is not a
    field on ANY config schema while ``settings.physics`` defaults to ``None`` — so
    both returned early on every call and the body was unreachable. A dead knob
    kept a dead import invisible.

    Restoring the import would fix nothing: ``_components["coil_sens"]`` was the
    only reference to that key tree-wide. ``estimate_smaps`` (called live from
    ``data_pipeline_director``) is the elected owner (non-negotiable 17).
    """
    import inspect

    from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder

    source = inspect.getsource(PhysicsBuilder.build_coil_sensitivity)
    body = source.split('"""')[2]
    assert "ESPIRiTSensitivity" not in body, (
        "the non-existent class is referenced outside the explanatory docstring"
    )
    assert "coil_sens" not in body, "a component nothing reads is being populated again"


def test_build_coil_sensitivity_still_chains():
    """It stays in the fluent chain (``director.py`` calls it) — a step that
    silently vanishes is harder to notice than one that says why it does nothing."""
    import inspect

    from spectramr.infrastructure.training.builders.physics_builder import PhysicsBuilder

    sig = inspect.signature(PhysicsBuilder.build_coil_sensitivity)
    assert "PhysicsBuilder" in str(sig.return_annotation)
