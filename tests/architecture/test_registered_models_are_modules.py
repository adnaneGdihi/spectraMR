"""Fitness function: every registered model is an ``nn.Module``, and none hides ``forward``.

The ``architecture`` marker's registered description ("AST-level fitness function")
does not describe this file: it reads class objects out of the populated registry, not
source text. Several siblings are not AST scans either, so that alone is unremarkable
-- what matters is that an AST scan *cannot* answer #801's question. A scan for
``self.x = nn.Foo(...)`` reads **zero** on all seven classes this change repairs, and
not for a spelling reason: none of them writes ``torch.nn.`` either. They reach their
submodules three ways, each defeating source text differently -- a factory whose return
type is not written down (``self.model = get_generator(...)``), a sibling class whose
own bases would have to be resolved transitively (``self.encoder =
FlexibleEncoder(...)``), and, in ``low_field_augmented``, ``__init__`` parameters, where
the class body constructs nothing and the types are the caller's to decide. That last
shape is why widening the scan to catch factories does not rescue it: its only
assignment from a call is ``logging.getLogger(...)``, while construction finds three
modules held. Asking the registry for the class object sees all three.

``populate_model_registry()`` is not a cost to justify -- it is what makes the census
complete. Every pytest run here already imports the model package through the root
conftest's ``pytest_plugins = ["tests.fixtures.builders"]``, and that import on its own
leaves ``MODEL_REGISTRY`` at **269** of 588: a partial count that reads plausible.
**Five of the nine classes #801 tables sit outside those 269, and all five are classes
this change repairs** -- so a census that skipped the fixture would scan 269 entries,
find five fewer offenders and pass. Green on the easy shape, which is the shape a
fitness function exists to not be. After that plugin import the call costs **0.05 s**
(5.4 s from genuinely cold), and ``--durations`` attributes 0.18 s to this fixture.

Price this file from the directory lane -- 85.3 s to 86.2 s wall, one run each way,
inside the noise of an 85 s lane -- and from nothing else. pytest's printed duration
excludes everything before ``pytest_sessionstart``, the 5.2 s plugin import included,
so ``--collect-only`` on this file prints ``0.09s`` against a 9.0 s wall clock. A solo
run is 8.9 s, but that is the floor every file pays: an AST-only sibling in this same
directory, run alone the same way, is 11.6 s.

Two invariants, one owner each:

(a) A registered entry's class is an ``nn.Module``. A non-Module holds its submodules
    outside any ``state_dict``: absent from checkpoints, unmoved by ``.to(device)``,
    invisible to the optimizer -- and nothing raises.
(b) A registered ``nn.Module`` does not override ``__call__``. ``nn.Module.__call__``
    *is* ``_call_impl``, so an override makes ``forward`` unreachable and every hook
    that hangs off it -- profiling, AMP, ``register_forward_hook`` -- silently dead.

(b) reads **0 at the base of the change that added it**, and that is the point rather
than a defect in it: at base the three classes that overrode ``__call__`` were *also*
non-Modules, so (a) claimed them first and (b) never saw them. It is a ratchet against
the regression shape #801's own prescription produces -- add the ``nn.Module`` base,
keep the ``__call__`` -- not a census of today's tree. The plants below are what prove
it is live; the base run is not evidence for it either way.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

pytestmark = pytest.mark.architecture

# Group 3 of #801. `ColdDiffusion` and `LaplaceDiffusion` subclass `Diffusion`, define
# no `forward`, and own no submodule between them: a degradation *process* may
# legitimately not be a module. Whether a process belongs in a MODEL registry at all is
# the open question, and it is an owner's to answer -- recorded here so it stays
# visible instead of being silently tolerated.
KNOWN_NON_MODULES = frozenset({"cold_diffusion_process", "laplace_diffusion"})


def census(registry: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return (keys whose class is not an ``nn.Module``, keys overriding ``__call__``).

    Reads ``entry["class"]``. A registry *value* is a metadata dict, so treating the
    value itself as a class makes this a count over the empty set -- which reads
    exactly like a clean tree.
    """
    not_modules: set[str] = set()
    call_overrides: set[str] = set()
    for key, entry in registry.items():
        cls = entry["class"]
        if not (isinstance(cls, type) and issubclass(cls, torch.nn.Module)):
            not_modules.add(key)
            continue
        # The RESOLVED attribute, not `vars(cls)`: an override inherited from an
        # intermediate base is invisible to `vars` and is exactly as much of a facade.
        if cls.__call__ is not torch.nn.Module.__call__:
            call_overrides.add(key)
    return not_modules, call_overrides


def _entry(cls: Any) -> dict[str, Any]:
    """A registry entry shaped like the real ones.

    Only ``"class"`` is read, but the other documented fields are present so a plant
    cannot pass merely because its entry was degenerate.
    """
    return {"class": cls, "mode": "reconstruction", "role": "generator", "capabilities": None}


@pytest.fixture(scope="module")
def registry() -> dict[str, Any]:
    """The populated live registry.

    Module-scoped and idempotent, so every test here shares one call. That call is a
    PRECONDITION rather than a convenience -- the module docstring measures why, and
    `test_the_census_reads_the_whole_registry` below is what goes red if it is ever
    dropped. Never mutate what this returns -- clearing
    MODEL_REGISTRY is not recoverable within a process, and `force=True` does not undo
    it. Every plant below builds its own copy.
    """
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    assert MODEL_REGISTRY, "populate_model_registry() left MODEL_REGISTRY empty"
    return MODEL_REGISTRY


def test_the_census_reads_the_whole_registry(registry: dict[str, Any]) -> None:
    """The fixture's ``populate_model_registry()`` call is a precondition; this guards it.

    Deleting that call is invisible to every other test in this file. Importing the model
    package -- which every pytest run here already does, through the root conftest's
    ``pytest_plugins`` -- leaves MODEL_REGISTRY at 269 of 588, and *both* keys in
    KNOWN_NON_MODULES are inside those 269, so the equality above still holds while the
    census silently scans under half the tree. The five #801 rows that sit outside the
    269 are all classes this change repairs, which is the exact shape NN15 is about: a
    detector green on the easy population.

    Pinned to a key the import path does not reach rather than to a total, because the
    total moves whenever a model is registered and a ratchet on it gets edited to green.
    A directory lane where some earlier file already populated the registry passes this
    honestly -- the census really is complete there.
    """
    assert "configurable_vae" in registry, (
        "The census ran over a PARTIALLY populated MODEL_REGISTRY: 'configurable_vae' is "
        "one of the 319 keys that importing the model package does not reach, so its "
        "absence means only that import happened. Restore populate_model_registry() in "
        "the `registry` fixture -- without it this file scans 269 of 588 entries, misses "
        "five of #801's nine rows, and passes."
    )


def test_every_registered_model_is_an_nn_module(registry: dict[str, Any]) -> None:
    not_modules, _ = census(registry)
    unexpected = not_modules - KNOWN_NON_MODULES
    retired = KNOWN_NON_MODULES - not_modules

    # Equality, not containment, and the message names WHICH direction broke: a
    # containment check goes green forever once group 3 is fixed and the record here
    # should have shrunk with it (NN20 -- the count moves down only).
    problems = []
    if unexpected:
        problems.append(
            f"NEW non-nn.Module registration(s) {sorted(unexpected)}: their submodules "
            "never register, so they are absent from state_dict, unmoved by .to(device) "
            "and invisible to the optimizer. Give each an nn.Module base AND call "
            "nn.Module.__init__ before the first submodule assignment -- adding the base "
            "alone raises 'cannot assign module before Module.__init__() call' (#801)."
        )
    if retired:
        problems.append(
            f"{sorted(retired)} is an nn.Module now: drop it from KNOWN_NON_MODULES in "
            "this file. The record is a ratchet and moves down only (NN20)."
        )
    assert not problems, " ".join(problems)


def test_no_registered_module_overrides_call(registry: dict[str, Any]) -> None:
    _, overrides = census(registry)
    assert overrides == set(), (
        f"{sorted(overrides)} override __call__. nn.Module.__call__ IS _call_impl, so "
        "the override makes forward unreachable and silently kills every hook that hangs "
        "off it -- forward/backward hooks, profiling, AMP. If the body is byte-identical "
        "to forward, delete it; if it differs, that difference belongs in forward."
    )


def test_the_census_sees_a_planted_non_module(registry: dict[str, Any]) -> None:
    class NotAModule:
        """Registered like a model, inherits nothing from torch."""

    planted = {**registry, "_planted_not_a_module": _entry(NotAModule)}
    not_modules, _ = census(planted)

    assert "_planted_not_a_module" in not_modules
    assert "_planted_not_a_module" not in registry, "the plant leaked into the live registry"


def test_the_census_sees_a_planted_call_override(registry: dict[str, Any]) -> None:
    class FacadeModule(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return self.forward(x)

    planted = {**registry, "_planted_facade": _entry(FacadeModule)}
    _, overrides = census(planted)

    assert "_planted_facade" in overrides
    assert "_planted_facade" not in registry, "the plant leaked into the live registry"


def test_the_census_sees_an_override_inherited_from_an_intermediate_base(
    registry: dict[str, Any],
) -> None:
    """The second shape of (b), and the one a `vars(cls)` implementation misses.

    A class that inherits its facade is exactly as unreachable as one that declares it,
    so this is a distinct shape rather than a restatement (NN15 -- one plant per shape).
    """

    class FacadeBase(torch.nn.Module):
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class Leaf(FacadeBase):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    assert "__call__" not in vars(Leaf), "premise: the override is not on Leaf itself"

    planted = {**registry, "_planted_inherited_facade": _entry(Leaf)}
    _, overrides = census(planted)

    assert "_planted_inherited_facade" in overrides


def test_a_non_class_registry_value_is_reported_rather_than_raising(
    registry: dict[str, Any],
) -> None:
    """An *instance* under "class" is a non-Module, not a crash.

    Without the `isinstance(cls, type)` guard `issubclass` raises TypeError here, and a
    detector that errors reports nothing about the other 587 entries.
    """
    planted = {**registry, "_planted_instance": _entry(torch.nn.Linear(2, 2))}
    not_modules, overrides = census(planted)

    assert "_planted_instance" in not_modules
    assert "_planted_instance" not in overrides


def test_a_well_formed_module_is_flagged_by_neither_census(registry: dict[str, Any]) -> None:
    """Pins the detector as narrow.

    A census that flagged everything would pass all four plants above and be useless;
    this is what makes those four mean something.
    """

    class WellFormed(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = torch.nn.Linear(2, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

    planted = {**registry, "_planted_ok": _entry(WellFormed)}
    not_modules, overrides = census(planted)

    assert "_planted_ok" not in not_modules
    assert "_planted_ok" not in overrides
