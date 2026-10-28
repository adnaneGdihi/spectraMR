"""Loss ownership declared on the strategy class is pinned against the source (mrixfields review 2026-09-03).

A flag can lie: ``folds_image_losses`` says whether the other declared image
losses reach the objective, and the source is the only witness of that. Every
cohort strategy that overrides ``_compute_losses_impl`` either calls the parent
(``super()._compute_losses_impl``) or the fold (``_apply_builder_image_losses``)
-- then it folds -- or neither -- then it must say False. The same set of
strategies computes its inline L1 weight from the loss-weight table, one owner.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from types import SimpleNamespace

import pytest

from spectramr.infrastructure.training.strategies.loss_folding import (
    declared_folds_image_losses,
    declared_inline_losses,
)
from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory

COHORT_MODES = (
    "doob_bridge",
    "confluence",
    "scattering_besov",
    "steerable_synthesis",
    "brenier_synthesis",
    "cross_field_translation",
    "bloch_synth",
    "ulf_redegrad_tta",
    "ulf_map",
    "ulf_dps",
    "recoverability_vib",
    "field_cold_diffusion",
    "cartoon_texture_safe",
    "monotone_field",
    "generative_refiner",
    "field_conditioned_inr",
    "heteroscedastic_ulf",
    "field_fno",
    "bloch_field",
    "field_wiener",
    "koopman_field",
    "field_flow",
    "field_bridge",
    "fisher_rao_geodesic",
    "field_guided_diffusion",
    "lora_modulation",
    "mccann_field_path",
    "field_cocycle",
)
L1_TABLE_READERS = (
    "scattering_besov",
    "steerable_synthesis",
    "brenier_synthesis",
    "ulf_redegrad_tta",
    "cartoon_texture_safe",
    "monotone_field",
    "field_conditioned_inr",
    "field_fno",
    "bloch_field",
    "field_wiener",
    "koopman_field",
    "fisher_rao_geodesic",
    "lora_modulation",
    "mccann_field_path",
    "recoverability_vib",
)

#: Every way a strategy routes the builder's ``losses.*_losses`` modules to its
#: objective. ONE owner (non-negotiable 17): both the mrixfields pin and the
#: computer-route pin below read this tuple, so teaching the vocabulary a new
#: route is a one-line change that widens both.
#:
#: Routes 4 and 5 were added 2026-09-07 (issue #1918). The pin previously knew
#: only 1-3 and so read the two diffusion strategies -- which reach the objective
#: through a unified loss computer and through a direct ``env.losses`` fold -- as
#: strategies that fold nothing. A truthful ``folds_image_losses = True`` on
#: either would have FAILED this pin, which is why the flags stayed undeclared
#: and 58 kspace_filling arms sat UNVERIFIED.
#: Route 4 alone, named: ``_reaches`` treats it differently from the other four
#: (it is a hand-off, not a fold), and referencing it by name rather than by
#: ``ROUTE_MARKERS[3]`` means reordering the tuple cannot silently re-aim the
#: drop-awareness at the wrong route.
_HANDOFF_MARKER = "losses_dict="

ROUTE_MARKERS = (
    "super()._compute_losses_impl",  # 1. delegate to the parent's builder path
    "_apply_builder_image_losses(",  # 2. the strategy's own fold call
    "fold_builder_image_losses(",  # 3. the shared fold helper
    _HANDOFF_MARKER,  # 4. forward env.losses to a unified loss computer
    "env_losses.items()",  # 5. iterate env.losses and accumulate weight * loss
)

#: Loss computers that ACCEPT ``losses_dict`` and never read it. Route 4 is a
#: hand-off, so it only reaches the objective if the receiver folds; these two
#: drop it on the floor, which makes the marker alone a false clear.
#: ONE owner (non-negotiable 17): ``_reaches`` consumes it and
#: ``test_every_dropping_computer_is_named`` below proves the set exact against
#: ``models/losses/computers/``, so a third dropper cannot appear unnamed.
DROPPING_COMPUTERS = frozenset({"UnifiedMAELossComputer", "UnifiedDisentangledLossComputer"})

#: Modes whose strategy reaches the objective by route 4 or 5. Kept apart from
#: ``COHORT_MODES`` deliberately: the two mrixfields-specific tests below
#: (inline-L1 detection, ``_declared_inline_l1_weight`` readers) encode
#: assumptions about that cohort's type-B strategies that do not hold here.
COMPUTER_ROUTE_MODES = ("diffusion", "x_diffusion")


def _reaches(src: str) -> bool:
    """True when the module source shows any route from the builder to the objective.

    Presence-in-source, so it is prose-satisfiable in BOTH directions now
    (memory: getsource-presence-pins-are-prose-satisfiable):

    * false GREEN -- a docstring quoting ``losses_dict=`` satisfies the marker
      without any code folding anything. Inherited from the three original
      markers, not widened by adding two more.
    * false RED -- new with drop-awareness: a class that really folds, whose
      prose merely *names* ``UnifiedMAELossComputer`` (say, a comment explaining
      why it does not use one), reads as a dropped hand-off and fails the pin.

    The pin's job is to catch a flag that contradicts an obviously absent route,
    not to prove reachability -- that is what the runtime observation in the
    strategy's own docstring, and ``test_masked_strategy.py``'s empirical probe
    of the two dropping computers, are for.
    """
    if any(m in src for m in ROUTE_MARKERS if m != _HANDOFF_MARKER):
        return True
    # Route 4 is a hand-off, not a fold: it reaches the objective only when the
    # receiver reads ``losses_dict``. Widened from ``losses_dict=losses_dict``
    # (3 of 24 call sites at dev) to ``losses_dict=`` (24 of 24) on 2026-09-08;
    # the naive widening alone would have CLEARED MaskedPretrainingStrategy,
    # which hands off to a computer that discards the dict.
    return _HANDOFF_MARKER in src and not any(d in src for d in DROPPING_COMPUTERS)


def _cls(mode: str) -> type:
    path = TrainingStrategyFactory.STRATEGY_CLASS_PATHS[mode]
    module, name = path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[name]), name)


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_every_cohort_strategy_declares_its_ownership(mode: str) -> None:
    cls = _cls(mode)
    assert declared_inline_losses(cls) is not None, f"{cls.__name__} declares no inline_losses"
    assert declared_folds_image_losses(cls) is not None, f"{cls.__name__} declares no folds flag"


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_the_folds_flag_agrees_with_the_source(mode: str) -> None:
    """Planted-violation shape: flip any strategy's flag and this goes red."""
    cls = _cls(mode)
    src = inspect.getsource(inspect.getmodule(cls))
    if "_compute_losses_impl" not in cls.__dict__ and "fold_builder_image_losses(" not in src:
        # Inherits the parent's loss hook (ulf_map via PnPStrategy): the nearest
        # declaration is the parent's, and the parent's module decides.
        parent = next(k for k in cls.__mro__[1:] if "_compute_losses_impl" in k.__dict__)
        src = inspect.getsource(inspect.getmodule(parent))
    reaches = _reaches(src)
    assert declared_folds_image_losses(cls) is reaches, (
        f"{cls.__name__}.folds_image_losses={declared_folds_image_losses(cls)} but its "
        f"_compute_losses_impl {'calls' if reaches else 'never calls'} the parent or the fold"
    )


@pytest.mark.parametrize("mode", COHORT_MODES)
def test_an_inline_l1_declaration_matches_an_inline_l1_computation(mode: str) -> None:
    """``l1`` is inline iff the strategy's module computes an L1 itself."""
    cls = _cls(mode)
    module_src = inspect.getsource(inspect.getmodule(cls))
    computes_l1 = "l1_loss(" in module_src or "F.l1" in module_src or ".abs().mean()" in module_src
    declared = declared_inline_losses(cls)
    assert declared is not None
    if cls.__name__ == "PnPStrategy" or mode == "ulf_map":
        return  # the parent path computes every declared entry; nothing is inline
    if mode == "field_bridge":
        # Its endpoint anchor ``mean(abs(x_hat - x))`` is an L1 gated by
        # ``lambda_endpoint_l1`` (default 0) and weighted by that knob, not by an
        # image_losses entry, so ``l1`` is deliberately NOT declared inline.
        assert "l1" not in declared
        return
    assert ("l1" in declared) is computes_l1, (cls.__name__, sorted(declared), computes_l1)


@pytest.mark.parametrize("mode", L1_TABLE_READERS)
def test_the_inline_l1_weight_is_read_from_the_table_not_a_training_block(mode: str) -> None:
    """One owner: no reader of ``training.<mode>.lambda_l1`` survives."""
    cls = _cls(mode)
    module_src = inspect.getsource(inspect.getmodule(cls))
    assert 'getattr(cfg, "lambda_l1"' not in module_src and '_g("lambda_l1"' not in module_src
    assert "_declared_inline_l1_weight()" in module_src


def test_the_weight_read_raises_without_an_l1_entry() -> None:
    cls = _cls("brenier_synthesis")
    s = cls.__new__(cls)
    s.config = SimpleNamespace(
        losses=SimpleNamespace(image_losses=[], kspace_losses=[], complex_losses=[])
    )
    with pytest.raises(ValueError, match=r"losses\.image_losses"):
        s._declared_inline_l1_weight()


def test_the_retired_training_block_weights_raise_at_load() -> None:
    """The 15 mode schemas lost ``lambda_l1`` (``lambda_recon`` on the VIB); a rename record
    names the owner instead of pydantic's bare 'extra forbidden'."""
    from spectramr.config.schemas.renames import RENAMES

    retired = {k for k in RENAMES if k.endswith(".lambda_l1") or k.endswith(".lambda_recon")}
    assert len(retired) == 15 and all(RENAMES[k].posture == "raise" for k in retired)
    assert "training.brenier_synthesis.lambda_l1" in retired
    assert "training.recoverability_vib.lambda_recon" in retired


# --------------------------------------------------------------------------
# The computer / env-losses route (issue #1918).
#
# ``DiffusionTrainingStrategy`` hands ``env.losses`` to
# ``UnifiedDiffusionLossComputer`` as ``losses_dict`` (route 4);
# ``XDiffusionTrainingStrategy`` iterates ``env.losses`` itself and accumulates
# ``weight * loss_val`` off the same loss-weight SSOT (route 5). Both therefore
# fold every declared entry, and both must say so -- 58 kspace_filling arms
# depend on that declaration to be verifiable at all.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", COMPUTER_ROUTE_MODES)
def test_every_computer_route_strategy_declares_its_ownership(mode: str) -> None:
    cls = _cls(mode)
    assert declared_inline_losses(cls) is not None, f"{cls.__name__} declares no inline_losses"
    assert declared_folds_image_losses(cls) is not None, f"{cls.__name__} declares no folds flag"


@pytest.mark.parametrize("mode", COMPUTER_ROUTE_MODES)
def test_the_computer_route_strategies_claim_nothing_inline(mode: str) -> None:
    """Neither computes a loss itself: the computer's fidelity slot stands down when
    the same term arrives via ``losses_dict`` (``_recon_fallback_name``), and
    XDiffusion's ``mse`` fallback fires only when ``env.losses`` is empty. A
    non-empty ``inline_losses`` here would make the witness EXCLUDE that name from
    the reachability check it is the whole point of."""
    assert declared_inline_losses(_cls(mode)) == frozenset()


# --- planted violations: the pin must go red on a lying flag, both directions ---
#
# Committed rather than run once by hand (non-negotiable 15). Each feeds
# ``_reaches`` a synthetic module source, because a dynamically-built class has
# no ``inspect.getsource``. The third case is the anti-vacuity control: without
# it, a ``_reaches`` that returned False unconditionally would pass both
# violation cases and the detector would be blind.

_NO_ROUTE_SRC = """
class Lying:
    folds_image_losses = True

    def _compute_losses_impl(self, pred, target, **kw):
        return {"loss": (pred - target).abs().mean()}
"""


@pytest.mark.parametrize(
    ("declared", "src", "should_agree"),
    [
        # PLANTED: claims to fold, source shows no route at all -> pin red.
        (True, _NO_ROUTE_SRC, False),
        # PLANTED: claims to fold nothing, but forwards to the computer -> pin red.
        (False, "        losses_dict=losses_dict if losses_dict else None,", False),
        # PLANTED: claims to fold nothing, but iterates env.losses -> pin red.
        (False, "            for loss_name, loss_fn in env_losses.items():", False),
        # CONTROL: a truthful route-4 declaration -> pin green.
        (True, "        losses_dict=losses_dict if losses_dict else None,", True),
        # CONTROL: a truthful route-5 declaration -> pin green.
        (True, "            for loss_name, loss_fn in env_losses.items():", True),
        # CONTROL: a truthful "folds nothing" declaration -> pin green.
        (False, _NO_ROUTE_SRC, True),
        # PLANTED: claims to fold, hands off to a computer that DROPS the dict
        # -> pin red. This is the row the naive marker widening would have lost.
        (
            True,
            "        self.c = UnifiedMAELossComputer(...)\n        losses_dict=env_losses,",
            False,
        ),
        # CONTROL: the same hand-off to a computer that reads it -> pin green.
        (
            True,
            "        self.c = UnifiedReconstructionLossComputer(...)\n        losses_dict=env_losses,",
            True,
        ),
    ],
)
def test_the_pin_distinguishes_a_truthful_flag_from_a_lying_one(
    declared: bool, src: str, should_agree: bool
) -> None:
    assert (declared is _reaches(src)) is should_agree


def test_every_route_marker_is_load_bearing() -> None:
    """Each marker names a route some real strategy actually takes.

    A marker nothing exercises is dead vocabulary that makes the pin look wider
    than it is (the ``dedup-gate-enumerates-only-known-owners`` shape). Scanned
    over the whole strategies package, not the parametrized modes: the 30 modes
    above happen to use routes 2-5 only, so a mode-scoped version of this check
    reported route 1 as dead while 37 call sites use it.
    """
    import pathlib

    import spectramr.infrastructure.training.strategies as pkg

    root = pathlib.Path(next(iter(pkg.__path__)))
    blob = "\n".join(
        f.read_text(encoding="utf-8", errors="replace") for f in sorted(root.rglob("*.py"))
    )
    assert blob, "strategies package read as empty -- the scan, not the markers, is broken"
    unused = [m for m in ROUTE_MARKERS if m not in blob]
    assert not unused, f"ROUTE_MARKERS entries no strategy uses: {unused}"


# --------------------------------------------------------------------------
# The wide gate (issue #1918).
#
# The two pins above read 30 of the 153 strategy classes. The other 123 were
# never asked whether their flag is true, and a strategy that INHERITS
# ``folds_image_losses = True`` from a base while overriding the hook with a
# body that folds nothing is a silent PASS in the witness -- the exact shape
# that left 58 kspace_filling arms unverifiable.
# --------------------------------------------------------------------------

STRATEGY_PATHS = tuple(sorted(set(TrainingStrategyFactory.STRATEGY_CLASS_PATHS.values())))


def _cls_from_path(path: str) -> type:
    module, name = path.rsplit(".", 1)
    return getattr(__import__(module, fromlist=[name]), name)


def _hook_owner(cls: type) -> type | None:
    """The class whose ``_compute_losses_impl`` actually runs for ``cls``.

    Not ``cls`` itself: a strategy may inherit the hook from a mixin or a base,
    and it is the RUNNING body that decides whether the builder's losses reach
    the objective -- so the scan follows the MRO to the definer, whoever
    declared the flag.
    """
    return next((k for k in cls.__mro__ if "_compute_losses_impl" in k.__dict__), None)


def _class_source(cls: type) -> str:
    """``inspect.getsource(cls)`` with the duplicate-definition trap closed.

    ``getsource`` finds a class by NAME in the module, so in a file that defines
    the name twice it may hand back the wrong body (memory:
    name-keyed-dict-collapses-duplicate-defs -- 8 strategy files do this). Fail
    loud rather than pin against a stub.
    """
    module_src = inspect.getsource(inspect.getmodule(cls))
    definitions = sum(
        1
        for node in ast.walk(ast.parse(module_src))
        if isinstance(node, ast.ClassDef) and node.name == cls.__name__
    )
    assert definitions == 1, (
        f"{cls.__module__} defines {cls.__name__} {definitions} times -- inspect.getsource "
        f"cannot tell them apart, so this pin would read an arbitrary one"
    )
    return inspect.getsource(cls)


@pytest.mark.parametrize("path", STRATEGY_PATHS, ids=lambda p: p.rsplit(".", 1)[1])
def test_a_strategy_that_claims_to_fold_shows_a_route(path: str) -> None:
    """Every class whose EFFECTIVE ``folds_image_losses`` is True must show a route.

    One-directional on purpose -- ``declared is True`` implies a route, but a
    False declaration is not required to show none. The asymmetry is in the
    consequences: declaring False makes the witness report every non-inline
    ``image_losses`` entry through ``unreachable_image_losses``, so
    under-declaring is loud and self-punishing, while over-declaring buys a
    silent PASS. Only the silent direction needs a gate. (It is also what lets
    ``SSDUReconstructionStrategy`` declare False truthfully: it matches route 5
    textually, but its loop body is ``if "ssdu" in name`` -- every other
    declared image loss is skipped, and no textual marker can see that filter.)

    Scanned at CLASS granularity, not the module: strategy files hold several
    classes, so a module scan lets one sibling's fold clear a neighbour that
    folds nothing. Not the method body either -- ``ReconstructionTrainingStrategy``
    keeps its fold in ``_apply_builder_image_losses``, a separate method of the
    same class, and a method-scoped scan falsely accuses it.
    """
    cls = _cls_from_path(path)
    if declared_folds_image_losses(cls) is not True:
        pytest.skip("does not claim to fold -- the witness reports it, no silent PASS to catch")
    owner = _hook_owner(cls)
    if owner is None:
        pytest.skip("defines no _compute_losses_impl anywhere in its MRO")
    assert _reaches(_class_source(owner)), (
        f"{cls.__name__}.folds_image_losses is True, but the _compute_losses_impl that runs "
        f"for it ({owner.__name__}) shows no route to the objective: none of {ROUTE_MARKERS} "
        f"appears in its class body (or route 4 appears but hands off to a dropping computer). "
        f"Either it folds and the route is unrecognised, or the flag is a lie and every "
        f"declared losses.image_losses entry on its arms is silently discarded."
    )


def test_every_dropping_computer_is_named() -> None:
    """``DROPPING_COMPUTERS`` is exact against the computers package.

    The one-owner half of non-negotiable 17: the constant decides whether route 4
    counts, so an unnamed third dropper would silently clear a strategy. Shrink-only
    -- a computer that LEARNS to read ``losses_dict`` must leave the set in the same
    change, and a new one that ignores it must be added here or this goes red.
    """
    import spectramr.models.losses.computers as computers_pkg

    root = pathlib.Path(next(iter(computers_pkg.__path__)))
    drops: set[str] = set()
    for file in sorted(root.rglob("*.py")):
        tree = ast.parse(file.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not node.name.endswith("LossComputer"):
                continue
            hooks = [
                fn
                for fn in node.body
                if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef)
                and fn.name in {"compute", "compute_forward_loss"}
            ]
            # An abstract ``compute`` declares the signature and computes nothing,
            # so it cannot drop what it never receives. Detected by the decorator,
            # not by a ``Base*`` name match: the next abstract layer may be spelled
            # anything (pitfall #15 -- a gate is only as wide as its ugliest shape).
            if any(
                isinstance(d, ast.Name) and d.id == "abstractmethod"
                for fn in hooks
                for d in fn.decorator_list
            ):
                continue
            body = "\n".join(ast.unparse(fn) for fn in hooks)
            if body and "losses_dict" not in body:
                drops.add(node.name)
    assert drops, "computers package scanned as empty -- the scan, not the set, is broken"
    assert drops == DROPPING_COMPUTERS, (
        f"DROPPING_COMPUTERS is stale: only in the set {sorted(DROPPING_COMPUTERS - drops)}, "
        f"only in the source {sorted(drops - DROPPING_COMPUTERS)}"
    )
