"""Registry census — the runtime backstop for model-registration integrity.

Where ``test_registry_integrity.py`` statically scans source for decorator
hygiene, this suite populates the *live* ``MODEL_REGISTRY`` and asserts:

* a global count floor + per-mode floors (a whole family going dark — e.g. the
  ~41 SOTA registrations silently dropped by a swallowed ``ImportError`` in
  ``init_registry`` — trips the floor);
* the full SOTA roster is present with the EXACT ``(name, mode)`` it registers
  (migration-agnostic: whether a name registers via ``sota_registry`` or, after
  the decorator migration, via the pkgutil walk, the pin is the same);
* no *dark decorator* — every statically-declared ``@register_model`` name is in
  the populated registry unless it is an acknowledged ``REJECTED_NAMES`` entry;
* ``REJECTED_NAMES`` hygiene — a rejected name is absent from the live registry.

This is the true guard behind the ``init_registry`` fail-loud change: even if a
future edit re-softens the guard, the roster/floor assertions here fail in CI.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def registry() -> dict:
    from spectramr.models.init_registry import populate_model_registry
    from spectramr.models.registry import MODEL_REGISTRY

    populate_model_registry()
    return MODEL_REGISTRY


# The SOTA innovation roster with the exact mode each name registers under.
# Pinned so a swallowed import (or a mode drift during the decorator migration)
# is a hard CI failure, not a silent catalog loss. Refresh deliberately.
SOTA_ROSTER: tuple[tuple[str, str], ...] = (
    ("pa_pcfm", "diffusion"),
    ("res_mocodiff", "diffusion"),
    ("mc_cm", "diffusion"),
    ("fourier_kan_swin", "reconstruction"),
    ("varnet_kan", "reconstruction"),
    ("kan_inr", "reconstruction"),
    ("d2_mamba", "reconstruction"),
    ("cv_ssd", "reconstruction"),
    ("bloch_mamba", "reconstruction"),
    ("bloch_mamba_v2", "reconstruction"),
    ("diff_mamba", "reconstruction"),
    ("neuro_mamba", "reconstruction"),
    ("vision_mamba", "reconstruction"),
    ("gen_3dgs", "synthesis"),
    ("dynamic_3dgs", "reconstruction"),
    ("gauss_mrf", "reconstruction"),
    ("hnf", "reconstruction"),
    ("k_hash", "reconstruction"),
    ("q_inr", "reconstruction"),
    ("cold_diffusion_process", "diffusion"),
    # metal_diffusion retired 2026-08-12 — facade alias of cold_diffusion_process
    # (the name promised a degradation the alias never set). ColdDiffusion stays
    # reachable under its canonical name, pinned by the row above.
    ("lie_diffusion", "diffusion"),
    ("graph_kan", "reconstruction"),
    ("graph_cuts_neural", "reconstruction"),
    ("spectral_graph", "reconstruction"),
    ("swin_mamba_kan", "reconstruction"),
    ("umr_fm", "reconstruction"),
    ("ttt_mamba", "reconstruction"),
    ("q_kan", "reconstruction"),
    ("causal_gan", "synthesis"),
    ("rl_mri", "acquisition"),
    ("eq_deq", "reconstruction"),
    ("neural_ode_recon", "reconstruction"),
    ("meta_tta", "reconstruction"),
    ("diff_seq", "reconstruction"),
    ("therm_pinn", "reconstruction"),
    ("gen_mrf", "reconstruction"),
    ("lagrange_euler", "reconstruction"),
    ("score_disent", "diffusion"),
    ("nsp_tta", "reconstruction"),
    ("homotopic_transport", "reconstruction"),
    ("q_fno", "reconstruction"),
)

# Per-mode count floors, set a little below the current live counts so ordinary
# additions never trip them but a whole family disappearing does.
MODE_FLOORS: dict[str, int] = {
    "reconstruction": 320,
    "diffusion": 70,
    "gan": 38,
    "vae": 26,
    "domain_adaptation": 6,
}

GLOBAL_FLOOR = 580


def test_global_count_floor(registry: dict) -> None:
    assert len(registry) >= GLOBAL_FLOOR, (
        f"registry has {len(registry)} models, below the floor {GLOBAL_FLOOR} — "
        "a subpackage likely went dark (swallowed import / walk-list regression)."
    )


def test_per_mode_floors(registry: dict) -> None:
    from collections import Counter

    counts = Counter(
        v["mode"] for v in registry.values() if isinstance(v, dict) and v.get("mode")
    )
    low = {m: (counts[m], floor) for m, floor in MODE_FLOORS.items() if counts[m] < floor}
    assert not low, f"mode families below floor (got, floor): {low}"


@pytest.mark.parametrize("name,mode", SOTA_ROSTER, ids=[n for n, _ in SOTA_ROSTER])
def test_sota_roster_present_with_mode(registry: dict, name: str, mode: str) -> None:
    assert name in registry, (
        f"SOTA model {name!r} missing — a swallowed ImportError in init_registry "
        "may have dropped the whole SOTA catalog."
    )
    assert registry[name]["mode"] == mode, (
        f"SOTA model {name!r} registered under mode {registry[name]['mode']!r}, "
        f"expected {mode!r} (registration-mode drift changes builder dispatch)."
    )


def _static_register_model_names() -> set[str]:
    """The name every ``@register_model`` decorator declares, parsed with ``ast``.

    Not a regex. The one this replaced required ``name=`` to be the decorator's
    FIRST argument, which stopped being true when a change put ``role=`` ahead of
    it: the scan then saw 509 names where the tree declares 530.

    The 21 it lost were invisible because of the DIRECTION this test asserts in.
    ``test_no_dark_decorators`` checks that every *statically declared* name is
    live, so dropping names from the static set cannot fail it — it just stops
    asking about them. A detector that quietly narrows still reports success,
    which is why the sibling scanner in ``test_registry_integrity.py`` is parsed
    the same way and carries planted decorators for each shape.

    The test-file skip is matched against the path RELATIVE to the scan root, so
    it does not depend on where the repository is checked out.
    """
    base = Path(__file__).resolve().parents[3] / "src" / "spectramr" / "models"
    names: set[str] = set()
    for py in base.rglob("*.py"):
        if any("test" in part for part in py.relative_to(base).parts):
            continue
        try:
            tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for node in ast.walk(tree):
            for dec in getattr(node, "decorator_list", []):
                if not isinstance(dec, ast.Call):
                    continue
                called = getattr(dec.func, "id", None) or getattr(dec.func, "attr", None)
                if called != "register_model":
                    continue
                declared = next(
                    (
                        kw.value.value
                        for kw in dec.keywords
                        if kw.arg == "name" and isinstance(kw.value, ast.Constant)
                    ),
                    None,
                )
                if declared is None and dec.args and isinstance(dec.args[0], ast.Constant):
                    declared = dec.args[0].value
                if isinstance(declared, str):
                    names.add(declared)
    return names


def test_no_dark_decorators(registry: dict) -> None:
    """Every statically-declared @register_model name is live (unless rejected).

    A ``@register_model`` decorator in a module the walk never imports is
    silently dead — the name is advertised in schemas but resolves to a
    ``KeyError`` at build time. This catches that class directly.
    """
    from spectramr.models.registry import REJECTED_NAMES

    declared = _static_register_model_names()
    rejected = set(REJECTED_NAMES)
    dark = sorted(n for n in declared if n not in registry and n not in rejected)
    assert not dark, (
        "dark @register_model decorators (declared in source, absent from the "
        f"populated registry, not in REJECTED_NAMES): {dark}"
    )


def test_rejected_names_absent_from_registry(registry: dict) -> None:
    from spectramr.models.registry import REJECTED_NAMES

    leaked = sorted(n for n in REJECTED_NAMES if n in registry)
    assert not leaked, (
        f"REJECTED_NAMES that are actually registered (stale rejection): {leaked}"
    )
