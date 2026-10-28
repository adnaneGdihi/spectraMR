"""Cold-import registration audit for the 2026 breakthrough components.

This test spawns a *fresh* Python subprocess for each registry check so
module-cache effects from sibling tests cannot mask a silently-dead
``@register_*`` decorator. The CLAUDE.md pitfall #9 ("a `@register_loss`
decorator that nothing imports is silently dead") becomes a regression
test: if anyone removes the `spectral_triple_loss` line from
`src/models/losses/__init__.py`, this test fails.

The eight registry surfaces checked are:

* 4 metrics aliases — ``galois_h1_certificate``, ``asymptotic_gfactor``,
  ``persistence_diameter``, ``mrf_persistence_stability``.
* 3 loss aliases — ``spectral_triple_intertwining``,
  ``connes_intertwining``, ``morita_morphism``.
* 5 generator names — ``toeplitz_attention_unet``,
  ``bloch_lrs_attention_unet``, ``lanczos_attention_unet``,
  ``tjl_mrf_estimator``, ``mps_attention_unet``.
* 2 strategy aliases — ``sle_compressed_sensing``, ``spectral_triple``.
* 1 mask type — ``MaskType.SLE_KAPPA``.
* 1 transforms-package re-export — ``build_sle_kspace_mask``.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cold(check: str) -> tuple[int, str]:
    """Run ``check`` as a fresh Python process so module imports start cold."""
    proc = subprocess.run(
        [sys.executable, "-c", check],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout + proc.stderr


_METRIC_ALIASES = [
    "galois_h1_certificate",
    "h1_cert",
    "cohomology_certificate",
    "asymptotic_gfactor",
    "free_prob_gfactor",
    "persistence_diameter",
    "mrf_persistence_stability",
    "mrf_bottleneck",
]


@pytest.mark.parametrize("alias", _METRIC_ALIASES)
def test_metric_alias_resolves_from_cold_import(alias: str) -> None:
    check = (
        "from spectramr.core.metrics import get_metric; "
        f"m = get_metric({alias!r}); "
        "assert m is not None, 'get_metric returned None'"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for metric {alias!r}:\n{out}"


_LOSS_ALIASES = [
    "spectral_triple_intertwining",
    "connes_intertwining",
    "morita_morphism",
]


@pytest.mark.parametrize("alias", _LOSS_ALIASES)
def test_loss_alias_resolves_from_cold_import(alias: str) -> None:
    """If `spectral_triple_loss` is dropped from src/models/losses/__init__.py,
    this test fails — the @register_loss decorator no longer fires."""
    check = (
        "from spectramr.models.losses.registry import LossRegistry; "
        f"loss = LossRegistry.create({alias!r}); "
        "assert loss is not None"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for loss {alias!r}:\n{out}"


_GENERATOR_NAMES = [
    "toeplitz_attention_unet",
    "bloch_lrs_attention_unet",
    "lanczos_attention_unet",
    "tjl_mrf_estimator",
    "mps_attention_unet",
    # Batch 1 geometric-prior generators.
    "hyperbolic_cine_unet",
    "heisenberg_recon_unet",
]


@pytest.mark.parametrize("name", _GENERATOR_NAMES)
def test_generator_resolves_from_cold_import(name: str) -> None:
    """A `@register_model` decorator on an un-imported generator file is dead.
    This test forces a cold load of `spectramr.models.generators` (which mounts the
    breakthrough module) and checks the registry."""
    check = (
        "import spectramr.models.generators; "
        "from spectramr.models.registry import MODEL_REGISTRY; "
        f"entry = MODEL_REGISTRY.get({name!r}); "
        f"assert entry is not None, 'missing {name}'"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for generator {name!r}:\n{out}"


_STRATEGY_ALIASES = [
    "sle_compressed_sensing",
    "spectral_triple",
    # Batch 1 + 2 thin reconstruction/diffusion aliases.
    "tropical_mrf",
    "fisher_rao_flow",
    "sheaf_kspace",
    "resetting_diffusion",
    "hyperbolic_cardiac",
    "heisenberg_phase",
    "hjb_trajectory",
    "primary_ideal",
    "symplectic_bloch",
    "frontdoor_federated",
    "hodge_motion",
    "levy_diffusion",
    "koopman_fmri",
    "sheaf_multi_contrast",
    "hjb_waveform",
    "stein_federated",
    "tropical_quantitative_maps",
    "frontdoor_scanner",
]


@pytest.mark.parametrize("alias", _STRATEGY_ALIASES)
def test_strategy_alias_in_class_paths(alias: str) -> None:
    check = (
        "from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory; "
        f"assert {alias!r} in TrainingStrategyFactory.STRATEGY_CLASS_PATHS"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for strategy alias {alias!r}:\n{out}"


def test_sle_kappa_mask_type_present() -> None:
    check = (
        "from spectramr.infrastructure.physics.sampling import MaskType; "
        "assert MaskType.SLE_KAPPA.value == 'sle_kappa'"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for MaskType.SLE_KAPPA:\n{out}"


def test_sle_trajectory_exposed_from_transforms_package() -> None:
    check = (
        "from spectramr.data.transforms import build_sle_kspace_mask, kappa_to_dimension, sample_sle_trace; "
        "assert callable(build_sle_kspace_mask)"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for transforms re-export:\n{out}"


_VF_PARM_MODES = [
    "equivariance_conformal",
    "bloch_manifold_dps",
    "se3_equivariant_navigator",
    "hamiltonian_acquisition",
]


@pytest.mark.parametrize("mode", _VF_PARM_MODES)
def test_vf_parm_mode_registered_in_strategy_paths_and_modes(mode: str) -> None:
    """VF-smoke-2026-05-25 regression: the four "P-arm" breakthrough strategies
    (B-1..B-4) must stay in both ``STRATEGY_CLASS_PATHS`` and ``VALID_TRAINING_MODES``."""
    check = (
        "from spectramr.infrastructure.training.strategy_factory import TrainingStrategyFactory; "
        "from spectramr.config.validation_constants import VALID_TRAINING_MODES; "
        f"assert {mode!r} in TrainingStrategyFactory.STRATEGY_CLASS_PATHS, 'missing from STRATEGY_CLASS_PATHS'; "
        f"assert {mode!r} in VALID_TRAINING_MODES, 'missing from VALID_TRAINING_MODES'"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for VF P-arm mode {mode!r}:\n{out}"


def test_training_modes_include_breakthrough_aliases() -> None:
    # Inline a list literal so the subprocess -c flag receives a single expression.
    aliases = (
        "['sle_compressed_sensing','spectral_triple','tropical_mrf',"
        "'fisher_rao_flow','sheaf_kspace','resetting_diffusion',"
        "'hyperbolic_cardiac','heisenberg_phase','hjb_trajectory',"
        "'primary_ideal','symplectic_bloch','frontdoor_federated',"
        "'hodge_motion','levy_diffusion','koopman_fmri',"
        "'sheaf_multi_contrast','hjb_waveform','stein_federated',"
        "'tropical_quantitative_maps','frontdoor_scanner']"
    )
    check = (
        "from spectramr.config.validation_constants import VALID_TRAINING_MODES; "
        f"aliases = {aliases}; "
        "missing_modes = [a for a in aliases if a not in VALID_TRAINING_MODES]; "
        "assert not missing_modes, f'missing from modes: {missing_modes}'"
    )
    rc, out = _run_cold(check)
    assert rc == 0, f"cold import failed for training-mode constants:\n{out}"
