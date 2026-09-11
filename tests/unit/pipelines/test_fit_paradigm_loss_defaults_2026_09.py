"""Every ``fit()`` paradigm default survives the loss builder call ``fit()`` makes.

``_PARADIGM_DEFAULTS`` injects a ``losses:`` block programmatically, so no file
under ``experiments/`` carries these shapes and a corpus census over YAML cannot
see them. That blind spot is not hypothetical: a guard added to ``LossBuilder``
in 2026-09 refused ``losses.latent.lambda_kl``, which only the ``vae`` and
``vqvae`` defaults declare, and the 656-arm YAML census stayed green.

The call under test is the one ``pipelines/fit.py`` makes, not a convenient
subset of it — the builder's eight ``build_*`` methods are aliases for the same
dynamic pass, so a chain that omits one proves nothing the single call does not.
"""

from __future__ import annotations

import pytest
import torch


def _paradigms() -> list[str]:
    from spectramr.pipelines.fit import _PARADIGM_ALIASES, _PARADIGM_DEFAULTS

    return sorted(set(_PARADIGM_DEFAULTS) | set(_PARADIGM_ALIASES))


def _settings_for(paradigm: str):
    """The config ``fit()`` assembles for one paradigm, minus the runtime knobs."""
    from spectramr.config.settings import TrainingSettings
    from spectramr.pipelines.fit import _PARADIGM_ALIASES, _PARADIGM_DEFAULTS

    family = _PARADIGM_ALIASES.get(paradigm, paradigm)
    defaults = _PARADIGM_DEFAULTS[family]
    training = dict(defaults["training"])
    training.setdefault("strategy_class", paradigm)
    training["epochs"] = 1
    return TrainingSettings.settings_from_dict(
        {
            "model": {"model_type": "unet"},
            "data": {"dataset_type": "synthetic"},
            "optimization": {},
            "logging": {},
            "checkpoint": {},
            "losses": defaults["losses"],
            "training": training,
        }
    )


def test_the_sweep_covers_every_declared_paradigm():
    """A shrunken target list would make the parametrised test below vacuous."""
    from spectramr.pipelines.fit import _PARADIGM_ALIASES, _PARADIGM_DEFAULTS

    paradigms = _paradigms()
    assert len(paradigms) >= 24, paradigms
    assert {"reconstruction", "gan", "vae", "vqvae", "diffusion"} <= set(paradigms)
    # Every alias must route to a real block, or its row is untested rather than passing.
    unrouted = sorted(t for t in _PARADIGM_ALIASES.values() if t not in _PARADIGM_DEFAULTS)
    assert not unrouted, unrouted


@pytest.mark.parametrize("paradigm", _paradigms())
def test_paradigm_default_losses_build(paradigm):
    """``pipelines/fit.py``'s own call, verbatim.

    A paradigm whose defaults cannot build is a run that cannot start, and the
    defaults are what an author gets when they name a paradigm and write no
    ``losses:`` block at all.
    """
    from spectramr.infrastructure.training.builders.loss_builder import LossBuilder

    settings = _settings_for(paradigm)
    LossBuilder(settings, torch.device("cpu")).build_reconstruction_losses().build()


def test_the_vae_defaults_declare_a_lambda_only_kl():
    """Pins the shape that made the guard's false refusal reachable.

    If ``vae`` ever migrates ``lambda_kl`` onto a domain list, the case above
    stops covering the lambda-only path and this test says so.
    """
    from spectramr.pipelines.fit import _PARADIGM_DEFAULTS

    losses = _PARADIGM_DEFAULTS["vae"]["losses"]
    assert losses["latent"]["lambda_kl"] == 1.0
    assert not any(e.get("name") == "kl" for e in losses.get("image_losses", []))
