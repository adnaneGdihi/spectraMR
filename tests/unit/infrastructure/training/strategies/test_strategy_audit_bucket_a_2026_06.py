"""Regression tests for the 2026-06 DESIGN bucket-A quick fixes.

Each pins one small DESIGN-bucket cleanup: motion_meta identity-FFT removal,
n2n repetition randomization, standard/meta_learning docstring accuracy,
synthetic_pathology total-key guard, and the sfc conformal-DC degrade warning.
"""

from __future__ import annotations

import inspect

import torch


def test_motion_meta_drops_identity_fft_roundtrip() -> None:
    from spectramr.infrastructure.training.strategies.motion_meta_strategy import (
        ConcreteMotionMetaTrainingStrategy,
    )

    src = inspect.getsource(ConcreteMotionMetaTrainingStrategy)
    assert "ifft2c(fft2c(target_complex))" not in src  # the no-op round-trip
    assert "target_image = target_complex" in src


def test_n2n_picks_two_distinct_repetitions() -> None:
    from spectramr.infrastructure.training.strategies.n2n_strategy import pair_from_stack

    # Slot k is filled with the constant k, so distinct slots have distinct means.
    stack = torch.cat([torch.full((1, 1, 4, 4), float(k)) for k in range(3)], dim=1)
    inp = stack[:, :1]
    for _ in range(8):  # randomized pick: distinctness must hold every time
        tgt = pair_from_stack(inp, stack, "random_pair")
        assert inp.mean().item() != tgt.mean().item()


def test_standard_docstring_drops_unread_training_mode() -> None:
    from spectramr.infrastructure.training.strategies.standard_strategy import (
        StandardTrainingStrategy,
    )

    doc = StandardTrainingStrategy.__doc__ or ""
    assert "`training.training_mode`: 'standard' or 'custom'" not in doc


def test_meta_learning_docstring_lists_real_homes() -> None:
    from spectramr.infrastructure.training.strategies.meta_learning_strategy import (
        MetaLearningTrainingStrategy,
    )

    doc = MetaLearningTrainingStrategy.__doc__ or ""
    # The docstring now lists the actual multi-home scan + aliases.
    assert "model.adaptation_config" in doc
    assert "config.meta_learning" in doc
    assert "meta_lr_inner" in doc


def test_synthetic_pathology_guards_missing_total_key() -> None:
    from spectramr.infrastructure.training.strategies.synthetic_pathology_aug_strategy import (
        SyntheticPathologyAugStrategy,
    )

    src = inspect.getsource(SyntheticPathologyAugStrategy)
    # Both lesion-term folds now raise (for/else) if no total key is present.
    assert src.count("would be silently dropped") >= 1
    assert "raise KeyError" in src


def test_sfc_conformal_missing_dc_inputs_raises_not_warns() -> None:
    """Missing ``mask``/``conformal_jacobian`` must RAISE, not warn-and-degrade.

    Rewritten 2026-07-14 (issue #189). This test used to grep the source for the
    strings ``"conformal data-consistency is SKIPPED"`` and ``_warned_no_dc`` —
    i.e. it asserted that the *degrade* path still existed. The strategy has since
    been upgraded to fail loud: without those batch keys the conformal DC step
    cannot run, the loss collapses to an ordinary Gaussian-denoising MSE, and the
    "conformal" paradigm is inert while the run smoke-PASSes (pitfall #16). So the
    source outgrew the test, which then pinned the very degradation the repo
    forbids — and had been red on ``dev`` ever since.

    Asserting behaviour rather than source text: a string-grep test cannot tell a
    warning from a raise, which is exactly the distinction that matters here.
    """
    from spectramr.infrastructure.training.strategies.sfc_conformal_kspace_strategies import (
        ConformalDiffusionReconStrategy,
    )

    src = inspect.getsource(ConformalDiffusionReconStrategy)
    assert (
        "conformal data-consistency is SKIPPED" not in src
    ), "the warn-and-run-plain-MSE degrade path is back — it must raise"
    assert "raise ValueError(" in src
    assert "conformal_jacobian" in src
