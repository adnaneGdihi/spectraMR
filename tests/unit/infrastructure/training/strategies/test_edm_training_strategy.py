"""EDM declares the tensor its network actually receives.

``EDMTrainingStrategy`` overrides ``_compute_losses_impl``, and so inherited
``DiffusionTrainingStrategy``'s ``snapshot_prepared_is_model_input = False`` /
``snapshot_model_input_tag = "diffusion_step"`` without ever emitting that tag —
its artifacts told a reader "the visible input is not the model input, see
``diffusion_step``" and no such snapshot existed (non-negotiable 14).

EDM is the sharpest case in the family, because capturing the *obvious* tensor
would still have been wrong. Karras preconditioning feeds ``c_in * noised``, not
``noised``; ``c_in`` spans orders of magnitude across the sigma schedule, so the
two do not even share a dynamic range. A snapshot of ``noised`` would look
plausible and misreport the input.

Bare-instance pattern: ``_compute_losses_impl`` is exercised directly with the
few seams it reads stubbed, so no config/DI/env build is needed.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from spectramr.infrastructure.training.strategies.edm_training_strategy import (
    EDMTrainingStrategy,
)


def _bare(recorder: dict) -> EDMTrainingStrategy:
    s = object.__new__(EDMTrainingStrategy)
    s.device = torch.device("cpu")
    s._resolve_legacy_batch = lambda input_batch, kwargs: input_batch  # type: ignore[method-assign]
    s._edm_schedule = SimpleNamespace(
        sample_sigma=lambda b, device: torch.full((b,), 1.5, device=device)
    )
    s._declared_model_input = None

    def _gen(x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        recorder["model_saw"] = x
        return torch.zeros_like(x)

    s.env = SimpleNamespace(generator=_gen)
    return s


def test_edm_declares_the_preconditioned_input_not_the_noised_one() -> None:
    recorder: dict = {}
    s = _bare(recorder)
    clean = torch.randn(2, 1, 8, 8)

    s._compute_losses_impl(input_batch={"target": clean}, target_batch=clean, epoch=0)

    assert s._declared_model_input is not None, (
        "EDM inherits the carve-out, so it owes the wrapper a declaration"
    )
    tensors, extra, in_kspace_keys = s._declared_model_input

    # The contract: what was declared IS what the network received.
    assert torch.equal(tensors["model_input"], recorder["model_saw"])

    # And the distinction that makes this worth testing: `noised` is captured
    # for contrast, but it is NOT the model input.
    assert not torch.equal(tensors["noised"], recorder["model_saw"]), (
        "c_in preconditioning must separate the declared input from `noised`"
    )
    assert extra["model_input_key"] == "model_input"


def test_edm_names_kspace_keys_explicitly() -> None:
    """`None` would fall back to a `"kspace"` substring match over key names.

    That fallback misses `target` entirely (findings booklet VIS-1). An empty
    set is the explicit answer: nothing here is k-space *by name*, and the
    canonical keys get unioned in from the config SSOT for a k-space arm — which
    is correct, since every tensor declared here is a linear function of the
    arm's own `target`.
    """
    recorder: dict = {}
    s = _bare(recorder)
    clean = torch.randn(2, 1, 8, 8)

    s._compute_losses_impl(input_batch={"target": clean}, target_batch=clean, epoch=0)

    _tensors, _extra, in_kspace_keys = s._declared_model_input
    assert in_kspace_keys == set(), "must be explicit, not None"


def test_edm_still_returns_its_loss() -> None:
    """The declaration is a side effect and must not disturb the objective."""
    recorder: dict = {}
    s = _bare(recorder)
    clean = torch.randn(2, 1, 8, 8)

    losses = s._compute_losses_impl(input_batch={"target": clean}, target_batch=clean, epoch=0)

    assert "loss_total" in losses and "loss_edm" in losses
    assert torch.isfinite(losses["loss_total"])


def test_it_declares_its_own_loss_ownership() -> None:
    """Issue #1918: EDM score matching computes no image loss and routes nowhere.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert EDMTrainingStrategy.__dict__["folds_image_losses"] is False
    assert EDMTrainingStrategy.__dict__["inline_losses"] == frozenset()
