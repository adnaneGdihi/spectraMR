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

    def _gen(x: torch.Tensor, c_noise: torch.Tensor) -> torch.Tensor:
        recorder["model_saw"] = x
        return torch.zeros_like(x)

    s.env = SimpleNamespace(generator=_gen)
    return s


def test_edm_declares_the_preconditioned_input_not_the_noised_one(declared) -> None:
    recorder: dict = {}
    s = _bare(recorder)
    clean = torch.randn(2, 1, 8, 8)

    s._compute_losses_impl(input_batch={"target": clean}, target_batch=clean, epoch=0)

    assert declared() is not None, (
        "EDM builds its model input inside the step, so it names it for the snapshot"
    )
    tensors, extra, _in_kspace_keys = declared()

    # The contract: what was declared IS what the network received.
    assert torch.equal(tensors["model_input"], recorder["model_saw"])

    # And the distinction that makes this worth testing: `noised` is captured
    # for contrast, but it is NOT the model input.
    assert not torch.equal(tensors["noised"], recorder["model_saw"]), (
        "c_in preconditioning must separate the declared input from `noised`"
    )
    assert extra["model_input_key"] == "model_input"


def test_edm_names_kspace_keys_explicitly(declared) -> None:
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

    _tensors, _extra, in_kspace_keys = declared()
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
    """Issue #1918: EDM computes its denoising term inline and folds declared image losses.

    Read off ``__dict__``, never the inherited value: this class sits under
    ``DiffusionTrainingStrategy``, whose ``folds_image_losses = True`` is truthful for ITSELF and
    becomes a lie the moment a subclass replaces ``_compute_losses_impl``. An
    inherited True makes the audit's ``image_losses_reach_the_objective`` witness
    PASS every declared ``losses.image_losses`` entry on this strategy's arms
    while the training step discards them.
    """
    assert EDMTrainingStrategy.__dict__["folds_image_losses"] is True
    assert EDMTrainingStrategy.__dict__["empty_loss_stack_is_design"] is True
    assert EDMTrainingStrategy.__dict__["inline_losses"] == frozenset()


def _image_cfg(weight: float) -> SimpleNamespace:
    entry = SimpleNamespace(name="l1", weight=weight, enabled=True, role="target")
    return SimpleNamespace(losses=SimpleNamespace(image_losses=[entry]))


class _RecordingLoss:
    """Image loss that records what it was asked to compare."""

    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self.calls.append((pred, target))
        return (pred - target).abs().mean()


def test_edm_declared_image_loss_is_folded_onto_the_denoised_estimate() -> None:
    clean = torch.randn(2, 1, 8, 8)
    weight = 0.5

    torch.manual_seed(0)
    s_off = _bare({})
    s_off.config = _image_cfg(weight)
    loss_off = s_off._compute_losses_impl(input_batch={"target": clean}, target_batch=clean)[
        "loss_total"
    ]

    torch.manual_seed(0)
    s_on = _bare({})
    s_on.config = _image_cfg(weight)
    l1 = _RecordingLoss()
    s_on.env.losses = {"l1": l1}
    out = s_on._compute_losses_impl(input_batch={"target": clean}, target_batch=clean)

    assert "loss_l1" in out, "a folded declared term must reach the per-step dict"
    ((pred, target),) = l1.calls
    assert torch.equal(target, clean), "the declared term compares the denoised estimate to clean"
    expected = weight * (pred - clean).abs().mean()
    assert torch.allclose(out["loss_total"] - loss_off, expected)


def test_edm_fold_ignores_terms_not_declared_as_image_losses() -> None:
    clean = torch.randn(2, 1, 8, 8)
    s = _bare({})
    s.config = _image_cfg(0.5)
    kspace = _RecordingLoss()
    s.env.losses = {"kspace_l1": kspace}
    s._compute_losses_impl(input_batch={"target": clean}, target_batch=clean)
    assert kspace.calls == [], "an undeclared (k-space) term must never be applied to image tensors"
