"""SToRM manifold regulariser — the Laplacian must actually be a Laplacian.

The headline test here is ``test_uniform_laplacian_is_psd_...``: the previous
implementation was not merely imprecise, it was **unbounded below**, so the
"smoothness penalty" rewarded blowing the reconstruction up.
"""

from __future__ import annotations

import pytest
import torch

from spectramr.config.schemas.enums import Regime
from spectramr.models.losses.storm_loss import (
    SToRMRegularizer,
    _uniform_temporal_laplacian,
)


def _old_buggy_laplacian(batch: int, frames: int) -> torch.Tensor:
    """The implementation this replaced, kept inline as the counter-example.

    Reproduced verbatim so the test below demonstrates the defect rather than
    asserting a claim about code that no longer exists. Note ``=`` not ``+=``:
    each iteration overwrites the previous one's diagonal entry, so the interior
    diagonal ends at 1 instead of the node degree 2.
    """
    lap = torch.zeros(batch, frames, frames)
    for i in range(frames - 1):
        lap[:, i, i] = 1
        lap[:, i, i + 1] = -1
        lap[:, i + 1, i] = -1
        lap[:, i + 1, i + 1] = 1
    return lap


@pytest.mark.parametrize("frames", [2, 3, 5, 8])
def test_uniform_laplacian_is_psd_and_penalises_frame_differences(frames: int) -> None:
    """The three properties that make it a Laplacian, on the tensor we ship."""
    lap = _uniform_temporal_laplacian(
        1, frames, device=torch.device("cpu"), dtype=torch.float64
    )
    single = lap[0]

    # 1. Row sums vanish: a constant sequence is perfectly smooth, cost 0.
    assert torch.allclose(
        single.sum(-1), torch.zeros(frames, dtype=torch.float64), atol=1e-12
    )
    # 2. Positive semi-definite: the penalty can never go below 0.
    assert float(torch.linalg.eigvalsh(single).min()) >= -1e-10
    # 3. Symmetric.
    assert torch.allclose(single, single.T)


@pytest.mark.parametrize("frames", [3, 5])
def test_the_old_laplacian_was_unbounded_below(frames: int) -> None:
    """RED-against-the-bug: pins WHY the loop had to be replaced.

    The old matrix had a negative eigenvalue, so ``Tr(X L X^T)`` had no lower
    bound: a constant frame sequence scored NEGATIVE, and scaling it drove the
    "penalty" toward -inf. Minimising it rewarded blowing the reconstruction up
    while the loss curve went satisfyingly down.
    """
    old = _old_buggy_laplacian(1, frames)[0].double()
    assert float(torch.linalg.eigvalsh(old).min()) < -0.1
    assert not torch.allclose(old.sum(-1), torch.zeros(frames, dtype=torch.float64))

    # The concrete failure: a constant sequence scores below zero.
    x = torch.ones(1, frames, 1, dtype=torch.float64)
    old_cost = float(x.transpose(1, 2) @ old.unsqueeze(0) @ x)
    assert old_cost < 0

    # The fix scores it at exactly zero — a constant sequence IS smooth.
    fixed = _uniform_temporal_laplacian(
        1, frames, device=torch.device("cpu"), dtype=torch.float64
    )
    new_cost = float(x.transpose(1, 2) @ fixed @ x)
    assert new_cost == pytest.approx(0.0, abs=1e-10)


def test_quadratic_form_equals_the_sum_of_squared_frame_differences() -> None:
    """The identity that makes Tr(X L X^T) mean "temporal smoothness"."""
    frames = 5
    x = torch.randn(1, frames, 3, dtype=torch.float64)
    lap = _uniform_temporal_laplacian(
        1, frames, device=torch.device("cpu"), dtype=torch.float64
    )
    quad = float(torch.diagonal(x.transpose(1, 2) @ lap @ x, dim1=1, dim2=2).sum())
    expected = float(((x[:, 1:] - x[:, :-1]) ** 2).sum())
    assert quad == pytest.approx(expected, rel=1e-9)


def test_single_frame_raises() -> None:
    with pytest.raises(ValueError, match="no temporal neighbour"):
        _uniform_temporal_laplacian(
            1, 1, device=torch.device("cpu"), dtype=torch.float32
        )


def test_learned_laplacian_without_navigators_raises_instead_of_degrading() -> None:
    """The silent fallback this loss used to take.

    With ``learn_laplacian=True`` and no navigators it quietly built the UNIFORM
    Laplacian — turning learned-manifold smoothness into plain temporal TV, a
    different regulariser, reported under the name SToRM. The caller asked for a
    learned manifold and got something else, with no warning (pitfall #9).
    """
    loss = SToRMRegularizer(learn_laplacian=True, num_frames=4, nav_dim=8)
    x = torch.randn(1, 4, 1, 8, 8)
    with pytest.raises(ValueError, match="navigators"):
        loss(x)


def test_uniform_laplacian_is_reachable_by_declaring_it() -> None:
    """learn_laplacian=False is a CHOICE, and must still work.

    The fix removes the silent fallback, not the uniform Laplacian itself. A
    caller that declares it is asking for it on purpose.
    """
    loss = SToRMRegularizer(learn_laplacian=False, num_frames=4)
    x = torch.randn(2, 4, 1, 6, 6)
    value = loss(x)
    assert torch.isfinite(value)
    # PSD Laplacian + non-negative lambda => the penalty cannot be negative.
    assert float(value) >= 0.0


def test_constant_sequence_is_the_smoothest_possible() -> None:
    """A behavioural consequence of PSD-ness, asserted end-to-end on the loss."""
    loss = SToRMRegularizer(learn_laplacian=False, num_frames=4)
    constant = torch.ones(1, 4, 1, 5, 5)
    varying = torch.randn(1, 4, 1, 5, 5)
    assert float(loss(constant)) == pytest.approx(0.0, abs=1e-5)
    assert float(loss(varying)) > 0.0


def test_storm_is_tagged_for_the_dynamic_regime() -> None:
    """It backs mri_dynamic's LIVE claim; a lost tag must fail here."""
    from spectramr.models.losses.registry import LossRegistry

    meta = LossRegistry._loss_domains["storm"]
    assert meta["workflows"] == frozenset({Regime.DYNAMIC})


def test_trace_is_computed_without_materialising_the_p_squared_matrix() -> None:
    """Equivalence to the [B,P,P] form it replaced, as an exact identity.

    The old form built X^T L X — [B, P, P] with P = C*H*W — via two bmms and then
    read only its diagonal, allocating P^2 to use P of it. That is 4.3 GB at
    128^2 and 68.7 GB at 256^2: a guaranteed OOM on exactly the dynamic MRI this
    regulariser exists for.

    The reference below is the old expression inline, so this is an equivalence
    proof rather than the function compared against itself.
    """
    torch.manual_seed(0)
    b, t, c, h, w = 2, 6, 1, 5, 5
    x = torch.randn(b, t, c, h, w, dtype=torch.float64)
    lap = _uniform_temporal_laplacian(
        b, t, device=torch.device("cpu"), dtype=torch.float64
    )

    x_flat = x.view(b, t, -1)
    xl = torch.bmm(x_flat.transpose(1, 2), lap)  # the OLD [B,P,P] route
    reference = torch.diagonal(torch.bmm(xl, x_flat), dim1=1, dim2=2).sum(dim=1).mean()

    loss = SToRMRegularizer(learn_laplacian=False, num_frames=t)
    assert float(loss(x)) == pytest.approx(float(reference), rel=1e-12)


def test_the_trace_still_equals_the_sum_of_squared_frame_differences() -> None:
    """The physics, asserted on the loss end-to-end after the rewrite."""
    torch.manual_seed(0)
    b, t = 1, 6
    x = torch.randn(b, t, 1, 4, 4, dtype=torch.float64)
    loss = SToRMRegularizer(learn_laplacian=False, num_frames=t)
    expected = float(((x[:, 1:] - x[:, :-1]) ** 2).sum())
    assert float(loss(x)) == pytest.approx(expected, rel=1e-9)


def test_registry_call_does_not_bind_the_target_to_navigators() -> None:
    """``loss(prediction, target)`` used to hand the target image to the manifold
    encoder as navigators, which died unpacking a 5-D tensor."""
    loss = SToRMRegularizer(learn_laplacian=True, num_frames=4, nav_dim=8)
    x = torch.randn(1, 4, 1, 6, 6)
    with pytest.raises(ValueError, match="navigators"):
        loss(x, torch.randn(1, 4, 1, 6, 6))
    value = loss(x, torch.randn(1, 4, 1, 6, 6), navigators=torch.randn(1, 4, 8))
    assert torch.isfinite(value)


def test_uniform_laplacian_ignores_the_target_and_accepts_non_contiguous_input() -> None:
    loss = SToRMRegularizer(learn_laplacian=False, num_frames=4)
    x = torch.randn(1, 4, 1, 6, 6)
    x_t = x.transpose(-1, -2)
    assert not x_t.is_contiguous()
    torch.testing.assert_close(loss(x_t, torch.randn(1, 4, 1, 6, 6)), loss(x_t.contiguous()))
