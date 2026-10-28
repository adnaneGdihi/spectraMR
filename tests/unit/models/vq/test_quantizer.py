"""VQ quantizer: the two halves of #1340.

1. EMA codebook updates must not participate in autograd. The EMA statistics
   are *state*, updated by assignment. Before the ``no_grad`` guard each
   in-place write appended to the autograd graph, so the graph grew
   monotonically for the life of the run -- an unbounded VRAM leak that
   presents as a mid-run OOM rather than as a wrong number.

2. ``vq_loss`` must be exactly ``commitment_cost * commitment_loss``. The term
   that used to sit beside it, labelled "codebook loss", was the commitment
   expression with its ``F.mse_loss`` arguments swapped -- and ``mse`` is
   symmetric, so it was the same tensor twice. The encoder was pulled at
   ``1 + commitment_cost`` = 5x the documented weight.
"""

from __future__ import annotations

import torch
from torch.nn.functional import mse_loss, one_hot

from spectramr.models.vq.quantizer import VectorQuantizer


def _quantizer() -> VectorQuantizer:
    return VectorQuantizer(num_embeddings=8, embedding_dim=4)


def test_ema_buffers_never_acquire_a_grad_fn() -> None:
    """The leak, stated as the property that was violated."""
    q = _quantizer().train()
    z = torch.randn(2, 4, 4, 4, requires_grad=True)
    for _ in range(3):
        _quantized, vq_loss, _ = q(z)
        vq_loss.backward(retain_graph=True)
    assert q.ema_count.grad_fn is None
    assert q.ema_weight.grad_fn is None
    assert not q.ema_count.requires_grad
    assert not q.ema_weight.requires_grad


def test_the_guard_is_what_keeps_the_graph_out() -> None:
    """Plant the violation: the same call without the guard DOES build a graph.

    Asserting only that the buffers are clean would still pass if some unrelated
    detail happened to keep them clean. Running the undecorated body next to the
    decorated one shows the decorator is the load-bearing part.
    """
    z = torch.randn(2, 4, 4, 4, requires_grad=True)

    guarded = _quantizer().train()
    guarded(z)
    assert guarded.ema_weight.grad_fn is None

    unguarded = _quantizer().train()
    # Mirror the module's own flattening so the planted call is the real one.
    z_flat = z.view(-1, unguarded.embedding_dim)
    distances = torch.cdist(z_flat, unguarded.embeddings)
    encodings = torch.nn.functional.one_hot(
        distances.argmin(dim=1), num_classes=8
    ).float()
    # __wrapped__ is the pre-decoration function object.
    type(unguarded)._update_codebook.__wrapped__(unguarded, z_flat, encodings)
    assert unguarded.ema_weight.grad_fn is not None, (
        "the undecorated body no longer builds a graph, so this test can no "
        "longer prove the decorator is what prevents the leak -- re-derive it"
    )


def test_straight_through_estimator_still_carries_the_gradient() -> None:
    """`no_grad` must scope the EMA update only -- not the encoder gradient."""
    q = _quantizer().train()
    z = torch.randn(2, 4, 4, 4, requires_grad=True)
    quantized, _vq_loss, _ = q(z)
    quantized.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def _terms(q: VectorQuantizer, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute the quantizer's own commitment term beside its reported loss.

    Mirrors ``forward``'s flattening and nearest-neighbour search so the
    comparison is against the real quantisation, not an approximation of it.
    """
    z_flat = z.view(-1, q.embedding_dim)
    distances = torch.sum((z_flat.unsqueeze(1) - q.embeddings.unsqueeze(0)) ** 2, dim=2)
    encodings = one_hot(distances.argmin(dim=1), q.num_embeddings).float()
    quantized = torch.matmul(encodings, q.embeddings).view(z.shape)
    return mse_loss(z, quantized.detach()), quantized


def test_vq_loss_is_exactly_the_weighted_commitment_term() -> None:
    """The reported loss carries the documented weight, not ``1 + weight``."""
    torch.manual_seed(0)
    q = _quantizer().eval()  # eval: no EMA step between the two evaluations
    z = torch.randn(6, 4, requires_grad=True)

    commitment, _quantized = _terms(q, z)
    _q, vq_loss, _i = q(z)

    torch.testing.assert_close(vq_loss, q.commitment_cost * commitment)
    assert vq_loss.item() < commitment.item(), (
        "vq_loss is at least the unweighted commitment term, so a second "
        f"copy of it is still being added: {vq_loss.item()} vs {commitment.item()}"
    )


def test_commitment_cost_scales_the_loss_linearly() -> None:
    """Doubling the weight doubles the loss -- no unweighted term hiding in it.

    This is the pin that a *value* assertion cannot give: any additive term
    that does not carry ``commitment_cost`` shows up as a ratio below 2. The
    duplicated term made this 1.5 / 1.25 = 1.2.
    """
    torch.manual_seed(0)
    z = torch.randn(6, 4)

    cheap = _quantizer().eval()
    dear = _quantizer().eval()
    dear.commitment_cost = 2 * cheap.commitment_cost
    dear.embeddings.copy_(cheap.embeddings)  # identical codebook, identical assignment

    _q1, loss_cheap, i1 = cheap(z)
    _q2, loss_dear, i2 = dear(z)

    assert torch.equal(i1, i2), "the two quantizers did not select the same entries"
    torch.testing.assert_close(loss_dear / loss_cheap, torch.tensor(2.0))


def test_no_codebook_term_is_possible_because_the_codebook_is_a_buffer() -> None:
    """Why the term is absent rather than corrected.

    ``||sg(z_e) - e||^2`` is the codebook term of van den Oord Eq. (3), and it
    is spelled ``F.mse_loss(quantized, z.detach())``. Here it is a *constant*:
    ``embeddings`` is a buffer moved by the EMA, so the expression carries no
    gradient and adding it back would change the reported number while
    teaching the codebook nothing. If this ever fails, the codebook became a
    parameter and the EMA/gradient decision must be re-made deliberately.
    """
    q = _quantizer().eval()
    z = torch.randn(6, 4, requires_grad=True)
    _commitment, quantized = _terms(q, z)

    assert not q.embeddings.requires_grad
    assert not mse_loss(quantized, z.detach()).requires_grad


def test_forward_does_no_discarded_work_on_the_step_path() -> None:
    """The per-step perplexity was computed and assigned to ``_``.

    Pinned by AST rather than by timing: a discarded expression is invisible
    to any behavioural assertion, which is why it survived. ``get_codebook_entropy``
    is the surface that owns this signal, and it is exercised beside the pin so
    the removal did not take the capability with it.
    """
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(VectorQuantizer.forward)))
    discarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_" for t in node.targets)
    ]
    assert not discarded, f"forward() assigns to `_`: {[ast.unparse(n) for n in discarded]}"

    q = _quantizer().train()
    q(torch.randn(2, 4, 4, 4))
    entropy = q.get_codebook_entropy()
    assert isinstance(entropy, float)
    assert entropy >= 0.0


def _reference_forward(q: VectorQuantizer, z: torch.Tensor):
    """The pre-fix lookup: a broadcast ``(z - e)**2`` that holds an (N, K, D) tensor."""
    z_flat = z.view(-1, q.embedding_dim)
    distances = ((z_flat.unsqueeze(1) - q.embeddings.unsqueeze(0)) ** 2).sum(2)
    indices = distances.argmin(dim=1)
    return indices, q.embeddings[indices].view(z.shape)


def test_lookup_matches_the_broadcast_formula() -> None:
    """The matmul expansion picks the same codes, and the same quantized tensor."""
    torch.manual_seed(0)
    q = VectorQuantizer(num_embeddings=64, embedding_dim=16).eval()
    q.embeddings.copy_(torch.randn(64, 16))
    z = torch.randn(4, 16, 8, 8)
    quantized, _, indices = q(z)
    ref_indices, ref_quantized = _reference_forward(q, z)
    assert torch.equal(indices, ref_indices)
    assert torch.allclose(quantized, ref_quantized.detach() + (z - z).detach(), atol=1e-6)


def test_lookup_never_materialises_the_pairwise_difference_tensor() -> None:
    """The planted violation for the 32 GiB OOM: no op may output an (N, K, D)-sized tensor."""
    from torch.utils._python_dispatch import TorchDispatchMode

    class _LargestOutput(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.largest = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for t in out if isinstance(out, (tuple, list)) else (out,):
                if isinstance(t, torch.Tensor):
                    self.largest = max(self.largest, t.numel())
            return out

    k, d = 64, 16
    q = VectorQuantizer(num_embeddings=k, embedding_dim=d).eval()
    z = torch.randn(2, d, 8, 8)
    n = z.numel() // d
    with _LargestOutput() as spy:
        q(z)
    assert spy.largest < n * k * d, "an (N, K, D) intermediate came back"


def test_training_update_still_moves_the_codebook() -> None:
    q = VectorQuantizer(num_embeddings=8, embedding_dim=4).train()
    before = q.embeddings.clone()
    q(torch.randn(2, 4, 4, 4))
    assert not torch.equal(before, q.embeddings)
    assert torch.isfinite(q.embeddings).all()
