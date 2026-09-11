"""Unit tests for :mod:`spectramr.models.generators.domain_adaptation_generators`.

Scope is :class:`DomainAdversarialGenerator`, the one class in the module this change
touches. The module's other eight registered generators have no paired coverage
either; that is pre-existing debt, stated rather than paid by a sweep here.
"""

import torch

from spectramr.models.generators.domain_adaptation_generators import (
    DomainAdversarialGenerator,
)


def _model() -> DomainAdversarialGenerator:
    return DomainAdversarialGenerator(in_channels=2, out_channels=2, base_features=8)


class TestReconstruction:
    """The half of the model that is wired."""

    def test_forward_restores_the_input_spatial_shape(self) -> None:
        model = _model()
        out = model(torch.randn(2, 2, 32, 32))
        assert out.shape == (2, 2, 32, 32)

    def test_forward_returns_a_single_tensor(self) -> None:
        # #1959's wiring would make this a (reconstruction, domain_logits) pair, so
        # the arity is the seam the change moves and is pinned rather than assumed.
        out = _model()(torch.randn(1, 2, 16, 16))
        assert isinstance(out, torch.Tensor)


class TestTheDomainHeadIsUnwired:
    """Pins the absence #1959 reports, by observation rather than by reading the source.

    **These tests go red when #1959's wiring lands, and that is their purpose.** They
    hold the class docstring and the real behaviour together for as long as the gap
    exists. When the domain head is wired, delete this class and rewrite the docstring
    -- do not relax the assertions to keep them green.
    """

    def test_the_domain_classifier_receives_no_gradient(self) -> None:
        model = _model()
        model(torch.randn(2, 2, 32, 32)).sum().backward()

        head = dict(model.domain_classifier.named_parameters())
        assert head, "the head is constructed, so there must be parameters to observe"
        # Sorted on both sides: `named_parameters()` yields registration order, and the
        # claim is that the set is exhausted, not that it arrives in any order.
        unreached = sorted(name for name, p in head.items() if p.grad is None)
        assert unreached == sorted(head), (
            "domain_classifier took gradient from the reconstruction loss, so the head "
            f"is now on the graph: {sorted(set(head) - set(unreached))}"
        )

    def test_the_reconstruction_path_does_take_gradient(self) -> None:
        # The control. Without it the assertion above passes whenever the fixture
        # fails to run backward at all, which is the same observation for a different
        # and uninteresting reason.
        model = _model()
        model(torch.randn(2, 2, 32, 32)).sum().backward()

        reached = [name for name, p in model.reconstructor.named_parameters() if p.grad is not None]
        assert reached, "backward did not reach the reconstructor; the pin above is vacuous"

    def test_forward_never_calls_the_domain_classifier(self) -> None:
        # Observed at the call site rather than inferred from the gradient: a head that
        # ran under no_grad would also show `grad is None`.
        model = _model()
        calls: list[int] = []
        model.domain_classifier.register_forward_hook(
            lambda _module, _args, _output: calls.append(1)
        )
        model(torch.randn(1, 2, 16, 16))
        assert calls == []
