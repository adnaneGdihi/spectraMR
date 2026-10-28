"""``DynamicOTFlow`` / ``KIDOTLoss``: one Euler pass must equal the separate passes."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spectramr.models.losses.optimal_transport_losses import DynamicOTFlow, KIDOTLoss


def _reference_terms(
    flow: DynamicOTFlow, source: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Endpoint and kinetic action from two independent explicit-Euler loops."""
    dt = 1.0 / flow.num_steps
    x = source
    for i in range(flow.num_steps):
        x = x + dt * flow.velocity_net(x, torch.tensor(i * dt))
    endpoint = x
    x, energy = source, source.new_zeros(())
    for i in range(flow.num_steps):
        v = flow.velocity_net(x, torch.tensor(i * dt))
        energy = energy + dt * (v**2).mean()
        x = x + dt * v
    return endpoint, energy


def test_dynamic_ot_single_pass_matches_the_two_pass_reference() -> None:
    torch.manual_seed(0)
    flow = DynamicOTFlow(in_channels=1, num_steps=4, hidden_dim=16)
    source = torch.randn(2, 1, 8, 8, requires_grad=True)
    target = torch.randn(2, 1, 8, 8)

    loss = flow(source, target, lambda_terminal=0.7)
    (grad,) = torch.autograd.grad(loss, source)

    endpoint, energy = _reference_terms(flow, source)
    ref = energy + 0.7 * F.mse_loss(endpoint, target)
    (ref_grad,) = torch.autograd.grad(ref, source)

    torch.testing.assert_close(loss, ref)
    torch.testing.assert_close(grad, ref_grad)
    torch.testing.assert_close(flow.integrate(source), endpoint)
    torch.testing.assert_close(flow.compute_kinetic_energy(source), energy)


def test_kidot_physics_term_uses_the_transported_endpoint() -> None:
    torch.manual_seed(0)
    op = torch.nn.Identity()
    loss_fn = KIDOTLoss(forward_operator=op, lambda_physics=0.3, lambda_terminal=0.5, num_steps=3)
    source = torch.randn(2, 1, 8, 8)
    target = torch.randn(2, 1, 8, 8)
    measurements = torch.randn(2, 1, 8, 8)

    endpoint, energy = _reference_terms(loss_fn.dynamic_ot, source)
    expected = (
        energy + 0.5 * F.mse_loss(endpoint, target) + 0.3 * F.mse_loss(endpoint, measurements)
    )
    torch.testing.assert_close(loss_fn(source, target, measurements), expected)
