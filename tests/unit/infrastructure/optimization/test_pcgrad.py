"""Unit tests for PCGrad gradient surgery.

Verifies that gradient projection correctly resolves destructive
interference between conflicting loss gradients.
"""

import torch

from spectramr.infrastructure.optimization.pcgrad import (
    apply_pcgrad_to_losses,
    project_conflicting_gradients,
)


class TestPCGrad:
    """Tests for Projecting Conflicting Gradients."""

    def test_conflicting_gradients_projected(self):
        """Conflicting gradients (dot < 0) should be projected to orthogonal."""
        # Two opposing 1D gradients
        g_primary = [torch.tensor([1.0, 0.0, 0.0])]
        g_secondary = [torch.tensor([-1.0, 0.5, 0.0])]

        projected = project_conflicting_gradients(g_primary, g_secondary)

        # After projection, the conflict component should be removed
        dot_after = torch.dot(g_primary[0].view(-1), projected[0].view(-1))
        assert (
            dot_after >= -1e-6
        ), f"Projected gradient should not conflict: dot={dot_after}"

    def test_non_conflicting_gradients_unchanged(self):
        """Non-conflicting gradients (dot >= 0) should be unchanged."""
        g_primary = [torch.tensor([1.0, 0.0])]
        g_secondary = [torch.tensor([0.5, 1.0])]

        projected = project_conflicting_gradients(g_primary, g_secondary)

        assert torch.allclose(projected[0], g_secondary[0])

    def test_orthogonal_gradients_unchanged(self):
        """Perfectly orthogonal gradients should be unchanged."""
        g_primary = [torch.tensor([1.0, 0.0])]
        g_secondary = [torch.tensor([0.0, 1.0])]

        projected = project_conflicting_gradients(g_primary, g_secondary)

        assert torch.allclose(projected[0], g_secondary[0])

    def test_none_gradients_handled(self):
        """None gradients (unused params) should be passed through."""
        g_primary = [None, torch.tensor([1.0])]
        g_secondary = [torch.tensor([-1.0]), None]

        projected = project_conflicting_gradients(g_primary, g_secondary)

        assert projected[0] is g_secondary[0]
        assert projected[1] is None

    def test_multiple_parameters(self):
        """Should handle multiple parameter gradients independently."""
        g_p = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        g_s = [torch.tensor([-0.5, 0.3]), torch.tensor([0.5, -0.5])]

        projected = project_conflicting_gradients(g_p, g_s)

        assert len(projected) == 2
        # First param: conflicting (dot = -0.5 < 0)
        dot0 = torch.dot(g_p[0], projected[0])
        assert dot0 >= -1e-6
        # Second param: conflicting (dot = -0.5 < 0)
        dot1 = torch.dot(g_p[1], projected[1])
        assert dot1 >= -1e-6

    def test_zero_primary_gradient(self):
        """Zero primary gradient should leave secondary unchanged."""
        g_primary = [torch.tensor([0.0, 0.0])]
        g_secondary = [torch.tensor([-1.0, 0.5])]

        projected = project_conflicting_gradients(g_primary, g_secondary)

        assert torch.allclose(projected[0], g_secondary[0])


    def test_non_contiguous_gradients_are_projected(self):
        """autograd.grad can return a non-contiguous gradient (e.g. through a transpose).

        ``view(-1)`` raised on it and killed physics 50 and 51 at step 1 once their
        discriminator was built (sabine job 17886489).
        """
        g_p = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]).t()
        g_s = -torch.tensor([[1.0, 0.5], [0.0, 1.0], [1.0, 0.0]]).t()
        assert not g_p.is_contiguous() and not g_s.is_contiguous()
        (out,) = project_conflicting_gradients([g_p], [g_s])
        assert out.shape == g_s.shape
        assert torch.dot(out.reshape(-1), g_p.reshape(-1)).abs() < 1e-5

    def test_projection_never_syncs_to_the_host(self, monkeypatch):
        """The conflict test is a tensor op: no ``bool(tensor)`` per parameter per step."""

        def no_sync(self):
            raise AssertionError("host sync in project_conflicting_gradients")

        g_p = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0])]
        g_s = [torch.tensor([-1.0, 1.0]), torch.tensor([1.0, 1.0])]
        monkeypatch.setattr(torch.Tensor, "__bool__", no_sync)
        out = project_conflicting_gradients(g_p, g_s)
        monkeypatch.undo()
        assert torch.allclose(out[0], torch.tensor([0.0, 1.0]))
        assert torch.allclose(out[1], torch.tensor([1.0, 1.0]))  # zero primary: unchanged


class TestApplyPCGradToLosses:
    """apply_pcgrad_to_losses must not crash when the two losses touch
    different parameter sets (allow_unused=True yields None secondary grads)."""

    def test_disjoint_parameter_sets_do_not_crash(self):
        # p_a feeds only the primary loss; p_b feeds only the secondary loss.
        # For p_b the primary grad is None; for p_a the projected secondary grad
        # is None. The old ``(g_p + g_s_proj)`` raised ``tensor + None``.
        p_a = torch.nn.Parameter(torch.randn(3))
        p_b = torch.nn.Parameter(torch.randn(3))
        loss_primary = (p_a**2).sum()
        loss_secondary = (p_b**2).sum()

        surrogate = apply_pcgrad_to_losses(
            loss_primary, loss_secondary, [p_a, p_b]
        )
        # A backward through the surrogate must succeed and populate p_a's grad.
        surrogate.backward()
        assert p_a.grad is not None
        assert torch.isfinite(surrogate)
