"""Projecting Conflicting Gradients (PCGrad) for Multi-Task Learning.

Implements gradient surgery to resolve destructive interference between
conflicting loss objectives (e.g., physics-based Bloch regression vs.
adversarial PatchGAN gradients).

When two gradient vectors point in opposing directions (negative dot product),
the conflicting component of one gradient is projected onto the normal plane
of the other, removing destructive interference while preserving constructive
contributions.

Reference:
    Yu et al., "Gradient Surgery for Multi-Task Learning," NeurIPS, 2020.
"""

from __future__ import annotations

import random
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class PCGrad:
    """Projecting Conflicting Gradients optimizer wrapper.

    Wraps any ``torch.optim.Optimizer`` and performs gradient surgery
    before each ``step()``.

    Example::

        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer = PCGrad(base_opt, num_tasks=3)

        # During training:
        losses = [loss_recon, loss_semantic, loss_perceptual]
        optimizer.pc_backward(losses)
        optimizer.step()

    Args:
        optimizer: Base optimizer wrapping model parameters.
        num_tasks: Number of task-specific losses. Must match
            the length of the loss list passed to ``pc_backward``.
        reduction: How to aggregate projected gradients across tasks.
            ``'sum'`` (default) or ``'mean'``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_tasks: int = 2,
        reduction: str = "sum",
    ) -> None:
        if reduction not in ("sum", "mean"):
            raise ValueError(f"reduction must be 'sum' or 'mean', got '{reduction}'")
        self._optimizer = optimizer
        self._num_tasks = num_tasks
        self._reduction = reduction

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    def zero_grad(self) -> None:
        self._optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self._optimizer.step()

    def state_dict(self) -> dict:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self._optimizer.load_state_dict(state_dict)

    def pc_backward(self, losses: list[Tensor]) -> None:
        """Compute PCGrad-projected gradients and store in ``.grad``.

        Replaces the standard ``loss.backward()`` call. Each loss is
        backpropagated independently; conflicting components between
        per-task gradients are projected away before aggregation.

        Args:
            losses: List of ``num_tasks`` scalar loss tensors.

        Raises:
            ValueError: If ``len(losses) != num_tasks``.
        """
        if len(losses) != self._num_tasks:
            raise ValueError(f"Expected {self._num_tasks} losses, got {len(losses)}")

        task_grads: list[list[Tensor | None]] = []
        shared_params = self._get_shared_params()

        for loss in losses:
            self._optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            grads = []
            for p in shared_params:
                grads.append(p.grad.detach().clone() if p.grad is not None else None)
            task_grads.append(grads)

        projected = self._project(task_grads)

        self._optimizer.zero_grad(set_to_none=True)
        for i, p in enumerate(shared_params):
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            for t in range(self._num_tasks):
                g = projected[t][i]
                if g is not None:
                    p.grad.add_(g)
            if self._reduction == "mean":
                p.grad.div_(self._num_tasks)

    def _project(self, task_grads: list[list[Tensor | None]]) -> list[list[Tensor | None]]:
        num_tasks = len(task_grads)
        num_params = len(task_grads[0])
        projected = [[g.clone() if g is not None else None for g in tg] for tg in task_grads]
        order = list(range(num_tasks))
        random.shuffle(order)

        for i_idx in range(num_tasks):
            i = order[i_idx]
            for j_idx in range(num_tasks):
                j = order[j_idx]
                if i == j:
                    continue
                gi_flat = self._flatten(projected[i])
                gj_flat = self._flatten(task_grads[j])
                if gi_flat is None or gj_flat is None:
                    continue
                dot = torch.dot(gi_flat, gj_flat)
                if dot < 0:
                    gj_norm_sq = torch.dot(gj_flat, gj_flat).clamp(min=1e-12)
                    coeff = dot / gj_norm_sq
                    for k in range(num_params):
                        gi_k = projected[i][k]
                        gj_k = task_grads[j][k]
                        if gi_k is not None and gj_k is not None:
                            projected[i][k] = gi_k - coeff * gj_k
        return projected

    def _flatten(self, grads: list[Tensor | None]) -> Tensor | None:
        flat = [g.flatten() for g in grads if g is not None]
        if not flat:
            return None
        return torch.cat(flat)

    def _get_shared_params(self) -> list[torch.nn.Parameter]:
        seen: set[int] = set()
        params: list[torch.nn.Parameter] = []
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                pid = id(p)
                if pid not in seen and p.requires_grad:
                    seen.add(pid)
                    params.append(p)
        return params


def project_conflicting_gradients(
    grad_primary: list[torch.Tensor],
    grad_secondary: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Project the secondary gradients to remove conflict with primary gradients.

    For each parameter, if the primary and secondary gradients conflict
    (dot product < 0), the conflicting component of the secondary gradient
    is surgically removed by projecting it onto the normal plane of the
    primary gradient.

    Args:
        grad_primary: Primary task gradients (e.g., physics/reconstruction).
            These are treated as the ground truth direction and are NOT modified.
        grad_secondary: Secondary task gradients (e.g., adversarial).
            Conflicting components are projected away.

    Returns:
        Modified secondary gradients with conflicting components removed.
        Non-conflicting gradients are returned unchanged.
    """
    projected = []
    for g_p, g_s in zip(grad_primary, grad_secondary, strict=False):
        if g_p is None or g_s is None:
            projected.append(g_s)
            continue

        # reshape, not view: autograd may return a non-contiguous gradient. The conflict
        # test stays on device (torch.where) so no parameter forces a host sync.
        g_p_flat = g_p.reshape(-1)
        g_s_flat = g_s.reshape(-1)
        dot = torch.dot(g_p_flat, g_s_flat)
        norm_sq = torch.dot(g_p_flat, g_p_flat)
        # Conflicting (dot < 0) with a non-zero primary: g_s - (g_s·g_p / ||g_p||²) g_p.
        coeff = torch.where(
            (dot < 0) & (norm_sq > 1e-12), dot / norm_sq.clamp_min(1e-12), torch.zeros_like(dot)
        )
        projected.append(g_s - coeff * g_p)

    return projected


def apply_pcgrad_to_losses(
    loss_primary: torch.Tensor,
    loss_secondary: torch.Tensor,
    parameters: list[torch.Tensor],
) -> torch.Tensor:
    """Apply PCGrad and return the combined loss with conflict-free gradients.

    Computes gradients for both losses, projects the secondary gradients
    to remove conflict, then constructs a synthetic scalar whose backward
    pass applies the corrected combined gradient.

    Args:
        loss_primary: Primary loss (physics/reconstruction). Its gradients
            define the protected direction.
        loss_secondary: Secondary loss (adversarial). Its conflicting
            gradient components are projected away.
        parameters: Model parameters to compute gradients for.

    Returns:
        Combined loss tensor. The backward pass of `loss_primary` proceeds
        normally; the backward pass of `loss_secondary` uses projected
        (conflict-free) gradients.
    """
    # Compute gradients for both losses (without updating parameters)
    grad_primary = torch.autograd.grad(
        loss_primary,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    grad_secondary = torch.autograd.grad(
        loss_secondary,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    # Project conflicting secondary gradients
    grad_projected = project_conflicting_gradients(list(grad_primary), list(grad_secondary))

    # Construct surrogate loss: the backward pass of this scalar
    # will apply the combined (primary + projected_secondary) gradients.
    # We do this by computing: sum(p * (g_primary + g_projected)) for each param
    surrogate = torch.tensor(0.0, device=loss_primary.device, requires_grad=True)
    for p, g_p, g_s_proj in zip(parameters, grad_primary, grad_projected, strict=False):
        if g_p is None:
            continue
        # ``g_s_proj`` can be None: project_conflicting_gradients passes the
        # allow_unused=True Nones through for params absent from the secondary
        # graph, so ``g_p + g_s_proj`` would raise ``tensor + None``. Treat a
        # missing projected secondary gradient as zero (only the primary applies).
        combined = g_p if g_s_proj is None else g_p + g_s_proj
        surrogate = surrogate + (p * combined.detach()).sum()

    return surrogate
