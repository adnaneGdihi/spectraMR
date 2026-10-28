"""LAMB — Layer-wise Adaptive Moments (You et al., 2019).

Adam's per-parameter update, rescaled by LARS's layer-wise trust ratio. Adam
supplies the direction; the trust ratio supplies the per-layer step size, which
is what makes very large batches trainable.

Like ``lars``, ``lamb`` sat in the ``OptimizerType`` enum with no implementation
anywhere and an xfail describing it as "needs apex / custom".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import overload

import torch
from torch.optim import Optimizer
from torch.optim.optimizer import ParamsT

from spectramr.infrastructure.training.optimizer_registry import register_optimizer

__all__ = ["LAMB"]

#: Denominator floor for the on-device trust ratio. ``torch.where`` evaluates
#: both branches, so the division must stay finite even where its result is
#: discarded -- a nan there survives the selection.
_TINY = 1e-38


@register_optimizer("lamb")
class LAMB(Optimizer):
    """Adam with a layer-wise trust ratio.

    Args:
        params: Parameters or param groups.
        lr: Base learning rate.
        betas: Adam moment coefficients.
        eps: Adam numerical-stability epsilon.
        weight_decay: **Decoupled** (AdamW-style): added to the normalised update
            rather than to the raw gradient. The paper's formulation is decoupled,
            and coupling it here would silently make the trust ratio a function of
            the decay strength.
        clamp_value: Upper clamp on ``phi(||w||)``. The paper leaves ``phi``
            unspecified; the reference implementation uses identity-with-clamp,
            and an unclamped ratio makes an early large-weight layer take an
            enormous step.
        adam: Disable the trust ratio entirely (ratio fixed at 1), which reduces
            LAMB to AdamW. Useful as an exact within-optimizer ablation for
            "was it the trust ratio?" without swapping optimizer_type and
            perturbing everything else.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        clamp_value: float = 10.0,
        adam: bool = False,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if clamp_value < 0.0:
            raise ValueError(f"Invalid clamp_value: {clamp_value}")

        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "clamp_value": clamp_value,
                "adam": adam,
            },
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss: float | None = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.is_sparse:
                        raise RuntimeError("LAMB does not support sparse gradients")
                    # A complex parameter is optimised as its (real, imag) pair, as
                    # torch.optim does: addcmul_(grad, grad) on a complex tensor is
                    # g**2, not |g|**2, and moved a purely imaginary gradient's real part.
                    param = torch.view_as_real(p) if p.is_complex() else p
                    if grad.is_complex():
                        grad = torch.view_as_real(grad)

                    state = self.state[p]
                    if not state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(param)
                        state["exp_avg_sq"] = torch.zeros_like(param)

                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    state["step"] += 1

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    # Bias-correct the moments themselves (You et al., Alg. 2). Folded
                    # into the step size instead, sqrt(bc2)/bc1 survives the trust
                    # ratio, which rescales only the update's norm, as an implicit
                    # warmup the method does not have (0.32x the step at t=1).
                    bias_c1 = 1 - beta1 ** state["step"]
                    bias_c2 = 1 - beta2 ** state["step"]
                    update = (exp_avg / bias_c1) / (exp_avg_sq / bias_c2).sqrt().add(group["eps"])
                    if group["weight_decay"] != 0:
                        update = update.add(param, alpha=group["weight_decay"])

                    if group["adam"]:
                        param.add_(update, alpha=-group["lr"])
                        continue

                    w_norm = torch.norm(param).clamp(0, group["clamp_value"])
                    u_norm = torch.norm(update)
                    # Either norm being zero leaves no meaningful ratio; 1.0 keeps
                    # the step on plain Adam rather than emitting nan.
                    #
                    # Kept ON DEVICE. This was
                    #     1.0 if (w_norm == 0 or u_norm == 0) else (w_norm/u_norm).item()
                    # which is three host round-trips per parameter tensor per step
                    # -- two tensor-to-bool conversions plus the .item() -- inside the
                    # training loop, against non-negotiable #9. Measured on CUDA over
                    # 80 parameter tensors that made a LAMB step 19.1x an AdamW one.
                    #
                    # clamp_min on the denominator keeps torch.where's discarded
                    # branch finite: where evaluates BOTH sides, so an unguarded
                    # 0/0 would produce a nan that the selection cannot remove.
                    ratio = w_norm / u_norm.clamp_min(_TINY)
                    trust = torch.where((w_norm > 0) & (u_norm > 0), ratio, torch.ones_like(ratio))
                    param.sub_(update * (group["lr"] * trust))

        return loss
