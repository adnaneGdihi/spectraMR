r"""InfoNCE critic loss for IB-VF (Phase 2 / Idea 1).

Two registered losses:

* :class:`InfoNCECritic` — symmetric bilinear-critic InfoNCE lower
  bound. Estimates :math:`I(z_m; \theta)` from a mini-batch of
  positive pairs and ``K`` negatives drawn from other samples in the
  batch (or a MoCo-style momentum bank).
* :class:`PhaseSmoothnessComplex` — *not* InfoNCE; pulled in here only
  as a friendly alias because the original VF YAMLs already reference
  it under that name (lives in ``models/losses/regularizers.py``).

The IB-VF strategy uses two instantiations of :class:`InfoNCECritic`:

  L_IB = - beta_+ * I(z_m; theta_hat)   (maximise)
         + beta_- * I(z_m; x)            (minimise — squeeze nuisance)

The class itself returns the (negative) InfoNCE lower bound so it can
be added to the total loss with the conventional sign. The IB sign
flip (minus on the maximisation term, plus on the minimisation term)
is the strategy's responsibility, not this loss's.

Reference: van den Oord, Li, Vinyals 2018 (CPC); He et al. 2020 (MoCo).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.losses.registry import register_loss


@register_loss(name="infonce_critic", aliases=["infonce", "cpc"], domain="image")
class InfoNCECritic(nn.Module):
    r"""Bilinear-critic InfoNCE lower bound on :math:`I(z; y)`.

    Args:
        temperature: Softmax temperature ``τ``. Lower values sharpen
            the critic (harder negatives); typical range 0.05–0.5.
        critic_hidden_dim: Width of the projection MLP applied to each
            tensor before the inner-product critic. Set to 0 to use
            identity projections (raw features).
        negative_pool: ``"in_batch"`` uses the other samples in the
            batch as negatives; ``"momentum_bank"`` augments those
            with a queue of pre-encoded samples (MoCo).
        bank_size: Capacity of the momentum bank when enabled.
    """

    _bank: torch.Tensor | None
    _bank_ptr: torch.Tensor | None

    def __init__(
        self,
        temperature: float = 0.1,
        critic_hidden_dim: int = 128,
        negative_pool: Literal["in_batch", "momentum_bank"] = "in_batch",
        bank_size: int = 0,
        feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0; got {temperature}.")
        if negative_pool not in ("in_batch", "momentum_bank"):
            raise ValueError(
                f"negative_pool must be 'in_batch' or 'momentum_bank'; got {negative_pool!r}."
            )
        if negative_pool == "momentum_bank" and bank_size <= 0:
            raise ValueError(
                "negative_pool='momentum_bank' needs bank_size > 0; with an empty bank "
                "the critic would score in-batch negatives only."
            )
        self.temperature = float(temperature)
        self.negative_pool = negative_pool
        self.bank_size = bank_size

        # Optional projection heads. Identity when critic_hidden_dim == 0.
        self.proj_z = (
            nn.Identity()
            if critic_hidden_dim <= 0
            else nn.Sequential(
                nn.Linear(feature_dim or critic_hidden_dim, critic_hidden_dim),
                nn.GELU(),
                nn.Linear(critic_hidden_dim, critic_hidden_dim),
            )
            if feature_dim is not None
            else nn.LazyLinear(critic_hidden_dim)
        )
        self.proj_y = (
            nn.Identity()
            if critic_hidden_dim <= 0
            else nn.Sequential(
                nn.Linear(feature_dim or critic_hidden_dim, critic_hidden_dim),
                nn.GELU(),
                nn.Linear(critic_hidden_dim, critic_hidden_dim),
            )
            if feature_dim is not None
            else nn.LazyLinear(critic_hidden_dim)
        )

        # Momentum bank (registered as a non-persistent buffer so it
        # doesn't pollute checkpoints; rotated FIFO).
        # ``_ptr`` mirrors ``_bank_ptr`` on the host so the FIFO write never reads
        # the device buffer back (one ``.item()`` sync per step).
        self._ptr = 0
        if negative_pool == "momentum_bank":
            self.register_buffer("_bank", torch.empty(0), persistent=False)
            self.register_buffer("_bank_ptr", torch.zeros(1, dtype=torch.long), persistent=False)
        else:
            self._bank = None
            self._bank_ptr = None

    def _enqueue(self, y_proj: torch.Tensor) -> None:
        """Push the projected features into the momentum bank (FIFO)."""
        if self.bank_size <= 0 or self._bank is None or self._bank_ptr is None:
            return
        if self._bank.numel() == 0:
            self._bank = torch.zeros(
                self.bank_size,
                y_proj.shape[-1],
                device=y_proj.device,
                dtype=y_proj.dtype,
            )
        # A batch larger than the bank can only contribute its most-recent
        # ``bank_size`` rows; clamp so the ring write stays well-defined. Without
        # this a 576-row batch into a 64-row bank made ``end = (ptr + b) % 64``
        # collapse the destination slice and raised "expanded size (64) must
        # match existing size (576)" (cluster crash exp_vf_ib_infonce,
        # diagnostics 2026-06-26).
        if y_proj.shape[0] > self.bank_size:
            y_proj = y_proj[-self.bank_size :]
        b = y_proj.shape[0]
        ptr = self._ptr
        # Contiguous FIFO write with explicit split — robust to the exact-fill
        # case ``ptr + b == bank_size`` where ``(ptr + b) % bank_size`` wraps to
        # 0 and an ``[ptr:0]`` slice would silently drop the whole batch.
        first = min(b, self.bank_size - ptr)
        self._bank[ptr : ptr + first] = y_proj[:first].detach()
        remainder = b - first
        if remainder > 0:
            self._bank[:remainder] = y_proj[first:].detach()
        self._ptr = (ptr + b) % self.bank_size
        self._bank_ptr[0] = self._ptr

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        r"""Return the *negative* InfoNCE lower bound (so smaller is better).

        ``prediction`` and ``target`` are matched ``(B, D)`` feature
        tensors. The critic logits are :math:`z_i \cdot y_j / \tau`;
        positives lie on the diagonal, negatives are off-diagonal +
        the momentum bank.
        """
        if prediction.ndim != 2 or target.ndim != 2:
            raise ValueError(
                "InfoNCECritic expects (B, D) feature tensors; got "
                f"prediction {tuple(prediction.shape)}, target {tuple(target.shape)}."
            )
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("Mismatched batch sizes between prediction and target.")
        z = self.proj_z(prediction)
        y = self.proj_y(target)
        z = F.normalize(z, dim=-1)
        y = F.normalize(y, dim=-1)

        b = z.shape[0]
        # In-batch logits: (B, B); positive pairs are the diagonal.
        logits = z @ y.T / self.temperature

        # Augment with the momentum bank's negatives.
        if (
            self.negative_pool == "momentum_bank"
            and self._bank is not None
            and self._bank.numel() > 0
        ):
            # Clone the bank before the matmul: ``z @ bank.T`` saves ``bank``
            # for backward (it is needed to compute z's gradient even though the
            # bank itself carries no grad), but ``_enqueue`` below writes into
            # ``self._bank`` in place, bumping its version. Without the clone the
            # next backward raises "a variable needed for gradient computation
            # has been modified by an inplace operation" — first visible at the
            # second step, once the bank is non-empty (ib_vf smoke 20260605).
            bank = self._bank.clone()
            bank_logits = z @ bank.T / self.temperature
            logits = torch.cat([logits, bank_logits], dim=1)

        labels = torch.arange(b, device=logits.device)
        loss_z2y = F.cross_entropy(logits, labels)

        # Symmetric pass.
        logits_sym = y @ z.T / self.temperature
        loss_y2z = F.cross_entropy(logits_sym, labels)

        if self.negative_pool == "momentum_bank":
            self._enqueue(y.detach())

        return 0.5 * (loss_z2y + loss_y2z)


__all__ = ["InfoNCECritic"]
