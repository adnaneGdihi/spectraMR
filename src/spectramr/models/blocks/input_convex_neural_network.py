"""Input-Convex Neural Network — the Brenier potential primitive (MICCAI B-1.5/3.9, T1 #20).

A scalar potential ``psi(x)`` CONVEX in its image input ``x`` by construction (Amos et al.
2017, "Input Convex Neural Networks"). The gradient of a convex potential is a monotone
(cyclically-monotone) map — the FORM of a Brenier/Monge optimal-transport map — which B-1.5
and B-3.9 build on. NOTE: the W2-OPTIMALITY of a Brenier map holds only for the specific
marginals its potential is the OT dual of; fitting this block by paired L1 (as the strategies
do) yields a convex-constrained monotone map, NOT a guaranteed W2-optimal map between the
field marginals. The gradient is also curl-free (it is grad of a scalar), bounding expressivity.

The map is in DISPLACEMENT form: ``T(x) = x + map_scale * grad_x psi(x)``. This corresponds
to the full Brenier potential ``phi(x) = 0.5*||x||^2 + map_scale*psi(x)`` (the ``0.5*||x||^2``
is implicit in the ``x +``); since ``phi`` is strongly convex, ``T`` is strictly monotone. The
displacement form gives a NEAR-IDENTITY init (``map_scale`` small -> ``T(x) ~ x``, a sane OT
starting point) and keeps the convexity ablation clean (it acts on ``psi`` alone, not on an
always-convex ``||x||^2`` base that would mask it).

Convexity of ``psi`` is structural, not learned:
- the ``z -> z`` (convex-path) conv weights are NON-NEGATIVE via a softplus reparameterisation
  (``W = softplus(W_raw)``) — differentiable and always >= 0, unlike a ``clamp_`` that mutates
  params under ``no_grad``;
- the activation is ``softplus`` (convex, non-decreasing, smooth — smoothness matters because
  training backprops through ``grad_x psi``, a double backward);
- the ``x -> z`` (input-skip) convs are AFFINE in ``x`` (unconstrained — affine is convex);
- mean-pool over space+channels (a non-negative combination) yields the scalar ``psi``.

``enforce_convexity=False`` drops the softplus reparam (raw, sign-free weights) so ``psi`` is a
generic CNN whose gradient is NOT a guaranteed OT map — the one-knob ablation for B-1.5/3.9.
It stays a trainable map (non-degenerate), isolating the convexity/Brenier guarantee.

``intensity_adaptation`` (``none`` | ``undamped`` | ``transfer``) controls how the map remaps
INTENSITY/CONTRAST — the residual that survives on co-registered, normalised source/target pairs:

- ``none`` (default): the original mean-pooled potential. The ``1/N`` mean damps ``grad psi`` to
  ``O(1/N)`` so ``T(x) ~ x`` is pinned at identity (the right diagnostic baseline, but it learns
  no contrast change).
- ``undamped`` (Route 1): SUM-pool the potential, so the per-pixel intensity gradient is ``O(1)``.
  A sum is still a non-negative reduction, so ``psi`` stays convex and ``T`` stays monotone. NOTE:
  a convex displacement map has ``T'(x) = 1 + map_scale*psi''(x) >= 1`` — it can only EXPAND
  intensities, never COMPRESS them.
- ``transfer`` (Route 2): compose a field-conditioned MONOTONE per-pixel intensity transfer
  ``g_b(x) = gain(b)*x + bias(b) + sum_k w_k(b)*softplus(s*(x - t_k))`` (``gain, w_k >= 0``) with
  the convex spatial map: ``T(x) = g_b(x) + map_scale*grad psi(x)``. ``diag(g_b') (PSD) +
  map_scale*Hess psi (PSD) = PSD`` keeps ``T`` a monotone Brenier map, and ``gain(b) < 1`` lifts
  the expand-only ceiling (it can COMPRESS contrast). ``g_b`` is identity at init (near-identity
  start preserved). An unknown value RAISES (pitfall #9/#15).

Magnitude-only (``in_channels=1``); ``potential`` -> ``[B]``, ``brenier_map`` -> ``[B,1,H,W]``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class InputConvexNeuralNetwork(nn.Module):
    """Convex scalar potential ``psi(x)``; ``brenier_map`` is its (displacement) OT map."""

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        kernel_size: int = 3,
        field_dim: int = 0,
        map_scale_init: float = 0.1,
        enforce_convexity: bool = True,
        intensity_adaptation: str = "none",
    ) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                f"InputConvexNeuralNetwork is magnitude-only (in=1); got {in_channels}."
            )
        if n_layers < 2:
            raise ValueError(f"n_layers must be >= 2; got {n_layers}.")
        if map_scale_init <= 0.0:
            raise ValueError(f"map_scale_init must be > 0; got {map_scale_init}.")
        # intensity_adaptation: how the (curl-free, near-identity) base map is allowed to remap
        # INTENSITY/CONTRAST. On co-registered+normalised source/7T pairs the residual is a
        # contrast change, not a geometric one, so the default mean-pool ("none") leaves the map
        # pinned at identity (grad psi ~ O(1/N)). "undamped" sum-pools the potential so the
        # intensity-space gradient is O(1); "transfer" composes a field-conditioned monotone
        # per-pixel intensity transfer. Both keep the map monotone (Brenier-valid) under convexity.
        _valid_ia = {"none", "undamped", "transfer"}
        if intensity_adaptation not in _valid_ia:  # pitfall #9/#15: raise, never silently degrade
            raise ValueError(
                f"intensity_adaptation must be one of {sorted(_valid_ia)}; got "
                f"{intensity_adaptation!r}."
            )
        self.intensity_adaptation = str(intensity_adaptation)
        if field_dim > 0 and n_layers < 3:
            # FiLM is applied inside range(n_layers-2); n_layers<3 would leave the field
            # conditioning entirely inert (a silent facade, pitfall #15/#16). Fail fast.
            raise ValueError(
                f"field_dim>0 requires n_layers>=3 (FiLM applies inside range(n_layers-2)); "
                f"got n_layers={n_layers} -> field conditioning would be inert."
            )
        self.hidden = int(hidden_channels)
        self.n_layers = int(n_layers)
        self.field_dim = int(field_dim)
        self.enforce_convexity = bool(enforce_convexity)
        self.pad = kernel_size // 2
        # Fan-in normalisation of the convex-path convs: without it the gradient grad_x psi
        # compounds MULTIPLICATIVELY through depth/width (softplus weights ~0.69 x fan-in per
        # layer) and explodes at the production config (h=64,n=4 -> |grad|~470, ~79% clamp
        # saturation — NOT the near-identity init the displacement form intends). Dividing each
        # conv by sqrt(fan_in) keeps the gradient O(1) and config-invariant.
        self._znorm = float((self.hidden * kernel_size * kernel_size) ** 0.5)

        # Input-skip (x -> hidden/out) convs: AFFINE in x, unconstrained.
        self.x_skips = nn.ModuleList(
            nn.Conv2d(in_channels, self.hidden, kernel_size, padding=self.pad)
            for _ in range(n_layers - 1)
        )
        self.x_final = nn.Conv2d(in_channels, 1, kernel_size, padding=self.pad)
        # Convex-path (z -> z) conv weights: kept non-negative via softplus(raw).
        self.z_raw = nn.ParameterList(
            nn.Parameter(torch.randn(self.hidden, self.hidden, kernel_size, kernel_size) * 0.05)
            for _ in range(n_layers - 2)
        )
        self.z_final_raw = nn.Parameter(
            torch.randn(1, self.hidden, kernel_size, kernel_size) * 0.05
        )
        # Map scale (softplus -> positive): small init => near-identity Brenier map at start.
        self.map_scale_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(float(map_scale_init))), dtype=torch.float32)
        )
        # Optional field conditioning (B-1.5): per convex-path layer (gamma>=0, beta) from the
        # field; gamma>=0 scaling + beta shift preserve convexity in x. Zero-init (inert start).
        if self.field_dim > 0:
            # One (gamma, beta) per MODULATED convex-path layer = (n_layers - 2) (the loop
            # body); the earlier (n_layers-1) sizing left slot 0 dead. Zero weight + gamma-raw
            # bias = inv_softplus(1) -> gamma=softplus=1.0, beta=0 => TRUE identity FiLM at init
            # (field-inert AND architecture-equivalent to field_dim=0; gamma>=0 keeps convexity).
            self.field_mlp = nn.Linear(self.field_dim, 2 * self.hidden * (n_layers - 2))
            nn.init.zeros_(self.field_mlp.weight)
            nn.init.zeros_(self.field_mlp.bias)
            with torch.no_grad():
                self.field_mlp.bias.view(n_layers - 2, 2, self.hidden)[:, 0, :] = math.log(
                    math.expm1(1.0)
                )

        # Route 2 ("transfer"): a field-conditioned MONOTONE per-pixel intensity transfer g_b(x)
        # composed with the convex spatial map: T(x) = g_b(x) + map_scale*grad psi(x). g_b is a
        # learned tone curve g_b(x) = gain(b)*x + bias(b) + sum_k w_k(b)*softplus(s*(x - t_k)) with
        # gain,w_k >= 0 (softplus) and s>0, so g_b is increasing (PSD diagonal Jacobian) and the
        # composite stays the gradient of a convex potential -> still a monotone Brenier map. Init
        # gain=1, bias=0, w_k~0 => g_b ~ identity (near-identity OT start preserved).
        self.transfer_k = 4
        if self.intensity_adaptation == "transfer":
            self.transfer_slope = 4.0
            self.register_buffer("transfer_knots", torch.linspace(0.0, 1.0, self.transfer_k))
            in_dim = self.field_dim if self.field_dim > 0 else 1
            self.transfer_mlp = nn.Linear(in_dim, 2 + self.transfer_k)
            nn.init.zeros_(self.transfer_mlp.weight)
            with torch.no_grad():
                self.transfer_mlp.bias.zero_()
                self.transfer_mlp.bias[0] = math.log(math.expm1(1.0))  # softplus -> gain = 1.0
                self.transfer_mlp.bias[2:] = -10.0  # softplus(-10) ~ 0 -> basis off at init

    def _zw(self, raw: torch.Tensor) -> torch.Tensor:
        """Convex-path weight: softplus(raw) (>=0) when convexity is enforced, else raw."""
        return F.softplus(raw) if self.enforce_convexity else raw

    def map_scale(self) -> torch.Tensor:
        """Positive displacement scale (softplus of the learnable raw parameter)."""
        return F.softplus(self.map_scale_raw)

    def potential(self, x: torch.Tensor, field: torch.Tensor | None = None) -> torch.Tensor:
        """Convex scalar potential ``psi(x)`` -> ``[B]`` (mean over channels + space)."""
        gammas = betas = None
        if self.field_dim > 0:
            if field is None:
                raise ValueError("InputConvexNeuralNetwork(field_dim>0) requires `field`.")
            fb = self.field_mlp(field.reshape(x.shape[0], self.field_dim).float())
            fb = fb.reshape(x.shape[0], self.n_layers - 2, 2, self.hidden)
            gammas = F.softplus(fb[:, :, 0])  # >=0 keeps convexity
            betas = fb[:, :, 1]

        z = F.softplus(self.x_skips[0](x))  # z_1 = softplus(affine(x))
        for i in range(self.n_layers - 2):
            # /_znorm (sqrt fan-in) keeps grad_x psi O(1) across width/depth (near-identity init)
            zc = F.conv2d(z, self._zw(self.z_raw[i]), padding=self.pad) / self._znorm  # non-neg
            if gammas is not None and betas is not None:
                zc = gammas[:, i][:, :, None, None] * zc + betas[:, i][:, :, None, None]
            z = F.softplus(zc + self.x_skips[i + 1](x))
        out = F.conv2d(
            z, self._zw(self.z_final_raw), padding=self.pad
        ) / self._znorm + self.x_final(x)
        flat = out.flatten(1)
        # "none"/"transfer": mean-pool (a non-negative combination; keeps convexity, bounds the
        # gradient to a near-identity base). "undamped": sum-pool so the per-pixel intensity
        # gradient is O(1) instead of O(1/N) -- still a non-negative combination, so still convex.
        if self.intensity_adaptation == "undamped":
            return flat.sum(dim=1)  # [B]
        return flat.mean(dim=1)  # [B]

    def _transfer(self, x: torch.Tensor, field: torch.Tensor | None) -> torch.Tensor:
        """Field-conditioned monotone per-pixel intensity transfer ``g_b(x)`` (Route 2 base)."""
        b = x.shape[0]
        if self.field_dim > 0:
            if field is None:
                raise ValueError(
                    "intensity_adaptation='transfer' with field_dim>0 requires `field`."
                )
            f = field.reshape(b, self.field_dim).float()
        else:
            f = x.new_ones(b, 1)
        params = self.transfer_mlp(f)  # [B, 2 + K]
        gain = F.softplus(params[:, 0]).view(b, 1, 1, 1)  # >= 0 -> monotone
        bias = params[:, 1].view(b, 1, 1, 1)
        w = F.softplus(params[:, 2:]).view(b, self.transfer_k, 1, 1, 1)  # >= 0
        knots = self.transfer_knots.view(1, self.transfer_k, 1, 1, 1)
        basis = F.softplus(
            self.transfer_slope * (x.unsqueeze(1) - knots)
        )  # [B,K,1,H,W], increasing
        return gain * x + bias + (w * basis).sum(dim=1)

    def brenier_map(self, x: torch.Tensor, field: torch.Tensor | None = None) -> torch.Tensor:
        """Displacement-form Brenier OT map ``T(x) = base(x) + map_scale * grad_x psi(x)``.

        Strictly monotone (the implicit ``0.5*||x||^2`` makes the full potential strongly
        convex). Needs an autograd graph; backprops through ``grad`` at train time. With
        ``intensity_adaptation='transfer'`` the identity base ``x`` is replaced by the monotone
        intensity transfer ``g_b(x)`` (still a gradient-of-convex base, so ``T`` stays monotone).
        """
        with torch.enable_grad():
            x_g = x.detach().requires_grad_(True)
            psi = self.potential(x_g, field)
            (grad,) = torch.autograd.grad(psi.sum(), x_g, create_graph=self.training)
        base = self._transfer(x, field) if self.intensity_adaptation == "transfer" else x
        return base + self.map_scale() * grad

    def forward(self, x: torch.Tensor, field: torch.Tensor | None = None, **_: Any) -> torch.Tensor:
        """Default forward returns the potential ``[B]``; use ``brenier_map`` for the map."""
        return self.potential(x, field)
