"""Bloch-manifold Schrodinger bridge for ULF-to-HF translation (PR-4 / idea N-A).

This is the **I2SB** bridge -- a stochastic interpolant whose endpoints are
DATA rather than data-to-noise: ``x0`` is the ULF (ultra-low-field) source and
``x1`` is the HF (high-field) target (Liu et al. 2023, *I2SB: Image-to-Image
Schrodinger Bridge*). On top of the bridge velocity-matching objective we add
the part no published Schrodinger bridge has: a **Bloch-manifold-consistency**
penalty that constrains the bridge's predicted clean endpoint to lie on the
manifold of physically realizable spin signals.

Bridge objective (inherited from :class:`StochasticInterpolantsStrategy`):

    ``i_t = (1-t) x0 + t x1 + sigma(t) z``,   ``z ~ N(0, I)``,
    target velocity ``b(t) = (x1 - x0) + sigma'(t) z``,
    ``L_vel = E||b_theta(i_t, t) - b(t)||^2``.

Bloch-manifold-consistency penalty:

    The bridge's predicted clean endpoint is estimated as
    ``x1_hat = x0 + b_theta(i_t, t)`` (the velocity is, in expectation,
    ``x1 - x0`` plus a zero-mean stochastic term, so ``x0 + b_theta`` is an
    estimator of ``x1``). The channel axis is treated as the relaxation
    triplet ``(M0, T1, T2)``; we measure how far ``x1_hat`` departs from the
    Bloch relaxation manifold ``M = {M0>0, T1>0, T2>0, T2<=T1}`` by the
    squared **Fisher-metric geodesic distance** between ``x1_hat`` and its
    projection onto ``M``:

        ``L_bloch = E[ d_M(x1_hat, Proj_M x1_hat)^2 ]``,

    where ``d_M`` is :meth:`BlochRelaxationManifold.geodesic_distance` (the
    Fisher-information distance induced by the SPGR signal model) and
    ``Proj_M`` is the manifold's reflection-to-interior operator. Because the
    distance is taken in the Fisher metric -- not a plain Euclidean ``L2`` --
    the penalty weights deviations by how distinguishable they are under the
    acquisition, which is the physically meaningful notion of "off-manifold".

Total objective:

    ``L = L_vel + lambda_bloch * L_bloch``.

**Trivial limit.** At ``lambda_bloch = 0`` the manifold term drops out and the
strategy reduces exactly to the plain I2SB bridge (``L = L_vel``); the
``loss_bloch_manifold`` entry is still reported (as ``0``) for provenance.

When ``lambda_bloch > 0`` the predicted endpoint MUST carry exactly three
channels ``(M0, T1, T2)`` or the manifold term is undefined. A mismatched
channel count is a **configuration error that raises** (it is the headline
novelty of this arm — silently zeroing it would make the run look trained
while the advertised regularizer never fired; pitfall #9/#10). Set
``lambda_bloch = 0`` to genuinely disable the penalty (the trivial I2SB
limit), in which case ``loss_bloch_manifold`` is reported as ``0`` and the
channel count is unconstrained.

``lambda_bloch`` and ``bloch_cache_resolution`` are constructor knobs (no new
YAML schema key); ``training_mode`` selection is enough to route to this
strategy.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import torch
import torch.nn.functional as torch_functional

from spectramr.data.batch_types import read_batch_field
from spectramr.infrastructure.physics.manifolds import BlochRelaxationManifold

from .stochastic_interpolants_strategy import StochasticInterpolantsStrategy

# Default tabulated-metric resolution for the Bloch Fisher manifold. > 0 routes
# ``metric_tensor`` through trilinear interpolation of a pre-built grid instead
# of a per-voxel autograd loop (the headline penalty fires once per training
# step over B·H·W voxels, so the loop is the dominant cost). 16³ ≈ 4096 Bloch
# evals at construction; queries are then a handful of gathers + matmuls.
_DEFAULT_BLOCH_CACHE_RESOLUTION = 16


class BlochSchrodingerBridgeStrategy(StochasticInterpolantsStrategy):
    """I2SB data↔data bridge with a Bloch-manifold-consistency penalty.

    Args:
        env: Training environment (provides ``generator`` and ``device``).
        lambda_bloch: Weight on the Bloch-manifold-consistency penalty.
            ``0.0`` recovers the plain I2SB bridge. Default ``0.1``.
        bloch_cache_resolution: Resolution of the tabulated Fisher-metric grid
            on the Bloch manifold. ``> 0`` (default) replaces the per-voxel
            autograd loop in :meth:`BlochRelaxationManifold.metric_tensor` with
            a one-time grid build + trilinear interpolation. ``0`` forces the
            exact per-point autograd path (slow; for parity testing only).
            Must be a non-negative int — anything else **raises** (pitfall #15).
        **kwargs: Forwarded to :class:`StochasticInterpolantsStrategy`.
    """

    #: Loss ownership (issue #1918). mse_loss(pred, true_velocity) regresses the bridge DRIFT, not the target
    #: image; the hook routes to neither the parent nor the computer.
    inline_losses: ClassVar[frozenset[str]] = frozenset()
    folds_image_losses: ClassVar[bool] = False

    def __init__(
        self,
        env: Any,
        *,
        lambda_bloch: float = 0.1,
        bloch_cache_resolution: int = _DEFAULT_BLOCH_CACHE_RESOLUTION,
        **kwargs: Any,
    ) -> None:
        super().__init__(env=env, **kwargs)
        # The typed training.schrodinger_bridge block (when present) is the
        # canonical source — the strategy factory forwards services only, so the
        # constructor kwargs were unreachable from YAML (pitfall #15). Config
        # overrides the kwarg defaults; the validation below still runs.
        _sb_cfg = getattr(self.config.training, "schrodinger_bridge", None)
        if _sb_cfg is not None:
            lambda_bloch = _sb_cfg.lambda_bloch
            bloch_cache_resolution = _sb_cfg.bloch_cache_resolution
        self.lambda_bloch = float(lambda_bloch)
        # Wire + validate + stamp the cache-resolution knob (pitfall #15): an
        # unread/illegal value must fail at startup, never silently degrade.
        if not isinstance(bloch_cache_resolution, int) or bloch_cache_resolution < 0:
            raise ValueError(
                "bloch_cache_resolution must be a non-negative int, got "
                f"{bloch_cache_resolution!r}."
            )
        self.bloch_cache_resolution = int(bloch_cache_resolution)
        # Fisher-information manifold over (M0, T1, T2): the physical-realizability
        # constraint that distinguishes this bridge from a vanilla I2SB. Building
        # with cache_resolution > 0 vectorizes the training-time metric query
        # (no per-voxel autograd loop on the hot path).
        self.manifold = BlochRelaxationManifold(cache_resolution=self.bloch_cache_resolution)

    def _bloch_manifold_penalty(self, x1_hat: torch.Tensor) -> torch.Tensor:
        """Squared Fisher-geodesic distance of ``x1_hat`` from the manifold.

        ``x1_hat`` is ``[B, C, H, W]``. The Bloch-manifold penalty is the
        *headline novelty* of this arm and is only defined when ``C == 3`` (the
        ``(M0, T1, T2)`` triplet). When ``lambda_bloch > 0`` a mismatched channel
        count is a configuration error, NOT something to silently zero: emitting
        a 0 would make the advertised regularizer a no-op while the run still
        looks trained (pitfall #9/#10). We therefore **raise** so the audit /
        smoke pass catches it at startup. Set ``lambda_bloch = 0`` (handled by
        the caller before this method is reached) to genuinely disable it.

        The departure is measured as ``d_M(p, Proj_M p)^2`` averaged over
        voxels, where ``Proj_M`` is :meth:`BlochRelaxationManifold._reflect_to_manifold`
        and ``d_M`` is :meth:`BlochRelaxationManifold.geodesic_distance`.
        """
        if x1_hat.dim() != 4 or x1_hat.shape[1] != 3:
            raise ValueError(
                "BlochSchrodingerBridgeStrategy.lambda_bloch > 0 requires the "
                "generator to emit a 4-D (B, 3, H, W) parameter map (M0, T1, T2) "
                "so the Bloch Fisher penalty is genuinely exercised, but got "
                f"shape {tuple(x1_hat.shape)}. Either set out_channels=3 in the "
                "model config or set lambda_bloch=0.0 to disable the penalty."
            )
        # [B, 3, H, W] -> [N, 3] triplets along the channel axis.
        pts = x1_hat.permute(0, 2, 3, 1).reshape(-1, 3)
        # Projection onto the manifold interior (reflection trick).
        proj = self.manifold._reflect_to_manifold(pts)
        # Genuine manifold call: Fisher-metric geodesic distance, NOT plain L2.
        dist = self.manifold.geodesic_distance(pts, proj)  # [N]
        return (dist**2).mean()

    def _compute_losses_impl(
        self,
        input_batch: Any = None,
        target_batch: Any = None,
        epoch: int = 0,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        batch = self._resolve_legacy_batch(input_batch, kwargs)
        gen = getattr(self.env, "generator", None)
        # ``read_batch_field``, not ``isinstance(batch, dict)``: the loop delivers a
        # ``TrainingBatch`` dataclass, so the isinstance leg is False and the
        # ``else`` handed the WHOLE BATCH OBJECT on as if it were a tensor.
        target = read_batch_field(batch, "target")
        if gen is None or target is None:
            return {"loss_total": torch.tensor(0.0, device=self.device)}

        x1 = target
        x0 = read_batch_field(batch, "source")
        if x0 is None:
            x0 = torch.randn_like(x1)

        b = x1.shape[0]
        t = torch.rand(b, device=x1.device).view(b, *([1] * (x1.dim() - 1)))
        z = torch.randn_like(x1)
        sigma_t = self._sigma(t)
        sigma_dot = self._sigma_dot(t)

        i_t = (1 - t) * x0 + t * x1 + sigma_t * z
        true_velocity = (x1 - x0) + sigma_dot * z

        # Dispatch via signature introspection (inherited from
        # FlowMatchingStrategy); never except-TypeError to branch, so a genuine
        # in-forward TypeError propagates instead of being silently retried.
        if self._generator_accepts_time(gen):
            pred = gen(i_t, t.flatten())
        else:
            pred = gen(i_t)

        velocity_loss = torch_functional.mse_loss(pred, true_velocity)

        # Predicted clean endpoint estimator: x0 + b_theta approx x1 (drift part).
        x1_hat = x0 + pred
        if self.lambda_bloch != 0.0:
            bloch_loss = self._bloch_manifold_penalty(x1_hat)
        else:
            # Trivial I2SB limit: skip the (potentially expensive) manifold
            # back-prop entirely, but still report the key as zero.
            bloch_loss = x1.new_zeros(())

        total = velocity_loss + self.lambda_bloch * bloch_loss
        return {
            "loss_total": total,
            "loss_sb_velocity": velocity_loss.detach(),
            "loss_bloch_manifold": bloch_loss.detach(),
        }

    @torch.no_grad()
    def sample(
        self,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x0: torch.Tensor,
        n_steps: int = 50,
        stochastic: bool = False,
        epsilon: float = 0.1,
    ) -> torch.Tensor:
        """ULF→HF inference: Euler-integrate the bridge ODE from ``x0`` (ULF).

        Inherits the stochastic-interpolant integrator: ``dx/dt = b(t, x)``
        from ``t=0`` (source ``x0``) to ``t=1`` (HF estimate). ``stochastic``
        adds the bridge diffusion term. The output has the same shape as
        ``x0``.
        """
        return super().sample(
            velocity_fn,
            x0,
            n_steps=n_steps,
            stochastic=stochastic,
            epsilon=epsilon,
        )


__all__ = ["BlochSchrodingerBridgeStrategy"]
