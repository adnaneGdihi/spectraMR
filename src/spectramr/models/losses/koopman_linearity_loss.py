r"""Koopman-linearity (dynamic-mode-decomposition) residual loss for fMRI.

Koopman operator theory states that for a nonlinear dynamical system
:math:`x_{t+1} = F(x_t)` there exists a *linear* operator :math:`K` acting
on a lifted observable space :math:`\Phi`:

.. math::

    \Phi(x_{t+1}) = K\,\Phi(x_t).

This loss makes the ``koopman_fmri`` paradigm honour that mathematics rather
than degrading to a vanilla reconstruction term. Given a *temporal* sequence
(an fMRI volume series), it

1. lifts each spatial snapshot into a fixed observable space :math:`\Phi`
   using the encoder of the existing primitive
   :class:`spectramr.models.temporal.koopman_operator.KoopmanMotionFilter`,
   seeded and frozen (extended DMD on random features);
2. fits the best-fit linear Koopman operator :math:`K` to the prediction's
   *own* snapshots by least squares (the exact-DMD estimator
   :math:`K = Z_{+}\,Z_{-}^{+}`, where :math:`Z_{-} = [\Phi(x_0)\dots
   \Phi(x_{T-2})]`, :math:`Z_{+} = [\Phi(x_1)\dots\Phi(x_{T-1})]` and
   :math:`{}^{+}` is the Moore-Penrose pseudo-inverse);
3. penalizes the linear-evolution residual
   :math:`\lVert \Phi(x_{t+1}) - K\,\Phi(x_t)\rVert`.

A sequence whose dynamics are genuinely linear in :math:`\Phi` (e.g. a
constant series, or one produced by propagating a state through a fixed
operator) yields a near-zero residual; an unstructured / nonlinear series
yields a positive one. The fitted :math:`K` is detached so the loss measures
*how far the prediction is from admitting a constant linear propagator*, not
how well a free matrix can memorise it.

The fit has ``L^2`` unknowns and ``(T-1) L`` equations, so with ``T - 1 <= L``
it interpolates any series exactly and the residual is ~0 whatever the dynamics
(#2279); the loss raises in that regime instead of returning a zero. The encoder
is frozen because nothing trains a loss module's parameters, and training it
against this residual alone would be minimised by a constant :math:`\Phi`.

Reference:
    Lusch, Kutz & Brunton, "Deep learning for universal linear embeddings of
    nonlinear dynamics," *Nature Communications*, 2018.
    Schmid, "Dynamic mode decomposition of numerical and experimental data,"
    *J. Fluid Mech.*, 2010.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from spectramr.models.losses.registry import register_loss
from spectramr.models.temporal.koopman_operator import KoopmanMotionFilter

#: Every rebuild of the observable encoder draws the same weights.
_ENCODER_SEED = 0


@register_loss(
    name="koopman_linearity",
    aliases=["koopman_dmd_residual", "koopman_linearity_residual"],
    domain="image",
)
class KoopmanLinearityLoss(nn.Module):
    r"""Penalize departure from linear Koopman dynamics in observable space.

    Input layout: ``[B, T, H, W]`` where ``T`` is the temporal axis, or the
    equivalent ``[B, C, H, W]`` where the channel axis ``C`` is interpreted as
    time (fMRI frames stacked on channels). Both are 4-D and handled
    identically — axis 1 is the snapshot index. ``prediction`` and ``target``
    must share the same shape; a mismatch raises :class:`ValueError`.

    The loss is computed on ``prediction`` (the model's reconstructed series);
    ``target`` is accepted for API parity and not required by the residual.
    The series needs ``T >= latent_dim + 2`` snapshots.

    Args:
        latent_dim: Dimension of the Koopman observable space :math:`\Phi`.
        hidden_dim: Hidden width of the encoder/decoder MLPs in the primitive.
        num_layers: Number of hidden layers in the primitive's encoder.
        ridge: Tikhonov regularization added to the Gram matrix when solving
            the least-squares fit for :math:`K` (numerical stability).
        reduction: ``"mean"`` or ``"sum"`` over the residual elements.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        ridge: float = 1e-4,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.latent_dim = latent_dim
        self.ridge = ridge
        self.reduction = reduction
        # H*W is unknown until the first forward pass, so the encoder is rebuilt
        # for it then; seeding makes the rebuild the same observable every time.
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.koopman: KoopmanMotionFilter | None = None
        self._build_filter(input_dim=64)

    def _build_filter(self, input_dim: int) -> None:
        # Seed only the CPU generator the fork restores: torch.manual_seed would
        # also reseed every CUDA generator, which fork_rng(devices=[]) leaves unrestored.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(_ENCODER_SEED)
            koopman = KoopmanMotionFilter(
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                prediction_horizon=1,
            )
        self.koopman = koopman.requires_grad_(False)

    def _ensure_filter(self, input_dim: int, device: torch.device) -> None:
        assert self.koopman is not None
        if self.koopman.input_dim != input_dim:
            self._build_filter(input_dim=input_dim)
        self.koopman.to(device)

    def _fit_koopman(self, z_minus: torch.Tensor, z_plus: torch.Tensor) -> torch.Tensor:
        r"""Exact-DMD least-squares fit ``K = Z_+ Z_-^+`` (per batch element).

        Args:
            z_minus: Snapshot embeddings ``[B, T-1, L]`` (states ``x_0..x_{T-2}``).
            z_plus: Shifted embeddings ``[B, T-1, L]`` (states ``x_1..x_{T-1}``).

        Returns:
            Fitted operators ``[B, L, L]``, detached (the residual measures
            distance from a *constant* linear propagator, not a fit capacity).
        """
        # Solve K Z_-^T = Z_+^T  ==>  per batch:  K = Z_+^T (Z_-)^{+T}
        # We instead solve the normal equations with ridge for stability:
        #   K = (Z_+^T Z_-)(Z_-^T Z_- + ridge I)^{-1}
        b, _, lat = z_minus.shape
        zm_t = z_minus.transpose(1, 2)  # [B, L, T-1]
        gram = zm_t @ z_minus  # [B, L, L]
        eye = torch.eye(lat, device=gram.device, dtype=gram.dtype).expand(b, lat, lat)
        gram = gram + self.ridge * eye
        cross = zm_t @ z_plus  # [B, L, L]  == Z_-^T Z_+
        # K^T solves gram @ K^T = cross  (since (Z_+^T Z_-) = cross^T)
        k_t: torch.Tensor = torch.linalg.solve(gram, cross)  # [B, L, L]
        return k_t.transpose(1, 2).detach()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        if prediction.dim() != 4:
            raise ValueError(
                "KoopmanLinearityLoss expects a 4-D tensor [B, T, H, W] "
                "(or [B, C, H, W] with C as the time axis); got shape "
                f"{tuple(prediction.shape)}."
            )
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target must share shape; got "
                f"{tuple(prediction.shape)} vs {tuple(target.shape)}."
            )
        b, t, h, w = prediction.shape
        if t - 1 <= self.latent_dim:
            raise ValueError(
                f"Koopman linearity needs T >= latent_dim + 2 = {self.latent_dim + 2} "
                f"snapshots on axis 1; got T={t}. With fewer, the least-squares operator "
                "interpolates any series exactly and the residual is ~0 (#2279)."
            )

        # Flatten spatial dims into the per-snapshot feature vector.
        snapshots = prediction.reshape(b, t, h * w)  # [B, T, D]
        self._ensure_filter(input_dim=h * w, device=prediction.device)
        assert self.koopman is not None

        # Lift to the fixed observable space Phi via the primitive's frozen encoder.
        z = self.koopman.encode(snapshots)  # [B, T, L]
        z_minus = z[:, :-1]  # Phi(x_t),    [B, T-1, L]
        z_plus = z[:, 1:]  # Phi(x_{t+1}), [B, T-1, L]

        # Best-fit constant linear Koopman operator from the prediction's own
        # snapshots (detached): the residual then measures how linearly the
        # observables actually evolve.
        k = self._fit_koopman(z_minus, z_plus)  # [B, L, L]
        z_pred = z_minus @ k.transpose(1, 2)  # K Phi(x_t), [B, T-1, L]

        residual = (z_plus - z_pred).pow(2)
        if self.reduction == "mean":
            return residual.mean()
        return residual.sum()
