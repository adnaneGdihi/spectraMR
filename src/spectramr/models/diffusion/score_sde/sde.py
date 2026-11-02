"""Stochastic Differential Equations for Score-Based Generative Modeling.

This module implements the abstract SDE class and concrete variants:
- VESDE (Variance Exploding)
- VPSDE (Variance Preserving)
"""

from __future__ import annotations

import abc

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


class SDE(abc.ABC):
    """Abstract base class for SDEs."""

    def __init__(self, N: int = 1000):
        """Initialize SDE.

        Args:
            N: Number of discretization steps.
        """
        self.N = N

    @property
    @abc.abstractmethod
    def T(self) -> float:
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        """Compute drift and diffusion coefficients.

        dx = f(x, t)dt + g(t)dw

        Returns:
            drift: f(x, t)
            diffusion: g(t)
        """
        pass

    @abc.abstractmethod
    def marginal_prob(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        """Compute mean and std of the marginal p_t(x).

        Returns:
            mean: E[x_t | x_0]
            std: Std[x_t | x_0]
        """
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Float[Tensor, ...]:
        """Sample from the prior distribution p_T(x)."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z: Float[Tensor, "B C D H W"]) -> Float[Tensor, B]:
        """Compute log-probability of the prior."""
        pass


class VESDE(SDE):
    """Variance Exploding SDE (Score Matching/NCSN).

    dx = sqrt(d/dt sigma(t)^2) dw
    """

    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 50.0, N: int = 1000):
        """__init__.

        Args:
            sigma_min (float): Description.
            sigma_max (float): Description.
            N (int): Description.
        """
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(sigma_min), np.log(sigma_max), N))

    @property
    def T(self) -> float:
        """T.

        Returns:
            float: Description.
        """
        return 1.0

    def sigma(self, t: Tensor) -> Tensor:
        """Noise level sigma(t) = sigma_min * (sigma_max / sigma_min) ** t, same shape as ``t``."""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def sde(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        """sde.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
            t (Float[Tensor, B]): Description.
        Returns:
            tuple[Float[Tensor, 'B C D H W'], Float[Tensor, 'B C D H W']]: Description.
        """
        sigma = self.sigma(t)
        drift = torch.zeros_like(x)

        # g(t) = sigma(t) * sqrt(2 log(sigma_max/sigma_min))
        # Ensure broadcasting for t: [B] -> [B, 1, 1, 1, 1]
        diffusion = sigma * torch.sqrt(
            torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)), device=t.device)
        )

        # Broadcast diffusion to match x shape
        view_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        diffusion = diffusion.view(*view_shape).expand_as(x)

        return drift, diffusion

    def marginal_prob(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        # p_0t(x(t)|x(0)) = N(x(t); x(0), (sigma(t)^2 - sigma(0)^2)I)
        # Often simplified to N(x(t); x(0), sigma(t)^2 I) for VE-SDE
        # Note: Exact p_0t with sigma(0) check is not implemented yet. # IMPL
        # Using simplified version for now.

        # sigma(t) = sigma_min * (sigma_max/sigma_min)^t
        """marginal_prob.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
            t (Float[Tensor, B]): Description.
        Returns:
            tuple[Float[Tensor, 'B C D H W'], Float[Tensor, 'B C D H W']]: Description.
        """
        t = t.to(x.device)
        std = self.sigma(t)

        # Broadcast
        view_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        std = std.view(*view_shape).expand_as(x)

        mean = x
        return mean, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Float[Tensor, ...]:
        """prior_sampling.

        Args:
            shape (tuple[int, ...]): Description.
            device (torch.device): Description.
        Returns:
            Float[Tensor, ...]: Description.
        """
        return torch.randn(*shape, device=device) * self.sigma_max

    def prior_logp(self, z: Float[Tensor, "B C D H W"]) -> Float[Tensor, B]:
        # N(0, sigma_max^2 I)
        """prior_logp.

        Args:
            z (Float[Tensor, 'B C D H W']): Description.
        Returns:
            Float[Tensor, B]: Description.
        """
        shape = z.shape
        N = np.prod(shape[1:])

        return -N / 2.0 * np.log(2 * np.pi * self.sigma_max**2) - torch.sum(
            z**2, dim=(1, 2, 3, 4)
        ) / (2 * self.sigma_max**2)


class VPSDE(SDE):
    """Variance Preserving SDE (DDPM).

    dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0, N: int = 1000):
        """__init__.

        Args:
            beta_min (float): Description.
            beta_max (float): Description.
            N (int): Description.
        """
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1.0 - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    @property
    def T(self) -> float:
        """T.

        Returns:
            float: Description.
        """
        return 1.0

    def sde(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        # beta(t) = beta_0 + t * (beta_1 - beta_0)
        """sde.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
            t (Float[Tensor, B]): Description.
        Returns:
            tuple[Float[Tensor, 'B C D H W'], Float[Tensor, 'B C D H W']]: Description.
        """
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)

        view_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        beta_t = beta_t.view(*view_shape).expand_as(x)

        drift = -0.5 * beta_t * x
        diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    def marginal_prob(
        self, x: Float[Tensor, "B C D H W"], t: Float[Tensor, B]
    ) -> tuple[Float[Tensor, "B C D H W"], Float[Tensor, "B C D H W"]]:
        # Integ beta(s) ds = 0.5 * t^2 * (b1-b0) + t * b0
        """marginal_prob.

        Args:
            x (Float[Tensor, 'B C D H W']): Description.
            t (Float[Tensor, B]): Description.
        Returns:
            tuple[Float[Tensor, 'B C D H W'], Float[Tensor, 'B C D H W']]: Description.
        """
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0

        view_shape = [t.shape[0]] + [1] * (x.ndim - 1)
        log_mean_coeff = log_mean_coeff.view(*view_shape).expand_as(x)

        mean_coeff = torch.exp(log_mean_coeff)
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        return mean_coeff * x, std

    def prior_sampling(self, shape: tuple[int, ...], device: torch.device) -> Float[Tensor, ...]:
        """prior_sampling.

        Args:
            shape (tuple[int, ...]): Description.
            device (torch.device): Description.
        Returns:
            Float[Tensor, ...]: Description.
        """
        return torch.randn(*shape, device=device)

    def prior_logp(self, z: Float[Tensor, "B C D H W"]) -> Float[Tensor, B]:
        # N(0, I)
        """prior_logp.

        Args:
            z (Float[Tensor, 'B C D H W']): Description.
        Returns:
            Float[Tensor, B]: Description.
        """
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2.0 * np.log(2 * np.pi) - torch.sum(z**2, dim=(1, 2, 3, 4)) / 2.0
