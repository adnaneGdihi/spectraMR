from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    pass


# --- Beta Schedules ---
def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear schedule"""
    return torch.linspace(beta_start, beta_end, timesteps)


# --- Time Embedding ---


# --- UNet Building Blocks with Time ---
class DoubleConvWithTime(nn.Module):
    """(Convolution => GroupNorm => GELU) * 2, with time embedding."""

    def __init__(self, in_channels, out_channels, mid_channels=None, time_emb_dim=None):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            mid_channels (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.time_mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim is not None
            else None
        )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels, affine=True),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.GELU(),
        )

    def forward(self, x, t=None):
        """forward.

        Args:
            x (Any): Description.
            t (Any): Description.
        Returns:
            Any: Description.

        forward method for DoubleConvWithTime.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        h = self.double_conv(x)
        if self.time_mlp is not None and t is not None:
            # Ensure t is on the same device as the model
            t = t.to(next(self.parameters()).device)
            time_emb = self.time_mlp(t)
            h = h + time_emb.unsqueeze(-1).unsqueeze(-1)
        return h


class DownWithTime(nn.Module):
    """Downscaling with maxpool then DoubleConvWithTime."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConvWithTime(
            in_channels,
            out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x, t):
        """forward.

        Args:
            x (Any): Description.
            t (Any): Description.
        Returns:
            Any: Description.

        forward method for DownWithTime.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = self.pool(x)
        return self.conv(x, t)


class UpWithTime(nn.Module):
    """Upscaling then DoubleConvWithTime."""

    def __init__(self, in_channels, out_channels, time_emb_dim):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            time_emb_dim (Any): Description.
        """
        super().__init__()
        # Using Upsample + Conv2d is often more stable and avoids checkerboard
        # artifacts
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
        )
        self.conv = DoubleConvWithTime(
            in_channels,
            out_channels,
            time_emb_dim=time_emb_dim,
        )

    def forward(self, x1, x2, t):
        """forward.

        Args:
            x1 (Any): Description.
            x2 (Any): Description.
            t (Any): Description.
        Returns:
            Any: Description.

        forward method for UpWithTime.

        Executes PyTorch tensor operations.

        Args:
            x1 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x2 (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, t)


# --- Base Diffusion Class ---
class Diffusion:
    """Base class for DDPM-style diffusion models."""

    def __init__(self, timesteps=1000, beta_schedule="linear", device=None, betas=None):
        """__init__.

        Args:
            timesteps (Any): Description.
            beta_schedule (Any): Description.
            device (Any): Description.
            betas: Optional explicit beta sequence. When given it is used
                verbatim and ``beta_schedule`` is kept only as a label.
                A schedule NAME is not a schedule: the training-side forward
                process (``DiffusionScheduler``) and this reverse process
                implement DIFFERENT cosine formulas -- ``s=0`` there,
                Nichol-Dhariwal ``s=0.008`` here -- so binding the two by name
                alone left sampling inverting a trajectory training never ran.
                Callers that own the forward schedule pass its betas here.
        """
        self.timesteps = timesteps

        # Set device - default to CPU if not specified.
        # Auto-selecting CUDA can be dangerous in multi-GPU/DDP setups.
        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        if betas is not None:
            betas = torch.as_tensor(betas, dtype=torch.float32).detach().clone().cpu()
            if betas.ndim != 1 or betas.numel() != timesteps:
                raise ValueError(
                    "Explicit betas must be 1-D of length "
                    f"timesteps={timesteps}; got shape {tuple(betas.shape)}."
                )
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"unknown beta schedule: {beta_schedule}")

        self.betas = betas.to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        # and reconstruction x_0 = f(x_t, epsilon)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

    def _extract(self, a, t, x_shape):
        """_extract.

        Args:
            a (Any): Description.
            t (Any): Description.
            x_shape (Any): Description.
        Returns:
            Any: Description.
        """
        batch_size = t.shape[0]
        # Ensure indices are on the same device as the source tensor
        out = a.gather(-1, t.to(a.device))
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.device)

    def q_sample(self, x_start, t, noise=None):
        """q_sample.

        Args:
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
        Returns:
            Any: Description.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        # Ensure all tensors are on the same device
        x_start = x_start.to(self.device)
        t = t.to(self.device)
        noise = noise.to(self.device)

        sqrt_alphas_cumprod_t = self._extract(
            self.sqrt_alphas_cumprod,
            t,
            x_start.shape,
        )
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x_start.shape,
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t, t, noise):
        """Reconstruct x_start from x_t and noise (epsilon).

        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
        """
        x_t = x_t.to(self.device)
        t = t.to(self.device)
        noise = noise.to(self.device)

        sqrt_recip_alphas_cumprod_t = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1_alphas_cumprod_t = self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise

    @torch.no_grad()
    def p_sample_step(self, x, t, predicted_noise, t_index):
        """Perform a single reverse diffusion step given predicted noise.

        Computes mean and variance, then samples:
        x_{t-1} = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon) + sigma_t * z

        ``t_index`` is the integer timestep, not a loop counter: the noise term is dropped
        only when it is 0, the final step (#482).
        """
        # Ensure inputs are on correct device
        x = x.to(self.device)
        t = t.to(self.device)
        predicted_noise = predicted_noise.to(self.device)

        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            t,
            x.shape,
        )
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Compute the model mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean

        posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Add noise to the mean to get the sample for the previous timestep
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def ddim_step(self, x, t, t_prev, predicted_noise, eta: float = 0.0):
        """Reverse one DDIM step from ``t`` to ``t_prev``, which may skip timesteps.

        ``p_sample_step`` is a DDPM ancestral step: it reads ``betas_t`` and
        ``sqrt_recip_alphas_t``, which are only defined for ``t -> t-1``. A
        strided sampler (``validation.sampler_steps``) jumps many timesteps at
        once, so it needs the DDIM update, which is parameterised purely by the
        two cumulative alphas and is therefore exact for any ``t > t_prev``.

        Args:
            x: Noised sample at timestep ``t``.
            t: Current timesteps, shape ``(B,)``.
            t_prev: Target timesteps, shape ``(B,)``. Entries ``< 0`` mean "the
                final hop" (``alpha_bar_prev = 1``), which returns the clean
                ``x_0`` estimate.
            predicted_noise: Model epsilon-prediction at ``t``.
            eta: DDIM stochasticity. ``0.0`` (default) is deterministic; ``1.0``
                recovers the DDPM posterior variance.

        Returns:
            The sample at ``t_prev``.
        """
        x = x.to(self.device)
        t = t.to(self.device)
        t_prev = t_prev.to(self.device)
        predicted_noise = predicted_noise.to(self.device)

        alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
        # Clamp before gather so a negative sentinel cannot index out of range,
        # then overwrite those lanes with 1.0 (the alpha_bar of "no noise").
        alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev.clamp(min=0), x.shape)
        is_final = (t_prev < 0).reshape(-1, *((1,) * (x.dim() - 1))).to(alpha_bar_prev.device)
        alpha_bar_prev = torch.where(is_final, torch.ones_like(alpha_bar_prev), alpha_bar_prev)

        # x_0 estimate from the epsilon-prediction (exact when eps is the true noise).
        x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_bar_t)

        sigma = eta * torch.sqrt(
            (1.0 - alpha_bar_prev)
            / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_prev).clamp(min=0.0)
        )
        # Direction pointing to x_t; the sigma^2 mass moves into the noise term.
        dir_xt = torch.sqrt((1.0 - alpha_bar_prev - sigma**2).clamp(min=0.0)) * predicted_noise
        out = torch.sqrt(alpha_bar_prev) * x0 + dir_xt

        if eta > 0.0:
            out = out + sigma * torch.randn_like(x)
        return out

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index, cond=None, guidance_scale=7.5):
        """Denoise a single step with optional classifier-free guidance."""
        # Infer device from model parameters if possible, otherwise rely on input
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Fallback for models without parameters or empty iterators
            device = x.device

        # Ensure all input tensors are on the correct device
        x = x.to(device)
        t = t.to(device)
        if cond is not None:
            cond = cond.to(device)

        # Check if model supports conditioning
        import inspect

        forward_sig = inspect.signature(model.forward)
        supports_cond = "cond" in forward_sig.parameters
        # F-DIFFUSION-T / 2026-05-20 — diffusion p_sample passes the
        # timestep ``t`` positionally. Some configured denoising_models
        # (e.g. plain ``standard_unet``) only accept ``(self, x)``, so
        # ``model(x, t)`` blows up with the cryptic
        # ``UNet.forward() takes 2 positional arguments but 3 were given``
        # mid-validation. Detect that case here and raise loudly with
        # the actual model class + the YAML knob the user needs to flip.
        params = forward_sig.parameters
        has_t_param = (
            "t" in params
            or any(
                p.kind == inspect.Parameter.VAR_POSITIONAL
                or p.kind == inspect.Parameter.VAR_KEYWORD
                for p in params.values()
            )
            or sum(
                1
                for p in params.values()
                if p.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
                and p.name != "self"
            )
            >= 2
        )
        if not has_t_param:
            raise TypeError(
                f"Diffusion denoising model {type(model).__name__} does "
                f"not accept a timestep argument — ``forward`` signature "
                f"is {forward_sig}. Diffusion sampling is meaningless "
                f"without time conditioning. Set the denoising_model "
                f"YAML knob to a time-conditioned variant (e.g. "
                f"``diffusion_unet`` / ``diffusion_recon`` from "
                f"src/spectramr/models/generators/diffusion_variants.py) "
                f"or any UNet whose forward accepts ``(x, t, **kwargs)``."
            )

        # Predict noise using the model
        # With classifier-free guidance, combine conditional and unconditional
        if cond is not None and guidance_scale > 1.0:
            # Unconditional prediction (omit cond argument)
            uncond_pred = model(x, t)

            # Conditional prediction
            if supports_cond:
                cond_pred = model(x, t, cond=cond)
            else:
                cond_pred = uncond_pred

            predicted_noise = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
        else:
            # Standard prediction
            if supports_cond and cond is not None:
                predicted_noise = model(x, t, cond=cond)
            else:
                predicted_noise = model(x, t)

        return self.p_sample_step(x, t, predicted_noise, t_index)

    @torch.no_grad()
    def p_sample_loop(self, model, shape, cond=None, guidance_scale=7.5):
        """The full reverse diffusion process (sampling).
        Supports conditional generation with classifier-free guidance.
        """
        img = torch.randn(shape, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            # Pass guidance parameters to the single-step sampler
            img = self.p_sample(
                model,
                img,
                t,
                i,
                cond=cond,
                guidance_scale=guidance_scale,
            )
        return img

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l1"):
        """p_losses.

        Args:
            denoise_model (Any): Description.
            x_start (Any): Description.
            t (Any): Description.
            noise (Any): Description.
            loss_type (Any): Description.
        Returns:
            Any: Description.
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = denoise_model(x_noisy, t)

        # Ensure all tensors are on the same device for loss computation (model output device)
        device = predicted_noise.device
        noise = noise.to(device)

        # Lazy import: losses/__init__ imports models.diffusion modules
        # (levy, resetting) — a module-scope import here would cycle.
        from spectramr.models.losses.elementary import resolve_elementary_loss

        return resolve_elementary_loss(loss_type)(predicted_noise, noise)

    def parameters(self):
        """Return an empty generator since diffusion processes don't have
        learnable parameters.
        """
        return iter([])

    def to(self, device):
        """Move the diffusion process to the specified device."""
        self.device = device
        # Move all tensor attributes to the device
        for attr in [
            "betas",
            "alphas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
            "sqrt_recip_alphas",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
        ]:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        return self
