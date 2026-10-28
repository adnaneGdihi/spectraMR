"""Cold MRI K-Space Scheduler.
============================

Architect-Grade Scheduler for Cold Diffusion MRI Reconstruction.
Implements Algorithm 2 from Bansal et al. (2022) with Physics-Informed Degradation.
Adapted to use KSpaceAccelerator for mask generation.
"""

import numpy as np
import torch

from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c
from spectramr.infrastructure.physics.sampling import KSpaceAccelerator


class ColdMRIKSpaceScheduler:
    """
    Scheduler for Cold Diffusion MRI Reconstruction.
    Implements Algorithm 2 from Bansal et al. (2022) with Physics-Informed Degradation.
    """

    def __init__(
        self,
        num_train_timesteps: int = 50,
        prediction_type: str = "sample",
        min_mask_density: float = 0.10,  # 10% sampling at t=T
        max_mask_density: float = 1.0,  # 100% sampling at t=0
        accelerator: KSpaceAccelerator | None = None,
        strategy: str = "linear",  # linear, cosine
    ):
        """__init__.

        Args:
            num_train_timesteps (int): Description.
            prediction_type (str): Description.
            min_mask_density (float): Description.
            max_mask_density (float): Description.
            accelerator (Optional[KSpaceAccelerator]): Description.
            strategy (str): Description.
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.min_density = min_mask_density
        self.max_density = max_mask_density
        self.accelerator = accelerator

        # 1. The Schedule: Linear ramp of Mask Density
        # t=0 (Clean) -> Density=1.0
        # t=T (Degraded) -> Density=min_density
        self.timesteps = torch.flip(torch.arange(0, num_train_timesteps), [0])

        if strategy == "linear":
            self.densities = torch.linspace(max_mask_density, min_mask_density, num_train_timesteps)
        elif strategy == "cosine":
            t = torch.linspace(0, 1, num_train_timesteps)
            # Dense at 0 (1.0), Sparse at 1 (min)
            # Cosine decay: 0.5 * (1 + cos(t * pi)) maps 0->1 to 1->0
            # We want map 0->1 to max->min
            cosine_factor = 0.5 * (1 + torch.cos(t * torch.pi))  # 1.0 -> 0.0
            self.densities = (
                min_mask_density + (max_mask_density - min_mask_density) * cosine_factor
            )
        else:
            # Default linear
            self.densities = torch.linspace(max_mask_density, min_mask_density, num_train_timesteps)

    def set_timesteps(self, num_inference_steps: int, device: str | torch.device = None):
        """
        Sets the discrete timesteps used for the diffusion chain.
        Supports fewer steps than training (strided sampling).
        """
        self.num_inference_steps = num_inference_steps

        # [FIX] Improved scheduling using linspace to cover full range (T-1 to 0)
        # We want 'num_inference_steps' transitions.
        # Ideally: t_start -> ... -> 0
        # If we include 0 in the list, we burn one step doing Identity 0->0.
        # So we generate steps + 1 points, and discard the last one (0).

        steps = np.linspace(0, self.num_train_timesteps - 1, num_inference_steps + 1)
        # Round and reverse: [999, ..., 20, 0]
        timesteps = np.round(steps).astype(np.int64)[::-1]

        # Keep only the top 'num_inference_steps' (Discard 0 at the end)
        # Result: [999, ..., 20] (approx)
        # This ensures the first step starts at max noise, and the last step targets 0.
        timesteps = timesteps[:-1]

        self.timesteps = torch.from_numpy(timesteps.copy()).to(device)

    def get_density(self, timestep):
        """Retrieves the sampling density for a specific timestep."""
        # Map timestep to array index
        # If timestep matches exactly
        if (self.timesteps == timestep).any():
            step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
            # We need to map this index to the original density array which has num_train_timesteps
            # Original linear map: 0 -> max, T -> min
            # We can just interpolate based on t value
            t_norm = timestep / self.num_train_timesteps
            # t=0 -> 1.0 (index 0 of densities if reversed? No, densities created linspace(max, min, T))
            # Wait, densities[0] is max (t=0 logic?).
            # In __init__: linspace(max, min).
            # t=0 should be max density?
            # User code: timesteps = arange(0, T)[::-1] (T-1, ..., 0)
            # densities = linspace(max, min, T)
            # So dens[0] corresponds to timesteps[0] (which is T-1, high t, degraded).
            # At t=T-1 (start of inference), we want MIN density.
            # At t=0 (end of inference), we want MAX density.

            # User code check:
            # self.timesteps = torch.arange(0, num_train_timesteps)[::-1] -> [49, 48, ..., 0]
            # self.densities = torch.linspace(max, min, num_train_timesteps) -> [1.0, ..., 0.1]
            # So index 0: t=49, dens=1.0.
            # Wait. t=49 (Start of diffusion) should be DEGRADED (min density).
            # t=0 (End) should be CLEAN (max density).

            # The user's blueprint says:
            # t=0 (Clean) -> Density=1.0
            # t=T (Degraded) -> Density=0.1
            # timesteps array: [49, ..., 0]
            # If densities is [1.0 ... 0.1], then at index 0 (t=49), density is 1.0 (Clear).
            # This seems INVERTED.
            # Standard diffusion: t=T is noisy (degraded). t=0 is clean.

            # Let's fix the density array to match t=T -> Min, t=0 -> Max.
            # If timesteps are [49, ..., 0], densities should be [0.1, ..., 1.0].
            pass

        # Robust implementation: direct calculation from t
        # t goes 0..num_train_timesteps
        t_val = timestep if isinstance(timestep, int) else timestep.item()
        progress = t_val / self.num_train_timesteps  # 0.0 to 1.0

        # t=0 -> progress=0 -> Density=Max
        # t=T -> progress=1 -> Density=Min
        # Typical diffusion: t=T (High t) is Noisy.
        # So at t=T (progress ~ 1), density should be MIN.
        # At t=0 (progress 0), density should be MAX.

        density = self.max_density - progress * (self.max_density - self.min_density)
        return density

    def degrade(self, image, timestep):
        """
        The Physics Operator D(x, t).
        Simulates k-space undersampling at the density defined by 'timestep'.
        """
        t_val = timestep if isinstance(timestep, int) else timestep.item()

        if t_val == 0:
            return image  # No degradation at t=0

        # Use Accelerator if valid, else fallback to naive
        current_density = self.get_density(timestep)

        # 1. FFT - Always use physics module for consistency
        # Image input: (B, C, H, W). Assuming C=2 (Real/Imag) or Complex.
        if image.shape[1] == 2:
            x_complex = torch.complex(image[:, 0], image[:, 1])  # (B, H, W)
        else:
            x_complex = image

        k_space = fft2c(x_complex)  # Use physics module for proper centering

        # 2. Generate Mask
        if self.accelerator:
            # Accelerator normally takes 'timestep' (0..1000) and maps to mask internally.
            # But here we calculated explicit density.
            # We can ask accelerator to generate mask for this 't' if it follows the same schedule.

            # Note: Accelerator usually calculates density from t internally.
            # We trust accelerator's mapping if we pass t.
            mask = self.accelerator.get_acceleration_mask(
                k_space.shape[1:], t_val, device=image.device
            )
            # ``k_space`` is (B, H, W) and ``k_space.shape[1:]`` declares one
            # channel, so the accelerator's contractual return is (1, H, W) --
            # already broadcastable over the batch. The previous ``unsqueeze(0)``
            # made it (1, 1, H, W), which broadcasts (B, H, W) up to (1, B, H, W)
            # and silently promotes the batch axis into a channel axis downstream.
            # It only escaped notice because the trajectory families used to return
            # a rank-2 mask here and skipped the branch entirely.
            if mask.ndim != 3:
                msg = (
                    f"accelerator returned rank-{mask.ndim} mask {tuple(mask.shape)}; "
                    f"expected (channels, height, width)"
                )
                raise ValueError(msg)

        else:
            # Naive fallback from blueprint
            B, H, W = k_space.shape
            num_lines = int(W * current_density)
            mask = torch.zeros_like(k_space)
            center_fraction = 0.04
            center_lines = int(W * center_fraction)
            center_start = (W - center_lines) // 2

            # Center
            mask[:, :, center_start : center_start + center_lines] = 1

            # Random
            # Create random mask per batch item? Or shared?
            # Blueprint used shared.
            # Ideally consistent.
            # Simplified:
            mask_lines = torch.randperm(W, device=image.device)[:num_lines]
            mask[:, :, mask_lines] = 1

        # 3. Apply
        k_space_undersampled = k_space * mask

        # 4. IFFT - Always use physics module for consistency
        x_degraded_complex = ifft2c(k_space_undersampled)

        if image.shape[1] == 2:
            x_degraded = torch.stack([x_degraded_complex.real, x_degraded_complex.imag], dim=1)
        else:
            x_degraded = x_degraded_complex

        return x_degraded

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
    ):
        """
        The Core Algorithm 2 Step.
        Args:
            model_output: The predicted clean image x_0 (from the UNet).
            timestep: Current step t.
            sample: The current degraded image x_t.
        """
        if self.prediction_type != "sample":
            raise ValueError(
                f"Cold Diffusion requires prediction_type='sample', got {self.prediction_type}"
            )

        # 1. Prediction: The model predicts x_0 directly
        pred_original_sample = model_output

        # 2. Calculate previous timestep (t-1)
        # Handle strided sampling: find the previous t in self.timesteps
        # timestep is the current t (e.g. 1000). We need t_prev (e.g. 980).

        # If set_timesteps was called, self.timesteps contains the schedule (e.g. 980, 960, ...)
        # Find index of current timestep
        try:
            step_index = (self.timesteps == timestep).nonzero(as_tuple=True)[0].item()
            if step_index + 1 < len(self.timesteps):
                prev_timestep = self.timesteps[step_index + 1].item()
            else:
                prev_timestep = 0
        except Exception:
            # Fallback if t not in schedule (e.g. during training)
            prev_timestep = max(0, timestep - 1)

        # 3. Compute Degradations (The "Cold" Forward Pass)
        # D(x_0_hat, t)
        d_x0_t = self.degrade(pred_original_sample, timestep)

        # D(x_0_hat, t-1)
        d_x0_prev = self.degrade(pred_original_sample, prev_timestep)

        # 4. Calculate Residual (The "Drift")
        # This captures the high-freq details present in x_t that the model missed
        residual = sample - d_x0_t

        # 5. The Reconstruction Update
        # x_{t-1} = D(x_0_hat, t-1) + Residual
        prev_sample = d_x0_prev + residual

        return prev_sample
