import logging

import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

# One `try` per optional package (non-negotiable 18), resolved ONCE at import.
# It used to sit inside `forward`, which is wrong twice over: Python does not
# negative-cache a failed import, so every step re-ran the whole `sys.meta_path`
# finder search (29 us/call, and `sys.modules` never gains the name), and the
# `except ImportError: pass` that followed chose a different integrator without
# saying so.
#
# `torchdiffeq` is declared in no `pyproject.toml` extra and appears in no
# lockfile, so no environment built from this repository's metadata has ever
# had it: every run of the 8 arms selecting `neural_ode` -- two of them under
# `experiments/validated/` -- took the built-in solver below.
try:
    from torchdiffeq import odeint

    HAVE_TORCHDIFFEQ = True
except ImportError:
    odeint = None
    HAVE_TORCHDIFFEQ = False
    logger.warning(
        "torchdiffeq is not installed, so NeuralODEGenerator integrates with its "
        "built-in fixed-step RK4 instead of torchdiffeq.odeint. Both walk the same "
        "grid (%d uniform steps over [0, integration_time]), so this is a change of "
        "implementation and not of discretisation -- but torchdiffeq's adjoint and "
        "adaptive solvers are unreachable. Install it with `pip install torchdiffeq`.",
        10,
    )


#: Integration steps of the continuous mode, and blocks of the fixed-depth control.
_STEPS = 10

#: ``model_kwargs.depth_mode`` vocabulary of :class:`NeuralODEGenerator`.
DEPTH_MODES = ("continuous", "fixed")


class ConvODEDerivative(nn.Module):
    """ODE Function f(t, h) - Convolutional Derivative Network"""

    def __init__(self, dim):
        """__init__.

        Args:
            dim (Any): Description.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(4, dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(4, dim),
            nn.ReLU(),
        )

    def forward(self, t, x):
        """forward.

        Args:
            t (Any): Description.
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ConvODEDerivative.

        Executes PyTorch tensor operations.

        Args:
            t (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.net(x)


@register_model(
    name="neural_ode",
    training_mode="reconstruction",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class NeuralODEGenerator(nn.Module, IGenerator):
    """
    Neural ODE for continuous-depth MRI reconstruction (Experiment 113).

    Models the transformation from input to output as an ODE: dh/dt = f(t, h).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: int = 64,
        integration_time: float = 1.0,
        depth_mode: str = "continuous",
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (int): Description.
            integration_time (float): Description.
            depth_mode: ``continuous`` integrates one derivative network with RK4
                over ``_STEPS`` steps; ``fixed`` is the fixed-depth control, a ResNet
                of ``_STEPS`` untied residual blocks of the same network. Anything
                else raises.
        """
        super().__init__()
        if depth_mode not in DEPTH_MODES:
            raise ValueError(f"neural_ode depth_mode={depth_mode!r}; expected one of {DEPTH_MODES}.")
        self.depth_mode = depth_mode

        # Handle list features input (take first element)
        if isinstance(features, (list, tuple)):
            features = features[0]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.integration_time = integration_time

        self.feature_extract = nn.Conv2d(in_channels, features, 3, padding=1)
        self.ode_func = ConvODEDerivative(features) if depth_mode == "continuous" else None
        self.residual_blocks = (
            nn.ModuleList(ConvODEDerivative(features) for _ in range(_STEPS))
            if depth_mode == "fixed"
            else None
        )
        self.reconstruct = nn.Conv2d(features, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t_eval: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input [B, C, H, W]
            t_eval: Optional target timestamps [T] or [B, T] in [0, 1].
                    If provided, returns reconstruction at these times.

        forward method for NeuralODEGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_eval (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Initial state
        h = self.feature_extract(x)
        if self.residual_blocks is not None:
            if t_eval is not None:
                raise ValueError("depth_mode='fixed' has no continuous time to evaluate at.")
            for block in self.residual_blocks:
                h = h + block(None, h)
            return self.reconstruct(h)

        # RK4 Solver (4th-order Runge-Kutta)
        steps = _STEPS
        dt = self.integration_time / steps

        if HAVE_TORCHDIFFEQ:
            # `steps + 1` POINTS, so torchdiffeq walks `steps` intervals of `dt`
            # -- the same grid the built-in solver below uses. It was
            # `linspace(..., steps)` before, i.e. 9 intervals of T/9, so the two
            # branches silently disagreed on the discretisation. Nothing
            # published is affected: torchdiffeq is in no extra and no lockfile,
            # so this branch has never executed.
            t_span = torch.linspace(0, self.integration_time, steps + 1, device=x.device)
            h_trajectory = odeint(self.ode_func, h, t_span, method="rk4")
            h = h_trajectory[-1]  # Final state
            return self.reconstruct(h)

        # A ten-iteration loop stood here that computed k1..k4 and never wrote
        # back to `h`. Its four stages were rebound each pass and discarded, so
        # it could not affect the result -- it only doubled the model's forward
        # cost (measured: 80 ode_func evaluations per forward, 40 of them dead)
        # and, on CUDA, added ten host->device transfers via torch.tensor(t).
        # The live RK4 update is the one below at the end of this method.
        current_t = 0.0

        # If t_eval is provided, we need to store intermediate states
        if t_eval is not None:
            if t_eval.dim() > 1:
                target_times = t_eval[0]  # Assume same for batch
            else:
                target_times = t_eval

            outputs = []
            # Start with the initial state at t=0
            state = h

            # We iterate through target times
            for target_t in target_times:
                target_t_float = float(target_t)
                # Integrate until current_t reaches target_t
                while current_t < target_t_float:
                    # Determine step size, ensuring we don't overshoot target_t
                    step_size = min(dt, target_t_float - current_t)
                    if step_size < 1e-6:  # Avoid infinite loops with tiny steps
                        break

                    # RK4 stages for this step
                    k1 = self.ode_func(current_t, state)
                    k2 = self.ode_func(current_t + step_size / 2, state + step_size / 2 * k1)
                    k3 = self.ode_func(current_t + step_size / 2, state + step_size / 2 * k2)
                    k4 = self.ode_func(current_t + step_size, state + step_size * k3)

                    # RK4 update
                    state = state + (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
                    current_t += step_size

                # We are at or past target_t, record the reconstruction
                rec = self.reconstruct(state)
                outputs.append(rec)

            return torch.stack(outputs, dim=1)  # [B, T, C, H, W]

        # If t_eval is None, perform standard fixed-step RK4 for the full integration_time
        for _ in range(steps):
            t_tensor = torch.tensor(current_t, device=h.device, dtype=h.dtype)

            # RK4 stages
            k1 = self.ode_func(t_tensor, h)
            k2 = self.ode_func(t_tensor + dt / 2, h + dt / 2 * k1)
            k3 = self.ode_func(t_tensor + dt / 2, h + dt / 2 * k2)
            k4 = self.ode_func(t_tensor + dt, h + dt * k3)

            # RK4 update
            h = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            current_t += dt  # Update current_t for the next step

        # Return the final reconstruction after full integration
        return self.reconstruct(h)

    def generate(self, z: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate samples - for Neural ODE, z is the input state."""
        return self.forward(z, **kwargs)

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return input_shape

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "neural_ode_generator"
