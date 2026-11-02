r"""Non-Cartesian MRI Simulation Transform.

This module provides a TorchIO-compatible transform that simulates
Non-Cartesian (Spiral, Radial) acquisition physics using NUFFT.

It replaces the "perfect" input image with a "zero-filled" reconstruction
from undersampled non-Cartesian k-space, and stores the raw k-space
and trajectory for downstream use (e.g. data consistency).

.. math::

    y_j = \sum_{n=0}^{N-1} x_n e^{-i 2\pi \textbf{k}_j \cdot \textbf{r}_n}

    \hat{x} = \text{AdjNUFFT}(y) = \sum_{j=0}^{M-1} w_j y_j e^{i 2\pi \textbf{k}_j \cdot \textbf{r}}
"""

import copy

import torch
import torchio as tio
from torchio.transforms import Transform

from spectramr.infrastructure.physics.trajectories import (
    NON_CARTESIAN_TRAJECTORIES,
    TRAJECTORY_TYPES,
    get_trajectory,
)

#: Re-exported so ``data.builders`` can reach the trajectory vocabulary without its own
#: ``spectramr.data -> infrastructure`` layer-direction violation. This module already
#: carries the recorded physics-SSOT exception for that import; the definitions stay in
#: ``trajectories.py`` (single SSOT), only the import path is shortened.
__all__ = [
    "NON_CARTESIAN_TRAJECTORIES",
    "TRAJECTORY_TYPES",
    "NonCartesianSimulationTransform",
]


class NonCartesianSimulationTransform(Transform):
    """Simulates Non-Cartesian acquisition artifacts via NUFFT.

    1. Generates trajectory (Spiral/Radial).
    2. Computes Forward NUFFT (Image -> K-space).
    3. Computes Adjoint NUFFT (K-space -> Aliased Image).
    4. Updates Subject with:
       - 'input': Aliased Image (for model input)
       - 'target': Original Image (for loss calculation if not present)
       - 'measured_kspace': Simulated Non-Cartesian K-space
       - 'trajectory': K-space coordinates
       - 'dcf': Density Compensation Function

    Args:
        pattern: one of ``NON_CARTESIAN_TRAJECTORIES`` -- "radial", "golden_angle",
            "spiral" or "epi". Note "epi" is Cartesian in geometry (it is a specific
            readout ORDERING), so simulating it through NUFFT is correct but does
            more work than a Cartesian mask would; it is offered for distortion
            models that want the true readout path.
        im_size: Expected image size (H, W). If None, inferred per-sample (slower).
        acceleration: Acceleration factor (controls num_arms/spokes).
        norm: FFT normalization ("ortho" not strictly applicable to NUFFT, usually None)
        **kwargs: Forwarded to the trajectory generator, overriding the density
            parameter derived from ``acceleration`` (e.g. an explicit ``num_arms``).
    """

    def __init__(
        self,
        pattern: str = "spiral",
        im_size: tuple[int, int] | None = (256, 256),
        acceleration: float = 4.0,
        enable_noise: bool = False,
        noise_level: float = 0.0,
        **kwargs,
    ):
        """__init__.

        Args:
            pattern (str): Description.
            im_size (Optional[tuple[int, int]]): Description.
            acceleration (float): Description.
            enable_noise (bool): Description.
            noise_level (float): Description.
        """
        super().__init__(**kwargs)
        self.pattern = pattern
        self.im_size = im_size
        self.acceleration = acceleration
        self.enable_noise = enable_noise
        self.noise_level = noise_level
        self.kwargs = kwargs

        # Lazy loading of torchkbnufft to avoid import errors if not installed
        self._check_tkbn()

        # Cache for operators
        self._nufft_op = None
        self._adj_op = None
        self._cached_im_size = None
        # Cache for the (shape-invariant) trajectory + DCF. These depend only
        # on ``(pattern, im_size)`` and the fixed accel/kwargs, yet were
        # regenerated on every ``apply_transform`` (worker hot path). Kept on
        # CPU here and moved to the item's device at use (mirrors the operator
        # cache pattern above). ``_cached_traj_key`` guards a shape change.
        self._cached_traj = None
        self._cached_dcf = None
        self._cached_traj_key: tuple[str, tuple[int, int]] | None = None

    def _density_kwargs(self, im_size: tuple[int, int]) -> dict:
        """Turn ``acceleration`` into the per-family density parameter.

        This is sampling-budget POLICY, which is why it lives here rather than in
        ``trajectories.get_trajectory``: that function generates the geometry a
        caller asks for, and different callers are entitled to different budgets.
        Keeping the split means routing through the shared generator (#1097) cannot
        move the sampling density of an existing arm.

        The radial and spiral expressions are reproduced verbatim from the
        hand-rolled dispatch this replaced -- six `spiral` and two `radial` arms are
        live science, so the arithmetic is a contract, not an implementation detail.

        Explicit ``kwargs`` override the derived value. The old code instead passed
        BOTH (``num_arms=num_arms, **self.kwargs``), which raised
        ``TypeError: got multiple values for keyword argument`` the moment a caller
        supplied one -- the override branch it looked like it had never worked.
        """
        if self.pattern in ("radial", "golden_angle"):
            # NB no ``max(1, ...)`` guard, matching the original. An acceleration
            # above ``max(im_size)`` therefore still yields 0 spokes; no arm is
            # anywhere near that regime (live arms run 256px at 4x -> 64), and
            # widening it here would be a physics change smuggled into a wiring fix.
            derived = {"num_spokes": int(max(im_size) / self.acceleration)}
        elif self.pattern == "spiral":
            base_arms = 48
            derived = {"num_arms": max(1, int(base_arms / self.acceleration))}
        elif self.pattern == "epi":
            # ``get_epi_trajectory`` takes acceleration natively and floors the
            # phase-encode count itself, so the budget needs no separate derivation.
            derived = {"acceleration": max(1, int(self.acceleration))}
        else:
            raise ValueError(
                f"Unknown pattern: {self.pattern}. Expected one of {NON_CARTESIAN_TRAJECTORIES}."
            )
        return {**derived, **self.kwargs}

    def _check_tkbn(self):
        """_check_tkbn.

        Returns:
            Any: Description.
        """
        try:
            import torchkbnufft as tkbn

            self.tkbn = tkbn
        except ImportError:
            raise ImportError(
                "torchkbnufft is required for Non-Cartesian simulation. "
                "Install it via pip install torchkbnufft."
            )

    def _get_operators(self, im_size: tuple[int, int], device: torch.device):
        """Get or create cached NUFFT operators."""
        # Check if we need to recreate operators
        needs_recreate = self._nufft_op is None or self._cached_im_size != im_size

        # For device check, get device from operator's parameters
        if self._nufft_op is not None:
            try:
                # KbNufft doesn't expose .device, get it from parameters
                op_device = next(self._nufft_op.parameters()).device
                if op_device != device:
                    needs_recreate = True
            except StopIteration:
                # No parameters, recreate to be safe
                needs_recreate = True

        if needs_recreate:
            self._nufft_op = self.tkbn.KbNufft(im_size=im_size).to(device)
            self._adj_op = self.tkbn.KbNufftAdjoint(im_size=im_size).to(device)
            self._cached_im_size = im_size

        return self._nufft_op, self._adj_op

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """Apply the transform to a Subject."""
        # Assume 'input' exists and is the high-res ground truth
        if "input" not in subject:
            return subject

        # Copy input to target if not present (save GT)
        if "target" not in subject:
            # ``Image.copy()`` is ``dict.copy`` and returns a plain dict, which patching drops.
            subject["target"] = copy.deepcopy(subject["input"])

        # Get data: (C, H, W, D) always in TorchIO
        image_tensor = subject["input"].data
        # Ensure real-valued float input for trajectory simulation.
        # Complex tensors must be converted to magnitude first to avoid
        # silently discarding the imaginary part.
        if image_tensor.is_complex():
            image_tensor = image_tensor.abs().float()
        else:
            image_tensor = image_tensor.float()

        # Handle dimensions
        if image_tensor.ndim != 4:
            # Should be 4D (C, H, W, D)
            raise ValueError(f"TorchIO Input tensor must be 4D, got {image_tensor.ndim}D")

        # Unpack assuming (C, H, W, D) where H,W are imaging plane and D is slices/batch
        C, H, W, D = image_tensor.shape
        im_size = (H, W)

        # Override im_size if configured (must match actual H,W or we crop? No, usually match)
        if self.im_size is not None and self.im_size != im_size:
            # Just warn or ignore? Using configured might break if mismatch
            # For simulation, we usually assume input is already standardized
            pass

        device = image_tensor.device

        # 1. Trajectory + DCF — shape-invariant, so build once per
        #    (pattern, im_size) and reuse (was regenerated every item).
        traj_key = (self.pattern, im_size)
        if self._cached_traj is None or self._cached_traj_key != traj_key:
            trajectory, dcf = get_trajectory(
                trajectory_type=self.pattern,
                im_size=im_size,
                **self._density_kwargs(im_size),
            )
            self._cached_traj = trajectory
            self._cached_dcf = dcf
            self._cached_traj_key = traj_key

        # Move physics objects to the item's device (cache stays CPU-resident).
        trajectory = self._cached_traj.to(device)
        dcf = self._cached_dcf.to(device)

        # 2. Forward NUFFT
        # KbNufft expects (Batch, C, H, W)
        # We treat D (slices) as Batch.
        # Input: (C, H, W, D) -> (D, C, H, W)
        batch_input = image_tensor.permute(3, 0, 1, 2)

        # Convert to complex for KbNufft (requires complex input or real with last dim=2)
        if not batch_input.is_complex():
            batch_input = batch_input.to(torch.complex64)

        # Get operators
        nufft, adj_nufft = self._get_operators(im_size, device)

        # K-Space: (B, C, Npoints)
        kspace = nufft(batch_input, trajectory)

        # Add Noise
        if self.enable_noise and self.noise_level > 0:
            noise = (
                torch.randn_like(kspace) * self.noise_level
                + 1j * torch.randn_like(kspace) * self.noise_level
            )
            kspace = kspace + noise

        # 3. Adjoint NUFFT (Zero-filled Recon)
        # (B, C, N) * (N,) broadcasting? -> (B, C, N) * (1, 1, N)
        # batch_input was (D, C, H, W) -> B=D.
        kspace_weighted = kspace * dcf.unsqueeze(0).unsqueeze(0)

        # Output: (B, C, H, W) - Complex
        aliased_image = adj_nufft(kspace_weighted, trajectory)

        # Reshape back to TorchIO format (C, H, W, D)
        # (D, C, H, W) -> (C, H, W, D)
        aliased_image = aliased_image.permute(1, 2, 3, 0)

        # 4. Update Subject
        # 'input' -> Zero-filled Recon Magnitude
        aliased_mag = aliased_image.abs()

        # Determine strict 4D shape
        if aliased_mag.ndim == 3:
            # Should not happen with permute, but safe guard
            aliased_mag = aliased_mag.unsqueeze(-1)  # add D=1

        subject["input"].set_data(aliased_mag)

        # Store physics data fields
        # Note: These are not TorchIO Images, just tensors.
        # Collation might be tricky if not wrapped, but ConsolidateDataset logic handles dicts usually.
        subject["measured_kspace"] = kspace.detach().cpu()  # (D, C, N)
        subject["trajectory"] = trajectory.detach().cpu()  # (2, N)
        subject["dcf"] = dcf.detach().cpu()  # (N,)
        # Patch sampling copies a subject from its attributes, so keys set by item
        # assignment alone never reach a training batch.
        subject.update_attributes()

        return subject
