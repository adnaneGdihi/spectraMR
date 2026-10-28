import torch
import torch.nn as nn

from spectramr.infrastructure.physics.fft_ops import ifft2c, sense_adjoint
from spectramr.models.blocks.fno_block import FNOBlock
from spectramr.models.registry import register_model


@register_model(
    name="fno",
    training_mode="reconstruction",
    # k-space input ([Re,Im]); applies IFFT at output so output_domain is image
    # by default (configurable to kspace). Declaring both output domains keeps
    # either config audit-clean.
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain=("image", "kspace"),
)
class FNOGenerator(nn.Module):
    """Fourier Neural Operator Generator for MRI Reconstruction.

    [ARCHITECTURE FIX] This version:
    - Uses frequency-domain grid encoding [-π, π] instead of spatial [0, 1]
    - Applies IFFT at output to convert k-space refinement to image domain
    - Suitable for k-space input with image-domain loss (SSIM, L1, etc.)
    """

    def __init__(
        self,
        in_channels: int = 2,  # [Re, Im] for complex k-space
        out_channels: int = 2,  # [Re, Im] for complex image
        width: int = 64,
        modes1: int = 64,
        modes2: int = 64,
        depth: int = 4,
        output_domain: str = "image",  # 'image' or 'kspace'
        output_form: str = "complex",  # 'complex' ([Re,Im]) or 'magnitude' (1-ch |x|)
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            width (int): Description.
            modes1 (int): Description.
            modes2 (int): Description.
            depth (int): Description.
            output_domain (str): Description.
            output_form (str): ``complex`` emits [Re, Im] (out_channels=2);
                ``magnitude`` emits |x| on image output (out_channels=1).
        """
        super().__init__()

        if "modes" in kwargs:
            modes = kwargs.pop("modes")
            if isinstance(modes, (list, tuple)) and len(modes) >= 2:
                modes1, modes2 = modes[0], modes[1]

        self.output_domain = output_domain
        if output_form not in ("complex", "magnitude"):
            raise ValueError(
                f"FNOGenerator output_form must be 'complex' or 'magnitude'; got {output_form!r}."
            )
        if output_form == "magnitude" and output_domain != "image":
            raise ValueError("FNOGenerator output_form='magnitude' requires output_domain='image'.")
        self.output_form = output_form

        self.fc0 = nn.Linear(in_channels + 2, width)  # +2 for grid encoding

        self.layers = nn.ModuleList([FNOBlock(width, modes1, modes2) for _ in range(depth)])

        self.fc1 = nn.Linear(width, 128)
        # F-FNO-OUTCHAN / 2026-05-20 — note: this layer always produces
        # 2 channels because the downstream forward path (lines ~170-181)
        # unconditionally interprets the output as complex re+im and
        # stacks via ``torch.stack([img.real, img.imag], dim=1)``. The
        # YAML's ``out_channels: 1`` would *appear* to be honoured here
        # but break the model's internal complex contract — so we keep
        # the architectural 2-channel form but validate at construction
        # that the user understands this. CLAUDE.md #9 wants no silent
        # override: surface the constraint as an early TypeError if the
        # user declared a non-2 ``out_channels``, with the fix hint
        # naming the magnitude-extraction adapter.
        expected_out = 1 if output_form == "magnitude" else 2
        if out_channels != expected_out:
            raise ValueError(
                f"FNOGenerator with output_form={output_form!r} emits {expected_out} "
                f"channel(s); YAML declared ``out_channels={out_channels}`` "
                f"(CLAUDE.md pitfall #9). Set ``model.out_channels: {expected_out}``, "
                f"or ``model_kwargs.output_form`` to match."
            )
        self.fc2 = nn.Linear(128, 2)  # Architectural — see note above.

        # Optimization: Pre-compute grid for the expected resolution
        self.grid_cache = {}

    def get_grid(self, shape, device):
        """Generate frequency-domain grid for k-space inputs.

        [ARCHITECTURE FIX] Uses centered frequency coordinates [-π, π]
        instead of spatial [0, 1] to match k-space topology.
        """
        # Robust dimension handling for various input shapes
        if len(shape) == 4:
            # Standard case: [B, C, H, W]
            batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        elif len(shape) == 3:
            # Handle [C, H, W] by assuming batch=1, or [B, H, W]
            # Check if first dim looks like channels (small) or batch
            if shape[0] <= 16:  # Likely channels
                batchsize = 1
                size_x, size_y = shape[1], shape[2]
            else:  # Likely batch
                batchsize = shape[0]
                size_x, size_y = shape[1], shape[2]
        elif len(shape) == 5:
            # 3D volume case: [B, C, D, H, W] - use last two spatial dims
            batchsize, size_x, size_y = shape[0], shape[3], shape[4]
        else:
            raise ValueError(f"FNOGenerator expects 3D-5D input, got {len(shape)}D: {shape}")

        # Robust string key
        device_key = f"{device.type}:{device.index if device.index is not None else 0}"
        grid_key = (size_x, size_y, device_key)

        if grid_key not in self.grid_cache:
            # [ARCHITECTURE FIX] Frequency coordinates for k-space
            gridx = torch.linspace(-torch.pi, torch.pi, size_x, dtype=torch.float, device=device)
            gridx = gridx.view(1, 1, size_x, 1).repeat([1, 1, 1, size_y])

            gridy = torch.linspace(-torch.pi, torch.pi, size_y, dtype=torch.float, device=device)
            gridy = gridy.view(1, 1, 1, size_y).repeat([1, 1, size_x, 1])

            self.grid_cache[grid_key] = torch.cat((gridx, gridy), dim=1)

        return self.grid_cache[grid_key].expand(batchsize, -1, -1, -1)

    def forward(self, x, sensitivity_maps=None, **kwargs):
        """Forward pass; applies the declared ``output_form`` head."""
        out = self._forward_complex(x, sensitivity_maps=sensitivity_maps, **kwargs)
        if self.output_form == "magnitude":
            return torch.sqrt(out[:, 0:1] ** 2 + out[:, 1:2] ** 2)
        return out

    def _forward_complex(self, x, sensitivity_maps=None, **kwargs):
        """Forward pass with optional IFFT output.

        Args:
            x: K-space input (B, 2, H, W) with [Re, Im] channels

        Returns:
            If output_domain='image': Image-domain output (B, 2, H, W)
            If output_domain='kspace': K-space refined output (B, 2, H, W)
        """
        # x: (B, C, H, W)
        if x.ndim == 3:
            x = x.unsqueeze(0)

        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        for layer in self.layers:
            x = layer(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        kspace_pred = x.permute(0, 3, 1, 2)  # (B, 2, H, W) k-space refined

        # [FIX] Physics Projection for Loss Calculation
        # Always preserve 2-channel [Re, Im] output for proper loss computation
        if self.training and kwargs.get("project_to_image", True):
            # Check for Multi-Coil SENSE
            sensitivity_maps = (
                kwargs.get("sensitivity_maps") if sensitivity_maps is None else sensitivity_maps
            )

            if sensitivity_maps is not None:
                # Multi-coil case
                if kspace_pred.shape[1] > 2 and kspace_pred.shape[1] % 2 == 0:
                    coils = kspace_pred.shape[1] // 2
                    k_complex = torch.complex(kspace_pred[:, :coils], kspace_pred[:, coils:])
                    img_complex = sense_adjoint(k_complex, sensitivity_maps)
                    # [FIX] Return [Re, Im] instead of magnitude
                    return torch.stack([img_complex.real, img_complex.imag], dim=1)

            # Single Coil - convert k-space to image domain
            if kspace_pred.shape[1] == 2:
                k_complex = torch.complex(kspace_pred[:, 0], kspace_pred[:, 1])
                img = ifft2c(k_complex)
                # [FIX] Return [B, 2, H, W] with [Re, Im], NOT magnitude!
                return torch.stack([img.real, img.imag], dim=1)

        # Convert to image domain for inference/validation if requested
        if self.output_domain == "image":
            x_complex = torch.complex(kspace_pred[:, 0], kspace_pred[:, 1])
            # Use canonical centered IFFT
            x_image = ifft2c(x_complex)
            return torch.stack([x_image.real, x_image.imag], dim=1)

        return kspace_pred


@register_model(
    name="uno",
    training_mode="reconstruction",
    spatial_dims=(2, 3),
    input_domain="image",
    output_domain="image",
    accepts_complex=False,
    requires_paired_data=True,
)
class UNOGenerator(nn.Module):
    """U-shaped Neural Operator (U-NO) for MRI.

    Combines U-Net's multi-scale processing with FNO's spectral learning.
    Reduces Gibbs ringing by handling local features in the contracting/expanding path.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        width: int = 64,
        modes: int = 32,  # Tuples handled in logic if needed
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            width (int): Description.
            modes (int): Description.
        """
        super().__init__()
        self.width = width
        # U-Net structure with FNO blocks

        # Projection
        self.fc0 = nn.Linear(in_channels + 2, width)

        # Encoder
        self.enc1 = FNOBlock(width, modes, modes)
        self.down1 = nn.Conv2d(width, width * 2, 2, stride=2)

        self.enc2 = FNOBlock(width * 2, modes // 2, modes // 2)
        self.down2 = nn.Conv2d(width * 2, width * 4, 2, stride=2)

        # Bottleneck
        self.bot = FNOBlock(width * 4, modes // 4, modes // 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = FNOBlock(width * 2, modes // 2, modes // 2)  # concat: width*4 -> proj?
        # Actually U-NO usually sums or cats. Let's cat and project.
        self.cat_proj2 = nn.Conv2d(width * 4, width * 2, 1)

        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.cat_proj1 = nn.Conv2d(width * 2, width, 1)
        self.dec1 = FNOBlock(width, modes, modes)

        # Output
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

        self.grid_cache = {}

    def get_grid(self, shape, device):
        # Same grid logic as FNOGenerator
        """get_grid.

        Args:
            shape (Any): Description.
            device (Any): Description.
        Returns:
            Any: Description.
        """
        batchsize, size_x, size_y = shape[0], shape[2], shape[3]
        device_key = str(device)
        grid_key = (size_x, size_y, device_key)

        if grid_key not in self.grid_cache:
            gridx = torch.linspace(-torch.pi, torch.pi, size_x, device=device)
            gridx = gridx.view(1, 1, size_x, 1).repeat([1, 1, 1, size_y])
            gridy = torch.linspace(-torch.pi, torch.pi, size_y, device=device)
            gridy = gridy.view(1, 1, 1, size_y).repeat([1, 1, size_x, 1])
            self.grid_cache[grid_key] = torch.cat((gridx, gridy), dim=1)

        return self.grid_cache[grid_key].expand(batchsize, -1, -1, -1)

    def forward(self, x, **kwargs):
        # x: (B, C, H, W)
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for UNOGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=1)

        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        # Encoder
        x1 = self.enc1(x)
        x2_in = self.down1(x1)
        x2 = self.enc2(x2_in)
        x3_in = self.down2(x2)

        # Bottleneck
        x3 = self.bot(x3_in)

        # Decoder
        x2_up = self.up2(x3)
        # Skip connection
        x2_cat = torch.cat([x2_up, x2], dim=1)
        x2_cat = self.cat_proj2(x2_cat)
        x2_out = self.dec2(x2_cat)

        x1_up = self.up1(x2_out)
        x1_cat = torch.cat([x1_up, x1], dim=1)
        x1_cat = self.cat_proj1(x1_cat)
        x_out = self.dec1(x1_cat)

        # Projection
        x_out = x_out.permute(0, 2, 3, 1)
        x_out = self.fc1(x_out)
        x_out = nn.functional.gelu(x_out)
        x_out = self.fc2(x_out)

        return x_out.permute(0, 3, 1, 2)
