"""CycleGenericGAN for Unpaired MRI Translation (e.g. 3T <-> 7T).

Based on Cluster.md Section 6.3.
Concept: Unpaired translation using two generators (G_AB, G_BA) and two discriminators (D_A, D_B).
Enforces Cycle Consistency: x_A -> G_AB -> x_B -> G_BA -> x_A_recon ~= x_A.
"""

import torch
import torch.nn as nn
from torch import Tensor

from spectramr.models.registry import register_model


# Simple ResNet Generator
@register_model(name="cyclegan_generator", training_mode="gan")
class ResNetGenerator(nn.Module):
    """ResNetGenerator class."""

    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=6):
        """__init__.

        Args:
            input_nc (Any): Description.
            output_nc (Any): Description.
            ngf (Any): Description.
            n_blocks (Any): Description.
        """
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # Downsample
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, 3, stride=2, padding=1),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]

        # ResBlocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Z_anat is the encoder output (after the ResBlocks): the split index below.
        self._anatomy_split = len(model)
        self.latent_channels = ngf * mult

        # Upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.ConvTranspose2d(
                    ngf * mult,
                    int(ngf * mult / 2),
                    3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                nn.InstanceNorm2d(int(ngf * mult / 2)),
                nn.ReLU(True),
            ]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward_with_anatomy(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(translation, Z_anat)`` from one pass; ``Z_anat`` is the encoder bottleneck.

        Splits the same ``Sequential`` ``forward`` runs, so the parameters, state-dict keys and
        output are unchanged; only the bottleneck tensor is also returned.
        """
        z = self.model[: self._anatomy_split](x)
        return self.model[self._anatomy_split :](z), z

    def forward(self, x: Tensor) -> Tensor:
        """forward.

        Args:
            x (Tensor): Description.
        Returns:
            Tensor: Description.

        forward method for ResNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model(x)


class ResnetBlock(nn.Module):
    """ResnetBlock class."""

    def __init__(self, dim):
        """__init__.

        Args:
            dim (Any): Description.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        """forward.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.

        forward method for ResnetBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return x + self.conv_block(x)


class NLayerDiscriminator(nn.Module):
    """NLayerDiscriminator class."""

    def __init__(self, input_nc=1, ndf=64, n_layers=3):
        """__init__.

        Args:
            input_nc (Any): Description.
            ndf (Any): Description.
            n_layers (Any): Description.
        """
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, stride=2, padding=padw),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """forward.

        Args:
            input (Any): Description.
        Returns:
            Any: Description.

        forward method for NLayerDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.model(input)


class CycleGAN(nn.Module):
    """Container for CycleGAN training steps."""

    def __init__(self, input_nc=1, output_nc=1):
        """__init__.

        Args:
            input_nc (Any): Description.
            output_nc (Any): Description.
        """
        super().__init__()
        self.netG_A = ResNetGenerator(input_nc, output_nc)  # A->B (e.g. 3T -> 7T)
        self.netG_B = ResNetGenerator(output_nc, input_nc)  # B->A (e.g. 7T -> 3T)

        self.netD_A = NLayerDiscriminator(input_nc)  # Real A vs Fake A
        self.netD_B = NLayerDiscriminator(output_nc)  # Real B vs Fake B

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, real_A: Tensor, real_B: Tensor) -> dict[str, Tensor]:
        """Performs a forward pass and computes losses for one batch.

        forward method for CycleGAN.

        Executes PyTorch tensor operations.

        Args:
            real_A (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            real_B (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        fake_B = self.netG_A(real_A)
        rec_A = self.netG_B(fake_B)

        fake_A = self.netG_B(real_B)
        rec_B = self.netG_A(fake_A)

        # Generator Losses
        # 1. GAN Loss
        # Train G_A to fool D_B
        pred_fake_B = self.netD_B(fake_B)
        loss_G_A = self.mse_loss(pred_fake_B, torch.ones_like(pred_fake_B))

        # Train G_B to fool D_A
        pred_fake_A = self.netD_A(fake_A)
        loss_G_B = self.mse_loss(pred_fake_A, torch.ones_like(pred_fake_A))

        # 2. Cycle Consistency
        loss_cycle_A = self.l1_loss(rec_A, real_A) * 10.0
        loss_cycle_B = self.l1_loss(rec_B, real_B) * 10.0

        # 3. Identity (Optional, crucial for MRI to preserve color/intensity scaling)
        id_A = self.netG_B(real_A)
        loss_id_A = self.l1_loss(id_A, real_A) * 0.5

        id_B = self.netG_A(real_B)
        loss_id_B = self.l1_loss(id_B, real_B) * 0.5

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_id_A + loss_id_B

        # Discriminator Losses (computed separately typically, but here for demo)
        # Real A
        pred_real_A = self.netD_A(real_A)
        loss_D_A_real = self.mse_loss(pred_real_A, torch.ones_like(pred_real_A))

        pred_fake_A_det = self.netD_A(fake_A.detach())
        loss_D_A_fake = self.mse_loss(pred_fake_A_det, torch.zeros_like(pred_fake_A_det))
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        # Real B
        pred_real_B = self.netD_B(real_B)
        loss_D_B_real = self.mse_loss(pred_real_B, torch.ones_like(pred_real_B))

        pred_fake_B_det = self.netD_B(fake_B.detach())
        loss_D_B_fake = self.mse_loss(pred_fake_B_det, torch.zeros_like(pred_fake_B_det))
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        return {
            "loss_G": loss_G,
            "loss_D_A": loss_D_A,
            "loss_D_B": loss_D_B,
            "fake_B": fake_B,
            "rec_A": rec_A,
        }
