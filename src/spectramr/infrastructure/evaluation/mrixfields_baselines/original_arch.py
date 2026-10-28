"""External reference architectures for loading original MRIxFields2026 pretrained weights.

This module is NOT a registered framework model and is NOT integrated into the
registry/DI system. Its sole purpose is to faithfully reproduce the original
clovaai/stargan-v2 architecture so that the official MRIxFields2026 pretrained
checkpoint state dicts (``nets_ema["generator"]``, ``nets_ema["mapping_network"]``)
can be loaded with ``strict=True`` — identical submodule attribute names, identical
parameter shapes.

Sources:
    - Original StarGAN v2 implementation: https://github.com/clovaai/stargan-v2/blob/master/core/model.py
    - MRIxFields2026 adaptation: /tmp/mrix/Baseline_mrixfields_models_stargan_v2.py
      (``StarGANv2Generator`` → ``ClovaaiStarGANv2Generator``,
       ``MappingNetwork``     → ``ClovaaiMappingNetwork``)

Reference:
    Choi et al., "StarGAN v2: Diverse Image Synthesis for Multiple Domains", CVPR 2020.

Do NOT import this module outside of the ``infrastructure/evaluation/`` layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Validation helper (verbatim from source)
# ---------------------------------------------------------------------------

_VALID_IMG_SIZES = {32, 64, 128, 256, 512}


def _validate_img_size(img_size: int) -> None:
    """Validate that img_size is a supported power of 2."""
    if img_size not in _VALID_IMG_SIZES:
        raise ValueError(f"img_size must be one of {sorted(_VALID_IMG_SIZES)}, got {img_size}")


# ---------------------------------------------------------------------------
# Building blocks (verbatim from source, attribute names preserved for strict load)
# ---------------------------------------------------------------------------


class ResBlk(nn.Module):
    """Residual block with optional downsampling and normalization.

    Based on StarGAN v2 official (clovaai/stargan-v2).
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        actv: nn.Module | None = None,
        normalize: bool = False,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv: nn.Module = nn.LeakyReLU(0.2) if actv is None else actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in: int, dim_out: int) -> None:
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization.

    Modulates features using style code: (1 + gamma) * norm(x) + beta.

    Attribute names ``norm`` and ``fc`` are preserved verbatim so that
    an original checkpoint loads with strict=True.
    """

    def __init__(self, style_dim: int, num_features: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta


class AdainResBlk(nn.Module):
    """Residual block with AdaIN for style injection.

    Based on StarGAN v2 official (clovaai/stargan-v2).
    Attribute names ``conv1``, ``conv2``, ``norm1``, ``norm2``, ``conv1x1``
    are preserved verbatim so that an original checkpoint loads with strict=True.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        style_dim: int = 64,
        actv: nn.Module | None = None,
        upsample: bool = False,
    ) -> None:
        super().__init__()
        self.actv: nn.Module = nn.LeakyReLU(0.2) if actv is None else actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in: int, dim_out: int, style_dim: int = 64) -> None:
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


# ---------------------------------------------------------------------------
# Generator (renamed from StarGANv2Generator)
# ---------------------------------------------------------------------------


class ClovaaiStarGANv2Generator(nn.Module):
    """StarGAN v2 generator with AdaIN style injection.

    Verbatim port of ``StarGANv2Generator`` from the MRIxFields2026 baseline code
    (itself derived from clovaai/stargan-v2). Renamed to avoid clashing with
    the framework's native StarGANv2Generator.

    Submodule attributes ``from_rgb``, ``encode``, ``decode``, ``to_rgb`` are
    preserved verbatim so that ``nets_ema["generator"]`` state dicts load with
    ``strict=True``.

    Parameters:
        img_size: Input image size (must be in {32, 64, 128, 256, 512}).
        style_dim: Style code dimension.
        max_conv_dim: Maximum conv channels.
        input_nc: Input channels (1 for grayscale MRI).
    """

    def __init__(
        self,
        img_size: int = 128,
        style_dim: int = 64,
        max_conv_dim: int = 512,
        input_nc: int = 1,
    ) -> None:
        super().__init__()
        _validate_img_size(img_size)
        # Initial channel count scales inversely with image size (StarGAN v2 convention).
        dim_in = 2**14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv2d(input_nc, dim_in, 3, 1, 1)
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm2d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(dim_in, input_nc, 1, 1, 0),
            nn.Tanh(),
        )

        # Down/up-sampling blocks
        repeat_num = int(math.log2(img_size)) - 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            self.encode.append(ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_in, style_dim, upsample=True))
            dim_in = dim_out

        # Bottleneck blocks
        for _ in range(2):
            self.encode.append(ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(0, AdainResBlk(dim_out, dim_out, style_dim))

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input image (B, 1, H, W).
            s: Style code (B, style_dim).

        Returns:
            Generated image (B, 1, H, W) with values in [-1, 1].
        """
        x = self.from_rgb(x)
        for block in self.encode:
            x = block(x)
        for block in self.decode:
            x = block(x, s)
        return self.to_rgb(x)


# ---------------------------------------------------------------------------
# Mapping Network (renamed from MappingNetwork)
# ---------------------------------------------------------------------------


class ClovaaiMappingNetwork(nn.Module):
    """Maps random latent codes to domain-specific style codes.

    Verbatim port of ``MappingNetwork`` from the MRIxFields2026 baseline code
    (itself derived from clovaai/stargan-v2). Renamed to avoid clashing with
    any framework-native ``MappingNetwork``.

    Submodule attributes ``shared`` and ``unshared`` are preserved verbatim so
    that ``nets_ema["mapping_network"]`` state dicts load with ``strict=True``.

    Parameters:
        latent_dim: Input latent dimension.
        style_dim: Output style dimension.
        num_domains: Number of domains (15 for joint task3 with 3 contrasts x 5 fields).
    """

    def __init__(self, latent_dim: int = 16, style_dim: int = 64, num_domains: int = 15) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(latent_dim, 512), nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512), nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared.append(
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, style_dim),
                )
            )

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Latent code (B, latent_dim).
            y: Target domain label (B,) as LongTensor.

        Returns:
            Style code (B, style_dim).
        """
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out.append(layer(h))
        out_t = torch.stack(out, dim=1)  # (B, num_domains, style_dim)
        idx = torch.arange(y.size(0), device=y.device)
        return out_t[idx, y]  # (B, style_dim)


# ---------------------------------------------------------------------------
# Checkpoint state extraction
# ---------------------------------------------------------------------------

_NETG_PREFIXES = ("netG.", "netG_AB.")


def extract_generator_state(state_dict: dict) -> dict:
    """Extract the generator sub-state from an original-repo checkpoint.

    Handles the following checkpoint layouts emitted by the original
    MRIxFields2026 training code:

    - ``{"model": {"netG.<key>": ..., "netD.<key>": ...}}``
      → strips ``"netG."`` prefix; excludes discriminator keys.
    - ``{"model": {"netG_AB.<key>": ..., "netG_BA.<key>": ...}}``
      → strips ``"netG_AB."`` prefix (CycleGAN A→B direction).
    - ``{"generator": {<key>: ...}}``
      → returns the ``"generator"`` sub-dict directly.
    - Bare dict with no ``"net*"``-prefixed top-level keys
      → returned as-is (already a generator state dict).

    Raises:
        ValueError: For any layout that does not match the above patterns
            (fail-loud per pitfall #9).

    Args:
        state_dict: Raw dict loaded from a ``.pth`` checkpoint file.

    Returns:
        Generator state dict ready for ``model.load_state_dict(strict=True)``.
    """
    if "model" in state_dict:
        for prefix in _NETG_PREFIXES:
            g = {
                k[len(prefix) :]: v for k, v in state_dict["model"].items() if k.startswith(prefix)
            }
            if g:
                return g
        top_groups = sorted({k.split(".")[0] for k in state_dict["model"]})
        raise ValueError(
            f"checkpoint 'model' has no netG./netG_AB. prefix; top-level key groups={top_groups}"
        )

    if "generator" in state_dict:
        return state_dict["generator"]

    if all(not k.startswith("net") for k in state_dict):
        return state_dict

    raise ValueError(f"unrecognized checkpoint layout; top-level keys={list(state_dict)}")
