#!/usr/bin/env python
"""K-Space Cold Diffusion Model

This module implements a cold diffusion model specifically designed for
k-space denoising. The model operates on k-space data, learning to
progressively denoise undersampled k-space through a series of timesteps,
ultimately reconstructing high-quality images via inverse FFT.

[ARCHITECTURE IMPROVEMENTS - Phase 5]
This generator now supports optional integration of three physics-informed components:

1. **FrequencyWeightedL1Loss**: K-space loss that upweights high-frequency components
   to counteract natural 1/f spectral decay in MRI. Prevents blur artifacts.
   Usage: Instantiate in training strategy and add to reconstruction loss term.
   from spectramr.models.losses import create_loss
   freq_loss = create_loss('frequency_weighted_l1_kspace', height=256, width=256, alpha=2.0)

2. **PhaseSafeDualAttention**: Attention mechanism that preserves phase information
   by computing weights on magnitude (phase-neutral) but applying to complex values.
   Prevents 180° phase flips that cause ghosting artifacts.
   Usage: Add to bottleneck of FourierBridgeNetwork or UNet
   from spectramr.models.blocks.attention import PhaseSafeDualAttention
   attn = PhaseSafeDualAttention(in_channels=64)

3. **Circular Padding**: All convolutions use circular padding (padding='circular') to
   preserve FFT periodicity. k-space is periodic (cyclic boundary conditions), so
   circular padding is physically appropriate. Eliminates Gibbs ringing artifacts
   that occur with zero-padding at domain boundaries.
"""

import inspect
import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.domain.exceptions import ConfigurationError
from spectramr.infrastructure.physics.data_consistency import (
    VALID_DC_METHODS,
    AdaptiveDataConsistency,
    HardDataConsistency,
    NoiseAdaptiveDataConsistency,
    NoiseAdjustedDataConsistency,
    SimpleDataConsistency,
    SoftDataConsistency,
    TargetAwareFSDC,
    dc_passthrough_center_patch,
)
from spectramr.infrastructure.physics.fft_ops import fft2c, ifft2c, sense_adjoint
from spectramr.infrastructure.physics.kan_data_consistency import (
    KANAdaptiveDataConsistency,
)
from spectramr.infrastructure.physics.wiener_gain_dc import WienerGainDataConsistency
from spectramr.models.blocks.activation import ComplexActivation
from spectramr.models.blocks.attention import (
    CBAMSpatialAttention,
    ChannelAttention,
    ComplexChannelAttention,
    IdentityAtInitAttention,
    KernelizedAttention,
    LinearAttention,
    PhaseSafeDualAttention,
    SparseAttention,
)
from spectramr.models.blocks.attention_domains import (
    ATTENTION_DOMAIN_SUPPORT,
    complex_to_interleaved,
    validate_feature_domain,
)
from spectramr.models.blocks.complex_channel_gain import ComplexChannelGain
from spectramr.models.blocks.contrast_conditioning import (
    expand_contrast_over_slices,
    fuse_contrast_embedding,
)
from spectramr.models.blocks.dual_domain import DualDomainBlock
from spectramr.models.blocks.dual_domain_attention import DualDomainAttention
from spectramr.models.blocks.dual_domain_attention_kan import (
    KANGatedDualDomainAttention,
    WaveletFreqAttentionBlock,
)
from spectramr.models.blocks.hermitian_null_space_attention import (
    HermitianNullSpaceAttention,
)
from spectramr.models.blocks.kspace_geometry_encoding import (
    KSPACE_GEOMETRY_CHANNELS,
    as_interleaved_complex,
    kspace_geometry_channels,
    measurement_support,
    resolve_geometry_planes,
)
from spectramr.models.blocks.null_space_attention import NullSpaceDualDomainAttention
from spectramr.models.blocks.radial_band_gain import DEFAULT_MAX_LOG_GAIN, RadialBandGain
from spectramr.models.blocks.timestep_embedding import sinusoidal_timestep_embedding
from spectramr.models.constants import DEFAULT_DIFFUSION_TIMESTEPS
from spectramr.models.generators.backbone_builders import (
    BackboneBuildContext,
    build_backbone,
    registered_backbone_names,
)
from spectramr.models.generators.kspace_line_tokens import LINE_AXES, LINE_BACKBONES
from spectramr.models.generators.option_parsing import parse_enum_option
from spectramr.models.generators.reference_conditioning import (
    prepend_reference,
    resolve_reference_channels,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.layers.complex_conv import ComplexConv2d
from spectramr.models.registry import register_model
from spectramr.models.utils.shape_validator import validate_5d_or_4d_tensor
from spectramr.models.generators.kan_param_groups import (
    build_kan_param_groups,
    resolve_kan_param_group_ratios,
)

logger = logging.getLogger(__name__)


def model_expects_smaps_concat(model: object, *, default: bool = False) -> bool:
    """Does ``model``'s backbone expect S-maps concatenated onto its input?

    **This is the one resolver for that question** (CLAUDE.md #17). Every site
    that sizes a stack for a cold-diffusion network -- the training strategy,
    ``forward_probe``, ``energy_probe``, ``ColdDiffusionInferenceStrategy`` --
    asks here rather than re-deriving the rule, because re-deriving it is
    exactly what broke: four call sites each read ``condition_with_smaps`` (or,
    in ``_assert_trained_width``, hard-coded ``2 * in_channels``) and so fed a
    doubled stack to the six ``diff_varnet`` / ``diff_varnet_kan`` arms, whose
    backbones run their own data consistency and are built at ``1x``.

    ``condition_with_smaps`` is the *arm's declaration*; it is NOT this answer.
    A generator may honour it and still be built at ``1x`` when its backbone is
    in ``_INTERNAL_DC_BACKBONES``. The generator resolves the two into
    ``expects_smaps_concat`` at construction, so that attribute -- read off the
    built object -- is authoritative.

    The fallback to ``condition_with_smaps`` covers models that predate the
    attribute (and the test stubs that model them); it is deliberately the only
    place in the codebase where that substitution is made.

    ``default`` is the answer for a model carrying *neither* attribute, and it
    is not the same everywhere: the probes and the inference path historically
    read ``getattr(..., "condition_with_smaps", False)`` and so must keep
    answering ``False``, while ``DiffusionTrainingStrategy`` concatenated
    unconditionally and must keep answering ``True`` for the non-cold
    generators it also drives. Passing it explicitly keeps that difference
    visible at the call site instead of hiding it in a second spelling.
    """
    concat = getattr(model, "expects_smaps_concat", None)
    if concat is not None:
        return bool(concat)
    declared = getattr(model, "condition_with_smaps", None)
    if declared is not None:
        return bool(declared)
    return default


#: ``KSpaceColdDiffusionGenerator.__init__`` defaults for the two knobs that
#: decide S-map concatenation. They live here, not inline in ``__init__``, so
#: :func:`config_expects_smaps_concat` cannot answer for an arm that omits them
#: while ``__init__`` answers differently -- a divergence that surfaces as a
#: wrong *number* rather than an error (CLAUDE.md #17).
DEFAULT_BACKBONE_TYPE = "unet"

#: The spellings the ``if/elif`` chain in ``FourierBridgeNetwork.__init__``
#: handles itself. Everything else falls through to ``backbone_builders``.
_CHAIN_BACKBONE_TYPES: tuple[str, ...] = (
    "complex_unet",
    "diff_varnet",
    "diff_varnet_kan",
    "mamba_unet",
    "nafnet",
    "restormer",
    "swin_diff_rec",
    "swin_diff_rec_kan",
    "swin_transformer",
    "swinir",
    "unet",
    "vision_mamba",
    "vision_transformer",
    "vit",
)
DEFAULT_CONDITION_WITH_SMAPS = True


def resolve_expects_smaps_concat(*, backbone_type: str, condition_with_smaps: bool) -> bool:
    """The rule deciding whether S-maps are concatenated onto the backbone input.

    **The one owner of that rule** (CLAUDE.md #17).
    :meth:`KSpaceColdDiffusionGenerator.__init__` calls this to set
    ``self.expects_smaps_concat``, and the config auditor reaches it through
    :func:`config_expects_smaps_concat`; neither restates the conjunction.

    It IS a conjunction, and that is the whole point: ``condition_with_smaps``
    is the arm's *declaration*, but an internal-DC backbone is built at ``1x``
    regardless, because its per-cascade ``MaskedReplacementDataConsistency`` compares
    against a channel-preserved measurement. Reading either half alone gives
    the wrong answer for one of the two arm families.
    """
    return bool(
        condition_with_smaps
        and backbone_type not in KSpaceColdDiffusionGenerator._INTERNAL_DC_BACKBONES
    )


def config_expects_smaps_concat(model_kwargs: Mapping[str, Any] | None) -> bool:
    """Same answer as :func:`resolve_expects_smaps_concat`, from an arm's YAML.

    Takes ``config.model.model_kwargs`` and applies the generator's own
    defaults, so a *static* consumer -- ``ConfigHealthChecker``, which only
    ever sees the frozen ``TrainingSettings`` and never a built module -- gets
    the identical verdict without constructing the model.

    This exists because :func:`model_expects_smaps_concat` reads
    ``expects_smaps_concat`` off a **built** object and therefore cannot serve
    a pre-construction caller. Issue #1387: the auditor keyed four channel
    checks on ``model_type == "kspace_cold_diffusion"`` alone, which is true
    for the internal-DC arms too -- so six arms that concatenate *nothing* had
    their in_channels checks silently waived.
    """
    kw = dict(model_kwargs or {})
    return resolve_expects_smaps_concat(
        backbone_type=str(kw.get("backbone_type", DEFAULT_BACKBONE_TYPE)),
        condition_with_smaps=bool(kw.get("condition_with_smaps", DEFAULT_CONDITION_WITH_SMAPS)),
    )


#: The radial-band-token knobs an arm may declare. ``radial_band_tokens_rungs`` is
#: deliberately absent: the generator derives it from the timestep count.
_RADIAL_BAND_TOKEN_KNOBS = frozenset(
    {"radial_band_tokens_bands", "radial_band_tokens_dim", "radial_band_tokens_max_log"}
)


#: The model_kwargs each transformer backbone's builder reads (besides ``heads``).
_TRANSFORMER_KNOBS: dict[str, frozenset[str]] = {
    "dit": frozenset({"depth", "patch_size", "mlp_ratio"}),
    "uvit": frozenset({"depth", "patch_size", "mlp_ratio"}),
    "diffit": frozenset({"depth", "patch_size", "mlp_ratio"}),
    "hat": frozenset({"depth", "window_size", "cab_weight"}),
}
_TRANSFORMER_ALIASES = {"u_vit": "uvit"}


def _transformer_knob_defects(backbone_type: str, kw: Mapping[str, Any]) -> list[str]:
    """Inert and misspelt knobs for dit/uvit/diffit/hat, which absorb unknown keys."""
    import difflib

    name = _TRANSFORMER_ALIASES.get(backbone_type, backbone_type)
    own = _TRANSFORMER_KNOBS.get(name)
    if own is None:
        return []
    out: list[str] = []
    others = frozenset().union(*_TRANSFORMER_KNOBS.values()) - own
    inert = sorted(k for k in kw if k in others)
    if inert:
        out.append(f"{inert} are never read by backbone_type={backbone_type!r}.")
    known = (
        own
        | others
        | set(inspect.signature(KSpaceColdDiffusionGenerator.__init__).parameters)
        | {"heads", "max_timesteps"}
    )
    for key in sorted(k for k in kw if k not in known):
        near = difflib.get_close_matches(key, sorted(own | {"heads"}), n=1, cutoff=0.75)
        if near:
            out.append(f"model_kwargs.{key} is not read by backbone_type={backbone_type!r}; did you mean {near[0]!r}?")
    return out


def _accepts_timesteps(backbone: nn.Module) -> bool:
    """True when ``backbone.forward`` names a ``timesteps`` parameter (exact name, not a substring)."""
    return "timesteps" in inspect.signature(backbone.forward).parameters


def backbone_kwargs_defects(model_kwargs: Mapping[str, Any] | None) -> list[str]:
    """Every ``backbone_type`` x ``model_kwargs`` incompatibility in an arm, as messages.

    **The one owner of these rules** (CLAUDE.md #17). The constructor raises on the
    first entry; the audit witness ``kspace_backbone_kwargs_compatible`` reports all
    of them from the frozen config, without building the model. Before #2047 the
    rules lived only in the constructor, so ``audit --strict`` passed four arms that
    then died at generator construction. Absent keys take the constructor's defaults.
    """
    kw = dict(model_kwargs or {})
    backbone_type = str(kw.get("backbone_type", DEFAULT_BACKBONE_TYPE))
    force_pure_kspace = bool(kw.get("force_pure_kspace", False))
    feature_norm = kw.get("kspace_feature_norm", "none")
    bands = kw.get("radial_band_tokens_bands", 0)
    defects: list[str] = []

    if "radial_band_tokens_rungs" in kw:
        defects.append(
            "radial_band_tokens_rungs is derived from the process's timestep count, "
            "not declared. Remove it; model_kwargs.timesteps already owns that fact."
        )
    unknown = sorted(
        k
        for k in kw
        if k.startswith("radial_band_tokens")
        and k not in _RADIAL_BAND_TOKEN_KNOBS
        and k != "radial_band_tokens_rungs"
    )
    if unknown:
        defects.append(
            f"Unrecognised radial band token knob(s) {unknown}. This constructor "
            "absorbs **kwargs, so a misspelled knob runs as the control in silence."
        )

    if "kspace_geometry_planes" in kw:
        if backbone_type in KSpaceColdDiffusionGenerator._GEOMETRY_EXEMPT_BACKBONES:
            defects.append(
                f"kspace_geometry_planes applies only to backbones built with geometry "
                f"planes, not backbone_type={backbone_type!r}."
            )
        try:
            resolve_geometry_planes(kw["kspace_geometry_planes"])
        except ValueError as exc:
            defects.append(str(exc))

    line_knobs = sorted(k for k in kw if k.startswith("line_ax") or k == "line_axes")
    if backbone_type in LINE_BACKBONES:
        if line_knobs not in ([], ["line_axis"]):
            defects.append(f"Unrecognised line-token knob(s) {line_knobs}; the knob is 'line_axis'.")
        elif "line_axis" in kw and kw["line_axis"] not in LINE_AXES:
            defects.append(f"line_axis={kw['line_axis']!r}; expected one of {LINE_AXES}.")
        inert = sorted(k for k in ("patch_size", "window_size", "cab_weight") if k in kw)
        if inert:
            defects.append(
                f"{inert} are patch/window knobs that backbone_type={backbone_type!r} never "
                "reads: it tokenises whole k-space lines (see line_axis)."
            )
    elif line_knobs:
        defects.append(f"line_axis applies only to the line-token backbones, not {backbone_type!r}.")

    defects.extend(_transformer_knob_defects(backbone_type, kw))

    if backbone_type != "complex_unet":
        if str(feature_norm).lower() != "none":
            defects.append(
                f"kspace_feature_norm={feature_norm!r} applies only to "
                f"backbone_type='complex_unet', not {backbone_type!r}."
            )
        if bands:
            defects.append(
                f"radial_band_tokens_bands={bands} applies only to "
                f"backbone_type='complex_unet', not {backbone_type!r}: no other "
                "backbone exposes a per-up-step seam for it."
            )
    elif bands and not force_pure_kspace:
        defects.append(
            "radial_band_tokens_bands needs force_pure_kspace: true -- the annuli are "
            "k-space annuli, and without it the decoder's feature maps are images."
        )

    # The bridge is the single owner of domain conversion: force_pure_kspace skips
    # its entry ifft2c, so an image-domain DC backbone would FFT k-space a second time
    # and enforce DC in the wrong domain with no shape error. A pure-k-space VarNet
    # needs a k-space-native DC operator, which is a new component, not a flag.
    if force_pure_kspace and backbone_type in KSpaceColdDiffusionGenerator._IMAGE_DOMAIN_DC_BACKBONES:
        defects.append(
            f"backbone_type={backbone_type!r} performs data consistency "
            "internally on IMAGE-domain tensors (its MaskedReplacementDataConsistency "
            "FFTs its own input), so force_pure_kspace=true would hand it "
            "k-space and cause a second forward transform -- DC applied in "
            "the wrong domain. Set model_kwargs.force_pure_kspace: false so "
            "FourierBridgeNetwork performs the entry ifft2c, or choose "
            "backbone_type: 'complex_unet', which is k-space-native."
        )
    return defects


class ChannelAdapter(nn.Module):
    """Learned channel projection for adaptive input/output adaptation.

    Instead of truncating or zero-padding channels (which lose information),
    this module learns a 1x1 convolution to project between different channel counts.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize channel adapter.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels != out_channels:
            # Learned projection using 1x1 convolution
            # bias=False: k-space constant term creates spatial Dirac delta singularity (Hammernik et al. 2018)
            self.adapter = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, padding=0, bias=False
            )
        else:
            # No adaptation needed
            self.adapter = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project channels while preserving gradients.

        Args:
            x: (B, C_in, H, W) input tensor

        Returns:
            (B, C_out, H, W) output tensor
        """
        return self.adapter(x)


class KSpaceCrop(nn.Module):
    """Downsamples spatial resolution by center-cropping frequencies in k-space."""

    def __init__(self, scale_factor: int = 2):
        """__init__.

        Args:
            scale_factor (int): Description.
        """
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            target_shape (tuple[int, int] | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpaceCrop.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_shape (tuple[int, int] | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C, H, W = x.shape
        if target_shape is not None:
            new_H, new_W = target_shape
        else:
            new_H = H // self.scale_factor
            new_W = W // self.scale_factor

        if new_H == H and new_W == W:
            return x

        if new_H > H or new_W > W:
            raise ValueError(f"KSpaceCrop cannot increase size from ({H},{W}) to ({new_H},{new_W})")

        start_H = (H - new_H) // 2
        start_W = (W - new_W) // 2

        # Due to norm="ortho", shrinking k-space size multiplies spatial amplitude by scale_factor.
        # We divide by scale_factor to preserve spatial amplitude variance.
        cropped = x[..., start_H : start_H + new_H, start_W : start_W + new_W]
        return cropped / self.scale_factor


class KSpacePad(nn.Module):
    """Upsamples spatial resolution by zero-padding high frequencies in k-space.

    Applies a smooth Tukey apodization window before zero-padding to prevent
    hard frequency-domain step functions that cause Sinc ringing artifacts
    (crosshair patterns) in the spatial domain.
    """

    def __init__(self, scale_factor: int = 2, taper_fraction: float = 0.3):
        """__init__.

        Args:
            scale_factor (int): Description.
            taper_fraction (float): Description.
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.taper_fraction = taper_fraction

    @staticmethod
    def _tukey_1d(N: int, alpha: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Generate 1D Tukey window: flat center with cosine taper at edges.

        alpha=0: rectangular (no taper), alpha=1: Hann window.
        """
        if N <= 1:
            return torch.ones(N, device=device, dtype=dtype)
        if alpha <= 0:
            return torch.ones(N, device=device, dtype=dtype)
        if alpha >= 1:
            return torch.hann_window(N, periodic=False, device=device, dtype=dtype)

        n = torch.arange(N, device=device, dtype=dtype)
        width = int(alpha * N / 2)
        if width < 1:
            return torch.ones(N, device=device, dtype=dtype)

        w = torch.ones(N, device=device, dtype=dtype)
        # Left cosine taper
        left = 0.5 * (1 - torch.cos(torch.pi * n[:width] / width))
        w[:width] = left
        # Right cosine taper
        right = 0.5 * (1 - torch.cos(torch.pi * (N - 1 - n[-width:]) / width))
        w[-width:] = right
        return w

    def forward(self, x: torch.Tensor, target_shape: tuple[int, int] | None = None) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            target_shape (tuple[int, int] | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpacePad.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_shape (tuple[int, int] | None): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        B, C, H, W = x.shape
        if target_shape is not None:
            new_H, new_W = target_shape
            current_scale = new_W / W
        else:
            new_H = H * self.scale_factor
            new_W = W * self.scale_factor
            current_scale = float(self.scale_factor)

        if new_H == H and new_W == W:
            return x

        if new_H < H or new_W < W:
            raise ValueError(f"KSpacePad cannot decrease size from ({H},{W}) to ({new_H},{new_W})")

        # Apply Tukey apodization BEFORE padding to smooth the crop boundary.
        # This prevents the hard rectangular step function that causes
        # 2D Sinc ringing (crosshair artifacts) in the spatial domain.
        window_h = self._tukey_1d(H, self.taper_fraction, x.device, x.dtype)
        window_w = self._tukey_1d(W, self.taper_fraction, x.device, x.dtype)
        window_2d = torch.outer(window_h, window_w).view(1, 1, H, W)
        x = x * window_2d

        pad_H_total = new_H - H
        pad_W_total = new_W - W

        pad_top = pad_H_total // 2
        pad_bottom = pad_H_total - pad_top
        pad_left = pad_W_total // 2
        pad_right = pad_W_total - pad_left

        padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=0.0)

        # Due to norm="ortho", expanding k-space size divides spatial amplitude by scale_factor.
        # We multiply by scale_factor to preserve spatial amplitude variance.
        return padded * current_scale


_REAL_ACTIVATION_FACTORIES = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
    "gelu": lambda: nn.GELU(),
}
_COMPLEX_ACTIVATION_NAMES = ("complex", "modrelu")
_VALID_ACTIVATIONS = (*_REAL_ACTIVATION_FACTORIES, *_COMPLEX_ACTIVATION_NAMES)


def _resolve_activation(name: str, channels: int) -> nn.Module:
    """Resolve an activation name for the k-space blocks.

    Raises on anything outside ``_VALID_ACTIVATIONS`` and on odd channel
    counts for the complex activations (which need interleaved real/imag
    pairs).
    """
    if name in _REAL_ACTIVATION_FACTORIES:
        return _REAL_ACTIVATION_FACTORIES[name]()
    if name in _COMPLEX_ACTIVATION_NAMES:
        if channels % 2 != 0:
            raise ValueError(
                f"Activation '{name}' requires an even number of channels "
                f"(interleaved real/imag pairs); got {channels}"
            )
        return ComplexActivation(channels // 2)
    raise ValueError(f"Unknown activation '{name}'. Valid options: {_VALID_ACTIVATIONS}")


class KSpaceUNetBlock(nn.Module):
    """Basic U-Net block for k-space processing with phase preservation via residual connections."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        padding_mode: str = "circular",
        use_batch_norm: bool = True,
        activation: str = "leaky_relu",
        use_complex_conv: bool = False,
        residual_weight: float = 0.1,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            kernel_size (int): Description.
            stride (int): Description.
            padding (int): Description.
            padding_mode (str): Description.
            use_batch_norm (bool): Description.
            activation (str): Description.
            use_complex_conv (bool): Description.
            residual_weight (float): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.residual_weight = residual_weight
        self.use_residual = residual_weight > 0 and in_channels == out_channels

        if use_complex_conv:
            # Complex convolution expects 2x channels (real+imag) or separate inputs
            # Here we assume concatenated inputs [B, 2*C, H, W]
            # Output is also [B, 2*C, H, W]
            self.conv1 = ComplexConv2d(
                in_channels // 2,
                out_channels // 2,
                kernel_size,
                stride,
                padding,
                padding_mode=padding_mode,
                bias=False,
            )
            self.conv2 = ComplexConv2d(
                out_channels // 2,
                out_channels // 2,
                kernel_size,
                stride,
                padding,
                padding_mode=padding_mode,
                bias=False,
            )
        else:
            self.conv1 = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode=padding_mode,
                bias=not use_batch_norm,
            )
            self.conv2 = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                padding_mode=padding_mode,
                bias=not use_batch_norm,
            )

        # [STABILIZATION] Disabled k-space normalization to preserve spectrum distribution
        # Standard BN/GN destroys 1/f energy distribution, flattening contrast and removing details.
        self.bn1 = self.bn2 = nn.Identity()

        self.activation = _resolve_activation(activation, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection (preserves original phase reference)
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpaceUNetBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        residual = x if self.use_residual else None

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)

        # Add scaled residual connection to preserve phase information
        # Phase = atan2(imag, real), residual preserves original phase reference
        if residual is not None:
            x = x + self.residual_weight * residual

        return x


class KSpaceDownsampleBlock(nn.Module):
    """Downsampling block for k-space U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_type: str = "self",
        activation: str = "leaky_relu",
        use_complex_conv: bool = False,
        time_embedding_dim: int | None = None,
        padding_mode: str = "circular",
        kan_dual_domain_kwargs: dict | None = None,
        *,
        feature_domain: str,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            attention_type (str): Description.
            activation (str): Description.
            use_complex_conv (bool): Description.
            time_embedding_dim (int | None): Description.
            padding_mode (str): Description.
            kan_dual_domain_kwargs (dict | None): Hyperparams forwarded to
                KANGatedDualDomainAttention when attention_type='kan_dual_domain'.
            feature_domain (str): Domain of the feature maps flowing through
                this block ("kspace" | "image"), derived from
                ``force_pure_kspace`` by ``FourierBridgeNetwork``. Forwarded
                to the domain-sensitive sub-blocks (DualDomainBlock and the
                dual-domain attention family) so their internal FFTs are
                oriented correctly. Raises on unknown values.
        """
        super().__init__()
        self._kan_dual_domain_kwargs = dict(kan_dual_domain_kwargs or {})
        self.feature_domain = validate_feature_domain(feature_domain)
        # Build-time mirror of the audit's domain-support check (the audit
        # rejects illegal YAML combos first; this guards non-YAML callers).
        supported = ATTENTION_DOMAIN_SUPPORT.get(attention_type)
        if supported is not None and self.feature_domain not in supported:
            raise ValueError(
                f"attention_type={attention_type!r} does not support "
                f"feature_domain={self.feature_domain!r} (supports: "
                f"{sorted(supported)})."
            )

        # Time embedding projection
        self.time_gain = None
        if time_embedding_dim is not None:
            self.time_gain = ComplexChannelGain(
                time_embedding_dim, out_channels, interleaved=use_complex_conv
            )

        # Adapter for DualDomainBlock channel matching
        if in_channels != out_channels:
            if use_complex_conv:
                self.adapter = ComplexConv2d(in_channels // 2, out_channels // 2, 1, bias=False)
            else:
                self.adapter = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        else:
            self.adapter = nn.Identity()

        # Dual Domain Block for processing
        self.unet_block = DualDomainBlock(
            out_channels,
            activation=_resolve_activation(activation, out_channels),
            use_complex_conv=use_complex_conv,
            time_embedding_dim=time_embedding_dim,
            feature_domain=self.feature_domain,
        )

        if use_complex_conv:
            self.downsample = nn.Sequential(
                ComplexConv2d(
                    out_channels // 2,
                    out_channels // 2,
                    3,
                    stride=1,
                    padding=1,
                    padding_mode=padding_mode,
                    bias=False,
                ),
                KSpaceCrop(scale_factor=2),
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, 1, 1),
                KSpaceCrop(scale_factor=2),
            )

        # F14 re-evaluated (2026-05-23): the prior guard raised here for
        # ``use_complex_conv`` + any real-valued attention, on the theory
        # that a true-complex tensor would hit a real bias. That premise
        # is false for this generator — ComplexConv2d (complex_conv.py:194)
        # and ComplexActivation (activation.py:44) both emit the
        # interleaved-real [B, 2C, H, W] layout, so every attention block
        # (incl. the phase-aware KAN / WaveletFreq novelty blocks) receives
        # interleaved-real and forwards cleanly. The real ``Input type
        # complex<float>`` crash lived in the separate image_cold_diffusion
        # path (ImageColdDiffusionUNet), which never uses these blocks.
        # Unsupported attention_type values still fail loud in the dispatch
        # ``else`` below. See docs/smoke_audit_20260523_fixes.rst.
        # ``zero_init_output=False``: IdentityAtInitAttention below is the ONE owner
        # of identity-at-init here, and stacking the block's own zero-init under it
        # freezes every parameter at an exact saddle (issue #471, non-negotiable 17).
        if attention_type == "self":
            self.attention = LinearAttention(
                out_channels, norm_type="instance", zero_init_output=False
            )
        elif attention_type == "kernelized":
            self.attention = KernelizedAttention(out_channels, zero_init_output=False)
        elif attention_type == "sparse":
            self.attention = SparseAttention(out_channels, zero_init_output=False)
        elif attention_type == "channel":
            # Interleaved complex features need one gain per complex channel; a
            # per-real-channel gate scales Re and Im differently and rotates phase.
            self.attention = (
                ComplexChannelAttention(out_channels)
                if use_complex_conv
                else ChannelAttention(out_channels)
            )
        elif attention_type == "dual_domain":
            self.attention = DualDomainAttention(out_channels, feature_domain=self.feature_domain)
        elif attention_type == "null_space_dual_domain":
            self.attention = NullSpaceDualDomainAttention(
                out_channels,
                num_heads=int(self._kan_dual_domain_kwargs.get("num_heads", 4)),
                feature_domain=self.feature_domain,
            )
        elif attention_type == "hermitian_null_space":
            self.attention = HermitianNullSpaceAttention(
                out_channels,
                num_features=int(self._kan_dual_domain_kwargs.get("num_features", 32)),
                feature_domain=self.feature_domain,
            )
        elif attention_type == "kan_dual_domain":
            kan_kw = dict(self._kan_dual_domain_kwargs)
            kan_kw.setdefault("num_heads", 4)
            self.attention = KANGatedDualDomainAttention(
                in_channels=out_channels,
                time_embedding_dim=time_embedding_dim,
                feature_domain=self.feature_domain,
                **kan_kw,
            )
        elif attention_type == "wavelet_freq":
            wv_kw = dict(self._kan_dual_domain_kwargs)
            wv_kw.setdefault("num_heads", 4)
            # The wavelet block doesn't use most KAN-specific knobs; filter
            # to just the ones it accepts so YAML reuse stays safe. ``gate_type``
            # + the KAN gate knobs ARE accepted now (they parameterise the
            # KAN/MLP fusion gate that makes attn_kan_wavelet != attn_mlp_wavelet).
            wv_accepted = {
                "num_heads",
                "num_levels",
                "score_fn",
                "topk_k",
                "gate_type",
                "kan_grid_size",
                "kan_spline_order",
                "kan_hidden",
            }
            wv_kw = {k: v for k, v in wv_kw.items() if k in wv_accepted}
            self.attention = WaveletFreqAttentionBlock(
                in_channels=out_channels,
                feature_domain=self.feature_domain,
                **wv_kw,
            )
        elif attention_type == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # Every block starts at an EXACT identity (rho = 1.000) so the shootout's
        # arm-vs-arm delta is the learned mechanism, not eight different effective
        # initialisations. Only 3 of the 8 blocks honoured that on their own --
        # kan_dual_domain ran 7-9x at init and this call site is REPLACE, not
        # residual. gamma=1 reproduces the raw block exactly, so nothing is lost.
        # Issue #471; see docs/attention_wiring_audit.rst.
        if not isinstance(self.attention, nn.Identity):
            self.attention = IdentityAtInitAttention(self.attention)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): Description.
            t_emb (torch.Tensor | None): Description.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Description.

        forward method for KSpaceDownsampleBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = self.adapter(x)

        # Phase-preserving multiplicative conditioning (see ComplexChannelGain).
        if self.time_gain is not None:
            x = self.time_gain(x, t_emb)

        x = self.unet_block(x, t_emb)
        skip = x  # Save for skip connection

        # IdentityAtInitAttention forwards t_emb and mask only to blocks whose
        # signature declares them, by NAME (detected once at construction). The
        # isinstance check this replaced would silently drop t_emb for any new
        # time-conditioned block, and positional arity cannot tell the two apart.
        if not isinstance(self.attention, nn.Identity):
            x = self.attention(x, t_emb, mask=mask)

        x = self.downsample(x)
        return x, skip


class KSpaceUpsampleBlock(nn.Module):
    """Upsampling block for k-space U-Net."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_type: str = "self",
        activation: str = "leaky_relu",
        use_complex_conv: bool = False,
        time_embedding_dim: int | None = None,
        padding_mode: str = "circular",
        kan_dual_domain_kwargs: dict | None = None,
        *,
        feature_domain: str,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            attention_type (str): Description.
            activation (str): Description.
            use_complex_conv (bool): Description.
            time_embedding_dim (int | None): Description.
            padding_mode (str): Description.
            kan_dual_domain_kwargs (dict | None): Hyperparams for KAN
                dual-domain attention when used.
            feature_domain (str): Domain of the feature maps flowing through
                this block ("kspace" | "image"), derived from
                ``force_pure_kspace`` by ``FourierBridgeNetwork``. Forwarded
                to the domain-sensitive sub-blocks. Raises on unknown values.
        """
        super().__init__()
        self._kan_dual_domain_kwargs = dict(kan_dual_domain_kwargs or {})
        self.feature_domain = validate_feature_domain(feature_domain)
        # Build-time mirror of the audit's domain-support check.
        supported = ATTENTION_DOMAIN_SUPPORT.get(attention_type)
        if supported is not None and self.feature_domain not in supported:
            raise ValueError(
                f"attention_type={attention_type!r} does not support "
                f"feature_domain={self.feature_domain!r} (supports: "
                f"{sorted(supported)})."
            )
        self.use_complex_conv = use_complex_conv

        if use_complex_conv:
            self.kspace_pad = KSpacePad(scale_factor=2)
            self.upsample_conv = ComplexConv2d(
                in_channels // 2,
                out_channels // 2,
                3,
                1,
                1,
                padding_mode=padding_mode,
                bias=False,
            )
        else:
            self.kspace_pad = KSpacePad(scale_factor=2)
            self.upsample_conv = nn.Conv2d(
                in_channels,
                out_channels,
                3,
                1,
                1,
                padding_mode=padding_mode,
            )

        self.time_gain = None
        if time_embedding_dim is not None:
            self.time_gain = ComplexChannelGain(
                time_embedding_dim, out_channels, interleaved=use_complex_conv
            )

        # Reduce channels after concatenation (out_channels + out_channels -> out_channels)
        if use_complex_conv:
            self.adapter = ComplexConv2d(out_channels, out_channels // 2, 1, bias=False)
        else:
            self.adapter = nn.Conv2d(out_channels * 2, out_channels, 1, bias=False)

        self.unet_block = DualDomainBlock(
            out_channels,
            activation=_resolve_activation(activation, out_channels),
            use_complex_conv=use_complex_conv,
            time_embedding_dim=time_embedding_dim,
            feature_domain=self.feature_domain,
        )

        # F14 re-evaluated (2026-05-23): mirror of the down-block change.
        # The prior guard's premise (complex tensor hits a real bias) is
        # false here — the block path is interleaved-real [B, 2C, H, W]
        # (line ~806 concatenation comment), so all attention types forward
        # cleanly. Unsupported values still fail loud in the dispatch
        # ``else`` below. See docs/smoke_audit_20260523_fixes.rst.
        # ``zero_init_output=False``: IdentityAtInitAttention below is the ONE owner
        # of identity-at-init here, and stacking the block's own zero-init under it
        # freezes every parameter at an exact saddle (issue #471, non-negotiable 17).
        if attention_type == "self":
            self.attention = LinearAttention(
                out_channels, norm_type="instance", zero_init_output=False
            )
        elif attention_type == "kernelized":
            self.attention = KernelizedAttention(out_channels, zero_init_output=False)
        elif attention_type == "sparse":
            self.attention = SparseAttention(out_channels, zero_init_output=False)
        elif attention_type == "channel":
            # Interleaved complex features need one gain per complex channel; a
            # per-real-channel gate scales Re and Im differently and rotates phase.
            self.attention = (
                ComplexChannelAttention(out_channels)
                if self.use_complex_conv
                else ChannelAttention(out_channels)
            )
        elif attention_type == "spatial":
            self.attention = CBAMSpatialAttention(
                7
            )  # Standardize on 7x7 CBAM for spatial consistency
        elif attention_type == "dual_domain":
            self.attention = DualDomainAttention(out_channels, feature_domain=self.feature_domain)
        elif attention_type == "null_space_dual_domain":
            self.attention = NullSpaceDualDomainAttention(
                out_channels,
                num_heads=int(self._kan_dual_domain_kwargs.get("num_heads", 4)),
                feature_domain=self.feature_domain,
            )
        elif attention_type == "hermitian_null_space":
            self.attention = HermitianNullSpaceAttention(
                out_channels,
                num_features=int(self._kan_dual_domain_kwargs.get("num_features", 32)),
                feature_domain=self.feature_domain,
            )
        elif attention_type == "kan_dual_domain":
            kan_kw = dict(self._kan_dual_domain_kwargs)
            kan_kw.setdefault("num_heads", 4)
            self.attention = KANGatedDualDomainAttention(
                in_channels=out_channels,
                time_embedding_dim=time_embedding_dim,
                feature_domain=self.feature_domain,
                **kan_kw,
            )
        elif attention_type == "wavelet_freq":
            wv_kw = dict(self._kan_dual_domain_kwargs)
            wv_kw.setdefault("num_heads", 4)
            # The wavelet block doesn't use most KAN-specific knobs; filter
            # to just the ones it accepts so YAML reuse stays safe. ``gate_type``
            # + the KAN gate knobs ARE accepted now (they parameterise the
            # KAN/MLP fusion gate that makes attn_kan_wavelet != attn_mlp_wavelet).
            wv_accepted = {
                "num_heads",
                "num_levels",
                "score_fn",
                "topk_k",
                "gate_type",
                "kan_grid_size",
                "kan_spline_order",
                "kan_hidden",
            }
            wv_kw = {k: v for k, v in wv_kw.items() if k in wv_accepted}
            self.attention = WaveletFreqAttentionBlock(
                in_channels=out_channels,
                feature_domain=self.feature_domain,
                **wv_kw,
            )
        elif attention_type == "none":
            self.attention = nn.Identity()
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")

        # Every block starts at an EXACT identity (rho = 1.000) so the shootout's
        # arm-vs-arm delta is the learned mechanism, not eight different effective
        # initialisations. Only 3 of the 8 blocks honoured that on their own --
        # kan_dual_domain ran 7-9x at init and this call site is REPLACE, not
        # residual. gamma=1 reproduces the raw block exactly, so nothing is lost.
        # Issue #471; see docs/attention_wiring_audit.rst.
        if not isinstance(self.attention, nn.Identity):
            self.attention = IdentityAtInitAttention(self.attention)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """forward.

        Args:
            x (torch.Tensor): Description.
            skip (torch.Tensor): Description.
            t_emb (torch.Tensor | None): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpaceUpsampleBlock.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            skip (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_emb (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        x = self.kspace_pad(x, target_shape=skip.shape[-2:])
        x = self.upsample_conv(x)

        # IdentityAtInitAttention forwards t_emb and mask only to blocks whose
        # signature declares them, by NAME (detected once at construction).
        if not isinstance(self.attention, nn.Identity):
            x = self.attention(x, t_emb, mask=mask)

        # Concatenate skip connection
        # Both x and skip are Interleaved [R1, I1, R2, I2, ...]
        # Standard concatenation along dim=1 preserves the interleaved layout for multi-coil data.
        x = torch.cat([x, skip], dim=1)

        # Reduce and process
        x = self.adapter(x)

        # Phase-preserving multiplicative conditioning (see ComplexChannelGain).
        if self.time_gain is not None:
            x = self.time_gain(x, t_emb)

        x = self.unet_block(x, t_emb)

        return x


from spectramr.models.reconstruction.unet import (
    AttentionType,
    BlockType,
    NormalizationType,
    UNet,
    UNetConfig,
)


class FourierBridgeNetwork(nn.Module):
    """
    Dual-Domain Bridge Network.

    Bridging the inductive bias gap by:
    1. Transforming K-Space -> Image Space (iFFT)
    2. applying CNN in Image Space (Local features)
    3. Transforming Image Space -> K-Space (FFT)
    """

    def __init__(
        self,
        config: UNetConfig,
        backbone_type: str = "unet",
        force_pure_kspace: bool = False,
        time_embedding_dim: int = 256,
        **backbone_kwargs,
    ):
        """__init__.

        Args:
            config (UNetConfig): Description.
            backbone_type (str): Description.
            force_pure_kspace (bool): Description.
        """
        super().__init__()
        self.config = config
        self.force_pure_kspace = force_pure_kspace
        self.backbone_type = backbone_type
        # Feature domain fed to the backbone: k-space when we skip the entry
        # iFFT (pure-k-space mode), image otherwise. This is the SSOT that the
        # domain-aware attention blocks consume — derived here, never a YAML
        # knob of its own (no unread-knob surface, pitfall #15).
        self.feature_domain = "kspace" if force_pure_kspace else "image"

        # Add RepetitionFusion for 5D data
        from spectramr.models.blocks.pseudo_3d_kspace import RepetitionFusion

        self.rep_fusion = RepetitionFusion(channels=config.in_channels, num_reps=18)

        # Backbones that support time conditioning
        self.time_conditioned_backbones = {
            "unet",
            "nafnet",
            "swin_diff_rec",
            "diff_varnet",
            "swin_diff_rec_kan",
            "diff_varnet_kan",
            "mamba_unet",
            "vision_mamba",
            "complex_unet",
            # The registry trunks all take ``timesteps``; a backbone that did not
            # would simply never see the key, which is the facade this set exists
            # to prevent.
            *registered_backbone_names(),
        }

        # Select backbone architecture
        # Prepare kwargs by removing explicit args to avoid collision
        common_filtered_keys = ["dropout", "img_size", "image_size"]
        base_clean_kwargs = {
            k: v for k, v in backbone_kwargs.items() if k not in common_filtered_keys
        }

        # ``kspace_feature_norm`` and the radial-band-token knobs are gated by
        # :func:`backbone_kwargs_defects`, which runs before EITHER backbone branch;
        # a gate sited here missed the PureKSpaceUNet one entirely. A direct caller
        # of this class is a Python caller, not a config, and carries its own kwargs.
        if backbone_type != "complex_unet":
            base_clean_kwargs.pop("kspace_feature_norm", None)

        # Standardize image_size from kwargs if present
        img_size = (
            backbone_kwargs.get("img_size") or backbone_kwargs.get("image_size") or (256, 256)
        )

        if self.force_pure_kspace:
            # Force complex convolutions for backbones that support it (UNet, NAFNet, SwinDiffRec, MambaUNet)
            # unless explicitly disabled by user (which would be weird for pure kspace but allowed?)
            # Let's default to True.
            if "use_complex_conv" not in base_clean_kwargs:
                base_clean_kwargs["use_complex_conv"] = True

        if backbone_type == "swin_diff_rec":
            from spectramr.models.generators.swin_diff_rec import SwinDiffRec

            self.backbone = SwinDiffRec(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                image_size=img_size,
                base_channels=config.features[0],
                dropout=config.dropout,
                **base_clean_kwargs,
            )
        elif backbone_type == "diff_varnet":
            from spectramr.models.generators.diff_varnet import DiffVarNet

            self.backbone = DiffVarNet(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                image_size=img_size,
                base_channels=config.features[0],
                dropout=config.dropout,
                **base_clean_kwargs,
            )
        elif backbone_type == "swin_diff_rec_kan":
            from spectramr.models.generators.swin_diff_rec_kan import SwinDiffRecKAN

            self.backbone = SwinDiffRecKAN(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                image_size=img_size,
                base_channels=config.features[0],
                dropout=config.dropout,
                **base_clean_kwargs,
            )
        elif backbone_type == "diff_varnet_kan":
            from spectramr.models.generators.diff_varnet_kan import DiffVarNetKAN

            self.backbone = DiffVarNetKAN(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                image_size=img_size,
                base_channels=config.features[0],
                dropout=config.dropout,
                **base_clean_kwargs,
            )
        elif backbone_type == "nafnet":
            from spectramr.models.generators.nafnet_generator import NAFNetGenerator

            # NAFNet takes specific args, clean duplicates
            # Note: NAFNet calculates time_embedding_dim internally as width * 4
            naf_kwargs = {
                k: v
                for k, v in base_clean_kwargs.items()
                if k not in ["time_embedding_dim", "width"]
            }
            self.backbone = NAFNetGenerator(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                width=config.features[0],
                # NAFNet conditions at width*4, which is NOT the width the
                # generator builds ``contrast_embedding`` at. Declare the
                # incoming width so it can project instead of shape-erroring on
                # the first contrast-conditioned forward (48*4=192 vs 256 on
                # experiment_11e_nafnet). Read off ``config.time_dim``, which the
                # outer generator already sets from time_embedding_dim -- that
                # kwarg itself is consumed upstream and never reaches
                # backbone_kwargs, so reading it from there returns None.
                contrast_emb_dim=getattr(config, "time_dim", None),
                **naf_kwargs,
            )
        elif backbone_type == "vision_mamba" or backbone_type == "mamba_unet":
            from spectramr.models.generators.mamba_unet import MambaUNet

            # Extract Mamba-specific parameters and put them in mamba_config dict
            mamba_depth = base_clean_kwargs.pop("mamba_depth", 4)
            mamba_hidden_dim = base_clean_kwargs.pop("mamba_hidden_dim", 128)
            mamba_config = base_clean_kwargs.pop("mamba_config", {})

            # Merge depth and hidden_dim into mamba_config
            mamba_config.setdefault("d_state", mamba_hidden_dim)
            mamba_config.setdefault("expand", 2)

            # Add time_embedding_dim for diffusion support
            time_emb_dim = base_clean_kwargs.pop("time_embedding_dim", 256)

            self.backbone = MambaUNet(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                features=list(config.features),  # Convert tuple to list
                mamba_config=mamba_config,
                time_embedding_dim=time_emb_dim,
                **base_clean_kwargs,
            )
        elif backbone_type == "vision_transformer" or backbone_type == "vit":
            from spectramr.models.generators.vision_transformer import (
                VisionTransformer,
            )

            # VisionTransformer expects int img_size, not tuple
            img_size_int = img_size[0] if isinstance(img_size, (tuple, list)) else img_size

            self.backbone = VisionTransformer(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                img_size=img_size_int,
                patch_size=base_clean_kwargs.pop("patch_size", 16),
                embed_dim=config.features[0],
                depth=config.depth,
                num_heads=base_clean_kwargs.pop("num_heads", 8),
                dropout=config.dropout,
                **base_clean_kwargs,
            )
        elif backbone_type == "swin_transformer":
            from spectramr.models.generators.swin_transformer_generator import (
                SwinTransformerGenerator,
            )

            # SwinTransformer expects int img_size, not tuple
            img_size_int = img_size[0] if isinstance(img_size, (tuple, list)) else img_size

            embed_dim = config.features[0]
            h0 = max(1, embed_dim // 16)
            default_heads = (h0, h0 * 2, h0 * 4, h0 * 8)

            self.backbone = SwinTransformerGenerator(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                img_size=img_size_int,
                embed_dim=embed_dim,
                depths=base_clean_kwargs.pop("depths", (2, 2, 6, 2)),
                num_heads=base_clean_kwargs.pop("num_heads", default_heads),
                window_size=base_clean_kwargs.pop("window_size", 7),
                **base_clean_kwargs,
            )
        elif backbone_type == "restormer":
            from spectramr.models.generators.restormer_generator import RestormerGenerator

            # Extract only kwargs that RestormerGenerator accepts
            restormer_kwargs = {k: v for k, v in base_clean_kwargs.items() if k in ["scale"]}
            self.backbone = RestormerGenerator(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                dim=config.features[0],
                num_blocks=base_clean_kwargs.pop("num_blocks", 6),
                num_refinement_blocks=base_clean_kwargs.pop("num_refinement_blocks", 6),
                heads=base_clean_kwargs.pop("heads", 8),
                ffn_expansion_factor=base_clean_kwargs.pop("ffn_expansion_factor", 2.66),
                bias=base_clean_kwargs.pop("bias", False),
                **restormer_kwargs,
            )
        elif backbone_type == "swinir":
            from spectramr.models.generators.swinir_generator import SwinIRGenerator

            # SwinIR expects int img_size, not tuple
            img_size_int = img_size[0] if isinstance(img_size, (tuple, list)) else img_size

            embed_dim = config.features[0]
            h0 = max(1, embed_dim // 16)
            depths = tuple(base_clean_kwargs.pop("depths", (6, 6, 6, 6)))

            self.backbone = SwinIRGenerator(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                in_chans=config.in_channels,
                img_size=img_size_int,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=base_clean_kwargs.pop("num_heads", (h0,) * len(depths)),
                window_size=base_clean_kwargs.pop("window_size", 8),
                **base_clean_kwargs,
            )
        elif backbone_type == "unet":
            self.backbone = UNet(config)
        elif backbone_type == "complex_unet":
            from spectramr.models.generators.complex_unet import ComplexUNet

            padding_mode = base_clean_kwargs.pop("padding_mode", None)
            if padding_mode is None and self.force_pure_kspace:
                padding_mode = (
                    "circular"  # 🧲 PHYSICS FIX: Circular is required for k-space periodicity
                )
            padding_mode = padding_mode or "circular"

            # ComplexUNet expects slightly different args. feature_domain is
            # passed EXPLICITLY here (not via base_clean_kwargs) so it reaches
            # only the backbone that consumes it — putting it in the shared
            # kwargs would break the other backbones (pitfall #16).
            self.backbone = ComplexUNet(
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                features=config.features,
                time_embedding_dim=time_embedding_dim,
                img_size=img_size,
                padding_mode=padding_mode,
                feature_domain=self.feature_domain,
                **base_clean_kwargs,
            )
        else:
            # Anything the chain above does not name is a registered builder or
            # nothing. Replacing the old hand-written "Supported types:" string
            # with a derived one also fixes a list that had already drifted off
            # the branches beside it.
            self.backbone = build_backbone(
                backbone_type,
                BackboneBuildContext(
                    in_channels=config.in_channels,
                    out_channels=config.out_channels,
                    features=tuple(config.features),
                    img_size=tuple(img_size) if isinstance(img_size, (tuple, list))
                    else (img_size, img_size),
                    feature_domain=self.feature_domain,
                    # The contrast embedding is built at ``time_embedding_dim``
                    # while a transformer backbone runs at its own ``dim``, and
                    # the backbones gated conditioning on those two being equal
                    # -- structurally false, so the FiLM silently never fired.
                    contrast_emb_dim=time_embedding_dim,
                    kwargs=dict(base_clean_kwargs),
                ),
                also_supported=_CHAIN_BACKBONE_TYPES,
            )

        # ✅ Create channel adapter for flexible input handling
        # Initialized on first forward pass when we know input channels
        self.channel_adapter: ChannelAdapter | None = None
        self._channel_adapter_initialized = False

        # ✅ Initialize phase-safe dual attention for bottleneck fusion (Phase 5: config-driven)
        # Queries from image-space features, guided by k-space magnitude
        # Only initialize if attention_type is 'dual_domain' in model_kwargs.
        # NOTE: this bridge-level attention is already domain-correct — it
        # re-derives the spatial query via ifft2c under force_pure_kspace in
        # forward() (see the phase_safe_attention block below). It is
        # independent of the in-ComplexUNet block-level feature_domain wiring.
        attention_type = backbone_kwargs.get("attention_type")
        if attention_type == "dual_domain":
            # Phase 5: Get optional attention config from model_kwargs
            # For complex k-space with in_channels=2 (real, imag),
            # PhaseSafeDualAttention expects in_channels=1 (one complex channel)
            num_heads = backbone_kwargs.get("phase_safe_attention_num_heads", 1)
            reduction = backbone_kwargs.get("phase_safe_attention_reduction", 1)
            # in_channels should be config.out_channels // 2 for complex data (1 complex = 2 real)
            attention_in_channels = max(1, config.out_channels // 2)
            self.phase_safe_attention = PhaseSafeDualAttention(
                in_channels=attention_in_channels,
                num_heads=num_heads,
                reduction=reduction,
            )
        else:
            self.phase_safe_attention = None

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Forward the request to the wrapped backbone.

        Raises rather than degrading when the selected backbone has no
        implementation (non-negotiable 3 / pitfall #9): gradient checkpointing is
        a *memory* claim, and an arm that asked for it and silently did not get
        it OOMs later with no indication of why. Callers who want the arm to run
        without it must say so with ``enable_checkpointing: false``.

        Args:
            enable: Whether the backbone should checkpoint its blocks.

        Raises:
            NotImplementedError: If the configured backbone cannot checkpoint.
        """
        setter = getattr(self.backbone, "set_grad_checkpointing", None)
        if setter is None:
            raise NotImplementedError(
                f"backbone_type={self.backbone_type!r} "
                f"({type(self.backbone).__name__}) does not implement "
                "set_grad_checkpointing, so optimization.gradient."
                "enable_checkpointing cannot be honored. Use "
                "backbone_type='complex_unet', or set enable_checkpointing: "
                "false for this arm."
            )
        setter(enable)

    def forward(
        self,
        kspace_input: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            kspace_input: (B, C, H, W) or (B, C, D, H, W) - Assumes C=2 (Real, Imag) or C is even.
            timesteps: (B,) time indices.
            **kwargs: Additional arguments (filtered to avoid passing incompatible params to UNet).

        Returns:
            Same shape as input (kspace_input)
        """
        # Real [N, K, H, W] planes the generator built; never forwarded to the backbone.
        geometry = kwargs.pop("geometry", None)

        # ✅ Validate input shape
        input_is_5d_flag, original_shape_check = validate_5d_or_4d_tensor(
            kspace_input, name="FourierBridgeNetwork input"
        )

        # SAFETY: Squeeze all trailing singleton dimensions to handle TorchIO tensors
        # (B, C, H, W, 1) or (B, C, H, W, 1, 1) -> (B, C, H, W)
        while kspace_input.dim() > 5 or (kspace_input.dim() == 5 and kspace_input.shape[-1] == 1):
            if kspace_input.shape[-1] == 1:
                kspace_input = kspace_input.squeeze(-1)
            else:
                break

        # Store original shape for later restoration
        original_shape = kspace_input.shape

        # Handle both 4D (2D slices) and 5D (3D volumes) inputs
        # [FIX] TorchIO patch_size [H, W, 1] produces 5D tensors with singleton depth.
        # Squeeze singleton depth BEFORE volumetric processing to avoid returning
        # 5D output [B, C, H, W, 1] when the downstream pipeline expects 4D.
        if kspace_input.dim() == 5 and kspace_input.shape[-1] == 1:
            kspace_input = kspace_input.squeeze(-1)  # (B, C, H, W, 1) -> (B, C, H, W)

        # Squeeze kwargs tensors as well
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                while v.dim() > 5 or (v.dim() == 5 and v.shape[-1] == 1):
                    if v.shape[-1] == 1:
                        v = v.squeeze(-1)
                    else:
                        break
                kwargs[k] = v

        input_is_5d = kspace_input.dim() == 5
        original_batch_size = kspace_input.shape[0]

        # Use RepetitionFusion for 5D data instead of flattening
        if input_is_5d:
            B, C, H, W, D = kspace_input.shape  # TorchIO format: depth is last

            # For volumetric TorchIO data, flatten depth into batch dimension
            # [B, C, H, W, D] -> [B*D, C, H, W]
            kspace_input = kspace_input.permute(0, 4, 1, 2, 3).reshape(B * D, C, H, W)

            # Now kspace_input is (B*D, C, H, W)

        # 1. PHYSICS TRANSFORM: k-Space -> Image Space (skipping if force_pure_kspace)
        # Check if input is complex tensor or stacked real
        input_was_complex = torch.is_complex(kspace_input)

        # Guard against single-channel magnitudes:
        # If C=1, zero-pad imaginary channel BEFORE any branching so the
        # `kspace_input.shape[1] == 2` fast-path is taken and no odd-channel
        # validation guard triggers downstream.
        if not torch.is_complex(kspace_input) and kspace_input.shape[1] == 1:
            kspace_input = torch.cat([kspace_input, torch.zeros_like(kspace_input)], dim=1)

        if torch.is_complex(kspace_input):
            kspace_complex = kspace_input
        else:
            # Assumes stacked real/imag (B, 2*N, H, W) -> (B, N, H, W) complex
            # If C=2, B,2,H,W -> B,H,W complex
            if kspace_input.shape[1] == 2:
                kspace_complex = torch.complex(kspace_input[:, 0], kspace_input[:, 1])
            else:
                # Group groups of 2
                # CRITICAL: At this point kspace_input MUST be 4D (B, C, H, W)
                if kspace_input.dim() != 4:
                    raise ValueError(
                        f"Expected 4D tensor for view_as_complex, got {kspace_input.dim()}D "
                        f"with shape {kspace_input.shape}. "
                        f"Channels: {kspace_input.shape[1] if kspace_input.dim() >= 2 else 'N/A'}"
                    )

                # Validate that we have even number of channels for complex conversion
                if kspace_input.shape[1] % 2 != 0:
                    if kspace_input.shape[1] == 1:
                        # Automatically pad single-channel inputs (magnitude/real)
                        # with a zero imaginary channel for complex conversion
                        kspace_input = torch.cat(
                            [kspace_input, torch.zeros_like(kspace_input)], dim=1
                        )
                    else:
                        # [FIX] Odd channels (e.g., C=3 from degradation mask concat):
                        # Zero-pad to next even number for complex conversion
                        logger.debug(
                            f"[FourierBridgeNetwork] Odd channels C={kspace_input.shape[1]}, "
                            f"zero-padding to C={kspace_input.shape[1] + 1}"
                        )
                        pad_ch = torch.zeros(
                            kspace_input.shape[0],
                            1,
                            *kspace_input.shape[2:],
                            device=kspace_input.device,
                            dtype=kspace_input.dtype,
                        )
                        kspace_input = torch.cat([kspace_input, pad_ch], dim=1)

                # Additional validation: check that after permute, last dimension is 2
                permuted = kspace_input.permute(0, 2, 3, 1).contiguous()
                if permuted.shape[-1] % 2 != 0:
                    raise ValueError(
                        f"After permute, last dimension must be even (for complex pairs). "
                        f"Got shape {permuted.shape}. Original shape: {kspace_input.shape}"
                    )

                # Reshape to isolate complex pairs: (B, H, W, C) -> (B, H, W, C//2, 2)
                # This ensures view_as_complex always sees a last dimension of 2
                B_dim, H_dim, W_dim, C_dim = permuted.shape
                permuted_reshaped = permuted.view(B_dim, H_dim, W_dim, C_dim // 2, 2)

                kspace_input_view = torch.view_as_complex(permuted_reshaped)  # (B, H, W, C/2)
                kspace_complex = kspace_input_view.permute(0, 3, 1, 2)  # (B, C/2, H, W)

        if self.force_pure_kspace:
            # SKIP FFT: Treat K-Space as the domain for CNN
            image_space_complex = kspace_complex
        else:
            # Use CENTERED inverse FFT (matches DualDomainBlock)
            # This handles fftshift/ifftshift internally
            image_space_complex = ifft2c(kspace_complex)

        # 2. CHANNEL ADAPTATION
        # Convert to interleaved real for CNN: (B, C_complex, H, W) -> (B, 2*C_complex, H, W)
        if image_space_complex.dim() == 3:  # (B, H, W) single complex channel
            # Stack real and imag as channel dim: (B, 1, 2, H, W) -> (B, 2, H, W) interleaved
            real = image_space_complex.real.unsqueeze(1)  # (B, 1, H, W)
            imag = image_space_complex.imag.unsqueeze(1)  # (B, 1, H, W)
            image_input = torch.cat([real, imag], dim=1)  # (B, 2, H, W)
        else:
            # (B, C_complex, H, W) complex tensor
            # Interleave real and imag: (B, 2*C_complex, H, W)
            # Stack real/imag along channel, then permute to interleave: [R1, I1, R2, I2, ...]
            B, C, H, W = image_space_complex.shape
            real = image_space_complex.real  # (B, C, H, W)
            imag = image_space_complex.imag  # (B, C, H, W)
            # Stack to (B, C, 2, H, W), then permute to (B, 2*C, H, W)
            stacked = torch.stack([real, imag], dim=2)  # (B, C, 2, H, W)
            image_input = stacked.permute(0, 1, 2, 3, 4).contiguous()
            image_input = image_input.view(B, C * 2, H, W)  # (B, 2*C, H, W)

        # Appended AFTER the entry transform: the planes describe the k-space grid
        # in either domain, and joining as (plane, 0) pairs keeps the Re/Im pairing
        # of every channel before them.
        geometry_as_gain = getattr(self.backbone, "geometry_gain", None) is not None
        if geometry is not None:
            if geometry.shape[0] != image_input.shape[0]:
                raise ValueError(
                    f"[FourierBridgeNetwork] geometry batch {geometry.shape[0]} does not "
                    f"match the backbone input batch {image_input.shape[0]}."
                )
        if geometry is not None and geometry_as_gain:
            kwargs["geometry"] = geometry
        elif geometry is not None:
            image_input = torch.cat(
                [image_input, as_interleaved_complex(geometry).to(image_input.dtype)], dim=1
            )

        # 3. BACKBONE PROCESSING
        # [WIDTH CONTRACT] The backbone was built for exactly
        # ``config.in_channels``. Any other width is a caller bug and MUST
        # raise here rather than be coerced (CLAUDE.md #3, pitfalls #9/#16).
        #
        # This block used to *rebuild* ``self.channel_adapter`` for whatever
        # width arrived — "handles training vs validation using different
        # repetition-fusion paths" — and the projection it built was never
        # "learned". ``self.channel_adapter`` starts as ``None`` in
        # ``__init__`` and is constructed ONLY here, i.e. after the optimizer
        # captured ``model.parameters()``, so a non-Identity adapter carries
        # random weights that receive no update, enter no checkpoint, and are
        # redrawn on every width flip. #1326 fixed one caller
        # (ColdDiffusionInferenceStrategy); the training and multi-step
        # validation callers kept flipping 16ch <-> 8ch against each other, so
        # experiment_11_attention_none sampled its whole reverse loop through a
        # fresh random 1x1 projection — two validations of identical weights on
        # identical input disagreed by 137%.
        actual_in_ch = image_input.shape[1]
        if actual_in_ch != self.config.in_channels:
            raise ValueError(
                f"[FourierBridgeNetwork] Received a {actual_in_ch}-channel "
                f"input but this network was built for "
                f"{self.config.in_channels}. Coercing the width here would "
                "build an untrained 1x1 ChannelAdapter and silently degrade "
                "the output instead of failing (#1326). On a cold-diffusion "
                "sampling path this almost always means the S-maps did not "
                "reach the generator: pass `smaps=` (or `sensitivity_maps=`) "
                "to `sample()`/`forward()` so it can concatenate them, or "
                "Resolve the caller's width through "
                "`model_expects_smaps_concat(model)` rather than the arm's "
                "`condition_with_smaps` declaration -- an internal-DC backbone "
                "honours the declaration and is still built at 1x. If the arm "
                "genuinely should not be conditioned, set "
                "`condition_with_smaps: false`."
            )
        if self.channel_adapter is None:
            # Width matches, so this is an Identity pass-through. It is kept as
            # a real submodule for module-tree and ``.to()`` compatibility.
            self.channel_adapter = ChannelAdapter(
                in_channels=actual_in_ch, out_channels=self.config.in_channels
            ).to(image_input.device)
            self._channel_adapter_in_channels = actual_in_ch
            self._channel_adapter_initialized = True

        image_input = self.channel_adapter(image_input)

        # Pass timesteps explicitly with proper parameter name for backbones that support it
        # Filter out any conflicting 'time' from kwargs to avoid duplicate parameter error
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "time" and k != "timesteps"}

        # Only pass time parameters to backbones that support them
        kwargs_to_pass = filtered_kwargs.copy()
        if self.backbone_type in self.time_conditioned_backbones:
            kwargs_to_pass["time"] = timesteps
            kwargs_to_pass["timesteps"] = timesteps

        import inspect

        try:
            allowed_kwargs = inspect.signature(self.backbone.forward).parameters
            if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed_kwargs.values()):
                kwargs_to_pass = {k: v for k, v in kwargs_to_pass.items() if k in allowed_kwargs}
        except Exception:
            pass
        image_output = self.backbone(image_input, **kwargs_to_pass)

        # Handle backbones that return tuples (e.g. NAFNet, or models with auxiliary outputs)
        if isinstance(image_output, tuple):
            image_output = image_output[0]

        # DEBUG: Validate UNet output shape
        expected_channels = self.config.out_channels
        if image_output.shape[1] != expected_channels:
            logger.debug(
                f"WARNING: UNet output has {image_output.shape[1]} channels, expected {expected_channels}. Shape: {image_output.shape}"
            )
            logger.debug(
                f"DEBUG: config.in_channels={self.config.in_channels}, config.out_channels={self.config.out_channels}"
            )

        # Phase-safe dual attention refines the BACKBONE's prediction
        # (``image_output``), using an image-domain query for guidance.
        #
        # ``PhaseSafeDualAttention`` returns ``x_kspace + gamma * out`` with
        # ``gamma`` initialised to 0, so whatever is passed as ``x_kspace`` (its
        # value + residual base) is what the model emits at initialisation and
        # what dominates early training. Two requirements, both load-bearing for
        # avoiding the centre "DC blob" artefact (smoke triage of
        # experiment_11_kspace_cold_diffusion, 2026-05-26):
        #
        #   1. ``x_kspace`` MUST be the backbone prediction ``image_output`` —
        #      NOT the raw input k-space. Passing the input k-space there made
        #      the model an identity passthrough of the undersampled input
        #      (whose IFFT is a bright DC blob) and discarded the backbone output
        #      entirely — it only steered the attention weights via the query.
        #      This now matches the no-attention path (``= image_output``).
        #   2. The query ``x_image`` MUST be genuine image-domain data. Under
        #      ``force_pure_kspace`` the (misnamed) ``image_space_complex`` is
        #      STILL k-space, so it is ``ifft2c``-ed here; otherwise it is
        #      already the spatial-domain inverse FFT.
        attentioned_output = image_output
        if self.phase_safe_attention is not None and image_output.shape[1] >= 4:
            spatial_complex = (
                ifft2c(kspace_complex) if self.force_pure_kspace else image_space_complex
            )
            if torch.is_complex(spatial_complex):
                # Interleaved [R1,I1,R2,I2,...], the layout this module and
                # ``PhaseSafeDualAttention`` (which reads ``0::2``/``1::2``)
                # both use. A blocked [Re...,Im...] stack truncates to the value
                # width by keeping every real part and dropping every imaginary
                # one, which halves the guidance without changing a shape.
                guidance = complex_to_interleaved(spatial_complex)
            else:
                guidance = spatial_complex

            # The query projection expects the same stacked-channel count as the
            # value tensor (``image_output``); truncation now drops whole
            # complex pairs rather than splitting real from imaginary.
            if guidance.shape[1] < image_output.shape[1]:
                pad_size = image_output.shape[1] - guidance.shape[1]
                guidance = F.pad(guidance, (0, 0, 0, 0, 0, pad_size), mode="replicate")
            elif guidance.shape[1] > image_output.shape[1]:
                guidance = guidance[:, : image_output.shape[1]]

            # x_kspace=image_output (value + residual base), x_image=guidance
            # (query). At init (gamma=0) this returns image_output, consistent
            # with the no-attention path above.
            attentioned_output = self.phase_safe_attention(image_output, guidance)

        # 4. PHYSICS TRANSFORM: Image Space -> k-Space (skipping if force_pure_kspace)
        # Convert back to complex (Interleaved parsing)
        # Guard against odd-channel inputs (CLAUDE.md #9): the 0::2/1::2
        # slicing of an odd-channel tensor produces a zero-channel imag
        # part, which collapses ``output_complex`` to shape [B, 0, H, W].
        # The downstream ``fft2c`` then raises ``cuFFT_INVALID_SIZE`` —
        # the failure mode that took out four cold-diffusion arms in the
        # 2026-05-10 cluster rerun. Refuse loudly here so the YAML mis-
        # configuration is obvious.
        if attentioned_output.shape[1] % 2 != 0:
            raise ValueError(
                "[KSpaceColdDiffusionGenerator] interleaved Re/Im layout "
                f"requires an even channel count but got {attentioned_output.shape}. "
                "Set in_channels and out_channels to even values (e.g. 2 for a "
                "single complex coil real-stacked)."
            )
        real = attentioned_output[:, 0::2, ...]
        imag = attentioned_output[:, 1::2, ...]
        output_complex = torch.complex(real, imag)

        if self.force_pure_kspace:
            # SKIP FFT
            kspace_prediction = output_complex
        else:
            # FFT back to k-space (CENTERED)
            kspace_prediction = fft2c(output_complex)

        # Return in same format as input (Stack Real/Imag or Complex)
        # Match the original kspace_input shape

        # Use original_shape[1] to handle both 4D and 5D cases correctly
        original_channels = original_shape[1]

        if input_was_complex:
            kspace_out = kspace_prediction
        else:
            # Squeeze trailing singleton dims (TorchIO/3D backbones may add them)
            while kspace_prediction.ndim > 4 and kspace_prediction.shape[-1] == 1:
                kspace_prediction = kspace_prediction.squeeze(-1)
            if kspace_prediction.ndim == 5:
                # Volumetric: flatten depth into batch for conversion, restore below
                kp_B, kp_C, kp_H, kp_W, kp_D = kspace_prediction.shape
                kspace_prediction = kspace_prediction.permute(0, 4, 1, 2, 3).reshape(
                    kp_B * kp_D, kp_C, kp_H, kp_W
                )
            B_out, C_out, H_out, W_out = kspace_prediction.shape
            # Interleave real and imag: [R1, I1, R2, I2, ...]
            kspace_out = torch.stack([kspace_prediction.real, kspace_prediction.imag], dim=2)
            kspace_out = kspace_out.reshape(B_out, C_out * 2, H_out, W_out)

            # Return native out_channels shape — do NOT zero-pad to match input.
            # The loss and validation pipelines handle channel mismatches via
            # truncation (see diffusion.py _extract_and_fix_output).
            # Zero-padding would inject zeros that corrupt ifft_magnitude output.
            expected_out = self.config.out_channels
            if kspace_out.shape[1] > expected_out:
                # Truncate excess channels (e.g. backbone returned more than expected)
                kspace_out = kspace_out[:, :expected_out]

        # ✅ Restore 5D shape if input was 5D
        if input_is_5d:
            # kspace_out is (B*D, C, H, W), reshape back to TorchIO format (B, C, H, W, D)
            B, C, H, W, D = original_shape
            _, C_out, H_out, W_out = kspace_out.shape

            # Reshape from (B*D, C_out, H_out, W_out) to (B, D, C_out, H_out, W_out)
            kspace_out = kspace_out.reshape(B, D, C_out, H_out, W_out)

            # Permute from (B, D, C, H, W) to (B, C, H, W, D) [TorchIO format]
            kspace_out = kspace_out.permute(0, 2, 3, 4, 1)

            # Validate spatial dimensions and batch match (channels may differ)
            assert kspace_out.shape[0] == B and kspace_out.shape[-1] == D, (
                f"Output shape {kspace_out.shape} spatial mismatch with input shape {original_shape}"
            )

        return kspace_out

    def to(self, *args: object, **kwargs: object) -> "FourierBridgeNetwork":
        """Override `to()` to handle channel adapter device transfer."""
        result = super().to(*args, **kwargs)
        # Move channel adapter to the same device if it exists
        if self.channel_adapter is not None:
            self.channel_adapter = self.channel_adapter.to(*args, **kwargs)
        return result


class PureKSpaceUNet(nn.Module):
    """Pure K-Space Complex U-Net backbone with time conditioning.

    Now supports time embedding for diffusion models.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: tuple[int, ...] = (64, 128, 256, 512),
        time_embedding_dim: int = 256,
        num_bottleneck_reflect_pad_layers: int = 2,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            features (tuple[int, ...]): Description.
            time_embedding_dim (int): Description.
            num_bottleneck_reflect_pad_layers (int): Description.
        """
        super().__init__()
        from spectramr.models.layers.complex_conv import ComplexConv2d

        self.encoder = nn.ModuleList()
        self.pooling = KSpaceCrop(scale_factor=2)
        self.time_embedding_dim = time_embedding_dim

        if in_channels % 2 != 0:
            raise ValueError(
                f"PureKSpaceUNet expects even input channels (Real+Imag), got {in_channels}"
            )

        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        # Time projection for each encoder level
        self.time_projs = nn.ModuleList()

        cin = in_channels // 2

        # Determine bottleneck threshold (Phase 5: config-driven)
        # Use num_bottleneck_reflect_pad_layers to determine which layers get reflect padding
        # e.g., if num_bottleneck_reflect_pad_layers=2 and features has 4 items,
        # apply reflect padding to indices 2 and 3 (the last 2 layers)
        bottleneck_start_idx = max(0, len(features) - num_bottleneck_reflect_pad_layers)

        for idx, f in enumerate(features):
            # Use ComplexConv2d with circular padding to prevent boundary ringing in k-space
            conv_class = ComplexConv2d

            self.encoder.append(
                nn.Sequential(
                    conv_class(cin, f, 3, padding=1, padding_mode="circular", bias=False),
                    nn.Identity(),  # NO normalization in k-space: frequency bins are orthogonal, not spatial features
                    ComplexActivation(
                        f
                    ),  # Phase-safe activation instead of nn.SiLU (Trabelsi et al. 2018)
                    conv_class(f, f, 3, padding=1, padding_mode="circular", bias=False),
                    nn.Identity(),  # NO normalization in k-space
                    ComplexActivation(
                        f
                    ),  # Phase-safe activation instead of nn.SiLU (Trabelsi et al. 2018)
                )
            )
            self.time_projs.append(ComplexChannelGain(time_embedding_dim, f * 2))
            cin = f

        self.decoder = nn.ModuleList()
        self.decoder_time_projs = nn.ModuleList()
        self.upsample = KSpacePad(scale_factor=2)

        features_rev = features[::-1]
        for i, f in enumerate(features_rev[:-1]):
            target_f = features_rev[i + 1]
            self.decoder_time_projs.append(ComplexChannelGain(time_embedding_dim, target_f * 2))
            self.decoder.append(
                nn.Sequential(
                    ComplexConv2d(f, target_f, 3, padding=1, padding_mode="circular", bias=False),
                    nn.Identity(),  # NO normalization in k-space
                    ComplexActivation(
                        target_f
                    ),  # Phase-safe activation instead of nn.SiLU (Trabelsi et al. 2018)
                )
            )

        if out_channels % 2 != 0:
            raise ValueError(f"PureKSpaceUNet expects even output channels, got {out_channels}")
        # bias=False: k-space constant term creates spatial Dirac delta singularity (Hammernik et al. 2018)
        self.final = ComplexConv2d(
            features[0],
            out_channels // 2,
            3,
            padding=1,
            padding_mode="circular",
            bias=False,
        )

    def _get_sinusoidal_embedding(self, timesteps: torch.Tensor, dim: int) -> torch.Tensor:
        """Generate sinusoidal timestep embeddings."""
        half_dim = dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass with optional time conditioning.

        Args:
            x: K-space input (B, C, H, W) with C=2 (real/imag)
            timesteps: Timestep indices (B,) for diffusion conditioning
        """
        # Generate time embedding
        if timesteps is not None:
            t_emb = self._get_sinusoidal_embedding(timesteps, self.time_embedding_dim)
            t_emb = self.time_mlp(t_emb)
        else:
            t_emb = None

        # The contrast conditions the encoder as well as the decoder: a decoder-only
        # contrast leaves every skip feature contrast-blind.
        combined_emb = fuse_contrast_embedding(
            t_emb, kwargs.get("contrast_emb"), owner="PureKSpaceUNet"
        )

        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x)
            x = self.time_projs[i](x, combined_emb)

            skips.append(x)
            x = self.pooling(x)

        # Decode
        for i, block in enumerate(self.decoder):
            skip = skips[-(i + 1)]
            x = self.upsample(x, target_shape=skip.shape[-2:])
            x = x + skip
            x = block(x)
            x = self.decoder_time_projs[i](x, combined_emb)

        x = self.upsample(x)
        x = x + skips[0]
        x = self.final(x)
        return x


from spectramr.infrastructure.physics.sense import SENSESubspaceProjector

#: Data-consistency methods that pin CARTESIAN bins, and so have no correct
#: reading under an off-grid acquisition.
GRID_DC_METHODS: frozenset[str] = frozenset(
    {
        "hard",
        "soft",
        "adaptive",
        "noise_adaptive",
        "noise_adjusted",
        "target_aware_fsdc",
        "kan_adaptive",
        "wiener_gain",
    }
)

#: Data-consistency methods whose blend is defined on LINEAR k-space, so they need
#: the arm's log1p-compression flag.
LINEAR_DOMAIN_DC_METHODS: frozenset[str] = frozenset(
    {"soft", "noise_adaptive", "noise_adjusted", "target_aware_fsdc", "wiener_gain"}
)

#: A model built under a non-default timestep scale carries the buffer
#: ``timestep_embedding_scale_<scale>``; a default-scale model carries none.
_DEFAULT_T_SCALE = "unit_interval"
_T_SCALE_MARKER_PREFIX = "timestep_embedding_scale_"

#: Backbones whose private sinusoid takes the raw step and ignores ``timestep_embedding_scale``.
_RAW_STEP_BACKBONES = frozenset({"unet", "nafnet"})

#: Architecture version stamped into every state dict as a buffer. Version 2:
#: phase-preserving per-complex-channel timestep gain in the k-space blocks, a
#: conditioned bottleneck, contrast conditioning in the encoder, and the k-space
#: geometry planes on every backbone input.
KSPACE_GENERATOR_ARCH_VERSION = 2
KSPACE_GENERATOR_ARCH_VERSION_KEY = "kspace_generator_arch_version"


@register_model(
    name="kspace_cold_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="kspace",
    # Training-time forward returns k-space (the FourierBridgeNetwork's
    # final FFT step, line ~2145). The `generate()` sampler can return
    # image when called with output_domain="image", but train losses see
    # k-space.
    output_domain="kspace",
    accepts_complex=False,
    expects_real_imag_interleaved=True,
    requires_paired_data=True,
    # Pattern C contrast-id conditioning: forward() accepts ``contrast_idx``
    # and routes it through ``self.contrast_embedding`` → ``contrast_emb``
    # which is added to the time embedding. See call sites at lines
    # ~1562–1569 (combined_emb) and ~2159–2162 (filtered_kwargs). Declaring
    # the capability here lets the audit
    # (``multi_contrast_model_support``) confirm that ``contrast_idx`` is
    # actually consumed instead of silently dropped at the model boundary.
    supports_contrast_conditioning=True,
)
class KSpaceColdDiffusionGenerator(nn.Module, IGenerator):
    """K-Space Cold Diffusion Generator.

    [ARCHITECT UPDATED] Now uses FourierBridgeNetwork to solve Inductive Bias Mismatch
    and explicit SENSE-Manifold Projections to enforce optimal coil topology.
    """

    # The synthetic-forward probe sees output ≈ input on a random k-space
    # tensor because the cold-diffusion process is built around a
    # degradation chain that, at t=0 (the probe condition), maps to a
    # near-identity transform of the (already-degraded) input. This is
    # by design — the model relies on the strategy's `t > 0` noising
    # schedule to produce non-trivial output. Suppress the false-positive
    # identity-collapse warning.
    synthetic_forward_probe_skip = {"identity_collapse"}

    # Read by the t=0 pre-DC probe to decide whether to emit `val_t0_predc_*`.
    # A class attribute, not a trial call: that decision selects the emitted
    # metric key set, and `pipelines/train.py:_all_reduce_val_metrics` packs
    # `sorted(val_accum.keys())` into a tensor and all-reduces it POSITIONALLY,
    # so a key set that differs between ranks reduces mismatched-length tensors
    # -- a hang, or one metric's value landing under another's name. Never an
    # error. A property of the class is the same on every rank by construction.
    exposes_pre_dc = True

    # Unrolled backbones whose internal ``MaskedReplacementDataConsistency`` keys against the
    # (un-doubled) measured k-space at EVERY cascade and preserve channel count
    # end-to-end — they have NO final projection to ``out_channels`` (unlike
    # ``swin_diff_rec``, whose ``final_conv`` reduces to ``out_channels`` BEFORE
    # its single DC). For these, the backbone input width MUST equal the measured
    # k-space width (== ``in_channels``); the S-map channel-doubling done in
    # ``__init__`` is skipped for them, otherwise ``k_guessed`` mismatches
    # ``measured`` inside the DC ("tensor a (8) vs b (4)" crash at iter 1 of
    # experiment_11_kspace_cold_diffusion_varnet). S-map conditioning still
    # reaches them via the learned ``ChannelAdapter`` 1x1 projection in
    # ``FourierBridgeNetwork``, which fuses ``[noisy || smaps]`` down to
    # ``in_channels``. This is a STRICT SUBSET of the forward-pass
    # ``_no_concat_backbones`` set (which also contains ``complex_unet`` — that
    # one has no internal DC, so doubling is harmless and is left in place).
    #
    # SCOPE: this set answers the CHANNEL-WIDTH question only. It is NOT the
    # set of backbones that do DC internally -- ``swin_diff_rec*`` do too, and
    # are deliberately absent here for the width reason given above. The DOMAIN
    # question has its own set below (CLAUDE.md #17: one owner per invariant).
    # Keying a second invariant off this membership is exactly the bug that let
    # ``force_pure_kspace: true`` reach the swin backbones unchecked.
    _INTERNAL_DC_BACKBONES: frozenset[str] = frozenset({"diff_varnet", "diff_varnet_kan"})

    #: Backbones built without the k-space geometry planes. The internal-DC
    #: unrolled networks iterate on their input as the reconstruction state and
    #: hold no input stem, so an appended plane would be iterated as k-space and
    #: compared against the measurement; conditioning them needs a path inside the
    #: unrolled block, which they do not have.
    _GEOMETRY_EXEMPT_BACKBONES: frozenset[str] = _INTERNAL_DC_BACKBONES

    #: Backbones whose internal ``MaskedReplacementDataConsistency`` consumes an
    #: IMAGE-domain tensor and FFTs it itself
    #: (``physics/data_consistency_layer.py`` step 1). Membership is the DOMAIN
    #: invariant and is a SUPERSET of :attr:`_INTERNAL_DC_BACKBONES`, which
    #: answers the unrelated width question.
    #:
    #: ``swin_diff_rec`` / ``swin_diff_rec_kan`` sit here but not there, and the
    #: difference is real rather than an oversight: their ``final_conv`` reduces
    #: to ``out_channels`` before their single DC, so the S-map doubling is
    #: harmless for WIDTH -- but neither file contains a single ``fft2c`` /
    #: ``torch.fft`` call, so whatever domain the bridge hands them is the
    #: domain their DC receives. ``swin_diff_rec``'s own comment states the
    #: precondition: "DC requires: image, measured_kspace, mask". The guard that
    #: reads this set is :func:`backbone_kwargs_defects`.
    _IMAGE_DOMAIN_DC_BACKBONES: frozenset[str] = frozenset(
        {"diff_varnet", "diff_varnet_kan", "swin_diff_rec", "swin_diff_rec_kan"}
    )

    #: Backbones with NO attention seam — they contain zero references to
    #: ``attention_type`` and absorb it through ``**kwargs``. Requesting anything
    #: but ``'none'`` on these raises instead of being silently dropped; an
    #: unspecified value resolves to ``'none'`` rather than the library default
    #: ``'self'``. See the guard and the resolution below for the measurement.
    _SEAMLESS_ATTENTION_BACKBONES: frozenset[str] = frozenset(
        {
            "swin_diff_rec",
            "swin_diff_rec_kan",
            "diff_varnet",
            "diff_varnet_kan",
            "nafnet",
            # Audited 2026-09-17, the measurement the guard below was waiting
            # for: sweeping attention_type over the whole registered vocabulary
            # leaves the parameter count of each of these EXACTLY unchanged
            # (restormer 2.378M, swinir 39.089M, vit 0.118M, swin_transformer
            # 0.059M at base_channels=32) while a bogus name still raises.
            # ``dual_domain`` is the one that looks alive -- it constructs 11
            # extra submodules -- and they carry zero parameters between them.
            "restormer",
            "swinir",
            "vision_transformer",
            "vit",
            "swin_transformer",
            # mamba_unet.py contains zero references to ``attention_type`` and
            # absorbs it through ``**kwargs`` -- the same static evidence that
            # admitted diff_varnet and nafnet. It cannot be construction-swept
            # without the ``.[mamba]`` extra, so this listing rests on the source,
            # not on a parameter count.
            "vision_mamba",
            "mamba_unet",
            # The four registry trunks are attention end to end, but none of them
            # implements THIS framework's pluggable block-attention seam, so an
            # arm naming one here would advertise a block it does not build.
            "dit",
            "uvit",
            "u_vit",
            "hat",
            "diffit",
            "kspace_ar",
            "kspace_masked",
        }
    )

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        num_layers: int = 4,
        attention_type: str | None = None,
        num_timesteps: int | None = None,
        sampling_steps: int | None = None,
        time_embedding_dim: int = 256,
        time_embedding_type: str = "sinusoidal",
        training_mode: str | None = None,
        activation: str = "complex",
        use_complex_conv: bool = True,
        acceleration_config: dict | None = None,
        kspace_log_scaled: bool | None = None,
        device: "torch.device | str | None" = None,
        phase_encode_axis: int | None = None,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            base_channels (int): Description.
            num_layers (int): Description.
            attention_type (str): Description.
            num_timesteps (int | None): The diffusion horizon T; the builder
                injects training.diffusion.timesteps. Also accepted as ``timesteps``.
            sampling_steps (int | None): Reverse steps of :meth:`sample`; the builder
                injects training.diffusion.sampling_steps, and ``None`` takes
                ``resolve_sampling_steps``'s logged default (#566).
            time_embedding_dim (int): Description.
            time_embedding_type (str): Description.
            training_mode (str | None): Description.
            activation (str): Description.
            use_complex_conv (bool): Description.
            acceleration_config (dict | None): Description.
            kspace_log_scaled: whether this arm's k-space is ``log1p``-compressed
                (``data.processing.enable_log_scaling``). DECLARED here, not
                defaulted, so ``ModelBuilder``'s signature contract injects the
                SSOT value the same way it injects ``acceleration_config``; no
                new YAML key is introduced. ``None`` means "never supplied": it
                is tolerated while the magnitude bound is off (the knob is
                opt-in) and RAISES at the point of use if the bound is on,
                because applying a physical ratio to compressed k-space without
                knowing which is what made a declared 1.3 realise 29.8x (#1281).
            device: the run's resolved compute device. DECLARED here, not
                defaulted, for the same reason as ``kspace_log_scaled`` above:
                ``resolve_generator_kwargs`` injects a contract-gated SSOT value
                only into constructors that name the parameter, and no other
                registered generator does. Forwarded to
                ``KSpaceUndersamplingProcess`` so its mask generator serves from
                a device-resident table; ``None`` leaves the historical CPU
                behaviour and its per-step host sync (#1508).
            phase_encode_axis: ``data.acquisition.phase_encode_axis``, injected
                by ``resolve_generator_kwargs``; ``mask_direction: phase``
                resolves through it and raises without it (#2160).
        """
        super().__init__()

        # Alias num_res_blocks from config to num_layers
        if "num_res_blocks" in kwargs:
            num_layers = kwargs.pop("num_res_blocks")

        # ``model_kwargs.timesteps`` and ``num_timesteps`` name one horizon, the
        # mask ladder's length and the time embedding's divisor; two differing
        # values raise, with None (not 1000) as the "not given" marker.
        if "timesteps" in kwargs:
            _ts = int(kwargs.pop("timesteps"))
            if num_timesteps is not None and num_timesteps != _ts:
                raise ValueError(
                    "KSpaceColdDiffusionGenerator received conflicting "
                    f"num_timesteps={num_timesteps} and timesteps={_ts}; they "
                    "are the same diffusion horizon / time-embedding max. Set "
                    "exactly one (prefer model_kwargs.timesteps)."
                )
            num_timesteps = _ts
        if num_timesteps is None:
            # Only direct construction lands here: the builder injects
            # training.diffusion.timesteps on every configured path.
            num_timesteps = DEFAULT_DIFFUSION_TIMESTEPS

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_timesteps = num_timesteps
        self.reference_channels = resolve_reference_channels(kwargs)

        # [PHYSICS INTEGRATION] Repetition (NEX) Fusion -- OPT-IN, not universal.
        #
        # Built ONLY when the arm explicitly declares ``num_repetitions``. This
        # used to be built unconditionally on every instance, which made both
        # capability probes unanswerable: ``hasattr(gen, "rep_fusion")`` was a
        # constant ``True``, so it reported "yes, I fuse repetitions" for arms
        # that never asked for it AND for arms whose data cannot supply the
        # declared count (#1173; CLAUDE.md pitfall #16 -- a probe that cannot
        # return False is a facade, not a check).
        #
        # The presence test is a REAL discriminator, not a tautology:
        # ``model.model_kwargs`` is an open ``dict[str, Any]``
        # (config/schemas/model.py:98), not a Pydantic model whose every
        # declared field is materialised on dump. A key is here only if the YAML
        # actually wrote it.
        #
        # The previous ``kwargs.get("num_repetitions", 4)`` also silently
        # substituted a default that NO M4Raw contrast can satisfy (real counts
        # are 3 for T1/T2 and 2 for FLAIR -- see
        # ``m4raw_dataset.M4RAW_REPETITIONS_BY_CONTRAST``), so an arm that never
        # mentioned repetitions still got a fusion layer sized for 4 of them.
        # Defaulting is exactly what non-negotiable 3 forbids here.
        # Import hoisted out of the conditional so the annotation below can name
        # the type unquoted; the module is imported here rather than at file
        # scope to keep the layers package off this module's import cycle.
        from spectramr.models.layers.complex_conv import ComplexRepetitionFusion

        self.rep_fusion: ComplexRepetitionFusion | None = None
        if "num_repetitions" in kwargs:
            num_physical_coils = kwargs.get("num_physical_coils", in_channels // 2)
            self.rep_fusion = ComplexRepetitionFusion(
                num_physical_coils=num_physical_coils,
                num_repetitions=kwargs["num_repetitions"],
            )

        # [PHYSICS ORCHESTRATION] Extract acceleration parameters
        # Priority: acceleration_config > individual kwargs > defaults.
        # Resolution lives in kspace_process so the CI ladder gate resolves the
        # SAME kwargs this constructor does (issue #550): the gate used to read
        # raw YAML with its own defaults, so it could certify a ladder the
        # runtime never built. ``acceleration_config`` may be a live
        # AccelerationConfigSchema, a model_dump() of one, or a raw dict.
        from spectramr.models.diffusion.kspace_process import (
            resolve_undersampling_kwargs,
        )

        accel_config = acceleration_config or kwargs.get("acceleration_config", {})
        process_kwargs = resolve_undersampling_kwargs(
            accel_config, kwargs, phase_encode_axis=phase_encode_axis
        )
        self.center_fraction = process_kwargs["center_fraction"]

        acceleration_params = kwargs.get("acceleration_params", {})
        if isinstance(acceleration_params, dict):
            keys_to_remove = [
                "max_acceleration",
                "center_fraction",
                "seed",
                "acceleration_type",
            ]
            acceleration_params = {
                k: v for k, v in acceleration_params.items() if k not in keys_to_remove
            }
        else:
            acceleration_params = {}

        # [PHYSICS ORCHESTRATION] The diffusion process handles restoration sampling
        # Consolidate on KSpaceUndersamplingProcess (explicit physics)
        from spectramr.models.diffusion.forward_process_registry import (
            build_forward_process,
        )

        # ``device`` is the run's already-resolved compute device, injected by
        # ``resolve_generator_kwargs`` step 3d because this constructor names it
        # explicitly. It is forwarded ONLY to the undersampling process, whose
        # mask generator needs a non-CPU device to serve masks from its
        # device-resident table instead of a per-step host sync (#1508). The
        # module's own parameters are placed by ``GeneratorBuilder.build``'s
        # ``.to(device)``, which is a separate concern and unchanged.
        # The acquisition model is the arm's choice, not this class's: a radial
        # arm degrades by dropping spokes off the grid, a Cartesian one by
        # dropping bins on it. Resolved through the registry so an unknown name
        # raises at build rather than silently acquiring Cartesian (pitfall 9).
        process_type = str(kwargs.get("kspace_process_type", "cartesian_mask"))
        extra_process_kwargs = {
            key: kwargs[key]
            for key in ("num_spokes", "samples_per_spoke", "density_compensation")
            if key in kwargs
        }
        if extra_process_kwargs and process_type == "cartesian_mask":
            raise ValueError(
                f"{sorted(extra_process_kwargs)} are non-Cartesian acquisition "
                f"knobs and kspace_process_type is 'cartesian_mask', which reads "
                f"none of them. Declare kspace_process_type explicitly or drop "
                f"the knobs (non-negotiable 8)."
            )
        # Training-time forward noise is read by the Cartesian process only; declared on
        # any other process it would be inert, so refuse it (non-negotiable 8).
        forward_noise = {
            key: kwargs[key] for key in ("forward_noise_sigma", "forward_noise_schedule") if key in kwargs
        }
        if forward_noise and process_type != "cartesian_mask":
            raise ValueError(
                f"{sorted(forward_noise)} are read only by the cartesian_mask forward process, "
                f"and kspace_process_type is {process_type!r}. Drop the knobs or declare "
                "kspace_process_type: cartesian_mask."
            )
        extra_process_kwargs.update(forward_noise)
        if process_type != "cartesian_mask":
            # The trajectory is built for one matrix size and the process raises
            # on a mismatch at the first forward. Take the size from the arm
            # rather than defaulting, so the failure is a config error at build.
            declared = kwargs.get("im_size") or kwargs.get("img_size") or kwargs.get("image_size")
            if declared is None:
                raise ValueError(
                    f"kspace_process_type={process_type!r} builds an off-grid "
                    f"trajectory and needs the matrix size; declare im_size in "
                    f"model.model_kwargs."
                )
            if isinstance(declared, int):
                declared = (declared, declared)
            extra_process_kwargs["im_size"] = tuple(declared)
            # An off-grid process resamples through a NUFFT, which does not commute
            # with log1p compression, so it must know the plane is compressed (#2341).
            extra_process_kwargs["kspace_log_scaled"] = bool(kspace_log_scaled)
        self.kspace_process = build_forward_process(
            process_type,
            num_timesteps=num_timesteps,
            device=device,
            **process_kwargs,
            **extra_process_kwargs,
        )
        self.kspace_process_type = process_type
        # Store DC method + sampler name for sample() use. Per
        # TODO/audit/11_diffusion_samplers_vae.md F1, ``_sampler_name``
        # was previously read but never written, so the YAML knob
        # ``training.diffusion.sampler`` (or ``inference_sampler``) had
        # no effect — every run silently used ``cold_mri``. Reading it
        # from kwargs here lets the ModelBuilder propagate the YAML
        # choice. Validation against ``SamplerRegistry.list_available()``
        # happens at sample-call time via ``get_sampler``.
        self._dc_method = kwargs.get("dc_method", "hard")
        # A grid DC layer pins bins it believes were measured. Under an off-grid
        # acquisition every Cartesian bin is interpolated, so `hard` would pin
        # gridded values as data, and with an empty support it degrades to a
        # silent no-op instead -- a run that trains with no data consistency and
        # reports success. Fail at build (pitfall 9).
        if process_type != "cartesian_mask" and self._dc_method in GRID_DC_METHODS:
            raise ValueError(
                f"dc_method={self._dc_method!r} is a grid-domain data-consistency "
                f"method and kspace_process_type={process_type!r} acquires off-grid "
                f"samples, where no Cartesian bin is a measurement. Declare a "
                f"sample-domain consistency term on this arm instead."
            )
        self._dc_weight = float(kwargs.get("dc_weight", 1.0))
        from spectramr.models.diffusion.kspace_process import resolve_sampling_steps

        self._sampling_steps = resolve_sampling_steps(sampling_steps, num_timesteps)
        self._sampler_name = str(
            kwargs.get("sampler") or kwargs.get("inference_sampler") or "cold_mri"
        )
        # Validated at BUILD by the module that actually looks the name up, so
        # an arm naming a sampler this reverse loop cannot resolve fails now
        # rather than mid-validation (pitfall 15).
        from spectramr.models.diffusion.samplers.registry import list_available

        _known = list_available()
        if _known and self._sampler_name not in _known:
            raise ValueError(
                f"Unknown sampler {self._sampler_name!r}. Registered: "
                f"{sorted(_known)}. Declaring an unregistered name used to run "
                f"`cold_mri` silently."
            )
        # Reverse-process variant for the cold_mri sampler. Validate at BUILD
        # (pitfall #15) so an illegal YAML value fails now, not mid-validation;
        # the resolved value is stamped into provenance via model_kwargs.
        from spectramr.models.diffusion.kspace_process import (
            VALID_REVERSE_MODES,
        )

        # How the backbone maps t before its sinusoid (#2315). The default keeps
        # the codes existing checkpoints were trained on; validated at build.
        from spectramr.models.blocks.timestep_embedding import validate_timestep_scale

        self._timestep_embedding_scale = validate_timestep_scale(
            str(kwargs.get("timestep_embedding_scale", _DEFAULT_T_SCALE))
        )
        # These two embed the raw step index with a private sinusoid and never read the scale, so a
        # declared non-integer scale would be stamped on a checkpoint the model never used.
        _declared_scale = kwargs.get("timestep_embedding_scale")
        _backbone_kind = str(kwargs.get("backbone_type", DEFAULT_BACKBONE_TYPE))
        if (
            _backbone_kind in _RAW_STEP_BACKBONES
            and _declared_scale is not None
            and self._timestep_embedding_scale != "integer"
        ):
            raise ConfigurationError(
                f"backbone_type={_backbone_kind!r} embeds the raw integer step and ignores "
                f"timestep_embedding_scale={_declared_scale!r}. Declare 'integer' (what it "
                "computes) or use complex_unet, which honours the scale."
            )
        # A checkpoint's time_mlp is fitted to one scale; the marker travels in
        # the state dict so the hook below refuses a load under any other.
        if self._timestep_embedding_scale != _DEFAULT_T_SCALE:
            self.register_buffer(
                _T_SCALE_MARKER_PREFIX + self._timestep_embedding_scale, torch.ones(1)
            )
        self.register_load_state_dict_pre_hook(self._refuse_mismatched_timestep_scale)
        # Version 2 changed the k-space block conditioning and widened every
        # backbone input; the stamp lets a version-1 checkpoint fail by name.
        self.register_buffer(
            KSPACE_GENERATOR_ARCH_VERSION_KEY,
            torch.tensor(KSPACE_GENERATOR_ARCH_VERSION, dtype=torch.long),
        )
        self.register_load_state_dict_pre_hook(self._refuse_other_arch_version)
        self._reverse_mode = str(kwargs.get("reverse_sampling_mode", "additive"))
        if self._reverse_mode not in VALID_REVERSE_MODES:
            raise ValueError(
                f"Unknown reverse_sampling_mode {self._reverse_mode!r}. "
                f"Valid: {sorted(VALID_REVERSE_MODES)}."
            )
        from spectramr.models.diffusion.reverse_refine import validate_refine_alpha

        self._refine_alpha = validate_refine_alpha(
            self._reverse_mode, kwargs.get("reverse_refine_alpha")
        )
        from spectramr.models.diffusion.reverse_integrity import validate_integrity_check

        self._integrity_check = validate_integrity_check(kwargs.get("reverse_integrity_check"))
        from spectramr.models.diffusion.reverse_physics import resolve_reverse_physics_kwargs

        self._reverse_physics = resolve_reverse_physics_kwargs(
            self._reverse_mode, str(self._dc_method), kwargs
        )
        self._reverse_clip_ratio = float(kwargs.get("reverse_clip_ratio", 4.0))
        if not (self._reverse_clip_ratio > 0):
            raise ValueError(f"reverse_clip_ratio must be > 0, got {self._reverse_clip_ratio!r}.")
        # Reference the magnitude ceiling to the whole tensor (legacy) or to each
        # coefficient's own radial band. ``max|measured|`` is the k-space DC peak,
        # ~37x the RMS coefficient here, so a global ratio near 1 bounds nothing
        # that matters (issue #536). Validated at BUILD, stamped via model_kwargs.
        from spectramr.models.diffusion.kspace_process import (
            VALID_CLIP_REFERENCES,
        )

        self._clip_reference = str(kwargs.get("output_kspace_clip_reference", "global_max"))
        if self._clip_reference not in VALID_CLIP_REFERENCES:
            raise ValueError(
                f"Unknown output_kspace_clip_reference {self._clip_reference!r}. "
                f"Valid: {sorted(VALID_CLIP_REFERENCES)}."
            )

        # Read by the forward process at build (above) and by the reverse loop through it;
        # the exclusivity with ``sampler_sigma`` is checked here so the build fails, not sample().
        self._forward_noise_sigma = float(self.kspace_process.forward_noise_sigma) if process_type == "cartesian_mask" else 0.0
        # C6 sampler-determinism knobs: ``sampler_sigma`` / ``sampler_seed`` /
        # ``selection_rule``. Read from model_kwargs here, validated at BUILD
        # through the SAME function both samplers use, and forwarded to the
        # ``cold_mri`` sampler in ``sample()``. Until 2026-09 the ``cold_mri``
        # wrapper declared none of them, so the three YAML keys were inert on
        # the training-validation path (issue #1286): ``sampler_sigma: 0.0`` in
        # a YAML was true by accident, and a non-zero value would have been
        # silently ignored.
        from spectramr.models.diffusion.kspace_process import (
            validate_sampler_determinism,
        )

        self._sampler_sigma = float(kwargs.get("sampler_sigma", 0.0))
        _sampler_seed = kwargs.get("sampler_seed")
        self._sampler_seed = None if _sampler_seed is None else int(_sampler_seed)
        self._selection_rule = str(kwargs.get("selection_rule", "fixed"))
        validate_sampler_determinism(self._sampler_sigma, self._selection_rule)
        from spectramr.models.diffusion.kspace_process import validate_forward_noise_exclusive

        validate_forward_noise_exclusive(self._forward_noise_sigma, self._sampler_sigma)

        # Set by :meth:`sample` from the per-call sampler before it is discarded.
        # ``None`` means no reverse sampling has run on this generator yet, or
        # the resolved sampler keeps no such record -- distinguishable from a
        # sampler that ran zero steps, which reports zeros.
        self._last_reverse_stats: dict[str, Any] | None = None

        # [SCALE CONTROL — Phase-1 divergence guard] Optional phase-preserving
        # bound on the model's OUTPUT k-space magnitude, ``ratio x max|measured|``
        # per sample, applied in ``forward`` whenever the measured k-space is
        # available (training). The pure-k-space ``complex_unet`` backbone has NO
        # inter-layer normalisation (spatial norm is inappropriate for frequency
        # bins), so nothing else caps output-scale drift over training — the
        # root of the experiment_11 measurement-independent collapse. ``None``
        # (default) leaves the cohort byte-identical. Validate at BUILD and stamp
        # the resolved value via ``model_kwargs`` (pitfall #15).
        _clip = kwargs.get("output_kspace_clip_ratio")
        if _clip is not None and not (float(_clip) > 0):
            raise ValueError(f"output_kspace_clip_ratio must be > 0 or null, got {_clip!r}.")
        self._output_kspace_clip_ratio = None if _clip is None else float(_clip)

        # The magnitude bound is a PHYSICAL ratio, so it cannot be applied
        # without knowing the domain of the k-space it bounds. Refuse loudly
        # rather than assume physical (CLAUDE.md #3): assuming wrong is the
        # 1.3-declared / 29.8x-realised failure of issue #1281. Only enforced
        # when the (opt-in) bound is actually on, so arms that leave it off are
        # unaffected and need no rebuild.
        self._kspace_log_scaled = kspace_log_scaled
        if self._output_kspace_clip_ratio is not None and self._kspace_log_scaled is None:
            raise ValueError(
                "output_kspace_clip_ratio is set but kspace_log_scaled was never "
                "supplied, so the magnitude ceiling cannot be built in the right "
                "domain. ModelBuilder injects it from "
                "data.processing.enable_log_scaling; pass kspace_log_scaled "
                "explicitly when constructing this generator directly."
            )
        # One warning per instance, not per step: the batch-mismatch case is
        # structural (it holds for every step of a D>1 arm), so repeating it
        # 30,000 times would bury the log it is meant to make readable.
        self._warned_measurement_batch = False

        # [ARCHITECT IMPLEMENTATION] Fourier Bridge Integration
        # We replace the manual U-Net construction with the Fourier Bridge.

        # ``attention_type=None`` means "not specified". Resolve it against the
        # backbone BEFORE any use, because the two defaults were mutually
        # incompatible: ``backbone_type`` defaults to ``'unet'`` and
        # ``attention_type`` defaulted to ``'self'``, which the guard further
        # down rejects — so ``KSpaceColdDiffusionGenerator()`` could not be
        # constructed at all, and **22 corpus arms that declare NEITHER knob**
        # inherited that. A new validation must not reject the library default;
        # it must guard the EXPLICIT argument, and the sentinel is what tells
        # "the user asked for self" apart from "nobody asked".
        #
        # ``none`` is ConfigurableResidualBlock's OWN default
        # (``ConfigurableResidualBlock.__init__``, models/reconstruction/unet.py),
        # so the unet branch matches the
        # block it builds rather than inventing a third answer. An explicit
        # unsupported value still raises, unchanged.
        #
        # The same reasoning extends to ``_SEAMLESS_ATTENTION_BACKBONES``: they
        # build no attention at all, so ``none`` is likewise the honest
        # resolution of "nobody asked", and the seam guard further down stays a
        # check on the EXPLICIT argument. Without this, the guard would reject
        # every config that merely omits the knob — including this module's own
        # diff_varnet tests — which is the exact regression described above.
        if attention_type is None:
            _bb = kwargs.get("backbone_type", "unet")
            _seamless = _bb in self._SEAMLESS_ATTENTION_BACKBONES
            attention_type = "none" if (_bb == "unet" or _seamless) else "self"
            logger.debug(
                "attention_type not specified; resolved to %r for backbone_type=%r",
                attention_type,
                _bb,
            )

        # Map Attention string to Enum — raise on unknown values; silent
        # fallbacks mask misconfigured experiments (CLAUDE.md §9).
        try:
            attn_enum = AttentionType(attention_type.lower())
        except ValueError:
            valid = [e.value for e in AttentionType]
            raise ValueError(
                f"attention_type '{attention_type}' is not valid. Valid options: {valid}"
            ) from None

        # Configure backbone
        # features list: [base, base*2, base*4...]
        features = tuple(base_channels * (2**i) for i in range(num_layers))

        # An unknown option raises: degrading to GROUP / STANDARD would train a different network
        # than the YAML declares (non-negotiable 3).
        norm_enum = parse_enum_option(
            NormalizationType, kwargs.get("norm_type", "group"), "norm_type"
        )
        block_enum = parse_enum_option(BlockType, kwargs.get("block_type", "standard"), "block_type")

        # backbone_type x model_kwargs legality has one owner, which the audit also
        # runs on the frozen config (#2047). It sits before EITHER backbone branch:
        # the ``force_pure_kspace and backbone_type == 'unet'`` branch below builds
        # PureKSpaceUNet directly and never constructs the bridge.
        defects = backbone_kwargs_defects(kwargs)
        if defects:
            raise ValueError(defects[0])

        force_pure_kspace = kwargs.pop("force_pure_kspace", False)
        backbone_type = kwargs.pop("backbone_type", DEFAULT_BACKBONE_TYPE)
        self.backbone_type = backbone_type
        geometry_planes = resolve_geometry_planes(kwargs.pop("kspace_geometry_planes", None))

        self._gate_complex_unet_knobs(kwargs, backbone_type)

        # F-SPADE (2026-05-24 smoke_audit_20260524): fail loud on a silent
        # SPADE fallback. ``SPADEBlock`` / ``SPADEEncoder``
        # (models/blocks/spade.py) have NO consumer in any
        # kspace_cold_diffusion backbone — the ``spade_hidden_channels`` /
        # ``spade_norm_type`` model_kwargs were silently dropped, so
        # ``experiment_11b_spade_cold_diffusion`` ran as a plain
        # ``complex_unet`` and its "SPADE" outputs were indistinguishable
        # from the non-SPADE base (a mislabeled arm that would pollute the
        # sim2rank pool). CLAUDE.md #9 forbids silent fallbacks: surface the
        # misconfiguration at build time instead of faking SPADE results.
        # To run a real SPADE arm, wire ``SPADEEncoder`` into the decoder
        # levels and plumb a marker tensor through ``forward`` (a feature
        # task — the marker semantics must be decided), then register the
        # backbone as SPADE-aware here. See
        # TODO/audit/smoke_audit_20260524.md §F-SPADE.
        _spade_only_kwargs = [
            k for k in ("spade_hidden_channels", "spade_norm_type") if k in kwargs
        ]
        if _spade_only_kwargs:
            raise ValueError(
                f"SPADE kwargs {_spade_only_kwargs} were supplied in "
                f"model_kwargs but no kspace_cold_diffusion backbone consumes "
                f"them — SPADEBlock/SPADEEncoder (models/blocks/spade.py) are "
                f"not wired into backbone_type={backbone_type!r}. This "
                f"previously ran silently as a plain backbone (CLAUDE.md #9 "
                f"forbids silent fallbacks). Wire SPADE into the decoder and "
                f"plumb a marker through forward(), or remove the spade_* "
                f"kwargs to run an honest non-SPADE baseline. See "
                f"TODO/audit/smoke_audit_20260524.md §F-SPADE."
            )

        # The training strategy ALWAYS concatenates S-maps onto the noisy
        # input along the channel dim for kspace_cold_diffusion (see
        # diffusion.py::_prepare_diffusion_inputs). The S-maps share the
        # noisy input's channel count, so the backbone receives 2 ×
        # in_channels. Toggleable via condition_with_smaps=False for
        # configs that want to disable S-map conditioning entirely (in
        # which case the strategy must also be configured to skip the
        # concat — TODO: surface a config flag for that path).
        self.condition_with_smaps = bool(
            kwargs.pop("condition_with_smaps", DEFAULT_CONDITION_WITH_SMAPS)
        )
        # Internal-DC backbones (diff_varnet / diff_varnet_kan) must receive an
        # input width equal to the measured k-space (== in_channels) because
        # their per-cascade MaskedReplacementDataConsistency compares the channel-preserved
        # prediction against the un-doubled measured k-space. Skip the S-map
        # channel-doubling for them.
        # See KSpaceColdDiffusionGenerator._INTERNAL_DC_BACKBONES.
        #
        # NOTE: they are NOT "still conditioned through the learned
        # ChannelAdapter projection" — that adapter is built inside
        # ``FourierBridgeNetwork.forward`` and is therefore never trained (see
        # the width-contract raise there). They simply receive no S-maps.
        #
        # [ONE OWNER — CLAUDE.md #17] ``expects_smaps_concat`` is the single
        # predicate that decides BOTH the backbone's input width here AND
        # whether any caller may concatenate S-maps onto the stack. It used to
        # be a local, so every downstream site invented its own spelling and
        # they disagreed: ``forward``'s dead ``in_channels == out_channels * 2``
        # gate, a ``_no_concat_backbones`` set that contradicted
        # ``_INTERNAL_DC_BACKBONES`` on ``complex_unet``, and a training
        # strategy that concatenated unconditionally. Read this attribute;
        # never re-derive the rule.
        # Hoisted to module scope (#1387) so the config auditor can reach the
        # SAME rule without building the model. Do not inline it back.
        self.expects_smaps_concat = resolve_expects_smaps_concat(
            backbone_type=backbone_type,
            condition_with_smaps=self.condition_with_smaps,
        )
        backbone_in_channels = in_channels * 2 if self.expects_smaps_concat else in_channels
        # PUBLISHED, not just consumed. ``self.in_channels`` above is the BARE
        # measurement width -- what the validation sampler re-enters with and
        # what the concat gate in ``forward`` keys on. It is NOT the width the
        # backbone sees on the training path, where the strategy pre-concatenates
        # and arrives at ``2 * in_channels``. Both are legitimate inputs to
        # ``forward``; only this one describes the tensor the debug snapshot
        # names ``model_input``, so the model-input contract
        # (``infrastructure/training/model_input_contract.py``) reads THIS
        # attribute first. Without it the contract compared a 16-channel
        # ``model_input`` against ``in_channels=8`` and reported a mismatch on
        # every smaps-conditioned cold-diffusion arm -- a diagnostic crying wolf
        # across most of the cohort, which is how a check gets muted rather than
        # read.
        # The reference is excluded: ``forward`` concatenates it after the
        # ``model_input`` this attribute describes, so only the build adds it.
        self.backbone_in_channels = backbone_in_channels
        # Geometry planes join after the reference and the entry transform, as
        # (plane, 0) complex pairs, so they widen the build but not the published
        # model-input width above.
        self.kspace_geometry_planes = (
            () if backbone_type in self._GEOMETRY_EXEMPT_BACKBONES else geometry_planes
        )
        self.kspace_geometry_channels = len(self.kspace_geometry_planes)
        geometry_width = 2 * self.kspace_geometry_channels
        # A pure-k-space complex_unet takes the planes as a per-bin gain: as input
        # channels they are a measurement-independent k-space term, which is the
        # centre dot and centre-row line in its output (KSpaceGeometryGain).
        if "kspace_geometry_gain_planes" in kwargs:
            raise ValueError(
                "kspace_geometry_gain_planes is derived from kspace_geometry_planes and "
                "the backbone; declare kspace_geometry_planes instead."
            )
        if self.kspace_geometry_channels and backbone_type == "complex_unet" and force_pure_kspace:
            kwargs["kspace_geometry_gain_planes"] = self.kspace_geometry_channels
            geometry_width = 0

        unet_config = UNetConfig(
            # noisy + smaps when conditioning enabled, the source reference, geometry
            in_channels=backbone_in_channels + self.reference_channels + geometry_width,
            out_channels=out_channels,  # Output image channels
            features=features,
            depth=num_layers,
            attention_type=attn_enum,
            use_attention=(attention_type != "none"),
            time_dim=time_embedding_dim,
            # Defaults for robustness
            norm_type=norm_enum,
            block_type=block_enum,
            use_residual=True,
        )

        if force_pure_kspace and backbone_type == "unet":
            # PureKSpaceUNet is specialized for unet structure in K-Space.
            # It has NO attention seam — its encoder/decoder are plain
            # ComplexConv + ComplexActivation. Requesting any attention here
            # would be silently dropped (pitfall #16 facade: the arm would
            # advertise e.g. attention_type=self yet run vanilla). Fail loud
            # so the misconfiguration surfaces at build time, not as a
            # mislabeled result in the sim2rank pool. The constructor default
            # is attention_type="self", so an arm that omits the key also
            # fails here — set it to "none" explicitly (or use complex_unet
            # to keep block-level attention).
            if attention_type != "none":
                raise ValueError(
                    "backbone_type='unet' + force_pure_kspace=true builds "
                    "PureKSpaceUNet, which has no attention seam — "
                    f"attention_type={attention_type!r} would be silently "
                    "dropped (pitfall #16). Set model_kwargs.attention_type: "
                    "'none', or use backbone_type: 'complex_unet' to keep "
                    "block-level attention."
                )
            # Phase 5: Pass config-driven reflect padding parameter
            num_bottleneck_reflect_pad_layers = kwargs.get("reflect_padding_bottleneck_layers", 2)
            self.backbone = PureKSpaceUNet(
                in_channels=backbone_in_channels + self.reference_channels + geometry_width,
                out_channels=out_channels,
                features=features,
                time_embedding_dim=time_embedding_dim,
                num_bottleneck_reflect_pad_layers=num_bottleneck_reflect_pad_layers,
            )
        else:
            # For all other backbones (or unet in bridge mode), use the unified network.
            # FourierBridgeNetwork handles strict validation of backbone_type.
            # It also handles force_pure_kspace by disabling FFT/IFFT ops internally.
            #
            # Fail loud for the ``unet`` (reconstruction-UNet) backbone with an
            # unsupported attention_type. That backbone builds
            # ``ConfigurableResidualBlock``, which implements ONLY channel/spatial
            # attention. An EXPLICIT self/dual_domain/kan/wavelet request would
            # otherwise hit ConfigurableResidualBlock's generic raise deep in
            # the stack, so surface an actionable message at the generator seam
            # instead (pitfall #9) — the attention seam for those types is
            # ``backbone_type='complex_unet'``. This can no longer fire on the
            # DEFAULT: an unspecified attention_type resolves to ``none`` for
            # this backbone above.
            _UNET_BLOCK_ATTENTION = {"none", "channel", "spatial"}
            if backbone_type == "unet" and attention_type not in _UNET_BLOCK_ATTENTION:
                raise ValueError(
                    "backbone_type='unet' (the reconstruction-UNet bridge) builds "
                    "ConfigurableResidualBlock, which implements only attention_type "
                    f"in {sorted(_UNET_BLOCK_ATTENTION)}; got {attention_type!r}. Use "
                    "backbone_type: 'complex_unet' to keep self/dual_domain/kan/"
                    "wavelet block-attention, or set model_kwargs.attention_type to a "
                    "supported value ('none'/'channel'/'spatial')."
                )
            # Same facade, different backbones. The seamless set builds NO
            # attention: those modules contain zero references to
            # ``attention_type`` and their constructors absorb it via
            # ``**kwargs``, so a request was validated against the registry and
            # then discarded. Measured on swin_diff_rec — sweeping
            # attention_type over 'self', 'none', 'channel', 'sparse' and
            # 'kan_dual_domain' produced a byte-identical 26.40M-parameter
            # model while a bogus value still raised, i.e. the name was checked
            # and dropped, the most misleading of the possible behaviours.
            # diff_varnet and nafnet build no attention module whatsoever.
            #
            # Still NOT listed: vision_mamba / mamba_unet. They need the
            # ``.[mamba]`` extra to construct at all, so the sweep that admitted
            # the five backbones above could not be run on them, and a guard on
            # an unmeasured backbone is the same untested assumption in the
            # other direction.
            if backbone_type in self._SEAMLESS_ATTENTION_BACKBONES and attention_type != "none":
                raise ValueError(
                    f"backbone_type={backbone_type!r} has no attention seam: it never "
                    f"reads attention_type, so {attention_type!r} would be silently "
                    "dropped and the arm would advertise attention it does not run "
                    "(pitfall #16 facade). Set model_kwargs.attention_type: 'none', or "
                    "use backbone_type: 'complex_unet', which implements the "
                    "self/channel/sparse/kernelized/dual_domain/kan_dual_domain/"
                    "wavelet_freq block-attention seam."
                )
            self.backbone = FourierBridgeNetwork(
                unet_config,
                backbone_type=backbone_type,
                force_pure_kspace=force_pure_kspace,
                time_embedding_dim=time_embedding_dim,
                attention_type=attention_type,  # ✅ Pass attention_type explicitly for Phase 5
                **kwargs,
            )

        # Contrast conditioning: optional FiLM-style embedding for multi-contrast datasets
        # (e.g. M4Raw with T1/T2/FLAIR/PD).  Disabled by default (num_contrasts=0).
        num_contrasts = kwargs.get("num_contrasts", 0)
        self.num_contrasts = int(num_contrasts)
        if self.num_contrasts > 0:
            self.contrast_embedding = nn.Embedding(self.num_contrasts, time_embedding_dim)
            nn.init.normal_(self.contrast_embedding.weight, std=0.02)

        # [PHYSICS INTEGRATION] Register Data Consistency as a submodule
        # This ensures that learnable parameters (like SoftDC's lambda_param)
        # are included in self.parameters() and hooked into the optimizer.
        self.use_dc = kwargs.get("use_dc", True)
        self.dc_method = kwargs.get("dc_method", "hard")
        self.dc_weight = kwargs.get("dc_weight", 1.0)
        # Noise-simulation levels, forwarded from ``physics.data_consistency``
        # (#1525). Before that they were schema fields with no consumer: every
        # layer fell back to its own hard-coded 0.01/0.005 and a declared value
        # was silently discarded. Only the branches whose layer accepts them
        # pass them on -- ``dc_settings.DCKnobReadership`` is the SSOT for which
        # those are, and the audit reports a declaration the method cannot read.
        #
        # The defaults are 0.0, i.e. data consistency pins to the measurement
        # and nothing else. They used to be 0.01/0.005, which no arm ever
        # declared -- 0 of the corpus sets either knob, so all 68 `dc_method:
        # hard` arms simulated acquisition noise nobody chose. The level is
        # ABSOLUTE and the k-space it perturbs is log1p-compressed, so a single
        # number lands as a different corruption in every band: measured on a
        # percentile-normalised phantom at 256x256, 0.005 is 0.9% of the mean
        # |k| in the DC annulus and 23.2% in the outer one, and the training
        # level 0.01 reaches 46.5% there. That randomises the PHASE of observed
        # outer-band lines by up to 23.9 degrees -- on the bins hard DC exists
        # to hold exactly, and which the model therefore cannot correct.
        #
        # The reverse loop already disagreed: `_apply_observed_dc` is
        # `x0*(1-obs) + measurement*obs` with no noise term, so training saw
        # corrupted measurements and the validation that reports the numbers saw
        # clean ones. Electing the reverse loop's semantics (non-negotiable 17)
        # makes the two paths one mechanism again.
        #
        # The knobs stay read, so an arm that wants acquisition-noise
        # augmentation declares it and the audit still reports a declaration the
        # method cannot consume.
        self.dc_train_noise_level = float(kwargs.get("train_noise_level", 0.0))
        self.dc_eval_noise_level = float(kwargs.get("eval_noise_level", 0.0))
        self.dc_noise_type = kwargs.get("noise_type", "gaussian")

        # 2026-05-28: ``dc_method: null`` (Python ``None``) and the empty
        # string used to fall through to ``SimpleDataConsistency`` with
        # ``method=None``, silently applying soft DC blending. The
        # experiment_11 YAML set ``dc_method: null`` intending to *disable*
        # the model-internal DC — the silent fall-through doubled DC with
        # ``physics.data_consistency`` (CLAUDE.md #9 violation; root cause
        # of the validation "DC blob" the smoke triage flagged on
        # 2026-05-28). Treat ``None`` / "" / "none" as an explicit disable.
        if self.dc_method in (None, "", "none", "off", "disabled"):
            self.use_dc = False

        # SSOT: the advertised DC-method set lives in physics.data_consistency
        # so the model-internal builder and the reverse-diffusion sampler stay
        # in lockstep (no divergence like the 2026-07-05 'adaptive' crash).
        if self.use_dc and self.dc_method not in VALID_DC_METHODS:
            raise ValueError(
                f"[KSpaceColdDiffusionGenerator] unknown dc_method="
                f"{self.dc_method!r}. Valid choices: "
                f"{sorted(VALID_DC_METHODS)} (or null / 'none' to "
                f"disable model-internal DC entirely). The previous code "
                f"silently fell through to SimpleDataConsistency which "
                f"caused the experiment_11 'DC blob' regression."
            )

        # Blend-based layers compute on LINEAR k-space; replacement (hard) commutes with
        # the log1p compression and needs no flag.
        if (
            self.use_dc
            and self.dc_method in LINEAR_DOMAIN_DC_METHODS
            and self._kspace_log_scaled is None
        ):
            raise ValueError(
                f"dc_method={self.dc_method!r} blends on linear k-space but "
                "kspace_log_scaled was never declared; pass it from "
                "data.processing.enable_log_scaling."
            )
        dc_log_scaled = bool(self._kspace_log_scaled)

        if not self.use_dc:
            self.dc_layer = None
        elif self.dc_method == "soft":
            self.dc_layer = SoftDataConsistency(
                lambda_init=self.dc_weight, log_scaled=dc_log_scaled
            )
        elif self.dc_method == "noise_adjusted":
            # One Wiener trust per sample from its noise level (dc_weight is beta).
            self.dc_layer = NoiseAdjustedDataConsistency(
                beta=self.dc_weight, log_scaled=dc_log_scaled
            )
        elif self.dc_method == "adaptive":
            self.dc_layer = AdaptiveDataConsistency()
        elif self.dc_method == "kan_adaptive":
            kan_dc_kwargs = kwargs.get("kan_dc_kwargs", {}) or {}
            self.dc_layer = KANAdaptiveDataConsistency(**kan_dc_kwargs)
        elif self.dc_method == "noise_adaptive":
            # Closed-form Wiener/SNR trust: denoise the low-field measured lines
            # (dc_weight is the trust temperature β). No learned per-pixel map, so
            # no CNN-adaptive blob risk. See NoiseAdaptiveDataConsistency.
            self.dc_layer = NoiseAdaptiveDataConsistency(
                beta=self.dc_weight, log_scaled=dc_log_scaled
            )
        elif self.dc_method == "wiener_gain":
            # Wiener trust plus a closed-form gain fit, both on LINEAR k-space, so
            # the layer must know whether this arm's plane is log1p-compressed.
            self.dc_layer = WienerGainDataConsistency(
                log_scaled=dc_log_scaled, beta=self.dc_weight
            )
        elif self.dc_method == "hard":
            # No weight argument, deliberately: hard DC REPLACES the acquired
            # bins ((1 - m) * recon + m * obs), which is weight 1.0 by
            # construction, so there is no blend coefficient for ``dc_weight``
            # to occupy. It does simulate acquisition noise, so the two noise
            # levels and the noise model are the knobs that reach it.
            self.dc_layer = HardDataConsistency(
                train_noise_level=self.dc_train_noise_level,
                eval_noise_level=self.dc_eval_noise_level,
                noise_type=self.dc_noise_type,
            )
        elif self.dc_method == "target_aware_fsdc":
            # Every output channel is the target: a source contrast reaches the
            # model as ``reference`` and never enters the diffusion state.
            dc_target_ch = out_channels
            # ``ta_fsdc_acs_taper`` opt-in (audit-2026-05-14 §1 round-5):
            # set ``None`` (legacy hard rectangle) or ``"hann"`` (Hann-tapered
            # ACS edge). The Hann taper mitigates the "white spot" central
            # cluster seen in ``experiment_130_ti_ccd`` (and similar
            # cold-diffusion experiments) during early training, when the
            # model's HF prediction is near-zero and a hard-rectangle ACS
            # replacement's IFFT produces a concentrated low-pass blob.
            self.dc_layer = TargetAwareFSDC(
                target_channels=dc_target_ch,
                center_fraction=self.center_fraction,
                hf_lambda=self.dc_weight,
                acs_taper=kwargs.get("ta_fsdc_acs_taper"),
                log_scaled=dc_log_scaled,
            )
        else:
            self.dc_layer = SimpleDataConsistency(
                method=self.dc_method,
                weight=self.dc_weight,
                train_noise_level=self.dc_train_noise_level,
                eval_noise_level=self.dc_eval_noise_level,
                noise_type=self.dc_noise_type,
            )

        # [PHYSICS INTEGRATION] Orthogonal Projection onto True Spatial Manifold
        self.sense_projector = SENSESubspaceProjector()
        self.last_backbone_output: torch.Tensor | None = None
        self.last_backbone_projected: torch.Tensor | None = None
        #: ``(fraction, mean overshoot)`` of the output clamp on the last training forward.
        self.last_clip_stats: tuple[torch.Tensor, torch.Tensor] | None = None

        # [DC PASSTHROUGH] Hardcoded centre-patch skip-connect.
        # The CNN should not be reconstructing the DC bin (or its
        # immediate neighbours) — those frequencies are determined by
        # the measurement and dominate k-space magnitude by 5+ decades,
        # producing the "white spot" centre-blob failure mode that
        # AdaptiveDataConsistency only mitigates after extensive
        # training. Specifying a patch size > 0 routes that region
        # around the CNN entirely. ``None`` / size 0 disables.
        # See docs/validation_image_audit.rst for the diagnosis.
        # [RADIAL BAND GAIN] Opt-in phase-exact per-annulus magnitude correction,
        # applied to the model's own k-space output before it is captured as
        # ``x_pre_dc``. Off unless an arm declares a band count, and identity at
        # initialisation when on, so the control's numbers are reproducible with
        # the mechanism built. The measured deficit it targets is the decoder's
        # 2^-(up_steps) radial attenuation (#2117), which no attention type can
        # correct because every one of them is gain-bounded.
        band_cfg = kwargs.get("radial_band_gain_bands")
        if band_cfg is None or int(band_cfg) == 0:
            self.radial_band_gain = None
        else:
            bands = int(band_cfg)
            if bands < 2:
                raise ValueError(
                    "radial_band_gain_bands must be 0 (disabled) or >= 2, got "
                    f"{bands}. One annulus is a global scalar wearing a band's "
                    "name, and nothing downstream would distinguish the two."
                )
            self.radial_band_gain = RadialBandGain(
                n_bands=bands,
                time_embed_dim=int(kwargs.get("radial_band_gain_time_embed_dim", 0) or 0),
                max_log_gain=float(
                    kwargs.get("radial_band_gain_max_log", DEFAULT_MAX_LOG_GAIN)
                ),
            )

        passthrough_cfg = kwargs.get("dc_passthrough_center_size")
        if passthrough_cfg is None or passthrough_cfg == 0:
            self._dc_passthrough_size: tuple[int, int] | None = None
        else:
            if isinstance(passthrough_cfg, (list, tuple)):
                if len(passthrough_cfg) != 2:
                    raise ValueError(
                        "dc_passthrough_center_size must be int or 2-tuple, "
                        f"got {passthrough_cfg!r}"
                    )
                self._dc_passthrough_size = (
                    int(passthrough_cfg[0]),
                    int(passthrough_cfg[1]),
                )
            else:
                s = int(passthrough_cfg)
                self._dc_passthrough_size = (s, s)
            if any(d < 1 for d in self._dc_passthrough_size):
                raise ValueError(
                    "dc_passthrough_center_size dimensions must be >= 1, "
                    f"got {self._dc_passthrough_size}"
                )

        # B-spline coefficients train at a reduced LR / weight decay because their
        # gradient scales differ from linear weights; ratio 1.0 (default) is inert.
        self._kan_lr_ratio, self._kan_weight_decay_ratio = resolve_kan_param_group_ratios(kwargs)
        # Filled when the optimizer builds its groups; train.py stamps it into provenance.
        self.kan_param_group_record: dict[str, Any] | None = None

        # Bind attention blocks now that all sub-modules are registered.
        # Must happen at the end of __init__ (not lazily in forward) so that
        # set_current_smaps() works correctly before the first forward pass.
        self._bind_attention_blocks_to_self()

    @property
    def supports_repetition_fusion(self) -> bool:
        """Whether this instance actually carries a repetition-fusion layer.

        Answers "did the arm opt into NEX/repetition fusion?" -- and, unlike the
        ``hasattr(gen, "rep_fusion")`` test it replaces, it CAN return ``False``.
        That test was a constant ``True`` for every instance of this class,
        because ``__init__`` built ``rep_fusion`` unconditionally (#1173).

        Scope, stated precisely so callers do not over-read it: ``True`` means
        the layer EXISTS and is sized to the declared ``num_repetitions``. It
        does not promise the layer will ever execute -- the forward gate that
        would invoke it expects a rep-major ``[B, Reps, Coils, Ky, Kx]`` layout
        that no dataset in this repository produces (see the comment on that
        gate in :meth:`forward`). Reporting existence honestly is this
        property's job; making the layer reachable is a separate, owner-scoped
        change.
        """
        return self.rep_fusion is not None

    @property
    def supports_5d_input(self) -> bool:
        """Whether ``forward`` can consume a 5D ``[B, C, H, W, D]`` batch.

        This is a DIFFERENT question from
        :attr:`supports_repetition_fusion`, and conflating the two is the defect
        this property exists to end. Two call sites in ``diffusion.py`` decided
        "may I keep this batch 5D?" by testing ``hasattr(gen, "rep_fusion")`` --
        i.e. they asked about repetition fusion in order to learn about 5D
        consumption. One predicate, two invariants (CLAUDE.md non-negotiable
        17).

        The answer here does not depend on ``rep_fusion`` at all: 5D input is
        handled by the ``FourierBridgeNetwork`` backbone, which reshapes
        ``(B, C, H, W, D) -> (B*D, C, H, W)`` internally and restores the depth
        axis afterwards. That path is unconditional, so this is ``True`` for
        every instance -- which is what the old predicate evaluated to in
        practice as well, making this substitution behaviour-preserving while
        the repetition-fusion probe becomes answerable.
        """
        return True

    def set_kan_sample_collection(self, enabled: bool) -> None:
        """Toggle input-sample collection on every KAN layer in the generator.

        Used by the diffusion strategy: enable at the start of training,
        disable after the grid-extension warm-up window (default 10K iters).
        While enabled, each KANLayer caches a small ring buffer of its
        recent inputs on CPU so ``update_kan_grids()`` can fit a new grid
        to the empirical distribution.
        """
        from spectramr.models.blocks.kan_layer import KANLayer

        n = 0
        for module in self.modules():
            if isinstance(module, KANLayer):
                module.set_sample_collection(enabled)
                n += 1
        return None if n else None

    def update_kan_grids(self) -> int:
        """Trigger grid extension on every KAN layer using its buffered samples.

        Plan §9 risk #1 mitigation: call this at iter 2K, 4K, 6K, 8K during
        the first 10K iters of training. After that, freeze (call
        ``set_kan_sample_collection(False)``).

        Returns:
            Number of KAN layers whose grid was actually updated. Layers
            with empty buffers (e.g. layers that haven't seen forwards yet)
            are silently skipped.
        """
        from spectramr.models.blocks.kan_layer import KANLayer

        updated = 0
        for module in self.modules():
            if isinstance(module, KANLayer) and module._collected_samples:
                module.update_grid_from_samples()  # uses buffered samples
                updated += 1
        return updated

    def get_kan_trust_map_telemetry(self) -> dict[str, torch.Tensor]:
        """Return the most-recent KAN ADC trust-map summary statistics.

        Reports four values: mean trust at the central 8% of k-space (DC
        region), mean trust at the periphery, overall mean, and overall
        std. Center > periphery is the expected pattern for any
        well-conditioned MRI reconstruction (low-frequency measurements
        should be trusted more than noisy high-frequency ones).

        Returns an empty dict if the model is not using KAN ADC.
        """
        from spectramr.infrastructure.physics.kan_data_consistency import (
            KANAdaptiveDataConsistency,
        )

        # Values are 0-d device tensors (cloned, since the buffer is rewritten in
        # place by the next forward); the training loop's fused host transfer is
        # the only sync (#1448).
        if isinstance(self.dc_layer, KANAdaptiveDataConsistency):
            prefix = "kan_trust"
        elif isinstance(self.dc_layer, NoiseAdaptiveDataConsistency):
            # Same [center, periphery, mean, std] trust layout; surfaced so the
            # DC shootout can watch for a blob (center≈1, periphery≈1 with a dark
            # output) vs healthy denoising (center high, periphery low).
            prefix = "noise_adaptive_trust"
        else:
            return {}
        stats = self.dc_layer._last_trust_stats.detach().clone()
        names = ("center", "periphery", "mean", "std")
        out = {f"{prefix}/{name}": stats[i] for i, name in enumerate(names)}
        if isinstance(self.dc_layer, WienerGainDataConsistency):
            out["wiener_gain/alpha"] = self.dc_layer._last_gain.detach().clone()
        return out

    def get_kan_gate_telemetry(self) -> dict[str, torch.Tensor]:
        """Aggregate the most-recent KAN gate values across all blocks.

        Each ``KANGatedDualDomainAttention`` instance writes its last batch's
        gate means into a 3-element ``_last_gates`` buffer ``(g_img, g_kspace,
        g_cross)``. This method averages across all blocks in the network and
        returns a flat dict of 0-d device tensors, keyed ``kan_gate/img``,
        ``kan_gate/kspace``, ``kan_gate/cross`` and ``kan_gate/img_active_blocks``
        (the block count). Nothing here syncs; the caller transfers them to the
        host with the rest of its metrics, at its log interval.

        Returns an empty dict if no KAN attention blocks are present.
        """
        from spectramr.models.blocks.dual_domain_attention_kan import (
            KANGatedDualDomainAttention,
        )

        gates = [
            m._last_gates.detach().float()
            for m in self.modules()
            if isinstance(m, KANGatedDualDomainAttention)
        ]
        if not gates:
            return {}
        # Reduced on the gates' own device; a host-side accumulator forced one
        # sync per block, and .tolist() another (#1448).
        means = torch.stack(gates).mean(dim=0)
        return {
            "kan_gate/img": means[0],
            "kan_gate/kspace": means[1],
            "kan_gate/cross": means[2],
            "kan_gate/img_active_blocks": torch.full(
                (), float(len(gates)), device=means.device
            ),
        }

    def get_differential_lr_param_groups(
        self,
        base_lr: float,
        weight_decay: float,
    ) -> list[dict]:
        """Return AdamW-style parameter groups with the B-spline coefficients on a separate LR.

        OptimizerBuilder calls this if it's defined on the model. The 'kan' group
        holds exactly the ``spline_weight`` of every ``KANLayer`` (including those
        inside ``KANAdaptiveDataConsistency``), trained at ``base_lr * kan_lr_ratio``
        and ``weight_decay * kan_weight_decay_ratio``; 'main' holds everything else,
        base weights, gates and attention projections included. The group's size and
        fraction of the trainable model are kept on ``kan_param_group_record``.
        """
        groups, self.kan_param_group_record = build_kan_param_groups(
            self,
            base_lr=base_lr,
            weight_decay=weight_decay,
            lr_ratio=self._kan_lr_ratio,
            weight_decay_ratio=self._kan_weight_decay_ratio,
        )
        # Scalar DC trust weights (soft lambda, Wiener beta, FSDC lambda_hf) are not
        # regularised: decay would pull them towards distrusting the measurement.
        dc_layer = getattr(self, "dc_layer", None)
        trust = [
            p
            for p in (dc_layer.parameters() if dc_layer is not None else ())
            if p.requires_grad and p.ndim == 0
        ]
        if trust:
            trust_ids = {id(p) for p in trust}
            for group in groups:
                group["params"] = [p for p in group["params"] if id(p) not in trust_ids]
            groups = [g for g in groups if g["params"]]
            groups.append(
                {"name": "dc_trust", "params": trust, "lr": base_lr, "weight_decay": 0.0}
            )
        return groups

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Enable activation checkpointing on the backbone.

        ``ModelBuilder`` probes for this method when
        ``optimization.gradient.enable_checkpointing`` is set; without it the
        builder falls back to a generic wrapper that only matches ``nn.Conv2d``/
        ``nn.Linear``/``nn.BatchNorm2d``. On this model that reaches 3.1 % of the
        parameter mass, because the network is built from ``ComplexConv2d``.
        Measured on the ``experiment_11`` shape (4 coils, 256x256, fp32),
        activations saved for backward — the term no ZeRO stage shards:

        =====  ==================  ===================
        Batch  no checkpointing    with checkpointing
        =====  ==================  ===================
        2      13317 MiB           681 MiB
        1      7051 MiB            341 MiB
        =====  ==================  ===================

        Against a fixed state of 1597 MiB (104.7 M params, Adam, fp32) the
        un-checkpointed B=2 total is what OOMs a 16 GB card.

        This is a **training-only** lever. ``_checkpointing_active()`` on the
        backbone requires ``self.training`` and ``torch.is_grad_enabled()``, so
        checkpointing is off under validation by design — there is nothing saved
        for backward to shrink there, and recomputing would be pure cost. A
        validation OOM is a different quantity (the sampler's transient working
        set) on a different device class, and is governed by
        ``validation.loader.chunk_size``; see
        ``DiffusionTrainingStrategy._sample_multistep_chunked``.

        Args:
            enable: Whether the backbone should checkpoint its blocks.
        """
        self.backbone.set_grad_checkpointing(enable)

    def set_current_smaps(self, smaps: torch.Tensor | None) -> None:
        """Stash coil-sensitivity maps so KAN attention blocks can FiLM on them.

        Plan §3.1: KAN attention with ``condition_on_smaps=True`` reads
        from this stash via ``_pull_smaps_from_context()`` rather than
        threading S-maps through every intermediate layer's signature.
        Set automatically at the start of forward() when S-maps are
        present in kwargs; cleared on exit. Tests can also drive this
        directly to exercise the FiLM path with mock data.
        """
        self._current_smaps = smaps
        # Invalidate the pooled-feature cache the attention blocks share
        # (first block to need them recomputes; see
        # ``KANGatedDualDomainAttention._pull_smap_feats_from_context``).
        self._current_smap_feats = None

    def set_current_mask(self, mask: torch.Tensor | None) -> None:
        """Stash the acquisition mask; ``forward`` reads it only inside :meth:`mask_scope`.

        The setter alone never conditions a forward: a stash that outlived the call
        that set it would condition a later bare forward on another batch's support.
        """
        self._current_mask = mask

    @contextmanager
    def mask_scope(self, mask: torch.Tensor | None) -> Iterator[None]:
        """Let a bare ``forward(x, t)`` read ``mask`` for the duration, then restore.

        A reverse sampler or a probe re-enters ``forward`` with no kwargs, so the
        support reaches it out of band; outside a scope a forward without ``mask=``
        gets none. Nested scopes restore the enclosing mask and state on exit.
        """
        prev_mask = getattr(self, "_current_mask", None)
        prev_active = getattr(self, "_mask_scope_active", False)
        self.set_current_mask(mask)
        self._mask_scope_active = True
        try:
            yield
        finally:
            self.set_current_mask(prev_mask)
            self._mask_scope_active = prev_active

    def set_current_contrast_idx(self, contrast_idx: torch.Tensor | None) -> None:
        """Stash the contrast id so every reverse step is conditioned on it.

        The third tensor that must survive the reverse loop, for the reason the
        two above it do: ``sample()`` drives ``forward(x_t, t)`` with no kwargs,
        so a contrast id threaded only through the forward signature reaches
        training and nothing else. Without this the cold multistep validation
        path grades an unconditioned model while training conditions every step.
        """
        self._current_contrast_idx = contrast_idx

    def set_current_reference(self, reference: torch.Tensor | None) -> None:
        """Stash the source-contrast k-space so every reverse step sees it.

        Same route as the smaps, mask and contrast id: the reverse sampler
        re-enters ``forward(x_t, t)`` with no kwargs.
        """
        self._current_reference = reference

    def _gate_complex_unet_knobs(self, kwargs: dict, backbone_type: str) -> None:
        """Route the complex_unet-only knobs to the backbone that consumes them.

        The legality of each knob is :func:`backbone_kwargs_defects`'s, checked
        before this runs. Both families are popped and re-injected rather than left
        in place: a key left in ``kwargs`` reaches ``build_backbone`` and every other
        backbone's ``**kwargs``, where it is swallowed without a word. Each key is
        popped with a literal string, never a loop, because
        ``check_model_kwargs_are_read`` harvests those literals as its vocabulary.
        """
        feature_norm = kwargs.pop("kspace_feature_norm", "none")
        bands = kwargs.pop("radial_band_tokens_bands", 0)
        token_dim = kwargs.pop("radial_band_tokens_dim", None)
        token_max_log = kwargs.pop("radial_band_tokens_max_log", None)
        if backbone_type != "complex_unet":
            return

        kwargs["kspace_feature_norm"] = feature_norm
        if not bands:
            return
        kwargs["radial_band_tokens_bands"] = int(bands)
        kwargs["radial_band_tokens_rungs"] = int(self.num_timesteps)
        if token_dim is not None:
            kwargs["radial_band_tokens_dim"] = int(token_dim)
        if token_max_log is not None:
            kwargs["radial_band_tokens_max_log"] = float(token_max_log)

    def _bind_attention_blocks_to_self(self) -> None:
        """Set ``_parent_generator`` on every KAN attention block.

        Called once at the end of __init__. The block uses this back-ref
        to pull S-maps from the generator's stash without needing to be
        threaded the maps through intermediate forward signatures.

        We must use a ``weakref`` here: a direct ``self`` assignment is
        auto-registered as a child by ``nn.Module.__setattr__`` (which
        treats any nn.Module attribute as a submodule). That creates a
        parent ↔ child cycle and ``self.to(device)`` / ``self.modules()``
        recurse forever (RecursionError: maximum recursion depth
        exceeded). The block already supports a callable ref — see
        ``_pull_smaps_from_context`` — so a weakref drops in cleanly.
        """
        import weakref

        from spectramr.models.blocks.dual_domain_attention_kan import (
            KANGatedDualDomainAttention,
        )

        gen_ref = weakref.ref(self)
        for module in self.modules():
            if isinstance(module, KANGatedDualDomainAttention):
                # Use object.__setattr__ to bypass nn.Module's __setattr__,
                # which would otherwise try to register the weakref as
                # a buffer/parameter. weakref.ref is not a Module so the
                # registration would silently slot it into _non_persistent_*,
                # but skipping the override removes any ambiguity.
                object.__setattr__(module, "_parent_generator", gen_ref)

    def synthetic_forward_probe_kwargs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The forward kwargs the Tier-2 probe must supply for THIS instance.

        ``forward`` takes ``kspace_measured`` and ``mask`` through ``**kwargs``,
        which ``inspect.signature`` cannot enumerate — so
        :func:`~spectramr.infrastructure.validation.forward_probe.synthetic_forward_probe`,
        which builds its call from the signature, called ``model(x)`` and tripped
        :meth:`_assert_measurement_reaches_declared_mechanisms`. Every arm
        declaring data consistency or an output magnitude bound therefore
        false-failed the audit's ``--probe`` gate: the model was healthy and the
        caller was incomplete. Declaring the contract here is the fix that keeps
        the guard — suppressing it for probes would have made the probe validate
        an identity path with DC and the output bound never exercised, which is
        the facade the guard exists to prevent (pitfall #16).

        Instance-scoped on purpose. ``dc_layer`` and ``_output_kspace_clip_ratio``
        are built from ``model_kwargs``, so whether these kwargs are REQUIRED is
        a property of the constructed arm, not of the class — which is why this
        is a method and not a ``ClassVar`` beside
        :attr:`synthetic_forward_probe_skip`.

        The values are deliberately synthetic-but-load-bearing:

        * ``kspace_measured`` — a detached copy of the probe's own input, which
          is what the training strategy passes (``_build_generator_kwargs`` sets
          it to ``input_batch``). Detached so the probe's backward pass measures
          gradient flow through the network, not through the measurement.
        * ``mask`` — a deterministic Cartesian undersampling mask (ACS band plus
          every ``_PROBE_MASK_STRIDE``-th phase-encode line), NOT all-ones. An
          all-ones mask would make hard DC overwrite the whole prediction with
          the measurement, so the probe would grade a copy of its own input.
          Deterministic (no RNG) so the probe's determinism re-runs stay
          bit-comparable.

        Returns:
            Kwargs to merge into the probe's forward call: the mask whenever the
            backbone carries geometry planes, plus the measurement when the arm
            declares a measurement-gated mechanism.
        """
        # A reference arm needs its source contrast on every call; the mirrored
        # input is deterministic and differs from x, so the concat is exercised.
        reference: dict[str, torch.Tensor] = {}
        if self.reference_channels:
            mirrored = x.detach().flip(-1)
            repeats = -(-self.reference_channels // int(x.shape[1]))
            reference["reference"] = mirrored.repeat(1, repeats, 1, 1)[
                :, : self.reference_channels
            ].contiguous()

        # [B, 1, H, W] — broadcasts over the channel axis, so the DC block's
        # `mask.shape[1] > 1` re-slicing is correctly a no-op.
        h, w = int(x.shape[-2]), int(x.shape[-1])
        mask = torch.zeros((int(x.shape[0]), 1, h, w), dtype=torch.float32, device=x.device)
        mask[..., :: self._PROBE_MASK_STRIDE] = 1.0
        acs = max(1, w // 16)
        centre = w // 2
        mask[..., max(0, centre - acs // 2) : centre + acs // 2 + 1] = 1.0
        # The geometry planes are built from the mask, so every v2 forward needs one.
        geometry = {"mask": mask} if self.kspace_geometry_channels else {}

        if (
            self.dc_layer is None
            and self._output_kspace_clip_ratio is None
            and self._dc_passthrough_size is None
        ):
            return {**reference, **geometry}

        probe_kwargs: dict[str, torch.Tensor] = {
            "kspace_measured": x.detach().clone(),
            **reference,
            **geometry,
        }
        if self.dc_layer is not None or self._dc_passthrough_size is not None:
            probe_kwargs["mask"] = mask

        return probe_kwargs

    #: Phase-encode stride for the synthetic probe mask above. 4 keeps the mask
    #: sparse enough that data consistency is a genuine constraint rather than a
    #: full overwrite, while leaving enough sampled bins for the output
    #: magnitude bound to have a meaningful per-sample reference.
    _PROBE_MASK_STRIDE: int = 4

    def _assert_measurement_reaches_declared_mechanisms(
        self, kwargs: dict, x_out: torch.Tensor
    ) -> None:
        """Fail loudly when a declared physics mechanism has nothing to act on.

        Data consistency and the output magnitude bound are BOTH gated on
        ``kwargs["kspace_measured"]``, and both used to skip in silence when it
        was absent or batch-mismatched. That makes ``dc_method: hard``,
        ``physics.data_consistency.enabled`` and ``output_kspace_clip_ratio``
        into facade knobs (pitfall #16): the YAML advertises them, the audit
        accepts them, provenance stamps them, and the forward quietly returns the
        raw backbone proposal instead.

        The signature is unmistakable once you look for it, and it is exactly
        what this guard prevents shipping unannounced. Measured on this
        generator at ``in/out_channels=8`` with compressed k-space peaking at
        4.7, an untrained net returns ``|k|max`` **6.11** with the measurement
        present -- pinned to the band-local ceiling, identical at t=6 and t=27 --
        against **18.95** without it. The experiment_11 ``attention_none``
        snapshot recorded 134.0 at step 6, 22x a bound the measurement-present
        path cannot exceed at any timestep.

        What that unbounded output does downstream is NOT settled, and the
        distinction matters enough to state: an out-of-range but finite
        prediction still renders. Measured against
        ``MetricsTracker._normalize_images`` -- percentile windowing, not
        min-max -- a prediction whose bulk sits at std 1.6 with up to 10% of
        bins past ``DECOMPRESS_MAGNITUDE_CEILING`` saves as an ordinary
        full-range PNG (``png_max=255``). That saver emits a bit-exact black
        image in three cases only: an all-zero prediction, a constant one, or
        one containing NaN. The experiment_11 validation PNGs are bit-exact
        black (``min=max=0``, all 8 samples) beside clean ground truth, so the
        prediction reaching the saver was one of those three -- which an
        unbounded head can reach via the ``expm1`` -> ``* scale`` overflow the
        ceiling above exists to prevent, but which this guard does not by
        itself prove. Establishing which requires the prediction tensor, i.e.
        the pre/post-DC snapshot that the tag split restores.

        Training-only by design: the reverse sampler enforces its own bound via
        ``reverse_clip_ratio`` and legitimately runs without this kwarg.

        A batch MISMATCH warns rather than raises, and the difference is not
        squeamishness. Absence is unambiguously a wiring bug -- the training
        strategy supplies the kwarg in ``_build_generator_kwargs``, so nothing
        legitimate reaches the forward without it. A mismatch, by contrast, has
        one known-legitimate cause that predates this guard: for a genuine 5-D
        (D>1) batch the prediction is restored to ``[B, C, H, W, D]`` while the
        measurement stays flattened to ``[B*D, C, H, W]``, and skipping the
        per-sample ceiling there was a priced decision, not an oversight.
        Raising would break every D>1 arm to fix a D=1 cohort's problem. The
        warning removes the silence, which is the part that actually cost time.

        The Tier-2 audit probe used to trip this raise on every arm that
        declares either mechanism, because
        :func:`~spectramr.infrastructure.validation.forward_probe.synthetic_forward_probe`
        builds its forward call from the signature and both kwargs arrive via
        ``**kwargs``. That is fixed at the caller, not here — see
        :meth:`synthetic_forward_probe_kwargs`, which declares the contract so
        the probe exercises DC and the output bound for real. Suppressing the
        guard under ``torch.no_grad`` or for probes would have been the wrong
        repair: it would make the probe grade an identity path.

        Raises:
            ValueError: When training with DC or the output clip declared and
                ``kspace_measured`` is missing entirely; or when DC specifically
                is declared and ``mask`` is missing, since the projection is
                gated on both and would otherwise skip in silence.
        """
        if not self.training:
            return

        declared = []
        if self.dc_layer is not None:
            declared.append(f"data consistency (dc_method={self.dc_method!r})")
        if self._output_kspace_clip_ratio is not None:
            declared.append(
                f"output magnitude bound (output_kspace_clip_ratio="
                f"{self._output_kspace_clip_ratio})"
            )
        if not declared:
            return

        measured = kwargs.get("kspace_measured")
        if measured is None:
            raise ValueError(
                f"[KSpaceColdDiffusionGenerator] this arm declares "
                f"{' and '.join(declared)}, but the training forward received no "
                f"`kspace_measured`. Both mechanisms are gated on it, so the "
                f"model would return the unconstrained backbone proposal while "
                f"the config, the audit and provenance all report them as "
                f"active. Pass the measured k-space (the training strategy does "
                f"this in `_build_generator_kwargs`), or drop the knobs from "
                f"`model_kwargs`/`physics.data_consistency` so the arm stops "
                f"advertising physics it does not apply."
            )
        # The DC branch is gated on `mask is not None` as WELL as the
        # measurement (see the `if mask is not None and measured_kspace is not
        # None` at the top of the data-consistency block), so a forward carrying
        # `kspace_measured` and no `mask` passed this guard and still skipped
        # data consistency in silence -- the same facade, one level behind the
        # check that was written to close it. Scoped to the mechanisms that
        # actually read the mask: the output magnitude bound needs only the
        # measurement, so an arm declaring the clip alone is untouched.
        if self.dc_layer is not None and kwargs.get("mask") is None:
            raise ValueError(
                f"[KSpaceColdDiffusionGenerator] this arm declares data "
                f"consistency (dc_method={self.dc_method!r}) and the training "
                f"forward received `kspace_measured` but no `mask`. The "
                f"data-consistency projection is gated on BOTH, so it would be "
                f"skipped silently while the config, the audit and provenance "
                f"report it as active -- there is no notion of a 'sampled bin' "
                f"to be consistent with without the sampling mask. Pass the "
                f"mask (the training strategy does this in "
                f"`_build_generator_kwargs`), or drop `dc_method` so the arm "
                f"stops advertising a projection it does not apply."
            )
        if measured.shape[0] != x_out.shape[0] and not self._warned_measurement_batch:
            self._warned_measurement_batch = True
            logger.warning(
                "[KSpaceColdDiffusionGenerator] `kspace_measured` batch %d does "
                "not match the prediction batch %d (shapes %s vs %s), so the "
                "per-sample ceiling for %s cannot broadcast and %s NOT applied "
                "on this forward. Expected for a genuine 5-D (D>1) batch, where "
                "the prediction is restored to [B, C, H, W, D] while the "
                "measurement stays flattened to [B*D, C, H, W]. If this arm is "
                "D=1, it is a wiring bug: flatten both the same way before the "
                "forward. Logged once per model instance.",
                measured.shape[0],
                x_out.shape[0],
                tuple(measured.shape),
                tuple(x_out.shape),
                " and ".join(declared),
                "they are" if len(declared) > 1 else "it is",
            )

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        *,
        return_pre_dc: bool = False,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for k-space cold diffusion.

        Args:
            x: Input k-space tensor (B, C, H, W) or (B, C, H, W, D)
            timesteps: Diffusion timesteps tensor (B,). CRITICAL: Must be passed for proper diffusion!
            return_pre_dc: Also return the pre-DC proposal in eval mode. Declared
                explicitly rather than read out of ``**kwargs`` so a misspelling
                is a ``TypeError`` at the call site instead of a silently
                ignored key that hands back a bare tensor -- the caller would
                then unpack a tensor and measure the wrong half of the model.
            **kwargs: Additional arguments (mask, kspace_measured,
                sensitivity_maps, etc.)

        Returns:
            ``x_out`` alone, or ``(x_out, x_pre_dc)`` when training or when
            ``return_pre_dc`` is set. The annotation used to claim a bare
            ``Tensor`` while the training branch already returned a 2-tuple.
        """
        # Stash S-maps so KAN attention can FiLM on them, and so the S-map
        # concat below can find them. The block walks self via
        # _parent_generator to find the stash.
        #
        # An explicit kwarg wins and refreshes the stash; when there is none we
        # FALL BACK to whatever is already stashed instead of overwriting it
        # with ``None``. ``sample()`` stashes the maps once and then drives the
        # reverse loop, which re-enters this method with only ``(x, t)`` — an
        # unconditional write here wiped the maps at reverse step 1 and left
        # every subsequent step unconditioned.
        smaps = kwargs.get("sensitivity_maps")
        if smaps is None:
            smaps = kwargs.get("smaps")
        if smaps is not None:
            self.set_current_smaps(smaps)
        else:
            smaps = getattr(self, "_current_smaps", None)

        # The stashed mask is read only inside ``mask_scope`` (``sample()`` and the
        # probes): an explicit mask never writes it, so no batch's support can
        # condition a later bare call.
        if kwargs.get("mask") is None and getattr(self, "_mask_scope_active", False):
            kwargs["mask"] = self._current_mask

        # The contrast id is read from the stash but only ``sample()`` writes it,
        # scoped to its reverse loop: a write here outlived the training batch and
        # conditioned every later call that carried no contrast on that batch's (#2435).
        if (
            kwargs.get("contrast_idx") is None
            and getattr(self, "_current_contrast_idx", None) is not None
        ):
            kwargs["contrast_idx"] = self._current_contrast_idx

        reference = kwargs.pop("reference", None)
        if reference is not None:
            self.set_current_reference(reference)
        else:
            reference = getattr(self, "_current_reference", None)

        # Handle TorchIO 5D tensors (B, C, H, W, D) - process each slice
        original_shape = x.shape
        is_5d = x.dim() == 5

        # [PHYSICS INTEGRATION] Learnable Temporal/NEX Fusion
        # Distinguish multi-repetition [B, Reps, Coils, Ky, Kx] from volumetric [B, C, H, W, D]
        # In multi-repetition, the 3rd dimension (index 2) is the number of coils (in_channels)
        # ``supports_repetition_fusion`` first: ``rep_fusion`` is now None unless
        # the arm opted in, so the shape test must not be reached without it.
        #
        # KNOWN-UNREACHABLE ON REAL DATA -- do not read this branch as live.
        # ``x.shape[2] == self.in_channels`` asks for a rep-MAJOR layout
        # ``[B, Reps, Coils, Ky, Kx]``. No dataset in this repository emits that:
        # M4Raw stacks repetitions LAST (``m4raw_dataset.py:411`` ->
        # ``(2, H, W, Reps)``, which TorchIO batches to ``(B, C, H, W, D=Reps)``),
        # so ``shape[2]`` is H (256) and ``in_channels`` is 8 or 16. The
        # comparison cannot hold. Wiring the layout contract is tracked
        # separately -- see the NN16 filing referenced from #1173. Left in place
        # rather than deleted (capability to wire, not dead weight to drop).
        if (
            is_5d
            and self.supports_repetition_fusion
            and x.shape[1] > 1
            and x.shape[2] == self.in_channels
        ):
            x = self.rep_fusion(x)

        # Squeeze all trailing singleton dimensions to ensure 4D or 5D
        # (B, C, H, W, 1) or (B, C, H, W, 1, 1) -> (B, C, H, W)
        while x.dim() > 5 or (x.dim() == 5 and x.shape[-1] == 1):
            if x.shape[-1] == 1:
                x = x.squeeze(-1)
            else:
                break

        # For 5D volumes, delegate to FourierBridgeNetwork which handles 5D properly
        # FourierBridgeNetwork will reshape (B,C,H,W,D) -> (B*D,C,H,W) and back

        # Use 'timesteps' consistently with strategy
        if timesteps is None:
            # F-NOTS-RATELIMIT / 2026-05-20 — the audit-time synthetic
            # forward probe calls ``forward(x)`` without ``timesteps``;
            # falling back to ``t=0`` is by-design for the probe (the
            # model declares ``synthetic_forward_probe_skip =
            # {"identity_collapse"}`` because t=0 IS identity for cold
            # diffusion). However, the previous warning fired on
            # *every* forward, spamming smoke logs by 12+ lines per arm.
            # Rate-limit to one warning per generator instance so the
            # real-world cases where a strategy bug omits timesteps
            # remain visible without the probe-path spam.
            if not getattr(self, "_warned_no_timesteps", False):
                logger.warning(
                    "[KSpaceColdDiffusionGenerator] No timesteps "
                    "provided! Defaulting to t=0 (suppressing further "
                    "occurrences for this instance). The probe path is "
                    "expected; if you see this during training, the "
                    "strategy is failing to pass ``t`` and the model "
                    "will identity-collapse."
                )
                self._warned_no_timesteps = True
            timesteps = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)

        # [REMOVED] log_scaling dynamic scale computation
        # It has been refactored into KSpaceNormalizationTransform in the data pipeline.

        # [ARCHITECT IMPLEMENTATION] Delegate to Fourier Bridge
        # The bridge handles:
        # 1. iFFT (k-space -> Image)
        # 2. UNet (Image Space Denoising)
        # 3. FFT (Image -> k-space)
        # Filter out any conflicting parameters that FourierBridgeNetwork doesn't accept
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k not in ["time", "timesteps", "acquired"]
        }

        # Pass max_timesteps down to bridge network for scaling
        filtered_kwargs["max_timesteps"] = float(self.num_timesteps)
        filtered_kwargs["timestep_scale"] = self._timestep_embedding_scale

        # Contrast conditioning: fuse contrast embedding into backbone kwargs
        contrast_idx = filtered_kwargs.pop("contrast_idx", None)
        if hasattr(self, "contrast_embedding"):
            if contrast_idx is None:
                raise ConfigurationError(
                    f"This generator was built with num_contrasts={self.num_contrasts} but forward "
                    "received no contrast_idx: it would run without its contrast embedding, "
                    "unconditioned and without a trace. Pass contrast_idx (predict: --contrast-idx)."
                )
            c_emb = self.contrast_embedding(contrast_idx.long().to(x.device))
            filtered_kwargs["contrast_emb"] = c_emb

        # [PHYSICS INTEGRATION] Force the prediction onto the SENSE manifold.
        # ``smaps`` was resolved from kwargs-or-stash at the top of this method.

        # [S-MAP CONDITIONING — ONE OWNER, CLAUDE.md #17]
        # Gate on ``self.expects_smaps_concat``, the same predicate that sized
        # the backbone in ``__init__``, plus "the stack arrived un-doubled".
        #
        # The retired gate required ``self.in_channels == self.out_channels *
        # 2``, which is False for every arm in the corpus (in==out==8 there;
        # the doubling is applied to the BACKBONE's width, not the generator's),
        # so this branch was dead code — and it was further filtered by a
        # ``_no_concat_backbones`` set that listed ``complex_unet`` even though
        # ``_INTERNAL_DC_BACKBONES`` does not, i.e. two owners disagreeing about
        # one backbone: 60 of the corpus's 96 cold-diffusion arms.
        #
        # With the gate live, every entry point converges on the trained width:
        # the training strategy and the inference strategy pre-concatenate and
        # arrive at ``2 * in_channels`` (skipped here), while the multi-step
        # validation sampler re-enters with a bare ``in_channels`` measurement
        # and is completed here, per reverse step, against the CURRENT ``x``.
        if (
            smaps is not None
            and self.expects_smaps_concat
            and x.shape[1] == self.in_channels
            and x.dim() == smaps.dim()
        ):
            # [DOMAIN] ``x`` is k-space; ``smaps`` arrives image-domain from
            # the strategy's ``_current_smaps``.  Same treatment as the
            # strategy-side concat, or this fallback path trains on a different
            # stack than the primary one.  LOCAL name: ``smaps`` itself must
            # stay image-domain for ``self.sense_projector`` below.
            from spectramr.infrastructure.physics.coil_sensitivity import (
                prepare_smaps_for_kspace_conditioning,
            )

            # [DTYPE] Match the maps' complexity to ``x`` BEFORE preparing them.
            # ``prepare_smaps_for_kspace_conditioning`` preserves the channel
            # count, so concatenating complex maps onto a real stack does not
            # fail — ``torch.cat`` promotes the whole stack to complex, and the
            # entry transform above then interleaves C complex channels into
            # 2*C real ones. A 4-coil complex map set therefore turns an 8+4
            # concat into a 24-channel backbone input. The strategy-side concat
            # aligns dtype first (diffusion.py::_prepare_diffusion_inputs); do
            # the same here so a caller that hands over raw ESPIRiT output is
            # correct rather than merely lucky. ``smaps`` itself stays
            # image-domain complex for ``self.sense_projector`` below.
            smaps_c = smaps
            if not torch.is_complex(x) and torch.is_complex(smaps_c):
                _spatial = smaps_c.shape[2:]
                _perm = [0, 1, len(_spatial) + 2, *range(2, 2 + len(_spatial))]
                smaps_c = (
                    torch.view_as_real(smaps_c)
                    .permute(*_perm)
                    .reshape(smaps_c.shape[0], -1, *_spatial)
                    .contiguous()
                )
            elif torch.is_complex(x) and not torch.is_complex(smaps_c):
                smaps_c = torch.complex(smaps_c, torch.zeros_like(smaps_c))

            smaps_k, _ = prepare_smaps_for_kspace_conditioning(smaps_c, x, channel_dim=1)
            x = torch.cat([x, smaps_k], dim=1)

        x = prepend_reference(x, reference, self.reference_channels)

        # [5D RESHAPE FIX] Reshape 5D to 4D
        is_volumetric_5d = x.dim() == 5
        if is_volumetric_5d:
            # upstream diffusion strategy permutes 5D inputs to [B, D, C, H, W]
            B, D_dim, C, H, W = x.shape
            x = x.contiguous().view(B * D_dim, C, H, W)
            if timesteps is not None:
                # Repeat timesteps for each depth slice
                timesteps = timesteps.repeat_interleave(D_dim)
            # The contrast embedding was built per volume above, so it follows
            # the timesteps onto the slices or the backbone refuses the batch.
            if filtered_kwargs.get("contrast_emb") is not None:
                filtered_kwargs["contrast_emb"] = expand_contrast_over_slices(
                    filtered_kwargs["contrast_emb"], B * D_dim, D_dim
                )

            # Flatten 5D kwargs to 4D to match x
            for kwarg_name in ["mask", "kspace_measured", "smaps"]:
                if kwarg_name in filtered_kwargs and filtered_kwargs[kwarg_name] is not None:
                    kwarg_tensor = filtered_kwargs[kwarg_name]
                    if kwarg_tensor.dim() == 5:
                        # kspace_measured and smaps are [B, D, C, H, W] due to upstream permute
                        # mask may be [B, C, H, W, D] or expanded incorrectly.
                        if kwarg_name == "mask":
                            # mask is [B, C, H, W, D]
                            k_b, k_c, k_h, k_w, k_d = kwarg_tensor.shape
                            if k_d == D_dim:
                                filtered_kwargs[kwarg_name] = kwarg_tensor.permute(
                                    0, 4, 1, 2, 3
                                ).reshape(k_b * k_d, k_c, k_h, k_w)
                            else:
                                # Fallback if mask is already [B, D, C, H, W]
                                filtered_kwargs[kwarg_name] = kwarg_tensor.contiguous().view(
                                    B * D_dim,
                                    kwarg_tensor.shape[2],
                                    kwarg_tensor.shape[3],
                                    kwarg_tensor.shape[4],
                                )
                        else:
                            # [B, D, C, H, W]
                            k_b, k_d, k_c, k_h, k_w = kwarg_tensor.shape
                            filtered_kwargs[kwarg_name] = kwarg_tensor.contiguous().view(
                                k_b * k_d, k_c, k_h, k_w
                            )

        geometry = self._geometry_planes(x, filtered_kwargs.get("mask"), kwargs.get("acquired"))
        if geometry is not None and not isinstance(self.backbone, FourierBridgeNetwork):
            # PureKSpaceUNet consumes k-space directly, so the planes join here.
            x = torch.cat([x, self._geometry_like(geometry, x)], dim=1)
        elif geometry is not None:
            # The bridge appends them after its entry transform: an iFFT'd geometry
            # plane would be a different, meaningless map.
            filtered_kwargs["geometry"] = geometry

        kwargs_to_pass = filtered_kwargs.copy()

        diffusion_backbones = [
            "diff_varnet",
            "swin_diff_rec",
            "diff_varnet_kan",
            "swin_diff_rec_kan",
            "complex_unet",
            "unet",
            "mamba_unet",
        ]

        if self.backbone_type in diffusion_backbones or _accepts_timesteps(self.backbone):
            kwargs_to_pass["timesteps"] = timesteps

        x_out = self.backbone(x, **kwargs_to_pass)

        if is_volumetric_5d:
            # Reshape back to 5D. ``D_dim`` was unpacked from the 5D input at
            # the start of this branch — ``D`` was the legacy name and raises
            # NameError now.
            _, C_out, H_out, W_out = x_out.shape
            x_out = x_out.view(B, D_dim, C_out, H_out, W_out).permute(0, 2, 3, 4, 1)

        # The backbone's own estimate, before the coil projection and DC, for the
        # debug snapshot: the pre-DC tensor already includes the projection.
        self.last_backbone_output = x_out.detach()
        self.last_backbone_projected = None

        if smaps is not None:
            # [ROBUST MULTI-COIL MAPPING FIX]
            # Verify if x_out has mathematically compatible dimensions to be projected along the physical SENSE manifold.
            # SENSE evaluates a sum across coils. If x_out predicts discrete features (like multi-contrasts)
            # while smaps delineates analog receive coils, PyTorch will trigger a dimensional broadcast crash.
            x_out_c = x_out.shape[1] if torch.is_complex(x_out) else x_out.shape[1] // 2
            smaps_c = smaps.shape[1] if torch.is_complex(smaps) else smaps.shape[1] // 2

            if x_out_c == smaps_c or x_out_c == 1 or smaps_c == 1:
                if self._kspace_log_scaled is None:
                    raise ValueError(
                        "[KSpaceColdDiffusionGenerator] smaps were passed, so the output is "
                        "projected onto the coil subspace, but kspace_log_scaled was never "
                        "supplied: the projection is linear-domain physics and cannot pick a "
                        "domain (#2488). ModelBuilder injects it from data.processing; pass "
                        "kspace_log_scaled when constructing this generator directly."
                    )
                x_out = self.sense_projector(x_out, smaps, log_scaled=self._kspace_log_scaled)
                # Snapshot twin of the raw backbone: separates "unprojected view" from a real defect.
                self.last_backbone_projected = x_out.detach()
            else:
                logger.debug(
                    f"[KSpaceColdDiffusionGenerator] Bypassing SENSE projection: mathematical shape incompatibility "
                    f"between predicted arrays ({x_out_c} complex ch) and hardware S-maps ({smaps_c} complex ch)."
                )

        # Pre-DC prediction: the model's OWN k-space output (post-SENSE) BEFORE
        # the measurement-injecting Data Consistency layer. Exposed to the
        # strategy for OPT-IN pre-DC fidelity supervision (Experiment-11 DC-blob
        # L1+): supervising this forces the net to predict measurement-dependent
        # HF itself instead of leaning on the soft-DC-injected (always-sampled)
        # ACS centre. See DiffusionTrainingStrategy._add_pre_dc_fidelity +
        # losses.reconstruction.lambda_pre_dc_kspace (default 0.0 -> unused).
        # The gain is part of the model's own prediction, so it runs BEFORE the
        # pre-DC capture: an arm supervising ``lambda_pre_dc_kspace`` must score
        # what the model actually emits, not the output one stage earlier.
        if self.radial_band_gain is not None:
            # The gain's timestep conditioning was declared by the class and never
            # passed, so it ran as a per-annulus scalar whatever the arm asked for.
            # The embedding comes from the canonical owner rather than a tenth
            # private sinusoid; this call site has no t_emb of its own.
            gain_embedding = None
            if self.radial_band_gain.time_embed_dim:
                gain_embedding = sinusoidal_timestep_embedding(
                    timesteps,
                    self.radial_band_gain.time_embed_dim,
                    max_timesteps=float(self.num_timesteps),
                    scale=self._timestep_embedding_scale,
                )
            x_out = self.radial_band_gain(x_out, gain_embedding)

        x_pre_dc = x_out

        # Both mechanisms below read `kspace_measured` and both used to skip in
        # silence without it, turning declared physics into a facade. Check once,
        # here, before either can no-op. Training-only (see the method docstring).
        self._assert_measurement_reaches_declared_mechanisms(kwargs, x_out)

        # The measurement and mask the DC layer pinned, so the output clamp below leaves them alone.
        dc_support: tuple[torch.Tensor, torch.Tensor] | None = None
        # [PHYSICS INTEGRATION] Apply Data Consistency if enabled and inputs provided
        if self.dc_layer is not None:
            mask = kwargs.get("mask")
            measured_kspace = kwargs.get("kspace_measured")

            if mask is not None and measured_kspace is not None:
                # 1. Type Alignment: Convert real-stacked to complex if prediction is complex
                if torch.is_complex(x_out) and not torch.is_complex(measured_kspace):
                    # Interleave real/imag channels to complex
                    B_m, C_m, H_m, W_m = measured_kspace.shape
                    real = measured_kspace[:, 0::2, ...]
                    imag = measured_kspace[:, 1::2, ...]
                    measured_kspace = torch.complex(real, imag)

                # 3. Channel Alignment: Match target prediction channels
                # [CROSS-CONTRAST FIX] In cross-contrast mode, kspace_measured is
                # [Source_T1, Target_T2] concatenated. The model predicts only the
                # Target-contrast channels, so we must take the LAST N channels
                # (Target) not the first N (Source) to enforce correct DC.
                # For single-contrast, shapes already match → this branch is a no-op.
                if measured_kspace.shape[1] > x_out.shape[1]:
                    measured_kspace = measured_kspace[:, -x_out.shape[1] :]
                if mask is not None and mask.shape[1] > 1 and mask.shape[1] != x_out.shape[1]:
                    # Extract the target-contrast channels from the mask too
                    mask = mask[:, -x_out.shape[1] :]

                # [STABILIZATION] Rely on sampling mask and Soft DC for center consistency
                # Manual acs_mask replacement causes sharp k-space edges and square artifacts.
                acs_mask = None

                # 4. Final Projection
                #
                # The domain is a property of this class, not of the shapes in
                # front of it: ``FourierBridgeNetwork`` returns k-space on BOTH
                # branches -- ``force_pure_kspace`` only decides whether the exit
                # ``fft2c`` is skipped because the features already are k-space.
                # So ``is_kspace_domain=True`` is unconditional, and a shape
                # mismatch is a misalignment to report rather than a domain to
                # infer (#2147).
                if x_out.shape != measured_kspace.shape:
                    raise ValueError(
                        "[KSpaceColdDiffusionGenerator] data consistency is declared "
                        f"(dc_method={self.dc_method!r}) but the prediction "
                        f"{tuple(x_out.shape)} and the measurement "
                        f"{tuple(measured_kspace.shape)} do not align, so the layer "
                        "cannot be applied. This used to skip DC in silence, which "
                        "trains an arm without the physics it declared. The known "
                        "trigger is a genuine 5-D volume: x_out is restored to "
                        "[B, C, H, W, D] above while kspace_measured stays flattened "
                        "to [B*D, C, H, W]. Flatten the measurement the same way, or "
                        "set use_dc: false if this arm is meant to run without DC."
                    )
                # Called once, unguarded. Every dc_method this generator can build
                # accepts both kwargs, so a `except TypeError` fallback could only
                # ever be entered by a TypeError raised INSIDE a layer's forward --
                # and its retry drops `is_kspace_domain`, whose default is False on
                # 6 of the 7. Measured on [1,2,16,16]: the flagged and unflagged
                # results differ by 0.42 (soft), 0.71 (adaptive), 1.00
                # (noise_adaptive), 1.27 (hard, the cohort's method) and 0.62
                # (kan_adaptive) relative max.
                x_out = self.dc_layer(
                    x_out,
                    measured_kspace,
                    mask,
                    acs_mask=acs_mask,
                    is_kspace_domain=True,
                )
                dc_support = (measured_kspace, mask)

                if not hasattr(self, "_dc_notified") or not self._dc_notified:
                    logger.info(
                        f"🧲 [DC VERIFY] Applied DC: x_out={x_out.shape}, measured={measured_kspace.shape}, mask={mask.shape}"
                    )
                    self._dc_notified = True

        # [DC PASSTHROUGH] Deterministic centre-patch skip-connect, run
        # AFTER the (possibly learned) data-consistency layer. This is
        # the last operation in k-space before returning — anything that
        # AdaptiveDC, SoftDC, or the CNN proposed at the centre patch is
        # hard-overwritten by the measurement at sampled bins. Unsampled
        # bins inside the patch keep their CNN-predicted value.
        if self._dc_passthrough_size is not None:
            _measured = kwargs.get("kspace_measured")
            _mask = kwargs.get("mask")
            x_out = dc_passthrough_center_patch(x_out, _measured, _mask, self._dc_passthrough_size)

        # [SCALE CONTROL] A radial, phase-preserving cap at ``ratio x`` the measured reference, in
        # physical units even under log1p (#1281), applied only where the measured k-space is passed
        # (training). The bounded reverse loops apply ``reverse_clip_ratio`` instead; ``additive`` bounds nothing (#2428).
        if self._output_kspace_clip_ratio is not None:
            _measured = kwargs.get("kspace_measured")
            # Batch-match guard: for genuine 5D (D>1) inputs ``x_out`` is restored
            # to [B, C, H, W, D] above while ``kspace_measured`` stays flattened to
            # [B*D, C, H, W], so the per-sample ceiling would not broadcast. Skip
            # the (best-effort) bound rather than raise; D=1 (the cohort) matches.
            if _measured is not None and _measured.shape[0] == x_out.shape[0]:
                from spectramr.models.diffusion.kspace_process import (
                    apply_ceiling_ratio,
                    band_local_magnitude_ceiling,
                    ceiling_binding_stats,
                    clamp_to_magnitude_ceiling,
                    measured_write_ceiling,
                    paired_magnitude,
                )

                if self._clip_reference == "band_local":
                    _ceil = band_local_magnitude_ceiling(
                        _measured,
                        self._output_kspace_clip_ratio,
                        log_scaled=self._kspace_log_scaled,
                    )
                else:
                    # TRUE complex modulus: ``_measured`` is the same interleaved
                    # Re/Im tensor this method slices with [:, 0::2] / [:, 1::2]
                    # ~75 lines above, so reading it elementwise here was an
                    # internal inconsistency as well as a physics error (#1281).
                    _mag = paired_magnitude(_measured)
                    _mdims = tuple(range(1, _mag.dim()))
                    _ref = _mag.amax(dim=_mdims, keepdim=True).clamp_min(1e-8)
                    _ref = _ref.reshape(_ref.shape[0], *([1] * (x_out.dim() - 1)))
                    _ceil = apply_ceiling_ratio(
                        _ref, self._output_kspace_clip_ratio, self._kspace_log_scaled
                    )
                if dc_support is not None and dc_support[1].dim() == x_out.dim():
                    _dc_measured, _dc_mask = dc_support
                    _ceil = measured_write_ceiling(
                        _ceil, _dc_measured, _dc_mask.amax(dim=1, keepdim=True) > 0, str(self._dc_method)
                    )
                if self.training:
                    self.last_clip_stats = ceiling_binding_stats(x_out, _ceil)
                x_out = clamp_to_magnitude_ceiling(x_out, _ceil, 1e-8)

        # We no longer handle inverse scaling here. The model output is strictly what the backbone proposes.
        # Inverse scaling is done independently (if needed) at the pipeline level.
        if self.training or return_pre_dc:
            # 2nd element = pre-DC prediction (was None) for OPT-IN pre-DC
            # fidelity supervision; consumers that only read [0] are unaffected.
            #
            # ``return_pre_dc`` opens the same door in eval. The alternative --
            # flipping the module to ``.train()`` for a probe -- would update
            # BatchNorm running statistics from validation data and corrupt the
            # model being measured, so the flag exists precisely to avoid it.
            return x_out, x_pre_dc
        else:
            return x_out

    def generate(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor | int | None = None,  # ignored in full loop
        output_domain: str = "image",
        **kwargs,
    ) -> torch.Tensor:
        """
        Full Iterative Diffusion Sampling + Physics Readout
        """
        # 1. Run the Diffusion Loop (K-Space -> Refined K-Space)
        # We assume 'x' is the measurement (starting point for Cold Diff)
        measurement = kwargs.get("measurement", x)
        mask = kwargs.get("mask")

        if mask is None:
            mask = measurement_support(measurement)

        # Run full sampling loop
        kspace_pred = self.sample(
            measurement=measurement,
            mask=mask,
            device=x.device,
            inference_timesteps=kwargs.get("inference_timesteps", self._sampling_steps),
        )

        # 2. Return Raw K-Space if requested
        if output_domain == "kspace":
            return kspace_pred

        # 3. Physics Projection: K-Space -> Image (SENSE)
        # This converts (Coils, H, W) -> (1, H, W) using sensitivity maps
        sensitivity_maps = kwargs.get("sensitivity_maps")

        if sensitivity_maps is not None:
            # Handle Multi-Coil Case (C > 2)
            if kspace_pred.shape[1] > 2 or sensitivity_maps.shape[1] > 1:
                # Ensure complex tensors
                if not torch.is_complex(kspace_pred):
                    # Combine channels if flattened
                    coils = kspace_pred.shape[1] // 2
                    k_complex = torch.complex(kspace_pred[:, :coils], kspace_pred[:, coils:])
                else:
                    k_complex = kspace_pred

                # SENSE Adjoint: Sum( IFFT(k) * conj(S) )
                image_complex = sense_adjoint(k_complex, sensitivity_maps)
                return torch.abs(image_complex).unsqueeze(1)

        # Fallback: Single Coil RSS or Standard IFFT
        # ... existing single coil logic ...
        if kspace_pred.shape[1] == 2:
            complex_pred = torch.complex(kspace_pred[:, 0], kspace_pred[:, 1])
            return torch.abs(ifft2c(complex_pred)).unsqueeze(1)

        return kspace_pred  # Should not reach here if smaps are valid

    def get_output_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """get_output_shape.

        Args:
            input_shape (tuple[int, ...]): Description.
        Returns:
            tuple[int, ...]: Description.
        """
        return (input_shape[0], input_shape[2], input_shape[3])

    def get_parameter_count(self) -> int:
        """get_parameter_count.

        Returns:
            int: Description.
        """
        return sum(p.numel() for p in self.parameters())

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        measurement: torch.Tensor | None = None,
        loss_type: str = "l1",
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Training step: compute losses for cold diffusion training.

        1. Take Ground Truth K-Space (x_start).
        2. Apply Physics Degradation (Acceleration) at step t.
        3. Predict Ground Truth from Degraded input.
        4. Compute Loss.

        Args:
            x_start: Fully sampled ground truth k-space (B, C, H, W)
            t: Timesteps (B,)
            measurement: Optional measurement for data consistency
            loss_type: "l1" or "l2" for reconstruction loss

        Returns:
            Dict with 'loss', 'recon_loss', 'consistency_loss'
        """
        # Ensure 4D shape if 5D with singleton depth (consistency with forward)
        while x_start.dim() > 5 or (x_start.dim() == 5 and x_start.shape[-1] == 1):
            if x_start.shape[-1] == 1:
                x_start = x_start.squeeze(-1)
            else:
                break

        device = x_start.device
        batch_size = x_start.shape[0]

        if self.accelerator is None:
            # Fallback: no accelerator, just predict directly
            prediction = self.forward(x_start, t, mask=torch.ones_like(x_start[:, :1].real))
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            loss = F.l1_loss(prediction, x_start)
            return {
                "loss": loss,
                "recon_loss": loss,
                "consistency_loss": torch.tensor(0.0),
            }

        # 1. Physics Degradation: x_t = D(x_0, t)
        x_t_list = []
        mask_list = []

        timesteps_list = t.tolist()
        for i in range(batch_size):
            timestep = timesteps_list[i]
            mask_i = self.accelerator.get_acceleration_mask(x_start.shape[1:], timestep, device)
            x_t_i = x_start[i] * mask_i.float()
            x_t_list.append(x_t_i)
            mask_list.append(mask_i)

        x_t = torch.stack(x_t_list)
        mask_t = torch.stack(mask_list)

        # 2. Restoration Prediction: x_0_hat = f(x_t, t)
        prediction = self.forward(x_t, t, mask=mask_t.float())
        if isinstance(prediction, tuple):
            prediction = prediction[0]

        # 3. Loss Computation
        if loss_type == "l1":
            recon_loss = F.l1_loss(prediction, x_start)
        else:
            recon_loss = F.mse_loss(prediction, x_start)

        # Frequency Consistency Loss: network must preserve sampled frequencies
        consistency_loss = F.mse_loss(
            prediction * mask_t.float(),
            x_start * mask_t.float(),
        )

        total_loss = recon_loss + 0.1 * consistency_loss

        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "consistency_loss": consistency_loss,
        }

    def sample(
        self,
        measurement: torch.Tensor,
        mask: torch.Tensor | None = None,
        device: torch.device | None = None,
        inference_timesteps: int | None = None,
        start_timestep: int | None = None,
        seed_offset: int = 0,
        **kwargs,  # Accept additional args like 'time' from caller
    ) -> torch.Tensor:
        """Inference: reconstruct from highly accelerated measurement.

        Args:
            measurement: The input undersampled k-space (x_T)
            mask: Optional measurement mask
            device: Device to run on
            inference_timesteps: Number of inference steps (default: use training timesteps)
                Set to smaller value (e.g. 50) for faster inference
            start_timestep: Timestep the measurement is degraded AT — the head of
                the reverse trajectory. ``None`` starts fully degraded
                (``num_timesteps - 1``), which is only correct when the input is at
                MAX acceleration. Forwarded only to samplers whose ``sample()``
                accepts it; for the others a warning is logged rather than the
                argument being dropped in silence (pitfall #15).
            seed_offset: Ensemble member index for a multi-sample validation
                (``validation.sampling.ensemble_samples``): the resolved sampler
                seeds member ``i``'s noise stream with ``sampler_seed + i``.
                Forwarded only when the sampler's ``sample()`` declares it, and a
                NON-ZERO offset the sampler cannot take RAISES rather than warns:
                a member that replays member 0's stream makes the ensemble a
                facade, which is worse than a missing head (non-negotiable 16).
            **kwargs: ``smaps`` / ``sensitivity_maps`` (image-domain, complex or
                real-interleaved) are stashed for the duration of the reverse
                loop so ``forward()`` can condition EVERY step on them. Omit
                them on an arm with ``expects_smaps_concat`` and the backbone
                raises on the first step rather than silently reconstructing
                through an untrained projection (#1326).

        Returns:
            Reconstructed fully-sampled k-space
        """
        _smaps_arg = kwargs.pop("smaps", None)
        if _smaps_arg is None:
            _smaps_arg = kwargs.pop("sensitivity_maps", None)
        _contrast_arg = kwargs.pop("contrast_idx", None)
        _reference_arg = kwargs.pop("reference", None)
        _measured_lines = kwargs.pop("measured_lines", None)
        # Ensure 4D shape if 5D with singleton depth (consistency with forward)
        while measurement.dim() > 5 or (measurement.dim() == 5 and measurement.shape[-1] == 1):
            if measurement.shape[-1] == 1:
                measurement = measurement.squeeze(-1)
            else:
                break

        # Also ensure mask matches measurement shape
        if mask is not None:
            while mask.dim() > 5 or (mask.dim() == 5 and mask.shape[-1] == 1):
                if mask.shape[-1] == 1:
                    mask = mask.squeeze(-1)
                else:
                    break

        if device is None:
            device = measurement.device

        self.eval()
        with torch.no_grad():
            # Route through the SamplerRegistry so YAML's `training.diffusion.sampler`
            # choice (cold_mri / cold_mri+guidance / dps_posterior / pi_gdm / dds /
            # red / pnp_admm) is honoured rather than silently bypassed.
            # Audit finding B3: previously this method directly constructed
            # PhysicsInformedColdDiffusion, ignoring guidance_scale/cond_drop_prob.
            from spectramr.models.diffusion.samplers import get_sampler

            if mask is None:
                mask = measurement_support(measurement)

            # Use configured steps or passed steps
            steps = self._sampling_steps if inference_timesteps is None else inference_timesteps

            sampler_name = getattr(self, "_sampler_name", "cold_mri")
            if self._forward_noise_sigma > 0 and sampler_name.lower() not in ("cold_mri", "coldmri"):
                raise ValueError(
                    f"forward_noise_sigma={self._forward_noise_sigma:g} re-noises the reverse "
                    f"denoiser input, which only the cold_mri sampler runs; sampler "
                    f"{sampler_name!r} would ignore it. Use cold_mri or set forward_noise_sigma: 0."
                )
            sampler_kwargs = {
                "model": self,
                "num_timesteps": self.num_timesteps,
                "max_acceleration": self.kspace_process.max_accel,
                "center_fraction": self.kspace_process.center_fraction,
                "dc_method": getattr(self, "_dc_method", "hard"),
                "dc_weight": getattr(self, "_dc_weight", 1.0),
                "sampling_steps": steps,
            }
            # reverse_mode / reverse_clip_ratio are cold_mri-only knobs; the
            # other samplers' constructors don't accept them, so gate on name.
            if sampler_name.lower() in ("cold_mri", "coldmri"):
                sampler_kwargs["reverse_mode"] = getattr(self, "_reverse_mode", "additive")
                sampler_kwargs["reverse_clip_ratio"] = getattr(self, "_reverse_clip_ratio", 4.0)
                sampler_kwargs["clip_reference"] = getattr(self, "_clip_reference", "global_max")
                # C6 determinism knobs (issue #1286): the wrapper now declares
                # them, so what the YAML set is what the reverse loop runs with.
                sampler_kwargs["sampler_sigma"] = getattr(self, "_sampler_sigma", 0.0)
                sampler_kwargs["sampler_seed"] = getattr(self, "_sampler_seed", None)
                sampler_kwargs["selection_rule"] = getattr(self, "_selection_rule", "fixed")
                sampler_kwargs["refine_alpha"] = getattr(self, "_refine_alpha", None)
                sampler_kwargs["integrity_check"] = getattr(self, "_integrity_check", "off")
                physics = self._reverse_physics
                sampler_kwargs["decay_envelope"] = physics.decay_envelope
                sampler_kwargs["projection"] = physics.projection
                sampler_kwargs["dc_trust"] = physics.dc_trust
                sampler_kwargs["support_radius"] = physics.support_radius
                sampler_kwargs["reverse_dc"] = physics.reverse_dc
                # Unlike the training-path bound this one is NOT opt-in --
                # reverse_clip_ratio always has a value, so a ceiling is always
                # built and the domain is always needed. Refuse rather than
                # assume physical (CLAUDE.md #3, issue #1281).
                if self._kspace_log_scaled is None:
                    raise ValueError(
                        "cold_mri reverse sampling needs kspace_log_scaled, which "
                        "was never supplied to this generator. ModelBuilder "
                        "injects it from data.processing.enable_log_scaling; pass "
                        "kspace_log_scaled explicitly when constructing directly."
                    )
                sampler_kwargs["kspace_log_scaled"] = self._kspace_log_scaled
            sampler = get_sampler(sampler_name, **sampler_kwargs)

            # Iterative Restoration x_T -> x_0.
            #
            # The sampler re-enters ``self.forward(x_t, t)`` with the bare
            # k-space state and no kwargs, so the S-maps have to reach it out
            # of band. Stash them for the loop and RESTORE the previous value
            # afterwards: validation runs between training steps, and leaking a
            # validation batch's maps into the next training step is the
            # mirror-image hazard of the one this fixes.
            # Forward the trajectory head only where the resolved sampler
            # understands it. Passing it unconditionally would TypeError on
            # dps_posterior / pi_gdm / dds / red / pnp_admm, which route through
            # this same method; dropping it in silence would make a caller's
            # explicit request vanish, so an unsupported sampler WARNS
            # (pitfall #15: never quietly ignore a knob that was set).
            _sample_kwargs: dict[str, object] = {}
            _sample_params = inspect.signature(sampler.sample).parameters
            if start_timestep is not None:
                if "start_timestep" in _sample_params:
                    _sample_kwargs["start_timestep"] = int(start_timestep)
                else:
                    logger.warning(
                        "sampler %r does not accept 'start_timestep': requested "
                        "trajectory head t=%d ignored, sampling starts fully "
                        "degraded at t=%d.",
                        sampler_name,
                        int(start_timestep),
                        self.num_timesteps - 1,
                    )
            if seed_offset:
                if "seed_offset" in _sample_params:
                    _sample_kwargs["seed_offset"] = int(seed_offset)
                else:
                    raise ValueError(
                        f"sampler {sampler_name!r} does not accept 'seed_offset': ensemble "
                        f"member {int(seed_offset)} would replay member 0's noise stream, "
                        "and an ensemble of identical samples is a facade. Use the cold_mri "
                        "sampler, or set validation.sampling.ensemble_samples to 1."
                    )

            # The lines each sample acquired (#2160): the reverse rungs must count R
            # over them exactly as the forward process that built the measurement did.
            if _measured_lines is not None:
                if "measured_lines" not in _sample_params:
                    raise ValueError(
                        f"sampler {sampler_name!r} does not accept 'measured_lines': its "
                        "reverse rungs would count R over never-acquired lines while the "
                        "measurement was built over acquired ones."
                    )
                _sample_kwargs["measured_lines"] = _measured_lines

            _prev_smaps = getattr(self, "_current_smaps", None)
            _prev_contrast = getattr(self, "_current_contrast_idx", None)
            _prev_reference = getattr(self, "_current_reference", None)
            if _reference_arg is not None:
                self.set_current_reference(_reference_arg)
            if _smaps_arg is not None:
                self.set_current_smaps(_smaps_arg)
            if _contrast_arg is not None:
                self.set_current_contrast_idx(_contrast_arg)
            try:
                with self.mask_scope(mask):
                    out = sampler.sample(measurement, mask, **_sample_kwargs)
                # The sampler calls forward() bare, so the centre-patch
                # passthrough gated on ``kspace_measured`` never ran on the
                # reverse path (#2162). It runs once here, after the sampler's
                # last DC step, as it runs after the DC layer in training.
                if self._dc_passthrough_size is not None and not isinstance(out, tuple):
                    out = dc_passthrough_center_patch(
                        out,
                        measurement[:, -out.shape[1] :],
                        mask if mask.shape[1] in (1, out.shape[1]) else mask[:, -out.shape[1] :],
                        self._dc_passthrough_size,
                    )
                # The sampler is built per call and discarded, so what its
                # reverse loop actually RAN has to be read off before it goes.
                # Samplers other than cold_mri expose no such record; absent is
                # stamped as absent, never inferred (CLAUDE.md #18).
                self._last_reverse_stats = getattr(sampler, "reverse_stats", None)
                return out
            finally:
                self.set_current_smaps(_prev_smaps)
                self.set_current_contrast_idx(_prev_contrast)
                self.set_current_reference(_prev_reference)

    def _geometry_planes(
        self, x: torch.Tensor, support: torch.Tensor | None, acquired: torch.Tensor | None
    ) -> torch.Tensor | None:
        """``[N, K, H, W]`` geometry planes for the backbone input ``x``, or None.

        ``support`` is the ``mask`` known at this rung: the rung mask in training,
        the current support in the reverse loops. ``acquired`` is the
        scanner-written plane; absent means the acquisition has no line axis and
        every bin is acquirable, the meaning ``measured_lines=None`` has in the
        reverse process.
        """
        if not self.kspace_geometry_channels:
            return None
        if support is None:
            raise ValueError(
                "[KSpaceColdDiffusionGenerator] forward needs mask= (the support known at "
                "this rung): architecture v2 conditions the backbone on it. Training passes "
                "the rung mask and the reverse loops the current support; a bare forward "
                "reads a stashed mask only inside mask_scope() (sample(), the probes). A "
                "caller holding only a zero-filled measurement passes "
                "kspace_geometry_encoding.support_forward_kwargs(model, measurement)."
            )
        n = x.shape[0]
        planes = []
        for plane in (support, acquired):
            if plane is not None and plane.dim() == 4 and plane.shape[0] not in (1, n):
                # A per-volume plane follows the depth flatten onto its slices.
                if n % plane.shape[0]:
                    raise ValueError(
                        f"geometry plane batch {plane.shape[0]} does not divide the "
                        f"backbone batch {n}."
                    )
                plane = plane.repeat_interleave(n // plane.shape[0], dim=0)
            planes.append(plane)
        return kspace_geometry_channels(
            planes[0],
            planes[1],
            tuple(x.shape[-2:]),
            batch=n,
            dtype=x.real.dtype,
            planes=self.kspace_geometry_planes,
        )

    @staticmethod
    def _geometry_like(planes: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """The planes in ``x``'s channel layout: complex channels or (plane, 0) pairs."""
        if torch.is_complex(x):
            return torch.complex(planes, torch.zeros_like(planes))
        return as_interleaved_complex(planes)

    def _refuse_other_arch_version(
        self, module: nn.Module, state_dict: dict, prefix: str, *args: Any
    ) -> None:
        """Raise on a checkpoint from another generator architecture version."""
        found = state_dict.get(prefix + KSPACE_GENERATOR_ARCH_VERSION_KEY)
        saved = 1 if found is None else int(found)
        if saved != KSPACE_GENERATOR_ARCH_VERSION:
            raise ConfigurationError(
                f"Checkpoint is KSpaceColdDiffusionGenerator architecture v{saved}; this "
                f"build is v{KSPACE_GENERATOR_ARCH_VERSION}. v2 replaced the per-real-channel "
                "timestep FiLM with a phase-preserving per-complex-channel gain "
                "(time_proj -> time_gain), conditioned the bottleneck, routed the contrast "
                "embedding into the encoder, and widened every backbone input by "
                f"{2 * KSPACE_GEOMETRY_CHANNELS} k-space geometry channels. The weights "
                "cannot be mapped onto this architecture; retrain the arm."
            )

    def _refuse_mismatched_timestep_scale(
        self, module: nn.Module, state_dict: dict, prefix: str, *args: Any
    ) -> None:
        """Raise when a checkpoint was trained under another timestep scale (#2315)."""
        marker = prefix + _T_SCALE_MARKER_PREFIX
        found = sorted(
            k[len(marker):] for k in state_dict if k.startswith(marker) and "." not in k[len(marker):]
        )
        if len(found) > 1:
            raise ValueError(f"Checkpoint carries more than one timestep-scale marker: {found}.")
        saved = found[0] if found else _DEFAULT_T_SCALE
        if saved != self._timestep_embedding_scale:
            raise ValueError(
                f"Checkpoint was trained with timestep_embedding_scale={saved!r} but this "
                f"model is built with {self._timestep_embedding_scale!r}. The time "
                "embedding codes differ, so the loaded time_mlp would condition on the "
                "wrong rungs. Set model.model_kwargs.timestep_embedding_scale to "
                f"{saved!r}, or start the run fresh."
            )

    @property
    def last_reverse_stats(self) -> dict[str, Any] | None:
        """What the last :meth:`sample` call's reverse loop ran, or ``None``.

        ``terminal_timestep_called`` is the lowest timestep the model was
        actually invoked at. It is NOT the tail of the schedule: an inert step
        is skipped, and on a ladder whose ``mask(0)`` is all-ones the t=0 step is
        always inert, so the rung that IS R=1 is scheduled and never evaluated
        (#2067). This reflects the LAST ``sample`` call, which under chunked
        validation is the last chunk and under a validation ensemble the last
        member. Both replay the same schedule -- chunks split the batch, members
        differ only in the C6 noise stream -- so the three numbers agree across
        them and the record is not order-dependent. ``decay_unfitted`` counts
        samples, so the chunked validation loop sums it across chunks.
        """
        return self._last_reverse_stats

    @property
    def sampler_sigma(self) -> float:
        """C6 reverse-step noise scale the ``cold_mri`` sampler runs with.

        0 means the reverse loop is deterministic. Public so the validation
        ensemble can check the RESOLVED sampler is stochastic before drawing N
        members (the schema checks the declaration; this is the other half).
        """
        return self._sampler_sigma

    @property
    def sampler_seed(self) -> int | None:
        """Seed of the dedicated reverse-step noise generator (``None`` = entropy)."""
        return self._sampler_seed

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "kspace_cold_diffusion"


# Factory functions
def create_kspace_cold_diffusion_generator(
    in_channels: int = 2,
    out_channels: int = 2,
    base_channels: int = 64,
    num_layers: int = 4,
    attention_type: str = "self",
    num_timesteps: int | None = None,
    time_embedding_dim: int = 256,
    time_embedding_type: str = "sinusoidal",
    physics_informed: bool = True,
    num_unrolls: int = 5,
    **kwargs,
) -> nn.Module:
    """create_kspace_cold_diffusion_generator.

    Args:
        in_channels (int): Description.
        out_channels (int): Description.
        base_channels (int): Description.
        num_layers (int): Description.
        attention_type (str): Description.
        num_timesteps (int): Description.
        time_embedding_dim (int): Description.
        time_embedding_type (str): Description.
        physics_informed (bool): Description.
        num_unrolls (int): Description.
    Returns:
        nn.Module: Description.
    """
    generator = KSpaceColdDiffusionGenerator(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        num_layers=num_layers,
        attention_type=attention_type,
        num_timesteps=num_timesteps,
        time_embedding_dim=time_embedding_dim,
        time_embedding_type=time_embedding_type,
        **kwargs,
    )

    return generator


def create_complex_kspace_cold_diffusion_generator(
    base_channels: int = 64,
    num_layers: int = 4,
    attention_type: str = "self",
    num_timesteps: int | None = None,
    time_embedding_dim: int = 256,
    time_embedding_type: str = "sinusoidal",
    physics_informed: bool = True,
    num_unrolls: int = 5,
    **kwargs,
) -> nn.Module:
    """
    [DEPRECATED] Redirects to the Unified KSpaceColdDiffusionGenerator using complex convolutions.
    The "Dual Generator" approach (independent real/imag networks) has been removed due to
    phase incoherence.
    """
    # Force use_complex_conv=True for "complex" requests
    kwargs["use_complex_conv"] = True
    kwargs["activation"] = "complex"

    # Map old 'out_channels' (usually 1 for dual) to strict 2 for unified
    # k-space is always 2 channels (Real, Imag)

    return create_kspace_cold_diffusion_generator(
        in_channels=2,
        out_channels=2,
        base_channels=base_channels,
        num_layers=num_layers,
        attention_type=attention_type,
        num_timesteps=num_timesteps,
        time_embedding_dim=time_embedding_dim,
        time_embedding_type=time_embedding_type,
        physics_informed=physics_informed,
        num_unrolls=num_unrolls,
        **kwargs,
    )
