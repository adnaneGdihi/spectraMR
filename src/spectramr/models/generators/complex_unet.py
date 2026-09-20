"""Complex U-Net Generator
=======================

A U-Net variant that strictly operates in the complex domain using ComplexConv2d
and complex-aware activations (ModReLU).

Designed for pure k-space processing where phase preservation is critical.
"""

import logging
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.models.blocks.attention import CrossContrastOLMPA
from spectramr.models.blocks.attention_domains import validate_feature_domain
from spectramr.models.blocks.radial_band_tokens import (
    DEFAULT_MAX_LOG_SCALE,
    RadialBandTokens,
)
from spectramr.models.generators.kspace_cold_diffusion_generator import (
    KSpaceDownsampleBlock,
    KSpaceUNetBlock,
    KSpaceUpsampleBlock,
)
from spectramr.models.layers.complex_conv import ComplexConv2d
from spectramr.models.layers.complex_norm import ComplexRMSNorm
from spectramr.models.registry import register_model

logger = logging.getLogger(__name__)

# Advertised values for the ``kspace_feature_norm`` model_kwarg. Validated at
# build (pitfall #15): an unknown value RAISES rather than silently no-op'ing.
_VALID_KSPACE_FEATURE_NORMS = ("none", "rms")


class ComplexTimeInjection(nn.Module):
    """Injects time embedding into complex feature maps.

    Projects a real-valued time embedding into complex space (real + imaginary channels)
    and adds it to the feature map.

    attributes:
        proj (nn.Sequential): Projection MLP from time_dim to 2*channels.
    """

    def __init__(self, channels: int, time_emb_dim: int):
        """Initialize the time injection module.

        Args:
            channels: Number of complex channels (tensor dim = 2*channels).
            time_emb_dim: Dimension of the input time embedding.
        """
        super().__init__()
        # Time embedding is real, but we need to inject it into complex features
        # We project time to 2*channels (real+imag) and add it
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels * 2))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Complex feature map of shape (B, 2*C, H, W).
            emb: Time embedding tensor of shape (B, time_dim).

        Returns:
            torch.Tensor: Feature map with time embedding added (B, 2*C, H, W).
        """
        # Project: (B, time_dim) -> (B, 2*C)
        t_emb = self.proj(emb)
        # Reshape for broadcasting: (B, 2*C, 1, 1)
        t_emb = t_emb[:, :, None, None]
        return x + t_emb


@register_model(
    name="complex_unet",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="kspace",
    output_domain="kspace",
    accepts_complex=True,
    expects_real_imag_interleaved=True,
)
class ComplexUNet(nn.Module):
    """U-Net that enforces Complex algebra throughout the network.

    This architecture avoids the "Real-Valued Fallacy" by ensuring that feature maps
    rotate correctly in the complex plane rather than mixing real/imaginary components
    arbitrarily. It uses `ComplexConv2d` and `ModReLU` (modulus activation).

    Architecture:
        - Input: Pure K-Space (B, 2*C, H, W)
        - Encoder: `KSpaceDownsampleBlock` (ComplexConv + ModReLU)
        - Bottleneck: `KSpaceUNetBlock` (ComplexConv + ModReLU)
        - Decoder: `KSpaceUpsampleBlock` (Transposed ComplexConv + ModReLU)
        - Output: Pure K-Space (B, 2*C, H, W)

    Attributes:
        in_channels (int): Number of input complex channels.
        out_channels (int): Number of output complex channels.
        features (Tuple[int, ...]): Number of features at each U-Net level.
        time_embedding_dim (int): Dimension of time embedding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: tuple[int, ...] = (32, 64, 128, 256),
        time_embedding_dim: int = 256,
        img_size: int | tuple[int, int] = (256, 256),
        padding_mode: str = "zeros",
        feature_domain: str = "kspace",
        **kwargs: Any,
    ):
        """Initialize the Complex U-Net.

        Args:
            in_channels: Number of input channels (complex).
            out_channels: Number of output channels (complex).
            features: Tuple defining the number of features at each depth.
            time_embedding_dim: Dimensionality of the time embedding.
            img_size: Target image size (unused, kept for factory compatibility).
            feature_domain: Domain of the feature maps this U-Net consumes,
                ``"kspace"`` or ``"image"``. ``FourierBridgeNetwork`` derives
                it from ``force_pure_kspace`` and always passes it explicitly;
                the default matches the registered ``input_domain="kspace"``
                contract for direct registry construction. Forwarded to every
                down/up block so domain-sensitive sub-blocks (DualDomainBlock
                + the dual-domain attention family) orient their internal
                FFTs correctly. Raises on unknown values.
            **kwargs: Additional keyword arguments (attention_type, num_contrasts, etc.).
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.time_embedding_dim = time_embedding_dim
        self.feature_domain = validate_feature_domain(feature_domain)
        # Rate-limits the "no max_timesteps" warning in forward() to once per
        # instance rather than once per step.
        self._warned_no_max_timesteps = False

        # Inter-layer scale control. The pure-k-space backbone has no normalization
        # (spatial BN/GN flatten the 1/f spectrum), so activation magnitude drifts
        # over training into the experiment_11 measurement-independent collapse.
        # ``rms`` inserts a phase-/spectrum-preserving ComplexRMSNorm at every
        # stage boundary; ``none`` (default) keeps the backbone byte-identical.
        # Validate + raise on unknown (pitfall #15).
        self.kspace_feature_norm = str(kwargs.pop("kspace_feature_norm", "none")).lower()
        if self.kspace_feature_norm not in _VALID_KSPACE_FEATURE_NORMS:
            raise ValueError(
                f"kspace_feature_norm must be one of {_VALID_KSPACE_FEATURE_NORMS}, "
                f"got {self.kspace_feature_norm!r}."
            )

        # CC-OLMPA cross-contrast attention (post-bottleneck)
        attention_type = kwargs.pop("attention_type", "none")
        attention_type_lc = str(attention_type).lower()
        self.use_cc_olmpa = attention_type_lc == "cross_contrast_olmpa"
        # Block-level attention is forwarded to KSpaceDownsample/Upsample blocks
        # for any non-CC-OLMPA, non-none type. CC-OLMPA stays at the bottleneck
        # only and the encoder/decoder blocks remain attention-free.
        if self.use_cc_olmpa or attention_type_lc == "none":
            self._block_attention_type = "none"
        else:
            self._block_attention_type = attention_type_lc
        # Hyperparams forwarded to KANGatedDualDomainAttention when the
        # block-level attention is 'kan_dual_domain'.
        self._kan_dual_domain_kwargs = kwargs.pop("kan_dual_domain_kwargs", None) or {}

        # Initial projection
        # Input is (B, in_channels, H, W) real-interleaved -> (B, features[0]*2, H, W) real-interleaved
        # Note: ComplexConv2d expects complex_in_channels, complex_out_channels.
        # in_channels is the total real channel count (e.g. 8 for 4 coils).
        # bias=False: k-space constant term creates spatial Dirac delta singularity (Hammernik et al. 2018)
        self.initial_conv = ComplexConv2d(
            in_channels // 2,
            features[0],  # Complex channels
            kernel_size=3,
            padding=1,
            padding_mode=padding_mode,
            bias=False,
        )

        # Encoder
        self.downs = nn.ModuleList()
        current_channels = features[0]
        for feature in features[1:]:
            self.downs.append(
                KSpaceDownsampleBlock(
                    in_channels=current_channels * 2,  # Block expects REAL channels count (2*C)
                    out_channels=feature * 2,  # Block expects REAL channels count
                    use_complex_conv=True,
                    activation="modrelu",
                    time_embedding_dim=time_embedding_dim,
                    attention_type=self._block_attention_type,
                    padding_mode=padding_mode,
                    kan_dual_domain_kwargs=self._kan_dual_domain_kwargs,
                    feature_domain=self.feature_domain,
                )
            )
            current_channels = feature

        # Bottleneck
        bottleneck_out_channels = current_channels * 2 * 2  # real channel count
        self.bottleneck = KSpaceUNetBlock(
            in_channels=current_channels * 2,
            out_channels=bottleneck_out_channels,
            use_complex_conv=True,
            activation="modrelu",
            padding_mode=padding_mode,
        )

        # CC-OLMPA attention at bottleneck resolution (lowest spatial dims)
        if self.use_cc_olmpa:
            phase_safe_dim = kwargs.pop("phase_safe_dim", 128)
            # `num_contrasts` is deliberately NOT read here. This block attends at
            # BOTTLENECK resolution, where the source|target split is 1:1 by
            # construction, so the contrast count is a data-level quantity that this
            # module has no parameter for. The previous spelling popped it into a
            # local and then hardcoded `num_contrasts=1` anyway, which read as a
            # consumed knob while discarding the declared value -- the shape of
            # pitfall #15 that is hardest to spot, because grep finds the key.
            # Target channels = half of bottleneck output (source | target split)
            target_ch = bottleneck_out_channels // 2
            self.cc_olmpa = CrossContrastOLMPA(
                in_channels=target_ch,
                num_contrasts=1,  # 1:1 source→target at feature level
                phase_safe_dim=phase_safe_dim,
            )
        else:
            self.cc_olmpa = None

        # Decoder
        self.ups = nn.ModuleList()
        reversed_features = list(reversed(features))
        current_dec_channels = current_channels * 2 * 2  # Bottleneck output (real)

        for i in range(len(features)):
            target_feature = reversed_features[i]  # complex count
            self.ups.append(
                KSpaceUpsampleBlock(
                    in_channels=current_dec_channels,
                    out_channels=target_feature * 2,
                    use_complex_conv=True,
                    activation="modrelu",
                    time_embedding_dim=time_embedding_dim,
                    attention_type=self._block_attention_type,
                    padding_mode=padding_mode,
                    kan_dual_domain_kwargs=self._kan_dual_domain_kwargs,
                    feature_domain=self.feature_domain,
                )
            )
            current_dec_channels = target_feature * 2

        # Final projection
        # bias=False: k-space constant term creates spatial Dirac delta singularity (Hammernik et al. 2018)
        self.final_conv = ComplexConv2d(
            features[0],
            out_channels // 2,  # out_channels is total real count
            kernel_size=1,
            padding=0,
            padding_mode=padding_mode,
            bias=False,
        )

        # Per-stage inter-layer normalization (phase-/spectrum-preserving). One
        # ComplexRMSNorm per stage boundary keyed to that stage's complex channel
        # count; ``none`` builds Identity so the forward is byte-identical. The
        # final_conv output is deliberately NOT normalized — its scale is owned by
        # the loss / data-consistency / output_kspace_clip_ratio path.
        self.norm_initial = self._make_stage_norm(features[0])
        self.norm_downs = nn.ModuleList(self._make_stage_norm(f) for f in features[1:])
        self.norm_bottleneck = self._make_stage_norm(features[-1] * 2)
        self.norm_ups = nn.ModuleList(self._make_stage_norm(f) for f in reversed(features))

        # Time Embedding MLP (Standard Sinusoidal Projection)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim),
        )

        # Off unless a caller opts in via set_grad_checkpointing(); the default
        # forward must stay allocation-identical to the pre-checkpointing one.
        self.grad_checkpointing = False

        # [RADIAL BAND TOKENS] Opt-in per-annulus, per-rung conditioning read at
        # every up-step, where the radial attenuation is introduced (#2117).
        # Built LAST: a module constructed earlier consumes RNG draws and shifts
        # every later init, so the arm would stop being bit-identical to its
        # control for a reason unrelated to the mechanism. Literal pop keys, never
        # a loop -- check_model_kwargs_are_read harvests those literals.
        bands = int(kwargs.pop("radial_band_tokens_bands", 0) or 0)
        rungs = kwargs.pop("radial_band_tokens_rungs", None)
        token_dim = int(kwargs.pop("radial_band_tokens_dim", 64))
        max_log = float(kwargs.pop("radial_band_tokens_max_log", DEFAULT_MAX_LOG_SCALE))
        self.radial_band_tokens = self._build_band_tokens(
            bands, rungs, token_dim, max_log, features, kwargs
        )

    def _build_band_tokens(
        self,
        bands: int,
        rungs: Any,
        token_dim: int,
        max_log: float,
        features: tuple[int, ...],
        leftover_kwargs: dict[str, Any],
    ) -> RadialBandTokens | None:
        """The block, or ``None``, refusing a knob in its namespace it cannot read.

        This constructor ends in ``**kwargs``, so a misspelled knob is otherwise
        absorbed and the arm runs as its control silently: the advertised-options
        audit skips a signature carrying ``**kwargs``, and the inert-knob scan
        sees declared parameters only.
        """
        unknown = sorted(k for k in leftover_kwargs if k.startswith("radial_band_tokens"))
        if unknown:
            raise ValueError(
                f"ComplexUNet: unrecognised radial band token knob(s) {unknown}. A knob in "
                "this namespace that nothing reads is indistinguishable from one that works."
            )
        if not bands:
            return None
        if self.feature_domain != "kspace":
            raise ValueError(
                "ComplexUNet: radial_band_tokens_bands needs feature_domain='kspace'; a "
                f"radial annulus is not defined on {self.feature_domain!r} feature maps."
            )
        if rungs is None:
            raise ValueError(
                "ComplexUNet: radial_band_tokens_bands was declared without a rung count. "
                "The generator derives it from num_timesteps; a direct caller must pass "
                "radial_band_tokens_rungs so the bank cannot disagree with the process."
            )
        return RadialBandTokens(
            level_channels=tuple(reversed(features)),
            n_bands=bands,
            n_rungs=int(rungs),
            token_dim=token_dim,
            max_log_scale=max_log,
        )

    def _up_step(
        self,
        level: int,
        trunk: torch.Tensor,
        skip: torch.Tensor,
        emb: torch.Tensor | None,
        mask: torch.Tensor | None,
        rung: torch.Tensor,
        full_size: tuple[int, int],
    ) -> torch.Tensor:
        """One up-step, band tokens applied AFTER the stage norm -- before it,
        ``ComplexRMSNorm``'s per-sample scalar divides the contribution back out.
        """
        out = self.norm_ups[level](self.ups[level](trunk, skip, emb, mask))
        return self.radial_band_tokens(out, rung, level=level, full_size=full_size)

    def _make_stage_norm(self, complex_channels: int) -> nn.Module:
        """ComplexRMSNorm for ``kspace_feature_norm='rms'``, else identity."""
        if self.kspace_feature_norm == "rms":
            return ComplexRMSNorm(complex_channels)
        return nn.Identity()

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        """Trade recompute for activation memory on the encoder/bottleneck/decoder.

        This is the hook ``ModelBuilder`` probes for when
        ``optimization.gradient.enable_checkpointing`` is set
        (``infrastructure/training/builders/model_builder.py``). Without it the
        builder falls through to a generic wrapper that only matches
        ``nn.Conv2d``/``nn.Linear``/``nn.BatchNorm2d`` — which on this network is
        30 of 159 leaf modules and **3.1 % of the parameter mass**, because the
        54 ``ComplexConv2d`` layers that carry the model are not ``nn.Conv2d``
        subclasses. The segments checkpointed here (``downs`` + ``bottleneck`` +
        ``ups``) are 99.4 % of the parameters.

        Granularity is the whole point: wrapping an individual convolution saves
        nothing, because a single layer has no interior to discard. Wrapping a
        *block* lets autograd drop every intermediate between the block's input
        and its output and regenerate them during backward.

        Args:
            enable: Whether to checkpoint the block calls in :meth:`forward`.
        """
        self.grad_checkpointing = bool(enable)

    def _checkpointing_active(self) -> bool:
        """Whether this forward should actually checkpoint.

        Recompute only pays for itself when a backward pass will follow. Under
        ``eval()``/``no_grad`` — notably the 28-step reverse sampler used for
        validation — checkpointing would re-run every block for nothing, so the
        plain path is taken instead.
        """
        return self.grad_checkpointing and self.training and torch.is_grad_enabled()

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
        self, x: torch.Tensor, timesteps: torch.Tensor | None = None, **kwargs: Any
    ) -> torch.Tensor:
        """Forward pass.

        forward method for ComplexUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService.
        """
        t_emb = None
        if timesteps is not None:
            # Normalise per-sample by the declared horizon, never by what else
            # shares the batch: gating the division on the batch maximum
            # embedded the reverse sampler's uniform t=1 as 1.0, the code for
            # the LAST rung, while the same t=1 scaled correctly in training
            # whenever a larger t happened to share its batch.
            max_t = kwargs.get("max_timesteps")
            if max_t is None:
                # Direct and probe construction never supplied a horizon, so
                # the legacy 1000-step ceiling stands in -- applied
                # unconditionally, and logged once rather than assumed silently,
                # because an undeclared horizon is the failure this guards.
                if not self._warned_no_max_timesteps:
                    logger.warning(
                        "[ComplexUNet] No max_timesteps supplied; assuming the "
                        "legacy 1000-step ceiling (suppressing further "
                        "occurrences for this instance). Pass "
                        "max_timesteps=<diffusion horizon> to condition on the "
                        "arm's actual schedule."
                    )
                    self._warned_no_max_timesteps = True
                max_t = 1000.0
            elif max_t <= 0:
                raise ValueError(
                    f"ComplexUNet received max_timesteps={max_t!r}; it must be "
                    "a positive diffusion horizon, not a value to guess around."
                )
            timesteps_scaled = timesteps.float() / float(max_t)

            # [STABILIZATION FIX] Use proper sinusoidal embeddings
            t_sin = self._get_sinusoidal_embedding(timesteps_scaled, self.time_embedding_dim)
            t_emb = self.time_mlp(t_sin)

        # The acquisition mask reaches the attention blocks as a forward argument
        # rather than module state, so a block that consumes it is unit-testable in
        # isolation. It is the FULL-resolution mask: each block center-crops it onto
        # its own grid, matching the KSpaceCrop the encoder downsamples with.
        mask = kwargs.get("mask")

        contrast_emb = kwargs.get("contrast_emb")
        combined_emb = t_emb
        if contrast_emb is not None:
            # Fix batch mismatch: contrast_emb may have training batch size
            # while t_emb has validation batch size during cascading validation
            if t_emb is not None and contrast_emb.shape[0] != t_emb.shape[0]:
                contrast_emb = contrast_emb[: t_emb.shape[0]]
            combined_emb = (t_emb + contrast_emb) if t_emb is not None else contrast_emb

        # Initial Conv (normalized once, reused as the deepest skip)
        x_start = self.norm_initial(self.initial_conv(x))

        ckpt = self._checkpointing_active()

        # Encoder — normalize the serial (depth) path where magnitude compounds
        skips = [x_start]
        current = x_start
        for idx, down_block in enumerate(self.downs):
            if ckpt:
                current, skip = checkpoint(down_block, current, t_emb, mask, use_reentrant=False)
            else:
                current, skip = down_block(current, t_emb, mask)
            current = self.norm_downs[idx](current)
            skips.append(skip)

        # Bottleneck
        if ckpt:
            current = self.norm_bottleneck(
                checkpoint(self.bottleneck, current, use_reentrant=False)
            )
        else:
            current = self.norm_bottleneck(self.bottleneck(current))

        # CC-OLMPA: cross-contrast attention at bottleneck resolution
        if self.cc_olmpa is not None:
            B, C, H, W = current.shape
            half_c = C // 2
            # Split source (first half) and target (second half)
            source_feat = current[:, :half_c]  # [B, C/2, H, W]
            target_feat = current[:, half_c:]  # [B, C/2, H, W]
            # Flatten to [B, N, C/2] for CC-OLMPA
            target_seq = target_feat.reshape(B, half_c, H * W).permute(0, 2, 1)
            source_seq = source_feat.reshape(B, half_c, H * W).permute(0, 2, 1)
            # Apply attention: target attends to source
            attended = self.cc_olmpa(target_seq, source_seq)
            # Reshape back and residual
            target_feat = attended.permute(0, 2, 1).reshape(B, half_c, H, W) + target_feat
            current = torch.cat([source_feat, target_feat], dim=1)

        # Decoder
        if self.radial_band_tokens is not None and timesteps is None:
            raise ValueError(
                "ComplexUNet: the radial band tokens are rung-indexed, but this forward "
                "received no timesteps. A bank silently defaulted to rung 0 is a global "
                "embedding that still reports as wired."
            )
        full_size = (x.shape[-2], x.shape[-1])
        for i, up_block in enumerate(self.ups):
            skip = skips[-(i + 1)]
            if self.radial_band_tokens is not None:
                # One checkpointed unit so the per-level scale tensor is
                # recomputed, not retained; widened only when the block is built,
                # so an arm without it keeps the boundary its tests measured.
                args = (i, current, skip, combined_emb, mask, timesteps, full_size)
                current = (
                    checkpoint(self._up_step, *args, use_reentrant=False)
                    if ckpt
                    else self._up_step(*args)
                )
                continue
            if ckpt:
                block_out = checkpoint(
                    up_block, current, skip, combined_emb, mask, use_reentrant=False
                )
            else:
                block_out = up_block(current, skip, combined_emb, mask)
            current = self.norm_ups[i](block_out)

        # Final Output
        out = self.final_conv(current)
        return torch.as_tensor(out)
