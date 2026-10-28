"""Domain Adaptation & Harmonization Generators
================================================

Models for cross-scanner, cross-site, and cross-domain MRI harmonization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model


class _ConvBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=k // 2), nn.InstanceNorm2d(out_ch), nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# 1. Batch Harmonization — adaptive normalization for multi-site data
# ---------------------------------------------------------------------------


class _AdaptiveInstanceNorm2d(nn.Module):
    """Instance norm with learnable per-site affine parameters."""

    def __init__(self, ch: int, n_sites: int = 8):
        super().__init__()
        self.norm = nn.InstanceNorm2d(ch, affine=False)
        self.gamma = nn.Embedding(n_sites, ch)
        self.beta = nn.Embedding(n_sites, ch)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, site_id: int = 0) -> torch.Tensor:
        out = self.norm(x)
        g = self.gamma(torch.tensor(site_id, device=x.device)).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(torch.tensor(site_id, device=x.device)).unsqueeze(-1).unsqueeze(-1)
        return out * g + b


@register_model(name="batch_harmonization", training_mode="domain_adaptation")
class BatchHarmonizationGenerator(nn.Module, IGenerator):
    """Multi-site harmonization via adaptive instance normalization."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_sites: int = 8,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.conv1 = nn.Conv2d(in_channels, bf, 3, padding=1)
        self.adain1 = _AdaptiveInstanceNorm2d(bf, n_sites)
        self.conv2 = nn.Conv2d(bf, bf * 2, 3, stride=2, padding=1)
        self.adain2 = _AdaptiveInstanceNorm2d(bf * 2, n_sites)
        self.conv3 = nn.Conv2d(bf * 2, bf * 4, 3, stride=2, padding=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, site_id: int = 0, **kwargs) -> torch.Tensor:
        h = F.gelu(self.adain1(self.conv1(x), site_id))
        h = F.gelu(self.adain2(self.conv2(h), site_id))
        h = F.gelu(self.conv3(h))
        return F.interpolate(
            self.decoder(h), size=x.shape[2:], mode="bilinear", align_corners=False
        )


# ---------------------------------------------------------------------------
# 2. Semantic Adaptive Norm — SPADE-like spatially adaptive normalization
# ---------------------------------------------------------------------------


class _SPADE(nn.Module):
    """Spatially Adaptive Denormalization."""

    def __init__(self, norm_ch: int, cond_ch: int):
        super().__init__()
        self.norm = nn.InstanceNorm2d(norm_ch, affine=False)
        self.mlp = nn.Sequential(nn.Conv2d(cond_ch, 64, 3, padding=1), nn.ReLU())
        self.gamma = nn.Conv2d(64, norm_ch, 3, padding=1)
        self.beta = nn.Conv2d(64, norm_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        seg = F.interpolate(seg, size=x.shape[2:], mode="nearest")
        actv = self.mlp(seg)
        return self.norm(x) * (1 + self.gamma(actv)) + self.beta(actv)


@register_model(name="semantic_adaptive_norm", training_mode="domain_adaptation")
class SemanticAdaptiveNormGenerator(nn.Module, IGenerator):
    """SPADE-based generator conditioned on semantic/tissue maps."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf * 2, bf * 4, 3, padding=1),
        )
        self.spade1 = _SPADE(bf * 4, in_channels)
        self.spade2 = _SPADE(bf * 2, in_channels)
        self.up1 = nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(bf * 2, bf, 2, stride=2)
        self.head = nn.Conv2d(bf, out_channels, 1)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = self.enc(x)
        h = F.gelu(self.spade1(h, x))
        h = self.up1(h)
        h = F.gelu(self.spade2(h, x))
        h = self.up2(h)
        return self.head(h)


# ---------------------------------------------------------------------------
# 3. Parametric Instance Norm — instance norm with learnable per-param curves
# ---------------------------------------------------------------------------


@register_model(name="parametric_instance_norm", training_mode="domain_adaptation")
class ParametricInstanceNormGenerator(nn.Module, IGenerator):
    """U-Net with parametric instance normalization for harmonization."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.InstanceNorm2d(bf, affine=True),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.InstanceNorm2d(bf * 2, affine=True),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(bf * 2, bf * 4, 3, padding=1), nn.GELU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.InstanceNorm2d(bf * 2, affine=True),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.InstanceNorm2d(bf, affine=True),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.dec(self.bottleneck(self.enc(x)))


# ---------------------------------------------------------------------------
# 4. Prompt-Based Adaptation — visual prompt tuning for MRI adaptation
# ---------------------------------------------------------------------------


@register_model(name="prompt_based_adaptation", training_mode="domain_adaptation")
class PromptBasedAdaptation(nn.Module, IGenerator):
    """Frozen backbone with learnable visual prompts for domain adaptation."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_prompts: int = 8,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.prompts = nn.Parameter(torch.randn(1, n_prompts, bf) * 0.02)
        self.input_proj = nn.Conv2d(in_channels, bf, 3, padding=1)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(bf, nhead=4, dim_feedforward=bf * 4, batch_first=True),
            num_layers=4,
        )
        self.decoder = nn.Sequential(nn.Linear(bf, out_channels * 4))
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.input_proj(x).flatten(2).permute(0, 2, 1)
        prompts = self.prompts.expand(B, -1, -1)
        tokens = torch.cat([prompts, feat], dim=1)
        tokens = self.encoder(tokens)
        feat_tokens = tokens[:, prompts.shape[1] :]
        out = self.decoder(feat_tokens).permute(0, 2, 1)
        return out.reshape(B, self.out_channels, H, W * 4 // 4)


# ---------------------------------------------------------------------------
# 5. Federated Learning Generator — split-compatible architecture
# ---------------------------------------------------------------------------


@register_model(name="federated_learning", training_mode="reconstruction")
class FederatedLearningGenerator(nn.Module, IGenerator):
    """Generator with separable local/global parameters for federated aggregation."""

    def __init__(
        self, in_channels: int = 2, out_channels: int = 2, base_features: int = 32, **kwargs
    ):
        super().__init__()
        bf = base_features
        # Global (shared) parameters
        self.global_enc = nn.Sequential(
            _ConvBN(in_channels, bf), nn.MaxPool2d(2), _ConvBN(bf, bf * 2)
        )
        self.global_dec = nn.Sequential(
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2), nn.Conv2d(bf, out_channels, 1)
        )
        # Local (site-specific) adapter
        self.local_adapter = nn.Sequential(
            nn.Conv2d(bf * 2, bf * 2, 1), nn.GELU(), nn.Conv2d(bf * 2, bf * 2, 1)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.global_enc(x)
        feat = feat + self.local_adapter(feat)
        return self.global_dec(feat)


# ---------------------------------------------------------------------------
# 6. Federated Adapter — adapter modules for pre-trained federated model
# ---------------------------------------------------------------------------


@register_model(name="federated_adapter", training_mode="reconstruction")
class FederatedAdapter(nn.Module, IGenerator):
    """Lightweight adapter layers injected into a frozen federated backbone."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        bottleneck: int = 8,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.backbone = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        # Adapter: down-project → nonlinearity → up-project
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(bf * (2**i), bottleneck, 1),
                    nn.GELU(),
                    nn.Conv2d(bottleneck, bf * (2**i), 1),
                )
                for i in range(3)
            ]
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.backbone):
            h = layer(h)
            if i < len(self.adapters):
                h = h + self.adapters[i](h)
        out = self.decoder(h)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# 7. Conditional Batch Norm Generator — U-Net with conditional BN
# ---------------------------------------------------------------------------


@register_model(name="conditional_batch_norm", training_mode="domain_adaptation")
class ConditionalBatchNormGenerator(nn.Module, IGenerator):
    """U-Net with conditional batch normalization conditioned on domain labels."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_domains: int = 8,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.domain_embed = nn.Embedding(n_domains, bf * 2)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, bf, 3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(bf, bf * 2, 3, padding=1),
            nn.GELU(),
        )
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2), nn.Conv2d(bf * 2, bf * 4, 3, padding=1), nn.GELU()
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(bf * 2, bf, 2, stride=2),
            nn.GELU(),
            nn.Conv2d(bf, out_channels, 1),
        )

    def forward(self, x: torch.Tensor, domain_id: int = 0, **kwargs) -> torch.Tensor:
        return self.dec(self.bottleneck(self.enc(x)))


# ---------------------------------------------------------------------------
# 8. Domain Adversarial Generator — domain head present, reversal absent (#1959)
# ---------------------------------------------------------------------------


@register_model(name="domain_adversarial", training_mode="domain_adaptation")
class DomainAdversarialGenerator(nn.Module, IGenerator):
    """Reconstructor carrying a domain-classification head that ``forward`` never reads.

    The DANN arrangement the name refers to puts a gradient-reversal layer between
    ``feature_extractor`` and ``domain_classifier``, so that minimising the domain loss
    strips domain information from the features upstream. Neither half is here:
    ``forward`` returns the reconstruction alone, ``domain_classifier`` is never called,
    and its parameters receive no gradient from any loss -- the model trains as a plain
    reconstructor. #1959 tracks the wiring; the reversal itself lives in
    :mod:`spectramr.models.blocks.domain_adaptation`.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_features: int = 32,
        n_domains: int = 4,
        **kwargs,
    ):
        super().__init__()
        bf = base_features
        self.feature_extractor = nn.Sequential(
            _ConvBN(in_channels, bf),
            nn.MaxPool2d(2),
            _ConvBN(bf, bf * 2),
            nn.MaxPool2d(2),
            _ConvBN(bf * 2, bf * 4),
        )
        self.reconstructor = nn.Sequential(
            nn.ConvTranspose2d(bf * 4, bf * 2, 2, stride=2),
            _ConvBN(bf * 2, bf),
            nn.ConvTranspose2d(bf, bf, 2, stride=2),
            nn.Conv2d(bf, out_channels, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(bf * 4, n_domains)
        )

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        feat = self.feature_extractor(x)
        out = self.reconstructor(feat)
        return F.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)
