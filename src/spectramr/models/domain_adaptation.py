"""Domain Adaptation for Cross-Scanner Generalization

This module provides domain adaptation techniques for MRI
super-resolution to enable cross-scanner generalization. It
includes scanner-specific normalization, domain discriminators,
and adaptation layers.
"""

import torch
import torch.nn.functional as F
from torch import nn

from spectramr.models.blocks.domain_adaptation import grad_reverse
from spectramr.models.registry import register_model


class ScannerSpecificNormalization(nn.Module):
    """Scanner-specific normalization layer."""

    def __init__(
        self,
        num_scanners: int,
        channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        """__init__.

        Args:
            num_scanners (int): Description.
            channels (int): Description.
            eps (float): Description.
            momentum (float): Description.
        """
        super().__init__()
        self.num_scanners = num_scanners
        self.channels = channels
        self.eps = eps
        self.momentum = momentum

        # Scanner-specific affine parameters
        self.scanner_weight = nn.Parameter(torch.ones(num_scanners, channels))
        self.scanner_bias = nn.Parameter(torch.zeros(num_scanners, channels))

        # Running statistics per scanner
        self.register_buffer(
            "scanner_running_mean",
            torch.zeros(num_scanners, channels),
        )
        self.register_buffer("scanner_running_var", torch.ones(num_scanners, channels))

    def forward(self, x: torch.Tensor, scanner_ids: torch.Tensor) -> torch.Tensor:
        """Apply scanner-specific normalization.

        Args:
            x: Input tensor (B, C, H, W)
            scanner_ids: Scanner IDs for each sample (B,)

        Returns:
            Normalized tensor

        """
        if self.training:
            # Update running stats (keep loop for correctness during training)
            # This allows correct calculation of mean/var per scanner even in mixed batches
            for scanner_id in range(self.num_scanners):
                mask = scanner_ids == scanner_id
                if mask.any():
                    scanner_data = x[mask]  # (N_scanner, C, H, W)
                    batch_mean = scanner_data.mean(dim=[0, 2, 3])
                    batch_var = scanner_data.var(dim=[0, 2, 3], unbiased=False)

                    # Update running statistics
                    self.scanner_running_mean[scanner_id].mul_(1 - self.momentum)
                    self.scanner_running_mean[scanner_id].add_(
                        batch_mean * self.momentum,
                    )

                    self.scanner_running_var[scanner_id].mul_(1 - self.momentum)
                    self.scanner_running_var[scanner_id].add_(batch_var * self.momentum)

        # 1. Gather params for the whole batch (Vectorized Lookup)
        # indexing: [Scanners, C] -> [B, C]
        batch_mean = self.scanner_running_mean[scanner_ids]
        batch_var = self.scanner_running_var[scanner_ids]
        batch_weight = self.scanner_weight[scanner_ids]
        batch_bias = self.scanner_bias[scanner_ids]

        # 2. Reshape for broadcasting [B, C, 1, 1]
        batch_mean = batch_mean.view(-1, self.channels, 1, 1)
        batch_var = batch_var.view(-1, self.channels, 1, 1)
        batch_weight = batch_weight.view(-1, self.channels, 1, 1)
        batch_bias = batch_bias.view(-1, self.channels, 1, 1)

        # 3. Apply Normalization (One Fused Kernel)
        return (x - batch_mean) / torch.sqrt(batch_var + self.eps) * batch_weight + batch_bias


# ``input_domain`` is deliberately LEFT UNDECLARED (#1920). This critic does not score an
# MRI signal at all: its first layer is ``AdaptiveAvgPool2d(1)`` -> ``Flatten`` ->
# ``Linear(in_channels, ...)``, i.e. it pools an ENCODER FEATURE MAP to a vector and
# classifies which of ``num_domains`` it came from. No word in the ``Domain`` vocabulary
# ("image", "kspace", "complex_image", "latent", ...) names that input, and a wrong
# declaration is a hard error at ``resolve_conversion`` while an absent one is only
# advisory (NN18: absent is a state to report, never one to infer).
@register_model(role="discriminator", name="domain_discriminator", training_mode="ssl")
class DomainDiscriminator(nn.Module):
    """Domain discriminator for adversarial domain adaptation."""

    def __init__(
        self,
        in_channels: int,
        num_domains: int = 2,
        hiddein_channels: int = 512,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            num_domains (int): Description.
            hiddein_channels (int): Description.
        """
        super().__init__()
        self.num_domains = num_domains

        self.discriminator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hiddein_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hiddein_channels, hiddein_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hiddein_channels // 2, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate domain of input features.

        forward method for DomainDiscriminator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return self.discriminator(x)


class DomainAdaptationLayer(nn.Module):
    """Domain adaptation layer with gradient reversal."""

    def __init__(
        self,
        in_channels: int,
        num_domains: int = 2,
        hiddein_channels: int = 512,
        alpha: float = 1.0,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            num_domains (int): Description.
            hiddein_channels (int): Description.
            alpha (float): Description.
        """
        super().__init__()
        self.alpha = alpha
        self.discriminator = DomainDiscriminator(
            in_channels,
            num_domains,
            hiddein_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal and domain discrimination.

        forward method for DomainAdaptationLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        reversed_x = grad_reverse(x, self.alpha)
        domain_logits = self.discriminator(reversed_x)
        return domain_logits


class FeatureAlignmentLayer(nn.Module):
    """Feature alignment layer for domain adaptation."""

    def __init__(self, channels: int, reduction_ratio: int = 16):
        """__init__.

        Args:
            channels (int): Description.
            reduction_ratio (int): Description.
        """
        super().__init__()
        self.channels = channels

        # Channel attention for alignment
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid(),
        )

        # Spatial attention for alignment
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid(),
        )

        # Alignment transformation
        self.alignment = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align source and target features.

        Args:
            source_features: Features from source domain
            target_features: Features from target domain

        Returns:
            Aligned source and target features

        forward method for FeatureAlignmentLayer.

        Executes PyTorch tensor operations.

        Args:
            source_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            target_features (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Compute channel attention weights
        source_channel_weights = self.channel_attention(source_features)
        target_channel_weights = self.channel_attention(target_features)

        # Apply channel attention
        source_aligned = source_features * source_channel_weights
        target_aligned = target_features * target_channel_weights

        # Compute spatial attention
        source_avg = torch.mean(source_aligned, dim=1, keepdim=True)
        source_max = torch.max(source_aligned, dim=1, keepdim=True)[0]
        source_spatial = torch.cat([source_avg, source_max], dim=1)

        target_avg = torch.mean(target_aligned, dim=1, keepdim=True)
        target_max = torch.max(target_aligned, dim=1, keepdim=True)[0]
        target_spatial = torch.cat([target_avg, target_max], dim=1)

        source_spatial_weights = self.spatial_attention(source_spatial)
        target_spatial_weights = self.spatial_attention(target_spatial)

        # Apply spatial attention
        source_aligned = source_aligned * source_spatial_weights
        target_aligned = target_aligned * target_spatial_weights

        # Final alignment transformation
        source_output = self.alignment(source_aligned)
        target_output = self.alignment(target_aligned)

        return source_output, target_output


@register_model(name="cross_scanner_adaptation", training_mode="adaptation")
class CrossScannerAdaptationNetwork(nn.Module):
    """Complete cross-scanner adaptation network."""

    def __init__(
        self,
        in_channels: int,
        num_scanners: int,
        feature_channels: int = 256,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            num_scanners (int): Description.
            feature_channels (int): Description.
        """
        super().__init__()
        self.num_scanners = num_scanners

        # Scanner-specific normalization
        self.scanner_norm = ScannerSpecificNormalization(num_scanners, in_channels)

        # Feature extractor (shared across scanners)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(inplace=True),
        )

        # Domain adaptation layer
        self.domain_adapter = DomainAdaptationLayer(feature_channels, num_scanners)

        # Feature alignment
        self.feature_aligner = FeatureAlignmentLayer(feature_channels)

        # Scanner-specific reconstruction heads
        self.reconstruction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
                    nn.BatchNorm2d(feature_channels // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature_channels // 2, in_channels, 3, padding=1),
                    nn.Tanh(),  # Output in [-1, 1] range
                )
                for _ in range(num_scanners)
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        scanner_ids: torch.Tensor,
        adaptation_mode: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with cross-scanner adaptation.

        Args:
            x: Input images (B, C, H, W)
            scanner_ids: Scanner IDs (B,)
            adaptation_mode: Whether to return domain logits for adaptation

        Returns:
            Dictionary with reconstruction and optionally domain logits

        """
        if x.size(0) == 0:
            raise RuntimeError("Input batch size cannot be zero")

        # Scanner-specific normalization
        normalized = self.scanner_norm(x, scanner_ids)

        # Feature extraction
        features = self.feature_extractor(normalized)

        # Domain adaptation (if requested)
        domain_logits = None
        if adaptation_mode:
            domain_logits = self.domain_adapter(features)

        # Scanner-specific reconstruction
        reconstruction = torch.zeros_like(x)

        for scanner_id in range(self.num_scanners):
            mask = scanner_ids == scanner_id
            if mask.any():
                scanner_indices = mask.nonzero(as_tuple=True)[0]
                scanner_features = features[scanner_indices]

                # Apply scanner-specific head
                scanner_recon = self.reconstruction_heads[scanner_id](scanner_features)

                reconstruction[scanner_indices] = scanner_recon

        result = {"reconstruction": reconstruction}
        if domain_logits is not None:
            result["domain_logits"] = domain_logits

        return result

    def adapt_features(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adapt features between source and target domains."""
        return self.feature_aligner(source_features, target_features)


@register_model(name="multi_domain_extractor", training_mode="adaptation")
class MultiDomainFeatureExtractor(nn.Module):
    """Feature extractor that handles multiple domains/scanners."""

    def __init__(
        self,
        in_channels: int,
        feature_channels: int = 256,
        num_domains: int = 3,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            feature_channels (int): Description.
            num_domains (int): Description.
        """
        super().__init__()
        self.num_domains = num_domains

        # Domain-specific feature extractors
        self.domain_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, feature_channels, 3, padding=1),
                    nn.BatchNorm2d(feature_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                    nn.BatchNorm2d(feature_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_domains)
            ],
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, feature_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_channels // 2, num_domains),
        )

    def forward(
        self,
        x: torch.Tensor,
        domain_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Extract features with domain awareness.

        Args:
            x: Input images
            domain_ids: Domain/scanner IDs (if None, use all extractors)

        Returns:
            Dictionary with features and domain predictions

        forward method for MultiDomainFeatureExtractor.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            domain_ids (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            dict[str, torch.Tensor]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if domain_ids is None:
            # Use ensemble of all extractors
            all_features = []
            for extractor in self.domain_extractors:
                features = extractor(x)
                all_features.append(features)

            # Average features across domains
            features = torch.mean(torch.stack(all_features), dim=0)

            # Domain classification
            domain_logits = self.domain_classifier(features)
        else:
            # Use domain-specific extractors
            batch_size = x.shape[0]
            features = torch.zeros(
                batch_size,
                self.domain_extractors[0][3].out_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
            )

            domain_logits = torch.zeros(batch_size, self.num_domains, device=x.device)

            for domain_id in range(self.num_domains):
                mask = domain_ids == domain_id
                if mask.any():
                    domain_indices = mask.nonzero(as_tuple=True)[0]
                    domain_features = self.domain_extractors[domain_id](
                        x[domain_indices],
                    )
                    features[domain_indices] = domain_features

                    # Domain classification for this domain
                    domain_domain_logits = self.domain_classifier(domain_features)
                    domain_logits[domain_indices] = domain_domain_logits

        return {"features": features, "domain_logits": domain_logits}


def create_domain_adaptation_network(
    in_channels: int,
    num_scanners: int,
    feature_channels: int = 256,
) -> CrossScannerAdaptationNetwork:
    """Factory function for creating domain adaptation networks."""
    return CrossScannerAdaptationNetwork(in_channels, num_scanners, feature_channels)


def create_multi_domain_extractor(
    in_channels: int,
    num_domains: int,
    feature_channels: int = 256,
) -> MultiDomainFeatureExtractor:
    """Factory function for creating multi-domain feature extractors."""
    return MultiDomainFeatureExtractor(in_channels, feature_channels, num_domains)


def compute_domain_adversarial_loss(
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
) -> torch.Tensor:
    """Compute domain adversarial loss for adaptation."""
    loss = F.cross_entropy(domain_logits, domain_labels)
    return loss


def compute_feature_alignment_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
) -> torch.Tensor:
    """Compute feature alignment loss between domains."""
    # Use maximum mean discrepancy (MMD) for feature alignment
    source_mean = torch.mean(source_features, dim=0)
    target_mean = torch.mean(target_features, dim=0)

    mmd_loss = torch.mean((source_mean - target_mean) ** 2)
    return mmd_loss


def adapt_model_to_scanner(
    model: nn.Module,
    target_scanner_data: torch.Tensor,
    num_epochs: int = 10,
) -> nn.Module:
    """Adapt a trained model to a new scanner using domain adaptation.

    Args:
        model: Pre-trained model
        target_scanner_data: Data from target scanner
        num_epochs: Number of adaptation epochs

    Returns:
        Adapted model

    """
    model.train()

    from spectramr.infrastructure.training.builders import OptimizationBuilder

    # Simple adaptation: fine-tune on target scanner data
    optimizer = OptimizationBuilder.create_single_optimizer(model.parameters(), learning_rate=1e-4)

    for _epoch in range(num_epochs):
        # Forward pass on target data
        target_output = model(target_scanner_data)

        # Compute adaptation loss (reconstruction loss, domain loss, etc.)
        # This is a simplified example - in practice, you'd use more # IMPL
        # sophisticated adaptation
        loss = torch.mean((target_output - target_scanner_data) ** 2)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
