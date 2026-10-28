"""Perceptual Loss (VGG-based) implementation.

This module implements Perceptual Loss (also known as VGG Loss or Content Loss)
using a pre-trained VGG network. It is commonly used in GANs and super-resolution
to preserve high-frequency details and perceptual quality.

Metrics can be optionally computed during loss calculation by setting
`compute_metrics=True` in the constructor.
"""

from typing import Any

import torch
import torch.nn as nn

from spectramr.models.losses.metrics_aware_loss import MetricsAwareLossMixin
from spectramr.models.losses.registry import register_loss

#: VGG19 ``features`` index of each ReLU output, shared by every extractor here.
VGG19_RELU_INDICES: dict[str, int] = {
    "relu1_1": 1,
    "relu1_2": 3,
    "relu2_1": 6,
    "relu2_2": 8,
    "relu3_1": 11,
    "relu3_2": 13,
    "relu3_3": 15,
    "relu3_4": 17,
    "relu4_1": 20,
    "relu4_2": 22,
    "relu4_3": 24,
    "relu4_4": 26,
    "relu5_1": 29,
    "relu5_2": 31,
    "relu5_3": 33,
    "relu5_4": 35,
}


def rss_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Root-sum-of-squares magnitude over the channel axis, kept as one channel.

    A real ``[B, C, ...]`` input with even ``C`` is read as interleaved
    ``[R1, I1, R2, I2, ...]``; a complex input is used as is. ``vector_norm`` is
    used rather than ``sqrt(sum(|z|**2))`` because its gradient at an all-zero
    pixel is zero, where the square root's is NaN.
    """
    if x.is_complex():
        mag: torch.Tensor = torch.linalg.vector_norm(x, dim=1, keepdim=True)
        return mag
    b, c = x.shape[:2]
    pairs = x.reshape(b, c // 2, 2, *x.shape[2:])
    mag = torch.linalg.vector_norm(pairs, dim=(1, 2))
    return mag.unsqueeze(1)


def _check_layer_names(names: Any, owner: str) -> None:
    """Raise on a VGG19 layer name the index table does not know."""
    unknown = sorted(set(names) - set(VGG19_RELU_INDICES))
    if unknown:
        raise ValueError(
            f"{owner}: unknown VGG19 layer(s) {unknown}; expected names from "
            f"{sorted(VGG19_RELU_INDICES)}."
        )


@register_loss(name="perceptual", aliases=["PerceptualLoss", "vgg", "content"], domain="image")
class PerceptualLoss(MetricsAwareLossMixin, nn.Module):
    r"""Perceptual Loss using VGG19 features.

    **DOMAIN**: IMAGE-SPACE ONLY
    **Input**: [B, 1, H, W] or [B, 3, H, W] (auto-converts to magnitude/RGB)
    **3D Support**: No (2D VGG patches)

    Computes L1/L2 distance between feature maps of pre-trained VGG19.
    Do NOT use with k-space [B, 2, H, W].
    Mathematical Formulation:
    .. math::

        \mathcal{L}_{Perceptual} = \sum_l w_l \| \Phi_l(\hat{y}) - \Phi_l(y) \|_2^2"""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(
        self,
        layer_weights: dict[str, float] | None = None,
        use_input_norm: bool = True,
        range_norm: bool = False,
        criterion: str = "l1",
        compute_metrics: bool = False,
    ):
        """Initialize Perceptual Loss.

        Args:
            layer_weights: Dictionary mapping layer names to weights.
                           Default: {'relu5_4': 1.0}
            use_input_norm: Whether to normalize input with ImageNet stats.
            range_norm: Whether to normalize input from [-1, 1] to [0, 1].
            criterion: Loss criterion ('l1' or 'mse').
            compute_metrics: Whether to compute metrics during forward pass.
        """
        try:
            from torchvision import models  # type: ignore[import-untyped]
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise ImportError(
                "PerceptualLoss requires torchvision, which is a core dependency "
                "(see pyproject.toml) -- this failure means the install is broken "
                "or ABI-mismatched, not that an extra is missing. torchvision ships "
                "compiled ops built against a fixed torch ABI, so it must resolve "
                "from the same index as torch ('pytorch-cu126' in [tool.uv.sources]); "
                "a torchvision resolved from PyPI instead raises RuntimeError: "
                "operator torchvision::nms does not exist. Reinstall torch and "
                "torchvision together from that index."
            ) from exc

        super().__init__()

        if layer_weights is None:
            layer_weights = {"relu5_4": 1.0}
        # An unknown name would contribute nothing and train the term at zero.
        _check_layer_names(layer_weights, "PerceptualLoss")
        self.layer_weights = layer_weights
        self.use_input_norm = use_input_norm
        self.range_norm = range_norm
        self.compute_metrics_flag = compute_metrics

        self.criterion: nn.Module
        if criterion == "l1":
            self.criterion = nn.L1Loss()
        elif criterion == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

        # Load VGG19
        # We use the features only
        try:
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        except Exception as e:
            raise ImportError(
                "Could not load VGG19 model. Ensure torchvision is installed "
                "and internet connection is available for first run."
            ) from e

        self.vgg_layers = vgg
        self.vgg_layers.eval()

        # Freeze parameters
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.layer_indices = dict(VGG19_RELU_INDICES)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
        """Compute perceptual loss.

        Args:
            x: Input image (generated)
            y: Target image (real)

        Returns:
            Scalar loss value, or tuple of (loss, metrics) if compute_metrics=True

        forward method for PerceptualLoss.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            y (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor | tuple[torch.Tensor, dict[str, float]]: Dictionary containing tensor outputs.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Handle input range [-1, 1] -> [0, 1]
        if self.range_norm:
            x = (x + 1) / 2
            y = (y + 1) / 2

        # Auto-detect complex multi-channel input [B, C, H, W] where C is even
        if x.is_complex() or x.shape[1] % 2 == 0:
            x = rss_magnitude(x)
        if y.is_complex() or y.shape[1] % 2 == 0:
            y = rss_magnitude(y)

        # Handle grayscale (1 channel) -> RGB (3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if y.shape[1] == 1:
            y = y.repeat(1, 3, 1, 1)

        # Normalize with ImageNet stats
        if self.use_input_norm:
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std

        # A NaN or Inf here is the divergence signal. Returning a zero loss made
        # the term vanish on exactly the batches that carried it, and the run
        # continued with a corrupted model (2026-09-03, non-negotiable 3).
        if not torch.isfinite(x).all():
            raise ValueError(
                "PerceptualLoss: non-finite values in the prediction after normalisation."
            )
        if not torch.isfinite(y).all():
            raise ValueError("PerceptualLoss: non-finite values in the target after normalisation.")

        loss = torch.tensor(0.0, device=x.device)

        # Extract features
        x_features = self._extract_features(x)
        y_features = self._extract_features(y)

        for layer_name, weight in self.layer_weights.items():
            if layer_name in x_features and layer_name in y_features:
                loss = loss + (
                    weight * self.criterion(x_features[layer_name], y_features[layer_name])
                )

        if self.compute_metrics_flag:
            metrics = self.compute_metrics(x, y, loss)
            return loss, metrics
        return loss

    def _extract_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract the requested layers, running VGG only as deep as the last one."""
        wanted = {self.layer_indices[name]: name for name in self.layer_weights}
        last = max(wanted)
        features: dict[str, torch.Tensor] = {}
        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if idx in wanted:
                features[wanted[idx]] = x
            if idx >= last:
                break
        return features

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_value: torch.Tensor,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute perceptual-specific metrics.

        Args:
            pred: Predicted tensor
            target: Target tensor
            loss_value: Computed loss value
            **kwargs: Additional arguments

        Returns:
            Dictionary with perceptual metrics
        """
        metrics = {
            "loss": (loss_value.item() if loss_value.numel() == 1 else loss_value.mean().item()),
            "pred_mean": pred.mean().item(),
            "pred_std": pred.std().item(),
            "target_mean": target.mean().item(),
            "target_std": target.std().item(),
            "pred_range": (pred.max().item() - pred.min().item()),
            "target_range": (target.max().item() - target.min().item()),
        }

        # Add feature statistics if available
        with torch.no_grad():
            pred_features = self.forward_to_layers(pred)
            target_features = self.forward_to_layers(target)

            for layer_name, (pred_feat, target_feat) in zip(
                pred_features.keys(),
                zip(pred_features.values(), target_features.values(), strict=False),
                strict=False,
            ):
                if pred_feat is not None and target_feat is not None:
                    feat_diff = (pred_feat - target_feat).abs().mean().item()
                    metrics[f"feat_diff_{layer_name}"] = feat_diff

        return metrics

    def forward_to_layers(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        """Extract features at specified layers.

        One pass through VGG serves every requested layer; restarting at layer 0
        per name would feed a later layer's activations back into ``conv1_1``.

        Args:
            x: Input tensor

        Returns:
            Dictionary mapping layer names to feature tensors
        """
        # Normalize input
        if self.range_norm:
            x = (x + 1) / 2
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        with torch.no_grad():
            features = self._extract_features(x)
        return {name: feat.detach() for name, feat in features.items()}


class VGGFeatureExtractor(nn.Module):
    """VGG19 Feature Extractor for Perceptual Loss."""

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, layers: list[str] | None = None, use_input_norm: bool = True) -> None:
        """__init__.

        Args:
            layers (Any): Description.
            use_input_norm (Any): Description.
        """
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise ImportError(
                "VGGFeatureExtractor requires torchvision, which is a core "
                "dependency (see pyproject.toml) -- this failure means the "
                "install is broken or ABI-mismatched, not that an extra is "
                "missing. torchvision ships compiled ops built against a fixed "
                "torch ABI, so it must resolve from the same index as torch "
                "('pytorch-cu126' in [tool.uv.sources]); a torchvision resolved "
                "from PyPI instead raises RuntimeError: operator torchvision::nms "
                "does not exist. Reinstall torch and torchvision together from "
                "that index."
            ) from exc

        super().__init__()
        self.use_input_norm = use_input_norm
        # NOTE: the modern weights= API is the ONLY correct call. The old
        # `try weights= / except TypeError: pretrained=True` fallback was
        # deleted because it MASKED the real error: a weights-download failure
        # surfaced as a confusing second exception from the fallback arm. Let
        # the real error surface instead.
        #
        # Corrected 2026-08-13, measured on the pinned torchvision 0.28.0: an
        # earlier revision of this comment said the kwarg "no longer exists in
        # torchvision >= 0.15, so the fallback itself raised TypeError". It
        # does still exist — deprecated in 0.13, still resolving, emitting two
        # UserWarnings. Avoiding it remains right; the TypeError rationale was
        # not (#961).
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.vgg_layers = vgg
        self.vgg_layers.eval()
        for param in self.vgg_layers.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Default layers if None
        if layers is None:
            layers = ["relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1"]
        _check_layer_names(layers, "VGGFeatureExtractor")
        self.output_layers = layers
        self.layer_indices = dict(VGG19_RELU_INDICES)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            list[torch.Tensor]: Description.

        forward method for VGGFeatureExtractor.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            list[torch.Tensor]: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        if self.use_input_norm:
            x = (x - self.mean) / self.std

        # Outputs come back in network order, one per requested name; the layers
        # past the deepest requested one are not evaluated.
        wanted = [self.layer_indices[name] for name in self.output_layers]
        last = max(wanted)
        outputs: list[torch.Tensor] = []
        for idx, layer in enumerate(self.vgg_layers):
            x = layer(x)
            for _ in range(wanted.count(idx)):
                outputs.append(x)
            if idx >= last:
                break
        return outputs


class VGGPerceptualLoss(nn.Module):
    """
    Robust Multi-Scale Perceptual Loss using sliced VGG19.

    Efficiently computes loss by slicing the network and forwarding
    through sequential blocks, accumulating loss at each scale.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, feature_layers: list[int] | None = None, use_l1: bool = True) -> None:
        """__init__.

        Args:
            feature_layers (list[int] | None): VGG layer indices for feature extraction.
                Defaults to [3, 8, 17, 26, 35].
            use_l1 (bool): Use L1 loss (True) or MSE loss (False).
        """
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise ImportError(
                "VGGPerceptualLoss requires torchvision, which is a core "
                "dependency (see pyproject.toml) -- this failure means the "
                "install is broken or ABI-mismatched, not that an extra is "
                "missing. torchvision ships compiled ops built against a fixed "
                "torch ABI, so it must resolve from the same index as torch "
                "('pytorch-cu126' in [tool.uv.sources]); a torchvision resolved "
                "from PyPI instead raises RuntimeError: operator torchvision::nms "
                "does not exist. Reinstall torch and torchvision together from "
                "that index."
            ) from exc

        super().__init__()
        if feature_layers is None:
            feature_layers = [3, 8, 17, 26, 35]
        # Modern weights= API only; the removed `pretrained=True` fallback
        # itself raises TypeError on torchvision >= 0.15 and masked the real
        # error. Let the real error surface.
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        self.slices = nn.ModuleList()

        # Create slices for relu1_2, relu2_2, relu3_4, etc.
        prev_idx = 0
        # feature_layers indices correspond to Relu layers typically.
        # [3, 8, 17, 26, 35] are standard checks.
        for idx in feature_layers:
            self.slices.append(vgg[prev_idx:idx])
            prev_idx = idx

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss() if use_l1 else nn.MSELoss()

        # ImageNet Mean/Std for normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # Assume x is [0, 1] or similar scale.
        # If input is [-1, 1], shift first. We assume [0, 1] for now or handle outside.
        # PerceptualLoss standard behavior.

        # Auto-detect complex multi-channel input [B, C, H, W] where C is even
        """normalize.

        Args:
            x (Any): Description.
        Returns:
            Any: Description.
        """
        if x.is_complex() or x.shape[1] % 2 == 0:
            x = rss_magnitude(x)

        # 1-channel to 3-channel
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        return (x - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """forward.

        Args:
            pred (Any): Description.
            target (Any): Description.
        Returns:
            Any: Description.

        forward method for VGGPerceptualLoss.

        Executes PyTorch tensor operations.

        Args:
            pred (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            target (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)

        loss = torch.tensor(0.0, device=pred.device)
        x = pred_norm
        y = target_norm

        for block in self.slices:
            x = block(x)
            y = block(y)
            loss = loss + self.criterion(x, y)

        return loss


@register_loss(name="dists", aliases=["DISTS", "structure_texture_loss"])
class DISTS(nn.Module):
    """Deep Image Structure and Texture Similarity Loss.

    Unified metric combining structure (edges) and texture (statistics) perception.

    Reference:
        Ding et al. "Image Quality Assessment: Unifying Structure and Texture Similarity"
        IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2020.

    Key innovation:
        - Structure branch: Gradient-based features (detect edges, shapes)
        - Texture branch: Statistical features (local mean, variance, contrast)
        - Unified weighting: Adaptive emphasis on perceptually relevant differences

    Advantage over alternatives:
        - SSIM alone: Misses texture details
        - Perceptual (L2 on VGG): Misses structure at certain scales
        - LPIPS alone: Texture-biased
        - DISTS: Balances both simultaneously
    """

    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self, backbone: str = "vgg19", requires_grad: bool = False):
        """Initialize DISTS loss.

        Args:
            backbone: 'vgg19' or 'vgg16' for feature extraction
            requires_grad: If False, freeze backbone weights
        """
        try:
            from torchvision import models
        except (ImportError, RuntimeError) as exc:  # pragma: no cover
            raise ImportError(
                "DISTS requires torchvision, which is a core dependency (see "
                "pyproject.toml) -- this failure means the install is broken or "
                "ABI-mismatched, not that an extra is missing. torchvision ships "
                "compiled ops built against a fixed torch ABI, so it must resolve "
                "from the same index as torch ('pytorch-cu126' in [tool.uv.sources]); "
                "a torchvision resolved from PyPI instead raises RuntimeError: "
                "operator torchvision::nms does not exist. Reinstall torch and "
                "torchvision together from that index."
            ) from exc

        super().__init__()

        # Modern `weights=` API, matching PerceptualLoss above and
        # VGGFeatureExtractor in gan_loss_library. `pretrained=True` is the
        # torchvision 0.12 spelling: deprecated in 0.13, still resolving on
        # 0.28.0 but emitting two UserWarnings per construction, and slated
        # for removal. It reached here because every DISTS test is
        # @pytest.mark.slow behind a weights download, so no fast-lane run
        # ever constructed the class (#961).
        if backbone == "vgg19":
            vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            self.layer_names = ["relu1_2", "relu2_2", "relu3_4", "relu4_4", "relu5_4"]
            max_layer = 30
        elif backbone == "vgg16":
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.layer_names = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
            max_layer = 30
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.features = vgg.features[:max_layer]

        # Freeze backbone
        if not requires_grad:
            for param in self.features.parameters():
                param.requires_grad = False

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.num_layers = len(self.layer_names)

    def _extract_features(
        self, x: torch.Tensor, layers: list[int] | None = None
    ) -> list[torch.Tensor]:
        """Extract multi-scale features from VGG.

        Args:
            x: [B, C, H, W] RGB image (assume no preprocessing)
            layers: Which layer indices to extract (default: all)

        Returns:
            List of feature maps at different scales
        """
        # Expect x ∈ [0, 1] or [-1, 1]
        # Normalize to ImageNet stats
        # [FIX] Robust check for [-1, 1] range:
        # Standard [0, 1] data shouldn't be negative. [-1, 1] data will be < -0.1
        if x.min() < -0.05 or x.max() > 1.1:
            x = (x + 1) / 2

        x = (x - self.mean) / self.std

        features: list[torch.Tensor] = []
        layer_idx = 0

        for module in self.features.children():
            x = module(x)

            # Collect at specific ReLU layers
            if isinstance(module, nn.ReLU):
                if layers is None or layer_idx in layers:
                    features.append(x)
                layer_idx += 1

        return features

    def forward(
        self,
        x_pred: torch.Tensor,
        x_target: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DISTS loss.

        Args:
            x_pred: [B, C, H, W] Predicted image
            x_target: [B, C, H, W] Target image
            return_components: If True, return (loss, loss_structure, loss_texture)

        Returns:
            loss: Scalar loss value

        forward method for DISTS.

        Executes PyTorch tensor operations.

        Args:
            x_pred (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            x_target (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            return_components (bool): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # Ensure RGB
        # Auto-detect complex multi-channel input [B, C, H, W] where C is even
        if x_pred.is_complex() or x_pred.shape[1] % 2 == 0:
            x_pred = rss_magnitude(x_pred)
        if x_target.is_complex() or x_target.shape[1] % 2 == 0:
            x_target = rss_magnitude(x_target)

        if x_pred.shape[1] == 1:
            x_pred = x_pred.repeat(1, 3, 1, 1)
        if x_target.shape[1] == 1:
            x_target = x_target.repeat(1, 3, 1, 1)

        # Extract features
        features_pred = self._extract_features(x_pred)
        features_target = self._extract_features(x_target)

        loss_s = torch.tensor(0.0, device=x_pred.device)  # Structure loss
        loss_t = torch.tensor(0.0, device=x_pred.device)  # Texture loss

        # Compute losses at each scale
        for feat_pred, feat_target in zip(features_pred, features_target, strict=False):
            # Structure: Gradient magnitude difference
            grad_x_pred = torch.abs(feat_pred[:, :, :, :-1] - feat_pred[:, :, :, 1:]).mean(
                dim=[2, 3, 1]
            )  # [B]
            grad_x_target = torch.abs(feat_target[:, :, :, :-1] - feat_target[:, :, :, 1:]).mean(
                dim=[2, 3, 1]
            )  # [B]

            grad_y_pred = torch.abs(feat_pred[:, :, :-1, :] - feat_pred[:, :, 1:, :]).mean(
                dim=[2, 3, 1]
            )  # [B]
            grad_y_target = torch.abs(feat_target[:, :, :-1, :] - feat_target[:, :, 1:, :]).mean(
                dim=[2, 3, 1]
            )  # [B]

            loss_s = loss_s + torch.nn.functional.l1_loss(grad_x_pred, grad_x_target)
            loss_s = loss_s + torch.nn.functional.l1_loss(grad_y_pred, grad_y_target)

            # Texture: Statistics (mean, variance)
            mean_pred = feat_pred.mean(dim=[2, 3])  # [B, C]
            mean_target = feat_target.mean(dim=[2, 3])  # [B, C]

            var_pred = feat_pred.var(dim=[2, 3])  # [B, C]
            var_target = feat_target.var(dim=[2, 3])  # [B, C]

            loss_t = loss_t + torch.nn.functional.l1_loss(mean_pred, mean_target)
            loss_t = loss_t + torch.nn.functional.l1_loss(var_pred, var_target)

        # Average over scales
        loss_s = loss_s / (2 * self.num_layers)  # 2 for x,y gradients
        loss_t = loss_t / (2 * self.num_layers)  # 2 for mean,var

        # Combined DISTS loss
        loss = (loss_s + loss_t) / 2.0

        if return_components:
            return loss, loss_s, loss_t
        else:
            return loss
