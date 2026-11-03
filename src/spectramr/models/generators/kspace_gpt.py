import torch
import torch.nn as nn

from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

__all__ = ["KSpaceGPT"]


@register_model(name="kspace_gpt", training_mode="reconstruction", spatial_dims=(2,))
class KSpaceGPT(nn.Module, IGenerator):
    """Transformer over k-space tokens: bidirectional patches, or causal lines.

    ``causal_mask=False`` tokenises ``patch_size`` patches and attends over all of them.
    ``causal_mask=True`` makes each row of the ``[..., H, W]`` plane (dim -2, the axis the
    Cartesian loader mask removes lines along) one token, in raster order, and masks
    attention so the output row ``j`` reads input rows ``0..j`` only. An unacquired row
    arrives as zeros, so its prediction comes from the rows before it. One causal pass,
    teacher-forced on the measured context; no sequential rollout.
    """

    def __init__(
        self,
        in_channels: int = 2,  # Real/Imag
        out_channels: int = 2,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        img_size: int = 256,
        patch_size: int = 16,
        features: list[int] = None,  # Compatibility with UNet-style configs
        causal_mask: bool = False,
        max_seq_length: int | None = None,
        **kwargs,  # legacy corpora (experiments/training, validated) declare unread keys
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            d_model (int): Description.
            nhead (int): Description.
            num_layers (int): Description.
            dim_feedforward (int): Description.
            img_size (int): Description.
            patch_size (int): Description.
            features (list[int]): Description.
            causal_mask (bool): Causal line tokens instead of bidirectional patches.
            max_seq_length (int | None): Longest line sequence the causal path accepts
                (its positional table); defaults to ``img_size``. Causal path only.
        """
        super().__init__()

        # [COMPATIBILITY FIX] Handle 'features' from config if d_model is default/mismatched
        if features is not None and len(features) > 0:
            # Use largest feature dimension as d_model if not explicitly set to something else
            # (Heuristic: usually the bottleneck dimension)
            if d_model == 256 and features[-1] != 256:
                d_model = features[-1]

        # [COMPATIBILITY FIX] Handle 'embed_dim' alias for 'd_model'
        if "embed_dim" in kwargs:
            d_model = kwargs["embed_dim"]

        if not isinstance(causal_mask, bool):
            raise ValueError(f"causal_mask must be a bool, got {causal_mask!r}.")
        if not causal_mask and max_seq_length is not None:
            raise ValueError(
                "max_seq_length sizes the causal line sequence and is unread with "
                "causal_mask=False; remove it or set causal_mask: true."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.causal_mask = causal_mask
        self.img_size = img_size

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        if causal_mask:
            seq_len = img_size if max_seq_length is None else max_seq_length
            if not isinstance(seq_len, int) or isinstance(seq_len, bool) or seq_len < 1:
                raise ValueError(f"max_seq_length must be a positive int, got {seq_len!r}.")
            self.max_seq_length = seq_len
            # One token per row: a per-row Linear in and out, so nothing mixes rows
            # except the masked attention.
            self.line_embed = nn.Linear(in_channels * img_size, d_model)
            self.line_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
            self.line_out = nn.Linear(d_model, out_channels * img_size)
            # True = may not attend; built once and sliced per forward.
            self.register_buffer(
                "attn_mask",
                torch.ones(seq_len, seq_len, dtype=torch.bool).triu(1),
                persistent=False,
            )
            return

        self.max_seq_length = None
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

        # Positional encoding
        num_patches = (img_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))

        # Output projection
        self.output_proj = nn.ConvTranspose2d(
            d_model, out_channels, kernel_size=patch_size, stride=patch_size
        )

        # Cache for positional embeddings
        self._cached_pos_emb = None
        self._cached_shape = None

    def _get_pos_embedding(self, seq_len: int) -> torch.Tensor:
        """Get positional embedding with caching to avoid graph breaks."""
        # Zero-cost check if shape matches
        if self._cached_shape == seq_len and self._cached_pos_emb is not None:
            return self._cached_pos_emb

        # Resize logic runs ONLY on shape change
        if seq_len <= self.pos_embedding.shape[1]:
            # Slice if smaller
            self._cached_pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            # Interpolate if larger (better for spatial coherence than repeat)
            # pos_embedding is [1, N, D] -> [1, D, N] for interpolation
            pos_emb_perm = self.pos_embedding.transpose(1, 2)

            # Interpolate to new length
            pos_emb_interp = torch.nn.functional.interpolate(
                pos_emb_perm, size=seq_len, mode="linear", align_corners=False
            )

            # Permute back to [1, N, D]
            self._cached_pos_emb = pos_emb_interp.transpose(1, 2)

        self._cached_shape = seq_len
        return self._cached_pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        """forward.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.

        forward method for KSpaceGPT.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if self.causal_mask:
            return self._forward_causal_lines(x)
        B, C, H, W = x.shape

        # Embed patches
        x = self.patch_embed(x)  # [B, d_model, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]

        # Add pos embedding (O(1) lookup most of the time)
        pos_emb = self._get_pos_embedding(x.shape[1])
        x = x + pos_emb

        # Transformer
        x = self.transformer(x)

        # Reshape back
        h, w = H // self.patch_size, W // self.patch_size
        x = x.transpose(1, 2).reshape(B, self.d_model, h, w)

        # Output projection
        x = self.output_proj(x)

        return x

    def _forward_causal_lines(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, C, H, W]`` -> same, row ``j`` of the output a function of input rows ``<= j``."""
        b, c, n_lines, width = x.shape
        if n_lines > self.max_seq_length:
            raise ValueError(
                f"{n_lines} k-space lines exceed max_seq_length={self.max_seq_length}; raise "
                "model_kwargs.max_seq_length or crop the plane."
            )
        if (c, width) != (self.in_channels, self.img_size):
            raise ValueError(
                f"causal line tokens are {self.in_channels} channels x {self.img_size} "
                f"samples (in_channels x img_size); got {c} x {width}."
            )
        tokens = x.permute(0, 2, 1, 3).reshape(b, n_lines, c * width)
        h = self.line_embed(tokens) + self.line_pos[:, :n_lines]
        h = self.transformer(h, mask=self.attn_mask[:n_lines, :n_lines], is_causal=True)
        out = self.line_out(h).reshape(b, n_lines, self.out_channels, width)
        return out.permute(0, 2, 1, 3).contiguous()

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
        return (input_shape[0], self.out_channels, input_shape[2], input_shape[3])

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x)

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "kspace_gpt"
