import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from spectramr.models.blocks.grad_checkpointing import (
    GradCheckpointingMixin,
    NestedGradCheckpointingMixin,
)
from spectramr.models.interfaces.models import IGenerator
from spectramr.models.registry import register_model

try:
    import torchkbnufft as tkbn
except ImportError:
    tkbn = None


__all__ = [
    "B0AwareGraphUNet",
    "DensityAwareGraphConv",
    "DynamicEdgeConv",
    "GraphUNetGenerator",
]


# =============================================================================
# Dynamic EdgeConv Components (DGCNN-style)
# Reference: Wang, Y., et al. (2019). "Dynamic Graph CNN for Learning on Point Clouds."
# =============================================================================


def knn_graph(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build k-NN graph from point features.

    Args:
        x: [B, C, N] Point features (or coordinates)
        k: Number of nearest neighbors

    Returns:
        idx: [B, N, k] Indices of k nearest neighbors
    """
    # [FIX] Handle complex features
    if torch.is_complex(x):
        x = torch.abs(x)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (B, N, k)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 20, idx: torch.Tensor = None) -> torch.Tensor:
    """Get edge features from k-NN graph for EdgeConv.

    Args:
        x: [B, C, N] Node features
        k: Number of neighbors
        idx: Optional pre-computed neighbor indices

    Returns:
        feature: [B, 2C, N, k] Edge features (x_j - x_i, x_i)
    """
    batch_size, num_dims, num_points = x.size()
    x = x.view(batch_size, -1, num_points)

    if idx is None:
        idx = knn_graph(x, k)

    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    # [B*N*k, C]
    feature = x.transpose(2, 1).contiguous().view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    # x: [B, N, 1, C]
    x = x.transpose(2, 1).contiguous().view(batch_size, num_points, 1, num_dims)
    # feature: [B, N, k, C*2] -> (x_j - x_i, x_i)
    feature = torch.cat((feature - x, x.repeat(1, 1, k, 1)), dim=3)

    return feature.permute(0, 3, 1, 2).contiguous()  # [B, 2C, N, k]


class DynamicEdgeConv(nn.Module):
    """Dynamic Edge Convolution layer.

    Unlike standard GNNs, this rebuilds the graph at every forward pass
    based on k-space coordinates, capturing time-varying correlations.
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            k (int): Description.
        """
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        """Forward pass with dynamic graph construction.

        Args:
            x: [B, Cin, N] Signal features
            coords: [B, 2, N] k-space coordinates (for graph construction)

        Returns:
            [B, Cout, N] Updated features

        forward method for DynamicEdgeConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            coords (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Handle complex features
        if torch.is_complex(x):
            x = x.real
        if coords is not None and torch.is_complex(coords):
            coords = coords.real

        # Construct graph based on PHYSICAL proximity (coords), not feature similarity
        if coords is not None:
            idx = knn_graph(coords, self.k)
        else:
            idx = knn_graph(x, self.k)

        x_graph = get_graph_feature(x, k=self.k, idx=idx)
        x_out = self.conv(x_graph)

        # Aggregation (Max Pooling over neighbors)
        return x_out.max(dim=-1, keepdim=False)[0]


# =============================================================================
# Graph Attention Components (GAT-style)
# =============================================================================


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer.
    """

    def __init__(self, in_features: int, out_features: int, heads: int = 1, concat: bool = True):
        """__init__.

        Args:
            in_features (int): Description.
            out_features (int): Description.
            heads (int): Description.
            concat (bool): Description.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat

        self.W = nn.Parameter(torch.Tensor(heads, in_features, out_features))
        self.a = nn.Parameter(torch.Tensor(heads, 2 * out_features, 1))

        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        adj: [B, N, N] or sparse

        forward method for GraphAttentionLayer.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, N, C = x.shape

        # [B, N, C] -> [B, N, heads, out_features]
        # First linear transform for all heads
        # We can implement this as [heads, in, out] but need efficient matmul
        # Reshape x to [B, N, 1, C]
        # x @ W^T : [B, N, 1, C] @ [heads, C, out]^T -> [B, N, heads, out]

        # Simplified implementation: Loop over heads (less efficient but clearer)
        # Or better: Concatenate inputs?

        # Fast implementation: weight specific linear transform
        # x: [B, N, C]
        # W: [H, C, F]
        # result: [B, N, H, F]

        # Using functional linear per head or einsum
        # Einsum: bnc, hcf -> bnhf
        h = torch.einsum("bnc,hcf->bnhf", x, self.W)

        # Attention mechanism
        # [B, N, N, H] attention scores
        # Can be memory intensive.
        # Efficient GAT usually works on edges.
        # Given we might have full graph or dense blocks, let's optimize for density

        # For memory efficiency with large N (~4096+), we calculate attention per edge
        if adj.is_sparse:
            # GAT on sparse adj
            # Not fully implemented here for brevity; assume dense block or limited N
            # Fallback to simple mean if sparse for now to avoid OOM
            # Real GAT requires edge index handling which is complex without PyG
            pass

        # For this experiment, we will use a simplified attention mechanism
        # Attention = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))

        # Memory-efficient implementation using broadcasting
        # h: [B, N, H, F]

        # Prepare for attention
        # Shape: [B, N, H, F]

        # We will compute attention scores ONLY for connected nodes if sparse
        # But here we implement dense attention for safety, assuming small blocks or N
        # If N is large, this will OOM. We rely on block-diagonal approximation in caller.

        # [B, N, H, F]

        # Expand for broadcasting [B, N, N, H, F] -> OOM
        # Trick: a [H, 2F, 1] -> a1 [H, F, 1], a2 [H, F, 1]
        a1 = self.a[:, : self.out_features, :]  # [H, F, 1]
        a2 = self.a[:, self.out_features :, :]  # [H, F, 1]

        # h @ a1 : [B, N, H, F] @ [H, F, 1] -> [B, N, H, 1]
        # Use einsum
        e1 = torch.einsum("bnhf,hfl->bnhl", h, a1)  # [B, N, H, 1]
        e2 = torch.einsum("bnhf,hfl->bnhl", h, a2)  # [B, N, H, 1]

        # e = e1 + e2.transpose(1, 2) roughly
        # e_ij = e1_i + e2_j
        # [B, N, 1, H] + [B, 1, N, H] -> [B, N, N, H]
        e = e1.permute(0, 1, 3, 2) + e2.permute(0, 3, 1, 2)  # [B, N, N, H]
        e = self.leakyrelu(e)

        # Mask with adjacency
        if adj is not None:
            # adj [B, N, N]
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).unsqueeze(-1)  # [1, N, N, 1]
            elif adj.dim() == 3:
                adj = adj.unsqueeze(-1)  # [B, N, N, 1]

            zero_vec = -9e15 * torch.ones_like(e)
            if adj.is_sparse:
                adj_dense = adj.to_dense()  # Warning: Dense conversion
                attention = torch.where(adj_dense > 0, e, zero_vec)
            else:
                attention = torch.where(adj > 0, e, zero_vec)
        else:
            attention = e

        attention = F.softmax(attention, dim=2)  # Softmax over neighbors (dim 2)
        # [B, N, N, H]

        # Aggregation: h' = sum(attention * h)
        # [B, N, N, H] @ [B, N, H, F] -> [B, N, H, F] (weighted sum over dimension 2)
        # Einsum: bnkh, bkhf -> bnhf (sum over k neighbors)
        h_prime = torch.einsum("bnkh,bkhf->bnhf", attention, h)

        if self.concat:
            # [B, N, H*F]
            return h_prime.reshape(B, N, self.heads * self.out_features)
        else:
            # [B, N, F] (mean)
            return h_prime.mean(dim=2)


class GraphAttentionUNet(GradCheckpointingMixin, nn.Module):
    """
    Graph Attention U-Net (GAT-UNet)
    """

    def __init__(self, in_channels, out_channels, hidden_dim=64, depth=3, heads=4):
        """__init__.

        Args:
            in_channels (Any): Description.
            out_channels (Any): Description.
            hidden_dim (Any): Description.
            depth (Any): Description.
            heads (Any): Description.
        """
        super().__init__()

        self.feature_embedder = nn.Linear(2, hidden_dim)
        self.register_buffer("coord_freqs", torch.randn(2, hidden_dim) * 2 * torch.pi)
        self.coord_embedder = nn.Linear(hidden_dim, hidden_dim)

        # Encoder
        self.encoder_layers = nn.ModuleList()
        curr_dim = hidden_dim
        skip_dims = []

        # GAT dimensions must account for heads if concatenated
        # To keep dimensions consistent, we make sure hidden_dim is divisible by heads
        # Or we project back.

        for _ in range(depth):
            # Input: curr_dim -> Output: gat_out_dim (concat heads)
            head_dim = max(curr_dim // heads, 1)
            gat_out_dim = head_dim * heads
            self.encoder_layers.append(
                GraphAttentionLayer(curr_dim, head_dim, heads=heads, concat=True)
            )
            skip_dims.append(gat_out_dim)

            # Projection to next dim
            self.encoder_layers.append(nn.Linear(gat_out_dim, curr_dim * 2))
            curr_dim *= 2

        # Middle
        head_dim = max(curr_dim // heads, 1)
        self.middle_layer = GraphAttentionLayer(curr_dim, head_dim, heads=heads, concat=True)
        curr_dim = head_dim * heads

        # Decoder
        self.decoder_layers = nn.ModuleList()

        for _ in range(depth):
            skip_dim = skip_dims.pop()

            # Reduce via linear first
            self.decoder_layers.append(nn.Linear(curr_dim + skip_dim, curr_dim))

            # GAT
            # Output: skip_dim (halving)
            out_dim = skip_dim
            head_dim_out = max(out_dim // heads, 1)

            self.decoder_layers.append(
                GraphAttentionLayer(curr_dim, head_dim_out, heads=heads, concat=True)
            )

            curr_dim = head_dim_out * heads

        self.final_layer = nn.Linear(curr_dim, out_channels)

    def _gat(self, layer: nn.Module, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """One GAT layer, checkpointed when active: its N x N attention dominates memory."""
        if self._checkpointing_active():
            return checkpoint(layer, x, adj, use_reentrant=False)
        return layer(x, adj)

    def forward(self, x, adj, dcf=None):
        # x: [B, N, 5]
        # [FIX] Handle complex input from k-space dataloaders
        """forward.

        Args:
            x (Any): Description.
            adj (Any): Description.
            dcf (Any): Description.
        Returns:
            Any: Description.

        forward method for GraphAttentionUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            dcf (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        if torch.is_complex(x):
            x = x.real

        # Entry-point shape contract.  The 2026-05-10 smoke (exp_42c_gat)
        # crashed deep in ``final_layer`` with the cryptic message
        # ``mat1 and mat2 shapes cannot be multiplied (4096x2 and 256x1)``
        # — a 2-feature tensor reaching a Linear(256, 1).  That can only
        # happen if the input arrived with too few features and the
        # downstream asserts were stripped (``-O``, ``PYTHONOPTIMIZE=1``,
        # or asserts compiled out of a wheel).  Turn the entry check
        # into an unconditional ``raise`` so the next smoke surfaces the
        # actual contract violation at this boundary rather than
        # tunneling down to a Linear's matmul error.
        if x.dim() != 3 or x.shape[-1] < 5:
            raise ValueError(
                f"GraphAttentionUNet expects [B, N, 5] graph nodes "
                f"([Re, Im, Kx, Ky, DCF]); got shape {tuple(x.shape)}. "
                f"Check that GraphUNetGenerator._grid_to_graph ran "
                f"(input was 4-D grid) or that the dataset's "
                f"`graph_encoding` transform produced 5-feature nodes."
            )

        signal = x[..., 0:2]
        coords = x[..., 2:4]
        dcf_vals = x[..., 4:5] if dcf is None else dcf.unsqueeze(-1) if dcf.dim() == 1 else dcf

        # Embedding
        pos_emb = torch.sin(coords @ self.coord_freqs)
        x_emb = self.feature_embedder(signal * dcf_vals) + self.coord_embedder(pos_emb)
        x = F.relu(x_emb)

        skips = []

        # Encoder.  Asserts → raises so an ``-O`` interpreter (or a
        # pre-compiled ``.pyc``) still surfaces the contract violation
        # instead of letting a downstream Linear matmul fail with an
        # opaque shape error.
        for i in range(0, len(self.encoder_layers), 2):
            gat = self.encoder_layers[i]
            proj = self.encoder_layers[i + 1]

            if x.shape[-1] != gat.in_features:
                raise ValueError(
                    f"Encoder GAT[{i}] expected {gat.in_features} features, got {x.shape[-1]}"
                )
            x = self._gat(gat, x, adj)
            x = F.relu(x)
            skips.append(x)

            if x.shape[-1] != proj.in_features:
                raise ValueError(
                    f"Encoder Proj[{i + 1}] expected {proj.in_features} features, got {x.shape[-1]}"
                )
            x = proj(x)
            x = F.relu(x)

        # Middle
        if x.shape[-1] != self.middle_layer.in_features:
            raise ValueError(
                f"Middle GAT expected {self.middle_layer.in_features} features, got {x.shape[-1]}"
            )
        x = self._gat(self.middle_layer, x, adj)
        x = F.relu(x)

        # Decoder
        for i in range(0, len(self.decoder_layers), 2):
            proj = self.decoder_layers[i]
            gat = self.decoder_layers[i + 1]

            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)

            if x.shape[-1] != proj.in_features:
                raise ValueError(
                    f"Decoder Proj[{i}] expected {proj.in_features} features, got {x.shape[-1]}"
                )
            x = proj(x)
            x = F.relu(x)

            if x.shape[-1] != gat.in_features:
                raise ValueError(
                    f"Decoder GAT[{i + 1}] expected {gat.in_features} features, got {x.shape[-1]}"
                )
            x = self._gat(gat, x, adj)
            x = F.relu(x)

        if x.shape[-1] != self.final_layer.in_features:
            raise ValueError(
                f"Final Linear expected {self.final_layer.in_features} "
                f"features, got {x.shape[-1]} — the encoder/decoder cascade "
                f"did not produce the expected hidden dim. Check that the "
                f"YAML's hidden_dim matches what the model was constructed "
                f"with, and that ``feature_embedder(signal)`` actually "
                f"lifted the per-node feature count from 2 → hidden_dim."
            )
        x = self.final_layer(x)
        return x


def _sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: [B] integer timestep indices.
        dim: Embedding dimensionality (must be even).

    Returns:
        emb: [B, dim] float embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, dtype=torch.float32, device=timesteps.device)
        * (torch.log(torch.tensor(10000.0)) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)  # [B, half]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]


class DynamicGraphUNet(nn.Module):
    """Graph U-Net Generator with diffusion timestep and contrast conditioning.

    Conditions every encoder and decoder layer on a sinusoidal timestep
    embedding (via additive projection) and optionally on a discrete contrast
    class label (decoder-only, via learned class embedding).
    """

    # TODO: Migrate to PyTorch Geometric for optimized message passing

    def __init__(
        self,
        in_channels: int = 5,  # [ECO-42] 5 Channels: (Re, Im, Kx, Ky, DCF)
        out_channels: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        trajectory_points: torch.Tensor | None = None,
        k_neighbors: int = 8,
        im_size: tuple[int, int] = (256, 256),
        time_embedding_dim: int = 128,
        num_classes: int = 0,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_dim (int): Description.
            depth (int): Description.
            trajectory_points (torch.Tensor | None): Description.
            k_neighbors (int): Description.
            im_size (tuple[int, int]): Description.
            time_embedding_dim (int): Description.
            num_classes (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.im_size = im_size
        self.time_embedding_dim = time_embedding_dim
        self.num_classes = num_classes

        # [ECO-42] Pre-compute topology from trajectory coordinates
        if trajectory_points is not None:
            adj = self._build_knn_graph(trajectory_points, k=k_neighbors)
            self.register_buffer("static_adjacency", adj, persistent=False)
        else:
            self.register_buffer("static_adjacency", None)

        self.k_neighbors = k_neighbors
        self.use_dynamic_graph = True

        # [ECO-42] Physics Input Embeddings
        # [STABILITY FIX] LayerNorm on raw signal BEFORE embedding.
        # K-space signal has extreme dynamic range (center ~1000x periphery).
        # Without normalization, feature_embedder amplifies this into magnitude
        # explosion that compounds through every encoder layer.
        self.signal_norm = nn.LayerNorm(2)

        # 1. Signal Embedding (Real/Imag) — operates on normalized signal
        self.feature_embedder = nn.Linear(2, hidden_dim)

        # 2. Coordinate Embedding (Fourier Features)
        self.register_buffer("coord_freqs", torch.randn(2, hidden_dim) * 2 * torch.pi)
        self.coord_embedder = nn.Linear(hidden_dim, hidden_dim)

        # --- Diffusion timestep conditioning ---
        if time_embedding_dim > 0:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embedding_dim, time_embedding_dim * 4),
                nn.SiLU(),
                nn.Linear(time_embedding_dim * 4, time_embedding_dim),
            )
        else:
            self.time_mlp = None

        # Encoder with dimension tracking for conditioning projections
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()  # [STABILITY FIX] Inter-layer normalization
        curr_dim = hidden_dim
        hidden_dim_start = hidden_dim
        enc_out_dims: list[int] = []

        for _ in range(depth):
            self.encoder_layers.append(DensityAwareGraphConv(curr_dim, hidden_dim))
            self.encoder_norms.append(nn.LayerNorm(hidden_dim))
            enc_out_dims.append(hidden_dim)
            curr_dim = hidden_dim
            hidden_dim *= 2

        self.middle_layer = DensityAwareGraphConv(curr_dim, curr_dim)
        self.middle_norm = nn.LayerNorm(curr_dim)  # [STABILITY FIX]
        mid_out_dim = curr_dim

        # Decoder dimension tracking
        enc_dims: list[int] = []
        h = hidden_dim_start
        for _ in range(depth):
            enc_dims.append(h)
            h *= 2

        dec_out_dims: list[int] = []
        for i in range(depth):
            skip_dim = enc_dims[-(i + 1)]
            if i < depth - 1:
                out_dim = enc_dims[-(i + 2)]
            else:
                out_dim = max(enc_dims[0] // 2, out_channels)
            dec_out_dims.append(out_dim)

        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()  # [STABILITY FIX]
        curr_dim_dec = mid_out_dim
        for i in range(depth):
            skip_dim = enc_dims[-(i + 1)]
            out_dim = dec_out_dims[i]
            self.decoder_layers.append(DensityAwareGraphConv(curr_dim_dec + skip_dim, out_dim))
            self.decoder_norms.append(nn.LayerNorm(out_dim))
            curr_dim_dec = out_dim

        self.final_layer = DensityAwareGraphConv(curr_dim_dec, out_channels)

        # Per-layer time-embedding projections (additive conditioning)
        if time_embedding_dim > 0:
            self.time_proj_enc = nn.ModuleList(
                [nn.Linear(time_embedding_dim, d) for d in enc_out_dims]
            )
            self.time_proj_mid = nn.Linear(time_embedding_dim, mid_out_dim)
            self.time_proj_dec = nn.ModuleList(
                [nn.Linear(time_embedding_dim, d) for d in dec_out_dims]
            )
        else:
            self.time_proj_enc = nn.ModuleList()
            self.time_proj_mid = None
            self.time_proj_dec = nn.ModuleList()

        # Contrast class conditioning (decoder-only)
        if num_classes > 0 and time_embedding_dim > 0:
            self.class_embedding = nn.Embedding(num_classes, time_embedding_dim)
            self.class_proj_dec = nn.ModuleList(
                [nn.Linear(time_embedding_dim, d) for d in dec_out_dims]
            )
        else:
            self.class_embedding = None
            self.class_proj_dec = nn.ModuleList()

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        dcf: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        class_labels: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features [B, N, 5] (Signal + Coords + DCF)
               Channels: 0:2=Signal(Re,Im), 2:4=Coords(Kx,Ky), 4=DCF
            adj: Adjacency Matrix
            dcf: Optional explicit density compensation factor [B, N]
            timesteps: Optional diffusion timestep indices [B] (long)
            class_labels: Optional contrast class indices [B] (long)

        forward method for DynamicGraphUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            dcf (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            class_labels (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Handle complex input from k-space dataloaders
        if torch.is_complex(x):
            x = x.real

        signal = x[..., 0:2]  # Real, Imag
        coords = x[..., 2:4]  # Kx, Ky
        dcf = x[..., 4]  # Density (B, N)

        # [STABILITY FIX] DCF Pre-weighting with clamping + mean-normalization
        # Non-Cartesian DCF values span 4+ orders of magnitude (1e-4 to 1e+3),
        # causing feature explosion in sparse k-space regions.
        dcf_weight = dcf.unsqueeze(-1)  # [B, N, 1]
        dcf_weight = dcf_weight.clamp(min=1e-6, max=1e3)
        dcf_weight = dcf_weight / (dcf_weight.mean(dim=1, keepdim=True) + 1e-8)

        # [STABILITY FIX] Normalize signal BEFORE embedding.
        # K-space center has 100-1000x larger values than periphery.
        # Without normalization, feature_embedder amplifies this into
        # exponential magnitude growth through the encoder stack.
        signal_normed = self.signal_norm(signal)
        signal_weighted = signal_normed * dcf_weight

        # Store raw signal for residual output connection
        signal_residual = signal

        # [ECO-42] Coordinate Embedding (with output clamping)
        pos_emb = torch.sin(coords @ self.coord_freqs)
        pos_emb = torch.nan_to_num(pos_emb, nan=0.0, posinf=1.0, neginf=-1.0)

        # Fuse Embeddings
        x_emb = self.feature_embedder(signal_weighted) + self.coord_embedder(pos_emb)
        x = F.relu(x_emb)

        # --- Diffusion timestep embedding ---
        t_emb = None
        if self.time_mlp is not None and timesteps is not None:
            sin_emb = _sinusoidal_timestep_embedding(timesteps, self.time_embedding_dim)
            t_emb = self.time_mlp(sin_emb)  # [B, time_embedding_dim]

        # --- Contrast class embedding ---
        c_emb = None
        if self.class_embedding is not None and class_labels is not None:
            c_emb = self.class_embedding(class_labels)  # [B, time_embedding_dim]

        skips = []

        # Encoder — inject time embedding at each layer
        # [STABILITY FIX] LayerNorm after each conv to prevent magnitude growth
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x, adj, dcf)
            x = self.encoder_norms[i](x)
            if t_emb is not None:
                x = x + self.time_proj_enc[i](t_emb).unsqueeze(1)  # [B, 1, C]
            skips.append(x)

        # Bottleneck — inject time embedding
        x = self.middle_layer(x, adj, dcf)
        x = self.middle_norm(x)
        if t_emb is not None and self.time_proj_mid is not None:
            x = x + self.time_proj_mid(t_emb).unsqueeze(1)  # [B, 1, C]

        # Decoder — inject time + contrast embeddings
        for i, layer in enumerate(self.decoder_layers):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=-1)
            x = layer(x, adj, dcf)
            x = self.decoder_norms[i](x)
            if t_emb is not None:
                x = x + self.time_proj_dec[i](t_emb).unsqueeze(1)  # [B, 1, C]
            if c_emb is not None:
                x = x + self.class_proj_dec[i](c_emb).unsqueeze(1)  # [B, 1, C]

        x = self.final_layer(x, adj, dcf)

        # [STABILITY FIX] Residual connection from input signal to output.
        # This provides a direct gradient path and anchors the output
        # magnitude to the input scale, preventing drift during training.
        # Only applies when output dim matches signal dim (Re/Im = 2).
        if x.shape[-1] == signal_residual.shape[-1]:
            x = x + signal_residual

        # [STABILITY FIX] Clamp final output to prevent NaN/Inf propagation
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        return x


    # Internal _build_knn_graph logic duplicated or inherited?
    # Since we are moving class, we should copy the method or make it a mixin.
    # For simplicity, we assume the wrapper handles graph construction or we copy logic.
    # Actually, DynamicGraphUNet implementation above removed the `_build_knn_graph` method.
    # We must restore it or delegate.

    def _build_knn_graph(
        self, points: torch.Tensor, k: int, block_size: int = 1024
    ) -> torch.Tensor:
        # Same implementation as before
        """_build_knn_graph.

        Args:
            points (torch.Tensor): Description.
            k (int): Description.
            block_size (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        N, D = points.shape
        device = points.device

        # [FIX] Ensure points are real for torch.cdist
        if torch.is_complex(points):
            points = points.real

        if block_size >= N:
            dists = torch.cdist(points, points)
            _, indices = torch.topk(-dists, min(k + 1, N), dim=1)
            row = torch.arange(N, device=device).view(-1, 1).repeat(1, indices.shape[1]).flatten()
            col = indices.flatten()
            edge_index = torch.stack([row, col], dim=0)
            values = torch.ones(edge_index.shape[1], device=device)
            return torch.sparse_coo_tensor(edge_index, values, (N, N))

        num_blocks = (N + block_size - 1) // block_size
        all_rows, all_cols = [], []
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, N)
            block_n = end - start
            if block_n <= 0:
                continue

            block_points = points[start:end]
            dists = torch.cdist(block_points, block_points)
            local_k = min(k + 1, block_n)
            _, local_indices = torch.topk(-dists, local_k, dim=1)
            global_indices = local_indices + start
            local_rows = torch.arange(block_n, device=device).view(-1, 1).repeat(1, local_k) + start
            all_rows.append(local_rows.flatten())
            all_cols.append(global_indices.flatten())

        row = torch.cat(all_rows)
        col = torch.cat(all_cols)
        edge_index = torch.stack([row, col], dim=0)
        values = torch.ones(edge_index.shape[1], device=device)
        return torch.sparse_coo_tensor(edge_index, values, (N, N))


@register_model(
    name="graph_unet",
    training_mode="reconstruction",
    spatial_dims=(2,),
    # Despite "graph" in the name, this generator's forward consumes
    # standard [B, C, H, W] image-domain tensors and uses graph
    # adjacency only as an internal connectivity prior.
    input_domain="image",
    output_domain="image",
    # ``_grid_to_graph`` (~line 1076) handles ``torch.is_complex(grid)``
    # by splitting into real/imag and concatenating along the channel
    # dim, so complex input IS supported. The decorator previously
    # claimed False, causing the audit to reject pre_model iFFT
    # chains that produce a complex_image tensor. Fixed 2026-05-12.
    accepts_complex=True,
    requires_paired_data=True,
)
@register_model(
    name="graph_unet_diffusion",
    training_mode="diffusion",
    spatial_dims=(2,),
    input_domain="image",
    output_domain="image",
    accepts_complex=True,
    requires_paired_data=True,
)
class GraphUNetGenerator(NestedGradCheckpointingMixin, nn.Module, IGenerator):
    """Wrapper for Graph U-Net Generators (Dynamic Graph UNet, GAT-UNet).

    Supports diffusion timestep conditioning and contrast class conditioning
    when backbone_type="dynamic_graph_unet" (the default).
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        trajectory_points: torch.Tensor | None = None,
        k_neighbors: int = 8,
        im_size: tuple[int, int] = (256, 256),
        backbone_type: str = "dynamic_graph_unet",
        time_embedding_dim: int = 128,
        num_classes: int = 0,
        gat_max_nodes: int = 4096,
        **kwargs,
    ):
        """__init__.

        Args:
            in_channels (int): Description.
            out_channels (int): Description.
            hidden_dim (int): Description.
            depth (int): Description.
            trajectory_points (torch.Tensor | None): Description.
            k_neighbors (int): Description.
            im_size (tuple[int, int]): Description.
            backbone_type (str): Description.
            time_embedding_dim (int): Description.
            num_classes (int): Description.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.im_size = im_size
        self.backbone_type = backbone_type
        # GAT path: dense softmax over [B, N, N, H] is O(N^2) memory.
        # Cap node count to keep memory feasible; full-res output is
        # recovered by bilinear upsampling after the backbone.
        self.gat_max_nodes = int(gat_max_nodes)

        # [ECO-42] Optional Adjoint for Visualization/Validation
        if tkbn:
            self.adjoint_op = tkbn.KbNufftAdjoint(im_size=im_size)
        else:
            self.adjoint_op = None

        # Select Backbone
        if backbone_type == "dynamic_graph_unet" or backbone_type == "default":
            self.backbone = DynamicGraphUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=hidden_dim,
                depth=depth,
                trajectory_points=trajectory_points,
                k_neighbors=k_neighbors,
                im_size=im_size,
                time_embedding_dim=time_embedding_dim,
                num_classes=num_classes,
                **kwargs,
            )
        elif backbone_type == "gat_unet":
            # GAT specific params
            heads = kwargs.get("heads", 4)
            self.backbone = GraphAttentionUNet(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_dim=hidden_dim,
                depth=depth,
                heads=heads,
            )
        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # Graph Construction Helpers (Delegated to Backbone if possible, or managed here)
        # We manage graph construction here to share between backbones
        self.k_neighbors = k_neighbors
        self.use_dynamic_graph = True

        if trajectory_points is not None:
            # Use DynamicGraphUNet's helper or move it to util?
            # For now we use the method attached to the wrapper for shared utility
            adj = self._build_knn_graph(trajectory_points, k=k_neighbors)
            self.register_buffer("static_adjacency", adj, persistent=False)
        else:
            self.register_buffer("static_adjacency", None)

    def dynamic_knn_graph(self, trajectory: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        [FIX 11] Build k-NN graph dynamically from ACTUAL trajectory coordinates.
        """
        if trajectory.dim() == 3:
            trajectory = trajectory[0]

        # [FIX] Ensure trajectory is real for torch.cdist
        if torch.is_complex(trajectory):
            trajectory = trajectory.real

        N = trajectory.shape[0]
        device = trajectory.device

        if N > 4096:
            return self._build_knn_graph(trajectory, k=k)

        dist = torch.cdist(trajectory, trajectory)
        _, indices = torch.topk(-dist, min(k + 1, N), dim=1)
        row = torch.arange(N, device=device).view(-1, 1).expand(-1, indices.shape[1]).flatten()
        col = indices.flatten()
        edge_index = torch.stack([row, col], dim=0)
        values = torch.ones(edge_index.shape[1], device=device)
        return torch.sparse_coo_tensor(edge_index, values, (N, N))

    def _build_knn_graph(
        self, points: torch.Tensor, k: int, block_size: int = 1024
    ) -> torch.Tensor:
        # Duplicated helper for wrapper context
        """_build_knn_graph.

        Args:
            points (torch.Tensor): Description.
            k (int): Description.
            block_size (int): Description.
        Returns:
            torch.Tensor: Description.
        """
        N, D = points.shape
        device = points.device

        # [FIX] Ensure points are real for torch.cdist
        if torch.is_complex(points):
            points = points.real

        if block_size >= N:
            dists = torch.cdist(points, points)
            _, indices = torch.topk(-dists, min(k + 1, N), dim=1)
            row = torch.arange(N, device=device).view(-1, 1).repeat(1, indices.shape[1]).flatten()
            col = indices.flatten()
            edge_index = torch.stack([row, col], dim=0)
            values = torch.ones(edge_index.shape[1], device=device)
            return torch.sparse_coo_tensor(edge_index, values, (N, N))

        num_blocks = (N + block_size - 1) // block_size
        all_rows, all_cols = [], []
        for i in range(num_blocks):
            start = i * block_size
            end = min((i + 1) * block_size, N)
            block_n = end - start
            if block_n <= 0:
                continue

            block_points = points[start:end]
            dists = torch.cdist(block_points, block_points)
            local_k = min(k + 1, block_n)
            _, local_indices = torch.topk(-dists, local_k, dim=1)
            global_indices = local_indices + start
            local_rows = torch.arange(block_n, device=device).view(-1, 1).repeat(1, local_k) + start
            all_rows.append(local_rows.flatten())
            all_cols.append(global_indices.flatten())

        row = torch.cat(all_rows)
        col = torch.cat(all_cols)
        edge_index = torch.stack([row, col], dim=0)
        values = torch.ones(edge_index.shape[1], device=device)
        return torch.sparse_coo_tensor(edge_index, values, (N, N))

    def _grid_to_graph(self, grid: torch.Tensor) -> torch.Tensor:
        """Convert grid image to graph nodes."""
        if grid.dim() != 4:
            raise ValueError(f"Expected 4D grid input [B, C, H, W], got {grid.shape}")

        # [FIX] Handle complex grid
        if torch.is_complex(grid):
            # Split into real and imaginary
            real_part = grid.real
            imag_part = grid.imag
            grid = torch.cat([real_part, imag_part], dim=1)

        B, C, H, W = grid.shape
        N = H * W
        device = grid.device
        signal = grid.permute(0, 2, 3, 1).reshape(B, N, C)
        if C == 1:
            signal = torch.cat([signal, torch.zeros_like(signal)], dim=-1)
        elif C > 2:
            signal = signal[..., :2]
        kx = torch.linspace(-1, 1, W, device=device)
        ky = torch.linspace(-1, 1, H, device=device)
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        coords = torch.stack([grid_x, grid_y], dim=-1).reshape(1, N, 2)
        coords = coords.repeat(B, 1, 1)
        dcf = torch.ones(B, N, 1, device=device)
        nodes = torch.cat([signal, coords, dcf], dim=-1)
        return nodes

    def _graph_to_grid(self, nodes: torch.Tensor, original_shape: tuple) -> torch.Tensor:
        """Convert graph nodes back to grid."""
        B, C_in, H, W = original_shape
        if nodes.dim() != 3:
            return nodes
        OutC = nodes.shape[-1]
        grid = nodes.reshape(B, H, W, OutC).permute(0, 3, 1, 2)
        return grid

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        class_labels: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass with optional diffusion timestep and contrast conditioning.

        Args:
            x: Input tensor — either graph nodes [B, N, 5] or grid [B, C, H, W].
            adj: Pre-built adjacency matrix (optional; built dynamically if None).
            timesteps: Diffusion timestep indices [B] (long); passed to backbone.
            class_labels: Contrast class indices [B] (long); passed to backbone.

        forward method for GraphUNetGenerator.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            timesteps (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            class_labels (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # [FIX] Handle complex input early
        if torch.is_complex(x):
            if x.dim() == 4:
                # Grid input - will be handled in _grid_to_graph
                pass
            else:
                # Graph input - assume it's already in the right format but promoted to complex?
                # Or maybe it needs splitting? For now, just take real.
                x = x.real

        # Pre-process inputs (Grid -> Graph)
        is_grid_input = x.dim() == 4
        original_shape = x.shape if is_grid_input else None

        # GAT backbone uses dense [B, N, N, H] attention. Cap N before encoding
        # by adaptively pooling the grid; we'll upsample the output back at the end.
        gat_downsampled = False
        gat_orig_hw: tuple[int, int] | None = None
        if (
            is_grid_input
            and isinstance(self.backbone, GraphAttentionUNet)
            and (x.shape[-2] * x.shape[-1]) > self.gat_max_nodes
        ):
            B, _, H, W = x.shape
            gat_orig_hw = (H, W)
            target_side = max(1, int(self.gat_max_nodes**0.5))
            tgt_h = min(H, target_side)
            tgt_w = min(W, target_side)
            if torch.is_complex(x):
                x_real = F.adaptive_avg_pool2d(x.real, (tgt_h, tgt_w))
                x_imag = F.adaptive_avg_pool2d(x.imag, (tgt_h, tgt_w))
                x = torch.complex(x_real, x_imag)
            else:
                x = F.adaptive_avg_pool2d(x, (tgt_h, tgt_w))
            original_shape = x.shape  # graph_to_grid uses the pooled shape
            gat_downsampled = True

        if is_grid_input:
            x = self._grid_to_graph(x)

        # Graph Construction (Shared)
        if adj is None:
            coords = x[0, :, 2:4].detach()
            if getattr(self, "use_dynamic_graph", True):
                adj = self.dynamic_knn_graph(coords, k=self.k_neighbors)
                self.register_buffer("cached_coords", coords.contiguous(), persistent=False)
            elif self.static_adjacency is not None:
                adj = self.static_adjacency
            else:
                adj = self._build_knn_graph(coords, k=self.k_neighbors)
                if adj is not None and adj.dim() >= 2:
                    self.register_buffer("static_adjacency", adj, persistent=False)
                    self.register_buffer("cached_coords", coords.contiguous(), persistent=False)

        # Forward through backbone, passing conditioning signals if supported
        if isinstance(self.backbone, DynamicGraphUNet):
            x = self.backbone(
                x,
                adj,
                timesteps=timesteps,
                class_labels=class_labels,
            )
        elif isinstance(self.backbone, GraphAttentionUNet):
            x = self.backbone(x, adj)
        else:
            x = self.backbone(x)

        # Post-process (Graph -> Grid)
        if is_grid_input and original_shape is not None:
            x = self._graph_to_grid(x, original_shape)

        # Restore original H,W if we down-pooled for the GAT path.
        if gat_downsampled and gat_orig_hw is not None and x.dim() == 4:
            x = F.interpolate(x, size=gat_orig_hw, mode="bilinear", align_corners=False)

        return x

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
        return input_shape

    def generate(self, x: torch.Tensor, adj: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """generate.

        Args:
            x (torch.Tensor): Description.
            adj (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        return self.forward(x, adj, **kwargs)

    @torch.no_grad()
    def predict_image(self, x_nodes: torch.Tensor) -> torch.Tensor:
        # Shared implementation for visualization
        """predict_image.

        Args:
            x_nodes (torch.Tensor): Description.
        Returns:
            torch.Tensor: Description.
        """
        if self.adjoint_op is None:
            return torch.zeros(x_nodes.shape[0], 1, *self.im_size, device=x_nodes.device)

        if not hasattr(self, "cached_coords") or self.cached_coords is None:
            return torch.zeros(x_nodes.shape[0], 1, *self.im_size, device=x_nodes.device)

        # [FIX] Handle Image Input
        if x_nodes.ndim == 4:
            if x_nodes.shape[1] == 2:
                return torch.complex(x_nodes[:, 0:1], x_nodes[:, 1:2]).abs()
            return x_nodes.abs()

        if x_nodes.dim() == 3 and x_nodes.shape[-1] == 2:
            kspace = torch.complex(x_nodes[..., 0], x_nodes[..., 1])
            kspace = kspace.unsqueeze(1)
        else:
            return torch.zeros(x_nodes.shape[0], 1, *self.im_size, device=x_nodes.device)

        traj = self.cached_coords.transpose(0, 1) * torch.pi
        image = self.adjoint_op(kspace, traj)
        return image.abs()

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "graph_unet"


class DensityAwareGraphConv(nn.Module):
    """
    [ECO-42] Density-Aware Graph Convolution.

    Implements a Monte Carlo approximation of the continuous convolution integral
    by weighting message passing with the inverse sampling density (DCF).
    """

    def __init__(self, in_features, out_features):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj, dcf=None):
        """
        x: [B, N, C] Node features
        adj: [B, N, N] or [N, N] Adjacency Matrix
        dcf: [B, N] or [N] Density Compensation Weights

        forward method for DensityAwareGraphConv.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            adj (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.
            dcf (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        # 1. Apply Density Compensation to Inputs
        # Nodes in dense regions are downweighted to normalize energy density
        if dcf is not None:
            # dcf shape handling
            if dcf.dim() == 1:
                w = dcf.view(1, -1, 1)  # [1, N, 1]
            elif dcf.dim() == 2:
                w = dcf.unsqueeze(-1)  # [B, N, 1]
            else:
                w = dcf  # Assume compatible

            # [STABILITY FIX] Clamp DCF to bounded range and mean-normalize
            # Non-Cartesian DCF values routinely span 4+ orders of magnitude
            # (1e-4 to 1e+3). Without normalization, this causes feature
            # explosion in sparse k-space regions and starvation in dense regions.
            w = w.clamp(min=1e-6, max=1e3)
            w = w / (w.mean(dim=1, keepdim=True) + 1e-8)

            x_weighted = x * w
        else:
            x_weighted = x

        # 2. Sparse Aggregation
        # Result = Sum(Neighbor_Features * Neighbor_Density_Weight)

        # [FIX] Handle malformed adj (0D, 1D, or None)
        # This can happen if graph construction fails or returns scalar
        if adj is None or adj.dim() < 2:
            # Build simple identity adjacency as fallback
            # x_weighted is [B, N, C] at this point
            N = x_weighted.size(1) if x_weighted.dim() >= 2 else x_weighted.size(0)
            B = x_weighted.size(0) if x_weighted.dim() >= 2 else 1
            device = x_weighted.device
            # Identity matrix means each node only receives its own features
            adj = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)

        # Handle batching for sparse mm (optimized block-diagonal approach)
        if adj.is_sparse or (hasattr(adj, "layout") and adj.layout == torch.sparse_coo):
            if adj.dim() == 2:
                # Shared adj - use JIT compiled aggregate
                agg = self._fast_sparse_aggregate(x_weighted, adj)
            else:
                # Batched sparse using block-diagonal construction for efficiency
                # Instead of looping, we construct a block-diagonal sparse matrix
                B = x_weighted.size(0)
                N = adj.size(-1)

                if B <= 4:
                    # For small batches, vectorized loop is faster
                    outs = [torch.sparse.mm(adj[i], x_weighted[i]) for i in range(B)]
                    agg = torch.stack(outs, dim=0)
                else:
                    # Construct block-diagonal sparse matrix for large batches
                    # This allows single sparse mm call
                    indices_list = []
                    values_list = []

                    for i in range(B):
                        sparse_adj = adj[i].coalesce() if adj[i].is_sparse else adj[i].to_sparse()
                        idx = sparse_adj.indices()
                        # Offset indices for block-diagonal structure
                        idx = idx + i * N
                        indices_list.append(idx)
                        values_list.append(sparse_adj.values())

                    # Combine into single block-diagonal sparse matrix
                    all_indices = torch.cat(indices_list, dim=1)
                    all_values = torch.cat(values_list, dim=0)
                    block_diag = torch.sparse_coo_tensor(
                        all_indices,
                        all_values,
                        (B * N, B * N),
                        device=x_weighted.device,
                    )

                    # Flatten x, apply sparse mm, reshape back
                    x_flat = x_weighted.reshape(B * N, -1)
                    agg_flat = torch.sparse.mm(block_diag, x_flat)
                    agg = agg_flat.reshape(B, N, -1)
        else:
            # Dense fallback - ensure both tensors are 3D for bmm
            # Expected: adj [B, N, N], x_weighted [B, N, C]

            # Get batch size from x (always use x as the source of truth for B)
            if x_weighted.dim() == 3 or x_weighted.dim() == 4:
                B = x_weighted.size(0)
            elif x_weighted.dim() == 2:
                B = 1
            else:
                raise ValueError(f"Unexpected x_weighted dims: {x_weighted.dim()}")

            # Handle x_weighted shape
            if x_weighted.dim() == 4:
                # [B, C, H, W] -> flatten to [B, H*W, C]
                B_x, C_x, H_x, W_x = x_weighted.shape
                x_weighted = x_weighted.permute(0, 2, 3, 1).reshape(B_x, H_x * W_x, C_x)
            elif x_weighted.dim() == 2:
                # [N, C] -> [1, N, C] and expand
                x_weighted = x_weighted.unsqueeze(0)
                if B > 1:
                    x_weighted = x_weighted.expand(B, -1, -1)
            # If dim == 3, already correct shape [B, N, C]

            # Handle adj shape - expand 2D to 3D if needed
            if adj.dim() == 2:
                adj = adj.unsqueeze(0).expand(B, -1, -1)  # [N,N] -> [B,N,N]
            elif adj.dim() == 1:
                raise ValueError(f"adj must be at least 2D, got shape {adj.shape}")

            # Validate shapes before bmm
            # adj: [B, N, N], x_weighted: [B, N, C]
            # bmm expects (B, n, m) @ (B, m, p) -> (B, n, p)
            if adj.shape[0] != x_weighted.shape[0]:
                # Batch mismatch - try to expand adj
                if adj.shape[0] == 1:
                    adj = adj.expand(x_weighted.shape[0], -1, -1)
                else:
                    raise ValueError(
                        f"Batch size mismatch: adj {adj.shape} vs x_weighted {x_weighted.shape}"
                    )

            if adj.shape[2] != x_weighted.shape[1]:
                raise ValueError(
                    f"Inner dimension mismatch for bmm: adj {adj.shape} vs x_weighted {x_weighted.shape}. "
                    f"adj.shape[2]={adj.shape[2]} must equal x_weighted.shape[1]={x_weighted.shape[1]}"
                )

            # [STABILITY FIX] Symmetric degree normalization (D^{-1/2} A D^{-1/2})
            # Without this, high-degree nodes accumulate unbounded energy from
            # neighbors during message passing. This is standard GCN practice
            # (Kipf & Welling, 2017).
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)  # [B, N, 1]
            deg_inv_sqrt = deg.pow(-0.5)  # [B, N, 1]
            # D^{-1/2} A D^{-1/2}: scale rows and columns by degree
            adj_norm = adj * deg_inv_sqrt * deg_inv_sqrt.transpose(-1, -2)

            agg = torch.bmm(adj_norm, x_weighted)

        # 3. Update
        out = self.linear(agg)

        # [ECO-42] No residual connection in Conv block itself?
        # Standard GCN usually is just transformation.
        out = F.relu(out)

        # [STABILITY FIX] NaN/Inf safety net on output
        # This is a last-resort guard — the DCF clamping and degree
        # normalization above should prevent NaN in practice.
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out

    @torch.jit.export
    def _fast_sparse_aggregate(self, x_weighted: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        JIT-compiled sparse aggregation loop to reduce Python overhead.
        adj: [N, N] sparse tensor
        x_weighted: [B, N, C] dense tensor
        """
        # Note: torch.sparse.mm expects 2D tensors.
        outs = torch.jit.annotate(list[torch.Tensor], [])
        for i in range(x_weighted.size(0)):
            # adj [N, N] @ x [N, C] -> [N, C]
            outs.append(torch.sparse.mm(adj, x_weighted[i]))
        return torch.stack(outs)


# =============================================================================
# B0-Aware Graph UNet (Time-Informed for Off-Resonance Correction)
# =============================================================================


class B0AwareGraphUNet(nn.Module, IGenerator):
    """B0-Aware Graph U-Net for Non-Cartesian MRI with Off-Resonance Correction.

    Physics Motivation:
    For Spiral/Radial trajectories, B0 inhomogeneity causes phase accumulation
    φ(r, t) = ΔΩ(r) * t that blurs the reconstructed image. Traditional NUFFT
    handles this via time-segmented gridding. In graph-based approaches, we
    inject the readout time as a node feature, enabling the network to learn
    the conjugate-phase demodulation implicitly.

    Input Features: [Re, Im, Kx, Ky, DCF, t_readout]
    - Re, Im: Complex k-space signal
    - Kx, Ky: Trajectory coordinates in [-π, π]
    - DCF: Density compensation factor
    - t_readout: Readout time (normalized 0-1)

    The network learns: f(signal, coords, t) -> corrected_signal

    Reference:
    - Noll DC et al., IEEE TMI 1991 (Conjugate Phase Reconstruction)
    - Wang Y et al., "Dynamic Graph CNN for Point Clouds", DGCNN 2019
    """

    def __init__(
        self,
        in_channels: int = 2,  # Signal channels (Re, Im)
        out_channels: int = 2,
        hidden_dim: int = 64,
        depth: int = 3,
        k_neighbors: int = 16,
        im_size: tuple[int, int] = (256, 256),
        use_residual: bool = True,
    ):
        """Initialize B0-Aware Graph UNet.

        Args:
            in_channels: Input signal channels (default 2 for Re/Im)
            out_channels: Output signal channels
            hidden_dim: Hidden feature dimension
            depth: Number of encoder/decoder layers
            k_neighbors: K for k-NN graph construction
            im_size: Image size for optional NUFFT adjoint
            use_residual: Add residual connection from input to output
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k_neighbors
        self.use_residual = use_residual
        self.im_size = im_size

        # Feature dimensions:
        # Input = Signal(2) + Time(1) = 3 channels for base embedding
        # Coordinates are used for graph construction, not concatenated

        # Input embedding: [Re, Im, t] -> hidden_dim
        self.input_embed = nn.Sequential(
            nn.Conv1d(in_channels + 1, hidden_dim, 1),  # +1 for time
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Encoder with DynamicEdgeConv
        self.encoder = nn.ModuleList()
        curr_dim = hidden_dim
        for i in range(depth):
            out_dim = hidden_dim * (2 ** min(i + 1, 3))
            self.encoder.append(DynamicEdgeConv(curr_dim, out_dim, k=k_neighbors))
            curr_dim = out_dim

        # Bottleneck
        self.bottleneck = DynamicEdgeConv(curr_dim, curr_dim, k=k_neighbors)

        # Decoder with skip connections
        self.decoder = nn.ModuleList()
        enc_dims = [hidden_dim * (2 ** min(i + 1, 3)) for i in range(depth)]

        for i in range(depth):
            skip_dim = enc_dims[-(i + 1)]
            if i < depth - 1:
                out_dim = enc_dims[-(i + 2)]
            else:
                out_dim = hidden_dim
            # Input is curr_dim + skip_dim due to skip connection
            self.decoder.append(DynamicEdgeConv(curr_dim + skip_dim, out_dim, k=k_neighbors))
            curr_dim = out_dim

        # Final projection: hidden_dim -> out_channels
        self.output_proj = nn.Conv1d(hidden_dim, out_channels, 1)

        # Optional NUFFT adjoint for image-domain output
        if tkbn:
            self.adjoint_op = tkbn.KbNufftAdjoint(im_size=im_size)
        else:
            self.adjoint_op = None

    def forward(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor,
        t_readout: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with time-aware B0 correction.

        Args:
            x: K-space signal [B, 2, N] (Re/Im channels)
            trajectory: K-space coordinates [B, 2, N] or [2, N] (kx, ky in [-π, π])
            t_readout: Readout time [B, 1, N] or [N] (normalized 0-1)
                       If None, computed from sample index

        Returns:
            Corrected k-space signal [B, 2, N]

        forward method for B0AwareGraphUNet.

        Executes PyTorch tensor operations.

        Args:
            x (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            trajectory (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.
            t_readout (torch.Tensor, shape (B, C, H, W) or (B, C, D, H, W)): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        B, C, N = x.shape
        device = x.device

        # Ensure trajectory has batch dimension
        if trajectory.dim() == 2:
            trajectory = trajectory.unsqueeze(0).expand(B, -1, -1)

        # Generate readout time if not provided
        if t_readout is None:
            # Linear time from 0 to 1 (typical for single-shot spiral/radial)
            t_readout = torch.linspace(0, 1, N, device=device)
            t_readout = t_readout.view(1, 1, N).expand(B, -1, -1)
        elif t_readout.dim() == 1:
            t_readout = t_readout.view(1, 1, N).expand(B, -1, -1)
        elif t_readout.dim() == 2:
            t_readout = t_readout.unsqueeze(1)

        # Concatenate signal with time: [B, 3, N]
        x_with_time = torch.cat([x, t_readout], dim=1)

        # Input embedding
        feat = self.input_embed(x_with_time)  # [B, hidden_dim, N]

        # Encoder with skip connections
        skips = []
        for layer in self.encoder:
            feat = layer(feat, coords=trajectory)
            skips.append(feat)

        # Bottleneck
        feat = self.bottleneck(feat, coords=trajectory)

        # Decoder with skip connections
        for i, layer in enumerate(self.decoder):
            skip = skips.pop()
            feat = torch.cat([feat, skip], dim=1)
            feat = layer(feat, coords=trajectory)

        # Output projection
        out = self.output_proj(feat)  # [B, out_channels, N]

        # Residual connection (signal correction)
        if self.use_residual:
            out = out + x

        return out

    def predict_image(
        self,
        x: torch.Tensor,
        trajectory: torch.Tensor,
        dcf: torch.Tensor = None,
        t_readout: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run forward pass and grid to image domain.

        Args:
            x: K-space signal [B, 2, N]
            trajectory: Coordinates [B, 2, N] in [-π, π]
            dcf: Density compensation [B, N] or [N]
            t_readout: Readout time [B, 1, N]

        Returns:
            Image [B, 1, H, W]
        """
        # Correct k-space
        corrected = self.forward(x, trajectory, t_readout)

        if self.adjoint_op is None:
            raise RuntimeError("torchkbnufft required for image prediction")

        # Convert to complex
        kspace_complex = torch.complex(corrected[:, 0], corrected[:, 1])
        kspace_complex = kspace_complex.unsqueeze(1)  # [B, 1, N]

        # Apply DCF if provided
        if dcf is not None:
            if dcf.dim() == 1:
                dcf = dcf.unsqueeze(0)
            kspace_complex = kspace_complex * dcf.unsqueeze(1)

        # Grid to image via NUFFT adjoint
        # trajectory for torchkbnufft: [2, N]
        traj = trajectory[0] if trajectory.dim() == 3 else trajectory
        img = self.adjoint_op(kspace_complex, traj)

        return img.abs()

    # IGenerator interface
    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Generate output (interface compliance)."""
        # Assume x is [B, 5, N] with [Re, Im, Kx, Ky, DCF]
        signal = x[:, :2]  # [B, 2, N]
        trajectory = x[:, 2:4]  # [B, 2, N]
        return self.forward(signal, trajectory)

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
        return input_shape  # Same shape as input

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "b0_aware_graph_unet"
