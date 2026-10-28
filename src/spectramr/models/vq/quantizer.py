"""Vector Quantized VAE Quantizer with EMA Codebook

This module implements the vector quantization component for VQ-VAE,
including EMA-based codebook updates for stable training.
"""

import torch
import torch.nn.functional as F
from torch import nn


class VQQuantizer(nn.Module):
    """Vector Quantizer with Exponential Moving Average (EMA) codebook updates.

    This implementation follows the VQ-VAE paper with EMA updates for
    stable codebook learning and commitment loss for reconstruction.

    Reference:
    van den Oord et al. "Neural Discrete Representation Learning"
    NeurIPS 2017
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 64,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        """Initialize VQ quantizer.

        Args:
            num_embeddings: Number of codebook entries
            embedding_dim: Dimension of each embedding vector
            commitment_cost: Weight for commitment loss
            decay: EMA decay rate for codebook updates
            epsilon: Small value for numerical stability

        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Initialize codebook
        self.register_buffer("embeddings", torch.randn(num_embeddings, embedding_dim))
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embeddings.clone())

        # Initialize with uniform distribution
        limit = 1 / self.num_embeddings
        self.embeddings.data.uniform_(-limit, limit)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through quantizer.

        Args:
            z: Input tensor of shape (B, C, H, W) or (B, N, C)

        Returns:
            Tuple of (quantized, loss, indices)
            - quantized: Quantized tensor same shape as input
            - loss: Total VQ loss
            - indices: Encoding indices

        """
        # Flatten input for quantization
        z_flattened = z.view(-1, self.embedding_dim)

        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2 z.e: an (N, K) matmul. The broadcast
        # difference ``(z - e)**2`` materialises (N, K, D) -- 32 GiB for the 64x64
        # latent, K=1024, D=256 of experiment_54, which OOMs a 16 GiB card. Autocast is
        # off because a half-precision matmul would flip the argmin on near-ties.
        with torch.autocast(device_type=z.device.type, enabled=False):
            zf = z_flattened.float()
            emb = self.embeddings.float()
            distances = (
                zf.pow(2).sum(1, keepdim=True) + emb.pow(2).sum(1).unsqueeze(0) - 2 * zf @ emb.t()
            )

        # Find closest embeddings
        encoding_indices = torch.argmin(distances, dim=1)

        # Quantize
        quantized = F.embedding(encoding_indices, self.embeddings)

        # Reshape back to original shape
        quantized = quantized.view(z.shape)

        # Commitment loss: ``||z_e - sg(e)||^2`` -- the encoder's pull toward
        # the codebook entry it selected.
        commitment_loss = F.mse_loss(z, quantized.detach())

        # There is deliberately no codebook term beside it. This is an *EMA*
        # quantizer: ``embeddings`` is a buffer moved by ``_update_codebook``,
        # which is van den Oord Appendix A.1's replacement for the codebook
        # term of Eq. (3). No gradient can reach a buffer, so a codebook term
        # here would be a constant.
        #
        # What used to sit here was ``F.mse_loss(quantized.detach(), z)``
        # labelled "codebook loss". ``F.mse_loss`` is symmetric and both
        # spellings detach ``quantized``, so that line was ``commitment_loss``
        # computed a second time -- bit-identical value, identical graph. The
        # total was therefore ``(1 + commitment_cost) * commitment_loss``: the
        # encoder was pulled at 5x the documented ``commitment_cost`` while the
        # codebook learned nothing from the extra term.
        vq_loss = self.commitment_cost * commitment_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Update codebook with EMA
        if self.training:
            # The dense one-hot exists only for the EMA update's two reductions.
            encodings = F.one_hot(encoding_indices, self.num_embeddings).to(z_flattened.dtype)
            self._update_codebook(z_flattened, encodings)

        # No perplexity is computed here. A `torch.exp(-sum(p log p))` over the
        # codebook used to be evaluated every step and assigned to `_`: the
        # return signature is `(quantized, loss, indices)` and three callers
        # unpack exactly that, so the value was discarded. `get_codebook_entropy`
        # below is the monitoring surface, on demand and off the step path
        # (non-negotiable 9). It is the entropy of the EMA *usage* distribution
        # rather than of this batch's assignments -- a smoothed reading of the
        # same signal, not the identical number.
        return quantized, vq_loss, encoding_indices

    @torch.no_grad()
    def _update_codebook(self, z_flattened: torch.Tensor, encodings: torch.Tensor):
        """Update codebook using exponential moving average.

        EMA statistics are *state*, not a differentiable path: the codebook is
        updated by assignment, and the straight-through estimator at the call
        site already carries the gradient. Without ``no_grad`` each in-place EMA
        write appended to the autograd graph, growing it monotonically for the
        life of the run (measured +7 nodes/step) -- an unbounded VRAM leak that
        presents as a mid-run OOM, not as a wrong number.
        """
        # Update EMA count
        self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(
            encodings,
            dim=0,
        )

        # Laplace smoothing
        n = torch.sum(self.ema_count)
        smoothing = self.num_embeddings * self.epsilon
        self.ema_count = (self.ema_count + self.epsilon) / (n + smoothing) * n

        # Update EMA weight
        encodings_sum = torch.matmul(encodings.t(), z_flattened)
        self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * encodings_sum

        # Update embeddings
        self.embeddings = self.ema_weight / self.ema_count.unsqueeze(1)

    def get_codebook_usage(self) -> torch.Tensor:
        """Get usage statistics for codebook entries."""
        return self.ema_count / torch.sum(self.ema_count)

    def get_codebook_entropy(self) -> float:
        """Get entropy of codebook usage distribution."""
        usage = self.get_codebook_usage()
        entropy = -torch.sum(usage * torch.log(usage + self.epsilon))
        return entropy.item()

    def quantize_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert indices to quantized vectors."""
        return F.embedding(indices, self.embeddings)

    def get_num_embeddings(self) -> int:
        """Get number of embeddings in codebook."""
        return self.num_embeddings

    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        return self.embedding_dim


# Backward compatibility alias
VectorQuantizer = VQQuantizer
