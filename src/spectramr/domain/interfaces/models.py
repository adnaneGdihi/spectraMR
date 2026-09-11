"""Structural (``Protocol``) contracts for model shapes.

These are typing contracts only -- nothing inherits from them and no isinstance
check runs against them, so none is ``@runtime_checkable``.

``IGenerator``, ``IDiscriminator``, ``IModel`` and the other nominal interfaces
are ABCs owned by :mod:`spectramr.models.interfaces.models`, and this package's
``__init__`` re-exports them from there. A same-named ``Protocol`` defined here
is unreachable through ``from spectramr.domain.interfaces import ...``.
"""

from typing import Protocol

from torch import Tensor

# We use ComplexTensor as a type hint alias for Tensor, typical in PyTorch 2.0+
# where complex tensors are just Tensors with complex dtype.
ComplexTensor = Tensor


class GenerativeModel(Protocol):
    """
    Protocol for Generative Models (GAN, VAE, Diffusion).
    Takes a latent vector or noise and optional condition, outputs a real-space image.

    ### Model Hierarchy
    ```mermaid
    classDiagram
        class GenerativeModel {
            <<protocol>>
            +forward(z, c) Tensor
        }
        class ReconstructionModel {
            <<protocol>>
            +forward(kspace, mask) Tensor
        }
        class IModel {
            <<interface>>
            +name str
            +forward(x)
            +get_parameter_count() int
        }
        class IGenerator {
            <<interface>>
            +generate(x)
        }
        class IDiscriminator {
            <<interface>>
            +discriminate(x)
        }
        IModel <|-- IGenerator
        IModel <|-- IDiscriminator
    ```
    """

    def forward(self, z: Tensor, c: Tensor | None = None) -> Tensor:
        """
        Args:
            z: Latent vector or noise [B, latent_dim] or [B, C, H, W]
            c: Optional conditioning vector/image
        Returns:
            x: Generated image [B, C, H, W]
        """
        ...


class ReconstructionModel(Protocol):
    """
    Protocol for Reconstruction Models (Unrolled Networks, Variational Networks).
    Takes k-space data and a sampling mask, outputs a real-space image.
    """

    def forward(self, kspace: ComplexTensor, mask: Tensor) -> Tensor:
        """
        Args:
            kspace: Input k-space data [B, C, H, W] (complex)
            mask: Sampling mask [B, 1, H, W] or [B, C, H, W]
        Returns:
            x: Reconstructed image [B, C, H, W]
        """
        ...


class IEncoder(Protocol):
    """
    Protocol for Encoder models (VAE encoder, feature extractor).

    Encodes input to latent representation.
    """

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Encode input to latent space.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Tuple of (mu, log_var) for VAE-style encoders,
            or (latent, None) for deterministic encoders.
        """
        ...

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent space."""
        ...


class IDecoder(Protocol):
    """
    Protocol for Decoder models (VAE decoder, upsampling networks).

    Decodes latent representation to output space.
    """

    def forward(self, z: Tensor) -> Tensor:
        """
        Decode latent to output.

        Args:
            z: Latent tensor [B, latent_dim] or [B, C, H, W]

        Returns:
            Decoded output [B, C_out, H_out, W_out]
        """
        ...
